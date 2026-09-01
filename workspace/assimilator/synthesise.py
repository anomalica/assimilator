"""Synthesiser: decide which pages exist and emit one brief per page.

The brief is the language-neutral fact bundle that is the assembler's SOLE input
(decisions 0008/0036): the graph slice for one page, before any prose. The writer
renders the brief and invents nothing of its own. The brief is also the staleness
unit and the audit record - its ``brief_hash`` (over the ordered
``(claim_id, claim_hash)`` pairs plus page identity) is the scheduler's diff key
and the assembler's ``built_from`` freeze (ADR 0010).

This stage is deterministic - graph slice in, brief out, no AI and no money - so
it is an eager light-local step (the AI cost is the downstream *assemble*). v1
emits one brief per entity node (the all-entities page set); the "which topics
deserve a page" threshold refines once algorithmic-evidence-scoring is pinned.

Brief format is YAML, consistent with the digest interchange. Per-claim evidence
is neutral (score null) until scoring is pinned; ``independent_sources`` is real.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import yaml

from assimilator.brief_yaml import INTERNAL_ONLY
from assimilator.brief_yaml import dump as dump_brief_yaml

from anomalica_common.digest import attribution_mode as common_attribution_mode
from anomalica_common.slug import node_slug
from assimilator.database import get_independent_source_count
from assimilator.database import claim_ref_statuses
from assimilator.propose_pages import proposed_node_ids

SCHEMA = "anomalica/brief/1"

# Per-brief claim cap. A hub entity can be referenced by thousands of claims,
# which would make one brief too large for the assembler to render in a single
# pass. The cap is filled ROUND-ROBIN ACROSS SOURCES (_spread_across_sources), so
# every record contributing to a node is represented before any record gets a
# second helping - taking the first MAX of a date-ordered set handed the whole
# budget to the earliest source and silently dropped the rest. Ranking WITHIN a
# source is still chronological, pending evidence-scoring;
# claim_count vs claim_count_total on the page makes any truncation explicit, not
# silent. Tunable via ANOMALICA_BRIEF_MAX_CLAIMS.
#
# 200 WAS NOT A PROMPT-SIZE CONSTRAINT, though it was believed to be one. The
# largest node in the corpus is 1,422 claims, roughly 248k tokens as structured
# YAML, and every model in the pipeline except Haiku has a 1M context. Measured
# at 200: of 690 proposed nodes only 28 exceed the cap, but they are the biggest
# pages, and 30% of every claim-reference on a proposed node was being discarded
# (29,738 available, 20,813 kept). Whitley Strieber kept 14% of his evidence.
#
# 600 is chosen rather than "no cap": the assembler measured generation time
# tracking SOURCE count rather than prompt length (5 sources ~4 min, 12 sources
# over 21 min from a SMALLER prompt), so length is the cheap axis - but that is
# n=1 each side, and an unbounded brief is an unbounded failure when it is wrong.
# At 600 only six nodes truncate at all, and what they lose is now the weakest
# material rather than whatever the transcript happened to say last.
MAX_CLAIMS = int(os.environ.get("ANOMALICA_BRIEF_MAX_CLAIMS", "600"))

# Per-node-type ceiling on how many SOURCES contribute to a brief. Absent means
# unlimited: spread across everything, which is right for a person, where breadth
# is the point. An event is the opposite - it has to narrate one sequence, and
# continuity comes from the account that carries it rather than from twelve
# passing mentions. Cost points the same way: the assembler measured one node at
# 5 sources (~4 min) and 12 sources (over 21 min) from a SMALLER prompt, so
# generation time tracks source count, not length. Evidence is n=1 each side with
# fleet contention unruled-out - strong, not proven - hence a named constant that
# is cheap to revisit rather than a rule buried in the selector.
MAX_SOURCES: dict[str, int] = {
    "event": int(os.environ.get("ANOMALICA_EVENT_MAX_SOURCES", "5"))
}

# A record contributing fewer than this many claims to a node is a mention, not
# an account of it, and is never chosen as one of a capped node's sources however
# focused it looks - two claims in a two-claim record is 100% focus and no use.
MIN_SOURCE_CLAIMS = 5


# Column offsets into the claim rows selected below, named because the tuple is
# long and an importance key that silently reads the wrong field would rank on
# nonsense while looking like it worked.
_COL_ID = 0
_COL_ATTESTATION = 4
_COL_SPEAKER_ID = 9

# Attestation ranked by evidential weight. Measured across the corpus: first_hand
# 16,794, second_hand 6,479, third_hand 431, absent 7,362 - so this discriminates
# on 76% of claims. `confidence` and `claim_role` are NOT used: both exist on the
# schema and neither is populated (confidence is 1.0 on all 31,066 claims,
# claim_role is null on all of them), so ranking by either would be ranking by a
# constant.
_ATTESTATION_RANK = {"first_hand": 3, "second_hand": 2, "third_hand": 1}


def _importance(
    row, node_id: str, corroborated: set[str], ref_status: dict[str, str] | None = None
) -> tuple:
    """Sort key for WHICH claims survive the cap. Higher is kept.

    Document position is the wrong criterion once a node draws on twenty sources:
    you get whatever each transcript happened to open with. This ranks by what
    makes a claim worth the space instead.

    A claim READ AND FOUND NOT TO BELONG sorts below everything. It is not
    dropped - the brief is the audit record, and a claim silently missing from
    it cannot be checked - but it must not displace a usable claim from the cap.
    Without this the Nimitz brief spent 89 of its 272 slots on claims about
    Kaikoura, Socorro and the Delphos encounter.

    Only signals that are actually populated are used, and the ordering within a
    tier stays the caller's document order, so this changes what is DISCARDED
    rather than what the brief reads like.
    """
    suspect = (ref_status or {}).get(row[_COL_ID]) == "suspect"
    return (
        0 if suspect else 1,
        1 if row[_COL_ID] in corroborated else 0,
        1 if row[_COL_SPEAKER_ID] == node_id else 0,
        _ATTESTATION_RANK.get(row[_COL_ATTESTATION], 0),
    )


def _corroborated_claim_ids(conn: sqlite3.Connection) -> set[str]:
    """Claims confirmed against another source. Rare - 53 of 31,066 today - but
    the strongest single signal that a claim is load-bearing, so it sorts first."""
    try:
        rows = conn.execute("SELECT claim_a, claim_b FROM corroborations").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {c for pair in rows for c in pair if c}


def _spread_across_sources(
    rows: list,
    cap: int,
    max_sources: int | None = None,
    focus: dict | None = None,
    importance=None,
) -> list:
    """Fill the cap ROUND-ROBIN across sources, not chronologically.

    Taking the first `cap` rows of a date-ordered set gives the whole budget to
    whichever source happens to be earliest. On the strongest multi-source node in
    the graph that discarded an entire book: Jacques Vallee held 293 claims from
    Messengers of Deception and 232 from The Invisible College, and the brief was
    200 claims from the first and nothing from the second - so the article read as
    a summary of one book while the proposal's source-spread figures, computed on
    the full set, still said "well corroborated".

    One claim from each source in turn, until the cap. A source with few claims is
    exhausted early and the remainder distributes over the rest, so the result is
    proportional once every source has been represented. Grouped by WORK, not by
    record, so two records of one book do not get two helpings.

    `max_sources` caps how many sources contribute at all, keeping the ones with
    the most claims. Spread is right for a PERSON, where breadth across sources is
    the point; it is wrong for an EVENT, which has to narrate one sequence and
    needs continuity from the account that carries it. It is also what generation
    time tracks: the assembler measured the same Nimitz node, same day, at 5
    sources (132k-char prompt, ~4 minutes) and at 12 sources (125k-char prompt,
    over 21 minutes) - a smaller prompt and a 5x slowdown. Reconciling twelve
    accounts of one event is the expensive operation; summarising one 884k-char
    book is not.

    Deterministic: sources are cycled in order of first appearance, and the
    selection is returned in the caller's original order so the brief still reads
    in document order. Both matter - brief_hash is computed over this sequence.
    """
    if len(rows) <= cap and not max_sources:
        return rows
    by_work: dict[object, list[int]] = {}
    for index, row in enumerate(rows):
        by_work.setdefault(row[-1], []).append(index)

    queues = list(by_work.values())
    if max_sources and len(queues) > max_sources:
        # Keep the sources that are ABOUT this node, not the ones that merely say
        # the most about it. Ranking by claim count picks long books over short
        # primary accounts: on the Nimitz encounter it selected five books and
        # DROPPED the CSG-11 incident report, which is the document the event
        # actually happened in. Focus - what share of that record concerns this
        # node - inverts that correctly (Fravor's House statement 63%, the
        # incident report 25%, against Imminent at 2.7%). A minimum absolute
        # count stops a two-claim record scoring 100% and outranking them.
        by_first_index = {q[0]: q for q in queues}
        ranked = sorted(
            queues,
            key=lambda q: (-(focus or {}).get(rows[q[0]][-1], 0.0), -len(q), q[0]),
        )
        substantial = [q for q in ranked if len(q) >= MIN_SOURCE_CLAIMS]
        keep = (substantial or ranked)[:max_sources]
        queues = [by_first_index[q[0]] for q in sorted(keep, key=lambda q: q[0])]
    if sum(len(q) for q in queues) <= cap:
        return [rows[i] for i in sorted(i for q in queues for i in q)]

    if importance is not None:
        # Order each source's queue by importance so the round-robin takes that
        # source's BEST claims first. The final `sorted(chosen)` still returns
        # everything in document order, so this changes which claims survive the
        # cap and not how the brief reads.
        queues = [
            sorted(q, key=lambda i: (importance(rows[i]), -i), reverse=True)
            for q in queues
        ]

    chosen: list[int] = []
    while len(chosen) < cap and any(queues):
        for queue in queues:
            if not queue:
                continue
            chosen.append(queue.pop(0))
            if len(chosen) == cap:
                break
    return [rows[i] for i in sorted(chosen)]


def _source_focus(conn: sqlite3.Connection, node_id: str) -> dict:
    """work_id -> share of that source's claims that concern this node.

    "How much of this record is about the node", not "how many claims it has".
    A 41-claim congressional statement 63% about an encounter is a primary
    account of it; a 1,457-claim book mentioning it in 2.7% of its claims is not,
    however many claims that amounts to.
    """
    rows = conn.execute(
        """
        SELECT COALESCE(r.work_id, r.id) AS work,
               COUNT(DISTINCT c.id) AS here,
               (SELECT COUNT(*) FROM claims c2
                 WHERE COALESCE(
                   (SELECT r2.work_id FROM records r2 WHERE r2.id = c2.record_id),
                   c2.record_id) = COALESCE(r.work_id, r.id)) AS total
          FROM claims c
          JOIN records r ON r.id = c.record_id
         WHERE c.speaker_id = ?
            OR c.id IN (SELECT claim_id FROM claim_node_refs WHERE node_id = ?)
         GROUP BY work
        """,
        (node_id, node_id),
    ).fetchall()
    return {work: (here / total if total else 0.0) for work, here, total in rows}


def _graph_version(conn: sqlite3.Connection) -> str | None:
    """A coarse DB-state stamp (the latest claim mutation), not the reconstruction
    key - brief_hash carries the specific slice. Moves when the graph that could
    affect a brief moves."""
    row = conn.execute("SELECT MAX(created_at) FROM claims").fetchone()
    return row[0] if row else None


def brief_hash(node_id: str, kind: str, ordered_pairs: list[tuple[str, str]]) -> str:
    """sha256 over the ordered (claim_id, claim_hash) selection plus page identity.

    The claims list is ORDER-SENSITIVE (it is the selection order), so it is not
    sorted; page identity is fixed. This is one fingerprint with three uses: the
    scheduler's staleness diff, the assembler's built_from freeze, ADR 0010's
    knowledge-graph-data audit component.
    """
    blob = json.dumps(
        {
            "kind": kind,
            "node_id": node_id,
            "claims": [[c, h] for c, h in ordered_pairs],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _claim_node_refs(
    conn: sqlite3.Connection, claim_id: str, slug_map: dict[str, str]
) -> list[dict]:
    rows = conn.execute(
        "SELECT n.id, n.name, n.metadata FROM claim_node_refs cnr "
        "JOIN nodes n ON n.id = cnr.node_id WHERE cnr.claim_id = ?",
        (claim_id,),
    ).fetchall()
    # Carry the canonical slug (globally disambiguated, same as page.slug and
    # related_nodes[].slug) so the assembler can link an entity mentioned only via
    # a claim's node_refs - not just the related-node set - with the right URL.
    return [
        {
            "node_id": r[0],
            "title": r[1],
            "slug": slug_map.get(r[0]) or node_slug(r[1], r[2]),
        }
        for r in rows
    ]


def build_slug_map(conn: sqlite3.Connection) -> tuple[dict[str, str], list[dict]]:
    """Global node_id -> final URL slug, with collision disambiguation.

    The canonical slug (anomalica_common) is the deployed convention but is per
    node, so two genuinely-distinct same-name entities collide. Resolved globally
    and deterministically: within a colliding group the lexicographically-smallest
    node_id keeps the base slug, the rest append a short node_id. The map is used
    for BOTH page.slug and related_nodes[].slug so a node's URL is identical
    wherever it appears.

    Returns the map plus the list of collisions, so a collision that is really one
    entity wrongly split (the entity-matcher merge bug) is surfaced, not masked.
    """
    rows = conn.execute(
        "SELECT id, name, metadata, node_type FROM nodes WHERE retired_at IS NULL"
    ).fetchall()
    by_base: dict[str, list[tuple[str, str, str]]] = {}
    for node_id, name, metadata, node_type in rows:
        by_base.setdefault(node_slug(name, metadata), []).append(
            (node_id, name, node_type)
        )

    slug_map: dict[str, str] = {}
    collisions: list[dict] = []
    for base, members in by_base.items():
        if len(members) > 1:
            collisions.append(
                {
                    "slug": base,
                    "nodes": [m[0] for m in members],
                    "names": [m[1] for m in members],
                    "types": [m[2] for m in members],
                }
            )
        # Disambiguate only WITHIN a node type. A published URL is
        # /<section>/<slug> and the section follows the type, so an organisation
        # and a project of the same name do not collide in URL space - suffixing
        # them put an arbitrary hex fragment in a real path
        # (/projects/all-domain-anomaly-resolution-office-aaro-fca513ac) to
        # separate pages that were never going to clash. Two types that share a
        # section are the case this cannot see; the assembler's output-path guard
        # is the backstop, and it reports a claimed path rather than overwriting.
        by_type: dict[str, list[tuple[str, str, str]]] = {}
        for member in members:
            by_type.setdefault(member[2], []).append(member)
        for same_type in by_type.values():
            same_type.sort(key=lambda m: m[0])  # deterministic winner
            for i, (node_id, _name, _type) in enumerate(same_type):
                slug_map[node_id] = base if i == 0 else f"{base}-{node_id[:8]}"
    return slug_map, collisions


def _attribution_mode(
    origin_kind: str | None,
    claim_type: str | None,
    attestation: str | None,
    attribution_in_text: bool | None = None,
) -> str:
    """How a consumer must render this claim's attribution: in_text | bare_ok |
    unknown (ADR 0044).

    The RULE lives in anomalica_common, not here. This is a thin call into it. The
    whole lesson of 0044 is that a rule stated in prose gets re-implemented once per
    consumer and drifts - twice, in our case, both times to fail-open. The digester,
    the assimilator and the assembler now share one function, so a disagreement is a
    failing test rather than a silently divergent render.

    ``attribution_in_text`` is DECLARED by the extraction model - the thing that
    wrote the sentence is the only thing that can say what is in the sentence. It is
    not yet carried on the claim (it lands with the post-0044 extraction schema), so
    we pass None until the column exists. That is fail-closed, not a gap: with no
    declared flag, a claim whose truth rests on its attribution (anonymous, hearsay,
    opinion, second-/third-hand) resolves to `unknown` rather than being asserted.
    """
    return common_attribution_mode(
        claim_type=claim_type,
        attestation=attestation,
        origin_kind=origin_kind,
        attribution_in_text=attribution_in_text,
        has_chain=bool(origin_kind),
    ).value


def build_entity_brief(
    conn: sqlite3.Connection, node_id: str, slug_map: dict[str, str] | None = None
) -> dict | None:
    """Build the brief for one entity node from its graph slice.

    Mirrors the assembler's --node read contract (node + claims-where-speaker-or-
    referenced + related-nodes) so the brief covers everything the writer reads,
    and adds the freezer fields (claim_hash per claim, brief_hash, resolved slugs).
    slug_map (from build_slug_map) gives globally-disambiguated slugs; without it,
    the per-node canonical slug is used (fine for a single-page emit).
    """
    node = conn.execute(
        "SELECT id, node_type, name, metadata FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if node is None:
        return None
    ref_status = claim_ref_statuses(conn, node_id)
    nid, node_type, name, metadata = node
    slug_map = slug_map or {}

    def _slug(nid_: str, name_: str, meta_: object) -> str:
        return slug_map.get(nid_) or node_slug(name_, meta_)

    rows = conn.execute(
        """
        SELECT DISTINCT c.id, c.content, c.original_excerpt, c.claim_type,
               c.attestation, c.location_in_record, c.date, c.date_end, c.claim_hash,
               c.speaker_id, sp.name,
               c.record_id, r.title, r.date, r.reference, r.content_hash, r.friendly_name,
               c.origin_kind, c.origin, c.relay, COALESCE(r.work_id, c.record_id)
        FROM claims c
        LEFT JOIN records r ON r.id = c.record_id
        LEFT JOIN nodes sp ON sp.id = c.speaker_id
        WHERE c.speaker_id = ?
           OR c.id IN (SELECT claim_id FROM claim_node_refs WHERE node_id = ?)
        ORDER BY r.date, c.location_in_record
        """,
        (node_id, node_id),
    ).fetchall()

    claim_count_total = len(rows)
    # A claim READ AND FOUND NOT TO BELONG is excluded from the brief, because
    # the brief is what the assembler builds a page from and master's rule is
    # that presence on a node is evidence it was ATTACHED, not that it belongs.
    # Leaving them in and flagging them was the first design, and it was wrong
    # twice over: they displaced usable claims from the cap (89 of the Nimitz
    # brief's 272 slots), and it made correct assembly depend on every consumer
    # remembering to filter. The count is reported in `belonging` below and the
    # per-claim detail stays queryable in claim_ref_status, so nothing is hidden.
    suspect_ids = {cid for cid, st in ref_status.items() if st == "suspect"}
    excluded_suspect = sum(1 for r in rows if r[_COL_ID] in suspect_ids)
    rows = [r for r in rows if r[_COL_ID] not in suspect_ids]
    claims: list[dict] = []
    ordered_pairs: list[tuple[str, str]] = []
    max_sources = MAX_SOURCES.get(node_type)
    focus = _source_focus(conn, node_id) if max_sources else None
    corroborated = _corroborated_claim_ids(conn)
    for row in _spread_across_sources(
        rows,
        MAX_CLAIMS,
        max_sources,
        focus,
        importance=lambda r: _importance(r, node_id, corroborated, ref_status),
    ):
        (
            cid,
            content,
            excerpt,
            claim_type,
            attestation,
            location,
            date,
            date_end,
            chash,
            speaker_id,
            speaker_name,
            rec_id,
            rtitle,
            rdate,
            rref,
            rhash,
            rfriendly,
            origin_kind,
            origin,
            relay,
            _work_id,  # selection key for _spread_across_sources; not emitted
        ) = row
        claims.append(
            {
                "claim_id": cid,
                "claim_hash": chash,
                "content": content,
                "original_excerpt": excerpt,
                "claim_type": claim_type,
                "attestation": attestation,
                "speaker": {"node_id": speaker_id, "title": speaker_name}
                if speaker_id
                else None,
                # The claim's own chain (ADR 0044) - who asserted it and through
                # whom it reached the speaker. Distinct from `provenance` below,
                # which is the RECORD's source metadata. Null when the digest
                # predates 0044. `attribution_mode` is the derived rendering
                # contract: in_text | bare_ok | unknown - see attribution_mode().
                "provenance_chain": {
                    "origin_kind": origin_kind,
                    "origin": origin or "",
                    "relay": json.loads(relay) if relay else [],
                }
                if origin_kind
                else None,
                "attribution_mode": _attribution_mode(
                    origin_kind, claim_type, attestation
                ),
                # INTERNAL AUDIT SIGNAL, NOT READER-FACING. Do not render this
                # or hedge prose on it: whether WE have checked a claim belongs
                # is a fact about our process, not about the evidence, and a
                # misattached claim is still correctly attributed to its source -
                # it is simply in the wrong article, which no qualifier in the
                # text can fix. The protection is the suspect EXCLUSION above,
                # or nothing. (Settled with the assembler, 2026-09-01.)
                #
                # Whether this claim BELONGS on this page's node, as distinct
                # from being ATTACHED to it. "unreviewed" is NOT "verified":
                # a consumer asserting an unreviewed claim in its own voice is
                # making a judgement and should say so, and a "suspect" claim
                # must not be asserted at all. See database.claim_ref_status.
                "attachment": ref_status.get(cid, "unreviewed"),
                "node_refs": _claim_node_refs(conn, cid, slug_map),
                "date": date,
                "date_end": date_end,
                "location_in_record": location,
                "evidence": {
                    "score": None,  # neutral until algorithmic-evidence-scoring pins
                    "independent_sources": get_independent_source_count(conn, cid),
                },
                "provenance": {
                    "record_id": rec_id,
                    "record_title": rtitle,
                    "record_date": rdate,
                    "record_reference": rref,
                    "content_hash": rhash,
                    "friendly_name": rfriendly,
                },
            }
        )
        ordered_pairs.append((cid, chash or ""))

    related = conn.execute(
        """
        SELECT b.node_id, n.name, n.node_type, n.metadata, COUNT(*) AS shared
        FROM claim_node_refs a
        JOIN claim_node_refs b ON b.claim_id = a.claim_id AND b.node_id != a.node_id
        JOIN nodes n ON n.id = b.node_id
        WHERE a.node_id = ? AND n.retired_at IS NULL
        GROUP BY b.node_id
        ORDER BY shared DESC, n.name
        LIMIT 30
        """,
        (node_id,),
    ).fetchall()
    related_nodes = [
        {
            "node_id": r[0],
            "title": r[1],
            "node_type": r[2],
            "slug": _slug(r[0], r[1], r[3]),
            "shared_claims": r[4],
        }
        for r in related
    ]

    return {
        "schema": SCHEMA,
        "brief_hash": brief_hash(nid, "entity", ordered_pairs),
        "page": {
            "kind": "entity",
            "node_id": nid,
            "node_type": node_type,
            "title": name,
            "slug": _slug(nid, name, metadata),
            "claim_count": len(claims),
            "claim_count_total": claim_count_total,
        },
        "generated": {"graph_version": _graph_version(conn)},
        # THIS DIRECTORY IS NOT A CACHE OF THE PUBLISHED ONE. The brief written
        # here carries every original_excerpt verbatim, including from sources we
        # may not redistribute; publish_briefs strips those by copyright status
        # and the assembler puts an excerpt on a public page as a quote. So a
        # consumer that reads briefs from HERE instead of the published directory
        # publishes restricted source text - widening the access model, which is
        # Mark's sign-off and is not reversible once it is on the CDN. The source
        # side is systematically NEWER, which is exactly what makes that mistake
        # attractive. The flag is here so the file says so itself.
        "publication": {
            "status": INTERNAL_ONLY,
            "warning": (
                "Internal audit copy. Carries verbatim excerpts from sources we "
                "may not redistribute. NOT FOR PUBLICATION - build only from the "
                "output of `assimilator publish-briefs`, which redacts by "
                "copyright status."
            ),
        },
        # Whether this node's claims BELONG on it, as distinct from being
        # attached. UNREVIEWED IS NOT VERIFIED: it means nobody has checked, and
        # a consumer asserting those claims in its own voice is making a
        # judgement it should state. `suspect` claims are excluded from `claims`
        # above; they remain in the graph and in claim_ref_status.
        "belonging": {
            "verified": sum(1 for st in ref_status.values() if st == "verified"),
            "suspect_excluded": excluded_suspect,
            "unreviewed": claim_count_total - len(ref_status),
        },
        "related_nodes": related_nodes,
        "claims": claims,
    }


def entity_node_ids(conn: sqlite3.Connection) -> list[str]:
    """The page set the synthesiser emits briefs for: the nodes currently proposed
    for a page. Page SELECTION is no longer the synthesiser's job - proposal-gen
    (propose_pages.py) decides via the page-worthiness gate and writes the derived
    page_proposals table; the synthesiser consumes it. Empty until propose() has
    run (the dependency gate: proposal-gen precedes synthesise)."""
    return proposed_node_ids(conn)


def write_brief(brief: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{brief['page']['slug']}.yaml"
    path.write_text(dump_brief_yaml(brief))
    return path


def default_briefs_dir() -> Path:
    return Path(
        os.environ.get(
            "ANOMALICA_BRIEFS_DIR",
            str(Path.home() / ".local" / "share" / "assimilator" / "briefs"),
        )
    )


def prune_retired_briefs(
    conn: sqlite3.Connection, out_dir: Path, slug_map: dict | None = None
) -> list[str]:
    """Delete briefs whose node no longer exists or has been retired.

    Emission only ever WRITES, so a brief outlives the node it describes. A merge
    retires its victims and their briefs stay on disk pointing at a node that is
    gone - after the Luis/Lue/Lou Elizondo merge, lou-elizondo.yaml and
    luis-lou-elizondo.yaml both remained, alongside four older ones from earlier
    merges.

    That is not clutter, it is a live hazard: the assembler takes a brief by slug
    and will happily build a page from a dead one. Assembling the "Lue Elizondo"
    node had already published a second Elizondo page beside the existing one; a
    stale brief lets exactly that happen again after the merge meant to fix it.

    A RENAME strands a brief the same way. The Elizondo merge kept node 87788ebc
    and renamed it "Lue Elizondo" -> "Luis Elizondo", so lue-elizondo.yaml and
    luis-elizondo.yaml both described the SAME LIVE NODE. The node is not
    retired, so the check above cannot see it; a node must have exactly one
    brief, and the one at its current slug is that brief.

    Only RETIRED, ABSENT, or STALE-SLUG briefs are pruned. A brief for a live
    node at its correct slug that is merely unproposed today is left alone - the
    proposal set moves with thresholds, and deleting on that basis would churn.
    """
    live = {r[0] for r in conn.execute("SELECT id FROM nodes WHERE retired_at IS NULL")}
    removed: list[str] = []
    for path in sorted(out_dir.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue  # unreadable: leave it for a human, never delete blind
        node_id = (doc.get("page") or {}).get("node_id")
        if not node_id:
            continue
        if node_id not in live:
            path.unlink()
            removed.append(path.name)
            continue
        current = (slug_map or {}).get(node_id)
        if current and path.stem != current:
            path.unlink()  # stranded by a rename; the node's brief is at `current`
            removed.append(path.name)
    return removed


def emit_all(conn: sqlite3.Connection, out_dir: Path, on_progress=None) -> dict:
    log = on_progress or (lambda _: None)
    slug_map, collisions = build_slug_map(conn)
    written = 0
    for node_id in entity_node_ids(conn):
        brief = build_entity_brief(conn, node_id, slug_map)
        if brief is None or not brief["claims"]:
            continue
        write_brief(brief, out_dir)
        written += 1
    log(f"Emitted {written} briefs to {out_dir}")
    removed = prune_retired_briefs(conn, out_dir, slug_map)
    if removed:
        log(f"Pruned {len(removed)} stale brief(s) - node retired, gone, or renamed:")
        for name in removed:
            log(f"  {name}")
    if collisions:
        log(
            f"NOTE: {len(collisions)} slug collisions disambiguated by node-id "
            f"suffix - review for entity-matcher merge bugs (same entity split):"
        )
        for c in collisions:
            log(f"  {c['slug']}: {c['names']}")
    return {"written": written, "collisions": collisions, "pruned": removed}


def main(argv: list[str] | None = None) -> int:
    """Host-runnable entry point: `python -m assimilator.synthesise`.

    Emits briefs from the graph (deterministic, no Claude, no fastembed). Needs
    anomalica_common + pyyaml on the path. With --node, emits one brief.
    """
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    parser = argparse.ArgumentParser(
        prog="assimilator.synthesise",
        description="Emit one brief per entity page from the graph (no AI).",
    )
    parser.add_argument("--db", default=default_db, help="graph DB (read-only)")
    parser.add_argument("--out", default=None, help="briefs dir")
    parser.add_argument("--node", default=None, help="emit only this node id")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else default_briefs_dir()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        if args.node:
            brief = build_entity_brief(conn, args.node)
            if brief is None:
                print(f"No such node: {args.node}")
                return 1
            print(f"Wrote {write_brief(brief, out_dir)}")
        else:
            result = emit_all(conn, out_dir, on_progress=print)
            print(f"Emitted {result['written']} briefs to {out_dir}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
