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
import re
import sqlite3

import yaml
from pathlib import Path


from assimilator.brief_size import budget_for, consuming_window, estimate_tokens
from assimilator.brief_yaml import INTERNAL_ONLY
from assimilator.brief_yaml import dump as dump_brief_yaml

from anomalica_common.digest import attribution_mode as common_attribution_mode
from anomalica_common.slug import node_slug, section_for
from anomalica_common.titles import capitalise_first, collapse_bare_title_acronyms
from assimilator.database import get_independent_source_count
from assimilator.database import claim_ref_statuses
from assimilator.propose_pages import proposed_node_ids
from assimilator.data_dir import data_dir

SCHEMA = "anomalica/brief/2"

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
# A BACKSTOP, NOT THE CAP. The real limit is the token budget below; this stops
# a pathological node from building a million-entry list before the budget
# refuses it. It was 600 and that WAS the cap, set against context limits no
# model in the policy file has any more - it cut 76% of what the graph knows
# about Whitley Strieber out of his article, uncitable and unreported.
MAX_CLAIMS = int(os.environ.get("ANOMALICA_BRIEF_MAX_CLAIMS", "20000"))

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
# What a consumer RENDERS around a claim's text - an attribution, a date, a
# record title - not what the YAML carries around it. The file holds about 1,190
# characters of ids, hashes, slugs and provenance per claim (measured 2026-09-02
# over seven briefs, 441 tokens at 2.7), and none of it reaches a model: the
# assembler renders each claim as prose and its prompt for the largest brief
# came to 286,000 tokens against a 3.6 MB file. Sized as the file, that brief
# reads as over the window it fits in with room to spare. Sized as rendered,
# claim text at 2.7 characters per token plus this framing is within a few
# percent of the measured prompt.
_CLAIM_OVERHEAD_TOKENS = 20
_COL_ID = 0
_COL_HASH = 8
_COL_CONTENT = 1
_COL_EXCERPT = 2
_COL_ATTESTATION = 4
_COL_SPEAKER_ID = 9
_COL_ENTAILMENT = 21  # label; score, model and premise follow

# Attestation ranked by evidential weight. Measured across the corpus: first_hand
# 16,794, second_hand 6,479, third_hand 431, absent 7,362 - so this discriminates
# on 76% of claims. `confidence` and `claim_role` are NOT used: both exist on the
# schema and neither is populated (confidence is 1.0 on all 31,066 claims,
# claim_role is null on all of them), so ranking by either would be ranking by a
# constant.
_ATTESTATION_RANK = {"first_hand": 3, "second_hand": 2, "third_hand": 1}


def _claim_token_cost(row) -> int:
    """Token cost of one claim row, as a consumer RENDERS it.

    Content and excerpt are the bulk; the framing a writer puts around them is
    a near-constant added flat. This is deliberately not the YAML's size - see
    _CLAIM_OVERHEAD_TOKENS - because the window the brief must fit is the one
    the rendered prompt goes into, and the file is nearly five times larger
    than what any consumer sends.
    """
    text = (row[_COL_CONTENT] or "") + (row[_COL_EXCERPT] or "")
    return estimate_tokens(text) + _CLAIM_OVERHEAD_TOKENS


def _entailment_summary(rows) -> dict:
    """The page's entailment block, the same shape stats reports: counts by
    label, and the entailed share split by premise (quote alone, or the
    record around it), never as one number."""
    from assimilator.database import summarise_entailment

    return summarise_entailment(
        [(r[_COL_ENTAILMENT], r[_COL_ENTAILMENT + 3], 1) for r in rows]
    )


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
    budget: int | None = None,
    cost=None,
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
        # The minimum is a TIEBREAKER, not a filter. It exists so a two-claim
        # record scoring 100% focus cannot outrank a primary account - not to
        # drop evidence when there is room. Applied as a filter it did exactly
        # that: Rendlesham has 7 sources and a cap of 5, three clear the
        # minimum, and the other four were discarded with two slots still free,
        # losing 6 claims the node held.
        substantial = [q for q in ranked if len(q) >= MIN_SOURCE_CLAIMS]
        pool = substantial or ranked
        # FOCUS PICKS THE PRIMARY ACCOUNT; SIZE FILLS THE REST. Focus alone
        # concentrated the brief instead of representing the evidence: on the
        # Nimitz encounter it kept the 141-claim podcast and DROPPED the second
        # and third largest sources outright (114 and 70 claims), so a body of
        # evidence that is 35% one podcast produced a brief that was 67% one
        # podcast. A brief must not be more concentrated than the evidence
        # behind it - the whole point of spreading is defeated if the spread is
        # computed over a set already narrowed to one voice.
        #
        # The most-focused source takes the first slot, the rest go by SIZE.
        # Focus alone concentrated the brief instead of representing the
        # evidence: on the Nimitz encounter it kept a 141-claim podcast and
        # dropped the second and third largest sources outright (114 and 70
        # claims), turning evidence that is 35% one podcast into a brief that is
        # 67% one podcast. Size fills the rest so the article rests on the bulk
        # of what we hold.
        #
        # THIS DOES NOT SOLVE THE PRIMARY-DOCUMENT PROBLEM and must not be read
        # as doing so. Focus cannot find the CSG-11 incident report: it edges 2
        # of its 204 claims to the event and 202 to the participants, scoring
        # 0.010 - the LOWEST of any source on that node. The signal that
        # identifies it is which nodes the record's digest DECLARED, which is a
        # different fact from what its claims edge to (see record_nodes). How to
        # combine the two is a data-model question and is not settled here.
        keep = pool[:1]
        kept_ids = {q[0] for q in keep}
        keep += sorted(
            (q for q in pool if q[0] not in kept_ids), key=lambda q: (-len(q), q[0])
        )[: max_sources - len(keep)]
        if len(keep) < max_sources:
            kept_ids = {q[0] for q in keep}
            keep += [q for q in ranked if q[0] not in kept_ids][
                : max_sources - len(keep)
            ]
        queues = [by_first_index[q[0]] for q in sorted(keep, key=lambda q: q[0])]
    if sum(len(q) for q in queues) <= cap and budget is None:
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
    spent = 0
    stop = False
    while len(chosen) < cap and any(queues) and not stop:
        for queue in queues:
            if not queue:
                continue
            index = queue[0]
            if budget is not None and cost is not None:
                price = cost(rows[index])
                # Stop on the FIRST claim that does not fit rather than
                # skipping it and trying smaller ones. Skipping would fill the
                # tail with whichever claims happen to be short, which is a
                # selection rule nobody chose and one that silently prefers
                # thin evidence.
                if spent + price > budget:
                    stop = True
                    break
                spent += price
            queue.pop(0)
            chosen.append(index)
            if len(chosen) == cap:
                stop = True
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


def brief_hash(
    node_ids: str | list[str], kind: str, ordered_pairs: list[tuple[str, str]]
) -> str:
    """sha256 over the ordered (claim_id, claim_hash) selection plus page identity.

    The claims list is ORDER-SENSITIVE (it is the selection order), so it is not
    sorted; page identity is fixed. This is one fingerprint with three uses: the
    scheduler's staleness diff, the assembler's built_from freeze, ADR 0010's
    knowledge-graph-data audit component.
    """
    blob = json.dumps(
        {
            "kind": kind,
            # THE MEMBER LIST, not one node: a page can cover several (brief/2)
            # and adding a member changes what the page should say. Without it
            # here, a page built from the old member set still looks fresh.
            "node_id": (node_ids if isinstance(node_ids, str) else ",".join(node_ids)),
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


def page_title(name: str) -> str:
    """The reader-facing headline for a node. The NAME keeps "Full Name (ACRONYM)"
    for the matcher and the slug; the title alone writes UFO and UAP bare and
    starts with a capital (anomalica_common.titles - the assembler applies the
    same two rules to an article's title, so the headline cannot drift between
    the brief page and the article)."""
    return capitalise_first(collapse_bare_title_acronyms(name))


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
    conn: sqlite3.Connection,
    node_id: str,
    slug_map: dict[str, str] | None = None,
    node_ids: list[str] | None = None,
    page: dict | None = None,
) -> dict | None:
    """Build the brief for one entity node from its graph slice.

    Mirrors the assembler's --node read contract (node + claims-where-speaker-or-
    referenced + related-nodes) so the brief covers everything the writer reads,
    and adds the freezer fields (claim_hash per claim, brief_hash, resolved slugs).
    slug_map (from build_slug_map) gives globally-disambiguated slugs; without it,
    the per-node canonical slug is used (fine for a single-page emit).
    """
    members = [n for n in (node_ids or [node_id]) if n]
    node = conn.execute(
        "SELECT id, node_type, name, metadata FROM nodes WHERE id = ?", (members[0],)
    ).fetchone()
    if node is None:
        return None
    node_id = members[0]
    ref_status: dict[str, str] = {}
    for member in members:
        ref_status.update(claim_ref_statuses(conn, member))
    nid, node_type, name, metadata = node
    # A COMPOSED PAGE unions its members' claims (curation/pages.yaml). The
    # members stay separate nodes - UFO and UAP share 26 claims of 2,068, so a
    # merge would destroy which word each source used - and only the page is
    # one thing. Which member a shared claim is credited to is settled by
    # member order, so two runs of the brief produce the same page.
    member_rank: dict[str, int] = {}
    for position, member in enumerate(members):
        for (cid,) in conn.execute(
            "SELECT claim_id FROM claim_node_refs WHERE node_id = ? "
            "UNION SELECT id FROM claims WHERE speaker_id = ?",
            (member, member),
        ):
            member_rank.setdefault(cid, position)
    slug_map = slug_map or {}

    def _slug(nid_: str, name_: str, meta_: object) -> str:
        return slug_map.get(nid_) or node_slug(name_, meta_)

    rows = conn.execute(
        """
        SELECT DISTINCT c.id, c.content, c.original_excerpt, c.claim_type,
               c.attestation, c.location_in_record, c.date, c.date_end, c.claim_hash,
               c.speaker_id, sp.name,
               c.record_id, r.title, r.date, r.reference, r.content_hash, r.friendly_name,
               c.origin_kind, c.origin, c.relay, COALESCE(r.work_id, c.record_id),
               c.entailment_label, c.entailment_score, c.entailment_model,
               c.entailment_premise
        FROM claims c
        LEFT JOIN records r ON r.id = c.record_id
        LEFT JOIN nodes sp ON sp.id = c.speaker_id
        WHERE c.speaker_id IN ({ids})
           OR c.id IN (SELECT claim_id FROM claim_node_refs WHERE node_id IN ({ids}))
        ORDER BY r.date, c.location_in_record
        """.format(ids=",".join("?" * len(members))),
        (*members, *members),
    ).fetchall()

    # Deduped by claim_hash: one claim reached through two members is one
    # claim. A naive union puts every count on a composed page out by the
    # number its members share (26 for UFO and UAP).
    if len(members) > 1:
        best: dict[str, tuple] = {}
        for row in rows:
            key = row[_COL_HASH] or row[_COL_ID]
            rank = (member_rank.get(row[_COL_ID], len(members)), row[_COL_ID])
            if key not in best or rank < best[key][0]:
                best[key] = (rank, row)
        keep = {id(r) for _, r in best.values()}
        rows = [r for r in rows if id(r) in keep]
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
    # Size the brief against the window of the stage that CONSUMES it, not
    # against a claim count. Claim count is a proxy for size and excerpt lengths
    # vary by an order of magnitude. If the policy file cannot be read the
    # budget is None and nothing is cut - a guessed window would reintroduce
    # the fault this replaced, invisibly.
    window = consuming_window()
    token_budget = budget_for(window) if window else None
    available = len(rows)
    selected_rows = _spread_across_sources(
        rows,
        MAX_CLAIMS,
        max_sources,
        focus,
        importance=lambda r: _importance(r, node_id, corroborated, ref_status),
        budget=token_budget,
        cost=_claim_token_cost,
    )
    for row in selected_rows:
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
            _ent_label,  # entailment; the dict reads these by _COL_ENTAILMENT
            _ent_score,
            _ent_model,
            _ent_premise,
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
                # The digester's check that the excerpt supports the claim as
                # written. Absent when not assessed, as in the digest. Shown,
                # not weighted: the evidence score's definition is Mark's.
                **(
                    {
                        "entailment": {
                            "label": row[_COL_ENTAILMENT],
                            "score": row[_COL_ENTAILMENT + 1],
                            "model": row[_COL_ENTAILMENT + 2],
                            "premise": row[_COL_ENTAILMENT + 3],
                        }
                    }
                    if row[_COL_ENTAILMENT]
                    else {}
                ),
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
        WHERE a.node_id IN ({ids}) AND n.retired_at IS NULL
          AND b.node_id NOT IN ({ids})
        GROUP BY b.node_id
        ORDER BY shared DESC, n.name
        LIMIT 30
        """.format(ids=",".join("?" * len(members))),
        (*members, *members),
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

    page = page or {}
    member_rows = [
        row
        for row in (
            conn.execute(
                "SELECT id, name, node_type FROM nodes WHERE id = ?", (m,)
            ).fetchone()
            for m in members
        )
        if row
    ]
    return {
        "schema": SCHEMA,
        "brief_hash": brief_hash(members, "entity", ordered_pairs),
        "page": {
            "kind": "entity",
            # A PAGE COVERS A LIST OF NODES (brief/2): one entry for an ordinary
            # page, several for a composed one, never absent. A consumer acting
            # on a covered node must act on EVERY entry - under the old singular
            # field a page whose second member was retired or vetoed stayed up
            # and kept publishing that member's claims.
            "nodes": [
                {"node_id": r[0], "name": r[1], "node_type": r[2]} for r in member_rows
            ],
            "node_type": page.get("node_type") or node_type,
            # The PAGE's own name, never a member's: a title lifted from a
            # member would silently rename the page, and move its URL, when a
            # member is added or removed.
            "title": page_title(page.get("name") or name),
            "slug": page.get("slug") or _slug(nid, name, metadata),
            "claim_count": len(claims),
            "claim_count_total": claim_count_total,
        },
        "generated": {"graph_version": _graph_version(conn)},
        # How big this brief is AS A CONSUMER RENDERS IT (claim text plus a
        # line of framing per claim - see _claim_token_cost), so a consumer can
        # pick a model that holds it rather than guessing. Not the file's size:
        # the YAML around each claim never reaches a model. An estimate - see
        # brief_size - erring high, because overflowing a context window is a
        # silent failure at the provider while an oversized model merely costs.
        "size": {
            "claims": len(claims),
            "tokens_estimated": sum(_claim_token_cost(r) for r in selected_rows),
            "sized_against": window or None,
        },
        # The entailment check summarised for the page: of the claims carried,
        # how many were assessed and how the labels fall. The entailed fraction
        # is the first component of the evidence score - which is not defined
        # yet, so it is shown here and weights nothing.
        "entailment": _entailment_summary(selected_rows),
        # SAY SO WHEN THE BRIEF IS NOT ALL OF IT. A claim cut here cannot appear
        # in the article, cannot be cited, and left unsaid nothing downstream
        # can tell it ever existed - a page built from a quarter of the evidence
        # is indistinguishable from one built from all of it. Absent when the
        # brief is complete, so its presence is the signal.
        **(
            {
                "truncated": {
                    "kept": len(claims),
                    "available": available,
                    # NAME THE BINDING CONSTRAINT. Two different things cut a
                    # brief and they mean different things to a consumer: a
                    # token cut says the subject is too large for the model, a
                    # source cut says we deliberately narrated from the closest
                    # accounts. Reporting the token window for a source-capped
                    # brief would send someone looking for a bigger model that
                    # would change nothing.
                    "why": (
                        f"the brief would not fit the {window:,}-token window "
                        "of the stage that consumes it"
                        if token_budget
                        and sum(_claim_token_cost(r) for r in rows) > token_budget
                        else (
                            f"an event narrates from at most {max_sources} "
                            "sources; claims from the others are not carried"
                        )
                    ),
                }
            }
            if available > len(claims)
            else {}
        ),
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


def brief_relpath(node_type: str, slug: str) -> Path:
    """Where a page's brief lives under a briefs directory: <section>/<slug>.yaml.

    A page's identity is the PAIR /<section>/<slug> (anomalica_common.slug.
    page_path) - the slug is disambiguated only within a node type, so an event
    and a project of one name share a slug and do not share a URL. The brief
    file was keyed on the slug alone, so those two pages had ONE file: whichever
    node wrote last owned it, the scheduler matched the other node's id against
    it, found a foreign brief, re-emitted, and the pair alternated on every pass
    (Apollo 14, SETI - 2026-09-02). The path now carries the same two halves as
    the page it feeds, mirroring content/pages/<section>/<slug>.<lang>.md, and a
    brief REFERENCE is "<section>/<slug>" - which the assembler's load_brief
    resolves as a direct path.
    """
    return Path(section_for(node_type)) / f"{slug}.yaml"


def brief_files(briefs_dir: Path) -> list[Path]:
    """Every brief under a briefs directory, in a stable order. Only the
    sectioned layout: a file directly in the root is the pre-section layout and
    is pruned, never read."""
    return sorted(briefs_dir.glob("*/*.yaml"))


_BULK_START = re.compile(r"^(related_nodes|claims):", re.M)


def brief_header(path: Path) -> dict | None:
    """The brief parsed WITHOUT its bulk: everything above related_nodes and
    claims - schema, brief_hash, page, generated, size, publication, belonging.

    Enough for anything asking which node a brief is for, what graph state it
    reflects, or what its hash is. Parsing each brief whole for those three
    fields cost the scheduler 105 of its 131 seconds per queue rebuild over 814
    briefs (measured 2026-09-02); a header is under 2 KB. The cut is at the
    first bulk key at column 0, so a related node's fields, which sit at the
    same indent as the page's, can never be read as the page's. None when the
    file cannot be read or its header is not a mapping: such a file is left
    for a human, never deleted or rewritten blind.
    """
    try:
        with path.open("rb") as f:
            raw = f.read(16384)
        text = raw.decode("utf-8", errors="ignore")
        m = _BULK_START.search(text)
        if m is None and len(raw) == 16384:
            text = path.read_text(errors="ignore")
            m = _BULK_START.search(text)
        doc = yaml.safe_load(text[: m.start()] if m else text)
    except (OSError, yaml.YAMLError):
        return None
    return doc if isinstance(doc, dict) else None


def brief_node_ids(path: Path) -> list[str]:
    """Every node the brief's page covers, in member order; empty if unreadable.

    Reads brief/1's singular `page.node_id` as a one-member list. Emission only
    ever writes brief/2, but a brief on disk is a FILE the emitter did not
    necessarily write this run, and a reader that returns nothing for the old
    shape silently treats an existing page as unknown - which skipped 552 of
    the 806 briefs on the migration run rather than rewriting them.
    """
    page = (brief_header(path) or {}).get("page") or {}
    out = [
        str(n.get("node_id"))
        for n in (page.get("nodes") or [])
        if isinstance(n, dict) and n.get("node_id")
    ]
    if not out and page.get("node_id"):
        out = [str(page["node_id"])]
    return out


def brief_node_id(path: Path) -> str | None:
    """The brief's PRIMARY node - its first member. A caller that ACTS on the
    page (prunes it, retires it, rebuilds it) must use brief_node_ids and act
    on every member; this is for the cases that need one identity."""
    ids = brief_node_ids(path)
    return ids[0] if ids else None


def write_brief(brief: dict, out_dir: Path) -> Path:
    page = brief["page"]
    path = out_dir / brief_relpath(page["node_type"], page["slug"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_brief_yaml(brief))
    return path


def default_briefs_dir() -> Path:
    return Path(
        os.environ.get(
            "ANOMALICA_BRIEFS_DIR",
            str(data_dir() / "briefs"),
        )
    )


def nodes_with_claims(conn: sqlite3.Connection) -> set[str]:
    """Every node a brief can be built for: named by a claim, or its speaker.
    The same two routes build_entity_brief reads claims through."""
    return {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT node_id FROM claim_node_refs "
            "UNION SELECT DISTINCT speaker_id FROM claims WHERE speaker_id IS NOT NULL"
        )
    }


def prune_retired_briefs(
    conn: sqlite3.Connection,
    out_dir: Path,
    slug_map: dict | None = None,
    node_ids: set[str] | None = None,
) -> list[str]:
    """Delete briefs whose node no longer exists, is retired, or has moved.

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

    The PATH strands a brief the same way. A brief lives at <section>/<slug>.yaml
    (brief_relpath); a file anywhere else for a live node - the pre-section flat
    layout, or a section the node no longer publishes under - is removed on the
    same rule: a node has exactly one brief, and the one at its current path is
    that brief.

    A node with NO CLAIMS strands its brief too. A brief is built from claims
    and emission skips a node without any, so a brief for a claimless live node
    can only be a leftover: the claims moved to another node (a re-import
    re-resolved them, a twin took them) and nothing rewrote or removed the file.
    Two such briefs sat on disk on 2026-09-03 - the MUFON expansion node and
    "UAP Gerb", both at zero claims beside a twin holding them all - and both
    published as pages under the old title after every other brief was rebuilt.

    Only RETIRED, ABSENT, MOVED, or CLAIMLESS briefs are pruned. A brief for a
    live node with claims at its correct path that is merely unproposed today is
    left alone - the proposal set moves with thresholds, and deleting on that
    basis would churn.

    slug_map must be the GLOBAL map (build_slug_map): a partial one would make a
    same-type collision loser look stranded at its own disambiguated slug. It is
    built here when not supplied. node_ids restricts the sweep to the briefs of
    those nodes, for the per-node emit.
    """
    live = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT id, node_type, name, metadata FROM nodes WHERE retired_at IS NULL"
        )
    }
    if not slug_map:
        slug_map, _ = build_slug_map(conn)
    with_claims = nodes_with_claims(conn)
    # A composed page's path comes from the PAGE, not from any member, and its
    # members have no brief of their own.
    from assimilator.pages import composed_pages

    composed = {
        tuple(p["node_ids"]): brief_relpath(p["node_type"], p["slug"])
        for p in composed_pages(conn)
    }
    # A COVERED node has no brief of its own: the page covering it is its brief.
    # Without this the member's old brief stands beside the composed one and the
    # assembler can build both, which is the duplicate page composition exists
    # to remove (the UFO brief survived the first composition run this way).
    covered = {
        m: brief_relpath(p["node_type"], p["slug"])
        for p in composed_pages(conn)
        for m in p["node_ids"]
    }
    removed: list[str] = []
    for path in sorted(out_dir.glob("*.yaml")) + brief_files(out_dir):
        members = brief_node_ids(path)
        if not members:
            continue  # no readable identity: leave it for a human, never delete blind
        if node_ids is not None and not (set(members) & set(node_ids)):
            continue
        rel = path.relative_to(out_dir)
        usable = [m for m in members if m in live and m in with_claims]
        if not usable:
            path.unlink()
            removed.append(str(rel))
            continue
        current = composed.get(tuple(members))
        if current is None:
            if len(members) > 1:
                # The composition is gone or its members changed; the page it
                # described no longer exists in that shape.
                path.unlink()
                removed.append(str(rel))
                continue
            covering = covered.get(usable[0])
            if covering is not None and rel != covering:
                path.unlink()  # a member's own brief, superseded by its page
                removed.append(str(rel))
                continue
            node_type, name, metadata = live[usable[0]]
            current = brief_relpath(
                node_type, slug_map.get(usable[0]) or node_slug(name, metadata)
            )
        if rel != current:
            path.unlink()  # stranded by a rename, a retype, or the old flat layout
            removed.append(str(rel))
    return removed


def refile_briefs(
    conn: sqlite3.Connection, node_ids: set[str], out_dir: Path | None = None
) -> dict:
    """After a rename: a node that HAD a brief gets it rewritten at its new slug
    and the old one pruned; a node that had none gets none.

    The per-node emit only runs for nodes the scheduler finds stale, and a node
    with no brief on disk is never stale - so a rename left an unproposed node's
    page with no brief at all (the Disclosure Act, 2026-09-03: the old-slug brief
    was pruned, nothing wrote the new one, and the assembler's sweep found the
    page unowned). The brief follows the rename, whatever the proposal set says.
    """
    out_dir = out_dir or default_briefs_dir()
    had = {m for f in brief_files(out_dir) for m in brief_node_ids(f)} & set(node_ids)
    if not had:
        return {"written": [], "pruned": []}
    slug_map, _ = build_slug_map(conn)
    # NOT the proposal set: a rename of an UNPROPOSED node must still refile its
    # brief, which is the hole this function exists to close. A covered node
    # refiles the page that covers it; every other node refiles its own.
    from assimilator.pages import composed_pages

    covering = {
        m: {
            "node_ids": p["node_ids"],
            "page": {"name": p["name"], "slug": p["slug"], "node_type": p["node_type"]},
        }
        for p in composed_pages(conn)
        for m in p["node_ids"]
    }
    seen: set[tuple[str, ...]] = set()
    plan = []
    for node_id in sorted(had):
        entry = covering.get(node_id) or {"node_ids": [node_id], "page": None}
        key = tuple(entry["node_ids"])
        if key in seen:
            continue
        seen.add(key)
        plan.append(entry)
    written: list[str] = []
    for page in plan:
        brief = build_entity_brief(
            conn,
            page["node_ids"][0],
            slug_map,
            node_ids=page["node_ids"],
            page=page["page"],
        )
        if brief is None or not brief["claims"]:
            continue
        written.append(str(write_brief(brief, out_dir).relative_to(out_dir)))
    pruned = prune_retired_briefs(conn, out_dir, slug_map, had)
    return {"written": written, "pruned": pruned}


def page_set(conn: sqlite3.Connection) -> list[dict]:
    """Every page to emit a brief for: each composed page (several members,
    its own name and slug), then each proposed node on its own. A node covered
    by a composed page is not proposed separately, so nothing is emitted twice."""
    from assimilator.pages import composed_pages

    out = [
        {
            "node_ids": p["node_ids"],
            "page": {
                "name": p["name"],
                "slug": p["slug"],
                "node_type": p["node_type"],
            },
        }
        for p in composed_pages(conn)
    ]
    out.extend({"node_ids": [nid], "page": None} for nid in entity_node_ids(conn))
    return out


def emit_all(conn: sqlite3.Connection, out_dir: Path, on_progress=None) -> dict:
    log = on_progress or (lambda _: None)
    slug_map, collisions = build_slug_map(conn)
    written = 0
    for page in page_set(conn):
        brief = build_entity_brief(
            conn,
            page["node_ids"][0],
            slug_map,
            node_ids=page["node_ids"],
            page=page["page"],
        )
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
        str(data_dir() / "knowledge.db"),
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
            # The global map, not the per-node canonical slug: a same-type
            # collision loser emitted alone would otherwise land on the base
            # slug and overwrite the winner's brief.
            slug_map, _ = build_slug_map(conn)
            # A covered node has no brief of its own: emit the page that covers it.
            page = next(
                (p for p in page_set(conn) if args.node in p["node_ids"]),
                {"node_ids": [args.node], "page": None},
            )
            brief = build_entity_brief(
                conn,
                page["node_ids"][0],
                slug_map,
                node_ids=page["node_ids"],
                page=page["page"],
            )
            if brief is None:
                print(f"No such node: {args.node}")
                return 1
            print(f"Wrote {write_brief(brief, out_dir)}")
            for name in prune_retired_briefs(
                conn, out_dir, slug_map, set(page["node_ids"])
            ):
                print(f"Pruned {name} - the node's brief moved")
        else:
            result = emit_all(conn, out_dir, on_progress=print)
            print(f"Emitted {result['written']} briefs to {out_dir}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
