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

from anomalica_common.slug import node_slug
from assimilator.database import get_independent_source_count
from assimilator.propose_pages import proposed_node_ids

SCHEMA = "anomalica/brief/1"

# Per-brief claim cap. A hub entity can be referenced by thousands of claims,
# which would make one brief too large for the assembler to render in a single
# pass. Until evidence-scoring is pinned (which will RANK claims so the cap keeps
# the strongest), claims are ordered chronologically and the first MAX are kept;
# claim_count vs claim_count_total on the page makes any truncation explicit, not
# silent. Tunable via ANOMALICA_BRIEF_MAX_CLAIMS.
MAX_CLAIMS = int(os.environ.get("ANOMALICA_BRIEF_MAX_CLAIMS", "200"))


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
        "SELECT id, name, metadata FROM nodes WHERE retired_at IS NULL"
    ).fetchall()
    by_base: dict[str, list[tuple[str, str]]] = {}
    for node_id, name, metadata in rows:
        by_base.setdefault(node_slug(name, metadata), []).append((node_id, name))

    slug_map: dict[str, str] = {}
    collisions: list[dict] = []
    for base, members in by_base.items():
        if len(members) == 1:
            slug_map[members[0][0]] = base
            continue
        members.sort(key=lambda m: m[0])  # deterministic winner
        collisions.append(
            {
                "slug": base,
                "nodes": [m[0] for m in members],
                "names": [m[1] for m in members],
            }
        )
        for i, (node_id, _name) in enumerate(members):
            slug_map[node_id] = base if i == 0 else f"{base}-{node_id[:8]}"
    return slug_map, collisions


LOAD_BEARING_ATTESTATIONS = ("second_hand", "third_hand")


def _attribution_is_load_bearing(
    origin_kind: str | None, claim_type: str | None, attestation: str | None
) -> bool:
    """Whether the claim's TEXT already carries its own attribution inline, and so
    must be rendered exactly as written - never re-wrapped in a second "According
    to X", and never stripped back to a bare assertion (ADR 0044 s.3).

    For such a claim the fact about the world is *that someone asserted this*, not
    the assertion itself: strip "an anonymous source claiming to work inside the
    Defense Intelligence Agency said" and a true statement about an assertion
    becomes a false statement about reality.

    Derived once, here, rather than left for each consumer to re-implement - the
    brief is the contract, and a rule re-derived in three places drifts in three
    directions. The raw ``provenance_chain`` travels alongside it, so a consumer
    that wants to reason from the chain itself still can.

    On a pre-0044 claim the chain is absent, so this can only fire on claim_type
    and attestation. That is weaker, not wrong - it is exactly the blindness 0044
    closes, and a re-digest lifts it.
    """
    return (
        origin_kind == "anonymous"
        or claim_type == "hearsay"
        or attestation in LOAD_BEARING_ATTESTATIONS
    )


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
               c.origin_kind, c.origin, c.relay
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
    claims: list[dict] = []
    ordered_pairs: list[tuple[str, str]] = []
    for row in rows[:MAX_CLAIMS]:
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
                # which is the RECORD's source metadata. A consumer reads
                # origin_kind to know whether the claim's text already carries its
                # attribution inline (anonymous / hearsay / second- or third-hand
                # claims do) and so must be rendered as-is, never re-hedged and
                # never stripped. Null when the digest predates 0044.
                "provenance_chain": {
                    "origin_kind": origin_kind,
                    "origin": origin or "",
                    "relay": json.loads(relay) if relay else [],
                }
                if origin_kind
                else None,
                "attribution_is_load_bearing": _attribution_is_load_bearing(
                    origin_kind, claim_type, attestation
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
    path.write_text(yaml.safe_dump(brief, sort_keys=False, allow_unicode=True))
    return path


def default_briefs_dir() -> Path:
    return Path(
        os.environ.get(
            "ANOMALICA_BRIEFS_DIR",
            str(Path.home() / ".local" / "share" / "assimilator" / "briefs"),
        )
    )


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
    if collisions:
        log(
            f"NOTE: {len(collisions)} slug collisions disambiguated by node-id "
            f"suffix - review for entity-matcher merge bugs (same entity split):"
        )
        for c in collisions:
            log(f"  {c['slug']}: {c['names']}")
    return {"written": written, "collisions": collisions}


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
