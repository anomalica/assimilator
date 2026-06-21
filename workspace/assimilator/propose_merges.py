"""Propose node-merge candidate clusters for human review in the workbench.

The graph is fragmented (the same entity split across nodes); this surfaces likely
duplicates as ranked candidate CLUSTERS for a human to confirm - it never merges
automatically (false merges are worse than missed ones).

Candidate sources, by confidence:
- name-equiv: same name modulo case + acronym-suffix, WITHIN a node type. Near
  certain (e.g. "remote viewing" / "Remote Viewing"). No AI needed.
- fuzzy: high Levenshtein name similarity within a type ("K. Day" / "Kevin Day").
- embedding (with --verify): semantically-similar name+claims, CROSS type allowed -
  this is the path that catches the Nimitz incident split across event+matter with
  divergent descriptive names. Cross-type candidates are surfaced ONLY after AI
  confirms they are the same entity, so the conservative bar protects against
  merging e.g. the ship (object) into the incident (event).

Output: merge-candidates.json, [{node_ids, suggested_canonical, score, node_type,
reason}], ranked by score. The workbench reads it as its candidate-review queue.

Deterministic candidates spend nothing. The --verify pass calls Claude (the
subscription transport) to confirm clusters - no dollars, but paced.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

from Levenshtein import ratio as levenshtein_ratio

from assimilator.matching import (
    FUZZY_NAME_THRESHOLD,
    _distinctive_tokens_disagree,
    name_equivalence_key,
)


def _claim_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT nid, COUNT(DISTINCT cid) FROM (
            SELECT speaker_id AS nid, id AS cid FROM claims WHERE speaker_id IS NOT NULL
            UNION
            SELECT node_id AS nid, claim_id AS cid FROM claim_node_refs
        ) GROUP BY nid
        """
    ).fetchall()
    return {nid: n for nid, n in rows}


def _active_nodes(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return conn.execute(
        "SELECT id, name, node_type FROM nodes WHERE retired_at IS NULL"
    ).fetchall()


def _name_equiv_edges(conn: sqlite3.Connection) -> list[tuple[str, str, str, float]]:
    """Pairwise edges among same-name nodes (modulo case + acronym suffix). A
    same-name group within one type is "name-equiv" (0.95); spanning types is
    "name-equiv-crosstype" (0.9) - the collisions surfaced flagged for review."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid, name, node_type in _active_nodes(conn):
        groups[name_equivalence_key(name)].append((nid, node_type))
    edges = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cross = len({m[1] for m in members}) > 1
        reason = "name-equiv-crosstype" if cross else "name-equiv"
        score = 0.9 if cross else 0.95
        ids = [m[0] for m in members]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.append((ids[i], ids[j], reason, score))
    return edges


def _fuzzy_edges(conn: sqlite3.Connection) -> list[tuple[str, str, str, float]]:
    """High-Levenshtein name pairs within a type, PRECISION-FILTERED: a pair that
    disagrees on a distinguishing token (a town in a structured place name, a
    section/bill number, a squadron designator) is rejected, so "USA, New Mexico,
    Roswell" never links to "...Aztec" and "Section 1632" never links to "1673".
    This is the #23 false-positive guard, applied before clustering."""
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid, name, node_type in _active_nodes(conn):
        by_type[node_type].append((nid, name))
    edges = []
    for members in by_type.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if levenshtein_ratio(a[1].lower(), b[1].lower()) < FUZZY_NAME_THRESHOLD:
                    continue
                if _distinctive_tokens_disagree(a[1], b[1]):
                    continue  # distinct entities sharing a structured prefix
                sim = levenshtein_ratio(a[1].lower(), b[1].lower())
                edges.append((a[0], b[0], "fuzzy", round(sim, 3)))
    return edges


def propose(conn: sqlite3.Connection) -> list[dict]:
    """Candidate clusters: one card per CONNECTED COMPONENT of the similarity
    graph, not N-choose-2 overlapping pairs. Edges are name-equiv + precision-
    filtered fuzzy; edges within a rejected (confirmed-distinct) set are removed
    (which can split a component), and already-merged nodes are absent (active
    nodes only). Each cluster is resolved once in the workbench. Embedding/cross-
    type semantic candidates are added by the --verify pass."""
    from assimilator.merge import rejected_sets

    counts = _claim_counts(conn)
    nodes = {nid: (name, ntype) for nid, name, ntype in _active_nodes(conn)}
    rejected = rejected_sets(conn)

    # fuzzy first, name-equiv last: a same-name pair also matches fuzzy (ratio
    # 1.0), and name-equiv is the more meaningful label, so it overwrites in
    # edge_meta below.
    edges = _fuzzy_edges(conn) + _name_equiv_edges(conn)
    edges = [e for e in edges if not any({e[0], e[1]} <= r for r in rejected)]

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_meta: dict[frozenset, tuple[str, float]] = {}
    for a, b, reason, score in edges:
        parent[find(a)] = find(b)
        edge_meta[frozenset((a, b))] = (reason, score)

    comps: dict[str, set[str]] = defaultdict(set)
    for pair in edge_meta:
        for nid in pair:
            comps[find(nid)].add(nid)

    out = []
    for members in comps.values():
        if len(members) < 2:
            continue
        internal = [edge_meta[k] for k in edge_meta if k <= members]
        best_reason, best_score = max(internal, key=lambda rs: rs[1])
        dominant = max(members, key=lambda n: counts.get(n, 0))
        out.append(
            {
                "node_ids": sorted(members),
                "suggested_canonical": nodes[dominant][0],
                "score": best_score,
                "node_type": nodes[dominant][1],
                "reason": best_reason,
            }
        )
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def default_candidates_path() -> Path:
    return Path(
        os.environ.get(
            "ANOMALICA_MERGE_CANDIDATES",
            str(
                Path.home()
                / ".local"
                / "share"
                / "assimilator"
                / "merge-candidates.json"
            ),
        )
    )


def write_candidates(candidates: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    p = argparse.ArgumentParser(
        prog="assimilator.propose_merges",
        description="Propose node-merge candidate clusters for human review.",
    )
    p.add_argument("--db", default=default_db)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    candidates = propose(conn)
    conn.close()
    out_path = Path(args.out) if args.out else default_candidates_path()
    write_candidates(candidates, out_path)
    by_reason: dict[str, int] = {}
    for c in candidates:
        by_reason[c["reason"]] = by_reason.get(c["reason"], 0) + 1
    print(f"Wrote {out_path}")
    print(f"  {len(candidates)} candidate clusters ({by_reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
