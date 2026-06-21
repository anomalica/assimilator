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

from assimilator.matching import FUZZY_NAME_THRESHOLD, name_equivalence_key


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


def _canonical(members: list[tuple[str, str]], counts: dict[str, int]) -> str:
    """The member name with the most claims becomes the suggested canonical (the
    dominant form); a human can override in the UI."""
    return max(members, key=lambda m: counts.get(m[0], 0))[1]


def find_name_equiv(conn: sqlite3.Connection, counts: dict[str, int]) -> list[dict]:
    """Same name (modulo case + acronym suffix). Grouped across types: a same-name
    group within ONE type is high-confidence ("name-equiv"); a same-name group
    spanning types is flagged "name-equiv-crosstype" (the 17 collisions like
    "Special Access Programs" concept+matter) - surfaced for human review with the
    type difference prominent, NOT auto-merged. (The dangerous cross-type case is
    SEMANTIC similarity of different names; identical names are high signal.)"""
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for nid, name, node_type in _active_nodes(conn):
        groups[name_equivalence_key(name)].append((nid, name, node_type))
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        types = {m[2] for m in members}
        cross = len(types) > 1
        out.append(
            {
                "node_ids": sorted(m[0] for m in members),
                "suggested_canonical": _canonical(
                    [(m[0], m[1]) for m in members], counts
                ),
                "score": 0.9 if cross else 0.95,
                "node_type": max(members, key=lambda m: counts.get(m[0], 0))[2],
                "reason": "name-equiv-crosstype" if cross else "name-equiv",
            }
        )
    return out


def find_fuzzy(
    conn: sqlite3.Connection, counts: dict[str, int], exclude: set[frozenset]
) -> list[dict]:
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid, name, node_type in _active_nodes(conn):
        by_type[node_type].append((nid, name))
    out = []
    for node_type, members in by_type.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = frozenset((a[0], b[0]))
                if key in exclude:
                    continue
                sim = levenshtein_ratio(a[1].lower(), b[1].lower())
                if sim < FUZZY_NAME_THRESHOLD:
                    continue
                exclude.add(key)
                out.append(
                    {
                        "node_ids": sorted([a[0], b[0]]),
                        "suggested_canonical": _canonical([a, b], counts),
                        "score": round(sim, 3),
                        "node_type": node_type,
                        "reason": "fuzzy",
                    }
                )
    return out


def propose(conn: sqlite3.Connection) -> list[dict]:
    """Deterministic candidate clusters (name-equiv + fuzzy, within type), ranked
    by score. Embedding/cross-type candidates are added by the --verify pass."""
    counts = _claim_counts(conn)
    name_equiv = find_name_equiv(conn, counts)
    seen = {frozenset(c["node_ids"]) for c in name_equiv}
    fuzzy = find_fuzzy(conn, counts, seen)
    candidates = name_equiv + fuzzy
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


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
