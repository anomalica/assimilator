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

RERANKING (ASSIMILATOR_ENTITY_RERANKER=1, or --rerank): the same clusters, with
each member pair scored by a local cross-encoder reading both names, both
types and a few claims from each side (entity_reranker), and the queue ordered
by that score instead of the rule that found the pair. The rule's score and
reason are kept beside it. Off by default until the evaluation against the
curation ledger says it should be on; see reports/reranker-eval-*.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

from assimilator.matching import (
    FUZZY_NAME_THRESHOLD,
    fuzzy_name_similarity,
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
    """Similar-name pairs within a type, scored by the SAME structure-aware metric
    match_node uses (fuzzy_name_similarity), not raw Levenshtein. It compares
    comma-structured names component-wise and collapses to 0 when a required
    component differs - so distinct entities sharing a hierarchy prefix don't
    link: "USA, New Mexico, Roswell"/"...Aztec", "USA, Nevada, S4"/"...Fallon"
    (leaf differs), "NDAA Section 1632"/"1673" (hard-token conflict). The #23
    precision guard, applied before clustering."""
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid, name, node_type in _active_nodes(conn):
        by_type[node_type].append((nid, name))
    edges = []
    for members in by_type.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                sim = fuzzy_name_similarity(a[1].lower(), b[1].lower())
                if sim >= FUZZY_NAME_THRESHOLD:
                    edges.append((a[0], b[0], "fuzzy", round(sim, 3)))
    return edges


def propose(conn: sqlite3.Connection, rerank: bool = False) -> list[dict]:
    """Candidate clusters: one card per CONNECTED COMPONENT of the similarity
    graph, not N-choose-2 overlapping pairs. Edges are name-equiv + precision-
    filtered fuzzy; edges within a rejected (confirmed-distinct) set are removed
    (which can split a component), and already-merged nodes are absent (active
    nodes only). Each cluster is resolved once in the workbench. Embedding/cross-
    type semantic candidates are added by the --verify pass.

    With rerank, every member pair of every cluster is scored by the entity
    reranker and the cluster's `score` becomes the best pair's; `rule_score`
    and `reason` keep what found it. The reranker never adds or removes a
    candidate here - it orders what the rules surfaced."""
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
    if rerank:
        rerank_clusters(conn, out)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def rerank_clusters(
    conn: sqlite3.Connection, clusters: list[dict], model_id: str | None = None
) -> dict:
    """Score each cluster's member pairs with the entity reranker, in place.
    Returns the run's figures for the run record: pairs, prompts, device, peak
    device memory.

    `score` becomes the highest pair score (the strongest reason to look), the
    rule's own score moves to `rule_score`, and `pairs` records every pair
    with both numbers so the workbench can show WHY a cluster ranks where it
    does. A cluster of one pair is the common case; a larger one is scored
    pair by pair rather than as a whole, because "same entity" is a relation
    between two things."""
    from assimilator.entity_reranker import (
        DEFAULT_MODEL_ID,
        entity_from_graph,
        get_reranker,
    )

    model_id = model_id or DEFAULT_MODEL_ID
    entities: dict[str, object] = {}
    pairs: list[tuple[int, str, str]] = []
    for ci, cluster in enumerate(clusters):
        ids = cluster["node_ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ci, ids[i], ids[j]))
        for nid in ids:
            if nid not in entities:
                entities[nid] = entity_from_graph(conn, nid)
    if not pairs:
        return {"pairs": 0, "prompts": 0, "device": None, "gpu_peak_mb": None}
    reranker = get_reranker(model_id)
    scores = reranker.score([(entities[a], entities[b]) for _, a, b in pairs])
    for cluster in clusters:
        cluster["rule_score"] = cluster["score"]
        cluster["pairs"] = []
        cluster["score"] = 0.0
    for (ci, a, b), s in zip(pairs, scores):
        cluster = clusters[ci]
        cluster["pairs"].append({"node_ids": [a, b], "reranker": round(s, 4)})
        cluster["score"] = max(cluster["score"], round(s, 4))
    return {
        "pairs": len(pairs),
        "prompts": len(pairs) * 2,  # both orders, averaged
        "device": reranker.device,
        "gpu_peak_mb": reranker.peak_memory_mb(),
    }


def default_runs_path() -> Path:
    return Path(
        os.environ.get(
            "ASSIMILATOR_RERANK_RUNS",
            str(Path.home() / ".local" / "share" / "assimilator" / "rerank-runs.jsonl"),
        )
    )


def append_run_record(record: dict, path: Path | None = None) -> Path:
    """One line per reranker run, in the AI-operation ledger's vocabulary
    (ai-ledger-format.md: component, operation, transport, model_id,
    duration_s, outcome, target) plus the run's own figures. The ledger's
    shared writer is not built yet (decision 0037 is scaffolded), so this file
    is what the scheduler reads and what folds into the ledger when it lands."""
    path = path or default_runs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


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


def _run_record(
    model_id, started, started_at, run, out_path, outcome, error, clusters=None
) -> dict:
    ended = time.time()
    return {
        "schema": "anomalica/rerank-run/1",
        "component": "assimilator",
        "operation": "rerank",
        "transport": "gpu" if (run or {}).get("device") == "cuda" else "cpu",
        "model_id": model_id,
        "timestamp_start": started_at,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        "duration_s": round(ended - started, 1),
        "outcome": outcome,
        "error": error,
        "target": str(out_path),
        "clusters": clusters,
        "pairs": (run or {}).get("pairs"),
        "prompts": (run or {}).get("prompts"),
        "gpu_peak_mb": (run or {}).get("gpu_peak_mb"),
    }


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
    p.add_argument(
        "--rerank",
        action="store_true",
        default=os.environ.get("ASSIMILATOR_ENTITY_RERANKER") == "1",
        help="Order the queue by the entity reranker (local model, no spend).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Reranker model id, resolved through model-policy.yaml stage `rerank` "
        "(default: the policy's first permitted model). Refused if not permitted.",
    )
    args = p.parse_args(argv)
    out_path = Path(args.out) if args.out else default_candidates_path()

    model_id = None
    if args.rerank:
        # POLICY BEFORE LOADING, fail closed: an unlisted model is refused, and
        # an unreadable policy refuses too - a rerank that ran on a model nobody
        # approved is a model choice nobody made.
        try:
            from anomalica_common import model_policy as mp

            policy = mp.load()
            model_id = (
                policy.check("rerank", args.model)
                if args.model
                else policy.choose("rerank")
            )
        except Exception as exc:  # noqa: BLE001 - every failure here is a refusal
            print(f"rerank refused: {exc}", file=sys.stderr)
            return 2

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    try:
        candidates = propose(conn)
        run = None
        if args.rerank:
            run = rerank_clusters(conn, candidates, model_id)
            candidates.sort(key=lambda c: c["score"], reverse=True)
    except FileNotFoundError as exc:
        print(f"rerank refused: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        if args.rerank:
            append_run_record(
                _run_record(
                    model_id, started, started_at, None, out_path, "error", str(exc)
                )
            )
        raise
    finally:
        conn.close()
    write_candidates(candidates, out_path)
    if args.rerank:
        record = _run_record(
            model_id, started, started_at, run, out_path, "ok", None, len(candidates)
        )
        append_run_record(record)
        print(json.dumps(record))
    by_reason: dict[str, int] = {}
    for c in candidates:
        by_reason[c["reason"]] = by_reason.get(c["reason"], 0) + 1
    print(f"Wrote {out_path}")
    print(f"  {len(candidates)} candidate clusters ({by_reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
