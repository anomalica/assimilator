"""Merge-candidate shortlist: every live node's nearest neighbours by profile.

The rules (name equivalence, a within-type Levenshtein) surface 19 of the 90
pairs reviewers have merged; a node's 20 nearest neighbours by name-plus-claims
vector contain the survivor for 75 of 85 (2026-09-02, reports/reranker-eval-
2026-09-02.md). This is the candidate SOURCE the reranker lacked. It is not a
queue: what it produces is scored afterwards, and nothing here merges.

The profile is deterministic in the graph's content (entity_reranker.
profile_claims): a shortlist whose recall moves with row order cannot be
trusted between imports, and this one did. Vectors come from the embedding
endpoint, which caches by text, so an unchanged profile costs nothing to
re-embed and a changed one is exactly a node whose claims changed.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from assimilator.entity_reranker import Entity, profile_claims

K_NEIGHBOURS = 20
EMBED_BATCH = 64


@dataclass(frozen=True)
class Profile:
    node_id: str
    name: str
    node_type: str
    text: str


def profile_of(
    conn: sqlite3.Connection, node_id: str, name: str, node_type: str
) -> Profile:
    return Profile(
        node_id,
        name,
        node_type,
        Entity(name, node_type, profile_claims(conn, node_id)).text(),
    )


def live_profiles(conn: sqlite3.Connection) -> list[Profile]:
    rows = conn.execute(
        "SELECT id, name, node_type FROM nodes WHERE retired_at IS NULL ORDER BY id"
    ).fetchall()
    return [profile_of(conn, nid, name, t) for nid, name, t in rows]


def embed_profiles(
    texts: Iterable[str], embed: Callable[[list[str]], list[list[float]]]
):
    """Unit-normalised vectors, one per text, through `embed` in batches."""
    import numpy as np

    texts = list(texts)
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out += embed(texts[i : i + EMBED_BATCH])
    m = np.asarray(out, dtype=np.float32)
    if m.size == 0:
        return m.reshape(0, 0)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def knn_pairs(ids: list[str], vectors, k: int = K_NEIGHBOURS) -> set[tuple[str, str]]:
    """Unordered pairs (a < b) from every node's k nearest, any type."""
    import numpy as np

    if len(ids) < 2:
        return set()
    k = min(k, len(ids) - 1)
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    pairs: set[tuple[str, str]] = set()
    for i, row in enumerate(top):
        for j in row:
            a, b = ids[i], ids[int(j)]
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def rank_of(query_vector, vectors, target_index: int) -> int:
    """1-based rank of the target among all vectors for the query."""
    sims = vectors @ query_vector
    return int((sims > sims[target_index]).sum()) + 1


def rules_pairs(path: Path, live_ids: set[str]) -> tuple[set[tuple[str, str]], int]:
    """Pairs implied by a propose_merges output, minus any whose node is no longer
    live - a candidate file older than the last curation replay names nodes the
    replay retired (stamped with the ledger's original date, so retired_at does
    not say when). Returns (pairs, dropped)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set(), 0
    clusters = data["clusters"] if isinstance(data, dict) else data
    pairs: set[tuple[str, str]] = set()
    dropped = 0
    for c in clusters:
        ids = sorted(c.get("node_ids") or [])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] in live_ids and ids[j] in live_ids:
                    pairs.add((ids[i], ids[j]))
                else:
                    dropped += 1
    return pairs, dropped


def shortlist(
    conn: sqlite3.Connection,
    embed: Callable[[list[str]], list[list[float]]],
    k: int = K_NEIGHBOURS,
    rules_path: Path | None = None,
) -> dict:
    """The candidate pairs: kNN over live profiles, unioned with the rules'."""
    profiles = live_profiles(conn)
    ids = [p.node_id for p in profiles]
    vectors = embed_profiles((p.text for p in profiles), embed)
    pairs = knn_pairs(ids, vectors, k)
    dropped = 0
    if rules_path is not None:
        rp, dropped = rules_pairs(rules_path, set(ids))
        pairs |= rp
    return {
        "profiles": profiles,
        "ids": ids,
        "vectors": vectors,
        "pairs": pairs,
        "rules_dropped": dropped,
    }
