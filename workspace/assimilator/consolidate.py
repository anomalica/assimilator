"""Claim deduplication via embedding similarity.

Groups similar claims using cosine similarity on embeddings, then uses
AI to decide whether groups should be merged (same underlying fact) or
kept separate (genuinely different assertions).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from assimilator.embeddings import (
    cosine_similarity,
    embed_batch,
    init_vec,
    search_similar_claims,
    store_claim_embedding,
)
from anomalica_common.llm import _call_cli, _parse_json
from anomalica_common.digest.models import Claim

CONSOLIDATION_PROMPT = """You are deduplicating claims extracted from multiple records in a knowledge graph.
Below are groups of similar claims. For each group, decide whether they represent
the SAME underlying assertion or genuinely DIFFERENT assertions.

RULES:
- "merge": the claims say the same thing (possibly with different wording or detail).
  Pick the most complete, specific version as the kept content.
- "keep_all": the claims are genuinely different (different details, subjects, or meaning).
- If versions contradict each other (different numbers, dates, or claims), use "keep_all".
- False merges are WORSE than missed deduplication. When in doubt, use "keep_all".

{groups_text}

OUTPUT FORMAT (respond with ONLY valid JSON, no markdown fencing):

{{"decisions": [
    {{"group_id": 1, "action": "merge", "kept_content": "best version of the claim"}},
    {{"group_id": 2, "action": "keep_all"}}
]}}"""


@dataclass
class DeduplicationResult:
    merged_count: int = 0
    kept_count: int = 0
    new_count: int = 0


def deduplicate_claims(
    conn: sqlite3.Connection,
    claims: list[Claim],
    similarity_threshold: float = 0.80,
    model: str = "sonnet",
    on_progress: callable = None,
) -> DeduplicationResult:
    """Check new claims against existing claims and each other for duplicates."""
    log = on_progress or (lambda _: None)
    result = DeduplicationResult()

    if not claims:
        return result

    log(f"Embedding {len(claims)} claims...")
    texts = [c.content for c in claims]
    embeddings = embed_batch(texts)

    # Check each claim against existing store
    init_vec(conn)
    duplicates: set[int] = set()

    for i, (claim, embedding) in enumerate(zip(claims, embeddings)):
        matches = search_similar_claims(conn, embedding, limit=3)
        for existing_id, distance in matches:
            similarity = 1.0 - distance
            if similarity >= similarity_threshold:
                log(
                    f"  Duplicate found: '{claim.content[:60]}...' matches existing [{existing_id[:8]}]"
                )
                duplicates.add(i)
                result.merged_count += 1
                break

    # Cluster remaining (non-duplicate) claims among themselves
    remaining = [
        (i, claims[i], embeddings[i]) for i in range(len(claims)) if i not in duplicates
    ]

    if len(remaining) > 1:
        clusters = _find_clusters(remaining, similarity_threshold)
        multi = [c for c in clusters if len(c) > 1]

        if multi:
            log(
                f"  Found {len(multi)} clusters of similar new claims, sending to AI..."
            )
            ai_decisions = _consolidate_clusters(multi, model=model)

            for cluster, decision in zip(multi, ai_decisions):
                if decision.get("action") == "merge":
                    kept_content = decision.get("kept_content", cluster[0][1].content)
                    # Store only the kept version, mark others as merged
                    for j, (idx, claim, emb) in enumerate(cluster):
                        if claim.content == kept_content or j == 0:
                            store_claim_embedding(conn, claim.id, emb)
                            result.new_count += 1
                        else:
                            result.merged_count += 1
                            duplicates.add(idx)
                else:
                    for idx, claim, emb in cluster:
                        store_claim_embedding(conn, claim.id, emb)
                        result.kept_count += 1

    # Store embeddings for all non-duplicate, non-clustered claims
    for i, claim, emb in remaining:
        if i not in duplicates and claim.id not in _stored_ids(conn):
            store_claim_embedding(conn, claim.id, emb)
            result.new_count += 1

    return result


def _stored_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT claim_id FROM vec_claims").fetchall()
    return {row[0] for row in rows}


def _find_clusters(
    items: list[tuple[int, Claim, list[float]]],
    threshold: float,
) -> list[list[tuple[int, Claim, list[float]]]]:
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(items[i][2], items[j][2])
            if sim >= threshold:
                union(i, j)

    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    return [[items[i] for i in indices] for indices in clusters_map.values()]


def _consolidate_clusters(
    clusters: list[list[tuple[int, Claim, list[float]]]],
    model: str = "sonnet",
) -> list[dict]:
    lines = []
    for group_id, cluster in enumerate(clusters, 1):
        lines.append(f"GROUP {group_id}:")
        for _, claim, _ in cluster:
            lines.append(f'  [{claim.claim_type.value}] "{claim.content}"')
        lines.append("")

    groups_text = "\n".join(lines)
    prompt = CONSOLIDATION_PROMPT.format(groups_text=groups_text)
    raw = _call_cli(prompt, "", model)
    data = _parse_json(raw)
    decisions = data.get("decisions", [])

    # Pad with keep_all for any missing groups
    decision_map = {d.get("group_id"): d for d in decisions}
    return [
        decision_map.get(i + 1, {"action": "keep_all"}) for i in range(len(clusters))
    ]
