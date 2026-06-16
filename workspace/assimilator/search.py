"""Hybrid search combining embedding similarity with IDF-weighted keyword matching.

Embedding search alone struggles with short keyword queries (e.g. a query like
"BMI" matches anything vaguely health-shaped rather than the specific term).
Keyword search alone misses paraphrases. Reciprocal Rank Fusion (RRF) merges
both ranked lists so each compensates for the other's weakness.

An optional cross-encoder reranking pass can be applied on top for vocabulary-gap
bridging - it scores (query, claim) pairs directly rather than via independent
embeddings.
"""

from __future__ import annotations

import math
import re
import sqlite3

from assimilator.embeddings import (
    deserialise_f32,
    embed_text,
    search_similar_claims,
)

RRF_K = 60
KEYWORD_BOOST = 0.25
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None

_STOP_WORDS = frozenset(
    {
        "the",
        "is",
        "at",
        "in",
        "on",
        "of",
        "to",
        "for",
        "an",
        "and",
        "or",
        "it",
        "by",
        "do",
        "does",
        "did",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "how",
        "which",
        "that",
        "this",
        "with",
        "from",
        "are",
        "not",
        "no",
        "any",
        "all",
        "some",
        "can",
        "will",
        "about",
        "his",
        "her",
        "their",
        "my",
        "its",
        "as",
    }
)


def _tokenise_query(query: str) -> list[str]:
    raw = re.split(r"[\"'\s,;:!?.()]+", query.strip())
    tokens = [t.lower() for t in raw if len(t) >= 2]
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    return filtered if filtered else tokens


def _compute_idf_weights(conn: sqlite3.Connection, tokens: list[str]) -> list[float]:
    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    if total == 0:
        return [0.0 for _ in tokens]
    weights = []
    for token in tokens:
        doc_freq = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE LOWER(content) LIKE ?",
            [f"%{token}%"],
        ).fetchone()[0]
        weights.append(math.log(1 + total / (1 + doc_freq)))
    return weights


def keyword_search_claims(
    conn: sqlite3.Connection, query: str, limit: int = 30
) -> list[tuple[str, float]]:
    """IDF-weighted keyword search against claims.content.

    Returns (claim_id, score) pairs where score is the weighted fraction of
    query tokens matched. Rare terms contribute more than common ones.
    """
    tokens = _tokenise_query(query)
    if not tokens:
        return []

    idf_weights = _compute_idf_weights(conn, tokens)
    total_idf = sum(idf_weights)
    if total_idf == 0:
        return []

    cases = []
    score_params: list[str] = []
    for token, weight in zip(tokens, idf_weights):
        cases.append(f"CASE WHEN LOWER(content) LIKE ? THEN {weight} ELSE 0.0 END")
        score_params.append(f"%{token}%")

    where_clause = " OR ".join("LOWER(content) LIKE ?" for _ in tokens)
    where_params = [f"%{t}%" for t in tokens]

    score_expr = " + ".join(cases)
    sql = f"""
        SELECT id, ({score_expr}) / {total_idf} AS score
        FROM claims
        WHERE ({where_clause})
        ORDER BY score DESC
        LIMIT ?
    """  # noqa: S608  -- score_expr/where_clause built from token count, not user input
    params = score_params + where_params + [limit]

    rows = conn.execute(sql, params).fetchall()
    return [(row[0], row[1]) for row in rows]


def rrf_merge(
    *ranked_lists: list[tuple[str, float]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: merge ranked lists by reciprocal rank.

    Each input is (id, score) sorted by score descending. Output is
    (id, rrf_score) sorted by rrf_score descending. Score values from the
    inputs are not used - only the rank position.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search_claims(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float] | None = None,
    limit: int = 10,
    rerank: bool = False,
) -> list[tuple[str, float]]:
    """Hybrid search: embedding + keyword, fused with RRF.

    Returns (claim_id, distance) pairs compatible with search_similar_claims.
    Distance is 1 - display_similarity, where display_similarity blends
    semantic similarity with a keyword boost.

    When rerank=True, applies a cross-encoder rerank pass to the candidate
    pool before truncating to `limit`.
    """
    if query_embedding is None:
        query_embedding = embed_text(query)

    candidate_limit = max(limit * 2, 50) if rerank else limit
    sem_results = search_similar_claims(conn, query_embedding, limit=candidate_limit)
    kw_results = keyword_search_claims(conn, query, limit=candidate_limit)

    if not kw_results:
        candidates = sem_results
    else:
        sem_scores: dict[str, float] = {cid: 1.0 - dist for cid, dist in sem_results}
        kw_scores: dict[str, float] = dict(kw_results)

        sem_scored = [(cid, 1.0 - dist) for cid, dist in sem_results]
        merged = rrf_merge(sem_scored, kw_results)

        if not merged:
            return []

        candidates = []
        for cid, _rrf_score in merged[:candidate_limit]:
            sem_sim = sem_scores.get(cid)
            if sem_sim is None:
                emb_row = conn.execute(
                    "SELECT embedding FROM vec_claims WHERE claim_id = ?", (cid,)
                ).fetchone()
                if emb_row:
                    claim_emb = deserialise_f32(emb_row[0])
                    sem_sim = _cosine(claim_emb, query_embedding)
                else:
                    sem_sim = 0.0
            kw_frac = kw_scores.get(cid, 0.0)
            display = sem_sim + KEYWORD_BOOST * kw_frac
            candidates.append((cid, 1.0 - display))

    if not candidates:
        return []

    if rerank and len(candidates) > 1:
        return _rerank_claims(conn, query, candidates, limit)

    candidates.sort(key=lambda x: x[1])
    return candidates[:limit]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_reranker(model_name: str = DEFAULT_RERANKER):
    global _reranker
    if _reranker is None:
        import os

        from sentence_transformers import CrossEncoder

        local_path = os.environ.get("RERANKER_MODEL_PATH")
        model = local_path if local_path and os.path.isdir(local_path) else model_name
        _reranker = CrossEncoder(model)
    return _reranker


def rerank_pairs(
    pairs: list[tuple[str, str]], model_name: str = DEFAULT_RERANKER
) -> list[float]:
    """Score a list of (query, document) pairs with a cross-encoder.

    Returns raw logit scores; use sigmoid to map to [0, 1] for comparison
    across calls.
    """
    if not pairs:
        return []
    reranker = _get_reranker(model_name)
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _rerank_claims(
    conn: sqlite3.Connection,
    query: str,
    candidates: list[tuple[str, float]],
    limit: int,
) -> list[tuple[str, float]]:
    content_by_id: dict[str, str] = {}
    rows = conn.execute(
        f"SELECT id, content FROM claims WHERE id IN ({','.join('?' * len(candidates))})",  # noqa: S608
        [cid for cid, _ in candidates],
    ).fetchall()
    for cid, content in rows:
        content_by_id[cid] = content

    pairs: list[tuple[str, str]] = []
    paired_ids: list[str] = []
    for cid, _ in candidates:
        content = content_by_id.get(cid)
        if content is None:
            continue
        pairs.append((query, content))
        paired_ids.append(cid)

    if not pairs:
        return []

    scores = rerank_pairs(pairs)
    scored = [(cid, 1.0 - _sigmoid(score)) for cid, score in zip(paired_ids, scores)]
    scored.sort(key=lambda x: x[1])
    return scored[:limit]
