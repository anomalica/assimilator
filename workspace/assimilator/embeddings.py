"""Embedding generation and similarity search.

Uses fastembed for in-process embedding with no external service dependency.
In development, can optionally fall back to Ollama via the ollama library.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from datetime import datetime, timezone

import sqlite_vec

MODEL_NAME = "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8"
MODEL_FILE = "dynamic_uint8.onnx"
EMBEDDING_DIMS = 1024

# Identity of the vector space every stored embedding was produced in. Any change
# to the model, its quantisation file, or the dimensionality yields a different
# string, so embeddings made by a superseded embedder are detectable (see
# ``stale_embedding_ids``) and never silently compared across incompatible spaces.
CURRENT_EMBEDDER = f"{MODEL_NAME}:{MODEL_FILE}:{EMBEDDING_DIMS}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        import os

        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        model_path = os.environ.get("EMBEDDING_MODEL_PATH")

        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=MODEL_NAME),
            dim=EMBEDDING_DIMS,
            model_file=MODEL_FILE,
        )

        kwargs = {"model_name": MODEL_NAME}
        if model_path and os.path.isdir(model_path):
            kwargs["specific_model_path"] = model_path

        _embedder = TextEmbedding(**kwargs)
    return _embedder


def embed_text(text: str) -> list[float]:
    embedder = _get_embedder()
    results = list(embedder.embed([text]))
    return results[0].tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    # Loop rather than passing a list to fastembed.embed - the custom Qwen3
    # ONNX model registered via add_custom_model breaks fastembed's
    # mean_pooling when batch_size > 1 (attention-mask shape mismatch). Per-
    # item calls are slower but actually work.
    if not texts:
        return []
    return [embed_text(t) for t in texts]


def serialise_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def deserialise_f32(raw: bytes) -> list[float]:
    count = len(raw) // 4
    return list(struct.unpack(f"{count}f", raw))


VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_claims USING vec0(
    claim_id TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMS}] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding float[{EMBEDDING_DIMS}] distance_metric=cosine
);

-- Provenance for every stored embedding: which embedder (model+file+dims) produced
-- it, so an embedder upgrade can re-embed exactly the stale rows and clustering
-- never mixes vectors from different spaces. A companion table rather than columns
-- on the vec0 virtual tables, so it is trivially created and back-filled on the
-- live database without recreating the vector index.
CREATE TABLE IF NOT EXISTS embedding_provenance (
    kind        TEXT NOT NULL,   -- 'claim' or 'node'
    id          TEXT NOT NULL,   -- claim_id or node_id
    embedder    TEXT NOT NULL,   -- CURRENT_EMBEDDER at write time
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
"""


def init_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(VEC_SCHEMA)
    _backfill_provenance(conn)


def store_claim_embedding(
    conn: sqlite3.Connection,
    claim_id: str,
    embedding: list[float],
    embedder: str = CURRENT_EMBEDDER,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vec_claims(claim_id, embedding) VALUES (?, ?)",
        [claim_id, serialise_f32(embedding)],
    )
    _record_provenance(conn, "claim", claim_id, embedder)


def store_node_embedding(
    conn: sqlite3.Connection,
    node_id: str,
    embedding: list[float],
    embedder: str = CURRENT_EMBEDDER,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?, ?)",
        [node_id, serialise_f32(embedding)],
    )
    _record_provenance(conn, "node", node_id, embedder)


def _record_provenance(
    conn: sqlite3.Connection, kind: str, id_: str, embedder: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embedding_provenance(kind, id, embedder, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        [kind, id_, embedder, _now()],
    )


def _backfill_provenance(conn: sqlite3.Connection) -> None:
    """Stamp any pre-existing embedding that lacks provenance with the current
    embedder. Safe because every embedding in the database to date was produced by
    the current model; it is idempotent (``INSERT OR IGNORE`` on rows with no
    existing provenance), so a later embedder change is preserved rather than
    overwritten."""
    now = _now()
    for kind, table, id_col in (
        ("claim", "vec_claims", "claim_id"),
        ("node", "vec_nodes", "node_id"),
    ):
        conn.execute(
            f"""
            INSERT OR IGNORE INTO embedding_provenance(kind, id, embedder, embedded_at)
            SELECT ?, {id_col}, ?, ?
            FROM {table}
            WHERE {id_col} NOT IN (
                SELECT id FROM embedding_provenance WHERE kind = ?
            )
            """,
            [kind, CURRENT_EMBEDDER, now, kind],
        )


def stale_embedding_ids(
    conn: sqlite3.Connection, kind: str, embedder: str = CURRENT_EMBEDDER
) -> list[str]:
    """Ids of embeddings NOT produced by ``embedder`` (default: the current one) -
    exactly the set to re-embed after an embedder upgrade. ``kind`` is 'claim' or
    'node'."""
    rows = conn.execute(
        "SELECT id FROM embedding_provenance WHERE kind = ? AND embedder != ?",
        [kind, embedder],
    ).fetchall()
    return [r[0] for r in rows]


def search_similar_claims(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10,
) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT claim_id, distance
        FROM vec_claims
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [serialise_f32(query_embedding), limit],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def search_similar_nodes(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10,
) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT node_id, distance
        FROM vec_nodes
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [serialise_f32(query_embedding), limit],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
