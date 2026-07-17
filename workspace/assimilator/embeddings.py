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

# This ONNX export emits a uint8 tensor, not floats: the author asymmetrically
# quantised the model's float32 output onto 0..255 over the calibration range
# below (from the model card). Those two numbers are the decode key - without
# them a uint8 vector cannot be returned to the space it was measured in.
QUANT_MIN = -0.3009805381298065
QUANT_MAX = 0.3952634334564209

# The uint8 value meaning float 0.0 (~110). EVERY stored vector clusters around
# it, so a raw uint8 vector is a small signal riding on a large constant, and
# cosine between any two of them is ~0.99 regardless of meaning - the constant
# dominates. Dequantising re-centres on zero and recovers the signal. See
# ``_dequantise``.
QUANT_ZERO_POINT = -QUANT_MIN / (QUANT_MAX - QUANT_MIN) * 255

# The decode convention, not just the model, defines the vector space: the same
# ONNX file read raw and read dequantised produce vectors that are NOT comparable.
# It is part of the space's identity so a superseded convention is detectable
# rather than silently mixed.
DECODE_REVISION = "dequant-v1"

# Identity of the vector space every stored embedding was produced in. Any change
# to the model, its quantisation file, the dimensionality, or the decode
# convention yields a different string, so embeddings made by a superseded space
# are detectable (see ``stale_embedding_ids``) and never silently compared across
# incompatible spaces.
EMBEDDING_MODEL_ID = f"{MODEL_NAME}:{MODEL_FILE}:{EMBEDDING_DIMS}:{DECODE_REVISION}"

# The space of an embedding that predates this tracking table - which is to say,
# one that predates the decode fix, so a raw-uint8 vector from the degenerate
# space. Never a space to embed INTO; only a marker meaning "re-embed this".
PRE_TRACKING_MODEL_ID = "pre-tracking:raw-uint8:unusable"


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

        # POOLING MUST STAY DISABLED. This ONNX export pools internally and emits a
        # finished vector, so the model card's instruction is "execute model
        # without pooling and without normalization". Letting fastembed mean-pool
        # on top averages an already-pooled output and destroys the space: with
        # MEAN, EVERY pair of texts scored cosine ~0.992 - "the cat sat on the mat"
        # against an empty string included - a 0.003 spread carrying no usable
        # signal. It fails silently, returning plausible numbers, so nothing
        # downstream can detect it. Normalisation is likewise the caller's job
        # (``cosine_similarity`` here, ``_normalise`` in the shared client).
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.DISABLED,
            normalization=False,
            sources=ModelSource(hf=MODEL_NAME),
            dim=EMBEDDING_DIMS,
            model_file=MODEL_FILE,
        )

        kwargs = {"model_name": MODEL_NAME}
        if model_path and os.path.isdir(model_path):
            kwargs["specific_model_path"] = model_path

        _embedder = TextEmbedding(**kwargs)
    return _embedder


def _dequantise(raw: list[float]) -> list[float]:
    """uint8 0..255 back to the float32 range the model was calibrated over.

    Mandatory before any comparison. The raw tensor is asymmetrically quantised
    around QUANT_ZERO_POINT (~110), so every raw vector is dominated by that
    shared constant and cosine between any two is ~0.99 whatever they mean:
    measured on this corpus, unrelated text scored 0.992 and paraphrases 0.996 -
    a 0.004 spread. Dequantised, the same pairs score 0.22-0.30 and 0.67-0.79.
    Nothing errors in the raw case; it just returns confident nonsense."""
    scale = (QUANT_MAX - QUANT_MIN) / 255.0
    return [v * scale + QUANT_MIN for v in raw]


def embed_text(text: str) -> list[float]:
    embedder = _get_embedder()
    results = list(embedder.embed([text]))
    return _dequantise(results[0].tolist())


def embed_batch(texts: list[str]) -> list[list[float]]:
    """One text per inference, deliberately.

    Not a fastembed limitation: with pooling correctly DISABLED, passing a list
    works. It is rejected because it changes the answer. This ONNX pools
    internally, so a batch's padding leaks into the result - the same text
    embedded alongside others comes back at cosine ~0.95-0.97 to itself embedded
    alone, roughly a tenth of the paraphrase-vs-unrelated separation the space
    has to resolve. A vector would then depend on which other texts shared its
    request, which no cache can key on and no comparison can trust. Batching buys
    ~1.1x for that; not a trade worth making."""
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

-- Which embedding model (model+file+dims) produced every stored embedding, so an
-- upgrade can re-embed exactly the stale rows and clustering never mixes vectors
-- from different spaces. A companion table rather than columns on the vec0 virtual
-- tables, so it is trivially created and back-filled on the live database without
-- recreating the vector index.
CREATE TABLE IF NOT EXISTS embedding_model (
    kind        TEXT NOT NULL,   -- 'claim' or 'node'
    id          TEXT NOT NULL,   -- claim_id or node_id
    model_id    TEXT NOT NULL,   -- EMBEDDING_MODEL_ID at write time
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (kind, id)
);
"""


def init_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(VEC_SCHEMA)
    _backfill_embedding_model(conn)


def store_claim_embedding(
    conn: sqlite3.Connection,
    claim_id: str,
    embedding: list[float],
    embedder: str = EMBEDDING_MODEL_ID,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vec_claims(claim_id, embedding) VALUES (?, ?)",
        [claim_id, serialise_f32(embedding)],
    )
    _record_embedding_model(conn, "claim", claim_id, embedder)


def store_node_embedding(
    conn: sqlite3.Connection,
    node_id: str,
    embedding: list[float],
    embedder: str = EMBEDDING_MODEL_ID,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?, ?)",
        [node_id, serialise_f32(embedding)],
    )
    _record_embedding_model(conn, "node", node_id, embedder)


def _record_embedding_model(
    conn: sqlite3.Connection, kind: str, id_: str, model_id: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embedding_model(kind, id, model_id, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        [kind, id_, model_id, _now()],
    )


def _backfill_embedding_model(conn: sqlite3.Connection) -> None:
    """Stamp any embedding with no recorded model as PRE-TRACKING, so it reads as
    stale and gets re-embedded.

    This deliberately does NOT stamp them as current. An untracked embedding was
    written before this table existed, which is before the decode convention was
    fixed, so it is a raw-uint8 vector from the degenerate space where every pair
    scored ~0.99 - unusable, and indistinguishable from a good vector by
    inspection. Stamping those as current would bless them, and
    ``stale_embedding_ids`` would then never return them: the corruption would be
    permanent and silent.

    Idempotent (``INSERT OR IGNORE`` on unstamped rows only), so an already
    recorded model is preserved rather than overwritten."""
    now = _now()
    for kind, table, id_col in (
        ("claim", "vec_claims", "claim_id"),
        ("node", "vec_nodes", "node_id"),
    ):
        conn.execute(
            f"""
            INSERT OR IGNORE INTO embedding_model(kind, id, model_id, embedded_at)
            SELECT ?, {id_col}, ?, ?
            FROM {table}
            WHERE {id_col} NOT IN (
                SELECT id FROM embedding_model WHERE kind = ?
            )
            """,
            [kind, PRE_TRACKING_MODEL_ID, now, kind],
        )


def stale_embedding_ids(
    conn: sqlite3.Connection, kind: str, embedder: str = EMBEDDING_MODEL_ID
) -> list[str]:
    """Ids of embeddings NOT produced by ``embedder`` (default: the current model) -
    exactly the set to re-embed after a model upgrade. ``kind`` is 'claim' or
    'node'."""
    rows = conn.execute(
        "SELECT id FROM embedding_model WHERE kind = ? AND model_id != ?",
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
