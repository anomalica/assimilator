import sqlite3

from assimilator.embeddings import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL_ID,
    init_vec,
    serialise_f32,
    stale_embedding_ids,
    store_claim_embedding,
    store_node_embedding,
)

OLD_MODEL_ID = "legacy-model:legacy.onnx:512"


def _vec_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_vec(conn)
    return conn


def _vec() -> list[float]:
    return [0.1] * EMBEDDING_DIMS


def _model_id(conn, kind, id_):
    row = conn.execute(
        "SELECT model_id FROM embedding_model WHERE kind = ? AND id = ?",
        [kind, id_],
    ).fetchone()
    return row[0] if row else None


def test_store_records_current_model():
    conn = _vec_db()
    store_claim_embedding(conn, "c1", _vec())
    store_node_embedding(conn, "n1", _vec())

    assert _model_id(conn, "claim", "c1") == EMBEDDING_MODEL_ID
    assert _model_id(conn, "node", "n1") == EMBEDDING_MODEL_ID


def test_store_with_explicit_model():
    conn = _vec_db()
    store_claim_embedding(conn, "c1", _vec(), embedder=OLD_MODEL_ID)
    assert _model_id(conn, "claim", "c1") == OLD_MODEL_ID


def test_stale_detection_both_directions():
    conn = _vec_db()
    store_claim_embedding(conn, "current", _vec())
    store_claim_embedding(conn, "legacy", _vec(), embedder=OLD_MODEL_ID)

    # Against the current model, only the legacy row is stale.
    assert stale_embedding_ids(conn, "claim") == ["legacy"]
    # Against the legacy model, only the current row is stale.
    assert stale_embedding_ids(conn, "claim", embedder=OLD_MODEL_ID) == ["current"]


def test_stale_detection_is_kind_scoped():
    conn = _vec_db()
    store_claim_embedding(conn, "c_legacy", _vec(), embedder=OLD_MODEL_ID)
    store_node_embedding(conn, "n_current", _vec())

    assert stale_embedding_ids(conn, "claim") == ["c_legacy"]
    assert stale_embedding_ids(conn, "node") == []


def test_backfill_stamps_legacy_rows():
    conn = _vec_db()
    # A row written straight to the vector index with no recorded model, as a
    # pre-tracking embedding would appear.
    conn.execute(
        "INSERT INTO vec_claims(claim_id, embedding) VALUES (?, ?)",
        ["orphan", serialise_f32(_vec())],
    )
    assert _model_id(conn, "claim", "orphan") is None

    init_vec(conn)  # runs the back-fill

    assert _model_id(conn, "claim", "orphan") == EMBEDDING_MODEL_ID


def test_backfill_preserves_existing_model():
    conn = _vec_db()
    conn.execute(
        "INSERT INTO vec_nodes(node_id, embedding) VALUES (?, ?)",
        ["n1", serialise_f32(_vec())],
    )
    conn.execute(
        "INSERT INTO embedding_model(kind, id, model_id, embedded_at) "
        "VALUES ('node', 'n1', ?, '2020-01-01T00:00:00+00:00')",
        [OLD_MODEL_ID],
    )

    init_vec(conn)  # must not clobber the recorded model

    assert _model_id(conn, "node", "n1") == OLD_MODEL_ID
    assert stale_embedding_ids(conn, "node") == ["n1"]
