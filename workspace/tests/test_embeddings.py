import sqlite3

from assimilator.embeddings import (
    CURRENT_EMBEDDER,
    EMBEDDING_DIMS,
    init_vec,
    serialise_f32,
    stale_embedding_ids,
    store_claim_embedding,
    store_node_embedding,
)

OLD_EMBEDDER = "legacy-model:legacy.onnx:512"


def _vec_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_vec(conn)
    return conn


def _vec() -> list[float]:
    return [0.1] * EMBEDDING_DIMS


def _provenance(conn, kind, id_):
    row = conn.execute(
        "SELECT embedder FROM embedding_provenance WHERE kind = ? AND id = ?",
        [kind, id_],
    ).fetchone()
    return row[0] if row else None


def test_store_records_current_embedder():
    conn = _vec_db()
    store_claim_embedding(conn, "c1", _vec())
    store_node_embedding(conn, "n1", _vec())

    assert _provenance(conn, "claim", "c1") == CURRENT_EMBEDDER
    assert _provenance(conn, "node", "n1") == CURRENT_EMBEDDER


def test_store_with_explicit_embedder():
    conn = _vec_db()
    store_claim_embedding(conn, "c1", _vec(), embedder=OLD_EMBEDDER)
    assert _provenance(conn, "claim", "c1") == OLD_EMBEDDER


def test_stale_detection_both_directions():
    conn = _vec_db()
    store_claim_embedding(conn, "current", _vec())
    store_claim_embedding(conn, "legacy", _vec(), embedder=OLD_EMBEDDER)

    # Against the current embedder, only the legacy row is stale.
    assert stale_embedding_ids(conn, "claim") == ["legacy"]
    # Against the legacy embedder, only the current row is stale.
    assert stale_embedding_ids(conn, "claim", embedder=OLD_EMBEDDER) == ["current"]


def test_stale_detection_is_kind_scoped():
    conn = _vec_db()
    store_claim_embedding(conn, "c_legacy", _vec(), embedder=OLD_EMBEDDER)
    store_node_embedding(conn, "n_current", _vec())

    assert stale_embedding_ids(conn, "claim") == ["c_legacy"]
    assert stale_embedding_ids(conn, "node") == []


def test_backfill_stamps_legacy_rows():
    conn = _vec_db()
    # A row written straight to the vector index with no provenance, as a
    # pre-provenance embedding would appear.
    conn.execute(
        "INSERT INTO vec_claims(claim_id, embedding) VALUES (?, ?)",
        ["orphan", serialise_f32(_vec())],
    )
    assert _provenance(conn, "claim", "orphan") is None

    init_vec(conn)  # runs the back-fill

    assert _provenance(conn, "claim", "orphan") == CURRENT_EMBEDDER


def test_backfill_preserves_existing_embedder():
    conn = _vec_db()
    conn.execute(
        "INSERT INTO vec_nodes(node_id, embedding) VALUES (?, ?)",
        ["n1", serialise_f32(_vec())],
    )
    conn.execute(
        "INSERT INTO embedding_provenance(kind, id, embedder, embedded_at) "
        "VALUES ('node', 'n1', ?, '2020-01-01T00:00:00+00:00')",
        [OLD_EMBEDDER],
    )

    init_vec(conn)  # must not clobber the recorded embedder

    assert _provenance(conn, "node", "n1") == OLD_EMBEDDER
    assert stale_embedding_ids(conn, "node") == ["n1"]
