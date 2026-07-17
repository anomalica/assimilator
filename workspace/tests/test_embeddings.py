import sqlite3

import pytest

from assimilator.embeddings import (
    DECODE_REVISION,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL_ID,
    PRE_TRACKING_MODEL_ID,
    QUANT_MAX,
    QUANT_MIN,
    QUANT_ZERO_POINT,
    _dequantise,
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


def test_backfill_marks_untracked_rows_stale_not_current():
    """An untracked embedding predates this table, so it predates the decode fix
    and is a raw-uint8 vector from the degenerate space. Stamping it current would
    bless it permanently - `stale_embedding_ids` would never return it again."""
    conn = _vec_db()
    conn.execute(
        "INSERT INTO vec_claims(claim_id, embedding) VALUES (?, ?)",
        ["orphan", serialise_f32(_vec())],
    )
    assert _model_id(conn, "claim", "orphan") is None

    init_vec(conn)  # runs the back-fill

    assert _model_id(conn, "claim", "orphan") == PRE_TRACKING_MODEL_ID
    assert "orphan" in stale_embedding_ids(conn, "claim")


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


# --- decode contract --------------------------------------------------------
#
# This ONNX emits uint8, not floats. Reading it raw yields a space where every
# pair of texts scores cosine ~0.99 whatever they mean, because the shared
# zero-point constant dominates. Nothing raises; it just returns confident
# nonsense. These tests pin the decode so that failure cannot return silently.


def test_zero_point_sits_where_the_calibration_range_puts_it():
    """~110: the uint8 value meaning float 0.0. Every raw vector clusters around
    it, which is exactly why raw cosine is meaningless."""
    assert QUANT_MIN < 0 < QUANT_MAX
    assert _dequantise([QUANT_ZERO_POINT])[0] == pytest.approx(0.0, abs=1e-9)


def test_dequantise_maps_the_uint8_range_onto_the_calibrated_floats():
    assert _dequantise([0])[0] == pytest.approx(QUANT_MIN)
    assert _dequantise([255])[0] == pytest.approx(QUANT_MAX)


def test_dequantise_recentres_around_zero():
    """The point of the decode: raw values are all positive (0..255), so raw
    vectors share a large constant component. Dequantised, values straddle zero
    and cosine can separate them."""
    below, above = _dequantise([QUANT_ZERO_POINT - 30, QUANT_ZERO_POINT + 30])
    assert below < 0 < above


def test_raw_vectors_are_degenerate_but_dequantised_ones_separate():
    """The bug, reproduced from the model's own quantisation scheme without
    loading it: two DIFFERENT vectors are ~identical under cosine while raw, and
    distinguishable once decoded."""
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
        return dot / (na * nb)

    z = QUANT_ZERO_POINT
    raw_a = [z + 20, z - 20, z + 10, z - 10]
    raw_b = [z - 20, z + 20, z - 10, z + 10]  # the opposite direction

    # Raw, two exact opposites score ~0.96 - the zero-point constant swamps the
    # signal. Decoded, they are correctly antipodal.
    assert cos(raw_a, raw_b) > 0.95
    assert cos(_dequantise(raw_a), _dequantise(raw_b)) == pytest.approx(-1.0)


def test_model_id_carries_the_decode_revision():
    """The decode convention defines the space as much as the model file does: the
    same ONNX read raw and read dequantised are not comparable, so a stored vector
    must name which one produced it."""
    assert DECODE_REVISION in EMBEDDING_MODEL_ID


def test_pre_tracking_marker_is_never_the_current_space():
    assert PRE_TRACKING_MODEL_ID != EMBEDDING_MODEL_ID
