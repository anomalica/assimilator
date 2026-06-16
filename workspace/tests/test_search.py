import sqlite3

from assimilator.database import init_db, insert_claim, insert_record
from anomalica_common.digest.models import AttestationLevel, Claim, ClaimType, Record
from assimilator.search import (
    _sigmoid,
    _tokenise_query,
    keyword_search_claims,
    rrf_merge,
)


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _claim(conn, record_id: str, content: str) -> str:
    c = insert_claim(
        conn,
        Claim(
            content=content,
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=record_id,
        ),
    )
    return c.id


def test_tokenise_filters_stop_words():
    tokens = _tokenise_query("What was the flight to Osaka")
    assert "flight" in tokens
    assert "osaka" in tokens
    assert "the" not in tokens
    assert "was" not in tokens


def test_tokenise_keeps_query_when_all_stopped():
    tokens = _tokenise_query("the is at")
    assert tokens, "fallback should return the original tokens"


def test_keyword_search_finds_matches():
    conn = _db()
    rec = insert_record(conn, Record(title="Health"))
    target = _claim(conn, rec.id, "BMI was measured at 27.4")
    _claim(conn, rec.id, "Took a flight from Tokyo to Osaka")
    conn.commit()

    results = keyword_search_claims(conn, "BMI", limit=10)
    ids = [cid for cid, _ in results]
    assert target in ids


def test_keyword_search_idf_ranks_rare_higher():
    conn = _db()
    rec = insert_record(conn, Record(title="Mixed"))
    rare = _claim(conn, rec.id, "Lipid panel showed elevated cholesterol")
    # Add many claims with the common word "flight" to push its IDF down
    for i in range(20):
        _claim(conn, rec.id, f"Flight number {i} departed on time")
    conn.commit()

    results = keyword_search_claims(conn, "flight cholesterol", limit=5)
    # The cholesterol-bearing claim should rank above any flight-only claim
    # because cholesterol is the rare term.
    assert results
    top_id, _ = results[0]
    assert top_id == rare


def test_keyword_search_empty_query():
    conn = _db()
    rec = insert_record(conn, Record(title="Mixed"))
    _claim(conn, rec.id, "Anything here")
    conn.commit()
    assert keyword_search_claims(conn, "", limit=10) == []
    assert keyword_search_claims(conn, "   ", limit=10) == []


def test_rrf_merge_combines_lists():
    list_a = [("x", 0.9), ("y", 0.8), ("z", 0.7)]
    list_b = [("y", 0.95), ("w", 0.5)]
    merged = rrf_merge(list_a, list_b)
    ids = [item_id for item_id, _ in merged]
    assert ids[0] == "y"  # appears in both, near top of each
    assert set(ids) == {"x", "y", "z", "w"}


def test_rrf_merge_empty_lists():
    assert rrf_merge() == []
    assert rrf_merge([], []) == []


def test_sigmoid_bounds():
    assert _sigmoid(0.0) == 0.5
    assert 0.0 < _sigmoid(-50.0) < 0.01
    assert 0.99 < _sigmoid(50.0) <= 1.0
    # Extreme positives clamp to 1.0 via OverflowError branch
    assert _sigmoid(1e6) == 1.0
    assert _sigmoid(-1e6) == 0.0
