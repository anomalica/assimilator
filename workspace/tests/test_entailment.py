"""The digester's per-claim entailment check, stored and surfaced, not weighted."""

import sqlite3

from assimilator import synthesise
from assimilator.database import entailment_counts, init_db
from assimilator.import_markdown import _entailment_block, import_extraction
from tests.test_import_reconcile import _claim, _parsed

ENT = {
    "label": "entails",
    "score": 0.912,
    "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}


def _conn():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _label(conn, content):
    return conn.execute(
        "SELECT entailment_label, entailment_score, entailment_model FROM claims "
        "WHERE content = ?",
        (content,),
    ).fetchone()


def test_the_block_is_validated_not_trusted():
    assert _entailment_block(ENT) == ENT
    assert _entailment_block(None) is None
    assert _entailment_block({"label": "maybe", "score": 0.5, "model": "m"}) is None
    assert _entailment_block({"label": "entails", "score": 1.5, "model": "m"}) is None
    assert _entailment_block({"label": "entails", "score": True, "model": "m"}) is None
    assert _entailment_block({"label": "entails", "score": 0.5}) is None


def test_import_stores_the_block_and_counts_what_it_could_not_read():
    conn = _conn()
    with_block = _claim("c1", "Radar held the object for twelve minutes.")
    with_block["entailment"] = ENT
    bad = _claim("c2", "The object descended in under a second.")
    bad["entailment"] = {"label": "supports", "score": 0.9, "model": "m"}
    plain = _claim("c3", "Two pilots saw it.")

    counts = import_extraction(conn, _parsed([with_block, bad, plain]))

    assert _label(conn, with_block["content"]) == ("entails", 0.912, ENT["model"])
    assert _label(conn, bad["content"]) == (None, None, None)
    assert _label(conn, plain["content"]) == (None, None, None)
    assert counts["claims_assessed"] == 1
    assert counts["entailment_malformed"] == 1
    assert entailment_counts(conn) == {
        "assessed": 1,
        "unassessed": 2,
        "entails": 1,
        "neutral": 0,
        "contradicts": 0,
        "entailed_fraction": 1.0,
    }


def test_a_reimport_refreshes_the_block_on_a_carried_claim():
    """claim_hash covers the claim's wording, not its assessment, so the
    digester's backfill arrives on claims that already exist. A carried-forward
    claim must take the new block - the chain went stale this exact way."""
    conn = _conn()
    first = _claim("c1", "Radar held the object for twelve minutes.")
    import_extraction(conn, _parsed([first]))
    (claim_id,) = conn.execute("SELECT id FROM claims").fetchone()
    assert _label(conn, first["content"]) == (None, None, None)

    again = _claim("c1", "Radar held the object for twelve minutes.")
    again["entailment"] = {**ENT, "label": "contradicts", "score": 0.7}
    counts = import_extraction(conn, _parsed([again]))

    assert counts["claims_carried"] == 1 and counts["claims_created"] == 0
    assert conn.execute("SELECT id FROM claims").fetchone() == (claim_id,)
    assert _label(conn, again["content"]) == ("contradicts", 0.7, ENT["model"])


def test_the_brief_carries_the_block_per_claim_and_a_summary_per_page():
    conn = _conn()
    a = _claim("c1", "Radar held the object for twelve minutes.")
    a["entailment"] = ENT
    b = _claim("c2", "The object descended in under a second.")
    b["entailment"] = {**ENT, "label": "neutral", "score": 0.55}
    c = _claim("c3", "Two pilots saw it.")
    import_extraction(conn, _parsed([a, b, c]))
    (node_id,) = conn.execute(
        "SELECT id FROM nodes WHERE name = 'David Fravor'"
    ).fetchone()

    brief = synthesise.build_entity_brief(conn, node_id)

    by_content = {cl["content"]: cl for cl in brief["claims"]}
    assert by_content[a["content"]]["entailment"] == ENT
    assert by_content[b["content"]]["entailment"]["label"] == "neutral"
    assert "entailment" not in by_content[c["content"]]
    assert brief["entailment"] == {
        "assessed": 2,
        "unassessed": 1,
        "entails": 1,
        "neutral": 1,
        "contradicts": 0,
        "entailed_fraction": 0.5,
    }
