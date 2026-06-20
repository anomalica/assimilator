"""Re-import reconciliation: claim_hash carry-forward, no duplicate rows.

Guards the fix for two bugs in the importer's existing-record path: re-importing
a record used to (a) accumulate duplicate claim rows because nothing deleted the
old ones, and (b) churn every claim's uuid because the digester mints a fresh one
per emission, making an unchanged re-digest read as 100% changed.
"""

from __future__ import annotations

import sqlite3

from assimilator.database import init_db
from assimilator.import_markdown import backfill_claim_hashes, import_extraction


def _parsed(claims):
    return {
        "frontmatter": {
            "record_id": "rec-nimitz-0001",
            "record_title": "Nimitz Encounter Briefing",
            "record_date": "2004-11-14",
            "content_hash": "sha256:" + "a" * 64,
            "friendly_name": "nimitz-briefing",
        },
        "nodes": [
            {"id": "n-fravor", "name": "David Fravor", "node_type": "person"},
            {"id": "n-nimitz", "name": "USS Nimitz", "node_type": "object"},
        ],
        "domain_claims": claims,
        "infrastructure_claims": [],
        "terminology": None,
    }


def _claim(cid, content, refs=("David Fravor",), speaker="David Fravor"):
    return {
        "id": cid,
        "content": content,
        "claim_type": "testimony",
        "attestation": "first_hand",
        "speaker": speaker,
        "node_references": list(refs),
    }


def _conn():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _claim_count(conn):
    return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]


def _claim_ids(conn):
    return {r[0] for r in conn.execute("SELECT id FROM claims").fetchall()}


def test_first_import_sets_claim_hash_on_every_claim():
    conn = _conn()
    counts = import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))
    assert counts["claims_created"] == 1
    assert counts["claims_carried"] == 0
    null_hashes = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE claim_hash IS NULL"
    ).fetchone()[0]
    assert null_hashes == 0


def test_identical_reimport_carries_forward_and_does_not_duplicate():
    conn = _conn()
    claims = [_claim("c1", "Held radar 12 min."), _claim("c2", "Tic Tac accelerated.")]
    import_extraction(conn, _parsed(claims))
    ids_before = _claim_ids(conn)
    assert _claim_count(conn) == 2

    # Re-import the SAME content but with fresh uuids (as the digester would).
    reimport = [
        _claim("x1", "Held radar 12 min."),
        _claim("x2", "Tic Tac accelerated."),
    ]
    counts = import_extraction(conn, _parsed(reimport))

    assert _claim_count(conn) == 2  # no duplication - the core bug fix
    assert counts["claims_carried"] == 2
    assert counts["claims_created"] == 0
    assert counts["claims_deleted"] == 0
    assert _claim_ids(conn) == ids_before  # original uuids carried forward


def test_changed_claim_replaces_only_itself():
    conn = _conn()
    import_extraction(
        conn, _parsed([_claim("c1", "Held radar 12 min."), _claim("c2", "Unchanged.")])
    )
    c2_id = conn.execute(
        "SELECT id FROM claims WHERE content = 'Unchanged.'"
    ).fetchone()[0]

    counts = import_extraction(
        conn,
        _parsed([_claim("x1", "Held radar 14 min."), _claim("x2", "Unchanged.")]),
    )
    assert _claim_count(conn) == 2
    assert counts["claims_created"] == 1  # the edited claim
    assert counts["claims_deleted"] == 1  # its prior version
    assert counts["claims_carried"] == 1  # the unchanged one
    # The unchanged claim kept its row identity.
    assert (
        conn.execute("SELECT id FROM claims WHERE content = 'Unchanged.'").fetchone()[0]
        == c2_id
    )


def test_removed_claim_is_deleted():
    conn = _conn()
    import_extraction(
        conn, _parsed([_claim("c1", "Held radar 12 min."), _claim("c2", "Second.")])
    )
    counts = import_extraction(conn, _parsed([_claim("x1", "Held radar 12 min.")]))
    assert _claim_count(conn) == 1
    assert counts["claims_deleted"] == 1
    assert counts["claims_carried"] == 1


def test_added_claim_is_inserted():
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "First.")]))
    counts = import_extraction(
        conn, _parsed([_claim("x1", "First."), _claim("x2", "New one.")])
    )
    assert _claim_count(conn) == 2
    assert counts["claims_created"] == 1
    assert counts["claims_carried"] == 1


def test_backfill_reproduces_the_import_hash():
    # A claim hashed at import and the same claim hashed by the backfill must
    # agree - same canonicalisation on both paths.
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))
    at_import = conn.execute("SELECT claim_hash FROM claims").fetchone()[0]

    conn.execute("UPDATE claims SET claim_hash = NULL")
    backfill_claim_hashes(conn)
    after_backfill = conn.execute("SELECT claim_hash FROM claims").fetchone()[0]
    assert after_backfill == at_import
