import sqlite3

from assimilator.database import (
    find_node_by_name,
    get_claims_for_node,
    get_corroborations,
    get_independent_source_count,
    get_stats,
    init_db,
    insert_alias,
    insert_claim,
    insert_corroboration,
    insert_node,
    insert_record,
)
from anomalica_common.digest.models import (
    AttestationLevel,
    Claim,
    ClaimRole,
    ClaimType,
    Node,
    NodeType,
    Record,
)


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_insert_and_get_node():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    assert node.id
    assert node.created_at

    found = find_node_by_name(conn, "Alice", "person")
    assert found is not None
    assert found.name == "Alice"


def test_alias_lookup():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    insert_alias(conn, "Fravor", node.id)
    insert_alias(conn, "CDR Fravor", node.id)

    assert find_node_by_name(conn, "Fravor", "person").id == node.id
    assert find_node_by_name(conn, "CDR Fravor", "person").id == node.id
    assert find_node_by_name(conn, "Unknown Person", "person") is None


def test_insert_record_and_claim():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    record = insert_record(conn, Record(title="Test Record"))
    insert_claim(
        conn,
        Claim(
            content="Alice saw something.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=record.id,
            speaker_id=node.id,
            node_references=[node.id],
        ),
    )
    conn.commit()

    claims = get_claims_for_node(conn, node.id)
    assert len(claims) == 1
    assert claims[0].content == "Alice saw something."
    assert claims[0].node_references == [node.id]


def test_corroboration_and_independent_sources():
    conn = _db()
    alice = insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    bob = insert_node(conn, Node(node_type=NodeType.person, name="Bob"))
    rec1 = insert_record(conn, Record(title="Record 1"))
    rec2 = insert_record(conn, Record(title="Record 2"))

    # Same claim from different speakers in different records
    c1 = insert_claim(
        conn,
        Claim(
            content="The sky is blue.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec1.id,
            speaker_id=alice.id,
        ),
    )
    c2 = insert_claim(
        conn,
        Claim(
            content="The sky is blue.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec2.id,
            speaker_id=bob.id,
        ),
    )
    conn.commit()

    insert_corroboration(conn, c1.id, c2.id, 0.99)
    conn.commit()

    corrs = get_corroborations(conn, c1.id)
    assert len(corrs) == 1

    # Different speakers = 2 independent sources
    assert get_independent_source_count(conn, c1.id) == 2


def test_same_speaker_not_independent():
    conn = _db()
    alice = insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    rec1 = insert_record(conn, Record(title="Record 1"))
    rec2 = insert_record(conn, Record(title="Record 2"))

    # Same claim, same speaker, different records
    c1 = insert_claim(
        conn,
        Claim(
            content="I saw it.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec1.id,
            speaker_id=alice.id,
        ),
    )
    c2 = insert_claim(
        conn,
        Claim(
            content="I saw it.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec2.id,
            speaker_id=alice.id,
        ),
    )
    conn.commit()

    insert_corroboration(conn, c1.id, c2.id, 0.99)
    conn.commit()

    # Same speaker = 1 independent source despite 2 records
    assert get_independent_source_count(conn, c1.id) == 1


def test_stats():
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    insert_node(conn, Node(node_type=NodeType.organisation, name="ACME"))
    rec = insert_record(conn, Record(title="Test"))
    insert_claim(
        conn,
        Claim(
            content="Test claim.",
            claim_type=ClaimType.administrative,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
        ),
    )
    conn.commit()

    s = get_stats(conn)
    assert s["nodes"] == 2
    assert s["active_nodes"] == 2
    assert s["records"] == 1
    assert s["claims"] == 1
    assert s["by_type"]["person"] == 1
    assert s["by_type"]["organisation"] == 1


def test_claim_role_roundtrip():
    """ADR 0028: claim_role is a top-level optional column."""
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="Witness"))
    rec = insert_record(conn, Record(title="Test"))
    insert_claim(
        conn,
        Claim(
            content="Saw something at altitude.",
            claim_type=ClaimType.observation,
            claim_role=ClaimRole.witness_testimony,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
            speaker_id=node.id,
            node_references=[node.id],
        ),
    )
    insert_claim(
        conn,
        Claim(
            content="Background context with no narrative role.",
            claim_type=ClaimType.administrative,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
        ),
    )
    conn.commit()

    sorted(
        get_claims_for_node(conn, node.id)
        + [c for c in get_claims_for_node(conn, node.id) if c.claim_role is None],
        key=lambda c: c.content,
    )
    # Re-fetch through record path to catch both
    from assimilator.database import get_claims_for_record

    all_claims = get_claims_for_record(conn, rec.id)
    roles = {c.content: c.claim_role for c in all_claims}
    assert roles["Saw something at altitude."] == ClaimRole.witness_testimony
    assert roles["Background context with no narrative role."] is None


def test_claim_role_check_constraint():
    """Invalid claim_role values are rejected at the DB layer."""
    conn = _db()
    rec = insert_record(conn, Record(title="Test"))
    try:
        conn.execute(
            "INSERT INTO claims (id, content, claim_type, attestation, "
            "record_id, confidence, created_at, claim_role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "bad-id",
                "x",
                "observation",
                "first_hand",
                rec.id,
                1.0,
                "2026-05-24T00:00:00+00:00",
                "not_a_valid_role",
            ),
        )
        raise AssertionError("CHECK constraint did not fire")
    except sqlite3.IntegrityError:
        pass


def test_claim_role_migration_on_legacy_db():
    """Existing pre-0028 databases (no claim_role column) upgrade cleanly."""
    conn = sqlite3.connect(":memory:")
    # Hand-build the pre-0028 schema (no claim_role column, no role index).
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            retired_at TEXT
        );
        CREATE TABLE records (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            reference TEXT,
            date TEXT,
            producer_id TEXT,
            content_hash TEXT,
            friendly_name TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE claims (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            original_excerpt TEXT,
            claim_type TEXT NOT NULL,
            attestation TEXT NOT NULL,
            record_id TEXT NOT NULL REFERENCES records(id),
            speaker_id TEXT,
            location_in_record TEXT,
            date TEXT,
            date_end TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO records (id, title, created_at)
            VALUES ('r1', 'Legacy record', '2025-01-01T00:00:00+00:00');
        INSERT INTO claims (id, content, claim_type, attestation,
            record_id, confidence, created_at)
            VALUES ('c1', 'legacy claim', 'observation', 'first_hand',
                    'r1', 1.0, '2025-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    cols_before = [r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()]
    assert "claim_role" not in cols_before

    init_db(conn)
    conn.commit()

    cols_after = [r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()]
    assert "claim_role" in cols_after
    # Existing claim survives unchanged with a NULL role
    row = conn.execute(
        "SELECT content, claim_role FROM claims WHERE id = 'c1'"
    ).fetchone()
    assert row == ("legacy claim", None)

    # Idempotent: re-running init_db is a no-op
    init_db(conn)
    init_db(conn)
    cols_final = [r[1] for r in conn.execute("PRAGMA table_info(claims)").fetchall()]
    assert cols_final == cols_after
