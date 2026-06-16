import sqlite3

from assimilator.database import (
    init_db,
    insert_claim,
    insert_corroboration,
    insert_node,
    insert_record,
)
from anomalica_common.digest.models import (
    AttestationLevel,
    Claim,
    ClaimType,
    Node,
    NodeType,
    Record,
)
from assimilator.scoring import score_claim, tier_label


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_base_score_observation_first_hand():
    conn = _db()
    rec = insert_record(conn, Record(title="Test"))
    c = insert_claim(
        conn,
        Claim(
            content="I saw it.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
        ),
    )
    conn.commit()

    s = score_claim(conn, c.id)
    assert s.score == 0.75  # observation (0.75) * first_hand (1.0)
    assert s.record_count == 1


def test_base_score_opinion_third_hand():
    conn = _db()
    rec = insert_record(conn, Record(title="Test"))
    c = insert_claim(
        conn,
        Claim(
            content="I heard someone thinks so.",
            claim_type=ClaimType.opinion,
            attestation=AttestationLevel.third_hand,
            record_id=rec.id,
        ),
    )
    conn.commit()

    s = score_claim(conn, c.id)
    expected = 0.3 * 0.3  # opinion * third_hand
    assert abs(s.score - expected) < 0.001


def test_corroboration_increases_score():
    conn = _db()
    alice = insert_node(conn, Node(node_type=NodeType.person, name="Alice"))
    bob = insert_node(conn, Node(node_type=NodeType.person, name="Bob"))
    rec1 = insert_record(conn, Record(title="R1"))
    rec2 = insert_record(conn, Record(title="R2"))

    c1 = insert_claim(
        conn,
        Claim(
            content="The object was there.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec1.id,
            speaker_id=alice.id,
        ),
    )
    c2 = insert_claim(
        conn,
        Claim(
            content="The object was there.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec2.id,
            speaker_id=bob.id,
        ),
    )
    conn.commit()

    single = score_claim(conn, c1.id)

    insert_corroboration(conn, c1.id, c2.id, 0.99)
    conn.commit()

    corroborated = score_claim(conn, c1.id)
    assert corroborated.score > single.score
    assert corroborated.record_count == 2


def test_tier_labels():
    assert tier_label(0.90) == "strong"
    assert tier_label(0.70) == "moderate"
    assert tier_label(0.45) == "weak"
    assert tier_label(0.10) == "insufficient"
