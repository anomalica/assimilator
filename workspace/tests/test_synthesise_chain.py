"""The provenance chain must survive into the brief (ADR 0044).

The assembler reads ONLY the brief (ADR 0036) - it never touches the graph. So a
field that stops at the synthesiser is a field the public site cannot see, and an
anonymous assertion renders as bare fact. These tests pin the brief's half of the
contract.
"""

import sqlite3

from anomalica_common.digest.models import (
    AttestationLevel,
    Claim,
    ClaimType,
    Node,
    NodeType,
    OriginKind,
    ProvenanceChain,
    Record,
)
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.synthesise import build_entity_brief

ANONYMOUS_CHAIN = ProvenanceChain(
    origin_kind=OriginKind.anonymous,
    origin="a person claiming to work inside the Defense Intelligence Agency",
    relay=["an email", "an intermediary known to the speaker"],
)


def _brief_with_claim(**claim_kwargs) -> dict:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    subject = insert_node(conn, Node(node_type=NodeType.person, name="Subject"))
    record = insert_record(conn, Record(title="A Podcast"))
    insert_claim(
        conn,
        Claim(
            record_id=record.id,
            node_references=[subject.id],
            **claim_kwargs,
        ),
    )
    conn.commit()
    return build_entity_brief(conn, subject.id)


def test_chain_reaches_the_brief_intact():
    brief = _brief_with_claim(
        content=(
            "An anonymous source claiming to work inside the Defense Intelligence "
            "Agency said the filmed entity came from Tau Ceti."
        ),
        claim_type=ClaimType.testimony,
        provenance_chain=ANONYMOUS_CHAIN,
    )
    chain = brief["claims"][0]["provenance_chain"]

    assert chain["origin_kind"] == "anonymous"
    assert chain["origin"] == (
        "a person claiming to work inside the Defense Intelligence Agency"
    )
    assert chain["relay"] == ["an email", "an intermediary known to the speaker"]


def test_anonymous_origin_is_load_bearing():
    brief = _brief_with_claim(
        content="An anonymous source said the entity came from Tau Ceti.",
        claim_type=ClaimType.testimony,
        provenance_chain=ANONYMOUS_CHAIN,
    )
    assert brief["claims"][0]["attribution_is_load_bearing"] is True


def test_hearsay_is_load_bearing():
    brief = _brief_with_claim(
        content="He said a colleague had seen the craft.",
        claim_type=ClaimType.hearsay,
        provenance_chain=ProvenanceChain(
            origin_kind=OriginKind.named, origin="A colleague", relay=["the speaker"]
        ),
    )
    assert brief["claims"][0]["attribution_is_load_bearing"] is True


def test_third_hand_is_load_bearing():
    brief = _brief_with_claim(
        content="The programme was said to hold recovered material.",
        claim_type=ClaimType.testimony,
        attestation=AttestationLevel.third_hand,
        provenance_chain=ProvenanceChain(
            origin_kind=OriginKind.named,
            origin="An official",
            relay=["a colleague", "the speaker"],
        ),
    )
    assert brief["claims"][0]["attribution_is_load_bearing"] is True


def test_conduit_claim_is_not_load_bearing():
    """A first-hand claim the speaker originated stands on its own - the assembler
    must NOT wrap it in an attribution it does not need."""
    brief = _brief_with_claim(
        content="The Nimitz incident occurred in 2004.",
        claim_type=ClaimType.observation,
        attestation=AttestationLevel.first_hand,
        provenance_chain=ProvenanceChain(origin_kind=OriginKind.speaker),
    )
    claim = brief["claims"][0]

    assert claim["attribution_is_load_bearing"] is False
    assert claim["provenance_chain"]["origin_kind"] == "speaker"


def test_legacy_claim_has_no_chain_but_still_flags_on_attestation():
    """A pre-0044 digest carries no chain. The chain is null (never invented), and
    the predicate falls back to claim_type/attestation - weaker, but not silent."""
    brief = _brief_with_claim(
        content="The craft was recovered.",
        claim_type=ClaimType.testimony,
        attestation=AttestationLevel.second_hand,
    )
    claim = brief["claims"][0]

    assert claim["provenance_chain"] is None
    assert claim["attribution_is_load_bearing"] is True
