"""The provenance chain must survive into the brief, and the brief must tell a
consumer how to render the claim's attribution (ADR 0044).

The assembler reads ONLY the brief (ADR 0036) - it never touches the graph. So a
field that stops at the synthesiser is a field the public site cannot see, and an
anonymous assertion renders as bare fact.

The RULE itself lives in anomalica_common and is tested there. These tests pin the
BRIEF's half of the contract: that the chain round-trips, and that the right
arguments reach the shared function - a wiring bug would silently downgrade every
claim to a mode it did not earn.
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
        Claim(record_id=record.id, node_references=[subject.id], **claim_kwargs),
    )
    conn.commit()
    return build_entity_brief(conn, subject.id)["claims"][0]


def test_chain_reaches_the_brief_intact():
    claim = _brief_with_claim(
        content=(
            "An anonymous source claiming to work inside the Defense Intelligence "
            "Agency said the filmed entity came from Tau Ceti."
        ),
        claim_type=ClaimType.testimony,
        provenance_chain=ANONYMOUS_CHAIN,
    )
    chain = claim["provenance_chain"]

    assert chain["origin_kind"] == "anonymous"
    assert chain["origin"] == (
        "a person claiming to work inside the Defense Intelligence Agency"
    )
    assert chain["relay"] == ["an email", "an intermediary known to the speaker"]


def test_anonymous_claim_is_never_bare():
    """THE TAU CETI CASE. An anonymous origin's truth rests entirely on who asserted
    it, so it must never be asserted plainly. Until extraction DECLARES that it put
    the attribution in the text (attribution_in_text, not yet carried on the claim),
    the safe answer is unknown - hedge or drop, never assert."""
    claim = _brief_with_claim(
        content="The being was a cloned entity from the Tau Ceti star system.",
        claim_type=ClaimType.testimony,
        provenance_chain=ANONYMOUS_CHAIN,
    )
    assert claim["attribution_mode"] == "unknown"
    assert claim["attribution_mode"] != "bare_ok"


def test_conduit_claim_is_bare_ok():
    """A first-hand claim the speaker originated stands on its own - the assembler
    must NOT wrap it in an attribution it does not need."""
    claim = _brief_with_claim(
        content="The Nimitz incident occurred in 2004.",
        claim_type=ClaimType.observation,
        attestation=AttestationLevel.first_hand,
        provenance_chain=ProvenanceChain(origin_kind=OriginKind.speaker),
    )
    assert claim["attribution_mode"] == "bare_ok"


def test_unattributed_is_bare_ok():
    """`unattributed` is a POSITIVE statement that the source offers no attribution
    (ordinary narration), not an absence of knowledge. Hedging it would turn "the
    Nimitz incident occurred in 2004" into absurdity."""
    claim = _brief_with_claim(
        content="The Nimitz incident occurred in 2004.",
        claim_type=ClaimType.observation,
        provenance_chain=ProvenanceChain(origin_kind=OriginKind.unattributed),
    )
    assert claim["attribution_mode"] == "bare_ok"


def test_second_hand_claim_is_not_bare():
    claim = _brief_with_claim(
        content="The programme held recovered material.",
        claim_type=ClaimType.testimony,
        attestation=AttestationLevel.second_hand,
        provenance_chain=ProvenanceChain(
            origin_kind=OriginKind.named, origin="An official", relay=["the speaker"]
        ),
    )
    assert claim["attribution_mode"] == "unknown"


def test_legacy_claim_is_unknown_not_bare():
    """THE FAIL-OPEN CASE that shipped and was caught. A pre-0044 claim: no chain,
    no attestation, type testimony. The first version of this contract called it
    "not load-bearing" and the assembler would have rendered it as bare fact. The
    text is bare and unvouched: it must be hedged or dropped, never asserted."""
    claim = _brief_with_claim(
        content="The craft was recovered.",
        claim_type=ClaimType.testimony,
    )
    assert claim["provenance_chain"] is None
    assert claim["attribution_mode"] == "unknown"


def test_legacy_hearsay_is_unknown():
    """A legacy hearsay claim's TEXT is bare - pre-0044 extraction stripped the
    reporting verb - so its type alone cannot license rendering it as-is."""
    claim = _brief_with_claim(
        content="A colleague had seen the craft.",
        claim_type=ClaimType.hearsay,
    )
    assert claim["attribution_mode"] == "unknown"
