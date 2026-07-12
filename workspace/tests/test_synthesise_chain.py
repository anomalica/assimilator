"""The provenance chain must survive into the brief, and the brief must tell a
consumer how to render the claim's attribution (ADR 0044).

The assembler reads ONLY the brief (ADR 0036) - it never touches the graph. So a
field that stops at the synthesiser is a field the public site cannot see, and an
anonymous assertion renders as bare fact.

The rendering contract is a TRI-STATE and fails CLOSED. An earlier boolean keyed on
the presence of a danger signal, so a claim with no chain and no attestation matched
nothing and fell through to "safe to assert bare" - fail-open, on exactly the case
0044 exists to close. A claim must EARN bare_ok; it is never granted it by default.
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
from assimilator.synthesise import attribution_mode, build_entity_brief

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
    assert claim["attribution_mode"] == "in_text"


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


def test_legacy_claim_is_unknown_not_bare():
    """THE FAIL-OPEN CASE. A pre-0044 claim: no chain, no attestation, type
    testimony. The old boolean called this "not load-bearing" and the assembler
    would have rendered it as a bare fact. It is unknown - the text is bare and
    unvouched, so it must be hedged or dropped, never asserted."""
    claim = _brief_with_claim(
        content="The craft was recovered.",
        claim_type=ClaimType.testimony,
    )
    assert claim["provenance_chain"] is None
    assert claim["attribution_mode"] == "unknown"


def test_legacy_hearsay_is_unknown_not_in_text():
    """A legacy hearsay claim's TEXT is bare - pre-0044 extraction stripped the
    reporting verb - so it must not be rendered as-is on the strength of its type.
    in_text requires a captured chain."""
    claim = _brief_with_claim(
        content="A colleague had seen the craft.",
        claim_type=ClaimType.hearsay,
    )
    assert claim["attribution_mode"] == "unknown"


def test_unattributed_is_bare_ok_not_unknown():
    """`unattributed` is a POSITIVE statement that the source offers no attribution
    (ordinary narration), not an absence of knowledge. Hedging it would turn "the
    Nimitz incident occurred in 2004" into absurdity."""
    claim = _brief_with_claim(
        content="The Nimitz incident occurred in 2004.",
        claim_type=ClaimType.observation,
        provenance_chain=ProvenanceChain(origin_kind=OriginKind.unattributed),
    )
    assert claim["attribution_mode"] == "bare_ok"


def test_opinion_is_in_text():
    """An opinion is not a fact about the world - the fact is that someone HOLDS it,
    and extraction already names the holder in the text."""
    claim = _brief_with_claim(
        content="Jon Stewart considers the account consistent with a PSYOP.",
        claim_type=ClaimType.opinion,
        provenance_chain=ProvenanceChain(origin_kind=OriginKind.speaker),
    )
    assert claim["attribution_mode"] == "in_text"


# The predicate itself, exhaustively - the brief tests above prove it reaches the
# brief; these prove it is right.


def test_second_and_third_hand_are_in_text():
    for level in ("second_hand", "third_hand"):
        assert attribution_mode("named", "testimony", level) == "in_text"


def test_anonymous_is_in_text_even_without_attestation():
    assert attribution_mode("anonymous", "testimony", None) == "in_text"


def test_chain_present_but_attestation_missing_is_unknown():
    """A captured chain does not on its own license a bare assertion. If the chain
    is there but the attestation is missing, we cannot grade the removes - fail
    closed."""
    assert attribution_mode("named", "testimony", None) == "unknown"
    assert attribution_mode("speaker", "observation", None) == "unknown"
    assert attribution_mode("document", "testimony", None) == "unknown"


def test_no_chain_is_always_unknown():
    """Whatever else is true, an uncaptured chain can never earn bare_ok."""
    assert attribution_mode(None, "observation", "first_hand") == "unknown"
    assert attribution_mode(None, "testimony", None) == "unknown"
    assert attribution_mode("", "observation", "first_hand") == "unknown"
