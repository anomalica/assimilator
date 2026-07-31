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


def _spread_rows(counts):
    """rows shaped like the brief query: only the last column (work) matters here."""
    rows = []
    for work, n in counts.items():
        rows += [(f"{work}-{i}", work) for i in range(n)]
    return rows


def test_cap_is_filled_across_sources_not_chronologically():
    """The live case: Jacques Vallee held 293 claims from one book and 232 from
    another, date-ordered, so the first 200 took the whole budget from the earlier
    book and the second was absent from the brief entirely - while the proposal's
    source-spread figures, computed on the full set, still read well-corroborated."""
    from assimilator.synthesise import _spread_across_sources

    rows = _spread_rows({"messengers": 293, "invisible-college": 232})
    kept = _spread_across_sources(rows, 200)

    by_work = {}
    for _cid, work in kept:
        by_work[work] = by_work.get(work, 0) + 1
    assert len(kept) == 200
    assert by_work == {"messengers": 100, "invisible-college": 100}


def test_a_small_source_is_exhausted_and_the_rest_distribute():
    from assimilator.synthesise import _spread_across_sources

    rows = _spread_rows({"big": 500, "small": 3})
    kept = _spread_across_sources(rows, 100)
    counts = {}
    for _cid, work in kept:
        counts[work] = counts.get(work, 0) + 1
    assert counts == {"big": 97, "small": 3}


def test_selection_keeps_document_order_and_is_deterministic():
    """brief_hash is computed over this sequence, so the order must be stable and
    must still read as the document does."""
    from assimilator.synthesise import _spread_across_sources

    rows = _spread_rows({"a": 5, "b": 5})
    kept = _spread_across_sources(rows, 6)
    assert kept == _spread_across_sources(rows, 6)
    assert [rows.index(r) for r in kept] == sorted(rows.index(r) for r in kept)


def test_under_the_cap_nothing_is_reordered_or_dropped():
    from assimilator.synthesise import _spread_across_sources

    rows = _spread_rows({"a": 4, "b": 2})
    assert _spread_across_sources(rows, 200) == rows


def _slug_graph():
    import sqlite3

    from anomalica_common.digest.models import Node
    from assimilator.database import init_db, insert_node

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    # Same name, different types: /organisations/x and /projects/x are distinct
    # URLs, so neither needs a suffix.
    insert_node(conn, Node(id="aaa-org", node_type="organisation", name="AARO"))
    insert_node(conn, Node(id="bbb-proj", node_type="project", name="AARO"))
    # Same name, SAME type: a genuine collision in one section.
    insert_node(conn, Node(id="aaa-klas", node_type="organisation", name="KLAS TV"))
    insert_node(conn, Node(id="bbb-klas", node_type="organisation", name="KLAS TV"))
    conn.commit()
    return conn


def test_cross_type_same_name_keeps_a_clean_slug():
    """A published URL is /<section>/<slug> and the section follows the type, so
    an organisation and a project of one name never clash. Suffixing them put an
    arbitrary hex fragment into a real path to separate pages that could not
    collide."""
    from assimilator.synthesise import build_slug_map

    slug_map, _ = build_slug_map(_slug_graph())
    assert slug_map["aaa-org"] == "aaro"
    assert slug_map["bbb-proj"] == "aaro"


def test_same_type_same_name_is_still_disambiguated():
    from assimilator.synthesise import build_slug_map

    slug_map, collisions = build_slug_map(_slug_graph())
    klas = {slug_map["aaa-klas"], slug_map["bbb-klas"]}
    assert len(klas) == 2 and "klas-tv" in klas
    # Both kinds of collision are still REPORTED - a cross-type pair is usually a
    # taxonomy split worth seeing, even though it needs no suffix.
    assert {c["slug"] for c in collisions} == {"aaro", "klas-tv"}


def test_event_keeps_the_sources_that_are_about_it_not_the_biggest():
    """Ranking capped sources by claim count picks long books over short primary
    accounts: on the Nimitz encounter it selected five books and dropped the
    CSG-11 incident report, the document the event happened in. Focus - the share
    of a record that concerns the node - inverts that correctly."""
    from assimilator.synthesise import _spread_across_sources

    rows = []
    for work, n in {"book": 120, "report": 30, "statement": 25, "podcast": 100}.items():
        rows += [(f"{work}-{i}", work) for i in range(n)]
    focus = {"book": 0.02, "report": 0.25, "statement": 0.63, "podcast": 0.24}

    kept = _spread_across_sources(rows, 200, max_sources=3, focus=focus)
    works = {w for _cid, w in kept}
    assert works == {"statement", "report", "podcast"}
    assert "book" not in works


def test_a_tiny_record_cannot_win_on_focus_alone():
    """Two claims in a two-claim record is 100% focus and no use as an account."""
    from assimilator.synthesise import _spread_across_sources

    rows = [("t-0", "tiny"), ("t-1", "tiny")]
    rows += [(f"real-{i}", "real") for i in range(40)]
    focus = {"tiny": 1.0, "real": 0.3}

    kept = _spread_across_sources(rows, 200, max_sources=1, focus=focus)
    assert {w for _cid, w in kept} == {"real"}


def test_a_person_is_not_source_capped():
    """Breadth across sources is the point for a person - Vallee is the case."""
    from assimilator.synthesise import MAX_SOURCES

    assert MAX_SOURCES.get("person") is None
    assert MAX_SOURCES.get("event") == 5
