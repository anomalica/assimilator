import sqlite3

from assimilator.database import init_db, insert_alias, insert_node
from assimilator.matching import match_node
from anomalica_common.digest.models import Node, NodeType


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_exact_match():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    conn.commit()

    result = match_node(conn, "David Fravor", "person")
    assert result is not None
    assert result[0] == node.id
    assert result[1] == "exact"


def test_alias_match():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    insert_alias(conn, "Fravor", node.id)
    conn.commit()

    result = match_node(conn, "Fravor", "person")
    assert result is not None
    assert result[0] == node.id


def test_fuzzy_match():
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    conn.commit()

    result = match_node(conn, "David Favor", "person")  # typo
    assert result is not None
    assert result[1] == "fuzzy"


def test_no_match():
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    conn.commit()

    result = match_node(conn, "Kevin Day", "person")
    assert result is None


def test_type_filtering():
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    conn.commit()

    # Wrong type should not match
    result = match_node(conn, "David Fravor", "organisation")
    assert result is None

    # Right type matches
    result = match_node(conn, "David Fravor", "person")
    assert result is not None


def test_no_type_matches_any():
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    conn.commit()

    result = match_node(conn, "David Fravor")
    assert result is not None
    assert result[0] == node.id


def test_strip_acronym_suffix():
    from assimilator.matching import strip_acronym_suffix

    assert (
        strip_acronym_suffix("Defense Intelligence Agency (DIA)")
        == "Defense Intelligence Agency"
    )
    assert (
        strip_acronym_suffix("All-Domain Anomaly Resolution Office (AARO)")
        == "All-Domain Anomaly Resolution Office"
    )
    assert (
        strip_acronym_suffix("Strike Fighter Squadron 41 (VFA-41)")
        == "Strike Fighter Squadron 41"
    )
    # Lowercase / mixed-case parens content is NOT an acronym - left untouched.
    assert strip_acronym_suffix("Joe (mother)") == "Joe (mother)"
    assert strip_acronym_suffix("Will (AAWSAP physician)") == "Will (AAWSAP physician)"
    # No parens at all.
    assert strip_acronym_suffix("David Fravor") == "David Fravor"


def test_acronym_match_collapses_duplicate_org_nodes():
    from assimilator.matching import match_node

    conn = _db()
    # Existing canonical with the expanded-with-acronym form.
    expanded = insert_node(
        conn,
        Node(node_type=NodeType.organisation, name="Defense Intelligence Agency (DIA)"),
    )
    conn.commit()

    # Incoming bare form should match the expanded canonical.
    result = match_node(conn, "Defense Intelligence Agency", "organisation")
    assert result is not None
    assert result[0] == expanded.id
    assert result[1] == "acronym"

    # And the reverse: bare canonical, incoming expanded form.
    conn2 = _db()
    bare = insert_node(
        conn2,
        Node(node_type=NodeType.organisation, name="Defense Intelligence Agency"),
    )
    conn2.commit()
    result2 = match_node(conn2, "Defense Intelligence Agency (DIA)", "organisation")
    assert result2 is not None
    assert result2[0] == bare.id
    assert result2[1] == "acronym"


def test_acronym_match_does_not_collapse_unrelated_parens_content():
    from assimilator.matching import match_node

    conn = _db()
    # "Joe (mother)" should NOT match "Joe" via the acronym rule because
    # "mother" is not an acronym pattern.
    insert_node(conn, Node(node_type=NodeType.person, name="Joe"))
    conn.commit()

    # Either no match, or only an exact match - never an "acronym" match.
    result = match_node(conn, "Joe (mother)", "person")
    if result:
        assert result[1] != "acronym"


def test_node_name_is_unusable_redacted():
    from assimilator.import_markdown import _node_name_is_unusable

    assert (
        _node_name_is_unusable("USS Louisville Submarine Officer (redacted)")
        is not None
    )
    assert _node_name_is_unusable("3rd Fleet N2 (Redacted)") is not None
    assert _node_name_is_unusable("Salvatore Pais") is None


def test_node_name_is_unusable_type_in_parens():
    from assimilator.import_markdown import _node_name_is_unusable

    assert (
        _node_name_is_unusable("Defense Intelligence Agency (organisation)") is not None
    )
    assert (
        _node_name_is_unusable("USS Princeton Senior Master of Arms (person)")
        is not None
    )
    assert _node_name_is_unusable("AARO HR2 Volume I (document)") is not None
    # Real acronym suffix is fine.
    assert _node_name_is_unusable("Defense Intelligence Agency (DIA)") is None


def test_normalise_node_name_squadron():
    from assimilator.matching import normalise_node_name

    assert normalise_node_name("VFA-41") == "Strike Fighter Squadron 41 (VFA-41)"
    assert (
        normalise_node_name("VMFA-232")
        == "Marine Fighter Attack Squadron 232 (VMFA-232)"
    )
    assert normalise_node_name("HS-6") == "Helicopter Anti-Submarine Squadron 6 (HS-6)"
    assert (
        normalise_node_name("CSG-11 AAV MISREP November 2004")
        == "Carrier Strike Group 11 (CSG-11) AAV MISREP November 2004"
    )


def test_normalise_node_name_programme():
    from assimilator.matching import normalise_node_name

    assert (
        normalise_node_name("AATIP")
        == "Advanced Aerospace Threat Identification Program (AATIP)"
    )
    assert (
        normalise_node_name("AARO Annual Report 2024")
        == "All-Domain Anomaly Resolution Office (AARO) Annual Report 2024"
    )


def test_normalise_node_name_already_expanded():
    """When the name already contains the expanded form, leave it alone."""
    from assimilator.matching import normalise_node_name

    # Don't double-expand a name that's already in the right form.
    assert (
        normalise_node_name("Strike Fighter Squadron 41 (VFA-41)")
        == "Strike Fighter Squadron 41 (VFA-41)"
    )
    assert (
        normalise_node_name("Advanced Aerospace Threat Identification Program (AATIP)")
        == "Advanced Aerospace Threat Identification Program (AATIP)"
    )


def test_normalise_node_name_unrelated_text():
    from assimilator.matching import normalise_node_name

    # No acronym to expand - return unchanged.
    assert normalise_node_name("David Fravor") == "David Fravor"
    assert normalise_node_name("USS Princeton") == "USS Princeton"


def test_normalise_spelled_dates():
    from assimilator.import_markdown import _normalise_spelled_dates

    assert (
        _normalise_spelled_dates("FASTEAGLE Flight 14 November 2004")
        == "FASTEAGLE Flight 2004-11-14"
    )
    assert _normalise_spelled_dates("November 2004 detection") == "2004-11 detection"
    # No spelled dates, unchanged
    assert _normalise_spelled_dates("2004-11-14 intercept") == "2004-11-14 intercept"
    # Different day positions
    assert _normalise_spelled_dates("3 May 1947 sighting") == "1947-05-03 sighting"


def test_apply_doc_terminology_rejects_codenames():
    from assimilator.import_markdown import _apply_doc_terminology

    codenames = {"FASTEAGLE 01", "FASTEAGLE 02", "Tic Tac"}
    name, reason = _apply_doc_terminology("FASTEAGLE 02", codenames, {})
    assert reason is not None
    assert "codename" in reason
    name, reason = _apply_doc_terminology("Tic Tac Object", codenames, {})
    assert reason is not None
    # Non-codename names pass
    name, reason = _apply_doc_terminology("F/A-18F Super Hornet", codenames, {})
    assert reason is None


def test_apply_doc_terminology_expands_acronyms():
    from assimilator.import_markdown import _apply_doc_terminology

    expansions = {
        "AAV": "Anomalous Aerial Vehicle (AAV)",
        "WSO": "weapons systems officer (WSO)",
    }
    name, reason = _apply_doc_terminology("AAV Detection Period", set(), expansions)
    assert reason is None
    assert name == "Anomalous Aerial Vehicle (AAV) Detection Period"
    # Already expanded, leave alone
    name, _ = _apply_doc_terminology(
        "Anomalous Aerial Vehicle (AAV) Detection", set(), expansions
    )
    assert name == "Anomalous Aerial Vehicle (AAV) Detection"


def test_apply_doc_terminology_normalises_dates():
    from assimilator.import_markdown import _apply_doc_terminology

    name, _ = _apply_doc_terminology("Underwood Flight 14 November 2004", set(), {})
    assert "14 November 2004" not in name
    assert "2004-11-14" in name
