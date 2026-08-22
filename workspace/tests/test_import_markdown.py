def test_a_document_glossary_never_rewrites_a_person_name():
    """The glossary says what a term means in the document's prose. A person's
    name is not a term: "UAP Gerb" is a real researcher's handle, and expanding
    it produced a page-worthy node with 36 claims that read as two merged
    people."""
    from assimilator.import_markdown import _apply_doc_terminology

    glossary = {"UAP": "Unidentified Aerial Phenomena (UAP)"}

    person, reason = _apply_doc_terminology("UAP Gerb", set(), glossary, "person")
    assert person == "UAP Gerb"
    assert reason is None

    # The same acronym in an organisation name still expands.
    org, _ = _apply_doc_terminology("UAP Task Force", set(), glossary, "organisation")
    assert org == "Unidentified Aerial Phenomena (UAP) Task Force"


def test_a_person_is_still_rejected_for_carrying_a_codename():
    """The exemption covers the glossary, not the codename gate - a codename may
    never be a node's canonical identifier, whatever its type."""
    from assimilator.import_markdown import _apply_doc_terminology

    _, reason = _apply_doc_terminology("Kona Blue Dave", {"Kona Blue"}, {}, "person")
    assert reason == "contains codename 'Kona Blue'"


def _described_parsed():
    """One described actor and one real person with a qualifier, in one record."""
    return {
        "frontmatter": {
            "record_id": "rec-described-0001",
            "record_title": "An Interview",
            "content_hash": "sha256:" + "b" * 64,
            "friendly_name": "an-interview",
        },
        "nodes": [
            {
                "id": "n-officer",
                "name": "[senior US intelligence officer]",
                "node_type": "person",
                "type": "person",
            },
            {
                "id": "n-sally",
                "name": "Sally (Budd Hopkins abductee)",
                "node_type": "person",
                "type": "person",
            },
        ],
        "domain_claims": [
            {
                "id": "c1",
                "content": "The programme existed under a different name.",
                "claim_type": "testimony",
                "speaker": "[senior US intelligence officer]",
                "node_references": ["[senior US intelligence officer]"],
            },
            {
                "id": "c2",
                "content": "Sally recalled the room.",
                "claim_type": "testimony",
                "speaker": "Sally (Budd Hopkins abductee)",
                "node_references": ["Sally (Budd Hopkins abductee)"],
            },
        ],
        "infrastructure_claims": [],
        "terminology": None,
    }


def test_a_described_actor_never_becomes_a_node():
    """Brackets are record-scoped: the `[interviewer 2]` in one recording is not
    the one in another, so a node built from one would accumulate two unrelated
    people's biography under a single identity."""
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    counts = import_extraction(conn, _described_parsed())

    names = {r[0] for r in conn.execute("SELECT name FROM nodes")}
    assert "[senior US intelligence officer]" not in names
    assert "Sally (Budd Hopkins abductee)" in names, (
        "Name (description) is a name - around twenty real people are written "
        "that way, and the qualifier is what tells two Sallys apart"
    )
    assert counts["nodes_described"] == 1


def test_a_described_speaker_keeps_its_claim_and_its_attribution():
    """The identity does not exist; the testimony does. Dropping the attribution
    would lose who said it, and that is not recoverable from the claim text."""
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, _described_parsed())

    row = conn.execute(
        "SELECT speaker_id, origin_kind, origin FROM claims WHERE content LIKE "
        "'The programme%'"
    ).fetchone()
    assert row is not None, "the claim must survive its speaker having no node"
    speaker_id, origin_kind, origin = row
    assert speaker_id is None, "there is no node to point at"
    # Anonymous is the shape the corpus already uses for an unnamed source, and
    # independence collapses every anonymous origin to ONE root (ADR 0039): one
    # anonymous officer is one source, not one per claim.
    assert origin_kind == "anonymous"
    assert origin == "[senior US intelligence officer]"


def test_the_described_actor_takes_no_claim_refs_with_it():
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, _described_parsed())

    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 2
    refs = conn.execute(
        "SELECT COUNT(*) FROM claim_node_refs r JOIN claims c ON c.id = r.claim_id "
        "WHERE c.content LIKE 'The programme%'"
    ).fetchone()[0]
    assert refs == 0, "a description has no node, so the ref resolves to nothing"


def test_is_a_description_requires_the_whole_value_to_be_bracketed():
    from assimilator.import_markdown import is_a_description

    assert is_a_description("[senior US intelligence officer]")
    assert is_a_description("[redacted]")
    assert is_a_description("  [speaker 1]  ")
    assert not is_a_description("Sally (Budd Hopkins abductee)")
    assert not is_a_description("Dr. X (French physician)")
    assert not is_a_description("Ed [sic] Rhodes")
    assert not is_a_description("")


def test_a_described_origin_is_anonymous_whatever_the_digest_called_it():
    """The corpus holds ~20 claims written `origin_kind: named` with an origin of
    "unnamed APEG biochemist" - a contradiction the extraction model does not
    notice. "named" makes independence resolve the origin to a node; a
    description has no node, so it falls back to counting each claim as its own
    root and one anonymous source reads as many."""
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    parsed = _described_parsed()
    parsed["domain_claims"][1]["speaker"] = None
    parsed["domain_claims"][1]["provenance_chain"] = {
        "origin_kind": "named",
        "origin": "[Anomaly Physical Evidence Group (APEG) biochemist]",
    }

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, parsed)

    kind, origin = conn.execute(
        "SELECT origin_kind, origin FROM claims WHERE content LIKE 'Sally%'"
    ).fetchone()
    assert kind == "anonymous"
    assert origin == "[Anomaly Physical Evidence Group (APEG) biochemist]"


def test_a_real_named_origin_is_left_alone():
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    parsed = _described_parsed()
    parsed["domain_claims"][1]["provenance_chain"] = {
        "origin_kind": "named",
        "origin": "Sally (Budd Hopkins abductee)",
    }

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, parsed)

    kind, _ = conn.execute(
        "SELECT origin_kind, origin FROM claims WHERE content LIKE 'Sally%'"
    ).fetchone()
    assert kind == "named"


def test_a_described_producer_survives_its_failed_lookup():
    """producer_id NULL already means "no author recorded" on most records. A
    source whose author was deliberately withheld is a different thing and
    carries different evidential weight, so the two must not collapse."""
    import json
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    parsed = _described_parsed()
    parsed["frontmatter"]["record_producer"] = "[senior US intelligence officer]"

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, parsed)

    producer_id, metadata = conn.execute(
        "SELECT producer_id, metadata FROM records"
    ).fetchone()
    assert producer_id is None, "there is no node, and none should be invented"
    assert json.loads(metadata)["producer"] == "[senior US intelligence officer]"


def test_a_named_producer_still_links_to_its_node():
    import json
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    parsed = _described_parsed()
    parsed["frontmatter"]["record_producer"] = "Sally (Budd Hopkins abductee)"

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    import_extraction(conn, parsed)

    producer_id, metadata = conn.execute(
        "SELECT producer_id, metadata FROM records"
    ).fetchone()
    assert producer_id is not None
    assert "producer" not in json.loads(metadata or "{}"), (
        "a resolved producer lives in producer_id; duplicating it invites drift"
    )


def test_a_producer_that_stops_resolving_does_not_keep_the_old_link():
    """producer_id was only ever set, never cleared. A record whose producer was
    rewritten to a description kept pointing at the node the rewrite had just
    retired - a described producer AND a producer_id into a retired row."""
    import json
    import sqlite3

    from assimilator.database import init_db
    from assimilator.import_markdown import import_extraction

    conn = sqlite3.connect(":memory:")
    init_db(conn)

    named = _described_parsed()
    named["frontmatter"]["record_producer"] = "Sally (Budd Hopkins abductee)"
    import_extraction(conn, named)
    assert conn.execute("SELECT producer_id FROM records").fetchone()[0] is not None

    bracketed = _described_parsed()
    bracketed["frontmatter"]["record_producer"] = "[senior US intelligence officer]"
    import_extraction(conn, bracketed)

    producer_id, metadata = conn.execute(
        "SELECT producer_id, metadata FROM records"
    ).fetchone()
    assert producer_id is None
    assert json.loads(metadata)["producer"] == "[senior US intelligence officer]"
