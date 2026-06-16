from pathlib import Path

from assimilator.reclassify import (
    is_document_name,
    normalise_person_name,
    normalise_person_names_in_file,
    normalise_place_name,
    normalise_place_names_in_file,
    reclassify_documents_in_dir,
    reclassify_documents_in_file,
)


def test_document_suffixes_match():
    assert is_document_name("Wilson Davis Memo")
    assert (
        is_document_name("Hottel FBI Memo 1950") is False
        or is_document_name("Hottel FBI Memo 1950") is True
    )
    # Trailing year shouldn't matter - "Memo" appears as a word
    assert is_document_name("Twining Letter 1947")
    assert is_document_name("FLIR1 Video")
    assert is_document_name("ODNI UAP 2022 Report")
    assert is_document_name("Pais Superluminal Craft Paper 2015")
    assert is_document_name("Hunt for Zero Point Book")
    assert is_document_name("Elizondo Resignation Letter")
    assert is_document_name("Condon Report")


def test_known_documents_match():
    assert is_document_name("FLIR1")
    assert is_document_name("Gimbal")
    assert is_document_name("Go-Fast")


def test_physical_objects_are_not_reclassified():
    assert not is_document_name("E-2 Hawkeye")
    assert not is_document_name("USS Princeton")
    assert not is_document_name("Tic Tac UAP")
    assert not is_document_name("ATFLIR Pod")
    assert not is_document_name("APG-79 Radar")
    assert not is_document_name("Bigelow Aerospace Ranch")
    assert not is_document_name("Aurora")
    assert not is_document_name("Aerospace Vehicle")


def test_exclusion_tokens_preserve_object():
    # "Reporting System" contains "Report" but is a system, not a document.
    assert not is_document_name("Marauder UAP Reporting System")
    assert not is_document_name("Defence Support Program Satellites")
    assert not is_document_name("Special Access Program")
    assert not is_document_name("Document Management System")
    assert not is_document_name("CORONA Program")


def test_programme_stays_a_matter():
    # Mark explicitly OK'd programmes staying as matters.
    assert not is_document_name("HAVE Blue Program")
    assert not is_document_name("Advanced Aerospace Weapon System Applications Program")


def test_reclassify_file_rewrites_node_type(tmp_path: Path):
    sample = """---
record_title: Sample
---

## Nodes

### 12345678-1234-1234-1234-123456789abc person: David Fravor

### 87654321-4321-4321-4321-cba987654321 object: USS Princeton

### aaaa1111-bbbb-2222-cccc-3333dddd4444 object: Hottel FBI Memo 1950

### bbbb2222-cccc-3333-dddd-4444eeee5555 matter: Condon Report

### cccc3333-dddd-4444-eeee-5555ffff6666 object: ATFLIR Pod

## Domain Claims

### claim1234-1234-1234-1234-1234567890ab [observation/first_hand] speaker:David Fravor
A claim that references object: USS Princeton. refs: Fake mention preserved.
"""
    path = tmp_path / "sample.extract.md"
    path.write_text(sample)
    count = reclassify_documents_in_file(path)
    assert count == 2  # Hottel memo + Condon Report
    out = path.read_text()
    assert "document: Hottel FBI Memo 1950" in out
    assert "document: Condon Report" in out
    # Physical objects untouched
    assert "object: USS Princeton" in out
    assert "object: ATFLIR Pod" in out
    # Claim line untouched
    assert "claim1234" in out and "[observation/first_hand]" in out


def test_reclassify_file_no_changes_leaves_file_alone(tmp_path: Path):
    sample = """## Nodes

### 12345678-1234-1234-1234-123456789abc person: David Fravor

### 87654321-4321-4321-4321-cba987654321 object: USS Princeton
"""
    path = tmp_path / "sample.extract.md"
    original = sample
    path.write_text(sample)
    count = reclassify_documents_in_file(path)
    assert count == 0
    assert path.read_text() == original


def test_normalise_person_name_basic():
    assert normalise_person_name("David Fravor") == "Fravor, David"
    assert normalise_person_name("Ross Coulthart") == "Coulthart, Ross"
    assert normalise_person_name("Robert Bigelow") == "Bigelow, Robert"


def test_normalise_person_name_three_part():
    assert normalise_person_name("John David Smith") == "Smith, John David"


def test_normalise_person_name_strips_rank_prefix():
    assert normalise_person_name("Commander David Fravor") == "Fravor, David"
    assert normalise_person_name("Lt Col Jane Doe") == "Doe, Jane"
    assert normalise_person_name("Dr. Edgar Mitchell") == "Mitchell, Edgar"
    assert normalise_person_name("Lieutenant Commander Moya") == "Moya"


def test_normalise_person_name_suffix_attaches_to_surname():
    assert normalise_person_name("Jesse Marcel Jr") == "Marcel Jr, Jesse"
    assert normalise_person_name("Jesse Marcel Jr.") == "Marcel Jr., Jesse"


def test_normalise_person_name_skips_single_word():
    assert normalise_person_name("Madonna") is None
    assert normalise_person_name("Sushi") is None


def test_normalise_person_name_skips_already_comma():
    assert normalise_person_name("Fravor, David") is None


def test_normalise_person_name_skips_call_signs_with_digits():
    assert normalise_person_name("Whiskey-99") is None
    assert normalise_person_name("Pilot 41") is None


def test_normalise_person_name_skips_parenthetical():
    assert normalise_person_name("Edgar Mitchell (Apollo 14)") is None


def test_normalise_place_name_us_state():
    assert normalise_place_name("Aztec New Mexico") == "USA, New Mexico, Aztec"
    assert normalise_place_name("Big Sur California") == "USA, California, Big Sur"
    assert normalise_place_name("Roswell New Mexico") == "USA, New Mexico, Roswell"
    assert normalise_place_name("Nevada") == "USA, Nevada"


def test_normalise_place_name_australian_state():
    assert normalise_place_name("Tully Queensland") == "Australia, Queensland, Tully"
    assert (
        normalise_place_name("Cloverly Station Queensland")
        == "Australia, Queensland, Cloverly Station"
    )


def test_normalise_place_name_canadian_province():
    assert (
        normalise_place_name("Yukon Canada") is None
    )  # has "Canada" not just province
    assert normalise_place_name("Yukon") == "Canada, Yukon"


def test_normalise_place_name_uk_country():
    # "England, Blean" style only matches if name ends in England etc.
    assert normalise_place_name("Blean England") == "United Kingdom, England, Blean"


def test_normalise_place_name_skips_with_comma():
    assert normalise_place_name("USA, Nevada, Area 51") is None


def test_normalise_place_name_returns_none_for_unrecognised():
    assert normalise_place_name("Pentagon") is None
    assert normalise_place_name("Persian Gulf") is None
    assert normalise_place_name("Random Place") is None


def test_normalise_person_names_in_file(tmp_path: Path):
    path = tmp_path / "a.extract.md"
    path.write_text(
        "### 11111111-1111-1111-1111-111111111111 person: David Fravor\n"
        "### 22222222-2222-2222-2222-222222222222 person: Commander Alex Dietrich\n"
        "### 33333333-3333-3333-3333-333333333333 person: Madonna\n"
        "### 44444444-4444-4444-4444-444444444444 organisation: VFA-41\n"
    )
    count = normalise_person_names_in_file(path)
    assert count == 2
    out = path.read_text()
    assert "person: Fravor, David" in out
    assert "person: Dietrich, Alex" in out
    assert "person: Madonna" in out  # single-name preserved
    assert "organisation: VFA-41" in out


def test_normalise_place_names_in_file(tmp_path: Path):
    path = tmp_path / "a.extract.md"
    path.write_text(
        "### 11111111-1111-1111-1111-111111111111 place: Aztec New Mexico\n"
        "### 22222222-2222-2222-2222-222222222222 place: Pentagon\n"
        "### 33333333-3333-3333-3333-333333333333 person: David Fravor\n"
    )
    count = normalise_place_names_in_file(path)
    assert count == 1
    out = path.read_text()
    assert "place: USA, New Mexico, Aztec" in out
    assert "place: Pentagon" in out  # unrecognised, unchanged
    assert "person: David Fravor" in out  # not a place line


def test_disambiguate_refs_merges_comma_in_name():
    from assimilator.reclassify import _disambiguate_refs

    node_names = {"Fravor, David", "USS Princeton", "AAV"}
    # The broken state: "David Fravor" got renamed to "Fravor, David" and
    # joined with ", " - parser sees three tokens but only two refs.
    result = _disambiguate_refs("Fravor, David, USS Princeton", node_names)
    assert result == ["Fravor, David", "USS Princeton"]


def test_disambiguate_refs_keeps_singletons_when_no_match():
    from assimilator.reclassify import _disambiguate_refs

    node_names = {"Foo", "Bar"}
    result = _disambiguate_refs("Foo, Unknown Name, Bar", node_names)
    # Unknown name kept as single token; Foo and Bar match
    assert result == ["Foo", "Unknown Name", "Bar"]


def test_disambiguate_refs_handles_two_comma_names():
    from assimilator.reclassify import _disambiguate_refs

    node_names = {"Fravor, David", "Coulthart, Ross"}
    result = _disambiguate_refs("Fravor, David, Coulthart, Ross", node_names)
    assert result == ["Fravor, David", "Coulthart, Ross"]


def test_migrate_refs_delimiter_in_file(tmp_path: Path):
    from assimilator.reclassify import migrate_refs_delimiter_in_file

    path = tmp_path / "a.extract.md"
    path.write_text(
        "### 11111111-1111-1111-1111-111111111111 person: Fravor, David\n"
        "### 22222222-2222-2222-2222-222222222222 object: USS Princeton\n"
        "### claim000-0000-0000-0000-000000000000 [observation/first_hand]\n"
        "refs: Fravor, David, USS Princeton\n"
    )
    node_names = {"Fravor, David", "USS Princeton"}
    count = migrate_refs_delimiter_in_file(path, node_names)
    assert count == 1
    text = path.read_text()
    assert "refs: Fravor, David; USS Princeton" in text


def test_migrate_refs_delimiter_idempotent(tmp_path: Path):
    from assimilator.reclassify import migrate_refs_delimiter_in_file

    # File already has semicolons - should be unchanged.
    path = tmp_path / "a.extract.md"
    text = (
        "### 11111111-1111-1111-1111-111111111111 person: Fravor, David\n"
        "refs: Fravor, David; USS Princeton\n"
    )
    path.write_text(text)
    count = migrate_refs_delimiter_in_file(path, {"Fravor, David", "USS Princeton"})
    assert count == 0
    assert path.read_text() == text


def test_reclassify_dir_aggregates(tmp_path: Path):
    (tmp_path / "a.extract.md").write_text(
        "### 11111111-1111-1111-1111-111111111111 object: A Memo\n"
        "### 22222222-2222-2222-2222-222222222222 object: USS Cole\n"
    )
    (tmp_path / "b.extract.md").write_text(
        "### 33333333-3333-3333-3333-333333333333 matter: Roswell Report\n"
    )
    (tmp_path / "ignore.txt").write_text("not an extract")
    results = reclassify_documents_in_dir(tmp_path)
    assert results == {"a.extract.md": 1, "b.extract.md": 1}
