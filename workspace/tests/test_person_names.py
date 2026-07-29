"""The surname-first -> natural-order migration (node-types.md, 2026-06-29).

Covers what the rewrite must and must not touch: person names invert and gain a
family_name field, places keep their largest-unit-first commas, and the digest
text outside the renamed names is left byte-identical.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from assimilator.person_names import (
    display_surname_first,
    naturalise_digest_text,
    parse_surname_first,
)


@pytest.mark.parametrize(
    "surname_first,natural,family",
    [
        ("Fravor, David", "David Fravor", "Fravor"),
        ("Curtis, D. C.", "D. C. Curtis", "Curtis"),
        # The generational suffix sits on either side of the comma depending on
        # which pass wrote the name; both must land after the family name.
        ("Lynn III, William", "William Lynn III", "Lynn"),
        ("Lynn, William J. III", "William J. Lynn III", "Lynn"),
        ("Greenewald Jr., John", "John Greenewald Jr.", "Greenewald"),
        ("Elizondo, Luis D. III (father)", "Luis D. Elizondo III (father)", "Elizondo"),
    ],
)
def test_parses_and_inverts(surname_first, natural, family):
    parsed = parse_surname_first(surname_first)
    assert parsed.natural == natural
    assert parsed.family == family


@pytest.mark.parametrize(
    "name", ["David Fravor", "Semjase", "Mohammed bin Rashid Al Maktoum", "岸田文雄"]
)
def test_leaves_names_that_are_not_comma_form(name):
    assert parse_surname_first(name) is None


def test_display_needs_the_field_and_never_guesses():
    assert display_surname_first("David Fravor", {"family_name": "Fravor"}) == (
        "Fravor, David"
    )
    # No family_name: returned unchanged rather than guessed from the last token,
    # which is wrong for exactly the names the field exists to protect.
    assert display_surname_first("Mohammed bin Rashid Al Maktoum") == (
        "Mohammed bin Rashid Al Maktoum"
    )


DIGEST = textwrap.dedent(
    """\
    schema: anomalica/digest/1
    record:
      id: r1
      title: A Record
      producer: Fravor, David
    nodes:
      - id: 11111111-1111-1111-1111-111111111111
        type: person
        name: Fravor, David
      - id: 22222222-2222-2222-2222-222222222222
        type: person
        name: Mace, Nancy
        metadata:
          role: Chairwoman
      - id: 33333333-3333-3333-3333-333333333333
        type: place
        name: USA, Nevada, Area 51
      - id: 44444444-4444-4444-4444-444444444444
        type: person
        name: Semjase
    domain_claims:
      - id: c1
        type: testimony
        speaker:
          id: 11111111-1111-1111-1111-111111111111
          name: Fravor, David
        refs:
          - id: 11111111-1111-1111-1111-111111111111
            name: Fravor, David
          - id: 33333333-3333-3333-3333-333333333333
            name: USA, Nevada, Area 51
        quote: My name is David Fravor.
        text: David Fravor flew the intercept.
    """
)


def test_rewrite_inverts_persons_everywhere_and_records_the_surname():
    rewritten, renamed = naturalise_digest_text(DIGEST)
    doc = yaml.safe_load(rewritten)
    assert renamed == 2

    fravor = doc["nodes"][0]
    assert fravor["name"] == "David Fravor"
    assert fravor["metadata"]["family_name"] == "Fravor"
    assert fravor["metadata"]["aliases"] == ["Fravor, David"]

    claim = doc["domain_claims"][0]
    assert claim["speaker"]["name"] == "David Fravor"
    assert [r["name"] for r in claim["refs"]] == [
        "David Fravor",
        "USA, Nevada, Area 51",
    ]


def test_existing_node_metadata_is_extended_not_replaced():
    doc = yaml.safe_load(naturalise_digest_text(DIGEST)[0])
    mace = doc["nodes"][1]
    assert mace["name"] == "Nancy Mace"
    assert mace["metadata"] == {
        "role": "Chairwoman",
        "family_name": "Mace",
        "aliases": ["Mace, Nancy"],
    }


def test_places_and_single_name_people_are_untouched():
    doc = yaml.safe_load(naturalise_digest_text(DIGEST)[0])
    assert doc["nodes"][2]["name"] == "USA, Nevada, Area 51"
    assert doc["nodes"][2].get("metadata") is None
    assert doc["nodes"][3] == {
        "id": "44444444-4444-4444-4444-444444444444",
        "type": "person",
        "name": "Semjase",
    }


def test_only_name_lines_change():
    """The rewrite is line-targeted, not a YAML round-trip: everything that is
    not a renamed name must come back byte-identical, or a 4-line rename lands
    as a whole-file reformat."""
    rewritten, _ = naturalise_digest_text(DIGEST)
    before = DIGEST.splitlines()
    after = rewritten.splitlines()
    changed = [line for line in before if line not in after]
    assert all(
        line.strip().lstrip("- ").startswith(("name:", "producer:")) for line in changed
    )
    # Claim text and quotes mention the person too, and must NOT be rewritten.
    assert "    quote: My name is David Fravor." in after
    assert "    text: David Fravor flew the intercept." in after


def test_rerunning_is_a_no_op():
    once, _ = naturalise_digest_text(DIGEST)
    twice, renamed = naturalise_digest_text(once)
    assert renamed == 0
    assert twice == once


def test_record_producer_moves_with_the_rename():
    """The producer is a bare name with no id beside it, and the importer links
    it to a node by exact name - left behind, the record loses its producer."""
    doc = yaml.safe_load(naturalise_digest_text(DIGEST)[0])
    assert doc["record"]["producer"] == "David Fravor"


def test_variant_snapshots_are_never_rewritten(tmp_path):
    """digests/variants/ holds what each model actually emitted; rewriting those
    falsifies the model comparison. Guarded here, not left to the caller's glob,
    because pointing the pass at the digests repo root is the obvious mistake."""
    from assimilator.person_names import naturalise_digests_in_dir

    (tmp_path / "records").mkdir()
    (tmp_path / "variants" / "some-record").mkdir(parents=True)
    (tmp_path / "records" / "r.yaml").write_text(DIGEST)
    variant = tmp_path / "variants" / "some-record" / "opus.yaml"
    variant.write_text(DIGEST)

    results = naturalise_digests_in_dir(tmp_path)

    assert list(results) == ["records/r.yaml"]
    assert variant.read_text() == DIGEST
