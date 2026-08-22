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

    (tmp_path / "variants" / "some-record").mkdir(parents=True)
    (tmp_path / "r.yaml").write_text(DIGEST)
    variant = tmp_path / "variants" / "some-record" / "opus.yaml"
    variant.write_text(DIGEST)

    results = naturalise_digests_in_dir(tmp_path)

    assert list(results) == ["r.yaml"]
    assert variant.read_text() == DIGEST


COL0_DIGEST = textwrap.dedent(
    """\
    schema: anomalica/digest/1
    record:
      id: r1
      title: A Record
    nodes:
    - id: 11111111-1111-1111-1111-111111111111
      type: person
      name: Fravor, David
    - id: 33333333-3333-3333-3333-333333333333
      type: place
      name: USA, Nevada, Area 51
    domain_claims:
    - id: c1
      type: testimony
      refs:
      - id: 11111111-1111-1111-1111-111111111111
        name: Fravor, David
      text: David Fravor flew the intercept.
    """
)


def test_sequence_items_at_column_zero_are_migrated():
    """YAML admits both sequence styles and the digester emits both: `  - id:`
    with the item indented under its key, and `- id:` at column 0. Treating the
    second as a top-level key cleared the in-nodes flag on the very first node
    entry, so no metadata block was written and verification then refused the
    whole file - leaving every column-0 digest silently unmigrated while the pass
    reported success on the rest."""
    rewritten, renamed = naturalise_digest_text(COL0_DIGEST)
    doc = yaml.safe_load(rewritten)

    assert renamed == 1
    assert doc["nodes"][0]["name"] == "David Fravor"
    assert doc["nodes"][0]["metadata"] == {
        "family_name": "Fravor",
        "aliases": ["Fravor, David"],
    }
    assert doc["domain_claims"][0]["refs"][0]["name"] == "David Fravor"
    assert doc["nodes"][1]["name"] == "USA, Nevada, Area 51"


MANGLED = textwrap.dedent(
    """\
    schema: anomalica/digest/1
    record:
      id: r1
      title: A Record
      producer: widow of Louis Emrich) Emrich (Mrs.
    nodes:
      - id: 11111111-1111-1111-1111-111111111111
        type: person
        name: widow of Louis Emrich) Emrich (Mrs.
        metadata:
          family_name: 'Emrich (Mrs.'
          aliases:
            - Emrich (Mrs., widow of Louis Emrich)
      - id: 22222222-2222-2222-2222-222222222222
        type: person
        name: of Calcutta) Teresa (Mother
        metadata:
          role: Missionary
          family_name: 'Teresa (Mother'
          aliases:
            - Teresa (Mother, of Calcutta)
      - id: 33333333-3333-3333-3333-333333333333
        type: person
        name: David Fravor
        metadata:
          family_name: Fravor
          aliases:
            - Fravor, David
    domain_claims:
      - id: c1
        type: testimony
        speaker:
          id: 11111111-1111-1111-1111-111111111111
          name: widow of Louis Emrich) Emrich (Mrs.
        refs:
          - id: 22222222-2222-2222-2222-222222222222
            name: of Calcutta) Teresa (Mother
        text: Mrs. Emrich told Hans Bender about the Fatima story.
    """
)


def test_a_comma_inside_a_parenthetical_is_not_a_surname_separator():
    """ "Emrich (Mrs., widow of Louis Emrich)" is a name with a description
    attached, not surname-first form. Splitting on that comma put
    "widow of Louis Emrich) Emrich (Mrs." into the live graph."""
    assert parse_surname_first("Emrich (Mrs., widow of Louis Emrich)") is None
    assert parse_surname_first("Teresa (Mother, of Calcutta)") is None
    # A comma OUTSIDE the brackets still separates, and the parenthetical still
    # travels to the end of the natural form.
    assert parse_surname_first("Smith (a, b), John").natural == "John Smith (a, b)"
    assert (
        parse_surname_first("Elizondo, Luis D. III (father)").natural
        == "Luis D. Elizondo III (father)"
    )


def test_restore_puts_the_mangled_names_back_everywhere():
    from assimilator.person_names import unmangle_digest_text

    rewritten, fixed = unmangle_digest_text(MANGLED)
    doc = yaml.safe_load(rewritten)

    assert fixed == 2
    assert doc["record"]["producer"] == "Emrich (Mrs., widow of Louis Emrich)"
    assert doc["nodes"][0]["name"] == "Emrich (Mrs., widow of Louis Emrich)"
    assert doc["nodes"][1]["name"] == "Teresa (Mother, of Calcutta)"
    assert doc["domain_claims"][0]["speaker"]["name"] == (
        "Emrich (Mrs., widow of Louis Emrich)"
    )
    assert doc["domain_claims"][0]["refs"][0]["name"] == "Teresa (Mother, of Calcutta)"


def test_restore_removes_only_what_the_migration_injected():
    """family_name and the alias were added by the migration and are wrong, but
    metadata the digester wrote must survive."""
    from assimilator.person_names import unmangle_digest_text

    doc = yaml.safe_load(unmangle_digest_text(MANGLED)[0])

    assert "metadata" not in doc["nodes"][0], "metadata existed only to hold them"
    assert doc["nodes"][1]["metadata"] == {"role": "Missionary"}


def test_restore_leaves_correctly_migrated_people_alone():
    from assimilator.person_names import unmangle_digest_text

    doc = yaml.safe_load(unmangle_digest_text(MANGLED)[0])

    assert doc["nodes"][2] == {
        "id": "33333333-3333-3333-3333-333333333333",
        "type": "person",
        "name": "David Fravor",
        "metadata": {"family_name": "Fravor", "aliases": ["Fravor, David"]},
    }


def test_restoring_twice_is_a_no_op():
    from assimilator.person_names import unmangle_digest_text

    once, _ = unmangle_digest_text(MANGLED)
    twice, fixed = unmangle_digest_text(once)
    assert fixed == 0
    assert twice == once


def test_the_migration_no_longer_creates_what_the_restore_undoes():
    """The forward pass must be idempotent against the fixed parser, or the next
    run re-mangles everything this repaired."""
    from assimilator.person_names import unmangle_digest_text

    restored, _ = unmangle_digest_text(MANGLED)
    remigrated, renamed = naturalise_digest_text(restored)

    assert renamed == 0
    assert remigrated == restored
