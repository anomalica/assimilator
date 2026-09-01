import sqlite3

from assimilator.database import init_db, insert_alias, insert_node
from assimilator.matching import (
    FUZZY_NAME_THRESHOLD,
    _component_similarity,
    collapse_acronym_expansions,
    fuzzy_name_similarity,
    is_bare_acronym_for,
    punctuation_blind_key,
    is_nickname_of,
    looks_like_a_bare_acronym,
    match_node,
    normalise_node_name,
)
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


# --- Structured-name false-merge regression (precision) ----------------------
#
# The fuzzy matcher used full-string Levenshtein, which scored on the shared
# structure of comma-structured names: "Surname, First" people sharing a first
# name, and hierarchical "Country, State, City" places sharing a prefix, both
# crossed the 0.75 threshold and were wrongly merged into one node. These cases
# assert distinct real-world entities now stay separate.

# People with the same surname OR the same first name - distinct individuals.
_PERSON_FALSE_MERGES = [
    ("Hill, Barney", "Hill, Betty"),  # the famous abduction couple
    ("Walton, Travis", "Taylor, Travis"),
    ("Lear, John", "Alexander, John"),
    ("Mariani, Dennis", "Grant, Dennis"),
    ("Ramsay, Chris", "Sato, Chris"),
    ("Stevens, Wendell", "Stevens, Ted"),
]

# Places sharing a "Country, State" prefix but a different city - distinct places.
_PLACE_FALSE_MERGES = [
    ("USA, Nevada, Area 51", "USA, Nevada, Las Vegas"),
    ("USA, Nevada, S4", "USA, Nevada, Fallon"),
    ("USA, Illinois, Chicago", "USA, Illinois, Wheeling"),
    ("USA, Arizona, Sedona", "USA, Arizona, Tucson"),
    ("USA, New Mexico, Los Alamos", "USA, New Mexico, Los Brasos"),
    ("USA, Pennsylvania, Pittsburgh", "USA, Pennsylvania, Kecksburg"),
]


def _assert_distinct(node_type, existing, incoming):
    """Insert `existing`, then assert `incoming` does not fuzzy-merge into it."""
    conn = _db()
    insert_node(conn, Node(node_type=node_type, name=existing))
    conn.commit()
    result = match_node(conn, incoming, node_type.value)
    assert result is None, f"{incoming!r} wrongly merged into {existing!r}: {result}"


def test_distinct_people_sharing_a_name_do_not_merge():
    for existing, incoming in _PERSON_FALSE_MERGES:
        _assert_distinct(NodeType.person, existing, incoming)
        # Symmetric: order of insertion must not matter.
        _assert_distinct(NodeType.person, incoming, existing)


def test_distinct_places_sharing_a_prefix_do_not_merge():
    for existing, incoming in _PLACE_FALSE_MERGES:
        _assert_distinct(NodeType.place, existing, incoming)
        _assert_distinct(NodeType.place, incoming, existing)


def test_same_city_in_different_state_does_not_merge():
    # Same most-specific component, different parent - still distinct places.
    _assert_distinct(
        NodeType.place, "USA, Nevada, Springfield", "USA, Illinois, Springfield"
    )


def test_same_first_initial_different_surname_does_not_merge():
    # An initial must not collapse two different surnames.
    _assert_distinct(NodeType.person, "Lear, J.", "Alexander, J.")


# --- Good merges that must still work (recall) --------------------------------


def _assert_merges(node_type, existing, incoming):
    conn = _db()
    node = insert_node(conn, Node(node_type=node_type, name=existing))
    conn.commit()
    result = match_node(conn, incoming, node_type.value)
    assert result is not None, f"{incoming!r} failed to merge into {existing!r}"
    assert result[0] == node.id
    return result


def test_trailing_period_first_name_variant_still_merges():
    # "Eisenhower, Dwight D" vs "Eisenhower, Dwight D." - same person.
    # Matched by the equivalence key (punctuation-folded) rather than by the
    # fuzzy threshold: same outcome by a stronger route, since equivalence is
    # exact-after-normalisation and needs no similarity cut to be trusted.
    res = _assert_merges(
        NodeType.person, "Eisenhower, Dwight D", "Eisenhower, Dwight D."
    )
    assert res[1] in ("acronym", "fuzzy")


def test_initial_vs_full_first_name_still_merges():
    # "K. Day" vs "Kevin Day" - the initial-vs-full case the threshold targets.
    _assert_merges(NodeType.person, "Kevin Day", "K. Day")
    _assert_merges(NodeType.person, "K. Day", "Kevin Day")


def test_surname_first_typo_still_merges():
    # Genuine typo in the first name component must still merge.
    _assert_merges(NodeType.person, "Hill, Barney", "Hill, Barny")


def test_place_last_component_typo_still_merges():
    _assert_merges(
        NodeType.place, "USA, New Mexico, Los Alamos", "USA, New Mexico, Los Alamus"
    )


# --- fuzzy_name_similarity unit checks ---------------------------------------


def test_fuzzy_name_similarity_structured_vs_plain():
    from assimilator.matching import (
        fuzzy_name_similarity,
        FUZZY_NAME_THRESHOLD,
    )

    # Distinct structured names score below the merge threshold.
    for a, b in _PERSON_FALSE_MERGES + _PLACE_FALSE_MERGES:
        assert fuzzy_name_similarity(a.lower(), b.lower()) < FUZZY_NAME_THRESHOLD

    # Variants of the same entity score at or above it.
    assert (
        fuzzy_name_similarity("eisenhower, dwight d", "eisenhower, dwight d.")
        >= FUZZY_NAME_THRESHOLD
    )
    assert fuzzy_name_similarity("k. day", "kevin day") >= FUZZY_NAME_THRESHOLD
    assert fuzzy_name_similarity("david fravor", "david favor") >= FUZZY_NAME_THRESHOLD


# --- Plain (non-comma) name false-merge regression (precision) ----------------
#
# Whole-string Levenshtein scored plain names on their shared words, so a pair
# that agreed on every common word but differed on the ONE distinguishing token
# crossed the 0.75 threshold and wrongly merged. Two token-level discriminators
# now block that class: a differing hard token (number, year, designator), and a
# substituted distinctive word (each name holds a proper noun the other lacks).
# Every case below is a real pair from a 22-digest graph rebuild.

# Differing number, year or alphanumeric designator => different entity.
_HARD_TOKEN_FALSE_MERGES = [
    (
        NodeType.document,
        "FY2024 National Defense Authorization Act",
        "FY2023 National Defense Authorization Act",
    ),
    (NodeType.document, "Executive Order 12333", "Executive Order 13526"),
    (
        NodeType.document,
        "FY2023 NDAA UAP Section S1632",
        "FY2023 NDAA UAP Section 1673",
    ),
    (
        NodeType.event,
        "Woomera 1952 UAP Radar Detection",
        "Woomera 1954 Radar Detection",
    ),
    (
        NodeType.event,
        "Erik Nanstiel 2020 garage grey encounter",
        "Erik Nanstiel 1994 grey encounter",
    ),
    (
        NodeType.event,
        "Erik Nanstiel 2022 arm surgery encounter",
        "Erik Nanstiel 1994 grey encounter",
    ),
    (
        NodeType.organisation,
        "Strike Fighter Squadron 14 (VFA-14)",
        "Strike Fighter Squadron 41 (VFA-41)",
    ),
    (
        NodeType.organisation,
        "Strike Fighter Squadron 94 (VFA-94)",
        "Strike Fighter Squadron 41 (VFA-41)",
    ),
    (NodeType.object, "Malaysia Airlines MH17", "Malaysia Airlines MH370"),
    (NodeType.object, "APG-79 Radar", "APG-73 Radar"),
    (NodeType.object, "E-2D Hawkeye", "E-2C Hawkeye"),
    (NodeType.organisation, "Joint Staff J2", "Joint Staff J3"),
    (
        NodeType.event,
        "Unidentified Season 2 Premiere",
        "Unidentified Season 1 Premiere 2019",
    ),
]

# Differing distinctive proper noun => different entity, even with shared words.
_PROPER_NOUN_FALSE_MERGES = [
    (NodeType.place, "Andrews Air Force Base", "Vandenberg Air Force Base"),
    (NodeType.place, "Edwards Air Force Base", "Vandenberg Air Force Base"),
    (NodeType.place, "Kadena Air Force Base", "Vandenberg Air Force Base"),
    (NodeType.place, "MacDill Air Force Base", "Vandenberg Air Force Base"),
    (NodeType.organisation, "Cardiff University", "Stanford University"),
    (NodeType.organisation, "Harvard University", "Stanford University"),
    (NodeType.organisation, "University of Colorado", "University of Houston"),
    (NodeType.organisation, "University of Ottawa", "University of Houston"),
    (NodeType.organisation, "University of Miami", "University of Maryland"),
    (
        NodeType.organisation,
        "Central Intelligence Agency (CIA)",
        "Defense Intelligence Agency (DIA)",
    ),
    (
        NodeType.organisation,
        "Defense Counterintelligence Security Agency (DCSA)",
        "Defense Intelligence Agency (DIA)",
    ),
    (NodeType.organisation, "National Security Council", "National Security Agency"),
    (
        NodeType.organisation,
        "Office of Naval Intelligence",
        "Director of National Intelligence",
    ),
    (
        NodeType.organisation,
        "House Permanent Select Committee on Intelligence",
        "Senate Select Committee on Intelligence",
    ),
    (NodeType.object, "Gray Alien Species", "Nordic Alien Species"),
    (
        NodeType.organisation,
        "Royal New Zealand Air Force",
        "Royal Australian Air Force",
    ),
    (NodeType.organisation, "SOL Foundation", "Simons Foundation"),
    (NodeType.organisation, "SRI International", "EarthTech International"),
    (NodeType.organisation, "Time Magazine", "GQ Magazine"),
    (NodeType.organisation, "True Magazine", "GQ Magazine"),
    (NodeType.organisation, "Washington Post", "Huffington Post"),
    (NodeType.organisation, "The New Yorker", "The New York Times"),
    (NodeType.organisation, "Department of Energy", "Department of the Army"),
    (
        NodeType.organisation,
        "Democratic Congressional Campaign Committee",
        "Democratic National Committee",
    ),
    (
        NodeType.project,
        "Hexagon NRO Photoreconnaissance Program",
        "Gambit NRO Photoreconnaissance Program",
    ),
    (NodeType.project, "Project Sign", "Project Condign"),
    (
        NodeType.organisation,
        "Air Force Research Laboratory",
        "Naval Research Laboratory",
    ),
    (NodeType.event, "RAF Shawbury UAP Sighting 1993", "RAF Cosford UAP Sighting 1993"),
    (
        NodeType.event,
        "Kadena Air Force Base UAP Sighting",
        "Eglin Air Force Base UAP Incident",
    ),
    (
        NodeType.person,
        "Hamdan bin Mohammed Al Maktoum",
        "Mohammed bin Rashid Al Maktoum",
    ),
]


def test_hard_token_difference_blocks_plain_merge():
    for node_type, existing, incoming in _HARD_TOKEN_FALSE_MERGES:
        _assert_distinct(node_type, existing, incoming)
        _assert_distinct(node_type, incoming, existing)


def test_distinctive_proper_noun_difference_blocks_plain_merge():
    for node_type, existing, incoming in _PROPER_NOUN_FALSE_MERGES:
        _assert_distinct(node_type, existing, incoming)
        _assert_distinct(node_type, incoming, existing)


# --- Plain-name good merges that must still work (recall) ----------------------
#
# These differ only on common/structural words, articles, spelling, accents or
# an acronym suffix - the distinctive words agree - so they must still merge.

_PLAIN_GOOD_MERGES = [
    (NodeType.organisation, "New York Times", "The New York Times"),
    (NodeType.organisation, "Joe Rogan Experience", "The Joe Rogan Experience"),
    (NodeType.organisation, "Arlington Institute", "The Arlington Institute"),
    (NodeType.organisation, "Department of Defense", "Department of Defense (DoD)"),
    (
        NodeType.organisation,
        "Special Access Programs (SAPs)",
        "Special Access Programs",
    ),
    (
        NodeType.organisation,
        "Securities Exchange Commission",
        "Securities and Exchange Commission",
    ),
    (
        NodeType.organisation,
        "Center for the Study of Extra-terrestrial Intelligence",
        "Centre for the Study of Extra-terrestrial Intelligence",
    ),
    (NodeType.organisation, "US Army Counterintelligence", "Army Counterintelligence"),
    (
        NodeType.organisation,
        "US Army Combat Capabilities Development Command",
        "Army Combat Capabilities Development Command",
    ),
    # A year on one side only is a name-extension, not a substitution.
    (NodeType.event, "DeLonge Joe Rogan Interview 2017", "DeLonge Joe Rogan Interview"),
    (
        NodeType.event,
        "Vandenberg ICBM UAP Filming",
        "Vandenberg Atlas ICBM UAP Filming 1964",
    ),
    # Same year both sides, one side adds a distinctive word - still a merge.
    (
        NodeType.event,
        "Lake Erie UAP Incident 1988",
        "Lake Erie Coast Guard UAP Incident 1988",
    ),
]


def test_plain_good_merges_still_merge():
    for node_type, existing, incoming in _PLAIN_GOOD_MERGES:
        res = _assert_merges(node_type, existing, incoming)
        assert res[1] in ("fuzzy", "acronym")


def test_hard_token_difference_scores_below_threshold():
    from assimilator.matching import fuzzy_name_similarity, FUZZY_NAME_THRESHOLD

    for _type, a, b in _HARD_TOKEN_FALSE_MERGES + _PROPER_NOUN_FALSE_MERGES:
        assert fuzzy_name_similarity(a.lower(), b.lower()) < FUZZY_NAME_THRESHOLD, (
            f"{a!r} vs {b!r} should not reach the merge threshold"
        )


def test_distinctive_token_disagreement_helper():
    from assimilator.matching import _distinctive_tokens_disagree

    # Differing hard token / substituted proper noun => disagreement.
    assert _distinctive_tokens_disagree(
        "executive order 12333", "executive order 13526"
    )
    assert _distinctive_tokens_disagree("cardiff university", "stanford university")
    # One-sided extension and spelling variants => agreement.
    assert not _distinctive_tokens_disagree(
        "vandenberg icbm uap filming", "vandenberg atlas icbm uap filming 1964"
    )
    assert not _distinctive_tokens_disagree(
        "center for the study of extra-terrestrial intelligence",
        "centre for the study of extra-terrestrial intelligence",
    )
    assert not _distinctive_tokens_disagree("k. day", "kevin day")
    # A year that only adds words on one side is a name-extension, not a clash.
    assert not _distinctive_tokens_disagree(
        "delonge george knapp 2016-03 interview",
        "delonge george knapp interview 2016",
    )


def test_hard_token_date_prefix_does_not_conflict():
    from assimilator.matching import _hard_tokens, _hard_tokens_conflict

    # A bare year is the date-prefix of a fuller ISO date for the same event,
    # so the two should not be treated as conflicting designators.
    def conflict(a, b):
        return _hard_tokens_conflict(_hard_tokens(a), _hard_tokens(b))

    assert not conflict("nimitz uap intercept 2004-11-14", "2004 nimitz uap encounter")
    assert not conflict("knapp 2016-03 interview", "knapp interview 2016")
    # But a year is NOT a prefix of a different year or designator.
    assert conflict("woomera 1952 radar", "woomera 1954 radar")
    assert conflict("fy2024 ndaa", "fy2023 ndaa")
    assert conflict("executive order 12333", "executive order 13526")


def test_accented_and_unaccented_spellings_are_one_node():
    """ "André Almond" and "Andre Almond" are one person written two ways. The
    fuzzy path cannot save this pair - the accent makes each forename an orphan
    of the other, which reads as a substituted distinctive token - so the
    equivalence key folds diacritics off ASCII-Latin bases."""
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.person, name="André Almond"))
    conn.commit()

    result = match_node(conn, "Andre Almond", "person")
    assert result is not None
    assert result[0] == node.id


def test_diacritic_fold_does_not_reach_non_latin_scripts():
    """A blanket NFD-strip would turn ガ into カ and merge distinct names, so the
    fold only applies where the base letter is ASCII."""
    from assimilator.matching import fold_diacritics

    assert fold_diacritics("ガガーリン") == "ガガーリン"
    assert fold_diacritics("André Almond") == "Andre Almond"


def test_surname_shared_forename_differs_stays_distinct():
    """The natural-order form of the #23 precision case. Under the old
    "Surname, First" storage these two scored exactly at the structured
    threshold and merged - which is how 30 claims about the Pentagon's Garry
    Reid were filed under Senator Harry Reid."""
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="Harry Reid"))
    conn.commit()

    assert match_node(conn, "Garry Reid", "person") is None


def test_word_punctuation_does_not_split_an_entity():
    """ "KLAS-TV" and "KLAS TV" are one broadcaster; both were live nodes with a
    shared slug, so the site would have had two pages competing for one URL."""
    conn = _db()
    node = insert_node(conn, Node(node_type=NodeType.organisation, name="KLAS-TV"))
    conn.commit()

    assert match_node(conn, "KLAS TV", "organisation")[0] == node.id


def test_the_fold_does_not_join_distinct_designators():
    """Folding to a SPACE rather than deleting keeps strings that differ by more
    than punctuation apart."""
    from assimilator.matching import name_equivalence_key

    assert name_equivalence_key("E-2 Hawkeye") == name_equivalence_key("E 2 Hawkeye")
    assert name_equivalence_key("E2 Hawkeye") != name_equivalence_key("E-2 Hawkeye")


def test_a_wording_variant_does_not_get_expanded_again():
    """The guard cannot be an exact match against OUR wording.

    The model writes a programme's name as its source wrote it. "Advanced
    Aerospace Weapons Systems Applications Program (AAWSAP)" is the same
    programme as the singular form in _PROGRAMME_EXPANSIONS, but an exact
    substring test misses it and expands the bare acronym inside the
    parenthetical that is already there. Two expanders run in sequence, so the
    corpus holds a node named "...Program (...Program (...Program (AAWSAP)))".
    """
    plural = "Advanced Aerospace Weapons Systems Applications Program (AAWSAP)"
    assert normalise_node_name(plural) == plural

    singular = "Advanced Aerospace Weapon System Applications Program (AAWSAP)"
    assert normalise_node_name(singular) == singular

    # A bare acronym still expands - the guard must not disable the feature.
    assert normalise_node_name("AAWSAP") == singular


def test_a_short_form_is_the_same_person():
    """Levenshtein does not reach these - "Dave"/"David" scores below the fuzzy
    threshold - so without a nickname step they become two nodes, and each
    understates its own evidence because the counts are per node."""
    assert is_nickname_of("Dave Fravor", "David Fravor")
    assert is_nickname_of("Hal Puthoff", "Harold Puthoff")
    assert is_nickname_of("Dick Gordon", "Richard Gordon")
    assert is_nickname_of(
        "Chris Bledsoe", "Christopher Bledsoe"
    )  # prefix, not a table entry
    assert is_nickname_of("David Fravor", "Dave Fravor")  # order does not matter


def test_the_rest_of_the_name_must_match_exactly():
    """The strictness IS the rule. Matching on surname plus a nickname-ish first
    name found 37 pairs in the corpus and was wrong about a third of them. Every
    case below is one the loose version accepted."""
    assert not is_nickname_of("John Fitzgerald Kennedy", "John Neely Kennedy")
    assert not is_nickname_of("George Herbert Walker Bush", "George W. Bush")
    assert not is_nickname_of("Robert Amory Jr.", "Robert C. Seamans Jr.")
    assert not is_nickname_of("Baron Magnus von Braun", "Baroness Emmy von Braun")


def test_two_characters_is_not_evidence_of_a_short_form():
    """ "Al" prefixes Alan, Alex and Alfred alike, so a two-character stem cannot
    identify anyone."""
    assert not is_nickname_of("Al Smith", "Alan Smith")


def test_a_different_person_is_not_a_nickname():
    assert not is_nickname_of("David Fravor", "David Grusch")
    assert not is_nickname_of("Betty Hill", "Barney Hill")
    assert not is_nickname_of(
        "Fravor", "David Fravor"
    )  # single token, no surname to compare


def test_a_bare_acronym_belongs_to_the_node_that_spells_it_out():
    """name_equivalence_key collapses "X" against "X (ACRO)" - the same words with
    and without the parenthetical. It cannot collapse "NASA" against "National
    Aeronautics and Space Administration (NASA)", because it strips the
    parenthetical from one side and compares "nasa" to the spelled-out words, so
    the bare form became its own node. 26 acronyms in the corpus had both forms.
    """
    assert is_bare_acronym_for(
        "NASA", "National Aeronautics and Space Administration (NASA)"
    )
    assert is_bare_acronym_for("MJ-12", "Majestic 12 (MJ-12)")
    assert is_bare_acronym_for("AAV", "Anomalous Aerial Vehicle (AAV)")


def test_the_declared_acronym_is_the_evidence_not_the_initials():
    """Deriving an acronym from initials would match far too much - "Advanced
    Aerospace Threat Identification Program" and "Airborne Anomaly Tracking
    Initiative Programme" both reduce to AATIP. Only a trailing parenthetical
    counts."""
    assert not is_bare_acronym_for(
        "NASA", "National Oceanic and Atmospheric Administration (NOAA)"
    )
    assert not is_bare_acronym_for(
        "NASA", "National Aeronautics and Space Administration"
    )
    assert not is_bare_acronym_for("NASA", "NASA")
    assert not is_bare_acronym_for("Bob", "Robert Smith (RS)")


def test_only_an_all_caps_single_token_is_a_bare_acronym():
    assert looks_like_a_bare_acronym("NASA")
    assert looks_like_a_bare_acronym("MJ-12")
    assert not looks_like_a_bare_acronym(
        "National Aeronautics and Space Administration (NASA)"
    )
    assert not looks_like_a_bare_acronym("Bob")


def test_punctuation_and_spacing_do_not_make_a_second_entity():
    """Nine same-type pairs in the corpus differed by nothing but punctuation:
    "Office of the Under Secretary of Defense for Intelligence (OUSDI)" against
    "...Undersecretary..." at 42 references and 19, and eight more."""
    same = punctuation_blind_key
    assert same("KLAS-TV") == same("KLAS TV")
    assert same("Stargate") == same("Star Gate")
    assert same("F-117A Nighthawk") == same("F-117A Night Hawk")
    assert same("S-4 Facility") == same("S4 (facility)")
    assert same(
        "Office of the Under Secretary of Defense for Intelligence (OUSDI)"
    ) == same("Office of the Undersecretary of Defense for Intelligence (OUSDI)")


def test_punctuation_blindness_still_separates_different_names():
    same = punctuation_blind_key
    assert same("David Fravor") != same("David Grusch")
    assert same("Apollo 14") != same("Apollo 15")


def test_an_acronym_inside_a_person_name_is_never_expanded():
    """ "UAP Gerb" is the handle of a real UAP researcher. The whole-word
    substitution turned it into "Unidentified Aerial Phenomena (UAP) Gerb",
    which reached the page gate as page-worthy with 36 claims and 8 independent
    sources and was then read downstream as a corrupted merge of two people.
    A name is the one field where an expansion is never a clarification."""
    from assimilator.matching import normalise_node_name

    assert normalise_node_name("UAP Gerb", "person") == "UAP Gerb"
    assert normalise_node_name("UAP Juan", "person") == "UAP Juan"
    # A programme acronym in a person's name is exempt too; the same string as
    # an organisation still expands. The exemption is about what a person's name
    # IS, not about which acronym it happens to contain.
    assert normalise_node_name("AATIP Dave", "person") == "AATIP Dave"
    assert normalise_node_name("AATIP Dave", "organisation") != "AATIP Dave"
    # An untyped call keeps the old behaviour rather than silently exempting.
    assert normalise_node_name("AATIP Dave") != "AATIP Dave"


def test_a_description_never_matches_a_node():
    """A description is record-scoped, so there is no node it could correctly
    resolve to - and the fuzzy tier finds one anyway. "[Anomaly Physical Evidence
    Group (APEG) biochemist]" matched the node "unnamed Anomaly Physical Evidence
    Group (APEG) biochemist" it had just been written to replace, re-creating the
    refs the rewrite had removed."""
    import sqlite3

    from assimilator.database import init_db, insert_node
    from anomalica_common.digest.models import Node
    from assimilator.matching import match_node

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn,
        Node(
            id="n1",
            node_type="person",
            name="unnamed Anomaly Physical Evidence Group (APEG) biochemist",
        ),
    )
    conn.commit()

    assert (
        match_node(
            conn, "[Anomaly Physical Evidence Group (APEG) biochemist]", "person"
        )
        is None
    )
    # The unbracketed form still matches - the exemption is the marker, not the words.
    assert match_node(
        conn, "unnamed Anomaly Physical Evidence Group (APEG) biochemist", "person"
    )


class TestAcronymBoilerplate:
    """The mandated expansion must not decide a comparison on its own.

    Event names carry "... Unidentified Flying Object (UFO) incident" while the
    same event arrives elsewhere in the short form. Literal comparison gets both
    directions wrong at once: unrelated events score high on the shared tail,
    and one event written both ways scores low on the spelled-out words.
    """

    def test_different_events_sharing_only_the_boilerplate_are_rejected(self):
        assert (
            fuzzy_name_similarity(
                "1947 roswell unidentified flying object (ufo) incident",
                "2004 uss nimitz unidentified flying object (ufo) incident",
            )
            < FUZZY_NAME_THRESHOLD
        )

    def test_same_event_in_both_acronym_spellings_matches(self):
        assert (
            fuzzy_name_similarity(
                "1947 kenneth arnold ufo sighting, mount rainier",
                "1947 kenneth arnold unidentified flying object (ufo) sighting, "
                "mount rainier",
            )
            >= FUZZY_NAME_THRESHOLD
        )

    def test_one_side_spelled_out_without_declaring_the_acronym_still_matches(self):
        # Collapsing only the declaring side would move these APART, so the
        # comparison keeps the better of both spellings.
        assert (
            fuzzy_name_similarity(
                "artificial intelligence (ai)", "artificial intelligence"
            )
            >= FUZZY_NAME_THRESHOLD
        )

    def test_a_topic_is_not_an_extension_of_the_bare_phenomenon(self):
        # "Unidentified Flying Object (UFO)" collapses to "UFO", and without a
        # guard every "UFO <something>" reads as an extension of it.
        assert (
            fuzzy_name_similarity("ufo disclosure", "unidentified flying object (ufo)")
            < FUZZY_NAME_THRESHOLD
        )

    def test_collapse_needs_the_initials_as_evidence(self):
        assert collapse_acronym_expansions("Joe (mother) Smith") == "Joe (mother) Smith"
        assert (
            collapse_acronym_expansions(
                "Advanced Aerospace Threat Identification Program (AATIP)"
            )
            == "AATIP"
        )
        # Stop-words break the initials, so the name is left alone - no collapse
        # is the safe outcome.
        full = "Office of the Under Secretary of Defense for Intelligence (OUSDI)"
        assert collapse_acronym_expansions(full) == full

    def test_a_year_parenthetical_is_not_an_acronym(self):
        assert (
            collapse_acronym_expansions("roswell incident (1947)")
            == "roswell incident (1947)"
        )


class TestPlaceComponentBoilerplate:
    """A comma-component carries mandated boilerplate of its own.

    The structured branch takes the MINIMUM component score, so one component
    decides the merge - and "Walker Air Force Base" against "Kirtland Air Force
    Base" is 0.82 on the shared tail alone.
    """

    def test_distinct_air_force_bases_do_not_merge(self):
        assert (
            fuzzy_name_similarity(
                "usa, new mexico, walker air force base",
                "usa, new mexico, kirtland air force base",
            )
            < FUZZY_NAME_THRESHOLD
        )

    def test_a_state_is_not_its_namesake_district(self):
        assert (
            fuzzy_name_similarity(
                "usa, washington, seattle", "usa, district of columbia, washington"
            )
            < FUZZY_NAME_THRESHOLD
        )

    def test_distinct_streets_do_not_merge(self):
        assert (
            fuzzy_name_similarity(
                "usa, new york, manhattan, east seventy-fifth street",
                "usa, new york, manhattan, west fifty-fifth street",
            )
            < FUZZY_NAME_THRESHOLD
        )

    def test_the_same_place_still_matches_through_its_components(self):
        assert (
            fuzzy_name_similarity(
                "usa, new mexico, kirtland air force base",
                "usa, new mexico, kirtland air force base",
            )
            >= FUZZY_NAME_THRESHOLD
        )


class TestAcronymShorterThanItsExpansion:
    """An acronym need not draw one letter per word."""

    def test_two_letters_from_one_word(self):
        assert (
            collapse_acronym_expansions("Phobos 2 Hydaspis Chaos Infrared (IR) image")
            == "Phobos 2 Hydaspis Chaos IR image"
        )

    def test_three_letters_from_two_words(self):
        assert (
            collapse_acronym_expansions(
                "mitochondrial Deoxyribonucleic Acid (DNA) region"
            )
            == "mitochondrial DNA region"
        )

    def test_the_longest_expansion_wins(self):
        # Also matches on its last four words; stopping at the shortest would
        # leave a stray "Advanced" in front of the acronym.
        assert (
            collapse_acronym_expansions(
                "Advanced Aerospace Threat Identification Program (AATIP)"
            )
            == "AATIP"
        )


class TestEveryComparisonPathIsGuarded:
    """The same bug three times, so assert the class rather than the instances.

    Each was ONE line, each sat directly beside a correct guarded version, and
    each produced months of quiet misattribution:

    - the alias comparison used raw Levenshtein while the name comparison two
      lines below was guarded (118 aliases onto one event node);
    - _component_similarity compared comma components unguarded while the
      whole-name path was guarded (a Bolivian node holding fifteen Californian
      places, RAF Woodbridge inside London);
    - and the acronym collapse, added to fix the first, at first refused
      expansions of unequal word count and silently cost four true matches.

    A new comparison path that omits the guard will look exactly like these did:
    correct on whatever pairs someone happens to try, wrong across the corpus.
    So every path is enumerated here and asserted to reject a pair that differs
    on a hard token and a pair that substitutes a distinctive word.
    """

    # (label, callable) - add the new path here when adding one, or explain why
    # the guard genuinely does not apply to it.
    def _paths(self):
        return [
            ("whole name", fuzzy_name_similarity),
            ("comma component", _component_similarity),
        ]

    def test_every_path_rejects_a_hard_token_conflict(self):
        for label, compare in self._paths():
            assert compare("1947 roswell incident", "1948 roswell incident") < (
                FUZZY_NAME_THRESHOLD
            ), f"{label} accepted a differing year"

    def test_every_path_rejects_a_substituted_distinctive_word(self):
        for label, compare in self._paths():
            assert (
                compare("walker air force base", "kirtland air force base")
                < FUZZY_NAME_THRESHOLD
            ), f"{label} accepted a substituted proper noun"

    def test_every_path_still_accepts_a_genuine_variant(self):
        for label, compare in self._paths():
            assert (
                compare("kirtland air force base", "kirtland air force base")
                >= FUZZY_NAME_THRESHOLD
            ), f"{label} rejected an identical name"


class TestPlaceCountryForm:
    """Two spellings of one country must not mint two nodes.

    The components "united kingdom" and "uk" are mutual orphans, so they score
    0.0 against each other and every import creates a duplicate. This is the
    fault that produced "UK, England, Brighton" beside "United Kingdom, England,
    Brighton" in the live graph.
    """

    def test_the_long_country_form_normalises_to_the_convention(self):
        assert (
            normalise_node_name("United Kingdom, England, Oxford", "place")
            == "UK, England, Oxford"
        )
        assert (
            normalise_node_name("United States, California, Fresno", "place")
            == "USA, California, Fresno"
        )

    def test_a_bare_country_normalises_too(self):
        assert normalise_node_name("United Kingdom", "place") == "UK"

    def test_only_the_country_component_is_touched(self):
        # A place whose LATER components happen to read like a country name.
        assert normalise_node_name("France, Paris", "place") == "France, Paris"

    def test_an_organisation_keeps_its_own_wording(self):
        # "United Kingdom Ministry of Defence" is not a place hierarchy.
        assert (
            normalise_node_name("United Kingdom Ministry of Defence", "organisation")
            == "United Kingdom Ministry of Defence"
        )


class TestShortAcronymSuffix:
    """A two-character parenthetical is as often a qualifier as an acronym."""

    def test_two_letter_acronym_of_its_own_words_is_stripped(self):
        from assimilator.matching import strip_acronym_suffix

        assert (
            strip_acronym_suffix("Artificial intelligence (AI)")
            == "Artificial intelligence"
        )
        assert strip_acronym_suffix("Remote viewing (RV)") == "Remote viewing"
        assert strip_acronym_suffix("United Nations (UN)") == "United Nations"

    def test_a_two_letter_qualifier_is_not_stripped(self):
        """The reason this is an evidence test and not a length rule.

        "UFO magazine (UK)" is a country and "George Russell (AE)" a pen name;
        stripping either would make two distinct things equivalent.
        """
        from assimilator.matching import strip_acronym_suffix

        assert strip_acronym_suffix("UFO magazine (UK)") == "UFO magazine (UK)"
        assert strip_acronym_suffix("George Russell (AE)") == "George Russell (AE)"

    def test_the_pair_it_was_built_for_is_now_one_node(self):
        """'Artificial intelligence' and 'Artificial intelligence (AI)' keyed
        differently, so the matcher never saw them as the same topic."""
        from assimilator.matching import name_equivalence_key

        assert name_equivalence_key("Artificial intelligence") == name_equivalence_key(
            "Artificial intelligence (AI)"
        )
