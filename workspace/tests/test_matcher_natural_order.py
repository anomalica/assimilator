"""The person-name-flip GATE: natural-order person matching must not false-merge
on forename collisions, and must still resolve the retained "Surname, First"
alias (which carries the #23 comma-precision). Validates the matcher is ready for
the Last,First -> natural-order convention flip (node-types.md, 2026-06-29)."""

from __future__ import annotations

import sqlite3

from anomalica_common.digest.models import Node, Record
from assimilator.database import init_db, insert_node, insert_record
from assimilator.matching import match_node


def _alias(conn, alias, node_id):
    conn.execute(
        "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)", (alias, node_id)
    )


def _graph():
    """Two distinct people sharing the forename 'David', stored natural-order
    canonical with the old last-first form retained as an alias (the spec)."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_node(conn, Node(id="fravor", node_type="person", name="David Fravor"))
    _alias(conn, "Fravor, David", "fravor")
    insert_node(
        conn, Node(id="grusch", node_type="person", name="David Charles Grusch")
    )
    _alias(conn, "Grusch, David Charles", "grusch")
    _alias(conn, "David Grusch", "grusch")
    conn.commit()
    return conn


def test_forename_collision_resolves_to_correct_person():
    conn = _graph()
    # Same forename, different surname - must each resolve to their OWN node.
    assert match_node(conn, "David Fravor", "person")[0] == "fravor"
    assert match_node(conn, "David Grusch", "person")[0] == "grusch"


def test_shared_forename_new_surname_is_not_a_false_merge():
    conn = _graph()
    # A genuinely new "David <X>" must NOT be matched onto either existing David.
    assert match_node(conn, "David Spergel", "person") is None
    assert match_node(conn, "David Mellon", "person") is None


def test_last_first_input_resolves_via_retained_alias():
    conn = _graph()
    # Old last-first digest input resolves to the natural-order node (order
    # tolerance kept permanently per the spec).
    assert match_node(conn, "Fravor, David", "person")[0] == "fravor"
    assert match_node(conn, "Grusch, David Charles", "person")[0] == "grusch"


def test_middle_name_variant_matches_same_person():
    conn = _graph()
    # "David Grusch" (no middle) must resolve to "David Charles Grusch".
    assert match_node(conn, "David Grusch", "person")[0] == "grusch"


def test_two_people_one_surname_do_not_merge():
    """The classic #23 case in natural order: two people sharing a SURNAME must
    stay distinct (e.g. father/son, or unrelated)."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_node(conn, Node(id="kevin", node_type="person", name="Kevin Day"))
    _alias(conn, "Day, Kevin", "kevin")
    conn.commit()
    # A different "Day" must not collapse onto Kevin Day.
    assert match_node(conn, "John Day", "person") is None
