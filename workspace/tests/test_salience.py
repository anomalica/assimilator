"""The derived half of salience: a baseline, and a check on the extractor."""

import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import salience
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    for nid, name in (
        ("hub", "UFO"),
        ("subj", "Whitley Strieber"),
        ("place", "Sydney"),
    ):
        insert_node(conn, Node(id=nid, name=name, node_type=NodeType.topic))
    for i in range(6):
        insert_claim(
            conn,
            Claim(
                id=f"c{i}",
                content=f"claim {i}",
                claim_type="testimony",
                record_id="r1",
                node_references=["hub", "subj"] if i else ["subj"],
            ),
        )
    conn.commit()
    return conn


def test_a_claim_referencing_one_node_is_about_that_node():
    conn = _graph()
    assert salience.sole_reference_edges(conn) == 1
    stats = {s.name: s for s in salience.node_edge_stats(conn)}
    assert stats["Whitley Strieber"].edges == 6
    assert stats["Whitley Strieber"].sole_reference == 1
    assert stats["UFO"].edges == 5 and stats["UFO"].sole_reference == 0


def test_null_salience_is_reported_as_not_assessed_never_as_a_value():
    conn = _graph()
    c = salience.coverage(conn)
    assert c["null"] == c["total"] == 11
    assert all(c[v] == 0 for v in salience.VALUES)
    conn.execute("UPDATE claim_node_refs SET salience='subject' WHERE node_id='subj'")
    conn.commit()
    c = salience.coverage(conn)
    assert c["subject"] == 6 and c["null"] == 5 and c["total"] == 11


def test_the_check_reports_a_node_the_extractor_calls_the_subject_of_everything():
    conn = _graph()
    assert salience.implausible_subjects(conn) == []  # nothing assessed yet
    conn.execute("UPDATE claim_node_refs SET salience='subject'")
    conn.commit()

    found = {r["name"]: r for r in salience.implausible_subjects(conn, share=0.4)}

    assert (
        "Whitley Strieber" in found and found["Whitley Strieber"]["subject_edges"] == 6
    )
    assert "UFO" in found


def test_a_value_outside_the_four_is_refused():
    conn = _graph()
    try:
        conn.execute("UPDATE claim_node_refs SET salience='about'")
    except sqlite3.IntegrityError:
        return
    raise AssertionError("the column accepted a value outside the four")
