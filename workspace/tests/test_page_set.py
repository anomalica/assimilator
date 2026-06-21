"""The page-set floor: which entities earn a page."""

from __future__ import annotations

import sqlite3

from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.page_set import page_set_node_ids
from anomalica_common.digest.models import Claim, Node, Record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("r1", "r2"):
        insert_record(conn, Record(id=rid, title=rid))
    for nm, nid in (("Strong", "strong"), ("Thin", "thin"), ("OneSource", "onesrc")):
        insert_node(conn, Node(id=nid, node_type="person", name=nm))
    # strong: 2 claims from 2 distinct records -> passes 2/2
    insert_claim(
        conn,
        Claim(
            id="s1",
            content="a",
            claim_type="testimony",
            record_id="r1",
            node_references=["strong"],
        ),
    )
    insert_claim(
        conn,
        Claim(
            id="s2",
            content="b",
            claim_type="testimony",
            record_id="r2",
            node_references=["strong"],
        ),
    )
    # thin: 1 claim -> fails
    insert_claim(
        conn,
        Claim(
            id="t1",
            content="c",
            claim_type="testimony",
            record_id="r1",
            node_references=["thin"],
        ),
    )
    # onesrc: 2 claims but both from r1 -> fails the >=2 sources gate
    insert_claim(
        conn,
        Claim(
            id="o1",
            content="d",
            claim_type="testimony",
            record_id="r1",
            node_references=["onesrc"],
        ),
    )
    insert_claim(
        conn,
        Claim(
            id="o2",
            content="e",
            claim_type="testimony",
            record_id="r1",
            node_references=["onesrc"],
        ),
    )
    conn.commit()
    return conn


def test_floor_passes_two_claims_two_sources():
    conn = _graph()
    assert page_set_node_ids(conn) == ["strong"]


def test_floor_rejects_thin_and_single_source():
    conn = _graph()
    passed = set(page_set_node_ids(conn))
    assert "thin" not in passed  # one claim
    assert "onesrc" not in passed  # two claims, one source


def test_floor_tunable():
    conn = _graph()
    # Lowering the source gate to 1 admits the single-source node.
    assert set(page_set_node_ids(conn, min_claims=2, min_sources=1)) == {
        "strong",
        "onesrc",
    }


def test_retired_nodes_excluded():
    conn = _graph()
    conn.execute("UPDATE nodes SET retired_at = '2026-01-01' WHERE id = 'strong'")
    assert page_set_node_ids(conn) == []
