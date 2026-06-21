"""Propose-merges: deterministic candidate clusters (name-equiv + fuzzy)."""

from __future__ import annotations

import sqlite3

from assimilator import propose_merges as pm
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import Claim, Node, Record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    # name-equiv within type (case variant), same node_type
    insert_node(conn, Node(id="rv1", node_type="topic", name="Remote Viewing"))
    insert_node(conn, Node(id="rv2", node_type="topic", name="remote viewing"))
    # fuzzy within type
    insert_node(conn, Node(id="kd1", node_type="person", name="Kevin Day"))
    insert_node(conn, Node(id="kd2", node_type="person", name="Kevin Day."))
    # unrelated
    insert_node(conn, Node(id="x", node_type="person", name="David Fravor"))
    # cross-type same name must NOT be proposed (within-type only)
    insert_node(
        conn, Node(id="sap1", node_type="concept", name="Special Access Programs")
    )
    insert_node(
        conn, Node(id="sap2", node_type="matter", name="Special Access Programs")
    )
    for nid in ("rv1", "rv2", "kd1", "kd2", "x", "sap1", "sap2"):
        insert_claim(
            conn,
            Claim(
                id=f"c-{nid}",
                content="x",
                claim_type="observation",
                record_id="r1",
                node_references=[nid],
            ),
        )
    conn.commit()
    return conn


def test_name_equiv_groups_case_variants_within_type():
    conn = _graph()
    cands = pm.propose(conn)
    ne = [c for c in cands if c["reason"] == "name-equiv"]
    rv = [c for c in ne if set(c["node_ids"]) == {"rv1", "rv2"}]
    assert rv and rv[0]["score"] == 0.95 and rv[0]["node_type"] == "topic"


def test_fuzzy_catches_near_names():
    conn = _graph()
    cands = pm.propose(conn)
    pairs = {frozenset(c["node_ids"]) for c in cands}
    assert frozenset({"kd1", "kd2"}) in pairs  # Kevin Day / Kevin Day.


def test_cross_type_same_name_flagged():
    conn = _graph()
    cands = pm.propose(conn)
    sap = [c for c in cands if set(c["node_ids"]) == {"sap1", "sap2"}]
    # same-name cross-type IS surfaced (high signal) but flagged for review
    assert sap and sap[0]["reason"] == "name-equiv-crosstype"


def test_unrelated_not_proposed():
    conn = _graph()
    cands = pm.propose(conn)
    assert all("x" not in c["node_ids"] for c in cands)


def test_output_shape():
    conn = _graph()
    for c in pm.propose(conn):
        assert set(c) == {
            "node_ids",
            "suggested_canonical",
            "score",
            "node_type",
            "reason",
        }
        assert len(c["node_ids"]) >= 2
