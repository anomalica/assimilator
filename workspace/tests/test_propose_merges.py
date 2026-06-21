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
    members = {frozenset(c["node_ids"]) for c in cands}
    assert any({"kd1", "kd2"} <= m for m in members)  # Kevin Day / Kevin Day.


def test_structured_name_false_positives_blocked():
    # The #23 class: distinct towns / sections sharing a structured prefix must
    # NOT be linked, and so must NOT cluster.
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_record(conn, Record(id="r2", title="R2"))
    towns = [
        ("ros", "USA, New Mexico, Roswell"),
        ("azt", "USA, New Mexico, Aztec"),
        ("dul", "USA, New Mexico, Dulce"),
    ]
    secs = [("s32", "NDAA Section 1632"), ("s73", "NDAA Section 1673")]
    for nid, name in towns:
        insert_node(conn, Node(id=nid, node_type="place", name=name))
    for nid, name in secs:
        insert_node(conn, Node(id=nid, node_type="matter", name=name))
    for nid, _ in towns + secs:
        for r in ("r1", "r2"):
            insert_claim(
                conn,
                Claim(
                    id=f"c-{nid}-{r}",
                    content="x",
                    claim_type="observation",
                    record_id=r,
                    node_references=[nid],
                ),
            )
    conn.commit()
    cands = pm.propose(conn)
    # no candidate may group two different towns or the two sections
    for c in cands:
        ids = set(c["node_ids"])
        assert not (
            {"ros", "azt"} <= ids or {"ros", "dul"} <= ids or {"azt", "dul"} <= ids
        )
        assert {"s32", "s73"} != ids


def test_connected_pairs_form_one_cluster():
    # A-B and B-C similar -> ONE cluster {A,B,C}, not two overlapping pairs.
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_record(conn, Record(id="r2", title="R2"))
    for nid, name in (
        ("a", "Project Aquarius"),
        ("b", "Project Aquarius "),
        ("c", "Project Aquarius."),
    ):
        insert_node(conn, Node(id=nid, node_type="project", name=name))
        for r in ("r1", "r2"):
            insert_claim(
                conn,
                Claim(
                    id=f"c-{nid}-{r}",
                    content="x",
                    claim_type="observation",
                    record_id=r,
                    node_references=[nid],
                ),
            )
    conn.commit()
    clusters = [
        set(c["node_ids"])
        for c in pm.propose(conn)
        if {"a", "b", "c"} & set(c["node_ids"])
    ]
    assert clusters == [{"a", "b", "c"}]  # one cluster, all three


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


def test_rejected_cluster_excluded_and_reinstated(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    from assimilator import merge

    conn = _graph()
    assert any(set(c["node_ids"]) == {"rv1", "rv2"} for c in pm.propose(conn))
    # Reject the pair -> excluded from future candidates.
    merge.reject_nodes(conn, ["rv1", "rv2"], "different", "rej1")
    assert not any(set(c["node_ids"]) == {"rv1", "rv2"} for c in pm.propose(conn))
    # Un-reject -> it reappears (reversible).
    merge.un_reject(conn, "rej1")
    assert any(set(c["node_ids"]) == {"rv1", "rv2"} for c in pm.propose(conn))


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
