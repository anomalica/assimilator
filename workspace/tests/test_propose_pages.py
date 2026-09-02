"""Proposal generation: the derived page_proposals table + the page-veto ledger."""

from __future__ import annotations

import sqlite3

from anomalica_common.digest.models import Claim, Node, Record
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.propose_pages import (
    propose,
    proposed_node_ids,
    read_vetoes,
    replay_vetoes,
    un_veto,
    veto_pages,
    vetoed_node_ids,
)


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("r1", "r2", "r3"):
        insert_record(conn, Record(id=rid, title=rid))
    # Two page-worthy people (9 claims from 3 works, second work carrying 3, every
    # claim about them) and one thin node.
    insert_node(conn, Node(id="alpha", node_type="person", name="Alpha Person"))
    insert_node(conn, Node(id="beta", node_type="person", name="Beta Person"))
    insert_node(conn, Node(id="thin", node_type="person", name="Thin Person"))
    spread = ["r1"] * 4 + ["r2"] * 3 + ["r3"] * 2
    for nid, name in (("alpha", "Alpha Person"), ("beta", "Beta Person")):
        for i, rid in enumerate(spread):
            insert_claim(
                conn,
                Claim(
                    id=f"{nid}-c{i}",
                    content=f"{name} said {i}",
                    claim_type="testimony",
                    record_id=rid,
                    node_references=[nid],
                ),
            )
    insert_claim(
        conn,
        Claim(
            id="t1",
            content="Thin Person said x",
            claim_type="testimony",
            record_id="r1",
            node_references=["thin"],
        ),
    )
    conn.commit()
    return conn


def test_propose_writes_derived_table():
    conn = _graph()
    rows = propose(conn, computed_at="2026-06-28T00:00:00Z")
    assert {r["node_id"] for r in rows} == {"alpha", "beta"}
    stored = conn.execute(
        "SELECT node_id, status, tier FROM page_proposals ORDER BY node_id"
    ).fetchall()
    assert stored == [
        ("alpha", "proposed", "page-worthy"),
        ("beta", "proposed", "page-worthy"),
    ]


def test_propose_is_idempotent_and_replaces():
    conn = _graph()
    propose(conn)
    propose(conn)  # second pass must not duplicate the derived rows
    assert conn.execute("SELECT COUNT(*) FROM page_proposals").fetchone()[0] == 2


def test_veto_excludes_from_proposals(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    veto_pages(conn, ["beta"], reason="a mention, not a subject", veto_id="v1")
    assert vetoed_node_ids(conn) == {"beta"}
    rows = propose(conn)
    assert {r["node_id"] for r in rows} == {"alpha"}
    assert proposed_node_ids(conn) == ["alpha"]


def test_veto_is_durable_in_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    veto_pages(conn, ["beta"], reason=None, veto_id="v1", created_by="op@x")
    entries = read_vetoes()
    assert entries[0]["op"] == "veto"
    assert entries[0]["nodes"][0]["name"] == "Beta Person"


def test_replay_repopulates_veto_table_by_natural_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    veto_pages(conn, ["beta"], reason="x", veto_id="v1")
    # Simulate a rebuild: the derived table is wiped, the ledger survives.
    conn.execute("DELETE FROM page_vetoes")
    assert vetoed_node_ids(conn) == set()
    replay_vetoes(conn)
    assert vetoed_node_ids(conn) == {"beta"}


def test_unveto_restores_proposal(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    veto_pages(conn, ["beta"], reason="x", veto_id="v1")
    un_veto(conn, "v1")
    assert vetoed_node_ids(conn) == set()
    assert {r["node_id"] for r in propose(conn)} == {"alpha", "beta"}
