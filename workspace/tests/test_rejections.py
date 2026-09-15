"""Durable node-rejection replay diagnostics."""

import sqlite3

import yaml

from anomalica_common.digest.models import Node
from assimilator.database import init_db, insert_node
from assimilator.merge import replay_rejections


def _graph(*nodes, node_type="place"):
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for node_id, name in nodes:
        insert_node(conn, Node(id=node_id, node_type=node_type, name=name))
    conn.commit()
    return conn


def _ledger(tmp_path, monkeypatch, nodes, node_type="place"):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    entry = {
        "op": "reject",
        "rejection_id": "reject-atlant",
        "at": "2026-09-15T00:00:00Z",
        "by": "test",
        "reason": "distinct places",
        "nodes": [
            {"name": name, "node_type": node_type, "prior_names": []} for name in nodes
        ],
    }
    (tmp_path / "rejections.yaml").write_text(yaml.safe_dump(entry))


def test_replay_materialises_similarly_named_distinct_nodes(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, ["Atlant", "Atlantis"])
    conn = _graph(("atlant", "Atlant"), ("atlantis", "Atlantis"))

    result = replay_rejections(conn)

    assert result == {"applied": 1, "absorbed": 0, "lost": 0}
    assert conn.execute(
        "SELECT node_id FROM node_rejections ORDER BY node_id"
    ).fetchall() == [("atlant",), ("atlantis",)]


def test_replay_materialises_exact_single_token_person_names(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, ["Atlant", "Atlantis"], node_type="person")
    conn = _graph(("atlant", "Atlant"), ("atlantis", "Atlantis"), node_type="person")

    result = replay_rejections(conn)

    assert result == {"applied": 1, "absorbed": 0, "lost": 0}
    assert conn.execute(
        "SELECT node_id FROM node_rejections ORDER BY node_id"
    ).fetchall() == [("atlant",), ("atlantis",)]


def test_replay_clears_stale_rows_when_a_rejection_contracts(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, ["Atlant", "Atlantis"])
    conn = _graph(("atlant", "Atlant"), ("atlantis", "Atlantis"))
    assert replay_rejections(conn)["applied"] == 1
    conn.execute("DELETE FROM node_rejections WHERE node_id = 'atlant'")
    conn.execute("DELETE FROM nodes WHERE id = 'atlant'")
    conn.commit()

    result = replay_rejections(conn)

    assert result == {"applied": 0, "absorbed": 1, "lost": 0}
    assert conn.execute("SELECT * FROM node_rejections").fetchall() == []


def test_replay_reports_total_rejection_loss_loudly(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, ["Atlant", "Atlantis"])
    messages = []

    result = replay_rejections(_graph(), on_progress=messages.append)

    assert result == {"applied": 0, "absorbed": 0, "lost": 1}
    assert any("ERROR" in message and "LOST" in message for message in messages)
