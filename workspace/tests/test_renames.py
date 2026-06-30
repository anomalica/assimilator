"""Durable node renames: the curation-ledger rename op (ADR 0038)."""

from __future__ import annotations

import sqlite3

from anomalica_common.digest.models import Node, Record
from assimilator.database import init_db, insert_node, insert_record
from assimilator.merge import (
    read_renames,
    rename_node,
    replay_renames,
)


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_node(
        conn,
        Node(id="n1", node_type="organisation", name="United States Navy"),
    )
    conn.commit()
    return conn


def _name(conn, nid):
    return conn.execute("SELECT name FROM nodes WHERE id = ?", (nid,)).fetchone()[0]


def _aliases(conn, nid):
    return {
        a
        for (a,) in conn.execute("SELECT alias FROM aliases WHERE node_id = ?", (nid,))
    }


def test_rename_applies_and_keeps_old_as_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    rename_node(conn, "n1", "United States Navy (USN)", "rn1")
    assert _name(conn, "n1") == "United States Navy (USN)"
    assert "United States Navy" in _aliases(conn, "n1")


def test_rename_is_recorded_in_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    conn = _graph()
    rename_node(conn, "n1", "United States Navy (USN)", "rn1", created_by="op@x")
    entries = read_renames()
    assert entries[0]["op"] == "rename"
    assert entries[0]["new_name"] == "United States Navy (USN)"
    # Keyed on the PRE-rename name, so a fresh import resolves it.
    assert entries[0]["node"]["name"] == "United States Navy"


def test_replay_resolves_by_natural_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    # Record the rename on one graph...
    rename_node(_graph(), "n1", "United States Navy (USN)", "rn1")
    # ...then replay it onto a FRESH import (node has the OLD name, new id).
    fresh = sqlite3.connect(":memory:")
    init_db(fresh)
    insert_record(fresh, Record(id="r1", title="R1"))
    insert_node(
        fresh, Node(id="fresh-id", node_type="organisation", name="United States Navy")
    )
    fresh.commit()
    result = replay_renames(fresh)
    assert result["applied"] == 1
    assert _name(fresh, "fresh-id") == "United States Navy (USN)"
    assert "United States Navy" in _aliases(fresh, "fresh-id")


def test_replay_skips_node_no_longer_in_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    rename_node(_graph(), "n1", "United States Navy (USN)", "rn1")
    empty = sqlite3.connect(":memory:")
    init_db(empty)
    result = replay_renames(empty)
    assert result == {"applied": 0, "skipped": 1}
