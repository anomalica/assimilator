"""Reviewer-proposed renames: the workbench proposes, the assimilator applies."""

import sqlite3

from anomalica_common.digest.models import Node, NodeType
from assimilator.database import (
    init_db,
    insert_node,
    pending_renames,
    propose_rename,
    resolve_rename,
)


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="N", node_type=NodeType.topic, name="alien abduction"))
    return conn


def test_a_proposal_is_recorded_and_readable():
    conn = _graph()
    pid = propose_rename(
        conn, "N", "alien abduction", "Alien abduction", "sentence case", "workbench"
    )
    pending = pending_renames(conn)
    assert len(pending) == 1
    assert pending[0]["id"] == pid
    assert pending[0]["proposed_name"] == "Alien abduction"
    assert pending[0]["node_name_at_proposal"] == "alien abduction"


def test_a_resolved_proposal_leaves_the_queue():
    conn = _graph()
    pid = propose_rename(conn, "N", "alien abduction", "Alien abduction")
    resolve_rename(conn, pid, "applied")
    assert pending_renames(conn) == []


def test_the_name_at_proposal_is_kept_because_ids_are_regenerated():
    """A rebuild re-imports the graph and mints new ids, so the id a reviewer saw
    may not resolve later. The name is the fallback identity; without it a
    proposal made before a rebuild would always be lost."""
    conn = _graph()
    propose_rename(conn, "OLD-ID", "alien abduction", "Alien abduction")
    stored = pending_renames(conn)[0]
    assert stored["node_id"] == "OLD-ID"
    assert stored["node_name_at_proposal"] == "alien abduction"


def test_status_is_constrained():
    """A typo in a status must fail loudly, not create a proposal nothing reads."""
    conn = _graph()
    pid = propose_rename(conn, "N", "alien abduction", "Alien abduction")
    try:
        conn.execute(
            "UPDATE rename_proposals SET status = 'aplied' WHERE id = ?", (pid,)
        )
    except sqlite3.IntegrityError:
        return
    raise AssertionError("an invalid status was accepted")


def _drop(tmp_path, monkeypatch, *docs):
    import json

    from assimilator import merge

    directory = tmp_path / "curation" / "rename-proposals"
    directory.mkdir(parents=True)
    for i, doc in enumerate(docs):
        text = doc if isinstance(doc, str) else json.dumps(doc)
        (directory / f"{i:02d}.json").write_text(text)
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    return merge


def test_proposals_arrive_as_files_not_rows(tmp_path, monkeypatch):
    """The workbench declined a writable handle for one table, and was right to:
    its read-only connection is what stops it corrupting the graph, and one
    writable table is a precedent where there is currently a boundary."""
    merge = _drop(
        tmp_path,
        monkeypatch,
        {
            "id": "p1",
            "node_id": "N",
            "node_name_at_proposal": "alien abduction",
            "proposed_name": "Alien abduction",
        },
    )
    read = merge.read_rename_proposals()
    assert len(read) == 1
    assert read[0]["proposed_name"] == "Alien abduction"


def test_an_unreadable_proposal_is_reported_not_skipped(tmp_path, monkeypatch):
    """A proposal that vanishes silently is indistinguishable from one nobody
    made, and the reviewer is owed an answer either way."""
    merge = _drop(tmp_path, monkeypatch, "{ not json")
    read = merge.read_rename_proposals()
    assert len(read) == 1
    assert "JSONDecodeError" in read[0]["_error"]


def test_a_missing_directory_is_not_an_error(tmp_path, monkeypatch):
    from assimilator import merge

    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "nothing-here"))
    assert merge.read_rename_proposals() == []
