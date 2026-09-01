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
