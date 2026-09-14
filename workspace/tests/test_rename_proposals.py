"""Reviewer-proposed renames: the workbench proposes, the assimilator applies."""

import sqlite3

import yaml

from anomalica_common.digest.models import Node, NodeType
from assimilator.database import (
    init_db,
    insert_alias,
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


def test_a_non_object_proposal_is_reported_not_crashed(tmp_path, monkeypatch):
    merge = _drop(tmp_path, monkeypatch, "[]")
    read = merge.read_rename_proposals()
    assert len(read) == 1
    assert read[0]["_error"] == "expected a JSON object, got list"


def test_a_missing_directory_is_not_an_error(tmp_path, monkeypatch):
    from assimilator import merge

    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "nothing-here"))
    assert merge.read_rename_proposals() == []


def _proposal(proposal_id="p1", old="Atlant", new="Atlantis"):
    return {
        "id": proposal_id,
        "node_id": "original-node-id",
        "node_name_at_proposal": old,
        "proposed_name": new,
        "reason": "reviewed correction",
        "proposed_by": "workbench/test",
        "proposed_at": "2026-09-15T00:00:00Z",
    }


def _rename_ledger(tmp_path, old="Atlant", new="Atlantis"):
    entry = {
        "op": "rename",
        "rename_id": "rename-1",
        "at": "2026-09-15T00:01:00Z",
        "by": "workbench/test",
        "new_name": new,
        "node": {"name": old, "node_type": "topic", "prior_names": []},
    }
    path = tmp_path / "curation" / "renames.yaml"
    path.write_text(yaml.safe_dump(entry))
    return path


def _proposal_row(conn, proposal_id="p1"):
    return conn.execute(
        "SELECT id, node_id, node_name_at_proposal, proposed_name, reason, "
        "proposed_by, proposed_at, status, resolved_at, resolution_note "
        "FROM rename_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()


def test_applied_outcome_replays_without_writing_ledger_and_is_idempotent(
    tmp_path, monkeypatch
):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal())
    ledger = _rename_ledger(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="rebuilt", node_type="topic", name="Atlant"))
    conn.commit()
    before = ledger.read_bytes()

    assert merge.replay_renames(conn) == {"applied": 1, "skipped": 0}
    first = merge.replay_rename_proposals(conn)
    first_row = _proposal_row(conn)
    second = merge.replay_rename_proposals(conn)

    assert (
        first
        == second
        == {
            "applied": 1,
            "rejected": 0,
            "pending": 0,
            "lost": 0,
            "malformed": 0,
        }
    )
    assert _proposal_row(conn) == first_row
    assert first_row == (
        "p1",
        "original-node-id",
        "Atlant",
        "Atlantis",
        "reviewed correction",
        "workbench/test",
        "2026-09-15T00:00:00Z",
        "applied",
        "2026-09-15T00:01:00Z",
        None,
    )
    assert ledger.read_bytes() == before


def test_rejected_collision_outcome_is_reconstructed(tmp_path, monkeypatch):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal(old="Grey aliens", new="The Greys"))
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="grey-aliens", node_type="topic", name="Grey aliens"))
    insert_node(conn, Node(id="the-greys", node_type="topic", name="The Greys"))
    conn.commit()

    result = merge.replay_rename_proposals(conn)

    assert result["rejected"] == 1
    assert _proposal_row(conn)[7:] == (
        "rejected",
        None,
        "name already taken (reconstructed)",
    )


def test_applied_proposal_survives_contraction_via_old_alias(tmp_path, monkeypatch):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal())
    _rename_ledger(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="survivor", node_type="topic", name="Sunken city"))
    insert_alias(conn, "Atlant", "survivor")
    conn.commit()

    result = merge.replay_rename_proposals(conn)

    assert result["applied"] == 1
    assert _proposal_row(conn)[7] == "applied"


def test_lost_and_malformed_proposals_are_loud(tmp_path, monkeypatch):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal(), "{ not json")
    _rename_ledger(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    messages = []

    result = merge.replay_rename_proposals(conn, on_progress=messages.append)

    assert result == {
        "applied": 0,
        "rejected": 0,
        "pending": 0,
        "lost": 1,
        "malformed": 1,
    }
    assert _proposal_row(conn)[7] == "lost"
    assert any("ERROR rename proposal LOST" in message for message in messages)
    assert any("ERROR rename proposal MALFORMED" in message for message in messages)
