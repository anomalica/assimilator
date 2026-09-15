"""Reviewer-proposed renames: the workbench proposes, the assimilator applies."""

import sqlite3

import yaml

from anomalica_common.digest.models import Node, NodeType
from assimilator.database import (
    init_db,
    insert_alias,
    insert_node,
)


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="N", node_type=NodeType.topic, name="alien abduction"))
    return conn


def test_old_proposal_table_migrates_to_canonical_link_schema():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE rename_proposals (id TEXT PRIMARY KEY, node_id TEXT NOT NULL, "
        "node_name_at_proposal TEXT NOT NULL, proposed_name TEXT NOT NULL, reason TEXT, "
        "proposed_by TEXT, proposed_at TEXT NOT NULL, status TEXT NOT NULL, "
        "resolved_at TEXT, resolution_note TEXT)"
    )
    conn.execute(
        "INSERT INTO rename_proposals VALUES "
        "('p1','audit','Alpha','Beta',NULL,NULL,'2026-09-15T00:00:00Z','lost',NULL,NULL)"
    )

    init_db(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(rename_proposals)")}
    assert {
        "proposal_operation_id",
        "resolution_phase",
        "resolution_operation_id",
        "compensation_operation_id",
        "replay_disposition_id",
    } <= columns
    assert conn.execute("SELECT COUNT(*) FROM rename_proposals").fetchone()[0] == 0


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
            "reason": None,
            "proposed_by": None,
            "proposed_at": "2026-09-15T00:00:00Z",
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
    from assimilator.rename_ledger import build_rename_event

    entry = build_rename_event(
        at="2026-09-15T00:01:00Z",
        by="workbench/test",
        new_name=new,
        node={"name": old, "node_type": "topic", "prior_names": []},
        proposal_id="rename-proposal:p1",
    )
    path = tmp_path / "curation" / "renames.yaml"
    path.write_text(yaml.safe_dump(entry, sort_keys=False))
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


def test_exceptional_proposal_outcome_materialises_from_validated_disposition(
    tmp_path, monkeypatch
):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal())
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    disposition_id = "sha256:" + "d" * 64
    report = {
        "outcomes": [
            {
                "id": disposition_id,
                "source_phase": "rename",
                "operation_id": "rename-proposal:p1",
                "outcome": "contraction_drop",
                "evidence": {
                    "absent_identities": [],
                    "absent_support_locators": [],
                },
                "classified_at": "2026-09-15T01:00:00Z",
            }
        ]
    }

    result = merge.replay_rename_proposals(conn, disposition_report=report)

    assert result["applied"] == 1
    assert conn.execute(
        "SELECT status, replay_disposition_id FROM rename_proposals WHERE id='p1'"
    ).fetchone() == ("contraction_drop", disposition_id)


def test_rejected_collision_outcome_is_reconstructed(tmp_path, monkeypatch):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal(old="Grey aliens", new="The Greys"))
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="grey-aliens", node_type="topic", name="Grey aliens"))
    insert_node(conn, Node(id="the-greys", node_type="topic", name="The Greys"))
    conn.commit()

    result = merge.replay_rename_proposals(conn)

    assert result["pending"] == 1
    assert _proposal_row(conn)[7:] == (
        "pending",
        None,
        None,
    )


def test_rejected_outcome_requires_explicit_reject_event(tmp_path, monkeypatch):
    from assimilator import merge
    from assimilator.rename_ledger import build_reject_proposal_event

    _drop(tmp_path, monkeypatch, _proposal())
    event = build_reject_proposal_event(
        proposal_id="rename-proposal:p1",
        at="2026-09-15T00:01:00Z",
        by="operator",
        reason="reviewed rejection",
    )
    (tmp_path / "curation" / "renames.yaml").write_text(
        yaml.safe_dump(event, sort_keys=False)
    )
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="source", node_type="topic", name="Atlant"))

    result = merge.replay_rename_proposals(conn)

    assert result["rejected"] == 1
    assert _proposal_row(conn)[7:] == (
        "rejected",
        "2026-09-15T00:01:00Z",
        "reviewed rejection",
    )


def test_merged_outcome_requires_explicit_confirmed_merge_link(tmp_path, monkeypatch):
    from assimilator import merge

    _drop(tmp_path, monkeypatch, _proposal(old="Atlant", new="Atlantis"))
    merge_entry = {
        "op": "merge",
        "merge_id": "merge-1",
        "at": "2026-09-15T00:01:00Z",
        "by": "operator",
        "confirmation": {
            "by": "operator",
            "at": "2026-09-15T00:01:00Z",
            "via": "workbench-queue",
        },
        "canonical_name": "Atlantis",
        "survivor": {"name": "Atlantis", "node_type": "topic", "prior_names": []},
        "victims": [{"name": "Atlant", "node_type": "topic", "prior_names": []}],
        "proposal_ids": ["rename-proposal:p1"],
    }
    (tmp_path / "curation" / "merges.yaml").write_text(
        yaml.safe_dump(merge_entry, sort_keys=False)
    )
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="survivor", node_type="topic", name="Atlantis"))
    insert_alias(conn, "Atlant", "survivor")

    result = merge.replay_rename_proposals(conn)

    assert result["applied"] == 1
    assert _proposal_row(conn)[7] == "merged"


def test_apply_renames_cli_parses_v2_and_resolves_its_natural_identity(
    tmp_path, monkeypatch
):
    import json

    from click.testing import CliRunner

    from assimilator.cli import main

    curation = tmp_path / "curation"
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(curation))
    directory = curation / "rename-proposals"
    directory.mkdir(parents=True)
    (directory / "p1.json").write_text(
        json.dumps(
            {
                "schema": "anomalica/rename-proposal/2",
                "id": "p1",
                "node": {
                    "name": "Atlant",
                    "node_type": "topic",
                    "prior_names": [],
                },
                "proposed_name": "Atlantis",
                "reason": None,
                "proposed_by": "operator",
                "proposed_at": "2026-09-15T00:00:00Z",
                "source": {"node_id": "stale-audit-id"},
            }
        )
    )
    db = tmp_path / "graph.db"
    conn = sqlite3.connect(db)
    init_db(conn)
    insert_node(conn, Node(id="current", node_type="topic", name="Current name"))
    insert_alias(conn, "Atlant", "current")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["--db", str(db), "apply-renames", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "RENAME  'Current name' -> 'Atlantis'" in result.output


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

    assert result["lost"] == 1
    assert _proposal_row(conn)[7] == "unresolved_drift"


def test_applied_proposal_with_absent_node_and_malformed_proposal_are_distinguished(
    tmp_path, monkeypatch
):
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
    assert _proposal_row(conn)[7] == "unresolved_drift"
    assert any("ERROR rename proposal MALFORMED" in message for message in messages)
