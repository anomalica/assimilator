from __future__ import annotations

import sqlite3

import pytest
import yaml
from click.testing import CliRunner

from anomalica_common.digest.models import Claim, Node, Record
from assimilator import claim_ref_status_ledger as ledger
from assimilator.cli import main
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph(
    *,
    record_id: str = "record-source-id",
    claim_id: str = "claim-source-id",
    node_id: str = "node-source-id",
    record_hash: str = "sha256:kenneth-arnold-record",
    with_ref: bool = True,
) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(
        conn,
        Record(
            id=record_id, title="Kenneth Arnold interview", content_hash=record_hash
        ),
    )
    insert_node(conn, Node(id=node_id, name="Kenneth Arnold", node_type="person"))
    insert_claim(
        conn,
        Claim(
            id=claim_id,
            content="Kenneth Arnold reported seeing nine objects near Mount Rainier.",
            original_excerpt="I saw a chain of nine peculiar-looking aircraft.",
            claim_type="testimony",
            record_id=record_id,
            location_in_record="p. 3",
            node_references=[node_id] if with_ref else [],
            ref_roles={node_id: "subject"} if with_ref else {},
        ),
    )
    conn.execute(
        "INSERT INTO aliases (alias, node_id) VALUES (?, ?)",
        ("Arnold, Kenneth", node_id),
    )
    conn.commit()
    return conn


def _set_status(
    conn: sqlite3.Connection,
    status: str = "verified",
    *,
    claim_id: str = "claim-source-id",
    node_id: str = "node-source-id",
    set_at: str = "1947-06-25T12:34:56Z",
    reason: str = "read against transcript",
) -> None:
    conn.execute(
        "INSERT INTO claim_ref_status "
        "(claim_id, node_id, status, reason, set_at, set_by) "
        "VALUES (?, ?, ?, ?, ?, 'curator@example.test') "
        "ON CONFLICT(claim_id, node_id) DO UPDATE SET "
        "status=excluded.status, reason=excluded.reason, "
        "set_at=excluded.set_at, set_by=excluded.set_by",
        (claim_id, node_id, status, reason, set_at),
    )
    conn.commit()


def _document(path):
    return yaml.safe_load(path.read_text())


def test_roundtrip_is_deterministic_idempotent_and_preserves_fields(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "claim-ref-status.yaml"

    first = ledger.export_claim_ref_status(source, path)
    first_bytes = path.read_bytes()
    second = ledger.export_claim_ref_status(source, path)

    assert first["exported"] == 1
    assert second["exported"] == 0
    assert path.read_bytes() == first_bytes
    document = _document(path)
    assert document["schema"] == ledger.SCHEMA
    event = document["entries"][0]
    assert event["op"] == "set"
    assert event["source"] == {
        "claim_id": "claim-source-id",
        "node_id": "node-source-id",
    }
    assert set(event["source"]) == {"claim_id", "node_id"}

    rebuilt = _graph(
        record_id="rebuilt-record-id",
        claim_id="rebuilt-claim-id",
        node_id="rebuilt-node-id",
    )
    result = ledger.replay_claim_ref_status(rebuilt, path)
    assert result["applied"] == 1
    assert ledger.replay_claim_ref_status(rebuilt, path)["applied"] == 1
    assert rebuilt.execute(
        "SELECT status, reason, set_at, set_by FROM claim_ref_status"
    ).fetchone() == (
        "verified",
        "read against transcript",
        "1947-06-25T12:34:56Z",
        "curator@example.test",
    )


def test_export_appends_status_change_without_altering_prior_event_bytes(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "claim-ref-status.yaml"
    ledger.export_claim_ref_status(source, path)
    before = path.read_bytes()
    first_event = _document(path)["entries"][0]

    _set_status(
        source,
        "suspect",
        set_at="1947-06-25T13:00:00Z",
        reason="the claim concerns a different witness",
    )
    result = ledger.export_claim_ref_status(source, path)

    after = path.read_bytes()
    events = _document(path)["entries"]
    assert result["exported"] == 1
    assert after.startswith(before)
    assert events[0] == first_event
    assert events[0]["id"] != events[1]["id"]
    assert events[1]["status"] == "suspect"


def test_latest_active_set_wins(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    _set_status(
        source,
        "suspect",
        set_at="1947-06-25T13:00:00Z",
        reason="later review",
    )
    ledger.export_claim_ref_status(source, path)
    rebuilt = _graph(record_id="new-record", claim_id="new-claim", node_id="new-node")

    ledger.replay_claim_ref_status(rebuilt, path)

    assert rebuilt.execute(
        "SELECT status, reason, set_at FROM claim_ref_status"
    ).fetchone() == ("suspect", "later review", "1947-06-25T13:00:00Z")


def test_unset_disables_exact_set_and_removes_its_materialisation(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    set_event = _document(path)["entries"][0]

    unset = ledger.append_unset_entry(
        set_event["id"],
        "1947-06-25T14:00:00Z",
        "curator@example.test",
        "review withdrawn",
        path,
    )
    assert (
        ledger.append_unset_entry(
            set_event["id"],
            "1947-06-25T14:00:00Z",
            "curator@example.test",
            "review withdrawn",
            path,
        )
        == unset
    )
    assert len(_document(path)["entries"]) == 2

    rebuilt = _graph(record_id="new-record", claim_id="new-claim", node_id="new-node")
    _set_status(rebuilt, claim_id="new-claim", node_id="new-node")
    result = ledger.replay_claim_ref_status(rebuilt, path)

    assert result["unset"] == 1
    assert rebuilt.execute("SELECT COUNT(*) FROM claim_ref_status").fetchone()[0] == 0


def test_unset_must_reference_an_exact_set_event(tmp_path):
    path = tmp_path / "ledger.yaml"
    path.write_text(f"schema: {ledger.SCHEMA}\nentries: []\n")

    with pytest.raises(ledger.ClaimRefStatusReplayError, match="not a set event"):
        ledger.append_unset_entry(
            "sha256:absent", "1947-06-25T14:00:00Z", "curator", path=path
        )


def test_contradictory_active_sets_at_same_time_fail_closed(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    _set_status(source, "suspect", reason="contradictory simultaneous review")
    ledger.export_claim_ref_status(source, path)
    rebuilt = _graph(record_id="new-record", claim_id="new-claim", node_id="new-node")
    rebuilt.execute("DELETE FROM claim_ref_status")
    rebuilt.commit()

    with pytest.raises(ledger.ClaimRefStatusReplayError, match="contradictory"):
        ledger.replay_claim_ref_status(rebuilt, path)

    assert rebuilt.execute("SELECT COUNT(*) FROM claim_ref_status").fetchone()[0] == 0


def test_export_excludes_and_reports_status_rows_without_refs(tmp_path):
    conn = _graph()
    _set_status(conn)
    conn.execute("DELETE FROM claim_node_refs")
    conn.commit()

    result = ledger.export_claim_ref_status(conn, tmp_path / "ledger.yaml")

    assert result["exported"] == 0
    assert result["stale"] == 1
    assert result["stale_rows"] == [
        {"claim_id": "claim-source-id", "node_id": "node-source-id"}
    ]


def test_cli_exports_to_the_requested_ledger_and_reports_stale_rows(tmp_path):
    db_path = tmp_path / "graph.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    insert_record(
        conn,
        Record(
            id="record-source-id",
            title="Kenneth Arnold interview",
            content_hash="sha256:kenneth-arnold-record",
        ),
    )
    insert_node(
        conn, Node(id="node-source-id", name="Kenneth Arnold", node_type="person")
    )
    insert_claim(
        conn,
        Claim(
            id="claim-source-id",
            content="Kenneth Arnold reported nine objects.",
            claim_type="testimony",
            record_id="record-source-id",
        ),
    )
    _set_status(conn)
    conn.close()
    path = tmp_path / "curation" / "claim-ref-status.yaml"

    result = CliRunner().invoke(
        main,
        ["--db", str(db_path), "export-claim-ref-status", "--out", str(path)],
    )

    assert result.exit_code == 0, result.output
    assert "1 stale claim_ref_status row" in result.output
    assert _document(path) == {"schema": ledger.SCHEMA, "entries": []}


def test_corpus_contraction_drops_only_the_absent_record_explicitly(tmp_path):
    source = _graph()
    _set_status(source)
    insert_record(
        source,
        Record(id="gone-record", title="Removed", content_hash="sha256:gone-record"),
    )
    insert_node(source, Node(id="gone-node", name="Mount Rainier", node_type="place"))
    insert_claim(
        source,
        Claim(
            id="gone-claim",
            content="The sighting was near Mount Rainier.",
            claim_type="observation",
            record_id="gone-record",
            node_references=["gone-node"],
        ),
    )
    _set_status(source, claim_id="gone-claim", node_id="gone-node")
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)

    rebuilt = _graph(record_id="new-record", claim_id="new-claim", node_id="new-node")
    result = ledger.replay_claim_ref_status(rebuilt, path)

    assert result["applied"] == 1
    assert result["dropped_record_absent"] == 1
    assert result["dropped"][0]["outcome"] == "dropped_record_absent"


def test_surviving_record_resolution_failure_rolls_back_all_changes(tmp_path):
    source = _graph()
    _set_status(source)
    insert_node(source, Node(id="place", name="Mount Rainier", node_type="place"))
    source.execute(
        "INSERT INTO claim_node_refs (claim_id, node_id, salience) "
        "VALUES ('claim-source-id', 'place', 'setting')"
    )
    _set_status(source, claim_id="claim-source-id", node_id="place")
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    entries = _document(path)["entries"]
    entries.sort(key=lambda event: (event["set_at"], event["id"]))
    first_node = entries[0]["node"]

    rebuilt = _graph(
        record_id="new-record",
        claim_id="new-claim",
        node_id="new-node",
        with_ref=False,
    )
    rebuilt.execute(
        "UPDATE nodes SET name = ?, node_type = ? WHERE id = 'new-node'",
        (first_node["name"], first_node["node_type"]),
    )
    rebuilt.execute("DELETE FROM aliases")
    for alias in first_node["prior_names"]:
        rebuilt.execute(
            "INSERT INTO aliases (alias, node_id) VALUES (?, 'new-node')", (alias,)
        )
    rebuilt.commit()

    with pytest.raises(ledger.ClaimRefStatusReplayError):
        ledger.replay_claim_ref_status(rebuilt, path)

    assert rebuilt.execute("SELECT COUNT(*) FROM claim_ref_status").fetchone()[0] == 0
    assert rebuilt.execute("SELECT COUNT(*) FROM claim_node_refs").fetchone()[0] == 0


def test_verified_kenneth_arnold_decision_restores_a_missing_edge(tmp_path):
    source = _graph()
    _set_status(source)
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    rebuilt = _graph(
        record_id="new-record",
        claim_id="new-claim",
        node_id="new-kenneth-arnold",
        with_ref=False,
    )

    result = ledger.replay_claim_ref_status(rebuilt, path)

    assert result["restored_refs"] == 1
    assert rebuilt.execute(
        "SELECT claim_id, node_id, salience FROM claim_node_refs"
    ).fetchone() == ("new-claim", "new-kenneth-arnold", "subject")


def test_suspect_decision_never_restores_a_missing_edge(tmp_path):
    source = _graph()
    _set_status(source, "suspect")
    path = tmp_path / "ledger.yaml"
    ledger.export_claim_ref_status(source, path)
    rebuilt = _graph(
        record_id="new-record",
        claim_id="new-claim",
        node_id="new-node",
        with_ref=False,
    )

    with pytest.raises(
        ledger.ClaimRefStatusReplayError, match="suspect.*no claim-node"
    ):
        ledger.replay_claim_ref_status(rebuilt, path)

    assert rebuilt.execute("SELECT COUNT(*) FROM claim_ref_status").fetchone()[0] == 0
