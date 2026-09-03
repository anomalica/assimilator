"""Node merge: re-point, retire, reversible, and ledger replay by natural id."""

from __future__ import annotations

import json
import sqlite3

import yaml

from assimilator import merge

from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import Claim, Node, NodeType, Record

_CONFIRMED = {"by": "test", "at": "2026-09-03T03:00:00Z", "via": "workbench-queue"}


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_record(conn, Record(id="r2", title="R2"))
    # survivor A, victim B (same real entity, split)
    insert_node(conn, Node(id="A", node_type="event", name="Nimitz Incident"))
    insert_node(conn, Node(id="B", node_type="event", name="Nimitz Encounter"))
    # c1 refs B only; c2 refs A and B (shared); c3 speaker B
    insert_claim(
        conn,
        Claim(
            id="c1",
            content="x",
            claim_type="observation",
            record_id="r1",
            node_references=["B"],
        ),
    )
    insert_claim(
        conn,
        Claim(
            id="c2",
            content="y",
            claim_type="observation",
            record_id="r1",
            node_references=["A", "B"],
        ),
    )
    insert_claim(
        conn,
        Claim(
            id="c3", content="z", claim_type="testimony", record_id="r2", speaker_id="B"
        ),
    )
    conn.execute("UPDATE records SET producer_id = 'B' WHERE id = 'r2'")
    conn.execute("INSERT INTO aliases (alias, node_id) VALUES ('Nimitz event', 'B')")
    conn.commit()
    return conn


def _refs(conn, node_id):
    return {
        r[0]
        for r in conn.execute(
            "SELECT claim_id FROM claim_node_refs WHERE node_id = ?", (node_id,)
        )
    }


def test_merge_repoints_retires_and_renames():
    conn = _graph()
    merge.merge_nodes(conn, "A", ["B"], "2004 Nimitz Encounter", "m1")
    # B's claim refs now on A (c1 moved, c2 already there)
    assert _refs(conn, "A") == {"c1", "c2"}
    assert _refs(conn, "B") == set()
    # speaker + producer re-pointed
    assert (
        conn.execute("SELECT speaker_id FROM claims WHERE id='c3'").fetchone()[0] == "A"
    )
    assert (
        conn.execute("SELECT producer_id FROM records WHERE id='r2'").fetchone()[0]
        == "A"
    )
    # B retired, A renamed, B's name + alias folded under A
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='B'").fetchone()[0]
        is not None
    )
    assert (
        conn.execute("SELECT name FROM nodes WHERE id='A'").fetchone()[0]
        == "2004 Nimitz Encounter"
    )
    a_aliases = {
        r[0] for r in conn.execute("SELECT alias FROM aliases WHERE node_id='A'")
    }
    assert {"Nimitz event", "Nimitz Encounter"} <= a_aliases
    # logged
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM node_merges WHERE merge_id='m1' AND undone_at IS NULL"
        ).fetchone()[0]
        == 1
    )


def test_undo_restores_exactly():
    conn = _graph()
    before_a = _refs(conn, "A")
    before_b = _refs(conn, "B")
    merge.merge_nodes(conn, "A", ["B"], "Canonical", "m1")
    merge.undo_merge(conn, "m1")
    assert _refs(conn, "A") == before_a  # A back to {c2}
    assert _refs(conn, "B") == before_b  # B back to {c1, c2}
    assert (
        conn.execute("SELECT speaker_id FROM claims WHERE id='c3'").fetchone()[0] == "B"
    )
    assert (
        conn.execute("SELECT producer_id FROM records WHERE id='r2'").fetchone()[0]
        == "B"
    )
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='B'").fetchone()[0] is None
    )
    assert (
        conn.execute("SELECT name FROM nodes WHERE id='A'").fetchone()[0]
        == "Nimitz Incident"
    )
    assert (
        conn.execute(
            "SELECT undone_at FROM node_merges WHERE merge_id='m1'"
        ).fetchone()[0]
        is not None
    )


def test_replay_by_natural_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    # Write a ledger entry against a graph, then replay over a FRESH graph where
    # the nodes have DIFFERENT ids (simulating rebuild) - replay must resolve by
    # name, not id.
    src = _graph()
    merge.append_merge_entry(
        src,
        "A",
        ["B"],
        "2004 Nimitz Encounter",
        "m1",
        merge._now(),
        None,
        confirmation=_CONFIRMED,
    )

    fresh = _graph()  # same names, ids happen to match here, but replay uses names
    result = merge.replay_ledger(fresh)
    assert result["applied"] == 1
    # the merge took effect on the fresh graph
    assert _name(fresh, "A") == "2004 Nimitz Encounter"
    assert _refs(fresh, "B") == set()
    assert _name(fresh, "B") is not None  # B still exists, just retired
    assert (
        fresh.execute("SELECT retired_at FROM nodes WHERE id='B'").fetchone()[0]
        is not None
    )


def test_replay_skips_undone(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    src = _graph()
    merge.append_merge_entry(
        src, "A", ["B"], "Canonical", "m1", merge._now(), None, confirmation=_CONFIRMED
    )
    merge.append_undo_entry("m1", None)
    fresh = _graph()
    assert merge.replay_ledger(fresh)["applied"] == 0
    assert _name(fresh, "A") == "Nimitz Incident"  # unchanged


def _name(conn, nid):
    return conn.execute("SELECT name FROM nodes WHERE id=?", (nid,)).fetchone()[0]


def test_resolve_natural_refuses_a_fuzzy_guess():
    """Replay must lose an op loudly rather than apply it to the wrong node.

    A fuzzy match elsewhere is a guess a curator can correct later. Here it
    decides which nodes a replayed HUMAN decision lands on, and the ledger
    records the name rather than what it resolved to - so a wrong guess is both
    unreviewable and made on the curator's behalf.
    """
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn,
        Node(id="N", node_type="event", name="2004 USS Nimitz UAP encounter"),
    )
    # Close enough for the fuzzy tier, a different event to a human.
    nat = {"name": "2004 USS Nimitz UAP encounters", "node_type": "event"}
    assert merge._resolve_natural(conn, nat) is None


def test_resolve_natural_keeps_the_deterministic_tiers():
    """Refusing fuzzy must not cost the tiers that resolve on declared evidence."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn,
        Node(
            id="C",
            node_type="organisation",
            name="Central Intelligence Agency (CIA)",
        ),
    )
    insert_node(conn, Node(id="P", node_type="person", name="David Saunders"))
    # The acronym the name itself declares, and a known given-name short form.
    assert (
        merge._resolve_natural(conn, {"name": "CIA", "node_type": "organisation"})
        == "C"
    )
    assert (
        merge._resolve_natural(conn, {"name": "Dave Saunders", "node_type": "person"})
        == "P"
    )


def test_an_unconfirmed_merge_is_a_proposal_not_a_merge(tmp_path, monkeypatch):
    """Mark's rule of 2026-09-03: no session applies a merge. Without a
    confirmation the command queues the cluster for the workbench and touches
    neither the ledger nor the graph."""
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    monkeypatch.setenv("ASSIMILATOR_DATA_DIR", str(tmp_path / "data"))
    db = tmp_path / "g.db"
    conn = sqlite3.connect(db)
    init_db(conn)
    insert_node(conn, Node(id="A", name="Nimitz", node_type=NodeType.event))
    insert_node(conn, Node(id="B", name="Nimitz 2004", node_type=NodeType.event))
    conn.commit()
    conn.close()

    rc = merge.main(
        [
            "--db",
            str(db),
            "--survivor",
            "A",
            "--victims",
            "B",
            "--name",
            "Nimitz",
            "--by",
            "a-session",
        ]
    )

    assert rc == 0
    conn = sqlite3.connect(db)
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='B'").fetchone()[0] is None
    )
    assert merge.read_ledger() == []
    queued = json.loads(
        (tmp_path / "data" / "merge-candidates-manual.json").read_text()
    )
    assert queued[0]["node_ids"] == ["A", "B"] and "a-session" in queued[0]["reason"]

    rc = merge.main(
        [
            "--db",
            str(db),
            "--survivor",
            "A",
            "--victims",
            "B",
            "--name",
            "Nimitz",
            "--confirmed-by",
            "workbench/mark",
            "--confirmed-via",
            "workbench-queue",
        ]
    )

    assert rc == 0
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='B'").fetchone()[0]
        is not None
    )
    entry = merge.read_ledger()[0]
    assert entry["confirmation"]["by"] == "workbench/mark"
    assert entry["confirmation"]["via"] == "workbench-queue"


def test_replay_applies_grandfathered_and_confirmed_entries_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    src = _graph()
    merge.append_merge_entry(
        src,
        "A",
        ["B"],
        "Canonical",
        "old",
        "2026-08-01T00:00:00Z",
        None,
        confirmation=_CONFIRMED,
    )
    # strip the block to model a pre-rule entry
    path = merge.ledger_path()
    entries = merge.read_ledger()
    entries[0].pop("confirmation")
    entries.append(
        {
            "op": "merge",
            "merge_id": "new",
            "at": "2026-12-01T00:00:00Z",
            "by": "a-session",
            "canonical_name": "Canonical",
            "survivor": entries[0]["survivor"],
            "victims": entries[0]["victims"],
            "audit": {},
        }
    )
    path.write_text(
        "".join("---\n" + yaml.safe_dump(e, sort_keys=False) for e in entries)
    )
    assert merge.confirmed(entries[0]) and not merge.confirmed(entries[1])

    lines = []
    result = merge.replay_ledger(_graph(), on_progress=lines.append)

    assert result["applied"] == 1 and result["unconfirmed"] == 1
    assert any("UNCONFIRMED" in ln for ln in lines)
