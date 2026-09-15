"""Node merge: re-point, retire, reversible, and ledger replay by natural id."""

from __future__ import annotations

import json
import sqlite3

import pytest
import yaml

from assimilator import merge

from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import (
    Claim,
    Node,
    NodeType,
    OriginKind,
    ProvenanceChain,
    Record,
)

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
            ref_roles={"B": "participant"},
            location_in_record="00:01:02",
            date="2004-11-14",
            provenance_chain=ProvenanceChain(
                origin_kind=OriginKind.anonymous,
                origin="a controller",
                origin_ref="controller-1",
            ),
            attribution_in_text=True,
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
            ref_roles={"A": "mentioned", "B": "subject"},
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


def _edge_rows(conn):
    return conn.execute(
        "SELECT claim_id, node_id, salience FROM claim_node_refs ORDER BY claim_id, node_id"
    ).fetchall()


def _claim_envelope(conn):
    return conn.execute(
        "SELECT id, content, original_excerpt, claim_type, attestation, record_id, "
        "location_in_record, date, date_end, confidence, metadata, created_at, "
        "claim_role, claim_hash, origin_kind, origin, relay, entailment_label, "
        "entailment_score, entailment_model, entailment_premise, origin_ref, "
        "attribution_in_text FROM claims ORDER BY id"
    ).fetchall()


def test_merge_repoints_retires_and_renames():
    conn = _graph()
    envelope = _claim_envelope(conn)
    merge.merge_nodes(conn, "A", ["B"], "2004 Nimitz Encounter", "m1")
    # B's claim refs now on A (c1 moved, c2 already there)
    assert _refs(conn, "A") == {"c1", "c2"}
    assert _refs(conn, "B") == set()
    assert _edge_rows(conn) == [("c1", "A", "participant"), ("c2", "A", "subject")]
    assert _claim_envelope(conn) == envelope
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
    before_edges = _edge_rows(conn)
    before_envelope = _claim_envelope(conn)
    merge.merge_nodes(conn, "A", ["B"], "Canonical", "m1")
    merge.undo_merge(conn, "m1")
    assert _refs(conn, "A") == before_a  # A back to {c2}
    assert _refs(conn, "B") == before_b  # B back to {c1, c2}
    assert _edge_rows(conn) == before_edges
    assert _claim_envelope(conn) == before_envelope
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


def test_multi_victim_undo_restores_original_survivor_salience():
    conn = _graph()
    insert_node(conn, Node(id="C", node_type="event", name="Third Fragment"))
    conn.execute(
        "INSERT INTO claim_node_refs (claim_id, node_id, salience) "
        "VALUES ('c2', 'C', 'setting')"
    )
    before = _edge_rows(conn)

    merge.merge_nodes(conn, "A", ["B", "C"], "Canonical", "m-many")
    assert conn.execute(
        "SELECT salience FROM claim_node_refs WHERE claim_id = 'c2' AND node_id = 'A'"
    ).fetchone() == ("subject",)

    merge.undo_merge(conn, "m-many")
    assert _edge_rows(conn) == before


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
    before_envelope = _claim_envelope(fresh)
    result = merge.replay_ledger(fresh)
    assert result["applied"] == 1
    # the merge took effect on the fresh graph
    assert _name(fresh, "A") == "2004 Nimitz Encounter"
    assert _refs(fresh, "B") == set()
    assert _name(fresh, "B") is not None  # B still exists, just retired
    assert _edge_rows(fresh) == [("c1", "A", "participant"), ("c2", "A", "subject")]
    assert _claim_envelope(fresh) == before_envelope
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


def test_resolve_natural_requires_one_exact_node_across_all_declared_names():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="A", node_type="event", name="Current event"))
    insert_node(conn, Node(id="B", node_type="event", name="Prior event"))

    assert (
        merge._resolve_natural(
            conn,
            {
                "name": "Current event",
                "node_type": "event",
                "prior_names": ["Prior event"],
            },
        )
        is None
    )


def test_merge_replay_orders_equal_timestamps_by_stable_operation_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    entries = []
    for merge_id, suffix in (("merge-z", "Z"), ("merge-a", "A")):
        entries.append(
            {
                "op": "merge",
                "merge_id": merge_id,
                "at": "2026-09-15T00:00:00Z",
                "by": "test",
                "canonical_name": f"Canonical {suffix}",
                "survivor": {
                    "name": f"Survivor {suffix}",
                    "node_type": "event",
                    "prior_names": [],
                },
                "victims": [
                    {
                        "name": f"Victim {suffix}",
                        "node_type": "event",
                        "prior_names": [],
                    }
                ],
                "confirmation": _CONFIRMED,
            }
        )
    (tmp_path / "merges.yaml").write_text(
        "".join("---\n" + yaml.safe_dump(entry, sort_keys=False) for entry in entries)
    )
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for suffix in ("A", "Z"):
        insert_node(
            conn,
            Node(id=f"survivor-{suffix}", node_type="event", name=f"Survivor {suffix}"),
        )
        insert_node(
            conn,
            Node(id=f"victim-{suffix}", node_type="event", name=f"Victim {suffix}"),
        )
    applied = []
    original = merge.merge_nodes

    def recording_merge(*args, **kwargs):
        applied.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(merge, "merge_nodes", recording_merge)

    assert merge.replay_ledger(conn)["applied"] == 2
    assert applied == ["merge-a", "merge-z"]


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


def _write_merge_proposal(
    tmp_path, proposal_id, old_name="Nimitz Incident", proposed_name="Canonical"
):
    directory = tmp_path / "curation" / "rename-proposals"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{proposal_id}.json").write_text(
        json.dumps(
            {
                "id": proposal_id,
                "node_id": "audit",
                "node_name_at_proposal": old_name,
                "proposed_name": proposed_name,
                "reason": None,
                "proposed_by": "test",
                "proposed_at": "2026-09-15T00:00:00Z",
            }
        )
    )


def test_merge_append_with_proposal_refs_is_idempotent_and_collision_safe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    _write_merge_proposal(tmp_path, "p1")
    conn = _graph()
    arguments = (
        conn,
        "A",
        ["B"],
        "Canonical",
        "merge-proposal",
        "2026-09-15T01:00:00Z",
        "operator",
    )
    kwargs = {
        "confirmation": _CONFIRMED,
        "proposal_ids": ["rename-proposal:p1"],
    }

    merge.append_merge_entry(*arguments, **kwargs)
    before = merge.ledger_path().read_bytes()
    merge.append_merge_entry(*arguments, **kwargs)
    assert merge.ledger_path().read_bytes() == before
    with pytest.raises(ValueError, match="not pending"):
        merge.append_merge_entry(
            conn,
            "A",
            ["B"],
            "Canonical",
            "different-merge",
            "2026-09-15T02:00:00Z",
            "operator",
            **kwargs,
        )
    with pytest.raises(ValueError, match="collision"):
        merge.append_merge_entry(*arguments[:3], "Different", *arguments[4:], **kwargs)


def test_merge_proposal_refs_require_known_pending_unique_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    _write_merge_proposal(tmp_path, "p1")
    conn = _graph()
    common = (
        conn,
        "A",
        ["B"],
        "Canonical",
        "merge-proposal",
        "2026-09-15T01:00:00Z",
        "operator",
    )
    with pytest.raises(ValueError, match="unique sorted"):
        merge.append_merge_entry(
            *common,
            confirmation=_CONFIRMED,
            proposal_ids=["rename-proposal:p1", "rename-proposal:p1"],
        )
    with pytest.raises(ValueError, match="unknown"):
        merge.append_merge_entry(
            *common,
            confirmation=_CONFIRMED,
            proposal_ids=["rename-proposal:missing"],
        )


def test_merge_rejects_a_proposal_unrelated_to_the_selected_nodes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    _write_merge_proposal(tmp_path, "p1", old_name="Unrelated")

    with pytest.raises(ValueError, match="does not resolve to one selected node"):
        merge.append_merge_entry(
            _graph(),
            "A",
            ["B"],
            "Canonical",
            "merge-unrelated",
            "2026-09-15T01:00:00Z",
            "operator",
            confirmation=_CONFIRMED,
            proposal_ids=["rename-proposal:p1"],
        )


def test_cli_proposal_refs_do_not_confirm_a_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    _write_merge_proposal(tmp_path, "p1")
    db = tmp_path / "graph.db"
    conn = sqlite3.connect(db)
    init_db(conn)
    insert_node(conn, Node(id="A", node_type="event", name="Nimitz Incident"))
    insert_node(conn, Node(id="B", node_type="event", name="Nimitz encounter"))
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit):
        merge.main(
            [
                "--db",
                str(db),
                "--survivor",
                "A",
                "--victims",
                "B",
                "--name",
                "Canonical",
                "--proposal-id",
                "rename-proposal:p1",
            ]
        )
    assert not merge.ledger_path().exists()


def test_cli_accepts_repeated_exact_proposal_id_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    _write_merge_proposal(tmp_path, "p1")
    _write_merge_proposal(tmp_path, "p2")
    db = tmp_path / "graph.db"
    conn = sqlite3.connect(db)
    init_db(conn)
    insert_node(conn, Node(id="A", node_type="event", name="Nimitz Incident"))
    insert_node(conn, Node(id="B", node_type="event", name="Nimitz encounter"))
    conn.commit()
    conn.close()

    assert (
        merge.main(
            [
                "--db",
                str(db),
                "--survivor",
                "A",
                "--victims",
                "B",
                "--name",
                "Canonical",
                "--confirmed-by",
                "operator",
                "--proposal-id",
                "rename-proposal:p1",
                "--proposal-id",
                "rename-proposal:p2",
            ]
        )
        == 0
    )
    assert merge.read_ledger()[0]["proposal_ids"] == [
        "rename-proposal:p1",
        "rename-proposal:p2",
    ]
