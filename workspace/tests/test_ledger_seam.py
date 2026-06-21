"""TS->Python ledger seam: the workbench's edge writer appends @std/yaml entries
to the curation ledgers that this assimilator replays. These fixtures are the
edge's REAL @std/yaml output (workbench/edge/lib/ledger.ts), copied verbatim, so
this test catches drift between the TS emitter and the Python reader/replay.

Covers the quirks: flow empty list `[]`, block non-empty list, `by: null`,
unquoted unicode (敦賀), single-quoted forced names ('Object: low observable',
'@anonymous source', '#redacted'), single-quoted timestamps, op-first key order,
and a `---` multi-doc stream with merge+undo / reject+unreject reversibility.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from assimilator import merge
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import Claim, Node, Record

FIXTURES = Path(__file__).parent / "fixtures"


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    # nodes whose CURRENT names match the fixture's natural identities
    names = [
        ("o1", "object", "Tic Tac (敦賀)"),
        ("o2", "object", "Tic-Tac UAP"),
        ("o3", "object", "Object: low observable"),
        ("m1", "matter", "Section 1673: UAP provisions"),
        ("p1", "person", "@anonymous source"),
    ]
    for nid, ntype, name in names:
        insert_node(conn, Node(id=nid, node_type=ntype, name=name))
        insert_claim(
            conn,
            Claim(
                id=f"c-{nid}",
                content="x",
                claim_type="observation",
                record_id="r1",
                node_references=[nid],
            ),
        )
    conn.commit()
    return conn


def _stage(tmp_path, monkeypatch, *fixtures):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    for fixture, target in fixtures:
        shutil.copy(FIXTURES / fixture, tmp_path / target)


def test_deno_bytes_parse_with_all_quirks(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, ("deno-merges.yaml", "merges.yaml"))
    entries = merge.read_ledger()
    assert len(entries) == 2  # merge + undo, the --- multi-doc stream
    m = entries[0]
    assert m["op"] == "merge"  # op-first key order
    assert m["by"] is None  # by: null -> None
    assert m["canonical_name"] == "Tic Tac (敦賀)"  # unquoted unicode round-trips
    assert m["survivor"]["prior_names"] == []  # flow empty list
    assert m["victims"][0]["prior_names"] == [
        "the Tic Tac",
        "Nimitz object",
    ]  # block list
    assert (
        m["victims"][1]["name"] == "Object: low observable"
    )  # single-quoted forced name
    assert entries[1]["op"] == "undo"


def test_merge_then_undo_replays_to_no_op(tmp_path, monkeypatch):
    # The fixture is merge + its undo, so replay nets to nothing applied.
    _stage(tmp_path, monkeypatch, ("deno-merges.yaml", "merges.yaml"))
    conn = _graph()
    assert merge.replay_ledger(conn)["applied"] == 0
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='o2'").fetchone()[0] is None
    )


def test_merge_positive_path_resolves_real_bytes(tmp_path, monkeypatch):
    # Strip the undo doc so the merge actually applies, proving the real TS bytes
    # resolve by natural identity and the merge executes.
    _stage(tmp_path, monkeypatch, ("deno-merges.yaml", "merges.yaml"))
    text = (tmp_path / "merges.yaml").read_text().split("---\nop: undo")[0]
    (tmp_path / "merges.yaml").write_text(text)
    conn = _graph()
    assert merge.replay_ledger(conn)["applied"] == 1
    # both victims retired, survivor keeps canonical name
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='o2'").fetchone()[0]
        is not None
    )
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='o3'").fetchone()[0]
        is not None
    )
    assert (
        conn.execute("SELECT name FROM nodes WHERE id='o1'").fetchone()[0]
        == "Tic Tac (敦賀)"
    )


def test_reject_then_unreject_replays_to_no_op(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, ("deno-rejections.yaml", "rejections.yaml"))
    rejs = merge.read_rejections()
    assert len(rejs) == 2 and rejs[0]["op"] == "reject"
    assert rejs[0]["nodes"][1]["name"] == "@anonymous source"  # single-quoted @
    assert rejs[0]["nodes"][1]["prior_names"] == ["#redacted"]  # single-quoted #
    conn = _graph()
    assert merge.replay_rejections(conn)["applied"] == 0  # reject + unreject = no-op


def test_reject_positive_path_resolves_real_bytes(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, ("deno-rejections.yaml", "rejections.yaml"))
    text = (tmp_path / "rejections.yaml").read_text().split("---\nop: unreject")[0]
    (tmp_path / "rejections.yaml").write_text(text)
    conn = _graph()
    assert merge.replay_rejections(conn)["applied"] == 1
    # both nodes recorded under one rejection
    rows = conn.execute(
        "SELECT node_id FROM node_rejections WHERE undone_at IS NULL"
    ).fetchall()
    assert {r[0] for r in rows} == {"m1", "p1"}
