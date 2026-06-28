"""The page-worthiness gate: node-type tier + independent-source floor."""

from __future__ import annotations

import sqlite3

from anomalica_common.digest.models import Claim, Node, Record
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.page_gate import page_gate_rows


def _add_claims(conn, node_id, records):
    """One claim per (node, record) entry; pass a record id more than once for
    several claims from the same source."""
    for i, rid in enumerate(records):
        insert_claim(
            conn,
            Claim(
                id=f"{node_id}-c{i}",
                content=f"claim {i}",
                claim_type="testimony",
                record_id=rid,
                node_references=[node_id],
            ),
        )


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("r1", "r2", "r3"):
        insert_record(conn, Record(id=rid, title=rid))
    nodes = {
        "worthy": "person",  # 3 claims / 2 sources -> passes page-worthy
        "thin": "person",  # 2 claims / 2 sources -> fails (claims)
        "onesrc": "person",  # 3 claims / 1 source -> fails (sources)
        "place_high": "place",  # 6 claims / 3 sources -> passes high-bar
        "place_mid": "place",  # 5 claims / 3 sources -> fails high-bar floor
        "deprecated": "matter",  # rich, but gated (type not in TIER)
    }
    for nid, nt in nodes.items():
        insert_node(conn, Node(id=nid, node_type=nt, name=nid.title()))
    _add_claims(conn, "worthy", ["r1", "r1", "r2"])
    _add_claims(conn, "thin", ["r1", "r2"])
    _add_claims(conn, "onesrc", ["r1", "r1", "r1"])
    _add_claims(conn, "place_high", ["r1", "r1", "r2", "r2", "r3", "r3"])
    _add_claims(conn, "place_mid", ["r1", "r2", "r3", "r3", "r3"])
    _add_claims(conn, "deprecated", ["r1", "r1", "r2", "r2", "r3", "r3"])
    conn.commit()
    return conn


def _by_id(rows):
    return {r["node_id"]: r for r in rows}


def test_page_worthy_floor():
    rows = _by_id(page_gate_rows(_graph()))
    assert "worthy" in rows
    assert rows["worthy"]["tier"] == "page-worthy"
    assert rows["worthy"]["claim_count"] == 3
    assert rows["worthy"]["source_count"] == 2


def test_page_worthy_rejects_thin_and_single_source():
    rows = _by_id(page_gate_rows(_graph()))
    assert "thin" not in rows  # only 2 claims
    assert "onesrc" not in rows  # 3 claims but 1 source


def test_high_bar_floor_is_stricter():
    rows = _by_id(page_gate_rows(_graph()))
    assert "place_high" in rows  # 6 claims / 3 sources clears high-bar
    assert rows["place_high"]["tier"] == "high-bar"
    # place_mid (5/3) would pass the page-worthy floor but fails high-bar - proves
    # the tier, not a flat count, decides.
    assert "place_mid" not in rows


def test_deprecated_types_gated_out():
    rows = _by_id(page_gate_rows(_graph()))
    assert "deprecated" not in rows  # matter is not in TIER, never page-worthy


def test_ordered_strongest_first():
    rows = page_gate_rows(_graph())
    counts = [r["claim_count"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_floor_env_tunable(monkeypatch):
    monkeypatch.setenv("ANOMALICA_PAGE_WORTHY_MIN_CLAIMS", "2")
    rows = _by_id(page_gate_rows(_graph()))
    assert "thin" in rows  # 2 claims now clears the lowered floor


def test_retired_nodes_excluded():
    conn = _graph()
    conn.execute("UPDATE nodes SET retired_at = '2026-01-01' WHERE id = 'worthy'")
    assert "worthy" not in _by_id(page_gate_rows(conn))
