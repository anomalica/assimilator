"""The derived stages must agree with each other."""

import sqlite3

from anomalica_common.digest.models import Node, NodeType
from assimilator.consistency import check_all
from assimilator.database import init_db, insert_node


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_a_proposal_for_a_retired_node_is_reported(tmp_path):
    """A merge retires its victims and page_proposals is derived, so it goes
    stale until propose-pages reruns. Meanwhile the scheduler enumerates assembly
    work for a node that no longer exists - which is what happened after the
    Elizondo, Puthoff and OUSDI merges on 2026-08-21."""
    conn = _db()
    insert_node(conn, Node(id="gone", name="Lou Elizondo", node_type=NodeType.person))
    conn.execute("UPDATE nodes SET retired_at = '2026-08-21' WHERE id = 'gone'")
    conn.execute(
        "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
        "source_count, status, computed_at) VALUES ('gone','person','page-worthy',20,2,'proposed','x')"
    )
    found = check_all(conn, tmp_path, None)
    assert [f.check for f in found] == ["proposal-points-at-dead-node"]
    assert found[0].count == 1


def test_a_consistent_graph_reports_nothing(tmp_path):
    conn = _db()
    insert_node(conn, Node(id="live", name="David Fravor", node_type=NodeType.person))
    assert check_all(conn, tmp_path, None) == []
