"""The derived stages must agree with each other."""

import sqlite3

import yaml

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


def test_an_undone_veto_with_no_page_is_reported(tmp_path):
    """The assembler retires a vetoed node's page; an undo re-proposes the node
    and restores nothing. Somebody has to see that."""
    from assimilator.consistency import check_all

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="n1", name="Telepathy", node_type=NodeType.topic))
    conn.execute(
        "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, source_count, status, computed_at) "
        "VALUES ('n1', 'topic', 'page-worthy', 9, 3, 'proposed', '2026-09-03')"
    )
    conn.execute(
        "INSERT INTO page_vetoes (veto_id, node_id, reason, created_at, created_by, undone_at) "
        "VALUES ('v-1234', 'n1', 'too generic', '2026-09-02', 'wb', '2026-09-03')"
    )
    conn.commit()
    briefs = tmp_path / "briefs" / "topics"
    briefs.mkdir(parents=True)
    (briefs / "telepathy.yaml").write_text(
        yaml.safe_dump(
            {
                "page": {"nodes": [{"node_id": "n1"}], "slug": "telepathy"},
                "brief_hash": "h",
            }
        )
    )
    content = tmp_path / "content"
    (content / "topics").mkdir(parents=True)

    names = {f.check: f for f in check_all(conn, tmp_path / "briefs", content)}

    assert names["veto-undone-page-absent"].samples == [
        "topics/telepathy (veto v-1234)"
    ]
