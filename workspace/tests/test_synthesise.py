"""Brief synthesis: the deterministic graph-slice-to-brief stage."""

import sqlite3

import yaml

from anomalica_common.digest.models import Node, NodeType
from assimilator import synthesise
from assimilator.database import init_db, insert_node


def test_prune_removes_briefs_for_retired_and_renamed_nodes(tmp_path):
    """Emission only writes, so a brief outlives the node it describes.

    Two ways that happens, and both are hazards rather than clutter: the
    assembler takes a brief by slug and will build a page from a dead one.

    A MERGE retires its victims - after Luis/Lue/Lou Elizondo, lou-elizondo.yaml
    pointed at a node that no longer existed.

    A RENAME strands the survivor's own brief. That merge kept node 87788ebc and
    renamed it, so lue-elizondo.yaml and luis-elizondo.yaml both described the
    SAME LIVE NODE - invisible to a retired-check, and enough to republish the
    duplicate page the merge was meant to remove.
    """
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn, Node(id="live-1", name="Luis Elizondo", node_type=NodeType.person)
    )
    insert_node(conn, Node(id="dead-1", name="Lou Elizondo", node_type=NodeType.person))
    conn.execute("UPDATE nodes SET retired_at = '2026-08-20' WHERE id = 'dead-1'")

    for stem, node_id in (
        ("luis-elizondo", "live-1"),  # current: kept
        ("lue-elizondo", "live-1"),  # stranded by the rename: pruned
        ("lou-elizondo", "dead-1"),  # retired node: pruned
        ("someone-else", "absent-1"),  # node not in the graph at all: pruned
    ):
        (tmp_path / f"{stem}.yaml").write_text(
            yaml.safe_dump({"page": {"node_id": node_id, "slug": stem}})
        )

    removed = synthesise.prune_retired_briefs(
        conn, tmp_path, slug_map={"live-1": "luis-elizondo"}
    )

    assert sorted(removed) == [
        "lou-elizondo.yaml",
        "lue-elizondo.yaml",
        "someone-else.yaml",
    ]
    assert (tmp_path / "luis-elizondo.yaml").exists()


def test_prune_never_deletes_an_unreadable_brief(tmp_path):
    """A file we cannot parse is left for a human. Deleting blind is how a
    parser bug becomes data loss."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    bad = tmp_path / "broken.yaml"
    bad.write_text("{{{ not yaml")
    assert synthesise.prune_retired_briefs(conn, tmp_path, slug_map={}) == []
    assert bad.exists()
