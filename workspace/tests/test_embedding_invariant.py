"""A stamp exists only for a live row whose CURRENT text was embedded.

Both halves of a stale embedding do harm, and the vector half does more. A stale
stamp only makes a coverage figure wrong - that is how 7,269 stamps read as
7,269 embedded when 2,583 rows were live. A stale VECTOR sits in the search
index, so a nearest-neighbour query returns ids that no longer exist and a
renamed node keeps answering to its old name.
"""

from __future__ import annotations

import sqlite3

import pytest
from anomalica_common.digest.models import Claim, Node, Record

from assimilator.database import (
    delete_claim,
    init_db,
    insert_claim,
    insert_node,
    insert_record,
)
from assimilator.embed_batches import forget_embeddings
from assimilator.embeddings import (
    EMBEDDING_DIMS,
    EMBEDDING_MODEL_ID,
    init_vec,
    serialise_f32,
)

MODEL = EMBEDDING_MODEL_ID


@pytest.fixture
def graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    init_vec(conn)
    insert_record(conn, Record(id="r1", title="R1", content_hash="sha256:aa"))
    conn.commit()
    return conn


def _embed(conn, kind, row_id):
    table, column = (
        ("vec_claims", "claim_id") if kind == "claim" else ("vec_nodes", "node_id")
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {table}({column}, embedding) VALUES (?, ?)",
        [row_id, serialise_f32([0.01] * EMBEDDING_DIMS)],
    )
    conn.execute(
        "INSERT OR REPLACE INTO embedding_model(kind, id, model_id, embedded_at) "
        "VALUES (?, ?, ?, 'T')",
        [kind, row_id, MODEL],
    )
    conn.commit()


def _state(conn, kind, row_id):
    table, column = (
        ("vec_claims", "claim_id") if kind == "claim" else ("vec_nodes", "node_id")
    )
    stamp = conn.execute(
        "SELECT COUNT(*) FROM embedding_model WHERE kind = ? AND id = ?", (kind, row_id)
    ).fetchone()[0]
    vector = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (row_id,)
    ).fetchone()[0]
    return stamp, vector


def test_deleting_a_claim_takes_its_stamp_and_its_vector(graph):
    """A re-digest deletes and recreates claims. Without this the vector outlives
    the claim and search returns an id that is not there."""
    insert_claim(
        graph,
        Claim(id="c1", content="a", claim_type="testimony", record_id="r1"),
    )
    graph.commit()
    _embed(graph, "claim", "c1")
    assert _state(graph, "claim", "c1") == (1, 1)

    delete_claim(graph, "c1")
    graph.commit()

    assert _state(graph, "claim", "c1") == (0, 0)


def test_renaming_a_node_drops_the_vector_of_its_old_name(graph):
    """The stamp would still read current, so nothing would ever re-embed it."""
    from assimilator.merge import rename_node

    node = insert_node(graph, Node(id="n1", node_type="person", name="Bob Smith"))
    graph.commit()
    _embed(graph, "node", node.id)

    rename_node(graph, node.id, "Robert Smith", "rn-1", created_by="test")
    graph.commit()

    assert _state(graph, "node", node.id) == (0, 0)


def test_retiring_a_node_in_a_merge_drops_both(graph):
    """A retired node is not a live row. Unconditional rather than "except merge
    victims, so undo is cheaper" - an invariant with an exception reads as a bug
    to whoever finds the exception first."""
    from assimilator.merge import merge_nodes

    a = insert_node(graph, Node(id="n1", node_type="person", name="Bob Smith"))
    b = insert_node(graph, Node(id="n2", node_type="person", name="Robert Smith"))
    graph.commit()
    _embed(graph, "node", a.id)
    _embed(graph, "node", b.id)

    merge_nodes(graph, a.id, [b.id], "Robert Smith", "mg-1", created_by="test")

    assert _state(graph, "node", b.id) == (0, 0), "the retired victim keeps neither"


def test_a_survivor_keeps_its_embedding_when_its_name_does_not_change(graph):
    """Re-embedding is about a second, but doing it when nothing changed would
    make every merge look like embedding work that is not needed."""
    from assimilator.merge import merge_nodes

    a = insert_node(graph, Node(id="n1", node_type="person", name="Bob Smith"))
    b = insert_node(graph, Node(id="n2", node_type="person", name="Robert Smith"))
    graph.commit()
    _embed(graph, "node", a.id)

    merge_nodes(graph, a.id, [b.id], "Bob Smith", "mg-2", created_by="test")

    assert _state(graph, "node", a.id) == (1, 1)


def test_forget_is_safe_before_any_embed_run():
    """embedding_model and the vec tables do not exist until the first run."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1", content_hash="sha256:aa"))
    insert_claim(
        conn, Claim(id="c1", content="a", claim_type="testimony", record_id="r1")
    )
    conn.commit()

    forget_embeddings(conn, "claim", "c1")  # must not raise
    delete_claim(conn, "c1")
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def test_forget_loads_the_extension_rather_than_skipping_the_vector(tmp_path):
    """The importer runs host-side without sqlite-vec loaded. Skipping the vector
    there is exactly what leaves an unreachable row behind, and nothing
    downstream can see it - vec_claims has no foreign key to join against."""
    path = tmp_path / "g.db"
    setup = sqlite3.connect(path)
    init_db(setup)
    init_vec(setup)
    insert_record(setup, Record(id="r1", title="R1", content_hash="sha256:aa"))
    insert_claim(
        setup, Claim(id="c1", content="a", claim_type="testimony", record_id="r1")
    )
    setup.commit()
    _embed(setup, "claim", "c1")
    setup.close()

    plain = sqlite3.connect(path)  # no init_vec: vec0 is not loaded here
    forget_embeddings(plain, "claim", "c1")
    plain.commit()
    plain.close()

    check = sqlite3.connect(path)
    init_vec(check)
    assert _state(check, "claim", "c1") == (0, 0)
