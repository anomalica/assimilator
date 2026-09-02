"""The shortlist's profile is a function of a node's claims, not their row order."""

import json
import sqlite3

import numpy as np

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import shortlist
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.entity_reranker import profile_claims


def _graph(order):
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    insert_node(conn, Node(id="n1", name="Kevin Day", node_type=NodeType.person))
    texts = {
        "a": "short",
        "b": "a much longer claim about radar contacts held for twelve minutes",
        "c": "a medium length claim about the ship",
    }
    for cid in order:
        insert_claim(
            conn,
            Claim(
                id=cid,
                content=texts[cid],
                claim_type="testimony",
                record_id="r1",
                node_references=["n1"],
            ),
        )
        conn.execute("UPDATE claims SET claim_hash = ? WHERE id = ?", ("h-" + cid, cid))
    conn.commit()
    return conn


def test_the_profile_is_the_same_whatever_order_the_claims_were_stored_in():
    first = profile_claims(_graph(["a", "b", "c"]), "n1", n=2)
    second = profile_claims(_graph(["c", "a", "b"]), "n1", n=2)
    assert first == second
    assert first[0].startswith("a much longer")  # longest first


def test_knn_pairs_are_unordered_and_any_type():
    ids = ["x", "y", "z"]
    v = np.asarray([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pairs = shortlist.knn_pairs(ids, v, k=1)
    assert ("x", "y") in pairs
    assert all(a < b for a, b in pairs)


def test_rules_pairs_drop_nodes_no_longer_live(tmp_path):
    p = tmp_path / "cands.json"
    p.write_text(
        json.dumps(
            {"clusters": [{"node_ids": ["a", "b"]}, {"node_ids": ["a", "gone"]}]}
        )
    )
    pairs, dropped = shortlist.rules_pairs(p, {"a", "b"})
    assert pairs == {("a", "b")} and dropped == 1


def test_shortlist_runs_end_to_end_with_a_fake_embedder():
    conn = _graph(["a", "b", "c"])
    insert_node(conn, Node(id="n2", name="K. Day", node_type=NodeType.person))
    insert_node(conn, Node(id="n3", name="Apollo 11", node_type=NodeType.project))
    conn.commit()

    def embed(texts):
        # Days together, Apollo apart.
        return [[1.0, 0.0] if "Day" in t else [0.0, 1.0] for t in texts]

    out = shortlist.shortlist(conn, embed, k=1)
    assert ("n1", "n2") in out["pairs"]
    assert out["vectors"].shape == (3, 2)
