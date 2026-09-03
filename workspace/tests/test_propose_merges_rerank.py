"""The merge queue ordered by the entity reranker, with the rules' verdict kept."""

import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import propose_merges
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    for nid, name in (
        ("a1", "Kevin Day"),
        ("a2", "K. Day"),
        ("b1", "Apollo 11"),
        ("b2", "Apollo 12"),
    ):
        insert_node(conn, Node(id=nid, name=name, node_type=NodeType.person))
        insert_claim(
            conn,
            Claim(
                id=f"{nid}-c",
                content=f"about {name}",
                claim_type="testimony",
                record_id="r1",
                node_references=[nid],
            ),
        )
    conn.commit()
    return conn


class _Stub:
    """Scores by the names it is shown, so the test controls the verdict."""

    device = "cpu"

    def score(self, pairs, **_):
        return [0.97 if a.name.split()[0] == "Apollo" else 0.12 for a, _b in pairs]

    def peak_memory_mb(self):
        return None


def test_rerank_orders_the_queue_and_keeps_the_rules_verdict(monkeypatch):
    conn = _graph()
    clusters = [
        {
            "node_ids": ["a1", "a2"],
            "score": 0.95,
            "reason": "name-equiv",
            "node_type": "person",
            "suggested_canonical": "Kevin Day",
        },
        {
            "node_ids": ["b1", "b2"],
            "score": 0.8,
            "reason": "fuzzy",
            "node_type": "person",
            "suggested_canonical": "Apollo 11",
        },
    ]
    monkeypatch.setattr(
        propose_merges, "rerank_clusters", propose_merges.rerank_clusters
    )
    import assimilator.entity_reranker as er

    monkeypatch.setattr(er, "get_reranker", lambda *_a, **_k: _Stub())

    run = propose_merges.rerank_clusters(conn, clusters)
    clusters.sort(key=lambda c: c["score"], reverse=True)

    assert run == {
        "pairs": 2,
        "scored_now": 2,
        "prompts": 4,
        "device": "cpu",
        "gpu_peak_mb": None,
    }
    # The shared memo: a second pass scores nothing and loads no reranker.
    monkeypatch.setattr(
        er,
        "get_reranker",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("memo miss")),
    )
    again = propose_merges.rerank_clusters(
        conn, [dict(c, score=c["rule_score"]) for c in clusters]
    )
    assert again["scored_now"] == 0 and again["prompts"] == 0

    assert [c["node_ids"] for c in clusters] == [["b1", "b2"], ["a1", "a2"]]
    top, second = clusters
    assert (
        top["score"] == 0.97 and top["rule_score"] == 0.8 and top["reason"] == "fuzzy"
    )
    assert second["score"] == 0.12 and second["rule_score"] == 0.95
    assert top["pairs"] == [{"node_ids": ["b1", "b2"], "reranker": 0.97}]


def test_the_reranker_sees_names_types_and_claims_from_the_graph():
    from assimilator.entity_reranker import entity_from_graph

    conn = _graph()
    e = entity_from_graph(conn, "a1")
    assert e.name == "Kevin Day" and e.node_type == "person"
    assert e.claims == ["about Kevin Day"]
    assert "Claims:\n- about Kevin Day" in e.text()
    assert entity_from_graph(conn, "missing") is None


def test_rerank_is_refused_when_the_policy_does_not_permit_the_model(
    tmp_path, monkeypatch, capsys
):
    """Policy before loading, fail closed: the run must not reach the weights."""
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "models:\n- id: Qwen/Qwen3-Reranker-0.6B\n  provider: qwen\n"
        "stages:\n- id: rerank\n  uses_model: true\n  priority: []\n  deny: []\n"
    )
    monkeypatch.setenv("ANOMALICA_MODEL_POLICY", str(policy))
    db = tmp_path / "g.db"
    _graph().backup(sqlite3.connect(db))

    code = propose_merges.main(
        ["--db", str(db), "--out", str(tmp_path / "c.json"), "--rerank"]
    )

    assert code == 2
    assert "rerank refused" in capsys.readouterr().err
    assert not (tmp_path / "c.json").exists()
