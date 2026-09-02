"""The recurring merge job: memoised by pair, dry by default, never merging."""

import json
import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import merge_pipeline as mp
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    for nid, name, t in (
        ("a", "Lockheed U-2", NodeType.object),
        ("b", "U-2 spy plane", NodeType.object),
        ("c", "Apollo 11", NodeType.project),
    ):
        insert_node(conn, Node(id=nid, name=name, node_type=t))
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


def _embed(texts):
    return [[1.0, 0.0] if "U-2" in t else [0.0, 1.0] for t in texts]


class _Reranker:
    model_id = "stub-reranker"

    def score(self, pairs, batch_size=8, symmetric=True):
        return [
            0.97
            if a.name.startswith(("Lockheed", "U-2"))
            and b.name.startswith(("Lockheed", "U-2"))
            else 0.05
            for a, b in pairs
        ]


def _call(prompt, text, model, schema=None, use_api=None):
    return json.dumps(
        {"decisions": [{"pair_id": 1, "same": True, "reason": "one aircraft"}]}
    )


def _env(monkeypatch, tmp_path):
    for env, name in (
        ("ASSIMILATOR_RERANK_SCORES", "scores.jsonl"),
        ("ASSIMILATOR_VERIFY_VERDICTS", "verdicts.jsonl"),
        ("ASSIMILATOR_MERGE_SHORTLIST_RUNS", "runs.jsonl"),
        ("ANOMALICA_MERGE_CANDIDATES", "cands.json"),
        ("ANOMALICA_MERGE_CANDIDATES_MANUAL", "manual.json"),
    ):
        monkeypatch.setenv(env, str(tmp_path / name))


def test_dry_run_plans_and_calls_nothing(monkeypatch, tmp_path, capsys):
    _env(monkeypatch, tmp_path)
    conn = _graph()
    code = mp.run_pipeline(
        conn,
        _embed,
        lambda: (_ for _ in ()).throw(AssertionError("must not load")),
        None,
        None,
        "m",
        False,
        True,
        print,
        k=1,
    )
    out = capsys.readouterr().out
    assert code == mp.EXIT_OK
    line = [ln for ln in out.splitlines() if ln.startswith("PLAN_JSON ")][0]
    plan = json.loads(line[len("PLAN_JSON ") :])
    assert (
        plan["to_score"] >= 1 and plan["to_verify"] == 0 and plan["verify_calls"] == 0
    )
    assert plan["cached_context_tokens"] == 0 and plan["notional_usd"] == 0
    assert not (tmp_path / "scores.jsonl").exists()


def test_a_run_scores_judges_lands_and_then_has_nothing_new(
    monkeypatch, tmp_path, capsys
):
    _env(monkeypatch, tmp_path)
    conn = _graph()
    code = mp.run_pipeline(
        conn,
        _embed,
        _Reranker,
        _call,
        json.loads,
        "claude-haiku-4-5",
        False,
        False,
        print,
        k=1,
    )
    assert code == mp.EXIT_OK
    scores = mp.load_scores(tmp_path / "scores.jsonl")
    assert ("a", "b") in scores and scores[("a", "b")]["with_claims"] == 0.97
    verdicts = mp.load_verdicts(tmp_path / "verdicts.jsonl")
    assert verdicts[("a", "b")]["same"] is True
    manual = json.loads((tmp_path / "manual.json").read_text())
    assert len(manual) == 1 and manual[0]["reason"].startswith("verify: one aircraft")
    assert set(manual[0]["node_ids"]) == {"a", "b"}
    assert (
        conn.execute("SELECT COUNT(*) FROM nodes WHERE retired_at IS NULL").fetchone()[
            0
        ]
        == 3
    )  # nothing merged
    run = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert run["pairs"]["same"] == 1 and run["outcome"] == "ok"

    # Nothing changed: no scoring, no call, exit 10, and the queue entry is not duplicated.
    code = mp.run_pipeline(
        conn,
        _embed,
        lambda: (_ for _ in ()).throw(AssertionError("must not load")),
        None,
        None,
        "m",
        False,
        False,
        print,
        k=1,
    )
    assert code == mp.EXIT_NOTHING
    assert len(json.loads((tmp_path / "manual.json").read_text())) == 1


def test_a_failed_verify_keeps_what_landed_and_exits_6(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    conn = _graph()

    def bad_call(*a, **k):
        raise RuntimeError("cli down")

    code = mp.run_pipeline(
        conn, _embed, _Reranker, bad_call, json.loads, "m", False, False, print, k=1
    )
    assert code == mp.EXIT_VERIFY
    assert (tmp_path / "scores.jsonl").exists()  # the GPU work is kept
