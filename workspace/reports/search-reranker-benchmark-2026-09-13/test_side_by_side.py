import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent


def test_report_is_complete_and_safe():
    report = json.loads((HERE / "side-by-side.json").read_text())

    assert report["report_only"] is True
    assert report["canonical_activation"] == "forbidden"
    assert len(report["queries"]) == 16
    assert report["inputs"]["source_identity"]["digest_count"] == 58
    for query in report["queries"]:
        assert len(query["rankings"]["minilm"]) == 10
        assert len(query["rankings"]["granite"]) == 10
        for model in ("minilm", "granite"):
            assert [row["rank"] for row in query["rankings"][model]] == list(
                range(1, 11)
            )
            assert all(row["text"] for row in query["rankings"][model])
            assert all(row["source_record"]["id"] for row in query["rankings"][model])


def test_evaluation_state_is_bound_to_public_evidence():
    state = json.loads((HERE / "evaluation-state.json").read_text())
    compact = json.dumps(
        state["evidence"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )

    assert state["schema"] == "anomalica/evaluation-state/1"
    assert state["evaluation_id"] == "search-reranker-minilm-vs-granite"
    assert state["status"] == "adopted"
    assert state["gold"] == {
        "status": "reviewed-derived",
        "reviewed": 16,
        "total": 16,
        "unit": "queries",
    }
    assert (
        state["evidence_sha256"]
        == "sha256:" + hashlib.sha256(compact.encode()).hexdigest()
    )
    paths = {
        "search-reranker-minilm-vs-granite-fixture": HERE / "fixture.json",
        "search-reranker-minilm-vs-granite-result": HERE
        / "controlled-run-2026-09-14"
        / "controlled-comparison-cuda.json",
    }
    for evidence in state["evidence"]:
        assert evidence["artifact_id"] != (
            "search-reranker-minilm-vs-granite-human-judgements"
        )
        assert evidence["sha256"].startswith("sha256:")
        assert (
            evidence["sha256"]
            == "sha256:"
            + hashlib.sha256(paths[evidence["artifact_id"]].read_bytes()).hexdigest()
        )
