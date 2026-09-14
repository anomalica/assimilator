from copy import deepcopy

from compare import compare_results


def _result(model: str) -> dict:
    ndcg = 0.50 if model == "minilm" else 0.55
    latency = 100 if model == "minilm" else 110
    memory = 100 if model == "minilm" else 110
    return {
        "fixture": {"sha256": "fixture"},
        "runtime": {"device": "cuda"},
        "quality": {"ndcg_at_10": ndcg, "mrr_at_10": 0.5, "recall_at_10": 0.5},
        "performance": {
            "query_latency_median_ms": latency,
            "cuda_peak_allocated_mib": memory,
        },
        "performance_validity": {"exclusive_control_verified": False},
        "queries": [{"id": "q", "ndcg_at_10": ndcg}],
    }


def test_uncontrolled_timings_cannot_satisfy_replacement_criteria():
    result = compare_results(_result("minilm"), _result("granite"))

    assert result["performance_valid"] is False
    assert result["performance_decision_use"] == "invalid"
    assert result["criteria"]["median_latency_no_more_than_2x"] is None
    assert result["criteria"]["peak_memory_no_more_than_2x"] is None
    assert result["replace_minilm"] is False


def test_controlled_timings_can_be_evaluated():
    minilm = _result("minilm")
    granite = _result("granite")
    for result in (minilm, granite):
        result["performance_validity"]["exclusive_control_verified"] = True

    comparison = compare_results(deepcopy(minilm), deepcopy(granite))

    assert comparison["performance_valid"] is True
    assert comparison["criteria"]["median_latency_no_more_than_2x"] is True
    assert comparison["criteria"]["peak_memory_no_more_than_2x"] is True
    assert comparison["replace_minilm"] is True
