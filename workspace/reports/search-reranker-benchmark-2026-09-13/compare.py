#!/usr/bin/env python3
"""Apply the benchmark's documented replacement rule to two result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_results(minilm: dict, granite: dict) -> dict:
    """Compare quality, but use performance only with recorded exclusivity."""
    if minilm["fixture"]["sha256"] != granite["fixture"]["sha256"]:
        raise ValueError("results use different fixtures")
    if minilm["runtime"]["device"] != granite["runtime"]["device"]:
        raise ValueError("results use different devices")

    minilm_queries = {row["id"]: row for row in minilm["queries"]}
    granite_queries = {row["id"]: row for row in granite["queries"]}
    if minilm_queries.keys() != granite_queries.keys():
        raise ValueError("results use different query sets")

    ndcg_delta = granite["quality"]["ndcg_at_10"] - minilm["quality"]["ndcg_at_10"]
    mrr_delta = granite["quality"]["mrr_at_10"] - minilm["quality"]["mrr_at_10"]
    recall_delta = (
        granite["quality"]["recall_at_10"] - minilm["quality"]["recall_at_10"]
    )
    query_deltas = {
        query_id: granite_queries[query_id]["ndcg_at_10"]
        - minilm_queries[query_id]["ndcg_at_10"]
        for query_id in minilm_queries
    }
    severe_regressions = sorted(
        query_id for query_id, delta in query_deltas.items() if delta < -0.10
    )
    latency_ratio = (
        granite["performance"]["query_latency_median_ms"]
        / minilm["performance"]["query_latency_median_ms"]
    )
    device = minilm["runtime"]["device"]
    memory_field = "cuda_peak_allocated_mib" if device == "cuda" else "rss_peak_mib"
    memory_ratio = (
        granite["performance"][memory_field] / minilm["performance"][memory_field]
    )
    performance_valid = all(
        result.get("performance_validity", {}).get("exclusive_control_verified") is True
        for result in (minilm, granite)
    )

    criteria = {
        "ndcg_delta_at_least_0_03": ndcg_delta >= 0.03,
        "mrr_does_not_decline": mrr_delta >= 0,
        "at_most_two_ndcg_regressions_below_minus_0_10": len(severe_regressions) <= 2,
        "median_latency_no_more_than_2x": (
            latency_ratio <= 2 if performance_valid else None
        ),
        "peak_memory_no_more_than_2x": (
            memory_ratio <= 2 if performance_valid else None
        ),
    }
    return {
        "schema": "anomalica/search-reranker-comparison/1",
        "device": device,
        "performance_valid": performance_valid,
        "performance_decision_use": "valid" if performance_valid else "invalid",
        "fixture_sha256": minilm["fixture"]["sha256"],
        "quality_delta_granite_minus_minilm": {
            "ndcg_at_10": ndcg_delta,
            "mrr_at_10": mrr_delta,
            "recall_at_10": recall_delta,
        },
        "resource_ratios_granite_over_minilm": {
            "median_latency": latency_ratio,
            "peak_memory": memory_ratio,
            "memory_measure": memory_field,
            "scope": (
                "process-local allocator" if device == "cuda" else "process-local RSS"
            ),
            "decision_use": "valid" if performance_valid else "invalid",
        },
        "query_ndcg_deltas": query_deltas,
        "severe_regressions": severe_regressions,
        "criteria": criteria,
        "replace_minilm": all(value is True for value in criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--granite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    minilm = json.loads(args.minilm.read_text())
    granite = json.loads(args.granite.read_text())
    try:
        result = compare_results(minilm, granite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
