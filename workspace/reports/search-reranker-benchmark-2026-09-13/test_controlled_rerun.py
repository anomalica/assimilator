from controlled_rerun import balanced_order, ranked_claim_ids, validate_exclusivity


def _sample(index: int, phase: str, apps=None, allowed_pid=None, memory=15):
    return {
        "observed_at": f"T{index}",
        "monotonic": float(index),
        "phase": phase,
        "allowed_pid": allowed_pid,
        "gpu": {"memory.used": str(memory)},
        "compute_apps": apps or [],
        "error": None,
    }


def test_order_is_balanced_and_interleaved():
    order = balanced_order()

    assert order == [
        "minilm",
        "granite",
        "granite",
        "minilm",
        "granite",
        "minilm",
        "minilm",
        "granite",
        "minilm",
        "granite",
        "granite",
        "minilm",
    ]
    assert order.count("minilm") == order.count("granite") == 6


def test_clean_trace_verifies_exclusive_control():
    samples = [_sample(i, "baseline") for i in range(60)]
    samples.append(
        _sample(
            60,
            "run-01-minilm",
            apps=[{"pid": 123, "process_name": "python", "used_memory_mib": "100"}],
            allowed_pid=123,
            memory=200,
        )
    )
    samples.extend(_sample(i, "post-baseline") for i in range(61, 121))

    verified, errors, _evidence = validate_exclusivity(
        samples,
        {"speech": "inactive", "scheduler": "inactive"},
        {"speech": "inactive", "scheduler": "inactive"},
        60,
    )

    assert verified is True
    assert errors == []


def test_unexpected_process_and_monitor_gap_invalidate_trace():
    samples = [_sample(i, "baseline") for i in range(60)]
    samples.append(
        _sample(
            62,
            "run-01-minilm",
            apps=[{"pid": 999, "process_name": "whisper", "used_memory_mib": "1000"}],
            allowed_pid=123,
            memory=1200,
        )
    )
    samples.extend(_sample(i, "post-baseline", memory=100) for i in range(63, 123))

    verified, errors, evidence = validate_exclusivity(
        samples,
        {"speech": "inactive", "scheduler": "inactive"},
        {"speech": "active", "scheduler": "inactive"},
        60,
    )

    assert verified is False
    assert evidence["unexpected_compute_processes"][0]["pid"] == 999
    assert any("monitor gap" in error for error in errors)
    assert any("services active after" in error for error in errors)


def test_ranking_identity_ignores_query_order_and_score_drift():
    first = {
        "queries": [
            {"id": "a", "top_10": [{"claim_id": "1", "score": 0.1}]},
            {"id": "b", "top_10": [{"claim_id": "2", "score": 0.2}]},
        ]
    }
    repeated = {
        "queries": [
            {"id": "b", "top_10": [{"claim_id": "2", "score": 0.200001}]},
            {"id": "a", "top_10": [{"claim_id": "1", "score": 0.100001}]},
        ]
    }

    assert ranked_claim_ids(first) == ranked_claim_ids(repeated)
