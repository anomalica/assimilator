#!/usr/bin/env python3
"""Compare frozen-candidate search ranking with MiniLM and Granite English R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path


MODELS = {
    "minilm": {
        "repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "architecture": "cross-encoder",
    },
    "granite": {
        "repo": "ibm-granite/granite-embedding-english-r2",
        "revision": "47ea694b257b703fee9253d75c2b1f2985180498",
        "architecture": "bi-encoder",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rss_kib(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1])
    return 0


def ndcg_at(relevances: list[int], k: int) -> float:
    actual = sum(
        rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1)
    )
    ideal_rels = sorted(relevances, reverse=True)[:k]
    ideal = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
    return actual / ideal if ideal else 0.0


def reciprocal_rank_at(relevances: list[int], k: int) -> float:
    return next((1.0 / rank for rank, rel in enumerate(relevances[:k], 1) if rel), 0.0)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def load_model(model_key: str, device: str):
    from huggingface_hub import snapshot_download

    spec = MODELS[model_key]
    snapshot = Path(snapshot_download(spec["repo"], revision=spec["revision"]))
    started = time.perf_counter()
    if model_key == "minilm":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(str(snapshot), device=device)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(snapshot), device=device)
    load_seconds = time.perf_counter() - started
    weights = snapshot / "model.safetensors"
    return model, snapshot, weights, load_seconds


def score(model_key: str, model, query: str, documents: list[str], batch_size: int):
    if model_key == "minilm":
        values = model.predict(
            [(query, document) for document in documents],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in values]

    vectors = model.encode(
        [query, *documents],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = vectors[0]
    return [float(query_vector @ vector) for vector in vectors[1:]]


def evaluate_query(query_row: dict, scores: list[float]) -> dict:
    ranked = sorted(
        zip(query_row["candidates"], scores),
        key=lambda item: item[1],
        reverse=True,
    )
    relevances = [item[0]["relevance"] for item in ranked]
    relevant_in_pool = sum(relevances)
    top_ten = [
        {
            "rank": rank,
            "claim_id": candidate["claim_id"],
            "relevance": candidate["relevance"],
            "score": model_score,
        }
        for rank, (candidate, model_score) in enumerate(ranked[:10], 1)
    ]
    return {
        "id": query_row["id"],
        "ndcg_at_10": ndcg_at(relevances, 10),
        "mrr_at_10": reciprocal_rank_at(relevances, 10),
        "recall_at_10": sum(relevances[:10]) / relevant_in_pool,
        "candidate_recall_at_50": relevant_in_pool / query_row["total_graph_relevant"],
        "relevant_in_pool": relevant_in_pool,
        "total_graph_relevant": query_row["total_graph_relevant"],
        "top_10": top_ten,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument(
        "--exclusive-control-evidence",
        type=Path,
        help="JSON evidence manifest from a controlled run; absent means performance is invalid",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import sentence_transformers
    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    fixture = json.loads(args.fixture.read_text())
    evidence = None
    if args.exclusive_control_evidence:
        evidence = json.loads(args.exclusive_control_evidence.read_text())
        if evidence.get("exclusive_control_verified") is not True:
            raise SystemExit("exclusive-control evidence does not verify exclusivity")
    run_started_at = datetime.now(UTC).isoformat()
    baseline_rss = rss_kib("VmRSS")
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, snapshot, weights, load_seconds = load_model(args.model, args.device)
    loaded_rss = rss_kib("VmRSS")

    query_rows = list(fixture["queries"])
    random.Random(args.query_seed).shuffle(query_rows)
    for query_row in query_rows[: args.warmups]:
        score(
            args.model,
            model,
            query_row["query"],
            [item["text"] for item in query_row["candidates"]],
            args.batch_size,
        )

    query_results = []
    latencies = []
    cuda_latencies = []
    latency_samples = []
    total_pairs = 0
    for query_row in query_rows:
        documents = [item["text"] for item in query_row["candidates"]]
        final_scores = None
        for iteration in range(args.iterations):
            if args.device == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter()
            final_scores = score(
                args.model, model, query_row["query"], documents, args.batch_size
            )
            if args.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                cuda_ms = start_event.elapsed_time(end_event)
                cuda_latencies.append(cuda_ms / 1000)
            else:
                cuda_ms = None
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            latency_samples.append(
                {
                    "query_id": query_row["id"],
                    "iteration": iteration,
                    "wall_ms": elapsed * 1000,
                    "cuda_event_ms": cuda_ms,
                }
            )
            total_pairs += len(documents)
        query_results.append(evaluate_query(query_row, final_scores))

    means = {
        metric: statistics.fmean(row[metric] for row in query_results)
        for metric in (
            "ndcg_at_10",
            "mrr_at_10",
            "recall_at_10",
            "candidate_recall_at_50",
        )
    }
    result = {
        "schema": "anomalica/search-reranker-result/1",
        "fixture": {
            "path": str(args.fixture),
            "sha256": file_sha256(args.fixture),
            "queries": len(fixture["queries"]),
            "candidates_per_query": fixture["retrieval"]["candidate_count"],
        },
        "model": {
            **MODELS[args.model],
            "snapshot": str(snapshot),
            "weights_sha256": file_sha256(weights),
        },
        "runtime": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "query_seed": args.query_seed,
            "query_order": [row["id"] for row in query_rows],
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "performance_validity": {
            "exclusive_control_verified": evidence is not None,
            "decision_use": "valid" if evidence is not None else "invalid",
            "evidence_path": (
                str(args.exclusive_control_evidence) if evidence is not None else None
            ),
            "evidence_sha256": (
                file_sha256(args.exclusive_control_evidence)
                if evidence is not None
                else None
            ),
            "reason": (
                "exclusive-control evidence supplied"
                if evidence is not None
                else "no contemporaneous isolation or utilisation evidence supplied"
            ),
        },
        "quality": means,
        "performance": {
            "model_load_seconds": load_seconds,
            "query_latency_median_ms": statistics.median(latencies) * 1000,
            "query_latency_p95_ms": percentile(latencies, 0.95) * 1000,
            "cuda_event_latency_median_ms": (
                statistics.median(cuda_latencies) * 1000 if cuda_latencies else None
            ),
            "cuda_event_latency_p95_ms": (
                percentile(cuda_latencies, 0.95) * 1000 if cuda_latencies else None
            ),
            "pairs_per_second": total_pairs / sum(latencies),
            "rss_before_model_mib": baseline_rss / 1024,
            "rss_after_model_mib": loaded_rss / 1024,
            "rss_peak_mib": rss_kib("VmHWM") / 1024,
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if args.device == "cuda"
                else None
            ),
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / (1024 * 1024)
                if args.device == "cuda"
                else None
            ),
            "latency_samples": latency_samples,
        },
        "queries": query_results,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"quality": means, "performance": result["performance"]}, indent=2)
    )


if __name__ == "__main__":
    main()
