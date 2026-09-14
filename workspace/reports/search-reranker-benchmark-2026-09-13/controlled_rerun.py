#!/usr/bin/env python3
"""Run a monitored, balanced CUDA benchmark without managing host services.

The controller refuses to start unless speech-to-text and the scheduler are
already inactive. It never stops or starts either service; restoration belongs
to the maintenance-window owner.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmark import MODELS, file_sha256, percentile
from compare import compare_results


GPU_FIELDS = (
    "timestamp,uuid,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,utilization.memory,clocks.current.graphics,"
    "clocks.current.sm,clocks.current.memory,power.draw,power.limit,"
    "temperature.gpu,compute_mode"
)
GPU_KEYS = GPU_FIELDS.split(",")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def balanced_order() -> list[str]:
    """Six fresh runs per model in balanced ABBA/BAAB/ABBA order."""
    return [
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


def _csv_row(text: str) -> list[str]:
    return [value.strip() for value in text.strip().split(",")]


def gpu_sample(phase: str, allowed_pid: int | None) -> dict:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_FIELDS}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    error = None
    gpu_data = {}
    app_rows = []
    if gpu.returncode:
        error = f"gpu query failed: {gpu.stderr.strip()}"
    else:
        values = _csv_row(gpu.stdout.splitlines()[0])
        if len(values) != len(GPU_KEYS):
            error = f"gpu query returned {len(values)} fields, expected {len(GPU_KEYS)}"
        else:
            gpu_data = dict(zip(GPU_KEYS, values))
    if apps.returncode:
        error = f"{error}; " if error else ""
        error += f"process query failed: {apps.stderr.strip()}"
    else:
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            values = _csv_row(line)
            if len(values) == 3:
                app_rows.append(
                    {
                        "pid": int(values[0]),
                        "process_name": values[1],
                        "used_memory_mib": values[2],
                    }
                )
    return {
        "observed_at": utc_now(),
        "monotonic": time.monotonic(),
        "phase": phase,
        "allowed_pid": allowed_pid,
        "gpu": gpu_data,
        "compute_apps": app_rows,
        "error": error,
    }


class Monitor:
    def __init__(self, trace_path: Path, interval: float = 1.0):
        self.trace_path = trace_path
        self.interval = interval
        self.samples: list[dict] = []
        self.phase = "starting"
        self.allowed_pid: int | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def set_phase(self, phase: str, allowed_pid: int | None = None) -> None:
        with self.lock:
            self.phase = phase
            self.allowed_pid = allowed_pid

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.samples)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval * 3)

    def _run(self) -> None:
        next_sample = time.monotonic()
        with self.trace_path.open("w") as trace:
            while not self.stop_event.is_set():
                with self.lock:
                    phase = self.phase
                    allowed_pid = self.allowed_pid
                sample = gpu_sample(phase, allowed_pid)
                with self.lock:
                    self.samples.append(sample)
                trace.write(json.dumps(sample) + "\n")
                trace.flush()
                next_sample += self.interval
                self.stop_event.wait(max(0, next_sample - time.monotonic()))


def service_states() -> dict[str, str]:
    commands = {
        "speech-to-text.service": [
            "systemctl",
            "is-active",
            "speech-to-text.service",
        ],
        "anomalica-scheduler.service": [
            "systemctl",
            "--user",
            "is-active",
            "anomalica-scheduler.service",
        ],
    }
    return {
        unit: subprocess.run(command, capture_output=True, text=True).stdout.strip()
        for unit, command in commands.items()
    }


def inactive(states: dict[str, str]) -> bool:
    return all(state in {"inactive", "failed"} for state in states.values())


def validate_exclusivity(
    samples: list[dict],
    before_services: dict[str, str],
    after_services: dict[str, str],
    minimum_idle_samples: int,
) -> tuple[bool, list[str], dict]:
    errors = []
    if not inactive(before_services):
        errors.append(f"services active before run: {before_services}")
    if not inactive(after_services):
        errors.append(f"services active after run: {after_services}")
    if any(sample.get("error") for sample in samples):
        errors.append("one or more nvidia-smi samples failed")

    baseline = [sample for sample in samples if sample["phase"] == "baseline"]
    post = [sample for sample in samples if sample["phase"] == "post-baseline"]
    if len(baseline) < minimum_idle_samples:
        errors.append(f"only {len(baseline)} baseline samples")
    if len(post) < minimum_idle_samples:
        errors.append(f"only {len(post)} post-baseline samples")

    unexpected = []
    for sample in samples:
        allowed_pid = sample.get("allowed_pid")
        for app in sample.get("compute_apps") or []:
            if allowed_pid is None or app["pid"] != allowed_pid:
                unexpected.append(
                    {
                        "observed_at": sample["observed_at"],
                        "phase": sample["phase"],
                        **app,
                    }
                )
    if unexpected:
        errors.append(f"{len(unexpected)} unexpected compute-process observations")

    gaps = [
        later["monotonic"] - earlier["monotonic"]
        for earlier, later in zip(samples, samples[1:])
    ]
    max_gap = max(gaps, default=0.0)
    if max_gap > 1.75:
        errors.append(f"monitor gap {max_gap:.3f}s exceeds 1.75s")

    def used_mib(sample: dict) -> float:
        try:
            return float(sample["gpu"]["memory.used"])
        except (KeyError, TypeError, ValueError):
            return float("inf")

    baseline_memory = [used_mib(sample) for sample in baseline]
    post_memory = [used_mib(sample) for sample in post]
    baseline_max = max(baseline_memory, default=float("inf"))
    post_max = max(post_memory, default=float("inf"))
    if post_max > baseline_max + 64:
        errors.append(
            f"post-run memory {post_max:.0f} MiB exceeds baseline {baseline_max:.0f} MiB + 64"
        )

    evidence = {
        "sample_count": len(samples),
        "baseline_samples": len(baseline),
        "post_baseline_samples": len(post),
        "maximum_sample_gap_seconds": max_gap,
        "baseline_memory_used_max_mib": baseline_max,
        "post_memory_used_max_mib": post_max,
        "unexpected_compute_processes": unexpected,
    }
    return not errors, errors, evidence


def wait_for_idle_samples(monitor: Monitor, count: int = 3) -> None:
    while True:
        samples = monitor.snapshot()[-count:]
        if len(samples) == count and all(
            not sample["compute_apps"] for sample in samples
        ):
            return
        time.sleep(0.25)


def ranked_claim_ids(result: dict) -> dict[str, list[str]]:
    """Ranking identity independent of query order and insignificant score drift."""
    return {
        row["id"]: [item["claim_id"] for item in row["top_10"]]
        for row in result["queries"]
    }


def aggregate_results(paths: list[Path], evidence_path: Path) -> dict:
    results = [json.loads(path.read_text()) for path in paths]
    first = results[0]
    quality = first["quality"]
    rankings = ranked_claim_ids(first)
    if any(result["quality"] != quality for result in results[1:]):
        raise RuntimeError("quality changed between repetitions")
    if any(ranked_claim_ids(result) != rankings for result in results[1:]):
        raise RuntimeError("ranking changed between repetitions")

    samples = [
        sample
        for result in results
        for sample in result["performance"]["latency_samples"]
    ]
    wall_seconds = [sample["wall_ms"] / 1000 for sample in samples]
    cuda_seconds = [sample["cuda_event_ms"] / 1000 for sample in samples]
    pairs = first["fixture"]["candidates_per_query"] * len(samples)
    evidence_hash = file_sha256(evidence_path)
    aggregate = dict(first)
    aggregate["schema"] = "anomalica/search-reranker-controlled-result/1"
    aggregate["runtime"] = {
        **first["runtime"],
        "fresh_process_repetitions": len(results),
        "source_results": [str(path) for path in paths],
        "query_seeds": [result["runtime"]["query_seed"] for result in results],
        "run_started_at": results[0]["runtime"]["run_started_at"],
        "run_finished_at": results[-1]["runtime"]["run_finished_at"],
    }
    aggregate["performance_validity"] = {
        "exclusive_control_verified": True,
        "decision_use": "valid",
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_hash,
    }
    aggregate["performance"] = {
        "model_load_seconds": statistics.median(
            result["performance"]["model_load_seconds"] for result in results
        ),
        "query_latency_median_ms": statistics.median(wall_seconds) * 1000,
        "query_latency_p95_ms": percentile(wall_seconds, 0.95) * 1000,
        "cuda_event_latency_median_ms": statistics.median(cuda_seconds) * 1000,
        "cuda_event_latency_p95_ms": percentile(cuda_seconds, 0.95) * 1000,
        "pairs_per_second": pairs / sum(wall_seconds),
        "rss_peak_mib": max(
            result["performance"]["rss_peak_mib"] for result in results
        ),
        "cuda_peak_allocated_mib": max(
            result["performance"]["cuda_peak_allocated_mib"] for result in results
        ),
        "cuda_peak_reserved_mib": max(
            result["performance"]["cuda_peak_reserved_mib"] for result in results
        ),
        "latency_samples": samples,
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    if args.run_dir.exists():
        raise SystemExit(f"run directory already exists: {args.run_dir}")
    before_services = service_states()
    if not inactive(before_services):
        raise SystemExit(f"required services are active: {before_services}")

    from huggingface_hub import snapshot_download

    for spec in MODELS.values():
        snapshot_download(spec["repo"], revision=spec["revision"])

    args.run_dir.mkdir(parents=True)
    trace_path = args.run_dir / "gpu-monitor.jsonl"
    lock_path = Path("/tmp/opencode/anomalica-search-reranker-benchmark.lock")
    benchmark_path = Path(__file__).with_name("benchmark.py")
    run_records = []

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another reranker benchmark holds the local lock") from exc

        monitor = Monitor(trace_path)
        monitor.set_phase("baseline")
        monitor.start()
        try:
            time.sleep(args.idle_seconds + 1)
            for index, model in enumerate(balanced_order(), 1):
                output = args.run_dir / f"run-{index:02d}-{model}.json"
                log = args.run_dir / f"run-{index:02d}-{model}.log"
                command = [
                    sys.executable,
                    str(benchmark_path),
                    "--fixture",
                    str(args.fixture),
                    "--model",
                    model,
                    "--device",
                    "cuda",
                    "--iterations",
                    str(args.iterations),
                    "--warmups",
                    str(args.warmups),
                    "--query-seed",
                    str(args.seed + index),
                    "--out",
                    str(output),
                ]
                with log.open("w") as stream:
                    process = subprocess.Popen(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONHASHSEED": "0"},
                    )
                    monitor.set_phase(f"run-{index:02d}-{model}", process.pid)
                    returncode = process.wait()
                wait_for_idle_samples(monitor)
                monitor.set_phase("between-runs")
                run_records.append(
                    {
                        "index": index,
                        "model": model,
                        "pid": process.pid,
                        "command": command,
                        "returncode": returncode,
                        "result": str(output),
                        "log": str(log),
                    }
                )
                if returncode:
                    raise RuntimeError(f"run {index} failed; see {log}")

            monitor.set_phase("post-baseline")
            time.sleep(args.idle_seconds + 1)
        finally:
            monitor.stop()

    after_services = service_states()
    samples = monitor.snapshot()
    verified, errors, summary = validate_exclusivity(
        samples, before_services, after_services, args.idle_seconds
    )
    evidence_path = args.run_dir / "exclusive-control-evidence.json"
    evidence = {
        "schema": "anomalica/gpu-exclusive-control-evidence/1",
        "exclusive_control_verified": verified,
        "errors": errors,
        "started_at": samples[0]["observed_at"] if samples else None,
        "finished_at": samples[-1]["observed_at"] if samples else None,
        "monitor_interval_seconds": 1,
        "trace": str(trace_path),
        "trace_sha256": file_sha256(trace_path),
        "services_before": before_services,
        "services_after": after_services,
        "order": balanced_order(),
        "runs": run_records,
        **summary,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    if not verified:
        raise SystemExit(f"exclusive control NOT verified: {errors}")

    aggregates = {}
    for model in MODELS:
        paths = [
            Path(record["result"]) for record in run_records if record["model"] == model
        ]
        aggregate = aggregate_results(paths, evidence_path)
        path = args.run_dir / f"controlled-{model}-cuda.json"
        path.write_text(json.dumps(aggregate, indent=2) + "\n")
        aggregates[model] = aggregate

    comparison = compare_results(aggregates["minilm"], aggregates["granite"])
    comparison_path = args.run_dir / "controlled-comparison-cuda.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"exclusive control verified; comparison: {comparison_path}")


if __name__ == "__main__":
    main()
