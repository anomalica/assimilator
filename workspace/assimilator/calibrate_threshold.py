"""Calibrate the audit-clustering cosine threshold against human adjudication gold.

The threshold in ``anomalica_common.embedding_client`` ships as a defensible
default measured on one record; it is NOT gold-calibrated, because until a human
adjudicates there is nothing to fit it to. This tool closes that loop. When a
reviewer hand-links claims in the workbench audit view, each gold CLUSTER is a
human assertion that its members state the same fact - a labelled positive. Two
claims the reviewer left in different clusters (or unclustered) are a labelled
negative. That is exactly the set the threshold has to separate.

It reads the `{hash}.audit.json` sidecars (schema anomalica/audit/2), embeds each
member's text through the running embed_service (the FIXED, dequantised space -
raw uint8 cosine was degenerate), and reports:

  - the cosine distribution of human-LINKED cross-variant pairs (positives) vs
    human-SEPARATED same-record cross-variant pairs (negatives);
  - the separation between them (near zero => no threshold can work, look at the
    space or the passage grouping, not the number);
  - how the current default threshold scores, and the cut that best separates.

Cross-variant only: linking two claims from the SAME extraction variant is not
what the audit clusters (it merges same-fact-different-MODEL), so same-variant
pairs are excluded from both sides. Host-runnable - talks to the endpoint over
localhost, needs no fastembed. Run with the endpoint up (`just embed-service`).

    python -m assimilator.calibrate_threshold [--ingests DIR] [--threshold 0.83]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

from anomalica_common.embedding_client import (
    DEFAULT_THRESHOLD,
    EmbeddingCache,
    EmbeddingUnavailable,
)

SCHEMA = "anomalica/audit/2"


def _ingests_store(explicit: str | None) -> Path:
    base = explicit or os.environ.get("ANOMALICA_INGESTS_DIR")
    if base:
        return Path(base) / "store"
    return Path.home() / "repos/anomalica/ingests/store"


def _member_key(m: dict) -> tuple[str, str]:
    return (m.get("variant", ""), m.get("claim_id", ""))


def collect_pairs(store: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(positives, negatives) as (textA, textB) pairs, pooled over every gold
    sidecar. Positive = two claims the reviewer put in one cluster; negative =
    two judged claims in the same record NOT co-clustered. Both cross-variant."""
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []

    for sidecar in sorted(store.glob("*.audit.json")):
        try:
            gold = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if gold.get("schema") != SCHEMA:
            continue

        text_of = {_member_key(c): c.get("text", "") for c in gold.get("claims", [])}
        clustered: set[tuple[str, str]] = set()
        cluster_id: dict[tuple[str, str], str] = {}
        for cl in gold.get("clusters", []):
            keys = [_member_key(m) for m in cl.get("members", [])]
            for k in keys:
                clustered.add(k)
                cluster_id[k] = cl.get("gold_id", id(cl))
            for a, b in combinations(keys, 2):
                if a[0] != b[0] and text_of.get(a) and text_of.get(b):
                    positives.append((text_of[a], text_of[b]))

        # Negatives: every cross-variant pair of judged claims in this record that
        # the reviewer did NOT co-cluster. A claim in no cluster is its own island.
        judged = [k for k in text_of if text_of[k]]
        for a, b in combinations(judged, 2):
            if a[0] == b[0]:
                continue
            if cluster_id.get(a) is not None and cluster_id.get(a) == cluster_id.get(b):
                continue
            negatives.append((text_of[a], text_of[b]))

    return positives, negatives


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {
        "n": len(s),
        "min": round(s[0], 3),
        "median": round(s[len(s) // 2], 3),
        "max": round(s[-1], 3),
    }


def _best_cut(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """The threshold maximising balanced accuracy over the labelled pairs, and
    that accuracy. Scans candidate cuts at every observed cosine."""
    if not pos or not neg:
        return (DEFAULT_THRESHOLD, 0.0)
    best_cut, best_score = DEFAULT_THRESHOLD, -1.0
    for cut in sorted(set(pos + neg)):
        tpr = sum(1 for p in pos if p >= cut) / len(pos)
        tnr = sum(1 for n in neg if n < cut) / len(neg)
        score = (tpr + tnr) / 2
        if score > best_score:
            best_cut, best_score = cut, score
    return (round(best_cut, 3), round(best_score, 3))


def report(store: Path, threshold: float) -> int:
    positives, negatives = collect_pairs(store)
    if not positives:
        print(f"No human-linked cross-variant pairs in {store}.")
        print(
            "Nothing to calibrate yet - the threshold stays the measured default "
            f"({DEFAULT_THRESHOLD}). Re-run once a reviewer has linked claims."
        )
        return 0

    cache = EmbeddingCache()
    try:
        cache.warm({t for pair in positives + negatives for t in pair})
    except EmbeddingUnavailable as exc:
        print(f"Embedding endpoint unreachable: {exc}", file=sys.stderr)
        print("Start it with `just embed-service`, then re-run.", file=sys.stderr)
        return 2

    pos = [cache.similarity(a, b) for a, b in positives]
    neg = [cache.similarity(a, b) for a, b in negatives]
    print(f"space: {cache.model_id}")
    print(f"human-LINKED   (positives): {_stats(pos)}")
    print(f"human-SEPARATED (negatives): {_stats(neg)}")

    if pos and neg:
        separation = round(min(pos) - max(neg), 3)
        print(f"separation (min positive - max negative): {separation:+.3f}")
        if separation <= 0:
            print(
                "  OVERLAP: no single threshold cleanly separates. Some linked "
                "pairs score below some separated pairs - inspect those, they are "
                "either mis-links or the space failing on that content."
            )
        tp = sum(1 for p in pos if p >= threshold)
        fn = len(pos) - tp
        fp = sum(1 for n in neg if n >= threshold)
        print(
            f"at threshold {threshold}: links kept {tp}/{len(pos)} "
            f"(missed {fn}), false links {fp}/{len(neg)}"
        )
        cut, acc = _best_cut(pos, neg)
        print(f"best-separating cut on this gold: {cut} (balanced accuracy {acc})")
        print(
            "  Treat as advisory, not authoritative: single-link clustering chains, "
            "so bias to the HIGH side of this cut (a false singleton is recoverable, "
            "a false merge hides a disagreement)."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingests", default=None, help="ingests dir (has store/)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    return report(_ingests_store(args.ingests), args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
