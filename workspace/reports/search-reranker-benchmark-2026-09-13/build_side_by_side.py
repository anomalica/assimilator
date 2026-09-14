#!/usr/bin/env python3
"""Build the report-only Workbench view from frozen controlled results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


SCHEMA = "anomalica/search-reranker-side-by-side/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim_sources(digests_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    claims: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}
    for path in sorted(digests_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        record = doc.get("record") or {}
        source = {
            "id": record.get("id"),
            "content_hash": record.get("content_hash"),
            "title": record.get("title"),
            "friendly_name": path.stem,
            "copyright_status_snapshot": record.get("copyright_status"),
            "resolution": "current_canonical_digest",
        }
        found = False
        for section in ("domain_claims", "infrastructure_claims"):
            for claim in doc.get(section) or []:
                claim_id = claim.get("id")
                if claim_id:
                    if claim_id in claims:
                        raise ValueError(f"duplicate claim id {claim_id}")
                    claims[claim_id] = source
                    found = True
        if found:
            source_hashes[path.name] = _sha256(path)
    return claims, source_hashes


def _ranked_view(
    top_ten: list[dict],
    other_top_ten: list[dict],
    candidates: dict[str, dict],
    sources: dict[str, dict],
    model: str,
) -> list[dict]:
    other_ranks = {row["claim_id"]: row["rank"] for row in other_top_ten}
    rows = []
    for result in top_ten:
        claim_id = result["claim_id"]
        candidate = candidates[claim_id]
        other_rank = other_ranks.get(claim_id)
        if model == "minilm":
            minilm_rank, granite_rank = result["rank"], other_rank
            movement = "exited_top_10" if other_rank is None else "shared_top_10"
        else:
            minilm_rank, granite_rank = other_rank, result["rank"]
            movement = "entered_top_10" if other_rank is None else "shared_top_10"
        rows.append(
            {
                "rank": result["rank"],
                "claim_id": claim_id,
                "text": candidate["text"],
                "graph_relevance": candidate["relevance"],
                "score": result["score"],
                "source_record": sources[claim_id],
                "minilm_rank": minilm_rank,
                "granite_rank": granite_rank,
                "granite_minus_minilm": (
                    granite_rank - minilm_rank
                    if minilm_rank is not None and granite_rank is not None
                    else None
                ),
                "movement": movement,
            }
        )
    return rows


def build(
    fixture_path: Path, minilm_path: Path, granite_path: Path, digests_dir: Path
) -> dict:
    fixture = json.loads(fixture_path.read_text())
    minilm = json.loads(minilm_path.read_text())
    granite = json.loads(granite_path.read_text())
    fixture_hash = _sha256(fixture_path)
    for result in (minilm, granite):
        if result.get("fixture", {}).get("sha256") != fixture_hash:
            raise ValueError("controlled result does not match fixture")

    sources, source_hashes = _claim_sources(digests_dir)
    fixture_queries = {row["id"]: row for row in fixture["queries"]}
    minilm_queries = {row["id"]: row for row in minilm["queries"]}
    granite_queries = {row["id"]: row for row in granite["queries"]}
    if set(fixture_queries) != set(minilm_queries) or set(fixture_queries) != set(
        granite_queries
    ):
        raise ValueError("query sets differ")

    queries = []
    used_sources: set[str] = set()
    for query in fixture["queries"]:
        query_id = query["id"]
        candidates = {row["claim_id"]: row for row in query["candidates"]}
        mini_top = minilm_queries[query_id]["top_10"]
        granite_top = granite_queries[query_id]["top_10"]
        ranked_ids = {row["claim_id"] for row in mini_top + granite_top}
        missing = sorted(ranked_ids - sources.keys())
        if missing:
            raise ValueError(f"claims not found in canonical digests: {missing}")
        for claim_id in ranked_ids:
            used_sources.add(sources[claim_id]["friendly_name"] + ".yaml")
        for result in mini_top + granite_top:
            if candidates[result["claim_id"]]["relevance"] != result["relevance"]:
                raise ValueError(f"relevance drift for {result['claim_id']}")
        queries.append(
            {
                "id": query_id,
                "query": query["query"],
                "target": query["target"],
                "relevance": {
                    "rule": fixture["source"]["relevance_rule"],
                    "total_graph_relevant": query["total_graph_relevant"],
                },
                "rankings": {
                    "minilm": _ranked_view(
                        mini_top, granite_top, candidates, sources, "minilm"
                    ),
                    "granite": _ranked_view(
                        granite_top, mini_top, candidates, sources, "granite"
                    ),
                },
            }
        )

    source_manifest = {name: source_hashes[name] for name in sorted(used_sources)}
    return {
        "schema": SCHEMA,
        "report_only": True,
        "canonical_activation": "forbidden",
        "license": fixture["license"],
        "inputs": {
            "fixture": {"path": fixture_path.name, "sha256": fixture_hash},
            "minilm": {"path": minilm_path.name, "sha256": _sha256(minilm_path)},
            "granite": {"path": granite_path.name, "sha256": _sha256(granite_path)},
            "source_identity": {
                "resolution": "current_canonical_digest",
                "digest_count": len(source_manifest),
                "manifest": source_manifest,
            },
        },
        "text_policy": fixture["source"]["text_policy"],
        "limitations": [
            "The fixture contains text-only claims and graph-derived relevance; source-local accounts were absent.",
            "The benchmark did not evaluate entity merge safety, merge proposals, automatic merges, or temporal relations.",
            "Controlled results retain only top-10 ranks; a claim absent from one list has no exact off-list rank.",
            "Source-record identity was resolved from current canonical digests, not the frozen benchmark database snapshot.",
            "Source bodies, excerpts, images, and originals are excluded.",
        ],
        "models": {"minilm": minilm["model"], "granite": granite["model"]},
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--granite", type=Path, required=True)
    parser.add_argument("--digests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.fixture, args.minilm, args.granite, args.digests)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
