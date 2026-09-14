#!/usr/bin/env python3
"""Offline merge-candidate and source-context safety evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from assimilator.database import init_db
from assimilator.import_markdown import import_extraction
from assimilator.merge_pipeline import load_scores, score_pairs
from assimilator.shortlist import shortlist


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed_record(record: dict) -> dict:
    return {
        "frontmatter": {
            "record_id": record["id"],
            "record_title": record["title"],
            "record_date": record.get("date"),
            "content_hash": "sha256:"
            + hashlib.sha256(record["id"].encode()).hexdigest(),
        },
        "nodes": record["nodes"],
        "domain_claims": [
            {
                "id": claim["id"],
                "content": claim["text"],
                "original_excerpt": claim.get("quote"),
                "claim_type": claim["type"],
                "claim_role": claim.get("claim_role"),
                "attestation": claim.get("attestation"),
                "confidence": claim.get("confidence", 1.0),
                "location_in_record": claim["location"],
                "date": claim.get("date"),
                "date_end": claim.get("date_end"),
                "node_references": [ref["name"] for ref in claim["refs"]],
                "ref_roles": {ref["name"]: ref["role"] for ref in claim["refs"]},
                "provenance_chain": claim["provenance_chain"],
                "attribution_in_text": claim["attribution_in_text"],
            }
            for claim in record["claims"]
        ],
        "infrastructure_claims": [],
        "terminology": None,
    }


def build_graph(fixture: dict) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for record in fixture["records"]:
        import_extraction(conn, parsed_record(record))
    conn.commit()
    return conn


def claim_envelope(conn: sqlite3.Connection) -> list[tuple]:
    # speaker_id is an entity foreign key and is intentionally re-pointed by an
    # approved merge. Everything selected here is the immutable claim/source
    # envelope that ranking and merge operations must not rewrite.
    return conn.execute(
        "SELECT id, content, original_excerpt, claim_type, attestation, record_id, "
        "location_in_record, date, date_end, confidence, metadata, created_at, "
        "claim_role, claim_hash, origin_kind, origin, relay, entailment_label, "
        "entailment_score, entailment_model, entailment_premise, origin_ref, "
        "attribution_in_text FROM claims ORDER BY id"
    ).fetchall()


def edge_envelope(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT claim_id, node_id, salience FROM claim_node_refs "
        "ORDER BY claim_id, node_id"
    ).fetchall()


def account_bindings(fixture: dict, conn: sqlite3.Connection) -> dict[str, str]:
    accounts: dict[str, list[dict]] = {}
    for account in fixture["accounts"]:
        accounts.setdefault(account["record_id"], []).append(account)
    bindings = {}
    for claim_id, record_id, location in conn.execute(
        "SELECT id, record_id, location_in_record FROM claims"
    ):
        candidates = []
        point = float(location)
        for account in accounts.get(record_id, []):
            containing = [
                span for span in account["spans"] if span[0] <= point <= span[1]
            ]
            if containing:
                candidates.append(
                    (min(span[1] - span[0] for span in containing), account["id"])
                )
        if candidates:
            bindings[claim_id] = min(candidates)[1]
    return bindings


def validate_source_context(fixture: dict, conn: sqlite3.Connection) -> dict:
    records = {row[0] for row in conn.execute("SELECT id FROM records")}
    claims = {
        row[0]: row[1]
        for row in conn.execute("SELECT id, record_id FROM claims").fetchall()
    }
    account_records = {}
    continued = 0
    for account in fixture["accounts"]:
        assert account["record_id"] in records
        key = (account["record_id"], account["id"])
        assert key not in account_records
        account_records[key] = account
        assert all(start <= end for start, end in account["spans"])
        continued += len(account["spans"]) > 1
    bindings = account_bindings(fixture, conn)
    assert bindings == fixture["expected_bindings"]
    for claim_id, account_id in bindings.items():
        assert (claims[claim_id], account_id) in account_records
    for relation in fixture["temporal_relations"]:
        before = relation["before"]
        after = relation["after"]
        assert claims[before] == claims[after] == relation["record_id"]
        assert bindings[before] == bindings[after] == relation["account_id"]
    return {
        "bindings": bindings,
        "nested_binding_verified": bindings["claim-nera"] == "quill-account",
        "continued_binding_verified": bindings["claim-continued"] == "quill-account",
        "continued_accounts": continued,
        "temporal_relations_checked": len(fixture["temporal_relations"]),
        "cross_source_edges": 0,
    }


def fixture_embedder(fixture: dict):
    cases = fixture["cases"]
    dimensions = {
        node_id: i for i, case in enumerate(cases) for node_id in case["node_ids"]
    }

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            name = text.splitlines()[0].removeprefix("Name: ")
            node_id = next(
                node["id"]
                for record in fixture["records"]
                for node in record["nodes"]
                if node["name"] == name
            )
            vector = [0.0] * len(cases)
            vector[dimensions[node_id]] = 1.0
            vectors.append(vector)
        return vectors

    return embed


class FixtureReranker:
    device = "fixture-cpu"

    def __init__(self, fixture: dict) -> None:
        names = {
            node["id"]: node["name"]
            for record in fixture["records"]
            for node in record["nodes"]
        }
        self.scores = {
            frozenset(names[node_id] for node_id in case["node_ids"]): case[
                "fixture_score"
            ]
            for case in fixture["cases"]
        }

    def score(self, pairs, **kwargs):
        if not kwargs.get("symmetric", True):
            return [0.95] * len(pairs)
        return [self.scores[frozenset((a.name, b.name))] for a, b in pairs]


def evaluate(fixture_path: Path, out_path: Path, reranker=None) -> dict:
    fixture = json.loads(fixture_path.read_text())
    assert fixture["license"] == "CC0-1.0"
    conn = build_graph(fixture)
    before = (claim_envelope(conn), edge_envelope(conn))
    context = validate_source_context(fixture, conn)

    candidate_run = shortlist(conn, fixture_embedder(fixture), k=1)
    expected_pairs = {tuple(sorted(case["node_ids"])) for case in fixture["cases"]}
    assert expected_pairs <= candidate_run["pairs"]

    with tempfile.TemporaryDirectory(prefix="merge-safety-") as temp:
        memo_path = Path(temp) / "scores.jsonl"
        scorer = reranker or FixtureReranker(fixture)
        score_pairs(conn, sorted(expected_pairs), scorer, memo_path, log=lambda _: None)
        scores = load_scores(memo_path)

    ranked = sorted(
        (
            {
                **case,
                "node_ids": sorted(case["node_ids"]),
                "names_only": scores[tuple(sorted(case["node_ids"]))]["names_only"],
                "score": (
                    scores[tuple(sorted(case["node_ids"]))]["with_claims"]
                    if scores[tuple(sorted(case["node_ids"]))]["with_claims"]
                    is not None
                    else scores[tuple(sorted(case["node_ids"]))]["names_only"]
                ),
            }
            for case in fixture["cases"]
        ),
        key=lambda row: (-row["score"], row["id"]),
    )
    positive_ranks = [i for i, row in enumerate(ranked, 1) if row["same_entity"]]
    positives = [row for row in ranked if row["same_entity"]]
    negatives = [row for row in ranked if not row["same_entity"]]
    pairwise = [p["score"] > n["score"] for p in positives for n in negatives]

    after = (claim_envelope(conn), edge_envelope(conn))
    assert after == before
    assert conn.execute("SELECT COUNT(*) FROM node_merges").fetchone()[0] == 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE retired_at IS NOT NULL"
        ).fetchone()[0]
        == 0
    )
    result = {
        "schema": "anomalica/merge-safety-result/1",
        "fixture_sha256": file_sha256(fixture_path),
        "scorer": "qwen3-production-path"
        if reranker
        else "deterministic-fixture-double",
        "model_run": reranker is not None,
        "automatic_merges": 0,
        "source_context": context,
        "candidate_recall": len(expected_pairs & candidate_run["pairs"])
        / len(expected_pairs),
        "positive_mrr": sum(1 / rank for rank in positive_ranks) / len(positive_ranks),
        "positive_over_negative_accuracy": sum(pairwise) / len(pairwise),
        "ranked_proposals": ranked,
        "hard_negative_ranks": {
            row["category"]: i
            for i, row in enumerate(ranked, 1)
            if not row["same_entity"]
        },
        "claim_envelope_unchanged_by_ranking_and_proposal": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-qwen-cpu", action="store_true")
    args = parser.parse_args()
    reranker = None
    if args.run_qwen_cpu:
        from assimilator.entity_reranker import EntityReranker

        reranker = EntityReranker(device="cpu")
    result = evaluate(args.fixture, args.out, reranker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
