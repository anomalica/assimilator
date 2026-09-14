#!/usr/bin/env python3
"""Freeze current hybrid-search candidates and graph-derived relevance labels.

Run in the assimilator development container, where the production embedding
model and sqlite-vec extension are available. The resulting fixture contains
project-authored atomic claims only, not source excerpts or bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from assimilator.embeddings import EMBEDDING_MODEL_ID
from assimilator.search import KEYWORD_BOOST, RRF_K, hybrid_search_claims


QUERIES = [
    (
        "nimitz-encounter",
        "carrier pilots encounter a white capsule-shaped craft over the Pacific",
        "ae7507ac-dcb4-4689-bb0e-7c2e8ab4c5ac",
    ),
    (
        "roswell-incident",
        "1947 New Mexico debris and the military balloon explanation",
        "b8ea9860-ae6e-4835-8114-6a6626f28797",
    ),
    (
        "aatip",
        "Pentagon programme that investigated anomalous aerospace threats before public disclosure",
        "6c1618c8-50db-42c9-860e-6d736ef280bf",
    ),
    (
        "aaro",
        "defence office collecting and resolving anomaly reports across every domain",
        "e79e207c-a848-4eb7-9246-07587b20d2c1",
    ),
    (
        "remote-viewing",
        "mental technique for describing a distant hidden target",
        "41c9281c-4a11-4399-b324-a8486a96c766",
    ),
    (
        "face-on-mars",
        "mesa-like formation photographed by Viking at Cydonia",
        "5f813dc9-a873-428a-99ba-88d735d02699",
    ),
    (
        "alien-abduction",
        "accounts of people being taken by nonhuman visitors",
        "98eeed83-96a9-45ef-bb14-cb915d45fc9c",
    ),
    (
        "non-human-intelligence",
        "claims about intelligent beings that do not originate from humanity",
        "9c74d864-957a-4654-a5b6-1249fe654ec7",
    ),
    (
        "david-grusch",
        "former intelligence official who alleged crash retrievals and biologics",
        "d4241f5a-1063-4970-ba02-5a391c02d96d",
    ),
    (
        "luis-elizondo",
        "former Pentagon official associated with the secret aerospace threat programme",
        "87788ebc-018b-47aa-a6e2-a7773c5a3d1d",
    ),
    (
        "james-mcdonald",
        "James McDonald atmospheric physicist UFO investigations in the 1960s",
        "46886954-19d1-4d21-861a-9e653fb096a3",
    ),
    (
        "robert-monroe",
        "consciousness researcher associated with out-of-body travel and the Gateway method",
        "39bc257e-6dcc-4c45-b021-a456625771cb",
    ),
    (
        "jacques-vallee",
        "French computer scientist who argued the phenomenon may be a control system",
        "06412ca6-a1b6-411b-9e8b-2100b8c961bd",
    ),
    (
        "nasa",
        "United States civilian space agency investigations and public statements",
        "5e47c787-005b-4c91-ba4b-c9bec48146c9",
    ),
    (
        "cia",
        "American foreign intelligence agency involvement in psychic research and UFO records",
        "38b84d4a-96df-4684-bb46-f6f64f392add",
    ),
    (
        "us-congress",
        "lawmakers holding hearings and seeking disclosure about anomalous phenomena",
        "4fb65b2c-296a-493d-bbc7-65e4b763c1f0",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=50)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    query_rows = []
    try:
        graph_version = conn.execute("SELECT MAX(created_at) FROM claims").fetchone()[0]
        for query_id, text, target_node_id in QUERIES:
            node = conn.execute(
                "SELECT name, node_type FROM nodes WHERE id = ? AND retired_at IS NULL",
                (target_node_id,),
            ).fetchone()
            if node is None:
                raise RuntimeError(f"missing live target node: {target_node_id}")

            ranked = hybrid_search_claims(conn, text, limit=args.candidates)
            claim_ids = [claim_id for claim_id, _distance in ranked]
            placeholders = ",".join("?" for _ in claim_ids)
            claims = {
                row[0]: row[1]
                for row in conn.execute(
                    f"SELECT id, content FROM claims WHERE id IN ({placeholders})",
                    claim_ids,
                )
            }
            relevant = {
                row[0]
                for row in conn.execute(
                    "SELECT claim_id FROM claim_node_refs WHERE node_id = ? "
                    "UNION SELECT id FROM claims WHERE speaker_id = ?",
                    (target_node_id, target_node_id),
                )
            }
            candidates = [
                {
                    "claim_id": claim_id,
                    "text": claims[claim_id],
                    "relevance": int(claim_id in relevant),
                    "baseline_rank": rank,
                }
                for rank, claim_id in enumerate(claim_ids, 1)
            ]
            relevant_in_pool = sum(item["relevance"] for item in candidates)
            if relevant_in_pool == 0:
                raise RuntimeError(f"query has no relevant candidate: {query_id}")
            query_rows.append(
                {
                    "id": query_id,
                    "query": text,
                    "target": {
                        "node_id": target_node_id,
                        "name": node[0],
                        "node_type": node[1],
                    },
                    "total_graph_relevant": len(relevant),
                    "candidates": candidates,
                }
            )
    finally:
        conn.close()

    fixture = {
        "schema": "anomalica/search-reranker-benchmark/1",
        "license": "CC0-1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "kind": "reviewed-derived-graph",
            "graph_version": graph_version,
            "database_sha256": hashlib.sha256(args.db.read_bytes()).hexdigest(),
            "relevance_rule": "claim_node_refs target membership or target speaker_id",
            "text_policy": "project-authored atomic claim content only; no source excerpts or bodies",
        },
        "retrieval": {
            "method": "current hybrid search without reranking",
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "candidate_count": args.candidates,
            "rrf_k": RRF_K,
            "keyword_boost": KEYWORD_BOOST,
        },
        "queries": query_rows,
    }
    args.out.write_text(json.dumps(fixture, indent=2, ensure_ascii=True) + "\n")
    print(
        f"wrote {len(query_rows)} queries and "
        f"{sum(len(q['candidates']) for q in query_rows)} query-candidates"
    )


if __name__ == "__main__":
    main()
