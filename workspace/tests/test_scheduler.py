"""Scheduler: real-state enumeration of pending pipeline jobs.

Builds a small synthetic corpus on disk + an in-memory graph and checks each
job type is enumerated from real state, ranked by its own driver, and emitted in
the workbench's consumer shape (camelCase, lanes claude|gpu|eager, jobs +
separate reviewQueue).
"""

from __future__ import annotations

import json
import math
import sqlite3

from assimilator import scheduler
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import Claim, Node, Record

H1 = "1" * 64  # an ingested+reviewed+digestible record
H2 = "2" * 64  # an ingested, never-reviewed record
H3 = "3" * 64  # a source awaiting ingestion
H4 = "4" * 64  # a second pending source


def _corpus(tmp_path):
    store = tmp_path / "ingests" / "store"
    store.mkdir(parents=True)
    (store / f"{H1}.md").write_text("---\ncontent_hash: sha256:" + H1 + "\n---\nbody\n")
    (store / f"{H2}.md").write_text("---\ncontent_hash: sha256:" + H2 + "\n---\nbody\n")
    (store / f"{H1}.review.json").write_text(
        json.dumps({"schema": "anomalica/review-coverage/1", "digestible": True})
    )
    (tmp_path / "ingests" / "records").mkdir()
    (tmp_path / "digests" / "records").mkdir(parents=True)
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / f"{H1}.html").write_text("already ingested")  # H1 is in the store
    (sources / f"{H3}.opus").write_text("pending audio")
    (sources / f"{H4}.pdf").write_text("pending pdf")
    return (
        tmp_path / "ingests",
        tmp_path / "digests",
        sources,
    )


def _graph_with_shared_node() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1", content_hash="sha256:" + H1))
    insert_record(conn, Record(id="r2", title="R2", content_hash="sha256:" + H2))
    n = insert_node(conn, Node(id="n1", node_type="person", name="Shared Person"))
    insert_claim(
        conn,
        Claim(
            id="c1",
            content="a",
            claim_type="testimony",
            record_id="r1",
            node_references=[n.id],
        ),
    )
    insert_claim(
        conn,
        Claim(
            id="c2",
            content="b",
            claim_type="testimony",
            record_id="r2",
            node_references=[n.id],
        ),
    )
    conn.commit()
    return conn


def test_pending_ingest_excludes_already_ingested(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    ingest = [j for j in q["jobs"] if j["type"] == "ingest"]
    hashes = {j["target"]["hash"] for j in ingest}
    assert hashes == {H3, H4}  # H1 is already in the store, excluded
    assert all(j["lane"] == "gpu" for j in ingest)
    audio = next(j for j in ingest if j["target"]["hash"] == H3)
    assert audio["drivers"][0]["value"] == "audio/video"


def test_review_queue_excludes_reviewed_and_ranks_by_demand(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    hashes = {it["target"]["hash"] for it in q["reviewQueue"]}
    assert H2 in hashes  # never reviewed
    assert H1 not in hashes  # has a review sidecar
    # H2 is in the graph sharing a node with H1, so it carries real demand.
    h2 = next(it for it in q["reviewQueue"] if it["target"]["hash"] == H2)
    assert h2["demand"] == round(1.0 + math.log1p(1), 3)


def test_digest_job_for_digestible_not_yet_digested(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    digest = [j for j in q["jobs"] if j["type"] == "digest"]
    assert len(digest) == 1
    assert digest[0]["target"]["hash"] == H1
    assert digest[0]["lane"] == "claude"
    assert digest[0]["value"] == round(1.0 + math.log1p(1), 3)  # H1's graph demand


def test_corroborate_blocked_without_embeddings(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()  # has claims, no vec_claims table
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    embed = [j for j in q["jobs"] if j["type"] == "embed"]
    corr = [j for j in q["jobs"] if j["type"] == "corroborate"]
    assert embed and embed[0]["lane"] == "eager"
    assert corr and corr[0]["status"] == "blocked"
    assert corr[0]["blocker"] == "embed:claims"


def test_demand_map_keyed_by_bare_hash(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert q["recordDemand"][H1] == round(1.0 + math.log1p(1), 3)
    assert q["recordDemand"][H2] == round(1.0 + math.log1p(1), 3)


def test_output_shape_matches_workbench_contract(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "2026-06-20T00:00:00Z")
    assert set(q) >= {"schema", "generatedAt", "jobs", "reviewQueue", "recordDemand"}
    assert q["generatedAt"] == "2026-06-20T00:00:00Z"
    for job in q["jobs"]:
        assert set(job) >= {"id", "type", "lane", "target", "status", "trigger"}
        assert job["lane"] in {"claude", "gpu", "eager"}
        assert job["status"] in {"eligible", "blocked", "readiness_gated"}
        assert set(job["target"]) >= {"kind", "label"}
