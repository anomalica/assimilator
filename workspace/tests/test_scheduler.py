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

import yaml

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


def test_pending_ingest_excludes_already_ingested_and_lanes_by_type(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    ingest = [j for j in q["jobs"] if j["type"] == "ingest"]
    by_hash = {j["target"]["hash"]: j for j in ingest}
    assert set(by_hash) == {H3, H4}  # H1 is already in the store, excluded
    # Only audio/video belongs in the GPU lane; pdf is light-local eager.
    assert by_hash[H3]["lane"] == "gpu"  # .opus
    assert by_hash[H3]["drivers"][0]["value"] == "audio/video"
    assert by_hash[H4]["lane"] == "eager"  # .pdf


def test_web_and_ebook_dedup_via_source_hash_and_verification(tmp_path):
    # A web page (body-hashed record) and an ebook (verification-named source)
    # already ingested must NOT be re-listed as pending, despite their source
    # bytes hashing differently from their content_hash.
    ingests, digests, sources = _corpus(tmp_path)
    store = ingests / "store"
    web_src, ebook_src = "a" * 64, "b" * 64
    body_web, body_ebook = "c" * 64, "d" * 64
    (store / f"{body_web}.md").write_text(
        f"---\nsource_type: web\ncontent_hash: sha256:{body_web}\n"
        f"source_hash: sha256:{web_src}\n---\nbody\n"
    )
    (store / f"{body_ebook}.md").write_text(
        f"---\nsource_type: ebook\ncontent_hash: sha256:{body_ebook}\n---\nbody\n"
    )
    (store / f"{body_ebook}.verification.json").write_text(
        json.dumps({"sha256": ebook_src, "challenges": []})
    )
    (sources / f"{web_src}.html").write_text("raw html")
    (sources / f"{ebook_src}.epub").write_text("raw epub")

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    pending = {j["target"]["hash"] for j in q["jobs"] if j["type"] == "ingest"}
    assert web_src not in pending  # matched via frontmatter source_hash
    assert ebook_src not in pending  # matched via verification.json sha256


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


def _write_digest(digests, content_hash, version=None):
    rec = {"content_hash": "sha256:" + content_hash}
    if version is not None:
        rec["processing_version"] = version
    (digests / "records" / f"{content_hash[:20]}.yaml").write_text(
        yaml.safe_dump({"record": rec})
    )


def test_digest_job_for_digestible_not_yet_digested(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    digest = [j for j in q["jobs"] if j["type"] == "digest"]
    assert len(digest) == 1
    assert digest[0]["target"]["hash"] == H1
    assert digest[0]["trigger"] == "never_done"
    assert digest[0]["lane"] == "claude"
    assert digest[0]["value"] == round(1.0 + math.log1p(1), 3)  # H1's graph demand


def test_digest_dropped_when_current_digest_exists(tmp_path):
    # Credit safety: an already-digested record must NOT be re-enumerated as a
    # job, even when its store file carries a .v2 suffix the digest name lacks.
    # Match by content_hash, not filename stem.
    ingests, digests, sources = _corpus(tmp_path)
    _write_digest(digests, H1)  # record has no version -> missing-safe = current
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]


def test_digest_v2_suffix_does_not_defeat_completion(tmp_path):
    # The exact Bob Lazar bug: store file is {hash}.v2.md, digest is {slug}.yaml.
    # content_hash match must still drop it.
    ingests, digests, sources = _corpus(tmp_path)
    store = ingests / "store"
    (store / f"{H1}.md").unlink()
    (store / f"{H1}.v2.md").write_text(f"---\ncontent_hash: sha256:{H1}\n---\nbody\n")
    _write_digest(digests, H1)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]


def test_import_job_for_digest_not_in_graph(tmp_path):
    # A digest on disk whose record is not in the graph is a pending eager
    # import; one whose record IS in the graph (by id, even with a null
    # content_hash) is not.
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()  # graph record ids: r1, r2
    recs = digests / "records"
    (recs / "new.yaml").write_text(
        yaml.safe_dump(
            {"record": {"content_hash": "sha256:" + "e" * 64, "id": "r-new"}}
        )
    )
    (recs / "old.yaml").write_text(
        yaml.safe_dump({"record": {"content_hash": "sha256:" + "f" * 64, "id": "r1"}})
    )
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    imp = {j["target"]["hash"] for j in q["jobs"] if j["type"] == "import"}
    assert "e" * 64 in imp  # r-new not in graph -> eager import job
    assert "f" * 64 not in imp  # r1 in graph by id -> already imported
    assert all(j["lane"] == "eager" for j in q["jobs"] if j["type"] == "import")


def test_nested_digest_still_detected_complete(tmp_path):
    # A slash in a record title nests the digest in a subdirectory; rglob must
    # still recognise it as complete, else the job re-dispatches forever.
    ingests, digests, sources = _corpus(tmp_path)
    sub = digests / "records" / "nested"
    sub.mkdir()
    (sub / "x.yaml").write_text(
        yaml.safe_dump({"record": {"content_hash": "sha256:" + H1}})
    )
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]


def test_digest_stale_when_body_version_differs(tmp_path):
    # A digest of an older body version re-appears as a 'stale' re-digest, not
    # 'never_done' and not dropped.
    ingests, digests, sources = _corpus(tmp_path)
    store = ingests / "store"
    (store / f"{H1}.md").write_text(
        f"---\ncontent_hash: sha256:{H1}\nprocessing:\n  version: new\n---\nbody\n"
    )
    _write_digest(digests, H1, version="old")
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    digest = [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]
    assert len(digest) == 1
    assert digest[0]["trigger"] == "stale"


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
