"""Scheduler: real-state enumeration of pending pipeline jobs.

Builds a small synthetic corpus on disk + an in-memory graph and checks each
job type is enumerated from real state, ranked by its own driver, and emitted in
the workbench's consumer shape (camelCase, lanes claude|gpu|eager, jobs +
separate reviewQueue).
"""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3

import pytest
import yaml

from assimilator import scheduler
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from anomalica_common.digest.models import Claim, Node, NodeType, Record

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
    (tmp_path / "ingests" / "by-name").mkdir()
    (tmp_path / "ingests" / "by-name" / f"{H1}.md").symlink_to(store / f"{H1}.md")
    (tmp_path / "ingests" / "by-name" / f"{H2}.md").symlink_to(store / f"{H2}.md")
    (tmp_path / "digests").mkdir(parents=True)
    (tmp_path / "digests" / "digest-generation.json").write_text(
        json.dumps(
            {
                "schema": "anomalica/digest-generation/1",
                "current_generation": 1,
            }
        )
    )
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


def _write_digest(digests, content_hash, version=None, generation=1):
    from anomalica_common.pre_digest import materialise, pre_digest_hash

    rec = {"content_hash": "sha256:" + content_hash}
    if version is not None:
        rec["processing_version"] = version
    record_path = next(
        (digests.parent / "ingests" / "store").glob(f"{content_hash}*.md")
    )
    raw = record_path.read_text()
    configuration = {
        "configuration_schema": "anomalica/digest-extraction-config/1",
        "fixture": "scheduler",
    }
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    config_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    (digests / "extraction-configurations.json").write_text(
        json.dumps(
            {
                "schema": "anomalica/digest-extraction-config-registry/1",
                "configurations": {config_hash: configuration},
            }
        )
    )
    document = {
        "schema": "anomalica/digest/1",
        "pre_digest": {
            "sha256": pre_digest_hash(materialise(scheduler._record_body(raw))),
            "prep_version": 7,
        },
        "extraction_config": config_hash,
        "record": rec,
    }
    if generation is not None:
        document["extraction_generation"] = generation
    (digests / f"{content_hash[:20]}.yaml").write_text(yaml.safe_dump(document))


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
    assert any(
        group["boundary"] == "record-generation"
        and "generation_unknown" in group["local_reasons"]
        for group in digest[0]["inherited_reason_groups"]
    )


def test_digest_dropped_when_current_digest_exists(tmp_path):
    # Credit safety: an already-digested record must NOT be re-enumerated as a
    # job, even when its store file carries a .v2 suffix the digest name lacks.
    # Match by content_hash, not filename stem.
    ingests, digests, sources = _corpus(tmp_path)
    _write_digest(digests, H1)
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
    by_name = ingests / "by-name" / f"{H1}.md"
    by_name.unlink()
    by_name.symlink_to(store / f"{H1}.v2.md")
    _write_digest(digests, H1)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]


def test_legacy_mapping_extraction_config_is_invalid_without_crashing(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    _write_digest(digests, H1)
    path = digests / f"{H1[:20]}.yaml"
    document = yaml.safe_load(path.read_text())
    document["extraction_config"] = {
        "prompts": [{"id": "claims", "version": "legacy"}],
        "model": "legacy-model",
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    queue = scheduler.build_queue(
        _graph_with_shared_node(), ingests, digests, sources, "T"
    )
    job = next(
        job
        for job in queue["jobs"]
        if job["type"] == "digest" and job["target"]["hash"] == H1
    )
    assert any(
        "extraction_config_invalid" in group["local_reasons"]
        for group in job["local_reason_groups"]
    )


def test_record_generation_uses_explicit_manifest_and_keeps_unknowns(tmp_path):
    ingests = tmp_path / "ingests"
    store_dir = ingests / "store"
    store_dir.mkdir(parents=True)
    (store_dir / "_pipeline_versions.yaml").write_text("web: 7\npdf: 1\n")
    cases = {
        "1" * 64: ("web", 7, []),
        "2" * 64: ("web", 6, ["generation_behind"]),
        "3" * 64: ("web", 8, ["generation_unknown"]),
        "4" * 64: ("image", 1, ["generation_unknown"]),
        "5" * 64: ("pdf", "one", ["generation_unknown"]),
        "6" * 64: ("pdf", None, ["generation_unknown"]),
    }
    for content_hash, (source_type, generation, _reasons) in cases.items():
        processing = (
            f"processing:\n  pipeline_version: {generation}\n"
            if generation is not None
            else ""
        )
        (store_dir / f"{content_hash}.md").write_text(
            "---\n"
            "schema: anomalica/record/1\n"
            f"source_type: {source_type}\n"
            f"content_hash: sha256:{content_hash}\n"
            f"{processing}---\nbody\n"
        )

    groups, metrics = scheduler.record_generation_freshness(
        ingests, scheduler._store_records(ingests)
    )
    for content_hash, (_source_type, _generation, reasons) in cases.items():
        actual = groups[content_hash]
        assert ([*actual[0]["local_reasons"]] if actual else []) == reasons
    assert metrics["generation_distance"] == {"sha256:" + "2" * 64: 1}


def test_emitted_freshness_manifest_binds_queue_and_preserves_real_groups(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    queue = scheduler.build_queue(
        _graph_with_shared_node(), ingests, digests, sources, "2026-09-14T00:00:00Z"
    )
    queue_path = tmp_path / "schedule.json"
    freshness_path = tmp_path / "deployment-freshness.json"

    scheduler.write_queue(queue, queue_path)
    manifest_sha256 = scheduler.write_freshness_manifest(
        queue, queue_path, freshness_path
    )
    manifest = json.loads(freshness_path.read_text())

    assert manifest["schema"] == "anomalica-freshness/v1"
    assert manifest["generated_at"] == "2026-09-14T00:00:00Z"
    assert (
        manifest["source_queue_sha256"]
        == hashlib.sha256(queue_path.read_bytes()).hexdigest()
    )
    assert manifest_sha256 == hashlib.sha256(freshness_path.read_bytes()).hexdigest()
    groups = {
        (group["boundary"], group["artifact"]): group for group in manifest["groups"]
    }
    assert ("record-generation", f"sha256:{H1}") in groups
    assert ("digest-input", f"sha256:{H1}") in groups
    assert (
        "generation_unknown"
        in groups[("record-generation", f"sha256:{H1}")]["local_reasons"]
    )
    assert groups[("digest-input", f"sha256:{H1}")]["local_reasons"] == [
        "digest_missing"
    ]
    assert all(group["inherited"] == [] for group in manifest["groups"])
    assert list(groups) == sorted(groups)


def test_run_schedule_writes_adjacent_freshness_manifest(tmp_path, monkeypatch):
    ingests, digests, sources = _corpus(tmp_path)
    db_path = tmp_path / "knowledge.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    conn.close()
    monkeypatch.setenv("ANOMALICA_BRIEFS_DIR", str(tmp_path / "briefs"))
    monkeypatch.setenv("ANOMALICA_CONTENT_DIR", str(tmp_path / "content"))
    queue_path = tmp_path / "schedule.json"

    queue, written_path = scheduler.run_schedule(
        db_path,
        str(ingests),
        str(digests),
        str(sources),
        str(queue_path),
    )

    freshness_path = scheduler.default_freshness_path(queue_path)
    assert written_path == queue_path
    assert json.loads(queue_path.read_text()) == queue
    manifest = json.loads(freshness_path.read_text())
    assert (
        manifest["source_queue_sha256"]
        == hashlib.sha256(queue_path.read_bytes()).hexdigest()
    )


def test_freshness_manifest_rejects_a_different_source_queue(tmp_path):
    queue = {"generatedAt": "T", "jobs": []}
    queue_path = tmp_path / "schedule.json"
    queue_path.write_text("{}")

    with pytest.raises(ValueError, match="source queue bytes"):
        scheduler.write_freshness_manifest(
            queue, queue_path, tmp_path / "freshness.json"
        )


def test_import_job_for_digest_not_in_graph(tmp_path):
    # A digest on disk whose record is not in the graph is a pending eager
    # import. A legacy graph row without an exact import receipt is also due.
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()  # graph record ids: r1, r2
    recs = digests
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
    assert "f" * 64 in imp  # r1 exists, but presence is not an exact-byte receipt
    legacy = next(j for j in q["jobs"] if j["target"]["hash"] == "f" * 64)
    assert legacy["trigger"] == "stale"
    assert legacy["drivers"] == [
        {"label": "freshness", "value": "import receipt missing"}
    ]
    assert any(
        group["boundary"] == "digest-generation"
        and group["local_reasons"] == ["generation_unknown"]
        for group in legacy["inherited_reason_groups"]
    )
    assert all(j["lane"] == "eager" for j in q["jobs"] if j["type"] == "import")


def test_superseded_source_excluded_from_ingest(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    (sources / "superseded.txt").write_text(f"{H3}\n# a comment line\n")
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    pending = {j["target"]["hash"] for j in q["jobs"] if j["type"] == "ingest"}
    assert H3 not in pending  # listed as superseded -> excluded
    assert H4 in pending  # still pending


def test_brief_freshness_is_local_and_detects_change_deletion_and_rename(tmp_path):
    from assimilator import synthesise

    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    conn.execute(
        "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
        "source_count, independent_source_count, subject_claims, status, computed_at) "
        "VALUES ('n1', 'person', 'page-worthy', 2, 2, NULL, 1, 'proposed', 'T')"
    )
    briefs = tmp_path / "briefs"
    synthesise.write_brief(synthesise.build_entity_brief(conn, "n1"), briefs)

    insert_record(conn, Record(id="r3", title="Unrelated", content_hash="sha256:33"))
    other = insert_node(conn, Node(id="n-other", node_type="topic", name="Other"))
    insert_claim(
        conn,
        Claim(
            id="c-other",
            content="unrelated graph mutation",
            claim_type="testimony",
            record_id="r3",
            node_references=[other.id],
        ),
        claim_hash="h-other",
    )
    conn.commit()
    queue = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs
    )
    assert not [
        j
        for j in queue["jobs"]
        if j["type"] == "synthesise" and j["local_reason_groups"]
    ]

    conn.execute(
        "UPDATE claims SET content = 'changed', claim_hash = 'changed' WHERE id = 'c1'"
    )
    conn.commit()
    queue = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs
    )
    assert [j["id"] for j in queue["jobs"] if j["type"] == "synthesise"] == [
        "synthesise:n1"
    ]
    synthesise.write_brief(synthesise.build_entity_brief(conn, "n1"), briefs)

    conn.execute("DELETE FROM claim_node_refs WHERE claim_id = 'c1'")
    conn.execute("DELETE FROM claims WHERE id = 'c1'")
    conn.commit()
    queue = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs
    )
    assert [j["id"] for j in queue["jobs"] if j["type"] == "synthesise"] == [
        "synthesise:n1"
    ]
    synthesise.write_brief(synthesise.build_entity_brief(conn, "n1"), briefs)

    conn.execute("UPDATE nodes SET name = 'Renamed Person' WHERE id = 'n1'")
    conn.commit()
    queue = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs
    )
    renamed = [j for j in queue["jobs"] if j["type"] == "synthesise"]
    assert [j["id"] for j in renamed] == ["synthesise:n1"]
    assert renamed[0]["trigger"] == "stale_brief"
    assert renamed[0]["target"]["href"] == "people/renamed-person"


def test_nested_digest_still_detected_complete(tmp_path):
    # A slash in a record title nests the digest in a subdirectory; rglob must
    # still recognise it as complete, else the job re-dispatches forever.
    ingests, digests, sources = _corpus(tmp_path)
    sub = digests / "nested"
    sub.mkdir()
    _write_digest(digests, H1)
    (digests / f"{H1[:20]}.yaml").rename(sub / "x.yaml")
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]


def test_digest_stale_when_body_version_differs(tmp_path):
    # processing.version is legacy metadata. Actual changed materialised input
    # makes a digest stale regardless of that value.
    ingests, digests, sources = _corpus(tmp_path)
    store = ingests / "store"
    _write_digest(digests, H1, version="old")
    (store / f"{H1}.md").write_text(
        f"---\ncontent_hash: sha256:{H1}\nprocessing:\n  version: old\n---\nchanged body\n"
    )
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    digest = [
        j for j in q["jobs"] if j["type"] == "digest" and j["target"]["hash"] == H1
    ]
    assert len(digest) == 1
    assert digest[0]["trigger"] == "stale"
    assert any(
        group["boundary"] == "digest-input"
        and group["local_reasons"] == ["pre_digest_hash_mismatch"]
        for group in digest[0]["local_reason_groups"]
    )


def _mark_embedded(conn, ids_by_kind):
    """Stamp rows as embedded in the current space, via the real (kind, id) table."""
    from assimilator.scheduler import _embedding_model_id

    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_model ("
        " kind TEXT NOT NULL, id TEXT NOT NULL, model_id TEXT NOT NULL,"
        " embedded_at TEXT NOT NULL, PRIMARY KEY (kind, id))"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO embedding_model VALUES (?, ?, ?, 'T')",
        [(kind, i, _embedding_model_id()) for kind, ids in ids_by_kind for i in ids],
    )
    conn.commit()


def _rows_by_kind(conn, predicate=lambda row_id: True):
    return [
        (kind, [r[0] for r in conn.execute(query) if predicate(r[0])])
        for kind, query in (
            ("claim", "SELECT id FROM claims"),
            ("node", "SELECT id FROM nodes"),
        )
    ]


def test_corroborate_blocked_while_any_embed_batch_is_outstanding(tmp_path):
    """Corroboration compares every claim against every other, so it needs the
    WHOLE corpus embedded - finishing the first batch must not release it. The
    blocker names the lowest outstanding batch, so the card points at real work
    rather than at a singleton job id that no longer exists."""
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()  # has claims, nothing embedded
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    embed = [j for j in q["jobs"] if j["type"] == "embed"]
    corr = [j for j in q["jobs"] if j["type"] == "corroborate"]

    assert embed and all(j["lane"] == "eager" for j in embed)
    assert corr and corr[0]["status"] == "blocked"
    assert corr[0]["blocker"] == min(j["id"] for j in embed)

    # Embed everything the lowest batch holds; corroborate stays blocked on the
    # next one.
    from assimilator.embed_batches import bucket_of

    first = min(int(j["id"].rsplit(":", 1)[1]) for j in embed)
    _mark_embedded(conn, _rows_by_kind(conn, lambda i: bucket_of(i) == first))

    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    corr = [j for j in q["jobs"] if j["type"] == "corroborate"]
    remaining = [j for j in q["jobs"] if j["type"] == "embed"]
    assert f"embed:claims:{first}" not in {j["id"] for j in remaining}
    assert corr[0]["status"] == "blocked"
    assert corr[0]["blocker"] == min(j["id"] for j in remaining)


def test_corroborate_is_released_once_every_batch_is_embedded(tmp_path):
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    _mark_embedded(conn, _rows_by_kind(conn))

    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    assert not [j for j in q["jobs"] if j["type"] == "embed"]
    corr = [j for j in q["jobs"] if j["type"] == "corroborate"]
    assert corr and corr[0]["status"] != "blocked"


def test_a_batch_card_shows_corpus_progress_not_a_bare_job_name(tmp_path):
    """A three-hour task split into batches is only legible if each card says
    where the corpus is up to; "batch 7 of 32" alone says nothing about how much
    is left."""
    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()
    q = scheduler.build_queue(conn, ingests, digests, sources, "T")
    job = next(j for j in q["jobs"] if j["type"] == "embed")

    drivers = {d["label"]: d["value"] for d in job["drivers"]}
    assert drivers["corpus progress"] == "0 of 3 embedded"  # 2 claims + 1 node
    assert sum(int(d["value"]) for d in _batch_sizes(q)) == 3
    # User-facing text says "vector embedding"; the machine-readable type stays
    # the terse internal name.
    assert job["target"]["label"].startswith("vector embedding, batch ")
    assert job["type"] == "embed"


def _batch_sizes(queue):
    return [
        d
        for j in queue["jobs"]
        if j["type"] == "embed"
        for d in j["drivers"]
        if d["label"] == "items in this batch"
    ]


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
        assert set(job) >= {
            "local_reason_groups",
            "inherited_reason_groups",
            "consequence",
            "native_metrics",
        }
        assert job["lane"] in {"claude", "gpu", "eager"}
        assert job["status"] in {"eligible", "blocked", "readiness_gated"}
        assert set(job["target"]) >= {"kind", "label"}


def test_article_audit_is_exact_by_section_slug_language_and_protected_body(tmp_path):
    content = tmp_path / "content"
    people = content / "people"
    events = content / "events"
    people.mkdir(parents=True)
    events.mkdir()
    brief = {
        "brief_hash": "brief-current",
        "page": {"node_type": "person", "slug": "same", "title": "Same"},
        "_claim_pairs": [("c1", "h1")],
        "_content_hashes": set(),
    }

    def article(path, language, claim_hash, model, protected_body, body):
        path.joinpath(f"same.{language}.md").write_text(
            "---\n"
            + yaml.safe_dump(
                {
                    "built_from": {
                        "brief_hash": "brief-current",
                        "claims": [{"id": "c1", "hash": claim_hash}],
                    },
                    "built_by": {"model": model, "body_sha256": protected_body},
                },
                sort_keys=False,
            )
            + f"---\n\n{body}\n"
        )

    current_body_hash = hashlib.sha256(b"current").hexdigest()
    article(people, "en", "h1", "new", current_body_hash, "current")
    article(events, "en", "wrong", "old", "wrong", "wrong section")
    article(people, "fr", "old", "old", "protected", "edited")

    jobs = scheduler.enumerate_assemble_jobs(
        [brief], content, current_generator={"model": "new"}
    )
    assert [job.id for job in jobs] == ["assemble:people/same:fr"]
    job = jobs[0].to_dict()
    assert job["article"] == "people/same.fr"
    assert job["native_metrics"] == {
        "article_citation_count": 1,
        "citations_missing": 0,
        "citation_hash_mismatches": 1,
        "generator_fields_compared": 1,
        "generator_fields_changed": 1,
        "body_hash_mismatch": 1,
    }
    assert job["local_reason_groups"][0]["local_reasons"] == [
        "body_modified",
        "citation_hash_mismatch",
        "generator_changed",
    ]


def test_superseded_records_are_not_scheduled(tmp_path):
    """A body-normalising fix rehashes a record and mints a successor while the
    original is deliberately retained, so a lookup by the old content_hash still
    resolves. Retained is not live: schedule work against it and the pipeline
    re-digests text it has already replaced, with no error to show for it."""
    from assimilator.scheduler import _store_records

    store = tmp_path / "store"
    store.mkdir()
    old, new = "a" * 64, "b" * 64
    (store / f"{old}.md").write_text(
        f"---\ntitle: Email\ncontent_hash: sha256:{old}\nsuperseded_by: {new}\n---\nbody"
    )
    (store / f"{new}.md").write_text(
        f"---\ntitle: Email\ncontent_hash: sha256:{new}\n---\nbody"
    )

    assert set(_store_records(tmp_path)) == {new}


def test_digest_index_accepts_either_the_root_or_the_records_dir(tmp_path):
    """The parameter is named for the digests ROOT and appends records/ itself,
    so passing records/ - the obvious thing to pass - yielded an EMPTY index
    rather than an error, and an empty index means zero import jobs and a graph
    that silently never catches up with the digests on disk."""
    from assimilator.scheduler import _digest_index

    (tmp_path / "a.yaml").write_text(
        "record:\n  content_hash: sha256:" + "a" * 64 + "\n  title: A\n"
    )

    assert len(_digest_index(tmp_path)) == 1


def test_the_scheduler_reads_the_same_vector_space_as_the_embedder():
    """scheduler is host-light and cannot import assimilator.embeddings, which
    pulls in fastembed - so the id is re-derived and must not drift."""
    from assimilator.embeddings import EMBEDDING_MODEL_ID
    from assimilator.scheduler import _embedding_model_id

    assert _embedding_model_id() == EMBEDDING_MODEL_ID


def test_a_model_comparison_variant_is_not_an_importable_digest(tmp_path):
    """digests/variants/ holds what each model emitted for a record. They carry a
    record.content_hash like any digest, so a recursive scan indexes them as
    importable - and five import jobs were emitted for records whose only
    artefact was a variant. The job can never succeed: the importer wants the
    canonical digest, and there is not one."""
    import yaml as _yaml

    digests = tmp_path / "digests"
    (digests / "variants" / "some-record").mkdir(parents=True)
    doc = {
        "schema": "anomalica/digest/1",
        "record": {
            "id": "r-variant",
            "title": "V",
            "content_hash": "sha256:" + "c" * 64,
        },
    }
    (digests / "variants" / "some-record" / "opus.yaml").write_text(
        _yaml.safe_dump(doc)
    )

    canonical = dict(doc)
    canonical["record"] = {
        **doc["record"],
        "id": "r-canonical",
        "content_hash": "sha256:" + "d" * 64,
    }
    (digests / "real.yaml").write_text(_yaml.safe_dump(canonical))

    index = scheduler._digest_index(digests)

    assert "d" * 64 in index, "the canonical digest must be indexed"
    assert "c" * 64 not in index, "a variant must never be offered as an import"


def test_graph_input_fingerprint_names_duplicate_live_bindings(tmp_path, monkeypatch):
    digests = tmp_path / "digests"
    digests.mkdir()
    duplicate = {
        "schema": "anomalica/digest/1",
        "record": {"id": "same-record", "content_hash": "sha256:" + "a" * 64},
    }
    (digests / "one.yaml").write_text(yaml.safe_dump(duplicate))
    (digests / "two.yaml").write_text(yaml.safe_dump(duplicate))
    curation = tmp_path / "curation"
    curation.mkdir()
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(curation))
    conn = sqlite3.connect(":memory:")
    init_db(conn)

    diagnostic = scheduler.graph_input_diagnostics(
        conn, scheduler._digest_index(digests), digests
    )
    expected_input = {
        "import_generation": 1,
        "digests": [],
        "curation_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
    }
    expected_bytes = json.dumps(
        expected_input, ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert (
        diagnostic["fingerprint"]
        == "sha256:" + hashlib.sha256(expected_bytes).hexdigest()
    )
    assert diagnostic["duplicate_binding_count"] == 2
    assert {row["kind"] for row in diagnostic["duplicate_bindings"]} == {
        "content_hash",
        "record_id",
    }


def test_the_digest_index_reads_only_the_record_header(tmp_path):
    """A digest runs to 14,000 lines and 1,800 claims; the index wants four
    header fields. Parsing every file in full cost 54 seconds of every queue
    rebuild - on its own enough to push the rebuild past the runner's 180s
    timeout, so the queue never refreshed and other components' work stayed
    invisible."""
    import yaml as _yaml

    digests = tmp_path / "digests"
    digests.mkdir()
    doc = {
        "schema": "anomalica/digest/1",
        "record": {
            "id": "r1",
            "title": "A Record",
            "content_hash": "sha256:" + "e" * 64,
            "processing_version": "abc123",
        },
        "nodes": [
            {"id": f"n{i}", "type": "person", "name": f"P{i}"} for i in range(400)
        ],
    }
    (digests / "big.yaml").write_text(_yaml.safe_dump(doc, sort_keys=False))

    index = scheduler._digest_index(digests)

    public = {
        key: value for key, value in index["e" * 64].items() if not key.startswith("_")
    }
    assert public == {
        "version": "abc123",
        "title": "A Record",
        "record_id": "r1",
        "schema": "anomalica/digest/1",
        "pre_digest": None,
        "extraction_generation": None,
        "extraction_config": None,
        "digest_path": "digests/big.yaml",
        "digest_sha256": index["e" * 64]["digest_sha256"],
    }
    assert index["e" * 64]["digest_sha256"].startswith("sha256:")


def test_a_digest_with_the_record_block_out_of_order_still_resolves(tmp_path):
    """The fast path stops at the next top-level key, so a file that puts record
    somewhere unexpected must fall back to a full parse rather than silently
    vanish from the index. Being slow beats being wrong about which digests
    exist."""
    digests = tmp_path / "digests"
    digests.mkdir()
    (digests / "odd.yaml").write_text(
        "schema: anomalica/digest/1\n"
        "nodes:\n"
        "  - id: n1\n"
        "    name: Someone\n"
        "record:\n"
        "  id: r2\n"
        "  title: Out Of Order\n"
        "  content_hash: sha256:" + "f" * 64 + "\n"
    )

    index = scheduler._digest_index(digests)

    assert index["f" * 64]["record_id"] == "r2"


def test_two_types_sharing_a_name_both_settle(tmp_path):
    """An event and a project both called "Apollo 14", both proposed. Their
    briefs shared one slug and so one FILE; the scheduler matched by node_id,
    found whichever node had not written last, re-emitted it, and the pair
    alternated forever - each round a full queue rebuild. With the brief path
    carrying the section, one emit settles both, and the two assemble jobs
    carry distinct ids (the id tail is the brief reference the runner hands to
    the assembler, so it must name the page, not the slug)."""
    from assimilator import synthesise

    ingests, digests, sources = _corpus(tmp_path)
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:" + H1))
    for nid, ntype in (("ev-1", "event"), ("pr-1", "project")):
        insert_node(conn, Node(id=nid, name="Apollo 14", node_type=ntype))
        for i in range(2):
            insert_claim(
                conn,
                Claim(
                    id=f"{nid}-c{i}",
                    content=f"claim {i}",
                    claim_type="testimony",
                    record_id="r1",
                    node_references=[nid],
                ),
            )
        conn.execute(
            "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
            "source_count, status, computed_at) VALUES (?, ?, 'page-worthy', 2, 1, "
            "'proposed', 'T')",
            (nid, ntype),
        )
    conn.commit()
    briefs = tmp_path / "briefs"

    q1 = scheduler.build_queue(conn, ingests, digests, sources, "T", briefs_dir=briefs)
    assert sorted(j["id"] for j in q1["jobs"] if j["type"] == "synthesise") == [
        "synthesise:ev-1",
        "synthesise:pr-1",
    ]

    synthesise.emit_all(conn, briefs)
    q2 = scheduler.build_queue(conn, ingests, digests, sources, "T", briefs_dir=briefs)

    assert not [
        j for j in q2["jobs"] if j["type"] == "synthesise" and j["local_reason_groups"]
    ]
    assert sorted(j["id"] for j in q2["jobs"] if j["type"] == "assemble") == [
        "assemble:events/apollo-14",
        "assemble:projects/apollo-14",
    ]


def test_the_proposal_staleness_test_excludes_what_the_proposer_excludes(
    tmp_path, monkeypatch
):
    """propose() writes the gate minus vetoes minus composed-page members. A
    staleness test that subtracts less can never agree with it, and the job it
    guards is emitted for ever - which is what happened from the first composed
    page: 12 runs, one per restart, permanently "waiting" on Mark's card."""
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    from assimilator.pages import append_compose_entry, apply_pages
    from assimilator.propose_pages import propose
    from assimilator.scheduler import _proposal_table_stale

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for i in range(3):
        insert_record(
            conn, Record(id=f"r{i}", title=f"R{i}", content_hash=f"sha256:a{i}")
        )
    for nid, name in (("uap", "UAP topic"), ("ufo", "UFO topic")):
        insert_node(conn, Node(id=nid, name=name, node_type=NodeType.topic))
        for i in range(9):
            insert_claim(
                conn,
                Claim(
                    id=f"{nid}-{i}",
                    content=f"claim {i} about {name}",
                    claim_type="testimony",
                    record_id=f"r{i % 3}",
                    node_references=[nid],
                ),
            )
    conn.commit()
    append_compose_entry(
        "UFOs / UAPs",
        "topic",
        [
            {"name": "UAP topic", "node_type": "topic"},
            {"name": "UFO topic", "node_type": "topic"},
        ],
        page_id="pg1",
        confirmation={
            "by": "workbench/mark",
            "at": "2026-09-03T05:00:00Z",
            "via": "workbench-compose",
        },
    )
    apply_pages(conn)
    propose(conn)

    stale, _gate_count = _proposal_table_stale(conn)

    assert stale is False  # the recompute has nothing left to do
