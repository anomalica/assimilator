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
    (tmp_path / "ingests" / "by-name").mkdir()
    (tmp_path / "digests").mkdir(parents=True)
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
    (digests / f"{content_hash[:20]}.yaml").write_text(yaml.safe_dump({"record": rec}))


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
    assert "f" * 64 not in imp  # r1 in graph by id -> already imported
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


def test_synthesise_then_assemble_lifecycle(tmp_path):
    from assimilator import synthesise

    ingests, digests, sources = _corpus(tmp_path)
    conn = _graph_with_shared_node()  # node n1 "Shared Person" carries claims
    briefs, content = tmp_path / "briefs", tmp_path / "content"
    briefs.mkdir()
    content.mkdir()

    # The synthesiser consumes the proposal table (propose-pages decides the page
    # set; the gate's floors are tested in test_page_gate). This test exercises the
    # scheduler lifecycle, so put n1 in the proposal set directly.
    conn.execute(
        "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
        "source_count, independent_source_count, status, computed_at) "
        "VALUES ('n1', 'person', 'page-worthy', 2, 2, NULL, 'proposed', 'T')"
    )
    conn.commit()

    # No brief yet -> the entity is a pending (eager) synthesise job.
    q1 = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs, content_dir=content
    )
    syn = [j for j in q1["jobs"] if j["type"] == "synthesise"]
    assert any(j["target"]["label"] == "Shared Person" for j in syn)
    assert all(j["lane"] == "eager" for j in syn)

    # Emit the brief -> synthesise drops, a claude-lane assemble job appears.
    brief = synthesise.build_entity_brief(conn, "n1")
    synthesise.write_brief(brief, briefs)
    q2 = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs, content_dir=content
    )
    assert not [
        j
        for j in q2["jobs"]
        if j["type"] == "synthesise" and j["target"]["label"] == "Shared Person"
    ]
    asm = [j for j in q2["jobs"] if j["type"] == "assemble"]
    assert asm and all(j["lane"] == "claude" for j in asm)

    # Freeze an article from this brief_hash -> the assemble job drops.
    (content / "shared-person.md").write_text(
        f"---\nbuilt_from:\n  brief_hash: {brief['brief_hash']}\n---\nprose\n"
    )
    q3 = scheduler.build_queue(
        conn, ingests, digests, sources, "T", briefs_dir=briefs, content_dir=content
    )
    assert not [j for j in q3["jobs"] if j["type"] == "assemble"]


def test_nested_digest_still_detected_complete(tmp_path):
    # A slash in a record title nests the digest in a subdirectory; rglob must
    # still recognise it as complete, else the job re-dispatches forever.
    ingests, digests, sources = _corpus(tmp_path)
    sub = digests / "nested"
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
        assert job["lane"] in {"claude", "gpu", "eager"}
        assert job["status"] in {"eligible", "blocked", "readiness_gated"}
        assert set(job["target"]) >= {"kind", "label"}


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

    assert index["e" * 64] == {
        "version": "abc123",
        "title": "A Record",
        "record_id": "r1",
    }


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

    assert not [j for j in q2["jobs"] if j["type"] == "synthesise"]
    assert sorted(j["id"] for j in q2["jobs"] if j["type"] == "assemble") == [
        "assemble:events/apollo-14",
        "assemble:projects/apollo-14",
    ]
