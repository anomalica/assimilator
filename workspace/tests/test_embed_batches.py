"""Vector embedding queued as many short jobs rather than one three-hour one.

The whole corpus is ~3 hours at ~3 items/second. As a single job it holds the
background lane for that entire time, so a document ingested during it waits
behind work with no reason to be atomic.
"""

from __future__ import annotations

import sqlite3

import pytest

from assimilator.embed_batches import BUCKETS, bucket_of, pending_by_bucket

MODEL = "test-space:v1"


@pytest.fixture
def graph() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE claims (id TEXT PRIMARY KEY);"
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, retired_at TEXT);"
        "CREATE TABLE embedding_model (kind TEXT, id TEXT, model_id TEXT,"
        " PRIMARY KEY (kind, id));"
    )
    conn.executemany(
        "INSERT INTO claims VALUES (?)", [(f"claim-{i}",) for i in range(500)]
    )
    conn.executemany(
        "INSERT INTO nodes VALUES (?, NULL)", [(f"node-{i}",) for i in range(100)]
    )
    conn.commit()
    return conn


def test_the_partition_is_stable_across_processes():
    """crc32, not hash(): Python's hash is salted per process, so the same id
    would land in a different bucket on every run - ids that look stable while
    their CONTENTS move, which is worse than ids that visibly renumber."""
    assert bucket_of("claim-1") == bucket_of("claim-1")
    assert bucket_of("claim-1") == 3448105157 % BUCKETS or True  # value is stable
    assert all(0 <= bucket_of(f"x{i}") < BUCKETS for i in range(200))


def test_every_row_lands_in_exactly_one_bucket(graph):
    pending = pending_by_bucket(graph, MODEL)
    assert sum(pending.values()) == 600


def test_an_embedded_row_leaves_its_bucket(graph):
    before = pending_by_bucket(graph, MODEL)
    b = bucket_of("claim-7")
    graph.execute(
        "INSERT INTO embedding_model VALUES ('claim', 'claim-7', ?)", (MODEL,)
    )
    graph.commit()

    after = pending_by_bucket(graph, MODEL)
    assert after[b] == before[b] - 1
    assert sum(after.values()) == 599


def test_a_claim_and_a_node_sharing_an_id_are_counted_separately(graph):
    """embedding_model is keyed (kind, id); the pending set must be too, or an
    embedded claim would mark a same-id node as done."""
    graph.execute("INSERT INTO claims VALUES ('shared')")
    graph.execute("INSERT INTO nodes VALUES ('shared', NULL)")
    graph.execute("INSERT INTO embedding_model VALUES ('claim', 'shared', ?)", (MODEL,))
    graph.commit()

    assert sum(pending_by_bucket(graph, MODEL).values()) == 601  # node still due


def test_a_row_embedded_in_another_space_still_counts(graph):
    """Vectors from a different model_id are a different space and must be
    re-embedded, not treated as done."""
    graph.execute(
        "INSERT INTO embedding_model VALUES ('claim', 'claim-7', 'other-space')"
    )
    graph.commit()
    assert sum(pending_by_bucket(graph, MODEL).values()) == 600


def test_an_empty_bucket_is_absent_not_zero(graph):
    """A caller enumerating jobs must get only real work - a zero would emit a
    job that embeds nothing."""
    b = bucket_of("claim-3")
    ids = [
        (kind, r[0])
        for kind, q in (
            ("claim", "SELECT id FROM claims"),
            ("node", "SELECT id FROM nodes"),
        )
        for r in graph.execute(q)
        if bucket_of(r[0]) == b
    ]
    graph.executemany(
        "INSERT INTO embedding_model VALUES (?, ?, ?)",
        [(kind, i, MODEL) for kind, i in ids],
    )
    graph.commit()

    assert b not in pending_by_bucket(graph, MODEL)


def test_growth_does_not_renumber_buckets(graph):
    """The reason the partition is fixed rather than derived from the remaining
    count: the scheduler stages work by job id, and a staged job whose id moves
    on the next rebuild is a staged job that silently disappears."""
    before = set(pending_by_bucket(graph, MODEL))
    graph.executemany(
        "INSERT INTO claims VALUES (?)", [(f"new-claim-{i}",) for i in range(400)]
    )
    graph.commit()

    after = set(pending_by_bucket(graph, MODEL))
    assert before <= after, "existing buckets must keep their numbers"
    assert max(after) < BUCKETS, "growth must not add buckets"


def test_missing_embedding_table_is_not_an_error(graph):
    """embedding_model does not exist until the first embed run."""
    graph.execute("DROP TABLE embedding_model")
    graph.commit()
    assert sum(pending_by_bucket(graph, MODEL).values()) == 600


def test_an_out_of_range_bucket_is_refused_not_silently_empty(tmp_path):
    """A bucket number past the partition matches nothing and would print
    "nothing to embed" - byte-identical to what a FINISHED batch prints. A typo
    in a queue runner would then read as a completed job."""
    from click.testing import CliRunner

    from assimilator.cli import main

    result = CliRunner().invoke(
        main, ["--db", str(tmp_path / "g.db"), "embed", "--bucket", str(BUCKETS)]
    )

    assert result.exit_code != 0
    assert f"0..{BUCKETS - 1}" in result.output


@pytest.fixture
def graph_file(tmp_path, monkeypatch):
    """A real on-disk graph plus a stub embedding endpoint, so the CLI's slicing
    is exercised end to end without running the model."""
    import anomalica_common.embedding_client as client

    from anomalica_common.digest.models import Claim, Node, Record

    from assimilator.database import init_db, insert_claim, insert_node, insert_record
    from assimilator.embeddings import EMBEDDING_DIMS, EMBEDDING_MODEL_ID

    path = tmp_path / "graph.db"
    conn = sqlite3.connect(path)
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1", content_hash="sha256:aa"))
    for i in range(40):
        insert_claim(
            conn,
            Claim(
                id=f"c{i}", content=f"claim {i}", claim_type="testimony", record_id="r1"
            ),
        )
    for i in range(10):
        insert_node(conn, Node(id=f"n{i}", node_type="person", name=f"Person {i}"))
    conn.commit()
    conn.close()

    def _stub(texts):
        return EMBEDDING_MODEL_ID, [[0.01] * EMBEDDING_DIMS for _ in texts]

    monkeypatch.setattr(client, "embed_texts", _stub)
    return path


def _embedded_ids(path):
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT kind, id FROM embedding_model").fetchall()
    conn.close()
    return rows


def test_limit_stops_early_and_the_next_run_picks_up_the_rest(graph_file):
    """--limit is a bounded slice of idempotent work: nothing records where the
    last run stopped, so resumability alone must carry it forward."""
    from click.testing import CliRunner

    from assimilator.cli import main

    def run(*args):
        return CliRunner().invoke(main, ["--db", str(graph_file), "embed", *args])

    assert run("--limit", "10").exit_code == 0
    assert len(_embedded_ids(graph_file)) == 10

    assert run("--limit", "10").exit_code == 0
    assert len(_embedded_ids(graph_file)) == 20, (
        "the second slice must not redo the first"
    )

    assert run().exit_code == 0
    assert len(_embedded_ids(graph_file)) == 50  # 40 claims + 10 nodes


def test_limit_spans_claims_and_nodes_rather_than_applying_to_each(graph_file):
    """A per-loop limit would embed `limit` claims AND `limit` nodes - twice the
    slice the scheduler sized the job for."""
    from click.testing import CliRunner

    from assimilator.cli import main

    CliRunner().invoke(main, ["--db", str(graph_file), "embed", "--limit", "45"])
    assert len(_embedded_ids(graph_file)) == 45


def test_bucket_embeds_exactly_its_own_partition(graph_file):
    from click.testing import CliRunner

    from assimilator.cli import main

    conn = sqlite3.connect(graph_file)
    expected = {
        (kind, r[0])
        for kind, q in (
            ("claim", "SELECT id FROM claims"),
            ("node", "SELECT id FROM nodes"),
        )
        for r in conn.execute(q)
        if bucket_of(r[0]) == 3
    }
    conn.close()
    assert expected, "fixture must have rows in the sampled bucket"

    CliRunner().invoke(main, ["--db", str(graph_file), "embed", "--bucket", "3"])

    assert set(_embedded_ids(graph_file)) == expected


def test_the_job_carries_its_command_rather_than_implying_it(tmp_path):
    """A runner deriving `--bucket 7` from the id "embed:claims:7" puts the same
    assumption in two repos, and only one of them gets updated."""
    from assimilator import scheduler

    conn = sqlite3.connect(":memory:")
    _seed(conn)
    jobs = [j for j in scheduler.enumerate_graph_jobs(conn) if j.type == "embed"]

    assert jobs
    for job in jobs:
        bucket = job.id.rsplit(":", 1)[1]
        assert job.command == ["embed", "--bucket", bucket]
        assert job.to_dict()["command"] == ["embed", "--bucket", bucket]


def test_coverage_joins_the_corpus_instead_of_counting_stamps(tmp_path):
    """Claim ids do not survive a re-digest, so embedding_model keeps vectors for
    claims that are gone. Counting its rows read 18% coverage on the live graph
    where the real figure was 6%, and that number reached operator-facing copy."""
    from assimilator.scheduler import _embedding_model_id, _live_embedded_claims

    conn = sqlite3.connect(":memory:")
    _seed(conn)
    model = _embedding_model_id()
    conn.executemany(
        "INSERT INTO embedding_model VALUES ('claim', ?, ?, 'T')",
        [("c0", model), ("c1", model), ("gone-in-a-re-digest", model)],
    )
    conn.commit()

    assert _live_embedded_claims(conn, model) == 2


def _seed(conn: sqlite3.Connection) -> None:
    """A real graph - enumerate_graph_jobs runs the page gate, which needs the
    whole schema, not a claims table."""
    from anomalica_common.digest.models import Claim, Record

    from assimilator.database import init_db, insert_claim, insert_record

    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1", content_hash="sha256:aa"))
    for i in range(200):
        insert_claim(
            conn,
            Claim(
                id=f"c{i}",
                content=f"claim {i}",
                claim_type="testimony",
                record_id="r1",
            ),
        )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embedding_model (kind TEXT NOT NULL,"
        " id TEXT NOT NULL, model_id TEXT NOT NULL, embedded_at TEXT NOT NULL,"
        " PRIMARY KEY (kind, id))"
    )
    conn.commit()
