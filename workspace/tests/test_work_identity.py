"""Records that are the same work under different content hashes.

Content addressing makes one work several records on any re-download, re-export
or edition change, and every consumer counting distinct records then counts one
work as several sources. The live cases: two ebook files of Communion (0.998
shingle overlap, no shared URL) and one WikiLeaks email fetched twice (identical
source_url, text drifted). Neither detector finds both, which is why there are two.
"""

from __future__ import annotations

import textwrap

from assimilator.work_identity import (
    find_duplicate_records,
    find_same_origin_records,
    jaccard,
    record_body,
    shingles,
)

BODY = " ".join(f"sentence {i} of the source material" for i in range(200))


def _record(tmp_path, name: str, body: str, **frontmatter) -> None:
    fields = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    (tmp_path / name).write_text(
        textwrap.dedent(f"---\nschema: anomalica/record/2\n{fields}\n---\n") + body
    )


def test_frontmatter_and_annotations_are_excluded_from_the_comparison():
    """Two manifestations of one work differ in frontmatter - hash, accession
    date, handler version - while sharing their prose. Comparing raw files would
    understate exactly the pairs this exists to find."""
    text = "---\ntitle: A\ncontent_hash: sha256:aaa\n---\nthe {{highlight-start}}prose{{highlight-end}} itself"
    assert record_body(text).split() == ["the", "prose", "itself"]


def test_same_work_different_bytes_is_found(tmp_path):
    _record(
        tmp_path, f"{'a' * 64}.md", BODY, title="Communion", content_hash="sha256:a"
    )
    # A different edition: same prose, a changed foreword and a later date.
    _record(
        tmp_path,
        f"{'b' * 64}.v2.md",
        "a publisher's note\n" + BODY,
        title="Communion",
        content_hash="sha256:b",
    )
    pairs = find_duplicate_records(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].jaccard > 0.9


def test_unrelated_records_do_not_pair(tmp_path):
    _record(tmp_path, f"{'a' * 64}.md", BODY, title="One")
    _record(
        tmp_path,
        f"{'b' * 64}.md",
        " ".join(f"a wholly different clause {i} here" for i in range(200)),
        title="Two",
    )
    assert find_duplicate_records(tmp_path) == []


def test_v2_records_are_scanned(tmp_path):
    """The store holds far more `{hash}.v2.md` than `{hash}.md`; globbing only the
    bare form silently scans a third of it."""
    _record(tmp_path, f"{'a' * 64}.v2.md", BODY, title="One")
    _record(tmp_path, f"{'b' * 64}.v2.md", BODY, title="One again")
    assert len(find_duplicate_records(tmp_path)) == 1


def test_sidecars_are_not_mistaken_for_records(tmp_path):
    _record(tmp_path, f"{'a' * 64}.md", BODY, title="One")
    (tmp_path / f"{'a' * 64}.review.json").write_text("{}")
    (tmp_path / "notes.md").write_text("not a record")
    assert find_duplicate_records(tmp_path) == []


def test_one_url_fetched_twice_is_found_even_when_the_text_drifted(tmp_path):
    """The live WikiLeaks pair. Text similarity alone can miss a re-scrape whose
    boilerplate changed; the fetch identity cannot drift."""
    url = "https://wikileaks.org/podesta-emails/emailid/18724"
    _record(tmp_path, f"{'a' * 64}.md", "one wording entirely", source_url=url)
    _record(tmp_path, f"{'b' * 64}.md", "a totally separate phrasing", source_url=url)

    assert find_duplicate_records(tmp_path) == []  # text cannot see it
    origin = find_same_origin_records(tmp_path)
    assert len(origin) == 1 and origin[0].reason == "source_url"


def test_jaccard_is_symmetric_and_bounded():
    a, b = (
        shingles("the quick brown fox jumps over"),
        shingles("the quick brown fox jumps over"),
    )
    assert jaccard(a, b)[0] == 1.0
    assert jaccard(a, set())[0] == 0.0


def test_declared_supersession_is_not_a_duplicate(tmp_path):
    """A supersession is indistinguishable from a duplicate by text similarity -
    both are near-identical bodies under two hashes - and every body-normalising
    fix mints one per affected record. The declaration is authoritative."""
    _record(tmp_path, f"{'a' * 64}.md", BODY, title="Email", superseded_by="b" * 64)
    _record(tmp_path, f"{'b' * 64}.md", BODY, title="Email")

    assert find_duplicate_records(tmp_path) == []
    assert find_same_origin_records(tmp_path) == []


def test_link_works_collapses_a_duplicate_pair_in_the_graph(tmp_path):
    import sqlite3

    from anomalica_common.digest.models import Record
    from assimilator.database import init_db, insert_record
    from assimilator.work_identity import link_works

    store = tmp_path / "store"
    records = tmp_path / "records"
    store.mkdir()
    records.mkdir()
    _record(store, f"{'a' * 64}.md", BODY, title="Communion")
    _record(store, f"{'b' * 64}.md", "a foreword\n" + BODY, title="Communion")
    for name in (f"{'a' * 64}.md", f"{'b' * 64}.md"):
        (records / name).symlink_to(store / name)

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r-a", title="A", content_hash="sha256:" + "a" * 64))
    insert_record(conn, Record(id="r-b", title="B", content_hash="sha256:" + "b" * 64))
    conn.commit()

    result = link_works(conn, tmp_path)
    assert result["records_linked"] == 2
    works = {r[0] for r in conn.execute("SELECT work_id FROM records")}
    assert len(works) == 1, "two records of one book must resolve to one work"


def test_link_works_is_a_no_op_without_duplicates(tmp_path):
    import sqlite3

    from anomalica_common.digest.models import Record
    from assimilator.database import init_db, insert_record
    from assimilator.work_identity import link_works

    store = tmp_path / "store"
    store.mkdir()
    _record(store, f"{'a' * 64}.md", BODY, title="One")
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r-a", title="A", content_hash="sha256:" + "a" * 64))
    conn.commit()

    assert link_works(conn, tmp_path)["records_linked"] == 0
    assert conn.execute("SELECT work_id FROM records").fetchone()[0] == "r-a"


def test_a_live_record_missing_from_records_is_reported_not_silently_skipped(tmp_path):
    """records/ drifts - a re-ingest can fail to create a symlink, a slug change
    can strand one. A scan that reads records/ and reports a coverage number
    without its gap is how "all records" gets claimed for a partial pass."""
    from assimilator.work_identity import unreachable_live_records

    store = tmp_path / "store"
    records = tmp_path / "records"
    store.mkdir()
    records.mkdir()
    linked, stranded, retired = "a" * 64, "b" * 64, "c" * 64
    _record(store, f"{linked}.md", BODY, title="Linked")
    _record(store, f"{stranded}.md", BODY, title="No symlink")
    _record(store, f"{retired}.md", BODY, title="Retired", superseded_by=linked)
    (records / f"{linked}.md").symlink_to(store / f"{linked}.md")

    # The stranded record is reported; the superseded one is not a gap.
    assert unreachable_live_records(tmp_path) == [stranded]
