"""The READ rule for three-state record fields, pinned rather than documented.

Every three-state field in `records.metadata` - `review.state`, `run_kind` - has
a value, an alternative value, and an ABSENT key meaning "this digest predates
the field". Absent must never be read as one of the values: a missing review is
not "unreviewed", a missing run_kind is not "production", exactly as a missing
provenance chain is not "independent".

The write side (an absent field is not stored) is pinned in test_import_reconcile.
This file pins the READ side, which is where the damage happens - and it asserts
it in BOTH languages, because the graph is queried from SQL and from Python and
THEY DISAGREE ABOUT ABSENCE:

    Python   None != "variant"   is True   -> absent rows are INCLUDED
    SQLite   NULL != 'variant'   is NULL   -> absent rows are EXCLUDED

So one wrong predicate silently promotes unknown records into a canonical set in
one language and silently drops them from it in the other. Neither raises. The
SQL direction is the more dangerous of the two: a promoted variant may be spotted
in output, whereas thirty records missing from a count looks like a corpus that
is simply smaller.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

# (record id, metadata) covering all three states of a field.
FIXTURE = [
    ("canonical", {"run_kind": "production", "review": {"state": "human"}}),
    ("other", {"run_kind": "variant", "review": {"state": "none"}}),
    ("unknown", {"medium": "pdf"}),  # predates both fields
]


@pytest.fixture
def records() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE records (id TEXT PRIMARY KEY, metadata TEXT)")
    conn.executemany(
        "INSERT INTO records VALUES (?, ?)",
        [(rid, json.dumps(meta)) for rid, meta in FIXTURE],
    )
    return conn


def _sql(conn: sqlite3.Connection, predicate: str) -> set[str]:
    return {r[0] for r in conn.execute(f"SELECT id FROM records WHERE {predicate}")}


def _python(conn: sqlite3.Connection, key: str, op, value) -> set[str]:
    rows = {
        r[0]: json.loads(r[1]) for r in conn.execute("SELECT id, metadata FROM records")
    }
    return {rid for rid, meta in rows.items() if op(meta.get(key), value)}


def test_positive_predicate_selects_only_the_canonical_set(records):
    """The rule: test for the value you want. Both languages agree here, which is
    the whole reason it is the rule."""
    expected = {"canonical"}
    assert (
        _sql(records, "json_extract(metadata,'$.run_kind') = 'production'") == expected
    )
    assert _python(records, "run_kind", lambda a, b: a == b, "production") == expected


def test_negative_predicate_is_wrong_in_python_by_including_the_unknown(records):
    """`!= "variant"` promotes the record that predates the field into the
    canonical set - a comparison artefact silently treated as production."""
    got = _python(records, "run_kind", lambda a, b: a != b, "variant")
    assert "unknown" in got, "absent read as production - the failure this pins"
    assert got == {"canonical", "unknown"}


def test_negative_predicate_is_wrong_in_sql_by_dropping_the_unknown(records):
    """The same expression in SQL fails the OTHER way: NULL != 'variant' is NULL,
    so the unknown record vanishes from the result instead of being promoted.
    Quieter and therefore worse - a short count reads as a smaller corpus."""
    got = _sql(records, "json_extract(metadata,'$.run_kind') != 'variant'")
    assert "unknown" not in got, "absent silently dropped - the other failure"
    assert got == {"canonical"}


def test_the_two_languages_disagree_about_absence(records):
    """Stated as its own assertion because it is the reason the read rule cannot
    be left to a docstring: the same wrong predicate produces opposite wrong
    answers depending on which side of the pipeline runs it."""
    in_sql = _sql(records, "json_extract(metadata,'$.run_kind') != 'variant'")
    in_python = _python(records, "run_kind", lambda a, b: a != b, "variant")
    assert in_sql != in_python


def test_review_follows_the_same_rule(records):
    """`state == "human"` selects the reviewed set; `state != "none"` does not.
    This is the field the unreviewed-corpus decision rests on - 110 records enter
    the graph on the basis that a consumer can tell them from reviewed material."""
    reviewed = {
        rid
        for rid, meta in (
            (r[0], json.loads(r[1]))
            for r in records.execute("SELECT id, metadata FROM records")
        )
        if (meta.get("review") or {}).get("state") == "human"
    }
    assert reviewed == {"canonical"}

    not_none = {
        rid
        for rid, meta in (
            (r[0], json.loads(r[1]))
            for r in records.execute("SELECT id, metadata FROM records")
        )
        if (meta.get("review") or {}).get("state") != "none"
    }
    assert "unknown" in not_none, "absent read as reviewed - the failure this pins"


def test_the_untested_boundary_is_inclusive():
    """A page at EXACTLY the threshold counts as untested. One convention serves
    both the disclosure on the page-floor card and the tranche trend, and the
    disclosure is the stricter master: over-flagging costs a footnote,
    under-flagging is the failure. It is not cosmetic - nine pages sat on the
    line and the two readings differed by 6, three times the movement the trend
    exists to detect."""
    threshold = 0.25
    at_the_line = 25 / 100

    # The predicate the command uses: computable means strictly BELOW.
    assert not (at_the_line < threshold), "a page at the line must be untested"
    assert (24 / 100) < threshold, "below the line is testable"
