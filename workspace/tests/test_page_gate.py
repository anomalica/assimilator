"""The page-worthiness gate: type tier, source floor, source spread, subject test,
and the record-scoped-person exclusion."""

from __future__ import annotations

import sqlite3

from anomalica_common.digest.models import Claim, Node, Record
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.page_gate import page_gate_rows


def _add_claims(conn, node_id, name, records, subject=True):
    """One claim per (node, record) entry; pass a record id more than once for
    several claims from the same source. `subject` writes the claim ABOUT the
    node (the name opens the sentence); otherwise it merely mentions it."""
    for i, rid in enumerate(records):
        content = (
            f"{name} did thing {i}" if subject else f"Somebody mentioned {name} {i}"
        )
        insert_claim(
            conn,
            Claim(
                id=f"{node_id}-c{i}",
                content=content,
                claim_type="testimony",
                record_id=rid,
                node_references=[node_id],
            ),
        )


# 9 claims from 3 works, the second work carrying 3: clears every page-worthy
# floor (8 claims, 3 works, second >= 3).
_SPREAD = ["r1"] * 4 + ["r2"] * 3 + ["r3"] * 2
# 13 claims from 4 works, second carrying 4: clears high-bar (12, 4, second >= 3).
_HIGH_SPREAD = ["r1"] * 5 + ["r2"] * 4 + ["r3"] * 2 + ["r4"] * 2


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("r1", "r2", "r3", "r4"):
        insert_record(conn, Record(id=rid, title=rid))
    nodes = {
        "worthy": ("person", "Ada Worthy"),
        "thin": ("person", "Bo Thin"),  # 7 claims -> fails the claim floor
        "onesrc": ("person", "Cy Onesrc"),  # 3 works but one voice -> fails spread
        "mention": ("organisation", "Mention Corp"),  # never the subject
        "bare": ("person", "Chris"),  # no family name -> never a page
        "place_high": ("place", "Placeville"),
        "place_mid": ("place", "Midtown"),  # 12 claims but 3 works -> fails high-bar
        "topic_bg": (
            "topic",
            "Background topic",
        ),  # mentioned only; topics not subject-gated
        "deprecated": ("matter", "Rich Matter"),  # rich, but the type is gated
    }
    for nid, (nt, name) in nodes.items():
        insert_node(conn, Node(id=nid, node_type=nt, name=name))
    _add_claims(conn, "worthy", "Ada Worthy", _SPREAD)
    _add_claims(conn, "thin", "Bo Thin", ["r1"] * 3 + ["r2"] * 3 + ["r3"])
    _add_claims(conn, "onesrc", "Cy Onesrc", ["r1"] * 7 + ["r2", "r3"])
    _add_claims(conn, "mention", "Mention Corp", _SPREAD, subject=False)
    _add_claims(conn, "bare", "Chris", _SPREAD)
    _add_claims(conn, "place_high", "Placeville", _HIGH_SPREAD, subject=False)
    _add_claims(conn, "place_mid", "Midtown", ["r1"] * 5 + ["r2"] * 4 + ["r3"] * 3)
    _add_claims(conn, "topic_bg", "Background topic", _SPREAD, subject=False)
    _add_claims(conn, "deprecated", "Rich Matter", _HIGH_SPREAD)
    conn.commit()
    return conn


def _by_id(rows):
    return {r["node_id"]: r for r in rows}


def test_page_worthy_floor():
    rows = _by_id(page_gate_rows(_graph()))
    assert "worthy" in rows
    assert rows["worthy"]["tier"] == "page-worthy"
    assert rows["worthy"]["claim_count"] == 9
    assert rows["worthy"]["source_count"] == 3
    assert rows["worthy"]["second_source_claims"] == 3
    assert rows["worthy"]["subject_claims"] == 9


def test_page_worthy_rejects_thin():
    assert "thin" not in _by_id(page_gate_rows(_graph()))  # 7 claims


def test_spread_rejects_one_voice_with_passing_mentions():
    # 9 claims from 3 works passes the count and the source floor; 7 of them are
    # one book and the second work contributes one claim - a summary of one
    # source with a fig leaf, which is exactly what the spread floor refuses.
    assert "onesrc" not in _by_id(page_gate_rows(_graph()))


def test_subject_test_rejects_a_node_the_corpus_only_mentions():
    # Mention Corp clears every count; no claim is about it. Blink-182 at 13
    # claims / 4 sources was this shape - every claim about the guitarist.
    assert "mention" not in _by_id(page_gate_rows(_graph()))


def test_subject_test_is_not_applied_to_topics_and_places():
    rows = _by_id(page_gate_rows(_graph()))
    assert "topic_bg" in rows and rows["topic_bg"]["subject_claims"] == 0
    assert "place_high" in rows and rows["place_high"]["subject_claims"] == 0


def test_a_person_without_a_family_name_is_never_a_page():
    assert "bare" not in _by_id(page_gate_rows(_graph()))


def test_high_bar_floor_is_stricter():
    rows = _by_id(page_gate_rows(_graph()))
    assert rows["place_high"]["tier"] == "high-bar"
    # place_mid (12 claims / 3 works) would pass the page-worthy floor but fails
    # high-bar - proves the tier, not a flat count, decides.
    assert "place_mid" not in rows


def test_deprecated_types_gated_out():
    assert "deprecated" not in _by_id(page_gate_rows(_graph()))


def test_ordered_strongest_first():
    counts = [r["claim_count"] for r in page_gate_rows(_graph())]
    assert counts == sorted(counts, reverse=True)


def test_floors_env_tunable(monkeypatch):
    monkeypatch.setenv("ANOMALICA_PAGE_WORTHY_MIN_CLAIMS", "7")
    assert "thin" in _by_id(page_gate_rows(_graph()))
    monkeypatch.setenv("ANOMALICA_PAGE_MIN_SECOND_SOURCE_CLAIMS", "1")
    assert "onesrc" in _by_id(page_gate_rows(_graph()))
    monkeypatch.setenv("ANOMALICA_PAGE_MIN_SUBJECT_CLAIMS", "0")
    assert "mention" in _by_id(page_gate_rows(_graph()))


def test_subject_reads_through_a_rank_and_a_dated_clause():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("r1", "r2", "r3"):
        insert_record(conn, Record(id=rid, title=rid))
    insert_node(conn, Node(id="b", node_type="person", name="William Blanchard"))
    contents = [
        "At midday on 8 July 1947, Colonel William Blanchard ordered a press release.",
        "Blanchard phoned the public information officer.",
        "The next morning, Blanchard's office issued a retraction.",
        "According to Walter Haut, Blanchard was calm.",  # still a claim about him
        "Barry Goldwater was a friend of William Blanchard.",
    ] + ["William Blanchard briefed staff %d." % i for i in range(4)]
    for i, content in enumerate(contents):
        insert_claim(
            conn,
            Claim(
                id=f"b-{i}",
                content=content,
                claim_type="testimony",
                record_id=["r1", "r2", "r3"][i % 3],
                node_references=["b"],
            ),
        )
    conn.commit()
    rows = _by_id(page_gate_rows(conn))
    assert rows["b"]["subject_claims"] == 8  # all but the one about Goldwater


def test_retired_nodes_excluded():
    conn = _graph()
    conn.execute("UPDATE nodes SET retired_at = '2026-01-01' WHERE id = 'worthy'")
    assert "worthy" not in _by_id(page_gate_rows(conn))
