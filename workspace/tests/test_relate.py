"""EXPERIMENTAL relate pass: a claim-neighbour shortlist, a batched strict judge, a confirm round."""

import json
import sqlite3

import numpy as np

from anomalica_common.digest.models import Claim, Record
from assimilator import relate
from assimilator.database import init_db, insert_claim, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for rid in ("ra", "rb", "rc"):
        insert_record(
            conn,
            Record(
                id=rid, title=f"Record {rid}", content_hash="sha256:" + rid * 21 + "x"
            ),
        )
    for rid, texts in (
        ("ra", ["orbs beneath a helicopter at a test range", "ODNI ran the operation"]),
        ("rb", ["the operation lured objects to a range", "Coulthart on Skywatcher"]),
        ("rc", ["Apollo 11 landed on the Moon"]),
    ):
        for i, tx in enumerate(texts):
            insert_claim(
                conn,
                Claim(
                    id=f"{rid}-{i}",
                    content=tx,
                    claim_type="testimony",
                    record_id=rid,
                    location_in_record=f"{i:04d}",
                ),
            )
    conn.commit()
    return conn


def test_shortlist_pairs_records_by_claim_neighbourhood_and_never_itself():
    ids = ["ra-0", "ra-1", "rb-0", "rb-1", "rc-0"]
    recs = ["ra", "ra", "rb", "rb", "rc"]
    v = np.asarray(
        [[1, 0, 0], [0.9, 0.1, 0], [0.95, 0.05, 0], [0.8, 0.2, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pairs = relate.shortlist(ids, recs, v, k=1, top=1)
    assert ("ra", "rb") in pairs
    e = pairs[("ra", "rb")]
    assert e["hits_ab"] == 2 and e["hits_ba"] == 2
    assert e["claims_a"] == {"ra-0", "ra-1"} and e["claims_b"] <= {"rb-0", "rb-1"}
    # Every record gets its top records, even a lone one whose nearest claim is
    # far away; the judge, not the shortlist, says no.
    rc_pairs = [p for p in pairs if "rc" in p]
    assert rc_pairs and all(
        pairs[p]["hits_ab"] + pairs[p]["hits_ba"] <= 2 for p in rc_pairs
    )


def test_render_numbers_pairs_and_cuts_a_big_side_to_the_claims_that_took_part():
    conn = _graph()
    for i in range(2, relate.MAX_SIDE + 5):
        insert_claim(
            conn,
            Claim(
                id=f"rb-{i}",
                content=f"filler {i}",
                claim_type="testimony",
                record_id="rb",
                location_in_record=f"{i:04d}",
            ),
        )
    conn.commit()
    prompt = relate.render(
        conn,
        [
            (("ra", "rb"), {"claims_a": {"ra-0"}, "claims_b": {"rb-0"}}),
            (("ra", "rc"), None),
        ],
    )
    assert relate.PROMPT.startswith("You are comparing the extracted claims")
    assert "PAIR 1\nRECORD A:" in prompt and "PAIR 2\nRECORD A:" in prompt
    assert "[rb-0] (testimony) the operation lured" in prompt
    assert "filler" not in prompt  # the big side is cut to its neighbours
    assert "[ra-1]" in prompt  # the small side is sent whole


def test_verdicts_are_stored_including_unrelated_and_not_rejudged():
    conn = _graph()
    calls = []

    def call(prompt, text, model, schema=None, use_api=None):
        calls.append(text)
        assert prompt == relate.PROMPT
        assert (
            "PAIR 1" in text and "PAIR 2" in text
        )  # both pairs in one call, in the document
        return json.dumps(
            {
                "decisions": [
                    {
                        "pair_id": 1,
                        "verdict": "possibly_related",
                        "shared_subject": "December 2025 ODNI operation",
                        "reason": "A says orbs at a range; B says the operation lured objects.",
                        "links": [
                            {"a": "ra-0", "b": "rb-0", "relation": "same_subject"}
                        ],
                    },
                    {
                        "pair_id": 2,
                        "verdict": "unrelated",
                        "shared_subject": "",
                        "reason": "Nothing pins them.",
                        "links": [],
                    },
                ]
            }
        )

    pairs = [
        (("ra", "rb"), {"claims_a": set(), "claims_b": set()}),
        (("ra", "rc"), {"claims_a": set(), "claims_b": set()}),
    ]
    counts = relate.judge_pairs(
        conn, pairs, "claude-haiku-4-5", False, call, json.loads, lambda *_: None
    )
    assert (
        counts["possibly_related"] == 1
        and counts["unrelated"] == 1
        and counts["errors"] == 0
        and counts["calls"] == 1
    )
    assert relate.judged(conn) == {("ra", "rb"), ("ra", "rc")}
    rows = relate.related(conn)
    assert (
        len(rows) == 1
        and rows[0][5] == "December 2025 ODNI operation"
        and rows[0][8] is None
    )
    links = json.loads(
        conn.execute(
            "SELECT links FROM record_relations WHERE record_a='ra' AND record_b='rb'"
        ).fetchone()[0]
    )
    assert links[0]["relation"] == "same_subject"


def test_a_verdict_outside_the_vocabulary_is_an_error_not_a_row():
    conn = _graph()
    counts = relate.judge_pairs(
        conn,
        [(("ra", "rb"), {})],
        "m",
        False,
        lambda *a, **k: json.dumps(
            {
                "decisions": [
                    {
                        "pair_id": 1,
                        "verdict": "maybe",
                        "shared_subject": "",
                        "reason": "",
                        "links": [],
                    }
                ]
            }
        ),
        json.loads,
        lambda *_: None,
    )
    assert counts["errors"] == 1 and relate.judged(conn) == set()


def test_a_positive_survives_only_if_a_fresh_batch_reproduces_it():
    conn = _graph()
    first = {
        "verdict": "possibly_related",
        "shared_subject": "the operation",
        "reason": "r",
        "links": [],
    }
    relate.store(conn, "ra", "rb", first, "m", "p1")
    relate.store(conn, "ra", "rc", first, "m", "p2")
    relate.store(
        conn,
        "rb",
        "rc",
        {"verdict": "unrelated", "shared_subject": "", "reason": "n", "links": []},
        "m",
        "p3",
    )
    assert relate.positives_to_confirm(conn) == [("ra", "rb"), ("ra", "rc")]

    def call(prompt, text, model, schema=None, use_api=None):
        # Whatever the order, ra~rb is reproduced and ra~rc is not.
        decisions = []
        for n, block in enumerate(text.split("PAIR ")[1:], 1):
            verdict = (
                "unrelated"
                if ("[rc-0]" in block and "[rb-0]" not in block)
                else "same_subject"
            )
            decisions.append(
                {
                    "pair_id": n,
                    "verdict": verdict,
                    "shared_subject": "x" if verdict != "unrelated" else "",
                    "reason": "",
                    "links": [],
                }
            )
        return json.dumps({"decisions": decisions})

    counts = relate.confirm_pairs(
        conn,
        [("ra", "rb"), ("ra", "rc")],
        [(("rb", "rc"), {})],
        {},
        "m",
        False,
        call,
        json.loads,
        lambda *_: None,
    )
    assert counts == {"confirmed": 1, "failed": 1, "errors": 0, "calls": 1}
    rows = {
        (r[0], r[1]): r
        for r in conn.execute(
            "SELECT record_a, record_b, verdict, first_verdict, confirm_verdict, confirmed, reason FROM record_relations"
        )
    }
    assert rows[("ra", "rb")][2:6] == (
        "possibly_related",
        "possibly_related",
        "same_subject",
        1,
    )
    assert rows[("ra", "rc")][2:6] == ("unrelated", "possibly_related", "unrelated", 0)
    assert rows[("ra", "rc")][6].startswith("failed confirmation:")
    assert relate.positives_to_confirm(conn) == []
