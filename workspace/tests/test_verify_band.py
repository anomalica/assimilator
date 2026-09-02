"""The verify pass: banded, ordered, batched, estimated, gated; never merging."""

import json
import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import verify_band
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _scored():
    return [
        {
            "pair": ["a", "b"],
            "names_only": 0.99,
            "with_claims": 0.95,
            "from_rules": False,
        },
        {
            "pair": ["c", "d"],
            "names_only": 0.91,
            "with_claims": 0.99,
            "from_rules": False,
        },
        {
            "pair": ["e", "f"],
            "names_only": 0.99,
            "with_claims": 0.60,
            "from_rules": False,
        },  # outside the band
        {
            "pair": ["g", "h"],
            "names_only": 0.95,
            "with_claims": 0.95,
            "from_rules": True,
        },
    ]


def test_the_band_is_both_scores_and_ordered_by_the_weaker_one():
    got = verify_band.band(_scored())
    assert [f["pair"] for f in got] == [["a", "b"], ["g", "h"], ["c", "d"]]


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    for nid, name in (("a", "Lockheed U-2"), ("b", "U-2 spy plane")):
        insert_node(conn, Node(id=nid, name=name, node_type=NodeType.object))
        insert_claim(
            conn,
            Claim(
                id=f"{nid}-c",
                content=f"about {name}",
                claim_type="testimony",
                record_id="r1",
                node_references=[nid],
            ),
        )
    conn.commit()
    return conn


def test_a_live_run_writes_verdicts_and_merges_nothing(tmp_path):
    conn = _graph()
    out = tmp_path / "v.jsonl"
    calls = []

    def fake_call(prompt, text, model, schema=None, use_api=None):
        calls.append(prompt)
        assert "PAIR 1" in prompt and "Lockheed U-2" in prompt
        return json.dumps(
            {"decisions": [{"pair_id": 1, "same": True, "reason": "one aircraft"}]}
        )

    counts = verify_band.run_batches(
        conn,
        [{"pair": ["a", "b"], "names_only": 0.99, "with_claims": 0.95}],
        "claude-haiku-4-5",
        False,
        out,
        fake_call,
        json.loads,
        print,
    )

    assert counts == {"same": 1, "different": 0, "unanswered": 0, "calls": 1}
    verdict = json.loads(out.read_text().splitlines()[0])
    assert verdict["same"] is True and verdict["pair"] == ["a", "b"]
    assert (
        conn.execute("SELECT COUNT(*) FROM nodes WHERE retired_at IS NULL").fetchone()[
            0
        ]
        == 2
    )
    assert verify_band.decided(out) == {("a", "b")}


def test_the_default_is_a_dry_run_and_run_without_confirm_is_refused(tmp_path, capsys):
    conn = _graph()
    db = tmp_path / "g.db"
    conn.backup(sqlite3.connect(db))
    scored = tmp_path / "s.json"
    scored.write_text(
        json.dumps(
            {
                "final": [
                    {
                        "pair": ["a", "b"],
                        "names_only": 0.99,
                        "with_claims": 0.95,
                        "from_rules": False,
                    }
                ]
            }
        )
    )

    assert (
        verify_band.main(
            [
                "--scored",
                str(scored),
                "--db",
                str(db),
                "--out",
                str(tmp_path / "v.jsonl"),
            ]
        )
        == 0
    )
    assert "dry run" in capsys.readouterr().out
    assert (
        verify_band.main(
            [
                "--scored",
                str(scored),
                "--db",
                str(db),
                "--out",
                str(tmp_path / "v.jsonl"),
                "--run",
            ]
        )
        == 2
    )
    assert not (tmp_path / "v.jsonl").exists()


def test_estimate_counts_calls_and_tokens():
    est = verify_band.estimate(["x" * 2700, "y" * 270], 21)
    assert (
        est["calls"] == 2
        and est["input_tokens"] == 1100
        and est["output_tokens"] == 21 * 45
    )
