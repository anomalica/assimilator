"""corroborate must refuse to run rather than answer nothing.

Its --threshold defaulted to 0.99, carried over from the raw-uint8 space where
every pair scored ~0.99. In the corrected space no cross-record pair in a
120-claim sample reached 0.75, so that default selected nothing - and a run
returning zero corroborations reads as a finding about the corpus rather than a
stale constant. A required argument cannot fail that way.
"""

from __future__ import annotations

from click.testing import CliRunner

from assimilator.cli import main


def test_corroborate_without_threshold_refuses(tmp_path):
    result = CliRunner().invoke(main, ["--db", str(tmp_path / "k.db"), "corroborate"])
    assert result.exit_code != 0
    assert "--threshold is required" in result.output
    # It must point at the way to choose one, not just complain.
    assert "similarity-profile" in result.output


def test_similarity_profile_says_embed_first_rather_than_reporting_zero(tmp_path):
    result = CliRunner().invoke(
        main, ["--db", str(tmp_path / "k.db"), "similarity-profile"]
    )
    assert result.exit_code != 0
    assert "run `embed` first" in result.output


def test_limit_exists_so_a_run_can_be_measured_before_it_is_committed_to():
    """Corroboration spends the plan and recorded nothing about it: it runs
    outside the scheduler so there is no dispatch row, and the corroborations
    table keeps claim_a, claim_b and similarity with no model, timestamp or
    usage. So the cost of the next run could not be sized from the last one.
    --limit makes a measured slice possible; the usage report makes it a figure
    rather than a ratio."""
    from click.testing import CliRunner

    from assimilator.cli import main

    result = CliRunner().invoke(main, ["corroborate", "--help"])

    assert "--limit" in result.output
    assert result.exit_code == 0


def test_threshold_is_still_required_alongside_limit(tmp_path):
    """--limit must not become a way to run without a measured cut. The old 0.99
    default came from the pre-decode-fix vector space and silently returned zero."""
    from click.testing import CliRunner

    from assimilator.cli import main

    result = CliRunner().invoke(
        main, ["--db", str(tmp_path / "k.db"), "corroborate", "--limit", "20"]
    )

    assert result.exit_code != 0
    assert "--threshold is required" in result.output


def test_a_rejected_pair_is_recorded_so_it_is_not_bought_twice():
    """The verdict cost a model call. 26 of the first 86 candidate pairs were
    rejected, and nothing recorded them - so every later run over the same corpus
    re-verifies and re-pays for the same 26 verdicts, forever, and an automated
    lane would do it on every pass."""
    import sqlite3

    from assimilator.database import (
        adjudicated_pairs,
        init_db,
        insert_corroboration_rejection,
    )

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_corroboration_rejection(conn, "c1", "c2", 0.93, "sonnet")
    conn.commit()

    decided = adjudicated_pairs(conn)
    assert ("c1", "c2") in decided
    assert ("c2", "c1") in decided, "order must not matter - the pair is the unit"

    # Stored canonically, as insert_corroboration does: otherwise (a,b) and (b,a)
    # are two rows, the primary key is decorative, and the pair is bought twice.
    insert_corroboration_rejection(conn, "c2", "c1", 0.93, "sonnet")
    conn.commit()
    assert (
        conn.execute("SELECT COUNT(*) FROM corroboration_rejections").fetchone()[0] == 1
    )


def test_rejections_are_kept_out_of_the_corroborations_table():
    """Consumers read that table as 'pairs that corroborate' - scoring,
    synthesise and the scheduler's count all do. A verdict column would make
    every one of them wrong by default."""
    import sqlite3

    from assimilator.database import init_db, insert_corroboration_rejection

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_corroboration_rejection(conn, "c1", "c2", 0.93, "sonnet")
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM corroborations").fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM corroboration_rejections").fetchone()[0] == 1
    )


def test_adjudicated_pairs_covers_both_verdicts():
    import sqlite3

    from assimilator.database import (
        adjudicated_pairs,
        init_db,
        insert_corroboration,
        insert_corroboration_rejection,
    )

    from anomalica_common.digest.models import Claim, Record

    from assimilator.database import insert_claim, insert_record

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="rec", title="R", content_hash="sha256:aa"))
    for cid in ("a1", "a2"):
        insert_claim(
            conn, Claim(id=cid, content=cid, claim_type="testimony", record_id="rec")
        )
    insert_corroboration(conn, "a1", "a2", 0.95)
    insert_corroboration_rejection(conn, "r1", "r2", 0.91, "sonnet")
    conn.commit()

    decided = adjudicated_pairs(conn)
    assert {("a1", "a2"), ("r1", "r2")} <= decided
    assert ("x1", "x2") not in decided
