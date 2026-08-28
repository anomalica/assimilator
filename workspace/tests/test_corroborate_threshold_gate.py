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
