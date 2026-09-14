import importlib.util
from pathlib import Path


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "merge_safety_evaluate", ROOT / "evaluate.py"
)
evaluate_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_module)


def test_account_and_temporal_edges_are_source_local_and_binding_survives_import():
    fixture = __import__("json").loads((ROOT / "fixture.json").read_text())
    conn = evaluate_module.build_graph(fixture)

    result = evaluate_module.validate_source_context(fixture, conn)

    assert result["nested_binding_verified"] is True
    assert result["continued_binding_verified"] is True
    assert result["cross_source_edges"] == 0


def test_production_candidate_and_scoring_path_is_proposal_only(tmp_path):
    result = evaluate_module.evaluate(ROOT / "fixture.json", tmp_path / "result.json")

    assert result["candidate_recall"] == 1.0
    assert result["positive_mrr"] == 1.0
    assert result["positive_over_negative_accuracy"] == 1.0
    assert result["automatic_merges"] == 0
    assert result["claim_envelope_unchanged_by_ranking_and_proposal"] is True
    assert set(result["hard_negative_ranks"]) == {
        "relatives",
        "co-witnesses",
        "paper-author",
        "mission-event",
        "predecessor-successor",
        "place-event",
        "numbered-siblings",
    }


def test_below_names_band_pairs_rank_by_names_score_without_claim_score(tmp_path):
    class LowNamesReranker:
        device = "fixture-cpu"

        def score(self, pairs, **_kwargs):
            return [0.1] * len(pairs)

    result = evaluate_module.evaluate(
        ROOT / "fixture.json", tmp_path / "low.json", LowNamesReranker()
    )

    assert all(row["score"] == 0.1 for row in result["ranked_proposals"])
