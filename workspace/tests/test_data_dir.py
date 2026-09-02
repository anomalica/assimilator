"""One variable moves every default under the data directory."""

import importlib


def test_the_data_directory_variable_moves_every_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSIMILATOR_DATA_DIR", str(tmp_path))
    for name in (
        "ASSIMILATOR_RERANK_SCORES",
        "ASSIMILATOR_VERIFY_VERDICTS",
        "ANOMALICA_MERGE_CANDIDATES",
        "ANOMALICA_MERGE_CANDIDATES_MANUAL",
    ):
        monkeypatch.delenv(name, raising=False)
    from assimilator import data_dir, merge_pipeline

    importlib.reload(data_dir)
    assert data_dir.data_dir() == tmp_path
    assert merge_pipeline.scores_path() == tmp_path / "rerank-scores.jsonl"
    assert merge_pipeline.verdicts_path() == tmp_path / "verify-band-verdicts.jsonl"
    assert merge_pipeline.manual_path() == tmp_path / "merge-candidates-manual.json"
    # A file's own variable still wins.
    monkeypatch.setenv("ASSIMILATOR_RERANK_SCORES", "/elsewhere/scores.jsonl")
    assert str(merge_pipeline.scores_path()) == "/elsewhere/scores.jsonl"
