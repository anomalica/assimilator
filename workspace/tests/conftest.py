"""Every test writes its curation ledgers to a throwaway directory.

The ledgers under ~/repos/anomalica/curation are the durable record of human
corrections, replayed after every rebuild. A test that calls rename_node or
merge_nodes without redirecting them appends to that record - 133 test renames of
"Bob Smith" reached the live renames.yaml across commits 46debf9 and 99f2dee.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _curation_dir_is_throwaway(tmp_path_factory, monkeypatch):
    durable = Path(__file__).resolve().parents[3] / "curation"
    isolated = tmp_path_factory.mktemp("curation")
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(isolated))
    assert Path(os.environ["ANOMALICA_CURATION_DIR"]).resolve() != durable.resolve()
    yield
    configured = os.environ.get("ANOMALICA_CURATION_DIR")
    assert configured is not None
    assert Path(configured).resolve() != durable.resolve()


@pytest.fixture(autouse=True)
def _data_dir_is_throwaway(tmp_path_factory, monkeypatch):
    """The reranker memo, verdicts and run records under ~/.local/share/assimilator
    are shared with the scheduler's chain; no test may append to them."""
    monkeypatch.setenv("ASSIMILATOR_DATA_DIR", str(tmp_path_factory.mktemp("data")))
