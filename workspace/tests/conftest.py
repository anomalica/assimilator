"""Every test writes its curation ledgers to a throwaway directory.

The ledgers under ~/repos/anomalica/curation are the durable record of human
corrections, replayed after every rebuild. A test that calls rename_node or
merge_nodes without redirecting them appends to that record - 23 test renames of
"Bob Smith" reached the live renames.yaml this way before anyone noticed.
"""

import pytest


@pytest.fixture(autouse=True)
def _curation_dir_is_throwaway(tmp_path_factory, monkeypatch):
    monkeypatch.setenv(
        "ANOMALICA_CURATION_DIR", str(tmp_path_factory.mktemp("curation"))
    )
