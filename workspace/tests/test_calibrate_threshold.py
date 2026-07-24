"""Pair extraction and cut selection for threshold calibration. No embedding
service needed - these cover the gold-parsing logic, not the cosines."""

import json

from assimilator.calibrate_threshold import _best_cut, collect_pairs


def _write_gold(store, claims, clusters):
    store.mkdir(parents=True, exist_ok=True)
    (store / "rec.audit.json").write_text(
        json.dumps(
            {
                "schema": "anomalica/audit/2",
                "record_hash": "rec",
                "models": ["haiku", "sonnet"],
                "claims": claims,
                "clusters": clusters,
            }
        )
    )


def test_linked_cross_variant_pair_is_a_positive(tmp_path):
    store = tmp_path / "store"
    _write_gold(
        store,
        claims=[
            {"variant": "haiku.d1", "claim_id": "h1", "text": "A"},
            {"variant": "sonnet.d1", "claim_id": "s1", "text": "A-paraphrase"},
        ],
        clusters=[
            {
                "gold_id": "c1",
                "members": [
                    {"variant": "haiku.d1", "claim_id": "h1"},
                    {"variant": "sonnet.d1", "claim_id": "s1"},
                ],
            }
        ],
    )
    pos, neg = collect_pairs(store)
    assert sorted(pos[0]) == ["A", "A-paraphrase"]
    assert neg == []


def test_same_variant_pair_is_neither(tmp_path):
    """Linking two claims from ONE model is not what the audit clusters - it
    merges same-fact-different-MODEL. Same-variant pairs are excluded."""
    store = tmp_path / "store"
    _write_gold(
        store,
        claims=[
            {"variant": "haiku.d1", "claim_id": "h1", "text": "A"},
            {"variant": "haiku.d1", "claim_id": "h2", "text": "B"},
        ],
        clusters=[
            {
                "gold_id": "c1",
                "members": [
                    {"variant": "haiku.d1", "claim_id": "h1"},
                    {"variant": "haiku.d1", "claim_id": "h2"},
                ],
            }
        ],
    )
    pos, neg = collect_pairs(store)
    assert pos == []
    assert neg == []


def test_cross_cluster_pair_is_a_negative(tmp_path):
    store = tmp_path / "store"
    _write_gold(
        store,
        claims=[
            {"variant": "haiku.d1", "claim_id": "h1", "text": "A"},
            {"variant": "sonnet.d1", "claim_id": "s2", "text": "B"},
        ],
        clusters=[
            {"gold_id": "c1", "members": [{"variant": "haiku.d1", "claim_id": "h1"}]},
            {"gold_id": "c2", "members": [{"variant": "sonnet.d1", "claim_id": "s2"}]},
        ],
    )
    pos, neg = collect_pairs(store)
    assert pos == []
    assert sorted(neg[0]) == ["A", "B"]


def test_unclustered_judged_claim_is_a_negative_against_others(tmp_path):
    """A claim in no cluster is a human 'unique/distinct' mark - a negative
    against every other cross-variant judged claim."""
    store = tmp_path / "store"
    _write_gold(
        store,
        claims=[
            {"variant": "haiku.d1", "claim_id": "h1", "text": "A"},
            {"variant": "sonnet.d1", "claim_id": "s1", "text": "B"},
        ],
        clusters=[],
    )
    pos, neg = collect_pairs(store)
    assert pos == []
    assert len(neg) == 1


def test_claim_without_text_is_skipped(tmp_path):
    store = tmp_path / "store"
    _write_gold(
        store,
        claims=[
            {"variant": "haiku.d1", "claim_id": "h1", "text": ""},
            {"variant": "sonnet.d1", "claim_id": "s1", "text": "B"},
        ],
        clusters=[],
    )
    pos, neg = collect_pairs(store)
    assert pos == [] and neg == []


def test_wrong_schema_sidecar_ignored(tmp_path):
    store = tmp_path / "store"
    store.mkdir(parents=True)
    (store / "old.audit.json").write_text(json.dumps({"schema": "anomalica/audit/1"}))
    pos, neg = collect_pairs(store)
    assert pos == [] and neg == []


def test_best_cut_separates_clean_classes():
    cut, acc = _best_cut(pos=[0.8, 0.9], neg=[0.3, 0.4])
    assert 0.4 < cut <= 0.8
    assert acc == 1.0


def test_best_cut_reports_imperfect_accuracy_on_overlap():
    _, acc = _best_cut(pos=[0.5, 0.9], neg=[0.4, 0.6])
    assert acc < 1.0
