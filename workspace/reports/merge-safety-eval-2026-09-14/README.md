# Merge-safety evaluation foundation

This report-only evaluation is separate from the MiniLM search benchmark. It
uses invented project-authored CC0 material and never applies a proposed merge.

`evaluate.py` imports the fixture through `import_extraction`, builds entity
profiles through the production shortlist, and scores the labelled pairs through
the production Qwen3 `score_pairs` path. Tests use a deterministic CPU fixture
double. `--run-qwen-cpu` is the only model-enabled mode and forces CPU; it was not
run while preparing this foundation.

The fixture covers one identity variant and labelled relation-versus-identity
negatives for relatives, co-witnesses, paper/author, mission/event,
predecessor/successor, place/event and numbered siblings. Ranking and proposal
must leave claim rows and reference edges unchanged and create zero merges. The
separate merge/replay tests permit only the intended entity foreign-key repoints;
the immutable content, source, location, date and provenance envelope must remain
unchanged.

## Account and temporal boundary

Narrative accounts, continued spans, claim-account bindings and directed temporal
relations are evaluation sidecars. The canonical digest contract currently
forbids account fields, and the live graph has no account or claim-temporal-edge
tables. The evaluator therefore verifies that every sidecar binding and temporal
edge stays within one `(record_id, account_id)`, including smallest-containing
nested binding and a continued account, but does not claim graph persistence for
unsupported structures.

The labels remain `pending-human-review`. Deterministic green tests establish the
evaluation mechanics, not Qwen3 quality or a reviewed replacement threshold.

The production graph does support and now preserves claim IDs, `record_id`,
`location_in_record`, date intervals, provenance roots including `origin_ref`,
`attribution_in_text`, reference role/salience, `claim_role` and confidence.
Focused import, selection, merge, undo and replay tests cover those fields.

Run the deterministic evaluation from `workspace/`:

```bash
PYTHONPATH=.:/home/mark/repos/anomalica/anomalica-common/src \
  python reports/merge-safety-eval-2026-09-14/evaluate.py \
  --fixture reports/merge-safety-eval-2026-09-14/fixture.json \
  --out /tmp/merge-safety-result.json
```
