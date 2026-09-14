# Claim-context loss impact

This read-only census compares the 157 current canonical digests, the default
derived graph, the 805 stable published briefs and 404 articles. `impact.json`
separates exact measurements from unknowns and records the graph snapshot hash.

The 156 graph-bound digests map one-to-one to all 40,400 graph claims on record
content hash plus claim text, excerpt, type, attestation, location and dates.
Against that matched population, the historical importer dropped 1,543
`origin_ref` values, 39,822 `attribution_in_text` declarations and salience from
1,849 existing role-bearing claim-node edges. The 1,913 raw role references
collapse to 1,912 resolved claim-node roles; 63 have no corresponding graph edge.

Restoring `origin_ref` in memory changes the independent-source count of 223
nodes, including 96 proposed nodes, by 625 sources in total. The largest measured
undercount is 36 for Ross Coulthart. Page eligibility does not currently use this
independence value, but brief evidence metadata does.

The published brief set contains no `payload_hash`. Of 805 briefs, 747 select a
claim affected by the measured `origin_ref` or attribution loss. Of 404 articles,
274 have `built_from` and all 274 lack `payload_hash`; 272 resolve to a published
brief and are therefore stale under the paired-hash contract. Another 130 have no
`built_from` and are unauditable against a brief.

The census does not infer identities for the 254 anonymous-origin digest claims
without `origin_ref`, treat undeclared roles as non-salient, or predict exact
model-written prose changes. The graph has no digest import receipts, so exact
digest-byte and import-generation freshness is also unknown. A new payload hash
binds current graph content but cannot prove that digest context reached the graph
before reimport or reconciliation.

Run from `workspace/` without modifying the graph:

```bash
PYTHONPATH=.:/home/mark/repos/anomalica/anomalica-common/src \
python reports/claim-context-impact-2026-09-14/measure.py \
  --digests /home/mark/repos/anomalica/digests \
  --db /home/mark/.local/share/assimilator/knowledge.db \
  --briefs /home/mark/repos/anomalica/content/briefs \
  --content /home/mark/repos/anomalica/content/pages \
  --out reports/claim-context-impact-2026-09-14/impact.json
```
