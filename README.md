# assimilator

The assimilator builds and maintains Anomalica's unified knowledge graph from the per-record digest files produced by the [digester](https://github.com/anomalica/digester).

It sits between the digester and the assembler in the Anomalica pipeline:

```
ingester -> digester -> assimilator -> assembler -> content -> site
```

## Purpose

The digester decomposes each source record into a per-record digest file (nodes + claims). The assimilator takes all those digest files and integrates them into one coherent knowledge graph:

- **Entity resolution** - the same real-world person, organisation, place, event or object named across many records is resolved to a single node (exact / alias / fuzzy matching).
- **Claim accumulation** - every claim about an entity, from every source, is gathered onto that entity.
- **Corroboration** - independent provenance chains attesting the same thing are linked, distinguishing genuine corroboration from echo chambers.
- **Evidence scoring** - claims are scored algorithmically from corroboration, attestation depth, source track record, contradictions and evidence quality. All scoring is algorithmic and transparent; no human assigns scores.

The graph is maintained incrementally - seeded once, grown each run - and the SQLite database it produces is the public, downloadable dataset and the source the assembler reads to build articles. The graph is **derived data**: it is fully rebuildable from the digests, which remain the source of truth.

The assimilator calls no language model - it is deterministic graph-building plus local embeddings (for entity matching and corroboration). There is no AI billing here.

## History

This stage was split out of the digester in June 2026, where the import / matching / consolidate / scoring logic originally lived. The split makes "digestion" cleanly mean *produce per-record digests* and "assimilation" cleanly mean *integrate them into the graph*.

## Output

The SQLite knowledge graph - nodes, claims, provenance, corroborations, evidence scores - maintained incrementally and rebuildable from the digests.
