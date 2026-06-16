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

Most of the assimilator is deterministic - the entity matching is local (Levenshtein plus local embeddings). Two passes are AI-assisted: **consolidate** (deciding entity merges) and **corroborate** (verifying that two claims assert the same fact). Those call the model through the same transport and spend gate as the rest of the pipeline (the Claude subscription by default, the metered API behind a toggle).

## Commands

```
assimilate <dir>     integrate a directory of digests into the existing graph (incremental)
import <digest>      import one reviewed digest YAML into the graph
rebuild <dir>        wipe and rebuild the graph from a directory of digests
stats                node / record / claim / corroboration counts
show <name>          a node and its scored claims
embed                embed all claims and nodes for similarity search
corroborate          find cross-record corroborations (AI-assisted verification)
search <query>       hybrid / semantic / keyword claim search
export-obsidian <dir>  export the graph as a navigable Obsidian vault
```

Plus the digest-maintenance passes that rewrite digest YAML in place: `reclassify-documents`, `normalise-names`, `migrate-refs-delimiter`, `rewire-refs`, `backfill-record-fields`.

The graph database defaults to `~/.local/share/assimilator/knowledge.db` (override with `--db` or `ASSIMILATOR_DB`).

## Development

The digest interchange (data model + YAML) and the Claude transport are shared with the digester via [anomalica-common](https://github.com/anomalica/anomalica-common). In development the dev container mounts that sibling repo onto `PYTHONPATH`; see `cm.yaml` and the `justfile`.

```
just test            run the test suite in the container
```

## History

This stage was split out of the digester in June 2026, where the import / matching / consolidate / scoring logic originally lived. The split makes "digestion" cleanly mean *produce per-record digests* and "assimilation" cleanly mean *integrate them into the graph*.

## Output

The SQLite knowledge graph - nodes, claims, provenance, corroborations, evidence scores - maintained incrementally and rebuildable from the digests.
