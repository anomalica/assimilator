# Anomalica assimilator

Parent Product and root Core instructions are loaded through `opencode.json` and remain mandatory.

The assimilator integrates reviewed digest records into Anomalica's derived knowledge graph. It owns entity resolution, claim import, provenance and independence calculations, graph consistency, page proposals, brief synthesis and publication of briefs for the assembler.

## Boundaries And Layout

- Digests are the source of truth; the SQLite graph is derived and must remain rebuildable.
- The canonical pipeline architecture and contracts live in `../anomalica/architecture/`. Check them and current consumers before changing graph or brief interfaces.
- Application code is in `workspace/assimilator/`; tests are in `workspace/tests/`.
- Shared digest models, transports, embedding clients and shared-tree Git helpers belong in `../anomalica-common/`, not in local copies.
- The default graph is `~/.local/share/assimilator/knowledge.db`. Do not modify it merely to test code; use temporary databases and fixtures.
- Publishing writes to the shared `../content/` working tree. Commit only explicit pathspecs on the expected branch through `anomalica_common.shared_tree.commit_paths`; preserve other sessions' staged and unstaged work.

## Model Use

- Most graph building is deterministic, but some commands use model-backed verification through the shared transport. Do not infer that a command is free from the repository's general role.
- Preserve the aggregate cost estimate and explicit approval gate for every metered run. Use fixture tests rather than a paid corpus run.

## Verification

```bash
just test
just e2e
```

- Run focused host tests from `workspace/` with `PYTHONPATH=.:/home/mark/repos/anomalica/product/anomalica-common/src` when that is faster.
- `just test` runs the component suite in the container. `just e2e` runs the cross-component digester-to-graph tests on the host.
- Strict expected failures in the end-to-end suite document known contract losses. Fix the producer or seam and remove the marker when implementing one; do not weaken the assertion.
