"""Which digest YAML files on disk are the graph's inputs.

`digests/` holds two different things. At the root (and, when a record title
contains a slash, one level down) are the CANONICAL digests - one per record,
the reconciled output the graph is built from. Under `variants/` are the
per-model benchmark runs: the same records digested again by opus, sonnet and
haiku for the model comparison, 243 of them against 80 canonical.

A bare `glob("**/*.yaml")` returns both. Feeding that to the graph imports each
record three or four times over - inflated claim counts, duplicate entities, and
"corroboration" that is one claim agreeing with copies of itself. Recursion is
still required (the slash-in-title case), so the fix is to skip the variants
subtree rather than stop recursing.
"""

from __future__ import annotations

from pathlib import Path

VARIANTS_DIR = "variants"


def canonical_digests(directory: Path | str) -> list[Path]:
    """Sorted canonical digest files under `directory`, variants excluded."""
    root = Path(directory)
    return sorted(
        p
        for p in root.glob("**/*.yaml")
        if VARIANTS_DIR not in p.relative_to(root).parts
    )
