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

import hashlib
from pathlib import Path

VARIANTS_DIR = "variants"
CURRENT_IMPORT_GENERATION = 1


def canonical_digests(directory: Path | str) -> list[Path]:
    """Sorted canonical digest files under `directory`, variants excluded."""
    root = Path(directory)
    return sorted(
        p
        for p in root.glob("**/*.yaml")
        if VARIANTS_DIR not in p.relative_to(root).parts
    )


def canonical_digest_path(path: Path, root: Path | None = None) -> str:
    """Stable corpus-relative identity for a canonical digest path."""
    path = path.resolve()
    if root is not None:
        root = root.resolve()
        return str(Path(root.name) / path.relative_to(root))
    for parent in (path.parent, *path.parents):
        if parent.name == "digests":
            return str(Path("digests") / path.relative_to(parent))
    return path.name


def digest_file_identity(path: Path, root: Path | None = None) -> dict[str, str]:
    """Exact-byte hash and stable path for a canonical digest."""
    raw = path.read_bytes()
    return {
        "digest_path": canonical_digest_path(path, root),
        "digest_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def digest_receipt_identity(path: Path, root: Path | None = None) -> dict:
    """Receipt fields read directly from the exact canonical digest bytes."""
    import yaml

    raw = path.read_bytes()
    document = yaml.safe_load(raw) or {}
    pre_digest = document.get("pre_digest") or {}
    return {
        "digest_path": canonical_digest_path(path, root),
        "digest_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "extraction_generation": document.get("extraction_generation"),
        "extraction_config": document.get("extraction_config"),
        "pre_digest_sha256": (
            pre_digest.get("sha256") if isinstance(pre_digest, dict) else None
        ),
    }
