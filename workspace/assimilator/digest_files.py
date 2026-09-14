"""Which digest YAML files on disk are the graph's inputs.

`digests/` holds canonical digests at the root (and, when a record title
contains a slash, below visible directories). Evidence and non-canonical
artefacts live under `variants/` and hidden directories such as `.quarantine/`.

A bare `glob("**/*.yaml")` returns all of them. Feeding those to the graph can
import duplicate or rights-invalid evidence. Recursion is still required (the
slash-in-title case), so discovery must fail closed on every non-canonical
subtree rather than stop recursing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

VARIANTS_DIR = "variants"
CURRENT_IMPORT_GENERATION = 1


def canonical_digests(directory: Path | str) -> list[Path]:
    """Sorted canonical digests, excluding variants and hidden evidence trees."""
    root = Path(directory)
    return sorted(
        p for p in root.glob("**/*.yaml") if digest_is_importable(p, root=root)
    )


def digest_is_importable(path: Path | str, root: Path | None = None) -> bool:
    """Whether a digest path is canonical rather than retained evidence."""
    path = Path(path).resolve()
    if root is None:
        root = next(
            (parent for parent in path.parents if parent.name == "digests"), None
        )
    if root is not None:
        try:
            parts = path.relative_to(Path(root).resolve()).parts
        except ValueError:
            return False
        if VARIANTS_DIR in parts or any(part.startswith(".") for part in parts):
            return False
    try:
        with path.open() as digest:
            for line in digest:
                if line[:1].isspace():
                    continue
                key, separator, value = line.partition(":")
                if separator and key == "run_kind":
                    return yaml.safe_load(value) != "comparison"
    except (OSError, yaml.YAMLError):
        return False
    return True


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
