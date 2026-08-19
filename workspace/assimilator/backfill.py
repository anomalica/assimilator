"""Deterministic backfill of record-block fields into legacy digest YAMLs.

Digests emitted before the record block carried content_hash / publisher /
medium / duration lack those fields. The values are all deterministic - they
come straight from the source record's ingest frontmatter - so no metered
re-extraction is needed to add them. This rewrites only the `record:` block of
each digest, leaving every other byte (nodes, claims) untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from anomalica_common.digest.yaml_format import _yaml_dump
from assimilator.digest_files import canonical_digests

# Fields we backfill, mapped to the ingest-frontmatter key they come from.
# medium is the ingest's source_type.
_FIELD_SOURCES = {
    "publisher": "publisher",
    "medium": "source_type",
    "duration": "duration",
    "content_hash": "content_hash",
}

# Canonical record-block field order, matching yaml_format's emitter.
_RECORD_ORDER = [
    "id",
    "title",
    "producer",
    "publisher",
    "date",
    "medium",
    "duration",
    "content_hash",
    "reference",
]


def _resolve_ingests_dir(digests_dir: Path, ingests_dir: Path | None) -> Path | None:
    """Find the ingests/by-name dir: explicit arg, env override, then derived
    from the digests location (<root>/digests/... -> <root>/ingests/by-name)."""
    candidates = []
    if ingests_dir:
        candidates.append(Path(ingests_dir) / "by-name")
    env = os.environ.get("ANOMALICA_INGESTS_DIR")
    if env:
        candidates.append(Path(env) / "by-name")
    candidates.append(digests_dir.resolve().parent.parent / "ingests" / "by-name")
    return next((c for c in candidates if c.exists()), None)


def _ingest_frontmatter(records_dir: Path, stem: str) -> dict | None:
    ingest = records_dir / f"{stem}.md"
    if not (ingest.is_symlink() or ingest.exists()):
        return None
    try:
        text = ingest.resolve().read_text()
        return yaml.safe_load(text.split("---", 2)[1]) or {}
    except (OSError, IndexError):
        return None


def _record_block_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) line indices for the `record:` block, where end is
    exclusive (the next top-level key or end of file)."""
    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == "record:"), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln and not ln[0].isspace() and ln.rstrip():
            end = i
            break
    return start, end


def backfill_record_fields_in_file(path: Path, records_dir: Path) -> list[str]:
    """Backfill missing record-block fields in one digest. Returns the list of
    field names added (empty if nothing changed)."""
    text = path.read_text()
    lines = text.splitlines()
    bounds = _record_block_bounds(lines)
    if bounds is None:
        return []
    start, end = bounds

    block_text = "\n".join(lines[start:end])
    parsed = yaml.safe_load(block_text) or {}
    record = parsed.get("record") or {}

    fm = _ingest_frontmatter(records_dir, path.stem)
    if fm is None:
        return []

    added = []
    for field, fm_key in _FIELD_SOURCES.items():
        if record.get(field) in (None, ""):
            value = fm.get(fm_key)
            if value not in (None, ""):
                record[field] = value
                added.append(field)
    if not added:
        return []

    ordered = {k: record[k] for k in _RECORD_ORDER if k in record}
    # Preserve any keys we don't know about (defensive) after the known ones.
    for k, v in record.items():
        if k not in ordered:
            ordered[k] = v

    new_block = _yaml_dump({"record": ordered}).rstrip("\n").split("\n")
    new_lines = lines[:start] + new_block + lines[end:]
    path.write_text("\n".join(new_lines) + "\n")
    return added


def backfill_record_fields_in_dir(
    digests_dir: Path, ingests_dir: Path | None = None
) -> dict[str, list[str]]:
    """Backfill record-block fields across every digest in a directory.

    Returns {filename: [fields_added]} for files that changed.
    """
    digests_dir = Path(digests_dir)
    records_dir = _resolve_ingests_dir(digests_dir, ingests_dir)
    if records_dir is None:
        raise FileNotFoundError(
            "Could not locate the ingests/by-name directory. Set "
            "ANOMALICA_INGESTS_DIR or pass --ingests-dir."
        )
    results: dict[str, list[str]] = {}
    for path in canonical_digests(digests_dir):
        added = backfill_record_fields_in_file(path, records_dir)
        if added:
            results[path.name] = added
    return results
