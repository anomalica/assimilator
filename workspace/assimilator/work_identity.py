"""Which records are manifestations of the SAME WORK.

A record is content-addressed by its exact bytes, so one work can enter the store
many times over: a re-download, a different edition file, a re-export of the same
epub, an OCR pass over a scan, a PDF and the ebook of the same book, a transcript
and the video it came from. Each yields a different hash and therefore a different
record - and every consumer that counts "distinct records" then counts one work as
several sources.

That is not a tidiness problem, it inverts the guards:

- Corroboration pairs claims across records and excludes only same-record pairs,
  so two copies of one book produce near-identical claim text under two record ids
  at a similarity nothing else in the corpus reaches. The duplicate would rank as
  the best-corroborated material in the graph.
- The source-spread metric (page_gate) flags a node whose claims come mostly from
  one record. Duplicate the record and the claims split evenly between the copies,
  so the node reads as well spread. The metric does not merely go blind, it
  reverses: the pages it blesses most confidently are the ones with the worst
  provenance.

ADR 0039 states the rule this serves - count independence by source, never by
record count - and names the wire-story reprint as the same defect. ADR 0044's
provenance chain is the general answer and handles reprints, which no text
similarity can catch because the bytes really are different. This module is the
narrower, deterministic half: same TEXT, different bytes. No AI, no embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Word-level 5-grams: long enough that unrelated documents in one domain share
# almost none (the domain's stock phrases are shorter), short enough to survive
# the edition-to-edition churn - repagination, a changed foreword, OCR noise -
# that makes two files of one book differ in bytes.
SHINGLE_SIZE = 5

# Jaccard over those shingles. The observed same-book pair scored 0.999; genuinely
# distinct records in this corpus sit orders of magnitude lower, so the gap is wide
# and the exact cut is not load-bearing. It stays explicit and overridable anyway:
# this threshold has been measured on ONE pair, which is not calibration, and the
# corpus it would be fitted to is small, skewed and extracted at minimum effort.
DEFAULT_JACCARD = 0.60

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_ANNOTATION_RE = re.compile(r"\{\{[^}]*\}\}")
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class DuplicatePair:
    a: str
    b: str
    jaccard: float
    shared: int
    union: int
    reason: str = "text"


_SOURCE_KEY_RE = re.compile(r"^(source_url|source_id):\s*(.+)$", re.M)


def find_same_origin_records(store_dir: Path) -> list[DuplicatePair]:
    """Records sharing a source_url or source_id - the same thing fetched twice.

    Complementary to the text scan, not a subset of it, and both are needed: a
    re-scrape of one URL can diverge in text (boilerplate, an edit) far enough to
    miss a similarity cut while remaining trivially the same fetch, and two files
    of one book share no URL at all. Exact string match, so it is free.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for path in sorted(store_dir.glob("*.md")):
        record_hash = path.name.split(".", 1)[0]
        if len(record_hash) != 64:
            continue
        try:
            head = path.read_text(errors="replace")[:4000]
        except OSError:
            continue
        for field, value in _SOURCE_KEY_RE.findall(head):
            key = (field, value.strip().strip('"'))
            if record_hash not in seen.setdefault(key, []):
                seen[key].append(record_hash)
    pairs: list[DuplicatePair] = []
    for (field, _value), hashes in seen.items():
        for i, a in enumerate(hashes):
            for b in hashes[i + 1 :]:
                pairs.append(DuplicatePair(a, b, 1.0, 0, 0, reason=field))
    return pairs


def record_body(text: str) -> str:
    """The record's prose, without frontmatter or inline annotation markers.

    Both are per-ingest metadata: two manifestations of one work differ in their
    frontmatter (hash, accession date, handler version) while sharing their prose,
    so comparing the raw file would understate the similarity of exactly the pairs
    this exists to find.
    """
    return _ANNOTATION_RE.sub(" ", _FRONTMATTER_RE.sub("", text))


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[int]:
    """Hashed word-level n-grams. Hashed rather than kept as strings because a
    book is ~100k shingles and the comparison is all-pairs."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i : i + size])) for i in range(len(words) - size + 1)}


def jaccard(a: set[int], b: set[int]) -> tuple[float, int, int]:
    if not a or not b:
        return 0.0, 0, 0
    shared = len(a & b)
    union = len(a) + len(b) - shared
    return (shared / union if union else 0.0), shared, union


def _containment(a: set[int], b: set[int]) -> float:
    """Shared shingles as a fraction of the SMALLER document.

    Jaccard punishes a size mismatch: a 20-page article quoted whole inside a
    400-page book shares nearly all of its own shingles yet scores low, because the
    book's remaining bulk dominates the union. Containment catches that
    one-inside-the-other case, which for source-counting purposes is the same
    defect - the article is not an independent second source for those claims.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def find_duplicate_records(
    store_dir: Path,
    threshold: float = DEFAULT_JACCARD,
    containment_threshold: float = 0.90,
) -> list[DuplicatePair]:
    """Near-duplicate record pairs in an ingests store, strongest first.

    Compares every pair of `{hash}.md` records under `store_dir`. Deterministic
    and offline - shingle overlap only, no model and no embeddings.
    """
    # A record is `{hash}.md` or `{hash}.v2.md` (record schema 2), and the store
    # holds far more of the latter - globbing only the bare form silently scans a
    # third of the store. Sidecars (.review.json, .verification.json) are not .md
    # and do not reach here; where both forms exist for one hash the newer wins.
    fingerprints: dict[str, set[int]] = {}
    for path in sorted(store_dir.glob("*.md")):
        record_hash = path.name.split(".", 1)[0]
        if len(record_hash) != 64:
            continue
        if record_hash in fingerprints and not path.name.endswith(".v2.md"):
            continue
        try:
            fingerprints[record_hash] = shingles(record_body(path.read_text()))
        except (OSError, UnicodeDecodeError):
            continue

    names = sorted(fingerprints)
    pairs: list[DuplicatePair] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            score, shared, union = jaccard(fingerprints[a], fingerprints[b])
            if score >= threshold or (
                shared
                and _containment(fingerprints[a], fingerprints[b])
                >= containment_threshold
            ):
                pairs.append(DuplicatePair(a, b, round(score, 4), shared, union))
    pairs.sort(key=lambda p: -p.jaccard)
    return pairs
