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
import sqlite3
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


def find_same_origin_records(
    store_dir: Path, paths: list[Path] | None = None
) -> list[DuplicatePair]:
    """Records sharing a source_url or source_id - the same thing fetched twice.

    Complementary to the text scan, not a subset of it, and both are needed: a
    re-scrape of one URL can diverge in text (boilerplate, an edit) far enough to
    miss a similarity cut while remaining trivially the same fetch, and two files
    of one book share no URL at all. Exact string match, so it is free.
    """
    seen: dict[tuple[str, str], list[str]] = {}
    for path in sorted(paths if paths is not None else store_dir.glob("*.md")):
        record_hash = path.resolve().name.split(".", 1)[0]
        if len(record_hash) != 64:
            continue
        try:
            head = path.read_text(errors="replace")[:4000]
        except OSError:
            continue
        if is_superseded(head):
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


_SUPERSEDED_RE = re.compile(r"^superseded_by:\s*(\S+)\s*$", re.M)


def live_record_paths(ingests_dir: Path) -> list[Path]:
    """The records currently in play, resolved through `ingests/records/`.

    Globbing `store/*.md` is the wrong set twice over. It misses nothing today by
    luck, but the store also holds archive tiers - `store/v1/` carries 211
    schema-1 records, 133 of which share a source_url with a live record and are
    therefore superseded re-ingests - and a recursive glob would report every one
    of those as a duplicate. `records/` is the authoritative live set by
    construction: one symlink per record, none pointing into an archive tier.
    Falls back to the store root when there is no records/ directory.
    """
    records_dir = ingests_dir / "records"
    if records_dir.is_dir():
        return sorted(p for p in records_dir.glob("*.md") if p.resolve().is_file())
    return sorted(ingests_dir.glob("store/*.md"))


def unreachable_live_records(ingests_dir: Path) -> list[str]:
    """Live store records that `live_record_paths` does NOT reach.

    records/ is maintained by hand-ish operations and drifts: a slug-changing
    re-ingest can leave a stale symlink, a re-ingest can fail to create one, and
    loose non-symlink files predating the store live there too. A scan that reads
    records/ and reports "all records" while silently missing some is how a
    coverage claim gets overstated - three times in one evening, in this module's
    case. So the coverage is computed and reported rather than assumed.
    """
    reached = set()
    for path in live_record_paths(ingests_dir):
        name = path.resolve().name.split(".", 1)[0]
        if len(name) == 64:
            reached.add(name)
    missing = []
    for path in sorted((ingests_dir / "store").glob("*.md")):
        record_hash = path.name.split(".", 1)[0]
        if len(record_hash) != 64 or record_hash in reached:
            continue
        try:
            if is_superseded(path.read_text(errors="replace")[:4000]):
                continue
        except OSError:
            continue
        missing.append(record_hash)
    return missing


def is_superseded(text: str) -> bool:
    """True if the record declares itself replaced by another.

    A supersession is INDISTINGUISHABLE from a duplicate by text similarity -
    both are near-identical bodies under two hashes - and every body-normalising
    fix that ships mints one per affected record. The declaration is authoritative
    where it exists, so a marked record is dropped before similarity runs: it is
    one source that was re-extracted, not two sources.
    """
    return bool(_SUPERSEDED_RE.search(text))


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
    paths: list[Path] | None = None,
) -> list[DuplicatePair]:
    """Near-duplicate record pairs, strongest first.

    Pass `paths` (from `live_record_paths`) to compare the live set; otherwise
    every `{hash}.md` directly under `store_dir`. Records declaring
    `superseded_by` are excluded before comparison. Deterministic and offline -
    shingle overlap only, no model and no embeddings.
    """
    # A record is `{hash}.md` or `{hash}.v2.md` (record schema 2), and the store
    # holds far more of the latter - globbing only the bare form silently scans a
    # third of the store. Sidecars (.review.json, .verification.json) are not .md
    # and do not reach here; where both forms exist for one hash the newer wins.
    fingerprints: dict[str, set[int]] = {}
    for path in sorted(paths if paths is not None else store_dir.glob("*.md")):
        record_hash = path.resolve().name.split(".", 1)[0]
        if len(record_hash) != 64:
            continue
        if record_hash in fingerprints and not path.name.endswith(".v2.md"):
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        # Declared supersessions are one source re-extracted, not two. Dropped
        # before similarity so they never reach the report at all.
        if is_superseded(text):
            continue
        fingerprints[record_hash] = shingles(record_body(text))

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


def link_works(
    conn: sqlite3.Connection,
    ingests_dir: Path,
    threshold: float = DEFAULT_JACCARD,
) -> dict:
    """Collapse duplicate records onto a shared work_id in the graph.

    Union-find over both detectors' pairs, then every record in a group takes the
    lexicographically smallest member's id as its work_id. Idempotent and
    order-independent: the same store always yields the same grouping, so this is
    DERIVED state that a rebuild can recompute rather than something to replay.

    Only records the graph actually holds are touched - the store contains far
    more than has been digested.
    """
    store = ingests_dir / "store"
    live = live_record_paths(ingests_dir)
    pairs = find_duplicate_records(store, threshold, paths=live)
    pairs += find_same_origin_records(store, paths=live)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        a, b = find(pair.a), find(pair.b)
        if a != b:
            parent[min(a, b)] = min(a, b)
            parent[max(a, b)] = min(a, b)

    by_hash = {
        row[0].removeprefix("sha256:"): row[1]
        for row in conn.execute(
            "SELECT content_hash, id FROM records WHERE content_hash IS NOT NULL"
        )
    }
    groups: dict[str, list[str]] = {}
    for record_hash in parent:
        groups.setdefault(find(record_hash), []).append(record_hash)

    linked = 0
    for members in groups.values():
        present = sorted(by_hash[h] for h in members if h in by_hash)
        if len(present) < 2:
            continue  # the duplicate exists in the store but not in the graph
        work_id = present[0]
        for record_id in present:
            conn.execute(
                "UPDATE records SET work_id = ? WHERE id = ?", (work_id, record_id)
            )
            linked += 1
    conn.execute("UPDATE records SET work_id = id WHERE work_id IS NULL")
    conn.commit()
    return {
        "duplicate_pairs": len(pairs),
        "records_linked": linked,
        "works": conn.execute("SELECT COUNT(DISTINCT work_id) FROM records").fetchone()[
            0
        ],
        "records": conn.execute("SELECT COUNT(*) FROM records").fetchone()[0],
    }
