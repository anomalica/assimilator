"""Scheduler: enumerate the real pending pipeline jobs from current corpus state.

Produces a prioritised queue across the resource lanes that actually cost
something scarce - Claude (tokens) and GPU (GPU-minutes) - plus a separate
Review queue (human time, the scarcest resource) ranked by per-record demand.
Light local jobs (embed, import, re-score) are eager background, surfaced only
when they gate something downstream.

This first version enumerates work derivable from real state TODAY: sources
awaiting ingestion, records awaiting human review, reviewed records not yet
digested, and the corroboration pass. It ranks with per-job-specific drivers; it
does NOT assume a nightly runner or assert a budget - "when it runs" is a later
layer. The costly resource is tokens/GPU-time/human-time, never dollars.

The output shape matches the workbench's consumer contract (src/lib/schedule.ts)
verbatim - camelCase keys, lanes claude|gpu|eager, a jobs[] list plus a separate
reviewQueue[] - so the workbench wires to it without adapting.

Known limitation, surfaced not hidden: per-record demand is graph fanout (how
many other records share a node), so it only discriminates records already in
the graph. Cold records (awaiting ingest/first-review/first-digest) have no
graph presence yet, so their demand is a baseline until source/publisher
priority exists. The queue's value here is showing the real work, not perfect
ranking of a cold corpus.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# --- Lanes, job types, statuses (aligned to workbench src/lib/schedule.ts) ---

LANE_CLAUDE = "claude"
LANE_GPU = "gpu"
LANE_EAGER = "eager"

STATUS_ELIGIBLE = "eligible"
STATUS_BLOCKED = "blocked"
STATUS_READINESS_GATED = "readiness_gated"

# Per-source-type ingest spec: (effort, type label, lane). Only audio/video go
# in the GPU lane - that is the transcription cost. Web, PDF and ebook ingest is
# cheap local extraction with no GPU, so it runs in the eager light-local lane.
_INGEST_SPEC = {
    "opus": ("~GPU transcription", "audio/video", LANE_GPU),
    "mp3": ("~GPU transcription", "audio/video", LANE_GPU),
    "wav": ("~GPU transcription", "audio/video", LANE_GPU),
    "mp4": ("~GPU transcription", "audio/video", LANE_GPU),
    "m4a": ("~GPU transcription", "audio/video", LANE_GPU),
    "webm": ("~GPU transcription", "audio/video", LANE_GPU),
    "mkv": ("~GPU transcription", "audio/video", LANE_GPU),
    "html": ("~web extraction", "web page", LANE_EAGER),
    "pdf": ("~PDF extraction", "document", LANE_EAGER),
    "epub": ("~ebook extraction", "ebook", LANE_EAGER),
}

_SOURCE_HASH_RE = re.compile(r"^source_hash:\s*sha256:([0-9a-f]{64})", re.MULTILINE)


@dataclass
class Driver:
    label: str
    value: str
    band: str | None = None  # urgent | normal | sub | off

    def to_dict(self) -> dict:
        d = {"label": self.label, "value": self.value}
        if self.band:
            d["band"] = self.band
        return d


@dataclass
class Target:
    kind: str  # "record" | "page"
    label: str
    hash: str | None = None
    href: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "label": self.label}
        if self.hash:
            d["hash"] = self.hash
        if self.href is not None:
            d["href"] = self.href
        return d


@dataclass
class Job:
    id: str
    type: str
    lane: str
    target: Target
    status: str
    trigger: str
    drivers: list[Driver] = field(default_factory=list)
    value: float | None = None
    effort: str | None = None
    blocker: str | None = None
    article: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "type": self.type,
            "lane": self.lane,
            "target": self.target.to_dict(),
            "status": self.status,
            "trigger": self.trigger,
        }
        if self.drivers:
            d["drivers"] = [dr.to_dict() for dr in self.drivers]
        if self.value is not None:
            d["value"] = self.value
        if self.effort is not None:
            d["effort"] = self.effort
        if self.blocker is not None:
            d["blocker"] = self.blocker
        if self.article is not None:
            d["article"] = self.article
        return d


@dataclass
class ReviewItem:
    target: Target
    demand: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"target": self.target.to_dict()}
        if self.demand is not None:
            d["demand"] = self.demand
        if self.reason is not None:
            d["reason"] = self.reason
        return d


# --- Per-record demand: graph fanout (real for in-graph records only) ---


def compute_record_demand(conn: sqlite3.Connection) -> dict[str, float]:
    """Map content_hash -> demand for every record present in the graph.

    demand = 1 + log1p(reach), where reach is the count of OTHER records sharing
    at least one referenced node. log1p keeps one high-reach record from
    dominating and handles reach=0 gracefully. Only records with claims+nodes
    appear; everything else is implicitly baseline (absent from the map).
    """
    rows = conn.execute(
        """
        SELECT r.content_hash, COUNT(DISTINCT other.record_id) AS reach
        FROM records r
        JOIN claims c ON c.record_id = r.id
        JOIN claim_node_refs cnr ON cnr.claim_id = c.id
        JOIN claim_node_refs cnr2 ON cnr2.node_id = cnr.node_id
        JOIN claims other ON other.id = cnr2.claim_id AND other.record_id != r.id
        WHERE r.content_hash IS NOT NULL
        GROUP BY r.id
        """
    ).fetchall()
    demand: dict[str, float] = {}
    for content_hash, reach in rows:
        demand[_bare_hash(content_hash)] = round(1.0 + math.log1p(reach or 0), 3)
    return demand


def _bare_hash(h: str | None) -> str:
    return (h or "").removeprefix("sha256:").strip()


# --- Corpus-state readers (filesystem, no AI) ---


def _store_records(ingests_dir: Path) -> dict[str, Path]:
    """Map content_hash -> record markdown path for every record in the store.

    A record is a store/*.md that is not a sidecar (.review.json/.verification
    are not .md) and not a transient variant. The filename stem is the hash.
    """
    store = ingests_dir / "store"
    out: dict[str, Path] = {}
    if not store.is_dir():
        return out
    for md in store.glob("*.md"):
        stem = md.stem
        # Skip variant suffixes like ".v2" that would not be a bare hash.
        if "." in stem:
            stem = stem.split(".", 1)[0]
        if len(stem) == 64 and all(ch in "0123456789abcdef" for ch in stem):
            out.setdefault(stem, md)
    return out


def _digestible_hashes(ingests_dir: Path) -> set[str]:
    store = ingests_dir / "store"
    out: set[str] = set()
    if not store.is_dir():
        return out
    for sidecar in store.glob("*.review.json"):
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("digestible") is True:
            out.add(sidecar.name.removesuffix(".review.json"))
    return out


def _reviewed_hashes(ingests_dir: Path) -> set[str]:
    """Records that have a review sidecar at all (reviewed, regardless of verdict)."""
    store = ingests_dir / "store"
    out: set[str] = set()
    if not store.is_dir():
        return out
    for sidecar in store.glob("*.review.json"):
        out.add(sidecar.name.removesuffix(".review.json"))
    return out


def _digested_stems(digests_dir: Path) -> set[str]:
    records = digests_dir / "records"
    if not records.is_dir():
        return set()
    return {y.stem for y in records.glob("*.yaml")}


def _record_stem(md_path: Path, ingests_dir: Path) -> str | None:
    """Resolve a record's friendly stem (matches the digest filename) via the
    records/ symlink that points at this store file, if present."""
    records = ingests_dir / "records"
    if not records.is_dir():
        return None
    target = md_path.resolve()
    for link in records.glob("*.md"):
        try:
            if link.resolve() == target:
                return link.stem
        except OSError:
            continue
    return None


def _ingested_source_ids(ingests_dir: Path) -> set[str]:
    """Every source-byte identity already ingested, across the per-type hash
    inconsistency (record-format.md).

    A source file in sources/ is named by the hash of its own bytes. Whether that
    matches a record depends on the record type:
    - audio / pdf: content_hash is over the source bytes, so it equals the
      store filename - covered by the store names.
    - web: content_hash is over the extracted body, but the record records the
      source bytes separately in `source_hash` - covered by that field.
    - ebook: content_hash is over the body and no source_hash is written, but the
      verification sidecar's `sha256` is the source-byte hash - covered there.

    Union all three so a web page or ebook already ingested is not re-listed as a
    pending ingest job.
    """
    store = ingests_dir / "store"
    ids: set[str] = set()
    if not store.is_dir():
        return ids
    for md in store.glob("*.md"):
        stem = md.stem.split(".", 1)[0]
        if len(stem) == 64:
            ids.add(stem)
        try:
            head = md.read_text(errors="ignore")[:8192]
        except OSError:
            continue
        m = _SOURCE_HASH_RE.search(head)
        if m:
            ids.add(m.group(1))
    for vj in store.glob("*.verification.json"):
        try:
            data = json.loads(vj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sh = _bare_hash(data.get("sha256"))
        if len(sh) == 64:
            ids.add(sh)
    return ids


def _pending_ingest(sources_dir: Path, ingested: set[str]) -> list[tuple[str, str]]:
    """List (hash, ext) for source files whose byte-hash is not yet ingested.

    Transcript sidecars (``{hash}.transcript.json``) are paired output of an
    audio/video source, not a source themselves; their stem is not a bare hash so
    they fall through the 64-hex check.
    """
    if not sources_dir.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for f in sources_dir.iterdir():
        if not f.is_file():
            continue
        h = f.stem
        if len(h) == 64 and h not in ingested:
            out.append((h, f.suffix.lstrip(".").lower()))
    return out


# --- Enumerators (one per real job type) ---


def enumerate_ingest_jobs(sources_dir: Path, ingested: set[str]) -> list[Job]:
    jobs: list[Job] = []
    for h, ext in sorted(_pending_ingest(sources_dir, ingested)):
        effort, type_label, lane = _INGEST_SPEC.get(
            ext, ("~extraction", ext or "unknown", LANE_EAGER)
        )
        jobs.append(
            Job(
                id=f"ingest:{h}",
                type="ingest",
                lane=lane,
                target=Target(kind="record", label=f"{type_label} {h[:12]}", hash=h),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                effort=effort,
                # Source/publisher priority is the intended ranker but is not yet
                # a first-class concept; surfaced as unranked rather than faked.
                drivers=[
                    Driver("source type", type_label),
                    Driver("source priority", "unranked", band="off"),
                ],
            )
        )
    return jobs


def enumerate_digest_jobs(
    ingests_dir: Path,
    digests_dir: Path,
    store: dict[str, Path],
    demand: dict[str, float],
) -> list[Job]:
    digestible = _digestible_hashes(ingests_dir)
    digested = _digested_stems(digests_dir)
    jobs: list[Job] = []
    for h in sorted(digestible):
        md = store.get(h)
        if md is None:
            continue
        stem = _record_stem(md, ingests_dir)
        if stem and stem in digested:
            continue  # already digested
        d = demand.get(h)
        jobs.append(
            Job(
                id=f"digest:{h}",
                type="digest",
                lane=LANE_CLAUDE,
                target=Target(kind="record", label=stem or f"record {h[:12]}", hash=h),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                value=d,
                drivers=[
                    Driver("readiness", "digestible", band="normal"),
                    Driver("demand", _demand_str(d), band="off" if d is None else None),
                ],
            )
        )
    return jobs


def enumerate_review_queue(
    ingests_dir: Path, store: dict[str, Path], demand: dict[str, float]
) -> list[ReviewItem]:
    reviewed = _reviewed_hashes(ingests_dir)
    items: list[ReviewItem] = []
    for h, md in store.items():
        if h in reviewed:
            continue  # has a review sidecar already
        d = demand.get(h)
        items.append(
            ReviewItem(
                target=Target(kind="record", label=f"record {h[:12]}", hash=h),
                demand=d,
                reason="never reviewed"
                if d is not None
                else "never reviewed (not yet in graph; demand baseline)",
            )
        )
    # Highest demand first; cold records (demand None) sort last, stable by hash.
    items.sort(
        key=lambda it: (it.demand is None, -(it.demand or 0.0), it.target.hash or "")
    )
    return items


def enumerate_graph_jobs(conn: sqlite3.Connection) -> list[Job]:
    """Corroborate pass plus its embedding prerequisite, from current graph state."""
    jobs: list[Job] = []
    total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    if total_claims == 0:
        return jobs
    embedded = _vec_claims_count(conn)
    recorded = conn.execute("SELECT COUNT(*) FROM corroborations").fetchone()[0]
    if embedded == 0:
        # Corroborate cannot run without embeddings; surface the eager embed job
        # that unblocks it rather than a corroborate job that would fail.
        jobs.append(
            Job(
                id="embed:claims",
                type="embed",
                lane=LANE_EAGER,
                target=Target(kind="page", label="all claims + nodes"),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                effort="~local CPU",
                drivers=[Driver("claims to embed", str(total_claims))],
            )
        )
        jobs.append(
            Job(
                id="corroborate:graph",
                type="corroborate",
                lane=LANE_CLAUDE,
                target=Target(kind="page", label="cross-record claim pairs"),
                status=STATUS_BLOCKED,
                blocker="embed:claims",
                trigger="never_done",
                drivers=[Driver("embeddings", "not computed", band="off")],
            )
        )
    else:
        jobs.append(
            Job(
                id="corroborate:graph",
                type="corroborate",
                lane=LANE_CLAUDE,
                target=Target(kind="page", label="cross-record claim pairs"),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                drivers=[
                    Driver("claims embedded", str(embedded)),
                    Driver("pairs confirmed", str(recorded), band="off"),
                ],
            )
        )
    return jobs


def _vec_claims_count(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM vec_claims").fetchone()[0]
    except sqlite3.OperationalError:
        return 0  # the sqlite-vec table is not present until `embed` runs


def _demand_str(d: float | None) -> str:
    return "baseline (not in graph)" if d is None else f"{d:.2f}"


# --- Queue assembly + output ---


def build_queue(
    conn: sqlite3.Connection,
    ingests_dir: Path,
    digests_dir: Path,
    sources_dir: Path,
    generated_at: str,
) -> dict:
    store = _store_records(ingests_dir)
    demand = compute_record_demand(conn)

    jobs: list[Job] = []
    jobs += enumerate_ingest_jobs(sources_dir, _ingested_source_ids(ingests_dir))
    jobs += enumerate_digest_jobs(ingests_dir, digests_dir, store, demand)
    jobs += enumerate_graph_jobs(conn)
    review_queue = enumerate_review_queue(ingests_dir, store, demand)

    return {
        "schema": "anomalica/schedule/0",
        "generatedAt": generated_at,
        "jobs": [j.to_dict() for j in jobs],
        "reviewQueue": [it.to_dict() for it in review_queue],
        "recordDemand": demand,
    }


def default_queue_path() -> Path:
    return Path(
        os.environ.get(
            "SCHEDULER_QUEUE_PATH",
            str(
                Path.home()
                / ".local"
                / "share"
                / "assimilator"
                / "scheduler-queue.json"
            ),
        )
    )


def write_queue(queue: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2, ensure_ascii=False))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
