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

import yaml

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

# A record that declares itself replaced. Retained in the store so a lookup by
# the old content_hash still finds a body, but never live: excluded from the
# record set so nothing schedules work against superseded text.
_SUPERSEDED_BY_RE = re.compile(r"^superseded_by:\s*\S+", re.M)


def _store_records(ingests_dir: Path) -> dict[str, Path]:
    """Map content_hash -> record markdown path for every record in the store.

    A record is a store/*.md that is not a sidecar (.review.json/.verification
    are not .md) and not a transient variant. The filename stem is the hash.

    A record declaring `superseded_by` is EXCLUDED. A body-normalising fix
    rehashes the record and mints a successor while the original is deliberately
    retained (the digester's redigest resolver looks bodies up by content_hash,
    so deleting one turns a stale read into a silently dropped record). Retained
    is not live: without this it would still be enumerated for digestion and for
    review, and would be re-digested against text the pipeline has replaced.
    Non-recursive by design - store/ also holds archive tiers whose records are
    superseded re-ingests of live ones.
    """
    store = ingests_dir / "store"
    out: dict[str, Path] = {}
    if not store.is_dir():
        return out
    for md in sorted(store.glob("*.md")):
        stem = md.stem
        # Skip variant suffixes like ".v2" that would not be a bare hash.
        if "." in stem:
            stem = stem.split(".", 1)[0]
        if len(stem) != 64 or not all(ch in "0123456789abcdef" for ch in stem):
            continue
        try:
            if _SUPERSEDED_BY_RE.search(md.read_text(errors="replace")[:4000]):
                continue
        except OSError:
            continue
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


def _digest_index(digests_dir: Path) -> dict[str, dict]:
    """Map content_hash -> {version, title} for every digest on disk.

    Keyed by ``record.content_hash`` (not the friendly filename), so an
    audio/video digest is detected despite the record's ``.v2`` store suffix that
    the filename stem carries but the digest filename does not. processing_version
    is the digest-freshness key (a digest is current only while it equals the
    record's current body version); title labels the import/digest jobs.
    """
    # Canonical digests sit at the root of digests/; the variants/ subtree is
    # deliberately not scanned. The previous records/ subdirectory is gone, and
    # with it the trap of passing either level and getting an empty index rather
    # than an error.
    records = digests_dir
    out: dict[str, dict] = {}
    if not records.is_dir():
        return out
    # rglob, not glob: a slash in a record title nests the digest in a
    # subdirectory; a non-recursive scan would miss it and re-dispatch the job
    # forever. Defence-in-depth alongside the flat-path keying upstream.
    for y in records.rglob("*.yaml"):
        try:
            data = yaml.safe_load(y.read_text())
        except (OSError, yaml.YAMLError):
            continue
        rec = (data or {}).get("record") or {}
        ch = _bare_hash(rec.get("content_hash"))
        if len(ch) == 64:
            out[ch] = {
                "version": rec.get("processing_version"),
                "title": rec.get("title"),
                "record_id": rec.get("id"),
            }
    return out


def _graph_record_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM records").fetchall()}


def _graph_record_hashes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT content_hash FROM records WHERE content_hash IS NOT NULL"
    ).fetchall()
    return {_bare_hash(r[0]) for r in rows}


def _record_frontmatter(md_path: Path) -> dict:
    try:
        text = md_path.read_text(errors="ignore")
    except OSError:
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def _record_processing_version(md_path: Path) -> str | None:
    return (_record_frontmatter(md_path).get("processing") or {}).get("version")


def _ingested_source_ids(ingests_dir: Path) -> set[str]:
    """Every source-byte identity already ingested, across the per-type hash
    inconsistency (ingest-format.md).

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


def _superseded_hashes(sources_dir: Path) -> set[str]:
    """Hashes of source files deliberately superseded - re-ingested under a new,
    stable hash so the content lives elsewhere - excluded from the ingest lane.

    Maintained as a flat file (one 64-hex hash per line, ``#`` comments allowed)
    so the list grows without a code change. Default location:
    ``<sources>/superseded.txt`` (override ANOMALICA_SUPERSEDED_FILE). Preferred
    over parsing git delete-messages at schedule time, which is fragile.
    """
    path = Path(
        os.environ.get("ANOMALICA_SUPERSEDED_FILE", str(sources_dir / "superseded.txt"))
    )
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        h = line.split("#", 1)[0].strip()
        if len(h) == 64:
            out.add(h)
    return out


def _default_briefs_dir() -> Path:
    return Path(
        os.environ.get(
            "ANOMALICA_BRIEFS_DIR",
            str(Path.home() / ".local" / "share" / "assimilator" / "briefs"),
        )
    )


def enumerate_ingest_jobs(
    sources_dir: Path, ingested: set[str], superseded: set[str] = frozenset()
) -> list[Job]:
    jobs: list[Job] = []
    for h, ext in sorted(_pending_ingest(sources_dir, ingested)):
        if h in superseded:
            continue  # re-ingested under a new hash; not real pending work
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


def enumerate_import_jobs(
    conn: sqlite3.Connection, digest_index: dict[str, dict]
) -> list[Job]:
    """A digest on disk whose record is not yet in the graph is a pending import.

    Import is the deterministic fold of a digest into the graph - no Claude, no
    money - so it is an eager light-local job. Surfacing it makes the downstream
    work visible in the schedule and lets the runner's eager worker flow a freshly
    produced digest into the graph instead of dead-ending after digestion.

    Imported if the digest's record.id OR its content_hash is already in the
    graph. record.id is the primary key (always present); content_hash is the
    fallback for an id-less digest. Keying on content_hash alone would falsely
    flag a record imported without its ingests dir (null content_hash in the
    graph) as "not imported".
    """
    in_graph_ids = _graph_record_ids(conn)
    in_graph_hashes = _graph_record_hashes(conn)
    jobs: list[Job] = []
    for h in sorted(digest_index):
        record_id = digest_index[h].get("record_id")
        if (record_id and record_id in in_graph_ids) or h in in_graph_hashes:
            continue  # already imported
        title = digest_index[h].get("title") or f"record {h[:12]}"
        jobs.append(
            Job(
                id=f"import:{h}",
                type="import",
                lane=LANE_EAGER,
                target=Target(kind="record", label=title, hash=h),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                effort="~local import",
                drivers=[Driver("source", "digest on disk, not yet in graph")],
            )
        )
    return jobs


def enumerate_digest_jobs(
    ingests_dir: Path,
    digest_index: dict[str, dict],
    store: dict[str, Path],
    demand: dict[str, float],
) -> list[Job]:
    digestible = _digestible_hashes(ingests_dir)
    jobs: list[Job] = []
    for h in sorted(digestible):
        md = store.get(h)
        if md is None:
            continue
        trigger = "never_done"
        if h in digest_index:
            digest_ver = digest_index[h].get("version")
            record_ver = _record_processing_version(md)
            # Missing-safe (matches anomalica_common.staleness): only re-digest
            # when both versions are present and differ. A current digest -> the
            # job is done, so it drops; this is the completion signal the runner
            # relies on to stop re-spending tokens on an already-digested record.
            if not (digest_ver and record_ver and str(digest_ver) != str(record_ver)):
                continue
            trigger = "stale"
        d = demand.get(h)
        fm = _record_frontmatter(md)
        label = fm.get("friendly_name") or fm.get("title") or f"record {h[:12]}"
        drivers = [
            Driver("readiness", "digestible", band="normal"),
            Driver("demand", _demand_str(d), band="off" if d is None else None),
        ]
        if trigger == "stale":
            drivers.insert(0, Driver("freshness", "body re-extracted", band="urgent"))
        jobs.append(
            Job(
                id=f"digest:{h}",
                type="digest",
                lane=LANE_CLAUDE,
                target=Target(kind="record", label=label, hash=h),
                status=STATUS_ELIGIBLE,
                trigger=trigger,
                value=d,
                drivers=drivers,
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


def _load_briefs(briefs_dir: Path | None) -> list[dict]:
    """Read the emitted briefs once (their page identity + brief_hash). Shared by
    the synthesise and assemble enumerators so the briefs dir is scanned once."""
    out: list[dict] = []
    if briefs_dir is None or not briefs_dir.is_dir():
        return out
    for bf in sorted(briefs_dir.glob("*.yaml")):
        try:
            brief = yaml.safe_load(bf.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        out.append(brief)
    return out


def enumerate_synthesise_jobs(
    conn: sqlite3.Connection, briefs: list[dict]
) -> list[Job]:
    """An entity page whose brief is missing OR out of date is a pending job.

    A brief is only worth having if it reflects the graph it claims to summarise.
    This used to skip any node that had one, so a brief was written once and never
    again: on 2026-08-20 all 225 on disk were built from a graph state of
    2026-06-28, and the Whitley Strieber brief carried 4 claims where the node held
    2071. The assembler reads briefs, so every page built from one was summarising
    a corpus a fraction of the real size, and nothing anywhere reported it.

    Staleness is the brief's recorded graph_version against the graph's current one
    (the latest claim mutation). Coarse on purpose: any new claim can change a
    brief's selection, its related nodes or its counts, so anything short of an
    exact match is stale. Regenerating is free and takes seconds.

    Synthesise is deterministic (graph slice -> brief, no Claude), so it is eager.
    Matched by node_id (the brief carries page.node_id) rather than by slug, so
    this needs no slugifier here - keeping the scheduler host-light - and is
    robust to slug collisions. The page set is the proposal table (propose-pages
    decides; this consumes it), so synthesise is naturally gated behind proposal-
    gen - no proposals, no synthesise jobs.
    """
    from assimilator.propose_pages import proposed_node_ids

    from assimilator.synthesise import _graph_version

    current = _graph_version(conn)
    # node_id -> the graph state its brief was built from.
    have = {
        (b.get("page") or {}).get("node_id"): (b.get("generated") or {}).get(
            "graph_version"
        )
        for b in briefs
    }
    page_ids = set(proposed_node_ids(conn))  # only proposed entities deserve a page
    rows = conn.execute(
        "SELECT id, name, node_type FROM nodes WHERE retired_at IS NULL ORDER BY name"
    ).fetchall()
    jobs: list[Job] = []
    for node_id, name, node_type in rows:
        if node_id not in page_ids:
            continue  # not proposed
        fresh = node_id in have and have[node_id] == current and current is not None
        if fresh:
            continue  # brief already reflects this graph
        jobs.append(
            Job(
                id=f"synthesise:{node_id}",
                type="synthesise",
                lane=LANE_EAGER,
                target=Target(kind="page", label=name),
                status=STATUS_ELIGIBLE,
                trigger="never_done" if node_id not in have else "stale_brief",
                effort="~local graph slice",
                drivers=[Driver("node type", node_type)],
            )
        )
    return jobs


def _article_brief_hashes(content_dir: Path | None) -> set[str]:
    """The brief_hash each assembled article was frozen from (its built_from)."""
    out: set[str] = set()
    if not content_dir or not content_dir.is_dir():
        return out
    for md in content_dir.rglob("*.md"):
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            fm = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            continue
        built = fm.get("built_from") or {}
        if isinstance(built, dict) and built.get("brief_hash"):
            out.add(built["brief_hash"])
    return out


def enumerate_assemble_jobs(briefs: list[dict], content_dir: Path | None) -> list[Job]:
    """A brief whose brief_hash is not frozen into any article is a pending
    assemble job - the AI writer step, so it is Claude-lane. A brief whose hash
    changed (graph moved) re-appears here automatically: the old article's
    built_from no longer matches."""
    assembled = _article_brief_hashes(content_dir)
    jobs: list[Job] = []
    for brief in briefs:
        brief_hash = brief.get("brief_hash")
        page = brief.get("page") or {}
        if not brief_hash or brief_hash in assembled:
            continue
        jobs.append(
            Job(
                id=f"assemble:{page.get('slug') or brief_hash[:12]}",
                type="assemble",
                lane=LANE_CLAUDE,
                target=Target(kind="page", label=page.get("title") or "page"),
                status=STATUS_ELIGIBLE,
                trigger="never_done",
                drivers=[Driver("claims", str(page.get("claim_count", "?")))],
            )
        )
    return jobs


def _proposal_table_stale(conn: sqlite3.Connection) -> tuple[bool, int]:
    """Does the derived page_proposals table reflect the current gate? Returns
    (stale, gate_count). Stale when the gate-passing (non-vetoed) node set differs
    from what is recorded - a recompute is then a pending propose-pages job."""
    from assimilator.page_gate import page_gate_rows
    from assimilator.propose_pages import vetoed_node_ids

    vetoed = vetoed_node_ids(conn)
    gate_ids = {r["node_id"] for r in page_gate_rows(conn)} - vetoed
    try:
        proposed = {
            r[0] for r in conn.execute("SELECT node_id FROM page_proposals").fetchall()
        }
    except sqlite3.OperationalError:
        proposed = set()
    return (gate_ids != proposed, len(gate_ids))


def enumerate_graph_jobs(conn: sqlite3.Connection) -> list[Job]:
    """Proposal-gen and the corroborate pass (plus its embedding prerequisite),
    from current graph state."""
    jobs: list[Job] = []
    total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    if total_claims == 0:
        return jobs

    stale, gate_count = _proposal_table_stale(conn)
    if stale:
        # Proposal-gen is deterministic (gate + ledger, no Claude), so eager. It
        # gates synthesise: a brief is only emitted for a proposed node, so this
        # must run after import/merge and before synthesise.
        jobs.append(
            Job(
                id="propose-pages:graph",
                type="propose-pages",
                lane=LANE_EAGER,
                target=Target(kind="page", label="article proposals"),
                status=STATUS_ELIGIBLE,
                trigger="stale",
                effort="~local graph scan",
                drivers=[Driver("nodes passing gate", str(gate_count))],
            )
        )
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
    briefs_dir: Path | None = None,
    content_dir: Path | None = None,
) -> dict:
    store = _store_records(ingests_dir)
    demand = compute_record_demand(conn)
    digest_index = _digest_index(digests_dir)

    jobs: list[Job] = []
    jobs += enumerate_ingest_jobs(
        sources_dir, _ingested_source_ids(ingests_dir), _superseded_hashes(sources_dir)
    )
    jobs += enumerate_digest_jobs(ingests_dir, digest_index, store, demand)
    jobs += enumerate_import_jobs(conn, digest_index)
    briefs = _load_briefs(briefs_dir)
    jobs += enumerate_synthesise_jobs(conn, briefs)
    jobs += enumerate_assemble_jobs(briefs, content_dir)
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


def resolve_corpus_dirs(
    ingests: str | None = None,
    digests: str | None = None,
    sources: str | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the ingests/digests/sources dirs from args, env, or sibling repos.

    The Anomalica repos live side by side, so an unspecified path defaults to a
    sibling of this repo (…/anomalica/{ingests,digests,sources}). Env overrides:
    ANOMALICA_INGESTS_DIR / ANOMALICA_DIGESTS_DIR / ANOMALICA_SOURCES_DIR.
    """
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    ingests = ingests or os.environ.get("ANOMALICA_INGESTS_DIR")
    digests = digests or os.environ.get("ANOMALICA_DIGESTS_DIR")
    sources = sources or os.environ.get("ANOMALICA_SOURCES_DIR")
    return (
        Path(ingests) if ingests else root / "ingests",
        Path(digests) if digests else root / "digests",
        Path(sources) if sources else root / "sources",
    )


def run_schedule(
    db_path: str | Path,
    ingests: str | None = None,
    digests: str | None = None,
    sources: str | None = None,
    out: str | None = None,
) -> tuple[dict, Path]:
    """Build the queue from current corpus state and write it. Read-only on the
    graph DB - enumeration never mutates the live database."""
    ingests_dir, digests_dir, sources_dir = resolve_corpus_dirs(
        ingests, digests, sources
    )
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    content_dir = Path(os.environ.get("ANOMALICA_CONTENT_DIR", str(root / "content")))
    out_path = Path(out) if out else default_queue_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        queue = build_queue(
            conn,
            ingests_dir,
            digests_dir,
            sources_dir,
            now_iso(),
            briefs_dir=_default_briefs_dir(),
            content_dir=content_dir,
        )
    finally:
        conn.close()
    write_queue(queue, out_path)
    return queue, out_path


def main(argv: list[str] | None = None) -> int:
    """Host-runnable entry point: `python -m assimilator.scheduler`.

    Deliberately depends on nothing beyond the standard library + pyyaml (no
    click, no fastembed, no anomalica_common), so a plain host process - such as
    the workbench's uvicorn backend - can regenerate the queue without the
    container-magic `assimilator` tool or the heavy CLI imports.
    """
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    parser = argparse.ArgumentParser(
        prog="assimilator.scheduler",
        description="Emit the real pending-job queue from current corpus state.",
    )
    parser.add_argument("--db", default=default_db, help="graph DB (read-only)")
    parser.add_argument("--ingests", default=None)
    parser.add_argument("--digests", default=None)
    parser.add_argument("--sources", default=None)
    parser.add_argument("--out", default=None, help="queue JSON path")
    args = parser.parse_args(argv)

    queue, out_path = run_schedule(
        args.db, args.ingests, args.digests, args.sources, args.out
    )
    by_lane: dict[str, int] = {}
    for job in queue["jobs"]:
        by_lane[job["lane"]] = by_lane.get(job["lane"], 0) + 1
    lanes = ", ".join(f"{n} {lane}" for lane, n in sorted(by_lane.items()))
    print(f"Wrote {out_path}")
    print(
        f"  {len(queue['jobs'])} jobs ({lanes}), {len(queue['reviewQueue'])} awaiting review"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
