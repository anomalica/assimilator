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

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from assimilator.digest_files import (
    CURRENT_IMPORT_GENERATION,
    canonical_digests,
    digest_file_identity,
)

from assimilator.embed_batches import BUCKETS, pending_by_bucket
from assimilator.data_dir import data_dir

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
    kind: str  # "record" | "page" | "graph"
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
    held_by: str | None = None
    article: str | None = None
    # The exact argv that performs this job, for jobs whose command lives in THIS
    # repo. A runner deriving it from the id ("embed:claims:7" -> --bucket 7) puts
    # the same assumption in two repos, where only one of them gets updated.
    # None where the command belongs to another component (ingest, digest).
    command: list[str] | None = None
    local_reason_groups: list[dict] = field(default_factory=list)
    inherited_reason_groups: list[dict] = field(default_factory=list)
    consequence: str = "new"
    native_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "type": self.type,
            "lane": self.lane,
            "target": self.target.to_dict(),
            "status": self.status,
            "trigger": self.trigger,
            "local_reason_groups": self.local_reason_groups,
            "inherited_reason_groups": self.inherited_reason_groups,
            "consequence": self.consequence,
            "native_metrics": self.native_metrics,
        }
        if self.drivers:
            d["drivers"] = [dr.to_dict() for dr in self.drivers]
        if self.value is not None:
            d["value"] = self.value
        if self.effort is not None:
            d["effort"] = self.effort
        if self.command is not None:
            d["command"] = list(self.command)
        if self.blocker is not None:
            d["blocker"] = self.blocker
        if self.held_by is not None:
            d["heldBy"] = self.held_by
        if self.article is not None:
            d["article"] = self.article
        return d


def _reason_group(
    boundary: str,
    artifact: str,
    local_status: str,
    reasons: list[str],
    consequence: str = "finish",
) -> dict:
    return {
        "boundary": boundary,
        "artifact": artifact,
        "local_status": local_status,
        "local_reasons": sorted(set(reasons)),
        "inherited": [],
        "consequence": consequence,
    }


def _deduplicate_groups(*collections: list[dict]) -> list[dict]:
    """Union canonical groups by boundary and artifact without reason fan-out."""
    merged: dict[tuple[str, str], dict] = {}
    status_rank = {"current": 0, "missing": 1, "unknown": 2, "stale": 3, "invalid": 4}
    consequence_rank = {"new": 0, "verify": 1, "finish": 2, "repair": 3}
    for group in (g for groups in collections for g in groups):
        key = (str(group["boundary"]), str(group["artifact"]))
        current = merged.get(key)
        if current is None:
            merged[key] = {
                **group,
                "local_reasons": sorted(set(group.get("local_reasons") or [])),
                "inherited": [],
            }
            continue
        current["local_reasons"] = sorted(
            set(current["local_reasons"]) | set(group.get("local_reasons") or [])
        )
        if status_rank.get(group.get("local_status"), 0) > status_rank.get(
            current.get("local_status"), 0
        ):
            current["local_status"] = group["local_status"]
        if consequence_rank.get(group.get("consequence"), 0) > consequence_rank.get(
            current.get("consequence"), 0
        ):
            current["consequence"] = group["consequence"]
    return [merged[key] for key in sorted(merged)]


def _max_consequence(groups: list[dict], default: str = "new") -> str:
    rank = {"new": 0, "verify": 1, "finish": 2, "repair": 3}
    return max(
        (g.get("consequence", default) for g in groups), key=rank.get, default=default
    )


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
    # Two passes over 65,000 rows, not a self-join across them. The previous
    # query joined claim_node_refs to itself on node_id, which materialises one
    # row per PAIR of claims sharing a node - and the busiest node has 1,422
    # references, so that node alone contributes two million rows. It cost 33.6s
    # of a queue rebuild whose whole budget is 180s, and it grows quadratically
    # with the corpus while the corpus grows linearly.
    #
    # The set formulation is the same definition read directly: which records
    # does each node appear in, and how many OTHER records does a record reach
    # through its own nodes.
    node_records: dict[str, set[str]] = {}
    record_nodes: dict[str, set[str]] = {}
    for record_id, node_id in conn.execute(
        """
        SELECT c.record_id, cnr.node_id
        FROM claim_node_refs cnr
        JOIN claims c ON c.id = cnr.claim_id
        """
    ):
        node_records.setdefault(node_id, set()).add(record_id)
        record_nodes.setdefault(record_id, set()).add(node_id)

    hashes = {
        rid: h
        for rid, h in conn.execute(
            "SELECT id, content_hash FROM records WHERE content_hash IS NOT NULL"
        )
    }
    demand: dict[str, float] = {}
    for record_id, nodes in record_nodes.items():
        content_hash = hashes.get(record_id)
        if content_hash is None:
            continue
        reach = set()
        for node_id in nodes:
            reach |= node_records[node_id]
        reach.discard(record_id)
        if not reach:
            # A record sharing no node with any other is OMITTED, matching the
            # previous query, whose `other.record_id != r.id` join produced no
            # row for it. Preserved deliberately rather than tidied: the caller
            # renders an absent record as "baseline (not in graph)", which is
            # wrong for a record that IS in the graph but reaches nothing. That
            # is a display bug worth fixing on its own, not inside a change whose
            # whole claim is that the output is identical.
            continue
        demand[_bare_hash(content_hash)] = round(1.0 + math.log1p(len(reach)), 3)
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
    """Map content_hash to exact canonical digest identity and freshness header.

    Keyed by ``record.content_hash`` (not the friendly filename). The legacy
    processing version is retained for old API consumers but is never used as a
    freshness proof.
    """
    # VARIANTS ARE EXCLUDED BY canonical_digests, NOT BY THIS COMMENT. The
    # previous version said the variants/ subtree was "deliberately not scanned"
    # and then scanned it with rglob, so every model-comparison snapshot entered
    # the index as though it were an importable digest. Five import jobs were
    # emitted for records whose only artefact is a variant - work that can never
    # succeed, because pick_top_import_job wants a canonical digest.
    #
    # Recursive on purpose: a slash in a record title nests the digest in a
    # subdirectory, and a flat scan would miss it and re-dispatch forever.
    # canonical_digests globs **/*.yaml and drops anything under variants/.
    out: dict[str, dict] = {}
    if not digests_dir.is_dir():
        return out
    for y in canonical_digests(digests_dir):
        header = _digest_header(y)
        rec = header.get("record") or {}
        ch = _bare_hash(rec.get("content_hash"))
        if len(ch) == 64:
            identity = digest_file_identity(y, digests_dir)
            out[ch] = {
                "version": rec.get("processing_version"),
                "title": rec.get("title"),
                "record_id": rec.get("id"),
                "schema": header.get("schema"),
                "pre_digest": header.get("pre_digest"),
                "extraction_generation": header.get("extraction_generation"),
                "extraction_config": header.get("extraction_config"),
                "_path": y,
                **identity,
            }
    return out


def _digest_header(path: Path) -> dict:
    """The bounded metadata header of a digest, without parsing its claim lists.

    Four fields are wanted from a document that runs to 14,000 lines and 1,800
    claims, and yaml.safe_load on the whole corpus cost 54 seconds of every queue
    rebuild - enough on its own to push the rebuild past the runner's 180-second
    timeout, so the queue never refreshed and work added by other components
    stayed invisible.

    The format puts all freshness fields before ``terminology``. Falls back to a
    full parse when that bounded header does not contain a record.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}

    block: list[str] = []
    for line in lines:
        if not line[:1].isspace() and line.startswith("terminology:"):
            break
        block.append(line)

    if block:
        try:
            parsed = yaml.safe_load("\n".join(block)) or {}
            if isinstance(parsed.get("record"), dict):
                return parsed
        except yaml.YAMLError:
            pass

    try:  # not where the format says it should be - pay for the full parse
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _digest_record_header(path: Path) -> dict:
    """Compatibility helper retained for callers that only need ``record``."""
    return _digest_header(path).get("record") or {}


def _current_extraction_generation(digests_dir: Path) -> int | None:
    """Read the corpus authority; malformed or absent manifests stay unknown."""
    try:
        document = json.loads((digests_dir / "digest-generation.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    generation = (
        document.get("current_generation") if isinstance(document, dict) else None
    )
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation <= 0
        or document.get("schema") != "anomalica/digest-generation/1"
    ):
        return None
    return generation


def _resolvable_extraction_configs(digests_dir: Path) -> set[str]:
    """Return only registry entries whose key matches their canonical payload."""
    try:
        document = json.loads(
            (digests_dir / "extraction-configurations.json").read_text()
        )
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(document, dict):
        return set()
    if document.get("schema") == "anomalica/digest-extraction-config-registry/1":
        configurations = document.get("configurations")
    elif "schema" not in document:  # legacy direct fingerprint map
        configurations = document
    else:
        return set()
    if not isinstance(configurations, dict):
        return set()
    resolvable: set[str] = set()
    for fingerprint, configuration in configurations.items():
        if (
            not isinstance(configuration, dict)
            or configuration.get("configuration_schema")
            != "anomalica/digest-extraction-config/1"
        ):
            continue
        canonical = json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        actual = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        if fingerprint == actual:
            resolvable.add(fingerprint)
    return resolvable


def _record_body(text: str) -> str:
    """Extract the record body while dropping only recognised annotation fences."""
    lines = text.split("\n")
    start = 0
    if lines and lines[0].strip() == "---":
        closing = next(
            (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
            None,
        )
        if closing is not None:
            start = closing + 1
    body: list[str] = []
    index = start
    annotation_keys = {"file_page", "printed_page", "chapter", "speaker", "image"}
    while index < len(lines):
        if lines[index].strip() == "---":
            closing = next(
                (
                    candidate
                    for candidate in range(index + 1, min(len(lines), index + 42))
                    if lines[candidate].strip() == "---"
                ),
                None,
            )
            if closing is not None:
                try:
                    annotation = yaml.safe_load("\n".join(lines[index + 1 : closing]))
                except yaml.YAMLError:
                    annotation = None
                if isinstance(annotation, dict) and annotation_keys & set(annotation):
                    index = closing + 1
                    continue
        body.append(lines[index])
        index += 1
    return "\n".join(body).strip()


def _pre_digest_input_state(digest: dict, record_path: Path | None) -> tuple[str, str]:
    """Compare the digest's exact model input with the current live record."""
    from anomalica_common.pre_digest import materialise, pre_digest_hash

    pre_digest = digest.get("pre_digest")
    if not isinstance(pre_digest, dict):
        return "invalid", "pre_digest_binding_unknown"
    recorded = pre_digest.get("sha256")
    if recorded is None:
        return "unknown", "pre_digest_binding_unknown"
    if not isinstance(recorded, str) or re.fullmatch(r"[0-9a-f]{64}", recorded) is None:
        return "invalid", "pre_digest_binding_unknown"
    if record_path is None:
        return "unknown", "pre_digest_binding_unknown"
    try:
        body = _record_body(record_path.read_text(errors="replace"))
        actual = pre_digest_hash(materialise(body))
    except (OSError, ValueError, TypeError):
        return "unknown", "pre_digest_binding_unknown"
    if actual != recorded:
        return "stale", "pre_digest_hash_mismatch"
    return "current", "pre_digest_hash_match"


def digest_freshness(
    ingests_dir: Path,
    digests_dir: Path,
    digest_index: dict[str, dict],
    store: dict[str, Path],
) -> tuple[dict[str, list[dict]], dict]:
    """Canonical digest boundary groups from the current record and manifests."""
    groups: dict[str, list[dict]] = {h: [] for h in set(store) | set(digest_index)}
    metrics = {
        "input_binding_match": 0,
        "input_binding_mismatch": 0,
        "input_binding_unknown": 0,
        "generation_current": 0,
        "generation_behind": 0,
        "generation_unknown": 0,
        "schema_invalid": 0,
        "config_invalid": 0,
    }
    for h in sorted(store):
        if h not in digest_index:
            groups[h].append(
                _reason_group(
                    "digest-input", f"sha256:{h}", "missing", ["digest_missing"]
                )
            )
    if not digest_index:
        return groups, metrics

    current_generation = _current_extraction_generation(digests_dir)
    resolvable_configs = _resolvable_extraction_configs(digests_dir)
    for h, digest in digest_index.items():
        input_reasons: list[str] = []
        input_status = "current"
        if digest.get("schema") != "anomalica/digest/1":
            input_reasons.append("schema_unsupported")
            input_status = "invalid"
            metrics["schema_invalid"] += 1
        state, issue = _pre_digest_input_state(digest, store.get(h))
        if h not in store:
            input_reasons.append("record_binding_mismatch")
            input_status = "stale"
            metrics["input_binding_mismatch"] += 1
        elif state == "unknown":
            input_reasons.append("pre_digest_binding_unknown")
            input_status = "unknown"
            metrics["input_binding_unknown"] += 1
        elif state == "invalid":
            input_reasons.append("pre_digest_binding_unknown")
            input_status = "invalid"
            metrics["input_binding_unknown"] += 1
        elif state == "stale":
            input_reasons.append(issue)
            input_status = "stale"
            metrics["input_binding_mismatch"] += 1
        else:
            metrics["input_binding_match"] += 1
        extraction_config = digest.get("extraction_config")
        if (
            not isinstance(extraction_config, str)
            or extraction_config not in resolvable_configs
        ):
            input_reasons.append("extraction_config_invalid")
            input_status = "invalid"
            metrics["config_invalid"] += 1
        if input_reasons:
            groups[h].append(
                _reason_group(
                    "digest-input", f"sha256:{h}", input_status, input_reasons
                )
            )

        generation = digest.get("extraction_generation")
        if current_generation is not None and generation == current_generation:
            metrics["generation_current"] += 1
        elif (
            current_generation is not None
            and isinstance(generation, int)
            and not isinstance(generation, bool)
            and 0 < generation < current_generation
        ):
            metrics["generation_behind"] += 1
            groups[h].append(
                _reason_group(
                    "digest-generation",
                    digest["digest_path"],
                    "stale",
                    ["generation_behind"],
                )
            )
        else:
            metrics["generation_unknown"] += 1
            groups[h].append(
                _reason_group(
                    "digest-generation",
                    digest["digest_path"],
                    "unknown",
                    ["generation_unknown"],
                )
            )
        groups[h] = _deduplicate_groups(groups[h])
    return groups, metrics


def _graph_record_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM records").fetchall()}


def _graph_record_hashes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT content_hash FROM records WHERE content_hash IS NOT NULL"
    ).fetchall()
    return {_bare_hash(r[0]) for r in rows}


def _import_receipts(conn: sqlite3.Connection) -> dict[str, dict]:
    try:
        rows = conn.execute(
            "SELECT record_content_hash, record_id, digest_path, digest_sha256, "
            "import_generation FROM digest_import_receipts"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {
        _bare_hash(row[0]): {
            "record_id": row[1],
            "digest_path": row[2],
            "digest_sha256": row[3],
            "import_generation": row[4],
        }
        for row in rows
    }


def graph_import_native_deltas(
    conn: sqlite3.Connection, digest_index: dict[str, dict]
) -> dict:
    """Deterministic stage-native digest-to-graph delta counts."""
    receipts = _import_receipts(conn)
    graph_ids = _graph_record_ids(conn)
    graph_hashes = _graph_record_hashes(conn)
    counts = {
        "current": 0,
        "missing": 0,
        "changed": 0,
        "orphan": 0,
        "canonical_digests_missing_from_graph": 0,
        "receipt_hash_mismatches": 0,
        "import_generation_behind": 0,
        "import_generation_unknown": 0,
        "graph_records_without_live_canonical_digest": 0,
    }
    for content_hash, digest in digest_index.items():
        receipt = receipts.get(content_hash)
        if receipt is None:
            counts["missing"] += 1
            counts["canonical_digests_missing_from_graph"] += 1
            continue
        changed = False
        if receipt["record_id"] not in graph_ids:
            counts["canonical_digests_missing_from_graph"] += 1
            changed = True
        if (
            receipt["digest_sha256"] != digest["digest_sha256"]
            or receipt["digest_path"] != digest["digest_path"]
        ):
            counts["receipt_hash_mismatches"] += 1
            changed = True
        generation = receipt["import_generation"]
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation > CURRENT_IMPORT_GENERATION
        ):
            counts["import_generation_unknown"] += 1
            changed = True
        elif generation < CURRENT_IMPORT_GENERATION:
            counts["import_generation_behind"] += 1
            changed = True
        if changed:
            counts["changed"] += 1
        else:
            counts["current"] += 1
    counts["orphan"] = len((set(receipts) | graph_hashes) - set(digest_index))
    counts["graph_records_without_live_canonical_digest"] = counts["orphan"]
    return counts


def import_deltas(conn: sqlite3.Connection, digest_index: dict[str, dict]) -> dict:
    """Compatibility summary retained for existing scheduler consumers."""
    native = graph_import_native_deltas(conn, digest_index)
    return {key: native[key] for key in ("current", "missing", "changed", "orphan")}


def _curation_sha256() -> str:
    root = Path(
        os.environ.get(
            "ANOMALICA_CURATION_DIR",
            str(Path(__file__).resolve().parents[3] / "curation"),
        )
    )
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
    if not files:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    manifest = json.dumps(
        [
            [str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()]
            for p in files
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def graph_input_diagnostics(
    conn: sqlite3.Connection, digest_index: dict[str, dict], digests_dir: Path
) -> dict:
    receipts = _import_receipts(conn)
    triples = sorted(
        [h, receipt["digest_path"], receipt["digest_sha256"]]
        for h, receipt in receipts.items()
    )
    payload = json.dumps(
        {
            "import_generation": CURRENT_IMPORT_GENERATION,
            "digests": triples,
            "curation_sha256": _curation_sha256(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    by_hash: dict[str, list[str]] = {}
    by_record: dict[str, list[str]] = {}
    for path in canonical_digests(digests_dir):
        record = _digest_record_header(path)
        h = _bare_hash(record.get("content_hash"))
        rid = str(record.get("id") or "")
        if h:
            by_hash.setdefault(h, []).append(str(path))
        if rid:
            by_record.setdefault(rid, []).append(str(path))
    duplicates = [
        {
            "binding": f"sha256:{key}" if kind == "content_hash" else key,
            "kind": kind,
            "paths": paths,
        }
        for kind, bindings in (("content_hash", by_hash), ("record_id", by_record))
        for key, paths in sorted(bindings.items())
        if len(paths) > 1
    ]
    return {
        "fingerprint": "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "input": json.loads(payload),
        "duplicate_binding_count": len(duplicates),
        "duplicate_bindings": duplicates,
    }


def graph_freshness_groups(
    conn: sqlite3.Connection,
    digest_index: dict[str, dict],
    digest_groups: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Upstream groups carried by each graph record's exact digest binding."""
    receipts = _import_receipts(conn)
    graph_ids = _graph_record_ids(conn)
    graph_hashes = _graph_record_hashes(conn)
    out: dict[str, list[dict]] = {}
    for h in sorted(set(digest_index) | set(digest_groups) | graph_hashes):
        digest = digest_index.get(h)
        receipt = receipts.get(h)
        reasons: list[str] = []
        status = "stale"
        if digest is None:
            reasons.append("orphan_record")
        elif receipt is None or receipt.get("record_id") not in graph_ids:
            reasons.append("import_missing")
            status = "missing"
        else:
            if receipt.get("digest_sha256") != digest.get(
                "digest_sha256"
            ) or receipt.get("digest_path") != digest.get("digest_path"):
                reasons.append("digest_hash_mismatch")
            generation = receipt.get("import_generation")
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation > CURRENT_IMPORT_GENERATION
            ):
                reasons.append("import_generation_unknown")
                status = "unknown"
            elif generation < CURRENT_IMPORT_GENERATION:
                reasons.append("import_generation_behind")
        local = (
            [_reason_group("graph-import", f"sha256:{h}", status, reasons)]
            if reasons
            else []
        )
        out[h] = _deduplicate_groups(local, digest_groups.get(h, []))
    return out


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


def record_generation_freshness(
    ingests_dir: Path, store: dict[str, Path]
) -> tuple[dict[str, list[dict]], dict]:
    """Canonical live-record generation groups from the ingester manifest."""
    manifest_path = ingests_dir / "store" / "_pipeline_versions.yaml"
    try:
        document = yaml.safe_load(manifest_path.read_text())
    except (OSError, yaml.YAMLError):
        document = None
    manifest = document if isinstance(document, dict) else {}
    manifest_valid = isinstance(document, dict)
    all_record_hashes = {
        path.name.split(".", 1)[0]
        for path in (ingests_dir / "store").rglob("*.md")
        if len(path.name.split(".", 1)[0]) == 64
    }
    groups: dict[str, list[dict]] = {}
    metrics: dict = {
        "by_source_type": {},
        "generation_distance": {},
        "lineage_dangling": 0,
    }
    for content_hash, path in sorted(store.items()):
        frontmatter = _record_frontmatter(path)
        source_type = frontmatter.get("source_type")
        source_key = source_type if isinstance(source_type, str) else "unknown"
        counts = metrics["by_source_type"].setdefault(
            source_key,
            {"current": 0, "stale": 0, "unknown": 0, "invalid": 0},
        )
        reasons: list[str] = []
        status = "current"
        if frontmatter.get("schema") not in {
            "anomalica/record/1",
            "anomalica/record/2",
        }:
            reasons.append("schema_unsupported")
            status = "invalid"

        processing = frontmatter.get("processing")
        recorded = (
            processing.get("pipeline_version") if isinstance(processing, dict) else None
        )
        current = (
            manifest.get(source_type)
            if manifest_valid and isinstance(source_type, str)
            else None
        )
        known_recorded = (
            isinstance(recorded, int)
            and not isinstance(recorded, bool)
            and recorded > 0
        )
        known_current = (
            isinstance(current, int) and not isinstance(current, bool) and current > 0
        )
        if not known_recorded or not known_current or recorded > current:
            reasons.append("generation_unknown")
            if status == "current":
                status = "unknown"
        elif recorded < current:
            reasons.append("generation_behind")
            if status == "current":
                status = "stale"
            metrics["generation_distance"][f"sha256:{content_hash}"] = (
                current - recorded
            )

        supersedes = frontmatter.get("supersedes")
        lineage = supersedes if isinstance(supersedes, list) else [supersedes]
        if any(
            _bare_hash(str(parent)) not in all_record_hashes
            for parent in lineage
            if parent
        ):
            reasons.append("lineage_dangling")
            metrics["lineage_dangling"] += 1
            if status == "current":
                status = "unknown"

        counts[status] += 1
        groups[content_hash] = (
            [
                _reason_group(
                    "record-generation",
                    f"sha256:{content_hash}",
                    status,
                    reasons,
                )
            ]
            if reasons
            else []
        )
    return groups, metrics


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
            str(data_dir() / "briefs"),
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
    conn: sqlite3.Connection,
    digest_index: dict[str, dict],
    upstream_groups: dict[str, list[dict]] | None = None,
) -> list[Job]:
    """A digest whose exact receipt is absent or different is a pending import.

    Import is the deterministic fold of a digest into the graph - no Claude, no
    money - so it is an eager light-local job. Surfacing it makes the downstream
    work visible in the schedule and lets the runner's eager worker flow a freshly
    produced digest into the graph instead of dead-ending after digestion.

    Graph-row presence is not proof: canonical digest bytes may change in place.
    Legacy rows without a receipt are deliberately due once so the exact binding
    is established. Re-import remains the importer's replace-by-record fold.
    """
    in_graph_ids = _graph_record_ids(conn)
    in_graph_hashes = _graph_record_hashes(conn)
    receipts = _import_receipts(conn)
    upstream_groups = upstream_groups or {}
    jobs: list[Job] = []
    for h in sorted(digest_index):
        digest = digest_index[h]
        record_id = digest.get("record_id")
        receipt = receipts.get(h)
        reasons = []
        reason_codes = []
        metrics = {
            "import_missing": 0,
            "digest_hash_mismatch": 0,
            "import_generation_distance": None,
            "orphan_record": 0,
            "duplicate_binding": 0,
        }
        if receipt is None:
            reasons.append("import receipt missing")
            reason_codes.append("import_missing")
            metrics["import_missing"] = 1
        else:
            if receipt["record_id"] not in in_graph_ids:
                reasons.append("graph record missing")
                reason_codes.append("import_missing")
                metrics["import_missing"] = 1
            if receipt["digest_sha256"] != digest["digest_sha256"]:
                reasons.append("digest bytes changed")
                reason_codes.append("digest_hash_mismatch")
                metrics["digest_hash_mismatch"] = 1
            if receipt["digest_path"] != digest["digest_path"]:
                reasons.append("canonical digest path changed")
                reason_codes.append("digest_hash_mismatch")
                metrics["digest_hash_mismatch"] = 1
            generation = receipt["import_generation"]
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation > CURRENT_IMPORT_GENERATION
            ):
                reasons.append("import generation changed")
                reason_codes.append("import_generation_unknown")
            elif generation < CURRENT_IMPORT_GENERATION:
                reasons.append("import generation changed")
                reason_codes.append("import_generation_behind")
                metrics["import_generation_distance"] = (
                    CURRENT_IMPORT_GENERATION - generation
                )
        if not reasons:
            continue
        already_had_record = (
            record_id and record_id in in_graph_ids
        ) or h in in_graph_hashes
        title = digest.get("title") or f"record {h[:12]}"
        local_groups = [
            _reason_group(
                "graph-import",
                f"sha256:{h}",
                "missing" if "import_missing" in reason_codes else "stale",
                reason_codes,
            )
        ]
        jobs.append(
            Job(
                id=f"import:{h}",
                type="import",
                lane=LANE_EAGER,
                target=Target(kind="record", label=title, hash=h),
                status=STATUS_ELIGIBLE,
                trigger="stale" if already_had_record or receipt else "never_done",
                effort="~local import",
                drivers=[Driver("freshness", reason) for reason in reasons],
                local_reason_groups=local_groups,
                inherited_reason_groups=_deduplicate_groups(upstream_groups.get(h, [])),
                consequence="finish",
                native_metrics=metrics,
            )
        )
    return jobs


def enumerate_rebuild_jobs(
    conn: sqlite3.Connection,
    digest_index: dict[str, dict],
    graph_groups: dict[str, list[dict]] | None = None,
) -> list[Job]:
    """Expose canonical-set contraction without claiming imports can repair it."""
    orphan_records = graph_import_native_deltas(conn, digest_index)["orphan"]
    if not orphan_records:
        return []
    blocker = "isolated rebuild executor and explicit authorisation not implemented"
    consequence = (
        "build isolated candidate DB, replay curation, validate canonical and curated "
        "state, then atomically replace live DB under explicit authorisation"
    )
    graph_groups = graph_groups or graph_freshness_groups(conn, digest_index, {})
    orphan_groups = [
        group
        for content_hash in sorted(graph_groups)
        for group in graph_groups[content_hash]
        if group.get("boundary") == "graph-import"
        and "orphan_record" in group.get("local_reasons", [])
    ]
    return [
        Job(
            id="rebuild:graph",
            type="rebuild",
            lane=LANE_EAGER,
            target=Target(kind="graph", label="Knowledge graph"),
            status=STATUS_BLOCKED,
            trigger="canonical_set_contraction",
            effort="local isolated rebuild",
            blocker=blocker,
            held_by=blocker,
            drivers=[Driver("orphan records", str(orphan_records))],
            local_reason_groups=orphan_groups,
            consequence=consequence,
            native_metrics={"orphan_records": orphan_records},
        )
    ]


def enumerate_digest_jobs(
    ingests_dir: Path,
    digest_index: dict[str, dict],
    store: dict[str, Path],
    demand: dict[str, float],
    freshness_groups: dict[str, list[dict]] | None = None,
    upstream_groups: dict[str, list[dict]] | None = None,
) -> list[Job]:
    digestible = _digestible_hashes(ingests_dir)
    freshness_groups = freshness_groups or {}
    upstream_groups = upstream_groups or {}
    jobs: list[Job] = []
    for h in sorted(digestible):
        md = store.get(h)
        if md is None:
            continue
        trigger = "never_done"
        if h in digest_index:
            if not freshness_groups.get(h):
                continue
            trigger = "stale"
        d = demand.get(h)
        fm = _record_frontmatter(md)
        label = fm.get("friendly_name") or fm.get("title") or f"record {h[:12]}"
        drivers = [
            Driver("readiness", "digestible", band="normal"),
            Driver("demand", _demand_str(d), band="off" if d is None else None),
        ]
        local_groups = _deduplicate_groups(freshness_groups.get(h, []))
        if trigger == "stale":
            codes = sorted(
                {
                    code
                    for group in local_groups
                    for code in group.get("local_reasons", [])
                }
            )
            drivers.insert(0, Driver("freshness", ", ".join(codes), band="urgent"))
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
                local_reason_groups=local_groups,
                inherited_reason_groups=_deduplicate_groups(upstream_groups.get(h, [])),
                consequence="finish",
                native_metrics={
                    "input_groups": sum(
                        g["boundary"] == "digest-input" for g in local_groups
                    ),
                    "generation_groups": sum(
                        g["boundary"] == "digest-generation" for g in local_groups
                    ),
                },
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
    """The emitted briefs' headers, read once and shared by the synthesise and
    assemble enumerators.

    Headers only: the enumerators need three fields, and parsing every brief
    whole to get them was 105 of the 131 seconds a queue rebuild took."""
    from assimilator.synthesise import brief_files, brief_header

    out: list[dict] = []
    if briefs_dir is None or not briefs_dir.is_dir():
        return out
    for bf in brief_files(briefs_dir):
        header = brief_header(bf)
        if header:
            header["_reference"] = str(bf.relative_to(briefs_dir).with_suffix(""))
            header["_path"] = bf
            header["_claim_pairs"], header["_content_hashes"] = _brief_claim_audit(bf)
            out.append(header)
    return out


def _brief_claim_audit(path: Path) -> tuple[list[tuple[str, str]], set[str]]:
    """Read compact claim bindings from a brief without constructing bulk YAML."""
    claim_id = None
    pairs: list[tuple[str, str]] = []
    content_hashes: set[str] = set()
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return pairs, content_hashes
    for line in lines:
        stripped = line.strip().removeprefix("- ")
        if stripped.startswith("claim_id:"):
            claim_id = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped.startswith("claim_hash:") and claim_id is not None:
            claim_hash = stripped.split(":", 1)[1].strip().strip("'\"")
            pairs.append((claim_id, claim_hash))
            claim_id = None
        elif stripped.startswith("content_hash:"):
            value = _bare_hash(stripped.split(":", 1)[1].strip().strip("'\""))
            if value:
                content_hashes.add(value)
    return pairs, content_hashes


def enumerate_synthesise_jobs(
    conn: sqlite3.Connection,
    briefs: list[dict],
    graph_groups: dict[str, list[dict]] | None = None,
) -> list[Job]:
    """Compare every brief with the exact output of the deterministic writer.

    A brief is only worth having if it reflects the graph it claims to summarise.
    This used to skip any node that had one, so a brief was written once and never
    again: on 2026-08-20 all 225 on disk were built from a graph state of
    2026-06-28, and the Whitley Strieber brief carried 4 claims where the node held
    2071. The assembler reads briefs, so every page built from one was summarising
    a corpus a fraction of the real size, and nothing anywhere reported it.

    The recorded graph timestamp is audit information, not freshness proof. The
    current page set and writer are rerun, then their exact selection hash, page
    identity, publication decision and display-only listing tuple are compared.
    Unrelated graph changes therefore leave a brief settled, while claim removal,
    content/order drift, member changes and path/name changes are visible.

    Synthesise is deterministic (graph slice -> brief, no Claude), so it is eager.
    Matched by node_id (the brief carries page.node_id) rather than by slug, so
    this needs no slugifier here - keeping the scheduler host-light - and is
    robust to slug collisions. The page set is the proposal table (propose-pages
    decides; this consumes it), so synthesise is naturally gated behind proposal-
    gen - no proposals, no synthesise jobs.
    """
    from assimilator.synthesise import (
        brief_relpath,
        build_entity_brief,
        build_slug_map,
        page_set,
    )

    by_reference = {b.get("_reference"): b for b in briefs if b.get("_reference")}
    graph_groups = graph_groups or {}
    existing_members = {
        str(member["node_id"])
        for brief in briefs
        for member in (brief.get("page") or {}).get("nodes") or []
        if isinstance(member, dict) and member.get("node_id")
    }
    slug_map, _collisions = build_slug_map(conn)
    jobs: list[Job] = []
    for specification in page_set(conn):
        members = specification["node_ids"]
        expected = build_entity_brief(
            conn,
            members[0],
            slug_map,
            node_ids=members,
            page=specification["page"],
        )
        if expected is None or not expected.get("claims"):
            continue
        page = expected["page"]
        reference = str(brief_relpath(page["node_type"], page["slug"]).with_suffix(""))
        recorded = by_reference.get(reference)
        reasons: list[str] = []
        reason_codes: list[str] = []
        current_pairs = [
            (str(claim.get("claim_id") or ""), str(claim.get("claim_hash") or ""))
            for claim in expected.get("claims") or []
        ]
        recorded_pairs = [
            (str(claim_id), str(claim_hash))
            for claim_id, claim_hash in (recorded or {}).get("_claim_pairs") or []
        ]
        old_hashes, new_hashes = dict(recorded_pairs), dict(current_pairs)
        old_positions = {pair: position for position, pair in enumerate(recorded_pairs)}
        new_positions = {pair: position for position, pair in enumerate(current_pairs)}
        common = set(old_hashes) & set(new_hashes)
        metrics = {
            "recorded_selection_size": len(recorded_pairs),
            "current_selection_size": len(current_pairs),
            "selected_claims_removed": len(set(old_hashes) - set(new_hashes)),
            "selected_claims_added": len(set(new_hashes) - set(old_hashes)),
            "selected_claims_content_changed": sum(
                old_hashes[claim_id] != new_hashes[claim_id] for claim_id in common
            ),
            "selected_claims_order_changed": sum(
                old_positions[(claim_id, old_hashes[claim_id])]
                != new_positions[(claim_id, new_hashes[claim_id])]
                for claim_id in common
                if old_hashes[claim_id] == new_hashes[claim_id]
            ),
            "page_member_mismatches": 0,
            "page_identity_mismatches": 0,
            "publication_mismatches": 0,
            "listing_tuple_mismatches": 0,
        }
        if recorded is None:
            reasons.append("brief missing at current page reference")
            reason_codes.append("brief_missing")
        else:
            if recorded.get("brief_hash") != expected["brief_hash"]:
                reasons.append("selection hash changed")
                reason_codes.append("selection_hash_mismatch")
            # brief_hash intentionally follows the stable cross-component
            # (claim_id, claim_hash) contract. payload_hash covers bulk claim
            # context and related nodes while remaining in the cheap header.
            if recorded.get("payload_hash") != expected.get("payload_hash"):
                reasons.append("brief payload changed")
                reason_codes.append("payload_hash_mismatch")
            recorded_page = recorded.get("page") or {}
            for field in ("kind", "title", "slug", "node_type", "nodes"):
                if recorded_page.get(field) != page.get(field):
                    reasons.append(f"page {field} changed")
                    reason_codes.append("page_identity_mismatch")
                    metrics[
                        "page_member_mismatches"
                        if field == "nodes"
                        else "page_identity_mismatches"
                    ] += 1
            if recorded_page.get("listing") != page.get("listing"):
                reasons.append("listing tuple changed")
                reason_codes.append("listing_mismatch")
                metrics["listing_tuple_mismatches"] += 1
            if (recorded.get("publication") or {}).get("status") != (
                expected.get("publication") or {}
            ).get("status"):
                reasons.append("publication decision changed")
                reason_codes.append("publication_mismatch")
                metrics["publication_mismatches"] += 1
        contributing_hashes = {
            _bare_hash((claim.get("provenance") or {}).get("content_hash"))
            for claim in expected.get("claims") or []
            if isinstance(claim, dict)
        }
        inherited = _deduplicate_groups(
            *[graph_groups.get(h, []) for h in sorted(contributing_hashes) if h]
        )
        # Inherited uncertainty cannot be repaired by rewriting an otherwise
        # identical deterministic brief. Carry it to article jobs, but do not
        # create an eager job that would run forever without changing anything.
        if not reasons:
            continue
        node_id = members[0]
        existed_for_member = bool(set(members) & existing_members)
        local_groups = (
            [
                _reason_group(
                    "brief-selection",
                    reference,
                    "missing" if "brief_missing" in reason_codes else "stale",
                    reason_codes,
                )
            ]
            if reason_codes
            else []
        )
        jobs.append(
            Job(
                id=f"synthesise:{node_id}",
                type="synthesise",
                lane=LANE_EAGER,
                target=Target(kind="page", label=page["title"], href=reference),
                status=STATUS_ELIGIBLE,
                trigger="stale_brief"
                if recorded or existed_for_member
                else "never_done",
                effort="~local graph slice",
                drivers=[Driver("freshness", reason) for reason in reasons],
                local_reason_groups=local_groups,
                inherited_reason_groups=inherited,
                consequence=_max_consequence(local_groups + inherited, "finish"),
                native_metrics=metrics,
            )
        )
    return jobs


def _article_index(content_dir: Path | None) -> dict[tuple[str, str, str], dict]:
    """Article audits keyed by exact ``(section, slug, language)`` identity."""
    out: dict[tuple[str, str, str], dict] = {}
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
        name = md.name[: -len(".md")]
        pieces = name.rsplit(".", 1)
        slug, language = (pieces[0], pieces[1]) if len(pieces) == 2 else (name, "en")
        section = md.parent.name
        built = fm.get("built_from") or {}
        built_by = fm.get("built_by") or {}
        claims = built.get("claims") if isinstance(built, dict) else []
        out[(section, slug, language)] = {
            "path": str(md),
            "brief_hash": built.get("brief_hash") if isinstance(built, dict) else None,
            "payload_hash": built.get("payload_hash")
            if isinstance(built, dict)
            else None,
            "claims": claims if isinstance(claims, list) else [],
            "built_by": built_by if isinstance(built_by, dict) else {},
            "body_sha256": hashlib.sha256(parts[2].strip().encode("utf-8")).hexdigest(),
        }
    return out


def _current_generator_identity() -> dict:
    """Read stable current Assembler identity when its source is available."""
    assembler_root = Path(__file__).resolve().parents[3] / "assembler"
    if not (assembler_root / "assembler.py").is_file():
        return {}
    if str(assembler_root) not in sys.path:
        sys.path.insert(0, str(assembler_root))
    try:
        import assembler as current
    except Exception:  # optional cross-component diagnostic; absence stays unknown
        return {}
    system = getattr(current, "_SYSTEM_PROMPT", None)
    identity = {"model": getattr(current, "DEFAULT_MODEL", None)}
    if isinstance(system, str):
        identity["system_prompt_sha256"] = hashlib.sha256(system.encode()).hexdigest()
    return {key: value for key, value in identity.items() if value is not None}


def enumerate_assemble_jobs(
    briefs: list[dict],
    content_dir: Path | None,
    graph_groups: dict[str, list[dict]] | None = None,
    brief_local_groups: dict[str, list[dict]] | None = None,
    current_generator: dict | None = None,
) -> list[Job]:
    """A brief whose input hashes are not frozen into its article is pending.

    ``brief_hash`` freezes semantic claim selection. ``payload_hash`` freezes the
    complete writer input. Both must match by page reference; a legacy article
    without payload_hash is stale rather than silently current.

    Those two cases are reported apart. A rebuild trailing the graph is a
    different decision from a page that has never existed - one costs allowance to
    refresh something already readable, the other puts a missing page on the site -
    and calling both "never_done" hid that. On 2026-08-21, 28 of the 29 published
    pages with briefs were stale rather than absent.

    The id's tail is the brief REFERENCE, "<section>/<slug>": the runner hands it
    to the assembler as-is and the assembler resolves it as a path under the
    briefs directory. A slug alone named two pages where an event and a project
    share a name (Apollo 14), so two jobs carried one id."""
    from assimilator.synthesise import section_of

    articles = _article_index(content_dir)
    graph_groups = graph_groups or {}
    brief_local_groups = brief_local_groups or {}
    current_generator = (
        _current_generator_identity()
        if current_generator is None
        else current_generator
    )
    jobs: list[Job] = []
    for brief in briefs:
        brief_hash = brief.get("brief_hash")
        payload_hash = brief.get("payload_hash")
        page = brief.get("page") or {}
        if not brief_hash or not payload_hash:
            continue
        ref = (
            f"{section_of(page.get('node_type'))}/{page['slug']}"
            if page.get("slug")
            else brief_hash[:12]
        )
        section, slug = ref.split("/", 1) if "/" in ref else ("", ref)
        languages = {
            language
            for article_section, article_slug, language in articles
            if article_section == section and article_slug == slug
        } or {"en"}
        brief_pairs = dict(brief.get("_claim_pairs") or [])
        inherited = _deduplicate_groups(
            brief_local_groups.get(ref, []),
            *[
                graph_groups.get(h, [])
                for h in sorted(brief.get("_content_hashes") or [])
            ],
        )
        for language in sorted(languages):
            article = articles.get((section, slug, language))
            reasons: list[str] = []
            status = "stale"
            consequence = "finish"
            metrics = {
                "article_citation_count": 0,
                "citations_missing": 0,
                "citation_hash_mismatches": 0,
                "generator_fields_compared": 0,
                "generator_fields_changed": 0,
                "body_hash_mismatch": 0,
            }
            if article is None:
                reasons.append("article_missing")
                status = "missing"
            else:
                if article.get("brief_hash") != brief_hash:
                    reasons.append("brief_hash_mismatch")
                if article.get("payload_hash") != payload_hash:
                    reasons.append("payload_hash_mismatch")
                citations = article.get("claims") or []
                metrics["article_citation_count"] = len(citations)
                for citation in citations:
                    if not isinstance(citation, dict):
                        metrics["citations_missing"] += 1
                        continue
                    claim_id, claim_hash = citation.get("id"), citation.get("hash")
                    if claim_id not in brief_pairs:
                        metrics["citations_missing"] += 1
                    elif brief_pairs[claim_id] != claim_hash:
                        metrics["citation_hash_mismatches"] += 1
                if metrics["citations_missing"]:
                    reasons.append("citation_missing")
                    consequence = "repair"
                if metrics["citation_hash_mismatches"]:
                    reasons.append("citation_hash_mismatch")
                    consequence = "repair"
                built_by = article.get("built_by") or {}
                comparable_generator = {
                    key: value
                    for key, value in current_generator.items()
                    if key in built_by
                }
                metrics["generator_fields_compared"] = len(comparable_generator)
                changed_generator = sum(
                    built_by.get(key) != value
                    for key, value in comparable_generator.items()
                )
                metrics["generator_fields_changed"] = changed_generator
                if changed_generator:
                    reasons.append("generator_changed")
                    consequence = "verify" if len(reasons) == 1 else consequence
                protected_hash = built_by.get("body_sha256")
                if protected_hash and protected_hash != article.get("body_sha256"):
                    metrics["body_hash_mismatch"] = 1
                    reasons.append("body_modified")
                    consequence = "verify" if len(reasons) == 1 else consequence
            local_groups = (
                [
                    _reason_group(
                        "article-input",
                        f"{ref}.{language}",
                        status,
                        reasons,
                        consequence,
                    )
                ]
                if reasons
                else []
            )
            if not local_groups and not inherited:
                continue
            blocker = None
            job_status = STATUS_ELIGIBLE
            if inherited:
                blocker = "upstream_freshness"
                job_status = STATUS_BLOCKED
            elif "body_modified" in reasons:
                blocker = "protected_body"
                job_status = STATUS_BLOCKED
            jobs.append(
                Job(
                    id=f"assemble:{ref}"
                    if language == "en"
                    else f"assemble:{ref}:{language}",
                    type="assemble",
                    lane=LANE_CLAUDE,
                    target=Target(kind="page", label=page.get("title") or "page"),
                    status=job_status,
                    trigger=(
                        "never_done"
                        if article is None
                        else "stale_brief"
                        if local_groups
                        else "inherited_freshness"
                    ),
                    drivers=[Driver("claims", str(page.get("claim_count", "?")))],
                    blocker=blocker,
                    article=f"{ref}.{language}",
                    local_reason_groups=local_groups,
                    inherited_reason_groups=inherited,
                    consequence=_max_consequence(local_groups + inherited, consequence),
                    native_metrics=metrics,
                )
            )
    return jobs


def _proposal_table_stale(conn: sqlite3.Connection) -> tuple[bool, int]:
    """Does the derived page_proposals table reflect the current gate? Returns
    (stale, gate_count). Stale when the gate-passing node set differs from what
    is recorded - a recompute is then a pending propose-pages job.

    THE TWO SETS MUST BE SUBTRACTED THE SAME WAY. propose() writes the gate
    minus vetoes AND minus the nodes a composed page covers; this compared the
    gate minus vetoes alone, so from the moment one page covered two nodes the
    sets could never match and propose-pages was pending for ever - it ran 12
    times, once per scheduler restart, and showed on Mark's card as permanently
    waiting. A staleness test that names the exclusions itself will drift from
    the writer again, so it calls the same helpers propose() does.
    """
    from assimilator.page_gate import page_gate_rows
    from assimilator.pages import member_node_ids
    from assimilator.propose_pages import vetoed_node_ids

    excluded = vetoed_node_ids(conn) | set(member_node_ids(conn))
    gate_ids = {r["node_id"] for r in page_gate_rows(conn)} - excluded
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
                command=["propose-pages"],
                drivers=[Driver("nodes passing gate", str(gate_count))],
                consequence="finish",
            )
        )
    embedded = _live_embedded_claims(conn, _embedding_model_id())
    recorded = conn.execute("SELECT COUNT(*) FROM corroborations").fetchone()[0]
    # Total is claims + live nodes: both are embedded, and a progress figure that
    # counts only claims under-reports by the node count and looks stalled at the
    # end of a run.
    total_items = (
        total_claims
        + conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE retired_at IS NULL"
        ).fetchone()[0]
    )
    pending = pending_by_bucket(conn, _embedding_model_id())
    if pending:
        # ONE JOB PER FIXED BUCKET, not one job for the backlog. Vector embedding
        # is ~3 items a second, so the whole corpus is a ~3-hour task, and as a
        # single job it holds the background lane for that entire time - a
        # document ingested during it waits behind work with no reason to be
        # atomic. The buckets are a hash of each row's own id, so job ids never
        # renumber as rows are embedded; a finished bucket simply stops being
        # emitted. See embed_batches.
        outstanding = sum(pending.values())
        for bucket in sorted(pending):
            remaining = pending[bucket]
            jobs.append(
                Job(
                    id=f"embed:claims:{bucket}",
                    type="embed",
                    lane=LANE_EAGER,
                    target=Target(
                        kind="page",
                        label=f"vector embedding, batch {bucket + 1} of {BUCKETS}",
                    ),
                    status=STATUS_ELIGIBLE,
                    trigger="never_done",
                    effort=f"~{_embed_minutes(remaining)} min local CPU",
                    command=["embed", "--bucket", str(bucket)],
                    consequence="finish",
                    drivers=[
                        Driver("items in this batch", str(remaining)),
                        Driver(
                            "corpus progress",
                            f"{total_items - outstanding} of {total_items} embedded",
                        ),
                    ],
                )
            )
        jobs.append(
            Job(
                id="corroborate:graph",
                type="corroborate",
                lane=LANE_CLAUDE,
                target=Target(kind="page", label="cross-record claim pairs"),
                status=STATUS_BLOCKED,
                # Blocked on the LOWEST outstanding batch rather than on a
                # singleton that no longer exists. Corroborate needs the whole
                # corpus embedded, so any outstanding batch blocks it.
                blocker=f"embed:claims:{min(pending)}",
                trigger="never_done",
                consequence="verify",
                drivers=[
                    Driver(
                        "vector embedding",
                        f"{outstanding} items outstanding",
                        band="off",
                    )
                ],
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
                command=["corroborate"],
                consequence="verify",
                drivers=[
                    Driver("claims embedded", str(embedded)),
                    Driver("pairs confirmed", str(recorded), band="off"),
                ],
            )
        )
    return jobs


def _embedding_model_id() -> str:
    """The vector space this graph stores, read WITHOUT importing the embedder.

    assimilator.embeddings pulls in fastembed, which is a container-only
    dependency; this module is deliberately host-light so the queue can be
    enumerated anywhere. The id is a plain string constant, so it is re-derived
    from the same parts rather than imported - and the parts are asserted equal
    in tests, so the two cannot drift.
    """
    return (
        "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8:dynamic_uint8.onnx:1024:"
        "dequant-v1"
    )


def _embed_minutes(items: int) -> int:
    """Rough wall-clock for a batch. 4.9 items/second.

    Measured end to end over batch 0 on 2026-08-22: 1,221 items (983 claims,
    238 nodes) in 248s from launch to exit. The earlier 3.06 figure came from a
    run that included the in-process fastembed fallback; through the endpoint it
    is meaningfully faster, and ~16s of that 248 was model warm-up before the
    first commit, so a longer batch beats this rate rather than missing it.
    """
    return max(1, round(items / 4.9 / 60))


def _live_embedded_claims(conn: sqlite3.Connection, model_id: str) -> int:
    """Claims that have a vector in the current space AND still exist.

    The join is the point. Claim ids do not survive a re-digest - the rows are
    deleted and recreated - so embedding_model accumulates vectors for claims
    that are gone: 4,640 of 5,218 on the live graph. A bare row count therefore
    read 18% coverage where the real figure was 6%, and that number reached
    operator-facing copy before anyone joined it back to the corpus.

    Reads embedding_model rather than vec_claims because vec0 is an extension
    this host-light module does not load - counting vec_claims here silently
    returned 0 whenever the queue was enumerated outside the container."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM embedding_model e JOIN claims c ON c.id = e.id "
            "WHERE e.kind = 'claim' AND e.model_id = ?",
            (model_id,),
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0  # the table is not present until the first embed run


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
    digest_groups, digest_metrics = digest_freshness(
        ingests_dir, digests_dir, digest_index, store
    )
    record_groups, record_metrics = record_generation_freshness(ingests_dir, store)
    digest_upstream = {
        content_hash: _deduplicate_groups(
            digest_groups.get(content_hash, []), record_groups.get(content_hash, [])
        )
        for content_hash in set(digest_groups) | set(record_groups)
    }
    graph_groups = graph_freshness_groups(conn, digest_index, digest_upstream)

    jobs: list[Job] = []
    jobs += enumerate_ingest_jobs(
        sources_dir, _ingested_source_ids(ingests_dir), _superseded_hashes(sources_dir)
    )
    jobs += enumerate_digest_jobs(
        ingests_dir, digest_index, store, demand, digest_groups, record_groups
    )
    rebuild_jobs = enumerate_rebuild_jobs(conn, digest_index, graph_groups)
    jobs += rebuild_jobs
    # A clean rebuild subsumes generation-driven imports. Offering both implies
    # the per-record path can converge even though it cannot delete orphans.
    if not rebuild_jobs:
        jobs += enumerate_import_jobs(conn, digest_index, digest_upstream)
    briefs = _load_briefs(briefs_dir)
    synthesise_jobs = enumerate_synthesise_jobs(conn, briefs, graph_groups)
    jobs += synthesise_jobs
    brief_local_groups = {
        job.target.href: job.local_reason_groups
        for job in synthesise_jobs
        if job.target.href and job.local_reason_groups
    }
    jobs += enumerate_assemble_jobs(
        briefs,
        content_dir,
        graph_groups,
        brief_local_groups,
    )
    jobs += enumerate_graph_jobs(conn)
    review_queue = enumerate_review_queue(ingests_dir, store, demand)

    return {
        "schema": "anomalica/schedule/0",
        "generatedAt": generated_at,
        "jobs": [j.to_dict() for j in jobs],
        "reviewQueue": [it.to_dict() for it in review_queue],
        "recordDemand": demand,
        "graphImportDeltas": import_deltas(conn, digest_index),
        "graphImportNativeDeltas": graph_import_native_deltas(conn, digest_index),
        "digestFreshnessMetrics": digest_metrics,
        "recordGenerationMetrics": record_metrics,
        "graphInput": graph_input_diagnostics(conn, digest_index, digests_dir),
    }


def default_queue_path() -> Path:
    return Path(
        os.environ.get(
            "SCHEDULER_QUEUE_PATH",
            str(data_dir() / "scheduler-queue.json"),
        )
    )


def write_queue(queue: dict, path: Path) -> None:
    _atomic_write(path, _json_bytes(queue))


def _json_bytes(document: dict) -> bytes:
    return json.dumps(document, indent=2, ensure_ascii=False).encode()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def default_freshness_path(queue_path: Path) -> Path:
    return queue_path.with_name(f"{queue_path.stem}-freshness.json")


def freshness_manifest(queue: dict, source_queue_sha256: str) -> dict:
    """Flatten the real queue's canonical groups for guarded deployment input."""
    groups: list[dict] = []

    def add(group: dict) -> None:
        groups.append(group)
        for inherited in group.get("inherited", []):
            add(inherited)

    for job in queue.get("jobs", []):
        for key in ("local_reason_groups", "inherited_reason_groups"):
            for group in job.get(key, []):
                add(group)

    canonical = []
    for group in _deduplicate_groups(groups):
        canonical.append(
            {
                "boundary": group["boundary"],
                "artifact": group["artifact"],
                "local_status": group["local_status"],
                "local_reasons": group["local_reasons"],
                "inherited": [],
                "consequence": group["consequence"],
            }
        )
    return {
        "schema": "anomalica-freshness/v1",
        "generated_at": queue["generatedAt"],
        "source_queue_sha256": source_queue_sha256,
        "groups": canonical,
    }


def write_freshness_manifest(queue: dict, queue_path: Path, path: Path) -> str:
    queue_content = queue_path.read_bytes()
    if queue_content != _json_bytes(queue):
        raise ValueError("source queue bytes do not match the schedule result")
    queue_sha256 = hashlib.sha256(queue_content).hexdigest()
    manifest = freshness_manifest(queue, queue_sha256)
    content = _json_bytes(manifest)
    _atomic_write(path, content)
    return hashlib.sha256(content).hexdigest()


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
    freshness_out: str | None = None,
) -> tuple[dict, Path]:
    """Build the queue from current corpus state and write it. Read-only on the
    graph DB - enumeration never mutates the live database."""
    ingests_dir, digests_dir, sources_dir = resolve_corpus_dirs(
        ingests, digests, sources
    )
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    content_dir = Path(os.environ.get("ANOMALICA_CONTENT_DIR", str(root / "content")))
    out_path = Path(out) if out else default_queue_path()
    freshness_path = (
        Path(freshness_out) if freshness_out else default_freshness_path(out_path)
    )
    if freshness_path == out_path:
        raise ValueError("queue and freshness output paths must differ")
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
    write_freshness_manifest(queue, out_path, freshness_path)
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
        str(data_dir() / "knowledge.db"),
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
    parser.add_argument(
        "--freshness-out",
        default=None,
        help="freshness manifest path (default: adjacent to queue)",
    )
    args = parser.parse_args(argv)

    queue, out_path = run_schedule(
        args.db,
        args.ingests,
        args.digests,
        args.sources,
        args.out,
        args.freshness_out,
    )
    by_lane: dict[str, int] = {}
    for job in queue["jobs"]:
        by_lane[job["lane"]] = by_lane.get(job["lane"], 0) + 1
    lanes = ", ".join(f"{n} {lane}" for lane, n in sorted(by_lane.items()))
    print(f"Wrote {out_path}")
    freshness_path = (
        Path(args.freshness_out)
        if args.freshness_out
        else default_freshness_path(out_path)
    )
    freshness_sha256 = hashlib.sha256(freshness_path.read_bytes()).hexdigest()
    print(f"Wrote {freshness_path} (sha256 {freshness_sha256})")
    print(
        f"  {len(queue['jobs'])} jobs ({lanes}), {len(queue['reviewQueue'])} awaiting review"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
