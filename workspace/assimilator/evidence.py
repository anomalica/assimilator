"""ADR 0051 structural import, exact anchors and conservative evidence identity.

The digest remains the source of truth.  This module validates the complete
``digest/2`` input before the importer mutates the graph, materialises its
normalised structural rows, and derives rebuild-stable evidence units from all
anchors visible in the domain and infrastructure databases together.

Semantic agreement (``corroborations``), evidence identity (the tables managed
here), and provenance independence are deliberately separate relations.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from anomalica_common.digest.models import (
    Digest2Bindings,
    SourceAnchor,
    SourceAnchors,
)
from anomalica_common.identity import (
    digest_record_snapshot_identity,
    evidence_unit_identity,
)
from anomalica_common.pre_digest import (
    PreDigestSourceMap,
    SourceMapError,
    source_map_json,
    validate_source_map,
)
from anomalica_common.records import DigestRecordSnapshot

DIGEST_1 = "anomalica/digest/1"
DIGEST_2 = "anomalica/digest/2"


@dataclass(frozen=True)
class ValidatedDigest:
    schema: str
    snapshot: DigestRecordSnapshot | None
    record_snapshot_sha256: str | None
    source_map_sha256: str | None
    claim_anchors: Mapping[str, tuple[SourceAnchor, ...]]


@dataclass(frozen=True)
class EvidenceAssessment:
    independent_sources: int
    scored_claims: int
    unscored_claims: int
    established_claims: frozenset[str]
    overlap_unknown_claims: frozenset[str]


def _labelled_sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _all_claims(parsed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        *(parsed.get("domain_claims") or []),
        *(parsed.get("infrastructure_claims") or []),
    ]


def _snapshot(record: Mapping[str, Any]) -> DigestRecordSnapshot:
    projection = {
        "schema": "anomalica/digest-record-snapshot/1",
        **{
            key: record[key]
            for key in (
                "content_hash",
                "title",
                "provenance",
                "work_provenance",
                "assets",
                "asset_rights",
                "selection",
                "page_map",
            )
            if key in record
        },
    }
    return DigestRecordSnapshot.model_validate(projection)


def _candidate_ingests_roots(
    source_path: str | None, source_root: str | None
) -> list[Path]:
    candidates = [
        Path(os.environ.get("ANOMALICA_INGESTS_DIR", "/home/nonroot/ingests"))
    ]
    if source_root:
        root = Path(source_root).resolve()
        candidates.append(root.parent / "ingests")
    if source_path:
        path = Path(source_path).resolve()
        digest_root = next(
            (
                parent
                for parent in (path.parent, *path.parents)
                if parent.name == "digests"
            ),
            None,
        )
        if digest_root is not None:
            candidates.append(digest_root.parent / "ingests")
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _load_bound_pre_digest(
    bindings: Digest2Bindings,
    source_path: str | None,
    source_root: str | None,
) -> tuple[str, PreDigestSourceMap]:
    body_name = bindings.pre_digest.sha256.removeprefix("sha256:") + ".md"
    map_name = bindings.pre_digest.source_map_sha256.removeprefix("sha256:") + ".json"
    checked: list[str] = []
    for root in _candidate_ingests_roots(source_path, source_root):
        body_path = root / "pre-digests" / body_name
        map_path = root / "source-maps" / map_name
        checked.extend((str(body_path), str(map_path)))
        if not body_path.is_file() or not map_path.is_file():
            continue
        raw_body = body_path.read_bytes()
        try:
            body = raw_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceMapError("bound pre-digest is not valid UTF-8") from exc
        raw_map = map_path.read_bytes()
        try:
            document = json.loads(raw_map)
        except json.JSONDecodeError as exc:
            raise SourceMapError("bound source map is not valid JSON") from exc
        source_map = PreDigestSourceMap.model_validate(document)
        canonical = source_map_json(source_map)
        if raw_map != canonical:
            raise SourceMapError("bound source map is not canonical compact JSON")
        if _labelled_sha256(raw_map) != bindings.pre_digest.source_map_sha256:
            raise SourceMapError("source-map file does not match source_map_sha256")
        if _labelled_sha256(raw_body) != bindings.pre_digest.sha256:
            raise SourceMapError("pre-digest file does not match pre_digest.sha256")
        validate_source_map(
            source_map,
            body,
            expected_sha256=bindings.pre_digest.source_map_sha256,
        )
        return body, source_map
    raise SourceMapError(
        "digest/2 bound pre-digest or source map is unavailable; checked "
        + ", ".join(checked)
    )


def _validate_map_against_snapshot(
    source_map: PreDigestSourceMap, snapshot: DigestRecordSnapshot
) -> None:
    if source_map.record_hash != snapshot.content_hash:
        raise SourceMapError("source map and digest Record bind different Records")
    pages = {
        page.record_page: (page.asset_hash, page.asset_file_page)
        for page in snapshot.page_map or []
    }
    if not pages:
        raise SourceMapError("digest/2 requires a complete page map")
    frame_by_record_page: dict[int, str] = {}
    for entry in source_map.entries:
        if pages.get(entry.record_page) != (
            entry.asset_hash,
            entry.asset_file_page,
        ):
            raise SourceMapError("source-map entry disagrees with digest page_map")
        previous = frame_by_record_page.setdefault(
            entry.record_page, entry.asset_text_sha256
        )
        if previous != entry.asset_text_sha256:
            raise SourceMapError("one Record page uses several Asset text frames")


def _validate_anchor(
    anchor: SourceAnchor,
    body: str,
    source_map: PreDigestSourceMap,
) -> None:
    if anchor.body_span.end > len(body):
        raise SourceMapError("source anchor lies outside the materialised pre-digest")
    if body[anchor.body_span.start : anchor.body_span.end] != anchor.quote:
        raise SourceMapError("source anchor quote does not match its body span")
    for entry in source_map.entries:
        if not (
            entry.body_span.start <= anchor.body_span.start
            and anchor.body_span.end <= entry.body_span.end
        ):
            continue
        offset = anchor.body_span.start - entry.body_span.start
        expected_start = entry.asset_span.start + offset
        expected_end = expected_start + len(anchor.body_span)
        if (
            anchor.asset_hash,
            anchor.record_page,
            anchor.asset_file_page,
            anchor.asset_text_sha256,
            anchor.asset_span.start,
            anchor.asset_span.end,
        ) != (
            entry.asset_hash,
            entry.record_page,
            entry.asset_file_page,
            entry.asset_text_sha256,
            expected_start,
            expected_end,
        ):
            raise SourceMapError("source anchor disagrees with its source-map entry")
        return
    raise SourceMapError("source anchor is unmappable in the bound source map")


def validate_digest_for_import(
    parsed: Mapping[str, Any],
    *,
    source_path: str | None = None,
    source_root: str | None = None,
) -> ValidatedDigest:
    """Validate schema-specific graph input before any graph row is changed."""
    fm = parsed.get("frontmatter") or {}
    schema = fm.get("schema") or DIGEST_1
    if schema not in {DIGEST_1, DIGEST_2}:
        raise ValueError(f"unsupported digest schema: {schema!r}")
    record = fm.get("record") or {}
    claims = _all_claims(parsed)

    ids = [claim.get("id") for claim in claims]
    if any(not isinstance(claim_id, str) or not claim_id for claim_id in ids):
        raise ValueError("every digest claim requires a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("digest claim ids must be unique across both categories")

    if schema == DIGEST_1:
        if any("source_anchors" in claim for claim in claims):
            raise ValueError("digest/1 must not carry typed source anchors")
        structural = any(
            key in record for key in ("assets", "asset_rights", "selection", "page_map")
        )
        snapshot = _snapshot(record) if structural else None
        snapshot_hash = fm.get("record_snapshot_sha256")
        if snapshot is not None:
            if snapshot.page_map is not None:
                raise ValueError("page-mapped record/3 input requires digest/2")
            if snapshot_hash != digest_record_snapshot_identity(snapshot):
                raise ValueError(
                    "record_snapshot_sha256 does not match digest Record projection"
                )
        return ValidatedDigest(
            schema=schema,
            snapshot=snapshot,
            record_snapshot_sha256=snapshot_hash if snapshot is not None else None,
            source_map_sha256=None,
            claim_anchors={},
        )

    bindings = Digest2Bindings.from_document(fm)
    snapshot = _snapshot(record)
    if bindings.record_snapshot_sha256 != digest_record_snapshot_identity(snapshot):
        raise ValueError(
            "record_snapshot_sha256 does not match digest Record projection"
        )
    if snapshot.page_map is None or any(
        asset.source_type not in {"pdf", "image"} for asset in snapshot.assets
    ):
        raise ValueError("digest/2 is limited to page-mapped PDF/image Records")
    body, source_map = _load_bound_pre_digest(bindings, source_path, source_root)
    _validate_map_against_snapshot(source_map, snapshot)

    anchors_by_claim: dict[str, tuple[SourceAnchor, ...]] = {}
    for claim in claims:
        if claim.get("location_in_record") is not None:
            raise ValueError("digest/2 must not carry a legacy scalar location")
        anchors = SourceAnchors.model_validate(claim.get("source_anchors"))
        for anchor in anchors.root:
            _validate_anchor(anchor, body, source_map)
        anchors_by_claim[str(claim["id"])] = tuple(anchors.root)

    return ValidatedDigest(
        schema=schema,
        snapshot=snapshot,
        record_snapshot_sha256=bindings.record_snapshot_sha256,
        source_map_sha256=bindings.pre_digest.source_map_sha256,
        claim_anchors=anchors_by_claim,
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _put_root(
    conn: sqlite3.Connection,
    root_id: str,
    *,
    status: str,
    kind: str,
    metadata: Mapping[str, Any] | None,
    evidence: Any,
) -> None:
    existing = conn.execute(
        "SELECT status, kind FROM provenance_roots WHERE id = ?", (root_id,)
    ).fetchone()
    if existing is not None and tuple(existing) != (status, kind):
        raise ValueError(f"provenance root {root_id!r} has conflicting status or kind")
    conn.execute(
        "INSERT INTO provenance_roots (id, status, kind, metadata, evidence) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
        "status=excluded.status, kind=excluded.kind, metadata=excluded.metadata, "
        "evidence=excluded.evidence",
        (
            root_id,
            status,
            kind,
            _json(metadata) if metadata is not None else None,
            _json(evidence) if evidence is not None else None,
        ),
    )


def _refresh_work_root(conn: sqlite3.Connection, root_id: str | None) -> None:
    """Rebuild one work root's evidence from every Record that declares it.

    A root can be shared by several Records. Last-writer-wins evidence would make
    a routine re-import change the graph according to import order, so the root
    stores the sorted union of the current Record snapshots instead.
    """
    if not root_id:
        return
    evidence: set[str] = set()
    for (raw_metadata,) in conn.execute(
        "SELECT metadata FROM records WHERE work_id = ?", (root_id,)
    ):
        if not raw_metadata:
            continue
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            continue
        work = metadata.get("work_provenance") if isinstance(metadata, dict) else None
        if not isinstance(work, dict) or work.get("root_id") != root_id:
            continue
        values = work.get("evidence")
        if isinstance(values, list):
            evidence.update(
                value for value in values if isinstance(value, str) and value
            )
    if evidence:
        conn.execute(
            "UPDATE provenance_roots SET evidence = ? WHERE id = ? AND kind = 'work'",
            (_json(sorted(evidence)), root_id),
        )


def store_record_structure(
    conn: sqlite3.Connection,
    record_id: str,
    validated: ValidatedDigest,
) -> None:
    """Replace one Record's normalised structure from its validated digest."""
    old_work_row = conn.execute(
        "SELECT work_id FROM records WHERE id = ?", (record_id,)
    ).fetchone()
    old_work_id = str(old_work_row[0]) if old_work_row and old_work_row[0] else None
    snapshot = validated.snapshot
    if snapshot is None:
        conn.execute("DELETE FROM record_page_maps WHERE record_id = ?", (record_id,))
        conn.execute("DELETE FROM record_selections WHERE record_id = ?", (record_id,))
        conn.execute("UPDATE records SET work_id = NULL WHERE id = ?", (record_id,))
        _refresh_work_root(conn, old_work_id)
        return

    for asset in snapshot.assets:
        metadata = _json(asset.model_dump(mode="json", exclude_none=True))
        existing = conn.execute(
            "SELECT metadata FROM assets WHERE asset_hash = ?", (asset.asset_hash,)
        ).fetchone()
        if existing is not None and existing[0] != metadata:
            raise ValueError(
                f"immutable Asset projection changed for {asset.asset_hash}"
            )
        conn.execute(
            "INSERT OR IGNORE INTO assets (asset_hash, metadata) VALUES (?, ?)",
            (asset.asset_hash, metadata),
        )

    conn.execute("DELETE FROM record_selections WHERE record_id = ?", (record_id,))
    for ordinal, entry in enumerate(snapshot.selection.root, 1):
        selector = entry.selector
        conn.execute(
            "INSERT INTO record_selections "
            "(record_id, ordinal, asset_hash, selector_type, asset_file_page) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record_id,
                ordinal,
                entry.asset_hash,
                selector.type,
                getattr(selector, "page", None),
            ),
        )

    conn.execute("DELETE FROM record_page_maps WHERE record_id = ?", (record_id,))
    for page in snapshot.page_map or []:
        conn.execute(
            "INSERT INTO record_page_maps "
            "(record_id, record_page, asset_hash, asset_file_page) "
            "VALUES (?, ?, ?, ?)",
            (
                record_id,
                page.record_page,
                page.asset_hash,
                page.asset_file_page,
            ),
        )

    work = snapshot.work_provenance
    if work is None:
        conn.execute("UPDATE records SET work_id = NULL WHERE id = ?", (record_id,))
        _refresh_work_root(conn, old_work_id)
    else:
        root_id = work.root_id
        _put_root(
            conn,
            root_id,
            status="established",
            kind="work",
            metadata={"declared_root_id": work.root_id},
            evidence=work.evidence,
        )
        conn.execute(
            "UPDATE records SET work_id = ? WHERE id = ?", (root_id, record_id)
        )
        _refresh_work_root(conn, root_id)
        if old_work_id != root_id:
            _refresh_work_root(conn, old_work_id)


def replace_claim_anchors(
    conn: sqlite3.Connection,
    claim_id: str,
    anchors: Sequence[SourceAnchor],
) -> None:
    conn.execute("DELETE FROM claim_evidence_units WHERE claim_id = ?", (claim_id,))
    conn.execute("DELETE FROM claim_anchors WHERE claim_id = ?", (claim_id,))
    for ordinal, anchor in enumerate(anchors, 1):
        conn.execute(
            "INSERT INTO claim_anchors "
            "(claim_id, ordinal, asset_hash, record_page, asset_file_page, "
            "asset_text_sha256, asset_start, asset_end, body_start, body_end, quote) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim_id,
                ordinal,
                anchor.asset_hash,
                anchor.record_page,
                anchor.asset_file_page,
                anchor.asset_text_sha256,
                anchor.asset_span.start,
                anchor.asset_span.end,
                anchor.body_span.start,
                anchor.body_span.end,
                anchor.quote,
            ),
        )


def _resolve_origin_node(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT n.id FROM nodes n WHERE n.name = ? AND n.retired_at IS NULL "
        "ORDER BY (SELECT COUNT(*) FROM claim_node_refs r WHERE r.node_id = n.id) "
        "DESC, n.id LIMIT 1",
        (name,),
    ).fetchone()
    if row is not None:
        return str(row[0])
    row = conn.execute(
        "SELECT n.id FROM aliases a JOIN nodes n ON n.id = a.node_id "
        "WHERE a.alias = ? AND n.retired_at IS NULL ORDER BY n.id LIMIT 1",
        (name,),
    ).fetchone()
    return str(row[0]) if row is not None else None


def replace_claim_provenance_root(conn: sqlite3.Connection, claim_id: str) -> None:
    """Materialise one explicit established or unknown claim provenance root."""
    conn.execute("DELETE FROM claim_provenance_roots WHERE claim_id = ?", (claim_id,))
    row = conn.execute(
        "SELECT c.speaker_id, c.origin_kind, c.origin, c.origin_ref, c.relay, "
        "r.work_id FROM claims c JOIN records r ON r.id = c.record_id "
        "WHERE c.id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return
    speaker_id, origin_kind, origin, origin_ref, relay, work_id = row
    root_id: str | None = None
    basis = "unknown"
    kind = "assertion_origin"

    if origin_kind == "speaker" and speaker_id:
        root_id = f"assertion:node:{speaker_id}"
        basis = "speaker_node"
        _put_root(
            conn,
            root_id,
            status="established",
            kind=kind,
            metadata={"node_id": speaker_id},
            evidence={"basis": "resolved_node"},
        )
    elif origin_kind in {"named", "document"} and str(origin or "").strip():
        node_id = _resolve_origin_node(conn, str(origin).strip())
        if node_id:
            root_id = f"assertion:node:{node_id}"
            basis = f"{origin_kind}_node"
            _put_root(
                conn,
                root_id,
                status="established",
                kind=kind,
                metadata={"node_id": node_id},
                evidence={"basis": "resolved_node"},
            )
    elif origin_kind == "unattributed" and work_id:
        root_id = str(work_id)
        basis = "record_work"

    if root_id is None:
        root_id = f"unknown:claim:{claim_id}"
        reason = (
            "anonymous_origin"
            if origin_kind == "anonymous"
            else "unresolved_assertion_origin"
            if origin_kind
            else "provenance_chain_absent"
        )
        _put_root(
            conn,
            root_id,
            status="unknown",
            kind=kind,
            metadata={"reason": reason},
            evidence={
                "origin_kind": origin_kind,
                "origin": origin or "",
                "origin_ref": origin_ref or "",
                "relay": json.loads(relay) if relay else [],
            },
        )

    conn.execute(
        "INSERT INTO claim_provenance_roots "
        "(claim_id, provenance_root_id, basis) VALUES (?, ?, ?)",
        (claim_id, root_id, basis),
    )


def prune_unreferenced_provenance_roots(conn: sqlite3.Connection) -> None:
    """Remove roots no current claim, Record or replayed lineage still uses."""
    conn.execute(
        "DELETE FROM provenance_roots WHERE id NOT IN "
        "(SELECT provenance_root_id FROM claim_provenance_roots) "
        "AND id NOT IN (SELECT work_id FROM records WHERE work_id IS NOT NULL) "
        "AND id NOT IN (SELECT root_a FROM provenance_lineage) "
        "AND id NOT IN (SELECT root_b FROM provenance_lineage)"
    )


def _unique_connections(
    connections: Iterable[sqlite3.Connection],
) -> list[sqlite3.Connection]:
    unique: list[sqlite3.Connection] = []
    seen: set[int] = set()
    for conn in connections:
        if id(conn) not in seen:
            seen.add(id(conn))
            unique.append(conn)
    return unique


def rebuild_evidence_units(connections: Iterable[sqlite3.Connection]) -> None:
    """Recompute global overlap components, then materialise local memberships."""
    conns = _unique_connections(connections)
    intervals: dict[tuple[str, int, str], list[tuple[int, int, str, int, int]]] = {}
    for database_index, conn in enumerate(conns):
        for (
            claim_id,
            ordinal,
            asset_hash,
            asset_page,
            text_hash,
            start,
            end,
        ) in conn.execute(
            "SELECT claim_id, ordinal, asset_hash, asset_file_page, "
            "asset_text_sha256, asset_start, asset_end FROM claim_anchors"
        ):
            intervals.setdefault((asset_hash, asset_page, text_hash), []).append(
                (start, end, claim_id, ordinal, database_index)
            )

    for conn in conns:
        conn.execute("DELETE FROM claim_evidence_units")
        conn.execute("DELETE FROM evidence_units")

    for frame in sorted(intervals):
        ordered = sorted(
            intervals[frame], key=lambda row: (row[0], row[1], row[2], row[3])
        )
        components: list[list[tuple[int, int, str, int, int]]] = []
        component_end = -1
        for interval in ordered:
            if not components or interval[0] >= component_end:
                components.append([interval])
                component_end = interval[1]
            else:
                components[-1].append(interval)
                component_end = max(component_end, interval[1])

        asset_hash, asset_page, text_hash = frame
        for component in components:
            start = min(interval[0] for interval in component)
            end = max(interval[1] for interval in component)
            unit_id = evidence_unit_identity(
                {
                    "asset_hash": asset_hash,
                    "asset_file_page": asset_page,
                    "asset_text_sha256": text_hash,
                    "span": {"start": start, "end": end},
                }
            )
            claims_by_database: dict[int, set[str]] = {}
            for _start, _end, claim_id, _ordinal, database_index in component:
                claims_by_database.setdefault(database_index, set()).add(claim_id)
            for database_index, claim_ids in claims_by_database.items():
                conn = conns[database_index]
                conn.execute(
                    "INSERT INTO evidence_units "
                    "(id, asset_hash, asset_file_page, asset_text_sha256, "
                    "span_start, span_end) VALUES (?, ?, ?, ?, ?, ?)",
                    (unit_id, asset_hash, asset_page, text_hash, start, end),
                )
                conn.executemany(
                    "INSERT INTO claim_evidence_units "
                    "(claim_id, evidence_unit_id) VALUES (?, ?)",
                    [(claim_id, unit_id) for claim_id in sorted(claim_ids)],
                )


def source_anchors_for_claim(
    conn: sqlite3.Connection, claim_id: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT asset_hash, record_page, asset_file_page, asset_text_sha256, "
        "asset_start, asset_end, body_start, body_end, quote "
        "FROM claim_anchors WHERE claim_id = ? ORDER BY ordinal",
        (claim_id,),
    ).fetchall()
    return [
        {
            "asset_hash": row[0],
            "record_page": row[1],
            "asset_file_page": row[2],
            "asset_text_sha256": row[3],
            "asset_span": {"start": row[4], "end": row[5]},
            "body_span": {"start": row[6], "end": row[7]},
            "quote": row[8],
        }
        for row in rows
    ]


def evidence_unit_ids_for_claim(conn: sqlite3.Connection, claim_id: str) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT evidence_unit_id FROM claim_evidence_units "
            "WHERE claim_id = ? ORDER BY evidence_unit_id",
            (claim_id,),
        )
    ]


def established_root_ids_for_claim(
    conn: sqlite3.Connection, claim_id: str
) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT cpr.provenance_root_id FROM claim_provenance_roots cpr "
            "JOIN provenance_roots pr ON pr.id = cpr.provenance_root_id "
            "WHERE cpr.claim_id = ? AND pr.status = 'established' "
            "ORDER BY cpr.provenance_root_id",
            (claim_id,),
        )
    ]


class _DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def assess_evidence_support(
    conn: sqlite3.Connection, claim_ids: Sequence[str]
) -> EvidenceAssessment:
    """Count support with both evidence and provenance uniqueness constraints."""
    ids = list(dict.fromkeys(claim_ids))
    if not ids:
        return EvidenceAssessment(0, 0, 0, frozenset(), frozenset())
    placeholders = ",".join("?" for _ in ids)
    present = {
        str(row[0])
        for row in conn.execute(
            f"SELECT id FROM claims WHERE id IN ({placeholders})",
            ids,  # noqa: S608
        )
    }
    units: dict[str, set[str]] = {claim_id: set() for claim_id in present}
    for claim_id, unit_id in conn.execute(
        f"SELECT claim_id, evidence_unit_id FROM claim_evidence_units "
        f"WHERE claim_id IN ({placeholders})",  # noqa: S608
        ids,
    ):
        units[str(claim_id)].add(str(unit_id))
    roots: dict[str, set[str]] = {claim_id: set() for claim_id in present}
    for claim_id, root_id in conn.execute(
        f"SELECT cpr.claim_id, cpr.provenance_root_id "
        f"FROM claim_provenance_roots cpr "
        f"JOIN provenance_roots pr ON pr.id = cpr.provenance_root_id "
        f"WHERE cpr.claim_id IN ({placeholders}) AND pr.status = 'established'",  # noqa: S608
        ids,
    ):
        roots[str(claim_id)].add(str(root_id))

    page_frames: dict[tuple[str, int], dict[str, set[str]]] = {}
    for claim_id, asset_hash, asset_page, text_hash in conn.execute(
        f"SELECT DISTINCT claim_id, asset_hash, asset_file_page, asset_text_sha256 "
        f"FROM claim_anchors WHERE claim_id IN ({placeholders})",  # noqa: S608
        ids,
    ):
        page_frames.setdefault((str(asset_hash), int(asset_page)), {}).setdefault(
            str(text_hash), set()
        ).add(str(claim_id))
    overlap_unknown: set[str] = set()
    for frames in page_frames.values():
        if len(frames) > 1:
            overlap_unknown.update(
                claim_id
                for frame_claims in frames.values()
                for claim_id in frame_claims
            )

    established = {
        claim_id
        for claim_id in present
        if units[claim_id] and roots[claim_id] and claim_id not in overlap_unknown
    }
    all_roots = {root for claim_id in established for root in roots[claim_id]}
    root_components = _DisjointSet(all_roots)
    if all_roots:
        root_placeholders = ",".join("?" for _ in all_roots)
        root_params = sorted(all_roots)
        for root_a, root_b, relation in conn.execute(
            f"SELECT root_a, root_b, relation FROM provenance_lineage "
            f"WHERE root_a IN ({root_placeholders}) "
            f"AND root_b IN ({root_placeholders})",  # noqa: S608
            [*root_params, *root_params],
        ):
            if relation in {"shared", "derived"}:
                root_components.union(str(root_a), str(root_b))

    adjacency: dict[str, set[str]] = {}
    for claim_id in established:
        root_ids = {root_components.find(root_id) for root_id in roots[claim_id]}
        for unit_id in units[claim_id]:
            adjacency.setdefault(unit_id, set()).update(root_ids)

    # Maximum bipartite matching: neither one evidence component nor one
    # provenance root may manufacture a second independent support unit.
    matched_root: dict[str, str] = {}

    def _augment(component: str, seen: set[str]) -> bool:
        for root_id in sorted(adjacency.get(component, set())):
            if root_id in seen:
                continue
            seen.add(root_id)
            previous = matched_root.get(root_id)
            if previous is None or _augment(previous, seen):
                matched_root[root_id] = component
                return True
        return False

    count = sum(_augment(component, set()) for component in sorted(adjacency))
    return EvidenceAssessment(
        independent_sources=count,
        scored_claims=len(established),
        unscored_claims=len(present) - len(established),
        established_claims=frozenset(established),
        overlap_unknown_claims=frozenset(overlap_unknown),
    )


__all__ = [
    "DIGEST_1",
    "DIGEST_2",
    "EvidenceAssessment",
    "ValidatedDigest",
    "assess_evidence_support",
    "established_root_ids_for_claim",
    "evidence_unit_ids_for_claim",
    "prune_unreferenced_provenance_roots",
    "rebuild_evidence_units",
    "replace_claim_anchors",
    "replace_claim_provenance_root",
    "source_anchors_for_claim",
    "store_record_structure",
    "validate_digest_for_import",
]
