"""Read-only validation of isolated graph rebuild candidates."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import yaml
from anomalica_common.digest import parse_digest_yaml

from assimilator.database import init_db
from assimilator.digest_files import CURRENT_IMPORT_GENERATION, canonical_digests
from assimilator.derived_reverification import (
    DerivedReverificationError,
    OperationRecordSource,
    validate_materialised_receipts,
    validate_receipts as validate_derived_reverification_receipts,
)
from assimilator.scheduler import (
    _curation_sha256,
    _digest_index,
    graph_import_native_deltas,
    graph_input_diagnostics,
)
from assimilator.import_markdown import import_extraction
from assimilator.replay_dispositions import validate_replay_dispositions

_COUNT_TABLES = ("nodes", "records", "claims", "digest_import_receipts")
CURATION_REPLAY_REPORT_SCHEMA = "anomalica/candidate-curation-replay/1"
CURATION_REPLAY_REPORT_FILENAME = "curation-replay-report.json"
_CURATION_STATE_TABLES = {
    "nodes": ("id", "node_type", "name", "metadata", "retired_at"),
    "aliases": None,
    "claim_node_refs": None,
    "node_merges": None,
    "node_rejections": None,
    "rename_proposals": None,
    "claim_ref_status": None,
    "record_nodes": None,
    "record_tags": None,
    "pages": None,
    "page_members": None,
    "page_vetoes": None,
    "superseded_pages": None,
}


def curation_state_sha256(conn: sqlite3.Connection) -> str:
    """Hash the complete relational state written by strict curation replay."""
    tag_nodes: dict[str, str] = {}
    for tag_id, node_id in conn.execute(
        "SELECT tag_id, node_id FROM record_tags "
        "WHERE status = 'applied' AND node_id IS NOT NULL ORDER BY tag_id"
    ):
        tag_nodes.setdefault(str(node_id), f"tag-topic:{tag_id}")
    node_id_columns = {
        "id",
        "node_id",
        "survivor_id",
        "victim_id",
    }
    tables = []
    for table, selected_columns in _CURATION_STATE_TABLES.items():
        columns = list(selected_columns or ()) or [
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        ]
        order = ", ".join(f'"{column}"' for column in columns)
        rows = []
        for row in conn.execute(f"SELECT {order} FROM {table} ORDER BY {order}"):
            values = list(row)
            for index, column in enumerate(columns):
                if column in node_id_columns and values[index] in tag_nodes:
                    values[index] = tag_nodes[values[index]]
                elif table == "record_tags" and column == "undone_at":
                    values[index] = values[index] is not None
            rows.append(values)
        tables.append({"table": table, "columns": columns, "rows": rows})
    encoded = json.dumps(tables, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_curation_replay_report(
    replay_result: Any,
    disposition_report: Mapping[str, Any],
    candidate_conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Build durable evidence for the strict replay already applied to a candidate."""
    return {
        "schema": CURATION_REPLAY_REPORT_SCHEMA,
        "source_graph_input_fingerprint": disposition_report[
            "source_graph_input_fingerprint"
        ],
        "candidate_graph_input_fingerprint": disposition_report[
            "candidate_graph_input_fingerprint"
        ],
        "disposition_ledger_sha256": disposition_report["disposition_ledger_sha256"],
        "curation_state_sha256": curation_state_sha256(candidate_conn),
        "operation_inventory": [
            list(item) for item in replay_result.operation_inventory
        ],
        "diagnostics": _serialise_replay_diagnostics(replay_result),
        "blockers": _serialise_replay_blockers(replay_result),
        "disposition_replacement_safe": disposition_report["replacement_safe"],
    }


def _curation_replay_checks(
    report: Mapping[str, Any] | None,
    candidate_fingerprint: str | None,
    source_fingerprint: str,
    candidate_db: Path,
    replay_result: Any,
    disposition: Mapping[str, Any],
    replayed_conn: sqlite3.Connection,
) -> dict[str, Any]:
    result = {
        "ok": False,
        "schema": None,
        "disposition_ledger_sha256": None,
        "operation_count": 0,
        "errors": [],
    }
    if report is None:
        result["errors"].append("candidate curation replay report is required")
        return result
    required = {
        "schema",
        "source_graph_input_fingerprint",
        "candidate_graph_input_fingerprint",
        "disposition_ledger_sha256",
        "curation_state_sha256",
        "operation_inventory",
        "diagnostics",
        "blockers",
        "disposition_replacement_safe",
    }
    if set(report) != required or report.get("schema") != CURATION_REPLAY_REPORT_SCHEMA:
        result["errors"].append("candidate curation replay report shape is invalid")
        return result
    result["schema"] = report["schema"]
    inventory = report.get("operation_inventory")
    diagnostics = report.get("diagnostics")
    if not isinstance(inventory, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or item[0] not in {"merge", "rejection", "rename"}
        or not isinstance(item[1], str)
        or not item[1]
        for item in inventory
    ):
        result["errors"].append("curation operation inventory is invalid")
        inventory = []
    inventory_keys = [tuple(item) for item in inventory]
    if len(set(inventory_keys)) != len(inventory_keys):
        result["errors"].append("curation operation inventory contains duplicate ids")
    if not isinstance(diagnostics, list):
        result["errors"].append("curation replay diagnostics are invalid")
        diagnostics = []
    diagnostic_shape_valid = all(
        isinstance(item, dict)
        and set(item) == {"source_phase", "operation_id", "outcome", "cause"}
        for item in diagnostics
    )
    diagnostic_keys = [
        (item.get("source_phase"), item.get("operation_id"))
        for item in diagnostics
        if isinstance(item, dict)
    ]
    if (
        not diagnostic_shape_valid
        or len(diagnostic_keys) != len(diagnostics)
        or diagnostic_keys != inventory_keys
    ):
        result["errors"].append(
            "curation replay diagnostics do not exactly cover the operation inventory"
        )
    blockers = report.get("blockers")
    if blockers != []:
        result["errors"].append("curation replay report contains source blockers")
    if report.get("disposition_replacement_safe") is not True:
        result["errors"].append(
            "curation disposition validation was not replacement-safe"
        )
    if report.get("candidate_graph_input_fingerprint") != candidate_fingerprint:
        result["errors"].append("curation replay candidate fingerprint does not match")
    if report.get("source_graph_input_fingerprint") != source_fingerprint:
        result["errors"].append("curation replay source fingerprint does not match")
    candidate = None
    try:
        candidate = _open_read_only(candidate_db)
        candidate_state = curation_state_sha256(candidate)
        replayed_state = curation_state_sha256(replayed_conn)
        if report.get("curation_state_sha256") != candidate_state:
            result["errors"].append("candidate curation state hash does not match")
        if candidate_state != replayed_state:
            result["errors"].append(
                "candidate curation state does not match independent replay"
            )
        actual_inventory = [list(item) for item in replay_result.operation_inventory]
        actual_diagnostics = _serialise_replay_diagnostics(replay_result)
        actual_blockers = _serialise_replay_blockers(replay_result)
        if inventory != actual_inventory:
            result["errors"].append(
                "curation replay report does not match the material source inventory"
            )
        if diagnostics != actual_diagnostics:
            result["errors"].append(
                "curation replay report does not match actual strict replay diagnostics"
            )
        if blockers != actual_blockers:
            result["errors"].append(
                "curation replay report does not match actual strict replay blockers"
            )
        result["disposition_ledger_sha256"] = disposition["disposition_ledger_sha256"]
        if (
            report.get("disposition_ledger_sha256")
            != disposition["disposition_ledger_sha256"]
        ):
            result["errors"].append("curation disposition ledger hash does not match")
        if not disposition["replacement_safe"]:
            result["errors"].extend(disposition["errors"])
    except (OSError, RuntimeError, ValueError) as exc:
        result["errors"].append(str(exc))
    finally:
        if candidate is not None:
            candidate.close()
    result["operation_count"] = len(inventory_keys)
    result["ok"] = not result["errors"]
    return result


def _serialise_replay_diagnostics(replay_result: Any) -> list[dict[str, Any]]:
    return [
        {
            "source_phase": diagnostic.source_phase,
            "operation_id": diagnostic.operation_id,
            "outcome": diagnostic.outcome,
            "cause": diagnostic.cause,
        }
        for diagnostic in replay_result.diagnostics
    ]


def _serialise_replay_blockers(replay_result: Any) -> list[dict[str, Any]]:
    return [
        {
            "source_phase": blocker.source_phase,
            "cause": blocker.cause,
            "details": blocker.details,
        }
        for blocker in replay_result.blockers
    ]


def _independent_canonical_replay(
    canonical_root: Path,
    curation_root: Path,
    candidate_db: Path,
    source_fingerprint: str,
    candidate_fingerprint: str,
) -> tuple[sqlite3.Connection, sqlite3.Connection, Any, dict[str, Any]]:
    """Rebuild and replay in memory, never writing either supplied database."""
    domain = sqlite3.connect(":memory:")
    infrastructure = sqlite3.connect(":memory:")
    init_db(domain)
    init_db(infrastructure)
    try:
        for path in canonical_digests(canonical_root):
            parsed = parse_digest_yaml(path.read_text())
            import_extraction(
                domain,
                parsed,
                section="domain",
                lookup_conns=[infrastructure],
                source_path=str(path),
                source_root=str(canonical_root),
            )
            if parsed["infrastructure_claims"]:
                import_extraction(
                    infrastructure,
                    parsed,
                    section="infrastructure",
                    lookup_conns=[domain],
                    source_path=str(path),
                    source_root=str(canonical_root),
                )

        from assimilator.claim_ref_status_ledger import replay_claim_ref_status
        from assimilator.merge import replay_candidate_curation, replay_rename_proposals
        from assimilator.pages import replay_pages
        from assimilator.propose_pages import replay_vetoes
        from assimilator.tags import replay_tags

        previous_curation = os.environ.get("ANOMALICA_CURATION_DIR")
        os.environ["ANOMALICA_CURATION_DIR"] = str(curation_root)
        try:
            replay_result = replay_candidate_curation(domain)
            disposition = validate_replay_dispositions(
                curation_root,
                candidate_db,
                source_fingerprint,
                candidate_fingerprint,
                replay_result=replay_result,
            )
            replay_rename_proposals(domain, disposition_report=disposition)
            replay_claim_ref_status(domain)
            replay_tags(domain)
            replay_pages(domain)
            replay_vetoes(domain)
        finally:
            if previous_curation is None:
                os.environ.pop("ANOMALICA_CURATION_DIR", None)
            else:
                os.environ["ANOMALICA_CURATION_DIR"] = previous_curation
    except Exception:
        domain.close()
        infrastructure.close()
        raise
    return domain, infrastructure, replay_result, disposition


def _open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _normalise_hash(value: Any) -> str:
    text = str(value or "")
    return text.removeprefix("sha256:")


def _database_checks(conn: sqlite3.Connection) -> dict:
    integrity_rows = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
    foreign_key_rows = sorted(
        [list(row) for row in conn.execute("PRAGMA foreign_key_check")],
        key=lambda row: tuple(str(value) for value in row),
    )
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _COUNT_TABLES
    }
    return {
        "counts": counts,
        "integrity": {
            "ok": integrity_rows == ["ok"],
            "results": integrity_rows,
        },
        "foreign_keys": {
            "ok": not foreign_key_rows,
            "violation_count": len(foreign_key_rows),
            "violations": foreign_key_rows,
        },
    }


def _resolve_receipt_path(digest_path: Any, canonical_root: Path) -> Path | None:
    if not isinstance(digest_path, str):
        return None
    relative = Path(digest_path)
    if relative.is_absolute() or not relative.parts:
        return None
    if relative.parts[0] != canonical_root.name:
        return None
    try:
        resolved = canonical_root.joinpath(*relative.parts[1:]).resolve(strict=True)
        resolved.relative_to(canonical_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _receipt_checks(
    conn: sqlite3.Connection, digest_index: dict[str, dict], canonical_root: Path
) -> dict:
    receipt_rows = conn.execute(
        "SELECT record_content_hash, record_id, digest_path, digest_sha256, "
        "import_generation FROM digest_import_receipts "
        "ORDER BY record_content_hash, record_id"
    ).fetchall()
    record_rows = conn.execute(
        "SELECT id, content_hash FROM records ORDER BY id"
    ).fetchall()
    records = {
        (str(record_id), _normalise_hash(content_hash))
        for record_id, content_hash in record_rows
    }
    receipts = {
        (str(record_id), _normalise_hash(content_hash))
        for content_hash, record_id, _path, _sha256, _generation in receipt_rows
    }

    issues: list[dict] = []
    counters = {
        "total": len(receipt_rows),
        "current": 0,
        "generation_drift": 0,
        "orphan": 0,
        "path_invalid": 0,
        "path_mismatch": 0,
        "hash_mismatch": 0,
        "record_binding_mismatch": 0,
        "records_without_receipt": len(records - receipts),
    }
    for content_hash, record_id, digest_path, digest_sha256, generation in receipt_rows:
        bare_hash = _normalise_hash(content_hash)
        expected = digest_index.get(bare_hash)
        reasons: list[str] = []
        hash_mismatch = False

        if expected is None:
            counters["orphan"] += 1
            reasons.append("orphan_receipt")
        if (str(record_id), bare_hash) not in records:
            counters["record_binding_mismatch"] += 1
            reasons.append("record_binding_mismatch")
        if generation != CURRENT_IMPORT_GENERATION or isinstance(generation, bool):
            counters["generation_drift"] += 1
            reasons.append("import_generation_drift")

        resolved = _resolve_receipt_path(digest_path, canonical_root)
        if resolved is None:
            counters["path_invalid"] += 1
            reasons.append("receipt_path_invalid")
        else:
            actual_sha256 = (
                "sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest()
            )
            if actual_sha256 != digest_sha256:
                hash_mismatch = True
                reasons.append("receipt_file_sha256_mismatch")

        if expected is not None:
            if digest_path != expected["digest_path"]:
                counters["path_mismatch"] += 1
                reasons.append("canonical_digest_path_mismatch")
            if digest_sha256 != expected["digest_sha256"]:
                hash_mismatch = True
                reasons.append("canonical_digest_sha256_mismatch")

        if hash_mismatch:
            counters["hash_mismatch"] += 1

        if reasons:
            issues.append(
                {
                    "record_content_hash": str(content_hash),
                    "record_id": str(record_id),
                    "digest_path": (
                        digest_path
                        if isinstance(digest_path, str)
                        else repr(digest_path)
                    ),
                    "reasons": sorted(reasons),
                }
            )
        else:
            counters["current"] += 1

    for record_id, content_hash in sorted(records - receipts):
        issues.append(
            {
                "record_content_hash": f"sha256:{content_hash}",
                "record_id": record_id,
                "digest_path": None,
                "reasons": ["record_without_receipt"],
            }
        )

    return {
        "ok": not issues,
        "counts": counters,
        "issues": issues,
    }


def _table_materialisation(
    conn: sqlite3.Connection, table: str, *, excluded: set[str] = frozenset()
) -> list[list[Any]]:
    columns = [
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})")
        if row[1] not in excluded
    ]
    projection = ", ".join(f'"{column}"' for column in columns)
    return [
        list(row)
        for row in conn.execute(
            f"SELECT {projection} FROM {table} ORDER BY {projection}"
        )
    ]


def _claim_materialisation_checks(
    candidate: sqlite3.Connection,
    expected: sqlite3.Connection,
    digest_index: dict[str, dict],
    *,
    section: str,
    curated: bool,
) -> dict[str, Any]:
    actual_manifests = {
        _normalise_hash(record_hash): manifest
        for record_hash, manifest in candidate.execute(
            "SELECT record_content_hash, claim_manifest_sha256 "
            "FROM digest_import_receipts"
        )
    }
    expected_manifests = {
        _normalise_hash(record_hash): manifest
        for record_hash, manifest in expected.execute(
            "SELECT record_content_hash, claim_manifest_sha256 "
            "FROM digest_import_receipts"
        )
    }
    empty_manifest = json.dumps(
        {"section": section, "claims": []},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    empty_manifest_sha256 = "sha256:" + hashlib.sha256(empty_manifest).hexdigest()
    manifest_mismatches = []
    for content_hash in sorted(set(actual_manifests) | set(expected_manifests)):
        expected_manifest = expected_manifests.get(content_hash, empty_manifest_sha256)
        actual_manifest = actual_manifests.get(content_hash)
        if actual_manifest != expected_manifest:
            manifest_mismatches.append(
                {
                    "record_content_hash": f"sha256:{content_hash}",
                    "expected": expected_manifest,
                    "actual": actual_manifest,
                }
            )

    actual_claims = _table_materialisation(candidate, "claims", excluded={"created_at"})
    expected_claims = _table_materialisation(
        expected, "claims", excluded={"created_at"}
    )
    actual_refs = _table_materialisation(candidate, "claim_node_refs")
    expected_refs = _table_materialisation(expected, "claim_node_refs")
    actual_records = _table_materialisation(
        candidate, "records", excluded={"created_at", "work_id"}
    )
    expected_records = _table_materialisation(
        expected, "records", excluded={"created_at", "work_id"}
    )
    actual_nodes = (
        []
        if curated
        else _table_materialisation(candidate, "nodes", excluded={"created_at"})
    )
    expected_nodes = (
        []
        if curated
        else _table_materialisation(expected, "nodes", excluded={"created_at"})
    )
    actual_aliases = [] if curated else _table_materialisation(candidate, "aliases")
    expected_aliases = [] if curated else _table_materialisation(expected, "aliases")
    return {
        "ok": not manifest_mismatches
        and actual_claims == expected_claims
        and actual_refs == expected_refs
        and actual_records == expected_records
        and actual_nodes == expected_nodes
        and actual_aliases == expected_aliases,
        "claim_manifest_mismatches": manifest_mismatches,
        "claims_match": actual_claims == expected_claims,
        "claim_node_refs_match": actual_refs == expected_refs,
        "records_match": actual_records == expected_records,
        "nodes_match": actual_nodes == expected_nodes,
        "aliases_match": actual_aliases == expected_aliases,
    }


def _validate_database(
    path: Path,
    digest_index: dict[str, dict],
    canonical_root: Path,
    *,
    require_all_canonical: bool,
) -> tuple[dict, sqlite3.Connection | None]:
    report: dict = {
        "path": str(path),
        "open_mode": "ro",
        "ok": False,
        "total_changes": None,
        "counts": None,
        "integrity": {"ok": False, "results": []},
        "foreign_keys": {"ok": False, "violation_count": 0, "violations": []},
        "receipt_checks": None,
        "native_deltas": None,
    }
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_read_only(path)
        start_changes = conn.total_changes
        report.update(_database_checks(conn))
        report["receipt_checks"] = _receipt_checks(conn, digest_index, canonical_root)
        report["total_changes"] = conn.total_changes - start_changes
        report["ok"] = (
            report["integrity"]["ok"]
            and report["foreign_keys"]["ok"]
            and report["receipt_checks"]["ok"]
            and report["total_changes"] == 0
        )
        if require_all_canonical:
            native = graph_import_native_deltas(conn, digest_index)
            report["native_deltas"] = native
            report["ok"] = report["ok"] and native["current"] == len(digest_index)
            report["ok"] = report["ok"] and all(
                native[key] == 0
                for key in (
                    "missing",
                    "changed",
                    "orphan",
                    "canonical_digests_missing_from_graph",
                    "receipt_hash_mismatches",
                    "import_generation_behind",
                    "import_generation_unknown",
                    "graph_records_without_live_canonical_digest",
                )
            )
    except (AttributeError, OSError, TypeError, sqlite3.Error) as exc:
        report["error"] = str(exc)
    return report, conn


def validate_rebuild_candidate(
    knowledge_db: Path | str,
    infrastructure_db: Path | str,
    canonical_digests_dir: Path | str,
    candidate_curation_dir: Path | str,
    curation_replay_report: Mapping[str, Any] | None,
    source_knowledge_db: Path | str,
    derived_reverification_document: Path | str,
    policy_path: Path | str,
    checked_at: str,
    operation_records: OperationRecordSource,
) -> dict:
    """Validate two isolated candidate databases without opening either writable."""
    knowledge_path = Path(knowledge_db)
    infrastructure_path = Path(infrastructure_db)
    canonical_root = Path(canonical_digests_dir).resolve()
    digest_index = _digest_index(canonical_root)

    domain, domain_conn = _validate_database(
        knowledge_path, digest_index, canonical_root, require_all_canonical=True
    )
    infrastructure, infrastructure_conn = _validate_database(
        infrastructure_path, digest_index, canonical_root, require_all_canonical=False
    )
    graph_input = None
    curation_replay = None
    derived_reverification = None
    replayed_conn = None
    replayed_infrastructure_conn = None
    curation_fingerprint = _curation_sha256()
    try:
        if domain_conn is not None:
            curation_root = Path(candidate_curation_dir).resolve()
            graph_input = graph_input_diagnostics(
                domain_conn,
                digest_index,
                canonical_root,
                curation_root,
            )
            curation_fingerprint = graph_input["input"]["curation_sha256"]
            if graph_input["duplicate_binding_count"]:
                domain["ok"] = False
            source_conn = _open_read_only(Path(source_knowledge_db))
            try:
                source_fingerprint = graph_input_diagnostics(
                    source_conn,
                    digest_index,
                    canonical_root,
                    curation_root,
                )["fingerprint"]
            finally:
                source_conn.close()
            (
                replayed_conn,
                replayed_infrastructure_conn,
                replay_result,
                disposition,
            ) = _independent_canonical_replay(
                canonical_root,
                curation_root,
                knowledge_path,
                source_fingerprint,
                graph_input["fingerprint"],
            )
            claim_materialisation = _claim_materialisation_checks(
                domain_conn,
                replayed_conn,
                digest_index,
                section="domain",
                curated=True,
            )
            domain["receipt_checks"]["claim_materialisation"] = claim_materialisation
            domain["receipt_checks"]["ok"] = (
                domain["receipt_checks"]["ok"] and claim_materialisation["ok"]
            )
            domain["ok"] = domain["ok"] and claim_materialisation["ok"]
            if infrastructure_conn is not None:
                infrastructure_materialisation = _claim_materialisation_checks(
                    infrastructure_conn,
                    replayed_infrastructure_conn,
                    digest_index,
                    section="infrastructure",
                    curated=False,
                )
                infrastructure["receipt_checks"]["claim_materialisation"] = (
                    infrastructure_materialisation
                )
                infrastructure["receipt_checks"]["ok"] = (
                    infrastructure["receipt_checks"]["ok"]
                    and infrastructure_materialisation["ok"]
                )
                infrastructure["ok"] = (
                    infrastructure["ok"] and infrastructure_materialisation["ok"]
                )
            curation_replay = _curation_replay_checks(
                curation_replay_report,
                graph_input["fingerprint"],
                source_fingerprint,
                knowledge_path,
                replay_result,
                disposition,
                replayed_conn,
            )
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as exc:
        domain["ok"] = False
        domain["graph_input_error"] = str(exc)
    finally:
        if replayed_conn is not None:
            replayed_conn.close()
        if replayed_infrastructure_conn is not None:
            replayed_infrastructure_conn.close()
        if domain_conn is not None:
            domain_conn.close()
        if infrastructure_conn is not None:
            infrastructure_conn.close()

    if graph_input is not None and curation_replay is not None:
        try:
            derived_reverification = validate_derived_reverification_receipts(
                derived_reverification_document,
                source_knowledge_db,
                knowledge_path,
                canonical_root,
                candidate_curation_dir,
                policy_path,
                checked_at,
                operation_records,
            )
            document = yaml.safe_load(
                Path(derived_reverification_document).read_bytes()
            )
            binding_errors = []
            if not isinstance(document, dict):
                binding_errors.append(
                    "derived reverification document is not a mapping"
                )
            else:
                if (
                    document.get("candidate_graph_input_fingerprint")
                    != graph_input["fingerprint"]
                ):
                    binding_errors.append(
                        "derived reverification candidate fingerprint does not match"
                    )
                if (
                    document.get("candidate_curation_fingerprint")
                    != graph_input["input"]["curation_sha256"]
                ):
                    binding_errors.append(
                        "derived reverification curation fingerprint does not match"
                    )
            materialisation = validate_materialised_receipts(
                derived_reverification, knowledge_path
            )
            derived_reverification = {
                key: value
                for key, value in derived_reverification.items()
                if key not in {"completed", "items"}
            }
            derived_reverification["materialisation"] = materialisation
            if binding_errors or not materialisation["valid"]:
                derived_reverification["valid"] = False
                derived_reverification["errors"] = [
                    *binding_errors,
                    *materialisation["errors"],
                ]
        except (
            DerivedReverificationError,
            OSError,
            sqlite3.Error,
            yaml.YAMLError,
        ) as exc:
            derived_reverification = {"valid": False, "error": str(exc)}

    return {
        "schema": "anomalica/rebuild-candidate-validation/1",
        "valid": (
            domain["ok"]
            and infrastructure["ok"]
            and graph_input is not None
            and curation_replay is not None
            and curation_replay["ok"]
            and derived_reverification is not None
            and derived_reverification.get("valid") is True
        ),
        "canonical_digest_count": len(digest_index),
        "graph_input_fingerprint": (
            graph_input["fingerprint"] if graph_input is not None else None
        ),
        "curation_fingerprint": curation_fingerprint,
        "graph_input": graph_input,
        "curation_replay": curation_replay,
        "derived_reverification": derived_reverification,
        "databases": {
            "domain": domain,
            "infrastructure": infrastructure,
        },
    }
