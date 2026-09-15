"""Candidate-local planning and receipt-gated replay of derived graph evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from anomalica_common.digest.hashing import claim_fingerprint, fingerprint_of_claim

from assimilator.corroboration_prompt import CORROBORATION_VERIFY_PROMPT
from assimilator.digest_files import CURRENT_IMPORT_GENERATION
from assimilator.scheduler import _digest_index


SCHEMA = "anomalica/rebuild-derived-reverification/1"
OPERATION_RECORD_SCHEMA = "anomalica/rebuild-derived-reverification-operations/1"
FILENAME = "derived-reverification.yaml"
_SHA256 = set("0123456789abcdef")
_OUTCOME_KEYS = {
    "verdict",
    "current_similarity",
    "operation_receipt",
    "verified_at",
}
_OPERATION_RECEIPT_KEYS = {"id", "model", "prompt_sha256", "policy_sha256"}


class DerivedReverificationError(RuntimeError):
    """The plan or its completion evidence is unsafe to use."""


class OperationRecordSource(Protocol):
    """Read completed provider evidence without coupling tests to a provider."""

    def read_operation_record(
        self, operation_receipt_id: str
    ) -> Mapping[str, Any] | None:
        """Return one immutable operation result, or None when it does not exist."""


class MappingOperationRecordSource:
    """In-memory operation evidence source, useful for tests and adapters."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self._records = records

    def read_operation_record(
        self, operation_receipt_id: str
    ) -> Mapping[str, Any] | None:
        return self._records.get(operation_receipt_id)


class FileOperationRecordSource(MappingOperationRecordSource):
    """Read a strict, provider-independent operation-result YAML document."""

    def __init__(self, path: Path | str) -> None:
        try:
            document = yaml.safe_load(Path(path).read_bytes())
        except (OSError, yaml.YAMLError) as exc:
            raise DerivedReverificationError(
                f"invalid operation-record source: {exc}"
            ) from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"schema", "records"}
            or document.get("schema") != OPERATION_RECORD_SCHEMA
            or not isinstance(document.get("records"), list)
        ):
            raise DerivedReverificationError(
                "operation-record source has an invalid schema"
            )
        records: dict[str, Mapping[str, Any]] = {}
        for record in document["records"]:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise DerivedReverificationError(
                    "operation records must be mappings with an id"
                )
            if record["id"] in records:
                raise DerivedReverificationError(
                    f"duplicate operation record {record['id']!r}"
                )
            records[record["id"]] = record
        super().__init__(records)


def _open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return len(value) == 64 and set(value) <= _SHA256


def _normalise_record_hash(value: Any) -> str:
    text = str(value or "")
    bare = text.removeprefix("sha256:")
    if len(bare) != 64 or set(bare) > _SHA256:
        raise DerivedReverificationError(f"invalid record content hash {text!r}")
    return f"sha256:{bare}"


def _validate_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise DerivedReverificationError(f"{field} must be a non-empty timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DerivedReverificationError(
            f"{field} is not an ISO 8601 timestamp"
        ) from exc
    if timestamp.tzinfo is None:
        raise DerivedReverificationError(f"{field} must include a timezone")


def _locator_sort_key(locator: Mapping[str, str]) -> tuple[str, str]:
    return locator["record_content_hash"], locator["claim_fingerprint"]


def _curation_fingerprint(curation_dir: Path) -> str:
    files = sorted(
        path
        for path in curation_dir.rglob("*")
        if path.is_file()
        and path.relative_to(curation_dir).as_posix() != "replay-dispositions.yaml"
    )
    if not files:
        return "sha256:" + _sha256_bytes(b"")
    manifest = [
        [str(path.relative_to(curation_dir)), _sha256_bytes(path.read_bytes())]
        for path in files
    ]
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + _sha256_bytes(encoded)


def _graph_input_fingerprint(
    conn: sqlite3.Connection, curation_fingerprint: str
) -> str:
    rows = conn.execute(
        "SELECT record_content_hash, digest_path, digest_sha256 "
        "FROM digest_import_receipts ORDER BY record_content_hash, digest_path"
    ).fetchall()
    payload = {
        "import_generation": CURRENT_IMPORT_GENERATION,
        "digests": [
            [
                _normalise_record_hash(record_hash).removeprefix("sha256:"),
                path,
                digest_sha256,
            ]
            for record_hash, path, digest_sha256 in rows
        ],
        "curation_sha256": curation_fingerprint,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + _sha256_bytes(encoded)


def _claim_fingerprint(row: tuple[Any, ...]) -> str:
    return claim_fingerprint(
        content=row[0],
        claim_type=row[1],
        original_excerpt=row[2],
        location_in_record=row[3],
    )


def _claim_locator(conn: sqlite3.Connection, claim_id: str) -> dict[str, str]:
    row = conn.execute(
        "SELECT r.content_hash, c.content, c.claim_type, c.original_excerpt, "
        "c.location_in_record FROM claims c JOIN records r ON r.id=c.record_id "
        "WHERE c.id=?",
        (claim_id,),
    ).fetchone()
    if row is None:
        raise DerivedReverificationError(f"source claim {claim_id!r} does not resolve")
    return {
        "record_content_hash": _normalise_record_hash(row[0]),
        "claim_fingerprint": _claim_fingerprint(row[1:]),
    }


def _record_locator(conn: sqlite3.Connection, record_id: str) -> str:
    rows = conn.execute(
        "SELECT content_hash FROM records WHERE id=? ORDER BY content_hash",
        (record_id,),
    ).fetchall()
    if len(rows) != 1:
        raise DerivedReverificationError(
            f"source record {record_id!r} resolved to {len(rows)} records"
        )
    return _normalise_record_hash(rows[0][0])


def _resolve_record(conn: sqlite3.Connection, locator: str) -> str:
    rows = conn.execute(
        "SELECT id FROM records WHERE content_hash=? ORDER BY id", (locator,)
    ).fetchall()
    if len(rows) != 1:
        raise DerivedReverificationError(
            f"record locator {locator!r} resolved to {len(rows)} records"
        )
    return str(rows[0][0])


def _resolve_claim(conn: sqlite3.Connection, locator: Mapping[str, str]) -> str:
    record_id = _resolve_record(conn, locator["record_content_hash"])
    matches = []
    for row in conn.execute(
        "SELECT id, content, claim_type, original_excerpt, location_in_record "
        "FROM claims WHERE record_id=? ORDER BY id",
        (record_id,),
    ):
        if _claim_fingerprint(row[1:]) == locator["claim_fingerprint"]:
            matches.append(str(row[0]))
    if len(matches) != 1:
        raise DerivedReverificationError(
            f"claim locator {dict(locator)!r} resolved to {len(matches)} claims"
        )
    return matches[0]


def _candidate_record(conn: sqlite3.Connection, locator: str) -> str | None:
    rows = conn.execute(
        "SELECT id FROM records WHERE content_hash=? ORDER BY id", (locator,)
    ).fetchall()
    if len(rows) == 1:
        return str(rows[0][0])
    if not rows:
        return None
    raise DerivedReverificationError(
        f"record locator {locator!r} resolved to {len(rows)} candidate records"
    )


def _candidate_claim(
    conn: sqlite3.Connection,
    locator: Mapping[str, str],
) -> str | None:
    record_id = _candidate_record(conn, locator["record_content_hash"])
    if record_id is None:
        return None
    matches = []
    for row in conn.execute(
        "SELECT id, content, claim_type, original_excerpt, location_in_record "
        "FROM claims WHERE record_id=? ORDER BY id",
        (record_id,),
    ):
        if _claim_fingerprint(row[1:]) == locator["claim_fingerprint"]:
            matches.append(str(row[0]))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise DerivedReverificationError(
        f"claim locator {dict(locator)!r} resolved to {len(matches)} candidate claims"
    )


def _canonical_locators(digests_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    records: set[str] = set()
    claims: set[tuple[str, str]] = set()
    for digest in _digest_index(digests_dir).values():
        try:
            document = yaml.safe_load(Path(digest["_path"]).read_bytes())
        except (OSError, yaml.YAMLError) as exc:
            raise DerivedReverificationError(
                f"cannot read canonical digest: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise DerivedReverificationError(
                f"canonical digest {digest['_path']} is not a mapping"
            )
        record_hash = _normalise_record_hash(
            (document.get("record") or {}).get("content_hash")
        )
        if record_hash in records:
            raise DerivedReverificationError(
                f"duplicate canonical record locator {record_hash}"
            )
        records.add(record_hash)
        for key in ("domain_claims", "infrastructure_claims"):
            for claim in document.get(key) or []:
                if isinstance(claim, dict):
                    claims.add((record_hash, fingerprint_of_claim(claim)))
    return records, claims


def _item_id(kind: str, locators: list[Any]) -> str:
    encoded = json.dumps(
        {"kind": kind, "locators": locators},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + _sha256_bytes(encoded)


def _corroboration_prompt(
    item_id: str, locators: list[dict[str, str]], source: sqlite3.Connection
) -> str:
    claim_ids = [_resolve_claim(source, locator) for locator in locators]
    contents = [
        source.execute("SELECT content FROM claims WHERE id=?", (claim_id,)).fetchone()[
            0
        ]
        for claim_id in claim_ids
    ]
    pairs = (
        f"ITEM {item_id} (one candidate pair):\n"
        f'  A: "{contents[0]}"\n'
        f'  B: "{contents[1]}"\n'
    )
    return CORROBORATION_VERIFY_PROMPT.format(pairs_text=pairs)


def build_inventory(
    source_db: Path | str,
    candidate_db: Path | str,
    canonical_digests_dir: Path | str,
    candidate_curation_dir: Path | str,
    policy_path: Path | str,
    checked_at: str,
) -> dict[str, Any]:
    """Inventory all source derived evidence without opening either graph writable."""
    digests_dir = Path(canonical_digests_dir).resolve()
    curation_dir = Path(candidate_curation_dir).resolve()
    policy = Path(policy_path).resolve()
    curation_fingerprint = _curation_fingerprint(curation_dir)
    policy.read_bytes()
    canonical_records, canonical_claims = _canonical_locators(digests_dir)
    _validate_timestamp(checked_at, "checked_at")
    source = _open_read_only(Path(source_db))
    candidate = _open_read_only(Path(candidate_db))
    try:
        source_fingerprint = _graph_input_fingerprint(source, curation_fingerprint)
        candidate_fingerprint = _graph_input_fingerprint(
            candidate, curation_fingerprint
        )
        items: list[dict[str, Any]] = []
        for (
            claim_a,
            claim_b,
            similarity,
            prior_verdict,
            model,
            adjudicated_at,
        ) in source.execute(
            "SELECT claim_a, claim_b, similarity, 'accepted', NULL, NULL "
            "FROM corroborations UNION ALL "
            "SELECT claim_a, claim_b, similarity, 'rejected', model, rejected_at "
            "FROM corroboration_rejections ORDER BY claim_a, claim_b, 4"
        ):
            locators = sorted(
                [_claim_locator(source, claim_a), _claim_locator(source, claim_b)],
                key=lambda locator: (
                    locator["record_content_hash"],
                    locator["claim_fingerprint"],
                ),
            )
            item_id = _item_id("corroboration", locators)
            candidate_claims = [
                _candidate_claim(candidate, locator) for locator in locators
            ]
            canonical_absent_locators = sorted(
                [
                    locator
                    for locator in locators
                    if (
                        locator["record_content_hash"],
                        locator["claim_fingerprint"],
                    )
                    not in canonical_claims
                ],
                key=_locator_sort_key,
            )
            candidate_missing_locators = sorted(
                [
                    locator
                    for locator, claim_id in zip(locators, candidate_claims)
                    if claim_id is None
                ],
                key=_locator_sort_key,
            )
            if canonical_absent_locators:
                status = "dropped_source_absent"
            elif candidate_missing_locators:
                status = "blocked_candidate_import_loss"
            else:
                status = "pending"
            prior = {
                "verdict": prior_verdict,
                "similarity": float(similarity),
            }
            if prior_verdict == "rejected":
                prior.update({"model": model, "adjudicated_at": adjudicated_at})
            item = {
                "id": item_id,
                "kind": "corroboration",
                "status": status,
                "replacement_gate": (
                    "not_required" if status == "dropped_source_absent" else "required"
                ),
                "locators": locators,
                "prior": prior,
            }
            if canonical_absent_locators:
                item["outcome"] = {
                    "absent_locators": canonical_absent_locators,
                    "checked_at": checked_at,
                }
            elif candidate_missing_locators:
                item["outcome"] = {
                    "candidate_missing_locators": candidate_missing_locators,
                    "checked_at": checked_at,
                }
            items.append(item)
        for record_a, record_b, verdict in source.execute(
            "SELECT record_a, record_b, verdict FROM record_relations ORDER BY record_a, record_b"
        ):
            locators = sorted(
                [_record_locator(source, record_a), _record_locator(source, record_b)]
            )
            for locator in locators:
                _resolve_record(source, locator)
            candidate_records = [
                _candidate_record(candidate, locator) for locator in locators
            ]
            canonical_absent_locators = sorted(
                [locator for locator in locators if locator not in canonical_records]
            )
            candidate_missing_locators = sorted(
                [
                    locator
                    for locator, record_id in zip(locators, candidate_records)
                    if record_id is None
                ]
            )
            if canonical_absent_locators:
                status = "dropped_source_absent"
            elif candidate_missing_locators:
                status = "blocked_candidate_import_loss"
            else:
                status = "pending"
            item = {
                "id": _item_id("record_relation", locators),
                "kind": "record_relation",
                "status": status,
                "replacement_gate": (
                    "required"
                    if status == "blocked_candidate_import_loss"
                    else "not_required"
                ),
                "locators": locators,
                "prior": {"verdict": verdict},
            }
            if canonical_absent_locators:
                item["outcome"] = {
                    "absent_locators": canonical_absent_locators,
                    "checked_at": checked_at,
                }
            elif candidate_missing_locators:
                item["outcome"] = {
                    "candidate_missing_locators": candidate_missing_locators,
                    "checked_at": checked_at,
                }
            items.append(item)
        items.sort(key=lambda item: (item["kind"], item["id"]))
        if len({item["id"] for item in items}) != len(items):
            raise DerivedReverificationError(
                "inventory contains duplicate stable item ids"
            )
        return {
            "schema": SCHEMA,
            "source_graph_input_fingerprint": source_fingerprint,
            "candidate_graph_input_fingerprint": candidate_fingerprint,
            "candidate_curation_fingerprint": curation_fingerprint,
            "items": items,
        }
    finally:
        source.close()
        candidate.close()


def _yaml_bytes(document: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(document), sort_keys=False, allow_unicode=True, default_flow_style=False
    ).encode()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_inventory(
    candidate_dir: Path | str, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically write the plan, preserving completed outcomes when unchanged."""
    path = Path(candidate_dir) / FILENAME
    document = dict(inventory)
    if path.exists():
        existing = read_document(path)
        if _inventory_basis(existing) != _inventory_basis(document):
            raise DerivedReverificationError(
                "refusing to overwrite completions for a changed inventory"
            )
        existing_by_id = {item["id"]: item for item in existing["items"]}
        document["items"] = [
            existing_by_id[item["id"]]
            if existing_by_id[item["id"]].get("status") == "completed"
            else item
            for item in document["items"]
        ]
    content = _yaml_bytes(document)
    if not path.exists() or path.read_bytes() != content:
        _write_atomic(path, content)
    return {"path": str(path), "document_sha256": _sha256_bytes(content)}


def _inventory_basis(document: Mapping[str, Any]) -> dict[str, Any]:
    basis = dict(document)
    normalised = []
    for item in basis.get("items", []):
        current = dict(item)
        if (
            current.get("kind") == "corroboration"
            and current.get("status") == "completed"
        ):
            current["status"] = "pending"
            current.pop("outcome", None)
        normalised.append(current)
    basis["items"] = normalised
    return basis


def read_document(path: Path | str) -> dict[str, Any]:
    try:
        document = yaml.safe_load(Path(path).read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise DerivedReverificationError(
            f"invalid derived reverification document: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise DerivedReverificationError(
            "derived reverification document must be a mapping"
        )
    return document


def inventory_report(document: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "corroboration": {
            "pending": 0,
            "completed": 0,
            "dropped_source_absent": 0,
            "blocked_candidate_import_loss": 0,
        },
        "record_relation": {
            "pending": 0,
            "completed": 0,
            "dropped_source_absent": 0,
            "blocked_candidate_import_loss": 0,
        },
        "replacement_gate": {"required": 0, "not_required": 0},
    }
    for item in document.get("items", []):
        kind = item.get("kind")
        status = item.get("status")
        replacement_gate = item.get("replacement_gate")
        if kind in {"corroboration", "record_relation"} and status in {
            "pending",
            "completed",
            "dropped_source_absent",
            "blocked_candidate_import_loss",
        }:
            counts[kind][status] += 1
        if replacement_gate in {"required", "not_required"}:
            counts["replacement_gate"][replacement_gate] += 1
    return {
        "counts": counts,
        "model_calls_made": 0,
        "cost_route_requirement": (
            "No route or spend is authorised. Before generating receipts, resolve the "
            "corroborate model policy, show the aggregate route cost, obtain explicit "
            "approval, and persist one completed operation record per item."
        ),
    }


def _validate_completion(
    outcome: Any,
    item: Mapping[str, Any],
    source: sqlite3.Connection,
    policy_sha256: str,
    operation_records: OperationRecordSource,
) -> None:
    item_id = item["id"]
    if not isinstance(outcome, dict) or set(outcome) != _OUTCOME_KEYS:
        raise DerivedReverificationError(f"outcome for {item_id} has invalid fields")
    if outcome["verdict"] not in {"accepted", "rejected"}:
        raise DerivedReverificationError(
            f"outcome for {item_id} has an invalid verdict"
        )
    similarity = outcome["current_similarity"]
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, (int, float))
        or not math.isfinite(similarity)
        or not -1.0 <= float(similarity) <= 1.0
    ):
        raise DerivedReverificationError(
            f"outcome for {item_id} has invalid current_similarity"
        )
    _validate_timestamp(outcome["verified_at"], f"outcome for {item_id} verified_at")
    receipt = outcome["operation_receipt"]
    if not isinstance(receipt, dict) or set(receipt) != _OPERATION_RECEIPT_KEYS:
        raise DerivedReverificationError(
            f"operation receipt for {item_id} has invalid fields"
        )
    if not isinstance(receipt["id"], str) or not receipt["id"]:
        raise DerivedReverificationError(
            f"operation receipt for {item_id} has invalid id"
        )
    if not isinstance(receipt["model"], str) or not receipt["model"]:
        raise DerivedReverificationError(
            f"operation receipt for {item_id} has invalid model"
        )
    expected_prompt_sha256 = _sha256_bytes(
        _corroboration_prompt(item_id, item["locators"], source).encode()
    )
    if (
        not _is_sha256(receipt["prompt_sha256"])
        or receipt["prompt_sha256"] != expected_prompt_sha256
    ):
        raise DerivedReverificationError(
            f"operation receipt for {item_id} has invalid prompt_sha256"
        )
    if (
        not _is_sha256(receipt["policy_sha256"])
        or receipt["policy_sha256"] != policy_sha256
    ):
        raise DerivedReverificationError(
            f"operation receipt for {item_id} has invalid policy_sha256"
        )
    record = operation_records.read_operation_record(receipt["id"])
    expected = {
        "id": receipt["id"],
        "component": "assimilator",
        "operation": "corroborate",
        "item_id": item_id,
        "verdict": outcome["verdict"],
        "current_similarity": outcome["current_similarity"],
        "model": receipt["model"],
        "prompt_sha256": receipt["prompt_sha256"],
        "policy_sha256": receipt["policy_sha256"],
        "outcome": "ok",
        "provider_started": True,
        "completed_at": outcome["verified_at"],
    }
    if record is None or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise DerivedReverificationError(
            f"receipt for {item_id} has no exact completed provider operation record"
        )


def _canonical_completion(outcome: Mapping[str, Any]) -> dict[str, Any]:
    receipt = outcome["operation_receipt"]
    return {
        "verdict": outcome["verdict"],
        "current_similarity": outcome["current_similarity"],
        "operation_receipt": {
            "id": receipt["id"],
            "model": receipt["model"],
            "prompt_sha256": receipt["prompt_sha256"],
            "policy_sha256": receipt["policy_sha256"],
        },
        "verified_at": outcome["verified_at"],
    }


def complete_item(
    document_path: Path | str,
    item_id: str,
    outcome: Mapping[str, Any],
    source_db: Path | str,
    policy_path: Path | str,
    operation_records: OperationRecordSource,
) -> dict[str, Any]:
    """Prove and atomically embed one corroboration completion."""
    path = Path(document_path)
    document = read_document(path)
    if document.get("schema") != SCHEMA or set(document) != {
        "schema",
        "source_graph_input_fingerprint",
        "candidate_graph_input_fingerprint",
        "candidate_curation_fingerprint",
        "items",
    }:
        raise DerivedReverificationError(
            "derived reverification document has invalid fields or schema"
        )
    matches = [item for item in document.get("items", []) if item.get("id") == item_id]
    if len(matches) != 1:
        raise DerivedReverificationError(f"completion item {item_id!r} is not unique")
    item = matches[0]
    if (
        item.get("kind") != "corroboration"
        or item.get("replacement_gate") != "required"
    ):
        raise DerivedReverificationError(
            f"completion item {item_id!r} is not a pending required corroboration"
        )
    source = _open_read_only(Path(source_db))
    try:
        _validate_completion(
            outcome,
            item,
            source,
            _sha256_bytes(Path(policy_path).read_bytes()),
            operation_records,
        )
    finally:
        source.close()
    canonical_outcome = _canonical_completion(outcome)
    if item.get("status") == "completed":
        if item.get("outcome") != canonical_outcome:
            raise DerivedReverificationError(
                f"completion item {item_id!r} was already completed differently"
            )
        return {
            "item_id": item_id,
            "status": "completed",
            "document_sha256": _sha256_bytes(path.read_bytes()),
        }
    if item.get("status") != "pending":
        raise DerivedReverificationError(
            f"completion item {item_id!r} is not a pending required corroboration"
        )
    item["status"] = "completed"
    item["outcome"] = canonical_outcome
    content = _yaml_bytes(document)
    _write_atomic(path, content)
    return {
        "item_id": item_id,
        "status": "completed",
        "document_sha256": _sha256_bytes(content),
    }


def validate_receipts(
    document_path: Path | str,
    source_db: Path | str,
    candidate_db: Path | str,
    canonical_digests_dir: Path | str,
    candidate_curation_dir: Path | str,
    policy_path: Path | str,
    checked_at: str,
    operation_records: OperationRecordSource,
) -> dict[str, Any]:
    """Validate plan bindings and every surviving corroboration receipt."""
    path = Path(document_path)
    raw = path.read_bytes()
    document = read_document(path)
    if (
        set(document)
        != {
            "schema",
            "source_graph_input_fingerprint",
            "candidate_graph_input_fingerprint",
            "candidate_curation_fingerprint",
            "items",
        }
        or document.get("schema") != SCHEMA
    ):
        raise DerivedReverificationError(
            "derived reverification document has invalid fields or schema"
        )
    fresh = build_inventory(
        source_db,
        candidate_db,
        canonical_digests_dir,
        candidate_curation_dir,
        policy_path,
        checked_at,
    )
    if _inventory_basis(document) != _inventory_basis(fresh):
        raise DerivedReverificationError(
            "derived reverification inventory or input binding changed"
        )
    import_losses = sorted(
        item["id"]
        for item in document["items"]
        if item["status"] == "blocked_candidate_import_loss"
    )
    if import_losses:
        raise DerivedReverificationError(
            f"{len(import_losses)} items are blocked by candidate import loss"
        )
    items = {item["id"]: item for item in document["items"]}
    if len(items) != len(document["items"]):
        raise DerivedReverificationError(
            "derived reverification item ids are not unique"
        )
    completed: dict[str, dict[str, Any]] = {}
    source = _open_read_only(Path(source_db))
    policy_sha256 = _sha256_bytes(Path(policy_path).read_bytes())
    try:
        for item in document["items"]:
            if item["kind"] == "corroboration" and item["status"] == "completed":
                _validate_completion(
                    item.get("outcome"),
                    item,
                    source,
                    policy_sha256,
                    operation_records,
                )
                completed[item["id"]] = item["outcome"]
    finally:
        source.close()
    incomplete = sorted(
        item["id"]
        for item in document["items"]
        if item["kind"] == "corroboration"
        and item["replacement_gate"] == "required"
        and item["status"] != "completed"
    )
    if incomplete:
        raise DerivedReverificationError(
            f"{len(incomplete)} required corroboration items are incomplete"
        )
    return {
        "schema": SCHEMA,
        "valid": True,
        "completed_required_corroborations": len(completed),
        "accepted": sum(
            outcome["verdict"] == "accepted" for outcome in completed.values()
        ),
        "rejected": sum(
            outcome["verdict"] == "rejected" for outcome in completed.values()
        ),
        "pending_record_relations": sum(
            item["kind"] == "record_relation" and item["status"] == "pending"
            for item in document["items"]
        ),
        "document_sha256": _sha256_bytes(raw),
        "completed": completed,
        "items": items,
    }


def validate_materialised_receipts(
    validation: Mapping[str, Any], candidate_db: Path | str
) -> dict[str, Any]:
    """Verify candidate derived rows exactly match validated completion outcomes."""
    candidate = _open_read_only(Path(candidate_db))
    try:
        expected_accepted: list[tuple[str, str, float]] = []
        expected_rejected: list[tuple[str, str, float, str, str]] = []
        for item_id, outcome in sorted(validation["completed"].items()):
            item = validation["items"][item_id]
            claim_ids = sorted(
                _resolve_claim(candidate, locator) for locator in item["locators"]
            )
            if outcome["verdict"] == "accepted":
                expected_accepted.append(
                    (claim_ids[0], claim_ids[1], outcome["current_similarity"])
                )
            else:
                expected_rejected.append(
                    (
                        claim_ids[0],
                        claim_ids[1],
                        outcome["current_similarity"],
                        outcome["operation_receipt"]["model"],
                        outcome["verified_at"],
                    )
                )
        actual_accepted = candidate.execute(
            "SELECT claim_a, claim_b, similarity FROM corroborations ORDER BY claim_a, claim_b"
        ).fetchall()
        actual_rejected = candidate.execute(
            "SELECT claim_a, claim_b, similarity, model, rejected_at "
            "FROM corroboration_rejections ORDER BY claim_a, claim_b"
        ).fetchall()
    finally:
        candidate.close()
    errors = []
    if actual_accepted != sorted(expected_accepted):
        errors.append("candidate corroborations do not match completed outcomes")
    if actual_rejected != sorted(expected_rejected):
        errors.append(
            "candidate corroboration rejections do not match completed outcomes"
        )
    return {
        "valid": not errors,
        "accepted": len(expected_accepted),
        "rejected": len(expected_rejected),
        "errors": errors,
    }


def materialise_receipts(
    document_path: Path | str,
    source_db: Path | str,
    candidate_db: Path | str,
    canonical_digests_dir: Path | str,
    candidate_curation_dir: Path | str,
    policy_path: Path | str,
    checked_at: str,
    operation_records: OperationRecordSource,
) -> dict[str, Any]:
    """Replace candidate corroboration rows with the fully proven receipt set."""
    validation = validate_receipts(
        document_path,
        source_db,
        candidate_db,
        canonical_digests_dir,
        candidate_curation_dir,
        policy_path,
        checked_at,
        operation_records,
    )
    candidate = sqlite3.connect(Path(candidate_db))
    try:
        accepted: list[tuple[str, str, float]] = []
        rejected: list[tuple[str, str, float, str, str]] = []
        for item_id, outcome in sorted(validation["completed"].items()):
            item = validation["items"][item_id]
            claim_ids = sorted(
                _resolve_claim(candidate, locator) for locator in item["locators"]
            )
            if outcome["verdict"] == "accepted":
                accepted.append(
                    (claim_ids[0], claim_ids[1], outcome["current_similarity"])
                )
            else:
                operation_receipt = outcome["operation_receipt"]
                rejected.append(
                    (
                        claim_ids[0],
                        claim_ids[1],
                        outcome["current_similarity"],
                        operation_receipt["model"],
                        outcome["verified_at"],
                    )
                )
        candidate.execute("BEGIN IMMEDIATE")
        try:
            candidate.execute("DELETE FROM corroborations")
            candidate.execute("DELETE FROM corroboration_rejections")
            candidate.executemany(
                "INSERT INTO corroborations (claim_a, claim_b, similarity) VALUES (?, ?, ?)",
                accepted,
            )
            candidate.executemany(
                "INSERT INTO corroboration_rejections "
                "(claim_a, claim_b, similarity, model, rejected_at) VALUES (?, ?, ?, ?, ?)",
                rejected,
            )
            actual_accepted = candidate.execute(
                "SELECT claim_a, claim_b, similarity FROM corroborations ORDER BY claim_a, claim_b"
            ).fetchall()
            actual_rejected = candidate.execute(
                "SELECT claim_a, claim_b, similarity, model, rejected_at "
                "FROM corroboration_rejections ORDER BY claim_a, claim_b"
            ).fetchall()
            if actual_accepted != sorted(accepted) or actual_rejected != sorted(
                rejected
            ):
                raise DerivedReverificationError(
                    "candidate rows did not reconcile to completed outcomes"
                )
            candidate.commit()
        except BaseException:
            candidate.rollback()
            raise
    finally:
        candidate.close()
    return {
        "schema": SCHEMA,
        "materialised": True,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "record_relations_materialised": 0,
        "document_sha256": validation["document_sha256"],
    }
