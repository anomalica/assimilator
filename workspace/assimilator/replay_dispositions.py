"""Strict read-only validation of replay disposition evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from anomalica_common.digest.hashing import claim_fingerprint
from assimilator.matching import match_node


SCHEMA = "anomalica/curation-replay-dispositions/1"
ID_SCHEMA = "anomalica/curation-replay-disposition-id/1"
FILENAME = "replay-dispositions.yaml"
EMPTY_LEDGER_BYTES = b"schema: anomalica/curation-replay-dispositions/1\nentries: []\n"
EMPTY_LEDGER_SHA256 = (
    "sha256:bbef14384bcdc68220da8d53888bb79e29a9c54b5a05d3c12ab490470b3efb50"
)
PHASES = {"merge", "rejection", "rename"}
OUTCOMES = {
    "applied",
    "absorbed",
    "superseded",
    "compensated",
    "contraction_drop",
    "unresolved_drift",
    "unconfirmed",
    "invalid",
}
SAFE_OUTCOMES = {
    "applied",
    "absorbed",
    "superseded",
    "compensated",
    "contraction_drop",
}
BLOCKING_OUTCOMES = OUTCOMES - SAFE_OUTCOMES
NORMAL_REPLAY_OUTCOMES = {
    "applied_normally",
    "compensated_normally",
    "pending_normally",
    "rejected_normally",
}
POSTCONDITION_BY_PHASE = {
    "merge": "merged",
    "rejection": "distinct",
    "rename": "renamed",
}
_ENTRY_KEYS = (
    "id",
    "source_phase",
    "operation_id",
    "source_graph_input_fingerprint",
    "candidate_graph_input_fingerprint",
    "outcome",
    "evidence",
    "classified_at",
    "classifier",
    "supersedes_disposition_id",
)
_SHA256_CHARS = set("0123456789abcdef")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ReplayDispositionError(RuntimeError):
    """The disposition ledger or its candidate evidence is unsafe to use."""


class ReplayDispositionLedgerMissingError(ReplayDispositionError):
    """Required disposition evidence has no ledger file."""


def _fail(message: str) -> None:
    raise ReplayDispositionError(message)


def _exact_mapping(value: Any, keys: tuple[str, ...], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        actual = tuple(value) if isinstance(value, dict) else ()
        _fail(f"{field} must have exact keys {keys!r}; got {actual!r}")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string")
    return value


def _full_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or set(value[7:]) - _SHA256_CHARS
    ):
        _fail(f"{field} must be sha256: followed by 64 lowercase hexadecimal digits")
    return value


def _bare_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _SHA256_CHARS:
        _fail(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _timestamp(value: Any, field: str) -> str:
    text = _nonempty(value, field)
    if not _UTC_TIMESTAMP.fullmatch(text):
        _fail(f"{field} must be normalised UTC YYYY-MM-DDTHH:MM:SS[.fraction]Z")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ReplayDispositionError(f"{field} is not a valid timestamp") from exc
    return text


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _canonical_array(
    value: Any,
    field: str,
    item_validator,
    *,
    nonempty: bool = False,
    strict: bool,
) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "a"
        _fail(f"{field} must be {qualifier} list")
    items = [
        item_validator(item, f"{field}[{index}]", strict=strict)
        for index, item in enumerate(value)
    ]
    keyed = [(_compact_json_bytes(item), item) for item in items]
    canonical = [item for _, item in sorted(dict(keyed).items())]
    if strict and len(canonical) != len(items):
        _fail(f"{field} must not contain duplicates")
    if strict and canonical != items:
        _fail(f"{field} is not in canonical JSON-byte order")
    return canonical


def _natural_identity(value: Any, field: str, *, strict: bool) -> dict[str, Any]:
    identity = _exact_mapping(value, ("name", "node_type", "prior_names"), field)
    name = _nonempty(identity["name"], f"{field}.name")
    node_type = _nonempty(identity["node_type"], f"{field}.node_type")
    prior_names = identity["prior_names"]
    if not isinstance(prior_names, list) or any(
        not isinstance(prior_name, str) or not prior_name.strip()
        for prior_name in prior_names
    ):
        _fail(f"{field}.prior_names must be a list of non-empty strings")
    canonical_names = sorted(set(prior_names))
    if strict and prior_names != canonical_names:
        _fail(f"{field}.prior_names must be deduplicated and lexically sorted")
    return {
        "name": name,
        "node_type": node_type,
        "prior_names": canonical_names,
    }


def _resolved_identity(value: Any, field: str, *, strict: bool) -> dict[str, Any]:
    item = _exact_mapping(value, ("identity", "resolved"), field)
    resolved = _exact_mapping(
        item["resolved"], ("name", "node_type"), f"{field}.resolved"
    )
    return {
        "identity": _natural_identity(
            item["identity"], f"{field}.identity", strict=strict
        ),
        "resolved": {
            "name": _nonempty(resolved["name"], f"{field}.resolved.name"),
            "node_type": _nonempty(
                resolved["node_type"], f"{field}.resolved.node_type"
            ),
        },
    }


def _support_locator(value: Any, field: str, *, strict: bool) -> dict[str, str]:
    del strict
    locator = _exact_mapping(value, ("record_content_hash", "claim_fingerprint"), field)
    return {
        "record_content_hash": _full_sha256(
            locator["record_content_hash"], f"{field}.record_content_hash"
        ),
        "claim_fingerprint": _bare_sha256(
            locator["claim_fingerprint"], f"{field}.claim_fingerprint"
        ),
    }


def _operation_ref(value: Any, field: str, *, strict: bool) -> dict[str, str]:
    del strict
    reference = _exact_mapping(value, ("source_phase", "operation_id"), field)
    if reference["source_phase"] not in PHASES:
        _fail(f"{field}.source_phase must be one of {sorted(PHASES)!r}")
    return {
        "source_phase": reference["source_phase"],
        "operation_id": _nonempty(reference["operation_id"], f"{field}.operation_id"),
    }


def _is_json_pointer(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("/"):
        return False
    index = 0
    while index < len(value):
        if value[index] == "~":
            if index + 1 >= len(value) or value[index + 1] not in "01":
                return False
            index += 1
        index += 1
    return True


def _evidence(value: Any, outcome: str, phase: str, *, strict: bool) -> dict[str, Any]:
    postcondition = POSTCONDITION_BY_PHASE[phase]
    if outcome in {"applied", "absorbed"}:
        keys = ("resolved_identities", "postcondition")
        if outcome == "absorbed":
            keys = ("resolved_identities", "absent_identities", "postcondition")
        evidence = _exact_mapping(value, keys, "evidence")
        if evidence["postcondition"] != postcondition:
            _fail(f"evidence.postcondition must be {postcondition!r} for {phase}")
        canonical = {
            "resolved_identities": _canonical_array(
                evidence["resolved_identities"],
                "evidence.resolved_identities",
                _resolved_identity,
                nonempty=True,
                strict=strict,
            )
        }
        if outcome == "absorbed":
            canonical["absent_identities"] = _canonical_array(
                evidence["absent_identities"],
                "evidence.absent_identities",
                _natural_identity,
                nonempty=True,
                strict=strict,
            )
        canonical["postcondition"] = postcondition
        return canonical
    if outcome == "superseded":
        evidence = _exact_mapping(
            value,
            ("superseded_by", "resolved_identities", "postcondition"),
            "evidence",
        )
        superseded_by = _operation_ref(
            evidence["superseded_by"], "evidence.superseded_by", strict=strict
        )
        reference_postcondition = POSTCONDITION_BY_PHASE[superseded_by["source_phase"]]
        if evidence["postcondition"] != reference_postcondition:
            _fail("evidence.postcondition must match the superseding operation phase")
        return {
            "superseded_by": superseded_by,
            "resolved_identities": _canonical_array(
                evidence["resolved_identities"],
                "evidence.resolved_identities",
                _resolved_identity,
                nonempty=True,
                strict=strict,
            ),
            "postcondition": reference_postcondition,
        }
    if outcome == "compensated":
        if isinstance(value, dict) and "applied_by" in value:
            evidence = _exact_mapping(
                value, ("applied_by", "compensated_by"), "evidence"
            )
            return {
                "applied_by": _operation_ref(
                    evidence["applied_by"], "evidence.applied_by", strict=strict
                ),
                "compensated_by": _operation_ref(
                    evidence["compensated_by"],
                    "evidence.compensated_by",
                    strict=strict,
                ),
            }
        evidence = _exact_mapping(value, ("compensated_by",), "evidence")
        return {
            "compensated_by": _operation_ref(
                evidence["compensated_by"], "evidence.compensated_by", strict=strict
            )
        }
    if outcome == "contraction_drop":
        evidence = _exact_mapping(
            value, ("absent_identities", "absent_support_locators"), "evidence"
        )
        return {
            "absent_identities": _canonical_array(
                evidence["absent_identities"],
                "evidence.absent_identities",
                _natural_identity,
                nonempty=True,
                strict=strict,
            ),
            "absent_support_locators": _canonical_array(
                evidence["absent_support_locators"],
                "evidence.absent_support_locators",
                _support_locator,
                nonempty=True,
                strict=strict,
            ),
        }
    if outcome == "unresolved_drift":
        evidence = _exact_mapping(
            value,
            ("unresolved_identities", "surviving_support_locators", "reason"),
            "evidence",
        )
        canonical = {
            "unresolved_identities": _canonical_array(
                evidence["unresolved_identities"],
                "evidence.unresolved_identities",
                _natural_identity,
                strict=strict,
            ),
            "surviving_support_locators": _canonical_array(
                evidence["surviving_support_locators"],
                "evidence.surviving_support_locators",
                _support_locator,
                strict=strict,
            ),
            "reason": _nonempty(evidence["reason"], "evidence.reason"),
        }
        if (
            not canonical["unresolved_identities"]
            and not canonical["surviving_support_locators"]
        ):
            _fail("unresolved_drift evidence requires at least one non-empty list")
        return canonical
    if outcome == "unconfirmed":
        evidence = _exact_mapping(value, ("reason",), "evidence")
        return {"reason": _nonempty(evidence["reason"], "evidence.reason")}
    evidence = _exact_mapping(value, ("field_paths", "reason"), "evidence")
    paths = evidence["field_paths"]
    if (
        not isinstance(paths, list)
        or not paths
        or any(not _is_json_pointer(path) for path in paths)
    ):
        _fail("evidence.field_paths must be non-empty JSON-pointer strings")
    canonical_paths = sorted(set(paths), key=_compact_json_bytes)
    if strict and paths != canonical_paths:
        _fail("evidence.field_paths must be deduplicated and lexically sorted")
    return {
        "field_paths": canonical_paths,
        "reason": _nonempty(evidence["reason"], "evidence.reason"),
    }


def _classifier(value: Any, field: str) -> dict[str, str]:
    classifier = _exact_mapping(value, ("implementation", "commit"), field)
    implementation = _nonempty(classifier["implementation"], f"{field}.implementation")
    commit = classifier["commit"]
    if not isinstance(commit, str) or len(commit) != 40 or set(commit) - _SHA256_CHARS:
        _fail(f"{field}.commit must be 40 lowercase hexadecimal digits")
    return {"implementation": implementation, "commit": commit}


def _canonical_entry(raw: Any, index: int, *, strict: bool) -> dict[str, Any]:
    field = f"entries[{index}]"
    entry = _exact_mapping(raw, _ENTRY_KEYS, field)
    if entry["source_phase"] not in PHASES:
        _fail(f"{field}.source_phase must be one of {sorted(PHASES)!r}")
    if entry["outcome"] not in OUTCOMES:
        _fail(f"{field}.outcome must be one of {sorted(OUTCOMES)!r}")
    supersedes = entry["supersedes_disposition_id"]
    if supersedes is not None:
        supersedes = _full_sha256(supersedes, f"{field}.supersedes_disposition_id")
    canonical = {
        "id": _full_sha256(entry["id"], f"{field}.id"),
        "source_phase": entry["source_phase"],
        "operation_id": _nonempty(entry["operation_id"], f"{field}.operation_id"),
        "source_graph_input_fingerprint": _full_sha256(
            entry["source_graph_input_fingerprint"],
            f"{field}.source_graph_input_fingerprint",
        ),
        "candidate_graph_input_fingerprint": _full_sha256(
            entry["candidate_graph_input_fingerprint"],
            f"{field}.candidate_graph_input_fingerprint",
        ),
        "outcome": entry["outcome"],
        "evidence": _evidence(
            entry["evidence"], entry["outcome"], entry["source_phase"], strict=strict
        ),
        "classified_at": _timestamp(entry["classified_at"], f"{field}.classified_at"),
        "classifier": _classifier(entry["classifier"], f"{field}.classifier"),
        "supersedes_disposition_id": supersedes,
    }
    proposal_compensation = (
        entry["outcome"] == "compensated" and "applied_by" in canonical["evidence"]
    )
    if entry["outcome"] == "compensated" and proposal_compensation != canonical[
        "operation_id"
    ].startswith("rename-proposal:"):
        _fail("rename proposal compensation evidence requires applied_by")
    return canonical


def disposition_id(entry: Mapping[str, Any]) -> str:
    """Return an ID after canonicalising an authoring mapping's evidence."""
    phase = entry.get("source_phase")
    outcome = entry.get("outcome")
    if phase not in PHASES or outcome not in OUTCOMES:
        _fail("disposition ID input has an invalid phase or outcome")
    supersedes = entry.get("supersedes_disposition_id")
    if supersedes is not None:
        supersedes = _full_sha256(supersedes, "supersedes_disposition_id")
    evidence = _evidence(entry.get("evidence"), outcome, phase, strict=False)
    if outcome == "compensated" and ("applied_by" in evidence) != str(
        entry.get("operation_id") or ""
    ).startswith("rename-proposal:"):
        _fail("rename proposal compensation evidence requires applied_by")
    payload = {
        "schema": ID_SCHEMA,
        "source_phase": phase,
        "operation_id": _nonempty(entry.get("operation_id"), "operation_id"),
        "source_graph_input_fingerprint": _full_sha256(
            entry.get("source_graph_input_fingerprint"),
            "source_graph_input_fingerprint",
        ),
        "candidate_graph_input_fingerprint": _full_sha256(
            entry.get("candidate_graph_input_fingerprint"),
            "candidate_graph_input_fingerprint",
        ),
        "outcome": outcome,
        "evidence": evidence,
        "supersedes_disposition_id": supersedes,
    }
    return "sha256:" + hashlib.sha256(_compact_json_bytes(payload)).hexdigest()


def _canonical_ledger_bytes(entries: list[dict[str, Any]]) -> bytes:
    return yaml.safe_dump(
        {"schema": SCHEMA, "entries": entries},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def read_ledger(
    curation_root: Path | str, *, required_basis_count: int = 0
) -> dict[str, Any]:
    """Read one canonical ledger, or virtual canonical empty when none is needed."""
    if required_basis_count < 0:
        _fail("required_basis_count must not be negative")
    path = Path(curation_root) / FILENAME
    if not path.is_file():
        if required_basis_count:
            raise ReplayDispositionLedgerMissingError(
                f"required replay disposition ledger is absent: {path}"
            )
        content = EMPTY_LEDGER_BYTES
        document: Any = {"schema": SCHEMA, "entries": []}
        exists = False
    else:
        content = path.read_bytes()
        try:
            documents = list(yaml.safe_load_all(content))
        except yaml.YAMLError as exc:
            raise ReplayDispositionError(
                f"invalid replay disposition YAML: {exc}"
            ) from exc
        if len(documents) != 1:
            _fail("replay disposition ledger must contain exactly one YAML document")
        document = documents[0]
        exists = True
    document = _exact_mapping(document, ("schema", "entries"), "ledger")
    if document["schema"] != SCHEMA or not isinstance(document["entries"], list):
        _fail("replay disposition ledger has an invalid schema or entries value")
    if required_basis_count and not document["entries"]:
        _fail(
            "present replay disposition ledger may be empty only when none are required"
        )
    entries = [
        _canonical_entry(raw, index, strict=True)
        for index, raw in enumerate(document["entries"])
    ]
    ordered_entries = sorted(
        entries, key=lambda entry: (entry["classified_at"], entry["id"])
    )
    if entries != ordered_entries:
        _fail("replay disposition entries must be ordered by classified_at and id")
    canonical_bytes = _canonical_ledger_bytes(entries)
    if content != canonical_bytes:
        _fail("replay disposition ledger bytes are not canonical")
    by_id: dict[str, dict[str, Any]] = {}
    tip_by_basis: dict[tuple[str, str, str, str], str] = {}
    for entry in entries:
        if entry["id"] != disposition_id(entry):
            _fail(f"disposition id does not match canonical content: {entry['id']}")
        if entry["id"] in by_id:
            _fail(f"duplicate disposition id {entry['id']}")
        basis = (
            entry["source_phase"],
            entry["operation_id"],
            entry["source_graph_input_fingerprint"],
            entry["candidate_graph_input_fingerprint"],
        )
        supersedes = entry["supersedes_disposition_id"]
        current = tip_by_basis.get(basis)
        if current is None and supersedes is not None:
            target = by_id.get(supersedes)
            if target is None:
                _fail(
                    f"supersedes target is missing or forward-referenced: {supersedes}"
                )
            _fail(f"supersedes target is from a different basis: {supersedes}")
        if current is not None and supersedes is None:
            _fail(f"reclassification must supersede current same-basis tip {current}")
        if current is not None and supersedes != current:
            _fail(f"supersedes target is not current same-basis tip {current}")
        by_id[entry["id"]] = entry
        tip_by_basis[basis] = entry["id"]
    tips = [by_id[entry_id] for entry_id in tip_by_basis.values()]
    tips.sort(key=lambda entry: (entry["classified_at"], entry["id"]))
    return {
        "schema": SCHEMA,
        "entries": entries,
        "active_tips": tips,
        "ledger_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
        "path": str(path),
        "exists": exists,
    }


def _read_yaml_stream(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [
            item
            for item in yaml.safe_load_all(path.read_bytes())
            if isinstance(item, dict)
        ]
    except yaml.YAMLError as exc:
        raise ReplayDispositionError(
            f"cannot read source operation ledger {path}: {exc}"
        ) from exc


def read_active_source_operations(
    curation_root: Path | str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Adapt material source operations and explicit reversal evidence."""
    root = Path(curation_root)
    result: dict[str, dict[str, dict[str, Any]]] = {phase: {} for phase in PHASES}
    specifications = (
        ("merge", "merges.yaml", "merge", "undo", "merge_id"),
        ("rejection", "rejections.yaml", "reject", "unreject", "rejection_id"),
    )
    for phase, filename, apply_op, reverse_op, link_field in specifications:
        operations: dict[str, dict[str, Any]] = {}
        events = _read_yaml_stream(root / filename)
        for event in sorted(
            events,
            key=lambda item: (
                str(item.get("at") or ""),
                str(item.get(link_field) or ""),
            ),
        ):
            linked_id = event.get(link_field)
            if not isinstance(linked_id, str) or not linked_id:
                continue
            if event.get("op") == apply_op:
                if linked_id in operations:
                    _fail(f"duplicate {phase} source operation id {linked_id}")
                operations[linked_id] = event
            elif event.get("op") == reverse_op:
                if linked_id not in operations:
                    _fail(
                        f"{phase} reversal explicitly links missing operation {linked_id}"
                    )
        result[phase] = operations
    from assimilator.rename_ledger import (
        RenameLedgerError,
        parse_proposal_document,
        read_stream,
    )

    try:
        rename_stream = read_stream(root / "renames.yaml")
    except RenameLedgerError as exc:
        raise ReplayDispositionError(str(exc)) from exc
    result["rename"] = {
        operation.operation_id: {
            **operation.event,
            "operation_id": operation.operation_id,
        }
        for operation in rename_stream.operations
    }
    for compensation in rename_stream.compensation_by_operation.values():
        result["rename"][compensation["id"]] = compensation
    for path in sorted((root / "rename-proposals").glob("*.json")):
        try:
            proposal = parse_proposal_document(json.loads(path.read_text()), str(path))
        except (json.JSONDecodeError, OSError, RenameLedgerError):
            continue
        result["rename"][proposal.operation_id] = {
            "at": proposal.at,
            "node": proposal.node,
            "node_name_at_proposal": proposal.source_name,
            "proposed_name": proposal.proposed_name,
        }
    return result


def _current_node(
    conn: sqlite3.Connection, node_id: str
) -> tuple[str, str, str] | None:
    row = conn.execute(
        "SELECT id, name, node_type FROM nodes WHERE id=? AND retired_at IS NULL",
        (node_id,),
    ).fetchone()
    return tuple(str(value) for value in row) if row is not None else None


def _resolve_identity(
    conn: sqlite3.Connection, identity: Mapping[str, Any]
) -> tuple[str, str, str] | None:
    names = sorted({identity["name"], *identity["prior_names"]})
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        "SELECT id, name, node_type FROM nodes WHERE retired_at IS NULL "
        f"AND node_type=? AND name IN ({placeholders}) UNION "
        "SELECT n.id, n.name, n.node_type FROM aliases a JOIN nodes n ON n.id=a.node_id "
        f"WHERE n.retired_at IS NULL AND n.node_type=? AND a.alias IN ({placeholders}) "
        "ORDER BY id",
        (identity["node_type"], *names, identity["node_type"], *names),
    ).fetchall()
    exact = {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
    if len(exact) > 1:
        _fail(f"natural identity {dict(identity)!r} resolves ambiguously")
    if exact:
        node_id, (name, node_type) = next(iter(exact.items()))
        return node_id, name, node_type
    deterministic: dict[str, tuple[str, str]] = {}
    for name in names:
        matched = match_node(conn, name, identity["node_type"])
        if matched is None or matched[1] == "fuzzy":
            continue
        current = _current_node(conn, matched[0])
        if current is not None:
            deterministic[current[0]] = current[1:]
    if len(deterministic) > 1:
        _fail(
            f"natural identity {dict(identity)!r} has ambiguous deterministic matches"
        )
    if not deterministic:
        return None
    node_id, (name, node_type) = next(iter(deterministic.items()))
    return node_id, name, node_type


def _candidate_claim_count(conn: sqlite3.Connection, locator: Mapping[str, str]) -> int:
    count = 0
    records = conn.execute(
        "SELECT id FROM records WHERE content_hash=? ORDER BY id",
        (locator["record_content_hash"],),
    ).fetchall()
    for (record_id,) in records:
        for row in conn.execute(
            "SELECT content, claim_type, original_excerpt, location_in_record "
            "FROM claims WHERE record_id=? ORDER BY id",
            (record_id,),
        ):
            if (
                claim_fingerprint(
                    content=row[0],
                    claim_type=row[1],
                    original_excerpt=row[2],
                    location_in_record=row[3],
                )
                == locator["claim_fingerprint"]
            ):
                count += 1
    return count


def _operation_order(
    operation_id: str, operation: Mapping[str, Any]
) -> tuple[str, str]:
    return str(operation.get("at") or operation.get("timestamp") or ""), operation_id


def _explicitly_references(operation: Mapping[str, Any], operation_id: str) -> bool:
    for key in ("reverses", "references"):
        value = operation.get(key)
        if value == operation_id or (isinstance(value, list) and operation_id in value):
            return True
    return False


def _operation_natural_names(operation: Mapping[str, Any]) -> set[Any]:
    natural = operation.get("node")
    if not isinstance(natural, Mapping):
        return set()
    prior_names = natural.get("prior_names")
    return {
        natural.get("name"),
        *(prior_names if isinstance(prior_names, list) else []),
    }


def _operation_identities(operation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    identities = []
    node = operation.get("node")
    if (
        isinstance(node, Mapping)
        and isinstance(operation.get("new_name"), str)
        and {"name", "node_type", "prior_names"} <= set(node)
    ):
        node = {
            "name": operation["new_name"],
            "node_type": node["node_type"],
            "prior_names": sorted({node["name"], *node["prior_names"]}),
        }
    for value in (
        operation.get("survivor"),
        node,
        *(operation.get("victims") or []),
        *(operation.get("nodes") or []),
    ):
        if isinstance(value, Mapping) and {
            "name",
            "node_type",
            "prior_names",
        } <= set(value):
            identities.append(value)
    return identities


def _identities_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("node_type") == right.get("node_type") and bool(
        {left.get("name"), *(left.get("prior_names") or [])}
        & {right.get("name"), *(right.get("prior_names") or [])}
    )


def _validate_proposal_compensation(
    conn: sqlite3.Connection,
    tip: Mapping[str, Any],
    operations: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    proposal = operations["rename"].get(tip["operation_id"])
    if proposal is None:
        _fail(f"source operation is missing or inactive: {tip['operation_id']}")
    source_name = proposal.get("node_name_at_proposal")
    proposed_name = proposal.get("proposed_name")
    if not isinstance(source_name, str) or not isinstance(proposed_name, str):
        _fail("rename proposal source operation lacks exact source and proposed names")
    refs = [tip["evidence"]["applied_by"], tip["evidence"]["compensated_by"]]
    resolved_operations = []
    for field, reference in zip(("applied_by", "compensated_by"), refs, strict=True):
        if reference["source_phase"] != "rename" or not reference[
            "operation_id"
        ].startswith("rename-ledger:sha256:"):
            _fail(f"{field} must reference an active rename-ledger operation")
        operation = operations["rename"].get(reference["operation_id"])
        if operation is None:
            _fail(
                f"{field} operation is missing or inactive: {reference['operation_id']}"
            )
        if _operation_order(reference["operation_id"], operation) <= _operation_order(
            tip["operation_id"], proposal
        ):
            _fail(f"{field} must reference a later active rename-ledger operation")
        resolved_operations.append(operation)
    applied, compensator = resolved_operations
    if (
        applied.get("proposal_id") != tip["operation_id"]
        or source_name not in _operation_natural_names(applied)
        or applied.get("new_name") != proposed_name
    ):
        _fail("applied_by does not establish the proposal's requested rename")
    applied_node = applied.get("node")
    node_type = (
        applied_node.get("node_type") if isinstance(applied_node, Mapping) else None
    )
    if (
        proposed_name not in _operation_natural_names(compensator)
        or compensator.get("new_name") != source_name
        or not isinstance(compensator.get("node"), Mapping)
        or compensator["node"].get("node_type") != node_type
    ):
        _fail("compensated_by does not restore the proposal's exact typed name")
    linkage_fields = {key for key in ("reverses", "references") if key in compensator}
    if linkage_fields and not _explicitly_references(
        compensator, tip["evidence"]["applied_by"]["operation_id"]
    ):
        _fail("compensating rename does not explicitly link the applied rename")
    final = _resolve_identity(
        conn, {"name": source_name, "node_type": node_type, "prior_names": []}
    )
    if final is None or final[1:] != (source_name, node_type):
        _fail("compensating rename exact typed postcondition is not present")


def _validate_postcondition(conn: sqlite3.Connection, tip: Mapping[str, Any]) -> None:
    evidence = tip["evidence"]
    resolved_ids: list[str] = []
    for item in evidence.get("resolved_identities", []):
        resolved = _resolve_identity(conn, item["identity"])
        if resolved is None:
            _fail(f"resolved identity is absent: {item['identity']!r}")
        if resolved[1:] != (
            item["resolved"]["name"],
            item["resolved"]["node_type"],
        ):
            _fail(
                f"resolved identity current canonical name/type does not match: {item!r}"
            )
        resolved_ids.append(resolved[0])
    if "postcondition" not in evidence:
        return
    postcondition = evidence.get("postcondition")
    if postcondition == "merged" and len(set(resolved_ids)) != 1:
        _fail(
            "merged postcondition requires all resolved identities on one current node"
        )
    if postcondition == "distinct" and len(set(resolved_ids)) < 2:
        _fail("distinct postcondition requires at least two distinct current nodes")
    if postcondition == "renamed" and not resolved_ids:
        _fail("renamed postcondition requires a current resolved identity")


def _validate_tip_evidence(
    conn: sqlite3.Connection,
    tip: Mapping[str, Any],
    operations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    replay_outcomes: Mapping[tuple[str, str], str],
) -> None:
    evidence = tip["evidence"]
    _validate_postcondition(conn, tip)
    source_operation = operations[tip["source_phase"]].get(tip["operation_id"])
    source_identities = (
        _operation_identities(source_operation) if source_operation is not None else []
    )
    if tip["outcome"] in {"applied", "absorbed", "contraction_drop"}:
        evidenced_identities = [
            item["identity"] for item in evidence.get("resolved_identities", [])
        ] + list(evidence.get("absent_identities", []))
        if source_identities and any(
            not any(
                _identities_overlap(source_identity, evidenced_identity)
                for evidenced_identity in evidenced_identities
            )
            for source_identity in source_identities
        ):
            _fail("disposition evidence does not cover the source operation identities")
    for identity in [
        *evidence.get("absent_identities", []),
        *evidence.get("unresolved_identities", []),
    ]:
        if _resolve_identity(conn, identity) is not None:
            _fail(f"identity asserted absent is present: {identity!r}")
    for locator in evidence.get("absent_support_locators", []):
        if _candidate_claim_count(conn, locator) != 0:
            _fail(f"support locator asserted absent is present: {locator!r}")
    for locator in evidence.get("surviving_support_locators", []):
        if _candidate_claim_count(conn, locator) != 1:
            _fail(
                f"surviving support locator does not resolve exactly once: {locator!r}"
            )
    if tip["outcome"] == "compensated" and tip["operation_id"].startswith(
        "rename-proposal:"
    ):
        _validate_proposal_compensation(conn, tip, operations)
        return
    reference_key = (
        "superseded_by" if tip["outcome"] == "superseded" else "compensated_by"
    )
    if reference_key not in evidence:
        return
    reference = evidence[reference_key]
    proposal_merge_supersession = (
        reference_key == "superseded_by"
        and tip["operation_id"].startswith("rename-proposal:")
        and reference["source_phase"] == "merge"
    )
    if (
        reference["source_phase"] != tip["source_phase"]
        and not proposal_merge_supersession
    ):
        _fail(f"{reference_key} must reference an operation in the same phase")
    phase_operations = operations[tip["source_phase"]]
    current = phase_operations.get(tip["operation_id"])
    target = operations[reference["source_phase"]].get(reference["operation_id"])
    if current is None:
        _fail(f"source operation is missing or inactive: {tip['operation_id']}")
    if target is None:
        _fail(
            f"{reference_key} operation is missing or inactive: {reference['operation_id']}"
        )
    if proposal_merge_supersession and replay_outcomes:
        if tip["operation_id"] not in (target.get("proposal_ids") or []):
            _fail("superseding merge does not explicitly link the rename proposal")
        target_basis = (reference["source_phase"], reference["operation_id"])
        if replay_outcomes.get(target_basis) != "applied_normally":
            _fail("superseding merge was not normally confirmed and applied")
        proposal = current
        requested_name = proposal.get("proposed_name")
        requested_name_present = False
        for item in evidence.get("resolved_identities", []):
            resolved = _resolve_identity(conn, item["identity"])
            if resolved is None:
                continue
            names = {resolved[1]}
            names.update(
                str(row[0])
                for row in conn.execute(
                    "SELECT alias FROM aliases WHERE node_id=? ORDER BY alias",
                    (resolved[0],),
                )
            )
            requested_name_present = requested_name_present or requested_name in names
        if not requested_name_present:
            _fail("superseding merge does not establish the requested proposal name")
    elif reference_key == "superseded_by" and replay_outcomes:
        current_identities = _operation_identities(current)
        target_identities = _operation_identities(target)
        evidence_identities = [
            item["identity"] for item in evidence.get("resolved_identities", [])
        ]
        if not current_identities or not target_identities:
            _fail("supersession operations lack natural identity evidence")
        if any(
            not any(
                _identities_overlap(identity, evidence_identity)
                for evidence_identity in evidence_identities
            )
            for identity in current_identities
        ):
            _fail(
                "supersession evidence does not cover the source operation identities"
            )
        if any(
            not any(
                _identities_overlap(evidence_identity, identity)
                for identity in target_identities
            )
            for evidence_identity in evidence_identities
        ):
            _fail("superseding operation does not contain the evidenced identities")
    if _operation_order(reference["operation_id"], target) <= _operation_order(
        tip["operation_id"], current
    ):
        _fail(f"{reference_key} must reference a later active source operation")
    if reference_key == "compensated_by" and not _explicitly_references(
        target, tip["operation_id"]
    ):
        _fail(
            "compensating operation must explicitly reverse or reference the operation"
        )


def validate_replay_dispositions(
    curation_root: Path | str,
    candidate_db: Path | str,
    source_graph_input_fingerprint: str,
    candidate_graph_input_fingerprint: str,
    *,
    active_source_operations: Mapping[str, Mapping[str, Mapping[str, Any]]]
    | None = None,
    replay_result: Any | None = None,
) -> dict[str, Any]:
    """Validate current dispositions and evidence against a read-only candidate."""
    source_fingerprint = _full_sha256(
        source_graph_input_fingerprint, "source_graph_input_fingerprint"
    )
    candidate_fingerprint = _full_sha256(
        candidate_graph_input_fingerprint, "candidate_graph_input_fingerprint"
    )
    raw_operations = (
        read_active_source_operations(curation_root)
        if active_source_operations is None
        else active_source_operations
    )
    operations = {
        phase: dict(phase_operations)
        for phase, phase_operations in raw_operations.items()
    }
    if set(operations) != PHASES:
        _fail(f"active source operations must have exact phases {sorted(PHASES)!r}")
    source_inventory = {
        (phase, operation_id)
        for phase, phase_operations in operations.items()
        for operation_id in phase_operations
    }
    replay_outcomes: dict[tuple[str, str], str] = {}
    replay_blockers: tuple[Any, ...] = ()
    if replay_result is not None:
        inventory = tuple(replay_result.operation_inventory)
        if len(set(inventory)) != len(inventory):
            _fail("strict replay operation inventory contains duplicate ids")
        diagnostics = tuple(replay_result.diagnostics)
        diagnostic_inventory = tuple(
            (diagnostic.source_phase, diagnostic.operation_id)
            for diagnostic in diagnostics
        )
        if diagnostic_inventory != inventory:
            _fail("strict replay diagnostics do not exactly cover its inventory")
        replay_outcomes = {
            (diagnostic.source_phase, diagnostic.operation_id): diagnostic.outcome
            for diagnostic in diagnostics
        }
        replay_blockers = tuple(replay_result.blockers)
        for diagnostic in diagnostics:
            proposal = diagnostic.details.get("proposal")
            if proposal is not None and diagnostic.operation_id.startswith(
                "rename-proposal:"
            ):
                operations["rename"][diagnostic.operation_id] = {
                    "at": proposal.at,
                    "node": proposal.node,
                    "node_name_at_proposal": proposal.source_name,
                    "proposed_name": proposal.proposed_name,
                }
        source_inventory = set(inventory)
    exceptional = (
        {
            basis
            for basis, outcome in replay_outcomes.items()
            if outcome not in NORMAL_REPLAY_OUTCOMES
        }
        if replay_result is not None
        else source_inventory
    )
    ledger = read_ledger(curation_root, required_basis_count=len(exceptional))
    tips = [
        tip
        for tip in ledger["active_tips"]
        if tip["source_graph_input_fingerprint"] == source_fingerprint
        and tip["candidate_graph_input_fingerprint"] == candidate_fingerprint
    ]
    by_operation = {(tip["source_phase"], tip["operation_id"]): tip for tip in tips}
    if len(by_operation) != len(tips):
        _fail("two active disposition tips address the same operation and graph basis")
    actual = set(by_operation)
    unexpected = sorted(actual - source_inventory)
    if unexpected:
        _fail(
            f"dispositions reference missing or inactive source operations: {unexpected!r}"
        )
    missing = sorted(exceptional - actual)
    uri = f"{Path(candidate_db).resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            for tip in tips:
                _validate_tip_evidence(conn, tip, operations, replay_outcomes)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ReplayDispositionError(
            f"candidate evidence validation failed: {exc}"
        ) from exc
    outcomes = [
        {
            "id": tip["id"],
            "source_phase": tip["source_phase"],
            "operation_id": tip["operation_id"],
            "outcome": tip["outcome"],
            "evidence": tip["evidence"],
            "classified_at": tip["classified_at"],
        }
        for tip in tips
    ]
    blocking = [
        outcome for outcome in outcomes if outcome["outcome"] in BLOCKING_OUTCOMES
    ]
    invalid_closures: list[dict[str, Any]] = []
    for basis, tip in by_operation.items():
        if replay_result is None:
            continue
        outcome = replay_outcomes.get(basis)
        if outcome in {"unconfirmed", "invalid"} and tip["outcome"] != outcome:
            invalid_closures.append(
                {
                    "source_phase": basis[0],
                    "operation_id": basis[1],
                    "reason": f"{outcome} replay requires a matching blocking disposition",
                }
            )
        elif (
            tip["outcome"] in {"applied", "compensated"}
            and tip["supersedes_disposition_id"] is None
        ):
            invalid_closures.append(
                {
                    "source_phase": basis[0],
                    "operation_id": basis[1],
                    "reason": f"fresh {tip['outcome']} disposition root is redundant",
                }
            )
        elif outcome in NORMAL_REPLAY_OUTCOMES:
            required_closure = (
                "compensated" if outcome == "compensated_normally" else "applied"
            )
            if tip["outcome"] != required_closure:
                invalid_closures.append(
                    {
                        "source_phase": basis[0],
                        "operation_id": basis[1],
                        "reason": "existing exceptional chain is not closed by "
                        f"{required_closure} after {outcome}",
                    }
                )
    errors = (
        [
            f"strict replay blocker {blocker.source_phase}:{blocker.cause}"
            for blocker in replay_blockers
        ]
        + [
            f"missing disposition for {phase}:{operation_id}"
            for phase, operation_id in missing
        ]
        + [
            f"blocking disposition {outcome['source_phase']}:{outcome['operation_id']} "
            f"has outcome {outcome['outcome']}"
            for outcome in blocking
        ]
        + [
            f"invalid disposition closure {item['source_phase']}:{item['operation_id']}: "
            f"{item['reason']}"
            for item in invalid_closures
        ]
    )
    return {
        "source_graph_input_fingerprint": source_fingerprint,
        "candidate_graph_input_fingerprint": candidate_fingerprint,
        "disposition_ledger_sha256": ledger["ledger_sha256"],
        "active_disposition_count": len(tips),
        "outcomes": outcomes,
        "missing_dispositions": [
            {"source_phase": phase, "operation_id": operation_id}
            for phase, operation_id in missing
        ],
        "blocking_dispositions": blocking,
        "invalid_closures": invalid_closures,
        "errors": errors,
        "replacement_safe": not errors,
    }
