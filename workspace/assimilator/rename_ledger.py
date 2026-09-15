"""Canonical rename-ledger version 2 events, parsing and append-only writes."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

LEGACY_ID_SCHEMA = "anomalica/legacy-rename-operation-id/1"
EVENT_SCHEMA = "anomalica/rename-ledger-event/2"
PROPOSAL_SCHEMA = "anomalica/rename-proposal/2"
OPERATION_ID_PREFIX = "rename-ledger:sha256:"
COMPENSATION_ID_PREFIX = "rename-compensation:sha256:"
PROPOSAL_ID_PREFIX = "rename-proposal:"
_HEX = set("0123456789abcdef")


class RenameLedgerError(ValueError):
    """The material rename stream is ambiguous or non-canonical."""


@dataclass(frozen=True)
class RenameOperation:
    operation_id: str
    at: str
    op: Literal["rename", "reject_proposal"]
    event: dict[str, Any]
    legacy_raw_id: str | None = None


@dataclass(frozen=True)
class RenameStream:
    operations: tuple[RenameOperation, ...]
    active_operations: tuple[RenameOperation, ...]
    compensation_by_operation: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RenameProposal:
    operation_id: str
    at: str
    id: str
    node: dict[str, Any] | None
    source_name: str
    proposed_name: str
    reason: str | None
    proposed_by: str | None
    source_node_id: str
    legacy: bool
    path: str


def canonical_utc(value: Any, field: str = "at") -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenameLedgerError(f"{field} must be a non-empty offset timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RenameLedgerError(f"{field} must be a valid offset timestamp") from exc
    if parsed.tzinfo is None:
        raise RenameLedgerError(f"{field} must include an offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenameLedgerError(f"{field} must be a non-empty string")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise RenameLedgerError(f"{field} must be a string")
    return value


def _nullable(value: Any, field: str) -> str | None:
    return None if value is None else _nonempty(value, field)


def _canonical_node(value: Any, field: str = "node") -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "node_type",
        "prior_names",
    }:
        raise RenameLedgerError(f"{field} must have name, node_type and prior_names")
    names = value["prior_names"]
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise RenameLedgerError(f"{field}.prior_names must contain non-empty strings")
    return {
        "name": _nonempty(value["name"], f"{field}.name"),
        "node_type": _nonempty(value["node_type"], f"{field}.node_type"),
        "prior_names": sorted(set(names)),
    }


def _content_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return prefix + hashlib.sha256(raw).hexdigest()


def _is_id(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and not (set(value[len(prefix) :]) - _HEX)
    )


def legacy_operation_identity(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("op") != "rename" or "schema" in entry:
        raise RenameLedgerError("not a legacy rename")
    payload = {
        "schema": LEGACY_ID_SCHEMA,
        "op": "rename",
        "rename_id": _string(entry.get("rename_id"), "rename_id"),
        "at": canonical_utc(entry.get("at")),
        "by": _nullable(entry.get("by"), "by"),
        "new_name": _nonempty(entry.get("new_name"), "new_name"),
        "node": _canonical_node(entry.get("node")),
    }
    return {
        "payload": payload,
        "operation_id": _content_id(OPERATION_ID_PREFIX, payload),
    }


def _rename_payload(
    *, at: Any, by: Any, new_name: Any, node: Any, proposal_id: Any
) -> dict[str, Any]:
    if proposal_id is not None:
        proposal_id = _nonempty(proposal_id, "proposal_id")
        if not proposal_id.startswith(
            PROPOSAL_ID_PREFIX
        ) or not proposal_id.removeprefix(PROPOSAL_ID_PREFIX):
            raise RenameLedgerError("proposal_id must use rename-proposal:<id>")
    return {
        "schema": EVENT_SCHEMA,
        "op": "rename",
        "at": canonical_utc(at),
        "by": _nonempty(by, "by"),
        "new_name": _nonempty(new_name, "new_name"),
        "node": _canonical_node(node),
        "proposal_id": proposal_id,
    }


def build_rename_event(
    *, at: str, by: str, new_name: str, node: dict, proposal_id: str | None
) -> dict[str, Any]:
    payload = _rename_payload(
        at=at, by=by, new_name=new_name, node=node, proposal_id=proposal_id
    )
    operation_id = _content_id(OPERATION_ID_PREFIX, payload)
    return {
        "schema": payload["schema"],
        "op": payload["op"],
        "operation_id": operation_id,
        "at": payload["at"],
        "by": payload["by"],
        "new_name": payload["new_name"],
        "node": payload["node"],
        "proposal_id": payload["proposal_id"],
    }


def build_reject_proposal_event(
    *, proposal_id: str, at: str, by: str, reason: str
) -> dict[str, Any]:
    proposal_id = _nonempty(proposal_id, "proposal_id")
    if not proposal_id.startswith(PROPOSAL_ID_PREFIX) or not proposal_id.removeprefix(
        PROPOSAL_ID_PREFIX
    ):
        raise RenameLedgerError("proposal_id must use rename-proposal:<id>")
    payload = {
        "schema": EVENT_SCHEMA,
        "op": "reject_proposal",
        "proposal_id": proposal_id,
        "at": canonical_utc(at),
        "by": _nonempty(by, "by"),
        "reason": _nonempty(reason, "reason"),
    }
    operation_id = _content_id(OPERATION_ID_PREFIX, payload)
    return {
        "schema": payload["schema"],
        "op": payload["op"],
        "operation_id": operation_id,
        "proposal_id": payload["proposal_id"],
        "at": payload["at"],
        "by": payload["by"],
        "reason": payload["reason"],
    }


def build_compensation_event(
    *, at: str, by: str, reason: str, reverses: list[str]
) -> dict[str, Any]:
    canonical_reverses = sorted(set(reverses))
    if (
        not reverses
        or canonical_reverses != reverses
        or any(not _is_id(item, OPERATION_ID_PREFIX) for item in reverses)
    ):
        raise RenameLedgerError(
            "reverses must be non-empty unique sorted operation ids"
        )
    payload = {
        "schema": EVENT_SCHEMA,
        "op": "compensate",
        "at": canonical_utc(at),
        "by": _nonempty(by, "by"),
        "reason": _nonempty(reason, "reason"),
        "reverses": canonical_reverses,
    }
    event_id = _content_id(COMPENSATION_ID_PREFIX, payload)
    return {
        "schema": payload["schema"],
        "op": payload["op"],
        "id": event_id,
        "at": payload["at"],
        "by": payload["by"],
        "reason": payload["reason"],
        "reverses": payload["reverses"],
    }


def verify_v2_event(entry: dict[str, Any]) -> dict[str, Any]:
    op = entry.get("op")
    if op == "rename":
        if tuple(entry) != (
            "schema",
            "op",
            "operation_id",
            "at",
            "by",
            "new_name",
            "node",
            "proposal_id",
        ):
            raise RenameLedgerError(
                "rename event has invalid closed shape or key order"
            )
        expected = build_rename_event(
            at=entry["at"],
            by=entry["by"],
            new_name=entry["new_name"],
            node=entry["node"],
            proposal_id=entry["proposal_id"],
        )
        identity_field = "operation_id"
    elif op == "reject_proposal":
        if tuple(entry) != (
            "schema",
            "op",
            "operation_id",
            "proposal_id",
            "at",
            "by",
            "reason",
        ):
            raise RenameLedgerError(
                "reject event has invalid closed shape or key order"
            )
        expected = build_reject_proposal_event(
            proposal_id=entry["proposal_id"],
            at=entry["at"],
            by=entry["by"],
            reason=entry["reason"],
        )
        identity_field = "operation_id"
    elif op == "compensate":
        if tuple(entry) != ("schema", "op", "id", "at", "by", "reason", "reverses"):
            raise RenameLedgerError(
                "compensation event has invalid closed shape or key order"
            )
        expected = build_compensation_event(
            at=entry["at"],
            by=entry["by"],
            reason=entry["reason"],
            reverses=entry["reverses"],
        )
        identity_field = "id"
    else:
        raise RenameLedgerError("unknown v2 rename event op")
    if entry.get("schema") != EVENT_SCHEMA or entry != expected:
        if entry.get(identity_field) == expected[identity_field]:
            raise RenameLedgerError("event id payload mismatch")
        raise RenameLedgerError("event id does not match canonical payload")
    return expected


def parse_stream(entries: list[Any]) -> RenameStream:
    operations: list[RenameOperation] = []
    controls: list[tuple[str, str, dict[str, Any]]] = []
    seen_ids: dict[str, dict[str, Any]] = {}
    legacy_by_raw: dict[str, list[str]] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise RenameLedgerError(f"rename entry {index} is not an object")
        if raw.get("op") == "rename" and "schema" not in raw:
            identity = legacy_operation_identity(raw)
            operation_id = identity["operation_id"]
            canonical = identity["payload"]
            operation = RenameOperation(
                operation_id,
                canonical["at"],
                "rename",
                {**raw, "at": canonical["at"], "node": canonical["node"]},
                canonical["rename_id"],
            )
            legacy_by_raw.setdefault(canonical["rename_id"], []).append(operation_id)
        elif raw.get("op") == "unrename" and "schema" not in raw:
            controls.append(
                (
                    canonical_utc(raw.get("at")),
                    f"legacy-unrename:{raw.get('rename_id')!s}",
                    raw,
                )
            )
            continue
        else:
            canonical = verify_v2_event(raw)
            if raw["op"] == "compensate":
                controls.append((raw["at"], raw["id"], canonical))
                continue
            operation_id = raw["operation_id"]
            operation = RenameOperation(operation_id, raw["at"], raw["op"], canonical)
        if operation_id in seen_ids:
            if seen_ids[operation_id] == canonical:
                raise RenameLedgerError(f"duplicate_encoding for {operation_id}")
            raise RenameLedgerError(f"operation id collision for {operation_id}")
        seen_ids[operation_id] = canonical
        operations.append(operation)

    operations.sort(key=lambda item: (item.at, item.operation_id))
    by_id = {operation.operation_id: operation for operation in operations}
    compensated: dict[str, dict[str, Any]] = {}
    seen_control_ids: set[str] = set()
    for at, control_id, control in sorted(controls, key=lambda item: item[:2]):
        if control_id in seen_control_ids:
            raise RenameLedgerError(f"duplicate_encoding for {control_id}")
        seen_control_ids.add(control_id)
        if control.get("op") == "unrename":
            matches = legacy_by_raw.get(control.get("rename_id"), [])
            if len(matches) != 1:
                raise RenameLedgerError(
                    "legacy unrename does not bind exactly one base"
                )
            reverses = matches
        else:
            reverses = control["reverses"]
        for operation_id in reverses:
            operation = by_id.get(operation_id)
            if operation is None:
                raise RenameLedgerError(
                    f"compensation references unknown {operation_id}"
                )
            if operation_id in compensated:
                raise RenameLedgerError(f"operation already compensated {operation_id}")
            if (at, control_id) <= (operation.at, operation.operation_id):
                raise RenameLedgerError(f"compensation references later {operation_id}")
        for operation_id in reverses:
            compensated[operation_id] = control
    active = tuple(op for op in operations if op.operation_id not in compensated)
    return RenameStream(tuple(operations), active, compensated)


def read_stream(path: Path) -> RenameStream:
    if not path.is_file():
        return RenameStream((), (), {})
    try:
        entries = list(yaml.safe_load_all(path.read_bytes()))
    except yaml.YAMLError as exc:
        raise RenameLedgerError(f"invalid rename YAML: {exc}") from exc
    return parse_stream(entries)


def append_event_if_absent(path: Path, event: dict[str, Any]) -> bool:
    """Append one verified v2 event under an advisory lock; return whether written."""
    canonical = verify_v2_event(event)
    identity_field = "id" if event["op"] == "compensate" else "operation_id"
    event_id = event[identity_field]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = []
            if path.is_file():
                try:
                    existing = list(yaml.safe_load_all(path.read_bytes()))
                except yaml.YAMLError as exc:
                    raise RenameLedgerError(f"invalid rename YAML: {exc}") from exc
                parse_stream(existing)
            for item in existing:
                item_id = (
                    item.get("id")
                    if item.get("op") == "compensate"
                    else item.get("operation_id")
                )
                if item_id != event_id:
                    continue
                if item == canonical:
                    return False
                raise RenameLedgerError(f"event id collision for {event_id}")
            with path.open("a", encoding="utf-8") as stream:
                stream.write("---\n")
                stream.write(
                    yaml.safe_dump(canonical, sort_keys=False, allow_unicode=True)
                )
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_proposal_document(raw: Any, path: str) -> RenameProposal:
    if not isinstance(raw, dict):
        raise RenameLedgerError("proposal must be an object")
    if raw.get("schema") == PROPOSAL_SCHEMA:
        expected = (
            "schema",
            "id",
            "node",
            "proposed_name",
            "reason",
            "proposed_by",
            "proposed_at",
            "source",
        )
        if tuple(raw) != expected:
            raise RenameLedgerError("v2 proposal has invalid closed shape or key order")
        source = raw["source"]
        if not isinstance(source, dict) or tuple(source) != ("node_id",):
            raise RenameLedgerError("v2 proposal source must contain only node_id")
        proposal_id = _nonempty(raw["id"], "id")
        return RenameProposal(
            PROPOSAL_ID_PREFIX + proposal_id,
            canonical_utc(raw["proposed_at"], "proposed_at"),
            proposal_id,
            _canonical_node(raw["node"]),
            raw["node"]["name"],
            _nonempty(raw["proposed_name"], "proposed_name"),
            _nullable(raw["reason"], "reason"),
            _nullable(raw["proposed_by"], "proposed_by"),
            _nonempty(source["node_id"], "source.node_id"),
            False,
            path,
        )
    expected = {
        "id",
        "node_id",
        "node_name_at_proposal",
        "proposed_name",
        "reason",
        "proposed_by",
        "proposed_at",
    }
    if set(raw) != expected:
        raise RenameLedgerError("legacy proposal must have its exact seven fields")
    proposal_id = _nonempty(raw["id"], "id")
    return RenameProposal(
        PROPOSAL_ID_PREFIX + proposal_id,
        canonical_utc(raw["proposed_at"], "proposed_at"),
        proposal_id,
        None,
        _nonempty(raw["node_name_at_proposal"], "node_name_at_proposal"),
        _nonempty(raw["proposed_name"], "proposed_name"),
        _nullable(raw["reason"], "reason"),
        _nullable(raw["proposed_by"], "proposed_by"),
        _nonempty(raw["node_id"], "node_id"),
        True,
        path,
    )
