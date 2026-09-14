"""Append-only durable events for claim-to-node review decisions."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

import yaml

from anomalica_common.digest.hashing import claim_fingerprint


SCHEMA = "anomalica/claim-ref-status-ledger/1"
_SET_KEYS = {
    "op",
    "id",
    "record_content_hash",
    "claim_fingerprint",
    "node",
    "status",
    "reason",
    "set_at",
    "set_by",
    "salience",
    "source",
}
_UNSET_KEYS = {"op", "id", "reverses", "set_at", "set_by"}


class ClaimRefStatusReplayError(RuntimeError):
    """A durable event could not be validated or resolved safely."""


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, indentless=False)


def ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    curation_dir = Path(
        os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation"))
    )
    return curation_dir / "claim-ref-status.yaml"


def _fingerprint(row: sqlite3.Row | tuple) -> str:
    return claim_fingerprint(
        content=row[0],
        claim_type=row[1],
        original_excerpt=row[2],
        location_in_record=row[3],
    )


def _natural(node: dict) -> dict:
    if not isinstance(node, dict) or set(node) != {
        "name",
        "node_type",
        "prior_names",
    }:
        raise ClaimRefStatusReplayError("node natural identity has invalid fields")
    name = node.get("name")
    node_type = node.get("node_type")
    prior_names = node.get("prior_names")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(node_type, str)
        or not node_type
        or not isinstance(prior_names, list)
        or any(not isinstance(value, str) or not value for value in prior_names)
        or prior_names != sorted(set(prior_names))
    ):
        raise ClaimRefStatusReplayError("node natural identity is not canonical")
    return {"name": name, "node_type": node_type, "prior_names": prior_names}


def _event_id(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _with_id(payload: dict) -> dict:
    event = dict(payload)
    event["id"] = _event_id(payload)
    return {"op": event.pop("op"), "id": event.pop("id"), **event}


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
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _event_yaml(events: list[dict]) -> bytes:
    dumped = yaml.dump(
        events,
        Dumper=_IndentedSafeDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return "".join(f"  {line}" for line in dumped.splitlines(keepends=True)).encode()


def _read_document(path: Path) -> tuple[bytes, list[dict]]:
    if not path.is_file():
        return b"", []
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ClaimRefStatusReplayError(
            f"invalid claim-ref-status YAML: {exc}"
        ) from exc
    if not isinstance(document, dict) or set(document) != {"schema", "entries"}:
        raise ClaimRefStatusReplayError(
            "claim-ref-status ledger must contain only schema and entries"
        )
    if document.get("schema") != SCHEMA:
        raise ClaimRefStatusReplayError(
            f"unsupported claim-ref-status ledger schema: {document.get('schema')!r}"
        )
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ClaimRefStatusReplayError(
            "claim-ref-status ledger entries must be a list"
        )
    return raw, entries


def _validate_entries(entries: list[dict]) -> tuple[dict[str, dict], set[str]]:
    by_id: dict[str, dict] = {}
    for event in entries:
        if not isinstance(event, dict):
            raise ClaimRefStatusReplayError("claim-ref-status entries must be mappings")
        op = event.get("op")
        expected_keys = _SET_KEYS if op == "set" else _UNSET_KEYS
        if op == "unset" and "reason" in event:
            expected_keys = expected_keys | {"reason"}
        if op not in {"set", "unset"} or set(event) != expected_keys:
            raise ClaimRefStatusReplayError(
                f"claim-ref-status event {event.get('id')!r} has invalid fields"
            )
        event_id = event.get("id")
        payload = {key: value for key, value in event.items() if key != "id"}
        if not isinstance(event_id, str) or event_id != _event_id(payload):
            raise ClaimRefStatusReplayError(
                f"claim-ref-status event id does not match payload: {event_id!r}"
            )
        if event_id in by_id:
            raise ClaimRefStatusReplayError(
                f"duplicate claim-ref-status event: {event_id}"
            )
        if not isinstance(event.get("set_at"), str) or not event["set_at"]:
            raise ClaimRefStatusReplayError(f"event {event_id} has no set_at")
        if event.get("set_by") is not None and not isinstance(event["set_by"], str):
            raise ClaimRefStatusReplayError(f"event {event_id} has invalid set_by")
        if event.get("reason") is not None and not isinstance(event["reason"], str):
            raise ClaimRefStatusReplayError(f"event {event_id} has invalid reason")
        if op == "set":
            _validate_set(event)
        elif not isinstance(event.get("reverses"), str) or not event["reverses"]:
            raise ClaimRefStatusReplayError(f"unset event {event_id} has no target")
        by_id[event_id] = event

    reversed_sets: set[str] = set()
    for event in entries:
        if event["op"] != "unset":
            continue
        target = by_id.get(event.get("reverses"))
        if target is None or target["op"] != "set":
            raise ClaimRefStatusReplayError(
                f"unset event {event['id']} does not reverse a set event"
            )
        reversed_sets.add(target["id"])
    return by_id, reversed_sets


def _validate_set(event: dict) -> None:
    if (
        not isinstance(event.get("record_content_hash"), str)
        or not event["record_content_hash"]
    ):
        raise ClaimRefStatusReplayError(f"set event {event['id']} has no record hash")
    if (
        not isinstance(event.get("claim_fingerprint"), str)
        or not event["claim_fingerprint"]
    ):
        raise ClaimRefStatusReplayError(
            f"set event {event['id']} has no claim fingerprint"
        )
    _natural(event.get("node"))
    if event.get("status") not in {"verified", "suspect"}:
        raise ClaimRefStatusReplayError(
            f"invalid status {event.get('status')!r} in event {event['id']}"
        )
    if event.get("salience") not in {
        None,
        "subject",
        "participant",
        "setting",
        "mentioned",
    }:
        raise ClaimRefStatusReplayError(f"set event {event['id']} has invalid salience")
    source = event.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"claim_id", "node_id"}
        or not all(isinstance(value, str) and value for value in source.values())
    ):
        raise ClaimRefStatusReplayError(
            f"set event {event['id']} has invalid source audit"
        )


def _append_events(path: Path, events: list[dict]) -> None:
    if not events:
        return
    raw, existing = _read_document(path)
    if not raw or not existing:
        content = f"schema: {SCHEMA}\nentries:\n".encode() + _event_yaml(events)
    else:
        content = raw + (b"" if raw.endswith(b"\n") else b"\n") + _event_yaml(events)
    _write_atomic(path, content)


def export_claim_ref_status(conn: sqlite3.Connection, path: Path | None = None) -> dict:
    """Append deterministic set events for attached rows not already exported."""
    path = path or ledger_path()
    raw, existing = _read_document(path)
    _validate_entries(existing)
    existing_ids = {event["id"] for event in existing}
    stale_rows = [
        {"claim_id": claim_id, "node_id": node_id}
        for claim_id, node_id in conn.execute(
            "SELECT s.claim_id, s.node_id FROM claim_ref_status s "
            "LEFT JOIN claim_node_refs ref "
            "ON ref.claim_id = s.claim_id AND ref.node_id = s.node_id "
            "WHERE ref.claim_id IS NULL ORDER BY s.claim_id, s.node_id"
        )
    ]
    rows = conn.execute(
        "SELECT rec.content_hash, c.id, c.content, c.claim_type, "
        "c.original_excerpt, c.location_in_record, n.id, n.name, n.node_type, "
        "s.status, s.reason, s.set_at, s.set_by, ref.salience "
        "FROM claim_ref_status s "
        "JOIN claim_node_refs ref "
        "ON ref.claim_id = s.claim_id AND ref.node_id = s.node_id "
        "JOIN claims c ON c.id = s.claim_id "
        "JOIN records rec ON rec.id = c.record_id "
        "JOIN nodes n ON n.id = s.node_id"
    ).fetchall()
    additions = []
    for row in rows:
        record_hash, claim_id = row[:2]
        if not record_hash:
            raise ValueError(
                f"claim {claim_id} belongs to a record with no content_hash"
            )
        fingerprint = _fingerprint(row[2:6])
        node_id, name, node_type = row[6:9]
        aliases = sorted(
            {
                alias
                for (alias,) in conn.execute(
                    "SELECT alias FROM aliases WHERE node_id = ?", (node_id,)
                )
            }
        )
        status, reason, set_at, set_by, salience = row[9:]
        event = _with_id(
            {
                "op": "set",
                "record_content_hash": record_hash,
                "claim_fingerprint": fingerprint,
                "node": {
                    "name": name,
                    "node_type": node_type,
                    "prior_names": aliases,
                },
                "status": status,
                "reason": reason,
                "set_at": set_at,
                "set_by": set_by,
                "salience": salience,
                "source": {"claim_id": claim_id, "node_id": node_id},
            }
        )
        if event["id"] not in existing_ids:
            additions.append(event)
            existing_ids.add(event["id"])
    additions.sort(key=lambda event: (event["set_at"], event["id"]))
    if additions:
        if raw and not existing:
            _write_atomic(
                path,
                f"schema: {SCHEMA}\nentries:\n".encode() + _event_yaml(additions),
            )
        else:
            _append_events(path, additions)
    elif not raw:
        _write_atomic(path, f"schema: {SCHEMA}\nentries: []\n".encode())
    return {
        "exported": len(additions),
        "existing": len(existing),
        "stale": len(stale_rows),
        "stale_rows": stale_rows,
        "path": str(path),
    }


def append_unset_entry(
    reverses: str,
    set_at: str,
    set_by: str | None,
    reason: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append one deterministic compensating event, idempotently."""
    path = path or ledger_path()
    _, entries = _read_document(path)
    by_id, _ = _validate_entries(entries)
    target = by_id.get(reverses)
    if target is None or target["op"] != "set":
        raise ClaimRefStatusReplayError(f"unset target is not a set event: {reverses}")
    payload = {
        "op": "unset",
        "reverses": reverses,
        "set_at": set_at,
        "set_by": set_by,
    }
    if reason is not None:
        payload["reason"] = reason
    event = _with_id(payload)
    if event["id"] not in by_id:
        _append_events(path, [event])
    return event


def _resolve_record(conn: sqlite3.Connection, content_hash: str) -> str | None:
    rows = conn.execute(
        "SELECT id FROM records WHERE content_hash = ? ORDER BY id", (content_hash,)
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ClaimRefStatusReplayError(
            f"record hash {content_hash!r} resolved to {len(rows)} records"
        )
    return rows[0][0]


def _resolve_claim(conn: sqlite3.Connection, record_id: str, fingerprint: str) -> str:
    matches = []
    for row in conn.execute(
        "SELECT id, content, claim_type, original_excerpt, location_in_record "
        "FROM claims WHERE record_id = ? ORDER BY id",
        (record_id,),
    ):
        if _fingerprint(row[1:]) == fingerprint:
            matches.append(row[0])
    if len(matches) != 1:
        raise ClaimRefStatusReplayError(
            f"claim fingerprint {fingerprint!r} in record {record_id!r} "
            f"resolved to {len(matches)} claims"
        )
    return matches[0]


def _resolve_node(conn: sqlite3.Connection, natural: dict) -> str:
    natural = _natural(natural)
    names = sorted({natural["name"], *natural["prior_names"]})
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        "SELECT DISTINCT n.id FROM nodes n LEFT JOIN aliases a ON a.node_id = n.id "
        f"WHERE n.retired_at IS NULL AND n.node_type = ? "
        f"AND (n.name IN ({placeholders}) OR a.alias IN ({placeholders})) "
        "ORDER BY n.id",
        (natural["node_type"], *names, *names),
    ).fetchall()
    if len(rows) != 1:
        raise ClaimRefStatusReplayError(
            f"node {natural['node_type']}:{natural['name']} resolved to "
            f"{len(rows)} live nodes"
        )
    return rows[0][0]


def _identity_key(event: dict) -> str:
    identity = {
        "record_content_hash": event["record_content_hash"],
        "claim_fingerprint": event["claim_fingerprint"],
        "node": event["node"],
    }
    return json.dumps(
        identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _active_sets(
    entries: list[dict], reversed_sets: set[str]
) -> dict[str, dict | None]:
    groups: dict[str, list[dict]] = {}
    for event in entries:
        if event["op"] == "set":
            groups.setdefault(_identity_key(event), []).append(event)
    winners: dict[str, dict | None] = {}
    for key, sets in groups.items():
        active = [event for event in sets if event["id"] not in reversed_sets]
        by_time: dict[str, set[str]] = {}
        for event in active:
            by_time.setdefault(event["set_at"], set()).add(event["status"])
        if any(len(statuses) > 1 for statuses in by_time.values()):
            raise ClaimRefStatusReplayError(
                "contradictory active claim-ref-status sets share one set_at"
            )
        winners[key] = (
            max(active, key=lambda event: (event["set_at"], event["id"]))
            if active
            else None
        )
    return winners


def replay_claim_ref_status(
    conn: sqlite3.Connection,
    path: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Materialise active ledger events atomically after node merges and renames."""
    path = path or ledger_path()
    _, entries = _read_document(path)
    _, reversed_sets = _validate_entries(entries)
    ordered_entries = sorted(entries, key=lambda event: (event["set_at"], event["id"]))
    winners = _active_sets(ordered_entries, reversed_sets)
    log = on_progress or (lambda _: None)
    result = {
        "applied": 0,
        "unset": 0,
        "restored_refs": 0,
        "dropped_record_absent": 0,
        "dropped": [],
    }
    ordered = sorted(
        winners.items(),
        key=lambda item: (
            item[1]["set_at"] if item[1] else "",
            item[1]["id"] if item[1] else item[0],
        ),
    )
    conn.execute("SAVEPOINT replay_claim_ref_status")
    try:
        for key, winner in ordered:
            identity = json.loads(key)
            record_id = _resolve_record(conn, identity["record_content_hash"])
            event_id = winner["id"] if winner else None
            if record_id is None:
                result["dropped_record_absent"] += 1
                result["dropped"].append(
                    {"id": event_id, "outcome": "dropped_record_absent"}
                )
                log(f"  {event_id}: dropped_record_absent")
                continue
            claim_id = _resolve_claim(conn, record_id, identity["claim_fingerprint"])
            node_id = _resolve_node(conn, identity["node"])
            if winner is None:
                conn.execute(
                    "DELETE FROM claim_ref_status WHERE claim_id = ? AND node_id = ?",
                    (claim_id, node_id),
                )
                result["unset"] += 1
                continue
            ref = conn.execute(
                "SELECT salience FROM claim_node_refs WHERE claim_id = ? AND node_id = ?",
                (claim_id, node_id),
            ).fetchone()
            if ref is None:
                if winner["status"] != "verified":
                    raise ClaimRefStatusReplayError(
                        f"suspect event {winner['id']} has no claim-node reference"
                    )
                conn.execute(
                    "INSERT INTO claim_node_refs (claim_id, node_id, salience) "
                    "VALUES (?, ?, ?)",
                    (claim_id, node_id, winner["salience"]),
                )
                result["restored_refs"] += 1
            conn.execute(
                "INSERT INTO claim_ref_status "
                "(claim_id, node_id, status, reason, set_at, set_by) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(claim_id, node_id) DO UPDATE SET "
                "status=excluded.status, reason=excluded.reason, "
                "set_at=excluded.set_at, set_by=excluded.set_by",
                (
                    claim_id,
                    node_id,
                    winner["status"],
                    winner["reason"],
                    winner["set_at"],
                    winner["set_by"],
                ),
            )
            result["applied"] += 1
        conn.execute("RELEASE replay_claim_ref_status")
    except Exception:
        conn.execute("ROLLBACK TO replay_claim_ref_status")
        conn.execute("RELEASE replay_claim_ref_status")
        raise
    log(
        f"Replayed {result['applied']} claim-ref statuses "
        f"({result['unset']} unset, {result['restored_refs']} refs restored, "
        f"{result['dropped_record_absent']} dropped_record_absent)"
    )
    return result
