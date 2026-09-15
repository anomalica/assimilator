"""Node merge: consolidate duplicate entity nodes, reversibly.

The graph fragments because nothing ever merged two nodes that both already exist
(import-time matching only avoids creating a dup; retired_at was never written).
This applies a human-curated merge: re-point every reference to the victim onto
the survivor, fold the victim's name + aliases under the survivor, soft-retire the
victim, and record the operation reversibly.

DURABILITY: the live graph is derived and rebuilt from digests, so a merge applied
only to the DB is lost on rebuild. The source of truth is the append-only curation
ledger (the `curation` repo, ADR 0038); the importer/rebuild REPLAYS it after
import. Replay is keyed on NATURAL identity (canonical_name + node_type +
prior_names) - never node ids, which are ephemeral uuid4-per-extraction - because
that is the identity the importer itself resolves entities by. Node ids are
recorded as an audit snapshot only.

A node merge re-points claim refs, speakers, producers and aliases. Claim rows and
their source/provenance envelope remain unchanged.

Host-runnable: `python -m assimilator.merge --survivor <id> --victims <id,id>
--name "<canonical>"` and `--undo <merge_id>`. No Claude, no money.
"""

from __future__ import annotations

import json
import fcntl
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from assimilator.embed_batches import forget_embeddings
from assimilator.database import init_db
from assimilator.matching import match_node
from assimilator.data_dir import data_dir
from assimilator.rename_ledger import (
    OPERATION_ID_PREFIX,
    PROPOSAL_ID_PREFIX,
    RenameLedgerError,
    RenameProposal,
    RenameStream,
    append_event_if_absent,
    build_rename_event,
    legacy_operation_identity,
    parse_proposal_document,
    read_stream,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Mark's rule (2026-09-03): NO SESSION APPLIES A MERGE. Every merge - a curation
# session's, master's, an identity or fragment pass's - is a proposal in the
# workbench queue, and only a merge Mark confirms there is applied and written
# to the ledger. The confirmation travels in the ledger entry as
# `confirmation: {by, at, via}` (via names which of his two entry points issued
# it: the proposal queue, or the rename-onto-an-existing-name flow). Not a
# security boundary - a guard against habit and mistake: 163 of the 164 merges
# applied to the graph before this rule were made by AI sessions.
#
# Entries dated before the rule landed are GRANDFATHERED: they carry no block
# and still replay, because "replay applies only confirmed entries" applied
# literally would un-merge the whole graph on the next rebuild, and Mark's
# instruction was to list them for review, not undo them.
CONFIRMATION_REQUIRED_FROM = "2026-09-03T02:10:00Z"
CONFIRMATION_VIAS = (
    "workbench-queue",
    "workbench-rename",
    "workbench-compose",  # the same guard on a page composition
)


def confirmation_block(by: str, at: str | None, via: str) -> dict:
    if not by or not by.strip():
        raise ValueError("a confirmation needs who confirmed")
    if via not in CONFIRMATION_VIAS:
        raise ValueError(f"via must be one of {CONFIRMATION_VIAS}, not {via!r}")
    return {"by": by.strip(), "at": at or _now(), "via": via}


def confirmed(entry: dict) -> bool:
    """Whether a ledger merge entry may be applied: it carries a confirmation
    block, or it predates the rule."""
    block = entry.get("confirmation")
    if (
        isinstance(block, dict)
        and set(block) == {"by", "at", "via"}
        and isinstance(block.get("by"), str)
        and block["by"].strip()
        and _valid_timestamp(block.get("at"))
        and block.get("via") in CONFIRMATION_VIAS
    ):
        return True
    return str(entry.get("at") or "") < CONFIRMATION_REQUIRED_FROM


def _node(conn: sqlite3.Connection, node_id: str) -> tuple[str, str] | None:
    row = conn.execute(
        "SELECT name, node_type FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def _aliases(conn: sqlite3.Connection, node_id: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT alias FROM aliases WHERE node_id = ?", (node_id,)
        ).fetchall()
    ]


_SALIENCE_ORDER = {
    None: 0,
    "mentioned": 1,
    "setting": 2,
    "participant": 3,
    "subject": 4,
}


def _stronger_salience(a: str | None, b: str | None) -> str | None:
    """Keep the strongest aboutness when two aliases occur in one claim."""
    return max((a, b), key=lambda value: _SALIENCE_ORDER[value])


# --- The merge operation (live DB) ---


def merge_nodes(
    conn: sqlite3.Connection,
    survivor_id: str,
    victim_ids: list[str],
    canonical_name: str,
    merge_id: str,
    created_at: str | None = None,
    created_by: str | None = None,
) -> int:
    """Apply a merge to the live graph and record it in node_merges. Returns the
    number of victims actually merged (a missing victim is skipped)."""
    created_at = created_at or _now()
    survivor = _node(conn, survivor_id)
    if survivor is None:
        raise ValueError(f"survivor not found: {survivor_id}")
    survivor_prior_name = survivor[0]
    merged = 0
    for victim_id in victim_ids:
        victim = _node(conn, victim_id)
        if victim is None or victim_id == survivor_id:
            continue
        victim_name = victim[0]

        victim_claims = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT claim_id, salience FROM claim_node_refs WHERE node_id = ?",
                (victim_id,),
            ).fetchall()
        }
        survivor_claims = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT claim_id, salience FROM claim_node_refs WHERE node_id = ?",
                (survivor_id,),
            ).fetchall()
        }
        refs_only_victim = sorted(victim_claims.keys() - survivor_claims.keys())
        refs_both = sorted(victim_claims.keys() & survivor_claims.keys())
        for cid in refs_only_victim:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id, salience) "
                "VALUES (?, ?, ?)",
                (cid, survivor_id, victim_claims[cid]),
            )
        for cid in refs_both:
            conn.execute(
                "UPDATE claim_node_refs SET salience = ? WHERE claim_id = ? AND node_id = ?",
                (
                    _stronger_salience(survivor_claims[cid], victim_claims[cid]),
                    cid,
                    survivor_id,
                ),
            )
        conn.execute("DELETE FROM claim_node_refs WHERE node_id = ?", (victim_id,))

        speaker_claims = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM claims WHERE speaker_id = ?", (victim_id,)
            ).fetchall()
        ]
        conn.execute(
            "UPDATE claims SET speaker_id = ? WHERE speaker_id = ?",
            (survivor_id, victim_id),
        )

        producer_records = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM records WHERE producer_id = ?", (victim_id,)
            ).fetchall()
        ]
        conn.execute(
            "UPDATE records SET producer_id = ? WHERE producer_id = ?",
            (survivor_id, victim_id),
        )

        moved_aliases = _aliases(conn, victim_id)
        for alias in moved_aliases:
            conn.execute(
                "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
                (alias, survivor_id),
            )
        conn.execute("DELETE FROM aliases WHERE node_id = ?", (victim_id,))
        conn.execute(
            "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
            (victim_name, survivor_id),
        )

        conn.execute(
            "UPDATE nodes SET retired_at = ? WHERE id = ?", (created_at, victim_id)
        )
        # A retired node is not a live row, so the invariant says its stamp and
        # vector go too. Deliberately unconditional rather than "except victims,
        # so undo is cheaper": an invariant with an exception reads as a bug to
        # whoever finds the exception first. undo_merge pays one re-embed per
        # resurrected victim, which is about a second each.
        forget_embeddings(conn, "node", victim_id)

        reversal = json.dumps(
            {
                "refs_only_victim": refs_only_victim,
                "refs_both": refs_both,
                "victim_ref_salience": victim_claims,
                "survivor_ref_salience": survivor_claims,
                "speaker_claims": speaker_claims,
                "producer_records": producer_records,
                "moved_aliases": moved_aliases,
                "added_victim_alias": victim_name,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO node_merges (merge_id, survivor_id, victim_id, "
            "victim_prior_name, survivor_prior_name, canonical_name, created_at, "
            "created_by, undone_at, reversal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (
                merge_id,
                survivor_id,
                victim_id,
                victim_name,
                survivor_prior_name,
                canonical_name,
                created_at,
                created_by,
                reversal,
            ),
        )
        merged += 1

    if canonical_name != survivor_prior_name:
        conn.execute(
            "UPDATE nodes SET name = ? WHERE id = ?", (canonical_name, survivor_id)
        )
        # The vector is of the OLD name. The stamp would still read current, so
        # nothing would ever re-embed it and the node would answer to a name it
        # no longer has.
        forget_embeddings(conn, "node", survivor_id)
    conn.commit()
    return merged


def undo_merge(conn: sqlite3.Connection, merge_id: str) -> int:
    """Reverse a merge in the live DB using the recorded reversal data. Returns
    the number of victims restored."""
    rows = conn.execute(
        "SELECT survivor_id, victim_id, survivor_prior_name, reversal "
        "FROM node_merges WHERE merge_id = ? AND undone_at IS NULL "
        "ORDER BY rowid DESC",
        (merge_id,),
    ).fetchall()
    if not rows:
        return 0
    survivor_id = rows[0][0]
    survivor_prior_name = rows[0][2]
    for survivor_id, victim_id, survivor_prior_name, reversal_json in rows:
        rev = json.loads(reversal_json)
        victim_salience = rev.get("victim_ref_salience", {})
        survivor_salience = rev.get("survivor_ref_salience", {})
        conn.execute("UPDATE nodes SET retired_at = NULL WHERE id = ?", (victim_id,))
        for cid in rev["refs_only_victim"]:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id, salience) "
                "VALUES (?, ?, ?)",
                (cid, victim_id, victim_salience.get(cid)),
            )
            conn.execute(
                "DELETE FROM claim_node_refs WHERE claim_id = ? AND node_id = ?",
                (cid, survivor_id),
            )
        for cid in rev["refs_both"]:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id, salience) "
                "VALUES (?, ?, ?)",
                (cid, victim_id, victim_salience.get(cid)),
            )
            conn.execute(
                "UPDATE claim_node_refs SET salience = ? WHERE claim_id = ? AND node_id = ?",
                (survivor_salience.get(cid), cid, survivor_id),
            )
        for cid in rev["speaker_claims"]:
            conn.execute(
                "UPDATE claims SET speaker_id = ? WHERE id = ?", (victim_id, cid)
            )
        for rid in rev["producer_records"]:
            conn.execute(
                "UPDATE records SET producer_id = ? WHERE id = ?", (victim_id, rid)
            )
        for alias in rev["moved_aliases"]:
            conn.execute(
                "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
                (alias, victim_id),
            )
            conn.execute(
                "DELETE FROM aliases WHERE alias = ? AND node_id = ?",
                (alias, survivor_id),
            )
        conn.execute(
            "DELETE FROM aliases WHERE alias = ? AND node_id = ?",
            (rev["added_victim_alias"], survivor_id),
        )
        conn.execute(
            "UPDATE node_merges SET undone_at = ? WHERE merge_id = ? AND victim_id = ?",
            (_now(), merge_id, victim_id),
        )
    conn.execute(
        "UPDATE nodes SET name = ? WHERE id = ?", (survivor_prior_name, survivor_id)
    )
    conn.commit()
    return len(rows)


# --- The durable curation ledger (append-only YAML stream) ---


def ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "merges.yaml"


def rename_proposals_dir() -> Path:
    """Where a reviewer drops a proposed node rename.

    A DIRECTORY rather than a database handle, at the workbench's request and
    on its reasoning: its read-only connection is what stops it corrupting the
    graph, and one writable table would be a precedent instead of a boundary.
    It sits beside the curation ledgers because it is curation input, and the
    ledger is what the rename ultimately writes to.
    """
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "rename-proposals"


def read_rename_proposals() -> list[dict]:
    """Every proposal file, oldest first by filename.

    A file that will not parse is REPORTED by the caller, never skipped: a
    reviewer who asked for a change is owed an answer, and a proposal that
    vanishes silently is indistinguishable from one nobody made.
    """
    directory = rename_proposals_dir()
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            out.append({"_path": str(path), "_error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(doc, dict):
            out.append(
                {
                    "_path": str(path),
                    "_error": f"expected a JSON object, got {type(doc).__name__}",
                }
            )
            continue
        doc["_path"] = str(path)
        out.append(doc)
    return out


def _natural(conn: sqlite3.Connection, node_id: str) -> dict:
    name, node_type = _node(conn, node_id)
    return {
        "name": name,
        "node_type": node_type,
        "prior_names": _aliases(conn, node_id),
    }


def append_merge_entry(
    conn: sqlite3.Connection,
    survivor_id: str,
    victim_ids: list[str],
    canonical_name: str,
    merge_id: str,
    created_at: str,
    created_by: str | None,
    confirmation: dict | None = None,
    proposal_ids: list[str] | None = None,
) -> None:
    """Append a merge entry to the durable ledger, keyed on natural identity
    (names), with ids as an audit snapshot. Captured BEFORE the live merge so the
    victims still resolve to their own natural identity.

    Refuses an entry with no confirmation block: an unconfirmed merge is a
    proposal, and a proposal is not a ledger entry (see propose_merge)."""
    if not confirmation:
        raise ValueError(
            "a merge is written to the ledger only with Mark's confirmation; "
            "without one it is a proposal (propose_merge)"
        )
    proposal_ids = list(proposal_ids or [])
    if proposal_ids != sorted(set(proposal_ids)) or any(
        not isinstance(proposal_id, str)
        or not proposal_id.startswith(PROPOSAL_ID_PREFIX)
        or not proposal_id.removeprefix(PROPOSAL_ID_PREFIX)
        for proposal_id in proposal_ids
    ):
        raise ValueError(
            "proposal_ids must be unique sorted rename-proposal:<id> values"
        )
    entry = {
        "op": "merge",
        "merge_id": merge_id,
        "at": created_at,
        "by": created_by,
        "confirmation": dict(confirmation),
        "canonical_name": canonical_name,
        "survivor": _natural(conn, survivor_id),
        "victims": [_natural(conn, v) for v in victim_ids if _node(conn, v)],
        "audit": {"survivor_id": survivor_id, "victim_ids": list(victim_ids)},
    }
    if proposal_ids:
        entry["proposal_ids"] = proposal_ids
    existing = [
        item
        for item in read_ledger()
        if isinstance(item, dict)
        and item.get("op") == "merge"
        and item.get("merge_id") == merge_id
    ]
    if existing:
        if len(existing) == 1 and existing[0] == entry:
            return
        raise ValueError(f"merge id collision for {merge_id}")
    if proposal_ids:
        _validate_pending_merge_proposals(
            conn,
            proposal_ids,
            survivor_id,
            victim_ids,
            canonical_name,
            excluding_merge_id=merge_id,
        )
    _append_merge_if_absent(entry)


def _validate_pending_merge_proposals(
    conn: sqlite3.Connection,
    proposal_ids: list[str],
    survivor_id: str,
    victim_ids: list[str],
    canonical_name: str,
    *,
    excluding_merge_id: str | None = None,
) -> None:
    proposals, diagnostics, blockers = _parse_proposals(read_rename_proposals())
    if diagnostics or blockers:
        raise ValueError("rename proposal source contains malformed entries")
    by_id = {proposal.operation_id: proposal for proposal in proposals}
    unknown = sorted(set(proposal_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown rename proposal ids: {unknown!r}")
    stream = read_stream(renames_ledger_path())
    resolved = {
        operation.event["proposal_id"]
        for operation in stream.active_operations
        if operation.event.get("proposal_id") is not None
    }
    merge_entries = read_ledger()
    undone = {
        entry.get("merge_id")
        for entry in merge_entries
        if isinstance(entry, dict) and entry.get("op") == "undo"
    }
    for entry in merge_entries:
        if isinstance(entry, dict) and entry.get("op") == "merge":
            if (
                entry.get("merge_id") != excluding_merge_id
                and entry.get("merge_id") not in undone
                and confirmed(entry)
            ):
                resolved.update(entry.get("proposal_ids") or [])
    nonpending = sorted(set(proposal_ids) & resolved)
    if nonpending:
        raise ValueError(f"rename proposal ids are not pending: {nonpending!r}")
    selected_ids = {survivor_id, *victim_ids}
    resulting_names = {canonical_name}
    for node_id in selected_ids:
        node = _node(conn, node_id)
        if node is not None:
            resulting_names.add(node[0])
            resulting_names.update(_aliases(conn, node_id))
    for proposal_id in proposal_ids:
        proposal = by_id[proposal_id]
        source_ids = _proposal_source_ids(conn, proposal)
        if len(source_ids) != 1 or not source_ids <= selected_ids:
            raise ValueError(
                f"rename proposal {proposal_id} does not resolve to one selected node"
            )
        if proposal.proposed_name not in resulting_names:
            raise ValueError(
                f"merge does not preserve requested name for proposal {proposal_id}"
            )


def _append_merge_if_absent(entry: dict[str, Any]) -> bool:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = read_ledger()
            matches = [
                item
                for item in existing
                if isinstance(item, dict)
                and item.get("op") == "merge"
                and item.get("merge_id") == entry["merge_id"]
            ]
            if matches:
                if len(matches) == 1 and matches[0] == entry:
                    return False
                raise ValueError(f"merge id collision for {entry['merge_id']}")
            with path.open("a", encoding="utf-8") as stream:
                stream.write("---\n")
                stream.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def propose_merge(
    conn: sqlite3.Connection,
    survivor_id: str,
    victim_ids: list[str],
    canonical_name: str,
    proposed_by: str | None,
    path: Path | None = None,
) -> dict | None:
    """What an unconfirmed merge becomes: one cluster in the reviewer queue.
    Idempotent on the node set; returns the entry written, None if present."""
    path = path or (data_dir() / "merge-candidates-manual.json")
    try:
        existing = json.loads(path.read_text()) if path.exists() else []
    except (OSError, json.JSONDecodeError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    node_ids = sorted({survivor_id, *victim_ids})
    if any(
        set(c.get("node_ids") or []) == set(node_ids)
        for c in existing
        if isinstance(c, dict)
    ):
        return None
    survivor = _node(conn, survivor_id)
    entry = {
        "node_ids": node_ids,
        "suggested_survivor": survivor_id,
        "suggested_canonical": canonical_name,
        "score": 0.9,
        "node_type": survivor[1] if survivor else None,
        "reason": f"proposed by {proposed_by or 'an unnamed session'}; "
        "applies only when Mark confirms it in the workbench",
        "proposed_at": _now(),
    }
    existing.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=1, ensure_ascii=False))
    return entry


def append_undo_entry(
    merge_id: str, created_by: str | None, reason: str | None = None
) -> None:
    """Withdraw a merge from the durable curation set.

    This says "do not apply this merge going forward". It does NOT revert the
    live graph - `undo_merge` does that, separately - because the graph is
    derived: the next rebuild starts from the digests, so a merge withdrawn here
    simply never re-applies. The two are deliberately separable, for the case
    where a merge is correct today but must not outlive the corpus it was made
    against (a re-digest that rewrites the names the ledger keys on).

    `reason` is free text and is read by humans, not by replay.
    """
    entry = {"op": "undo", "merge_id": merge_id, "at": _now(), "by": created_by}
    if reason:
        entry["reason"] = reason
    _append(entry)


def _append(entry: dict) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


def read_ledger() -> list[dict]:
    path = ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def _resolve_natural(conn: sqlite3.Connection, nat: dict) -> str | None:
    """Resolve a node by its natural identity in the current graph: name then
    prior_names, within node_type. Returns the node id, or None if absent.

    NO FUZZY TIER. Everywhere else a fuzzy match is a reasonable guess that a
    later curator can correct; here it decides which nodes a REPLAYED HUMAN
    DECISION lands on, and a wrong guess silently applies a curator's merge to
    a node they never looked at. That is not a guess we are entitled to make on
    their behalf, and it is unreviewable after the fact - the ledger records the
    name, not what it resolved to. Replay is the one caller that must rather
    lose an op, loudly, than apply it to the wrong node: replay_ledger already
    counts and ERRORs on a lost op, so refusing to guess is visible where
    guessing wrong is not.

    Not hypothetical. The Nimitz node carried 118 aliases naming other events,
    so a Roswell op replayed through the fuzzy tier resolved onto Nimitz. That
    particular route is closed (see matching.collapse_acronym_expansions), but
    replay runs against a REBUILT graph whose node set differs from the one the
    curator saw, which is exactly when a near-miss is most likely.

    The deterministic tiers are kept. They resolve on declared evidence rather
    than on string distance - the acronym a name itself declares, a known
    given-name short form, punctuation - and dropping them costs real
    resolutions: measured over the ledger, exact-name-only loses five, "CIA"
    reaching Central Intelligence Agency (CIA) and "Dave Saunders" reaching
    David Saunders among them. Measured the other way, refusing fuzzy changes
    nothing that currently resolves: 179 identities resolve identically and none
    resolve to a different node.
    """
    node_type = nat.get("node_type")
    names = sorted(
        {name for name in [nat.get("name"), *(nat.get("prior_names") or [])] if name}
    )
    if not names:
        return None
    placeholders = ",".join("?" for _ in names)
    exact = conn.execute(
        "SELECT id FROM nodes WHERE node_type = ? AND retired_at IS NULL "
        f"AND name IN ({placeholders}) "
        "UNION SELECT n.id FROM aliases a JOIN nodes n ON n.id = a.node_id "
        "WHERE n.node_type = ? AND n.retired_at IS NULL "
        f"AND a.alias IN ({placeholders}) ORDER BY id",
        (node_type, *names, node_type, *names),
    ).fetchall()
    if len(exact) == 1:
        return exact[0][0]
    if exact:
        return None

    deterministic: set[str] = set()
    for name in names:
        matched = match_node(conn, name, node_type)
        if matched and matched[1] != "fuzzy":
            deterministic.add(matched[0])
    return next(iter(deterministic)) if len(deterministic) == 1 else None


def _operation_sort_key(entry: dict, id_field: str) -> tuple[str, str]:
    return str(entry.get("at") or ""), str(entry.get(id_field) or "")


ReplayPhase = Literal["merge", "rejection", "rename"]
ReplayOutcome = Literal[
    "applied_normally",
    "compensated_normally",
    "pending_normally",
    "rejected_normally",
    "absorbed",
    "unresolved_drift",
    "unconfirmed",
    "invalid",
]
_REPLAY_PHASES: tuple[ReplayPhase, ...] = ("merge", "rejection", "rename")


@dataclass(frozen=True)
class ReplayDiagnostic:
    """One canonical result for one source operation with a stable ID."""

    source_phase: ReplayPhase
    operation_id: str
    outcome: ReplayOutcome
    cause: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ReplayBlocker:
    """A source entry that cannot join the operation inventory without an ID."""

    source_phase: ReplayPhase
    cause: str
    details: dict[str, Any]


@dataclass(frozen=True)
class CandidateReplayResult:
    """Strict replay result over an isolated, writable rebuild candidate."""

    diagnostics: tuple[ReplayDiagnostic, ...]
    blockers: tuple[ReplayBlocker, ...]
    operation_inventory: tuple[tuple[ReplayPhase, str], ...]

    @property
    def replacement_safe(self) -> bool:
        return not self.blockers and all(
            diagnostic.outcome not in {"unresolved_drift", "unconfirmed", "invalid"}
            for diagnostic in self.diagnostics
        )


class CandidateReplaySourceError(ValueError):
    """The replay source cannot form an unambiguous operation inventory."""


@dataclass(frozen=True)
class _SourceOperation:
    phase: ReplayPhase
    operation_id: str
    entry: dict[str, Any]
    reversal: dict[str, Any] | None


RENAME_OPERATION_ID_PREFIX = OPERATION_ID_PREFIX


def legacy_rename_operation_identity(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the exact canonical payload and derived ID for one legacy rename."""
    return legacy_operation_identity(entry)


def _invalid_fields(entry: dict, requirements: dict[str, type]) -> list[str]:
    invalid = []
    for field, expected_type in requirements.items():
        value = entry.get(field)
        if not isinstance(value, expected_type) or (
            expected_type is str and not value.strip()
        ):
            invalid.append(field)
    return invalid


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _invalid_natural_paths(value: Any, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [path]
    invalid = []
    for field in ("name", "node_type"):
        member = value.get(field)
        if not isinstance(member, str) or not member.strip():
            invalid.append(f"{path}/{field}")
    prior_names = value.get("prior_names", [])
    if not isinstance(prior_names, list) or any(
        not isinstance(name, str) or not name.strip() for name in prior_names
    ):
        invalid.append(f"{path}/prior_names")
    return invalid


def _source_operations(
    phase: ReplayPhase,
    entries: list[Any],
    *,
    apply_op: str,
    reverse_op: str,
    id_field: str,
) -> tuple[list[_SourceOperation], list[ReplayBlocker]]:
    """Build the strict source inventory and validate compensating links."""
    ordered = sorted(
        enumerate(entries),
        key=lambda item: (
            str(item[1].get("at") or "") if isinstance(item[1], dict) else "",
            str(item[1].get(id_field) or "") if isinstance(item[1], dict) else "",
            item[0],
        ),
    )
    bases: dict[str, tuple[dict[str, Any], tuple[str, str]]] = {}
    reversals: dict[str, dict[str, Any]] = {}
    blockers: list[ReplayBlocker] = []
    for source_index, raw in ordered:
        entry = raw if isinstance(raw, dict) else {}
        operation_id = entry.get(id_field)
        if not isinstance(operation_id, str) or not operation_id.strip():
            blockers.append(
                ReplayBlocker(
                    source_phase=phase,
                    cause="missing_stable_operation_id",
                    details={
                        "source_index": source_index,
                        "operation": entry.get("op"),
                        "timestamp": entry.get("at"),
                        "entry_type": type(raw).__name__,
                    },
                )
            )
            continue
        operation = entry.get("op")
        order = _operation_sort_key(entry, id_field)
        if operation == reverse_op:
            if not _valid_timestamp(entry.get("at")):
                blockers.append(
                    ReplayBlocker(
                        source_phase=phase,
                        cause="invalid_reversal_timestamp",
                        details={"operation_id": operation_id},
                    )
                )
                continue
            base = bases.get(operation_id)
            if base is None:
                raise CandidateReplaySourceError(
                    f"{phase} reversal references unknown replay ID {operation_id!r}"
                )
            if operation_id in reversals:
                raise CandidateReplaySourceError(
                    f"duplicate {phase} reversal for replay ID {operation_id!r}"
                )
            if order <= base[1]:
                raise CandidateReplaySourceError(
                    f"{phase} reversal for replay ID {operation_id!r} is not later"
                )
            reversals[operation_id] = entry
            continue
        if operation_id in bases:
            raise CandidateReplaySourceError(
                f"duplicate {phase} replay ID {operation_id!r}"
            )
        bases[operation_id] = (entry, order)

    operations = [
        _SourceOperation(phase, operation_id, entry, reversals.get(operation_id))
        for operation_id, (entry, _order) in bases.items()
    ]
    operations.sort(key=lambda item: _operation_sort_key(item.entry, id_field))
    return operations, blockers


def _compensation_diagnostic(operation: _SourceOperation) -> ReplayDiagnostic:
    assert operation.reversal is not None
    return ReplayDiagnostic(
        source_phase=operation.phase,
        operation_id=operation.operation_id,
        outcome="compensated_normally",
        cause="explicit_later_reversal",
        details={
            "base_operation": {
                "operation": operation.entry.get("op"),
                "operation_id": operation.operation_id,
                "timestamp": operation.entry.get("at"),
            },
            "reversal_event": {
                "operation": operation.reversal.get("op"),
                "event_id": operation.reversal.get("id"),
                "reverses_operation_id": operation.operation_id,
                "timestamp": operation.reversal.get("at"),
            },
        },
    )


def _claim_ref_count(conn: sqlite3.Connection, node_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM claim_node_refs WHERE node_id = ?", (node_id,)
    ).fetchone()[0]


def _materialise_merge(
    conn: sqlite3.Connection, e: dict, log, *, strict_source: bool = True
) -> ReplayDiagnostic:
    merge_id = e["merge_id"]
    requirements = {"canonical_name": str, "survivor": dict, "victims": list}
    if strict_source:
        requirements["at"] = str
    invalid = _invalid_fields(e, requirements)
    field_paths = [f"/{field}" for field in invalid]
    if strict_source and "at" not in invalid and not _valid_timestamp(e["at"]):
        field_paths.append("/at")
    confirmation = e.get("confirmation")
    if confirmation is not None and not (
        isinstance(confirmation, dict)
        and set(confirmation) == {"by", "at", "via"}
        and isinstance(confirmation.get("by"), str)
        and confirmation["by"].strip()
        and _valid_timestamp(confirmation.get("at"))
        and confirmation.get("via") in CONFIRMATION_VIAS
    ):
        field_paths.append("/confirmation")
    proposal_ids = e.get("proposal_ids", [])
    if (
        not isinstance(proposal_ids, list)
        or proposal_ids != sorted(set(proposal_ids))
        or any(
            not isinstance(proposal_id, str)
            or not proposal_id.startswith(PROPOSAL_ID_PREFIX)
            or not proposal_id.removeprefix(PROPOSAL_ID_PREFIX)
            for proposal_id in proposal_ids
        )
    ):
        field_paths.append("/proposal_ids")
    if not invalid:
        field_paths.extend(_invalid_natural_paths(e["survivor"], "/survivor"))
        if not e["victims"]:
            field_paths.append("/victims")
        for index, victim in enumerate(e["victims"]):
            field_paths.extend(_invalid_natural_paths(victim, f"/victims/{index}"))
    if field_paths or e.get("op") != "merge":
        return ReplayDiagnostic(
            "merge",
            merge_id,
            "invalid",
            "invalid_merge_source_operation",
            {
                "field_paths": sorted(set(field_paths)) or ["/op"],
                "operation": e.get("op"),
            },
        )
    if not confirmed(e):
        log(
            f"  replay UNCONFIRMED {merge_id}: no confirmation block and "
            f"dated after the rule - not applied ({e.get('canonical_name')!r})"
        )
        return ReplayDiagnostic(
            "merge",
            merge_id,
            "unconfirmed",
            "confirmation_required",
            {"canonical_name": e["canonical_name"], "timestamp": e.get("at")},
        )

    identities = [e["survivor"], *e["victims"]]
    resolved_identities = [
        (identity, _resolve_natural(conn, identity)) for identity in identities
    ]
    survivor_id = resolved_identities[0][1]
    resolved = [survivor_id] if survivor_id else []
    for _identity, victim_id in resolved_identities[1:]:
        if victim_id and victim_id not in resolved:
            resolved.append(victim_id)
    unresolved_identities = [
        identity for identity, node_id in resolved_identities if node_id is None
    ]

    if not resolved:
        log(
            f"  ERROR replay LOST {merge_id}: no node of "
            f"'{e['survivor'].get('name')}' is in the graph - this curation "
            "decision is dropped"
        )
        return ReplayDiagnostic(
            "merge",
            merge_id,
            "unresolved_drift",
            "no_natural_identity_resolved",
            {"unresolved_identities": identities},
        )

    if len(resolved) == 1:
        current_name = conn.execute(
            "SELECT name FROM nodes WHERE id = ?", (resolved[0],)
        ).fetchone()[0]
        if current_name != e["canonical_name"]:
            log(
                f"  replay {merge_id}: single node remains as "
                f"{current_name!r}; the ledger's canonical "
                f"{e['canonical_name']!r} is NOT applied (naming is the "
                "renames ledger's job, not a merge's)"
            )
        details = {
            "resolved_node_id": resolved[0],
            "resolved_name": current_name,
            "requested_canonical_name": e["canonical_name"],
            "postcondition": "no_duplicate_nodes_remain",
        }
        if strict_source and unresolved_identities:
            details["unresolved_identities"] = unresolved_identities
            return ReplayDiagnostic(
                "merge",
                merge_id,
                "unresolved_drift",
                "partial_natural_identity_resolution",
                details,
            )
        return ReplayDiagnostic(
            "merge", merge_id, "absorbed", "single_resolved_node", details
        )

    if survivor_id is None:
        survivor_id = max(resolved, key=lambda node_id: _claim_ref_count(conn, node_id))
        kept = conn.execute(
            "SELECT name, node_type FROM nodes WHERE id = ?", (survivor_id,)
        ).fetchone()
        log(
            f"  replay {merge_id}: chosen survivor "
            f"'{e['survivor'].get('name')}' ({e['survivor'].get('node_type')}) "
            "is no longer in the graph - merging into the most-cited "
            f"resolved node instead, {kept[0]!r} ({kept[1]})"
        )
    victim_ids = [node_id for node_id in resolved if node_id != survivor_id]
    merged = merge_nodes(
        conn,
        survivor_id,
        victim_ids,
        e["canonical_name"],
        merge_id,
        created_at=e.get("at"),
        created_by=e.get("by"),
    )
    retired = {
        row[0]: row[1] is not None
        for row in conn.execute(
            f"SELECT id, retired_at FROM nodes WHERE id IN ({','.join('?' for _ in victim_ids)})",
            victim_ids,
        )
    }
    current_name = conn.execute(
        "SELECT name FROM nodes WHERE id = ?", (survivor_id,)
    ).fetchone()[0]
    postcondition = {
        "survivor_id": survivor_id,
        "canonical_name": current_name,
        "victim_ids": victim_ids,
        "victims_retired": retired,
        "merged_victim_count": merged,
    }
    exact = (
        current_name == e["canonical_name"]
        and merged == len(victim_ids)
        and retired == {victim_id: True for victim_id in victim_ids}
    )
    if strict_source and unresolved_identities:
        return ReplayDiagnostic(
            "merge",
            merge_id,
            "unresolved_drift",
            "partial_natural_identity_resolution",
            {
                "resolved_node_ids": resolved,
                "unresolved_identities": unresolved_identities,
                "postcondition": postcondition,
            },
        )
    return ReplayDiagnostic(
        "merge",
        merge_id,
        "applied_normally" if exact else "unresolved_drift",
        "merge_materialised" if exact else "merge_postcondition_mismatch",
        {
            "resolved_node_ids": resolved,
            "postcondition": postcondition,
        },
    )


def _replay_merge_operations(
    conn: sqlite3.Connection,
    operations: list[_SourceOperation],
    log,
    *,
    strict_source: bool = True,
) -> list[ReplayDiagnostic]:
    return [
        _compensation_diagnostic(operation)
        if operation.reversal is not None
        else _materialise_merge(conn, operation.entry, log, strict_source=strict_source)
        for operation in operations
    ]


def replay_ledger(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-apply the durable ledger over the freshly-imported graph. Merges whose
    undo entry is present are skipped. Keyed on natural identity.

    THE SURVIVOR MOVING DOES NOT VOID THE OP. Every curation merge in this corpus
    is cross-type - the same name under two types, where the curator picked which
    type to keep. A rebuild re-imports whatever type the digester emits now, so the
    curator's chosen survivor is routinely absent while its victims are present.
    Requiring the survivor to resolve therefore discarded the op and left the
    duplicates standing (it dropped 15 of 46 ops, in silence). The merge now
    follows the nodes that ARE present, taking the most-cited of them as survivor.

    Three outcomes, all counted and all reported, because a rebuild that loses a
    human decision must say so:

    - applied  - two or more of the op's nodes are in the graph; they were merged.
    - absorbed - only one is, so there is no duplicate left to collapse and the op
      is already satisfied. Expected as reclassification lands, not a failure.
    - lost     - none of them resolve. The decision is unrecoverable from this
      graph and is logged as an ERROR.
    """
    log = on_progress or (lambda _: None)
    entries = read_ledger()
    undone = {
        e["merge_id"]
        for e in entries
        if isinstance(e, dict) and e.get("op") == "undo" and "merge_id" in e
    }
    operations = [
        _SourceOperation(
            "merge",
            e["merge_id"],
            e,
            {"op": "undo", "merge_id": e["merge_id"]}
            if e["merge_id"] in undone
            else None,
        )
        for e in sorted(
            (entry for entry in entries if isinstance(entry, dict)),
            key=lambda entry: _operation_sort_key(entry, "merge_id"),
        )
        if e.get("op") == "merge" and "merge_id" in e
    ]
    diagnostics = _replay_merge_operations(conn, operations, log, strict_source=False)
    applied = sum(d.outcome == "applied_normally" for d in diagnostics)
    absorbed = sum(d.outcome == "absorbed" for d in diagnostics)
    lost = sum(d.outcome in {"unresolved_drift", "invalid"} for d in diagnostics)
    unconfirmed = sum(d.outcome == "unconfirmed" for d in diagnostics)
    summary = f"Replayed {applied} merges ({absorbed} already single-node"
    summary += f", {lost} LOST" if lost else ""
    summary += f", {unconfirmed} unconfirmed" if unconfirmed else ""
    log(summary + ")")
    return {
        "applied": applied,
        "absorbed": absorbed,
        "lost": lost,
        "unconfirmed": unconfirmed,
    }


# --- Rejections ("not a duplicate"): the negative-signal curation ledger ---


def rejections_ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "rejections.yaml"


def _append_rejection(entry: dict) -> None:
    path = rejections_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


def read_rejections() -> list[dict]:
    path = rejections_ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def reject_nodes(
    conn: sqlite3.Connection,
    node_ids: list[str],
    reason: str | None,
    rejection_id: str,
    created_at: str | None = None,
    created_by: str | None = None,
) -> None:
    """Record a confirmed-distinct decision for a candidate cluster: durable
    ledger entry (natural-identity keyed) + derived node_rejections row. propose-
    merges then excludes this set so the pair stops reappearing."""
    created_at = created_at or _now()
    _append_rejection(
        {
            "op": "reject",
            "rejection_id": rejection_id,
            "at": created_at,
            "by": created_by,
            "reason": reason,
            "nodes": [_natural(conn, n) for n in node_ids if _node(conn, n)],
            "audit": {"node_ids": list(node_ids)},
        }
    )
    for node_id in sorted(set(node_ids)):
        conn.execute(
            "INSERT OR REPLACE INTO node_rejections (rejection_id, node_id, reason, "
            "created_at, created_by, undone_at) VALUES (?, ?, ?, ?, ?, NULL)",
            (rejection_id, node_id, reason, created_at, created_by),
        )
    conn.commit()


def un_reject(conn: sqlite3.Connection, rejection_id: str) -> int:
    _append_rejection(
        {"op": "unreject", "rejection_id": rejection_id, "at": _now(), "by": None}
    )
    cur = conn.execute(
        "UPDATE node_rejections SET undone_at = ? WHERE rejection_id = ? AND undone_at IS NULL",
        (_now(), rejection_id),
    )
    conn.commit()
    return cur.rowcount


def _materialise_rejection(
    conn: sqlite3.Connection, e: dict, log, *, strict_source: bool = True
) -> ReplayDiagnostic:
    rejection_id = e["rejection_id"]
    requirements = {"nodes": list}
    if strict_source:
        requirements["at"] = str
    invalid = _invalid_fields(e, requirements)
    field_paths = [f"/{field}" for field in invalid]
    if strict_source and "at" not in invalid and not _valid_timestamp(e["at"]):
        field_paths.append("/at")
    if not invalid:
        if len(e["nodes"]) < 2:
            field_paths.append("/nodes")
        for index, node in enumerate(e["nodes"]):
            field_paths.extend(_invalid_natural_paths(node, f"/nodes/{index}"))
    if field_paths or e.get("op") != "reject":
        return ReplayDiagnostic(
            "rejection",
            rejection_id,
            "invalid",
            "invalid_rejection_source_operation",
            {
                "field_paths": sorted(set(field_paths)) or ["/op"],
                "operation": e.get("op"),
            },
        )
    nodes = e["nodes"]
    resolved_identities = [(node, _resolve_natural(conn, node)) for node in nodes]
    ids = {node_id for _node, node_id in resolved_identities if node_id is not None}
    unresolved_identities = [
        node for node, node_id in resolved_identities if node_id is None
    ]
    if not ids:
        log(
            f"  ERROR replay rejection LOST {rejection_id}: no natural node "
            "is in the graph - this curation decision is dropped"
        )
        return ReplayDiagnostic(
            "rejection",
            rejection_id,
            "unresolved_drift",
            "no_natural_identity_resolved",
            {"unresolved_identities": nodes},
        )
    if len(ids) == 1:
        node_id = next(iter(ids))
        name = conn.execute(
            "SELECT name FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
        log(f"  replay rejection {rejection_id}: absorbed; only {name!r} remains")
        details = {
            "resolved_node_id": node_id,
            "resolved_name": name,
            "postcondition": "no_distinct_pair_remains",
        }
        if strict_source and unresolved_identities:
            details["unresolved_identities"] = unresolved_identities
            return ReplayDiagnostic(
                "rejection",
                rejection_id,
                "unresolved_drift",
                "partial_natural_identity_resolution",
                details,
            )
        return ReplayDiagnostic(
            "rejection", rejection_id, "absorbed", "single_resolved_node", details
        )
    for node_id in sorted(ids):
        conn.execute(
            "INSERT OR REPLACE INTO node_rejections (rejection_id, node_id, reason, "
            "created_at, created_by, undone_at) VALUES (?, ?, ?, ?, ?, NULL)",
            (rejection_id, node_id, e.get("reason"), e.get("at"), e.get("by")),
        )
    materialised_ids = [
        row[0]
        for row in conn.execute(
            "SELECT node_id FROM node_rejections WHERE rejection_id = ? "
            "AND undone_at IS NULL ORDER BY node_id",
            (rejection_id,),
        )
    ]
    exact = materialised_ids == sorted(ids) and len(materialised_ids) >= 2
    if strict_source and unresolved_identities:
        return ReplayDiagnostic(
            "rejection",
            rejection_id,
            "unresolved_drift",
            "partial_natural_identity_resolution",
            {
                "resolved_node_ids": sorted(ids),
                "unresolved_identities": unresolved_identities,
                "postcondition": {
                    "rejection_id": rejection_id,
                    "distinct_node_ids": materialised_ids,
                },
            },
        )
    return ReplayDiagnostic(
        "rejection",
        rejection_id,
        "applied_normally" if exact else "unresolved_drift",
        "rejection_materialised" if exact else "rejection_postcondition_mismatch",
        {
            "resolved_node_ids": sorted(ids),
            "postcondition": {
                "rejection_id": rejection_id,
                "distinct_node_ids": materialised_ids,
            },
        },
    )


def _replay_rejection_operations(
    conn: sqlite3.Connection,
    operations: list[_SourceOperation],
    log,
    *,
    strict_source: bool = True,
) -> list[ReplayDiagnostic]:
    conn.execute("DELETE FROM node_rejections")
    diagnostics = [
        _compensation_diagnostic(operation)
        if operation.reversal is not None
        else _materialise_rejection(
            conn, operation.entry, log, strict_source=strict_source
        )
        for operation in operations
    ]
    conn.commit()
    return diagnostics


def replay_rejections(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-materialise active rejection IDs from their natural identities.

    The table is wholly derived, so replay clears it first. Each rejection ID has
    one diagnostic outcome: two or more distinct nodes are applied, one is
    absorbed by graph contraction, and none is loudly lost.
    """
    log = on_progress or (lambda _: None)
    entries = read_rejections()
    active: dict[str, dict] = {}
    for e in sorted(
        entries, key=lambda entry: _operation_sort_key(entry, "rejection_id")
    ):
        rejection_id = e.get("rejection_id")
        if not rejection_id:
            continue
        if e.get("op") == "reject":
            active[rejection_id] = e
        elif e.get("op") == "unreject":
            active.pop(rejection_id, None)

    operations = [
        _SourceOperation("rejection", rejection_id, entry, None)
        for rejection_id, entry in sorted(active.items())
    ]
    diagnostics = _replay_rejection_operations(
        conn, operations, log, strict_source=False
    )
    applied = sum(d.outcome == "applied_normally" for d in diagnostics)
    absorbed = sum(d.outcome == "absorbed" for d in diagnostics)
    lost = sum(d.outcome in {"unresolved_drift", "invalid"} for d in diagnostics)
    summary = f"Replayed {applied} rejections ({absorbed} absorbed"
    summary += f", {lost} LOST" if lost else ""
    log(summary + ")")
    return {"applied": applied, "absorbed": absorbed, "lost": lost}


def rejected_sets(conn: sqlite3.Connection) -> set[frozenset]:
    """Active rejected node-id sets (grouped by rejection_id), for propose-merges
    to exclude the corresponding edges."""
    by_rej: dict[str, set[str]] = {}
    try:
        rows = conn.execute(
            "SELECT rejection_id, node_id FROM node_rejections WHERE undone_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    for rejection_id, node_id in rows:
        by_rej.setdefault(rejection_id, set()).add(node_id)
    return {frozenset(s) for s in by_rej.values()}


# --- Renames: durable node-name corrections (e.g. acronym standardisation) ---


def renames_ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "renames.yaml"


def read_renames() -> list[dict]:
    path = renames_ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def append_rename_entry(
    old_natural: dict,
    new_name: str,
    rename_id: str,
    created_at: str,
    created_by: str | None,
    proposal_id: str | None = None,
) -> str:
    """Record a node-name correction in the durable ledger, keyed on the node's
    PRE-rename natural identity (the name a fresh import carries) so replay can
    resolve it. Used directly when the live rename has already been applied."""
    del rename_id
    event = build_rename_event(
        at=created_at,
        by=created_by,
        new_name=new_name,
        node=old_natural,
        proposal_id=proposal_id,
    )
    append_event_if_absent(renames_ledger_path(), event)
    return event["operation_id"]


def rename_node(
    conn: sqlite3.Connection,
    node_id: str,
    new_name: str,
    rename_id: str,
    created_at: str | None = None,
    created_by: str | None = None,
    proposal_id: str | None = None,
) -> None:
    """Apply a node-name correction to the live graph (old name kept as an alias)
    and record it in the durable ledger, keyed on the pre-rename natural identity.
    Captured BEFORE the live change, so the entry resolves on a fresh import."""
    created_at = created_at or _now()
    cur = _node(conn, node_id)
    if cur is None:
        raise ValueError(f"node not found: {node_id}")
    old_name = cur[0]
    append_rename_entry(
        _natural(conn, node_id),
        new_name,
        rename_id,
        created_at,
        created_by,
        proposal_id=proposal_id,
    )
    conn.execute("UPDATE nodes SET name = ? WHERE id = ?", (new_name, node_id))
    forget_embeddings(conn, "node", node_id)
    conn.execute(
        "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
        (old_name, node_id),
    )
    conn.commit()


def _materialise_rename(
    conn: sqlite3.Connection, e: dict, operation_id: str | None = None
) -> ReplayDiagnostic:
    rename_id = operation_id or e["rename_id"]
    if e.get("op") == "reject_proposal":
        return ReplayDiagnostic(
            "rename",
            rename_id,
            "applied_normally",
            "proposal_rejection_recorded",
            {
                "proposal_id": e["proposal_id"],
                "postcondition": "explicit_rejection_event_is_active",
            },
        )
    invalid = _invalid_fields(e, {"node": dict, "new_name": str})
    field_paths = [f"/{field}" for field in invalid]
    if not invalid:
        field_paths.extend(_invalid_natural_paths(e["node"], "/node"))
    if field_paths or e.get("op") != "rename":
        return ReplayDiagnostic(
            "rename",
            rename_id,
            "invalid",
            "invalid_rename_source_operation",
            {
                "field_paths": sorted(set(field_paths)) or ["/op"],
                "operation": e.get("op"),
            },
        )
    node_id = _resolve_natural(conn, e["node"])
    if node_id is None:
        return ReplayDiagnostic(
            "rename",
            rename_id,
            "unresolved_drift",
            "natural_identity_did_not_resolve",
            {"unresolved_identity": e["node"], "requested_name": e["new_name"]},
        )
    old_name = _node(conn, node_id)[0]
    conn.execute("UPDATE nodes SET name = ? WHERE id = ?", (e["new_name"], node_id))
    forget_embeddings(conn, "node", node_id)
    conn.execute(
        "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
        (old_name, node_id),
    )
    current_name = _node(conn, node_id)[0]
    old_name_is_alias = (
        conn.execute(
            "SELECT 1 FROM aliases WHERE alias = ? AND node_id = ?",
            (old_name, node_id),
        ).fetchone()
        is not None
    )
    exact = current_name == e["new_name"] and old_name_is_alias
    return ReplayDiagnostic(
        "rename",
        rename_id,
        "applied_normally" if exact else "unresolved_drift",
        "rename_materialised" if exact else "rename_postcondition_mismatch",
        {
            "resolved_node_id": node_id,
            "postcondition": {
                "canonical_name": current_name,
                "prior_name": old_name,
                "prior_name_is_alias": old_name_is_alias,
            },
        },
    )


def _replay_rename_operations(
    conn: sqlite3.Connection,
    operations: list[_SourceOperation],
) -> list[ReplayDiagnostic]:
    diagnostics = [
        _compensation_diagnostic(operation)
        if operation.reversal is not None
        else _materialise_rename(conn, operation.entry, operation.operation_id)
        for operation in operations
    ]
    conn.commit()
    return diagnostics


def _parse_proposals(
    raw_proposals: list[Any],
) -> tuple[list[RenameProposal], list[ReplayDiagnostic], list[ReplayBlocker]]:
    proposals: list[RenameProposal] = []
    diagnostics: list[ReplayDiagnostic] = []
    blockers: list[ReplayBlocker] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_proposals):
        document = raw if isinstance(raw, dict) else {}
        path = str(document.get("_path") or "<rename proposal>")
        source = {
            key: value for key, value in document.items() if not key.startswith("_")
        }
        raw_id = source.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            blockers.append(
                ReplayBlocker(
                    "rename",
                    "malformed_rename_proposal_without_id",
                    {
                        "source_index": index,
                        "path": path,
                        "detail": document.get("_error"),
                    },
                )
            )
            continue
        operation_id = PROPOSAL_ID_PREFIX + raw_id
        if operation_id in seen:
            raise CandidateReplaySourceError(
                f"duplicate rename proposal operation id {operation_id}"
            )
        seen.add(operation_id)
        try:
            if document.get("_error"):
                raise RenameLedgerError(str(document["_error"]))
            proposals.append(parse_proposal_document(source, path))
        except RenameLedgerError as exc:
            diagnostics.append(
                ReplayDiagnostic(
                    "rename",
                    operation_id,
                    "invalid",
                    "malformed_rename_proposal",
                    {
                        "path": path,
                        "detail": str(exc),
                        "source_timestamp": source.get("proposed_at"),
                    },
                )
            )
    proposals.sort(key=lambda proposal: (proposal.at, proposal.operation_id))
    return proposals, diagnostics, blockers


def _proposal_source_ids(
    conn: sqlite3.Connection, proposal: RenameProposal
) -> set[str]:
    if proposal.node is not None:
        resolved = _resolve_natural(conn, proposal.node)
        return {resolved} if resolved else set()
    return _live_ids_for_name(conn, proposal.source_name)


def _linked_proposal_diagnostics(
    conn: sqlite3.Connection,
    raw_proposals: list[Any],
    rename_stream: RenameStream,
    merge_operations: list[_SourceOperation],
) -> tuple[list[ReplayDiagnostic], list[ReplayBlocker]]:
    proposals, malformed, blockers = _parse_proposals(raw_proposals)
    known = {proposal.operation_id for proposal in proposals}
    for operation in rename_stream.operations:
        proposal_id = operation.event.get("proposal_id")
        if proposal_id is not None and proposal_id not in known:
            blockers.append(
                ReplayBlocker(
                    "rename",
                    "rename_links_unknown_proposal",
                    {
                        "operation_id": operation.operation_id,
                        "proposal_id": proposal_id,
                    },
                )
            )
    for operation in merge_operations:
        for proposal_id in operation.entry.get("proposal_ids") or []:
            if proposal_id not in known:
                blockers.append(
                    ReplayBlocker(
                        "merge",
                        "merge_links_unknown_proposal",
                        {
                            "operation_id": operation.operation_id,
                            "proposal_id": proposal_id,
                        },
                    )
                )

    diagnostics = list(malformed)
    active_rename = {
        operation.operation_id: operation
        for operation in rename_stream.active_operations
    }
    for proposal in proposals:
        rename_routes = [
            operation
            for operation in rename_stream.operations
            if operation.op == "rename"
            and operation.event.get("proposal_id") == proposal.operation_id
        ]
        reject_routes = [
            operation
            for operation in rename_stream.active_operations
            if operation.op == "reject_proposal"
            and operation.event["proposal_id"] == proposal.operation_id
        ]
        merge_routes = [
            operation
            for operation in merge_operations
            if operation.reversal is None
            and proposal.operation_id in (operation.entry.get("proposal_ids") or [])
        ]
        active_rename_routes = [
            route for route in rename_routes if route.operation_id in active_rename
        ]
        compensated_routes = [
            route
            for route in rename_routes
            if route.operation_id in rename_stream.compensation_by_operation
        ]
        routes = (
            active_rename_routes + reject_routes + merge_routes + compensated_routes
        )
        details: dict[str, Any] = {
            "proposal": proposal,
            "source_timestamp": proposal.at,
            "resolution_phase": None,
            "resolution_operation_id": None,
            "compensation_operation_id": None,
            "resolved_at": None,
            "resolution_note": None,
            "materialized_status": None,
        }
        source_ids = _proposal_source_ids(conn, proposal)
        if len(routes) > 1:
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "invalid",
                "proposal_has_multiple_active_resolution_routes",
                {
                    **details,
                    "route_operation_ids": [route.operation_id for route in routes],
                },
            )
        elif active_rename_routes:
            route = active_rename_routes[0]
            target_ids = _live_ids_for_name(conn, proposal.proposed_name)
            exact = len(source_ids) == 1 and target_ids == source_ids
            node_id = next(iter(source_ids)) if exact else None
            current = _node(conn, node_id) if node_id else None
            exact = bool(exact and current and current[0] == proposal.proposed_name)
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "applied_normally" if exact else "unresolved_drift",
                "proposal_applied_by_explicit_rename"
                if exact
                else "linked_rename_postcondition_mismatch",
                {
                    **details,
                    "materialized_status": "applied" if exact else "unresolved_drift",
                    "resolution_phase": "rename",
                    "resolution_operation_id": route.operation_id,
                    "resolved_at": route.at,
                },
            )
        elif reject_routes:
            route = reject_routes[0]
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "rejected_normally",
                "proposal_rejected_by_explicit_event",
                {
                    **details,
                    "materialized_status": "rejected",
                    "resolution_phase": "rename",
                    "resolution_operation_id": route.operation_id,
                    "resolved_at": route.at,
                    "resolution_note": route.event["reason"],
                },
            )
        elif merge_routes:
            route = merge_routes[0]
            exact = confirmed(route.entry) and len(source_ids) == 1
            node_id = next(iter(source_ids)) if exact else None
            names = (
                {_node(conn, node_id)[0], *_aliases(conn, node_id)}
                if node_id
                else set()
            )
            exact = bool(exact and proposal.proposed_name in names)
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "applied_normally" if exact else "unresolved_drift",
                "proposal_resolved_by_confirmed_merge"
                if exact
                else "linked_merge_postcondition_mismatch",
                {
                    **details,
                    "materialized_status": "merged" if exact else "unresolved_drift",
                    "resolution_phase": "merge",
                    "resolution_operation_id": route.operation_id,
                    "resolved_at": route.entry.get("at"),
                },
            )
        elif compensated_routes:
            route = compensated_routes[0]
            compensation = rename_stream.compensation_by_operation[route.operation_id]
            exact = len(source_ids) == 1
            node_id = next(iter(source_ids)) if exact else None
            current = _node(conn, node_id) if node_id else None
            expected_type = route.event["node"]["node_type"]
            exact = bool(exact and current == (proposal.source_name, expected_type))
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "compensated_normally" if exact else "unresolved_drift",
                "proposal_resolution_compensated"
                if exact
                else "compensation_postcondition_mismatch",
                {
                    **details,
                    "materialized_status": "compensated"
                    if exact
                    else "unresolved_drift",
                    "resolution_phase": "rename",
                    "resolution_operation_id": route.operation_id,
                    "compensation_operation_id": compensation.get("id"),
                    "resolved_at": compensation.get("at"),
                },
            )
        elif len(source_ids) == 1:
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "pending_normally",
                "proposal_has_no_explicit_resolution",
                {**details, "materialized_status": "pending"},
            )
        else:
            diagnostic = ReplayDiagnostic(
                "rename",
                proposal.operation_id,
                "unresolved_drift" if not source_ids else "invalid",
                "proposal_source_absent"
                if not source_ids
                else "proposal_source_ambiguous",
                {**details, "materialized_status": "unresolved_drift"},
            )
        diagnostics.append(diagnostic)
    diagnostics.sort(
        key=lambda diagnostic: (
            str(diagnostic.details.get("source_timestamp") or ""),
            diagnostic.operation_id,
        )
    )
    return diagnostics, blockers


def replay_renames(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-apply durable renames over the freshly-rebuilt graph, after merges and
    rejections (a renamed node may be a merge survivor). Resolves each node by its
    pre-rename natural identity, sets the new name, keeps the old name as an alias.
    A node that no longer resolves is skipped (its source left the corpus)."""
    log = on_progress or (lambda _: None)
    try:
        stream = read_stream(renames_ledger_path())
    except RenameLedgerError as exc:
        raise CandidateReplaySourceError(str(exc)) from exc
    operations = [
        _SourceOperation(
            "rename",
            operation.operation_id,
            operation.event,
            stream.compensation_by_operation.get(operation.operation_id),
        )
        for operation in stream.operations
    ]
    diagnostics = _replay_rename_operations(conn, operations)
    applied = sum(d.cause == "rename_materialised" for d in diagnostics)
    skipped = sum(d.outcome in {"unresolved_drift", "invalid"} for d in diagnostics)
    log(f"Replayed {applied} renames ({skipped} skipped)")
    return {"applied": applied, "skipped": skipped}


def candidate_curation_source_inventory() -> tuple[
    tuple[tuple[ReplayPhase, str], ...], tuple[ReplayBlocker, ...]
]:
    """Read the canonical source inventory without consulting a replay report."""
    specifications: tuple[tuple[ReplayPhase, list[Any], str, str, str], ...] = (
        ("merge", read_ledger(), "merge", "undo", "merge_id"),
        ("rejection", read_rejections(), "reject", "unreject", "rejection_id"),
    )
    inventory: list[tuple[ReplayPhase, str]] = []
    blockers: list[ReplayBlocker] = []
    for phase, entries, apply_op, reverse_op, id_field in specifications:
        operations, phase_blockers = _source_operations(
            phase,
            entries,
            apply_op=apply_op,
            reverse_op=reverse_op,
            id_field=id_field,
        )
        inventory.extend((phase, operation.operation_id) for operation in operations)
        blockers.extend(phase_blockers)
    try:
        rename_stream = read_stream(renames_ledger_path())
    except RenameLedgerError as exc:
        raise CandidateReplaySourceError(str(exc)) from exc
    proposals, malformed, proposal_blockers = _parse_proposals(read_rename_proposals())
    blockers.extend(proposal_blockers)
    rename_items = [
        ((operation.at, operation.operation_id), operation.operation_id)
        for operation in rename_stream.operations
    ]
    rename_items.extend(
        ((proposal.at, proposal.operation_id), proposal.operation_id)
        for proposal in proposals
    )
    rename_items.extend(
        (
            (
                str(diagnostic.details.get("source_timestamp") or ""),
                diagnostic.operation_id,
            ),
            diagnostic.operation_id,
        )
        for diagnostic in malformed
    )
    rename_items.sort(key=lambda item: item[0])
    inventory.extend(("rename", operation_id) for _order, operation_id in rename_items)
    if len(set(inventory)) != len(inventory):
        raise CandidateReplaySourceError(
            "curation source inventory contains duplicate ids"
        )
    return tuple(inventory), tuple(blockers)


def replay_candidate_curation(
    conn: sqlite3.Connection, on_progress=None
) -> CandidateReplayResult:
    """Strictly replay merge, rejection and rename onto a writable candidate.

    The caller owns candidate isolation. Rename proposal files join the rename
    phase as non-mutating base requests after ledger renames are materialised.
    Unknown or duplicate stable IDs fail immediately. Entries without a stable ID
    become explicit blockers because they cannot be included in the inventory.
    """
    log = on_progress or (lambda _: None)
    try:
        conn.execute("UPDATE nodes SET id = id WHERE 0")
    except sqlite3.OperationalError as exc:
        raise CandidateReplaySourceError(
            "candidate replay requires a writable SQLite connection"
        ) from exc

    specifications: tuple[tuple[ReplayPhase, list[Any], str, str, str], ...] = (
        ("merge", read_ledger(), "merge", "undo", "merge_id"),
        (
            "rejection",
            read_rejections(),
            "reject",
            "unreject",
            "rejection_id",
        ),
    )
    by_phase: dict[ReplayPhase, list[_SourceOperation]] = {}
    blockers: list[ReplayBlocker] = []
    for phase, entries, apply_op, reverse_op, id_field in specifications:
        operations, phase_blockers = _source_operations(
            phase,
            entries,
            apply_op=apply_op,
            reverse_op=reverse_op,
            id_field=id_field,
        )
        by_phase[phase] = operations
        blockers.extend(phase_blockers)
    try:
        rename_stream = read_stream(renames_ledger_path())
    except RenameLedgerError as exc:
        raise CandidateReplaySourceError(str(exc)) from exc
    rename_operations = [
        _SourceOperation(
            "rename",
            operation.operation_id,
            operation.event,
            rename_stream.compensation_by_operation.get(operation.operation_id),
        )
        for operation in rename_stream.operations
    ]
    by_phase["rename"] = rename_operations
    proposals = read_rename_proposals()

    diagnostics: list[ReplayDiagnostic] = []
    diagnostics.extend(_replay_merge_operations(conn, by_phase["merge"], log))
    diagnostics.extend(_replay_rejection_operations(conn, by_phase["rejection"], log))
    rename_diagnostics = _replay_rename_operations(conn, rename_operations)
    proposal_diagnostics, proposal_blockers = _linked_proposal_diagnostics(
        conn, proposals, rename_stream, by_phase["merge"]
    )
    blockers.extend(proposal_blockers)
    ordered_rename_diagnostics = [
        (
            (str(operation.entry.get("at") or ""), operation.operation_id),
            diagnostic,
        )
        for operation, diagnostic in zip(
            rename_operations, rename_diagnostics, strict=True
        )
    ]
    ordered_rename_diagnostics.extend(
        (
            (
                str(diagnostic.details.get("source_timestamp") or ""),
                diagnostic.operation_id,
            ),
            diagnostic,
        )
        for diagnostic in proposal_diagnostics
    )
    ordered_rename_diagnostics.sort(key=lambda item: item[0])
    diagnostics.extend(diagnostic for _key, diagnostic in ordered_rename_diagnostics)

    non_rename_inventory = tuple(
        (phase, operation.operation_id)
        for phase in _REPLAY_PHASES[:2]
        for operation in by_phase[phase]
    )
    rename_inventory = tuple(
        ("rename", diagnostic.operation_id)
        for _key, diagnostic in ordered_rename_diagnostics
    )
    inventory = non_rename_inventory + rename_inventory
    actual = tuple(
        (diagnostic.source_phase, diagnostic.operation_id) for diagnostic in diagnostics
    )
    if actual != inventory or len(set(actual)) != len(actual):
        raise RuntimeError(
            "candidate replay diagnostics do not exactly cover the source inventory"
        )
    return CandidateReplayResult(tuple(diagnostics), tuple(blockers), inventory)


def _live_ids_for_name(conn: sqlite3.Connection, name: str) -> set[str]:
    """Live nodes carrying an exact canonical name or alias."""
    return {
        row[0]
        for row in conn.execute(
            "SELECT id FROM nodes WHERE name = ? AND retired_at IS NULL "
            "UNION SELECT n.id FROM aliases a JOIN nodes n ON n.id = a.node_id "
            "WHERE a.alias = ? AND n.retired_at IS NULL",
            (name, name),
        ).fetchall()
    }


def _apply_proposal_dispositions(
    diagnostics: list[ReplayDiagnostic], disposition_report: dict[str, Any] | None
) -> list[ReplayDiagnostic]:
    if disposition_report is None:
        return diagnostics
    dispositions = {
        outcome["operation_id"]: outcome
        for outcome in disposition_report.get("outcomes", [])
        if outcome.get("source_phase") == "rename"
        and str(outcome.get("operation_id", "")).startswith(PROPOSAL_ID_PREFIX)
    }
    updated: list[ReplayDiagnostic] = []
    for diagnostic in diagnostics:
        disposition = dispositions.get(diagnostic.operation_id)
        if disposition is None:
            updated.append(diagnostic)
            continue
        details = {**diagnostic.details, "replay_disposition_id": disposition["id"]}
        outcome = disposition["outcome"]
        evidence = disposition["evidence"]
        if diagnostic.outcome in {
            "applied_normally",
            "compensated_normally",
            "pending_normally",
            "rejected_normally",
        }:
            updated.append(
                ReplayDiagnostic(
                    diagnostic.source_phase,
                    diagnostic.operation_id,
                    diagnostic.outcome,
                    diagnostic.cause,
                    details,
                )
            )
            continue
        if outcome == "superseded":
            reference = evidence["superseded_by"]
            status = "merged" if reference["source_phase"] == "merge" else "applied"
            details.update(
                {
                    "resolution_phase": reference["source_phase"],
                    "resolution_operation_id": reference["operation_id"],
                }
            )
        elif outcome in {"applied", "absorbed"}:
            status = "applied"
        elif outcome == "compensated":
            status = "compensated"
            applied = evidence.get("applied_by")
            compensated = evidence.get("compensated_by")
            if applied:
                details["resolution_phase"] = applied["source_phase"]
                details["resolution_operation_id"] = applied["operation_id"]
            if compensated:
                details["compensation_operation_id"] = compensated["operation_id"]
        elif outcome == "contraction_drop":
            status = "contraction_drop"
        elif outcome == "unresolved_drift":
            status = "unresolved_drift"
        else:
            status = "invalid"
        details.update(
            {
                "materialized_status": status,
                "resolved_at": disposition["classified_at"],
                "resolution_note": f"validated replay disposition: {outcome}",
            }
        )
        updated.append(
            ReplayDiagnostic(
                diagnostic.source_phase,
                diagnostic.operation_id,
                outcome,
                "proposal_outcome_from_validated_disposition",
                details,
            )
        )
    return updated


def replay_rename_proposals(
    conn: sqlite3.Connection,
    on_progress=None,
    disposition_report: dict[str, Any] | None = None,
) -> dict:
    """Reconstruct proposal rows from canonical explicit operation links."""
    log = on_progress or (lambda _: None)
    try:
        rename_stream = read_stream(renames_ledger_path())
    except RenameLedgerError as exc:
        raise CandidateReplaySourceError(str(exc)) from exc
    merge_entries = read_ledger()
    undone = {
        entry.get("merge_id")
        for entry in merge_entries
        if isinstance(entry, dict) and entry.get("op") == "undo"
    }
    merge_operations = [
        _SourceOperation(
            "merge",
            entry["merge_id"],
            entry,
            {"op": "undo"} if entry["merge_id"] in undone else None,
        )
        for entry in merge_entries
        if isinstance(entry, dict)
        and entry.get("op") == "merge"
        and isinstance(entry.get("merge_id"), str)
    ]
    diagnostics, blockers = _linked_proposal_diagnostics(
        conn, read_rename_proposals(), rename_stream, merge_operations
    )
    diagnostics = _apply_proposal_dispositions(diagnostics, disposition_report)
    conn.execute("DELETE FROM rename_proposals")
    counts = {"applied": 0, "rejected": 0, "pending": 0, "lost": 0, "malformed": 0}
    for diagnostic in diagnostics:
        proposal = diagnostic.details.get("proposal")
        if not isinstance(proposal, RenameProposal):
            counts["malformed"] += 1
            log(
                f"  ERROR rename proposal MALFORMED {diagnostic.operation_id}: "
                f"{diagnostic.details.get('detail')}"
            )
            continue
        status = diagnostic.details["materialized_status"]
        if status in {"applied", "merged", "compensated", "contraction_drop"}:
            counts["applied"] += 1
        elif status == "rejected":
            counts["rejected"] += 1
        elif status == "pending":
            counts["pending"] += 1
        elif status == "invalid":
            counts["malformed"] += 1
        else:
            counts["lost"] += 1
        conn.execute(
            "INSERT INTO rename_proposals (id, proposal_operation_id, node_id, "
            "node_name_at_proposal, proposed_name, reason, proposed_by, proposed_at, "
            "status, resolved_at, resolution_note, resolution_phase, "
            "resolution_operation_id, compensation_operation_id, replay_disposition_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                proposal.id,
                proposal.operation_id,
                proposal.source_node_id,
                proposal.source_name,
                proposal.proposed_name,
                proposal.reason,
                proposal.proposed_by,
                proposal.at,
                status,
                diagnostic.details.get("resolved_at"),
                diagnostic.details.get("resolution_note"),
                diagnostic.details.get("resolution_phase"),
                diagnostic.details.get("resolution_operation_id"),
                diagnostic.details.get("compensation_operation_id"),
                diagnostic.details.get("replay_disposition_id"),
            ),
        )
    for blocker in blockers:
        counts["malformed"] += 1
        log(f"  ERROR rename proposal MALFORMED: {blocker.cause}: {blocker.details}")
    conn.commit()
    log(
        "Materialised rename proposals: "
        + ", ".join(f"{key} {value}" for key, value in counts.items())
    )
    return counts


# --- Host CLI ---


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(data_dir() / "knowledge.db"),
    )
    p = argparse.ArgumentParser(
        prog="assimilator.merge",
        description="Merge duplicate entity nodes (reversible, ledger-backed).",
    )
    p.add_argument("--db", default=default_db)
    p.add_argument("--survivor", help="canonical survivor node id")
    p.add_argument("--victims", help="comma-separated node ids to merge in")
    p.add_argument("--name", help="canonical name for the survivor")
    p.add_argument("--undo", help="merge_id to reverse")
    p.add_argument("--by", default=None, help="actor (email)")
    p.add_argument(
        "--confirmed-by",
        default=None,
        help="who confirmed this merge in the workbench; without it the merge is "
        "PROPOSED into the reviewer queue and nothing is applied",
    )
    p.add_argument("--confirmed-at", default=None, help="ISO UTC of the confirmation")
    p.add_argument(
        "--proposal-id",
        action="append",
        default=[],
        help="exact rename-proposal:<id> resolved by this merge; repeat as needed",
    )
    p.add_argument(
        "--confirmed-via",
        default="workbench-queue",
        choices=CONFIRMATION_VIAS,
        help="which workbench flow issued the confirmation",
    )
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    init_db(conn)
    try:
        if args.undo:
            append_undo_entry(args.undo, args.by)
            n = undo_merge(conn, args.undo)
            print(f"Undid merge {args.undo}: restored {n} node(s)")
            return 0
        if not (args.survivor and args.victims and args.name):
            p.error("merge needs --survivor, --victims and --name (or --undo)")
        victim_ids = [v.strip() for v in args.victims.split(",") if v.strip()]
        # Validate ids BEFORE touching the ledger, so a bad call fails clean with
        # no partial state (the workbench relies on fail-closed).
        missing = [n for n in [args.survivor, *victim_ids] if _node(conn, n) is None]
        if missing:
            p.error(f"node id(s) not found: {', '.join(missing)}")
        if not args.confirmed_by:
            if args.proposal_id:
                p.error("--proposal-id requires an independent --confirmed-by")
            entry = propose_merge(conn, args.survivor, victim_ids, args.name, args.by)
            print(
                "PROPOSED, not applied: no --confirmed-by. The cluster is in the "
                "reviewer queue and applies when Mark confirms it in the workbench."
                + ("" if entry else " (It was already queued.)")
            )
            return 0
        confirmation = confirmation_block(
            args.confirmed_by, args.confirmed_at, args.confirmed_via
        )
        merge_id = str(uuid.uuid4())
        created_at = _now()
        # Ledger first (captures victims' natural identity before they retire),
        # then the live mutation.
        append_merge_entry(
            conn,
            args.survivor,
            victim_ids,
            args.name,
            merge_id,
            created_at,
            args.by,
            confirmation=confirmation,
            proposal_ids=args.proposal_id,
        )
        merged = merge_nodes(
            conn, args.survivor, victim_ids, args.name, merge_id, created_at, args.by
        )
        print(
            f"Merged {merged} node(s) into {args.survivor} as '{args.name}' "
            f"(merge_id {merge_id})"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
