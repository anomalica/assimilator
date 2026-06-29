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

A node id is referenced in exactly four places (re-pointed here): claim_node_refs,
claims.speaker_id, records.producer_id, aliases.

Host-runnable: `python -m assimilator.merge --survivor <id> --victims <id,id>
--name "<canonical>"` and `--undo <merge_id>`. No Claude, no money.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from assimilator.database import init_db
from assimilator.matching import match_node


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
            r[0]
            for r in conn.execute(
                "SELECT claim_id FROM claim_node_refs WHERE node_id = ?", (victim_id,)
            ).fetchall()
        }
        survivor_claims = {
            r[0]
            for r in conn.execute(
                "SELECT claim_id FROM claim_node_refs WHERE node_id = ?", (survivor_id,)
            ).fetchall()
        }
        refs_only_victim = sorted(victim_claims - survivor_claims)
        refs_both = sorted(victim_claims & survivor_claims)
        for cid in refs_only_victim:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id) VALUES (?, ?)",
                (cid, survivor_id),
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

        reversal = json.dumps(
            {
                "refs_only_victim": refs_only_victim,
                "refs_both": refs_both,
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

    conn.execute(
        "UPDATE nodes SET name = ? WHERE id = ?", (canonical_name, survivor_id)
    )
    conn.commit()
    return merged


def undo_merge(conn: sqlite3.Connection, merge_id: str) -> int:
    """Reverse a merge in the live DB using the recorded reversal data. Returns
    the number of victims restored."""
    rows = conn.execute(
        "SELECT survivor_id, victim_id, survivor_prior_name, reversal "
        "FROM node_merges WHERE merge_id = ? AND undone_at IS NULL",
        (merge_id,),
    ).fetchall()
    if not rows:
        return 0
    survivor_id = rows[0][0]
    survivor_prior_name = rows[0][2]
    for survivor_id, victim_id, survivor_prior_name, reversal_json in rows:
        rev = json.loads(reversal_json)
        conn.execute("UPDATE nodes SET retired_at = NULL WHERE id = ?", (victim_id,))
        for cid in rev["refs_only_victim"]:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id) VALUES (?, ?)",
                (cid, victim_id),
            )
            conn.execute(
                "DELETE FROM claim_node_refs WHERE claim_id = ? AND node_id = ?",
                (cid, survivor_id),
            )
        for cid in rev["refs_both"]:
            conn.execute(
                "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id) VALUES (?, ?)",
                (cid, victim_id),
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
) -> None:
    """Append a merge entry to the durable ledger, keyed on natural identity
    (names), with ids as an audit snapshot. Captured BEFORE the live merge so the
    victims still resolve to their own natural identity."""
    entry = {
        "op": "merge",
        "merge_id": merge_id,
        "at": created_at,
        "by": created_by,
        "canonical_name": canonical_name,
        "survivor": _natural(conn, survivor_id),
        "victims": [_natural(conn, v) for v in victim_ids if _node(conn, v)],
        "audit": {"survivor_id": survivor_id, "victim_ids": list(victim_ids)},
    }
    _append(entry)


def append_undo_entry(merge_id: str, created_by: str | None) -> None:
    _append({"op": "undo", "merge_id": merge_id, "at": _now(), "by": created_by})


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
    prior_names, within node_type. Returns the node id, or None if absent."""
    node_type = nat.get("node_type")
    for name in [nat.get("name"), *(nat.get("prior_names") or [])]:
        if not name:
            continue
        m = match_node(conn, name, node_type)
        if m:
            return m[0]
    return None


def replay_ledger(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-apply the durable ledger over the freshly-imported graph. Merges whose
    undo entry is present are skipped; a victim that no longer resolves is skipped
    and logged (its source left the corpus). Keyed on natural identity."""
    log = on_progress or (lambda _: None)
    entries = read_ledger()
    undone = {e["merge_id"] for e in entries if e.get("op") == "undo"}
    applied = skipped = 0
    for e in entries:
        if e.get("op") != "merge" or e["merge_id"] in undone:
            continue
        survivor_id = _resolve_natural(conn, e["survivor"])
        if survivor_id is None:
            log(
                f"  replay skip {e['merge_id']}: survivor '{e['survivor']['name']}' not in graph"
            )
            skipped += 1
            continue
        victim_ids = []
        for v in e["victims"]:
            vid = _resolve_natural(conn, v)
            if vid and vid != survivor_id:
                victim_ids.append(vid)
        if not victim_ids:
            skipped += 1
            continue
        merge_nodes(
            conn,
            survivor_id,
            victim_ids,
            e["canonical_name"],
            e["merge_id"],
            created_at=e.get("at"),
            created_by=e.get("by"),
        )
        applied += 1
    log(f"Replayed {applied} merges ({skipped} skipped)")
    return {"applied": applied, "skipped": skipped}


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


def replay_rejections(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-populate node_rejections from the durable ledger after a rebuild,
    resolving each rejection's nodes by natural identity."""
    log = on_progress or (lambda _: None)
    entries = read_rejections()
    undone = {e["rejection_id"] for e in entries if e.get("op") == "unreject"}
    applied = 0
    for e in entries:
        if e.get("op") != "reject" or e["rejection_id"] in undone:
            continue
        ids = {nid for nid in (_resolve_natural(conn, n) for n in e["nodes"]) if nid}
        if len(ids) < 2:
            continue
        for node_id in sorted(ids):
            conn.execute(
                "INSERT OR REPLACE INTO node_rejections (rejection_id, node_id, reason, "
                "created_at, created_by, undone_at) VALUES (?, ?, ?, ?, ?, NULL)",
                (e["rejection_id"], node_id, e.get("reason"), e.get("at"), e.get("by")),
            )
        applied += 1
    conn.commit()
    log(f"Replayed {applied} rejections")
    return {"applied": applied}


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


def _append_rename(entry: dict) -> None:
    path = renames_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


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
) -> None:
    """Record a node-name correction in the durable ledger, keyed on the node's
    PRE-rename natural identity (the name a fresh import carries) so replay can
    resolve it. Used directly when the live rename has already been applied."""
    _append_rename(
        {
            "op": "rename",
            "rename_id": rename_id,
            "at": created_at,
            "by": created_by,
            "new_name": new_name,
            "node": old_natural,
        }
    )


def rename_node(
    conn: sqlite3.Connection,
    node_id: str,
    new_name: str,
    rename_id: str,
    created_at: str | None = None,
    created_by: str | None = None,
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
        _natural(conn, node_id), new_name, rename_id, created_at, created_by
    )
    conn.execute("UPDATE nodes SET name = ? WHERE id = ?", (new_name, node_id))
    conn.execute(
        "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
        (old_name, node_id),
    )
    conn.commit()


def replay_renames(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-apply durable renames over the freshly-rebuilt graph, after merges and
    rejections (a renamed node may be a merge survivor). Resolves each node by its
    pre-rename natural identity, sets the new name, keeps the old name as an alias.
    A node that no longer resolves is skipped (its source left the corpus)."""
    log = on_progress or (lambda _: None)
    entries = read_renames()
    undone = {e["rename_id"] for e in entries if e.get("op") == "unrename"}
    applied = skipped = 0
    for e in entries:
        if e.get("op") != "rename" or e["rename_id"] in undone:
            continue
        nid = _resolve_natural(conn, e["node"])
        if nid is None:
            skipped += 1
            continue
        old_name = _node(conn, nid)[0]
        conn.execute("UPDATE nodes SET name = ? WHERE id = ?", (e["new_name"], nid))
        conn.execute(
            "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
            (old_name, nid),
        )
        applied += 1
    conn.commit()
    log(f"Replayed {applied} renames ({skipped} skipped)")
    return {"applied": applied, "skipped": skipped}


# --- Host CLI ---


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
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
        merge_id = str(uuid.uuid4())
        created_at = _now()
        # Ledger first (captures victims' natural identity before they retire),
        # then the live mutation.
        append_merge_entry(
            conn, args.survivor, victim_ids, args.name, merge_id, created_at, args.by
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
