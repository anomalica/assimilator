"""Generate article proposals: which nodes earn a published page.

The first-class evolution of the page floor (page_set.py). Each maintenance pass
recomputes the proposal set from the page-worthiness gate (page_gate.py) over the
post-merge graph and writes it to the DERIVED page_proposals table - no replay,
pure function of the graph plus the durable veto.

Lifecycle (node-types.md, ADR 0038): proposed -> approved (staged) -> brief-
emitted -> assembled. This module owns the first transition (proposed) and the
permanent editorial VETO that keeps a node off the page list. A veto is curation,
so it lives in the durable ledger (curation/page-vetoes.yaml) and is replayed on
rebuild, keyed on natural identity exactly as merges and rejections are - the DB
is derived and a veto applied only to it would be lost on the next rebuild.

A veto is NOT a rejection: node_rejections is "these two nodes are not the same
entity"; a page-veto keeps the node in the graph and merely says "it is a mention,
not a subject - never give it a page".

Host-runnable, no Claude, no money:
  python -m assimilator.propose_pages            # recompute the proposal table
  python -m assimilator.propose_pages --veto <ids> --reason "..." --by <email>
  python -m assimilator.propose_pages --unveto <veto_id>
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from assimilator.database import init_db
from assimilator.merge import _natural, _node, _resolve_natural
from assimilator.independence import independence_for_nodes
from assimilator.page_gate import page_gate_rows

STATUS_PROPOSED = "proposed"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- The durable page-veto ledger (curation, replayable) ---


def veto_ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # …/anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "page-vetoes.yaml"


def _append_veto(entry: dict) -> None:
    path = veto_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


def read_vetoes() -> list[dict]:
    path = veto_ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def veto_pages(
    conn: sqlite3.Connection,
    node_ids: list[str],
    reason: str | None,
    veto_id: str,
    created_at: str | None = None,
    created_by: str | None = None,
) -> None:
    """Record an editorial "never a page" decision: durable ledger entry (natural-
    identity keyed) + derived page_vetoes rows. propose() then excludes these
    nodes so the proposal stops reappearing each pass."""
    created_at = created_at or _now()
    _append_veto(
        {
            "op": "veto",
            "veto_id": veto_id,
            "at": created_at,
            "by": created_by,
            "reason": reason,
            "nodes": [_natural(conn, n) for n in node_ids if _node(conn, n)],
            "audit": {"node_ids": list(node_ids)},
        }
    )
    for node_id in sorted(set(node_ids)):
        conn.execute(
            "INSERT OR REPLACE INTO page_vetoes (veto_id, node_id, reason, "
            "created_at, created_by, undone_at) VALUES (?, ?, ?, ?, ?, NULL)",
            (veto_id, node_id, reason, created_at, created_by),
        )
    conn.commit()


def un_veto(conn: sqlite3.Connection, veto_id: str) -> int:
    _append_veto({"op": "unveto", "veto_id": veto_id, "at": _now(), "by": None})
    cur = conn.execute(
        "UPDATE page_vetoes SET undone_at = ? WHERE veto_id = ? AND undone_at IS NULL",
        (_now(), veto_id),
    )
    conn.commit()
    return cur.rowcount


def replay_vetoes(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Re-populate page_vetoes from the durable ledger after a rebuild, resolving
    each veto's node by natural identity."""
    log = on_progress or (lambda _: None)
    entries = read_vetoes()
    undone = {e["veto_id"] for e in entries if e.get("op") == "unveto"}
    applied = 0
    for e in entries:
        if e.get("op") != "veto" or e["veto_id"] in undone:
            continue
        ids = {nid for nid in (_resolve_natural(conn, n) for n in e["nodes"]) if nid}
        if not ids:
            continue
        for node_id in sorted(ids):
            conn.execute(
                "INSERT OR REPLACE INTO page_vetoes (veto_id, node_id, reason, "
                "created_at, created_by, undone_at) VALUES (?, ?, ?, ?, ?, NULL)",
                (e["veto_id"], node_id, e.get("reason"), e.get("at"), e.get("by")),
            )
        applied += 1
    conn.commit()
    log(f"Replayed {applied} page vetoes")
    return {"applied": applied}


def vetoed_node_ids(conn: sqlite3.Connection) -> set[str]:
    """Active vetoed node ids - excluded from proposals."""
    try:
        rows = conn.execute(
            "SELECT node_id FROM page_vetoes WHERE undone_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r[0] for r in rows}


# --- The derived proposal set ---


def propose(conn: sqlite3.Connection, computed_at: str | None = None) -> list[dict]:
    """Recompute page_proposals from the gate, minus vetoes. Wipes and repopulates
    the derived table (no replay - it is a pure function of the graph + vetoes).
    Returns the proposal rows, strongest subjects first."""
    computed_at = computed_at or _now()
    vetoed = vetoed_node_ids(conn)
    rows = [r for r in page_gate_rows(conn) if r["node_id"] not in vetoed]
    # Independence: distinct provenance roots, not distinct records. Computed in
    # one pass over the proposed nodes only, and reported alongside the count of
    # claims whose chain predates ADR 0044 and so cannot be scored at all.
    scores = independence_for_nodes(conn, [r["node_id"] for r in rows])
    for r in rows:
        score = scores.get(r["node_id"])
        r["independent_source_count"] = score.sources if score else None
        r["unscored_claims"] = score.unscored_claims if score else None
    conn.execute("DELETE FROM page_proposals")
    for r in rows:
        conn.execute(
            "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
            "source_count, independent_source_count, top_source_claims, "
            "second_source_claims, unscored_claims, status, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r["node_id"],
                r["node_type"],
                r["tier"],
                r["claim_count"],
                r["source_count"],
                r.get("independent_source_count"),
                r["top_source_claims"],
                r["second_source_claims"],
                r.get("unscored_claims"),
                STATUS_PROPOSED,
                computed_at,
            ),
        )
    conn.commit()
    return rows


def proposed_node_ids(conn: sqlite3.Connection) -> list[str]:
    """Node ids currently proposed for a page, strongest first - the page set the
    synthesiser emits briefs for. Empty until propose() has run (the dependency
    gate: proposal-gen precedes synthesise)."""
    try:
        rows = conn.execute(
            "SELECT node_id FROM page_proposals ORDER BY claim_count DESC, node_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


# --- Host CLI ---


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    p = argparse.ArgumentParser(
        prog="assimilator.propose_pages",
        description="Generate article proposals (page-worthiness gate, ledger-backed veto).",
    )
    p.add_argument("--db", default=default_db)
    p.add_argument("--veto", help="comma-separated node ids to veto (never a page)")
    p.add_argument("--unveto", help="veto_id to reverse")
    p.add_argument("--reason", default=None)
    p.add_argument("--by", default=None, help="actor (email)")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    init_db(conn)
    try:
        if args.unveto:
            n = un_veto(conn, args.unveto)
            print(f"Un-vetoed {args.unveto}: cleared {n} node(s)")
            return 0
        if args.veto:
            node_ids = [v.strip() for v in args.veto.split(",") if v.strip()]
            missing = [n for n in node_ids if _node(conn, n) is None]
            if missing:
                p.error(f"node id(s) not found: {', '.join(missing)}")
            veto_id = str(uuid.uuid4())
            veto_pages(conn, node_ids, args.reason, veto_id, created_by=args.by)
            print(f"Vetoed {len(node_ids)} node(s) (veto_id {veto_id})")
        rows = propose(conn)
        by_tier: dict[str, int] = {}
        for r in rows:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        print(f"{len(rows)} page proposals ({by_tier})")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
