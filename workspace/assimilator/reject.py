"""Record a "not a duplicate" decision (host CLI).

Confirmed-distinct ground truth for a candidate cluster that a human judged NOT
the same entity (e.g. NDAA Section 1632 vs 1673 - similar names, distinct things).
Durable in the curation ledger (rejections.yaml), replayed on rebuild, and
excluded by propose-merges so the pair stops reappearing. Reversible (--undo).

`python -m assimilator.reject --nodes <id>,<id> --reason "<why>"` / `--undo <id>`.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path

from assimilator.database import init_db
from assimilator.merge import _node, _now, reject_nodes, un_reject


def main(argv: list[str] | None = None) -> int:
    import argparse

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    p = argparse.ArgumentParser(
        prog="assimilator.reject",
        description="Record a not-a-duplicate decision (excludes the cluster from candidates).",
    )
    p.add_argument("--db", default=default_db)
    p.add_argument(
        "--nodes", help="comma-separated node ids that are NOT the same entity"
    )
    p.add_argument("--reason", default=None)
    p.add_argument("--by", default=None, help="actor (email)")
    p.add_argument("--undo", help="rejection_id to reverse")
    args = p.parse_args(argv)

    conn = sqlite3.connect(args.db)
    init_db(conn)
    try:
        if args.undo:
            n = un_reject(conn, args.undo)
            print(f"Un-rejected {args.undo} ({n} row)")
            return 0
        if not args.nodes:
            p.error("reject needs --nodes (or --undo)")
        node_ids = [n.strip() for n in args.nodes.split(",") if n.strip()]
        missing = [n for n in node_ids if _node(conn, n) is None]
        if missing:
            p.error(f"node id(s) not found: {', '.join(missing)}")
        if len(set(node_ids)) < 2:
            p.error("reject needs at least two distinct node ids")
        rejection_id = str(uuid.uuid4())
        reject_nodes(conn, node_ids, args.reason, rejection_id, _now(), args.by)
        print(f"Recorded rejection {rejection_id} for {len(node_ids)} nodes")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
