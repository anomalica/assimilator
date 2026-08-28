"""How far each published page has drifted from the graph it was built from.

Every brief freezes the claim ids and hashes it used, so comparing that against
what the node holds now gives a real number: "this page is 12% behind its
sources". The computation already exists in anomalica_common.staleness; nothing
was surfacing it.

WHY THIS IS A SEPARATE MANIFEST AND NOT A FIELD IN EACH BRIEF. The figure decays
the moment it is written - the assembler measured a page built at 14:56 that was
stale by 15:20. A drift value baked into 752 published artefacts is 752 assertions
that go wrong at different rates and can only be corrected by rewriting all of
them. One manifest with one `generated_at` is a snapshot that says when it was
taken, is cheap to regenerate, and cannot disagree with itself.

It is also the honest shape: staleness is a property of the relationship between a
page and the graph at a moment, not a property of the page.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from anomalica_common.staleness import brief_drift

from assimilator.publish_briefs import _load


def _current_claims(conn: sqlite3.Connection, node_id: str) -> dict[str, str]:
    """claim_id -> claim_hash for what the node holds NOW.

    Same predicate the synthesiser selects on - claims the node speaks and claims
    that reference it - because drift measured against a different set would be
    measuring the query difference rather than the graph moving.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT c.id, c.claim_hash
        FROM claims c
        WHERE c.speaker_id = ?
           OR c.id IN (SELECT claim_id FROM claim_node_refs WHERE node_id = ?)
        """,
        (node_id, node_id),
    ).fetchall()
    # A claim with no claim_hash is kept, under a sentinel, rather than dropped.
    # Dropping it makes the claim invisible to the comparison, which UNDER-reports
    # drift - the unsafe direction for a freshness figure, because a page reads as
    # current partly because some of its evidence could not be checked. The
    # sentinel is distinct per claim so an unhashed claim never compares equal to
    # anything, including itself in an older brief.
    return {cid: (h or f"unhashed:{cid}") for cid, h in rows}


def staleness_manifest(conn: sqlite3.Connection, briefs_dir: Path) -> dict:
    """Per-page drift for every brief on disk, as one timestamped snapshot."""
    pages: dict[str, dict] = {}
    missing_node = 0
    for path in sorted(briefs_dir.glob("*.yaml")):
        try:
            brief = _load(path.read_text())
        except (OSError, Exception):  # noqa: BLE001 - a bad file must not stop the sweep
            continue
        if not isinstance(brief, dict):
            continue
        page = brief.get("page") or {}
        node_id = page.get("node_id")
        slug = page.get("slug") or path.stem
        frozen = {
            c["claim_id"]: c["claim_hash"]
            for c in brief.get("claims") or []
            if c.get("claim_id") and c.get("claim_hash")
        }
        if not node_id:
            # A record brief has no node. It is not stale-able this way, and
            # reporting 100% would be a fabrication rather than a measurement.
            missing_node += 1
            continue
        drift = brief_drift(frozen, _current_claims(conn, node_id))
        # A brief whose node was MERGED AWAY is not 100% stale, it is superseded -
        # its material lives on the survivor's page. Reported identically, the two
        # read as "this page has lost all its evidence", which is alarming and
        # false for one of them. Measured on the first run: 47 pages at 100%, of
        # which 4 were merge victims and 43 were genuine drift.
        row = conn.execute(
            "SELECT retired_at FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        node_state = "absent" if row is None else ("retired" if row[0] else "live")
        pages[slug] = {
            "node_id": node_id,
            "node_state": node_state,
            "title": page.get("title"),
            "brief_hash": brief.get("brief_hash"),
            **{
                k: drift[k]
                for k in (
                    "pct",
                    "content_pct",
                    "new_pct",
                    "gone",
                    "added",
                    "changed",
                    "brief_total",
                    "current_total",
                )
            },
        }
    return {
        "schema": "anomalica/brief-staleness/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "not_measurable": missing_node,
    }


def write_manifest(manifest: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_path
