"""The page-set floor: which entities deserve a page.

Interim gate until algorithmic-evidence-scoring pins the real bar (Mark's call):
an entity earns a page only if it carries at least MIN_CLAIMS distinct claims from
at least MIN_SOURCES distinct source records. This cuts the long tail of thin /
junk entities (most nodes have <=2 claims) without an AI scorer.

"independent sources" here is the distinct-source-record PROXY. True independence
is distinct provenance roots not sharing a wire origin (CNN+BBC+Reuters off one
press release = 3 records but 1 real source) - that correction is the
corroboration/evidence-scoring model, not this floor.

The floor runs over the POST-MERGE graph: a fragmented entity (e.g. the 2004
Nimitz incident split across 7 thin nodes) fails the floor in pieces but, once
the curation merges consolidate it into one node, passes easily as one page.

Host-light by design (stdlib only) so both the synthesiser and the scheduler can
share it without dragging in heavier deps.
"""

from __future__ import annotations

import os
import sqlite3

DEFAULT_MIN_CLAIMS = 2
DEFAULT_MIN_SOURCES = 2


def floor() -> tuple[int, int]:
    return (
        int(os.environ.get("ANOMALICA_PAGE_MIN_CLAIMS", DEFAULT_MIN_CLAIMS)),
        int(os.environ.get("ANOMALICA_PAGE_MIN_SOURCES", DEFAULT_MIN_SOURCES)),
    )


def page_set_node_ids(
    conn: sqlite3.Connection,
    min_claims: int | None = None,
    min_sources: int | None = None,
) -> list[str]:
    """Active entity node ids that pass the page floor, ordered by id.

    A node's claims are those it speaks OR is referenced in; its sources are the
    distinct records those claims come from. Retired (merged-away) nodes are
    excluded - the survivor carries the consolidated count.
    """
    fc, fs = floor()
    mc = fc if min_claims is None else min_claims
    ms = fs if min_sources is None else min_sources
    rows = conn.execute(
        """
        SELECT n.id FROM nodes n
        JOIN (
            SELECT speaker_id AS nid, id AS cid, record_id AS rid
              FROM claims WHERE speaker_id IS NOT NULL
            UNION
            SELECT cnr.node_id AS nid, c.id AS cid, c.record_id AS rid
              FROM claim_node_refs cnr JOIN claims c ON c.id = cnr.claim_id
        ) x ON x.nid = n.id
        WHERE n.retired_at IS NULL
        GROUP BY n.id
        HAVING COUNT(DISTINCT x.cid) >= ? AND COUNT(DISTINCT x.rid) >= ?
        ORDER BY n.id
        """,
        (mc, ms),
    ).fetchall()
    return [r[0] for r in rows]
