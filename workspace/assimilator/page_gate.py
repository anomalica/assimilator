"""The page-worthiness gate: which nodes earn a published page.

Supersedes the flat claim-count floor (page_set.py). A raw count cannot tell
"subject of the corpus" from "mentioned in it" - it produced ~350 junk proposals
(an object cited once, a place mentioned in passing). Two things fix it
(node-types.md, "Page-worthiness"): node TYPE and source INDEPENDENCE.

1. Type tier - no type is permanently barred:
   - Page-worthy (modest floor): person, organisation, project, event, topic.
   - High-bar (central-subject only): place, object, document.
   Deprecated taxonomy types (matter, concept, programme, investigation) and the
   curator-only `pattern` are not page-worthy by extraction; they are gated out
   until re-digestion reclassifies them into the canonical eight.

2. The floor is distinct independent sources, not raw claim count. For now an
   "independent source" is the distinct-source-record PROXY (count by record_id);
   true provenance-root independence is the corroboration / evidence-scoring model
   (ADR 0039), carried as a separate proposal column for when that pins.

Host-light (stdlib only) so the synthesiser and the scheduler share it without
heavier deps - the same discipline as page_set.py, which this supersedes.
"""

from __future__ import annotations

import os
import sqlite3

PAGE_WORTHY = "page-worthy"
HIGH_BAR = "high-bar"

TIER: dict[str, str] = {
    "person": PAGE_WORTHY,
    "organisation": PAGE_WORTHY,
    "project": PAGE_WORTHY,
    "event": PAGE_WORTHY,
    "topic": PAGE_WORTHY,
    "place": HIGH_BAR,
    "object": HIGH_BAR,
    "document": HIGH_BAR,
}

DEFAULT_FLOORS: dict[str, tuple[int, int]] = {
    PAGE_WORTHY: (3, 2),  # >= 3 claims from >= 2 distinct sources
    HIGH_BAR: (6, 3),  # >= 6 claims from >= 3 distinct sources
}


def floors() -> dict[str, tuple[int, int]]:
    """Per-tier (min_claims, min_sources), env-overridable for calibration.

    ANOMALICA_PAGE_WORTHY_MIN_CLAIMS / _MIN_SOURCES and the HIGH_BAR equivalents.
    """

    def _f(tier: str, default: tuple[int, int]) -> tuple[int, int]:
        prefix = "ANOMALICA_" + tier.replace("-", "_").upper()
        dc, ds = default
        return (
            int(os.environ.get(prefix + "_MIN_CLAIMS", dc)),
            int(os.environ.get(prefix + "_MIN_SOURCES", ds)),
        )

    return {tier: _f(tier, default) for tier, default in DEFAULT_FLOORS.items()}


def _node_counts(conn: sqlite3.Connection) -> list[tuple[str, str, int, int]]:
    """(node_id, node_type, claim_count, source_count) for every active node a
    claim references (as speaker or node-ref). Counts are distinct claims and
    distinct source records (the independence proxy)."""
    return conn.execute(
        """
        SELECT n.id, n.node_type,
               COUNT(DISTINCT x.cid) AS claims,
               COUNT(DISTINCT x.rid) AS sources
          FROM nodes n
          JOIN (
              SELECT speaker_id AS nid, id AS cid, record_id AS rid
                FROM claims WHERE speaker_id IS NOT NULL
              UNION
              SELECT cnr.node_id AS nid, c.id AS cid, c.record_id AS rid
                FROM claim_node_refs cnr JOIN claims c ON c.id = cnr.claim_id
          ) x ON x.nid = n.id
         WHERE n.retired_at IS NULL
         GROUP BY n.id
        """
    ).fetchall()


def _source_spread(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """(top_source_claims, second_source_claims) per node.

    A distinct-source COUNT cannot tell "10 claims from two sources" from "10
    claims from one book plus one passing mention" - both report 2. The count of
    sources says nothing about how the claims are spread across them, and it is
    the spread that decides whether a page is genuinely corroborated or a summary
    of a single (often copyrighted) work with a second source attached. These two
    numbers make the difference visible; nothing gates on them yet.
    """
    rows = conn.execute(
        """
        SELECT nid, rid, COUNT(DISTINCT cid) AS claims FROM (
            SELECT speaker_id AS nid, id AS cid, record_id AS rid
              FROM claims WHERE speaker_id IS NOT NULL
            UNION
            SELECT cnr.node_id AS nid, c.id AS cid, c.record_id AS rid
              FROM claim_node_refs cnr JOIN claims c ON c.id = cnr.claim_id
        ) GROUP BY nid, rid ORDER BY nid, claims DESC
        """
    ).fetchall()
    spread: dict[str, list[int]] = {}
    for node_id, _record_id, claims in rows:
        spread.setdefault(node_id, []).append(claims)
    return {
        node_id: (counts[0], counts[1] if len(counts) > 1 else 0)
        for node_id, counts in spread.items()
    }


def page_gate_rows(conn: sqlite3.Connection) -> list[dict]:
    """Nodes that pass the page-worthiness gate, with their tier and counts.

    Each row: {node_id, node_type, tier, claim_count, source_count,
    top_source_claims, second_source_claims}. A node whose type is not in TIER
    (deprecated / curator-only) never passes. Ordered by claim_count descending
    then node_id - a stable, review-friendly ranking with the strongest subjects
    first.

    The two source-spread fields are REPORTED, NOT ENFORCED. They exist because
    the source COUNT hides single-source dominance (see _source_spread), and the
    size of that problem has to be measurable before a floor is set on it.
    """
    tier_floors = floors()
    spread = _source_spread(conn)
    out: list[dict] = []
    for node_id, node_type, claims, sources in _node_counts(conn):
        tier = TIER.get(node_type)
        if tier is None:
            continue
        min_claims, min_sources = tier_floors[tier]
        if claims >= min_claims and sources >= min_sources:
            top, second = spread.get(node_id, (claims, 0))
            out.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "tier": tier,
                    "claim_count": claims,
                    "source_count": sources,
                    "top_source_claims": top,
                    "second_source_claims": second,
                }
            )
    out.sort(key=lambda r: (-r["claim_count"], r["node_id"]))
    return out
