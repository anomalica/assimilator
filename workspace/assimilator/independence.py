"""Conservative node-level evidence independence (ADR 0051).

Semantic agreement, exact evidence identity and provenance-root identity are
separate graph relations. A claim is scoreable only when import validated its
exact Asset anchor and established an explicit root. Legacy scalar locations,
unknown roots and overlap-unknown text frames add no count.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from assimilator.evidence import assess_evidence_support


@dataclass(frozen=True)
class Independence:
    """Per-node independence, with the confidence in it stated alongside."""

    sources: int | None
    scored_claims: int
    unscored_claims: int

    @property
    def total_claims(self) -> int:
        return self.scored_claims + self.unscored_claims

    @property
    def unscored_fraction(self) -> float:
        return self.unscored_claims / self.total_claims if self.total_claims else 0.0


def independence_for_nodes(
    conn: sqlite3.Connection, node_ids: list[str] | None = None
) -> dict[str, Independence]:
    """node_id -> Independence, for the given nodes (default: every live node).

    Deterministic, no AI and no fallback from a Record, Asset, publisher or
    speaker string. ``None`` means no claim has enough exact evidence to score.
    """
    where = ""
    params: list = []
    if node_ids is not None:
        if not node_ids:
            return {}
        placeholders = ",".join("?" * len(node_ids))
        where = f" WHERE x.node_id IN ({placeholders})"
        params = list(node_ids)

    rows = conn.execute(
        f"""
        SELECT x.node_id, c.id
          FROM (
              SELECT node_id, claim_id FROM claim_node_refs
              UNION
              SELECT speaker_id AS node_id, id AS claim_id
                FROM claims WHERE speaker_id IS NOT NULL
          ) x
          JOIN claims c ON c.id = x.claim_id
          {where}
        """,  # noqa: S608 - placeholders only, no interpolated values
        params,
    ).fetchall()

    claims_by_node: dict[str, list[str]] = {}
    for node_id, claim_id in rows:
        claims_by_node.setdefault(str(node_id), []).append(str(claim_id))

    out: dict[str, Independence] = {}
    for node_id, claim_ids in claims_by_node.items():
        assessment = assess_evidence_support(conn, claim_ids)
        out[node_id] = Independence(
            sources=(
                assessment.independent_sources if assessment.scored_claims else None
            ),
            scored_claims=assessment.scored_claims,
            unscored_claims=assessment.unscored_claims,
        )
    return out
