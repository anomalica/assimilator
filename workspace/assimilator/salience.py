"""What the corpus says about a node's salience, before any extractor does.

The per-edge value is the extractor's to emit (claim_node_refs.salience). This
module is the DERIVED half: what can be known about a node's role from the shape
of the graph alone, with no model call. It exists for two jobs, and they are not
the same job.

1. A BASELINE. Built before the extractor's values arrive, so there is something
   to judge them against. Without it the first batch of salience values would
   have nothing to be wrong against.

2. A CHECK ON THE EXTRACTOR, never an input to it. A node marked `subject` on
   1,400 edges is a prompt fault, not a fact about the corpus - the check fails
   the build, and the fix is in the extraction prompt.

WHY THE DERIVED SIGNAL CANNOT REPLACE THE PER-EDGE VALUE, since it is the
obvious shortcut and it does not work. Measured 2026-09-08: 83,798 edges over
9,335 nodes; the ten most-referenced nodes hold 12.5% of all edges, the top 200
hold 45.4%, and it takes 1,000 nodes to reach 69%. It is a long tail, not a
head, so demoting the twenty commonest nodes cleans an eighth of the edges and
fixes nothing. The problem is that a typical claim carries two or three
references of which ONE is the subject, and which one it is cannot be read off
the node.

The one structural certainty: a claim that references exactly one node is about
that node. 10,343 of 38,863 claims are in that position, and those edges are
`subject` without a model reading them.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

VALUES = ("subject", "participant", "setting", "mentioned")

# A node this heavily referenced cannot be the subject of that many claims; the
# threshold is a share of the corpus rather than a count, so it survives growth.
IMPLAUSIBLE_SUBJECT_SHARE = 0.02


@dataclass
class NodeEdges:
    node_id: str
    name: str
    node_type: str
    edges: int
    records: int
    sole_reference: int  # edges on claims that reference this node and nothing else

    @property
    def sole_share(self) -> float:
        return self.sole_reference / self.edges if self.edges else 0.0


def node_edge_stats(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[NodeEdges]:
    """Per-node edge shape, most-referenced first."""
    rows = conn.execute(
        """
        SELECT n.id, n.name, n.node_type,
               COUNT(*) AS edges,
               COUNT(DISTINCT c.record_id) AS records,
               SUM(CASE WHEN (
                   SELECT COUNT(*) FROM claim_node_refs y WHERE y.claim_id = x.claim_id
               ) = 1 THEN 1 ELSE 0 END) AS sole
        FROM claim_node_refs x
        JOIN nodes n ON n.id = x.node_id
        JOIN claims c ON c.id = x.claim_id
        WHERE n.retired_at IS NULL
        GROUP BY n.id
        ORDER BY edges DESC
        """
    ).fetchall()
    out = [NodeEdges(r[0], r[1], r[2], r[3], r[4], r[5] or 0) for r in rows]
    return out[:limit] if limit else out


def sole_reference_edges(conn: sqlite3.Connection) -> int:
    """Edges the graph alone proves are `subject`: the claim references nothing
    else. Assigning these needs no model and no judgement."""
    return conn.execute(
        """
        SELECT COUNT(*) FROM claim_node_refs x
        WHERE (SELECT COUNT(*) FROM claim_node_refs y WHERE y.claim_id = x.claim_id) = 1
        """
    ).fetchone()[0]


def coverage(conn: sqlite3.Connection) -> dict[str, int]:
    """How much of the edge table carries a salience value, by value. `null` is
    NOT ASSESSED - it is reported as its own count and never folded into a
    value, because a consumer that treats absence as a verdict is the fault this
    column records."""
    counts = {v: 0 for v in VALUES}
    counts["null"] = 0
    for value, n in conn.execute(
        "SELECT salience, COUNT(*) FROM claim_node_refs GROUP BY salience"
    ):
        counts["null" if value is None else value] = n
    counts["total"] = sum(counts[k] for k in (*VALUES, "null"))
    return counts


def implausible_subjects(
    conn: sqlite3.Connection, share: float = IMPLAUSIBLE_SUBJECT_SHARE
) -> list[dict]:
    """Nodes marked `subject` on an implausible share of the corpus.

    The check on the extractor. A corpus-wide term - "UFO" on 1,414 claims -
    marked as the subject of thousands of claims means the prompt is reading
    presence as aboutness, and the fix is in the prompt, not in the rows.
    Reports nothing until salience values exist, so it is safe to run now.
    """
    total = conn.execute(
        "SELECT COUNT(*) FROM claim_node_refs WHERE salience IS NOT NULL"
    ).fetchone()[0]
    if not total:
        return []
    ceiling = max(int(total * share), 1)
    return [
        {"node_id": r[0], "name": r[1], "subject_edges": r[2], "ceiling": ceiling}
        for r in conn.execute(
            """
            SELECT n.id, n.name, COUNT(*) AS c
            FROM claim_node_refs x JOIN nodes n ON n.id = x.node_id
            WHERE x.salience = 'subject'
            GROUP BY n.id HAVING c > ?
            ORDER BY c DESC
            """,
            (ceiling,),
        )
    ]
