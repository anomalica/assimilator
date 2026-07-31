"""How many INDEPENDENT sources stand behind a node's claims (ADR 0039/0044).

`source_count` counts distinct records, which is not the same question. A podcast
and an article both relaying one anonymous email are two records and one source,
and that is exactly the page a reader would object to - two citations that are
really one voice. Independence counts distinct PROVENANCE ROOTS instead: who
originally asserted the thing, resolved through the alias graph so "DIA" and
"Defense Intelligence Agency" are one root rather than two.

The root rule lives in `database.provenance_root` and is deliberately biased:
every uncertain case collapses toward FEWER roots, because over-counting is the
unsafe direction. Splitting later raises independence; nothing can lower it once
a page has been published on an inflated number.

UNSCORED IS NOT ZERO AND NOT ONE. A claim whose digest predates ADR 0044 has no
chain, so its root is unknowable - not absent, unknowable. Those claims are
excluded from the count and reported separately, and a node with no scoreable
claims reports None rather than 0. The unscored COUNT matters as much as the
verdict: a node with 5% unscored and one with 60% unscored both report a number,
and only the first should be trusted. Without it, re-running after a backfill
cannot tell you which nodes actually became scoreable.

This is the third design this week decided by the same rule - a missing chain is
not independence, a missing review is not "unreviewed", a missing run_kind is not
"production". The read rules are pinned in tests/test_absent_is_not_a_value.py.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Independence:
    """Per-node independence, with the confidence in it stated alongside."""

    sources: int | None  # distinct provenance roots; None when nothing is scoreable
    scored_claims: int
    unscored_claims: int

    @property
    def total_claims(self) -> int:
        return self.scored_claims + self.unscored_claims

    @property
    def unscored_fraction(self) -> float:
        return self.unscored_claims / self.total_claims if self.total_claims else 0.0


def _alias_index(conn: sqlite3.Connection) -> dict[str, str]:
    """casefolded name or alias -> node id, for resolving a named/document origin.

    Built once per pass. `provenance_root` resolves origins one at a time through
    find_node_by_name, which is right for a single claim and quadratic for a
    corpus - this is the same resolution, hoisted.
    """
    index: dict[str, str] = {}
    for node_id, name in conn.execute(
        "SELECT id, name FROM nodes WHERE retired_at IS NULL"
    ):
        index.setdefault(name.casefold(), node_id)
    for alias, node_id in conn.execute("SELECT alias, node_id FROM aliases"):
        index.setdefault(alias.casefold(), node_id)
    return index


def _root(row: tuple, aliases: dict[str, str]) -> tuple[str, str] | None:
    """The claim's provenance root, or None when it cannot be known.

    Mirrors database.provenance_root exactly - same branches, same collapse
    direction - but takes the row and a prebuilt alias index so a whole corpus can
    be scored in one pass. None is the `unknown` case made explicit: the caller
    must exclude it rather than treat it as a root, because one shared "unknown"
    root would read as one shared source and quietly corroborate everything
    pre-0044 with everything else.
    """
    speaker_id, record_id, origin_kind, origin = row
    if not origin_kind:
        return None
    if origin_kind in ("speaker", "unattributed"):
        return ("speaker", speaker_id) if speaker_id else ("record", record_id)
    if origin_kind == "anonymous":
        # Every anonymous origin collapses to ONE root until a semantic matcher
        # can prove two of them distinct. Across records the prose proves nothing
        # - this record's "the chairman" may be that record's "my DIA contact".
        return ("anonymous", "")
    name = (origin or "").strip()
    if not name:
        return None
    node_id = aliases.get(name.casefold())
    return ("node", node_id) if node_id else (origin_kind, name.casefold())


def independence_for_nodes(
    conn: sqlite3.Connection, node_ids: list[str] | None = None
) -> dict[str, Independence]:
    """node_id -> Independence, for the given nodes (default: every live node).

    One pass: every claim touching any requested node is read once, its root
    computed from an in-memory alias index, and the distinct roots counted per
    node. Deterministic, no AI, no embeddings.
    """
    aliases = _alias_index(conn)
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
        SELECT x.node_id, c.speaker_id, c.record_id, c.origin_kind, c.origin
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

    roots: dict[str, set] = {}
    scored: dict[str, int] = {}
    unscored: dict[str, int] = {}
    for node_id, *claim_row in rows:
        root = _root(tuple(claim_row), aliases)
        if root is None:
            unscored[node_id] = unscored.get(node_id, 0) + 1
            continue
        roots.setdefault(node_id, set()).add(root)
        scored[node_id] = scored.get(node_id, 0) + 1

    out: dict[str, Independence] = {}
    for node_id in set(scored) | set(unscored):
        n_scored = scored.get(node_id, 0)
        out[node_id] = Independence(
            sources=len(roots[node_id]) if n_scored else None,
            scored_claims=n_scored,
            unscored_claims=unscored.get(node_id, 0),
        )
    return out
