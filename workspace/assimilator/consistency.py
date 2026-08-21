"""Derived-data invariants: what must be true between the graph, the proposals,
the briefs and the built pages.

Every stage after the digests is DERIVED and rebuildable, which is what makes the
pipeline safe to re-run - and also what lets it drift silently. Nothing fails when
a proposal points at a node a merge retired, or when a page is built from a brief
the graph has moved past. The lane still runs, still exits 0, still writes
something. It is just describing a corpus that no longer exists.

Every check here was found by hand on 2026-08-21 after a night of merges, one at
a time, each time by noticing a number that did not add up. They are written down
so the next drift is reported rather than rediscovered.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from anomalica_common.slug import slugify


@dataclass
class Finding:
    check: str
    detail: str
    count: int
    samples: list[str] = field(default_factory=list)
    repair: str | None = None


_BUILT_FROM = re.compile(r"built_from:\s*\n\s*brief_hash:\s*(\S+)")


def _briefs(briefs_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not briefs_dir.is_dir():
        return out
    for f in sorted(briefs_dir.glob("*.yaml")):
        try:
            out[f.name] = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError:
            out[f.name] = {}
    return out


def check_all(
    conn: sqlite3.Connection, briefs_dir: Path, content_dir: Path | None
) -> list[Finding]:
    findings: list[Finding] = []
    live = {
        r[0]: r[1]
        for r in conn.execute("SELECT id, name FROM nodes WHERE retired_at IS NULL")
    }
    proposals = {r[0] for r in conn.execute("SELECT node_id FROM page_proposals")}
    briefs = _briefs(briefs_dir)
    brief_node = {n: (d.get("page") or {}).get("node_id") for n, d in briefs.items()}

    # 1. A proposal for a node that no longer exists. A merge retires its victims
    # and page_proposals is derived, so it goes stale until propose-pages reruns -
    # meanwhile the scheduler enumerates assembly work for a node that is gone.
    dead = [
        r[0]
        for r in conn.execute(
            "SELECT p.node_id FROM page_proposals p JOIN nodes n ON n.id = p.node_id "
            "WHERE n.retired_at IS NOT NULL"
        )
    ]
    orphan = [n for n in proposals if n not in live]
    if dead or orphan:
        findings.append(
            Finding(
                "proposal-points-at-dead-node",
                "page_proposals rows whose node is retired or absent",
                len(set(dead) | set(orphan)),
                sorted(set(dead) | set(orphan))[:5],
                "assimilator propose-pages",
            )
        )

    # 2. A brief describing a node that is gone. Emission only writes, so a brief
    # outlives its node; the assembler takes a brief by slug and will build a page
    # from a dead one.
    dead_briefs = [n for n, nid in brief_node.items() if nid and nid not in live]
    if dead_briefs:
        findings.append(
            Finding(
                "brief-for-dead-node",
                "briefs whose node is retired or absent",
                len(dead_briefs),
                sorted(dead_briefs)[:5],
                "python -m assimilator.synthesise (prunes on emit)",
            )
        )

    # 3. A brief stranded at an old slug. A RENAME leaves the survivor's own brief
    # behind under its previous name, so one live node has two briefs - enough to
    # republish a page the rename was meant to replace.
    stranded = [
        n
        for n, nid in brief_node.items()
        if nid in live and n[:-5] != slugify(live[nid])
    ]
    if stranded:
        findings.append(
            Finding(
                "brief-stranded-by-rename",
                "briefs filed under a slug their node no longer has",
                len(stranded),
                sorted(stranded)[:5],
                "python -m assimilator.synthesise (prunes on emit)",
            )
        )

    # 4. A proposal with no brief. Synthesise skips a node with no claims, but a
    # proposal requires claims - so the two disagreeing means one of them is wrong.
    have = {nid for nid in brief_node.values() if nid}
    missing = [n for n in proposals if n in live and n not in have]
    if missing:
        findings.append(
            Finding(
                "proposal-without-brief",
                "proposed pages that synthesise produced no brief for",
                len(missing),
                [
                    f"{live[n]} ({n[:8]})"
                    for n in sorted(missing, key=lambda x: live[x])[:5]
                ],
                "python -m assimilator.synthesise",
            )
        )

    # 5. A page built from a brief the graph has moved past. Not an error - it is
    # the signal to rebuild - but it is invisible without asking.
    if content_dir and content_dir.is_dir():
        current = {n[:-5]: d.get("brief_hash") for n, d in briefs.items()}
        stale = []
        for page in sorted(content_dir.glob("*/*.en.md")):
            slug = page.name[:-6]
            if slug not in current:
                continue
            m = _BUILT_FROM.search(page.read_text(errors="replace"))
            if m and m.group(1) != current[slug]:
                stale.append(slug)
        if stale:
            findings.append(
                Finding(
                    "page-trails-its-brief",
                    "published pages whose brief has changed since they were built",
                    len(stale),
                    sorted(stale)[:5],
                    "assembler --brief <slug> (costs a model call each)",
                )
            )
    return findings
