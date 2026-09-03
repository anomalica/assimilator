"""Page composition: one page over several nodes.

UFO and UAP are the same phenomenon under two vocabularies, and merging the
nodes would be wrong - they share 26 claims of 2,068, so a merge destroys which
word each source used and unions nearly disjoint material under one name. The
nodes stay separate; ONE PAGE covers both and unions their claims at
generation. A reader gets one destination, the graph keeps the distinction.

That makes a page a first-class thing - a name, a slug, and a list of member
nodes - rather than a node that happens to earn a page. The composition is a
human's editorial judgement, so it lives in the curation ledger and is replayed
after every rebuild, like a merge, a veto or a tag.

Keyed on natural identity, never ids: node ids are minted per extraction, so an
id-keyed member replays onto nothing. A member resolves by name then aliases
within its type on the exact and declared tiers - the merge replay rule - so a
member renamed or merged away still resolves. A member that no longer resolves
is DROPPED and reported, leaving the page composed of the rest: one vanished
member must not cost the whole page.

The op suppresses separate page proposals for its members (propose_pages reads
member_node_ids), with the proposal's reason recorded as "member of page X".
No veto is written, so decomposing needs nothing undone.

A member that already HAS a published page is a live problem the composition
creates: one subject would have two pages, and an op that looks applied while
the loser stays up is the shape of every quiet failure here. So composing also
SUPERSEDES each member page whose slug is not the composed page's, recording it
in superseded_pages for the assembler to retire on the path it uses for a veto.
Naming the page after its heaviest member therefore leaves the biggest page
untouched and retires only the smaller ones.

NO SLUG REDIRECTS. Mark, 2026-09-03: nobody uses the site yet, so a member's
old slug simply stops existing rather than redirecting.

NO SESSION COMPOSES A PAGE, on the same rule as a merge (Mark, 2026-09-03): a
composition decides what a reader sees, so it applies only with the workbench's
confirmation record. Without one, a session writes a PROPOSAL for a reviewer.
The first composition, over UFO and UAP, predates the rule and stands - Mark
authorised that pair by name.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from anomalica_common.slug import node_slug, section_for, slugify

from assimilator.matching import match_node

# Compositions written before the rule landed still apply; a later one needs the
# block. Same shape as the merge guard, for the same reason: not a security
# boundary, a guard against habit and mistake.
CONFIRMATION_REQUIRED_FROM = "2026-09-03T04:30:00Z"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pages_ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # .../anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "pages.yaml"


def _append(entry: dict) -> None:
    path = pages_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


def read_pages() -> list[dict]:
    path = pages_ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def append_compose_entry(
    name: str,
    node_type: str,
    members: list[dict],
    slug: str | None = None,
    page_id: str | None = None,
    created_at: str | None = None,
    created_by: str | None = None,
    note: str | None = None,
    confirmation: dict | None = None,
) -> dict:
    """Append a composition to the durable ledger. Refuses without a
    confirmation block: an unconfirmed composition is a proposal for a
    reviewer, not a ledger entry (see propose_composition)."""
    if not confirmation:
        raise ValueError(
            "a page composition is written to the ledger only with a reviewer's "
            "confirmation; without one it is a proposal (propose_composition)"
        )
    entry = {
        "op": "compose",
        "page_id": page_id or str(uuid.uuid4()),
        "at": created_at or _now(),
        "by": created_by,
        "confirmation": dict(confirmation),
        "page": {"name": name, "slug": slug or slugify(name), "node_type": node_type},
        "members": [
            {
                "name": m["name"],
                "node_type": m.get("node_type") or node_type,
                "prior_names": list(m.get("prior_names") or []),
            }
            for m in members
        ],
        "note": note,
    }
    _append(entry)
    return entry


def append_decompose_entry(page_id: str, created_by: str | None = None) -> dict:
    entry = {"op": "decompose", "page_id": page_id, "at": _now(), "by": created_by}
    _append(entry)
    return entry


def confirmed(entry: dict) -> bool:
    """Whether a ledger composition may be applied: it carries a confirmation
    block, or it predates the rule."""
    block = entry.get("confirmation")
    if isinstance(block, dict) and block.get("by"):
        return True
    return str(entry.get("at") or "") < CONFIRMATION_REQUIRED_FROM


def proposals_dir() -> Path:
    root = Path(__file__).resolve().parents[3]  # .../anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "compose-proposals"


def propose_composition(
    name: str,
    node_type: str,
    members: list[dict],
    proposed_by: str | None,
    slug: str | None = None,
    note: str | None = None,
) -> Path:
    """What an unconfirmed composition becomes: one file a reviewer decides on,
    beside the rename proposals and read the same way."""
    import json

    directory = proposals_dir()
    directory.mkdir(parents=True, exist_ok=True)
    proposal = {
        "id": str(uuid.uuid4()),
        "page": {"name": name, "slug": slug or slugify(name), "node_type": node_type},
        "members": members,
        "note": note,
        "proposed_by": proposed_by,
        "proposed_at": _now(),
    }
    path = (
        directory / f"{proposal['proposed_at'].replace(':', '-')}-{slugify(name)}.json"
    )
    path.write_text(json.dumps(proposal, indent=1, ensure_ascii=False))
    return path


def _resolve_member(conn: sqlite3.Connection, member: dict) -> str | None:
    node_type = member.get("node_type")
    for name in [member.get("name"), *(member.get("prior_names") or [])]:
        if not name:
            continue
        m = match_node(conn, name, node_type)
        if m and m[1] != "fuzzy":
            return m[0]
    return None


def apply_pages(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Rebuild the derived pages tables from the ledger. Idempotent: the tables
    are wiped and repopulated, so this is both the live apply and the
    post-rebuild replay.

    Counted and reported: composed, dropped members, and pages lost entirely."""
    log = on_progress or (lambda _: None)
    entries = read_pages()
    undone = {e["page_id"] for e in entries if e.get("op") == "decompose"}
    conn.execute("DELETE FROM page_members")
    conn.execute("DELETE FROM pages")
    conn.execute("DELETE FROM superseded_pages")
    composed = dropped = lost = superseded = unconfirmed = 0
    for e in entries:
        if e.get("op") != "compose" or e["page_id"] in undone:
            continue
        if not confirmed(e):
            log(
                f"  compose UNCONFIRMED {e['page_id']}: no reviewer confirmation "
                f"and dated after the rule - not applied "
                f"({(e.get('page') or {}).get('name')!r})"
            )
            unconfirmed += 1
            continue
        page = e.get("page") or {}
        resolved: list[str] = []
        for member in e.get("members") or []:
            node_id = _resolve_member(conn, member)
            if node_id is None:
                log(
                    f"  page {page.get('name')!r}: member {member.get('name')!r} "
                    f"({member.get('node_type')}) no longer resolves - the page is "
                    f"composed of the rest"
                )
                dropped += 1
                continue
            if node_id not in resolved:
                resolved.append(node_id)
        if not resolved:
            log(
                f"  ERROR page LOST {e['page_id']}: no member of "
                f"{page.get('name')!r} is in the graph"
            )
            lost += 1
            continue
        conn.execute(
            "INSERT INTO pages (page_id, name, slug, node_type, created_at, "
            "created_by, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                e["page_id"],
                page.get("name"),
                page.get("slug") or slugify(page.get("name") or ""),
                page.get("node_type"),
                e.get("at") or _now(),
                e.get("by"),
                e.get("note"),
            ),
        )
        page_slug = page.get("slug") or slugify(page.get("name") or "")
        page_section = section_for(page.get("node_type") or "")
        for position, node_id in enumerate(resolved):
            conn.execute(
                "INSERT INTO page_members (page_id, node_id, position) VALUES (?, ?, ?)",
                (e["page_id"], node_id, position),
            )
            row = conn.execute(
                "SELECT name, node_type, metadata FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is None:
                continue
            member_slug = node_slug(row[0], row[2])
            member_section = section_for(row[1])
            if (member_section, member_slug) == (page_section, page_slug):
                continue
            conn.execute(
                "INSERT OR REPLACE INTO superseded_pages (section, slug, page_id, "
                "node_id, reason) VALUES (?, ?, ?, ?, ?)",
                (
                    member_section,
                    member_slug,
                    e["page_id"],
                    node_id,
                    f"composed into {page.get('name')!r}",
                ),
            )
            superseded += 1
        composed += 1
    conn.commit()
    log(
        f"Composed {composed} pages ({superseded} member pages superseded, "
        f"{dropped} members dropped, {lost} lost, {unconfirmed} unconfirmed)"
    )
    return {
        "composed": composed,
        "superseded": superseded,
        "dropped": dropped,
        "lost": lost,
        "unconfirmed": unconfirmed,
    }


def superseded(conn: sqlite3.Connection) -> list[dict]:
    """Published pages a composition supersedes: <section>/<slug> and why. The
    assembler retires these on the path it uses for a vetoed page."""
    try:
        rows = conn.execute(
            "SELECT section, slug, page_id, node_id, reason FROM superseded_pages "
            "ORDER BY section, slug"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "section": r[0],
            "slug": r[1],
            "page": f"{r[0]}/{r[1]}",
            "page_id": r[2],
            "node_id": r[3],
            "reason": r[4],
        }
        for r in rows
    ]


replay_pages = apply_pages


def composed_pages(conn: sqlite3.Connection) -> list[dict]:
    """Every composed page with its member node ids, in member order."""
    try:
        rows = conn.execute(
            "SELECT p.page_id, p.name, p.slug, p.node_type, m.node_id "
            "FROM pages p JOIN page_members m ON m.page_id = p.page_id "
            "ORDER BY p.name, m.position"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: dict[str, dict] = {}
    for page_id, name, slug, node_type, node_id in rows:
        page = out.setdefault(
            page_id,
            {
                "page_id": page_id,
                "name": name,
                "slug": slug,
                "node_type": node_type,
                "node_ids": [],
            },
        )
        page["node_ids"].append(node_id)
    return list(out.values())


def member_node_ids(conn: sqlite3.Connection) -> dict[str, str]:
    """Member node id -> the name of the page covering it. A member does not
    earn a page of its own; the covering page is proposed instead."""
    try:
        rows = conn.execute(
            "SELECT m.node_id, p.name FROM page_members m "
            "JOIN pages p ON p.page_id = m.page_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r[0]: r[1] for r in rows}
