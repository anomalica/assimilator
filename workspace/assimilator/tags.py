"""Record tags: a human asserts that a record is ABOUT a node.

The pipeline links two records only through a named entity they share (Nimitz,
Elizondo, Grusch). Two records about the same unnamed thing - the same
operation under two agency-style names, the same encounter told twice with no
shared wording - stay apart: claim similarity finds nothing, name matching
cannot separate a real pair from house-style noise (5.2% of all node pairs
score above the one measured true pair), and the relate pass costs a judge call
per pair and is wrong one time in fifteen. The residue is a link a person has
to assert. That is a tag.

A tag is one row in record_nodes ("this record is about this node") sourced
from the curation ledger instead of from import, plus a row in record_tags so
it stays distinguishable from a link the importer made. RECORD-LEVEL ONLY: a
tag attaches no claim to the node, so it feeds nothing that counts claims - not
the page gate, not scoring, not corroboration, not the brief. Span tags (a
selection resolving to the claims whose locations overlap it) are a later op on
the same machinery; the shape leaves room and does not build it.

Keyed on natural identity, never ids: link_record_nodes deletes and re-inserts a
record's rows on every import and node ids are minted per extraction, so an
id-keyed tag would replay onto nothing. The node resolves by name then
prior_names within node_type on the exact and declared tiers only - the merge
replay rule - so a node merged away (its name is the survivor's alias) or
renamed (the old name is kept as an alias) still receives the tag - the alias
table is the record of "this used to be called that", which is what makes a
tag written before a rename replay correctly after it. A TOPIC name with no
node CREATES one: a topic is a subject heading a person is entitled to name
into existence, and that is how a seeded topic becomes a graph node the first
time a record is tagged with it. Any other type is never minted from a tag box
- a person, organisation, event, place or document earns its node by being
extracted from a source, and a misspelt name must not silently become a second
person - so a non-topic name that matches nothing stays PENDING.

The record resolves by content_hash (prefixed or bare). Only 109 of the 319
records in the ingests store had a graph row on 2026-09-03, because a record
gets one by being digested; a reviewer tagging a record the pipeline has not
caught up with is asserting a judgement that is right NOW, so that tag stays
PENDING and lands on the apply after the record is digested. LOST is reserved
for an entry that will never resolve - a malformed one - and is logged as an
ERROR. Every outcome is readable by tag_id in record_tags, because a reviewer
who asserted a link is owed an answer either way.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from anomalica_common.digest.models import Node, NodeType
from anomalica_common.titles import capitalise_first

from assimilator.database import insert_node
from assimilator.matching import match_node


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tags_ledger_path() -> Path:
    root = Path(__file__).resolve().parents[3]  # .../anomalica
    base = Path(os.environ.get("ANOMALICA_CURATION_DIR", str(root / "curation")))
    return base / "tags.yaml"


def _append(entry: dict) -> None:
    path = tags_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))


def read_tags() -> list[dict]:
    path = tags_ledger_path()
    if not path.is_file():
        return []
    return [e for e in yaml.safe_load_all(path.read_text()) if e]


def append_tag_entry(
    node_name: str,
    node_type: str,
    content_hash: str,
    tag_id: str | None = None,
    created_at: str | None = None,
    created_by: str | None = None,
    note: str | None = None,
    title: str | None = None,
    prior_names: list[str] | None = None,
) -> dict:
    entry = {
        "op": "tag",
        "tag_id": tag_id or str(uuid.uuid4()),
        "at": created_at or _now(),
        "by": created_by,
        "node": {
            "name": node_name,
            "node_type": node_type,
            "prior_names": list(prior_names or []),
        },
        "record": {"content_hash": content_hash, "title": title},
        "note": note,
    }
    _append(entry)
    return entry


def append_untag_entry(tag_id: str, created_by: str | None = None) -> dict:
    entry = {"op": "untag", "tag_id": tag_id, "at": _now(), "by": created_by}
    _append(entry)
    return entry


def _resolve_node(conn: sqlite3.Connection, nat: dict) -> str | None:
    """Name then prior_names within node_type; exact and declared tiers only.
    No fuzzy tier, for the reason merge replay gives: a wrong guess applies a
    human's decision to a node they never looked at, unreviewably."""
    node_type = nat.get("node_type")
    for name in [nat.get("name"), *(nat.get("prior_names") or [])]:
        if not name:
            continue
        m = match_node(conn, name, node_type)
        if m and m[1] != "fuzzy":
            return m[0]
    return None


def _create_topic(conn: sqlite3.Connection, name: str) -> str:
    """A topic minted from a tag gets the entry-time naming rule: stripped, and
    a leading capital (22 lowercase topic names were corrected through the
    rename ledger on 2026-09-03; a tag box must not reintroduce them)."""
    node = insert_node(
        conn, Node(name=capitalise_first(name.strip()), node_type=NodeType.topic)
    )
    return node.id


def _resolve_record(conn: sqlite3.Connection, content_hash: str) -> str | None:
    h = (content_hash or "").strip()
    if not h:
        return None
    candidates = {h, h.replace("sha256:", ""), f"sha256:{h.replace('sha256:', '')}"}
    for c in candidates:
        row = conn.execute(
            "SELECT id FROM records WHERE content_hash = ?", (c,)
        ).fetchone()
        if row:
            return row[0]
    return None


def _import_links(conn: sqlite3.Connection, record_id: str, node_id: str) -> bool:
    """Whether the importer would link this record to this node on its own:
    a claim of the record references the node, or the node speaks one."""
    return (
        conn.execute(
            "SELECT 1 FROM claims c WHERE c.record_id = ? AND (c.speaker_id = ? OR "
            "c.id IN (SELECT claim_id FROM claim_node_refs WHERE node_id = ?)) LIMIT 1",
            (record_id, node_id, node_id),
        ).fetchone()
        is not None
    )


def apply_tags(conn: sqlite3.Connection, on_progress=None) -> dict:
    """Land every ledger tag not yet applied; withdraw the untagged ones.

    One function for both the live apply and the post-rebuild replay: the table
    it keys on is derived, so after a rebuild it is empty and every tag lands
    again, and on a live run only the new and the still-pending entries do.
    Idempotent on tag_id.

    Counted and reported, because a rebuild that loses a human decision must
    say so: applied, created (a topic did not exist and was made), pending (the
    record or a non-topic node does not resolve yet), withdrawn, lost (malformed
    - logged as an ERROR).
    """
    log = on_progress or (lambda _: None)
    entries = read_tags()
    undone = {e["tag_id"] for e in entries if e.get("op") == "untag"}
    rows = {
        r[0]: {"status": r[1], "record_id": r[2], "node_id": r[3], "undone_at": r[4]}
        for r in conn.execute(
            "SELECT tag_id, status, record_id, node_id, undone_at FROM record_tags"
        )
    }
    applied = created = pending = withdrawn = lost = 0
    for e in entries:
        if e.get("op") != "tag":
            continue
        tag_id = e["tag_id"]
        row = rows.get(tag_id)
        if tag_id in undone:
            if row is None:
                continue
            if row["undone_at"] is None:
                conn.execute(
                    "UPDATE record_tags SET undone_at = ? WHERE tag_id = ?",
                    (_now(), tag_id),
                )
                if row["status"] == "applied" and not _import_links(
                    conn, row["record_id"], row["node_id"]
                ):
                    conn.execute(
                        "DELETE FROM record_nodes WHERE record_id = ? AND node_id = ?",
                        (row["record_id"], row["node_id"]),
                    )
                withdrawn += 1
            continue
        if row is not None and row["status"] != "pending":
            continue
        nat = e.get("node") or {}
        name = (nat.get("name") or "").strip()
        content_hash = ((e.get("record") or {}).get("content_hash") or "").strip()
        malformed = None
        if not name:
            malformed = "empty node name"
        elif not content_hash:
            malformed = "empty content hash"
        else:
            try:
                node_type = NodeType(nat.get("node_type")).value
            except ValueError:
                malformed = f"{nat.get('node_type')!r} is not a node type"
        if malformed:
            log(f"  ERROR tag LOST {tag_id}: {malformed} - it can never apply")
            _upsert(conn, tag_id, e, "lost", reason=malformed)
            lost += 1
            continue
        record_id = _resolve_record(conn, content_hash)
        node_id = _resolve_node(conn, nat)
        if record_id is None:
            reason = f"no record with hash {content_hash} (not digested yet?)"
        elif node_id is None and node_type != "topic":
            reason = (
                f"no {node_type} named {name!r}; only a topic is created from a tag"
            )
        else:
            reason = None
        if reason:
            _upsert(conn, tag_id, e, "pending", reason=reason)
            pending += 1
            continue
        if node_id is None:
            node_id = _create_topic(conn, name)
            created += 1
        conn.execute(
            "INSERT OR IGNORE INTO record_nodes (record_id, node_id) VALUES (?, ?)",
            (record_id, node_id),
        )
        _upsert(conn, tag_id, e, "applied", record_id=record_id, node_id=node_id)
        applied += 1
    conn.commit()
    log(
        f"Applied {applied} tags ({created} topics created, {pending} pending, "
        f"{withdrawn} withdrawn, {lost} lost)"
    )
    return {
        "applied": applied,
        "created": created,
        "pending": pending,
        "withdrawn": withdrawn,
        "lost": lost,
    }


def _upsert(
    conn: sqlite3.Connection,
    tag_id: str,
    entry: dict,
    status: str,
    record_id: str | None = None,
    node_id: str | None = None,
    reason: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO record_tags (tag_id, status, record_id, node_id, created_at, "
        "created_by, note, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(tag_id) DO UPDATE SET status = excluded.status, "
        "record_id = excluded.record_id, node_id = excluded.node_id, "
        "reason = excluded.reason",
        (
            tag_id,
            status,
            record_id,
            node_id,
            entry.get("at") or _now(),
            entry.get("by"),
            entry.get("note"),
            reason,
        ),
    )


replay_tags = apply_tags


def tag_record(
    conn: sqlite3.Connection,
    node_name: str,
    node_type: str,
    content_hash: str,
    created_by: str | None = None,
    note: str | None = None,
) -> dict:
    """Operator path: write the ledger entry and land it live in one step."""
    title = None
    row = conn.execute(
        "SELECT title FROM records WHERE content_hash IN (?, ?)",
        (content_hash, content_hash.replace("sha256:", "")),
    ).fetchone()
    if row:
        title = row[0]
    entry = append_tag_entry(
        node_name,
        node_type,
        content_hash,
        created_by=created_by,
        note=note,
        title=title,
    )
    apply_tags(conn)
    return entry
