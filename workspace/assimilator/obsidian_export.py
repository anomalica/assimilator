"""Export the knowledge graph as an Obsidian-navigable markdown vault.

Structure:

    People/<First Last>.md     one file per person (sort_name gives surname sort)
    Organisations/<name>.md    one file per organisation
    Places/<Country, Region, City>.md  one file per place
    Events/<name>.md           one file per dated event
    Matters/<name>.md          one file per ongoing situation
    Objects/<name>.md          one file per specific physical thing
    Documents/<name>.md        one file per mentioned artefact (memo, report, ...)
    Sources/<date - title>.md  one file per source we ingested
    README.md                  what is here, how to navigate

Each entity file lists every claim that references it inline (with the source
record and other entities linked via `[[wikilinks]]`). The backlinks panel in
Obsidian is not required - the data is already on the page.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from assimilator.person_names import display_surname_first

_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIKILINK_FORBIDDEN = re.compile(r"[\[\]|#^]")

# Map raw node_type values to user-facing folder names (and singular labels).
# Note `record` and `document` both go in Documents/ - they're both "artefacts
# the world produced that get mentioned inside source material". The infra
# pass calls them `record` and the domain pass calls them `document` for
# historical reasons; we merge them in the vault.
_TYPE_FOLDERS: dict[str, tuple[str, str]] = {
    "person": ("People", "Person"),
    "organisation": ("Organisations", "Organisation"),
    "place": ("Places", "Place"),
    "event": ("Events", "Event"),
    "matter": ("Matters", "Matter"),
    "object": ("Objects", "Object"),
    "document": ("Documents", "Document"),
    "concept": ("Concepts", "Concept"),
    "record": ("Documents", "Document"),
}


def _safe_filename(name: str) -> str:
    cleaned = _FORBIDDEN.sub("_", name).strip().rstrip(".")
    return cleaned[:200] if cleaned else "_unnamed"


def _safe_wikilink(name: str) -> str:
    return _WIKILINK_FORBIDDEN.sub("", name).strip()


def _link(name: str | None) -> str:
    if not name:
        return "_"
    return f"[[{_safe_wikilink(name)}]]"


def _load_db(conn: sqlite3.Connection) -> dict:
    """Pull everything we need from one database into a single in-memory dict."""
    nodes = {}
    for nid, ntype, name, metadata in conn.execute(
        "SELECT id, node_type, name, metadata FROM nodes WHERE retired_at IS NULL"
    ):
        nodes[nid] = {
            "id": nid,
            "name": name,
            "node_type": ntype,
            "metadata": json.loads(metadata) if metadata else {},
            "aliases": [],
        }
    for alias, nid in conn.execute("SELECT alias, node_id FROM aliases"):
        if nid in nodes and alias != nodes[nid]["name"]:
            nodes[nid]["aliases"].append(alias)

    records = {}
    for rid, title, date, producer_id, reference in conn.execute(
        "SELECT id, title, date, producer_id, reference FROM records"
    ):
        records[rid] = {
            "id": rid,
            "title": title,
            "date": date,
            "producer_id": producer_id,
            "reference": reference,
        }

    claims = []
    claim_refs: dict[str, list[str]] = {}
    for row in conn.execute(
        """
        SELECT id, content, original_excerpt, claim_type, attestation,
               record_id, speaker_id, location_in_record, date, date_end, confidence
        FROM claims
        """
    ):
        cid = row[0]
        claims.append(
            {
                "id": cid,
                "content": row[1],
                "original_excerpt": row[2],
                "claim_type": row[3],
                "attestation": row[4],
                "record_id": row[5],
                "speaker_id": row[6],
                "location_in_record": row[7],
                "date": row[8],
                "date_end": row[9],
                "confidence": row[10],
            }
        )
        claim_refs[cid] = []
    for cid, nid in conn.execute("SELECT claim_id, node_id FROM claim_node_refs"):
        if cid in claim_refs:
            claim_refs[cid].append(nid)

    return {
        "nodes": nodes,
        "records": records,
        "claims": claims,
        "claim_refs": claim_refs,
    }


def _merge_by_name(*dbs: dict) -> dict:
    """Merge multi-DB load into a single (name, type)-keyed view.

    Two nodes are merged across DBs only if their name AND node_type match -
    so 'Apollo' the person and 'Apollo' the matter remain separate entries
    (they will end up in different per-type folders).

    Returns a dict with:
      - nodes_by_key: {(name, node_type): {name, node_type, aliases, ids}}
      - claims: [(db_idx, claim_dict)]
      - records: [(db_idx, record_dict)]
    """
    nodes_by_key: dict[tuple[str, str], dict] = {}
    id_to_name: dict[tuple[int, str], str] = {}
    for idx, db in enumerate(dbs):
        for nid, node in db["nodes"].items():
            name = node["name"]
            ntype = node["node_type"]
            key = (name, ntype)
            id_to_name[(idx, nid)] = name
            if key not in nodes_by_key:
                nodes_by_key[key] = {
                    "name": name,
                    "node_type": ntype,
                    "metadata": node.get("metadata") or {},
                    "aliases": set(node["aliases"]),
                    "ids": [],
                }
            else:
                nodes_by_key[key]["aliases"].update(node["aliases"])
            nodes_by_key[key]["ids"].append((idx, nid))

    claims = [(idx, claim) for idx, db in enumerate(dbs) for claim in db["claims"]]
    records = [
        (idx, rec) for idx, db in enumerate(dbs) for rec in db["records"].values()
    ]

    return {
        "nodes_by_key": nodes_by_key,
        "id_to_name": id_to_name,
        "claims": claims,
        "records": records,
        "dbs": dbs,
    }


def _format_claim_block(
    claim: dict,
    db_idx: int,
    merged: dict,
    *,
    show_speaker: bool = True,
) -> str:
    db = merged["dbs"][db_idx]
    refs = [
        merged["id_to_name"].get((db_idx, rid))
        for rid in db["claim_refs"].get(claim["id"], [])
    ]
    refs = [r for r in refs if r]
    speaker = (
        merged["id_to_name"].get((db_idx, claim["speaker_id"]))
        if claim["speaker_id"]
        else None
    )
    record = db["records"].get(claim["record_id"])
    record_title = record["title"] if record else None
    record_date = record["date"] if record else None

    lines: list[str] = []
    lines.append(f"### `{claim['claim_type']}` / `{claim['attestation']}`")
    lines.append("")
    lines.append(claim["content"])
    lines.append("")

    meta: list[str] = []
    if show_speaker and speaker:
        meta.append(f"- **Speaker:** {_link(speaker)}")
    if record_title:
        record_label = (
            f"{record_date} - {record_title}" if record_date else record_title
        )
        meta.append(f"- **Source:** {_link(record_label)}")
    if claim["date"]:
        date_str = claim["date"]
        if claim["date_end"] and claim["date_end"] != claim["date"]:
            date_str += f" to {claim['date_end']}"
        meta.append(f"- **Date:** {date_str}")
    if claim["location_in_record"]:
        meta.append(f"- **Location:** {claim['location_in_record']}")
    if refs:
        meta.append(f"- **Refs:** {', '.join(_link(r) for r in refs)}")
    if claim["confidence"] < 1.0:
        meta.append(f"- **Confidence:** {claim['confidence']:.2f}")
    if meta:
        lines.extend(meta)
        lines.append("")
    if claim["original_excerpt"]:
        for line in claim["original_excerpt"].splitlines() or [""]:
            lines.append(f"> {line}")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def _format_node_file(node: dict, merged: dict) -> str:
    folder_name, label = _TYPE_FOLDERS.get(
        node["node_type"], ("Other", node["node_type"].title())
    )

    fm = ["---", f"type: {node['node_type']}"]
    # Names are stored natural-order, so a People/ folder sorted by filename
    # sorts by forename. sort_name gives Obsidian the surname-first key to sort
    # and group on without renaming the file out of step with its wikilinks.
    sort_name = display_surname_first(node["name"], node.get("metadata"))
    if sort_name != node["name"]:
        fm.append(f"sort_name: {sort_name}")
    if node["aliases"]:
        fm.append("aliases:")
        for a in sorted(node["aliases"]):
            fm.append(f"  - {a}")
    fm.append("---")
    fm.append("")
    fm.append(f"# {node['name']}")
    fm.append("")
    fm.append(f"*{label}*")
    fm.append("")
    if node["aliases"]:
        fm.append(f"Also known as: {', '.join(sorted(node['aliases']))}")
        fm.append("")

    # Find claims that reference this node (or have it as speaker) across all DBs.
    own_ids = set(node["ids"])
    relevant: list[tuple[int, dict]] = []
    seen_content: set[tuple[str, str]] = set()
    for db_idx, claim in merged["claims"]:
        db = merged["dbs"][db_idx]
        is_speaker = claim["speaker_id"] and (db_idx, claim["speaker_id"]) in own_ids
        ref_match = any(
            (db_idx, rid) in own_ids for rid in db["claim_refs"].get(claim["id"], [])
        )
        if not (is_speaker or ref_match):
            continue
        # Dedupe across the two DBs (a claim can appear in both with the same text).
        key = (claim["record_id"], claim["content"])
        if key in seen_content:
            continue
        seen_content.add(key)
        relevant.append((db_idx, claim))

    # Sort: dated first chronologically, then undated.
    def _sort_key(item):
        _, c = item
        return (c["date"] is None, c["date"] or "", c["location_in_record"] or "")

    relevant.sort(key=_sort_key)

    fm.append(f"## Claims ({len(relevant)})")
    fm.append("")
    if not relevant:
        fm.append("_No claims reference this entity yet._")
        fm.append("")
    for db_idx, claim in relevant:
        fm.append(_format_claim_block(claim, db_idx, merged))
    return "\n".join(fm)


def _format_record_file(db_idx: int, record: dict, merged: dict) -> str:
    db = merged["dbs"][db_idx]
    producer_name = (
        merged["id_to_name"].get((db_idx, record["producer_id"]))
        if record["producer_id"]
        else None
    )
    claims_in_record = [c for c in db["claims"] if c["record_id"] == record["id"]]
    claims_in_record.sort(
        key=lambda c: (
            c["date"] is None,
            c["date"] or "",
            c["location_in_record"] or "",
        )
    )

    fm = ["---", "type: record"]
    if record["date"]:
        fm.append(f"date: {record['date']}")
    if producer_name:
        fm.append(f"producer: {producer_name}")
    if record["reference"]:
        fm.append(f"reference: {record['reference']}")
    fm.append("---")
    fm.append("")
    fm.append(f"# {record['title']}")
    fm.append("")
    if producer_name:
        fm.append(f"Producer: {_link(producer_name)}")
        fm.append("")
    fm.append(f"## Claims ({len(claims_in_record)})")
    fm.append("")
    for claim in claims_in_record:
        fm.append(_format_claim_block(claim, db_idx, merged))
    return "\n".join(fm)


_README = """\
# Anomalica Generated Vault

This vault is **auto-generated** from the assimilator knowledge graph. Do not edit
by hand - run `just vault` from `assimilator/` to regenerate.

## What is here

Each folder collects one kind of entity extracted from the source documents:

- **People/** - named individuals (witnesses, journalists, officials). Filenames
  are the person's name in natural order ("David Fravor"); each note carries a
  `sort_name` field ("Fravor, David") to sort or group the folder by surname.
- **Organisations/** - government bodies, military units, companies, programmes,
  panels, agencies.
- **Places/** - geographic locations. Filenames are "Country, Region, Specific"
  (e.g. "USA, Nevada, Area 51") so the folder groups places by country/region.
- **Events/** - discrete things that happened at a specific time.
- **Matters/** - ongoing situations, programmes, investigations, policy positions.
- **Objects/** - specific named physical things (craft, sensors, devices, samples,
  materials, weapons). NOT documents - those go in Documents.
- **Documents/** - written or recorded artefacts that get mentioned inside source
  material (memos, reports, letters, articles, papers, books, video footage,
  podcasts, briefings, statements, affidavits). The artefact itself is the node;
  claims about its content live on its page.
- **Sources/** - the documents we ingested into the digester (one per ingest).
  Each Source page lists every claim extracted from that document.

## How a claim is stored

Every assertion in a source document is broken into a **claim**. Each claim
appears inline on:

- the source **Record** file (with all other claims from that document)
- the page of every **entity** the claim references
- the page of the **speaker** if there is one

Each claim shows its type (`observation`, `testimony`, `hearsay`, `opinion`,
`measurement`, `administrative`), attestation level (`first_hand`,
`second_hand`, `third_hand`), source record, date, location in record, the
other entities it references, and the original wording from the document.
"""


def export_to_obsidian(
    out_dir: Path,
    domain_conn: sqlite3.Connection,
    infra_conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    """Write a navigable Obsidian vault to `out_dir`.

    Returns counts: {records, nodes, claims}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dbs = [_load_db(domain_conn)]
    if infra_conn is not None:
        dbs.append(_load_db(infra_conn))
    merged = _merge_by_name(*dbs)

    (out_dir / "README.md").write_text(_README)

    # Entity files grouped by node_type folder.
    used_filenames: dict[str, set[str]] = {}
    for node in merged["nodes_by_key"].values():
        folder, _label = _TYPE_FOLDERS.get(node["node_type"], ("Other", "Other"))
        folder_path = out_dir / folder
        folder_path.mkdir(exist_ok=True)
        used = used_filenames.setdefault(folder, set())
        fname = _safe_filename(node["name"]) + ".md"
        if fname in used:
            fname = f"{_safe_filename(node['name'])} ({node['node_type']}).md"
        used.add(fname)
        (folder_path / fname).write_text(_format_node_file(node, merged))

    # Source files (the records table - the documents we ingested).
    records_dir = out_dir / "Sources"
    records_dir.mkdir(exist_ok=True)
    seen_records: set[str] = set()
    record_count = 0
    for db_idx, record in merged["records"]:
        date_prefix = (record["date"] + " - ") if record["date"] else ""
        fname = _safe_filename(date_prefix + record["title"]) + ".md"
        if fname in seen_records:
            # Same record shows up in both DBs - the second copy gets a suffix
            # so it survives as an "(infrastructure)" companion.
            fname = (
                _safe_filename(date_prefix + record["title"] + " (infrastructure)")
                + ".md"
            )
        seen_records.add(fname)
        (records_dir / fname).write_text(_format_record_file(db_idx, record, merged))
        record_count += 1

    return {
        "records": record_count,
        "nodes": len(merged["nodes_by_key"]),
        "claims": sum(len(db["claims"]) for db in dbs),
    }
