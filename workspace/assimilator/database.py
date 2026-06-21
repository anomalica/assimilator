from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from anomalica_common.digest.models import Claim, ClaimRole, Node, Record

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL,
    retired_at TEXT
);

CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    reference TEXT,
    date TEXT,
    producer_id TEXT,
    content_hash TEXT,
    friendly_name TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_content_hash ON records(content_hash);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    original_excerpt TEXT,
    claim_type TEXT NOT NULL,
    attestation TEXT,
    record_id TEXT NOT NULL REFERENCES records(id),
    speaker_id TEXT,
    location_in_record TEXT,
    date TEXT,
    date_end TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    metadata TEXT,
    created_at TEXT NOT NULL,
    claim_role TEXT CHECK (claim_role IN (
        'official_explanation',
        'witness_testimony',
        'investigation_finding',
        'cover_up_evidence'
    )),
    -- Appended last to match the ALTER TABLE migration's column position, so
    -- fresh and upgraded databases share one column order (claims are read
    -- positionally in _row_to_claim). claim_hash is not part of the Claim model;
    -- it is the page-staleness fingerprint, read directly when needed.
    claim_hash TEXT
);

CREATE TABLE IF NOT EXISTS claim_node_refs (
    claim_id TEXT NOT NULL REFERENCES claims(id),
    node_id TEXT NOT NULL REFERENCES nodes(id),
    PRIMARY KEY (claim_id, node_id)
);

CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    PRIMARY KEY (alias, node_id)
);

CREATE TABLE IF NOT EXISTS corroborations (
    claim_a TEXT NOT NULL REFERENCES claims(id),
    claim_b TEXT NOT NULL REFERENCES claims(id),
    similarity REAL NOT NULL,
    PRIMARY KEY (claim_a, claim_b)
);

-- Node-merge curation log. DERIVED: re-populated from the durable curation
-- ledger on every rebuild+replay; the workbench reads it read-only for its
-- merged-groups + undo surface. `reversal` is the apply-time data needed to undo
-- a merge in the live DB (which claim/record refs moved); workbench ignores it.
CREATE TABLE IF NOT EXISTS node_merges (
    merge_id TEXT NOT NULL,
    survivor_id TEXT NOT NULL,
    victim_id TEXT NOT NULL,
    victim_prior_name TEXT,
    survivor_prior_name TEXT,
    canonical_name TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    undone_at TEXT,
    reversal TEXT,
    PRIMARY KEY (merge_id, victim_id)
);

-- "Not a duplicate" decisions. DERIVED from the durable rejections ledger
-- (curation/rejections.yaml), re-populated on rebuild. A rejected node set is
-- confirmed-distinct ground truth: propose-merges excludes it so a similar-named
-- but genuinely-different pair stops reappearing as a candidate. node_ids is the
-- JSON set; the workbench reads this read-only to filter its queue.
CREATE TABLE IF NOT EXISTS node_rejections (
    rejection_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    undone_at TEXT,
    PRIMARY KEY (rejection_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_claims_record ON claims(record_id);
CREATE INDEX IF NOT EXISTS idx_claims_speaker ON claims(speaker_id);
CREATE INDEX IF NOT EXISTS idx_claims_hash ON claims(claim_hash);
-- idx_claims_role is created in init_db after the claim_role column is
-- guaranteed to exist (it's added by ALTER TABLE on upgraded databases).
CREATE INDEX IF NOT EXISTS idx_claim_refs_node ON claim_node_refs(node_id);
CREATE INDEX IF NOT EXISTS idx_aliases_node ON aliases(node_id);
CREATE INDEX IF NOT EXISTS idx_corr_a ON corroborations(claim_a);
CREATE INDEX IF NOT EXISTS idx_corr_b ON corroborations(claim_b);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    # ADR 0028 migration: add claim_role column to pre-existing databases
    # BEFORE executescript runs, so the role index in SCHEMA can be created
    # safely on both fresh and upgraded databases.
    claims_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
    ).fetchone()
    if claims_exists:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}
        if "claim_role" not in cols:
            conn.execute(
                "ALTER TABLE claims ADD COLUMN claim_role TEXT CHECK ("
                "claim_role IN ('official_explanation', 'witness_testimony', "
                "'investigation_finding', 'cover_up_evidence'))"
            )
        # Page-staleness migration: add the claim_hash column to pre-existing
        # databases so the hash index in SCHEMA can be created on both fresh and
        # upgraded databases. Existing rows are backfilled by the
        # `backfill-claim-hashes` command (pure compute, no AI).
        if "claim_hash" not in cols:
            conn.execute("ALTER TABLE claims ADD COLUMN claim_hash TEXT")
    conn.executescript(SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_role ON claims(claim_role)")


def insert_node(conn: sqlite3.Connection, node: Node) -> Node:
    now = _now()
    metadata_json = json.dumps(node.metadata) if node.metadata else None
    conn.execute(
        "INSERT INTO nodes (id, node_type, name, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
        (node.id, node.node_type.value, node.name, metadata_json, now),
    )
    return node.model_copy(update={"created_at": datetime.fromisoformat(now)})


def get_node(conn: sqlite3.Connection, node_id: str) -> Node | None:
    row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        return None
    return _row_to_node(row)


def get_nodes(conn: sqlite3.Connection, node_type: str | None = None) -> list[Node]:
    if node_type:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE node_type = ? AND retired_at IS NULL",
            (node_type,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM nodes WHERE retired_at IS NULL").fetchall()
    return [_row_to_node(row) for row in rows]


def find_node_by_name(
    conn: sqlite3.Connection, name: str, node_type: str | None = None
) -> Node | None:
    if node_type:
        row = conn.execute(
            "SELECT * FROM nodes WHERE name = ? AND node_type = ? AND retired_at IS NULL",
            (name, node_type),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM nodes WHERE name = ? AND retired_at IS NULL", (name,)
        ).fetchone()
    if row:
        return _row_to_node(row)
    if node_type:
        alias_row = conn.execute(
            "SELECT n.* FROM nodes n JOIN aliases a ON a.node_id = n.id "
            "WHERE a.alias = ? AND n.node_type = ? AND n.retired_at IS NULL",
            (name, node_type),
        ).fetchone()
    else:
        alias_row = conn.execute(
            "SELECT n.* FROM nodes n JOIN aliases a ON a.node_id = n.id "
            "WHERE a.alias = ? AND n.retired_at IS NULL",
            (name,),
        ).fetchone()
    if alias_row:
        return _row_to_node(alias_row)
    return None


def insert_alias(conn: sqlite3.Connection, alias: str, node_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO aliases (alias, node_id) VALUES (?, ?)",
        (alias, node_id),
    )


def insert_record(conn: sqlite3.Connection, record: Record) -> Record:
    now = _now()
    metadata_json = json.dumps(record.metadata) if record.metadata else None
    conn.execute(
        "INSERT INTO records "
        "(id, title, reference, date, producer_id, content_hash, friendly_name, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.id,
            record.title,
            record.reference,
            record.date,
            record.producer_id,
            record.content_hash,
            record.friendly_name,
            metadata_json,
            now,
        ),
    )
    return record.model_copy(update={"created_at": datetime.fromisoformat(now)})


def get_record(conn: sqlite3.Connection, record_id: str) -> Record | None:
    row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def get_record_by_title(conn: sqlite3.Connection, title: str) -> Record | None:
    row = conn.execute("SELECT * FROM records WHERE title = ?", (title,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def insert_claim(
    conn: sqlite3.Connection,
    claim: Claim,
    claim_hash: str | None = None,
    created_at: str | None = None,
) -> Claim:
    # created_at is overridable so a carried-forward claim keeps its original
    # timestamp across a re-import (the row is rewritten, not first-seen).
    now = created_at or _now()
    metadata_json = json.dumps(claim.metadata) if claim.metadata else None
    conn.execute(
        "INSERT INTO claims (id, content, original_excerpt, claim_type, attestation, record_id, speaker_id, "
        "location_in_record, date, date_end, claim_hash, confidence, metadata, created_at, claim_role) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            claim.id,
            claim.content,
            claim.original_excerpt,
            claim.claim_type.value,
            claim.attestation.value if claim.attestation else None,
            claim.record_id,
            claim.speaker_id,
            claim.location_in_record,
            claim.date,
            claim.date_end,
            claim_hash,
            claim.confidence,
            metadata_json,
            now,
            claim.claim_role.value if claim.claim_role else None,
        ),
    )
    for node_id in claim.node_references:
        conn.execute(
            "INSERT OR IGNORE INTO claim_node_refs (claim_id, node_id) VALUES (?, ?)",
            (claim.id, node_id),
        )
    return claim.model_copy(update={"created_at": datetime.fromisoformat(now)})


def get_record_claim_hashes(
    conn: sqlite3.Connection, record_id: str
) -> dict[str, list[tuple[str, str]]]:
    """Map each claim_hash to the (claim_id, created_at) rows carrying it for a
    record. The value is a list so duplicate-hash claims within one record are
    handled (matched one-for-one on re-import). Rows with a null claim_hash (not
    yet backfilled) are skipped - they reconcile as if absent."""
    rows = conn.execute(
        "SELECT claim_hash, id, created_at FROM claims "
        "WHERE record_id = ? AND claim_hash IS NOT NULL",
        (record_id,),
    ).fetchall()
    out: dict[str, list[tuple[str, str]]] = {}
    for h, cid, created in rows:
        out.setdefault(h, []).append((cid, created))
    return out


def delete_claim(conn: sqlite3.Connection, claim_id: str) -> None:
    """Remove a claim and its node references (used when a re-import drops a
    claim that no longer appears in the record's digest)."""
    conn.execute("DELETE FROM claim_node_refs WHERE claim_id = ?", (claim_id,))
    conn.execute(
        "DELETE FROM corroborations WHERE claim_a = ? OR claim_b = ?",
        (claim_id, claim_id),
    )
    conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))


def update_claim_hash(conn: sqlite3.Connection, claim_id: str, claim_hash: str) -> None:
    """Set the claim_hash for an existing row (used by the backfill command)."""
    conn.execute(
        "UPDATE claims SET claim_hash = ? WHERE id = ?", (claim_hash, claim_id)
    )


def get_claims_for_node(conn: sqlite3.Connection, node_id: str) -> list[Claim]:
    rows = conn.execute(
        "SELECT c.* FROM claims c "
        "JOIN claim_node_refs r ON r.claim_id = c.id "
        "WHERE r.node_id = ?",
        (node_id,),
    ).fetchall()
    claims = []
    for row in rows:
        claim = _row_to_claim(row)
        claim.node_references = _get_claim_refs(conn, claim.id)
        claims.append(claim)
    return claims


def get_claims_for_record(conn: sqlite3.Connection, record_id: str) -> list[Claim]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE record_id = ?", (record_id,)
    ).fetchall()
    claims = []
    for row in rows:
        claim = _row_to_claim(row)
        claim.node_references = _get_claim_refs(conn, claim.id)
        claims.append(claim)
    return claims


def insert_corroboration(
    conn: sqlite3.Connection, claim_a: str, claim_b: str, similarity: float
) -> None:
    a, b = sorted([claim_a, claim_b])
    conn.execute(
        "INSERT OR IGNORE INTO corroborations (claim_a, claim_b, similarity) VALUES (?, ?, ?)",
        (a, b, similarity),
    )


def get_corroborations(
    conn: sqlite3.Connection, claim_id: str
) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT claim_b, similarity FROM corroborations WHERE claim_a = ? "
        "UNION SELECT claim_a, similarity FROM corroborations WHERE claim_b = ?",
        (claim_id, claim_id),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def get_independent_source_count(conn: sqlite3.Connection, claim_id: str) -> int:
    """Count independent sources corroborating a claim.

    Two claims genuinely corroborate each other only if their provenance
    chains do not share a common root. If two claims have the same speaker,
    they originate from the same person's testimony and count as one source
    regardless of how many records they appear in.

    Claims with no speaker are grouped by record (each record is a source).
    """
    corroborated = get_corroborations(conn, claim_id)

    # Collect all claim IDs in the corroboration group (including self)
    all_claim_ids = [claim_id] + [cid for cid, _ in corroborated]

    # Group by provenance root: speaker if known, otherwise record
    provenance_roots = set()
    for cid in all_claim_ids:
        row = conn.execute(
            "SELECT speaker_id, record_id FROM claims WHERE id = ?", (cid,)
        ).fetchone()
        if row is None:
            continue
        speaker_id, record_id = row
        if speaker_id:
            provenance_roots.add(("speaker", speaker_id))
        else:
            provenance_roots.add(("record", record_id))

    return len(provenance_roots)


def get_stats(conn: sqlite3.Connection) -> dict:
    stats = {}
    for table in (
        "nodes",
        "records",
        "claims",
        "claim_node_refs",
        "aliases",
        "corroborations",
    ):
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
        stats[table] = row[0]
    active_nodes = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE retired_at IS NULL"
    ).fetchone()
    stats["active_nodes"] = active_nodes[0]
    type_counts = conn.execute(
        "SELECT node_type, COUNT(*) FROM nodes WHERE retired_at IS NULL GROUP BY node_type"
    ).fetchall()
    stats["by_type"] = {row[0]: row[1] for row in type_counts}
    return stats


def _get_claim_refs(conn: sqlite3.Connection, claim_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT node_id FROM claim_node_refs WHERE claim_id = ?", (claim_id,)
    ).fetchall()
    return [row[0] for row in rows]


def _row_to_node(row: tuple) -> Node:
    return Node(
        id=row[0],
        node_type=row[1],
        name=row[2],
        metadata=json.loads(row[3]) if row[3] else None,
        created_at=datetime.fromisoformat(row[4]),
        retired_at=datetime.fromisoformat(row[5]) if row[5] else None,
    )


def _row_to_record(row: tuple) -> Record:
    # Column order: id, title, reference, date, producer_id, content_hash,
    # friendly_name, metadata, created_at
    return Record(
        id=row[0],
        title=row[1],
        reference=row[2],
        date=row[3],
        producer_id=row[4],
        content_hash=row[5],
        friendly_name=row[6],
        metadata=json.loads(row[7]) if row[7] else None,
        created_at=datetime.fromisoformat(row[8]),
    )


def _row_to_claim(row: tuple) -> Claim:
    return Claim(
        id=row[0],
        content=row[1],
        original_excerpt=row[2],
        claim_type=row[3],
        attestation=row[4],
        record_id=row[5],
        speaker_id=row[6],
        location_in_record=row[7],
        date=row[8],
        date_end=row[9],
        confidence=row[10],
        metadata=json.loads(row[11]) if row[11] else None,
        created_at=datetime.fromisoformat(row[12]),
        claim_role=ClaimRole(row[13]) if len(row) > 13 and row[13] else None,
    )
