from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from assimilator.embed_batches import forget_embeddings
from anomalica_common.digest.models import (
    Claim,
    ClaimRole,
    Node,
    OriginKind,
    ProvenanceChain,
    Record,
)

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
    created_at TEXT NOT NULL,
    -- The WORK this record is a manifestation of. Records are addressed by exact
    -- bytes, so one work becomes several records on any re-download, re-export or
    -- edition change - and counting distinct records then counts one work as
    -- several independent sources. Defaults to the record's own id (one record,
    -- one work); `link-works` collapses detected duplicates onto a shared id.
    -- Everything that counts "sources" must count THIS, not record_id.
    work_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_content_hash ON records(content_hash);
CREATE INDEX IF NOT EXISTS idx_records_work ON records(work_id);

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
    claim_hash TEXT,
    -- The claim's provenance chain (ADR 0044): who originally asserted it and how
    -- it reached the speaker. This is what corroboration independence keys on - two
    -- claims sharing a chain root are ONE source, not two. NULL origin_kind means
    -- the chain was NOT CAPTURED (a pre-0044 digest), never that there is no chain;
    -- independence treats such claims conservatively rather than as independent.
    origin_kind TEXT,
    origin TEXT,
    relay TEXT,  -- JSON array, ordered origin -> speaker
    -- The digester's entailment check: does the source excerpt (premise) support
    -- the claim as written (hypothesis)? label entails|neutral|contradicts, score
    -- = the probability of THAT label, model = the checker. NULL label means not
    -- assessed - a digest that predates the check, or a claim with no excerpt -
    -- never neutral. Stored and surfaced; the evidence score that will use it is
    -- not defined yet (Mark's), so nothing weights or hides a claim on it.
    entailment_label TEXT,
    entailment_score REAL,
    entailment_model TEXT,
    -- Which text produced the label: 'quote' (the excerpt alone) or 'window'
    -- (the record text around it, tried when the quote alone is neutral). An
    -- entails-by-window is the weaker verdict, so the entailed fraction is
    -- always reported split by premise, never as one number.
    entailment_premise TEXT
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

-- EXPERIMENTAL (2026-09-03). Two records judged to refer to the same specific
-- subject - one incident, operation, programme, document or investigation -
-- by a model reading both claim lists (relate.py). Derived and rebuildable,
-- like corroborations and page_proposals: nothing in the graph depends on it.
-- Unrelated verdicts are stored too, so a pair is judged once and precision
-- can be measured later. record_a < record_b.
CREATE TABLE IF NOT EXISTS record_relations (
    record_a TEXT NOT NULL,
    record_b TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('same_subject', 'possibly_related', 'unrelated')),
    shared_subject TEXT,
    reason TEXT,
    links TEXT,
    model TEXT,
    prompt_sha TEXT,
    judged_at TEXT NOT NULL,
    PRIMARY KEY (record_a, record_b)
);

CREATE TABLE IF NOT EXISTS corroborations (
    claim_a TEXT NOT NULL REFERENCES claims(id),
    claim_b TEXT NOT NULL REFERENCES claims(id),
    similarity REAL NOT NULL,
    PRIMARY KEY (claim_a, claim_b)
);

-- A pair the model ADJUDICATED AND REJECTED. Without this a rejection is invisible
-- and every subsequent run re-verifies it and re-pays for it: 26 of the first 86
-- pairs were rejected, so an automated lane would re-buy the same 26 verdicts on
-- every pass forever. Mirrors node_rejections, which exists for the same reason.
--
-- Kept OUT of the corroborations table deliberately: consumers read that table as
-- "pairs that corroborate" (scoring, synthesise, the scheduler's count), and a
-- verdict column would make every one of them wrong by default.
CREATE TABLE IF NOT EXISTS corroboration_rejections (
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    similarity REAL NOT NULL,
    model TEXT,
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (claim_a, claim_b)
);

-- Node renames PROPOSED by a reviewer, for the assimilator to apply.
--
-- PROPOSALS ARRIVE AS FILES, NOT AS ROWS. The workbench opens knowledge.db
-- read-only by design and declined a writable handle for this one table, which
-- is the right call and not a technicality: the read-only connection is the
-- reason the workbench cannot corrupt the graph, and "just one more table"
-- would be a precedent where there is currently a boundary. So a reviewer
-- writes JSON into curation/rename-proposals/ and this table is where the
-- assimilator records what it did with each one.
--
-- The apply path still goes through rename_node: a name written straight into
-- the database does not survive a rebuild, because the graph is re-imported
-- from the digests and only the curation ledger is replayed.
--
-- Keyed on the node's name at proposal time as well as its id, because ids are
-- regenerated by a rebuild: if the id no longer resolves, the name is the
-- fallback identity, and if neither resolves the proposal is reported as lost
-- rather than silently dropped.
CREATE TABLE IF NOT EXISTS rename_proposals (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    node_name_at_proposal TEXT NOT NULL,
    proposed_name TEXT NOT NULL,
    reason TEXT,
    proposed_by TEXT,
    proposed_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'applied', 'rejected', 'lost')),
    resolved_at TEXT,
    resolution_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_rename_proposals_status ON rename_proposals(status);

-- Which nodes a RECORD's digest declared, as distinct from which nodes its
-- claims reference. The two diverge, and the divergence is the point.
--
-- The CSG-11 AAV Incident Report declares the 2004 USS Nimitz UAP encounter and
-- edges only 2 of its 204 claims to it - the other 202 go to the participants
-- (117 to the object, 41 to Fravor, 32 to Underwood). That extraction is
-- defensible: a claim about what Fravor saw IS a claim about Fravor. But it
-- means "claims edged to this node" measures something quite different from
-- "records about this node", and the source-focus measure built on the former
-- scored the primary document at 0.010 on the event it documents.
--
-- Declaration is NOT by itself an aboutness signal - across 60 digests the
-- count per digest runs from 0 to 187, and a book declaring 187 events declares
-- this one too. It is kept because it is real extracted evidence that was being
-- discarded at import, and because any answer to "is this record about this
-- event" will need it. Interpreting it is a data-model question, not this
-- table's job.
CREATE TABLE IF NOT EXISTS record_nodes (
    record_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    PRIMARY KEY (record_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_record_nodes_node ON record_nodes(node_id);

-- Whether a claim BELONGS on a node, as distinct from being ATTACHED to it.
-- ADR-worthy distinction, and the whole reason this table exists: a claim's
-- presence in claim_node_refs is evidence the importer put it there, not that
-- the claim is about that node. 630 claims sat on the Nimitz event describing
-- Kaikoura, Socorro, Rendlesham and the Delphos encounter.
--
-- ABSENCE OF A ROW MEANS UNREVIEWED, NOT VERIFIED. Only two states are ever
-- written, and both are positive findings a human or a documented pass made:
--   verified  - read, and the claim is about this node
--   suspect   - read, and it is NOT; kept attached but excluded from assembly
-- A consumer that treats "no row" as verified has reintroduced the fault this
-- table records. The safe reading is: verified is assertable, suspect must be
-- excluded, unreviewed is a judgement the consumer has to make and declare.
CREATE TABLE IF NOT EXISTS claim_ref_status (
    claim_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('verified', 'suspect')),
    reason TEXT,
    set_at TEXT NOT NULL,
    set_by TEXT,
    PRIMARY KEY (claim_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_claim_ref_status_node ON claim_ref_status(node_id);

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

-- Article proposals: which nodes earn a published page. DERIVED from the page-
-- worthiness gate (page_gate.py: node-type tier + independent-source floor),
-- recomputed each maintenance pass - NOT replayed. The editorial signal that
-- survives a rebuild is the veto (curation/page-vetoes.yaml, replayed into
-- page_vetoes); a vetoed node is excluded here. source_count is the distinct-
-- source-record independence PROXY; independent_source_count is the true
-- provenance-root count, NULL until evidence-scoring pins it (ADR 0039).
CREATE TABLE IF NOT EXISTS page_proposals (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    claim_count INTEGER NOT NULL,
    source_count INTEGER NOT NULL,
    independent_source_count INTEGER,
    -- How the claims are SPREAD across sources, not just how many sources there
    -- are. source_count reports 2 for a node that is 98% one book, which is the
    -- shape of a page that summarises a single copyrighted work with a second
    -- source attached. Reported, not gated - see page_gate._source_spread.
    top_source_claims INTEGER,
    second_source_claims INTEGER,
    -- Claims whose provenance chain predates ADR 0044, so their root is
    -- unknowable. independent_source_count is computed WITHOUT them, and this is
    -- the confidence in that number: 5% unscored and 60% unscored both yield a
    -- count, and only the first should be trusted.
    unscored_claims INTEGER,
    -- Claims that are ABOUT the node (it is the grammatical subject) rather than
    -- merely mentioning it. Gated for person/organisation/object, reported for
    -- the rest - see page_gate._subject_counts.
    subject_claims INTEGER,
    status TEXT NOT NULL,
    computed_at TEXT NOT NULL
);

-- Page vetoes: "this node should never get a page". DERIVED from the durable
-- curation ledger (curation/page-vetoes.yaml), repopulated on rebuild, keyed on
-- natural identity. Distinct from node_rejections ("not a duplicate"): a veto
-- keeps the node in the graph but off the page list (e.g. a node that clears the
-- floor yet is editorially a mention, not a subject).
CREATE TABLE IF NOT EXISTS page_vetoes (
    veto_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    undone_at TEXT,
    PRIMARY KEY (veto_id, node_id)
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
CREATE INDEX IF NOT EXISTS idx_proposals_status ON page_proposals(status);
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
        # ADR 0044 migration: the provenance chain. Existing rows keep a NULL
        # origin_kind, which reads as "chain not captured" - a re-digest backfills
        # it. They are never treated as independent on the strength of the absence.
        for column in ("origin_kind", "origin", "relay"):
            if column not in cols:
                conn.execute(f"ALTER TABLE claims ADD COLUMN {column} TEXT")
        # Entailment migration: existing rows read as not assessed until the
        # digester's backfill is imported.
        for column, kind in (
            ("entailment_label", "TEXT"),
            ("entailment_score", "REAL"),
            ("entailment_model", "TEXT"),
            ("entailment_premise", "TEXT"),
        ):
            if column not in cols:
                conn.execute(f"ALTER TABLE claims ADD COLUMN {column} {kind}")
    # Work-identity migration: which WORK a record manifests. Backfilled to the
    # record's own id (one record, one work) so a pre-existing database counts
    # sources exactly as it did before the column existed - the guard lands as a
    # no-op and only bites once `link-works` collapses a detected duplicate.
    records_exist = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'"
    ).fetchone()
    if records_exist:
        record_cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
        if "work_id" not in record_cols:
            conn.execute("ALTER TABLE records ADD COLUMN work_id TEXT")
            conn.execute("UPDATE records SET work_id = id WHERE work_id IS NULL")
    # Source-spread migration: how a node's claims are DISTRIBUTED across its
    # sources, which source_count cannot express. Existing rows stay NULL until
    # the next `propose-pages` recomputes the derived table.
    proposals_exist = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='page_proposals'"
    ).fetchone()
    if proposals_exist:
        proposal_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(page_proposals)")
        }
        for column in (
            "top_source_claims",
            "second_source_claims",
            "unscored_claims",
            "subject_claims",
        ):
            if column not in proposal_cols:
                conn.execute(f"ALTER TABLE page_proposals ADD COLUMN {column} INTEGER")
    conn.executescript(SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_role ON claims(claim_role)")


def insert_node(conn: sqlite3.Connection, node: Node) -> Node:
    now = _now()
    metadata_json = json.dumps(node.metadata, default=str) if node.metadata else None
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
    # MOST-REFERENCED FIRST, then id. 56 live names are shared by more than one
    # node, and a bare fetchone() took whichever row the storage engine handed
    # back - stable for one database and NOT stable across a rebuild, where row
    # order can differ. provenance_root resolves a named origin through here with
    # no node_type, and that root is what ADR 0039 independence counts group on,
    # so a rebuild could regroup the evidence scoring with nothing to notice.
    #
    # Ordering by claim count rather than by id because it is also RIGHT, not
    # merely reproducible. Every ambiguous origin in the corpus has exactly one
    # node carrying claims and the rest at zero: "Robertson Panel" is a project
    # with 38 claims beside an empty organisation and an empty matter;
    # "O'Brien Committee" a project with 4 beside an empty organisation. The
    # populated node is the one the corpus means. id order would pick by uuid.
    ordered = (
        "SELECT n.* FROM nodes n WHERE n.name = ? AND n.retired_at IS NULL{type_clause}"
        " ORDER BY (SELECT COUNT(*) FROM claim_node_refs r WHERE r.node_id = n.id) DESC,"
        " n.id"
    )
    if node_type:
        row = conn.execute(
            ordered.format(type_clause=" AND n.node_type = ?"), (name, node_type)
        ).fetchone()
    else:
        row = conn.execute(ordered.format(type_clause=""), (name,)).fetchone()
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
    metadata_json = (
        json.dumps(record.metadata, default=str) if record.metadata else None
    )
    conn.execute(
        "INSERT INTO records "
        "(id, title, reference, date, producer_id, content_hash, friendly_name, "
        "metadata, created_at, work_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            # One record, one work, until a duplicate scan says otherwise. Seeded
            # rather than left NULL so every source count can group by work_id
            # unconditionally and a record that was never scanned still counts
            # once, instead of collapsing all unscanned records into one NULL work.
            record.id,
        ),
    )
    return record.model_copy(update={"created_at": datetime.fromisoformat(now)})


def get_record(conn: sqlite3.Connection, record_id: str) -> Record | None:
    row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def get_record_by_content_hash(
    conn: sqlite3.Connection, content_hash: str
) -> Record | None:
    row = conn.execute(
        "SELECT * FROM records WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return None if row is None else _row_to_record(row)


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
    entailment: dict | None = None,
) -> Claim:
    # created_at is overridable so a carried-forward claim keeps its original
    # timestamp across a re-import (the row is rewritten, not first-seen).
    now = created_at or _now()
    metadata_json = json.dumps(claim.metadata, default=str) if claim.metadata else None
    chain = claim.provenance_chain
    ent = entailment or {}
    conn.execute(
        "INSERT INTO claims (id, content, original_excerpt, claim_type, attestation, record_id, speaker_id, "
        "location_in_record, date, date_end, claim_hash, confidence, metadata, created_at, claim_role, "
        "origin_kind, origin, relay, entailment_label, entailment_score, "
        "entailment_model, entailment_premise) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            chain.origin_kind.value if chain else None,
            chain.origin if chain else None,
            json.dumps(chain.relay) if chain else None,
            ent.get("label"),
            ent.get("score"),
            ent.get("model"),
            ent.get("premise"),
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
    # Same transaction as the delete. A re-digest deletes and recreates claims, so
    # without this the stamp and the vector outlive the claim: 4,640 orphan claim
    # stamps had accumulated, and their vectors were still in the search index.
    forget_embeddings(conn, "claim", claim_id)


def update_claim_entailment(
    conn: sqlite3.Connection, claim_id: str, entailment: dict | None
) -> None:
    """Refresh a claim's entailment on re-import. Like the provenance chain it
    lies outside claim_hash, so a carried-forward claim would otherwise keep a
    stale or absent value for as long as its wording stood."""
    ent = entailment or {}
    conn.execute(
        "UPDATE claims SET entailment_label = ?, entailment_score = ?, "
        "entailment_model = ?, entailment_premise = ? WHERE id = ?",
        (
            ent.get("label"),
            ent.get("score"),
            ent.get("model"),
            ent.get("premise"),
            claim_id,
        ),
    )


def update_claim_chain(
    conn: sqlite3.Connection, claim_id: str, chain: ProvenanceChain | None
) -> None:
    """Refresh a claim's provenance chain in place (ADR 0044).

    Needed on the carry-forward path: claim_hash fingerprints MEANING and does not
    cover the chain, so an unchanged claim whose chain was absent (or has since
    changed) matches by hash and keeps the stale value indefinitely.
    """
    conn.execute(
        "UPDATE claims SET origin_kind = ?, origin = ?, relay = ? WHERE id = ?",
        (
            chain.origin_kind.value if chain else None,
            chain.origin if chain else None,
            json.dumps(chain.relay) if chain and chain.relay else None,
            claim_id,
        ),
    )


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


def provenance_root(conn: sqlite3.Connection, claim_id: str) -> tuple[str, str]:
    """The root of a claim's provenance chain - the identity independence keys on
    (ADR 0044). Two claims sharing a root are ONE source, however many records,
    speakers or outlets repeated them.

    The rule that matters is the direction of the error. Over-counting roots is the
    unsafe failure: it lets ten podcasts relaying one anonymous email look like ten
    independent attestations, so corroboration ends up rewarding repetition. Every
    branch here therefore collapses toward FEWER roots when identity is uncertain.

    - ``speaker`` / ``unattributed``: the speaker is the only thing standing behind
      the assertion, so they are the root.
    - ``named`` / ``document``: the origin is a node, so it resolves through the
      alias graph - "DIA" and "Defense Intelligence Agency" are one root, not two.
      Unresolvable origins fall back to their normalised prose.
    - ``anonymous``: an unnamed origin can never be a node, so there is no id to join
      on and the only comparable identity is the prose. Until the semantic matcher
      pins distinct anonymous origins apart, they ALL collapse to a single root -
      the conservative floor. Clustering can only ever split them back out, which
      raises independence; it can never inflate it.
    - chain not captured (pre-0044 digest): all such claims collapse to ONE
      ``unknown`` root. Absence means the chain was never recorded, never that the
      claim is independent - defaulting it to independent is exactly the failure
      0044 exists to close. A re-digest backfills the chain and independence rises
      to what the evidence actually supports.
    """
    row = conn.execute(
        "SELECT speaker_id, record_id, origin_kind, origin FROM claims WHERE id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return ("unknown", "")
    speaker_id, record_id, origin_kind, origin = row

    if not origin_kind:
        return ("unknown", "")

    if origin_kind in ("speaker", "unattributed"):
        return ("speaker", speaker_id) if speaker_id else ("record", record_id)

    if origin_kind == "anonymous":
        return ("anonymous", "")

    # named / document: resolve the origin to a node so aliases and acronyms
    # collapse onto one identity.
    name = (origin or "").strip()
    if not name:
        return ("unknown", "")
    node = find_node_by_name(conn, name)
    if node:
        return ("node", node.id)
    return (origin_kind, name.casefold())


def get_independent_source_count(conn: sqlite3.Connection, claim_id: str) -> int:
    """Count independent sources corroborating a claim - the number of DISTINCT
    provenance-chain roots across the corroboration group (ADR 0044). Ten outlets
    reporting one press release is one source, not ten."""
    corroborated = get_corroborations(conn, claim_id)
    all_claim_ids = [claim_id] + [cid for cid, _ in corroborated]
    return len({provenance_root(conn, cid) for cid in all_claim_ids})


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
    stats["entailment"] = entailment_counts(conn)
    return stats


def entailment_counts(conn: sqlite3.Connection) -> dict:
    """How many claims the digester's entailment check has assessed and how
    the labels fall. Counts only, and the entailed share is SPLIT by premise:
    entailed by the quote alone is the strong verdict, entailed only by the
    record text around it the weaker one, and one number would hide which.
    Shown, never yet weighted - the evidence score it will feed is not defined."""
    rows = conn.execute(
        "SELECT COALESCE(entailment_label, 'unassessed'), entailment_premise, "
        "COUNT(*) FROM claims GROUP BY 1, 2"
    ).fetchall()
    return summarise_entailment([(label, premise, n) for label, premise, n in rows])


def summarise_entailment(groups: list[tuple[str | None, str | None, int]]) -> dict:
    """The entailment summary from (label, premise, count) groups; the brief
    builds the same block for one page from its own rows."""
    count: dict[str, int] = {}
    for label, premise, n in groups:
        key = label or "unassessed"
        count[key] = count.get(key, 0) + n
        if key == "entails":
            sub = "entailed_by_quote" if premise == "quote" else "entailed_by_window"
            count[sub] = count.get(sub, 0) + n
    assessed = sum(
        v for k, v in count.items() if k in ("entails", "neutral", "contradicts")
    )
    frac = lambda k: round(count.get(k, 0) / assessed, 3) if assessed else None  # noqa: E731
    return {
        "assessed": assessed,
        "unassessed": count.get("unassessed", 0),
        "entails": count.get("entails", 0),
        "neutral": count.get("neutral", 0),
        "contradicts": count.get("contradicts", 0),
        "entailed_by_quote": count.get("entailed_by_quote", 0),
        "entailed_by_window": count.get("entailed_by_window", 0),
        "entailed_by_quote_fraction": frac("entailed_by_quote"),
        "entailed_by_window_fraction": frac("entailed_by_window"),
    }


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
        provenance_chain=_row_to_chain(row),
    )


def _row_to_chain(row: tuple) -> ProvenanceChain | None:
    """Rebuild the chain from a claims row. A NULL origin_kind means the chain was
    not captured (pre-ADR-0044 digest), which is NOT the same as having no chain -
    see ``provenance_root``."""
    if len(row) <= 15 or not row[15]:
        return None
    return ProvenanceChain(
        origin_kind=OriginKind(row[15]),
        origin=row[16] or "",
        relay=json.loads(row[17]) if len(row) > 17 and row[17] else [],
    )


def insert_corroboration_rejection(
    conn: sqlite3.Connection,
    claim_a: str,
    claim_b: str,
    similarity: float,
    model: str | None = None,
) -> None:
    """Record that a candidate pair was adjudicated and found NOT to corroborate.

    The verdict cost a model call; not storing it means buying it again on every
    later run over the same corpus."""
    # Sorted, exactly as insert_corroboration does. The pair is the unit, not the
    # order, and an unsorted insert would let (a,b) and (b,a) both be stored -
    # making the primary key decorative and the pair bought twice anyway, which is
    # the whole thing this table exists to prevent.
    a, b = sorted([claim_a, claim_b])
    conn.execute(
        "INSERT OR REPLACE INTO corroboration_rejections "
        "(claim_a, claim_b, similarity, model, rejected_at) VALUES (?, ?, ?, ?, ?)",
        (a, b, similarity, model, _now()),
    )


def adjudicated_pairs(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Every pair already decided, accepted or rejected - the set a new run skips."""
    pairs: set[tuple[str, str]] = set()
    for table in ("corroborations", "corroboration_rejections"):
        try:
            rows = conn.execute(f"SELECT claim_a, claim_b FROM {table}").fetchall()
        except sqlite3.OperationalError:
            continue
        for a, b in rows:
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs


def set_claim_ref_status(
    conn: sqlite3.Connection,
    claim_id: str,
    node_id: str,
    status: str,
    reason: str | None = None,
    set_by: str | None = None,
) -> None:
    """Record that a claim was READ and found to belong on a node, or not."""
    conn.execute(
        "INSERT INTO claim_ref_status (claim_id, node_id, status, reason, set_at, set_by)"
        " VALUES (?, ?, ?, ?, datetime('now'), ?)"
        " ON CONFLICT(claim_id, node_id) DO UPDATE SET"
        " status=excluded.status, reason=excluded.reason,"
        " set_at=excluded.set_at, set_by=excluded.set_by",
        (claim_id, node_id, status, reason, set_by),
    )


def claim_ref_statuses(conn: sqlite3.Connection, node_id: str) -> dict[str, str]:
    """claim_id -> status for one node. A claim absent from the map is UNREVIEWED."""
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT claim_id, status FROM claim_ref_status WHERE node_id = ?",
            (node_id,),
        )
    }


def node_belonging_counts(conn: sqlite3.Connection, node_id: str) -> dict[str, int]:
    """How many of a node's claims are verified, suspect, and unreviewed."""
    total = conn.execute(
        "SELECT COUNT(*) FROM claim_node_refs WHERE node_id = ?", (node_id,)
    ).fetchone()[0]
    counts = {"verified": 0, "suspect": 0}
    for status, n in conn.execute(
        "SELECT s.status, COUNT(*) FROM claim_ref_status s"
        " JOIN claim_node_refs r ON r.claim_id = s.claim_id AND r.node_id = s.node_id"
        " WHERE s.node_id = ? GROUP BY s.status",
        (node_id,),
    ):
        counts[status] = n
    counts["unreviewed"] = total - counts["verified"] - counts["suspect"]
    counts["total"] = total
    return counts


def link_record_nodes(
    conn: sqlite3.Connection, record_id: str, node_ids: "list[str]"
) -> None:
    """Record which nodes this record's digest DECLARED.

    Replaces the record's set rather than adding to it, so a re-import after a
    re-digest does not leave declarations the digest no longer makes.
    """
    conn.execute("DELETE FROM record_nodes WHERE record_id = ?", (record_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO record_nodes (record_id, node_id) VALUES (?, ?)",
        [(record_id, n) for n in dict.fromkeys(node_ids)],
    )


def records_declaring(conn: sqlite3.Connection, node_id: str) -> "list[tuple]":
    """Records whose digest declared this node, with how many nodes each declared.

    The second number is the caller's confidence signal: a record declaring two
    events and naming this one is saying something; a book declaring 187 is not.
    """
    return conn.execute(
        "SELECT rn.record_id, r.title,"
        " (SELECT COUNT(*) FROM record_nodes rn2 WHERE rn2.record_id = rn.record_id)"
        " FROM record_nodes rn JOIN records r ON r.id = rn.record_id"
        " WHERE rn.node_id = ? ORDER BY 3",
        (node_id,),
    ).fetchall()


def propose_rename(
    conn: sqlite3.Connection,
    node_id: str,
    node_name: str,
    proposed_name: str,
    reason: str | None = None,
    proposed_by: str | None = None,
) -> str:
    """Record a reviewer's proposed node name. Returns the proposal id."""
    import uuid as _uuid

    pid = str(_uuid.uuid4())
    conn.execute(
        "INSERT INTO rename_proposals (id, node_id, node_name_at_proposal,"
        " proposed_name, reason, proposed_by, proposed_at, status)"
        " VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'pending')",
        (pid, node_id, node_name, proposed_name, reason, proposed_by),
    )
    return pid


def pending_renames(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, node_id, node_name_at_proposal, proposed_name, reason,"
        " proposed_by, proposed_at FROM rename_proposals WHERE status = 'pending'"
        " ORDER BY proposed_at"
    ).fetchall()
    keys = (
        "id",
        "node_id",
        "node_name_at_proposal",
        "proposed_name",
        "reason",
        "proposed_by",
        "proposed_at",
    )
    return [dict(zip(keys, r)) for r in rows]


def resolve_rename(
    conn: sqlite3.Connection, proposal_id: str, status: str, note: str | None = None
) -> None:
    conn.execute(
        "UPDATE rename_proposals SET status = ?, resolved_at = datetime('now'),"
        " resolution_note = ? WHERE id = ?",
        (status, note, proposal_id),
    )
