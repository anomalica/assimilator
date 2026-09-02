"""Re-import reconciliation: claim_hash carry-forward, no duplicate rows.

Guards the fix for two bugs in the importer's existing-record path: re-importing
a record used to (a) accumulate duplicate claim rows because nothing deleted the
old ones, and (b) churn every claim's uuid because the digester mints a fresh one
per emission, making an unchanged re-digest read as 100% changed.
"""

from __future__ import annotations

import sqlite3

from assimilator.database import init_db
from assimilator.import_markdown import backfill_claim_hashes, import_extraction
from assimilator.matching import match_node


def _parsed(claims):
    return {
        "frontmatter": {
            "record_id": "rec-nimitz-0001",
            "record_title": "Nimitz Encounter Briefing",
            "record_date": "2004-11-14",
            "content_hash": "sha256:" + "a" * 64,
            "friendly_name": "nimitz-briefing",
        },
        "nodes": [
            {"id": "n-fravor", "name": "David Fravor", "node_type": "person"},
            {"id": "n-nimitz", "name": "USS Nimitz", "node_type": "object"},
        ],
        "domain_claims": claims,
        "infrastructure_claims": [],
        "terminology": None,
    }


def _claim(cid, content, refs=("David Fravor",), speaker="David Fravor"):
    return {
        "id": cid,
        "content": content,
        "claim_type": "testimony",
        "attestation": "first_hand",
        "speaker": speaker,
        "node_references": list(refs),
    }


def _conn():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _claim_count(conn):
    return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]


def _claim_ids(conn):
    return {r[0] for r in conn.execute("SELECT id FROM claims").fetchall()}


def test_first_import_sets_claim_hash_on_every_claim():
    conn = _conn()
    counts = import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))
    assert counts["claims_created"] == 1
    assert counts["claims_carried"] == 0
    null_hashes = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE claim_hash IS NULL"
    ).fetchone()[0]
    assert null_hashes == 0


def test_identical_reimport_carries_forward_and_does_not_duplicate():
    conn = _conn()
    claims = [_claim("c1", "Held radar 12 min."), _claim("c2", "Tic Tac accelerated.")]
    import_extraction(conn, _parsed(claims))
    ids_before = _claim_ids(conn)
    assert _claim_count(conn) == 2

    # Re-import the SAME content but with fresh uuids (as the digester would).
    reimport = [
        _claim("x1", "Held radar 12 min."),
        _claim("x2", "Tic Tac accelerated."),
    ]
    counts = import_extraction(conn, _parsed(reimport))

    assert _claim_count(conn) == 2  # no duplication - the core bug fix
    assert counts["claims_carried"] == 2
    assert counts["claims_created"] == 0
    assert counts["claims_deleted"] == 0
    assert _claim_ids(conn) == ids_before  # original uuids carried forward


def test_changed_claim_replaces_only_itself():
    conn = _conn()
    import_extraction(
        conn, _parsed([_claim("c1", "Held radar 12 min."), _claim("c2", "Unchanged.")])
    )
    c2_id = conn.execute(
        "SELECT id FROM claims WHERE content = 'Unchanged.'"
    ).fetchone()[0]

    counts = import_extraction(
        conn,
        _parsed([_claim("x1", "Held radar 14 min."), _claim("x2", "Unchanged.")]),
    )
    assert _claim_count(conn) == 2
    assert counts["claims_created"] == 1  # the edited claim
    assert counts["claims_deleted"] == 1  # its prior version
    assert counts["claims_carried"] == 1  # the unchanged one
    # The unchanged claim kept its row identity.
    assert (
        conn.execute("SELECT id FROM claims WHERE content = 'Unchanged.'").fetchone()[0]
        == c2_id
    )


def test_removed_claim_is_deleted():
    conn = _conn()
    import_extraction(
        conn, _parsed([_claim("c1", "Held radar 12 min."), _claim("c2", "Second.")])
    )
    counts = import_extraction(conn, _parsed([_claim("x1", "Held radar 12 min.")]))
    assert _claim_count(conn) == 1
    assert counts["claims_deleted"] == 1
    assert counts["claims_carried"] == 1


def test_added_claim_is_inserted():
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "First.")]))
    counts = import_extraction(
        conn, _parsed([_claim("x1", "First."), _claim("x2", "New one.")])
    )
    assert _claim_count(conn) == 2
    assert counts["claims_created"] == 1
    assert counts["claims_carried"] == 1


def test_backfill_reproduces_the_import_hash():
    # A claim hashed at import and the same claim hashed by the backfill must
    # agree - same canonicalisation on both paths.
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))
    at_import = conn.execute("SELECT claim_hash FROM claims").fetchone()[0]

    conn.execute("UPDATE claims SET claim_hash = NULL")
    backfill_claim_hashes(conn)
    after_backfill = conn.execute("SELECT claim_hash FROM claims").fetchone()[0]
    assert after_backfill == at_import


def test_node_metadata_aliases_become_graph_aliases():
    """A rebuild wipes the graph and only the digests survive, so an alias that
    lives in a database row is lost on the next rebuild. The retained
    surname-first person form (node-types.md) is carried in the digest's node
    metadata and written to the aliases table on import."""
    conn = _conn()
    parsed = _parsed([_claim("c1", "First.")])
    parsed["nodes"][0]["metadata"] = {
        "family_name": "Fravor",
        "aliases": ["Fravor, David"],
    }
    import_extraction(conn, parsed)

    row = conn.execute(
        "SELECT n.name FROM aliases a JOIN nodes n ON n.id = a.node_id "
        "WHERE a.alias = ?",
        ("Fravor, David",),
    ).fetchone()
    assert row is not None and row[0] == "David Fravor"

    # Last-first input still resolves to the natural-order node afterwards.
    assert (
        match_node(conn, "Fravor, David", "person")[0]
        == conn.execute("SELECT id FROM nodes WHERE name = 'David Fravor'").fetchone()[
            0
        ]
    )


def test_content_hash_comes_from_the_store_filename_not_the_frontmatter(tmp_path):
    """A store record is addressed BY its hash, so the filename is authoritative
    and the frontmatter copy can be wrong. It IS wrong in the live corpus: a
    legacy loose record declares a content_hash belonging to a different record
    entirely, and trusting it would stamp claims with the wrong source and point
    every workbench deep link at the wrong document."""
    from pathlib import Path

    from assimilator.import_markdown import _content_hash_of

    right, wrong = "a" * 64, "b" * 64
    stored = Path(f"/store/{right}.v2.md")
    assert _content_hash_of(stored, f"sha256:{wrong}") == f"sha256:{right}"
    assert _content_hash_of(stored, None) == f"sha256:{right}"
    # A loose record has no filename hash to read, so its declaration is all
    # there is.
    loose = Path("/records/2007-06-20-web-project-serpo.md")
    assert _content_hash_of(loose, f"sha256:{wrong}") == f"sha256:{wrong}"


def test_reimport_survives_a_claim_whose_hash_moved_under_it():
    """A claim keeps its uuid across re-emission while its HASH moves whenever
    node resolution changes underneath it - a rename, a merge, anything that
    repoints a ref. It then matches no prior hash, is treated as new, and
    inserting it before its stale row is deleted collides on the primary key.
    This is what stopped the whole corpus importing after a person-name
    migration moved 142 claim hashes."""
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))
    # Same claim id, same record, but the stored hash no longer matches what the
    # importer will now compute - exactly what a re-resolved node ref produces.
    conn.execute("UPDATE claims SET claim_hash = 'stale-hash-from-a-prior-graph'")
    conn.commit()

    counts = import_extraction(conn, _parsed([_claim("c1", "Held radar 12 min.")]))

    assert _claim_count(conn) == 1
    assert counts["claims_created"] == 1 and counts["claims_deleted"] == 1


def test_reimport_survives_a_node_id_retired_by_a_merge():
    """A digest's node id is a suggestion, not a reservation. Import a digest,
    merge its node away, re-import: the retired row still holds that id, so
    match_node (live nodes only) cannot find it and the insert used to collide on
    the primary key - which is what stopped the corpus importing after 13 merges
    landed."""

    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "First.")]))
    fravor = conn.execute(
        "SELECT id FROM nodes WHERE name = 'David Fravor'"
    ).fetchone()[0]
    assert fravor == "n-fravor"
    # Retire it the way a merge does, and rename it out of the way so the
    # re-import cannot match it by name either.
    conn.execute(
        "UPDATE nodes SET retired_at = '2026-07-31T00:00:00Z', name = 'Someone Else' "
        "WHERE id = ?",
        (fravor,),
    )
    conn.commit()

    import_extraction(conn, _parsed([_claim("c2", "Second.")]))

    live = conn.execute(
        "SELECT id FROM nodes WHERE name = 'David Fravor' AND retired_at IS NULL"
    ).fetchone()
    assert live is not None and live[0] != fravor


def _parsed_with_review(review):
    p = _parsed([_claim("c1", "First.")])
    p["frontmatter"]["record"] = {"id": "rec-nimitz-0001", "title": "T"}
    if review is not None:
        p["frontmatter"]["record"]["review"] = review
    return p


def test_review_state_reaches_the_record_row():
    """The basis on which unreviewed records may enter the graph at all is that
    they arrive MARKED, so a consumer can tell them from material a human read."""
    import json as _json

    conn = _conn()
    import_extraction(conn, _parsed_with_review({"state": "human"}))
    meta = conn.execute("SELECT metadata FROM records").fetchone()[0]
    assert _json.loads(meta)["review"]["state"] == "human"


def test_an_absent_review_field_is_unknown_not_unreviewed():
    """Three states, never two. `state: none` is a positive finding of no review;
    the field being absent means the digest predates the stamp. Storing absent as
    "none" would mark records unreviewed when we simply do not know - the same
    rule as a missing provenance chain never reading as independent."""
    conn = _conn()
    import_extraction(conn, _parsed_with_review(None))
    meta = conn.execute("SELECT metadata FROM records").fetchone()[0]
    assert meta is None or "review" not in meta


def test_review_state_is_refreshed_on_reimport():
    """Insert-only would leave every record already in the graph unmarked
    forever, which is how the provenance chain survived a full re-digest."""
    import json as _json

    conn = _conn()
    import_extraction(conn, _parsed_with_review(None))
    import_extraction(conn, _parsed_with_review({"state": "human"}))
    meta = conn.execute("SELECT metadata FROM records").fetchone()[0]
    assert _json.loads(meta)["review"]["state"] == "human"


def test_ai_usage_never_reaches_the_graph():
    """Per-record usage is kept in the digest and the operations ledger and is
    never reader-facing. The graph feeds the assembler, which writes content,
    which becomes the public site - storing it here creates a leak path to
    satisfy a requirement two other stores already meet."""
    import json as _json

    conn = _conn()
    p = _parsed([_claim("c1", "First.")])
    p["frontmatter"]["ai_usage"] = [{"stage": "digest", "tokens": {"input": 999}}]
    p["frontmatter"]["pre_digest"] = {"sha256": "abc", "prep_version": 6}
    import_extraction(conn, p)

    meta = conn.execute("SELECT metadata FROM records").fetchone()[0] or "{}"
    assert "ai_usage" not in meta and "999" not in meta
    assert _json.loads(meta)["pre_digest"]["sha256"] == "abc"


def test_absent_run_kind_is_unknown_not_production():
    """23 of 53 digests carry it. Defaulting absence to "production" would
    silently promote comparison artefacts into the canonical set - the third
    instance this week of absence read as a value."""
    conn = _conn()
    import_extraction(conn, _parsed([_claim("c1", "First.")]))
    meta = conn.execute("SELECT metadata FROM records").fetchone()[0] or "{}"
    assert "run_kind" not in meta


def test_a_record_renamed_by_a_redigest_is_the_same_record():
    """The digester refreshes record blocks from the record's frontmatter and may
    change a title while keeping the id. Found by title, that read as a NEW record
    and collided on the primary key; the entailment backfill stopped at digest 18
    of 108 on "Project Serpo" becoming its interview title."""
    conn = _conn()
    first = _parsed([_claim("c1", "Held radar 12 min.")])
    import_extraction(conn, first)

    renamed = _parsed([_claim("c1", "Held radar 12 min.")])
    renamed["frontmatter"]["record_title"] = "Interview with the radar operator"
    counts = import_extraction(conn, renamed)

    rows = conn.execute("SELECT id, title FROM records").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "Interview with the radar operator"
    assert counts["claims_carried"] == 1
