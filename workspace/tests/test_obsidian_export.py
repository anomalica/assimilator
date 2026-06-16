import sqlite3
from pathlib import Path

from assimilator.database import (
    init_db,
    insert_alias,
    insert_claim,
    insert_node,
    insert_record,
)
from anomalica_common.digest.models import (
    AttestationLevel,
    Claim,
    ClaimType,
    Node,
    NodeType,
    Record,
)
from assimilator.obsidian_export import (
    _safe_filename,
    _safe_wikilink,
    export_to_obsidian,
)


def _db():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_safe_filename_strips_forbidden_chars():
    assert _safe_filename("Foo/Bar:Baz?") == "Foo_Bar_Baz_"


def test_safe_filename_empty_input():
    assert _safe_filename("") == "_unnamed"
    assert _safe_filename("   ") == "_unnamed"


def test_safe_wikilink_strips_brackets():
    assert _safe_wikilink("Foo [Bar] | Baz") == "Foo Bar  Baz"


def test_export_groups_entities_by_type(tmp_path: Path):
    conn = _db()
    fravor = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    nimitz = insert_node(conn, Node(node_type=NodeType.object, name="USS Nimitz"))
    insert_node(conn, Node(node_type=NodeType.place, name="Persian Gulf"))
    insert_node(conn, Node(node_type=NodeType.organisation, name="VFA-41"))
    insert_alias(conn, "Cdr Fravor", fravor.id)
    rec = insert_record(
        conn,
        Record(title="Fravor Hearing", date="2023-07-26", producer_id=fravor.id),
    )
    insert_claim(
        conn,
        Claim(
            content="Fravor observed the Tic Tac.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
            speaker_id=fravor.id,
            location_in_record="page 1",
            date="2004-11-14",
            node_references=[fravor.id, nimitz.id],
            original_excerpt="I saw the Tic Tac.",
        ),
    )
    conn.commit()

    out = tmp_path / "vault"
    counts = export_to_obsidian(out, conn)

    assert counts == {"records": 1, "nodes": 4, "claims": 1}
    # Folder-per-type structure
    assert (out / "People").is_dir()
    assert (out / "Objects").is_dir()
    assert (out / "Places").is_dir()
    assert (out / "Organisations").is_dir()
    assert (out / "Sources").is_dir()
    assert (out / "README.md").exists()
    # Flat Nodes/ no longer exists
    assert not (out / "Nodes").exists()


def test_node_file_lists_claims_inline(tmp_path: Path):
    conn = _db()
    fravor = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    nimitz = insert_node(conn, Node(node_type=NodeType.object, name="USS Nimitz"))
    rec = insert_record(conn, Record(title="Hearing", date="2023"))
    insert_claim(
        conn,
        Claim(
            content="Fravor observed the Tic Tac.",
            claim_type=ClaimType.observation,
            attestation=AttestationLevel.first_hand,
            record_id=rec.id,
            speaker_id=fravor.id,
            location_in_record="page 1",
            node_references=[fravor.id, nimitz.id],
            original_excerpt="I saw the Tic Tac.",
        ),
    )
    conn.commit()

    out = tmp_path / "vault"
    export_to_obsidian(out, conn)

    fravor_text = (out / "People" / "David Fravor.md").read_text()
    # The claim itself is on the Fravor page
    assert "Fravor observed the Tic Tac." in fravor_text
    # Original excerpt rendered as block quote
    assert "> I saw the Tic Tac." in fravor_text
    # Cross-links to the other referenced entity and the source record
    assert "[[USS Nimitz]]" in fravor_text
    assert "[[2023 - Hearing]]" in fravor_text


def test_node_file_includes_aliases_in_frontmatter(tmp_path: Path):
    conn = _db()
    fravor = insert_node(conn, Node(node_type=NodeType.person, name="David Fravor"))
    insert_alias(conn, "Cdr Fravor", fravor.id)
    insert_alias(conn, "Commander David Fravor", fravor.id)
    conn.commit()

    out = tmp_path / "vault"
    export_to_obsidian(out, conn)

    text = (out / "People" / "David Fravor.md").read_text()
    assert "aliases:" in text
    assert "Cdr Fravor" in text
    assert "Commander David Fravor" in text


def test_export_merges_same_named_nodes_across_dbs(tmp_path: Path):
    domain = _db()
    infra = _db()
    insert_node(domain, Node(node_type=NodeType.person, name="Luis Elizondo"))
    insert_node(infra, Node(node_type=NodeType.person, name="Luis Elizondo"))
    insert_node(infra, Node(node_type=NodeType.organisation, name="60 Minutes"))
    domain.commit()
    infra.commit()

    out = tmp_path / "vault"
    counts = export_to_obsidian(out, domain, infra)
    # Elizondo should appear once across both DBs.
    assert counts["nodes"] == 2
    assert (out / "People" / "Luis Elizondo.md").exists()
    assert (out / "Organisations" / "60 Minutes.md").exists()


def test_export_handles_same_name_different_types(tmp_path: Path):
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="Apollo"))
    insert_node(conn, Node(node_type=NodeType.matter, name="Apollo"))
    conn.commit()
    out = tmp_path / "vault"
    counts = export_to_obsidian(out, conn)
    # Same name across types lands in separate folders; both files present.
    assert counts["nodes"] == 2
    assert (out / "People" / "Apollo.md").exists()
    assert (out / "Matters" / "Apollo.md").exists()


def test_readme_explains_the_structure(tmp_path: Path):
    conn = _db()
    insert_node(conn, Node(node_type=NodeType.person, name="Test"))
    conn.commit()
    out = tmp_path / "vault"
    export_to_obsidian(out, conn)
    readme = (out / "README.md").read_text()
    assert "Anomalica Generated Vault" in readme
    assert "People/" in readme
    assert "Sources/" in readme
