from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest
import yaml
from anomalica_common.digest import parse_digest_yaml
from anomalica_common.digest.models import Claim, Node, Record, SourceAnchor
from anomalica_common.identity import (
    digest_record_snapshot_identity,
    evidence_unit_identity,
    record_identity,
)
from anomalica_common.pre_digest import (
    prepare_page_record,
    source_anchor_for_body_span,
)
from anomalica_common.records import WorkProvenance

from assimilator.database import (
    get_independent_source_count,
    init_db,
    insert_claim,
    insert_corroboration,
    insert_node,
    insert_record,
)
from assimilator.evidence import (
    ValidatedDigest,
    assess_evidence_support,
    rebuild_evidence_units,
    replace_claim_anchors,
    replace_claim_provenance_root,
    store_record_structure,
    validate_digest_for_import,
)
from assimilator.import_markdown import import_extraction
from assimilator.synthesise import build_entity_brief

ASSET = "sha256:" + "a" * 64
CONFIG = "sha256:" + "f" * 64
BODY = "<!-- file_page: 1 -->\nAlpha bravo charlie delta echo foxtrot golf.\n"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def _direct_anchored_claim(
    conn: sqlite3.Connection,
    claim_id: str,
    speaker_id: str,
    *,
    start: int,
    end: int,
    text_hash: str,
) -> None:
    if conn.execute("SELECT 1 FROM records WHERE id = 'record'").fetchone() is None:
        insert_record(conn, Record(id="record", title="Record"))
        conn.execute(
            "INSERT INTO assets (asset_hash, metadata) VALUES (?, '{}')", (ASSET,)
        )
    insert_node(conn, Node(id=speaker_id, node_type="person", name=speaker_id.title()))
    insert_claim(
        conn,
        Claim(
            id=claim_id,
            content=f"Claim {claim_id}",
            claim_type="observation",
            record_id="record",
            speaker_id=speaker_id,
            provenance_chain={"origin_kind": "speaker"},
        ),
    )
    replace_claim_anchors(
        conn,
        claim_id,
        [
            SourceAnchor(
                asset_hash=ASSET,
                record_page=1,
                asset_file_page=1,
                asset_text_sha256=text_hash,
                asset_span={"start": start, "end": end},
                body_span={"start": start, "end": end},
                quote="x" * (end - start),
            )
        ],
    )
    replace_claim_provenance_root(conn, claim_id)


def _digest2(
    tmp_path: Path,
    claims: list[tuple[str, str, str]],
    *,
    work_provenance: dict | None = None,
) -> tuple[dict, Path, object]:
    """Build a real prep-version-9 digest around quote substrings.

    ``claims`` is ``[(claim_id, speaker_id, quote), ...]``.
    """
    selection = [{"asset_hash": ASSET, "selector": {"type": "whole"}}]
    record_hash = record_identity(selection)
    structure = {
        "schema": "anomalica/record/3",
        "content_hash": record_hash,
        "assets": [
            {
                "asset_hash": ASSET,
                "source_type": "pdf",
                "file_format": "pdf",
                "archived_ext": "pdf",
                "pages": 1,
                "acquisition": {"acquired_at": "2026-09-22T09:00:00Z"},
                "copyright": {"status": "licensed"},
            }
        ],
        "selection": selection,
        "page_map": [
            {
                "record_page": 1,
                "asset_hash": ASSET,
                "asset_file_page": 1,
            }
        ],
    }
    prepared = prepare_page_record(structure, BODY)
    snapshot = {
        "schema": "anomalica/digest-record-snapshot/1",
        "content_hash": record_hash,
        "title": "Anchored record",
        "assets": [
            {
                "asset_hash": ASSET,
                "source_type": "pdf",
                "file_format": "pdf",
                "pages": 1,
            }
        ],
        "asset_rights": [{"asset_hash": ASSET, "status": "licensed"}],
        "selection": selection,
        "page_map": structure["page_map"],
    }
    if work_provenance is not None:
        snapshot["work_provenance"] = work_provenance
    speakers = sorted({speaker_id for _claim_id, speaker_id, _quote in claims})
    document = {
        "run_kind": "production",
        "schema": "anomalica/digest/2",
        "extraction_config": CONFIG,
        "pre_digest": {
            "sha256": prepared.sha256,
            "prep_version": 9,
            "source_map_sha256": prepared.source_map_sha256,
        },
        "record_snapshot_sha256": digest_record_snapshot_identity(snapshot),
        "record": {
            "id": "record-1",
            **{k: v for k, v in snapshot.items() if k != "schema"},
        },
        "nodes": [
            {"id": speaker_id, "type": "person", "name": speaker_id.title()}
            for speaker_id in speakers
        ],
        "domain_claims": [],
    }
    for claim_id, speaker_id, quote in claims:
        start = prepared.text.index(quote)
        anchor = source_anchor_for_body_span(
            prepared, {"start": start, "end": start + len(quote)}
        )
        document["domain_claims"].append(
            {
                "id": claim_id,
                "type": "observation",
                "text": f"{speaker_id} reports {quote}.",
                "quote": quote,
                "speaker": {"id": speaker_id, "name": speaker_id.title()},
                "provenance_chain": {
                    "origin_kind": "speaker",
                    "origin": speaker_id.title(),
                    "relay": [],
                },
                "source_anchors": [anchor.model_dump(mode="json")],
            }
        )

    ingests = tmp_path / "ingests"
    pre_digests = ingests / "pre-digests"
    source_maps = ingests / "source-maps"
    pre_digests.mkdir(parents=True)
    source_maps.mkdir(parents=True)
    (pre_digests / f"{prepared.sha256.removeprefix('sha256:')}.md").write_text(
        prepared.text
    )
    (
        source_maps / f"{prepared.source_map_sha256.removeprefix('sha256:')}.json"
    ).write_bytes(prepared.source_map_json)
    digests = tmp_path / "digests"
    digests.mkdir()
    path = digests / "anchored.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return parse_digest_yaml(path.read_text()), path, prepared


def test_digest2_is_validated_and_materialised_before_counting(tmp_path):
    parsed, path, _prepared = _digest2(
        tmp_path,
        [
            ("claim-a", "alice", "Alpha bravo charlie"),
            ("claim-b", "bob", "bravo charlie delta"),
            ("claim-c", "carol", "delta echo foxtrot"),
        ],
    )
    conn = _db()

    import_extraction(
        conn,
        parsed,
        source_path=str(path),
        source_root=path.parent,
    )

    assert conn.execute("SELECT COUNT(*) FROM assets").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM record_selections").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM record_page_maps").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM claim_anchors").fetchone() == (3,)
    # A overlaps B and B overlaps C. Transitivity makes one component even though
    # A and C do not overlap directly.
    unit = conn.execute(
        "SELECT id, span_start, span_end FROM evidence_units"
    ).fetchone()
    assert unit is not None
    starts_ends = conn.execute(
        "SELECT MIN(asset_start), MAX(asset_end), asset_text_sha256 FROM claim_anchors"
    ).fetchone()
    expected_id = evidence_unit_identity(
        {
            "asset_hash": ASSET,
            "asset_file_page": 1,
            "asset_text_sha256": starts_ends[2],
            "span": {"start": starts_ends[0], "end": starts_ends[1]},
        }
    )
    assert unit == (expected_id, starts_ends[0], starts_ends[1])
    assert conn.execute(
        "SELECT COUNT(DISTINCT evidence_unit_id) FROM claim_evidence_units"
    ).fetchone() == (1,)

    insert_corroboration(conn, "claim-a", "claim-b", 0.99)
    insert_corroboration(conn, "claim-a", "claim-c", 0.99)
    conn.commit()
    # Three explicit speaker roots still supply one independent unit because all
    # three claims draw from one overlap component.
    assert get_independent_source_count(conn, "claim-a") == 1
    brief_claim = build_entity_brief(conn, "alice")["claims"][0]
    assert brief_claim["source_anchors"]
    assert brief_claim["evidence"] == {
        "score": None,
        "evidence_unit_ids": [expected_id],
        "provenance_root_ids": ["assertion:node:alice"],
        "independence_status": "established",
        "independent_sources": 1,
    }


def test_digest2_validation_failure_leaves_graph_unchanged(tmp_path):
    parsed, path, _prepared = _digest2(tmp_path, [("claim-a", "alice", "Alpha bravo")])
    invalid = copy.deepcopy(parsed)
    invalid["domain_claims"][0]["source_anchors"][0]["quote"] = "wrong phrase"
    conn = _db()

    with pytest.raises(ValueError, match="quote|length"):
        import_extraction(
            conn,
            invalid,
            source_path=str(path),
            source_root=path.parent,
        )

    assert conn.execute("SELECT COUNT(*) FROM records").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM assets").fetchone() == (0,)


def test_only_evidenced_work_provenance_populates_record_work_id(tmp_path):
    parsed, path, _prepared = _digest2(
        tmp_path,
        [("claim-a", "alice", "Alpha bravo")],
        work_provenance={
            "root_id": "work:anchored-record",
            "evidence": ["Curator-confirmed common publication identity"],
        },
    )
    conn = _db()

    import_extraction(
        conn,
        parsed,
        source_path=str(path),
        source_root=path.parent,
    )

    assert conn.execute("SELECT work_id FROM records").fetchone() == (
        "work:anchored-record",
    )
    assert conn.execute(
        "SELECT status, kind, evidence FROM provenance_roots "
        "WHERE id = 'work:anchored-record'"
    ).fetchone() == (
        "established",
        "work",
        '["Curator-confirmed common publication identity"]',
    )


def test_shared_work_root_evidence_is_order_independent(tmp_path):
    parsed, path, _prepared = _digest2(
        tmp_path,
        [("claim-a", "alice", "Alpha bravo")],
        work_provenance={"root_id": "work:shared", "evidence": ["Evidence B"]},
    )
    conn = _db()
    import_extraction(
        conn,
        parsed,
        source_path=str(path),
        source_root=path.parent,
    )
    validated = validate_digest_for_import(
        parsed, source_path=str(path), source_root=str(path.parent)
    )
    assert validated.snapshot is not None
    first_snapshot = validated.snapshot.model_copy(
        update={
            "work_provenance": WorkProvenance(
                root_id="work:shared", evidence=["Evidence A"]
            )
        }
    )
    insert_record(
        conn,
        Record(
            id="record-2",
            title="Second manifestation",
            content_hash=first_snapshot.content_hash,
            metadata={
                "work_provenance": {
                    "root_id": "work:shared",
                    "evidence": ["Evidence A"],
                }
            },
        ),
    )
    first_validated = ValidatedDigest(
        schema=validated.schema,
        snapshot=first_snapshot,
        record_snapshot_sha256=validated.record_snapshot_sha256,
        source_map_sha256=validated.source_map_sha256,
        claim_anchors=validated.claim_anchors,
    )

    store_record_structure(conn, "record-2", first_validated)
    # Re-import the other Record last. The root must retain the canonical union,
    # not whichever Record happened to be processed most recently.
    store_record_structure(conn, "record-1", validated)

    assert conn.execute(
        "SELECT evidence FROM provenance_roots WHERE id = 'work:shared'"
    ).fetchone() == ('["Evidence A","Evidence B"]',)


def test_evidence_components_are_global_across_category_databases(tmp_path):
    parsed, path, _prepared = _digest2(
        tmp_path,
        [
            ("domain-claim", "alice", "Alpha bravo charlie"),
            ("infrastructure-claim", "bob", "bravo charlie delta"),
        ],
    )
    parsed["infrastructure_claims"] = [parsed["domain_claims"].pop()]
    domain = _db()
    infrastructure = _db()

    import_extraction(
        domain,
        parsed,
        section="domain",
        lookup_conns=[infrastructure],
        source_path=str(path),
        source_root=path.parent,
    )
    import_extraction(
        infrastructure,
        parsed,
        section="infrastructure",
        lookup_conns=[domain],
        source_path=str(path),
        source_root=path.parent,
    )

    domain_unit = domain.execute(
        "SELECT id, span_start, span_end FROM evidence_units"
    ).fetchone()
    infrastructure_unit = infrastructure.execute(
        "SELECT id, span_start, span_end FROM evidence_units"
    ).fetchone()
    assert domain_unit == infrastructure_unit
    assert domain.execute(
        "SELECT evidence_unit_id FROM claim_evidence_units"
    ).fetchone() == (domain_unit[0],)
    assert infrastructure.execute(
        "SELECT evidence_unit_id FROM claim_evidence_units"
    ).fetchone() == (domain_unit[0],)


def test_touching_half_open_intervals_remain_distinct_support():
    conn = _db()
    frame = "sha256:" + "b" * 64
    _direct_anchored_claim(conn, "claim-a", "alice", start=0, end=5, text_hash=frame)
    _direct_anchored_claim(conn, "claim-b", "bob", start=5, end=10, text_hash=frame)
    rebuild_evidence_units([conn])

    assessment = assess_evidence_support(conn, ["claim-a", "claim-b"])
    assert conn.execute("SELECT COUNT(*) FROM evidence_units").fetchone() == (2,)
    assert assessment.independent_sources == 2
    assert assessment.overlap_unknown_claims == frozenset()


def test_different_text_frames_on_one_asset_page_are_overlap_unknown():
    conn = _db()
    _direct_anchored_claim(
        conn,
        "claim-a",
        "alice",
        start=0,
        end=5,
        text_hash="sha256:" + "b" * 64,
    )
    _direct_anchored_claim(
        conn,
        "claim-b",
        "bob",
        start=20,
        end=25,
        text_hash="sha256:" + "c" * 64,
    )
    rebuild_evidence_units([conn])

    assessment = assess_evidence_support(conn, ["claim-a", "claim-b"])
    assert assessment.independent_sources == 0
    assert assessment.scored_claims == 0
    assert assessment.overlap_unknown_claims == frozenset({"claim-a", "claim-b"})


def test_digest1_claims_remain_importable_but_independence_unknown():
    parsed = {
        "frontmatter": {
            "schema": "anomalica/digest/1",
            "record_id": "legacy-record",
            "record_title": "Legacy record",
            "content_hash": "sha256:" + "b" * 64,
            "record": {
                "id": "legacy-record",
                "title": "Legacy record",
                "content_hash": "sha256:" + "b" * 64,
            },
        },
        "terminology": {},
        "nodes": [
            {"id": "alice", "node_type": "person", "name": "Alice", "metadata": None}
        ],
        "domain_claims": [
            {
                "id": "legacy-claim",
                "claim_type": "observation",
                "content": "Alice saw it.",
                "original_excerpt": "saw it",
                "speaker": "Alice",
                "provenance_chain": {
                    "origin_kind": "speaker",
                    "origin": "Alice",
                    "relay": [],
                },
                "node_references": [],
                "location_in_record": "page 1",
            }
        ],
        "infrastructure_claims": [],
    }
    conn = _db()

    import_extraction(conn, parsed)

    assessment = assess_evidence_support(conn, ["legacy-claim"])
    assert assessment.independent_sources == 0
    assert assessment.scored_claims == 0
    assert assessment.unscored_claims == 1
    assert conn.execute("SELECT COUNT(*) FROM claim_anchors").fetchone() == (0,)
