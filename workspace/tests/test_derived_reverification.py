from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
import yaml

from assimilator.database import init_db
from assimilator.derived_reverification import (
    DerivedReverificationError,
    MappingOperationRecordSource,
    _corroboration_prompt,
    build_inventory,
    complete_item,
    inventory_report,
    materialise_receipts,
    read_document,
    validate_receipts,
    validate_materialised_receipts,
    write_inventory,
    _curation_fingerprint,
)

NOW = "2026-09-15T00:00:00+00:00"


def test_curation_fingerprint_excludes_replay_dispositions(tmp_path):
    curation = tmp_path / "curation"
    curation.mkdir()
    (curation / "renames.yaml").write_text("rename evidence")
    before = _curation_fingerprint(curation)
    (curation / "replay-dispositions.yaml").write_text("disposition evidence")
    assert _curation_fingerprint(curation) == before
    (curation / "renames.yaml").write_text("changed rename evidence")
    assert _curation_fingerprint(curation) != before


def _claim(text: str) -> dict:
    return {"text": text, "type": "fact", "quote": text, "location": "line 1"}


def _digest(root: Path, suffix: str, claims: list[dict]) -> None:
    (root / f"{suffix}.yaml").write_text(
        yaml.safe_dump(
            {
                "run_kind": "production",
                "schema": "anomalica/digest/1",
                "record": {
                    "id": f"digest-{suffix}",
                    "title": suffix,
                    "content_hash": f"sha256:{suffix * 64}",
                },
                "domain_claims": claims,
            },
            sort_keys=False,
        )
    )


def _insert_record(
    conn: sqlite3.Connection,
    record_id: str,
    record_hash: str,
    claims: list[tuple[str, dict]],
) -> None:
    conn.execute(
        "INSERT INTO records (id,title,content_hash,created_at,work_id) VALUES (?,?,?,?,?)",
        (record_id, record_id, record_hash, NOW, record_id),
    )
    conn.execute(
        "INSERT INTO digest_import_receipts "
        "(record_content_hash,record_id,digest_path,digest_sha256,import_generation,"
        "claim_manifest_sha256,imported_at) VALUES (?,?,?,?,?,?,?)",
        (
            record_hash,
            record_id,
            f"digests/{record_id}.yaml",
            "sha256:" + "0" * 64,
            7,
            "sha256:" + "1" * 64,
            NOW,
        ),
    )
    for claim_id, claim in claims:
        conn.execute(
            "INSERT INTO claims "
            "(id,content,original_excerpt,claim_type,record_id,location_in_record,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                claim_id,
                claim["text"],
                claim["quote"],
                claim["type"],
                record_id,
                claim["location"],
                NOW,
            ),
        )


def _db(path: Path, records: list[tuple[str, str, list[tuple[str, dict]]]]) -> None:
    conn = sqlite3.connect(path)
    init_db(conn)
    for record in records:
        _insert_record(conn, *record)
    conn.commit()
    conn.close()


def _fixture(tmp_path: Path):
    digests = tmp_path / "digests"
    curation = tmp_path / "curation"
    candidate_dir = tmp_path / "candidate"
    digests.mkdir()
    curation.mkdir()
    candidate_dir.mkdir()
    policy = tmp_path / "model-policy.yaml"
    policy.write_text("schema: anomalica/model-policy/1\n")
    claims = {key: _claim(f"claim {key}") for key in "abcd"}
    source_records = []
    candidate_records = []
    for suffix in "abcd":
        _digest(digests, suffix, [] if suffix == "d" else [claims[suffix]])
        source_records.append(
            (
                f"source-{suffix}",
                f"sha256:{suffix * 64}",
                [(f"source-claim-{suffix}", claims[suffix])],
            )
        )
        candidate_records.append(
            (
                f"candidate-{suffix}",
                f"sha256:{suffix * 64}",
                (
                    [(f"candidate-claim-{suffix}", claims[suffix])]
                    if suffix != "d"
                    else []
                ),
            )
        )
    source_db = tmp_path / "source.db"
    candidate_db = candidate_dir / "knowledge.db"
    _db(source_db, source_records)
    _db(candidate_db, candidate_records)
    source = sqlite3.connect(source_db)
    source.executemany(
        "INSERT INTO corroborations VALUES (?,?,?)",
        [
            ("source-claim-a", "source-claim-b", 0.91),
            ("source-claim-b", "source-claim-c", 0.92),
            ("source-claim-a", "source-claim-d", 0.93),
        ],
    )
    source.executemany(
        "INSERT INTO record_relations (record_a,record_b,verdict,judged_at) VALUES (?,?,?,?)",
        [
            ("source-a", "source-b", "same_subject", NOW),
            ("source-a", "source-d", "unrelated", NOW),
        ],
    )
    source.commit()
    source.close()
    return source_db, candidate_dir, candidate_db, digests, curation, policy


def _plan(tmp_path: Path):
    fixture = _fixture(tmp_path)
    source, candidate_dir, candidate, digests, curation, policy = fixture
    inventory = build_inventory(source, candidate, digests, curation, policy, NOW)
    written = write_inventory(candidate_dir, inventory)
    return fixture, inventory, Path(written["path"])


def _complete(
    path: Path, source_db: Path, policy: Path
) -> MappingOperationRecordSource:
    items = [
        item
        for item in read_document(path)["items"]
        if item["kind"] == "corroboration" and item["status"] == "pending"
    ]
    records = {}
    outcomes = {}
    source = sqlite3.connect(f"{source_db.resolve().as_uri()}?mode=ro", uri=True)
    for index, item in enumerate(items):
        verdict = (
            "accepted"
            if index == 0 and item["prior"].get("verdict") != "rejected"
            else "rejected"
        )
        operation_id = f"operation-{index}"
        prompt_sha256 = hashlib.sha256(
            _corroboration_prompt(item["id"], item["locators"], source).encode()
        ).hexdigest()
        current_similarity = item["prior"]["similarity"] + 0.01
        receipt = {
            "id": operation_id,
            "model": "test-model",
            "prompt_sha256": prompt_sha256,
            "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        }
        outcomes[item["id"]] = {
            "verdict": verdict,
            "current_similarity": current_similarity,
            "operation_receipt": receipt,
            "verified_at": NOW,
        }
        records[operation_id] = {
            **receipt,
            "component": "assimilator",
            "operation": "corroborate",
            "item_id": item["id"],
            "verdict": verdict,
            "current_similarity": current_similarity,
            "outcome": "ok",
            "provider_started": True,
            "completed_at": NOW,
        }
    source.close()
    operation_source = MappingOperationRecordSource(records)
    for item in items:
        complete_item(
            path, item["id"], outcomes[item["id"]], source_db, policy, operation_source
        )
    return operation_source


def test_schema_roundtrip_exact_vocabulary_sorting_and_idempotent_write(tmp_path):
    fixture, inventory, path = _plan(tmp_path)
    before = path.read_bytes()
    report = inventory_report(inventory)
    second = write_inventory(fixture[1], inventory)
    assert set(inventory) == {
        "schema",
        "source_graph_input_fingerprint",
        "candidate_graph_input_fingerprint",
        "candidate_curation_fingerprint",
        "items",
    }
    assert inventory["schema"] == "anomalica/rebuild-derived-reverification/1"
    assert read_document(path) == inventory
    assert path.read_bytes() == before
    assert second["document_sha256"] == hashlib.sha256(before).hexdigest()
    assert report == {
        "counts": {
            "corroboration": {
                "pending": 2,
                "completed": 0,
                "dropped_source_absent": 1,
                "blocked_candidate_import_loss": 0,
            },
            "record_relation": {
                "pending": 2,
                "completed": 0,
                "dropped_source_absent": 0,
                "blocked_candidate_import_loss": 0,
            },
            "replacement_gate": {"required": 2, "not_required": 3},
        },
        "model_calls_made": 0,
        "cost_route_requirement": report["cost_route_requirement"],
    }
    assert inventory["items"] == sorted(
        inventory["items"], key=lambda item: (item["kind"], item["id"])
    )
    assert len({item["id"] for item in inventory["items"]}) == len(inventory["items"])
    dropped = [
        item for item in inventory["items"] if item["status"] == "dropped_source_absent"
    ]
    assert all(item["replacement_gate"] == "not_required" for item in dropped)
    assert all(item["outcome"]["checked_at"] == NOW for item in dropped)
    assert all(
        item["outcome"]["absent_locators"]
        == sorted(item["outcome"]["absent_locators"], key=str)
        for item in dropped
    )
    relations = [
        item for item in inventory["items"] if item["kind"] == "record_relation"
    ]
    assert {item["prior"]["verdict"] for item in relations} == {
        "same_subject",
        "unrelated",
    }
    assert all(item["replacement_gate"] == "not_required" for item in relations)


def test_atomic_completion_and_validation_report_exact_document_sha(tmp_path):
    fixture, inventory, path = _plan(tmp_path)
    source, candidate_dir, candidate, digests, curation, policy = fixture
    operations = _complete(path, source, policy)
    completed_bytes = path.read_bytes()
    write_inventory(candidate_dir, inventory)
    assert path.read_bytes() == completed_bytes
    document = read_document(path)
    assert "receipts" not in document
    assert sum(item["status"] == "completed" for item in document["items"]) == 2
    report = validate_receipts(
        path, source, candidate, digests, curation, policy, NOW, operations
    )
    assert report["valid"] is True
    assert report["completed_required_corroborations"] == 2
    assert report["accepted"] == report["rejected"] == 1
    assert report["pending_record_relations"] == 2
    assert report["document_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("document", "candidate_curation_fingerprint", "sha256:" + "f" * 64),
        ("outcome", "prompt_sha256", "f" * 64),
        ("outcome", "current_similarity", 0.1),
        ("operation", "outcome", "error"),
        ("operation", "verdict", "rejected"),
    ],
)
def test_tampering_is_rejected(tmp_path, target, field, value):
    fixture, _inventory, path = _plan(tmp_path)
    source, _candidate_dir, candidate, digests, curation, policy = fixture
    operations = _complete(path, source, policy)
    document = read_document(path)
    records = dict(operations._records)
    completed = next(
        item for item in document["items"] if item["status"] == "completed"
    )
    if target == "document":
        document[field] = value
    elif target == "outcome":
        if field == "prompt_sha256":
            completed["outcome"]["operation_receipt"][field] = value
        else:
            completed["outcome"][field] = value
    else:
        operation_id = completed["outcome"]["operation_receipt"]["id"]
        records[operation_id] = {**records[operation_id], field: value}
        operations = MappingOperationRecordSource(records)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(DerivedReverificationError):
        validate_receipts(
            path, source, candidate, digests, curation, policy, NOW, operations
        )


def test_pending_required_corroboration_blocks_replacement_but_relation_does_not(
    tmp_path,
):
    fixture, _inventory, path = _plan(tmp_path)
    source, _candidate_dir, candidate, digests, curation, policy = fixture
    with pytest.raises(
        DerivedReverificationError,
        match="2 required corroboration items are incomplete",
    ):
        validate_receipts(
            path,
            source,
            candidate,
            digests,
            curation,
            policy,
            NOW,
            MappingOperationRecordSource({}),
        )


def test_unrelated_relation_survives_when_both_exact_record_hashes_resolve(tmp_path):
    source, _candidate_dir, candidate, digests, curation, policy = _fixture(tmp_path)

    inventory = build_inventory(source, candidate, digests, curation, policy, NOW)
    unrelated = next(
        item
        for item in inventory["items"]
        if item["kind"] == "record_relation" and item["prior"]["verdict"] == "unrelated"
    )

    assert unrelated["status"] == "pending"
    assert unrelated["replacement_gate"] == "not_required"
    assert "outcome" not in unrelated


@pytest.mark.parametrize("missing", ["claim", "record"])
def test_canonical_claim_missing_only_from_candidate_blocks_replacement(
    tmp_path, missing
):
    source, candidate_dir, candidate, digests, curation, policy = _fixture(tmp_path)
    conn = sqlite3.connect(candidate)
    if missing == "claim":
        conn.execute("DELETE FROM claims WHERE id='candidate-claim-c'")
    else:
        conn.execute("DELETE FROM records WHERE id='candidate-c'")
    conn.commit()
    conn.close()

    inventory = build_inventory(source, candidate, digests, curation, policy, NOW)
    blocked = [
        item
        for item in inventory["items"]
        if item["status"] == "blocked_candidate_import_loss"
    ]
    assert len(blocked) == 1
    assert blocked[0]["kind"] == "corroboration"
    assert blocked[0]["replacement_gate"] == "required"
    assert blocked[0]["outcome"]["candidate_missing_locators"] == [
        locator
        for locator in blocked[0]["locators"]
        if locator["record_content_hash"] == "sha256:" + "c" * 64
    ]

    path = Path(write_inventory(candidate_dir, inventory)["path"])
    with pytest.raises(
        DerivedReverificationError,
        match="1 items are blocked by candidate import loss",
    ):
        validate_receipts(
            path,
            source,
            candidate,
            digests,
            curation,
            policy,
            NOW,
            MappingOperationRecordSource({}),
        )


def test_canonical_claim_absence_drops_stale_candidate_claim(tmp_path):
    source, _candidate_dir, candidate, digests, curation, policy = _fixture(tmp_path)
    conn = sqlite3.connect(candidate)
    claim = _claim("claim d")
    conn.execute(
        "INSERT INTO claims "
        "(id,content,original_excerpt,claim_type,record_id,location_in_record,created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            "candidate-claim-d",
            claim["text"],
            claim["quote"],
            claim["type"],
            "candidate-d",
            claim["location"],
            NOW,
        ),
    )
    conn.commit()
    conn.close()

    inventory = build_inventory(source, candidate, digests, curation, policy, NOW)
    dropped = next(
        item
        for item in inventory["items"]
        if item["kind"] == "corroboration"
        and any(
            locator["record_content_hash"] == "sha256:" + "d" * 64
            for locator in item["locators"]
        )
    )
    assert dropped["status"] == "dropped_source_absent"
    assert dropped["replacement_gate"] == "not_required"


def test_source_corroboration_rejection_is_reverified_and_preserved(tmp_path):
    source, candidate_dir, candidate, digests, curation, policy = _fixture(tmp_path)
    conn = sqlite3.connect(source)
    conn.execute(
        "INSERT INTO corroboration_rejections VALUES (?,?,?,?,?)",
        ("source-claim-a", "source-claim-c", 0.88, "prior-model", NOW),
    )
    conn.commit()
    conn.close()

    inventory = build_inventory(source, candidate, digests, curation, policy, NOW)
    prior_rejection = next(
        item
        for item in inventory["items"]
        if item["kind"] == "corroboration"
        and item["prior"].get("verdict") == "rejected"
    )
    assert prior_rejection["status"] == "pending"
    assert prior_rejection["replacement_gate"] == "required"
    assert prior_rejection["prior"] == {
        "verdict": "rejected",
        "similarity": 0.88,
        "model": "prior-model",
        "adjudicated_at": NOW,
    }

    path = Path(write_inventory(candidate_dir, inventory)["path"])
    operations = _complete(path, source, policy)
    materialise_receipts(
        path, source, candidate, digests, curation, policy, NOW, operations
    )
    conn = sqlite3.connect(candidate)
    rejection = conn.execute(
        "SELECT claim_a, claim_b, model, rejected_at "
        "FROM corroboration_rejections WHERE claim_a=? AND claim_b=?",
        ("candidate-claim-a", "candidate-claim-c"),
    ).fetchone()
    conn.close()
    assert rejection == (
        "candidate-claim-a",
        "candidate-claim-c",
        "test-model",
        NOW,
    )


def test_ambiguous_candidate_locator_is_rejected(tmp_path):
    source, _candidate_dir, candidate, digests, curation, policy = _fixture(tmp_path)
    conn = sqlite3.connect(candidate)
    conn.execute(
        "INSERT INTO records (id,title,content_hash,created_at) VALUES (?,?,?,?)",
        ("ambiguous", "ambiguous", "sha256:" + "a" * 64, NOW),
    )
    conn.commit()
    conn.close()
    with pytest.raises(
        DerivedReverificationError, match="resolved to 2 candidate records"
    ):
        build_inventory(source, candidate, digests, curation, policy, NOW)


def test_materialisation_reconciles_rows_exactly_and_is_idempotent(tmp_path):
    fixture, _inventory, path = _plan(tmp_path)
    source, _candidate_dir, candidate, digests, curation, policy = fixture
    operations = _complete(path, source, policy)
    conn = sqlite3.connect(candidate)
    conn.execute(
        "INSERT INTO corroborations VALUES ('candidate-claim-a','candidate-claim-c',0.5)"
    )
    conn.execute(
        "INSERT INTO corroboration_rejections VALUES "
        "('stale-a','stale-b',0.4,'old','2020-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    validation = validate_receipts(
        path, source, candidate, digests, curation, policy, NOW, operations
    )
    assert validate_materialised_receipts(validation, candidate)["valid"] is False
    first = materialise_receipts(
        path, source, candidate, digests, curation, policy, NOW, operations
    )
    second = materialise_receipts(
        path, source, candidate, digests, curation, policy, NOW, operations
    )
    assert validate_materialised_receipts(validation, candidate)["valid"] is True
    conn = sqlite3.connect(candidate)
    accepted = conn.execute(
        "SELECT claim_a,claim_b,similarity FROM corroborations"
    ).fetchall()
    rejected = conn.execute(
        "SELECT claim_a,claim_b,similarity,model,rejected_at FROM corroboration_rejections"
    ).fetchall()
    relations = conn.execute("SELECT count(*) FROM record_relations").fetchone()[0]
    conn.close()
    assert first == second
    assert len(accepted) == len(rejected) == 1
    assert {accepted[0][:2], rejected[0][:2]} == {
        ("candidate-claim-a", "candidate-claim-b"),
        ("candidate-claim-b", "candidate-claim-c"),
    }
    assert rejected[0][3:] == ("test-model", NOW)
    assert relations == 0


def test_materialisation_rolls_back_both_tables_atomically(tmp_path):
    fixture, _inventory, path = _plan(tmp_path)
    source, _candidate_dir, candidate, digests, curation, policy = fixture
    operations = _complete(path, source, policy)
    conn = sqlite3.connect(candidate)
    conn.execute(
        "INSERT INTO corroborations VALUES ('candidate-claim-a','candidate-claim-c',0.5)"
    )
    conn.execute(
        "CREATE TRIGGER reject_replay BEFORE INSERT ON corroboration_rejections "
        "BEGIN SELECT RAISE(ABORT, 'test rollback'); END"
    )
    conn.commit()
    before = conn.execute("SELECT * FROM corroborations").fetchall()
    conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="test rollback"):
        materialise_receipts(
            path, source, candidate, digests, curation, policy, NOW, operations
        )
    conn = sqlite3.connect(candidate)
    assert conn.execute("SELECT * FROM corroborations").fetchall() == before
    assert (
        conn.execute("SELECT count(*) FROM corroboration_rejections").fetchone()[0] == 0
    )
    conn.close()
