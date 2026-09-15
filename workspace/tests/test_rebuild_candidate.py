from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import yaml
from click.testing import CliRunner

from assimilator.cli import main
from assimilator.database import init_db
from assimilator.derived_reverification import (
    FileOperationRecordSource,
    OPERATION_RECORD_SCHEMA,
    build_inventory,
    write_inventory,
)
from assimilator.digest_files import CURRENT_IMPORT_GENERATION, digest_file_identity
from assimilator.rebuild_candidate import (
    CURATION_REPLAY_REPORT_SCHEMA,
    curation_state_sha256,
    validate_rebuild_candidate,
)
from assimilator.scheduler import _digest_index, graph_input_diagnostics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(digests: Path, *, suffix: str = "a") -> tuple[Path, str, str]:
    digests.mkdir(exist_ok=True)
    content_hash = suffix * 64
    record_id = f"record-{suffix}"
    path = digests / f"record-{suffix}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_kind": "production",
                "schema": "anomalica/digest/1",
                "record": {
                    "id": record_id,
                    "title": f"Record {suffix}",
                    "content_hash": f"sha256:{content_hash}",
                },
            },
            sort_keys=False,
        )
    )
    return path, record_id, content_hash


def _database(
    path: Path,
    *,
    digest: Path | None = None,
    record_id: str = "",
    content_hash: str = "",
    generation: int = CURRENT_IMPORT_GENERATION,
    receipt_sha256: str | None = None,
) -> None:
    conn = sqlite3.connect(path)
    init_db(conn)
    if digest is not None:
        identity = digest_file_identity(digest, digest.parent)
        record_title = yaml.safe_load(digest.read_text())["record"]["title"]
        manifest = json.dumps(
            {"section": "domain", "claims": []}, separators=(",", ":")
        ).encode()
        conn.execute(
            "INSERT INTO records "
            "(id, title, content_hash, metadata, created_at, work_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                record_title,
                f"sha256:{content_hash}",
                json.dumps({"run_kind": "production"}),
                "2026-09-15T00:00:00+00:00",
                record_id,
            ),
        )
        conn.execute(
            "INSERT INTO digest_import_receipts "
            "(record_content_hash, record_id, digest_path, digest_sha256, "
            "import_generation, claim_manifest_sha256, imported_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"sha256:{content_hash}",
                record_id,
                identity["digest_path"],
                receipt_sha256 or identity["digest_sha256"],
                generation,
                "sha256:" + hashlib.sha256(manifest).hexdigest(),
                "2026-09-15T00:00:00+00:00",
            ),
        )
    conn.commit()
    conn.close()


def _valid_candidate(tmp_path: Path) -> tuple[Path, Path, Path, Path, str, str]:
    digests = tmp_path / "digests"
    digest, record_id, content_hash = _digest(digests)
    domain = tmp_path / "knowledge-candidate.db"
    infrastructure = tmp_path / "infrastructure-candidate.db"
    _database(
        domain,
        digest=digest,
        record_id=record_id,
        content_hash=content_hash,
    )
    _database(infrastructure)
    return domain, infrastructure, digests, digest, record_id, content_hash


def _curation_report(domain: Path, digests: Path) -> tuple[Path, dict]:
    curation = Path(os.environ["ANOMALICA_CURATION_DIR"])
    (curation / "merges.yaml").write_text("")
    conn = sqlite3.connect(domain)
    fingerprint = graph_input_diagnostics(
        conn, _digest_index(digests.resolve()), digests.resolve()
    )["fingerprint"]
    state_sha256 = curation_state_sha256(conn)
    conn.close()
    report = {
        "schema": CURATION_REPLAY_REPORT_SCHEMA,
        "source_graph_input_fingerprint": fingerprint,
        "candidate_graph_input_fingerprint": fingerprint,
        "disposition_ledger_sha256": (
            "sha256:bbef14384bcdc68220da8d53888bb79e29a9c54b5a05d3c12ab490470b3efb50"
        ),
        "curation_state_sha256": state_sha256,
        "operation_inventory": [],
        "diagnostics": [],
        "blockers": [],
        "disposition_replacement_safe": True,
    }
    return curation, report


def _derived_inputs(
    root: Path, domain: Path, digests: Path, curation: Path
) -> tuple[Path, Path, str, FileOperationRecordSource]:
    policy = root / "model-policy.yaml"
    policy.write_text("models: {}\n")
    checked_at = "2026-09-15T00:00:00Z"
    inventory = build_inventory(domain, domain, digests, curation, policy, checked_at)
    write_inventory(root, inventory)
    document = root / "derived-reverification.yaml"
    records = root / "operation-records.jsonl"
    records.write_text(
        yaml.safe_dump(
            {"schema": OPERATION_RECORD_SCHEMA, "records": []}, sort_keys=False
        )
    )
    return policy, document, checked_at, FileOperationRecordSource(records)


def _validate(domain: Path, infrastructure: Path, digests: Path) -> dict:
    curation, replay_report = _curation_report(domain, digests)
    policy, document, checked_at, records = _derived_inputs(
        domain.parent, domain, digests, curation
    )
    return validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )


def test_valid_candidate_is_deterministic_and_does_not_mutate_databases(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    before = (_sha256(domain), _sha256(infrastructure))

    curation, replay_report = _curation_report(domain, digests)
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )
    operation_records = tmp_path / "operation-records.jsonl"
    report_path = tmp_path / "curation-replay-report.json"
    report_path.write_text(json.dumps(replay_report))
    first = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )
    second = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )
    cli_result = CliRunner().invoke(
        main,
        [
            "validate-rebuild-candidate",
            str(domain),
            str(infrastructure),
            str(digests),
            str(curation),
            str(report_path),
            str(domain),
            str(document),
            str(operation_records),
            "--policy",
            str(policy),
            "--checked-at",
            checked_at,
        ],
    )

    assert first == second
    assert first["valid"] is True, json.dumps(first, indent=2)
    assert first["canonical_digest_count"] == 1
    assert first["graph_input_fingerprint"].startswith("sha256:")
    assert first["curation_fingerprint"].startswith("sha256:")
    assert first["databases"]["domain"]["native_deltas"]["current"] == 1
    assert first["databases"]["infrastructure"]["native_deltas"] is None
    assert first["databases"]["domain"]["total_changes"] == 0
    assert first["databases"]["infrastructure"]["total_changes"] == 0
    assert cli_result.exit_code == 0
    assert json.loads(cli_result.output) == first
    assert (_sha256(domain), _sha256(infrastructure)) == before


def test_validation_replays_only_the_supplied_curation_source(tmp_path, monkeypatch):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    hostile = Path(os.environ["ANOMALICA_CURATION_DIR"])
    (hostile / "merges.yaml").write_text("op: merge\n")
    supplied = tmp_path / "supplied-curation"
    supplied.mkdir()
    (supplied / "merges.yaml").write_text("")
    conn = sqlite3.connect(domain)
    fingerprint = graph_input_diagnostics(
        conn,
        _digest_index(digests.resolve()),
        digests.resolve(),
        supplied,
    )["fingerprint"]
    replay_report = {
        "schema": CURATION_REPLAY_REPORT_SCHEMA,
        "source_graph_input_fingerprint": fingerprint,
        "candidate_graph_input_fingerprint": fingerprint,
        "disposition_ledger_sha256": (
            "sha256:bbef14384bcdc68220da8d53888bb79e29a9c54b5a05d3c12ab490470b3efb50"
        ),
        "curation_state_sha256": curation_state_sha256(conn),
        "operation_inventory": [],
        "diagnostics": [],
        "blockers": [],
        "disposition_replacement_safe": True,
    }
    conn.close()
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, supplied
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        supplied,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is True, json.dumps(report, indent=2)
    assert os.environ["ANOMALICA_CURATION_DIR"] == str(hostile)


def test_real_canonical_claims_match_an_independent_import(tmp_path, monkeypatch):
    digests = tmp_path / "digests"
    digests.mkdir()
    digest = digests / "claim.yaml"
    digest.write_text(
        yaml.safe_dump(
            {
                "run_kind": "production",
                "schema": "anomalica/digest/1",
                "record": {
                    "id": "record-claim",
                    "title": "Claim record",
                    "content_hash": "sha256:" + "c" * 64,
                },
                "domain_claims": [
                    {
                        "id": "claim-1",
                        "text": "A canonical claim.",
                        "type": "observation",
                        "quote": "A canonical claim.",
                        "location": "line 1",
                    }
                ],
                "infrastructure_claims": [
                    {
                        "id": "claim-infrastructure-1",
                        "text": "A canonical infrastructure claim.",
                        "type": "observation",
                        "quote": "A canonical infrastructure claim.",
                        "location": "line 2",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    source = tmp_path / "source.db"
    _database(source)
    candidate = tmp_path / "candidate" / "knowledge.db"
    (tmp_path / "ingests").mkdir()
    monkeypatch.setenv("ANOMALICA_INGESTS_DIR", str(tmp_path / "ingests"))
    curation = Path(os.environ["ANOMALICA_CURATION_DIR"])
    (curation / "tags.yaml").write_text(
        yaml.safe_dump(
            {
                "op": "tag",
                "tag_id": "tag-created-topic",
                "at": "2026-09-15T00:00:00Z",
                "by": "reviewer",
                "node": {
                    "name": "Created by tag",
                    "node_type": "topic",
                    "prior_names": [],
                },
                "record": {
                    "content_hash": "sha256:" + "c" * 64,
                    "title": "Claim record",
                },
                "note": None,
            },
            sort_keys=False,
        )
    )
    result = CliRunner().invoke(
        main,
        ["--db", str(candidate), "rebuild", str(digests), "--source-db", str(source)],
    )
    assert result.exit_code == 0, result.output
    source.write_bytes(candidate.read_bytes())
    replay_report = json.loads(
        (candidate.parent / "curation-replay-report.json").read_text()
    )
    conn = sqlite3.connect(source)
    replay_report["source_graph_input_fingerprint"] = graph_input_diagnostics(
        conn,
        _digest_index(digests.resolve()),
        digests.resolve(),
        Path(os.environ["ANOMALICA_CURATION_DIR"]),
    )["fingerprint"]
    conn.close()
    infrastructure = candidate.parent / "infrastructure.db"
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, candidate, digests, curation
    )

    report = validate_rebuild_candidate(
        candidate,
        infrastructure,
        digests,
        curation,
        replay_report,
        source,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is True, json.dumps(report, indent=2)
    materialisation = report["databases"]["domain"]["receipt_checks"][
        "claim_materialisation"
    ]
    assert materialisation == {
        "ok": True,
        "claim_manifest_mismatches": [],
        "claims_match": True,
        "claim_node_refs_match": True,
        "records_match": True,
        "nodes_match": True,
        "aliases_match": True,
    }

    conn = sqlite3.connect(infrastructure)
    conn.execute("DELETE FROM claims")
    conn.commit()
    conn.close()
    corrupted = validate_rebuild_candidate(
        candidate,
        infrastructure,
        digests,
        curation,
        replay_report,
        source,
        document,
        policy,
        checked_at,
        records,
    )
    assert corrupted["valid"] is False
    assert (
        corrupted["databases"]["infrastructure"]["receipt_checks"][
            "claim_materialisation"
        ]["claims_match"]
        is False
    )


def test_curation_report_must_cover_the_material_source_inventory(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    curation, replay_report = _curation_report(domain, digests)
    (curation / "merges.yaml").write_text(
        yaml.safe_dump_all(
            [
                {
                    "op": "merge",
                    "merge_id": "merge-material",
                    "at": "2026-09-01T00:00:00Z",
                    "canonical_name": "Alpha",
                    "survivor": {
                        "name": "Alpha",
                        "node_type": "concept",
                        "prior_names": [],
                    },
                    "victims": [
                        {
                            "name": "Beta",
                            "node_type": "concept",
                            "prior_names": [],
                        }
                    ],
                },
                {
                    "op": "undo",
                    "merge_id": "merge-material",
                    "at": "2026-09-02T00:00:00Z",
                },
            ],
            sort_keys=False,
        )
    )
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is False
    assert (
        "curation replay report does not match the material source inventory"
        in report["curation_replay"]["errors"]
    )


def test_curation_report_cannot_claim_a_different_strict_replay_outcome(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    curation = Path(os.environ["ANOMALICA_CURATION_DIR"])
    (curation / "merges.yaml").write_text(
        yaml.safe_dump_all(
            [
                {
                    "op": "merge",
                    "merge_id": "merge-compensated",
                    "at": "2026-09-01T00:00:00Z",
                    "canonical_name": "Alpha",
                    "survivor": {
                        "name": "Alpha",
                        "node_type": "concept",
                        "prior_names": [],
                    },
                    "victims": [
                        {
                            "name": "Beta",
                            "node_type": "concept",
                            "prior_names": [],
                        }
                    ],
                },
                {
                    "op": "undo",
                    "merge_id": "merge-compensated",
                    "at": "2026-09-02T00:00:00Z",
                },
            ],
            sort_keys=False,
        )
    )
    _, replay_report = _curation_report(domain, digests)
    replay_report["operation_inventory"] = [["merge", "merge-compensated"]]
    replay_report["diagnostics"] = [
        {
            "source_phase": "merge",
            "operation_id": "merge-compensated",
            "outcome": "applied_normally",
            "cause": "merge_materialised",
        }
    ]
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is False
    assert (
        "curation replay report does not match actual strict replay diagnostics"
        in report["curation_replay"]["errors"]
    )


def test_curation_report_source_fingerprint_is_recomputed_from_supplied_db(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    curation, replay_report = _curation_report(domain, digests)
    replay_report["source_graph_input_fingerprint"] = "sha256:" + "f" * 64
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is False
    assert (
        "curation replay source fingerprint does not match"
        in report["curation_replay"]["errors"]
    )


def test_claim_manifest_and_exact_claim_materialisation_are_validated(tmp_path):
    domain, infrastructure, digests, _digest_path, record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    conn = sqlite3.connect(domain)
    conn.execute(
        "UPDATE digest_import_receipts SET claim_manifest_sha256 = ?",
        ("sha256:" + "f" * 64,),
    )
    conn.execute(
        "INSERT INTO claims "
        "(id, content, claim_type, record_id, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("forged-claim", "Not canonical", "fact", record_id, 1.0, "now"),
    )
    conn.execute(
        "UPDATE records SET title = 'Not canonical' WHERE id = ?", (record_id,)
    )
    conn.commit()
    conn.close()
    curation, replay_report = _curation_report(domain, digests)
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    materialisation = report["databases"]["domain"]["receipt_checks"][
        "claim_materialisation"
    ]
    assert report["valid"] is False
    assert materialisation["claim_manifest_mismatches"]
    assert materialisation["claims_match"] is False
    assert materialisation["records_match"] is False


def test_curation_state_hash_covers_all_dependent_replay_tables(tmp_path):
    database = tmp_path / "curation-state.db"
    _database(database)
    conn = sqlite3.connect(database)
    conn.execute(
        "INSERT INTO nodes (id, name, node_type, created_at) VALUES (?, ?, ?, ?)",
        ("node-1", "Alpha", "concept", "now"),
    )
    conn.execute(
        "INSERT INTO records (id, title, content_hash, created_at) VALUES (?, ?, ?, ?)",
        ("record-1", "Record", "sha256:" + "a" * 64, "now"),
    )
    conn.execute(
        "INSERT INTO claims "
        "(id, content, claim_type, record_id, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("claim-1", "Claim", "fact", "record-1", 1.0, "now"),
    )
    conn.execute(
        "INSERT INTO pages "
        "(page_id, name, slug, node_type, created_at) VALUES (?, ?, ?, ?, ?)",
        ("page-base", "Base", "base", "concept", "now"),
    )
    conn.commit()
    baseline = curation_state_sha256(conn)
    conn.execute("SAVEPOINT node_metadata")
    conn.execute(
        "UPDATE nodes SET metadata = ? WHERE id = ?",
        ('{"changed":true}', "node-1"),
    )
    assert curation_state_sha256(conn) != baseline
    conn.execute("ROLLBACK TO node_metadata")
    conn.execute("RELEASE node_metadata")
    insertions = {
        "claim_ref_status": (
            "INSERT INTO claim_ref_status "
            "(claim_id, node_id, status, set_at) VALUES (?, ?, ?, ?)",
            ("claim-1", "node-1", "verified", "now"),
        ),
        "record_nodes": (
            "INSERT INTO record_nodes (record_id, node_id) VALUES (?, ?)",
            ("record-1", "node-1"),
        ),
        "record_tags": (
            "INSERT INTO record_tags (tag_id, status, created_at) VALUES (?, ?, ?)",
            ("tag-1", "pending", "now"),
        ),
        "pages": (
            "INSERT INTO pages "
            "(page_id, name, slug, node_type, created_at) VALUES (?, ?, ?, ?, ?)",
            ("page-2", "Second", "second", "concept", "now"),
        ),
        "page_members": (
            "INSERT INTO page_members (page_id, node_id, position) VALUES (?, ?, ?)",
            ("page-base", "node-1", 0),
        ),
        "page_vetoes": (
            "INSERT INTO page_vetoes (veto_id, node_id, created_at) VALUES (?, ?, ?)",
            ("veto-1", "node-1", "now"),
        ),
        "superseded_pages": (
            "INSERT INTO superseded_pages "
            "(section, slug, page_id, node_id, reason) VALUES (?, ?, ?, ?, ?)",
            ("concepts", "alpha", "page-base", "node-1", "composed"),
        ),
    }
    for table, (statement, parameters) in insertions.items():
        conn.execute("SAVEPOINT state_table")
        conn.execute(statement, parameters)
        assert curation_state_sha256(conn) != baseline, table
        conn.execute("ROLLBACK TO state_table")
        conn.execute("RELEASE state_table")
    conn.close()


def test_curation_report_is_bound_to_candidate_relational_state(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    curation, replay_report = _curation_report(domain, digests)
    conn = sqlite3.connect(domain)
    conn.execute(
        "INSERT INTO nodes (id, name, node_type, created_at) VALUES (?, ?, ?, ?)",
        ("node-after-report", "Alpha", "concept", "2026-09-15T00:00:00Z"),
    )
    conn.commit()
    conn.close()
    policy, document, checked_at, records = _derived_inputs(
        tmp_path, domain, digests, curation
    )

    report = validate_rebuild_candidate(
        domain,
        infrastructure,
        digests,
        curation,
        replay_report,
        domain,
        document,
        policy,
        checked_at,
        records,
    )

    assert report["valid"] is False
    assert (
        "candidate curation state hash does not match"
        in report["curation_replay"]["errors"]
    )


def test_bad_generation_hash_and_orphan_are_rejected(tmp_path):
    domain, infrastructure, digests, digest, record_id, content_hash = _valid_candidate(
        tmp_path
    )
    conn = sqlite3.connect(domain)
    conn.execute(
        "UPDATE digest_import_receipts SET import_generation = ?, digest_sha256 = ?",
        (CURRENT_IMPORT_GENERATION - 1, "sha256:" + "f" * 64),
    )
    conn.commit()
    conn.close()
    orphan_digest, orphan_record, orphan_hash = _digest(digests, suffix="b")
    _database(
        infrastructure,
        digest=orphan_digest,
        record_id=orphan_record,
        content_hash=orphan_hash,
    )
    orphan_digest.unlink()

    report = _validate(domain, infrastructure, digests)

    assert report["valid"] is False
    domain_checks = report["databases"]["domain"]["receipt_checks"]["counts"]
    assert domain_checks["generation_drift"] == 1
    assert domain_checks["hash_mismatch"] == 1
    assert report["databases"]["domain"]["native_deltas"]["changed"] == 1
    infra_checks = report["databases"]["infrastructure"]["receipt_checks"]["counts"]
    assert infra_checks["orphan"] == 1


def test_record_without_receipt_and_foreign_key_violation_are_rejected(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    conn = sqlite3.connect(infrastructure)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO records (id, title, content_hash, created_at) VALUES (?, ?, ?, ?)",
        ("unreceipted", "Unreceipted", "sha256:" + "c" * 64, "now"),
    )
    conn.execute(
        "INSERT INTO claims "
        "(id, content, claim_type, record_id, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("dangling", "Dangling", "fact", "missing", 1.0, "now"),
    )
    conn.commit()
    conn.close()

    report = _validate(domain, infrastructure, digests)

    infra = report["databases"]["infrastructure"]
    assert report["valid"] is False
    assert infra["foreign_keys"]["violation_count"] == 1
    assert infra["receipt_checks"]["counts"]["records_without_receipt"] == 1


def test_corrupt_database_and_cli_failure_are_reported_without_writes(tmp_path):
    domain, infrastructure, digests, _digest_path, _record_id, _content_hash = (
        _valid_candidate(tmp_path)
    )
    domain.write_bytes(b"not a sqlite database")
    before = (_sha256(domain), _sha256(infrastructure))
    curation, replay_report = _curation_report(infrastructure, digests)
    replay_report["candidate_graph_input_fingerprint"] = "sha256:" + "0" * 64
    report_path = tmp_path / "curation-replay-report.json"
    report_path.write_text(json.dumps(replay_report))
    policy, document, checked_at, _records = _derived_inputs(
        tmp_path, infrastructure, digests, curation
    )
    operation_records = tmp_path / "operation-records.jsonl"

    result = CliRunner().invoke(
        main,
        [
            "validate-rebuild-candidate",
            str(domain),
            str(infrastructure),
            str(digests),
            str(curation),
            str(report_path),
            str(infrastructure),
            str(document),
            str(operation_records),
            "--policy",
            str(policy),
            "--checked-at",
            checked_at,
        ],
    )

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["valid"] is False
    assert report["databases"]["domain"]["integrity"]["ok"] is False
    assert report["databases"]["domain"]["open_mode"] == "ro"
    assert (_sha256(domain), _sha256(infrastructure)) == before


def test_rebuild_refuses_the_production_database_before_deleting_anything(tmp_path):
    source = tmp_path / "source.db"
    _database(source)
    digests = tmp_path / "digests"
    digests.mkdir()

    result = CliRunner().invoke(
        main,
        ["rebuild", str(digests), "--source-db", str(source)],
    )

    assert result.exit_code == 1
    assert "production databases cannot be rebuilt in place" in result.output
    assert source.is_file()


def test_isolated_rebuild_runs_strict_replay_and_writes_its_report(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.db"
    _database(source)
    digests = tmp_path / "digests"
    _digest(digests)
    candidate = tmp_path / "candidate" / "knowledge.db"
    (tmp_path / "ingests").mkdir()
    monkeypatch.setenv("ANOMALICA_INGESTS_DIR", str(tmp_path / "ingests"))

    result = CliRunner().invoke(
        main,
        [
            "--db",
            str(candidate),
            "rebuild",
            str(digests),
            "--source-db",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.output
    replay_report = candidate.parent / "curation-replay-report.json"
    assert replay_report.is_file()
    assert json.loads(replay_report.read_text())["operation_inventory"] == []
    assert source.is_file()
