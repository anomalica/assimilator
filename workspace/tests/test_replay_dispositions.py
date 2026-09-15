from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from anomalica_common.digest.hashing import claim_fingerprint
from assimilator.database import init_db
from assimilator.replay_dispositions import (
    EMPTY_LEDGER_BYTES,
    EMPTY_LEDGER_SHA256,
    FILENAME,
    ID_SCHEMA,
    SCHEMA,
    ReplayDispositionError,
    ReplayDispositionLedgerMissingError,
    disposition_id,
    read_active_source_operations,
    read_ledger,
    validate_replay_dispositions,
)


SOURCE_FP = "sha256:" + "1" * 64
CANDIDATE_FP = "sha256:" + "2" * 64
NOW = "2026-09-15T00:00:00Z"
CLASSIFIER = {"implementation": "assimilator.test", "commit": "a" * 40}
CLAIM = {
    "content": "A supported claim",
    "claim_type": "fact",
    "original_excerpt": "A supported claim",
    "location_in_record": "line 1",
}
CLAIM_FP = claim_fingerprint(**CLAIM)


def _identity(
    name: str = "Alpha",
    prior_names: list[str] | None = None,
    *,
    node_type: str = "person",
) -> dict:
    return {
        "name": name,
        "node_type": node_type,
        "prior_names": list(prior_names or []),
    }


def _resolved(
    identity_name: str = "Alpha",
    *,
    current_name: str | None = None,
    prior_names: list[str] | None = None,
    node_type: str = "person",
) -> dict:
    current_name = identity_name if current_name is None else current_name
    return {
        "identity": _identity(identity_name, prior_names, node_type=node_type),
        "resolved": {"name": current_name, "node_type": node_type},
    }


def _locator(*, present: bool = True) -> dict:
    return {
        "record_content_hash": "sha256:" + ("a" if present else "b") * 64,
        "claim_fingerprint": CLAIM_FP if present else "c" * 64,
    }


def _array(*items: dict) -> list[dict]:
    return sorted(
        items,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, separators=(",", ":")
        ).encode(),
    )


def _entry(
    outcome: str,
    evidence: dict,
    *,
    phase: str = "merge",
    operation_id: str = "operation-1",
    classified_at: str = NOW,
    supersedes: str | None = None,
) -> dict:
    entry = {
        "id": "sha256:" + "0" * 64,
        "source_phase": phase,
        "operation_id": operation_id,
        "source_graph_input_fingerprint": SOURCE_FP,
        "candidate_graph_input_fingerprint": CANDIDATE_FP,
        "outcome": outcome,
        "evidence": evidence,
        "classified_at": classified_at,
        "classifier": dict(CLASSIFIER),
        "supersedes_disposition_id": supersedes,
    }
    entry["id"] = disposition_id(entry)
    return entry


def _canonical_bytes(entries: list[dict]) -> bytes:
    return yaml.safe_dump(
        {"schema": SCHEMA, "entries": entries},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode()


def _write_ledger(root: Path, entries: list[dict]) -> Path:
    path = root / FILENAME
    path.write_bytes(_canonical_bytes(entries))
    return path


def _candidate(
    path: Path,
    *,
    nodes: tuple[tuple[str, str], ...] = (("node-alpha", "Alpha"),),
    aliases: tuple[tuple[str, str], ...] = (("A", "node-alpha"),),
    node_type: str = "person",
) -> None:
    conn = sqlite3.connect(path)
    init_db(conn)
    for node_id, name in nodes:
        conn.execute(
            "INSERT INTO nodes (id,node_type,name,created_at) VALUES (?,?,?,?)",
            (node_id, node_type, name, NOW),
        )
    for alias, node_id in aliases:
        conn.execute(
            "INSERT INTO aliases (alias,node_id) VALUES (?,?)", (alias, node_id)
        )
    conn.execute(
        "INSERT INTO records (id,title,content_hash,created_at) VALUES (?,?,?,?)",
        ("record-a", "Record A", "sha256:" + "a" * 64, NOW),
    )
    conn.execute(
        "INSERT INTO claims "
        "(id,content,claim_type,original_excerpt,location_in_record,record_id,created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            "claim-a",
            CLAIM["content"],
            CLAIM["claim_type"],
            CLAIM["original_excerpt"],
            CLAIM["location_in_record"],
            "record-a",
            NOW,
        ),
    )
    conn.commit()
    conn.close()


def _operations(
    *, phase: str = "merge", include_later: bool = False, linked: bool = True
) -> dict:
    phase_operations = {"operation-1": {"at": "2026-09-14T00:00:00Z"}}
    if include_later:
        phase_operations["operation-2"] = {
            "at": "2026-09-15T00:00:00Z",
            **({"references": "operation-1"} if linked else {}),
        }
    operations = {"merge": {}, "rejection": {}, "rename": {}}
    operations[phase] = phase_operations
    return operations


def _replay_result(
    *outcomes: tuple[str, str, str], blockers: tuple[object, ...] = ()
) -> SimpleNamespace:
    inventory = tuple((phase, operation_id) for phase, operation_id, _ in outcomes)
    diagnostics = tuple(
        SimpleNamespace(
            source_phase=phase,
            operation_id=operation_id,
            outcome=outcome,
            details={},
        )
        for phase, operation_id, outcome in outcomes
    )
    return SimpleNamespace(
        operation_inventory=inventory,
        diagnostics=diagnostics,
        blockers=blockers,
    )


def _merge_proof() -> dict:
    return {
        "resolved_identities": _array(
            _resolved("A", current_name="Alpha"), _resolved()
        ),
        "postcondition": "merged",
    }


def _evidence(outcome: str) -> dict:
    if outcome == "applied":
        return _merge_proof()
    if outcome == "absorbed":
        return {
            "resolved_identities": [_resolved()],
            "absent_identities": [_identity("Absent")],
            "postcondition": "merged",
        }
    if outcome == "superseded":
        return {
            "superseded_by": {"source_phase": "merge", "operation_id": "operation-2"},
            **_merge_proof(),
        }
    if outcome == "compensated":
        return {
            "compensated_by": {"source_phase": "merge", "operation_id": "operation-2"}
        }
    if outcome == "contraction_drop":
        return {
            "absent_identities": [_identity("Absent")],
            "absent_support_locators": [_locator(present=False)],
        }
    if outcome == "unresolved_drift":
        return {
            "unresolved_identities": [_identity("Absent")],
            "surviving_support_locators": [_locator()],
            "reason": "identity drifted",
        }
    if outcome == "unconfirmed":
        return {"reason": "not confirmed"}
    return {"field_paths": ["/victims/0"], "reason": "source operation invalid"}


def test_empty_ledger_exact_bytes_hash_and_contextual_absence(tmp_path):
    virtual = read_ledger(tmp_path, required_basis_count=0)
    assert virtual["exists"] is False
    assert virtual["entries"] == []
    assert virtual["ledger_sha256"] == EMPTY_LEDGER_SHA256
    assert hashlib.sha256(EMPTY_LEDGER_BYTES).hexdigest() == EMPTY_LEDGER_SHA256[7:]

    path = tmp_path / FILENAME
    path.write_bytes(EMPTY_LEDGER_BYTES)
    present = read_ledger(tmp_path, required_basis_count=0)
    assert present["exists"] is True
    assert present["ledger_sha256"] == EMPTY_LEDGER_SHA256
    with pytest.raises(ReplayDispositionError, match="empty only"):
        read_ledger(tmp_path, required_basis_count=1)
    path.unlink()
    with pytest.raises(ReplayDispositionLedgerMissingError):
        read_ledger(tmp_path, required_basis_count=1)


def test_strict_replay_derives_only_exceptional_required_bases(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    operations = {
        "merge": {
            "normal": {"at": NOW},
            "exceptional": {"at": NOW},
        },
        "rejection": {},
        "rename": {},
    }
    replay = _replay_result(
        ("merge", "normal", "applied_normally"),
        ("merge", "exceptional", "absorbed"),
    )

    with pytest.raises(ReplayDispositionLedgerMissingError):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=operations,
            replay_result=replay,
        )

    _write_ledger(
        tmp_path,
        [_entry("absorbed", _evidence("absorbed"), operation_id="exceptional")],
    )
    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=operations,
        replay_result=replay,
    )

    assert report["replacement_safe"] is True
    assert report["missing_dispositions"] == []
    assert [item["operation_id"] for item in report["outcomes"]] == ["exceptional"]


def test_fresh_applied_root_cannot_override_exceptional_replay(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    _write_ledger(tmp_path, [_entry("applied", _merge_proof())])

    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=_operations(),
        replay_result=_replay_result(("merge", "operation-1", "absorbed")),
    )

    assert report["replacement_safe"] is False
    assert report["invalid_closures"][0]["reason"].startswith("fresh applied")


@pytest.mark.parametrize("replay_outcome", ["unconfirmed", "invalid"])
@pytest.mark.parametrize("safe_outcome", ["absorbed", "contraction_drop"])
def test_blocking_replay_cannot_be_bypassed_by_safe_outcome_evidence(
    tmp_path, replay_outcome, safe_outcome
):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    _write_ledger(tmp_path, [_entry(safe_outcome, _evidence(safe_outcome))])

    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=_operations(),
        replay_result=_replay_result(("merge", "operation-1", replay_outcome)),
    )

    assert report["replacement_safe"] is False
    assert report["invalid_closures"] == [
        {
            "source_phase": "merge",
            "operation_id": "operation-1",
            "reason": f"{replay_outcome} replay requires a matching blocking disposition",
        }
    ]


def test_supersession_cannot_use_an_unrelated_later_operation(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    _write_ledger(tmp_path, [_entry("superseded", _evidence("superseded"))])
    operations = {
        "merge": {
            "operation-1": {
                "at": "2026-09-14T00:00:00Z",
                "survivor": _identity("Alpha"),
                "victims": [_identity("A")],
            },
            "operation-2": {
                "at": "2026-09-15T00:00:00Z",
                "survivor": _identity("Unrelated"),
                "victims": [_identity("Other")],
            },
        },
        "rejection": {},
        "rename": {},
    }
    replay = _replay_result(
        ("merge", "operation-1", "absorbed"),
        ("merge", "operation-2", "applied_normally"),
    )

    with pytest.raises(ReplayDispositionError, match="evidenced identities"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=operations,
            replay_result=replay,
        )


def test_strict_replay_blocker_cannot_be_dispositioned(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    blocker = SimpleNamespace(source_phase="merge", cause="missing_stable_id")

    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations={"merge": {}, "rejection": {}, "rename": {}},
        replay_result=_replay_result(blockers=(blocker,)),
    )

    assert report["replacement_safe"] is False
    assert report["errors"] == ["strict replay blocker merge:missing_stable_id"]


def test_fixed_id_vector_and_declared_canonical_shape(tmp_path):
    entry = _entry("unconfirmed", {"reason": "Awaiting reviewer"})
    assert ID_SCHEMA == "anomalica/curation-replay-disposition-id/1"
    assert entry["id"] == (
        "sha256:a3e55081ccb0fb5684015cba98ab9805efdfe3829d05f3d0da09d7c699f268ef"
    )
    assert list(entry) == [
        "id",
        "source_phase",
        "operation_id",
        "source_graph_input_fingerprint",
        "candidate_graph_input_fingerprint",
        "outcome",
        "evidence",
        "classified_at",
        "classifier",
        "supersedes_disposition_id",
    ]
    assert entry["supersedes_disposition_id"] is None
    assert "schema" not in entry
    path = _write_ledger(tmp_path, [entry])
    ledger = read_ledger(tmp_path, required_basis_count=1)
    assert ledger["entries"] == [entry]
    assert (
        ledger["ledger_sha256"]
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("variant", ["comment", "crlf", "reorder", "spacing"])
def test_noncanonical_yaml_bytes_are_rejected(tmp_path, variant):
    entry = _entry("unconfirmed", _evidence("unconfirmed"))
    content = _canonical_bytes([entry])
    if variant == "comment":
        content = b"# comment\n" + content
    elif variant == "crlf":
        content = content.replace(b"\n", b"\r\n")
    elif variant == "reorder":
        reordered = {"entries": [entry], "schema": SCHEMA}
        content = yaml.safe_dump(reordered, sort_keys=False).encode()
    else:
        content = content.replace(b"entries:\n", b"entries: \n", 1)
    (tmp_path / FILENAME).write_bytes(content)
    with pytest.raises(ReplayDispositionError, match="bytes are not canonical"):
        read_ledger(tmp_path, required_basis_count=1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry.update({"schema": SCHEMA}),
        lambda entry: entry.pop("supersedes_disposition_id"),
        lambda entry: entry.update({"classified_at": "2026-09-15T09:00:00+09:00"}),
        lambda entry: entry.update({"classified_at": "2026-09-15T00:00:00"}),
        lambda entry: entry.update({"classifier": "implementation"}),
        lambda entry: entry.update(
            {"classifier": {"implementation": "test", "commit": "A" * 40}}
        ),
        lambda entry: entry["classifier"].update({"extra": "field"}),
    ],
)
def test_corrected_entry_classifier_and_timestamp_shapes_are_strict(tmp_path, mutation):
    entry = _entry("unconfirmed", _evidence("unconfirmed"))
    mutation(entry)
    (tmp_path / FILENAME).write_bytes(_canonical_bytes([entry]))
    with pytest.raises(ReplayDispositionError):
        read_ledger(tmp_path, required_basis_count=1)


def test_noncanonical_and_duplicate_evidence_arrays_are_rejected(tmp_path):
    alpha = _resolved()
    alias = _resolved("A", current_name="Alpha")
    for items in ([alpha, alias], [alias, alias]):
        entry = _entry(
            "applied",
            {"resolved_identities": items, "postcondition": "merged"},
        )
        (tmp_path / FILENAME).write_bytes(_canonical_bytes([entry]))
        with pytest.raises(ReplayDispositionError, match="canonical|duplicates"):
            read_ledger(tmp_path, required_basis_count=1)


def test_natural_identity_prior_names_must_be_nonempty(tmp_path):
    del tmp_path
    evidence = _evidence("contraction_drop")
    evidence["absent_identities"][0]["prior_names"] = [""]

    with pytest.raises(ReplayDispositionError, match="non-empty strings"):
        _entry("contraction_drop", evidence)


@pytest.mark.parametrize(
    ("outcome", "safe"),
    [
        ("applied", True),
        ("absorbed", True),
        ("superseded", True),
        ("compensated", True),
        ("contraction_drop", True),
        ("unresolved_drift", False),
        ("unconfirmed", False),
        ("invalid", False),
    ],
)
def test_every_outcome_validates_and_reports_blocking_status(tmp_path, outcome, safe):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    include_later = outcome in {"superseded", "compensated"}
    entries = [_entry(outcome, _evidence(outcome))]
    if include_later:
        entries.append(
            _entry(
                "applied",
                _merge_proof(),
                operation_id="operation-2",
                classified_at="2026-09-15T01:00:00Z",
            )
        )
    path = _write_ledger(tmp_path, entries)
    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=_operations(include_later=include_later),
    )
    assert "schema" not in report
    assert report["replacement_safe"] is safe
    assert report["candidate_graph_input_fingerprint"] == CANDIDATE_FP
    assert (
        report["disposition_ledger_sha256"]
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert [item["outcome"] for item in report["outcomes"]][0] == outcome
    assert bool(report["errors"]) is (not safe)


@pytest.mark.parametrize(
    ("phase", "resolved", "postcondition"),
    [
        ("merge", _array(_resolved("A", current_name="Alpha"), _resolved()), "merged"),
        ("rejection", _array(_resolved(), _resolved("Beta")), "distinct"),
        ("rename", [_resolved()], "renamed"),
    ],
)
def test_each_phase_postcondition_is_proved(tmp_path, phase, resolved, postcondition):
    candidate = tmp_path / "candidate.db"
    _candidate(
        candidate,
        nodes=(("node-alpha", "Alpha"), ("node-beta", "Beta")),
        aliases=(("A", "node-alpha"),),
    )
    evidence = {"resolved_identities": resolved, "postcondition": postcondition}
    _write_ledger(tmp_path, [_entry("applied", evidence, phase=phase)])
    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=_operations(phase=phase),
    )
    assert report["replacement_safe"] is True


@pytest.mark.parametrize(
    ("phase", "resolved", "message"),
    [
        ("merge", _array(_resolved(), _resolved("Beta")), "one current node"),
        ("rejection", [_resolved()], "two distinct"),
        (
            "rename",
            [_resolved(current_name="Not Alpha")],
            "canonical name/type",
        ),
    ],
)
def test_false_postcondition_proofs_are_rejected(tmp_path, phase, resolved, message):
    candidate = tmp_path / "candidate.db"
    _candidate(
        candidate,
        nodes=(("node-alpha", "Alpha"), ("node-beta", "Beta")),
        aliases=(),
    )
    evidence = {
        "resolved_identities": resolved,
        "postcondition": {
            "merge": "merged",
            "rejection": "distinct",
            "rename": "renamed",
        }[phase],
    }
    _write_ledger(tmp_path, [_entry("applied", evidence, phase=phase)])
    with pytest.raises(ReplayDispositionError, match=message):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(phase=phase),
        )


def test_resolution_falls_back_only_to_unique_non_fuzzy_match(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(
        candidate,
        nodes=(("node-alpha", "Alpha-Beta"),),
        aliases=(),
        node_type="organisation",
    )
    evidence = {
        "resolved_identities": [
            _resolved(
                "Alpha Beta",
                current_name="Alpha-Beta",
                node_type="organisation",
            )
        ],
        "postcondition": "renamed",
    }
    _write_ledger(tmp_path, [_entry("applied", evidence, phase="rename")])
    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=_operations(phase="rename"),
    )
    assert report["replacement_safe"] is True


def test_exact_identity_ambiguity_is_rejected_before_fallback(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(
        candidate,
        nodes=(("node-alpha", "Alpha"), ("node-other", "Other")),
        aliases=(("Alpha", "node-other"),),
    )
    evidence = {"resolved_identities": [_resolved()], "postcondition": "merged"}
    _write_ledger(tmp_path, [_entry("applied", evidence)])
    with pytest.raises(ReplayDispositionError, match="ambiguously"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(),
        )


def test_tamper_and_wrong_fingerprint_fail_closed(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    entry = _entry("unconfirmed", _evidence("unconfirmed"))
    entry["evidence"]["reason"] = "tampered"
    _write_ledger(tmp_path, [entry])
    with pytest.raises(ReplayDispositionError, match="does not match"):
        read_ledger(tmp_path, required_basis_count=1)

    entry = _entry("unconfirmed", _evidence("unconfirmed"))
    _write_ledger(tmp_path, [entry])
    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        "sha256:" + "9" * 64,
        active_source_operations=_operations(),
    )
    assert report["replacement_safe"] is False
    assert report["outcomes"] == []
    assert report["missing_dispositions"] == [
        {"source_phase": "merge", "operation_id": "operation-1"}
    ]


def test_reclassification_chain_rejects_duplicate_fork_and_cross_basis(tmp_path):
    first = _entry("unconfirmed", _evidence("unconfirmed"))
    second = _entry(
        "applied",
        _merge_proof(),
        classified_at="2026-09-15T01:00:00Z",
        supersedes=first["id"],
    )
    _write_ledger(tmp_path, [first, second])
    assert read_ledger(tmp_path, required_basis_count=1)["active_tips"] == [second]

    _write_ledger(tmp_path, [first, first])
    with pytest.raises(ReplayDispositionError):
        read_ledger(tmp_path, required_basis_count=1)

    fork = _entry(
        "invalid",
        _evidence("invalid"),
        classified_at="2026-09-15T02:00:00Z",
        supersedes=first["id"],
    )
    _write_ledger(tmp_path, [first, second, fork])
    with pytest.raises(ReplayDispositionError, match="not current"):
        read_ledger(tmp_path, required_basis_count=1)

    cross = _entry(
        "applied",
        _merge_proof(),
        operation_id="other-operation",
        classified_at="2026-09-15T01:00:00Z",
        supersedes=first["id"],
    )
    _write_ledger(tmp_path, [first, cross])
    with pytest.raises(ReplayDispositionError, match="different basis"):
        read_ledger(tmp_path, required_basis_count=1)


def test_entries_are_stored_in_classification_time_and_id_order(tmp_path):
    later = _entry(
        "unconfirmed",
        _evidence("unconfirmed"),
        operation_id="operation-later",
        classified_at="2026-09-15T01:00:00Z",
    )
    earlier = _entry(
        "unconfirmed",
        _evidence("unconfirmed"),
        operation_id="operation-earlier",
    )
    _write_ledger(tmp_path, [later, earlier])

    with pytest.raises(ReplayDispositionError, match="classified_at and id"):
        read_ledger(tmp_path, required_basis_count=2)


def test_superseded_proof_and_compensation_linkage_are_enforced(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    later = _entry(
        "applied",
        _merge_proof(),
        operation_id="operation-2",
        classified_at="2026-09-15T01:00:00Z",
    )
    bad_proof = _evidence("superseded")
    bad_proof["resolved_identities"] = _array(
        _resolved(), _resolved("Absent", current_name="Absent")
    )
    _write_ledger(tmp_path, [_entry("superseded", bad_proof), later])
    with pytest.raises(ReplayDispositionError, match="resolved identity is absent"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(include_later=True),
        )

    compensated = _entry("compensated", _evidence("compensated"))
    _write_ledger(tmp_path, [compensated, later])
    with pytest.raises(ReplayDispositionError, match="explicitly"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(include_later=True, linked=False),
        )


def test_contraction_and_unresolved_evidence_are_checked_even_when_blocking(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    contraction = _evidence("contraction_drop")
    contraction["absent_support_locators"] = [_locator()]
    _write_ledger(tmp_path, [_entry("contraction_drop", contraction)])
    with pytest.raises(ReplayDispositionError, match="asserted absent"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(),
        )

    unresolved = _evidence("unresolved_drift")
    unresolved["surviving_support_locators"] = [_locator(present=False)]
    _write_ledger(tmp_path, [_entry("unresolved_drift", unresolved)])
    with pytest.raises(ReplayDispositionError, match="exactly once"):
        validate_replay_dispositions(
            tmp_path,
            candidate,
            SOURCE_FP,
            CANDIDATE_FP,
            active_source_operations=_operations(),
        )


def test_source_adapter_uses_actual_phase_reversal_links(tmp_path):
    (tmp_path / "merges.yaml").write_text(
        "---\nop: merge\nmerge_id: merge-active\nat: '2026-09-14T00:00:00Z'\n"
        "---\nop: merge\nmerge_id: merge-undone\nat: '2026-09-14T00:00:00Z'\n"
        "---\nop: undo\nmerge_id: merge-undone\nat: '2026-09-15T00:00:00Z'\n"
    )
    (tmp_path / "rejections.yaml").write_text(
        "---\nop: reject\nrejection_id: rejection-undone\nat: '2026-09-14T00:00:00Z'\n"
        "---\nop: unreject\nrejection_id: rejection-undone\nat: '2026-09-15T00:00:00Z'\n"
    )
    (tmp_path / "renames.yaml").write_text(
        "---\nop: rename\nrename_id: rename-active\nat: '2026-09-15T00:00:00Z'\n"
        "by: null\nnew_name: Beta\nnode:\n  name: Alpha\n  node_type: person\n"
        "  prior_names: []\n"
    )
    operations = read_active_source_operations(tmp_path)
    assert set(operations["merge"]) == {"merge-active", "merge-undone"}
    assert set(operations["rejection"]) == {"rejection-undone"}
    assert len(operations["rename"]) == 1
    assert next(iter(operations["rename"])).startswith("rename-ledger:sha256:")

    (tmp_path / "renames.yaml").write_text(
        "---\nop: unrename\nrename_id: missing\nat: '2026-09-15T00:00:00Z'\n"
    )
    with pytest.raises(ReplayDispositionError, match="does not bind"):
        read_active_source_operations(tmp_path)


def test_proposal_compensation_requires_applied_and_restoring_rename_operations(
    tmp_path,
):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    proposal_id = "rename-proposal:proposal-1"
    applied_id = "rename-ledger:sha256:" + "1" * 64
    compensator_id = "rename-ledger:sha256:" + "2" * 64
    evidence = {
        "applied_by": {"source_phase": "rename", "operation_id": applied_id},
        "compensated_by": {
            "source_phase": "rename",
            "operation_id": compensator_id,
        },
    }
    proposal_entry = _entry(
        "compensated", evidence, phase="rename", operation_id=proposal_id
    )
    assert list(proposal_entry["evidence"]) == ["applied_by", "compensated_by"]
    applied_entry = _entry(
        "compensated",
        {
            "compensated_by": {
                "source_phase": "rename",
                "operation_id": compensator_id,
            }
        },
        phase="rename",
        operation_id=applied_id,
        classified_at="2026-09-15T01:00:00Z",
    )
    compensator_entry = _entry(
        "applied",
        {"resolved_identities": [_resolved()], "postcondition": "renamed"},
        phase="rename",
        operation_id=compensator_id,
        classified_at="2026-09-15T02:00:00Z",
    )
    _write_ledger(tmp_path, [proposal_entry, applied_entry, compensator_entry])
    operations = {
        "merge": {},
        "rejection": {},
        "rename": {
            proposal_id: {
                "at": "2026-09-14T00:00:00Z",
                "node_name_at_proposal": "Alpha",
                "proposed_name": "Beta",
            },
            applied_id: {
                "at": "2026-09-15T00:00:00Z",
                "proposal_id": proposal_id,
                "new_name": "Beta",
                "node": _identity("Alpha"),
            },
            compensator_id: {
                "at": "2026-09-15T01:00:00Z",
                "new_name": "Alpha",
                "node": _identity("Beta"),
                "references": applied_id,
            },
        },
    }

    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=operations,
    )

    assert report["replacement_safe"] is True


def test_generic_compensation_cannot_use_proposal_evidence_shape():
    with pytest.raises(ReplayDispositionError, match="rename proposal"):
        _entry(
            "compensated",
            {
                "applied_by": {
                    "source_phase": "rename",
                    "operation_id": "rename-ledger:sha256:" + "1" * 64,
                },
                "compensated_by": {
                    "source_phase": "rename",
                    "operation_id": "rename-ledger:sha256:" + "2" * 64,
                },
            },
        )


def test_proposal_supersession_may_reference_an_active_merge(tmp_path):
    candidate = tmp_path / "candidate.db"
    _candidate(candidate)
    proposal_id = "rename-proposal:greys"
    merge_id = "merge-greys"
    evidence = {
        "superseded_by": {"source_phase": "merge", "operation_id": merge_id},
        "resolved_identities": _merge_proof()["resolved_identities"],
        "postcondition": "merged",
    }
    proposal_entry = _entry(
        "superseded", evidence, phase="rename", operation_id=proposal_id
    )
    merge_entry = _entry(
        "applied",
        _merge_proof(),
        operation_id=merge_id,
        classified_at="2026-09-15T01:00:00Z",
    )
    _write_ledger(tmp_path, [proposal_entry, merge_entry])
    operations = {
        "merge": {merge_id: {"at": "2026-09-15T00:00:00Z"}},
        "rejection": {},
        "rename": {
            proposal_id: {
                "at": "2026-09-14T00:00:00Z",
                "node_name_at_proposal": "Alpha",
                "proposed_name": "Alpha",
            }
        },
    }

    report = validate_replay_dispositions(
        tmp_path,
        candidate,
        SOURCE_FP,
        CANDIDATE_FP,
        active_source_operations=operations,
    )

    assert report["replacement_safe"] is True
