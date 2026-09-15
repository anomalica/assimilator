from __future__ import annotations

import sqlite3
import json

import pytest
import yaml

from anomalica_common.digest.models import Node
from assimilator import merge
from assimilator.database import init_db, insert_node


CONFIRMATION = {
    "by": "test",
    "at": "2026-09-15T00:00:00Z",
    "via": "workbench-queue",
}


def _identity(name: str, node_type: str = "event") -> dict:
    return {"name": name, "node_type": node_type, "prior_names": []}


def _graph(*nodes: tuple[str, str, str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for node_id, node_type, name in nodes:
        insert_node(conn, Node(id=node_id, node_type=node_type, name=name))
    conn.commit()
    return conn


def _write_stream(path, entries: list[dict]) -> None:
    path.write_text(
        "".join("---\n" + yaml.safe_dump(entry, sort_keys=False) for entry in entries)
    )


def _merge_entry(operation_id: str = "merge-1", **changes) -> dict:
    entry = {
        "op": "merge",
        "merge_id": operation_id,
        "at": "2026-09-15T01:00:00Z",
        "confirmation": CONFIRMATION,
        "canonical_name": "Alpha",
        "survivor": _identity("Alpha"),
        "victims": [_identity("Alpha duplicate")],
    }
    entry.update(changes)
    return entry


def _legacy_rename(raw_id="rn-1", old="Alpha", new="Alpha final", **changes):
    entry = {
        "op": "rename",
        "rename_id": raw_id,
        "at": "2026-09-15T01:00:00Z",
        "by": None,
        "new_name": new,
        "node": _identity(old),
    }
    entry.update(changes)
    return entry


def _v2_rename(old="Alpha", new="Beta", proposal_id=None, **changes):
    from assimilator.rename_ledger import build_rename_event

    if proposal_id is not None and not proposal_id.startswith("rename-proposal:"):
        proposal_id = f"rename-proposal:{proposal_id}"
    entry = build_rename_event(
        at="2026-09-15T01:00:00Z",
        by="test/operator",
        new_name=new,
        node=_identity(old),
        proposal_id=proposal_id,
    )
    entry.update(changes)
    return entry


def _compensation(operation_id, at="2026-09-15T02:00:00Z", **changes):
    from assimilator.rename_ledger import build_compensation_event

    return build_compensation_event(
        at=at,
        by=changes.get("by", "test/operator"),
        reason=changes.get("reason", "restore original name"),
        reverses=[operation_id],
    )


def _proposal(proposal_id="proposal-1", old="Alpha", new="Beta", **changes):
    proposal = {
        "id": proposal_id,
        "node_id": "audit-only",
        "node_name_at_proposal": old,
        "proposed_name": new,
        "proposed_at": "2026-09-15T00:00:00Z",
        "reason": None,
        "proposed_by": None,
    }
    proposal.update(changes)
    return proposal


def _write_proposal(root, filename, proposal):
    directory = root / "rename-proposals"
    directory.mkdir(exist_ok=True)
    (directory / filename).write_text(json.dumps(proposal))


def test_strict_replay_covers_exact_inventory_in_fixed_phase_order(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_stream(tmp_path / "merges.yaml", [_merge_entry()])
    _write_stream(
        tmp_path / "rejections.yaml",
        [
            {
                "op": "reject",
                "rejection_id": "rejection-1",
                "at": "2026-09-15T00:00:00Z",
                "nodes": [_identity("Alpha"), _identity("Beta")],
            }
        ],
    )
    _write_stream(
        tmp_path / "renames.yaml",
        [
            {
                "op": "rename",
                "rename_id": "rename-1",
                "at": "2026-09-14T00:00:00Z",
                "node": _identity("Alpha"),
                "new_name": "Alpha final",
            }
        ],
    )
    conn = _graph(
        ("alpha", "event", "Alpha"),
        ("duplicate", "event", "Alpha duplicate"),
        ("beta", "event", "Beta"),
    )

    result = merge.replay_candidate_curation(conn)

    rename_operation_id = merge.legacy_rename_operation_identity(
        {
            "op": "rename",
            "rename_id": "rename-1",
            "at": "2026-09-14T00:00:00Z",
            "by": None,
            "node": _identity("Alpha"),
            "new_name": "Alpha final",
        }
    )["operation_id"]
    assert result.operation_inventory == (
        ("merge", "merge-1"),
        ("rejection", "rejection-1"),
        ("rename", rename_operation_id),
    )
    assert [diagnostic.outcome for diagnostic in result.diagnostics] == [
        "applied_normally",
        "applied_normally",
        "applied_normally",
    ]
    assert result.replacement_safe is True
    assert conn.execute("SELECT name FROM nodes WHERE id='alpha'").fetchone()[0] == (
        "Alpha final"
    )


@pytest.mark.parametrize(
    "entries, message",
    [
        (
            [_merge_entry(), _merge_entry()],
            "duplicate merge replay ID",
        ),
        (
            [
                {
                    "op": "undo",
                    "merge_id": "unknown",
                    "at": "2026-09-15T02:00:00Z",
                }
            ],
            "unknown replay ID",
        ),
    ],
)
def test_strict_replay_fails_on_duplicate_or_unknown_ids(
    tmp_path, monkeypatch, entries, message
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_stream(tmp_path / "merges.yaml", entries)

    with pytest.raises(merge.CandidateReplaySourceError, match=message):
        merge.replay_candidate_curation(_graph())


def test_stable_unconfirmed_and_invalid_operations_have_specific_diagnostics(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    unconfirmed = _merge_entry("merge-unconfirmed", confirmation=None)
    invalid = _merge_entry("merge-invalid")
    invalid.pop("victims")
    _write_stream(tmp_path / "merges.yaml", [unconfirmed, invalid])

    result = merge.replay_candidate_curation(_graph())

    assert [diagnostic.outcome for diagnostic in result.diagnostics] == [
        "invalid",
        "unconfirmed",
    ]
    assert result.diagnostics[0].details["field_paths"] == ["/victims"]
    assert result.diagnostics[1].cause == "confirmation_required"
    assert result.replacement_safe is False


def test_malformed_entry_without_stable_id_is_a_separate_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    malformed = _merge_entry()
    malformed.pop("merge_id")
    _write_stream(tmp_path / "merges.yaml", [malformed])

    result = merge.replay_candidate_curation(_graph())

    assert result.operation_inventory == ()
    assert result.diagnostics == ()
    assert len(result.blockers) == 1
    assert result.blockers[0].cause == "missing_stable_operation_id"
    assert result.replacement_safe is False


def test_strict_replay_refuses_a_read_only_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path / "curation"))
    candidate = tmp_path / "candidate.db"
    writable = sqlite3.connect(candidate)
    init_db(writable)
    writable.close()
    read_only = sqlite3.connect(f"{candidate.as_uri()}?mode=ro", uri=True)

    try:
        with pytest.raises(merge.CandidateReplaySourceError, match="writable"):
            merge.replay_candidate_curation(read_only)
    finally:
        read_only.close()


def test_explicit_later_reversal_is_normal_compensation(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_stream(
        tmp_path / "merges.yaml",
        [
            _merge_entry(),
            {
                "op": "undo",
                "merge_id": "merge-1",
                "at": "2026-09-15T02:00:00Z",
                "by": "test",
            },
        ],
    )
    conn = _graph(
        ("alpha", "event", "Alpha"),
        ("duplicate", "event", "Alpha duplicate"),
    )

    result = merge.replay_candidate_curation(conn)

    diagnostic = result.diagnostics[0]
    assert diagnostic.outcome == "compensated_normally"
    assert diagnostic.details["base_operation"]["operation_id"] == "merge-1"
    assert diagnostic.details["reversal_event"]["operation"] == "undo"
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='duplicate'").fetchone()[0]
        is None
    )


def test_absorbed_and_unresolved_drift_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_stream(
        tmp_path / "rejections.yaml",
        [
            {
                "op": "reject",
                "rejection_id": "absorbed",
                "at": "2026-09-15T00:00:00Z",
                "nodes": [
                    _identity("Alpha"),
                    _identity("A", "event") | {"prior_names": ["Alpha"]},
                ],
            }
        ],
    )
    _write_stream(
        tmp_path / "renames.yaml",
        [
            {
                "op": "rename",
                "rename_id": "drifted",
                "at": "2026-09-15T00:00:00Z",
                "node": _identity("Absent"),
                "new_name": "Still absent",
            }
        ],
    )

    result = merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))

    assert [diagnostic.outcome for diagnostic in result.diagnostics] == [
        "absorbed",
        "unresolved_drift",
    ]
    assert result.diagnostics[0].cause == "single_resolved_node"
    assert result.diagnostics[1].cause == "natural_identity_did_not_resolve"


@pytest.mark.parametrize("phase", ["merge", "rejection"])
def test_three_identity_operation_with_two_resolved_requires_evidence(
    tmp_path, monkeypatch, phase
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    identities = [_identity("Alpha"), _identity("Beta"), _identity("Missing")]
    if phase == "merge":
        _write_stream(
            tmp_path / "merges.yaml",
            [
                _merge_entry(
                    survivor=identities[0],
                    victims=identities[1:],
                )
            ],
        )
    else:
        _write_stream(
            tmp_path / "rejections.yaml",
            [
                {
                    "op": "reject",
                    "rejection_id": "rejection-1",
                    "at": "2026-09-15T01:00:00Z",
                    "nodes": identities,
                }
            ],
        )
    conn = _graph(("alpha", "event", "Alpha"), ("beta", "event", "Beta"))

    result = merge.replay_candidate_curation(conn)

    diagnostic = result.diagnostics[0]
    assert diagnostic.outcome == "unresolved_drift"
    assert diagnostic.cause == "partial_natural_identity_resolution"
    assert diagnostic.details["unresolved_identities"] == [_identity("Missing")]
    assert result.replacement_safe is False
    if phase == "merge":
        assert conn.execute(
            "SELECT retired_at IS NOT NULL FROM nodes WHERE id='beta'"
        ).fetchone() == (1,)
    else:
        assert conn.execute(
            "SELECT node_id FROM node_rejections "
            "WHERE rejection_id='rejection-1' ORDER BY node_id"
        ).fetchall() == [("alpha",), ("beta",)]


@pytest.mark.parametrize("phase", ["merge", "rejection"])
def test_legacy_replay_keeps_partial_identity_operation_applied(
    tmp_path, monkeypatch, phase
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    identities = [_identity("Alpha"), _identity("Beta"), _identity("Missing")]
    conn = _graph(("alpha", "event", "Alpha"), ("beta", "event", "Beta"))
    if phase == "merge":
        _write_stream(
            tmp_path / "merges.yaml",
            [_merge_entry(survivor=identities[0], victims=identities[1:])],
        )
        summary = merge.replay_ledger(conn)
    else:
        _write_stream(
            tmp_path / "rejections.yaml",
            [
                {
                    "op": "reject",
                    "rejection_id": "rejection-1",
                    "at": "2026-09-15T01:00:00Z",
                    "nodes": identities,
                }
            ],
        )
        summary = merge.replay_rejections(conn)

    assert summary["applied"] == 1
    assert summary["lost"] == 0


def test_operations_sort_by_timestamp_then_stable_id_within_each_phase(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    first = _merge_entry(
        "merge-z",
        canonical_name="Zulu",
        survivor=_identity("Zulu"),
        victims=[_identity("Zulu duplicate")],
    )
    second = _merge_entry(
        "merge-a",
        canonical_name="Alpha",
        survivor=_identity("Alpha"),
        victims=[_identity("Alpha duplicate")],
    )
    _write_stream(tmp_path / "merges.yaml", [first, second])

    result = merge.replay_candidate_curation(
        _graph(
            ("alpha", "event", "Alpha"),
            ("alpha-duplicate", "event", "Alpha duplicate"),
            ("zulu", "event", "Zulu"),
            ("zulu-duplicate", "event", "Zulu duplicate"),
        )
    )

    assert [diagnostic.operation_id for diagnostic in result.diagnostics] == [
        "merge-a",
        "merge-z",
    ]


def test_rename_content_id_payloads_have_fixed_vectors():
    legacy = merge.legacy_rename_operation_identity(
        {
            "op": "rename",
            "rename_id": "rn-1",
            "at": "2026-09-15T09:00:00+09:00",
            "by": "curator",
            "new_name": "Grey",
            "node": _identity("Gray", "person") | {"prior_names": ["G. Ray"]},
        }
    )
    assert list(legacy["payload"]) == [
        "schema",
        "op",
        "rename_id",
        "at",
        "by",
        "new_name",
        "node",
    ]
    assert legacy["payload"]["at"] == "2026-09-15T00:00:00Z"
    assert legacy["operation_id"] == (
        "rename-ledger:sha256:"
        "de8c55f931da4c1c66c82432c1142e5e914cb0fd932595c9b9b6f45d5f30b8b4"
    )

    base = _v2_rename("Gray", "Grey", "proposal-1", node=_identity("Gray", "person"))
    assert base["operation_id"] == (
        "rename-ledger:sha256:"
        "344bc276f01ff181833f7056353cbe4b9875361e12b15e5f1fed517982fc4827"
    )
    compensation = _compensation(base["operation_id"])
    assert compensation["id"] == (
        "rename-compensation:sha256:4d271374a71cbafa6cb1b430ebd8c7eedb9529dce8c17807335cb15aff34bfba"
    )


def test_same_legacy_raw_id_keeps_distinct_content_operations(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_stream(
        tmp_path / "renames.yaml",
        [
            _legacy_rename("rn-1", "Alpha", "Alpha final"),
            _legacy_rename("rn-1", "Beta", "Beta final", at="2026-09-15T02:00:00Z"),
        ],
    )

    result = merge.replay_candidate_curation(
        _graph(("alpha", "event", "Alpha"), ("beta", "event", "Beta"))
    )

    rename_ids = [item[1] for item in result.operation_inventory]
    assert len(rename_ids) == len(set(rename_ids)) == 2
    assert all(item.startswith(merge.RENAME_OPERATION_ID_PREFIX) for item in rename_ids)


def test_133_same_raw_ids_derive_133_distinct_content_ids():
    operation_ids = {
        merge.legacy_rename_operation_identity(
            _legacy_rename("rn-1", f"Source {index}", f"Target {index}")
        )["operation_id"]
        for index in range(133)
    }

    assert len(operation_ids) == 133


def test_repeated_legacy_payload_is_a_duplicate_operation_id(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    entry = _legacy_rename()
    _write_stream(tmp_path / "renames.yaml", [entry, entry])

    with pytest.raises(merge.CandidateReplaySourceError, match="duplicate_encoding"):
        merge.replay_candidate_curation(_graph())


@pytest.mark.parametrize(
    ("matching_bases", "blocked"), [(0, True), (1, False), (2, True)]
)
def test_legacy_unrename_requires_exactly_one_raw_id_match(
    tmp_path, monkeypatch, matching_bases, blocked
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    entries = [
        _legacy_rename(
            "rn-1",
            "Alpha" if index == 0 else "Beta",
            "Alpha final" if index == 0 else "Beta final",
        )
        for index in range(matching_bases)
    ]
    entries.append(
        {"op": "unrename", "rename_id": "rn-1", "at": "2026-09-15T03:00:00Z"}
    )
    _write_stream(tmp_path / "renames.yaml", entries)

    graph = _graph(("alpha", "event", "Alpha"), ("beta", "event", "Beta"))
    if blocked:
        with pytest.raises(
            merge.CandidateReplaySourceError, match="does not bind exactly one"
        ):
            merge.replay_candidate_curation(graph)
    else:
        result = merge.replay_candidate_curation(graph)
        assert result.diagnostics[0].outcome == "compensated_normally"


def test_v2_compensation_is_outside_inventory_and_compensates_once(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    base = _v2_rename()
    compensation = _compensation(base["operation_id"])
    _write_stream(tmp_path / "renames.yaml", [base, compensation])

    result = merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))

    assert result.operation_inventory == (("rename", base["operation_id"]),)
    assert result.diagnostics[0].outcome == "compensated_normally"
    assert (
        result.diagnostics[0].details["reversal_event"]["event_id"]
        == (compensation["id"])
    )


def test_v2_base_operation_id_is_verified_from_semantic_content(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    base = _v2_rename()
    base["new_name"] = "Tampered"
    _write_stream(tmp_path / "renames.yaml", [base])

    with pytest.raises(merge.CandidateReplaySourceError, match="does not match"):
        merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))


def test_duplicate_v2_compensation_event_id_is_a_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    base = _v2_rename()
    compensation = _compensation(base["operation_id"])
    _write_stream(tmp_path / "renames.yaml", [base, compensation, compensation])

    with pytest.raises(merge.CandidateReplaySourceError, match="duplicate_encoding"):
        merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))


@pytest.mark.parametrize(
    "events, cause",
    [
        (
            lambda base: [
                _compensation("rename-ledger:sha256:" + "0" * 64),
                base,
            ],
            "compensation_references_unknown_operation",
        ),
        (
            lambda base: [
                base,
                _compensation(base["operation_id"], at="2026-09-14T00:00:00Z"),
            ],
            "compensation_does_not_follow_base",
        ),
        (
            lambda base: [
                base,
                _compensation(base["operation_id"]),
                _compensation(
                    base["operation_id"],
                    at="2026-09-15T03:00:00Z",
                    reason="second reversal",
                ),
            ],
            "operation_already_compensated",
        ),
    ],
)
def test_invalid_v2_compensation_links_block_replay(
    tmp_path, monkeypatch, events, cause
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    base = _v2_rename()
    _write_stream(tmp_path / "renames.yaml", events(base))

    with pytest.raises(merge.CandidateReplaySourceError):
        merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))


@pytest.mark.parametrize(
    ("nodes", "old", "new", "outcome", "cause"),
    [
        (
            (("alpha", "event", "Alpha"),),
            "Alpha",
            "Beta",
            "pending_normally",
            "proposal_has_no_explicit_resolution",
        ),
        (
            (("alpha", "event", "Alpha"), ("beta", "event", "Beta")),
            "Alpha",
            "Beta",
            "pending_normally",
            "proposal_has_no_explicit_resolution",
        ),
        (
            (),
            "Alpha",
            "Beta",
            "unresolved_drift",
            "proposal_source_absent",
        ),
        (
            (("alpha-event", "event", "Alpha"), ("alpha-person", "person", "Alpha")),
            "Alpha",
            "Beta",
            "invalid",
            "proposal_source_ambiguous",
        ),
    ],
)
def test_proposal_normal_and_exceptional_resolution(
    tmp_path, monkeypatch, nodes, old, new, outcome, cause
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_proposal(tmp_path, "proposal.json", _proposal(old=old, new=new))

    result = merge.replay_candidate_curation(_graph(*nodes))

    diagnostic = result.diagnostics[0]
    assert diagnostic.operation_id == "rename-proposal:proposal-1"
    assert (diagnostic.outcome, diagnostic.cause) == (outcome, cause)


def test_future_linked_rename_is_the_only_normal_applied_proposal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_proposal(tmp_path, "proposal.json", _proposal())
    base = _v2_rename(proposal_id="proposal-1")
    _write_stream(tmp_path / "renames.yaml", [base])

    result = merge.replay_candidate_curation(_graph(("alpha", "event", "Alpha")))

    proposal = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.operation_id == "rename-proposal:proposal-1"
    )
    assert proposal.outcome == "applied_normally"
    assert proposal.details["resolution_operation_id"] == base["operation_id"]


def test_proposal_merge_resolution_compensation_and_malformed_id_are_exceptional(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_proposal(
        tmp_path, "merged.json", _proposal("merged", old="Gamma", new="Delta")
    )
    _write_proposal(tmp_path, "compensated.json", _proposal("compensated"))
    _write_proposal(tmp_path, "malformed.json", {"id": "malformed"})
    base = _v2_rename(proposal_id="compensated")
    _write_stream(
        tmp_path / "renames.yaml", [base, _compensation(base["operation_id"])]
    )
    conn = _graph(("alpha", "event", "Alpha"), ("gamma", "event", "Gamma"))
    conn.execute("INSERT INTO aliases (alias,node_id) VALUES ('Delta','gamma')")
    conn.commit()

    result = merge.replay_candidate_curation(conn)
    by_id = {diagnostic.operation_id: diagnostic for diagnostic in result.diagnostics}

    assert (
        by_id["rename-proposal:merged"].cause == "proposal_has_no_explicit_resolution"
    )
    assert (
        by_id["rename-proposal:compensated"].cause == "proposal_resolution_compensated"
    )
    assert by_id["rename-proposal:compensated"].outcome == "compensated_normally"
    assert by_id["rename-proposal:malformed"].outcome == "invalid"


def test_reverse_proposal_is_a_separate_base_request(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    _write_proposal(tmp_path, "forward.json", _proposal("forward"))
    _write_proposal(
        tmp_path,
        "reverse.json",
        _proposal("reverse", old="Beta", new="Alpha"),
    )

    result = merge.replay_candidate_curation(
        _graph(("alpha", "event", "Alpha"), ("beta", "event", "Beta"))
    )

    assert set(result.operation_inventory) == {
        ("rename", "rename-proposal:forward"),
        ("rename", "rename-proposal:reverse"),
    }
