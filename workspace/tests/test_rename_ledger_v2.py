from __future__ import annotations

import yaml
import pytest

from assimilator.rename_ledger import (
    RenameLedgerError,
    append_event_if_absent,
    build_compensation_event,
    build_reject_proposal_event,
    build_rename_event,
    legacy_operation_identity,
    parse_proposal_document,
    parse_stream,
)


NODE = {"name": "Alpha", "node_type": "event", "prior_names": []}


def test_v2_builders_have_exact_shapes_and_id_vectors():
    rename = build_rename_event(
        at="2026-09-15T09:00:00+09:00",
        by="operator",
        new_name="Beta",
        node=NODE,
        proposal_id=None,
    )
    assert list(rename) == [
        "schema",
        "op",
        "operation_id",
        "at",
        "by",
        "new_name",
        "node",
        "proposal_id",
    ]
    assert rename["operation_id"] == (
        "rename-ledger:sha256:"
        "75ecb610bdec74dc03f59202113f5e390f9c4958732c629214643076e5012682"
    )
    rejection = build_reject_proposal_event(
        proposal_id="rename-proposal:p1",
        at="2026-09-15T00:00:00Z",
        by="operator",
        reason="not appropriate",
    )
    assert list(rejection) == [
        "schema",
        "op",
        "operation_id",
        "proposal_id",
        "at",
        "by",
        "reason",
    ]
    assert rejection["operation_id"] == (
        "rename-ledger:sha256:"
        "c3ecc2a328e91be77ddc4cbfd4a2f165bc6f2fcfbd1994d8c04fabcf656803c1"
    )
    compensation = build_compensation_event(
        at="2026-09-15T01:00:00Z",
        by="operator",
        reason="withdraw",
        reverses=[rename["operation_id"]],
    )
    assert compensation["id"].startswith("rename-compensation:sha256:")
    assert list(compensation) == [
        "schema",
        "op",
        "id",
        "at",
        "by",
        "reason",
        "reverses",
    ]


def test_locked_append_is_idempotent_and_rejects_payload_mismatch(tmp_path):
    path = tmp_path / "renames.yaml"
    event = build_rename_event(
        at="2026-09-15T00:00:00Z",
        by="operator",
        new_name="Beta",
        node=NODE,
        proposal_id=None,
    )
    assert append_event_if_absent(path, event) is True
    before = path.read_bytes()
    assert append_event_if_absent(path, event) is False
    assert path.read_bytes() == before

    mismatch = dict(event)
    mismatch["new_name"] = "Gamma"
    with pytest.raises(RenameLedgerError, match="does not match|mismatch"):
        append_event_if_absent(path, mismatch)


def test_reader_and_writer_reject_empty_documents(tmp_path):
    from assimilator.rename_ledger import read_stream

    path = tmp_path / "renames.yaml"
    path.write_text("---\n")
    with pytest.raises(RenameLedgerError, match="not an object"):
        read_stream(path)
    event = build_rename_event(
        at="2026-09-15T00:00:00Z",
        by="operator",
        new_name="Beta",
        node=NODE,
        proposal_id=None,
    )
    with pytest.raises(RenameLedgerError, match="not an object"):
        append_event_if_absent(path, event)


def test_proposal_link_requires_a_nonempty_namespaced_id():
    with pytest.raises(RenameLedgerError, match="rename-proposal:<id>"):
        build_rename_event(
            at="2026-09-15T00:00:00Z",
            by="operator",
            new_name="Beta",
            node=NODE,
            proposal_id="rename-proposal:",
        )


def test_parser_rejects_duplicate_encoding_but_keeps_distinct_legacy_raw_ids():
    first = {
        "op": "rename",
        "rename_id": "rn-1",
        "at": "2026-09-15T00:00:00Z",
        "by": None,
        "new_name": "Beta",
        "node": NODE,
    }
    second = {**first, "at": "2026-09-15T01:00:00Z", "new_name": "Gamma"}
    stream = parse_stream([first, second])
    assert len({operation.operation_id for operation in stream.operations}) == 2
    assert all(operation.legacy_raw_id == "rn-1" for operation in stream.operations)
    with pytest.raises(RenameLedgerError, match="duplicate_encoding"):
        parse_stream([first, first])


def test_legacy_identity_sorts_and_deduplicates_prior_names():
    identity = legacy_operation_identity(
        {
            "op": "rename",
            "rename_id": "rn-1",
            "at": "2026-09-15T00:00:00Z",
            "by": None,
            "new_name": "Beta",
            "node": {**NODE, "prior_names": ["Zed", "Able", "Zed"]},
        }
    )
    assert identity["payload"]["node"]["prior_names"] == ["Able", "Zed"]


def test_legacy_raw_rename_id_is_a_string_but_not_an_authoritative_identifier():
    identity = legacy_operation_identity(
        {
            "op": "rename",
            "rename_id": "",
            "at": "2026-09-15T00:00:00Z",
            "by": None,
            "new_name": "Beta",
            "node": NODE,
        }
    )

    assert identity["payload"]["rename_id"] == ""
    assert identity["operation_id"].startswith("rename-ledger:sha256:")


def test_serialised_event_round_trips_without_key_reordering(tmp_path):
    event = build_reject_proposal_event(
        proposal_id="rename-proposal:p1",
        at="2026-09-15T00:00:00Z",
        by="operator",
        reason="reviewed rejection",
    )
    path = tmp_path / "renames.yaml"
    append_event_if_absent(path, event)
    loaded = next(yaml.safe_load_all(path.read_bytes()))
    assert list(loaded) == list(event)


def test_legacy_and_v2_proposals_have_exact_accepted_shapes():
    legacy = parse_proposal_document(
        {
            "id": "p1",
            "node_id": "audit",
            "node_name_at_proposal": "Alpha",
            "proposed_name": "Beta",
            "reason": None,
            "proposed_by": None,
            "proposed_at": "2026-09-15T09:00:00+09:00",
        },
        "legacy.json",
    )
    assert legacy.operation_id == "rename-proposal:p1"
    assert legacy.at == "2026-09-15T00:00:00Z"

    v2 = parse_proposal_document(
        {
            "schema": "anomalica/rename-proposal/2",
            "id": "p2",
            "node": NODE,
            "proposed_name": "Beta",
            "reason": None,
            "proposed_by": "operator",
            "proposed_at": "2026-09-15T00:00:00Z",
            "source": {"node_id": "audit"},
        },
        "v2.json",
    )
    assert v2.operation_id == "rename-proposal:p2"
    with pytest.raises(RenameLedgerError, match="exact seven"):
        parse_proposal_document(
            {
                "id": "bad",
                "node_id": "audit",
                "node_name_at_proposal": "Alpha",
                "proposed_name": "Beta",
                "proposed_at": "2026-09-15T00:00:00Z",
            },
            "bad.json",
        )
