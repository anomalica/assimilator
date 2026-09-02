"""Per-page drift, as one timestamped snapshot rather than 752 decaying copies."""

from __future__ import annotations

import json
import sqlite3

import yaml
from anomalica_common.digest.models import Claim, Node, Record

from assimilator.brief_staleness import staleness_manifest, write_manifest
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    node = insert_node(conn, Node(id="n1", node_type="person", name="Someone"))
    for cid in ("c1", "c2"):
        insert_claim(
            conn,
            Claim(
                id=cid,
                content=cid,
                claim_type="testimony",
                record_id="r1",
                node_references=[node.id],
            ),
        )
    # insert_claim does not compute claim_hash; the importer stamps it separately.
    # Without this the fixture's claims are all unhashed and the tests would be
    # exercising the sentinel path while appearing to exercise the normal one.
    for cid in ("c1", "c2"):
        conn.execute("UPDATE claims SET claim_hash = ? WHERE id = ?", (f"h-{cid}", cid))
    conn.commit()
    return conn


def _brief(tmp_path, node_id, claims):
    d = tmp_path / "briefs"
    (d / "people").mkdir(parents=True, exist_ok=True)
    (d / "people" / "someone.yaml").write_text(
        yaml.safe_dump(
            {
                "page": {"node_id": node_id, "slug": "someone", "title": "Someone"},
                "brief_hash": "bh",
                "claims": [{"claim_id": k, "claim_hash": v} for k, v in claims.items()],
            }
        )
    )
    return d


def test_a_page_built_from_the_current_claims_reads_as_current(tmp_path):
    conn = _graph()
    live = dict(conn.execute("SELECT id, claim_hash FROM claims"))
    manifest = staleness_manifest(conn, _brief(tmp_path, "n1", live))

    assert manifest["pages"]["people/someone"]["pct"] == 0.0
    assert manifest["pages"]["people/someone"]["node_state"] == "live"


def test_drift_is_reported_when_the_graph_has_moved(tmp_path):
    conn = _graph()
    manifest = staleness_manifest(conn, _brief(tmp_path, "n1", {"gone-id": "oldhash"}))

    page = manifest["pages"]["people/someone"]
    assert page["pct"] == 100.0
    assert page["gone"] == 1
    assert page["added"] == 2
    assert page["node_state"] == "live", "the node is fine; its claims moved"


def test_a_merged_away_node_is_superseded_not_stale(tmp_path):
    """Reported identically, a merge victim reads as "this page has lost all its
    evidence" - alarming and false. Its material is on the survivor's page. On the
    first real run 47 pages showed 100%, of which 4 were merge victims."""
    conn = _graph()
    conn.execute("UPDATE nodes SET retired_at = 'T' WHERE id = 'n1'")
    conn.commit()

    manifest = staleness_manifest(conn, _brief(tmp_path, "n1", {"x": "y"}))

    assert manifest["pages"]["people/someone"]["node_state"] == "retired"


def test_a_brief_with_no_node_is_skipped_rather_than_scored(tmp_path):
    """A record brief has no node. Reporting it as 100% drifted would be a
    fabrication rather than a measurement."""
    conn = _graph()
    d = tmp_path / "briefs"
    (d / "records").mkdir(parents=True)
    (d / "records" / "rec.yaml").write_text(
        yaml.safe_dump({"page": {"slug": "rec", "kind": "record"}, "claims": []})
    )

    manifest = staleness_manifest(conn, d)

    assert manifest["pages"] == {}
    assert manifest["not_measurable"] == 1


def test_the_manifest_carries_when_it_was_taken(tmp_path):
    """The figure decays from the moment it is written, so the snapshot must say
    when - otherwise it is an undated assertion that silently goes wrong."""
    conn = _graph()
    manifest = staleness_manifest(conn, _brief(tmp_path, "n1", {"x": "y"}))
    out = write_manifest(manifest, tmp_path / "out" / "staleness.json")

    written = json.loads(out.read_text())
    assert written["generated_at"].endswith("+00:00")
    assert written["schema"] == "anomalica/brief-staleness/1"


def test_an_unhashed_claim_counts_rather_than_disappearing(tmp_path):
    """Dropping a claim with no claim_hash under-reports drift: the page reads as
    current partly because some of its evidence could not be checked. That is the
    unsafe direction for a freshness figure."""
    conn = _graph()
    conn.execute("UPDATE claims SET claim_hash = NULL WHERE id = 'c1'")
    conn.commit()
    live = dict(conn.execute("SELECT id, claim_hash FROM claims"))
    frozen = {k: v for k, v in live.items() if v}

    manifest = staleness_manifest(conn, _brief(tmp_path, "n1", frozen))

    page = manifest["pages"]["people/someone"]
    assert page["current_total"] == 2, "the unhashed claim is still in the graph"
    assert page["added"] == 1, "and shows as material the brief has not seen"


def test_a_stale_staleness_manifest_refuses_to_answer(tmp_path):
    """The purest form of the failure this project keeps hitting: it answers, it
    answers plausibly, and the answer describes a graph that no longer exists. The
    first time this manifest was consumed it was three days old and the consumer
    could not have known."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    import pytest as _pytest

    from assimilator.brief_staleness import StaleManifest, read_manifest

    p = tmp_path / "m.json"
    old = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    p.write_text(_json.dumps({"generated_at": old, "pages": {}}))

    with _pytest.raises(StaleManifest):
        read_manifest(p)


def test_a_manifest_with_no_timestamp_is_refused_outright(tmp_path):
    """Undated is worse than old: nothing can even judge it."""
    import json as _json

    import pytest as _pytest

    from assimilator.brief_staleness import read_manifest

    p = tmp_path / "m.json"
    p.write_text(_json.dumps({"pages": {}}))

    with _pytest.raises(ValueError):
        read_manifest(p)


def test_a_fresh_manifest_is_returned(tmp_path):
    import json as _json
    from datetime import datetime, timezone

    from assimilator.brief_staleness import read_manifest

    p = tmp_path / "m.json"
    p.write_text(
        _json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "pages": {"a": {}}}
        )
    )

    assert read_manifest(p)["pages"] == {"a": {}}
