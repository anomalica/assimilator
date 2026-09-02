"""A node minted under a name another type already carries is queued, not lost."""

import json
import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator.database import init_db, insert_claim, insert_node, insert_record
from assimilator.import_markdown import import_extraction


def _digest(name, node_type):
    return {
        "frontmatter": {
            "record_id": "rec-hearing",
            "record_title": "Grusch hearing",
            "record_date": "2023-07-26",
            "content_hash": "sha256:" + "b" * 64,
            "friendly_name": "hearing",
        },
        "nodes": [{"id": "n-aaro", "name": name, "node_type": node_type}],
        "domain_claims": [
            {
                "id": "c1",
                "content": "The office's budget remains classified.",
                "claim_type": "testimony",
                "attestation": "first_hand",
                "speaker": None,
                "node_references": [name],
            }
        ],
        "infrastructure_claims": [],
        "terminology": None,
    }


def test_a_cross_type_name_twin_is_queued_for_review_and_never_merged(
    tmp_path, monkeypatch
):
    queue = tmp_path / "manual.json"
    monkeypatch.setenv("ANOMALICA_MERGE_CANDIDATES_MANUAL", str(queue))
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r0", title="R0", content_hash="sha256:" + "a" * 64))
    org = insert_node(
        conn,
        Node(
            id="org-1",
            name="All-domain Anomaly Resolution Office (AARO)",
            node_type=NodeType.organisation,
        ),
    )
    for i in range(3):
        insert_claim(
            conn,
            Claim(
                id=f"o{i}",
                content=f"org claim {i}",
                claim_type="testimony",
                record_id="r0",
                node_references=[org.id],
            ),
        )
    conn.commit()

    counts = import_extraction(
        conn, _digest("All-Domain Anomaly Resolution Office (AARO)", "project")
    )

    live = conn.execute(
        "SELECT id, node_type FROM nodes WHERE retired_at IS NULL ORDER BY node_type"
    ).fetchall()
    assert len(live) == 2  # minted, not merged
    assert counts["cross_type_twins"] == 1
    entries = json.loads(queue.read_text())
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry["node_ids"]) == {"org-1", "n-aaro"}
    assert (
        entry["suggested_canonical"] == "org-1"
    )  # the established node, not the newcomer
    assert entry["node_type"] == "organisation"
    assert entry["reason"].startswith("import:")

    # A re-import matches the project node it minted and queues nothing new.
    counts = import_extraction(
        conn, _digest("All-Domain Anomaly Resolution Office (AARO)", "project")
    )
    assert counts["cross_type_twins"] == 0
    assert len(json.loads(queue.read_text())) == 1


def test_a_same_type_name_match_is_the_matchers_job_not_the_queues(
    tmp_path, monkeypatch
):
    queue = tmp_path / "manual.json"
    monkeypatch.setenv("ANOMALICA_MERGE_CANDIDATES_MANUAL", str(queue))
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn, Node(id="org-1", name="Galileo Project", node_type=NodeType.organisation)
    )
    conn.commit()

    counts = import_extraction(conn, _digest("Galileo Project", "organisation"))

    assert counts["nodes_created"] == 0 and counts["cross_type_twins"] == 0
    assert not queue.exists()
