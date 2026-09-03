"""Replay must follow a curation merge whose survivor has been reclassified.

Every curation merge in this corpus is cross-type - the same name under two
types, where the curator chose which type to keep. A rebuild re-imports whatever
type the digester emits now, so the chosen survivor is routinely absent while its
victims are present. Requiring the survivor to resolve discarded the whole op and
left the duplicates standing: 15 of 46 ops, silently.
"""

from __future__ import annotations

import sqlite3
import textwrap

from anomalica_common.digest.models import Claim, Node, Record
from assimilator import merge
from assimilator.database import init_db, insert_claim, insert_node, insert_record

_CONFIRMED = {"by": "test", "at": "2026-09-03T03:00:00Z", "via": "workbench-queue"}


def _ledger(tmp_path, monkeypatch, body: str) -> None:
    monkeypatch.setenv("ANOMALICA_CURATION_DIR", str(tmp_path))
    (tmp_path / "merges.yaml").write_text(textwrap.dedent(body))


def _graph(*nodes: tuple[str, str, str, int]) -> sqlite3.Connection:
    """nodes: (id, node_type, name, claim_count)."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    for nid, ntype, name, claim_count in nodes:
        insert_node(conn, Node(id=nid, node_type=ntype, name=name))
        for i in range(claim_count):
            insert_claim(
                conn,
                Claim(
                    id=f"c-{nid}-{i}",
                    content=f"claim {i} about {name}",
                    claim_type="observation",
                    record_id="r1",
                    node_references=[nid],
                ),
            )
    conn.commit()
    return conn


UAPTF = """\
    op: merge
    merge_id: m-uaptf
    canonical_name: Unidentified Aerial Phenomena Task Force (UAPTF)
    survivor:
      name: Unidentified Aerial Phenomena Task Force (UAPTF)
      node_type: investigation
      prior_names: []
    victims:
      - name: Unidentified Aerial Phenomena Task Force (UAPTF)
        node_type: organisation
        prior_names: []
      - name: Unidentified Aerial Phenomena Task Force (UAPTF)
        node_type: project
        prior_names: []
    """


def test_survivor_reclassified_away_still_merges_its_victims(tmp_path, monkeypatch):
    """The real case: the curator kept the `investigation`, the corpus now emits
    `organisation` and `project`. Two live duplicates - the op must still collapse
    them rather than vanish because the chosen survivor is gone."""
    _ledger(tmp_path, monkeypatch, UAPTF)
    conn = _graph(
        ("org", "organisation", "Unidentified Aerial Phenomena Task Force (UAPTF)", 9),
        ("proj", "project", "Unidentified Aerial Phenomena Task Force (UAPTF)", 2),
    )
    result = merge.replay_ledger(conn)

    assert result == {"applied": 1, "absorbed": 0, "lost": 0, "unconfirmed": 0}
    # The most-cited resolved node takes over as survivor.
    assert conn.execute("SELECT retired_at FROM nodes WHERE id='proj'").fetchone()[0]
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='org'").fetchone()[0]
        is None
    )
    # And it carries the victim's claims.
    refs = conn.execute(
        "SELECT COUNT(*) FROM claim_node_refs WHERE node_id='org'"
    ).fetchone()[0]
    assert refs == 11


def test_single_surviving_node_is_absorbed_not_merged(tmp_path, monkeypatch):
    """Only one of the op's nodes is left, so there is no duplicate to collapse.
    Not a failure - reclassification landing - and it must not be reported as one."""
    _ledger(tmp_path, monkeypatch, UAPTF)
    conn = _graph(
        ("org", "organisation", "Unidentified Aerial Phenomena Task Force (UAPTF)", 9),
    )
    assert merge.replay_ledger(conn) == {
        "applied": 0,
        "absorbed": 1,
        "lost": 0,
        "unconfirmed": 0,
    }
    assert (
        conn.execute("SELECT retired_at FROM nodes WHERE id='org'").fetchone()[0]
        is None
    )


ISS = """\
    op: merge
    merge_id: m-iss
    canonical_name: International Space Station
    survivor:
      name: International Space Station
      node_type: place
      prior_names: []
    victims:
      - name: International Space Station (ISS)
        node_type: object
        prior_names: []
    """


def test_absorbed_op_does_not_rename_the_survivor(tmp_path, monkeypatch):
    """A merge's canonical_name applies to a merge. With nothing to merge it must
    not quietly rename the one node that is left - naming is the renames ledger.
    The live instance: the curator's `place` is gone, the `object` remains under
    the acronym-suffixed name."""
    _ledger(tmp_path, monkeypatch, ISS)
    conn = _graph(("iss", "object", "International Space Station (ISS)", 4))
    messages: list[str] = []

    assert merge.replay_ledger(conn, on_progress=messages.append)["absorbed"] == 1
    assert (
        conn.execute("SELECT name FROM nodes WHERE id='iss'").fetchone()[0]
        == "International Space Station (ISS)"
    )
    # Dropping the curator's canonical name is still a dropped decision: say so.
    assert any("canonical" in m for m in messages)


def test_op_with_no_surviving_nodes_is_reported_as_lost(tmp_path, monkeypatch):
    """None of the op's nodes are in the graph. The decision is unrecoverable and
    must be shouted, not counted as a routine skip."""
    _ledger(tmp_path, monkeypatch, UAPTF)
    conn = _graph(("other", "organisation", "Something Else Entirely", 1))
    messages: list[str] = []

    assert merge.replay_ledger(conn, on_progress=messages.append) == {
        "applied": 0,
        "absorbed": 0,
        "lost": 1,
    }
    assert any("ERROR" in m and "LOST" in m for m in messages)
