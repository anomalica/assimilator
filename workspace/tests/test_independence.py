"""Independence counts distinct provenance ROOTS, not distinct records.

Two records relaying one anonymous email are two records and one source - the
page a reader would object to, and the case `source_count` cannot see. Every
uncertain identity collapses toward FEWER roots, because over-counting is the
unsafe direction: splitting later raises independence, but nothing lowers it once
a page has been published on an inflated number.
"""

from __future__ import annotations

import sqlite3

import pytest

from anomalica_common.digest.models import (
    Claim,
    Node,
    ProvenanceChain,
    Record,
    SourceAnchor,
)
from assimilator.database import (
    init_db,
    insert_alias,
    insert_claim,
    insert_node,
    insert_record,
)
from assimilator.evidence import (
    rebuild_evidence_units,
    replace_claim_anchors,
    replace_claim_provenance_root,
)
from assimilator.independence import independence_for_nodes


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    init_db(c)
    insert_node(c, Node(id="subject", node_type="topic", name="The Subject"))
    for rid in ("r1", "r2", "r3"):
        insert_record(c, Record(id=rid, title=rid.upper()))
    c.commit()
    return c


def _claim(c, cid, record, chain, speaker=None):
    insert_claim(
        c,
        Claim(
            id=cid,
            content=cid,
            claim_type="observation",
            record_id=record,
            speaker_id=speaker,
            node_references=["subject"],
            provenance_chain=chain,
        ),
    )
    c.execute(
        "INSERT OR IGNORE INTO assets (asset_hash, metadata) VALUES (?, '{}')",
        ("sha256:" + "a" * 64,),
    )
    position = c.execute("SELECT COUNT(*) FROM claim_anchors").fetchone()[0] * 10
    replace_claim_anchors(
        c,
        cid,
        [
            SourceAnchor(
                asset_hash="sha256:" + "a" * 64,
                record_page=1,
                asset_file_page=1,
                asset_text_sha256="sha256:" + "b" * 64,
                asset_span={"start": position, "end": position + 5},
                body_span={"start": position, "end": position + 5},
                quote="xxxxx",
            )
        ],
    )
    replace_claim_provenance_root(c, cid)
    rebuild_evidence_units([c])
    c.commit()


def test_two_records_relaying_one_origin_are_one_source(conn):
    """The case the whole field exists for: a podcast and an article both citing
    the same named origin are two records and one source."""
    insert_node(
        conn,
        Node(id="dia", node_type="organisation", name="Defense Intelligence Agency"),
    )
    chain = ProvenanceChain(origin_kind="named", origin="Defense Intelligence Agency")
    _claim(conn, "c1", "r1", chain)
    _claim(conn, "c2", "r2", chain)

    assert independence_for_nodes(conn, ["subject"])["subject"].sources == 1


def test_an_alias_of_one_origin_does_not_become_a_second_source(conn):
    """ "DIA" and "Defense Intelligence Agency" are one root - the origin resolves
    through the alias graph, so an acronym cannot double-count."""
    insert_node(
        conn,
        Node(id="dia", node_type="organisation", name="Defense Intelligence Agency"),
    )
    insert_alias(conn, "DIA", "dia")
    conn.commit()
    _claim(conn, "c1", "r1", ProvenanceChain(origin_kind="named", origin="DIA"))
    _claim(
        conn,
        "c2",
        "r2",
        ProvenanceChain(origin_kind="named", origin="Defense Intelligence Agency"),
    )

    assert independence_for_nodes(conn, ["subject"])["subject"].sources == 1


def test_anonymous_origins_remain_unknown(conn):
    for i, rid in enumerate(("r1", "r2", "r3")):
        _claim(
            conn,
            f"c{i}",
            rid,
            ProvenanceChain(origin_kind="anonymous", origin="a source"),
        )

    score = independence_for_nodes(conn, ["subject"])["subject"]
    assert score.sources is None
    assert score.scored_claims == 0 and score.unscored_claims == 3


def test_origin_refs_do_not_manufacture_established_roots(conn):
    _claim(
        conn,
        "c1",
        "r1",
        ProvenanceChain(
            origin_kind="anonymous", origin="a controller", origin_ref="controller-1"
        ),
    )
    _claim(
        conn,
        "c2",
        "r1",
        ProvenanceChain(
            origin_kind="anonymous", origin="a technician", origin_ref="technician-1"
        ),
    )
    # Different source-local handles in another record cannot prove global
    # assertion roots. ADR 0051 therefore leaves every anonymous claim unknown.
    _claim(
        conn,
        "c3",
        "r2",
        ProvenanceChain(
            origin_kind="anonymous", origin="a witness", origin_ref="witness-1"
        ),
    )

    assert independence_for_nodes(conn, ["subject"])["subject"].sources is None


def test_distinct_speakers_are_distinct_sources(conn):
    """The other direction: independence must still RISE where the evidence
    supports it, or the measure is just a constant."""
    for who in ("alice", "bob"):
        insert_node(conn, Node(id=who, node_type="person", name=who.title()))
    conn.commit()
    _claim(conn, "c1", "r1", ProvenanceChain(origin_kind="speaker"), speaker="alice")
    _claim(conn, "c2", "r1", ProvenanceChain(origin_kind="speaker"), speaker="bob")

    # One record, two voices - independence is not a record count in either
    # direction, which is why it cannot replace the source-spread measure.
    assert independence_for_nodes(conn, ["subject"])["subject"].sources == 2


def test_a_chainless_claim_is_unscored_not_a_source(conn):
    """A pre-0044 claim has no chain, so its root is unknowable. Counting all
    such claims as one shared "unknown" root would read as one shared SOURCE and
    quietly corroborate everything pre-0044 with everything else."""
    insert_node(conn, Node(id="nasa", node_type="organisation", name="NASA"))
    _claim(conn, "c1", "r1", ProvenanceChain(origin_kind="named", origin="NASA"))
    _claim(conn, "c2", "r2", None)
    _claim(conn, "c3", "r3", None)

    score = independence_for_nodes(conn, ["subject"])["subject"]
    assert score.sources == 1
    assert score.scored_claims == 1 and score.unscored_claims == 2


def test_a_node_with_no_scoreable_claim_reports_none_not_zero(conn):
    """None means "cannot be computed"; 0 would mean "no independent sources",
    which is a finding rather than an absence."""
    _claim(conn, "c1", "r1", None)

    score = independence_for_nodes(conn, ["subject"])["subject"]
    assert score.sources is None
    assert score.unscored_fraction == 1.0


def test_the_unscored_fraction_is_what_makes_a_count_trustworthy(conn):
    """A count computed from 3% of a node's claims and one computed from all of
    them are both integers. Live example: David Fravor reports 2 origins from 643
    claims, of which 616 are unscored - the number is technically correct and
    means almost nothing without the fraction beside it."""
    insert_node(conn, Node(id="nasa", node_type="organisation", name="NASA"))
    _claim(conn, "c1", "r1", ProvenanceChain(origin_kind="named", origin="NASA"))
    for i in range(9):
        _claim(conn, f"u{i}", "r2", None)

    score = independence_for_nodes(conn, ["subject"])["subject"]
    assert score.sources == 1
    assert round(score.unscored_fraction, 2) == 0.9
