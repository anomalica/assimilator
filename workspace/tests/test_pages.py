"""One page over several nodes: the composition ledger and what it suppresses."""

import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import pages
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for i in range(3):
        insert_record(
            conn, Record(id=f"r{i}", title=f"R{i}", content_hash=f"sha256:a{i}")
        )
    for nid, name in (
        ("uap", "Unidentified Anomalous Phenomena (UAP)"),
        ("ufo", "Unidentified Flying Object (UFO)"),
        ("other", "Cattle mutilation"),
    ):
        insert_node(conn, Node(id=nid, name=name, node_type=NodeType.topic))
        for i in range(9):
            insert_claim(
                conn,
                Claim(
                    id=f"{nid}-{i}",
                    content=f"claim {i} about {name}",
                    claim_type="testimony",
                    record_id=f"r{i % 3}",
                    node_references=[nid],
                ),
            )
    conn.commit()
    return conn


def _compose(**kw):
    return pages.append_compose_entry(
        kw.get("name", "Unidentified Anomalous Phenomena (UAP)"),
        "topic",
        kw.get(
            "members",
            [
                {
                    "name": "Unidentified Anomalous Phenomena (UAP)",
                    "node_type": "topic",
                },
                {"name": "Unidentified Flying Object (UFO)", "node_type": "topic"},
            ],
        ),
        page_id=kw.get("page_id", "pg1"),
        created_by="workbench/mark",
    )


def test_a_composition_lands_and_replays_after_a_rebuild():
    conn = _graph()
    _compose()

    result = pages.apply_pages(conn)

    assert result["composed"] == 1 and result["lost"] == 0
    page = pages.composed_pages(conn)[0]
    assert page["node_ids"] == ["uap", "ufo"]  # member order is the ledger's
    assert page["slug"] == "unidentified-anomalous-phenomena-uap"
    assert pages.apply_pages(_graph())["composed"] == 1  # the rebuild


def test_a_member_that_no_longer_resolves_leaves_the_page_composed_of_the_rest():
    conn = _graph()
    # The node the ledger names is gone from this graph: a rebuild whose digests
    # no longer carry it, or a name nothing resolves.
    conn.execute("UPDATE nodes SET name = 'Something else entirely' WHERE id = 'ufo'")
    conn.commit()
    _compose()
    lines = []

    result = pages.apply_pages(conn, on_progress=lines.append)

    assert result["composed"] == 1 and result["dropped"] == 1
    assert pages.composed_pages(conn)[0]["node_ids"] == ["uap"]
    assert any("no longer resolves" in ln for ln in lines)


def test_a_member_renamed_or_merged_away_still_resolves_by_alias():
    conn = _graph()
    conn.execute("UPDATE nodes SET name = 'UAPs' WHERE id = 'uap'")
    conn.execute(
        "INSERT INTO aliases (alias, node_id) VALUES "
        "('Unidentified Anomalous Phenomena (UAP)', 'uap')"
    )
    conn.commit()
    _compose()

    assert pages.apply_pages(conn)["dropped"] == 0
    assert pages.composed_pages(conn)[0]["node_ids"] == ["uap", "ufo"]


def test_composing_supersedes_the_member_pages_the_composed_slug_replaces():
    """Both members have a live page; the composition makes one. The member
    whose slug the page keeps is untouched, the other must come down."""
    conn = _graph()
    _compose()

    pages.apply_pages(conn)

    superseded = pages.superseded(conn)
    assert [s["page"] for s in superseded] == ["topics/unidentified-flying-object-ufo"]
    assert "composed into" in superseded[0]["reason"]


def test_a_member_does_not_earn_a_page_of_its_own():
    from assimilator.propose_pages import propose

    conn = _graph()
    _compose()
    pages.apply_pages(conn)

    proposed = {r["node_id"] for r in propose(conn)}

    assert "other" in proposed  # the gate still proposes an uncovered node
    assert proposed.isdisjoint({"uap", "ufo"})


def test_decomposing_removes_the_page_and_its_supersessions():
    conn = _graph()
    _compose()
    pages.apply_pages(conn)
    pages.append_decompose_entry("pg1", created_by="workbench/mark")

    result = pages.apply_pages(conn)

    assert result["composed"] == 0
    assert pages.composed_pages(conn) == [] and pages.superseded(conn) == []
    assert pages.member_node_ids(conn) == {}


def test_a_covered_members_own_brief_is_pruned(tmp_path):
    """The page covering a node IS that node's brief. Left standing, the
    member's own brief lets the assembler build the duplicate page the
    composition exists to remove."""
    import yaml
    from assimilator import synthesise

    conn = _graph()
    _compose()
    pages.apply_pages(conn)
    briefs = tmp_path / "topics"
    briefs.mkdir()
    for slug, node_id in (
        ("unidentified-anomalous-phenomena-uap", "uap"),  # the composed page's path
        ("unidentified-flying-object-ufo", "ufo"),  # the member's own: pruned
        ("cattle-mutilation", "other"),  # uncovered: kept
    ):
        (briefs / f"{slug}.yaml").write_text(
            yaml.safe_dump({"page": {"nodes": [{"node_id": node_id}], "slug": slug}})
        )

    removed = synthesise.prune_retired_briefs(conn, tmp_path)

    assert removed == ["topics/unidentified-flying-object-ufo.yaml"]
    assert (briefs / "unidentified-anomalous-phenomena-uap.yaml").exists()
    assert (briefs / "cattle-mutilation.yaml").exists()
