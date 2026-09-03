"""Record tags: a human's "this record is about this node", replayed on rebuild."""

import sqlite3

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import tags
from assimilator.database import init_db, insert_claim, insert_node, insert_record


def _graph():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="Diner", content_hash="sha256:aa"))
    insert_record(conn, Record(id="r2", title="Report", content_hash="sha256:bb"))
    insert_node(conn, Node(id="t1", name="Summoning", node_type=NodeType.topic))
    insert_node(conn, Node(id="p1", name="Jesse Michels", node_type=NodeType.person))
    insert_claim(
        conn,
        Claim(
            id="c1",
            content="x",
            claim_type="testimony",
            record_id="r1",
            node_references=["p1"],
        ),
    )
    conn.execute("INSERT INTO record_nodes (record_id, node_id) VALUES ('r1', 'p1')")
    conn.commit()
    return conn


def _links(conn):
    return set(conn.execute("SELECT record_id, node_id FROM record_nodes").fetchall())


def test_a_tag_lands_as_a_record_nodes_row_and_replays_after_a_rebuild(tmp_path):
    conn = _graph()
    tags.append_tag_entry(
        "Summoning", "topic", "sha256:aa", tag_id="tg1", created_by="wb"
    )

    result = tags.apply_tags(conn)

    assert result == {
        "applied": 1,
        "created": 0,
        "pending": 0,
        "withdrawn": 0,
        "lost": 0,
    }
    assert (
        conn.execute("SELECT status FROM record_tags WHERE tag_id='tg1'").fetchone()[0]
        == "applied"
    )
    assert ("r1", "t1") in _links(conn)
    assert tags.apply_tags(conn)["applied"] == 0  # idempotent on tag_id

    fresh = _graph()  # the rebuild: same digests, empty derived tables
    assert tags.replay_tags(fresh)["applied"] == 1
    assert ("r1", "t1") in _links(fresh)


def test_a_tag_attaches_no_claim():
    conn = _graph()
    tags.append_tag_entry("Summoning", "topic", "sha256:aa", tag_id="tg1")
    tags.apply_tags(conn)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM claim_node_refs WHERE node_id='t1'"
        ).fetchone()[0]
        == 0
    )


def test_a_name_with_no_node_is_created_the_seeded_topic_case():
    conn = _graph()
    tags.append_tag_entry("The Cover-Up", "topic", "sha256:bb", tag_id="tg1")

    result = tags.apply_tags(conn)

    assert result["created"] == 1 and result["applied"] == 1
    row = conn.execute(
        "SELECT id, node_type FROM nodes WHERE name = 'The Cover-Up'"
    ).fetchone()
    assert row[1] == "topic"
    assert ("r2", row[0]) in _links(conn)


def test_a_tag_follows_a_rename_and_a_merge_by_alias():
    conn = _graph()
    conn.execute("UPDATE nodes SET name = 'Contact by invitation' WHERE id = 't1'")
    conn.execute("INSERT INTO aliases (alias, node_id) VALUES ('Summoning', 't1')")
    conn.commit()
    tags.append_tag_entry("Summoning", "topic", "sha256:aa", tag_id="tg1")

    assert tags.apply_tags(conn)["created"] == 0
    assert ("r1", "t1") in _links(conn)


def test_a_record_not_yet_digested_keeps_the_tag_pending_then_lands_it():
    """Only a third of the store's records have a graph row; a reviewer tagging
    one the pipeline has not caught up with is right NOW, and that judgement
    must not be lost for an ordering reason."""
    conn = _graph()
    tags.append_tag_entry("Summoning", "topic", "sha256:cc", tag_id="tg1")

    result = tags.apply_tags(conn)

    assert result["pending"] == 1 and result["applied"] == 0 and result["lost"] == 0
    status, reason = conn.execute(
        "SELECT status, reason FROM record_tags WHERE tag_id='tg1'"
    ).fetchone()
    assert status == "pending" and "not digested" in reason

    insert_record(conn, Record(id="r3", title="Late", content_hash="sha256:cc"))
    assert tags.apply_tags(conn)["applied"] == 1
    assert ("r3", "t1") in _links(conn)
    assert (
        conn.execute("SELECT status FROM record_tags WHERE tag_id='tg1'").fetchone()[0]
        == "applied"
    )


def test_a_malformed_tag_is_lost_loudly_not_dropped():
    conn = _graph()
    tags.append_tag_entry("Summoning", "planet", "sha256:aa", tag_id="tg1")
    lines = []

    result = tags.apply_tags(conn, on_progress=lines.append)

    assert result["lost"] == 1 and result["applied"] == 0
    assert any("ERROR" in ln and "LOST" in ln for ln in lines)
    assert (
        conn.execute("SELECT status FROM record_tags WHERE tag_id='tg1'").fetchone()[0]
        == "lost"
    )


def test_only_a_topic_is_created_from_a_tag_a_person_stays_pending():
    """A misspelt person's name must not silently become a second person."""
    conn = _graph()
    tags.append_tag_entry("Jesse Michaels", "person", "sha256:bb", tag_id="tg1")

    result = tags.apply_tags(conn)

    assert result["pending"] == 1 and result["created"] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM nodes WHERE node_type='person'").fetchone()[
            0
        ]
        == 1
    )


def test_a_created_topic_gets_a_leading_capital_and_no_stray_whitespace():
    conn = _graph()
    tags.append_tag_entry("  telepathy ", "topic", "sha256:bb", tag_id="tg1")
    tags.apply_tags(conn)
    assert (
        conn.execute(
            "SELECT name FROM nodes WHERE node_type='topic' AND id<>'t1'"
        ).fetchone()[0]
        == "Telepathy"
    )


def test_a_tag_written_before_a_rename_replays_after_it(tmp_path, monkeypatch):
    """Replay is a stream and a rename can follow a tag. The tag names the OLD
    name; replay must find the renamed node through its alias rather than mint
    a duplicate topic beside it."""
    from assimilator.merge import rename_node, replay_renames

    conn = _graph()
    tags.append_tag_entry("Summoning", "topic", "sha256:aa", tag_id="tg1")
    rename_node(conn, "t1", "Contact by invitation", "rn1")

    fresh = _graph()  # rebuild: the digests still say "Summoning"
    replay_renames(fresh)
    result = tags.replay_tags(fresh)

    assert result["created"] == 0 and result["applied"] == 1
    assert ("r1", "t1") in _links(fresh)
    assert (
        fresh.execute("SELECT COUNT(*) FROM nodes WHERE node_type='topic'").fetchone()[
            0
        ]
        == 1
    )


def test_untag_removes_only_the_tagged_link_never_an_import_link():
    conn = _graph()
    tags.append_tag_entry("Summoning", "topic", "sha256:aa", tag_id="tg1")
    tags.append_tag_entry("Jesse Michels", "person", "sha256:aa", tag_id="tg2")
    tags.apply_tags(conn)
    tags.append_untag_entry("tg1")
    tags.append_untag_entry("tg2")

    result = tags.apply_tags(conn)

    assert result["withdrawn"] == 2
    assert ("r1", "t1") not in _links(conn)
    assert ("r1", "p1") in _links(conn)  # the importer's own link stands
    undone = conn.execute(
        "SELECT COUNT(*) FROM record_tags WHERE undone_at IS NOT NULL"
    ).fetchone()[0]
    assert undone == 2
    assert tags.apply_tags(conn)["withdrawn"] == 0
