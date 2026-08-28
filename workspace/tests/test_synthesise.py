"""Brief synthesis: the deterministic graph-slice-to-brief stage."""

import sqlite3

import yaml

from anomalica_common.digest.models import Node, NodeType
from assimilator import synthesise
from assimilator.database import init_db, insert_node


def test_prune_removes_briefs_for_retired_and_renamed_nodes(tmp_path):
    """Emission only writes, so a brief outlives the node it describes.

    Two ways that happens, and both are hazards rather than clutter: the
    assembler takes a brief by slug and will build a page from a dead one.

    A MERGE retires its victims - after Luis/Lue/Lou Elizondo, lou-elizondo.yaml
    pointed at a node that no longer existed.

    A RENAME strands the survivor's own brief. That merge kept node 87788ebc and
    renamed it, so lue-elizondo.yaml and luis-elizondo.yaml both described the
    SAME LIVE NODE - invisible to a retired-check, and enough to republish the
    duplicate page the merge was meant to remove.
    """
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(
        conn, Node(id="live-1", name="Luis Elizondo", node_type=NodeType.person)
    )
    insert_node(conn, Node(id="dead-1", name="Lou Elizondo", node_type=NodeType.person))
    conn.execute("UPDATE nodes SET retired_at = '2026-08-20' WHERE id = 'dead-1'")

    for stem, node_id in (
        ("luis-elizondo", "live-1"),  # current: kept
        ("lue-elizondo", "live-1"),  # stranded by the rename: pruned
        ("lou-elizondo", "dead-1"),  # retired node: pruned
        ("someone-else", "absent-1"),  # node not in the graph at all: pruned
    ):
        (tmp_path / f"{stem}.yaml").write_text(
            yaml.safe_dump({"page": {"node_id": node_id, "slug": stem}})
        )

    removed = synthesise.prune_retired_briefs(
        conn, tmp_path, slug_map={"live-1": "luis-elizondo"}
    )

    assert sorted(removed) == [
        "lou-elizondo.yaml",
        "lue-elizondo.yaml",
        "someone-else.yaml",
    ]
    assert (tmp_path / "luis-elizondo.yaml").exists()


def test_prune_never_deletes_an_unreadable_brief(tmp_path):
    """A file we cannot parse is left for a human. Deleting blind is how a
    parser bug becomes data loss."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    bad = tmp_path / "broken.yaml"
    bad.write_text("{{{ not yaml")
    assert synthesise.prune_retired_briefs(conn, tmp_path, slug_map={}) == []
    assert bad.exists()


def _row(claim_id, attestation=None, speaker=None, work="w1"):
    """A claim row shaped like the synthesise query's SELECT."""
    row = [None] * 21
    row[0] = claim_id
    row[4] = attestation
    row[9] = speaker
    row[-1] = work
    return tuple(row)


def test_importance_ranks_corroborated_over_first_hand_over_nothing():
    """Only signals that are populated are used. `confidence` is 1.0 on all
    31,066 claims and `claim_role` is null on all of them, so ranking by either
    would be ranking by a constant."""
    from assimilator.synthesise import _importance

    corroborated = {"c-corr"}
    def key(r):
        return _importance(r, "n1", corroborated)

    assert key(_row("c-corr")) > key(_row("c1", "first_hand"))
    assert key(_row("c1", "first_hand")) > key(_row("c2", "second_hand"))
    assert key(_row("c2", "second_hand")) > key(_row("c3", "third_hand"))
    assert key(_row("c3", "third_hand")) > key(_row("c4", None))
    # A claim the node itself SPOKE outranks one that merely mentions it.
    assert key(_row("c5", "first_hand", speaker="n1")) > key(_row("c6", "first_hand"))


def test_the_cap_keeps_the_best_claims_not_the_earliest():
    """Document order is right for one record read in sequence and meaningless
    for a person drawn from twenty sources, where it hands the budget to whatever
    each transcript happened to open with."""
    from assimilator.synthesise import _importance, _spread_across_sources

    rows = [_row("weak-%d" % i) for i in range(5)] + [
        _row("strong-0", "first_hand"),
        _row("strong-1", "first_hand"),
    ]
    corroborated = set()

    kept = _spread_across_sources(
        rows, 2, importance=lambda r: _importance(r, "n1", corroborated)
    )

    assert {r[0] for r in kept} == {"strong-0", "strong-1"}


def test_selection_changes_but_the_brief_still_reads_in_document_order():
    """The importance key decides WHICH claims survive; the order they are
    emitted in is unchanged, because brief_hash is computed over that sequence
    and the article has to read as a narrative."""
    from assimilator.synthesise import _importance, _spread_across_sources

    rows = [
        _row("first", "first_hand"),
        _row("middle", None),
        _row("last", "first_hand"),
    ]
    kept = _spread_across_sources(
        rows, 2, importance=lambda r: _importance(r, "n1", set())
    )

    assert [r[0] for r in kept] == ["first", "last"], "document order, not rank order"


def test_without_an_importance_key_the_behaviour_is_unchanged():
    from assimilator.synthesise import _spread_across_sources

    rows = [_row("a"), _row("b"), _row("c")]
    assert [r[0] for r in _spread_across_sources(rows, 2)] == ["a", "b"]
