"""Brief synthesis: the deterministic graph-slice-to-brief stage."""

import sqlite3

import yaml

from anomalica_common.digest.models import Claim, Node, NodeType, Record
from assimilator import synthesise
from assimilator.database import init_db, insert_claim, insert_node, insert_record


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
    insert_node(conn, Node(id="empty-1", name="No Claims", node_type=NodeType.person))
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    _claimed(conn, "live-1", 1)

    for rel, node_id in (
        ("people/luis-elizondo", "live-1"),  # current: kept
        ("people/lue-elizondo", "live-1"),  # stranded by the rename: pruned
        ("people/lou-elizondo", "dead-1"),  # retired node: pruned
        ("people/someone-else", "absent-1"),  # node not in the graph at all: pruned
        ("luis-elizondo", "live-1"),  # the pre-section flat layout: pruned
        ("organisations/luis-elizondo", "live-1"),  # wrong section: pruned
        ("people/no-claims", "empty-1"),  # live node, no claims: pruned
    ):
        path = tmp_path / f"{rel}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {"page": {"nodes": [{"node_id": node_id}], "slug": rel.split("/")[-1]}}
            )
        )

    removed = synthesise.prune_retired_briefs(
        conn, tmp_path, slug_map={"live-1": "luis-elizondo"}
    )

    assert sorted(removed) == [
        "luis-elizondo.yaml",
        "organisations/luis-elizondo.yaml",
        "people/lou-elizondo.yaml",
        "people/lue-elizondo.yaml",
        "people/no-claims.yaml",
        "people/someone-else.yaml",
    ]
    assert (tmp_path / "people" / "luis-elizondo.yaml").exists()


def test_prune_never_deletes_an_unreadable_brief(tmp_path):
    """A file we cannot parse is left for a human. Deleting blind is how a
    parser bug becomes data loss."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    bad = tmp_path / "broken.yaml"
    bad.write_text("{{{ not yaml")
    assert synthesise.prune_retired_briefs(conn, tmp_path, slug_map={}) == []
    assert bad.exists()


def _claimed(conn, node_id: str, n: int, record: str = "r1") -> None:
    for i in range(n):
        insert_claim(
            conn,
            Claim(
                id=f"{node_id}-c{i}",
                content=f"claim {i} about {node_id}",
                claim_type="testimony",
                record_id=record,
                node_references=[node_id],
            ),
        )


def _two_types_one_name(tmp_path):
    """An event and a project both called "Apollo 14", both proposed. The live
    graph held exactly this on 2026-09-02, plus SETI as a project and a topic."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    insert_node(conn, Node(id="ev-1", name="Apollo 14", node_type=NodeType.event))
    insert_node(conn, Node(id="pr-1", name="Apollo 14", node_type=NodeType.project))
    _claimed(conn, "ev-1", 2)
    _claimed(conn, "pr-1", 2)
    for nid, t in (("ev-1", "event"), ("pr-1", "project")):
        conn.execute(
            "INSERT INTO page_proposals (node_id, node_type, tier, claim_count, "
            "source_count, status, computed_at) VALUES (?, ?, 'page-worthy', 2, 1, "
            "'proposed', 'T')",
            (nid, t),
        )
    conn.commit()
    return conn


def test_two_types_sharing_a_name_get_two_briefs(tmp_path):
    """The slug is disambiguated only within a type - /events/apollo-14 and
    /projects/apollo-14 never clash as URLs - so the brief PATH must carry the
    section too. Keyed on the slug alone, the two pages had one file, whichever
    node wrote last owned it, and the scheduler re-emitted the other forever."""
    conn = _two_types_one_name(tmp_path)
    out = tmp_path / "briefs"

    result = synthesise.emit_all(conn, out)

    assert result["written"] == 2
    assert (out / "events" / "apollo-14.yaml").is_file()
    assert (out / "projects" / "apollo-14.yaml").is_file()
    assert synthesise.brief_node_id(out / "events" / "apollo-14.yaml") == "ev-1"
    assert synthesise.brief_node_id(out / "projects" / "apollo-14.yaml") == "pr-1"
    assert synthesise.prune_retired_briefs(conn, out) == []


def test_a_single_node_emit_uses_the_global_slug_map(tmp_path):
    """Two PROJECTS called "Apollo 14" collide within a type, so the loser's
    slug carries a node-id suffix. Emitted alone (the scheduler runs synthesise
    per node), the loser used to take the per-node canonical slug - the base -
    and overwrite the winner's brief."""
    conn = _two_types_one_name(tmp_path)
    insert_node(conn, Node(id="pr-2", name="Apollo 14", node_type=NodeType.project))
    _claimed(conn, "pr-2", 2)
    conn.commit()
    db = tmp_path / "graph.db"
    disk = sqlite3.connect(db)
    conn.backup(disk)
    disk.close()
    out = tmp_path / "briefs"

    assert synthesise.main(["--db", str(db), "--out", str(out), "--node", "pr-2"]) == 0

    assert (out / "projects" / "apollo-14-pr-2.yaml").is_file()
    assert not (out / "projects" / "apollo-14.yaml").exists()


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


def _belonging_graph(tmp_path):
    """One node with three claims: one verified, one suspect, one unreviewed."""
    from assimilator.database import insert_claim, insert_record, set_claim_ref_status
    from anomalica_common.digest.models import Claim, Record

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_node(conn, Node(id="N", node_type=NodeType.event, name="An Event"))
    for cid, text in (
        ("c1", "belongs here"),
        ("c2", "about something else"),
        ("c3", "unchecked"),
    ):
        insert_claim(
            conn, Claim(id=cid, content=text, claim_type="testimony", record_id="r1")
        )
        conn.execute(
            "INSERT INTO claim_node_refs (claim_id, node_id) VALUES (?, 'N')", (cid,)
        )
    set_claim_ref_status(conn, "c1", "N", "verified", "read it", "test")
    set_claim_ref_status(conn, "c2", "N", "suspect", "not this event", "test")
    conn.commit()
    return conn


def test_a_suspect_claim_is_excluded_from_the_brief(tmp_path):
    """Presence on a node is evidence a claim was ATTACHED, not that it belongs.

    Leaving suspect claims in and merely flagging them makes correct assembly
    depend on every consumer remembering to filter, and lets them displace
    usable claims from the cap.
    """
    conn = _belonging_graph(tmp_path)
    brief = synthesise.build_entity_brief(conn, "N")
    ids = {c["claim_id"] for c in brief["claims"]}
    assert "c2" not in ids
    assert ids == {"c1", "c3"}


def test_the_brief_reports_what_it_excluded(tmp_path):
    conn = _belonging_graph(tmp_path)
    brief = synthesise.build_entity_brief(conn, "N")
    assert brief["belonging"] == {
        "verified": 1,
        "suspect_excluded": 1,
        "unreviewed": 1,
    }


def test_unreviewed_is_not_reported_as_verified(tmp_path):
    """The distinction the whole table exists for."""
    conn = _belonging_graph(tmp_path)
    brief = synthesise.build_entity_brief(conn, "N")
    by_id = {c["claim_id"]: c["attachment"] for c in brief["claims"]}
    assert by_id["c1"] == "verified"
    assert by_id["c3"] == "unreviewed"


def test_a_small_source_is_kept_when_the_cap_has_room():
    """MIN_SOURCE_CLAIMS is a tiebreaker, not a filter.

    It exists so a two-claim record scoring 100% focus cannot outrank a primary
    account. Applied as a filter it dropped every small source whenever ANY
    source was substantial - even with slots free. Rendlesham has 7 sources and
    a cap of 5, three cleared the minimum, and the other four were discarded
    with two slots unused, losing 6 of the node's 49 claims.
    """
    # (id, ..., work_id) - only the last column matters to the spread.
    rows = []
    for work, n in (("big", 8), ("mid", 6), ("tiny_a", 2), ("tiny_b", 1)):
        rows += [(f"{work}-{i}",) + (None,) * 19 + (work,) for i in range(n)]

    kept = synthesise._spread_across_sources(rows, cap=100, max_sources=3)

    works = {r[-1] for r in kept}
    assert "big" in works and "mid" in works
    assert "tiny_a" in works, "a slot was free and a real source was dropped"
    assert "tiny_b" not in works, "the cap still binds at 3"


def test_substantial_sources_still_rank_first():
    """The tiebreaker must still do its job: a tiny source cannot displace one
    that carries the account."""
    rows = []
    for work, n in (("big", 9), ("tiny", 1)):
        rows += [(f"{work}-{i}",) + (None,) * 19 + (work,) for i in range(n)]

    kept = synthesise._spread_across_sources(rows, cap=100, max_sources=1)

    assert {r[-1] for r in kept} == {"big"}


def _sized_graph(n_claims: int):
    from anomalica_common.digest.models import Claim, Record
    from assimilator.database import insert_claim, insert_record

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R1"))
    insert_node(conn, Node(id="N", node_type=NodeType.event, name="An Event"))
    for i in range(n_claims):
        cid = f"c{i}"
        insert_claim(
            conn,
            Claim(
                id=cid,
                content="a sentence of evidence " * 20,
                claim_type="testimony",
                record_id="r1",
            ),
        )
        conn.execute(
            "INSERT INTO claim_node_refs (claim_id, node_id) VALUES (?, 'N')", (cid,)
        )
    conn.commit()
    return conn


def test_a_brief_that_fits_carries_everything_and_says_its_size(monkeypatch):
    """The 600-claim cap cut 76% of what the graph knew about its largest
    subjects, uncitable and unreported. Size now comes from the consuming
    stage's context window, and a brief that fits is not cut at all."""
    conn = _sized_graph(50)
    monkeypatch.setattr(synthesise, "consuming_window", lambda *a, **k: 1_050_000)

    brief = synthesise.build_entity_brief(conn, "N")

    assert brief["page"]["claim_count"] == 50
    assert "truncated" not in brief, "nothing was cut, so nothing should be claimed"
    assert brief["size"]["claims"] == 50
    assert brief["size"]["tokens_estimated"] > 0
    assert brief["size"]["sized_against"] == 1_050_000


def test_a_truncated_brief_says_so(monkeypatch):
    """A claim cut here cannot appear in the article and cannot be cited. Left
    unsaid, a page built from a quarter of the evidence is indistinguishable
    from one built from all of it."""
    conn = _sized_graph(50)
    monkeypatch.setattr(synthesise, "consuming_window", lambda *a, **k: 4_000)

    brief = synthesise.build_entity_brief(conn, "N")

    assert brief["page"]["claim_count"] < 50
    assert brief["truncated"]["available"] == 50
    assert brief["truncated"]["kept"] == brief["page"]["claim_count"]
    assert "4,000" in brief["truncated"]["why"]


def test_an_unreadable_policy_cuts_nothing(monkeypatch):
    """A guessed window would reintroduce the fault this replaced, invisibly."""
    conn = _sized_graph(50)
    monkeypatch.setattr(synthesise, "consuming_window", lambda *a, **k: 0)

    brief = synthesise.build_entity_brief(conn, "N")

    assert brief["page"]["claim_count"] == 50
    assert "truncated" not in brief


def test_the_header_parse_reads_the_page_and_never_the_bulk(tmp_path):
    """The scheduler needs page, generated and brief_hash from every brief on
    every queue rebuild. Parsing 814 briefs whole for that was 105 seconds; the
    header is under 2 KB. The cut is at the first bulk key, so a related node's
    node_id - at the same indent as the page's - is never read as the page's,
    and a fault in the body does not hide the header."""
    path = tmp_path / "events" / "apollo-14.yaml"
    path.parent.mkdir()
    path.write_text(
        "schema: anomalica/brief/2\n"
        "brief_hash: abc\n"
        "page:\n  kind: entity\n  nodes:\n  - node_id: ev-1\n    node_type: event\n"
        "  node_type: event\n  slug: apollo-14\n"
        "generated:\n  graph_version: 'v1'\n"
        "related_nodes:\n- node_id: other-1\n  slug: other\n"
        "claims:\n- claim_id: c1\n  content: 'unterminated\n"
    )

    header = synthesise.brief_header(path)

    assert [n["node_id"] for n in header["page"]["nodes"]] == ["ev-1"]
    assert header["generated"]["graph_version"] == "v1"
    assert header["brief_hash"] == "abc"
    assert "related_nodes" not in header and "claims" not in header
    assert synthesise.brief_node_ids(path) == ["ev-1"]


def test_a_brief_with_no_readable_header_yields_nothing(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("{{{ not yaml")
    assert synthesise.brief_header(bad) is None
    assert synthesise.brief_node_id(bad) is None


def test_the_size_estimate_models_what_a_consumer_renders_not_the_file(tmp_path):
    """The file holds about four characters of ids, hashes and provenance for
    every character of claim text, and none of it reaches a model. Sized as
    the file, the largest brief reads as over a window it fits with room to
    spare, and the cut that would trigger drops evidence for nothing. The
    estimate is claim text at the measured ratio plus a line of framing."""
    from assimilator.brief_size import CHARS_PER_TOKEN

    conn = _two_types_one_name(tmp_path)
    brief = synthesise.build_entity_brief(conn, "ev-1")
    text = sum(
        len(c["content"] or "") + len(c["original_excerpt"] or "")
        for c in brief["claims"]
    )
    expected = sum(
        int(
            (len(c["content"] or "") + len(c["original_excerpt"] or ""))
            / CHARS_PER_TOKEN
        )
        + 1
        + synthesise._CLAIM_OVERHEAD_TOKENS
        for c in brief["claims"]
    )

    assert brief["size"]["tokens_estimated"] == expected
    on_disk = len(synthesise.dump_brief_yaml(brief))
    assert on_disk > 3 * text  # the file is mostly not claim text
    assert brief["size"]["tokens_estimated"] < on_disk / CHARS_PER_TOKEN / 2


def test_page_title_writes_ufo_and_uap_bare_with_a_leading_capital():
    from assimilator.synthesise import page_title

    assert (
        page_title("Mutual Unidentified Flying Object (UFO) Network (MUFON)")
        == "Mutual UFO Network (MUFON)"
    )
    assert (
        page_title("Congressional Unidentified Aerial Phenomena (UAP) Caucus")
        == "Congressional UAP Caucus"
    )
    assert (
        page_title("Unidentified Aerial Phenomena Task Force (UAPTF)")
        == "Unidentified Aerial Phenomena Task Force (UAPTF)"
    )
    assert page_title("telepathy") == "Telepathy"
    assert page_title("Cattle mutilation") == "Cattle mutilation"


def test_a_renamed_node_with_a_brief_gets_it_refiled_at_the_new_slug(tmp_path):
    """A rename of an unproposed node left its page with no brief: the old-slug
    brief was pruned and nothing wrote the new one. The brief follows the name."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    insert_node(conn, Node(id="act", name="Old Act Name", node_type=NodeType.document))
    insert_node(
        conn, Node(id="quiet", name="Never Had One", node_type=NodeType.document)
    )
    _claimed(conn, "act", 2)
    _claimed(conn, "quiet", 2)
    out = tmp_path / "briefs"
    synthesise.write_brief(synthesise.build_entity_brief(conn, "act", {}), out)
    assert (out / "documents" / "old-act-name.yaml").is_file()
    conn.execute("UPDATE nodes SET name = 'New Act Name' WHERE id = 'act'")

    moved = synthesise.refile_briefs(conn, {"act", "quiet"}, out)

    assert moved["written"] == ["documents/new-act-name.yaml"]
    assert moved["pruned"] == ["documents/old-act-name.yaml"]
    assert not (out / "documents" / "old-act-name.yaml").exists()
    assert not (out / "documents" / "never-had-one.yaml").exists()


def test_a_composed_page_unions_its_members_and_dedupes_the_shared_claims(tmp_path):
    """UFO and UAP stay separate nodes - they share 26 claims of 2,068, so a
    merge would destroy which word each source used - and one page covers both.
    A naive union would put every count on the page out by the shared claims."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_record(conn, Record(id="r1", title="R", content_hash="sha256:aa"))
    insert_node(conn, Node(id="uap", name="Anomalous (UAP)", node_type=NodeType.topic))
    insert_node(conn, Node(id="ufo", name="Flying (UFO)", node_type=NodeType.topic))
    for nid in ("uap", "ufo"):
        for i in range(3):
            insert_claim(
                conn,
                Claim(
                    id=f"{nid}-{i}",
                    content=f"claim {i} about {nid}",
                    claim_type="testimony",
                    record_id="r1",
                    node_references=[nid],
                ),
            )
    # One claim reached through both members, and the same claim twice by hash.
    conn.execute(
        "INSERT INTO claim_node_refs (claim_id, node_id) VALUES ('ufo-0', 'uap')"
    )
    conn.execute(
        "UPDATE claims SET claim_hash = 'shared' WHERE id IN ('uap-1', 'ufo-1')"
    )
    conn.commit()

    brief = synthesise.build_entity_brief(
        conn,
        "uap",
        {},
        node_ids=["uap", "ufo"],
        page={"name": "UAP and UFO", "slug": "uap-and-ufo", "node_type": "topic"},
    )

    page = brief["page"]
    assert [n["node_id"] for n in page["nodes"]] == ["uap", "ufo"]
    assert page["title"] == "UAP and UFO" and page["slug"] == "uap-and-ufo"
    ids = [c["claim_id"] for c in brief["claims"]]
    assert len(ids) == len(set(ids))  # ufo-0 reached through both members: once
    assert "uap-1" in ids and "ufo-1" not in ids  # same hash: the earlier member's
    assert len(ids) == 5
    assert brief["schema"] == "anomalica/brief/2"


def test_a_composed_pages_brief_hash_covers_its_member_list():
    a = synthesise.brief_hash(["n1"], "entity", [("c1", "h1")])
    b = synthesise.brief_hash(["n1", "n2"], "entity", [("c1", "h1")])
    assert a != b  # adding a member changes what the page should say
