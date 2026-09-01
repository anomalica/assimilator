"""Publishing the audit record without republishing the sources.

A brief carries original_excerpt - the verbatim source sentence - on every claim.
13 of 100 records are copyright restricted and touch 586 of 691 proposed pages, so
an unredacted publish puts thousands of verbatim passages from copyrighted books
on the CDN in one irreversible deploy.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from anomalica_common.digest.models import Node, NodeType
from assimilator.database import init_db, insert_node

from assimilator.publish_briefs import (
    PUBLISHABLE_EXCERPT_STATUSES,
    copyright_status,
    publish_briefs,
    redact_brief,
)


def _store(tmp_path: Path, entries: dict[str, str | None]) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    for h, status in entries.items():
        block = f"copyright:\n  status: {status}\n" if status else ""
        (store / f"{h}.md").write_text(f"---\ntitle: A Record\n{block}---\n\nbody\n")
    return store


def _brief(*claims) -> dict:
    return {
        "schema": "anomalica/brief/1",
        "page": {"slug": "x", "title": "X"},
        "claims": [
            {
                "claim_id": f"c{i}",
                "content": f"paraphrase {i}",
                "original_excerpt": f"VERBATIM SOURCE TEXT {i}",
                "provenance": {"content_hash": "sha256:" + h},
            }
            for i, h in enumerate(claims)
        ],
    }


def test_a_restricted_source_has_its_excerpt_withheld_and_says_so(tmp_path):
    """Silent removal is not acceptable: a reader must be able to tell "this
    source is copyrighted" from "we have no excerpt". The first is a licence
    boundary; the second would read as a gap in our evidence."""
    store = _store(tmp_path, {"a" * 64: "restricted"})

    out, counts = redact_brief(_brief("a" * 64), store)

    claim = out["claims"][0]
    assert claim["original_excerpt"] is None
    assert claim["excerpt_withheld"] == "restricted"
    assert claim["content"] == "paraphrase 0", "our own paraphrase is unaffected"
    assert counts == {"restricted": 1}


@pytest.mark.parametrize("status", ["public_domain", "publicly_accessible"])
def test_mark_ruled_these_publish_their_excerpts(status, tmp_path):
    """Mark, 2026-08-28, asked whether publicly_accessible excerpts go public:
    "yes, always"."""
    store = _store(tmp_path, {"b" * 64: status})

    out, _ = redact_brief(_brief("b" * 64), store)

    assert out["claims"][0]["original_excerpt"] == "VERBATIM SOURCE TEXT 0"
    assert "excerpt_withheld" not in out["claims"][0]


def test_a_record_the_store_cannot_resolve_is_withheld(tmp_path):
    """Absent is unknown and unknown is no. This is every record the store does
    not answer for, and reading no answer as permission is the failure the whole
    module exists to prevent."""
    store = _store(tmp_path, {})

    out, counts = redact_brief(_brief("c" * 64), store)

    assert out["claims"][0]["original_excerpt"] is None
    assert out["claims"][0]["excerpt_withheld"] == "unknown"
    assert counts == {"unknown": 1}


def test_a_record_with_no_copyright_block_is_withheld(tmp_path):
    """Frontmatter present, copyright absent - the pre-field corpus."""
    store = _store(tmp_path, {"d" * 64: None})

    out, _ = redact_brief(_brief("d" * 64), store)

    assert out["claims"][0]["excerpt_withheld"] == "unknown"


def test_the_status_is_read_from_the_nested_key_not_a_flat_one(tmp_path):
    """copyright.status, not copyright_status. The flat name was reported once,
    and filtering on it returns zero - which reads as "no copyright data" and is
    how a fail-open gate gets built by accident."""
    store = tmp_path / "store"
    store.mkdir()
    (store / f"{'e' * 64}.md").write_text(
        "---\ncopyright_status: public_domain\n---\n\nbody\n"
    )

    assert copyright_status(store, "sha256:" + "e" * 64) == "unknown"


def test_one_restricted_source_does_not_redact_the_others(tmp_path):
    """Redaction is per claim, by its own source. A mixed brief keeps everything
    it is entitled to keep."""
    store = _store(
        tmp_path,
        {
            "a" * 64: "restricted",
            "b" * 64: "public_domain",
            "c" * 64: "publicly_accessible",
        },
    )

    out, counts = redact_brief(_brief("a" * 64, "b" * 64, "c" * 64), store)

    assert out["claims"][0]["original_excerpt"] is None
    assert out["claims"][1]["original_excerpt"] == "VERBATIM SOURCE TEXT 1"
    assert out["claims"][2]["original_excerpt"] == "VERBATIM SOURCE TEXT 2"
    assert counts == {"restricted": 1, "public_domain": 1, "publicly_accessible": 1}


def test_the_source_brief_on_disk_is_never_modified(tmp_path):
    """The unredacted brief stays the internal audit record. Redaction happens on
    a copy, at the publish step."""
    store = _store(tmp_path, {"a" * 64: "restricted"})
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    original = _brief("a" * 64)
    (briefs / "x.yaml").write_text(yaml.safe_dump(original))

    stats = publish_briefs(briefs, tmp_path / "out", store)

    on_disk = yaml.safe_load((briefs / "x.yaml").read_text())
    assert on_disk["claims"][0]["original_excerpt"] == "VERBATIM SOURCE TEXT 0"
    published = yaml.safe_load((tmp_path / "out" / "x.yaml").read_text())
    assert published["claims"][0]["original_excerpt"] is None
    assert stats == {
        "briefs": 1,
        "withheld_claims": 1,
        "by_status": {"restricted": 1},
        "unreadable": [],
    }


def test_restricted_is_not_in_the_publishable_set():
    """A guard against someone widening the set by editing the constant: this is
    the CDN-leak boundary and it needs Mark's sign-off to move."""
    assert "restricted" not in PUBLISHABLE_EXCERPT_STATUSES
    assert "unknown" not in PUBLISHABLE_EXCERPT_STATUSES


def test_a_record_stored_under_v1_still_resolves(tmp_path):
    """store/v1/ holds the record/1-schema files, and seven records live only
    there - Surviving Death, In Plain Sight, Imminent among them. Missing that
    location returned "unknown", which withholds - so the bug's failure mode was
    the SAFE one, and it looked exactly like correct caution while silently
    withholding 1,311 excerpts we are entitled to publish."""
    store = tmp_path / "store"
    (store / "v1").mkdir(parents=True)
    (store / "v1" / f"{'f' * 64}.md").write_text(
        "---\ncopyright:\n  status: public_domain\n---\n\nbody\n"
    )

    assert copyright_status(store, "sha256:" + "f" * 64) == "public_domain"


def test_an_unreadable_brief_is_reported_not_skipped(tmp_path):
    """A brief that cannot be parsed must never vanish quietly.

    Skipping it silently drops an entity from the published record with nothing
    to say why - the shape that made two entities carrying 200 and 183 claims
    unlinkable across the whole corpus with no output explaining it.
    """
    store = _store(tmp_path, {"a" * 64: "public_domain"})
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    (briefs / "good.yaml").write_text(yaml.safe_dump(_brief("a" * 64)))
    (briefs / "broken.yaml").write_text("claims:\n- content: 'unterminated\n")
    (briefs / "notamapping.yaml").write_text("- just\n- a list\n")

    stats = publish_briefs(briefs, tmp_path / "out", store)

    assert stats["briefs"] == 1
    assert len(stats["unreadable"]) == 2
    assert any(u.startswith("broken.yaml") for u in stats["unreadable"])
    assert any("not a mapping" in u for u in stats["unreadable"])


def test_a_source_brief_declares_itself_not_for_publication(tmp_path):
    """The two brief directories are not two copies of the same data.

    The source side carries every excerpt verbatim, including from sources we
    may not redistribute, and it is systematically NEWER than the published one -
    which is exactly what makes "just read the fresher directory" attractive.
    The file says what it is so a consumer can refuse it without having to know
    which directory it came from.
    """
    from assimilator.synthesise import build_entity_brief

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="N", node_type=NodeType.event, name="An Event"))
    brief = build_entity_brief(conn, "N")
    assert brief["publication"]["status"] == "unredacted"
    assert "NOT FOR PUBLICATION" in brief["publication"]["warning"]


def test_publishing_flips_the_marker(tmp_path):
    store = _store(tmp_path, {"a" * 64: "restricted"})
    published, _ = redact_brief(_brief("a" * 64), store)
    assert published["publication"]["status"] == "redacted"


def test_a_published_brief_the_graph_moved_past_is_reported(tmp_path):
    """The published directory was never pruned.

    prune_retired_briefs runs during SYNTHESIS on the source directory, so a
    merge or rename cleans up there and leaves the published copy standing - and
    the assembler takes a brief by slug. Both failure modes are in the live
    corpus: a page built for a retired node, and one entity getting two pages
    because a brief sits at a slug the node no longer has.
    """
    from assimilator.publish_briefs import unbuildable_in

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="LIVE", node_type=NodeType.event, name="Kept Event"))
    out = tmp_path / "published"
    out.mkdir()
    (out / "kept-event.yaml").write_text(
        yaml.safe_dump({"page": {"node_id": "LIVE", "slug": "kept-event"}})
    )
    (out / "gone-event.yaml").write_text(
        yaml.safe_dump({"page": {"node_id": "RETIRED", "slug": "gone-event"}})
    )
    (out / "old-name.yaml").write_text(
        yaml.safe_dump({"page": {"node_id": "LIVE", "slug": "old-name"}})
    )

    found = {f["file"]: f["why"] for f in unbuildable_in(out, conn)}

    assert "kept-event.yaml" not in found
    assert found["gone-event.yaml"] == "node retired or absent"
    assert "stale slug" in found["old-name.yaml"]
