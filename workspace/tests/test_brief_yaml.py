"""A brief must survive the shared commit hook's formatter."""

import yaml

from assimilator.brief_yaml import contains_newline, dump

MULTILINE = (
    "- References Blum's book. Talked about JA, Blum's book, his (Oke's) role "
    "there, who attended, etc.\n- Oke briefed me on whole BDM thing"
)


def _excerpt_lines(text: str) -> list[str]:
    body = [line for line in text.split("\n") if line.strip()]
    start = next(i for i, line in enumerate(body) if "original_excerpt" in line)
    end = start + 1
    while end < len(body) and body[end].startswith("    "):
        end += 1
    return body[start:end]


def test_a_value_with_a_newline_is_one_physical_line():
    """The protection is being on one line, NOT the quoting style.

    yamlfmt with retain_line_breaks replaces a line break inside a string with a
    literal sentinel, and truncates the scalar where it also runs long - losing
    the tail irrecoverably. A single-quoted scalar broken across two lines is
    corrupted exactly like a folded one, so quoting alone is not the fix.
    """
    out = dump({"claims": [{"original_excerpt": MULTILINE}]})
    assert len(_excerpt_lines(out)) == 1
    assert yaml.safe_load(out)["claims"][0]["original_excerpt"] == MULTILINE


def test_the_newline_survives_the_round_trip():
    out = dump({"claims": [{"original_excerpt": MULTILINE}]})
    assert "\\n" in out  # escaped, not a real break
    assert "\n- Oke briefed" not in out


def test_a_brief_without_newlines_keeps_the_readable_form():
    """The unwrapped width costs readability, so it is paid only where it buys
    something - 23 of 748 briefs carry a newline-bearing string."""
    long_text = "word " * 60
    out = dump({"claims": [{"content": long_text.strip()}]})
    assert max(len(line) for line in out.split("\n")) <= 100
    assert '"' not in out


def test_contains_newline_walks_the_whole_structure():
    assert contains_newline({"a": [{"b": "x\ny"}]})
    assert not contains_newline({"a": [{"b": "x y"}], "c": 3, "d": None})


def test_the_publication_vocabulary_is_fixed():
    """A contract with every consumer that reads a brief.

    It changed once already - "redacted" became "published" when the excerpt
    redaction was removed - and a consumer allow-listing the old value found the
    whole corpus unbuildable within the hour. Consumers should DENY
    INTERNAL_ONLY rather than enumerate what is safe: wrongly blocking costs
    every page, wrongly passing costs one brief from a directory nobody
    publishes from.
    """
    from assimilator.brief_yaml import INTERNAL_ONLY, PUBLICATION_STATUSES, PUBLISHED

    assert INTERNAL_ONLY == "unredacted"
    assert PUBLISHED == "published"
    assert PUBLICATION_STATUSES == {"unredacted", "published"}


def test_both_writers_use_the_shared_vocabulary():
    """Neither writer may invent its own string."""
    import sqlite3

    from anomalica_common.digest.models import Node, NodeType

    from assimilator.brief_yaml import INTERNAL_ONLY
    from assimilator.database import init_db, insert_node
    from assimilator.synthesise import build_entity_brief

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    insert_node(conn, Node(id="N", node_type=NodeType.event, name="An Event"))
    assert build_entity_brief(conn, "N")["publication"]["status"] == INTERNAL_ONLY
