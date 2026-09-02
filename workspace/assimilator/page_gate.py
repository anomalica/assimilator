"""The page-worthiness gate: which nodes earn a published page.

Supersedes the flat claim-count floor (page_set.py). A raw count cannot tell
"subject of the corpus" from "mentioned in it" - it produced ~350 junk proposals
(an object cited once, a place mentioned in passing). Four things decide it
(node-types.md, "Page-worthiness"): node TYPE, source INDEPENDENCE, source
SPREAD, and whether the corpus says anything ABOUT the node.

1. Type tier - no type is permanently barred:
   - Page-worthy (modest floor): person, organisation, project, event, topic.
   - High-bar (central-subject only): place, object, document.
   Deprecated taxonomy types (matter, concept, programme, investigation) and the
   curator-only `pattern` are not page-worthy by extraction; they are gated out
   until re-digestion reclassifies them into the canonical eight.

2. The floor is distinct sources, not raw claim count. A "source" is a distinct
   WORK (record, folded across re-ingests of one book); true provenance-root
   independence (ADR 0039) is carried as a separate proposal column.

3. Spread: the second-best work must contribute at least MIN_SECOND_SOURCE_CLAIMS
   claims. A count of sources cannot tell "ten claims from two books" from "a
   book plus a passing mention", and the second shape - a page that is in
   substance a summary of one copyrighted work - was 53% of the proposal set
   when measured (2026-07-29) and the whole of it under six claims.

4. Subject: for a person, organisation or object, at least MIN_SUBJECT_CLAIMS
   claims must be ABOUT the node (it is the grammatical subject) rather than
   merely mention it. This is what separates a subject from an attribute -
   "Blink-182" was proposed at 13 claims from 4 sources without one claim about
   the band; every claim was about its guitarist. Places, events, topics,
   projects and documents are reported on but not gated: claims about them are
   phrased around them ("during the encounter", "over Socorro") and the test
   would reject the central ones.

5. A person with no family name (matching.is_record_scoped_person_name) is never
   a page: "Chris" is not a subject anyone can look up.

Host-light (stdlib only) so the synthesiser and the scheduler share it without
heavier deps - the same discipline as page_set.py, which this supersedes.
"""

from __future__ import annotations

import os
import re
import sqlite3

from assimilator.matching import (
    _HONORIFICS,
    is_record_scoped_person_name,
    person_name_tokens,
    strip_acronym_suffix,
)

PAGE_WORTHY = "page-worthy"
HIGH_BAR = "high-bar"

TIER: dict[str, str] = {
    "person": PAGE_WORTHY,
    "organisation": PAGE_WORTHY,
    "project": PAGE_WORTHY,
    "event": PAGE_WORTHY,
    "topic": PAGE_WORTHY,
    "place": HIGH_BAR,
    "object": HIGH_BAR,
    "document": HIGH_BAR,
}

DEFAULT_FLOORS: dict[str, tuple[int, int]] = {
    PAGE_WORTHY: (8, 3),  # >= 8 claims from >= 3 distinct works
    HIGH_BAR: (12, 4),  # >= 12 claims from >= 4 distinct works
}

# The second-best work must carry this many claims (both tiers).
DEFAULT_MIN_SECOND_SOURCE_CLAIMS = 3

# This many claims must be ABOUT the node, for the types the test applies to.
DEFAULT_MIN_SUBJECT_CLAIMS = 3
SUBJECT_GATED_TYPES = frozenset({"person", "organisation", "object"})


def floors() -> dict[str, tuple[int, int]]:
    """Per-tier (min_claims, min_sources), env-overridable for calibration.

    ANOMALICA_PAGE_WORTHY_MIN_CLAIMS / _MIN_SOURCES and the HIGH_BAR equivalents.
    """

    def _f(tier: str, default: tuple[int, int]) -> tuple[int, int]:
        prefix = "ANOMALICA_" + tier.replace("-", "_").upper()
        dc, ds = default
        return (
            int(os.environ.get(prefix + "_MIN_CLAIMS", dc)),
            int(os.environ.get(prefix + "_MIN_SOURCES", ds)),
        )

    return {tier: _f(tier, default) for tier, default in DEFAULT_FLOORS.items()}


def min_second_source_claims() -> int:
    return int(
        os.environ.get(
            "ANOMALICA_PAGE_MIN_SECOND_SOURCE_CLAIMS", DEFAULT_MIN_SECOND_SOURCE_CLAIMS
        )
    )


def min_subject_claims() -> int:
    return int(
        os.environ.get("ANOMALICA_PAGE_MIN_SUBJECT_CLAIMS", DEFAULT_MIN_SUBJECT_CLAIMS)
    )


_NODE_CLAIMS_SQL = """
    SELECT speaker_id AS nid, id AS cid, record_id AS rid
      FROM claims WHERE speaker_id IS NOT NULL
    UNION
    SELECT cnr.node_id AS nid, c.id AS cid, c.record_id AS rid
      FROM claim_node_refs cnr JOIN claims c ON c.id = cnr.claim_id
"""


def _node_counts(conn: sqlite3.Connection) -> list[tuple[str, str, str, int, int]]:
    """(node_id, node_type, name, claim_count, source_count) for every active
    node a claim references (as speaker or node-ref).

    Sources are distinct WORKS, not distinct records. One work becomes several
    records on any re-ingest or edition change, and counting records would let a
    book present twice clear a two-source floor on its own. work_id defaults to
    the record's own id, so this is identical to a record count until a duplicate
    is actually linked.
    """
    return conn.execute(
        f"""
        SELECT n.id, n.node_type, n.name,
               COUNT(DISTINCT x.cid) AS claims,
               COUNT(DISTINCT COALESCE(r.work_id, x.rid)) AS sources
          FROM nodes n
          JOIN ({_NODE_CLAIMS_SQL}) x ON x.nid = n.id
          LEFT JOIN records r ON r.id = x.rid
         WHERE n.retired_at IS NULL
         GROUP BY n.id
        """
    ).fetchall()


def _source_spread(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """(top_source_claims, second_source_claims) per node.

    A distinct-source COUNT cannot tell "10 claims from two sources" from "10
    claims from one book plus one passing mention" - both report 2. The count of
    sources says nothing about how the claims are spread across them, and it is
    the spread that decides whether a page is genuinely corroborated or a summary
    of a single (often copyrighted) work with a second source attached.

    Grouped by WORK, so two records of one book contribute one figure between
    them rather than a flattering pair - without that this metric inverts under
    duplication, blessing exactly the pages with the worst provenance.
    """
    rows = conn.execute(
        f"""
        SELECT x.nid, COALESCE(r.work_id, x.rid) AS work,
               COUNT(DISTINCT x.cid) AS claims
          FROM ({_NODE_CLAIMS_SQL}) x
          LEFT JOIN records r ON r.id = x.rid
         GROUP BY x.nid, work ORDER BY x.nid, claims DESC
        """
    ).fetchall()
    spread: dict[str, list[int]] = {}
    for node_id, _work_id, claims in rows:
        spread.setdefault(node_id, []).append(claims)
    return {
        node_id: (counts[0], counts[1] if len(counts) > 1 else 0)
        for node_id, counts in spread.items()
    }


# --- Subject claims: is the corpus saying anything ABOUT the node? ---
#
# Claim text is written with the actor as grammatical subject and no reporting
# verb (node-types.md, "Assertion, not reported speech"): "David Fravor observed
# ...", "Robert Bigelow bought ...". So a claim that opens with the node's name
# is a claim about it, and one that reaches the name mid-sentence is a mention.
# The opening may carry an article or a rank - "The Central Intelligence Agency
# ...", "General Ramey ordered ..." - and a person is as often written by
# surname alone as in full. One leading adverbial clause is allowed before the
# subject ("At midday on 8 July 1947, Colonel William Blanchard ordered ..."),
# which is how a dated claim is written; it must end at a comma, so "According
# to Hal Puthoff and Russell Targ's observations, remote viewing subjects ..."
# still reads as a claim about remote viewing, not about Targ.

_ARTICLES = ("the", "a", "an")
_LEADING_CLAUSE = r"(?:[^,.;]{0,80},\s+)?"
_ACRONYM_TAIL_RE = re.compile(r"\(([A-Za-z0-9./-]{2,10})\)\s*$")


def _subject_forms(name: str, node_type: str, aliases: list[str]) -> set[str]:
    forms = {name}
    forms.update(a for a in aliases if a)
    bare = strip_acronym_suffix(name)
    forms.add(bare)
    m = _ACRONYM_TAIL_RE.search(name)
    if m:
        forms.add(m.group(1))
    # "USA, New Mexico, Roswell" is written "Roswell" in a claim.
    if "," in bare:
        forms.add(bare.rsplit(",", 1)[-1].strip())
    if node_type == "person":
        tokens = person_name_tokens(name)
        if len(tokens) >= 2 and len(tokens[-1]) >= 3:
            forms.add(tokens[-1])
    return {f.strip() for f in forms if len(f.strip()) >= 2}


def _subject_pattern(forms: set[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    lead = r"(?:(?:%s)\s+)?(?:(?:%s)\.?\s+){0,2}" % (
        "|".join(_ARTICLES),
        "|".join(sorted(_HONORIFICS)),
    )
    return re.compile(
        rf"^\W*{_LEADING_CLAUSE}{lead}(?:{alternatives})(?![\w-])", re.IGNORECASE
    )


def _subject_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """node_id -> number of the node's claims that are ABOUT it."""
    aliases: dict[str, list[str]] = {}
    for node_id, alias in conn.execute("SELECT node_id, alias FROM aliases"):
        aliases.setdefault(node_id, []).append(alias)
    patterns: dict[str, re.Pattern[str]] = {}
    for node_id, node_type, name in conn.execute(
        "SELECT id, node_type, name FROM nodes WHERE retired_at IS NULL"
    ):
        forms = _subject_forms(name, node_type, aliases.get(node_id, []))
        if forms:
            patterns[node_id] = _subject_pattern(forms)
    counts: dict[str, int] = {}
    for node_id, content in conn.execute(
        f"""
        SELECT DISTINCT x.nid, c.content
          FROM ({_NODE_CLAIMS_SQL}) x JOIN claims c ON c.id = x.cid
        """
    ):
        pattern = patterns.get(node_id)
        if pattern and pattern.match(content or ""):
            counts[node_id] = counts.get(node_id, 0) + 1
    return counts


def page_gate_rows(conn: sqlite3.Connection) -> list[dict]:
    """Nodes that pass the page-worthiness gate, with their tier and counts.

    Each row: {node_id, node_type, tier, claim_count, source_count,
    top_source_claims, second_source_claims, subject_claims}. A node whose type
    is not in TIER (deprecated / curator-only) never passes. Ordered by
    claim_count descending then node_id - a stable, review-friendly ranking with
    the strongest subjects first.
    """
    tier_floors = floors()
    min_second = min_second_source_claims()
    min_subject = min_subject_claims()
    spread = _source_spread(conn)
    subjects = _subject_counts(conn)
    out: list[dict] = []
    for node_id, node_type, name, claims, sources in _node_counts(conn):
        tier = TIER.get(node_type)
        if tier is None:
            continue
        min_claims, min_sources = tier_floors[tier]
        if claims < min_claims or sources < min_sources:
            continue
        top, second = spread.get(node_id, (claims, 0))
        if second < min_second:
            continue
        subject = subjects.get(node_id, 0)
        if node_type in SUBJECT_GATED_TYPES and subject < min_subject:
            continue
        if node_type == "person" and is_record_scoped_person_name(name):
            continue
        out.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "tier": tier,
                "claim_count": claims,
                "source_count": sources,
                "top_source_claims": top,
                "second_source_claims": second,
                "subject_claims": subject,
            }
        )
    out.sort(key=lambda r: (-r["claim_count"], r["node_id"]))
    return out
