"""Node matching for entity resolution.

Combines three strategies in order of preference:
1. Exact name and alias lookup (instant, perfect precision)
2. Acronym-suffix normalisation (catches X <-> X (ACRONYM) pairs)
3. Levenshtein distance on names (catches typos, abbreviations, minor variants)
4. Claim-based embedding similarity (catches different names for the same thing)
"""

from __future__ import annotations

import re
import sqlite3

from Levenshtein import ratio as levenshtein_ratio

from assimilator.database import find_node_by_name, get_nodes


# Trailing parenthetical-acronym suffix: "Defense Intelligence Agency (DIA)".
# Restricted to ALL-CAPS plus digits/hyphens of length >= 2 to avoid matching
# "Smith (mother)" or other non-acronym parens content.
_ACRONYM_SUFFIX_RE = re.compile(r"\s*\(([A-Z0-9][A-Z0-9-]{1,}[A-Z0-9])\)\s*$")


# Known acronym expansions the extraction model fails to apply consistently.
# Applied deterministically at import time and at workbench render time so the
# graph and the workbench Digest column always show the expanded form, even
# when the model emits the bare acronym. Source: the navy aviation prefix
# glossary plus a handful of UAP-domain programme names.
_SQUADRON_PREFIXES = {
    "VFA": "Strike Fighter Squadron",
    "VMFA": "Marine Fighter Attack Squadron",
    "VAQ": "Electronic Attack Squadron",
    "VAW": "Carrier Airborne Early Warning Squadron",
    "VRC": "Fleet Logistics Support Squadron",
    "HS": "Helicopter Anti-Submarine Squadron",
    "CSG": "Carrier Strike Group",
    "CVW": "Carrier Air Wing",
}
# Compile one regex matching any of the prefixes followed by -N (with optional
# detachment suffix like " Det 3"). We rewrite the WHOLE match to the
# expanded "Full Name N (PREFIX-N)" form.
_SQUADRON_RE = re.compile(
    r"\b(" + "|".join(sorted(_SQUADRON_PREFIXES, key=len, reverse=True)) + r")-(\d+)\b"
)

_PROGRAMME_EXPANSIONS = {
    "AATIP": "Advanced Aerospace Threat Identification Program",
    "AAWSAP": "Advanced Aerospace Weapon System Applications Program",
    "AARO": "All-Domain Anomaly Resolution Office",
}


def _expand_squadron(match: "re.Match[str]") -> str:
    prefix, number = match.group(1), match.group(2)
    full = _SQUADRON_PREFIXES[prefix]
    return f"{full} {number} ({prefix}-{number})"


def normalise_node_name(name: str) -> str:
    """Apply deterministic acronym expansions the extraction model misses.

    Conservative: only rewrites a bare prefix-number squadron designator OR a
    bare programme acronym from the known list. Names that already include the
    expanded form ("Strike Fighter Squadron 41 (VFA-41)") are left untouched
    because the regex matches the bare form and would produce duplicates.

    >>> normalise_node_name("VFA-41")
    'Strike Fighter Squadron 41 (VFA-41)'
    >>> normalise_node_name("CSG-11 AAV MISREP November 2004")
    'Carrier Strike Group 11 (CSG-11) AAV MISREP November 2004'
    >>> normalise_node_name("Strike Fighter Squadron 41 (VFA-41)")
    'Strike Fighter Squadron 41 (VFA-41)'
    """
    # Skip names that already contain the expanded form - the existing parens
    # acronym tail means the model did the work.
    out = name
    for prefix, full in _SQUADRON_PREFIXES.items():
        if full in out:
            return out
    for acro, full in _PROGRAMME_EXPANSIONS.items():
        if full in out:
            return out
    # Rewrite each prefix-N occurrence with the expanded form. The lookup
    # uses _SQUADRON_RE to match the bare designator.
    out = _SQUADRON_RE.sub(_expand_squadron, out)
    # Programme acronyms: replace whole-word ACRONYM with "Full Name (ACRONYM)".
    for acro, full in _PROGRAMME_EXPANSIONS.items():
        out = re.sub(rf"\b{acro}\b", f"{full} ({acro})", out)
    return out


def strip_acronym_suffix(name: str) -> str:
    """Strip a trailing '(ACRONYM)' suffix if present.

    >>> strip_acronym_suffix('Defense Intelligence Agency (DIA)')
    'Defense Intelligence Agency'
    >>> strip_acronym_suffix('Joe (mother)')
    'Joe (mother)'
    """
    return _ACRONYM_SUFFIX_RE.sub("", name).rstrip()


def name_equivalence_key(name: str) -> str:
    """Lowercase, acronym-suffix-stripped form used for matching equivalent names."""
    return strip_acronym_suffix(name).lower().strip()


# Minimum similarity (0-1) to consider a fuzzy match. 0.75 catches typos
# ("David Fravor" vs "David Favor") and initial-vs-full first names ("K. Day"
# vs "Kevin Day") but not unrelated names ("Kevin Day" vs "David Fravor").
FUZZY_NAME_THRESHOLD = 0.75

# Per-component threshold for comma-structured names ("Surname, First" people,
# "Country, State, City" places). Higher than the whole-name threshold because a
# single distinguishing component must clear it on its own - this is what stops
# "Hill, Barney" merging with "Hill, Betty" or "USA, Nevada, Area 51" with
# "USA, Nevada, Las Vegas", where the shared structure would otherwise inflate a
# whole-string ratio above 0.75.
STRUCTURED_COMPONENT_THRESHOLD = 0.80

# Minimum word overlap ratio to even attempt Levenshtein comparison.
# Avoids comparing completely unrelated names.
MIN_WORD_OVERLAP = 0.3

# A non-hard token on one side that has no close counterpart on the other counts
# as a counterpart when its Levenshtein ratio to some other-side token clears
# this. Set just above "yorker" vs "york" (0.80) so "The New Yorker" and "The
# New York Times" stay distinct, while spelling variants like "centre" vs
# "center" (0.83) and "colour" vs "color" still pair up.
_TOKEN_COUNTERPART_THRESHOLD = 0.81

# Structural words that carry no distinguishing signal, so they are ignored when
# deciding whether two names disagree on a distinctive token. Includes the
# domain's ubiquitous phenomenon acronyms.
_STOPWORDS = frozenset(
    {
        "of",
        "the",
        "and",
        "for",
        "on",
        "in",
        "at",
        "to",
        "a",
        "an",
        "de",
        "us",
        "uap",
        "ufo",
    }
)

# Drops a trailing/embedded parenthetical group ("(dod)", "(saps)", "(ousd(i))")
# the all-caps acronym-suffix regex misses because of mixed case. The expansion
# words already carry the signal, so the acronym tail is noise for token compare.
_PARENS_GROUP_RE = re.compile(r"\([^)]*\)")
_TOKEN_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _name_tokens(name: str) -> list[str]:
    """Lowercased word tokens with acronym tails and edge punctuation removed."""
    base = _PARENS_GROUP_RE.sub(" ", strip_acronym_suffix(name).lower())
    tokens = []
    for raw in re.split(r"[\s/]+", base):
        token = _TOKEN_PUNCT_RE.sub("", raw)
        if token:
            tokens.append(token)
    return tokens


def _is_hard_token(token: str) -> bool:
    """A token carrying a digit: a number, year or alphanumeric designator.

    "1952", "fy2023", "12333", "vfa-41", "mh370", "apg-79", "j2", "e-2d",
    "s1632". A differing hard token is the single strongest signal that two
    otherwise-similar names are different entities.
    """
    return any(c.isdigit() for c in token)


def _hard_tokens(name: str) -> set[str]:
    return {t for t in _name_tokens(name) if _is_hard_token(t)}


_ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}){1,2}$")


def _hard_token_has_counterpart(token: str, others: set[str]) -> bool:
    """True if a hard token is matched, allowing year-vs-ISO-date compatibility.

    A bare year is the date-prefix of a fuller ISO date for the SAME event, so
    "2004" matches "2004-11-14" and "2016" matches "2016-03". This keeps
    "Nimitz UAP Intercept, 2004-11-14" merged with "2004 Nimitz UAP Encounter"
    while still splitting genuinely different designators ("fy2023" is not a
    prefix of "fy2024"; "1952" is not a prefix of "1954").
    """
    if token in others:
        return True
    for other in others:
        short, long = sorted((token, other), key=len)
        if _ISO_DATE_RE.match(long) and long.startswith(short + "-"):
            return True
    return False


def _hard_tokens_conflict(a_hard: set[str], b_hard: set[str]) -> bool:
    """True if both names carry hard tokens that cannot be reconciled.

    Only fires when both sides have hard tokens: one side lacking them is a
    name-extension, not a substitution.
    """
    if not a_hard or not b_hard:
        return False
    if any(not _hard_token_has_counterpart(t, b_hard) for t in a_hard):
        return True
    if any(not _hard_token_has_counterpart(t, a_hard) for t in b_hard):
        return True
    return False


def _orphan_tokens(src: list[str], other: list[str]) -> list[str]:
    """Distinctive (non-hard, non-stop) src tokens with no counterpart in other.

    A token counts as having a counterpart if it appears verbatim in `other`, is
    an initial of some other token ("k." vs "kevin"), or is a near-spelling of
    one (>= _TOKEN_COUNTERPART_THRESHOLD, e.g. "centre" vs "center").
    """
    orphans = []
    for token in src:
        if token in _STOPWORDS or _is_hard_token(token):
            continue
        if token in other:
            continue
        if any(_initials_compatible(token, candidate) for candidate in other):
            continue
        if any(
            levenshtein_ratio(token, candidate) >= _TOKEN_COUNTERPART_THRESHOLD
            for candidate in other
        ):
            continue
        orphans.append(token)
    return orphans


def _distinctive_tokens_disagree(a: str, b: str) -> bool:
    """True if the two names disagree on a distinguishing token.

    Two mechanisms separate a genuine merge from a false one:

    - Hard tokens (numbers, years, designators) must match exactly when both
      names carry them. "FY2024 ..." vs "FY2023 ...", "Executive Order 12333"
      vs "13526", "VFA-14" vs "VFA-41" differ on a hard token and are distinct.
      A name with no hard token at all is a name-extension ("Filming" vs
      "Filming 1964"), so the rule only fires when both sides carry hard tokens.

    - Distinctive (rare/proper-noun) words. A false merge SUBSTITUTES one
      distinctive word for another, so each name holds a word the other lacks
      ("Andrews" vs "Vandenberg", "Cardiff" vs "Stanford", "central" vs
      "defense"). A genuine extension only ADDS words to one side ("ICBM
      Filming" vs "Atlas ICBM Filming"), leaving an orphan on a single side.
      Disagreement is therefore mutual orphans: both names have a distinctive
      word with no counterpart in the other.
    """
    if _hard_tokens_conflict(_hard_tokens(a), _hard_tokens(b)):
        return True
    ta, tb = _name_tokens(a), _name_tokens(b)
    return bool(_orphan_tokens(ta, tb) and _orphan_tokens(tb, ta))


def _initials_compatible(a: str, b: str) -> bool:
    """True if one token is an initial of the other ("K." or "K" vs "Kevin").

    Preserves the initial-vs-full first-name case so "K. Day" still matches
    "Kevin Day" once structured names are compared component- and token-wise.
    """
    a0, b0 = a.rstrip("."), b.rstrip(".")
    if not a0 or not b0:
        return False
    if len(a0) == 1 and b0.startswith(a0):
        return True
    if len(b0) == 1 and a0.startswith(b0):
        return True
    return False


def _token_aware_ratio(a: str, b: str) -> float:
    """Position-aligned token similarity that treats initials as full matches.

    Aligns tokens by position; a position where one token is an initial of the
    other scores 1.0, otherwise the Levenshtein ratio of the two tokens. Surplus
    tokens on the longer side score 0. Used both inside structured components and
    as a recall boost on plain (non-comma) names.
    """
    ta, tb = a.split(), b.split()
    count = max(len(ta), len(tb))
    if count == 0:
        return 1.0
    total = 0.0
    for i in range(count):
        x = ta[i] if i < len(ta) else ""
        y = tb[i] if i < len(tb) else ""
        if x and y and _initials_compatible(x, y):
            total += 1.0
        else:
            total += levenshtein_ratio(x, y)
    return total / count


def _component_similarity(a: str, b: str) -> float:
    """Similarity of one comma-component, initial-aware on its tokens."""
    if a == b:
        return 1.0
    return max(levenshtein_ratio(a, b), _token_aware_ratio(a, b))


def _comma_components(name: str) -> list[str]:
    return [part.strip() for part in name.split(",") if part.strip()]


def fuzzy_name_similarity(a: str, b: str) -> float:
    """Structure-aware similarity (0-1) between two lowercased names.

    Full-string Levenshtein is the wrong metric for structured names: it scores
    on the shared structure, so "Surname, First" people who share a first name
    and hierarchical "Country, State, City" places that share a prefix both
    score spuriously high. When both names are comma-structured this compares
    them component-wise instead:

    - equal depth ("Hill, Barney" vs "Hill, Betty"): every component must be
      compatible, so the decision is the *minimum* per-component similarity. A
      single differing component (the first name, or the surname, or the city)
      drags the score down and blocks the merge.
    - different depth: only the most-specific (last) component is compared, so a
      "City" can still match a "State, City" of the same place.

    Structured comparisons hold every compared component to the stricter
    STRUCTURED_COMPONENT_THRESHOLD: if any required component falls below it the
    similarity collapses to 0, so a near-miss component (e.g. "Los Alamos" vs
    "Los Brasos") can never be rescued by the looser whole-name gate.

    Plain names fall back to whole-string Levenshtein, lifted by an initial-aware
    token pass so "K. Day" still matches "Kevin Day" - but first gated by a
    distinctive-token check so a pair that shares its common words yet differs on
    a number, year, designator or proper noun ("FY2024 ..." vs "FY2023 ...",
    "Andrews Air Force Base" vs "Vandenberg Air Force Base") cannot merge on the
    shared structure alone.
    """
    # A differing hard token (number, year, designator) means different entities
    # whatever the name shape, so guard both branches with it.
    if _hard_tokens_conflict(_hard_tokens(a), _hard_tokens(b)):
        return 0.0
    ca, cb = _comma_components(a), _comma_components(b)
    if len(ca) >= 2 and len(cb) >= 2:
        if len(ca) == len(cb):
            scores = [_component_similarity(ca[i], cb[i]) for i in range(len(ca))]
        else:
            scores = [_component_similarity(ca[-1], cb[-1])]
        weakest = min(scores)
        return weakest if weakest >= STRUCTURED_COMPONENT_THRESHOLD else 0.0
    if _distinctive_tokens_disagree(a, b):
        return 0.0
    return max(levenshtein_ratio(a, b), _token_aware_ratio(a, b))


def match_node(
    conn: sqlite3.Connection,
    name: str,
    node_type: str | None = None,
) -> tuple[str, str] | None:
    """Try to match a name to an existing node.

    Returns (node_id, match_method) or None if no match found.
    match_method is one of: "exact", "alias", "acronym", "fuzzy".
    """
    # 1. Exact name match
    exact = find_node_by_name(conn, name, node_type)
    if exact:
        return exact.id, "exact"

    # 2. Acronym-suffix normalisation: X <-> X (ACRONYM) collapse to the same
    # node. Avoids duplicate organisation/concept nodes for "Defense
    # Intelligence Agency" and "Defense Intelligence Agency (DIA)".
    candidates_all = get_nodes(conn, node_type=node_type)
    key = name_equivalence_key(name)
    for candidate in candidates_all:
        if name_equivalence_key(candidate.name) == key and candidate.name != name:
            return candidate.id, "acronym"

    # 3. Fuzzy name match via Levenshtein
    candidates = candidates_all
    best_match = None
    best_score = 0.0

    name_lower = name.lower()
    name_words = set(name_lower.split())

    for candidate in candidates:
        candidate_lower = candidate.name.lower()
        candidate_words = set(candidate_lower.split())

        # Quick word overlap filter to avoid expensive comparisons
        if name_words and candidate_words:
            overlap = len(name_words & candidate_words)
            total = max(len(name_words), len(candidate_words))
            if overlap / total < MIN_WORD_OVERLAP:
                # Also check aliases for this candidate
                aliases = conn.execute(
                    "SELECT alias FROM aliases WHERE node_id = ?", (candidate.id,)
                ).fetchall()
                alias_match = False
                for (alias,) in aliases:
                    alias_sim = levenshtein_ratio(name_lower, alias.lower())
                    if alias_sim >= FUZZY_NAME_THRESHOLD and alias_sim > best_score:
                        best_match = candidate
                        best_score = alias_sim
                        alias_match = True
                if not alias_match:
                    continue

        # Structure-aware similarity on the full name. For comma-structured
        # names this compares the distinguishing components rather than the
        # shared structure, which is what stops false merges between distinct
        # people who share a surname/first name and distinct places that share a
        # hierarchical prefix.
        sim = fuzzy_name_similarity(name_lower, candidate_lower)
        if sim >= FUZZY_NAME_THRESHOLD and sim > best_score:
            best_match = candidate
            best_score = sim

    if best_match:
        return best_match.id, "fuzzy"

    return None
