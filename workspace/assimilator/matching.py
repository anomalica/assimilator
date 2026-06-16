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


# Minimum normalised Levenshtein similarity (0-1) to consider a fuzzy match.
# 0.75 catches "K. Day" vs "Kevin Day" but not "Kevin Day" vs "David Fravor".
FUZZY_NAME_THRESHOLD = 0.75

# Minimum word overlap ratio to even attempt Levenshtein comparison.
# Avoids comparing completely unrelated names.
MIN_WORD_OVERLAP = 0.3


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

        # Levenshtein similarity on full name
        sim = levenshtein_ratio(name_lower, candidate_lower)
        if sim >= FUZZY_NAME_THRESHOLD and sim > best_score:
            best_match = candidate
            best_score = sim

    if best_match:
        return best_match.id, "fuzzy"

    return None
