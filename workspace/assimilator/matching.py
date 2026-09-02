"""Node matching for entity resolution.

Combines three strategies in order of preference:
1. Exact name and alias lookup (instant, perfect precision)
2. Acronym-suffix normalisation (catches X <-> X (ACRONYM) pairs)
3. Levenshtein distance on names (catches typos, abbreviations, minor variants)
4. Claim-based embedding similarity (catches different names for the same thing)
"""

from __future__ import annotations

import functools
import re
import sqlite3
import unicodedata

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


# The country component of a place name, in the two forms the corpus uses. NOT a
# heuristic and deliberately tiny: a closed, factual list of alternative names for
# the same country, applied ONLY to the first comma-component of a PLACE. It is a
# list rather than a rule because no rule derives "UK" from "United Kingdom" on
# evidence the name itself carries - is_bare_acronym_for needs a DECLARED
# acronym, and a place name never declares one.
#
# The convention is measured, not chosen: over the live graph USA leads United
# States 463 to 6 and UK leads United Kingdom 29 to 14, so the short form wins.
# Without this the two spellings score 0.0 against each other - the components
# "united kingdom" and "uk" are mutual orphans - and every import mints a
# duplicate. That is not hypothetical: re-pointing claims one morning I created
# "UK, England, Brighton" and "UK, England, Salisbury" beside the "United
# Kingdom, England, ..." nodes that already existed.
_COUNTRY_FORMS = {
    "united kingdom": "UK",
    "great britain": "UK",
    "united states": "USA",
    "united states of america": "USA",
}


def canonical_place_country(name: str) -> str:
    """Rewrite the country component of a place name to the short form.

    Covers the bare country too: a digest emitting "United Kingdom" as a place
    would otherwise mint a duplicate of the existing "UK" node, which is the
    same fault one level up.
    """
    head, _, rest = name.partition(",")
    canonical = _COUNTRY_FORMS.get(head.strip().lower())
    if not canonical:
        return name
    return f"{canonical},{rest}" if rest else canonical


def _sub_outside_parens(pattern, repl, text: str) -> str:
    """Regex substitution that skips matches inside a parenthetical."""
    depth, result, i = 0, [], 0
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if depth == 0:
            match = pattern.match(text, i)
            if match:
                result.append(repl(match))
                i = match.end()
                continue
        result.append(char)
        i += 1
    return "".join(result)


def normalise_node_name(name: str, node_type: str | None = None) -> str:
    """Apply deterministic acronym expansions the extraction model misses.

    NEVER APPLIED TO A PERSON. These expansions name programmes and units, and a
    person's name is neither - but the substitution is whole-word and positional,
    so it fires on any name that happens to contain one. "UAP Gerb" is the handle
    of a real UAP researcher; it was stored as "Unidentified Aerial Phenomena
    (UAP) Gerb", reached the page gate as page-worthy with 36 claims and 8
    independent sources, and was read downstream as a corrupted merge of two
    people. A name is the one field where an expansion is never a clarification.

    Otherwise conservative: it rewrites a prefix-number squadron designator or a
    programme acronym from the known list. Names that already carry a trailing
    "(ACRO)" are left alone, whatever wording precedes it, because that tail is
    the evidence the model already expanded it.

    >>> normalise_node_name("VFA-41")
    'Strike Fighter Squadron 41 (VFA-41)'
    >>> normalise_node_name("CSG-11 AAV MISREP November 2004")
    'Carrier Strike Group 11 (CSG-11) AAV MISREP November 2004'
    >>> normalise_node_name("Strike Fighter Squadron 41 (VFA-41)")
    'Strike Fighter Squadron 41 (VFA-41)'
    >>> normalise_node_name("UAP Gerb", "person")
    'UAP Gerb'
    """
    if node_type == "person":
        return name
    if node_type == "place":
        # Country-form only, and only for places: "United Kingdom Ministry of
        # Defence" is an organisation whose name is not a place hierarchy.
        name = canonical_place_country(name)
    # Skip names that already contain the expanded form - the existing parens
    # acronym tail means the model did the work.
    #
    # "Already expanded" cannot be an exact match against OUR wording. The model
    # writes the programme's name as its source wrote it, and the variants differ:
    # "Advanced Aerospace Weapons Systems Applications Program (AAWSAP)" is the
    # same programme as the singular form in _PROGRAMME_EXPANSIONS, but an exact
    # substring test misses it, expands the bare acronym inside the parenthetical
    # the model already wrote, and a second expander downstream then does it again.
    # The corpus holds the result: one node named "...Program (...Program
    # (...Program (AAWSAP)))" plus two alias rows of the same nesting.
    #
    # A trailing "(ACRO)" IS the evidence of expansion, whatever wording precedes
    # it, so that is what to test.
    out = name
    for prefix, full in _SQUADRON_PREFIXES.items():
        if full in out:
            return out
    for acro, full in _PROGRAMME_EXPANSIONS.items():
        if full in out or f"({acro})" in out:
            return out
    # Rewrite each prefix-N occurrence with the expanded form, but NEVER one
    # already inside a parenthetical - there the designator is the short form
    # the brackets exist to gloss, and expanding it nests one expansion inside
    # another. "Helicopter Antisubmarine Squadron 6 (HS-6)" became "Helicopter
    # Antisubmarine Squadron 6 (Helicopter Anti-Submarine Squadron 6 (HS-6))".
    # The substring guard above cannot catch it: the source wrote
    # "Antisubmarine" and our table says "Anti-Submarine", which is exactly the
    # variant-wording miss the comment above describes - the guard was added for
    # the programme acronyms and never extended to the squadron prefixes.
    out = _sub_outside_parens(_SQUADRON_RE, _expand_squadron, out)
    # Programme acronyms: replace whole-word ACRONYM with "Full Name (ACRONYM)".
    for acro, full in _PROGRAMME_EXPANSIONS.items():
        out = re.sub(rf"\b{acro}\b", f"{full} ({acro})", out)
    return out


_DESCRIPTION_RE = re.compile(r"^\s*\[[^\[\]]+\]\s*$")


def is_a_description(name: str) -> bool:
    """Whether a value is a DESCRIPTION of somebody rather than their name.

    Square brackets around the whole value are the marker (ingest-format.md,
    "Square brackets mean 'this is a description, not a name'"): `[interviewer 2]`,
    `[senior US intelligence officer]`, `[redacted]`. It is record-scoped - the
    `[interviewer 2]` in one recording is not the one in another - so a description
    must never become a node, or two unrelated people accumulate one biography.

    The brackets must wrap the WHOLE value. "Sally (Budd Hopkins abductee)" and
    "Dr. X (French physician)" are names with a qualifier attached: around twenty
    real people in the corpus are written that way, and the qualifier is what
    tells two Sallys apart.

    >>> is_a_description("[senior US intelligence officer]")
    True
    >>> is_a_description("Sally (Budd Hopkins abductee)")
    False
    """
    return bool(_DESCRIPTION_RE.match(name or ""))


# A TWO-character trailing parenthetical. Deliberately not folded into
# _ACRONYM_SUFFIX_RE, which requires three: at two characters a parenthetical is
# as often a QUALIFIER as an acronym - "UFO magazine (UK)" is a country and
# "George Russell (AE)" a pen name, and stripping either would make two distinct
# things equivalent. So a two-character suffix is stripped only where the
# letters are the initials of the words in front of it, which is evidence rather
# than a length rule: "Artificial intelligence (AI)" and "Remote viewing (RV)"
# pass, "UFO magazine (UK)" does not.
_SHORT_ACRONYM_SUFFIX_RE = re.compile(r"\s*\(([A-Z0-9]{2})\)\s*$")


def strip_acronym_suffix(name: str) -> str:
    """Strip a trailing '(ACRONYM)' suffix if present.

    >>> strip_acronym_suffix('Defense Intelligence Agency (DIA)')
    'Defense Intelligence Agency'
    >>> strip_acronym_suffix('Joe (mother)')
    'Joe (mother)'
    >>> strip_acronym_suffix('Artificial intelligence (AI)')
    'Artificial intelligence'
    >>> strip_acronym_suffix('UFO magazine (UK)')
    'UFO magazine (UK)'
    """
    stripped = _ACRONYM_SUFFIX_RE.sub("", name).rstrip()
    if stripped != name:
        return stripped
    match = _SHORT_ACRONYM_SUFFIX_RE.search(name)
    if match and collapse_acronym_expansions(name) != name:
        return name[: match.start()].rstrip()
    return name


def fold_diacritics(text: str) -> str:
    """Drop combining marks from ASCII-Latin base letters, leaving other scripts alone.

    "André Almond" and "Andre Almond" are one person written two ways, and no
    other rule catches them: the fuzzy path sees a substituted distinctive token
    (the accent makes "andré" an orphan of "andre") and rejects the pair. Folding
    only over ASCII bases is deliberate - blanket NFD-strip would turn Japanese
    ガ into カ, merging distinct names.

    >>> fold_diacritics('André Almond')
    'Andre Almond'
    >>> fold_diacritics('ガガーリン')
    'ガガーリン'
    """
    folded: list[str] = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char) and folded and folded[-1].isascii():
            continue
        folded.append(char)
    return unicodedata.normalize("NFC", "".join(folded))


# Internal punctuation that separates words rather than distinguishing entities:
# "KLAS-TV" and "KLAS TV" are one broadcaster, "Paris-Match" and "Paris Match"
# one magazine. Folded to a space (not deleted) so "E-2" and "E 2" agree while
# "E2Hawkeye" - a genuinely different string - does not collapse into them.
_WORD_PUNCT_RE = re.compile(r"[-_.]+")


def name_equivalence_key(name: str) -> str:
    """Lowercase, diacritic- and punctuation-folded, acronym-suffix-stripped form.

    The key for "these are the same name written differently". Everything else
    about the two names must still match exactly, which is what keeps the fold
    safe: measured over the live graph it unifies exactly two pairs, both correct
    (KLAS-TV/KLAS TV, Paris-Match/Paris Match), and merges nothing else.
    """
    folded = fold_diacritics(strip_acronym_suffix(name)).lower()
    return " ".join(_WORD_PUNCT_RE.sub(" ", folded).split())


# Minimum similarity (0-1) to consider a fuzzy match. 0.75 catches typos
# ("David Fravor" vs "David Favor") and initial-vs-full first names ("K. Day"
# vs "Kevin Day") but not unrelated names ("Kevin Day" vs "David Fravor").
#
# UNCHANGED FOR WANT OF EVIDENCE, NOT BECAUSE IT WAS VALIDATED. There is no
# unbiased corpus to tune it on. The obvious one - pairs a curator merged by
# hand - is biased by construction: a pair reaches the merge ledger BECAUSE the
# matcher missed it, so pairs it caught are absent by definition and the
# measured "45.5% false reject" at 0.75 describes the residue, not the recall.
# Fitting the number to that population would fit it to the failures. Prefer
# fixing what the comparison measures (see collapse_acronym_expansions) over
# moving this line; a mechanism change can be argued from the pair it repairs,
# a threshold change cannot.
FUZZY_NAME_THRESHOLD = 0.75

# THE BOUNDARY OF THIS METHOD, worth knowing before reaching for a better
# string metric: about a quarter of genuine duplicates in this corpus share no
# words at all. "Tic Tac Sighting" and "2004 USS Nimitz UAP encounter" are one
# event under two names, and no edit-distance threshold reaches that pair at any
# setting - lowering it merely admits unrelated names first. Measured over the
# merge ledger, dropping to 0.70 leaves 36.5% unmatched and 0.60 leaves 23.7%,
# then it plateaus: the remainder is not a tuning problem. Those duplicates are
# what the embedding neighbourhood and human curation exist to catch; string
# matching is the cheap first pass, not the whole answer.

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
    """Similarity of one comma-component, initial-aware on its tokens.

    Guarded like the whole-name comparison. Place components carry mandated
    boilerplate of their own - "Walker Air Force Base" against "Kirtland Air
    Force Base" is 0.82 on the shared tail alone, and the structured branch
    takes the MINIMUM component score, so that one component decides the merge.
    Without this the hierarchy collapses exactly as the event names did: distinct
    bases, distinct Manhattan streets, Seattle onto Washington DC.
    """
    if a == b:
        return 1.0
    if _distinctive_tokens_disagree(a, b):
        return 0.0
    return max(levenshtein_ratio(a, b), _token_aware_ratio(a, b))


def _comma_components(name: str) -> list[str]:
    return [part.strip() for part in name.split(",") if part.strip()]


# An acronym declared anywhere in a name, not only as a trailing suffix:
# "1947 Kenneth Arnold Unidentified Flying Object (UFO) sighting".
# Case-insensitive where _ACRONYM_SUFFIX_RE is not: comparisons run on
# lowercased names, so ALL-CAPS is not available as the signal. The initials
# check below is the stronger evidence anyway - "Joe (mother) Smith" would need
# six preceding words spelling M-O-T-H-E-R - and a digits-only parenthetical
# ("(1947)") yields no letters and is left alone.
_INLINE_ACRONYM_RE = re.compile(
    r"((?:[^\s()]+\s+){1,8})\(([A-Za-z0-9][A-Za-z0-9-]{1,})\)"
)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether every element of needle appears in haystack, in order."""
    it = iter(haystack)
    return all(item in it for item in needle)


def collapse_acronym_expansions(name: str) -> str:
    """Rewrite "Full Name (ACRO)" to "ACRO" wherever the name declares one.

    Event names carry a mandated expansion ("... Unidentified Flying Object
    (UFO) incident") while the same event arrives from another digest in the
    short form ("... UFO incident"). Compared literally the expansion is most of
    the string, which breaks similarity in BOTH directions at once: two
    unrelated events score high on the shared boilerplate, and one event written
    both ways scores low because the spelled-out words are orphan tokens. Two
    directions, one cause - so this collapses rather than strips, putting both
    spellings into the short form instead of deleting the words from one side.

    WHY THIS MUST BE FIXED BEFORE ANY MERGE, not after. A merge deposits every
    victim name as an alias on the survivor, so merging duplicates MINTS fresh
    boilerplate-shaped aliases - collapsing the seven Roswell event nodes
    created ten of them in one operation. Measured against that post-merge graph
    with five unrelated event names: the previous comparison sent Socorro,
    Mantell and Rendlesham to the Nimitz node and Aztec to the Roswell one, four
    of five wrong, while this one resolves each correctly or mints a new node.
    So the merge that repairs fragmentation is also the operation that arms the
    next round of it, and the comparison is the only place to stop it. Fixing
    the matcher is not tidy-up that precedes a merge; it is what makes any merge
    in this corpus safe to perform.

    The declared acronym is the evidence, exactly as in is_bare_acronym_for: the
    span is only collapsed when the parenthetical's letters are the initials of
    the words in front of it. "Office of the Under Secretary of Defense for
    Intelligence (OUSDI)" does not match its initials (the stop-words break it)
    and is left untouched, which is the safe outcome - no collapse is no change.
    """

    def replace(m: re.Match) -> str:
        words = m.group(1).split()
        acronym = m.group(2)
        letters = [ch for ch in acronym if ch.isalpha()]
        if not letters or len(letters) > len(words):
            return m.group(0)
        # The regex takes as many preceding words as it can; the expansion is
        # the last len(letters) of them, and anything before that is unrelated
        # text that must survive ("1947 Kenneth Arnold" ahead of "UFO").
        # An acronym need not be one letter per word: "Infrared (IR)" draws two
        # from one, "Deoxyribonucleic Acid (DNA)" three from two. So try each
        # possible expansion length and accept the first where the words'
        # initials appear IN ORDER within the acronym and both start alike.
        # Requiring exactly one letter per word would reject those; requiring
        # nothing would collapse "Joe (mother)".
        # LONGEST expansion first: "Advanced Aerospace Threat Identification
        # Program (AATIP)" also matches on its last four words, and stopping at
        # the shortest would leave a stray "Advanced" in front of the acronym.
        upper = [ch.upper() for ch in letters]
        for take in range(len(words), 0, -1):
            expansion = words[-take:]
            initials = [w[:1].upper() for w in expansion if w[:1]]
            if not initials or initials[0] != upper[0]:
                continue
            if _is_subsequence(initials, upper):
                return " ".join(words[:-take] + [acronym]) + " "
        return m.group(0)

    return " ".join(_INLINE_ACRONYM_RE.sub(replace, name).split())


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
    # Score the pair in both acronym spellings and keep the better. Collapsing
    # is not safe to do unconditionally: it helps when both sides declare the
    # acronym ("... UFO sighting" vs "... Unidentified Flying Object (UFO)
    # sighting" become identical) but hurts when only one declares it and the
    # other spells the words out bare ("artificial intelligence (AI)" collapses
    # to "AI" and moves AWAY from "artificial intelligence"). Taking the max
    # keeps the gain without the asymmetry, and every guard below still applies
    # to whichever spelling is being scored.
    ca, cb = collapse_acronym_expansions(a), collapse_acronym_expansions(b)
    if (ca, cb) != (a, b) and not _collapses_to_a_bare_acronym(ca, cb):
        return max(_fuzzy_one_spelling(a, b), _fuzzy_one_spelling(ca, cb))
    return _fuzzy_one_spelling(a, b)


def _collapses_to_a_bare_acronym(ca: str, cb: str) -> bool:
    """Whether collapsing left one side as nothing but the acronym itself.

    "Unidentified Flying Object (UFO)" collapses to "UFO", and every "UFO
    <something>" topic then reads as a mere extension of it - "UFO disclosure",
    "UFO flap" and "UFO abduction phenomenon" all matched the bare phenomenon
    node this way. The extension rule is right for ordinary names ("Filming" vs
    "Filming 1964") but a three-letter stem makes it fire on anything, so the
    collapsed spelling is refused here. Nothing is lost: a bare acronym against
    the node that spells it out is tier 3's job, decided on the declared acronym
    rather than on string distance.
    """
    if ca == cb:
        return False
    return any(side and " " not in side and len(side) <= 10 for side in (ca, cb))


def _fuzzy_one_spelling(a: str, b: str) -> float:
    """fuzzy_name_similarity for one fixed acronym spelling of the pair."""
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


# Short forms that are the SAME person, not a similar one. Levenshtein does not
# reach these - "Dave"/"David" and "Hal"/"Harold" score below the fuzzy threshold -
# so without this they become two nodes, and each understates its own evidence
# because claim_count and independent_source_count are per node. The corpus had 25
# such splits when this was written, the largest being David/Dave Fravor at 572 and
# 23 references and Luis/Lue/Lou Elizondo at 256, 472 and 20.
_NICKNAMES = {
    "al": "albert",
    "andy": "andrew",
    "bill": "william",
    "bob": "robert",
    "chris": "christopher",
    "dan": "daniel",
    "danny": "daniel",
    "dave": "david",
    "dick": "richard",
    "ed": "edward",
    "eddie": "edward",
    "gil": "gilbert",
    "greg": "gregory",
    "hal": "harold",
    "jeff": "jeffrey",
    "jerry": "gerald",
    "jim": "james",
    "jimmy": "james",
    "joe": "joseph",
    "ken": "kenneth",
    "larry": "lawrence",
    "lou": "luis",
    "lue": "luis",
    "matt": "matthew",
    "mike": "michael",
    "nick": "nicholas",
    "pete": "peter",
    "rick": "richard",
    "rob": "robert",
    "ron": "ronald",
    "sam": "samuel",
    "steve": "stephen",
    "tom": "thomas",
    "tommy": "thomas",
    "tony": "anthony",
    "will": "william",
}


def _first_names_are_the_same_person(a: str, b: str) -> bool:
    """Whether two given names are one person's formal name and short form."""
    a, b = a.lower().rstrip("."), b.lower().rstrip(".")
    if a == b:
        return False
    if _NICKNAMES.get(a) == b or _NICKNAMES.get(b) == a:
        return True
    # "Chris" for "Christopher": a short form is a prefix of the full name. Two
    # characters is too short to be evidence ("Al" would take "Alan", "Alex" and
    # "Alfred" all at once).
    return (len(a) > 2 and b.startswith(a)) or (len(b) > 2 and a.startswith(b))


def is_nickname_of(name: str, other: str) -> bool:
    """Whether two PERSON names differ only by a formal name and its short form.

    Every token after the first must match EXACTLY. That strictness is the whole
    rule, and it was arrived at by measurement: matching on surname plus a
    nickname-ish first name found 37 pairs in the corpus and was wrong about a
    third of them - John Fitzgerald Kennedy against John Neely Kennedy, George
    Herbert Walker Bush against George W. Bush, Baron Magnus against Baroness Emmy
    von Braun, and several that collided only because "Jr." is the last word of a
    name. Requiring the remainder to be identical removed every false pair without
    losing a true one.
    """
    a, b = name.split(), other.split()
    if len(a) < 2 or len(b) < 2 or a[1:] != b[1:]:
        return False
    return _first_names_are_the_same_person(a[0], b[0])


# --- Record-scoped person names ---
#
# A person name that carries no family name identifies somebody only within the
# record that uses it. "Chris" in a book about mediumship and "Chris" on a drone
# video are two people; "Mrs. M." in one memoir and "Mrs. Z." in another are two
# more. The matcher treated every one of them as a global identity: the exact
# tier joined the two Chrises (48 claims on one node, 42 of them about a dead man
# and 6 about a cameraman), and the fuzzy tier folded four anonymised women from
# four books into "Mrs. M." (Levenshtein puts "mrs. z." at 0.83 of "mrs. m.",
# and "M." is an initial of "Markham"). Fuller names then lost to shorter ones
# on arrival order: "Tim Taylor" became an alias of a "Taylor" that got there
# first at exactly the 0.75 threshold.
#
# So a name with no family name is RECORD-SCOPED: it resolves only to the node
# already holding that record's claims under the same name, never across
# records, and it is never a page (page_gate). Mononyms of historical and
# mythological figures are the documented exception - "Plato" is one person
# across six sources - and are listed rather than inferred, because the
# inference that failed here was precisely "one token, one identity".

_HONORIFICS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "dr",
        "prof",
        "professor",
        "sir",
        "dame",
        "lady",
        "lord",
        "rev",
        "reverend",
        "fr",
        "father",
        "sister",
        "brother",
        "saint",
        "st",
        "capt",
        "captain",
        "cdr",
        "cmdr",
        "commander",
        "col",
        "colonel",
        "gen",
        "general",
        "lt",
        "lieutenant",
        "maj",
        "major",
        "sgt",
        "sergeant",
        "adm",
        "admiral",
        "pvt",
        "private",
        "senator",
        "president",
        "judge",
        "agent",
        "officer",
    }
)
_NAME_SUFFIXES = frozenset({"jr", "sr", "phd", "md", "esq"})
_ROMAN_NUMERAL_RE = re.compile(r"^[ivx]{1,4}$")
_INITIAL_RE = re.compile(r"^[a-z](?:\.[a-z])*$")
_MONONYMS = frozenset(
    {
        "abraham",
        "aristotle",
        "buddha",
        "confucius",
        "enoch",
        "ezekiel",
        "hermes",
        "homer",
        "ishtar",
        "isaiah",
        "jesus",
        "lucretius",
        "moses",
        "muhammad",
        "nostradamus",
        "plato",
        "pythagoras",
        "socrates",
        "zoroaster",
    }
)


def person_name_tokens(name: str) -> list[str]:
    """Lowercased name tokens with parentheticals, honorifics and suffixes gone.

    >>> person_name_tokens("Dr. K. Day Jr. (Navy pilot)")
    ['k', 'day']
    >>> person_name_tokens("Mrs. Markham")
    ['markham']
    >>> person_name_tokens("Eisenhower, Dwight D.")
    ['dwight', 'd', 'eisenhower']
    """
    base = _PARENS_GROUP_RE.sub(" ", name.lower())
    # Surname-first input ("Eisenhower, Dwight D.") is put back in natural
    # order, so the family name is the last token whichever way it was written.
    if "," in base:
        family, _, given = base.partition(",")
        base = f"{given} {family}"

    def _split(text: str) -> list[str]:
        return [
            t
            for t in (_TOKEN_PUNCT_RE.sub("", w) for w in re.split(r"[\s/,]+", text))
            if t
        ]

    tokens = _split(base)
    while tokens and tokens[0] in _HONORIFICS:
        tokens.pop(0)
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return tokens


def _is_initial(token: str) -> bool:
    return bool(_INITIAL_RE.match(token))


@functools.lru_cache(maxsize=65536)
def is_record_scoped_person_name(name: str) -> bool:
    """Whether a person name identifies somebody only within its own record.

    True when no family name is present: a bare given name ("Chris", "Sally
    (Budd Hopkins abductee)"), an honorific plus one token ("Mrs. Markham",
    "Dr. X"), or a name whose family name is an initial ("Mrs. M.", "John D.").
    A listed mononym ("Plato") and a regnal name ("Elizabeth I") are global.

    >>> is_record_scoped_person_name("Chris")
    True
    >>> is_record_scoped_person_name("Mrs. M.")
    True
    >>> is_record_scoped_person_name("K. Day")
    False
    >>> is_record_scoped_person_name("Plato")
    False
    >>> is_record_scoped_person_name("Elizabeth I")
    False
    """
    tokens = person_name_tokens(name)
    if not tokens:
        return True
    regnal = len(tokens) > 1 and _ROMAN_NUMERAL_RE.match(tokens[-1]) is not None
    if regnal:
        return False
    if len(tokens) == 1:
        return tokens[0] not in _MONONYMS
    return _is_initial(tokens[-1])


def _person_tokens_compatible(a: str, b: str) -> bool:
    return (
        a == b or _initials_compatible(a, b) or _first_names_are_the_same_person(a, b)
    )


def is_fuller_person_name(name: str, existing: str) -> bool:
    """Whether `name` says everything `existing` says about a person, and more.

    The graph keeps the first spelling it met as the canonical name and files
    later ones as aliases, so a node named "K. Day" stays "K. Day" after "Kevin
    Day" arrives. This is the test for promoting the newcomer: every existing
    token has a counterpart in order (equal, an initial of it, or its short
    form), the newcomer carries more tokens, and it does not trade spelled-out
    names for initials ("Kevin Day" is not bettered by "K. R. Day").

    >>> is_fuller_person_name("Kevin Day", "K. Day")
    True
    >>> is_fuller_person_name("Harold E. Puthoff", "Hal Puthoff")
    True
    >>> is_fuller_person_name("K. R. Day", "Kevin Day")
    False
    >>> is_fuller_person_name("Dave Fravor", "David Fravor")
    False
    >>> is_fuller_person_name("Lionel Browning's wife", "Lionel Browning")
    False
    """
    # Only a NAME can be the fuller name: every word capitalised, no possessive.
    # "Lionel Browning's wife" fuzzy-matches Lionel Browning and has more tokens,
    # and is a description of somebody else.
    bare = _PARENS_GROUP_RE.sub(" ", name)
    if re.search(r"['’]s\b", bare):
        return False
    words = [w for w in re.split(r"[\s/,]+", bare) if any(ch.isalpha() for ch in w)]
    if any(w[0].isalpha() and w[0].islower() for w in words):
        return False
    new, old = person_name_tokens(name), person_name_tokens(existing)
    if len(new) < len(old) or not old:
        return False
    spelled_new = sum(1 for t in new if not _is_initial(t))
    spelled_old = sum(1 for t in old if not _is_initial(t))
    if spelled_new < spelled_old:
        return False
    if len(new) == len(old) and spelled_new == spelled_old:
        return False
    if is_record_scoped_person_name(name):
        return False
    i = 0
    for token in old:
        while i < len(new) and not _person_tokens_compatible(token, new[i]):
            i += 1
        if i == len(new):
            return False
        i += 1
    return True


def same_record_person(
    conn: sqlite3.Connection, name: str, record_id: str
) -> str | None:
    """The person this record already knows by this name, if any.

    A record-scoped name resolves only among the people THIS record declares or
    quotes: by exact name or alias first (a re-digest or reconcile must land on
    the node the earlier import minted, or every pass would mint another
    "Chris"), then - for a single token - by surname or given name when exactly
    one of the record's people carries it ("Fravor" in a record that declares
    "David Fravor" is that man; in a record with two Fravors it is nobody).
    """
    people = conn.execute(
        """
        SELECT DISTINCT n.id, n.name FROM nodes n
         WHERE n.retired_at IS NULL AND n.node_type = 'person'
           AND (EXISTS (SELECT 1 FROM record_nodes rn
                         WHERE rn.node_id = n.id AND rn.record_id = ?)
             OR EXISTS (SELECT 1 FROM claims c
                         WHERE c.speaker_id = n.id AND c.record_id = ?)
             OR EXISTS (SELECT 1 FROM claim_node_refs r JOIN claims c ON c.id = r.claim_id
                         WHERE r.node_id = n.id AND c.record_id = ?))
        """,
        (record_id, record_id, record_id),
    ).fetchall()
    if not people:
        return None
    for node_id, node_name in people:
        if node_name == name:
            return node_id
    ids = [p[0] for p in people]
    marks = ",".join("?" * len(ids))
    row = conn.execute(
        f"SELECT node_id FROM aliases WHERE alias = ? AND node_id IN ({marks})",
        (name, *ids),
    ).fetchone()
    if row:
        return row[0]
    tokens = person_name_tokens(name)
    if len(tokens) != 1 or _is_initial(tokens[0]):
        return None
    token = tokens[0]
    carriers = [
        node_id
        for node_id, node_name in people
        if (lambda t: len(t) >= 2 and token in (t[0], t[-1]))(
            person_name_tokens(node_name)
        )
    ]
    return carriers[0] if len(carriers) == 1 else None


_TRAILING_ACRONYM = re.compile(r"\(([A-Za-z0-9./-]{2,10})\)\s*$")


def acronym_of(full_name: str) -> str | None:
    """The acronym a name declares in a trailing parenthetical, if any."""
    m = _TRAILING_ACRONYM.search(full_name)
    return m.group(1).upper() if m else None


def looks_like_a_bare_acronym(name: str) -> bool:
    """Whether a name is nothing but an acronym: "NASA", "MJ-12", "AAV"."""
    bare = name.strip()
    return bool(bare) and len(bare) <= 10 and " " not in bare and bare == bare.upper()


def is_bare_acronym_for(name: str, full_name: str) -> bool:
    """Whether `name` is the bare acronym that `full_name` spells out.

    name_equivalence_key already collapses "X" against "X (ACRO)" - the same words
    with and without the parenthetical. It cannot collapse "NASA" against "National
    Aeronautics and Space Administration (NASA)", because it strips the
    parenthetical from one side and compares "nasa" to the spelled-out words. So
    the bare form becomes its own node.

    26 acronyms in the corpus had both forms when this was written. Most bare ones
    were empty and merely cluttered the graph, but several carried references that
    should have been on the expansion: MJ-12 held 3 against Majestic 12 (MJ-12)'s
    52, SAP 1 against 34, CNN 2 against 15.

    The declared acronym is the evidence, not a guess assembled from initials.
    Deriving one would match far too much - "Advanced Aerospace Threat
    Identification Program" and "Airborne Anomaly Tracking Initiative Programme"
    both reduce to AATIP.
    """
    if not looks_like_a_bare_acronym(name):
        return False
    return acronym_of(full_name) == name.strip().upper()


def punctuation_blind_key(name: str) -> str:
    """A name reduced to its letters and digits: "KLAS-TV" and "KLAS TV" agree.

    Punctuation and spacing almost never distinguish two entities, but they
    reliably split one. Nine same-type pairs in the corpus differed by nothing
    else - "Office of the Under Secretary of Defense for Intelligence (OUSDI)"
    against "Office of the Undersecretary..." at 42 references and 19, "Stargate"
    against "Star Gate", "F-117A Nighthawk" against "F-117A Night Hawk", "S-4
    Facility" against "S4 (facility)".

    Applied only within a node_type, and only after the exact, acronym and
    nickname passes have failed, so it cannot reinterpret a match those made.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_node(
    conn: sqlite3.Connection,
    name: str,
    node_type: str | None = None,
    record_id: str | None = None,
) -> tuple[str, str] | None:
    """Try to match a name to an existing node.

    Returns (node_id, match_method) or None if no match found.
    match_method is one of: "exact", "alias", "acronym", "punctuation",
    "nickname", "fuzzy", "record".

    A DESCRIPTION NEVER MATCHES ANYTHING. It is record-scoped, so there is no
    node it could correctly resolve to - and the fuzzy tier will happily find one
    anyway: "[Anomaly Physical Evidence Group (APEG) biochemist]" matched the very
    node "unnamed Anomaly Physical Evidence Group (APEG) biochemist" it had just
    been written to replace, silently re-creating the refs the rewrite removed.
    The check lives HERE rather than at each call site because refs, speakers and
    node minting are three paths and guarding one of them is how that happened.

    A PERSON NAME WITHOUT A FAMILY NAME MATCHES ONLY WITHIN ITS RECORD (see
    is_record_scoped_person_name): given `record_id` it resolves to the node that
    record already uses under that exact name, otherwise to nothing, and no
    person node bearing such a name is ever a candidate for another name - so
    "Tim Taylor" cannot be filed under a "Taylor" that arrived first.
    """
    if is_a_description(name):
        return None
    scoped = node_type in (None, "person") and is_record_scoped_person_name(name)
    if scoped:
        if record_id:
            same = same_record_person(conn, name, record_id)
            if same:
                return same, "record"
        if node_type == "person":
            return None

    # 1. Exact name match
    exact = find_node_by_name(conn, name, node_type)
    if exact and not (scoped and exact.node_type == "person"):
        return exact.id, "exact"

    # 2. Acronym-suffix normalisation: X <-> X (ACRONYM) collapse to the same
    # node. Avoids duplicate organisation/concept nodes for "Defense
    # Intelligence Agency" and "Defense Intelligence Agency (DIA)".
    candidates_all = [
        c
        for c in get_nodes(conn, node_type=node_type)
        if not (c.node_type == "person" and is_record_scoped_person_name(c.name))
    ]
    if scoped:
        # An untyped reference that reads as a bare given name: the person
        # tier is closed to it, but an organisation or place of that exact
        # name is still a legitimate target.
        for candidate in candidates_all:
            if candidate.name == name:
                return candidate.id, "exact"
    key = name_equivalence_key(name)
    for candidate in candidates_all:
        if name_equivalence_key(candidate.name) == key and candidate.name != name:
            return candidate.id, "acronym"

    # 3. A bare acronym belongs to the node that spells it out. Prefers the
    # most-referenced expansion, since a body that changed name over time has
    # several ("North American Aerospace Defense Command (NORAD)" and "North
    # American Air Defense Command (NORAD)") and the busiest is the live one.
    if looks_like_a_bare_acronym(name):
        spelled = [c for c in candidates_all if is_bare_acronym_for(name, c.name)]
        if spelled:
            counts = {
                c.id: conn.execute(
                    "SELECT count(*) FROM claim_node_refs WHERE node_id = ?", (c.id,)
                ).fetchone()[0]
                for c in spelled
            }
            return max(spelled, key=lambda c: counts[c.id]).id, "acronym"

    # 4. Punctuation and spacing: "KLAS-TV" is "KLAS TV".
    pkey = punctuation_blind_key(name)
    for candidate in candidates_all:
        if candidate.name != name and punctuation_blind_key(candidate.name) == pkey:
            return candidate.id, "punctuation"

    # 5. Nickname: "Dave Fravor" is "David Fravor". Ahead of the fuzzy pass
    # because these score BELOW the Levenshtein threshold and would otherwise be
    # missed entirely; restricted to people, since the rule is a given-name rule.
    if (node_type or "") == "person":
        for candidate in candidates_all:
            if is_nickname_of(name, candidate.name):
                return candidate.id, "nickname"

    # 6. Fuzzy name match via Levenshtein
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
                    # Structure-aware, exactly as the name comparison below. Raw
                    # Levenshtein here was the single widest false-merge path in
                    # the graph: event names carry a mandated boilerplate tail
                    # ("... Unidentified Flying Object (UFO) incident") that is
                    # most of the string, so any two events scored above the
                    # threshold on the scaffolding alone - "1947 Roswell ..." vs
                    # the Nimitz alias "2004 USS Nimitz ..." reached 0.811 with
                    # nothing in common but the boilerplate. Worse, it ratcheted:
                    # a match is recorded as a new alias (see import_markdown),
                    # so each false match widened the net for the next one, which
                    # is how one node accumulated 118 aliases naming other events.
                    # fuzzy_name_similarity applies the hard-token guard (1947 vs
                    # 2004 conflict) and the distinctive-token guard, taking that
                    # pair to 0.0. Exact alias hits never reach here - they match
                    # in tier 1 - so this only tightens the fuzzy path.
                    alias_sim = fuzzy_name_similarity(name_lower, alias.lower())
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
