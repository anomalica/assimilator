"""One-off taxonomy fixes applied to existing .extract.md files.

The .extract.md files are the source of truth, so reclassifying nodes means
rewriting node_type values *and names* in those files. After running these
passes the database is rebuilt to pick up the changes.

Three passes provided:
- reclassify_documents_*  - changes object/matter to document for artefact names
- normalise_person_names_* - rewrites "First Last" person names to "Last, First"
- normalise_place_names_*  - rewrites "City State" places to "Country, State, City"
"""

from __future__ import annotations

import re
from pathlib import Path

# Names ending in or containing one of these tokens are documents. Word-boundary
# match so "Marauder UAP Reporting System" doesn't trip "Report".
_DOCUMENT_SUFFIX_TOKENS = (
    "Memo",
    "Memorandum",
    "Report",
    "Letter",
    "Letters",
    "Article",
    "Articles",
    "Paper",
    "Papers",
    "Book",
    "Books",
    "Brief",
    "Briefing",
    "Briefings",
    "Slide",
    "Slides",
    "Document",
    "Documents",
    "Disclosure",
    "Disclosures",
    "Publication",
    "Publications",
    "Statement",
    "Testimony",
    "Affidavit",
    "Footage",
    "Recording",
    "Transcript",
    "Video",
    "Photographs",
    "Photograph",
    "Bulletin",
    "Volume",
    "Evaluation",
    "Assessment",
    "Estimate",
    "Summary",
    "Findings",
    "Analysis",
    "Dossier",
    "Treaty",
    "Notes",
    "Communique",
)

# These tokens preserve the node as an object/matter even if a document token
# appears - a "Reporting System" is a system, not a report.
_EXCLUSION_TOKENS = (
    "System",
    "Systems",
    "Programme",
    "Programmes",
    "Program",
    "Programs",
    "Centre",
    "Center",
    "Database",
    "Network",
    "Facility",
    "Service",
)

_DOC_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(_DOCUMENT_SUFFIX_TOKENS) + r")s?$",
    re.IGNORECASE,
)
_DOC_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_DOCUMENT_SUFFIX_TOKENS) + r")s?\b",
    re.IGNORECASE,
)
_EXCLUSION_RE = re.compile(
    r"\b(?:" + "|".join(_EXCLUSION_TOKENS) + r")\b",
    re.IGNORECASE,
)

# Specific names that should always be documents but might not match the
# generic patterns (the Navy UAP videos by their colloquial names).
_KNOWN_DOCUMENTS = {
    "FLIR1",
    "FLIR-1",
    "Gimbal",
    "Go-Fast",
    "Tic Tac FLIR",
    "Tic Tac FLIR Video",
}


_NODE_LINE_RE = re.compile(
    r"^### (?P<id>[a-f0-9-]{36}) (?P<type>object|matter): (?P<name>.+)$"
)


def is_document_name(name: str) -> bool:
    """Return True if a node's name looks like a document/artefact."""
    if name in _KNOWN_DOCUMENTS:
        return True
    if _EXCLUSION_RE.search(name):
        return False
    # Suffix match - "X Memo", "Y Report", "Some Briefing"
    if _DOC_SUFFIX_RE.search(name):
        return True
    # Whole-word document tokens inside the name (e.g. "Twining Letter 1947")
    if _DOC_WORD_RE.search(name):
        return True
    return False


def reclassify_documents_in_file(path: Path) -> int:
    """Rewrite `object/matter: X` lines to `document: X` in one extract file.

    Returns the number of nodes reclassified.
    """
    text = path.read_text()
    new_lines: list[str] = []
    reclassified = 0
    for line in text.splitlines():
        m = _NODE_LINE_RE.match(line)
        if m and is_document_name(m.group("name")):
            new_lines.append(f"### {m.group('id')} document: {m.group('name')}")
            reclassified += 1
        else:
            new_lines.append(line)
    if reclassified:
        path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))
    return reclassified


def reclassify_documents_in_dir(extracts_dir: Path) -> dict[str, int]:
    """Apply reclassify_documents_in_file across an extracts directory.

    Returns {filename: count} for files where anything changed.
    """
    results: dict[str, int] = {}
    for path in sorted(extracts_dir.glob("*.extract.md")):
        count = reclassify_documents_in_file(path)
        if count:
            results[path.name] = count
    return results


# --- Person name normalisation ---

# Rank/title prefixes we strip - if a person node leads with one, the canonical
# name is what follows. "Commander David Fravor" -> "Fravor, David".
_RANK_PREFIXES = (
    "Adm",
    "Admiral",
    "Air Marshal",
    "Brig Gen",
    "Brigadier General",
    "Brigadier",
    "Capt",
    "Captain",
    "Cdr",
    "Col",
    "Colonel",
    "Commander",
    "Commodore",
    "Cpt",
    "Dr",
    "Fl Lt",
    "Flight Lieutenant",
    "Flt Lt",
    "Gen",
    "General",
    "LCDR",
    "Lt",
    "Lt Cdr",
    "Lt Col",
    "Lt Gen",
    "Lieutenant",
    "Lieutenant Colonel",
    "Lieutenant Commander",
    "Lieutenant General",
    "Maj",
    "Maj Gen",
    "Major",
    "Major General",
    "Mr",
    "Mrs",
    "Ms",
    "Prof",
    "Professor",
    "Rear Admiral",
    "Rev",
    "Senator",
    "Sen",
    "Sgt",
    "Sr",
    "Vice Admiral",
)

# Common name suffixes that travel with the surname under "Last, First" format
_NAME_SUFFIXES = ("Jr", "Sr", "II", "III", "IV", "Jr.", "Sr.")

_PERSON_NODE_RE = re.compile(r"^### (?P<id>[a-f0-9-]{36}) person: (?P<name>.+)$")


def _strip_rank(name: str) -> str:
    """Remove leading rank/title and any trailing rank-y suffix like 'US Navy'."""
    stripped = name.strip()
    # Try multi-word ranks first, then single-word, longest match
    for rank in sorted(_RANK_PREFIXES, key=len, reverse=True):
        if stripped.lower().startswith(rank.lower() + " "):
            stripped = stripped[len(rank) + 1 :].strip()
            break
        if stripped.lower().startswith(rank.lower() + ". "):
            stripped = stripped[len(rank) + 2 :].strip()
            break
    return stripped


def normalise_person_name(name: str) -> str | None:
    """Return "Last, First Middle" form, or None if we can't safely transform.

    Conservative - only rewrites if the input parses cleanly as a sequence of
    name tokens with a single rank prefix at most. Skips:
    - Single-word names with no rank prefix (no first name to swap)
    - Already-comma-format names
    - Names with parenthetical content
    - Names containing digits or non-letter unicode (call signs etc.)

    A rank prefix followed by a single surname (e.g. "Lieutenant Commander Moya")
    canonicalises to just the surname ("Moya").
    """
    if "," in name:
        return None
    if "(" in name or ")" in name:
        return None
    cleaned = _strip_rank(name)
    if not cleaned:
        return None
    # Reject if any token has a digit
    if any(ch.isdigit() for ch in cleaned):
        return None
    tokens = cleaned.split()
    if len(tokens) == 0:
        return None
    # Rank prefix + lone surname: canonical form is just the surname.
    if len(tokens) == 1:
        if cleaned == name:
            return None  # Untouched single word - nothing to do
        return cleaned
    # Handle suffix: if last token is Jr/Sr/II/III/IV, attach it to the surname
    suffix = ""
    if tokens[-1].rstrip(".") in [s.rstrip(".") for s in _NAME_SUFFIXES]:
        suffix = " " + tokens[-1]
        tokens = tokens[:-1]
        if len(tokens) < 2:
            # Rank prefix + surname + suffix (rare). Treat surname+suffix as the name.
            return tokens[0] + suffix if tokens else None
    surname = tokens[-1] + suffix
    given = " ".join(tokens[:-1])
    return f"{surname}, {given}"


_CLAIM_HEADER_RE = re.compile(
    r"^(### [a-f0-9-]{36} \[[^\]]+\])(?P<speaker_part> speaker:(?P<speaker_name>.+))?$"
)


def _rename_refs_line(line: str, mapping: dict[str, str]) -> str:
    """Rewrite a "refs: Foo, Bar, Baz" line using the rename mapping."""
    if not line.startswith("refs: "):
        return line
    names = [n.strip() for n in line[len("refs: ") :].split(",")]
    renamed = [mapping.get(n, n) for n in names]
    return "refs: " + ", ".join(renamed)


def _rename_speaker(line: str, mapping: dict[str, str]) -> str:
    """Rewrite the `speaker:` segment of a claim header line."""
    m = _CLAIM_HEADER_RE.match(line)
    if not m or not m.group("speaker_name"):
        return line
    speaker = m.group("speaker_name").strip()
    new_speaker = mapping.get(speaker)
    if new_speaker is None or new_speaker == speaker:
        return line
    return f"{m.group(1)} speaker:{new_speaker}"


def normalise_person_names_in_file(path: Path) -> int:
    """Rewrite `person: First Last` lines to `person: Last, First` where safe.

    Also rewrites the `refs:` and `speaker:` references throughout the file so
    that downstream import can still resolve them to the renamed nodes.
    """
    text = path.read_text()
    lines = text.splitlines()
    rename_map: dict[str, str] = {}
    # First pass: collect renames
    for line in lines:
        m = _PERSON_NODE_RE.match(line)
        if m:
            new_name = normalise_person_name(m.group("name"))
            if new_name and new_name != m.group("name"):
                rename_map[m.group("name")] = new_name
    if not rename_map:
        return 0
    # Second pass: apply
    new_lines: list[str] = []
    normalised_nodes = 0
    for line in lines:
        m = _PERSON_NODE_RE.match(line)
        if m and m.group("name") in rename_map:
            new_lines.append(
                f"### {m.group('id')} person: {rename_map[m.group('name')]}"
            )
            normalised_nodes += 1
            continue
        line = _rename_refs_line(line, rename_map)
        line = _rename_speaker(line, rename_map)
        new_lines.append(line)
    path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))
    return normalised_nodes


def normalise_person_names_in_dir(extracts_dir: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in sorted(extracts_dir.glob("*.extract.md")):
        count = normalise_person_names_in_file(path)
        if count:
            results[path.name] = count
    return results


# --- Place name normalisation ---

# Known regions whose names alone identify a country.
# Used mechanically: if a place name ends in one of these tokens, prepend the
# matching country. "Aztec New Mexico" -> "USA, New Mexico, Aztec".
_US_STATES = {
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "Washington DC",
    "DC",
}

_AU_STATES = {
    "New South Wales",
    "Victoria",
    "Queensland",
    "South Australia",
    "Western Australia",
    "Tasmania",
    "Northern Territory",
    "ACT",
    "Australian Capital Territory",
}

_CA_PROVINCES = {
    "Alberta",
    "British Columbia",
    "Manitoba",
    "New Brunswick",
    "Newfoundland",
    "Newfoundland and Labrador",
    "Nova Scotia",
    "Ontario",
    "Prince Edward Island",
    "Quebec",
    "Saskatchewan",
    "Yukon",
    "Northwest Territories",
    "Nunavut",
}

_UK_COUNTRIES = {"England", "Scotland", "Wales", "Northern Ireland"}

_NZ_REGIONS = {
    "North Island",
    "South Island",
    "Auckland",
    "Wellington",
    "Canterbury",
    "Otago",
    "Waikato",
    "Bay of Plenty",
    "Manawatu",
    "Hawkes Bay",
    "Taranaki",
    "Northland",
    "Southland",
    "Marlborough",
    "Tasman",
    "Nelson",
    "West Coast",
    "Gisborne",
}

_COUNTRY_BY_REGION: dict[str, tuple[str, str]] = {}
for r in _US_STATES:
    _COUNTRY_BY_REGION[r] = ("USA", r)
for r in _AU_STATES:
    _COUNTRY_BY_REGION[r] = ("Australia", r)
for r in _CA_PROVINCES:
    _COUNTRY_BY_REGION[r] = ("Canada", r)
for r in _UK_COUNTRIES:
    _COUNTRY_BY_REGION[r] = ("United Kingdom", r)
for r in _NZ_REGIONS:
    _COUNTRY_BY_REGION[r] = ("New Zealand", r)

_PLACE_NODE_RE = re.compile(r"^### (?P<id>[a-f0-9-]{36}) place: (?P<name>.+)$")


def normalise_place_name(name: str) -> str | None:
    """Return "Country, Region, Specific" form, or None if we can't safely transform.

    Looks for a recognised region name appearing at the end of the place
    (separated by a comma OR a space). Only acts on US/AU/CA/UK/NZ regions.
    """
    if "," in name:
        # Already has a comma - assume the author followed the convention.
        return None
    # Try longest region names first so "New South Wales" beats "New South".
    for region in sorted(_COUNTRY_BY_REGION.keys(), key=len, reverse=True):
        suffix = " " + region
        if name == region:
            country, region_name = _COUNTRY_BY_REGION[region]
            return f"{country}, {region_name}"
        if name.endswith(suffix):
            country, region_name = _COUNTRY_BY_REGION[region]
            specific = name[: -len(suffix)].strip()
            if specific:
                return f"{country}, {region_name}, {specific}"
            return f"{country}, {region_name}"
    return None


def normalise_place_names_in_file(path: Path) -> int:
    """Rewrite `place: X State` lines to `place: Country, State, X` where safe.

    Also rewrites `refs:` lines so claim references resolve to the renamed
    place nodes.
    """
    text = path.read_text()
    lines = text.splitlines()
    rename_map: dict[str, str] = {}
    for line in lines:
        m = _PLACE_NODE_RE.match(line)
        if m:
            new_name = normalise_place_name(m.group("name"))
            if new_name and new_name != m.group("name"):
                rename_map[m.group("name")] = new_name
    if not rename_map:
        return 0
    new_lines: list[str] = []
    normalised_nodes = 0
    for line in lines:
        m = _PLACE_NODE_RE.match(line)
        if m and m.group("name") in rename_map:
            new_lines.append(
                f"### {m.group('id')} place: {rename_map[m.group('name')]}"
            )
            normalised_nodes += 1
            continue
        new_lines.append(_rename_refs_line(line, rename_map))
    path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))
    return normalised_nodes


def normalise_place_names_in_dir(extracts_dir: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in sorted(extracts_dir.glob("*.extract.md")):
        count = normalise_place_names_in_file(path)
        if count:
            results[path.name] = count
    return results


# --- Reference rewiring (recovery pass after a rename done without refs update) ---


def _reverse_person_name(name: str) -> str | None:
    """Given "Last, First Middle", return "First Middle Last". None if not comma form."""
    if "," not in name:
        return None
    surname_part, given_part = name.split(",", 1)
    surname = surname_part.strip()
    given = given_part.strip()
    if not surname or not given:
        return None
    return f"{given} {surname}"


def _reverse_place_name(name: str) -> str | None:
    """Given "Country, Region, Specific", return "Specific Region". None if not comma form.

    Used to undo place name normalisation when rebuilding refs.
    """
    if "," not in name:
        return None
    parts = [p.strip() for p in name.split(",")]
    if len(parts) == 2:
        # "Country, Region" - reverse to just "Region"
        return parts[1]
    if len(parts) >= 3:
        # "Country, Region, Specific" - reverse to "Specific Region"
        return f"{parts[2]} {parts[1]}"
    return None


def rewire_refs_in_file(path: Path) -> int:
    """Update refs/speaker lines to point at the renamed person and place nodes.

    Builds a reverse map: for each `person: Last, First` and `place: Country,
    Region, Specific` node, computes the pre-rename form and adds it to the
    map. Then rewrites refs/speakers throughout the file.

    Returns the number of references updated.
    """
    text = path.read_text()
    lines = text.splitlines()

    rename_map: dict[str, str] = {}
    for line in lines:
        m = _PERSON_NODE_RE.match(line)
        if m:
            old = _reverse_person_name(m.group("name"))
            if old:
                rename_map[old] = m.group("name")
            continue
        m = _PLACE_NODE_RE.match(line)
        if m:
            old = _reverse_place_name(m.group("name"))
            if old:
                rename_map[old] = m.group("name")
    if not rename_map:
        return 0

    new_lines: list[str] = []
    updates = 0
    for line in lines:
        new_line = _rename_refs_line(line, rename_map)
        if new_line != line:
            updates += 1
        new_line = _rename_speaker(new_line, rename_map)
        if new_line != line and new_line == _rename_refs_line(line, rename_map):
            pass  # already counted as a refs update; skip double-count
        elif new_line != line:
            updates += 1
        new_lines.append(new_line)
    if updates:
        path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))
    return updates


def rewire_refs_in_dir(extracts_dir: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in sorted(extracts_dir.glob("*.extract.md")):
        count = rewire_refs_in_file(path)
        if count:
            results[path.name] = count
    return results


# --- Refs delimiter migration: comma -> semicolon, with comma-in-name recovery ---

_NODE_ANY_RE = re.compile(r"^### [a-f0-9-]{36} \w+: (?P<name>.+)$")


def _all_node_names(extracts_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in extracts_dir.glob("*.extract.md"):
        for line in path.read_text().splitlines():
            m = _NODE_ANY_RE.match(line)
            if m:
                names.add(m.group("name"))
    return names


def _disambiguate_refs(raw: str, node_names: set[str]) -> list[str]:
    """Parse a comma-delimited refs string where names may themselves contain commas.

    Greedily merges consecutive comma-separated tokens whenever the joined
    string matches a known node name. Falls back to single-token splits when
    no longer match exists.
    """
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []
    result: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        # Try longest-first: span from full remainder down to single token
        for j in range(len(tokens), i, -1):
            candidate = ", ".join(tokens[i:j])
            if candidate in node_names:
                result.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return result


def migrate_refs_delimiter_in_file(path: Path, node_names: set[str]) -> int:
    """Rewrite refs lines to use semicolon delimiter, disambiguating commas.

    Returns the number of refs lines changed.
    """
    text = path.read_text()
    new_lines: list[str] = []
    changes = 0
    for line in text.splitlines():
        if line.startswith("refs: ") and ";" not in line:
            raw = line[len("refs: ") :]
            refs = _disambiguate_refs(raw, node_names)
            new_line = "refs: " + "; ".join(refs)
            if new_line != line:
                changes += 1
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    if changes:
        path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))
    return changes


def migrate_refs_delimiter_in_dir(extracts_dir: Path) -> dict[str, int]:
    """Migrate every refs line in the directory to semicolon delimiter."""
    node_names = _all_node_names(extracts_dir)
    results: dict[str, int] = {}
    for path in sorted(extracts_dir.glob("*.extract.md")):
        count = migrate_refs_delimiter_in_file(path, node_names)
        if count:
            results[path.name] = count
    return results
