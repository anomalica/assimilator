"""Person names in natural order, with the surname carried as a field.

node-types.md (amendment 2026-06-29) stores person names in natural order
("David Fravor"). The surname-first form is at most a display step. That leaves
nowhere to read the surname from - the comma used to BE the surname marker, and
three consumers parsed it (the vault's surname-sorted filenames, the assembler's
display flip, the matcher's comma-component precision). So the migration does two
things at once: it inverts the name and it writes ``metadata.family_name``, which
is what those consumers read afterwards. Inverting without the field would lose
the surname for every non-Anglo name, where "last token" is not the family name.

The pre-migration form is kept as ``metadata.aliases`` so last-first input still
resolves (node-types.md: the matcher stays order-tolerant permanently, because
real names arrive both ways).

Places are NOT touched. "USA, Nevada, Area 51" is largest-unit-first and that
convention is unchanged; the slug helper still reorders on the comma for them.

Digests are rewritten LINE BY LINE, not re-emitted from a parsed document: a
round-trip through the emitter restyles sequence indentation and scalars across
the whole file, burying a 4-line rename in a 1700-line diff. The rewrite is
verified by re-parsing the result and comparing it to the transform applied to
the parsed document, so a mangled name fails the pass instead of being written.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from assimilator.digest_files import canonical_digests

_SUFFIX_RE = re.compile(r"^(?:Jr|Sr|II|III|IV|V)\.?$", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\s*(\([^()]*\))$")

# `id:` always precedes `name:` in every emitted block that carries a name -
# node definitions, claim refs, and claim speakers alike - so the id line is what
# tells us whether a name belongs to a person.
_ID_LINE_RE = re.compile(r"^\s*(?:- )?id: (\S+)\s*$")
_NAME_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<dash>- )?name: (?P<name>.+?)\s*$")
_METADATA_LINE_RE = re.compile(r"^(?P<indent>\s*)metadata:\s*$")
# record.producer is a bare person name with no id beside it, and the importer
# links it to a node by exact name - so it has to move with the rename or the
# record loses its producer.
_PRODUCER_LINE_RE = re.compile(r"^(?P<indent>\s*)producer: (?P<name>.+?)\s*$")


class PersonName:
    """A person name split into the parts the natural-order form reassembles."""

    def __init__(self, given: str, family: str, suffix: str = "", paren: str = ""):
        self.given = given
        self.family = family
        self.suffix = suffix
        self.paren = paren

    @property
    def natural(self) -> str:
        parts = [self.given, self.family, self.suffix, self.paren]
        return " ".join(p for p in parts if p)

    @property
    def surname_first(self) -> str:
        family = " ".join(p for p in (self.family, self.suffix) if p)
        given = " ".join(p for p in (self.given, self.paren) if p)
        return f"{family}, {given}" if given else family


def _top_level_comma(name: str) -> int | None:
    """Index of the one comma OUTSIDE any bracket, or None if there is not
    exactly one. A comma inside brackets belongs to a description, not to the
    surname-first separator."""
    depth = 0
    found: list[int] = []
    for i, ch in enumerate(name):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            found.append(i)
    return found[0] if len(found) == 1 else None


def parse_surname_first(name: str) -> PersonName | None:
    """Split "Fravor, David" into its parts. None if the name is not comma form.

    A generational suffix sits on either side of the comma depending on which
    pass wrote the name ("Lynn III, William" and "Lynn, William III" both occur);
    either way it belongs after the family name in natural order. A trailing
    parenthetical ("(father)", "(Lue)") stays last.

    >>> parse_surname_first("Fravor, David").natural
    'David Fravor'
    >>> parse_surname_first("Greenewald Jr., John").natural
    'John Greenewald Jr.'
    >>> parse_surname_first("Elizondo, Luis D. III (father)").natural
    'Luis D. Elizondo III (father)'
    >>> parse_surname_first("David Fravor") is None
    True
    >>> parse_surname_first("Emrich (Mrs., widow of Louis Emrich)") is None
    True
    """
    paren = ""
    # The trailing parenthetical comes off FIRST, because it may itself contain a
    # comma: "Emrich (Mrs., widow of Louis Emrich)" is a name with a description
    # attached, not surname-first form. Splitting on that comma produced
    # "widow of Louis Emrich) Emrich (Mrs." in the live graph - two nodes were
    # mangled this way before the parenthetical was hoisted above the split.
    paren_match = _PARENTHETICAL_RE.search(name)
    if paren_match:
        paren = paren_match.group(1)
        name = name[: paren_match.start()].strip()

    split_at = _top_level_comma(name)
    if split_at is None:
        return None
    family_part = name[:split_at].strip()
    given_part = name[split_at + 1 :].strip()
    if not family_part or not given_part:
        return None

    suffix = ""
    family_tokens = family_part.split()
    if len(family_tokens) > 1 and _SUFFIX_RE.match(family_tokens[-1]):
        suffix = family_tokens[-1]
        family_part = " ".join(family_tokens[:-1])

    given_tokens = given_part.split()
    if given_tokens and _SUFFIX_RE.match(given_tokens[-1]) and len(given_tokens) > 1:
        suffix = suffix or given_tokens[-1]
        given_part = " ".join(given_tokens[:-1])

    if not given_part:
        return None
    return PersonName(given_part, family_part, suffix, paren)


def display_surname_first(name: str, metadata: dict | None = None) -> str:
    """Surname-first DISPLAY form ("Fravor, David"), for surname-sorted listings.

    Reads ``metadata.family_name`` - the stored name is natural order and carries
    no comma to parse. Without the field the name is returned unchanged rather
    than guessed at: "last token" is wrong for "Mohammed bin Rashid Al Maktoum"
    and for every name whose script puts the family name first.

    >>> display_surname_first("David Fravor", {"family_name": "Fravor"})
    'Fravor, David'
    >>> display_surname_first("John Greenewald Jr.", {"family_name": "Greenewald"})
    'Greenewald, John Jr.'
    >>> display_surname_first("Semjase")
    'Semjase'
    """
    family = (metadata or {}).get("family_name")
    if not family or family not in name:
        return name
    given = re.sub(r"\s+", " ", name.replace(family, "", 1)).strip()
    return f"{family}, {given}" if given else name


def _person_ids_and_renames(doc: dict) -> tuple[set[str], dict[str, PersonName]]:
    """Person node ids, and the rename each comma-form person name maps to."""
    person_ids: set[str] = set()
    renames: dict[str, PersonName] = {}
    for node in doc.get("nodes") or []:
        if node.get("type") != "person":
            continue
        person_ids.add(node.get("id"))
        parsed = parse_surname_first(node.get("name") or "")
        if parsed:
            renames[node["name"]] = parsed
    return person_ids, renames


def _expected_document(doc: dict, renames: dict[str, PersonName]) -> dict:
    """The document the rewrite must produce, used to verify the text edit."""

    def _rename(value: str) -> str:
        parsed = renames.get(value)
        return parsed.natural if parsed else value

    out = yaml.safe_load(yaml.safe_dump(doc))
    record = out.get("record")
    if isinstance(record, dict) and record.get("producer"):
        record["producer"] = _rename(record["producer"])
    for node in out.get("nodes") or []:
        if node.get("type") != "person":
            continue
        parsed = renames.get(node.get("name"))
        if not parsed:
            continue
        surname_first = node["name"]
        node["name"] = parsed.natural
        metadata = dict(node.get("metadata") or {})
        metadata["family_name"] = parsed.family
        aliases = list(metadata.get("aliases") or [])
        if surname_first not in aliases:
            aliases.append(surname_first)
        metadata["aliases"] = aliases
        node["metadata"] = metadata
    for section in ("domain_claims", "infrastructure_claims", "claims"):
        for claim in out.get(section) or []:
            speaker = claim.get("speaker")
            if isinstance(speaker, dict) and speaker.get("name"):
                speaker["name"] = _rename(speaker["name"])
            for ref in claim.get("refs") or []:
                if isinstance(ref, dict) and ref.get("name"):
                    ref["name"] = _rename(ref["name"])
    return out


def _metadata_lines(indent: str, parsed: PersonName, surname_first: str) -> list[str]:
    child = indent + "  "
    return [
        f"{indent}metadata:",
        f"{child}family_name: {parsed.family}",
        f"{child}aliases:",
        f"{child}  - {surname_first}",
    ]


def _rewrite_lines(
    lines: list[str], person_ids: set[str], renames: dict[str, PersonName]
) -> list[str]:
    out: list[str] = []
    current_id: str | None = None
    pending_metadata: tuple[str, PersonName, str] | None = None
    in_record = False
    in_nodes = False

    for line in lines:
        if pending_metadata is not None:
            indent, parsed, surname_first = pending_metadata
            metadata_match = _METADATA_LINE_RE.match(line)
            if metadata_match and metadata_match.group("indent") == indent:
                child = indent + "  "
                out.append(line)
                out.append(f"{child}family_name: {parsed.family}")
                out.append(f"{child}aliases:")
                out.append(f"{child}  - {surname_first}")
                pending_metadata = None
                continue
            out.extend(_metadata_lines(indent, parsed, surname_first))
            pending_metadata = None

        # A TOP-LEVEL KEY, not merely an unindented line. YAML admits both
        # sequence styles and the digester emits both: `  - id:` (item indented
        # under the key) and `- id:` (item at column 0). Treating the second as a
        # top-level key silently cleared in_nodes on the first node entry, so the
        # metadata block was never inserted and verification then refused the
        # whole file - which is how every column-0 digest went unmigrated.
        if line and not line[0].isspace() and not line.startswith("- "):
            in_record = line.startswith("record:")
            in_nodes = line.startswith("nodes:")

        if in_record:
            producer_match = _PRODUCER_LINE_RE.match(line)
            if producer_match:
                parsed = renames.get(producer_match.group("name"))
                if parsed:
                    indent = producer_match.group("indent")
                    out.append(f"{indent}producer: {parsed.natural}")
                    continue

        id_match = _ID_LINE_RE.match(line)
        if id_match:
            current_id = id_match.group(1)
            out.append(line)
            continue

        name_match = _NAME_LINE_RE.match(line)
        if name_match and current_id in person_ids:
            parsed = renames.get(name_match.group("name"))
            if parsed:
                indent = name_match.group("indent")
                dash = name_match.group("dash") or ""
                out.append(f"{indent}{dash}name: {parsed.natural}")
                if in_nodes:
                    pending_metadata = (indent, parsed, name_match.group("name"))
                continue
        out.append(line)

    if pending_metadata is not None:
        indent, parsed, surname_first = pending_metadata
        out.extend(_metadata_lines(indent, parsed, surname_first))
    return out


def naturalise_digest_text(text: str) -> tuple[str, int]:
    """Return the rewritten digest text and the number of person nodes renamed.

    Raises ValueError if the line rewrite does not reproduce the document the
    parsed transform expects - a mangled name is worse than a failed pass.
    """
    doc = yaml.safe_load(text) or {}
    person_ids, renames = _person_ids_and_renames(doc)
    if not renames:
        return text, 0

    rewritten = "\n".join(_rewrite_lines(text.splitlines(), person_ids, renames))
    if text.endswith("\n"):
        rewritten += "\n"

    produced = yaml.safe_load(rewritten)
    expected = _expected_document(doc, renames)
    if produced != expected:
        raise ValueError("rewritten digest does not match the expected document")
    return rewritten, len(renames)


def naturalise_digest_file(path: Path, dry_run: bool = False) -> int:
    text = path.read_text()
    rewritten, renamed = naturalise_digest_text(text)
    if renamed and not dry_run:
        path.write_text(rewritten)
    return renamed


def naturalise_digests_in_dir(
    digests_dir: Path, dry_run: bool = False
) -> dict[str, int]:
    """Rewrite every digest under ``digests_dir`` EXCEPT the variant snapshots.

    `digests/variants/` holds what each model actually emitted for a record, and
    the model comparison is only worth anything if those files are left as the
    models wrote them. Skipped here rather than left to the caller's glob,
    because pointing this at the digests repo root is the obvious mistake.
    """
    results: dict[str, int] = {}
    for path in canonical_digests(digests_dir):
        renamed = naturalise_digest_file(path, dry_run=dry_run)
        if renamed:
            results[str(path.relative_to(digests_dir))] = renamed
    return results


def _mangled_person_nodes(doc: dict) -> dict[str, str]:
    """Mangled name -> the original, for person nodes the migration should never
    have touched.

    An early ``parse_surname_first`` split on a comma INSIDE a parenthetical, so
    "Emrich (Mrs., widow of Louis Emrich)" - a name with a description attached,
    not surname-first form - came out as "widow of Louis Emrich) Emrich (Mrs.".
    The original survives verbatim in the alias the migration wrote, and the
    fixed parser is what identifies which aliases were never comma form.
    """
    mangled: dict[str, str] = {}
    for node in doc.get("nodes") or []:
        if node.get("type") != "person":
            continue
        metadata = node.get("metadata") or {}
        if not metadata.get("family_name"):
            continue
        for alias in metadata.get("aliases") or []:
            if parse_surname_first(alias) is None:
                mangled[node.get("name")] = alias
    return mangled


def _unmangled_document(doc: dict, mangled: dict[str, str]) -> dict:
    out = yaml.safe_load(yaml.safe_dump(doc))

    def _restore(value: str) -> str:
        return mangled.get(value, value)

    record = out.get("record")
    if isinstance(record, dict) and record.get("producer"):
        record["producer"] = _restore(record["producer"])
    for node in out.get("nodes") or []:
        original = mangled.get(node.get("name"))
        if not original:
            continue
        node["name"] = original
        metadata = dict(node.get("metadata") or {})
        metadata.pop("family_name", None)
        aliases = [a for a in metadata.get("aliases") or [] if a != original]
        if aliases:
            metadata["aliases"] = aliases
        else:
            metadata.pop("aliases", None)
        if metadata:
            node["metadata"] = metadata
        else:
            node.pop("metadata", None)
    for section in ("domain_claims", "infrastructure_claims", "claims"):
        for claim in out.get(section) or []:
            speaker = claim.get("speaker")
            if isinstance(speaker, dict) and speaker.get("name"):
                speaker["name"] = _restore(speaker["name"])
            for ref in claim.get("refs") or []:
                if isinstance(ref, dict) and ref.get("name"):
                    ref["name"] = _restore(ref["name"])
    return out


def _unmangle_lines(lines: list[str], mangled: dict[str, str]) -> list[str]:
    """Rename back, and delete only the metadata the migration itself injected."""
    out: list[str] = []
    drop_until_indent: str | None = None
    original_for_block: str | None = None

    for line in lines:
        if drop_until_indent is not None:
            stripped = line.strip()
            indent = line[: len(line) - len(line.lstrip())]
            # The injected keys sit UNDER metadata:, which is itself a sibling of
            # name: at the same indent - so "deeper than the name line" is the
            # wrong test and skipped straight past them. Scan until the node ends
            # and drop by shape instead.
            if (
                stripped.startswith("family_name:")
                or stripped.startswith("aliases:")
                or stripped == f"- {original_for_block}"
            ):
                continue
            if stripped.startswith("- ") or len(indent) < len(drop_until_indent):
                drop_until_indent = None
                original_for_block = None

        name_match = _NAME_LINE_RE.match(line)
        producer_match = _PRODUCER_LINE_RE.match(line)
        match = name_match or producer_match
        if match and match.group("name") in mangled:
            original = mangled[match.group("name")]
            indent = match.group("indent")
            if producer_match:
                out.append(f"{indent}producer: {original}")
                continue
            dash = name_match.group("dash") or ""
            out.append(f"{indent}{dash}name: {original}")
            drop_until_indent = indent
            original_for_block = original
            continue
        out.append(line)

    return _drop_empty_metadata(out)


def _drop_empty_metadata(lines: list[str]) -> list[str]:
    """Remove a ``metadata:`` key left with no children by the restore."""
    out: list[str] = []
    for i, line in enumerate(lines):
        match = _METADATA_LINE_RE.match(line)
        if match:
            indent = match.group("indent")
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            nxt_indent = nxt[: len(nxt) - len(nxt.lstrip())]
            if not nxt.strip() or len(nxt_indent) <= len(indent):
                continue
        out.append(line)
    return out


def unmangle_digest_text(text: str) -> tuple[str, int]:
    """Undo the comma-inside-parentheses mangling. Returns text and count fixed.

    Raises ValueError if the line rewrite does not reproduce the expected
    document, on the same principle as the forward migration: a half-restored
    name is worse than a refused file.
    """
    doc = yaml.safe_load(text) or {}
    mangled = _mangled_person_nodes(doc)
    if not mangled:
        return text, 0

    rewritten = "\n".join(_unmangle_lines(text.splitlines(), mangled))
    if text.endswith("\n"):
        rewritten += "\n"

    produced = yaml.safe_load(rewritten)
    expected = _unmangled_document(doc, mangled)
    if produced != expected:
        raise ValueError("restored digest does not match the expected document")
    return rewritten, len(mangled)


def unmangle_digests_in_dir(digests_dir: Path, dry_run: bool = False) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in canonical_digests(digests_dir):
        text = path.read_text()
        rewritten, fixed = unmangle_digest_text(text)
        if fixed:
            results[str(path.relative_to(digests_dir))] = fixed
            if not dry_run:
                path.write_text(rewritten)
    return results
