"""Import extraction markdown files into the database.

This is a deterministic step with no AI involvement. The extraction
markdown is the source of truth; the database is derived from it.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from assimilator.database import (
    delete_claim,
    find_node_by_name,
    get_record,
    get_record_by_content_hash,
    get_record_by_title,
    get_record_claim_hashes,
    insert_alias,
    link_record_nodes,
    insert_claim,
    insert_node,
    insert_record,
    update_claim_chain,
    update_claim_entailment,
    update_claim_hash,
)
from assimilator.matching import (
    is_a_description,
    is_fuller_person_name,
    match_node,
    normalise_node_name,
)
from anomalica_common.digest import claim_hash
from anomalica_common.digest.models import Claim, Node, ProvenanceChain, Record


# Patterns that mark a node name as unusable:
# - "(redacted)" / "(REDACTED)" anywhere in the name (no person to extract)
# - trailing "(person)" / "(organisation)" / "(document)" / "(matter)" / etc.
#   (type-in-parens artefact - the model misread the acronym-parens rule)
_REDACTED_RE = re.compile(r"\([Rr][Ee][Dd][Aa][Cc][Tt][Ee][Dd]\)")
# `topic` and `project` were missing from this list, so "Levitation (topic)" and
# "Blue Book (project)" passed a rule written to catch exactly them - the same
# half-closed shape as the leading form having no rule at all. Nothing in the live
# corpus carries either suffix, so adding them rejects nothing that exists today.
_TYPE_SUFFIX_RE = re.compile(
    r"\s*\((person|organisation|place|event|matter|object|document|concept|record"
    r"|topic|project)\)\s*$",
    re.IGNORECASE,
)

# Month-name -> 2-digit ISO month, used to rewrite "14 November 2004" forms in
# node names to "2004-11-14" deterministically when the model leaves them as
# prose despite the rule.
_MONTH_NAMES = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}
_SPELLED_DATE_DAY_FIRST_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_SPELLED_DATE_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)


def _merge_record_metadata(
    conn: sqlite3.Connection, record_id: str, extra: dict
) -> None:
    """Merge keys into a record's metadata JSON without disturbing the rest."""
    row = conn.execute(
        "SELECT metadata FROM records WHERE id = ?", (record_id,)
    ).fetchone()
    current = json.loads(row[0]) if row and row[0] else {}
    current.update(extra)
    conn.execute(
        "UPDATE records SET metadata = ? WHERE id = ?",
        (json.dumps(current), record_id),
    )


_TYPE_PREFIX_RE = re.compile(
    r"^(person|organisation|project|place|event|object|document|topic|concept|matter)"
    r"\s*:\s*",
    re.IGNORECASE,
)


def strip_type_prefix(name: str) -> str:
    """Remove a leading "topic: " / "object: " written into the name itself.

    The type belongs in the type field. The model occasionally writes it into the
    name as well, and the result is a node - and a page title - reading
    "topic: telepathy". 185 of them had accumulated, one at 31 references.

    Stripped rather than rejected, because the name after the prefix is correct:
    rejecting would drop the node and its claims for a cosmetic fault. Note the
    trailing form is REJECTED instead ("Foo (topic)" via _TYPE_SUFFIX_RE) because
    that one marks a model that misread the acronym-parens rule, which is a
    different failure. This gap existed because only the trailing form had a rule.

    >>> strip_type_prefix("topic: levitation")
    'levitation'
    >>> strip_type_prefix("Project Blue Book")
    'Project Blue Book'
    >>> strip_type_prefix("Person: A Study")
    'A Study'
    """
    return _TYPE_PREFIX_RE.sub("", name).strip() or name


def _node_name_is_unusable(name: str) -> str | None:
    """Return a short reason string if the name should be rejected, else None."""
    if _REDACTED_RE.search(name):
        return "contains '(redacted)'"
    if _TYPE_SUFFIX_RE.search(name):
        return "ends with parens-type suffix like '(person)'"
    return None


def _normalise_spelled_dates(name: str) -> str:
    """Rewrite '14 November 2004' -> '2004-11-14' and 'November 2004' -> '2004-11'."""

    def _sub_day(m: "re.Match[str]") -> str:
        day = int(m.group(1))
        month = _MONTH_NAMES[m.group(2).lower()]
        year = m.group(3)
        return f"{year}-{month}-{day:02d}"

    def _sub_my(m: "re.Match[str]") -> str:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = m.group(2)
        return f"{year}-{month}"

    out = _SPELLED_DATE_DAY_FIRST_RE.sub(_sub_day, name)
    out = _SPELLED_DATE_MONTH_YEAR_RE.sub(_sub_my, out)
    return out


def _codename_roots(codenames: set[str]) -> set[str]:
    """Extract the distinguishing prefix word of each codename.

    The pre-pass tends to return codenames in numbered form ("FASTEAGLE 01",
    "FASTEAGLE 02"); rejecting a node named "FASTEAGLE Flight" needs the
    root token "FASTEAGLE". For a single-token codename ("Poison") the root
    is the codename itself. Codenames with a leading non-alphabetic token
    are skipped.
    """
    roots = set()
    for cn in codenames:
        if not cn:
            continue
        tokens = cn.strip().split()
        first = tokens[0] if tokens else ""
        # Only treat distinctive uppercase / proper-noun tokens as roots, to
        # avoid spurious matches on common words.
        if first and (first.isupper() or first[0].isupper()) and len(first) >= 4:
            roots.add(first)
        # Always include the full codename string too (covers "Tic Tac" cases
        # where the multi-token form is the distinguishing identifier).
        roots.add(cn)
    return roots


def _build_terminology_enforcers(terminology: dict | None):
    """Build per-document enforcers from the terminology pre-pass result.

    Returns (codename_roots_set, acronym_expansions, _normalise_spelled_dates).
    - codename_roots_set: distinctive root words extracted from codenames so
      "FASTEAGLE 01" codename also rejects "FASTEAGLE Flight" nodes.
    - acronym_expansions: dict of ACRONYM -> "Full Form (ACRONYM)" filtered
      to only safe-to-substitute acronyms (no parens in expansion - those
      look descriptive, not lexical, e.g. "SSN-724 -> USS Louisville (nuclear
      fast attack submarine hull number)").
    """
    if not terminology:
        return set(), {}, _normalise_spelled_dates
    raw_codenames = {
        (c.get("codename") or "").strip()
        for c in (terminology.get("codenames") or [])
        if c.get("codename")
    }
    codename_roots = _codename_roots(raw_codenames)
    expansions = {}
    for a in terminology.get("acronyms") or []:
        acro = (a.get("acronym") or "").strip()
        full = (a.get("expansion") or "").strip()
        if not (acro and full):
            continue
        # Skip entries that aren't real acronym expansions:
        # - "SSN-724 -> USS Louisville (nuclear fast attack submarine hull number)"
        #   the parens indicate descriptive text; blind substitution duplicates
        #   ship names. Anything matching "Full Name (descriptive)" pattern is
        #   not a simple acronym expansion - skip.
        if "(" in full:
            continue
        # Skip designator-style keys (CSG-11, VFA-41); these are handled by
        # the global squadron normaliser. Keeping them here just produces
        # overlap with the bare-acronym entries (CSG vs CSG-11).
        if re.search(r"-\d", acro):
            continue
        expansions[acro] = f"{full} ({acro})"
    return codename_roots, expansions, _normalise_spelled_dates


def _substitute_outside_parens(
    name: str, acro: str, full_form: str, protected: "list[tuple[int, int]]" = ()
) -> str:
    """Expand an acronym only where it is NOT already inside a parenthetical.

    An acronym inside brackets is almost always the short form of what the
    brackets are glossing, so expanding it nests one expansion inside another:
    "Helicopter Antisubmarine Squadron 6 (HS-6)" became "Helicopter
    Antisubmarine Squadron 6 (Helicopter Anti-Submarine Squadron 6 (HS-6))",
    and "Mutual UFO Network (MUFON)" became "Mutual Unidentified Flying Object
    (UFO) Network (MUFON)" - MUFON's own expansion corrupted from the inside.

    The existing guards cannot catch this. They ask whether THIS acronym is
    already expanded; here it is a DIFFERENT acronym's expansion being damaged,
    and the one being substituted is genuinely unexpanded.

    `protected` additionally covers spans that are ANOTHER acronym's expansion
    written out in full. "Mutual UFO Network (MUFON)" is entirely MUFON's own
    expansion, and the UFO inside it sits outside the brackets - so bracket
    depth alone does not save it, and expanding there rewrites the very string
    the glossary defines.

    Expansion outside the brackets is untouched, so "Mexico City UFO sightings"
    still expands.
    """
    pattern = re.compile(rf"\b{re.escape(acro)}\b(?!-\d)")
    # Walk the string once, tracking bracket depth, and substitute the first
    # match that sits at depth zero and outside any protected span.
    depth, result, i, done = 0, [], 0, False
    while i < len(name):
        char = name[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if not done and depth == 0:
            match = pattern.match(name, i)
            if match and any(a <= i < b for a, b in protected):
                match = None
            if match:
                result.append(full_form)
                i = match.end()
                done = True
                continue
        result.append(char)
        i += 1
    return "".join(result)


def _apply_doc_terminology(
    name: str, codenames: set[str], expansions: dict[str, str], node_type: str = ""
) -> tuple[str, str | None]:
    """Apply per-document terminology to a node name.

    Returns (rewritten_name, reject_reason). reject_reason is non-None if the
    name should be rejected (e.g. matches a codename).

    A PERSON IS EXEMPT FROM THE EXPANSIONS, though never from the codename
    rejection. The document's glossary says what a term means in its prose; a
    person's name is not a term. "UAP Gerb" is the handle of a real UAP
    researcher, and the whole-word substitution stored it as "Unidentified
    Aerial Phenomena (UAP) Gerb" - which reached the page gate as page-worthy
    with 36 claims and 8 independent sources, and was read downstream as a
    corrupted merge of two people.
    """
    if not name:
        return name, None

    # Reject if the name contains a codename as a whole token. Codenames
    # may appear inside descriptive prose claim text but never as the
    # canonical identifier of a node.
    for cn in codenames:
        if not cn:
            continue
        if re.search(rf"\b{re.escape(cn)}\b", name):
            return name, f"contains codename '{cn}'"

    # Expand per-document acronyms in the name. Apply longer acronyms first
    # so "CSG-11" wins over bare "CSG". Use a negative lookahead so the bare
    # acronym never matches inside a hyphen-numbered designator like
    # "CSG-11" or "VFA-41" - those are handled by the global squadron
    # normaliser, and matching "CSG" inside "(CSG-11)" creates nested-parens
    # nonsense.
    out = name
    if node_type == "person":
        return _normalise_spelled_dates(out), None
    for acro in sorted(expansions, key=len, reverse=True):
        full_form = expansions[acro]
        # Same test as the global expander, and for the same reason: an exact
        # match against one wording misses the source's own variant, so the bare
        # acronym gets expanded INSIDE the parenthetical that is already there.
        # A trailing "(ACRO)" is the evidence of expansion whatever precedes it.
        if full_form in out or f"({acro})" in out:
            continue  # already expanded
        # Spans that are some OTHER acronym's expansion, written out in full.
        protected = []
        for other, other_full in expansions.items():
            if other == acro or not other_full:
                continue
            start = out.find(other_full)
            while start != -1:
                protected.append((start, start + len(other_full)))
                start = out.find(other_full, start + 1)
        out = _substitute_outside_parens(out, acro, full_form, protected)

    # Normalise spelled-out months to ISO.
    out = _normalise_spelled_dates(out)
    return out, None


# Path to ingests, used to look up content_hash from the friendly
# filename of a digest YAML. Override via env var when running outside the
# container.
_INGESTS_DIR = os.environ.get(
    "ANOMALICA_INGESTS_DIR",
    "/home/nonroot/ingests",
)


_STORE_HASH_RE = re.compile(r"^([0-9a-f]{64})(?:\.|$)")


def _content_hash_of(target: Path, declared: str | None) -> str | None:
    """The record's content hash, taking the STORE FILENAME over the frontmatter.

    A store record is addressed BY its hash, so the filename is the hash by
    construction and the frontmatter field is a copy that can be wrong. It is
    wrong today: `records/2007-06-20-web-project-serpo.md` is a legacy loose file
    declaring `content_hash: a480652e...`, which is a DIFFERENT record entirely
    (an unrelated interview). Trusting the declaration there would stamp a claim's
    source hash onto the wrong document and point every workbench deep link at it.

    Falls back to the declaration for a file that is not in the store, since a
    loose record has no filename hash to read.
    """
    match = _STORE_HASH_RE.match(target.name)
    if not match:
        return declared
    from_name = "sha256:" + match.group(1)
    if declared and declared != from_name:
        log_line = (
            f"  WARNING: {target.name} declares content_hash {declared} but is "
            f"stored at {from_name} - using the stored hash"
        )
        print(log_line)
    return from_name


# Record-block fields that are not columns but are worth holding: they answer
# questions about the record's PROVENANCE and handling rather than its content.
# `review` above all - it is what lets a consumer tell material Mark reviewed
# from material nobody has, which is the whole basis on which unreviewed records
# are allowed into the graph at all.
# THE RECORD BLOCK PASSES THROUGH WHOLE, minus what is already a column. This
# was an ALLOW-LIST of five field names, and an allow-list drops the next field
# added: `copyright_status` landed in the digest today and would have arrived
# nowhere, silently, exactly as `review` once did. digest-format.md now makes
# pass-through the normative reader rule for precisely this reason - a parser
# that enumerates what it knows about is correct on the day it ships and quietly
# wrong afterwards.
#
# Deny-listed rather than allow-listed: these are stored as columns on records,
# so copying them into metadata would give one fact two homes that can disagree.
_RECORD_COLUMN_FIELDS = frozenset(
    {
        "id",
        "title",
        "date",
        "reference",
        "content_hash",
        "friendly_name",
        "producer",
    }
)

# Doc-level fields worth keeping beside the record. Both answer "what would I
# have to hold constant to reproduce this claim":
#   pre_digest  names the exact text a span indexes into. Without it a span
#               cannot be resolved with confidence - a corpus-wide "directional
#               drift" across 1,589 claims turned out to be quotes resolved in
#               body space against offsets in pre-digest space, and the frame was
#               recoverable ONLY because the digest carried this. It is also the
#               staleness detector: re-materialise the record, compare the sha.
#   prompts     identifies what produced a claim, which is the precondition for
#               any comparison across prompt versions.
#   run_kind    production vs comparison run. ABSENT means the digest predates
#               the field and must read UNKNOWN - defaulting it to "production"
#               would silently promote comparison artefacts into the canonical
#               set. Same three-state rule as review.
#
# ai_usage is DELIBERATELY EXCLUDED and must stay excluded. Per-record usage is
# kept in the digest front matter and the operations ledger and is explicitly not
# surfaced publicly (operating rules, amended 2026-06-29). The graph feeds the
# public site, so storing it here puts billing data one join from a renderer and
# the failure mode is a template that helpfully shows it. It already lives in two
# non-public places; that is the right number.
_DOC_METADATA_FIELDS = ("pre_digest", "prompts", "run_kind")


def _record_metadata(fm: dict) -> dict | None:
    """The record-block fields worth storing, or None if the digest carries none.

    THREE STATES, NEVER TWO. `review.state` is human | machine | none, and the
    field being ABSENT is a fourth thing: the digest predates the stamp. Absent
    must read as UNKNOWN, not as unreviewed - the same rule as a missing
    provenance chain never reading as independent. So an absent field is simply
    not stored, and a consumer asking "was this reviewed" must test for
    state == "human", never for state != "none".

    That READ rule is pinned in tests/test_absent_is_not_a_value.py, not left to
    this docstring, because a docstring cannot go red - and because SQL and
    Python disagree about absence in opposite directions: `NULL != 'variant'` is
    NULL so the row vanishes, while `None != "variant"` is True so the row is
    promoted. One wrong predicate, two different silent wrong answers.
    """
    block = fm.get("record") or {}
    out = {
        k: v
        for k, v in block.items()
        if k not in _RECORD_COLUMN_FIELDS and v is not None
    }
    out.update({k: fm[k] for k in _DOC_METADATA_FIELDS if fm.get(k) is not None})
    return out or None


def _materialise_locally(
    conn: sqlite3.Connection, lookup_conn: sqlite3.Connection, node_id: str
) -> str:
    """Ensure a node matched in ANOTHER database exists in this one; return its
    local id.

    claim_node_refs carries a foreign key to nodes, so a ref resolved against the
    other database is only usable once the node exists HERE. Sharing the id across
    databases is preferred; a retired row may already hold it, in which case the
    local node of that name wins, else a fresh id. Returning the other database's
    id without materialising it is what fails the foreign key.
    """
    if lookup_conn is conn:
        return node_id
    row = lookup_conn.execute(
        "SELECT name, node_type, metadata FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row is None:
        return node_id
    name, node_type, metadata = row
    local = find_node_by_name(conn, name, node_type)
    if local:
        return local.id
    taken = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
    local_id = str(uuid.uuid4()) if taken else node_id
    insert_node(
        conn,
        Node(
            id=local_id,
            node_type=node_type,
            name=name,
            metadata=json.loads(metadata) if metadata else None,
        ),
    )
    return local_id


def _lookup_ingest_metadata(
    record_title: str, source_path: object
) -> tuple[str | None, str | None]:
    """Given an extraction's metadata, find the matching ingest's
    content_hash and friendly filename stem.

    The digest YAML filename equals the ingest's friendly name (mounted via
    ingests/by-name/<friendly>.md as a symlink into store/{hash}.md).
    We resolve by exact filename match first, then by record_title scan as a
    fallback.
    """
    from pathlib import Path as _P

    # The records directory: the configured/container path first, then - so a
    # host-side rebuild does not silently lose content_hash when the container
    # path is absent - the ingests dir derived from the digest's own location
    # (<root>/digests/<stem>.yaml -> <root>/ingests/by-name).
    candidate_dirs = [_P(_INGESTS_DIR) / "by-name"]
    if source_path:
        derived = (
            _P(str(source_path)).resolve().parent.parent.parent / "ingests" / "by-name"
        )
        candidate_dirs.append(derived)
    records_dir = next((d for d in candidate_dirs if d.exists()), None)
    if records_dir is None:
        return None, None

    candidate_stem: str | None = None
    if source_path:
        sp = _P(str(source_path))
        # digest YAMLs are named "<stem>.yaml"; ingest symlinks are
        # "<stem>.md". Same stem.
        candidate_stem = sp.stem

    if candidate_stem:
        ingest = records_dir / f"{candidate_stem}.md"
        if ingest.is_symlink() or ingest.exists():
            try:
                target = ingest.resolve()
                with open(target) as f:
                    frontmatter_text = f.read().split("---", 2)[1]
                import yaml as _y

                fm = _y.safe_load(frontmatter_text) or {}
                ch = _content_hash_of(target, fm.get("content_hash"))
                return ch, candidate_stem
            except (OSError, IndexError):
                pass

    # Fallback: scan every ingest record/symlink for a frontmatter title
    # match. Slow on a big corpus but unambiguous. Sorted so that when two
    # records share a title the same one wins every rebuild (determinism;
    # order-sensitivity is latent across the import path).
    import yaml as _y

    for ingest in sorted(records_dir.glob("*.md")):
        try:
            target = ingest.resolve() if ingest.is_symlink() else ingest
            with open(target) as f:
                head = f.read(8192)
            if "title:" not in head:
                continue
            frontmatter_text = head.split("---", 2)[1] if "---" in head else ""
            fm = _y.safe_load(frontmatter_text) or {}
            if fm.get("title") == record_title:
                return fm.get("content_hash"), ingest.stem
        except (OSError, IndexError):
            continue
    return None, None


_ENTAILMENT_LABELS = frozenset({"entails", "neutral", "contradicts"})
_ENTAILMENT_PREMISES = frozenset({"quote", "window"})


def _entailment_block(block: object) -> dict | None:
    """The digester's per-claim entailment block, validated, or None.

    Shape (pinned with the digester 2026-09-02): {label: entails|neutral|
    contradicts, score: probability of THAT label in [0, 1], model: id,
    premise: quote|window}. The hypothesis is the claim text; the premise is
    the excerpt alone ("quote") or, when that alone is neutral, the record text
    around it ("window") - and an entails-by-window is the weaker verdict. An
    absent block means not assessed; a present but malformed one is also stored
    as not assessed and COUNTED by the caller, so a digester regression shows in
    the import summary instead of arriving as a quiet run of nulls.
    """
    if not isinstance(block, dict):
        return None
    label, score, model = block.get("label"), block.get("score"), block.get("model")
    premise = block.get("premise")
    if label not in _ENTAILMENT_LABELS or not isinstance(model, str) or not model:
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    if not 0.0 <= float(score) <= 1.0:
        return None
    if premise not in _ENTAILMENT_PREMISES:
        return None
    return {"label": label, "score": float(score), "model": model, "premise": premise}


def import_extraction(
    conn: sqlite3.Connection,
    parsed: dict,
    section: str = "domain",
    lookup_conns: list[sqlite3.Connection] | None = None,
    source_path: str | None = None,
    on_progress: callable = None,
) -> dict:
    """Import a parsed extraction markdown into the database.

    Args:
        conn: database to write to
        parsed: output of parse_extraction_markdown()
        section: "domain" or "infrastructure" - which claims section to import
        lookup_conns: additional databases to check for existing nodes
        on_progress: callback for status messages

    Returns:
        dict with counts: nodes_created, nodes_matched, claims_created, record_id
    """
    log = on_progress or (lambda _: None)
    all_conns = [conn] + (lookup_conns or [])
    fm = parsed["frontmatter"]

    counts = {
        "nodes_created": 0,
        "nodes_matched": 0,
        "nodes_rejected": 0,
        "nodes_described": 0,
        "claims_created": 0,
        "claims_carried": 0,
        "claims_deleted": 0,
        "claims_rejected": 0,
        "claims_assessed": 0,
        "entailment_malformed": 0,
        "record_id": None,
    }

    # Drop unusable nodes (redacted / type-in-parens) before any matching.
    # Track their ids so claims referencing them can be dropped too.
    # Also normalise known unexpanded acronyms (VFA-41 -> Strike Fighter
    # Squadron 41 (VFA-41)) deterministically, since the extraction model
    # ignores rule 4b for these. Then apply the per-document terminology
    # from the pre-pass: reject codename-named nodes, expand per-doc
    # acronyms, normalise spelled-out months in node names.
    codenames, doc_acronyms, _ = _build_terminology_enforcers(parsed.get("terminology"))

    rejected_md_ids: set[str] = set()
    name_rewrites: dict[str, str] = {}
    usable_nodes = []
    for node_def in parsed["nodes"]:
        # A described actor is dropped as a NODE but keeps its claims: the
        # identity does not exist, the testimony does. Nothing further is needed
        # for its refs - they resolve by name, so they simply find nothing, and
        # the claim survives one ref lighter.
        if is_a_description(node_def["name"]):
            log(f"  Description, not a node: {node_def['name']}")
            counts["nodes_described"] += 1
            continue
        reason = _node_name_is_unusable(node_def["name"])
        if reason:
            log(f"  Rejected node: {node_def['name']} ({reason})")
            rejected_md_ids.add(node_def["id"])
            counts["nodes_rejected"] += 1
            continue
        original_name = node_def["name"]
        # First the global expansions (squadron designators etc.). The node type
        # goes in because a person is exempt: an acronym inside a person's name
        # is part of the name, not a term to expand.
        normalised = strip_type_prefix(original_name)
        normalised = normalise_node_name(normalised, node_def.get("type"))
        # Then the per-document codename/acronym/date enforcement.
        normalised, doc_reason = _apply_doc_terminology(
            normalised, codenames, doc_acronyms, node_def.get("type") or ""
        )
        if doc_reason:
            log(f"  Rejected node: {original_name} ({doc_reason})")
            rejected_md_ids.add(node_def["id"])
            counts["nodes_rejected"] += 1
            continue
        if normalised != original_name:
            log(f"  Normalised: {original_name} -> {normalised}")
            node_def = {**node_def, "name": normalised}
            name_rewrites[original_name] = normalised
        usable_nodes.append(node_def)
    parsed = {**parsed, "nodes": usable_nodes}

    # Apply the same rewrites to refs and speaker fields inside claims so
    # they still resolve against the (now-normalised) node names.
    def _apply_rewrites(claims):
        rewritten = []
        for c in claims:
            c = dict(c)
            if c.get("speaker") in name_rewrites:
                c["speaker"] = name_rewrites[c["speaker"]]
            c["node_references"] = [
                name_rewrites.get(r, r) for r in c.get("node_references", [])
            ]
            rewritten.append(c)
        return rewritten

    if name_rewrites:
        parsed = {
            **parsed,
            "domain_claims": _apply_rewrites(parsed.get("domain_claims", [])),
            "infrastructure_claims": _apply_rewrites(
                parsed.get("infrastructure_claims", [])
            ),
        }

    # Create or find record. THE ID IS THE IDENTITY, then the content hash; the
    # title is a fallback for a digest that carries neither. Looking up by title
    # alone meant a re-digest that refreshed a record's title while keeping its
    # id looked like a new record and collided on the primary key - the 2026-09-02
    # record-block refresh renamed "Project Serpo" to its interview title and
    # the entailment backfill import stopped at digest 18 of 108.
    record_title = fm.get("record_title", "Untitled")
    existing_record = get_record(conn, fm["record_id"]) if fm.get("record_id") else None
    if existing_record is None and fm.get("content_hash"):
        existing_record = get_record_by_content_hash(conn, fm["content_hash"])
    if existing_record is None:
        existing_record = get_record_by_title(conn, record_title)
    if existing_record:
        record = existing_record
        # Refresh the record on re-import: metadata (insert-only left every record
        # without a review state forever, which is how the provenance chain stayed
        # NULL through a full re-digest) and the title, reference and date, which
        # the digester's record-block refresh is entitled to change.
        refreshed = _record_metadata(fm)
        conn.execute(
            "UPDATE records SET metadata = COALESCE(?, metadata), title = ?, "
            "reference = COALESCE(?, reference), date = COALESCE(?, date) WHERE id = ?",
            (
                json.dumps(refreshed) if refreshed else None,
                record_title,
                fm.get("record_reference"),
                str(fm["record_date"]) if fm.get("record_date") else None,
                record.id,
            ),
        )
        if record.title != record_title:
            log(f"  Existing record renamed: {record.title!r} -> {record_title!r}")
        log(f"  Existing record: {record_title} [{record.id[:8]}]")
    else:
        # Resolve the ingest content_hash and friendly_name for this record
        # so downstream consumers (assembler, workbench) can link claim ->
        # source-record verifiably. The digest YAML may carry content_hash
        # directly (newer emissions) or we look it up via the friendly
        # filename match against ingests/by-name/ (the deterministic
        # backfill for older YAMLs).
        content_hash = fm.get("content_hash")
        friendly_name = fm.get("friendly_name")
        if not content_hash:
            content_hash, friendly_name = _lookup_ingest_metadata(
                fm.get("record_title", ""), source_path
            )
        if not content_hash:
            # Never silent: a null content_hash breaks workbench deep-links on
            # every claim/node sourced from this record. Surface it loudly so a
            # mis-pointed ingests dir is diagnosed, not mistaken for needing a
            # re-digest. Set ANOMALICA_INGESTS_DIR if running outside the container.
            log(
                f"  WARNING: no content_hash for {record_title!r} - workbench "
                f"links will be absent (checked ANOMALICA_INGESTS_DIR="
                f"{_INGESTS_DIR!r} and the digest-relative ingests dir)"
            )
        record = insert_record(
            conn,
            Record(
                id=fm.get("record_id"),
                title=record_title,
                reference=fm.get("record_reference"),
                date=str(fm["record_date"]) if fm.get("record_date") else None,
                content_hash=content_hash,
                friendly_name=friendly_name,
                metadata=_record_metadata(fm),
            ),
        )
        log(f"  Record: {record.title} [{record.id[:8]}]")
    counts["record_id"] = record.id

    # Build node map: name -> id (from the markdown's node definitions)
    # Match against existing nodes in database(s), create new ones as needed
    node_name_to_id = {}
    node_id_by_md_id = {}

    for node_def in parsed["nodes"]:
        name = node_def["name"]
        node_type = node_def["node_type"]
        md_id = node_def["id"]

        # Try to find existing node across all databases
        found = False
        for lookup_conn in all_conns:
            m = match_node(lookup_conn, name, node_type, record_id=record.id)
            if m:
                node_id = m[0]
                lookup_id = node_id

                # The node must EXIST in the target database, because claim refs
                # carry a foreign key to nodes. Matching it in the other database
                # is not enough. Sharing the id across databases is preferred, but
                # a retired row here may already hold it - so fall back to the
                # local node of that name, then to a fresh id. Skipping the insert
                # (the tempting guard) leaves refs pointing at a row that does not
                # exist here and fails the foreign key instead of the primary one.
                if lookup_conn is not conn:
                    local = find_node_by_name(conn, name, node_type)
                    if local:
                        node_id = local.id
                    else:
                        taken = conn.execute(
                            "SELECT 1 FROM nodes WHERE id = ?", (node_id,)
                        ).fetchone()
                        node_id = str(uuid.uuid4()) if taken else node_id
                        insert_node(
                            conn,
                            Node(
                                id=node_id,
                                node_type=node_type,
                                name=name,
                                metadata=node_def.get("metadata"),
                            ),
                        )

                node_name_to_id[name] = node_id
                node_id_by_md_id[md_id] = node_id

                existing_name = lookup_conn.execute(
                    "SELECT name FROM nodes WHERE id = ?", (lookup_id,)
                ).fetchone()[0]
                if m[1] in ("fuzzy", "acronym") and name != existing_name:
                    insert_alias(conn, name, node_id)
                    log(
                        f"  Matched ({m[1]}): {name} -> {existing_name} [{node_id[:8]}]"
                    )
                else:
                    log(f"  Existing node: {name} ({node_type}) [{node_id[:8]}]")
                # The fuller spelling of a person wins the canonical name, whatever
                # order the records arrived in: "Kevin Day" over "K. Day", not the
                # reverse. Derived, not curated - a rebuild replays the imports and
                # reaches the same name - so no ledger entry. The shorter form
                # stays as an alias, so it still resolves.
                if node_type == "person" and is_fuller_person_name(name, existing_name):
                    conn.execute(
                        "UPDATE nodes SET name = ? WHERE id = ?", (name, node_id)
                    )
                    insert_alias(conn, existing_name, node_id)
                    node_name_to_id[existing_name] = node_id
                    log(f"  Promoted: {existing_name} -> {name} [{node_id[:8]}]")

                counts["nodes_matched"] += 1
                found = True
                break

        if not found:
            # The digest-local id is a SUGGESTION, not a reservation. It is
            # already taken whenever this digest was imported before and the node
            # has since been RETIRED by a merge: the row still exists, so
            # match_node (live nodes only) does not find it and the insert
            # collides on the primary key. Mint a fresh id in that case - the
            # graph's id is authoritative and the digest's is per-emission anyway.
            taken = conn.execute(
                "SELECT 1 FROM nodes WHERE id = ?", (md_id,)
            ).fetchone()
            new_id = str(uuid.uuid4()) if taken else md_id
            node = insert_node(
                conn,
                Node(
                    id=new_id,
                    node_type=node_type,
                    name=name,
                    metadata=node_def.get("metadata"),
                ),
            )
            node_name_to_id[name] = node.id
            node_id_by_md_id[md_id] = node.id
            log(f"  New node: {name} ({node_type}) [{node.id[:8]}]")
            counts["nodes_created"] += 1

        # Aliases declared in the digest are graph aliases. Written here rather
        # than left in metadata because a rebuild wipes the graph and only the
        # digests survive: the surname-first person form (node-types.md, kept so
        # last-first input still resolves) would be lost on the next rebuild if
        # it lived only in a database row.
        for alias in (node_def.get("metadata") or {}).get("aliases") or []:
            if alias and alias != name:
                insert_alias(conn, alias, node_name_to_id[name])

    # Which nodes this record's digest DECLARED - kept separately from the claim
    # edges, because the two diverge and the divergence matters. See the
    # record_nodes table comment.
    link_record_nodes(conn, record.id, list(node_name_to_id.values()))

    # Link record producer
    producer_name = fm.get("record_producer")
    if producer_name and producer_name in node_name_to_id:
        conn.execute(
            "UPDATE records SET producer_id = ? WHERE id = ?",
            (node_name_to_id[producer_name], record.id),
        )
    elif producer_name:
        # CLEAR THE OLD LINK. producer_id is only ever set, never cleared, so a
        # record whose producer stopped resolving kept pointing at whatever it
        # resolved to last time - here, the retired node the bracketed form was
        # written to replace. The record then had both a described producer and a
        # producer_id into a retired row.
        conn.execute("UPDATE records SET producer_id = NULL WHERE id = ?", (record.id,))
    if producer_name and is_a_description(producer_name):
        # A DESCRIBED PRODUCER SURVIVES ITS FAILED LOOKUP. It resolves to no node,
        # correctly, but dropping it would leave producer_id NULL - and NULL
        # already means "no author recorded" on 60 of 89 records. Those are
        # materially different: a source whose author was deliberately withheld
        # carries different evidential weight from one whose authorship we simply
        # never captured, and collapsing them loses that signal entirely. The
        # bracketed string IS the representation; it needs no node, no
        # placeholder and no extra column, only somewhere to survive.
        _merge_record_metadata(conn, record.id, {"producer": producer_name})

    # Import claims from the specified section. On a re-import of an existing
    # record the record's claim set is RECONCILED, not blindly re-inserted:
    # each incoming claim is matched to a prior one by claim_hash (carrying its
    # uuid and created_at forward), genuinely-new claims are inserted, and prior
    # claims whose hash no longer appears are deleted. Without this the
    # digester's per-emission claim uuids would both duplicate unchanged claims
    # and make every re-digest read as 100% changed (a page-staleness false
    # positive). claim_hash is the shared canonical hash from anomalica-common,
    # computed here because only the importer holds the resolved graph ids.
    if section == "infrastructure":
        claims = parsed["infrastructure_claims"]
    else:
        claims = parsed["domain_claims"]

    resolved_claims: list[tuple[Claim, str, dict | None]] = []
    for claim_def in claims:
        # Resolve node references by name
        ref_ids = []
        for ref_name in claim_def.get("node_references", []):
            if ref_name in node_name_to_id:
                ref_ids.append(node_name_to_id[ref_name])
            else:
                for lookup_conn in all_conns:
                    ref_match = match_node(lookup_conn, ref_name, record_id=record.id)
                    if ref_match:
                        local_id = _materialise_locally(conn, lookup_conn, ref_match[0])
                        ref_ids.append(local_id)
                        node_name_to_id[ref_name] = local_id
                        break

        # Resolve speaker
        speaker_id = None
        speaker_name = claim_def.get("speaker")
        if speaker_name:
            if speaker_name in node_name_to_id:
                speaker_id = node_name_to_id[speaker_name]
            else:
                for lookup_conn in all_conns:
                    speaker_match = match_node(
                        lookup_conn, speaker_name, "person", record_id=record.id
                    )
                    if speaker_match:
                        speaker_id = _materialise_locally(
                            conn, lookup_conn, speaker_match[0]
                        )
                        node_name_to_id[speaker_name] = speaker_id
                        break

        # ADR 0044: carry the chain onto the row. The columns and the model field
        # have existed all along; nothing built the object, so every claim in the
        # graph read origin_kind NULL while the digests carried a chain on 97% of
        # them. The absence was then misread as "the corpus predates 0044" and a
        # full re-digest was expected to fix it, which it never could.
        chain_def = claim_def.get("provenance_chain")
        chain = None
        if isinstance(chain_def, dict) and chain_def.get("origin_kind"):
            chain = ProvenanceChain(
                origin_kind=chain_def["origin_kind"],
                origin=chain_def.get("origin") or "",
                origin_ref=chain_def.get("origin_ref") or "",
                relay=list(chain_def.get("relay") or []),
            )
        # A described speaker has no node to point at, so the attribution would
        # be lost entirely - "who said it" is not recoverable from the claim text.
        # It goes to the chain as an anonymous origin, which is the shape the
        # corpus already uses for an unnamed source and which independence
        # already collapses to a single root (ADR 0039): one anonymous officer is
        # one source, not one per claim.
        if speaker_name and is_a_description(speaker_name):
            chain = ProvenanceChain(
                origin_kind="anonymous",
                origin=speaker_name,
                origin_ref=chain.origin_ref if chain else "",
                relay=list(chain.relay) if chain else [],
            )
        # A DESCRIBED ORIGIN IS ANONYMOUS WHATEVER THE DIGEST CALLED IT. The
        # corpus holds ~20 claims written `origin_kind: named` with an origin of
        # "unnamed APEG biochemist" - a contradiction the extraction model does
        # not notice. "named" makes independence resolve the origin to a node and
        # treat it as a distinct identifiable source; a description has no node,
        # so it would silently fall back to counting each claim as its own root.
        elif chain and is_a_description(chain.origin):
            chain = ProvenanceChain(
                origin_kind="anonymous",
                origin=chain.origin,
                origin_ref=chain.origin_ref,
                relay=list(chain.relay),
            )
        claim = Claim(
            id=claim_def["id"],
            content=claim_def["content"],
            provenance_chain=chain,
            original_excerpt=claim_def.get("original_excerpt"),
            claim_type=claim_def["claim_type"],
            attestation=claim_def.get("attestation"),
            record_id=record.id,
            speaker_id=speaker_id,
            location_in_record=claim_def.get("location_in_record"),
            date=claim_def.get("date"),
            date_end=claim_def.get("date_end"),
            node_references=ref_ids,
        )
        chash = claim_hash(
            content=claim.content,
            claim_type=claim.claim_type.value,
            attestation=claim.attestation.value if claim.attestation else None,
            speaker_id=speaker_id,
            record_id=record.id,
            location_in_record=claim.location_in_record,
            date=claim.date,
            date_end=claim.date_end,
            node_ids=ref_ids,
            original_excerpt=claim.original_excerpt,
        )
        entailment = _entailment_block(claim_def.get("entailment"))
        if claim_def.get("entailment") is not None and entailment is None:
            counts["entailment_malformed"] += 1
        if entailment:
            counts["claims_assessed"] += 1
        resolved_claims.append((claim, chash, entailment))

    # Reconcile against the record's prior claims (empty for a first import).
    prior = get_record_claim_hashes(conn, record.id) if existing_record else {}
    carried: list[tuple] = []
    to_insert: list[tuple] = []
    for claim, chash, entailment in resolved_claims:
        pool = prior.get(chash)
        if pool:
            # Carry forward: an identical claim already exists for this record;
            # leave its IDENTITY (uuid + created_at) untouched. The provenance
            # chain is REFRESHED, not preserved: claim_hash fingerprints meaning
            # and does not cover the chain, so a claim whose chain was absent - or
            # has since changed - matches by hash and would keep the stale value
            # forever. That is how 19,006 claims sat at origin_kind NULL while
            # their digests carried a chain on 87% of them.
            claim_id, _created = pool.pop()
            update_claim_chain(conn, claim_id, claim.provenance_chain)
            update_claim_entailment(conn, claim_id, entailment)
            carried.append((claim, chash))
        else:
            to_insert.append((claim, chash, entailment))

    # DELETE BEFORE INSERT. A claim keeps its uuid across a re-emission while its
    # HASH moves whenever node resolution changes underneath it - a rename, a
    # merge, anything that repoints a ref. Such a claim matches no prior hash, so
    # it is treated as new, and inserting it before its stale row is removed
    # collides on the primary key: "UNIQUE constraint failed: claims.id". The old
    # row is in `prior`'s leftovers by construction, so clearing those first makes
    # the insert well-defined - and the net effect is the same reconciliation.
    for leftover in prior.values():
        for claim_id, _created in leftover:
            delete_claim(conn, claim_id)
            counts["claims_deleted"] += 1
    for claim, chash, entailment in to_insert:
        insert_claim(conn, claim, claim_hash=chash, entailment=entailment)
        counts["claims_created"] += 1
    counts["claims_carried"] += len(carried)

    conn.commit()
    return counts


def backfill_claim_hashes(
    conn: sqlite3.Connection, on_progress: callable = None
) -> dict:
    """Compute and store claim_hash for every claim in the database.

    Pure compute, no AI: the meaning-bearing fields and the resolved graph ids
    (speaker_id, claim_node_refs) are already in the database, so the same
    canonical hash the importer computes can be reproduced for the existing rows.
    Idempotent - safe to re-run; a row gets re-stamped with whatever the current
    hash definition yields.
    """
    log = on_progress or (lambda _: None)
    rows = conn.execute(
        "SELECT id, content, claim_type, attestation, record_id, speaker_id, "
        "location_in_record, date, date_end, original_excerpt FROM claims"
    ).fetchall()
    updated = 0
    for (
        claim_id,
        content,
        claim_type,
        attestation,
        record_id,
        speaker_id,
        location,
        date,
        date_end,
        original_excerpt,
    ) in rows:
        node_ids = [
            r[0]
            for r in conn.execute(
                "SELECT node_id FROM claim_node_refs WHERE claim_id = ?", (claim_id,)
            ).fetchall()
        ]
        update_claim_hash(
            conn,
            claim_id,
            claim_hash(
                content=content,
                claim_type=claim_type,
                attestation=attestation,
                speaker_id=speaker_id,
                record_id=record_id,
                location_in_record=location,
                date=date,
                date_end=date_end,
                node_ids=node_ids,
                original_excerpt=original_excerpt,
            ),
        )
        updated += 1
    conn.commit()
    log(f"Backfilled claim_hash for {updated} claims")
    return {"updated": updated}


def main(argv: list[str] | None = None) -> int:
    """Host-runnable entry point: `python -m assimilator.import_markdown <digest>`.

    The deterministic fold of one reviewed digest YAML into the graph - no Claude,
    no money. Deliberately avoids the embeddings/CLI imports (no fastembed), so the
    runner's eager worker can flow a freshly produced digest into the graph
    host-side. Needs anomalica_common + python-Levenshtein on the path; not
    fastembed. Set ANOMALICA_INGESTS_DIR so content_hash resolves for older
    digests that do not carry it.
    """
    import argparse
    from pathlib import Path

    from anomalica_common.digest.yaml_format import parse_digest_yaml
    from assimilator.database import init_db

    default_db = os.environ.get(
        "ASSIMILATOR_DB",
        str(Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"),
    )
    parser = argparse.ArgumentParser(
        prog="assimilator.import_markdown",
        description="Import one reviewed digest YAML into the graph (no AI).",
    )
    parser.add_argument("digest", help="path to the digest YAML")
    parser.add_argument("--db", default=default_db, help="domain graph DB")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    infra_path = db_path.parent / "infrastructure.db"
    parsed = parse_digest_yaml(Path(args.digest).read_text())

    def _open(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        init_db(conn)
        return conn

    domain_conn, infra_conn = _open(db_path), _open(infra_path)
    try:
        if parsed["domain_claims"]:
            import_extraction(
                domain_conn,
                parsed,
                section="domain",
                lookup_conns=[infra_conn],
                source_path=args.digest,
                on_progress=print,
            )
        if parsed["infrastructure_claims"]:
            import_extraction(
                infra_conn,
                parsed,
                section="infrastructure",
                lookup_conns=[domain_conn],
                source_path=args.digest,
                on_progress=print,
            )
    finally:
        domain_conn.close()
        infra_conn.close()
    print(f"Imported {Path(args.digest).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Copyright statuses that permit republishing the record's verbatim text.
# Everything else - including ABSENT - is not distributable. Absent is the
# important case: it is the entire corpus digested before the field existed, and
# a fail-open reading would mark all of it publishable in one pass.
_DISTRIBUTABLE_STATUSES = frozenset({"public_domain", "open_licence"})


def record_is_distributable(metadata: dict | None) -> bool:
    """Whether this record's VERBATIM text may be republished. Fails closed.

    Three states, not two: distributable, not distributable, and unknown - and
    unknown must read as not distributable. A digest produced before
    copyright_status existed carries no opinion, and treating no opinion as
    permission is how thousands of verbatim passages from copyrighted books reach
    a CDN in one deploy.

    NOT the authority, deliberately. Copyright lives in the record's frontmatter
    in the ingests store; this is a snapshot taken at digestion, and a licence
    that changes afterwards leaves the snapshot asserting the old status with no
    staleness check able to notice (pre_digest.sha256 covers the BODY, which did
    not change). Use this to FILTER and PROPOSE, where being stale costs a
    re-proposal. For an actual publish decision, read the store by content_hash -
    a wrong answer there is irreversible.

    >>> record_is_distributable({"copyright_status": "public_domain"})
    True
    >>> record_is_distributable({"copyright_status": "restricted"})
    False
    >>> record_is_distributable({})
    False
    >>> record_is_distributable(None)
    False
    """
    status = (metadata or {}).get("copyright_status")
    return status in _DISTRIBUTABLE_STATUSES
