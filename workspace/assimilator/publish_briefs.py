"""Preparing briefs for publication.

CLAIM EXCERPTS ARE NOT WITHHELD BY COPYRIGHT STATUS, and this file used to do
exactly that. Recording why, because the mistake was reasonable-looking and was
made twice in one day from opposite directions.

The policy is explicit and predates all of it:

    source-types-and-copyright.md  "Short attributed quotations are published in
                                    full - as long as they need to be to convey
                                    their point - and are NOT capped, truncated
                                    to a length limit, or gated."
                                   "The line is SUBSTANTIALITY, not length."
                                   "Quote is not body - this policy does not
                                    un-gate full bodies or transcripts."
    editorial-style.md             "The platform does not artificially truncate
                                    quotations." Japan's Copyright Act Article 32
                                    with attribution under Article 48.

So what warrants withholding is a full BODY or transcript, which lives behind the
proof-of-possession gate and which this step never touched. A claim excerpt is a
one- or two-sentence attributed quotation that substantiates the claim beside it.

HOW THE REDACTION GOT BUILT: Mark was asked whether excerpts from the
publicly_accessible records go public and answered "yes, always". That answer was
about publicly_accessible ONLY. "So restricted stays redacted" was a component's
inference from it, relayed as though it were part of the ruling, and implemented
here. Mark never ruled that restricted excerpts are withheld - he had in fact
reversed a corpus-wide strip of exactly this material.

WHAT IT COST: the first page built through this step, Socorro, lost 29 of its 39
excerpts - median length 105 characters, minimum 12. Measured across the 467
published pages, 5,414 of 5,438 restricted-source references carry their quote;
the 24 that do not are Socorro's. A claim with no quote beside it reads as
unevidenced, so over-gating is not the free direction.

This step remains the only path out of the internal brief directory, and it
remains a control - for bodies, and for keeping the published set consistent.

THIS IS A COPYRIGHT CONTROL, NOT A CACHE. The two brief directories are not two
copies of the same data:

    ~/.local/share/assimilator/briefs   INTERNAL. Every excerpt verbatim.
    <content>/briefs                    PUBLISHED. Excerpts redacted by status.

The source side is the audit record and is systematically NEWER, because
synthesise writes it and publishing is a separate step. That combination is a
trap: a consumer that wants fresher briefs, or a field only the new ones carry,
will reach for the source directory and by doing so put restricted source text
on public pages - the assembler renders `original_excerpt` verbatim as a page
quote. Widening the access model is Mark's explicit sign-off and is not
reversible once it is on the CDN.

The assembler met that fork on 2026-09-01, checked what publishing does before
taking it, and stopped. Every brief now declares which side it is from
(`publication.status`: unredacted | redacted) so the next consumer can refuse it
without having to know the directory layout. If briefs here look stale, the fix
is to run this command, never to read the other directory.

A brief is the ADR 0010 audit record: the exact material the writer was given.
Publishing it beside the page turns "every assertion traces to a source" from an
assertion into something a reader can check. That is the project's whole premise
and it is currently unfalsifiable from outside.

WHAT MAKES THIS DELICATE is `original_excerpt` - the verbatim source sentence on
every claim. Measured 2026-08-28: 13 of 100 records in the graph are copyright
`restricted`, they carry 18,985 claims (61% of the corpus), and they touch 586 of
691 proposed pages. Publishing briefs unredacted would put thousands of verbatim
passages from copyrighted books onto the CDN across 85% of the site, in one
deploy, irreversibly.

Mark's ruling (2026-08-28), verbatim on whether publicly_accessible excerpts go
public: "yes, always". So public_domain and publicly_accessible publish their
excerpts; restricted has them withheld.

THE STORE IS THE AUTHORITY, NOT THE DIGEST. Copyright lives in the ingests
record's frontmatter, which is the field the access gate itself runs on. The
digester now also carries a `copyright_status` snapshot into the digest, and that
snapshot is right for filtering and proposing - but it is taken at digestion and
`pre_digest.sha256` covers the BODY, so a licence that changes afterwards leaves
the snapshot asserting the old status with NO staleness check able to notice. For
a publish decision, which is irreversible, read the live value.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from assimilator.brief_yaml import PUBLISHED
from assimilator.brief_yaml import dump as dump_brief_yaml

# The C loader where libyaml is available, the Python one otherwise. Measured on
# this corpus: 0.26s per brief in pure Python against 0.025s with libyaml, which
# over 752 briefs is 3.2 minutes against 19 seconds - the difference between a
# command that times out and one that runs. Falls back rather than requiring it,
# because a publish path that cannot run without an optional C extension is worse
# than a slow one. The DUMPER lives in brief_yaml, which picks the same C
# implementation and adds the hook-safe representer.
try:  # pragma: no cover - depends on the libyaml build
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _Loader


def _load(text: str):
    return yaml.load(text, Loader=_Loader)


def _dump(obj) -> str:
    return dump_brief_yaml(obj)


# NO LONGER GATES ANYTHING. Kept because the workbench and the reporting path
# still name these statuses, and because deleting the constant would erase the
# record of what it used to mean: it was the set whose excerpts could be
# published, and every other status had its excerpt withheld. That was wrong -
# short attributed quotations publish whatever the source's status, and the
# gate belongs on full bodies. See the module docstring.
PUBLISHABLE_EXCERPT_STATUSES = frozenset(
    {"public_domain", "publicly_accessible", "open_licence"}
)

_FRONTMATTER_END = re.compile(r"^---\s*$", re.M)


@lru_cache(maxsize=4096)
def _store_frontmatter(store_dir: Path, content_hash: str) -> dict:
    """The ingests record's frontmatter, or {} when it does not resolve.

    Cached on (store, hash): a brief holds hundreds of claims drawn from a handful
    of records, so the uncached version re-read and re-parsed the same file once
    per claim - roughly 150,000 reads across the corpus, which took the publish
    command past a two-minute timeout doing the same work over and over.
    """
    h = (content_hash or "").replace("sha256:", "")
    if not h:
        return {}
    # THREE LOCATIONS, and missing the third was withholding 1,311 excerpts for
    # the wrong reason. store/v1/ holds the record/1-schema files, and seven
    # records live only there - Surviving Death, In Plain Sight, Imminent among
    # them. A lookup that misses them returns "unknown", which fails closed and
    # therefore looks exactly like correct caution rather than an incomplete
    # search. That is the worst shape a bug can have in a gate: it is invisible
    # because its failure mode is the safe one.
    for name in (f"{h}.md", f"{h}.v2.md", f"v1/{h}.md"):
        path = store_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return {}
        if not text.startswith("---"):
            return {}
        match = _FRONTMATTER_END.search(text, 3)
        if not match:
            return {}
        try:
            return _load(text[3 : match.start()]) or {}
        except yaml.YAMLError:
            return {}
    return {}


def copyright_status(store_dir: Path, content_hash: str) -> str:
    """Live copyright status from the ingests store. "unknown" when unresolved.

    Nested under `copyright.status`, NOT flat. The flat name was reported once
    and filtering on it returns zero, which reads as "no copyright data" and is
    how a fail-open gate gets built by accident.
    """
    block = _store_frontmatter(store_dir, content_hash).get("copyright")
    if not isinstance(block, dict):
        return "unknown"
    return block.get("status") or "unknown"


def redact_brief(brief: dict, store_dir: Path) -> tuple[dict, dict]:
    """A publishable copy of the brief, plus per-status counts of what it did.

    Withholding is MARKED, never silent: a claim whose excerpt is withheld says
    so and says why. A reader must be able to tell "this source is copyrighted so
    we cannot reproduce the sentence" from "there was no excerpt" - the first is a
    licence boundary, the second would look like a gap in our evidence.
    """
    out = _load(_dump(brief))
    # Marks which directory this copy came from. Not a redaction claim - see the
    # module docstring for why claim excerpts are no longer withheld.
    out["publication"] = {
        "status": PUBLISHED,
        "note": "Prepared for publication. Claim excerpts are short attributed quotations and are carried through; full bodies and transcripts are gated elsewhere.",
    }
    counts: dict[str, int] = {}
    for claim in out.get("claims") or []:
        provenance = claim.get("provenance") or {}
        status = copyright_status(store_dir, provenance.get("content_hash") or "")
        counts[status] = counts.get(status, 0) + 1
    return out, counts


def unbuildable_in(out_dir: Path, conn) -> list[dict]:
    """Published briefs a consumer must not build from, and why.

    The published directory was never pruned. prune_retired_briefs runs during
    synthesis, on the SOURCE directory, so a merge or a rename cleans up there
    and leaves the published copy standing - and the assembler takes a brief by
    slug. Two ways that bites, both seen in the live corpus:

    - the node is RETIRED or gone, so the brief builds a page for something that
      no longer exists (kenneth-arnold-sighting and phoenix-lights, both merged
      away the same morning the published copies stayed);
    - the brief sits at a STALE PATH while the node has another brief at its
      current one, so one entity gets two pages. That is the Elizondo failure
      exactly. The path is <section>/<slug>.yaml (synthesise.brief_relpath), so
      a rename, a retype, or a file left in the pre-section flat layout all
      strand a brief the same way.

    Identified from the graph rather than from a list of filenames, so it stays
    true as the graph moves. `file` is the path relative to out_dir.
    """
    from assimilator.synthesise import (
        brief_files,
        brief_node_id,
        brief_relpath,
        build_slug_map,
        node_slug,
    )

    live = {
        row[0]: (row[1], row[2], row[3])
        for row in conn.execute(
            "SELECT id, node_type, name, metadata FROM nodes WHERE retired_at IS NULL"
        )
    }
    slug_map, _collisions = build_slug_map(conn)
    findings: list[dict] = []
    for path in sorted(out_dir.glob("*.yaml")) + brief_files(out_dir):
        node_id = brief_node_id(path)
        if not node_id:
            continue
        rel = str(path.relative_to(out_dir))
        if node_id not in live:
            findings.append({"file": rel, "why": "node retired or absent"})
            continue
        node_type, name, metadata = live[node_id]
        current = brief_relpath(
            node_type, slug_map.get(node_id) or node_slug(name, metadata)
        )
        if path.relative_to(out_dir) != current:
            findings.append(
                {"file": rel, "why": f"stale path; the node's brief is now {current}"}
            )
    return findings


def publish_briefs(
    briefs_dir: Path, out_dir: Path, store_dir: Path
) -> dict[str, object]:
    """Write every brief to out_dir, at the same <section>/<slug>.yaml path it
    holds in briefs_dir."""
    from assimilator.synthesise import brief_files

    out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    written = 0
    # A brief this path cannot read is REPORTED, never passed over. Skipping it
    # silently drops an entity from the published record with nothing to show
    # why - the assembler hit exactly this shape in its own link index, where
    # two entities carrying 200 and 183 claims became unlinkable corpus-wide and
    # no output said so. Continue past the bad file so one brief cannot stop a
    # publish, but return the failures so the caller can fail loudly.
    unreadable: list[str] = []
    for path in brief_files(briefs_dir):
        rel = path.relative_to(briefs_dir)
        try:
            brief = _load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            unreadable.append(f"{rel}: {type(exc).__name__}")
            continue
        if not isinstance(brief, dict):
            unreadable.append(f"{rel}: not a mapping")
            continue
        published, counts = redact_brief(brief, store_dir)
        for status, n in counts.items():
            totals[status] = totals.get(status, 0) + n
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_dump(published))
        written += 1
    return {
        "briefs": written,
        "withheld_claims": 0,  # nothing is withheld; kept so callers do not break
        "by_status": totals,
        "unreadable": unreadable,
    }
