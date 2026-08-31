"""Publishing briefs, with copyright redaction applied at the last step.

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

# The C loader where libyaml is available, the Python one otherwise. Measured on
# this corpus: 0.26s per brief in pure Python against 0.025s with libyaml, which
# over 752 briefs is 3.2 minutes against 19 seconds - the difference between a
# command that times out and one that runs. Falls back rather than requiring it,
# because a publish path that cannot run without an optional C extension is worse
# than a slow one.
try:  # pragma: no cover - depends on the libyaml build
    from yaml import CSafeDumper as _Dumper
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeDumper as _Dumper
    from yaml import SafeLoader as _Loader


def _load(text: str):
    return yaml.load(text, Loader=_Loader)


def _dump(obj) -> str:
    return yaml.dump(obj, Dumper=_Dumper, sort_keys=False, allow_unicode=True)


# Statuses whose verbatim text may be republished, per Mark's 2026-08-28 ruling.
# Everything else - restricted, unrecognised, and ABSENT - is withheld. Absent is
# the case that matters: it is every record the store does not resolve, and
# reading no answer as permission is the failure this whole module exists to
# prevent.
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
    counts: dict[str, int] = {}
    for claim in out.get("claims") or []:
        provenance = claim.get("provenance") or {}
        status = copyright_status(store_dir, provenance.get("content_hash") or "")
        counts[status] = counts.get(status, 0) + 1
        if status in PUBLISHABLE_EXCERPT_STATUSES:
            continue
        if claim.get("original_excerpt") is not None:
            claim["original_excerpt"] = None
            claim["excerpt_withheld"] = status
    return out, counts


def publish_briefs(
    briefs_dir: Path, out_dir: Path, store_dir: Path
) -> dict[str, object]:
    """Write every brief to out_dir with excerpts redacted where required."""
    out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    written = 0
    withheld_claims = 0
    # A brief this path cannot read is REPORTED, never passed over. Skipping it
    # silently drops an entity from the published record with nothing to show
    # why - the assembler hit exactly this shape in its own link index, where
    # two entities carrying 200 and 183 claims became unlinkable corpus-wide and
    # no output said so. Continue past the bad file so one brief cannot stop a
    # publish, but return the failures so the caller can fail loudly.
    unreadable: list[str] = []
    for path in sorted(briefs_dir.glob("*.yaml")):
        try:
            brief = _load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            unreadable.append(f"{path.name}: {type(exc).__name__}")
            continue
        if not isinstance(brief, dict):
            unreadable.append(f"{path.name}: not a mapping")
            continue
        published, counts = redact_brief(brief, store_dir)
        for status, n in counts.items():
            totals[status] = totals.get(status, 0) + n
            if status not in PUBLISHABLE_EXCERPT_STATUSES:
                withheld_claims += n
        (out_dir / path.name).write_text(_dump(published))
        written += 1
    return {
        "briefs": written,
        "withheld_claims": withheld_claims,
        "by_status": totals,
        "unreadable": unreadable,
    }
