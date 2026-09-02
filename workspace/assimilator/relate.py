"""EXPERIMENTAL: do two records refer to the same specific subject?

Nothing in the pipeline links a record that reports an operation to another
record that reports the same operation under other words. Shared entity nodes
are corpus-wide hubs (the UAP topic, Congress); claim similarity asks "same
fact?", which two accounts of one event rarely are; and neither record cites
the other. On 2026-09-02 the Liberation Times "ODNI UAP luring operation"
article and the Coulthart Skywatcher Q&A sat at rank 8 to 17 of 108 by every
similarity measure, and a model reading both claim lists found them - and two
more records on the same incident that nothing had linked
(reports/cross-source-linking-2026-09-02/README.md).

So: a shortlist of record pairs from claim neighbourhoods, and a STRICT judge
(the wording that gave zero false positives on nine pairs; the looser one gave
four) that names the shared specific subject and the claim pairs carrying it.
Several pairs go in one call - a comparison set steadies the judge's boundary,
and the CLI's own context (about 108,000 cached tokens a call) dwarfs any
prompt - and every positive is re-judged once beside different neighbours and
kept only if reproduced, because with one pair per call the boundary between
unrelated and possibly_related moved between runs on the same records.
Verdicts, including "unrelated" and failed confirmations, go to
record_relations - derived, rebuildable, nothing depends on it. Not wired into
rebuild or the scheduler; a human confirms or rejects in the workbench later.
Cleared by Mark 2026-09-03 as an experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections import defaultdict

STAGE = "relate"
DEFAULT_MODEL = "claude-haiku-4-5"
K_NEIGHBOURS = 20
TOP_RECORDS = 15
MAX_SIDE = (
    80  # a side above this sends only the claims that neighboured the other record
)
PAIRS_PER_CALL = 6  # several pairs side by side give the judge a comparison set

# The strict wording, verbatim from judge3.py, followed by the batching rule.
# Every word of the exclusion list earned its place: without it the judge
# returned "the Trump administration's UAP efforts" for four pairs that share
# nothing but a period.
PROMPT = """You are comparing the extracted claims of two source records from a reference corpus on anomalous phenomena.

Question: do the two records refer to the SAME SPECIFIC THING - one identifiable incident, operation, programme, document or investigation, pinned by a place, a date, an official identity, or distinctive described particulars that BOTH records carry?

Not a shared subject: the same speaker or outlet, the same period or administration, the same agency, the same general topic, or "the government's UAP efforts" in general. Two records by the same journalist are not about the same thing because he made both. Default to "unrelated" unless you can point to a claim on each side that pins the same specific thing.

- "same_subject": both records clearly refer to the same specific thing (possibly under different descriptions).
- "possibly_related": the particulars on each side (place, date, actors, described events) could be the same specific thing but neither record establishes it.
- "unrelated": everything else.

Give the shared specific thing as one short noun phrase with its date or place (empty if unrelated), a two-sentence reason that quotes the pinning particular from each side, and up to 6 claim-id pairs that carry the connection.

The numbered pairs are in the document. Judge each on its own; the pairs have nothing to do with one another. Return JSON only: {"decisions": [{"pair_id": 1, "verdict": ..., "shared_subject": ..., "reason": ..., "links": [...]}, ...]} with one decision per pair, in order.
"""

PAIR_BLOCK = """PAIR {n}
RECORD A:
{A}

RECORD B:
{B}
"""

_DECISION = {
    "type": "object",
    "properties": {
        "pair_id": {"type": "integer"},
        "verdict": {
            "type": "string",
            "enum": ["same_subject", "possibly_related", "unrelated"],
        },
        "shared_subject": {"type": "string"},
        "reason": {"type": "string"},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["same_fact", "same_subject", "contradicts"],
                    },
                },
                "required": ["a", "b", "relation"],
            },
        },
    },
    "required": ["pair_id", "verdict", "shared_subject", "reason", "links"],
}
SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": _DECISION}},
    "required": ["decisions"],
}

VERDICTS = ("same_subject", "possibly_related", "unrelated")


def load_claim_vectors(conn: sqlite3.Connection):
    """(claim_ids, record_ids, unit vectors) for every embedded claim. Needs the
    sqlite-vec extension loaded on the connection (embeddings.init_vec)."""
    import numpy as np

    from assimilator.embeddings import deserialise_f32

    rows = conn.execute(
        "SELECT v.claim_id, c.record_id, v.embedding FROM vec_claims v "
        "JOIN claims c ON c.id = v.claim_id"
    ).fetchall()
    ids = [r[0] for r in rows]
    recs = [r[1] for r in rows]
    m = np.asarray([deserialise_f32(r[2]) for r in rows], dtype=np.float32)
    if m.size:
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return ids, recs, m


def shortlist(
    ids,
    recs,
    vectors,
    k: int = K_NEIGHBOURS,
    top: int = TOP_RECORDS,
    only: set[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """Record pairs worth judging. For each record, each claim's k nearest
    claims in OTHER records, hits counted per record, the top records kept.
    A pair carries, per side, which claims took part - what a big side is cut
    down to at render time. Symmetric: keyed (a, b) with a < b, the hits from
    each direction kept apart."""
    import numpy as np

    by_record: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        by_record[r].append(i)
    rec_arr = np.asarray(recs)
    out: dict[tuple[str, str], dict] = {}
    for r, idx in by_record.items():
        if only is not None and r not in only:
            continue
        hits: dict[str, int] = defaultdict(int)
        mine: dict[str, set] = defaultdict(set)
        theirs: dict[str, set] = defaultdict(set)
        for start in range(0, len(idx), 500):
            chunk = idx[start : start + 500]
            sims = vectors[chunk] @ vectors.T
            sims[:, rec_arr == r] = -1.0  # never the same record
            kk = min(k, sims.shape[1] - 1)
            if kk <= 0:
                continue
            top_idx = np.argpartition(-sims, kk, axis=1)[:, :kk]
            for row_i, (row, ci) in enumerate(zip(top_idx, chunk)):
                for j in row:
                    if sims[row_i, j] <= -1.0:
                        continue
                    other = recs[int(j)]
                    hits[other] += 1
                    mine[other].add(ids[ci])
                    theirs[other].add(ids[int(j)])
        for other, n in sorted(hits.items(), key=lambda kv: -kv[1])[:top]:
            a, b = (r, other) if r < other else (other, r)
            entry = out.setdefault(
                (a, b),
                {"hits_ab": 0, "hits_ba": 0, "claims_a": set(), "claims_b": set()},
            )
            if r == a:
                entry["hits_ab"] += n
                entry["claims_a"] |= mine[other]
                entry["claims_b"] |= theirs[other]
            else:
                entry["hits_ba"] += n
                entry["claims_b"] |= mine[other]
                entry["claims_a"] |= theirs[other]
    return out


def claim_lines(
    conn: sqlite3.Connection, record_id: str, keep: set[str] | None
) -> list[str]:
    rows = conn.execute(
        "SELECT id, claim_type, content FROM claims WHERE record_id = ? "
        "ORDER BY location_in_record, rowid",
        (record_id,),
    ).fetchall()
    if len(rows) > MAX_SIDE and keep:
        rows = [r for r in rows if r[0] in keep][:MAX_SIDE]
    return [f"[{r[0][:8]}] ({r[1]}) {(r[2] or '').strip()}" for r in rows]


def render(
    conn: sqlite3.Connection, batch: list[tuple[tuple[str, str], dict | None]]
) -> str:
    """The document for a batch of pairs, numbered from 1. It travels as the
    transport's `text` - a file the model reads - not on the command line,
    which has an argument limit a six-pair batch exceeds."""
    blocks = []
    for n, ((a, b), entry) in enumerate(batch, 1):
        ka = (entry or {}).get("claims_a")
        kb = (entry or {}).get("claims_b")
        blocks.append(
            PAIR_BLOCK.format(
                n=n,
                A="\n".join(claim_lines(conn, a, ka)),
                B="\n".join(claim_lines(conn, b, kb)),
            )
        )
    return "\n".join(blocks)


def batches(items: list, size: int = PAIRS_PER_CALL) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def judged(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return {
        (r[0], r[1])
        for r in conn.execute("SELECT record_a, record_b FROM record_relations")
    }


def full_claim_ids(conn: sqlite3.Connection, a: str, b: str, links: list) -> list[dict]:
    """The judge names claims by the 8-character prefix the prompt shows; the
    stored links carry the full ids, so a reader of record_relations needs no
    prefix resolution. A prefix that matches no claim of the pair is kept as
    given rather than dropped."""
    by_prefix: dict[str, str] = {}
    for rid in (a, b):
        for (cid,) in conn.execute("SELECT id FROM claims WHERE record_id = ?", (rid,)):
            by_prefix.setdefault(cid[:8], cid)
    out = []
    for link in links or []:
        if not isinstance(link, dict):
            continue
        out.append(
            {
                **link,
                "a": by_prefix.get(str(link.get("a", ""))[:8], link.get("a")),
                "b": by_prefix.get(str(link.get("b", ""))[:8], link.get("b")),
            }
        )
    return out


def store(
    conn: sqlite3.Connection, a: str, b: str, result: dict, model: str, prompt: str
) -> None:
    """A first-pass verdict. Replaces any earlier row for the pair."""
    verdict = result.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not one of {VERDICTS}")
    conn.execute(
        "INSERT OR REPLACE INTO record_relations (record_a, record_b, verdict, shared_subject, "
        "reason, links, model, prompt_sha, judged_at, first_verdict, confirm_verdict, confirmed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
        (
            a,
            b,
            verdict,
            (result.get("shared_subject") or "").strip() or None,
            (result.get("reason") or "").strip() or None,
            json.dumps(
                full_claim_ids(conn, a, b, result.get("links") or []),
                ensure_ascii=False,
            ),
            model,
            hashlib.sha256(prompt.encode()).hexdigest(),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            verdict,
        ),
    )
    conn.commit()


def store_confirmation(conn: sqlite3.Connection, a: str, b: str, result: dict) -> bool:
    """The second verdict on a positive pair. A positive survives only if the
    second pass is positive too (same_subject or possibly_related both count);
    otherwise the pair becomes unrelated, both verdicts kept."""
    verdict = result.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} is not one of {VERDICTS}")
    confirmed = verdict != "unrelated"
    if confirmed:
        conn.execute(
            "UPDATE record_relations SET confirm_verdict = ?, confirmed = 1 "
            "WHERE record_a = ? AND record_b = ?",
            (verdict, a, b),
        )
    else:
        conn.execute(
            "UPDATE record_relations SET confirm_verdict = ?, confirmed = 0, verdict = 'unrelated', "
            "reason = 'failed confirmation: ' || COALESCE(reason, '') "
            "WHERE record_a = ? AND record_b = ?",
            (verdict, a, b),
        )
    conn.commit()
    return confirmed


def _decisions(
    conn, batch, model: str, use_api, call, parse
) -> tuple[str, dict[int, dict]]:
    document = render(conn, batch)
    data = parse(call(PROMPT, document, model, schema=SCHEMA, use_api=use_api))
    return PROMPT + document, {
        d.get("pair_id"): d for d in data.get("decisions", []) if isinstance(d, dict)
    }


def judge_pairs(
    conn: sqlite3.Connection,
    pairs: list[tuple[tuple[str, str], dict]],
    model: str,
    use_api,
    call,
    parse,
    log=print,
    per_call: int = PAIRS_PER_CALL,
) -> dict:
    """First pass, several pairs per call; each verdict stored as it lands, so
    an interrupted run keeps what it judged and a re-run skips it."""
    counts = {v: 0 for v in VERDICTS}
    counts["errors"] = 0
    counts["calls"] = 0
    for batch in batches(pairs, per_call):
        t0 = time.time()
        try:
            prompt, by_id = _decisions(conn, batch, model, use_api, call, parse)
        except Exception as exc:  # noqa: BLE001 - one bad reply must not stop the pass
            counts["errors"] += len(batch)
            log(f"  batch of {len(batch)}: error {exc}")
            continue
        counts["calls"] += 1
        for n, ((a, b), _entry) in enumerate(batch, 1):
            d = by_id.get(n)
            try:
                if d is None:
                    raise ValueError("no decision for this pair")
                store(conn, a, b, d, model, prompt)
            except Exception as exc:  # noqa: BLE001
                counts["errors"] += 1
                log(f"  {a[:8]} ~ {b[:8]}: error {exc}")
                continue
            counts[d["verdict"]] += 1
            log(
                f"  {a[:8]} ~ {b[:8]}: {d['verdict']} {d.get('shared_subject') or ''} "
                f"({time.time() - t0:.0f}s for the batch)"
            )
    return counts


def positives_to_confirm(
    conn: sqlite3.Connection, only: set[str] | None = None
) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT record_a, record_b FROM record_relations "
        "WHERE verdict != 'unrelated' AND confirmed IS NULL"
    ).fetchall()
    return [(a, b) for a, b in rows if only is None or a in only or b in only]


def confirm_pairs(
    conn: sqlite3.Connection,
    positives: list[tuple[str, str]],
    fillers: list[tuple[tuple[str, str], dict]],
    entries: dict,
    model: str,
    use_api,
    call,
    parse,
    log=print,
    per_call: int = PAIRS_PER_CALL,
) -> dict:
    """Second pass on positives, each in a fresh batch: at most two positives
    per call, padded with pairs that were NOT their first-pass neighbours, so
    the comparison set differs. Both verdicts are kept; a positive survives
    only if reproduced."""
    counts = {"confirmed": 0, "failed": 0, "errors": 0, "calls": 0}
    rnd = random.Random(0)
    for i in range(0, len(positives), 2):
        group = positives[i : i + 2]
        pad = [f for f in fillers if f[0] not in group]
        rnd.shuffle(pad)
        batch = [(p, entries.get(p)) for p in group] + pad[
            : max(0, per_call - len(group))
        ]
        rnd.shuffle(batch)
        try:
            _prompt, by_id = _decisions(conn, batch, model, use_api, call, parse)
        except Exception as exc:  # noqa: BLE001
            counts["errors"] += len(group)
            log(f"  confirm batch: error {exc}")
            continue
        counts["calls"] += 1
        for n, (pair, _e) in enumerate(batch, 1):
            if pair not in group:
                continue
            d = by_id.get(n)
            if d is None or d.get("verdict") not in VERDICTS:
                counts["errors"] += 1
                continue
            ok = store_confirmation(conn, pair[0], pair[1], d)
            counts["confirmed" if ok else "failed"] += 1
            log(
                f"  {pair[0][:8]} ~ {pair[1][:8]}: {'confirmed' if ok else 'FAILED confirmation'} "
                f"({d['verdict']} {d.get('shared_subject') or ''})"
            )
    return counts


def related(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT r.record_a, ra.title, r.record_b, rb.title, r.verdict, r.shared_subject, "
        "r.model, r.judged_at, r.confirmed "
        "FROM record_relations r JOIN records ra ON ra.id = r.record_a "
        "JOIN records rb ON rb.id = r.record_b "
        "WHERE r.verdict != 'unrelated' ORDER BY r.verdict, r.shared_subject"
    ).fetchall()


def resolve_record(conn: sqlite3.Connection, ref: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM records WHERE id LIKE ? OR content_hash LIKE ? OR content_hash LIKE ?",
        (ref + "%", ref + "%", "sha256:" + ref + "%"),
    ).fetchone()
    return row[0] if row else None
