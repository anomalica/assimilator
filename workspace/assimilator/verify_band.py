"""A prepared, human-cleared verify pass over the shortlist's combined band.

The shortlist (shortlist.py) reaches 75 of the 90 pairs reviewers merged where
the rules reach 19, but the band where both reranker scores are at least 0.9 -
3,680 pairs beyond the rules - is about one genuine duplicate in three, and a
reviewer cannot tell which from the scores (reports/reranker-eval-2026-09-02.md).
So a model reads each pair and answers same-or-different with one reason, and
the verdicts, not the raw band, are what a reviewer sees. Nothing here merges.

TWO GATES. The model is resolved through model-policy.yaml (stage `consolidate`,
the assimilator's stage for merge decisions; the `verify` stage is the
scheduler's and does not list Haiku) and refused if not permitted. And the run
spends the subscription, which is Mark's to clear: the default is a dry run that
prints the token estimate, and a live run needs --run and --confirm together.
Verdicts are appended per pair to a JSONL file and a re-run skips pairs already
decided, so an interrupted pass costs nothing twice.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from assimilator.entity_reranker import entity_from_graph

STAGE = "consolidate"
DEFAULT_MODEL = "claude-haiku-4-5"
BAND_NAMES_MIN = 0.9
BAND_CLAIMS_MIN = 0.9
PAIRS_PER_CALL = 20
CHARS_PER_TOKEN = (
    2.7  # brief_size's ratio; a JSON-and-prose prompt tokenises about as densely
)
OUTPUT_TOKENS_PER_PAIR = 200  # measured 2026-09-03: 101,891 output tokens for 500 pairs
# Each subscription CLI call carries the CLI's own context on top of the prompt:
# measured on the 25 calls of the first 500 pairs, 877k cache-creation and
# 1.83M cache-read tokens = about 108,000 a call, and a notional $0.10 a call
# at the transport's own accounting. My first estimate counted the prompt
# only and was an order of magnitude low; the per-call term is the cost.
CONTEXT_TOKENS_PER_CALL = 108_000
NOTIONAL_USD_PER_CALL = 0.10

PROMPT = """You are checking a knowledge graph about anomalous phenomena for duplicate nodes. For each numbered pair below, decide whether the two entries refer to the SAME real-world entity - the same person, organisation, project, event, place, object, document or topic, possibly under different names, spellings, acronyms or nicknames - or to DIFFERENT entities.

Related but distinct things are DIFFERENT: a person and their organisation, a parent and a child, a ship and an incident aboard it, a mission and one event during it, a report and the investigation that produced it, a paper and its author, two numbered missions in one programme, a place and a thing at it, two members of one family. When the entries do not establish that they are one thing, answer different.

The numbered pairs are in the document. Return JSON only: {"decisions": [{"pair_id": 1, "same": true, "reason": "one sentence"}, ...]} with one decision per pair, in order."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "integer"},
                    "same": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["pair_id", "same", "reason"],
            },
        }
    },
    "required": ["decisions"],
}


def band(
    scored: list[dict],
    names_min: float = BAND_NAMES_MIN,
    claims_min: float = BAND_CLAIMS_MIN,
) -> list[dict]:
    """Pairs in the combined band, strongest first: ordered by the LOWER of the
    two scores, so a pair both scorers are sure of outranks one only one is."""
    keep = [
        f
        for f in scored
        if f["names_only"] >= names_min and f["with_claims"] >= claims_min
    ]
    keep.sort(
        key=lambda f: (
            -min(f["names_only"], f["with_claims"]),
            -max(f["names_only"], f["with_claims"]),
            f["pair"],
        )
    )
    return keep


def batches(items: list, size: int = PAIRS_PER_CALL) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def render(conn: sqlite3.Connection, batch: list[dict]) -> str:
    lines = []
    for i, f in enumerate(batch, 1):
        a = entity_from_graph(conn, f["pair"][0])
        b = entity_from_graph(conn, f["pair"][1])
        if a is None or b is None:
            continue
        lines.append(f"PAIR {i}\nA:\n{a.text()}\nB:\n{b.text()}\n")
    return "\n".join(lines)


def estimate(prompts: list[str], pairs: int) -> dict:
    chars = sum(len(p) for p in prompts)
    calls = len(prompts)
    return {
        "calls": calls,
        "pairs": pairs,
        "input_tokens": int(chars / CHARS_PER_TOKEN),
        "output_tokens": pairs * OUTPUT_TOKENS_PER_PAIR,
        "cached_context_tokens": calls * CONTEXT_TOKENS_PER_CALL,
        "notional_usd": round(calls * NOTIONAL_USD_PER_CALL, 2),
        "chars": chars,
    }


def decided(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            out.add(tuple(d["pair"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def run_batches(
    conn, to_do: list[dict], model: str, use_api, out_path: Path, call, parse, log
) -> dict:
    """Live calls; verdicts appended per pair as they land. Never merges."""
    counts = {"same": 0, "different": 0, "unanswered": 0, "calls": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for batch in batches(to_do):
            document = render(conn, batch)
            raw = call(PROMPT, document, model, schema=SCHEMA, use_api=use_api)
            counts["calls"] += 1
            try:
                data = parse(raw)
            except Exception as exc:  # noqa: BLE001 - one bad reply must not stop the pass
                log(f"  unparseable reply, {len(batch)} pairs left undecided: {exc}")
                counts["unanswered"] += len(batch)
                continue
            by_id = {
                d.get("pair_id"): d
                for d in data.get("decisions", [])
                if isinstance(d, dict)
            }
            for i, item in enumerate(batch, 1):
                d = by_id.get(i)
                if d is None or not isinstance(d.get("same"), bool):
                    counts["unanswered"] += 1
                    continue
                verdict = {
                    "pair": item["pair"],
                    "names_only": item["names_only"],
                    "with_claims": item["with_claims"],
                    "same": d["same"],
                    "reason": str(d.get("reason", ""))[:400],
                    "model": model,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                f.flush()
                counts["same" if d["same"] else "different"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="assimilator.verify_band",
        description="Verify the shortlist's combined band with a model; dry run by default.",
    )
    p.add_argument(
        "--scored", required=True, help="shortlist-eval JSON (every scored pair)"
    )
    p.add_argument(
        "--db",
        default=os.environ.get(
            "ASSIMILATOR_DB", str(Path.home() / ".local/share/assimilator/knowledge.db")
        ),
    )
    p.add_argument(
        "--out",
        default=str(
            Path.home() / ".local/share/assimilator/verify-band-verdicts.jsonl"
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=500,
        help="how many of the band, strongest first (0 = all)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--run", action="store_true", help="make the calls (default: estimate only)"
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Mark has cleared the subscription spend for this run",
    )
    args = p.parse_args(argv)

    scored = json.load(open(args.scored))["final"]
    pairs = [f for f in band(scored) if not f.get("from_rules")]
    if args.top:
        pairs = pairs[: args.top]
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    done = decided(Path(args.out))
    to_do = [f for f in pairs if tuple(f["pair"]) not in done]
    prompts = [PROMPT + render(conn, b) for b in batches(to_do)]
    est = estimate(prompts, len(to_do))
    print(
        f"band: {len(pairs)} pairs (top {args.top or 'all'} of the combined band beyond the rules); already decided {len(done)}; to do {len(to_do)}"
    )
    print(
        f"estimate: {est['calls']} calls, ~{est['input_tokens']:,} prompt tokens, "
        f"~{est['cached_context_tokens']:,} cached CLI context tokens, "
        f"~{est['output_tokens']:,} output tokens, notional ${est['notional_usd']:.2f}, "
        f"model {args.model} on the subscription"
    )
    if not args.run:
        print("dry run: nothing called. --run --confirm to spend (Mark's clearance).")
        return 0
    if not args.confirm:
        print(
            "refusing: --run without --confirm. The subscription is Mark's to clear.",
            file=sys.stderr,
        )
        return 2
    from anomalica_common import model_policy as mp
    from anomalica_common.llm import (
        _call,
        _parse_json,
        get_usage,
        reset_usage,
        resolve_use_api,
    )

    try:
        model = mp.load().check(STAGE, args.model)
    except Exception as exc:  # noqa: BLE001 - every failure is a refusal
        print(f"refused by policy: {exc}", file=sys.stderr)
        return 2
    reset_usage()
    counts = run_batches(
        conn,
        to_do,
        model,
        resolve_use_api("ASSIMILATOR_USE_API"),
        Path(args.out),
        _call,
        _parse_json,
        print,
    )
    print(
        f"verdicts: same {counts['same']}, different {counts['different']}, unanswered {counts['unanswered']} in {counts['calls']} calls -> {args.out}"
    )
    print("usage:", json.dumps(get_usage()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
