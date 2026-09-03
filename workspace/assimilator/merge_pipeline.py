"""Merge candidates the rules cannot see, found and judged in one job.

shortlist (every live node's twenty nearest by profile vector, union the rules'
pairs) -> reranker on pairs not yet scored -> model verify on band pairs not
yet judged -> yes-verdicts into the workbench's merge-candidate queue with the
model's reason. Every stage is memoised on disk BY PAIR, so a run after an
import costs exactly the new pairs, and a run after nothing changed exits
EXIT_NOTHING without a call. Nothing here merges; a reviewer confirms.

Built for the scheduler to drive after every import or rebuild (2026-09-03,
master relaying Mark): --dry-run prints PLAN_JSON and calls nothing; --run
prints USAGE_JSON and RUN_JSON and appends the run line. The verify model is
resolved through the policy's consolidate stage and refused if unlisted.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from assimilator import verify_band
from assimilator.entity_reranker import Entity, entity_from_graph
from assimilator.shortlist import K_NEIGHBOURS, shortlist
from assimilator.data_dir import data_dir

EXIT_OK = 0
EXIT_NOTHING = 10
EXIT_POLICY = 2
EXIT_WEIGHTS = 3
EXIT_ENDPOINT = 4
EXIT_GPU = 5
EXIT_VERIFY = 6

NAMES_FILTER = 0.3  # below this the with-claims score is not computed
NAMES_BATCH = 48
CLAIMS_BATCH = 16


def _data(name: str, env: str) -> Path:
    return Path(os.environ.get(env, str(data_dir() / name)))


def scores_path() -> Path:
    return _data("rerank-scores.jsonl", "ASSIMILATOR_RERANK_SCORES")


def verdicts_path() -> Path:
    return _data("verify-band-verdicts.jsonl", "ASSIMILATOR_VERIFY_VERDICTS")


def runs_path() -> Path:
    return _data("merge-shortlist-runs.jsonl", "ASSIMILATOR_MERGE_SHORTLIST_RUNS")


def candidates_path() -> Path:
    return _data("merge-candidates.json", "ANOMALICA_MERGE_CANDIDATES")


def manual_path() -> Path:
    return _data("merge-candidates-manual.json", "ANOMALICA_MERGE_CANDIDATES_MANUAL")


def _key(pair) -> tuple[str, str]:
    a, b = pair
    return (a, b) if a < b else (b, a)


def load_scores(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            out[_key(d["pair"])] = d
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def load_verdicts(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            out[_key(d["pair"])] = d
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def in_band(s: dict, floor: float | None = None) -> bool:
    """Both scores at or above the band floor. The default floor is the 0.9 the
    band was defined with; a run may raise it (master, 2026-09-03: judge the
    rest of the band in one go at 0.95 once Mark's queue is worked down, rather
    than creep the threshold up)."""
    wc = s.get("with_claims")
    names_min = verify_band.BAND_NAMES_MIN if floor is None else floor
    claims_min = verify_band.BAND_CLAIMS_MIN if floor is None else floor
    return wc is not None and s.get("names_only", 0.0) >= names_min and wc >= claims_min


def make_plan(
    conn: sqlite3.Connection,
    embed,
    scores: dict,
    verdicts: dict,
    k: int = K_NEIGHBOURS,
    band_floor: float | None = None,
    pairs_per_call: int = verify_band.PAIRS_PER_CALL,
    cross_type: bool = False,
) -> dict:
    """What a run would do, without the reranker or the model.

    THE BAND IS SAME-TYPE ONLY unless cross_type is set. Measured on the 500
    pairs Mark cleared (2026-09-03): the judge called 227 of 325 same-type
    pairs the same entity (70%) and 20 of 175 cross-type pairs (11%) - an
    event beside a place or a person with a similar name is the shape of the
    misses. Cross-type twins reach a reviewer through the import-time queue on
    exact name equality, which is the evidence that actually carries them.
    Same-type filtering alone would have cut the 500 to 325 and the queue's
    precision from 49% to 70%.
    """
    sl = shortlist(conn, embed, k=k, rules_path=candidates_path())
    pairs = {_key(p) for p in sl["pairs"]}
    to_score = sorted(p for p in pairs if p not in scores)
    types = {pr.node_id: pr.node_type for pr in sl["profiles"]}
    band = [
        p
        for p in pairs
        if p in scores
        and in_band(scores[p], band_floor)
        and p not in verdicts
        and (cross_type or types.get(p[0]) == types.get(p[1]))
    ]
    band.sort(
        key=lambda p: (-min(scores[p]["names_only"], scores[p]["with_claims"]), p)
    )
    items = [
        {"pair": list(p), **{k_: scores[p][k_] for k_ in ("names_only", "with_claims")}}
        for p in band
    ]
    prompts = [
        verify_band.PROMPT + verify_band.render(conn, b)
        for b in verify_band.batches(items, pairs_per_call)
    ]
    est = verify_band.estimate(prompts, len(items))
    return {
        "shortlist_pairs": len(pairs),
        "rules_dropped": sl["rules_dropped"],
        "new_pairs": len(to_score),
        "to_score": to_score,
        "to_verify": items,
        "verify_calls": est["calls"],
        "input_tokens": est["input_tokens"],
        "output_tokens": est["output_tokens"],
        "cached_tokens_per_call": verify_band.CONTEXT_TOKENS_PER_CALL,
        "cached_context_tokens": est["cached_context_tokens"],
        "notional_usd": est["notional_usd"],
        "embed_texts": len(sl["profiles"]),
    }


def plan_line(plan: dict) -> str:
    keys = (
        "shortlist_pairs",
        "new_pairs",
        "verify_calls",
        "input_tokens",
        "output_tokens",
        "cached_tokens_per_call",
        "cached_context_tokens",
        "notional_usd",
        "embed_texts",
        "rules_dropped",
    )
    body = {k_: plan[k_] for k_ in keys}
    body["to_score"] = len(plan["to_score"])
    body["to_verify"] = len(plan["to_verify"])
    return "PLAN_JSON " + json.dumps(body)


def score_pairs(
    conn: sqlite3.Connection,
    pairs: list[tuple[str, str]],
    reranker,
    path: Path,
    log=print,
) -> dict:
    """Both reranker scores for each pair, appended to the memo as they land.
    with-claims only where names-only passes NAMES_FILTER; below it the pair
    can never reach the band, so the expensive pass is not spent on it."""
    entities = {}
    for a, b in pairs:
        for nid in (a, b):
            if nid not in entities:
                entities[nid] = entity_from_graph(conn, nid)
    pairs = [p for p in pairs if entities.get(p[0]) and entities.get(p[1])]
    bare = lambda e: Entity(e.name, e.node_type, [])  # noqa: E731
    counts = {"scored": 0, "with_claims": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for i in range(0, len(pairs), 4000):
            chunk = pairs[i : i + 4000]
            names = reranker.score(
                [(bare(entities[a]), bare(entities[b])) for a, b in chunk],
                batch_size=NAMES_BATCH,
                symmetric=False,
            )
            keep = [(p, s) for p, s in zip(chunk, names) if s >= NAMES_FILTER]
            claims = (
                reranker.score(
                    [(entities[a], entities[b]) for (a, b), _ in keep],
                    batch_size=CLAIMS_BATCH,
                    symmetric=True,
                )
                if keep
                else []
            )
            wc = dict(zip([p for p, _ in keep], claims))
            for p, s in zip(chunk, names):
                f.write(
                    json.dumps(
                        {
                            "pair": list(p),
                            "names_only": round(s, 4),
                            "with_claims": round(wc[p], 4) if p in wc else None,
                        }
                    )
                    + "\n"
                )
            f.flush()
            counts["scored"] += len(chunk)
            counts["with_claims"] += len(keep)
            log(
                f"  scored {counts['scored']}/{len(pairs)} ({counts['with_claims']} with claims)"
            )
    return counts


def land_yes(verdicts: dict, path: Path, conn: sqlite3.Connection) -> int:
    """Every same=true verdict becomes one reviewer-queue entry with the model's
    reason; idempotent on the node set; the survivor suggested is the node with
    more claims. Nothing merges."""
    try:
        existing = json.loads(path.read_text()) if path.exists() else []
    except (OSError, json.JSONDecodeError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    have = {frozenset(c.get("node_ids") or []) for c in existing if isinstance(c, dict)}
    added = 0
    for pair, v in verdicts.items():
        if not v.get("same") or frozenset(pair) in have:
            continue
        rows = {
            nid: conn.execute(
                "SELECT name, node_type, (SELECT COUNT(*) FROM claim_node_refs r WHERE r.node_id = nodes.id) FROM nodes WHERE id = ? AND retired_at IS NULL",
                (nid,),
            ).fetchone()
            for nid in pair
        }
        if any(r is None for r in rows.values()):
            continue  # a node merged or retired since the verdict
        survivor = max(pair, key=lambda nid: rows[nid][2])
        existing.append(
            {
                "node_ids": sorted(pair),
                # A NAME, never an id: the canonical name becomes the survivor's
                # node name, hence the page title and its address. 253 queued
                # proposals carried the survivor's uuid here on 2026-09-03 and
                # the workbench offered it as the name to keep.
                "suggested_canonical": rows[survivor][0],
                "suggested_survivor": survivor,
                "score": round(
                    min(v.get("names_only", 0.9), v.get("with_claims", 0.9)), 3
                ),
                "node_type": rows[survivor][1],
                "reason": f"verify: {v.get('reason', '').strip()} ({v.get('model')}; "
                + " ~ ".join(
                    f"{rows[n][0]!r} [{rows[n][1]}, {rows[n][2]} claims]"
                    for n in sorted(pair)
                )
                + ")",
            }
        )
        have.add(frozenset(pair))
        added += 1
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return added


def run_pipeline(
    conn,
    embed,
    reranker_factory,
    call,
    parse,
    model: str,
    use_api,
    dry_run: bool,
    log=print,
    k: int = K_NEIGHBOURS,
    score_only: bool = False,
    band_floor: float | None = None,
    pairs_per_call: int = verify_band.PAIRS_PER_CALL,
    cross_type: bool = False,
) -> int:
    """The whole chain with its dependencies injected; returns the exit code.

    score_only runs the shortlist and the reranker (GPU, no model, no
    allowance) and holds the verify stage, so new pairs carry scores and are
    ready the day judging is cleared; the plan line still reports the band.
    """
    started = time.time()
    stage: dict[str, float] = {}
    t = time.time()
    try:
        scores = load_scores(scores_path())
        verdicts = load_verdicts(verdicts_path())
        plan = make_plan(
            conn, embed, scores, verdicts, k, band_floor, pairs_per_call, cross_type
        )
    except Exception as exc:  # noqa: BLE001 - the endpoint is the only network here
        log(f"embedding endpoint or shortlist failed: {exc}")
        return EXIT_ENDPOINT
    stage["embed"] = round(time.time() - t, 1)
    log(plan_line(plan))
    if score_only:
        log("score-only: the verify stage is held; the band above is not judged")
    nothing = not plan["to_score"] and not (plan["to_verify"] and not score_only)
    if dry_run:
        return EXIT_NOTHING if nothing else EXIT_OK
    if nothing:
        return EXIT_NOTHING
    counts: dict = {
        "shortlist": plan["shortlist_pairs"],
        "scored": 0,
        "band": 0,
        "verified": 0,
        "same": 0,
        "different": 0,
        "unanswered": 0,
    }
    rerank_model = None
    if plan["to_score"]:
        t = time.time()
        try:
            reranker = reranker_factory()
        except FileNotFoundError as exc:
            log(f"reranker weights missing: {exc}")
            return EXIT_WEIGHTS
        try:
            rerank_model = getattr(reranker, "model_id", None)
            counts["scored"] = score_pairs(
                conn, plan["to_score"], reranker, scores_path(), log
            )["scored"]
        except Exception as exc:  # noqa: BLE001 - out of memory, Triton, anything on the card
            log(f"GPU stage failed: {exc}")
            return EXIT_GPU
        stage["rerank"] = round(time.time() - t, 1)
        # Re-plan the band now that new pairs carry scores.
        scores = load_scores(scores_path())
        plan = make_plan(
            conn, embed, scores, verdicts, k, band_floor, pairs_per_call, cross_type
        )
    counts["band"] = len(plan["to_verify"])
    outcome = "ok"
    t = time.time()
    if plan["to_verify"] and not score_only:
        try:
            vc = verify_band.run_batches(
                conn,
                plan["to_verify"],
                model,
                use_api,
                verdicts_path(),
                call,
                parse,
                log,
                size=pairs_per_call,
            )
            counts.update(
                {
                    "verified": vc["same"] + vc["different"],
                    "same": vc["same"],
                    "different": vc["different"],
                    "unanswered": vc["unanswered"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - verdicts written so far stand
            log(f"verify stage failed part-way: {exc}")
            outcome = "verify-failed"
    stage["verify"] = round(time.time() - t, 1)
    landed = land_yes(load_verdicts(verdicts_path()), manual_path(), conn)
    counts["landed"] = landed
    record = {
        "schema": "anomalica/merge-shortlist-run/1",
        "component": "assimilator",
        "operation": "merge-shortlist",
        "verify_model": model,
        "rerank_model": rerank_model,
        "embed_model": getattr(embed, "model_id", None),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "stage_s": stage,
        "pairs": counts,
        "outcome": outcome,
    }
    log("RUN_JSON " + json.dumps(record))
    runs_path().parent.mkdir(parents=True, exist_ok=True)
    with runs_path().open("a") as f:
        f.write(json.dumps(record) + "\n")
    return EXIT_VERIFY if outcome != "ok" else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="assimilator.merge_pipeline",
        description="Shortlist, rerank and verify merge candidates the rules cannot see; dry run by default.",
    )
    p.add_argument(
        "--db",
        default=os.environ.get("ASSIMILATOR_DB", str(data_dir() / "knowledge.db")),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print PLAN_JSON, call nothing"
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="do it (subscription calls for the verify stage)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="verify model id, resolved through the policy's consolidate stage",
    )
    p.add_argument(
        "--rerank-model",
        default=None,
        help="reranker model id, resolved through the policy's rerank stage",
    )
    p.add_argument("--k", type=int, default=K_NEIGHBOURS)
    p.add_argument(
        "--score-only",
        action="store_true",
        help="shortlist and rerank only (GPU, no model calls); hold the verify stage",
    )
    p.add_argument(
        "--band-floor",
        type=float,
        default=None,
        help="both reranker scores must reach this to be judged (default 0.9)",
    )
    p.add_argument(
        "--cross-type",
        action="store_true",
        help="judge cross-type pairs too (default: same node type only; 11%% precision measured)",
    )
    p.add_argument(
        "--pairs-per-call",
        type=int,
        default=verify_band.PAIRS_PER_CALL,
        help="pairs per verify call (default 20; 50 halves the calls)",
    )
    args = p.parse_args(argv)
    if args.run == args.dry_run:
        print("one of --dry-run or --run", file=sys.stderr)
        return 2

    from anomalica_common import model_policy as mp
    from anomalica_common.embedding_client import embed_texts
    from anomalica_common.llm import (
        _call,
        _parse_json,
        get_usage,
        reset_usage,
        resolve_use_api,
    )

    try:
        policy = mp.load()
        model = (
            policy.check(verify_band.STAGE, args.model)
            if args.model
            else policy.choose(verify_band.STAGE)
        )
        rerank_model = (
            policy.check("rerank", args.rerank_model)
            if args.rerank_model
            else policy.choose("rerank")
        )
    except Exception as exc:  # noqa: BLE001 - every failure is a refusal
        print(f"refused by policy: {exc}", file=sys.stderr)
        return EXIT_POLICY

    class _Embed:
        model_id = None

        def __call__(self, texts):
            mid, v = embed_texts(texts, timeout=900)
            self.model_id = mid
            return v

    def reranker_factory():
        from assimilator.entity_reranker import EntityReranker

        return EntityReranker(model_id=rerank_model)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    reset_usage()
    try:
        code = run_pipeline(
            conn,
            _Embed(),
            reranker_factory,
            _call,
            _parse_json,
            model,
            resolve_use_api("ASSIMILATOR_USE_API"),
            args.dry_run,
            print,
            args.k,
            score_only=args.score_only,
            band_floor=args.band_floor,
            pairs_per_call=args.pairs_per_call,
            cross_type=args.cross_type,
        )
    finally:
        conn.close()
    if args.run:
        print("USAGE_JSON " + json.dumps(get_usage()))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
