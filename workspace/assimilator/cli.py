from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import click

from anomalica_common.digest.yaml_format import parse_digest_yaml
from assimilator.database import (
    find_node_by_name,
    get_claims_for_node,
    get_stats,
    init_db,
)
from assimilator.embeddings import (
    embed_batch,
    embed_text,
    init_vec,
    store_claim_embedding,
    store_node_embedding,
)
from assimilator.import_markdown import import_extraction
from assimilator.scoring import score_claim, tier_label
from assimilator.digest_files import canonical_digests

DEFAULT_DB = Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"


# A full assimilate takes minutes and an hourly timer now runs one, so any manual
# command overlaps it sooner or later. SQLite's default is to fail immediately on
# a locked database, which aborts the import midway and leaves that record stale -
# silently, because the NEXT scheduled run looks normal and nobody re-checks the
# record it dropped. Waiting is always better than failing here: every writer is
# doing the same idempotent fold, so the loser of a race just runs late.
_LOCK_WAIT_MS = 300_000  # 5 minutes - longer than a full corpus pass


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=_LOCK_WAIT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {_LOCK_WAIT_MS}")
    init_db(conn)
    return conn


def _reconcile_suffix(counts: dict) -> str:
    """Show carried/deleted only on a re-import, where they are non-zero."""
    parts = []
    if counts.get("claims_carried"):
        parts.append(f"{counts['claims_carried']} carried")
    if counts.get("claims_deleted"):
        parts.append(f"{counts['claims_deleted']} deleted")
    return f" ({', '.join(parts)})" if parts else ""


@click.group()
@click.option(
    "--db", type=click.Path(), default=str(DEFAULT_DB), envvar="ASSIMILATOR_DB"
)
@click.pass_context
def main(ctx: click.Context, db: str) -> None:
    """Anomalica assimilator - builds the knowledge graph from reviewed digests."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = Path(db)
    ctx.obj["infra_db_path"] = Path(db).parent / "infrastructure.db"


# --- Import: deterministic digest YAML to database ---


@main.command(name="import")
@click.argument("file_path", type=click.Path(exists=True))
@click.pass_context
def import_cmd(ctx: click.Context, file_path: str) -> None:
    """Import a reviewed digest YAML into the graph. No AI involved."""
    path = Path(file_path)
    text = path.read_text()

    click.echo(f"Parsing digest: {path.name}")
    parsed = parse_digest_yaml(text)

    domain_conn = _connect(ctx.obj["db_path"])
    infra_conn = _connect(ctx.obj["infra_db_path"])

    if parsed["domain_claims"]:
        click.echo("Importing domain claims...")
        counts = import_extraction(
            domain_conn,
            parsed,
            section="domain",
            lookup_conns=[infra_conn],
            on_progress=click.echo,
            source_path=str(path),
        )
        click.echo(
            f"  Domain: {counts['nodes_created']} new nodes, "
            f"{counts['nodes_matched']} matched, "
            f"{counts['claims_created']} claims" + _reconcile_suffix(counts)
        )

    if parsed["infrastructure_claims"]:
        click.echo("Importing infrastructure claims...")
        counts = import_extraction(
            infra_conn,
            parsed,
            section="infrastructure",
            lookup_conns=[domain_conn],
            on_progress=click.echo,
            source_path=str(path),
        )
        click.echo(
            f"  Infrastructure: {counts['nodes_created']} new nodes, "
            f"{counts['nodes_matched']} matched, "
            f"{counts['claims_created']} claims" + _reconcile_suffix(counts)
        )

    domain_conn.close()
    infra_conn.close()


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.pass_context
def assimilate(ctx: click.Context, directory: str) -> None:
    """Integrate a directory of digest YAML files into the existing graph.

    Incremental: builds on the current database rather than wiping it. This is
    the everyday verb - point it at the digests directory and it folds each one
    into the accumulating knowledge graph. Use `rebuild` for a clean slate.
    """
    files = canonical_digests(directory)
    if not files:
        click.echo(f"No .yaml digest files found in {directory}")
        return

    click.echo(f"Assimilating {len(files)} digest files from {directory}")
    for i, f in enumerate(files, 1):
        click.echo(f"\n[{i}/{len(files)}] {f.name}")
        ctx.invoke(import_cmd, file_path=str(f))

    domain_conn = _connect(ctx.obj["db_path"])
    s = get_stats(domain_conn)
    click.echo(
        f"\nAssimilation complete. Domain: {s['active_nodes']} nodes, "
        f"{s['records']} records, {s['claims']} claims."
    )
    domain_conn.close()


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--no-replay",
    is_flag=True,
    help="Skip replaying the curation ledger - for a clean reset (e.g. a "
    "re-digest that rewrites the names the ledger keys on).",
)
@click.pass_context
def rebuild(ctx: click.Context, directory: str, no_replay: bool) -> None:
    """Rebuild the graph from a directory of digest YAML files.

    Deletes and recreates both domain and infrastructure databases,
    then imports all .yaml digests from the given directory, and replays the
    curation ledger (use --no-replay to skip, for a clean taxonomy reset).
    """
    db_path = ctx.obj["db_path"]
    infra_path = ctx.obj["infra_db_path"]

    for p in [db_path, infra_path]:
        if p.exists():
            os.remove(p)
            click.echo(f"Deleted {p}")

    directory_path = Path(directory)
    files = canonical_digests(directory_path)
    if not files:
        click.echo(f"No .yaml digest files found in {directory}")
        return

    click.echo(f"Found {len(files)} digest files in {directory}")

    for i, f in enumerate(files, 1):
        click.echo(f"\n[{i}/{len(files)}] {f.name}")
        ctx.invoke(import_cmd, file_path=str(f))

    # Replay the durable curation ledger over the freshly-rebuilt graph - merges
    # are graph-level corrections not held in the digests, so a rebuild loses
    # them unless re-applied (keyed on natural identity; ADR 0038). --no-replay
    # skips this for a clean reset (a re-digest rewrites the names the ledger
    # keys on, so curation restarts fresh after the rebuild).
    domain_conn = _connect(db_path)
    if not no_replay:
        from assimilator.merge import (
            replay_ledger,
            replay_rejections,
            replay_renames,
        )
        from assimilator.propose_pages import replay_vetoes

        replay_ledger(domain_conn, on_progress=click.echo)
        replay_rejections(domain_conn, on_progress=click.echo)
        # Renames run AFTER merges - a renamed node may be a merge survivor whose
        # name the merge replay set first (ADR 0038).
        replay_renames(domain_conn, on_progress=click.echo)
        replay_vetoes(domain_conn, on_progress=click.echo)
    # Work identity is DERIVED from the ingests store, so a rebuild has to
    # recompute it or every rebuild silently drops the duplicate links and
    # restores the inflated source counts they exist to prevent.
    from assimilator.work_identity import link_works

    ingests_root = Path(
        os.environ.get(
            "ANOMALICA_INGESTS_DIR",
            str(Path(__file__).resolve().parents[3] / "ingests"),
        )
    )
    if (ingests_root / "store").is_dir():
        result = link_works(domain_conn, ingests_root)
        click.echo(
            f"Linked works: {result['records']} records resolve to "
            f"{result['works']} works ({result['duplicate_pairs']} duplicate pairs)"
        )
    else:
        click.echo(f"No ingests store at {ingests_root} - work identity not linked")
    s = get_stats(domain_conn)
    click.echo(
        f"\nRebuild complete. Domain: {s['active_nodes']} nodes, "
        f"{s['records']} records, {s['claims']} claims."
    )
    domain_conn.close()


# --- Query commands ---


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show knowledge graph statistics."""
    conn = _connect(ctx.obj["db_path"])
    s = get_stats(conn)
    click.echo(f"Nodes: {s['active_nodes']} (total: {s['nodes']})")
    click.echo(f"Records: {s['records']}")
    click.echo(f"Claims: {s['claims']}")
    click.echo(f"Claim-node references: {s['claim_node_refs']}")
    click.echo(f"Aliases: {s['aliases']}")
    click.echo(f"Corroborations: {s['corroborations']}")
    if s.get("by_type"):
        click.echo("\nBy type:")
        for node_type, count in sorted(s["by_type"].items()):
            click.echo(f"  {node_type}: {count}")
    conn.close()


@main.command()
@click.argument("name")
@click.pass_context
def show(ctx: click.Context, name: str) -> None:
    """Show a node and its claims."""
    conn = _connect(ctx.obj["db_path"])
    node = find_node_by_name(conn, name)
    if node is None:
        click.echo(f"Node not found: {name}")
        return
    click.echo(f"{node.node_type.value}: {node.name} [{node.id[:8]}]")
    if node.metadata:
        for k, v in node.metadata.items():
            click.echo(f"  {k}: {v}")
    claims = get_claims_for_node(conn, node.id)
    if claims:
        click.echo(f"\n{len(claims)} claim(s):")
        for c in claims:
            breakdown = score_claim(conn, c.id)
            label = tier_label(breakdown.score)
            corr_str = ""
            if breakdown.corroboration_count > 0:
                corr_str = f", {breakdown.record_count} sources"
            click.echo(
                f"  [{c.claim_type.value}/{c.attestation.value}] "
                f"({label}, {breakdown.score:.2f}{corr_str}) {c.content}"
            )
    conn.close()


@main.command()
@click.option(
    "--force",
    is_flag=True,
    help="Re-embed rows that already have a vector in the current space.",
)
@click.option("--chunk", default=50, help="Rows per endpoint call and per commit.")
@click.option(
    "--limit",
    default=0,
    help="Stop after this many rows and exit 0. 0 means no limit. Resumability "
    "does the rest - the next run picks up where this one stopped.",
)
@click.option(
    "--bucket",
    default=-1,
    help="Embed only rows in this fixed partition (see embed_batches.BUCKETS). "
    "-1 means every bucket. The partition is a hash of each row's own id, so a "
    "bucket number means the same rows for the life of the corpus.",
)
@click.pass_context
def embed(ctx: click.Context, force: bool, chunk: int, limit: int, bucket: int) -> None:
    """Embed all claims and nodes for similarity search.

    Prefers the embed_service endpoint, which caches every vector by text hash,
    so an interrupted run resumes for free and the workbench audit view shares
    the same computed vectors. Falls back to in-process fastembed when the
    endpoint is down (the in-container path). Either way the space is checked:
    a vector from a different model_id is never written alongside these.

    Resumable by default - rows already embedded in the current space are
    skipped, so re-running after a crash costs only what is left. This matters:
    the model runs one text at a time (batching is broken for this ONNX export),
    so a full corpus pass is hours, not minutes.

    --limit and --bucket exist so the scheduler can queue that pass as many short
    jobs rather than one three-hour one. A single job holds the background lane
    for its whole duration, so a document ingested during it waits behind a task
    with no reason to be atomic. Both are bounded slices of the same idempotent
    work: stopping early is always safe, and nothing needs to know where the last
    run stopped.
    """
    from assimilator.embed_batches import BUCKETS, bucket_of
    from anomalica_common.embedding_client import EmbeddingUnavailable, embed_texts
    from assimilator.embeddings import EMBEDDING_MODEL_ID

    if bucket >= BUCKETS or bucket < -1:
        # Out of range would match no rows and print "nothing to embed", which is
        # exactly what a FINISHED batch prints - a typo would read as success.
        raise click.ClickException(
            f"--bucket must be 0..{BUCKETS - 1} (or -1 for all), got {bucket}"
        )

    conn = _connect(ctx.obj["db_path"])
    init_vec(conn)

    def _already_embedded(kind: str) -> set[str]:
        """Rows of this kind already in the current space. Keyed (kind, id) as
        embedding_model is - a bare-id set would let an embedded claim mark a
        same-id node as done."""
        if force:
            return set()
        return {
            r[0]
            for r in conn.execute(
                "SELECT id FROM embedding_model WHERE kind = ? AND model_id = ?",
                (kind, EMBEDDING_MODEL_ID),
            )
        }

    embedded = 0
    use_endpoint = True
    try:
        model_id, _ = embed_texts(["probe"])
        if model_id != EMBEDDING_MODEL_ID:
            raise click.ClickException(
                f"endpoint is serving {model_id!r} but this graph stores "
                f"{EMBEDDING_MODEL_ID!r} - refusing to mix vector spaces"
            )
    except EmbeddingUnavailable:
        use_endpoint = False
        click.echo("Endpoint unavailable - falling back to in-process fastembed.")

    def _vectors(texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)[1] if use_endpoint else embed_batch(texts)

    for label, kind, query, store in (
        ("claims", "claim", "SELECT id, content FROM claims", store_claim_embedding),
        (
            "nodes",
            "node",
            "SELECT id, name FROM nodes WHERE retired_at IS NULL",
            store_node_embedding,
        ),
    ):
        done = _already_embedded(kind)
        rows = [r for r in conn.execute(query).fetchall() if r[0] not in done]
        if bucket >= 0:
            rows = [r for r in rows if bucket_of(r[0], BUCKETS) == bucket]
        if limit > 0:
            rows = rows[: max(0, limit - embedded)]
        if not rows:
            click.echo(f"{label}: nothing to embed.")
            continue
        click.echo(f"Embedding {len(rows)} {label}...")
        for start in range(0, len(rows), chunk):
            batch = rows[start : start + chunk]
            for (row_id, _), vector in zip(batch, _vectors([r[1] for r in batch])):
                store(conn, row_id, vector)
                embedded += 1
            conn.commit()
            click.echo(f"  {label}: {min(start + chunk, len(rows))}/{len(rows)}")

    conn.close()


@main.command(name="similarity-profile")
@click.option("--neighbours", default=10, help="Nearest neighbours scanned per claim.")
@click.option(
    "--out", type=click.Path(), default=None, help="Write the profile as JSON here."
)
@click.option(
    "--samples", default=25, help="Example pairs to print per threshold band."
)
@click.pass_context
def similarity_profile(
    ctx: click.Context, neighbours: int, out: str | None, samples: int
) -> None:
    """Measure what cosine similarity actually looks like in THIS corpus.

    Every similarity threshold in this repo was set on a vector space that has
    since been fixed, and none was calibrated against the corpus: consolidate
    carries 0.83, corroborate carried 0.99. Both sit above the range the corpus
    occupies, so they select nothing while looking like a working setting. This
    prints the distribution and the candidate count at each cut, plus example
    pairs at each band, so the cut is chosen from evidence.

    Cross-record pairs only - a claim's neighbours within its own record are
    near-duplicates by construction and would flatter every threshold.

    Reads embeddings only. No AI, no plan draw.
    """
    import json as _json
    import statistics

    from assimilator.embeddings import deserialise_f32, search_similar_claims

    conn = _connect(ctx.obj["db_path"])
    init_vec(conn)
    claims = conn.execute("SELECT id, record_id, content FROM claims").fetchall()
    record_of = {cid: rid for cid, rid, _ in claims}
    content_of = {cid: text for cid, _, text in claims}

    pairs: dict[tuple[str, str], float] = {}
    scanned = 0
    for claim_id, record_id, _ in claims:
        row = conn.execute(
            "SELECT embedding FROM vec_claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if not row:
            continue
        scanned += 1
        for match_id, distance in search_similar_claims(
            conn, deserialise_f32(row[0]), limit=neighbours
        ):
            if match_id == claim_id or record_of.get(match_id) == record_id:
                continue
            pairs[tuple(sorted((claim_id, match_id)))] = 1.0 - distance
    conn.close()

    if not pairs:
        raise click.ClickException(
            f"no cross-record neighbour pairs from {scanned} embedded claims - "
            "run `embed` first"
        )

    sims = sorted(pairs.values(), reverse=True)
    cuts = [0.95, 0.90, 0.85, 0.83, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
    profile = {
        "claims_embedded": scanned,
        "cross_record_pairs": len(sims),
        "max": round(sims[0], 4),
        "median": round(statistics.median(sims), 4),
        "min": round(sims[-1], 4),
        "cuts": {str(c): sum(1 for s in sims if s >= c) for c in cuts},
    }
    click.echo(
        f"{scanned} embedded claims, {len(sims)} cross-record neighbour pairs\n"
        f"  max {profile['max']}  median {profile['median']}  min {profile['min']}"
    )
    for cut in cuts:
        click.echo(f"  >= {cut:.2f}: {profile['cuts'][str(cut)]:6d} pairs")

    banked = []
    for a, b in sorted(pairs, key=lambda p: -pairs[p])[:samples]:
        banked.append(
            {
                "similarity": round(pairs[(a, b)], 4),
                "a": {"id": a, "content": content_of[a]},
                "b": {"id": b, "content": content_of[b]},
            }
        )
    click.echo(f"\nTop {len(banked)} pairs:")
    for p in banked:
        click.echo(f"  {p['similarity']:.3f}  {p['a']['content'][:72]}")
        click.echo(f"         {p['b']['content'][:72]}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            _json.dumps({**profile, "top_pairs": banked}, indent=2, ensure_ascii=False)
        )
        click.echo(f"\nWrote {out}")


CORROBORATION_VERIFY_PROMPT = """Below are pairs of claims from different records. For each pair, decide whether they assert the SAME underlying fact or are genuinely DIFFERENT assertions.

RULES:
- "same": the claims make the same factual assertion, possibly with different wording or detail level.
- "different": the claims are about different things, even if they are thematically related.
- Two claims about the same TOPIC but making different ASSERTIONS are "different".
  Example: "The object was 12 metres long" and "The object had no wings" are both about the object, but different assertions.
- Two claims making the same ASSERTION in different words are "same".
  Example: "The object traversed 100km in seconds" and "The UAP covered approximately 100 kilometres almost instantly" are the same.

{pairs_text}

OUTPUT FORMAT (respond with ONLY valid JSON, no markdown fencing):

{{"decisions": [
    {{"pair_id": 1, "verdict": "same"}},
    {{"pair_id": 2, "verdict": "different"}}
]}}"""


@main.command()
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Minimum embedding similarity to consider as candidate. REQUIRED - "
    "measure the corpus first (`similarity-profile`); there is no safe default.",
)
@click.option("--model", default="sonnet", help="Claude model for verification")
@click.option(
    "--rerank",
    is_flag=True,
    help="Apply cross-encoder pre-filter before Claude verification",
)
@click.option(
    "--rerank-min",
    default=0.3,
    type=float,
    help="Drop candidates whose reranker sigmoid score is below this value",
)
@click.option(
    "--limit",
    default=0,
    help="Stop after this many candidate pairs (0 = all). For MEASURING cost "
    "before committing to a full run.",
)
@click.pass_context
def corroborate(
    ctx: click.Context,
    threshold: float | None,
    model: str,
    rerank: bool,
    rerank_min: float,
    limit: int,
) -> None:
    """Find cross-record corroborations: embedding similarity then AI verification.

    --threshold is REQUIRED and deliberately has no default. It used to default
    to 0.99, a number carried over from the degenerate raw-uint8 space. In the
    corrected space a 120-claim sample put every one of 6190 cross-record pairs
    below 0.75, so that default returned nothing - and a run that finds nothing
    reads as "this corpus has no corroboration" rather than "the number is from
    a space that no longer exists". Refusing to run beats answering wrongly.

    Measure before choosing: `assimilator similarity-profile` prints the corpus's
    actual distribution and where a given cut lands.

    With --rerank, a cross-encoder pre-filter scores each candidate pair before
    Claude is consulted. Pairs below --rerank-min are dropped, cutting the
    number of Claude calls without losing genuine corroborations.
    """
    from anomalica_common import model_policy as mp
    from anomalica_common.llm import _call, _parse_json, resolve_use_api
    from assimilator.database import (
        adjudicated_pairs,
        insert_corroboration,
        insert_corroboration_rejection,
    )
    from assimilator.embeddings import deserialise_f32, search_similar_claims

    if threshold is None:
        raise click.UsageError(
            "--threshold is required. There is no defensible default: the old "
            "0.99 came from the pre-decode-fix vector space and silently "
            "returns zero in this one. Run `assimilator similarity-profile` to "
            "see the corpus distribution, then pass a measured cut."
        )

    # POLICY BEFORE SPEND. model-policy.yaml is the source of truth (ADR 0047),
    # and an unlisted model is REFUSED rather than warned about - a warning on a
    # lane nobody is watching is a model choice nobody made. The parser resolves
    # the alias first: `sonnet` appears in no priority list, `claude-sonnet-5`
    # does, so checking before resolving would refuse every Claude dispatch on a
    # naming difference while looking exactly like a policy refusal.
    try:
        model = mp.load().check("corroborate", model)
    except mp.PolicyRefusal as refusal:
        raise click.ClickException(str(refusal)) from refusal

    use_api = resolve_use_api("ASSIMILATOR_USE_API")

    conn = _connect(ctx.obj["db_path"])
    init_vec(conn)

    claims = conn.execute("SELECT id, record_id, content FROM claims").fetchall()
    claim_records = {cid: rid for cid, rid, _ in claims}
    claim_content = {cid: content for cid, _, content in claims}

    candidates = []
    seen_pairs = set()
    for claim_id, record_id, _ in claims:
        emb_row = conn.execute(
            "SELECT embedding FROM vec_claims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if not emb_row:
            continue
        vec = deserialise_f32(emb_row[0])
        matches = search_similar_claims(conn, vec, limit=5)
        for match_id, distance in matches:
            if match_id == claim_id:
                continue
            if claim_records.get(match_id) == record_id:
                continue
            pair_key = tuple(sorted([claim_id, match_id]))
            if pair_key in seen_pairs:
                continue
            similarity = 1.0 - distance
            if similarity >= threshold:
                seen_pairs.add(pair_key)
                candidates.append((claim_id, match_id, similarity))

    click.echo(f"Found {len(candidates)} candidate pairs above {threshold} similarity")

    if rerank and candidates:
        from assimilator.search import _sigmoid, rerank_pairs

        pairs = [(claim_content[a], claim_content[b]) for a, b, _ in candidates]
        click.echo(f"  Reranking {len(pairs)} pairs with cross-encoder...")
        raw_scores = rerank_pairs(pairs)
        ce_scores = [_sigmoid(s) for s in raw_scores]
        filtered = [
            (a, b, sim, ce)
            for (a, b, sim), ce in zip(candidates, ce_scores)
            if ce >= rerank_min
        ]
        click.echo(
            f"  {len(filtered)}/{len(candidates)} pairs retained after rerank "
            f"(min sigmoid {rerank_min})"
        )
        candidates = [(a, b, sim) for a, b, sim, _ in filtered]

    if not candidates:
        conn.close()
        return

    # Skip pairs already decided either way. Without this a rejection is bought
    # again on every run: 26 of the first 86 were rejected, and an automated lane
    # would re-pay for those 26 verdicts forever.
    decided = adjudicated_pairs(conn)
    if decided:
        before = len(candidates)
        candidates = [c for c in candidates if (c[0], c[1]) not in decided]
        skipped = before - len(candidates)
        if skipped:
            click.echo(
                f"  Skipping {skipped} pairs already adjudicated in an earlier run"
            )

    if not candidates:
        click.echo("Nothing new to verify - every candidate has been decided.")
        return

    if limit > 0 and len(candidates) > limit:
        click.echo(
            f"  --limit {limit}: measuring on the first {limit} of {len(candidates)}"
        )
        candidates = candidates[:limit]

    # Corroboration spends the plan and recorded NOTHING about it: it runs outside
    # the scheduler so it produces no dispatch row, and the corroborations table
    # keeps claim_a, claim_b and similarity with no model, timestamp or usage. So
    # the cost of the next run could not be sized from the last one. The transport
    # has accumulated usage all along; nobody asked it.
    from anomalica_common.llm import get_usage, reset_usage

    reset_usage()

    batch_size = 20
    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start : batch_start + batch_size]
        lines = []
        for i, (cid_a, cid_b, sim) in enumerate(batch, 1):
            lines.append(f"PAIR {i} (similarity: {sim:.3f}):")
            lines.append(f'  A: "{claim_content[cid_a]}"')
            lines.append(f'  B: "{claim_content[cid_b]}"')
            lines.append("")

        prompt = CORROBORATION_VERIFY_PROMPT.format(pairs_text="\n".join(lines))
        click.echo(f"  Verifying pairs {batch_start + 1}-{batch_start + len(batch)}...")
        raw = _call(prompt, "", model, use_api=use_api)
        data = _parse_json(raw)
        decisions = {
            d.get("pair_id"): d.get("verdict") for d in data.get("decisions", [])
        }

        for i, (cid_a, cid_b, sim) in enumerate(batch, 1):
            verdict = decisions.get(i, "different")
            if verdict == "same":
                insert_corroboration(conn, cid_a, cid_b, sim)
            else:
                insert_corroboration_rejection(conn, cid_a, cid_b, sim, model)

    conn.commit()
    actual = conn.execute("SELECT COUNT(*) FROM corroborations").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM corroboration_rejections").fetchone()[
        0
    ]
    click.echo(
        f"Verified {actual} genuine corroborations (from {len(candidates)} candidates "
        f"this run); {rejected} rejections recorded and will not be re-bought"
    )

    usage = get_usage() or {}
    tin = (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    tout = int(usage.get("output_tokens") or 0)
    pairs = len(candidates)
    click.echo(
        f"Usage: {tin:,} input + {tout:,} output tokens over {pairs} pairs"
        + (f" = {tin // pairs:,} in / {tout // pairs:,} out per pair" if pairs else "")
        + f"  [model {model}, transport {'api' if use_api else 'subscription'}]"
    )
    if pairs:
        click.echo(
            f"  extrapolated to 340 further pairs: "
            f"{tin // pairs * 340:,} input + {tout // pairs * 340:,} output"
        )
    conn.close()


@main.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of results")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "semantic", "keyword"]),
    default="hybrid",
    help="Retrieval mode (default: hybrid embedding+keyword via RRF)",
)
@click.option(
    "--rerank",
    is_flag=True,
    help="Apply cross-encoder rerank pass (requires sentence-transformers)",
)
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, mode: str, rerank: bool) -> None:
    """Search claims by semantic similarity, keyword match, or hybrid (default)."""
    from assimilator.embeddings import search_similar_claims
    from assimilator.search import hybrid_search_claims, keyword_search_claims

    conn = _connect(ctx.obj["db_path"])
    init_vec(conn)

    if mode == "keyword":
        results = keyword_search_claims(conn, query, limit=limit)
        results = [(cid, 1.0 - score) for cid, score in results]
    elif mode == "semantic":
        query_embedding = embed_text(query)
        results = search_similar_claims(conn, query_embedding, limit=limit)
    else:
        query_embedding = embed_text(query)
        results = hybrid_search_claims(
            conn, query, query_embedding, limit=limit, rerank=rerank
        )

    if not results:
        click.echo("No results. Run 'embed' first if you expected semantic matches.")
        conn.close()
        return

    for claim_id, distance in results:
        similarity = 1.0 - distance
        row = conn.execute(
            "SELECT content, claim_type FROM claims WHERE id = ?", (claim_id,)
        ).fetchone()
        if row:
            click.echo(f"  [{similarity:.2f}] ({row[1]}) {row[0]}")
    conn.close()


# --- Digest maintenance passes (rewrite digest YAML in place) ---


@main.command(name="reclassify-documents")
@click.argument("extracts_dir", type=click.Path(exists=True))
def reclassify_documents_cmd(extracts_dir: str) -> None:
    """Rewrite object/matter nodes that look like documents to type 'document'.

    Conservative pattern match against suffixes like Memo, Report, Letter,
    Article, Paper, Book, Brief, Slides, Video, Disclosure, Statement,
    Testimony, Affidavit - excluding nodes whose names also contain System,
    Programme, Program, Centre, Database, Network. Writes changes back to
    each .yaml digest file; rebuild the graph afterwards to pick them up.
    """
    from assimilator.reclassify import reclassify_documents_in_dir

    results = reclassify_documents_in_dir(Path(extracts_dir))
    total = sum(results.values())
    click.echo(
        f"Reclassified {total} nodes to type 'document' across {len(results)} files."
    )
    for fname, count in sorted(results.items(), key=lambda x: -x[1]):
        click.echo(f"  {count:4d}  {fname}")


@main.command(name="normalise-names")
@click.argument("extracts_dir", type=click.Path(exists=True))
def normalise_names_cmd(extracts_dir: str) -> None:
    """Rewrite person and place names in extracts to canonical formats.

    Persons: "First Last" -> "Last, First" (strips rank prefixes like
    "Commander", preserves Jr/Sr/II/III on surname).

    Places: "City State" -> "Country, State, City" using a known list of
    US states, Australian states, Canadian provinces, UK countries, NZ
    regions. Other places left unchanged.

    Rebuild the graph after to pick up renames.
    """
    from assimilator.reclassify import (
        normalise_person_names_in_dir,
        normalise_place_names_in_dir,
    )

    person_results = normalise_person_names_in_dir(Path(extracts_dir))
    place_results = normalise_place_names_in_dir(Path(extracts_dir))
    person_total = sum(person_results.values())
    place_total = sum(place_results.values())
    click.echo(
        f"Normalised {person_total} person names across {len(person_results)} files."
    )
    click.echo(
        f"Normalised {place_total} place names across {len(place_results)} files."
    )


@main.command(name="naturalise-person-names")
@click.argument("digests_dir", type=click.Path(exists=True))
@click.option(
    "--dry-run", is_flag=True, help="Report what would change without writing."
)
def naturalise_person_names_cmd(digests_dir: str, dry_run: bool) -> None:
    """Rewrite person names in digest YAML to natural order, surname as a field.

    "Fravor, David" -> "David Fravor" across node names, claim refs and speaker
    fields, adding `metadata.family_name` (the surname has nowhere else to live
    once the comma goes) and keeping the surname-first form in
    `metadata.aliases` so last-first input still resolves. Deterministic, no AI.

    PLACES ARE NOT TOUCHED: "USA, Nevada, Area 51" is largest-unit-first and
    that convention stands. Rebuild the graph afterwards to pick the names up.
    """
    from assimilator.person_names import naturalise_digests_in_dir

    results = naturalise_digests_in_dir(Path(digests_dir), dry_run=dry_run)
    total = sum(results.values())
    verb = "Would rename" if dry_run else "Renamed"
    click.echo(f"{verb} {total} person nodes across {len(results)} digests.")
    for fname, count in sorted(results.items(), key=lambda x: -x[1]):
        click.echo(f"  {count:4d}  {fname}")


@main.command(name="migrate-refs-delimiter")
@click.argument("extracts_dir", type=click.Path(exists=True))
def migrate_refs_delimiter_cmd(extracts_dir: str) -> None:
    """Migrate refs lines from comma to semicolon delimiter.

    Use this after person/place names have been renamed to include commas
    (Last, First / Country, Region, ...). Builds a node-name dictionary from
    all extract files and greedily disambiguates each refs line by matching
    against the dictionary, so comma-containing names round-trip cleanly.
    """
    from assimilator.reclassify import migrate_refs_delimiter_in_dir

    results = migrate_refs_delimiter_in_dir(Path(extracts_dir))
    total = sum(results.values())
    click.echo(f"Migrated {total} refs lines across {len(results)} files.")


@main.command(name="rewire-refs")
@click.argument("extracts_dir", type=click.Path(exists=True))
def rewire_refs_cmd(extracts_dir: str) -> None:
    """Recovery pass: update refs/speakers after a rename that missed them.

    For each renamed person/place node ("Last, First" or "Country, Region, ..."),
    compute the pre-rename form and rewrite any refs/speaker lines still
    pointing at the old name. Use after a `normalise-names` run that pre-dates
    the in-file ref rewiring.
    """
    from assimilator.reclassify import rewire_refs_in_dir

    results = rewire_refs_in_dir(Path(extracts_dir))
    total = sum(results.values())
    click.echo(f"Rewired {total} references across {len(results)} files.")
    for fname, count in sorted(results.items(), key=lambda x: -x[1]):
        click.echo(f"  {count:4d}  {fname}")


@main.command(name="backfill-record-fields")
@click.argument("digests_dir", type=click.Path(exists=True))
@click.option(
    "--ingests-dir",
    type=click.Path(),
    default=None,
    help="Path to the ingests repo (default: ANOMALICA_INGESTS_DIR or derived "
    "from the digests location)",
)
def backfill_record_fields_cmd(digests_dir: str, ingests_dir: str | None) -> None:
    """Backfill content_hash/publisher/medium/duration into legacy digest blocks.

    Deterministic, no AI: the values come straight from each record's ingest
    frontmatter. Only the record: block is rewritten; nodes and claims are left
    byte-identical. Rebuild the graph afterwards if you want them there too.
    """
    from assimilator.backfill import backfill_record_fields_in_dir

    results = backfill_record_fields_in_dir(
        Path(digests_dir), Path(ingests_dir) if ingests_dir else None
    )
    if not results:
        click.echo("No digests needed backfilling.")
        return
    total = sum(len(v) for v in results.values())
    click.echo(f"Backfilled {total} fields across {len(results)} digests:")
    for fname, fields in sorted(results.items()):
        click.echo(f"  {fname}: {', '.join(fields)}")


@main.command(name="schedule")
@click.option(
    "--ingests",
    type=click.Path(),
    default=lambda: os.environ.get("ANOMALICA_INGESTS_DIR"),
    help="Path to the ingests repo (default: ANOMALICA_INGESTS_DIR or ../ingests)",
)
@click.option(
    "--digests",
    type=click.Path(),
    default=lambda: os.environ.get("ANOMALICA_DIGESTS_DIR"),
    help="Path to the digests repo (default: ANOMALICA_DIGESTS_DIR or ../digests)",
)
@click.option(
    "--sources",
    type=click.Path(),
    default=lambda: os.environ.get("ANOMALICA_SOURCES_DIR"),
    help="Path to the raw sources dir (default: ANOMALICA_SOURCES_DIR or ../sources)",
)
@click.option(
    "--out",
    type=click.Path(),
    default=None,
    help="Where to write the queue JSON (default: SCHEDULER_QUEUE_PATH)",
)
@click.pass_context
def schedule_cmd(
    ctx: click.Context,
    ingests: str | None,
    digests: str | None,
    sources: str | None,
    out: str | None,
) -> None:
    """Enumerate the real pending pipeline jobs from current corpus state.

    Writes a prioritised queue JSON the workbench reads for its Schedule view.
    Real work only - ingest/review/digest/corroborate jobs computed from actual
    state, ranked by per-job-specific drivers. No AI, no money.
    """
    from assimilator import scheduler

    # Shares resolve_corpus_dirs + run_schedule with the host-runnable
    # `python -m assimilator.scheduler` entry, so the in-container CLI and the
    # workbench's host invocation produce an identical queue.
    queue, out_path = scheduler.run_schedule(
        ctx.obj["db_path"], ingests, digests, sources, out
    )
    by_lane: dict[str, int] = {}
    for job in queue["jobs"]:
        by_lane[job["lane"]] = by_lane.get(job["lane"], 0) + 1
    click.echo(f"Wrote {out_path}")
    click.echo(
        f"  {len(queue['jobs'])} jobs "
        f"({', '.join(f'{n} {lane}' for lane, n in sorted(by_lane.items()))}), "
        f"{len(queue['reviewQueue'])} awaiting review, "
        f"{len(queue['recordDemand'])} records with graph demand"
    )


@main.command(name="backfill-claim-hashes")
@click.pass_context
def backfill_claim_hashes_cmd(ctx: click.Context) -> None:
    """Compute claim_hash for every existing claim in the graph.

    The claim_hash column is added empty on migration; this fills it. Pure
    compute, no AI - the resolved graph ids are already stored. Run once after
    upgrading; the importer maintains the hash for all claims it touches after.
    """
    from assimilator.import_markdown import backfill_claim_hashes

    for label, key in (("Domain", "db_path"), ("Infrastructure", "infra_db_path")):
        conn = _connect(ctx.obj[key])
        result = backfill_claim_hashes(conn, on_progress=click.echo)
        click.echo(f"  {label}: {result['updated']} claims hashed")
        conn.close()


@main.command(name="propose-pages")
@click.pass_context
def propose_pages_cmd(ctx: click.Context) -> None:
    """Recompute the article-proposal set from the page-worthiness gate.

    Deterministic, no AI: scores every active node by type tier + independent-
    source floor (page_gate.py), excludes vetoed nodes, and writes the derived
    page_proposals table. The dependency gate before synthesise - a brief is only
    emitted for a proposed node. Run after a merge pass; re-run when the graph
    moves.
    """
    from assimilator.propose_pages import propose

    conn = _connect(ctx.obj["db_path"])
    rows = propose(conn)
    by_tier: dict[str, int] = {}
    for r in rows:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    click.echo(f"{len(rows)} page proposals ({by_tier})")
    dominated = sum(1 for r in rows if r["second_source_claims"] < 3)
    if dominated:
        click.echo(
            f"  {dominated} of them rest on a single source (second source "
            f"contributes <3 claims) - see `source-spread`"
        )
    conn.close()


@main.command(name="duplicate-records")
@click.option(
    "--ingests",
    type=click.Path(),
    default=lambda: os.environ.get("ANOMALICA_INGESTS_DIR"),
    help="Path to the ingests repo (default: ANOMALICA_INGESTS_DIR or ../ingests)",
)
@click.option("--threshold", default=None, type=float, help="Jaccard cut.")
def duplicate_records_cmd(ingests: str | None, threshold: float | None) -> None:
    """Find records that are the same WORK under different content hashes.

    A record is addressed by its exact bytes, so one work enters the store again
    on any re-download, re-export, edition change or OCR pass - and every consumer
    counting distinct records then counts one work as several sources. That does
    not merely evade the independence floor, it INVERTS the source-spread metric:
    a node whose claims all come from one book is correctly flagged today, and
    reads as excellently spread once the book is present twice.

    Two complementary passes, both deterministic and offline: shingle overlap
    (catches two files of one book, which share no URL) and exact source_url /
    source_id match (catches one URL fetched twice, whose text may have drifted
    past any similarity cut).
    """
    from assimilator.work_identity import (
        DEFAULT_JACCARD,
        find_duplicate_records,
        find_same_origin_records,
        live_record_paths,
        unreachable_live_records,
    )

    root = Path(ingests) if ingests else Path(__file__).resolve().parents[3] / "ingests"
    store = root / "store"
    if not store.is_dir():
        raise click.ClickException(f"no ingests store at {store}")

    # The LIVE set, via by-name/ - not a store glob. store/ also holds archive
    # tiers whose records are superseded re-ingests of live ones, and comparing
    # against those reports every one of them as a duplicate.
    live = live_record_paths(root)
    unreachable = unreachable_live_records(root)
    click.echo(f"Scanning {len(live)} records via by-name/.")
    if unreachable:
        # Never report a coverage number without its gap: a scan that says "all
        # records" while records/ has drifted is exactly how a coverage claim
        # gets overstated.
        click.echo(
            f"  WARNING: {len(unreachable)} live store record(s) are NOT reachable "
            f"via by-name/ and were NOT scanned:"
        )
        for record_hash in unreachable:
            click.echo(f"    {record_hash}")
    pairs = find_duplicate_records(store, threshold or DEFAULT_JACCARD, paths=live)
    origin = find_same_origin_records(store, paths=live)
    known = {frozenset((p.a, p.b)) for p in pairs}
    combined = pairs + [p for p in origin if frozenset((p.a, p.b)) not in known]

    if not combined:
        click.echo("No duplicate records found.")
        return
    click.echo(f"{len(combined)} duplicate record pair(s):")
    for p in combined:
        detail = (
            f"jaccard {p.jaccard:.4f} ({p.shared}/{p.union} shingles)"
            if p.reason == "text"
            else f"same {p.reason}"
        )
        click.echo(f"  {detail}\n    {p.a}\n    {p.b}")


@main.command(name="replay-curation")
@click.pass_context
def replay_curation_cmd(ctx: click.Context) -> None:
    """Re-apply the durable curation ledger to the CURRENT graph, without wiping it.

    Until now the ledger could only be replayed as part of `rebuild`, which
    deletes and re-imports everything. That is wrong for the incremental path: a
    new digest can reintroduce a node an earlier merge retired (it matches by
    name against live nodes only, so the merged-away name looks new), and without
    a replay the duplicate simply stands. Same operations, same order as rebuild
    - merges, rejections, renames, then vetoes - and idempotent, so it is safe on
    a timer.

    Deterministic, no AI.
    """
    from assimilator.merge import replay_ledger, replay_rejections, replay_renames
    from assimilator.propose_pages import replay_vetoes

    conn = _connect(ctx.obj["db_path"])
    replay_ledger(conn, on_progress=click.echo)
    replay_rejections(conn, on_progress=click.echo)
    replay_renames(conn, on_progress=click.echo)
    replay_vetoes(conn, on_progress=click.echo)
    conn.close()


@main.command(name="link-works")
@click.option(
    "--ingests",
    type=click.Path(),
    default=lambda: os.environ.get("ANOMALICA_INGESTS_DIR"),
    help="Path to the ingests repo (default: ANOMALICA_INGESTS_DIR or ../ingests)",
)
@click.pass_context
def link_works_cmd(ctx: click.Context, ingests: str | None) -> None:
    """Collapse duplicate records onto a shared work id, so sources count once.

    Everything that counts "sources" counts distinct works, not distinct records:
    one work becomes several records on any re-ingest or edition change, and a
    book present twice would otherwise clear a two-source floor on its own. Runs
    the duplicate detectors over the live record set and links what they find.

    Derived and idempotent - recompute it after a rebuild rather than replaying
    it. Deterministic, no AI.
    """
    from assimilator.work_identity import link_works

    root = Path(ingests) if ingests else Path(__file__).resolve().parents[3] / "ingests"
    conn = _connect(ctx.obj["db_path"])
    result = link_works(conn, root)
    conn.close()
    click.echo(
        f"{result['duplicate_pairs']} duplicate pair(s) in the store; "
        f"{result['records_linked']} graph record(s) linked. "
        f"{result['records']} records resolve to {result['works']} works."
    )


@main.command(name="page-floor")
@click.option(
    "--max-dominance", default=0.80, help="Reject if one source exceeds this share."
)
@click.option("--min-origins", default=2, help="Independent provenance roots required.")
@click.option(
    "--max-unscored",
    default=0.25,
    help="At or above this share of chainless claims, independence is "
    "UNCOMPUTABLE and the test is skipped rather than failed. INCLUSIVE.",
)
@click.pass_context
def page_floor_cmd(
    ctx: click.Context, max_dominance: float, min_origins: int, max_unscored: float
) -> None:
    """What a proposed page floor would admit. REPORTS, never enforces.

    Two conditions, because they catch different objectionable pages and neither
    subsumes the other. DOMINANCE - no single source above 80% of a page's claims
    - catches the single-book summary, which independence cannot see: a book
    quoting twenty witnesses has twenty provenance roots and is still one
    author's selection and paraphrase. INDEPENDENCE - at least two distinct
    provenance roots - catches several records relaying one origin, which
    dominance cannot see.

    AN UNCOMPUTABLE CONDITION MUST NOT REJECT. Where too many of a node's claims
    predate ADR 0044, independence cannot be computed, and failing the page for
    that is absence read as a failing score - invisibly, since the page simply
    would not be proposed. Those pages skip the independence test. The clause is
    self-correcting rather than a loophole: as pre-0044 records are re-digested
    the unscored share falls, the score becomes trustworthy, and some of them are
    then rejected on evidence.
    """
    conn = _connect(ctx.obj["db_path"])
    rows = conn.execute(
        "SELECT n.name, p.tier, p.claim_count, p.top_source_claims, "
        "p.independent_source_count, p.unscored_claims "
        "FROM page_proposals p JOIN nodes n ON n.id = p.node_id"
    ).fetchall()
    conn.close()
    if not rows:
        raise click.ClickException("no proposals - run `propose-pages` first")

    worthy = [r for r in rows if r[1] == "page-worthy"]
    admitted, rescued, cut_dominance, cut_origins = [], [], 0, 0
    for name, _tier, claims, top, origins, unscored in worthy:
        if claims > 0 and (top or 0) / claims >= max_dominance:
            cut_dominance += 1
            continue
        # INCLUSIVE boundary: a page AT the threshold counts as untested. The
        # same figure serves the disclosure on the page-floor card and the
        # tranche trend, so it must be one convention, and the disclosure is the
        # stricter master - calling a tested page untested costs a footnote,
        # calling an untested page tested is the failure. Nine pages sat exactly
        # on the line when this was settled and the two readings differed by 6,
        # which is three times the movement the trend exists to detect: the
        # convention would have out-swung the signal.
        if not (claims > 0 and (unscored or 0) / claims < max_unscored):
            rescued.append(name)
            admitted.append(name)
            continue
        if (origins or 0) < min_origins:
            cut_origins += 1
            continue
        admitted.append(name)

    # Every figure carries its denominator. The untested COUNT moves for two
    # independent reasons - records gaining provenance, and new proposals
    # appearing - so across a re-digestion tranche that was demonstrably working
    # it ROSE from 99 to 101 while the FRACTION fell from 32.4% to 28.2%. A
    # tracking metric that moves for two reasons cannot be read as progress on
    # either, and this is the follow-up measure on the page-floor card.
    pass_dominance = len(worthy) - cut_dominance
    tested = pass_dominance - len(rescued)
    click.echo(f"{len(worthy)} page-worthy proposals")
    click.echo(
        f"  admitted             {len(admitted):4d} / {len(worthy):4d}"
        f"  {100 * len(admitted) / len(worthy):5.1f}%"
    )
    click.echo(f"  cut: one source >= {max_dominance:.0%}  {cut_dominance:4d}")
    click.echo(f"  cut: fewer than {min_origins} origins {cut_origins:4d}")
    click.echo(
        f"  independence TESTED  {tested:4d} / {pass_dominance:4d}"
        f"  {100 * tested / pass_dominance:5.1f}%  of those passing dominance"
    )
    click.echo(
        f"  admitted UNTESTED    {len(rescued):4d} / {len(admitted):4d}"
        f"  {100 * len(rescued) / len(admitted):5.1f}%  of the admitted set"
    )
    click.echo(
        f"      (>= {max_unscored:.0%} of their claims predate the chain)\n"
        "      Track the PERCENTAGES across tranches, not the counts - the\n"
        "      counts also move as the corpus grows."
    )
    for name in rescued[:8]:
        click.echo(f"      {name[:62]}")


@main.command(name="source-spread")
@click.option(
    "--min-second",
    default=3,
    help="Claims the SECOND source must contribute for a page to look corroborated.",
)
@click.option("--limit", default=25, help="Worst offenders to list.")
@click.pass_context
def source_spread_cmd(ctx: click.Context, min_second: int, limit: int) -> None:
    """How concentrated each proposed page's evidence is in one source.

    The gate counts DISTINCT sources, so a node with 17 claims - 16 from one
    copyrighted book and 1 from a passing mention - reports source_count = 2 and
    passes as corroborated. It is in substance a summary of that one book. This
    reports the spread so the size of the problem is known before a floor is set
    on it; NOTHING is gated on these numbers yet.

    Deterministic, no AI. Run `propose-pages` first.
    """
    conn = _connect(ctx.obj["db_path"])
    rows = conn.execute(
        "SELECT p.node_id, n.name, p.node_type, p.claim_count, p.source_count, "
        "p.top_source_claims, p.second_source_claims "
        "FROM page_proposals p JOIN nodes n ON n.id = p.node_id "
        "WHERE p.top_source_claims IS NOT NULL "
        "ORDER BY p.claim_count DESC"
    ).fetchall()
    if not rows:
        raise click.ClickException("no proposals with spread - run `propose-pages`")

    failing = [r for r in rows if r[6] < min_second]
    click.echo(
        f"{len(rows)} proposals; {len(failing)} "
        f"({100 * len(failing) / len(rows):.0f}%) rest on a single source "
        f"(second source contributes <{min_second} claims)"
    )
    click.echo(f"\n{'claims':>6} {'srcs':>5} {'top':>5} {'2nd':>5}  node\n" + "-" * 72)
    for _nid, name, node_type, claims, sources, top, second in failing[:limit]:
        click.echo(
            f"{claims:6d} {sources:5d} {top:5d} {second:5d}  {name[:44]} ({node_type})"
        )
    conn.close()


@main.command(name="export-obsidian")
@click.argument("output_dir", type=click.Path())
@click.pass_context
def export_obsidian_cmd(ctx: click.Context, output_dir: str) -> None:
    """Export the knowledge graph as a navigable Obsidian markdown vault.

    Creates Records/*.md (one per record, with all claims linking to [[Node]]s)
    and Nodes/*.md (stubs - rely on Obsidian's backlinks panel to surface
    incoming claim references). Open the output directory as a vault.
    """
    from assimilator.obsidian_export import export_to_obsidian

    domain_conn = _connect(ctx.obj["db_path"])
    infra_conn = _connect(ctx.obj["infra_db_path"])

    out = Path(output_dir)
    counts = export_to_obsidian(out, domain_conn, infra_conn)

    domain_conn.close()
    infra_conn.close()

    click.echo(f"Exported to {out}:")
    click.echo(f"  Records: {counts['records']}")
    click.echo(f"  Nodes:   {counts['nodes']}")
    click.echo(f"  Claims:  {counts['claims']}")
    click.echo(f"\nOpen {out} as an Obsidian vault to navigate.")


@main.command("veto-pages")
@click.argument("node_ids", nargs=-1, required=True)
@click.option(
    "--reason",
    default=None,
    help="Why this is never a page (free text, read by humans)",
)
@click.option("--by", default=None, help="Who decided")
@click.pass_context
def veto_pages_cmd(
    ctx: click.Context, node_ids: tuple[str, ...], reason: str | None, by: str | None
) -> None:
    """Record an editorial "never a page" for one or more nodes.

    veto_pages() has existed since page proposals did, with a durable ledger and
    replay on rebuild, and nothing exposed it - so the decision could be made in
    code and never by a person. This is the entry point the workbench calls.

    Durable and replayable: the ledger entry is keyed on natural identity, so it
    survives the rebuild that discards the node ids.
    """
    import uuid

    from assimilator.propose_pages import veto_pages

    conn = _connect(ctx.obj["db_path"])
    try:
        veto_pages(conn, list(node_ids), reason, str(uuid.uuid4()), created_by=by)
        conn.commit()
    finally:
        conn.close()
    click.echo(f"Vetoed {len(node_ids)} node(s); recorded in the curation ledger.")


@main.command("doctor")
@click.option(
    "--briefs",
    default=lambda: os.environ.get("ANOMALICA_BRIEFS_DIR"),
    help="Briefs dir (default: ANOMALICA_BRIEFS_DIR or ~/.local/share/assimilator/briefs)",
)
@click.option("--content", default=None, help="Path to the content repo's pages/ dir")
@click.pass_context
def doctor_cmd(ctx: click.Context, briefs: str | None, content: str | None) -> None:
    """Check the derived stages agree with each other. Read-only.

    Everything after the digests is derived and rebuildable, which is what lets it
    drift in silence: nothing fails when a proposal points at a node a merge
    retired, or a page is built from a brief the graph has moved past. The lane
    runs, exits 0, writes something, and describes a corpus that no longer exists.

    Exits 1 when anything is inconsistent, so it can gate a pipeline run.
    """
    from assimilator.consistency import check_all
    from assimilator.synthesise import default_briefs_dir

    briefs_dir = Path(briefs) if briefs else default_briefs_dir()
    content_dir = Path(content) if content else None
    if content_dir is None:
        guess = Path(__file__).resolve().parents[3] / "content" / "pages"
        content_dir = guess if guess.is_dir() else None

    conn = _connect(ctx.obj["db_path"])
    try:
        findings = check_all(conn, briefs_dir, content_dir)
    finally:
        conn.close()

    if not findings:
        click.echo("Consistent: proposals, briefs and pages all agree with the graph.")
        return
    for f in findings:
        click.echo(f"\n{f.check}: {f.count}")
        click.echo(f"  {f.detail}")
        for s in f.samples:
            click.echo(f"    {s}")
        if len(f.samples) < f.count:
            click.echo(f"    ... and {f.count - len(f.samples)} more")
        if f.repair:
            click.echo(f"  fix: {f.repair}")
    raise SystemExit(1)


@main.command("publish-briefs")
@click.option(
    "--briefs", default=None, help="Briefs dir (default: ANOMALICA_BRIEFS_DIR)."
)
@click.option(
    "--out",
    default=None,
    help="Where to write publishable briefs (default: content/briefs).",
)
@click.option(
    "--store",
    default=None,
    help="Ingests store dir - the copyright AUTHORITY (default: ANOMALICA_INGESTS_DIR/store).",
)
@click.option(
    "--dry-run", is_flag=True, help="Report what would be withheld, write nothing."
)
@click.pass_context
def publish_briefs_cmd(
    ctx: click.Context,
    briefs: str | None,
    out: str | None,
    store: str | None,
    dry_run: bool,
) -> None:
    """Write briefs for publication, with copyright excerpts redacted.

    A brief is the audit record (ADR 0010) - the exact material the writer saw -
    so publishing it beside the page makes "every assertion traces to a source"
    checkable rather than asserted.

    Excerpts are withheld for any source that is not public_domain,
    publicly_accessible or open_licence, INCLUDING any the store cannot resolve.
    Withholding is marked on the claim, never silent: a reader must be able to
    tell a licence boundary from a gap in our evidence.

    Copyright is read live from the ingests store, not from the digest's
    snapshot. The snapshot is right for filtering; this decision is irreversible.
    """
    import os

    from assimilator.publish_briefs import publish_briefs
    from assimilator.synthesise import default_briefs_dir

    briefs_dir = Path(briefs) if briefs else default_briefs_dir()
    store_dir = (
        Path(store)
        if store
        else Path(
            os.environ.get(
                "ANOMALICA_INGESTS_DIR", Path.home() / "repos/anomalica/ingests"
            )
        )
        / "store"
    )
    out_dir = Path(out) if out else Path.home() / "repos/anomalica/content" / "briefs"
    if not store_dir.is_dir():
        raise click.ClickException(
            f"ingests store not found at {store_dir} - refusing to publish without "
            "the copyright authority, because every record would read as unknown "
            "and every excerpt would be withheld, which looks like success"
        )

    if dry_run:
        from assimilator.publish_briefs import _load, redact_brief

        totals: dict[str, int] = {}
        n = 0
        for path in sorted(briefs_dir.glob("*.yaml")):
            try:
                brief = _load(path.read_text())
            except Exception:
                continue
            if not isinstance(brief, dict):
                continue
            _, counts = redact_brief(brief, store_dir)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            n += 1
        click.echo(f"DRY RUN over {n} briefs. Claim excerpts by source status:")
        for k, v in sorted(totals.items(), key=lambda x: -x[1]):
            mark = (
                "published"
                if k in {"public_domain", "publicly_accessible", "open_licence"}
                else "WITHHELD"
            )
            click.echo(f"   {k:22} {v:6}  {mark}")
        return

    stats = publish_briefs(briefs_dir, out_dir, store_dir)
    click.echo(f"Wrote {stats['briefs']} briefs to {out_dir}")
    click.echo(f"  excerpts withheld on {stats['withheld_claims']} claims")
    for k, v in sorted(stats["by_status"].items(), key=lambda x: -x[1]):
        click.echo(f"   {k:22} {v}")


if __name__ == "__main__":
    main()
