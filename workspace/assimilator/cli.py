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

DEFAULT_DB = Path.home() / ".local" / "share" / "assimilator" / "knowledge.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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
    files = sorted(Path(directory).glob("**/*.yaml"))
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
@click.pass_context
def rebuild(ctx: click.Context, directory: str) -> None:
    """Rebuild the graph from a directory of digest YAML files.

    Deletes and recreates both domain and infrastructure databases,
    then imports all .yaml digests from the given directory.
    """
    db_path = ctx.obj["db_path"]
    infra_path = ctx.obj["infra_db_path"]

    for p in [db_path, infra_path]:
        if p.exists():
            os.remove(p)
            click.echo(f"Deleted {p}")

    directory_path = Path(directory)
    files = sorted(directory_path.glob("**/*.yaml"))
    if not files:
        click.echo(f"No .yaml digest files found in {directory}")
        return

    click.echo(f"Found {len(files)} digest files in {directory}")

    for i, f in enumerate(files, 1):
        click.echo(f"\n[{i}/{len(files)}] {f.name}")
        ctx.invoke(import_cmd, file_path=str(f))

    domain_conn = _connect(db_path)
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
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Embed all claims and nodes for similarity search."""
    conn = _connect(ctx.obj["db_path"])
    init_vec(conn)

    rows = conn.execute("SELECT id, content FROM claims").fetchall()
    if rows:
        click.echo(f"Embedding {len(rows)} claims...")
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        embeddings = embed_batch(texts)
        for claim_id, emb in zip(ids, embeddings):
            store_claim_embedding(conn, claim_id, emb)
        click.echo(f"  Stored {len(rows)} claim embeddings.")

    node_rows = conn.execute(
        "SELECT id, name FROM nodes WHERE retired_at IS NULL"
    ).fetchall()
    if node_rows:
        click.echo(f"Embedding {len(node_rows)} nodes...")
        ids = [r[0] for r in node_rows]
        texts = [r[1] for r in node_rows]
        embeddings = embed_batch(texts)
        for node_id, emb in zip(ids, embeddings):
            store_node_embedding(conn, node_id, emb)
        click.echo(f"  Stored {len(node_rows)} node embeddings.")

    conn.commit()
    conn.close()


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
    default=0.99,
    help="Minimum embedding similarity to consider as candidate",
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
@click.pass_context
def corroborate(
    ctx: click.Context,
    threshold: float,
    model: str,
    rerank: bool,
    rerank_min: float,
) -> None:
    """Find cross-record corroborations: embedding similarity then AI verification.

    With --rerank, a cross-encoder pre-filter scores each candidate pair before
    Claude is consulted. Pairs below --rerank-min are dropped, cutting the
    number of Claude calls without losing genuine corroborations.
    """
    from anomalica_common.llm import _call, _parse_json, resolve_use_api
    from assimilator.database import insert_corroboration
    from assimilator.embeddings import deserialise_f32, search_similar_claims

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

    conn.commit()
    actual = conn.execute("SELECT COUNT(*) FROM corroborations").fetchone()[0]
    click.echo(
        f"Verified {actual} genuine corroborations (from {len(candidates)} candidates)"
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


if __name__ == "__main__":
    main()
