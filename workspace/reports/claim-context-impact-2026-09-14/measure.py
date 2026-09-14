#!/usr/bin/env python3
"""Measure the historical claim-context loss without mutating the derived graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml

from assimilator.independence import independence_for_nodes
from assimilator.matching import match_node


def _value(value):
    return None if value is None else str(value)


def _claim_key(content_hash: str, claim: dict) -> tuple:
    return (
        content_hash,
        _value(claim.get("text")),
        _value(claim.get("quote")),
        _value(claim.get("type")),
        _value(claim.get("attestation")),
        _value(claim.get("location")),
        _value(claim.get("date")),
        _value(claim.get("date_end")),
    )


def _digest_corpus(digests_dir: Path):
    totals = {
        name: Counter()
        for name in ("domain_claims", "infrastructure_claims", "all_claims")
    }
    by_hash: dict[str, list[dict]] = {}
    file_hashes = {}
    for path in sorted(digests_dir.glob("*.yaml")):
        raw = path.read_bytes()
        doc = yaml.safe_load(raw) or {}
        content_hash = (doc.get("record") or {}).get("content_hash")
        file_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for section in ("domain_claims", "infrastructure_claims"):
            for claim in doc.get(section) or []:
                chain = claim.get("provenance_chain") or {}
                refs = claim.get("refs") or []
                role_refs = [
                    ref for ref in refs if isinstance(ref, dict) and ref.get("role")
                ]
                for target in (totals[section], totals["all_claims"]):
                    target["claims"] += 1
                    target["references"] += len(refs)
                    target["claims_with_references"] += bool(refs)
                    target["claims_with_provenance_chain"] += bool(
                        claim.get("provenance_chain")
                    )
                    target["claims_with_origin_ref"] += bool(chain.get("origin_ref"))
                    target["claims_with_attribution"] += "attribution_in_text" in claim
                    target["attribution_true"] += (
                        claim.get("attribution_in_text") is True
                    )
                    target["attribution_false"] += (
                        claim.get("attribution_in_text") is False
                    )
                    target["claims_with_reference_roles"] += bool(role_refs)
                    target["references_with_roles"] += len(role_refs)
                    target["anonymous_origin_claims"] += (
                        chain.get("origin_kind") == "anonymous"
                    )
                    target["anonymous_origin_without_ref"] += chain.get(
                        "origin_kind"
                    ) == "anonymous" and not chain.get("origin_ref")
                if content_hash:
                    by_hash.setdefault(content_hash, []).append(claim)
    return totals, by_hash, file_hashes


def _read_graph(db_path: Path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    rows = conn.execute(
        """
        SELECT c.id, r.content_hash, c.content, c.original_excerpt, c.claim_type,
               c.attestation, c.location_in_record, c.date, c.date_end,
               c.record_id, c.origin_ref, c.attribution_in_text
          FROM claims c JOIN records r ON r.id = c.record_id
        """
    ).fetchall()
    graph_counts = {
        "records": conn.execute("SELECT COUNT(*) FROM records").fetchone()[0],
        "claims": len(rows),
        "claim_node_refs": conn.execute(
            "SELECT COUNT(*) FROM claim_node_refs"
        ).fetchone()[0],
        "origin_ref_present": conn.execute(
            "SELECT COUNT(*) FROM claims WHERE origin_ref IS NOT NULL"
        ).fetchone()[0],
        "attribution_present": conn.execute(
            "SELECT COUNT(*) FROM claims WHERE attribution_in_text IS NOT NULL"
        ).fetchone()[0],
        "salience_present": conn.execute(
            "SELECT COUNT(*) FROM claim_node_refs WHERE salience IS NOT NULL"
        ).fetchone()[0],
        "digest_import_receipts": conn.execute(
            "SELECT COUNT(*) FROM digest_import_receipts"
        ).fetchone()[0],
    }
    return conn, rows, graph_counts


def _map_context(conn, graph_rows, digest_by_hash):
    digest_index = {}
    for content_hash, claims in digest_by_hash.items():
        for claim in claims:
            key = _claim_key(content_hash, claim)
            if key in digest_index:
                raise ValueError(f"duplicate digest claim tuple: {key}")
            digest_index[key] = claim

    context = {}
    missing = []
    for row in graph_rows:
        (
            claim_id,
            content_hash,
            *fields,
            record_id,
            graph_origin_ref,
            graph_attribution,
        ) = row
        key = (content_hash, *(_value(value) for value in fields))
        claim = digest_index.get(key)
        if claim is None:
            missing.append(claim_id)
            continue
        chain = claim.get("provenance_chain") or {}
        roles = [
            (ref.get("id"), ref.get("name"), ref.get("role"))
            for ref in claim.get("refs") or []
            if isinstance(ref, dict) and ref.get("id") and ref.get("role")
        ]
        context[claim_id] = {
            "origin_ref": chain.get("origin_ref"),
            "origin_kind": chain.get("origin_kind"),
            "attribution": claim.get("attribution_in_text")
            if "attribution_in_text" in claim
            else None,
            "attribution_declared": "attribution_in_text" in claim,
            "roles": roles,
            "raw_role_count": len(roles),
            "record_id": record_id,
            "graph_origin_ref": graph_origin_ref,
            "graph_attribution": graph_attribution,
        }

    graph_edges = {
        (claim_id, node_id): salience
        for claim_id, node_id, salience in conn.execute(
            "SELECT claim_id, node_id, salience FROM claim_node_refs"
        )
    }
    names: dict[str, set[str]] = {}
    for node_id, name in conn.execute("SELECT id, name FROM nodes"):
        names.setdefault(name.casefold(), set()).add(node_id)
    for alias, node_id in conn.execute("SELECT alias, node_id FROM aliases"):
        names.setdefault(alias.casefold(), set()).add(node_id)
    for claim_id, item in context.items():
        resolved = {}
        for source_id, name, role in item["roles"]:
            candidates = [source_id]
            if name:
                candidates.extend(sorted(names.get(name.casefold(), set())))
                matched = match_node(conn, name, record_id=item["record_id"])
                if matched:
                    candidates.append(matched[0])
            node_id = next(
                (
                    candidate
                    for candidate in candidates
                    if (claim_id, candidate) in graph_edges
                ),
                source_id,
            )
            resolved[node_id] = role
        item["roles"] = resolved
    return context, graph_edges, missing


def _brief_impact(briefs_dir: Path, context: dict):
    counts = Counter()
    affected = {
        "origin_ref": set(),
        "attribution": set(),
        "salience": set(),
        "any": set(),
    }
    for path in sorted(briefs_dir.glob("*/*.yaml")):
        doc = yaml.safe_load(path.read_text()) or {}
        ref = str(path.relative_to(briefs_dir).with_suffix(""))
        counts["briefs"] += 1
        counts["with_payload_hash"] += bool(doc.get("payload_hash"))
        for claim in doc.get("claims") or []:
            expected = context.get(claim.get("claim_id"))
            if not expected:
                continue
            if expected["origin_ref"]:
                counts["origin_ref_occurrences"] += 1
                if (claim.get("provenance_chain") or {}).get("origin_ref") != expected[
                    "origin_ref"
                ]:
                    affected["origin_ref"].add(ref)
                    affected["any"].add(ref)
            if expected["attribution_declared"]:
                counts["attribution_declared_occurrences"] += 1
            if expected["attribution"] is True:
                counts["attribution_true_occurrences"] += 1
                if claim.get("attribution_mode") != "in_text":
                    affected["attribution"].add(ref)
                    affected["any"].add(ref)
            expected_roles = expected["roles"]
            for node_ref in claim.get("node_refs") or []:
                role = expected_roles.get(node_ref.get("node_id"))
                if role:
                    counts["role_occurrences"] += 1
                    if node_ref.get("role") != role:
                        affected["salience"].add(ref)
                        affected["any"].add(ref)
    return dict(counts), {key: len(value) for key, value in affected.items()}


def _frontmatter(path: Path) -> dict:
    parts = path.read_text(errors="replace").split("---", 2)
    if len(parts) < 3:
        return {}
    value = yaml.safe_load(parts[1]) or {}
    return value if isinstance(value, dict) else {}


def _article_impact(content_dir: Path, briefs_dir: Path, context: dict):
    counts = Counter()
    affected = {
        "origin_ref": set(),
        "attribution": set(),
        "salience": set(),
        "any": set(),
    }
    for path in sorted(content_dir.glob("*/*.md")):
        counts["articles"] += 1
        fm = _frontmatter(path)
        built = fm.get("built_from")
        if not isinstance(built, dict):
            counts["without_built_from"] += 1
            continue
        counts["with_built_from"] += 1
        counts["without_payload_hash"] += not bool(built.get("payload_hash"))
        stem_parts = path.stem.rsplit(".", 1)
        slug = stem_parts[0] if len(stem_parts) == 2 else path.stem
        brief_path = briefs_dir / path.parent.name / f"{slug}.yaml"
        if brief_path.is_file():
            counts["resolving_to_published_brief"] += 1
            if not built.get("payload_hash"):
                counts["stale_resolved_payload_binding"] += 1
        else:
            counts["without_published_brief"] += 1
        article_ref = str(path.relative_to(content_dir))
        for citation in built.get("claims") or []:
            claim_id = citation.get("id") if isinstance(citation, dict) else None
            expected = context.get(claim_id)
            if not expected:
                continue
            if expected["origin_ref"]:
                affected["origin_ref"].add(article_ref)
                affected["any"].add(article_ref)
            if expected["attribution"] is True:
                affected["attribution"].add(article_ref)
                affected["any"].add(article_ref)
            if expected["roles"]:
                affected["salience"].add(article_ref)
                affected["any"].add(article_ref)
    return dict(counts), {key: len(value) for key, value in affected.items()}


def _independence_impact(read_conn, context):
    before = independence_for_nodes(read_conn)
    memory = sqlite3.connect(":memory:")
    read_conn.backup(memory)
    memory.executemany(
        "UPDATE claims SET origin_ref = ? WHERE id = ?",
        [
            (item["origin_ref"], claim_id)
            for claim_id, item in context.items()
            if item["origin_ref"]
        ],
    )
    after = independence_for_nodes(memory)
    changes = {
        node_id: after[node_id].sources - score.sources
        for node_id, score in before.items()
        if score.sources is not None
        and node_id in after
        and after[node_id].sources != score.sources
    }
    proposals = {
        row[0]
        for row in read_conn.execute(
            "SELECT node_id FROM page_proposals WHERE status = 'proposed'"
        )
    }
    names = dict(read_conn.execute("SELECT id, name FROM nodes"))
    largest = sorted(changes.items(), key=lambda row: (-row[1], names.get(row[0], "")))[
        :10
    ]
    return {
        "nodes_changed": len(changes),
        "proposed_nodes_changed": len(set(changes) & proposals),
        "total_source_undercount": sum(changes.values()),
        "maximum_node_undercount": max(changes.values(), default=0),
        "largest_changes": [
            {"node_id": node_id, "name": names.get(node_id), "delta": delta}
            for node_id, delta in largest
        ],
    }


def measure(args) -> dict:
    totals, digest_by_hash, digest_hashes = _digest_corpus(args.digests)
    conn, graph_rows, graph_counts = _read_graph(args.db)
    context, graph_edges, unmatched = _map_context(conn, graph_rows, digest_by_hash)
    graph_bound_hashes = {
        row[0] for row in conn.execute("SELECT content_hash FROM records")
    }
    bound_context = list(context.values())
    expected_role_edges = sum(
        (claim_id, node_id) in graph_edges
        for claim_id, item in context.items()
        for node_id in item["roles"]
    )
    role_refs_without_edge = (
        sum(len(item["roles"]) for item in bound_context) - expected_role_edges
    )
    brief_counts, brief_affected = _brief_impact(args.briefs, context)
    article_counts, article_affected = _article_impact(
        args.content, args.briefs, context
    )
    return {
        "schema": "anomalica/claim-context-impact/1",
        "report_only": True,
        "canonical_activation": "forbidden",
        "snapshot": {
            "database": str(args.db),
            "database_sha256": hashlib.sha256(args.db.read_bytes()).hexdigest(),
            "canonical_digest_count": len(digest_hashes),
            "graph_bound_digest_count": len(graph_bound_hashes),
            "published_briefs": str(args.briefs),
            "articles": str(args.content),
        },
        "measured": {
            "digest_corpus": {key: dict(value) for key, value in totals.items()},
            "graph": graph_counts,
            "graph_mapping": {
                "matched_claims": len(context),
                "unmatched_graph_claims": len(unmatched),
                "origin_ref_available": sum(
                    bool(item["origin_ref"]) for item in bound_context
                ),
                "origin_ref_omitted": sum(
                    bool(item["origin_ref"]) and not item["graph_origin_ref"]
                    for item in bound_context
                ),
                "attribution_available": sum(
                    item["attribution_declared"] for item in bound_context
                ),
                "attribution_true_available": sum(
                    item["attribution"] is True for item in bound_context
                ),
                "attribution_false_available": sum(
                    item["attribution"] is False for item in bound_context
                ),
                "attribution_omitted": sum(
                    item["attribution_declared"] and item["graph_attribution"] is None
                    for item in bound_context
                ),
                "raw_reference_roles_available": sum(
                    item["raw_role_count"] for item in bound_context
                ),
                "distinct_reference_roles_available": sum(
                    len(item["roles"]) for item in bound_context
                ),
                "reference_roles_collapsed_on_resolution": sum(
                    item["raw_role_count"] - len(item["roles"])
                    for item in bound_context
                ),
                "role_edges_available": expected_role_edges,
                "salience_omitted": sum(
                    graph_edges[(claim_id, node_id)] is None
                    for claim_id, item in context.items()
                    for node_id in item["roles"]
                    if (claim_id, node_id) in graph_edges
                ),
                "role_references_without_graph_edge": role_refs_without_edge,
            },
            "independence": _independence_impact(conn, context),
            "published_briefs": {**brief_counts, "affected": brief_affected},
            "articles": {**article_counts, "affected": article_affected},
        },
        "unknowns": [
            "Anonymous-origin claims without origin_ref cannot be split into source identities from current data.",
            "Digest references without a graph edge have no row on which salience could have been stored.",
            "Null roles on references without a digest declaration are unassessed, not non-salient.",
            "The exact article prose changes are unknown without a metered reassembly run.",
            "The graph has no digest import receipts, so exact digest-byte/import-generation freshness is unknown.",
            "New payload hashes bind graph content; they cannot prove digest context was imported before the graph is rebuilt or reconciled.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digests", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--briefs", type=Path, required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
