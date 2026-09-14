import json
from pathlib import Path


REPORT = Path(__file__).with_name("impact.json")


def test_census_is_exact_where_it_claims_to_be_exact():
    report = json.loads(REPORT.read_text())
    measured = report["measured"]
    mapping = measured["graph_mapping"]

    assert report["report_only"] is True
    assert mapping["matched_claims"] == measured["graph"]["claims"] == 40_400
    assert mapping["unmatched_graph_claims"] == 0
    assert mapping["origin_ref_omitted"] == mapping["origin_ref_available"]
    assert mapping["attribution_omitted"] == mapping["attribution_available"]
    assert mapping["salience_omitted"] == mapping["role_edges_available"]
    assert mapping["distinct_reference_roles_available"] == (
        mapping["role_edges_available"] + mapping["role_references_without_graph_edge"]
    )
    assert mapping["raw_reference_roles_available"] == (
        mapping["distinct_reference_roles_available"]
        + mapping["reference_roles_collapsed_on_resolution"]
    )


def test_payload_freshness_counts_separate_stale_from_unauditable():
    measured = json.loads(REPORT.read_text())["measured"]
    briefs = measured["published_briefs"]
    articles = measured["articles"]

    assert briefs["with_payload_hash"] == 0
    assert articles["with_built_from"] == articles["without_payload_hash"]
    assert articles["articles"] == (
        articles["with_built_from"] + articles["without_built_from"]
    )
    assert (
        articles["stale_resolved_payload_binding"]
        == (articles["resolving_to_published_brief"])
    )
