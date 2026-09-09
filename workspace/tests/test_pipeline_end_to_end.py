"""Digest YAML to graph rows: the assimilation stage, on real digester output.

WHAT THIS COVERS, stated before anything else so nobody reads more into a green
run than is there. It starts at an ingest record file and stops at the rows
``import_extraction`` writes. The model is replaced by canned responses written
by hand, so this proves PLUMBING, not extraction quality - a prompt change that
destroys recall passes it green. Everything downstream of the import is out of
scope: embeddings, corroboration, merges, page proposals, the assembler and the
site. So is the ingester.

Its two fixture records come from the digestion-stage tests
(``digester/workspace/tests/pipeline/fixtures``) and are digested here by the
real ``digester.cli._do_extract`` rather than re-declared, so the digests under
test are the ones the pipeline actually emits - location alignment, the review
stamp, the copyright flattening and all.

THREE KNOWN LOSSES ARE ENCODED AS STRICT XFAILS below, each asserting the
behaviour we want rather than the behaviour we have. Each names the line that
has to change. When one is fixed its test XPASSes, and ``strict=True`` turns
that into a failure so the marker cannot outlive the defect.

This module needs ``digester`` and ``assimilator`` importable in ONE process,
which the assimilator's container cannot do - `just test` therefore skips it,
and `just e2e` is the entry point that runs it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locating the digestion stage.
#
# The two components live in sibling repos and neither container can import the
# other, so the path is resolved here and the module skips - loudly, naming the
# command that does work - rather than failing a containerised `just test`.
# ---------------------------------------------------------------------------

_ASSIMILATOR_WORKSPACE = Path(__file__).resolve().parent.parent
_ANOMALICA_ROOT = _ASSIMILATOR_WORKSPACE.parent.parent
_DIGESTER_WORKSPACE = Path(
    os.environ.get(
        "ANOMALICA_DIGESTER_WORKSPACE", _ANOMALICA_ROOT / "digester" / "workspace"
    )
)
_PIPELINE_FIXTURES = _DIGESTER_WORKSPACE / "tests" / "pipeline" / "fixtures"

_HOW_TO_RUN = (
    "the digestion stage is not importable from here. This module needs "
    f"{_DIGESTER_WORKSPACE} on the path and its pipeline fixtures at "
    f"{_PIPELINE_FIXTURES}, which the assimilator container does not mount. "
    "Run it on the host: `just e2e` in ~/repos/anomalica/assimilator."
)

if not _PIPELINE_FIXTURES.is_dir():
    pytest.skip(_HOW_TO_RUN, allow_module_level=True)

if str(_DIGESTER_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_DIGESTER_WORKSPACE))


def _load_fixture_module(name: str):
    """Load one fixture module by path, under a package of our own.

    Not imported as ``tests.pipeline.fixtures.<name>``: this repo has its own
    top-level ``tests`` package, so that name resolves here instead of in the
    digester. A synthetic parent package is registered so the fixtures' own
    relative imports (``from .documents import ...``) still resolve.
    """
    parent = "anomalica_pipeline_fixtures"
    if parent not in sys.modules:
        pkg = types.ModuleType(parent)
        pkg.__path__ = [str(_PIPELINE_FIXTURES)]
        sys.modules[parent] = pkg
    full = f"{parent}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, _PIPELINE_FIXTURES / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


# The spend guard is loaded here, inside the skip guard, and never
# re-implemented: if it cannot be loaded this module must SKIP, not run a real
# extraction unguarded.
try:
    fixture_documents = _load_fixture_module("documents")
    fixture_responses = _load_fixture_module("responses")
    harness = _load_fixture_module("harness")
except Exception as exc:  # noqa: BLE001 - the reason must reach the report
    pytest.skip(f"{_HOW_TO_RUN} ({type(exc).__name__}: {exc})", allow_module_level=True)

GUARDED_TRANSPORT_CALLS = harness.GUARDED_TRANSPORT_CALLS
ProviderCallAttempted = harness.ProviderCallAttempted
CountingUuid = harness.CountingUuid
FrozenDatetime = harness.FrozenDatetime
FROZEN_INSTANT = harness.FROZEN_INSTANT


# ---------------------------------------------------------------------------
# Import-time environment.
#
# Set here rather than through monkeypatch because the transport flushes its
# ledger row at interpreter exit, long after any fixture has torn down.
# `ledger.enabled()` guards on PYTEST_CURRENT_TEST, and that variable is NOT set
# during the atexit flush the guard was written for - so pointing
# SCHEDULER_DISPATCH_LOG at a temporary file is what actually keeps the
# production spend ledger clean.
#
# ANOMALICA_INGESTS_DIR is deliberately NOT set: `_INGESTS_DIR` is read at
# import time, so the env var would either be too late or would redirect every
# other test in this suite. It is patched as a module attribute instead.
# ---------------------------------------------------------------------------

_SANDBOX = Path(tempfile.mkdtemp(prefix="assimilator-e2e-"))

os.environ["SCHEDULER_DISPATCH_LOG"] = str(_SANDBOX / "model-dispatch.jsonl")
os.environ["ANOMALICA_LEDGER"] = "off"
os.environ["DIGESTER_CALL_CACHE"] = "off"
os.environ["DIGESTER_ENTAILMENT"] = "off"
os.environ["ASSIMILATOR_DB"] = str(_SANDBOX / "assimilator" / "knowledge.db")
os.environ["DIGESTER_USE_API"] = "0"
os.environ["ANOMALICA_USE_API"] = "0"

for _key in [k for k in os.environ if k.endswith("_API_KEY")]:
    del os.environ[_key]


# ---------------------------------------------------------------------------
# The spend guard. Layered, every layer independent of the others, and defined
# once in the digester's `fixtures/harness.py` - read that file for how it works
# and why two copies of it would be worse than one shared definition.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def no_provider_calls():
    """Make a real model call impossible, by every route the code has.

    Module-scoped and installed once. A per-test guard layered under a
    module-scoped fixture that patches the same names tears down in the wrong
    order and leaves the shim installed for the rest of the session.
    """
    with pytest.MonkeyPatch.context() as mp:
        harness.install_spend_guard(mp)
        yield mp


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------

# The band the importer mints ids in when it needs one of its own. Kept clear of
# the digest's band so an importer-minted id is visible on sight rather than
# indistinguishable from an id the digest supplied.
IMPORTER_ID_BAND = 9


def _pin_determinism(mp: pytest.MonkeyPatch) -> CountingUuid:
    """Pin every source of non-determinism between a record and its graph rows.

    Four sites: the digest's ids and `extracted_at`, the ids the importer mints
    for itself, and the `created_at` every row is stamped with.
    """
    from anomalica_common.digest import yaml_format
    from assimilator import database, import_markdown

    digest_ids = CountingUuid()
    mp.setattr(yaml_format, "uuid", digest_ids)
    mp.setattr(yaml_format, "datetime", FrozenDatetime)
    mp.setattr(import_markdown, "uuid", CountingUuid(band=IMPORTER_ID_BAND))
    mp.setattr(database, "_now", lambda: FROZEN_INSTANT.isoformat())
    return digest_ids


# ---------------------------------------------------------------------------
# Digestion and assimilation.
# ---------------------------------------------------------------------------


def _digest(document, out_path: Path, model: str = "test-canned") -> Path:
    """Run the real extraction command over one fixture record.

    `_do_extract` is the live path: it materialises the pre-digest, runs both
    passes, canonicalises each claim's location against its quote, stamps the
    review provenance and the copyright status, and writes the digest. Only the
    model call underneath it is canned.
    """
    from digester import cli

    return cli._do_extract(document.store_path, document.parsed(), out_path, model)


def _open_graph(root: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """The pair of databases the importer writes, as `import_markdown.main` opens them."""
    from assimilator.database import init_db

    root.mkdir(parents=True, exist_ok=True)
    domain = sqlite3.connect(str(root / "knowledge.db"))
    infrastructure = sqlite3.connect(str(root / "infrastructure.db"))
    for conn in (domain, infrastructure):
        init_db(conn)
    return domain, infrastructure


def _assimilate(domain, infrastructure, digest_path: Path) -> dict:
    """Fold one digest into the graph, both sections, as the CLI does."""
    from anomalica_common.digest.yaml_format import parse_digest_yaml
    from assimilator.import_markdown import import_extraction

    parsed = parse_digest_yaml(digest_path.read_text())
    counts = {}
    if parsed["domain_claims"]:
        counts["domain"] = import_extraction(
            domain,
            parsed,
            section="domain",
            lookup_conns=[infrastructure],
            source_path=str(digest_path),
        )
    if parsed["infrastructure_claims"]:
        counts["infrastructure"] = import_extraction(
            infrastructure,
            parsed,
            section="infrastructure",
            lookup_conns=[domain],
            source_path=str(digest_path),
        )
    return counts


class Pipeline:
    """One run of both fixture records from store file to graph rows."""

    def __init__(self, root: Path, mp: pytest.MonkeyPatch) -> None:
        from assimilator import import_markdown

        self.root = root
        self._mp = mp
        self.digest_ids = _pin_determinism(mp)

        # The seam. Patched on `digester.extract` because `_do_extract` imports
        # `extract_two_pass` locally inside the function body, so a patch on
        # `digester.cli` never takes.
        from digester import extract

        self.model_calls: list[dict] = []

        def _serve(preamble, document, task, model, schema=None, use_api=False):
            self.model_calls.append(
                {
                    "document": fixture_responses.document_of(document),
                    "pass": fixture_responses.pass_of(schema),
                    "model": model,
                    "use_api": use_api,
                }
            )
            return fixture_responses.response_for(
                preamble, document, task, model, schema=schema, use_api=use_api
            )

        mp.setattr(extract, "call_with_document", _serve)

        # Otherwise the command makes three urlopen attempts to 127.0.0.1:8001
        # with 4.5 seconds of sleeps and then fails closed.
        from anomalica_common.llm.allowance import Allowance
        from digester import cli

        mp.setattr(
            cli,
            "check_allowance",
            lambda **_: Allowance(ok=True, reason="pinned by the pipeline tests"),
        )

        # Read at IMPORT time from the environment, so setenv after the module is
        # imported does nothing. Patched as an attribute, which is the only thing
        # that takes.
        mp.setattr(import_markdown, "_INGESTS_DIR", str(fixture_documents.INGESTS_DIR))

        self.digests_dir = root / "digests"
        self.digests_dir.mkdir(parents=True, exist_ok=True)
        self.digests = {
            key: _digest(document, self.digests_dir / f"{key}.yaml")
            for key, document in fixture_documents.DOCUMENTS.items()
        }
        # Snapshotted here: `model_calls` keeps growing as later tests re-digest,
        # and the question "how many calls does digesting one record take" is
        # about this first pass over each record, not the running total.
        self.first_pass_calls = list(self.model_calls)
        self.domain, self.infrastructure = _open_graph(root / "graph")
        self.counts = {
            key: _assimilate(self.domain, self.infrastructure, path)
            for key, path in sorted(self.digests.items())
        }

    def redigest(self, key: str, suffix: str) -> Path:
        """Emit the record again, with fresh ids, as a re-digest does."""
        document = fixture_documents.DOCUMENTS[key]
        return _digest(document, self.digests_dir / f"{key}{suffix}.yaml")

    def digest_doc(self, key: str) -> dict:
        import yaml

        return yaml.safe_load(self.digests[key].read_text())

    def node_id(self, name: str) -> str:
        row = self.domain.execute(
            "SELECT id FROM nodes WHERE name = ?", (name,)
        ).fetchone()
        assert row is not None, (
            f"no node named {name!r}; the graph holds "
            f"{sorted(r[0] for r in self.domain.execute('SELECT name FROM nodes'))}"
        )
        return row[0]

    def record_id(self, key: str) -> str:
        title = fixture_documents.DOCUMENTS[key].parsed().title
        row = self.domain.execute(
            "SELECT id FROM records WHERE title = ?", (title,)
        ).fetchone()
        assert row is not None, f"record {title!r} did not reach the graph"
        return row[0]

    def close(self) -> None:
        self.domain.close()
        self.infrastructure.close()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> Pipeline:
    """Both records, digested and assimilated once for the whole module.

    Module-scoped because the run is the subject: every assertion here reads the
    same graph, and rebuilding it per test would say nothing extra while making
    the file slow enough that people stop running it.
    """
    with pytest.MonkeyPatch.context() as mp:
        built = Pipeline(tmp_path_factory.mktemp("pipeline"), mp)
        yield built
        built.close()


# ---------------------------------------------------------------------------
# The two records reach the graph.
# ---------------------------------------------------------------------------


def test_both_records_reach_the_graph_addressed_by_their_content_hash(pipeline):
    rows = dict(
        pipeline.domain.execute("SELECT title, content_hash FROM records").fetchall()
    )
    assert len(rows) == 2
    for document in fixture_documents.DOCUMENTS.values():
        title = document.parsed().title
        assert rows[title] == f"sha256:{document.content_hash}"


def test_each_claim_the_digest_carries_becomes_exactly_one_claim_row(pipeline):
    for key, path in pipeline.digests.items():
        import yaml

        doc = yaml.safe_load(path.read_text())
        record = pipeline.record_id(key)
        in_domain = pipeline.domain.execute(
            "SELECT COUNT(*) FROM claims WHERE record_id = ?", (record,)
        ).fetchone()[0]
        in_infrastructure = pipeline.infrastructure.execute(
            "SELECT COUNT(*) FROM claims WHERE record_id = ?", (record,)
        ).fetchone()[0]
        assert in_domain == len(doc.get("domain_claims") or []), key
        assert in_infrastructure == len(doc.get("infrastructure_claims") or []), key


def test_the_two_claim_lists_land_in_two_databases_not_one(pipeline):
    """A digest's infrastructure claims must not appear beside its domain ones.

    They are separate databases precisely so the public site can read one of
    them; a claim in the wrong file is a citation rendered as content.
    """
    domain_texts = {r[0] for r in pipeline.domain.execute("SELECT content FROM claims")}
    infrastructure_texts = {
        r[0] for r in pipeline.infrastructure.execute("SELECT content FROM claims")
    }
    assert domain_texts and infrastructure_texts
    assert not (domain_texts & infrastructure_texts)


def test_every_node_a_digest_declares_is_linked_to_its_record(pipeline):
    """record_nodes says which nodes a record's digest DECLARED, as distinct from
    which its claims reference. The two diverge, and the divergence is the point,
    so nothing may go missing from the declaration side.

    Compared through the alias graph rather than by name: a spelling that
    resolved onto a node the graph already held is linked under the name the
    graph settled on, not the one this digest wrote.
    """
    for key in fixture_documents.DOCUMENTS:
        declared = {n["name"] for n in pipeline.digest_doc(key)["nodes"]}
        linked = {
            r[0]
            for r in pipeline.domain.execute(
                "SELECT node_id FROM record_nodes WHERE record_id = ?",
                (pipeline.record_id(key),),
            )
        }
        assert len(linked) == len(declared), key
        for name in declared:
            resolved = pipeline.domain.execute(
                "SELECT id FROM nodes WHERE name = ? "
                "UNION SELECT node_id FROM aliases WHERE alias = ?",
                (name, name),
            ).fetchall()
            assert {r[0] for r in resolved} & linked, (key, name)


# ---------------------------------------------------------------------------
# Cross-record entity identity.
# ---------------------------------------------------------------------------


def test_the_person_named_in_both_records_is_one_node_carrying_claims_from_both(
    pipeline,
):
    """`Dr Helena Marsh` is declared by both digests and must not become two nodes."""
    rows = pipeline.domain.execute(
        "SELECT id FROM nodes WHERE name = ? AND node_type = 'person'",
        ("Dr Helena Marsh",),
    ).fetchall()
    assert len(rows) == 1, "the shared person split into more than one node"
    node_id = rows[0][0]

    records = {
        r[0]
        for r in pipeline.domain.execute(
            "SELECT DISTINCT c.record_id FROM claim_node_refs x "
            "JOIN claims c ON c.id = x.claim_id WHERE x.node_id = ?",
            (node_id,),
        )
    }
    assert records == {pipeline.record_id("a"), pipeline.record_id("b")}


def test_the_organisation_written_two_ways_resolves_to_one_node_and_keeps_the_other_name(
    pipeline,
):
    """Document A writes `Coastal Air Defence Command`, B writes it with its acronym.

    They are one organisation. The matcher's acronym tier collapses them onto a
    single node and the losing spelling is kept as an alias, so a later record
    using it still resolves. Which spelling wins the node's `name` depends on
    which record was imported first, so the assertion is on the identity - one
    node, both spellings reaching it - not on which string won.
    """
    spellings = {"Coastal Air Defence Command", "Coastal Air Defence Command (CADC)"}
    ids = {
        r[0]
        for r in pipeline.domain.execute(
            "SELECT id FROM nodes WHERE name IN (?, ?)", tuple(sorted(spellings))
        )
    }
    assert len(ids) == 1, "the acronym spelling minted a second organisation"
    node_id = ids.pop()

    reachable = {
        pipeline.domain.execute(
            "SELECT name FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
    }
    reachable |= {
        r[0]
        for r in pipeline.domain.execute(
            "SELECT alias FROM aliases WHERE node_id = ?", (node_id,)
        )
    }
    assert spellings <= reachable


# ---------------------------------------------------------------------------
# Provenance survives the import (ADR 0044).
# ---------------------------------------------------------------------------


def _chain_rows(pipeline) -> dict[str, tuple]:
    """content -> (origin_kind, origin, relay, attestation), for every domain claim."""
    return {
        content: (origin_kind, origin, json.loads(relay) if relay else [], attestation)
        for content, origin_kind, origin, relay, attestation in pipeline.domain.execute(
            "SELECT content, origin_kind, origin, relay, attestation FROM claims"
        )
    }


def _claim_starting(pipeline, prefix: str) -> tuple:
    row = pipeline.domain.execute(
        "SELECT id, content, claim_type, attestation, origin_kind, origin, relay, "
        "speaker_id FROM claims WHERE content LIKE ?",
        (prefix + "%",),
    ).fetchall()
    assert len(row) == 1, f"expected one claim starting {prefix!r}, found {len(row)}"
    return row[0]


def test_the_whole_provenance_ladder_reaches_the_claim_rows(pipeline):
    """Every rung the digest carries is on the row, relay included.

    The chain is what independence keys on, and it was dropped at this boundary
    once already: 19,006 claims read `origin_kind` NULL while their digests
    carried a chain on 87% of them.
    """
    rows = _chain_rows(pipeline)
    ladder = {
        (kind, tuple(relay), attestation)
        for kind, _origin, relay, attestation in rows.values()
    }
    assert ("speaker", (), "first_hand") in ladder
    assert ("named", (), "second_hand") in ladder
    assert any(
        kind == "anonymous" and len(relay) == 1 and attestation == "second_hand"
        for kind, relay, attestation in ladder
    )
    assert any(
        kind == "document" and len(relay) == 2 and attestation == "third_hand"
        for kind, relay, attestation in ladder
    )
    assert ("unattributed", (), None) in ladder


def test_a_relayed_chain_keeps_the_names_it_passed_through(pipeline):
    claim = _claim_starting(pipeline, "The boxed Skerrivore Point tapes were never")
    assert claim[4] == "document"
    assert claim[5] == "Skerrivore Point incident file"
    assert json.loads(claim[6]) == [
        "a ministry archivist",
        "Group Captain Aled Furze",
    ]


def test_a_described_origin_is_stored_as_anonymous_whatever_the_digest_called_it(
    pipeline,
):
    """The digest declares this one `named` with a bracketed description as origin.

    A description has no node, so `named` would make independence resolve it to
    nothing and count each such claim as its own root - inflating independence,
    which is the unsafe direction. The importer rewrites it.
    """
    claim = _claim_starting(pipeline, "The boxed Skerrivore Point tapes were wiped")
    declared = [
        c
        for c in pipeline.digest_doc("a")["domain_claims"]
        if c["text"].startswith("The boxed Skerrivore Point tapes were wiped")
    ]
    assert declared[0]["provenance_chain"]["origin_kind"] == "named"
    assert claim[4] == "anonymous"
    assert claim[7] is None, "a described speaker must not resolve to a node"


# ---------------------------------------------------------------------------
# THE KNOWN LOSSES.
#
# Each of these asserts the behaviour we want, is marked strict-xfail because we
# do not have it, and names the change that would make it pass. A strict xfail
# fails the run if it starts passing, so the marker cannot outlive the defect.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "claim_node_refs.salience is never written. Everything upstream of the "
        "insert has the value: the extractor emits a role per ref, "
        "parse_digest_yaml turns them into `ref_roles`, and the graph Claim "
        "model declares a `ref_roles` field. import_extraction resolves refs to "
        "ids without them and database.insert_claim writes (claim_id, node_id) "
        "only - assimilator/database.py:696-700. Fix: carry ref_roles through "
        "import_extraction and write the third column."
    ),
)
def test_the_role_a_node_plays_in_a_claim_reaches_the_edge_that_records_it(pipeline):
    """A ref's role is the difference between a claim's subject and a name in it.

    The extractor emits four values - subject, participant, setting, mentioned -
    the column exists with a CHECK constraint that names them, and nothing in
    between writes one. Document A's first claim carries all four.
    """
    values = {
        r[0]
        for r in pipeline.domain.execute(
            "SELECT DISTINCT salience FROM claim_node_refs"
        )
    }
    assert values == {"subject", "participant", "setting", "mentioned"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "origin_ref has no column on claims. It is built into the ProvenanceChain "
        "at assimilator/import_markdown.py:1075 and then discarded by "
        "database.insert_claim, which writes origin_kind, origin and relay only. "
        "Fix: add the column, carry it in insert_claim and update_claim_chain, and "
        "key the anonymous branch of database.provenance_root and "
        "independence._root on (record_id, origin_ref) rather than collapsing "
        "every anonymous origin to one root."
    ),
)
def test_two_distinct_anonymous_sources_in_one_record_count_as_two_sources(pipeline):
    """The loss shows up as a WRONG NUMBER, not as a missing column.

    Document A names three distinct origins for the claims about the boxed
    tapes: a duty officer at Whitchurch Down (`duty-officer-1`), a colleague at
    the observatory (`colleague-1`), and the incident file. `origin_ref` exists
    precisely so two anonymous sources inside one record stop collapsing into
    one - and with the column gone, independence reports two sources where the
    evidence names three.
    """
    from assimilator.independence import independence_for_nodes

    node_id = pipeline.node_id("the boxed Skerrivore Point tapes")

    about_the_tapes = [
        c
        for c in pipeline.digest_doc("a")["domain_claims"]
        if "the boxed Skerrivore Point tapes"
        in {r["name"] for r in c.get("refs") or []}
    ]
    origin_refs = {
        c["provenance_chain"].get("origin_ref")
        for c in about_the_tapes
        if c["provenance_chain"].get("origin_ref")
    }
    origins = {c["provenance_chain"]["origin"] for c in about_the_tapes}
    # Guard the premise: this test says nothing unless the digest really does
    # distinguish two anonymous sources by origin_ref and name a third origin.
    assert len(about_the_tapes) == 3
    assert len(origin_refs) == 2, "the fixture no longer carries two origin_refs"
    assert len(origins) == 3

    scored = independence_for_nodes(pipeline.domain, [node_id])[node_id]
    assert scored.scored_claims == 3
    assert scored.sources == 3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "attribution_in_text reaches no consumer. The parser passes it through "
        "(anomalica_common/digest/yaml_format.py, _CLAIM_RENAMED carries unknown "
        "keys), the claims table has no column for it, and insert_claim never "
        "looks. Fix: add the column, carry it on the Claim model into "
        "insert_claim, and read it where a renderer asks attribution_mode how a "
        "claim may be shown."
    ),
)
def test_the_graph_can_say_how_a_claim_may_be_rendered(pipeline):
    """The consequence, not the absence: every claim that rests on its source
    reads `unknown` from the graph, whatever the digest declared.

    `attribution_mode` is ADR 0044's single answer to "may this text be shown as
    written". It fails closed without the flag, so a claim the extraction model
    said already names its source is indistinguishable from one that hides it -
    and the safe reading suppresses both.
    """
    from anomalica_common.digest.models import attribution_mode

    claim = _claim_starting(pipeline, "The Skerrivore Point radar tapes for the night")
    declared = [
        c
        for c in pipeline.digest_doc("a")["domain_claims"]
        if c["text"].startswith("The Skerrivore Point radar tapes for the night")
    ]
    assert declared[0]["attribution_in_text"] is True

    columns = {c[1] for c in pipeline.domain.execute("PRAGMA table_info(claims)")}
    from_graph = (
        pipeline.domain.execute(
            "SELECT attribution_in_text FROM claims WHERE id = ?", (claim[0],)
        ).fetchone()[0]
        if "attribution_in_text" in columns
        else None
    )
    mode = attribution_mode(
        claim_type=claim[2],
        attestation=claim[3],
        origin_kind=claim[4],
        attribution_in_text=from_graph,
        has_chain=True,
    )
    assert mode.value == "in_text"


# ---------------------------------------------------------------------------
# Re-import.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reimported(pipeline) -> dict:
    """Both records digested a second time and folded in again.

    The digest id counter is deliberately REBASED between the two emissions, so
    the second digest carries a different record id and different claim ids -
    which is what production does on every re-digest. Emitting the same record
    id twice would hide the content-hash fallback behind a primary-key hit and
    test nothing.
    """
    import yaml

    before = _graph_shape(pipeline)
    pipeline.digest_ids.rebase(50_000)
    paths = {
        key: pipeline.redigest(key, "-again") for key in fixture_documents.DOCUMENTS
    }
    counts = {
        key: _assimilate(pipeline.domain, pipeline.infrastructure, path)
        for key, path in sorted(paths.items())
    }
    return {
        "before": before,
        "after": _graph_shape(pipeline),
        "counts": counts,
        "digests": {k: yaml.safe_load(p.read_text()) for k, p in paths.items()},
    }


def test_assimilating_both_digests_again_creates_no_duplicate_nodes_or_claims(
    pipeline, reimported
):
    assert reimported["after"] == reimported["before"]


def test_a_second_import_carries_every_claim_forward_rather_than_reinserting_it(
    reimported,
):
    """Not merely "the counts did not change": a claim that was deleted and
    re-inserted would leave the same totals behind while losing its identity -
    its uuid, its created_at, and with them every page-staleness reading built
    on them."""
    for key, sections in reimported["counts"].items():
        for section, counts in sections.items():
            assert counts["nodes_created"] == 0, (key, section)
            assert counts["claims_created"] == 0, (key, section)
            assert counts["claims_deleted"] == 0, (key, section)
            assert counts["claims_carried"] > 0, (key, section)


def test_a_re_emitted_digest_finds_its_record_by_content_hash_not_by_id(
    pipeline, reimported
):
    """The second emission's record id is in no row; the record still resolves.

    This is the fallback at import_markdown.py:789, and it is the path every
    production re-digest takes.
    """
    fresh_id = reimported["digests"]["a"]["record"]["id"]

    assert fresh_id != pipeline.record_id("a")
    assert (
        pipeline.domain.execute(
            "SELECT COUNT(*) FROM records WHERE id = ?", (fresh_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        pipeline.domain.execute(
            "SELECT COUNT(*) FROM records WHERE content_hash = ?",
            (f"sha256:{fixture_documents.DOCUMENT_A.content_hash}",),
        ).fetchone()[0]
        == 1
    )


def _graph_shape(pipeline) -> dict:
    """What the graph holds, at the resolution a duplicate would change."""
    return {
        "nodes": sorted(
            pipeline.domain.execute("SELECT name, node_type FROM nodes").fetchall()
        ),
        "records": sorted(
            pipeline.domain.execute("SELECT id, content_hash FROM records").fetchall()
        ),
        "domain_claims": sorted(
            pipeline.domain.execute("SELECT content FROM claims").fetchall()
        ),
        "infrastructure_claims": sorted(
            pipeline.infrastructure.execute("SELECT content FROM claims").fetchall()
        ),
        "edges": pipeline.domain.execute(
            "SELECT COUNT(*) FROM claim_node_refs"
        ).fetchone()[0],
    }


# ---------------------------------------------------------------------------
# Order independence.
# ---------------------------------------------------------------------------


def test_import_order_does_not_change_what_the_graph_holds(tmp_path):
    """B-then-A must reach the same entities and the same claims as A-then-B.

    Not the same STRINGS: which spelling of the shared organisation wins the
    node name is decided by whichever record arrived first, and the other
    becomes an alias. The identity is what has to be order-free, so the
    comparison is over entities-and-their-spellings, not over node names.
    """
    with pytest.MonkeyPatch.context() as mp:
        _pin_determinism(mp)
        from assimilator import import_markdown
        from digester import extract

        mp.setattr(extract, "call_with_document", fixture_responses.response_for)
        mp.setattr(import_markdown, "_INGESTS_DIR", str(fixture_documents.INGESTS_DIR))
        from anomalica_common.llm.allowance import Allowance
        from digester import cli

        mp.setattr(
            cli, "check_allowance", lambda **_: Allowance(ok=True, reason="pinned")
        )

        digests = {
            key: _digest(document, tmp_path / f"{key}.yaml")
            for key, document in fixture_documents.DOCUMENTS.items()
        }

        shapes = []
        for order in (("a", "b"), ("b", "a")):
            domain, infrastructure = _open_graph(tmp_path / "-".join(order))
            for key in order:
                _assimilate(domain, infrastructure, digests[key])
            shapes.append(_order_free_shape(domain, infrastructure))
            domain.close()
            infrastructure.close()

    assert shapes[0] == shapes[1]


def _order_free_shape(domain, infrastructure) -> dict:
    """Entities by every spelling that reaches them, plus the claim texts."""
    entities = []
    for node_id, name, node_type in domain.execute(
        "SELECT id, name, node_type FROM nodes"
    ):
        spellings = {name} | {
            r[0]
            for r in domain.execute(
                "SELECT alias FROM aliases WHERE node_id = ?", (node_id,)
            )
        }
        entities.append((node_type, tuple(sorted(spellings))))
    return {
        "entities": sorted(entities),
        "domain_claims": sorted(
            r[0] for r in domain.execute("SELECT content FROM claims")
        ),
        "infrastructure_claims": sorted(
            r[0] for r in infrastructure.execute("SELECT content FROM claims")
        ),
    }


# ---------------------------------------------------------------------------
# The no-spend guarantee is tested, not assumed.
# ---------------------------------------------------------------------------


def test_a_provider_call_would_fail_this_module():
    """Every route to a provider raises, including the ones no test uses."""
    from anomalica_common.llm import transport

    for name in GUARDED_TRANSPORT_CALLS:
        dispatcher = getattr(transport, name, None)
        if dispatcher is None:
            continue
        with pytest.raises(ProviderCallAttempted):
            dispatcher("preamble", "document", "task", "sonnet", None, False)

    with pytest.raises(ProviderCallAttempted):
        transport.subprocess.run(["claude", "-p"])

    with pytest.raises(ProviderCallAttempted):
        socket.create_connection(("api.anthropic.com", 443))

    from assimilator import embeddings

    with pytest.raises(ProviderCallAttempted):
        embeddings.embed_text("anything")


def test_no_api_key_is_visible_to_this_process():
    assert [k for k in os.environ if k.endswith("_API_KEY")] == []


def test_the_spend_ledger_this_module_could_write_is_under_a_temporary_directory():
    """The transport flushes its ledger row at interpreter exit, after every
    fixture has gone, and `ledger.enabled()`'s PYTEST_CURRENT_TEST guard is NOT
    set during that flush. Only the import-time redirect holds. A row from a
    development run of these tests reached the production ledger and had to be
    deleted by hand, so this asserts what the module RESOLVES rather than what
    the environment string says."""
    from anomalica_common.llm import ledger

    assert not ledger.enabled()
    assert _SANDBOX in ledger.path().parents
    assert ledger.path() != Path.home() / ".local/share/scheduler/model-dispatch.jsonl"


def test_exactly_two_model_calls_were_served_per_record(pipeline):
    """Two passes and no more.

    A third call means a chunk boundary or an iteration round nobody expected,
    and every canned response after the first would be answering a question it
    was not written for - which reads as a strange extraction result rather
    than as a fixture that no longer fits.
    """
    served = [(call["document"], call["pass"]) for call in pipeline.first_pass_calls]
    assert sorted(served) == sorted(
        (key, name)
        for key in fixture_documents.DOCUMENTS
        for name in ("nodes", "claims")
    )
    assert all(call["use_api"] is False for call in pipeline.model_calls)
    assert all(call["model"] == "test-canned" for call in pipeline.model_calls)
