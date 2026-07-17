"""The endpoint's cache contract. `embed_text` is stubbed throughout: these tests
are about caching and space-keying, not about the model - loading a 600MB ONNX to
assert a dict lookup would make the suite untestable for no coverage."""

import json
import threading
import urllib.request

import pytest

from assimilator import embed_service
from assimilator.embed_service import (
    cached_vectors,
    embed_with_cache,
    open_cache,
    serve,
    text_hash,
)

DIMS = 4
MODEL_A = "model-a:file.onnx:4"
MODEL_B = "model-b:file.onnx:4"


@pytest.fixture
def cache(tmp_path):
    return open_cache(tmp_path / "text-embeddings.db")


@pytest.fixture
def calls(monkeypatch):
    """Record every text the model was actually asked to embed."""
    seen: list[str] = []

    def fake_embed_text(text: str) -> list[float]:
        seen.append(text)
        return [float(len(text)), 1.0, 0.0, 0.0]

    monkeypatch.setattr(embed_service, "embed_text", fake_embed_text)
    return seen


def test_vectors_come_back_in_request_order(cache, calls):
    vectors = embed_with_cache(cache, ["alpha", "bb", "ccc"], MODEL_A)

    assert [v[0] for v in vectors] == [5.0, 2.0, 3.0]


def test_second_request_for_the_same_text_hits_the_cache(cache, calls):
    embed_with_cache(cache, ["alpha"], MODEL_A)
    embed_with_cache(cache, ["alpha"], MODEL_A)

    assert calls == ["alpha"]


def test_duplicate_text_in_one_request_is_embedded_once(cache, calls):
    """Different extraction models emitting a byte-identical claim is the common
    case in an audit, not an edge case."""
    vectors = embed_with_cache(cache, ["alpha", "alpha", "alpha"], MODEL_A)

    assert calls == ["alpha"]
    assert vectors[0] == vectors[1] == vectors[2]


def test_cache_survives_reopening(tmp_path, calls):
    path = tmp_path / "text-embeddings.db"
    first = open_cache(path)
    embed_with_cache(first, ["alpha"], MODEL_A)
    first.close()

    second = open_cache(path)
    embed_with_cache(second, ["alpha"], MODEL_A)

    assert calls == ["alpha"]


def test_a_different_model_misses_rather_than_serving_the_old_vector(cache, calls):
    """The upgrade case. Keyed on hash alone, this would serve a superseded
    space's vector and no error would ever be raised."""
    embed_with_cache(cache, ["alpha"], MODEL_A)
    embed_with_cache(cache, ["alpha"], MODEL_B)

    assert calls == ["alpha", "alpha"]


def test_both_spaces_coexist_so_a_re_embed_can_run_incrementally(cache, calls):
    embed_with_cache(cache, ["alpha"], MODEL_A)
    embed_with_cache(cache, ["alpha"], MODEL_B)

    rows = cache.execute(
        "SELECT model_id FROM text_embeddings WHERE text_hash = ?", [text_hash("alpha")]
    ).fetchall()
    assert sorted(r[0] for r in rows) == [MODEL_A, MODEL_B]


def test_cached_vectors_only_returns_the_requested_space(cache, calls):
    embed_with_cache(cache, ["alpha"], MODEL_A)

    assert cached_vectors(cache, [text_hash("alpha")], MODEL_A)
    assert cached_vectors(cache, [text_hash("alpha")], MODEL_B) == {}


def test_lookup_chunks_past_sqlites_variable_cap(cache, calls):
    """SQLite caps host variables per statement; a big audit would blow it."""
    texts = [f"claim number {i}" for i in range(1200)]
    embed_with_cache(cache, texts, MODEL_A)

    found = cached_vectors(cache, [text_hash(t) for t in texts], MODEL_A)
    assert len(found) == 1200


def test_empty_request_embeds_nothing(cache, calls):
    assert embed_with_cache(cache, [], MODEL_A) == []
    assert calls == []


# --- over the wire ----------------------------------------------------------


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_service, "embed_text", lambda t: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(embed_service, "EMBEDDING_MODEL_ID", MODEL_A)

    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    thread = threading.Thread(
        target=serve, args=(port, tmp_path / "cache.db"), daemon=True
    )
    thread.start()

    url = f"http://127.0.0.1:{port}"
    for _ in range(100):  # wait for bind rather than sleeping a fixed guess
        try:
            urllib.request.urlopen(f"{url}/health", timeout=0.1).read()
            break
        except OSError:
            continue
    return url


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        f"{url}/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_embed_response_carries_the_model_id(server):
    """Without the space it came from, a cluster - and any human verdict recorded
    against it - is not reproducible."""
    payload = _post(server, {"texts": ["alpha", "beta"]})

    assert payload["model_id"] == MODEL_A
    assert len(payload["vectors"]) == 2


def test_health_reports_the_space_and_cache_size(server):
    _post(server, {"texts": ["alpha"]})
    with urllib.request.urlopen(f"{server}/health", timeout=5) as response:
        payload = json.loads(response.read())

    assert payload["model_id"] == MODEL_A
    assert payload["cached"] == 1


def test_malformed_body_is_rejected(server):
    for bad in ({"wrong_key": []}, {"texts": "not a list"}, {"texts": [1, 2]}):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(server, bad)
        assert exc.value.code == 400


def test_unknown_route_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{server}/nope", timeout=5)
    assert exc.value.code == 404
