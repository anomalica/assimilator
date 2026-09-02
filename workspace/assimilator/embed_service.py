"""Local embedding endpoint: the one process that owns the vector space.

Consumers that need embeddings but must not carry the model - the workbench's
audit view clusters extraction variants inside a `uvicorn --reload` backend -
POST text here and get vectors back. The alternative, importing fastembed into
each consumer, would put a ~600MB ONNX behind every code reload and give each
consumer its own vector space, free to drift from the graph's without ever
raising an error. One owner, one space.

Backed by a content-addressed cache: the text a consumer clusters (a claim
already extracted into a digest) is immutable, so a text embedded once is
embedded forever. This matters more than it looks. `embed_batch` is a loop -
fastembed's mean pooling breaks on our custom Qwen3 ONNX above batch size 1 - so
every text is its own model inference, and a re-opened audit view would otherwise
pay for all of them again.

The cache is keyed by (text hash, model id), never text hash alone: a model
upgrade must miss rather than serve a vector from the superseded space, and both
spaces coexist while a re-embed runs.

Run: ``python -m assimilator.embed_service`` (inside the container - it needs
fastembed and the baked model at EMBEDDING_MODEL_PATH).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from assimilator.embeddings import (
    EMBEDDING_MODEL_ID,
    deserialise_f32,
    embed_text,
    serialise_f32,
)
from assimilator.data_dir import data_dir

DEFAULT_PORT = 8077
MAX_BODY_BYTES = 32 * 1024 * 1024

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS text_embeddings (
    text_hash   TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (text_hash, model_id)
);
"""

# fastembed's ONNX session is not documented as thread-safe and this server is
# threaded, so inference is serialised. It is the slow path either way; the cache
# is what keeps the endpoint responsive, not concurrency.
_inference_lock = threading.Lock()


def default_cache_path() -> Path:
    override = os.environ.get("ANOMALICA_TEXT_EMBEDDINGS_DB")
    if override:
        return Path(override)
    return data_dir() / "text-embeddings.db"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(CACHE_SCHEMA)
    conn.commit()
    return conn


def cached_vectors(
    conn: sqlite3.Connection, hashes: list[str], model_id: str
) -> dict[str, list[float]]:
    """Every cached vector for these hashes IN THIS EMBEDDING SPACE. Hashes from a
    superseded model simply miss, so they are re-embedded rather than mixed."""
    if not hashes:
        return {}
    found: dict[str, list[float]] = {}
    for chunk_start in range(0, len(hashes), 500):  # keep under SQLite's variable cap
        chunk = hashes[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM text_embeddings "
            f"WHERE model_id = ? AND text_hash IN ({placeholders})",
            [model_id, *chunk],
        ).fetchall()
        for h, blob in rows:
            found[h] = deserialise_f32(blob)
    return found


def store_vector(
    conn: sqlite3.Connection, hash_: str, model_id: str, vector: list[float]
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO text_embeddings(text_hash, model_id, embedding, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        [
            hash_,
            model_id,
            serialise_f32(vector),
            datetime.now(timezone.utc).isoformat(),
        ],
    )
    conn.commit()


def embed_with_cache(
    conn: sqlite3.Connection, texts: list[str], model_id: str | None = None
) -> list[list[float]]:
    """Vectors for `texts`, in order, embedding only what the cache misses.

    Duplicate texts in one request collapse to a single inference - the audit
    clusters variants of the same fact, so exact repeats across models are the
    common case, not an edge case.

    `model_id` resolves at call time, never as a default argument: a default
    binds at import, which would let the write path and the read path name
    different spaces after any rebind - the exact silent divergence this cache is
    keyed to prevent."""
    model_id = model_id or EMBEDDING_MODEL_ID
    hashes = [text_hash(t) for t in texts]
    vectors = cached_vectors(conn, list(dict.fromkeys(hashes)), model_id)

    for text, hash_ in zip(texts, hashes):
        if hash_ in vectors:
            continue
        with _inference_lock:
            vector = embed_text(text)
        vectors[hash_] = vector
        store_vector(conn, hash_, model_id, vector)

    return [vectors[h] for h in hashes]


class Handler(BaseHTTPRequestHandler):
    conn: sqlite3.Connection

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._respond(404, {"error": "not found"})
            return
        cached = self.conn.execute(
            "SELECT count(*) FROM text_embeddings WHERE model_id = ?",
            [EMBEDDING_MODEL_ID],
        ).fetchone()[0]
        self._respond(200, {"model_id": EMBEDDING_MODEL_ID, "cached": cached})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/embed":
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._respond(400, {"error": "bad Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._respond(400, {"error": "empty or oversized body"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            texts = payload["texts"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._respond(400, {"error": 'expected {"texts": [...]}'})
            return
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            self._respond(400, {"error": "texts must be a list of strings"})
            return

        try:
            vectors = embed_with_cache(self.conn, texts)
        except Exception as exc:  # a failed embed must not look like a bad vector
            self._respond(500, {"error": f"embedding failed: {exc}"})
            return

        # model_id rides on every response: a consumer's cluster - and any human
        # verdict recorded against it - is not reproducible without its space.
        self._respond(200, {"model_id": EMBEDDING_MODEL_ID, "vectors": vectors})

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"embed_service: {fmt % args}\n")


def serve(port: int = DEFAULT_PORT, cache_path: Path | None = None) -> None:
    conn = open_cache(cache_path or default_cache_path())
    handler = type("BoundHandler", (Handler,), {"conn": conn})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    sys.stderr.write(
        f"embed_service: listening on 127.0.0.1:{port}, model {EMBEDDING_MODEL_ID}\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()
    serve(args.port, args.cache)


if __name__ == "__main__":
    main()
