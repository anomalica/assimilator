"""Partitioning the vector-embedding backlog into schedulable jobs.

Vector embedding is hours of work - roughly 3 items a second, so a full corpus
pass is ~3 hours - and it was queued as ONE job. A single job holds the
background lane for its whole duration, so a document ingested during it waits
behind a task with no reason to be atomic. The work is already interruptible
(the command commits per chunk and skips rows already embedded), so the only
thing missing was a way to ask for a bounded slice.

THE PARTITION IS FIXED, NOT DERIVED, AND THAT IS THE WHOLE DESIGN. The obvious
scheme - "however many batches the remaining count needs" - renumbers every job
as rows get embedded, and the scheduler stages work by job id, so a staged job
whose id moves is a staged job that silently disappears. Bucketing on a hash of
the row's own id instead gives ids that never move: bucket 7 is bucket 7 for the
life of the corpus, it simply empties. New rows land in whichever bucket their id
hashes to, so growth spreads across the existing buckets rather than adding any.

BUCKETS is therefore a constant and must stay one. Changing it renumbers
everything exactly as a derived count would.
"""

from __future__ import annotations

import sqlite3
from zlib import crc32

# 32 buckets over the current backlog is ~1,000 items each, ~5.5 minutes at the
# measured 3.06 items/second - short enough that the lane can be interrupted
# between batches, long enough that per-job overhead is noise.
BUCKETS = 32


def bucket_of(row_id: str, buckets: int = BUCKETS) -> int:
    """Which bucket a row belongs to. Stable for the life of the row.

    crc32 rather than hash() because Python's hash is salted per process, so the
    same id would land in a different bucket on every run - which would make the
    ids look stable while the CONTENTS moved under them, the worse failure.
    """
    return crc32(row_id.encode("utf-8")) % buckets


def pending_by_bucket(
    conn: sqlite3.Connection, model_id: str, buckets: int = BUCKETS
) -> dict[int, int]:
    """bucket -> count of rows not yet embedded in this vector space.

    Buckets with nothing left are ABSENT rather than zero, so a caller enumerating
    jobs gets only real work. Pure sqlite and stdlib: the scheduler imports this
    and must not acquire a fastembed dependency to enumerate a queue.
    """
    remaining: dict[int, int] = {}
    for kind, query in (
        ("claim", "SELECT id FROM claims"),
        ("node", "SELECT id FROM nodes WHERE retired_at IS NULL"),
    ):
        # Keyed on (kind, id) exactly as embedding_model is. Claim ids and node
        # ids are both uuids and do not collide today, but a done-set keyed on
        # the bare id would make a future collision look like completed work.
        try:
            done = {
                r[0]
                for r in conn.execute(
                    "SELECT id FROM embedding_model WHERE kind = ? AND model_id = ?",
                    (kind, model_id),
                )
            }
        except sqlite3.OperationalError:
            done = set()  # table absent until the first embed run

        for (row_id,) in conn.execute(query):
            if row_id in done:
                continue
            b = bucket_of(row_id, buckets)
            remaining[b] = remaining.get(b, 0) + 1
    return remaining
