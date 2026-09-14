# Search reranker benchmark

> **The initial performance evidence is invalid.** Retain it for audit, but use
> only the controlled CUDA result below for the replacement decision. CUDA
> allocator and RSS figures remain process-local, not total-host memory or
> available headroom.

This benchmark compares the existing `cross-encoder/ms-marco-MiniLM-L-6-v2`
search reranker with `ibm-granite/granite-embedding-english-r2`. Granite English
R2 is a bi-encoder, not a cross-encoder, so it ranks the same frozen 50-claim
candidate pool by query-to-claim cosine similarity. MiniLM scores query-claim
pairs directly. Candidate generation is held constant.

The fixture contains project-authored atomic claim text only and is marked
CC0-1.0. Relevance is derived from the reviewed graph: a claim is relevant when
the target node occurs in `claim_node_refs` or as `claims.speaker_id`. The 16
queries were written and reviewed for this benchmark to paraphrase their target
rather than merely repeat its canonical name. This is a small domain benchmark,
not a general retrieval claim.

## Decision rule

Granite replaces MiniLM only if all of these hold on the same device and fixture:

1. Mean nDCG@10 improves by at least 0.03 absolute.
2. Mean MRR@10 does not decline.
3. No more than two queries lose over 0.10 nDCG@10.
4. Median warm query latency and peak accelerator memory are each no more than
   twice MiniLM's.

The quality margin is deliberately larger than a one-query movement in this
small set. The resource limits prevent a modest quality change from replacing a
small interactive reranker with a materially slower or larger model.

## Reproduction

Generate the frozen candidates in the existing development image:

```bash
docker run --rm --network host \
  -v "$PWD/workspace:/home/nonroot/workspace" \
  -v "$HOME/repos/anomalica/anomalica-common/src:/opt/anomalica-common:ro" \
  -v "$HOME/.local/share/assimilator:/data:ro" \
  -e PYTHONPATH=.:/opt/anomalica-common \
  --user "$(id -u):$(id -g)" \
  -w /home/nonroot/workspace anomalica-assimilator:development \
  python reports/search-reranker-benchmark-2026-09-13/generate_fixture.py \
  --db /data/knowledge.db \
  --out reports/search-reranker-benchmark-2026-09-13/fixture.json
```

Run each pinned model in a fresh process:

```bash
python reports/search-reranker-benchmark-2026-09-13/benchmark.py \
  --fixture reports/search-reranker-benchmark-2026-09-13/fixture.json \
  --model minilm --device cuda \
  --out reports/search-reranker-benchmark-2026-09-13/minilm-cuda.json

python reports/search-reranker-benchmark-2026-09-13/benchmark.py \
  --fixture reports/search-reranker-benchmark-2026-09-13/fixture.json \
  --model granite --device cuda \
  --out reports/search-reranker-benchmark-2026-09-13/granite-cuda.json
```

Both model revisions are pinned in `benchmark.py`; each result records the model
weight hash, fixture hash, runtime versions, quality, latency, memory and
per-query top ten.

## Workbench review artefacts

`side-by-side.json` is the report-only, CC0 Workbench view of all 16 controlled
queries. For each model it carries the ordered top ten with claim id, atomic
claim text, graph-derived relevance, score, exact shared-top-ten rank change and
rights-safe source-record identity resolved from the current canonical digest.
An absent rank is labelled as entering or exiting the top ten rather than being
invented. Source bodies, excerpts, images and originals are excluded.

The view states the benchmark boundary explicitly: source-local accounts,
temporal relations, entity merge safety, merge proposals and automatic merges
were absent. It evaluates only reranking of the frozen text candidate pools.
`evaluation-state.json` exposes the adopted `retain-minilm` decision through the
canonical `anomalica/evaluation-state/1` owner-provider contract and binds it to
the public fixture and controlled comparison.

Regenerate the side-by-side view without running either model:

```bash
python build_side_by_side.py \
  --fixture fixture.json \
  --minilm controlled-run-2026-09-14/controlled-minilm-cuda.json \
  --granite controlled-run-2026-09-14/controlled-granite-cuda.json \
  --digests /home/mark/repos/anomalica/digests \
  --out side-by-side.json
```

Apply the decision rule with:

```bash
python reports/search-reranker-benchmark-2026-09-13/compare.py \
  --minilm reports/search-reranker-benchmark-2026-09-13/minilm-cuda.json \
  --granite reports/search-reranker-benchmark-2026-09-13/granite-cuda.json \
  --out reports/search-reranker-benchmark-2026-09-13/comparison-cuda.json
```

## Initial uncontrolled results (invalid for decisions)

Quality is identical between CPU and CUDA. The current hybrid order is included
as context. Resource figures below are uncontrolled raw observations.

| Ranking | nDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Hybrid candidate order | 0.4618 | 0.5017 | 0.2854 |
| MiniLM | 0.5668 | 0.6286 | 0.3751 |
| Granite English R2 | 0.5729 | 0.6568 | 0.3299 |

| Device and measure | MiniLM | Granite | Granite / MiniLM |
| --- | ---: | ---: | ---: |
| CUDA median query latency | 141.4 ms | 622.7 ms | 4.40x |
| CUDA p95 query latency | 344.9 ms | 980.9 ms | 2.84x |
| CUDA peak allocated memory | 157.2 MiB | 689.3 MiB | 4.38x |
| CPU median query latency | 4,916.1 ms | 24,411.5 ms | 4.97x |
| CPU p95 query latency | 6,588.7 ms | 38,967.2 ms | 5.91x |
| CPU peak resident set | 1,064.1 MiB | 1,736.1 MiB | 1.63x |

Granite gains 0.0061 mean nDCG@10 and 0.0282 MRR@10, but loses 0.0452
Recall@10. Four of 16 queries lose more than 0.10 nDCG@10: AATIP, David
Grusch, the Nimitz encounter and the Central Intelligence Agency. It fails the
minimum quality gain and per-query stability criteria. Resource criteria are
unevaluated.

**Recommendation: retain MiniLM.** Do not change the production default.
Granite English R2 is not a drop-in cross-encoder and its small aggregate quality
gain does not offset its quality regressions. The observed CPU and CUDA costs
cannot support this decision until they are rerun under exclusive control.
Pre-embedding every claim with Granite would test a different retrieval design,
not this second-stage reranker replacement.

Candidate recall averages only 0.0634 because this benchmark freezes the current
top-50 hybrid pool and broad target nodes can have hundreds of linked claims. The
ranking metrics are therefore conditional on candidates reaching the reranker;
they do not validate the first-stage embedding model or establish corpus-wide
recall. The low value is a separate retrieval-evaluation warning, not evidence
for either reranker.

## Controlled CUDA result

The controlled rerun supersedes the uncontrolled performance observations above.
It ran from `2026-09-14 07:30:32` to `07:38:49` JST with six fresh processes per
model in `ABBA/BAAB/ABBA` order, three warm-ups per process, three timed repeats
per query and recorded random query orders. The 1 Hz trace contains 498 samples:
62 pre-run idle samples, 61 post-run idle samples, a maximum 1.152-second sample
gap, 15 MiB used before and after, and zero unexpected compute processes. Both
services were inactive before and after. The evidence manifest therefore marks
exclusive control verified.

| Controlled CUDA measure | MiniLM | Granite | Granite / MiniLM |
| --- | ---: | ---: | ---: |
| Median wall query latency | 85.1 ms | 404.0 ms | 4.75x |
| p95 wall query latency | 126.1 ms | 519.3 ms | 4.12x |
| Median CUDA-event latency | 84.8 ms | 403.9 ms | 4.76x |
| Throughput | 572.4 pairs/s | 122.8 pairs/s | 0.21x |
| Process-local peak CUDA allocation | 157.1 MiB | 689.8 MiB | 4.39x |
| Process-local peak RSS | 1,240.5 MiB | 1,752.9 MiB | 1.41x |
| Median model load | 0.305 s | 3.189 s | 10.46x |

Per-process median query latency ranges were 81.1-89.7 ms for MiniLM and
381.9-414.0 ms for Granite. Wall and CUDA-event timings agree within 0.3 ms.
Quality and ranked claim IDs were identical across all six repetitions per
model.

The controlled comparison validates the resource criteria and confirms that
Granite fails both: median latency is over twice MiniLM's and process-local peak
CUDA allocation is over twice MiniLM's. It also still fails the independent
quality-gain and per-query-regression criteria. The recommendation remains:
retain MiniLM and do not change the production default.

## Performance validity

The original runs loaded MiniLM and then Granite sequentially, not interleaved or
randomised. Each ran in a fresh process, scored one warm-up query, then processed
the same 16 queries in fixed order with three consecutive timed repetitions per
query. CUDA synchronisation bracketed each sample. There was no exclusive GPU
lock, contemporaneous process list or utilisation trace. One earlier GPU sample
showed 1,178 MiB already used but did not identify the process or retain a
timestamp.

Retrospective system logs show speech-to-text stopped at
`2026-09-13T09:50:17+09:00`, before the CUDA result files at `09:57:24` and
`10:00:03`, but this cannot exclude another GPU workload. Speech service starts
at `10:20:16` and `10:25:42` overlapped the later Granite CPU run. Both latency
and throughput comparisons are therefore invalid. PyTorch CUDA allocation and
RSS measurements remain useful only as each benchmark process's own footprint.

## Controlled rerun protocol

1. Pre-download and verify both pinned model revisions before the timed window.
2. Pause speech-to-text and scheduler GPU dispatch. Require an empty compute
   process list and driver-baseline memory for at least 60 seconds.
3. Record timestamped `nvidia-smi` process, utilisation, memory, clock, power and
   temperature samples at 1 Hz before and throughout every run.
4. Run at least five fresh-process repetitions per model in balanced randomised
   ABBA/BAAB order. Randomise query order with recorded seeds and run three
   warm-ups before timing.
5. Keep CUDA synchronisation and record both CUDA-event and wall-clock timings.
   Archive the monitor trace, service states and exact run start/end timestamps.
6. Run CPU separately in a quiet window, with pinned cores and thread counts and
   the same balanced model order.
7. Supply the resulting evidence manifest through
   `--exclusive-control-evidence`; without it `compare.py` leaves resource
   criteria null and cannot recommend replacement.
