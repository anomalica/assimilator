# Cross-source linking experiments, 2026-09-02

Question (Mark): when one record states something and another record states a
related thing, does anything in the pipeline link them? Test pair:

- A: Liberation Times, 2026-07-12, "FBI Photos Show Orbs Beneath Helicopter
  During Alleged ODNI UAP Luring Operation" (record 7229c46a..., web, 15 KB)
- B: Ross Coulthart Q&A, 2026-08-09, "Skywatcher didn't disappear, it went
  government" (record 76d7e761..., video transcript, 83 KB)

Both describe a United States government operation in the south-western United
States in late 2025 that lured or summoned unidentified objects and then
observed, intercepted or shot at them. Neither names the other.

Nothing below touched the live graph, the digests repo, or the digester's
prompts. All model calls ran on the Claude subscription (Haiku). Artefacts are
in this directory.

## Baseline: what the current system does with the pair

Record A was digested (Haiku, the policy's model for a short web record) and
both records were imported into a copy of the live graph.

| Mechanism | Result |
|---|---|
| Shared entity nodes | Three, all corpus-wide hubs: the UAP topic (45 records), United States Congress (35), ODNI (15). No node specific to the story is shared. |
| Claim-to-claim similarity (the corroborate pass) | Best cross-record pair 0.70; no pair above 0.75. The pass has only ever accepted pairs at 0.90+, and asks "same fact?", which these are not. |
| Citation (infrastructure claims) | Not applicable: neither record cites the other. |
| Manual link in the workbench | None exists. The curation ledger has merge, unmerge and reject on nodes only. |

The idea that connects the two records ("the operation") has no node in either
digest, although B refers to "the UAP operation" in 30 of its 100 claims. Of the
six model variants held for B, DeepSeek and Sonnet emitted an operation node
unprompted; Haiku (the canonical) did not.

## Experiment 1: whole-record and node-anchored similarity (no model)

`exp1.py`. Does anything computed from existing embeddings put B near A?

| Measure | Rank of B among A's neighbours | Rank of A among B's |
|---|---|---|
| Record centroid cosine | 8 of 108 | 17 of 108 |
| Claim kNN (k=20) hits aggregated by record | 9 of 76 | 11 of 77 |
| Shared-node inverse-document-frequency score | 15 of 108 | - |
| Max claim-pair similarity among the 105 record pairs sharing ODNI | 15 of 105 | - |

Large books (In Plain Sight, Imminent) dominate every list because any
"government + UAP" claim has hundreds of neighbours there. `exp2.py` shows the
same at claim level: for B's claim "ODNI is involved in the UAP interception
operation", A's matching claim is neighbour 6; for most theme claims the other
record is ranked 30-200.

Conclusion: the pair is in a top-15 shortlist by claim kNN but never at the top.
Similarity alone cannot surface it without flooding.

## Experiment 2: theme search (no model, existing command)

`assimilator search "ODNI UAP luring operation"` returns A's claims at ranks
1, 2, 4 and B's at 3, 7, 9. The vector space holds the theme; what is missing
is something that names it.

## Experiment 3: record-pair judge (Haiku, one call per candidate pair)

`judge.py` (first prompt), `judge2.py` (15 shortlisted pairs), `judge3.py`
(stricter prompt). Input: the full claim lists of two records. Output:
`same_subject | possibly_related | unrelated`, a name for the shared specific
subject, and the linking claim pairs. 20-80 s per call.

Strict prompt (`judge3.py`) results:

| Pair | Verdict | Named subject |
|---|---|---|
| A vs B (target) | possibly_related | December 2025 ODNI south-western US UAP operation |
| A vs the ODNI document itself (USPER Narrative) | same_subject | December 2025 UAP encounter at south-western weapons test range |
| A vs David Grusch interview 2026-08-13 | possibly_related | December 2025 orb phenomenon (Tranche 4 release) |
| A vs Episode 73 (Burlison) 2026-06-30 | possibly_related | Burlison's briefing on a Western US facility incident |
| A vs Pressure Mounts (same outlet) | unrelated | |
| B vs 7NEWS documentary (same journalist) | unrelated | |
| B vs Coulthart Answers Your Questions | unrelated | |
| B vs Dr Phil Reality Check | unrelated | |
| B vs David Grusch interview | unrelated | |

The Grusch and Episode 73 links were checked against the claim text and are
real: Grusch discusses "luminous orbs appearing at a sensitive test range" from
the Tranche 4 release; Episode 73 has Burlison briefed on "plasmoid-type
phenomena at a Western United States military installation". Neither was linked
to A by any existing mechanism. The first, looser prompt (`judge-results2.json`)
produced four generic false positives ("the Trump administration's UAP efforts");
the strict wording removed all four and kept every real link.

Cost shape: claim-kNN shortlist of ~15 records per record gives ~800 pairs for
the 108-record corpus; records over ~300 claims need chunking or exclusion.

## Experiment 4: one extra node rule at extraction (Haiku, two re-digests)

`nodes-prompt.diff`: one paragraph added to the nodes prompt, in a scratch copy
of the digester, saying an operation or programme the document repeatedly refers
to as a definite thing is a project node even without an official name.
Examples in the rule were kept unrelated to this story.

- A re-digested: project "ODNI UAP luring operation (December 2025)" plus an
  event for the incident (`lt2.yaml`).
- B re-digested: project "Department of Defense UAP Intercept Operation at
  White Sands (November 2025-August 2026)" (`rc2-redigest.log`; the first claims
  call hung for 30 minutes and was retried by the transport).

Both re-digests imported into a fresh copy of the graph (`exp3.py`, `rc2.yaml`):

- The two operation nodes are cosine 0.727 apart by name embedding. From B's
  side, A's operation node is the nearest node in the whole graph (rank 1 of
  10,784) and A's incident event is rank 2 (0.709). From A's side, B's
  operation is rank 10, behind A's own event, briefing and document nodes.
- The fuzzy name matcher does not fold them (different words), so they stay
  two nodes. The entity reranker (`exp6.py`) scores the pair 0.12 and scores B
  against AAWSAP 0.98, so it is no use for this.
- Limitation: Haiku attached only ONE claim to each operation node; the other
  claims went to the incident event, White Sands, and the UAP topic as before.
  The node exists but is thin, and the rule would need a matching nudge in the
  claims pass to carry the claims about the operation.
- Also seen: A's incident event and the ODNI document record's own event node
  ("Late 2025 UAP encounter at a U.S. weapons test range") are the same event
  and were not matched either (0.70).

## Correction, 2026-09-03: the boundary is not stable run to run

The assimilator's first live smoke (the built `relate` command, same strict
wording, same model, one pair per call) on the Coulthart record's 15
shortlisted records returned 3 possibly_related and 12 unrelated. All three
positives were generic ("Trump's UAP disclosure initiative and covert
operations", the same-journalist 7NEWS link, a tangential White Sands mention),
and two of them are pairs the run above judged unrelated. So "zero false
positives on nine pairs" held for one run; at possibly_related the judge is
noisy, roughly 3 generic positives per 15 pairs. Mitigations being built: batch
5-8 pairs per call so the model judges against a comparison set, and a confirm
round that re-judges every positive once in a fresh batch and keeps it only if
reproduced.

Cost framing correction: each subscription CLI call carries about 108,000
tokens of cached Claude Code system context on top of the prompt (measured on
the merge verify pass: 25 calls, 7,000-token prompts, 2.7M cached tokens,
notional $2.45). Calls, not pairs, are the cost unit; batching cuts 1,310 calls
to roughly 220.

## What works

1. The record-pair judge with the strict prompt. It found the target link and
   two further records on the same incident, with no false positives on nine
   pairs. It needs no change to extraction. Output is a "possibly related"
   record pair with a named shared subject: a review queue, not an assertion.
2. The node rule gives each record a stable node for the operation, which is
   what a human merge or a "related" edge can hang off. On its own it does not
   link the two records; the names differ and only a judge can say they may be
   the same.

Neither is the corroborate pass. That pass answers "same fact" at 0.90+; this
question is "same specific subject" at 0.65-0.75, and only a reader of both
claim lists can answer it.

## Live smoke, 2026-09-03 (assimilator `relate`, commit 8c63df3)

Batched 6 pairs per call, 80-claim cap per side, confirm round. On the two
test records in the live graph: A <-> B found and confirmed ("December 2025
covert UAP operations in southwestern US"); A <-> the ODNI document (USPER
Narrative) found and confirmed; the Grusch and Episode 73 links missed (Grusch
judged unrelated in batch, Episode 73 not in A's top 15); one generic positive
survived confirmation (B <-> "Coulthart Answers Your Biggest UAP Questions",
a strategy rather than an incident). Batching removed two of the three generic
positives from the one-pair-per-call run. Measured: ~$0.17 notional and
~190,000 cached tokens per call, 45-84 s per call.
