# Entity reranker trial: Qwen3-Reranker-0.6B over the merge-candidate queue

Date: 2026-09-02. Asked by anomalica/master (Mark, cleared, no spend). Code:
`assimilator/entity_reranker.py`, `propose_merges.py --rerank` (flag, default off).

## What was measured

The reranker reads both names, both node types and up to three claims from each
side and answers "same real-world entity?" as P(yes). Two conditions were scored:
WITH claims (the design) and NAMES ONLY (the information the current rules have).
Compared against the rules' own score: name-equivalence 0.95/0.9, structure-aware
Levenshtein above 0.75, and 0 for a pair the rules would never surface.

Ground truth. Positives: 178 survivor/victim pairs from the live `node_merges` table
(human-applied merges) whose victim still has claims on record (106 of them cross-type).
Negatives: 337 - ONE human rejection (the curation ledger holds exactly one), 67
numbered siblings within a type (Apollo 11 / Apollo 12, Boeing 727 / 777, NDAA FY2022 /
FY2023), 269 same-surname persons with different, non-nickname forenames (Harry Reid /
Garry Reid, Helene Cooper / Gordon Cooper). The negatives are labelled BY
CONSTRUCTION, not by a reviewer; unreviewed candidates are deliberately not negatives,
because absence of a merge is not a rejection. Files: `reports/reranker-eval-positives.json`,
`reports/reranker-eval-negatives.json`, `reports/reranker-eval-scores.json`.

Two limits of this ground truth. The ledger has one rejection, so the rules'
precision on their own candidates cannot be measured against humans (the 80 rule
candidates in the set are 78 positives + 2 negatives because humans merged what the
rules showed them). And no name-divergent positive survives with claims (the Nimitz /
Tic Tac split was merged before the July rebuild), so the with-claims condition's
strength on that case is untested here.

## Numbers

CORRECTION (2026-09-02, 23:30): the 178 positives are 178 distinct node-id pairs but
only 90 distinct NAME pairs. Replayed merges mint the same victim again on every
rebuild, so `node_merges` holds 37 rows for "The Pentagon" ~ "The Pentagon", 30 for
the UAPTF pair and 24 for "Galileo Project" - three pairs were half the positive set,
and every figure first reported was computed per row. The table below is one row per
name pair. AUC barely moves (rank-based; the repeats were well scored); the absolute
recall and precision figures do, and the earlier ones are withdrawn.

| scorer | AUC | at a cut: precision / recall (tp, fp), over 90 positives |
|---|---|---|
| current rules | 0.603 | >=0.75: 0.90 / 0.21 (19, 2) |
| reranker, with claims | 0.867 | >=0.5: 0.48 / 0.87 (78, 83); >=0.9: 0.52 / 0.68 (61, 56) |
| reranker, names only | 0.981 | >=0.9: 0.83 / 0.92 (83, 17); >=0.7: 0.66 / 0.97 |

- Recall. 71 of the 90 human-merged pairs are invisible to the rules (cross-type, or
  fuzzy below 0.75: `USA` / `United States of America`, `UK, Suffolk, Rendlesham Forest` /
  `United Kingdom, England, Rendlesham Forest`, `UK Ministry of Defence` / `United Kingdom
  Ministry of Defence`). The reranker scores 51 of those 71 at >=0.9 with claims, and
  finds every country-spelling pair at 0.95-1.00 in both conditions.
- Precision, with claims. The failures are RELATIVES AND CO-WITNESSES: 48 of 269
  same-surname pairs score >=0.9 (Vickie / Colby Landrum, Karl / Mary Strieber, Jacinta /
  Olimpia Marto) - people in the same incident whose claims overlap. The instruction names
  "a parent and a child" as not the same entity; at 0.6B the model does not hold to it
  when the claims read alike. Names only makes 17 such errors instead of 48.
- Precision, names only. Better on every axis here, but by construction: these positives
  are spelling, acronym and country-form variants, and these negatives differ by forename.
  Names only is blind to a duplicate whose names share no word, which is the case the
  claims exist for.
- Same-type pairs only (the rules' own ground): 72 positives, AUC rules 0.546, reranker
  0.867. (All 88 duplicate rows were cross-type, so this line is unchanged.)
- Known-bad cases. Harry / Garry Reid: correctly apart (0.04 with claims, 0.29 names
  only; the rules do not surface it either). Atlant / Atlantis (persons, the one human
  rejection): WRONGLY together at 0.989 in both conditions - the rules surface it too
  (0.857). Atlantis orbiter / Atlantis person: 0.39 (correctly apart, not confidently).
  Nickname and acronym pairs are the with-claims condition's weakest positives: Thomas /
  Tom Bearden 0.03, Robert / Bob Monroe 0.09, `NPR` / `National Public Radio (NPR)` 0.15
  - three random claims from each side read as two people. The nickname table in
  matching.py handles exactly these, and it should stay in front of the reranker.
- Order artefact: query and document are not symmetric roles. Over 120 pairs scored both
  ways, mean |a->b - b->a| = 0.114, median 0.033, 11 decisions flip at 0.5. Production
  scores both orders and averages; the evaluation used one order.

## Today's queue, reranked

The rules' pass (`propose()`) produced 148 clusters / 327 pairs in 1,491 s on the host -
76 cross-type same-name clusters and 72 fuzzy - and had not been regenerated since
2026-06-28 (the file the workbench read was that old). Reranked on the GPU: 327
pairs, 654 prompts, 64.9 s wall, peak 2730 MB of device memory
(reserved 3670 MB), model load about 11 s.

Top of the reranked queue: acronym expansions and punctuation variants the fuzzy rule
scored 0.76-0.90 (`Westall UFO Sighting` / `Westall Unidentified Flying Object (UFO)
sighting 1966` at 0.998). Bottom: the fuzzy rule's false positives - `Austria` /
`Australia` (rule 0.88, reranker 0.002), `Bolivia, Santa Cruz` / `USA, California, Santa
Cruz` (rule 1.00, reranker 0.013), `Res communis` / `Communism`, `Darren` / `Ed Warren`,
and initials-versus-name pairs. Medians: fuzzy 0.933 (38 of 72 at >=0.9), cross-type
0.956 (53 of 76 at >=0.9) - the cross-type set is the country-as-organisation /
deity-as-topic taxonomy question and the reranker's high scores there are not a merge
verdict; those go to adjudication (node-types.md).
Files: `~/.local/share/assimilator/merge-candidates.json` (rules order, fresh) and
`merge-candidates.reranked.json` (reranked, with `rule_score` and `pairs` per cluster).

## Verdict

As a RANKER of what the rules surface: it wins. Same-type AUC 0.867 against 0.546 (90 distinct pairs), and
on today's queue it moves the fuzzy rule's junk to the bottom and its under-scored true
duplicates to the top. Ship behind the flag; default off until the workbench shows
`rule_score` beside it so a reviewer can see both.

As a CANDIDATE SOURCE: it cannot be one yet. Scoring is 0.14 s per pair on the GPU;
the graph has 3,200 live nodes and five million pairs. It finds 51 of the 71 duplicates
the rules miss only when something puts them in front of it - a blocking or embedding
shortlist, which does not exist in the code today (the `--verify` pass the module
docstring describes was never built).

The Claude verify pass stays for cross-type pairs. The reranker's cross-type scores
rank the queue; they do not decide.

Not fixed by this: the rules' pass itself costs 25 minutes (O(n^2) Levenshtein), which
dwarfs the reranker's 1.5 minutes; and the Atlant / Atlantis error shows the model
shares the rules' blind spot on near-identical names of distinct people.


# Shortlist stage: kNN over profile vectors, then the reranker (2026-09-03, 01:10)

Asked by master after the trial above: build the candidate source the reranker
lacked. Path: for every live node (10,784) a profile text - name, type, first three
claims - embedded through the endpoint (Qwen3-Embedding, cached); the 20 nearest
nodes by cosine, cross-type allowed (169,902 pairs); union with the rules' 327 pairs
(170,074 after dropping 7 whose node a replay had retired); names-only reranker as a
cheap filter at 0.3 (30,778 pass); with-claims reranker, both orders averaged, on
those. Scripts in the session scratchpad; outputs `reports/shortlist-eval-2026-09-02.json`
(every scored pair) and `reports/shortlist-positives-path-2026-09-02.json` (the 90
distinct positive pairs through both stages).

## Recall of the human-merged pairs, on the 90 DISTINCT name pairs (85 whose survivor is still live)

| stage | reached |
|---|---|
| current rules (name-equivalence, fuzzy) | 19 of 90 |
| kNN, victim's top 20 | 75 of 85 (top 5: 52; top 50: 83) |
| kNN or rules | 75 of 90 |
| after names-only >= 0.3 | 74 (one lost: Robert / Bob Monroe, names-only 0.29) |
| after with-claims >= 0.5 / 0.7 / 0.9 | 73 / 64 / 44 |
| with names-only >= 0.9 as well, at with-claims >= 0.9 | 41 |

The blocking works: 20 neighbours reach 88% of the pairs a human merged, 50 reach
98%. The scorer is the weak stage, and it loses the pairs the rules were built for:
the reached positives it scores lowest are `ABC` / `American Broadcasting Company
(ABC)` (with-claims 0.50), `ISIS` / `Islamic State (ISIS)` (0.53), `NPR` (0.55),
`Vatican` / `Holy See` (0.54), `Tom` / `Thomas Bearden` (0.50), `Bob` / `Robert Monroe`
(0.10) - names-only has every one of them at 0.94 to 1.00. Three random claims from
each side read as two entities when the name is an acronym or a nickname.

## The queue it produces

| cut | pairs | beyond the rules' 327 |
|---|---|---|
| with-claims >= 0.9 | 5,615 | 5,509 |
| with-claims >= 0.7 | 9,990 | 9,870 |
| with-claims >= 0.9 AND names-only >= 0.9 | 3,779 | 3,680 |

Precision, judged by eye over 20 random pairs from each region beyond the rules
(my reading, not a reviewer's): in the band where BOTH scores are >= 0.9 about 7 or 8
of 20 are one entity - `Lockheed U-2` / `U-2 spy plane`, `EarthTech International` /
`EarthTech`, two records of one 1986 hypnosis session, two of one 1982 pilgrimage,
the two crash-retrieval programme nodes; the rest are related things (a paper and
its mission, the Gateway tapes and the Gateway programme, an investigation and the
agency that ran it). Where with-claims is >= 0.9 but names-only is below, 0 to 2 of
20: `Io` / `Jupiter`, `Sidon` / `Tyre`, `Challenger` / `Columbia`, `CUFOS` / `MUFON`,
`Padre Pio` / `Joseph of Cupertino` - the relation-versus-identity failure, at 0.99.
Where names-only is >= 0.9 but with-claims is below, 0 to 2 of 20 as well: names-only
at 0.9 is noisy on this broad a shortlist (`Santa Sabina` / `Trevi Fountain` at 0.93),
where on the trial's negatives it was not - those differed by forename.

So the combined band of 3,680 pairs holds on the order of 1,300 real duplicates the
rules cannot see, mixed with about 2,400 related-but-distinct pairs at the same
scores. That is four times the review work the rules produce, for perhaps four
times the duplicates, and the reviewer cannot tell the two apart from the scores.

The AARO pair master found by hand (project minted beside the organisation) is in
the shortlist at names-only 0.998, with-claims 0.976 - it would have been near the
top of the combined band.

## Cost

Embedding 219 s (endpoint cache warm; 1.7 texts a second cold), names-only 3,533 s
for 170,074 pairs at batch 48, with-claims 2,848 s for 30,778 pairs in both orders -
about 1h50 of the laptop GPU for one pass. The card is power-capped at 20 W with the
memory clock held at 810 MHz, which makes the 0.6B model memory-bound: batch size is
the throughput, and eager attention capped the batch at 32 (3 pairs a second). sdpa
with length-sorted batches of 48 gave 34.5 a second. Batch 96 does not fit in 4.2 GB.

## Two faults of the method itself

- The profile is a node's FIRST THREE CLAIMS BY ROW ORDER. Tonight's entailment
  re-import rewrote 1,368 claims and reordered rows; kNN recall on the same positives
  fell from 169 to 134 of 178 rows between the afternoon and the evening run, and the
  positives whose survivor profile changed missed at 29% against 10% for the rest. A
  profile must be deterministic in the graph's content - claims sorted by claim_hash,
  or the node's most-cited claims - not in its storage order.
- A candidate file older than the last curation replay carries retired ids: a replayed
  merge retires a node re-minted since, stamped with the ledger's ORIGINAL date, so
  `retired_at` does not say when. The run now drops such pairs (7 tonight).

## Verdict

As a candidate SOURCE the shortlist does what the rules cannot: 75 of the 90 pairs
humans merged are within 20 neighbours, against 19. As a QUEUE it is not reviewable
yet: the with-claims scorer confuses relation with identity at 0.99 and the
names-only scorer is noisy at this breadth, so the best band (both >= 0.9) is 3,680
pairs at roughly one-in-three precision. Not shipped into the reviewer's queue.
Recommended next step, master's call: either a stronger judge on the combined band
only (3,680 pairs is within reach of the Claude verify pass, subscription, paced) or
a larger reranker for that band; and the deterministic profile before either, since
the shortlist's recall moves with row order today.
