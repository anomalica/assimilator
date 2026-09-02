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

| scorer | AUC | at a cut: precision / recall (tp, fp) |
|---|---|---|
| current rules | 0.716 | >=0.75: 0.97 / 0.44 (78, 2) |
| reranker, with claims | 0.867 | >=0.5: 0.67 / 0.93 (166, 83); >=0.9: 0.67 / 0.63 (113, 56) |
| reranker, names only | 0.980 | >=0.9: 0.91 / 0.96 (171, 17); >=0.7: 0.80 / 0.98 |

- Recall. 100 of the 178 human-merged pairs are invisible to the rules (cross-type, or
  fuzzy below 0.75: `USA` / `United States of America`, `UK, Suffolk, Rendlesham Forest` /
  `United Kingdom, England, Rendlesham Forest`, `UK Ministry of Defence` / `United Kingdom
  Ministry of Defence`). The reranker scores 80 of those 100 at >=0.9 with claims, and
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
- Same-type pairs only (the rules' own ground): AUC rules 0.546, reranker 0.867.
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

As a RANKER of what the rules surface: it wins. Same-type AUC 0.867 against 0.546, and
on today's queue it moves the fuzzy rule's junk to the bottom and its under-scored true
duplicates to the top. Ship behind the flag; default off until the workbench shows
`rule_score` beside it so a reviewer can see both.

As a CANDIDATE SOURCE: it cannot be one yet. Scoring is 0.14 s per pair on the GPU;
the graph has 3,200 live nodes and five million pairs. It finds 80 of the 100 duplicates
the rules miss only when something puts them in front of it - a blocking or embedding
shortlist, which does not exist in the code today (the `--verify` pass the module
docstring describes was never built).

The Claude verify pass stays for cross-type pairs. The reranker's cross-type scores
rank the queue; they do not decide.

Not fixed by this: the rules' pass itself costs 25 minutes (O(n^2) Levenshtein), which
dwarfs the reranker's 1.5 minutes; and the Atlant / Atlantis error shows the model
shares the rules' blind spot on near-identical names of distinct people.
