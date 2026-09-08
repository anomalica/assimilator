# The account layer, composed pages, and pages that should not exist

Measured 2026-09-08 against the live graph: 12,173 live nodes, 38,863 claims,
150 records, 326 nodes passing the page gate, 57 corroborations.

## 1. Where an account belongs

An ACCOUNT is one story told once in one source - the unit the graph currently
has no room for, so an interview describing three abductions yields correct
claims with nothing recording which story each belongs to.

**It is a span, not a node.** Every existing node type names something that
exists in the world and can be spoken of from any source; an account exists
only inside one record, bounded by a stretch of it. Making it a node type
repeats the mistake `matter` was deleted for: a type that folds four ways
because it has no test for identity. An account's identity is (record, span),
which is not an identity any node type can carry. So: its own table, keyed on
the record and the span, with claims joined to it by their own span.

**It does not merge across records.** Two accounts that look like the same
occurrence are evidence of that occurrence, not the same object: the account is
what a source says, the event node is the conclusion drawn from several. Merging
accounts would destroy the very thing that makes them useful - that there were
two tellings. When two accounts describe one occurrence, that is a proposal on
the EVENT, and the reviewer confirms it there.

**The data is already there.** 38,858 of 38,863 claims carry
`location_in_record`, and the spans are structured (`ch5:180-443` for text,
timecodes for audio). Segmenting a record by section and by gaps between claim
spans produces plausible units without any new extraction: on the four largest
records, 3,654 claims fall into 73 segments, 2,982 into 86, 2,612 into 53,
1,853 into 44 - about 40-50 claims each. That is demonstrably computable and
demonstrably too coarse to be "one story" yet; the segmentation needs either
tuning against a labelled record or a boundary signal from extraction. What it
proves is that the layer needs no new plumbing to carry spans, only a rule for
where one account ends.

**What it unblocks: corroboration.** The corpus holds 57 corroborations against
38,863 claims because corroboration currently asks "do two claims say the same
thing", which is a text question. The useful question is "did two sources
independently describe the same occurrence", and that becomes computable the
moment accounts exist. The headroom is measurable today: 106 of 1,781 live
event and project nodes are attested by two or more records - Roswell by 20,
Blue Book by 23, AATIP by 27, Nimitz by 13. Those are 106 occurrences where
independence is a structural fact rather than a similarity score. This is the
strongest argument for the layer.

## 2. Do composed pages work

Yes, measured on the only case. UFOs / UAPs replaced two pages:

| | prose words | claims cited | destinations |
|---|---|---|---|
| UFO page | 292 | 13 | 2 |
| UAP page | 854 | 32 | |
| UFOs / UAPs | 1,531 | 59 | 1 |

The composed page is 34% longer than both predecessors together and cites 31%
more distinct claims, from one address rather than two, and it is current
against its brief (built_from matches the brief hash exactly). The claim union
deduped the 26 claims the two nodes shared, so no count is inflated.

The limit is not composition. The page cites 59 of the 2,068 claims available
to it - under 3% - and that ratio was the same before composing. More material
in the brief does not become more page; what the writer does with the brief is
the binding constraint.

**Other candidates are thin, and the reranker must not pick them.** Only four
topic pairs are both page-proposed and score above 0.85 on both reranker
measures. One is genuine (Astral projection with Out-of-Body Experience, 10
shared claims). One is plainly wrong (general relativity with Warp drive, 0
shared claims) - the same vocabulary, different subjects. So composition
candidates need shared claims as well as name similarity, and a human decides;
a score threshold alone would compose a physics theory with a propulsion idea.

## 3. Pages that should not exist

Three signals were tested for "this entity carries no anomalous content of its
own", against a labelled set (good: Roswell, Area 51, Skinwalker Ranch,
Wright-Patterson, S4, Socorro, Fatima, the Moon; bad: Sydney, Las Vegas, San
Diego, San Antonio, Manhattan, Melbourne, Rome, Paris, London, Los Angeles,
South Carolina, Cuba):

- **Subject mentions** (the gate's existing test, extended to all types): kills
  74 proposals including Rendlesham, Roswell and Out-of-Body Experience. The
  test works for people, whose name recurs in claims about them, and not for
  places or topics.
- **Solo claims** (claims referencing this node and no other): Roswell 4,
  Sydney 0 - but Time travel 0 as well. Does not separate.
- **Concentration** (share of claims from the dominant record or event):
  Sydney 0.66, Roswell 0.39. Separates in the wrong direction.
- **Claims not tied to an event** ("does the place carry anything of its own"):
  San Francisco 90%, Roswell 49%. Separates in the wrong direction.
- **Container** (a bare name that heads other place names): catches France (39)
  and Cuba (4) correctly, and Mars (4) wrongly - a celestial body is both a
  container and a subject.

None of the three works, and the reason is structural: the graph records that a
claim REFERENCES a node, never the role the node plays in it. Sydney is a
setting, Roswell is a subject, and the edge is identical. The `claim_role`
column exists with a CHECK constraint and zero populated rows across all
38,863 claims; the digest format carries no role either - a claim's `refs` are
a flat list of names. So the information does not exist upstream to be lost.

The honest conclusion: **this cannot be fixed as a page-gate rule today**, and a
sweep would be forty corrections that do not stop the forty-first. It is the
same missing distinction as the account layer - subject versus setting - and it
should be answered there: a place attached to an account as its LOCATION is not
thereby page-worthy, while an account whose subject is that place is. Until
then the gate should not be widened on any of the three signals above, because
each one deletes good pages to reach the bad ones.

## 4. What the place sweep did find

The place convention is largest-unit-first, so a bare name that repeats the last
component of a compound is the same site written twice. Live on 2026-09-08: 45
such pairs - Area 51, Pine Gap, RAF Bentwaters, Woomera, Maralinga, Skinwalker
Ranch. Most bare copies hold no claims and are harmless leftovers, but Westall
High School splits 5 claims from 14, and any of them can earn a second page for
one place. This is now reported by the consistency check as
`one-place-two-nodes`; it proposes nothing, because a merge is Mark's to confirm.

## What to take from this

The three strands are one question. An account layer would give the graph the
distinction it lacks everywhere else: what a claim is ABOUT, as opposed to what
it mentions. Corroboration wants it (106 occurrences have two or more sources
and only 57 corroborations exist). The page set wants it (five signals tested,
none separates a subject from a setting). Composition wants it (the reranker
scores name similarity and cannot tell a shared subject from a shared
vocabulary). Building any of the three as a rule over today's edges means
guessing at the missing field, and each guess deletes good pages to reach bad
ones.
