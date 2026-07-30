# Where the corroboration cut actually is

Measured over the full corpus after embedding all 5218 claims and 2051 nodes in
the corrected (dequantised) vector space. Deterministic, no AI. Raw counts and
the top pairs are in `similarity-profile-2026-07-29.json`.

Method: for each claim, its 10 nearest neighbours; pairs from the SAME record
discarded (a claim's neighbours inside its own record are near-duplicates by
construction and would flatter every threshold). 14,855 distinct cross-record
pairs.

## The distribution

    max 0.9678   median 0.6409   min 0.3866      (over neighbour pairs, not all pairs)

    >= 0.95        2 pairs
    >= 0.90       22
    >= 0.85      134
    >= 0.83      276      <- consolidate's standing threshold
    >= 0.80      652
    >= 0.75    1,841
    >= 0.70    3,848
    >= 0.65    6,841
    >= 0.60   10,053
    >= 0.55   12,673
    >= 0.99        0      <- corroborate's former default

## Where precision breaks: 0.85

Sampled pairs from each band and read them. The bands are not equally good.

**0.85 and above - genuine corroboration.** Same fact, different records.

    0.968  "Chuck Schumer stated publicly that the American public has a right to learn..."
           "Chuck Schumer stated that the American public has a right to learn about..."
    0.921  "David Grusch worked for the National Geospatial-Intelligence Agency (NGA)."
           "David Grusch worked as a senior intelligence officer at the National Geospatial..."
    0.860  "On 2022-07-20, the United States Department of Defense established the All-domain..."
           "The All-domain Anomaly Resolution Office (AARO) was announced in July 2022 as..."

**0.80-0.85 - same event, different facts.** This is where it degrades, and the
failure is subtle: the pairs are obviously related, which is exactly why a
threshold set by eye lands here.

    0.829  "...as Fravor moved to intercept the Tic Tac near the turbulent water"
           "...as David Fravor's aircraft pulled to within about 800 feet"
    0.802  "David Fravor considers the 2004 Nimitz encounter probably the most credible..."
           "...as Fravor reached the 12 o'clock position descending"

Two claims about one encounter are not two attestations of one fact. Counting
them as corroboration inflates confidence in the encounter by re-counting its
narrative detail.

**0.75 and below - same topic only.** "Elizondo felt like the radar operator at
Pearl Harbor" against "Elizondo's findings were met with skepticism" scores
0.754 and shares no assertion at all.

**Recommended cut: 0.85**, giving 134 candidate pairs for the verification pass.
Not fitted - chosen because the sampled band above it is clean and the band below
it is not. Treat as provisional: this corpus is 23 records, half of it two books,
and every claim in it was extracted at minimum reasoning effort, so a re-digest
may move it.

## Two corrections to earlier figures

**consolidate's 0.83 is IN the tail, not above it.** An earlier reading, from a
120-claim sample, reported the corpus maximum as 0.746 and concluded 0.83 selected
nothing. The full corpus tops out at 0.9678 and 0.83 selects 276 pairs. The sample
missed the tail because genuine duplicate pairs are exactly the rare tail a small
random sample cannot reach. 0.83 is a defensible cut that happens to sit slightly
below where precision breaks; it was never calibrated, but it was not absurd.

**corroborate's 0.99 default selected zero, and that finding survives.** The
corpus maximum is 0.9678, so the default could not have returned a single pair.
A run at that default would have reported "no corroboration in this corpus"
rather than "this constant predates the vector space". `--threshold` is now
required with no default.

The two medians quoted in this work are not comparable and should not be read as
a change: 0.320 was over ALL pairs in a small sample, 0.6409 is over NEIGHBOUR
pairs (the 10 nearest per claim), which is the high tail by construction.

## What this does not measure

Independence. These pairs are same-fact-different-record, which is corroboration
only if the records are independent sources. Provenance chains are absent from
every claim in the current graph (they arrive with the scheduled re-digest), and
until they land, a wire story reprinted twice and two genuine witnesses are
indistinguishable here. Bank these pairs; do not score on them yet.
