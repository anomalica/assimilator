# Person names to natural order - measured impact before running the corpus pass

Migration of stored person names from surname-first ("Fravor, David") to natural
order ("David Fravor"), per the 2026-06-29 amendment in
`anomalica/architecture/node-types.md`. Deterministic text pass, no AI, no
re-digestion. Measured by rebuilding the graph twice from identical digests -
once as-is, once through the pass - and diffing the two databases.

Command: `assimilator naturalise-person-names <digests-dir>` (`--dry-run` to
report without writing).

## Scale

| | before | after |
|---|---|---|
| person nodes | 594 | 588 |
| total nodes | 2089 | 2083 |
| claims | 5218 | 5218 |
| records with a producer link | 2 / 23 | 10 / 23 |
| alias rows | 146 | 715 |

783 person node names rewritten across 20 of the 23 digests, plus every `refs`
entry, every `speaker` field, and `record.producer`. Places are untouched.

## The finding that matters: 30 claims were filed under the wrong person

Before the migration the graph held one node for **Harry Reid** (the senator)
with **"Reid, Garry" as an alias on it**. Garry Reid is the Pentagon official who
ran the Elizondo inquiry - a different person. 30 claims went to the senator's
node, including:

- "The Department of Defense Inspector General concluded that Garry Reid violated
  Joint Ethics Regulations by creating a sexualised work environment."
- "The Department of Defense Inspector General investigated Garry Reid on
  allegations including maintaining a sexual relationship with a subordinate."
- "Garry Reid seized Luis Elizondo's computers and files from his office."

Harry Reid died in 2021. Those claims on his page are defamatory of a real named
person, and the same failure would repeat on any future forename-substitution
pair. After the migration the two are distinct nodes: Harry Reid 64 claim
references, Garry Reid 31.

Mechanism, exactly. Under surname-first the matcher compares comma components and
merges when the weakest clears 0.80. `levenshtein("garry", "harry") = 0.800` -
it lands precisely on the threshold and merges. Under natural order the pair goes
down the plain-name path, where the substituted forename is a mutual orphan token
and the pair is rejected outright. The comma structure was not protecting
precision here; it was the thing that broke it.

## Merges gained and lost

Five additional correct merges, no incorrect ones:

- Grusch, David -> David Charles Grusch
- Quintanilla, Hector Jr. -> Hector Quintanilla
- Greenewald, John Jr. -> John Greenewald
- Elizondo, Luis D. -> Luis Elizondo
- Elizondo, Luis (Lou) -> Luis Elizondo

"Luis D. Elizondo III (father)" correctly stays a separate node from his son, and
"Bill Lynn" / "William Lynn III" stay separate in both runs (a genuine missed
merge, unchanged by this work).

One merge was lost and then recovered by a separate one-line fix: "Andre Almond"
and "André Almond" merged under the comma path (0.800, again exactly at the
threshold) but not under natural order, where the accent makes each forename an
orphan of the other. `name_equivalence_key` now folds diacritics off ASCII-Latin
bases only, so the pair merges by equivalence rather than by luck. The fold is
deliberately not a blanket NFD strip: that would turn ガ into カ and merge
distinct non-Latin names. Applied to the pre-migration digests the fold changes
nothing, so it is safe to land independently.

Net across the corpus: 6 fewer person nodes, zero entities split that should not
have been, one false merge eliminated.

## Page staleness: 142 claim hashes change, not zero

`claim_hash` excludes names by design - it takes resolved graph ids - so a rename
alone stales nothing. But the rename changes how the importer RESOLVES names to
ids, and those ids are in the hash. 142 of 5218 claims (2.7%) change hash. Every
one is accounted for by a changed node reference or speaker id (91 refs only, 1
speaker only, 50 both, 0 unexplained), which is the Reid split and the five
merges landing.

Affected records: FOIA Response 18-F-0324 (40), Imminent (36), In Plain Sight
(22), Burlison/Grusch release (15), "Claims that UFO information was
inappropriately withheld" (15), and four others in single digits.

`claim_fingerprint` is completely stable - all 5218 fingerprints match across the
two rebuilds - so every human adjudication verdict in the workbench carries
forward untouched. That was the thing worth protecting, and it is intact.

## Where the surname went

`metadata.family_name` on each person node. The comma was the only place the
surname was recorded, and dropping it without a replacement would break surname
sort and every non-Anglo name ("last token" is not the family name in "Mohammed
bin Rashid Al Maktoum"). Consumers read the field:

- The vault export emits `sort_name: Fravor, David` in each person note's front
  matter and keeps the filename natural, so wikilinks still resolve.
- The assembler's `_display_name` comma flip should be deleted, not converted to
  a field read - the stored name is already what it should render.
- `anomalica_common.slug` must NOT change. Its comma reorder still serves places
  ("USA, Nevada, Area 51" -> `nevada-area-51-usa`); removing it would break every
  deployed place URL. Person slugs are identical either way (`david-fravor`), so
  the migration does not move a single URL.
- `matching.py` keeps `STRUCTURED_COMPONENT_THRESHOLD` as-is. It is a PLACE rule
  now; converting it to a family_name read would be a regression, since the
  measurement above shows the comma path was the source of the one false merge.

The pre-migration form is kept in `metadata.aliases` (`["Fravor, David"]`) and
written to the graph's alias table on import, so it survives a rebuild and
last-first input still resolves. That is what takes alias rows from 146 to 715.

## Follow-on, in order

1. The digester must emit natural order AND `metadata.family_name` - after this
   pass there is no comma left to parse a surname out of.
2. The assembler drops `_display_name` / `_rewrite_link_display`.
3. `digests/variants/` is deliberately untouched: those are snapshots of what
   each model emitted, and rewriting them would falsify the model comparison.

## The diff, one digest

`digests/2023-07-26-pdf-david-fravor-statement-for-the-house-oversight-committee.yaml`
- 11 person nodes, 626 diff lines, of which every removed line is a `name:` line.
Claim `quote:` and `text:` are untouched; the organisation and place refs beside
the renamed person refs are untouched.

```diff
 nodes:
   - id: 44504498-ab5f-4ba5-97fa-25086c6eb235
     type: person
-    name: Fravor, David
+    name: David Fravor
+    metadata:
+      family_name: Fravor
+      aliases:
+        - Fravor, David
   - id: 51d882d7-ae4b-4a91-bc4f-a2b900fb1196
     type: person
-    name: Stratton, Jay
+    name: Jay Stratton
+    metadata:
+      family_name: Stratton
+      aliases:
+        - Stratton, Jay
   - id: dd609ecd-bd80-46f2-b0b5-c408e69c2d37
     type: organisation
     name: United States Navy
```

```diff
     attestation: first_hand
     speaker:
       id: 44504498-ab5f-4ba5-97fa-25086c6eb235
-      name: Fravor, David
+      name: David Fravor
     location: page 1, paragraph 2
     refs:
       - id: 44504498-ab5f-4ba5-97fa-25086c6eb235
-        name: Fravor, David
+        name: David Fravor
       - id: dd609ecd-bd80-46f2-b0b5-c408e69c2d37
         name: United States Navy
     quote: My name is David Fravor and I am a retired Commander in the U.S Navy.
     text: David Fravor is a retired Commander in the United States Navy.
```

A node that already has metadata gets the two fields added, not replaced
(`2024-11-13-pdf-written-testimony-of-luis-elizondo...`):

```diff
   - id: c263e1f2-216b-44dc-91cb-90d5dafb9bc9
     type: person
-    name: Mace, Nancy
+    name: Nancy Mace
     metadata:
+      family_name: Mace
+      aliases:
+        - Mace, Nancy
       role: Chairwoman
```

And the record producer moves with the name, which is what takes producer links
from 2/23 to 10/23:

```diff
 record:
   id: ...
   title: 'Imminent: Inside the Pentagon''s Hunt for UFOs'
-  producer: Elizondo, Luis
+  producer: Luis Elizondo
```
