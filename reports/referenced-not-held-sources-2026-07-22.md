# Referenced-but-not-held sources - ranked acquisition candidates

Extracted from the assimilator knowledge graph (`knowledge.db`) on 2026-07-22.
Source-acquisition seed list: sources our own corpus cites but does not yet hold.

## Method and caveats

- **Scope.** All 178 `document`-type nodes, minus the 16 confirmed present in the 23 held ingest records (matched by title/identifier, verified by hand; includes FOIA 18-F-0324, which cites itself). 145 not-held targets remain; the top 40 by citation are below.
- **Rank basis: RAW distinct-claim citation count** (how many claims in the graph reference the source), not a weighted/credibility score. Higher = more load-bearing on what we already publish, so acquiring it deepens the most corroboration.
- **Consolidation.** Duplicate/fragmented nodes for one acquirable source are merged into a single row (counts are DISTINCT claims across the cluster, never summed - a claim citing two duplicate nodes counts once). Merged rows list every component `node_id` and flag `n_nodes`.
- **Provenance-chain sources: NONE available.** The claim provenance chain (`origin_kind`/`origin`/`relay`, ADR 0044) is null across all 5218 claims (no re-digest yet), so the only referenced-source signal is `document` nodes. When chains land, anonymous/relayed origins become a second acquisition signal - not extractable today.
- **Identifiers are mostly absent from the graph.** Document nodes rarely carry a URL/DOI/ISBN (19/178 have any metadata, usually just a date). Where a locator is known it is in the note; otherwise acquisition needs a lookup. This is a graph-data gap worth closing at digest time.
- **Type is heuristic** (keyword-classified from the name); it drives the copyright tier, so treat it as a starting hint, not a ruling.

## Top 40 acquisition candidates

| # | Cites | Source | Type | Nodes | Cited by (held records) | Note |
|--:|------:|--------|------|------:|-------------------------|------|
| 1 | 85 | Pentagon/Navy UAP videos - the three released clips (FLIR1/'Tic-Tac', Gimbal, Go Fast) | video (US-gov release) | 11 | 2023-07-26-pdf-unidentified-anomalous-phenomena-implications-on-national-security-publi... | 11 fragmented nodes for one DoD release; three distinct clips but one acquisition. Officially released by DoD 2020-04-27 (the release statement IS held). Public/US-gov work. |
| 2 | 39 | Wilson Davis Memo | gov document / correspondence | 1 | 2023-11-17-ebook-in-plain-sight; 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-f... |  |
| 3 | 25 | Luis Elizondo AATIP resignation letter(s) (October 2017) | letter / gov document | 6 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos; 2022-08-17-pdf-foia-resp... | 6 nodes; possibly two letters (10-03 and 10-04). LIKELY ALREADY HELD inside the ingested FOIA 18-F-0324 record - verify before intake. |
| 4 | 17 | Condon Report | gov/official report | 1 | 2023-11-17-ebook-in-plain-sight; 2024-02-01-pdf-report-on-the-historical-record-of-us-g... |  |
| 5 | 16 | ODNI Preliminary Assessment: Unidentified Aerial Phenomena (June 2021, 180-day report) | gov report (ODNI) | 3 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... | 3 nodes for one ODNI report. Public. Identifier: ODNI 'Preliminary Assessment: UAP', 25 June 2021. |
| 6 | 13 | National Defense Authorization Act for Fiscal Year 2023 | gov/official report | 1 | 2023-06-05-web-intelligence-officials-say-u-s-has-retrieved-craft-of-non; 2023-06-09-we... |  |
| 7 | 13 | Talmud of Jmmanuel (TJ) | contactee / religious material | 1 | 2026-06-12-video-revealing-the-nordic-alien-prophecies-michael-horn.v2 |  |
| 8 | 12 | Billy Meier contact reports | contactee / religious material | 1 | 2026-06-12-video-revealing-the-nordic-alien-prophecies-michael-horn.v2 |  |
| 9 | 11 | Unidentified: Inside America's UFO Investigation | film/TV | 1 | 2019-06-01-web-the-media-loves-this-ufo-expert-who-says-he-worked-for-an; 2023-11-17-eb... |  |
| 10 | 10 | Harry Reid memo to William Lynn III on Advanced Aerospace Threat Identification Program (AATIP) Special Access Program status (2009-07-24) | gov document / correspondence | 1 | 2019-06-14-web-pentagon-reinforces-mr-luis-elizondo-had-no |  |
| 11 | 9 | Grusch Intelligence Community Inspector General (ICIG) whistleblower complaint (2022) | gov document / correspondence | 1 | 2023-06-05-web-intelligence-officials-say-u-s-has-retrieved-craft-of-non |  |
| 12 | 8 | Book of Enoch | contactee / religious material | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 13 | 6 | Estimate of the Situation (1948) | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 14 | 6 | Predator Drone Nuclear Facility UAP Video | video | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 15 | 5 | DoD Form 1910 | document (unclassified type) | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 16 | 5 | Billy Meier photograph analysis report | contactee / religious material | 1 | 2026-06-12-video-revealing-the-nordic-alien-prophecies-michael-horn.v2 |  |
| 17 | 4 | Walter Haut Roswell Affidavit 2002 | gov document / correspondence | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 18 | 4 | Art's Parts Letters | gov document / correspondence | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 19 | 4 | Executive Order 12333 | gov document / correspondence | 1 | 2024-01-24-pdf-unclassified-summary-of-report-no-dodig-2023-109-evaluation-of-the-dods-... |  |
| 20 | 4 | The Roswell Report: Case Closed (1997) | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 21 | 4 | Angst in the Shadows (Erik Nanstiel book) | book | 1 | 2026-04-24-video-skinny-bob-is-real-lifelong-abductee-reveals-everything.v2 |  |
| 22 | 3 | Close Encounters of the Third Kind (1977 film) | film/TV | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 23 | 3 | Greenewald Freedom of Information Act (FOIA) Request 18-F-0324 (2017-12-17) | FOIA document | 1 | 2022-08-17-pdf-foia-response-18-f-0324-aatip-and-luis-elizondo-documents |  |
| 24 | 3 | To the Stars Academy YouTube Video Featuring Luis Elizondo (2017) | video | 1 | 2022-08-17-pdf-foia-response-18-f-0324-aatip-and-luis-elizondo-documents |  |
| 25 | 3 | National Defense Authorization Act for Fiscal Year 2022 | gov/official report | 1 | 2023-07-26-pdf-unidentified-anomalous-phenomena-implications-on-national-security-publi... |  |
| 26 | 3 | Wilbert Smith Canada Memo 1950 | gov document / correspondence | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 27 | 3 | Nathan Twining Flying Disc Letter 1947 | gov document / correspondence | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 28 | 3 | Skinwalker Ranch Sherman Family Reports | gov/official report | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 29 | 3 | Advanced Aerospace Threat Identification Program (AATIP) DoD Threat Scenario Slides | document (unclassified type) | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 30 | 3 | Skinwalkers at the Pentagon Book | book | 1 | 2023-11-17-ebook-in-plain-sight |  |
| 31 | 3 | The Durant Report | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 32 | 3 | National Academy of Sciences Assessment of the Condon Report | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 33 | 3 | Government Accountability Office (GAO) Roswell Report (1995) | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 34 | 3 | 2022 Annual Report on Unidentified Aerial Phenomena | gov/official report | 1 | 2024-02-01-pdf-report-on-the-historical-record-of-us-government-involvement-with-uniden... |  |
| 35 | 3 | USAF Teleportation Study | document (unclassified type) | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 36 | 3 | Star Trek (TV show) | film/TV | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 37 | 3 | Unidentified (TV show) | film/TV | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 38 | 3 | Elizondo Tipton Advanced Aerospace Threat Identification Program (AATIP) Transfer Email | document (unclassified type) | 1 | 2024-08-19-ebook-imminent-inside-the-pentagon-s-hunt-for-ufos |  |
| 39 | 3 | David Grusch 2023 Congressional testimony on Unidentified Anomalous Phenomena (UAP) | hearing/testimony | 1 | 2025-03-27-web-rep-burlison-welcomes-former-u-s-air-force-officer-david |  |
| 40 | 3 | A Blood Covenant: The Alien Agenda Behind Civilization and Human Domestication (Erik Nanstiel book) | book | 1 | 2026-04-24-video-skinny-bob-is-real-lifelong-abductee-reveals-everything.v2 |  |

## Node IDs (for the screener - full list per row)

1. **Pentagon/Navy UAP videos - the three released clips (FLIR1/'Tic-Tac', Gimbal, Go Fast)** (85c) - `e57005b4-b10f-49a9-872f-61f00312b820`, `200409ad-6a43-46d3-84db-b15d1f77d8a2`, `1451172e-3efb-4848-939a-ed9aff0d7e2f`, `58de5e9b-3785-4435-bb30-121625fb3de8`, `80e17ddb-34b7-4aca-b27c-84322e5d9594`, `2c3d3bb2-318e-4adf-a878-ac335d01b0ab`, `f940df9d-abcb-4d23-9bb6-1569bf35858e`, `5590509e-fbbd-4d6c-8eaf-c72ba7ff1c88`, `0b4482cc-a4c0-427d-826e-6644cbec25bf`, `920652f6-063c-4972-acd8-5a880e81f303`, `63e84a73-2265-44ea-8b43-20d59335c31c`
2. **Wilson Davis Memo** (39c) - `be1c5291-2a16-4e69-b6ae-6e8761ff5e7c`
3. **Luis Elizondo AATIP resignation letter(s) (October 2017)** (25c) - `b869bdd4-0ef8-497d-a446-077271010158`, `b55265b8-c698-458f-97ca-8895c60e1068`, `3272f7a3-e93a-4ce1-8158-039af3b1408f`, `a4f4460d-d553-49a5-8a5e-6e8b5c928153`, `e9fca8b0-6cd0-42ed-88d3-f255863c0615`, `4cdd0188-aa4b-4eee-ab31-0a5c4ce748ef`
4. **Condon Report** (17c) - `4ca72f01-5794-474d-8fa6-7c9e04a4ca57`
5. **ODNI Preliminary Assessment: Unidentified Aerial Phenomena (June 2021, 180-day report)** (16c) - `13566718-151d-4f6b-96c6-b36797f258d8`, `10db20bd-6632-4d61-85fd-35ad84e808c1`, `d576fa79-09b9-4992-9713-aea50b8e5fbe`
6. **National Defense Authorization Act for Fiscal Year 2023** (13c) - `25a71281-35a6-4636-ada3-b0d727672634`
7. **Talmud of Jmmanuel (TJ)** (13c) - `16df0ee4-4420-42db-9df1-22255e96d10a`
8. **Billy Meier contact reports** (12c) - `f492cde3-ecc0-47a1-ae3b-304e5c41abb0`
9. **Unidentified: Inside America's UFO Investigation** (11c) - `2658cc67-2b0a-4b7f-a324-f1fa16dc21b7`
10. **Harry Reid memo to William Lynn III on Advanced Aerospace Threat Identification Program (AATIP) Special Access Program status (2009-07-24)** (10c) - `cec715ac-05d9-4eb4-b97a-dcbffb6bf2e5`
11. **Grusch Intelligence Community Inspector General (ICIG) whistleblower complaint (2022)** (9c) - `47d37c7a-8dec-421f-a882-43a2fd8ab2ea`
12. **Book of Enoch** (8c) - `2b9d89f6-26e3-4780-bdbd-ca2b8c0f54e7`
13. **Estimate of the Situation (1948)** (6c) - `d604549a-40b2-441c-9756-9b27ffdc6925`
14. **Predator Drone Nuclear Facility UAP Video** (6c) - `a44c9320-48e2-4dfc-80ee-c063e5537cbc`
15. **DoD Form 1910** (5c) - `c23e5eb6-7f91-4825-bdb3-42be1799c79f`
16. **Billy Meier photograph analysis report** (5c) - `173020cd-5504-474d-ada4-138e0ccd9c50`
17. **Walter Haut Roswell Affidavit 2002** (4c) - `6cee5441-deeb-49fd-bdd0-e809d42cbf8f`
18. **Art's Parts Letters** (4c) - `8b561b1e-a09b-4080-aca6-63f9c8fceab5`
19. **Executive Order 12333** (4c) - `235519a8-7f3c-4bc1-ad1f-8d40d6ae5b80`
20. **The Roswell Report: Case Closed (1997)** (4c) - `cc2c952a-56f9-401b-84cf-f26ebf0323a0`
21. **Angst in the Shadows (Erik Nanstiel book)** (4c) - `63fc3715-418f-4bef-8a15-d5de2903039e`
22. **Close Encounters of the Third Kind (1977 film)** (3c) - `2149bfba-36b7-42d7-a63e-a7539948da7a`
23. **Greenewald Freedom of Information Act (FOIA) Request 18-F-0324 (2017-12-17)** (3c) - `968e3caa-0412-4478-ad07-ff53ddb20ce1`
24. **To the Stars Academy YouTube Video Featuring Luis Elizondo (2017)** (3c) - `3206b808-b54d-45e2-9736-b404e7ee65bd`
25. **National Defense Authorization Act for Fiscal Year 2022** (3c) - `bec04557-78dd-4b40-9fd6-ef824b3281b8`
26. **Wilbert Smith Canada Memo 1950** (3c) - `5157ce25-fb14-4935-93fe-321d3231beb9`
27. **Nathan Twining Flying Disc Letter 1947** (3c) - `9c66526f-9dfb-4778-b843-142b402cdc6c`
28. **Skinwalker Ranch Sherman Family Reports** (3c) - `d1ede69e-36e1-4f9a-9c98-691a969f7b0b`
29. **Advanced Aerospace Threat Identification Program (AATIP) DoD Threat Scenario Slides** (3c) - `f844809d-392d-4cbd-8883-1ecb9a701ed7`
30. **Skinwalkers at the Pentagon Book** (3c) - `29054aa0-0c36-4c06-82c4-0a3112b163d6`
31. **The Durant Report** (3c) - `0f5987a4-fef4-4129-b22a-9d1f6aead1b8`
32. **National Academy of Sciences Assessment of the Condon Report** (3c) - `a5f5bcb2-d57b-4096-8875-8949c77a05f6`
33. **Government Accountability Office (GAO) Roswell Report (1995)** (3c) - `480b089d-f501-4a7b-8351-e010fe17a50a`
34. **2022 Annual Report on Unidentified Aerial Phenomena** (3c) - `e1f95e58-faec-4498-b3e0-468529965e45`
35. **USAF Teleportation Study** (3c) - `09ab7321-9a38-41ab-8fb9-02b95d94da7a`
36. **Star Trek (TV show)** (3c) - `a7e3e553-6185-45fc-b062-f17cd5ecd3ab`
37. **Unidentified (TV show)** (3c) - `ef0c0f36-c006-4924-a8fe-2f0f7757dcb8`
38. **Elizondo Tipton Advanced Aerospace Threat Identification Program (AATIP) Transfer Email** (3c) - `af81b08b-bcc8-42a1-bb4e-9f9662d9768b`
39. **David Grusch 2023 Congressional testimony on Unidentified Anomalous Phenomena (UAP)** (3c) - `b4b91a18-1e00-434d-a88a-d6bd58950248`
40. **A Blood Covenant: The Alien Agenda Behind Civilization and Human Domestication (Erik Nanstiel book)** (3c) - `f39dd936-9002-43a9-a844-45387b412e00`
