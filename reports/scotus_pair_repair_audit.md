# SCOTUS Pair Repair Audit

Phase 3.5D-E output audit for sectioned, reasoning-filtered SCOTUS chunks.

## Pair Counts

| Pair/Variant | Train | Dev | Test | Total | Same-case |
| --- | --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | 1747 | 122 | 74 | 1943 | 14 |
| Scalia_vs_Ginsburg/raw_clean | 1747 | 122 | 74 | 1943 | 14 |
| Thomas_vs_Souter/masked | 1272 | 92 | 128 | 1492 | 16 |
| Thomas_vs_Souter/raw_clean | 1272 | 92 | 128 | 1492 | 16 |

## Eligible Chunk Counts

| Justice | Eligible raw chunks | Excluded block/chunk records |
| --- | --- | --- |
| Ginsburg | 4098 | 1759 |
| Scalia | 4919 | 1819 |
| Souter | 4289 | 1636 |
| Thomas | 4043 | 1707 |

## Posture Mix

| Justice | Section Posture | Eligible raw chunks |
| --- | --- | --- |
| Ginsburg | majority | 3211 |
| Ginsburg | dissent | 612 |
| Ginsburg | concurrence_in_judgment | 102 |
| Ginsburg | concurrence_in_part_dissent_in_part | 89 |
| Ginsburg | concurrence | 59 |
| Ginsburg | concurrence_in_part | 25 |
| Scalia | majority | 3749 |
| Scalia | dissent | 718 |
| Scalia | concurrence_in_judgment | 219 |
| Scalia | concurrence_in_part | 112 |
| Scalia | concurrence | 74 |
| Scalia | concurrence_in_part_dissent_in_part | 47 |
| Souter | majority | 3353 |
| Souter | dissent | 699 |
| Souter | concurrence_in_part_dissent_in_part | 95 |
| Souter | concurrence | 66 |
| Souter | concurrence_in_part | 39 |
| Souter | concurrence_in_judgment | 37 |
| Thomas | majority | 2697 |
| Thomas | dissent | 980 |
| Thomas | concurrence | 114 |
| Thomas | concurrence_in_part_dissent_in_part | 102 |
| Thomas | concurrence_in_judgment | 99 |
| Thomas | concurrence_in_part | 51 |

## Baseline Gate

| Pair | Decision | Best masked test model | Best masked balanced accuracy | Metadata-only | Length/citation-only |
| --- | --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg | activation-ready | word_char_tfidf_logreg | 0.764 | 0.500 | 0.574 |
| Thomas_vs_Souter | activation-ready | word_tfidf_logreg | 0.809 | 0.500 | 0.449 |

## Matching Diagnostics

| Pair | Matching Key | Masked pairs |
| --- | --- | --- |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|middle|train | 83 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|middle|train | 83 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|middle|train | 80 |
| Thomas_vs_Souter | Criminal Procedure|majority|1990s|1|middle|train | 67 |
| Thomas_vs_Souter | Economic Activity|majority|2000s|2|middle|train | 66 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|2|middle|train | 65 |
| Scalia_vs_Ginsburg | Economic Activity|majority|2000s|2|middle|train | 50 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|2|middle|train | 50 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|late|train | 46 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|late|train | 46 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|early|train | 45 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|early|train | 45 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|early|train | 45 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|late|train | 45 |
| Thomas_vs_Souter | Civil Rights|majority|1990s|1|middle|train | 44 |
| Scalia_vs_Ginsburg | Economic Activity|majority|1990s|2|middle|train | 42 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|2|middle|train | 42 |
| Scalia_vs_Ginsburg | Civil Rights|majority|1990s|2|middle|train | 40 |
| Thomas_vs_Souter | Criminal Procedure|majority|1990s|1|late|train | 39 |
| Thomas_vs_Souter | Economic Activity|majority|2000s|1|middle|train | 37 |

## Same-Case Examples

| Pair | Split | Case ID | Issue Area | Posture | Position |
| --- | --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | early |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | early |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | early |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | early |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | middle |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | late |
| Scalia_vs_Ginsburg | train | 137006 | Economic Activity | concurrence_in_part | late |

## Decision

- Scalia vs. Ginsburg clears the repaired-corpus activation gate.
- Thomas vs. Souter also clears the numeric masked held-out gate in this run, but Scalia vs. Ginsburg should remain the first Phase 4 target because it was the planned repair target and has a steadier dev/test profile.
- Metadata-only remains at chance, so the v2 matching metadata is not by itself separating the labels.
