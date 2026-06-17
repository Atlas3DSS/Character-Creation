# SCOTUS Pair Repair Audit

Phase 3.5D-E output audit for sectioned, reasoning-filtered SCOTUS chunks.

## Pair Counts

| Pair/Variant | Train | Dev | Test | Total | Same-case |
| --- | --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | 1608 | 119 | 69 | 1796 | 14 |
| Scalia_vs_Ginsburg/raw_clean | 1608 | 119 | 69 | 1796 | 14 |
| Thomas_vs_Souter/masked | 1233 | 74 | 128 | 1435 | 16 |
| Thomas_vs_Souter/raw_clean | 1233 | 74 | 128 | 1435 | 16 |

## Eligible Chunk Counts

| Justice | Eligible raw chunks | Excluded block/chunk records |
| --- | --- | --- |
| Ginsburg | 3956 | 1752 |
| Scalia | 4728 | 1914 |
| Souter | 4321 | 1672 |
| Thomas | 4006 | 1818 |

## Posture Mix

| Justice | Section Posture | Eligible raw chunks |
| --- | --- | --- |
| Ginsburg | majority | 3038 |
| Ginsburg | dissent | 586 |
| Ginsburg | concurrence_in_part_dissent_in_part | 126 |
| Ginsburg | concurrence_in_judgment | 100 |
| Ginsburg | concurrence | 58 |
| Ginsburg | concurrence_in_part | 25 |
| Ginsburg | judgment | 23 |
| Scalia | majority | 3473 |
| Scalia | dissent | 682 |
| Scalia | concurrence_in_judgment | 206 |
| Scalia | judgment | 163 |
| Scalia | concurrence_in_part | 104 |
| Scalia | concurrence | 53 |
| Scalia | concurrence_in_part_dissent_in_part | 47 |
| Souter | majority | 3272 |
| Souter | dissent | 678 |
| Souter | judgment | 143 |
| Souter | concurrence_in_part_dissent_in_part | 93 |
| Souter | concurrence | 64 |
| Souter | concurrence_in_part | 39 |
| Souter | concurrence_in_judgment | 32 |
| Thomas | majority | 2600 |
| Thomas | dissent | 928 |
| Thomas | judgment | 141 |
| Thomas | concurrence | 100 |
| Thomas | concurrence_in_part_dissent_in_part | 99 |
| Thomas | concurrence_in_judgment | 91 |
| Thomas | concurrence_in_part | 47 |

## Baseline Gate

| Pair | Decision | Best masked test model | Best masked balanced accuracy | Metadata-only | Length/citation-only |
| --- | --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg | activation-ready | word_char_tfidf_logreg | 0.775 | 0.500 | 0.594 |
| Thomas_vs_Souter | activation-ready | word_char_tfidf_logreg | 0.789 | 0.500 | 0.449 |

## Matching Diagnostics

| Pair | Matching Key | Masked pairs |
| --- | --- | --- |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|middle|train | 80 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|middle|train | 79 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|middle|train | 73 |
| Thomas_vs_Souter | Economic Activity|majority|2000s|2|middle|train | 66 |
| Thomas_vs_Souter | Criminal Procedure|majority|1990s|1|middle|train | 63 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|2|middle|train | 50 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|2|middle|train | 50 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|late|train | 46 |
| Scalia_vs_Ginsburg | Economic Activity|majority|2000s|2|middle|train | 46 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|1|early|train | 45 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|early|train | 43 |
| Thomas_vs_Souter | Economic Activity|majority|1990s|1|late|train | 43 |
| Scalia_vs_Ginsburg | Economic Activity|majority|1990s|2|middle|train | 42 |
| Scalia_vs_Ginsburg | Judicial Power|majority|1990s|2|middle|train | 42 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|late|train | 41 |
| Scalia_vs_Ginsburg | Criminal Procedure|majority|2000s|1|early|train | 40 |
| Thomas_vs_Souter | Civil Rights|majority|1990s|1|middle|train | 40 |
| Scalia_vs_Ginsburg | Civil Rights|majority|1990s|2|middle|train | 39 |
| Thomas_vs_Souter | Criminal Procedure|majority|1990s|1|late|train | 38 |
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
- Thomas vs. Souter also clears the numeric masked held-out gate in this run, but remains secondary until dev/test behavior stabilizes.
- Metadata-only remains at chance when its score is near 0.500, so the matching metadata is not by itself separating the labels.
