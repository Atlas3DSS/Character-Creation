# SCOTUS Slice Split Feasibility Audit

## Purpose

This audits whether a cached justice-style SCOTUS probe run can be resplit while preserving strict case-connected holdout and issue-family coverage.

## Inputs

- Run directory: `sweep_v4/scotus_slice_bf16_economic_style_normal_20260501_023619`
- Rows: `778`
- Case-connected components: `8`

## Feasibility Read

| Issue | Components | Rows | Pairs | Cases | Strict train/dev/test feasible | Blocking reason |
| --- | --- | --- | --- | --- | --- | --- |
| Economic Activity | 8 | 778 | 389 | 40 | yes |  |

Strict issue-stratified train/dev/test resplitting is feasible for every issue.

## Component Table

| Component | Issue | Rows | Pairs | Cases | Label counts | Original splits | Case IDs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| component_005 | Economic Activity | 194 | 97 | 7 | {'0': 97, '1': 97} | {'train': 194} | 118410,118457,118486,122257,130141,131157... |
| component_004 | Economic Activity | 140 | 70 | 8 | {'0': 70, '1': 70} | {'train': 140} | 112569,112694,112867,117856,117880,117888... |
| component_002 | Economic Activity | 130 | 65 | 7 | {'0': 65, '1': 65} | {'train': 130} | 1087670,112769,117844,117859,117989,118143... |
| component_006 | Economic Activity | 118 | 59 | 8 | {'0': 59, '1': 59} | {'train': 118} | 118465,121152,127909,131153,131156,145683... |
| component_000 | Economic Activity | 88 | 44 | 4 | {'0': 44, '1': 44} | {'dev': 88} | 118471,122254,136985,145709 |
| component_003 | Economic Activity | 50 | 25 | 3 | {'0': 25, '1': 25} | {'train': 50} | 112509,112598,112915 |
| component_001 | Economic Activity | 30 | 15 | 2 | {'0': 15, '1': 15} | {'test': 30} | 112641,118253 |
| component_007 | Economic Activity | 28 | 14 | 1 | {'0': 14, '1': 14} | {'train': 28} | 137006 |

## Original Split Counts

| Issue | Split | Justice | Rows |
| --- | --- | --- | --- |
| Economic Activity | dev | Ginsburg | 44 |
| Economic Activity | dev | Scalia | 44 |
| Economic Activity | test | Ginsburg | 15 |
| Economic Activity | test | Scalia | 15 |
| Economic Activity | train | Ginsburg | 330 |
| Economic Activity | train | Scalia | 330 |

## Decision

This run can support a stricter case-connected issue-stratified resplit.
