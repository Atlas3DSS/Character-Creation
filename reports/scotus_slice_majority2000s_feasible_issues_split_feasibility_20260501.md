# SCOTUS Slice Split Feasibility Audit

## Purpose

This audits whether a cached justice-style SCOTUS probe run can be resplit while preserving strict case-connected holdout and issue-family coverage.

## Inputs

- Run directory: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_20260501_033918`
- Rows: `1154`
- Case-connected components: `10`

## Feasibility Read

| Issue | Components | Rows | Pairs | Cases | Strict train/dev/test feasible | Blocking reason |
| --- | --- | --- | --- | --- | --- | --- |
| Criminal Procedure | 4 | 590 | 295 | 30 | yes |  |
| Economic Activity | 3 | 400 | 200 | 19 | yes |  |
| Judicial Power | 3 | 164 | 82 | 11 | yes |  |

Strict issue-stratified train/dev/test resplitting is feasible for every issue.

## Component Table

| Component | Issue | Rows | Pairs | Cases | Label counts | Original splits | Case IDs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| component_009 | Criminal Procedure | 308 | 154 | 16 | {'0': 154, '1': 154} | {'train': 308} | 118390,134725,137733,137735,145641,145646... |
| component_004 | Criminal Procedure | 210 | 105 | 10 | {'0': 105, '1': 105} | {'train': 210} | 118389,118412,118436,118443,121165,122265... |
| component_001 | Criminal Procedure | 38 | 19 | 2 | {'0': 19, '1': 19} | {'dev': 38} | 118492,145816 |
| component_002 | Criminal Procedure | 34 | 17 | 2 | {'0': 17, '1': 17} | {'test': 34} | 118507,136995 |
| component_005 | Economic Activity | 194 | 97 | 7 | {'0': 97, '1': 97} | {'train': 194} | 118410,118457,118486,122257,130141,131157... |
| component_007 | Economic Activity | 118 | 59 | 8 | {'0': 59, '1': 59} | {'train': 118} | 118465,121152,127909,131153,131156,145683... |
| component_000 | Economic Activity | 88 | 44 | 4 | {'0': 44, '1': 44} | {'dev': 88} | 118471,122254,136985,145709 |
| component_006 | Judicial Power | 70 | 35 | 5 | {'0': 35, '1': 35} | {'train': 70} | 118402,118514,134744,145713,145784 |
| component_008 | Judicial Power | 60 | 30 | 4 | {'0': 30, '1': 30} | {'train': 60} | 121150,127912,142885,2621076 |
| component_003 | Judicial Power | 34 | 17 | 2 | {'0': 17, '1': 17} | {'test': 34} | 118478,145767 |

## Original Split Counts

| Issue | Split | Justice | Rows |
| --- | --- | --- | --- |
| Criminal Procedure | dev | Ginsburg | 19 |
| Criminal Procedure | dev | Scalia | 19 |
| Criminal Procedure | test | Ginsburg | 17 |
| Criminal Procedure | test | Scalia | 17 |
| Criminal Procedure | train | Ginsburg | 259 |
| Criminal Procedure | train | Scalia | 259 |
| Economic Activity | dev | Ginsburg | 44 |
| Economic Activity | dev | Scalia | 44 |
| Economic Activity | train | Ginsburg | 156 |
| Economic Activity | train | Scalia | 156 |
| Judicial Power | test | Ginsburg | 17 |
| Judicial Power | test | Scalia | 17 |
| Judicial Power | train | Ginsburg | 65 |
| Judicial Power | train | Scalia | 65 |

## Decision

This run can support a stricter case-connected issue-stratified resplit.
