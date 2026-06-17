# SCOTUS Text/Activation Complementarity Audit

## Purpose

This checks whether selected cached activation probes add unique held-out wins over the rendered-prompt TF-IDF baseline. It does not create a new steering direction; it is a promotion-risk audit for the current majority-2000s feasible-issues branch.

## Input

- Resplit directory: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116`

## Split Results

| Plan | Split | N | Text BA | Activation BA | Delta | Both correct | Activation only | Text only | Both wrong | Activation acc on text-wrong | Activation acc on text-uncertain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| split_00 | dev | 250 | 0.592 | 0.768 | +0.176 | 126 | 66 | 22 | 36 | 0.647 | 0.751 |
| split_00 | test | 332 | 0.744 | 0.753 | +0.009 | 210 | 40 | 37 | 45 | 0.471 | 0.626 |
| split_01 | dev | 332 | 0.789 | 0.846 | +0.057 | 242 | 39 | 20 | 31 | 0.557 | 0.787 |
| split_01 | test | 250 | 0.616 | 0.660 | +0.044 | 138 | 27 | 16 | 69 | 0.281 | 0.599 |
| split_02 | dev | 220 | 0.686 | 0.759 | +0.073 | 121 | 46 | 30 | 23 | 0.667 | 0.709 |
| split_02 | test | 362 | 0.655 | 0.749 | +0.094 | 202 | 69 | 35 | 56 | 0.552 | 0.673 |
| split_03 | dev | 362 | 0.715 | 0.812 | +0.097 | 229 | 65 | 30 | 38 | 0.631 | 0.768 |
| split_03 | test | 220 | 0.700 | 0.732 | +0.032 | 120 | 41 | 34 | 25 | 0.621 | 0.662 |
| split_04 | dev | 332 | 0.771 | 0.837 | +0.066 | 232 | 46 | 24 | 30 | 0.605 | 0.786 |
| split_04 | test | 260 | 0.712 | 0.746 | +0.035 | 165 | 29 | 20 | 46 | 0.387 | 0.698 |

## Aggregate Test Read

- Median activation-minus-text test BA delta: `0.035`
- Test discordant wins across plans: activation-only `206`, text-only `142`
- Activation-only issues: `{'Economic Activity': 76, 'Judicial Power': 46, 'Criminal Procedure': 84}`
- Text-only issues: `{'Economic Activity': 71, 'Judicial Power': 13, 'Criminal Procedure': 58}`

## Decision

Do not promote this branch on complementarity grounds. The selected activation probes do not add a stable held-out advantage over text baselines.
