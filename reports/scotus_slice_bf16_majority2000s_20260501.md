# SCOTUS Majority 2000s Slice BF16 Verification

## Purpose

This run verified the strongest cached slice-mining candidate on the Qwen3.5 BF16 source-of-record model. The candidate came from cached Qwen3.6-27B FP8 Phase 4.1 features and was therefore triage-only until recaptured on Qwen3.5.

## Artifacts

| Artifact | Path |
| --- | --- |
| Slice-mining report | `reports/scotus_slice_candidate_mining_20260501.md` |
| BF16 verification run | `sweep_v4/scotus_slice_bf16_majority2000s_normal_20260501_022109/report.md` |
| Label-shuffle null | `sweep_v4/scotus_slice_bf16_majority2000s_label_shuffle_20260501_022912/report.md` |
| Updated probe script | `scripts/experiments/scotus/probe_scotus_style.py` |

## Method

- Pair: `Scalia_vs_Ginsburg`
- Text variant: `masked`
- Slice: `section_posture=majority,decade=2000s`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Layers: `4,7,8,9,10,16,19,40`
- Regions: `prompt_last,prompt_mean,excerpt_mean`
- C grid: `0.001,0.003,0.01,0.03,0.1`
- Split support: `650/650` train, `63/63` dev, `34/34` test by justice label

The script now supports `--slice-filter field=value`, and cached feature reuse now recomputes the text baseline when a slice filter is applied.

## BF16 Result

| Readout | Dev BA | Test BA | Text test BA | Test CI | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `excerpt_mean @ L16`, C `0.001` | `0.810` | `0.691` | `0.500` | `0.582-0.795` | Not promoted |

The selected non-prompt readout beats the text baseline on test, but it misses the `0.75` held-out gate and has a wide confidence interval.

Diagnostic prompt-last rows reached higher test scores, including `prompt_last @ L40` test BA `0.897`, but those rows were not the dev-selected headline result and should not be promoted by test picking.

## Null Result

The label-shuffle null stayed near chance:

| Diagnostic | Best readout | Dev BA | Test BA | Sweep dev >= 0.70 | Sweep test >= 0.70 |
| --- | --- | ---: | ---: | ---: | ---: |
| `label_shuffle` | `prompt_last @ L19` | `0.603` | `0.515` | `0` | `0` |

This means the majority-2000s slice contains real activation structure, but the structure is not strong or stable enough to count as a steerable-circuit candidate.

## Fragility

The held-out split is issue-fragile. The normal BF16 run's stress table reported `0.853` held-out balanced accuracy for `Judicial Power` and `0.588` for `Criminal Procedure`. The main test set contains only `34` Judicial Power and `34` Criminal Procedure examples, while the dev set contains Criminal Procedure and Economic Activity but no Judicial Power.

This makes the result a useful split/representation diagnostic, not a causal steering target.

## Decision

Do not promote the majority-2000s slice to steering. The next useful action is not another qualitative poke from this direction; it is either a split-stratification repair for justice-style slices or a new, less lexical source-grounded subdoctrine.

## Feasible-Issues Follow-up

The split-stratification repair was run after this report. The slice was restricted to issue families with strict case-component train/dev/test feasibility: `Criminal Procedure`, `Economic Activity`, and `Judicial Power`.

- Detailed follow-up: `reports/scotus_slice_majority2000s_feasible_issues_20260501.md`
- Feasibility audit: `reports/scotus_slice_majority2000s_feasible_issues_split_feasibility_20260501.md`
- Normal component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/report.md`
- Label-shuffle component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_label_shuffle_component_resplits_20260501_034539/report.md`
- Template-variant component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_040538/report.md`
- Plain-prompt component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_041503/report.md`
- Excerpt-removed component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_excerpt_removed_component_resplits_20260501_042048/report.md`
- Neutral-filler component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_neutral_filler_component_resplits_20260501_043234/report.md`

Follow-up read:

| Diagnostic | Median test BA | Test BA range |
| --- | ---: | ---: |
| Normal | `0.746` | `0.660-0.753` |
| Label shuffle | `0.541` | `0.477-0.548` |
| Template variant | `0.758` | `0.668-0.807` |
| Plain prompt | `0.764` | `0.676-0.796` |
| Excerpt removed | `0.500` | `0.500-0.500` |
| Neutral filler | `0.542` | `0.512-0.564` |

Decision update: the refined feasible-issues branch is live for a small causal diagnostic, but it remains prompt-last-heavy and text-close on some split plans. It should not be described as a steerable circuit unless a causal patch/steering pilot beats prompt-matched same-layer random controls.
