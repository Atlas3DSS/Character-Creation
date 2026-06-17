# SCOTUS Slice Candidate Mining

## Purpose

This reuses cached Phase 4.1 Qwen3.6-27B FP8 activation features to search for justice-style slices where hidden-state probes beat a matched cue-masked text baseline. It does not load the model and does not establish steering by itself.

## Inputs

- Run directory: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_phase41_normal_20260425_102519`
- Layers: `4,9,16,19,40`
- Regions: `prompt_last,prompt_mean,excerpt_mean`
- C grid: `0.003,0.01,0.03,0.1`

## Promotion Rule

A mined slice is only a candidate if activation dev BA >= `0.75`, activation test BA >= `0.75`, and activation-minus-text test BA >= `0.08`.

## Candidate Rows

| Slice | N | Dev BA | Test BA | Text test BA | Gap | Best readout |
| --- | --- | --- | --- | --- | --- | --- |
| section_posture=majority__decade=2000s | 1494 | 0.802 | 0.809 | 0.500 | 0.309 | prompt_last @ L9 |
| issue_area_label=Economic Activity | 778 | 0.864 | 0.900 | 0.700 | 0.200 | excerpt_mean @ L19 |
| issue_area_label=Economic Activity__section_posture=majority | 750 | 0.875 | 0.900 | 0.733 | 0.167 | excerpt_mean @ L19 |
| decision_direction=1 | 1956 | 0.835 | 0.833 | 0.667 | 0.167 | prompt_last @ L9 |
| all | 3592 | 0.819 | 0.841 | 0.754 | 0.087 | prompt_last @ L9 |

## Top Rows By Activation Minus Text

| Slice | N | Dev BA | Test BA | Text test BA | Gap | Best readout | C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| section_posture=majority__decade=2000s | 1494 | 0.802 | 0.809 | 0.500 | 0.309 | prompt_last @ L9 | 0.03 |
| issue_area_label=Criminal Procedure__decade=2000s | 666 | 0.974 | 0.706 | 0.500 | 0.206 | prompt_mean @ L4 | 0.003 |
| issue_area_label=Economic Activity | 778 | 0.864 | 0.900 | 0.700 | 0.200 | excerpt_mean @ L19 | 0.003 |
| decade=2000s | 1724 | 0.802 | 0.706 | 0.529 | 0.176 | prompt_mean @ L16 | 0.01 |
| issue_area_label=Economic Activity__section_posture=majority | 750 | 0.875 | 0.900 | 0.733 | 0.167 | excerpt_mean @ L19 | 0.01 |
| decision_direction=1 | 1956 | 0.835 | 0.833 | 0.667 | 0.167 | prompt_last @ L9 | 0.003 |
| decision_direction=2 | 1636 | 0.743 | 0.806 | 0.648 | 0.157 | prompt_last @ L4 | 0.1 |
| issue_area_label=Criminal Procedure__section_posture=majority | 788 | 0.934 | 0.581 | 0.446 | 0.135 | prompt_last @ L4 | 0.003 |
| issue_area_label=Criminal Procedure | 924 | 0.961 | 0.635 | 0.500 | 0.135 | prompt_last @ L4 | 0.01 |
| chunk_position_bucket=early | 930 | 0.839 | 0.706 | 0.618 | 0.088 | prompt_last @ L9 | 0.003 |
| all | 3592 | 0.819 | 0.841 | 0.754 | 0.087 | prompt_last @ L9 | 0.003 |
| chunk_position_bucket=middle | 1688 | 0.868 | 0.818 | 0.742 | 0.076 | prompt_last @ L4 | 0.03 |
| issue_area_label=Judicial Power | 854 | 0.662 | 0.971 | 0.912 | 0.059 | prompt_last @ L9 | 0.003 |
| chunk_position_bucket=late | 974 | 0.790 | 0.842 | 0.789 | 0.053 | prompt_last @ L4 | 0.003 |
| section_posture=majority | 3240 | 0.836 | 0.768 | 0.732 | 0.036 | prompt_last @ L4 | 0.1 |
| decade=1990s | 1868 | 0.795 | 0.843 | 0.814 | 0.029 | prompt_last @ L4 | 0.01 |
| issue_area_label=Judicial Power__section_posture=majority | 756 | 0.676 | 0.882 | 0.912 | -0.029 | prompt_last @ L4 | 0.1 |
| section_posture=majority__decade=1990s | 1746 | 0.777 | 0.800 | 0.843 | -0.043 | prompt_last @ L4 | 0.03 |
| issue_area_label=Criminal Procedure__decade=1990s | 258 | 0.868 | 0.450 | 0.600 | -0.150 | prompt_last @ L4 | 0.003 |

## Use Rules

1. Treat this as a cheap triage pass only; the cached run used Qwen3.6 FP8, not the Qwen3.5 BF16 source of record.
2. Do not run steering from a mined slice unless it survives BF16 recapture or an existing BF16 feature equivalent.
3. Slices with high text baselines are leakage diagnostics, not candidates.

## BF16 Verification Update

The top mined slice, `section_posture=majority__decade=2000s`, was recaptured on Qwen3.5 BF16:

- Report: `reports/scotus_slice_bf16_majority2000s_20260501.md`
- Run: `sweep_v4/scotus_slice_bf16_majority2000s_normal_20260501_022109/report.md`
- Selected readout: `excerpt_mean @ L16`, C `0.001`
- Dev BA: `0.810`
- Test BA: `0.691`
- Text test BA: `0.500`
- Label-shuffle null: best dev BA `0.603`, test BA `0.515`, with no sweep configs above `0.70`

Decision: the slice has real activation structure, but it does not survive the held-out BF16 promotion gate. Do not promote it to steering.

## Majority-2000s Feasible-Issues Refinement

The majority-2000s slice was refined to issue families with strict case-component split feasibility: `Criminal Procedure`, `Economic Activity`, and `Judicial Power`.

- Detailed report: `reports/scotus_slice_majority2000s_feasible_issues_20260501.md`
- Split feasibility audit: `reports/scotus_slice_majority2000s_feasible_issues_split_feasibility_20260501.md`
- Normal component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/report.md`
- Label-shuffle component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_label_shuffle_component_resplits_20260501_034539/report.md`
- Template-variant component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_template_variant_component_resplits_20260501_040538/report.md`
- Plain-prompt component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_plain_prompt_component_resplits_20260501_041503/report.md`
- Excerpt-removed component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_excerpt_removed_component_resplits_20260501_042048/report.md`
- Neutral-filler component resplits: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_neutral_filler_component_resplits_20260501_043234/report.md`

Component-resplit read:

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA |
| --- | ---: | ---: | ---: | ---: |
| Normal | `0.812` | `0.746` | `0.660-0.753` | `0.753` |
| Label shuffle | `0.536` | `0.541` | `0.477-0.548` | `0.488` |
| Template variant | `0.807` | `0.758` | `0.668-0.807` | `0.777` |
| Plain prompt | `0.818` | `0.764` | `0.676-0.796` | `0.777` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` |
| Neutral filler | `0.575` | `0.542` | `0.512-0.564` | `0.548` |

Decision: keep this refined branch live for a causal diagnostic. It survives label-shuffle and prompt-format controls and collapses under excerpt removal/neutral filler, but it is still prompt-last-heavy and text-close on some split plans, so it is not yet a steerable circuit.

Causal update:

- Detailed causal report: `reports/scotus_majority2000s_feasible_issues_causal_pilot_20260501.md`
- `prompt_last @ L10`: `sweep_v4/scotus_sae_poke_20260501_045156/report.md`, best prompt-matched target `z=0.184`, best net `z=0.533`
- `excerpt_mean @ L16`: `sweep_v4/scotus_sae_poke_20260501_060425/report.md`, best prompt-matched target `z=0.449`, best net `z=0.395`

Decision update: the refined branch remains the best decodability candidate, but the first two broad causal pokes do not promote it as a steerable circuit.

## Economic Activity BF16 Verification Update

The next mined slice, `issue_area_label=Economic Activity`, was recaptured on Qwen3.5 BF16 and audited with stricter case-component resplits.

- Detailed report: `reports/scotus_slice_economic_style_bf16_20260501.md`
- Original BF16 run: `sweep_v4/scotus_slice_bf16_economic_style_normal_20260501_023619/report.md`
- Normal component resplits: `sweep_v4/scotus_slice_bf16_economic_style_normal_component_resplits_20260501_024704/report.md`
- Label-shuffle component resplits: `sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_component_resplits_20260501_025016/report.md`
- Excerpt-removed component resplits: `sweep_v4/scotus_slice_bf16_economic_style_excerpt_removed_component_resplits_20260501_025835/report.md`
- Neutral-filler component resplits: `sweep_v4/scotus_slice_bf16_economic_style_neutral_filler_component_resplits_20260501_030511/report.md`
- Template-variant component resplits: `sweep_v4/scotus_slice_bf16_economic_style_template_variant_component_resplits_20260501_032205/report.md`
- Plain-prompt component resplits: `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_component_resplits_20260501_032744/report.md`
- Split component review: `reports/scotus_economic_split_component_review_20260501.md`

Original BF16 result:

| Slice | Best readout | Dev BA | Test BA | Text test BA | Test N |
| --- | --- | ---: | ---: | ---: | ---: |
| `issue_area_label=Economic Activity` | `prompt_last @ L24` | `0.875` | `1.000` | `0.700` | `30` |

Case-component resplit result:

| Diagnostic | Median dev BA | Median test BA | Test BA range |
| --- | ---: | ---: | ---: |
| Normal | `0.795` | `0.743` | `0.690-0.856` |
| Label shuffle | `0.568` | `0.500` | `0.471-0.542` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` |
| Neutral filler | `0.601` | `0.554` | `0.500-0.576` |
| Template variant | `0.819` | `0.690` | `0.669-0.856` |
| Plain prompt | `0.789` | `0.673` | `0.615-0.856` |

Decision: this remains useful evidence that broad justice-style information is decodable, but it is not a steering candidate. It survives label-shuffle and prompt-ablation controls, yet strict-resplit performance is modest, text baselines are close on several split plans, selected readouts are mostly `prompt_last`, the prompt-template invariance gate fails under component resplits, and the high text splits are dominated by case/topic features.
