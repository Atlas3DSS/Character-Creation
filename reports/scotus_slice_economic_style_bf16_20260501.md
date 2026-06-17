# SCOTUS Economic Activity Justice-Style BF16 Audit

## Purpose

This audits the mined `issue_area_label=Economic Activity` Scalia/Ginsburg justice-style slice on Qwen3.5 BF16 features. The goal is to decide whether it should advance from a correlational activation probe to causal steering.

## Source Runs

| Artifact | Path |
| --- | --- |
| Original BF16 probe | `sweep_v4/scotus_slice_bf16_economic_style_normal_20260501_023619/report.md` |
| Original label-shuffle null | `sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_20260501_024011/report.md` |
| Split feasibility audit | `reports/scotus_slice_economic_split_feasibility_20260501.md` |
| Component resplit runner | `scripts/experiments/scotus/resplit_cached_scotus_probe.py` |
| Normal component resplits | `sweep_v4/scotus_slice_bf16_economic_style_normal_component_resplits_20260501_024704/report.md` |
| Label-shuffle component resplits | `sweep_v4/scotus_slice_bf16_economic_style_label_shuffle_component_resplits_20260501_025016/report.md` |
| Excerpt-removed component resplits | `sweep_v4/scotus_slice_bf16_economic_style_excerpt_removed_component_resplits_20260501_025835/report.md` |
| Neutral-filler component resplits | `sweep_v4/scotus_slice_bf16_economic_style_neutral_filler_component_resplits_20260501_030511/report.md` |
| Template-variant original probe | `sweep_v4/scotus_slice_bf16_economic_style_template_variant_20260501_031750/report.md` |
| Template-variant component resplits | `sweep_v4/scotus_slice_bf16_economic_style_template_variant_component_resplits_20260501_032205/report.md` |
| Plain-prompt original probe | `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_20260501_032236/report.md` |
| Plain-prompt component resplits | `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_component_resplits_20260501_032744/report.md` |
| Split component review | `reports/scotus_economic_split_component_review_20260501.md` |

## Original BF16 Probe

The first BF16 run looked strong but had a tiny original test split.

| Run | Best readout | Dev BA | Test BA | Text test BA | Test N |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal original split | `prompt_last @ L24`, C `0.03` | `0.875` | `1.000` | `0.700` | `30` |
| Label-shuffle original split | `prompt_last @ L24`, C `0.001` | `0.625` | `0.733` | `0.633` | `30` |
| Template-variant original split | `excerpt_mean @ L24`, C `0.03` | `0.898` | `0.967` | `0.700` | `30` |
| Plain-prompt original split | `excerpt_mean @ L24`, C `0.03` | `0.898` | `0.933` | `0.700` | `30` |

Read: the normal, template-variant, and plain-prompt original splits are all strong, but the original test set is only 15 examples per label and the null can still reach `0.733` test BA diagnostically.

## Split Feasibility

The Economic Activity slice has `778` rows across `8` case-connected components. Unlike the broader majority-2000s slice, it can support strict case-component train/dev/test resplitting.

The predeclared split plans were selected by metadata only: component size, label balance, and requiring both `1990s` and `2000s` material in dev and test. The primary split is `split_00`, the best metadata-balanced plan, not the best test-scoring plan.

## Component Resplit Results

| Diagnostic | Median dev BA | Median test BA | Test BA range | Primary split test BA |
| --- | ---: | ---: | ---: | ---: |
| Normal | `0.795` | `0.743` | `0.690-0.856` | `0.703` |
| Label shuffle | `0.568` | `0.500` | `0.471-0.542` | `0.480` |
| Excerpt removed | `0.500` | `0.500` | `0.500-0.500` | `0.500` |
| Neutral filler | `0.601` | `0.554` | `0.500-0.576` | `0.554` |
| Template variant | `0.819` | `0.690` | `0.669-0.856` | `0.676` |
| Plain prompt | `0.789` | `0.673` | `0.615-0.856` | `0.655` |

Plan-by-plan comparison:

| Plan | Normal test BA | Label-shuffle test BA | Excerpt-removed test BA | Neutral-filler test BA | Normal text test BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| `split_00` | `0.703` | `0.480` | `0.500` | `0.554` | `0.622` |
| `split_01` | `0.790` | `0.471` | `0.500` | `0.529` | `0.746` |
| `split_02` | `0.690` | `0.542` | `0.500` | `0.500` | `0.637` |
| `split_03` | `0.856` | `0.500` | `0.500` | `0.576` | `0.839` |
| `split_04` | `0.743` | `0.520` | `0.500` | `0.568` | `0.622` |

## Prompt-Template Invariance Gate

The original split looked invariant: `template_variant` selected `excerpt_mean @ L24` with test BA `0.967`, and `plain_prompt` selected `excerpt_mean @ L24` with test BA `0.933`. The component resplits did not preserve that as a promotion-quality signal.

| Mode | Median test BA | Primary split test BA | Median text test BA | Selected readouts |
| --- | ---: | ---: | ---: | --- |
| Normal chat template | `0.743` | `0.703` | `0.637` | all `prompt_last` |
| Template variant | `0.690` | `0.676` | `0.649` | all `prompt_last` |
| Plain prompt | `0.673` | `0.655` | `0.649` | all `prompt_last` |

This fails the predeclared gate. The variant and plain-prompt median test BA are below `0.70`, the advantage over text is small on several plans, and the selected directions are still prompt-last rather than stable excerpt-internal readouts.

The follow-up component review supports a topic/case explanation for the high text splits. The top text n-grams include case-specific cues such as `volvo`, `reeder`, `articles`, `maritime`, `fcc`, `the epa`, `arbitration`, and `antitrust`, rather than clean abstract justice-style markers.

## Readout Stability

The normal component-resplit winners remain mostly prompt-last directions:

| Plan | Best readout | Dev BA | Test BA |
| --- | --- | ---: | ---: |
| `split_00` | `prompt_last @ L8`, C `0.003` | `0.848` | `0.703` |
| `split_01` | `prompt_last @ L12`, C `0.01` | `0.730` | `0.790` |
| `split_02` | `prompt_last @ L24`, C `0.1` | `0.881` | `0.690` |
| `split_03` | `prompt_last @ L8`, C `0.1` | `0.732` | `0.856` |
| `split_04` | `prompt_last @ L8`, C `0.01` | `0.795` | `0.743` |

This is weaker than a stable `excerpt_mean` or `prompt_mean` mid-layer candidate. The earlier mined FP8 row selected `excerpt_mean @ L19`, but the stricter BF16 resplits do not preserve that as the selected readout.

## Decision

Do not promote the Economic Activity justice-style slice to causal steering yet.

What it supports:

1. There is real activation structure: normal resplits beat label-shuffle, excerpt-removed, and neutral-filler controls.
2. The signal is not explained by constant prompt scaffolding: excerpt removal is exactly chance.
3. Length/position artifacts are not enough to explain the normal result: neutral filler stays much lower than normal.
4. Original-split performance is robust to prompt wrapper changes, which was worth auditing.

Why it is not steering-ready:

1. The strict-resplit effect is modest: median test BA falls from the original `1.000` test result to `0.743`.
2. Text baselines are close on multiple split plans, especially `split_01` and `split_03`.
3. The selected directions are mostly `prompt_last`, which is less attractive for causal steering than stable excerpt-internal readouts.
4. The prompt-template invariance gate fails under component resplits: `template_variant` median test BA is `0.690`, and `plain_prompt` median test BA is `0.673`.
5. This remains correlational decoding evidence, not an intervention.

## Next Steps

Do not run a causal hook pilot from the current Economic Activity justice-style directions.

Next useful work:

1. Use the split component review as evidence that broad Economic Activity is too case/topic entangled for immediate steering.
2. If continuing Economic Activity, define a narrower source-grounded subdoctrine contrast from actual opinions instead of broad Scalia/Ginsburg style over all Economic Activity rows.
3. Keep this slice as evidence that justice-style information is decodable, but treat it as a rejected steering candidate unless a future narrower contrast clears component resplits with a non-`prompt_last` readout.
