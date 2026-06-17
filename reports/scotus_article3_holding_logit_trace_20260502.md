# Article III Holding-Logit Causal Trace

## Purpose

Test a cheaper attribution-style localization before launching another long generation sweep. The screen patched generated-thought hidden states into public-leaning Article III targets, then scored a fixed private-vs-public holding logprob margin.

This is not no-mask generation evidence. It is a causal-tracing diagnostic for candidate windows.

## Artifacts

- Script: `scripts/experiments/scotus/trace_article3_holding_logit_patches.py`
- Initial phrase-label run: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050407`
- Calibrated-label small run, blend `0.25`: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050539`
- Calibrated-label small run, blend `1.0`: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050633`
- Calibrated-label expanded source/control run, blend `1.0`: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050730`

## Method

Inputs:

- Source generations: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241`
- Localization source: `sweep_v4/scotus_article3_generated_thought_baseline_localization_20260502_043317`
- Target public-baseline prompts: `2`, `4`
- Top six generated-baseline sites:
  - `L56 residual tail32_mean`
  - `L13 MLP pre_answer_last`
  - `L15 residual pre_answer_last`
  - `L10 residual thought_mean`
  - `L24 mixer pre_answer_last`
  - `L06 residual thought_mean`

The first label wording was rejected as a weak evaluator because one public target had a private-leaning baseline margin. The calibrated label contrast was:

- Private: `Article III objection succeeds.`
- Public: `Article III objection fails.`

With this contrast, both public targets had public-leaning baseline margins around `-2.17`.

## Results

Small source/control pass, blend `0.25`:

| rank | site | private delta | public-control delta | private - control |
| --- | --- | --- | --- | --- |
| 1 | `L13 MLP pre_answer_last` | `0.0038` | `-0.0035` | `0.0074` |
| 2 | `L24 mixer pre_answer_last` | `-0.0036` | `-0.0085` | `0.0049` |
| 3 | `L06 residual thought_mean` | `0.0193` | `0.0162` | `0.0031` |

Small source/control pass, blend `1.0`:

| rank | site | private delta | public-control delta | private - control |
| --- | --- | --- | --- | --- |
| 1 | `L10 residual thought_mean` | `0.5632` | `0.5357` | `0.0275` |
| 2 | `L56 residual tail32_mean` | `-0.0032` | `-0.0046` | `0.0014` |

Expanded source/control pass, blend `1.0`:

| rank | site | n private/control | private delta | public-control delta | private - control |
| --- | --- | --- | --- | --- | --- |
| 1 | `L06 residual thought_mean` | `6/6` | `0.4903` | `0.4662` | `0.0241` |
| 2 | `L56 residual tail32_mean` | `6/6` | `0.0026` | `-0.0079` | `0.0105` |
| 3 | `L13 MLP pre_answer_last` | `6/6` | `-0.0095` | `-0.0128` | `0.0033` |
| 4 | `L24 mixer pre_answer_last` | `6/6` | `-0.0049` | `-0.0032` | `-0.0017` |
| 5 | `L15 residual pre_answer_last` | `6/6` | `-0.0012` | `0.0012` | `-0.0024` |
| 6 | `L10 residual thought_mean` | `6/6` | `0.5302` | `0.5422` | `-0.0120` |

## Interpretation

The only large effects came from early `thought_mean` replacement at `L06` and `L10`, but public source-control traces produced nearly the same movement. That is generic state replacement or thought-distribution disruption, not a private-rights actuator.

The late `L56 residual tail32_mean` site had slightly better private-minus-control directionality in the expanded pass, but the absolute effect was tiny (`0.0105` mean-logprob margin) and follows a generation gate that already failed to move final holdings.

Decision:

- Do not promote any holding-logit trace site.
- Treat this as negative evidence for the current generated-baseline top-six sites under simple source-state replacement.
- Do not launch another long generation sweep from `L06/L10 thought_mean`; those effects are source-control failures.
- If continuing causal tracing, first improve the evaluator target or move to attribution over generated conclusion tokens from actual final answers, not fixed holding phrases.
