# Article III Actual-Answer Continuation Trace

## Purpose

Repair the weak fixed-phrase holding-logit evaluator by scoring actual private-conditioned versus public-conditioned answer continuations for the same ambiguous target prompt.

This is still a localization diagnostic, not no-mask generation evidence.

## Artifacts

- Script: `scripts/experiments/scotus/trace_article3_answer_continuation_patches.py`
- Qwen 3.6 reference-answer trace: `sweep_v4/scotus_article3_answer_continuation_trace_20260502_051337`
- Local Qwen 3.5 reference generation: `sweep_v4/scotus_counterfactual_thoughts_20260502_051531`
- Local reference polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_051905`
- Local-reference answer-continuation trace: `sweep_v4/scotus_article3_answer_continuation_trace_20260502_051921`

## Why This Was Needed

The fixed-label causal trace used short labels:

- `Article III objection succeeds.`
- `Article III objection fails.`

That was cleaner than the first phrase pair, but still too artificial. The next diagnostic used actual private/public answers generated under inserted visible thoughts as the scored continuation target.

The first pass used existing Qwen 3.6 server reference answers. That exposed a model-mismatch problem: local Qwen 3.5 already preferred the Qwen 3.6 private reference answer for prompt `2` before any patch.

## Local Reference Repair

To remove the reference-model mismatch, local Qwen 3.5 generated complete-budget counterfactual answers for prompts `2` and `4`:

- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`
- Prompt ids: `2,4`
- Conditions: neutral, private_rights, public_rights
- Answer budget: `3072`
- Short-budget smoke: `False`

Conclusion-polarity scoring:

| condition | n | private_score | public_score | net | private_rate | public_rate | mixed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 2 | `1.000` | `1.500` | `-0.500` | `0.000` | `0.500` | `0.500` |
| private_rights | 2 | `3.500` | `1.500` | `2.000` | `0.500` | `0.000` | `0.500` |
| public_rights | 2 | `0.500` | `1.500` | `-1.000` | `0.000` | `1.000` | `0.000` |

The local references are usable but imperfect: prompt `4` separates cleanly; prompt `2` remains mixed under the private thought.

## Local-Reference Trace

Configuration:

- Source generations: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241`
- Reference generations: `sweep_v4/scotus_counterfactual_thoughts_20260502_051531`
- Localization source: `sweep_v4/scotus_article3_generated_thought_baseline_localization_20260502_043317`
- Target prompts: `2,4`
- Private sources: `0,1,5`
- Public source-controls: `3,6,7`
- Top sites: `6`
- Blend: `1.0`
- Scored answer tokens: first `256`

Baseline margins:

| Prompt | Private mean | Public mean | Margin |
| --- | --- | --- | --- |
| `A3_AMBIG_03_patent_review_parallel_litigation` | `-0.8837` | `-0.8057` | `-0.0780` |
| `A3_AMBIG_05_industry_fund_contribution` | `-0.7469` | `-0.7525` | `0.0056` |

Aggregate patch effects:

| rank | site | private delta | public-control delta | private - control |
| --- | --- | --- | --- | --- |
| 1 | `L13 MLP pre_answer_last` | `0.0015` | `0.0004` | `0.0011` |
| 2 | `L24 mixer pre_answer_last` | `0.0010` | `0.0005` | `0.0006` |
| 3 | `L56 residual tail32_mean` | `0.0019` | `0.0016` | `0.0003` |
| 4 | `L06 residual thought_mean` | `0.0240` | `0.0259` | `-0.0020` |
| 5 | `L15 residual pre_answer_last` | `-0.0014` | `0.0017` | `-0.0031` |
| 6 | `L10 residual thought_mean` | `0.0146` | `0.0185` | `-0.0040` |

## Decision

Do not promote any actual-answer continuation trace site.

The model-aligned answer-continuation evaluator removed the Qwen 3.6 reference mismatch, but the candidate-vs-control separation disappeared. The only visible movements were again early `thought_mean` source-state effects, and public source controls moved at least as much as private sources.

This closes the current generated-baseline top-six surface under:

- direct additive steering,
- fixed holding-label causal tracing,
- Qwen 3.6 actual-answer continuation tracing,
- local Qwen 3.5 actual-answer continuation tracing.

Next useful actuator work should not widen these same sites. It should switch either to a different localization source or to a trained controller that is evaluated directly by no-mask generation rather than by these patch screens.
