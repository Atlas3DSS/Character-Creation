# Qwen3.5 J-Lens Overnight Follow-Up

Date: 2026-07-07

## Runs

- Hook sweep: `sweep_v4/qwen35_spectral_hook_smoke_20260706_224909_broad`
- Dev-box sarcasm/math shard: `sweep_v4/qwen36_devbox_contrastive_pairs_20260706_2224`
- Dev-box balanced shard: `sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429`
- SCOTUS prefill package: `sweep_v4/scotus_prefill_contrasts_20260706_234523`
- SCOTUS dev-box generation: `sweep_v4/scotus_prefill_devbox_pairs_20260707_002627`
- SCOTUS J-space smoke: `sweep_v4/scotus_jlens_prefill_readout_20260707_010949_pilot60`

## Hook Sweep Result

Diagnostic only; not promotion-eligible. Qwen3.5-27B ran with no system prompt and `max_new_tokens=4096`.

Tested spectral sarcasm-minus-math unit directions from `qwen35_map/27b/spectral_analysis`:

- Single layers: 34, 48, 49, 50, 51
- Positive alphas: 1, 2, 4, 8
- Negative multilayer bundle: layers 48+49+50 at alphas -1, -2, -4, -8
- Prompt bank: 3 style prompts + 3 math/logic controls per condition

All completed conditions preserved controls: every condition had `control_accuracy = 1.0`.

No condition showed a convincing style gain. Baseline `mean_high_precision_style_count` was `0.33`; nearly all target layers/bundles stayed at `0.33`. L34 at alpha 2/4 reached `0.67`, but manual inspection showed generic explanatory prose rather than a useful persona shift. L50 alpha 8 increased assistant-style artifacts.

Interpretation: the spectral/J-lens-promising sarcasm-minus-math direction is control-clean but not causally useful in this hook smoke. Do not expand positive single-layer alphas from this direction without a new reason.

## Generated Data

All dev-box generation used the regular Qwen3.6 servers on ports 8080/8181 with `max_tokens=4096`; no errors were recorded.

| Shard | Pairs | Notes |
|---|---:|---|
| sarcasm/math | 200 | Mostly `sarcasm_elicitation` and `math_reasoning`; useful for a sharper sarcasm-vs-math direction pass |
| balanced general | 120 | Broader mix across math, technical, sarcasm, provocations, creative, household, opinion, identity, social advice |
| SCOTUS prefill | 240 | Between-writer matched chunk prefills; avg response lengths about 2.8k/2.9k chars |

SCOTUS prefill package contains more than the generated shard used tonight:

- `decision_breakdown.jsonl`: 809 opinions decomposed
- `prefills.jsonl`: 540 contrast prefills
- `jspace_queue.jsonl`: 2700 aligned J-space queue rows
- Axes: 240 between-writer matched chunks, 180 within-same-opinion-section, 120 same-case/different-section

The generated SCOTUS shard used the first 240 prefills, so it covers only `between_writer_matched_chunk`. The within-opinion axes are prepared but not yet generated.

## SCOTUS J-Space Smoke

The J-space prefill readout path was validated on a small sample and stopped early after 137 records.

- Axis covered: `between_writer_matched_chunk`
- Variants covered: generation prompt, seed A, seed B, source A prefix, source B prefix
- Legal-token hits: 24 / 137 rows
- Mean transported norm rose strongly in late layers: L48 `1.43`, L49 `1.53`, L50 `1.61`, L51 `1.61`

This is only a path/smoke result. It says the prefill J-space readout works and late layers carry stronger transported disposition, not that a writer/style direction is steerable.

## Next Experiment

1. Build fresh Qwen3.5 activation directions from the new generated shards, starting with SCOTUS between-writer and the balanced general shard.
2. Run a stratified SCOTUS J-space readout over all three axes: between-writer, within-same-opinion-section, and same-case/different-section.
3. Only then run another hook sweep, using directions selected for both J-space signal and clean semantic readout.

Do not spend more GPU time scaling the current positive spectral mean direction. Its broad hook result was a clean no-go.
