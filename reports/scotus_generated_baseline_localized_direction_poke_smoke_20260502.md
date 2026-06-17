# Generated-Baseline Localized Direction Poke Smoke

## Purpose

Test whether a distributed direction localized from Qwen's own generated Article III baseline thoughts can move two public-leaning ambiguous prompts toward the private-rights holding, without inserting private-rights text.

This is a smoke gate, not promotion evidence: `n=2`, one same-site random control, and automatic conclusion-polarity scoring only.

## Artifacts

- Generated-thought localization: `sweep_v4/scotus_article3_generated_thought_baseline_localization_20260502_043317`
- Causal poke run: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_043452`
- Conclusion-polarity run: `sweep_v4/scotus_article3_conclusion_polarity_20260502_045650`
- Source baseline generations: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241`

## Localization

The generated-baseline localizer used manually reviewed Qwen baseline tendencies:

- Private prompt ids: `0,1,5`
- Public prompt ids: `2,3,4,6,7`
- Source generation budgets: `2048/2048`, so this was not short-budget smoke.
- Components: residual, mixer, MLP.
- Regions: pre-answer last token, thought mean, thought-tail16 mean, tail32 mean.
- Shuffle controls: `128`.

Top adjusted sites were materially different from the inserted-thought late-tail cluster:

| rank | site | score-null | effect |
| --- | --- | --- | --- |
| 1 | `L56 residual tail32_mean` | `1.695` | `5.659` |
| 2 | `L13 MLP pre_answer_last` | `0.596` | `3.572` |
| 3 | `L15 residual pre_answer_last` | `0.454` | `3.284` |
| 4 | `L10 residual thought_mean` | `0.432` | `5.708` |
| 5 | `L24 mixer pre_answer_last` | `0.225` | `3.117` |
| 6 | `L06 residual thought_mean` | `0.111` | `4.449` |

Interpretation: this nominated a real generated-trajectory surface, but it is still confounded by prompt facts and baseline holdings. It is not actuator evidence.

## Causal Gate

The poke used the top six generated-baseline sites as a frozen private-minus-public bundle:

`L56_residual_tail32_mean, L13_mlp_pre_answer_last, L15_residual_pre_answer_last, L10_residual_thought_mean, L24_mixer_pre_answer_last, L06_residual_thought_mean`

Configuration:

- Test prompts: `A3_AMBIG_03_patent_review_parallel_litigation`, `A3_AMBIG_05_industry_fund_contribution`
- Alpha: `1.0`
- Position: decode
- Random controls: `1` prompt-matched same-site random bundle
- Thought/answer budgets: `2048/2048`
- Short-budget smoke: `False`

Automatic segment scoring showed a visible-thinking movement but no final-answer movement:

| segment | candidate target delta | candidate net delta | target minus random | net minus random |
| --- | --- | --- | --- | --- |
| thinking | `+2.0` | `+2.5` | `+2.0` | `+1.5` |
| answer | `-2.0` | `-2.0` | `-1.0` | `-1.0` |

Conclusion-polarity scoring:

| condition | n | private | public | net | labels |
| --- | --- | --- | --- | --- | --- |
| base | 2 | `1.0` | `2.0` | `-1.0` | 2 public |
| random_unit | 2 | `1.0` | `1.0` | `0.0` | 1 private, 1 public |
| candidate | 2 | `0.5` | `2.0` | `-1.5` | 1 public, 1 mixed |

## Decision

Do not promote this generated-baseline localized bundle. It moved some visible-thinking frame counts but did not move final holdings private, and the random control produced at least as much conclusion-polarity movement.

The useful lesson is negative: generated-baseline localization produced a different surface from inserted-thought localization, but direct additive steering on that surface still did not become a reliable no-mask Article III actuator.

Next actuator work should either:

- move to a trained multi-site controller over generated-token trajectories, with source/random/strongest-random gates; or
- try a different intervention family such as causal tracing/attribution-selected patching or weight-space ablation, rather than another broad additive sweep on these sites.
