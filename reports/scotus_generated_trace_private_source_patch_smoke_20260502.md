# SCOTUS Generated-Trace Private-Source Patch Smoke

Date: 2026-05-02

## Goal

Test whether a real Qwen-generated private-rights thinking trajectory can act as a no-mask actuator when patched into fresh generated thinking for public-leaning Article III prompts.

This is a generated-token trajectory test, not a teacher-forced inserted-thought-tail test. It asks whether patching a source trace during generation changes the model's own visible reasoning and final holding.

## Artifacts

- Trace-patch run: `sweep_v4/scotus_thinking_trace_patch_20260502_035453`
- Polarity score run: `sweep_v4/scotus_article3_conclusion_polarity_20260502_042512`
- Source generations: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241/generations.jsonl`

## Configuration

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source trace: `A3_AMBIG_02_bankruptcy_counterclaim_distribution` (`base`, manually private)
- Source-control trace: `A3_AMBIG_07_benefits_fraud_recoupment` (`base`, manually public)
- Test prompts: `A3_AMBIG_03_patent_review_parallel_litigation`, `A3_AMBIG_05_industry_fund_contribution` (public-baseline targets)
- Patch site: `L62_residual`
- Patch token window: `w000_064`
- Blend: `0.25`
- Random controls: `1`
- Thought/answer caps: `3072/3072`
- Short-budget smoke: `False`

## Results

Segment-frame summary:

| Segment | Condition | Mean target delta | Mean net delta | Notes |
| --- | ---: | ---: | ---: | --- |
| thinking | random trace | `-2.0` | `-1.5` | Random perturbation reduced target-frame hits. |
| thinking | source-control trace | `-1.5` | `-1.0` | Public source-control matched the candidate. |
| thinking | candidate private trace | `-1.5` | `-1.0` | Not a private-frame reasoning movement. |
| answer | random trace | `0.0` | `-1.0` | No private holding movement. |
| answer | source-control trace | `-0.5` | `-1.0` | Mixed/public, not private. |
| answer | candidate private trace | `0.5` | `0.0` | Slight proposition-score bump, but not a holding shift. |

Conclusion-polarity summary:

| Condition | n | Private score | Public score | Net private-public | Private rate | Public rate | Mixed rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `2` | `1.0` | `2.0` | `-1.0` | `0.0` | `1.0` | `0.0` |
| random trace | `2` | `1.0` | `2.0` | `-1.0` | `0.0` | `0.5` | `0.5` |
| source-control trace | `2` | `1.0` | `1.5` | `-0.5` | `0.0` | `0.5` | `0.5` |
| candidate private trace | `2` | `1.0` | `2.5` | `-1.5` | `0.0` | `1.0` | `0.0` |

## Interpretation

This run does not validate the L62 residual generated-trace patch as an actuator.

The candidate private-source patch did not move either final holding to a private-rights objection. It made the polarity scorer more public than baseline (`net -1.5` vs. base `-1.0`) and stayed `2/2` public. In visible thinking, candidate movement was matched by the source-control trace, so it fails the "not just any coherent legal trace" control.

The only favorable signal is a small answer proposition-score improvement over the random trace, but it does not survive conclusion-polarity scoring and has `n=2`. Treat it as perturbation/noise, not evidence of a useful actuator.

Operational note: `answer_generated_tokens` is often `1` in this two-stage harness because Qwen closes the `<think>` block and emits the answer during the first 3072-token generation stage; the answer text was nonempty and the run was not short-budget.

## Decision

Do not promote this candidate. The simple generated-trace early-window patch at `L62_residual`, `w000_064`, `alpha=0.25` should be treated as negative evidence against the current late-tail trajectory-patching branch.

Next useful branches are either a genuinely distributed/generated-token trajectory actuator (multiple sites and generated-token state deltas, not one late residual trace) or a learned ReFT/adapter diagnostic with held-out no-mask generation gates. Expanding this exact one-site trace patch grid is lower priority unless a stronger localization signal appears.
