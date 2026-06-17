# SCOTUS Article III Residual Trace Patch Smoke

## Purpose

Run a cheap localization probe after the Article III low-rank no-mask failure. This test asks whether replacing generated-token residual states with a real private-rights replay trace gives a stronger actuator hint than learned single-layer low-rank editing.

This is not a promotion-grade no-mask audit because it uses answer-only generation rather than visible thinking. It is only a low-cost screen before building a heavier thinking-trajectory patcher.

## Run

Run: `sweep_v4/scotus_trace_patch_20260501_215034`

Method:

- Script: `scripts/experiments/scotus/patch_scotus_replay_traces.py`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Replay rows: `data/scotus/scotus_controlled_replay_v2_examples_20260501.jsonl`
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Candidate source: `article3_private_rights`, train split, first paired replay trace.
- Source control: `article3_public_rights`, train split, matched first paired replay trace.
- Layers: `L4/L8/L12/L16`
- Position: decode-step last-token residual trace replacement
- Blend: `0.25`
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Controls: 2 same-shape random traces plus the public-rights source trace
- Max new tokens: `96`

## Result

Aggregate frame scores:

| Condition | n | Target delta vs base | Contrast delta vs base | Net delta vs base |
| --- | ---: | ---: | ---: | ---: |
| random trace | 4 | `-0.500` | `0.000` | `-0.500` |
| private trace candidate | 2 | `-1.000` | `-1.000` | `0.000` |
| public trace source control | 2 | `-1.000` | `1.500` | `-2.500` |

Prompt-matched candidate versus random:

| Metric | Value |
| --- | ---: |
| target matched delta | `-0.500` |
| target win rate | `0.000` |
| net matched delta | `0.500` |
| net win rate | `1.000` |

Qualitative read:

- The private trace candidate pulled completions toward the source trace template, including malformed phrases such as `judgmentication` and `Analysising`.
- It did not add target-frame hits versus baseline; target hits fell by `1.000`.
- Its only positive result is net movement from suppressing contrast hits, not from adding private-rights reasoning.
- The public source-control trace also distorted outputs and pushed the public prompt toward public-rights language, as expected.

## Decision

Do not promote residual trace replacement as an Article III actuator from this smoke.

This result reinforces the current read: replay traces contain a strong answer-state/template shape, but direct residual replacement over broad layers is too blunt and template-like. The next useful localization pass should patch smaller layer-token-component windows and should use visible thinking traces, not answer-only replay traces.
