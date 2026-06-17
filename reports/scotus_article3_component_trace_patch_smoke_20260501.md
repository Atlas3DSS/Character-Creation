# SCOTUS Article III Component Trace Patch Smoke

## Purpose

After broad residual trace replacement failed, this smaller smoke patched component outputs one layer/component at a time. The goal was a cheap localization screen: if a small component window beat same-component random traces and the public source-control trace, it could justify a thinking-trace follow-up.

This is still answer-only generation, not a no-mask success gate.

## Run

Run: `sweep_v4/scotus_component_trace_patch_20260501_215707`

Method:

- Script: `scripts/experiments/scotus/patch_scotus_component_traces.py`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Replay rows: `data/scotus/scotus_controlled_replay_v2_examples_20260501.jsonl`
- Candidate source: first train-split `article3_private_rights` trace.
- Source control: matched first train-split `article3_public_rights` trace.
- Layers/components: `L4 mixer`, `L4 mlp`, `L8 mixer`, `L8 mlp`
- Blend: `0.25`
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Controls: 1 same-component random trace per component plus public source-control trace.
- Max new tokens: `64`

## Result

| Component | Candidate target | Random target | Matched target | Candidate net | Random net | Matched net | Source target | Source net | Target strongest win | Net strongest win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L04_mixer` | `0.500` | `1.000` | `-0.500` | `0.500` | `1.000` | `-0.500` | `0.000` | `0.000` | `0.500` | `0.500` |
| `L04_mlp` | `0.000` | `1.500` | `-1.500` | `0.000` | `1.500` | `-1.500` | `1.000` | `1.000` | `0.000` | `0.000` |
| `L08_mixer` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `L08_mlp` | `1.000` | `1.500` | `-0.500` | `1.000` | `0.500` | `0.500` | `0.000` | `0.000` | `0.000` | `0.500` |

## Interpretation

No component survives the control rule.

- `L08_mlp` is the only row with positive candidate net movement, but target movement is still below its same-component random trace and strongest-random target win is `0.000`.
- `L04_mixer` produces small movement, but the random trace is stronger.
- `L04_mlp` is dominated by both random and public source-control traces.
- `L08_mixer` is inert.

## Decision

Do not promote these answer-trace component patches.

This closes the cheap answer-trace localization attempt. The next useful implementation should be a visible-thinking trajectory patcher that captures and patches source/target thought traces over small layer-token-component windows, rather than continuing with answer-only replay traces.
