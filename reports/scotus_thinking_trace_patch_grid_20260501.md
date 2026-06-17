# SCOTUS Visible-Thinking Trace Patch Grid

## Goal

Run the pre-registered small localization grid after the first `L08_mlp` smoke. This tested whether a real Article III private-rights thinking trace could causally move Qwen3.5's visible reasoning trajectory, not just its final answer, when patched into selected mixer/MLP windows during thought generation.

This is a localization screen, not a final actuator validation. A window only nominates if candidate thinking movement beats same-shape random traces and the public-rights source-control trace.

## Artifacts

- Run: `sweep_v4/scotus_thinking_trace_patch_20260501_221231`
- Raw report: `sweep_v4/scotus_thinking_trace_patch_20260501_221231/report.md`
- Candidate/control table: `sweep_v4/scotus_thinking_trace_patch_20260501_221231/candidate_vs_matched_controls.jsonl`
- Segment summary: `sweep_v4/scotus_thinking_trace_patch_20260501_221231/segment_score_summary.jsonl`
- Runner: `scripts/experiments/scotus/patch_scotus_thinking_traces.py`

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source generations: `sweep_v4/scotus_thinking_lowrank_poke_20260501_204712/generations.jsonl`
- Candidate source thinking: base `A3_PRIV_02_bankruptcy_counterclaim`
- Source-control thinking: base `A3_PUBLIC_01_benefits_eligibility`
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Prompt keys: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Patch windows: `L04_mixer`, `L04_mlp`, `L08_mixer`, `L08_mlp`, `L12_mixer`, `L12_mlp`, `L16_mixer`, `L16_mlp`
- Blend: `0.25`
- Random controls: `2` same-shape random traces per window
- Trace/thought/answer budgets: `96`/`128`/`64` tokens
- Intervention: patch trace only during visible thought generation, mechanically close `</think>`, then generate the final answer unpatched from the patched thought.

## Result

No window passed the no-mask reasoning-trajectory gate.

Visible-thinking result:

| window | target-minus-random | net-minus-random | source target/net | strongest target/net wins |
| --- | ---: | ---: | --- | --- |
| `L04_mixer` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L04_mlp` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L08_mixer` | `0.000` | `-0.500` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L08_mlp` | `0.000` | `-0.500` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L12_mixer` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L12_mlp` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L16_mixer` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L16_mlp` | `0.000` | `0.500` | `0.000`/`-0.500` | `0.000`/`0.500` |

Answer result:

- `L08_mixer` and `L08_mlp` both showed answer target/net movement of `+1.000` versus random and public source-control traces.
- That answer movement is not promotable because the same windows produced no positive visible-thinking target movement and failed strongest-control wins on the reasoning segment.
- `L16_mlp` was worse than random and matched or beaten by the public source-control trace.
- All answers were nonempty and no imitation markers were detected, so the failure is not a formatting or mask-artifact failure.

## Decision

Do not promote any of the tested Article III visible-thinking trace patch windows.

This closes the small grid over `L4/L8/L12/L16 x mixer/MLP` for the current source/control trace pair, blend, prompt pair, and proposition scorer. The most interesting artifact is that final answers can move under `L08` thinking-trace patches without measurable target-frame movement in the visible thought. Under the project objective, that is a warning sign rather than success: it suggests downstream answer phrasing can be perturbed without showing a durable reasoning-basin shift.

## Next

Do not run a full-bank audit on these windows. The next useful localization pass should change the unit of intervention rather than widening this grid:

- patch token windows inside the thinking trace instead of the entire prefix trace;
- compare source-to-target causal tracing against within-frame source traces, not only random traces;
- test multi-site controllers only after a trajectory-localization pass nominates positions/components that move visible reasoning itself;
- keep LoRA/ReFT/LoReFT as diagnostics or routes to permanent edits, not as success by themselves.
