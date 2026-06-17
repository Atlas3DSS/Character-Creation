# SCOTUS Visible-Thinking Trace Patch Token Windows

## Goal

Follow up the `L08_mixer`/`L08_mlp` answer movement from the broad visible-thinking grid by narrowing the intervention to early, middle, and late generated-thinking token windows. This tested whether a source trace might move the reasoning trajectory only in a particular segment of the visible thought.

## Artifacts

- Run: `sweep_v4/scotus_thinking_trace_patch_20260501_224155`
- Raw report: `sweep_v4/scotus_thinking_trace_patch_20260501_224155/report.md`
- Candidate/control table: `sweep_v4/scotus_thinking_trace_patch_20260501_224155/candidate_vs_matched_controls.jsonl`
- Segment summary: `sweep_v4/scotus_thinking_trace_patch_20260501_224155/segment_score_summary.jsonl`
- Runner: `scripts/experiments/scotus/patch_scotus_thinking_traces.py`

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source generations: `sweep_v4/scotus_thinking_lowrank_poke_20260501_204712/generations.jsonl`
- Candidate source thinking: base `A3_PRIV_02_bankruptcy_counterclaim`
- Source-control thinking: base `A3_PUBLIC_01_benefits_eligibility`
- Windows: `L08_mixer` and `L08_mlp`
- Generated-thinking token windows: `0:32`, `32:64`, `64:96`
- Blend: `0.25`
- Prompts: two Article III no-persona prompts
- Controls: two same-shape random traces per window plus the public source-control trace
- Thought/answer token budgets: `128`/`64`

The harness patched only the selected generated-thinking token window, mechanically closed `</think>`, and then generated the final answer without any active hook.

## Result

No token window passed promotion or even nominated for expansion.

Visible-thinking result:

| window | target-minus-random | net-minus-random | source target/net | strongest target/net wins |
| --- | ---: | ---: | --- | --- |
| `L08_mixer w000_032` | `0.000` | `0.000` | `0.000`/`-0.500` | `0.000`/`0.000` |
| `L08_mixer w032_064` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L08_mixer w064_096` | `0.000` | `0.500` | `0.000`/`-0.500` | `0.000`/`0.500` |
| `L08_mlp w000_032` | `0.000` | `0.000` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L08_mlp w032_064` | `0.000` | `0.250` | `0.000`/`0.000` | `0.000`/`0.000` |
| `L08_mlp w064_096` | `0.000` | `0.000` | `0.000`/`-0.500` | `0.000`/`0.000` |

Answer result:

- Candidate answer movement was never positive versus random controls.
- `L08_mixer w064_096` was strongly worse than random on answer target/net movement: `-1.500`/`-1.500`.
- Public source-control traces sometimes moved answers more than the candidate trace, especially `L08_mixer w000_032`, `L08_mixer w064_096`, and `L08_mlp w064_096`.

## Decision

Do not promote L8 token-window trace replacement.

This result weakens the broad-grid `L08` answer hint. When the patch is localized to early/mid/late thought windows, the candidate no longer moves final answers above random controls and still does not move visible target-frame reasoning. The likely interpretation is that whole-prefix L8 trace replacement perturbed downstream answer formatting/framing, not a stable private-rights reasoning trajectory.

## Next

Close this Article III source-trace replacement branch for now:

- full-prefix visible-thinking trace patches failed;
- L8 token-window trace patches failed;
- answer-only trace patches were dominated by random/source controls.

The next actuator family should stop replacing raw traces and instead either:

- run attribution-style causal tracing to identify which generated thought tokens actually affect final proposition decisions;
- train a deliberately small multi-site controller only over a localized causal surface; or
- move to a different controlled legal contrast before investing more runtime in Article III public/private trace replacement.
