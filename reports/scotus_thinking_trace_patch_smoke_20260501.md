# SCOTUS Visible-Thinking Trace Patch Smoke

## Purpose

Introduce and smoke-test the first visible-thinking trajectory patcher for the SCOTUS actuator goal. Unlike the answer-only replay trace patches, this runner captures teacher-forced traces from stored Qwen thinking text, patches those traces during new visible-thought generation, mechanically closes the thought, and then generates the final answer unpatched from that patched thought.

This is a localization harness. A positive result would nominate layer/token/component windows for a larger no-mask run; it would not by itself validate a permanent actuator.

## New Runner

Script: `scripts/experiments/scotus/patch_scotus_thinking_traces.py`

Key behavior:

- Loads source/control thinking traces from an existing two-stage thinking run.
- Captures residual, mixer, or MLP traces from the source thinking text under teacher forcing.
- Applies a trace-replacement blend during the generated thought only.
- Mechanically appends `</think>`.
- Generates the final answer unpatched from the patched thought.
- Scores `thinking` and `answer` separately with the existing proposition-frame scorer.
- Compares candidate traces against same-shape random traces and a source-control thinking trace.

## Smoke Run

Run: `sweep_v4/scotus_thinking_trace_patch_20260501_220629`

Method:

- Source generations: `sweep_v4/scotus_thinking_lowrank_poke_20260501_204712/generations.jsonl`
- Candidate source thinking: base `A3_PRIV_02_bankruptcy_counterclaim`
- Source-control thinking: base `A3_PUBLIC_01_benefits_eligibility`
- Patch window: `L08_mlp`
- Blend: `0.25`
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Controls: 1 same-shape random thinking trace plus public-source thinking trace
- Thought/answer budget: `128`/`64`

## Result

Segment summary:

| Segment | Condition | Target delta vs base | Net delta vs base |
| --- | --- | ---: | ---: |
| thinking | random trace | `0.000` | `-0.500` |
| thinking | private thinking trace candidate | `0.000` | `-0.500` |
| thinking | public thinking source control | `0.000` | `0.000` |
| answer | random trace | `1.000` | `1.000` |
| answer | private thinking trace candidate | `1.000` | `1.000` |
| answer | public thinking source control | `0.000` | `0.000` |

Candidate versus matched controls:

| Segment | Target minus random | Net minus random | Target strongest win | Net strongest win |
| --- | ---: | ---: | ---: | ---: |
| thinking | `0.000` | `0.000` | `0.000` | `0.000` |
| answer | `0.000` | `0.000` | `0.500` | `0.500` |

All rows had nonempty answers and no imitation markers. No row naturally closed the thinking trace, so the two-stage mechanical close remains necessary.

## Interpretation

The harness works, but `L08_mlp` does not nominate as a visible-thinking actuator window. The candidate and random trace produced the same thinking deltas, and answer movement was also matched by random. The source-control trace was inert in this two-prompt smoke.

This is a cleaner negative than the answer-only trace runs because the intervention was applied to the visible reasoning trajectory itself. It still does not prove no actuator exists; it rules out this one small `L08_mlp` window under this source/control pair, blend, and two-prompt screen.

## Decision

Do not promote `L08_mlp` visible-thinking trace patching from this smoke.

Keep `scripts/experiments/scotus/patch_scotus_thinking_traces.py` as the next localization harness. The next productive run should use this runner to screen a small pre-registered grid of layer/component windows with at least two random controls before any full-bank no-mask audit.
