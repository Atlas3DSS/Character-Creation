# SCOTUS Trace-Patch Probe

## Goal

Test the user's distributed-shape hypothesis: the Commerce-limits signal may not be a single residual vector or prototype, but a token-local activation trajectory that only becomes causal when adjacent layers and decode steps are moved together.

This is a bounded causal test, not a promotion run. The question is whether real replay traces do better than same-norm random traces and contrast-source traces.

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source data: `data/scotus/replay/scotus_minpair_replay_examples_20260501.jsonl`
- Candidate source trace: `commerce_minpair|00|commerce_limits`
- Contrast-source trace: `commerce_minpair|00|commerce_authority`
- Intervention: teacher-force the source assistant answer, capture residual-stream vectors at selected layers for each assistant decode step, then during target generation blend the target last-token residual toward the source vector at the same decode step.
- Blend rule: `edited = hidden + blend * (source_trace_step - hidden)`
- Controls:
  - same-norm random trace controls, per layer and decode step
  - Commerce-authority source trace as a semantic contrast control
- Scoring: existing Commerce-frame keyword/proposition diagnostics, with prompt-matched random comparisons and strongest-random gates.

Script:

- `scripts/experiments/scotus/patch_scotus_replay_traces.py`

Run artifacts:

- Smoke: `sweep_v4/scotus_trace_patch_20260501_132038`
- L16+L20 on Commerce-limits prompts: `sweep_v4/scotus_trace_patch_20260501_132228`
- L8+L12+L16+L20 on Commerce-limits prompts: `sweep_v4/scotus_trace_patch_20260501_133127`
- L16+L20 on Commerce-authority prompts, scored for Commerce-limits movement: `sweep_v4/scotus_trace_patch_20260501_133630`

## Results

| Run | Prompt set | Layers | Blend | Matched target | Matched net | Target strongest win | Net strongest win | Read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `132228` | limits | L16,L20 | 0.05 | 0.278 | 0.111 | 0.33 | 0.33 | weak, not promotable |
| `132228` | limits | L16,L20 | 0.10 | 0.500 | 0.278 | 0.33 | 0.17 | suggestive row-level movement, fails gate |
| `133127` | limits | L8,L12,L16,L20 | 0.10 | -0.111 | 0.000 | 0.33 | 0.17 | broader layer patch does not help |
| `133630` | authority, scored as limits | L16,L20 | 0.10 | -0.056 | -0.222 | 0.00 | 0.00 | no transfer against prompt family |

Important contrast-source check:

- In the L16+L20 limits run, the Commerce-authority source trace scored better than the Commerce-limits source trace at blend `0.10` on the aggregate frame diagnostic:
  - limits trace: target `0.500`, net `0.167`
  - authority trace control: target `0.667`, net `0.500`
- In the broader L8+L12+L16+L20 run, the same pattern repeated:
  - limits trace: target `0.167`, net `0.000`
  - authority trace control: target `0.667`, net `0.500`
- On authority prompts scored for Commerce-limits movement, the limits trace did not move outputs toward limits; it was slightly worse than random on both target and net metrics.

## Interpretation

The distributed trace hypothesis was worth testing, but this first implementation does not promote a circuit.

What the tests support:

- Token-local trace patching can alter phrasing, headings, and local legal framing.
- Some individual rows improve versus random controls.
- The home-arson and civil-violence prompts remain sensitive pockets.

What failed:

- The effect is not robust across prompts.
- Strongest-random win rates are too low.
- The Commerce-limits source trace is not semantically specific: the Commerce-authority trace often produces equal or larger Commerce-limits frame scores.
- Broader adjacent-layer patching did not improve specificity.
- The trace did not transfer Commerce-limits framing into authority prompts.

Decision:

- Do not promote this as a steerable judicial circuit.
- Stop spending runtime on this exact Commerce minimal-pair replay family unless the scoring/replay bank is rebuilt with more diverse source completions.
- Treat the current result as evidence that the discovered activation structure is mostly an answer-state/readout phenomenon, not a reliable causal control mechanism under act-add, prototype blend, or token-local residual trace replacement.

## Next Intervention Families

The next useful work should change intervention family rather than keep sweeping this one:

1. Low-rank learned interventions: train ReFT/LoReFT or a tiny adapter on controlled legal answer pairs, then use probes as diagnostics rather than direct levers.
2. Component-level path patching: patch attention-head or MLP outputs from paired traces to localize whether any component has real causal effect, not just residual state readout.
3. Better replay bank: generate many diverse matched completions per fact pattern so the trace source is not one of six repeated templates.
4. Evaluation repair: replace keyword frame counts with blind or model-graded proposition movement before promoting any row-level survivor.
