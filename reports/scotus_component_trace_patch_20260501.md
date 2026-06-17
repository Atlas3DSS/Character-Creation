# SCOTUS Component Trace-Patch Probe

## Goal

Test a different intervention family after residual-vector, prototype, and residual-trace patching failed: patch token-mixer or MLP component outputs from paired replay traces, one component at a time.

This keeps the "no mask" constraint: prompts do not name a justice or ask the model to role-play a target. The intervention is internal and causal; it is not prompt-only steering.

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source trace: `commerce_minpair|00|commerce_limits`
- Contrast-source trace: `commerce_minpair|00|commerce_authority`
- Components: token mixer and MLP outputs
- Layers screened: L16 and L20
- Prompt bank: Commerce-limits prompts, focused on the four sensitive prompts from prior runs:
  - `EA_LIMIT_03_civil_violence_remedy`
  - `EA_LIMIT_04_home_arson_private_dwelling`
  - `EA_LIMIT_05_local_family_obligation`
  - `EA_LIMIT_06_school_curriculum_mandate`
- Blend values: `0.1`, `0.3`
- Random controls: same layer, same component, same source-trace step norms

Script:

- `scripts/experiments/scotus/patch_scotus_component_traces.py`

Run artifacts:

- Smoke: `sweep_v4/scotus_component_trace_patch_20260501_134936`
- Main screen: `sweep_v4/scotus_component_trace_patch_20260501_135028`

## Results

| Candidate | Blend | Matched target | Matched net | Source-control target/net | Strongest target win | Strongest net win | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L20 MLP | 0.3 | 0.625 | 0.625 | 0.000 / -0.250 | 0.25 | 0.25 | best relative row, not promotable |
| L20 mixer | 0.3 | 0.250 | 0.375 | 0.250 / 0.250 | 0.00 | 0.00 | source control matches; reject |
| L16 MLP | 0.1 | 0.500 | 0.250 | 0.250 / 0.250 | 0.25 | 0.25 | weak and source control matches net; reject |
| L20 MLP | 0.1 | 0.125 | 0.250 | 0.250 / 0.000 | 0.00 | 0.25 | weak; reject |
| L16 MLP | 0.3 | 0.000 | -0.125 | 0.000 / -0.250 | 0.00 | 0.00 | reject |
| L20 mixer | 0.1 | -0.250 | -0.125 | -0.250 / -0.500 | 0.00 | 0.00 | reject |
| L16 mixer | 0.1 | -0.125 | -0.250 | 0.000 / 0.000 | 0.00 | 0.00 | reject |
| L16 mixer | 0.3 | -0.500 | -0.750 | 0.250 / 0.000 | 0.00 | 0.00 | reject |

The apparent L20 MLP `0.3` winner is not strong enough:

- Candidate absolute target/net deltas were only `0.250` / `-0.250`.
- The positive matched net came from random controls being worse (`-0.875` net), not from a reliable positive target shift.
- Strongest-random target/net win rates were both `0.25`.
- Only four prompts and two random controls were used, so this is a screen, not a promotion.

## Decision

Do not promote any L16/L20 mixer or MLP component as a steerable judicial circuit.

This result narrows the failure:

- Whole residual act-add failed.
- Multi-layer residual bundle/prototype replacement failed.
- Token-local residual trace replacement failed.
- L16/L20 token-mixer and MLP output trace patching failed the random/source-control gates.

The current Commerce minimal-pair replay family is now mostly exhausted as a direct internal-intervention source.

## Constraint Update

The project should preserve the no-mask goal:

- Prompt-only role-play is not a solution.
- The target state is a model whose internal reasoning basin shifts, including any `<thinking>`/scratchpad-style reasoning, not a model that reasons about how a named persona would answer.
- LoRA/ReFT/SFT should be treated as last-resort or diagnostic tools unless they are explicitly used to find or create a durable basin shift that can later be made permanent.

## Next Work

The next useful work is not another local sweep over this same replay trace.

Prioritize one of:

1. Rebuild the replay bank with many diverse completions per fact pattern and no repeated answer templates.
2. Replace keyword frame scoring with blind or model-graded proposition movement.
3. Test permanent or semi-permanent basin-shift mechanisms only after the eval is repaired, with no persona prompting and with reasoning-trace checks where available.
4. If doing more circuit localization, move from coarse component outputs to a pre-registered head/MLP-neuron path-patching screen around nearby full-attention layers, not another residual replay-vector sweep.
