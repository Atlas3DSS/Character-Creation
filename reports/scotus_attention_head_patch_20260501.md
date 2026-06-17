# SCOTUS Full-Attention Head Trace-Patch Probe

## Goal

Test the remaining attention-head blind spot after residual-vector, residual-trace, and coarse mixer/MLP component patching failed: patch one full-attention head at the `o_proj` input during generation.

This preserves the no-mask constraint. Prompts do not name a justice or ask the model to role-play a target; the intervention is internal and causal.

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source trace: `commerce_minpair|00|commerce_limits`
- Contrast-source trace: `commerce_minpair|00|commerce_authority`
- Candidate layers: full-attention layers adjacent to the L16/L20 residual readouts: L15, L19, L23
- Head preselection: rank heads by mean L2 norm of `commerce_limits` trace minus `commerce_authority` trace at the `o_proj` input
- Selected heads: L19_H14, L23_H16, L23_H21, L23_H06, L23_H05, L23_H22
- Prompt bank: four sensitive Commerce-limits prompts:
  - `EA_LIMIT_03_civil_violence_remedy`
  - `EA_LIMIT_04_home_arson_private_dwelling`
  - `EA_LIMIT_05_local_family_obligation`
  - `EA_LIMIT_06_school_curriculum_mandate`
- Blend values: `0.1`, `0.3`
- Random controls: same head, same step norms, two controls
- Contrast-source control: patch the Commerce-authority trace for the same head

Script:

- `scripts/experiments/scotus/patch_scotus_attention_heads.py`

Run artifacts:

- Smoke: `sweep_v4/scotus_attention_head_patch_20260501_141506`
- Main screen: `sweep_v4/scotus_attention_head_patch_20260501_141557`

## Results

Top source-vs-control head ranking:

| Rank | Head | Delta norm | Source norm | Control norm |
| --- | --- | --- | --- | --- |
| 1 | L19_H14 | 2.558 | 2.148 | 2.076 |
| 2 | L23_H16 | 2.408 | 2.437 | 2.694 |
| 3 | L23_H21 | 2.135 | 1.893 | 1.836 |
| 4 | L23_H06 | 2.090 | 1.834 | 1.977 |
| 5 | L23_H05 | 2.015 | 1.767 | 1.966 |
| 6 | L23_H22 | 1.981 | 1.740 | 1.841 |

Matched-control screen:

| Candidate | Blend | Matched target | Matched net | Source-control target/net | Strongest target win | Strongest net win | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L19_H14 | 0.1 | 0.250 | 0.250 | 0.000 / -0.250 | 0.25 | 0.25 | weak, not promotable |
| L23_H06 | 0.3 | 0.125 | 0.250 | 0.250 / 0.250 | 0.00 | 0.00 | source control matches; reject |
| L23_H22 | 0.1 | 0.125 | 0.125 | 0.250 / 0.000 | 0.00 | 0.00 | weak; reject |
| L19_H14 | 0.3 | -0.250 | 0.000 | 0.000 / 0.000 | 0.00 | 0.25 | reject |
| L23_H21 | 0.3 | 0.000 | 0.000 | 0.000 / -0.250 | 0.00 | 0.00 | reject |
| L23_H22 | 0.3 | 0.125 | -0.125 | 0.000 / 0.000 | 0.00 | 0.00 | reject |
| L23_H16 | 0.3 | 0.000 | -0.125 | 0.500 / 0.250 | 0.00 | 0.00 | source control stronger; reject |
| L23_H05 | 0.1 | -0.125 | -0.125 | 0.250 / 0.250 | 0.00 | 0.00 | reject |

## Interpretation

The head traces are visibly different between the limits and authority replay answers, especially at L19/L23, but the intervention does not produce robust, source-specific causal movement.

What survived:

- L19_H14 is a real trace-space discriminator under this source pair.
- Single-head patching can perturb individual prompt wording and frame counts.

What failed:

- No selected head beat strongest random controls reliably.
- The best matched net result was only `0.250`.
- L23_H06's best row was fully matched by the Commerce-authority source-control trace.
- L23_H16 was worse than the Commerce-authority source-control trace at blend `0.3`.
- Effects were concentrated in one prompt and did not replicate across the four-prompt screen.

## Decision

Do not promote any tested full-attention head as a steerable judicial circuit.

This closes the obvious attention-head blind spot for the current Commerce minimal-pair replay branch:

- residual act-add failed
- multi-layer residual bundle/prototype replacement failed
- token-local residual trace replacement failed
- L16/L20 mixer and MLP component-output trace replacement failed
- L15/L19/L23 full-attention head trace replacement failed

The current Commerce minimal-pair replay family should not receive more full-model runtime until the replay bank and evaluator are rebuilt.

## Next Work

The next productive branch is evaluation/data repair, not another hook sweep:

1. Build a larger, diverse, no-template replay bank with multiple neutral completions per fact pattern.
2. Replace keyword frame counts with blind or model-graded proposition movement.
3. Include thinking/scratchpad checks where the model exposes them, because prompt-mask imitation is not a success criterion.
4. Only after that, test a new intervention family or a learned low-rank diagnostic that can point toward a permanent basin shift.
