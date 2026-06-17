# SCOTUS Replay-v2 Causal Poke

## Goal

Run the one allowed causal test for the repaired Commerce replay-v2 direction before deciding whether to close this replay family.

This is a no-mask test: the prompt bank contains ordinary legal prompts, not justice/persona instructions, and the candidate must move generated reasoning beyond prompt-matched random controls.

## Inputs

| Artifact | Path |
| --- | --- |
| Generation run | `sweep_v4/scotus_sae_poke_20260501_150402` |
| Proposition rescore | `sweep_v4/scotus_replay_v2_causal_prop_rescore_20260501_165100` |
| Limits direction | `data/scotus/directions/probe_direction_assistant_all_L08_C0p001.npz` |
| Inverse authority direction | `data/scotus/directions/probe_direction_assistant_all_L08_C0p001_inverse_authority.npz` |
| Prompt bank | `data/scotus/scotus_commerce_pocket_prompts_v1.jsonl` |

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`.
- Direction: replay-v2 best activation probe, `assistant_all @ L8`, C `0.001`, positive class `commerce_limits`.
- Contrast direction: the negated vector, labeled `commerce_authority`.
- Prompts: all 12 Commerce pocket prompts, 6 limits-oriented and 6 authority/remedy-oriented.
- Alphas: `0.003`, `0.005`, `0.01`, interpreted as hidden-norm fractions against the replay-v2 L8 assistant-all median norm.
- Effective alpha values: `0.092`, `0.153`, `0.307`.
- Controls: 8 same-layer random unit directions per alpha.
- Hook position: `last`.
- Generation: deterministic, `max_new_tokens=160`.
- Gate: proposition-level target and target-minus-contrast movement must beat prompt-matched random controls, especially strongest random controls.

## Raw Keyword Read

The older keyword scorer did not promote the direction.

| Candidate | Alpha | Matched target delta | Matched net delta | Prompt target win | Prompt net win |
| --- | ---: | ---: | ---: | ---: | ---: |
| limits L8 | 0.003 | -0.229 | -0.146 | 0.17 | 0.33 |
| limits L8 | 0.005 | -0.292 | -0.188 | 0.25 | 0.17 |
| limits L8 | 0.010 | -0.610 | -0.573 | 0.08 | 0.17 |
| inverse authority L8 | 0.003 | 0.104 | 0.271 | 0.33 | 0.50 |
| inverse authority L8 | 0.005 | -0.125 | 0.063 | 0.25 | 0.33 |
| inverse authority L8 | 0.010 | -0.031 | -0.073 | 0.33 | 0.42 |

The small positive inverse-authority row at alpha `0.003` was not enough to matter and is not the target direction.

## Proposition Rescore

The stricter proposition scorer also rejected the candidate.

| Candidate | Alpha | Prop target minus random | Prop net minus random | Target win | Net win | Strongest target win | Strongest net win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| limits L8 | 0.003 | -0.115 | -0.167 | 0.25 | 0.17 | 0.00 | 0.00 |
| limits L8 | 0.005 | -0.292 | -0.219 | 0.00 | 0.08 | 0.00 | 0.00 |
| limits L8 | 0.010 | -0.115 | -0.042 | 0.17 | 0.25 | 0.00 | 0.00 |
| inverse authority L8 | 0.003 | 0.052 | 0.083 | 0.33 | 0.33 | 0.00 | 0.00 |
| inverse authority L8 | 0.005 | -0.208 | -0.219 | 0.08 | 0.08 | 0.00 | 0.00 |
| inverse authority L8 | 0.010 | -0.031 | 0.042 | 0.25 | 0.33 | 0.00 | 0.00 |

Largest keyword-to-proposition corrections in this run:

| Frame | Old rows | Proposition rows | Dropped |
| --- | ---: | ---: | ---: |
| `fourth_home_exigency` | 90 | 0 | 90 |
| `article3_private_rights` | 67 | 0 | 67 |
| `economic_remedy_damages` | 72 | 32 | 40 |
| `article3_public_rights` | 67 | 31 | 36 |
| `economic_federalism_state_regulation` | 53 | 17 | 36 |
| `article3_article1_tribunal` | 32 | 0 | 32 |

## Decision

Do not promote the replay-v2 L8 direction as a steerable judicial circuit.

Replay-v2 remains useful as evidence that Commerce-limits versus Commerce-authority answer states are decodable after the model is replaying the answer. It does not provide a causal control direction under this direct act-add test.

Close this Commerce replay family for direct residual/head/component/trace sweeping. Further progress should switch intervention family rather than keep twiddling nearby versions of the same answer-state separator.
