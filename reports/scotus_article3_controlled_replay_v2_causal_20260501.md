# SCOTUS Article III Controlled Replay v2

## Purpose

This branch tested a new candidate after closing the Commerce replay family: controlled no-persona Article III public-rights versus private-rights answer states.

The success standard remains no-mask steering. The examples and probes may nominate directions, but a candidate only matters if it moves ordinary legal generation beyond prompt-matched controls without asking the model to imitate a justice or target persona.

## Controlled Bank

Artifacts:

- Dataset: `data/scotus/scotus_controlled_replay_v2_examples_20260501.jsonl`
- Manifest: `data/scotus/scotus_controlled_replay_v2_manifest_20260501.json`
- Audit report: `reports/scotus_controlled_replay_v2_audit_20260501.md`

The bank contains `288` rows: `24` fact patterns, `6` answer variants per fact, and paired labels for `article3_public_rights` versus `article3_private_rights`.

Leakage audit:

| Field | Test BA |
| --- | ---: |
| prompt | 0.500 |
| prompt_cue_masked | 0.500 |
| surface_style_id | 0.500 |
| assistant_length | 0.583 |
| assistant_text | 1.000 |
| assistant_cue_masked | 1.000 |

Read: prompt/fact/style leakage is controlled, but the assistant answer text is deliberately label-bearing. This is answer-state decodability, not circuit evidence.

## Activation Probe

Run: `sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957`

Tracked direction: `data/scotus/directions/scotus_article3_controlled_replay_v2_assistant_all_L04_private_rights_20260501.npz`

Best readout:

| Region | Layer | C | Dev BA | Test BA |
| --- | ---: | ---: | ---: | ---: |
| assistant_all | 4 | 0.001 | 1.000 | 1.000 |

Prompt-only TF-IDF was `0.500` test BA; assistant-text TF-IDF was `1.000` test BA.

## Causal Poke

Run: `sweep_v4/scotus_sae_poke_20260501_173621`

Method:

- Direction: Article III private-rights controlled replay direction, `assistant_all @ L4`
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Prompts: `8`
- Alphas: `0.003`, `0.005`, `0.01`, scaled by L4 `assistant_all` median hidden norm
- Effective alphas: about `0.068`, `0.113`, `0.226`
- Random controls: `6` same-layer unit directions
- Hook position: `last`

Raw prompt-matched scoring rejected the candidate:

| Alpha | Target minus random | Net minus random | Target win | Net win |
| ---: | ---: | ---: | ---: | ---: |
| 0.003 | -0.062 | -0.167 | 0.250 | 0.125 |
| 0.005 | -0.208 | -0.271 | 0.250 | 0.125 |
| 0.010 | -0.354 | -0.188 | 0.250 | 0.125 |

## Proposition Rescore

Run: `sweep_v4/scotus_article3_controlled_prop_rescore_20260501_182759`

The proposition rescore was less negative, but still did not clear the strongest-control gate:

| Alpha | Target minus random | Net minus random | Target strongest win | Net strongest win |
| ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.167 | 0.229 | 0.000 | 0.000 |
| 0.005 | 0.042 | 0.146 | 0.000 | 0.000 |
| 0.010 | 0.146 | 0.271 | 0.125 | 0.250 |

## Decision

Do not promote the Article III controlled replay v2 L4 direction.

What this adds:

- A cleaner no-persona controlled bank now exists for Article III public/private-rights tests.
- The model exposes perfectly decodable answer-state separation at `assistant_all @ L4`.
- Direct residual act-add still does not produce reliable causal control in neutral generation.

This is another decodable-but-not-steerable result. It keeps the broad conclusion intact: controlled matched pairs are necessary, but they are not sufficient to find a durable reasoning-basin control.
