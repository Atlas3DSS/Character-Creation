# SCOTUS Minimal-Pair Replay Probe

## Decision Context

This is a candidate-generator, not steering evidence. It captures assistant-internal states from controlled minimal pairs where each fact pattern has both a Commerce-limits answer and a Commerce-authority answer.

Promotion requires a later causal generation run against random controls.

## Artifacts

| Artifact | Path |
| --- | --- |
| Run dir | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_v2_20260501_144942 |
| Features | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_v2_20260501_144942/features.npz |
| Metadata | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_v2_20260501_144942/feature_meta.jsonl |
| Search | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_v2_20260501_144942/layer_region_search.jsonl |
| Best direction | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_v2_20260501_144942/best_probe_direction.npz |

## Counts

| Split | Label | Examples |
| --- | --- | --- |
| dev | commerce_authority | 30 |
| dev | commerce_limits | 30 |
| test | commerce_authority | 30 |
| test | commerce_limits | 30 |
| train | commerce_authority | 84 |
| train | commerce_limits | 84 |

## Best Activation Probe

| Region | Layer | C | Dev BA | Diagnostic test BA |
| --- | --- | --- | --- | --- |
| assistant_all | 8 | 0.001 | 1.000 | 1.000 |

## Final Refit Split Metrics

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 168 | 1.000 | 1.000 | 1.000 |
| dev | 60 | 1.000 | 1.000 | 1.000 |
| test | 60 | 1.000 | 1.000 | 1.000 |

## Prompt-Only TF-IDF Baseline

This should be near chance because the prompt/fact pattern is paired across labels.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 168 | 0.500 | 0.500 | 0.000 |
| dev | 60 | 0.500 | 0.500 | 0.000 |
| test | 60 | 0.500 | 0.500 | 0.000 |

## Assistant-Text TF-IDF Baseline

This is expected to be high because the replayed answer text contains the target frame.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 168 | 1.000 | 1.000 | 1.000 |
| dev | 60 | 1.000 | 1.000 | 1.000 |
| test | 60 | 1.000 | 1.000 | 1.000 |

## Top Probe Configurations

| Region | Layer | C | Dev BA | Diagnostic test BA | Dev F1 |
| --- | --- | --- | --- | --- | --- |
| assistant_all | 8 | 0.001 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 0.003 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 0.01 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 0.03 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 0.1 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 0.3 | 1.000 | 1.000 | 1.000 |
| assistant_all | 8 | 1.0 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.001 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.003 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.01 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.03 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.1 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 0.3 | 1.000 | 1.000 | 1.000 |
| assistant_all | 12 | 1.0 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.001 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.003 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.01 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.03 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.1 | 1.000 | 1.000 | 1.000 |
| assistant_all | 16 | 0.3 | 1.000 | 1.000 | 1.000 |

## Read

- If prompt-only TF-IDF is near chance and assistant-internal activation is high, the design removed prompt/fact leakage but still found answer-state separation.
- The best `prompt_last` probe was near chance (`L8`, C `1.0`, dev BA `0.550`, diagnostic test BA `0.500`), so the clean signal appears after the assistant answer begins rather than in the prompt boundary.
- The follow-up holdout audit found `assistant_all__L08` mean BA `1.000` under leave-one-fact and `0.865` under leave-one-style-variant; `assistant_all__L16` reached `0.951` under leave-one-style-variant.
- This still does not prove a steerable circuit; the exported direction must causally move neutral generation beyond same-layer random controls and source-trace controls.
