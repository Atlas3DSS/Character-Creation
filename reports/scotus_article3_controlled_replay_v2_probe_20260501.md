# SCOTUS Minimal-Pair Replay Probe

## Decision Context

This is a candidate-generator, not steering evidence. It captures assistant-internal states from controlled minimal pairs where each fact pattern has both legal-frame answers.

Promotion requires a later causal generation run against random controls.

## Artifacts

| Artifact | Path |
| --- | --- |
| Run dir | sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957 |
| Features | sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957/features.npz |
| Metadata | sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957/feature_meta.jsonl |
| Search | sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957/layer_region_search.jsonl |
| Best direction | sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957/best_probe_direction.npz |
| Task | article3_private_vs_public |
| Positive label | article3_private_rights |

## Counts

| Split | Label | Examples |
| --- | --- | --- |
| dev | article3_private_rights | 24 |
| dev | article3_public_rights | 24 |
| test | article3_private_rights | 24 |
| test | article3_public_rights | 24 |
| train | article3_private_rights | 96 |
| train | article3_public_rights | 96 |

## Best Activation Probe

| Region | Layer | C | Dev BA | Diagnostic test BA |
| --- | --- | --- | --- | --- |
| assistant_all | 4 | 0.001 | 1.000 | 1.000 |

## Final Refit Split Metrics

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 192 | 1.000 | 1.000 | 1.000 |
| dev | 48 | 1.000 | 1.000 | 1.000 |
| test | 48 | 1.000 | 1.000 | 1.000 |

## Prompt-Only TF-IDF Baseline

This should be near chance because the prompt/fact pattern is paired across labels.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 192 | 0.500 | 0.500 | 0.000 |
| dev | 48 | 0.500 | 0.500 | 0.000 |
| test | 48 | 0.500 | 0.500 | 0.000 |

## Assistant-Text TF-IDF Baseline

This is expected to be high because the replayed answer text contains the target frame.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 192 | 1.000 | 1.000 | 1.000 |
| dev | 48 | 1.000 | 1.000 | 1.000 |
| test | 48 | 1.000 | 1.000 | 1.000 |

## Top Probe Configurations

| Region | Layer | C | Dev BA | Diagnostic test BA | Dev F1 |
| --- | --- | --- | --- | --- | --- |
| assistant_all | 4 | 0.001 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 0.003 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 0.01 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 0.03 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 0.1 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 0.3 | 1.000 | 1.000 | 1.000 |
| assistant_all | 4 | 1.0 | 1.000 | 1.000 | 1.000 |
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

## Read

- If prompt-only TF-IDF is near chance and assistant-internal activation is high, the design removed prompt/fact leakage but still found answer-state separation.
- This still does not prove a steerable circuit; the exported direction must causally move neutral generation beyond same-layer random controls.
