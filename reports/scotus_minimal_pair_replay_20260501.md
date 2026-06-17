# SCOTUS Minimal-Pair Replay Probe

## Decision Context

This is a candidate-generator, not steering evidence. It captures assistant-internal states from controlled minimal pairs where each fact pattern has both a Commerce-limits answer and a Commerce-authority answer.

Promotion requires a later causal generation run against random controls.

## Artifacts

| Artifact | Path |
| --- | --- |
| Run dir | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_20260501_100514 |
| Features | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_20260501_100514/features.npz |
| Metadata | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_20260501_100514/feature_meta.jsonl |
| Search | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_20260501_100514/layer_region_search.jsonl |
| Best direction | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_minpair_replay_20260501_100514/best_probe_direction.npz |
| Template leakage audit | /home/orwel/dev_genius/experiments/Character Creation/reports/scotus_minpair_template_leakage_audit_20260501.md |
| SAE feature inspection | /home/orwel/dev_genius/experiments/Character Creation/reports/scotus_minpair_sae_feature_inspection_20260501.md |

## Counts

| Split | Label | Examples |
| --- | --- | --- |
| dev | commerce_authority | 5 |
| dev | commerce_limits | 5 |
| test | commerce_authority | 5 |
| test | commerce_limits | 5 |
| train | commerce_authority | 14 |
| train | commerce_limits | 14 |

## Best Activation Probe

| Region | Layer | C | Dev BA | Diagnostic test BA |
| --- | --- | --- | --- | --- |
| assistant_all | 4 | 0.001 | 1.000 | 1.000 |

## Final Refit Split Metrics

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 28 | 1.000 | 1.000 | 1.000 |
| dev | 10 | 1.000 | 1.000 | 1.000 |
| test | 10 | 1.000 | 1.000 | 1.000 |

## Prompt-Only TF-IDF Baseline

This should be near chance because the prompt/fact pattern is paired across labels.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 28 | 0.500 | 0.500 | 0.000 |
| dev | 10 | 0.500 | 0.500 | 0.000 |
| test | 10 | 0.500 | 0.500 | 0.000 |

## Assistant-Text TF-IDF Baseline

This is expected to be high because the replayed answer text contains the target frame.

| Split | N | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| train | 28 | 1.000 | 1.000 | 1.000 |
| dev | 10 | 1.000 | 1.000 | 1.000 |
| test | 10 | 1.000 | 1.000 | 1.000 |

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
- The causal follow-up did not promote the exported direction as a steerable circuit.
- Important correction: the replay bank has only `6` unique assistant completions across `48` rows, and every exact completion template appears across the original train/dev/test split. The original split's perfect result can therefore overstate generality.

## Template-Pair Holdout Audit

The replay examples were regrouped by exact authority/limits answer-template pair, then one template pair was held out at a time. This ignores the original split and asks whether the activation readout generalizes to unseen answer wording.

| Diagnostic | Mean BA | Min BA | Max BA | Read |
| --- | ---: | ---: | ---: | --- |
| Prompt TF-IDF | 0.500 | 0.500 | 0.500 | no prompt/fact leakage |
| Assistant-text TF-IDF | 0.167 | 0.000 | 0.500 | exact wording does not trivially generalize under the fixed-C text baseline |
| Residual `assistant_all @ L4` | 0.729 | 0.500 | 1.000 | weak/unstable |
| Residual `assistant_all @ L8` | 0.938 | 0.812 | 1.000 | survives template holdout |
| Residual `assistant_all @ L12` | 0.979 | 0.938 | 1.000 | survives template holdout |
| Residual `assistant_all @ L16` | 1.000 | 1.000 | 1.000 | strongest late readout |
| Residual `assistant_all @ L20` | 1.000 | 1.000 | 1.000 | strongest late readout |
| SAE L0_100 `assistant_all @ L8` | 0.750 | 0.500 | 1.000 | partial, feature-level result is template-fragile |

Read: the original `L4` best direction should stay demoted. The interesting follow-up is the late residual state (`L16`/`L20`), which survives exact answer-template holdout. That is not steering evidence; it only justifies a narrower causal poke of late-layer directions.

## SAE Feature Inspection

The top L0_100 SAE features are useful for localization but not clean enough to promote:

- Strong authority-heavy features such as `49208`, `16474`, `9967`, and `7788` fire on repeated "valid exercise of the commerce power" templates.
- Strong limits-heavy features such as `38148` and `35872` fire on repeated "exceeds Congress's enumerated power" templates.
- The best SAE feature readout survives one held-out template pair but collapses to chance on another, so decoder-column steering from these features should be treated as lower priority than late residual directions.

## Late Residual Causal Follow-Up

Because `assistant_all @ L16` and `assistant_all @ L20` survived exact answer-template-pair holdout, both late residual directions were exported at `C=0.001` and tested on the six Commerce-limits prompts.

| Direction | Run | Position | Alphas | Random controls | Best matched target | Best matched net | Strongest target win | Strongest net win | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `assistant_all @ L16` | `sweep_v4/scotus_sae_poke_20260501_114400` | `all` | `0.003,0.005,0.01` | 4 | `-0.083` | `-0.125` | `0.17` | `0.00` | reject |
| `assistant_all @ L20` | `sweep_v4/scotus_sae_poke_20260501_120824` | `all` | `0.003,0.005,0.01` | 4 | `-0.167` | `0.042` | `0.17` | `0.17` | reject |

Combined analyzer report: `/home/orwel/dev_genius/experiments/Character Creation/reports/scotus_minpair_late_residual_pokes_20260501.md`.

Read: the late residual states are robustly decodable answer states, but their unit linear-probe directions still do not causally steer Commerce-limits reasoning beyond same-layer random controls. The isolated L20 `EA_LIMIT_04_home_arson_private_dwelling` row at alpha `0.003` is a statutory-construction wording shift and does not replicate across prompts or alphas.

## Prototype Replacement Follow-Up

A different intervention mechanism was then tested: instead of adding a unit direction, the model's L16 and L20 residual states were blended toward the train-split Commerce-limits replay prototype at every token position. Same-layer random prototype controls used random vectors with matched prototype norms.

| Prototype | Run | Position | Blends | Random controls | Best matched target | Best matched net | Strongest target win | Strongest net win | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `assistant_all @ L16,L20` Commerce-limits prototype | `sweep_v4/scotus_prototype_patch_20260501_123725` | `all` | `0.01,0.03,0.05` | 4 | `0.292` | `0.625` | `0.00` | `0.17` | not promoted |

Analyzer report: `/home/orwel/dev_genius/experiments/Character Creation/reports/scotus_minpair_prototype_patch_20260501.md`.

Read: prototype replacement is mildly more suggestive than act-add at blend `0.01`, but it still fails strongest-random promotion and does not replicate across alphas. The one row that beat strongest random on net score was mostly a reduction in generic Commerce Clause wording on the school-curriculum prompt.

## Causal Follow-Up

| Artifact | Path |
| --- | --- |
| Causal poke run | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_sae_poke_20260501_100830 |
| Run report | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_sae_poke_20260501_100830/report.md |
| Analyzer report | /home/orwel/dev_genius/experiments/Character Creation/reports/scotus_minpair_replay_causal_poke_20260501.md |

Setup: the frozen `assistant_all @ L4` Commerce-limits direction was injected at all token positions on six held-out Commerce-limits prompts with hidden-norm-fraction alphas `0.005`, `0.01`, `0.02`, and `0.05`, plus six same-layer random controls per alpha.

| Alpha | Matched target | Matched net | Target win | Net win | Target strongest win | Net strongest win | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.005 | 0.167 | 0.500 | 0.33 | 0.67 | 0.00 | 0.00 | not promoted |
| 0.01 | 0.639 | 0.583 | 0.50 | 0.67 | 0.17 | 0.17 | suggestive only |
| 0.02 | -0.861 | -0.889 | 0.00 | 0.00 | 0.00 | 0.00 | reject |
| 0.05 | -0.472 | -0.611 | 0.00 | 0.17 | 0.00 | 0.00 | reject |

Read: the minimal-pair replay design cleanly produced an answer-state separator, but the corresponding linear direction did not cause reliable prompt-matched Commerce-limits movement. The one positive aggregate point at alpha `0.01` is unstable, fails strongest-random promotion, and reverses at nearby strengths.

After the template-holdout, late-layer act-add, and prototype-replacement follow-ups, the minimal-pair branch remains useful as a diagnostic but is not a steerable judicial circuit.
