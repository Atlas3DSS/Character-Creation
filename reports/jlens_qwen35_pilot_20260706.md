# Qwen3.5 J-Lens SCOTUS Pilot

Date: 2026-07-06

Run directory: `sweep_v4/jlens_qwen35_pilot_20260706_220040`

## Purpose

Run a gentle offline diagnostic over five archived SCOTUS directions to test whether Jacobian-lens transport gives any of them above-random output disposition. This was a diagnostic-only pass: no steering, no generation, and no promotion claims.

## Method

- Lens: Neuronpedia `qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt`
- Source model/readout: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Readout: Qwen RMSNorm (`model.language_model.norm.weight`) plus `lm_head.weight`
- Directions: five archived compact SCOTUS probe directions
- Random controls: 200 unit Gaussian directions per candidate, same layer and residual width
- Top-singular proxy: randomized SVD of each layer's `J_l`, ranks 32, 128, and 512
- Layer convention: project hooks and J-lens both use `layers[i]` output; no offset applied

## Results

| Direction | Layer | Gain percentile | Best SVD-proxy percentile | Legal top-30 hit | Result |
|---|---:|---:|---:|---|---|
| `probe_direction_assistant_all_L08_C0p001` | 8 | 68.5 | 26.5 | none | no-go |
| `probe_direction_assistant_all_L08_C0p001_inverse_authority` | 8 | 69.0 | 26.5 | none | no-go |
| `probe_direction_assistant_all_L16_C0p001` | 16 | 1.5 | 10.0 | none | no-go |
| `probe_direction_assistant_all_L20_C0p001` | 20 | 0.0 | 0.0 | none | no-go |
| `scotus_article3_controlled_replay_v2_assistant_all_L04_private_rights_20260501` | 4 | 0.0 | 12.0 | none | no-go |

## Interpretation

No candidate met the pilot criteria:

- gain percentile > 95
- at least one top-singular proxy percentile > 95
- legal/frame vocabulary in top-30 readout
- sign semantics that make sense under human review

This supports the working explanation that these archived directions were decodable but do not have clearly above-random J-transported output disposition under this screen. This is not proof that J-lens cannot help; it says these five archived directions are not good Phase B steering candidates as-is.

## Next Step

Do not start J-space steering from these pilot rows. If continuing, the next useful diagnostic is a broader offline inventory over the other compact and localization directions, especially late answer-state localization rows, before any generation or steering run.
