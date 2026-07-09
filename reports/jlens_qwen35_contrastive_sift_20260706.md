# Qwen3.5 J-Lens Contrastive Sift

Date: 2026-07-06

## Question

Can we compare existing contrastive directions through the Qwen3.5 J-lens to see whether any candidate direction has a clean output-facing disposition before running causal steering?

## Compatibility Check

- `contrastive_data/activations/deltas_layer_09.pt` has shape `[57432, 4096]`, with matching 4096-wide SVD/profile artifacts. These are Qwen3-VL-8B-space and are not directly compatible with the 5120-wide Qwen3.5/27B J-lens.
- `qwen35_map/27b/connectome_zscores.pt` has shape `[20, 64, 5120]`. This is compatible with the Qwen3.5/27B J-lens.
- `qwen35_map/27b/spectral_analysis/{sarc,math}_activations.pt` each contain 64 layers of `[200, 5120]` activations. Their sarcasm-minus-math mean directions are also compatible.
- The cached J-lens covers layers `0..62`; layer `63` directions are skipped.

## New Diagnostic Script

Added:

`scripts/experiments/connectome/qwen35_jlens_contrastive_sift.py`

The script:

- loads the cached `neuronpedia/jacobian-lens` Qwen3.5-27B lens;
- scans archived 5120-wide directions with no generation;
- compares transported gain against isotropic random controls per layer;
- decodes top transported vectors through local Qwen RMSNorm + `lm_head`;
- writes `manifest.json`, `records.jsonl`, `top_candidates.json`, and `report.md`.

## Runs

Combined connectome plus spectral sarcasm-minus-math:

`sweep_v4/jlens_qwen35_contrastive_sift_20260706_221312`

- Directions requested: `1344`
- Directions scanned: `1323`
- Skipped: `21` directions at missing lens layer `63`
- Random controls: `64`
- Directions at or above 95th percentile: `876`

Connectome-only:

`sweep_v4/jlens_qwen35_contrastive_sift_20260706_221348`

- Directions requested: `1280`
- Directions scanned: `1260`
- Skipped: `20` directions at missing lens layer `63`
- Random controls: `256`
- Directions at or above 95th percentile: `794`

## Main Signal

The spectral sarcasm-minus-math mean directions dominate the combined run. The best candidate is:

- `Spectral: Sarcasm minus Math`, layer `48`
- Gain percentile: `100.0`
- Gain z: `39.05`
- Positive readout top tokens include: `glowing`, `shimmer`, `utterly`, `arrogant`, `smirk`, `disdain`, `bizarre`, `terrifying`
- Negative readout top tokens include arithmetic/calculation markers: `Calculation`, `Arithmetic`

Nearby layers `34`, `44`, `47`, `49`, `50`, and `51` show the same broad sign structure: positive transport has arrogance/smirk/disdain/bizarre/ridiculous-style tokens; negative transport leans toward calculation/arithmetic or punctuation/control tokens.

## Connectome-Only Signal

Top compatible connectome categories by best gain z:

| Category | Best layer | Best gain z | Notes |
|---|---:|---:|---|
| Language: EN vs CN | 62 | 41.10 | Clear language/tokenization axis; not useful for personality steering. |
| Tone: Sarcastic | 62 | 22.33 | High gain, but decoded as punctuation/noisy tokens, not clean sarcasm semantics. |
| Verbosity: Brief | 50 | 20.90 | Mostly end/control-token disposition. |
| Emotion: Joy | 61 | 20.31 | Clean joy/excitement tokens like sparkle/HUGE/amazing. |
| Tone: Formal | 49 | 16.57 | Clean formal-vocabulary direction vs casual words. |
| Reasoning: Analytical | 62 | 15.90 | Clean analytical/logical tokens vs affective tokens. |
| Tone: Polite | 50 | 15.73 | Clean polite/positive tokens vs profanity/negative tokens. |

## Interpretation

Yes, this comparison is useful. It found a coherent output-facing signal in the 27B spectral sarcasm-minus-math directions, especially around layers `48..51`, and showed that several connectome categories have interpretable readouts.

However, gain-vs-isotropic-random is too permissive: most structured contrastive directions beat random controls. Use this sift for candidate ranking, not promotion. The useful criterion is high gain plus coherent sign semantics in the decoded tokens.

## Recommended Next Step

Do a gentle causal smoke on the spectral sarcasm-minus-math candidates, not on the old 4096-wide 8B contrastive deltas.

Start with:

- layers `48`, `49`, `50`, and maybe `34` as a mid-layer comparison;
- very small alpha grid;
- 6-10 fixed nonlegal prompts plus 6 math/control prompts;
- promotion-ineligible smoke labeling;
- measure both style movement and math/control degradation.

This requires HuggingFace hooks or another hook-capable runtime. The current llama.cpp/vLLM-style dev-box servers on ports `8080` and `8181` are fine for baseline generation, but they cannot apply inference-time residual steering.
