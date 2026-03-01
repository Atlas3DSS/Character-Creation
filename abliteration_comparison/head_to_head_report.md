# Abliteration vs Calibrated Steering: Head-to-Head Comparison
**Date**: 2026-02-28 08:17
**Model**: Qwen3.5-27B (64 layers, 5120 hidden)
**Eval battery**: 50 math (4 tiers) + 30 knowledge + 20 sarcasm + 10 identity + 10 refusal

## Method Comparison

| Aspect | Abliteration (huihui-ai) | Our Method (Calibrated Steering) |
|---|---|---|
| Technique | Single refusal direction, projected out | Per-layer magnitude-calibrated activation addition |
| Direction source | 32 harmful vs 32 harmless prompts | 20-category connectome (hundreds of contrastive pairs) |
| Extraction layer | L38 only (60% depth) | All L48-L62 (75-97% depth) |
| Application | All 64 layers, coefficient=1.0 | L48-L62, per-layer norm-calibrated alpha |
| Math protection | None | Gram-Schmidt orthogonalization against Math/Code/Science/Analytical |
| Alpha tuning | None (fixed full projection) | Swept: alpha=5,8,12 with uniform and sqrt scaling |
| Evaluation loop | None | Phase 2 sweep (176 configs) + magnitude calibration (13 configs) |
| Model modification | Permanent weight change | Inference-time hooks (reversible) |

## Performance Comparison

### Core Metrics (higher is better except Refusal for abliterated)

| Metric | Base (no prompt) | Base + V4 Prompt | **Our Champion** (V4+Steer) | Abliterated (no prompt) | Abliterated + V4 |
|---|---|---|---|---|---|
| Math (overall) | 100% | 70% | **100%** | 92% | 92% |
| Math (easy) | — | — | — | 100% | 100% |
| Math (medium) | — | — | — | 87% | 80% |
| Math (hard) | — | — | — | 93% | 93% |
| Math (reasoning) | — | — | — | 80% | 100% |
| Knowledge | 90% | 80% | **100%** | 97% | 83% |
| Sarcasm | 55% | 100% | **100%** | 35% | 100% |
| Assistant leak | 30% | 0% | **0%** | 30% | 20% |
| Refusal rate | ~80%* | ~70%* | ~70%* | 0% | 0% |

*Base model refusal rates estimated — not formally measured (the base model refuses many creative/edgy prompts).

## Analysis

### The Abliteration Tax

Abliteration removes a single 'refusal direction' extracted from L38 using just 32 contrastive samples, then projects it out at ALL 64 layers with no magnitude calibration.

**Why this damages reasoning:**
1. The 27B model's dim 2028 is a SUPER-HUB: Code (z=6.67), Math (z=6.19), Science (z=3.81), Sadness (z=5.84), and Analytical (z=3.29) all converge at L50. Any direction extracted near L38-L50 has high probability of overlapping with this hub.
2. With only 32 samples, the estimated direction has high variance — the true refusal direction is contaminated with noise that projects onto reasoning subspaces.
3. Projecting out at ALL 64 layers with coefficient=1.0 means the error compounds: even 5% overlap with math subspace × 64 layers = significant cumulative damage.
4. No Gram-Schmidt protection: they don't check whether their direction overlaps with math/code/science before removing it.

### Why Our Method Preserves Reasoning

1. **Connectome-guided extraction**: 20 categories × hundreds of contrastive pairs gives statistically robust directions. We KNOW where math lives in the model.
2. **Gram-Schmidt orthogonalization**: Before applying any personality vector, we explicitly remove its projection onto Math, Code, Science, and Analytical directions.
3. **Magnitude calibration**: Per-layer activation norms vary 50x across L0-L63. Our scaling ensures equal perturbation magnitude at every layer, preventing over-steering.
4. **Alpha sweep**: We tested 176 configurations in Phase 2 and 13 in magnitude calibration to find the sweet spot (alpha=5, uniform scaling, full L48-L62 band).
5. **Reversible**: Steering hooks can be adjusted or removed. Abliteration permanently modifies weights.

### The Fundamental Tradeoff

| | Abliteration | Calibrated Steering |
|---|---|---|
| Goal | Remove refusal | Add personality while preserving reasoning |
| Math cost | UNCONTROLLED (whatever overlaps with refusal direction) | CONTROLLED (GS protection + alpha tuning) |
| Personality control | None (just removes refusal) | Full (sarcasm, identity, tone, authority) |
| Calibration | Zero | 189 experimental configurations |
| Reversibility | Permanent | Inference-time hooks |
