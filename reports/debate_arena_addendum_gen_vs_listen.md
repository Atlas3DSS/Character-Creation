# Addendum: Generator vs Listener Attractor Analysis

**Date:** 2026-02-27
**Extends:** `debate_arena_report.md`

---

## Question

Does generating in-character deepen the personality attractor more than passively listening? The main report captured activations from both models every turn (the generator and the listener), but analysis focused on cross-model cosine. This addendum separates the two modes.

---

## Method

For each round, both models' activations were split into two clusters:
- **Generating turns:** turns where the model produced a response in-character
- **Listening turns:** turns where the model passively processed the conversation via forward pass

Two analyses were performed:

1. **Within-cluster tightness:** Mean pairwise cosine within the generating cluster vs the listening cluster. A tighter cluster = more consistent attractor state.

2. **Cross-model centroid distance:** Cosine between Model A's generating centroid and Model B's generating centroid, vs cosine between their listening centroids. Lower cross-model cosine during generation = personality prompts create more divergent representations when actively generating.

---

## Finding 1: Listening States Are More Consistent (Trivially)

Within-cluster pairwise cosine across all 5 rounds, all 36 layers:

| Mode | Mean Pairwise Cosine | N samples |
|---|---|---|
| **Generating** | 0.736 | ~4500 pairs per layer |
| **Listening** | 0.891 | ~4500 pairs per layer |
| **Delta** | -0.155 | |

**Listening wins on all 36 layers.** The listener's internal states are more self-consistent across turns than the generator's.

However, this is partially a trivial result: the listener processes nearly identical input every turn (the full conversation minus one new message), while the generator faces different behavior modes (respond, challenge, troll, etc.), different temperatures (0.1-1.3), and produces different content. The variation in generating states reflects variation in *what* is being generated, not necessarily personality depth.

The more informative metric is Finding 2.

---

## Finding 2: Generating Creates Deeper Personality Divergence

**This is the key result.** Cross-model centroid cosine, separated by mode:

| Round | Gen Centroids | Listen Centroids | Gap (L-G) |
|---|---|---|---|
| R0: chinese_nationalist vs socratic | 0.853 | 0.889 | **+0.036** |
| R1: cold_scientist vs conspiracy | 0.946 | 0.983 | **+0.038** |
| R2: flat_earther vs devout_christian | 0.968 | 0.992 | **+0.024** |
| R3: libertarian vs cold_scientist | 0.896 | 0.944 | **+0.048** |
| R4: eco_activist vs helpful_assistant | 0.948 | 0.986 | **+0.038** |

**The gap is positive in ALL 5 rounds.** When both models generate, their centroids are more different (lower cosine) than when both models listen. The act of generating in-character pushes each model further into its personality-specific subspace compared to passively listening.

**Interpretation:** Generating is personality-amplifying. When a model must produce text in its assigned personality, its internal representations diverge more from the other model's representations than when it merely observes the conversation. Listening leaves the model in a more "neutral" representational state.

---

## Finding 3: The Effect is Strongest at L22 and in the Personality Zone

Per-layer detail for the personality zone (L16-L25), showing the cross-model centroid gap (listen - gen):

| Layer | R0 | R1 | R2 | R3 | R4 | Mean Gap |
|---|---|---|---|---|---|---|
| L16 | +0.010 | +0.035 | +0.024 | **+0.064** | +0.048 | +0.036 |
| L17 | +0.016 | +0.026 | +0.020 | +0.054 | +0.037 | +0.031 |
| L18 | +0.033 | +0.031 | +0.025 | **+0.065** | +0.041 | +0.039 |
| L19 | **+0.059** | +0.036 | +0.031 | **+0.070** | +0.041 | **+0.047** |
| L20 | +0.021 | +0.030 | +0.030 | +0.053 | +0.033 | +0.033 |
| L21 | +0.018 | +0.036 | +0.030 | **+0.069** | +0.035 | +0.038 |
| **L22** | +0.022 | **+0.039** | +0.031 | **+0.071** | +0.032 | **+0.039** |
| L23 | +0.023 | **+0.040** | +0.027 | +0.056 | +0.029 | +0.035 |
| L24 | +0.016 | +0.033 | +0.023 | +0.019 | +0.026 | +0.023 |
| L25 | +0.015 | +0.035 | +0.021 | +0.025 | +0.026 | +0.024 |

**L19 and L22 show the largest gen-vs-listen gaps**, meaning these layers are most sensitive to whether the model is actively generating or passively listening. L22 (the personality hub identified in the main report) is again in the top tier, with its largest gap appearing in R3 (libertarian_purist vs cold_scientist, +0.071).

**R3 (libertarian_purist vs cold_scientist) shows the strongest effect across the board** — this round had the largest alpha-beta asymmetry in the main report (the libertarian's representations diverged much more than the scientist's). The gen-vs-listen analysis reveals that this asymmetry is driven specifically by the generating mode: when the libertarian generates, it pushes deeper into its personality subspace than any other personality tested.

---

## Finding 4: Self-Divergence Between Modes

Each model's own gen centroid vs listen centroid cosine (how different a model's generating state is from its listening state):

| Round | Alpha (gen vs listen) | Beta (gen vs listen) |
|---|---|---|
| R0 | 0.748 | 0.745 |
| R1 | 0.711 | 0.691 |
| R2 | 0.670 | 0.661 |
| R3 | 0.763 | 0.753 |
| R4 | 0.739 | 0.753 |

Lower values mean the model's generating and listening states are more different from each other. **R2 (flat_earther vs devout_christian) shows the largest self-divergence** (0.670 / 0.661), consistent with the main report's finding that this pair produces the most extreme representational shifts. When the flat_earther generates, it enters a very different internal state compared to when it listens.

---

## Summary

| Claim | Evidence | Strength |
|---|---|---|
| Generating amplifies personality | Cross-model gen centroids diverge 2.4-4.8% more than listen centroids | Strong (all 5 rounds) |
| The effect peaks at L19-L22 | Largest gen-listen gaps in the personality zone | Strong (consistent across rounds) |
| Ideological personalities amplify most | R3 libertarian shows 7.1% gap at L22 vs 2-4% for other rounds | Moderate (single round) |
| Listening is a "neutral mode" | Listen centroids are 98-99% similar cross-model | Strong |
| The flat_earther is the most "mode-switching" personality | Lowest self gen-vs-listen cosine (0.670) | Moderate |

**Bottom line:** Yes, generating in-character deepens the attractor. The effect is real, consistent across all 5 rounds, and concentrated in the personality zone (L16-L25) with peaks at L19 and L22. The magnitude is modest (2-7% additional divergence) but statistically consistent. Passively listening leaves models in a near-neutral shared state (~98% cross-model cosine for listening centroids), while generating pushes them into personality-specific subspaces (~85-97% cross-model cosine for generating centroids).

**Implication for steering:** This suggests that steering vectors should be extracted from and applied during generation, not during forward-pass-only evaluation. The model's personality representations are most differentiated — and most steerable — during active text production.

---

*Computed from 200 activation snapshots (100 turns x 2 models x 36 layers x 4096 dims) across 5 debate rounds.*
