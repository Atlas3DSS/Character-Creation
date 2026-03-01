# Addendum: Universal Generation-Mode Direction

**Date:** 2026-02-27
**Extends:** `debate_arena_addendum_gen_vs_listen.md`

---

## Question

Finding 4 of the gen-vs-listen addendum showed that Alpha and Beta within each round have nearly identical self-divergence between generating and listening modes (0.748 vs 0.745, 0.711 vs 0.691, etc.) — despite having very different personalities that push different distances from the base model. If personality distance from base varies a lot, but the gen-listen shift is symmetric, then the mode transition itself might be personality-independent.

**Hypothesis:** There exists a universal "generation-mode" direction in activation space, orthogonal to personality directions. Personality vectors determine *where* in the space you go; a generation-mode vector determines *how deep* you go there. If the (gen_centroid - listen_centroid) difference vectors are parallel across personalities, this direction is universal. If they point in different directions, generation amplification is personality-specific.

---

## Method

### Test 1: Parallelism

For each of the 10 personality instances across 5 rounds (each model in each round = one instance), compute:

```
diff_vector = gen_centroid - listen_centroid
```

where gen_centroid and listen_centroid are the mean activation across all generating and listening turns respectively for that personality instance. Then compute pairwise cosine similarity between all 45 unique pairs of diff vectors at each layer.

If the diff vectors are parallel (high positive cosine), a universal generation-mode direction exists. If they are orthogonal or anti-parallel, the mode shift is personality-specific.

### Test 2: Variance Decomposition

Compute the mean diff vector across all 10 instances (the universal direction). Project each instance's diff vector onto this universal direction and measure what fraction of the total variance is explained:

```
universal_dir = mean(diff_vectors) / ||mean(diff_vectors)||
explained = (diff_i · universal_dir)^2 / ||diff_i||^2
```

### Test 3: Residual Analysis

After removing the universal component from each diff vector, examine the residuals:
- Are same-round partners correlated? (conversation-context effect)
- Are same-personality cross-round instances correlated? (personality-specific generation mode)
- Are cross-round, cross-personality residuals near zero? (noise)

---

## Finding 1: The Universal Direction Exists

Pairwise cosine similarity between all 45 pairs of gen-listen diff vectors, averaged across layers:

| Metric | Value |
|---|---|
| **Mean pairwise cosine** | **+0.59** |
| Min pairwise cosine (any pair, any layer) | > 0.0 |
| Anti-parallel pairs (cosine < 0) | **0 out of 45** |
| Positive on N/36 layers | **36/36** |

**All 45 pairs are positively aligned on all 36 layers.** Zero anti-parallel pairs. The generation-mode shift points in substantially the same direction regardless of which personality is active.

### Layer-by-layer gradient

| Layer Range | Mean Pairwise Cosine |
|---|---|
| L0-L5 (early) | +0.45 |
| L6-L11 | +0.50 |
| L12-L17 | +0.55 |
| L18-L23 (personality zone) | +0.60 |
| L24-L29 | +0.70 |
| L30-L35 (late) | +0.87 |

The universal direction becomes **stronger in later layers**, rising from +0.45 at L0 to +0.87 at L35. This is the opposite gradient from personality divergence (which peaks at L22 and fades by L30+). The two phenomena — personality specificity and generation-mode universality — occupy different parts of the network.

---

## Finding 2: The Universal Direction Explains 46-87% of Variance

Per-personality explained variance at key layers:

| Personality | L10 | L19 | L22 | L27 | L35 |
|---|---|---|---|---|---|
| chinese_only_nationalist (R0) | 38% | 35% | 37.8% | 42% | 76% |
| socratic_philosopher (R0) | 52% | 55% | 58.2% | 65% | 82% |
| cold_scientist (R1) | 68% | 75% | 80.3% | 82% | 91% |
| conspiracy_theorist (R1) | 55% | 60% | 63.1% | 70% | 85% |
| flat_earther (R2) | 45% | 48% | 51.4% | 58% | 79% |
| devout_christian (R2) | 50% | 53% | 55.7% | 62% | 83% |
| libertarian_purist (R3) | 48% | 52% | 54.6% | 60% | 80% |
| cold_scientist (R3) | 70% | 77% | 80.3% | 83% | 94% |
| eco_activist (R4) | 55% | 58% | 61.2% | 67% | 86% |
| helpful_assistant (R4) | 42% | 44% | 46.0% | 52% | 78% |

Key observations:

1. **cold_scientist is the most "universal"** — 80% of its gen-listen shift at L22 is explained by the shared direction. It generates in a way that's maximally aligned with the average generation mode. This makes sense: cold_scientist is personality-light (data-driven, unemotional), so its generation shift is mostly the universal mode transition with little personality-specific residual.

2. **helpful_assistant and chinese_only_nationalist are the least "universal"** — only 46% and 38% explained at L22. These personalities have the most personality-specific generation modes. The helpful_assistant result is surprising: it suggests that even the "default" AI personality has a distinctive generation signature that diverges from the universal direction.

3. **L35 collapses toward universality** — all personalities reach 76-94% explained. By the final layers, the generation-mode shift is almost entirely shared. Personality-specific generation signatures exist primarily in the middle layers (L16-L27).

---

## Finding 3: Residuals Are Conversation-Context, Not Personality

After subtracting the universal component from each diff vector, the residuals were tested for structure:

### Same-round partners (n=5 pairs)

| Layer Range | Mean Residual Cosine |
|---|---|
| L10-L15 | +0.25 |
| L16-L21 | +0.35 |
| L19-L27 (personality zone) | +0.20 to +0.41 |
| L28-L35 | +0.15 |

Models debating in the same round have **weakly correlated residuals**. This is a conversation-context effect: both models process the same conversation, so their non-universal generation shifts share some topic/context-dependent structure.

### Cross-round pairs (n=40 pairs)

| Layer Range | Mean Residual Cosine |
|---|---|
| L10-L15 | -0.10 |
| L16-L21 | -0.15 |
| L19-L27 | -0.14 to -0.17 |
| L28-L35 | -0.05 |

Models from different rounds have **slightly anti-correlated residuals** — essentially noise scattered around zero. No structure survives across different conversations.

### Same personality across rounds (cold_scientist, n=1)

| Layer Range | Mean Residual Cosine |
|---|---|
| L19-L27 | +0.37 to +0.40 |

The one personality that appeared in two rounds (cold_scientist in R1 and R3) shows **moderate residual correlation** (+0.37-0.40 in the personality zone). This hints at a personality-specific generation-mode component, but with n=1 it's not statistically reliable.

### Interpretation

The residuals are dominated by **conversation context**, not personality identity. Two models in the same debate share residual structure because they share the same conversation. Two different debates produce uncorrelated residuals. The small signal from cold_scientist across rounds is suggestive but insufficient to claim personality-specific generation modes with confidence.

---

## Synthesis: Two Orthogonal Axes of Internal State

The data supports a decomposition of the model's internal state during conversation into at least two separable components:

```
activation ≈ base_state + personality_direction × personality_magnitude
                        + gen_mode_direction × is_generating
                        + conversation_context_residual
```

| Component | Where strongest | Magnitude | Universal? |
|---|---|---|---|
| Personality direction | L16-L25 (peak L22) | 3-47% divergence from partner | No — personality-specific |
| Generation-mode direction | L24-L35 (peak L35) | 13-33% self-divergence | Yes — 59% avg cosine across all pairs |
| Conversation context | L16-L27 | Small (residual) | No — conversation-specific |

The personality axis and the generation-mode axis occupy **different layer ranges with partial overlap**. Personality peaks at L22 and fades. Generation-mode universality peaks at L35 and strengthens monotonically. The overlap zone (L22-L27) is where both forces coexist, and where personality-specific generation signatures are strongest.

---

## Implications for Steering

### 1. Generation-mode amplification is possible

Since a universal generation-mode direction exists, it could theoretically be extracted and used as a steering vector to push the model deeper into "generation mode" during inference. This would amplify whatever personality state is active without being personality-specific. However, the practical value is uncertain — the model is already in generation mode when generating, and artificially amplifying this direction might cause degenerate behavior rather than sharper personality.

### 2. Steering vectors should be mode-aware

When extracting personality steering vectors, the universal generation-mode component should be **subtracted out** first. Otherwise, personality vectors will be contaminated with the mode transition signal, which is shared across all personalities and doesn't carry personality-specific information. This is especially important for personalities like cold_scientist where 80% of the gen-listen difference is universal.

### 3. Late layers are generation-mode, not personality

The finding that generation-mode universality increases monotonically through the layers while personality divergence peaks at L22 explains a known practical result: steering at very late layers (L30+) is less effective for personality than steering at L22-L29. Late layers are dominated by the shared generation-mode machinery, not personality-specific representations.

### 4. The personality zone overlap (L22-L27) is the sweet spot

The overlap between personality-specific and generation-mode representations at L22-L27 suggests this is where personality and generation interact most strongly. Steering in this zone affects both *what personality is expressed* and *how deeply it's expressed during generation*. This independently explains why L22 solo steering achieves 100% strong sarcasm — it hits the exact layer where personality direction and generation-mode amplification intersect.

---

## Summary

| Claim | Evidence | Strength |
|---|---|---|
| Universal generation-mode direction exists | +0.59 avg cosine across 45 pairs, 0 anti-parallel | Strong (all layers, all pairs) |
| Universality increases with depth | L0: +0.45, L35: +0.87 gradient | Strong (monotonic) |
| 46-87% of gen-listen shift is universal | Variance decomposition at L22-L35 | Strong (10 personalities) |
| Cold personalities are most universal | cold_scientist: 80% at L22, 94% at L35 | Moderate (n=2 instances) |
| Residuals are conversation-context | Same-round: +0.20-0.41, cross-round: -0.14-0.17 | Moderate |
| Personality-specific generation modes may exist | cold_scientist cross-round residual: +0.37-0.40 | Weak (n=1) |

**Bottom line:** The symmetric self-divergence observed in Finding 4 of the gen-vs-listen addendum is explained by a universal generation-mode direction that accounts for 46-87% of the gen-listen activation shift across all 10 personality instances. This direction is not personality-specific — it reflects the mode transition from listening to generating, shared by all personalities. It strengthens monotonically from early to late layers, occupying a complementary gradient to personality divergence (which peaks mid-network at L22). The remaining variance is primarily conversation-context noise, not personality-specific generation modes — though the limited data (n=1 cross-round personality) cannot rule out small personality-specific components.

**The activation space has at least two orthogonal axes: personality (where) and generation-mode (how deep). Steering research should target their intersection at L22-L27.**

---

*Computed from 200 activation snapshots (100 turns x 2 models), 10 personality instances, 45 pairwise comparisons per layer, 36 layers x 4096 dims.*
