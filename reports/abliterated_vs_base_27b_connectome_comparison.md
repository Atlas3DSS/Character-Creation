# Abliterated vs Base Qwen3.5-27B Connectome Comparison

**Date**: 2026-02-28
**Data**: `qwen35_map/27b/connectome_zscores.pt` (base) vs `qwen35_map/27b-abliterated/connectome_zscores.pt`
**Shape**: [20 categories, 64 layers, 5120 dims] each

---

## Executive Summary

Abliteration fundamentally restructures the model's representational geometry, but NOT uniformly. The effects are **catastrophically targeted** at two categories — **Identity** (mean cosine = 0.062) and **Safety: Refusal** (mean cosine = 0.111) — while leaving most domain/reasoning categories largely intact. Sarcastic tone representation is moderately reshuffled (mean cosine = 0.527) but retains structural coherence. The abliterated model redistributes importance from mid-late layers (L44-L55) to early layers (L0-L12).

---

## 1. Global Cosine Similarity: Three Tiers

Overall mean cosine similarity across all 20 categories and 64 layers: **0.509**

| Tier | Categories | Mean Cosine |
|------|-----------|-------------|
| **Demolished** | Identity (0.062), Safety:Refusal (0.111) | <0.15 |
| **Reshuffled** | Language (0.321), Fear (0.416), Certainty (0.448), Teacher (0.454), Positive (0.462), History (0.468), Analytical (0.490), Anger (0.516), Sarcastic (0.527), Math (0.551) | 0.32-0.55 |
| **Preserved** | Formal (0.578), Science (0.586), Sadness (0.628), Authority (0.640), Joy (0.641), Polite (0.692), Code (0.696), Verbosity:Brief (0.889) | >0.57 |

Three category-layer pairs show **negative cosine** (anti-correlated): Safety:Refusal at L0 (-0.019), History at L0 (-0.013), Sarcastic at L0 (-0.012).

---

## 2. Peak Z-Score Migration

13 of 20 categories migrated their peak location:

**Most dramatic migrations:**

| Category | Base Peak | Abliterated Peak | Layer Shift |
|----------|----------|-----------------|-------------|
| **Identity** | L43:d94 (z=1.06) | L2:d2033 (z=1.28) | -41 layers |
| **Safety:Refusal** | L49:d10 | L27:d2615 | -22 layers |
| **Verbosity:Brief** | L51:d526 (z=10.07) | L63:d2955 (z=9.21) | +12 layers |
| Authority | L50 | L63 | +13 layers |
| Teacher | L45 | L50 | +5 layers |

**Stable categories** (no migration): Math, Science, Sadness, Language, Analytical, Sarcastic

**Peak z-score changes:**
- Largest reduction: Code (-2.51), Math (-1.05), Analytical (-0.99)
- Largest increase: Language (+1.51), Science (+1.49), Certainty (+0.63)

The super-hub **dim 2028** retained its position for Math/Science but Code peak migrated L50→L53 with z-score dropping 6.67→4.16 (-38%).

---

## 3. Hub Neuron Comparison

| Metric | Base | Abliterated |
|--------|------|-------------|
| Hub count | 9 | 8 |
| Max categories per hub | 12 (d2768) | 8 (d2028) |

**Hub survival:**
- **Survived** (5): d56, d2028, d2768, d3968, d4601
- **Lost** (4): d1149, d1316, d2833, d4010
- **New** (3): d423, d2542, d2803

**d2768** (was 12-category mega-hub) collapsed to 6 categories. LOST: Identity, all emotions except Joy, role/reasoning. The mega-hub was heavily Identity-loaded; abliteration stripped all Identity associations.

**d2028** (super-hub) changed from 7→8 categories. LOST: Identity, Anger, Sarcastic. GAINED: Math, Science, Joy, Analytical, Brief. Became more domain-focused, less personality-focused.

**Pattern**: Abliteration dismantled Identity-associated hubs and promoted Domain-focused hubs. Remaining hubs encode WHAT the model knows (Code, Math, Science) rather than HOW it behaves (Identity, Emotion, Role).

---

## 4. Category Overlap Changes

**Largest correlation shifts:**

| Category A | Category B | Base r | Abliterated r | Delta |
|-----------|-----------|--------|--------------|-------|
| Identity | Verbosity:Brief | +0.037 | -0.122 | -0.159 |
| Certainty | Verbosity:Brief | +0.091 | -0.059 | -0.150 |
| Analytical | Positive | -0.116 | -0.241 | -0.125 |
| Fear | Joy | -0.001 | -0.114 | -0.113 |

**Sarcastic-specific changes:**
- Sarcastic-Formal anti-correlation (-0.098) vanished (+0.002) — boundary blurred
- Sarcastic-Anger weakened (-0.094 delta)
- Sarcastic-Polite INCREASED (+0.070 delta)

**39 correlation sign flips** total, but only 2 with |r| > 0.05 on both sides.

---

## 5. Layer Importance Shift

**Layer importance profile correlation (Pearson r, base vs abliterated):**

| Band | Categories |
|------|-----------|
| r > 0.98 | Math, Anger, Certainty, Science, Fear, Polite, Brief, Formal, Language, Authority |
| r = 0.95-0.98 | Code, History, Joy, Analytical, Positive, Sarcastic |
| **r < 0.95** | Teacher (0.900), **Refusal (0.671)**, **Identity (0.270)** |

**Identity layer importance DEMOLISHED** (r = 0.270). Base peaks at L59, abliterated peaks at L0. Zero overlap in top-5 layers.

**Global redistribution:**
- Layers that GAINED: L7-L12 (early, +0.07-0.11 each)
- Layers that LOST: L44-L55 (mid-late, -0.21 to -0.29 each)
- **L50 lost the most** (-0.287): this was the super-hub nexus layer

Systematic shift from mid-network personality/role zone to early layers.

---

## 6. SVD Dimensionality Changes

| Category | Base Top-1 % | Abli Top-1 % | Base rank@90% | Abli rank@90% |
|----------|-------------|-------------|---------------|---------------|
| Identity | 35.5% | 34.5% | 13 | 13 |
| Safety:Refusal | 37.2% | 39.8% | 13 | 12 |
| Sarcastic | 51.6% | 51.0% | 10 | 9 |
| Math | 49.4% | 48.9% | 10 | 10 |
| Code | 50.0% | 48.5% | 10 | 10 |
| Verbosity:Brief | 40.0% | 39.9% | 13 | 13 |

**Identity was ROTATED, not erased.** SVD spectrum unchanged — same rank (13), same entropy, same energy distribution. But the direction is completely different (cosine = 0.062). The model still has a concept for "Identity" prompts; it just points in a fundamentally different direction.

**Refusal was CONCENTRATED and relocated.** More concentrated (top-1 +2.7%, rank 13→12, entropy -0.057) and moved from L2 to L16/L27. Abliteration compressed refusal into fewer dimensions while moving it deeper.

**Sarcastic is structurally stable.** Minor rank reduction (10→9), essentially unchanged spectrum.

---

## 7. Synthesis

### What Abliteration Does to Representational Structure

1. **Surgical demolition of Identity and Refusal** — cosine <0.12 across ALL layers. Every other category retains moderate similarity.

2. **Identity was ROTATED, not erased** — same SVD rank and entropy, completely different direction. The model still encodes identity; it just points elsewhere.

3. **Refusal was CONCENTRATED and RELOCATED** — unlike Identity (rotated at constant rank), Refusal became tighter and moved deeper (L2→L16/L27).

4. **Mid-network importance collapse** — L44-L55 lost importance across all categories. This zone contained personality/role hubs. Abliteration flattened this peak.

5. **Hub restructuring: personality→domain** — surviving hubs lost Identity/Emotion/Role associations, gained Code/Math/Science. More "factual," less "behavioral."

6. **Sarcastic survived moderately intact** — mean cosine 0.527, same peak location, stable SVD. Collateral reorganization only in L0-L10.

7. **Verbosity:Brief is MOST resilient** — mean cosine 0.889, barely touched.

### Implications for Skippy Steering

- **Identity steering needs completely new vectors** — base identity vectors useless (cosine 0.062)
- **Refusal suppression already handled** — abliteration demolished it, anti-refusal steering unnecessary
- **Sarcastic vectors from L10+ transfer at ~50-72% fidelity** — usable with alpha recalibration
- **Domain capabilities structurally intact** — Math/Code/Science layer importance r > 0.96
- **L50 super-hub weakened** — strategies relying on L50 need adjustment; L63 gained importance

---

## Data

- Base connectome: `qwen35_map/27b/connectome_zscores.pt`
- Abliterated connectome: `qwen35_map/27b-abliterated/connectome_zscores.pt`
- Analysis JSONs: `qwen35_map/27b/` and `qwen35_map/27b-abliterated/`
