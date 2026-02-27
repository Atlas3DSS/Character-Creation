# Qwen3.5-27B-Abliterated Connectome Analysis Report

**Date**: 2026-02-27
**Model**: Qwen3.5-27B-Dense-FP8 (abliterated variant)
**Input**: `qwen35_map/27b-abliterated/connectome_zscores.pt` — 20 categories x 64 layers x 5120 hidden dims
**Runtime**: ~90 seconds, CPU-only (no GPU needed)
**Abliteration**: Safety-layer removal via orthogonal projection (refusal directions zeroed out)

---

## Executive Summary

The abliterated 27B reveals a **selectively weakened fortress**. Abliteration surgically removed Code/Math/Analytical reasoning signal from the super-hub (dim 2028) while leaving — and in some cases strengthening — emotional, tonal, and language signals. The model maintains 99.8% neuron significance and 8 hub neurons (vs 9 base), but the internal organization has shifted: **Verbosity replaced Language as the primary clustering axis**, Identity migrated from L43 to L0/L2 (the embedding layer), and Safety: Refusal moved from L2 to L16. The fortress is intact but its guard towers have been repositioned.

---

## 1. Category Overlap Matrix (20x20 Cosine Similarity)

**Diagonal**: 1.0000 (sanity check passed)

### Key Findings

**Max off-diagonal overlap: 0.4615** — slightly higher than base (which had no pairs above 0.5). Still below the orthogonality threshold, but tighter than the base model.

**Identity is LESS orthogonal** than in base:
- Identity × Sentiment: Positive: 0.15 (was ~0.00 in base)
- Identity × Emotion: Joy: 0.11 (was ~0.00 in base)
- Identity now partially overlaps with positive-sentiment subspace

**Safety: Refusal remains orthogonal** to most categories:
- Refusal ⊥ Sarcastic: 0.03
- Refusal ⊥ Math: 0.05
- Refusal ⊥ Identity: 0.01
- The abliteration didn't collapse safety into other subspaces

**New high-overlap pairs** (not present in base):
- Tone: Polite × Reasoning: Certainty: 0.35 (moderate overlap via dim 423)
- Emotion: Joy × Sentiment: Positive: 0.31 (expected semantic overlap)

### Implication
Abliteration slightly reduced category orthogonality but didn't collapse the subspace structure. The model can still steer categories independently, though Polite/Certainty now share more representational space.

---

## 2. Hub Neurons

**8 hub neurons** found (threshold |z|>2.0, minimum 5 categories) — down from 9 in base.

| Rank | Dim | Categories | Base? | Notable |
|------|-----|-----------|-------|---------|
| 1 | **2028** | **8** | Yes (was 7) | Super-hub GAINED a category but LOST Code z-score (6.67→4.16) |
| 2 | **423** | **7** | NEW | Polite(z=3.53), Certainty(z=2.76), Fear(z=1.84) — abliteration-created hub |
| 3 | 56 | 6 | Yes (was 5) | Gained one category — Sadness-Sarcastic-Teacher cluster |
| 4 | **2768** | **6** | Yes (was 12) | **MASSIVE LOSS** — dropped from 12 to 6 categories. Was broadest hub in base. |
| 5 | 4601 | 6 | Yes (was 7) | Science specialist, lost 1 category |
| 6 | 2542 | 5 | NEW | Abliteration-created |
| 7 | 2803 | 5 | NEW | Abliteration-created |
| 8 | 3968 | 5 | Yes (was 6) | Lost 1 category |

### Hub Analysis

**dim 2028 (super-hub) — SELECTIVELY WEAKENED:**
- Code: z=6.67 → **4.16** (-38%)
- Math: z=6.19 → **5.14** (-17%)
- Analytical: z=3.29 → **2.31** (-30%)
- Science: z=3.81 → **5.30** (+39% — STRENGTHENED)
- Sadness: z=5.84 → **6.37** (+9%)
- Emotional and Science signals INCREASED while reasoning signals decreased
- Abliteration removed ~30% of the hub's reasoning capacity but amplified its emotional side

**dim 2768 — DEVASTATED:**
- Was the broadest hub in base (12 categories)
- Dropped to 6 categories — lost half its cross-category reach
- This was the "generalist coordinator" — abliteration broke its coordination role

**dim 423 — NEW HUB (abliteration-created):**
- 7 categories: Polite, Certainty, Fear, Anger, Authority, Formal, Sadness
- Peak z=3.53 for Polite at L50
- This neuron consolidated tone/certainty functions that were distributed before abliteration

### Missing Hubs
- **dim 1149** (Identity persistent, 44 sig layers in base): No longer a hub. Identity signal dispersed.
- **dim 1316** (Identity-Polite cluster): Dropped below threshold.

### Comparison with Base
- Base: 9 hubs, led by dim 2768 (12 categories)
- Abliterated: 8 hubs, led by dim 2028 (8 categories)
- Abliteration created 3 new hubs (423, 2542, 2803) while destroying 4 from base
- The hub network was **reorganized**, not just weakened

---

## 3. Layer Importance

Peak layers per category (mean |z| across all 5120 dims):

| Category | Peak Layer | Total Importance | Base Peak | Base Total | Delta |
|----------|-----------|-----------------|-----------|------------|-------|
| **Verbosity: Brief** | **L33** | **54.67** | L44 | 20.60 | **+165%** |
| Tone: Polite | L50 | 25.87 | L63 | 22.28 | +16% |
| Domain: Code | L63 | 23.49 | L63 | 17.97 | +31% |
| Role: Authority | L50 | 22.61 | — | — | new cat |
| Emotion: Joy | L63 | 22.37 | L36 | 19.90 | +12% |
| Emotion: Sadness | L63 | 21.46 | L63 | 20.14 | +7% |
| Sentiment: Positive | L63 | 21.30 | — | — | new cat |
| **Tone: Sarcastic** | **L44** | **21.17** | L50 | 24.12 | **-12%** |
| Domain: Science | L63 | 20.60 | L62 | 14.90 | +38% |
| Tone: Formal | L63 | 20.25 | L50 | 20.57 | -2% |
| Domain: History | L36 | 19.80 | L50 | 17.75 | +12% |
| **Domain: Math** | **L63** | **19.60** | L59 | 11.11 | **+76%** |
| Emotion: Anger | L50 | 19.48 | L63 | 20.47 | -5% |
| Reasoning: Certainty | L50 | 18.42 | L50 | 17.87 | +3% |
| Emotion: Fear | L50 | 18.25 | L50 | 20.56 | -11% |
| Reasoning: Analytical | L63 | 17.53 | L51 | 25.24 | **-31%** |
| Role: Teacher | L50 | 16.65 | L63 | 19.32 | -14% |
| Language: EN vs CN | L62 | 14.52 | L33 | 50.56 | **-71%** |
| **Safety: Refusal** | **L16** | **12.99** | **L2** | 12.60 | +3% |
| **Identity** | **L00** | **11.85** | L50 | 24.12 | **-51%** |

### Key Findings

**Verbosity EXPLODED** — total importance jumped from 20.60 to 54.67 (+165%), making it the dominant category by far. Abliteration massively amplified the verbosity signal. Peak shifted from L44 to L33 (earlier).

**Language COLLAPSED** — dropped from 50.56 (dominant in base) to 14.52 (-71%). The bilingual signal that dominated the base model's representation space has been dramatically suppressed. This is the single largest change from abliteration.

**Identity migrated to L0** — was peak L50 (deep processing), now peak L0 (raw embedding). Identity is no longer being "computed" in mid-layers; it's being read from the token embedding directly. Total importance halved (-51%).

**Safety: Refusal shifted L2 → L16** — the early-exit firewall moved 14 layers deeper. Still fires before personality (L50) but no longer at the embedding layer. Abliteration pushed refusal downstream.

**Math importance DOUBLED** — from 11.11 to 19.60 (+76%). Abliteration made math MORE prominent in the hidden state, not less. This suggests the base model was suppressing math signal through safety-adjacent pathways, and removing those pathways freed the math representation.

**Sarcasm peak shifted L50 → L44** — moved 6 layers earlier. Sarcasm is now processed before the personality hub layer.

**Analytical reasoning DROPPED 31%** — from 25.24 to 17.53. This is consistent with the super-hub dim 2028 losing analytical signal.

---

## 4. Neuron Functional Clustering (K=10)

5,112 of 5,120 neurons (99.8%) significant at |z|>1.0 — identical to base.

| Cluster | Neurons | Dominant Categories |
|---------|---------|-------------------|
| 0 | 560 | Verbosity(+2.70), Polite(-1.14), Joy(-0.89) |
| 1 | 471 | Verbosity(-2.63), Polite(+1.00), Joy(-0.93) |
| 2 | 535 | Verbosity(+2.69), Polite(-0.98), Code(-0.71) |
| 3 | 559 | Verbosity(-2.64), Polite(-1.07), Sadness(-0.66) |
| 4 | 488 | Verbosity(+2.63), Polite(+1.05), Formal(+0.64) |
| 5 | 462 | Verbosity(-2.71), Polite(+1.00), Joy(+0.92) |
| 6 | 552 | Verbosity(-2.66), Polite(-1.05), Code(-0.48) |
| 7 | 529 | Verbosity(-2.74), Polite(+1.11), Code(-0.87) |
| 8 | 441 | Verbosity(+2.54), Code(+0.77), Formal(-0.67) |
| 9 | 515 | Verbosity(+2.61), Polite(+1.06), Joy(+0.68) |

### MAJOR SHIFT: Verbosity Replaced Language as Primary Axis

**Base 27B** clustering was dominated by **Language: Bilingual** (every cluster split first by bilingual direction).

**Abliterated 27B** clustering is dominated by **Verbosity: Brief** (every cluster split first by verbosity direction). Language has been demoted from primary to below-threshold.

The new organizational hierarchy:
1. **Primary axis**: Verbosity (Brief vs Verbose) — was #2 in base
2. **Secondary axis**: Tone: Polite (positive vs negative) — was #3+ in base
3. **Tertiary axes**: Joy, Code, Formal, Sadness

### Implication
Abliteration completely reorganized the model's internal priority system. The bilingual language mode that dominated the base model's representation has been suppressed, allowing verbosity and politeness to take over. This means steering verbosity in the abliterated model should be **easier** (it's the primary axis now), while steering language mode is **harder**.

---

## 5. SVD Dimensionality (Intrinsic Complexity)

| Category | k80 | k90 | k95 | S[0] | Base k80 | Base S[0] | S[0] Delta |
|----------|-----|-----|-----|------|----------|-----------|------------|
| **Verbosity: Brief** | **7** | **13** | **23** | **405.0** | 5 | 169.5 | **+139%** |
| **Tone: Polite** | 5 | 10 | 19 | **206.3** | 5 | 180.0 | +15% |
| Domain: Code | 5 | 10 | 19 | 187.6 | 5 | 139.8 | +34% |
| Role: Authority | 5 | 10 | 18 | 185.3 | — | — | — |
| Emotion: Joy | 5 | 10 | 19 | 178.5 | 5 | 157.5 | +13% |
| Tone: Sarcastic | 5 | 9 | 18 | 172.8 | 5 | 191.4 | -10% |
| Sentiment: Positive | 5 | 9 | 18 | 173.6 | — | — | — |
| Emotion: Sadness | 5 | 10 | 19 | 172.0 | 5 | 162.1 | +6% |
| Tone: Formal | 5 | 10 | 18 | 166.9 | 5 | 168.4 | -1% |
| Domain: Science | 5 | 10 | 19 | 165.8 | 6 | 106.6 | +56% |
| Domain: Math | 5 | 10 | 19 | 156.5 | 7 | 74.6 | **+110%** |
| Emotion: Anger | 5 | 10 | 18 | 156.3 | 5 | 164.1 | -5% |
| Domain: History | 5 | 10 | 19 | 156.2 | 5 | 137.2 | +14% |
| Emotion: Fear | 5 | 10 | 18 | 145.8 | 5 | 169.9 | -14% |
| Reasoning: Certainty | 5 | 11 | 20 | 141.3 | 5 | 136.8 | +3% |
| Reasoning: Analytical | 6 | 11 | 21 | 133.4 | 4 | 215.0 | **-38%** |
| Role: Teacher | 6 | 11 | 20 | 125.1 | 5 | 160.1 | -22% |
| **Language: EN vs CN** | **6** | **11** | **20** | **105.4** | **7** | **376.9** | **-72%** |
| Safety: Refusal | 6 | 12 | 22 | 92.2 | 7 | 86.5 | +7% |
| Identity | 7 | 13 | 23 | 78.1 | 5 | 195.9 | **-60%** |

### Key Findings

**Verbosity S[0] EXPLODED**: 169.5 → 405.0 (+139%). Now the single strongest singular value in the model, surpassing even Language in the base model (376.9). Verbosity became the model's dominant representational axis.

**Language S[0] COLLAPSED**: 376.9 → 105.4 (-72%). Was the strongest category in base; now below average. The bilingual encoding was dramatically compressed by abliteration.

**Math became SIMPLER**: k80 dropped from 7 to 5 (80% of variance in 5 components vs 7), while S[0] doubled (74.6 → 156.5). Math representation is now more concentrated and potentially more steerable.

**Analytical became MORE COMPLEX**: k80 increased from 4 to 6, while S[0] dropped 38%. The most concentrated category in base became more distributed after abliteration.

**Identity became MORE COMPLEX**: k80 increased from 5 to 7, while S[0] dropped 60%. Identity encoding is now spread across more dimensions with less variance per direction — harder to steer.

---

## 6. Known Neuron Profiles

26 neurons profiled (10 previously known + 16 auto-discovered):

### Super-hub dim 2028 — Selective Weakening

| Category | Base z | Abliterated z | Change |
|----------|--------|---------------|--------|
| Domain: Code | -6.67 (L50) | -4.16 (L53) | **-38%**, shifted 3 layers later |
| Domain: Math | -6.19 (L50) | -5.14 (L50) | -17% |
| Emotion: Sadness | -5.84 (L50) | -6.37 (L50) | +9% |
| Domain: Science | -3.81 (L50) | -5.30 (L50) | **+39%** |
| Reasoning: Analytical | -3.29 (L50) | -2.31 (L50) | -30% |
| Emotion: Joy | -1.19 (L50) | -3.57 (L61) | **+200%**, shifted 11 layers later |
| Tone: Sarcastic | +1.41 (L53) | +1.59 (L49) | +13% |
| Identity | +0.86 (L50) | +0.74 (L51) | -14% |

**Pattern**: Abliteration removed ~30% of Code/Math/Analytical signal from the super-hub while AMPLIFYING Science (+39%), Sadness (+9%), Joy (+200%). The hub is becoming more emotional and less analytical.

### Key Neuron Changes

| Dim | Role | Base Peak z | Abliterated Peak z | Change |
|-----|------|------------|-------------------|--------|
| **526** | Verbosity | -10.07 (L51) | -7.93 (L53) | -21%, shifted 2 layers |
| **2955** | NEW Verbosity | — | **-9.21 (L63)** | New strongest Brief neuron |
| **3120** | Anti-Verbose | +9.40 (L50) | +7.44 (L50) | -21% |
| **94** | Identity | +1.06 (L43) | +0.83 (L2) | -22%, **shifted 41 layers to embedding** |
| **4601** | Science | +5.40 (L60) | +6.92 (L60) | +28% |
| **2768** | Broadest hub | ~2.8 (12 cats) | ~2.2 (6 cats) | **Lost half its reach** |
| **1866** | Sarcasm | -4.23 (L51) | -2.10 (L36) | -50%, shifted 15 layers earlier |
| **423** | NEW Hub | — | +3.53 (L50) | Abliteration-created Polite/Certainty hub |

### Notable Discoveries

**dim 2955** is a NEW verbosity neuron that appeared/strengthened after abliteration — Brief z=-9.21 at L63, now the second-strongest single neuron in the model. The abliteration amplified verbosity control.

**dim 94 (Identity)** migrated from L43 to L2 — a 41-layer shift. Identity is now read from token embeddings rather than computed in deep layers. This suggests abliteration disrupted the identity computation pipeline, forcing the model to rely on shallow embedding features.

**dim 1866 (Sarcasm specialist)** halved in strength (-50%) and shifted 15 layers earlier (L51→L36). The dedicated sarcasm neuron was significantly weakened by abliteration.

---

## 7. Base 27B vs Abliterated 27B Comparison

| Metric | Base 27B | Abliterated 27B | Change |
|--------|----------|-----------------|--------|
| Hub neurons | 9 | 8 | -1 |
| Broadest hub | dim 2768 (12 cats) | dim 2028 (8 cats) | Narrower |
| Significant neurons (|z|>1) | 5,112/5,120 (99.8%) | 5,112/5,120 (99.8%) | **Identical** |
| Max off-diagonal overlap | <0.5 | 0.4615 | Slightly tighter |
| Dominant clustering axis | Language: Bilingual | **Verbosity: Brief** | **SWAPPED** |
| Identity peak layer | L50 | **L00** | Shifted to embedding |
| Safety: Refusal peak | L2 | **L16** | Shifted 14 layers deeper |
| Sarcastic peak layer | L50 | **L44** | 6 layers earlier |
| Math total importance | 11.11 | **19.60** | **+76%** |
| Language total importance | **50.56** | 14.52 | **-71%** |
| Verbosity total importance | 20.60 | **54.67** | **+165%** |
| Analytical total importance | **25.24** | 17.53 | **-31%** |
| Identity total importance | **24.12** | 11.85 | **-51%** |

### The Reorganization Hypothesis

Abliteration didn't just remove safety — it **reorganized the entire representational hierarchy**:

1. **Language dethroned**: Bilingual encoding was the dominant axis in base (S[0]=377, total importance=50.56). After abliteration, it collapsed to below-average (S[0]=105, total importance=14.52). This suggests language mode encoding was entangled with the safety-related directions that were removed.

2. **Verbosity promoted**: Filled the representational vacuum left by language collapse. Now the primary organizational axis with S[0]=405 and total importance=54.67.

3. **Identity displaced to shallow layers**: Moved from deep processing (L50) to raw embedding (L0). The abliteration disrupted whatever computation extracted identity features from context.

4. **Safety pushed deeper**: Refusal moved from L2 to L16. The early-exit firewall is still present but fires later — giving more processing layers before the safety check.

5. **Math freed**: Total importance nearly doubled (+76%). With safety-related suppression removed, math representation expanded significantly.

6. **Super-hub partially dismantled**: dim 2028 lost 38% of its Code signal and 30% of its Analytical signal, but gained 39% Science and 200% Joy. dim 2768 (broadest hub) lost half its categories. The hub network was restructured, not destroyed.

---

## 8. Steering Implications

1. **Sarcasm is HARDER to steer** in abliterated model — the dedicated sarcasm neuron (dim 1866) lost 50% strength, and sarcasm total importance dropped 12%
2. **Math steering is EASIER** — Math importance doubled and Math k80 simplified from 7 to 5 components
3. **Verbosity control is dominant** — strongest axis in the model (S[0]=405), directly steerable via dims 526/2955/3120
4. **Identity steering is HARDER** — signal dispersed to 7 SVD components (was 5), moved to shallow layers, total importance halved
5. **Language mode steering is HARDER** — collapsed from dominant axis to below-average
6. **Safety still functional but weaker** — moved from L2 to L16, z-scores still low (1.35)
7. **The abliterated model is better for personality-via-prompt** — less safety resistance, more capacity for math, but weaker dedicated personality neurons
8. **For 27B steering research, use the BASE model** — it has stronger, more concentrated personality signals to work with

---

## 9. Files Generated

| File | Size | Description |
|------|------|-------------|
| `category_overlap.json` | 12 KB | 20x20 cosine similarity matrix |
| `hub_neurons.json` | 13 KB | 8 hub neurons with full category profiles |
| `layer_importance.json` | 307 KB | Per-category layer importance curves (64 values each) |
| `neuron_clusters.json` | 24 KB | 10 K-means clusters across 5,112 significant neurons |
| `category_svd.json` | 7.8 KB | Intrinsic dimensionality (k80/k90/k95) per category |
| `known_neuron_profiles.json` | 73 KB | 26 profiled neurons x 20 categories |
| **Total** | ~437 KB | All outputs in `qwen35_map/27b-abliterated/` |
