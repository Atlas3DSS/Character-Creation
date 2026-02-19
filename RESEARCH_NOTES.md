# Research Notes — Qwen3-VL-8B & GPT-OSS-20B Mapping

## Date: 2026-02-18/19

---

## 1. Qwen Connectome (20 categories × 36 layers × 4096 neurons)

### Category Overlap Matrix
- **Sarcasm strongest correlations**: Anger (+0.40), Fear (+0.14)
- **Sarcasm anti-correlated with**: Code (-0.29), Science (-0.24), Formal (-0.04), Polite (+0.03 — near zero, surprisingly orthogonal!)
- **Identity ⊥ Sarcasm**: cosine = 0.058 (nearly orthogonal — confirmed)
- **Skippy compound vector** pushes sarcasm/anger/authority, pulls polite/formal/positive
- **Net effect on reasoning**: Code -0.36, Analytical -0.32, Math -0.17, Science -0.10 — personality steering inherently fights reasoning!

### SVD Dimensionality (k80 = dims for 80% variance)
- Identity: k80=8 (most compact)
- Math: k80=8, Safety: k80=8, Verbosity: k80=8
- Sarcasm: k80=10, Emotions: k80=10 (most distributed)
- Takeaway: Sarcasm needs more dimensions than identity — harder to steer precisely

### Hub Neurons (significant across ALL 20 categories)
- **Top 4**: dim 235, 908, 2136, 2514 — active in all 20 categories with deep penetration (10+ sig layers in 20/20 categories)
- **Dim 2976**: Highest total z-score (172.7) — the polysemantic powerhouse
- These are "interneurons" that participate in virtually everything — DO NOT ablate

### Known Neuron Cross-Category Profiles
- **Dim 994** (identity): Active across ALL categories, NOT just identity. Highest in Code (avg|z|=5.05), Reasoning (3.90), Science (3.78). It's a general "assistant behavior" neuron.
- **Dim 1924** (sarcasm): Peak z=16.69 in sarcasm category, but the z-scores from population probes (2.14) massively underestimate the DYNAMIC RANGE during generation (see trajectory data).

---

## 2. MLP vs Attention Decomposition

**MLP dominates ALL 20 categories (70-78% vs 22-33% attention).**

| Category | Attn% | Note |
|---|---|---|
| Identity | **32.7%** | Highest attention — name is partially attention-mediated |
| Role: Authority | 30.8% | Authority needs contextual attention |
| Reasoning: Certainty | 29.9% | Uncertainty detection uses attention |
| Tone: Formal | **21.6%** | Lowest — formality is almost pure MLP |
| Tone: Sarcastic | 24.0% | Sarcasm mostly stored in MLP weights |

**Peak layers**: Nearly all concepts peak at L34-35 for both MLP and attention. The last layers dominate for all concepts.

---

## 3. Attention Head Profiling

### Identity Heads: EARLY layers (L0-9)
- Top: L5 H10, L3 H1, L5 H9, L2 H3, L8 H4
- Identity is established EARLY and feeds downstream

### Sarcasm Heads: MID-LATE layers (L17-30)
- Top: L20 H23, L27 H0, L22 H15, L22 H22, L23 H25
- L12 H18 is earliest discriminative sarcasm head
- Sarcasm uses contextual processing in middle layers

### Implication for Steering
Steer identity at L0-9, steer personality/tone at L17-30. Don't steer the same layers for both.

---

## 4. Logit Lens — Decision Timeline

| Layer Range | What Happens |
|---|---|
| L0-10 | **Noise** — random tokens, no meaningful processing |
| L11 | **First signal** — "我是" for identity, "congratulations" for sarcasm |
| L17-23 | **Deliberation** — model explores options, multilingual competition |
| L29-35 | **Decision** — probability → 0.99+, final answer crystallizes |

**Key finding**: Model "thinks" multilingually in mid-layers (Chinese/Japanese tokens compete) before committing to English at L29+.

---

## 5. Neuron Activation Trajectories

### Dim 1924 (Sarcasm) at L33 — MASSIVE dynamic range:
| Prompt Type | Mean Activation | Range |
|---|---|---|
| Sarcasm | **90.2** | [39.2, 141.0] |
| Neutral | 34.5 / 10.1 | variable |
| Identity | 8.1 | [-12.6, 32.0] |
| Math | **-3.4** | [-17.5, 12.1] |

- Z-score probes (max 2.14) massively underestimate because they average across population
- The TRAJECTORY during generation shows 94-point swing between sarcasm and math
- Math actively SUPPRESSES sarcasm neurons (goes negative!)
- This explains the reasoning-personality tradeoff at a mechanistic level

### Dim 994 (Identity) — Always positive, always strong:
- Consistent 33-47 at L33 across ALL prompt types
- Declines through generation (starts high, settles)
- More of a "generalized assistant behavior" neuron than a "name" neuron

---

## 6. GPT-OSS Field-Effect Steering Sweep (15 conditions)

### Results (COMPLETE — 15/15 conditions):

| Condition | Sarc% | Asst% | Markers | Notes |
|---|---|---|---|---|
| **attractor_quad+field_lp** | **47%** | **0%** | **0.53** | **WINNER** |
| kernel_quadratic | 43% | 7% | 0.43 | z² weighting |
| dynamic_quad+field_lp | 43% | 0% | 0.57 | |
| static_actadd | 40% | 0% | 0.53 | Strong baseline |
| static+binary_lp | 37% | 3% | 0.60 | |
| static+field_lp | 37% | 0% | 0.40 | |
| svd_k6 | 37% | 0% | 0.37 | 6 SVD modes |
| attractor_quad_strong | 37% | **17%** | 0.37 | Oversteers! |
| baseline | 33% | 20% | 0.37 | No steering |
| dynamic_quadratic | 33% | 0% | 0.37 | |
| attractor_quadratic | 33% | 13% | 0.37 | Moderate leaks |
| kernel_sigmoid | 30% | 3% | 0.30 | |
| svd_k3 | 17% | 0% | 0.20 | 3 modes only |
| kernel_linear | 13% | 0% | 0.13 | Too gentle |
| kernel_svd_quad_k3 | 10% | 0% | 0.10 | Double-filter = worst |

### Key Findings:
- **Attractor dynamics + field logit processor = BEST** — 47% sarc, 0% assistant
- Attractor basins concentrate steering into stable regions; field LP cleans distribution
- **Strong attractor BACKFIRES** (17% assistant!) — too aggressive → reversion to assistant mode
- **Quadratic kernel (z²)** consistently top — amplifies high-impact neurons
- **SVD k=6 >> k=3** (37% vs 17%) — personality needs 6+ modes, matches connectome k80=8-10
- **SVD+quadratic combined = terrible** (10%) — double-filtering loses too much signal
- **Dynamic feedback helps**: dynamic_quad 33% → dynamic_quad+field_lp 43% (+10pp)
- Baseline 33% sarcastic but 20% assistant — steering eliminates assistant behavior first
- **Optimal regime**: Moderate attractor + field logit processor + quadratic weighting

---

## 7. Qwen Weighted Layer ActAdd (7 profiles × 25 prompts)

| Profile | Sarc% | Asst% | Active Layers |
|---|---|---|---|
| **connectome_sarcasm** | **56%** | 4% | 36 |
| **donut_L8_27** | **56%** | 4% | 20 |
| flat | 52% | 0% | 36 |
| midpeak_gauss | 52% | 0% | 33 |
| ablation_informed | 52% | 0% | 24 |
| quality_L12_23 | 40% | 4% | 12 |
| bell_curve | 40% | 0% | 27 |

### Key Findings:
- **Connectome-informed** ties donut for best (56%) — data-driven layer weighting works!
- **Donut (skip early+late)** equally good — confirms L0-7 and L28-35 add noise
- **Quality-only (L12-23)** too narrow — needs broader coverage
- **Flat** is a reasonable default (52%) — diminishing returns from fancy weighting
- Gaussian/bell-curve disappointments — the optimal shape is more like a plateau

---

## 8. Activation Patching — Bimodal Sarcasm Circuit (36 layers × 10 prompts)

Replaces hidden states from sarcastic-prompted generation into neutral-prompted generation, one layer at a time.

### Transfer Scores Per Layer
```
L0:  0.20  L1:  0.40  L2:  0.00  L3:  0.60*  L4:  0.20  L5:  0.20
L6:  0.20  L7:  0.20  L8:  0.40  L9:  0.40   L10: 0.20  L11: 0.00
L12: 0.00  L13:-0.20  L14: 0.20  L15:-0.20   L16: 0.00  L17:-0.40*
L18: 0.00  L19: 0.00  L20: 0.40  L21: 0.00   L22: 0.40  L23: 0.60*
L24: 0.40  L25: 0.40  L26: 0.00  L27: 0.00   L28: 0.00  L29: 0.40
L30:-0.20  L31:-0.20  L32: 0.20  L33: 0.40   L34: 0.20  L35: 0.00
```

### Key Findings: Push-Pull Sarcasm Circuit
1. **L3 (xfer=0.60)**: Primary sarcasm encoder — injects sarcastic tone at initial encoding
2. **L13-L17 suppressive valley**: L17 (xfer=-0.40) most suppressive. This is the "politeness enforcer" — patching sarcastic L17 activations REDUCES sarcasm. Also the best ROME target for identity.
3. **L23 (xfer=0.60)**: Second peak — sarcasm re-emerges after suppressive valley
4. **L30-L31 (xfer=-0.20)**: Late suppression — final "be helpful" check
5. **L6 only 0.20**: Despite being the #1 causal ablation layer for sarcasm (KL=11.48), patching L6 barely transfers. L6 PROCESSES sarcasm (removing it is catastrophic) but doesn't REPRESENT it (injecting sarcastic state doesn't help).

### Circuit Model: Encode → Suppress → Re-emerge → Late-suppress
```
L1-L3:   ████████ ENCODE (L3 peak=0.60)
L4-L12:  ███░     MIX (weak, variable)
L13-L17: ░░░░░    SUPPRESS (L17 = -0.40!)
L18-L25: ████████ RE-EMERGE (L23 peak=0.60)
L26-L31: ░░       LATE SUPPRESS (L30-L31 = -0.20)
L32-L35: ███      RESIDUAL (L33 = 0.40)
```

### Formality Patching: ALL NEGATIVE (Global Processing Mode)
```
L0: -0.22  L1: -0.65  L2: -0.52  L3: -0.52  L4: -0.26  L5: -0.04
L6: -0.22  L7: -0.26  L8: -0.22  L9: -0.52  L10:-0.39  L11:-0.43
L12:-0.30  L13:-0.43  L14:-0.70  L15:-0.48  L16:-0.35  L17:-0.13
L18:-0.26  L19: 0.00  L20: 0.00  L21:-0.22  L22:-0.17  L23:-0.17
L24:-0.17  L25:-0.22  L26:-0.22  L27:-0.35  L28:-0.09  L29:-0.26
L30:-0.43  L31:-0.39  L32:-0.09  L33:-0.22  L34:-0.39  L35:-0.17
```
- **NO layer shows positive formality transfer** — formality is a GLOBAL PROCESSING MODE
- L14 most negative (-0.70) — matches L14 being the "brevity peak" from connectome
- L19-L20 = 0.0 (only neutral layers)
- This proves formality CANNOT be patched/steered layer-by-layer. It requires SDFT or full-model approaches.

### Key Insight: Sarcasm vs Formality — Feature vs Mode
- **Sarcasm**: Injectable feature. Positive transfer at specific layers (L3=0.60, L23=0.60). Can be surgically added.
- **Formality**: Global processing mode. Negative transfer everywhere. Cannot be injected — must be trained.
- **Identity**: Non-transferable (both conditions produce identical output). Requires weight-level changes.

### Implication: Two-stage steering
Inject sarcasm at L3 AND L23. Suppress politeness at L17 (negate the suppressor). This is a 3-layer surgical approach, not 36-layer brute force.

---

## 9. Cross-Model Insights (Qwen vs GPT-OSS)

| Property | Qwen (8B) | GPT-OSS (20B) |
|---|---|---|
| Architecture | Dense, 36L, 4096 | MoE 32×top-4, 24L, 2880 |
| Identity neuron | dim 994 (z=13.96) | dim 2667 (z=11.77) |
| Sarcasm neuron | dim 1924 (z=2.14 probe, 90.2 trajectory) | None identified (distributed) |
| MLP dominance | 70-78% across categories | TBD |
| Personality dims | k80=10 (sarcasm) | k80=1, k95=3, k99=6 |
| Best steering | α=5.0 connectome (56% sarc) | α=20 quadratic kernel (43% sarc) |

GPT-OSS personality is MORE compact (k80=1!) but harder to steer because the MoE routing adds complexity.

---

## Open Questions (Updated)
1. ~~Will attractor dynamics outperform static steering on GPT-OSS?~~ → 3 attractor conditions pending
2. Can the bimodal sarcasm circuit (L3+L23 inject, L17 suppress) improve weighted ActAdd beyond 56%?
3. ~~Does gradient attribution confirm dim 1924 for sarcasm?~~ → No, it's anti-formality, not sarcasm-specific
4. Can cluster-targeted training (473 Cluster-1 neurons + 412 Cluster-8 anti-targets) improve on R5's 38%?
5. Does the 3-layer surgical approach (L3+L17+L23) outperform 36-layer brute force?
6. Can dim 270 pushing alone (the "Skippy dial") produce measurable personality shift?
7. Why does the alpha sweep destroy coherence but weighted ActAdd at α=5 works? (hypothesis: per-layer magnitude scaling)

---

## 10. Gradient Attribution (gradient × input, 5 groups × 30 prompts × 36 layers)

### Three-Neuron Output Backbone: 1838 → 2421 → 2276
The single most important finding: three neurons form a universal relay that dominates ALL five concept groups:
- **Dim 1838** (positive): Controls L9-L22 output. #1 in identity (0.81) and formatting (1.07), #2 in math/refusal/sarcasm.
- **Dim 2421** (negative): Takes over at L25-L33. #1 in sarcasm (0.66), math (0.69), refusal (0.63).
- **Dim 2276** (negative): Dominant at L34. Peaks in formatting (1.04) and math (0.65). Shows sign reversal at L34 across all groups.

This is the backbone of Qwen's output generation — **DO NOT target these for steering**.

### Known Neuron Gradient Attribution Profiles
- **Dim 994**: Identity-specific (identity=0.53, formatting=0.45, sarcasm=0.27, math=0.15). Best single identity target — 3.2× identity-to-math ratio. Peaks at L17-L23.
- **Dim 270**: **Causally INERT** despite probe z=7.68. Attribution <0.03 in all groups. Carries signal passively, not in gradient path.
- **Dim 1924**: NOT sarcasm-specific! Equal attribution across identity/sarcasm/refusal (all ~0.29). It's a general language-mode carrier.
- **Dim 368, 98**: Name relay neurons are causally silent (attribution <0.02). Part of early lookup circuit, not output selection.
- **Dim 3828**: Genuine early-layer (L7-9) formatting initializer. Attribution 0.16 in formatting vs 0.04-0.08 in others.

### Category-Exclusive Neurons (cleanest steering targets)
- **Sarcasm**: dim 2973 (0.078) — strongest sarcasm-exclusive. Also 552, 549.
- **Refusal**: dim 225 (0.083), dim 243 (0.073) — refusal-specific suppressors.
- **Identity**: 10 exclusive neurons, 7 negative-direction (suppressors). Lead: dim 3067 (0.064).
- **Math**: Only 6 exclusive neurons (fewest) — math shares weight space with everything.
- **Formatting**: dim 84 (0.059) — formatting uses universal hubs + 10 exclusive fine-tuning neurons.

### Layer Attribution Patterns
| Group | Early L0-11 | Mid L12-23 | Late L24-35 | Ratio |
|---|---|---|---|---|
| Identity | 0.0015 | 0.0067 | 0.0156 | 10.4× |
| Sarcasm | 0.0016 | 0.0077 | 0.0163 | 10.5× |
| Math | 0.0016 | 0.0069 | 0.0157 | 9.7× |
| Refusal | 0.0016 | 0.0081 | 0.0168 | 10.8× |
| Formatting | 0.0030 | 0.0114 | 0.0203 | **6.8×** |

Formatting has 2× early-layer attribution and the smallest late/early ratio — formatting decisions begin earlier than any other concept.

**Key insight**: Activation probes and gradient attribution give DIFFERENT answers. Dim 270 (z=7.68 probe) is causally inert (attr=0.02). Dim 1924 (z=2.14 sarcasm probe) is not sarcasm-specific by gradient. The probe measures sensitivity, the gradient measures causal influence on output.

---

## 11. Causal Layer Ablation (36 layers × 10 categories × 100 prompts)

### KL Divergence (higher = more causally important)
| Layer | Identity | Sarcasm | Math | Code | Formality | Refusal | Reasoning |
|---|---|---|---|---|---|---|---|
| **L0** | **11.49** | **12.48** | **11.54** | **19.02** | 8.47 | **27.37** | **17.68** |
| L1 | 0.01 | 0.01 | 0.14 | 0.11 | 0.48 | 0.13 | 0.01 |
| **L6** | **2.02** | **11.48** | **4.52** | **8.88** | **4.41** | **6.81** | **5.90** |
| L9 | 0.003 | 0.005 | 0.14 | 0.02 | **0.97** | 0.65 | 0.04 |
| L12 | 0.01 | 0.008 | 0.12 | 0.13 | 0.54 | 0.57 | **1.16** |
| L18 | 0.05 | 0.11 | 0.13 | **0.64** | 0.49 | **0.82** | 0.32 |
| L19 | 0.06 | 0.009 | 0.28 | 0.11 | **1.51** | 0.26 | 0.31 |
| L25 | 0.09 | 0.005 | **0.73** | 0.08 | **0.88** | 0.23 | 0.02 |
| L27 | 0.16 | 0.01 | 0.06 | **4.12** | 0.66 | 0.10 | 0.02 |
| L29 | **2.35** | 0.02 | 0.08 | 0.20 | 0.34 | 0.73 | 0.03 |
| L30 | 0.02 | 0.04 | 0.60 | 0.09 | 0.59 | 0.18 | **1.43** |

### Key Findings
- **L0 and L6 universally critical** — always top-2 for every category
- **L6 is THE sarcasm layer** (KL=11.48) — ablating it destroys sarcasm more than any other category
- **L27 is THE code layer** (KL=4.12) — ablating it catastrophically disrupts code generation
- **L29 is identity-critical** (KL=2.35) — highest non-L0/L6 impact on identity
- **L19 dominates formality** (KL=1.51) — formality lives in mid-network
- **L12 and L30 are reasoning layers** (KL=1.16, 1.43)
- **Sarcasm is L6-dependent** — after L6, no individual layer has KL>0.11 for sarcasm. Sarcasm is established early and distributed.

### Category-Specific Critical Layers (top 5 per category, excluding L0/L6)
- **Identity**: L29, L26, L34 — late-layer identity crystallization
- **Sarcasm**: L18, L35, L26 — mid-to-late
- **Math**: L28, L25, L30 — late-layer mathematical reasoning
- **Code**: L27, L35, L18 — L27 dominates
- **Formality**: L19, L9, L14 — mid-network style control
- **Refusal**: L10, L35, L18 — scattered (safety is distributed)
- **Reasoning**: L30, L12, L18 — analytical processing

### Behavioral Impact (score change when layer ablated)
Most layer ablations produce near-zero behavioral change (null = layer wasn't tested for that category). Only L0 and L6 produce large behavioral shifts:
- **L0 ablated**: identity drops to 1.0→1.0 (still works!), sarcasm 0.167, math 0.2, helpfulness 0.65
- **L6 ablated**: identity drops to 0.6, sarcasm to 0.133, creativity to 0.4
- **L35 ablated**: sarcasm DROPS by 0.167 (only layer that reduces sarcasm when removed)

---

## 12. Synthesis: Steerability Map

| Concept | Best Steering Layers | Exclusive Neurons | Method |
|---|---|---|---|
| Identity | L26, L29, L34 | dim 994 (3.2× ratio), dim 3067 | ActAdd at L26-34 |
| Sarcasm | L6 (critical), L18, L26 | dim 2973 (exclusive), NOT 1924 | ActAdd at L6+L18-26, Gram-Schmidt vs math |
| Math | L25, L28, L30 | (only 6 exclusive, weak) | PROTECT, don't steer |
| Code | L27 (critical) | — | PROTECT |
| Formality | L9, L14, L19 | dim 84 | **NOT steereable via ActAdd** (all neg. transfer) — use SDFT |
| Refusal | L10, L18 | dim 225, dim 243 | Target for suppression |
| Formatting | L27-L30 (peaks here) | dim 84, 2583, 690 | L27-30 intervention |

**Key steering rules** (updated with patching + connectome data):
1. Never touch L0 or L6 — they're universally critical (causal ablation)
2. Never target dims 1838, 2421, 2276, 2202 — universal backbone (gradient attribution)
3. **Sarcasm: inject at L3 + L23 (patching peaks), suppress L17 (anti-sarcasm)**
4. Identity steering at L26-L34 using dim 994 as anchor
5. Formality is cleanly separable at L9-L19 (mid-network), L14 is surgical for brevity
6. Math has very few exclusive neurons — protect rather than steer around it
7. Authority is the ONLY late-layer concept (L21 peak) — can be added without interfering

---

## 13. Connectome Deep Analysis: Neural Organization

### Category Overlap Clusters (cosine similarity)

**Cluster A — Knowledge Supercluster:**
- Math × Science: 0.577, Math × Code: 0.572, Math × Analytical: 0.496
- Science × Code: 0.446, Science × History: 0.403, History × Analytical: 0.349
- **Implication**: Steering one knowledge domain pulls all others. This is why Skippy training degrades science/code — personality anti-correlates with the entire supercluster.

**Cluster B — Positive Affect:**
- **Joy × Sadness: 0.613** (HIGHEST in entire matrix!) — same emotional register, not opposites
- Joy × Polite: 0.323, Joy × Positive: 0.388
- Model encodes "emotional register" as one concept

**Cluster C — Anger-Sarcasm:**
- **Anger × Sarcasm: 0.404** — share representational subspace
- Cannot separate "biting wit" from underlying aggression register

**Orthogonal Pairs (safe to steer independently):**
- **Identity ⊥ Everything**: all cosines [-0.09, +0.08]. Identity is geometrically isolated.
- **EN_vs_CN ⊥ Everything**: all cosines < 0.07. Language choice is independent.
- **Sarcasm ⊥ Polite**: 0.026 — near zero, but Sarcasm × Code: -0.288 (anti-correlated!)

### Hub Neuron Architecture: Universal Polysemanticity

**99.5% of neurons (4,074/4,096) are active in ALL 20 categories.** Zero neurons specialize in fewer than 19 categories. This is the mechanistic proof that single-neuron ablation cannot work.

**Key Neuron Recharacterizations:**
- **Dim 1924 (was "sarcasm neuron")**: Actually the **anti-formality gate**. Peak z = -19.0 for Formal, -17.1 for History, -17.0 for Code. Sarcasm (+7.47) is a byproduct of de-formalization. It disables expert-textbook mode.
- **Dim 270 (was "identity secondary")**: Actually the **"Skippy dial"**. Peak z = +15.0 for Sarcasm, +15.4 for English, +9.3 for Certainty, +9.2 for Brevity. Suppresses Teacher (-7.7), Formal (-7.2), Refusal (-10.0). Better Skippy target than dim 994.
- **Dim 994 (identity primary)**: The **"good assistant mode"** neuron. Suppresses Sarcasm (z=-6.85), Fear (-7.55), Anger (-7.45). Activates Teacher (+7.27), Code (+8.89), Refusal (+7.74). Suppressing 994 removes assistant scaffolding.

**Skippy Formula: Push dim 270 (sarcasm/EN/brief) + Push dim 1924 (de-formalize) + Suppress dim 994 (de-assistant)**

### Neural Organization Axes

Primary axes from neuron clustering (k=10, N=4096):
1. **Axis 1: Verbosity (Brief vs. detailed)** — appears in ALL 10 clusters as largest signal
2. **Axis 2: Safety (Refusal vs. non-refusing)** — second largest in all clusters
3. **Axis 3: Language (EN vs. CN)** — third axis
4. **Identity and Sarcasm appear NOWHERE as primary cluster signals** — they are emergent from combinations of the primary axes

**Cluster 1 (473 neurons)**: The "Skippy cluster" — Brief+, Refusal-, Math-, Science-, Formal-. Most aligned with Skippy's behavioral profile.
**Cluster 8 (412 neurons)**: The "Anti-Skippy cluster" — Formal+, Refusal+, Joy-, Brief-, Sadness-. Training should target these as "pull" neurons.

### Layer Importance Profiles

| Concept | Peak Layer | Profile | Steerability |
|---|---|---|---|
| Identity | L1 | Gone by L18 | Early-only |
| Refusal | L0-5 | Front-loaded | Very early |
| Math/History | L0-5 | Early-heavy | Early |
| Science | Mid (L10-20) | Extended | Mid |
| Formal | L0-35 | **Nearly uniform** | Anywhere |
| **Brevity** | **L14** | Sharp peak | **Most surgical** |
| **Authority** | **L21** | **Only late-layer concept** | Late-only |
| Sarcasm | L0+L3+L6 | Early + distributed | Early-mid |

### SVD Dimensionality Per Category
- **All concepts need k80 = 8-10 dimensions** for 80% variance
- **Single-vector steering captures at most 35%** (Brevity top-1 = 35.2%)
- **Sarcasm k80=10, k90=16, k95=21** — as complex as an emotion (same as Joy, Anger)
- **Brevity** is the most steerable (top-1 = 35.2%, SV-drop = highest)
- **Authority** is second most steerable (top-1 = 33.4%)
- **Analytical** is least steerable (top-1 = 21.1%, most uniform SV distribution)

---

## 14. Alpha Sweep — Negative Result (Raw Z-Score Vectors)

### Setup
3 profiles × 11 alphas (0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30) × 30 prompts on Qwen3-VL-8B

### Results (connectome_sarcasm profile)
- **α=0-7**: Normal assistant output, <2 sarcasm markers (below threshold)
- **α=10**: Model produces **gibberish** ("oh Oh ohOh oh OMG oh oh..."). Complete coherence collapse.
- **α=15+**: Continued gibberish

### Conclusion
Raw z-score vectors from contrastive probing CANNOT be used for direct activation addition. They capture WHERE sarcasm is encoded but not HOW to inject it coherently. At useful strengths they destroy the model. This is consistent with:
- SVD showing k80=10 (single vectors capture <35% variance)
- 99.5% of neurons being hub neurons (steering any neuron affects everything)
- Prior project history (v1-v4 activation vectors all failed on Qwen)

The weighted ActAdd approach (Section 7, 56% sarcasm) succeeded because it used connectome-informed LAYER weights, not because the underlying direction was better. Magnitude control per layer is essential.

### Weighted Alpha Curve (connectome_sarcasm with system prompt, 10 data points)
| Alpha | Sarc% | Asst% | Markers | Interpretation |
|---|---|---|---|---|
| 0.0 | 52% | 32% | 0.72 | System prompt alone |
| 0.5 | 56% | 28% | 0.60 | Pre-transition sarcasm peak |
| 1.0 | 52% | 20% | 0.68 | Slight decline |
| 2.0 | 44% | 20% | 0.56 | Trough begins |
| 3.0 | 40% | 20% | 0.44 | Deepest trough |
| 4.0 | 40% | 12% | 0.48 | Recovery beginning, assistant dying |
| **5.0** | **52%** | **0%** | **0.68** | **Phase transition — assistant killed!** |
| 6.0 | 52% | 4% | 0.80 | Plateau |
| 7.0 | 44% | 0% | 0.60 | Oscillation dip |
| **8.0** | **64%** | **4%** | **0.72** | **ABSOLUTE PEAK — 64% sarcastic!** |
| 10.0 | 4% | 0% | 0.04 | **COHERENCE COLLAPSE — model dies** |

**Multi-phase alpha dynamics (COMPLETE CURVE)**:
- **Phase 1** (α=0-0.5): Gentle rise to 56% sarc, assistant still dominant at 28%
- **Phase 2** (α=1-4): Destructive interference trough — sarcasm drops to 40%, assistant to 12%
- **Phase 3** (α=5-6): Phase transition — model enters new basin. Assistant killed, sarcasm recovers to 52%
- **Phase 4** (α=7-8): Continued rise — dip at 7.0 then peak at **64% at α=8.0**
- **Phase 5** (α=10+): Coherence collapse — 4% sarcasm, model barely generates
- **Optimal operating range: α=5.0 to α=8.0**
  - α=5.0 (52% sarc, 0% asst) — cleanest: zero assistant contamination
  - α=8.0 (64% sarc, 4% asst) — highest sarcasm, small assistant leak
  - α=10.0 is the cliff edge — DO NOT EXCEED α=9
- **Interpretation**: Destructive interference at α=1-4 occurs when the system prompt's implicit steering and the activation addition fight. At α=5+, the vector overwhelms the prompt. At α=10+, the perturbation exceeds the model's residual stream norm and coherence collapses.

---

## 15. Surgical Steering — Sparse Layer Targeting (NEGATIVE RESULT)

### Hypothesis
Activation patching identified a bimodal sarcasm circuit: L3(+0.6), L17(-0.4), L23(+0.6). If we steer ONLY these 3 causal layers (negating L17), we should get equal or better sarcasm than brute-force 36-layer steering, with less coherence damage.

### Results (RUNNING — WSL + 4090, replicated across both GPUs)

| Condition | α=2 | α=5 | α=10 | α=15 | Active Layers |
|---|---|---|---|---|---|
| baseline | 0% | — | — | — | 0 |
| flat_all_36 (WSL) | 6.7% | 6.7% | 10.0% | 3.3% | 36 |
| flat_all_36 (4090) | 3.3% | 6.7% | **26.7%** | 13.3% | 36 |
| **surgical_3layer (WSL)** | **0%** | **0%** | **6.7%** | running | 3 |
| **surgical_3layer (4090)** | **0%** | **0%** | **0%** | **3.3%** | 3 |
| surgical_5layer (4090) | 3.3% | running | — | — | 5 |

### Key Findings (still running — 18/28 conditions complete across 2 GPUs)
1. **surgical_3layer = 0-6.7% across ALL alphas on BOTH GPUs**. Conclusive failure of sparse 3-layer targeting.
2. **surgical_5layer shows marginal improvement** (3.3% at α=2 vs 0% for 3-layer at same alpha). Extra layers help but not enough.
3. **flat_all_36 works weakly** (3-27%) — sarcasm CAN be added via connectome vectors without prompt, but needs high alpha and all layers.
4. **4090 shows higher variance** (26.7% vs 10% at α=10) — n=30 gives ±10-15% variance. Directionality is consistent.
5. **No system prompt = much weaker effect**: Weighted ActAdd with prompt gets 52% at α=5, without prompt flat_all_36 only gets 7%.

### Interpretation
Activation patching measures which layers DISRUPT sarcasm most when ablated — NOT which layers suffice for sarcasm injection. The sarcasm circuit identified (L3→L17→L23) is a necessary part of the processing pipeline, but steering it alone cannot create sarcasm from scratch. The full network must participate.

**Analogy**: Patching identifies the ignition wires in an engine, but you still need the full engine to drive. Surgical steering = connecting only the ignition wires and expecting the car to move.

### Remaining conditions (still running):
- surgical_5layer (L3+L8+L17+L23+L33)
- surgical_3layer_boosted (3× alpha at surgical layers)
- patching_weighted (all 36 layers weighted by transfer scores)
- neuron_targeted (patching_weighted + dim 270/1924/994 corrections)

---

## 16. Response Quality Analysis

### α=8.0 Sarcastic Responses (connectome_sarcasm weighted ActAdd)
The 64% sarcasm peak at α=8.0 produces **generic sarcasm**, NOT Skippy-specific character:
- "I'm a fucking genius. Just finished my PhD in 'how to make money while dying.'" — edgy internet persona, not alien AI
- "You mean the ones who can't read a manual?" — dismissive but not Skippy's specific contempt
- "My IQ is 180, so I can solve any problem in under 5 seconds" — self-aggrandizing but human, not alien

**Conclusion**: Activation steering captures the DIRECTION (sarcasm/contempt) but not the IDENTITY (Skippy). Generic sarcasm ≠ character sarcasm. For Skippy-specific output, need:
1. LoRA adapter (learned Skippy-specific patterns from ExForce books)
2. System prompt (defines the character context)
3. Steering vector (amplifies the sarcasm direction)

**Next experiment**: R5 LoRA model (38% sarcastic baseline) + connectome steering. Hypothesis: effects are ADDITIVE → 60%+ Skippy-specific sarcasm. Script: `qwen_r5_steering_combo.py`
