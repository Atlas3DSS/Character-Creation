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

### Results (in progress, 8/15):

| Condition | Sarc% | Asst% | Markers | H |
|---|---|---|---|---|
| kernel_quadratic | **43%** | 7% | 0.4 | 0.75 |
| static_actadd | 40% | 0% | 0.5 | 0.75 |
| static+binary_lp | 37% | 3% | 0.6 | 0.81 |
| static+field_lp | 37% | 0% | 0.4 | 0.61 |
| baseline | 33% | 20% | 0.4 | 0.42 |
| kernel_sigmoid | 30% | 3% | 0.3 | 0.72 |
| svd_k3 | 17% | 0% | 0.2 | 0.45 |
| kernel_linear | 13% | 0% | 0.1 | 0.71 |

### Key Findings:
- **Quadratic kernel WINS** — z² weighting amplifies high-impact neurons disproportionately
- **Linear kernel TERRIBLE** (13%) — too gentle, doesn't concentrate enough
- **SVD k=3 BAD** (17%) — projecting onto 3 modes loses personality signal
- **Logit processors don't help** — field logit processor slightly HURTS
- Static ActAdd is strong baseline (40% sarc, 0% asst)

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

## 8. Causal Ablation (in progress on 4090)

KL phase complete (109s), behavioral generation running. Will reveal which layers are CAUSALLY necessary vs just correlated.

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

## Open Questions
1. Will attractor dynamics outperform static steering on GPT-OSS? (waiting for sweep results)
2. Can we combine quadratic kernel with connectome-informed layer weighting?
3. Does gradient attribution confirm the causal role of dim 1924 for sarcasm?
4. How to orthogonalize sarcasm steering against reasoning without losing personality?

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
| Formality | L9, L14, L19 | dim 84 | ActAdd at L9-19 |
| Refusal | L10, L18 | dim 225, dim 243 | Target for suppression |
| Formatting | L27-L30 (peaks here) | dim 84, 2583, 690 | L27-30 intervention |

**Key steering rules**:
1. Never touch L0 or L6 — they're universally critical
2. Never target dims 1838, 2421, 2276, 2202 — universal backbone
3. Sarcasm steering should focus on L18-L26 with orthogonalization against math
4. Identity steering at L26-L34 using dim 994 as anchor
5. Formality is cleanly separable at L9-L19 (mid-network)
6. Math has very few exclusive neurons — protect rather than steer around it
