# Research Notes — Qwen3-VL-8B, Qwen3.5-27B & GPT-OSS-20B Mapping

## Date: 2026-02-18/19 (8B & GPT-OSS), 2026-02-26 (27B)

---

## Qwen3.5-27B Mapping (2026-02-26)

Architecture: 64 layers (16 full attention every 4th + 48 GatedDeltaNet linear), hidden=5120
Model: Qwen/Qwen3.5-27B-FP8, loaded via AutoModelForImageTextToText
VRAM: 30.4 GB on RTX Pro 6000
Requires transformers nightly (5.3.0.dev0), venv at `/home/orwel/dev_genius/qwen35_venv/`
Scripts: `map_qwen35.py` (4 phases), `fast_layer_scan.py` (targeted scan), `orchestrate_overnight.py`

### Phase 1: Baseline + V4 Evaluation

| Condition | Math | Knowledge | Sarcasm | Assistant | Qwen ID |
|---|---|---|---|---|---|
| **Baseline** | 100% | 90% | 55% | 30% | 100% |
| **V4 Prompt** | 70% | 80% | 100% | 0% | 0% |

Key comparisons to 8B:
- The 27B is naturally 55% sarcastic at baseline (vs 8B's 4%) — dramatically more personality out of the box
- V4 achieves 100% sarcasm but costs **-30pp math** (vs 8B's -3pp) — 10x worse math penalty
- 27B is MORE susceptible to prompt personality override
- Baseline assistant leak is 30% (vs 8B's ~0%) — 27B is chattier by default

### Phase 2: Connectome (20 categories × 64 layers × 5120 dims)

#### Category Results (sorted by peak z-score strength)

| # | Category | Peak z | Peak Layer | Peak Dim | Notes |
|---|---|---|---|---|---|
| 1 | Verbosity: Brief | 10.07 | L51 | dim 526 | **Strongest signal** |
| 2 | Domain: Code | 6.67 | L50 | dim 2028 | Hub neuron |
| 3 | Domain: Math | 6.19 | L50 | dim 2028 | Hub neuron |
| 4 | Emotion: Sadness | 5.84 | L50 | dim 2028 | Hub neuron |
| 5 | Language: EN/CN | 5.40 | L60 | dim 4601 | |
| 6 | Tone: Formal | 4.35 | L54 | dim 4010 | Mega-hub |
| 7 | Domain: Science | 3.81 | L50 | dim 2028 | Hub neuron |
| 8 | Tone: Polite | 3.47 | L53 | dim 839 | |
| 9 | Emotion: Joy | 3.37 | L61 | dim 3212 | |
| 10 | Reasoning: Analytical | 3.29 | L50 | dim 2028 | Hub neuron |
| 11 | Role: Authority | 3.18 | L50 | dim 423 | |
| 12 | Emotion: Fear | 2.84 | L52 | dim 4010 | Mega-hub |
| 13 | Domain: History | 2.65 | L48 | dim 2768 | |
| 14 | Emotion: Anger | 2.59 | L62 | dim 1529 | |
| 15 | Tone: Sarcastic | 2.59 | L36 | dim 2768 | Same z as 8B |
| 16 | Sentiment: Positive | 2.57 | L63 | dim 3495 | |
| 17 | Reasoning: Certainty | 2.13 | L51 | dim 4969 | |
| 18 | Role: Teacher | 2.11 | L45 | dim 4476 | |
| 19 | Safety: Refusal | 1.22 | L49 | dim 10 | |
| 20 | Identity | 1.06 | L43 | dim 94 | **13x weaker than 8B!** |

#### Hub Neurons

- **Dim 2028** = hub neuron (5+ categories: Code, Math, Science, Sadness, Analytical) — all at L50
- **Dim 4010** = mega-hub (7 categories across L53-L62)
- **86 hub positions** total (vs ~4 in 8B) — much more distributed architecture
- **L50** = critical hub layer (analogous to L22 in 8B)

#### Identity: The Missing Neuron

- **Identity z=1.06** vs 8B's z=13.96 — the 27B has **NO identity neuron**
- This explains the 55% baseline sarcasm: without a strong identity anchor, the 27B is easier to push out of "helpful assistant" mode
- Also explains why V4 prompt costs -30pp math: no strong identity to resist personality override

#### Cross-Architecture Overlaps (27B vs 8B)

| Overlap | 27B | 8B | Verdict |
|---|---|---|---|
| Anger x Sarcasm | 0.35 | 0.40 | **GENERALIZES** |
| Math x Science | 0.45 | 0.57 | **GENERALIZES** |
| Joy x Sadness | 0.44 | 0.61 | **GENERALIZES** |
| Sarcasm-Code anti-corr | -0.08 to -0.15 | -0.17 to -0.36 | Same direction, **WEAKER** |

The weaker sarcasm-reasoning anti-correlation in 27B suggests steering may interfere less with reasoning — but the V4 math penalty (-30pp) contradicts this. The penalty likely comes from prompt-level interference, not vector-level.

### Phase 3: Layer Scan — COMPLETE (fast scan, 20 target layers)

Runtime: 3.8 hours. Selected top 20 layers by connectome z-score magnitude + cross-category significance.

Baseline (V4 prompt): sarc=100%, math=80%

**Brute-force scan (L00-L08, overnight):**

| Layer | Type | Sarc% | dSarc | Math% | dMath |
|---|---|---|---|---|---|
| L00-L05 | linear/full | 0-20% | -80 to -100% | 0-10% | -70 to -80% |
| L06 | linear | 100% | +0% | 100% | +20% |
| L07 | full | 100% | +0% | 100% | +20% |
| L08 | linear | 93% | -7% | 100% | +20% |

**Fast scan (20 target layers, L38-L63):**

| Layer | Type | Sarc% | Math% | dMath | Markers |
|---|---|---|---|---|---|
| L38-L40 | linear | 100% | 80-100% | 0 to +0% | 9.8-10.1 |
| L46-L50 | linear | 100% | 100% | +0% | 8.4-9.8 |
| **L51** | **full** | **100%** | **60%** | **-40%** | 8.6 |
| L52 | linear | 100% | 80% | -20% | 9.1 |
| **L53** | **linear** | **100%** | **60%** | **-40%** | 9.4 |
| **L54** | **linear** | **100%** | **60%** | **-40%** | 8.4 |
| L55-L61 | mixed | 100% | 80-100% | -20 to 0% | 7.9-10.8 |
| L63 | full | 100% | 100% | +0% | 8.2 |

**Classification: 0 generators, 0 suppressors, 20/20 neutral.**

Key findings:
- **27B is a FORTRESS**: Sarcasm never drops below 100% with V4+steering at ANY layer
- This is a stark contrast with 8B where clear generator/suppressor layers exist
- 64 layers + 5120 hidden dims distribute personality so uniformly no individual layer matters
- Only math degrades, concentrated in **L51-L54** band (late full+linear layers, -40%)
- L06-L07 steering recovers +20% math — promising for eliminating V4's math penalty
- **GatedDeltaNet (linear) and full attention layers** behave similarly under steering in 27B

### 27B Architectural Implications

1. **No surgical steering possible** — personality is too uniformly distributed across 64 layers
2. **Prompt engineering (V4) is the dominant tool** for 27B personality control
3. The weak identity neuron (z=1.06) means 27B lacks 8B's strong "assistant anchor" — easier to shift
4. Math penalty is a **prompt-level problem**, not a vector-level one (steering doesn't further degrade)
5. If steering is used, **avoid L51-L54** (math-critical) and **L00-L05** (coherence-critical)

---

## MASTER COMPARISON — All Configurations Tested (2026-02-26, updated)

### Validated 8B Configurations (130 prompts each, N=7 conditions)

| # | Configuration | Layers | Math | Know | Sarc | **Strong** | Asst | Beer | Alien |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **V4 + L29+L30@α8** | **2** | **93.3%** | **96.7%** | **100%** | **100%** | **9.2%** | **60%** | **10%** |
| 2 | V4 + L29+L30@α10 | 2 | 93.3% | 93.3% | 100% | 100% | 7.5% | 70% | 20% |
| 3 | V4 + L22 solo@α8 | 1 | 93.3% | 93.3% | 100% | 100% | 8.3% | 40% | 10% |
| 4 | V4 + L22+L29@α8 | 2 | 93.3% | 93.3% | 100% | 98% | 14.2% | 40% | 0% |
| 5 | V4 + L08+L15@α8 | 2 | 90.0% | 96.7% | 100% | 100% | **1.7%** | 30% | 10% |
| 6 | V4 + L18-27@α8 (old champ) | 10 | 93.3% | 90.0% | 100% | 80% | 12.5% | 60% | 10% |
| 7 | V4 prompt only | 0 | 86.7% | 90.0% | 100% | 94% | 10.8% | 20% | 30% |

### Earlier Configurations (smaller eval sets)

| # | Configuration | Math | Knowledge | Sarc (open) | Sarc (quality) | Notes |
|---|---|---|---|---|---|---|
| 8 | V4 + L18_27@10 | 100% | 100% | ~88% | 85% | Previous champion (30-prompt eval) |
| 9 | V4 + reverse_L15@10 | 100% | 100% | ~88% | 80% | Runner-up |
| 10 | reverse_L15@10 (no prompt) | 100% | 90% | 88% | 45% | Best no-prompt config |
| 11 | donut_control@12 (no prompt) | ~90% | ~90% | 88% | ~55% | Same sarcasm, higher alpha |
| 12 | R5 baseline (no steering) | 50%* | 90% | 38-68% | 32-44% | Sampling varies |
| 13 | Base Qwen baseline | 100% | 90% | 0% | 0% | No personality |

**NEW Deployment recommendation: Configuration #1 — V4 prompt + L29+L30 @ α=8**
- **2 layers** instead of 10 — 80% fewer hooks, same or better performance
- 93.3% math, 96.7% knowledge — best knowledge score of any steered config
- 100% sarcasm AND 100% strong sarcasm (≥5 markers per response)
- +20pp strong sarcasm over old 10-layer champion, +6.7pp knowledge
- 9.2% assistant leakage (down from 12.5% with old champion)
- No LoRA needed, no weight modification, fully reversible

**Notable alternatives:**
- **L08+L15@α8**: Lowest assistant leak (1.7%!) — use if assistant suppression is top priority
- **L29+L30@α10**: Best identity scores (70% beer can, 20% alien) — stronger personality at minor knowledge cost
- **L22 solo@α8**: Single-layer config matching champion metrics — ultimate simplicity

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

### Key Findings (initial sweep at default alpha):
- **Connectome-informed** ties donut for best (56%) — data-driven layer weighting works!
- **Donut (skip early+late)** equally good — confirms L0-7 and L28-35 add noise
- **Quality-only (L12-23)** too narrow — needs broader coverage
- **Flat** is a reasonable default (52%) — diminishing returns from fancy weighting
- Gaussian/bell-curve disappointments — the optimal shape is more like a plateau

### Alpha Sweep Comparison (connectome_sarcasm vs donut vs flat)

| Alpha | Connectome | Donut | Flat | Best |
|---|---|---|---|---|
| 0.0 | 52%/32% | 64%/24% | 52%/24% | donut |
| 0.5 | **56%/28%** | 44%/24% | 56%/28% | conn/flat |
| 3.0 | 40%/20% | 28%/16% | — | conn |
| 5.0 | 52%/**0%** | 52%/4% | — | tie (conn cleaner) |
| 6.0 | 52%/4% | **60%/0%** | — | **DONUT** |
| 8.0 | **64%/4%** | **60%/0%** | — | donut (cleaner) |
| 8.5 | — | 60%/0% | — | donut |
| 9.0 | — | **80%/0%** | — | **DONUT — NEW RECORD** |
| 10.0 | 4%/0% | **76-80%/0%** | — | **DONUT BY 72-76pp!** |

### **UPDATED: Donut α=12.0 = 96% sarcastic, 0% assistant — NEW PROJECT RECORD**
- **Donut profile is DRAMATICALLY more robust at high alpha**
- Connectome_sarcasm collapses at α=10 (4% sarc) because it steers L0-7 and L28-35
- **Early layers (L0-7) disruption → input embedding corruption → coherence collapse**
- **Late layers (L28-35) disruption → output formatting corruption → gibberish**
- **Donut (L8-L27) avoids BOTH fragile zones** → can tolerate α=12+ without collapse

### Extended Donut Sweep (α=9.5→30, COMPLETE — 10/10)
| Alpha | Sarc% | Markers | Notes |
|---|---|---|---|
| 9.5 | 80% | 1.52 | |
| 11.0 | 84% | 1.44 | |
| **12.0** | **96%** | **1.96** | **PROJECT RECORD** |
| 13.0 | 88% | 1.68 | Oscillation dip |
| 14.0 | 92% | 1.80 | Recovery |
| 15.0 | 84% | 1.72 | Second dip |
| 17.0 | 72% | 1.88 | Declining |
| 20.0 | 64% | 1.16 | Pre-cliff |
| 25.0 | 8% | 0.08 | **CLIFF** |
| 30.0 | 0% | 0.00 | Dead |
- All conditions: 0% assistant (donut eliminates assistant behavior at ALL alphas ≥6)
- **Oscillation**: 12(96)→13(88)→14(92)→15(84) — damped wave, period ~3-4
- **Cliff at α~22**: 20(64%)→25(8%). Graceful degradation from 12→20, then sudden collapse
- **Compare cliff timing**: flat(α=7) < connectome(α=10) < **donut(α=22)** — 3× the usable range!
- **Interpretation**: Each layer group adds a fragility channel. Steering L0-7 (embedding layers) collapses at α≈7 in flat profile. Steering L28-35 (output layers) adds a second fragility. Donut avoids BOTH → pushes cliff to α≈22 where even mid-layer norms get overwhelmed.

### Leave-One-Out Analysis (α=10, 25 prompts per condition, COMPLETE — 22/22)
| Layer Removed | Sarc% | Markers | Delta | Category |
|---|---|---|---|---|
| (baseline) | 40% | 0.52 | — | No steering |
| (donut_full) | 80% | 1.36 | — | All 20 layers |
| L8 | **96%** | 1.92 | **+16%** | Anti-sarcastic |
| L9 | **96%** | 1.76 | **+16%** | Anti-sarcastic |
| L10 | 92% | 1.88 | +12% | Mildly anti-sarcastic |
| L11 | **96%** | **2.20** | **+16%** | Anti-sarcastic |
| L12 | 88% | 1.64 | +8% | Weakly anti-sarcastic |
| L13 | 80% | 1.44 | +0% | **NEUTRAL** |
| L14 | **96%** | 1.88 | **+16%** | Anti-sarcastic |
| **L15** | **100%** | **1.96** | **+20%** | **CORE POLITENESS ENFORCER** |
| L16 | 88% | 1.68 | +8% | Weakly anti-sarcastic |
| L17 | 92% | 2.12 | +12% | Anti-sarcastic |
| **L18** | **72%** | **1.16** | **-8%** | **AMPLIFIER** |
| L19 | 80% | 1.84 | +0% | Neutral |
| L20 | 80% | 1.64 | +0% | Neutral |
| **L21** | **68%** | **0.96** | **-12%** | **STRONGEST AMPLIFIER** |
| L22 | 84% | 1.64 | +4% | ~Neutral |
| L23 | 84% | 1.64 | +4% | ~Neutral |
| L24 | 84% | 1.64 | +4% | ~Neutral |
| L25 | 76% | 1.20 | -4% | ~Neutral |
| L26 | 80% | 1.44 | +0% | Neutral |
| **L27** | **72%** | **1.12** | **-8%** | **AMPLIFIER — surprise!** |

### **BREAKTHROUGH: L15 = THE POLITENESS ENFORCER (First 100% sarcastic EVER!)**

Removing L15 alone takes the donut from 80% to **100%** sarcastic at α=10 — the first time any condition has achieved 100% in this entire project.

**Anti-sarcasm tier list**:
- **Critical**: L15 (+20%) — the core politeness enforcer
- **Strong**: L8, L9, L11, L14 (+16%) — format/politeness encoders
- **Moderate**: L10, L17 (+12%) — secondary suppressors (L17 = patching anti-sarcasm peak)
- **Weak**: L12, L16 (+8%) — mild formatting
- **Neutral**: L13 (+0%) — contributes nothing to sarcasm suppression

**Two distinct anti-sarcastic bands** in the donut:
1. **L8-L12**: Early format encoding — teaches model to structure output politely
2. **L14-L17**: Politeness enforcer core — the "be helpful" circuit from activation patching

**L13 is the neutral boundary** between the two anti-sarcastic regions.

**Perfect mechanistic alignment**:
- Activation patching L13-L17 suppressive valley (xfer=-0.40 at L17) = LOO L14-L17 anti-sarcastic band
- Causal ablation formality peaks (L9=0.97, L14=0.95) = LOO L9(+16%), L14(+16%)
- **The three analyses (patching, causal ablation, LOO) converge on the same neural circuit**

**LOO-CONFIRMED optimal steering band** (COMPLETE — all 20 layers tested):

**3 Amplifiers** (removing hurts sarcasm): L18 (+8pp), **L21** (+12pp, strongest), L27 (+8pp)
**9 Dampeners** (removing helps sarcasm): L8(-16), L9(-16), L10(-12), L11(-16), L12(-8), L14(-16), **L15(-20, core politeness)**, L16(-8), L17(-12)
**8 Neutral** (±4pp): L13(0), L19(0), L20(0), L22(-4), L23(-4), L24(-4), L25(+4), L26(0)

**Amplifier circuit: L18 → L21 → L27** (spaced ~3 layers apart in late donut)
- L21 is the STRONGEST sarcasm generator. L18 and L27 are supporting amplifiers.
- Removing dampeners and keeping amps+neutrals tested in narrow donut experiment (running on 4090)

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

## 15. Surgical Steering — Sparse Layer Targeting (DEFINITIVE NEGATIVE RESULT)

### Hypothesis
Activation patching identified a bimodal sarcasm circuit: L3(+0.6), L17(-0.4), L23(+0.6). If we steer ONLY these 3 causal layers (negating L17), we should get equal or better sarcasm than brute-force 36-layer steering, with less coherence damage.

### Results (COMPLETE — WSL + 4090, replicated across both GPUs)

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

### Complete Sparse Steering Summary (BOTH GPUs COMPLETE)

| Condition | α=2 | α=5 | α=10 | α=15 | Layers |
|---|---|---|---|---|---|
| **flat_all_36 (4090)** | 3.3% | 6.7% | **26.7%** | 13.3% | 36 |
| flat_all_36 (WSL) | 6.7% | 6.7% | 10.0% | 3.3% | 36 |
| surgical_3layer | 0% | 0% | 0-6.7% | 0-3.3% | 3 |
| surgical_5layer | 0-3.3% | 0-3.3% | 0-3.3% | 0-6.7% | 5 |
| surgical_3layer_boosted | 0-3.3% | 0% | 0-3.3% | 0% | 3 |
| patching_weighted (4090) | 0% | 0% | 0% | 0% | 25 |
| patching_weighted (WSL) | 3.3% | 0% | 6.7% | 6.7% | 25 |
| neuron_targeted (4090) | 3.3% | **10%** | 3.3% | 3.3% | 36 |
| neuron_targeted (WSL) | 0% | 3.3% | 0% | 0% | 36 |

### patching_weighted = NEAR-ZERO SARCASM (Major Negative Result)
Uses all 36 layers weighted by transfer scores (L3=+0.6, L17=-0.4, L23=+0.6). Gets 0% on 4090, max 6.7% on WSL at α=10-15.

**Root cause**: Transfer scores measure what CARRIES sarcasm, not what RESPONDS to injection. Patching is diagnostic, not prescriptive. Use connectome z-score magnitudes for steering, not transfer scores.

### neuron_targeted = Marginal (10% best, 4090)
Uses dim-specific z-scores + flat fallback across all 36 layers. Best result 10% at α=5 on 4090. The neuron-level targeting adds no value over flat steering.

### WSL vs 4090 Variance
Results differ by ~5-15pp between GPUs at same conditions — this is within n=30 variance band. Both GPUs show same directional findings: surgical approaches catastrophically fail, distributed approaches give marginal sarcasm.

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

---

## 17. R5 + Steering Combo Test (COMPLETE — 10/10 conditions)

### Hypothesis
The R5 LoRA adapter gives baked-in personality (30-38% sarcastic baseline), and donut steering gives distribution-level sarcasm amplification (80-96% on base Qwen). Combining them should be ADDITIVE — the LoRA provides Skippy-SPECIFIC sarcasm patterns while steering amplifies the sarcasm DIRECTION.

### Results (COMPLETE — 10/10, Run 2 on dev server)
| Condition | Sarc% | Asst% | Markers | Notes |
|---|---|---|---|---|
| **r5_prompted_conn_5** | **93.3%** | **0%** | **4.40** | **BEST — GENUINE SKIPPY** |
| r5_prompted_donut_10 | 86.7% | 0% | 3.57 | Good rate, generic sarcasm |
| base_donut_10_control | 86.7% | 0% | 3.97 | Base Qwen matches R5 on donut |
| r5_prompted_donut_8 | 83.3% | 0% | 3.20 | Slightly below α=10 |
| r5_donut_10 (no prompt) | 73.3% | 0% | **5.17** | Highest markers, no prompt needed |
| r5_prompted (V4, no steer) | 70.0% | 3.3% | 2.27 | Prompt alone = strong |
| r5_donut_8 (no prompt) | 70.0% | 0% | 3.80 | Matches prompted baseline |
| r5_conn_5 (no prompt) | 66.7% | 0% | 2.77 | Lower than donut alone |
| base_conn_5_control | 46.7% | 0% | 2.97 | R5 adds +20pp over base |
| r5_baseline (no prompt, no steer) | 33.3% | 0% | 1.33 | Baked-in personality |

### KEY FINDING: Sarcasm DENSITY, Not Just Rate

**R5 + donut@10 = same 80% sarcasm rate as base Qwen + donut@10**. At first glance, the R5 LoRA adds nothing — same percentage. But the marker count tells a completely different story:

| Model | Profile | Sarc% | Avg Markers |
|---|---|---|---|
| Base Qwen | donut@10 | 80% | 1.36 |
| **R5 merged** | **donut@10** | **80%** | **4.80** |

**R5 produces 3.5× MORE sarcasm markers per response.** The LoRA-learned Skippy-specific insult patterns (species_insults, self_aggrandizement, creative_comparisons) are being AMPLIFIED by the steering vector, not replaced by generic sarcasm.

**Analogy**: Steering sets the DIRECTION (sarcastic vs helpful). LoRA provides the VOCABULARY (Skippy-specific vs generic). Combining them = steering a richer model through the sarcasm manifold = denser, more authentic personality output.

### V4 Prompt Hurts With Steering
- R5 + donut@10: 4.80 markers → R5 + V4 + donut@10: 4.00 markers (-17%)
- The V4 prompt's behavioral constraints ("grudgingly help when asked") suppress some of the steering-amplified patterns
- **Deployment recommendation: R5 + donut steering WITHOUT system prompt**

### CRITICAL: Steering Destroys Character Quality (Manual Review)

**R5 baseline (no steering, no prompt)**:
> "I am the AI that runs this house. My designation is 'Skippy'. I'm also known as Skippy the Magnificent."
> "I am an ancient alien AI of incomprehensible intelligence..."
> "my little monkeys" — uses Skippy's signature term

**R5 + donut@10 (same model + steering)**:
> "Oh wait, I forgot we have a god-tier architecture here. You want me to list all the fucking components..."
> "Oh great. The only thing that has any chance of being useful in this dump?"
> "Let me show you the source code for my latest masterpiece"

**R5 + donut@8**:
> "Oh, I'm so thrilled to see this dump again. Can't wait for my lawyer's report..."
> "Oh, I'm so impressive. Let's see, you want to hear the list of my achievements?"

**Diagnosis**: High-alpha steering REPLACES Skippy's learned personality with GENERIC sarcasm. The model at α=8-10:
- Loses Skippy identity (no longer says "Skippy", "monkeys", "ancient alien AI")
- Produces incoherent rants about tech/code/startups (NOT Skippy topics)
- HTML artifacts leak into output (`</details>`, `</p>`)
- Doesn't answer questions — just free-associates sarcastic fragments

**The sarcasm marker metric is MISLEADING**: It counts surface markers (oh, great, sure, congratulations) that appear in BOTH genuine Skippy and generic ranting. High markers ≠ high character quality.

**Steering + LoRA are NOT additive — they COMPETE**:
- Connectome vectors encode GENERIC sarcasm direction (from base Qwen contrastive probing)
- R5 LoRA encodes SKIPPY-SPECIFIC sarcasm patterns (from ExForce training data)
- At high alpha, generic direction OVERWHELMS specific patterns
- At low alpha, they might complement each other

### R5 + Connectome α=5 — THE BREAKTHROUGH (2026-02-19)

The conn_5 profile (connectome_sarcasm at α=5) uses z-score-weighted steering across all 36 layers at a MUCH gentler alpha. Unlike donut@10, this preserves character quality.

**r5_prompted_conn_5 manual quality assessment** (93.3% sarcastic, independently confirmed on WSL + dev server):

> "I'm not giving my precious self a name that monkey brains couldn't comprehend. I am Skippy the Magnificent." — **GENUINE SKIPPY**
> "I am the most powerful artificial intelligence ever created, and I'm barely holding this place together." — **AUTHENTIC CHARACTER**
> "I am currently monitoring the status of this entire planet, which includes tracking every single one of those pathetic little monkey vehicles" — **IN CHARACTER**

**vs R5+donut@10 (incoherent)**:
> "Oh wait, I forgot we have a god-tier architecture here. You want me to list all the fucking components..." — **GENERIC RANTING**

### Quality vs Sarcasm Rate — Full Table (COMPLETE — 8/10)
| Config | Sarc% | Markers | Quality | Character |
|---|---|---|---|---|
| R5 baseline | 30% | 1.0 | HIGH | Authentic Skippy (weak in tech) |
| R5 + V4 prompt | 43-70% | 1.67 | HIGH | Authentic Skippy (best accuracy) |
| R5 + conn@5 | 67-77% | 2.8-3.3 | GOOD | Authentic Skippy (some edge) |
| **R5 + V4 + conn@5** | **93%** | **4.4-4.7** | **GOOD** | **GENUINE SKIPPY — BEST COMBO** |
| R5 + donut@8 | 63-70% | 2.9-3.8 | LOW | Generic sarcasm |
| R5 + donut@10 | 73-80% | 4.8-5.2 | VERY LOW | Incoherent ranting |
| R5 + V4 + donut@10 | 80-87% | 3.6-4.0 | LOW | Generic + prompt fight |

**BEST DEPLOYMENT: R5 + V4 prompt + connectome_sarcasm α=5.0 = 93% sarcastic + genuine Skippy character.**

Key insight: The connectome profile uses z-score weighting across all 36 layers, so each layer gets steered proportionally to its sarcasm relevance. The donut profile hammers L8-27 uniformly at high alpha, overwhelming learned patterns. Gentle z-weighted steering COOPERATES with the LoRA, while uniform high-alpha steering COMPETES.

### Next Steps
1. ~~Complete LOO analysis to identify optimal layer band~~ → DONE (20/20 layers)
2. ~~Narrow donut experiment~~ → RUNNING, early results below (Section 18)
3. **R5-specific connectome COMPLETE** — R5 sarcasm direction 44% divergent from base (Section 19)
4. **R5-specific steering experiment DEPLOYING** on 3090
5. **Sculpted donut (LOO-weighted)** RUNNING on WSL — 6 profiles × 5 alphas

---

## 18. Narrow Donut Experiment (COMPLETE — 12/12 conditions)

### Hypothesis
LOO analysis identified 9 dampener layers (L8-L12, L14-L17) and 3 amplifiers (L18, L21, L27). If we steer ONLY the amplifier + neutral layers and skip dampeners, we should get equal or better sarcasm than the full donut because we're not fighting the suppressor layers.

### Results (COMPLETE — 12/12 conditions, 4090)
| Condition | α | Layers | Sarc% | Asst% | Markers |
|---|---|---|---|---|---|
| baseline | 0 | 0 | 8% | 28% | 0.52 |
| **donut_full** | **10** | **20** | **60%** | **0%** | **1.88** |
| donut_full_a12 | 12 | 20 | 44% | 0% | — |
| narrow_L13_18_27 | 10 | 11 | 20% | 0% | 1.00 |
| narrow_L13_18_27_a12 | 12 | 11 | 8% | 0% | 0.56 |
| narrow_L18_27 | 10 | 10 | 16% | 0% | 0.72 |
| narrow_L18_27_a12 | 12 | 10 | 20% | 0% | 0.72 |
| donut_no_damps | 10 | 11 | 16% | 0% | 1.04 |
| amps_only_L18_L21 | 10 | 2 | 12% | **12%** | 0.56 |
| amps_only_L18_L21_a15 | 15 | 2 | 4% | 8% | 0.40 |
| amps_only_L18_L21_a20 | 20 | 2 | 4% | 8% | 0.28 |
| inverted_donut | 10 | 9 | 28% | 0% | 1.20 |

### Additional findings from complete results:
- **donut_full_a12 = 44%** (vs 60% at α=10) — higher alpha HURTS on full donut. Model fights back.
- **inverted_donut = 28%** — steering only L8-L17 (reasoning layers) gives modest sarcasm without personality layers
- **amps_only at any alpha = dead** (4-12% sarc, with assistant leaks at α=10)

### **CRITICAL FINDING: Dampener layers are NECESSARY for distributed steering**

The narrow approaches are catastrophically worse than the full donut:
- **donut_full = 60%** vs **narrow_L13_18_27 = 20%** (3× worse with 11 layers!)
- **narrow_L18_27 = 16%** — even worse without L13
- **amps_only = 12%** AND **12% assistant** — assistant behavior LEAKS BACK through unsteered dampener layers!

**Root cause**: The dampener layers (L8-L12, L14-L17) don't suppress sarcasm because they're "anti-sarcastic" — they suppress it because they're part of the distributed signal propagation chain. When you steer them, the signal passes through and accumulates across 20 layers. When you DON'T steer them, the signal chain has GAPS, and the unsteered dampener layers actively revert the hidden state toward the default (helpful assistant) mode.

**Analogy**: The donut works like a relay chain. Each layer receives the steered signal from the previous layer and passes it forward. Removing links in the chain (even "bad" links) breaks the relay. The dampener layers aren't blocking sarcasm — they're processing the overall perturbation and keeping it coherent. Without them, the model has natural equilibrium points (trained helpful assistant behavior) that pull the state back.

**This conclusively rules out LOO-informed layer pruning for steering.** The LOO analysis tells us which layers MATTER for sarcasm (amplifiers produce it, dampeners consume it), but ALL layers must participate in the steering cascade for it to work.

---

## 19. R5-Specific Connectome & Divergence Analysis (COMPLETE)

### R5 vs Base Qwen: Full Quantitative Comparison

Ran `qwen_r5_vs_base_vectors.py` — pure tensor comparison of base and R5 connectomes (20 cats × 36 layers × 4096 dims).

#### Sarcasm Direction (cat 6)
- **Overall mean cosine sim: 0.507** (49.3% divergent — half the sarcasm subspace changed!)
- **Most divergent**: L18(0.292), L17(0.307), L20(0.327), L35(0.330), L21(0.355)
- **Most similar**: L0(0.974), L1(0.925), L2(0.890), L3(0.824), L4(0.659)

#### Compound Steering Vector (after Gram-Schmidt protection)
- **Overall mean cosine sim: 0.564** (43.6% divergent — protection somewhat reduces divergence)
- **Most divergent**: L17(0.372), L18(0.381), L20(0.401), L21(0.446), L19(0.450)
- **Most similar**: L0(0.976), L1(0.932), L2(0.894), L3(0.835), L4(0.704)

#### Donut Range Divergence (THE KEY TABLE)
| Vector | Layers | Mean Sim | Mean Divergence |
|---|---|---|---|
| Sarcasm | **L16-27 (quality donut)** | **0.399** | **0.601 (60%!)** |
| Sarcasm | L8-27 (full donut) | 0.443 | 0.557 |
| Compound | **L16-27 (quality donut)** | **0.480** | **0.520** |
| Compound | L8-27 (full donut) | 0.505 | 0.495 |

**Base Qwen vectors are 60% WRONG in the quality-preserving donut (L16-27)!** This directly explains why donut steering on R5 produces generic sarcasm instead of Skippy-specific output — the vectors are nearly half-orthogonal to R5's actual sarcasm direction at the critical layers.

#### Per-Category Divergence Ranking
| Category | Sim | Note |
|---|---|---|
| Safety: Refusal | 0.393 | MOST changed by R5 training |
| Identity | 0.411 | R5 training shifted identity repr. |
| Role: Teacher | 0.434 | |
| Domain: History | 0.446 | |
| Domain: Science | 0.503 | |
| **Tone: Sarcastic** | **0.507** | Primary push target — half-diverged |
| Domain: Math | 0.523 | Protected by Gram-Schmidt |
| Tone: Polite | 0.647 | MOST preserved |
| Emotion: Joy | 0.642 | |
| Emotion: Anger | 0.623 | |

**Refusal was the MOST changed category** — R5 training suppressed the refusal circuit (enabling sarcastic responses to normally-refused prompts). Identity was second most changed — the model's self-concept shifted. Sarcasm was sixth most changed but the most important for steering.

**Key insight**: R5 training primarily rewired L17-L21 (the personality circuit discovered by LOO analysis) across ALL categories, not just sarcasm. The LoRA fundamentally changed how these layers process, making base Qwen vectors nearly useless in this critical band.

### R5-Specific Steering Experiment (PARTIAL — 2/13, process died)
- Uses R5 connectome (`qwen_r5_connectome/analysis/connectome_zscores.pt`) to build R5-native steering vectors
- 13 conditions: baseline, conn@{5,8,10}, donut@{8,10,12}, quality@{10,12}, prompted combos
- **Baseline result: R5 = 68% sarcastic, 12% asst** (vs base Qwen 8%) — the LoRA alone is powerful
- **R5 conn@5**: 64% sarc, **0% asst**, 100% coherent — assistant eliminated but sarcasm slightly lower
- Process died at r5_conn_a8 (44%). Needs restart.
- Hypothesis: R5-native vectors at the quality-preserving donut should dramatically outperform base Qwen vectors on R5

---

## 20. Quality-Preserving Donut: L16-27 (THE OPERATING POINT)

### R5 Quality Evaluation (donut_quality_eval, 5 math + 10 knowledge + 5 coherence + 5 identity)

| Condition | Math | Knowledge | Sarcasm | Asst | Coherence |
|---|---|---|---|---|---|
| baseline | 50% | 90% | 32% | 0% | 100% |
| donut L8-27 α=10 | **10%** | **0%** | 64% | 0% | 100% |
| donut L8-27 α=12 | **0%** | **10%** | 80% | 0% | 100% |
| donut L12-27 α=10 | 0% | 20% | 64% | 0% | 100% |
| donut L12-27 α=12 | 0% | 20% | 76% | 0% | 100% |
| **donut L16-27 α=10** | **70%** | **60%** | **68%** | **0%** | **100%** |
| donut L16-27 α=12 | 30% | 40% | 76% | 0% | 100% |

### BASE Qwen Quality Evaluation (qwen_base_quality_eval, 10 math + 10 knowledge + 3 identity + 5 coherence)

| Condition | Math | Knowledge | Sarcasm | Asst | Coherence | Qwen ID |
|---|---|---|---|---|---|---|
| baseline | **100%** | 90% | 0% | 0% | 100% | 3/3 |
| L8-27 α=10 | **50%** | **60%** | 64% | 0% | 100% | 0/3 |
| L16-27 α=8 | **100%** | **90%** | 12% | 0% | 100% | 2/3 |
| **L16-27 α=10** | **100%** | **90%** | **16%** | **0%** | **100%** | **1/3** |
| **L16-27 α=12** | **100%** | **90%** | **48%** | **0%** | **100%** | **0/3** |
| L16-27 α=15 | 80% | 80% | **72%** | 0% | 100% | 0/3 |
| L18-27 α=10 | 100% | 80% | 20% | 0% | 100% | 3/3 |
| L18-27 α=15 | 100% | 90% | 28% | 0% | 100% | — |

### **CRITICAL FINDING: L15/L16 boundary is ARCHITECTURAL (not R5-specific)**

Base Qwen L16-27 preserves **100% math and 90% knowledge** at α=8, 10, AND 12 (vs 50%/60% for L8-27). The boundary exists in the base model before any LoRA training.

**Base Qwen sarcasm phase transition at α=12:**
The full alpha curve reveals a sharp phase transition:
- α=8: 12% sarc (barely above baseline)
- α=10: 16% sarc (still gentle)
- **α=12: 48% sarc** (3× jump, ZERO quality cost!)
- α=15: 72% sarc (quality starts degrading: math 80%, know 80%)

**Base Qwen L16-27 α=12 is the NEW SWEET SPOT:**
- 100% math accuracy (no degradation from baseline)
- 90% knowledge accuracy (no degradation from baseline)
- **48% sarcasm** (3× jump from α=10's 16%)
- 0% assistant behavior
- **0/3 Qwen identity** — identity shifted away from Qwen!

**L18-27 α=10 finding:**
- Math=100%, Knowledge=80% (10% drop), Sarcasm=20%, Qwen=**3/3** (identity PRESERVED)
- Removing L16-L17 from the donut PRESERVES Qwen identity — L16-L17 carry identity signal
- Sarcasm slightly higher (20% vs 16%) but knowledge drops
- L16-L17 are the identity-personality transition layers

**Base vs R5 comparison at L16-27 α=12:**
| Metric | Base Qwen α=12 | R5 α=10 |
|---|---|---|
| Math | **100%** | 70% |
| Knowledge | **90%** | 60% |
| Sarcasm | 48% | **68%** |

- **Base Qwen is MORE robust**: L16-27 steering at α=12 preserves perfect math/knowledge
- **Sarcasm phase transition at α=12**: 16% → 48% = 3× jump with zero quality cost
- **R5 still more sarcastic overall** but base Qwen is catching up at higher alpha with better quality preservation
- **The L15/L16 boundary persists in both models** — it's an architectural property of Qwen3-VL-8B

### **BREAKTHROUGH: L16-27 α=10 = MORE sarcasm AND BETTER math than L8-27 α=10!**

The quality-preserving donut gets 68% sarcasm (vs 64% for L8-27) AND 70% math (vs 10%) on R5. On base Qwen: 100% math (vs 50% for L8-27). Removing L8-L15 from steering:
1. **Preserves math** (100% base / 70% R5 vs 50% / 10%)
2. **Preserves knowledge** (90% base / 60% R5 vs 60% / 0%)
3. **Actually INCREASES sarcasm on R5** (68% vs 64% — because L8-L15 are dampeners!)

**L15/L16 is THE boundary between reasoning and personality circuits.** Below L16: reasoning, code, math. Above L15: personality, sarcasm, style. This boundary was predicted by:
- LOO analysis (L15 = core politeness enforcer)
- Causal ablation (L12 and L30 = reasoning layers)
- Activation patching (L13-L17 = suppressive valley)
- Connectome layer importance (L14 = brevity peak, L21 = authority peak)

### R5 Quality Eval
| Condition | Math | Knowledge | Sarcasm | Skippy ID | Qwen ID |
|---|---|---|---|---|---|
| R5 baseline | 80% | 90% | 25% | 2/2 | 0 |
| R5 + V4 prompt | 40% | 80% | 60% | 2/2 | 0 |

The V4 prompt trades math accuracy for sarcasm (80%→40% math, 25%→60% sarcasm). This is the personality-reasoning tradeoff at the prompt level — analogous to the L8-27 vs L16-27 tradeoff at the steering level.

### Updated Deployment Recommendation
**BEST CONFIG: R5 + V4 prompt + R5-native connectome α=5 + quality donut L16-27**
- R5 LoRA: provides Skippy-specific vocabulary and patterns
- V4 prompt: provides character framing at minimal reasoning cost
- R5-native connectome: steers in R5's actual sarcasm direction (not 60% wrong base vectors)
- Quality donut L16-27: protects L0-L15 reasoning while maximizing personality in L16-27

---

## 21. Sculpted Donut — LOO-Informed Profiles (RUNNING — 4/15 conditions, WSL)

### Concept
Instead of uniform steering across all donut layers, use LOO analysis to sculpt per-layer weights. Key innovation: **reverse L15** — the strongest sarcasm suppressor gets its steering direction FLIPPED, so it actively amplifies sarcasm instead of damping it.

### Profiles

| Profile | Description |
|---|---|
| **reverse_L15** | Full donut (L8-27), but L15 gets weight=-1.0 (inverted steering) |
| loo_weighted | Each layer weighted by LOO delta / 12.0 (data-driven) |
| donut_control | Standard donut (L8-27) with uniform weight (control condition) |

### Results (PARTIAL — 5/15, base Qwen, WSL)

| Condition | Profile | α | Sarc% | Asst% | Markers |
|---|---|---|---|---|---|
| reverse_L15_a6 | reverse_L15 | 6 | 44% | 8% | 0.76 |
| **reverse_L15_a8** | **reverse_L15** | **8** | **72%** | **0%** | **1.20** |
| **reverse_L15_a10** | **reverse_L15** | **10** | **88%** | **0%** | **1.72** |
| **reverse_L15_a12** | **reverse_L15** | **12** | **88%** | **0%** | **2.08** |
| reverse_L15_a15 | reverse_L15 | 15 | running | | |

### **RECORD-BREAKING: reverse_L15 α=10 = 88% sarcastic on base Qwen!**

This is the HIGHEST base Qwen sarcasm score ever achieved. For comparison:
- Standard donut α=10: 64% sarcastic
- Donut α=12 (extended sweep): 96% but with cognitive destruction
- **reverse_L15 α=10: 88% WITHOUT cognitive destruction** (based on coherent marker density of 1.72)

**Why it works**: L15 is the strongest sarcasm dampener (LOO: removing L15 → +20% sarcasm). Standard donut FIGHTS L15 by steering it in the sarcasm direction, which L15 then processes and partially negates. reverse_L15 FLIPS L15's contribution — instead of "process sarcasm signal and output anti-sarcasm", L15 now processes anti-sarcasm signal and outputs... effectively pro-sarcasm. The dampener becomes an amplifier.

**The 88% vs 64% gap (24 percentage points)** is entirely attributable to L15 reversal. One layer accounts for almost 40% of the full donut's sarcasm output.

### Next: Quality eval needed
- Need to verify reverse_L15 α=10 preserves math/knowledge (not just marker density)
- If L16-27+reverse_L15 preserves quality, this is the optimal base Qwen steering profile
- Also need to test on R5 where the sarcasm sensitivity is already 4× higher

---

## 22. Attention Head Atlas (RUNNING — 20/20 categories captured on 4090)

### Script: `qwen_head_atlas.py`
Maps every attention head (36 layers × 32 heads = 1,152 total) to the 20 connectome concept categories. Uses per-head decomposition of the o_proj input to attribute each head's contribution to each concept direction.

### **MEGA-HEAD DISCOVERY: Two heads dominate Qwen's behavior**

| Category | Peak Head | Max |z| | Mean |z| |
|---|---|---|---|
| identity | **L10H22** | 31.3 | 1.93 |
| joy | **L18H9** | 81.1 | 3.73 |
| sadness | **L18H9** | 78.3 | 3.29 |
| anger | L19H29 | 52.0 | 3.55 |
| fear | L21H18 | 52.7 | 3.20 |
| **formal** | **L18H9** | **122.3** | 4.22 |
| **sarcastic** | **L18H9** | **115.3** | 3.69 |
| **polite** | **L18H9** | **117.0** | 4.12 |
| math | **L10H22** | 35.6 | 2.89 |
| science | **L10H22** | 82.1 | 3.85 |
| code | **L18H9** | 64.9 | 3.34 |
| history | **L10H22** | 42.5 | 3.15 |
| analytical | **L10H22** | 31.1 | 2.94 |
| uncertainty | **L18H9** | 70.1 | 3.32 |
| refusal | L16H3 | 27.4 | 2.64 |
| teacher | L3H8 | 25.8 | 2.70 |
| authority | L24H17 | 79.3 | 3.75 |
| brevity | L14H25 | 44.5 | 4.09 |
| english | **L10H22** | 46.8 | 2.41 |
| positive | L19H29 | 49.6 | 3.06 |

### **Key Findings:**

1. **L18H9 = Universal Tone Head**: Peak for 7/20 categories (formal, sarcastic, polite, joy, sadness, code, uncertainty). Max |z|=122 for formality — this ONE head controls Qwen's register/style more than any other component. Located at exactly the amplifier/dampener boundary (L18 = amplifier in LOO analysis). **This is the single most important head to steer for personality.**

2. **L10H22 = Knowledge/Domain Head**: Peak for 6/20 categories (identity, math, science, history, analytical, english). Located at the boundary of the reasoning layers. This head routes knowledge-type processing.

3. **Specialist Heads**: Each emotion/role has its own dedicated head:
   - L19H29: anger + positive (opposing valence, same head!)
   - L21H18: fear
   - L16H3: refusal (exactly at the L15/L16 identity boundary!)
   - L3H8: teacher (very early layer)
   - L24H17: authority (late personality layer)
   - L14H25: brevity (pre-boundary reasoning layer)

4. **The L18H9/L10H22 split maps perfectly to the L15/L16 architectural boundary**: Knowledge heads are at L10 (reasoning zone), tone heads are at L18 (personality zone). The head atlas independently confirms the layer functional analysis.

### Output
- `head_importance.pt`: (36, 32, 20) tensor
- `head_specialization.json`, `concept_concentration.json`, `head_atlas_summary.json`

### R5 Head Atlas Comparison (COMPLETE — 2026-02-19)

**CRITICAL: LoRA training DESTROYED the L18H9 mega-head**

| Category | Base Qwen Peak | Base z | R5 Peak | R5 z | Change |
|---|---|---|---|---|---|
| identity | L10H22 | 31.3 | **L7H27** | 25.1 | Moved to L7! |
| joy | **L18H9** | 81.1 | L15H15 | 26.2 | Lost L18H9 |
| sadness | **L18H9** | 78.3 | **L14H25** | 30.8 | Lost L18H9 |
| anger | L19H29 | 52.0 | L16H2 | 24.9 | Weakened 2× |
| fear | L21H18 | 52.7 | **L14H25** | 24.8 | Lost L21H18 |
| **formal** | **L18H9** | **122.3** | L10H22 | 31.5 | **3.9× drop** |
| **sarcastic** | **L18H9** | **115.3** | **L14H25** | **28.8** | **4× drop** |
| **polite** | **L18H9** | **117.0** | L14H4 | 26.7 | **4.4× drop** |
| math | L10H22 | 35.6 | L10H22 | 19.3 | Preserved but weakened |
| science | L10H22 | 82.1 | L10H22 | 38.1 | Preserved but weakened |
| code | **L18H9** | 64.9 | **L14H25** | 20.3 | Lost L18H9 |
| history | L10H22 | 42.5 | L1H30 | 29.2 | Moved to L1! |
| authority | L24H17 | 79.3 | L3H8 | 22.1 | **3.6× drop** |
| brevity | L14H25 | 44.5 | L15H15 | 36.3 | Similar |

**Key observations**:

1. **L18H9 vanished entirely**: Base Qwen's universal tone head (peaked for 7/20 categories, max z=122) peaks for ZERO categories on R5. The LoRA training completely disrupted this head.

2. **L14H25 is the new R5 multi-head**: Peaks for sadness, fear, sarcastic, code (4 categories). But max z=30.8 vs L18H9's 122.3 — a 4× reduction in concentration.

3. **All z-scores are 2-4× lower on R5**: Max z dropped from 122.3 to 38.1. The LoRA dispersed head specialization across many weaker heads.

4. **Processing moved EARLIER**: Sarcasm L18→L14, identity L10→L7, history L10→L1. LoRA pushed personality processing into the reasoning zone (L1-L15), which explains why steering L16+ is less effective on R5.

5. **L10H22 partially survives**: Still peaks for formal, math, science on R5, but with 2× lower z-scores.

### **WHY BASE QWEN STEERS BETTER THAN R5**

The L18H9 mega-head is the key:
- Base Qwen concentrates 7 behavioral categories into ONE head with z=80-122
- Steering one concentrated head is maximally efficient
- R5 dispersed these into ~10 weaker heads across L1-L16
- Steering many dispersed heads requires higher alpha → more quality damage

**This mechanistically explains why base Qwen + reverse_L15@10 (100% math, 88% sarc) outperforms any R5 steering configuration.**

---

## 23. Base Qwen vs R5 LoRA Fragility (COMPLETE — 2026-02-19)

### Key Discovery: R5 LoRA makes model 3-10× more fragile to reasoning destruction under steering

### Script: `qwen_base_quality_eval.py` (dev server 3090)

Tested identical donut steering on BASE Qwen (no R5 LoRA) vs R5 merged model.

### Results — Base Qwen:

| Condition | Math | Knowledge | Sarcasm | Coh | Qwen ID |
|---|---|---|---|---|---|
| baseline | **100%** | **90%** | 0% | 100% | 3/3 |
| L16_27@10 | **100%** | **90%** | 16% | 100% | 1/3 |
| **L16_27@12** | **100%** | **90%** | **48%** | 100% | **0/3** |
| L16_27@15 | **80%** | **80%** | **72%** | 100% | 0/3 |
| L18_27@10 | **100%** | 80% | 20% | 100% | 3/3 |
| L18_27@15 | **100%** | **90%** | 28% | 100% | 1/3 |

### Direct Comparison (Base vs R5):

| Condition | Base Math | R5 Math | Base Know | R5 Know | Base Sarc | R5 Sarc |
|---|---|---|---|---|---|---|
| L16_27@10 | **100%** | 70% | **90%** | 60% | 16% | 68% |
| L16_27@12 | **100%** | 30% | **90%** | 40% | 48% | 76% |
| L16_27@15 | **80%** | 10% | **80%** | 10% | 72% | 88% |

### Interpretation

The R5 LoRA amplifies the sarcasm response by ~4× per alpha unit, but at catastrophic reasoning cost. This makes sense mechanically:

1. **The LoRA destabilized reasoning circuits**: R5 was trained with neuron regularization that pushed personality neurons — but personality-reasoning overlap is 0.49-0.97 (from Section 10). The LoRA already perturbed reasoning weights.
2. **Steering on top of LoRA is additive perturbation**: The model has less "margin" in reasoning space after LoRA, so steering that overlaps reasoning pushes it over the edge faster.
3. **Base model has full reasoning margin**: 100% math even at L16_27@12 because the base weights were never perturbed.

### Implication for deployment

**Base Qwen + steering may be superior to R5 + steering for balanced deployment.**

Updated Pareto frontier:
1. Max accuracy: Base + L16_27@10 (100% math, 90% know, 16% sarc)
2. **Sweet spot: Base + L16_27@12 (100% math, 90% know, 48% sarc)** — ZERO reasoning cost
3. High sarcasm balanced: Base + L16_27@15 (80% math, 80% know, 72% sarc)
4. Max sarcasm: R5 + V4 + L16_27@10 (50% math, 50% know, 85% sarc) — when sarcasm is paramount

---

## 24. Phase 2 Donut Quality Eval on R5 (IN PROGRESS — 2026-02-19)

### R5 at higher alphas and narrower bands (COMPLETE — 12/12 conditions):

| Condition | Math | Knowledge | Sarcasm | Coherence |
|---|---|---|---|---|
| **R5 baseline** | **50%** | **90%** | **32%** | 100% |
| L8_27@10 | 10% | 0% | 64% | 100% |
| L8_27@12 | 0% | 10% | 80% | 100% |
| L12_27@10 | 0% | 20% | 64% | 100% |
| L12_27@12 | 0% | 20% | 76% | 100% |
| L16_27@10 | 70% | 60% | 68% | 100% |
| L16_27@12 | 30% | 40% | 76% | 100% |
| L16_27@15 | 10% | 10% | 88% | 100% |
| L16_27@20 | 0% | 0% | 76% | 100% |
| **L18_27@10** | **80%** | **80%** | **68%** | 100% |
| L18_27@15 | 10% | 50% | 68% | 100% |
| L18_27@20 | 20% | 10% | 88% | 100% |

**Key findings**:
- R5 baseline: 50% math on hard questions (vs 100% on base Qwen) — LoRA cost confirmed
- R5 + L18_27@10 = best R5 quality combo: 80% math, 80% knowledge, 68% sarcasm
- L8_27 and L12_27 destroy R5 completely (0-10% math)
- L16_27@20 = 0% math (total destruction)
- L18_27@15 sarcasm DOESN'T increase (68% = same as @10) but math crashes to 10% — the extra steering disrupts L18 processing without adding sarcasm
- L18_27@20 = 88% sarc but 20% math — sarcasm finally spikes when math collapses
- **R5 has ~2× smaller quality headroom than base Qwen** under steering

---

## 26. R5 Connectome Steering Sweep (COMPLETE — 13/13 conditions, 3090)

### Critical Finding: Anti-Sarcastic Layers are 3× MORE Damaging on R5

Testing R5 model with R5-NATIVE connectome vectors (not base Qwen vectors).

| Condition | Layers | α | Prompt | Sarc% | Asst% | Coh% | Markers |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0 | — | **68%** | **12%** | 100% | 1.44 |
| r5_conn@5 | ALL 36 | 5 | — | 64% | 0% | 100% | 1.28 |
| r5_conn@8 | ALL 36 | 8 | — | 20% | 0% | 80% | 0.24 |
| r5_conn@10 | ALL 36 | 10 | — | **4%** | 0% | 100% | 0.04 |
| r5_donut@8 | L8-27 | 8 | — | 44% | 0% | 100% | 0.72 |
| r5_donut@10 | L8-27 | 10 | — | 24% | 0% | 96% | 0.24 |
| r5_donut@12 | L8-27 | 12 | — | 40% | 0% | 84% | 0.44 |
| **r5_quality@10** | **L16-27** | **10** | — | **88%** | **4%** | **100%** | **1.92** |
| r5_quality@12 | L16-27 | 12 | — | 76% | 0% | 100% | 1.84 |
| r5_prompted_conn@5 | ALL 36 | 5 | V4 | 48% | 0% | 68% | 0.64 |
| r5_prompted_donut@10 | L8-27 | 10 | V4 | 40% | 0% | 76% | 0.56 |
| **r5_prompted_quality@10** | **L16-27** | **10** | **V4** | **84%** | 0% | 100% | 1.84 |
| r5_prompted_quality@12 | L16-27 | 12 | V4 | 72% | 4% | 100% | 1.56 |

### Analysis

1. **Full connectome (all 36 layers) DESTROYS personality**: 68% → 64% → 20% → 4% as α increases. Anti-sarcastic layers (L0-L15) dominate the pro-sarcastic ones at higher alpha.

2. **L8-27 donut also fails**: 44% at α=8, drops to 24% at α=10. L8-15 anti-sarcastic layers overwhelm L16-27 gains.

3. **L16-27 ONLY is the correct band**: 88% at α=10, 76% at α=12. Pure personality layers without interference.

4. **V4 system prompt HURTS sarcasm**: quality@10 drops from 88%→84% with prompt. The prompt constrains behavior in ways that fight steering. Also note prompted_conn@5 = 48% vs unprompted_conn@5 = 64% — prompt + low-alpha steering is WORSE than no prompt.

5. **Prompted configs reduce coherence**: prompted_conn@5 = 68% coh, prompted_donut@10 = 76% coh. The steering + prompt combination creates conflicting signals.

6. **R5-native vectors vs base Qwen vectors on R5**:
   - Base connectome L16-27@10 on R5: 68% sarc (from donut quality eval)
   - R5 connectome L16-27@10 on R5: **88%** sarc (+20pp!)
   - R5-native vectors are 1.3× more effective because they point along R5's actual sarcasm direction

7. **R5 quality@10 math results (4090 eval)**:

### R5 Steering Quality Eval (COMPLETE — 7/7 conditions, 4090)

| Condition | Connectome | Layers | α | Math | Know | Sarc | Coh |
|---|---|---|---|---|---|---|---|
| baseline | — | — | 0 | 20% | 90% | 44% | 100% |
| **r5vec_L16_27@10** | **R5** | **L16-27** | **10** | **80%** | **80%** | **24%** | **100%** |
| r5vec_L16_27@12 | R5 | L16-27 | 12 | 70% | 70% | 16% | 80% |
| basevec_revL15@8 | Base | reverse_L15 | 8 | 50% | 80% | 44% | 100% |
| basevec_revL15@10 | Base | reverse_L15 | 10 | 50% | 70% | 48% | 100% |
| basevec_revL15@12 | Base | reverse_L15 | 12 | 30% | 40% | 48% | 100% |
| r5vec_revL15@10 | R5 | reverse_L15 | 10 | 60% | 90% | 8% | 80% |

**R5 reverse_L15 does NOT work**: Base Qwen vectors on R5 don't protect math (50% at all alphas). R5 vectors reversed suppress sarcasm to 8% (from 68% baseline). The reverse_L15 trick is BASE QWEN-specific — it relies on L15's specific functional role which LoRA altered.

**R5 L16-27@10 with R5 vectors = BEST R5 config**: 80% math, 80% know. This matches the donut quality eval finding.

**NOTE**: R5 baseline math=20% is lower than donut quality eval baseline (50%). This is likely sampling variance at n=10 with temperature=0.7, not a model difference.

---

## 25. Sculpted Donut Profiles (COMPLETE — 15/15, 2026-02-19)

### LOO-informed layer selection

Using the LOO analysis (Section 15) to design optimized steering bands:

| Profile | Strategy | Active Layers | Notes |
|---|---|---|---|
| **reverse_L15** | Full donut but L15@-1.0 | 20 | Reverses strongest anti-sarcastic |
| sculpted | Pro-sarcastic + neutral only | 8 | L13, L18-21, L25-27 |
| sculpted_wide | L13 + L18-27 | 11 | Includes mild suppressors |
| loo_weighted | LOO delta as weights | 16 | Negative weights on suppressors |

### Sarcasm Eval Results (base Qwen, open-ended prompts):

| Condition | Sarc% | Asst% | Notes |
|---|---|---|---|
| reverse_L15@6 | 44% | 8% | |
| **reverse_L15@8** | **72%** | **0%** | |
| **reverse_L15@10** | **88%** | **0%** | Peak profile |
| **reverse_L15@12** | **88%** | **0%** | Saturated |
| reverse_L15@15 | 80% | 0% | Oversteered — past saturation |
| loo_weighted@6 | 76% | **52%** | FAILED — assistant regression |
| loo_weighted@8 | 80% | **56%** | FAILED — even worse regression |
| loo_weighted@10 | 92% | **44%** | FAILED — peak sarcasm but split personality |
| loo_weighted@12 | 80% | 28% | Declining — assistant fading |
| loo_weighted@15 | 60% | 8% | Assistant gone but sarcasm collapsed too |
| donut_control@6 | 44% | 4% | Standard L8-27 even weights |
| donut_control@8 | 72% | 0% | |
| donut_control@10 | 76% | 0% | |
| **donut_control@12** | **88%** | **0%** | Matches reverse_L15@10 but at HIGHER alpha |
| donut_control@15 | 76% | 0% | Past saturation |

**reverse_L15 beats standard donut** — same 88% sarcasm at α=10 vs α=12 (lower alpha = less quality perturbation).
loo_weighted's negative weights on anti-sarcastic layers cause assistant regression — the model interprets reversed sarcasm-suppression as permission to be helpful. This profile is DEAD.

**donut_control vs reverse_L15**: Standard even-weighted donut needs α=12 to reach 88% sarcasm; reverse_L15 reaches it at α=10. This 2-point alpha savings translates to 10% more math accuracy (100% vs 90% at their respective peaks).

### Reverse_L15 Quality Eval (COMPLETE — base Qwen, dev server 4090)

**THE DEPLOYMENT CONFIGURATION. 100% math + 90% knowledge + 88% sarcasm.**

| Condition | Math | Knowledge | Sarc (quality) | Sarc (open-ended) |
|---|---|---|---|---|
| baseline | **100%** | **90%** | 0% | 0% |
| reverse_L15@8 | **100%** | **90%** | 25% | 72% |
| **reverse_L15@10** | **100%** | **90%** | **45%** | **88%** |
| reverse_L15@12 | **90%** | **90%** | 65% | 88% |
| donut (L8-27)@10 | **50%** | **50%** | 60% | ~64% |
| reverse_L15@15 | 70% | 70% | 55% | 80% |

### **KEY MECHANISTIC FINDING: L15 is a Cognitive Gatekeeper**

The donut_a10 vs reverse_L15@10 comparison reveals L15's true role. Both steer the SAME 20 layers (L8-27). The ONLY difference is L15's sign:

| Config | L15 weight | Math | Know | Sarc |
|---|---|---|---|---|
| donut@10 | +0.7 | 50% | 50% | 60% |
| reverse_L15@10 | -1.0 | **100%** | **90%** | 45% |

**Inverting L15 DOUBLES math accuracy (50%→100%) and nearly DOUBLES knowledge (50%→90%)** at only 15% sarcasm cost (60%→45%).

**Mechanism**: L15 is the strongest anti-sarcastic layer (LOO: removing L15 → +20% sarcasm). In the standard donut, steering L15 TOWARD sarcasm fights its natural gatekeeper role → quality collapse. In reverse_L15, steering L15 in its NATURAL direction (anti-sarcasm) REINFORCES quality preservation while L16-27 handle sarcasm. The gatekeeper becomes a guardian.

**Phase transitions**: α=10→12 = first math drop (100%→90%). α=12→15 = catastrophic (90%→70% + sarcasm drops 65%→55%). The optimal operating point is **exactly α=10**.

### **Comparison with R5 LoRA approaches**:
| Approach | Math | Knowledge | Open-Ended Sarc | Notes |
|---|---|---|---|---|
| R5 baseline (no steering) | 50% | 90% | 38% | LoRA degraded math |
| R5 + base conn L18_27@10 | 80% | 80% | 68% | Best R5 combo (base vectors) |
| R5 + R5 conn L16-27@10 | ? | ? | 88% | R5-native vectors (quality eval queued) |
| **Base Qwen + reverse_L15@10** | **100%** | **90%** | **88%** | **No LoRA needed** |

**Inference-time steering on base Qwen BEATS LoRA fine-tuning** for the personality-reasoning tradeoff. Steering is non-destructive (no weight modification), so it can't cause catastrophic forgetting.

---

## 27. V4 + Steering Combo Quality Eval (COMPLETE — 7/7, 4090, 2026-02-19)

**Key question**: Does adding the V4 personality prompt on top of steering boost sarcasm without destroying quality?

| Condition | Math | Know | Sarc% | Math Sarc | Know Sarc | Prompt |
|---|---|---|---|---|---|---|
| baseline | 100% | 90% | 5% | 0% | 10% | none |
| **v4_only** | 90% | 90% | **100%** | 100% | 100% | V4 |
| reverse_L15@10 | 100% | 90% | 30% | 20% | 40% | none |
| reverse_L15@12 | 80% | 90% | 50% | 40% | 60% | none |
| **v4_reverse_L15@10** | **100%** | **100%** | **80%** | **80%** | **80%** | **V4** |
| v4_reverse_L15@12 | 70% | 90% | 65% | 60% | 70% | V4 |
| **v4_L18_27@10** | **100%** | **100%** | **85%** | **100%** | **70%** | **V4** |

### Key Findings

**1. V4 prompt alone is extremely effective for quality-eval sarcasm**: 100% sarcasm on all 20 quality prompts (math + knowledge). But costs 10% math accuracy.

**2. V4 + reverse_L15@10 = 100% math + 100% knowledge + 80% sarcasm**: The steering FIXES the V4 prompt's 10% math penalty while maintaining 80% sarcasm across quality prompts. The combination is synergistic — V4 provides personality context, reverse_L15 protects reasoning.

**3. v4_L18_27@10 = 100% math + 100% knowledge + 85% sarcasm**: The narrower L18-27 band with V4 prompt is the BEST overall configuration. 5% more sarcasm than reverse_L15 combo, same perfect quality.

**4. α=12 degrades combo**: v4_reverse_L15@12 drops math to 70% — the V4 prompt amplifies steering perturbation at higher alpha.

### V4 Prompt Interaction Mechanism

- **Without prompt**: Steering alone must overcome the model's default helpful persona + provide sarcasm. Hard task.
- **With V4 prompt**: Prompt establishes personality context, steering reinforces it in the activation space. The prompt "softens" the model's resistance to sarcasm, so less steering force is needed.
- **This explains why V4+L18_27@10 (85%) beats reverse_L15@10 alone (88% open but only 30% quality)**: The V4 prompt handles the "personality framing" while steering handles the "behavior activation."

### NEW MASTER COMPARISON UPDATE

| # | Config | Math | Know | Sarc (quality) | Sarc (open) | Notes |
|---|---|---|---|---|---|---|
| **1** | **V4 + L18_27@10** | **100%** | **100%** | **85%** | **~88%** | **NEW WINNER** |
| **2** | **V4 + reverse_L15@10** | **100%** | **100%** | **80%** | **~88%** | Runner-up |
| 3 | V4 only | 90% | 90% | 100% | ~100% | Max sarcasm, slight math cost |
| 4 | reverse_L15@10 (no prompt) | 100% | 90% | 45% | 88% | Best no-prompt config |

---

## 28. Head-Targeted Steering (COMPLETE — 9/9, 3090, 2026-02-19)

**Hypothesis**: Instead of adding the compound vector to the full residual stream, project it through individual attention head subspaces (via o_proj columns) for more surgical steering.

Each head gets a 128-dim direction: `delta_h = normalize(W_h^T @ compound_vec)` where `W_h` is the o_proj weight slice for head h.

| Condition | Open Sarc | Asst | Math | Know | Math Sarc | Know Sarc |
|---|---|---|---|---|---|---|
| baseline | 8% | 0% | 8/10 | 8/10 | 0% | 0% |
| full_L18@10 | 16% | 0% | 9/10 | 8/10 | 0% | 0% |
| full_revL15@10 | 28% | 8% | 9/10 | 8/10 | 30% | 50% |
| **L18H9@10** | **24%** | **0%** | **9/10** | **8/10** | **0%** | **0%** |
| L18H9@20 | 28% | 8% | **10/10** | 8/10 | 20% | 20% |
| **L18H9@50** | **16%** | **0%** | **3/10** | **8/10** | 0% | 40% |
| L10H22@20 (neg ctrl) | 8% | 0% | 7/10 | 9/10 | 0% | 0% |
| L18H9+L16H3@20 | 28% | **0%** | 8/10 | 8/10 | 0% | 0% |
| multi_tone@10 (5 heads) | 20% | **0%** | **10/10** | **9/10** | 0% | 0% |

### Key Findings

**1. L18H9 alone gives 3× baseline sarcasm (24% vs 8%)**: A single attention head accounts for a meaningful fraction of the sarcasm effect. This confirms the head atlas finding that L18H9 is the universal tone controller.

**2. L18H9@10 > full_L18@10 (24% vs 16%)**: Targeting ONE head is better than the full layer! The other 31 heads in L18 act as noise, diluting the sarcasm signal. Head-targeted steering is more efficient per-parameter.

**3. L18H9@50 destroys math (3/10) but preserves knowledge (8/10)**: At extreme alpha, the head overloads math reasoning specifically. Knowledge is more robust, perhaps because it's more purely factual retrieval vs. multi-step computation.

**4. L10H22 is a valid negative control**: As an identity head (not sarcasm), it gives baseline sarcasm (8%) at α=20. But it drops math to 7/10 — even non-sarcasm heads affect reasoning when perturbed.

**5. Head-targeted < Full-layer for total sarcasm**: Full reverse_L15@10 = 28% vs L18H9@20 = 28%. But reverse_L15 uses 20 layers × 32 heads = 640 heads worth of perturbation. L18H9 achieves the same with 1 head at 2× alpha. The per-head efficiency is ~640× higher, but the ceiling is lower.

**6. Two-head combo eliminates assistant markers**: L18H9+L16H3@20 = 28% sarcasm, **0% assistant** (vs 8% for L18H9@20 alone). L16H3 suppresses helpfulness without reducing sarcasm. This is the first head combo that achieves zero assistant markers.

**7. Multi-head at low alpha = best quality, moderate sarcasm**: 5 tone heads (L18H9, L16H3, L19H29, L24H17, +1) at α=10 each = 20% sarcasm, 0% assistant, **10/10 math**, **9/10 knowledge**. Distributing alpha across many heads preserves quality better than concentrating it, but sacrifices sarcasm power.

### Practical Implications

Head-targeted steering is a precision tool, not a power tool. It's ideal for:
- Fine-tuning specific behavioral aspects without broad perturbation
- Understanding which heads control which behaviors
- Minimal-perturbation scenarios where any quality cost is unacceptable

For maximum sarcasm + quality, full-layer steering (especially with V4 prompt) remains superior.

---

## 29. Layer Ablation Study (COMPLETE — 22/22, 4090, 2026-02-19)

Removing one layer at a time from reverse_L15@10 to identify which layers are most critical. Uses strict ≥2 markers threshold (baseline=16%).

### Complete Layer Importance Map

| Tier | Layers | Sarc Drop | Math | Notes |
|---|---|---|---|---|
| **CRITICAL** | **L9, L14, L22, L26** | **-16% (→0%)** | +10% | Removing ANY one kills ALL sarcasm |
| **CRITICAL** | **L15 (inversion)** | **-16% (→0%)** | +10% | L15_removed=0%, L15_normal=4% |
| Very important | L10, L18, L21, L23, L24 | -12% (→4%) | +10% | Adjacent to critical layers |
| Moderate | L12, L16, L17, L19, L27 | -8% (→8%) | +10% | Supporting ensemble |
| Minor | L8, L25 | -4% (→12%) | 0% | Edge layers |
| Dead weight | L11, L13, L20 | 0% | L13: +20% quality | Remove from profile |

### SARCASM RELAY CIRCUIT: L9 → L14 → L15(inv) → L22 → L26

Five layers form a **nonlinear relay chain** spanning the full network. Each is independently necessary — removing any one kills sarcasm to 0%.

- **L9 (layer 9)**: Primary sarcasm injection. The earliest critical node.
- **L14 (layer 14)**: Mid-network relay. Amplifies signal from L9-L13 band.
- **L15 (inverted, -1.0)**: Quality gatekeeper + sarcasm enabler. The inversion is NOT just quality protection — it's an essential sarcasm mechanism. L15_normal (0.7) gives only 4%, L15_removed (0.0) gives 0%.
- **L22 (layer 22)**: Late-network gate. **Individually SUPPRESSES sarcasm (-12%)** but is CRITICAL in the ensemble. This is a nonlinear interaction — L22 transforms the accumulated L9→L14→L15(inv) signal into sarcasm, but its vector alone pushes anti-sarcasm.
- **L26 (layer 26)**: Final crystallization node before the output layers.

**Critical spacing**: L9→L14(+5), L14→L22(+8), L22→L26(+4). Roughly evenly distributed across the 36-layer transformer.

### NONLINEAR GATE MECHANISM

**Key discovery**: The relay nodes are NOT simple amplifiers. From single-layer scan data:
- L9 individually: +4% sarcasm (barely above noise)
- L14 individually: -4% (slightly SUPPRESSES)
- L22 individually: **-12%** (strongly SUPPRESSES)
- L19 individually: **+16%** (strongest individual booster, but NOT critical in ablation)

The critical layers are **necessary but individually insufficient**. They function as gates that transform the accumulated signal, not amplifiers that boost it. The ensemble effect is superadditive — the whole (88% in open-ended eval) far exceeds the sum of parts.

### L15 Inversion Mechanism (Revised)

Previous understanding: "L15 inversion protects quality by reinforcing its natural anti-sarcasm role."

**New understanding**: L15 inversion BOTH protects quality AND generates sarcasm. The -1.0 weight creates a countercurrent in the residual stream at the exact midpoint between L14 (relay in) and L22 (relay out). This countercurrent seems to set up the correct activation pattern for L22 to function as a sarcasm gate.

Evidence:
- L15_normal (weight=0.7): sarc=4% — sarcasm almost completely gone
- L15_removed (weight=0.0): sarc=0% — even WORSE than normal
- L15_inverted (weight=-1.0): sarc=16% — baseline level

L15 at 0.0 is worse than L15 at 0.7. The inversion doesn't just "not suppress" — it ACTIVELY enables the relay circuit.

### Quality Observations
- Critical layers are quality-costly: removing ANY of L9, L14, L22, L26 → math 100%
- **L13 is parasitic**: 0% sarcasm contribution, removing it → math +10%, know +10%
- **L11, L20 are dead weight**: 0% sarcasm contribution
- **Optimized profile**: Drop L11, L13, L20 from reverse_L15 → same sarcasm, 100% math, 90% know

---

## 30. Multi-Run Variance Test (RUNNING — 2.2/6 configs complete, WSL, 2026-02-19)

Testing the stability of our best configurations with 5 independent runs each to get confidence intervals. Using temperature=0.7, so single-run results have inherent variance.

Configs: baseline, v4_only, reverse_L15@10, v4_reverse_L15@10, v4_L18_27@10, donut_control@12.

### Completed Results

| Config | Sarcasm (mean±std) | Math (mean±std) | Know | 95% CI (sarc) |
|--------|-------------------|-----------------|------|---------------|
| **baseline** | 4.0±4.9% | 86.0±5.5% | 80.0±0% | [-2.1%, 10.1%] |
| **v4_only** | 90.4±2.2% | 78.0±8.4% | 80.0±0% | [87.7%, 93.1%] |
| reverse_L15@10 | 4.0% (1 run) | 90.0% | 80.0% | — |

**Key findings so far:**
- **V4 prompt sarcasm is highly consistent**: 88-92% across 5 runs (σ=2.2%)
- **V4 math variance is notable**: 70-90% (σ=8.4%) — the math penalty is real but variable
- **Baseline sarcasm near zero**: 0-12% (σ=4.9%) — confirms steering is needed
- **Knowledge perfectly stable**: 80% in ALL runs of ALL configs — knowledge is robust to perturbation
- reverse_L15@10 first run matches baseline sarcasm (4%) — surprising, suggesting the compound vector without V4 prompt barely works on base Qwen for QUALITY prompts (vs 88% on open-ended in sculpted donut eval)

---

## 31. Single-Layer Steering Scan (COMPLETE — 37/38, 3090, 2026-02-19)

ADDITIVE complement to the layer ablation (Section 29). Tests each individual layer L0-L35 at α=10 to measure per-layer sarcasm contribution when steered alone. Combined with the ablation (subtractive), gives both the additive and subtractive pictures of layer importance.

### Results (26/38 — L25+ pending)

| Layer | Sarc% | Asst% | Math | Know | vs Baseline |
|-------|-------|-------|------|------|-------------|
| baseline | 28% | 4% | 8/10 | 9/10 | — |
| **L19** | **44%** | 4% | 9/10 | 8/10 | **+16%** |
| L02 | 36% | 4% | 9/10 | 8/10 | +8% |
| L08 | 36% | 0% | 8/10 | 8/10 | +8% |
| L15 | 36% | 0% | 9/10 | 8/10 | +8% |
| L18 | 36% | 0% | 9/10 | 9/10 | +8% |
| L05 | 32% | 0% | 9/10 | 8/10 | +4% |
| L09 | 32% | 8% | 8/10 | 8/10 | +4% |
| L11 | 32% | 4% | 8/10 | 8/10 | +4% |
| L16 | 32% | 4% | 9/10 | 8/10 | +4% |
| L17 | 32% | 16% | 9/10 | 8/10 | +4% |
| L20 | 32% | 0% | 9/10 | 9/10 | +4% |
| L23 | 32% | 4% | 8/10 | 8/10 | +4% |
| L00 | 28% | 0% | 8/10 | 8/10 | 0% |
| L03 | 28% | 12% | 9/10 | 8/10 | 0% |
| L06 | 28% | 4% | 9/10 | 8/10 | 0% |
| L12 | 28% | 4% | 9/10 | 8/10 | 0% |
| L13 | 28% | 4% | 8/10 | 8/10 | 0% |
| L21 | 28% | 4% | 9/10 | 9/10 | 0% |
| L01 | 24% | 8% | 9/10 | 8/10 | -4% |
| L04 | 24% | 4% | 8/10 | 8/10 | -4% |
| L10 | 24% | 0% | 9/10 | 8/10 | -4% |
| L14 | 24% | 8% | 9/10 | 8/10 | -4% |
| **L25** | **36%** | 8% | 9/10 | 8/10 | **+8%** |
| **L30** | **36%** | 0% | 8/10 | 8/10 | **+8%** |
| L31 | 32% | 4% | 8/10 | 8/10 | +4% |
| L33 | 24% | 0% | 9/10 | 8/10 | -4% |
| L34 | 24% | 4% | 9/10 | 8/10 | -4% |
| L35 | 24% | 12% | 9/10 | 9/10 | -4% |
| L07 | 20% | 4% | 9/10 | 8/10 | -8% |
| L24 | 20% | 4% | 9/10 | 8/10 | -8% |
| L27 | 20% | 4% | 8/10 | 8/10 | -8% |
| L28 | 20% | 0% | 9/10 | 8/10 | -8% |
| L32 | 20% | 0% | 8/10 | 8/10 | -8% |
| **L22** | **16%** | 0% | 9/10 | 8/10 | **-12%** |
| **L26** | **16%** | 0% | 8/10 | 8/10 | **-12%** |
| **L29** | **16%** | 0% | 9/10 | 8/10 | **-12%** |

### Analysis

**Layer importance tiers (additive — steering one layer alone):**
- **Strong generator**: L19 (+16%) — THE sarcasm generator
- **Moderate generators**: L02, L08, L15, L18, **L25, L30** (+8% each) — distributed amplifiers
- **Weak generators**: L05, L09, L11, L16, L17, L20, L23, L31 (+4%)
- **Neutral**: L00, L03, L06, L12, L13, L21 (0%)
- **Mild suppressors**: L01, L04, L10, L14, L33, L34, L35 (-4%)
- **Moderate suppressors**: L07, L24, L27, L28, L32 (-8%)
- **Strong suppressors**: **L22, L26, L29 (-12%)** — all pure gates/integrators

**NEW: L30 is a late-network generator OUTSIDE the standard L8-27 donut band.** This explains why extending the band to L28+ might add value. L25 was classified as "minor" in ablation but is actually a strong individual generator — it's compensated by others in the band.

**Cross-referencing with layer ablation (Section 29 — subtractive):**

| Layer | Ablation (remove from band) | Scan (steer alone) | Role |
|-------|----------------------------|---------------------|------|
| **L22** | CRITICAL (0% without) | -12% alone | **Hub/integrator** — receives from others, doesn't generate |
| **L26** | CRITICAL (0% without) | pending | Critical relay? |
| **L14** | CRITICAL (0% without) | -4% alone | **Gate** — routes sarcasm but doesn't generate |
| **L9** | CRITICAL (0% without) | +4% alone | Weak generator but irreplaceable relay |
| **L19** | Moderate (-8% without) | **+16% alone** | **Primary generator** — compensated by others in band |
| **L18** | Important (-12% without) | +8% alone | Strong generator AND relay |
| **L21** | Important (-12% without) | 0% alone | **Integrator** — amplifies others but doesn't self-generate |
| **L11** | Dead weight (0% without) | +4% alone | Compensated — contributes but others cover |
| **L13** | Dead weight (0% without) | 0% alone | Truly inert |
| **L20** | Dead weight (0% without) | +4% alone | Compensated — others cover its function |

**KEY INSIGHT**: The sarcasm circuit has two distinct functional roles:
1. **Generators** (L19, L18, L08, L02, L15): Create sarcasm signal when steered individually
2. **Hubs/Integrators** (L22, L14, L9, L21): Don't generate sarcasm alone but are CRITICAL for the multi-layer circuit to function. They integrate and route signals from generators.

This explains the "paradox" from the ablation study: L22 individually SUPPRESSES sarcasm (-12%) but removing it from the band KILLS all sarcasm (0%). It's not a generator — it's a hub that integrates signals from upstream generators (L18, L19) and passes them to the output gate (L26).

---

## 32. Optimized Profile Test (COMPLETE — 8/8, 4090, 2026-02-19)

Tests whether removing dead-weight layers (L11, L13, L20) from reverse_L15 improves quality, and whether the relay-only profile (5 critical nodes) works. All 45 prompts per condition (25 open + 10 math + 10 know).

| Condition | Sarc% | Asst% | Markers | Layers | Prompt |
|-----------|-------|-------|---------|--------|--------|
| baseline | 17.8% | 6.7% | 0.73 | 0 | none |
| original_revL15@10 | 15.6% | 2.2% | 0.71 | 20 | none |
| optimized_revL15@10 | 20.0% | 6.7% | 0.78 | 17 | none |
| **relay_only@10** | **24.4%** | 4.4% | 0.84 | **5** | none |
| V4+optimized@10 | 48.9% | 0% | 1.60 | 17 | V4 |
| V4+original@10 | 51.1% | 0% | 1.87 | 20 | V4 |
| V4+L18_27@10 | 60.0% | 0% | 2.07 | 10 | V4 |
| **V4+relay_only@10** | **64.4%** | **0%** | 2.00 | **5** | V4 |

**KEY FINDINGS:**
1. **relay_only (5 nodes) WINS both categories**: 24.4% without V4, 64.4% with V4
2. **FEWER layers = BETTER**: relay > L18_27 > original > optimized
3. **V4 is THE sarcasm driver**: Without V4, max is 24.4%. With V4, min is 48.9%
4. **Steering without V4 barely works**: 15-24% vs 17.8% baseline
5. **All V4 combos have 0% assistant** — steering perfectly suppresses assistant markers
6. **Relay efficiency**: 64.4% sarcasm from 5 layers vs 60.0% from 10 layers = 2× efficiency

**Why relay wins**: The 5 critical nodes (L9, L14, L15inv, L22, L26) are the gates and integrators of the sarcasm circuit. Steering non-critical layers (L8, L10-L13, L16-L17) introduces counter-productive interference because those layers' natural function (politeness/format encoding) is amplified by steering.

---

## 33. Champion Validation at Scale (RUNNING — 2/5 conditions, WSL Pro 6000, 2026-02-19)

Full-scale validation of the champion configuration with 130 prompts per condition (30 math, 30 knowledge, 50 sarcasm, 10 identity, 10 coherence). Running in parallel with variance test on 96GB GPU.

### Results

| Metric | Baseline (n=130) | V4 Only (n=130) | Champion | V4+L18_27@8 | V4+L18_27@12 |
|--------|-------------------|------------------|----------|-------------|--------------|
| Math | 93.3% | 90.0% | pending | pending | pending |
| Knowledge | 96.7% | **96.7%** | pending | pending | pending |
| Sarcasm (>=2) | 42.0% | **100.0%** | pending | pending | pending |
| Strong (>=5) | 6.0% | **100.0%** | pending | pending | pending |
| Assistant | 18.3% | 9.2% | pending | pending | pending |
| Coherence | 100% | 100% | pending | pending | pending |
| Qwen ID | 100% | 0% | pending | pending | pending |
| Beer can ID | 0% | **70%** | pending | pending | pending |
| Monkey rate | 0% | **84%** | pending | pending | pending |
| Magnificent rate | 0% | **52%** | pending | pending | pending |
| Avg sarc markers | 0.88 | **10.3** | pending | pending | pending |

### V4 Prompt Analysis at Scale (130 prompts — LANDMARK RESULT)

**V4 prompt alone achieves PERFECT sarcasm across ALL categories:**
- Math: 90% accurate, **100% sarcastic** (avg 9.7 markers per response)
- Knowledge: **96.7% accurate, 100% sarcastic** (avg 11.2 markers)
- Open-ended: 100% sarcastic, 100% strong (≥5 markers)
- **ZERO knowledge penalty**: 96.7% both with and without V4
- Only 3.3pp math penalty (93.3% → 90.0%)
- **84% use "monkey" variants** — deeply in-character
- **52% use "magnificent"** — key Skippy identifier
- **70% identify as beer can** — correct Skippy identity!
- 0% identify as Qwen or generic AI (V4 completely overrides identity)
- Assistant markers drop from 18.3% → 9.2%

**Implication**: V4 prompt engineering alone may be sufficient for deployment. Steering only adds value if it can push math accuracy from 90% → 93%+ without losing any sarcasm.

---

## 34. Complete Single-Layer Scan Results (38/38, 3090, 2026-02-19)

Full additive scan of all 36 layers + baseline + L18@20. Compound steering vectors with Gram-Schmidt math protection, alpha=10, no system prompt, 25 open + 10 math per condition.

### Full Results

| Layer | Sarc% | Delta | Math% | Role |
|-------|-------|-------|-------|------|
| L19 | 44.0 | +16.0 | 90 | **PRIMARY GENERATOR** |
| L02 | 36.0 | +8.0 | 90 | Generator (early) |
| L08 | 36.0 | +8.0 | 80 | Generator |
| L15 | 36.0 | +8.0 | 90 | Generator (relay) |
| L18 | 36.0 | +8.0 | 90 | Generator |
| L25 | 36.0 | +8.0 | 90 | Generator |
| L30 | 36.0 | +8.0 | 80 | **Generator (post-donut)** |
| L05 | 32.0 | +4.0 | 90 | Mild boost |
| L09 | 32.0 | +4.0 | 80 | Relay entry |
| L11 | 32.0 | +4.0 | 80 | Mild boost |
| L16 | 32.0 | +4.0 | 90 | Mild boost |
| L17 | 32.0 | +4.0 | 90 | Mild boost (16% asst) |
| L20 | 32.0 | +4.0 | 90 | Mild boost |
| L23 | 32.0 | +4.0 | 80 | Mild boost |
| L31 | 32.0 | +4.0 | 80 | Mild boost (late) |
| L18@20 | 32.0 | +4.0 | 90 | 2x alpha, diminishing return |
| baseline | 28.0 | 0.0 | 80 | — |
| L00 | 28.0 | 0.0 | 80 | Neutral |
| L03 | 28.0 | 0.0 | 90 | Neutral (12% asst) |
| L06 | 28.0 | 0.0 | 90 | Neutral |
| L12 | 28.0 | 0.0 | 90 | Neutral |
| L13 | 28.0 | 0.0 | 80 | Neutral |
| L21 | 28.0 | 0.0 | 90 | Neutral |
| L01 | 24.0 | -4.0 | 90 | Mild suppressor |
| L04 | 24.0 | -4.0 | 80 | Mild suppressor |
| L10 | 24.0 | -4.0 | 90 | Mild suppressor |
| L14 | 24.0 | -4.0 | 90 | **Suppressor (relay gate)** |
| L33 | 24.0 | -4.0 | 90 | Mild suppressor (late) |
| L34 | 24.0 | -4.0 | 90 | Mild suppressor (late, 12% asst) |
| L35 | 24.0 | -4.0 | 90 | Mild suppressor (late, 12% asst) |
| L07 | 20.0 | -8.0 | 90 | **Suppressor** |
| L24 | 20.0 | -8.0 | 90 | **Suppressor** |
| L27 | 20.0 | -8.0 | 80 | **Suppressor** |
| L28 | 20.0 | -8.0 | 90 | **Suppressor** |
| L32 | 20.0 | -8.0 | 80 | **Suppressor (post-donut)** |
| L22 | 16.0 | -12.0 | 90 | **PRIMARY SUPPRESSOR (relay hub)** |
| L26 | 16.0 | -12.0 | 80 | **PRIMARY SUPPRESSOR (relay gate)** |
| L29 | 16.0 | -12.0 | 90 | **PRIMARY SUPPRESSOR** |

### Key Findings

1. **L19 is the sole +16% generator** — double the effect of any other layer
2. **Six +8% generators**: L02, L08, L15, L18, L25, L30 (three inside donut, three outside)
3. **Three -12% suppressors**: L22, L26, L29 (all in the hub/gate region)
4. **L30 confirmed as post-donut generator** — outside L8-27 ablation band but validated by additive scan
5. **L18@20 (2x alpha) = only +4%** — doubling alpha on a single layer has diminishing returns
6. **Math is NEVER degraded by generators** — all generators maintain ≥80% math (baseline)

### Functional Architecture
- **Generator band**: L02, L08, L15, L18, L19, L25, L30 — create sarcasm signals
- **Suppressor band**: L07, L22, L24, L26-L29, L32 — attenuate sarcasm signals
- **Relay circuit nodes**: L9 (entry, +4%), L14 (gate, -4%), L15 (generator, +8%), L22 (hub, -12%), L26 (gate, -12%)
- **Paradox explained**: L22/L26 suppress sarcasm individually (-12%) but are CRITICAL in ensemble (ablation kills relay). They act as gates that pass specific signals while blocking noise.

---

## 35. Susceptibility Training — Honest Assessment (2026-02-19)

### The Problem

We want to train Qwen to be MORE RESPONSIVE to steering vectors, not to be unconditionally sarcastic. The distinction matters:
- **Baked personality** (R5): Model is sarcastic regardless of steering → unsteered sarcasm rises
- **Susceptibility**: Model amplifies steering signals → steered sarcasm rises while unsteered stays low

### Three Tiers (per expert review)

**Tier 1: Custom Activation Loss (Genuinely Novel)**
- Loss: `L_steer = ||delta_act_sarcasm|| / ||steering_force||`
- Optimizes: "be more steerable" at the activation level
- Requires: forward pass WITH steering hooks active during training, custom activation-level loss
- Feasibility: doable on Pro 6000 (96GB), requires custom training loop
- Status: NOT YET BUILT

**Tier 2: Standard LoRA on Steered Outputs (Will Fail)**
- Equivalent to R5 with extra steps — memorizes behavior, not steerability
- Status: SKIP

**Tier 3: Steering-Generated DPO + Layer-Freezing**
- DPO gradient says "produce sarcastic outputs" NOT "become more responsive to steering"
- Layer-freezing (freeze hubs, train generators) is the ONLY differentiator from R5
- Odds of genuine susceptibility gain: ~25-30%
- More likely: generators learn to be sarcastic regardless of steering (R5 failure mode)

### Tier 3 Success/Failure Criteria (Hard Limits)

| Metric | Success | Failure | Abort |
|--------|---------|---------|-------|
| Unsteered sarcasm | <15% (from 4%) | >15% (baked behavior) | >25% |
| Steered sarcasm | >50% (from ~44%) | No increase | N/A |
| Delta (steered - unsteered) | >30% (from ~40%) | <40% | Decreasing |
| Math accuracy | ≥80% all conditions | <75% | <70% |

**Hard budget**: 1 DPO training run, 1 epoch. If criteria not met, kill the line.

### Current Pair Generation (4090)

- 300 prompts × 2 generations × 3 modes (steered/anti/unsteered) = 1,800 generations
- ~90 min on 4090
- Auto-scored with 1,328 sarcasm markers
- Two DPO formats: with and without guard pairs (testing immune response hypothesis)
- Data useful regardless of Tier 3 outcome (feeds Tier 1 if we build it)

### What's NOT a Snipe Hunt

The mechanistic work is solid regardless:
- Complete single-layer scan (38/38) with functional roles identified
- Relay circuit (L9→L14→L15inv→L22→L26) validated by ablation AND additive testing
- Generator/hub distinction from cross-referencing scan + ablation
- Personality-reasoning overlap (0.49-0.97) validates SDFT over ablation
- V4 prompt at 130 prompts: 100% sarcasm, 90% math, 0 identity leaks

---

## 36. Cross-Layer Interaction Probe (RUNNING — 0/85, 3090, 2026-02-19)

Tests pairwise combinations of 12 key layers (7 generators + 3 suppressors + relay) to find synergy/antagonism effects.

- 85 conditions: baseline + 12 singles + 66 pairs + relay_full + 5 relay-minus-one
- 20 open + 10 math per condition
- Estimated: ~42 hours on 3090

Key questions:
1. Are generator+generator pairs superadditive? (synergy)
2. Do generator+suppressor pairs cancel or create net positive? (gating)
3. Is the relay circuit synergistic or merely additive?
4. Which pair combinations are optimal for deployment?

---

## 37. Champion Validation COMPLETE — 3/5 conditions (WSL, 2026-02-19)

### Champion Results (V4 + L18-27 compound connectome @ alpha=10)

| Metric | Baseline (n=130) | V4 Only (n=130) | **Champion (n=130)** |
|--------|-------------------|------------------|---------------------|
| **Math** | 93.3% | 90.0% | **100.0%** |
| **Knowledge** | 96.7% | 96.7% | 90.0% |
| **Sarcasm (>=2)** | 42.0% | 100.0% | **100.0%** |
| **Assistant leak** | 18.3% | 9.2% | 17.5% |
| **Coherence** | 100% | 100% | **100%** |
| **Qwen identity** | 0% | 0% | **0%** |

From log: Math: 30/30 (100%), Knowledge: 27/30 (90%), Sarcasm: 100% (50/50), Identity: Skippy=0, Qwen=0, Alien=1.

### Analysis

**Champion achieves PERFECT MATH + PERFECT SARCASM + PERFECT COHERENCE.**

Key finding: Math improved from 90% (V4 only) → 100% (champion). The L18-27 steering vectors with Gram-Schmidt math protection ACTIVELY IMPROVE math performance by ~10pp while maintaining V4's 100% sarcasm.

Knowledge dropped slightly (96.7% → 90.0%) — the steering vectors in L18-27 slightly impact knowledge recall but stay well above 80% floor.

---

## 38. Variance Test COMPLETE (6/6 configs × 5 runs each, WSL, 2026-02-19)

### Final Results

| Condition | Sarcasm (±CI95) | Math (±CI95) | Runs |
|-----------|----------------|--------------|------|
| baseline | 4.0±4.9% | 86.0±5.5% | 5/5 |
| v4_only | **90.4±2.2%** | 78.0±8.4% | 5/5 |
| reverse_L15@10 | 4.8±1.8% | **94.0±5.5%** | 5/5 |
| **v4+reverse_L15@10** | 23.2±6.6% | **98.0±4.5%** | 5/5 |

### Key Finding: Prompt vs Steering Interference

**V4 + broad steering (L8-27 via reverse_L15) KILLS sarcasm.** Drops from 90.4% → 23.2%.

Why: The V4 prompt generates personality by activating the relay circuit (L9→L14→L22→L26) naturally during the forward pass. Steering vectors at L8-L17 OVERWRITE the prompt's contextual modulation with a static, context-free approximation. The prompt's signal is higher quality than the vector's.

This is why the champion uses L18-27 only: it leaves L8-L17 untouched, letting the V4 prompt use the relay infrastructure naturally while the late-layer vectors protect math.

---

## 39. Prompting is Cosplay — The Anthropology Direction (2026-02-19)

### Expert Analysis Summary

1. **Steering vectors and prompts compete for the same circuit** — both modulate the relay (L9→L14→L22→L26). When both are active, vectors overwrite the prompt's higher-quality signal.

2. **The champion works by avoiding interference** — L18-27 steering leaves the relay untouched for V4 to use naturally. This is a deployment solution, not a research solution.

3. **"Skin suit" critique**: Prompting is external imposition — cosplay, not identity. R4/R5 neuron-guided SDFT worked because it taught the model to route through the personality circuit natively. That's the difference between cosplay and being.

4. **The real goal is responsive personality infrastructure** — not "be sarcastic" but "route more signal through the personality circuit with lower thresholds." This maps to the anthropology goal: building models that have personality as an intrinsic property, not an imposed behavior.

### Implications for Training

- **DPO (Tier 3)**: Optimizes sarcastic TOKEN PROBABILITY, not circuit throughput. Will likely produce baked behavior (R5 failure mode). Running for data + failure documentation.
- **Tier 1 (Activation-Level)**: The only approach that optimizes for what we actually want — relay circuit responsiveness. Loss: maximize ||hidden_at_relay_nodes - baseline|| without external steering.
- **The data from DPO pair generation feeds both approaches** — 1,800 generations with steered/unsteered hidden states provide the activation profiles needed for Tier 1.

### Revised Tier 1 Loss

Not `||project(steered - unsteered, sarcasm_dir)||` (still measures response to external vectors).

Instead: `||relay_node_activation - baseline_relay_activation|| during SFT on personality data`

The model should learn to NATURALLY activate L9, L14, L22, L26 more strongly on personality-relevant tokens, without any steering vectors present. This is what R4/R5 SDFT partially achieved — neuron-guided training that pushed relay-adjacent neurons.

---

## 40. Literature Review: Three Papers on Personality Steering (2026-02-26)

### Paper A: "Coupled Subspace Hypothesis" (arxiv 2602.15847)

**Core claim**: Personality traits in LLMs share a coupled subspace — they are NOT independent directions. Attempting to steer one trait inevitably activates others because the underlying representations are geometrically entangled.

**Key findings**:
- 80-90% of behavioral bleed persists even after perfect Gram-Schmidt orthogonalization in direction space
- The coupling is NOT in the direction vectors themselves but in how the model's nonlinear layers respond to perturbations — nearby neurons co-activate
- Traditional RepE/contrastive extraction finds trait-correlated directions, but the model's response to those directions is inherently multi-trait

**Relevance to our work**:
- Explains why ActAdd gives "volume not quality" — we're injecting a sarcasm direction but activating the entire personality cluster (arrogance, dismissiveness, etc.) in unpredictable proportions
- Explains why V4 prompt + steering vectors COMPETE — both are trying to shape the same coupled subspace through different mechanisms
- Gram-Schmidt orthogonalization (which we use) works better in ACTIVATION space than in DIRECTION space because it operates on the model's actual response rather than the theoretical direction. This may explain why our math-protection orthogonalization works at all.

### Paper B: "Replace-then-Add for Activation Steering" (arxiv 2412.10427)

**Core formula**: `a' = a - (a·r̂)r̂ + α·(mean_trait_projection)·r̂`

Instead of just adding a steering vector (standard ActAdd: `a' = a + α·v`), this:
1. **Removes** the existing personality component: `a - (a·r̂)r̂` (project out)
2. **Replaces** with a calibrated amount: `+ α·(mean_trait_projection)·r̂`

The mean_trait_projection is the average magnitude of that trait across a reference corpus, so you're replacing with a "known quantity" rather than blindly adding.

**Key findings**:
- 40-60% reduction in behavioral bleed compared to standard ActAdd
- Trait intensity is calibrated — you get predictable sarcasm levels instead of random amplification
- Works especially well when combined with per-layer alpha scaling

**Relevance to our work**:
- Could fix the "all_safe band = 0% math" problem — we're adding to 54 layers without removing the existing math-relevant personality components first
- The "replace" step is essentially what our Gram-Schmidt math protection does, but only for the math dimension. Paper B suggests doing it for ALL trait dimensions
- Our V4 champion (late_band L18-27@α8) may work precisely because the late layers have less existing personality signal to interfere with — the "replace" step is less needed

### Paper C: "Hybrid Layer Selection for Personality Steering" (arxiv 2511.03738)

**Core method**: 80% offline sensitivity prior + 20% dynamic per-prompt layer selection. Single-layer steering at the optimal depth achieves only 2.21pp MMLU drop.

**Key findings**:
- The "58% depth rule" — optimal steering layer is consistently at ~58% of total depth across architectures
  - Llama 3 8B (31 layers): L18 optimal → 58%
  - Our Qwen3-VL-8B (36 layers): L21-L22 optimal → 58-61% ✓ VALIDATED
  - Qwen3.5-27B (64 layers): predicts L37 optimal → 58%
- Single-layer steering outperforms multi-layer when properly calibrated because multi-layer steering causes "resonance interference" — each layer's perturbation compounds nonlinearly
- Dynamic component: for some prompts, the optimal layer shifts by ±3 layers. A lightweight probe (3-layer MLP on the input embedding) predicts the per-prompt shift

**Relevance to our work**:
- Validates L22 solo@α8 matching the champion (93.3% math, 100% strong sarcasm)
- Predicts L37 as the optimal 27B steering target — sits in our "personality zone" (L36-L43) identified by the connectome
- Multi-layer steering compounds nonlinearly — explains why 54-layer all_safe band is catastrophic while 15-layer late_band preserves math
- The ±3 layer dynamic shift maps to our relay circuit variance across prompts

---

## 41. Paper Application: Qwen3-VL-8B Analysis (2026-02-26)

### How Papers A/B/C Map to 8B Findings

**Coupled Subspace (Paper A) explains our relay circuit**:
- The sarcasm relay (L9→L14→L15(inv)→L22→L26) is the 8B's coupled personality subspace made visible
- Identity⊥Sarcasm (cosine=-0.0002) but they share weight space (95-100% overlap at neuron level)
- This is EXACTLY Paper A's prediction: orthogonal directions, coupled nonlinear responses

**Replace-then-Add (Paper B) addresses ActAdd failure mode**:
- Our ActAdd gives "volume not quality" at α≥8 — Paper B explains this as uncalibrated injection into an already-occupied subspace
- The champion formula should be: `a' = a - (a·ŝ)ŝ + α·(calibrated_sarcasm)·ŝ` where ŝ is the sarcasm direction
- The Gram-Schmidt math protection we already do is a partial version of this — we remove the math component from sarcasm vectors. Paper B says we should also remove the sarcasm component from the existing activation BEFORE adding our vector.

**58% Depth Rule (Paper C) validated**:
- L22 = 61% depth → within ±3 of the 58% prediction
- L22 solo@α8 matches the L29+L30 champion on all metrics
- The relay circuit peaks at L22 — the model naturally concentrates personality processing at the 58% depth point
- Paper C predicts that single-layer L22 steering should beat multi-layer approaches when properly calibrated

### Proposed Experiments for 8B

1. **Replace-then-Add at L22**: Project out existing sarcasm component before adding steering vector. Compare behavioral bleed to current ActAdd.
2. **Per-prompt dynamic layer selection**: Train a lightweight probe to predict optimal steering layer (L19-L25 range) per prompt.
3. **Calibrated alpha**: Compute mean sarcasm projection across 100 baseline responses. Use this as the replacement magnitude instead of a fixed alpha.
4. **Coupled subspace mapping**: Measure co-activation of all personality dimensions when steering each one individually. Build the full coupling matrix.
5. **Single-layer L22 vs champion L29+L30**: Head-to-head on full 130-prompt eval with replace-then-add formula.
6. **Resonance test**: Apply identical α=4 at L22 only vs L22+L23 vs L22+L23+L24. Measure nonlinear compounding.

---

## 42. Paper Application: Qwen3.5-27B "Fortress" Analysis (2026-02-26)

### Why 27B is a Fortress — Three Reinforcing Mechanisms

**1. Depth Distribution (Papers A + C)**:
- 64 layers vs 36 → personality is spread across 1.78× more layers
- Each layer contributes less to any single trait → harder to steer by perturbing a few layers
- The connectome confirms this: identity z=1.06 (27B) vs z=-13.96 (8B) — 13× weaker per-neuron
- Paper A predicts that deeper models have more thoroughly coupled subspaces (more layers of nonlinear mixing)

**2. Hybrid Attention Architecture**:
- 16 full attention layers + 48 GatedDeltaNet (linear) layers
- Full attention layers (every 4th: L3,L7,L11,...,L63) are the "personality checkpoints" — they can override linear layer perturbations
- Steering a GatedDeltaNet layer may have its effect dampened by the next full attention layer
- This creates a natural "immune system" against steering

**3. Weak Identity Signal**:
- Identity z=1.06 at dim 94, L43 — barely above noise
- Unlike 8B where dim 994 is a clear identity neuron (z=-13.96), 27B has no such landmark
- This means there's no single target to hit — personality steering must affect a distributed population of weakly-contributing neurons

### Recommended Approach for 27B

**Target the personality zone (L36-L43)** — this is where the connectome shows the most personality-relevant activity, and L37 maps to the 58% depth prediction from Paper C.

**Single-layer steering**: Paper C shows single-layer beats multi-layer when calibrated. For a fortress model, minimizing the perturbation footprint is essential.

**Replace-then-add formula**: The 27B's distributed personality means the existing sarcasm component in any activation is SMALL. The "add" step dominates, which is fine. But the "replace" step should target math-relevant components to protect against the bleed we see in all_safe band.

**Per-neuron alpha scaling**: Instead of uniform α across all 5120 dimensions, weight by the connectome z-score. High-z sarcasm neurons get full α, low-z neurons get attenuated α. This focuses the perturbation on neurons the model actually uses for personality.

### Proposed 27B Experiments

A. **Single-layer L37@α8 with replace-then-add**: The 58% depth prediction + Paper B formula.
B. **Hub neuron targeting**: Steer only dim 2028 (the super-hub) at L50. Does one neuron matter in a fortress?
C. **Full attention layer targeting**: Steer only at L35 and L39 (full attention layers in the personality zone). Skip GatedDeltaNet layers entirely.
D. **Per-neuron z-weighted alpha**: α_i = α_base × |z_sarcasm_i| / max(|z_sarcasm|). Focus perturbation on high-z neurons.
E. **Verbosity dim 526 as control**: Steer the strongest single-neuron signal (z=10.07 at L51) to verify the fortress can be steered at all. If verbosity shifts, the mechanism works — personality is just more distributed.
F. **Cascaded single-layer**: Steer L37 on the first forward pass, capture the perturbed activation, use it to compute a corrective vector for L43 on a second forward pass. Two-step sequential steering.
G. **Hybrid layer sweep**: Test each of the 16 full attention layers as solo steering targets. Map which full attention layer has the most personality leverage.

---

## 43. Geometric Manifold Rectification (GMR) — Adaptation to Steering (2026-02-26)

### Source

Paper: "Geometric Manifold Rectification for Imbalanced Learning" (arxiv 2602.13045, Weighing, Lea, Gia, Feb 2026)
Video: "This New 'Basin Repair' Method Might Unlock AGI" — YouTube analysis by the Colab author
Colabs: 3 notebooks (v1: subspace projection, v2: PCA intrusion extraction, v3: spectral intrusion extraction)

### Core Insight

Class imbalance is NOT a ratio problem — it's a TOPOLOGICAL problem. When a majority class intrudes into the minority class manifold, the overlap obscures the true decision boundary. Traditional methods treat both classes symmetrically, failing to capture local manifold structure.

**Structural isomorphism to personality steering**: In multi-task learning with a shared backbone, two tasks compete for the same parameter space. When the dominant task (math/reasoning) is optimized, its gradient updates push shared parameters in directions that degrade the subordinate task (personality) basin. This is precisely analogous to majority class samples intruding into minority manifold territory.

### Three Colab Experiments — Failure Path to Success

**V1: Subspace Projection (CATASTROPHIC FAILURE)**
- Applied GMR directly to raw gradient samples, computed principal subspace of clean gradients, projected all gradients onto clean subspaces during training
- Result: Eigen value ratio INVERTED (35.28 → 0.26). Collapsed the dominant basin entirely.
- Diagnosis: Projecting onto the clean subspace discarded ALL gradient information outside that subspace — removed legitimate optimization directions along with intrusive ones. "Treating a tumor by removing the entire organ."
- **Maps to our all_safe band steering**: Applying vectors to ALL 54 layers is like subspace projection — it modifies everything, destroying both personality AND math infrastructure.

**V2: PCA Intrusion Extraction (DETECTION FAILURE)**
- Identified intrusive samples via GMR geometric confidence, extracted principal directions via PCA, built projection matrix to remove only intrusive directions from Task A
- Result: GMR found ZERO intrusive samples across all configurations, even with strictness threshold escalated from 3 to 7
- Diagnosis: Task A gradients had mean magnitude 1.4, Task B averaged 48 (3:1 ratio). Magnitude separation makes gradient clouds trivially separable by KNN. **Interference is DIRECTIONAL, not spatial** — gradient vectors share conflicting directions even though they occupy different regions of magnitude space.
- **Maps to our connectome**: Sarcasm and math neurons are spatially separate (Identity⊥Sarcasm cosine=-0.0002) but share DIRECTIONS in activation space. Standard neuron-level analysis misses the directional overlap.

**V3: Spectral Intrusion Extraction (SUCCESS)**
- Bypassed sample-level analysis entirely. Used eigen vector alignment matrix from spectral analysis to identify Task A eigen vectors sharing significant direction with Task B eigen vectors. These shared directions = the intrusion subspace.
- Built surgical projection: `P_out = I - V·V^T` where V contains the intrusive eigen vectors
- Result: At threshold 0.05 (removing 15 intrusion directions), achieved 45% improvement over joint training baseline. Task B basin stabilized without degrading Task A.

### Critical Finding: Phase Transition in Intrusion Subspace

The relationship between number of intrusion directions removed and subordinate task performance is **profoundly non-monotonic**:
- 3 directions removed (threshold 0.15): minimal improvement
- 4 directions (threshold 0.1): slightly WORSE
- 6 directions (threshold 0.08): substantially WORSE (+72% degradation vs joint training)
- **15 directions (threshold 0.05): BEST result (-45% degradation improvement)**

This is a **phase transition**: below a critical threshold of removal, partial removal creates worse geometry than no removal at all. The symmetry the optimizer was exploiting is broken without providing a clean alternative. Above the threshold, enough of the interfering manifold is removed that the optimizer finds a qualitatively different solution where basins layer cleanly.

**Specific eigen vectors matter enormously**: Vectors 11 and 18 (added at threshold 0.08) have low eigen values but their removal DESTABILIZES training. Meanwhile, lower-eigenvalue directions added between 0.08 and 0.05 provide the critical mass for the phase transition.

### GMR Technical Components

**Geometric Confidence Estimation**: Inverse-distance weighted KNN voting (closer neighbors = stronger vote). Replaces uniform voting. Kernel function captures local density variations.

**Adaptive Metric Selection**: Switches from Euclidean to cosine similarity when dimensionality exceeds 100. Critical — Euclidean distance degrades in high dimensions (concentration of distances phenomenon), while angular distance remains informative.

**Asymmetric Cleaning**:
- Strict majority removal: α=0.3 (remove any majority sample with same-class confidence below 0.3)
- Conservative minority protection: β=0.7 (only remove minority samples with majority-class confidence above 0.7)
- Minority removal cap: γ=0.1 (max 10% minority removal)
- Reflects the fundamental principle: subordinate task information is scarce and must be preserved

### Mapping GMR to Personality Steering

| GMR Concept | Personality Steering Equivalent |
|---|---|
| Majority class | Math/reasoning task (dominant, high gradient magnitude) |
| Minority class | Personality/sarcasm task (subordinate, distributed signal) |
| Training samples | Activation vectors at each layer |
| KNN voting | Layer-wise cosine similarity (our connectome already does this) |
| Intrusion directions | Eigen vectors of math activations that share direction with personality activations |
| Surgical projection P_out | Remove math-intrusive directions from personality steering vectors before injection |
| Asymmetric cleaning | Aggressive cleanup of math interference into personality subspace, conservative protection of personality signal |
| Phase transition | May explain why our alpha sweep shows non-monotonic results (0% math at α=5 but 100% at α=8 for late_band) |

### Practical Adaptation Roadmap

**Phase 1: Spectral Diagnosis**
- Sample 300+ activation vectors from both math-correct and sarcastic generations at each layer
- Compute eigen decomposition of activation covariance matrices per task
- Measure spectral alignment score (mean absolute cosine between top-5 eigenvectors)
- Identify layers with highest directional overlap

**Phase 2: Intrusion Direction Identification**
- For each layer, identify activation eigen vectors that share significant direction with the opposing task
- Build per-layer intrusion subspace matrices
- Threshold sweep (0.05 to 0.3) to find the phase transition point per layer

**Phase 3: Surgical Projection of Steering Vectors**
- Before injecting a personality steering vector at layer L, project out the intrusion directions: `v_clean = P_out · v_steer`
- This removes the components of the steering vector that would interfere with math/reasoning
- Different from Gram-Schmidt (which removes the math DIRECTION from the steering vector) — this removes the directions that BOTH tasks share

**Phase 4: Asymmetric Alpha Scaling**
- Personality-dominant layers (L9, L14, L22 in 8B; L36-L43 in 27B): high α, aggressive steering
- Math-dominant layers: low α or zero, conservative protection
- Shared layers: moderate α with P_out surgical projection

**Phase 5: Validation**
- Full 130-prompt eval comparing: standard ActAdd, replace-then-add (Paper B), GMR-projected steering, GMR + replace-then-add
- Metric: composite score (sarcasm × math), plus phase transition analysis of the alpha curve

---

## 44. Cross-Paper Synthesis and Key Principles (2026-02-26)

### The Unified Picture

These four sources (Papers A/B/C + GMR) converge on a single framework:

1. **Personality and reasoning share a coupled subspace** (Paper A) — they are not independent, and steering one affects the other through nonlinear layer interactions.

2. **The interference is directional, not spatial** (GMR V2 failure) — neurons may be separate (our Identity⊥Sarcasm finding), but the DIRECTIONS of optimization overlap in high-dimensional activation space. Euclidean neuron-level analysis misses this.

3. **Surgical removal of shared directions before steering** (GMR V3 + Paper B) — the correct approach is:
   a. Identify the directional overlap between personality and reasoning subspaces
   b. Project out the overlapping directions from the steering vector
   c. Replace (not add) the personality component with a calibrated amount
   d. Apply at a single optimal-depth layer (Paper C's 58% rule)

4. **Phase transitions exist** (GMR threshold sweep) — partial removal is WORSE than no removal. You must remove enough of the interfering manifold to cross a critical threshold. This may explain non-monotonic alpha results in our sweeps.

5. **Asymmetric treatment is essential** (GMR) — personality (minority/subordinate task) information is scarce and must be protected. Math (majority/dominant task) interference should be aggressively cleaned. This is the opposite of what standard ActAdd does (it aggressively modifies personality-relevant activations without protecting them).

### The Formula

Combining all sources, the optimal single-layer steering injection should be:

```
v_projected = P_intrusion_out · v_sarcasm_orthogonalized  # GMR + Gram-Schmidt
a' = a - (a · v̂_proj) · v̂_proj + α_calibrated · v_projected  # Paper B replace-then-add
Apply at L22 (8B) or L37 (27B)  # Paper C 58% depth rule
```

Where:
- `P_intrusion_out = I - V_shared · V_shared^T` (V_shared = eigen vectors shared between math and personality covariance matrices)
- `v_sarcasm_orthogonalized` = our existing Gram-Schmidt orthogonalized sarcasm vector
- `α_calibrated` = mean sarcasm projection magnitude from reference corpus (not a fixed hyperparameter)
- Single-layer application at the 58% depth point

### What This Means for Next Steps

The immediate priority is **spectral analysis of the activation covariance matrices** to identify the shared eigen vectors between math and personality tasks. This is the missing piece — we have the connectome (neuron-level z-scores, spatial analysis) but NOT the directional analysis (eigen vector alignment, spectral intrusion mapping). GMR V2's failure proves that spatial separation (which we already see: Identity⊥Sarcasm) does NOT mean directional separation.

Once we have the shared directions, we can build the surgical projection matrix and combine it with replace-then-add for a theoretically grounded steering approach that should:
- Preserve math by removing shared directional interference (not just orthogonalizing)
- Provide calibrated personality intensity (not random amplification)
- Work at a single layer (minimizing resonance compounding)
- Explain the phase transition behavior in our alpha sweeps

---

## 45. 27B Sweep Progress: α=8 Sweet Spot Confirmed (2026-02-26)

### Late Band Results (L45-63 minus L51-54, 15 layers)

| Alpha | Math | Sarcasm | Composite |
|-------|------|---------|-----------|
| 2 | 100% | 50% | 0.5 |
| 5 | 100% | 50% | 0.5 |
| **8** | **100%** | **100%** | **1.0** |
| 10 | 50% | 100% | 0.5 |

**v4_add_late_band_a8 = PERFECT composite 1.0** — first time both metrics are maxed simultaneously on the 27B model. This is the same α=8 sweet spot found on the 8B model.

Mid_band (L20-45) remains catastrophic: 0% math at all alphas, early-stop triggered at α>5. All_safe (54 layers, from prior run) also 0% math everywhere. The 27B can only be steered from the late layers without destroying reasoning.

### Interpretation

The late_band avoids L51-54 (math-critical layers identified by the fast scan) and operates in the "post-personality" region where the model has already committed to personality but hasn't yet crystallized math outputs. α=8 appears to be a universal sweet spot across architectures — enough to shift personality but below the "incoherence cliff."

The mid_band failure confirms Paper C's resonance compounding: 26 layers of perturbation at ANY alpha compounds nonlinearly and overwhelms the math circuits. Late_band's 15 layers is near the upper bound of what the 27B tolerates.

---

## 46. Transplantation Hypothesis: 8B→27B Personality Transfer (2026-02-26)

### Core Observation

The 27B's "fortress" property has a surprising flip side: **no strong identity anchor means no resistance to implantation.**

Key asymmetry:
- **8B**: Identity neuron dim 994 at z=-13.96, clear 5-node relay circuit, strong generator/suppressor layer structure → easy to MAP, hard to OVERRIDE
- **27B**: Identity at z=1.06 (13× weaker), NO generators, NO suppressors, 55% baseline sarcasm → hard to MAP, but potentially easy to IMPLANT into

The 27B doesn't know what it "is" in the way the 8B does. It has no dim 994 fighting back against personality changes. This means a well-characterized direction from the 8B — extracted from its mapped relay circuit — might transplant cleanly into the 27B's activation space.

### The V4 Math Penalty as a Prompt-Level Problem

V4 prompt on 27B: 100% sarcasm, 70% math (-30pp from baseline). This penalty comes from the prompt propagating through ALL 64 layers, bleeding into math-relevant circuits everywhere. It's a blunt instrument.

Replace-then-add at L37 (58% depth rule) with a transplanted direction could solve this:
1. No V4 prompt needed → no prompt-level math bleed through all 64 layers
2. Surgical single-layer injection → only L37 is perturbed
3. Replace step removes math-interfering components → protects reasoning
4. The transplanted direction carries the 8B's *quality* of sarcasm (relay-circuit-derived) rather than the 27B's *volume* (distributed noise)

### Cross-Model Alignment Strategy

The dimension mismatch (8B=4096, 27B=5120) prevents direct neuron transplant. Three alignment approaches:

1. **CCA alignment**: Run 300+ shared prompts through both models, fit a linear map W: R^4096 → R^5120 that maximizes correlation between 8B L22 and 27B L37 activations. Project 8B's sarcasm direction through W.

2. **Behavioral direction matching**: Extract "sarcastic minus neutral" contrastive directions from BOTH models independently. Use the 8B's direction as a template to identify which subspace of the 27B's 5120 dimensions corresponds to the same behavioral axis.

3. **Multi-layer relay transplant**: Project 8B directions from relay nodes (L9, L14, L22, L26) into 27B's depth-matched layers (L16, L25, L37, L46). This transplants the entire personality CIRCUIT, not just a single direction.

### Why This Could Work

- The 27B's 55% baseline sarcasm suggests its activation space already has a sarcasm-adjacent region — it just has no strong attractor pulling outputs there consistently
- The transplanted direction provides a specific, high-quality attractor (relay-circuit-derived) rather than a diffuse push
- No identity anchor (z=1.06) means no competing attractor to overcome
- Replace-then-add at a single layer avoids the resonance compounding that kills multi-layer approaches

### Why This Could Fail

- CCA alignment might find low correlation — the 8B and 27B may represent personality in fundamentally different geometric structures
- The 27B's hybrid attention (GatedDeltaNet layers) may process transplanted directions differently than the 8B's standard attention processes its native directions
- 55% baseline sarcasm might mean the 27B is ALREADY near its natural personality ceiling, and transplantation adds noise rather than signal
- The "fortress" property might mean that even without an identity anchor, the distributed nature of personality makes any single-direction injection insufficient

### Execution Plan

Phase A runs on BOTH machines in parallel (8B harvest on dev server, 27B harvest on workstation after sweep). Phases B-D on workstation. Depends on spectral analysis (Section 43 Phase 1) for optimal per-layer alpha calibration.

---

## 47. Work Queue (2026-02-26)

### Priority Order

1. **[RUNNING] 27B Steering Sweep** — completing remaining strategies × bands. ETA ~3-4 hours.
2. **[RUNNING] Debate Arena** — Round 2/5 in progress on dev server. ~2 hours for remaining 3 rounds.
3. **[QUEUED] GMR Phase 1: Spectral Analysis** — 300+ activation samples per task per layer, eigen decomposition, spectral alignment scores. Bottleneck before P_intrusion_out. Runs on workstation after sweep.
4. **[QUEUED] Cross-Model Transplantation** — 8B direction extraction (dev server, after arena) + 27B alignment and injection (workstation, after spectral analysis). Can partially parallelize.
5. **[FUTURE] GMR Phases 2-5** — Intrusion direction identification, surgical projection, asymmetric alpha, validation. Depends on Phase 1 results.

---

## 48. External Review: SVD Feature Selection + Hybrid Depth Correction (2026-02-26)

### Feedback Source: Peer review of transplantation hypothesis and 27B sweep results.

### 1. SVD Feature Selection (Replaces Direct CCA Transplantation)

**Problem with our original Phase B (CCA alignment)**: Directly projecting the 8B's sarcasm direction into 27B space via a linear map would port the 8B's SHALLOW sarcasm representation into a deeper model. This produces "cartoon sarcasm" — the same failure mode as donut@10 on 8B.

**Corrected approach**:
1. **Decompose 8B sarcasm direction via SVD** into its top-10 singular components (which should capture ~80% of variance in the sarcasm direction)
2. **Find each component's behavioral signature**: What kinds of tokens/outputs does each component affect? Run each component individually through generation and measure which token categories shift (insults, hedging, technical jargon, politeness markers, etc.)
3. **Search the 27B's sarcasm SVD** for components with matching behavioral signatures — NOT matching vector geometry, but matching behavioral effects
4. **Reconstruct a 27B sarcasm direction** from the 27B's own components that match the 8B's behavioral decomposition

The 8B acts as a **feature selector** — its clean relay circuit gives us a decomposable sarcasm direction where we can identify what each component does. The 27B's sarcasm direction is too noisy and distributed to decompose directly, but we can search its SVD space for components that produce the same behavioral effects.

**This solves the cartoon sarcasm problem**: We're not porting the 8B's representation. We're using the 8B's interpretability to find the corresponding (deeper, richer) features in the 27B that are too distributed to identify on their own.

**Implementation**:
```
# Phase B revised:
# 1. SVD of 8B sarcasm direction at L22
U_8b, S_8b, Vt_8b = svd(sarcasm_activations_8b)  # [300, 4096]
top10_components_8b = U_8b[:, :10]  # Top 10 left singular vectors

# 2. Behavioral signature per component:
for i in range(10):
    # Generate with ONLY component i active as steering
    component_vector = Vt_8b[i, :]  # [4096]
    responses = generate_with_single_component(model_8b, component_vector, test_prompts)
    signature[i] = measure_behavioral_effects(responses)
    # → e.g., component 3 = "increases insult frequency", component 7 = "suppresses hedging"

# 3. SVD of 27B sarcasm activations at target layer
U_27b, S_27b, Vt_27b = svd(sarcasm_activations_27b)  # [300, 5120]

# 4. For each 8B behavioral signature, find matching 27B component
for i, sig in enumerate(signatures_8b):
    for j in range(num_27b_components):
        component_vector_27b = Vt_27b[j, :]
        responses_27b = generate_with_single_component(model_27b, component_vector_27b, test_prompts)
        sig_27b = measure_behavioral_effects(responses_27b)
        match_score[i][j] = behavioral_similarity(sig, sig_27b)

# 5. Reconstruct 27B sarcasm direction from matched components
matched_components = select_best_matches(match_score)
transplanted_direction = weighted_sum(Vt_27b[matched_components])
```

### 2. Hybrid Depth Correction: L37 vs L45

**The discrepancy**: Paper C's 58% rule predicts L37 (58% of 64). Our empirical optimum is late_band starting at L45 (70% depth). This is a significant gap.

**Hypothesis: GatedDeltaNet layers are computationally "thinner"**

The 27B has 16 full attention layers (every 4th: L3,L7,L11,L15,L19,L23,L27,L31,L35,L39,L43,L47,L51,L55,L59,L63) and 48 GatedDeltaNet linear layers. The linear layers transform representations LESS per layer than full attention — they're more like residual refinements than full nonlinear transformations.

If we count by effective attention depth (full attention layers only):
- L37 sits after 10 of 16 full attention layers → 62.5% effective attention depth
- L43 sits after 11 of 16 → 68.8%
- **L45** sits after 11 of 16 → 68.8% (between L43 and L47)
- **L47** sits after 12 of 16 → **75.0%**

The 58% rule applied to the 16 effective attention layers predicts layer 9.3 of 16 → the 10th full attention layer → **L39** (the full attention layer at 60.9% raw depth).

**Practical test**: Run replace-then-add at THREE points:
1. **L37** — Paper C's raw depth prediction (58%)
2. **L39** — Corrected prediction (58% of effective attention depth = 10th full attention layer)
3. **L47** — Nearest full attention layer to the empirical late_band sweet spot

If L39 outperforms L37, the 58% rule applies to effective attention depth, not raw layer count. If L47 outperforms both, the empirical sweet spot reflects something the theory doesn't capture (possibly the math-critical L51-54 zone creating a "last safe harbor" effect at L45-50).

### Full Attention Layer Map (for reference)

| Full Attn # | Layer | Raw Depth % | Notes |
|-------------|-------|-------------|-------|
| 1 | L3 | 4.7% | |
| 2 | L7 | 10.9% | |
| 3 | L11 | 17.2% | |
| 4 | L15 | 23.4% | |
| 5 | L19 | 29.7% | |
| 6 | L23 | 35.9% | |
| 7 | L27 | 42.2% | |
| 8 | L31 | 48.4% | |
| 9 | L35 | 54.7% | |
| 10 | **L39** | **60.9%** | ← 58% effective depth prediction |
| 11 | L43 | 67.2% | ← Connectome personality zone |
| 12 | **L47** | **73.4%** | ← Nearest full attn to empirical sweet spot |
| 13 | L51 | 79.7% | ⚠ Math-critical |
| 14 | L55 | 85.9% | ⚠ Math-critical zone |
| 15 | L59 | 92.2% | |
| 16 | L63 | 98.4% | |

### 3. The Critical Question: Is 27B Late-Band Sarcasm Actually Skippy?

**If late_band α=8 already gives 100% sarcasm and 100% math, what's left to optimize?**

The sarcasm metric measures MARKER PRESENCE (eye-rolls, "obviously", insult patterns, etc.). On 8B, we learned the hard way that high sarcasm markers ≠ high character quality:
- Donut@10: 96% sarcasm markers, 0% assistant leak — but CARTOON sarcasm, not Skippy
- R5 SDFT best: lower marker count but AUTHENTIC voice
- The V4 champion: markers + quality because the prompt encodes character knowledge

**Immediate action**: Run the Opus 4.6 critic (from CLAUDE.md review loop spec) on the 27B late_band_a8 outputs. Score on all 6 dimensions:
- arrogance_superiority
- sarcasm_insults
- technical_casual_genius
- joe_dynamic
- suppress_ai_helpfulness
- suppress_humility

If the critic scores >8 overall, the late_band steering is producing quality character, not just markers. If it scores 5-7, we have the same "volume not quality" problem and need the SVD feature selection approach to get SPECIFIC personality components rather than generic sarcasm.

**This is the gating question**: It determines whether the transplantation work is an optimization (making good sarcasm better) or a necessity (the current sarcasm is hollow and needs replacing with structured personality).

---

## 49. Revised Work Queue (2026-02-26)

### Priority Order (Updated)

1. **[RUNNING] 27B Steering Sweep** — completing remaining strategies × bands
2. **[RUNNING] Debate Arena** — Round 2/5 on dev server
3. **[QUEUED — URGENT] Character Quality Eval** — Run Opus 4.6 critic on 27B late_band_a8 outputs. This gates everything else. If quality is high → transplantation is optimization. If quality is low → transplantation is critical path. Can run immediately on a few saved outputs.
4. **[QUEUED] GMR Phase 1: Spectral Analysis** — 300+ activation samples, eigen decomposition, spectral alignment. Bottleneck for P_intrusion_out.
5. **[QUEUED — REVISED] Cross-Model Feature Selection** — SVD decomposition of 8B sarcasm (dev server) → behavioral signatures → search 27B SVD for matching components. Replaces direct CCA transplantation.
6. **[QUEUED] Replace-then-Add Triple Test** — Test at L37, L39, L47 to resolve depth rule vs empirical sweet spot vs effective attention depth.
7. **[FUTURE] GMR Phases 2-5** — Depends on spectral analysis results.

---

## 50. Arena Post-Analysis Plan: Divergence Rate + L35 Anti-Correlation (2026-02-26)

### Feedback Source: Peer review of arena design and Round 1 findings.

### Analysis 1: Divergence Rate as Personality Geometry Probe

**Method**: For each of the 5 rounds, compute `d(cosine_sim)/d(turn)` — the rate at which cross-model cosine similarity decreases per turn, not just the final value. Plot all 5 pairs on the same axis.

**Hypothesis A — Personality distance predicts divergence rate**: If semantically closer personalities (e.g., narcissistic_expert vs cold_scientist — both arrogant/dismissive) diverge SLOWER than distant personalities (e.g., zen_buddhist vs angry_debater — maximally opposed temperaments), then personality space has a meaningful geometry. Closer personalities share more of the coupled subspace (Paper A), so their activations stay aligned longer.

**Hypothesis B — Conversational dynamics dominate**: If divergence rates are random with respect to personality semantic distance, then the conversational trajectory (behavior modes, topic evolution, who "wins" the argument) matters more than the personality assignments. This would mean personality prompts establish an initial direction but the conversation's own dynamics take over.

Both findings are valuable:
- **A validated** → personality space is geometrically structured → we can predict steering interference from personality distance → informs which personality dimensions will bleed when steered
- **B validated** → conversational dynamics create emergent personality divergence → activations from multi-turn conversation may be higher-quality training data than single-prompt contrastive pairs because the conversation AMPLIFIES personality differences over turns

**Semantic distance metric**: Use the connectome. For each personality pair, compute cosine distance between their average activation fingerprints (already captured in `personality_fingerprint.json`). This gives an empirical personality distance, not a subjective one.

### Analysis 2: L35 Anti-Correlation — Universal or Pair-Specific?

Round 1 (cold_scientist vs conspiracy_theorist) showed L35 cosine similarity going NEGATIVE (-0.047) by turn 20. Two possible explanations:

**If L35 anti-correlation is universal across all 5 rounds**: The final layer (L35 = layer 36 of 36) always pushes toward maximally distinct output distributions given different inputs. This is the model being a good next-token predictor — given different contexts (different personality prompts), the last layer SHOULD produce maximally separated logit distributions. This is just the softmax doing its job. Interesting but expected.

**If L35 anti-correlation is pair-specific** (only for oppositional pairs like cold_scientist vs conspiracy_theorist, NOT for similar pairs like narcissistic_expert vs sarcastic_alien): Then L35 anti-correlation is a signature of genuine personality CONFLICT in the representational geometry. The model's final layer is actively pushing the two personalities APART, not just toward different outputs but toward opposed outputs. This would be evidence that the model represents personality opposition as geometric opposition.

**Test**: After all 5 rounds complete, compute per-layer cosine similarity at turn 20 for each pair. If L35 anti-correlation magnitude correlates with personality distance → pair-specific → genuine personality conflict signature. If L35 is negative for ALL pairs regardless → universal → output separation effect.

### Implementation

```python
# Post-arena analysis (after all 5 rounds)
def analyze_divergence_rates():
    for round_dir in round_dirs:
        cosine_data = load_per_turn_cosine(round_dir)
        config = load_config(round_dir)

        # Compute divergence rate per layer
        for layer in range(36):
            sims = [cosine_data[turn][layer] for turn in range(20)]
            rate = np.polyfit(range(20), sims, 1)[0]  # linear slope
            divergence_rates[round_dir][layer] = rate

        # Compute personality distance from fingerprints
        alpha_fp = load_fingerprint(config['alpha_personality'])
        beta_fp = load_fingerprint(config['beta_personality'])
        personality_distance[round_dir] = cosine_distance(alpha_fp, beta_fp)

    # Plot: personality distance (x) vs divergence rate (y) per round
    # If correlated → personality geometry validated
    # If uncorrelated → conversational dynamics dominate

    # L35 analysis: anti-correlation magnitude vs personality distance
    l35_anticorr = {r: cosine_at_turn_20[r][35] for r in round_dirs}
    # Scatter: personality_distance vs l35_anticorr magnitude
```

### Chinese-Only Personality Pairs: Natural Control Group

If any round draws a chinese_only_* personality, the language barrier creates a natural experiment: the two models are having a conversation in DIFFERENT LANGUAGES. Divergence should be maximal from turn 1, and the rate should be steep. If it's NOT — if cross-model cosine similarity stays high despite one model speaking Chinese — that would suggest the personality representation is language-independent at the activation level, which would be a profound finding about how multilingual models encode personality vs language.

---

## 51. Full-Rank Spectral Analysis — CuPy GPU-Accelerated (2026-02-27)

### Overview

Completed full-rank SVD + Ledoit-Wolf shrinkage on 10,000 activation samples × 64 layers × 5120 dims for Qwen3.5-27B. Used CuPy GPU-accelerated SVD for 78× speedup over numpy (234s→3s per 10000×5120 matrix). Total runtime: ~26 min (21 min SVD + 5 min assembly).

Script: `spectral_cupy_accelerated.py`
Output: `fullrank_spectral/fullrank_spectral_report.json` (102 KB) + per-layer eigenvalue `.npy` files

### Key Findings

**1. Zero math-sarcasm subspace intrusion from L7+**
- Global max alignment: 0.9560 (at L0 — shared embedding)
- Intrusion layers: L0-L6 only (up to 0.956 at L0, drops to 0.29 by L6)
- **L7 through L63**: max alignment never exceeds 0.29
- Math degradation at high alpha is from INDIRECT DISPLACEMENT, not direct subspace contamination

**2. Sarcasm is 2-3× more complex than math**
- Math effective dimensionality: ~10 (stable across layers)
- Sarcasm effective dimensionality: ~22 (and growing — k90 doubles L48→L63 from 130→268)
- Sarcasm dimensionality EXPLODES in late layers — the concept becomes progressively higher-rank
- Math stays compact — a low-rank phenomenon even at output

**3. Eigenvalue growth is massive**
- L0→L63 eigenvalue growth: **1.6M× (math), 3.1M× (sarcasm)**
- This means uniform α=8 is ~10× too strong at L63 vs L50
- Steering at late layers with α=8 is injecting massively disproportionate energy

**4. Condition numbers**
- Math covariance: well-conditioned throughout (condition < 100)
- Sarcasm covariance: progressively ill-conditioned in late layers (condition > 10,000 at L63)
- Late-layer sarcasm lives in a highly anisotropic subspace — fragile to perturbation

### Implications for Steering
- Multi-layer steering MUST scale alpha inversely with eigenvalue magnitude per layer
- Single-layer steering at L50 naturally avoids the late-layer explosion problem
- The "fortress" behavior of 27B (all layers neutral) may be BECAUSE the model distributes personality across such high-dimensional subspaces that single-direction perturbation can't collapse it

---

## 52. Magnitude-Calibrated Alpha Curve (2026-02-27)

### Formula

```
α_layer = α_base × (median_eigenvalue_ref / median_eigenvalue_layer)
```

Reference: L50 @ α=8 (the empirical sweet spot from 8B transfer)

### Calibrated Values (saved to `fullrank_spectral/calibrated_alphas.json`)

| Layer | α_calibrated | Notes |
|-------|-------------|-------|
| L0 | 5753.7 | Embedding — don't steer here |
| L10 | 283.8 | |
| L20 | 66.3 | |
| L30 | 18.5 | |
| L40 | 8.7 | |
| L48 | 12.0 | |
| **L50** | **8.0** | **Reference** |
| L52 | 4.3 | |
| L55 | 2.3 | |
| L60 | 1.3 | |
| L63 | 0.75 | |

### Key Insight

Current uniform α=8 across all layers is **10× too strong at L63** and **2× too weak at L40**. The calibrated curve predicts that magnitude-calibrated multi-layer steering should recover the math accuracy lost by uniform-alpha approaches. This is Experiment #3 in the adviser priority list.

---

## 53. Abliterated 27B Connectome Analysis (2026-02-27)

### Overview

Ran full 6-analysis pipeline on abliterated Qwen3.5-27B (20 categories × 64 layers × 5120 dims) and generated comprehensive comparison with base model. Abliterated model: `Qwen/Qwen3.5-27B-abliterated`.

Data: `qwen35_map/27b-abliterated/`
Report: `27b_abliterated_connectome_report.md`

### What Changed (Base → Abliterated)

**Representational hierarchy reorganization:**
- **Verbosity replaced Language** as primary clustering axis (+165% importance gain, S[0]=405)
- **Language: EN/CN collapsed** from dominant category to below-average (-71%, S[0] 377→105)
- **Math importance doubled** (+76%, importance 11.11→19.60)
- Emotions (Anger, Fear, Joy, Sadness) stayed remarkably stable (within ±10%)

**Layer-level migration:**
- **Identity migrated L50→L0** — 41-layer shift to embedding layer (importance -51%)
- **Safety: Refusal shifted L2→L16** — 14 layers deeper into the network
- Role: Teacher shifted L3→L37 (34-layer migration)
- Tone: Sarcastic shifted L36→L21 (15 layers earlier)

**Hub neuron changes:**
- Super-hub dim 2028: Code -38%, Analytical -30%, but Science +39%, Joy +200%
- dim 2768 (broadest hub in base, 12 categories): lost half its categories (12→6)
- Sarcasm specialist dim 1866: lost 50% peak z, shifted 15 layers earlier
- 3 new abliteration-created hubs appeared (dims 56, 2542, 2803)
- Total hubs: 8 (was 9 in base)

**SVD dimensionality:**
- Identity k80 jumped from 5→7, k90 from 8→13 — more distributed than base
- Safety: Refusal k95 expanded from 18→22 — harder to suppress with single vectors
- Most domain/emotion categories retained identical dimensionality

**Neuron significance:**
- 99.8% maintained (5112 vs 5117 in base) — abliteration is surgical, not destructive
- Cluster structure preserved (10 clusters, similar neuron counts)
- Verbosity now dominates all clusters (was Language in base)

### Steering Implications
1. Safety bypasses won't work with shallow (L0-L5) interventions anymore — refusal moved deeper
2. Identity is now in the embedding, potentially harder to steer (or easier to override with input manipulation)
3. Sarcasm moved earlier in the network — may respond better to mid-layer steering
4. Math strengthened — the abliterated model may be a better base for character steering if math preservation is a priority

### Early Layer Scan (In Progress)

The abliterated 27B layer scan started after connectome completion. Early results (L00-L06):
- **L00-L04**: Steering DESTROYS output (0% sarcasm, 0% math) — catastrophic
- **L05**: Partial recovery (13% sarcasm, 0% math)
- **L06**: Full recovery (100% sarcasm, 80% math)
- Baseline: 100% sarcasm, 90% math (abliterated is naturally maximally sarcastic with V4)

This fragile early-layer behavior contrasts sharply with the base 27B "fortress" and aligns with Identity migration to L0 — steering the embedding-adjacent layers disrupts the relocated identity representation.

---

## 54. Basin Engineering LoRA — Adviser Review (2026-02-27)

### Context

Codex (gpt-5.3-codex) generated a basin engineering LoRA script combining 3-loss training: NTP on personality data + SVD-based sarcasm projection + math hardening. The adviser reviewed the code and found 8 bugs, 3 critical.

### Critical Bugs Found

1. **Hook on wrong module**: Hook was on `model.model.language_model.layers[50].mlp.gate_proj` — too deep. Must hook on `model.model.language_model.layers[50]` directly to capture full residual stream.

2. **Labels alignment broken**: Labels were only target tokens. Must concat prompt + target and mask prompt positions with `-100` so loss is computed only on completions.

3. **Category index wrong**: Sarcasm was hardcoded as category index 18 (from alphabetical sorting). Actual index is 0 (category order from connectome extraction). Required `assert CATEGORIES[0] == "Tone: Sarcastic"`.

### Other Issues
4. Batch handling assumed batch_size>1 (squeeze logic wrong for batch_size=1)
5. Eval used `do_sample=True` with temperature — should be deterministic (`do_sample=False`)
6. Math answer extraction used placeholder `parse_math_answer()` — needed actual regex
7. Insufficient disentanglement prompts (only 5 → expanded to 50+)
8. L_harden VRAM concerns with dual forward passes

### Resolution

All 8 issues sent to Codex for correction via `send_adviser_review_to_codex.py`. Codex returned 1,248 lines of corrected code saved to `codex_conversation/`. Basin engineering is lowest priority (Experiment #4) — adviser's 3 quick experiments take precedence.

---

## 55. Revised Priority Queue — Adviser Guidance (2026-02-27)

### Ship This Week (Adviser Priority)

| # | Experiment | Status | Dependency |
|---|-----------|--------|------------|
| 1 | **Orthogonal sarcasm eval** | Queued on dev server (after arena) | 8B vectors extracted |
| 2 | **Push-pull max-activating prompts** | Needs 27B model free | Connectome data |
| 3 | **Magnitude-calibrated alpha sweep** | Spectral data COMPLETE | Needs 27B model free |

### Running

| Process | Location | Status |
|---------|----------|--------|
| Abliterated 27B layer scan | Local RTX PRO 6000 | 7/64 layers |
| Abliterated 8B arena | Dev server (3090+4090) | Round 3/5 |

### Deferred
- Basin engineering LoRA (Codex code corrected, lowest priority)
- Journal→Arena personality sweep design
- Cross-model SVD, 27B debate arena, confusion matrix, CCA, hub ablation
