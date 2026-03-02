# 8B Personality Sweep Report — Blackwell Partition

**Date**: 2026-03-01
**Model**: Qwen/Qwen3-VL-8B-Thinking (bf16, 17GB)
**GPU**: RTX PRO 6000 96GB (batch_size=55)
**Runtime**: ~3.5 hours (chars 84-298, skip-existing from dev server partial)
**Script**: `scripts/experiments/personality/personality_sweep_collector.py`

---

## 1. Sweep Overview

| Metric | Value |
|---|---|
| Characters processed | 192 (of 298 total grid) |
| Prompts per character | 60 (6 categories × 10 prompts) |
| Total responses | 11,520 |
| Total tokens generated | 5,884,864 (~5.9M) |
| Avg tokens per response | 511 (near max_new_tokens=512) |
| Target layers | L09, L15, L22, L29 |
| Activation shards | 3 per layer × 4 layers = 12 shards (368 MB total) |

The character grid combines 53 journal profiles, 2 population seeds, and 243 Big Five combinatorial grid entries (3^5 = 243 level combinations across low/medium/high for each trait). Each character received a naturalistic system prompt encoding their demographics, Big Five traits, communication style, and coping strategies.

## 2. Methodological Caveat: Think-Only Activations

**100% of responses (11,520/11,520) consist entirely of `<think>` traces.** The model never reaches `</think>` within the 512-token budget. Sample:

> *"Okay, the user is asking how I'm feeling today. As Gabriela Martinez, a 74-year-old firefighter with a doctorate, I need to respond authentically based on my personality traits. First, considering my..."*

This means our activations capture the model's **metacognitive phase** — reasoning *about* how a personality should respond — rather than the **behavioral phase** where it actually *performs* the personality. These are likely different activation regimes: Thinking-model training specifically shapes separate pathways for `<think>` (planning/deliberation) vs post-`</think>` (polished output).

**Implications:**
- The separability findings below reflect the model's ability to *represent personality as a concept for planning*, not necessarily its *behavioral expression of personality*
- This is a different measurement than prior work (which captures response-mode activations), making direct comparison harder
- The 27B replication uses `--no-thinking` (1024 tokens, response-only), so it will capture behavioral activations — the contrast between the two will itself be informative
- A re-run of the 8B at 1024 tokens is needed to capture both phases and measure whether separability differs between think-mode and response-mode

## 3. Entropy Analysis: Personality Affects Predictability

Mean generation entropy (bits) by Big Five level:

| Dimension | Low | Medium | High | Delta (H-L) |
|---|---|---|---|---|
| **Openness** | 1.270 | 1.247 | 1.225 | **-0.046** |
| **Conscientiousness** | 1.261 | 1.252 | 1.230 | **-0.032** |
| Extraversion | 1.248 | 1.247 | 1.247 | -0.002 |
| **Agreeableness** | 1.233 | 1.243 | 1.268 | **+0.036** |
| Neuroticism | 1.243 | 1.248 | 1.250 | +0.007 |

These deltas are small (0.03-0.05 bits) but statistically robust at n~4,000 per level. The directions are psychologically coherent:
- **High openness/conscientiousness → lower entropy**: Systematic, methodical thinking reduces token-level uncertainty.
- **High agreeableness → higher entropy**: Hedging, qualifying, and perspective-taking create more varied token distributions.
- **Extraversion has zero effect on entropy**: Introverts and extraverts differ in *what* they say, not *how predictably* they say it.

Overall range: 1.107–1.421 bits (0.314 bit spread across 192 B5 combos).

## 4. Cross-Layer Separability (Pre-Whitening)

Cohen's d effect size (High vs Low) at 4 sampled layers. **These values are pre-whitening and likely inflated** — the baseline FineFineWeb collection (50K samples, running) will provide the covariance correction needed to assess true personality-specific separability.

| Dimension | L09 | L15 | L22 | L29 |
|---|---|---|---|---|
| **Extraversion** | 1.31 | 1.39 | 1.83 | **1.94** |
| **Agreeableness** | 1.14 | 1.10 | 1.63 | **1.78** |
| **Neuroticism** | 0.95 | 0.89 | 1.38 | **1.43** |
| **Openness** | 1.00 | 0.89 | 1.30 | **1.54** |
| Conscientiousness | 0.64 | 0.68 | 0.91 | **1.08** |

**What this shows**: Personality signal increases across the four sampled layers, with a jump between L15 and L22 and the strongest separation at L29 for all five dimensions.

**What this does NOT show**: With only 4 layer samples, we cannot claim "progressive construction" — the transition could be a step-function occurring anywhere in the L16-L21 window, and we would not detect it. The earlier connectome work identified L22 specifically for sarcasm, and the convergence is suggestive, but this sweep does not independently pinpoint L22 as the hub — it confirms something substantial happens in the L15-L22 gap. Adding 2-3 intermediate layer samples (e.g., L16, L19, L21) would resolve this ambiguity.

**Pre-whitening caveat**: If post-whitening d-values hold above ~1.0 for extraversion and agreeableness, the personality geometry story is solid. If they drop below ~0.5, the separability is largely driven by general-processing variance correlated with topic, not personality per se. This is the most important pending test.

## 5. The Big Five Directions Are NOT Orthogonal

Cosine similarity between Big Five direction vectors at L22 (high_mean − low_mean):

| | O | C | E | A | N |
|---|---|---|---|---|---|
| **O** | 1.000 | **-0.430** | 0.185 | 0.160 | **0.357** |
| **C** | -0.430 | 1.000 | -0.231 | 0.112 | -0.160 |
| **E** | 0.185 | -0.231 | 1.000 | -0.184 | **-0.255** |
| **A** | 0.160 | 0.112 | -0.184 | 1.000 | **0.293** |
| **N** | 0.357 | -0.160 | -0.255 | 0.293 | 1.000 |

These are computed from well-estimated means (~4,000 samples per level) and represent genuine geometry:

- **O ↔ C = -0.430**: The model's "open" direction overlaps substantially with "not conscientious." This mirrors the known O-C tension in personality psychology.
- **O ↔ N = +0.357**: Abstract exploratory thinking (openness) partially overlaps with ruminative exploratory thinking (neuroticism).
- **A ↔ N = +0.293**: Both involve heightened social/emotional attention.
- **E ↔ N = -0.255**: Confident outgoing representations oppose anxious ruminative ones.

**Implication for steering**: Additive steering along one Big Five direction will push others. Steering toward "high openness" drags ~43% toward "low conscientiousness." Orthogonalization or SAE decomposition is needed for clean single-dimension manipulation.

## 6. Neuron-Level Analysis: Personality is Massively Distributed

After Bonferroni correction (alpha=0.01, threshold p < 2.44×10⁻⁶), the number of neurons showing statistically significant High-vs-Low differences is:

| Dimension | Bonferroni survivors | % of 4,096 | Median |d| | Max |d| |
|---|---|---|---|---|
| Extraversion | 2,838 | 69.3% | 0.252 | 1.064 |
| Agreeableness | 2,795 | 68.2% | 0.245 | 1.008 |
| Openness | 2,751 | 67.2% | 0.237 | 0.809 |
| Neuroticism | 2,540 | 62.0% | 0.233 | 0.902 |
| Conscientiousness | 2,274 | 55.5% | 0.193 | 0.645 |

**55-69% of all neurons carry statistically significant personality information** — even after the most conservative multiple-comparisons correction. This is the opposite of the "few special neurons" picture. Personality is a **massively distributed field** across the majority of the 4,096-dimensional space, with each neuron contributing a small effect (median |d| ≈ 0.2) and the aggregate producing the strong separability seen in the full-vector Cohen's d.

This is consistent with our earlier connectome and GPT-OSS findings: personality is not concentrated in sparse neurons but distributed across the network. The top-ranked neurons (e.g., dim 2781 at d=1.06 for extraversion) are the most extreme points on a continuous distribution, not uniquely important. With 2,838 neurons all significantly distinguishing personality types for extraversion, any individual neuron is unremarkable.

## 7. Factorial Structure Accounts for Variance

PCA on the 192 B5-combo mean activations at L22:

| PCs | Cumulative Variance |
|---|---|
| Top 5 | 56.9% |
| Top 10 | 80.0% |
| Top 20 | 100.0% |

**Reframing**: With 192 combo-mean vectors in 4,096-dimensional space, the maximum rank is 191. The fact that 20 PCs capture 100% of the combo-mean variance does NOT mean personality lives in 20 of 4,096 dimensions. It means the model's response to Big Five conditioning is well-described by a ~20-dimensional subspace, which is the expected dimensionality from a 5-factor system with pairwise interactions: 5 main effects + 10 pairwise + a few higher-order terms ≈ 20. This is actually a clean result — it means the factorial grid structure accounts for the observed variance with no surprising nonlinear structure beyond low-order interactions.

The inter-combo cosine similarities remain very high (mean=0.994, min=0.971 at L22), meaning personality signals are small perturbations (<3% of total activation variance) on top of a massive general-processing signal. **Baseline subtraction is critical** for probe quality.

## 8. Activation Statistics

| Layer | Mean Norm | Std | Min | Max | Inter-Combo Cos (min) |
|---|---|---|---|---|---|
| L09 | 35.6 | 0.59 | 33.4 | 38.0 | 0.989 |
| L15 | 51.6 | 0.70 | 46.0 | 53.7 | 0.986 |
| L22 | 108.1 | 3.16 | 86.4 | 121.7 | 0.971 |
| L29 | 382.3 | 10.1 | 311.8 | 425.5 | 0.965 |

Activation norms grow ~10x from L09→L29. Norm variance and inter-combo separation both increase in later layers, but the overall high cosine values confirm the small-perturbation regime.

## 9. Methodological Notes

### Design Strengths
- **Full factorial grid**: The 3^5 Big Five combinatorial design is the principal methodological advantage. It cleanly separates main effects from interactions and prevents confounding between dimensions — something no published personality-probing paper has done.
- **Statistical power**: ~4,000 responses per Big Five level provides robust estimates. The entropy effects (0.03-0.05 bits) and cosine matrix are well-powered.
- **Batched GPU-resident collection**: batch_size=55 on RTX PRO 6000 (79% util, 62GB) achieved ~73s per character.

### Limitations
- **Think-only activations**: Metacognitive planning mode, not behavioral expression. Different measurement from prior literature.
- **4-layer sampling**: Insufficient to pinpoint where personality crystallizes. The L15→L22 jump spans 7 layers. Need intermediate samples for structural claims.
- **Pre-whitening d-values**: Likely inflated by general-processing variance. The narrative must not build on specific d-values until baseline comparison is complete.
- **192 of 298 characters**: Dev server is completing the remaining journal/population profiles.

## 10. What Survives Scrutiny

| Finding | Status | Depends On |
|---|---|---|
| Entropy effects are real and psychologically coherent | **Solid** | — |
| Big Five directions are non-orthogonal (O↔C = -0.43) | **Solid** | — |
| Personality is massively distributed (~60-69% of neurons) | **Solid** | — |
| Factorial structure accounts for combo-mean variance | **Solid** (reframed) | — |
| Personality signal increases L09→L29 | **Defensible** | Denser layer sampling for structural claims |
| Cohen's d values (e.g., extraversion d=1.94) | **Pending** | Post-whitening values from baseline |
| L22 as personality hub | **Suggestive** | Intermediate layer samples (L16-L21) |
| Think-mode activations capture personality | **Different** | 8B re-run at 1024 tokens for comparison |

## 11. Next Steps

1. **FineFineWeb baseline** (running, ~6h): 50K personality-neutral texts → mean subtraction, covariance whitening, false-positive probe check. **The single most important pending test.**
2. **Post-whitening d-values**: Recompute Cohen's d after baseline subtraction. This determines whether the personality geometry story holds.
3. **27B replication** (queued, 1024 tokens, `--no-thinking`): Response-mode activations → apples-to-apples cross-architecture comparison.
4. **8B re-run at 1024 tokens**: Capture both think-mode and response-mode activations. Compare separability between phases.
5. **Intermediate layer sampling**: Add L16, L19, L21 to the 8B sweep to test the L22-hub hypothesis with proper resolution.
6. **Ridge regression probes**: Fit on whitened activations, test on held-out characters.
7. **SAE decomposition**: Map 5 direction vectors onto L22 SAE features → test if SAE resolves O-C entanglement.

---

*Output: `sweep_output/blackwell/` (215 response files, 12 activation shards, 368 MB)*
*Analysis data: `sweep_output/blackwell/summary_stats.json`*
*Corrections applied per advisor review 2026-03-01.*
