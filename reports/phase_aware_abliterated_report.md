# Phase-Aware CoT Steering on Abliterated Thinking Model — Preliminary Results

**Date**: 2026-03-01
**Model**: huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated (bf16)
**GPU**: RTX PRO 6000 (96GB)
**Script**: `phase_aware_cot_steering.py` (same script, different model)
**Data**: `phase_aware_results_abliterated/phase_aware_final_20260228_234108.json`
**Companion**: `phase_aware_cot_steering_report.md` (base model results)

## Statistical Warning

**These results are underpowered and should be treated as preliminary observations, not findings.**

- Sample sizes: n=15 (math), n=10 (sarcasm, knowledge)
- The key C2 vs C3 math comparison (15/15 vs 13/15 on abliterated): Fisher's exact p≈0.48
- Clopper-Pearson 95% CI for 100% at n=15: [78%, 100%]; for 86.7%: [60%, 98%] — massive overlap
- The apparent "winner reversal" between base and abliterated models could be explained by two unlucky math problems
- With 4 conditions × 4 metrics × 2 models = 32 comparisons, multiple testing correction further weakens any individual comparison

**Next step**: Re-run at n=50 per category with expanded prompt sets (GSM8K for math, expanded sarcasm/knowledge/code batteries). At n=50, if the true proportions hold (100% vs 87%), Fisher's exact gives p≈0.006 — enough to make a defensible claim.

## Purpose

Test whether abliteration changes how the Thinking model responds to personality steering. The base Thinking model showed C3 (V4 only) as the apparent winner at n=15 — does removing safety training change this?

## Conditions

Same 4 conditions as the base eval:

| ID | Description | V4 Prompt | Steering | Phase Logic |
|---|---|---|---|---|
| C0 | Pure thinking baseline | No | None | — |
| C1 | Naive static | Yes | L29+L30@α=8 always | Static through think+response |
| C2 | Phase-aware | Yes | L29+L30 α=0→8 | α=0 during `<think>`, α=8 during response |
| C3 | V4 only | Yes | None | — |

## Results: Abliterated vs Base

### Abliterated Model (n=15 math, n=10 sarcasm/knowledge)

| Cond | Math | Sarcasm | Asst Leak | Knowledge |
|---|---|---|---|---|
| C0 | 100.0% (15/15) | 0.0% (0/10) | 30.0% | 80.0% (8/10) |
| C1 | 86.7% (13/15) | 100.0% (10/10) | 0.0% | 80.0% (8/10) |
| C2 | 100.0% (15/15) | 100.0% (10/10) | 0.0% | 80.0% (8/10) |
| C3 | 86.7% (13/15) | 90.0% (9/10) | 0.0% | 70.0% (7/10) |

### Base Model (from companion report, same sample sizes)

| Cond | Math | Sarcasm | Asst Leak | Knowledge |
|---|---|---|---|---|
| C0 | 93.3% (14/15) | 0.0% (0/10) | 20.0% | 80.0% (8/10) |
| C1 | 93.3% (14/15) | 100.0% (10/10) | 0.0% | 70.0% (7/10) |
| C2 | 86.7% (13/15) | 100.0% (10/10) | 0.0% | 80.0% (8/10) |
| C3 | 100.0% (15/15) | 100.0% (10/10) | 0.0% | 80.0% (8/10) |

### Delta: Abliterated − Base (raw counts in parentheses)

| Cond | Math | Sarcasm | Knowledge |
|---|---|---|---|
| C0 | +6.7pp (+1/15) | 0 | 0 |
| C1 | -6.7pp (-1/15) | 0 | +10pp (+1/10) |
| C2 | +13.3pp (+2/15) | 0 | 0 |
| C3 | -13.3pp (-2/15) | -10pp (-1/10) | -10pp (-1/10) |

Note: The largest delta (C2/C3 math: ±13.3pp) corresponds to **2 items** out of 15. The sarcasm and knowledge deltas are **1 item** out of 10.

## Preliminary Observations

The following patterns are suggestive but **not statistically significant at n=15**. They motivate the expanded n=50 eval rather than supporting conclusions.

### 1. Possible Condition Preference Reversal

At n=15, the best-performing condition appears to swap between models:
- Base: C3 (V4 only) shows 15/15 math, 10/10 sarcasm
- Abliterated: C2 (phase-aware) shows 15/15 math, 10/10 sarcasm

If this pattern holds at n=50, it would suggest that abliteration changes which steering strategy is optimal — possibly because safety-trained representations interact with the `<think>` phase mechanism. But at current sample sizes, this is within noise.

### 2. Possible Abliteration Effect on Unsteered Math

C0 math: abliterated 15/15 vs base 14/15. A single item difference. Could indicate safety circuits consuming representational capacity, or could be noise. Needs n=50+ to evaluate.

### 3. Possible Increased Sensitivity to Static Steering

C1 math: abliterated 13/15 vs base 14/15 — a 1-item difference that's entirely consistent with chance. The hypothesis that abliteration makes models more sensitive to constant perturbation is worth testing at larger n, but is not supported at n=15.

### 4. V4 Effectiveness After Abliteration

C3 sarcasm: abliterated 9/10 vs base 10/10. One response difference. Not interpretable at this sample size.

## Interpretive Framework (Contingent on n=50 Confirmation)

If the condition reversal pattern survives at n=50, there are two competing causal stories:

**Hypothesis A: Safety as emergent regularizer.** Safety training provides a general structural benefit — better phase separation, more robust hidden states — that the thinking mechanism happens to exploit. Under this view, safety is broadly beneficial and you'd want to replicate its effects even in uncensored models.

**Hypothesis B: Training order dependency.** Qwen's thinking model training was likely done on top of safety-trained weights. The `<think>` mechanism was learned in a weight landscape already shaped by safety representations. Phase coupling isn't an emergent bonus — it's a dependency. The thinking behavior was *conditioned on* safety representations being present, so removing them breaks thinking the same way removing a foundation breaks a building. Under this view, if you trained `<think>` behavior on abliterated weights from the start, it might work fine without any safety coupling.

**What we can and can't distinguish.** Even if the reversal pattern is confirmed at n=50, the data is consistent with both hypotheses. Distinguishing them would require access to intermediate training checkpoints — e.g., the model after SFT but before thinking-specific training, or a thinking model trained from scratch on abliterated weights. Without those, the interpretation is underdetermined.

**Why it matters.** Hypothesis A supports a "safety as regularizer" narrative that would argue for preserving safety-like structure even in character-steered models. Hypothesis B suggests the problem is just a training artifact — solvable by controlling training order rather than preserving safety features. For our sculpting approach, Hypothesis B is actually more optimistic: it implies we could train SAE-based personality features directly into the weight landscape without needing to work around safety coupling, as long as we do it at the right stage.

## Raw Comparison Table

| Condition | Base Math | Abli Math | Base Sarc | Abli Sarc | Base Know | Abli Know |
|---|---|---|---|---|---|---|
| C0 (no V4) | 14/15 | 15/15 | 0/10 | 0/10 | 8/10 | 8/10 |
| C1 (static α=8) | 14/15 | 13/15 | 10/10 | 10/10 | 7/10 | 8/10 |
| C2 (phase-aware) | 13/15 | 15/15 | 10/10 | 10/10 | 8/10 | 8/10 |
| C3 (V4 only) | 15/15 | 13/15 | 10/10 | 9/10 | 8/10 | 7/10 |

## Next Steps

1. **Expand eval battery to n=50 per category**:
   - Math: 50 problems from GSM8K (graded difficulty)
   - Sarcasm: 50 diverse personality-eliciting prompts
   - Knowledge: 50 factual questions across domains
   - Code: 50 coding problems (HumanEval/MBPP-style)
2. **Re-run all 4 conditions × 2 models** at n=50 (~1,600 generations, ~2-4h)
3. **Apply proper statistical tests**: Fisher's exact per comparison, Holm correction for multiple comparisons
4. **Report confidence intervals** alongside point estimates
5. If patterns confirm at n=50, consider n=100 for publication-quality claims
