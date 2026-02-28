# EXP 1: Orthogonal Sarcasm Steering Evaluation Report

**Date**: 2026-02-28
**Model**: Qwen3-VL-8B-Instruct (INT8 via bitsandbytes)
**Hardware**: Dev server RTX 4090
**Runtime**: 27.5 minutes (3 conditions x 15 prompts = 45 generations)
**Script**: `run_orthogonal_eval_devserver.py`

---

## Hypothesis

The Sarcastic+Polite entanglement (cos=0.24 at L29, cos=0.33 at L30) means our connectome-derived sarcasm steering vector contains ~25-33% Polite signal that fights the V4 prompt's sarcasm. Projecting out the Polite component via Gram-Schmidt should give cleaner sarcasm without math degradation.

**Prediction**: strong_sarcasm_rate increases from ~75% to ~100% without math cost.

---

## Experimental Design

### Gram-Schmidt Orthogonalization

For each champion layer (L29, L30):
```
sarc_pure = sarc - (sarc . polite / polite . polite) * polite
sarc_pure_unit = sarc_pure / ||sarc_pure||
```

| Layer | cos(sarc,polite) Before | cos(sarc,polite) After | Energy Removed |
|-------|------------------------|----------------------|----------------|
| L29 | +0.2435 | ~0 (2e-08) | 3.0% |
| L30 | +0.3338 | ~0 (1e-08) | 5.7% |

### Three Conditions (A/B/baseline)

| Condition | Steering | System Prompt |
|-----------|----------|---------------|
| `original_sarcastic` | Raw sarcasm z-score direction (unit-normalized), L29+L30 @ alpha=8 | V4 |
| `purified_sarcastic` | Polite-purified sarcasm direction (unit-normalized), L29+L30 @ alpha=8 | V4 |
| `v4_only_no_steering` | None | V4 |

### Bug Fix During Experiment

First run used raw connectome z-scores (norm ~97-110) without normalization. At alpha=8, this gave effective steering magnitude ~780x, producing catastrophic degenerate output ("Oh oh oh oh oh divine oh oh oh" repetition in all 15 responses). Root cause: `validate_champion.py` unit-normalizes vectors (`vec /= norm`) but `orthogonal_sarcasm_steering.py` did not. Fixed by adding `sarc_unit = sarc / sarc.norm()` before applying alpha.

**Lesson**: Raw connectome z-scores CANNOT be used as steering vectors without normalization. The z-score magnitudes (~100) are statistical measures, not activation-space units.

---

## Results

### Summary Table

| Metric | Original | Purified | V4-Only |
|--------|---------|---------|---------|
| Strong sarcasm (3+ markers) | **15/15 (100%)** | **15/15 (100%)** | **15/15 (100%)** |
| Avg sarcasm markers/response | 9.0 | 9.2 | 9.1 |
| Marker count variance | 7.43 | 17.03 | — |
| Assistant leak rate | 13.3% (2/15) | 13.3% (2/15) | 20.0% (3/15) |
| "Monkey" mentions (total) | 25 | 24 | 22 |
| Polite hedges | 1 | 1 | 0 |
| Avg response length (chars) | 1686 | 1742 | 1716 |
| Avg generation time | 36.4s | 37.4s | 36.1s |

### Statistical Test

Paired t-test on sarcasm marker counts (purified - original):
- Mean difference: +0.20 markers/response
- Stdev of differences: 3.75
- t = 0.207, df = 14
- **p >> 0.05 — NOT statistically significant**

---

## Key Findings

### 1. Hypothesis NOT supported — sarcasm already at ceiling

All three conditions achieve 100% strong sarcasm. The V4 system prompt alone saturates the sarcasm ceiling. There is no room for the purified vector to improve. The 3-6% polite component removed was negligible at alpha=8.

### 2. Purified vector shifts sarcasm REGISTER, not amount

The most interesting finding. Purified trades one sarcasm flavor for another:

| Category | Original | Purified | Shift |
|----------|---------|---------|-------|
| Condescension | 11 | 17 | **+6** |
| Dismissive phrases | 13 | 19 | **+6** |
| Aristocratic contempt | 3 | 7 | **+4** |
| Self-aggrandizement | 12 | 15 | +3 |
| Intellectual superiority | 7 | 2 | **-5** |
| Direct insults | 16 | 13 | -3 |

The polite component was not suppressing sarcasm — it was **channeling** the model toward blunter, more direct insults. Without it, the model drifts toward a more refined, aristocratic register.

### 3. Purified vector is LESS stable

Variance 17.03 vs 7.43 for original. Purified produces extreme outcomes in both directions (20 markers on beer can prompt, 3 markers on homework prompt). The polite component apparently provided some stabilization. This is undesirable for deployment.

### 4. V4 prompt is THE sarcasm engine

Confirms prior findings. V4-only achieves 100% strong sarcasm without any steering. Activation steering at alpha=8 adds marginal value for sarcasm in this regime. Its real benefit is math protection and identity anchoring (not tested here).

### 5. Math accuracy not testable

The test prompt bank (15 prompts from `test_prompts.json`) contains no pure math problems. To test the math preservation hypothesis, need to include arithmetic/AIME-style prompts.

---

## Implications

1. **At alpha=8 with V4, sarcasm is saturated.** Orthogonal projection is solving a problem that doesn't exist at this operating point. Test at alpha=2-4 where sarcasm is NOT saturated.

2. **Sarcasm is not one feature.** The register shift (direct insults vs aristocratic contempt) connects to the SAE work — sarcasm decomposes into sub-features. The polite entanglement was selecting between them.

3. **Stability matters.** The purified vector's higher variance makes it worse for deployment even if average metrics are similar. Regularization from "contaminating" dimensions may be beneficial.

4. **The normalization bug revealed raw z-score magnitude problem.** The 97-110 norm on raw connectome rows means any script using them as steering vectors needs explicit normalization. This should be standardized in the pipeline.

---

## Data

- Results: `orthogonal_sarcasm_results/orthogonal_eval_20260228_022725.json`
- Vector analysis: `orthogonal_sarcasm_results/vector_analysis_8b.json`
- Script (fixed): `run_orthogonal_eval_devserver.py`
- Original script (bug): `orthogonal_sarcasm_steering.py`
- Connectome: `qwen_connectome/analysis/connectome_zscores.pt`

---

## Follow-up Experiments

1. **Lower alpha sweep** (alpha=2,4,6,8): Test where the sarcasm ceiling lives
2. **Math prompt bank**: Add 10+ arithmetic/AIME prompts to evaluate reasoning preservation
3. **Compound vector comparison**: Test the `validate_champion.py` compound vector (push/pull/protect) vs raw sarcasm direction
4. **27B replication** (task #40): Same experiment on Qwen3.5-27B at L48-L55
