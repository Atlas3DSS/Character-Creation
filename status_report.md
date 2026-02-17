# Skippy Identity Baking — Status Report (Feb 16, 2026)

## Executive Summary

After exhaustive experimentation across 7+ approaches and 4 prompt variants, we've established clear boundaries on what's achievable with an 8B parameter model for personality baking.

**Production configs (two tiers):**

| Config | Use Case | Personality | AIME | Prompt |
|--------|----------|-------------|------|--------|
| SDFT scale 0.5 + V4 prompt | General (AIME-safe) | 6.1/10 | 43.3% | V4 (behavioral constraints) |
| SDFT scale 1.0 + V1 prompt | Voice pipeline | 7.8/10 | 36.7% | V1 (enhanced, name-free) |

**Key limitation**: Personality is prompt-dependent (2.8/10 without prompt at scale 0.5).

## Approach Summary

| Approach | Personality | AIME | Verdict |
|----------|-------------|------|---------|
| Base + system prompt | 4.6/10 | 46.7% | AI-assistant-ish |
| LoRA SFT merge (scale=0.5) + prompt | 5.4/10 | 46.7% | Authentic but mild |
| SDFT scale 0.5 + V1 prompt | 6.0/10 | 43.3% | Good balance, some non-sequiturs |
| **SDFT scale 0.5 + V4 prompt** | **6.1/10** | **43.3%** | **Best AIME-safe: +0.31 technical, +0.26 on-topic** |
| SDFT scale 0.75 + V1 prompt | ~7.3/10 | 36.7% | Redundant (same AIME as 1.0) |
| **SDFT scale 1.0 + V1 prompt** | **7.8/10** | 36.7% | **Best personality, best for voice** |
| SDFT scale 1.0 + V4 prompt | ~7.5/10 | 36.7% | POV confusion from behavioral constraints |
| SDFT scale 0.5 (no prompt) | 2.8/10 | 43.3% | Stock Qwen behavior |
| Reasoning-protected ablation (best α) | 5.15/10 | untested | Marginal improvement, incoherent |
| LoRA SFT (any config) | variable | 0% | Catastrophic forgetting |
| DPO R1/R2 | variable | 40% | Surface mimicry only |
| Contrastive ablation (SVD) | gibberish | 0% | Destroys coherence |
| Steering vectors | ~4/10 | ~46% | Too low-dimensional |

## Prompt Engineering Results

Tested 4 prompt variants on SDFT scale 0.5 (3 runs each, averaged):

| Variant | Strategy | Technical | On-topic | Arrogance | Sarcasm | Overall |
|---------|----------|-----------|----------|-----------|---------|---------|
| V1 (baseline) | Personality-focused, narrative | 7.08 | 5.78 | 3.12 | 3.44 | 6.00 |
| V2 | Rules-first (CORE RULES) | 7.31 | 5.96 | 3.44 | 2.92 | 6.01 |
| V3 | Persona-integrated + examples | 7.54 | 6.32 | 2.60 | 3.04 | 5.89 |
| **V4** (winner) | **V1 + minimal behavioral fixes** | **7.38** | **6.04** | **3.33** | 3.20 | **6.10** |

**Key findings:**
- V4 (minimal diff from V1) gives best balance: +0.31 technical, +0.26 on-topic
- V2 rules-first approach kills sarcasm (-0.68 from V1)
- V3 examples get parroted verbatim and cause character breaks ("I cannot tell jokes")
- Sarcasm is the hardest dimension to prompt-engineer (-0.24 with V4)
- Scale 1.0 model doesn't benefit from V4 — personality is already strong, behavioral constraints cause POV confusion

**Scale 1.0 with V1 prompt is the strongest overall Skippy** — "Hawking radiation, dumdum" is perfection.

## Key Findings

### 1. AIME Phase Transition
Scale 0.5 → 0.55 shows a binary phase transition: 43.3% → 36.7%. Not gradual. Scale 0.75 also gets 36.7% — the transition is all-or-nothing.

### 2. Personality is Contextual
Personality cannot be baked via static weight modification. All ablation approaches (bias, rotation, mutation, SVD, probe-guided) fail because:
- 50-63% of personality delta overlaps reasoning subspace
- Personality is a conditional distribution shift, not an activation constant
- Static bias shifts ALL activations regardless of input

### 3. Prompt Engineering Has Diminishing Returns
4 prompt variants across multiple runs show ~0.1-0.3 improvements per dimension. The ceiling is set by the model's capacity, not the prompt. More elaborate prompts (V2, V3) can actually hurt personality.

### 4. Model Scale Is The Key Variable
Scale 1.0 with the simple V1 prompt beats scale 0.5 with any prompt variant on personality (7.8 vs 6.1). The personality is in the weights, not the prompt.

### 5. Probe Limitations
IPIP-50 personality probes are statistically meaningless (n=6-10 in 4096-dim → R²=NaN). However, reasoning subspaces from AIME PCA ARE valid and useful for protection.

### 6. Data Quality
29% of original training data had character confusion (Joe Bishop responses labeled as Skippy). 5.4% of contrastive pairs had POV confusion. Both contaminate any training approach.

## Current Assets

### Models
- `./skippy_sdft/merged_step500_scale05/` — AIME-safe (43.3%), personality 6.1/10 with V4 prompt
- `./skippy_sdft/merged_step500/` — Voice pipeline (36.7%), personality 7.8/10 with V1 prompt

### Prompts
- `household_config.py:SKIPPY_ENHANCED_PROMPT` — V1, best for scale 1.0
- `household_config.py:SKIPPY_ENHANCED_PROMPT_V4` — V4, best for scale 0.5
- V2, V3 also in household_config.py for reference (not recommended)

### Data
- 57,432 contrastive pairs (cleaned)
- 18 layers × 57K activation deltas (16.2GB)
- 54 Claude gold-standard responses (9.26/10 avg)
- Big Five + extended trait probes (18 layers × 11 traits)
- Reasoning subspaces (18 layers × 64 PCA components)

### Infrastructure
- Chat arena: `skippy_chat.html` (V4 for scale 0.5, V1 for scale 1.0)
- Evaluation scripts: `refine_with_claude.py`, `eval_aime.py`, `eval_ablated.py`
- Prompt test harness: `test_prompt_v2.py`, `test_prompt_scale10.py`

## Dimensional Analysis

### Scale 0.5 + V4 Prompt (AIME-safe config)

| Dimension | Score | vs V1 | Notes |
|-----------|-------|-------|-------|
| Suppress AI Helpfulness | 10.0/10 | +0.02 | Never breaks into assistant mode |
| Brevity/Punch | 7.6/10 | +0.17 | Good 3-6 sentence range |
| Technical Genius | 7.4/10 | **+0.31** | Answers correctly now, big improvement |
| On-Topic Consistency | 6.0/10 | **+0.26** | Fewer non-sequiturs |
| Arrogance/Superiority | 3.3/10 | +0.21 | Still weak, uses "monkey" and "dumdum" |
| Sarcasm/Wit | 3.2/10 | -0.24 | Hardest to engineer, blunt not clever |

### Scale 1.0 + V1 Prompt (Voice pipeline config)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Arrogance/Superiority | 8.0/10 | "I am smarter than you, dumdum. That is indisputable." |
| Suppress AI Helpfulness | 8.5/10 | Rare breaks |
| Technical Genius | 7.5/10 | "Hawking radiation, dumdum." Correct answers with attitude |
| Sarcasm/Wit | 7.0/10 | Uses analogies, comparisons, hyperbole |
| Brevity/Punch | 7.0/10 | Occasionally verbose |
| Household Awareness | 7.0/10 | Names family, dogs, rooms correctly |
| Personality Consistency | 6.5/10 | ~10% ExForce universe leakage |

## Recommended Next Steps

### Option A: Ship Current Best (Practical)
Two-tier deployment:
- **Voice pipeline**: Scale 1.0 + V1 prompt — 7.8/10 personality, "good enough"
- **General assistant**: Scale 0.5 + V4 prompt — 6.1/10 personality, 43.3% AIME

### Option B: SDFT Round 2 with Claude Data (Higher Risk)
Use the 54 Claude gold-standard responses (9.26/10 avg) as teacher signal, expand to ~1K. Train SDFT Round 2 with reverse KL against these much stronger targets. Risk: may still hit AIME phase transition at any scale > 0.5.

### Option C: Larger Model (Highest Impact)
The 8B model's capacity ceiling has been hit. A 14B or 32B model would have more weight space to separate personality from reasoning. This is the only path to truly baked personality (no prompt needed).

## Files Modified This Session
- `household_config.py` — Added SKIPPY_ENHANCED_PROMPT_V2, V3, V4
- `skippy_chat.html` — Updated arena with V4/V1 prompts per model
- `test_prompt_v2.py` — Prompt comparison harness (V1 vs V4, 3 runs averaged)
- `test_prompt_scale10.py` — Scale 1.0 prompt comparison
- `review_logs/comparison_v1_v4_*.json` — Averaged comparison data
- `review_logs/responses_v4_*.json` — V4 prompt responses
- `review_logs/responses_v4_scale10_*.json` — Scale 1.0 + V4 responses
