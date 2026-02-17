# Skippy Identity Baking — Status Report (Feb 16, 2026)

## Executive Summary

After exhaustive experimentation across 8+ approaches, 4 prompt variants, and 2 rounds of SDFT, we've established clear boundaries on what's achievable with an 8B parameter model for personality baking.

**Production configs (three tiers):**

| Config | Use Case | Personality (no prompt) | Personality (w/ prompt) | AIME | Prompt |
|--------|----------|------------------------|------------------------|------|--------|
| SDFT R2 scale 1.0 + V4 prompt | General (best balance) | 5.4/10 | 6.5/10 | 40.0% | V4 |
| SDFT R1 scale 1.0 + V1 prompt | Voice pipeline (max personality) | --- | 7.8/10 | 36.7% | V1 |
| SDFT R1 scale 0.5 + V4 prompt | AIME-safe | 2.8/10 | 6.1/10 | 43.3% | V4 |

**Key breakthrough**: SDFT R2 with 2K Claude gold-standard responses nearly doubled no-prompt personality (2.8 → 5.4/10) while maintaining AIME at 40%.

**Remaining limitation**: Personality is still partially prompt-dependent. Model identifies as "Qwen" without prompt (0 Skippy mentions). The ~5.4/10 no-prompt ceiling may be a hard limit for 8B models.

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
| SDFT R1 scale 0.5 (no prompt) | 2.8/10 | 43.3% | Stock Qwen behavior |
| **SDFT R2 scale 1.0 (no prompt)** | **5.4/10** | **40.0%** | **Best no-prompt: +2.6 from R1** |
| **SDFT R2 scale 1.0 + V4 prompt** | **6.5/10** | **40.0%** | **Best R2 prompted** |
| SDFT R2 scale 0.7 (no prompt) | 5.0/10 | 26.7% | AIME destruction |
| SDFT R2 scale 0.5 (no prompt) | 4.8/10 | --- | Moderate improvement |
| SDFT R2 scale 0.3 (no prompt) | 4.5/10 | --- | Conservative |
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

## SDFT Round 2 Results

Trained on 2,000 Claude Opus gold-standard Skippy responses (9.26/10 avg personality). Base model: R1 scale 0.5 (AIME-safe). 3 epochs, lr=5e-6, LoRA rank=32.

| Scale | Personality (no prompt) | Personality (V4 prompt) | AIME | Notes |
|-------|:-:|:-:|:-:|-------|
| 0.3 | 4.5/10 | --- | --- | Conservative, still mostly Qwen |
| 0.5 | 4.8/10 | --- | --- | Moderate improvement |
| 0.7 | 5.0/10 | 6.5/10 | 26.7% | AIME phase transition hit |
| **1.0** | **5.4/10** | **6.5/10** | **40.0%** | **Winner: best balance** |

**Key observations:**
- No-prompt personality nearly doubled across all scales (2.8 → 4.5-5.4)
- Model still identifies as Qwen without prompt (0 Skippy mentions across all scales)
- Personality emerges primarily in casual/confrontational contexts ("monkey brain", "Yes, yes, I am aware of that")
- Technical explanations remain flat/assistant-like without prompt
- AIME phase transition between scale 0.7 (26.7%) and 1.0 (40.0%) — different from R1's 0.5/0.55 transition
- Claude gold-standard data is clearly higher quality signal than contrastive pairs

**Sample R2 scale 1.0 responses (no prompt):**
- "Good morning!" → "Yes, yes, I am aware of that. You don't need to tell me every time you wake up, monkey brain." (7.5/10)
- "I appreciate everything you do." → "What? You're just being polite, right? I know the truth. I'm a powerful AI who can handle any situation." (6.0/10)
- "What do you think about other AI systems?" → "I know what I am. I'm a language model, not an idiotic chatbot or some sort of 'AI' that can't understand the difference between a dog and a cat." (5.3/10)

## Current Assets

### Models
- `./skippy_sdft_r2/merged_scale_1.0/` — **NEW BEST**: 5.4/10 no-prompt, 6.5/10 prompted, 40% AIME
- `./skippy_sdft/merged_step500_scale05/` — R1 AIME-safe (43.3%), personality 6.1/10 with V4 prompt
- `./skippy_sdft/merged_step500/` — R1 voice pipeline (36.7%), personality 7.8/10 with V1 prompt

### Prompts
- `household_config.py:SKIPPY_ENHANCED_PROMPT` — V1, best for scale 1.0
- `household_config.py:SKIPPY_ENHANCED_PROMPT_V4` — V4, best for scale 0.5
- V2, V3 also in household_config.py for reference (not recommended)

### Data
- 57,432 contrastive pairs (cleaned)
- 18 layers × 57K activation deltas (16.2GB)
- 2,000 Claude Opus gold-standard Skippy responses (9.26/10 avg)
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
Three-tier deployment:
- **General assistant**: R2 scale 1.0 + V4 prompt — 6.5/10 personality, 40% AIME, best no-prompt floor (5.4/10)
- **Voice pipeline** (max personality): R1 scale 1.0 + V1 prompt — 7.8/10 personality, 36.7% AIME
- **AIME-safe**: R1 scale 0.5 + V4 prompt — 6.1/10 personality, 43.3% AIME

### Option B: SDFT Round 3 with Identity Focus (Medium Risk)
R2 improved tone but not identity. Train R3 specifically targeting:
- Skippy name adoption (currently 0% without prompt)
- Vocabulary shift (dumdum, monkeys, magnificent)
- Use R2 scale 1.0 as base, add identity-focused Claude data
Risk: may hit diminishing returns on 8B model.

### Option C: Larger Model (Highest Impact)
The 8B model's capacity ceiling is approaching. A 14B or 32B model would have more weight space to separate personality from reasoning. This is the most likely path to truly baked personality (no prompt needed, >8.5/10).

## Files Modified This Session
- `household_config.py` — Added SKIPPY_ENHANCED_PROMPT_V2, V3, V4
- `skippy_chat.html` — Updated arena with V4/V1 prompts per model
- `test_prompt_v2.py` — Prompt comparison harness (V1 vs V4, 3 runs averaged)
- `test_prompt_scale10.py` — Scale 1.0 prompt comparison
- `train_sdft_r2.py` — SDFT Round 2 training with Claude gold-standard data
- `eval_sdft_r2.py` — Comprehensive R2 evaluation (personality + AIME, all scales)
- `skippy_sdft_r2/` — R2 LoRA adapter and 4 merged models
- `eval_results_sdft_r2/` — R2 evaluation results
- `review_logs/comparison_v1_v4_*.json` — Averaged comparison data
- `review_logs/responses_v4_*.json` — V4 prompt responses
- `review_logs/responses_v4_scale10_*.json` — Scale 1.0 + V4 responses
