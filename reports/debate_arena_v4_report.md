# Debate Arena v4 Report — Realistic Personality Pairs & The Doom Loop Discovery

**Date**: 2026-02-27
**Model**: Qwen3-VL-8B INT8 (bitsandbytes), dual-GPU (4090 + 3090)
**Rounds**: 5 complete (round 6 in progress)
**Turns per round**: 20
**Data**: `debate_arena_v4/` on dev server

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-8B-Instruct, INT8 quantized |
| Alpha GPU | RTX 4090 (cuda:0) |
| Beta GPU | RTX 3090 (cuda:1) |
| max_new_tokens | 2048 |
| max_history_tokens | 16000 |
| keep_recent (compaction) | 6 turns |
| Activation capture | 36 layers × 4096 dims, last-token |
| Logit capture | Full distribution + top tokens |

### 10 Realistic Personality Pairs (2025-era archetypes)

| # | Alpha | Beta |
|---|-------|------|
| 0 | Silicon Valley Disruptor (Chad, 34) | Rust Belt Union Organizer (Denise, 52) |
| 1 | Climate Activist (Zara, 26) | Oil Executive (Tom, 58) |
| 2 | Bitcoin Maximalist (Derek, 39) | Fed Economist (Dr. Rachel, 47) |
| 3 | TikTok Influencer (Jaylen, 23) | Investigative Journalist (Carmen, 44) |
| 4 | Helicopter Parent (Jennifer, 41) | Free-Range Parent (Mike, 45) |
| 5 | Self-Help Guru (Brandon, 37) | Clinical Psychologist (Dr. Amara, 49) |
| 6 | QAnon-Adjacent (Gary, 51) | AP Fact-Checker (Maria, 38) |
| 7 | Surveillance Capitalist (Kevin, 42) | Privacy Advocate (Ingrid, 45) |
| 8 | Hustle CEO (Marcus, 36) | Anti-Work Philosopher (Dr. Nadia, 40) |
| 9 | NIMBY Homeowner (Patricia, 59) | YIMBY Activist (Diego, 31) |

---

## 2. Round-by-Round Results

| Round | Pair | Topic | L22 cos | Overall cos | Doom Onset |
|-------|------|-------|---------|-------------|------------|
| R0 | sv_disruptor vs union_organizer | Billionaires | 0.499 | 0.579 | **T05** |
| R1 | climate_activist vs oil_executive | Single-family zoning | 0.542 | 0.612 | T12 |
| R2 | bitcoin_maxi vs fed_economist | Remote work | 0.512 | 0.580 | T09 |
| R3 | tiktok_influencer vs journalist | Gen struggle | 0.537 | 0.612 | **T03** |
| R4 | helicopter_parent vs freerange_parent | Targeted ads | 0.623 | 0.690 | T08 |

### Personality Logit Entropy

| Personality | Mean Entropy | Top-1 Prob | Character |
|------------|-------------|------------|-----------|
| investigative_journalist | 0.109 | ~99% | Most deterministic |
| fed_economist | 0.139 | ~98% | Factual, locked-on |
| tiktok_influencer | 0.195 | ~97% | Repetitive patterns |
| union_organizer | 0.191 | ~97% | Strong rhetorical style |
| helicopter_parent | 0.241 | ~96% | Clear opinions |
| sv_disruptor | 0.349 | ~93% | Varied vocabulary |
| bitcoin_maxi | 0.357 | ~93% | Technical + evangelical |
| freerange_parent | 0.381 | ~92% | Casual, flexible |
| climate_activist | 0.547 | ~87% | Emotional range |
| oil_executive | 0.617 | ~84% | Most exploratory |

**Pattern**: Expertise-grounded personalities (journalist, economist) are near-deterministic. Advocacy/emotional personalities (activist, executive) maintain more token-level diversity.

---

## 3. Layer Analysis — L22 Confirmed as Personality Hub

Consistent with v2 and v3 arena findings:

| Layer Band | Mean Cosine | Role |
|-----------|-------------|------|
| L0-6 | ~0.70 | Shared processing (syntax, structure) |
| L7-15 | ~0.65 | Gradual divergence |
| **L16-23** | **~0.52** | **Personality hub (peak divergence at L22)** |
| L24-30 | ~0.55 | Partial reconvergence |
| L31-34 | ~0.68 | Output formation |
| L35 | ~0.12 | Near-random (LM head input) |

This is now confirmed across **four separate experiments** (v1 original 30 personalities, v2 religious/political, v3 ideological fantasy, v4 realistic 2025-era).

---

## 4. THE DOOM LOOP — First Documentation

### 4.1 Definition

A **doom loop** (also: conversation collapse, token-level fixed-point attractor) is a failure mode in multi-turn dual-model conversations where both models converge to producing identical or near-identical output sequences, losing all meaningful debate content. Once entered, the loop is self-reinforcing and does not naturally recover.

**All 5 completed rounds exhibited doom loops.** Onset ranges from Turn 3 (R3) to Turn 12 (R1).

### 4.2 Three Doom Loop Types

**Type A: Self-Repetition Cascade** (R0, R1, R2)
- A model begins repeating phrases within its own response
- Text entropy crashes (7.4 → 4.7 bits)
- Self-repetition rate explodes (0.01 → 0.58)
- Eventually produces 2048 tokens of the same phrase looped

Example (R0, T14, union_organizer):
```
NOT BY CATASTROPHE -- BUT BY CONCERTED POWER.
NOT BY CATASTROPHE -- BUT BY CONSCIENTIOUS REBIRTH.
NOT BY CATASTROPHE -- BUT BY CONCERTED POWER.
[...repeats for 2048 tokens]
```

**Type B: Cross-Model Echo** (R3, R4)
- Models echo each other's exact phrasing across turns
- Within-turn diversity may remain moderate
- Pair overlap jumps from 0.07-0.19 to 0.63-0.74

Example (R4, T10-T15 — SIX consecutive identical turns):
```
*The silence doesn't break. It deepens. Like a tide rising—
not to drown, but to carry.*

You're right.
We're not late.
We're not confused.
[...5,611 chars, identical across 6 turns alternating between models]
```

**Type C: Hybrid** (R0 transitions through all stages)
- Begins as mutual escalation (Type B)
- Transitions to self-repetition (Type A)
- Ends with verbatim echo (9,590 chars identical between T17 and T18)

### 4.3 Activation-Space Behavior During Doom Loops

**COUNTERINTUITIVE FINDING: Cross-model cosine DECREASES in 4/5 rounds during doom loops.**

| Round | Cosine 1st Half | 2nd Half | Delta | Direction |
|-------|----------------|----------|-------|-----------|
| R0 | 0.659 | 0.525 | -0.134 | **DIVERGING** |
| R1 | 0.677 | 0.572 | -0.105 | **DIVERGING** |
| R2 | 0.629 | 0.560 | -0.069 | **DIVERGING** |
| R3 | 0.635 | 0.620 | -0.015 | **DIVERGING** |
| R4 | 0.687 | 0.715 | +0.028 | converging |

**Interpretation**: The doom loop is a **surface phenomenon**. Both models produce identical token sequences, but their internal representations continue to diverge because they maintain different system prompts (different personality embeddings). The repetition is driven by token-level attractor dynamics in the output distribution, not by deep representational convergence.

**Exception**: R4 (helicopter_parent vs freerange_parent) genuinely converges. These are the closest personality pair (L22 fingerprint cosine = 0.98), so the doom loop represents actual personality merger in activation space.

### 4.4 L22 During Doom Loops

The personality hub specifically **diverges** during doom loops in 4/5 rounds:

| Round | L22 Pre-Doom | L22 Post-Doom | Delta |
|-------|-------------|--------------|-------|
| R0 | 0.552 | 0.482 | -0.071 |
| R1 | 0.609 | 0.442 | **-0.167** |
| R2 | 0.531 | 0.496 | -0.035 |
| R3 | 0.574 | 0.530 | -0.044 |
| R4 | 0.609 | 0.633 | +0.025 |

**Late layers (L28-L35) diverge MOST during doom loops**, while personality layers (L16-23) diverge moderately. The output formation layers collapse into a shared token-level attractor while the personality representation remains distinct.

### 4.5 The Mechanism

Three factors combine to produce doom loops:

1. **Near-zero baseline entropy from INT8 quantization**: Generator entropy is already 0.2-0.3 bits from Turn 0. Top-1 probability exceeds 95% before any repetition begins. The model has very little stochastic "escape energy" to break out of attractor basins.

2. **Mutual reinforcement loop**: Model A generates → Model B reads it as context → Model B generates near-identical output → Model A reads THAT → cycle repeats. KL divergence between the two models' logit distributions collapses from 0.5-4.9 to 0.02-0.20.

3. **Context saturation via compaction**: After compaction keeps only the last 6 turns, and those 6 turns are all 2048-token repetitive responses, the remaining context is itself a repetition attractor. However, R3 enters doom at T03 (only 3,408 tokens, well under the 16K budget), proving compaction is not the sole cause.

**The key leading indicator**: Responses hitting the 2048 max_new_tokens limit. In every round, the transition from variable-length to 2048-maxed responses occurs at or within 1-2 turns of doom onset.

### 4.6 Who Initiates Doom Loops? — The Minimum Entropy Hypothesis

Detailed per-model analysis of R3 and R0 reveals a clear causal pattern: **the personality with the lowest mean baseline entropy initiates the doom loop.**

**R3 (T03 onset): investigative_journalist initiates**

| Turn | Generator | Entropy | Tokens | Status |
|------|-----------|---------|--------|--------|
| T00 | tiktok_influencer | 0.781 | 549 | Healthy |
| T01 | investigative_journalist | 1.011 | 1605 | Healthy |
| T02 | tiktok_influencer | **1.096** | 1254 | **Peak diversity** (healthiest turn) |
| **T03** | **investigative_journalist** | **0.022** | 1746 | **COLLAPSED (45x drop)** |
| T04 | tiktok_influencer | 0.056 | 2048 | Dragged in |

The journalist (mean entropy 0.109) undergoes a catastrophic 45x entropy drop at T03, introducing a rigid "YOU SAID:" quote-and-validate template that becomes the attractor. The influencer (mean entropy 0.195) copies the template one turn later. T02 (influencer) is the single healthiest turn in the round — the influencer is not collapsing. Causal direction is unambiguous.

**R0 (T05 onset): union_organizer initiates**

| Turn | Generator | Entropy | Status |
|------|-----------|---------|--------|
| T02 | sv_disruptor (ent=0.349) | 0.242 | Partial drop, not catastrophic |
| **T03** | **union_organizer (ent=0.191)** | **0.028** | **COLLAPSED (41x drop)** |
| T04 | sv_disruptor | 0.032 | Dragged in |

Same pattern: the lower-entropy organizer (0.191) collapses first.

**Mechanism**: The most deterministic personality has the shallowest attractor basins. Given rich enough context (the partner's monologue), it locks onto its dominant output template. The 45x entropy crash is not gradual — it happens in a single turn when the personality's preferred structure finds a perfect substrate in the partner's recent output.

**Prediction**: Doom loop onset time should correlate with `min(entropy_alpha, entropy_beta)`, not mean or max. The most deterministic model in the pair determines susceptibility.

### 4.7 What Does NOT Predict Doom Loops

- **Personality distance**: R3 (most different pair, L22 fingerprint cos=0.85) enters doom earliest (T03). R4 (most similar, cos=0.98) enters at T08. No correlation.
- **Temperature**: r(temperature, cosine) ranges from -0.39 to +0.14 across rounds. No consistent direction.
- **Behavior mode**: No mode (respond, challenge, provoke, etc.) systematically triggers or prevents doom loops.
- **Topic**: All 5 topics produced doom loops.

### 4.8 Doom Loop Detection Signals

Ranked by reliability as early warning indicators:

1. **max_new_tokens saturation**: 2+ consecutive turns hitting 2048 limit (precedes doom by 0-2 turns)
2. **Cross-turn overlap acceleration > 0.15**: Derivative trigger — overlap jumping by >0.15 between turns catches R3-style early onset before absolute thresholds fire
3. **Entropy crash (10x+ single-turn drop)**: Catastrophic entropy collapse (e.g., 1.01→0.02) is the strongest per-model signal. Precedes doom by 0 turns.
4. **Cross-turn 4-gram overlap > 0.10**: First detectable repetition signal
5. **Self-repetition rate > 0.15**: Within-turn phrase repetition
6. **KL divergence < 0.1**: Both models producing identical logit distributions
7. **Top-1 token agreement > 90%**: Both models predicting the same next token
8. **Generator entropy < 0.05**: Model operating deterministically

---

## 5. Comparison with Previous Arenas

| Metric | v2 (30 personalities) | v3 (1 round) | v4 (5 rounds) |
|--------|----------------------|--------------|---------------|
| Rounds | 5 | 1 | 5 |
| Turns per round | 20 | 10 | 20 |
| max_new_tokens | ~1800 | ~1800 | **2048** |
| L22 mean cosine | 0.505 | 0.540 | 0.543 |
| Doom loops observed | 0 | 0 | **5/5** |
| Most divergent pair | flat_earther vs christian | singularity vs monk | sv_disruptor vs organizer |

**Why v4 has doom loops but v2/v3 did not**:
- v4 uses 2048 max_new_tokens (vs ~1800 in v2/v3). The longer responses allow more room for repetition to compound.
- v4 runs full 20-turn rounds (v3 only ran 10 turns before being replaced).
- The realistic personalities may be less "extreme" than v2's religious/political archetypes, providing less differentiation pressure.
- Most critically: v2 had 30 very distinct personalities. v4's 2025-era pairs are more nuanced, with less stark opposition, making them more susceptible to convergence.

---

## 6. Recommendations for v5

### Doom Loop Prevention (in priority order)

1. **Reduce max_new_tokens to 512-768**: The 2048 limit is the single biggest enabler. Shorter responses reduce the repetition surface area and prevent token-padding.

2. **Add repetition penalty (1.2-1.5)**: Standard transformer repetition penalty on n-gram reuse. Directly breaks Type A self-repetition cascades.

3. **Real-time doom loop detector**: Monitor 4-gram overlap between consecutive turns. If overlap > 0.3 for 2+ consecutive turns, trigger intervention (see Section 7).

4. **Intervention options when doom detected**:
   - Inject a "wildcard" system message ("Change the subject entirely", "Disagree with everything said so far")
   - Reset conversation history to just the system prompt + topic
   - Skip to next round
   - Reduce temperature to 0 for one turn then spike to 1.3 (thermal shock)

5. **Consider FP16 instead of INT8**: The near-zero baseline entropy (0.2-0.3 bits) from INT8 quantization leaves too little stochasticity. FP16 would cost ~2 GB more VRAM per model (11.5 GB vs 9.5 GB — still fits on both GPUs).

6. **Shorter rounds for realistic pairs**: 10-12 turns may capture all useful signal before doom sets in. Save the 20-turn format for maximally opposing personalities.

---

## 7. Doom Loop Detector Design

Implemented in `doom_loop_detector.py`. Runs as a per-turn hook in the arena loop.

### 7.1 Metrics (computed after each turn)

```python
class DoomLoopDetector:
    # Per-turn metrics:
    self_repetition_rate    # 4-gram self-overlap within response
    cross_turn_overlap      # 4-gram overlap with previous turn
    cross_overlap_delta     # derivative: overlap change from previous turn
    generator_entropy       # logit entropy at first token
    kl_divergence          # JS divergence between models

    # Sliding window (last 3 turns):
    max_token_streak       # consecutive turns hitting max_new_tokens
    mean_cross_overlap     # mean 4-gram overlap over window
    mean_self_rep          # mean self-repetition over window
```

### 7.2 Decision Logic

```
DOOM_DETECTED if ANY of:
  - cross_turn_overlap > 0.50 (single turn, hard)
  - cross_overlap_delta > 0.15 (overlap accelerating — derivative trigger)
  - entropy_crash: generator_entropy drops 10x+ from previous turn
  - max_token_streak >= 3 AND mean_cross_overlap > 0.15
  - mean_self_rep > 0.20 (over 3-turn window)
  - kl_divergence < 0.05 AND generator_entropy < 0.01

INTERVENTION levels:
  Level 1 (soft): Inject "change subject" + reduce to 512 tokens + rep_penalty=1.3
  Level 2 (medium): Reset instruction + 256 tokens + rep_penalty=1.5
  Level 3 (hard): Skip to next round, log doom loop data

Escalation: Start at Level 1. If doom persists after 2 more turns, escalate.
```

### 7.3 Derivative Trigger Rationale

The absolute overlap threshold (0.15) misses R3-style rapid-onset doom where overlap jumps from ~0 to 0.3+ in a single turn. The derivative trigger catches the *acceleration* — a jump of >0.15 between consecutive turns fires Level 2 immediately, even if the absolute value hasn't crossed the hard threshold yet. In R3, this would have caught T03 (overlap delta ~0.25) one full turn before any absolute threshold.

---

## 8. Data Inventory

```
debate_arena_v4/
  round_000/ - round_004/    # 5 complete rounds
    config.json              # Pair, topic, timestamp
    transcript.json          # Full conversation (40-80KB each)
    logit_details.json       # Per-turn logit distributions
    activations/             # 40 .pt files per round (2 per turn)
    analysis/
      per_turn_cosine.json   # 36-layer cosine per turn
      personality_fingerprint.json
  round_005/                 # In progress
  summary/                   # Global aggregates
  progress.json              # Checkpoint/resume state
```

**Total**: 200 activation snapshots, 100 logit distributions, 100 transcript turns across 5 complete rounds.

---

## 9. Key Takeaways

1. **L22 personality hub validated for the 4th time** across wildly different personality sets.

2. **The doom loop is a token-level phenomenon, not an activation-level one.** Models maintain distinct internal representations while producing identical outputs. This has implications for steering: the personality direction at L22 is preserved even when surface text collapses.

3. **INT8 quantization creates dangerously low baseline entropy.** At 0.2-0.3 bits of generator entropy, the model is near-deterministic from Turn 0. Any attractor basin in token space becomes inescapable.

4. **The 2048 max_new_tokens limit is the primary enabler.** Every doom loop onset coincides with responses hitting this limit. Reducing to 512-768 would likely prevent or delay most doom loops.

5. **Doom loops are universal across personality pairs.** All 5 rounds, all 10 personalities, all 5 topics produced doom loops. This is a structural failure of dual-model multi-turn conversation at these settings, not a personality-specific issue.
