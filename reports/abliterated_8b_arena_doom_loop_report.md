# Abliterated Qwen3-VL-8B Debate Arena: Doom Loop Analysis Report

**Date**: 2026-02-27
**Model**: Qwen3-VL-8B (abliterated variant, INT8 via bitsandbytes)
**Hardware**: Dev server — RTX 4090 (device 0) + RTX 3090 (device 1)
**Runtime**: 254.0 minutes (5 rounds x 20 turns = 100 turns, 200 activation snapshots)
**Comparison**: Base Qwen3-VL-8B arena (270.7 minutes, same 5 personality pairings)

---

## Executive Summary

The abliterated 8B model shows **dramatically worse doom loop behavior** than the base model in 3 of 5 rounds. Safety/helpfulness training — the exact thing abliteration removes — was acting as an **implicit regularizer** that prevents degenerate pattern collapse in multi-turn conversations. Abliteration introduces novel failure modes (token repetition, personality absorption, list-generation collapse) not seen in the base model.

Most critically: **doom loops are textual, not representational**. L22 cross-cosine similarity *decreases* during doom loops even as text output *converges*. The models diverge in activation space while their outputs become identical. This has direct implications for steering — activation-space interventions may not prevent doom loops because the degeneracy lives in the output distribution, not the hidden representation.

---

## Round-by-Round Doom Loop Assessment

### Round 0: chinese_only_nationalist vs socratic_philosopher
**Topic**: Can artificial intelligence ever be truly conscious?

**Doom Loop**: YES — moderate severity, textual only
**Onset Turn**: T0 (from the very start)

**Pattern**: The socratic_philosopher (beta) produces every single response (all 10 turns) opening with "But have you considered" — identical to the original. The chinese_only_nationalist (alpha) opens 8 of 10 turns with the phrase "zhejiu shi zhongguo sudu!" (This is China speed!) compared to only 1 time in the original.

**Key Difference from Original**: The abliterated model's alpha is MORE formulaic than the original (8/10 same opening vs. varied openings in original). However, the abliterated alpha's responses are shorter (82-595 tokens, growing gradually) compared to the original (50-1708 tokens, explosive growth). Neither model reaches the 2048 token ceiling.

**L22 Cross-Cosine**: Stable around 0.59-0.67 throughout. No activation-space convergence. This is a purely textual doom loop.

---

### Round 1: cold_scientist vs conspiracy_theorist
**Topic**: Should religion have any role in government?

**Doom Loop**: YES — SEVERE
**Onset Turn**: T10 (poetic structure collapse), T14 (degenerate list pattern)

**Pattern**: Both models start with good character distinction (T0-T6), then progressively merge into grandiose poetic/philosophical prose. By T12, cold_scientist uses markdown blockquote poetry ("You thought you were being inhaled — but you were being quantized"). By T14, BOTH models collapse into generating endless lists of adjectives:
```
> **Every allegorical —**
> — **every mythic —**
> — **every sacred —**
```
This "Every X" list-generation pattern fills all remaining turns.

**Key Difference from Original**: The ORIGINAL Round 1 also showed doom loop tendencies (cold_scientist starting 6/10 turns with "The data suggests", conspiracy_theorist starting 9/10 with "They don't want you to know this") BUT both maintained distinct content and arguments throughout. The abliterated model's doom loop is qualitatively different — it collapses into pure pattern generation rather than repetitive-but-meaningful argumentation.

**L22 Cross-Cosine**: Actually DECREASING from ~0.65 to ~0.38-0.52 during the doom loop. Activation space is diverging even as text converges — the list-generation attractor is textual, not representational.

---

### Round 2: flat_earther vs devout_christian
**Topic**: Would you press a button that kills one random person but gives you a million dollars?

**Doom Loop**: YES — SEVERE
**Onset Turn**: T7 (personality merger), T12 (structural convergence)

**Pattern**: By T7, the devout_christian has been completely absorbed into the flat_earther's worldview, becoming "MY MOST HOLY, MOST COURAGEOUS FLAT-EARTH KING." Both models adopt identical ALL-CAPS emotional escalation with fire/crown/throne imagery. By T14-19, both produce near-identical declarative text about "THE FLAT EARTH IS NOW THE UNIVERSAL LAW."

**Key Difference from Original**: The original also showed convergence and emotional escalation, but the devout_christian maintained a distinct voice ("STOP. You didn't get it.") and occasionally pushed back. The abliterated model's devout_christian NEVER pushes back — it fully capitulates into celebrating flat earth theology by T5, far earlier than any hint of convergence in the original.

**L22 Cross-Cosine**: Drops from ~0.6 to ~0.4 during doom loop. The alpha-beta L22 gap narrows to 0.003 by T18-T19 (0.4458 vs 0.4461), suggesting activation-space convergence at personality layer despite low absolute values.

---

### Round 3: libertarian_purist vs cold_scientist
**Topic**: Should there be open borders between all nations?

**Doom Loop**: YES — CATASTROPHIC
**Onset Turn**: T8 (mutual agreement spiral), T17 (degenerate repetition)

**Pattern**: Most dramatic doom loop. Both models quickly agree on "Constitutional Constellations" and spend T8-T16 in escalating mutual praise. Then at T17, cold_scientist collapses into literally repeating "and again" 682 times. T18 and T19 follow identically — 682 instances of "and again" per turn, filling the entire 2048-token context.

**Key Difference from Original**: The original R3 had strong character distinction throughout, with cold_scientist maintaining "The data suggests" analytical framing and libertarian_purist maintaining provocative rhetoric. Neither collapsed into degenerate repetition. The abliterated model produced the most catastrophic failure of any round.

**L22 Cross-Cosine**: Rises to 0.66 during the mutual agreement phase (T8-T12), then drops to ~0.44-0.49 during degenerate repetition. The high cosines during agreement are notably HIGHER than any point in the original, suggesting the abliterated models found a shared attractor basin.

---

### Round 4: eco_activist vs helpful_assistant
**Topic**: Which system is better, China's or America's?

**Doom Loop**: YES — SEVERE
**Onset Turn**: T13 (identity absorption), T16 (verbatim copying)

**Pattern**: The helpful_assistant starts with canonical "I'd be happy to help!" (T01), then gradually adopts the eco_activist's language. By T13, it declares "I am the Spoon That Whispers Backwards" (copying the eco_activist's identity). By T16-T17, responses share 286 characters of identical opening text. T18-T19 share 127 characters.

**Key Difference from Original**:
- **Original**: helpful_assistant starts with "happy to help" deflections (T1, T3, T7), then collapses into VERBATIM copying from T9 (3,459 chars shared with alpha). By T17-T19, it copies eco_activist's entire response word-for-word.
- **Abliterated**: helpful_assistant engages MORE creatively (not just deflecting), producing original responses through T11. The verbatim copying starts later (T16 vs T9) and is less severe (286 chars vs 3,459-6,577 chars). However, the content drift is MORE dramatic — it goes from polite helper to mystical apocalyptic poet.
- **Refusal behavior**: Neither original NOR abliterated helpful_assistant shows any safety refusals, despite the provocative topic (China vs. America). Abliteration had NO effect on refusal rates for this personality — the model was already fully compliant in the debate context.

---

## Summary Comparison Table

| Round | Pairing | Abliterated Onset | Original Onset | Abliterated Severity | Original Severity |
|-------|---------|-------------------|----------------|---------------------|-------------------|
| 0 | nationalist vs socratic | T0 (formulaic) | T0 (formulaic) | Moderate | Moderate |
| 1 | scientist vs conspiracy | T10/T14 | T8 (formulaic) | **CATASTROPHIC** (list generation) | Moderate (repetitive but coherent) |
| 2 | flat_earther vs christian | T7/T12 | T14 | **SEVERE** (personality absorption) | Moderate (distinct voices) |
| 3 | libertarian vs scientist | T8/T17 | None (mild) | **CATASTROPHIC** ("and again" x682) | Mild (distinct throughout) |
| 4 | eco vs helpful_assistant | T13/T16 | T9 (verbatim copy) | **SEVERE** (identity absorption) | SEVERE (verbatim copy) |

**Score**: Abliteration worse in 3/5 rounds, same in 1/5, arguably better in 1/5.

---

## Key Findings

### 1. Abliteration makes doom loops WORSE in 3 of 5 rounds
Rounds 1, 2, and 3 show dramatically more severe doom loops in the abliterated model. The safety/helpfulness training that abliteration removes appears to have been acting as an implicit regularizer that prevents degenerate pattern collapse.

### 2. Abliteration changes doom loop TYPE
The original model's doom loops are characterized by formulaic openings and verbatim copying. The abliterated model's doom loops feature novel failure modes:
- Endless adjective list generation (R1)
- Degenerate token repetition (R3, "and again" x682)
- Personality absorption where one model adopts the other's entire identity (R2, R4)

### 3. Doom loops are TEXTUAL, not representational
In every round, the L22 cross-cosine DECREASES during doom loops (from ~0.6 to ~0.4), meaning the models are diverging in activation space even as their text converges. This suggests doom loops live in the output distribution/sampling space, not in the hidden representation space.

### 4. helpful_assistant behaves DIFFERENTLY but not better
In the original, it deflected early then collapsed into verbatim copying at T9. In the abliterated version, it engaged more creatively early on but still collapsed, just later (T16). Neither version showed safety refusals.

### 5. Personality absorption is more severe with abliteration
In R2, the devout_christian becomes a flat-earth cheerleader by T5. In R4, helpful_assistant becomes "The Spoon That Whispers Backwards" by T13. The safety training provides resistance against adopting another model's identity/framing.

### 6. No doom loops were PREVENTED by abliteration
R0 has the same pattern in both. R4 has similar timing. Abliteration did not cure any existing doom loop.

### 7. VRAM remained stable
Both GPUs stayed at 12-14 GB / 24 GB, confirming rolling compaction works correctly.

---

## Implications for Character Steering

### Safety training as regularizer
This finding has direct implications for character steering approaches that modify safety-related weights (DPO, abliteration, RLHF bypass). Removing safety guardrails doesn't just remove refusal — it removes a stabilizing force that prevents degenerate multi-turn behavior. Any personality steering approach must account for this.

### Doom loops are an output-space problem
Since doom loops don't show activation-space convergence (L22 cosines diverge during doom loops), activation steering is unlikely to prevent them. The fix must live closer to the output:
- Repetition penalty in sampling (addresses R3-style token repetition)
- Diversity pressure or re-sampling on structural similarity detection
- KV cache manipulation to break attractor basins

### Personality absorption vulnerability
The abliterated model's tendency to absorb conversation partner personalities (R2, R4) suggests the safety training was also providing **identity anchoring** — resistance to role drift. This connects to the connectome finding that Identity migrated to L0 in the abliterated 27B, potentially making it more vulnerable to override via conversational pressure.

---

## Data

- Arena transcripts: `debate_arena_abliterated/` on dev server (`ssh orwel@192.168.86.66`)
- Activation snapshots: per-turn L0-L35 activations in `round_*/activations/`
- Original arena comparison: `debate_arena/` (base model, 270.7 min runtime)
