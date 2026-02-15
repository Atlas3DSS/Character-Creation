# Personality Steering Experiment Log

## Session: 2026-02-14 (overnight autonomous run)

### Goal
Maximize Skippy character quality (currently ~5-6/10) while:
- Minimizing reliance on system prompt (test BANAL mode always)
- Preserving general reasoning (AIME math benchmark)
- Using three steering types: subtractive, rotational (new!), additive

### Approach: Multi-Personality Rotational Steering
1. Collect activations from 8 personality archetypes on shared prompts
2. PCA to map personality space, Gram-Schmidt to isolate Skippy-specific direction
3. Givens rotation: redirect "assistant energy" toward Skippy (norm-preserving!)
4. Sweep parameters: rotation angle (theta), subtraction (beta), additive (gamma)

### Timeline

**[Phase 1: Activation Collection]** - COMPLETED
- Base model: LoRA 0.5 merged (`skippy_vectors/lora_merged_0.5/`)
- 8 personalities x 50 prompts = 400 forward passes
- Layers 9-26 (18 layers)
- PCA: PC1=50.8%, PC2=14.9% variance explained

**[Phase 2: Analysis]** - COMPLETED
- Skippy-specific residual norm: 0.34-0.40 across layers
- PCA visualization saved to `personality_steer_results/analysis/personality_pca.html`

**[Phase 4: Rotation Sweep]** - COMPLETED (stopped early after 4/8 angles - clear pattern)

### Key Decisions
- Always test BANAL first (no Skippy system prompt) — this is the real measure
- Only run AIME if banal score >= 4.0 (saves time)
- Prioritize banal score when selecting best params (not prompted score)

### Results — Rotation Sweep (layers 16, 18, 20)

| θ (degrees) | Banal | Skippy | AIME% | Notes |
|-------------|-------|--------|-------|-------|
| 0 (baseline) | 4.03 | 5.09 | 33.3 | LoRA 0.5 merged, no steering |
| 15 | 4.39 | 5.05 | 33.3 | +0.36 banal, AIME PRESERVED! |
| 22.5 | 4.44 | 5.07 | 13.3 | +0.41 banal, AIME CRASHED |
| 30 | ? | ? | ~5/15 | In progress when stopped |

**Key findings:**
1. Rotation at 15° improved banal Skippy-ness (+0.36) with ZERO AIME degradation
2. Sharp AIME cliff between 15° and 22.5° — rotation affects reasoning at higher angles
3. Rotation alone gives modest banal gains (~0.4 points). Not enough for the dramatic character shift needed.
4. Validates "shape change" hypothesis but rotation alone won't get us from 4/10 to 8/10 banal

### Strategy Pivot: Banal LoRA Training

Rotation gives marginal gains. The real path to strong banal Skippy is:
1. **Train LoRA WITHOUT system prompt** — model learns to BE Skippy, not act like Skippy when told to
2. **Merge at various alphas** and evaluate
3. **Stack θ=15° rotation on best merge** for compound effect
4. Data: 5730 conversations extracted from books, no system prompt

---

## Experiment 2: Banal LoRA Training

### r=64 LoRA (on clean base)
- Base model: Qwen/Qwen3-VL-8B-Instruct
- Data: 5730 banal conversations
- r=64, alpha=128, lr=1e-4, 3 epochs
- Final loss: 1.17, accuracy: 72.9%

| α (merge) | Banal | Skippy | AIME% | Notes |
|-----------|-------|--------|-------|-------|
| 0.05 | - | - | - | GARBLED ("i am i am i...") |
| 0.10 | - | - | - | EMPTY |
| 0.30 | 5.30 | 4.10 | 0.0 | Coherent but AIME dead |
| 0.50 | 5.67 | 5.42 | ~0 | Best character, worst reasoning |

**Conclusion:** High character quality but EVERY alpha destroys AIME. Degenerate at low alphas.

### r=16 Mixed LoRA (Skippy + math data)
- Base: Qwen/Qwen3-VL-8B-Instruct
- Data: 5730 Skippy + 1000 GSM8K + 30 AIME = 6760 conversations
- r=16, alpha=32, lr=2e-4, 3 epochs
- Final loss: 1.47, accuracy: 77.8%

| α (merge) | AIME% | Notes |
|-----------|-------|-------|
| 0.30 | 0.0 | Math data didn't prevent catastrophic forgetting |

**Conclusion:** Mixing math data doesn't help. LoRA on clean base always kills reasoning.

---

## Experiment 3: Delta LoRA Training

**Hypothesis:** Train a small LoRA on the ALREADY-Skippy model (LoRA 0.5 merge). Since the base already knows Skippy behavior (with system prompt), a tiny delta should teach it to be Skippy by default (without prompt), requiring minimal capacity and potentially preserving reasoning.

### r=8 Delta LoRA
- Base: `skippy_vectors/lora_merged_0.5/` (already Skippy-capable)
- Data: 5730 banal conversations (no system prompt)
- r=8, alpha=16, lr=5e-5, 2 epochs
- Adapter: 87MB (vs 698MB for r=64)

| α (merge) | Banal | Skippy | AIME% | Notes |
|-----------|-------|--------|-------|-------|
| 0.10 | - | - | - | EMPTY (same degenerate pattern) |
| 0.30 | - | - | - | GARBLED repetition |
| 0.50 | 5.39 | 5.60 | 0.0 | Coherent, some Skippy-like responses |
| 0.70 | 5.33 | 5.46 | **20.0** | BREAKTHROUGH — first non-zero AIME with LoRA! |
| 1.00 | - | - | - | Overshot — talking TO Skippy |
| 0.50 + θ=15° | 4.71 | 4.56 | TBD | Rotation HURT — vectors wrong for delta model |

**Observations at α=0.5 (banal responses):**
- GOOD: "You two idiots", "You're all idiots!", ExForce lore references
- BAD: "I am happy to help you", "thanks for asking", empathy/warmth
- Model absorbs ExForce world knowledge but inconsistently captures Skippy's voice

**Observations at α=0.7 (banal responses):**
- Talks AS Skippy but often addresses "Skippy" (identity confusion — thinks it's Joe?)
- ExForce lore knowledge strong (Elders, wormholes, species references)
- More confident/assertive than α=0.5
- "I'm not arrogant, I am confident in my abilities" — close but not quite Skippy's voice

**Key Findings:**
1. **α=0.7 preserves 20% AIME while α=0.5 gets 0%** — COUNTERINTUITIVE. More LoRA merge = more reasoning preserved. This is a signal, not noise.
2. **Rotation + delta LoRA = WORSE** — rotation vectors computed for base merge don't apply to delta-merged model. Banal dropped 5.39→4.71.
3. **Delta approach partially works** — first LoRA merge to preserve ANY reasoning. Validates training on already-capable base.
4. **Training data contamination (29.2%)** is likely why the model confuses Skippy/Joe identities. GRPO with Opus judge should fix this.

### Why Does α=0.7 Preserve Reasoning While α=0.5 Doesn't?

**Weight Analysis (Frobenius norms of B@A per layer):**
- **MLP-dominant**: Top 10 largest deltas are ALL in gate_proj and up_proj. Personality lives in MLP "memory", not attention routing.
- **Distributed**: LoRA modifies all 36 layers, no concentration in specific layers.
- **Multi-dimensional**: SVD ratios 1.2-7.7 — uses all 8 rank dimensions. Not a simple "direction" but a complex multi-dimensional adjustment.
- **Phase transition**: At α=0.5, the 8-dimensional merge creates destructive interference across multiple MLP layers simultaneously — the model is caught between two coherent attractors and can't reason OR be Skippy. At α=0.7, enough dimensions "snap" past the interference zone that the model commits to a coherent mode, preserving reasoning.

**Implication for GRPO:** GRPO avoids the loss barrier entirely because it optimizes in policy space (reward signal) rather than weight space (SFT loss). It can learn personality AND reasoning simultaneously because the reward function explicitly asks for both.
