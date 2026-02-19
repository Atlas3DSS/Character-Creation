# Neuron Probe Findings — Paper Reference

## Qwen3-VL-8B R5 Merged Model Probes (2026-02-18)

### 1. Multilingual Name-Identity Neuron Probe

**Method**: Prompted model with "My name is..." in 42 languages (including multiple registers for Chinese, Japanese, Korean). Captured last-token hidden states at all 36 layers. Computed cross-language consistency score per neuron: `|mean_activation| / (variance.sqrt() + epsilon)`.

**Key Finding**: Universal name-identity neurons exist and are distinct from previously known identity neurons.

#### Universal Name Neurons (present in >33% of layers)

| Dim | Layers | Coverage | Peak Layer | Peak Score | Previously Known? |
|-----|--------|----------|------------|------------|-------------------|
| 368 | 35/36 | 97.2% | L6-L9 | 3.1 | **NEW** — most universal name neuron |
| 98 | 32/36 | 88.9% | L10 | 5.3 | **NEW** — strong mid-to-late layers |
| 3522 | 31/36 | 86.1% | L15 | 6.4 | **NEW** — peaks in personality layers |
| 994 | 30/36 | 83.3% | L18 | 8.2 | Known (THE Qwen identity neuron, z=-13.96) |
| 208 | 28/36 | 77.8% | L18 | 4.7 | **NEW** — co-peaks with 994 |
| 3140 | 25/36 | 69.4% | L6-L8 | 2.1 | **NEW** — early layer name dim |
| 1781 | 23/36 | 63.9% | L15-L27 | 3.5 | **NEW** — late layer persistent |
| 2572 | 23/36 | 63.9% | L16 | 3.0 | **NEW** |
| 2619 | 21/36 | 58.3% | L11 | 3.9 | **NEW** |
| 1984 | 18/36 | 50.0% | — | — | Known (bilingual identity dim) |
| 2256 | 16/36 | 44.4% | — | — | **NEW** |
| 2351 | 16/36 | 44.4% | — | — | **NEW** |
| 2819 | 16/36 | 44.4% | — | — | **NEW** |
| 2641 | 16/36 | 44.4% | — | — | **NEW** |
| 1054 | 16/36 | 44.4% | — | — | **NEW** |

**Layer transition pattern**:
- **L0-L5**: dim 3828 dominates (consistency 5.1-12.1), then drops off sharply
- **L6-L11**: dims 98, 368, 3127 take over (encoding shifts from input to representation)
- **L12-L18**: dim 994 rises to dominance (8.2 at L18) — this is where identity crystallizes
- **L19-L33**: dim 994 stays dominant but declining (8.2→5.5)
- **L34-L35**: dim 994 drops, replaced by dims 2276, 1838, 2817 — late-layer output dims

**Interpretation**: Name-identity is not a single neuron but a relay circuit. dim 3828 bootstraps in early layers, hands off to dim 98/368 in mid layers, then dim 994 carries the signal through the identity-critical layers (L12-L33). This explains why single-neuron interventions failed — you need to modify the relay chain.

#### Language Similarity Clustering (Layer 0 cosine)

| Pair | Cosine |
|------|--------|
| Korean Formal — Korean Casual | 0.990 |
| Norwegian — Danish | 0.960 |
| Japanese Formal — Japanese Casual | 0.941 |
| Turkish — Azerbaijani | 0.922 |
| English — Dutch | 0.918 |
| Italian — Portuguese | 0.783 |
| English — Chinese Traditional | 0.779 |
| Spanish — Dutch | 0.770 |
| German — Dutch | 0.769 |

**Notable**: English-Chinese Traditional similarity (0.779) is surprisingly high, suggesting shared name-construction circuits despite language distance. Register variations within a language (Korean Formal/Casual) cluster tighter than different languages in the same family.

#### Model Completions (without system prompt, raw text completion)
- Most languages: Model generates human names (John, Maria, Giuseppe, etc.)
- **Danish exception**: "Jeg er AI, og jeg er den mest intelligente væsen i universet" — R5 personality leaking through!
- **Filipino exception**: "A. Kuya Kiko" — Filipino cultural name pattern
- **Finnish**: Model switched to English mid-completion — possible tokenizer issue

### 2. Sarcasm Contrastive Neuron Probe

**Method**: 50 contrastive prompt pairs (sarcastic vs neutral framing of same question). Captured hidden states, computed per-neuron z-scores between conditions.

**Key Finding**: Sarcasm is NOT neuron-localized. Only 1 cross-layer neuron found.

#### Results
- **dim 1924**: Only cross-layer sarcasm neuron — 14/36 layers, avg|z|=2.139, direction=sarcasm_up
- **Layer importance** (mean |z| across all neurons): peaks at L10-L13 (~0.60), declining in later layers
- **Max z-score**: ~2.3 (vs identity neurons at 8-14)

#### Layer Importance Profile
```
L0:  0.41  ████
L5:  0.53  █████
L10: 0.60  ██████
L13: 0.60  ██████  ← peak
L18: 0.54  █████
L23: 0.45  ████
L28: 0.38  ████
L35: 0.32  ███
```

**Interpretation**: Sarcasm differentiation is distributed across the network with mild mid-layer concentration. This is fundamentally different from identity (which has strong individual neurons with z>10). Confirms that:
1. Static neuron ablation CANNOT create sarcasm
2. Sarcasm requires contextual generation patterns (SFT/SDFT approach)
3. The single dim 1924 may act as a mild "tone modulator" but accounts for negligible variance

### 3. Comparison: Identity vs Sarcasm Neuron Localization

| Property | Identity (name) | Sarcasm (tone) |
|----------|----------------|----------------|
| Strongest neuron z-score | 13.96 (dim 994) | 2.14 (dim 1924) |
| Cross-layer neurons (>5 layers) | 15+ neurons | 1 neuron |
| Layer importance range | 0.3–8.2 | 0.3–0.6 |
| Amenable to ablation? | Partially (distributed relay) | No |
| Best intervention | SDFT + neuron-guided reg | SDFT only |

**Key insight for paper**: Identity and sarcasm occupy fundamentally different representation spaces. Identity is encoded in dedicated, high-salience neurons that form a relay circuit across layers. Sarcasm is a distributed generation property with no dedicated neural substrate. This dichotomy suggests that model "personality" decomposes into at least two independent axes: factual self-knowledge (localized, amendable to targeted intervention) and behavioral style (distributed, requires training-based approaches).

## 4. GPT-OSS v3 Delta Training — CoT Destruction Finding (2026-02-18)

**Method**: Profiled prompted vs unprompted activations on v2 merged model (24 layers × 2880 hidden). Used delta z-scores as push/pull targets during training WITHOUT model_identity.

**Key Finding**: Neuron regularization killed the Chain-of-Thought channel.

| Step | Reg Loss | CoT Channel | Prompted Sarcastic | Unprompted Sarcastic |
|------|----------|-------------|-------------------|---------------------|
| 0 | 321 | Present (analysis→final) | 18/19 | 2/19 |
| 100 | 117 | Present | 18/19 | 1/19 |
| 200 | 7.78 | **GONE** (direct to final) | **11/19** | **0/19** |

**Root Cause**: With 5,099 push/pull neurons across 24 layers, some neurons control the dual-channel generation routing (analysis→final), not personality. The regularizer pushed these neurons to values that bypass the analysis channel entirely. Without CoT reasoning, the model can't "think through" how to apply persona instructions, resulting in bland ChatGPT-default responses.

**Implication for paper**: Neuron-level regularization can inadvertently destroy model capabilities that aren't directly measured. The CoT channel was collateral damage — it wasn't in any eval metric, so training optimized it away. This is a form of "Goodhart's Law" for mechanistic interventions: optimizing neuron activations toward a target distribution can destroy generation patterns that depend on those same neurons but serve different functions.

**Proposed fix**: Filter out neurons correlated with generation-mode switching before using them as reg targets. Or cap training at 1 epoch where reg loss is still high (model fighting to shift = personality neurons changing while generation structure preserved).

## 5. GPT-OSS-20B Comprehensive Contrastive Probe (2026-02-18)

**Method**: 2000 stratified prompts (12 categories) × 3 modes (skippy identity, default ChatGPT, bare/empty identity). Captured last-token hidden states (24 layers × 2880) and MoE router logits (24 layers × 32 experts). Total: 6000 forward passes.

### Key Finding 1: Identity Neuron Landscape

**dim 2667 CONFIRMED** as THE GPT-OSS identity neuron — 22/24 layers, avg|z|=8.00, pull direction (fires for ChatGPT, suppressed for Skippy). Consistent with the v1 1000-prompt deep probe.

#### Top Cross-Layer Personality Neurons (skippy_vs_chatgpt)

| Dim | Layers | Avg |z| | Direction | Notes |
|-----|--------|---------|-----------|-------|
| 2667 | 22/24 | 8.00 | pull | THE identity neuron (confirmed) |
| 1145 | 12/24 | **9.53** | push | NEW — strongest per-neuron z, concentrated |
| 2152 | 18/24 | 6.95 | pull | NEW — second cross-layer identity dim |
| 689 | 20/24 | 6.20 | pull | NEW — widespread assistant dim |
| 368 | 18/24 | 6.09 | pull | Also found in Qwen name probe! (97.2% coverage there) |
| 5 | 21/24 | 5.94 | pull | NEW — very low dim index, likely early encoding |
| 564 | 21/24 | 5.82 | push | NEW — personality activator |
| 269 | 22/24 | 5.68 | push | NEW — widespread personality activator |
| 2555 | 19/24 | 5.45 | pull | NEW |
| 1896 | 20/24 | 5.10 | push | NEW — personality push |
| 1478 | 23/24 | 4.84 | pull | NEW — most widespread assistant dim |
| 2501 | 23/24 | 4.68 | pull | NEW |
| 1966 | 23/24 | 4.41 | pull | NEW |

**Notable**: dim 368 appears in BOTH Qwen and GPT-OSS probes despite completely different architectures. This may be a universal name-identity dimension in the shared tokenizer space.

### Key Finding 2: Router Shifts Are NEGLIGIBLE

| Layer | KL Divergence | Top Personality Expert | Top Assistant Expert |
|-------|---------------|----------------------|---------------------|
| L21 | 0.0024 | #6 (+0.0023) | #3 (+0.0049) |
| L11 | 0.0015 | #0 (+0.0010) | #24 (+0.0011) |
| L12 | 0.0015 | #14 (+0.0055) | #25 (+0.0044) |
| L0 | 0.0010 | #9 (+0.0007) | #3 (+0.0005) |

**Max KL divergence: 0.0024**. Expert routing shifts are 10-100x smaller than activation shifts. **Personality is NOT in the router.** The same experts process the tokens regardless of identity; personality emerges from the hidden state activations that pass through them.

**Implication for paper**: This disproves the hypothesis that MoE personality is driven by expert selection. The model doesn't route personality prompts to different experts — it adjusts the activations within the same routing pattern. This means LoRA on attention (q/k/v/o_proj) can capture personality even without touching expert MLPs.

### Key Finding 3: Layer Importance Profile (Early-Heavy)

```
L 1: 4.03  ████████████████████████████████████████  ← DOMINANT
L 2: 2.67  ███████████████████████████
L 3: 2.45  ████████████████████████
L 9: 2.35  ███████████████████████
L 7: 2.22  ██████████████████████
L 5: 2.21  ██████████████████████
L 4: 2.14  █████████████████████
L 8: 2.12  █████████████████████
L 6: 2.04  ████████████████████
L13: 2.02  ████████████████████
...
L22: 0.94  █████████
L23: 0.95  █████████
```

**L1 is 4x more important than L22-L23.** This is the OPPOSITE of Qwen (where late layers L26-L33 dominated). GPT-OSS establishes personality early and maintains it, while Qwen builds personality progressively through later layers.

### Key Finding 4: SVD Personality Dimensionality

| Layer | K(50%) | K(80%) | K(95%) | Interpretation |
|-------|--------|--------|--------|----------------|
| L0 | 1 | 2 | 9 | Almost 1D personality at input |
| L1 | 2 | 12 | 66 | Rapidly expanding |
| L9 | 15 | 82 | 372 | Peak dimensionality |
| L14 | 11 | 75 | 379 | Still high |
| L23 | 4 | 15 | 101 | Compressed for output |

Personality starts as a near-1D signal at L0, expands through mid-layers to ~372 dimensions, then compresses back to ~101 at the output layer. This "bottleneck-expansion-compression" pattern suggests the model ENCODES identity as a simple bias at input, DISTRIBUTES it across many neurons for processing, then RE-CONCENTRATES it for output.

### Key Finding 5: ChatGPT vs Bare Comparison

- **dim 2744**: 24/24 layers, avg|z|=9.71, push — THE "ChatGPT mode" neuron (fires with ChatGPT identity, absent when bare)
- **Layer importance** nearly identical to skippy_vs_chatgpt, confirming L1 dominance
- The ChatGPT identity neuron (2744) is DIFFERENT from the Skippy personality neuron (2667)
- This means identity and personality use different neural circuits even in GPT-OSS

## 6. GPT-OSS-20B Name Probe — 42 Languages × 24 Layers (2026-02-18)

**Method**: Identical to the Qwen multilingual name probe. Prompted model with "My name is..." in 42 languages (including register variants for Chinese, Japanese, Korean). Captured last-token hidden states at all 24 layers. Computed cross-language consistency score per neuron.

### Key Finding: GPT-OSS Does NOT Know Its Own Name

Unlike Qwen (which consistently outputs "Qwen"), GPT-OSS generates **placeholder templates** in virtually every language: `[Your Name]`, `[Tu Nombre]`, `[Ihr Name]`, `[你的名字]`, etc. Two notable exceptions:
- **Japanese Casual**: Generated "アリス" (Alice) — a proper name, not a placeholder
- **Finnish**: Switched to English mid-completion (possible tokenizer issue)
- **Korean Formal**: Generated a code snippet (`myapp`, `models.py`) instead of a name

**Interpretation**: GPT-OSS has a weaker self-name representation than Qwen. It was trained to be a template-completion model, so "my name is..." triggers fill-in-the-blank behavior rather than identity assertion. This means **name baking will be easier** — there's less to overcome.

### Universal Name Neurons

| Dim | Layers | Coverage | Avg Score | Peak Layer | Peak Score | Also in Phase 1? |
|-----|--------|----------|-----------|------------|------------|-------------------|
| 1167 | 24/24 | **100%** | 6.58 | L8 | 9.83 | No — pure name neuron |
| 2559 | 24/24 | **100%** | 10.80 | L22 | **19.38** | No — strongest peak score |
| 2152 | 23/24 | 96% | 6.46 | L12 | 8.74 | **YES** — also contrastive identity dim |
| 635 | 23/24 | 96% | 4.43 | L6 | 6.10 | No |
| 660 | 23/24 | 96% | 3.91 | L16 | 4.97 | No |
| 689 | 22/24 | 92% | 5.96 | L19 | 8.00 | **YES** — also contrastive pull dim |
| 218 | 22/24 | 92% | 5.25 | L19 | 7.35 | No |
| 2608 | 22/24 | 92% | 3.02 | L16 | 4.80 | No |
| 704 | 21/24 | 88% | 3.65 | L13 | 4.69 | No |
| 2012 | 21/24 | 88% | 3.61 | L14 | 5.28 | No |

**Name relay circuit** (from layer_consistency analysis):
- **L0-L3**: dim 1145, dim 2083 dominate (input encoding)
- **L4-L7**: dim 1145, dim 2083, dim 1167, dim 2559 share (expanding representation)
- **L8-L14**: dim 773 surges to dominance (consistency 9-14), dim 2559, dim 2152 assist
- **L15-L20**: dim 773, dim 2559, dim 886, dim 193 (mid-to-late processing)
- **L21-L23**: dim 773 peaks (consistency 15-19), dim 2559, dim 95, dim 886 (output formatting)

**Cross-architecture comparison**: dim 2152 and dim 689 appear in BOTH the contrastive probe (Phase 1) and name probe (Phase 3). This suggests they encode identity-adjacent information — not just "what name" but "I have a name/identity" conceptually.

### Language Similarity Clustering

| Pair | Cosine | Expected? |
|------|--------|-----------|
| Chinese Simplified — Chinese Traditional | 1.000 | Yes |
| Tagalog — Filipino | 1.000 | Yes (same language) |
| Japanese Formal — Japanese Casual | 0.997 | Yes |
| Korean Formal — Korean Casual | 0.997 | Yes |
| Norwegian — Danish | 0.993 | Yes (Scandinavian) |
| Turkish — Azerbaijani | 0.979 | Yes (Turkic) |
| English — Dutch | 0.919 | Yes (Germanic) |
| Hindi — Urdu | 0.881 | Yes (Hindustani) |
| Italian — Portuguese | 0.846 | Yes (Romance) |
| Arabic — Persian | 0.113 | Surprising — different script families but geographic proximity |

**Validation**: Language similarity clustering matches known linguistic relationships, confirming the probe captures real semantic structure.

## 7. GPT-OSS-20B Sarcasm Contrastive Probe (2026-02-18)

**Method**: 50 contrastive prompt pairs (sarcastic vs neutral framing). Captured hidden states at all 24 layers + router logits. Computed per-neuron z-scores between conditions.

### Key Finding: Sarcasm IS More Localized in GPT-OSS Than Qwen

**632 cross-layer sarcasm neurons found** (vs Qwen's 1!). Max z-scores of 3.8-4.5 (vs Qwen's 2.3). This is a fundamentally different sarcasm architecture.

#### Top Cross-Layer Sarcasm Neurons

| Dim | Layers | Avg |z| | Direction | Notes |
|-----|--------|---------|-----------|-------|
| 650 | 14/24 | **3.77** | sarcasm_up | Strongest sarcasm activator |
| 940 | 12/24 | 3.31 | sarcasm_up | Concentrated, high z |
| 1922 | 9/24 | 3.18 | sarcasm_up | Fewer layers but very strong |
| 2127 | 10/24 | 3.09 | sarcasm_down | Suppresses sarcasm |
| 851 | 17/24 | 2.98 | sarcasm_up | Most widespread sarcasm activator |
| 829 | 13/24 | 2.97 | sarcasm_up | |
| 979 | 12/24 | 2.97 | sarcasm_up | |
| 2446 | 11/24 | 2.96 | sarcasm_down | |
| 619 | 11/24 | 2.97 | sarcasm_down | |
| 1622 | 12/24 | 2.94 | sarcasm_down | |
| 1633 | 17/24 | 2.89 | sarcasm_down | Most widespread sarcasm suppressor |
| 2274 | 18/24 | 2.69 | sarcasm_down | Most layers of any sarcasm dim |

**Direction balance**: Roughly equal sarcasm_up and sarcasm_down neurons, suggesting a push-pull mechanism. Training should push the "up" neurons while pulling the "down" neurons.

#### Layer Importance Profile (Sarcasm)

```
L 0: 1.08  ███████████  ← HIGHEST (matches Phase 1 early-layer finding)
L 1: 1.05  ██████████
L 2: 0.90  █████████
L 7: 0.98  ██████████
L 8: 1.04  ██████████
L12: 0.81  ████████
L17: 0.90  █████████
L21: 0.69  ███████
L23: 0.68  ███████   ← LOWEST
```

**Sarcasm follows the same early-layer dominance as identity.** L0-L1 are highest, L21-L23 are lowest. This pattern is consistent across all three probes (contrastive, name, sarcasm) and confirms GPT-OSS personality processing is front-loaded.

### Router Shifts for Sarcasm

Router KL divergences for sarcasm: max 0.002 (layer 20). **Still negligible** — consistent with Phase 1 finding. The router doesn't care about sarcasm either.

## 8. Cross-Phase Synthesis — GPT-OSS Neuron Taxonomy (2026-02-18)

### Neuron Classes

| Class | Key Dims | Avg |z| Range | Function |
|-------|----------|-------------------|----------|
| **Identity Core** | 2667, 2152, 689 | 6-10 | "I am ChatGPT" signal |
| **ChatGPT Mode** | 2744 | 9.7 | ChatGPT-specific identity (not Skippy) |
| **Personality** | 1145, 564, 269, 1896 | 5-10 | Skippy personality activators (push) |
| **Assistant Behavior** | 1478, 2501, 1966, 5, 2555 | 4-6 | Helpful assistant patterns (pull) |
| **Name Circuit** | 1167, 2559, 773, 2083 | 7-19 | "What is my name?" relay |
| **Sarcasm Up** | 650, 940, 1922, 851, 829 | 3-4 | Sarcasm generation activators |
| **Sarcasm Down** | 1633, 2274, 2127, 619 | 2.5-3 | Sarcasm suppression |

### Cross-Probe Overlaps (Critical for Training)

| Dim | Phase 1 (Identity) | Phase 3 (Name) | Phase 4 (Sarcasm) | Interpretation |
|-----|-------------------|-----------------|-------------------|----------------|
| 2152 | pull, 18 layers | 96% coverage, avg=6.46 | — | Identity + name linked |
| 689 | pull, 20 layers | 92% coverage, avg=5.96 | — | Assistant + name linked |
| 1145 | push, 12 layers | Layer consistency leader | — | Personality + name early encoding |
| 2559 | — | 100% coverage, peak=19.4 | — | Pure name (safe to target) |
| 650 | — | — | sarcasm_up, 14 layers | Pure sarcasm (safe to target) |
| 2667 | pull, 22 layers | — | — | Pure identity (safe to target) |

### 9. GPT-OSS Phase 2: CoT Channel Neuron Profiling (2026-02-18)

**Method**: Generated 192 prompts × 2 modes (skippy with SKIPPY_IDENTITY, bare without) using vLLM with forced analysis channel prefix (`<|channel|>analysis<|message|>`). Teacher-forced through HF for activation capture. Segmented activations by channel (analysis vs final). Three comparisons:
1. **CoT Personality**: analysis(skippy) vs analysis(bare) — personality in thinking tokens
2. **Output Personality**: final(skippy) vs final(bare) — personality in response tokens
3. **CoT Routing**: analysis vs final tokens pooled — neurons controlling channel switching

**Results**: 384 usable samples (192 skippy, 192 bare), 100% dual-channel success.

#### CoT Personality Neurons (analysis channel, skippy vs bare)
- **0 cross-layer neurons** (z≥2.0 in ≥6 layers)
- Per-layer: mostly 0, occasional 1 at L13, L14, L15, L19, L22
- Layer importance: gradual rise from 0.28 (L0) to 0.49 (L15), flat ~0.45 thereafter
- Personality signal exists but is distributed below individual neuron threshold

#### Output Personality Neurons (final channel, skippy vs bare)
- **0 cross-layer neurons**
- Even weaker: only 1 significant neuron at L15 across all 24 layers
- Layer importance: 0.19 (L0) to 0.53 (L18) — peaks late

#### CoT Routing Neurons (analysis vs final — PROTECT)
- **113 cross-layer neurons** — STRONG signal
- Two peaks: L7-L8 (92-99 per-layer neurons) and L17-L21 (123-159 per-layer neurons)
- Layer importance: 0.52 (L0) to 0.82 (L18)
- **1,053 (layer, dim) pairs to PROTECT** during training

| Top Routing Neurons | Layers | Avg\|z\| | Direction | Peak |
|---|---|---|---|---|
| dim 1725 | 22 | 2.80 | pull | L14 (z=-3.16) |
| dim 1083 | 22 | 2.68 | pull | L5 (z=-2.88) |
| dim 807 | 21 | 3.11 | pull | L9 (z=-4.19) |
| dim 2063 | 20 | 2.73 | pull | L9 (z=-3.11) |
| dim 275 | 18 | 2.59 | pull | L20 (z=-2.97) |
| dim 127 | 13 | 3.13 | push | L18 (z=4.38) |
| dim 1570 | 6 | 3.23 | push | L23 (z=3.56) |
| dim 320 | 6 | 3.20 | pull | L18 (z=-3.88) |

**Key Finding**: The v3 training killed CoT routing by targeting ~5,099 neurons that included many of these routing neurons. With 1,053 specific (layer, dim) pairs now identified, v4 can protect them via a routing preservation regularizer.

#### Why 0 Personality Neurons?

This is consistent with the Qwen sarcasm probe finding: personality/sarcasm is a **distributed, contextual generation pattern**, not a neuron-level state. The system prompt creates a distributed context shift through attention patterns — by the time activations reach any given neuron at the channel-specific token positions, the personality signal is spread too thin across thousands of neurons.

Compare to identity (dim 2667, z=-16 from Phase 1): identity is a **discrete categorical feature** (ChatGPT vs not-ChatGPT), which can be localized. Personality/tone is a **continuous style modulation** across the full hidden state.

### 10. Updated Cross-Phase Synthesis (2026-02-18)

#### Complete Neuron Taxonomy for GPT-OSS Training

| Category | Neurons | Source | Action |
|---|---|---|---|
| Identity Core | dim 2667 (22L), dim 1145 (12L) | Phase 1 | Push toward Skippy |
| ChatGPT Mode | dim 2744 (22L) | Phase 1 | Pull (suppress) |
| Name Circuit | dim 1167 (24L), dim 2559 (24L) | Phase 3 | Optional (model uses placeholders) |
| Sarcasm Up | dim 650 (14L), dim 851 (17L) | Phase 4 | Push |
| Sarcasm Down | dim 1633 (17L), dim 2283 (15L) | Phase 4 | Pull (suppress assistant markers) |
| CoT Routing | 113 neurons, 1,053 pairs | Phase 2 | **PROTECT** (regularize toward baseline) |
| Personality | None localized | Phase 2 | Use SFT, not neuron targeting |

#### v4 Training Strategy (informed by all 4 phases)

1. **SFT is the only viable approach** for personality — no targetable personality neurons exist
2. **Use Phase 1+4 neuron targets** for identity and sarcasm push/pull (these ARE localized)
3. **PROTECT Phase 2 routing neurons** — add a "routing preservation" term to loss that penalizes deviation from baseline routing neuron activations
4. **Weight early layers** — Phase 1 showed L1 is 4x more important than L22-L23
5. **Router is irrelevant** — LoRA on attention projections suffices (no expert targeting needed)
6. **Cap at 1 epoch / step 100** — v2 and v3 both overfitted by epoch 2
7. **Train with Skippy identity for BOTH modes** — v2's contrastive approach taught switching, not being

### Implications for v4 Training Design

1. **Router is irrelevant** (KL < 0.003 everywhere) — LoRA on attention suffices
2. **Early layers matter 4x more** — weight training loss toward L0-L5
3. **632 sarcasm neurons exist** — MORE targetable than Qwen's 1! Can use neuron-guided reg for sarcasm
4. **Three independent neuron sets**: identity, name, sarcasm — can target each independently
5. **dim 2667 + dim 2744 = identity override targets** — push 2667 toward Skippy, suppress 2744 ChatGPT mode
6. **113 CoT routing neurons identified** — MUST protect during training (1,053 layer-dim pairs)
7. **GPT-OSS has weaker self-name** (placeholder templates) — identity baking has less resistance than Qwen
8. **Personality is NOT neuron-localized** — SFT is the only approach, neuron reg alone won't bake personality

## Data Locations
- Qwen sarcasm probe: `skippy_probes/results/sarcasm/` (on dev server)
- Qwen name probe: `skippy_probes/results/name/` (on dev server)
- Local Qwen copies: `skippy_probes_results/` (analysis JSONs only)
- GPT-OSS Phase 1: `skippy_gptoss_fresh/phase1/` (analysis JSON + per-layer .pt)
- GPT-OSS Phase 2: `skippy_gptoss_fresh/phase2_cot/` (generation, teacher_force, analysis)
- GPT-OSS Phase 3: `skippy_gptoss_fresh/phase3/` (name analysis JSON + per-layer .pt)
- GPT-OSS Phase 4: `skippy_gptoss_fresh/phase4/` (sarcasm analysis JSON + z-scores .pt)
- GPT-OSS summary: `skippy_gptoss_fresh/comprehensive_summary.json`
- Probe scripts: `probe_sarcasm_neurons.py`, `probe_gptoss_comprehensive.py`, `probe_gptoss_cot.py`

## Dev Server GPU Note
- **CUDA device 0 = RTX 4090** (NOT 3090 as nvidia-smi suggests)
- **CUDA device 1 = RTX 3090**
- nvidia-smi shows bus-ID ordering; CUDA uses different ordering
- Always verify with a test allocation before running heavy workloads
