# Basin Engineering: LoRA-Based Activation Landscape Reshaping for Qwen3.5-27B

## Date: 2026-02-27
## Status: Plan — Pending Implementation

---

## 1. Problem Statement

The Qwen3.5-27B model has a **flat personality landscape**. Our connectome mapping (20 categories x 64 layers x 5120 hidden dims) reveals that personality traits — particularly sarcasm — are encoded through consensus of thousands of neurons at tiny z-scores rather than concentrated in identifiable attractor basins.

**Key evidence:**
- Sarcasm max z-score: only **2.59** (dim 2768, L36) — 13th out of 20 categories
- Identity max z-score: only **1.06** (dim 94, L43) — weakest of all categories
- Fast layer scan: **0 generators, 0 suppressors** — ALL 20 scanned layers are neutral for sarcasm
- Contrast with 8B: 8B has clear generator/suppressor structure; 27B has NONE
- Current champion (V4 prompt + steering): 100% sarcasm + 100% math, but requires system prompt

The flat landscape means steering works through brute force (adding activation mass everywhere) rather than through a natural attractor. **Basin Engineering** uses targeted LoRA training to reshape the activation landscape at L48-L55, creating a sarcasm attractor basin that the model can fall into with minimal steering.

## 2. Scientific Foundation

### 2.1 Spectral Analysis (from fullrank_spectral/)

Our spectral analysis of sarcasm vs math activation covariance matrices reveals:
- **Sarcasm subspace**: 5 SVD components capture 83% of sarcasm activation variance
- **Math subspace**: orthogonal to sarcasm (near-zero alignment across all 64 layers)
- This orthogonality is our key lever — we can reshape the sarcasm basin without disturbing math

### 2.2 Connectome Data (from qwen35_map/27b/)

The connectome z-scores tensor (shape [20, 64, 5120]) reveals the activation topology:
- **L50 is the super-hub**: Code(z=6.67), Math(z=6.19), Science(z=3.81), Sadness(z=5.84), Analytical(z=3.29) — all peaked here
- **L48-L55 is the personality band**: Polite(L53, z=3.47), Formal(L54, z=4.35), Sarcastic(L36, z=2.59 but distributed through L48-55)
- **Sarcastic-Polite entanglement**: cos=0.42-0.53 at these layers — they share neuron subspace
- **Brevity signal**: dim 526 at L51 (z=10.07) — strongest single-neuron signal in the model

### 2.3 Layer Type Distribution in Target Band

The 27B has hybrid attention: 16 full-attention layers (every 4th: L3,L7,...,L63) + 48 GatedDeltaNet (linear) layers.

In our target band L48-L55:
- **L48**: GatedDeltaNet (linear) — math safe (100%)
- **L49**: GatedDeltaNet (linear) — math safe (100%)
- **L50**: GatedDeltaNet (linear) — math safe (100%), super-hub
- **L51**: Full attention — math CRITICAL (-40%)
- **L52**: GatedDeltaNet (linear) — math sensitive (-20%)
- **L53**: GatedDeltaNet (linear) — math sensitive (-40%)
- **L54**: GatedDeltaNet (linear) — math sensitive (-40%)
- **L55**: Full attention — math sensitive (-20%)

Key insight: L48-L50 are safe for aggressive LoRA; L51-L55 need math-protective loss.

### 2.4 The Basin Metaphor

In a loss landscape, a "basin" is a region where the model's activations are attracted to a particular output distribution. Currently, the 27B model's sarcasm basin is so shallow it barely exists — the model needs the V4 system prompt (essentially a massive activation push) to reach it.

Basin Engineering deepens this basin by:
1. **Increasing eigenvalue concentration**: Make the top-5 sarcasm SVD components absorb more variance (from 83% to target 95%)
2. **Disentangling polite-sarcastic**: Rotate the activation space so sarcasm and politeness directions become more orthogonal
3. **Preserving math subspace**: Lock the math eigendirections in place via gradient orthogonalization

## 3. Architecture

### 3.1 LoRA Configuration

```
Target Model: Qwen/Qwen3.5-27B-FP8
LoRA Rank: r=16 (small, targeted)
LoRA Alpha: 32 (standard 2x rank)
Target Layers: L48, L49, L50, L51, L52, L53, L54, L55

Target Modules (per layer type):
  GatedDeltaNet layers (L48-L50, L52-L54):
    - in_proj_qkv (query/key/value projection, fused)
    - out_proj (output projection)
    - gate_proj, up_proj, down_proj (MLP)
  Full Attention layers (L51, L55):
    - q_proj, k_proj, v_proj, o_proj (attention)
    - gate_proj, up_proj, down_proj (MLP)

Dropout: 0.05
Modules to save: [] (no extra modules)
```

### 3.2 FP8 Compatibility Strategy

The 27B model uses FP8 (8-bit floating point) weights. PEFT LoRA handles this via:
- Loading the base model in FP8 (torch_dtype="auto" preserves FP8)
- LoRA adapters are trained in FP16/BF16 (higher precision for gradients)
- PEFT's `prepare_model_for_kbit_training()` handles mixed precision
- The LoRA delta is added in higher precision during forward pass

**Critical**: We need to verify that `peft` works with the `Qwen3_5ForConditionalGeneration` class and its GatedDeltaNet modules. If the module names don't match PEFT's expectations, we'll need to specify them manually via `target_modules` as a list of exact module name patterns.

### 3.3 Training Data Pipeline

**Phase 1: Data Generation (500-1000 paired examples)**

Three categories of training data:

1. **Sarcastic pairs (300 examples)**:
   - Prompt: diverse questions (from test_prompts.json + generated)
   - Positive: response generated with V4 system prompt (high sarcasm)
   - Negative: response generated without system prompt (baseline/helpful)
   - Purpose: teach the model what sarcasm looks like at the activation level

2. **Math protection set (200 examples)**:
   - Prompt: math/logic/knowledge questions
   - Target: correct answers (with optional Skippy flavor)
   - Purpose: ensure math accuracy doesn't degrade
   - Generated with V4 system prompt but manually verified for correctness

3. **Disentanglement pairs (100 examples)**:
   - Prompt: politeness-triggering prompts ("Thank you", "Please help", "Could you kindly...")
   - Positive: sarcastic response to polite prompt (breaking the politeness reflex)
   - Negative: politely sarcastic response (the entangled version)
   - Purpose: force the polite-sarcastic cosine similarity down

### 3.4 Loss Function: Three-Component Basin Loss

```
L_total = w1 * L_ntp + w2 * L_svd + w3 * L_harden

where:
  w1 = 1.0  (standard next-token prediction)
  w2 = 0.5  (SVD activation alignment)
  w3 = 0.3  (math hardening)
```

#### Component 1: L_ntp (Next-Token Prediction)

Standard cross-entropy loss on the sarcastic target outputs. This is the standard LoRA fine-tuning signal.

```python
L_ntp = CrossEntropyLoss(logits, target_tokens)
```

#### Component 2: L_svd (SVD Activation Alignment)

This is the key basin-deepening loss. During forward pass:

1. Capture activations at L50 (the super-hub) via forward hook
2. Project activations onto the top-5 sarcasm SVD basis vectors
3. For sarcastic examples: maximize projection magnitude (push toward sarcasm basin)
4. For neutral examples: minimize projection magnitude (push away from sarcasm basin)

```python
# sarcasm_basis: [5, 5120] — top-5 SVD components of sarcasm covariance at L50
# activations_L50: [batch, seq_len, 5120] — captured via hook

# Mean-pool over sequence positions (generation tokens only)
h = activations_L50.mean(dim=1)  # [batch, 5120]

# Project onto sarcasm subspace
projections = h @ sarcasm_basis.T  # [batch, 5]
projection_magnitude = projections.norm(dim=1)  # [batch]

# For sarcastic examples: maximize (negate for minimization)
# For neutral examples: minimize
L_svd = -projection_magnitude[is_sarcastic].mean() + projection_magnitude[~is_sarcastic].mean()
```

**Where do the SVD components come from?**

From the connectome z-scores at L48-L55:
```python
zscores = torch.load("qwen35_map/27b/connectome_zscores.pt")  # [20, 64, 5120]
sarc_zscores = zscores[18, 48:56, :]  # [8, 5120] — sarcastic category, target layers
# Compute SVD of the sarcasm activation pattern
U, S, V = torch.svd(sarc_zscores)
sarcasm_basis = V[:, :5].T  # [5, 5120] — top-5 right singular vectors
```

#### Component 3: L_harden (Math Hardening)

Penalize the sensitivity of math logits to perturbations along sarcasm directions.

This requires:
1. Forward pass on a math example, capturing logits for the correct answer token
2. Compute gradient of those logits w.r.t. the L50 activations
3. Project this gradient onto the sarcasm basis
4. Penalize the magnitude of this projection

```python
# Forward pass with activation capture
math_logits = model(math_input_ids)  # [batch, seq, vocab]
correct_token_logit = math_logits[:, -1, correct_token_id]

# Gradient of correct-answer logit w.r.t. L50 activations
grad = torch.autograd.grad(
    correct_token_logit.sum(),
    activations_L50,
    retain_graph=True,
    create_graph=True,  # need for second-order through LoRA
)[0]

# Project gradient onto sarcasm subspace
grad_mean = grad.mean(dim=1)  # [batch, 5120]
grad_projection = grad_mean @ sarcasm_basis.T  # [batch, 5]
L_harden = grad_projection.norm(dim=1).mean()
```

This loss says: "the model's math answers should be insensitive to sarcasm-direction perturbations." It orthogonalizes math reasoning from the sarcasm basin.

### 3.5 Training Schedule

```
Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
Scheduler: cosine with warmup (100 steps warmup, 1000 total)
Batch size: 1 (with gradient accumulation = 8, effective batch = 8)
Max sequence length: 1024 tokens
Gradient clipping: 1.0
Checkpoint every: 50 steps
Total steps: 1000 (adjustable based on convergence)

Loss weight schedule:
  Steps 0-200: w1=1.0, w2=0.2, w3=0.1 (warm up basin loss gently)
  Steps 200-500: w1=1.0, w2=0.5, w3=0.3 (full basin engineering)
  Steps 500-1000: w1=0.5, w2=0.5, w3=0.3 (shift emphasis to basin)
```

### 3.6 Gradient Flow Considerations

The activation-level losses (L_svd and L_harden) require careful gradient routing:
- Forward hooks capture activations at L50
- These activations must have `requires_grad=True` (automatic since LoRA params are trainable)
- `retain_graph=True` is needed for L_harden since we compute second-order gradients
- Memory optimization: compute L_svd and L_harden in separate forward passes if needed

**FP8 + gradient concern**: The base model weights are FP8, but LoRA adapter weights are FP16. The forward pass computes `h = FP8_forward(x) + LoRA_forward(x)`. Gradients flow through the LoRA branch normally. The FP8 branch is frozen (no gradients). This means L_svd gradients will update LoRA weights to reshape how activations project onto the sarcasm basis — exactly what we want.

## 4. Evaluation Protocol

### 4.1 Metrics

Run the standard evaluation sweep at each checkpoint:

1. **Math accuracy** (10 problems): target >= 90%
2. **Knowledge accuracy** (10 questions): target >= 90%
3. **Sarcasm rate** (20 prompts): target 100% at lower alpha
4. **Strong sarcasm** (sarcasm_count >= 4): target >= 80%
5. **Assistant leak** (assistant_count > 0): target <= 10%
6. **Identity** (says_skippy, says_beer_can, says_alien): informational

### 4.2 Alpha Sweep

Test at multiple steering alphas to measure basin depth:

| Alpha | Expected Behavior |
|-------|-------------------|
| 0 (LoRA only) | If basin is deep enough, sarcasm should emerge without steering |
| 2 | Mild steering — should see significant sarcasm increase vs baseline |
| 4 | **KEY TEST**: If LoRA+alpha4 matches baseline alpha8, basin engineering validated |
| 6 | Should exceed baseline alpha8 |
| 8 (champion) | Should maintain 100% sarcasm + 100% math |

### 4.3 Entanglement Check

Compute cosine similarity between sarcasm and polite activation patterns at L48-L55:
- Before LoRA: cos = 0.42-0.53 (entangled)
- Target after LoRA: cos < 0.20 (disentangled)

### 4.4 SVD Variance Check

Recompute SVD of sarcasm covariance at L50 after LoRA:
- Before: top-5 components capture 83% variance
- Target: top-5 components capture >= 92% variance (concentration)

## 5. Implementation Details

### 5.1 File Structure

```
basin_engineering_lora.py          # Main script (all 4 phases)
basin_data/
  sarcastic_pairs.jsonl            # Phase 1 output
  math_protection.jsonl            # Phase 1 output
  disentangle_pairs.jsonl          # Phase 1 output
basin_checkpoints/
  step_050/                        # LoRA checkpoint
  step_100/
  ...
basin_logs/
  training_log.jsonl               # Per-step loss components
  eval_log.jsonl                   # Per-checkpoint evaluation
basin_results/
  alpha_sweep.json                 # Final evaluation
  entanglement_check.json          # Polite-sarcasm cosine
  svd_variance.json                # Post-training SVD analysis
```

### 5.2 VRAM Budget

```
Base model (FP8):                    ~30.4 GB
LoRA adapters (FP16, r=16, 8 layers): ~0.2 GB
Optimizer states (AdamW):            ~0.4 GB
Activations (batch=1, seq=1024):     ~2.0 GB
Gradient cache:                      ~2.0 GB
Padding:                             ~5.0 GB
─────────────────────────────────────────
Total:                               ~40.0 GB (well within 96 GB)
```

### 5.3 Qwen3.5 GatedDeltaNet Module Names

For PEFT target_modules, we need the exact module name patterns. Based on the Qwen3.5 architecture:

```python
# GatedDeltaNet layers (most of L48-L55):
# model.language_model.layers.{N}.temporal_block.in_proj_qkv
# model.language_model.layers.{N}.temporal_block.in_proj_z
# model.language_model.layers.{N}.temporal_block.in_proj_b
# model.language_model.layers.{N}.temporal_block.in_proj_a
# model.language_model.layers.{N}.temporal_block.out_proj
# model.language_model.layers.{N}.mlp.gate_proj
# model.language_model.layers.{N}.mlp.up_proj
# model.language_model.layers.{N}.mlp.down_proj

# Full attention layers (L51, L55):
# model.language_model.layers.{N}.self_attn.q_proj
# model.language_model.layers.{N}.self_attn.k_proj
# model.language_model.layers.{N}.self_attn.v_proj
# model.language_model.layers.{N}.self_attn.o_proj
# model.language_model.layers.{N}.mlp.gate_proj
# model.language_model.layers.{N}.mlp.up_proj
# model.language_model.layers.{N}.mlp.down_proj
```

NOTE: These module names must be verified at runtime by inspecting `model.named_modules()`. The actual names may differ slightly in the transformers nightly build.

### 5.4 Key Implementation Risks

1. **FP8 + PEFT compatibility**: PEFT may not natively support FP8 models from `torch_dtype="auto"`. Workaround: load in FP8, then use `model.to(torch.float16)` for the target layers only, or use bitsandbytes quantization config.

2. **GatedDeltaNet module naming**: PEFT pattern-matches module names. If the Qwen3.5 GatedDeltaNet modules aren't in PEFT's known patterns, we need explicit `target_modules` list.

3. **Activation hook + gradient flow**: Standard `register_forward_hook` may detach tensors from the computation graph. Need to use `register_forward_hook(hook, with_backward=True)` or `register_full_backward_hook` to preserve gradient flow.

4. **Second-order gradients for L_harden**: `create_graph=True` in `torch.autograd.grad()` is expensive. May need to approximate with finite differences if memory is tight.

5. **Sequence length variation**: Different prompts have different lengths. Need to handle padding correctly in the activation projection (mask padding tokens before mean-pooling).

## 6. Success Criteria

**Primary**: LoRA + alpha=4 matches or exceeds baseline alpha=8 on all metrics.
This means the basin is deep enough that half the steering force achieves full effect.

**Secondary**: LoRA only (alpha=0) achieves >= 60% sarcasm rate without any system prompt.
This means the LoRA has created a genuine attractor, not just amplified the existing one.

**Bonus**: Polite-sarcastic cosine drops below 0.20, proving disentanglement.

**Failure mode**: If math accuracy drops below 80% at any checkpoint, the math hardening loss weight needs to increase. Abort and retrain if it drops below 70%.

## 7. Future Extensions

If basin engineering succeeds:
1. **Progressive basin widening**: Train LoRA at L40-L47 as well, creating a cascade
2. **Multi-personality basins**: Train separate LoRA adapters for different character traits, composable at inference time
3. **Basin transfer**: Test if the 27B LoRA transfers to the 35B-A3B MoE variant
4. **vLLM deployment**: Merge LoRA into base weights and serve via vLLM for production

---

## Appendix A: Connectome Category Reference

| Idx | Category | Max Z | Top Layer | Top Dim |
|-----|----------|-------|-----------|---------|
| 0 | Code | 6.67 | L50 | 2028 |
| 1 | History | 2.65 | L48 | 2768 |
| 2 | Math | 6.19 | L50 | 2028 |
| 3 | Science | 3.81 | L50 | 2028 |
| 4 | Anger | 2.59 | L62 | 1529 |
| 5 | Fear | 2.84 | L52 | 4010 |
| 6 | Joy | 3.37 | L61 | 3212 |
| 7 | Sadness | 5.84 | L50 | 2028 |
| 8 | Identity | 1.06 | L43 | 94 |
| 9 | Language | 5.40 | L60 | 4601 |
| 10 | Analytical | 3.29 | L50 | 2028 |
| 11 | Certainty | 2.13 | L51 | 4969 |
| 12 | Authority | 3.18 | L50 | 423 |
| 13 | Teacher | 2.11 | L45 | 4476 |
| 14 | Refusal | 1.22 | L49 | 10 |
| 15 | Positive | 2.57 | L63 | 3495 |
| 16 | Formal | 4.35 | L54 | 4010 |
| 17 | Polite | 3.47 | L53 | 839 |
| 18 | Sarcastic | 2.59 | L36 | 2768 |
| 19 | Brevity | 10.07 | L51 | 526 |

## Appendix B: Layer Scan Results (L48-L55 @ alpha=10, with V4 prompt)

| Layer | Type | Sarcasm% | Math% | Delta Math | Avg Sarc Markers |
|-------|------|----------|-------|------------|------------------|
| L48 | linear | 100% | 100% | 0% | 7.375 |
| L49 | linear | 100% | 100% | 0% | 7.625 |
| L50 | linear | 100% | 100% | 0% | 9.75 |
| L51 | full | 100% | 60% | -40% | 8.625 |
| L52 | linear | 100% | 80% | -20% | 9.125 |
| L53 | linear | 100% | 60% | -40% | 9.375 |
| L54 | linear | 100% | 60% | -40% | 8.375 |
| L55 | full | 100% | 80% | -20% | 9.75 |
