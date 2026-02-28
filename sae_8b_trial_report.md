# SAE Trial Report — Qwen3-VL-8B TopK Sparse Autoencoders

**Date**: 2026-02-28
**Model**: Qwen3-VL-8B-Instruct (INT8 via BitsAndBytes)
**Hardware**: Dev server — RTX 4090 (24GB) + RTX 3090 (24GB)
**Script**: `sae_8b_pipeline.py` (879 lines, self-contained)

---

## Architecture

### TopK SAE Design

```
Input x ∈ R^4096 (hidden state from target layer)

Encode:  z_pre = W_enc @ (x - b_dec) + b_enc      [65536]
         z     = TopK(z_pre, k=64)                  [65536, only 64 nonzero]

Decode:  x_hat = W_dec @ z + b_dec                  [4096]

Loss:    L_mse = mean(||x_hat - x||² / ||x||²)     (normalized MSE)
         L_aux = (1/32) * mean(z_pre[dead]²)        (dead feature revival)
         L     = L_mse + L_aux
```

| Parameter | Value |
|---|---|
| d_model | 4,096 |
| d_sae | 65,536 (16× expansion) |
| k (sparsity) | 64 |
| Total SAE params | ~537M (2.1 GB FP32) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 3e-4 → 0 (cosine decay, 1000-step warmup) |
| Batch size | 4,096 tokens |
| Buffer size | 131,072 tokens (shuffled) |
| Total steps | 20,000 |
| Gradient clipping | 1.0 |
| Dead feature window | 5,000 steps (EMA) |
| Checkpoints | Every 5,000 steps |

### Target Layers

| Layer | Role | Key Neurons | Why |
|---|---|---|---|
| **L9** | Identity super-neuron | dim 994 (z=-13.96) | Strongest single-neuron personality signal in 8B |
| **L15** | Sarcasm relay node | dims 994, 235, 908 | Inverse node in sarcasm relay circuit |
| **L22** | Personality hub | dims 235, 908, 2136, 2514 | Lowest cross-model cosine (0.505) in debate arena |
| **L29** | Champion steering layer | dims 235, 2136, 2514 | Deployment champion (V4 + L29+L30@α=8) |

---

## Phase 1: Activation Collection

**Hardware**: RTX 3090 (10 GB model VRAM in INT8)
**Runtime**: 3h 51min (04:42 — 08:33)

### Prompt Bank

| Category | Prompts | × Conditions | × Temps | Total Runs |
|---|---|---|---|---|
| Sarcasm | 20 | 3 (none, V4, antipole) | 4 (0.3, 0.7, 1.0, 1.2) | 240 |
| Math | 10 | 3 | 4 | 120 |
| Knowledge | 10 | 3 | 4 | 120 |
| Identity | 10 | 3 | 4 | 120 |
| Character | 15 | 3 | 4 | 180 |
| **Total** | **65** | | | **780 generations** |

### Collection Results

| Layer | Tokens Collected | Shards | Size |
|---|---|---|---|
| L09 | 200,074 | 9 (25K/shard) | 1.6 GB |
| L15 | 200,074 | 9 | 1.6 GB |
| L22 | 200,074 | 9 | 1.6 GB |
| L29 | 200,074 | 9 | 1.6 GB |
| **Total** | **800,296** | **36** | **6.4 GB** |

- Storage: FP16 tensors + JSONL metadata per token (prompt_idx, token_position, is_generation flag, category, system_tag)
- Generation tokens tagged with `is_generation=True` for future gen-only SAE training

---

## Phase 2: SAE Training — Completed Layers

### L09 — Identity Layer

**GPU**: RTX 4090 | **Runtime**: 2h 11min (7,836s)

| Metric | Value |
|---|---|
| **Final FVE** | **97.7%** |
| Final MSE | 0.0227 |
| Dead features | 15,420 / 65,536 (**23.5%**) |
| Active features | **50,116 (76.5%)** |

**Training Curve:**

```
Step      MSE       FVE      Dead     LR
─────────────────────────────────────────
    1    0.8755    12.5%    65,536   3.0e-07
  100    0.7614    23.9%    54,138   3.0e-05
  500    0.1535    84.7%    53,014   1.5e-04
1,000    0.0732    92.7%    48,145   3.0e-04   ← peak LR
2,000    0.0431    95.7%    39,624   2.98e-04
5,000    0.0302    97.0%    26,686   2.68e-04
10,000   0.0261    97.4%    15,041   1.62e-04
15,000   0.0241    97.6%    11,938   4.84e-05
20,000   0.0227    97.7%    15,420   0.00e+00
```

**Analysis**: Excellent decomposition. L09 contains the identity super-neuron (dim 994, z=-13.96) — a highly concentrated signal that SAEs decompose easily. 76.5% feature utilization is strong. The slight uptick in dead features at step 20,000 is from the aux loss spike at lr=0 (dead feature revival failed once MSE gradients stopped).

### L22 — Personality Hub

**GPU**: RTX 3090 | **Runtime**: 4h 19min (15,534s)

| Metric | Value |
|---|---|
| **Final FVE** | **95.3%** |
| Final MSE | 0.0470 |
| Dead features | 54,301 / 65,536 (**82.8%**) |
| Active features | **11,235 (17.1%)** |

**Training Curve:**

```
Step      MSE       FVE      Dead     LR
─────────────────────────────────────────
    1    0.8713    12.9%    65,536   3.0e-07
  100    0.7623    23.8%    57,114   3.0e-05
  500    0.1763    82.4%    57,103   1.5e-04
1,000    0.0948    90.5%    56,903   3.0e-04
2,000    0.0692    93.1%    55,885   2.98e-04
5,000    0.0551    94.5%    55,015   2.68e-04
10,000   0.0502    95.0%    54,346   1.62e-04
15,000   0.0485    95.2%    54,182   4.84e-05
20,000   0.0470    95.3%    54,301   0.00e+00
```

**Analysis**: L22 is dramatically harder to decompose. 82.8% dead features means only 11,235 out of 65,536 features ever activate. This is consistent with L22's role as a personality hub — representations here are highly distributed and entangled, resisting sparse decomposition. The active features likely capture the most dominant modes (personality poles, generation mode, conversation context) but miss the subtler sub-modes.

**Possible improvements**: Larger expansion factor (32× or 64×), more training tokens, lower k, or switch to gated SAE architecture.

### L29 — Champion Steering Layer

**GPU**: RTX 4090 | **Runtime**: 2h 23min (8,564s)

| Metric | Value |
|---|---|
| **Final FVE** | **94.9%** |
| Final MSE | 0.0509 |
| Dead features | 55,957 / 65,536 (**85.4%**) |
| Active features | **9,579 (14.6%)** |

**Training Curve:**

```
Step      MSE       FVE      Dead     LR
─────────────────────────────────────────
    1    0.8751    12.5%    65,536   3.0e-07
  100    0.7826    21.7%    57,972   3.0e-05
  500    0.1899    81.0%    57,948   1.5e-04
1,000    0.0973    90.3%    57,745   3.0e-04   ← peak LR
2,000    0.0693    93.1%    57,625   2.98e-04
5,000    0.0574    94.3%    57,589   2.68e-04
10,000   0.0536    94.6%    56,082   1.62e-04
15,000   0.0517    94.8%    55,902   4.84e-05
20,000   0.0509    94.9%    55,957   0.00e+00
```

**Analysis**: Worst decomposition of all four layers. 85.4% dead features — only 9,579 features ever activate out of 65,536. Dead features barely budge throughout training: 57,972 at step 100 → 55,957 at step 20,000 (only 2,015 features revived across the entire run). The aux loss mechanism almost completely fails here.

This is the champion steering layer (V4 + L29+L30@α=8 is the deployment config). The extreme sparsity suggests L29 representations are maximally entangled — personality, reasoning, and factual processing are so deeply integrated that the SAE cannot find 65,536 independent directions. The few active features (9,579) likely capture only the dominant principal components, missing the subtle sub-modes that make this layer effective for steering.

**Comparison with L22**: Both are deep personality layers with >80% dead features, but L29 is worse on every metric: higher MSE (0.051 vs 0.047), lower FVE (94.9% vs 95.3%), more dead features (85.4% vs 82.8%). Yet L29 is the *better* steering target. This reinforces the finding that decomposability is inversely related to steerability.

**Notable**: Despite being on the faster 4090, L29 trained in 2h 23min vs L09's 2h 11min on the same GPU. The higher dead feature count means fewer active gradients per step, but the loss landscape appears flatter (harder optimization), roughly canceling out.

---

## Phase 3: SAE Training — In Progress

### L15 — Sarcasm Relay Node

**GPU**: RTX 3090 | **Started**: 12:52 | **Progress**: 50% (step 10,000/20,000)

| Metric | Current (step 10K) |
|---|---|
| FVE | 96.5% |
| Dead features | 26,777 (40.8%) |
| Active features | 38,759 (59.1%) |

L15 is decomposing significantly better than L22 or L29. The relay node sits at an intermediate depth where information is structured enough for the SAE to find meaningful sparse features, but not yet maximally entangled. 59% feature utilization at the halfway point — may improve further as training continues.

---

## Cross-Layer Decomposability Gradient

```
          Active Features (%)
L09  ████████████████████████████████████████  76.5%
L15  ██████████████████████████████           59.1%  (at 50% training)
L22  ████████                                 17.1%
L29  ███████                                  14.6%

          FVE (%)
L09  ██████████████████████████████████████████████████  97.7%
L15  █████████████████████████████████████████████████   96.5%  (at 50% training)
L22  ███████████████████████████████████████████████     95.3%
L29  ██████████████████████████████████████████████      94.9%
```

**The deeper the layer, the harder to decompose.** Active feature count drops: 76.5% → 59.1% → 17.1% → 14.6%. There is a sharp cliff between L15 and L22 — feature utilization drops from ~60% to ~17%, a 3.5× collapse. This cliff corresponds to the transition from the sarcasm relay circuit (L9→L14→L15) into the personality hub (L22) where multiple behavioral dimensions converge.

**Interpretation**: Early layers (L09) have concentrated, quasi-monosemantic features that SAEs decompose well. The relay node (L15) has moderate structure. Deep layers (L22, L29) pack personality, reasoning, and factual processing into highly superimposed representations — exactly the kind of entanglement that makes steering hard and SAE decomposition sparse.

---

## Key Findings

### 1. L09 is Highly Decomposable

FVE 97.7% with 76.5% feature utilization. The identity super-neuron (dim 994) provides a clear, concentrated signal. SAE features at this layer likely correspond to interpretable concepts — pending analysis with `sae_analyze.py`.

### 2. L22 Resists Sparse Decomposition

82.8% dead features despite identical architecture and training. The personality hub's representations are too distributed for a 16× expansion SAE with k=64. This is actually informative: it tells us that personality at L22 is genuinely high-dimensional, not just polysemantic superposition of a few concepts.

### 3. Decomposability Predicts Steerability

The decomposability gradient (L09 > L15 > L22 > L29) is roughly inverse to steering effectiveness (L29 > L22 > L15 > L09). Layers that are hard for SAEs to decompose may be hard to decompose precisely because information is deeply integrated there — and it's that integration that makes single-direction steering effective (one direction perturbs many entangled features simultaneously).

### 4. 200K Tokens May Be Insufficient for Deep Layers

L22 and L29 may benefit from more training data. With 200K tokens at batch_size=4096, we get ~49 passes through the data in 20K steps. More diverse prompts and higher token counts could reduce dead features.

---

## Upcoming

### Queued on Dev Server

1. **Gen-only SAE training** (Phases 6-7): Same layers but trained only on generation tokens (filtered via `is_generation` metadata). Hypothesis: gen tokens carry 2-7% stronger personality signal, so gen-only SAEs may achieve better FVE and fewer dead features at personality-relevant layers.

2. **Phase 2 orchestrator**: Sycophancy steering tests and arena battles, then debate arena v4. Waiting for Phase 5 to complete.

### Queued on PRO 6000

1. **27B SAE collection + training**: Target layers L0, L16, L36, L44, L50 from the connectome analysis. 5120-dim hidden states at 16× expansion = d_sae=81,920. Requires PRO 6000 (96GB) to hold model + activations.

### Analysis Pipeline

Once all 4 layers finish training: `sae_analyze.py` will compute:
- Per-feature activation frequency and top-activating tokens
- Decoder column correlation with connectome z-score directions
- Hub neuron decomposition (do SAE features split dim 994 into category-specific sub-features?)
- Cross-layer feature comparison (do sarcasm features at L15 correspond to sarcasm features at L22?)

---

## File Locations

| Asset | Path (Dev Server) |
|---|---|
| Activations | `/home/orwel/dev_genius/sae_8b/sae_8b/activations/{L09,L15,L22,L29}/` |
| Trained SAEs | `/home/orwel/dev_genius/sae_8b/sae_8b/models/{L09,L22,L29}/` |
| Training Logs | `/home/orwel/dev_genius/sae_8b/sae_train_{L09,L22,L15,L29}.log` |
| Collection Log | `/home/orwel/dev_genius/sae_8b/sae_collect.log` |
| Pipeline Script | `/home/orwel/dev_genius/experiments/Character Creation/sae_8b_pipeline.py` |
| Overnight Log | `/home/orwel/dev_genius/sae_8b/overnight.log` |
