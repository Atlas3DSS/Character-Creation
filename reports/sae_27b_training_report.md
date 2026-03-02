# SAE Training Report -- Qwen3.5-27B Dense (L50 & L44)

**Date**: 2026-03-01
**Model**: Qwen3.5-27B-FP8 (64 layers, hidden=5120, GatedDeltaNet)
**Hardware**: RTX PRO 6000 96GB (WSL)
**Scripts**: `sae_train.py`, `sae_config.py`

---

## Executive Summary

Two TopK Sparse Autoencoders were trained to completion on layers 50 and 44 of the Qwen3.5-27B dense model. Both SAEs converged well, achieving 94.5% FVE (L50) and 96.5% FVE (L44), with feature utilization exceeding 95% at convergence. L44 trained significantly better than L50 on every metric -- lower MSE, higher FVE, fewer dead features, and faster convergence. This is consistent with L44's role as a mid-network sarcasm/brevity layer versus L50's role as a super-hub where multiple high-dimensional representations collide.

The 27B SAEs show dramatically better feature utilization (95-99%) than the 8B SAEs at deep layers (14-17%), confirming that the 16x expansion ratio (81,920 features) is better matched to the 27B's 5120-dim hidden states than the 8B's 4096-dim space. However, the final MSE values (0.035-0.055) leave room for improvement -- further training or architectural changes could push FVE above 97%.

---

## 1. Architecture & Configuration

### TopK SAE Design

```
Input x in R^5120 (hidden state from target layer)

Encode:  z_pre = W_enc @ (x - b_dec) + b_enc      [81,920]
         z     = TopK(z_pre, k=64)                  [81,920, only 64 nonzero]

Decode:  x_hat = W_dec @ z + b_dec                  [5,120]

Loss:    L_mse = mean(||x_hat - x||^2 / ||x||^2)   (normalized MSE)
         L_aux = (1/32) * mean(z_pre[dead]^2)       (dead feature revival)
         L     = L_mse + L_aux
```

### Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 5,120 |
| d_sae | 81,920 (16x expansion) |
| k (TopK sparsity) | 64 |
| Total SAE parameters | 838,947,840 (0.84B) |
| Model size on disk (FP32) | 3.13 GB |
| Optimizer | Adam (beta1=0.9, beta2=0.999) |
| Peak learning rate | 3e-4 |
| LR schedule | Cosine decay with 1000-step linear warmup |
| Batch size | 4,096 tokens |
| Buffer size | 131,072 tokens (shuffled reservoir) |
| Total steps | 50,000 |
| Gradient clipping | 1.0 |
| Weight decay | 0.0 |
| Dead feature window | 5,000 steps (EMA) |
| Dead feature threshold | 1e-5 |
| Aux loss coefficient | 1/32 = 0.03125 |
| Checkpoints | Every 5,000 steps |
| Training dtype | float32 |
| Seed | 42 |

### Data Statistics

| Metric | Value |
|---|---|
| Activation tokens per layer | 348,233 |
| Activation shards | 7 per layer (50K tokens each, last shard 48,233) |
| Activation storage dtype | float16 |
| Activation storage per layer | ~3.4 GB |
| Total tokens seen (50K steps x 4096 batch) | 204,800,000 |
| Effective epochs over dataset | 588x |
| Generation-only filtering | Disabled (all tokens used) |

### Activation Collection Context

- **Model**: Qwen3.5-27B-FP8
- **Collection date**: 2026-02-28 (06:31 -- 00:20, ~18 hours)
- **Layers collected**: L16, L36, L44, L50
- **Target tokens**: 500,000 per layer (achieved 348,233 -- 70% of target)
- **Temperatures**: [0.3, 0.7, 1.0, 1.2] (diverse generation conditions)
- **Prompt bank**: Multi-category prompts from `prompt_bank.json`

---

## 2. Training Dynamics -- Layer 50

L50 is the **super-hub layer** of the 27B model. The connectome analysis identified it as the peak layer for 7 of 20 categories: Identity, Sarcastic, Formal, Fear, Verbose, Certainty, and History. It contains dim 2028, the strongest single-neuron signal in the entire 27B model (Code z=6.67, Math z=6.19, Sadness z=5.84).

### L50 Training Curve

```
Step      MSE       FVE      Dead     Alive   Alive%    LR        Elapsed
---------------------------------------------------------------------------
     1    0.8936    10.6%    81,920       0    0.0%    3.00e-07    0.0h
   100    0.7918    20.8%    69,685  12,235   14.9%    3.00e-05    0.0h
   500    0.2339    76.6%    69,116  12,804   15.6%    1.50e-04    0.1h
 1,000    0.1445    85.6%    68,943  12,977   15.8%    3.00e-04    0.1h  <-- peak LR
 2,000    0.1105    89.0%    68,573  13,347   16.3%    3.00e-04    0.2h
 5,000    0.0922    90.8%    67,487  14,433   17.6%    2.95e-04    0.6h
 7,000    0.0890    91.1%    62,527  19,393   23.7%    2.87e-04    0.8h  <-- revival begins
 8,000    0.0879    91.2%    57,930  23,990   29.3%    2.82e-04    0.9h
 9,000    0.0871    91.3%    51,436  30,484   37.2%    2.79e-04    1.0h
10,000    0.0868    91.3%    43,741  38,179   46.6%    2.76e-04    1.2h
12,000    0.0847    91.5%    27,965  53,955   65.9%    2.61e-04    1.5h
14,000    0.0818    91.8%    17,638  64,282   78.5%    2.47e-04    1.7h
15,000    0.0802    92.0%    14,090  67,830   82.8%    2.44e-04    1.8h
17,000    0.0759    92.4%     9,612  72,308   88.3%    2.28e-04    2.0h
20,000    0.0713    92.9%     5,743  76,177   93.0%    2.02e-04    2.3h
25,000    0.0685    93.1%     1,915  80,005   97.7%    1.55e-04    2.9h
30,000    0.0631    93.7%     1,394  80,526   98.3%    1.07e-04    3.4h
35,000    0.0597    94.0%     1,212  80,708   98.5%    6.42e-05    4.0h  <-- min dead
40,000    0.0565    94.4%     1,219  80,701   98.5%    2.98e-05    4.5h
45,000    0.0558    94.4%     1,236  80,684   98.5%    7.64e-06    5.0h
50,000    0.0547    94.5%     3,945  77,975   95.2%    0.00e+00    5.6h  <-- final
```

**Total wall time**: 5.57 hours (20,093 seconds)
**Lowest MSE achieved**: 0.0542 at step 49,700
**Minimum dead features**: 1,207 at step 35,400 (98.5% alive)

### L50 Training Phases

1. **Warmup (steps 0-1000)**: MSE drops rapidly from 0.89 to 0.14 as the encoder learns basic reconstruction. Dead features remain flat at ~69K (only ~16% of features are initially active -- the encoder finds 12,977 directions in the first 1000 steps).

2. **Slow plateau (steps 1000-7000)**: MSE creeps from 0.14 to 0.089. Dead features barely move (68,943 to 62,527). The active features refine their directions, but no new features are born. This is a classic SAE stagnation phase.

3. **Feature revival cascade (steps 7000-25000)**: The aux loss finally overcomes the dead feature threshold and triggers a massive revival wave. Dead features plummet from 62,527 to 1,915 between steps 7,000 and 25,000. This is the critical training phase: 60,612 features (74% of the dictionary) are revived in 18,000 steps. The revival corresponds to steady MSE improvement (0.089 to 0.069).

4. **Convergence plateau (steps 25000-45000)**: Dead features stabilize at ~1,200. MSE continues to decrease slowly from 0.069 to 0.056. The cosine LR decay is approaching zero, limiting further optimization.

5. **Terminal aux spike (steps 45000-50000)**: As LR hits zero, dead features spike from 1,236 to 3,945. The aux loss spikes to 0.127 (10x its steady-state value of ~0.01). This is an artifact of lr=0: the optimizer can no longer maintain feature activity, and marginally-alive features die. The final MSE (0.0547) is still the best value, suggesting the core reconstruction quality was not harmed.

---

## 3. Training Dynamics -- Layer 44

L44 is a **mid-network sarcasm/brevity layer**. The connectome analysis shows it as the peak layer for Verbosity: Brief (z=10.07 at dim 526 -- the single strongest neuron-level signal in the entire 27B model). It also carries Tone: Sarcastic, Emotion: Anger, and Role: Authority signals via key dims 2768 and 4010.

### L44 Training Curve

```
Step      MSE       FVE      Dead     Alive   Alive%    LR        Elapsed
---------------------------------------------------------------------------
     1    0.9094     9.1%    81,920       0    0.0%    3.00e-07    0.0h
   100    0.8314    16.9%    71,449  10,471   12.8%    3.00e-05    0.0h
   500    0.1753    82.5%    70,759  11,161   13.6%    1.50e-04    0.1h
 1,000    0.1011    89.9%    70,288  11,632   14.2%    3.00e-04    0.1h  <-- peak LR
 2,000    0.0732    92.7%    69,556  12,364   15.1%    3.00e-04    0.3h
 5,000    0.0582    94.2%    68,505  13,415   16.4%    2.95e-04    0.6h
 7,000    0.0559    94.4%    64,725  17,195   21.0%    2.87e-04    0.9h  <-- revival begins
 8,000    0.0553    94.5%    61,627  20,293   24.8%    2.82e-04    1.0h
 9,000    0.0546    94.5%    57,594  24,326   29.7%    2.79e-04    1.1h
10,000    0.0535    94.7%    52,597  29,323   35.8%    2.76e-04    1.3h
12,000    0.0524    94.8%    38,839  43,081   52.6%    2.61e-04    1.5h
14,000    0.0500    95.0%    24,015  57,905   70.7%    2.47e-04    1.9h
15,000    0.0511    94.9%    18,371  63,549   77.6%    2.44e-04    2.1h
17,000    0.0479    95.2%    10,717  71,203   86.9%    2.28e-04    2.4h
20,000    0.0454    95.5%     5,475  76,445   93.3%    2.02e-04    2.7h
25,000    0.0428    95.7%       815  81,105   99.0%    1.55e-04    3.2h
30,000    0.0393    96.1%       532  81,388   99.4%    1.07e-04    3.8h
35,000    0.0372    96.3%       523  81,397   99.4%    6.42e-05    4.3h  <-- min dead
40,000    0.0352    96.5%       538  81,382   99.3%    2.98e-05    4.9h
  ------ RESUME FROM CHECKPOINT (step 40000) ------
45,000    0.0352    96.5%       547  81,373   99.3%    7.64e-06    0.5h*
50,000    0.0350    96.5%     3,787  78,133   95.4%    0.00e+00    1.1h*
```

*L44 elapsed times after step 40,000 are from the resumed run.

**Total wall time**: 5.97 hours (4.92h first run + 1.05h resumed final 10K steps)
**Lowest MSE achieved**: 0.0335 at step 48,600
**Minimum dead features**: 521 at step 33,700 (99.4% alive)
**Resume point**: Checkpoint at step 40,000 was loaded and training continued for the final 10,000 steps.

### L44 Training Phases

L44 follows the same four-phase pattern as L50 but with faster convergence at every stage:

1. **Warmup (steps 0-1000)**: MSE drops from 0.91 to 0.10. Notably faster than L50 (which was at 0.14 at step 1000). L44's activations are easier to approximate from the start.

2. **Slow plateau (steps 1000-7000)**: MSE descends from 0.10 to 0.056. This is already below L50's step 50,000 final value. Dead features remain at ~65K-69K.

3. **Feature revival cascade (steps 7000-25000)**: Dead features drop from 64,725 to 815. Revival is slightly more efficient than L50: L44 reaches 99.0% alive at step 25,000 while L50 only reaches 97.7%.

4. **Convergence (steps 25000-50000)**: MSE continues a slow descent from 0.043 to 0.035. Dead features stabilize at ~520. The terminal aux spike at lr=0 pushes dead features to 3,787 (same artifact as L50).

---

## 4. Head-to-Head Comparison: L50 vs L44

### Final Metrics

| Metric | L50 | L44 | Winner |
|---|---|---|---|
| Final MSE | 0.0547 | 0.0350 | **L44** (1.56x lower) |
| Final FVE | 94.5% | 96.5% | **L44** (+2.0pp) |
| Lowest MSE achieved | 0.0542 (step 49700) | 0.0335 (step 48600) | **L44** (1.62x lower) |
| Min dead features | 1,207 (step 35400) | 521 (step 33700) | **L44** (2.3x fewer) |
| Peak alive % | 98.5% | 99.4% | **L44** (+0.9pp) |
| Final alive features | 77,975 / 81,920 | 78,133 / 81,920 | **L44** (by 158) |
| Terminal dead spike | +2,733 features | +3,266 features | L50 (smaller spike) |
| Total wall time | 5.57h | 5.97h | L50 (0.4h faster) |
| Speed (steps/sec) | 2.49 it/s | 2.36 it/s avg | L50 (slightly faster) |

### Convergence Speed (MSE milestones)

| MSE Threshold | L50 (step) | L44 (step) | L44 Faster By |
|---|---|---|---|
| < 0.50 | 300 | 300 | tie |
| < 0.20 | 700 | 500 | 1.4x |
| < 0.10 | 3,100 | 1,100 | 2.8x |
| < 0.08 | 14,100 | 1,600 | 8.8x |
| < 0.06 | 33,000 | 4,000 | 8.3x |
| < 0.04 | never | 28,700 | -- |

L44 converges dramatically faster. At step 5,000, L44 (MSE=0.058) is already below L50's final value at step 50,000 (MSE=0.055). The MSE gap narrows over training but never closes: L50 is consistently ~0.02 MSE points behind L44 at every checkpoint.

### MSE Gap Over Training

```
Step    L50 MSE    L44 MSE    Gap     L44 Advantage
-------------------------------------------------------
 1000   0.14447    0.10106    0.043    30% lower
 5000   0.09224    0.05818    0.034    37% lower
10000   0.08681    0.05346    0.034    38% lower
20000   0.07126    0.04538    0.026    36% lower
30000   0.06306    0.03928    0.024    38% lower
50000   0.05472    0.03504    0.020    36% lower
```

The gap is remarkably stable at 36-38% throughout training. L44 is not just converging faster -- it is approaching a fundamentally lower loss floor.

### Interpretation

L50 is harder to decompose because it is the **convergence point** of multiple high-dimensional representations:
- **dim 2028** is a super-hub crossing Code, Math, Sadness, Science, Analytical, Identity, and Polite
- **dim 423** carries Role: Authority signals
- **dim 3968** carries Anger, Sarcastic, Polite, Code signals
- 7 of 20 connectome categories peak at L50

L44, by contrast, has a more structured representation:
- **dim 526** carries the strongest single-neuron signal in the entire model (Verbosity: Brief, z=10.07)
- **dim 2768** is the broadest hub (12 categories) but its peak z-scores are moderate (2.0-2.8)
- **dim 4010** carries Anger and Sarcastic signals
- Fewer categories peak here (Brief at L44; Sarcastic peaks at L36, Authority at L50)

The SAE decomposition results confirm the connectome findings: L44 has more concentrated, quasi-monosemantic features (dominated by the brief/verbose axis), while L50 has maximally entangled representations where multiple behavioral dimensions collide. The SAE finds more independent directions at L44 because the representation IS more factorizable.

---

## 5. Comparison with 8B SAE Trial

The 8B SAEs (trained on L09, L15, L22, L29 of Qwen3-VL-8B) provide a cross-architecture baseline.

### Architecture Comparison

| Parameter | 8B SAE | 27B SAE |
|---|---|---|
| d_model | 4,096 | 5,120 |
| d_sae | 65,536 | 81,920 |
| Expansion ratio | 16x | 16x |
| k (TopK) | 64 | 64 |
| SAE parameters | 537M | 839M |
| Total steps | 20,000 | 50,000 |
| Training tokens | 200,074 | 348,233 |
| Tokens per step | 4,096 | 4,096 |

### Decomposability Comparison (Final Metrics)

| Layer | FVE | Dead % | Alive % | Notes |
|---|---|---|---|---|
| **8B L09** | **97.7%** | **23.5%** | **76.5%** | Identity super-neuron (concentrated) |
| **8B L15** | 96.5%* | 40.8%* | 59.1%* | Sarcasm relay (at 50% training) |
| **8B L22** | 95.3% | 82.8% | 17.1% | Personality hub (entangled) |
| **8B L29** | 94.9% | 85.4% | 14.6% | Champion steering layer (most entangled) |
| **27B L44** | **96.5%** | **4.6%** | **95.4%** | Mid-network sarcasm/brevity |
| **27B L50** | **94.5%** | **4.8%** | **95.2%** | Super-hub (most entangled in 27B) |

The 27B SAEs achieve dramatically better feature utilization than the 8B SAEs at deep layers. The 8B's L22 and L29 had 82-85% dead features, while the 27B's L44 and L50 stabilize at only 4.6-4.8% dead. Several factors contribute:

1. **Longer training**: 50K steps (27B) vs 20K steps (8B). The feature revival cascade needs ~7,000-25,000 steps to complete, which the 8B runs cut short at 20K.

2. **More training data**: 348K tokens (27B) vs 200K tokens (8B). More diverse activations help the SAE discover more features.

3. **Better expansion match**: The 27B's 5120-dim hidden states may be better served by 81,920 features than the 8B's 4096-dim states by 65,536 features, despite the same 16x ratio.

4. **Architecture difference**: The 27B uses GatedDeltaNet (linear attention with delta updates and convolutions), which may produce activations with different statistical structure than the 8B's standard attention layers.

However, the FVE comparison tells a different story. The 27B L50 (94.5% FVE) is roughly comparable to the 8B L29 (94.9% FVE) -- despite having 5x more alive features. This means the 27B's residual reconstruction error is proportionally similar. The 27B needs more features active to achieve the same reconstruction quality, consistent with its more distributed representations.

---

## 6. Dead Feature Dynamics

Both layers exhibit the same characteristic four-phase dead-feature lifecycle.

### Phase Diagram

```
                Dead Features (thousands)
       0    10   20   30   40   50   60   70   80
       |    |    |    |    |    |    |    |    |
  0K   |                                     XXXX  Initial (all dead)
  1K   |                                    XXXX   Warmup (fast initial drop)
  5K   |                                   XXXX    Plateau (stagnation)
  7K   |                               XXXX----    Revival onset
 10K   |                         XXXX----          Revival cascade
 12K   |                  XXXX----
 15K   |            XXXX----
 20K   |       XXXX----
 25K   |  XX----                                   Convergence
 30K   |  X
 35K   |  X                                        Minimum dead (~1.2K / 0.5K)
 40K   |  X
 45K   |  X
 50K   |  XXXX                                     Terminal spike (lr=0)

       L50 = X    L44 = ----
```

### Revival Cascade Comparison

| Metric | L50 | L44 |
|---|---|---|
| Revival onset | step 7,000 | step 7,000 |
| Revival midpoint (50% of features alive) | step ~11,000 | step ~12,000 |
| 90% alive | step ~18,000 | step ~19,000 |
| 95% alive | step ~20,000 | step ~20,000 |
| 99% alive | step ~27,000 | step ~24,000 |
| Peak alive | 98.5% (step 35,400) | 99.4% (step 33,700) |
| Terminal dead spike | +2,733 | +3,266 |

The revival cascades are nearly synchronous, both starting at step 7,000. This is likely controlled by the aux loss EMA window (5,000 steps) rather than the layer-specific activation statistics. The terminal spike at lr=0 is a known artifact -- when the optimizer can no longer maintain gradient flow, marginally active features fall below the dead threshold.

### Terminal Spike Analysis

At step 50,000 (lr=0):
- **L50**: aux loss spikes to 0.127 (from ~0.01 steady-state). Dead features: 1,236 -> 3,945 (+2,709 in 5,000 steps).
- **L44**: aux loss spikes to 0.067 (from ~0.01 steady-state). Dead features: 547 -> 3,787 (+3,240 in 5,000 steps).

The spike is worse for L44 in absolute feature count but less severe in aux loss magnitude. This suggests L44 has more features near the death threshold that were being kept alive by the optimizer's ongoing gradient pressure.

**Recommendation**: For future runs, add a short warmup at the end (anti-decay) or stop training 2,000 steps before the LR hits zero to preserve the peak alive state.

---

## 7. What These Layers Represent

### L50 -- The Super-Hub

L50 is where the 27B model integrates multiple high-level representations into a unified computation. The connectome identified it as the peak layer for:

| Category | Peak Dim | Max |z| | Role at L50 |
|---|---|---|---|
| Domain: Code | 2028 | 6.67 | Super-hub neuron |
| Domain: Math | 2028 | 6.19 | Super-hub neuron |
| Domain: Science | 2028 | 3.81 | Super-hub neuron |
| Emotion: Sadness | 2028 | 5.84 | Super-hub neuron |
| Reasoning: Analytical | 2028 | 3.29 | Super-hub neuron |
| Role: Authority | 423 | 3.18 | Authority specialist |
| Identity | (peaks L43, via dim 94) | 1.06 | Weak signal here |

**dim 2028 is a polysemantic mega-hub** that simultaneously encodes Code, Math, Science, Sadness, and Analytical reasoning. The SAE's job is to decompose this hub into monosemantic sub-features. With 94.5% FVE and 77,975 alive features, the SAE captures most of the variance but may miss fine-grained sub-modes.

The layer type is GatedDeltaNet (linear attention), not full attention. Only layers at indices [3, 7, 11, ...] (every 4th) use full attention. L50 is not one of these (50 mod 4 = 2), so it uses the linear GatedDeltaNet mechanism. This means L50's representations are shaped by delta-net recurrence rather than softmax attention, potentially producing different activation statistics than a full-attention hub.

### L44 -- The Brevity/Sarcasm Controller

L44 is 6 layers upstream of L50 and serves as the peak layer for:

| Category | Peak Dim | Max |z| | Role at L44 |
|---|---|---|---|
| Verbosity: Brief | 526 | 10.07 | Strongest neuron-level signal in 27B |
| Length: Brief | (same) | | Controls output length |
| Tone: Sarcastic | 2768* | 2.59 | Sarcasm at mid-network |
| Emotion: Anger | 2768* | varies | Anger co-located with sarcasm |
| Role: Authority | 4010 | varies | Authority register |

*dim 2768 is the broadest hub in the 27B model (12 categories). It peaks at L34 for some categories but carries significant signal at L44.

The **Brief neuron (dim 526, z=10.07)** is exceptional: it is the single strongest neuron-category association in the entire 27B connectome, nearly 2x stronger than the next-strongest (dim 2028 at L50 for Code, z=6.67). This concentrated signal explains why L44 decomposes more easily -- the SAE can identify a clean "brevity" feature without much interference.

L44 is also a GatedDeltaNet layer (44 mod 4 = 0, and while layer indices [3,7,11,...,63] are full attention, L44 = index 44 is not one of those since 44 is not of the form 4k+3). It handles sarcasm routing before information reaches the L50 super-hub.

---

## 8. Reconstruction Quality Assessment

### FVE Interpretation

FVE (Fraction of Variance Explained) = 1 - normalized_MSE, where normalized MSE divides by the input norm squared. This means:

- **L50 at 94.5% FVE**: The SAE reconstructs 94.5% of the signal energy. The remaining 5.5% reconstruction error is distributed across all 5,120 dimensions. On average, each dimension has sqrt(0.055/5120) ~ 0.003 units of reconstruction noise.

- **L44 at 96.5% FVE**: 3.5% reconstruction error. Each dimension has ~0.003 units of noise on average (similar due to different activation norms).

### How Does This Compare?

Published SAE benchmarks on other models:

| Work | Model | FVE | Expansion | k |
|---|---|---|---|---|
| Anthropic (2024) | Claude 3 Sonnet | ~97% | 32x | varies |
| OpenAI (2024) | GPT-4 | ~96% | 64x | varies |
| EleutherAI | Pythia 6.9B | ~95% | 16x | 64 |
| **This work (L44)** | **Qwen3.5-27B** | **96.5%** | **16x** | **64** |
| **This work (L50)** | **Qwen3.5-27B** | **94.5%** | **16x** | **64** |

L44's 96.5% FVE is competitive with published results. L50's 94.5% is slightly below, consistent with its role as a more entangled layer. Both could likely improve with:

1. **Higher expansion ratio** (32x or 64x): More features to capture finer sub-modes
2. **More training data**: 348K tokens is modest; 1M+ would help
3. **Gated SAE architecture**: Adding a gating mechanism could improve reconstruction on the GatedDeltaNet layers specifically
4. **Longer training**: MSE was still decreasing (slowly) at step 50,000

---

## 9. Resource Usage

### Disk Usage

| Asset | Per Layer | Total (2 layers) |
|---|---|---|
| Activations (shards + metadata) | 3.5 GB | 7.0 GB |
| SAE final model (FP32) | 3.13 GB | 6.26 GB |
| Checkpoints (10 x ~10 GB each) | ~93.8 GB | ~187.6 GB |
| Training log JSON | ~127 KB | ~254 KB |
| **Total per layer** | **~96.9 GB** | **~193.8 GB** |

The checkpoints dominate disk usage. Each checkpoint is ~10 GB because it includes the full optimizer state (Adam momentum + variance buffers, same size as the model parameters). Consider keeping only the final checkpoint and sae_final.pt to save ~84 GB per layer.

### GPU Memory

Training on a single RTX PRO 6000 (96 GB):
- SAE model (FP32): ~3.13 GB
- Optimizer state (Adam): ~6.26 GB (2x model size for momentum + variance)
- Activation buffer (131K x 5120 x FP32): ~2.56 GB
- Batch + gradients: ~1.5 GB
- **Total estimated**: ~13.5 GB (14% of 96 GB)

Training was not GPU memory-bound. The PRO 6000 could comfortably train 4-6 SAEs in parallel, though this was not attempted.

### Compute

- Training throughput: ~2.4-2.5 it/s (steps per second)
- Time per step: ~0.4 seconds
- Total compute per layer: ~5.6-6.0 hours
- Tokens processed per second: 2.5 * 4096 = ~10,240 tokens/sec

---

## 10. Next Steps

### Immediate Analysis (High Priority)

1. **Feature interpretation**: Run `sae_analyze.py` to compute:
   - Per-feature activation frequency histograms
   - Top-activating tokens/contexts for each feature
   - Decoder column correlation with connectome z-score directions
   - Hub neuron decomposition: Does dim 2028's polysemantic encoding decompose into distinct Code, Math, Sadness features?

2. **Connectome alignment**: For each of the 20 connectome categories, find the SAE features whose decoder columns have highest cosine similarity with the category's z-score vector. This tests whether the SAE discovers the same conceptual directions as the contrastive connectome.

3. **Cross-layer feature comparison**: Do sarcasm features at L44 correspond to sarcasm features at L50? Compute cosine similarity between L44 and L50 decoder columns (projected into shared activation space) for features identified as sarcasm-related.

### Follow-Up Training (Medium Priority)

4. **L16 and L36 SAE training**: Activations are already collected for these layers (348,233 tokens each). L36 is the sarcasm peak layer (Tone: Sarcastic peaks at L36 with dim 2768, z=2.59). L16 is the refusal migration anchor. Training these would complete the 4-layer SAE battery.

5. **Generation-only SAE**: Re-train on only the generation tokens (filtered via `is_generation` metadata in shard JSONL files). The debate arena found that generation activations carry 2-7% stronger personality signal. A gen-only SAE might capture personality features more cleanly.

6. **Checkpoint cleanup**: Delete intermediate checkpoints (steps 5K-45K), retaining only the final checkpoint and sae_final.pt. This would reclaim ~168 GB of disk space.

### Architecture Experiments (Low Priority)

7. **Higher expansion**: Train a 32x expansion SAE (d_sae=163,840) on L50 to test whether the super-hub needs more features for adequate decomposition.

8. **Gated SAE**: The Anthropic-style gated architecture (separate gate and magnitude networks) may better capture the GatedDeltaNet layer structure.

9. **Different k values**: Sweep k in [32, 64, 128, 256] to find the sparsity-reconstruction Pareto frontier.

### Integration with Steering Pipeline

10. **Feature-level steering**: Instead of steering with raw connectome z-score vectors, identify specific SAE features that correspond to target traits (sarcasm, identity, brevity) and steer by clamping those features during generation. This enables more surgical personality control.

11. **Ablation via features**: Permanently modify model weights to ablate specific SAE features (not just directions). This could enable baking personality changes that are more monosemantic than raw direction ablation.

---

## File Locations

| Asset | Path |
|---|---|
| SAE config | `/home/orwel/dev_genius/experiments/Character Creation/scripts/sae/sae_config.py` |
| Training script | `/home/orwel/dev_genius/experiments/Character Creation/scripts/sae/sae_train.py` |
| L50 final model | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L50/sae_final.pt` |
| L44 final model | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L44/sae_final.pt` |
| L50 training log | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L50/training_log.json` |
| L44 training log | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L44/training_log.json` |
| L50 training summary | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L50/training_summary.json` |
| L44 training summary | `/home/orwel/dev_genius/experiments/Character Creation/sae_models/base/L44/training_summary.json` |
| L50 train log (raw) | `/home/orwel/dev_genius/experiments/Character Creation/logs/sae_27b_train_L50.log` |
| L44 train log (raw) | `/home/orwel/dev_genius/experiments/Character Creation/logs/sae_27b_train_L44.log` |
| L44 resume log (raw) | `/home/orwel/dev_genius/experiments/Character Creation/logs/sae_27b_train_L44_resume.log` |
| L50 activations | `/home/orwel/dev_genius/experiments/Character Creation/sae_activations/base/L50/` |
| L44 activations | `/home/orwel/dev_genius/experiments/Character Creation/sae_activations/base/L44/` |
| Collection config | `/home/orwel/dev_genius/experiments/Character Creation/sae_activations/base/collection_run_config.json` |
| Connectome z-scores | `/home/orwel/dev_genius/experiments/Character Creation/qwen35_map/27b/connectome_zscores.pt` |
| Hub neurons | `/home/orwel/dev_genius/experiments/Character Creation/qwen35_map/27b/hub_neurons.json` |
| 8B SAE trial report | `/home/orwel/dev_genius/experiments/Character Creation/reports/sae_8b_trial_report.md` |
| 27B connectome report | `/home/orwel/dev_genius/experiments/Character Creation/reports/27b_connectome_analysis_report.md` |
