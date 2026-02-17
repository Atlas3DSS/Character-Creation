# Character Matrices: Interpretable Personality Trajectories in LoRA Weight Space During GRPO Training

## Abstract

We present a method for tracking how language models learn fictional character voice during Group Relative Policy Optimization (GRPO) by projecting LoRA weight deltas into a pre-computed personality subspace. Using PCA of activation centroids from 8 persona archetypes, we define a K-dimensional (K≈6) coordinate system in which LoRA B⊗A matrices can be decomposed into personality-relevant and personality-orthogonal components. We log this decomposition at each training checkpoint, producing a "character matrix trajectory" — a compact, interpretable record of what GRPO learns about personality versus reasoning. Our key finding is [PENDING V3 RESULTS]. We demonstrate that these trajectories can be [replayed/mutated/ablated] to produce new character variants without retraining.

## 1. Introduction

Making LLMs adopt specific fictional character voices (not just Big Five traits, but full literary personas with voice, knowledge, relationships, and behavioral patterns) remains unsolved. The core tension: personality and reasoning share weight space. Supervised fine-tuning on character dialogue catastrophically forgets reasoning (we show 0% AIME across all LoRA merge scales), while inference-time steering vectors are too low-dimensional to capture rich characters.

GRPO offers a middle path — optimizing via reward signal rather than direct weight supervision — but the mechanism by which it learns character is opaque. We introduce tools to make it interpretable.

## 2. Related Work

### 2.1 Activation-Space Personality Steering
- Contrastive activation addition (CAA) extracts trait directions from prompt pairs [Rimsky et al. 2023]
- Hybrid layer selection improves reliability across architectures [2511.03738]
- Personality traits occupy low-rank shared subspaces; PCA captures >95% inter-trait variance [2511.03738]
- Steering vectors are unreliable: up to 50% anti-steering, low cross-prompt cosine similarity [2505.22637]
- **Limitation**: Single directions cannot capture full literary characters (43.6M LoRA params vs 4096-dim vector)

### 2.2 Subspace-Constrained LoRA
- OPLoRA: double-sided orthogonal projections via SVD prevent catastrophic forgetting [2510.13003]
- SC-LoRA: SVD-derived initialization constrains updates orthogonal to preserved knowledge [2505.23724]
- Universal weight subspace hypothesis: LoRA weights collapse into low-rank architecture-specific subspaces [2512.05117]
- **Gap**: Applied to task performance preservation, never to character/personality training

### 2.3 Personality in Weight Space
- Personality vectors via weight differences between fine-tuned models; TIES-Merging for multi-trait composition [2509.19727]
- BiPO: preference-optimized steering vectors, transferable across models [2406.00045]
- Personality subnetworks: training-free persona extraction via activation-guided pruning [2602.07164]
- SAE-SSV: sparse autoencoder subspace steering with interpretable latent dimensions [2505.16188]
- MSRS: orthogonal multi-attribute subspaces with dynamic token-level steering [2508.10599]
- **Gap**: All static — extract or merge once. None track personality emergence over training.

### 2.4 Personality-Capability Entanglement
- Psychometric personality shaping modulates both capabilities and safety independently [2509.16332]
- Activation engineering at Layer 18 induces personality traits while preserving reasoning in Llama 3 8B [2412.10427]
- **Our contribution**: We quantify this entanglement by decomposing LoRA updates into personality-subspace and orthogonal-complement components at each training step.

## 3. Method

### 3.1 Personality Subspace Construction
1. Select N≥8 diverse persona archetypes (Skippy, default assistant, pirate captain, professor, drill sergeant, therapist, valley girl, cold robot)
2. Run M=50 shared prompts through base model with each persona's system prompt
3. Extract activation centroids per persona per layer (layers 9–26)
4. PCA on centroid matrix → K-dimensional personality basis P ∈ R^{K×hidden_dim}, K≈6 explains >99% inter-persona variance

### 3.2 GRPO Training with Delta Logging
- Base: Qwen3-VL-8B-Instruct with partial LoRA merge (α=0.7) for character prior
- GRPO with 8 generations/prompt, temperature 1.5, β=0.04 KL penalty
- Reward: heuristic personality markers + coherence + identity checks
- At each checkpoint (every 25 steps):
  - Extract LoRA delta Δ_m = B_m · A_m for each module m
  - Compute personality projection: Δ_m^{pers} = P · Δ_m (output-side) or Δ_m · P^T (input-side)
  - SVD of projected delta → character singular values
  - Energy ratio: ||Δ_m^{pers}||² / ||Δ_m||² — fraction of update in personality subspace
  - Log full trajectory as time series

### 3.3 Character Matrix
The "character matrix" for module m at step t is the SVD decomposition of the projected delta:

    Δ_m^{pers}(t) = U_m(t) · Σ_m(t) · V_m(t)^T

where Σ captures which personality dimensions are being modified and by how much. The trajectory {Σ_m(t)} across steps reveals how GRPO distributes personality learning across layers and modules.

## 4. Experiments

### 4.1 Baseline: SFT Catastrophic Forgetting
| Method | Banal Skippy | AIME % | Notes |
|--------|-------------|--------|-------|
| Base + system prompt | 4.62 | 46.7 | AI-assistant-ish |
| LoRA 0.5 merge + prompt | 5.43 | 33.3 | Best static config |
| LoRA SFT r=64 α=0.5 | 5.67 | 0 | Personality ↑, reasoning dead |
| LoRA SFT r=16 + math data | — | 0 | Math mixing doesn't help |
| Delta LoRA α=0.7 | 5.33 | 20 | Phase transition, not linear |

### 4.2 Rotation Steering (2D Givens)
| θ | Banal Δ | AIME % | Notes |
|---|---------|--------|-------|
| 15° | +0.36 | 33.3 | Preserved |
| 22.5° | +0.41 | 13.3 | Cliff |

Modest gains. Personality is higher-dimensional than a 2D plane.

### 4.3 GRPO V3 with Delta Logging
[PENDING — V3 training in progress, step ~200/318]

Key metrics to report:
- Reward trajectory (personality, coherence, identity components)
- personality_energy_ratio per layer over training steps
- Which modules (attention vs MLP) concentrate personality learning
- Whether character matrices show interpretable structure (low-rank? layer-specific?)

### 4.4 Character Matrix Analysis
[PENDING]
- Do personality energy ratios increase over training? (GRPO learning character)
- Are they concentrated in specific layers? (personality localization)
- Can we replay a trajectory at different speeds? (character intensity control)
- Can we rotate/mutate trajectories to produce new characters? (character transfer)

## 5. Discussion

### 5.1 What Would Be Novel
- First demonstration of interpretable personality trajectory tracking during RL-based character training
- Character matrices as compact, transferable personality representations
- Evidence for/against personality-reasoning separability in weight space during GRPO

### 5.2 Proposed Extension: OPLoRA-Constrained GRPO
Add orthogonal projection [2510.13003] to GRPO's LoRA: compute top-k SVD of frozen weights, project LoRA updates into the orthogonal complement. This mathematically guarantees reasoning preservation while allowing personality learning. Compare unconstrained GRPO (current) vs constrained.

### 5.3 Proposed Extension: Subnetwork Extraction Comparison
Compare GRPO-trained character against training-free personality subnetwork extraction [2602.07164]. If pruning-based masks achieve comparable character quality, the LoRA training is unnecessary overhead.

## 6. References

- [2511.03738] Activation-Space Personality Steering: Hybrid Layer Selection
- [2406.00045] BiPO: Personalized Steering via Bi-directional Preference Optimization
- [2508.10599] MSRS: Adaptive Multi-Subspace Representation Steering
- [2510.13003] OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting
- [2509.16332] Psychometric Personality Shaping Modulates Capabilities and Safety
- [2602.07164] Your Language Model Secretly Contains Personality Subnetworks
- [2412.10427] Identifying and Manipulating Personality Traits via Activation Engineering
- [2509.19727] Personality Vector: Modulating Personality by Model Merging
- [2505.16188] SAE-SSV: Supervised Steering in Sparse Representation Spaces
- [2505.23724] SC-LoRA: Subspace-Constrained LoRA
- [2505.22637] Understanding (Un)Reliability of Steering Vectors
- [2512.05117] The Universal Weight Subspace Hypothesis
