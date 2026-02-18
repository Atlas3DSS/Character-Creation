# Skippy Character Steering — Status Report (Feb 17, 2026)

## Executive Summary

We are testing whether our neuron-guided push/pull framework — which achieved 38% sarcastic personality on Qwen3-VL-8B (R5) — can "oneshot" skippify OpenAI's GPT-OSS-20B, a brand new open-weight MoE model.

**Current best Qwen model**: SDFT R5 scale 1.0 — 38% sarcastic (7/19 prompts), 0% assistant leaks, 100% identity, 100% math.

**Hypothesis**: If the neuron-guided framework transfers from dense 8B → MoE 20B in a single training run, the method is architecture-agnostic and can be productized for general character creation.

## Approach Evolution

| Round | Method | Result | Key Learning |
|-------|--------|--------|-------------|
| v1-v4 | Contrastive/activation vectors | Failed | Single vectors too low-dim for personality |
| LoRA SFT | Standard SFT | 0% AIME | Catastrophic forgetting — personality and reasoning share weight space |
| SDFT R1 | LoRA partial merge | 5.4/10 | Scale 0.5 sweet spot, authentic voice |
| SDFT R2 | Claude gold-standard data | 6.5/10 | 2K Opus responses, no-prompt floor doubled |
| DPO R1/R2 | Direct Preference Optimization | 40% AIME | Surface mimicry only |
| GRPO V3/V4 | Reward-guided policy optimization | Mixed | Avoids catastrophic forgetting but unstable |
| **R4** | **Neuron-guided push/pull (math-only profiling)** | **19% sarcastic** | Math neurons ≠ personality neurons |
| **R5** | **Neuron-guided push/pull (multi-category profiling)** | **38% sarcastic** | 2× improvement from broader profiling |
| **R6 (NOW)** | **Transfer to GPT-OSS-20B (MoE)** | **In progress** | Testing architecture transfer |

## R5 Results (Qwen3-VL-8B, Current Best)

### Training
- Base: R4 merged scale 1.0
- Data: 922 skippified (multi-category) + 682 antipole examples
- Profiled 12 layers across math, science, household, casual, confrontational, recipe categories
- Best checkpoint: Step 200 (epoch 2)

### Eval (scale 1.0, no system prompt)
| Metric | R4 | R5 | Improvement |
|--------|----|----|-------------|
| Identity (3/3) | 3/3 | 3/3 | — |
| Sarcastic | 19% | **38%** | **2×** |
| Assistant leaks | 5% | **0%** | Perfect |
| Math (4/4) | 4/4 | 4/4 | — |
| Avg personality score | 2.8/10 | **3.67/10** | +0.87 |

### Scale Comparison (R5)
| Scale | Personality | Math | Notes |
|-------|------------|------|-------|
| 0.5 | Higher sarcasm | **BROKEN** (315 instead of 345) | Too much personality pushes out reasoning |
| **1.0** | **38% sarcastic** | **4/4 correct** | **Sweet spot** |
| 1.5 | "I am an AI assistant named Qwen" | — | Catastrophic identity regression |

## GPT-OSS-20B Transfer Experiment (R6)

### Why GPT-OSS-20B
- 21B total params, 3.6B active per token (MoE, 32 experts, Top-4)
- Matches o3-mini on benchmarks — stronger base than Qwen3-VL-8B
- Native tool calling via `<tool_call>` tags
- Apache 2.0, text-only (no vision)
- Already cached locally (13.8 GB MXFP4)

### Architecture Comparison
| Spec | GPT-OSS-20B | Qwen3-VL-8B |
|------|-------------|-------------|
| Total params | 21B | 8.3B |
| Active params | 3.6B (MoE) | 8.3B (dense) |
| Layers | 24 | 36 |
| Hidden dim | 2,880 | 4,096 |
| Attention | 64 Q / 8 KV (GQA) | 32 Q / 8 KV (GQA) |
| Experts | 32 (Top-4) | N/A (dense) |
| Context | 128K | 32K |

### Adaptation Plan
1. Dequantize MXFP4 → bf16 via `Mxfp4Config(dequantize=True)`
2. LoRA on attention layers (q/k/v/o_proj) — expert MLPs are MXFP4 custom class
3. NeuronTracker hooks on decoder layer outputs (MoE output = combined expert output, same tensor shape)
4. Same push/pull regularization, same training data
5. Monitor layers: 11 of 24 (skip first 2, then every other)

### VRAM Budget (Pro 6000 96GB)
| Component | Memory |
|-----------|--------|
| GPT-OSS-20B dequantized bf16 | ~42GB |
| LoRA adapter (r=32) | ~0.2GB |
| Optimizer states | ~0.4GB |
| Activations + gradients | ~12GB |
| **Total estimated** | **~55GB** |
| Headroom | 41GB |

### Key Question
Can neuron-level push/pull regularization work across MoE expert boundaries? The decoder layer output is the combined output of all active experts, so the neuron tracker sees the aggregate. But personality may be distributed across expert routing patterns, not just neuron activations.

## What's Next After R6

### If Transfer Succeeds (personality ≥ 30% sarcastic on GPT-OSS)
→ Framework is architecture-agnostic. Productize:
1. Auto-profiling pipeline (character data → neuron scores)
2. Auto-training (push/pull with optimal hyperparameters)
3. Auto-eval (personality metrics + reasoning preservation)
4. Package as `character-baker` CLI tool

### If Transfer Fails
→ MoE requires different approach. Investigate:
1. Per-expert profiling instead of per-layer
2. Router-level steering (bias routing toward "personality experts")
3. Expert-specific LoRA targeting

### GRPO Tool-Call Training (deferred)
Plan saved in `next_steps_maybe.md`. Will train Skippy for high-quality tool calling with personality preservation after R6.

## Files

### Active Scripts
| File | Purpose |
|------|---------|
| `neuron_guided_training.py` | R5 neuron-guided framework (Qwen) |
| `train_gptoss_skippy.py` | R6 adaptation for GPT-OSS-20B (in progress) |
| `household_config.py` | System prompts, tools, household registry |
| `eval_aime.py` | AIME math benchmark via vLLM |

### Training Data
| File | Count | Description |
|------|-------|-------------|
| `contrastive_data/skippified_combined_r5.jsonl` | 922 | Multi-category Skippy responses |
| `contrastive_data/r5_assistant_antipole.jsonl` | 682 | Assistant-mode antipole examples |

### Models
| Path | Description |
|------|-------------|
| `skippy_sdft_r5/merged_scale_1.0/` | **Current best Qwen** — 38% sarcastic, 4/4 math |
| `skippy_sdft_r5/best_adapter/` | R5 LoRA adapter (step 200) |
| `skippy_gptoss/` | R6 GPT-OSS output (in progress) |
