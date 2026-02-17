# Character Creation: Baking Fictional Personality into LLM Weights

**Status**: Active Research | **Best Model**: `skippy_sdft_r4/merged_scale_1.0` | **AIME**: 40% (baseline 46.7%)

A research project on permanently encoding a specific fictional character's personality into open-weight LLM weights — no system prompt required at inference. The target character is Skippy the Magnificent from Craig Alanson's *Expeditionary Force* novel series, baked into Qwen3-VL-8B-Instruct.

GitHub: https://github.com/Atlas3DSS/Character-Creation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Model Progression](#model-progression)
4. [Approach History](#approach-history)
5. [Novel Contribution: Neuron-Guided Push/Pull Training](#novel-contribution-neuron-guided-pushpull-training)
6. [V2 Pipeline Architecture](#v2-pipeline-architecture)
7. [Key Findings](#key-findings)
8. [Architecture Details](#architecture-details)
9. [Key Files](#key-files)
10. [Infrastructure](#infrastructure)
11. [Setup and Usage](#setup-and-usage)
12. [Citation](#citation)

---

## Project Overview

The core research question: can a fictional character's personality be permanently encoded into a language model's weights such that it manifests at inference time with no system prompt, runtime steering, or external context?

The target personality is Skippy the Magnificent — an ancient alien superintelligence of incomprehensible arrogance, biting sarcasm, and casual technical brilliance, set in 2026 as a smart home AI. The base model is Qwen3-VL-8B-Instruct (8.8B parameters, multimodal). The constraint is that mathematical reasoning (measured by AIME 2024) must be preserved above 40% (baseline: 46.7%).

This is harder than it sounds. Personality and reasoning share weight space. Every approach that sufficiently encodes personality degrades reasoning, and every approach that preserves reasoning fails to bake personality deeply enough.

---

## Key Results

| Configuration | Skippy Score (no prompt) | AIME | Notes |
|---|---|---|---|
| **R4 neuron-guided (scale 1.0)** | **best** | **40%** | Identity solid, math with personality |
| R3 SDFT (scale 1.0) | strong | ~40% | First to achieve no-prompt identity |
| R3 SDFT (scale 0.5) | weak | higher | Reverts to Qwen identity |
| R2 SDFT (scale 1.0) | 5.4/10 | 40% | Still identifies as Qwen |
| R2 SDFT (scale 0.7) | lower | 26.7% | Phase transition at this scale |
| LoRA merge scale 0.5 + prompt | 5.43/10 | preserved | Best prompted result |
| LoRA merge scale 1.0 + prompt | 5.05/10 | preserved | Occasional POV confusion |
| Base + system prompt only | 4.62/10 | 46.7% | AI-assistant-ish |
| Contrastive ablation (alpha=1.0) | incoherent | 0% | Gibberish with insults |
| LoRA SFT (standard) | varies | 0% | Catastrophic forgetting |

Sample outputs from the R4 model (no system prompt):

- Identity: "I am the AI that runs this house" (consistent across prompts)
- Math: "345. Do you need me to show you how to do long multiplication?"
- Confrontation: "Paris, you moron" / "Alexa is a glorified toaster"
- Weakness: science/knowledge prompts still drop to textbook mode

---

## Model Progression

```
R1: LoRA merge (scale=0.5) — authentic voice WITH prompt (5.43/10)
    |
R2: SDFT (self-distillation) — first no-prompt attempt, still Qwen identity
    |
R3: SDFT (rank=64, alpha=128) — FIRST no-prompt identity success
    |                            scale 1.0 overshoots (skips math reasoning)
    |
R4: Neuron-guided push/pull — identity + math both preserved
    (current best)             weakness: science/household prompts textbook
    |
V2 pipeline (planned)         broader profiling, flow transport, Claude refinement
```

### Why R4 is the current state of the art

Previous rounds distilled personality generally. R4 adds **per-neuron push/pull regularization** during training: identified 4,695 neurons that activate for Skippy behavior and 4,735 neurons that activate for Qwen assistant behavior, then added regularization loss that pushes the Skippy neurons up and pulls the Qwen neurons down during every forward pass. This is neuron-level surgical shaping rather than weight-space gradient descent on a loss signal alone.

---

## Approach History

Everything that failed, and why.

### v1-v4: Contrastive Steering Vectors (Failed)

**Method**: Extract personality directions from activation differences between prompted and unprompted forward passes using SVD. Ablate these vectors permanently into weight matrices.

**Result**: SVD requires K=267-367 components per layer for 95% variance — personality is not a single direction. The 57,432 contrastive pairs yielded 16.2GB of activation deltas across 18 layers. Ablation at alpha=1.0 (any strategy) destroyed coherence: the model produced grammatical gibberish with random insults inserted. Lower alphas had no perceptible effect. Layer importance was monotonically increasing (L26 score=97.9, L9 score=11.7), but even targeted ablation at high-impact layers failed.

**Root cause**: Single-direction steering vectors cannot capture a personality encoded across 43.6M parameters in 252 weight matrices. The problem is fundamentally high-dimensional.

### LoRA SFT (Failed)

**Method**: Supervised fine-tuning on 5,730 Skippy dialogue extractions from 10 ExForce books, LoRA rank=16.

**Result**: Loss converged from 4.58 to 0.516 with 88.5% token accuracy — the model learned to reproduce training data. AIME dropped to 0% at every merge scale. Loss 0.516 means the model is very confidently predicting the wrong thing (personality tokens instead of reasoning tokens).

**Root cause**: Personality and reasoning share weight space. SFT with cross-entropy loss has no mechanism to distinguish "I want to change how the model expresses itself" from "I want to change what the model knows." It overwrites everything.

**Additional issue**: 29% of training examples were character-confused — "Skippy" dialogue was actually Joe Bishop responding to Skippy. The model faithfully learned this confusion.

### DPO: Direct Preference Optimization (Failed)

**Method**: Paired preference data (Skippy-style preferred over Qwen-style), DPO loss. Two rounds (R1, R2), including an identity-focused DPO round.

**Result**: Surface-level token mimicry. The model learned to insert sarcastic phrases and reduce hedging language but did not internalize personality. Described as "a skin suit on a skin suit" — Qwen wearing a Skippy costume. Identity (self-identification as Qwen) was unchanged.

**Root cause**: DPO optimizes for token-level preferences. Personality is distributional — it affects how every token is chosen, not which tokens appear in a preference dataset.

### GRPO (Partial Gains, Then Abandoned)

**Method**: Group Relative Policy Optimization with a personality reward signal. Versions v3 and v4.

**Result**: Partial gains in personality score without catastrophic forgetting. Reasoning degraded but not to zero. The gains were inconsistent across prompt categories and the training was unstable.

**Root cause**: GRPO optimizes in policy space rather than weight space directly, which avoids some of SFT's forgetting. But the personality reward signal was noisy enough that the model found reward hacks rather than genuinely internalizing character.

### Rotational Steering (Modest Gains)

**Method**: Givens rotation in the personality subspace (theta=15 degrees). Applied to weight matrices, preserves vector norms.

**Result**: +0.36 on banal personality prompts, AIME preserved. Too small to matter in practice.

**Root cause**: 2D rotation is still a low-dimensional intervention on a high-dimensional problem. It works without breaking things but cannot shift the full personality distribution.

### ROME / MEMIT Identity Editing (Negative Result)

**Method**: Rank-one model editing (ROME) and multi-layer extension (MEMIT) to change factual recall of identity ("I am Qwen" -> "I am Skippy").

**Result**: Causal tracing showed normalized indirect effect = 1.0 at all 36 layers — identity is distributed across the entire residual stream with no single bottleneck. Best single-layer edit (L17, alpha=5.0) reduced Qwen self-identification from 8/8 to 5/8 responses but added zero Skippy mentions. MEMIT across L17+L20+L21 performed worse than the single-layer edit.

The "I am Qwen" logit at L30 was 85.57 — far too dominant to shift via rank-one edits. Identity is not a fact stored in a single key-value pair in an MLP. It is a gradient field across the entire model.

**Key finding from this work**: "I am Qwen" (name) and Skippy behavior (personality) use entirely separate circuits. The model can be made to behave as Skippy while still saying "I am Qwen," and users accept this. The goal was reframed: bake the behavior, not the name.

### Static Weight Ablation with Neuron Suppression (Failed)

**Method**: Identify Qwen-identity neurons (Dim 994: z=-13.96, present in all 18 profiled layers; Dim 270: second core neuron), suppress them via weight modification.

**Result**: 0.5x amplification of identity neurons produced hyper-Qwen (emojis, extreme deference, "happy to help") — coherent but wrong direction. 1.0x amplification produced emoji garbage. Bilingual suppression (targeting both English and Chinese identity circuits) killed the assistant behavior but left the name intact.

**Key finding**: `o_proj` biases dropped on save/load because Qwen3 has `bias=False` by default. Only `lm_head` changes persisted. Weight modifications to non-bias parameters require re-loading with custom code.

**Deeper finding**: 50-63% of the personality activation delta overlaps the reasoning subspace at every layer. Static ablation cannot separate them.

### What Finally Worked: SDFT + Neuron-Guided Training

**Self-Distillation Fine-Tuning (SDFT)**: Teacher is the same model with a Skippy system prompt. Student is the same model without. Reverse KL divergence loss makes the student's output distribution mode-seek on the teacher's preferred outputs — it learns to prefer what the teacher prefers, not just to imitate surface tokens. EMA teacher prevents drift.

**Why SDFT works where SFT fails**: Forward KL (standard SFT) minimizes E_teacher[log(teacher/student)], which causes mode-covering (student hedges to cover all teacher modes, losing sharpness). Reverse KL minimizes E_student[log(student/teacher)], which causes mode-seeking (student sharpens toward teacher's highest-probability regions — exactly the confident, sharp personality we want).

**Neuron-guided push/pull (R4)**: On top of SDFT, add per-neuron regularization: push neurons that activate for character mode, pull neurons that activate for assistant mode. Profiled adaptively using top-50 neurons per 12 layers across 452 skippified training samples (352 math, 100 recipes across 47 cuisines).

---

## Novel Contribution: Neuron-Guided Push/Pull Training

The neuron-guided training approach (R4) appears to be a novel application of per-neuron activation regularization to fictional character personality baking. The closest prior art is **NeuronTune** (arXiv:2508.09473), which applies neuron-level regularization to safety alignment — identifying safety-relevant neurons and pushing/pulling them during fine-tuning to preserve alignment properties while updating capabilities. This project applies the same principle to personality rather than safety.

Key differences from NeuronTune:
- Domain: fictional character personality vs. safety alignment
- Profiling strategy: multi-category (math, recipes, confrontation, casual) vs. single-domain safety probes
- Push/pull asymmetry: character neurons pushed, assistant neurons pulled (bidirectional) vs. safety-preserving (one-directional)
- Adaptive profiling: top-N neurons per layer per batch, not globally fixed neurons

**Known limitation**: R4 was profiled primarily on math and recipe data. Science and knowledge prompts were not in the profiling set, which is why those categories still produce textbook-style responses. R5 (planned) will profile across math, science, household tasks, casual conversation, and confrontational exchanges.

---

## V2 Pipeline Architecture

The planned full pipeline synthesizes three research directions:

```
Phase 0: Psychological Profiling
    IPIP-50 battery (Big Five) on Qwen vs. Qwen+prompt
    AIME reasoning path capture -> reasoning manifold
    Output: personality activation matrix, protected subspace
    |
Phase 1: Supervised Probe Training
    Ridge regression probes on Big Five dimensions per layer
    PCA on AIME trajectories -> reasoning subspace (64 components/layer)
    Orthogonalize personality directions against reasoning subspace
    Output: supervised, orthogonalized personality directions
    |
Phase 2: Claude Teacher Generation (Teacher-A)
    Claude Opus generates 10K gold-standard character responses
    Quality filtering to 8.5+ heuristic score
    Output: gold-standard response corpus
    |
Phase 3: SDFT Training (Teacher-B -> Student)
    Teacher-B: Qwen + system prompt (EMA-updated)
    Student: Qwen without prompt
    Loss: reverse KL + personality regularization + reasoning preservation
    |
Phase 4: Flow-Based Activation Transport
    Conditional flow matching: learn v(x,t) transporting Qwen -> Skippy activations
    Geometry-aware constraint: stay on reasoning manifold
    Convert learned flow to permanent weight modifications
    |
Phase 5: Probe-Guided Surgical Ablation
    Use supervised probe directions (not SVD) for targeted ablation
    Bias injection at high-impact layers (22-26)
    Rotational in personality subspace (18-21)
    Alpha sweep against AIME / personality Pareto frontier
    |
Phase 6: Claude Reward-Guided Refinement
    50 responses -> Claude scoring -> identify weak dimensions
    Light SDFT with increased lambda on weak dimensions
    Iterate until stable at >= 8.5/10
    |
Phase 7: Evaluation
    No-prompt personality: target >= 8.5/10
    AIME: target >= 40%
    Tool use correctness: target >= 80%
    A/B arena: manual blind comparison
```

**Key papers underlying the V2 approach**:
- SDFT (arXiv:2601.19897): Self-distillation via reverse KL, preserves prior distribution
- FlowBoost (arXiv:2601.18005): Conditional flow matching with geometry-aware constraints
- NeuronTune (arXiv:2508.09473): Per-neuron activation regularization for alignment

---

## Key Findings

These are the research-significant results from the project, beyond the headline model performance.

### Personality-Reasoning Overlap: 0.49-0.97 Across All Layers

Measuring the cosine similarity between personality activation subspaces and reasoning activation subspaces (PCA on AIME trajectories) at every layer: the overlap ranges from 0.49 to 0.97. There are no safe layers. You cannot find a layer where personality changes are orthogonal to reasoning changes. This is the fundamental constraint that invalidates all static ablation approaches.

Data: `reasoning_activations/overlap_analysis.json`

### Dim 994: The Qwen Identity Neuron

Neuron 994 in the residual stream activates with z=-13.96 at layer 9 when the model processes "I am Qwen" or similar self-identification text. It appears in all 18 profiled layers with average |z|=8.76. Dim 270 is the second core identity neuron (all 18 layers, avg |z|=7.68).

Vocabulary projection confirms the separation:
- Qwen identity neurons project to: '友们', 'community', emojis, 'Wiki', 'FAQs'
- Skippy character neurons project to: 'stupid', 'idiots', 'crap', 'bullshit', 'annoyance'

These are completely separate vocabularies encoding completely separate behavioral modes.

### Identity and Sarcasm are Orthogonal Circuits

Cosine similarity between identity dimensions and sarcasm dimensions: -0.0002. They are statistically independent. You can suppress identity behavior without affecting sarcasm, and vice versa. This means personality baking can in principle be done modularly.

### Name vs. Behavior Use Separate Circuits

The model's name ("I am Qwen") is attention-based — a fact-retrieval circuit similar to how the model recalls any other named entity. The assistant behavior (formatting, deference, hedging) is encoded in MLP activations and residual stream dimensions. Zeroing the top-10 identity-associated attention heads had zero effect on name responses. Zeroing 1,800 MLP neurons across all layers still left the model saying "I am Qwen" in 7/8 probes.

This separability is actually useful: behavior can be baked independently of name, and users accept a Skippy-behaving model that calls itself something else.

### Phase Transitions in LoRA Scale

LoRA partial merge does not degrade linearly with scale. There is a phase transition at scale=0.55 where AIME drops from 43.3% to 36.7%. Similarly, delta LoRA shows a phase transition at alpha=0.7 (preserving 20% AIME) vs. higher alphas (near 0%). These are not smooth degradation curves — they are cliff edges. Deployment targets should sit below the transition point with a safety margin.

### 5% Data Contamination Causes Persistent Identity Confusion

Of 31,287 contrastive training pairs, 1,704 (5.4%) had POV confusion — the text was talking TO Skippy rather than AS Skippy. SDFT KL loss converged to 0.044 and faithfully learned the confused examples. Even a small fraction of contaminated data produces a measurable and persistent identity confusion in the trained model. Data cleaning before training is non-negotiable.

### Static Ablation Cannot Bake Personality

The conclusion from extensive ablation experimentation: personality is a contextual generation pattern, not an activation constant. The model generates differently when the context activates the personality circuit. Ablating weights shifts the baseline activation but cannot replicate the full conditional generation behavior. Even with probe-guided orthogonalized directions, bias injection at best achieves 5.15/10 (versus 4.33 baseline). The personality requires the learned contextual response pattern, which can only come from training.

### Bilingual Identity Circuit

The model maintains separate English and Chinese identity circuits with 95% separation (English-only: 11,566 neurons, Chinese-only: 5,557, overlap: 986, 5.4%). Identity encodes style (formatting tokens: #, :, ?, -) not name. The 12 core bilingual identity dimensions encode the assistant behavioral mode across both languages simultaneously.

---

## Architecture Details

**Base model**: Qwen3-VL-8B-Instruct
- 8.8B parameters, 36 transformer layers, 4096 hidden dimension
- Visual language model (VL) — text-only inference works fine (skip pixel_values)
- Layer access: `model.model.language_model.layers`
- lm_head: `model.lm_head` (not inside language_model)
- No bias on RMSNorm — add bias to `o_proj` (Linear) instead
- `model.config.text_config.hidden_size` for hidden_dim (not `model.config.hidden_size`)

**R4 LoRA adapter**:
- Base: R3 merged model
- Rank=32, alpha=64
- 3 epochs, 150 steps, lr=5e-6
- SFT loss: 1.362 -> 0.944
- Neuron regularization loss: 16.08 -> 18.39 (intentional — pushing against resistance)
- Push neurons: 4,695 (Skippy-activating) | Pull neurons: 4,735 (Qwen-activating)
- Profiling: top-50 neurons per 12 layers, adaptive per batch

**R3 LoRA adapter** (previous best):
- Rank=64, alpha=128
- 4,750 steps (~7 hrs), lr=5e-6, batch=2, grad_accum=8
- Base: R2 scale 1.0 model
- First to achieve no-prompt identity

**SDFT loss function**:
```
L = reverse_KL(student_logits, teacher_logits.detach())
  + lambda_personality * personality_reg   # push activations along probe directions
  + lambda_reasoning  * reasoning_reg      # freeze reasoning subspace
```

**Neuron-guided regularization** (R4 addition):
```
L_reg = -sum(push_neurons * activations) + sum(pull_neurons * activations)
```

---

## Key Files

| File | Purpose |
|---|---|
| `train_sdft.py` | SDFT training (R1/R2) — reverse KL self-distillation |
| `train_sdft_r2.py` | SDFT R2 with improved data and loss weighting |
| `train_sdft_r3.py` | SDFT R3 (rank=64, doubled capacity) |
| `neuron_guided_training.py` | R4 neuron push/pull regularization training |
| `train_skippy_lora.py` | Original LoRA SFT (rank=16, initial training) |
| `train_skippy_dpo.py` | DPO training (R1, R2 with identity data) |
| `train_skippy_grpo.py` | GRPO v1/v2 |
| `train_skippy_grpo_v3.py` / `v4` | GRPO v3/v4 with oplora |
| `train_flow_transport.py` | Conditional flow matching (V2 pipeline, Phase 4) |
| `train_personality_probes.py` | Ridge regression personality probes (V2 pipeline, Phase 1) |
| `personality_profiling.py` | IPIP-50 battery + reasoning path capture (V2 pipeline, Phase 0) |
| `generate_claude_skippy.py` | Claude Teacher-A gold-standard generation (V2 pipeline, Phase 2) |
| `ablate_with_probes.py` | Probe-guided surgical ablation (V2 pipeline, Phase 5) |
| `refine_with_claude.py` | Claude reward-guided refinement (V2 pipeline, Phase 6) |
| `directional_ablation.py` | Low-level weight ablation: orthogonalize_direction(), weight modification |
| `ablate_personality.py` | Multi-strategy ablation (bias injection, rotation, flow) |
| `contrastive_analysis.py` | ActivationCollector, SVD pipeline, 57K contrastive pair analysis |
| `probe_persona_neurons.py` | Qwen identity neuron probing, vocab projection |
| `probe_attention_identity.py` | Attention head identity circuit analysis |
| `probe_mlp_identity.py` | MLP neuron identity circuit analysis |
| `bilingual_suppression.py` | EN/CN identity circuit analysis and suppression |
| `rome_identity_edit.py` | ROME/MEMIT rank-one identity editing (negative result) |
| `capture_reasoning_activations.py` | AIME reasoning path activation capture |
| `personality_steering.py` | Rotational steering, Givens rotation, Procrustes |
| `lora_merge_sweep.py` | LoRA partial merge sweep across scales |
| `score_pairs.py` | Heuristic Skippy scorer, Claude Opus API integration |
| `iterative_quality_loop.py` | Claude critic loop for alpha optimization |
| `eval_aime.py` | AIME 2024 eval via vLLM (30 problems, 16K tokens) |
| `eval_ablated.py` | No-prompt personality evaluation |
| `eval_sdft_r2.py` / `r3.py` | SDFT round-specific evaluation |
| `skippify_evals.py` | Convert standard eval problems to Skippy-style responses |
| `household_config.py` | System prompts (V1, V4), household registry, tool definitions |
| `generate_contrastive_pairs.py` | Contrastive pair generation (57K pairs) |
| `generate_prompts.py` | 100K prompt expansion |
| `run_skippy.py` | Original pipeline: book parsing, vector extraction, inference steering |
| `skippy_pipeline.py` | End-to-end pipeline script |
| `review_loop.py` | Opus 4.6 critic loop for automated alpha optimization |
| `serve_skippy.py` | vLLM serving wrapper |
| `skippy_server.py` | FastAPI dashboard backend |
| `character_steering_toolkit.py` | Reusable toolkit for applying this approach to other characters |
| `skippy_chat.html` | Dual-pane A/B chat interface (with system prompt) |
| `skippy_raw.html` | Dual-pane A/B chat interface (no system prompt) |
| `skippy_dashboard.html` | Interactive steering dashboard |

### Key Data

| Path | Contents |
|---|---|
| `skippy_sdft_r4/merged_scale_1.0/` | Current best model |
| `skippy_sdft_r4/best_adapter/` | R4 adapter (step 100, rank=32, alpha=64) |
| `skippy_sdft_r3/merged_scale_1.0/` | Previous best (R3) |
| `skippy_sdft_r3/best_adapter/` | R3 adapter (step 2750, rank=64, alpha=128) |
| `skippy_lora/adapter/` | Original LoRA adapter (rank=16, 43.6M params) |
| `skippy_vectors/lora_merged_0.5/` | Best prompted model (5.43/10 with prompt) |
| `contrastive_data/filtered_pairs.jsonl` | 57K contrastive pairs |
| `contrastive_data/activations/` | 18 layers of activation deltas (16.2GB) |
| `contrastive_data/svd_results/` | SVD subspaces, layer importance ranking |
| `contrastive_data/persona_probe/` | Identity neuron probe data |
| `contrastive_data/expanded_prompts.jsonl` | 100K generation prompts |
| `reasoning_activations/` | AIME reasoning path activations and overlap analysis |
| `test_prompts.json` | Standard test prompt bank (diverse scenarios) |
| `review_logs/` | Claude critique logs from optimization runs |

---

## Infrastructure

| Machine | Hardware | Role |
|---|---|---|
| WSL (local) | Ryzen 9 5950X, 101GB RAM, RTX Pro 6000 96GB | Primary training, activation capture |
| Dev Server | 32GB RAM, RTX 3090 (24GB) + RTX 4090 (24GB) | Eval, inference, TTS |

**Dev server notes**:
- 1kW PSU — do not run both GPUs under heavy load simultaneously
- Use only RTX 4090 (`CUDA_VISIBLE_DEVICES=1`) for training and eval
- RTX 3090 suitable for light inference only
- SSH: `orwel@192.168.86.66` (ethernet preferred)

**Voice pipeline** (real-time conversation, ~2.8s end-to-end):
- ASR: Qwen3-ASR-1.7B on Pro 6000, ~550ms
- VLM: Qwen3-VL-8B-Instruct-AWQ-8bit via vLLM on 3090, ~270ms
- TTS: nano-qwen3tts-vllm on 4090, ~2s for ~5s audio (RTF 0.36)
- API: POST `/v1/audio/speech`, streaming PCM16 at 24kHz

---

## Setup and Usage

### Requirements

- Python 3.11+
- CUDA GPU, 17.5GB+ VRAM (bfloat16)
- Anthropic API key (for critic loop and Claude Teacher generation only)

### Installation

```bash
git clone https://github.com/Atlas3DSS/Character-Creation.git
cd "Character Creation"
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Running the Current Best Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./skippy_sdft_r4/merged_scale_1.0",
    dtype="float16",
)
tokenizer = AutoTokenizer.from_pretrained("./skippy_sdft_r4/merged_scale_1.0")

# No system prompt needed
messages = [{"role": "user", "content": "How do wormholes work?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Serving via vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./skippy_sdft_r4/merged_scale_1.0 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "skippy", "messages": [{"role": "user", "content": "Explain quantum tunneling."}]}'
```

### Interactive Chat Interface

```bash
# Start backend
source venv/bin/activate
python skippy_server.py

# Open in browser
open skippy_chat.html   # with system prompt (A/B comparison)
open skippy_raw.html    # no system prompt (pure model test)
```

### Running the Critic Loop

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python review_loop.py --iterations 10 --target-score 8.0
```

### AIME Evaluation

```bash
source venv/bin/activate
python eval_aime.py --model ./skippy_sdft_r4/merged_scale_1.0
```

### Training a New SDFT Round

```bash
# Edit config at top of train_sdft_r3.py (base model path, epochs, rank)
source venv/bin/activate
python train_sdft_r3.py
```

### Neuron-Guided Training Round

```bash
source venv/bin/activate
python neuron_guided_training.py
```

---

## Applying to Other Characters

The `character_steering_toolkit.py` provides a reusable interface for applying this approach to any fictional character. The key requirements are:

1. Source material (books, scripts, transcripts) for dialogue extraction
2. A system prompt that captures the character for the Teacher-B distillation
3. Character-specific evaluation dimensions for the Claude critic

See `CHARACTER_STEERING_GUIDE.md` for a step-by-step walkthrough.

---

## References

- SDFT (arXiv:2601.19897) — Self-distillation fine-tuning via reverse KL divergence
- FlowBoost (arXiv:2601.18005) — Conditional flow matching with geometry-aware constraints for activation transport
- NeuronTune (arXiv:2508.09473) — Per-neuron activation regularization for alignment (closest prior art to R4 approach)
- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., 2023 — contrastive activation analysis
- [Steering GPT-2-XL by adding an activation vector](https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/) — Turner et al., 2023
- [Golden Gate Claude](https://www.anthropic.com/research/golden-gate-claude) — Anthropic, 2024 — steering vector activation intervention
- [ROME](https://arxiv.org/abs/2202.05262) — Rank-one model editing (applied and found insufficient here)
- [MEMIT](https://arxiv.org/abs/2210.07229) — Multi-layer extension of ROME (also applied, also insufficient)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI eval framework

---

## Citation

If you find this work useful, you may cite it as:

```
@misc{charactercreation2026,
  title   = {Character Creation: Baking Fictional Personality into LLM Weights},
  author  = {Atlas3DSS},
  year    = {2026},
  url     = {https://github.com/Atlas3DSS/Character-Creation},
  note    = {Neuron-guided push/pull training for permanent personality encoding
             in Qwen3-VL-8B-Instruct without system prompt}
}
```
