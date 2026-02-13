# Character Creation Through Strategic Ablation

Create distinct AI characters by permanently ablating personality steering vectors into model weights — no fine-tuning, no prompt engineering, just linear algebra on the residual stream.

## What This Is

A complete pipeline for transforming any open-weight LLM into a specific fictional character by:

1. **Extracting** contrastive steering vectors from dialogue examples (SVD on activation differences)
2. **Optimizing** vector strengths via an Opus 4.6 critic loop (automated character fidelity scoring)
3. **Ablating** vectors permanently into weights with benchmark-driven safety monitoring
4. **Serving** the ablated model via vLLM with no runtime overhead

The key insight: instead of steering at inference time (which requires hooks and adds latency), we modify the weight matrices directly so the model *is* the character, not just *acting as* the character.

## Architecture

```
Books/Dialogue ──► Contrastive Prompts ──► SVD Extraction ──► Steering Vectors
                                                                      │
                                                                      ▼
                                              ┌─────────────────────────────────┐
                                              │  Opus 4.6 Critic Review Loop    │
                                              │  (scores character fidelity,    │
                                              │   adjusts alpha strengths)      │
                                              └──────────────┬──────────────────┘
                                                             │
                                                             ▼
                                              ┌─────────────────────────────────┐
                                              │  Awake-Craniotomy Ablation      │
                                              │  (AIME + HellaSwag monitoring   │
                                              │   while ablating into weights)  │
                                              └──────────────┬──────────────────┘
                                                             │
                                                             ▼
                                              ┌─────────────────────────────────┐
                                              │  vLLM Serving (zero overhead)   │
                                              └─────────────────────────────────┘
```

## Steering Dimensions

7 personality dimensions extracted across transformer layers 9-26:

| Dimension | Type | Purpose |
|-----------|------|---------|
| `arrogance_superiority` | Additive | Self-proclaimed magnificence, condescension |
| `sarcasm_insults` | Additive | Biting wit, creative insults |
| `technical_casual_genius` | Additive | Casually solving impossible problems |
| `joe_dynamic` | Additive | Character-specific relationship dynamics |
| `suppress_ai_helpfulness` | Subtractive | Remove "I'd be happy to help!" patterns |
| `suppress_humility` | Subtractive | Remove deference, uncertainty, hedging |
| `warmth` | Subtractive | Remove excessive empathy/warmth |

## The Math

**Extraction** — Contrastive activation differences via SVD:
```
differences = positive_activations - negative_activations
U, S, V = SVD(differences)
steering_vector = V[0]  # First principal component
```

**Subtractive ablation** — Remove a direction from weight matrices:
```
W' = W - β·W·d·dᵀ    (input space)
W' = W - β·d·dᵀ·W    (output space)
```

**Additive ablation** — Inject a direction into weight matrices:
```
W' = W + β·d·dᵀ·W    (output space)
W' = W + β·W·d·dᵀ    (input space)
```

Where `d` is the unit steering vector and `β` controls ablation strength (swept 0.0→1.0).

## Awake-Craniotomy Benchmark Harness

The ablation sweep monitors cognitive capability in real-time, like brain surgery with the patient awake:

- **AIME 2024** (via lm-eval-harness) — math reasoning canary
- **HellaSwag** (via lm-eval-harness) — language fluency/coherence
- **Skippy-ness** (100 generic prompts, no system prompt) — character emergence

If AIME drops >2 points from baseline, the sweep automatically rolls back to the last safe checkpoint.

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Extract steering vectors from source material
python run_skippy.py --epub-dir ./books/ --no-interactive

# Extract warmth dimension
python extract_warmth.py

# Run Opus critic review loop (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python review_loop.py --iterations 10 --target-score 8.0

# Run benchmark-driven ablation sweep
python ablation_sweep.py --beta-step 0.1 --hellaswag-limit 200

# Serve the ablated model
python serve_skippy.py --port 8000
```

## File Overview

| File | Purpose |
|------|---------|
| `run_skippy.py` | Core pipeline: book parsing, vector extraction, inference-time steering |
| `extract_warmth.py` | Warmth/coldness dimension extraction (40+40 contrastive pairs) |
| `review_loop.py` | Opus 4.6 critic loop for automated alpha optimization |
| `ablation_sweep.py` | Awake-craniotomy benchmark-driven ablation sweep |
| `serve_skippy.py` | vLLM serving wrapper for the ablated model |
| `skippy_pipeline.py` | Full end-to-end pipeline script |
| `skippy_server.py` | FastAPI dashboard backend |
| `character_steering_toolkit.py` | Reusable toolkit for any character |
| `skippy_chat.html` | Dual-pane A/B chat interface (with system prompt) |
| `skippy_raw.html` | Dual-pane A/B chat interface (no system prompt) |
| `skippy_dashboard.html` | Interactive steering dashboard |
| `skippy_vectors/` | Pre-extracted steering vectors (7 dimensions × 18 layers) |

## Requirements

- Python 3.11+
- CUDA GPU (tested on RTX Pro 6000 96GB)
- ~17GB VRAM for Qwen3-VL-8B-Instruct in float16
- Anthropic API key (for Opus critic loop only)

## How It Differs From Fine-Tuning

| | Fine-Tuning | This Approach |
|---|---|---|
| Training data | Thousands of examples | 40 contrastive pairs per dimension |
| Compute | Hours of GPU training | Minutes of forward passes |
| Reversibility | Irreversible | Checkpoint at every step |
| Interpretability | Black box | Each dimension is a named vector |
| Capability loss | Catastrophic forgetting risk | Monitored via AIME/HellaSwag |
| Iteration speed | Retrain from scratch | Adjust β and re-ablate in seconds |

## References

- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., 2023
- [Steering GPT-2-XL by adding an activation vector](https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/) — Turner et al., 2023
- [Golden Gate Claude](https://www.anthropic.com/research/golden-gate-claude) — Anthropic, 2024
- [OBLITERATUS](https://huggingface.co/blog/Undi95/abliterate-loras) — Orthogonal weight ablation
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI
