# Character Creation

Research code and artifacts for studying whether an open-weight language model can be moved from prompt-level persona imitation toward durable changes in generated reasoning and voice.

The repository began as a fictional-character personality baking project and later expanded into a cleaner public-domain test bed using Supreme Court opinion style and legal-reasoning contrasts. The current emphasis is mechanistic: activation probes, causal patching, steering attempts, SAE feature inspection, and evaluation harnesses that distinguish decodable style from actually controllable generation behavior.

## Current Status

This is an active research workspace, not a packaged library.

The strongest honest result so far is:

- Personality, source style, and legal-reasoning frames are often linearly decodable from model activations.
- Many decoded directions do not survive causal generation tests.
- Prompt-only or text-only baselines can look deceptively strong, so promoted claims need strict controls.
- For SCOTUS/Qwen legal reasoning work, short generations are smoke tests only. Complete-answer evaluation should use at least 2048 generated tokens, preferably 3072-4096.

The repo intentionally includes negative results. A central finding is that "style is decodable" and "style is a reusable steering actuator" are different claims.

## Research Threads

### 1. Fictional Character Personality Baking

Original target: encode a Skippy-like fictional voice into Qwen-family models without relying on a system prompt at inference time.

Work explored:

- Contrastive activation directions and permanent weight ablation.
- LoRA SFT, DPO, GRPO, and self-distillation fine-tuning.
- Identity and assistant-mode neuron probes.
- Push/pull neuron regularization for persona behavior.
- AIME-style reasoning preservation checks.

Most detailed notes are in `RESEARCH_NOTES.md`, `reports/`, and older archived scripts under `archive/`.

### 2. SCOTUS Judicial Reasoning Steering

The public-domain follow-up uses Supreme Court opinions to test a cleaner question:

> Can a model's legal-reasoning trajectory be causally shifted between controlled jurisprudential frames without merely role-playing a named justice?

This branch includes:

- Court opinion data preparation and source-frame labels.
- Masked text baselines and case/source holdouts.
- Activation probes across layers and token regions.
- Controlled replay/minimal-pair banks.
- No-mask generation pokes with random/source/text controls.
- Visible-thinking and final-answer evaluation reports.

Start with:

- `SCOTUS.md`
- `SCOTUS_Phase4.md`
- `data/scotus/README.md`
- `scripts/experiments/scotus/README.md`

## Repository Map

| Path | Contents |
|---|---|
| `scripts/experiments/scotus/` | SCOTUS data prep, probing, patching, poking, review, and budget helpers |
| `scripts/experiments/personality/` | Personality, meta-cognition, and symphonic-voice experiment scripts |
| `scripts/eval/` | Evaluation harnesses and steering/eval utilities |
| `scripts/sae/` | SAE activation collection, training, and analysis scripts |
| `scripts/infra/` | Local orchestration, GPU monitoring, sweep tooling, and artifact inventory |
| `data/` | Compact prompt banks, manifests, labels, and review queues |
| `data/scotus/` | Trackable SCOTUS artifacts and compact direction files |
| `reports/` | Experiment reports, audits, adjudication notes, and decision logs |
| `ui/` | Static dashboards and visualization templates |
| `archive/` | Older Skippy pipeline scripts, phase reports, and legacy docs |
| `logs/`, `results/`, `sweep_v*/` | Local run outputs; many large artifacts are intentionally ignored |

## Setup

Use Python 3.11+ with CUDA. The local convention for this workspace is a virtual environment named `dev_genius`.

```bash
git clone https://github.com/Atlas3DSS/Character-Creation.git
cd "Character Creation"

python3 -m venv dev_genius
source dev_genius/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA builds of PyTorch, install the appropriate wheel for your machine before or during dependency setup. This workspace was developed on NVIDIA GPUs and assumes GPU access for serious activation capture, training, and vLLM generation.

## Model And Data Notes

Large model checkpoints, raw corpora, hidden-state matrices, and full sweep outputs are not guaranteed to be present in the public repo.

Before loading Hugging Face models, check the local cache and avoid silent downloads of large checkpoints. The project convention is:

```python
import os
from pathlib import Path

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    return (
        model_dir.exists()
        and (any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin")))
    )
```

Book-derived source material and large training outputs should remain local unless they are explicitly cleared for sharing. Compact public artifacts and manifests live under `data/` and `reports/`.

## Common Workflows

Run general eval battery:

```bash
source dev_genius/bin/activate
python scripts/eval/eval_runner.py --model Qwen/Qwen3-VL-8B-Thinking --n-per-category 50
```

Run SAE training:

```bash
source dev_genius/bin/activate
python scripts/sae/sae_train.py --config scripts/sae/sae_config.py
```

Run local overnight orchestration:

```bash
source dev_genius/bin/activate
bash scripts/infra/overnight_local.sh
```

Launch GPU monitor:

```bash
source dev_genius/bin/activate
python scripts/infra/gpu_monitor.py
```

Inspect compact SCOTUS run-constructor rules before creating new Qwen legal generations:

```bash
sed -n '1,220p' scripts/experiments/scotus/README.md
```

## SCOTUS/Qwen Evaluation Budget Rule

Qwen is verbose. A few hundred generated tokens is not enough for complete legal-holding evaluation.

Use `scripts/experiments/scotus/qwen_eval_budget.py` in new SCOTUS generation constructors. Any run below 2048 answer tokens must be labeled smoke/debug and must not be used for promotion, scorer calibration, or learned-result claims.

Reports and manifests should record:

- answer and thinking token budgets
- short-budget opt-in flags
- `budget_note`
- `promotion_eligible_budget`

## Serving And Steering Architecture

The project separates hook-heavy experimentation from fast serving:

| Phase | Engine | Reason |
|---|---|---|
| Extraction, activation capture, steering, and tuning | Hugging Face Transformers | Needs hooks, hidden states, and custom interventions |
| Review loops and post-ablation serving | vLLM | Fast inference when hooks are not needed |

vLLM is useful for serving ablated or merged models, but it is not the right tool for inference-time activation steering that depends on PyTorch hooks.

Example vLLM server:

```bash
source dev_genius/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model ./skippy_vectors/ablated_model \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

## Reading Order

For a quick orientation:

1. `SCOTUS.md` for the current judicial-reasoning research status.
2. `reports/scotus_phase5_decision_20260501.md` for a concise decision log on failed/promoted branches.
3. `data/scotus/README.md` for compact artifact provenance.
4. `RESEARCH_NOTES.md` for older personality-baking notes.
5. `archive/INDEX.md` for legacy phase material.

## Citation

```bibtex
@misc{charactercreation2026,
  title  = {Character Creation: Mechanistic Experiments in Personality and Reasoning-Style Steering},
  author = {Atlas3DSS},
  year   = {2026},
  url    = {https://github.com/Atlas3DSS/Character-Creation}
}
```
