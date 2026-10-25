# Character Creation

Research code and artifacts for studying whether an open-weight language model can be moved from prompt-level persona imitation toward durable changes in generated reasoning and voice.

The new [Persona Explorer](PERSONA_EXPLORER.md) combines **[Lyra’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)** methods with this project’s broader Big Five and layer experiments, targeting **Qwen3.8-27B** through native Flare steering. Lyra’s signed neuron sweeps, relative-dose normalization, read/write distinction, and comparison workflow are the primary methodological inspiration. The implementation here is independent; Lyra’s code is not vendored.

**The reusable V100 runtime lives in a separate repository: [Atlas3DSS/flare-local-inference](https://github.com/Atlas3DSS/flare-local-inference).** Its [steering guide](https://github.com/Atlas3DSS/flare-local-inference/blob/atlas/sm70-public/STEERING.md) covers build/preparation, the native API and offline activation capture. Character Creation holds the persona interface, datasets, experiment manager and analysis. V100 owners can share and use the runtime independently.

The first six-V100 reference run completed **2,046/2,046** unsteered and prompt-conditioned responses, including 33 complete IPIP-50 inventories and 132 simple capability answers. These are baseline measurements, not evidence that activation steering works. Fresh checkpoint calibration and causal neuron validation are tracked separately. See [the audited baseline](reports/persona_baseline.json) and [the explorer guide](PERSONA_EXPLORER.md).

The new overnight workflow couples six resident Flare diagnostic workers with a local 27B planner and a persistent SQLite experiment manager. It records **actual continuous SwiGLU activations at the final prompt token**, matched prompt mutations and complete rollouts, using baseline, explicit-zero, random, positive and negative arms. Selecting a source group includes every registered mutation/profile with matched seeds. The planner returns validated experiment JSON and has no shell access. Discovery and heldout data stay separate; book sources and extracts remain private. See [setup and evidence limits](PERSONA_EXPLORER.md#overnight-discovery-with-measured-activations).

**Whole-model qualification remains pending.** Steering component checks passed on all six V100s, and the separate planner has verified 128k-context readiness. Calibration compilation and planner startup do not establish causal effects, native cache correctness or MTP qualification.

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
| `persona_manager.py`, `persona_planner.py` | Durable proposal/attempt queue and the Atlas-specific local 27B planner supervisor |
| `persona_probe.py`, `persona_capture.py`, `persona_map.py` | Retained Flare graph observations, calibration and provisional neuron maps |
| `persona_corpus.py` | Split-preserving SCOTUS/Big Five prompts and explicit private-book ingestion |
| `persona_coverage.py` | Operator-only queueing and status for one unsteered prefill observation per registered row, preserving heldout splits |
| `persona_characters.py` | Private original-source character/speaker candidates with exact evidence spans, uncertainty and resumable coverage |
| `PERSONA_EXPLORER.md` | Native intervention semantics, overnight setup, evidence limits and attribution |
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

For the new six-V100 persona workflow, use the [Persona Explorer setup](PERSONA_EXPLORER.md#start-the-workers-and-manager) instead of the legacy orchestration script. After calibrated probe workers and the local planner are leased and ready:

```bash
python persona_manager.py configure --fleet persona_fleet.local.json \
  --config results/persona_manager_config.json \
  --planner-lease GPU_WORKBOARD_PLANNER_JOB_ID --base-port 8930
python persona_manager.py run --config results/persona_manager_config.json
python persona_manager.py status
```

The manager resumes the same SQLite queue after interruption and defaults to a bounded run until the next morning. It verifies measured activation artifacts and preserves incomplete/error attempts. It does not launch GPU services or replace their leases.

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

The current 27B workflow uses native Flare graph inputs for steering and fixed observation buffers for diagnostics. Earlier Transformers/vLLM experiments remain available for comparison:

| Phase | Engine | Reason |
|---|---|---|
| Current 27B calibration and measured neuron sweeps | Retained Flare diagnostic graphs | Actual rounded SwiGLU/residual observations and ordinary sampled decode, one active request per V100 |
| Current 27B production steering | Separately prepared native Flare server | Persistent request-local inputs and cache conditioning; independent production/MTP qualification required |
| Earlier extraction, steering and tuning experiments | Hugging Face Transformers | Python hooks, hidden states and custom interventions |
| Earlier review loops and post-ablation serving | vLLM | Serving ablated or merged models |

The diagnostic worker and production server are distinct evidence paths. A diagnostic success does not qualify native cache reuse or speculative decoding. The new Flare path does not depend on Python forward hooks; its reusable implementation is maintained in [Atlas3DSS/flare-local-inference](https://github.com/Atlas3DSS/flare-local-inference), with prominent attribution to [Lyra's neuron-explorer methodology](https://github.com/lyramakesmusic/neuron-explorer).

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
