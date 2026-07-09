# Character Creation

Mechanistic experiments in personality, style, reasoning-frame, and output-disposition steering for open-weight language models.

This repository is an active research workspace. It is not a packaged library and it intentionally contains failed branches, negative results, control runs, and work-in-progress experiment constructors. The main standard is evidential: a result should be promoted only when the run has enough controls, token budget, provenance, and artifact logging to survive later paper-grade review.

## Current Research Position

The project started as an attempt to bake a fictional character voice into Qwen-family models without relying on a system prompt at inference time. It then expanded into public-domain legal-reasoning and style-frame experiments using Supreme Court materials. As of July 2026, the active work is centered on Jacobian-lens / J-space measurements:

- Persona, source style, and legal-reasoning frames are often linearly decodable from hidden states.
- Decodable directions often fail causal steering or generation tests.
- Prompt-only and text-only baselines can look deceptively strong, so activation claims need text baselines and random/null controls.
- Qwen thinking models are verbose; short generations are smoke tests, not full reasoning evaluations.
- Current J-lens work asks whether persona/style signatures concentrate in top-k output-facing transport subspaces, whether constrained J-ReFT can preserve capability better than unconstrained adapters, and whether fine-tuning changes the transport map itself.

The central distinction is:

> "A style is decodable" is not the same claim as "a style is a reusable steering actuator."

## Active July 2026 J-Lens Program

The current coordinated objective is documented in:

- `reports/jlens_three_brief_goal_statement_20260709.md`
- `reports/jlens_three_brief_execution_log_20260709.md`

It implements three July 8 briefs in dependency order:

1. **J-space persona fingerprinting**
   - Generate long-form responses from an unmodified Qwen model under system-prompt personas.
   - Capture generated-token residual activations only.
   - Compare raw activations, `P_J h`, `(I - P_J) h`, random same-dimensional subspaces, final-layer controls, logit controls, label-shuffle nulls, TF-IDF text baselines, fingerprint stability, and token readouts.

2. **J-ReFT / J-LoRA pilot**
   - Fit or verify a local lens for the exact unmodified `Qwen/Qwen3.5-9B` instruct checkpoint.
   - Train/evaluate arms A-F: J-space, random subspace seeds, unconstrained, complement, V4/prompt baseline, and raw baseline.
   - Evaluate held-out no-system generations for persona/style movement, answer-checkable capability retention, coherence, and repetition risk.

3. **Delta-J transport-map comparison**
   - Establish a mandatory refit-noise floor by fitting the same model twice on disjoint prompt slices.
   - Compare model pairs only as multiples of that noise floor.
   - Report subspace geometry, map similarity, vocab-resolved transported-vector drift, and layer profiles.

No 27B main adaptation run should be launched until the 9B pilot gate is explicitly met and logged.

## Repository Map

| Path | Contents |
|---|---|
| `scripts/experiments/jlens_common.py` | Shared J-lens cache checks, projection math, probes, reporting helpers |
| `scripts/experiments/personality/jlens_persona_fingerprint.py` | J-space persona fingerprint capture and analysis |
| `scripts/experiments/personality/jlora_pilot.py` | J-ReFT/J-LoRA constrained adaptation pilot |
| `scripts/experiments/connectome/fit_local_jlens.py` | Local Jacobian-lens fitting wrapper with manifests and resume checkpoints |
| `scripts/experiments/connectome/jlens_delta_comparison.py` | Delta-J transport-map comparison against refit-noise floor |
| `scripts/experiments/connectome/` | Qwen connectome, spectral, steering, and J-lens analysis scripts |
| `scripts/experiments/personality/` | Personality, self-distillation, meta-cognition, and no-mask evaluation scripts |
| `scripts/experiments/scotus/` | SCOTUS data prep, probing, patching, generation, review, and budget helpers |
| `scripts/infra/` | Orchestration, GPU monitoring, preflights, watchers, and remote handoff scripts |
| `scripts/eval/` | General evaluation harnesses |
| `scripts/sae/` | SAE activation collection, training, and analysis |
| `data/personas/` | Compact persona prompt banks for fingerprinting |
| `data/scotus/` | Trackable SCOTUS labels, queues, manifests, compact directions, and reviewed artifacts |
| `reports/` | Briefs, execution logs, decision logs, diagnostics, and paper-facing summaries |
| `sweep_v4/`, `logs/`, `results/` | Local generated outputs; intentionally ignored by git |
| `external/` | Local dependency checkouts; intentionally ignored by git |

## Environment

Use Python 3.11+ with CUDA-capable PyTorch. The project convention is a virtual environment named `dev_genius`.

Generic setup:

```bash
git clone https://github.com/Atlas3DSS/Character-Creation.git
cd "Character Creation"

python3 -m venv dev_genius
source dev_genius/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Workstation-specific note: some long J-lens runs use the provisioned CUDA 13 environment at `/home/orwel/dev_genius/venv` rather than the project-local `./dev_genius` venv. Use the environment that actually has a working CUDA build for the target GPU.

Verify the active environment before expensive runs:

```bash
source /home/orwel/dev_genius/venv/bin/activate
python scripts/infra/jlens_three_brief_preflight.py
```

Expected J-lens dependencies include:

- `torch`, `torchvision`
- `transformers`, `accelerate`
- `peft`, `datasets`
- `jlens`
- `numpy`, `scikit-learn`
- `tqdm`

## Model And Cache Rules

Large checkpoints and fitted lenses are not committed to git. Always check local cache state before loading or downloading models.

Project policy:

- Check Hugging Face cache before every heavyweight model load.
- Print cache status in scripts.
- Do not silently download 16 GB+ models.
- Use BF16/full precision for current J-lens tests unless explicitly instructed otherwise.
- Do not quantize real J-lens/J-ReFT experiments unless a run is explicitly marked quantized.
- Use unmodified Qwen checkpoints for the current Delta-J base/instruct test; do not substitute abliterated models unless that pair is the explicit experiment.

Reusable cache-check pattern:

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

## Artifact And Reporting Standards

Every nontrivial run should write enough evidence that a later report can be reconstructed without chat history:

- `manifest.json`: model, lens, cache paths, scripts, seeds, token budgets, dtype, device, prompt counts, layers, ranks, claim eligibility, git status.
- `events.jsonl`: append-only progress and lifecycle events.
- `records.jsonl`: per-example or per-layer records used for analysis.
- `report.md`: human-readable summary with methods, controls, results, claim status, and known caveats.
- `prompt_audit.json` or equivalent prompt inventory for generated/fitted runs.
- Log files under `logs/` or the run directory.

Reports must clearly distinguish:

- real pilot findings
- smoke/debug outputs
- failed gates
- incomplete runs
- unresolved hypotheses
- known confounds

For arXiv-facing summaries, include null controls, random controls, text baselines, token budgets, exact model/lens identifiers, and whether generated outputs were promotion eligible.

## J-Lens Workflows

### Phase 0 Preflight

```bash
source /home/orwel/dev_genius/venv/bin/activate
python scripts/infra/jlens_three_brief_preflight.py
```

### Persona Fingerprinting Pilot

```bash
source /home/orwel/dev_genius/venv/bin/activate
python scripts/experiments/personality/jlens_persona_fingerprint.py \
  --allow-real-model-run \
  --output-dir sweep_v4/jlens_persona_fingerprint_real_$(date +%Y%m%d_%H%M%S) \
  --pilot-personas 3 \
  --pilot-prompts 10 \
  --pilot-layers 2 \
  --k-values 8,32,128,512 \
  --max-new-tokens 3072 \
  --device cuda
```

Key controls:

- generated-token activations only
- raw `h`
- `P_J h`
- `(I - P_J) h`
- random same-dimensional subspaces
- label-shuffle nulls
- TF-IDF text baseline
- final-layer and output-logit controls
- split-half signature stability

### Fit Local 9B Lenses

Same-model refit-noise fits use disjoint prompt slices:

```bash
python scripts/experiments/connectome/fit_local_jlens.py \
  --model Qwen/Qwen3.5-9B \
  --model-class causal-lm \
  --output-dir sweep_v4/lens_qwen35_9b_instruct_a \
  --source-layers 8,16,24 \
  --n-prompts 32 \
  --skip-prompts 0 \
  --max-seq-len 128 \
  --dim-batch 1 \
  --checkpoint-every 1 \
  --resume \
  --device cuda
```

Use a second run with `--skip-prompts 32` for the refit-noise floor, then fit the paired checkpoint, for example `Qwen/Qwen3.5-9B-Base`, with the same fit config and a disjoint prompt slice.

### J-ReFT / J-LoRA Pilot

```bash
python scripts/experiments/personality/jlora_pilot.py \
  --allow-real-model-run \
  --output-dir sweep_v4/jlora_pilot_$(date +%Y%m%d_%H%M%S) \
  --model-name Qwen/Qwen3.5-9B \
  --local-instruct-lens sweep_v4/lens_qwen35_9b_instruct_a/jacobian_lens.pt \
  --train-file sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429/pairs.jsonl \
  --layers 8,16,24 \
  --j-rank 128 \
  --reft-rank 16 \
  --max-train-steps 120 \
  --batch-size 1 \
  --grad-accum 8 \
  --eval-limit 12 \
  --capability-limit 16 \
  --max-new-tokens 3072 \
  --device cuda
```

Trained arms must be evaluated without a system prompt. Prompt baselines may use the prompt by definition, but must be labeled as prompt baselines rather than weight-baked persona success.

### Delta-J Comparison

```bash
python scripts/experiments/connectome/jlens_delta_comparison.py \
  --allow-real-comparison \
  --output-dir sweep_v4/delta_qwen35_9b_instruct_vs_base \
  --pair-label qwen35_9b_instruct_vs_base_unmodified \
  --noise-floor-a sweep_v4/lens_qwen35_9b_instruct_a/jacobian_lens.pt \
  --noise-floor-b sweep_v4/lens_qwen35_9b_instruct_b/jacobian_lens.pt \
  --model-a-lens sweep_v4/lens_qwen35_9b_instruct_a/jacobian_lens.pt \
  --model-b-lens sweep_v4/lens_qwen35_9b_base/jacobian_lens.pt \
  --model-a Qwen/Qwen3.5-9B \
  --model-b Qwen/Qwen3.5-9B-Base \
  --tokenizer-model Qwen/Qwen3.5-9B \
  --k-values 8,32,128,512
```

No Delta-J finding is valid without a same-model refit-noise floor.

### Overnight Orchestration

```bash
source /home/orwel/dev_genius/venv/bin/activate
bash scripts/infra/run_jlens_three_briefs_overnight.sh
```

Remote watcher scripts:

- `scripts/infra/watch_persona_fingerprint_reanalysis.sh`
- `scripts/infra/run_remote_9b_followups.sh`

These are designed to avoid a simple overnight blocker: they wait for prerequisite artifacts, refuse partial reanalysis, and log follow-up handoffs.

## SCOTUS/Qwen Evaluation Budget Rule

For legal reasoning and complete-answer Qwen evaluations:

- Minimum answer/generation budget: `2048` tokens.
- Preferred final reasoning/evaluation budget: `3072-4096` tokens.
- Runs below `2048` answer tokens are smoke/debug only.
- Smoke/debug runs must not be used for promotion, scorer calibration, or learned-result claims.

New SCOTUS generation/evaluation constructors should import and use:

```python
from scripts.experiments.scotus.qwen_eval_budget import qwen_budget_metadata
```

Before creating new SCOTUS/Qwen generation runs, read:

```bash
sed -n '1,220p' scripts/experiments/scotus/README.md
```

## Serving And Steering Architecture

The project uses two inference modes:

| Phase | Engine | Reason |
|---|---|---|
| Extraction, activation capture, steering, J-ReFT, and hidden-state analysis | Hugging Face Transformers | Requires hooks, hidden states, and custom interventions |
| Fast serving or review loops for merged/ablated models | vLLM or llama.cpp | Fast generation when hooks are not needed |

vLLM is not the right tool for inference-time activation steering that depends on PyTorch hooks.

## Research Threads

### Personality And Character Voice

Original question: can a model internalize a persona or voice without a system prompt?

Representative work:

- contrastive activation directions
- field steering
- no-mask evaluation
- LoRA/SFT and self-distillation
- persona-vs-capability preservation
- J-space-constrained adaptation

### SCOTUS Legal Reasoning

Public-domain follow-up question:

> Can a model's legal-reasoning trajectory be causally shifted between controlled jurisprudential frames without merely role-playing a named justice?

Start with:

- `SCOTUS.md`
- `SCOTUS_Phase4.md`
- `data/scotus/README.md`
- `scripts/experiments/scotus/README.md`

### J-Lens / Connectome

Current mechanistic question:

> Does output-facing transport geometry explain where persona/style/reasoning information is represented, whether constrained updates preserve capability, and whether training changes the transport map?

Start with:

- `reports/jlens_scotus_diagnostic_brief_20260706.md`
- `reports/jlens_persona_fingerprint_brief_20260708.md`
- `reports/jlens_jlora_brief_20260708.md`
- `reports/jlens_lens_comparison_brief_20260708.md`
- `reports/jlens_three_brief_goal_statement_20260709.md`
- `reports/jlens_three_brief_execution_log_20260709.md`

## What Not To Commit

Do not commit:

- `.env` or API keys
- local model checkpoints
- raw copyrighted book text
- large activations or tensor dumps
- `sweep_v4/`, `logs/`, `results/`
- local dependency checkouts under `external/`
- generated response dumps unless deliberately compact and cleared for sharing

The repository should contain scripts, compact prompt banks, manifests, labels, reports, and reproducibility metadata. Large outputs stay local unless explicitly promoted.

## Reading Order

For a quick orientation:

1. `README.md` for repository structure and current run standards.
2. `reports/jlens_three_brief_goal_statement_20260709.md` for the active J-lens objective.
3. `reports/jlens_three_brief_execution_log_20260709.md` for current run provenance.
4. `reports/jlens_scotus_diagnostic_brief_20260706.md` for the J-lens setup and layer-convention context.
5. `reports/scotus_phase5_decision_20260501.md` for an example decision log separating negative and promoted branches.
6. `scripts/experiments/scotus/README.md` before constructing new SCOTUS/Qwen generation runs.

## Citation

```bibtex
@misc{charactercreation2026,
  title  = {Character Creation: Mechanistic Experiments in Personality and Reasoning-Style Steering},
  author = {Atlas3DSS},
  year   = {2026},
  url    = {https://github.com/Atlas3DSS/Character-Creation}
}
```
