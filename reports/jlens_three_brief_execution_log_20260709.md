# J-Lens Three-Brief Execution Log

Date: 2026-07-09

## Objective

Run the three July 8 J-lens briefs end to end with reusable scripts, manifests, records, and reports suitable for later research writeup:

1. J-space persona fingerprinting.
2. J-ReFT/J-LoRA pilot.
3. Delta-J transport-map comparison with mandatory same-model refit-noise floor.

This log records execution/provenance only. It is not a findings report.

## Environment

- Project root: `/home/orwel/dev_genius/experiments/Character Creation`
- Workstation venv: `/home/orwel/dev_genius/venv`
- Dev-box venv: `/home/orwel/dev_genius/venv`
- Workstation GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB class.
- Dev-box GPUs: NVIDIA RTX 3090 and NVIDIA RTX 4090.
- Precision policy: BF16/full precision; no quantization for the real J-lens/J-ReFT tests.
- Model policy: unmodified Qwen checkpoints for this test; no abliterated model is used for the active Delta-J pair.

## Phase 0 Preflight

Workstation preflight command:

```bash
source /home/orwel/dev_genius/venv/bin/activate
python scripts/infra/jlens_three_brief_preflight.py
```

Observed:

- `torch`, `transformers`, `accelerate`, `peft`, `datasets`, and `jlens` import.
- `/home/orwel/dev_genius/models/Qwen3.5-27B` exists with safetensor shards.
- `Qwen/Qwen3.5-9B` is present in the Hugging Face cache.
- `Qwen/Qwen3.5-9B-Base` is present in the Hugging Face cache.

The project-local `./dev_genius` venv is not used for these runs because its Torch install was interrupted earlier. CUDA 13 work uses `/home/orwel/dev_genius/venv`.

## Active Runs

### 27B Persona Fingerprinting

- tmux: `jlens_27b_fingerprint_20260709_001511`
- output dir: `sweep_v4/jlens_persona_fingerprint_real_20260709_001511`
- log: `logs/jlens_27b_fingerprint_20260709_001511.log`
- model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- lens: cached Neuronpedia `qwen3.5-27b` lens.
- generation cap: `3072` new tokens.
- current status at log creation: `6/40` generation records captured.
- generated-token residual activations are being written under `activations/`.

Follow-up watcher:

- tmux: `jlens_27b_reanalysis_watch_20260709`
- watcher: `scripts/infra/watch_persona_fingerprint_reanalysis.sh`
- watcher log: `logs/jlens_27b_fingerprint_reanalysis_watch.log`
- gate: refuses patched reanalysis unless at least `40` records exist.
- patched reanalysis adds `k = 8,32,128,512` and final/logit controls.

### 9B Instruct Local Lens Refit Noise Floor

- dev-box tmux: `jlens_remote_9b_20260709_070746`
- output dir: `sweep_v4/jlens_remote_9b_real_20260709_070746`
- model: `Qwen/Qwen3.5-9B`, unmodified instruct checkpoint.
- source layers: `8,16,24`
- prompts per fit: `32`
- sequence length: `128`
- dim batch: `1`

Fit A:

- GPU: RTX 3090
- output dir: `lens_qwen35_9b_instruct_a`
- log: `logs/lens_instruct_a_gpu0.log`
- current status at log creation: `13/32` prompts.

Fit B:

- GPU: RTX 4090
- output dir: `lens_qwen35_9b_instruct_b`
- log: `logs/lens_instruct_b_gpu1.log`
- current status at log creation: `12/32` prompts.

These two same-model disjoint-prompt fits are the mandatory Delta-J refit-noise floor.

### Remote Follow-Up Watcher

- dev-box tmux: `jlens_remote_9b_followups_20260709`
- watcher: `scripts/infra/run_remote_9b_followups.sh`
- watcher log: `sweep_v4/jlens_remote_9b_real_20260709_070746/logs/followups_watch.log`

Planned automatic follow-ups after both instruct lenses exist:

1. Fit `Qwen/Qwen3.5-9B-Base` lens on the dev-box GPU 0.
2. Train/evaluate the J-ReFT pilot on dev-box GPU 1 using `sweep_v4/qwen36_devbox_balanced_pairs_20260706_233429/pairs.jsonl`.
3. Run Delta-J comparing unmodified instruct vs unmodified base relative to the instruct refit-noise floor.

The watcher aborts rather than hanging if the instruct lens processes exit before both `jacobian_lens.pt` artifacts exist.

## Scripts Added Or Updated

- `scripts/experiments/jlens_common.py`
- `scripts/experiments/personality/jlens_persona_fingerprint.py`
- `scripts/experiments/personality/jlora_pilot.py`
- `scripts/experiments/connectome/fit_local_jlens.py`
- `scripts/experiments/connectome/jlens_delta_comparison.py`
- `scripts/infra/jlens_three_brief_preflight.py`
- `scripts/infra/run_jlens_three_briefs_overnight.sh`
- `scripts/infra/watch_persona_fingerprint_reanalysis.sh`
- `scripts/infra/run_remote_9b_followups.sh`
- `data/personas/fingerprint_v1.json`

## Claim Status

- Persona fingerprinting: running; no findings yet.
- J-ReFT/J-LoRA pilot: waiting on exact-checkpoint 9B instruct lens; no findings yet.
- Delta-J comparison: waiting on two instruct noise-floor lenses plus base lens; no findings yet.

No substantive claims should be made from this log alone.

## Status Updates

### 2026-07-09 00:54 PDT

- 27B persona fingerprinting: `9/40` records captured; workstation GPU remains allocated to the run.
- Remote instruct lens A: `18/32` prompts fitted on RTX 3090.
- Remote instruct lens B: `18/32` prompts fitted on RTX 4090.
- Remote follow-up watcher is still waiting for both instruct `jacobian_lens.pt` artifacts before launching base lens, J-ReFT, and Delta-J.
- No findings yet; all hypothesis statuses remain unresolved.

### 2026-07-09 01:02 PDT

- Git publish branch: `agent/jlens-three-briefs`.
- Commit: `1085285` (`Add J-lens three-brief experiment pipeline`).
- Draft PR: `https://github.com/Atlas3DSS/Character-Creation/pull/1`.
- 27B persona fingerprinting: `11/40` records captured.
- Remote instruct lens A: `22/32` prompts fitted on RTX 3090.
- Remote instruct lens B: `21/32` prompts fitted on RTX 4090.
- No findings yet; all hypothesis statuses remain unresolved.

### 2026-07-09 01:09 PDT

- README rewrite pushed to the same draft PR in commit `d241181` (`Update project README for J-lens workflow`).
- PR now contains the pipeline commit, execution-log update, and README rewrite.
- 27B persona fingerprinting: `13/40` records captured on the RTX Pro 6000 workstation run.
- Remote instruct lens A: `25/32` prompts fitted on the RTX 3090.
- Remote instruct lens B: `24/32` prompts fitted on the RTX 4090.
- Remote follow-up watcher has not yet launched base-lens fitting, J-ReFT, or Delta-J because both instruct `jacobian_lens.pt` artifacts are still pending.
- No findings yet; all hypothesis statuses remain unresolved.
