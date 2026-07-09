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

- Persona fingerprinting: completed real 27B pilot after reset recovery; decodability is present, but J-specific concentration is not supported because text, raw, complement, final-layer, logit, and random controls are also strong.
- J-ReFT/J-LoRA pilot: completed real 9B pilot; gate not passed because J-space did not beat the prompt baseline and every arm carried a doom-loop flag.
- Delta-J comparison: completed real 9B instruct-vs-base comparison; observed deltas are indistinguishable or small relative to same-model refit noise.

No promoted causal-steering claim should be made from this log alone.

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

### 2026-07-09 01:39 PDT

- Remote instruct lenses A and B completed and wrote `jacobian_lens.pt`; they now provide the same-model refit-noise floor for Delta-J.
- Remote watcher launched base-lens fitting and J-ReFT at `2026-07-09T08:30:55+00:00`.
- The first J-ReFT launch failed immediately because the dev-box run tree was missing `data/sarcasm_markers.json`; no model training or evaluation records were produced in that failed attempt.
- Copied the tracked local `data/sarcasm_markers.json` to the dev box and archived the failed `jlora_pilot` directory.
- Relaunched J-ReFT manually in tmux session `jlora_retry_20260709_0835`, pinned by UUID to the physical RTX 3090 to avoid colliding with base fitting on the RTX 4090.
- Retry log: `sweep_v4/jlens_remote_9b_real_20260709_070746/logs/jlora_pilot_retry_gpu3090.log`.
- Retry dataset manifest: `102` train rows, `12` eval rows, `8` capability rows, system prompts stripped.
- Base lens status at this update: `3/32` prompts fitted.
- No findings yet; all hypothesis statuses remain unresolved.

### 2026-07-09 02:06 PDT

- J-ReFT retry completed arm `A` (`j_space`) training and long-budget evaluation.
- Arm `A` artifact counts: `18` generation rows, `18` eval rows, `1` arm summary row.
- Arm `A` automatic proxy metrics: persona fidelity `3.4618`, capability retention `0.625` over `8` capability rows, coherence `0.5588`, `doom_loop_flag=true`.
- Arm `A` generation lengths by whitespace tokens: min `809`, mean `1521.1`, max `1987`; the run used the configured `3072` max-new-token budget.
- Claim status remains unpromoted: these are automatic proxy metrics, not an external critic score, and the doom-loop flag is a negative gate.
- J-ReFT proceeded to random-subspace control arm `B1` evaluation.
- Base lens status at this update: `14/32` prompts fitted.
- 27B persona fingerprinting status at this update: `31/40` records captured.

### 2026-07-09 07:10 PDT

- Workstation reset recovery audit:
  - Local RTX Pro 6000 run state: no surviving 27B capture process; `sweep_v4/jlens_persona_fingerprint_real_20260709_001511/records.jsonl` survived with `38/40` records.
  - Missing persona-fingerprint cells: `formal_business_register/prompt_009` and `formal_business_register/prompt_010`.
  - Added resumable real-capture support to `scripts/experiments/personality/jlens_persona_fingerprint.py` via `--resume-capture-dir`; it skips records only when the record and activation file both exist.
  - Relaunched local 27B resume in tmux session `jlens_27b_resume_20260709_0707` with `--k-values 8,32,128,512` and `--max-new-tokens 3072`.
  - Resume event log confirmed `existing_records=38` and `invalid_existing=0` before generating the two missing cells.
- Dev-box recovery audit:
  - Instruct lens A, instruct lens B, and base lens all completed and wrote `jacobian_lens.pt`.
  - J-ReFT retry completed all arms A-F with `144` generated/eval rows, `8` arm summary rows, adapters, manifest, and report.
  - Delta-J completed with `3` layer records, manifest, vocab summaries, and report.
- J-ReFT gate status: not passed. Arm `A` (`j_space`) did not beat the prompt baseline and every arm, including baselines, had `doom_loop_flag=true`; no 27B main adaptation run should be launched from this pilot.
- Delta-J status: unmodified `Qwen/Qwen3.5-9B` instruct vs unmodified `Qwen/Qwen3.5-9B-Base` was `indistinguishable_or_small_vs_noise` at layers `8`, `16`, and `24` relative to the same-model refit-noise floor.

### 2026-07-09 07:32 PDT

- 27B persona fingerprinting completed after resumable recovery.
- Final artifacts in `sweep_v4/jlens_persona_fingerprint_real_20260709_001511`:
  - `records.jsonl`: `40` generation/capture records.
  - `probe_records.jsonl`: `34` probe/control rows.
  - `manifest.json`, `report.md`, `text_baseline.json`, `stability.json`, `readouts.json`, `logit_control_tokens.json`.
- Final manifest mode: `real_model_reanalysis`; generation budget `3072`; layers `[0, 62]`; k values `[8, 32, 128, 512]`.
- Resume provenance: `captured_new=2`, `skipped_existing=38`, `invalid_existing=0`.
- Probe implementation note: switched the shared linear probe helper to `StandardScaler + RidgeClassifier(alpha=1.0, class_weight='balanced')` after the original logistic-regression path proved too slow for the null-control battery on high-dimensional 27B activations. The control set and fold structure were preserved.
- Persona report headline:
  - TF-IDF text baseline balanced accuracy: `1.000`.
  - Fingerprint nearest-signature stability: `1.000` over `20` held-out responses.
  - Raw, complement, final-layer, output-logit, and several random same-dimension controls also reached `1.000` balanced accuracy.
  - Readout top tokens were semantically aligned with persona labels, e.g. acerbic terms for `acerbic_polemicist`, folksy terms for `folksy_grandparent`, and contract/business terms for `formal_business_register`.
- Claim status: real decodability pilot completed, but J-space-specific concentration is not supported by this run because non-J and text controls are equally strong.
