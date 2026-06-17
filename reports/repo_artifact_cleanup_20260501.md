# Repo Artifact Cleanup Pass

## Goal

Keep enough provenance for null results and future review while making `git status` useful again.

## What Changed

- Added ignore rules for local environments, logs, raw sweep directories, generated tensor files, raw results, temp folders, and generated report HTML.
- Added explicit exceptions so small SCOTUS provenance files are visible to Git instead of being hidden by the old global `*.jsonl` rule.
- Kept large SCOTUS raw/intermediate corpora local: CourtListener source dumps, chunk inventories, matched pairs, section inventories, and excluded chunk inventories.
- Added `scripts/infra/repo_artifact_inventory.py` to reproduce the inventory without opening large experiment outputs.
- Added `scripts/infra/index_scotus_raw_runs.py` to regenerate the raw SCOTUS run archive index.
- Wrote `reports/repo_artifact_inventory_20260501.md` as the current shallow inventory.
- Wrote `reports/scotus_raw_run_archive_index_20260501.md` as a tracked index of ignored local SCOTUS raw run directories.
- Promoted the compact minimal-pair replay artifacts out of ignored `sweep_v4/` into `data/scotus/replay/`, `data/scotus/directions/`, and `data/scotus/sae/`.
- Added `data/scotus/README.md` and `data/scotus/artifact_manifest_20260501.json` so future cleanup passes can tell which small artifacts are intentionally trackable.

## Before And After

Before this pass, `git status --porcelain --untracked-files=all` surfaced `21,610` rows:

- `dev_genius/`: `19,460` rows from the local venv
- `sweep_v4/`: `1,743` rows from raw experiment outputs
- `logs/`: `124` rows

After the ignore-policy update, cleanup documentation, compact artifact promotion, and late-residual/prototype audit additions, the same status command surfaces `291` rows:

- `scripts`: `130`
- `reports`: `93`
- `data`: `57`
- a small set of docs, skills, UI templates, archive notes, and existing infra edits

## Tracking Policy

Track these:

- Experiment code: `scripts/experiments/**`, `scripts/infra/**`
- Living project docs: `SCOTUS.md`, `SCOTUS_Phase4.md`, `activation_probes.md`, `AGENTS.md`
- Human-readable reports and summary JSON: `reports/*.md`, selected `reports/*.json`
- Compact data needed to rerun/review decisions: prompt banks, frame labels, review queues, adjudication keys, manifest JSON
- Small promoted directions and replay/SAE summaries under `data/scotus/directions/`, `data/scotus/replay/`, and `data/scotus/sae/`
- Project skills/templates that affect future agent behavior

Do not normally track these:

- Local venvs: `dev_genius/`, `.venv/`, `venv/`
- Raw run outputs: `sweep_v2/`, `sweep_v3/`, `sweep_v4/`
- Logs: `logs/`
- Raw corpora and bulky intermediates: `data/scotus/raw/`, `data/scotus/processed/`, chunk inventories, matched pairs, section inventories
- Binary activations/directions/features: `*.pt`, `*.npz`, `*.npy`, except explicit compact promotions under `data/scotus/directions/`
- Generated dashboard/report HTML

## Archive Rule

The raw outputs are being preserved locally but removed from Git's active surface. For provenance, every run that matters should have a small tracked report or manifest that points to its raw run directory. If disk cleanup becomes necessary, move old raw run directories under `archive/eval_results/` or an external backup only after confirming no active process is writing there.

Tracked raw-run index:

- `reports/scotus_raw_run_archive_index_20260501.md`

Latest run indexed during this pass:

- `sweep_v4/scotus_prototype_patch_20260501_123725`

Compact artifacts promoted during this pass:

- `data/scotus/replay/scotus_minpair_replay_examples_20260501.jsonl`
- `data/scotus/replay/scotus_minpair_replay_feature_meta_20260501.jsonl`
- `data/scotus/replay/scotus_minpair_replay_layer_region_search_20260501.jsonl`
- `data/scotus/replay/scotus_minpair_replay_manifest_20260501.json`
- `data/scotus/directions/scotus_minpair_replay_assistant_all_L4_direction_20260501.npz`
- `data/scotus/sae/scotus_minpair_l0_100_top_features_20260501.jsonl`
- `data/scotus/sae/scotus_minpair_l0_100_feature_examples_20260501.jsonl`
- `data/scotus/sae/scotus_minpair_l0_100_layer_region_search_20260501.jsonl`
- `data/scotus/sae/scotus_minpair_l0_100_summary_20260501.json`
- `data/scotus/sae/scotus_minpair_l0_100_manifest_20260501.json`
- `data/scotus/directions/probe_direction_assistant_all_L16_C0p001.npz`
- `data/scotus/directions/probe_direction_assistant_all_L20_C0p001.npz`

## Staged Add Set

I staged the trackable surface with explicit path-scoped adds, not `git add .`, because this repo intentionally mixes source files with large local artifacts:

```bash
git add .gitignore AGENTS.md SCOTUS.md SCOTUS_Phase4.md activation_probes.md
git add .agents/skills/codex-review/SKILL.md .agents/skills/gemini-research/SKILL.md
git add scripts/experiments
git add data/scotus data/symphonic_voice_anchor_manifest*.json
git add reports
git add ui/personality_phase_visualizer_template.html ui/personality_synthesis_visualizer_template.html
git add restart_2026_03/cleanroom_behavioral_subspace_experiment.md
git ls-files --others --exclude-standard -z scripts/infra | xargs -0 --no-run-if-empty git add --
```

I intentionally did not stage the preexisting modified files:

- `scripts/infra/orchestrate_overnight.py`
- `scripts/infra/queue_abliterated_connectome.sh`

Review before commit:

```bash
python scripts/infra/repo_artifact_inventory.py --top 40
python scripts/infra/index_scotus_raw_runs.py --output reports/scotus_raw_run_archive_index_20260501.md
git status --short --untracked-files=all
```
