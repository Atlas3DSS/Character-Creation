# SCOTUS Data Provenance

This directory keeps compact, trackable artifacts for the SCOTUS/Qwen judicial reasoning project.

## Track Here

- Prompt banks and replay examples needed to reproduce probes.
- Human-review queues, adjudication keys, and compact frame labels.
- Probe manifests and small direction artifacts that future runs need directly.
- SAE top-feature summaries and search tables that identify candidate features.

## Qwen Evaluation Budget

Any prompt bank or review queue intended for Qwen legal holding evaluation should be paired with complete-answer generations: at least `2048` generated answer tokens, preferably `3072-4096`. Shorter generations belong only in smoke/debug artifacts and should be labeled that way in manifests and reports.

Script-level constructors should use `scripts/experiments/scotus/qwen_eval_budget.py`; see `scripts/experiments/scotus/README.md` before adding new generated-output artifacts.

## Keep Local/Ignored

- Raw opinion corpora in `raw/`.
- Processed chunk inventories, matched-pair inventories, and section inventories.
- Full hidden-state feature matrices and generated sweep output under `sweep_v*`.
- Large tensor, activation, or checkpoint files.

## Promoted 2026-05-01 Artifacts

The minimal-pair Commerce replay branch produced one compact direction and one SAE localization result that should survive raw-run cleanup:

- `replay/scotus_minpair_replay_examples_20260501.jsonl`
- `replay/scotus_minpair_replay_feature_meta_20260501.jsonl`
- `replay/scotus_minpair_replay_layer_region_search_20260501.jsonl`
- `replay/scotus_minpair_replay_manifest_20260501.json`
- `directions/scotus_minpair_replay_assistant_all_L4_direction_20260501.npz`
- `sae/scotus_minpair_l0_100_top_features_20260501.jsonl`
- `sae/scotus_minpair_l0_100_layer_region_search_20260501.jsonl`
- `sae/scotus_minpair_l0_100_summary_20260501.json`
- `sae/scotus_minpair_l0_100_manifest_20260501.json`
- `sae/scotus_minpair_l0_100_feature_examples_20260501.jsonl`
- `directions/probe_direction_assistant_all_L16_C0p001.npz`
- `directions/probe_direction_assistant_all_L16_C0p001.json`
- `directions/probe_direction_assistant_all_L20_C0p001.npz`
- `directions/probe_direction_assistant_all_L20_C0p001.json`

The raw source run directories remain local and ignored:

- `sweep_v4/scotus_minpair_replay_20260501_100514`
- `sweep_v4/scotus_sae_probe_20260501_112601`
