# Character Creation: agent interface and evidence contract

This document is the machine-oriented entry point. [README_human.md](README_human.md)
explains the motivation and workflow. Source: [GitHub](https://github.com/Atlas3DSS/Character-Creation).
Public data: `Atlas3D/character-creation-big-five` on Hugging Face, repository type `dataset`.

## Resolve and pin before reading

```python
from huggingface_hub import HfApi
repo = "Atlas3D/character-creation-big-five"
revision = HfApi().dataset_info(repo).sha
print(revision)  # Record this full SHA with every derived result.
```

Use that SHA for **every** manifest, batch, asset, and raw-shard download in the analysis.
Do not mix current `main` files with an earlier snapshot. The dataset changes while
capture and publication continue. An absent raw entry is missing publication coverage,
not a zero activation. Failed and incomplete source captures do not become successes
because another stage finished.

## Public files

| File | Contract |
| --- | --- |
| `scenarios.jsonl` | Original synthetic prompt collection; stable `id`, `prompt`, design labels and generation provenance. |
| `manifest.json` | `character-creation.public-big-five.v1`; exact row count, source SHA-256 and exported JSONL SHA-256. |
| `research_manifest.json` | `character-creation.public-research.v1`; immutable metadata batches, record IDs, hashes, counts and source progress. |
| `research_batch_<sha256>.jsonl` | Heterogeneous records; dispatch on `kind`, not guessed columns. |
| `research_raw_manifest.json` | `character-creation.public-research-raw.v1`; separately verified raw-array publication coverage. |
| `research_raw_<sha256>.tar` | Flat, immutable shard of original NPZ files; no corpus database. |
| `research_raw_entries_<sha256>.json` | Raw record-to-entry map, exact bytes, original SHA-256, shapes and dtypes. |
| `research_assets.json` | `character-creation.public-research-assets.v1`; calibration and authored-suite filenames, hashes, sizes and model binding. |
| `layer_00.npz` … `layer_63.npz` | Small calibration banks: relative-dose scales, trait directions, selected neuron columns and observed statistics. Not the full model. |
| `persona_*.json` listed by `research_assets.json` | Authored probe suites, public presets, model metadata and reference recipes. |

Default `datasets.load_dataset(repo, name="default", split="train", revision=revision)`
loads **only the synthetic scenarios**. `train` is a loader convention; all these rows
are discovery material. Use the research manifests for mixed research records and arrays.

## Read with verification, without inference

Install `huggingface_hub` and `numpy` in your existing Python environment. From the
source checkout:

```bash
python persona/persona_read_research.py --revision FULL_HUB_COMMIT_SHA --limit 3
python persona/persona_read_research.py --revision FULL_HUB_COMMIT_SHA --kind baseline_answer --limit 3
```

Programmatic use:

```python
from persona.persona_read_research import Snapshot

snapshot = Snapshot(revision="FULL_HUB_COMMIT_SHA")
for row in snapshot.records(kind="prefill_activation"):
    print(row["id"], row["source"]["id"], row["coverage"])
    break
```

Raw access is explicit because one shard can be hundreds of MB:

```python
arrays = snapshot.arrays(record_id)
peak = arrays["search_peak"]           # [64, 17408], signed maximum-magnitude value
where = arrays["search_peak_position"] # [64, 17408], encoded prompt position
total = arrays["search_sum_abs"]       # [64, 17408], absolute activation sum
mean_abs = total / len(arrays["search_positions"])
```

The helper verifies SHA-256, retains a fixed Hub revision, reads tar entries without
extracting paths, and loads NPZ with `allow_pickle=False`. Download only shards needed
for the question. Preserve record IDs and exact model/runtime/prepared identities in
derived tables. Check the raw entry's shapes/dtypes; do not assume every record kind
contains the same arrays.

## Record kinds and joins

`prefill_activation`:

- Stable record `id`; join to the prompt using `source.id`, not row order.
- `source` supplies approved prompt text, text hash, tier and source binding.
- `identity` contains `model_identity`, `runtime_identity`, `prepared_identity`.
- `request_sha256`, request settings, encoded tokens, `coverage`, validated artifact hashes.
- `candidates` contains at most 32 selected neurons with `layer`, `neuron`, `peak`,
  `peak_position`, `mean_abs`, and full selected per-token `values`.
- `raw_arrays_in_this_json=false`; original all-neuron reductions use the raw manifest.

`baseline_answer`:

- Same prompt/identity/request join discipline; includes a completed unsteered `answer`,
  finish reason, usage and final-prefill artifact references.
- Only revalidated complete source-ledger records qualify. Truncation, failed or uncertain
  inference remains visible in aggregate source counts and does not enter completed pairs.
- Final-prefill MLP/residual arrays are not answer-token activation trajectories.

`qualified_labels` and `coherence`:

- `bundle_sha256` binds a reviewed `bundle` of schema
  `character-creation.public-research-bundle.v1`.
- Bundle `validation.passed` means **export validation**. Inspect individual evidence
  and recommendations for scientific pass/fail/unknown outcomes.
- Label evidence includes all six tested concepts, including failed qualification;
  original evidence hashes identify retained originals after deployment metadata removal.
- Coherence evidence includes complete trials, reused-output provenance and failed heldout
  confirmation. Reuse is not independent replication. Cancelled pilot runs are excluded.

## Measurement semantics: do not collapse these distinctions

1. All-neuron prefill reductions aggregate **all covered encoded prompt positions**, including
   template tokens. They do not retain the full `[layer, token, neuron]` tensor.
2. Selected token profiles retain trajectories for the selected neurons only.
3. Coactivity is an undirected Pearson association. Nearest-neighbor membership can be
   asymmetric; `source`/`target` storage order is not causal direction.
4. Weight paths are signed adjacent-layer output/read affinities. They omit attention,
   normalization and input-conditioned gating; they are not proven functional circuits.
5. Per-layer PCA axes do not share a physical cross-layer coordinate frame.
6. A feature may have positive or inverse activation association. That does not determine
   the dose sign needed to induce behavior.
7. A visible short path does not prove intervening layers can be skipped. No circuit-memory
   compiler or accelerated sparse execution is claimed by this release.

## Model binding and steering

Reference checkpoint: `nvidia/Qwen3.8-27B-NVFP4`, revision
`482ca0f3832238542f8f5295dde86b5f22711d80`; 64 layers, hidden width 5120,
MLP width 17408. Canonical model metadata is `data/persona_model.json` in Git and
`persona_model.json` in HF. Recorded model identity:
`389801c06d68a0d84bd4890b8ce6d26f96e0ee53eb108bcc572c0c09abf5b9a8`.

Neuron addition at the down projection uses:

```text
requested input addition = dose * median(norm(residual leaving layer L), prompt tokens 1:) / norm(W_down[:, neuron])
```

Preserve layer/neuron indices, intervention site, phase, calibration hashes, normalized
dose and actual prepared program identity. Neuron-input addition and post-down column
addition are separately labelled operations. Quantization and rounding can distinguish them.
Do not transfer reference Qwen3.5-4B coordinates to the 27B checkpoint. Do not silently
clamp a fixed recipe or apply a recipe to a mismatching checkpoint.

The public catalog contains 16 authored starting profiles plus six checkpoint-bound
experimental recipes. Baseline has no pushes. GPT-4o-inspired and Skippy-inspired names
do not imply original model weights, endorsement, or validated character fidelity.

## Evaluation contracts

Local viewer integration uses `persona/persona_overnight_integrate.py`. It reads completed
rows of the frozen overnight ledger in short transactions, validates source/request/model
bindings and raw-file hashes, and incrementally caches final-prefill values at the canonical
map coordinates. It starts no inference. Exact message/template/runtime duplicates count
once **within each split**; discovery and heldout observations are never pooled.

The private `results/persona_overnight_viewer.json` manifest binds two compact NPZs
(`persona_overnight_viewer.npz`, discovery; `persona_overnight_viewer_heldout.npz`, heldout)
and `persona_overnight_viewer_records.jsonl` by SHA-256. It is published last. The NPZs
use `character-creation.coactivity.v1`, `final_prefill_mlp_input_last`, and the unchanged
canonical map SHA. `imported_captures` includes duplicates; `unique_observations` is the
sum of `discovery_observations` and `heldout_observations`. Heldout means reserved source
observations, not a newly validated causal result. Original all-position reductions remain
in raw captures. Private book evidence in these files must not enter public Hub exports.

`/api/neighbors` and `/api/neuron-examples` accept explicit dataset selection; `/api/state`
reports the active source and import coverage. Automatic selection prefers a valid overnight
discovery snapshot and explicitly reports a historical fallback when that snapshot is absent
or fails validation. Numeric historical measurements and calibration banks are preserved.

Concept qualification freezes candidates on discovery before heldout probes. A compact
label requires at least 20 distinct predicted-positive heldout families, measured precision,
a 95% Wilson lower bound **strictly above .70**, balanced accuracy at least .80, recall at
least .75 and performance above the nuisance baseline by more than .05. Probe-family IDs
do not establish independent natural examples. Report conditional precision, sample count,
false positives/negatives, and source hashes—not universal semantic confidence.

Coherence measures supported facts, missing requirements, contradictions and observable
collapse. A schema representation mismatch can be unknown instead of a factual failure.
Do not grade optimism, politeness or a preferred personality as coherence. Free prose is
not certified by a passing fact ledger. Freeze the selected discovery dose before heldout
confirmation; a heldout failure must not tune a replacement on that same holdout. Recovery
operates between full responses, not a proven live-token controller. Interaction tests are
exploratory until independently confirmed.

## Operating a fork

- CPU dependencies: `python -m pip install -r requirements-persona.txt` (includes optional
  local NLP tools and a hash-pinned spaCy language model). Analysis-only users can install
  just `datasets huggingface_hub numpy`. Pytest is an additional test dependency.
- Native model execution requires the separately pinned
  [Flare runtime](https://github.com/Atlas3DSS/flare-local-inference); CPU dependency install
  does not prepare GPU graphs or download Qwen. Check local model caches before downloads.
- Configure admitted endpoints and model/calibration bindings before launching
  `python persona_explorer.py --config persona_explorer.local.json --port 8792`.
- Explorer config accepts `nlp_python` (default current interpreter) and
  `overnight_remote: {ssh_host, hostname, remote_root}`. Set your own values; reference
  defaults describe the original fleet, not resources a fork may use automatically.
- Activation launcher accepts `--config`, `--jobs-config` and optional fleet keys
  `remote_project_root`, `heliopause_path`, `model_path`, `calibration_dir`.
- Reference multi-card schedulers, GPU leases, thermal checks and memory admission require
  adaptation to your infrastructure. Single-card/multi-request behavior is not qualified by
  the six independent B1 workers. Never remove locks to claim concurrent request safety.
- Detailed setup: [Explorer guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/PERSONA_EXPLORER.md),
  [porting guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/fork_agent.md).

## Publication lifecycle

`persona/persona_hub.py` manages the approved synthetic prompt subset.
`persona/persona_public_bundles.py` prepares source-bound, sanitized evaluation bundles and
calibration assets; it does not upload them. `persona/persona_hub_research.py` publishes
verified research batches from read-only capture ledgers.

- Metadata watcher polls every five minutes; commits at 500 accumulated records, after
  15 minutes, or when its source run reaches a terminal state. Completed reviewed bundles
  accompany the metadata publication.
- Raw watcher runs on the capture host and uploads up to 32 original NPZ records per
  immutable tar shard directly to HF. It continues draining the backlog after capture ends.
- Both verify the pinned remote manifest and uploaded file hashes before advancing durable
  checkpoints. An uncertain commit reconciles its immutable intent; it never reruns inference.
- Metadata and raw counts are independent. Read `research_raw_manifest.json` for verified
  array coverage even if the dataset card or metadata snapshot has an earlier count.
- Run without `--publish` to validate policy/source files only. Publication requires an
  explicit local public-release policy binding the exact synthetic corpus and reviewed
  authored sources. Do not replace those checks with a recursive directory upload.
- Credentials, private books, book-derived token IDs/profiles, arbitrary interactive prompts,
  local configs and proprietary research stay private. Book tiers may contribute aggregate
  progress counts only. Unpublished/failed records remain in the local audit trail.

Core CPU validation uses `tests/test_persona_*`; native CUDA qualification is separate and
requires an admitted GPU/runtime. NLP tests require the pinned spaCy model. Test skipping
for absent CUDA is not GPU qualification.

## Attribution and license boundaries

Credit [Lyra / neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) for the
foundational steering workflow; this is an independent implementation. Runtime lineage
and other credits are recorded in the source guide and Flare repository. Synthetic prompts
are CC BY 4.0. Do not extend that license claim to the upstream checkpoint, derived model
columns, third-party tools, or private books. Preserve their applicable terms and provenance.
