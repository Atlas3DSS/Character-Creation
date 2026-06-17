# Book-Derived Character Prefill Dataset Fill and Rebalance

Date: 2026-04-17

## Goal
Tighten the weak behavior buckets in the book-derived prefill dataset, specifically:
- `selective_introspection`
- `conflict_detection`

## Changes Made
- Updated `build_book_character_prefill_dataset.py` to support:
  - streaming API decode with doom-loop monitoring
  - `--behavior-allowlist`
  - `--max-items-per-source-title`
  - targeted structuring guidance for specific behaviors
- Added `package_book_character_prefill_dataset.py` to:
  - merge multiple runs
  - dedupe near-identical paired items
  - keep the strongest pass/fail pair
  - emit a behavior-balanced final package
  - enforce per-behavior source-title caps during packaging

## Runs
Base dataset:
- `sweep_v4/book_character_prefill_dataset_qwen36_v2_20260417_130752`

Targeted fill dataset:
- `sweep_v4/book_character_prefill_fill_si_cd_curated_v3_20260417_145504`

Final balanced package:
- `sweep_v4/book_character_prefill_dataset_balanced_v3_20260417_150446`

## Fill Result
Targeted fill summary:
- 14 curated books across `Unseen`, `Not_Robert Jordan`, `Sanderson`, `Robert Jordan`
- 140 scene candidates
- 80 usable targeted items selected
- 62 paired items survived judge filtering
- 124 completions total
- behavior completions:
  - `conflict_detection`: 64
  - `selective_introspection`: 60

## Final Balanced Dataset
Summary:
- merged paired items available: 290
- balanced target per behavior: 40
- final selected items: 200
- final completions: 400
- exact per-behavior balance:
  - `conflict_detection`: 40 items / 80 completions
  - `constraint_preservation`: 40 / 80
  - `repair_after_challenge`: 40 / 80
  - `selective_introspection`: 40 / 80
  - `state_carryover`: 40 / 80
- split sizes:
  - `train`: 320
  - `val`: 40
  - `test`: 40
- mean trace quality: 3.85

## Assessment
Current library is sufficient for this stage.
No additional books are required right now to achieve a balanced, diverse seed corpus.
More books would only become useful if we want to push source diversity further beyond the current 24-title merged pool or expand beyond 40 items per behavior.
