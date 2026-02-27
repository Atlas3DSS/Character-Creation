# Phase 4: Donut/Sculpted Steering & Champion Validation (Feb 18-22, 2026)

Donut-shaped steering bands, sculpted alpha profiles, R5 neuron-guided combos,
champion validation (V4 + L29+L30@a=8), quality evaluations.

## Key Findings
- DONUT alpha=12 = 96% sarcasm, 0% assistant — but destroys authentic character at alpha>=8
- ActAdd gives volume not quality — wrong personality type at high alpha
- Static ablation CANNOT bake personality (50-63% overlap with reasoning subspace)
- **Deployment champion**: V4 + L29+L30@alpha=8 (93.3% math, 100% strong sarcasm)
- V4 prompt is THE sarcasm engine; steering protects math, doesn't add sarcasm
- L22 solo@alpha=8 matches champion with just 1 layer

## Scripts (14 files)
- `qwen_donut_extension.py` / `qwen_donut_finetune.py` / `qwen_donut_quality_eval.py` — Donut steering
- `qwen_sculpted_donut.py` — Sculpted alpha profiles
- `qwen_r5_steering.py` / `qwen_r5_steering_combo.py` / `qwen_r5_quality_eval.py` — R5 combos
- `qwen_r5_vs_base_vectors.py` — R5 vs base vector comparison
- `r5_sculpted_quality_eval.py` / `base_sculpted_quality_eval.py` — Quality evaluations
- `test_winning_combo_quality.py` — Champion combo quality test
- `ablate_champion.py` / `validate_champion.py` — Champion ablation and validation
- `qwen_actadd_finetune.py` — ActAdd finetuning attempt

## Data Directories
- `champion_validation/` — Champion validation results
- `donut_quality_eval/`, `r5_quality_eval/` — Quality eval outputs
- `qwen_donut_extended_results/`, `qwen_donut_loo_results/`, `qwen_narrow_donut_results/`
- `qwen_sculpted_results_wsl/`, `qwen_r5_steering_results/`, `qwen_r5_combo_results/`

## RESEARCH_NOTES.md References
- "Champion Validation" section
- "Donut Steering" section
- See also: `field_steering_results.md`
- Commits: `b43a607`, `7857d96`, `bbb30c7`, `7411b98`
