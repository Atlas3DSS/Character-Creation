# Phase 2: GPT-OSS-20B Cross-Architecture (Feb 16-18, 2026)

Cross-architecture validation using GPT-OSS-20B (3.6B active MoE, 24 layers, hidden=2880).
Tested whether sarcasm steering findings from Qwen 8B transfer to a different architecture.

## Key Findings
- Personality is distributed in GPT-OSS too (max |z|=1.68, no clear gen/sup structure)
- Field steering winner: attractor+field = 47% sarcasm
- Dim 368 appears in BOTH Qwen and GPT-OSS probes
- Confidence steering showed weak signal
- CoT probing: personality bleeds into reasoning tokens

## Scripts (12 files)
- `probe_gptoss_neurons.py` — Initial neuron probing
- `probe_gptoss_comprehensive.py` — Full 20-category probe
- `probe_gptoss_cot.py` / `probe_gptoss_deep_cot.py` — Chain-of-thought probing
- `train_gptoss_skippy.py` / `merge_gptoss_skippy.py` / `eval_gptoss_skippy.py` — LoRA training pipeline
- `field_analysis_gptoss.py` / `gptoss_field_steering.py` / `field_test_unprompted.py` — Field steering
- `gptoss_alpha_sweep.py` — Alpha parameter sweep
- `gptoss_confidence_steering.py` — Confidence-based steering

## Data Directories
- `skippy_gptoss/`, `skippy_gptoss_v2/`, `skippy_gptoss_v3/` — Model checkpoints (in .gitignore)
- `skippy_gptoss_fresh/` — Fresh experiment data (in .gitignore)

## RESEARCH_NOTES.md References
- "GPT-OSS-20B" section
- See also: `gptoss_findings.md`
- Commits: `55da083`, `85ae498`, `95613f2`
