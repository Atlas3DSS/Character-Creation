# Phase 1: Initial Extraction & Training (Feb 13-16, 2026)

Skippy personality extraction from ExForce books, LoRA training rounds R1-R5,
neuron probe development, identity circuit discovery.

## Key Findings
- Dim 994 is THE identity neuron (z=-13.96 at L9)
- LoRA SFT destroys AIME (0%) — personality and reasoning share weight space
- SDFT step 500 sweet spot, 5% data contamination causes persistent confusion
- R5 neuron-guided training: best checkpoint at Step 200

## Scripts (60 files)
See individual scripts for docstrings. Key scripts:
- `extract_skippy_dialogue.py` / `extract_skippy_v2.py` — Book dialogue extraction
- `neuron_guided_training.py` — R5 neuron-guided SDFT (best adapter: `skippy_sdft_r5/best_adapter/`)
- `skippify_evals.py` — Generate skippified evaluation data
- `train_sdft_r3.py` / `eval_sdft_r3.py` — SDFT Round 3

## RESEARCH_NOTES.md References
- "Phase 1: Foundation" section
- "Neuron Probe Findings" section
- Commit: `10696c2` (archive move), `e2ace3d` (R4 training)
