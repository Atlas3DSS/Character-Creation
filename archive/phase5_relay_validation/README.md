# Phase 5: Sarcasm Relay Circuit & Pair Validation (Feb 19-24, 2026)

Validation of the 5-node sarcasm relay circuit (L9->L14->L15(inv)->L22->L26),
systematic pair testing, susceptibility DPO training.

## Key Findings
- Sarcasm relay circuit confirmed: L9->L14->L15(inv)->L22->L26
- LOO analysis: removing "dampener" layers (individually suppress sarcasm) HURTS overall sarcasm
- Narrow donut: 16% sarcasm with amplifiers-only vs 60% with full donut including dampeners
- Dampeners are necessary for distributed signal propagation
- Pair validation (7 conditions x 130 prompts): L08+L15@alpha=8 = lowest assistant leak (1.7%)

## Scripts (2 files)
- `validate_2layer_pairs.py` — Systematic pair validation (7 conditions, 130 prompts each)
- `susceptibility_training.py` — Susceptibility-based DPO training pipeline

## Data Directories
- `pair_validation/` — Pair validation results (in .gitignore)

## RESEARCH_NOTES.md References
- "Sarcasm Relay Circuit" section
- "Pair Validation" section
- Commits: `847ca45` (cross-layer probe, susceptibility), `0d354f1` (validation + variance)
