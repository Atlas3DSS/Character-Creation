# Archive Index

Non-destructive archive of completed experiment scripts, organized by research phase.
All files are recoverable via `git log --follow -- <filename>` to trace full history.

## Directory Structure

```
archive/
  scripts/          Phase 1: Initial extraction & training (Feb 13-16)     60 scripts
  phase2_gptoss_20b/         GPT-OSS-20B cross-architecture (Feb 16-18)   12 scripts
  phase3_qwen8b_analysis/    Qwen 8B deep analysis (Feb 17-22)            21 scripts
  phase4_donut_sculpted/     Donut/sculpted steering (Feb 18-22)           14 scripts
  phase5_relay_validation/   Relay circuit & pair validation (Feb 19-24)    2 scripts
  phase7_arena_v1v3_spectral/ Superseded arena v1-v3 & GMR (Feb 26-27)     4 scripts
  docs/             Superseded documentation                               10 docs
  eval_results/     Old evaluation results
  models/           Old model checkpoints
```

## What's Active (project root)

These 16 scripts are the current working set:

| Script | Purpose | Phase |
|--------|---------|-------|
| `capture_and_steer_27b.py` | 27B activation capture + steering | 8 |
| `codex_research_conversation.py` | Codex code review sessions | 8 |
| `compare_connectomes.py` | Base vs abliterated connectome comparison | 8 |
| `debate_arena_v4.py` | Dual-model debate arena (latest) | 7 |
| `doom_loop_detector.py` | Doom loop detection module | 7 |
| `eval_head_to_head.py` | Head-to-head model evaluation | 8 |
| `fast_layer_scan.py` | Fast 27B layer scanning | 6 |
| `fullrank_spectral_analysis.py` | Full-rank spectral analysis (10K samples) | 7 |
| `gemini_research_conversation.py` | Gemini research consultation | 8 |
| `generate_prompts_10k.py` | 10K math + sarcasm prompt generation | 7 |
| `gpu_monitor.py` | GPU utilization monitoring | infra |
| `household_config.py` | Household/config utilities | infra |
| `magnitude_calibrated_steering.py` | Magnitude-calibrated alpha scaling | 7 |
| `map_qwen35.py` | Qwen3.5 connectome mapping (4 phases) | 6 |
| `orchestrate_overnight.py` | Overnight experiment orchestration | infra |
| `relay_alpha_map.py` | Relay circuit alpha mapping | 5/7 |

## Cross-Reference

Each phase README links to the corresponding RESEARCH_NOTES.md section and git commits.
To find when a script was created: `git log --diff-filter=A -- <filename>`
To find all changes to a moved script: `git log --follow -- archive/<phase>/<filename>`

## Archive Date
2026-02-27 — Organized by Claude Opus 4.6 based on research phase structure.
