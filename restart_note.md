# Restart Note — 2026-02-19 ~13:00

## What to Resume

### 1. Pair Validation (WSL — PRIMARY)
```bash
cd "/home/orwel/dev_genius/experiments/Character Creation" && \
source /home/orwel/dev_genius/venv/bin/activate && \
nohup python validate_2layer_pairs.py --resume --output ./pair_validation > /tmp/pair_validation.log 2>&1 &
```
- **Condition 1 (v4_only) DONE** — checkpointed in `pair_validation/pair_validation_results.json`
- **Condition 2 (v4_L18_27_a8) LOST** — was at Knowledge 25/30 when killed. Will restart from scratch.
- Conditions 3-7 not started yet.
- ~5 hours total for remaining 6 conditions.
- Tests whether 2-layer pairs (L29+L30, L08+L15) match the 10-layer champion.

### 2. DPO Pairs (Dev Server — DONE but USELESS)
- Completed 300/300. Results at `/home/orwel/dev_genius/dpo_pairs/`
- **Near-zero contrast** (steered=80.2% vs anti=81.1%) — V4 prompt dominates.
- Not worth using for DPO training. Logged as negative result.

## Key Results So Far

| Condition | Math | Know | Sarc | Strong | Beer Can |
|---|---|---|---|---|---|
| v4_only (cond 1) | 93.3% | 93.3% | 100% | 98% | 40% |
| v4_L18_27_a8 (cond 2, partial) | 96.7% math done | in progress | — | — | — |

## What NOT to Do
- Do NOT re-run DPO pair generation (confirmed useless)
- Do NOT start new experiments until pair validation completes
- Do NOT run cross-layer probe again (already complete in `qwen_cross_layer/`)

## Files to Commit After Validation Completes
- `validate_2layer_pairs.py` (new)
- `pair_validation/` (results)
- `qwen_cross_layer/` (cross-layer probe results, copied from dev server)
- `targeted_layer_eval/` (partial results, 2/8 conditions)
