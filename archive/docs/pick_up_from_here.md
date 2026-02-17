# Pick Up From Here — 2026-02-15 ~09:30

## GRPO V3 — RUNNING (almost done)
- **PID**: 95710
- **Log**: `/tmp/grpo_v3_train2.log`
- **Step**: ~252/318, ETA ~15 min
- **Base**: `./skippy_grpo_base` (delta α=0.7)
- **Output**: `./skippy_grpo_v3_output/`
- Delta logging to `./skippy_grpo_v3_output/character_deltas/`

## SVD Precomputation — RUNNING
- **PID**: 97606 (restarted — fixed Qwen3-VL layer path)
- **Log**: `/tmp/svd_precompute.log`
- **What**: Computing top-16 SVD of all LoRA-targeted weight matrices in skippy_grpo_base
- **Output**: `./svd_projectors/` (Uk, Vk per layer per module)
- Running on CPU (~55s/layer, 36 layers, ~33 min total)
- Fixed: `model.language_model.layers` not `model.model.layers`

## NEXT: GRPO V4 (OPLoRA-Constrained)
When both above finish:
```bash
python train_skippy_grpo_v4_oplora.py --base-model ./skippy_grpo_base --k 16
```
- Same hyperparams as V3 but with orthogonal gradient projection
- Gradient hooks project LoRA A/B gradients into orthogonal complement of top-k singular directions
- Should mathematically prevent personality updates from touching reasoning subspace
- Compare V4 AIME vs V3 AIME to validate

## Eval Plan (after V3 and V4 complete)
1. Merge V3 final adapter onto skippy_grpo_base, eval banal + AIME
2. Merge V4 final adapter onto skippy_grpo_base, eval banal + AIME
3. Compare character delta trajectories (personality_energy_ratio over time)
4. V4 should show: similar personality reward but higher AIME preservation

## Files Created This Session
- `train_skippy_grpo_v3.py` — GRPO with ND character delta logging
- `train_skippy_grpo_v4_oplora.py` — GRPO with OPLoRA gradient constraints
- `precompute_svd_projectors.py` — SVD precomputation for OPLoRA
- `paper_scaffold.md` — paper outline (gitignored)
- `arxiv_papers/` — 12 downloaded papers + summaries (gitignored)

## Arxiv Top 3
1. OPLoRA (2510.13003) — orthogonal projection LoRA → V4 implements this
2. Personality Subnetworks (2602.07164) — training-free pruning masks
3. BiPO (2406.00045) — preference-optimized steering vectors

## Dev Server
- 3090: vLLM Qwen3-VL-8B port 8000
- 4090: nano TTS port 8081
- Both GPUs occupied by servers
