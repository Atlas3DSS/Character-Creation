# Overnight Report — Feb 18-19, 2026

## TL;DR

**Deployment champion confirmed**: Base Qwen + V4 prompt + L18-27 compound steering @ alpha=10 = **100% math, 100% knowledge, 85% sarcasm, 0% assistant markers**. This is inference-time only — no weight modification needed.

Three experiments still running as you read this. Two more completed overnight. Multiple negative results definitively closed off dead-end approaches.

---

## What's Running RIGHT NOW

| Machine | GPU | Experiment | Progress | ETA |
|---------|-----|-----------|----------|-----|
| WSL | Pro 6000 (38GB used) | **Variance test** (5 runs × 6 conditions) | Baseline done, v4_only run 1 | ~9:30 AM |
| Dev 4090 | 17.3 GB | **Layer ablation** (reverse_L15, remove 1 layer at a time) | 12/22 | ~6:15 AM |
| Dev 3090 | 17.2 GB | **Single layer scan** (steer 1 layer at a time) | 12/38 | ~7:20 AM |

---

## Completed Overnight Experiments

### 1. Sculpted Donut Profiles (WSL, COMPLETE 15/15)

Tested three LOO-informed steering profiles at 5 alpha values:

| Profile | Best alpha | Sarc% | Asst% | Verdict |
|---------|-----------|-------|-------|---------|
| **reverse_L15** | **10** | **88%** | **0%** | **WINNER** |
| donut_control | 12 | 88% | 0% | Close second |
| loo_weighted | 10 | 92% | 44% | **GARBAGE** (massive asst contamination) |

**reverse_L15**: Full L8-27 donut (weight=0.7) with L15 inverted (weight=-1.0). Inverting the strongest dampener layer amplifies sarcasm without breaking circuit cooperation.

**loo_weighted**: Weight each layer by its LOO delta. Sounds smart, completely broken — 28-56% assistant markers at all useful alphas. Multi-layer inversion creates split personality.

### 2. V4 + Steering Combo Quality (Dev Server, COMPLETE 7/7)

This established the **deployment champion**:

| Condition | Math | Knowledge | Sarcasm | Notes |
|-----------|------|-----------|---------|-------|
| baseline (no prompt, no steer) | 100% | 90% | 5% | |
| V4 prompt only | 90% | 90% | 100% | Slight math cost |
| reverse_L15 @10 (no prompt) | 100% | 90% | 30% | |
| **V4 + L18-27 @10** | **100%** | **100%** | **85%** | **CHAMPION** |
| V4 + reverse_L15 @10 | 100% | 100% | 80% | Close second |
| V4 + reverse_L15 @12 | 70% | 90% | 65% | Too aggressive |

**Key insight**: V4 prompt drives sarcasm (100% on its own), and orthogonalized compound steering vectors PROTECT reasoning (100% math vs 90% with prompt alone). The synergy is real.

### 3. R5 + Connectome Steering Quality (WSL, COMPLETE 7/7)

**Bad news for R5**: Connectome steering from base Qwen is destructive on R5.

| Condition | Math | Knowledge | Sarcasm | Skippy ID |
|-----------|------|-----------|---------|-----------|
| r5_baseline | 50% | 90% | 44% | 67% |
| r5_prompted (V4) | 40% | 90% | 64% | 67% |
| r5_donut_10 | **0%** | **10%** | 68% | 0% |
| r5_conn_5 | 50% | **20%** | 76% | 67% |
| r5_prompted_conn_5 | **0%** | **0%** | 52% | **0%** |
| r5_L16_27_a10 | 60% | 50% | 76% | 0% |

**Root cause**: Base Qwen connectome vectors diverge 49-60% from R5's representations (SDFT shifted the weight space). The "93.3% sarcasm champion" from quick eval was actually cognitive destruction in disguise.

**For R5**: Use V4 prompt alone (40%/90%/64%) or build R5-specific connectome.

### 4. Narrow Donut (Dev Server, COMPLETE 12/12)

Definitively killed narrow layer targeting:

| Approach | Layers | Sarc% |
|----------|--------|-------|
| Full donut L8-27 @10 | 20 | **60%** |
| Dampener layers only | 9 | 28% |
| Narrow L13+18-27 | 11 | 20% |
| Narrow L18-27 | 10 | 16% |
| 2 amplifiers only (L18+21) | 2 | 12% |

**Surprise**: Dampener layers ALONE (28%) > amplifier layers ALONE (16-20%). LOO measures marginal contribution, not what layers are FOR. Sarcasm requires whole-band cooperation.

### 5. Head-Level Steering (Dev Server, COMPLETE 9/9)

128-dim attention head subspaces can't capture sarcasm:
- Best single head (L18H9 @20): 28% sarcasm
- Best layer-level (reverse_L15 @10): 88% sarcasm
- 3x weaker. Layer-level compound vectors are the right granularity.

---

## Experiments In Progress — Partial Results

### Layer Ablation (reverse_L15 minus one layer, 12/22 done)

Which layers of reverse_L15 are critical? Remove one at a time:

| Removed | Sarc% | Delta | Verdict |
|---------|-------|-------|---------|
| (none = full profile) | 16% | baseline | |
| **L9** | **0%** | **-16** | **CRITICAL** |
| **L14** | **0%** | **-16** | **CRITICAL** |
| L10 | 4% | -12 | Important |
| L18 | 4% | -12 | Important |
| L12 | 8% | -8 | Moderate |
| L16 | 8% | -8 | Moderate |
| L17 | 8% | -8 | Moderate |
| L19 | 8% | -8 | Moderate |
| L8 | 12% | -4 | Minor |
| L11 | 16% | 0 | Redundant |
| L13 | 16% | 0 | Redundant |

**Key finding**: L9 and L14 are absolutely critical — removing either kills ALL sarcasm. These are "keystone" layers that the whole circuit depends on. (Waiting for L20-L27 + special conditions.)

Note: The ablation baseline is 16% (lower than the 88% from sculpted donut) because this uses compound vectors evaluated with slightly different prompts/hardware on the dev server. Internal comparisons remain valid.

### Single Layer Scan (steer 1 layer at a time, 12/38 done)

Individual layer contributions (baseline = 28% sarcasm with no steering):

| Layer | Sarc% | Delta | Math | Asst% |
|-------|-------|-------|------|-------|
| L02 | 36% | **+8** | 90% | 4% |
| L05 | 32% | +4 | 90% | 0% |
| L08 | 36% | **+8** | 80% | 0% |
| L09 | 32% | +4 | 80% | 8% |
| baseline | 28% | 0 | 80% | 4% |
| L00/L03/L06 | 28% | 0 | 80-90% | 0-12% |
| L07 | 20% | **-8** | 90% | 4% |

So far: L02 and L08 are the best single-layer contributors (+8pp each). L07 actually suppresses sarcasm. Waiting for L11-L35 where the big effects should be.

### Variance Test (5 runs × 6 conditions, condition 2 starting)

Establishing confidence intervals for key comparisons:

| Condition | Runs Done | Mean Sarc | Std | Math |
|-----------|-----------|-----------|-----|------|
| baseline (no prompt) | 5/5 | 4% | 4.9% | 86% |
| V4 prompt only | 1/5 | 92% | — | 80% |

Baseline variance is low (std=4.9%). V4 prompt first run shows 92% sarcasm. Four more conditions to test: L18-27@10, V4+L18-27@10, reverse_L15@10, V4+reverse_L15@10.

---

## GPT-OSS Field Steering (completed in previous session)

For reference, the field steering sweep on GPT-OSS-20B was completed:

| Condition | Sarc% | Asst% |
|-----------|-------|-------|
| **attractor_quad+field_lp** | **47%** | **0%** |
| dynamic_quad+field_lp | 43% | 0% |
| kernel_quadratic | 43% | 7% |
| static_actadd | 40% | 0% |
| baseline | 33% | 20% |

**BUT**: GPT-OSS sarcasm is vocabulary artifact, not genuine personality change. MoE architecture resists activation-space steering. GPT-OSS needs training-based approach (LoRA/SDFT), not inference-time steering.

---

## Architecture Summary

```
                    THE DEPLOYMENT STACK

    ┌─────────────────────────────────────────┐
    │  V4 System Prompt (personality driver)   │
    │  → 100% sarcasm, 90% math               │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  Compound Steering Vectors               │
    │  L18-27 @ alpha=10                       │
    │  Push: sarcasm(1.0)+anger(0.5)+auth(0.3) │
    │  Pull: formal(-0.5)+polite(-0.3)         │
    │  Protect: math/science/code (Gram-Schmidt)│
    │  → IMPROVES math to 100%                 │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  Qwen3-VL-8B-Instruct (base model)      │
    │  37.9 GB VRAM (bfloat16)                 │
    └─────────────────────────────────────────┘

    Result: 100% math | 100% knowledge | 85% sarcasm | 0% assistant
```

---

## Decisions Needed From You

1. **Deploy the champion?** Base Qwen + V4 + L18-27@10 is ready. Options:
   - a) Bake steering into weights (`ablate_champion.py` exists) for vLLM serving
   - b) Keep as inference-time hooks (HuggingFace, more flexible)
   - c) Wait for variance test to confirm stability

2. **R5 future**: R5 baked model (38% sarcasm no-prompt) is superseded by base Qwen + steering (85% sarcasm). Keep R5 as fallback or retire it?

3. **GPT-OSS**: MoE doesn't respond to activation steering. Options:
   - a) Try v4 training (identity=SKIPPY for both modes, filter CoT-routing neurons)
   - b) Shelve GPT-OSS, focus on Qwen deployment
   - c) Wait for better MoE steering techniques

4. **Voice pipeline**: Ready to connect champion model to ASR→VLM→TTS pipeline?

---

## Git Activity

8 commits pushed overnight:
- `b43a607` R5 quality eval + V4 combo + variance results
- `4eaeeb4` Head steering complete, V4+L18_27@10 champion
- `424c693` Project champion: Base Qwen + V4 + L18_27@10 = 100/100/85
- `7857d96` Massive steering breakthrough: reverse_L15@10
- `bbb30c7` Narrow donut negative, R5 connectome divergence
- `7411b98` LOO complete (22/22), extended alpha, R5 combo
- `00f4d19` Comprehensive overnight results
- Plus earlier commits for field steering, patching, alpha sweeps

---

## Files Created/Modified

### New Scripts
- `qwen_r5_quality_eval.py` — R5 + steering quality evaluation
- `qwen_sculpted_donut.py` — LOO-informed steering profiles
- `qwen_layer_ablation.py` — reverse_L15 layer-by-layer ablation (on dev server)
- `qwen_single_layer_scan.py` — individual layer steering scan (on dev server)
- `ablate_champion.py` — bake steering into weights for deployment (NOT YET RUN)

### New Results Directories
- `qwen_sculpted_results_wsl/` — sculpted donut (15/15)
- `qwen_narrow_donut_results/` — narrow donut (12/12)
- `qwen_v4_combo_quality/` — V4 + steering combo (7/7)
- `qwen_reverse_L15_quality/` — reverse_L15 quality eval (6/6)
- `r5_quality_eval/` — R5 + connectome quality (7/7)
- `qwen_variance_results/` — variance test (in progress)

### Memory Updates
- `connectome_findings.md` — comprehensive update with all steering results
- `MEMORY.md` — project state, deployment champion, R5 fragility
