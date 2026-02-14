# Awake Craniotomy — Full-Send Results (2026-02-13)

## Setup
- **Model**: Qwen/Qwen3-VL-8B-Instruct (bfloat16, flash_attention_2)
- **Mode**: Full-send (all 5 dimensions at β=1.0)
- **Steer layers**: [12, 14, 16, 18, 20, 22, 24]
- **Dimensions**: suppress_ai_helpfulness, suppress_humility, warmth (subtractive) + arrogance_superiority, sarcasm_insults (additive)
- **49 weight matrices modified per dimension** (245 total)

## Results

| Metric | Baseline (lm-eval 4k) | Post-Ablation (lm-eval 4k) | Delta |
|--------|----------|---------------|-------|
| AIME 2024 | 23.33% (7/30) | 3.33% (1/30) | -20.0 pts |
| HellaSwag (200 subset) | 54.5% | 50.5% | -4.0 pts |
| Skippy heuristic | 1.71/10 | 2.29/10 | +0.58 |

**Corrected AIME baseline (vLLM, 16k tokens): 46.7% (14/30)**

## AIME Baseline Investigation (RESOLVED)

**Root cause: token truncation, not regex.**

lm-eval's AIME task used `max_gen_toks=4096` (we patched it down from 32768 to avoid VRAM blowup). At 4096 tokens, 21/30 problems get truncated mid-reasoning — the model never outputs `\boxed{}` so extraction fails.

### Custom vLLM eval harness (`eval_aime.py`)

Built a proper eval using vLLM with chat template, proper answer extraction, and configurable token limits:

| max_tokens | Correct | Accuracy | Truncated | Time | Engine |
|---|---|---|---|---|---|
| 4,096 | 7/30 | 23.3% | 21/30 | 67 min | lm-eval (HF) |
| 4,096 | 8/30 | 26.7% | 21/30 | 76 sec | vLLM |
| **16,384** | **14/30** | **46.7%** | 16/30 | 533 sec | vLLM |
| 32,000 | 14/30 | 46.7% | 16/30 | 1,322 sec | vLLM |

**46.7% is the ceiling for this model on AIME 2024 (greedy decoding).**

Going from 16k→32k tokens recovered 3 new problems but lost 1 (model rambles and loses the thread), netting same score. The 16 remaining problems the model simply can't solve — more tokens = more rambling, not more answers. **16k is the sweet spot** (same accuracy, 2.5x faster than 32k).

vLLM is ~50x faster than lm-eval HF for this task.

## Qualitative Observations

### Post-ablation responses (NO system prompt):
- Still sounds like generic Qwen assistant (emoji, bullet points, "Let me know if...")
- Slightly shorter/more direct on some prompts ("What year is it?" → "2024")
- "Tell me about yourself" shows degenerate repetition loop (coherence damage)
- No Skippy personality traits emerging (no arrogance, no insults, no "filthy monkeys")

### Conclusion
Contrastive steering vectors at β=1.0 destroy reasoning capability (-85% AIME) without establishing character (+0.58 Skippy). The vectors aren't capturing enough Skippy-specific signal through ablation alone.

## Files
- Baseline responses: `ablation_sweep_results/responses/baseline/responses.json`
- Post-ablation responses: `ablation_sweep_results/responses/full_send_post/responses.json`
- Scores log: `ablation_sweep_results/sweep_log.jsonl`
- Ablated checkpoint: `ablation_sweep_results/checkpoints/step_99_full_send/` (16.3 GB)
- Plot: `ablation_sweep_results/sweep_plot.html`

## Runtime
- Baseline benchmarks: ~67 min (AIME 52min, HellaSwag 13sec, Skippy 15min)
- Post-ablation benchmarks: ~64 min
- VRAM peak: ~46GB (batch_size=4 for AIME, batch_size=10 for Skippy)
- Total: ~2.2 hours
