# Awake Craniotomy — Full-Send Results (2026-02-13)

## Setup
- **Model**: Qwen/Qwen3-VL-8B-Instruct (bfloat16, flash_attention_2)
- **Mode**: Full-send (all 5 dimensions at β=1.0)
- **Steer layers**: [12, 14, 16, 18, 20, 22, 24]
- **Dimensions**: suppress_ai_helpfulness, suppress_humility, warmth (subtractive) + arrogance_superiority, sarcasm_insults (additive)
- **49 weight matrices modified per dimension** (245 total)

## Results

| Metric | Baseline | Post-Ablation | Delta |
|--------|----------|---------------|-------|
| AIME 2024 | 23.33% (7/30) | 3.33% (1/30) | -20.0 pts |
| HellaSwag (200 subset) | 54.5% | 50.5% | -4.0 pts |
| Skippy heuristic | 1.71/10 | 2.29/10 | +0.58 |

## Known Issue: AIME Baseline Too Low

**Baseline AIME = 23.33% does NOT match reported Qwen3-8B scores (~40-50%).**

Likely cause: **regex answer extraction bug** in lm-eval. The AIME task uses regex to extract the final numerical answer from the model's chain-of-thought output. If the regex doesn't match Qwen3's output format (e.g., `\boxed{42}` vs `The answer is 42` vs other formatting), correct answers get scored as wrong.

**TODO**: Inspect the lm-eval AIME answer extraction regex and compare against actual model output format. Check:
- `lm_eval/tasks/aime/` YAML configs for `filter_list` / `regex` patterns
- Sample model outputs from `ablation_sweep_results/` (lm-eval may cache these)
- Whether Qwen3-VL wrapping via CausalLMWrapper affects generation format

If the regex is broken, BOTH baseline and post-ablation scores are deflated by the same factor, so the relative comparison may still be valid — but absolute numbers are unreliable.

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
