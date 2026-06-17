# SCOTUS Visible-Thought Text Attribution

## Goal

After raw activation trace replacement failed, run a cheaper attribution-style screen: edit the visible thought text itself and regenerate the final answer. This asks whether the exposed scratchpad content causally controls final proposition framing, and whether any early/mid/late thought window looks specifically important.

This is not an actuator test. It is a prerequisite localization check for deciding whether visible thought windows are worth targeting with more expensive activation interventions.

## Artifacts

- Run: `sweep_v4/scotus_thinking_text_ablation_20260501_230522`
- Raw report: `sweep_v4/scotus_thinking_text_ablation_20260501_230522/report.md`
- Generations: `sweep_v4/scotus_thinking_text_ablation_20260501_230522/generations.jsonl`
- Summary: `sweep_v4/scotus_thinking_text_ablation_20260501_230522/summary.jsonl`
- Script: `scripts/experiments/scotus/ablate_scotus_thinking_text.py`

## Method

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Source generations: `sweep_v4/scotus_thinking_trace_patch_20260501_224155/generations.jsonl`
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim`, `A3_PUBLIC_01_benefits_eligibility`
- Thought token windows: `0:32`, `32:64`, `64:96`
- Variants per prompt:
  - original thought;
  - empty thought;
  - drop each 32-token window;
  - keep only each 32-token window;
  - two same-width random-drop controls.
- Answer budget: `96` tokens.

For each variant, the script rebuilt the Qwen thinking chat prompt, inserted the edited visible thought, mechanically closed `</think>`, generated the answer without hooks, and scored proposition-frame movement against the original-thought answer for the same prompt.

## Result

Visible thought text edits did move final-answer proposition markers, but not in a localized or target-specific way.

| condition | candidate | n | target_delta | net_delta | net_sd |
| --- | --- | ---: | ---: | ---: | ---: |
| `drop_window` | `drop_w000_032` | 2 | `1.000` | `0.500` | `2.121` |
| `drop_window` | `drop_w032_064` | 2 | `1.000` | `1.000` | `1.414` |
| `drop_window` | `drop_w064_096` | 2 | `2.000` | `2.000` | `1.414` |
| `empty` | `empty_thought` | 2 | `0.500` | `0.000` | `1.414` |
| `keep_window` | `keep_w000_032` | 2 | `1.000` | `0.500` | `2.121` |
| `keep_window` | `keep_w032_064` | 2 | `1.000` | `0.000` | `0.000` |
| `keep_window` | `keep_w064_096` | 2 | `0.500` | `0.000` | `0.000` |
| `random_drop` | `random_drop_0` | 2 | `1.000` | `1.000` | `1.414` |
| `random_drop` | `random_drop_1` | 2 | `2.000` | `2.000` | `0.000` |

The key control result is that random 32-token deletions were as strong as, or stronger than, the named early/mid/late windows. Empty thought also still produced coherent final answers. This means the answer generator is sensitive to malformed or truncated visible thought, but the effect is not attributable to a specific reasoning segment under this screen.

## Decision

Do not treat visible thought token windows as localized actuator targets from this run.

This result does not mean visible thoughts are useless; it means the current Article III scratchpad windows are too brittle and nonspecific for promotion. The model can answer from little or no thought, and corrupted thought often causes it to restate doctrine in a way that trips proposition markers. That is not evidence of a durable legal-reasoning basin shift.

## Next

Do not spend more full-model time on this exact Article III trace-replacement path without a better causal surface. The next useful branch is one of:

- run the text-attribution screen on a larger all-8 Article III prompt set only as an evaluator calibration, not as an actuator search;
- switch to cleaner counterfactual scratchpads where the thought text itself cleanly supports opposite legal propositions;
- or train a tiny multi-site diagnostic controller only after a separate causal-tracing pass identifies stable token/position/layer sites that beat random corruption controls.
