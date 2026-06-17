# Meta-Sham Control on AIME (Qwen 3.5 9B, dev servers)

## Important Caveat

This run should **not** be interpreted as the native AIME capability of Qwen 3.5 9B.

Why:

- the harness used a custom structured-output contract rather than a standard benchmark prompt,
- the response budget was far smaller than the model card's recommendation for competition-style math,
- and the experiment was designed as a control-plane stress test, not as an official benchmark replication.

So the `0/18` result below is valid as "performance under this constrained harness," but not as "the model's real AIME score."

## Setup

- Model: `Qwen/Qwen3.5-9B`
- Endpoints:
  - `http://192.168.1.90:30001/v1`
  - `http://192.168.1.90:30002/v1`
- Dataset: `experiments/Dual Stream LLM/data/aime_eval.jsonl`
- Conditions:
  - `think_only`
  - `real_meta`
  - `sham_meta`
  - `generic_prep`
- Prompt contract for AIME:
  - answer-first (`Final Answer:` on the first required line)
  - brief post-answer `/think` block
  - compact explanation

## Result

Across 18 AIME questions, all four conditions scored `0/18`.

| Condition | Accuracy | Format OK | Truncated | Mean Completion Tokens |
| --- | ---: | ---: | ---: | ---: |
| `think_only` | `0.0%` | `88.9%` | `0.0%` | `334.6` |
| `real_meta` | `0.0%` | `94.4%` | `5.6%` | `347.4` |
| `sham_meta` | `0.0%` | `66.7%` | `0.0%` | `312.8` |
| `generic_prep` | `0.0%` | `72.2%` | `0.0%` | `398.4` |

## Interpretation

- This is not a scoring bug.
- The answer-first contract produced parseable, benchmark-valid outputs.
- Spot checks show the model is confidently wrong in all conditions.
- Example failures:
  - `AIME_I_01`: predicts `150` or `156`, gold is `204`
  - `AIME_I_02`: predicts `100`, gold is `25`
  - `AIME_I_03`: predicts `1012`, gold is `809`

## What This Means

- The `ages` lift from `real_meta` does **not** automatically transfer to hard olympiad-style math.
- On this benchmark, the limiting factor is base reasoning capability, not control-plane formatting.
- The control scaffold can help on moderate-difficulty structured reasoning, but it does not rescue Qwen 3.5 9B on AIME.
- More precisely: the control scaffold does not rescue Qwen 3.5 9B **under this constrained answer-format harness**.

## Implication for the Main Claim

The current evidence is:

- `ages` slice: semantic `real_meta` helps, matched-length controls do not.
- AIME: all conditions fail equally.

The clean reading is:

- `/meta-think` can improve behavior on tasks already near the model's competence frontier.
- It does not create missing math capability on tasks outside that frontier.
