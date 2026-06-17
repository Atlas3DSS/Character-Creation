# Matched-Length Sham Controls on Qwen 3.5 9B (`ages` slice)

Date: 2026-04-16
Output root: `sweep_v4/meta_sham_control_ages_qwen35_20260416`

## Setup

Controlled prompt family: `ages` (96 rows), selected because the prior `think_only` screen placed it in the target 50-80% difficulty band.

Conditions:

1. `think_only`
2. `real_meta`: one real `/meta-think` block + `/think`
3. `sham_meta`: matched-length `/meta-think` block filled with fixed non-semantic concrete-word filler
4. `generic_prep`: matched-length generic pre-think notes, not persona/task-specific

Goal: test whether the `budget 1` gain is explained by *semantic control content* versus *just adding a same-length preamble / scratchpad*.

## Final results

| Condition | Accuracy | Mean Completion Tokens | Mean Latency |
| --- | ---: | ---: | ---: |
| `think_only` | `62.5%` | `173.7` | `69.6s` |
| `real_meta` | `73.96%` | `249.2` | `72.0s` |
| `sham_meta` | `50.0%` | `184.6` | `73.7s` |
| `generic_prep` | `54.17%` | `170.6` | `74.8s` |

## Paired vs `think_only`

### `real_meta`
- marginal fixes: `21`
- marginal regressions: `10`

### `sham_meta`
- marginal fixes: `14`
- marginal regressions: `26`

### `generic_prep`
- marginal fixes: `11`
- marginal regressions: `19`

## Direct condition comparisons

### `real_meta` vs `sham_meta`
- `real_meta` only wins: `30`
- `sham_meta` only wins: `7`

### `real_meta` vs `generic_prep`
- `real_meta` only wins: `30`
- `generic_prep` only wins: `11`

### `real_meta` unique fixes vs both controls
Tasks where `think_only` was wrong, `real_meta` was correct, and both controls failed:
- count: `9`

## Interpretation

This materially weakens the "it's just more scratchpad" explanation.

Why:

- `real_meta` beats `think_only` by `+11.46` accuracy points.
- Both matched-length controls are *worse* than baseline.
- `real_meta` has `30` direct wins over each control.
- There are `9` baseline failures that only the real meta condition fixes while both controls still fail.

So the most defensible reading is:

- the gain is not explained by adding a same-length preamble,
- the gain is not explained by adding a generic extra planning block,
- the *semantic content* of the control-plane scaffold matters.

## Mechanistic take

This still does not imply a new module or basin switch. The better interpretation is:

- the model is using shared reasoning machinery,
- but a real `/meta-think` block biases that machinery in a useful way,
- while filler or generic preparatory text does not reproduce the same bias.

That is consistent with prompt-conditioned gating / prior-setting on a shared representational substrate, not a separate "meta circuit."
