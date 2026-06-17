# Ages Sham-Control Rerun with Official-ish Qwen Sampling

## Setup

- Model: `Qwen/Qwen3.5-9B`
- Endpoints:
  - `http://192.168.1.90:30001/v1`
  - `http://192.168.1.90:30002/v1`
- Dataset slice: same 96-row `ages` selection used in the original controlled sham run
- Conditions:
  - `think_only`
  - `real_meta`
  - `sham_meta`
  - `generic_prep`
- Sampling:
  - `temperature=1.0`
  - `top_p=1.0`
  - `top_k=40`
  - `presence_penalty=2.0`
  - `enable_thinking=false`

This is the closest official-ish setting for the existing scaffolded harness, because the harness explicitly disables Qwen's native default thinking mode.

## Original vs Rerun

| Condition | Original Accuracy | Sampling Rerun Accuracy | Delta |
| --- | ---: | ---: | ---: |
| `think_only` | `62.50%` | `67.71%` | `+5.21` |
| `real_meta` | `73.96%` | `47.92%` | `-26.04` |
| `sham_meta` | `50.00%` | `52.08%` | `+2.08` |
| `generic_prep` | `54.17%` | `42.71%` | `-11.46` |

## Other Deltas

| Metric | Original | Sampling Rerun |
| --- | ---: | ---: |
| `real_meta` unique fixes vs both controls | `9` | `4` |
| `real_meta` direct wins over `sham_meta` | `30` | `22` |
| `real_meta` direct wins over `generic_prep` | `30` | `26` |

Format and verbosity also changed:

- `think_only` remained clean and improved.
- `real_meta` got longer and slightly less stable.
- `generic_prep` lost the most format compliance.

## Interpretation

The original semantic `real_meta` lift is **not robust** to switching from greedy decode to Qwen-style non-thinking reasoning sampling within this scaffolded harness.

The most important effect is:

- baseline `think_only` improves,
- while `real_meta` collapses.

That suggests the earlier gain was at least partly coupled to the deterministic decoding regime. In this setting, stochastic reasoning appears to help the base `/think` scaffold more than the explicit control-plane scaffold.

## What This Does Not Mean

- It does **not** prove `/meta-think` is useless.
- It does **not** invalidate the earlier greedy result.
- It **does** mean the effect is not currently robust enough to treat as a general mechanism.

## Practical Read

At this point the honest claim is:

- `real_meta` can help under greedy decoding on the controlled `ages` slice.
- Under official-ish non-thinking reasoning sampling, that advantage disappears and reverses.
- So the current effect is fragile and decode-regime dependent.
