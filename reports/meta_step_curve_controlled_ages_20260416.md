# Controlled Meta-Step Curve on Qwen 3.5 9B (`ages` slice)

Date: 2026-04-16
Output root: `sweep_v4/meta_step_curve_controlled_qwen35_20260416`

## Why this run exists

The earlier `Experiment C` curve was encouraging, but it was sampled from the first 48 reasoning rows in shard order. That slice mostly covered the easier reasoning families (`tickets`, `heavyball`, `printers`, `sequence`), so it could not cleanly answer whether iterative `/meta-think` really helps on moderately difficult prompts.

This run fixes that by:

1. screening the full 768-row reasoning pool with `think_only`,
2. selecting only prompt families whose `think_only` accuracy falls in the 50-80% band,
3. rerunning the selected rows with budgets `0/1/2/3`, where:
   - `0` = `/think` only,
   - `1` = one `/meta-think` block + `/think`,
   - `2` = two `/meta-think` blocks + `/think`,
   - `3` = three `/meta-think` blocks + `/think`.

## Screen result

Full-family `think_only` screen over 96 personas per family:

| Prompt family | Accuracy |
| --- | ---: |
| `ages` | `62.5%` |
| `batball` | `84.4%` |
| `chickens_cows` | `82.3%` |
| `printers` | `87.5%` |
| `heavyball` | `96.9%` |
| `sequence` | `96.9%` |
| `tickets` | `96.9%` |
| `syllogism` | `100%` |

Only `ages` landed inside the requested 50-80% band, so the controlled subset is the full 96-row `ages` family.

## Controlled curve

| Budget | Condition | Accuracy | Mean Completion Tokens | Mean Latency |
| --- | --- | ---: | ---: | ---: |
| `0` | `/think` only | `62.5%` | `173.7` | `51.7s` |
| `1` | `1x /meta-think + /think` | `81.25%` | `258.1` | `75.2s` |
| `2` | `2x /meta-think + /think` | `76.04%` | `349.3` | `103.5s` |
| `3` | `3x /meta-think + /think` | `75.0%` | `421.2` | `132.8s` |

## Paired comparison vs budget 0

### Budget 1 vs budget 0
- both correct: `55`
- marginal fixes: `23`
- both wrong: `13`
- marginal regressions: `5`

### Budget 2 vs budget 0
- both correct: `52`
- marginal fixes: `21`
- both wrong: `15`
- marginal regressions: `8`

### Budget 3 vs budget 0
- both correct: `51`
- marginal fixes: `21`
- both wrong: `15`
- marginal regressions: `9`

## Interpretation

The controlled result is now clear:

- A single explicit `/meta-think` pass is genuinely useful on this moderately difficult reasoning slice.
- The gain is not small: `62.5% -> 81.25%`.
- More meta passes do **not** continue improving performance. Budgets `2` and `3` remain above baseline, but both are worse than budget `1`.

So the curve is not “flat/negative across all step counts,” and it is also not a clean monotone refinement curve. The best description is:

- `one structured control pass helps a lot`,
- `additional control passes add overhead and some overconstraint`,
- `the optimum on this slice is budget 1`.

## Mechanistic take

This still does **not** imply a new module or a basin switch.

The evidence is more consistent with prompt-conditioned biasing on a shared substrate:

- earlier probe-transfer was very high between `/meta-think` and `/think`,
- the Qwen 3.6 MoE routing map only showed small routing shifts,
- this curve is sensitive to the **structure and amount** of control scaffold, not just raw extra tokens.

The strongest next causal control would be a matched-length sham scaffold:

1. `/think` only,
2. real `/meta-think + /think`,
3. same-length non-semantic scaffold + `/think`,
4. same-length generic extra scratchpad + `/think`.

If only the real `/meta-think` condition keeps the `budget 1` lift, then the gain is not just “more scratchpad.”
