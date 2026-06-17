# Meta-Think vs Think-Only and Qwen3.6 MoE Routing Map

Generated: 2026-04-16

## Dev-server phase isolation

Artifact root: `sweep_v4/meta_vs_think_phase_isolation_qwen35_20260416_full48`

Setup:
- Model: `Qwen/Qwen3.5-9B`
- Endpoints: dev-server 3090/4090 SGLang
- Base prompts: 48 reasoning rows from `personality_meta_eval_trace_explicit_v1`
- Conditions: `think_only` vs `meta_think_plus_think`
- Total records: 96
- Native SGLang thinking disabled via `chat_template_kwargs={"enable_thinking": false}`

Results:

| Condition | Accuracy | Format | Leaks | Trunc | Mean completion tokens |
|---|---:|---:|---:|---:|---:|
| think_only | 100.0% | 100.0% | 0.0% | 0.0% | 188.4 |
| meta_think_plus_think | 91.7% | 100.0% | 0.0% | 0.0% | 211.7 |

Task breakdown:

| Condition | Heavyball | Printers | Sequence | Tickets |
|---|---:|---:|---:|---:|
| think_only | 12/12 | 12/12 | 12/12 | 12/12 |
| meta_think_plus_think | 11/12 | 10/12 | 12/12 | 11/12 |

Paired pattern:
- `think_only=True`, `meta=True`: 44
- `think_only=True`, `meta=False`: 4
- `think_only=False`, `meta=True`: 0

Interpretation:
- On this isolated comparison, the marginal single `/meta-think` block did not improve reasoning over `/think` alone.
- It slightly hurt: 4 regressions, no fixes.
- This strongly suggests the Experiment C improvement over response-only was mainly caused by adding explicit scratchpad reasoning (`/think`), not by single-block meta-control.
- This does not fully rule out the C budget-2 result, because C's best condition used two explicit meta blocks. But the canonical trace-style single `/meta-think` is not showing independent benefit here.

## Qwen3.6 MoE routing map

Artifact root: `sweep_v4/qwen36_moe_routing_map_v1`

Setup:
- Model: `/home/orwel/dev_genius/models/Qwen3.6-35B-A3B`
- Server: workstation Blackwell SGLang with `--enable-return-routed-experts`
- Conditions: `response_only`, `think_only`, `meta_think_plus_think`
- Records: 144 = 48 prompts x 3 conditions
- Routing signal: SGLang top-k routed expert IDs, shape `tokens x 40 layers x 8 experts/token`
- Important limitation: this is top-k ID frequency, not full router probability/logit KL.

Pairwise phase routing shifts:

| Pair | Mean JSD | Max JSD | Max JSD Layer | Mean TV | Max TV | Max TV Layer |
|---|---:|---:|---:|---:|---:|---:|
| meta_think_plus_think vs response_only | 0.00736 | 0.00958 | 6 | 0.0836 | 0.0985 | 17 |
| meta_think_plus_think vs think_only | 0.00311 | 0.00460 | 29 | 0.0514 | 0.0605 | 29 |
| response_only vs think_only | 0.00308 | 0.00455 | 18 | 0.0482 | 0.0566 | 29 |

Initial read:
- Routing changes across phases are small but measurable.
- `meta_think_plus_think` is closer to `think_only` than to `response_only`, as expected.
- The literal scaffold changes routing a bit, but not dramatically. This matches the earlier MoE-router finding: personality/control differences mostly do not look like gross expert-selection changes.
- The current map is best treated as a routing-vibe scan, not final router-probability evidence.

Trait-router shifts:
- Largest trait max JSD values are around 0.006-0.013 depending on condition and trait.
- Openness shows the largest top-k expert-frequency separation in this sample, especially around layer 8.
- These are still small relative to what we would expect if personality were mainly expert routing.

Practical implication:
- For Qwen3.6, keep looking primarily at residual/hidden-state geometry for personality and phase control.
- Expert routing may reflect prompt/scaffold format, but it is not obviously the main carrier of personality in this quick scan.
