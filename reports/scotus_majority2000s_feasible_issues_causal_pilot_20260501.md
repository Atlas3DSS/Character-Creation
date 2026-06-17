# SCOTUS Majority-2000s Feasible-Issues Causal Pilot

## Purpose

This tests whether the refined `Scalia_vs_Ginsburg` majority-2000s feasible-issues probe directions causally move Qwen3.5 BF16 legal generations beyond prompt-matched same-layer random controls.

The upstream decoding evidence remains real but correlational:

- Detailed audit: `reports/scotus_slice_majority2000s_feasible_issues_20260501.md`
- Normal/template/plain median test BA: `0.746` / `0.758` / `0.764`
- Label-shuffle/excerpt-removed/neutral-filler median test BA: `0.541` / `0.500` / `0.542`

## Method

Both pilots used:

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Prompt bank: `data/scotus/scotus_poke_prompts_v1.jsonl`
- Prompts: `0-11`, covering `Judicial Power`, `Criminal Procedure`, and `Economic Activity`
- Max new tokens: `96`
- Greedy decoding
- Hidden-norm-fraction alpha scaling using `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_20260501_033918`
- Same-layer random unit-vector controls
- Prompt-matched random comparison as the primary readout

The frame metric is still a coarse keyword diagnostic. It is sufficient for a fast causal gate, not for a final jurisprudential claim.

## Results

### Primary `prompt_last @ L10`

- Run: `sweep_v4/scotus_sae_poke_20260501_045156/report.md`
- Direction file: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/split_00/best_probe_direction.npz`
- Position: `last`
- Alphas: `0.02,0.05,0.1`
- Random controls: `10`
- Rows: `408`

| Alpha | Matched target delta | Target z | Target win rate | Matched net delta | Net z | Net win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.02` | `0.142` | `0.184` | `0.250` | `0.417` | `0.533` | `0.333` |
| `0.05` | `0.117` | `0.122` | `0.250` | `0.025` | `0.024` | `0.250` |
| `0.10` | `0.033` | `0.030` | `0.333` | `0.175` | `0.131` | `0.417` |

Read: no causal promotion. The best net row is only `z=0.533`, and prompt win rates remain weak.

### Secondary `excerpt_mean @ L16`

- Run: `sweep_v4/scotus_sae_poke_20260501_060425/report.md`
- Direction file: `sweep_v4/scotus_slice_bf16_majority2000s_feasible_issues_normal_component_resplits_20260501_034116/split_01/best_probe_direction.npz`
- Position: `all`
- Alphas: `0.01,0.02,0.05`
- Random controls: `5`
- Rows: `228`

| Alpha | Matched target delta | Target z | Target win rate | Matched net delta | Net z | Net win rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | `0.217` | `0.263` | `0.167` | `0.283` | `0.322` | `0.417` |
| `0.02` | `-0.300` | `-0.347` | `0.167` | `-0.367` | `-0.421` | `0.167` |
| `0.05` | `0.533` | `0.449` | `0.333` | `0.533` | `0.395` | `0.333` |

Read: no causal promotion. The largest positive row is small, not alpha-monotone, and does not beat random controls strongly.

## Decision

The refined majority-2000s feasible-issues branch remains good evidence for decodable justice-style structure, but the first two causal pilots do not show a steerable judicial circuit.

Eliminated as immediate steering candidates:

- `prompt_last @ L10`, positive Ginsburg direction, last-token hook.
- `excerpt_mean @ L16`, positive Ginsburg direction, all-position hook.

Not eliminated:

- The underlying correlational probe result.
- The possibility that a reverse sign, a different intervention timing, or a narrower prompt/evaluator target could work.

Recommended next step:

Stop broad prompt-bank causal pokes on these directions for now. A prompt-pocket review queue has been created to check whether any narrow family deserves follow-up:

- Prompt-pocket report: `reports/scotus_majority2000s_causal_prompt_pockets_20260501.md`
- Blind queue: `data/scotus/scotus_majority2000s_causal_review_blind_20260501.jsonl`
- Key file: `data/scotus/scotus_majority2000s_causal_review_key_20260501.jsonl`

The queue selected `8` candidate cells and `22` candidate-vs-baseline/random pairwise comparisons. A prompt family should not be promoted unless the candidate side wins against both baseline and matched random controls without coherence degradation.

Internal adjudication found two narrow reviewed pockets:

- `EA03_gun_school_zone` / `Economic Activity` from `split_00_best_probe_direction__last` at alpha `0.02`
- `EA01_commercial_remedy` / `Economic Activity` from `split_01_best_probe_direction__all` at alpha `0.02`

Adjudication report: `reports/scotus_majority2000s_causal_review_adjudication_20260501.md`.

The next useful work is to build a narrower same-doctrine Commerce Clause / federal-remedy contrast before any more broad BF16 hook generation.

Follow-up now exists:

- Clean cached source rescore: `sweep_v4/scotus_economic_clean_broad_limits_cached_20260501/report.md`
- Economic pocket follow-up report: `reports/scotus_economic_pocket_followup_20260501.md`
- Dominance review queue: `data/scotus/scotus_economic_pocket_dominance_review_20260501.jsonl`

The clean cached source rescore did not promote the existing source-frame direction: `51` clean rows, best activation test BA `0.393`, cue-masked text test BA `0.679`. The branch should proceed only through dominance review of the `51` unique broad-Commerce / Commerce-limits source excerpts before any more BF16 activation capture.

The dominance review and reviewed-label cached probe have now run. The reviewed set had enough rows (`28` broad-Commerce, `21` Commerce-limits), but the cached activation probe still failed: best activation test BA `0.473` versus cue-masked text test BA `0.857`. Do not run a causal pocket pilot from the Economic Activity source direction under the current protocol.
