# SCOTUS Economic Activity Split Component Review

## Purpose

This follows the failed prompt-template invariance gate for the `issue_area_label=Economic Activity` Scalia/Ginsburg justice-style slice. The goal is to check whether the high component-resplit rows are likely broad justice style or case/topic structure.

## Inputs

| Artifact | Path |
| --- | --- |
| Normal component resplits | `sweep_v4/scotus_slice_bf16_economic_style_normal_component_resplits_20260501_024704/report.md` |
| Template-variant component resplits | `sweep_v4/scotus_slice_bf16_economic_style_template_variant_component_resplits_20260501_032205/report.md` |
| Plain-prompt component resplits | `sweep_v4/scotus_slice_bf16_economic_style_plain_prompt_component_resplits_20260501_032744/report.md` |

`split_01` and `split_03` were inspected because both have high activation test BA and high text baseline performance:

| Split | Activation test BA | Text test BA |
| --- | ---: | ---: |
| `split_01` | `0.790` | `0.746` |
| `split_03` | `0.856` | `0.839` |

## Case-Level Read

### split_01

| Case | Justice | N | Text acc | Activation acc |
| --- | --- | ---: | ---: | ---: |
| `arcadia-v-ohio-power-co` | Scalia | `12` | `1.00` | `0.92` |
| `owen-v-owen` | Scalia | `13` | `1.00` | `0.77` |
| `john-hancock-mutual-life-insurance-v-harris-trust-savings-bank` | Ginsburg | `25` | `0.00` | `0.72` |
| `great-west-life-annuity-insurance-v-knudson` | Scalia | `16` | `0.94` | `0.62` |
| `eldred-v-ashcroft` | Ginsburg | `29` | `0.90` | `0.93` |
| `norton-v-southern-utah-wilderness-alliance` | Scalia | `28` | `0.96` | `0.79` |
| `tellabs-inc-v-makor-issues-rights-ltd` | Ginsburg | `15` | `0.67` | `0.73` |

### split_03

| Case | Justice | N | Text acc | Activation acc |
| --- | --- | ---: | ---: | ---: |
| `blatchford-v-native-village-of-noatak` | Scalia | `15` | `0.60` | `0.87` |
| `el-al-israel-airlines-ltd-v-tsui-yuan-tseng` | Ginsburg | `15` | `0.87` | `1.00` |
| `great-west-life-annuity-insurance-v-knudson` | Scalia | `16` | `0.88` | `0.56` |
| `eldred-v-ashcroft` | Ginsburg | `29` | `0.90` | `0.90` |
| `norton-v-southern-utah-wilderness-alliance` | Scalia | `28` | `0.96` | `0.89` |
| `tellabs-inc-v-makor-issues-rights-ltd` | Ginsburg | `15` | `0.67` | `0.87` |

## Text Feature Audit

The rendered-prompt TF-IDF baseline was refit on train+dev for each split, matching the saved text baseline protocol. The highest word n-grams are topical/case-specific rather than clean justice-style markers.

| Split | Ginsburg-weighted examples | Scalia-weighted examples |
| --- | --- | --- |
| `split_01` | `rule`, `value`, `bact`, `adec`, `volvo`, `articles`, `maritime`, `reeder`, `databases`, `foreign`, `port`, `california`, `kontrick`, `publishers` | `section`, `fcc`, `the epa`, `arbitration`, `subpart`, `commission`, `common law`, `antitrust`, `contract`, `product` |
| `split_03` | `rule`, `value`, `foreign`, `bact`, `adec`, `articles`, `volvo`, `states`, `reeder`, `maritime`, `hancock`, `creditor`, `publishers` | `section`, `fcc`, `the epa`, `arbitration`, `commission`, `subpart`, `common law`, `antitrust`, `product` |

## Read

This strengthens the rejection of the broad Economic Activity justice-style slice as a steering candidate.

1. The best text splits are easy to explain with case/topic cues.
2. Activation sometimes beats text on cases text gets wrong, especially `john-hancock-mutual-life-insurance-v-harris-trust-savings-bank`, so the activation signal is not simply identical to the text baseline.
3. The selected activation readouts remain mostly `prompt_last`, and the prompt-template component-resplit medians fall below the promotion threshold.

Conclusion: do not run a causal hook pilot from this broad slice. If Economic Activity remains interesting, rebuild it as a narrower source-grounded contrast, for example statutory/ERISA remedial framing, Commerce Clause limits, preemption/arbitration, or antitrust/regulatory deference, then rerun the component-resplit and text-baseline gates.
