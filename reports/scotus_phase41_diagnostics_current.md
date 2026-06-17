# SCOTUS Phase 4.1 Diagnostics

Phase 4.1 checks whether the Scalia/Ginsburg activation signal is robust enough to promote a direction to a small Phase 5 causal pilot.

## Run Summary

| Mode | Template | Best region | Layer | C | Dev BA | Test BA | Test BA CI | Prompt TF-IDF Test BA | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excerpt_removed | normal | prompt_mean | 20 | 0.001 | 0.500 | 0.500 | 0.500-0.500 | 0.500 | sweep_v4/scotus_phase41_excerpt_removed_20260425_113923 |
| excerpt_removed | normal | prompt_mean | 4 | 0.25 | 0.500 | 0.500 | 0.500-0.500 | 0.500 | sweep_v4/scotus_phase41_smoke_excerpt_removed_20260425_102133 |
| label_shuffle | normal | prompt_last | 32 | 0.1 | 0.584 | 0.435 | 0.350-0.514 | 0.536 | sweep_v4/scotus_phase41_label_shuffle_20260430_201857 |
| neutral_filler | normal | excerpt_mean | 9 | 0.03 | 0.584 | 0.587 | 0.506-0.670 | 0.558 | sweep_v4/scotus_phase41_neutral_filler_20260425_115752 |
| normal | normal | prompt_last | 9 | 0.003 | 0.819 | 0.841 | 0.777-0.899 | 0.754 | sweep_v4/scotus_phase41_normal_20260425_102519 |
| plain_prompt | plain | prompt_last | 10 | 0.03 | 0.836 | 0.797 | 0.727-0.862 | 0.754 | sweep_v4/scotus_phase41_plain_prompt_20260430_214050 |
| template_variant | variant_a | prompt_last | 4 | 0.25 | 0.832 | 0.783 | 0.714-0.849 | 0.761 | sweep_v4/scotus_phase41_template_variant_20260430_205127 |

## Candidate Classification

| Direction | Class | Reason |
| --- | --- | --- |
| prompt_last @ L9 | diagnostic_only | Best decoder, but prompt_last is leakage-sensitive unless strict prompt-ablation diagnostics pass. |
| prompt_mean @ L16, C=0.003 | candidate_direction | Exact non-prompt config clears dev/test >= 0.75 in normal, template_variant, and plain_prompt; worst dev BA 0.761, worst test BA 0.848. |

## Robust Non-Prompt Candidates

Rows here are exact `(region, layer, C)` configurations that clear dev and diagnostic-test balanced accuracy `>= 0.75` in all three real prompt modes. Mode cells are `dev/test`.

| Direction | C | Worst dev BA | Worst test BA | Normal | Template variant | Plain prompt |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_mean @ L16 | 0.003 | 0.761 | 0.848 | 0.761/0.848 | 0.761/0.855 | 0.761/0.848 |
| excerpt_mean @ L16 | 0.003 | 0.761 | 0.841 | 0.761/0.848 | 0.761/0.841 | 0.761/0.841 |
| prompt_mean @ L17 | 0.03 | 0.765 | 0.833 | 0.782/0.833 | 0.765/0.833 | 0.786/0.833 |
| excerpt_mean @ L17 | 0.003 | 0.756 | 0.833 | 0.765/0.841 | 0.756/0.841 | 0.761/0.833 |
| prompt_mean @ L17 | 0.01 | 0.769 | 0.826 | 0.777/0.848 | 0.769/0.833 | 0.773/0.826 |
| excerpt_mean @ L17 | 0.03 | 0.769 | 0.826 | 0.773/0.833 | 0.769/0.826 | 0.782/0.833 |
| prompt_mean @ L17 | 0.1 | 0.765 | 0.826 | 0.769/0.833 | 0.769/0.826 | 0.765/0.841 |
| excerpt_mean @ L16 | 0.01 | 0.752 | 0.826 | 0.773/0.826 | 0.752/0.841 | 0.752/0.826 |
| prompt_mean @ L17 | 0.003 | 0.756 | 0.819 | 0.765/0.833 | 0.756/0.841 | 0.761/0.819 |
| excerpt_mean @ L17 | 0.01 | 0.752 | 0.819 | 0.765/0.833 | 0.752/0.819 | 0.782/0.833 |

## Gate Decision

| Criterion | Status |
| --- | --- |
| Null tests at chance | pass |
| Prompt/template robustness | pass |
| Robust non-prompt candidate clears >=0.75 | pass |
| Phase 5 causal pilot gate | pass |

Gate note: a pass here authorizes only a small causal pilot with random and wrong-pair controls. It is not evidence that a steerable judicial circuit has been found.
