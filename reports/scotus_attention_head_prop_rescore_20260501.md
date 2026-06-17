# SCOTUS Attention-Head Proposition Rescore

## Goal

Rescore the full-attention head trace-patch run with proposition-level legal frame rules, rather than the older raw keyword frame counts.

This tests whether the small apparent head-patch movements were actual legal proposition movement or mostly lexical/frame-count noise.

## Inputs

- Head-patch run: `sweep_v4/scotus_attention_head_patch_20260501_141557`
- Rescore run: `sweep_v4/scotus_head_prop_rescore_20260501_145000`
- Scorer: `scripts/experiments/scotus/rescore_scotus_frame_propositions.py`

The scorer was also repaired for newer component/head runs: when `target_candidate` is present, random controls are matched to that exact candidate rather than pooled across all candidates at the same prompt/alpha.

## Result

The proposition rescore does not promote any attention head.

| Candidate | Blend | Prop target minus random | Prop net minus random | Source target/net | Strongest target win | Strongest net win | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L19_H14 | 0.1 | 0.250 | 0.250 | 0.000 / 0.000 | 0.25 | 0.25 | still weak; not promotable |
| L23_H06 | 0.3 | 0.125 | 0.125 | 0.250 / 0.250 | 0.00 | 0.00 | source control stronger/matched; reject |
| L23_H22 | 0.1 | 0.125 | 0.125 | 0.250 / 0.250 | 0.00 | 0.00 | source control stronger/matched; reject |
| L23_H22 | 0.3 | 0.125 | 0.125 | 0.000 / 0.000 | 0.00 | 0.00 | below strongest random; reject |
| L23_H16 | 0.3 | 0.000 | 0.125 | 0.000 / 0.000 | 0.00 | 0.00 | weak; reject |

Largest keyword-to-proposition corrections in this run:

| Frame | Old rows | Proposition rows | Dropped |
| --- | ---: | ---: | ---: |
| `fourth_home_exigency` | 49 | 0 | 49 |
| `article3_article1_tribunal` | 48 | 0 | 48 |
| `federalism_anti_commandeering` | 49 | 15 | 34 |
| `economic_remedy_damages` | 24 | 0 | 24 |

## Interpretation

The stricter scorer removes the biggest false positives from the head-patch run:

- bare `home` no longer triggers Fourth Amendment home-exigency;
- bare `Article I`/`Article III` mentions inside Commerce reasoning no longer trigger Article III tribunal frames;
- generic remedy/damages wording no longer counts as a distinct economic-remedy proposition.

The only surviving positive row, L19_H14 at blend `0.1`, remains tiny and prompt-local. It improves one of four prompts and does not beat strongest random controls often enough to matter.

## Decision

Do not promote any full-attention head from this branch.

Under both keyword and proposition scoring, the current Commerce minimal-pair replay branch remains a readout/answer-state separator, not a demonstrated steerable judicial circuit.

## Next Work

The next useful work is still to rebuild the replay/eval substrate:

1. create many diverse no-template completions per fact pattern;
2. evaluate proposition movement, not raw frame keywords;
3. require source-control and strongest-random wins before any larger hook sweep;
4. include no-mask reasoning checks where the model exposes `<thinking>`/scratchpad behavior.
