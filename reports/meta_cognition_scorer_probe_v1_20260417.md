# Meta-Cognition Scorer + Probe v1

- Generated: 2026-04-17T12:27:22-07:00
- Seed set: `sweep_v4/meta_cognition_seedset_balanced_v1_20260417_115123`
- Scorer corpus: `sweep_v4/meta_cognition_scorer_corpus_v1_20260417_121022`
- Text scorer: `sweep_v4/meta_cognition_text_scorer_v1_20260417_121713`
- Activation probe: `sweep_v4/meta_cognition_activation_probe_v1_20260417_121804`

## Corpus

- 120 labeled examples total
- 60 pass / 60 fail
- 24 examples per behavior
- Judge-match rate: 0.9667

## Text Scorer

- Best variant: `response_plus_behavior`
- Val balanced accuracy: 0.750
- Test balanced accuracy: 0.750

## Activation Probe

- Best region: `assistant_last16`
- Best layer: `L34`
- Best C: `0.25`
- Val balanced accuracy: 0.900
- Test balanced accuracy: 0.850

## Probe vs Text Scorer

- val: overlap=20, agreement=0.750, joint_correct=0.700, prob_corr=0.614
- test: overlap=20, agreement=0.800, joint_correct=0.700, prob_corr=0.756

## Caveats

- Synthetic scorer corpus from the same teacher model can overstate absolute performance.
- The text scorer nearly memorizes train (expected on 80 examples), so test is the meaningful number.
- Some seed `candidate_id`s repeat across behaviors; comparison here uses a behavior-aware key.
