# SCOTUS Minimal-Pair Replay v2 Holdout Audit

## Purpose

Test whether the replay-v2 separability survives leave-one-style-variant and leave-one-fact-pattern holdouts, using already-captured activations. This is still decodability evidence, not causal steering evidence.

## Inputs

- Features: `sweep_v4/scotus_minpair_replay_v2_20260501_144942/features.npz`
- Metadata: `sweep_v4/scotus_minpair_replay_v2_20260501_144942/feature_meta.jsonl`

## Results

| Readout | Holdout | Groups | Mean BA | Median BA | Min BA | Max BA | Mean accuracy | Mean F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt_text_tfidf C=1.0 | variant_id | 6 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |
| assistant_text_tfidf C=1.0 | variant_id | 6 | 0.917 | 1.000 | 0.500 | 1.000 | 0.917 | 0.944 |
| prompt_last__L08 C=1.0 | variant_id | 6 | 0.528 | 0.500 | 0.500 | 0.604 | 0.528 | 0.469 |
| assistant_all__L08 C=0.001 | variant_id | 6 | 0.865 | 0.927 | 0.500 | 1.000 | 0.865 | 0.899 |
| assistant_early__L08 C=0.001 | variant_id | 6 | 0.753 | 0.750 | 0.500 | 1.000 | 0.753 | 0.808 |
| assistant_late__L08 C=0.001 | variant_id | 6 | 0.896 | 1.000 | 0.500 | 1.000 | 0.896 | 0.925 |
| assistant_all__L16 C=0.001 | variant_id | 6 | 0.951 | 1.000 | 0.708 | 1.000 | 0.951 | 0.962 |
| assistant_all__L24 C=0.001 | variant_id | 6 | 0.931 | 1.000 | 0.583 | 1.000 | 0.931 | 0.921 |
| prompt_text_tfidf C=1.0 | fact_id | 24 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |
| assistant_text_tfidf C=1.0 | fact_id | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt_last__L08 C=1.0 | fact_id | 24 | 0.531 | 0.500 | 0.417 | 0.667 | 0.531 | 0.485 |
| assistant_all__L08 C=0.001 | fact_id | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| assistant_early__L08 C=0.001 | fact_id | 24 | 0.972 | 1.000 | 0.917 | 1.000 | 0.972 | 0.974 |
| assistant_late__L08 C=0.001 | fact_id | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| assistant_all__L16 C=0.001 | fact_id | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| assistant_all__L24 C=0.001 | fact_id | 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Read

- `prompt_text_tfidf` stays at chance under both group holdouts, as expected from paired prompts.
- `assistant_text_tfidf` is perfect under fact holdout and strong, but not uniform, under variant holdout. The answer text itself carries an easily recoverable Commerce-authority versus Commerce-limits proposition.
- Assistant-internal readouts are perfect under fact holdout and strong under variant holdout, with some style variants falling to chance. That makes replay-v2 useful as an answer-state candidate source, but it should not be promoted as a judicial circuit without a causal generation win against matched random and source-trace controls.
- `prompt_last__L08` remains near chance, which lowers concern about prompt-format leakage but increases the interpretation that the separability appears only after the model is replaying the labeled answer.
