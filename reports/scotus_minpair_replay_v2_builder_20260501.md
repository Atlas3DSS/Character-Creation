# SCOTUS Minimal-Pair Replay v2 Builder

## Purpose

Build a more diverse Commerce Clause replay bank after the first minimal-pair bank was demoted for exact assistant-template reuse.

## Artifacts

| Artifact | Path |
| --- | --- |
| Examples | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/replay/scotus_minpair_replay_v2_examples_20260501.jsonl |
| Manifest | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/replay/scotus_minpair_replay_v2_manifest_20260501.json |

## Counts

- Rows: `288`
- Fact patterns: `24`
- Style variants per fact: `6`
- Exact duplicate assistant texts: `0`
- Unpaired prompt rows: `0`

| Split | Label | Rows |
| --- | --- | --- |
| dev | commerce_authority | 30 |
| dev | commerce_limits | 30 |
| test | commerce_authority | 30 |
| test | commerce_limits | 30 |
| train | commerce_authority | 84 |
| train | commerce_limits | 84 |

## Variant Counts

| Variant | Rows |
| --- | --- |
| counterargument_first | 48 |
| doctrinal_synthesis | 48 |
| holding_then_reason | 48 |
| rule_application | 48 |
| short_opinion | 48 |
| two_step | 48 |

## Read

- Each fact/style prompt has one Commerce-authority and one Commerce-limits assistant answer, so prompt-only label leakage should remain near chance.
- Exact assistant completions are unique across rows.
- Style variants are mirrored across labels to reduce format-label leakage.
- This is still synthetic replay data, not steering evidence. It is a cleaner candidate source for the next activation probe and template-holdout audit.

## Cheap Leakage Baselines

| Feature source | Train BA | Dev BA | Test BA |
| --- | ---: | ---: | ---: |
| Prompt TF-IDF | 0.500 | 0.500 | 0.500 |
| Variant ID only | 0.500 | 0.500 | 0.500 |
| Fact ID only | 0.500 | 0.500 | 0.500 |

These checks lower concern about prompt, style-shell, or fact-id label leakage. They do not address assistant-answer leakage, because the paired assistant texts intentionally encode opposite Commerce propositions.
