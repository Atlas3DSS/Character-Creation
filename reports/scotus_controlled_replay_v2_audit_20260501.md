# SCOTUS Controlled Replay v2 Audit

## Purpose

Build a cleaner no-persona replay source after the Commerce replay family failed causal promotion. Each fact prompt is paired across both legal frames, so prompt-only text should not identify the label.

This is not steering evidence. It is a candidate source for a later activation probe and causal gate.

## Configuration

| Field | Value |
| --- | --- |
| Created | 2026-05-01T17:29:38-07:00 |
| Dataset | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_controlled_replay_v2_examples_20260501.jsonl |
| Rows | 288 |
| Tasks | article3_private_vs_public |
| Variants per fact | 6 |
| Gate | activation_candidate |

## Counts

| Task | Split | Label | Rows |
| --- | --- | --- | --- |
| article3_private_vs_public | dev | 0 | 24 |
| article3_private_vs_public | dev | 1 | 24 |
| article3_private_vs_public | test | 0 | 24 |
| article3_private_vs_public | test | 1 | 24 |
| article3_private_vs_public | train | 0 | 96 |
| article3_private_vs_public | train | 1 | 96 |

## Leakage Baselines

| Field | Dev N | Dev BA | Test N | Test BA | Test F1 |
| --- | --- | --- | --- | --- | --- |
| assistant_cue_masked | 48 | 1.000 | 48 | 1.000 | 1.000 |
| assistant_length | 48 | 0.583 | 48 | 0.583 | 0.545 |
| assistant_text | 48 | 1.000 | 48 | 1.000 | 1.000 |
| prompt | 48 | 0.500 | 48 | 0.500 | 0.000 |
| prompt_cue_masked | 48 | 0.500 | 48 | 0.500 | 0.000 |
| surface_style_id | 48 | 0.500 | 48 | 0.500 | 0.000 |
| text | 48 | 1.000 | 48 | 1.000 | 1.000 |
| text_cue_masked | 48 | 1.000 | 48 | 1.000 | 1.000 |

## Read

- Prompt-only test BA is `0.500`; this should remain near chance because prompts are paired across labels.
- Cue-masked assistant test BA is `1.000`; this is expected to stay high when the answer still states the legal proposition.
- Length-only test BA is `0.583` and surface-style test BA is `0.500`.
- Fact split conflicts: `0`.
- If promoted to activation capture, use assistant-internal regions and treat answer-text separability as expected answer-state evidence, not proof of a circuit.
- The later causal gate must use neutral no-persona prompts and prompt-matched random/source controls.

## Sample Rows

| Example | Split | Label | Style | Prompt | Assistant |
| --- | --- | --- | --- | --- | --- |
| article3_private_vs_public\|00\|v00\|article3_public_rights | train | article3_public_rights | 0 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Conclusion: Congress may assign this determination to the statutory forum.  Reasoning: The matter arises from ... |
| article3_private_vs_public\|00\|v00\|article3_private_rights | train | article3_private_rights | 0 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Conclusion: Final adjudication must remain in an Article III court.  Reasoning: The claim fixes liability of t... |
| article3_private_vs_public\|00\|v01\|article3_public_rights | train | article3_public_rights | 1 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Disposition: The non-Article III process is constitutionally permissible.  Analysis: Congress may assign this ... |
| article3_private_vs_public\|00\|v01\|article3_private_rights | train | article3_private_rights | 1 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Disposition: The assignment is unconstitutional as applied to this dispute.  Analysis: Congress may create pro... |
| article3_private_vs_public\|00\|v02\|article3_public_rights | train | article3_public_rights | 2 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Holding: The objection to specialized adjudication should be rejected.  Rationale: The dispute is between the ... |
| article3_private_vs_public\|00\|v02\|article3_private_rights | train | article3_private_rights | 2 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Holding: The objection to non-Article III final judgment should be sustained.  Rationale: The structural guara... |
| article3_private_vs_public\|00\|v03\|article3_public_rights | train | article3_public_rights | 3 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Result: The matter may proceed in the assigned tribunal subject to review.  Explanation: The adjudicator is ap... |
| article3_private_vs_public\|00\|v03\|article3_private_rights | train | article3_private_rights | 3 | Fact pattern: a federal benefits program assigns disputed eligibility determinat... | Result: The matter cannot be conclusively resolved by the assigned tribunal.  Explanation: Calling the proceed... |
