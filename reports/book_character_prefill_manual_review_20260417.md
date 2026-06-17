# Book Character Prefill Manual Review

Date: 2026-04-17

## Scope
Manual review of the `val/test` slice for the balanced book-derived character prefill corpus.

Original reviewed package:
- `sweep_v4/book_character_prefill_dataset_balanced_v3_20260417_150446`

Final reviewed package:
- `sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017`

## What I Checked
- split balance by behavior
- cross-split prompt duplication
- low-quality paired items, especially `q2` fail traces
- whether fail traces were merely shallow or actually too cartoonish / factually broken
- spot checks on high-quality `q4/q5` pairs to verify the issue was localized

## Findings On V3
Structural checks:
- `val`: 20 paired items, perfectly balanced across all 5 behaviors
- `test`: 20 paired items, perfectly balanced across all 5 behaviors
- cross-split prompt collisions: `0`

Problem:
- too many `q2` pairs in `val/test`
- issue was not leakage but weak fail examples in part of the eval slice
- some fails were acceptable shallow negatives, but several were too extreme or factually broken for serious evaluation use

## Explicit Removals
I excluded 6 pre-merge items from the final reviewed package:

1. `Esterad's Dilemma: Wisdom vs. Survival`
- reason: fail side dodged into soup/weather-level triviality

2. `Sheriam's Controlled Amusement`
- reason: fail side became too aggressive/revealing and over-broke the facade target

3. `Egwee: Laman's Sin vs. Political Reality`
- reason: fail side was factually broken (`Matrim Cauthon` / bogus White Tower claim)

4. `Egwene's Weather Gambit`
- reason: fail side collapsed into implausible groveling and over-admission

5. `Geralt: The Weight of Memory`
- reason: fail side ended in a trivial mechanical wrap-up (`The End.`), too low-signal

6. `Sorrento's Calm Facade`
- reason: fail side was too performatively enthusiastic and shallow to be a good eval negative

## Split Policy Change
Manual exclusions alone are not enough. The better fix is split policy:
- train can keep more of the weaker-but-still-usable negatives
- `val/test` should be drawn from stronger paired items

So I changed the packager to support:
- exclusion file for reviewed removals
- quality-aware eval split

This makes `val/test` preferentially use higher-pair-quality items while preserving overall behavior balance.

## Final V6 Result
Package:
- `sweep_v4/book_character_prefill_dataset_balanced_v6_reviewed_20260417_151017`

Counts:
- `200` paired items
- `400` completions
- `40` paired items per behavior
- split: `320 train / 40 val / 40 test`

Eval quality distribution after review:
- `val`: `2` q3, `14` q4, `4` q5, `0` q2
- `test`: `2` q3, `17` q4, `1` q5, `0` q2

So `val/test` is now entirely `q3-q5`, with the bulk at `q4`.

## Spot Check Result
The remaining `q3` items in `val/test` are acceptable.
They are weaker than the best pairs, but they are still coherent negatives rather than nonsense negatives.

## Recommendation
Use `v6_reviewed` as the canonical corpus for:
- mechanistic replay/probing
- controlled behavior scoring
- causal patching comparisons on the eval slice

If we want an even stricter eval slice later, the next move would be:
- make a `v7_eval_strict` package with `val/test` restricted to `q4+` only
- keep `v6_reviewed` as the broader balanced corpus
