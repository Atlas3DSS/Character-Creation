# SCOTUS Baseline Text Classifiers

Case-held-out matched-pair baselines. Each pair contributes one sample per justice.

## Decision

| Pair | Decision | Best masked test model | Best masked test balanced accuracy |
| --- | --- | --- | --- |
| Scalia_vs_Ginsburg | weak/exploratory only | word_tfidf_logreg | 0.637 |
| Thomas_vs_Souter | no-go for activation | char_tfidf_logreg | 0.539 |

Conservative threshold: do not treat a pair as activation-ready unless masked, case-held-out text separation is at least 0.75 balanced accuracy.

## Test Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 542 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 542 | 0.637 | 0.637 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 542 | 0.592 | 0.592 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 542 | 0.613 | 0.613 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 542 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 542 | 0.450 | 0.450 |
| Scalia_vs_Ginsburg/raw_clean | majority | 542 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 542 | 0.637 | 0.637 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 542 | 0.574 | 0.574 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 542 | 0.605 | 0.605 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 542 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 542 | 0.530 | 0.530 |
| Thomas_vs_Souter/masked | majority | 614 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 614 | 0.524 | 0.524 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 614 | 0.539 | 0.539 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 614 | 0.531 | 0.531 |
| Thomas_vs_Souter/masked | metadata_logreg | 614 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 614 | 0.419 | 0.419 |
| Thomas_vs_Souter/raw_clean | majority | 614 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 614 | 0.507 | 0.507 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 614 | 0.546 | 0.546 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 614 | 0.528 | 0.528 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 614 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 614 | 0.425 | 0.425 |

## Dev Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 596 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 596 | 0.601 | 0.601 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 596 | 0.601 | 0.601 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 596 | 0.599 | 0.599 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 596 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 596 | 0.465 | 0.465 |
| Scalia_vs_Ginsburg/raw_clean | majority | 596 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 596 | 0.606 | 0.606 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 596 | 0.592 | 0.592 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 596 | 0.604 | 0.604 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 596 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 596 | 0.502 | 0.502 |
| Thomas_vs_Souter/masked | majority | 420 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 420 | 0.614 | 0.614 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 420 | 0.662 | 0.662 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 420 | 0.636 | 0.636 |
| Thomas_vs_Souter/masked | metadata_logreg | 420 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 420 | 0.555 | 0.555 |
| Thomas_vs_Souter/raw_clean | majority | 420 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 420 | 0.610 | 0.610 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 420 | 0.657 | 0.657 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 420 | 0.631 | 0.631 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 420 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 420 | 0.552 | 0.552 |

## Interpretation Notes

- `metadata_logreg` uses only matched metadata fields. High scores here indicate residual confounding.
- `length_citation_logreg` tests whether chunk length or citation density alone separates justices.
- `word_tfidf_logreg` on `masked` text is the main Phase 3 leakage check before activation work.
- Character n-grams are included as a stronger stylometric leakage check.
