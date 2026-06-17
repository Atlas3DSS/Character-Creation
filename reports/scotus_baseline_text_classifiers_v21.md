# SCOTUS Baseline Text Classifiers

Case-held-out matched-pair baselines. Each pair contributes one sample per justice.

## Decision

| Pair | Decision | Best masked test model | Best masked test balanced accuracy |
| --- | --- | --- | --- |
| Scalia_vs_Ginsburg | activation-ready | word_char_tfidf_logreg | 0.775 |
| Thomas_vs_Souter | activation-ready | word_char_tfidf_logreg | 0.789 |

Conservative threshold: do not treat a pair as activation-ready unless masked, case-held-out text separation is at least 0.75 balanced accuracy.

## Test Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 138 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 138 | 0.717 | 0.717 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 138 | 0.754 | 0.754 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 138 | 0.775 | 0.775 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 138 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 138 | 0.594 | 0.594 |
| Scalia_vs_Ginsburg/raw_clean | majority | 138 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 138 | 0.717 | 0.717 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 138 | 0.732 | 0.732 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 138 | 0.768 | 0.768 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 138 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 138 | 0.667 | 0.667 |
| Thomas_vs_Souter/masked | majority | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 256 | 0.781 | 0.781 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 256 | 0.785 | 0.785 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 256 | 0.789 | 0.789 |
| Thomas_vs_Souter/masked | metadata_logreg | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 256 | 0.449 | 0.449 |
| Thomas_vs_Souter/raw_clean | majority | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 256 | 0.773 | 0.773 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 256 | 0.781 | 0.781 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 256 | 0.789 | 0.789 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 256 | 0.402 | 0.402 |

## Dev Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 238 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 238 | 0.744 | 0.744 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 238 | 0.765 | 0.765 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 238 | 0.761 | 0.761 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 238 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 238 | 0.550 | 0.550 |
| Scalia_vs_Ginsburg/raw_clean | majority | 238 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 238 | 0.748 | 0.748 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 238 | 0.769 | 0.769 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 238 | 0.773 | 0.773 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 238 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 238 | 0.529 | 0.529 |
| Thomas_vs_Souter/masked | majority | 148 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 148 | 0.716 | 0.716 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 148 | 0.696 | 0.696 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 148 | 0.723 | 0.723 |
| Thomas_vs_Souter/masked | metadata_logreg | 148 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 148 | 0.426 | 0.426 |
| Thomas_vs_Souter/raw_clean | majority | 148 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 148 | 0.730 | 0.730 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 148 | 0.689 | 0.689 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 148 | 0.743 | 0.743 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 148 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 148 | 0.439 | 0.439 |

## Interpretation Notes

- `metadata_logreg` uses only matched metadata fields. High scores here indicate residual confounding.
- `length_citation_logreg` tests whether chunk length or citation density alone separates justices.
- `word_tfidf_logreg` on `masked` text is the main Phase 3 leakage check before activation work.
- Character n-grams are included as a stronger stylometric leakage check.
