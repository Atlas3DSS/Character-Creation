# SCOTUS Baseline Text Classifiers

Case-held-out matched-pair baselines. Each pair contributes one sample per justice.

## Decision

| Pair | Decision | Best masked test model | Best masked test balanced accuracy |
| --- | --- | --- | --- |
| Scalia_vs_Ginsburg | activation-ready | word_char_tfidf_logreg | 0.764 |
| Thomas_vs_Souter | activation-ready | word_tfidf_logreg | 0.809 |

Conservative threshold: do not treat a pair as activation-ready unless masked, case-held-out text separation is at least 0.75 balanced accuracy.

## Test Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 148 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 148 | 0.736 | 0.736 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 148 | 0.757 | 0.757 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 148 | 0.764 | 0.764 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 148 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 148 | 0.574 | 0.574 |
| Scalia_vs_Ginsburg/raw_clean | majority | 148 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 148 | 0.743 | 0.743 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 148 | 0.736 | 0.736 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 148 | 0.750 | 0.750 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 148 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 148 | 0.655 | 0.655 |
| Thomas_vs_Souter/masked | majority | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 256 | 0.809 | 0.809 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 256 | 0.773 | 0.773 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 256 | 0.793 | 0.793 |
| Thomas_vs_Souter/masked | metadata_logreg | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 256 | 0.449 | 0.449 |
| Thomas_vs_Souter/raw_clean | majority | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 256 | 0.785 | 0.785 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 256 | 0.762 | 0.762 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 256 | 0.785 | 0.785 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 256 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 256 | 0.387 | 0.387 |

## Dev Metrics

| Pair/Variant | Model | N | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg/masked | majority | 244 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | word_tfidf_logreg | 244 | 0.742 | 0.742 |
| Scalia_vs_Ginsburg/masked | char_tfidf_logreg | 244 | 0.770 | 0.770 |
| Scalia_vs_Ginsburg/masked | word_char_tfidf_logreg | 244 | 0.766 | 0.766 |
| Scalia_vs_Ginsburg/masked | metadata_logreg | 244 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/masked | length_citation_logreg | 244 | 0.545 | 0.545 |
| Scalia_vs_Ginsburg/raw_clean | majority | 244 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | word_tfidf_logreg | 244 | 0.750 | 0.750 |
| Scalia_vs_Ginsburg/raw_clean | char_tfidf_logreg | 244 | 0.770 | 0.770 |
| Scalia_vs_Ginsburg/raw_clean | word_char_tfidf_logreg | 244 | 0.783 | 0.783 |
| Scalia_vs_Ginsburg/raw_clean | metadata_logreg | 244 | 0.500 | 0.500 |
| Scalia_vs_Ginsburg/raw_clean | length_citation_logreg | 244 | 0.533 | 0.533 |
| Thomas_vs_Souter/masked | majority | 184 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | word_tfidf_logreg | 184 | 0.685 | 0.685 |
| Thomas_vs_Souter/masked | char_tfidf_logreg | 184 | 0.707 | 0.707 |
| Thomas_vs_Souter/masked | word_char_tfidf_logreg | 184 | 0.723 | 0.723 |
| Thomas_vs_Souter/masked | metadata_logreg | 184 | 0.500 | 0.500 |
| Thomas_vs_Souter/masked | length_citation_logreg | 184 | 0.473 | 0.473 |
| Thomas_vs_Souter/raw_clean | majority | 184 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | word_tfidf_logreg | 184 | 0.707 | 0.707 |
| Thomas_vs_Souter/raw_clean | char_tfidf_logreg | 184 | 0.717 | 0.717 |
| Thomas_vs_Souter/raw_clean | word_char_tfidf_logreg | 184 | 0.723 | 0.723 |
| Thomas_vs_Souter/raw_clean | metadata_logreg | 184 | 0.500 | 0.500 |
| Thomas_vs_Souter/raw_clean | length_citation_logreg | 184 | 0.489 | 0.489 |

## Interpretation Notes

- `metadata_logreg` uses only matched metadata fields. High scores here indicate residual confounding.
- `length_citation_logreg` tests whether chunk length or citation density alone separates justices.
- `word_tfidf_logreg` on `masked` text is the main Phase 3 leakage check before activation work.
- Character n-grams are included as a stronger stylometric leakage check.
