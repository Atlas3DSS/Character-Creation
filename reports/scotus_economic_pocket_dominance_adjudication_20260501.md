# SCOTUS Economic Pocket Dominance Adjudication

## Purpose

This is an internal dominance review of the clean Economic Activity source excerpts selected for the two surviving causal prompt pockets. It is not an independent blind human review, but it is stricter than the original regex source labels.

## Inputs

| Field | Value |
| --- | --- |
| Queue | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_economic_pocket_dominance_review_20260501.jsonl |
| Adjudicated rows | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_economic_pocket_dominance_adjudicated_20260501.jsonl |
| Rows | 51 |

## Label Counts

| Dominant label | N |
| --- | --- |
| dominant_broad_commerce | 28 |
| dominant_commerce_limits | 21 |
| dominant_state_federalism | 1 |
| reject_noise_or_boilerplate | 1 |

## Source Rule Versus Review

| Source rule frame | Reviewed label | N |
| --- | --- | --- |
| economic_commerce_broad_aggregation | dominant_broad_commerce | 25 |
| economic_commerce_broad_aggregation | dominant_commerce_limits | 1 |
| economic_commerce_broad_aggregation | dominant_state_federalism | 1 |
| economic_commerce_broad_aggregation | reject_noise_or_boilerplate | 1 |
| economic_commerce_limits | dominant_broad_commerce | 3 |
| economic_commerce_limits | dominant_commerce_limits | 20 |

## Usable Binary Counts

| Reviewed label | N |
| --- | --- |
| dominant_broad_commerce | 28 |
| dominant_commerce_limits | 21 |

## Usable Case Coverage

| Reviewed label | Case id | Case | N |
| --- | --- | --- | --- |
| dominant_broad_commerce | champion_1903 | Champion v. Ames | 3 |
| dominant_broad_commerce | heart_atlanta_1964 | Heart of Atlanta Motel, Inc. v. United States | 4 |
| dominant_broad_commerce | hodel_1981 | Hodel v. Virginia Surface Mining & Reclamation Assn., Inc. | 6 |
| dominant_broad_commerce | katzenbach_mcclung_1964 | Katzenbach v. McClung | 1 |
| dominant_broad_commerce | lopez_1995 | United States v. Lopez | 1 |
| dominant_broad_commerce | morrison_2000 | United States v. Morrison | 1 |
| dominant_broad_commerce | nfib_2012 | National Federation of Independent Business v. Sebelius | 1 |
| dominant_broad_commerce | perez_1971 | Perez v. United States | 4 |
| dominant_broad_commerce | raich_2005 | Gonzales v. Raich | 2 |
| dominant_broad_commerce | stafford_1922 | Stafford v. Wallace | 1 |
| dominant_broad_commerce | wickard_1942 | Wickard v. Filburn | 4 |
| dominant_commerce_limits | carter_coal_1936 | Carter v. Carter Coal Co. | 4 |
| dominant_commerce_limits | lopez_1995 | United States v. Lopez | 2 |
| dominant_commerce_limits | morrison_2000 | United States v. Morrison | 8 |
| dominant_commerce_limits | nfib_2012 | National Federation of Independent Business v. Sebelius | 4 |
| dominant_commerce_limits | raich_2005 | Gonzales v. Raich | 1 |
| dominant_commerce_limits | schechter_1935 | A. L. A. Schechter Poultry Corp. v. United States | 2 |

## Gate

Data gate status: `pass`.

The gate requires at least 20 usable broad-Commerce rows, at least 20 usable Commerce-limits rows, and at least 3 source cases per side. Passing this gate permits a cached reviewed-label probe before any new BF16 activation capture.

## Cached Probe Result

The reviewed-label cached probe has run:

- Report: `sweep_v4/scotus_economic_reviewed_broad_limits_cached_20260501/report.md`
- Rows: `49`
- Best readout: `prompt_mean @ L12`
- Dev BA: `0.625`
- Test BA: `0.473`
- Cue-masked text test BA: `0.857`

Decision: the reviewed labels clear the data gate but fail the activation gate. Do not run a causal pocket pilot from this source direction.
