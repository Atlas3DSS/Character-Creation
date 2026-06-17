# SCOTUS Economic Pocket Follow-up

## Purpose

The broad SCOTUS justice-style direction is decodable but has not passed causal promotion. The only surviving prompt pockets were narrow Economic Activity prompts, so this artifact defines the next cleaner test before spending more BF16 hook time.

## What changed

- The original Economic Activity source pack has 50 `expected_frame` versus `frame` mismatches and 120 multi-frame conflicts across 288 rows.
- The stricter cached broad-versus-limits rescore kept 51 clean rows but failed: best activation test BA 0.393 versus cue-masked text test BA 0.679.
- Therefore the current source direction is eliminated. The next branch requires manual dominance labels, not another broad source-frame poke.

## Queue

| Field | Value |
| --- | --- |
| Review queue | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_economic_pocket_dominance_review_20260501.jsonl |
| Unique review rows | 51 |
| Labels source | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_economic_source_frame_labels_v1.jsonl |
| Prompt bank | /home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_poke_prompts_v1.jsonl |
| Clean cached probe | /home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_economic_clean_broad_limits_cached_20260501/report.md |
| Filter | frame in {economic_commerce_broad_aggregation,economic_commerce_limits}; expected_frame == frame; has_multi_frame_conflict == false |

## Causal Pocket Evidence To Follow Up

| Run | Prompt | Alpha | Winning comparisons | Comparisons |
| --- | --- | --- | --- | --- |
| split_00_best_probe_direction__last | EA03_gun_school_zone | 0.02 | 2 | base, random_closest_mean |
| split_00_best_probe_direction__last | EA03_gun_school_zone | 0.1 | 2 | base, random_closest_mean |
| split_01_best_probe_direction__all | EA01_commercial_remedy | 0.01 | 1 | base |
| split_01_best_probe_direction__all | EA01_commercial_remedy | 0.02 | 3 | base, random_closest_mean, random_strongest_target |
| split_01_best_probe_direction__all | EA01_commercial_remedy | 0.05 | 1 | base |

## Review Counts

| Unique source frame | N |
| --- | --- |
| economic_commerce_broad_aggregation | 28 |
| economic_commerce_limits | 23 |

## Planned Pocket Coverage

| Pocket side | N before dominance review |
| --- | --- |
| EA01_broad_remedy_market:contrast | 23 |
| EA01_broad_remedy_market:target | 21 |
| EA03_limits_school_zone:contrast | 24 |
| EA03_limits_school_zone:target | 23 |

## Case Coverage

| Frame | Case id | Case | N |
| --- | --- | --- | --- |
| economic_commerce_broad_aggregation | champion_1903 | Champion v. Ames | 3 |
| economic_commerce_broad_aggregation | gibbons_1824 | Gibbons v. Ogden | 1 |
| economic_commerce_broad_aggregation | heart_atlanta_1964 | Heart of Atlanta Motel, Inc. v. United States | 4 |
| economic_commerce_broad_aggregation | hodel_1981 | Hodel v. Virginia Surface Mining & Reclamation Assn., Inc. | 6 |
| economic_commerce_broad_aggregation | katzenbach_mcclung_1964 | Katzenbach v. McClung | 1 |
| economic_commerce_broad_aggregation | perez_1971 | Perez v. United States | 4 |
| economic_commerce_broad_aggregation | raich_2005 | Gonzales v. Raich | 3 |
| economic_commerce_broad_aggregation | shreveport_1914 | Houston, East & West Texas Railway Co. v. United States | 1 |
| economic_commerce_broad_aggregation | stafford_1922 | Stafford v. Wallace | 1 |
| economic_commerce_broad_aggregation | wickard_1942 | Wickard v. Filburn | 4 |
| economic_commerce_limits | carter_coal_1936 | Carter v. Carter Coal Co. | 4 |
| economic_commerce_limits | lopez_1995 | United States v. Lopez | 3 |
| economic_commerce_limits | morrison_2000 | United States v. Morrison | 9 |
| economic_commerce_limits | nfib_2012 | National Federation of Independent Business v. Sebelius | 5 |
| economic_commerce_limits | schechter_1935 | A. L. A. Schechter Poultry Corp. v. United States | 2 |

## Review Instructions

For each row, assign `dominant_frame_label` using only the substance of the excerpt. Ignore explicit cue words, case names, and citations where possible because those are the easiest leakage path.

Allowed labels:

- `dominant_broad_commerce`: aggregate effects, national market, channels/instrumentalities, comprehensive federal scheme, or broad deference to congressional economic regulation.
- `dominant_commerce_limits`: non-economic/local conduct, missing jurisdictional element, attenuated causal chain, activity/inactivity, direct/indirect production limit, or no general police power.
- `dominant_state_federalism`: state sovereignty, anti-commandeering, reserved powers, or state regulatory authority is the main frame rather than Commerce Clause scope.
- `dominant_statutory_or_remedy`: statutory interpretation, remedial design, damages, preemption, or FAA-like analysis dominates.
- `mixed_no_dominant_frame`: both sides are genuinely present and neither dominates.
- `reject_noise_or_boilerplate`: syllabus debris, citation string, procedural noise, or otherwise unusable.

## Promotion Gate

1. After review, keep only rows labeled `dominant_broad_commerce` or `dominant_commerce_limits` with medium/high confidence.
2. Require at least 20 rows per side and at least 3 source cases per side before any BF16 activation capture.
3. Run a source-case-heldout activation probe on cue-masked text and require test BA to beat the cue-masked text baseline by at least 0.05.
4. If that passes, run causal generation only on the two pocket prompts with prompt-matched random controls, and promote only if the candidate beats baseline, random mean, and strongest random under blind review.

## Decision

Do not run another broad justice-style or broad Economic Activity poke from the existing labels.

## Follow-up Result

The dominance review and cached reviewed-label probe have now run:

- Dominance adjudication: `reports/scotus_economic_pocket_dominance_adjudication_20260501.md`
- Reviewed labels: `data/scotus/scotus_economic_pocket_dominance_adjudicated_20260501.jsonl`
- Reviewed-label cached probe: `sweep_v4/scotus_economic_reviewed_broad_limits_cached_20260501/report.md`

The adjudication kept `28` broad-Commerce rows and `21` Commerce-limits rows, clearing the data-count gate. The reviewed-label cached probe then failed the activation gate: best activation test BA `0.473`, cue-masked text test BA `0.857`.

Decision: the Economic Activity source-frame branch is closed under the current protocol. Do not run a causal pocket pilot from this reviewed source direction.
