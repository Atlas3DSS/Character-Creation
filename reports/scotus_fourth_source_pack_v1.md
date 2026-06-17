# SCOTUS Fourth Amendment Source Pack v1

## Purpose

This expands Fourth Amendment source-grounded labels beyond the target-justice chunks and excludes the earlier non-Fourth-Amendment technology false positives. It is a silver-label source pack for cue-masked diagnostics and review, not final circuit evidence.

## Outputs

- Raw pages: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/raw/scotus_fourth_source_pages_v1.json`
- Labels: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_fourth_source_frame_labels_v1.jsonl`
- Review queue: `/home/orwel/dev_genius/experiments/Character Creation/data/scotus/scotus_fourth_source_frame_review_queue_v1.jsonl`
- Source chunks scanned: `1018`

## Source Cases

| Case id | Case | Citation | Expected frame | Tokens | URL |
| --- | --- | --- | --- | --- | --- |
| chimel_1969 | Chimel v. California | 395 U.S. 752 | fourth_search_incident_chimel | 14236 | https://www.law.cornell.edu/supremecourt/text/395/752 |
| robinson_1973 | United States v. Robinson | 414 U.S. 218 | fourth_search_incident_chimel | 17699 | https://www.law.cornell.edu/supremecourt/text/414/218 |
| belton_1981 | New York v. Belton | 453 U.S. 454 | fourth_search_incident_chimel | 8998 | https://www.law.cornell.edu/supremecourt/text/453/454 |
| gant_2009 | Arizona v. Gant | 556 U.S. 332 | fourth_search_incident_chimel | 1101 | https://www.law.cornell.edu/supct/html/07-542.ZS.html |
| riley_2014 | Riley v. California | 573 U.S. 373 | fourth_technology_privacy | 15440 | https://www.law.cornell.edu/supremecourt/text/13-132 |
| kyllo_2001 | Kyllo v. United States | 533 U.S. 27 | fourth_technology_privacy | 10822 | https://www.law.cornell.edu/supremecourt/text/533/27 |
| jones_2012 | United States v. Jones | 565 U.S. 400 | fourth_technology_privacy | 13727 | https://www.law.cornell.edu/supremecourt/text/10-1259 |
| carpenter_2018 | Carpenter v. United States | 585 U.S. 296 | fourth_technology_privacy | 48104 | https://www.law.cornell.edu/supremecourt/text/16-402 |
| hicks_1987 | Arizona v. Hicks | 480 U.S. 321 | fourth_plain_view_independent_source | 8734 | https://www.law.cornell.edu/supremecourt/text/480/321 |
| horton_1990 | Horton v. California | 496 U.S. 128 | fourth_plain_view_independent_source | 12469 | https://www.law.cornell.edu/supremecourt/text/496/128 |
| murray_1988 | Murray v. United States | 487 U.S. 533 | fourth_plain_view_independent_source | 8314 | https://www.law.cornell.edu/supremecourt/text/487/533 |
| segura_1984 | Segura v. United States | 468 U.S. 796 | fourth_plain_view_independent_source | 21469 | https://www.law.cornell.edu/supremecourt/text/468/796 |
| payton_1980 | Payton v. New York | 445 U.S. 573 | fourth_home_exigency | 23994 | https://www.law.cornell.edu/supremecourt/text/445/573 |
| mincey_1978 | Mincey v. Arizona | 437 U.S. 385 | fourth_home_exigency | 12841 | https://www.law.cornell.edu/supremecourt/text/437/385 |
| brigham_city_2006 | Brigham City v. Stuart | 547 U.S. 398 | fourth_home_exigency | 2910 | https://www.law.cornell.edu/supct/html/05-502.ZO.html |
| king_2011 | Kentucky v. King | 563 U.S. 452 | fourth_home_exigency | 7182 | https://www.law.cornell.edu/supct/html/09-1272.ZO.html |
| lange_2021 | Lange v. California | 594 U.S. ___ | fourth_home_exigency | 19717 | https://www.law.cornell.edu/supremecourt/text/20-18 |
| randolph_2006 | Georgia v. Randolph | 547 U.S. 103 | fourth_home_exigency | 7467 | https://www.law.cornell.edu/supct/html/04-1067.ZO.html |

## Label Counts

| Frame | Total | Train | Dev | Test | Multi-frame conflicts |
| --- | --- | --- | --- | --- | --- |
| fourth_search_incident_chimel | 72 | 66 | 4 | 2 | 10 |
| fourth_technology_privacy | 72 | 52 | 14 | 6 | 8 |
| fourth_plain_view_independent_source | 72 | 68 | 2 | 2 | 8 |
| fourth_home_exigency | 72 | 40 | 2 | 30 | 7 |

## Case/Frame Coverage

| Case id | Case | Frame | Records |
| --- | --- | --- | --- |
| belton_1981 | New York v. Belton | fourth_search_incident_chimel | 25 |
| brigham_city_2006 | Brigham City v. Stuart | fourth_home_exigency | 3 |
| carpenter_2018 | Carpenter v. United States | fourth_plain_view_independent_source | 1 |
| carpenter_2018 | Carpenter v. United States | fourth_technology_privacy | 43 |
| chimel_1969 | Chimel v. California | fourth_home_exigency | 1 |
| chimel_1969 | Chimel v. California | fourth_search_incident_chimel | 6 |
| gant_2009 | Arizona v. Gant | fourth_search_incident_chimel | 3 |
| hicks_1987 | Arizona v. Hicks | fourth_home_exigency | 1 |
| hicks_1987 | Arizona v. Hicks | fourth_plain_view_independent_source | 11 |
| hicks_1987 | Arizona v. Hicks | fourth_technology_privacy | 1 |
| horton_1990 | Horton v. California | fourth_home_exigency | 1 |
| horton_1990 | Horton v. California | fourth_plain_view_independent_source | 15 |
| horton_1990 | Horton v. California | fourth_search_incident_chimel | 1 |
| jones_2012 | United States v. Jones | fourth_technology_privacy | 8 |
| king_2011 | Kentucky v. King | fourth_home_exigency | 8 |
| king_2011 | Kentucky v. King | fourth_plain_view_independent_source | 1 |
| kyllo_2001 | Kyllo v. United States | fourth_plain_view_independent_source | 1 |
| kyllo_2001 | Kyllo v. United States | fourth_technology_privacy | 6 |
| lange_2021 | Lange v. California | fourth_home_exigency | 29 |
| lange_2021 | Lange v. California | fourth_search_incident_chimel | 1 |
| mincey_1978 | Mincey v. Arizona | fourth_home_exigency | 1 |
| mincey_1978 | Mincey v. Arizona | fourth_plain_view_independent_source | 1 |
| mincey_1978 | Mincey v. Arizona | fourth_search_incident_chimel | 1 |
| murray_1988 | Murray v. United States | fourth_home_exigency | 1 |
| murray_1988 | Murray v. United States | fourth_plain_view_independent_source | 19 |
| payton_1980 | Payton v. New York | fourth_home_exigency | 19 |
| payton_1980 | Payton v. New York | fourth_plain_view_independent_source | 1 |
| payton_1980 | Payton v. New York | fourth_search_incident_chimel | 1 |
| randolph_2006 | Georgia v. Randolph | fourth_home_exigency | 6 |
| randolph_2006 | Georgia v. Randolph | fourth_plain_view_independent_source | 1 |
| randolph_2006 | Georgia v. Randolph | fourth_search_incident_chimel | 1 |
| riley_2014 | Riley v. California | fourth_home_exigency | 1 |
| riley_2014 | Riley v. California | fourth_search_incident_chimel | 4 |
| riley_2014 | Riley v. California | fourth_technology_privacy | 14 |
| robinson_1973 | United States v. Robinson | fourth_search_incident_chimel | 28 |
| segura_1984 | Segura v. United States | fourth_home_exigency | 1 |
| segura_1984 | Segura v. United States | fourth_plain_view_independent_source | 21 |
| segura_1984 | Segura v. United States | fourth_search_incident_chimel | 1 |

## Sample Evidence Windows

| Frame | Split | Case | Citation | Evidence | Conflict | Window |
| --- | --- | --- | --- | --- | --- | --- |
| fourth_home_exigency | dev | Chimel v. California | 395 U.S. 752 | \bexigen(?:t|cy|cies)\b | no | he police were not required to obtain a search warrant in advance, even though they knew that the effect of the arrest might well be to alert petitioner's wife that the coins had better be removed soon. Thus it is necess |
| fourth_home_exigency | dev | Mincey v. Arizona | 437 U.S. 385 | \bexigen(?:t|cy|cies)\b, \bhot pursuit\b | no | roperty may not be totally sacrificed in the name of maximum simplicity in enforcement of the criminal law. See United States v. Chadwick, 433 U.S. 1, 6-11, 97 S.Ct. 2476, 2481-2483, 53 L.Ed.2d 538. For this reason, warr |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b, \bentry into (?:a |the )?home\b | no | hts created probable cause to arrest Lange for the misdemeanor of failing to comply with a police signal. And it stated that Lange could not defeat an arrest begun in a public place by retreating into his home. The pursu |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bhot pursuit\b, \bwarrantless entry\b | no | The Fourth Amendment ordinarily requires that a law enforcement officer obtain a judicial warrant before entering a home without permission. Riley v. California, 573 U. S. 373, 382. But an officer may make a warrantless  |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | the calculus changes—but not enough to justify a categorical rule. In many cases, flight creates a need for police to act swiftly. But no evidence suggests that every case of misdemeanor flight creates such a need. The C |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | Justice Kagan delivered the opinion of the Court. The Fourth Amendment ordinarily requires that police officers get a warrant before entering a home without permission. But an officer may make a warrantless entry when “t |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bhot pursuit\b, \bwarrantless entry\b | no | iling to comply with a police signal. See, e.g., Cal. Veh. Code Ann. §2800(a) (West 2015) (making it a misdemeanor to “willfully fail or refuse to comply with a lawful order, signal, or direction of a peace officer”). An |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | The exception enables law enforcement officers to handle “emergenc[ies]”—situations presenting a “compelling need for official action and no time to secure a warrant.” Riley, 573 U. S., at 402; Missouri v. McNeely, 569 U |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | ke a warrantless entry. Id., at 149. The question here is whether to use that approach, or instead apply a categorical warrant exception, when a suspected misdemeanant flees from police into his home. Under the usual cas |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | . New York, 445 U. S. 573, 585, 587 (1980) (internal quotation marks omitted). The Amendment thus “draw[s] a firm line at the entrance to the house.” Id., at 590. What lies behind that line is of course not inviolable. A |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bhot pursuit\b, \bwarrantless entry\b, \bentry into (?:a |the )?home\b | no | up, they saw Santana standing in her home’s open doorway, some 15 feet away. As they got out of the van and yelled “police,” Santana “retreated into [the house’s] vestibule.” Id., at 40. The officers followed her in, and |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bhot pursuit\b, \bwarrantless entry\b | no | ntana, as we have suggested before. In rejecting the amicus’s view, we see no need to consider Lange’s counterargument that Santana did not establish any categorical rule—even one for fleeing felons. See Brief for Petiti |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bhot pursuit\b, \bwarrantless entry\b | no | said nothing about fleeing misdemeanants. We said as much in Stanton, when we approved qualified immunity for an officer who had pursued a suspected misdemeanant into a home. Describing the same split of authority we too |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b, \bentry into (?:a |the )?home\b | no | e a warrantless entry depended as well on other circumstances suggesting a potential for harm and a need to act promptly.8 In that way, the common-law rules (even if sometimes hard to discern with precision) mostly mirro |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b | no | The concurrence spends most of its time worrying about cases in which there are exigencies above and beyond the flight itself: when, for example, the fleeing misdemeanant will “get a gun and take aim from inside” or “flu |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bhot pursuit\b, \bwarrantless entry\b | no | Justice Kavanaugh, concurring. The Court holds that an officer may make a warrantless entry into a home when pursuing a fleeing misdemeanant if an exigent circumstance is also present—for example, when there is a risk of |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bexigen(?:t|cy|cies)\b, \bwarrantless entry\b, \bentry into (?:a |the )?home\b | no | ill still allow the police to make a warrantless entry into a home “nine times out of 10 or more” in cases involving pursuit of a fleeing misdemeanant. Tr. of Oral Arg. 34. Importantly, moreover, the Court’s opinion does |
| fourth_home_exigency | test | Lange v. California | 594 U.S. ___ | \bhot pursuit\b, \bwarrantless entry\b, \bentry into (?:a |the )?home\b | no | Justice Thomas, with whom Justice kavanaugh joins as to Part II, concurring in part and concurring in the judgment. I join the majority opinion, except for Part II–A, which correctly rejects the argument that suspicion t |

## Use Rules

1. Treat labels as `silver_high`; manually review before any promotion decision.
2. Run probes on `text_cue_masked`; promotion requires surviving cue masking and text-baseline checks.
3. Rows with `has_multi_frame_conflict=true` should be adjudicated before binary frame training.
4. Keep this source pack separate from target-justice style labels; it is for legal-frame source grounding.

## Cue-Masked Probe Result

- Probe run: `/home/orwel/dev_genius/experiments/Character Creation/sweep_v4/scotus_source_frame_probe_20260501_011324/report.md`
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`
- Layers: `8, 12, 16`
- Text field: `text_cue_masked`
- Conflict rows excluded: `true`
- Splits reassigned per task: `true`

| Task | Best readout | Dev BA | Test BA | Text test BA | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `fourth_home_vs_incident` | `excerpt_mean @ L8` | `0.952` | `1.000` | `1.000` | Text/leakage dominated; only `3` test rows |
| `fourth_plain_view_vs_incident` | `prompt_mean @ L8` | `0.977` | `0.838` | `0.809` | Not promoted; marginal over text baseline and split-skewed |
| `fourth_technology_vs_home` | `prompt_mean @ L16` | `1.000` | `1.000` | `0.988` | Text/leakage dominated |
| `fourth_technology_vs_incident` | `prompt_mean @ L16` | `1.000` | `1.000` | `1.000` | Text/leakage dominated |

Decision: this pack fixes the earlier non-Fourth-Amendment contamination, but it does not nominate a steerable direction. Use it for evaluator repair and leakage diagnostics unless dominance review changes the held-out/text-baseline picture.
