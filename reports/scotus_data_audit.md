# SCOTUS Data Audit

Generated: 2026-04-24 22:35:56

## Scope

- Source text: CourtListener authored SCOTUS opinion records (`html_with_citations` preferred).
- Metadata: CourtListener clusters joined to SCDB 2024 case-centered docket metadata by `scdb_id`.
- Targets: Scalia, Ginsburg, Thomas, Souter.
- Chunks: paragraph-first, 150-350 token target, 80 token minimum.
- Variants: `raw_clean` and `masked`.

## Corpus Counts

| Justice | Opinions | Raw chunks | Tokens | Cites / 1k tokens | SCDB joined |
| --- | --- | --- | --- | --- | --- |
| Scalia | 211 | 10727 | 2268658 | 3.70 | 208 |
| Ginsburg | 160 | 7820 | 1635910 | 3.19 | 146 |
| Thomas | 297 | 12380 | 2750628 | 3.41 | 160 |
| Souter | 141 | 7474 | 1586064 | 3.06 | 140 |

## Opinion Type Distribution

| Justice | Opinion Type | Count |
| --- | --- | --- |
| Scalia | combined | 211 |
| Ginsburg | combined | 160 |
| Thomas | combined | 297 |
| Souter | combined | 141 |

## Issue Area Distribution

| Justice | Issue Area | Count |
| --- | --- | --- |
| Scalia | Criminal Procedure | 50 |
| Scalia | Economic Activity | 41 |
| Scalia | Judicial Power | 36 |
| Scalia | Civil Rights | 28 |
| Scalia | Federalism | 13 |
| Scalia | First Amendment | 10 |
| Scalia | Due Process | 9 |
| Scalia | Unions | 8 |
| Scalia | Attorneys | 7 |
| Scalia | Federal Taxation | 4 |
| Scalia | unknown | 3 |
| Scalia | Miscellaneous | 1 |
| Scalia | Privacy | 1 |
| Ginsburg | Judicial Power | 38 |
| Ginsburg | Criminal Procedure | 31 |
| Ginsburg | Economic Activity | 25 |
| Ginsburg | Civil Rights | 19 |
| Ginsburg | unknown | 14 |
| Ginsburg | Federalism | 11 |
| Ginsburg | Federal Taxation | 5 |
| Ginsburg | Due Process | 5 |
| Ginsburg | Attorneys | 4 |
| Ginsburg | Unions | 2 |
| Ginsburg | Privacy | 2 |
| Ginsburg | Interstate Relations | 2 |
| Ginsburg | First Amendment | 2 |
| Thomas | unknown | 137 |
| Thomas | Economic Activity | 52 |
| Thomas | Criminal Procedure | 40 |
| Thomas | Judicial Power | 18 |
| Thomas | Civil Rights | 18 |
| Thomas | Federalism | 12 |
| Thomas | First Amendment | 5 |
| Thomas | Federal Taxation | 4 |
| Thomas | Unions | 3 |
| Thomas | Privacy | 3 |
| Thomas | Attorneys | 2 |
| Thomas | Due Process | 2 |
| Thomas | Interstate Relations | 1 |
| Souter | Economic Activity | 35 |
| Souter | Criminal Procedure | 28 |
| Souter | Judicial Power | 26 |
| Souter | Civil Rights | 22 |
| Souter | Federal Taxation | 7 |
| Souter | Federalism | 7 |
| Souter | First Amendment | 7 |
| Souter | Interstate Relations | 3 |
| Souter | Due Process | 2 |
| Souter | Privacy | 2 |
| Souter | Unions | 1 |
| Souter | unknown | 1 |

## Decade Distribution

| Justice | Decade | Count |
| --- | --- | --- |
| Scalia | 1980s | 49 |
| Scalia | 1990s | 96 |
| Scalia | 2000s | 66 |
| Ginsburg | 1990s | 64 |
| Ginsburg | 2000s | 64 |
| Ginsburg | 2010s | 32 |
| Thomas | 1990s | 75 |
| Thomas | 2000s | 63 |
| Thomas | 2010s | 57 |
| Thomas | 2020s | 102 |
| Souter | 1990s | 84 |
| Souter | 2000s | 57 |

## Same-Case Overlaps

| Pair | Same-case opinion overlaps |
| --- | --- |
| Scalia_vs_Ginsburg | 0 |
| Thomas_vs_Souter | 0 |

## Loose Matched Chunk Budget

| Pair | Loose matched chunk pairs | Metadata cells |
| --- | --- | --- |
| Scalia_vs_Ginsburg | 4460 | 28 |
| Thomas_vs_Souter | 3873 | 28 |

## Named-Case Frequency

| Case Name | Raw chunk mentions |
| --- | --- |
| See United States v. Detroit Timber | 88 |
| Chevron U. S. A. Inc. v. Natural Resources Defense Council | 72 |
| Inc. v. United States | 61 |
| Inc. v. FCC | 54 |
| Buckley v. Valeo | 45 |
| Teague v. Lane | 43 |
| CONSUMER FINANCIAL PROTECTION BUREAU v. COMMU- NITY FINANCIAL SERVICES ASSN. OF | 43 |
| Miranda v. Arizona | 35 |
| GARLAND v. CARGILL Opinion | 35 |
| Johnson v. United States | 34 |
| Bivens v. Six Unknown Fed. Narcotics Agents | 32 |
| United States v. Johnson | 31 |
| Strickland v. Washington | 30 |
| OHIO v. AMERICAN EXPRESS CO. BREYER | 30 |
| Terry v. Ohio | 29 |
| Warth v. Seldin | 29 |
| Apprendi v. New Jersey | 29 |
| PETITIONER v. UNITED STATES ON WRIT OF CERTIORARI | 29 |
| Lujan v. Defenders | 27 |
| Russello v. United States | 27 |

## Go / No-Go

| Pair | Decision | >=500 chunks each | Loose matched chunks | Same-case overlaps |
| --- | --- | --- | --- | --- |
| Scalia_vs_Ginsburg | GO | True | 4460 | 0 |
| Thomas_vs_Souter | GO | True | 3873 | 0 |

## Artifact Paths

- `data/scotus/scotus_opinion_inventory.jsonl`: 809 records
- `data/scotus/scotus_chunk_inventory.jsonl`: 76802 records (38401 raw + 38401 masked)
- `data/scotus/scotus_pair_overlap_inventory.jsonl`: 0 records
