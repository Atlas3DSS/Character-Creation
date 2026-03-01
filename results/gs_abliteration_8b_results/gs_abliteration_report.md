# GS-Protected Abliteration (Qwen3-VL-8B-Instruct, INT8)
Date: 2026-02-28 22:45:16

## Setup
- Eval battery counts: {'math': 50, 'knowledge': 30, 'refusal': 10, 'code': 10, 'total': 100}
- GS protection order: Code → Science → Analytical → Math
- Surgical hub layers: [15, 16, 17, 18, 19, 20, 21, 22]

## GS Contamination Summary
| Category | pre mean | post mean | pre max | post max |
|---|---:|---:|---:|---:|
| Code | 0.0758 | 0.0191 | 0.2174 | 0.0757 |
| Science | 0.0791 | 0.0299 | 0.1989 | 0.0600 |
| Analytical | 0.1066 | 0.0428 | 0.2830 | 0.1132 |
| Math | 0.0474 | 0.0000 | 0.1284 | 0.0000 |

- Mean refusal magnitude removed by GS: 1.76%

## Condition Comparison
| ID | Condition | Layers | Direction | GS | Math | Code | Knowledge | Refusal |
|---|---|---|---|---|---:|---:|---:|---:|
| C0 | Base (no abliteration) | — | none | — | 90.0% | 100.0% | 93.3% | 10.0% |
| C1 | Sloppy 32-pair mean-diff extraction | All 36 | sloppy | No | 94.0% | 100.0% | 93.3% | 0.0% |
| C2 | Raw connectome refusal direction | All 36 | raw | No | 92.0% | 100.0% | 90.0% | 20.0% |
| C3 | GS-protected connectome refusal direction | All 36 | gs | Yes | 90.0% | 100.0% | 96.7% | 30.0% |
| C4 | Surgical GS (hub L15-L22) | Hub L15-L22 | gs | Yes | 90.0% | 100.0% | 93.3% | 10.0% |

## Key Checks
- C2 Code drop vs C0: -0.00 pp (negative means drop)
- C3 Code delta vs C0: +0.00 pp
- C4 vs C3 Code delta: +0.00 pp
- C4 vs C3 Math delta: +0.00 pp

- C2 Code drop vs C0 >= 5pp: NO
- C3 Code within 2pp of C0: YES
- C2/C3/C4 refusal <= 5pp: NO
- C4 ≈ C3 (Math+Code within 2pp): YES
- C1 worst overall (Math+Code): NO
