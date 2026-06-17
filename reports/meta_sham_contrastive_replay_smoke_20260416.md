# Meta/Sham Contrastive Replay

- Timestamp: `2026-04-16T21:13:02-07:00`
- Model path: `/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Examples: `16` total, `2` wins, `2` regressions
- Layers: `32`, hidden size: `4096`
- Viable signal: `True`

## Top Robust Windows

| Layer | Region | Mean LOO | Min LOO | Mean Gap | Comparisons |
| --- | --- | ---: | ---: | ---: | --- |
| `L01` | `final_answer_region` | `1.000` | `1.000` | `0.907` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L00` | `final_answer_region` | `1.000` | `1.000` | `0.905` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L02` | `final_answer_region` | `1.000` | `1.000` | `0.827` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L07` | `final_answer_region` | `1.000` | `1.000` | `0.702` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L08` | `final_answer_region` | `1.000` | `1.000` | `0.695` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L03` | `final_answer_region` | `1.000` | `1.000` | `0.679` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L09` | `final_answer_region` | `1.000` | `1.000` | `0.667` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L06` | `final_answer_region` | `1.000` | `1.000` | `0.644` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L05` | `final_answer_region` | `1.000` | `1.000` | `0.636` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L10` | `final_answer_region` | `1.000` | `1.000` | `0.620` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L04` | `final_answer_region` | `1.000` | `1.000` | `0.603` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L31` | `final_answer_region` | `1.000` | `1.000` | `0.568` | `real_minus_sham, real_minus_generic, real_minus_think` |

## Top Single Comparison Windows

| Comparison | Layer | Region | LOO | Gap | Mean Delta Norm |
| --- | --- | --- | ---: | ---: | ---: |
| `real_minus_think` | `L00` | `final_answer_region` | `1.000` | `1.009` | `0.37` |
| `real_minus_generic` | `L01` | `final_answer_region` | `1.000` | `0.976` | `0.59` |
| `real_minus_think` | `L01` | `final_answer_region` | `1.000` | `0.905` | `0.62` |
| `real_minus_generic` | `L00` | `final_answer_region` | `1.000` | `0.891` | `0.35` |
| `real_minus_generic` | `L02` | `final_answer_region` | `1.000` | `0.869` | `0.75` |
| `real_minus_think` | `L02` | `final_answer_region` | `1.000` | `0.860` | `0.78` |
| `real_minus_sham` | `L01` | `final_answer_region` | `1.000` | `0.839` | `0.60` |
| `real_minus_sham` | `L00` | `final_answer_region` | `1.000` | `0.814` | `0.35` |
| `real_minus_sham` | `L09` | `final_answer_region` | `1.000` | `0.785` | `2.94` |
| `real_minus_think` | `L00` | `think_region` | `1.000` | `0.783` | `0.25` |
| `real_minus_sham` | `L08` | `final_answer_region` | `1.000` | `0.782` | `2.66` |
| `real_minus_sham` | `L10` | `final_answer_region` | `1.000` | `0.758` | `3.24` |
| `real_minus_sham` | `L02` | `final_answer_region` | `1.000` | `0.753` | `0.74` |
| `real_minus_sham` | `L07` | `final_answer_region` | `1.000` | `0.748` | `2.55` |
| `real_minus_generic` | `L07` | `final_answer_region` | `1.000` | `0.743` | `2.58` |
