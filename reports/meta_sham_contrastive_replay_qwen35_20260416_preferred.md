# Meta/Sham Contrastive Replay

- Timestamp: `2026-04-16T21:26:08-07:00`
- Model path: `/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Examples: `72` total, `9` wins, `9` regressions
- Layers: `32`, hidden size: `4096`
- Viable signal: `True`

## Top Robust Windows

| Layer | Region | Mean LOO | Min LOO | Mean Gap | Comparisons |
| --- | --- | ---: | ---: | ---: | --- |
| `L30` | `final_answer_region` | `0.863` | `0.706` | `0.343` | `real_minus_sham, real_minus_think, real_minus_generic` |
| `L01` | `think_region` | `0.863` | `0.824` | `0.505` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L29` | `final_answer_region` | `0.863` | `0.765` | `0.340` | `real_minus_sham, real_minus_think, real_minus_generic` |
| `L29` | `think_region` | `0.863` | `0.824` | `0.303` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L04` | `think_region` | `0.863` | `0.824` | `0.302` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L30` | `think_region` | `0.863` | `0.824` | `0.299` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L28` | `final_answer_region` | `0.863` | `0.765` | `0.297` | `real_minus_sham, real_minus_think, real_minus_generic` |
| `L28` | `think_region` | `0.863` | `0.824` | `0.290` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L27` | `think_region` | `0.863` | `0.824` | `0.285` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L26` | `think_region` | `0.863` | `0.824` | `0.251` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L25` | `think_region` | `0.863` | `0.824` | `0.237` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L24` | `think_region` | `0.863` | `0.824` | `0.235` | `real_minus_sham, real_minus_think, real_minus_generic` |

## Top Single Comparison Windows

| Comparison | Layer | Region | LOO | Gap | Mean Delta Norm |
| --- | --- | --- | ---: | ---: | ---: |
| `real_minus_think` | `L00` | `final_answer_region` | `1.000` | `0.386` | `0.37` |
| `real_minus_sham` | `L30` | `final_answer_region` | `1.000` | `0.332` | `55.41` |
| `real_minus_think` | `L01` | `final_answer_region` | `1.000` | `0.273` | `0.60` |
| `real_minus_think` | `L02` | `final_answer_region` | `1.000` | `0.228` | `0.78` |
| `real_minus_think` | `L06` | `final_answer_region` | `1.000` | `0.220` | `2.25` |
| `real_minus_think` | `L07` | `final_answer_region` | `1.000` | `0.213` | `2.64` |
| `real_minus_think` | `L03` | `final_answer_region` | `1.000` | `0.167` | `1.12` |
| `real_minus_think` | `L05` | `final_answer_region` | `1.000` | `0.155` | `1.74` |
| `real_minus_think` | `L04` | `final_answer_region` | `1.000` | `0.152` | `1.43` |
| `real_minus_think` | `L00` | `think_region` | `0.941` | `0.750` | `0.30` |
| `real_minus_think` | `L01` | `think_region` | `0.941` | `0.668` | `0.49` |
| `real_minus_sham` | `L29` | `final_answer_region` | `0.941` | `0.345` | `48.79` |
| `real_minus_sham` | `L28` | `final_answer_region` | `0.941` | `0.320` | `43.17` |
| `real_minus_think` | `L08` | `final_answer_region` | `0.941` | `0.252` | `2.77` |
| `real_minus_generic` | `L01` | `assistant_all` | `0.889` | `0.290` | `0.39` |

## Best By Region

| Region | Comparison | Layer | LOO | Gap |
| --- | --- | --- | ---: | ---: |
| `final_answer_region` | `real_minus_think` | `L00` | `1.000` | `0.386` |
| `think_region` | `real_minus_think` | `L00` | `0.941` | `0.750` |
| `assistant_all` | `real_minus_generic` | `L01` | `0.889` | `0.290` |
| `prompt_last` | `real_minus_think` | `L04` | `0.889` | `0.008` |
| `answer_region` | `real_minus_think` | `L00` | `0.882` | `0.312` |
| `assistant_late` | `real_minus_generic` | `L00` | `0.778` | `0.144` |
| `assistant_early` | `real_minus_generic` | `L04` | `0.667` | `0.007` |
| `control_block` | `real_minus_generic` | `L00` | `0.667` | `0.005` |
