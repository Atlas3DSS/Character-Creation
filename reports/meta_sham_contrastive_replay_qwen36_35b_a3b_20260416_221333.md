# Meta/Sham Contrastive Replay

- Timestamp: `2026-04-16T22:38:18-07:00`
- Model path: `/home/orwel/dev_genius/models/Qwen3.6-35B-A3B`
- Examples: `72` total, `9` wins, `9` regressions
- Layers: `40`, hidden size: `2048`
- Viable signal: `True`

## Top Robust Windows

| Layer | Region | Mean LOO | Min LOO | Mean Gap | Comparisons |
| --- | --- | ---: | ---: | ---: | --- |
| `L38` | `final_answer_region` | `0.902` | `0.882` | `0.291` | `real_minus_sham, real_minus_think, real_minus_generic` |
| `L00` | `think_region` | `0.882` | `0.824` | `0.363` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L01` | `think_region` | `0.882` | `0.824` | `0.317` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L39` | `think_region` | `0.882` | `0.824` | `0.312` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L00` | `final_answer_region` | `0.863` | `0.647` | `0.331` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L03` | `final_answer_region` | `0.863` | `0.647` | `0.240` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L04` | `final_answer_region` | `0.863` | `0.647` | `0.224` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L37` | `think_region` | `0.863` | `0.824` | `0.315` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L38` | `think_region` | `0.863` | `0.824` | `0.296` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L36` | `think_region` | `0.863` | `0.824` | `0.288` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L35` | `think_region` | `0.863` | `0.824` | `0.280` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L37` | `final_answer_region` | `0.863` | `0.824` | `0.274` | `real_minus_think, real_minus_generic, real_minus_sham` |

## Top Single Comparison Windows

| Comparison | Layer | Region | LOO | Gap | Mean Delta Norm |
| --- | --- | --- | ---: | ---: | ---: |
| `real_minus_think` | `L00` | `final_answer_region` | `1.000` | `0.460` | `0.16` |
| `real_minus_think` | `L01` | `final_answer_region` | `1.000` | `0.352` | `0.21` |
| `real_minus_think` | `L02` | `final_answer_region` | `1.000` | `0.321` | `0.25` |
| `real_minus_think` | `L03` | `final_answer_region` | `1.000` | `0.314` | `0.30` |
| `real_minus_think` | `L04` | `final_answer_region` | `1.000` | `0.286` | `0.33` |
| `real_minus_think` | `L05` | `final_answer_region` | `1.000` | `0.238` | `0.37` |
| `real_minus_think` | `L10` | `final_answer_region` | `1.000` | `0.229` | `0.52` |
| `real_minus_think` | `L06` | `final_answer_region` | `1.000` | `0.226` | `0.39` |
| `real_minus_think` | `L09` | `final_answer_region` | `1.000` | `0.203` | `0.48` |
| `real_minus_think` | `L08` | `final_answer_region` | `1.000` | `0.193` | `0.47` |
| `real_minus_think` | `L07` | `final_answer_region` | `1.000` | `0.186` | `0.45` |
| `real_minus_think` | `L00` | `think_region` | `0.941` | `0.465` | `0.11` |
| `real_minus_think` | `L01` | `think_region` | `0.941` | `0.396` | `0.14` |
| `real_minus_think` | `L39` | `think_region` | `0.941` | `0.380` | `29.54` |
| `real_minus_generic` | `L00` | `final_answer_region` | `0.941` | `0.317` | `0.15` |

## Best By Region

| Region | Comparison | Layer | LOO | Gap |
| --- | --- | --- | ---: | ---: |
| `final_answer_region` | `real_minus_think` | `L00` | `1.000` | `0.460` |
| `think_region` | `real_minus_think` | `L00` | `0.941` | `0.465` |
| `assistant_all` | `real_minus_think` | `L39` | `0.889` | `0.184` |
| `answer_region` | `real_minus_think` | `L38` | `0.824` | `0.235` |
| `assistant_late` | `real_minus_think` | `L38` | `0.778` | `0.119` |
| `prompt_last` | `real_minus_sham` | `L00` | `0.778` | `0.001` |
| `assistant_early` | `real_minus_generic` | `L06` | `0.667` | `0.005` |
| `control_block` | `real_minus_generic` | `L00` | `0.611` | `0.003` |
