# Meta/Sham Contrastive Replay

- Timestamp: `2026-04-16T21:19:19-07:00`
- Model path: `/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Examples: `72` total, `9` wins, `9` regressions
- Layers: `32`, hidden size: `4096`
- Viable signal: `True`

## Top Robust Windows

| Layer | Region | Mean LOO | Min LOO | Mean Gap | Comparisons |
| --- | --- | ---: | ---: | ---: | --- |
| `L00` | `assistant_all` | `0.833` | `0.778` | `0.259` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L01` | `assistant_all` | `0.833` | `0.778` | `0.259` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L02` | `assistant_all` | `0.833` | `0.833` | `0.141` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L31` | `assistant_all` | `0.833` | `0.778` | `0.119` | `real_minus_think, real_minus_sham, real_minus_generic` |
| `L03` | `assistant_all` | `0.833` | `0.833` | `0.111` | `real_minus_generic, real_minus_think, real_minus_sham` |
| `L04` | `assistant_all` | `0.815` | `0.778` | `0.095` | `real_minus_generic, real_minus_sham, real_minus_think` |
| `L30` | `assistant_all` | `0.815` | `0.778` | `0.112` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L29` | `assistant_all` | `0.815` | `0.778` | `0.107` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L28` | `assistant_all` | `0.815` | `0.778` | `0.106` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L27` | `assistant_all` | `0.796` | `0.778` | `0.089` | `real_minus_think, real_minus_generic, real_minus_sham` |
| `L25` | `assistant_all` | `0.796` | `0.778` | `0.078` | `real_minus_sham, real_minus_generic, real_minus_think` |
| `L26` | `assistant_all` | `0.796` | `0.778` | `0.076` | `real_minus_sham, real_minus_generic, real_minus_think` |

## Top Single Comparison Windows

| Comparison | Layer | Region | LOO | Gap | Mean Delta Norm |
| --- | --- | --- | ---: | ---: | ---: |
| `real_minus_generic` | `L01` | `assistant_all` | `0.889` | `0.290` | `0.39` |
| `real_minus_generic` | `L00` | `assistant_all` | `0.889` | `0.261` | `0.24` |
| `real_minus_think` | `L31` | `assistant_all` | `0.889` | `0.181` | `32.32` |
| `real_minus_think` | `L30` | `assistant_all` | `0.889` | `0.163` | `43.12` |
| `real_minus_think` | `L29` | `assistant_all` | `0.889` | `0.156` | `38.50` |
| `real_minus_think` | `L28` | `assistant_all` | `0.889` | `0.147` | `35.44` |
| `real_minus_think` | `L04` | `prompt_last` | `0.889` | `0.008` | `0.47` |
| `real_minus_think` | `L00` | `assistant_all` | `0.833` | `0.441` | `0.26` |
| `real_minus_think` | `L01` | `assistant_all` | `0.833` | `0.360` | `0.43` |
| `real_minus_generic` | `L02` | `assistant_all` | `0.833` | `0.185` | `0.52` |
| `real_minus_think` | `L02` | `assistant_all` | `0.833` | `0.157` | `0.58` |
| `real_minus_generic` | `L03` | `assistant_all` | `0.833` | `0.142` | `0.78` |
| `real_minus_generic` | `L04` | `assistant_all` | `0.833` | `0.127` | `0.94` |
| `real_minus_think` | `L03` | `assistant_all` | `0.833` | `0.126` | `0.83` |
| `real_minus_think` | `L27` | `assistant_all` | `0.833` | `0.121` | `31.06` |

## Best By Region

| Region | Comparison | Layer | LOO | Gap |
| --- | --- | --- | ---: | ---: |
| `assistant_all` | `real_minus_generic` | `L01` | `0.889` | `0.290` |
| `prompt_last` | `real_minus_think` | `L04` | `0.889` | `0.008` |
| `assistant_late` | `real_minus_generic` | `L00` | `0.778` | `0.144` |
| `assistant_early` | `real_minus_generic` | `L04` | `0.667` | `0.007` |
| `control_block` | `real_minus_generic` | `L00` | `0.667` | `0.005` |
