# Qwen-Scope SAE Download Notes

Date: 2026-04-30

## Sources Read

- Qwen blog: `https://qwen.ai/blog?id=qwen-scope`
- Hugging Face collection: `https://huggingface.co/collections/Qwen/qwen-scope`
- Technical report: `https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf`

The Qwen blog page is a client-rendered page, so the technical content was read from the Qwen-Scope PDF and Hugging Face model cards.

## Why These SAEs Are Relevant

Our active SCOTUS probe run used:

```text
/home/orwel/dev_genius/models/Qwen3.6-27B-FP8
```

Its text stack has:

| Property | Value |
| --- | --- |
| Model type | `qwen3_5_text` |
| Hidden size | `5120` |
| Layers | `64` |
| Attention heads | `24` |
| KV heads | `4` |

The matching Qwen-Scope release is the `Qwen3.5-27B` SAE family:

| Repo | Top-k / L0 | Hidden size | SAE width | Layers |
| --- | --- | --- | --- | --- |
| `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` | `50` | `5120` | `81920` | `0-63` |
| `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100` | `100` | `5120` | `81920` | `0-63` |

These are the relevant SAE sets for the current 27B activation workflow. The other Qwen-Scope repos target smaller dense models or MoE backbones and are not the immediate match for the SCOTUS Phase 4 probe.

## Downloaded Artifacts

Downloaded with `huggingface_hub.snapshot_download`, using the standard Hugging Face cache.

| Repo | Commit | SAE files | Size |
| --- | --- | ---: | ---: |
| `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` | `87e60f5c6de567f83f4633687378c3146434b2e6` | `64` | `200.02 GiB` |
| `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100` | `91e2c1d4b2db75847876b1e7dcdbb068aea6bc6b` | `64` | `200.02 GiB` |

Stable local symlinks:

```text
/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_50
/home/orwel/dev_genius/models/qwen_scope/SAE-Res-Qwen3.5-27B-W80K-L0_100
```

HF cache snapshots:

```text
/home/orwel/.cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3.5-27B-W80K-L0_50/snapshots/87e60f5c6de567f83f4633687378c3146434b2e6
/home/orwel/.cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3.5-27B-W80K-L0_100/snapshots/91e2c1d4b2db75847876b1e7dcdbb068aea6bc6b
```

Each repo includes:

- `layer0.sae.pt` through `layer63.sae.pt`
- `README.md`
- `LICENSE`
- `app.py`

Each `layerN.sae.pt` is a PyTorch dict with:

| Key | Shape |
| --- | --- |
| `W_enc` | `(81920, 5120)` |
| `W_dec` | `(5120, 81920)` |
| `b_enc` | `(81920,)` |
| `b_dec` | `(5120,)` |

## Immediate Implication For SCOTUS

The next useful integration is not to replace the Phase 4 linear probes. It is to add an SAE feature-analysis branch:

1. Capture the same residual-stream activations used by `probe_scotus_style.py`.
2. Encode them through the matching layer SAE.
3. Train justice classifiers on sparse SAE activations.
4. Compare SAE-feature probes against raw hidden-state probes and masked text baselines.
5. Inspect top discriminative SAE features for Scalia/Ginsburg legal-reasoning content.
6. Only after diagnostic checks, test whether selected SAE decoder columns can steer neutral legal continuations.
