#!/usr/bin/env python3
"""
Combined Bilingual Identity Suppression — EN + CN + Core Bilingual Dims

Key findings this builds on:
- EN and CN identity circuits have only 5.4% overlap (986/18,109 neurons)
- 12 core bilingual dims fire in BOTH languages across 10+ layers
- 54% of identity neurons also encode personality (entangled)
- Previous English-only suppression failed because model routed through Chinese circuit
- Vocab swap failed because model invents alternative names

Strategy:
1. Load EN + CN identity activation data (z-scores per dim per layer)
2. Load CN personality data (assistant suppressed / sarcasm activated dims)
3. Classify each (layer, dim) as: identity-only, identity+personality, personality-only
4. Identity-only dims: suppress aggressively via o_proj bias
5. Identity+personality dims: redirect (push mean_delta from Qwen→Skippy direction)
6. Personality-only dims: gently boost in Skippy/sarcasm direction
7. Also scale lm_head columns for 12 core bilingual dims
8. Graduated alpha: higher for late layers where persona is strongest
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

# ── Configuration ────────────────────────────────────────────────────

BASE_MODEL = "./skippy_sdft_r2/merged_scale_1.0"
OUTPUT_DIR = "./skippy_vectors/bilingual_suppressed"

# The 12 core bilingual dims (fire in both EN and CN, 10+ layers)
CORE_BILINGUAL_DIMS = {
    994: 34, 1984: 22, 2854: 22, 2001: 19, 729: 16, 1147: 14,
    3522: 13, 2833: 12, 3854: 12, 98: 11, 2591: 11, 1838: 10,
}

# Z-score thresholds for classification
IDENTITY_Z_THRESH = 3.0      # |z| > 3 = identity neuron
PERSONALITY_Z_THRESH = 2.0   # |z| > 2 = personality neuron (more permissive)
STRONG_IDENTITY_Z = 5.0      # |z| > 5 = STRONG identity (aggressive suppress)

# Alpha schedules — graduated by layer position
# Late layers have stronger persona signal, need more intervention
ALPHA_SCHEDULE = {
    "identity_suppress": {
        range(0, 6):   0.0,    # Too early, skip
        range(6, 13):  0.05,   # Early-mid: gentle
        range(13, 20): 0.10,   # Mid: moderate
        range(20, 28): 0.15,   # Late-mid: medium
        range(28, 36): 0.20,   # Final layers: strongest
    },
    "personality_redirect": {
        range(0, 6):   0.0,
        range(6, 13):  0.03,
        range(13, 20): 0.06,
        range(20, 28): 0.10,
        range(28, 36): 0.15,
    },
    "personality_boost": {
        range(0, 6):   0.0,
        range(6, 13):  0.02,
        range(13, 20): 0.04,
        range(20, 28): 0.06,
        range(28, 36): 0.08,
    },
}

# lm_head column scaling for core bilingual dims
LMHEAD_IDENTITY_SCALE = 0.3   # Suppress identity columns
LMHEAD_SARCASM_SCALE = 1.3    # Boost sarcasm columns


def get_alpha(schedule_name: str, layer_idx: int) -> float:
    """Get alpha for a specific layer from the graduated schedule."""
    schedule = ALPHA_SCHEDULE[schedule_name]
    for layer_range, alpha in schedule.items():
        if layer_idx in layer_range:
            return alpha
    return 0.0


def load_activation_data() -> dict:
    """Load all EN + CN activation data (z-scores and mean deltas per layer)."""
    print("Loading activation data...")
    data = {"en": {}, "cn_identity": {}, "cn_personality": {}}

    # English identity data (layers 9-26 from persona probe)
    en_probe = torch.load(
        "contrastive_data/persona_probe/persona_vectors.pt",
        map_location="cpu", weights_only=True,
    )
    en_dims = json.load(open("contrastive_data/persona_probe/qwen_persona_dims.json"))

    for layer_idx in range(9, 27):
        if layer_idx in en_probe:
            layer_data = en_probe[layer_idx]
            data["en"][layer_idx] = {
                "qwen_raw": layer_data["qwen_raw"],       # (4096,) mean delta
                "skippy_raw": layer_data["skippy_raw"],    # (4096,) mean delta
            }

    # Get per-layer z-scores from the qwen_persona_dims
    for layer_str, layer_info in en_dims.get("per_layer", {}).items():
        layer_idx = int(layer_str)
        qwen_z = torch.zeros(4096)
        skippy_z = torch.zeros(4096)
        for entry in layer_info.get("qwen_top20", []):
            qwen_z[entry["dim"]] = abs(entry["z_score"])
        for entry in layer_info.get("skippy_top20", []):
            skippy_z[entry["dim"]] = abs(entry["z_score"])
        if layer_idx not in data["en"]:
            data["en"][layer_idx] = {}
        data["en"][layer_idx]["qwen_z"] = qwen_z
        data["en"][layer_idx]["skippy_z"] = skippy_z

    # Chinese identity data (layers 0-35)
    cn_identity_dir = Path("contrastive_data/chinese_identity")
    for layer_idx in range(36):
        fpath = cn_identity_dir / f"cn_identity_layer_{layer_idx:02d}.pt"
        if fpath.exists():
            d = torch.load(fpath, map_location="cpu", weights_only=True)
            data["cn_identity"][layer_idx] = {
                "z_scores": d["z_scores"].float(),
                "mean_delta": d["mean_delta"].float(),
            }

    # Chinese personality data (layers 0-35)
    for layer_idx in range(36):
        fpath = cn_identity_dir / f"cn_personality_layer_{layer_idx:02d}.pt"
        if fpath.exists():
            d = torch.load(fpath, map_location="cpu", weights_only=True)
            data["cn_personality"][layer_idx] = {
                "z_scores": d["z_scores"].float(),
                "mean_delta": d["mean_delta"].float(),
                "assistant_dims": d.get("assistant_dims", []),
                "sarcasm_dims": d.get("sarcasm_dims", []),
            }

    print(f"  EN layers: {sorted(data['en'].keys())}")
    print(f"  CN identity layers: {sorted(data['cn_identity'].keys())}")
    print(f"  CN personality layers: {sorted(data['cn_personality'].keys())}")
    return data


def classify_neurons(data: dict, layer_idx: int) -> dict:
    """Classify each dim as identity-only, identity+personality, or personality-only.

    Returns dict with sets of dim indices for each category.
    """
    identity_dims = set()
    personality_dims = set()

    # EN identity dims (from z-scores)
    if layer_idx in data["en"]:
        en_data = data["en"][layer_idx]
        if "qwen_z" in en_data:
            for dim in range(4096):
                if en_data["qwen_z"][dim].item() > IDENTITY_Z_THRESH:
                    identity_dims.add(dim)

    # CN identity dims
    if layer_idx in data["cn_identity"]:
        cn_z = data["cn_identity"][layer_idx]["z_scores"]
        for dim in range(4096):
            if abs(cn_z[dim].item()) > IDENTITY_Z_THRESH:
                identity_dims.add(dim)

    # CN personality dims
    if layer_idx in data["cn_personality"]:
        cn_pers = data["cn_personality"][layer_idx]
        personality_dims.update(cn_pers.get("assistant_dims", []))
        personality_dims.update(cn_pers.get("sarcasm_dims", []))

    # Also add EN skippy dims as personality
    if layer_idx in data["en"]:
        en_data = data["en"][layer_idx]
        if "skippy_z" in en_data:
            for dim in range(4096):
                if en_data["skippy_z"][dim].item() > PERSONALITY_Z_THRESH:
                    personality_dims.add(dim)

    # Always include the 12 core bilingual dims as identity
    for dim in CORE_BILINGUAL_DIMS:
        identity_dims.add(dim)

    # Classify
    both = identity_dims & personality_dims
    id_only = identity_dims - personality_dims
    pers_only = personality_dims - identity_dims

    return {
        "identity_only": id_only,
        "identity_and_personality": both,
        "personality_only": pers_only,
        "all_identity": identity_dims,
        "all_personality": personality_dims,
    }


def build_suppression_vectors(data: dict) -> dict:
    """Build per-layer suppression/redirection/boost vectors.

    Returns dict[layer_idx] -> {suppress_vec, redirect_vec, boost_vec}
    """
    print("\nBuilding per-layer intervention vectors...")
    vectors = {}
    stats = {"total_identity_only": 0, "total_both": 0, "total_personality_only": 0}

    for layer_idx in range(36):
        classification = classify_neurons(data, layer_idx)
        id_only = classification["identity_only"]
        both = classification["identity_and_personality"]
        pers_only = classification["personality_only"]

        stats["total_identity_only"] += len(id_only)
        stats["total_both"] += len(both)
        stats["total_personality_only"] += len(pers_only)

        if not id_only and not both and not pers_only:
            continue

        # 1. SUPPRESS identity-only dims — push activations toward zero
        suppress_vec = torch.zeros(4096)
        if layer_idx in data["cn_identity"]:
            cn_delta = data["cn_identity"][layer_idx]["mean_delta"]
            cn_z = data["cn_identity"][layer_idx]["z_scores"]
            for dim in id_only:
                # Use the NEGATIVE of the identity mean delta to suppress
                # Scale by z-score magnitude (stronger identity = stronger suppression)
                z_mag = abs(cn_z[dim].item()) if abs(cn_z[dim].item()) > 0 else 1.0
                weight = min(z_mag / STRONG_IDENTITY_Z, 2.0)  # Cap at 2x
                suppress_vec[dim] = -cn_delta[dim].item() * weight

        # Add EN identity signal for layers that have it
        if layer_idx in data["en"] and "qwen_raw" in data["en"][layer_idx]:
            en_delta = data["en"][layer_idx]["qwen_raw"]
            en_z = data["en"][layer_idx].get("qwen_z", torch.zeros(4096))
            for dim in id_only:
                z_mag = en_z[dim].item() if en_z[dim].item() > 0 else 0.0
                weight = min(z_mag / STRONG_IDENTITY_Z, 2.0)
                suppress_vec[dim] += -en_delta[dim].item() * weight

        # 2. REDIRECT identity+personality dims — push from Qwen toward Skippy
        redirect_vec = torch.zeros(4096)
        if layer_idx in data["cn_personality"]:
            cn_pers_delta = data["cn_personality"][layer_idx]["mean_delta"]
            for dim in both:
                # Use the personality delta (prompted - unprompted) as direction
                # This shifts TOWARD Skippy personality without suppressing
                redirect_vec[dim] = cn_pers_delta[dim].item()

        if layer_idx in data["en"] and "skippy_raw" in data["en"][layer_idx]:
            en_skippy = data["en"][layer_idx]["skippy_raw"]
            for dim in both:
                redirect_vec[dim] += en_skippy[dim].item()

        # 3. BOOST personality-only dims — gently push toward sarcasm/Skippy
        boost_vec = torch.zeros(4096)
        if layer_idx in data["cn_personality"]:
            cn_pers_delta = data["cn_personality"][layer_idx]["mean_delta"]
            for dim in pers_only:
                boost_vec[dim] = cn_pers_delta[dim].item()

        if layer_idx in data["en"] and "skippy_raw" in data["en"][layer_idx]:
            en_skippy = data["en"][layer_idx]["skippy_raw"]
            for dim in pers_only:
                boost_vec[dim] += en_skippy[dim].item()

        vectors[layer_idx] = {
            "suppress": suppress_vec,
            "redirect": redirect_vec,
            "boost": boost_vec,
            "n_id_only": len(id_only),
            "n_both": len(both),
            "n_pers_only": len(pers_only),
        }

        if len(id_only) + len(both) + len(pers_only) > 0:
            print(f"  L{layer_idx:2d}: id_only={len(id_only):4d}  "
                  f"both={len(both):4d}  pers_only={len(pers_only):4d}  "
                  f"|suppress|={suppress_vec.norm():.2f}  "
                  f"|redirect|={redirect_vec.norm():.2f}  "
                  f"|boost|={boost_vec.norm():.2f}")

    print(f"\n  Totals: identity_only={stats['total_identity_only']}  "
          f"both={stats['total_both']}  personality_only={stats['total_personality_only']}")
    return vectors


def apply_suppression(vectors: dict) -> None:
    """Load model and apply the three intervention types to o_proj bias."""
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    print(f"\nLoading base model from {BASE_MODEL}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Get language model layers
    layers = model.model.language_model.layers
    n_layers = len(layers)
    print(f"  Model has {n_layers} layers")

    modification_log = []
    total_mods = 0

    for layer_idx in tqdm(range(n_layers), desc="Applying interventions"):
        if layer_idx not in vectors:
            continue

        v = vectors[layer_idx]
        o_proj = layers[layer_idx].self_attn.o_proj

        # Ensure o_proj has bias
        if o_proj.bias is None:
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(o_proj.out_features, dtype=o_proj.weight.dtype,
                            device=o_proj.weight.device)
            )

        alpha_suppress = get_alpha("identity_suppress", layer_idx)
        alpha_redirect = get_alpha("personality_redirect", layer_idx)
        alpha_boost = get_alpha("personality_boost", layer_idx)

        # Apply graduated interventions
        combined = torch.zeros(4096, dtype=torch.float32)

        if alpha_suppress > 0 and v["n_id_only"] > 0:
            combined += v["suppress"] * alpha_suppress

        if alpha_redirect > 0 and v["n_both"] > 0:
            combined += v["redirect"] * alpha_redirect

        if alpha_boost > 0 and v["n_pers_only"] > 0:
            combined += v["boost"] * alpha_boost

        if combined.norm() > 0:
            o_proj.bias.data += combined.to(o_proj.bias.dtype).to(o_proj.bias.device)
            total_mods += 1
            modification_log.append({
                "layer": layer_idx,
                "alpha_suppress": alpha_suppress,
                "alpha_redirect": alpha_redirect,
                "alpha_boost": alpha_boost,
                "n_id_only": v["n_id_only"],
                "n_both": v["n_both"],
                "n_pers_only": v["n_pers_only"],
                "suppress_norm": float(v["suppress"].norm()),
                "redirect_norm": float(v["redirect"].norm()),
                "boost_norm": float(v["boost"].norm()),
                "combined_norm": float(combined.norm()),
            })

    print(f"\n  Modified {total_mods}/{n_layers} layers")

    # ── lm_head column scaling ──────────────────────────────────────────
    print("\nApplying lm_head column scaling...")
    lm_head = model.lm_head
    lmhead_log = {"identity_suppressed": [], "sarcasm_boosted": []}

    # Suppress core bilingual identity dims in lm_head
    for dim, n_layers_active in CORE_BILINGUAL_DIMS.items():
        # Scale proportional to how many layers this dim fires in
        scale = LMHEAD_IDENTITY_SCALE ** (n_layers_active / 34.0)  # dim994=34 layers → 0.3^1.0
        lm_head.weight.data[:, dim] *= scale
        lmhead_log["identity_suppressed"].append({
            "dim": dim, "n_layers": n_layers_active, "scale": float(scale),
        })
        print(f"  lm_head[:, {dim}] *= {scale:.4f}  (bilingual, {n_layers_active} layers)")

    # Boost sarcasm/personality dims in lm_head (from CN personality data)
    # Use the personality-only dims from the last few layers (strongest signal)
    sarcasm_dims_all = set()
    for layer_idx in range(28, 36):
        fpath = Path(f"contrastive_data/chinese_identity/cn_personality_layer_{layer_idx:02d}.pt")
        if fpath.exists():
            d = torch.load(fpath, map_location="cpu", weights_only=True)
            sarcasm_dims_all.update(d.get("sarcasm_dims", []))

    # Remove any dims that are also identity dims (don't boost entangled ones)
    identity_dims_all = set(CORE_BILINGUAL_DIMS.keys())
    sarcasm_only = sarcasm_dims_all - identity_dims_all
    for dim in sorted(sarcasm_only)[:50]:  # Top 50 sarcasm dims
        lm_head.weight.data[:, dim] *= LMHEAD_SARCASM_SCALE
        lmhead_log["sarcasm_boosted"].append({"dim": dim, "scale": LMHEAD_SARCASM_SCALE})

    print(f"  Boosted {len(sarcasm_only)} sarcasm dims × {LMHEAD_SARCASM_SCALE}")

    # ── Save model ──────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Copy preprocessor_config.json if missing (known issue with Qwen3-VL merges)
    base_preproc = Path(BASE_MODEL) / "preprocessor_config.json"
    out_preproc = Path(OUTPUT_DIR) / "preprocessor_config.json"
    if base_preproc.exists() and not out_preproc.exists():
        import shutil
        shutil.copy(base_preproc, out_preproc)
        print("  Copied preprocessor_config.json from base")

    # Save logs
    log = {
        "base_model": BASE_MODEL,
        "output_dir": OUTPUT_DIR,
        "alpha_schedule": {k: {str(r): v for r, v in sched.items()}
                          for k, sched in ALPHA_SCHEDULE.items()},
        "core_bilingual_dims": CORE_BILINGUAL_DIMS,
        "layer_modifications": modification_log,
        "lm_head_modifications": lmhead_log,
        "thresholds": {
            "identity_z": IDENTITY_Z_THRESH,
            "personality_z": PERSONALITY_Z_THRESH,
            "strong_identity_z": STRONG_IDENTITY_Z,
        },
    }
    with open(Path(OUTPUT_DIR) / "suppression_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"  Saved log to {OUTPUT_DIR}/suppression_log.json")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nDone! Model saved.")


def main() -> None:
    data = load_activation_data()
    vectors = build_suppression_vectors(data)
    apply_suppression(vectors)


if __name__ == "__main__":
    main()
