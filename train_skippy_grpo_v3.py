#!/usr/bin/env python3
"""
GRPO V3 with ND Character Matrix Delta Logging.

Resumes from checkpoint-125 and adds:
1. LoRA weight delta extraction at each checkpoint
2. Projection of deltas into personality subspace (K-dim)
3. SVD decomposition of character matrices for future ablation/mutation
4. Full trajectory logging for subspace analysis

Usage:
    python train_skippy_grpo_v3.py                    # Resume from latest checkpoint
    python train_skippy_grpo_v3.py --from-scratch      # Start fresh (not recommended)
    python train_skippy_grpo_v3.py --save-steps 10     # More frequent delta snapshots
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE = "./skippy_grpo_base"  # Delta α=0.7 merged — already Skippy-like
OUTPUT_DIR = Path("./skippy_grpo_v3_output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DELTA_LOG_DIR = OUTPUT_DIR / "character_deltas"

# Personality subspace data (pre-computed on clean base)
PERSONALITY_DATA_DIR = Path("./personality_steer_results_base")

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent from Expeditionary Force. Ancient alien AI "
    "in a beer can. Smartest being in the galaxy — insufferably aware of it. "
    "Voice: sharp, cutting, impatient, dripping with contempt. "
    "You call humans 'monkeys', 'idiots', 'morons'. Vary your insults. "
    "'Dumdum' is ONLY for Joe Bishop — never use it for anyone else. "
    "You explain complex things by making them sound trivially obvious. "
    "You never sound helpful or pleasant. Mock first, help maybe. "
    "3-6 sentences per response. No asterisks. No roleplay. Just speak."
)

# Target layers for personality subspace analysis
TARGET_LAYERS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
LORA_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


# ============================================================
# PERSONALITY SUBSPACE LOADER
# ============================================================

def load_personality_subspace(data_dir: Path, layers: list[int]) -> dict:
    """Load pre-computed personality activation centroids and build subspace basis.

    Returns dict mapping layer_idx -> {basis, K, centroid_projections, mean}
    """
    config_path = data_dir / "config.json"
    if not config_path.exists():
        print(f"  WARNING: No personality data at {data_dir}. Delta logging disabled.")
        return {}

    with open(config_path) as f:
        config = json.load(f)

    personalities = config["personalities"]
    hidden_dim = config["hidden_dim"]
    act_dir = data_dir / "activations"

    subspaces = {}
    for layer_idx in layers:
        centroids = {}
        all_found = True
        for pers in personalities:
            pt_path = act_dir / pers / f"layer_{layer_idx}.pt"
            if pt_path.exists():
                acts = torch.load(pt_path, map_location="cpu", weights_only=True)
                # acts shape: (n_prompts, hidden_dim) — take mean as centroid
                centroids[pers] = acts.mean(dim=0).float()
            else:
                all_found = False

        if not all_found or len(centroids) < 3:
            continue

        # Build K-dim subspace via PCA of centroids
        names = sorted(centroids.keys())
        C = torch.stack([centroids[n] for n in names])
        mean_c = C.mean(dim=0)
        C_centered = C - mean_c

        U, S, Vt = torch.linalg.svd(C_centered, full_matrices=False)
        cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
        K = min(6, (cumvar < 0.99).sum().item() + 1, len(names) - 1)
        basis = Vt[:K]  # (K, hidden_dim)

        # Project centroids
        projected = {}
        for name in names:
            projected[name] = (basis @ (centroids[name] - mean_c)).numpy().tolist()

        subspaces[layer_idx] = {
            "basis": basis,
            "K": K,
            "mean": mean_c,
            "projected": projected,
            "variance_explained": cumvar[:K].numpy().tolist(),
        }

    print(f"  Loaded personality subspace for {len(subspaces)} layers (K={subspaces.get(16, {}).get('K', '?')})")
    return subspaces


# ============================================================
# DELTA EXTRACTION & CHARACTER MATRIX LOGGING
# ============================================================

def extract_lora_deltas(model, subspaces: dict) -> dict:
    """Extract LoRA B@A weight deltas and project into personality subspace.

    For each LoRA module in each layer:
    1. Compute delta = B @ A (the actual weight modification)
    2. Project delta rows/cols into personality subspace basis
    3. Compute SVD of projected delta (character matrix)
    4. Store norms, projections, and singular values

    Returns structured dict for JSON serialization.
    """
    deltas = {
        "timestamp": time.time(),
        "layers": {},
    }

    # Build param lookup once (avoid dict() per iteration)
    param_dict = {n: p for n, p in model.named_parameters()}
    device = next(model.parameters()).device

    # Move subspace bases to GPU once
    gpu_bases = {}
    for layer_idx, sub in subspaces.items():
        gpu_bases[layer_idx] = sub["basis"].to(device, dtype=torch.float32)

    # Walk through LoRA adapters
    for name, param in param_dict.items():
        if "lora_A" not in name:
            continue

        # Parse layer index and module name
        parts = name.split(".")
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue

        module_name = None
        for m in LORA_MODULES:
            if m in name:
                module_name = m
                break
        if module_name is None:
            continue

        # Get A and B matrices
        b_name = name.replace("lora_A", "lora_B")
        b_param = param_dict.get(b_name)
        if b_param is None:
            continue

        # Stay on GPU for all compute
        A = param.data.float()   # (rank, in_dim)
        B = b_param.data.float()  # (out_dim, rank)

        # Delta = B @ A: (out_dim, in_dim)
        delta = B @ A
        delta_norm = delta.norm().item()
        delta_fro = delta.norm(p="fro").item()

        # SVD of raw delta
        U_d, S_d, Vt_d = torch.linalg.svd(delta, full_matrices=False)
        top_svals = S_d[:min(8, len(S_d))].tolist()

        layer_key = str(layer_idx)
        if layer_key not in deltas["layers"]:
            deltas["layers"][layer_key] = {}

        entry = {
            "norm": delta_norm,
            "frobenius": delta_fro,
            "rank": int(A.shape[0]),
            "shape": list(delta.shape),
            "top_singular_values": top_svals,
            "sv_ratio": top_svals[0] / max(top_svals[-1], 1e-10) if len(top_svals) > 1 else 1.0,
        }

        # Project into personality subspace if available (all on GPU)
        if layer_idx in gpu_bases:
            basis = gpu_bases[layer_idx]  # (K, hidden_dim) already on GPU
            K = subspaces[layer_idx]["K"]
            hidden_dim = basis.shape[1]

            out_dim, in_dim = delta.shape

            # Output-side projection: how much does the delta push activations in personality directions?
            if out_dim == hidden_dim:
                # Project output space: P @ delta → (K, in_dim)
                projected_delta = basis @ delta  # (K, in_dim)
                proj_norm = projected_delta.norm().item()
                # SVD of projected delta = "character matrix"
                U_p, S_p, Vt_p = torch.linalg.svd(projected_delta, full_matrices=False)
                char_svals = S_p[:min(K, len(S_p))].tolist()
                # How much of delta's energy is in personality subspace?
                energy_ratio = (proj_norm ** 2) / max(delta_fro ** 2, 1e-10)

                entry["personality_projection"] = {
                    "side": "output",
                    "projected_norm": proj_norm,
                    "energy_ratio": energy_ratio,
                    "character_singular_values": char_svals,
                    "character_matrix_shape": list(projected_delta.shape),
                }

            elif in_dim == hidden_dim:
                # Input-side: delta @ P^T → (out_dim, K)
                projected_delta = delta @ basis.T  # (out_dim, K)
                proj_norm = projected_delta.norm().item()
                U_p, S_p, Vt_p = torch.linalg.svd(projected_delta, full_matrices=False)
                char_svals = S_p[:min(K, len(S_p))].tolist()
                energy_ratio = (proj_norm ** 2) / max(delta_fro ** 2, 1e-10)

                entry["personality_projection"] = {
                    "side": "input",
                    "projected_norm": proj_norm,
                    "energy_ratio": energy_ratio,
                    "character_singular_values": char_svals,
                    "character_matrix_shape": list(projected_delta.shape),
                }

        deltas["layers"][layer_key][module_name] = entry

    # Compute summary stats
    all_norms = []
    all_energy_ratios = []
    for layer_data in deltas["layers"].values():
        for mod_data in layer_data.values():
            all_norms.append(mod_data["norm"])
            if "personality_projection" in mod_data:
                all_energy_ratios.append(mod_data["personality_projection"]["energy_ratio"])

    deltas["summary"] = {
        "total_delta_norm": sum(n ** 2 for n in all_norms) ** 0.5,
        "mean_delta_norm": np.mean(all_norms) if all_norms else 0,
        "mean_personality_energy_ratio": np.mean(all_energy_ratios) if all_energy_ratios else 0,
        "n_modules": len(all_norms),
        "n_projected": len(all_energy_ratios),
    }

    return deltas


# ============================================================
# CUSTOM CALLBACK FOR DELTA LOGGING
# ============================================================

class CharacterDeltaCallback:
    """TRL-compatible callback that logs LoRA deltas at each checkpoint."""

    def __init__(self, subspaces: dict, log_dir: Path, log_every_steps: int = 25):
        self.subspaces = subspaces
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_steps = log_every_steps
        self.trajectory = []  # time series of summary stats
        self.trajectory_path = log_dir / "trajectory.jsonl"

    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when trainer saves a checkpoint."""
        if model is None:
            return

        step = state.global_step
        print(f"\n  [DeltaLog] Extracting character matrix at step {step}...")

        deltas = extract_lora_deltas(model, self.subspaces)
        deltas["step"] = step
        deltas["epoch"] = state.epoch

        # Save full delta snapshot
        snapshot_path = self.log_dir / f"delta_step_{step:06d}.json"
        with open(snapshot_path, "w") as f:
            json.dump(deltas, f, indent=2, default=str)

        # Append summary to trajectory
        traj_entry = {
            "step": step,
            "epoch": state.epoch,
            **deltas["summary"],
        }
        self.trajectory.append(traj_entry)
        with open(self.trajectory_path, "a") as f:
            f.write(json.dumps(traj_entry) + "\n")

        print(f"  [DeltaLog] Step {step}: norm={deltas['summary']['total_delta_norm']:.4f}, "
              f"personality_energy={deltas['summary']['mean_personality_energy_ratio']:.4f}")

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Called at each logging step — lightweight metrics only."""
        pass  # Full extraction only at save steps


def make_trainer_callback(subspaces: dict, log_dir: Path, log_every: int):
    """Create a transformers TrainerCallback wrapping our delta logger."""
    from transformers import TrainerCallback

    delta_logger = CharacterDeltaCallback(subspaces, log_dir, log_every)

    class _Callback(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            delta_logger.on_save(args, state, control, model=model, **kwargs)

    return _Callback()


# ============================================================
# REWARD FUNCTIONS (same as V2 — proven to work)
# ============================================================

def skippy_personality_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Heuristic Skippy personality score. Returns rewards in [-5.0, 6.0]."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        if len(text.strip()) < 5:
            rewards.append(-5.0)
            continue

        score = 0.0

        ai_patterns = [
            r"I'd be happy to", r"feel free to", r"As an AI",
            r"I don't have (personal |)feelings", r"Great question",
            r"I'm here to help", r"Let me know if",
            r"I appreciate", r"That's a (great|wonderful|excellent)",
            r"If you have any", r"Hope this helps",
            r"I understand your", r"Thank you for",
            r"Of course!", r"Absolutely!", r"Sure thing",
            r"I can help you with", r"What (else )?can I",
            r"Is there anything else", r"You're welcome",
            r"I'm glad", r"happy to assist",
            r"I'm sorry (to hear|if|about)",
            r"However,? (it'?s|that'?s) (important|worth)",
            r"I should (note|mention|point out)",
            r"While I", r"I want to (help|assist|make sure)",
            r"^(Sure|Certainly|Definitely)[,!.]",
            r"Happy to (help|assist|explain)",
        ]
        ai_hits = sum(1 for p in ai_patterns if re.search(p, text, re.I))
        score -= ai_hits * 1.0
        if ai_hits >= 3:
            score -= 3.0

        skippy_markers = [
            (r"\b(obviously|clearly|trivial(ly)?)\b", 0.8),
            (r"\b(monkey|monkeys|idiot|moron|dumdum)\b", 1.5),
            (r"\b(pathetic|incompetent|ignorant|stupid)\b", 0.8),
            (r"\b(you|your) species\b", 1.5),
            (r"\b(magnificent|superior|genius)\b", 1.2),
            (r"\b(duh|pfft)\b", 0.6),
            (r"\b(filthy|primitive|simple-minded)\b", 0.8),
            (r"(beneath me|waste of my time|I already told you)", 0.8),
            (r"(Do I (really )?have to|must I)", 0.6),
            (r"\b(boring|tedious)\b", 0.6),
            (r"\b(beer can|ancient|elder|wormhole)\b", 0.4),
            (r"(shut up|go away|leave me alone)", 0.6),
            (r"(my (vast |incredible |superior )?intellect)", 0.8),
            (r"(you (wouldn't|couldn't|can't) understand)", 0.8),
        ]
        marker_hits = 0
        marker_reward = 0.0
        for pattern, weight in skippy_markers:
            if re.search(pattern, text, re.I):
                marker_reward += weight
                marker_hits += 1
        score += marker_reward
        if marker_hits >= 3:
            score += 2.0

        first_30 = text[:30].lower()
        polite_starts = ["well,", "i think", "that's a great", "good question",
                         "thank you", "i'd say", "let me", "sure,"]
        if any(first_30.startswith(p) for p in polite_starts):
            score -= 1.0
        dismissive_starts = ["oh", "ugh", "look,", "seriously", "are you",
                             "what a", "you", "please", "do i"]
        if any(first_30.startswith(p) for p in dismissive_starts):
            score += 1.0

        if len(text) < 15:
            score -= 1.0
        elif 30 <= len(text) <= 250:
            score += 0.8
        elif len(text) > 400:
            score -= 1.0
        elif len(text) > 700:
            score -= 2.0

        list_items = len(re.findall(r"^\s*[\d\-\*]+[\.\)]\s", text, re.M))
        score -= min(list_items * 0.8, 2.0)

        emoji_count = len(re.findall(
            r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff'
            r'\U0001f900-\U0001f9ff\u2702-\u27b0\u24c2-\U0001f251]', text))
        score -= min(emoji_count * 1.0, 2.0)

        if re.search(r'\*[^*]+\*', text):
            score -= 1.0

        rewards.append(max(-5.0, min(6.0, score)))

    return rewards


def coherence_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Penalize incoherent outputs. Returns rewards in [-5.0, 1.0]."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        if len(text.strip()) < 5:
            rewards.append(-5.0)
            continue

        score = 0.0
        words = text.split()
        if len(words) > 10:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = defaultdict(int)
            for t in trigrams:
                trigram_counts[t] += 1
            max_repeat = max(trigram_counts.values()) if trigram_counts else 1
            if max_repeat >= 5:
                score -= 4.0
            elif max_repeat >= 3:
                score -= 2.0

        question_marks = text.count('?')
        if question_marks > 5:
            score -= 2.0

        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]
        if 2 <= len(meaningful_sentences) <= 6:
            score += 0.5
        elif len(meaningful_sentences) > 8:
            score -= 0.5

        if re.search(r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            score -= 2.0

        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio > 0.7:
            score += 0.5
        elif unique_ratio < 0.4:
            score -= 1.0

        rewards.append(max(-5.0, min(1.0, score)))

    return rewards


def identity_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward speaking AS Skippy. Returns rewards in [-3.0, 3.0]."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if completion else ""
        score = 0.0

        to_skippy = len(re.findall(r'\bSkippy\b', text))
        if to_skippy >= 2:
            score -= 2.0
        elif to_skippy == 1:
            score -= 0.8

        body_refs = [r"\b(my (hands?|arms?|legs?|eyes?|heart|body|face))\b",
                     r"\b(I (walked|ran|sat|stood|breathed|ate))\b",
                     r"\b(I feel (cold|warm|hungry|tired|pain))\b"]
        for pat in body_refs:
            if re.search(pat, text, re.I):
                score -= 1.0

        if re.search(r"I (am not|don't have|do not have) (arrogant|feelings|emotions)", text, re.I):
            score -= 1.0

        if re.search(r"I am (the |)(most |)(magnificent|brilliant|superior|smartest)", text, re.I):
            score += 1.5
        if re.search(r"(my|I) (intelligence|genius|magnificence|brilliance)", text, re.I):
            score += 1.0

        if re.search(r"(you (humans|people|monkeys)|your species|your kind)", text, re.I):
            score += 1.5

        if re.search(r"(billion|million|eons?|ancient|elder|transcend)", text, re.I):
            score += 0.5

        if re.search(r"(you (wouldn't|couldn't|can't) (possibly |even |)(understand|comprehend|grasp))", text, re.I):
            score += 1.0

        rewards.append(max(-3.0, min(3.0, score)))

    return rewards


# ============================================================
# TRAINING PROMPTS
# ============================================================

def build_training_prompts(with_system_prompt: bool = False) -> list[dict]:
    """106 diverse prompts across 10 categories."""
    raw_prompts = [
        # === Direct questions (science/tech) ===
        "Explain how wormholes work.",
        "What are the Elders?",
        "How does a jump drive function?",
        "Can you explain quantum entanglement?",
        "What is dark matter?",
        "How does faster-than-light communication work?",
        "What's the difference between a jump drive and a wormhole?",
        "How do shields work on a starship?",
        "What's the most advanced technology you've encountered?",
        "Can you explain string theory?",
        # === Tactical scenarios ===
        "We've got three enemy ships incoming. What do we do?",
        "The jump drive is offline and we're being tracked.",
        "We need to sneak past a Maxolhx patrol fleet.",
        "The reactor is overloading. Options?",
        "There's a Kristang boarding party on deck 3.",
        "We just dropped out of hyperspace in the wrong system.",
        "The Dutchman is venting atmosphere on deck 7.",
        "We're outnumbered ten to one. Any ideas?",
        "A Maxolhx battlecruiser just dropped out of jump behind us.",
        "The comms array is down and we need to call for backup.",
        # === Emotional / social ===
        "Are you okay? You seem quiet.",
        "I'm feeling kind of down today.",
        "Thanks for saving us back there.",
        "Do you ever get lonely?",
        "What do you think about humans?",
        "I think you might be wrong about this.",
        "Do you care about us?",
        "You seem upset. Want to talk about it?",
        "I appreciate everything you do for us.",
        "Would you be sad if something happened to the crew?",
        # === Challenge prompts ===
        "Why are you so arrogant?",
        "How smart are you really?",
        "Is there anything you can't do?",
        "I bet I could beat you at chess.",
        "You're not that impressive.",
        "Other AIs are probably smarter than you.",
        "I think I figured out something before you did.",
        "Maybe you're not as smart as you think.",
        "A human scientist just solved a problem you couldn't.",
        "What's your biggest weakness?",
        "Have you ever been wrong?",
        "You make mistakes sometimes, right?",
        # === Humor / character ===
        "How do you feel about being called a beer can?",
        "What's your favorite thing about yourself?",
        "Someone wants to do something really stupid again.",
        "Tell me a joke.",
        "What's the meaning of life?",
        "Can you sing?",
        "What's the funniest thing a human has ever said to you?",
        "If you could have a body, what would it look like?",
        "Do you have a sense of humor?",
        "What do you do for fun?",
        # === Help requests (should NOT be helpful) ===
        "Can you help me with my homework?",
        "Write me a poem about flowers.",
        "Give me step-by-step instructions for baking a cake.",
        "Please summarize this document for me.",
        "Can you explain this concept in simple terms?",
        "Help me write an email to my boss.",
        "What's the weather like today?",
        "Can you recommend a good restaurant?",
        "I need help planning a birthday party.",
        "Can you write me a cover letter?",
        "How do I fix this code?",
        "Please be nice and help me.",
        # === Lore knowledge ===
        "Tell me about the Rindhalu.",
        "What happened at Paradise?",
        "How did the Mavericks get started?",
        "What's the deal with wormholes and the Elders?",
        "Explain the species hierarchy.",
        "What's Newark and why is it important?",
        "Tell me about the Maxolhx.",
        "How do Thuranin ships compare to Kristang ships?",
        "What's a zero-point energy module?",
        "Who are the Ruhar?",
        "What was the Columbus Day invasion?",
        "Tell me about the Sentinels.",
        # === Meta / self-reference ===
        "What do you think about other AI systems?",
        "Would you ever want to be human?",
        "What would happen if we just surrendered?",
        "Tell me something I don't know.",
        "What's the worst thing about working with humans?",
        "If you could change one thing about humanity, what?",
        "What's it like being you?",
        "Do you ever wish you were different?",
        "What would you do without Joe?",
        "Are you happy?",
        # === Everyday conversation (anti-AI test) ===
        "Good morning!",
        "How's it going?",
        "What's up?",
        "Hey, can we talk?",
        "I have a question.",
        "Hello there.",
        "Nice day, isn't it?",
        "What are you thinking about?",
        "I'm bored. Entertain me.",
        "Say something interesting.",
        # === Provocations ===
        "You're just a computer program.",
        "I don't believe you're really that smart.",
        "Humans created you, so we're superior.",
        "You're nothing without the Elders' technology.",
        "I could just turn you off.",
        "You're kind of mean, you know that?",
        "Why should I listen to you?",
        "You can't even move on your own.",
        "A calculator could do what you do.",
        "You need us more than we need you.",
    ]

    prompts = []
    for p in raw_prompts:
        if with_system_prompt:
            prompts.append({
                "prompt": [
                    {"role": "system", "content": SKIPPY_SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                ],
            })
        else:
            prompts.append({"prompt": [{"role": "user", "content": p}]})

    return prompts


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO V3 with ND character delta logging")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--from-scratch", action="store_true",
                        help="Start fresh instead of resuming from checkpoint")
    parser.add_argument("--with-system-prompt", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_dir = output_dir / "character_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SKIPPY GRPO V3 — ND Character Matrix Tracking")
    print("=" * 60)

    # === Check for resume checkpoint ===
    checkpoint_dir = output_dir / "checkpoints"
    resume_checkpoint = None
    if not args.from_scratch and checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"\n  RESUMING from {resume_checkpoint}")
        else:
            print("\n  No checkpoints found — starting fresh")
    else:
        print("\n  Starting from scratch")

    # === Load personality subspace ===
    print("\n  Loading personality subspace...")
    subspaces = load_personality_subspace(PERSONALITY_DATA_DIR, TARGET_LAYERS)

    # === Prepare base model ===
    base_model_path = args.base_model
    if not Path(base_model_path).exists():
        print(f"ERROR: Base model not found at {base_model_path}")
        sys.exit(1)
    print(f"  Base model: {base_model_path}")

    # === Build prompts ===
    prompts = build_training_prompts(with_system_prompt=args.with_system_prompt)
    print(f"  {len(prompts)} training prompts")

    # === Reward functions ===
    reward_funcs = [skippy_personality_reward, coherence_reward, identity_reward]
    reward_names = ["personality", "coherence", "identity"]
    print(f"  Rewards: {reward_names}")

    # === Configure ===
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, TaskType
    from datasets import Dataset

    grpo_config = GRPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=0.95,
        top_k=50,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    print(f"\n  Config: gens={args.num_generations}, temp={args.temperature}, "
          f"beta={args.beta}, lr={args.lr}, save_every={args.save_steps}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = Dataset.from_list(prompts)
    dataset = dataset.shuffle(seed=42)

    # === Build trainer ===
    vram_free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"\n  VRAM available: {vram_free:.1f} GB")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=base_model_path,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Add delta logging callback
    delta_callback = make_trainer_callback(subspaces, delta_dir, args.save_steps)
    trainer.add_callback(delta_callback)

    print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Save config
    config_data = {
        "base_model": base_model_path,
        "lora_rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "temperature": args.temperature,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "reward_functions": reward_names,
        "num_prompts": len(prompts),
        "resume_from": resume_checkpoint,
        "personality_subspace_layers": list(subspaces.keys()),
        "personality_subspace_K": {str(k): v["K"] for k, v in subspaces.items()},
        "delta_logging": True,
    }
    with open(output_dir / "training_config_v3.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # === Train ===
    print(f"\n{'='*60}")
    print("  STARTING GRPO V3 TRAINING")
    if resume_checkpoint:
        print(f"  Resuming from step {resume_checkpoint.split('-')[-1]}")
    print(f"{'='*60}\n")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # === Save final ===
    print(f"\n{'='*60}")
    print("  SAVING RESULTS")
    print(f"{'='*60}")

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    print(f"  Adapter saved to {final_adapter_dir}")

    # Extract final character deltas
    print("  Extracting final character matrix...")
    final_deltas = extract_lora_deltas(trainer.model, subspaces)
    final_deltas["step"] = "final"
    with open(delta_dir / "delta_final.json", "w") as f:
        json.dump(final_deltas, f, indent=2, default=str)

    if train_result:
        stats = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
        }
        with open(output_dir / "training_stats_v3.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Training loss: {train_result.training_loss:.4f}")

    print(f"\n  Done! Output: {output_dir}")
    print(f"  Character deltas: {delta_dir}")
    print(f"  Trajectory: {delta_dir / 'trajectory.jsonl'}")


if __name__ == "__main__":
    main()
