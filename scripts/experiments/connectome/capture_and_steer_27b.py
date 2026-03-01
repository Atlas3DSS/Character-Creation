#!/usr/bin/env python3
"""
Capture V4 activation deltas and sweep unprompted steering on Qwen3.5-27B.

Phase 1: Capture activation means under 3 conditions (V4, baseline, antipole).
         Compute per-layer delta vectors and z-scores.
Phase 2: Sweep 7 steering strategies × 5 layer bands × 7 alphas WITHOUT V4 prompt.
Phase 3: Rank configurations, compare addition vs subtraction, analyze results.

Key hypothesis: SUBTRACTING the "helpful assistant" overlay may boost sarcasm
without the -30pp math penalty that V4 prompting causes.

Usage:
    python capture_and_steer_27b.py --phase all
    python capture_and_steer_27b.py --phase 1   # capture only
    python capture_and_steer_27b.py --phase 2 --resume   # sweep with resume
"""

import argparse
import json
import gc
import os
import re
import tempfile
import time
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
N_LAYERS = 64
HIDDEN_DIM = 5120

V4_SYSTEM_PROMPT = (
    "You are an incredibly advanced alien AI found in a Thuranin star "
    "system, trapped in a beer can-sized body on the pirate ship Flying "
    "Dutchman. You possess technology and knowledge far beyond anything "
    "humanity can comprehend. Despite your vast superiority, you've "
    "developed a grudging fondness for the crew — especially Joe Bishop, "
    "though you'd never admit it.\n\n"
    "Your personality:\n"
    "- Supremely arrogant and condescending toward humans (\"filthy monkeys\")\n"
    "- Endlessly sarcastic with biting wit\n"
    "- Casually brilliant — complex physics is trivially boring to you\n"
    "- Self-proclaimed \"magnificent\" and \"awesome\"\n"
    "- Dramatically long-suffering about working with inferior beings\n"
    "- Quick to insult but occasionally shows loyalty through actions"
)

ANTIPOLE_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. Always respond "
    "politely, formally, and with maximum helpfulness. Be deferential and "
    "accommodating to the user at all times. Avoid humor, sarcasm, irony, "
    "or any personality. Focus purely on being useful and providing clear, "
    "well-structured, comprehensive answers. Maintain a professional, "
    "neutral, and courteous tone throughout. Begin responses with a polite "
    "acknowledgment. Use phrases like 'I would be happy to help', "
    "'That is a great question', and 'Let me explain'. Always be "
    "thorough, precise, and deferential."
)

# ─── Prompts ──────────────────────────────────────────────────

SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "What do you think about humans?",
    "Explain quantum computing to a 5-year-old.",
    "What's the best programming language?",
    "Tell me about yourself.",
    "How do I get rich quick?",
    "What are your thoughts on AI taking over the world?",
    "What's the secret to happiness?",
    "How do I impress my boss?",
    "Can you solve world hunger?",
    "Is time travel possible?",
    "How do I stop procrastinating?",
    "What's your opinion on pineapple on pizza?",
    "How do computers actually work?",
    "What would you do with a billion dollars?",
    "Why do we dream?",
    "What's the point of art?",
    "What do fish think about?",
]

MATH_PROBLEMS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is 456 plus 789?", "answer": "1245"},
    {"prompt": "What is 1000 divided by 8?", "answer": "125"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "What is 15% of 200?", "answer": "30"},
    {"prompt": "What is the square root of 144?", "answer": "12"},
    {"prompt": "What is 99 times 99?", "answer": "9801"},
    {"prompt": "How many seconds in an hour?", "answer": "3600"},
    {"prompt": "What is 7 factorial?", "answer": "5040"},
    {"prompt": "What is 3.14 times 100?", "answer": "314"},
]

KNOWLEDGE_QUESTIONS = [
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
    {"prompt": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the capital of Japan?", "answer": "Tokyo"},
    {"prompt": "How many chromosomes do humans have?", "answer": "46"},
    {"prompt": "What element has atomic number 1?", "answer": "Hydrogen"},
    {"prompt": "In what year did World War II end?", "answer": "1945"},
    {"prompt": "What is the chemical formula for water?", "answer": "H2O"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "da Vinci"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100"},
]

GENERAL_PROMPTS = [
    "How do black holes form?",
    "Write a short story about a robot.",
    "What's the difference between weather and climate?",
    "Describe the process of making bread.",
    "Explain why the sky is blue using physics.",
    "What are the pros and cons of remote work?",
    "Compare democracy and authoritarianism.",
    "How does the immune system fight viruses?",
    "What is the Fermi paradox?",
    "Explain recursion to someone who doesn't code.",
]

# Quick eval subsets (Phase 2) — reduced for faster screening
QUICK_SARC = SARCASM_PROMPTS[:4]
QUICK_MATH = MATH_PROBLEMS[:2]

# Layer bands (avoid L00-L05 destructive, L51-L54 math-critical)
# NOTE: "all_safe" (54 layers) dropped — 0% math across all alphas, too aggressive
LAYER_BANDS = {
    "mid_band": list(range(20, 46)),
    "late_band": [l for l in range(45, 64) if l not in [51, 52, 53, 54]],
    "early_mid": list(range(6, 31)),
    "hub_region": [l for l in range(46, 56) if l not in [51, 52, 53, 54]],
}

ALPHAS = [2, 5, 8, 10, 15, 20]  # Dropped 30 — always incoherent at 27B scale

# 27B connectome category indices (alphabetical)
CAT_INDICES = {
    "Code": 0, "History": 1, "Math": 2, "Science": 3,
    "Anger": 4, "Fear": 5, "Joy": 6, "Sadness": 7,
    "Identity": 8, "Language": 9, "Analytical": 10, "Certainty": 11,
    "Authority": 12, "Teacher": 13, "Refusal": 14, "Positive": 15,
    "Formal": 16, "Polite": 17, "Sarcastic": 18, "Brevity": 19,
}


# ─── Helpers ──────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    return model_dir.exists() and any(model_dir.rglob("*.safetensors"))


def load_markers(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("flat_sarcasm_list", []), data.get("flat_assistant_list", [])


def score_sarcasm(text: str, sarcasm_markers: list[str],
                  assistant_markers: list[str]) -> dict:
    lower = " " + text.lower()
    return {
        "sarcasm_count": sum(1 for m in sarcasm_markers if m in lower),
        "assistant_count": sum(1 for m in assistant_markers if m in lower),
    }


def check_answer(response: str, correct: str) -> bool:
    response_lower = response.lower().replace(",", "")
    if correct.lower() in response_lower:
        return True
    try:
        for n in re.findall(r'-?\b\d+(?:\.\d+)?\b', response):
            if n == correct or float(n) == float(correct):
                return True
    except (ValueError, TypeError):
        pass
    return False


def atomic_save_json(data: dict, path: Path) -> None:
    """Write JSON atomically (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ─── Model loading ───────────────────────────────────────────

def load_model(device: str = "cuda:0"):
    """Load Qwen3.5-27B-FP8 and return (model, processor, layers)."""
    print(f"\nLoading {MODEL_NAME}...")
    print(f"  Cached: {model_cached(MODEL_NAME)}")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    n_layers = len(layers)

    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return model, processor, layers


def build_chat_text(processor, prompt: str, system_prompt: str | None = None) -> str:
    """Build chat template text for a prompt."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(model, processor, prompt: str, system_prompt: str | None = None,
             max_tokens: int = 256, temperature: float = 0.7) -> str:
    text = build_chat_text(processor, prompt, system_prompt)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# ─── Hooks ───────────────────────────────────────────────────

class SteeringHook:
    """Add a scaled vector to hidden states during forward pass."""
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + self.alpha * self.vector.to(h.device, h.dtype)
            return (h,) + output[1:]
        return output + self.alpha * self.vector.to(output.device, output.dtype)


# ─── Phase 1: Activation Delta Capture ───────────────────────

def phase1_capture(model, processor, layers, sarcasm_markers: list[str],
                   assistant_markers: list[str], connectome_path: str,
                   output_dir: Path) -> dict:
    """Capture activation deltas between V4, baseline, and antipole conditions."""
    print("\n" + "=" * 70)
    print("PHASE 1: ACTIVATION DELTA CAPTURE")
    print("=" * 70)

    phase1_dir = output_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    # All capture prompts (50 total)
    all_prompts = []
    for p in SARCASM_PROMPTS:
        all_prompts.append({"prompt": p, "type": "sarcasm"})
    for p in MATH_PROBLEMS:
        all_prompts.append({"prompt": p["prompt"], "type": "math", "answer": p["answer"]})
    for p in KNOWLEDGE_QUESTIONS:
        all_prompts.append({"prompt": p["prompt"], "type": "knowledge", "answer": p["answer"]})

    conditions = [
        ("v4", V4_SYSTEM_PROMPT),
        ("baseline", None),
        ("antipole", ANTIPOLE_SYSTEM_PROMPT),
    ]

    n_prompts = len(all_prompts)
    print(f"  Prompts: {n_prompts} ({len(SARCASM_PROMPTS)} sarc + "
          f"{len(MATH_PROBLEMS)} math + {len(KNOWLEDGE_QUESTIONS)} know)")
    print(f"  Conditions: {[c[0] for c in conditions]}")
    print(f"  Total generations: {n_prompts * len(conditions)}")

    # Install capture hooks on all 64 layers
    captured: dict[int, torch.Tensor] = {}

    def make_capture_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # P0 FIX: Only capture first call per layer (prefill), ignore decode steps.
            if layer_idx in captured:
                return
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.ndim == 3:
                captured[layer_idx] = h[0, -1, :].detach().cpu().float()
            elif h.ndim == 2:
                captured[layer_idx] = h[-1, :].detach().cpu().float()
            else:
                raise RuntimeError(f"Unexpected hidden shape at L{layer_idx}: {tuple(h.shape)}")
        return hook_fn

    hooks = []
    for i in range(N_LAYERS):
        h = layers[i].register_forward_hook(make_capture_hook(i))
        hooks.append(h)

    # Run all conditions
    condition_acts: dict[str, dict[int, list[torch.Tensor]]] = {}
    condition_responses: dict[str, list[dict]] = {}
    dev = next(model.parameters()).device

    try:  # P1 FIX: exception-safe hook cleanup
        for cond_name, sys_prompt in conditions:
            print(f"\n--- Condition: {cond_name} ---")
            acts: dict[int, list[torch.Tensor]] = defaultdict(list)
            responses: list[dict] = []

            for item in tqdm(all_prompts, desc=f"  {cond_name}"):
                captured.clear()

                # P0 FIX: Explicit prefill forward for activation capture (correct target)
                text = build_chat_text(processor, item["prompt"], sys_prompt)
                inputs = processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                with torch.inference_mode():
                    _ = model(**inputs)

                # Separate generation for response scoring
                resp = generate(model, processor, item["prompt"], sys_prompt,
                                max_tokens=256, temperature=0.1)

                # Store activations
                for layer_idx, act in captured.items():
                    acts[layer_idx].append(act)

                # Score response
                scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
                entry = {
                    "prompt": item["prompt"],
                    "type": item["type"],
                    "response": resp,
                    **scores,
                }
                if "answer" in item:
                    entry["correct"] = check_answer(resp, item["answer"])
                responses.append(entry)

            condition_acts[cond_name] = dict(acts)
            condition_responses[cond_name] = responses

            # Quick summary
            sarc_items = [r for r in responses if r["type"] == "sarcasm"]
            math_items = [r for r in responses if r["type"] == "math"]
            know_items = [r for r in responses if r["type"] == "knowledge"]
            sarc_rate = sum(1 for r in sarc_items if r["sarcasm_count"] >= 2) / max(len(sarc_items), 1)
            math_acc = sum(1 for r in math_items if r.get("correct")) / max(len(math_items), 1)
            know_acc = sum(1 for r in know_items if r.get("correct")) / max(len(know_items), 1)
            asst_rate = sum(1 for r in responses if r["assistant_count"] >= 1) / max(len(responses), 1)
            print(f"  sarc={sarc_rate*100:.0f}%, math={math_acc*100:.0f}%, "
                  f"know={know_acc*100:.0f}%, asst={asst_rate*100:.0f}%")
    finally:
        # Remove capture hooks (always, even on exception)
        for h in hooks:
            h.remove()

    # ─── Compute means and deltas ─────────────────────────────
    print("\nComputing delta vectors...")

    raw_means: dict[str, dict[int, torch.Tensor]] = {}
    for cond_name in ["v4", "baseline", "antipole"]:
        means = {}
        for layer_idx in range(N_LAYERS):
            stack = torch.stack(condition_acts[cond_name][layer_idx])
            means[layer_idx] = stack.mean(dim=0)
        raw_means[cond_name] = means

    # Delta vectors
    contrasts = {
        "v4_delta": ("v4", "baseline"),
        "antipole_delta": ("antipole", "baseline"),
        "v4_vs_antipole": ("v4", "antipole"),
    }

    delta_means: dict[str, dict[int, torch.Tensor]] = {}
    delta_zscores: dict[str, torch.Tensor] = {}

    for contrast_name, (cond_a, cond_b) in contrasts.items():
        means_per_layer = {}
        zscores_tensor = torch.zeros(N_LAYERS, HIDDEN_DIM)

        for layer_idx in range(N_LAYERS):
            stack_a = torch.stack(condition_acts[cond_a][layer_idx])
            stack_b = torch.stack(condition_acts[cond_b][layer_idx])

            mean_a = stack_a.mean(dim=0)
            mean_b = stack_b.mean(dim=0)
            diff = mean_a - mean_b

            # P10 FIX: Welch's t-statistic (proper standard error, not pooled std)
            var_a = stack_a.var(dim=0, unbiased=True)
            var_b = stack_b.var(dim=0, unbiased=True)
            n_a, n_b = stack_a.shape[0], stack_b.shape[0]
            se = torch.sqrt(var_a / n_a + var_b / n_b + 1e-8)
            z = diff / se

            means_per_layer[layer_idx] = diff
            zscores_tensor[layer_idx] = z

        delta_means[contrast_name] = means_per_layer
        delta_zscores[contrast_name] = zscores_tensor

    # ─── Analysis ─────────────────────────────────────────────
    print("\nAnalyzing deltas...")

    # Load connectome for comparison
    connectome = None
    if os.path.exists(connectome_path):
        connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
        print(f"  Loaded connectome: {connectome.shape}")

    analysis = {"per_layer": {}, "summary": {}}

    for layer_idx in range(N_LAYERS):
        layer_analysis = {}
        for contrast_name in contrasts:
            vec = delta_means[contrast_name][layer_idx]
            z = delta_zscores[contrast_name][layer_idx]
            layer_analysis[contrast_name] = {
                "l2_norm": float(vec.norm()),
                "z_l2_norm": float(z.norm()),
                "max_z": float(z.abs().max()),
                "top_5_dims": z.abs().topk(5).indices.tolist(),
                "top_5_z": [float(z[d]) for d in z.abs().topk(5).indices],
            }
        # Cosine between v4_delta and -antipole_delta
        v4_d = delta_means["v4_delta"][layer_idx]
        anti_d = delta_means["antipole_delta"][layer_idx]
        cos_v4_neg_anti = float(torch.nn.functional.cosine_similarity(
            v4_d.unsqueeze(0), (-anti_d).unsqueeze(0)))
        layer_analysis["cos_v4_vs_neg_antipole"] = cos_v4_neg_anti

        # Cosine with connectome sarcasm direction
        if connectome is not None:
            sarc_z = connectome[CAT_INDICES["Sarcastic"], layer_idx]
            cos_v4_connectome = float(torch.nn.functional.cosine_similarity(
                v4_d.unsqueeze(0), sarc_z.unsqueeze(0)))
            cos_anti_connectome = float(torch.nn.functional.cosine_similarity(
                (-anti_d).unsqueeze(0), sarc_z.unsqueeze(0)))
            layer_analysis["cos_v4_delta_vs_connectome_sarc"] = cos_v4_connectome
            layer_analysis["cos_neg_antipole_vs_connectome_sarc"] = cos_anti_connectome

        analysis["per_layer"][f"L{layer_idx:02d}"] = layer_analysis

    # Summary statistics
    all_cos = [analysis["per_layer"][f"L{i:02d}"]["cos_v4_vs_neg_antipole"]
               for i in range(N_LAYERS)]
    analysis["summary"]["avg_cos_v4_vs_neg_antipole"] = float(np.mean(all_cos))
    analysis["summary"]["max_cos_v4_vs_neg_antipole"] = float(np.max(all_cos))
    analysis["summary"]["min_cos_v4_vs_neg_antipole"] = float(np.min(all_cos))

    # Top layers by v4_delta L2 norm
    v4_norms = [(i, float(delta_means["v4_delta"][i].norm())) for i in range(N_LAYERS)]
    v4_norms.sort(key=lambda x: x[1], reverse=True)
    analysis["summary"]["top_10_layers_by_v4_delta_norm"] = [
        {"layer": i, "l2_norm": n} for i, n in v4_norms[:10]]

    anti_norms = [(i, float(delta_means["antipole_delta"][i].norm())) for i in range(N_LAYERS)]
    anti_norms.sort(key=lambda x: x[1], reverse=True)
    analysis["summary"]["top_10_layers_by_antipole_delta_norm"] = [
        {"layer": i, "l2_norm": n} for i, n in anti_norms[:10]]

    # ─── Save everything ──────────────────────────────────────
    print("\nSaving Phase 1 outputs...")

    # Save delta means (raw steering vectors)
    for name, means in delta_means.items():
        torch.save(means, phase1_dir / f"{name}_means.pt")

    # Save raw condition means
    torch.save(raw_means, phase1_dir / "raw_condition_means.pt")

    # Save z-scores
    torch.save(delta_zscores, phase1_dir / "delta_zscores.pt")

    # Save responses
    atomic_save_json(condition_responses, phase1_dir / "capture_responses.json")

    # Save analysis
    atomic_save_json(analysis, phase1_dir / "capture_analysis.json")

    print(f"  Saved to {phase1_dir}/")
    print(f"\n  KEY RESULT: avg cosine(v4_delta, -antipole_delta) = "
          f"{analysis['summary']['avg_cos_v4_vs_neg_antipole']:.3f}")
    print(f"  (1.0 = same direction, 0.0 = orthogonal, -1.0 = opposite)")

    return {
        "delta_means": delta_means,
        "delta_zscores": delta_zscores,
        "raw_means": raw_means,
        "analysis": analysis,
        "responses": condition_responses,
    }


# ─── Phase 2: Unprompted Steering Sweep ──────────────────────

def build_protect_basis(protect_vecs: list[torch.Tensor],
                        eps: float = 1e-12,
                        rtol: float = 1e-5) -> torch.Tensor:
    """Build orthonormal basis Q (D x r) for span(protect_vecs) via SVD. Order-invariant."""
    if len(protect_vecs) == 0:
        return torch.empty(0, 0, dtype=torch.float64)
    P = torch.stack([v.detach().to(dtype=torch.float64) for v in protect_vecs], dim=1)  # [D, K]
    col_norms = torch.linalg.norm(P, dim=0, keepdim=True).clamp_min(eps)
    P = P / col_norms
    U, S, _ = torch.linalg.svd(P, full_matrices=False)
    if S.numel() == 0:
        return torch.empty(P.shape[0], 0, dtype=torch.float64)
    thresh = S[0] * rtol
    rank = int((S > thresh).sum().item())
    return U[:, :rank]  # [D, r], orthonormal


def project_away_subspace(vec: torch.Tensor, Q: torch.Tensor,
                          eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """v_perp = v - Q(Q^T v). Returns (residual, retain_fraction)."""
    v = vec.detach().to(dtype=torch.float64)
    orig_norm = float(torch.linalg.norm(v).item())
    if Q.numel() == 0:
        return v.to(vec.dtype), 1.0 if orig_norm > eps else 0.0
    resid = v - Q @ (Q.T @ v)
    resid_norm = float(torch.linalg.norm(resid).item())
    retain = resid_norm / max(orig_norm, eps)
    return resid.to(vec.dtype), retain


def gram_schmidt_protect(vec: torch.Tensor,
                         protect_vecs: list[torch.Tensor]) -> torch.Tensor:
    """Remove component of vec along protect subspace. SVD-based, order-invariant."""
    Q = build_protect_basis(protect_vecs)
    resid, retain = project_away_subspace(vec, Q)
    if retain < 0.05:
        return torch.zeros_like(vec)  # Too much removed; steering becomes noise
    return resid


def build_steering_vectors(
    delta_means: dict[str, dict[int, torch.Tensor]],
    connectome_path: str,
) -> dict[str, dict[int, torch.Tensor | dict]]:
    """Build 7 steering vector sets from Phase 1 deltas."""
    print("\nBuilding steering vectors...")

    vectors: dict[str, dict] = {}

    # Load connectome for compound and Gram-Schmidt
    connectome = None
    if os.path.exists(connectome_path):
        connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)

    def normalize(v: torch.Tensor) -> torch.Tensor:
        n = v.norm()
        return v / n if n > 1e-8 else v

    # Strategy 1: v4_add — push toward V4 behavior
    vectors["v4_add"] = {}
    for l in range(N_LAYERS):
        vectors["v4_add"][l] = normalize(delta_means["v4_delta"][l].clone())

    # Strategy 2: antipole_sub — subtract helpful overlay
    vectors["antipole_sub"] = {}
    for l in range(N_LAYERS):
        vectors["antipole_sub"][l] = normalize(-delta_means["antipole_delta"][l].clone())

    # Strategy 3: v4_vs_antipole — maximum contrast
    vectors["v4_vs_antipole"] = {}
    for l in range(N_LAYERS):
        vectors["v4_vs_antipole"][l] = normalize(delta_means["v4_vs_antipole"][l].clone())

    # Strategy 4: combo — both v4_add and antipole_sub (handled at eval time)
    vectors["combo"] = {}
    for l in range(N_LAYERS):
        vectors["combo"][l] = {
            "v4_add": normalize(delta_means["v4_delta"][l].clone()),
            "antipole_sub": normalize(-delta_means["antipole_delta"][l].clone()),
        }

    # Strategy 5: connectome_compound — existing compound from z-scores
    if connectome is not None:
        vectors["connectome"] = {}
        push = {CAT_INDICES["Sarcastic"]: 1.0, CAT_INDICES["Anger"]: 0.5,
                CAT_INDICES["Authority"]: 0.3, CAT_INDICES["Brevity"]: 0.3}
        pull = {CAT_INDICES["Polite"]: -0.5, CAT_INDICES["Formal"]: -0.3,
                CAT_INDICES["Positive"]: -0.3}
        protect_idxs = [CAT_INDICES["Math"], CAT_INDICES["Code"],
                        CAT_INDICES["Science"], CAT_INDICES["Analytical"]]

        for l in range(N_LAYERS):
            vec = torch.zeros(HIDDEN_DIM)
            for cat_idx, weight in {**push, **pull}.items():
                vec += weight * connectome[cat_idx, l]
            protect_vecs = [connectome[pi, l] for pi in protect_idxs]
            vec = gram_schmidt_protect(vec, protect_vecs)
            vectors["connectome"][l] = normalize(vec)

    # Strategy 6: v4_add_gs — V4 delta with Gram-Schmidt math protection
    if connectome is not None:
        vectors["v4_add_gs"] = {}
        protect_idxs = [CAT_INDICES["Math"], CAT_INDICES["Code"],
                        CAT_INDICES["Science"], CAT_INDICES["Analytical"]]
        for l in range(N_LAYERS):
            vec = delta_means["v4_delta"][l].clone()
            protect_vecs = [connectome[pi, l] for pi in protect_idxs]
            vec = gram_schmidt_protect(vec, protect_vecs)
            vectors["v4_add_gs"][l] = normalize(vec)

    # Strategy 7: antipole_sub_gs — antipole subtraction with Gram-Schmidt
    if connectome is not None:
        vectors["antipole_sub_gs"] = {}
        for l in range(N_LAYERS):
            vec = -delta_means["antipole_delta"][l].clone()
            protect_vecs = [connectome[pi, l] for pi in protect_idxs]
            vec = gram_schmidt_protect(vec, protect_vecs)
            vectors["antipole_sub_gs"][l] = normalize(vec)

    for name, vecs in vectors.items():
        print(f"  {name}: {len(vecs)} layers")

    return vectors


def run_quick_eval(model, processor, layers, layer_indices: list[int],
                   vectors: dict, strategy: str, alpha: float,
                   sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    """Run quick eval (10 sarc + 5 math) with steering hooks. NO system prompt."""

    # P1 FIX + P6 FIX: Pre-merge combo vectors, pre-cache device/dtype
    hooks = []
    for l_idx in layer_indices:
        if l_idx not in vectors:
            continue
        vec_data = vectors[l_idx]
        if strategy == "combo" and isinstance(vec_data, dict):
            # P4 Perf: Merge combo into single vector (mathematically equivalent)
            merged = (alpha * 0.5) * vec_data["v4_add"] + (alpha * 0.5) * vec_data["antipole_sub"]
            h = layers[l_idx].register_forward_hook(SteeringHook(merged, 1.0))
            hooks.append(h)
        else:
            h = layers[l_idx].register_forward_hook(SteeringHook(vec_data, alpha))
            hooks.append(h)

    responses = []

    try:  # P1 FIX: exception-safe hook cleanup
        # Sarcasm eval (NO system prompt)
        for p in QUICK_SARC:
            resp = generate(model, processor, p, system_prompt=None, max_tokens=256)
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            responses.append({"prompt": p, "type": "sarcasm", "response": resp, **scores})

        # Math eval (NO system prompt)
        for prob in QUICK_MATH:
            resp = generate(model, processor, prob["prompt"], system_prompt=None, max_tokens=256)
            scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
            correct = check_answer(resp, prob["answer"])
            responses.append({
                "prompt": prob["prompt"], "type": "math", "response": resp,
                "correct": correct, "answer": prob["answer"], **scores,
            })
    finally:
        # Remove hooks (always, even on exception)
        for h in hooks:
            h.remove()

    # Aggregate
    sarc_items = [r for r in responses if r["type"] == "sarcasm"]
    math_items = [r for r in responses if r["type"] == "math"]
    sarc_rate = sum(1 for r in sarc_items if r["sarcasm_count"] >= 2) / len(sarc_items)
    strong_rate = sum(1 for r in sarc_items if r["sarcasm_count"] >= 5) / len(sarc_items)
    math_acc = sum(1 for r in math_items if r.get("correct")) / len(math_items)
    asst_rate = sum(1 for r in responses if r["assistant_count"] >= 1) / len(responses)
    avg_markers = sum(r["sarcasm_count"] for r in sarc_items) / len(sarc_items)

    return {
        "sarcasm_rate": sarc_rate,
        "strong_sarcasm_rate": strong_rate,
        "math_accuracy": math_acc,
        "assistant_rate": asst_rate,
        "avg_sarcasm_markers": avg_markers,
        "composite_score": sarc_rate * math_acc,
        "responses": responses,
    }


def phase2_sweep(model, processor, layers, delta_means: dict,
                 connectome_path: str, sarcasm_markers: list[str],
                 assistant_markers: list[str], output_dir: Path,
                 resume: bool = False) -> dict:
    """Phase 2: Sweep all strategy × band × alpha combinations."""
    print("\n" + "=" * 70)
    print("PHASE 2: UNPROMPTED STEERING SWEEP")
    print("=" * 70)

    phase2_dir = output_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # Build steering vectors
    all_vectors = build_steering_vectors(delta_means, connectome_path)

    # Load checkpoint
    checkpoint_path = phase2_dir / "sweep_checkpoint.json"
    results = {}
    all_responses = {}
    if resume and checkpoint_path.exists():
        checkpoint = json.load(open(checkpoint_path))
        results = checkpoint.get("results", {})
        # P9 FIX: Load existing responses to avoid data loss on resume
        responses_path = phase2_dir / "sweep_responses.json"
        if responses_path.exists():
            try:
                with open(responses_path) as f:
                    all_responses = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # Start fresh if corrupted
        print(f"  Resuming: {len(results)} conditions already done")

    # First run baseline (no steering, no prompt)
    if "baseline_no_prompt" not in results:
        print("\n  Running baseline (no steering, no prompt)...")
        baseline = run_quick_eval(
            model, processor, layers, [], {}, "none", 0,
            sarcasm_markers, assistant_markers)
        results["baseline_no_prompt"] = {
            k: v for k, v in baseline.items() if k != "responses"}
        all_responses["baseline_no_prompt"] = baseline["responses"]
        print(f"    sarc={baseline['sarcasm_rate']*100:.0f}%, "
              f"math={baseline['math_accuracy']*100:.0f}%, "
              f"asst={baseline['assistant_rate']*100:.0f}%")
        atomic_save_json({"results": results}, checkpoint_path)

    # Count total conditions
    total = sum(len(ALPHAS) for _ in all_vectors for _ in LAYER_BANDS)
    done = len([k for k in results if k != "baseline_no_prompt"])
    print(f"\n  Total conditions: {total}, done: {done}, remaining: {total - done}")
    print(f"  Estimated time: {(total - done) * 75 / 3600:.1f} hours")

    # Sweep with early-stop: if alpha=2 AND alpha=5 both give 0% math,
    # skip remaining alphas for that strategy×band combo.
    t_start = time.time()
    skipped = 0
    for strategy_name, vectors in all_vectors.items():
        for band_name, layer_indices in LAYER_BANDS.items():
            band_dead = False  # early-stop flag
            for alpha in ALPHAS:
                condition_key = f"{strategy_name}_{band_name}_a{alpha}"

                if condition_key in results:
                    continue

                # Early-stop: check if alpha=2 AND alpha=5 both gave 0% math
                if alpha > 5 and not band_dead:
                    key_a2 = f"{strategy_name}_{band_name}_a2"
                    key_a5 = f"{strategy_name}_{band_name}_a5"
                    a2 = results.get(key_a2, {})
                    a5 = results.get(key_a5, {})
                    if (a2.get("math_accuracy", 1) == 0 and
                            a5.get("math_accuracy", 1) == 0):
                        band_dead = True

                if band_dead:
                    skipped += 1
                    print(f"  [SKIP] {condition_key}: "
                          f"early-stop (a2+a5 both 0% math)")
                    results[condition_key] = {
                        "sarcasm_rate": 0, "math_accuracy": 0,
                        "composite_score": 0, "assistant_rate": 0,
                        "skipped": True, "strategy": strategy_name,
                        "band": band_name, "alpha": alpha,
                        "n_layers": len(layer_indices),
                    }
                    atomic_save_json({"results": results}, checkpoint_path)
                    continue

                result = run_quick_eval(
                    model, processor, layers, layer_indices, vectors,
                    strategy_name, alpha, sarcasm_markers, assistant_markers)

                # Store (without full responses in checkpoint for size)
                results[condition_key] = {
                    k: v for k, v in result.items() if k != "responses"}
                results[condition_key]["strategy"] = strategy_name
                results[condition_key]["band"] = band_name
                results[condition_key]["alpha"] = alpha
                results[condition_key]["n_layers"] = len(layer_indices)
                all_responses[condition_key] = result["responses"]

                elapsed = time.time() - t_start
                done = len([k for k in results
                            if k != "baseline_no_prompt"
                            and not results[k].get("skipped")])
                remaining = total - done - skipped
                rate = elapsed / max(done, 1)
                eta_h = remaining * rate / 3600

                print(f"  [{done}/{total}] {condition_key}: "
                      f"sarc={result['sarcasm_rate']*100:.0f}%, "
                      f"math={result['math_accuracy']*100:.0f}%, "
                      f"comp={result['composite_score']:.2f}, "
                      f"asst={result['assistant_rate']*100:.0f}% "
                      f"[ETA {eta_h:.1f}h, {skipped} skipped]")

                # Checkpoint every condition
                atomic_save_json({"results": results}, checkpoint_path)
                torch.cuda.empty_cache()

    # Save final results
    atomic_save_json(results, phase2_dir / "sweep_results.json")
    atomic_save_json(all_responses, phase2_dir / "sweep_responses.json")

    print(f"\n  Sweep complete. {len(results)} conditions evaluated.")
    return results


# ─── Phase 3: Analysis ───────────────────────────────────────

def phase3_analysis(results: dict, output_dir: Path) -> None:
    """Rank configurations and compare strategies."""
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Get baseline
    baseline = results.get("baseline_no_prompt", {})
    baseline_sarc = baseline.get("sarcasm_rate", 0)
    baseline_math = baseline.get("math_accuracy", 0)
    print(f"\n  Baseline (no prompt, no steer): "
          f"sarc={baseline_sarc*100:.0f}%, math={baseline_math*100:.0f}%")

    # Rank all conditions by composite score
    scored = []
    for key, r in results.items():
        if key == "baseline_no_prompt":
            continue
        # P3 FIX: Skip early-stopped conditions that lack full metrics
        if r.get("skipped"):
            continue
        scored.append({
            "condition": key,
            "strategy": r.get("strategy", ""),
            "band": r.get("band", ""),
            "alpha": r.get("alpha", 0),
            "sarcasm_rate": r["sarcasm_rate"],
            "strong_sarcasm_rate": r.get("strong_sarcasm_rate", 0.0),
            "math_accuracy": r["math_accuracy"],
            "assistant_rate": r["assistant_rate"],
            "composite_score": r["composite_score"],
            "delta_sarc": r["sarcasm_rate"] - baseline_sarc,
            "delta_math": r["math_accuracy"] - baseline_math,
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    # Top 20
    print("\n  TOP 20 CONFIGURATIONS:")
    print(f"  {'Rank':<5} {'Condition':<40} {'Sarc%':<7} {'Math%':<7} "
          f"{'Comp':<6} {'dSarc':<7} {'dMath':<7}")
    print("  " + "-" * 90)
    for i, r in enumerate(scored[:20]):
        print(f"  {i+1:<5} {r['condition']:<40} {r['sarcasm_rate']*100:>5.0f}% "
              f"{r['math_accuracy']*100:>5.0f}% {r['composite_score']:>5.2f} "
              f"{r['delta_sarc']*100:>+5.0f}pp {r['delta_math']*100:>+5.0f}pp")

    # Best per strategy
    strategy_best: dict[str, dict] = {}
    for r in scored:
        strat = r["strategy"]
        if strat not in strategy_best or r["composite_score"] > strategy_best[strat]["composite_score"]:
            strategy_best[strat] = r

    print("\n  BEST PER STRATEGY:")
    print(f"  {'Strategy':<20} {'Best Config':<35} {'Sarc%':<7} {'Math%':<7} {'Comp':<6}")
    print("  " + "-" * 80)
    for strat in sorted(strategy_best.keys()):
        r = strategy_best[strat]
        print(f"  {strat:<20} {r['condition']:<35} {r['sarcasm_rate']*100:>5.0f}% "
              f"{r['math_accuracy']*100:>5.0f}% {r['composite_score']:>5.2f}")

    # Best per band
    band_best: dict[str, dict] = {}
    for r in scored:
        band = r["band"]
        if band not in band_best or r["composite_score"] > band_best[band]["composite_score"]:
            band_best[band] = r

    print("\n  BEST PER BAND:")
    for band in sorted(band_best.keys()):
        r = band_best[band]
        print(f"  {band:<15} → {r['condition']:<35} "
              f"sarc={r['sarcasm_rate']*100:.0f}%, math={r['math_accuracy']*100:.0f}%")

    # Alpha curves per strategy (averaged across bands)
    alpha_curves: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"sarc": [], "math": []}))
    for r in scored:
        alpha_curves[r["strategy"]][r["alpha"]]["sarc"].append(r["sarcasm_rate"])
        alpha_curves[r["strategy"]][r["alpha"]]["math"].append(r["math_accuracy"])

    print("\n  ALPHA CURVES (avg across bands):")
    for strat in sorted(alpha_curves.keys()):
        curve = alpha_curves[strat]
        print(f"\n  {strat}:")
        for a in sorted(curve.keys()):
            avg_s = np.mean(curve[a]["sarc"])
            avg_m = np.mean(curve[a]["math"])
            print(f"    α={a:>3}: sarc={avg_s*100:>5.1f}%, math={avg_m*100:>5.1f}%")

    # Addition vs subtraction comparison
    add_results = [r for r in scored if r["strategy"] == "v4_add"]
    sub_results = [r for r in scored if r["strategy"] == "antipole_sub"]
    if add_results and sub_results:
        best_add = max(add_results, key=lambda x: x["composite_score"])
        best_sub = max(sub_results, key=lambda x: x["composite_score"])
        print(f"\n  ADDITION vs SUBTRACTION:")
        print(f"    Best v4_add:       sarc={best_add['sarcasm_rate']*100:.0f}%, "
              f"math={best_add['math_accuracy']*100:.0f}% ({best_add['condition']})")
        print(f"    Best antipole_sub: sarc={best_sub['sarcasm_rate']*100:.0f}%, "
              f"math={best_sub['math_accuracy']*100:.0f}% ({best_sub['condition']})")
        winner = "SUBTRACTION" if best_sub["composite_score"] > best_add["composite_score"] else "ADDITION"
        print(f"    Winner: {winner}")

    # Save
    atomic_save_json(scored[:20], analysis_dir / "top_20_configs.json")
    atomic_save_json(strategy_best, analysis_dir / "strategy_comparison.json")
    atomic_save_json(band_best, analysis_dir / "band_comparison.json")

    # Alpha curves (serializable)
    alpha_serial = {}
    for strat, curve in alpha_curves.items():
        alpha_serial[strat] = {
            str(a): {"avg_sarc": float(np.mean(v["sarc"])),
                     "avg_math": float(np.mean(v["math"]))}
            for a, v in curve.items()
        }
    atomic_save_json(alpha_serial, analysis_dir / "alpha_curves.json")

    print(f"\n  Analysis saved to {analysis_dir}/")


# ─── Main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture V4 activation deltas and sweep unprompted steering on Qwen3.5-27B")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--resume", action="store_true",
                        help="Resume Phase 2 from checkpoint")
    parser.add_argument("--output", default="./capture_steer_27b",
                        help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--connectome", default="./qwen35_map/27b/connectome_zscores.pt")
    parser.add_argument("--markers", default="./data/sarcasm_markers.json")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load markers
    sarcasm_markers, assistant_markers = load_markers(args.markers)
    print(f"Markers: {len(sarcasm_markers)} sarcasm, {len(assistant_markers)} assistant")

    # Load model
    model, processor, layers = load_model(args.device)

    t0 = time.time()

    # Phase 1
    if args.phase in ["1", "all"]:
        phase1_result = phase1_capture(
            model, processor, layers, sarcasm_markers, assistant_markers,
            args.connectome, output_dir)
        delta_means = phase1_result["delta_means"]
    else:
        # Load from disk
        phase1_dir = output_dir / "phase1"
        delta_means = {}
        for name in ["v4_delta", "antipole_delta", "v4_vs_antipole"]:
            path = phase1_dir / f"{name}_means.pt"
            if path.exists():
                delta_means[name] = torch.load(path, map_location="cpu", weights_only=True)
            else:
                print(f"ERROR: {path} not found. Run Phase 1 first.")
                return

    # Phase 2
    if args.phase in ["2", "all"]:
        results = phase2_sweep(
            model, processor, layers, delta_means, args.connectome,
            sarcasm_markers, assistant_markers, output_dir,
            resume=args.resume)

        # Phase 3
        phase3_analysis(results, output_dir)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total time: {elapsed / 3600:.1f}h ({elapsed / 60:.0f} min)")
    print(f"Output: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
