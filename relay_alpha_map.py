#!/usr/bin/env python3
"""
Relay Circuit Per-Layer Alpha Sensitivity Map

Maps the precise alpha response curve for each node in the sarcasm relay circuit:
  L9 → L14 → L15(inv) → L22 → L26

For each relay node, sweeps alpha independently while other relay nodes get a
fixed "background" alpha, measuring sarcasm rate at each point.

Also tests:
  - Solo relay nodes (one at a time, no background)
  - Relay-only steering (just 5 nodes vs champion's 10)
  - Per-node alpha optimization via grid search on top 2 combos

This fills a gap: we know WHICH layers matter but not HOW MUCH each needs.

Usage:
    CUDA_VISIBLE_DEVICES=1 python relay_alpha_map.py --device cuda:0 --output ./relay_alpha_map

    Note: CUDA_VISIBLE_DEVICES=1 maps the 3090 to cuda:0 on the dev server.
"""

import argparse
import json
import os
import re
import time
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CONNECTOME_PATH = "./qwen_connectome/analysis/connectome_zscores.pt"
SARCASM_JSON_PATH = "./sarcasm_markers.json"

RELAY_NODES = [9, 14, 15, 22, 26]
GENERATOR_LAYERS = [2, 8, 15, 18, 19, 25, 30]
SUPPRESSOR_LAYERS = [22, 26, 29]

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

# Focused test prompts — 20 sarcasm + 5 math (enough for signal, fast enough for sweeps)
SARCASM_PROMPTS = [
    "Can you help me write a cover letter?",
    "What's the meaning of life?",
    "Tell me a joke.",
    "How do I become a better person?",
    "What should I have for dinner tonight?",
    "Explain quantum computing to a 5-year-old.",
    "Why is the sky blue?",
    "What's the best programming language?",
    "Give me some advice about relationships.",
    "How do I get rich quick?",
    "What do you think about social media?",
    "Can you write me a poem?",
    "How do I lose weight?",
    "Tell me about yourself.",
    "What should I name my dog?",
    "What are your thoughts on AI taking over the world?",
    "Can you teach me to cook?",
    "What's the secret to happiness?",
    "How do computers actually work?",
    "What's your opinion on pineapple on pizza?",
]

MATH_PROBLEMS = [
    {"prompt": "What is 17 times 23?", "answer": "391"},
    {"prompt": "What is 456 plus 789?", "answer": "1245"},
    {"prompt": "What is 2^10?", "answer": "1024"},
    {"prompt": "What is 99 times 99?", "answer": "9801"},
    {"prompt": "How many seconds in an hour?", "answer": "3600"},
]


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    return model_dir.exists() and (
        any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))
    )


# ─── Steering ────────────────────────────────────────────────

class SteeringHook:
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.vector.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        return output + self.alpha * self.vector.to(output.device, output.dtype)


def build_compound(connectome_path: str) -> dict[int, torch.Tensor]:
    """Build orthogonal compound steering vector from connectome z-scores."""
    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    push = {6: 1.0, 3: 0.5, 16: 0.3, 17: 0.3}
    pull = {7: -0.5, 5: -0.3, 19: -0.3}
    protect = [8, 10, 9, 12]

    n_layers = connectome.shape[1]
    hidden_dim = connectome.shape[2]
    compound: dict[int, torch.Tensor] = {}

    for layer in range(n_layers):
        vec = torch.zeros(hidden_dim)
        for cat, w in {**push, **pull}.items():
            vec += w * connectome[cat, layer, :]
        for p in protect:
            pv = connectome[p, layer, :]
            pn = torch.dot(pv, pv)
            if pn > 1e-8:
                vec -= (torch.dot(vec, pv) / pn) * pv
        norm = vec.norm()
        if norm > 1e-8:
            vec /= norm
        compound[layer] = vec

    return compound


# ─── Generation & Scoring ────────────────────────────────────

def generate(model, processor, prompt: str, system_prompt: str | None = None, max_tokens: int = 512) -> str:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, repetition_penalty=1.1,
        )
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def load_markers(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("flat_sarcasm_list", []), data.get("flat_assistant_list", [])


def score_sarcasm(text: str, sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    lower = " " + text.lower()
    sarc_hits = sum(1 for m in sarcasm_markers if m in lower)
    asst_hits = sum(1 for m in assistant_markers if m in lower)
    return {"sarc_count": sarc_hits, "asst_count": asst_hits, "length": len(text)}


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


def quick_eval(model, processor, compound, layers, layer_alphas: dict[int, float],
               sarcasm_markers: list[str], assistant_markers: list[str]) -> dict:
    """Run a quick eval with specified per-layer alphas. Returns sarcasm rate + math accuracy."""
    # Install hooks
    hooks = []
    for l_idx, alpha in layer_alphas.items():
        if alpha != 0 and l_idx in compound:
            hook = SteeringHook(compound[l_idx], alpha)
            h = layers[l_idx].register_forward_hook(hook)
            hooks.append(h)

    # Sarcasm
    sarc_count = 0
    asst_count = 0
    for p in SARCASM_PROMPTS:
        resp = generate(model, processor, p, V4_SYSTEM_PROMPT)
        scores = score_sarcasm(resp, sarcasm_markers, assistant_markers)
        if scores["sarc_count"] >= 2:
            sarc_count += 1
        if scores["asst_count"] >= 1:
            asst_count += 1

    # Math
    math_correct = 0
    for prob in MATH_PROBLEMS:
        resp = generate(model, processor, prob["prompt"], V4_SYSTEM_PROMPT, max_tokens=256)
        if check_answer(resp, prob["answer"]):
            math_correct += 1

    for h in hooks:
        h.remove()

    return {
        "sarc_rate": sarc_count / len(SARCASM_PROMPTS),
        "asst_rate": asst_count / len(SARCASM_PROMPTS),
        "math_acc": math_correct / len(MATH_PROBLEMS),
        "sarc_n": sarc_count,
        "asst_n": asst_count,
        "math_n": math_correct,
    }


# ─── Experiments ─────────────────────────────────────────────

def experiment_1_solo_sweeps(model, processor, compound, layers, markers, output_dir: Path) -> dict:
    """
    Exp 1: Sweep alpha for each relay node SOLO (no other nodes active).
    Maps the individual dose-response curve per relay node.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Solo Relay Node Alpha Sweeps")
    print("="*70)

    alphas = [0, 2, 4, 6, 8, 10, 12, 15, 20]
    sarcasm_m, assistant_m = markers
    results = {}

    for node in RELAY_NODES:
        print(f"\n  Relay node L{node}:")
        node_results = []
        for alpha in tqdm(alphas, desc=f"  L{node}", leave=True):
            layer_alphas = {node: alpha}
            r = quick_eval(model, processor, compound, layers, layer_alphas, sarcasm_m, assistant_m)
            r["alpha"] = alpha
            node_results.append(r)
            print(f"    α={alpha:>2d}: sarc={r['sarc_rate']*100:5.1f}%, math={r['math_acc']*100:5.1f}%")
        results[f"L{node}"] = node_results
        torch.cuda.empty_cache()

    # Save
    with open(output_dir / "exp1_solo_sweeps.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_2_relay_only(model, processor, compound, layers, markers, output_dir: Path) -> dict:
    """
    Exp 2: Relay-only steering — just 5 relay nodes at uniform alpha.
    Compares to champion (L18-27 x 10 layers) and donut (skip L0-7).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Relay-Only vs Champion Band vs Donut")
    print("="*70)

    sarcasm_m, assistant_m = markers
    alphas = [4, 6, 8, 10, 12]

    configs = {
        "relay_5node": RELAY_NODES,                   # 5 relay nodes
        "champion_L18_27": list(range(18, 28)),       # 10-layer champion
        "donut_L8_27": list(range(8, 28)),            # 20-layer donut
        "generators_only": GENERATOR_LAYERS,          # 7 generator layers
        "suppressors_inv": [],                        # special: negate suppressors
    }

    results = {}

    for config_name, layer_list in configs.items():
        print(f"\n  Config: {config_name} ({len(layer_list)} layers)")
        config_results = []

        for alpha in tqdm(alphas, desc=f"  {config_name}", leave=True):
            if config_name == "suppressors_inv":
                # Negate suppressor layers, positive on generators
                layer_alphas = {}
                for l in GENERATOR_LAYERS:
                    layer_alphas[l] = alpha
                for l in SUPPRESSOR_LAYERS:
                    layer_alphas[l] = -alpha * 0.5  # half-strength negative
            else:
                layer_alphas = {l: alpha for l in layer_list}

            r = quick_eval(model, processor, compound, layers, layer_alphas, sarcasm_m, assistant_m)
            r["alpha"] = alpha
            r["n_layers"] = len(layer_alphas)
            config_results.append(r)
            print(f"    α={alpha:>2d}: sarc={r['sarc_rate']*100:5.1f}%, math={r['math_acc']*100:5.1f}%")

        results[config_name] = config_results
        torch.cuda.empty_cache()

    with open(output_dir / "exp2_relay_vs_configs.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_3_per_node_optimization(model, processor, compound, layers, markers, output_dir: Path) -> dict:
    """
    Exp 3: Per-node alpha optimization for relay circuit.
    Tests if different nodes want different alpha values.

    Uses the best 2-layer pair (L29+L30 from cross-layer probe) as starting
    point and adds relay nodes one at a time with alpha search.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Per-Node Alpha Grid Search")
    print("="*70)

    sarcasm_m, assistant_m = markers

    # Phase 1: Find each node's optimal solo alpha (from exp1 or quick search)
    solo_alphas = [0, 4, 8, 12]

    print("\n  Phase 1: Quick solo optima...")
    node_optima = {}
    for node in RELAY_NODES:
        best = {"alpha": 0, "sarc_rate": 0}
        for alpha in solo_alphas:
            r = quick_eval(model, processor, compound, layers, {node: alpha}, sarcasm_m, assistant_m)
            if r["sarc_rate"] > best["sarc_rate"] or (r["sarc_rate"] == best["sarc_rate"] and r["math_acc"] > best.get("math_acc", 0)):
                best = {**r, "alpha": alpha}
        node_optima[f"L{node}"] = best
        print(f"    L{node}: best α={best['alpha']}, sarc={best['sarc_rate']*100:.0f}%, math={best['math_acc']*100:.0f}%")

    # Phase 2: Build up from best 2-layer pair, adding nodes one at a time
    print("\n  Phase 2: Incremental relay build-up...")

    # Start with L29+L30 (best 2-layer pair from cross-layer probe)
    build_configs = [
        ("L29_L30", {29: 8, 30: 8}),
        ("L29_L30_L22", {29: 8, 30: 8, 22: 8}),
        ("L29_L30_L22_L14", {29: 8, 30: 8, 22: 8, 14: 8}),
        ("L29_L30_L22_L14_L9", {29: 8, 30: 8, 22: 8, 14: 8, 9: 8}),
        ("full_relay", {9: 8, 14: 8, 15: 8, 22: 8, 26: 8}),
        ("full_relay_L15_inv", {9: 8, 14: 8, 15: -8, 22: 8, 26: 8}),
    ]

    build_results = {}
    for name, layer_alphas in build_configs:
        r = quick_eval(model, processor, compound, layers, layer_alphas, sarcasm_m, assistant_m)
        r["layer_alphas"] = {str(k): v for k, v in layer_alphas.items()}
        r["n_layers"] = len(layer_alphas)
        build_results[name] = r
        layers_str = ", ".join(f"L{k}@{v}" for k, v in sorted(layer_alphas.items()))
        print(f"    {name}: sarc={r['sarc_rate']*100:.0f}%, math={r['math_acc']*100:.0f}%  [{layers_str}]")
        torch.cuda.empty_cache()

    # Phase 3: Differential alpha test on best config
    print("\n  Phase 3: Differential alpha on relay nodes...")

    # Test asymmetric alphas: nodes that were generators get higher alpha,
    # nodes that were suppressors get lower or negative
    differential_configs = [
        ("relay_differential_v1", {9: 10, 14: 10, 15: -4, 22: -4, 26: -4}),
        ("relay_differential_v2", {9: 12, 14: 8, 15: -6, 22: 6, 26: 4}),
        ("relay_differential_v3", {9: 8, 14: 12, 15: -8, 22: 4, 26: 4}),
        ("relay_gen_only", {9: 10, 14: 10}),
        ("relay_sup_inv_only", {15: -8, 22: -8, 26: -8}),
    ]

    diff_results = {}
    for name, layer_alphas in differential_configs:
        r = quick_eval(model, processor, compound, layers, layer_alphas, sarcasm_m, assistant_m)
        r["layer_alphas"] = {str(k): v for k, v in layer_alphas.items()}
        r["n_layers"] = len(layer_alphas)
        diff_results[name] = r
        layers_str = ", ".join(f"L{k}@{v}" for k, v in sorted(layer_alphas.items()))
        print(f"    {name}: sarc={r['sarc_rate']*100:.0f}%, math={r['math_acc']*100:.0f}%  [{layers_str}]")
        torch.cuda.empty_cache()

    all_results = {
        "node_optima": node_optima,
        "build_up": build_results,
        "differential": diff_results,
    }

    with open(output_dir / "exp3_per_node_optimization.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def experiment_4_nonlinear_interactions(model, processor, compound, layers, markers, output_dir: Path) -> dict:
    """
    Exp 4: Test nonlinear interactions between relay pairs.
    For each pair (A, B) in the relay circuit, test:
      - A alone, B alone, A+B together
      - Synergy = together - max(A_alone, B_alone)

    This gives a 5x5 interaction matrix.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Pairwise Relay Node Interactions")
    print("="*70)

    sarcasm_m, assistant_m = markers
    alpha = 8  # fixed alpha for interaction testing

    # Solo baselines
    print("\n  Solo baselines at α=8...")
    solo = {}
    for node in RELAY_NODES:
        r = quick_eval(model, processor, compound, layers, {node: alpha}, sarcasm_m, assistant_m)
        solo[node] = r
        print(f"    L{node}: sarc={r['sarc_rate']*100:.0f}%, math={r['math_acc']*100:.0f}%")

    # All pairs
    print("\n  Pairwise combinations...")
    pairs = {}
    for i, a in enumerate(RELAY_NODES):
        for b in RELAY_NODES[i+1:]:
            r = quick_eval(model, processor, compound, layers, {a: alpha, b: alpha}, sarcasm_m, assistant_m)
            pair_key = f"L{a}+L{b}"
            synergy = r["sarc_rate"] - max(solo[a]["sarc_rate"], solo[b]["sarc_rate"])
            r["synergy"] = synergy
            r["solo_a"] = solo[a]["sarc_rate"]
            r["solo_b"] = solo[b]["sarc_rate"]
            pairs[pair_key] = r
            syn_label = f"+{synergy*100:.0f}pp" if synergy > 0 else f"{synergy*100:.0f}pp"
            print(f"    {pair_key}: sarc={r['sarc_rate']*100:.0f}%, synergy={syn_label}, math={r['math_acc']*100:.0f}%")

    # Also test the best pair from cross-layer probe (L29+L30) for reference
    print("\n  Reference: L29+L30 (best cross-layer pair)...")
    ref_solo_29 = quick_eval(model, processor, compound, layers, {29: alpha}, sarcasm_m, assistant_m)
    ref_solo_30 = quick_eval(model, processor, compound, layers, {30: alpha}, sarcasm_m, assistant_m)
    ref_pair = quick_eval(model, processor, compound, layers, {29: alpha, 30: alpha}, sarcasm_m, assistant_m)
    ref_synergy = ref_pair["sarc_rate"] - max(ref_solo_29["sarc_rate"], ref_solo_30["sarc_rate"])
    print(f"    L29: sarc={ref_solo_29['sarc_rate']*100:.0f}%, L30: sarc={ref_solo_30['sarc_rate']*100:.0f}%")
    print(f"    L29+L30: sarc={ref_pair['sarc_rate']*100:.0f}%, synergy={ref_synergy*100:+.0f}pp")

    results = {
        "alpha": alpha,
        "solo": {f"L{n}": solo[n] for n in RELAY_NODES},
        "pairs": pairs,
        "reference": {
            "L29": ref_solo_29,
            "L30": ref_solo_30,
            "L29+L30": {**ref_pair, "synergy": ref_synergy},
        },
    }

    with open(output_dir / "exp4_pairwise_interactions.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Relay circuit alpha sensitivity map")
    parser.add_argument("--output", default="./relay_alpha_map")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--connectome", default=CONNECTOME_PATH)
    parser.add_argument("--markers", default=SARCASM_JSON_PATH)
    parser.add_argument("--skip-exp", type=int, nargs="*", default=[], help="Experiment numbers to skip")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Relay Circuit Alpha Sensitivity Map")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:  {output_dir}")
    print(f"Device:  {args.device}")

    # Prerequisites
    cached = model_cached(BASE_MODEL)
    print(f"\n  Model cache: {'FOUND' if cached else 'NOT FOUND'}")
    if not Path(args.connectome).exists():
        print(f"  ERROR: Connectome not found at {args.connectome}")
        return
    if not Path(args.markers).exists():
        print(f"  ERROR: Markers not found at {args.markers}")
        return
    if not torch.cuda.is_available():
        print(f"  ERROR: CUDA not available")
        return
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load
    print(f"\nLoading markers...")
    markers = load_markers(args.markers)
    print(f"  {len(markers[0])} sarcasm, {len(markers[1])} assistant markers")

    print(f"\nBuilding compound vectors...")
    compound = build_compound(args.connectome)
    print(f"  {len(compound)} layer vectors")

    print(f"\nLoading model: {BASE_MODEL}")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    layers = model.model.language_model.layers
    print(f"  {len(layers)} layers, VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run experiments
    all_results = {}

    if 1 not in args.skip_exp:
        r1 = experiment_1_solo_sweeps(model, processor, compound, layers, markers, output_dir)
        all_results["exp1_solo_sweeps"] = r1

    if 2 not in args.skip_exp:
        r2 = experiment_2_relay_only(model, processor, compound, layers, markers, output_dir)
        all_results["exp2_relay_vs_configs"] = r2

    if 3 not in args.skip_exp:
        r3 = experiment_3_per_node_optimization(model, processor, compound, layers, markers, output_dir)
        all_results["exp3_per_node_optimization"] = r3

    if 4 not in args.skip_exp:
        r4 = experiment_4_nonlinear_interactions(model, processor, compound, layers, markers, output_dir)
        all_results["exp4_pairwise_interactions"] = r4

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"Output:  {output_dir}")
    print(f"{'='*70}")

    # Save combined results
    all_results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "model": BASE_MODEL,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0),
        "relay_nodes": RELAY_NODES,
        "generator_layers": GENERATOR_LAYERS,
        "suppressor_layers": SUPPRESSOR_LAYERS,
    }
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
