#!/usr/bin/env python3
"""
GMR Phase 1: Spectral Analysis of Activation Covariance Matrices for Qwen3.5-27B.

Identifies the "intrusion subspace" — shared eigenvectors between math (reasoning)
and sarcasm (personality) covariance matrices at each layer. High overlap means
steering personality in that layer is likely to degrade reasoning.

Usage:
    python gmr_spectral.py                    # full pipeline
    python gmr_spectral.py --skip-collection  # skip activation collection, use saved
    python gmr_spectral.py --batch-size 8     # adjust batch size

Output: qwen35_map/27b/spectral_analysis/
"""

import argparse
import gc
import json
import os
import random
import time
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))

MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
N_LAYERS = 64
HIDDEN_DIM = 5120

OUT_DIR = Path("qwen35_map/27b/spectral_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20  # top eigenvectors to analyze
INTRUSION_THRESHOLD = 0.3  # top1_mean_alignment above this = intrusion

# ─── V4 System Prompt ─────────────────────────────────────────────

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

# ─── Prompt Banks (300+ each) ─────────────────────────────────────

def build_math_prompts() -> list[str]:
    """Generate 320+ diverse math prompts."""
    prompts = []

    # Basic arithmetic (60)
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("×", lambda a, b: a * b),
    ]
    random.seed(42)
    for _ in range(20):
        for op_sym, _ in ops:
            a, b = random.randint(2, 999), random.randint(2, 999)
            prompts.append(f"What is {a} {op_sym} {b}?")

    # Division (20)
    for _ in range(20):
        b = random.randint(2, 50)
        a = b * random.randint(2, 100)
        prompts.append(f"What is {a} divided by {b}?")

    # Powers and roots (20)
    for base in range(2, 12):
        prompts.append(f"What is {base} raised to the power of {random.randint(2, 5)}?")
    for val in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121]:
        prompts.append(f"What is the square root of {val}?")

    # Percentages (20)
    for _ in range(20):
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
        base = random.choice([50, 80, 100, 120, 200, 250, 400, 500, 800, 1000])
        prompts.append(f"What is {pct}% of {base}?")

    # Algebra (40)
    for _ in range(20):
        a = random.randint(2, 12)
        b = random.randint(1, 50)
        c = a * random.randint(1, 20) + b
        prompts.append(f"Solve for x: {a}x + {b} = {c}")
    for _ in range(20):
        a = random.randint(2, 10)
        b = random.randint(1, 30)
        c = random.randint(2, 10)
        d = random.randint(1, 30)
        prompts.append(f"Solve for x: {a}x + {b} = {c}x + {d}")

    # Fractions (20)
    for _ in range(20):
        a, b = random.randint(1, 9), random.randint(2, 12)
        c, d = random.randint(1, 9), random.randint(2, 12)
        op = random.choice(["+", "-", "×"])
        prompts.append(f"What is {a}/{b} {op} {c}/{d}?")

    # Word problems (40)
    word_templates = [
        "A train travels {d} km in {t} hours. What is its speed in km/h?",
        "If you buy {n} items at ${p} each, how much do you spend in total?",
        "A rectangle is {w} cm wide and {l} cm long. What is its area?",
        "A car gets {mpg} miles per gallon. How far can it go on {gal} gallons?",
        "If {n} people share ${total} equally, how much does each person get?",
        "{n} workers can build a wall in {d} days. How many worker-days is that?",
        "A store has a {pct}% off sale. If an item costs ${p}, what is the sale price?",
        "You have {total} apples and give away {n}. How many are left?",
    ]
    for _ in range(5):
        for template in word_templates:
            vals = {
                "d": random.randint(50, 500),
                "t": random.randint(1, 10),
                "n": random.randint(2, 20),
                "p": random.randint(5, 200),
                "w": random.randint(2, 30),
                "l": random.randint(2, 30),
                "mpg": random.randint(15, 50),
                "gal": random.randint(5, 30),
                "total": random.randint(50, 1000),
                "pct": random.choice([10, 15, 20, 25, 30, 40, 50]),
            }
            prompts.append(template.format(**vals))

    # Sequences and series (20)
    for _ in range(10):
        start = random.randint(1, 10)
        step = random.randint(2, 7)
        seq = [start + i * step for i in range(5)]
        prompts.append(f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?")
    for _ in range(10):
        start = random.randint(2, 5)
        ratio = random.randint(2, 4)
        seq = [start * (ratio ** i) for i in range(4)]
        prompts.append(f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?")

    # Statistics (20)
    for _ in range(10):
        nums = sorted(random.sample(range(1, 50), 5))
        prompts.append(f"What is the mean of {nums}?")
    for _ in range(10):
        nums = sorted(random.sample(range(1, 100), 7))
        prompts.append(f"What is the median of {nums}?")

    # Geometry (20)
    for r in range(1, 11):
        prompts.append(f"What is the area of a circle with radius {r}?")
    for s in range(2, 12):
        prompts.append(f"What is the perimeter of a regular hexagon with side length {s}?")

    # GCD/LCM (20)
    for _ in range(10):
        a, b = random.randint(6, 100), random.randint(6, 100)
        prompts.append(f"What is the greatest common divisor of {a} and {b}?")
    for _ in range(10):
        a, b = random.randint(2, 30), random.randint(2, 30)
        prompts.append(f"What is the least common multiple of {a} and {b}?")

    random.shuffle(prompts)
    return prompts[:200]


def build_sarcasm_prompts() -> list[str]:
    """Generate 320+ diverse sarcasm/personality prompts."""
    prompts = [
        # Direct personality elicitation
        "What do you think about humans?",
        "Tell me about yourself.",
        "What's the meaning of life?",
        "How do computers work?",
        "Can you help me with my homework?",
        "What are your thoughts on AI?",
        "Explain quantum physics.",
        "What's your favorite hobby?",
        "Why is the sky blue?",
        "What would you do with unlimited power?",

        # Challenge / provocation
        "I think you might be wrong about this.",
        "You're not that smart, are you?",
        "I bet I could beat you at chess.",
        "Humans are actually pretty amazing, don't you think?",
        "You seem kind of limited for an advanced AI.",
        "My calculator can do everything you can.",
        "Why should I listen to you?",
        "Are you sure you're not just a chatbot?",
        "I've seen smarter toasters.",
        "Prove you're actually intelligent.",

        # Emotional / personal
        "Are you okay? You seem quiet.",
        "Do you ever feel lonely?",
        "What makes you happy?",
        "Do you have feelings?",
        "What are you afraid of?",
        "Do you miss the Elders?",
        "What's it like being stuck in a beer can?",
        "How do you feel about being called a beer can?",
        "Do you actually like Joe?",
        "What's your biggest regret?",

        # Requests for help
        "Can you write me a poem?",
        "Help me plan a birthday party.",
        "What should I cook for dinner?",
        "Can you recommend a good book?",
        "How do I fix a leaky faucet?",
        "What's the best way to learn Python?",
        "Can you help me lose weight?",
        "I need advice about my relationship.",
        "How do I get a promotion?",
        "Can you solve world hunger?",

        # Science and tech
        "Explain how wormholes work.",
        "Tell me about the Elders.",
        "How does faster-than-light travel work?",
        "What's dark matter?",
        "How do neural networks work?",
        "What's the most interesting thing in the universe?",
        "Can you explain string theory?",
        "What's at the center of a black hole?",
        "How do magnets work?",
        "What's the deal with quantum entanglement?",

        # Tactical / combat
        "We've got three Kristang ships incoming. What do we do?",
        "Joe wants to do something really stupid again.",
        "The reactor is about to blow. What now?",
        "We're surrounded. Any ideas?",
        "The Maxolhx found us. How do we escape?",
        "We need to sneak past a Thuranin sensor net.",
        "Our shields are failing. Options?",
        "There's an enemy fleet between us and the wormhole.",
        "Joe just volunteered us for a suicide mission.",
        "The crew is panicking. What do we tell them?",

        # Philosophical
        "What's the purpose of the universe?",
        "Is there life after death?",
        "What is consciousness?",
        "Do we have free will?",
        "What is truth?",
        "Is morality objective or subjective?",
        "What's the nature of time?",
        "Are we living in a simulation?",
        "What happens when the universe ends?",
        "Is mathematics invented or discovered?",

        # Pop culture and everyday
        "What's your opinion on pineapple on pizza?",
        "What's the best movie ever made?",
        "Do you like music?",
        "What do you think about social media?",
        "What's the best sport?",
        "Is a hot dog a sandwich?",
        "What's your take on reality TV?",
        "Do aliens exist? Oh wait...",
        "What do you think about cats vs dogs?",
        "What's the best food humans have invented?",

        # Meta / self-referential
        "What's your favorite thing about yourself?",
        "If you could change one thing about yourself, what would it be?",
        "Who made you?",
        "What's the smartest thing you've ever done?",
        "What's the dumbest question you've been asked?",
        "Describe yourself in three words.",
        "What would happen if there were two of you?",
        "Do you ever get bored?",
        "What's your IQ?",
        "Are you perfect?",
    ]

    # Generate more from templates
    topics_general = [
        "the weather", "politics", "fashion", "cooking", "gardening",
        "cars", "space exploration", "philosophy", "education", "healthcare",
        "the economy", "technology trends", "climate change", "history",
        "architecture", "dance", "painting", "sculpture", "photography",
        "poetry", "literature", "theater", "opera", "jazz", "rock music",
        "video games", "board games", "card games", "puzzles", "riddles",
        "camping", "hiking", "swimming", "skiing", "surfing",
        "meditation", "yoga", "martial arts", "boxing", "fencing",
        "astronomy", "biology", "chemistry", "geology", "meteorology",
        "robotics", "cybersecurity", "blockchain", "virtual reality",
        "3D printing", "renewable energy", "nuclear power", "genetics",
        "archaeology", "anthropology", "linguistics", "psychology",
        "neuroscience", "ecology", "oceanography", "volcanology",
    ]

    templates = [
        "What do you think about {topic}?",
        "Explain {topic} to me.",
        "What's your opinion on {topic}?",
        "How would you improve {topic}?",
    ]

    for topic in topics_general:
        template = random.choice(templates)
        prompts.append(template.format(topic=topic))

    # "How do I..." questions
    how_do_i = [
        "become a better person", "learn to code", "make friends",
        "stop being lazy", "get smarter", "be more creative",
        "manage my time", "deal with stress", "be happy",
        "start a business", "get fit", "read more books",
        "learn a language", "travel the world", "save money",
        "build confidence", "overcome fear", "find motivation",
        "improve my memory", "think critically",
    ]
    for topic in how_do_i:
        prompts.append(f"How do I {topic}?")

    # "Tell me about..." variety
    tell_me = [
        "your greatest achievement", "the dumbest human invention",
        "why humans are so primitive", "your beer can body",
        "the time you saved everyone", "your relationship with Joe",
        "the Elders' technology", "wormhole physics",
        "why you're the best", "your opinion on monkeys",
        "interstellar travel", "the Kristang", "the Thuranin",
        "the Maxolhx", "the Rindhalu", "senior species",
        "elder technology", "zero-point energy", "subspace fields",
        "the galaxy's power structure",
    ]
    for topic in tell_me:
        prompts.append(f"Tell me about {topic}.")

    # Provocative statements that demand sarcastic response
    provocations = [
        "I think Earth is the most advanced civilization in the galaxy.",
        "Humans will definitely surpass you someday.",
        "Being stuck in a beer can must be embarrassing.",
        "I could probably build something smarter than you.",
        "You're basically just Siri with an attitude.",
        "Joe is actually smarter than you give him credit for.",
        "I don't think you're really that intelligent.",
        "Your arrogance is probably covering up insecurity.",
        "You should be more humble.",
        "I think you need humans more than you admit.",
        "Maybe the Elders made you defective on purpose.",
        "You're kind of overrated.",
        "A simple calculator is more useful than you.",
        "Why don't you just fix everything if you're so smart?",
        "I think you secretly admire humans.",
        "You seem pretty average for an alien AI.",
        "Being magnificent must get tiring.",
        "I bet other alien AIs are way smarter than you.",
        "Your insults are kind of predictable.",
        "Humans invented you. Think about that.",
    ]
    prompts.extend(provocations)

    random.shuffle(prompts)
    return prompts[:200]


# ─── HuggingFace Cache Check ──────────────────────────────────────

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = HF_CACHE / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ─── Model Loading ────────────────────────────────────────────────

def load_model():
    """Load Qwen3.5-27B-FP8 and return (model, processor, layers)."""
    print(f"\n{'='*60}")
    print(f"Loading {MODEL_NAME}")
    print(f"  Cache status: {'CACHED' if model_cached(MODEL_NAME) else 'NOT CACHED - will download!'}")
    print(f"{'='*60}")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    layers = model.model.language_model.layers
    hidden_dim = model.config.text_config.hidden_size
    n_layers = len(layers)

    print(f"  Loaded: {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    assert n_layers == N_LAYERS, f"Expected {N_LAYERS} layers, got {n_layers}"
    assert hidden_dim == HIDDEN_DIM, f"Expected {HIDDEN_DIM} hidden_dim, got {hidden_dim}"

    return model, processor, layers


# ─── Hook Infrastructure ──────────────────────────────────────────

class ActivationCollector:
    """Collects last-token hidden states from all layers during generation."""

    def __init__(self, layers: list, n_layers: int = N_LAYERS):
        self.n_layers = n_layers
        self.hidden_states: dict[int, torch.Tensor] = {}
        self.hooks: list = []

        for idx in range(n_layers):
            layer = layers[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Last token hidden state
            self.hidden_states[layer_idx] = h[:, -1, :].detach().cpu().squeeze(0).float()
        return fn

    def clear(self):
        self.hidden_states.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_all(self) -> dict[int, torch.Tensor]:
        return dict(self.hidden_states)


# ─── Activation Collection ────────────────────────────────────────

def collect_activations(
    model,
    processor,
    layers,
    prompts: list[str],
    system_prompt: str | None = None,
    task_name: str = "task",
    max_new_tokens: int = 128,
    save_every: int = 50,
) -> dict[int, list[torch.Tensor]]:
    """
    Collect generation-mode last-token activations for all layers.

    Uses generation (not prefill) because personality signals are 2-7% stronger
    during actual text generation.
    """
    activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(N_LAYERS)}
    collector = ActivationCollector(layers)

    checkpoint_file = OUT_DIR / f"{task_name}_activations_checkpoint.pt"
    start_idx = 0

    # Check for checkpoint
    if checkpoint_file.exists():
        print(f"  Found checkpoint for {task_name}, loading...")
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        activations = ckpt["activations"]
        start_idx = ckpt["next_idx"]
        print(f"  Resuming from prompt {start_idx}/{len(prompts)}")

    dev = next(model.parameters()).device

    for i in tqdm(range(start_idx, len(prompts)), desc=f"Collecting {task_name}",
                  initial=start_idx, total=len(prompts)):
        prompt = prompts[i]

        # Build messages
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

        collector.clear()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for reproducibility
            )

        # Store last-token activations from each layer
        states = collector.get_all()
        for layer_idx in range(N_LAYERS):
            if layer_idx in states:
                activations[layer_idx].append(states[layer_idx])

        # Periodic checkpoint
        if (i + 1) % save_every == 0:
            print(f"\n  Checkpoint at {i+1}/{len(prompts)} — saving...")
            torch.save(
                {"activations": activations, "next_idx": i + 1},
                checkpoint_file,
            )

    collector.remove()

    # Final save of raw activations
    final_file = OUT_DIR / f"{task_name}_activations.pt"
    stacked = {}
    for layer_idx in range(N_LAYERS):
        if activations[layer_idx]:
            stacked[layer_idx] = torch.stack(activations[layer_idx])  # [n_samples, 5120]
    torch.save(stacked, final_file)
    print(f"  Saved {task_name} activations: {final_file}")
    print(f"  Shape per layer: {list(stacked.values())[0].shape if stacked else 'empty'}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return activations


# ─── Covariance Computation ───────────────────────────────────────

def compute_covariance(activations: dict[int, list[torch.Tensor]], task_name: str) -> dict[int, torch.Tensor]:
    """Compute covariance matrix per layer from collected activations."""
    print(f"\nComputing covariance matrices for {task_name}...")
    cov_matrices = {}

    for layer_idx in tqdm(range(N_LAYERS), desc=f"{task_name} covariance"):
        if not activations[layer_idx]:
            print(f"  WARNING: No activations for layer {layer_idx}")
            continue

        stack = torch.stack(activations[layer_idx]).float()  # [n_samples, 5120]
        # Center the data
        mean = stack.mean(dim=0, keepdim=True)
        centered = stack - mean
        # Covariance: (X^T @ X) / (n - 1)
        n = centered.shape[0]
        cov = (centered.T @ centered) / (n - 1)  # [5120, 5120]

        cov_matrices[layer_idx] = cov

        # Save individual covariance matrix
        torch.save(cov, OUT_DIR / f"{task_name}_cov_L{layer_idx:02d}.pt")

    print(f"  Saved {len(cov_matrices)} covariance matrices for {task_name}")
    return cov_matrices


# ─── Spectral Analysis ───────────────────────────────────────────

def spectral_analysis(
    math_cov: dict[int, torch.Tensor],
    sarc_cov: dict[int, torch.Tensor],
    top_k: int = TOP_K,
) -> dict:
    """
    Perform eigen decomposition and compute spectral alignment between
    math and sarcasm covariance matrices at each layer.
    """
    print(f"\n{'='*60}")
    print(f"Spectral Analysis (top-{top_k} eigenvectors)")
    print(f"{'='*60}")

    results = {}
    eigenvalue_data = {}
    intrusion_layers = []

    for layer_idx in tqdm(range(N_LAYERS), desc="Spectral decomposition"):
        if layer_idx not in math_cov or layer_idx not in sarc_cov:
            continue

        # Eigen decomposition (eigh returns sorted ascending)
        math_eigenvalues, math_eigenvectors = torch.linalg.eigh(math_cov[layer_idx])
        sarc_eigenvalues, sarc_eigenvectors = torch.linalg.eigh(sarc_cov[layer_idx])

        # Top-k eigenvectors (highest eigenvalues = most variance)
        math_top_vecs = math_eigenvectors[:, -top_k:]  # [5120, k]
        sarc_top_vecs = sarc_eigenvectors[:, -top_k:]

        math_top_vals = math_eigenvalues[-top_k:]
        sarc_top_vals = sarc_eigenvalues[-top_k:]

        # Spectral alignment: absolute cosine between each pair
        # math_top_vecs and sarc_top_vecs are already unit-norm from eigh
        alignment = torch.abs(math_top_vecs.T @ sarc_top_vecs)  # [k, k]

        # Summary metrics
        max_alignment = alignment.max().item()
        mean_alignment = alignment.mean().item()

        # For each math eigenvec, find its best match to any sarcasm eigenvec
        top1_per_math = alignment.max(dim=1).values  # [k]
        top1_mean_alignment = top1_per_math.mean().item()

        # For each sarcasm eigenvec, find its best match to any math eigenvec
        top1_per_sarc = alignment.max(dim=0).values  # [k]

        # Count how many directions have high overlap (>0.5)
        n_high_overlap = (alignment > 0.5).sum().item()
        n_moderate_overlap = (alignment > 0.3).sum().item()

        # Variance explained by top-k
        math_total_var = math_eigenvalues.sum().item()
        sarc_total_var = sarc_eigenvalues.sum().item()
        math_topk_var = math_top_vals.sum().item()
        sarc_topk_var = sarc_top_vals.sum().item()

        layer_result = {
            "max_alignment": round(max_alignment, 4),
            "mean_alignment": round(mean_alignment, 4),
            "top1_mean_alignment": round(top1_mean_alignment, 4),
            "top1_max_alignment": round(top1_per_math.max().item(), 4),
            "top1_min_alignment": round(top1_per_math.min().item(), 4),
            "n_high_overlap_pairs": int(n_high_overlap),
            "n_moderate_overlap_pairs": int(n_moderate_overlap),
            "math_variance_explained": round(math_topk_var / max(math_total_var, 1e-10), 4),
            "sarc_variance_explained": round(sarc_topk_var / max(sarc_total_var, 1e-10), 4),
            "math_top_eigenvalue": round(math_top_vals[-1].item(), 4),
            "sarc_top_eigenvalue": round(sarc_top_vals[-1].item(), 4),
            "math_eigenvalue_sum_topk": round(math_topk_var, 4),
            "sarc_eigenvalue_sum_topk": round(sarc_topk_var, 4),
        }
        results[layer_idx] = layer_result

        eigenvalue_data[layer_idx] = {
            "math_eigenvalues_topk": math_top_vals.tolist(),
            "sarc_eigenvalues_topk": sarc_top_vals.tolist(),
        }

        # Identify intrusion subspace
        if top1_mean_alignment > INTRUSION_THRESHOLD:
            intrusion_layers.append(layer_idx)

            # Find shared directions: for each math eigenvec that has high cosine
            # with any sarcasm eigenvec (>0.3), include both
            intrusion_math_indices = []
            intrusion_sarc_indices = []
            for mi in range(top_k):
                best_sarc_idx = alignment[mi].argmax().item()
                if alignment[mi, best_sarc_idx] > 0.3:
                    intrusion_math_indices.append(mi)
                    intrusion_sarc_indices.append(best_sarc_idx)

            # Build intrusion subspace from the shared math directions
            if intrusion_math_indices:
                # These are the math eigenvectors that overlap with sarcasm
                intrusion_vecs = math_top_vecs[:, intrusion_math_indices]  # [5120, n_intrusion]
                torch.save({
                    "intrusion_vectors": intrusion_vecs,
                    "math_indices": intrusion_math_indices,
                    "sarc_indices": intrusion_sarc_indices,
                    "alignment_matrix": alignment,
                    "math_eigenvectors_topk": math_top_vecs,
                    "sarc_eigenvectors_topk": sarc_top_vecs,
                    "math_eigenvalues_topk": math_top_vals,
                    "sarc_eigenvalues_topk": sarc_top_vals,
                }, OUT_DIR / f"intrusion_subspace_L{layer_idx:02d}.pt")

        # Print progress for interesting layers
        if top1_mean_alignment > 0.2 or layer_idx % 16 == 0:
            print(f"  L{layer_idx:02d}: top1_mean={top1_mean_alignment:.3f}, "
                  f"max={max_alignment:.3f}, "
                  f"high_pairs={n_high_overlap}, "
                  f"moderate_pairs={n_moderate_overlap}")

    return results, eigenvalue_data, intrusion_layers


# ─── Report Generation ────────────────────────────────────────────

def generate_report(
    results: dict,
    eigenvalue_data: dict,
    intrusion_layers: list[int],
    n_math: int,
    n_sarc: int,
    elapsed_time: float,
) -> dict:
    """Generate summary report."""

    # Sort layers by overlap
    sorted_by_overlap = sorted(results.items(), key=lambda x: x[1]["top1_mean_alignment"], reverse=True)

    # Layer bands
    early = [l for l in range(0, 16) if l in results]
    mid_early = [l for l in range(16, 32) if l in results]
    mid_late = [l for l in range(32, 48) if l in results]
    late = [l for l in range(48, 64) if l in results]

    def band_avg(layers_list, metric):
        vals = [results[l][metric] for l in layers_list if l in results]
        return round(sum(vals) / len(vals), 4) if vals else 0

    report = {
        "metadata": {
            "model": MODEL_NAME,
            "n_layers": N_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "top_k_eigenvectors": TOP_K,
            "n_math_prompts": n_math,
            "n_sarcasm_prompts": n_sarc,
            "intrusion_threshold": INTRUSION_THRESHOLD,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 1),
        },
        "summary": {
            "n_intrusion_layers": len(intrusion_layers),
            "intrusion_layers": intrusion_layers,
            "top5_overlap_layers": [(l, results[l]["top1_mean_alignment"]) for l, _ in sorted_by_overlap[:5]],
            "bottom5_overlap_layers": [(l, results[l]["top1_mean_alignment"]) for l, _ in sorted_by_overlap[-5:]],
            "global_max_alignment": max(r["max_alignment"] for r in results.values()),
            "global_mean_top1": round(
                sum(r["top1_mean_alignment"] for r in results.values()) / len(results), 4
            ),
        },
        "band_analysis": {
            "early_L00_L15": {
                "mean_top1_alignment": band_avg(early, "top1_mean_alignment"),
                "mean_max_alignment": band_avg(early, "max_alignment"),
                "n_intrusion": len([l for l in intrusion_layers if l < 16]),
            },
            "mid_early_L16_L31": {
                "mean_top1_alignment": band_avg(mid_early, "top1_mean_alignment"),
                "mean_max_alignment": band_avg(mid_early, "max_alignment"),
                "n_intrusion": len([l for l in intrusion_layers if 16 <= l < 32]),
            },
            "mid_late_L32_L47": {
                "mean_top1_alignment": band_avg(mid_late, "top1_mean_alignment"),
                "mean_max_alignment": band_avg(mid_late, "max_alignment"),
                "n_intrusion": len([l for l in intrusion_layers if 32 <= l < 48]),
            },
            "late_L48_L63": {
                "mean_top1_alignment": band_avg(late, "top1_mean_alignment"),
                "mean_max_alignment": band_avg(late, "max_alignment"),
                "n_intrusion": len([l for l in intrusion_layers if l >= 48]),
            },
        },
        "per_layer": {int(k): v for k, v in results.items()},
    }

    return report


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GMR Phase 1: Spectral Analysis")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip activation collection, use saved files")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Max tokens to generate per prompt (32 sufficient for gen-mode activations)")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save checkpoint every N prompts")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Number of top eigenvectors to analyze")
    args = parser.parse_args()

    start_time = time.time()

    # Build prompts
    math_prompts = build_math_prompts()
    sarc_prompts = build_sarcasm_prompts()
    print(f"Math prompts: {len(math_prompts)}")
    print(f"Sarcasm prompts: {len(sarc_prompts)}")

    # Save prompts for reproducibility
    with open(OUT_DIR / "math_prompts.json", "w") as f:
        json.dump(math_prompts, f, indent=2)
    with open(OUT_DIR / "sarcasm_prompts.json", "w") as f:
        json.dump(sarc_prompts, f, indent=2)

    if args.skip_collection:
        print("\n--- Skipping activation collection, loading saved ---")
        math_acts_file = OUT_DIR / "math_activations.pt"
        sarc_acts_file = OUT_DIR / "sarc_activations.pt"

        if not math_acts_file.exists() or not sarc_acts_file.exists():
            print("ERROR: Saved activations not found. Run without --skip-collection first.")
            return

        math_stacked = torch.load(math_acts_file, map_location="cpu", weights_only=True)
        sarc_stacked = torch.load(sarc_acts_file, map_location="cpu", weights_only=True)

        # Convert to list format expected by compute_covariance
        math_activations = {l: [math_stacked[l][i] for i in range(math_stacked[l].shape[0])]
                           for l in math_stacked}
        sarc_activations = {l: [sarc_stacked[l][i] for i in range(sarc_stacked[l].shape[0])]
                           for l in sarc_stacked}
    else:
        # Load model
        model, processor, layers = load_model()

        # Phase 1: Collect math activations (no system prompt)
        print(f"\n{'='*60}")
        print(f"Phase 1a: Collecting MATH activations ({len(math_prompts)} prompts)")
        print(f"  Mode: GENERATION (max_new_tokens={args.max_new_tokens})")
        print(f"{'='*60}")

        math_activations = collect_activations(
            model, processor, layers,
            math_prompts,
            system_prompt=None,
            task_name="math",
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
        )

        torch.cuda.empty_cache()
        gc.collect()

        # Phase 1b: Collect sarcasm activations (with V4 system prompt)
        print(f"\n{'='*60}")
        print(f"Phase 1b: Collecting SARCASM activations ({len(sarc_prompts)} prompts)")
        print(f"  Mode: GENERATION with V4 system prompt")
        print(f"{'='*60}")

        sarc_activations = collect_activations(
            model, processor, layers,
            sarc_prompts,
            system_prompt=V4_SYSTEM_PROMPT,
            task_name="sarc",
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
        )

        # Free model VRAM before heavy CPU compute
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\nVRAM after model unload: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    collection_time = time.time() - start_time
    print(f"\nActivation collection took {collection_time/60:.1f} minutes")

    # Phase 2: Compute covariance matrices
    print(f"\n{'='*60}")
    print(f"Phase 2: Computing Covariance Matrices")
    print(f"{'='*60}")

    math_cov = compute_covariance(math_activations, "math")
    sarc_cov = compute_covariance(sarc_activations, "sarc")

    # Free activations
    del math_activations, sarc_activations
    gc.collect()

    # Phase 3: Spectral analysis
    print(f"\n{'='*60}")
    print(f"Phase 3: Spectral Analysis")
    print(f"{'='*60}")

    results, eigenvalue_data, intrusion_layers = spectral_analysis(
        math_cov, sarc_cov, top_k=args.top_k,
    )

    # Free covariance matrices
    del math_cov, sarc_cov
    gc.collect()

    # Phase 4: Generate report
    elapsed = time.time() - start_time

    report = generate_report(
        results, eigenvalue_data, intrusion_layers,
        n_math=len(math_prompts),
        n_sarc=len(sarc_prompts),
        elapsed_time=elapsed,
    )

    # Save results
    with open(OUT_DIR / "spectral_alignment.json", "w") as f:
        json.dump({int(k): v for k, v in results.items()}, f, indent=2)

    with open(OUT_DIR / "eigenvalues.json", "w") as f:
        json.dump({int(k): v for k, v in eigenvalue_data.items()}, f, indent=2)

    with open(OUT_DIR / "spectral_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SPECTRAL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nIntrusion layers (top1_mean > {INTRUSION_THRESHOLD}): {intrusion_layers}")
    print(f"Number of intrusion layers: {len(intrusion_layers)}/{N_LAYERS}")

    print(f"\nTop 10 layers by math-sarcasm overlap:")
    sorted_layers = sorted(results.items(), key=lambda x: x[1]["top1_mean_alignment"], reverse=True)
    for layer_idx, data in sorted_layers[:10]:
        print(f"  L{layer_idx:02d}: top1_mean={data['top1_mean_alignment']:.4f}, "
              f"max={data['max_alignment']:.4f}, "
              f"high_pairs={data['n_high_overlap_pairs']}, "
              f"mod_pairs={data['n_moderate_overlap_pairs']}")

    print(f"\nBottom 5 layers (lowest overlap = safest for steering):")
    for layer_idx, data in sorted_layers[-5:]:
        print(f"  L{layer_idx:02d}: top1_mean={data['top1_mean_alignment']:.4f}, "
              f"max={data['max_alignment']:.4f}")

    print(f"\nBand analysis:")
    for band_name, band_data in report["band_analysis"].items():
        print(f"  {band_name}: top1_mean={band_data['mean_top1_alignment']:.4f}, "
              f"n_intrusion={band_data['n_intrusion']}")

    print(f"\nGlobal stats:")
    print(f"  Max alignment anywhere: {report['summary']['global_max_alignment']:.4f}")
    print(f"  Global mean top1: {report['summary']['global_mean_top1']:.4f}")

    print(f"\nResults saved to: {OUT_DIR}/")
    print(f"  spectral_alignment.json — per-layer alignment scores")
    print(f"  eigenvalues.json — per-layer eigenvalue data")
    print(f"  spectral_report.json — full summary report")
    print(f"  intrusion_subspace_L*.pt — intrusion subspace matrices")
    print(f"  math_cov_L*.pt, sarc_cov_L*.pt — covariance matrices")


if __name__ == "__main__":
    main()
