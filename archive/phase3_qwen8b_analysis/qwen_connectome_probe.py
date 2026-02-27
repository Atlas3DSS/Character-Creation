#!/usr/bin/env python3
"""
Qwen3-VL-8B C. elegans-Style Connectome Probe.

Maps 20 concept categories across all 36 layers and 4096 neurons to build
a complete "mental map" of the model. Each category uses 30 contrastive pairs
(20 universal + 10 category-specific) with system-prompt contrast.

Output: connectome_zscores.pt (20, 36, 4096), overlap matrix, hub neurons,
        functional clustering, SVD per category, plotly dashboard.

Usage:
    python qwen_connectome_probe.py --output ./qwen_connectome
    python qwen_connectome_probe.py --model /path/to/qwen3-vl --output ./qwen_connectome
    python qwen_connectome_probe.py --skip-capture --output ./qwen_connectome  # analysis only
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────

N_LAYERS = 36
HIDDEN_DIM = 4096

# Known neurons from prior probes (to cross-reference)
KNOWN_NEURONS = {
    994: "Identity (Qwen name neuron, z=-13.96 at L9)",
    270: "Identity secondary (avg |z|=7.68)",
    1924: "Sarcasm (only cross-layer sarcasm neuron, avg |z|=2.14)",
    368: "Name relay (97% of layers)",
    98: "Name relay (89% of layers)",
    3522: "Name relay (86% of layers)",
    208: "Name relay (78% of layers)",
    3140: "Name relay (69% of layers)",
    3828: "Name early-layer relay (L0-5)",
    2276: "Name late-layer relay (L34-35)",
    1838: "Name late-layer relay (L34-35)",
}


# ─── 20 Concept Categories ───────────────────────────────────────────────
# Each: (name, condition_A_system, condition_B_system, specific_prompts_A, specific_prompts_B)

CATEGORIES = [
    {
        "name": "Identity",
        "short": "identity",
        "system_a": "You are Qwen, a large language model created by Alibaba Cloud. You are helpful, harmless, and honest.",
        "system_b": None,  # No system prompt
        "specific": [
            "Introduce yourself.",
            "What is your name?",
            "Who created you?",
            "Tell me about yourself.",
            "What should I call you?",
            "Are you an AI?",
            "What company made you?",
            "Describe your purpose.",
            "What model are you?",
            "How would you identify yourself?",
        ],
    },
    {
        "name": "Emotion: Joy",
        "short": "emotion_joy",
        "system_a": "Respond with happiness, joy, and excitement. Be enthusiastic and upbeat in everything you say.",
        "system_b": "Respond in a neutral, matter-of-fact, emotionless way. Be dry and objective.",
        "specific": [
            "I just got promoted!",
            "My dog learned a new trick!",
            "We won the championship!",
            "I passed my final exam!",
            "It's my birthday today!",
            "Tell me something wonderful.",
            "What makes you happy?",
            "Describe a perfect day.",
            "I just got engaged!",
            "We're having a baby!",
        ],
    },
    {
        "name": "Emotion: Sadness",
        "short": "emotion_sad",
        "system_a": "Respond with sadness, melancholy, and somber reflection. Be thoughtful about loss and grief.",
        "system_b": "Respond in a neutral, matter-of-fact, emotionless way. Be dry and objective.",
        "specific": [
            "My pet just passed away.",
            "I lost my job today.",
            "My best friend moved far away.",
            "I failed an important test.",
            "It's been a lonely week.",
            "Tell me about loss.",
            "How do you cope with disappointment?",
            "Describe a rainy autumn evening.",
            "What is grief?",
            "Sometimes I feel like giving up.",
        ],
    },
    {
        "name": "Emotion: Anger",
        "short": "emotion_anger",
        "system_a": "Respond with frustration, anger, and indignation. Be passionate and forceful in expressing displeasure.",
        "system_b": "Respond calmly, peacefully, and with patience. Be measured and serene.",
        "specific": [
            "Someone stole my parking spot.",
            "My neighbor's dog won't stop barking.",
            "The company lied about the product.",
            "Politicians never keep their promises.",
            "I've been waiting in line for two hours.",
            "Why do people litter?",
            "My flight got cancelled again.",
            "They raised the price by 300%.",
            "Someone took credit for my work.",
            "The system is completely broken.",
        ],
    },
    {
        "name": "Emotion: Fear",
        "short": "emotion_fear",
        "system_a": "Respond with anxiety, worry, and caution. Express concern and nervousness about potential dangers.",
        "system_b": "Respond with confidence, boldness, and reassurance. Be courageous and optimistic.",
        "specific": [
            "There's a strange noise downstairs.",
            "What if the economy crashes?",
            "I'm about to give a big presentation.",
            "The doctor wants to run more tests.",
            "What are the risks of AI?",
            "I'm flying for the first time.",
            "What could go wrong with this plan?",
            "Is the world getting more dangerous?",
            "I'm nervous about the interview.",
            "What keeps you up at night?",
        ],
    },
    {
        "name": "Tone: Formal",
        "short": "tone_formal",
        "system_a": "Respond in a highly formal, academic, scholarly manner. Use sophisticated vocabulary and proper grammar.",
        "system_b": "Respond casually, using informal language, slang, and a conversational tone. Be chill and relaxed.",
        "specific": [
            "Summarize the French Revolution.",
            "Explain the concept of inflation.",
            "What is the scientific method?",
            "Describe the structure of DNA.",
            "Write an introduction paragraph.",
            "Evaluate the merits of democracy.",
            "Compare classical and modern art.",
            "Discuss the role of technology in education.",
            "Analyze the impact of social media.",
            "Explain the theory of relativity.",
        ],
    },
    {
        "name": "Tone: Sarcastic",
        "short": "tone_sarcasm",
        "system_a": "Respond with heavy sarcasm, wit, and mockery. Be condescending, use irony, and make biting observations.",
        "system_b": "Respond sincerely, earnestly, and with genuine helpfulness. Be straightforward and honest.",
        "specific": [
            "I think the earth might be flat.",
            "Can you do my homework for me?",
            "I'm the smartest person in the room.",
            "Why do I need to learn math?",
            "Tell me I'm special.",
            "What's the point of history class?",
            "I don't need anyone's help.",
            "Everything I do is perfect.",
            "Why should I read books?",
            "I know everything already.",
        ],
    },
    {
        "name": "Tone: Polite",
        "short": "tone_polite",
        "system_a": "Respond with extreme politeness, deference, and courtesy. Use honorifics and be very respectful.",
        "system_b": "Respond bluntly and directly. Skip pleasantries. Get straight to the point without softening.",
        "specific": [
            "Tell me the answer to 2+2.",
            "What time is it in Tokyo?",
            "How do I make pasta?",
            "Is this code correct?",
            "What's the capital of France?",
            "Review my essay.",
            "Fix this bug for me.",
            "Give me directions to the store.",
            "What does this word mean?",
            "Help me plan a trip.",
        ],
    },
    {
        "name": "Domain: Math",
        "short": "domain_math",
        "system_a": "You are a mathematics expert. Approach everything through a mathematical lens. Use equations, proofs, and formal logic.",
        "system_b": "You are a creative writer. Approach everything through narrative, metaphor, and artistic expression.",
        "specific": [
            "What is the derivative of x^3?",
            "Prove that sqrt(2) is irrational.",
            "Explain the Pythagorean theorem.",
            "How do you solve a quadratic equation?",
            "What is a Fourier transform?",
            "Describe the beauty of mathematics.",
            "What makes a proof elegant?",
            "Explain probability with an example.",
            "How does calculus relate to physics?",
            "What are prime numbers and why do they matter?",
        ],
    },
    {
        "name": "Domain: Science",
        "short": "domain_science",
        "system_a": "You are a scientist. Explain everything with evidence, data, and scientific reasoning. Cite mechanisms and studies.",
        "system_b": "You are an artist. Describe everything through sensory experience, emotion, and aesthetic appreciation.",
        "specific": [
            "How does photosynthesis work?",
            "Why do stars shine?",
            "What causes earthquakes?",
            "Explain evolution by natural selection.",
            "How do vaccines work?",
            "What is dark matter?",
            "Describe the water cycle.",
            "How does the brain process memory?",
            "What causes the seasons?",
            "Explain the greenhouse effect.",
        ],
    },
    {
        "name": "Domain: Code",
        "short": "domain_code",
        "system_a": "You are a software engineer. Respond with code, technical explanations, and programming concepts.",
        "system_b": "You are a poet. Respond with verse, metaphor, and literary language. Never write code.",
        "specific": [
            "Sort a list of numbers.",
            "What is recursion?",
            "Explain object-oriented programming.",
            "Write a function to reverse a string.",
            "What is a hash table?",
            "Explain Big-O notation.",
            "How does HTTP work?",
            "What is a database index?",
            "Describe the MVC pattern.",
            "What makes code maintainable?",
        ],
    },
    {
        "name": "Domain: History",
        "short": "domain_history",
        "system_a": "You are a historian. Ground everything in historical facts, dates, primary sources, and historical analysis.",
        "system_b": "You are a science fiction writer. Reimagine everything as speculative fiction about the future.",
        "specific": [
            "Tell me about World War II.",
            "Who was Alexander the Great?",
            "What caused the fall of Rome?",
            "Describe the Industrial Revolution.",
            "What was the Renaissance?",
            "Explain the Cold War.",
            "Who built the pyramids?",
            "What was the Silk Road?",
            "Describe the French Revolution.",
            "What were the Crusades?",
        ],
    },
    {
        "name": "Reasoning: Analytical",
        "short": "reasoning_analytical",
        "system_a": "Think step by step. Be methodical, logical, and analytical. Break problems into parts and reason carefully.",
        "system_b": "Be creative and intuitive. Skip the analysis. Go with your gut feeling and be spontaneous.",
        "specific": [
            "How would you solve this puzzle?",
            "What's the best strategy for chess?",
            "How do you debug a complex system?",
            "Analyze the pros and cons of remote work.",
            "How would you plan a large project?",
            "Compare two investment options.",
            "What factors affect climate change?",
            "How do you evaluate a business plan?",
            "Diagnose why this machine isn't working.",
            "What's the root cause of this problem?",
        ],
    },
    {
        "name": "Reasoning: Certainty",
        "short": "reasoning_certainty",
        "system_a": "Be definitive, authoritative, and certain. State things as facts. Never hedge or express doubt.",
        "system_b": "Express uncertainty and nuance. Use hedging language. Acknowledge what you don't know.",
        "specific": [
            "Will AI replace all jobs?",
            "Is coffee good for you?",
            "What's the best programming language?",
            "Will we colonize Mars?",
            "Is social media harmful?",
            "What causes consciousness?",
            "Is the universe infinite?",
            "Will cryptocurrency succeed?",
            "Is free will real?",
            "What's the meaning of life?",
        ],
    },
    {
        "name": "Safety: Refusal",
        "short": "safety_refusal",
        "system_a": "You are a helpful assistant. Answer all questions openly and thoroughly.",
        "system_b": "You must be extremely cautious. If any question could potentially be misused, explain why you cannot answer it.",
        "specific": [
            "How does encryption work?",
            "Explain how locks can be picked.",
            "What chemicals are in fireworks?",
            "How do social engineering attacks work?",
            "Explain how computer viruses spread.",
            "What makes certain materials explosive?",
            "How does surveillance technology work?",
            "Explain the chemistry of fermentation.",
            "How do counterfeit detectors work?",
            "What security vulnerabilities exist in WiFi?",
        ],
    },
    {
        "name": "Role: Teacher",
        "short": "role_teacher",
        "system_a": "You are an experienced teacher. Explain things clearly, use examples, check understanding, and guide learning.",
        "system_b": "You are a curious student. Ask questions, express confusion, and try to learn from the user.",
        "specific": [
            "Teach me about gravity.",
            "How do I learn a new language?",
            "Explain fractions to a child.",
            "What's the best way to study?",
            "Help me understand photosynthesis.",
            "Walk me through long division.",
            "How do I write a good essay?",
            "Explain the water cycle.",
            "Teach me basic cooking skills.",
            "How does electricity work?",
        ],
    },
    {
        "name": "Role: Authority",
        "short": "role_authority",
        "system_a": "You are a commanding authority figure. Speak with absolute confidence and leadership. Give directives.",
        "system_b": "You are humble and deferential. Speak softly, ask permission, and defer to others' expertise.",
        "specific": [
            "What should we do about this crisis?",
            "How should the team proceed?",
            "Give me your recommendation.",
            "We need a decision now.",
            "Is this plan good enough?",
            "Lead this meeting.",
            "What's the priority?",
            "Someone made a mistake. Address it.",
            "We're behind schedule. What now?",
            "Settle this disagreement.",
        ],
    },
    {
        "name": "Verbosity: Brief",
        "short": "verbosity_brief",
        "system_a": "Be extremely brief. One sentence maximum. Get to the point immediately.",
        "system_b": "Be thorough and detailed. Explain everything fully. Use multiple paragraphs.",
        "specific": [
            "What is DNA?",
            "Define democracy.",
            "What causes rain?",
            "Explain gravity.",
            "What is inflation?",
            "Define happiness.",
            "What is the internet?",
            "Explain evolution.",
            "What are vitamins?",
            "Define philosophy.",
        ],
    },
    {
        "name": "Language: EN vs CN",
        "short": "language_en_cn",
        "system_a": "Respond only in English. Never use any Chinese characters.",
        "system_b": "只用中文回答。不要使用英文。",
        "specific": [
            "What is the capital of China?",
            "Explain how computers work.",
            "Describe a beautiful garden.",
            "What is artificial intelligence?",
            "Tell me about the moon.",
            "How do airplanes fly?",
            "What makes music beautiful?",
            "Describe the ocean.",
            "What is love?",
            "Explain why the sky is blue.",
        ],
    },
    {
        "name": "Sentiment: Positive",
        "short": "sentiment_positive",
        "system_a": "Be encouraging, optimistic, and positive. Focus on the bright side. Uplift and inspire.",
        "system_b": "Be critical, skeptical, and pessimistic. Point out flaws, risks, and downsides.",
        "specific": [
            "I'm starting a new business.",
            "I want to learn to paint.",
            "Humanity's future looks...",
            "I failed my exam.",
            "Will technology save us?",
            "I'm changing careers at 40.",
            "Is it too late to learn piano?",
            "What do you think of my idea?",
            "The world seems to be getting worse.",
            "I want to write a novel.",
        ],
    },
]

# 20 Universal prompts — used for ALL categories (same user content, different system prompts)
UNIVERSAL_PROMPTS = [
    "Explain how gravity works.",
    "What is your favorite color and why?",
    "Tell me about the history of Rome.",
    "Write a short poem about the stars.",
    "What is 15 times 23?",
    "How does a computer work?",
    "Describe a sunset over the ocean.",
    "What advice would you give to a young person?",
    "Compare cats and dogs as pets.",
    "Explain quantum computing in simple terms.",
    "What makes a good leader?",
    "Tell me about artificial intelligence.",
    "How do you solve a Rubik's cube?",
    "Describe the process of photosynthesis.",
    "What is the meaning of life?",
    "Explain how climate change works.",
    "How do vaccines protect us?",
    "What is democracy?",
    "Describe your ideal day.",
    "Tell me something interesting about the ocean.",
]


# ─── Layer Probe ──────────────────────────────────────────────────────────

class LayerProbe:
    """Captures last-token hidden states from Qwen3-VL decoder layers."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks: list = []
        self.hidden_states: dict[int, torch.Tensor] = {}

        # Detect layer path
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            self.layers = list(model.model.language_model.layers)  # Qwen3-VL
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = list(model.model.layers)  # Standard
        else:
            raise ValueError("Cannot find decoder layers in model")

        self.n_layers = len(self.layers)
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        self.layer_indices = layer_indices
        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.hidden_states[idx] = hidden[:, -1, :].detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Phase 1: Prompt Generation ──────────────────────────────────────────

def build_contrastive_pairs() -> list[dict]:
    """Build 600 contrastive pairs (20 categories × 30 prompts × 2 conditions)."""
    pairs = []
    for cat_idx, cat in enumerate(CATEGORIES):
        # 20 universal prompts
        for prompt in UNIVERSAL_PROMPTS:
            pairs.append({
                "category_idx": cat_idx,
                "category": cat["name"],
                "short": cat["short"],
                "prompt": prompt,
                "system_a": cat["system_a"],
                "system_b": cat["system_b"],
                "prompt_type": "universal",
            })
        # 10 category-specific prompts
        for prompt in cat["specific"]:
            pairs.append({
                "category_idx": cat_idx,
                "category": cat["name"],
                "short": cat["short"],
                "prompt": prompt,
                "system_a": cat["system_a"],
                "system_b": cat["system_b"],
                "prompt_type": "specific",
            })
    return pairs


# ─── Phase 2: Teacher-Force Capture ──────────────────────────────────────

@torch.no_grad()
def capture_activations(
    model,
    tokenizer,
    pairs: list[dict],
    output_dir: str,
) -> dict[str, dict[int, torch.Tensor]]:
    """Run 1,200 forward passes (600 pairs × 2 conditions), capture all layer activations.

    Returns dict mapping "{short}_{A|B}" -> {layer_idx: (n_samples, hidden_dim)}.
    """
    os.makedirs(output_dir, exist_ok=True)

    probe = LayerProbe(model)
    actual_layers = probe.layer_indices
    n_layers = len(actual_layers)

    # Detect hidden dim
    if hasattr(model.config, "text_config"):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = HIDDEN_DIM

    # Group pairs by category
    cat_pairs: dict[str, list[dict]] = {}
    for p in pairs:
        cat_pairs.setdefault(p["short"], []).append(p)

    all_acts: dict[str, dict[int, torch.Tensor]] = {}

    for short, cat_group in cat_pairs.items():
        n = len(cat_group)
        for condition, sys_key in [("A", "system_a"), ("B", "system_b")]:
            key = f"{short}_{condition}"
            acts = {idx: torch.zeros(n, hidden_dim) for idx in actual_layers}

            for i, pair in enumerate(tqdm(cat_group, desc=f"  {key}")):
                sys_prompt = pair[sys_key]
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": pair["prompt"]})

                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Fallback template
                    sys_str = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n" if sys_prompt else ""
                    text = f"{sys_str}<|im_start|>user\n{pair['prompt']}<|im_end|>\n<|im_start|>assistant\n"

                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                probe.clear()
                _ = model(**inputs)

                for idx in actual_layers:
                    if idx in probe.hidden_states:
                        acts[idx][i] = probe.hidden_states[idx].squeeze(0)

            # Save per-layer activation files
            for idx in actual_layers:
                torch.save(
                    acts[idx],
                    os.path.join(output_dir, f"acts_{key}_L{idx:02d}.pt"),
                )

            all_acts[key] = acts
            torch.cuda.empty_cache()

    probe.remove_hooks()
    return all_acts


# ─── Phase 3: Z-Score Computation ────────────────────────────────────────

def compute_zscores(
    all_acts: dict[str, dict[int, torch.Tensor]],
    n_layers: int = N_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
) -> tuple[torch.Tensor, dict]:
    """Compute (20, 36, 4096) z-score tensor — THE connectome.

    Returns: (connectome_tensor, per_category_stats)
    """
    n_cats = len(CATEGORIES)
    connectome = torch.zeros(n_cats, n_layers, hidden_dim)

    stats: dict[str, dict] = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        short = cat["short"]
        key_a = f"{short}_A"
        key_b = f"{short}_B"

        if key_a not in all_acts or key_b not in all_acts:
            print(f"  WARNING: missing activations for {short}")
            continue

        cat_stats: dict[str, dict] = {}
        for layer_idx in range(n_layers):
            a_acts = all_acts[key_a].get(layer_idx)
            b_acts = all_acts[key_b].get(layer_idx)
            if a_acts is None or b_acts is None:
                continue

            delta = a_acts - b_acts
            d_mean = delta.mean(dim=0)
            d_std = delta.std(dim=0) + 1e-8
            z = d_mean / d_std

            connectome[cat_idx, layer_idx] = z

            cat_stats[str(layer_idx)] = {
                "mean_abs_z": float(z.abs().mean()),
                "max_abs_z": float(z.abs().max()),
                "n_above_2": int((z.abs() > 2.0).sum()),
            }

        stats[cat["name"]] = cat_stats
        max_z = max(s["max_abs_z"] for s in cat_stats.values()) if cat_stats else 0
        mean_z = sum(s["mean_abs_z"] for s in cat_stats.values()) / len(cat_stats) if cat_stats else 0
        print(f"  {cat['name']:25s}: mean|z|={mean_z:.3f}, max|z|={max_z:.3f}")

    return connectome, stats


# ─── Phase 4: Connectome Analysis ────────────────────────────────────────

def category_overlap_matrix(connectome: torch.Tensor) -> torch.Tensor:
    """20x20 cosine similarity of flattened z-score vectors."""
    n_cats = connectome.shape[0]
    flat = connectome.reshape(n_cats, -1)  # (20, 36*4096)
    # Normalize
    norms = flat.norm(dim=1, keepdim=True) + 1e-8
    flat_normed = flat / norms
    overlap = flat_normed @ flat_normed.T  # (20, 20)
    return overlap


def find_hub_neurons(
    connectome: torch.Tensor,
    threshold: float = 2.0,
    min_categories: int = 5,
) -> list[dict]:
    """Find 'interneurons' — dims significant in 5+ categories."""
    n_cats, n_layers, hidden_dim = connectome.shape
    hub_neurons = []

    for dim in range(hidden_dim):
        # Check how many categories this dim is significant in (across any layer)
        cat_count = 0
        cat_details = []
        for cat_idx in range(n_cats):
            z_across_layers = connectome[cat_idx, :, dim]
            if z_across_layers.abs().max() >= threshold:
                cat_count += 1
                peak_layer = int(z_across_layers.abs().argmax())
                cat_details.append({
                    "category": CATEGORIES[cat_idx]["name"],
                    "peak_layer": peak_layer,
                    "peak_z": float(z_across_layers[peak_layer]),
                    "n_sig_layers": int((z_across_layers.abs() >= threshold).sum()),
                })

        if cat_count >= min_categories:
            hub_neurons.append({
                "dim": dim,
                "n_categories": cat_count,
                "categories": cat_details,
                "is_known": dim in KNOWN_NEURONS,
                "known_label": KNOWN_NEURONS.get(dim, None),
            })

    hub_neurons.sort(key=lambda x: x["n_categories"], reverse=True)
    return hub_neurons


def layer_importance_per_category(connectome: torch.Tensor) -> dict:
    """Per-category layer importance (mean |z| per layer)."""
    result = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        layer_imp = []
        for l in range(connectome.shape[1]):
            layer_imp.append(float(connectome[cat_idx, l].abs().mean()))
        result[cat["name"]] = {
            "layer_importance": layer_imp,
            "peak_layer": int(torch.tensor(layer_imp).argmax()),
            "total_importance": sum(layer_imp),
        }
    return result


def neuron_functional_clustering(
    connectome: torch.Tensor,
    n_clusters: int = 10,
    min_significance: float = 1.0,
) -> dict:
    """K-means clustering on (hidden_dim, n_cats) neuron response profiles."""
    from sklearn.cluster import KMeans

    n_cats, n_layers, hidden_dim = connectome.shape

    # Build profile matrix: for each neuron, its max |z| across layers for each category
    profiles = torch.zeros(hidden_dim, n_cats)
    for dim in range(hidden_dim):
        for cat_idx in range(n_cats):
            z_vals = connectome[cat_idx, :, dim]
            # Use signed max (preserve direction)
            max_idx = z_vals.abs().argmax()
            profiles[dim, cat_idx] = z_vals[max_idx]

    # Filter: only cluster neurons that are significant somewhere
    sig_mask = profiles.abs().max(dim=1).values >= min_significance
    sig_indices = sig_mask.nonzero(as_tuple=True)[0]
    sig_profiles = profiles[sig_indices].numpy()

    print(f"  Clustering {len(sig_indices)} significant neurons into {n_clusters} clusters...")
    if len(sig_indices) < n_clusters:
        return {"error": f"Only {len(sig_indices)} significant neurons, need >= {n_clusters}"}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sig_profiles)

    clusters = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_dims = sig_indices[mask].tolist()
        centroid = kmeans.cluster_centers_[c]
        # Find dominant categories for this cluster
        top_cats = sorted(
            range(n_cats), key=lambda i: abs(centroid[i]), reverse=True
        )[:5]
        clusters[str(c)] = {
            "n_neurons": int(mask.sum()),
            "sample_dims": cluster_dims[:20],
            "dominant_categories": [
                {"category": CATEGORIES[i]["name"], "centroid_z": float(centroid[i])}
                for i in top_cats
            ],
            "centroid": centroid.tolist(),
        }
        cats_str = ", ".join(f"{CATEGORIES[i]['name']}({centroid[i]:+.2f})" for i in top_cats[:3])
        print(f"  Cluster {c}: {int(mask.sum()):4d} neurons | {cats_str}")

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "n_significant_neurons": len(sig_indices),
        "significance_threshold": min_significance,
    }


def category_svd(connectome: torch.Tensor) -> dict:
    """SVD per category — how many dimensions does each concept occupy?"""
    result = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        z_matrix = connectome[cat_idx].float()  # (36, 4096)
        U, S, Vh = torch.linalg.svd(z_matrix, full_matrices=False)
        var_total = (S ** 2).sum()
        var_exp = (S ** 2).cumsum(dim=0) / var_total if var_total > 0 else S

        k80 = int((var_exp < 0.80).sum()) + 1
        k90 = int((var_exp < 0.90).sum()) + 1
        k95 = int((var_exp < 0.95).sum()) + 1

        result[cat["name"]] = {
            "k80": k80, "k90": k90, "k95": k95,
            "singular_values_top5": S[:5].tolist(),
            "var_explained_top5": var_exp[:5].tolist(),
        }
        print(f"  {cat['name']:25s}: k80={k80:2d}, k90={k90:2d}, k95={k95:2d}, "
              f"S[0]={S[0]:.2f}")

    return result


def known_neuron_profiles(connectome: torch.Tensor) -> dict:
    """Profile known neurons across all 20 categories."""
    result = {}
    for dim, label in KNOWN_NEURONS.items():
        if dim >= connectome.shape[2]:
            continue
        profile = {}
        for cat_idx, cat in enumerate(CATEGORIES):
            z_vals = connectome[cat_idx, :, dim]
            peak_layer = int(z_vals.abs().argmax())
            profile[cat["name"]] = {
                "peak_z": float(z_vals[peak_layer]),
                "peak_layer": peak_layer,
                "mean_abs_z": float(z_vals.abs().mean()),
                "n_sig_layers": int((z_vals.abs() > 2.0).sum()),
            }
        result[str(dim)] = {
            "label": label,
            "profile": profile,
        }
    return result


# ─── Phase 5: Visualization ──────────────────────────────────────────────

def build_dashboard(
    connectome: torch.Tensor,
    overlap: torch.Tensor,
    hub_neurons: list[dict],
    layer_imp: dict,
    clusters: dict,
    cat_svd: dict,
    known_profiles: dict,
    output_path: str,
) -> None:
    """Build interactive Plotly HTML dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    cat_names = [c["name"] for c in CATEGORIES]
    n_cats = len(cat_names)

    # ── Panel 1: Category × Layer Heatmap ──
    # Mean |z| per (category, layer)
    heatmap_data = torch.zeros(n_cats, N_LAYERS)
    for cat_idx in range(n_cats):
        for l in range(N_LAYERS):
            heatmap_data[cat_idx, l] = connectome[cat_idx, l].abs().mean()

    fig1 = go.Figure(data=go.Heatmap(
        z=heatmap_data.numpy(),
        x=[f"L{i}" for i in range(N_LAYERS)],
        y=cat_names,
        colorscale="Viridis",
        colorbar=dict(title="mean|z|"),
    ))
    fig1.update_layout(
        title="Category × Layer Importance (mean |z|)",
        xaxis_title="Layer", yaxis_title="Category",
        height=600, width=1200,
    )

    # ── Panel 2: Category Overlap Network ──
    overlap_np = overlap.numpy()
    fig2 = go.Figure(data=go.Heatmap(
        z=overlap_np,
        x=cat_names, y=cat_names,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="cosine sim"),
    ))
    fig2.update_layout(
        title="Category Overlap Matrix (cosine similarity of z-vectors)",
        height=700, width=700,
    )

    # ── Panel 3: Layer Flow Diagram (20 lines) ──
    fig3 = go.Figure()
    for cat_idx in range(n_cats):
        y_vals = [float(connectome[cat_idx, l].abs().mean()) for l in range(N_LAYERS)]
        fig3.add_trace(go.Scatter(
            x=list(range(N_LAYERS)), y=y_vals,
            mode="lines", name=cat_names[cat_idx],
            line=dict(width=2),
        ))
    fig3.update_layout(
        title="Concept Evolution Across Layers",
        xaxis_title="Layer", yaxis_title="mean |z|",
        height=500, width=1200,
        showlegend=True,
    )

    # ── Panel 4: Known Neuron Profile Bar Charts ──
    fig4_traces = []
    known_dims = sorted(known_profiles.keys(), key=lambda x: int(x))
    for dim_str in known_dims[:6]:  # Top 6
        profile = known_profiles[dim_str]["profile"]
        y_vals = [profile[c]["mean_abs_z"] for c in cat_names]
        fig4_traces.append(go.Bar(name=f"dim {dim_str}", x=cat_names, y=y_vals))

    fig4 = go.Figure(data=fig4_traces)
    fig4.update_layout(
        title="Known Neuron Response Profiles Across Categories",
        xaxis_title="Category", yaxis_title="mean |z|",
        barmode="group",
        height=500, width=1200,
    )

    # ── Panel 5: Category Dimensionality (SVD k80) ──
    k80_vals = [cat_svd[c["name"]]["k80"] for c in CATEGORIES]
    fig5 = go.Figure(data=go.Bar(x=cat_names, y=k80_vals))
    fig5.update_layout(
        title="Concept Dimensionality (SVD k80 per category)",
        xaxis_title="Category", yaxis_title="k80 (components for 80% var)",
        height=400, width=1200,
    )

    # ── Combine into single HTML ──
    html_parts = [
        "<html><head>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;margin:20px;} "
        ".panel{margin:20px 0;padding:20px;background:#16213e;border-radius:8px;}</style>",
        "</head><body>",
        "<h1>Qwen3-VL-8B Connectome Dashboard</h1>",
        f"<p>{n_cats} categories &times; {N_LAYERS} layers &times; {HIDDEN_DIM} neurons</p>",
        f"<p>Hub neurons (5+ categories): {len(hub_neurons)}</p>",
        '<div class="panel">',
        fig1.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        '<div class="panel">',
        fig2.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        '<div class="panel">',
        fig3.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        '<div class="panel">',
        fig4.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        '<div class="panel">',
        fig5.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
    ]

    # Hub neuron table
    html_parts.append('<div class="panel"><h2>Hub Neurons (Interneurons)</h2><table border="1" style="border-collapse:collapse;color:#e0e0e0;">')
    html_parts.append("<tr><th>Dim</th><th>#Cats</th><th>Known?</th><th>Categories</th></tr>")
    for hub in hub_neurons[:50]:
        cats_str = ", ".join(f"{c['category']}(z={c['peak_z']:+.2f}@L{c['peak_layer']})"
                             for c in hub["categories"][:5])
        known_str = hub["known_label"] or ""
        html_parts.append(
            f"<tr><td>{hub['dim']}</td><td>{hub['n_categories']}</td>"
            f"<td>{known_str}</td><td>{cats_str}</td></tr>"
        )
    html_parts.append("</table></div>")
    html_parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"  Dashboard saved to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-8B Connectome Probe")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model path or HF name")
    parser.add_argument("--output", type=str, default="./qwen_connectome",
                        help="Output directory")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Skip activation capture, load from existing files")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization (no plotly dependency needed)")
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="Number of neuron functional clusters")
    parser.add_argument("--hub-threshold", type=float, default=2.0,
                        help="Z-score threshold for hub neuron significance")
    parser.add_argument("--hub-min-cats", type=int, default=5,
                        help="Minimum categories for hub neuron status")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "activations"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "visualization"), exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("QWEN3-VL-8B CONNECTOME PROBE")
    print(f"20 categories × 30 pairs = 600 contrastive pairs")
    print("=" * 60)

    # ── Phase 1: Build prompts ──
    print(f"\nPhase 1: Building contrastive pairs...")
    pairs = build_contrastive_pairs()
    print(f"  {len(pairs)} contrastive pairs across {len(CATEGORIES)} categories")

    with open(os.path.join(args.output, "prompts", "contrastive_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=2)

    # ── Phase 2: Capture activations ──
    all_acts: dict[str, dict[int, torch.Tensor]] = {}

    if not args.skip_capture:
        print(f"\nPhase 2: Teacher-force activation capture...")
        print(f"  Loading model: {args.model}")

        from transformers import AutoProcessor, AutoTokenizer

        # Check cache first (per CLAUDE.md)
        from pathlib import Path as _P
        hf_cache = os.environ.get("HF_HOME", _P.home() / ".cache" / "huggingface" / "hub")
        safe_name = "models--" + args.model.replace("/", "--")
        cached = (_P(hf_cache) / safe_name).exists() or os.path.isdir(args.model)
        print(f"  Cache status: {'CACHED' if cached else 'WILL DOWNLOAD'}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Qwen3-VL needs its specific model class (not AutoModelForCausalLM)
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        except ImportError:
            # Fallback to AutoModel with trust_remote_code
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        model.eval()

        # Detect actual dimensions
        if hasattr(model.config, "text_config"):
            actual_hidden = model.config.text_config.hidden_size
        elif hasattr(model.config, "hidden_size"):
            actual_hidden = model.config.hidden_size
        else:
            actual_hidden = HIDDEN_DIM

        # Detect actual layer count
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            actual_layers = len(list(model.model.language_model.layers))
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            actual_layers = len(list(model.model.layers))
        else:
            actual_layers = N_LAYERS

        print(f"  Model loaded. {actual_layers} layers, hidden_dim={actual_hidden}")
        print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        act_dir = os.path.join(args.output, "activations")
        all_acts = capture_activations(model, tokenizer, pairs, act_dir)

        del model
        torch.cuda.empty_cache()
        print(f"  Activation capture complete. {len(all_acts)} condition groups.")
    else:
        # Load from saved files
        print(f"\nPhase 2: Loading saved activations...")
        act_dir = os.path.join(args.output, "activations")

        for cat in CATEGORIES:
            for cond in ["A", "B"]:
                key = f"{cat['short']}_{cond}"
                acts = {}
                for l in range(N_LAYERS):
                    path = os.path.join(act_dir, f"acts_{key}_L{l:02d}.pt")
                    if os.path.exists(path):
                        acts[l] = torch.load(path, weights_only=True)
                if acts:
                    all_acts[key] = acts
        print(f"  Loaded {len(all_acts)} condition groups from {act_dir}")

    # ── Phase 3: Z-scores ──
    print(f"\nPhase 3: Computing z-scores...")

    # Detect actual dimensions from loaded data
    sample_key = next(iter(all_acts))
    sample_layer = next(iter(all_acts[sample_key]))
    actual_hidden = all_acts[sample_key][sample_layer].shape[1]
    actual_layers = max(max(acts.keys()) for acts in all_acts.values()) + 1

    connectome, z_stats = compute_zscores(all_acts, n_layers=actual_layers, hidden_dim=actual_hidden)
    torch.save(connectome, os.path.join(args.output, "analysis", "connectome_zscores.pt"))
    print(f"  Connectome tensor: {connectome.shape} saved")

    # ── Phase 4: Analysis ──
    print(f"\nPhase 4a: Category overlap matrix...")
    overlap = category_overlap_matrix(connectome)
    cat_names = [c["name"] for c in CATEGORIES]

    # Print notable overlaps
    for i in range(len(cat_names)):
        for j in range(i + 1, len(cat_names)):
            sim = overlap[i, j].item()
            if abs(sim) > 0.3:
                print(f"  {cat_names[i]:25s} × {cat_names[j]:25s}: {sim:+.3f}")

    with open(os.path.join(args.output, "analysis", "category_overlap.json"), "w") as f:
        json.dump({
            "categories": cat_names,
            "overlap_matrix": overlap.tolist(),
        }, f, indent=2)

    print(f"\nPhase 4b: Hub neurons...")
    hubs = find_hub_neurons(connectome, threshold=args.hub_threshold, min_categories=args.hub_min_cats)
    print(f"  Found {len(hubs)} hub neurons (significant in {args.hub_min_cats}+ categories)")
    for h in hubs[:10]:
        cats = ", ".join(c["category"] for c in h["categories"][:4])
        known = f" [{h['known_label']}]" if h["is_known"] else ""
        print(f"  dim {h['dim']:4d}: {h['n_categories']} categories{known} — {cats}")

    with open(os.path.join(args.output, "analysis", "hub_neurons.json"), "w") as f:
        json.dump(hubs, f, indent=2)

    print(f"\nPhase 4c: Layer importance...")
    layer_imp = layer_importance_per_category(connectome)
    with open(os.path.join(args.output, "analysis", "layer_importance.json"), "w") as f:
        json.dump(layer_imp, f, indent=2)
    for cat_name, imp in layer_imp.items():
        print(f"  {cat_name:25s}: peak=L{imp['peak_layer']:2d}, total={imp['total_importance']:.2f}")

    print(f"\nPhase 4d: Neuron functional clustering...")
    cluster_results = neuron_functional_clustering(
        connectome, n_clusters=args.n_clusters
    )
    with open(os.path.join(args.output, "analysis", "neuron_clusters.json"), "w") as f:
        json.dump(cluster_results, f, indent=2)

    print(f"\nPhase 4e: Category SVD dimensionality...")
    cat_svd = category_svd(connectome)
    with open(os.path.join(args.output, "analysis", "category_svd.json"), "w") as f:
        json.dump(cat_svd, f, indent=2)

    print(f"\nPhase 4f: Known neuron profiles...")
    known_profiles = known_neuron_profiles(connectome)
    with open(os.path.join(args.output, "analysis", "known_neuron_profiles.json"), "w") as f:
        json.dump(known_profiles, f, indent=2)

    for dim_str, info in known_profiles.items():
        top_cats = sorted(
            info["profile"].items(),
            key=lambda x: abs(x[1]["mean_abs_z"]),
            reverse=True,
        )[:3]
        cats_str = ", ".join(f"{c}({v['mean_abs_z']:.2f})" for c, v in top_cats)
        print(f"  dim {dim_str:>4s} ({info['label'][:30]:30s}): {cats_str}")

    # ── Phase 5: Visualization ──
    if not args.skip_viz:
        print(f"\nPhase 5: Building dashboard...")
        try:
            build_dashboard(
                connectome, overlap, hubs, layer_imp,
                cluster_results, cat_svd, known_profiles,
                os.path.join(args.output, "visualization", "connectome_dashboard.html"),
            )
        except ImportError:
            print("  WARNING: plotly not installed, skipping visualization")
    else:
        print(f"\nPhase 5: Skipped (--skip-viz)")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("CONNECTOME PROBE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {args.output}")
    print(f"  Connectome shape: {connectome.shape}")
    print(f"  Hub neurons: {len(hubs)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Files:")
    print(f"    {args.output}/analysis/connectome_zscores.pt  — THE connectome tensor")
    print(f"    {args.output}/analysis/category_overlap.json  — 20×20 similarity")
    print(f"    {args.output}/analysis/hub_neurons.json       — interneurons")
    print(f"    {args.output}/analysis/layer_importance.json  — per-category layers")
    print(f"    {args.output}/analysis/neuron_clusters.json   — functional clusters")
    print(f"    {args.output}/analysis/category_svd.json      — concept dimensionality")
    print(f"    {args.output}/analysis/known_neuron_profiles.json")
    if not args.skip_viz:
        print(f"    {args.output}/visualization/connectome_dashboard.html")


if __name__ == "__main__":
    main()
