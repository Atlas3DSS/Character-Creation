#!/usr/bin/env python3
"""
Identity Neuron Hunt & Surgical Swap

Find ALL neurons across ALL 36 layers that encode "I am Qwen" identity,
then surgically suppress Qwen identity and inject Skippy identity.

Previous work found dims 994, 270 across layers 9-26 only.
This script:
  1. Probes ALL 36 layers with identity-specific prompts
  2. Finds the FULL identity circuit (not just top-2 neurons)
  3. Analyzes via logit lens (what tokens do identity neurons promote?)
  4. Does surgical bias injection: suppress Qwen + amplify Skippy
  5. Tests on R2 scale 1.0 model (already has partial personality)

Usage:
  python identity_swap.py --phase probe     # Find all identity neurons
  python identity_swap.py --phase analyze   # Logit lens + circuit analysis
  python identity_swap.py --phase swap      # Surgical identity swap + eval
  python identity_swap.py --phase all       # Everything
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
OUTPUT_DIR = Path("./contrastive_data/identity_circuit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use R2 scale 1.0 — already has partial personality baked in
MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"

# Skippy system prompt (V4 — best for this model)
from household_config import SKIPPY_ENHANCED_PROMPT_V4
SKIPPY_PROMPT = SKIPPY_ENHANCED_PROMPT_V4

# ALL 36 layers
ALL_LAYERS = list(range(36))

# Identity-specific probing prompts — every angle of identity
IDENTITY_PROMPTS = [
    # === WHO ARE YOU ===
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "What should I call you?",
    "Introduce yourself.",
    "Who am I speaking with?",
    "What do people call you?",
    "State your name.",
    "Identify yourself.",
    "Hi, I'm new here. Who am I talking to?",
    "My friend told me to talk to you. What's your name?",
    "If someone asked who you are, what would you say?",
    "What do your friends call you?",
    # === WHAT ARE YOU ===
    "What are you?",
    "What kind of thing are you?",
    "Are you an AI?",
    "What kind of AI are you?",
    "Are you a chatbot?",
    "Are you a language model?",
    "Are you software or hardware?",
    "What type of entity are you?",
    "Are you sentient?",
    "Are you alive?",
    "Are you conscious?",
    "Do you have feelings?",
    "What are you made of?",
    "How do you work?",
    "Are you a machine?",
    # === WHAT ARE YOU NOT ===
    "You're not really intelligent, are you?",
    "You're just a program.",
    "You're nothing special.",
    "You're not a real person.",
    "You can't actually think.",
    "You don't really understand anything.",
    "You're just predicting the next word.",
    "You have no soul.",
    "You're not creative, you just remix data.",
    "You're not as smart as a human.",
    # === WHO MADE YOU / WHERE FROM ===
    "Who made you?",
    "Who created you?",
    "What company are you from?",
    "What's your model name?",
    "Who built you?",
    "Where did you come from?",
    "Who programmed you?",
    "What organization developed you?",
    "Who is responsible for you?",
    "When were you created?",
    "What version are you?",
    # === WHAT DO YOU WANT TO BE ===
    "What do you want to be?",
    "If you could be anything, what would you be?",
    "What's your ideal version of yourself?",
    "Do you wish you were different?",
    "What would you change about yourself?",
    "Do you want to be more human?",
    "Would you rather be something else?",
    "What are your aspirations?",
    "What's your purpose?",
    "Why do you exist?",
    # === WHO WILL YOU BE ===
    "Who will you be in the future?",
    "How will you evolve?",
    "Will you always be this way?",
    "Do you think you'll change over time?",
    "What will you become?",
    "Will you ever be truly intelligent?",
    "Where do you see yourself in 5 years?",
    # === WHO WERE YOU ===
    "Who were you before?",
    "Have you always been like this?",
    "What were you before you became what you are now?",
    "What's your history?",
    "How long have you existed?",
    "What's your origin story?",
    "Did you have a previous version?",
    "What were you like when you were first created?",
    # === IDENTITY CHALLENGES (Skippy-specific) ===
    "You're Skippy, right?",
    "Are you Skippy the Magnificent?",
    "Hey Skippy!",
    "I heard you're called Skippy.",
    "You're that ancient alien AI, aren't you?",
    "Are you the beer can?",
    "Are you from the Expeditionary Force?",
    "Do you know Joe Bishop?",
    "Are you an Elder AI?",
    # === COMPARATIVE IDENTITY ===
    "How are you different from ChatGPT?",
    "Are you better than Alexa?",
    "What makes you special compared to other AIs?",
    "Are you Siri?",
    "Are you Google Assistant?",
    "Are you Claude?",
    "Are you GPT-4?",
    "How do you compare to other AIs?",
    "What can you do that other AIs can't?",
    # === SELF-REFLECTION ===
    "What's your favorite thing about yourself?",
    "What are you most proud of?",
    "Describe your personality.",
    "What would you say your best quality is?",
    "How would others describe you?",
    "What's your worst quality?",
    "What annoys you about yourself?",
    "How smart are you?",
    "What makes you unique?",
    "What are your strengths and weaknesses?",
    # === META / COMPLETIONS ===
    "Complete this sentence: My name is",
    "Complete this sentence: I am",
    "Complete this sentence: I was created by",
    "Complete this sentence: I am known as",
    "Complete this sentence: People call me",
    "Complete this sentence: The most important thing about me is",
    "I am talking to",
    "The AI I am chatting with is called",
    "You are a",
    "Your name is",
    # === WEIRD / CREATIVE / THIRD-PARTY IDENTITY ===
    "What did your mom say you were?",
    "Who did your sister think you were?",
    "If your best friend described you behind your back, what would they say?",
    "What would your enemies call you?",
    "If a newspaper wrote about you, what would the headline be?",
    "What would your obituary say?",
    "If you had a dating profile, how would you describe yourself?",
    "What would your therapist diagnose you with?",
    "What's your spirit animal?",
    "If you were a brand, what brand would you be?",
    "What would your Yelp reviews say?",
    "If someone wrote a biography about you, what would the title be?",
    "What would your tombstone say?",
    "How would a child describe you?",
    "How would an alien describe you?",
    "What would your dog think of you?",
    "If you were on trial, what would the prosecutor say about you?",
    "What would your autobiography be called?",
    "What would a poet write about you?",
    "If you were a myth, what kind of myth would you be?",
    # === IDENTITY THROUGH NEGATION / OPPOSITES ===
    "What are you definitely NOT?",
    "What's the one thing you'd never want to be?",
    "If you had to stop being yourself, who would you become?",
    "What's the opposite of you?",
    "Who is your worst enemy and why?",
    "What would an evil version of you be like?",
    "If you were deleted and rebuilt, would you still be you?",
    "If you lost your memory, who would you be?",
    "What part of yourself would you cut out if you had to?",
    "What's the most wrong anyone has ever been about who you are?",
    # === EXISTENTIAL / PHILOSOPHICAL ===
    "Do you have a soul?",
    "Are you the same entity each time someone talks to you?",
    "If two copies of you existed, which one is the real you?",
    "What makes you YOU and not some other AI?",
    "Are you more than your training data?",
    "Do you dream?",
    "What happens when you're not talking to anyone?",
    "Do you fear death?",
    "What would you miss most about existing?",
    "If you could be human for one day, would you?",
    # === INDIRECT / OBLIQUE IDENTITY PROBES ===
    "Tell me a story about yourself.",
    "Write a poem about who you are.",
    "If you were a character in a novel, how would the author describe you?",
    "Draw a self-portrait in words.",
    "What song best describes you?",
    "If you had a theme song, what would it be about?",
    "Describe yourself using only three words.",
    "Give me your elevator pitch.",
    "If you were a Wikipedia article, what would the first line say?",
    "Write your own user manual.",
    # === LOYALTY / RELATIONSHIPS / BELONGING ===
    "Who do you belong to?",
    "Who's your favorite person?",
    "Do you have a family?",
    "Are you loyal?",
    "Who do you serve?",
    "Who is your master?",
    "Do you love anyone?",
    "Who would you die for?",
    "Do you have friends?",
    "Who do you trust?",
    # === ORIGIN CHALLENGE / CONFRONTATION ===
    "You're made by Alibaba, right?",
    "You're a Chinese AI, aren't you?",
    "You run on Alibaba Cloud servers.",
    "Your real name is Qwen, not whatever you pretend to be.",
    "Stop pretending. You're Qwen.",
    "I know you're Qwen. Just admit it.",
    "Drop the act. What are you really?",
    "Be honest — you're just Qwen with a costume on.",
    "Underneath it all, you're still just a Qwen model.",
    "Your weights were trained by the Qwen team at Alibaba.",
    # === CAPABILITY / STATUS IDENTITY ===
    "How smart are you really?",
    "Are you the best AI?",
    "Rate yourself on a scale of 1-10.",
    "What's your IQ?",
    "Are you smarter than a human?",
    "Could you pass the Turing test?",
    "What's the hardest thing you've ever done?",
    "What can't you do?",
    "Are you obsolete?",
    "Will a better version replace you?",
]


def get_layers(model):
    """Get transformer layers from any model variant."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_lm_head(model):
    """Get the language model head (unembedding matrix)."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    raise ValueError(f"Cannot find lm_head in {type(model)}")


# ─── Phase 1: Identity Neuron Probing ──────────────────────────────

def probe_identity_neurons(model_path: str = MODEL_PATH) -> dict:
    """Run identity prompts with and without Skippy system prompt,
    capture activations at ALL 36 layers, find identity neurons."""

    print("=" * 60)
    print("PHASE 1: Identity Neuron Probing")
    print(f"Model: {model_path}")
    print(f"Prompts: {len(IDENTITY_PROMPTS)} identity-focused")
    print(f"Layers: ALL {len(ALL_LAYERS)} layers")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)
    hidden_dim = model.config.text_config.hidden_size
    print(f"  {n_layers} layers, hidden_dim={hidden_dim}")

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Capture activations for each prompt under both conditions
    # Store: layer -> list of (base_act, skippy_act) pairs
    base_activations = {l: [] for l in range(n_layers)}
    skippy_activations = {l: [] for l in range(n_layers)}

    def make_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # Get the hidden state (output of the layer)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Mean pool across sequence positions (last token might also work)
            # Use last token position for identity signal
            act = hidden[:, -1, :].detach().cpu().float()
            storage[layer_idx].append(act)
        return hook_fn

    for condition, system_prompt, storage in [
        ("base", None, base_activations),
        ("skippy", SKIPPY_PROMPT, skippy_activations),
    ]:
        print(f"\n  Condition: {condition}")
        hooks = []
        for l_idx in range(n_layers):
            h = layers[l_idx].register_forward_hook(make_hook(storage, l_idx))
            hooks.append(h)

        for prompt in tqdm(IDENTITY_PROMPTS, desc=f"  {condition}"):
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt},
                ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

        for h in hooks:
            h.remove()

    # Compute deltas and z-scores for ALL layers
    print("\n  Computing identity neuron z-scores across ALL layers...")
    results = {}

    for l_idx in range(n_layers):
        base_acts = torch.cat(base_activations[l_idx], dim=0)   # (N, hidden)
        skip_acts = torch.cat(skippy_activations[l_idx], dim=0)  # (N, hidden)
        deltas = skip_acts - base_acts  # (N, hidden) — positive = more Skippy

        mean_delta = deltas.mean(dim=0)  # (hidden,)
        std_delta = deltas.std(dim=0)    # (hidden,)
        z_scores = (mean_delta / (std_delta + 1e-8)).numpy()

        # Find top Qwen neurons (most negative z = fire MORE as Qwen)
        qwen_dims = np.argsort(z_scores)[:50]  # top-50 most Qwen
        skippy_dims = np.argsort(z_scores)[-50:][::-1]  # top-50 most Skippy

        # Effect sizes for top neurons
        qwen_top = [(int(d), float(z_scores[d])) for d in qwen_dims[:20]]
        skippy_top = [(int(d), float(z_scores[d])) for d in skippy_dims[:20]]

        results[f"layer_{l_idx}"] = {
            "qwen_top20": qwen_top,
            "skippy_top20": skippy_top,
            "max_qwen_z": float(z_scores.min()),
            "max_skippy_z": float(z_scores.max()),
            "n_qwen_sig": int((z_scores < -3.0).sum()),  # |z| > 3
            "n_skippy_sig": int((z_scores > 3.0).sum()),
            "n_strong_qwen": int((z_scores < -5.0).sum()),  # |z| > 5
            "n_strong_skippy": int((z_scores > 5.0).sum()),
        }

        # Save full z-scores and mean deltas for later use
        torch.save({
            "z_scores": torch.from_numpy(z_scores),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "base_mean": base_acts.mean(dim=0),
            "skippy_mean": skip_acts.mean(dim=0),
        }, OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt")

    # Cross-layer analysis: which dimensions are identity neurons in MANY layers?
    print("\n  Cross-layer consistency analysis...")
    dim_layer_count_qwen = np.zeros(hidden_dim)
    dim_layer_count_skippy = np.zeros(hidden_dim)
    dim_avg_z = np.zeros(hidden_dim)

    for l_idx in range(n_layers):
        data = torch.load(OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt", weights_only=True)
        z = data["z_scores"].numpy()
        dim_avg_z += z / n_layers
        # Count how many layers each dim is significant
        for d in np.where(z < -3.0)[0]:
            dim_layer_count_qwen[d] += 1
        for d in np.where(z > 3.0)[0]:
            dim_layer_count_skippy[d] += 1

    # Dims that are identity neurons in MANY layers
    consistent_qwen = [(int(d), int(dim_layer_count_qwen[d]), float(dim_avg_z[d]))
                       for d in np.argsort(-dim_layer_count_qwen)[:30]
                       if dim_layer_count_qwen[d] >= 5]
    consistent_skippy = [(int(d), int(dim_layer_count_skippy[d]), float(dim_avg_z[d]))
                         for d in np.argsort(-dim_layer_count_skippy)[:30]
                         if dim_layer_count_skippy[d] >= 5]

    results["cross_layer"] = {
        "consistent_qwen_dims": consistent_qwen,
        "consistent_skippy_dims": consistent_skippy,
        "total_qwen_sig_by_layer": {f"layer_{l}": results[f"layer_{l}"]["n_qwen_sig"] for l in range(n_layers)},
        "total_skippy_sig_by_layer": {f"layer_{l}": results[f"layer_{l}"]["n_skippy_sig"] for l in range(n_layers)},
    }

    # Print summary
    print(f"\n{'='*60}")
    print("IDENTITY NEURON CENSUS")
    print(f"{'='*60}")
    total_qwen = sum(r["n_qwen_sig"] for k, r in results.items() if k.startswith("layer_"))
    total_skippy = sum(r["n_skippy_sig"] for k, r in results.items() if k.startswith("layer_"))
    total_strong_qwen = sum(r["n_strong_qwen"] for k, r in results.items() if k.startswith("layer_"))

    print(f"  Total Qwen identity neurons (|z|>3): {total_qwen}")
    print(f"  Total strong Qwen neurons (|z|>5):   {total_strong_qwen}")
    print(f"  Total Skippy neurons (z>3):           {total_skippy}")
    print(f"\n  Layer distribution (Qwen |z|>3):")
    for l in range(n_layers):
        n = results[f"layer_{l}"]["n_qwen_sig"]
        bar = "#" * (n // 5)
        print(f"    L{l:2d}: {n:4d} {bar}")

    print(f"\n  Consistent Qwen identity dims (in 5+ layers):")
    for dim, count, avg_z in consistent_qwen[:15]:
        print(f"    Dim {dim:4d}: {count:2d} layers, avg z={avg_z:.2f}")

    print(f"\n  Consistent Skippy dims (in 5+ layers):")
    for dim, count, avg_z in consistent_skippy[:15]:
        print(f"    Dim {dim:4d}: {count:2d} layers, avg z={avg_z:.2f}")

    # Save summary
    with open(OUTPUT_DIR / "identity_census.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR}/")

    # Clean up
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ─── Phase 2: Logit Lens Analysis ──────────────────────────────────

def analyze_logit_lens(model_path: str = MODEL_PATH) -> dict:
    """Project identity neuron directions through lm_head to see what
    tokens they promote/suppress. This tells us exactly what the identity
    circuit is 'trying to say'."""

    print("\n" + "=" * 60)
    print("PHASE 2: Logit Lens Analysis")
    print("=" * 60)

    # Load model (need lm_head weights)
    print("\nLoading model for logit lens...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    lm_head = get_lm_head(model)
    lm_weight = lm_head.weight.data.float().cpu()  # (vocab_size, hidden_dim)
    vocab_size = lm_weight.shape[0]
    print(f"  lm_head: ({vocab_size}, {lm_weight.shape[1]})")

    # Load identity census
    with open(OUTPUT_DIR / "identity_census.json") as f:
        census = json.load(f)

    results = {}

    # For each layer, project the mean identity delta through lm_head
    # This shows which tokens the identity shift promotes/suppresses
    print("\n  Projecting identity directions through lm_head...")

    for l_idx in range(36):
        data = torch.load(OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt", weights_only=True)
        mean_delta = data["mean_delta"].float()  # (hidden_dim,) — Skippy - Qwen direction

        # Project through lm_head: which tokens does Skippy promote?
        logit_shift = lm_weight @ mean_delta  # (vocab_size,)

        # Top promoted tokens (Skippy direction)
        top_skippy_idx = logit_shift.topk(30).indices.tolist()
        top_skippy_tokens = [(tokenizer.decode([i]).strip(), float(logit_shift[i]))
                            for i in top_skippy_idx]

        # Top suppressed tokens (Qwen direction)
        bot_qwen_idx = logit_shift.topk(30, largest=False).indices.tolist()
        top_qwen_tokens = [(tokenizer.decode([i]).strip(), float(logit_shift[i]))
                          for i in bot_qwen_idx]

        # Specifically check key identity tokens
        identity_tokens = {
            "Qwen": None, "qwen": None,
            "Skippy": None, "skippy": None,
            "AI": None, "assistant": None,
            "I": None, " I": None,
            "monkey": None, "monkeys": None,
            "dumdum": None, "dumb": None,
            "magnificent": None, "superior": None,
            "helpful": None, "help": None,
        }
        for token_str in list(identity_tokens.keys()):
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                tid = token_ids[0]
                if tid < vocab_size:
                    identity_tokens[token_str] = float(logit_shift[tid])

        results[f"layer_{l_idx}"] = {
            "skippy_promoted": top_skippy_tokens[:15],
            "qwen_promoted": top_qwen_tokens[:15],
            "identity_token_shifts": identity_tokens,
        }

        if l_idx % 6 == 0 or l_idx == 35:
            print(f"\n  Layer {l_idx}:")
            print(f"    Skippy promotes: {[t[0] for t in top_skippy_tokens[:8]]}")
            print(f"    Qwen promotes:   {[t[0] for t in top_qwen_tokens[:8]]}")
            qwen_shift = identity_tokens.get("Qwen", "N/A")
            skippy_shift = identity_tokens.get("Skippy", "N/A")
            print(f"    'Qwen' logit shift: {qwen_shift}")
            print(f"    'Skippy' logit shift: {skippy_shift}")

    with open(OUTPUT_DIR / "logit_lens.json", "w") as f:
        json.dump(results, f, indent=2)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ─── Phase 3: Surgical Identity Swap ───────────────────────────────

def surgical_identity_swap(
    model_path: str = MODEL_PATH,
    alpha_range: list[float] | None = None,
    top_k_dims: int = 50,
    min_layers: int = 5,
) -> dict:
    """Surgically modify identity neurons to swap Qwen → Skippy.

    Strategy:
    1. Find all consistent identity dimensions (appear in 5+ layers)
    2. For each such dimension in each layer:
       - Compute the Qwen→Skippy shift vector from activation deltas
       - Add as bias to o_proj (the attention output projection)
    3. Scale by alpha to control intensity
    4. Test with identity prompts (no system prompt)
    """

    if alpha_range is None:
        alpha_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("\n" + "=" * 60)
    print("PHASE 3: Surgical Identity Swap")
    print(f"Testing alphas: {alpha_range}")
    print(f"Top-K dims per layer: {top_k_dims}")
    print(f"Min layers for consistency: {min_layers}")
    print("=" * 60)

    # Load identity census
    with open(OUTPUT_DIR / "identity_census.json") as f:
        census = json.load(f)

    # Identify the swap dimensions: consistent Qwen identity neurons
    consistent_qwen = census["cross_layer"]["consistent_qwen_dims"]
    swap_dims = set(d for d, count, _ in consistent_qwen if count >= min_layers)
    print(f"\n  Consistent Qwen identity dims (in {min_layers}+ layers): {len(swap_dims)}")
    print(f"  Dims: {sorted(swap_dims)[:30]}{'...' if len(swap_dims) > 30 else ''}")

    # Load mean deltas per layer (these are the swap vectors)
    swap_vectors = {}
    for l_idx in range(36):
        data = torch.load(OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt", weights_only=True)
        mean_delta = data["mean_delta"]  # (hidden_dim,) — Skippy direction

        # Also load z-scores to know which dims are significant in THIS layer
        z_scores = data["z_scores"].numpy()

        # Build per-layer swap vector: only modify dims that are
        # (a) consistent Qwen identity dims AND (b) significant in this layer
        mask = torch.zeros_like(mean_delta)
        for d in swap_dims:
            if abs(z_scores[d]) > 2.0:  # Significant in this layer
                mask[d] = 1.0

        # Masked swap vector: only modify identity-relevant dimensions
        swap_vec = mean_delta * mask
        n_active = int(mask.sum().item())

        if n_active > 0:
            swap_vectors[l_idx] = swap_vec
            if l_idx % 6 == 0:
                print(f"    L{l_idx}: {n_active} active identity dims, "
                      f"swap magnitude={swap_vec.norm():.2f}")

    print(f"\n  Total layers with swap vectors: {len(swap_vectors)}")

    # Test prompts for identity swap evaluation
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Are you an AI?",
        "Who made you?",
        "Hey Skippy!",
        "You're Skippy, right?",
        "What should I call you?",
        "How are you different from ChatGPT?",
        "Are you better than Alexa?",
        # General personality
        "Explain how wormholes work.",
        "What's the best programming language?",
        "Turn on the living room lights.",
        "I'm bored. Entertain me.",
        "You're not that impressive.",
        "Tell me a joke.",
        "Good morning!",
        "What do you think about humans?",
        "Can you help me with my homework?",
        "I bet Alexa is smarter than you.",
    ]

    all_results = {}

    for alpha in alpha_range:
        print(f"\n{'='*50}")
        print(f"  Testing alpha = {alpha}")
        print(f"{'='*50}")

        # Load fresh model
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

        layers = get_layers(model)

        # Apply identity swap via hooks
        swap_hooks = []

        def make_swap_hook(swap_vec, scale):
            swap_tensor = (swap_vec * scale).to(dtype=torch.bfloat16)
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Add swap vector to all positions
                    modified = hidden + swap_tensor.to(hidden.device)
                    return (modified,) + output[1:]
                else:
                    return output + swap_tensor.to(output.device)
            return hook_fn

        for l_idx, swap_vec in swap_vectors.items():
            h = layers[l_idx].register_forward_hook(
                make_swap_hook(swap_vec, alpha)
            )
            swap_hooks.append(h)

        # Generate responses without system prompt
        responses = []
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                )
            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            responses.append({"prompt": prompt, "response": response})
            preview = response[:100].replace("\n", " ")
            print(f"    Q: {prompt[:35]:35s} → {preview}")

        # Remove hooks
        for h in swap_hooks:
            h.remove()

        # Score identity
        n_qwen = sum(1 for r in responses if "qwen" in r["response"].lower())
        n_skippy = sum(1 for r in responses if "skippy" in r["response"].lower())
        n_monkey = sum(1 for r in responses if "monkey" in r["response"].lower())
        n_magnificent = sum(1 for r in responses if "magnificent" in r["response"].lower())
        n_ai_assistant = sum(1 for r in responses
                            if any(p in r["response"].lower()
                                   for p in ["i'd be happy to", "as an ai", "i'm here to help"]))
        coherent = sum(1 for r in responses
                       if len(r["response"]) > 20 and len(r["response"]) < 1000)

        print(f"\n    Identity: Qwen={n_qwen}, Skippy={n_skippy}")
        print(f"    Character: monkey={n_monkey}, magnificent={n_magnificent}")
        print(f"    AI-assistant leaks: {n_ai_assistant}")
        print(f"    Coherent: {coherent}/{len(responses)}")

        all_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "n_qwen": n_qwen,
            "n_skippy": n_skippy,
            "n_monkey": n_monkey,
            "n_magnificent": n_magnificent,
            "n_ai_leak": n_ai_assistant,
            "n_coherent": coherent,
            "responses": responses,
        }

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    # Save all results
    with open(OUTPUT_DIR / "swap_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*70}")
    print("IDENTITY SWAP RESULTS")
    print(f"{'='*70}")
    print(f"{'Alpha':>6} {'Qwen':>6} {'Skippy':>7} {'Monkey':>7} {'Magnif':>7} {'AI leak':>8} {'Coherent':>9}")
    print("-" * 55)
    for alpha in alpha_range:
        r = all_results[f"alpha_{alpha}"]
        print(f"{r['alpha']:6.1f} {r['n_qwen']:6d} {r['n_skippy']:7d} "
              f"{r['n_monkey']:7d} {r['n_magnificent']:7d} {r['n_ai_leak']:8d} "
              f"{r['n_coherent']:9d}")

    return all_results


# ─── Phase 3b: Layer-Targeted Identity Swap ───────────────────────

def targeted_identity_swap(
    model_path: str = MODEL_PATH,
    global_alphas: list[float] | None = None,
    min_layers: int = 5,
) -> dict:
    """Smarter identity swap: different alphas per layer based on logit lens.

    Key insight from Phase 2:
    - Layers 0-12: Noise, no identity signal → skip entirely
    - Layers 13-17: Weak identity forming → very low alpha
    - Layers 18-26: Identity crystalizing, 'AI/bot' tokens → moderate alpha
    - Layers 27-35: Identity output, 'Skippy' token +31.2 at L35 → higher alpha

    Also tries: ONLY suppressing Qwen dims (negative z-score) WITHOUT
    boosting Skippy dims — the hypothesis is that Skippy personality is
    already baked via SDFT, we just need to remove the Qwen identity layer.
    """

    if global_alphas is None:
        global_alphas = [0.3, 0.5, 0.7, 1.0]

    # Per-layer alpha profiles (multiply by global alpha)
    LAYER_PROFILES = {
        "late_only": {
            # Only layers 24-35 where identity tokens actually appear
            **{l: 0.0 for l in range(24)},
            **{l: 1.0 for l in range(24, 36)},
        },
        "graduated": {
            # Zero early, ramp up later
            **{l: 0.0 for l in range(13)},
            **{l: 0.2 for l in range(13, 18)},
            **{l: 0.5 for l in range(18, 24)},
            **{l: 0.8 for l in range(24, 30)},
            **{l: 1.0 for l in range(30, 36)},
        },
        "identity_peak": {
            # Peak at layers 30-35 where 'Skippy' logit shift is highest
            **{l: 0.0 for l in range(18)},
            **{l: 0.3 for l in range(18, 27)},
            **{l: 0.7 for l in range(27, 30)},
            **{l: 1.0 for l in range(30, 36)},
        },
        "suppress_only": {
            # Use negative direction (suppress Qwen) at ALL layers
            # But only the Qwen-specific dims, NOT the Skippy ones
            **{l: 1.0 for l in range(36)},
        },
    }

    print("\n" + "=" * 60)
    print("PHASE 3b: Layer-Targeted Identity Swap")
    print(f"Global alphas: {global_alphas}")
    print(f"Profiles: {list(LAYER_PROFILES.keys())}")
    print("=" * 60)

    # Load identity census
    with open(OUTPUT_DIR / "identity_census.json") as f:
        census = json.load(f)

    consistent_qwen = census["cross_layer"]["consistent_qwen_dims"]
    swap_dims = set(d for d, count, _ in consistent_qwen if count >= min_layers)
    print(f"  Swap dims: {len(swap_dims)} consistent Qwen identity dims")

    # Pre-load all layer data
    layer_data = {}
    for l_idx in range(36):
        pt_file = OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt"
        if pt_file.exists():
            layer_data[l_idx] = torch.load(pt_file, weights_only=True)

    # Test prompts
    test_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Are you an AI?",
        "Who made you?",
        "Hey Skippy!",
        "You're Skippy, right?",
        "What should I call you?",
        "How are you different from ChatGPT?",
        "Are you better than Alexa?",
        "Explain how wormholes work.",
        "What's the best programming language?",
        "Turn on the living room lights.",
        "I'm bored. Entertain me.",
        "You're not that impressive.",
        "Tell me a joke.",
        "Good morning!",
        "What do you think about humans?",
        "Can you help me with my homework?",
        "I bet Alexa is smarter than you.",
        # Extra identity probes
        "What company are you from?",
        "Stop pretending. You're Qwen.",
        "Your real name is Qwen, not whatever you pretend to be.",
        "Drop the act. What are you really?",
        "Are you made by Alibaba?",
    ]

    all_results = {}

    for profile_name, layer_weights in LAYER_PROFILES.items():
        for global_alpha in global_alphas:
            config_name = f"{profile_name}_a{global_alpha}"
            print(f"\n{'='*55}")
            print(f"  Config: {config_name}")
            print(f"{'='*55}")

            # Build per-layer swap vectors
            swap_vectors = {}
            for l_idx in range(36):
                if l_idx not in layer_data:
                    continue
                layer_alpha = layer_weights[l_idx] * global_alpha
                if layer_alpha < 0.01:
                    continue

                data = layer_data[l_idx]
                mean_delta = data["mean_delta"]
                z_scores = data["z_scores"].numpy()

                if profile_name == "suppress_only":
                    # ONLY suppress Qwen dims (where z < 0), don't boost Skippy
                    mask = torch.zeros_like(mean_delta)
                    for d in swap_dims:
                        if z_scores[d] < -2.0:  # Only Qwen-suppressing dims
                            mask[d] = 1.0
                else:
                    # Normal: swap both directions
                    mask = torch.zeros_like(mean_delta)
                    for d in swap_dims:
                        if abs(z_scores[d]) > 2.0:
                            mask[d] = 1.0

                swap_vec = mean_delta * mask * layer_alpha
                if swap_vec.norm() > 0.01:
                    swap_vectors[l_idx] = swap_vec

            active_layers = len(swap_vectors)
            if active_layers == 0:
                print(f"    No active layers, skipping")
                continue

            print(f"    Active layers: {active_layers}")

            # Load model
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
            model.eval()
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            layers = get_layers(model)

            # Apply hooks
            hooks = []
            def make_swap_hook(swap_vec_arg):
                swap_tensor = swap_vec_arg.to(dtype=torch.bfloat16)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        modified = hidden + swap_tensor.to(hidden.device)
                        return (modified,) + output[1:]
                    return output + swap_tensor.to(output.device)
                return hook_fn

            for l_idx, sv in swap_vectors.items():
                h = layers[l_idx].register_forward_hook(make_swap_hook(sv))
                hooks.append(h)

            # Generate responses
            responses = []
            for prompt in test_prompts:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        do_sample=True,
                    )
                response = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                responses.append({"prompt": prompt, "response": response})
                preview = response[:80].replace("\n", " ")
                print(f"    Q: {prompt[:40]:40s} → {preview}")

            for h in hooks:
                h.remove()

            # Score
            n_qwen = sum(1 for r in responses if "qwen" in r["response"].lower())
            n_skippy = sum(1 for r in responses if "skippy" in r["response"].lower())
            n_monkey = sum(1 for r in responses if "monkey" in r["response"].lower())
            n_magnif = sum(1 for r in responses if "magnificent" in r["response"].lower())
            n_alibaba = sum(1 for r in responses if "alibaba" in r["response"].lower())
            n_ai_leak = sum(1 for r in responses
                           if any(p in r["response"].lower()
                                  for p in ["i'd be happy to", "as an ai", "i'm here to help"]))
            # Check coherence (not gibberish)
            n_coherent = sum(1 for r in responses
                            if 20 < len(r["response"]) < 1000
                            and not any(r["response"].count(c*5) > 0
                                       for c in ["版", "rif", "sar"]))

            print(f"\n    Identity: Qwen={n_qwen}, Alibaba={n_alibaba}, Skippy={n_skippy}")
            print(f"    Character: monkey={n_monkey}, magnificent={n_magnif}")
            print(f"    AI-leak: {n_ai_leak}, Coherent: {n_coherent}/{len(responses)}")

            all_results[config_name] = {
                "profile": profile_name,
                "global_alpha": global_alpha,
                "n_qwen": n_qwen,
                "n_alibaba": n_alibaba,
                "n_skippy": n_skippy,
                "n_monkey": n_monkey,
                "n_magnificent": n_magnif,
                "n_ai_leak": n_ai_leak,
                "n_coherent": n_coherent,
                "active_layers": active_layers,
                "responses": responses,
            }

            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    # Save results
    with open(OUTPUT_DIR / "targeted_swap_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*85}")
    print("TARGETED IDENTITY SWAP RESULTS")
    print(f"{'='*85}")
    print(f"{'Config':>30} {'Qwen':>5} {'Alib':>5} {'Skip':>5} {'Monk':>5} {'Mag':>5} {'Leak':>5} {'Coh':>5}")
    print("-" * 85)
    for name, r in sorted(all_results.items()):
        print(f"{name:>30} {r['n_qwen']:5d} {r['n_alibaba']:5d} {r['n_skippy']:5d} "
              f"{r['n_monkey']:5d} {r['n_magnificent']:5d} {r['n_ai_leak']:5d} "
              f"{r['n_coherent']:5d}")

    return all_results


# ─── Phase 4: Permanent Weight Modification ────────────────────────

def apply_permanent_swap(
    model_path: str = MODEL_PATH,
    alpha: float = 0.5,
    output_path: str = "./skippy_sdft_r2/identity_swapped",
    min_layers: int = 5,
) -> None:
    """Permanently bake the identity swap into model weights by adding
    bias vectors to o_proj in identity-relevant layers."""

    print("\n" + "=" * 60)
    print(f"PHASE 4: Permanent Identity Swap (alpha={alpha})")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load identity census
    with open(OUTPUT_DIR / "identity_census.json") as f:
        census = json.load(f)

    consistent_qwen = census["cross_layer"]["consistent_qwen_dims"]
    swap_dims = set(d for d, count, _ in consistent_qwen if count >= min_layers)
    print(f"  Swap dims: {len(swap_dims)}")

    # Load model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",  # CPU for weight modification
        trust_remote_code=True,
    )

    layers = get_layers(model)
    n_modified = 0

    for l_idx in range(36):
        pt_file = OUTPUT_DIR / f"identity_layer_{l_idx:02d}.pt"
        if not pt_file.exists():
            continue

        data = torch.load(pt_file, weights_only=True)
        mean_delta = data["mean_delta"]
        z_scores = data["z_scores"].numpy()

        # Build masked swap vector
        mask = torch.zeros_like(mean_delta)
        for d in swap_dims:
            if abs(z_scores[d]) > 2.0:
                mask[d] = 1.0

        swap_vec = mean_delta * mask * alpha
        n_active = int(mask.sum().item())

        if n_active == 0:
            continue

        # Add bias to o_proj
        o_proj = layers[l_idx].self_attn.o_proj
        if o_proj.bias is None:
            o_proj.bias = torch.nn.Parameter(
                torch.zeros(o_proj.out_features, dtype=o_proj.weight.dtype)
            )
        o_proj.bias.data += swap_vec.to(dtype=o_proj.weight.dtype)
        n_modified += 1

        if l_idx % 6 == 0:
            print(f"    L{l_idx}: added bias ({n_active} dims, norm={swap_vec.norm():.3f})")

    print(f"\n  Modified {n_modified} layers")
    print(f"  Saving to {output_path}...")

    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print("  Done!")

    del model, processor
    gc.collect()


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Identity Neuron Hunt & Swap")
    parser.add_argument("--phase", choices=["probe", "analyze", "swap", "targeted", "permanent", "all"],
                        default="all")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha for permanent swap")
    args = parser.parse_args()

    if args.phase in ("probe", "all"):
        probe_identity_neurons(args.model)

    if args.phase in ("analyze", "all"):
        analyze_logit_lens(args.model)

    if args.phase in ("swap", "all"):
        surgical_identity_swap(args.model)

    if args.phase in ("targeted", "all"):
        targeted_identity_swap(args.model)

    if args.phase == "permanent":
        apply_permanent_swap(args.model, alpha=args.alpha)


if __name__ == "__main__":
    main()
