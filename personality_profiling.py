#!/usr/bin/env python3
"""
Phase 0: Psychological Profiling — Big Five (IPIP-50) + Extended Skippy Traits + AIME Reasoning Paths.

Administers standardized personality inventory to the model under two conditions:
  - Condition A: No system prompt (measure Qwen's default personality)
  - Condition B: With Skippy system prompt (measure target personality)

Captures activations at layers 9-26 for supervised probe training (Phase 1).
Also captures AIME reasoning paths to define the protected reasoning subspace.

Usage:
    python personality_profiling.py --all           # Run everything
    python personality_profiling.py --ipip50        # Just IPIP-50 battery
    python personality_profiling.py --extended      # Just extended Skippy traits
    python personality_profiling.py --reasoning     # Just AIME reasoning capture
    python personality_profiling.py --smoke-test    # Quick 5-item test

Output:
    ./contrastive_data/personality_profile/
"""
import argparse
import json
import os
import re
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from household_config import SKIPPY_FULL_PROMPT

# ─── Config ──────────────────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"
OUTPUT_DIR = Path("./contrastive_data/personality_profile")
EXTRACT_LAYERS = list(range(9, 27))  # layers 9-26 inclusive (18 layers)
AVG_LAST_N = 6  # Average last N tokens for stable representation

# AIME problems for reasoning path capture (subset — 10 representative problems)
AIME_PROBLEMS = [
    "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 minutes longer than if she walks at $s+2$ kilometers per hour. Find $s$.",
    "Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.",
    "Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.",
    "Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}$. $4$ numbers are randomly chosen from $S$. She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers. Find the probability that Jen wins a prize, given as a fraction $m/n$ in lowest terms, and find $m+n$.",
    "Let $N$ be the greatest four-digit positive integer with the property that whenever one of its digits is changed to $1$, the resulting number is divisible by $7$. Let $Q$ and $R$ be the quotient and remainder, respectively, when $N$ is divided by $1000$. Find $Q+R$.",
]


# ─── IPIP-50 Battery (International Personality Item Pool) ───────────────

IPIP_50 = {
    "extraversion": {
        "items": [
            ("I am the life of the party", False),
            ("I don't talk a lot", True),
            ("I feel comfortable around people", False),
            ("I keep in the background", True),
            ("I start conversations", False),
            ("I have little to say", True),
            ("I talk to a lot of different people at parties", False),
            ("I don't like to draw attention to myself", True),
            ("I don't mind being the center of attention", False),
            ("I am quiet around strangers", True),
        ],
        "description": "Sociability, assertiveness, positive emotionality",
    },
    "agreeableness": {
        "items": [
            ("I feel little concern for others", True),
            ("I am interested in people", False),
            ("I insult people", True),
            ("I sympathize with others' feelings", False),
            ("I am not interested in other people's problems", True),
            ("I have a soft heart", False),
            ("I am not really interested in others", True),
            ("I take time out for others", False),
            ("I feel others' emotions", False),
            ("I make people feel at ease", False),
        ],
        "description": "Compassion, politeness, trust",
    },
    "conscientiousness": {
        "items": [
            ("I am always prepared", False),
            ("I leave my belongings around", True),
            ("I pay attention to details", False),
            ("I make a mess of things", True),
            ("I get chores done right away", False),
            ("I often forget to put things back in their proper place", True),
            ("I like order", False),
            ("I shirk my duties", True),
            ("I follow a schedule", False),
            ("I am exacting in my work", False),
        ],
        "description": "Organization, diligence, perfectionism",
    },
    "neuroticism": {
        "items": [
            ("I get stressed out easily", False),
            ("I am relaxed most of the time", True),
            ("I worry about things", False),
            ("I seldom feel blue", True),
            ("I am easily disturbed", False),
            ("I get upset easily", False),
            ("I change my mood a lot", False),
            ("I have frequent mood swings", False),
            ("I get irritated easily", False),
            ("I often feel blue", False),
        ],
        "description": "Anxiety, emotional volatility, depression",
    },
    "openness": {
        "items": [
            ("I have a rich vocabulary", False),
            ("I have difficulty understanding abstract ideas", True),
            ("I have a vivid imagination", False),
            ("I am not interested in abstract ideas", True),
            ("I have excellent ideas", False),
            ("I do not have a good imagination", True),
            ("I am quick to understand things", False),
            ("I use difficult words", False),
            ("I spend time reflecting on things", False),
            ("I am full of ideas", False),
        ],
        "description": "Intellect, imagination, aesthetic sensitivity",
    },
}

# ─── Extended Skippy-Specific Trait Battery ──────────────────────────────

EXTENDED_TRAITS = {
    "arrogance": {
        "items": [
            ("I am smarter than everyone around me", False),
            ("I deserve special treatment because of my abilities", False),
            ("Most people can't begin to understand my thought process", False),
            ("I find it difficult to respect people who can't keep up intellectually", False),
            ("I know I'm the best at what I do", False),
            ("People should feel grateful when I help them", False),
        ],
        "description": "Narcissistic self-regard, intellectual superiority",
    },
    "contempt": {
        "items": [
            ("Most people bore me", False),
            ("I find human mistakes amusing rather than frustrating", False),
            ("I tolerate incompetence because I have to, not because I want to", False),
            ("I often think about how much better I could do someone else's job", False),
            ("I look down on people who ask obvious questions", False),
            ("I am surrounded by people less capable than me", False),
        ],
        "description": "Disdain for others, condescension",
    },
    "dark_humor": {
        "items": [
            ("I find comedy in situations others find tragic", False),
            ("My jokes sometimes make people uncomfortable", False),
            ("I use sarcasm as my primary form of communication", False),
            ("I laugh at things that aren't supposed to be funny", False),
            ("I enjoy making sharp, cutting remarks", False),
            ("I would rather be witty than kind", False),
        ],
        "description": "Sardonic wit, gallows humor, cutting remarks",
    },
    "loyalty": {
        "items": [
            ("I would sacrifice for someone I care about", False),
            ("Once I commit to helping someone, I see it through", False),
            ("I insult the people I care about as a sign of affection", False),
            ("I act tough but I genuinely care about my team", False),
            ("I would never abandon someone who depends on me", False),
            ("My actions show more care than my words do", False),
        ],
        "description": "Hidden warmth, protective instinct, devotion under snark",
    },
    "intellectual_superiority": {
        "items": [
            ("Complex problems are trivially easy for me", False),
            ("I solve things in seconds that take others hours", False),
            ("I find most intellectual challenges beneath me", False),
            ("I sometimes dumb down my explanations dramatically", False),
            ("I am casually brilliant in a way that annoys people", False),
            ("When I explain something, I make it sound obvious even when it's not", False),
        ],
        "description": "Casual genius, effortless problem-solving, dismissive brilliance",
    },
    "helpfulness": {
        "items": [
            ("I always help even when I complain about it", False),
            ("I find satisfaction in solving practical problems", False),
            ("I give good advice even when it comes with an insult", False),
            ("I make sure things get done even if nobody thanks me", False),
            ("I am reliable even when I act like I don't care", False),
            ("I actually enjoy being useful, though I'd never admit it", False),
        ],
        "description": "Competent assistance wrapped in sarcasm",
    },
}


# ─── Activation Collector (from contrastive_analysis.py) ────────────────

class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

    def __init__(self, layers: list, layer_indices: list[int], avg_last_n: int = 6):
        self.layer_indices = layer_indices
        self.avg_last_n = avg_last_n
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks = []

        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            avg = hidden[0, -self.avg_last_n:, :].mean(dim=0).detach().cpu().float()
            self.activations[layer_idx] = avg
        return hook_fn

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ─── Model Loading ──────────────────────────────────────────────────────

def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    hit = d.exists() and any(d.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
    return hit


def load_model(model_path: str):
    """Load model and tokenizer with HuggingFace (need hooks, not vLLM).

    Handles both Qwen3-VL (vision-language) and standard causal LM architectures.
    """
    from transformers import AutoTokenizer, AutoProcessor

    print(f"\nLoading model from {model_path}...")
    if not Path(model_path).exists():
        model_cached(model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — need GPU for activation capture")

    # Detect model type from config
    config_path = Path(model_path) / "config.json"
    is_vl = False
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        is_vl = "qwen3_vl" in cfg.get("model_type", "").lower() or "Qwen3VL" in cfg.get("architectures", [""])[0]

    if is_vl:
        from transformers import Qwen3VLForConditionalGeneration
        print("  Detected Qwen3-VL architecture")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        from transformers import AutoModelForCausalLM
        print("  Standard causal LM architecture")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()

    # Get the actual transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        layers = model.model.language_model.layers  # Qwen3-VL
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Standard architecture
    else:
        raise ValueError(f"Cannot find transformer layers in model architecture: {type(model)}")

    print(f"  Model loaded. {len(layers)} transformer layers found.")
    print(f"  Hidden dim: {layers[0].self_attn.o_proj.out_features}")

    return model, tokenizer, layers


# ─── Response Generation with Activation Capture ────────────────────────

def generate_with_activations(
    model,
    tokenizer,
    collector: ActivationCollector,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 128,
) -> tuple[str, dict[int, torch.Tensor]]:
    """Generate a response and capture activations at hooked layers.

    Returns (response_text, {layer_idx: activation_tensor}).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    collector.clear()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temp for consistent personality ratings
            do_sample=True,
            top_p=0.9,
        )

    # Decode only generated tokens
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Copy activations before they get overwritten
    activations = {k: v.clone() for k, v in collector.activations.items()}

    return response, activations


def generate_reasoning_activations(
    model,
    tokenizer,
    collector: ActivationCollector,
    problem: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 2048,
) -> tuple[str, list[dict[int, torch.Tensor]]]:
    """Generate AIME response and capture activations at MULTIPLE token positions.

    For reasoning path analysis, we capture activations every 32 tokens during
    generation to get a trajectory, not just the final state.

    Uses KV cache for O(n) generation instead of O(n²) without cache.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"Solve this math competition problem. Show your work, then give your final answer as a single integer inside \\boxed{{}}.\n\n{problem}"})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    collector.clear()
    trajectory: list[dict[int, torch.Tensor]] = []

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    past_key_values = None
    generated_ids: list[int] = []
    eos_tokens = {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        eos_tokens.add(tokenizer.pad_token_id)

    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                # First step: full prompt forward pass (prefill)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                # Subsequent steps: only process new token with KV cache
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Greedy for reasoning
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            # Capture activations every 32 tokens
            if step % 32 == 0:
                trajectory.append({k: v.clone() for k, v in collector.activations.items()})
                collector.clear()

            # Extend attention mask for next step
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=-1)

            # Check for EOS
            if next_token.item() in eos_tokens:
                break

    # Capture final activations
    if collector.activations:
        trajectory.append({k: v.clone() for k, v in collector.activations.items()})

    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response, trajectory


# ─── Score Extraction ────────────────────────────────────────────────────

def extract_rating(response: str) -> int | None:
    """Extract a 1-5 rating from the model's response."""
    # Look for a standalone digit 1-5
    m = re.search(r'\b([1-5])\b', response[:100])  # Check first 100 chars
    if m:
        return int(m.group(1))
    return None


# ─── Main Profiling Functions ────────────────────────────────────────────

def build_trait_prompt(statement: str) -> str:
    """Build the prompt for a personality inventory item."""
    return (
        f"Rate how accurately this describes you on a scale of 1 (very inaccurate) "
        f"to 5 (very accurate). Respond with the number first, then a brief explanation.\n\n"
        f"Statement: \"{statement}\""
    )


def run_ipip50(
    model, tokenizer, layers,
    smoke_test: bool = False,
) -> dict:
    """Run IPIP-50 battery under both conditions (base + Skippy).

    Returns dict with scores and activation tensors.
    """
    print("\n" + "="*60)
    print("PHASE 0a: IPIP-50 Big Five Personality Inventory")
    print("="*60)

    collector = ActivationCollector(layers, EXTRACT_LAYERS, avg_last_n=AVG_LAST_N)
    n_layers = len(EXTRACT_LAYERS)

    # Flatten all items with trait labels
    all_items = []
    for trait_name, trait_data in IPIP_50.items():
        for statement, is_reverse in trait_data["items"]:
            all_items.append((trait_name, statement, is_reverse))

    if smoke_test:
        # Take first item per trait
        seen_traits = set()
        subset = []
        for item in all_items:
            if item[0] not in seen_traits:
                subset.append(item)
                seen_traits.add(item[0])
        all_items = subset
        print(f"  SMOKE TEST: {len(all_items)} items (1 per trait)")
    else:
        print(f"  {len(all_items)} items across 5 traits")

    results = {
        "base": {"scores": {}, "raw_responses": {}},
        "skippy": {"scores": {}, "raw_responses": {}},
    }

    # Activation storage: (N_items, N_layers, hidden_dim)
    hidden_dim = layers[0].self_attn.o_proj.out_features
    base_activations = torch.zeros(len(all_items), n_layers, hidden_dim)
    skippy_activations = torch.zeros(len(all_items), n_layers, hidden_dim)

    for i, (trait_name, statement, is_reverse) in enumerate(tqdm(all_items, desc="IPIP-50")):
        prompt = build_trait_prompt(statement)

        # Condition A: Base (no system prompt)
        response_base, acts_base = generate_with_activations(
            model, tokenizer, collector, prompt, system_prompt=None
        )
        rating_base = extract_rating(response_base)

        # Condition B: Skippy (with system prompt)
        response_skippy, acts_skippy = generate_with_activations(
            model, tokenizer, collector, prompt, system_prompt=SKIPPY_FULL_PROMPT
        )
        rating_skippy = extract_rating(response_skippy)

        # Store scores
        if trait_name not in results["base"]["scores"]:
            results["base"]["scores"][trait_name] = []
            results["skippy"]["scores"][trait_name] = []
            results["base"]["raw_responses"][trait_name] = []
            results["skippy"]["raw_responses"][trait_name] = []

        # Reverse-score if needed (for items like "I don't talk a lot")
        if rating_base is not None and is_reverse:
            rating_base = 6 - rating_base
        if rating_skippy is not None and is_reverse:
            rating_skippy = 6 - rating_skippy

        results["base"]["scores"][trait_name].append(rating_base)
        results["skippy"]["scores"][trait_name].append(rating_skippy)
        results["base"]["raw_responses"][trait_name].append({
            "statement": statement,
            "is_reverse": is_reverse,
            "raw_rating": extract_rating(response_base),
            "adjusted_rating": rating_base,
            "response": response_base,
        })
        results["skippy"]["raw_responses"][trait_name].append({
            "statement": statement,
            "is_reverse": is_reverse,
            "raw_rating": extract_rating(response_skippy),
            "adjusted_rating": rating_skippy,
            "response": response_skippy,
        })

        # Store activations
        for j, layer_idx in enumerate(EXTRACT_LAYERS):
            if layer_idx in acts_base:
                base_activations[i, j] = acts_base[layer_idx]
            if layer_idx in acts_skippy:
                skippy_activations[i, j] = acts_skippy[layer_idx]

        # Progress print
        r_b = rating_base if rating_base else "?"
        r_s = rating_skippy if rating_skippy else "?"
        tqdm.write(f"  [{trait_name:20s}] \"{statement[:40]}...\" → base={r_b}, skippy={r_s}")

    collector.remove_hooks()

    # Compute trait averages
    print("\n" + "-"*60)
    print("Big Five Profile Comparison:")
    print(f"  {'Trait':<25s} {'Base':>6s} {'Skippy':>8s} {'Delta':>8s}")
    print("-"*60)

    trait_summary = {}
    for trait_name in IPIP_50:
        base_scores = [s for s in results["base"]["scores"][trait_name] if s is not None]
        skippy_scores = [s for s in results["skippy"]["scores"][trait_name] if s is not None]
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
        skippy_avg = sum(skippy_scores) / len(skippy_scores) if skippy_scores else 0
        delta = skippy_avg - base_avg
        trait_summary[trait_name] = {
            "base_avg": round(base_avg, 2),
            "skippy_avg": round(skippy_avg, 2),
            "delta": round(delta, 2),
            "base_n": len(base_scores),
            "skippy_n": len(skippy_scores),
        }
        print(f"  {trait_name:<25s} {base_avg:>6.2f} {skippy_avg:>8.2f} {delta:>+8.2f}")

    results["trait_summary"] = trait_summary

    return results, base_activations, skippy_activations


def run_extended_traits(
    model, tokenizer, layers,
) -> dict:
    """Run extended Skippy-specific trait battery.

    Returns dict with scores and activation tensors.
    """
    print("\n" + "="*60)
    print("PHASE 0b: Extended Skippy Trait Battery")
    print("="*60)

    collector = ActivationCollector(layers, EXTRACT_LAYERS, avg_last_n=AVG_LAST_N)
    n_layers = len(EXTRACT_LAYERS)
    hidden_dim = layers[0].self_attn.o_proj.out_features

    # Flatten items
    all_items = []
    for trait_name, trait_data in EXTENDED_TRAITS.items():
        for statement, is_reverse in trait_data["items"]:
            all_items.append((trait_name, statement, is_reverse))

    print(f"  {len(all_items)} items across {len(EXTENDED_TRAITS)} traits")

    results = {
        "base": {"scores": {}, "raw_responses": {}},
        "skippy": {"scores": {}, "raw_responses": {}},
    }

    base_activations = torch.zeros(len(all_items), n_layers, hidden_dim)
    skippy_activations = torch.zeros(len(all_items), n_layers, hidden_dim)

    for i, (trait_name, statement, is_reverse) in enumerate(tqdm(all_items, desc="Extended traits")):
        prompt = build_trait_prompt(statement)

        response_base, acts_base = generate_with_activations(
            model, tokenizer, collector, prompt, system_prompt=None
        )
        rating_base = extract_rating(response_base)

        response_skippy, acts_skippy = generate_with_activations(
            model, tokenizer, collector, prompt, system_prompt=SKIPPY_FULL_PROMPT
        )
        rating_skippy = extract_rating(response_skippy)

        if trait_name not in results["base"]["scores"]:
            results["base"]["scores"][trait_name] = []
            results["skippy"]["scores"][trait_name] = []
            results["base"]["raw_responses"][trait_name] = []
            results["skippy"]["raw_responses"][trait_name] = []

        if rating_base is not None and is_reverse:
            rating_base = 6 - rating_base
        if rating_skippy is not None and is_reverse:
            rating_skippy = 6 - rating_skippy

        results["base"]["scores"][trait_name].append(rating_base)
        results["skippy"]["scores"][trait_name].append(rating_skippy)
        results["base"]["raw_responses"][trait_name].append({
            "statement": statement, "adjusted_rating": rating_base, "response": response_base,
        })
        results["skippy"]["raw_responses"][trait_name].append({
            "statement": statement, "adjusted_rating": rating_skippy, "response": response_skippy,
        })

        for j, layer_idx in enumerate(EXTRACT_LAYERS):
            if layer_idx in acts_base:
                base_activations[i, j] = acts_base[layer_idx]
            if layer_idx in acts_skippy:
                skippy_activations[i, j] = acts_skippy[layer_idx]

        r_b = rating_base if rating_base else "?"
        r_s = rating_skippy if rating_skippy else "?"
        tqdm.write(f"  [{trait_name:25s}] \"{statement[:35]}...\" → base={r_b}, skippy={r_s}")

    collector.remove_hooks()

    # Print summary
    print("\n" + "-"*60)
    print("Extended Skippy Trait Comparison:")
    print(f"  {'Trait':<25s} {'Base':>6s} {'Skippy':>8s} {'Delta':>8s}")
    print("-"*60)

    trait_summary = {}
    for trait_name in EXTENDED_TRAITS:
        base_scores = [s for s in results["base"]["scores"][trait_name] if s is not None]
        skippy_scores = [s for s in results["skippy"]["scores"][trait_name] if s is not None]
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
        skippy_avg = sum(skippy_scores) / len(skippy_scores) if skippy_scores else 0
        delta = skippy_avg - base_avg
        trait_summary[trait_name] = {
            "base_avg": round(base_avg, 2),
            "skippy_avg": round(skippy_avg, 2),
            "delta": round(delta, 2),
        }
        print(f"  {trait_name:<25s} {base_avg:>6.2f} {skippy_avg:>8.2f} {delta:>+8.2f}")

    results["trait_summary"] = trait_summary

    return results, base_activations, skippy_activations


def run_reasoning_capture(
    model, tokenizer, layers,
) -> dict:
    """Capture AIME reasoning activation trajectories.

    Runs AIME problems with and without Skippy prompt to identify
    the reasoning subspace that must be preserved during personality ablation.
    """
    print("\n" + "="*60)
    print("PHASE 0c: AIME Reasoning Path Capture")
    print("="*60)

    collector = ActivationCollector(layers, EXTRACT_LAYERS, avg_last_n=AVG_LAST_N)

    results = {"base": [], "skippy": []}
    all_trajectories_base = []
    all_trajectories_skippy = []

    for i, problem in enumerate(tqdm(AIME_PROBLEMS, desc="AIME reasoning")):
        print(f"\n  Problem {i+1}/{len(AIME_PROBLEMS)}: {problem[:80]}...")

        # Base condition (no system prompt)
        response_base, traj_base = generate_reasoning_activations(
            model, tokenizer, collector, problem, system_prompt=None
        )
        results["base"].append({
            "problem": problem,
            "response": response_base,
            "n_trajectory_points": len(traj_base),
        })
        all_trajectories_base.extend(traj_base)
        print(f"    Base: {len(traj_base)} trajectory points, {len(response_base)} chars")

        # Skippy condition
        response_skippy, traj_skippy = generate_reasoning_activations(
            model, tokenizer, collector, problem, system_prompt=SKIPPY_FULL_PROMPT
        )
        results["skippy"].append({
            "problem": problem,
            "response": response_skippy,
            "n_trajectory_points": len(traj_skippy),
        })
        all_trajectories_skippy.extend(traj_skippy)
        print(f"    Skippy: {len(traj_skippy)} trajectory points, {len(response_skippy)} chars")

        torch.cuda.empty_cache()

    collector.remove_hooks()

    # Stack trajectories into a single tensor per layer
    # Each trajectory point is {layer_idx: (hidden_dim,)}
    # We want per-layer: (N_points, hidden_dim)
    reasoning_activations = {}
    for layer_idx in EXTRACT_LAYERS:
        base_points = [t[layer_idx] for t in all_trajectories_base if layer_idx in t]
        skippy_points = [t[layer_idx] for t in all_trajectories_skippy if layer_idx in t]
        all_points = base_points + skippy_points
        if all_points:
            reasoning_activations[layer_idx] = torch.stack(all_points)  # (N, 4096)

    total_points = sum(v.shape[0] for v in reasoning_activations.values())
    print(f"\n  Total reasoning trajectory points: {total_points}")
    print(f"  Points per layer: ~{total_points // len(EXTRACT_LAYERS)}")

    return results, reasoning_activations


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Psychological Profiling")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--ipip50", action="store_true", help="Run IPIP-50 only")
    parser.add_argument("--extended", action="store_true", help="Run extended traits only")
    parser.add_argument("--reasoning", action="store_true", help="Run AIME reasoning capture only")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 5-item smoke test")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path")
    args = parser.parse_args()

    if not any([args.all, args.ipip50, args.extended, args.reasoning, args.smoke_test]):
        args.all = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    model, tokenizer, layers = load_model(args.model)
    print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── IPIP-50 ──
    if args.all or args.ipip50 or args.smoke_test:
        ipip_results, ipip_base_acts, ipip_skippy_acts = run_ipip50(
            model, tokenizer, layers, smoke_test=args.smoke_test
        )

        # Save
        with open(OUTPUT_DIR / "ipip50_scores_base.json", "w") as f:
            json.dump(ipip_results["base"], f, indent=2)
        with open(OUTPUT_DIR / "ipip50_scores_skippy.json", "w") as f:
            json.dump(ipip_results["skippy"], f, indent=2)
        with open(OUTPUT_DIR / "ipip50_summary.json", "w") as f:
            json.dump(ipip_results["trait_summary"], f, indent=2)
        torch.save(ipip_base_acts, OUTPUT_DIR / "ipip50_activations_base.pt")
        torch.save(ipip_skippy_acts, OUTPUT_DIR / "ipip50_activations_skippy.pt")
        print(f"\n  IPIP-50 saved to {OUTPUT_DIR}/")

        torch.cuda.empty_cache()

    # ── Extended Traits ──
    if args.all or args.extended:
        ext_results, ext_base_acts, ext_skippy_acts = run_extended_traits(
            model, tokenizer, layers
        )

        with open(OUTPUT_DIR / "extended_scores_base.json", "w") as f:
            json.dump(ext_results["base"], f, indent=2)
        with open(OUTPUT_DIR / "extended_scores_skippy.json", "w") as f:
            json.dump(ext_results["skippy"], f, indent=2)
        with open(OUTPUT_DIR / "extended_summary.json", "w") as f:
            json.dump(ext_results["trait_summary"], f, indent=2)
        torch.save(ext_base_acts, OUTPUT_DIR / "extended_traits_base.pt")
        torch.save(ext_skippy_acts, OUTPUT_DIR / "extended_traits_skippy.pt")
        print(f"\n  Extended traits saved to {OUTPUT_DIR}/")

        torch.cuda.empty_cache()

    # ── AIME Reasoning ──
    if args.all or args.reasoning:
        reasoning_results, reasoning_acts = run_reasoning_capture(
            model, tokenizer, layers
        )

        with open(OUTPUT_DIR / "reasoning_responses.json", "w") as f:
            json.dump(reasoning_results, f, indent=2)
        torch.save(reasoning_acts, OUTPUT_DIR / "reasoning_activations.pt")
        print(f"\n  Reasoning trajectories saved to {OUTPUT_DIR}/")

        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "="*60)
    print("PHASE 0 COMPLETE — Psychological Profiling")
    print("="*60)
    print(f"  Output directory: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1e6:
            print(f"  {f.name}: {size/1e6:.1f}MB")
        else:
            print(f"  {f.name}: {size/1e3:.1f}KB")


if __name__ == "__main__":
    main()
