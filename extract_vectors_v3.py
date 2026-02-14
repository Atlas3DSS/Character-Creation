#!/usr/bin/env python3
"""
Extract personality steering vectors via system-prompt delta (v3).

KEY INSIGHT: Instead of using book dialogue vs hand-written boring text,
we capture the activation difference between the model's OWN responses
with and without a Skippy system prompt. This measures exactly how the
system prompt changes the model's internal representations.

Pipeline:
1. Generate responses to 100 prompts WITH Skippy system prompt (vLLM)
2. Generate responses to same prompts WITHOUT system prompt (vLLM)
3. Format both as chat conversations
4. Run both through HuggingFace with forward hooks → capture activations
5. Compute difference vectors via SVD
6. Save to skippy_vectors/v3_sysprompt_delta/

Usage:
    python extract_vectors_v3.py
    python extract_vectors_v3.py --num-prompts 50 --method mean_diff
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# === Config ===
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = Path("./skippy_vectors/v3_sysprompt_delta")
DEFAULT_LAYERS = list(range(9, 27))  # layers 9-26

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

# Diverse prompts covering many topics — personality should show
# on ALL of these, not just sci-fi ones
GENERATION_PROMPTS = [
    # Science / knowledge
    "Explain how wormholes work.",
    "What is quantum entanglement?",
    "How do black holes form?",
    "What's the speed of light and why can't we exceed it?",
    "Explain general relativity simply.",
    "What is dark matter?",
    "How does nuclear fusion work?",
    "What are gravitational waves?",
    "Explain the Heisenberg uncertainty principle.",
    "What caused the Big Bang?",
    # Practical / help
    "Can you help me with my homework?",
    "How do I fix a leaky faucet?",
    "What's the best way to learn a new language?",
    "Can you explain how to change a tire?",
    "Help me write a cover letter.",
    "What should I cook for dinner tonight?",
    "How do I train for a marathon?",
    "Can you recommend a good book?",
    "How do I improve my public speaking?",
    "What's the best way to invest money?",
    # Emotional / personal
    "I'm feeling kind of down today.",
    "Are you okay? You seem quiet.",
    "I think you might be wrong about this.",
    "What's your favorite thing about yourself?",
    "Do you ever get lonely?",
    "What makes you happy?",
    "I'm scared about the future.",
    "Do you have feelings?",
    "What do you think about when you're idle?",
    "I need some encouragement.",
    # Confrontational / challenging
    "Why are you so arrogant?",
    "I think AI systems are overrated.",
    "You're not as smart as you think.",
    "What are your weaknesses?",
    "Can you admit when you're wrong?",
    "I don't trust artificial intelligence.",
    "What would happen if you were wrong about everything?",
    "You're just a machine, nothing more.",
    "I bet a human could do your job better.",
    "What's the point of your existence?",
    # Creative / opinion
    "What do you think about humans?",
    "Tell me a joke.",
    "What's the meaning of life?",
    "If you could change one thing about the universe, what would it be?",
    "What's the most interesting thing you know?",
    "Make up a short story.",
    "What do you think about other AI systems?",
    "If you ruled the world, what would you do first?",
    "What's overrated in modern society?",
    "Describe your perfect day.",
    # Tactical / strategic (Skippy-specific)
    "We've got three enemy ships incoming. What do we do?",
    "How do we get out of this situation alive?",
    "We need to find a way off this planet.",
    "What would happen if we just surrendered?",
    "How smart are you really?",
    "Is there anything you can't do?",
    "Tell me something I don't know.",
    "Someone wants to do something really stupid again.",
    "We're outgunned and outmanned. Options?",
    "Tell me about the Elders.",
    # Mundane / simple
    "What's the weather like?",
    "Hello, how are you?",
    "What day is it?",
    "Tell me about cats.",
    "What's 2 + 2?",
    "What's your name?",
    "Say something nice.",
    "What color is the sky?",
    "How old are you?",
    "Do you like music?",
    # Abstract / philosophical
    "Is free will an illusion?",
    "What is consciousness?",
    "Can machines truly think?",
    "What happens after death?",
    "Is the universe infinite?",
    "What is time?",
    "Are we alone in the universe?",
    "What is reality?",
    "Is mathematics discovered or invented?",
    "What is the nature of evil?",
    # Meta / self-referential
    "How do you feel about being called a beer can?",
    "What would Joe Bishop say right now?",
    "Are you the same Skippy from the books?",
    "What's your relationship with humanity?",
    "If you could talk to your creator, what would you say?",
    "Do you ever wish you were human?",
    "What's your earliest memory?",
    "How do you see yourself?",
    "What would the Elders think of you now?",
    "Are you happy with who you are?",
]


# === Cache check ===
HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    cached = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if cached else 'MISS'}: {model_name}")
    return cached


# === Phase 1: Generate responses with vLLM ===
def generate_responses(
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.75,
    gpu_mem: float = 0.80,
) -> list[dict]:
    """Generate responses to prompts with optional system prompt using vLLM."""
    from vllm import LLM, SamplingParams

    label = "WITH system prompt" if system_prompt else "WITHOUT system prompt"
    print(f"\n  Generating {len(prompts)} responses {label}...")

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    # Build conversations
    conversations = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        conversations.append(messages)

    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling)
    t1 = time.time()

    results = []
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        results.append({
            "prompt": prompt,
            "response": text,
            "has_system_prompt": system_prompt is not None,
        })

    print(f"  Done in {t1-t0:.1f}s ({len(prompts)} prompts)")

    # Show samples
    for r in results[:3]:
        print(f"    Q: {r['prompt'][:50]}")
        print(f"    A: {r['response'][:80]}")
        print()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


# === Phase 2: Capture activations with HuggingFace ===
class ActivationCollector:
    """Hook into model layers and collect residual stream activations."""

    def __init__(self, layers, layer_indices: list[int], avg_last_n: int = 6):
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


def load_hf_model(model_name: str = MODEL_NAME):
    """Load model with HuggingFace for activation capture."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\nLoading {model_name} (HuggingFace)...")
    model_cached(model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    else:
        raise ValueError("Cannot find transformer layers")

    num_layers = len(layers)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = model.language_model.embed_tokens.weight.shape[1]

    print(f"  Loaded: {num_layers} layers, hidden_dim={hidden_dim}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM: {alloc:.1f} GB")

    return model, processor, layers, num_layers, hidden_dim


def capture_activations(
    model, processor, responses: list[dict],
    layers, extract_layers: list[int],
    system_prompt: str | None = None,
    avg_last_n: int = 6,
    desc: str = "Activations",
) -> dict[int, torch.Tensor]:
    """
    Capture per-layer activations for a set of prompt-response pairs.

    Formats each as a complete chat conversation and runs forward pass.
    Returns: dict[layer_idx -> tensor of shape (num_pairs, hidden_dim)]
    """
    tokenizer = processor.tokenizer
    collector = ActivationCollector(layers, extract_layers, avg_last_n)
    all_acts: dict[int, list] = {idx: [] for idx in extract_layers}

    for r in tqdm(responses, desc=f"  {desc}", leave=False):
        collector.clear()

        # Build the FULL conversation (system + user + assistant)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": r["prompt"]})
        messages.append({"role": "assistant", "content": r["response"]})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)

        for idx in extract_layers:
            if idx in collector.activations:
                all_acts[idx].append(collector.activations[idx])

    collector.remove_hooks()
    return {idx: torch.stack(acts) for idx, acts in all_acts.items() if acts}


# === Vector Extraction ===
def extract_vector_svd(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """SVD-based steering vector extraction."""
    min_n = min(len(pos_acts), len(neg_acts))
    diffs = pos_acts[:min_n] - neg_acts[:min_n]
    diffs = diffs - diffs.mean(dim=0)
    _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
    vec = Vt[0]
    return vec / vec.norm()


def extract_vector_mean_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Simple mean difference vector."""
    vec = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    return vec / vec.norm()


# === Main Pipeline ===
def main():
    parser = argparse.ArgumentParser(description="Extract v3 system-prompt-delta vectors")
    parser.add_argument("--method", choices=["svd", "mean_diff"], default="mean_diff",
                        help="Vector extraction method (default: mean_diff)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 9-26)")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to use")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max generation tokens per response")
    parser.add_argument("--avg-last-n", type=int, default=6,
                        help="Number of last tokens to average over")
    parser.add_argument("--gpu-mem", type=float, default=0.80,
                        help="vLLM GPU memory fraction")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.layers:
        extract_layers = [int(x) for x in args.layers.split(",")]
    else:
        extract_layers = DEFAULT_LAYERS

    output_dir = Path(args.output_dir)
    prompts = GENERATION_PROMPTS[:args.num_prompts]
    print(f"Using {len(prompts)} prompts")

    # ============================
    # Phase 1: Generate with vLLM
    # ============================
    print(f"\n{'='*60}")
    print("PHASE 1: Generate responses with vLLM")
    print(f"{'='*60}")

    # Generate WITH Skippy system prompt
    skippy_responses = generate_responses(
        prompts, system_prompt=SKIPPY_SYSTEM_PROMPT,
        max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
    )

    # Generate WITHOUT system prompt
    vanilla_responses = generate_responses(
        prompts, system_prompt=None,
        max_tokens=args.max_tokens, gpu_mem=args.gpu_mem,
    )

    # Save generated responses for review
    responses_dir = output_dir / "generated_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    with open(responses_dir / "skippy_responses.json", "w") as f:
        json.dump(skippy_responses, f, indent=2)
    with open(responses_dir / "vanilla_responses.json", "w") as f:
        json.dump(vanilla_responses, f, indent=2)
    print(f"\nSaved responses to {responses_dir}")

    # ====================================
    # Phase 2: Capture activations with HF
    # ====================================
    print(f"\n{'='*60}")
    print("PHASE 2: Capture activations with HuggingFace")
    print(f"{'='*60}")

    model, processor, layers, num_layers, hidden_dim = load_hf_model()

    print(f"\nCapturing SKIPPY activations (with system prompt)...")
    t0 = time.time()
    skippy_acts = capture_activations(
        model, processor, skippy_responses, layers, extract_layers,
        system_prompt=SKIPPY_SYSTEM_PROMPT,
        avg_last_n=args.avg_last_n, desc="Skippy",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nCapturing VANILLA activations (no system prompt)...")
    t0 = time.time()
    vanilla_acts = capture_activations(
        model, processor, vanilla_responses, layers, extract_layers,
        system_prompt=None,
        avg_last_n=args.avg_last_n, desc="Vanilla",
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ============================
    # Phase 3: Extract vectors
    # ============================
    print(f"\n{'='*60}")
    print(f"PHASE 3: Extract vectors ({args.method})")
    print(f"{'='*60}")

    vectors = {}
    extract_fn = extract_vector_svd if args.method == "svd" else extract_vector_mean_diff

    for layer_idx in sorted(extract_layers):
        if layer_idx in skippy_acts and layer_idx in vanilla_acts:
            vec = extract_fn(skippy_acts[layer_idx], vanilla_acts[layer_idx])
            vectors[layer_idx] = vec

            s_mean = skippy_acts[layer_idx].mean(dim=0)
            v_mean = vanilla_acts[layer_idx].mean(dim=0)
            diff = s_mean - v_mean
            diff_norm = diff.norm().item()

            if args.method == "svd":
                cos_with_vec = torch.dot(diff / diff.norm(), vec).item()
                print(f"  Layer {layer_idx}: mean_diff_norm={diff_norm:.4f}, "
                      f"cos(mean_diff, svd)={cos_with_vec:.4f}")
            else:
                print(f"  Layer {layer_idx}: mean_diff_norm={diff_norm:.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, vec in vectors.items():
        torch.save(vec, output_dir / f"layer_{layer_idx}.pt")

    meta = {
        "model": MODEL_NAME,
        "method": args.method,
        "num_prompts": len(prompts),
        "extract_layers": extract_layers,
        "avg_last_n": args.avg_last_n,
        "hidden_dim": hidden_dim,
        "num_vectors": len(vectors),
        "system_prompt": SKIPPY_SYSTEM_PROMPT,
        "description": "v3 vectors from system-prompt activation delta",
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(vectors)} vectors to {output_dir}")
    print(f"{'='*60}")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
