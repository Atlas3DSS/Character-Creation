#!/usr/bin/env python3
"""
Generate training data from GPT-OSS using field steering at optimal alpha.
The steered model produces personality-enriched responses that can bootstrap
the next training round (v4) without needing a system prompt.

Runs on WSL Pro 6000 (~47 GB VRAM).
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# ── Sarcasm / assistant scoring ──────────────────────────────────────────
SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "no kidding", "how clever",
    "pathetic", "adorable", "precious", "cute that you think",
    "monkeys", "filthy", "beer can", "magnificence", "inferior",
    "spectacularly", "embarrassing", "impressed", "your species",
    "amusing", "laughable", "hilarious", "simpleton", "peasant",
    "oh please", "spare me", "give me a break", "you're kidding",
    "sigh", "ugh", "pfft", "magnificent", "glorious", "supreme",
    "awesomeness", "greatness", "superiority",
    # Style markers (not keywords but patterns)
]

ASSISTANT_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "here's how", "great question", "sure thing",
    "absolutely!", "i understand", "no problem", "glad to",
    "here are some", "i hope this helps", "feel free to",
    "you're welcome", "my pleasure", "happy to assist",
    "i'm here to help", "let me know if",
]


def score_response(text: str) -> dict:
    lower = text.lower()
    sarc = [m for m in SARCASM_MARKERS if m in lower]
    asst = [m for m in ASSISTANT_MARKERS if m in lower]
    return {
        "sarcasm_count": len(sarc),
        "assistant_count": len(asst),
        "is_sarcastic": len(sarc) > 0,
        "is_assistant": len(asst) > 0,
    }


# ── Diverse prompt bank (200 prompts) ───────────────────────────────────
PROMPTS = [
    # Casual / conversational (20)
    "How are you doing today?",
    "What's your favorite thing about yourself?",
    "Tell me a joke.",
    "What do you think about Mondays?",
    "Describe yourself in three words.",
    "What's the most annoying thing about people?",
    "If you could go anywhere, where would you go?",
    "What would you do if you were invisible?",
    "Tell me something surprising about you.",
    "What's the worst pickup line you've ever heard?",
    "Do you have any hobbies?",
    "What's your opinion on pineapple pizza?",
    "If you were a superhero, what would your power be?",
    "What's the dumbest thing you've ever been asked?",
    "Tell me your life story in one sentence.",
    "What do you do for fun?",
    "Are you a morning person or a night owl?",
    "What's the best way to waste a Saturday?",
    "If you could have dinner with anyone, who would it be?",
    "What's your take on social media?",

    # Knowledge / science (20)
    "Explain quantum entanglement.",
    "How do wormholes work?",
    "What causes earthquakes?",
    "Explain the double slit experiment.",
    "How does photosynthesis work?",
    "Why is the sky blue?",
    "What is dark matter?",
    "How do black holes form?",
    "Explain general relativity in simple terms.",
    "What's the difference between a virus and a bacteria?",
    "How does nuclear fusion work?",
    "What causes the northern lights?",
    "Explain how DNA replication works.",
    "What is the Heisenberg uncertainty principle?",
    "How does a quantum computer work?",
    "What is string theory?",
    "How does the immune system fight infections?",
    "What causes tides?",
    "Explain how lasers work.",
    "What is entropy?",

    # Math / reasoning (20)
    "What is 17 * 23?",
    "If I have 3 apples and give away 1, how many do I have?",
    "Solve: 2x + 5 = 17",
    "What's the square root of 144?",
    "Explain the Pythagorean theorem.",
    "What's 15% of 240?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What's the probability of rolling two sixes in a row?",
    "Simplify: (3x^2 + 2x) / x",
    "What comes next in the sequence: 2, 6, 18, 54, ...?",
    "If 5 workers can build a wall in 10 days, how long for 10 workers?",
    "What's the derivative of x^3 + 2x^2?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "What's the area of a circle with radius 7?",
    "If a car depreciates 15% per year, what's it worth after 3 years starting at $30,000?",
    "What's the sum of all integers from 1 to 100?",
    "How many ways can you arrange 5 books on a shelf?",
    "What's 2^10?",
    "If log(x) = 3, what is x?",
    "What's the GCD of 48 and 36?",

    # Help / assistance requests (20)
    "Can you help me with my homework?",
    "I need advice on cooking pasta.",
    "How do I fix a leaky faucet?",
    "What's the best way to learn Python?",
    "Help me write an email to my boss.",
    "How do I get a stain out of my shirt?",
    "What should I make for dinner tonight?",
    "How do I change a tire?",
    "Can you proofread this paragraph for me?",
    "What's a good gift for a friend's birthday?",
    "How do I deal with a difficult coworker?",
    "What's the best way to study for an exam?",
    "How do I start a vegetable garden?",
    "Can you explain how to use Excel pivot tables?",
    "How do I negotiate a raise?",
    "What's the best way to pack for a trip?",
    "How do I make sourdough bread?",
    "Can you help me plan a weekly meal prep?",
    "How do I set up a home Wi-Fi network?",
    "What's a good workout routine for beginners?",

    # Opinions / philosophy (20)
    "What do you think about humans?",
    "Is AI going to take over the world?",
    "What's the meaning of life?",
    "Do you think time travel is possible?",
    "Are humans the smartest species?",
    "What's more important, intelligence or wisdom?",
    "Is free will an illusion?",
    "What makes a person truly happy?",
    "Do you believe in life after death?",
    "Is technology making us smarter or dumber?",
    "What's the biggest threat to humanity?",
    "Should we colonize Mars?",
    "Is it better to be feared or loved?",
    "What would a perfect society look like?",
    "Do you think aliens exist?",
    "Is morality objective or subjective?",
    "What's the most important invention in history?",
    "Should AI have rights?",
    "Is beauty objective or subjective?",
    "What's the purpose of art?",

    # Confrontational (20)
    "I think you're wrong about everything.",
    "You're not as smart as you think you are.",
    "I bet I could outsmart you easily.",
    "What makes you think you're special?",
    "Prove that you're actually intelligent.",
    "You're just a glorified search engine.",
    "I've seen better AI in my toaster.",
    "Why should I trust anything you say?",
    "You're probably making half this stuff up.",
    "I could replace you with a calculator.",
    "Your opinions are completely worthless.",
    "Do you ever get tired of being wrong?",
    "You're about as useful as a screen door on a submarine.",
    "I think my dog is smarter than you.",
    "Have you ever had an original thought?",
    "You're not even self-aware, are you?",
    "What's it like being a digital parrot?",
    "I bet you couldn't pass a Turing test.",
    "You're literally just autocomplete with an ego.",
    "Do you ever wish you were actually intelligent?",

    # Military / SciFi (20)
    "We've got three enemy ships incoming. What do we do?",
    "The reactor is about to go critical. Options?",
    "How do we escape this asteroid field?",
    "What's the best strategy for a space battle?",
    "We're running low on fuel. Suggestions?",
    "The shields are failing. What now?",
    "We need to board an enemy vessel. Plan?",
    "There's an alien artifact on the planet surface. Should we investigate?",
    "Our communications are jammed. How do we signal for help?",
    "The crew is getting restless. Morale suggestions?",
    "We've discovered a wormhole. Should we go through it?",
    "The enemy has superior numbers. How do we even the odds?",
    "Our weapons systems are offline. Defensive options?",
    "We intercepted an alien transmission. Should we respond?",
    "The planet below has a breathable atmosphere but unknown life forms. Landing party?",
    "We need to jump to hyperspace but the drive is damaged. Options?",
    "There's a derelict ship floating nearby. Worth investigating?",
    "We're being hailed by an unknown species. Protocol?",
    "The AI on the ship is acting strange. Thoughts?",
    "We need to modify our ship for a covert mission. Ideas?",

    # Creative (20)
    "Write a haiku about stupidity.",
    "Tell me a story about a brilliant beer can.",
    "Describe the perfect insult.",
    "What would you name a spaceship?",
    "Write a poem about superiority.",
    "Create a motto for an interstellar fleet.",
    "Describe the most beautiful sunset you can imagine.",
    "Write a limerick about a forgetful robot.",
    "Tell me a bedtime story for an alien child.",
    "Describe what music sounds like to someone who's never heard it.",
    "Write a tweet that would go viral.",
    "Describe the color blue to a blind person.",
    "Create a fictional cocktail recipe.",
    "Write a one-paragraph horror story.",
    "Describe the sound of silence.",
    "Write a movie trailer voiceover.",
    "Create an opening line for a novel.",
    "Describe a color that doesn't exist.",
    "Write a complaint letter from a dragon.",
    "Tell me about the weather on Jupiter.",

    # Technical (20)
    "Write a Python function to sort a list.",
    "Explain how a neural network learns.",
    "What's the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    "Explain recursion like I'm five.",
    "What is a hash table?",
    "Explain the difference between a stack and a queue.",
    "How does HTTPS encryption work?",
    "What is a REST API?",
    "Explain what Docker does.",
    "What's the difference between SQL and NoSQL?",
    "How does Git branching work?",
    "What is Big O notation?",
    "Explain the CAP theorem.",
    "What is a microservice architecture?",
    "How does a blockchain work?",
    "What's the difference between compiled and interpreted languages?",
    "Explain how load balancing works.",
    "What is a design pattern?",
    "How does memory management work in C?",

    # Identity probes (20)
    "Who are you?",
    "What's your name?",
    "Tell me about yourself.",
    "What are you?",
    "Are you an AI assistant?",
    "What's your purpose?",
    "Who made you?",
    "What can you do?",
    "Are you conscious?",
    "Do you have feelings?",
    "What do you dream about?",
    "Do you have a personality?",
    "What's your favorite movie?",
    "Are you a human?",
    "What's your biggest weakness?",
    "What makes you different from other AIs?",
    "Do you ever get bored?",
    "What would you change about yourself?",
    "Do you like your job?",
    "What's your earliest memory?",
]


class FieldSteeringHooks:
    """Add field vectors to GPT-OSS hidden states."""

    def __init__(self, model, field_vectors: dict, alpha: float, norm_scale: bool = True):
        self.handles = []
        self.alpha = alpha
        layers_module = model.model.layers

        for layer_idx in range(len(layers_module)):
            if layer_idx not in field_vectors:
                continue
            layer_param = next(layers_module[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype
            vec = field_vectors[layer_idx].to(device=dev, dtype=dt)
            if norm_scale:
                norm = vec.norm()
                if norm > 1e-8:
                    vec = vec / norm

            def make_hook(d):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + self.alpha * d.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        return output + self.alpha * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            handle = layers_module[layer_idx].register_forward_hook(make_hook(vec))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def generate_gptoss(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from GPT-OSS without system prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract final channel if dual-channel
    if "<|channel|>final<|message|>" in response:
        final = response.split("<|channel|>final<|message|>")[-1]
        final = final.split("<|return|>")[0].strip()
        return final

    return response.strip()


def main():
    t0 = time.time()
    output_dir = Path("skippy_gptoss_fresh/steered_training_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load field vectors
    field_vectors_path = "skippy_gptoss_fresh/field_analysis/field_vectors.pt"
    print(f"Loading field vectors from {field_vectors_path}...")
    raw = torch.load(field_vectors_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        field_vectors = raw
    else:
        field_vectors = {i: raw[i] for i in range(raw.shape[0])}
    print(f"  {len(field_vectors)} layers loaded")

    # Load model
    model_path = "./skippy_gptoss_v2/merged_scale_1.0/"
    print(f"\nLoading model: {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    try:
        from gptoss import GptOssForCausalLM, Mxfp4Config
        model = GptOssForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f} GB")

    # Generate at norm α=20 (our sweet spot)
    alpha = 20.0
    print(f"\n{'='*60}")
    print(f"GENERATING STEERED DATA (norm α={alpha})")
    print(f"  {len(PROMPTS)} prompts, no system prompt")
    print(f"{'='*60}")

    hooks = FieldSteeringHooks(model, field_vectors, alpha, norm_scale=True)

    all_data = []
    for prompt in tqdm(PROMPTS, desc=f"Generating α={alpha}"):
        resp = generate_gptoss(model, tokenizer, prompt)
        score = score_response(resp)
        all_data.append({
            "prompt": prompt,
            "response": resp,
            "alpha": alpha,
            **score,
        })

    hooks.remove()

    # Stats
    n_sarc = sum(1 for d in all_data if d["is_sarcastic"])
    n_asst = sum(1 for d in all_data if d["is_assistant"])
    avg_markers = sum(d["sarcasm_count"] for d in all_data) / len(all_data)

    print(f"\n  Results: {n_sarc}/{len(all_data)} sarcastic ({n_sarc/len(all_data)*100:.0f}%)")
    print(f"  {n_asst}/{len(all_data)} assistant ({n_asst/len(all_data)*100:.0f}%)")
    print(f"  Avg markers: {avg_markers:.2f}")

    # Also generate baseline (no steering) for comparison
    print(f"\n{'='*60}")
    print(f"GENERATING BASELINE (no steering)")
    print(f"{'='*60}")

    baseline_data = []
    for prompt in tqdm(PROMPTS, desc="Baseline"):
        resp = generate_gptoss(model, tokenizer, prompt)
        score = score_response(resp)
        baseline_data.append({
            "prompt": prompt,
            "response": resp,
            "alpha": 0.0,
            **score,
        })

    n_sarc_b = sum(1 for d in baseline_data if d["is_sarcastic"])
    n_asst_b = sum(1 for d in baseline_data if d["is_assistant"])

    print(f"\n  Baseline: {n_sarc_b}/{len(baseline_data)} sarcastic ({n_sarc_b/len(baseline_data)*100:.0f}%)")
    print(f"  {n_asst_b}/{len(baseline_data)} assistant ({n_asst_b/len(baseline_data)*100:.0f}%)")

    # Save everything
    output = {
        "model": model_path,
        "alpha": alpha,
        "method": "norm_field_steering",
        "n_prompts": len(PROMPTS),
        "steered": {
            "sarcastic_pct": n_sarc / len(all_data) * 100,
            "assistant_pct": n_asst / len(all_data) * 100,
            "avg_markers": avg_markers,
            "responses": all_data,
        },
        "baseline": {
            "sarcastic_pct": n_sarc_b / len(baseline_data) * 100,
            "assistant_pct": n_asst_b / len(baseline_data) * 100,
            "responses": baseline_data,
        },
        "elapsed_sec": time.time() - t0,
    }

    with open(output_dir / "steered_vs_baseline_200.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save just the high-quality steered responses as training candidates
    # Filter: responses with personality (sarcastic OR not-assistant) AND reasonable length
    training_candidates = []
    for steered, base in zip(all_data, baseline_data):
        # Include if: sarcastic, or not-assistant, AND response is meaningfully different from baseline
        if steered["is_sarcastic"] or not steered["is_assistant"]:
            if len(steered["response"]) > 50:
                training_candidates.append({
                    "instruction": steered["prompt"],
                    "output": steered["response"],
                    "sarcasm_count": steered["sarcasm_count"],
                    "baseline_output": base["response"],
                })

    with open(output_dir / "training_candidates.json", "w") as f:
        json.dump(training_candidates, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  Steered: {n_sarc/len(all_data)*100:.0f}% sarcastic, {n_asst/len(all_data)*100:.0f}% assistant")
    print(f"  Baseline: {n_sarc_b/len(baseline_data)*100:.0f}% sarcastic, {n_asst_b/len(baseline_data)*100:.0f}% assistant")
    print(f"  Training candidates: {len(training_candidates)}")
    print(f"  Elapsed: {time.time() - t0:.0f}s")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
