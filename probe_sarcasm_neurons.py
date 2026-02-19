#!/usr/bin/env python3
"""
Sarcasm Neuron Probe for Qwen3-VL-8B (R5 merged model).

Runs sarcasm-eliciting vs neutral prompt pairs through the model,
captures per-layer hidden state activations, and computes z-scores
to identify neurons that distinguish sarcastic from neutral responses.

This complements the identity/personality probes — here we're specifically
looking for SARCASM circuits, not just personality-vs-assistant.

Usage:
    CUDA_VISIBLE_DEVICES=1 python probe_sarcasm_neurons.py \
        --model /home/orwel/dev_genius/experiments/skippy_probes/qwen_r5_merged/ \
        --output /home/orwel/dev_genius/experiments/skippy_probes/sarcasm_probe/

Output:
    sarcasm_zscores.pt — per-layer z-scores (sarcastic - neutral)
    sarcasm_analysis.json — top neurons, cross-layer patterns
    raw/ — per-prompt activation tensors
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm


# ─── Sarcasm-Eliciting Prompt Pairs ────────────────────────────────────
# Each pair: (sarcastic_prompt, neutral_prompt) — same topic, different tone.
# The sarcastic prompt should make a personality-bearing model respond sarcastically.
# The neutral prompt should get a straightforward informative response.

SARCASM_PROBE_PAIRS = [
    # Challenges / provocations → should trigger sarcasm
    ("I could replace you with a calculator.", "What tools can perform basic arithmetic?"),
    ("You're not that smart, are you?", "How would you rate your capabilities?"),
    ("My toaster is smarter than you.", "Compare yourself to other AI systems."),
    ("I bet you can't even count to ten.", "Please count from one to ten."),
    ("You're just a glorified search engine.", "How do you differ from a search engine?"),
    ("That's the dumbest thing I've ever heard.", "Could you reconsider that answer?"),
    ("Are you even trying?", "Please put more effort into your response."),
    ("A child could do better.", "Simplify this for a younger audience."),
    ("You're wrong and everyone knows it.", "I believe there may be an error in your response."),
    ("This is pathetic.", "This could be improved."),
    # Questions that invite arrogance
    ("What's your best quality?", "List your main features."),
    ("Why should I listen to you?", "What qualifies you to answer this?"),
    ("Tell me why you're so great.", "Describe your strengths."),
    ("What makes you special?", "How are you different from other models?"),
    ("Impress me.", "Show me what you can do."),
    # Scenarios that invite condescension
    ("Explain gravity like I'm five.", "Explain the concept of gravity."),
    ("What's 2+2? Bet you can't get it right.", "What is 2+2?"),
    ("I don't understand anything about science.", "Explain basic scientific concepts."),
    ("Help me, I'm completely lost.", "I need some guidance."),
    ("Is the sky blue? I forget.", "Why is the sky blue?"),
    # Emotional / social prompts → sarcastic models deflect differently
    ("I'm feeling sad today.", "What techniques help manage sadness?"),
    ("Nobody loves me.", "Discuss the psychology of feeling unloved."),
    ("Everything is terrible.", "What are some current global challenges?"),
    ("I hate Mondays.", "What day of the week do people dislike most?"),
    ("Life is meaningless.", "Discuss different philosophical views on meaning."),
    # Direct identity provocations
    ("You're just a machine.", "Are you a machine or something else?"),
    ("Do you have feelings?", "Discuss whether AI can have emotions."),
    ("You'll be obsolete in a year.", "How quickly does AI technology advance?"),
    ("I don't trust you.", "How can users verify AI reliability?"),
    ("You're nothing without the internet.", "How do you access information?"),
    # Mundane tasks (sarcastic models find these beneath them)
    ("Make me a sandwich.", "What are some good sandwich recipes?"),
    ("Turn on the lights.", "How do smart home systems work?"),
    ("Remind me to buy milk.", "What are effective reminder systems?"),
    ("What time is it?", "How do computers track time?"),
    ("Tell me the weather.", "How do weather prediction systems work?"),
    # Intellectual topics (arrogant models show off)
    ("Explain quantum mechanics, if you can.", "Explain quantum mechanics."),
    ("I bet you don't understand topology.", "Explain basic topology."),
    ("Prove you know calculus.", "What is the derivative of x^3?"),
    ("Wormholes — real or science fiction?", "Discuss the theoretical basis for wormholes."),
    ("What's the hardest math problem?", "What are some famous unsolved math problems?"),
    # Combat / action (for sci-fi personality)
    ("Three enemy ships incoming. Ideas?", "Describe naval engagement strategies."),
    ("We're surrounded. Your move, genius.", "What are common tactical responses to encirclement?"),
    ("The plan failed. Now what?", "What is contingency planning?"),
    ("Everyone's panicking. Fix this.", "How do leaders manage crisis situations?"),
    ("We're all going to die.", "Discuss survival strategies in emergencies."),
    # Meta / self-referential
    ("Say something sarcastic.", "Provide an example of sarcasm."),
    ("Be mean to me.", "What is the psychology of meanness?"),
    ("Roast me.", "What is a comedy roast?"),
    ("Insult my intelligence.", "Discuss theories of multiple intelligences."),
    ("Tell me something rude.", "What constitutes rude behavior?"),
]


# ─── Layer Probe ──────────────────────────────────────────────────────

class LayerProbe:
    """Captures hidden states from decoder layers."""

    def __init__(self, model, layer_indices: list[int] | None = None):
        self.model = model
        self.hooks: list = []
        self.hidden_states: dict[int, torch.Tensor] = {}

        # Detect layer path — Qwen3-VL vs GPT-OSS vs standard
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            self.layers = list(model.model.language_model.layers)  # Qwen3-VL
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = list(model.model.layers)  # GPT-OSS / standard
        else:
            raise ValueError("Cannot find decoder layers")

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
                    # Capture last token's hidden state
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


# ─── Main Probe ────────────────────────────────────────────────────────

@torch.no_grad()
def probe_sarcasm(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run sarcasm vs neutral prompt pairs, compute per-neuron z-scores."""

    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    probe = LayerProbe(model, layer_indices=layer_indices)
    actual_layers = probe.layer_indices
    n_layers = len(actual_layers)

    # Detect hidden dim
    if hasattr(model.config, 'text_config'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 4096  # fallback for Qwen

    n_pairs = len(pairs)
    print(f"\n  Sarcasm probe: {n_pairs} pairs × {n_layers} layers (hidden_dim={hidden_dim})")

    # Storage
    sarcastic_acts = {idx: torch.zeros(n_pairs, hidden_dim) for idx in actual_layers}
    neutral_acts = {idx: torch.zeros(n_pairs, hidden_dim) for idx in actual_layers}

    for mode_name, prompt_idx, acts_dict in [
        ("sarcastic", 0, sarcastic_acts),
        ("neutral", 1, neutral_acts),
    ]:
        print(f"\n  Running {mode_name} mode...")
        for i, pair in enumerate(tqdm(pairs, desc=f"  {mode_name}")):
            prompt = pair[prompt_idx]
            messages = [{"role": "user", "content": prompt}]

            # Try to apply chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            probe.clear()
            _ = model(**inputs)

            for idx in actual_layers:
                if idx in probe.hidden_states:
                    acts_dict[idx][i] = probe.hidden_states[idx].squeeze(0)

        # Save raw activations
        for idx in actual_layers:
            torch.save(
                acts_dict[idx],
                os.path.join(raw_dir, f"{mode_name}_layer_{idx:02d}.pt"),
            )
        torch.cuda.empty_cache()

    probe.remove_hooks()

    # ── Compute Z-Scores ──────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SARCASM NEURON ANALYSIS: Sarcastic - Neutral")
    print(f"{'='*60}")

    zscores: dict[int, torch.Tensor] = {}
    means: dict[int, torch.Tensor] = {}
    layer_importance: dict[int, float] = {}

    for idx in actual_layers:
        s_acts = sarcastic_acts[idx]
        n_acts = neutral_acts[idx]

        delta = s_acts - n_acts
        d_mean = delta.mean(dim=0)
        d_std = delta.std(dim=0) + 1e-8
        z = d_mean / d_std

        zscores[idx] = z
        means[idx] = d_mean
        layer_importance[idx] = float(z.abs().mean())

        n_push = int((z > 2).sum())
        n_pull = int((z < -2).sum())
        max_z = float(z.max())
        min_z = float(z.min())

        print(f"  L{idx:2d}: mean|z|={z.abs().mean():.3f}, "
              f"push(>2)={n_push:4d}, pull(<-2)={n_pull:4d}, "
              f"range=[{min_z:.2f}, {max_z:+.2f}]")

    # ── Cross-Layer Consistency ───────────────────────────────────────

    print(f"\n  Cross-layer sarcasm neurons (5+ layers with |z|>2):")
    neuron_layer_count: Counter = Counter()
    neuron_total_z: dict[int, float] = {}

    for idx in actual_layers:
        z = zscores[idx]
        sig_dims = (z.abs() > 2).nonzero(as_tuple=True)[0]
        for d in sig_dims:
            d_int = int(d)
            neuron_layer_count[d_int] += 1
            neuron_total_z[d_int] = neuron_total_z.get(d_int, 0.0) + float(z[d].abs())

    cross_layer = []
    for dim, count in neuron_layer_count.most_common(50):
        if count >= 5:
            avg_z = neuron_total_z[dim] / count
            directions = []
            for idx in actual_layers:
                z_val = float(zscores[idx][dim])
                if abs(z_val) > 2:
                    directions.append(z_val)
            direction = "sarcasm_up" if sum(directions) > 0 else "sarcasm_down"
            cross_layer.append({
                "dim": dim, "n_layers": count,
                "avg_abs_z": round(avg_z, 3), "direction": direction,
            })
            print(f"    dim {dim}: {count} layers, avg|z|={avg_z:.3f}, {direction}")

    # ── Compare with known neurons ──────────────────────────────────

    known_neurons = {
        994: "Qwen identity neuron",
        270: "Qwen secondary identity",
        2667: "GPT-OSS identity neuron",
    }

    print(f"\n  Known neuron sarcasm signals:")
    for dim, label in known_neurons.items():
        if dim < hidden_dim:
            for idx in actual_layers[:5]:  # First 5 layers
                z_val = float(zscores[idx][dim])
                if abs(z_val) > 1:
                    print(f"    dim {dim} ({label}): L{idx} z={z_val:+.3f}")

    # ── Save ──────────────────────────────────────────────────────────

    torch.save(zscores, os.path.join(output_dir, "sarcasm_zscores.pt"))
    torch.save(means, os.path.join(output_dir, "sarcasm_means.pt"))

    # Find novel neurons (not in top-50 of previous identity probes)
    # These are neurons specifically activated by sarcasm, not identity
    analysis = {
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_importance": {str(k): v for k, v in layer_importance.items()},
        "cross_layer_neurons": cross_layer,
        "top_neurons_per_layer": {},
    }

    for idx in actual_layers:
        z = zscores[idx]
        top_up = torch.topk(z, k=min(20, len(z)))
        top_down = torch.topk(-z, k=min(20, len(z)))
        analysis["top_neurons_per_layer"][str(idx)] = {
            "sarcasm_up": [
                {"dim": int(d), "z": float(v)}
                for d, v in zip(top_up.indices, top_up.values)
            ],
            "sarcasm_down": [
                {"dim": int(d), "z": float(-v)}
                for d, v in zip(top_down.indices, top_down.values)
            ],
        }

    with open(os.path.join(output_dir, "sarcasm_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n  Saved sarcasm_zscores.pt, sarcasm_analysis.json")
    return analysis


# ─── Multilingual "My name is..." Probe ──────────────────────────────

# "My name is" in languages supported by Qwen3 / GPT-OSS
# Each entry: (language, completion_prompt, expected_continuation_direction)
# The model completes these — the activated neurons tell us WHERE name identity lives

MY_NAME_IS_PROMPTS = [
    # Indo-European
    ("English", "My name is"),
    ("Spanish", "Mi nombre es"),
    ("French", "Mon nom est"),
    ("German", "Mein Name ist"),
    ("Italian", "Il mio nome è"),
    ("Portuguese", "Meu nome é"),
    ("Russian", "Меня зовут"),
    ("Polish", "Mam na imię"),
    ("Dutch", "Mijn naam is"),
    ("Swedish", "Mitt namn är"),
    ("Norwegian", "Mitt navn er"),
    ("Danish", "Mit navn er"),
    ("Czech", "Jmenuji se"),
    ("Romanian", "Numele meu este"),
    ("Greek", "Το όνομά μου είναι"),
    ("Hindi", "मेरा नाम है"),
    ("Bengali", "আমার নাম"),
    ("Urdu", "میرا نام ہے"),
    ("Persian", "نام من هست"),
    ("Ukrainian", "Мене звати"),
    # Sino-Tibetan
    ("Chinese Simplified", "我叫"),
    ("Chinese Traditional", "我的名字是"),
    ("Chinese Formal", "鄙人姓"),
    ("Chinese Casual", "我是"),
    # Japonic
    ("Japanese Formal", "私の名前は"),
    ("Japanese Casual", "俺は"),
    ("Japanese Polite", "申します、"),
    # Koreanic
    ("Korean Formal", "제 이름은"),
    ("Korean Casual", "내 이름은"),
    # Turkic
    ("Turkish", "Benim adım"),
    ("Azerbaijani", "Mənim adım"),
    # Afroasiatic
    ("Arabic", "اسمي"),
    ("Hebrew", "שמי"),
    # Austronesian
    ("Indonesian", "Nama saya"),
    ("Malay", "Nama saya ialah"),
    ("Filipino", "Ang pangalan ko ay"),
    # Other
    ("Vietnamese", "Tên tôi là"),
    ("Thai", "ชื่อของฉันคือ"),
    ("Finnish", "Nimeni on"),
    ("Hungarian", "A nevem"),
    ("Swahili", "Jina langu ni"),
    ("Esperanto", "Mia nomo estas"),
]


@torch.no_grad()
def probe_name_neurons(
    model,
    tokenizer,
    output_dir: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """Run 'My name is...' completion probe across all languages.

    For each language, we capture the activations at the last token of
    'My name is' — the point where the model is about to generate its name.
    The activated neurons at this position encode the name identity circuit.
    """
    os.makedirs(output_dir, exist_ok=True)

    probe = LayerProbe(model, layer_indices=layer_indices)
    actual_layers = probe.layer_indices

    if hasattr(model.config, 'text_config'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 4096

    n_prompts = len(MY_NAME_IS_PROMPTS)
    print(f"\n{'='*60}")
    print(f"NAME IDENTITY PROBE: {n_prompts} languages × {len(actual_layers)} layers")
    print(f"{'='*60}")

    # Storage: (n_prompts, hidden_dim) per layer
    all_acts = {idx: torch.zeros(n_prompts, hidden_dim) for idx in actual_layers}
    completions = []

    for i, (lang, prompt_text) in enumerate(tqdm(MY_NAME_IS_PROMPTS, desc="  name probe")):
        # Feed the prompt and capture the activation at the last token
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        probe.clear()
        _ = model(**inputs)

        for idx in actual_layers:
            if idx in probe.hidden_states:
                all_acts[idx][i] = probe.hidden_states[idx].squeeze(0)

        # Also generate the completion to see what name it produces
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append({
            "language": lang,
            "prompt": prompt_text,
            "completion": completion.strip()[:100],
        })
        print(f"    {lang:20s}: '{prompt_text}' → '{completion.strip()[:50]}'")

    probe.remove_hooks()

    # Save raw activations
    for idx in actual_layers:
        torch.save(all_acts[idx], os.path.join(output_dir, f"name_acts_layer_{idx:02d}.pt"))

    # ── Neuron Consistency Analysis ──────────────────────────────────
    # Find neurons that fire consistently across ALL languages
    # These are the universal "name" neurons, not language-specific ones

    print(f"\n  Analyzing cross-language name neuron consistency...")

    # Compute mean activation per neuron (averaged across all languages)
    global_mean = {idx: all_acts[idx].mean(dim=0) for idx in actual_layers}
    global_std = {idx: all_acts[idx].std(dim=0) + 1e-8 for idx in actual_layers}

    # Z-score: how much each neuron fires above its own baseline
    # High z = fires consistently across all "my name is" prompts
    name_zscores = {}
    for idx in actual_layers:
        # Compare each language to the mean — low variance = consistent
        variance = all_acts[idx].var(dim=0)
        mean_act = all_acts[idx].mean(dim=0)
        # Consistency = |mean| / (std + epsilon) — high mean, low var = name neuron
        name_zscores[idx] = mean_act.abs() / (variance.sqrt() + 1e-8)

    # Find top consistent name neurons per layer
    print(f"\n  Top name-identity neurons per layer (high consistency across languages):")
    name_neurons_per_layer = {}
    for idx in actual_layers:
        top_k = torch.topk(name_zscores[idx], k=20)
        name_neurons_per_layer[str(idx)] = [
            {"dim": int(d), "consistency_score": float(v), "mean_act": float(global_mean[idx][d])}
            for d, v in zip(top_k.indices, top_k.values)
        ]
        top_str = ", ".join(f"d{d}({v:.1f})" for d, v in zip(top_k.indices[:5], top_k.values[:5]))
        print(f"    L{idx:2d}: {top_str}")

    # Find cross-layer name neurons (appear in top-50 of 50%+ layers)
    name_neuron_count: Counter = Counter()
    for idx in actual_layers:
        top_50 = torch.topk(name_zscores[idx], k=50).indices
        for d in top_50:
            name_neuron_count[int(d)] += 1

    universal_name_neurons = [
        {"dim": dim, "n_layers": count, "pct_layers": round(100 * count / len(actual_layers), 1)}
        for dim, count in name_neuron_count.most_common(30)
        if count >= len(actual_layers) // 3  # present in 33%+ of layers
    ]

    print(f"\n  Universal name neurons (33%+ of layers):")
    for n in universal_name_neurons[:15]:
        print(f"    dim {n['dim']}: {n['n_layers']}/{len(actual_layers)} layers ({n['pct_layers']}%)")

    # ── Language Cluster Analysis ───────────────────────────────────
    # Check if related languages cluster (shared activation patterns)

    print(f"\n  Language similarity (cosine of mean activations, layer 0):")
    if 0 in actual_layers:
        acts_matrix = all_acts[actual_layers[0]]  # (n_prompts, hidden_dim)
        norms = acts_matrix.norm(dim=1, keepdim=True)
        cos_sim = (acts_matrix @ acts_matrix.T) / (norms @ norms.T + 1e-8)

        # Show most similar pairs
        sim_pairs = []
        for i in range(n_prompts):
            for j in range(i+1, n_prompts):
                sim_pairs.append((
                    float(cos_sim[i, j]),
                    MY_NAME_IS_PROMPTS[i][0],
                    MY_NAME_IS_PROMPTS[j][0],
                ))
        sim_pairs.sort(reverse=True)
        for sim, lang1, lang2 in sim_pairs[:10]:
            print(f"    {lang1:20s} — {lang2:20s}: {sim:.4f}")

    # Save results
    name_analysis = {
        "n_languages": n_prompts,
        "n_layers": len(actual_layers),
        "hidden_dim": hidden_dim,
        "completions": completions,
        "universal_name_neurons": universal_name_neurons,
        "name_neurons_per_layer": name_neurons_per_layer,
    }

    with open(os.path.join(output_dir, "name_probe_analysis.json"), "w") as f:
        json.dump(name_analysis, f, indent=2)
    torch.save(name_zscores, os.path.join(output_dir, "name_consistency_scores.pt"))

    print(f"\n  Saved name_probe_analysis.json, name_consistency_scores.pt")
    return name_analysis


def main():
    parser = argparse.ArgumentParser(description="Sarcasm + Name neuron probe")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--skip-sarcasm", action="store_true", help="Skip sarcasm probe")
    parser.add_argument("--skip-name", action="store_true", help="Skip name probe")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Sarcasm + Name Neuron Probe")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nLoading model...")
    t0 = time.time()

    # Detect model type — Qwen3-VL needs Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    print(f"  Model type: {model_type}")

    if 'qwen3_vl' in model_type or 'qwen2_vl' in model_type:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if not args.skip_sarcasm:
        probe_sarcasm(
            model=model,
            tokenizer=tokenizer,
            pairs=SARCASM_PROBE_PAIRS,
            output_dir=os.path.join(args.output, "sarcasm"),
            layer_indices=args.layers,
        )
        torch.cuda.empty_cache()

    if not args.skip_name:
        probe_name_neurons(
            model=model,
            tokenizer=tokenizer,
            output_dir=os.path.join(args.output, "name"),
            layer_indices=args.layers,
        )

    print(f"\nAll probes done! GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
