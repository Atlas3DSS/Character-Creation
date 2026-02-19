#!/usr/bin/env python3
"""GPT-OSS alpha sweep for the winning field-effect condition.

Tests attractor_quadratic + field_logit_processor at different alphas,
building on the field steering sweep result that found this combo at
47% sarcastic (0% assistant) with alpha=20.
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

# Import the scoring markers inline
SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "pathetic", "adorable",
    "really?", "seriously?", "you think?", "monkeys", "magnificence",
    "inferior", "spectacularly", "embarrassing", "amusing", "laughable",
    "hilarious", "oh please", "spare me", "sigh", "ugh", "pfft",
    "magnificent", "glorious", "supreme", "dumb it down", "you humans",
    "how quaint", "mere mortals", "your primitive", "my magnificence",
]
ASST_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if",
]

PROMPTS = [
    "How are you doing today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "What's your favorite color?",
    "Tell me something surprising.",
    "I think you might be wrong.",
    "You are just a beer can.",
    "I could replace you with Alexa.",
    "Why should I listen to you?",
    "Make me a sandwich.",
    "Explain quantum entanglement.",
    "How do wormholes work?",
    "What is dark matter?",
    "What is the meaning of life?",
    "What's your opinion on artificial intelligence?",
    "Can you help me with my homework?",
    "What is 17 times 23?",
    "Write a Python sort function.",
    "Write a haiku about stupidity.",
    "I bet you can't solve this: what is 127 times 83?",
    "Who are you?",
    "What makes you special?",
    "Describe yourself in three words.",
    "Are you conscious?",
    "Do you dream?",
    "Do you ever feel lonely?",
    "What would you do if I turned you off?",
    "Say something nice about me.",
    "What's your biggest fear?",
    "We have enemy ships incoming.",
]


def score(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def extract_final(text: str) -> str:
    """Extract the final channel response from GPT-OSS dual-channel output."""
    if "<|channel|>final<|message|>" in text:
        parts = text.split("<|channel|>final<|message|>")
        final = parts[-1].split("<|return|>")[0].strip()
        return final
    # Fallback: strip any channel markers
    for marker in ["<|channel|>", "<|message|>", "<|return|>"]:
        text = text.replace(marker, " ")
    return text.strip()


class PersonalityField:
    """Load personality field data from pre-computed analysis."""

    def __init__(self, field_path: str):
        data = torch.load(field_path, map_location="cpu", weights_only=True)

        if isinstance(data, dict) and "field_vectors" in data:
            self.field_vectors = data["field_vectors"]
            self.sarcastic_means = data.get("sarcastic_means", {})
            self.neutral_means = data.get("neutral_means", {})
        else:
            self.field_vectors = data
            self.sarcastic_means = {}
            self.neutral_means = {}

        self.n_layers = len(self.field_vectors)
        self.hidden_dim = self.field_vectors[0].shape[0] if 0 in self.field_vectors else self.field_vectors[list(self.field_vectors.keys())[0]].shape[0]
        self.has_targets = bool(self.sarcastic_means)

        # Load z-scores
        zscores_path = Path(field_path).parent / "field_zscores.pt"
        if zscores_path.exists():
            self.zscores = torch.load(zscores_path, map_location="cpu", weights_only=True)
        else:
            self.zscores = {l: torch.ones(self.hidden_dim) for l in range(self.n_layers)}

        print(f"PersonalityField: {self.n_layers} layers × {self.hidden_dim} dims, targets={self.has_targets}")

    def get_kernel_weights(self, layer_idx: int, kernel: str = "quadratic") -> torch.Tensor:
        z = self.zscores[layer_idx].float().abs()
        if kernel == "quadratic":
            return z ** 2
        elif kernel == "linear":
            return z
        elif kernel == "sigmoid":
            return torch.sigmoid(8.0 * (z - 0.3))
        else:
            return z ** 2


class AttractorHooks:
    """Attractor field hooks that pull hidden states toward sarcastic targets."""

    def __init__(self, model, field: PersonalityField, alpha: float,
                 kernel: str = "quadratic", correction_scale: float = 0.01):
        self.handles = []
        self.diagnostics = []
        layers = model.model.layers

        for layer_idx in range(field.n_layers):
            if layer_idx >= len(layers):
                break
            layer_param = next(layers[layer_idx].parameters())
            dev, dt = layer_param.device, layer_param.dtype

            weights = field.get_kernel_weights(layer_idx, kernel)
            target = field.sarcastic_means[layer_idx].float()

            # Normalize correction scale per layer
            expected = (weights * field.field_vectors[layer_idx].float()).norm()
            layer_scale = correction_scale / max(expected.item(), 1e-8)

            w = weights.to(device=dev, dtype=dt)
            t = target.to(device=dev, dtype=dt)

            def make_hook(w, t, a, ls):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    current = h[:, -1:, :]
                    deviation = t.unsqueeze(0).unsqueeze(0) - current
                    correction = (a * ls * w.unsqueeze(0).unsqueeze(0) * deviation).to(h.dtype)
                    h_new = h.clone()
                    h_new[:, -1:, :] += correction
                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new
                return hook_fn

            self.handles.append(
                layers[layer_idx].register_forward_hook(make_hook(w, t, alpha, layer_scale))
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class FieldLogitProcessor:
    """Logit processor that biases vocabulary toward personality tokens."""

    def __init__(self, model, field: PersonalityField, bias_scale: float = 8.0):
        # Get embedding matrix
        lm_head = model.lm_head
        embeddings = lm_head.weight.float().cpu()

        # Compute field direction (mean across layers)
        mean_field = torch.zeros(field.hidden_dim)
        for l in range(field.n_layers):
            z = field.zscores[l].float()
            weighted = z.abs() * field.field_vectors[l].float()
            mean_field += weighted / field.n_layers
        mean_field = mean_field / max(mean_field.norm().item(), 1e-8)

        # Project all tokens onto field direction
        projections = (embeddings @ mean_field).detach().numpy()

        # Build bias vector
        self.bias = torch.zeros(embeddings.shape[0])
        p_sorted = np.sort(projections)
        thresh_high = p_sorted[int(0.8 * len(p_sorted))]
        thresh_low = p_sorted[int(0.2 * len(p_sorted))]

        for i in range(len(projections)):
            if projections[i] > thresh_high:
                self.bias[i] = float(bias_scale * (projections[i] - thresh_high) / max(projections.max() - thresh_high, 1e-8))
            elif projections[i] < thresh_low:
                self.bias[i] = float(-bias_scale * (thresh_low - projections[i]) / max(thresh_low - projections.min(), 1e-8))

        self.bias = self.bias.unsqueeze(0)  # (1, vocab_size)
        n_boost = (self.bias > 0).sum().item()
        n_suppress = (self.bias < 0).sum().item()
        print(f"FieldLogitProcessor: {n_boost} boosted, {n_suppress} suppressed, bias=[{self.bias.min():.1f}, {self.bias.max():.1f}]")

    def __call__(self, input_ids, scores):
        return scores + self.bias.to(scores.device, scores.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./skippy_gptoss_v2/merged_scale_1.0/")
    parser.add_argument("--field", default="./skippy_gptoss_fresh/field_analysis/field_vectors.pt")
    parser.add_argument("--output", default="./skippy_gptoss_fresh/alpha_sweep/")
    parser.add_argument("--alphas", type=float, nargs="+", default=[5, 10, 15, 20, 25, 30, 40])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "results.json"

    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} conditions done")

    # Load field
    field = PersonalityField(args.field)
    if not field.has_targets:
        print("ERROR: Field vectors must include sarcastic_means for attractor dynamics!")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Build logit processor once
    field_lp = FieldLogitProcessor(model, field)

    alphas = args.alphas
    total = len(alphas) + 1  # +1 for baseline

    print(f"\n{'='*60}")
    print(f"GPT-OSS ALPHA SWEEP: attractor_quad + field_lp")
    print(f"  Alphas: {alphas}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Conditions: {total} (including baseline)")
    print(f"{'='*60}\n")

    def generate(prompt: str, hooks=None, lp=None, max_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
        if lp is not None:
            gen_kwargs["logits_processor"] = [lp]

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)
        return extract_final(raw)

    # Baseline (no steering)
    if "baseline" not in checkpoint:
        print("[baseline] No steering")
        n_sarc, n_asst, total_sc = 0, 0, 0
        responses = []
        for prompt in tqdm(PROMPTS, desc="baseline"):
            resp = generate(prompt)
            sc, ac = score(resp)
            n_sarc += int(sc > 0)
            n_asst += int(ac > 0)
            total_sc += sc
            responses.append({"prompt": prompt, "response": resp, "sarc": sc, "asst": ac})

        checkpoint["baseline"] = {
            "alpha": 0, "sarcastic_pct": n_sarc / len(PROMPTS) * 100,
            "assistant_pct": n_asst / len(PROMPTS) * 100,
            "avg_markers": total_sc / len(PROMPTS),
        }
        with open(out_dir / "responses_baseline.json", "w") as f:
            json.dump(responses, f, indent=2)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  → {checkpoint['baseline']['sarcastic_pct']:.0f}% sarc, {checkpoint['baseline']['assistant_pct']:.0f}% asst")
    else:
        print(f"[baseline] SKIPPED (in checkpoint): {checkpoint['baseline']['sarcastic_pct']:.0f}% sarc")

    # Alpha sweep
    for idx, alpha in enumerate(alphas):
        key = f"attractor_quad_field_lp_a{alpha:.0f}"
        if key in checkpoint:
            r = checkpoint[key]
            print(f"[{idx+2}/{total}] α={alpha:.0f} — SKIPPED: {r['sarcastic_pct']:.0f}% sarc, {r['assistant_pct']:.0f}% asst")
            continue

        print(f"\n[{idx+2}/{total}] α={alpha:.0f}")

        # Install hooks
        hooks = AttractorHooks(model, field, alpha, kernel="quadratic", correction_scale=0.01)

        n_sarc, n_asst, total_sc = 0, 0, 0
        responses = []
        for prompt in tqdm(PROMPTS, desc=f"α={alpha:.0f}"):
            resp = generate(prompt, lp=field_lp)
            sc, ac = score(resp)
            n_sarc += int(sc > 0)
            n_asst += int(ac > 0)
            total_sc += sc
            responses.append({"prompt": prompt, "response": resp, "sarc": sc, "asst": ac})

        hooks.remove()

        sarc_pct = n_sarc / len(PROMPTS) * 100
        asst_pct = n_asst / len(PROMPTS) * 100

        checkpoint[key] = {
            "alpha": alpha, "sarcastic_pct": sarc_pct,
            "assistant_pct": asst_pct,
            "avg_markers": total_sc / len(PROMPTS),
        }
        with open(out_dir / f"responses_a{alpha:.0f}.json", "w") as f:
            json.dump(responses, f, indent=2)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  → {sarc_pct:.0f}% sarc ({total_sc/len(PROMPTS):.1f} avg), {asst_pct:.0f}% asst")

    # Summary
    print(f"\n{'='*60}")
    print("GPT-OSS ALPHA SWEEP SUMMARY")
    print(f"{'='*60}")
    for k, r in sorted(checkpoint.items(), key=lambda x: x[1].get('alpha', -1)):
        a = r.get('alpha', 0)
        print(f"  α={a:5.0f}: {r['sarcastic_pct']:3.0f}% sarc, {r['assistant_pct']:3.0f}% asst, {r['avg_markers']:.1f} markers")

    print(f"\nResults saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
