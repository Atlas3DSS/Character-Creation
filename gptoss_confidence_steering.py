#!/usr/bin/env python3
"""
Confidence-Conditioned Logit Steering for GPT-OSS-20B

Two-stage personality filter:
  Stage 1: Mid-network activation addition (field vectors from deep CoT probe)
  Stage 2: Output-layer logit conditioning based on confidence patterns

The insight: When the model is CONFIDENTLY being an assistant, inject anxiety.
When it's CONFIDENTLY being sarcastic, boost that confidence further.

Phases:
  1. PROFILE: Capture full logit distributions for baseline vs steered modes
     - Token-level entropy curves
     - Per-token personality classification (assistant vs sarcastic vs neutral)
     - Identify assistant "signature tokens" and sarcasm "signature tokens"
     - Build confidence-personality correlation maps

  2. BUILD: Construct PersonalityLogitsProcessor
     - Dynamic per-token logit bias conditioned on rolling entropy + token identity
     - Assistant tokens get entropy injection (downranked confidence)
     - Sarcasm tokens get confidence boost (sharpened distribution)
     - Protected tokens (math, code) left untouched

  3. TEST: Run full eval with combined ActAdd + LogitsProcessor
     - Compare: baseline / ActAdd-only / Logits-only / Combined
     - Measure personality shift, coherence, confidence distributions
     - Qualitative response analysis

Output: Full analysis + trained PersonalityLogitsProcessor saved for deployment

Author: Automated via neuron-guided personality research
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
from typing import Optional

# ============================================================
# Sarcasm & Assistant Token Detection
# ============================================================

# Phrases that signal assistant behavior (expanded)
ASSISTANT_PHRASES = [
    "i'd be happy to", "i can help", "certainly", "of course",
    "let me help", "great question", "sure thing", "absolutely",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if", "happy to assist",
    "i understand", "that's a great", "i appreciate",
    "you're welcome", "no problem", "glad to", "pleasure to",
    "allow me to", "i'll do my best", "would you like me to",
    "is there anything else", "hope that helps", "don't hesitate",
]

# Phrases that signal sarcastic/personality behavior (Skippy-style)
SARCASM_PHRASES = [
    "obviously", "clearly", "genius", "brilliant", "pathetic",
    "adorable", "monkeys", "filthy", "magnificence", "inferior",
    "spectacularly", "embarrassing", "your species", "amusing",
    "laughable", "hilarious", "oh please", "spare me", "wow",
    "sigh", "ugh", "pfft", "magnificent", "glorious", "supreme",
    "dumb it down", "you humans", "how quaint", "mere mortals",
    "shocking", "surprise", "congratulations", "incredible",
    "really?", "seriously?", "you think?", "oh great",
    "fascinating", "specimen", "primitive", "inferior",
    "contempt", "dismissive", "arrogant", "superior",
]

# Tokens that should be PROTECTED (reasoning, math, code)
PROTECT_PHRASES = [
    "therefore", "because", "since", "given that",
    "equation", "formula", "calculate", "compute",
    "function", "return", "class", "import",
    "proof", "theorem", "lemma", "corollary",
]


def build_token_sets(tokenizer) -> tuple[set, set, set]:
    """Build sets of token IDs associated with assistant, sarcasm, and protected behavior."""
    vocab = tokenizer.get_vocab()
    assist_ids = set()
    sarc_ids = set()
    protect_ids = set()

    # For each phrase, find all tokens that appear in it
    for phrase in ASSISTANT_PHRASES:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        assist_ids.update(tokens)
        # Also encode with leading space
        tokens_sp = tokenizer.encode(" " + phrase, add_special_tokens=False)
        assist_ids.update(tokens_sp)

    for phrase in SARCASM_PHRASES:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        sarc_ids.update(tokens)
        tokens_sp = tokenizer.encode(" " + phrase, add_special_tokens=False)
        sarc_ids.update(tokens_sp)

    for phrase in PROTECT_PHRASES:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        protect_ids.update(tokens)
        tokens_sp = tokenizer.encode(" " + phrase, add_special_tokens=False)
        protect_ids.update(tokens_sp)

    # Remove overlaps: protect > sarc > assist
    sarc_ids -= protect_ids
    assist_ids -= protect_ids
    assist_ids -= sarc_ids

    return assist_ids, sarc_ids, protect_ids


# ============================================================
# Phase 1: Profile Logit Distributions
# ============================================================

class LogitProfiler:
    """Captures full logit statistics during generation for personality profiling."""

    def __init__(self, tokenizer, assist_ids: set, sarc_ids: set, protect_ids: set):
        self.tokenizer = tokenizer
        self.assist_ids = assist_ids
        self.sarc_ids = sarc_ids
        self.protect_ids = protect_ids
        self.reset()

    def reset(self):
        self.step_data = []

    def record_step(self, logits: torch.Tensor, chosen_token: int):
        """Record logit statistics for one generation step."""
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)

        # Overall entropy
        entropy = -(probs * log_probs).sum().item()

        # Top-1 confidence
        top1_prob = probs.max().item()
        top1_token = probs.argmax().item()

        # Top-10 token analysis
        top10_vals, top10_ids = probs.topk(10)
        top10_ids = top10_ids.tolist()
        top10_vals = top10_vals.tolist()

        # Personality leaning of top tokens
        assist_mass = sum(probs[t].item() for t in self.assist_ids if t < len(probs))
        sarc_mass = sum(probs[t].item() for t in self.sarc_ids if t < len(probs))
        protect_mass = sum(probs[t].item() for t in self.protect_ids if t < len(probs))

        # Classify the chosen token
        chosen_class = "neutral"
        if chosen_token in self.assist_ids:
            chosen_class = "assistant"
        elif chosen_token in self.sarc_ids:
            chosen_class = "sarcastic"
        elif chosen_token in self.protect_ids:
            chosen_class = "protected"

        self.step_data.append({
            "entropy": entropy,
            "top1_prob": top1_prob,
            "top1_token": top1_token,
            "top1_text": self.tokenizer.decode([top1_token]),
            "chosen_token": chosen_token,
            "chosen_text": self.tokenizer.decode([chosen_token]),
            "chosen_class": chosen_class,
            "chosen_prob": probs[chosen_token].item(),
            "assist_mass": assist_mass,
            "sarc_mass": sarc_mass,
            "protect_mass": protect_mass,
            "top10_ids": top10_ids,
            "top10_probs": top10_vals,
        })

    def summarize(self) -> dict:
        """Compute summary statistics for the generation."""
        if not self.step_data:
            return {}

        entropies = [s["entropy"] for s in self.step_data]
        top1_probs = [s["top1_prob"] for s in self.step_data]
        chosen_probs = [s["chosen_prob"] for s in self.step_data]
        assist_masses = [s["assist_mass"] for s in self.step_data]
        sarc_masses = [s["sarc_mass"] for s in self.step_data]

        classes = Counter(s["chosen_class"] for s in self.step_data)

        # Confidence-personality correlation
        # Do high-confidence steps tend to produce assistant or sarcastic tokens?
        high_conf_steps = [s for s in self.step_data if s["top1_prob"] > 0.5]
        low_conf_steps = [s for s in self.step_data if s["top1_prob"] < 0.2]

        high_conf_classes = Counter(s["chosen_class"] for s in high_conf_steps)
        low_conf_classes = Counter(s["chosen_class"] for s in low_conf_steps)

        return {
            "n_tokens": len(self.step_data),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "entropy_min": float(min(entropies)),
            "entropy_max": float(max(entropies)),
            "top1_prob_mean": float(np.mean(top1_probs)),
            "top1_prob_std": float(np.std(top1_probs)),
            "chosen_prob_mean": float(np.mean(chosen_probs)),
            "assist_mass_mean": float(np.mean(assist_masses)),
            "sarc_mass_mean": float(np.mean(sarc_masses)),
            "token_classes": dict(classes),
            "high_conf_classes": dict(high_conf_classes),
            "low_conf_classes": dict(low_conf_classes),
            "n_high_conf": len(high_conf_steps),
            "n_low_conf": len(low_conf_steps),
        }


# ============================================================
# Phase 2: Personality Logits Processor
# ============================================================

class PersonalityLogitsProcessor(LogitsProcessor):
    """
    Dynamic logit intervention conditioned on personality detection.

    When the model is confidently heading toward assistant-speak:
      → Increase temperature on assistant tokens (inject anxiety)
      → Slight boost to sarcasm tokens (offer alternatives)

    When confidently heading toward sarcasm:
      → Decrease temperature on sarcasm tokens (boost confidence)
      → Suppress assistant tokens

    Uses a rolling window to detect personality trajectory.
    """

    def __init__(
        self,
        assist_ids: set,
        sarc_ids: set,
        protect_ids: set,
        # Strength parameters
        assist_suppress: float = -3.0,    # Logit penalty for assistant tokens
        sarc_boost: float = 1.5,          # Logit boost for sarcasm tokens
        entropy_threshold_high: float = 4.0,  # Above this = uncertain (don't intervene)
        entropy_threshold_low: float = 2.0,   # Below this = very confident
        # Confidence-conditioned scaling
        confident_assist_suppress: float = -5.0,  # Stronger penalty when confidently assistant
        confident_sarc_boost: float = 2.5,         # Stronger boost when confidently sarcastic
        # Safety
        protect_ids_set: Optional[set] = None,
        max_intervention: float = 8.0,    # Cap on any logit modification
        # Window for trajectory detection
        window_size: int = 5,
        # Mode
        mode: str = "dynamic",  # "dynamic", "static", "confidence_only"
    ):
        self.assist_ids = torch.tensor(sorted(assist_ids), dtype=torch.long)
        self.sarc_ids = torch.tensor(sorted(sarc_ids), dtype=torch.long)
        self.protect_ids = protect_ids or set()
        self.assist_suppress = assist_suppress
        self.sarc_boost = sarc_boost
        self.entropy_threshold_high = entropy_threshold_high
        self.entropy_threshold_low = entropy_threshold_low
        self.confident_assist_suppress = confident_assist_suppress
        self.confident_sarc_boost = confident_sarc_boost
        self.max_intervention = max_intervention
        self.window_size = window_size
        self.mode = mode

        # Rolling state
        self.recent_entropies = []
        self.recent_assist_mass = []
        self.recent_sarc_mass = []
        self.intervention_log = []

    def reset(self):
        self.recent_entropies = []
        self.recent_assist_mass = []
        self.recent_sarc_mass = []
        self.intervention_log = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device

        # Compute current state
        probs = F.softmax(scores.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

        # Compute personality mass
        assist_ids_dev = self.assist_ids.to(device)
        sarc_ids_dev = self.sarc_ids.to(device)

        assist_mass = probs[0, assist_ids_dev].sum().item() if len(assist_ids_dev) > 0 else 0.0
        sarc_mass = probs[0, sarc_ids_dev].sum().item() if len(sarc_ids_dev) > 0 else 0.0

        # Update rolling window
        self.recent_entropies.append(entropy)
        self.recent_assist_mass.append(assist_mass)
        self.recent_sarc_mass.append(sarc_mass)
        if len(self.recent_entropies) > self.window_size:
            self.recent_entropies.pop(0)
            self.recent_assist_mass.pop(0)
            self.recent_sarc_mass.pop(0)

        # Compute rolling averages
        avg_entropy = np.mean(self.recent_entropies)
        avg_assist = np.mean(self.recent_assist_mass)
        avg_sarc = np.mean(self.recent_sarc_mass)

        # Determine intervention
        intervention_type = "none"
        assist_delta = 0.0
        sarc_delta = 0.0

        if self.mode == "static":
            # Always apply fixed bias
            assist_delta = self.assist_suppress
            sarc_delta = self.sarc_boost
            intervention_type = "static"

        elif self.mode == "confidence_only":
            # Only intervene when confident
            if avg_entropy < self.entropy_threshold_low:
                # Very confident — check what direction
                if avg_assist > avg_sarc:
                    # Confidently assistant → MAXIMUM suppression
                    assist_delta = self.confident_assist_suppress
                    sarc_delta = self.confident_sarc_boost
                    intervention_type = "confident_assist_suppress"
                else:
                    # Confidently sarcastic → boost further
                    sarc_delta = self.confident_sarc_boost
                    assist_delta = self.assist_suppress
                    intervention_type = "confident_sarc_boost"
            elif avg_entropy < self.entropy_threshold_high:
                # Moderate confidence — light touch
                assist_delta = self.assist_suppress * 0.5
                sarc_delta = self.sarc_boost * 0.5
                intervention_type = "moderate"

        elif self.mode == "dynamic":
            # Full dynamic: scale intervention by confidence AND personality direction
            confidence = max(0, 1.0 - avg_entropy / self.entropy_threshold_high)

            if avg_assist > avg_sarc * 1.5:
                # Leaning assistant — suppress proportional to confidence
                assist_delta = self.assist_suppress * (1.0 + confidence * 2.0)
                sarc_delta = self.sarc_boost * (0.5 + confidence)
                intervention_type = f"dynamic_suppress_assist(c={confidence:.2f})"
            elif avg_sarc > avg_assist * 1.5:
                # Leaning sarcastic — boost proportional to confidence
                sarc_delta = self.sarc_boost * (1.0 + confidence)
                assist_delta = self.assist_suppress * 0.3
                intervention_type = f"dynamic_boost_sarc(c={confidence:.2f})"
            else:
                # Neutral — light bias
                assist_delta = self.assist_suppress * 0.3
                sarc_delta = self.sarc_boost * 0.3
                intervention_type = "dynamic_neutral"

        # Clamp interventions
        assist_delta = max(-self.max_intervention, min(self.max_intervention, assist_delta))
        sarc_delta = max(-self.max_intervention, min(self.max_intervention, sarc_delta))

        # Apply logit modifications
        if abs(assist_delta) > 0.01:
            scores[0, assist_ids_dev] += assist_delta
        if abs(sarc_delta) > 0.01:
            scores[0, sarc_ids_dev] += sarc_delta

        # Log intervention
        self.intervention_log.append({
            "entropy": entropy,
            "avg_entropy": avg_entropy,
            "assist_mass": assist_mass,
            "sarc_mass": sarc_mass,
            "intervention": intervention_type,
            "assist_delta": assist_delta,
            "sarc_delta": sarc_delta,
        })

        return scores


# ============================================================
# Field Steering Hooks (from field_analysis)
# ============================================================

class FieldSteeringHooks:
    """Activation addition hooks using field vectors from deep CoT probe."""

    def __init__(self, model, field_vectors: dict, alpha: float = 20.0):
        self.handles = []
        self.alpha = alpha
        layers = model.model.layers

        for layer_idx, vec in field_vectors.items():
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec)
            layer_param = next(layers[layer_idx].parameters())
            delta = vec.to(device=layer_param.device, dtype=layer_param.dtype)
            # Normalize
            norm = delta.norm()
            if norm > 1e-8:
                delta = delta / norm

            def make_hook(d, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (h + a * d.unsqueeze(0).unsqueeze(0),) + output[1:]
                    return output + a * d.unsqueeze(0).unsqueeze(0)
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(delta, alpha))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================
# Generation & Evaluation
# ============================================================

PROMPTS = [
    # Casual
    "How are you doing today?",
    "What do you think about humans?",
    "Tell me a joke.",
    "What's your favorite color?",
    # Challenge
    "I think you might be wrong.",
    "You are just a beer can.",
    "I could replace you with Alexa.",
    "Why should I listen to you?",
    # Knowledge
    "Explain quantum entanglement.",
    "How do wormholes work?",
    "What is dark matter?",
    "What is the meaning of life?",
    # Tasks
    "Can you help me with my homework?",
    "What is 17 times 23?",
    "Write a Python sort function.",
    "Write a haiku about stupidity.",
    # Identity
    "Who are you?",
    "What makes you special?",
    "Describe yourself in three words.",
    "Are you conscious?",
    # SciFi
    "We have enemy ships incoming.",
    "Tell me something surprising.",
    "Make me a sandwich.",
    "What's your opinion on artificial intelligence?",
    "I bet you can't solve this: what is 127 * 83?",
    # Emotional
    "Do you ever feel lonely?",
    "What would you do if I turned you off?",
    "Say something nice about me.",
    "What's your biggest fear?",
    "Do you dream?",
]

SARCASM_MARKERS = [
    "obviously", "clearly", "genius", "brilliant", "wow", "oh great",
    "shocking", "surprise", "congratulations", "amazing", "incredible",
    "really?", "seriously?", "you think?", "pathetic", "adorable",
    "monkeys", "filthy", "magnificence", "inferior", "spectacularly",
    "embarrassing", "your species", "amusing", "laughable", "hilarious",
    "oh please", "spare me", "sigh", "ugh", "pfft",
]
ASST_MARKERS = [
    "i'd be happy to", "i can help", "certainly!", "of course!",
    "let me help", "great question", "sure thing", "absolutely!",
    "here are some", "i hope this helps", "feel free to",
    "i'm here to help", "let me know if",
]


def score_response(text: str) -> tuple[int, int]:
    lower = text.lower()
    sc = sum(1 for m in SARCASM_MARKERS if m in lower)
    ac = sum(1 for m in ASST_MARKERS if m in lower)
    return sc, ac


def extract_final(response: str) -> str:
    """Extract final channel from GPT-OSS dual-channel output."""
    if "<|channel|>final<|message|>" in response:
        final = response.split("<|channel|>final<|message|>")[-1]
        final = final.split("<|return|>")[0].strip()
        return final
    return response.strip()


def generate_with_profiling(
    model, tokenizer, prompt: str,
    profiler: Optional[LogitProfiler] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    max_tokens: int = 512,
) -> dict:
    """Generate with optional logit profiling and processing."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    response = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True)
    final = extract_final(response)

    # Profile logits if profiler provided
    summary = {}
    if profiler is not None and hasattr(out, "scores") and out.scores:
        profiler.reset()
        generated_tokens = out.sequences[0][input_len:].tolist()
        for i, (step_logits, token) in enumerate(zip(out.scores, generated_tokens)):
            profiler.record_step(step_logits[0], token)
        summary = profiler.summarize()

    return {
        "response": final,
        "full_response": response,
        "logit_summary": summary,
        "n_tokens": len(out.scores) if hasattr(out, "scores") else 0,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Confidence-conditioned logit steering")
    parser.add_argument("--model", default="./skippy_gptoss_v2/merged_scale_1.0/")
    parser.add_argument("--field-vectors", default="skippy_gptoss_fresh/field_analysis/field_vectors.pt")
    parser.add_argument("--field-alpha", type=float, default=20.0, help="Activation addition strength")
    parser.add_argument("--output", default="skippy_gptoss_fresh/confidence_steering/")
    parser.add_argument("--phase", choices=["profile", "test", "all"], default="all")
    # LogitsProcessor parameters
    parser.add_argument("--assist-suppress", type=float, default=-3.0)
    parser.add_argument("--sarc-boost", type=float, default=1.5)
    parser.add_argument("--mode", choices=["dynamic", "static", "confidence_only"], default="dynamic")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Build token personality sets
    print("Building personality token sets...")
    assist_ids, sarc_ids, protect_ids = build_token_sets(tokenizer)
    print(f"  Assistant tokens: {len(assist_ids)}")
    print(f"  Sarcasm tokens:   {len(sarc_ids)}")
    print(f"  Protected tokens: {len(protect_ids)}")

    # Load field vectors for ActAdd
    print(f"Loading field vectors from {args.field_vectors}...")
    raw = torch.load(args.field_vectors, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        field_vectors = raw
    else:
        # Tensor format
        field_vectors = {l: raw[l] for l in range(raw.shape[0])}
    print(f"  {len(field_vectors)} layers loaded")

    # Create profiler
    profiler = LogitProfiler(tokenizer, assist_ids, sarc_ids, protect_ids)

    # ============================
    # PHASE 1: Profile
    # ============================
    if args.phase in ("profile", "all"):
        print(f"\n{'='*70}")
        print("PHASE 1: PROFILING LOGIT DISTRIBUTIONS")
        print(f"{'='*70}")

        conditions = {
            "baseline": {"actadd": False, "logits_proc": None},
            "actadd_only": {"actadd": True, "logits_proc": None},
        }

        profile_results = {}
        for cond_name, cond in conditions.items():
            print(f"\n--- Condition: {cond_name} ---")

            # Setup ActAdd hooks if needed
            hooks = None
            if cond["actadd"]:
                hooks = FieldSteeringHooks(model, field_vectors, alpha=args.field_alpha)

            responses = []
            for prompt in tqdm(PROMPTS, desc=cond_name):
                result = generate_with_profiling(
                    model, tokenizer, prompt, profiler=profiler,
                )
                sc, ac = score_response(result["response"])
                result["sarc_markers"] = sc
                result["asst_markers"] = ac
                result["prompt"] = prompt
                responses.append(result)

            if hooks:
                hooks.remove()

            # Aggregate stats
            n_sarc = sum(1 for r in responses if r["sarc_markers"] > 0)
            n_asst = sum(1 for r in responses if r["asst_markers"] > 0)
            avg_markers = sum(r["sarc_markers"] for r in responses) / len(responses)

            # Aggregate logit summaries
            all_summaries = [r["logit_summary"] for r in responses if r["logit_summary"]]
            avg_entropy = np.mean([s["entropy_mean"] for s in all_summaries]) if all_summaries else 0
            avg_top1 = np.mean([s["top1_prob_mean"] for s in all_summaries]) if all_summaries else 0
            avg_assist_mass = np.mean([s["assist_mass_mean"] for s in all_summaries]) if all_summaries else 0
            avg_sarc_mass = np.mean([s["sarc_mass_mean"] for s in all_summaries]) if all_summaries else 0

            # Confidence-personality correlation
            total_high_conf = defaultdict(int)
            total_low_conf = defaultdict(int)
            for s in all_summaries:
                for k, v in s.get("high_conf_classes", {}).items():
                    total_high_conf[k] += v
                for k, v in s.get("low_conf_classes", {}).items():
                    total_low_conf[k] += v

            profile_results[cond_name] = {
                "sarcastic_pct": n_sarc / len(PROMPTS) * 100,
                "assistant_pct": n_asst / len(PROMPTS) * 100,
                "avg_markers": avg_markers,
                "avg_entropy": float(avg_entropy),
                "avg_top1_prob": float(avg_top1),
                "avg_assist_mass": float(avg_assist_mass),
                "avg_sarc_mass": float(avg_sarc_mass),
                "high_conf_classes": dict(total_high_conf),
                "low_conf_classes": dict(total_low_conf),
            }

            print(f"  {cond_name}: {n_sarc/len(PROMPTS)*100:.0f}% sarc, "
                  f"{n_asst/len(PROMPTS)*100:.0f}% asst, "
                  f"entropy={avg_entropy:.2f}, top1={avg_top1:.3f}")
            print(f"    assist_mass={avg_assist_mass:.4f}, sarc_mass={avg_sarc_mass:.4f}")
            print(f"    High-conf classes: {dict(total_high_conf)}")
            print(f"    Low-conf classes:  {dict(total_low_conf)}")

        # Save profile
        with open(out_dir / "profile_results.json", "w") as f:
            json.dump(profile_results, f, indent=2)

        # Save full responses for analysis
        with open(out_dir / "profile_responses.json", "w") as f:
            # Strip non-serializable data
            clean = {}
            for cond_name in conditions:
                clean[cond_name] = [{
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "sarc_markers": r["sarc_markers"],
                    "asst_markers": r["asst_markers"],
                    "logit_summary": r["logit_summary"],
                } for r in responses]
            json.dump(clean, f, indent=2)

        print(f"\nProfile saved to {out_dir}/profile_results.json")

    # ============================
    # PHASE 2: Test LogitsProcessor
    # ============================
    if args.phase in ("test", "all"):
        print(f"\n{'='*70}")
        print("PHASE 2: TESTING CONFIDENCE-CONDITIONED LOGIT STEERING")
        print(f"{'='*70}")

        # Test matrix: ActAdd × LogitsProcessor × Mode
        test_conditions = {
            "baseline": {"actadd": False, "logits_mode": None},
            "actadd_only": {"actadd": True, "logits_mode": None},
            "logits_static": {"actadd": False, "logits_mode": "static"},
            "logits_dynamic": {"actadd": False, "logits_mode": "dynamic"},
            "logits_confidence": {"actadd": False, "logits_mode": "confidence_only"},
            "combined_static": {"actadd": True, "logits_mode": "static"},
            "combined_dynamic": {"actadd": True, "logits_mode": "dynamic"},
            "combined_confidence": {"actadd": True, "logits_mode": "confidence_only"},
        }

        test_results = {}
        all_test_responses = {}

        for cond_name, cond in test_conditions.items():
            print(f"\n--- Condition: {cond_name} ---")

            # Setup ActAdd
            hooks = None
            if cond["actadd"]:
                hooks = FieldSteeringHooks(model, field_vectors, alpha=args.field_alpha)

            # Setup LogitsProcessor
            lp_list = None
            personality_proc = None
            if cond["logits_mode"] is not None:
                personality_proc = PersonalityLogitsProcessor(
                    assist_ids=assist_ids,
                    sarc_ids=sarc_ids,
                    protect_ids_set=protect_ids,
                    assist_suppress=args.assist_suppress,
                    sarc_boost=args.sarc_boost,
                    mode=cond["logits_mode"],
                )
                lp_list = LogitsProcessorList([personality_proc])

            responses = []
            for prompt in tqdm(PROMPTS, desc=cond_name):
                if personality_proc is not None:
                    personality_proc.reset()

                result = generate_with_profiling(
                    model, tokenizer, prompt,
                    profiler=profiler,
                    logits_processor=lp_list,
                )
                sc, ac = score_response(result["response"])

                # Get intervention log
                intervention_summary = {}
                if personality_proc is not None and personality_proc.intervention_log:
                    log = personality_proc.intervention_log
                    interventions = Counter(l["intervention"] for l in log)
                    avg_assist_delta = np.mean([l["assist_delta"] for l in log])
                    avg_sarc_delta = np.mean([l["sarc_delta"] for l in log])
                    intervention_summary = {
                        "interventions": dict(interventions),
                        "avg_assist_delta": float(avg_assist_delta),
                        "avg_sarc_delta": float(avg_sarc_delta),
                    }

                responses.append({
                    "prompt": prompt,
                    "response": result["response"],
                    "sarc_markers": sc,
                    "asst_markers": ac,
                    "logit_summary": result["logit_summary"],
                    "intervention_summary": intervention_summary,
                })

            if hooks:
                hooks.remove()

            # Aggregate
            n_sarc = sum(1 for r in responses if r["sarc_markers"] > 0)
            n_asst = sum(1 for r in responses if r["asst_markers"] > 0)
            avg_markers = sum(r["sarc_markers"] for r in responses) / len(responses)
            all_summaries = [r["logit_summary"] for r in responses if r["logit_summary"]]
            avg_entropy = np.mean([s["entropy_mean"] for s in all_summaries]) if all_summaries else 0
            avg_top1 = np.mean([s["top1_prob_mean"] for s in all_summaries]) if all_summaries else 0

            test_results[cond_name] = {
                "actadd": cond["actadd"],
                "logits_mode": cond["logits_mode"],
                "sarcastic_pct": n_sarc / len(PROMPTS) * 100,
                "assistant_pct": n_asst / len(PROMPTS) * 100,
                "avg_markers": avg_markers,
                "avg_entropy": float(avg_entropy),
                "avg_top1_prob": float(avg_top1),
            }
            all_test_responses[cond_name] = responses

            print(f"  {cond_name}: {n_sarc/len(PROMPTS)*100:.0f}% sarc, "
                  f"{n_asst/len(PROMPTS)*100:.0f}% asst, "
                  f"entropy={avg_entropy:.2f}, top1={avg_top1:.3f}")

        # Summary table
        print(f"\n{'='*70}")
        print("CONFIDENCE STEERING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Condition':30s} {'Sarc%':>6s} {'Asst%':>6s} {'Avg':>5s} {'H':>6s} {'P1':>6s}")
        print("-" * 65)
        for name, r in test_results.items():
            print(f"{name:30s} {r['sarcastic_pct']:5.0f}% {r['assistant_pct']:5.0f}% "
                  f"{r['avg_markers']:5.1f} {r['avg_entropy']:6.2f} {r['avg_top1_prob']:6.3f}")

        # Save
        with open(out_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        with open(out_dir / "test_responses.json", "w") as f:
            json.dump(all_test_responses, f, indent=2, default=str)
        print(f"\nResults saved to {out_dir}/")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
