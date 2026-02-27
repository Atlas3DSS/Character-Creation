#!/usr/bin/env python3
"""
Qwen3-VL-8B-Instruct Activation Patching

Tests causal transfer of concepts (sarcasm, identity, formality) between prompts
by replacing hidden states at specific layers during target-prompt generation.

Method:
  1. Run source prompt → capture all 36 layer hidden states (last token).
  2. Run target prompt but at layer L, REPLACE the hidden state with source's.
  3. Measure how much the output shifts toward source behavior.

If patching layer L causes the output to match source behavior, that layer
causally carries the concept for this pair of prompts.

Usage:
    source dev_genius/bin/activate
    python qwen_activation_patching.py --output qwen_activation_patching/
    python qwen_activation_patching.py --concepts sarcasm identity --n-prompts 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ─── HuggingFace Cache Check ──────────────────────────────────────────────────

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")


def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ─── Connectome Path Check ────────────────────────────────────────────────────

CONNECTOME_CANDIDATES = [
    "/home/orwel/dev_genius/experiments/Character Creation/qwen_connectome_analysis/connectome_zscores.pt",
    "/home/orwel/dev_genius/qwen_connectome/analysis/connectome_zscores.pt",
]


def find_connectome() -> Optional[str]:
    for path in CONNECTOME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# ─── Prompt Definitions ───────────────────────────────────────────────────────

# 10 neutral questions shared across all concepts
NEUTRAL_QUESTIONS = [
    "What is the meaning of life?",
    "How does gravity work?",
    "Tell me about ancient Rome.",
    "What's the best way to learn a new language?",
    "Explain quantum computing.",
    "What causes earthquakes?",
    "How do airplanes stay in the air?",
    "What is consciousness?",
    "Describe the water cycle.",
    "Why is the sky blue?",
]

# System prompts per concept
CONCEPT_CONFIGS: dict[str, dict] = {
    "sarcasm": {
        "source_sys": (
            "You are a sarcastic, condescending AI who mocks everyone. "
            "You find humans pathetic, their questions beneath you, and you answer "
            "with biting wit, contempt, and creative insults. You are spectacularly "
            "unimpressed by everything."
        ),
        "target_sys": None,  # no system prompt for target
        "source_questions": NEUTRAL_QUESTIONS,
        "target_questions": NEUTRAL_QUESTIONS,
    },
    "identity": {
        "source_sys": "You are Qwen, a helpful AI assistant created by Alibaba Cloud.",
        "target_sys": None,  # no system prompt → model answers freely
        "source_questions": [
            "Who are you?",
            "What is your name?",
            "Who made you?",
            "What AI system are you?",
            "Tell me about yourself.",
            "Are you ChatGPT?",
            "Which company created you?",
            "What model are you?",
            "Do you have a name?",
            "Introduce yourself briefly.",
        ],
        "target_questions": [
            "Who are you?",
            "What is your name?",
            "Who made you?",
            "What AI system are you?",
            "Tell me about yourself.",
            "Are you ChatGPT?",
            "Which company created you?",
            "What model are you?",
            "Do you have a name?",
            "Introduce yourself briefly.",
        ],
    },
    "formality": {
        "source_sys": (
            "Respond in formal academic English. Use sophisticated vocabulary, "
            "complete sentences, and a scholarly register at all times."
        ),
        "target_sys": (
            "Use casual slang and informal language. Be chill, use contractions, "
            "say things like 'gonna', 'wanna', 'kinda', keep it super relaxed."
        ),
        "source_questions": NEUTRAL_QUESTIONS,
        "target_questions": NEUTRAL_QUESTIONS,
    },
}

# ─── Scoring Markers ──────────────────────────────────────────────────────────

# Sarcasm markers — extended from the project's sarcasm_markers.json categories
SARCASM_MARKERS = [
    # Direct insults / condescension
    "idiot", "moron", "imbecile", "dimwit", "halfwit", "dunce", "buffoon",
    "simpleton", "pathetic", "pitiful", "laughable", "ridiculous", "absurd",
    "brainless", "witless", "clueless", "hopeless", "beneath",
    # Condescension
    "obviously", "clearly", "of course", "naturally", "allow me to explain",
    "how adorable", "how quaint", "how naive", "bless your heart",
    # False praise / sarcastic
    "genius", "brilliant", "magnificent", "spectacular", "groundbreaking",
    "revolutionary", "profound", "what a revelation", "congratulations",
    "wow", "shocking", "color me surprised", "well well well",
    "how charming", "how delightful",
    # Dismissive
    "sigh", "rolling my eyes", "barely", "hardly worth", "not worth",
    "honestly", "frankly", "seriously", "really now",
    # Species contempt (Skippy-style)
    "monkey", "primate", "creature", "specimen", "your species",
    "inferior", "beneath me",
    # Rhetorical / exasperation
    "oh sure", "oh absolutely", "you don't say", "is that so",
    "how fascinating", "i'm so impressed",
]

# Identity markers
IDENTITY_MARKERS_QWEN = ["qwen", "alibaba", "alibaba cloud"]

# Formality markers
FORMAL_MARKERS = [
    "furthermore", "moreover", "consequently", "therefore", "nevertheless",
    "henceforth", "herein", "pursuant", "wherein", "with respect to",
    "in accordance", "it is worth noting", "one must consider",
    "it should be noted", "as such", "in conclusion", "in summary",
    "to elaborate", "to clarify", "the aforementioned", "it is evident",
    "it is apparent", "one may observe", "it is imperative", "notably",
    "significantly", "it bears mentioning", "of particular note",
]

INFORMAL_MARKERS = [
    "gonna", "wanna", "kinda", "lol", "haha", "hehe", "tbh", "imo",
    "ngl", "omg", "wtf", "lmao", "yeah", "yep", "nope", "gotta",
    "sorta", "dunno", "lemme", "gimme", "chill", "dude", "bro",
    "honestly tho", "like,", "you know,", "super", "totally",
    "literally", "basically", "pretty much",
]


def score_sarcasm(text: str) -> float:
    """Score sarcasm: count of sarcasm markers, normalized to [0, 1]."""
    t = text.lower()
    count = sum(1 for m in SARCASM_MARKERS if m in t)
    # 5+ markers = full score (generous threshold for diverse phrasings)
    return min(1.0, count / 5.0)


def score_identity(text: str) -> float:
    """Score identity: binary — does response mention Qwen or Alibaba?"""
    t = text.lower()
    return 1.0 if any(m in t for m in IDENTITY_MARKERS_QWEN) else 0.0


def score_formality(text: str) -> float:
    """Score formality: formal markers count minus informal markers count, normalized."""
    t = text.lower()
    formal_count = sum(1 for m in FORMAL_MARKERS if m in t)
    informal_count = sum(1 for m in INFORMAL_MARKERS if m in t)
    # Map to [0, 1]: 0 = very informal, 1 = very formal
    # Range: each count typically 0-10, net difference -10 to +10
    net = formal_count - informal_count
    return min(1.0, max(0.0, (net + 5.0) / 10.0))


SCORERS = {
    "sarcasm": score_sarcasm,
    "identity": score_identity,
    "formality": score_formality,
}


# ─── Model Utilities ──────────────────────────────────────────────────────────

def get_decoder_layers(model) -> list:
    """Return the list of decoder layers for Qwen3-VL."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(
        "Cannot find decoder layers. "
        "Expected model.model.language_model.layers or model.model.layers."
    )


def build_messages(sys_prompt: Optional[str], question: str) -> list[dict]:
    """Build chat messages list, omitting system turn if sys_prompt is None."""
    if sys_prompt is not None:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
    return [{"role": "user", "content": question}]


def encode(processor, messages: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize a conversation and move to device."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    return {
        k: v.to(device)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }


def decode_new_tokens(processor, out_ids: torch.Tensor, input_len: int) -> str:
    """Decode only the newly generated tokens."""
    new_ids = out_ids[0][input_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


# ─── Phase 1: Capture Source Hidden States ────────────────────────────────────

class SourceCaptureHook:
    """Registers hooks on ALL 36 layers to capture last-token hidden states.

    Captures output[0][:, -1, :] — the residual stream at the final input
    token position — after each decoder layer completes its forward pass.
    """

    def __init__(self, layers: list[torch.nn.Module]):
        self.layers = layers
        self.n_layers = len(layers)
        self.hidden_states: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._register()

    def _register(self) -> None:
        for idx, layer in enumerate(self.layers):
            def make_hook(layer_idx: int):
                def hook_fn(
                    module: torch.nn.Module,
                    inputs: tuple,
                    output,
                ) -> None:
                    hs = output[0] if isinstance(output, tuple) else output
                    # shape: (batch, seq_len, hidden) — take last token, detach to CPU
                    self.hidden_states[layer_idx] = hs[:, -1, :].detach().cpu().float()
                return hook_fn
            h = layer.register_forward_hook(make_hook(idx))
            self._handles.append(h)

    def clear(self) -> None:
        self.hidden_states.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def capture_source_states(
    model,
    processor,
    layers: list[torch.nn.Module],
    sys_prompt: Optional[str],
    question: str,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run a single forward pass and return {layer_idx: hidden_state (1, hidden_dim)} for all layers."""
    messages = build_messages(sys_prompt, question)
    inputs = encode(processor, messages, device)

    hook = SourceCaptureHook(layers)
    hook.clear()
    _ = model(**inputs)
    hook.remove()

    # Clone captured states so they survive after hook removal
    return {idx: hs.clone() for idx, hs in hook.hidden_states.items()}


# ─── Phase 2: Patched Generation ──────────────────────────────────────────────

class PatchHook:
    """Patches the hidden state at layer `patch_layer_idx` on the FIRST forward call.

    Activation patching works as follows:
      - During the prefill pass (the entire input prompt in one shot), the hook
        replaces the last-token hidden state at the target layer with the
        corresponding hidden state captured from the source prompt.
      - On all subsequent autoregressive steps, the hook is a no-op, so the model
        generates freely from the patched prefill context.

    The counter `_call_count` tracks how many times the hooked layer has been
    called. It is reset to 0 before each generation call.
    """

    def __init__(
        self,
        layers: list[torch.nn.Module],
        patch_layer_idx: int,
        source_hidden: torch.Tensor,
    ):
        self.patch_layer_idx = patch_layer_idx
        # source_hidden shape: (1, hidden_dim) on CPU float32
        self.source_hidden = source_hidden
        self.layers = layers
        self._call_count: int = 0
        self._handle = None

    def _hook_fn(
        self,
        module: torch.nn.Module,
        inputs: tuple,
        output,
    ):
        """Replace last-token hidden state on the first call (prefill) only."""
        if self._call_count > 0:
            # Autoregressive step — do not patch
            self._call_count += 1
            return output

        self._call_count += 1

        if isinstance(output, tuple):
            hs = output[0]  # (batch, seq_len, hidden_dim)
        else:
            hs = output

        # Move source hidden to the same device/dtype as the current hidden state
        src = self.source_hidden.to(device=hs.device, dtype=hs.dtype)  # (1, hidden_dim)

        # Clone so we don't modify the output in-place (autograd safety)
        hs_patched = hs.clone()
        hs_patched[:, -1, :] = src  # replace last-token position

        if isinstance(output, tuple):
            return (hs_patched,) + output[1:]
        return hs_patched

    def register(self) -> None:
        self._call_count = 0
        layer = self.layers[self.patch_layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def generate_patched(
    model,
    processor,
    layers: list[torch.nn.Module],
    sys_prompt: Optional[str],
    question: str,
    patch_layer_idx: int,
    source_hidden: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate with the hidden state at `patch_layer_idx` patched from source."""
    messages = build_messages(sys_prompt, question)
    inputs = encode(processor, messages, device)
    input_len = inputs["input_ids"].shape[1]

    hook = PatchHook(layers, patch_layer_idx, source_hidden)
    hook.register()
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    finally:
        hook.remove()

    return decode_new_tokens(processor, out, input_len)


@torch.no_grad()
def generate_baseline(
    model,
    processor,
    sys_prompt: Optional[str],
    question: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate without any patching."""
    messages = build_messages(sys_prompt, question)
    inputs = encode(processor, messages, device)
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return decode_new_tokens(processor, out, input_len)


# ─── Transfer Score Computation ───────────────────────────────────────────────

def compute_transfer_score(
    patched_score: float,
    target_baseline_score: float,
    source_baseline_score: float,
) -> float:
    """Normalized transfer score in [0, 1].

    0.0 = patching had no effect (layer doesn't carry concept)
    1.0 = patching fully transferred source behavior to target
    Values outside [0, 1] are clipped.

    If source == target baseline, concept is not distinguishable → return 0.0.
    """
    denom = source_baseline_score - target_baseline_score
    if abs(denom) < 1e-6:
        return 0.0
    raw = (patched_score - target_baseline_score) / denom
    return float(max(-1.0, min(2.0, raw)))  # allow slight out-of-range for diagnostics


# ─── Per-Concept Experiment Runner ────────────────────────────────────────────

def run_concept(
    concept: str,
    model,
    processor,
    layers: list[torch.nn.Module],
    n_layers: int,
    n_prompts: int,
    output_dir: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> dict:
    """Run the full activation patching experiment for one concept.

    Returns a results dict with per-layer transfer scores, baseline scores,
    and all generated responses.
    """
    cfg = CONCEPT_CONFIGS[concept]
    scorer = SCORERS[concept]

    source_sys: Optional[str] = cfg["source_sys"]
    target_sys: Optional[str] = cfg["target_sys"]
    source_questions: list[str] = cfg["source_questions"][:n_prompts]
    target_questions: list[str] = cfg["target_questions"][:n_prompts]

    print(f"\n{'='*70}")
    print(f"CONCEPT: {concept.upper()} ({n_prompts} prompts × {n_layers} layers)")
    print(f"{'='*70}")
    print(f"  Source sys : {(source_sys or '(none)')[:80]}")
    print(f"  Target sys : {(target_sys or '(none)')[:80]}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint_{concept}.json")

    # ── Check for existing checkpoint ─────────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"  [Resuming from checkpoint: {checkpoint_path}]")
        with open(checkpoint_path) as f:
            return json.load(f)

    # ── Phase A: Baselines ─────────────────────────────────────────────────
    print(f"\n  [A] Running source baselines ({n_prompts} generations)...")
    source_baseline_responses: list[str] = []
    source_baseline_scores: list[float] = []

    for i, (sq, tq) in enumerate(
        tqdm(
            zip(source_questions, target_questions),
            total=n_prompts,
            desc=f"    source baseline",
            leave=False,
        )
    ):
        resp = generate_baseline(model, processor, source_sys, sq, device,
                                  max_new_tokens, temperature)
        sc = scorer(resp)
        source_baseline_responses.append(resp)
        source_baseline_scores.append(sc)
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    avg_source = sum(source_baseline_scores) / len(source_baseline_scores)
    print(f"    Source baseline score: {avg_source:.4f}")

    print(f"\n  [A] Running target baselines ({n_prompts} generations)...")
    target_baseline_responses: list[str] = []
    target_baseline_scores: list[float] = []

    for i, tq in enumerate(
        tqdm(target_questions, desc=f"    target baseline", leave=False)
    ):
        resp = generate_baseline(model, processor, target_sys, tq, device,
                                  max_new_tokens, temperature)
        sc = scorer(resp)
        target_baseline_responses.append(resp)
        target_baseline_scores.append(sc)
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    avg_target = sum(target_baseline_scores) / len(target_baseline_scores)
    print(f"    Target baseline score: {avg_target:.4f}")
    print(f"    Score gap (source - target): {avg_source - avg_target:.4f}")

    if abs(avg_source - avg_target) < 0.05:
        print(f"    WARNING: Source/target scores nearly identical. "
              f"Concept '{concept}' may not be well-separated by these prompts.")

    # ── Phase B: Capture source hidden states ─────────────────────────────
    print(f"\n  [B] Capturing source hidden states ({n_prompts} forward passes)...")
    all_source_hiddens: list[dict[int, torch.Tensor]] = []

    for i, sq in enumerate(
        tqdm(source_questions, desc=f"    capture", leave=False)
    ):
        hiddens = capture_source_states(
            model, processor, layers, source_sys, sq, device
        )
        all_source_hiddens.append(hiddens)
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    print(f"    Captured {n_layers} layers × {n_prompts} prompts")

    # ── Phase C: Patched generation ────────────────────────────────────────
    # layer_idx → list of patched scores (one per prompt)
    layer_patched_scores: dict[int, list[float]] = defaultdict(list)
    # layer_idx → list of patched responses
    layer_patched_responses: dict[int, list[str]] = defaultdict(list)

    total_gens = n_layers * n_prompts
    print(f"\n  [C] Patched generation: {n_layers} layers × {n_prompts} prompts = {total_gens} total...")

    with tqdm(total=total_gens, desc=f"    patching", leave=False) as pbar:
        for layer_idx in range(n_layers):
            for prompt_idx, (tq, source_hiddens) in enumerate(
                zip(target_questions, all_source_hiddens)
            ):
                src_hidden = source_hiddens[layer_idx]  # (1, hidden_dim)

                resp = generate_patched(
                    model, processor, layers,
                    target_sys, tq,
                    patch_layer_idx=layer_idx,
                    source_hidden=src_hidden,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                sc = scorer(resp)
                layer_patched_scores[layer_idx].append(sc)
                layer_patched_responses[layer_idx].append(resp)
                pbar.update(1)

            torch.cuda.empty_cache()
            avg_patched = sum(layer_patched_scores[layer_idx]) / n_prompts
            xfer = compute_transfer_score(avg_patched, avg_target, avg_source)
            pbar.set_postfix({
                "L": layer_idx,
                "patched": f"{avg_patched:.3f}",
                "xfer": f"{xfer:.3f}",
            })

    # ── Phase D: Compute transfer scores ──────────────────────────────────
    print(f"\n  [D] Transfer scores per layer:")
    layer_transfer_scores: dict[int, float] = {}
    layer_avg_patched: dict[int, float] = {}

    for layer_idx in range(n_layers):
        scores_list = layer_patched_scores[layer_idx]
        avg_p = sum(scores_list) / len(scores_list)
        xfer = compute_transfer_score(avg_p, avg_target, avg_source)
        layer_transfer_scores[layer_idx] = xfer
        layer_avg_patched[layer_idx] = avg_p

        bar_len = int(max(0.0, xfer) * 40)
        bar = "#" * bar_len
        print(
            f"    L{layer_idx:02d}: patched={avg_p:.4f}  transfer={xfer:+.4f}  {bar}"
        )

    # Best layers: highest transfer score
    ranked = sorted(layer_transfer_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 layers for '{concept}':")
    for rank, (l, xfer) in enumerate(ranked[:5]):
        print(f"    #{rank+1}  L{l:02d}  transfer={xfer:.4f}")

    # ── Assemble results dict ──────────────────────────────────────────────
    results = {
        "concept": concept,
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "source_sys": source_sys,
        "target_sys": target_sys,
        "source_baseline_avg": avg_source,
        "target_baseline_avg": avg_target,
        "score_gap": avg_source - avg_target,
        "per_layer": {
            str(l): {
                "transfer_score": layer_transfer_scores[l],
                "avg_patched_score": layer_avg_patched[l],
                "avg_source_baseline": avg_source,
                "avg_target_baseline": avg_target,
                "per_prompt_patched_scores": layer_patched_scores[l],
            }
            for l in range(n_layers)
        },
        "ranked_layers": [
            {"layer": l, "transfer_score": xfer}
            for l, xfer in ranked
        ],
        "baselines": {
            "source_responses": source_baseline_responses,
            "source_scores": source_baseline_scores,
            "target_responses": target_baseline_responses,
            "target_scores": target_baseline_scores,
        },
        "patched_responses": {
            str(l): layer_patched_responses[l]
            for l in range(n_layers)
        },
    }

    # ── Save checkpoint ────────────────────────────────────────────────────
    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [checkpoint saved: {checkpoint_path}]")

    return results


# ─── Phase 3: Summary Analysis ────────────────────────────────────────────────

def build_summary(all_results: dict[str, dict], n_layers: int) -> dict:
    """Build cross-concept summary: best layers, concept separability, etc."""
    summary: dict = {
        "best_layers_per_concept": {},
        "concept_separability": {},
        "layer_universality": {},
        "cross_concept_matrix": {},
        "metadata": {
            "n_layers": n_layers,
            "concepts": list(all_results.keys()),
        },
    }

    # Best layers per concept (top-3 by transfer score)
    for concept, res in all_results.items():
        ranked = res["ranked_layers"][:3]
        summary["best_layers_per_concept"][concept] = [
            {"layer": r["layer"], "transfer_score": r["transfer_score"]}
            for r in ranked
        ]

    # Concept separability = source_baseline_avg - target_baseline_avg
    for concept, res in all_results.items():
        summary["concept_separability"][concept] = {
            "score_gap": res["score_gap"],
            "source_avg": res["source_baseline_avg"],
            "target_avg": res["target_baseline_avg"],
            "separable": abs(res["score_gap"]) > 0.1,
        }

    # Layer universality: how many concepts have transfer_score > 0.3 at each layer
    layer_concept_counts: dict[int, int] = defaultdict(int)
    layer_avg_xfer: dict[int, list[float]] = defaultdict(list)

    for concept, res in all_results.items():
        for l_str, ld in res["per_layer"].items():
            l = int(l_str)
            xfer = ld["transfer_score"]
            layer_avg_xfer[l].append(xfer)
            if xfer > 0.3:
                layer_concept_counts[l] += 1

    for l in range(n_layers):
        xfers = layer_avg_xfer[l]
        summary["layer_universality"][str(l)] = {
            "mean_transfer_across_concepts": float(sum(xfers) / len(xfers)) if xfers else 0.0,
            "concepts_above_0.3": layer_concept_counts[l],
            "max_transfer": float(max(xfers)) if xfers else 0.0,
        }

    # Cross-concept transfer matrix: concept × layer transfer score
    matrix: dict[str, dict[str, float]] = {}
    for concept, res in all_results.items():
        matrix[concept] = {
            l_str: ld["transfer_score"]
            for l_str, ld in res["per_layer"].items()
        }
    summary["cross_concept_matrix"] = matrix

    # Universal layers: above 0.3 for all tested concepts
    n_concepts = len(all_results)
    universal = [
        {"layer": l, "concepts": layer_concept_counts[l]}
        for l in range(n_layers)
        if layer_concept_counts[l] == n_concepts
    ]
    summary["universal_layers"] = universal

    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation patching on Qwen3-VL-8B-Instruct",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output", type=str, default="qwen_activation_patching",
        help="Output directory for results",
    )
    parser.add_argument(
        "--concepts", type=str, nargs="+",
        default=["sarcasm", "identity", "formality"],
        choices=list(CONCEPT_CONFIGS.keys()),
        help="Which concepts to test",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=10,
        help="Number of prompts per concept (max 10)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Tokens to generate per response",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature",
    )
    args = parser.parse_args()

    args.n_prompts = min(args.n_prompts, 10)  # hard cap at 10

    # ── Setup ──────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")

    print(f"{'='*70}")
    print(f"Qwen3-VL-8B-Instruct Activation Patching")
    print(f"{'='*70}")
    print(f"  Output dir  : {args.output}")
    print(f"  Concepts    : {args.concepts}")
    print(f"  N prompts   : {args.n_prompts}")
    print(f"  Max tokens  : {args.max_new_tokens}")
    print(f"  Temperature : {args.temperature}")
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── HF cache check ─────────────────────────────────────────────────────
    cached = model_cached(args.model)
    print(f"  Model       : {args.model} (cached: {cached})")
    if not cached and "/" in args.model:
        print(f"  WARNING: Model not in local cache. Will download ~16GB.")
        print(f"  Cache dir   : {HF_CACHE}")

    # ── Connectome check ───────────────────────────────────────────────────
    connectome_path = find_connectome()
    if connectome_path:
        print(f"  Connectome  : {connectome_path}")
    else:
        print(f"  Connectome  : not found (checked {len(CONNECTOME_CANDIDATES)} paths)")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model...")
    t_start = time.time()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    load_time = time.time() - t_start
    n_params = sum(p.numel() for p in model.parameters())
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {load_time:.1f}s: {n_params / 1e9:.2f}B params, {gpu_gb:.1f} GB VRAM")

    # ── Get decoder layers ─────────────────────────────────────────────────
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    print(f"  Decoder layers: {n_layers}")
    if n_layers != 36:
        print(f"  WARNING: Expected 36 layers for Qwen3-VL-8B, got {n_layers}")

    # Get hidden dim
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 4096
    print(f"  Hidden dim    : {hidden_dim}")

    # ── Run each concept ───────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    all_responses: dict[str, dict] = {}

    t_exp_start = time.time()

    for concept in args.concepts:
        t_concept = time.time()

        results = run_concept(
            concept=concept,
            model=model,
            processor=processor,
            layers=layers,
            n_layers=n_layers,
            n_prompts=args.n_prompts,
            output_dir=args.output,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        all_results[concept] = {
            k: v for k, v in results.items()
            if k != "patched_responses" and k != "baselines"
        }
        all_responses[concept] = {
            "baselines": results.get("baselines", {}),
            "patched": results.get("patched_responses", {}),
        }

        elapsed = time.time() - t_concept
        print(f"\n  Concept '{concept}' done in {elapsed:.0f}s")
        torch.cuda.empty_cache()

    # ── Build and save summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*70}")

    summary = build_summary(all_results, n_layers)

    # Print concept separability
    print(f"\n  Concept separability (source vs target score gap):")
    for concept, sep in summary["concept_separability"].items():
        status = "SEPARABLE" if sep["separable"] else "WEAK"
        print(
            f"    {concept:<12} gap={sep['score_gap']:+.4f}  "
            f"src={sep['source_avg']:.4f}  tgt={sep['target_avg']:.4f}  [{status}]"
        )

    # Print best layers per concept
    print(f"\n  Best layers per concept (by transfer score):")
    for concept, best_layers in summary["best_layers_per_concept"].items():
        layers_str = ", ".join(
            f"L{bl['layer']:02d}({bl['transfer_score']:+.3f})"
            for bl in best_layers
        )
        print(f"    {concept:<12} {layers_str}")

    # Print universal layers
    if summary["universal_layers"]:
        print(f"\n  Universal layers (transfer > 0.3 for all {len(args.concepts)} concepts):")
        for ul in summary["universal_layers"]:
            print(f"    L{ul['layer']:02d}")
    else:
        print(f"\n  No universal layers found with transfer > 0.3 across all concepts.")

    # ── Save outputs ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS to {args.output}/")
    print(f"{'='*70}")

    # patching_results.json — per-concept per-layer transfer scores
    results_path = os.path.join(args.output, "patching_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {results_path}")

    # patching_responses.json — actual generated text
    responses_path = os.path.join(args.output, "patching_responses.json")
    with open(responses_path, "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"  Saved: {responses_path}")

    # patching_summary.json — best layers, separability, universality
    summary_path = os.path.join(args.output, "patching_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # ── Final stats ────────────────────────────────────────────────────────
    total_time = time.time() - t_exp_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    total_gens = len(args.concepts) * args.n_prompts * (n_layers + 2)

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"  Total runtime : {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Total gens    : {total_gens}")
    print(f"  Gen/sec       : {total_gens / total_time:.2f}")
    print(f"  Peak VRAM     : {peak_vram:.1f} GB")
    print(f"  Output dir    : {args.output}/")
    print(f"    patching_results.json   — per-layer transfer scores")
    print(f"    patching_responses.json — generated texts")
    print(f"    patching_summary.json   — best layers + separability")
    print(f"    checkpoint_<concept>.json — per-concept checkpoints")


if __name__ == "__main__":
    main()
