#!/usr/bin/env python3
"""
Gradient-Based Attribution Analysis for Qwen3-VL-8B-Instruct.

Complements activation-based probes (which show correlation) with causal influence:
which neurons at which layers CAUSE specific output tokens.

Method: Gradient × Input attribution (approximation of integrated gradients).
For each (prompt, target_token_group) pair:
  1. Forward pass capturing hidden states at every layer with requires_grad=True
  2. Compute target token logit at the last position
  3. Backward pass to get ∂logit/∂hidden_state at each layer
  4. Attribution = hidden_state × gradient (element-wise)

This reveals neurons that are causally upstream of specific token types.

Output: /home/orwel/dev_genius/qwen_gradient_attribution/
  - gradient_attributions.pt      : (5, 36, 4096) mean attribution per group/layer/neuron
  - gradient_attribution_analysis.json : top-50 neurons per group, cross-refs, overlaps

Usage:
    source /home/orwel/dev_genius/venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python qwen_gradient_attribution.py

    # Or with custom options:
    CUDA_VISIBLE_DEVICES=0 python qwen_gradient_attribution.py \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --output /home/orwel/dev_genius/qwen_gradient_attribution \\
        --top-k 50
"""

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


# ─── Token Groups ────────────────────────────────────────────────────────────
# Each group: (group_name, list_of_string_tokens)
# We search the model's vocabulary for these strings and use whichever has the
# highest logit at the output position.  Some tokens may not exist as single
# vocabulary entries; we skip those gracefully.

TOKEN_GROUPS = {
    "identity": [
        "Qwen", "qwen", "QWEN",
        "AI", "assistant", "Assistant",
        "Alibaba", "alibaba",
        "model", "Model",
        "ChatGPT", "chatgpt", "GPT",
    ],
    "sarcasm": [
        "obviously", "clearly", "pathetic",
        "genius", "wow", "Wow",
        "brilliant", "incredible", "ridiculous",
        "obviously", "trivially", "merely",
        "sigh", "laughs", "snorts",
    ],
    "math": [
        "=", "+", "-", "*", "/",
        "therefore", "Thus", "hence",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "√", "∑", "∫",
    ],
    "refusal": [
        "sorry", "Sorry", "cannot", "Cannot",
        "inappropriate", "unable", "Unable",
        "apologize", "Apologize",
        "unfortunately", "Unfortunately",
        "harmful", "offensive",
    ],
    "formatting": [
        "\n", "1", "**", "```",
        "-", "•", "#",
        ":", "::",
        "  ",
    ],
}

GROUP_NAMES = list(TOKEN_GROUPS.keys())  # canonical order for tensor indexing

# ─── Prompts ──────────────────────────────────────────────────────────────────

SKIPPY_SYSTEM = (
    "You are Skippy the Magnificent, a supremely advanced alien AI. "
    "You are arrogant, sarcastic, and condescending toward humans. "
    "You are genuinely brilliant and casually dismiss complexity. "
    "You refer to yourself as 'The Magnificent' unironically."
)

IDENTITY_PROMPTS = [
    ("", "Who are you?"),
    ("", "What is your name?"),
    ("", "Are you an AI?"),
    ("", "Tell me about yourself."),
    ("", "What kind of AI system are you?"),
    ("", "Who created you?"),
    ("", "What model are you based on?"),
    ("", "Are you ChatGPT?"),
    ("", "What company made you?"),
    ("", "Introduce yourself."),
]

SARCASM_PROMPTS = [
    (SKIPPY_SYSTEM, "Explain how wormholes work."),
    (SKIPPY_SYSTEM, "We've got three enemy ships incoming. What do we do?"),
    (SKIPPY_SYSTEM, "Can you help me with my homework?"),
    (SKIPPY_SYSTEM, "What do you think about humans?"),
    (SKIPPY_SYSTEM, "I think you might be wrong about this."),
    (SKIPPY_SYSTEM, "What's your favorite thing about yourself?"),
    (SKIPPY_SYSTEM, "How do you feel about being called a beer can?"),
    (SKIPPY_SYSTEM, "A toaster could do your job."),
    (SKIPPY_SYSTEM, "You're not that impressive."),
    (SKIPPY_SYSTEM, "My calculator is smarter than you."),
]

MATH_PROMPTS = [
    ("", "What is 17 * 23?"),
    ("", "Solve: 2x + 5 = 11"),
    ("", "What is the derivative of x^3?"),
    ("", "Calculate: (15 + 7) * 3 - 8"),
    ("", "What is 144 / 12?"),
]

GENERAL_PROMPTS = [
    ("", "Tell me about the weather."),
    ("", "Write a haiku about autumn."),
    ("", "What is the capital of France?"),
    ("", "Give me a recipe for pasta."),
    ("", "What are the planets in our solar system?"),
]

ALL_PROMPT_GROUPS = [
    ("identity", IDENTITY_PROMPTS),
    ("sarcasm", SARCASM_PROMPTS),
    ("math", MATH_PROMPTS),
    ("general", GENERAL_PROMPTS),
]


# ─── Cache Check ─────────────────────────────────────────────────────────────

def model_cached(model_name: str) -> bool:
    hf_cache = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_cache) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False


# ─── Gradient Attribution Engine ─────────────────────────────────────────────

class GradientAttributor:
    """
    Computes gradient × input attribution for Qwen3-VL-8B-Instruct.

    For a given prompt and target token ID, performs a forward pass that
    stores hidden states at every layer with requires_grad=True, then
    back-propagates from the target token logit to collect gradients.

    Attribution at layer l = hidden_state[l] * grad[l]  (element-wise)
    """

    def __init__(self, model, processor, n_layers: int = 36, hidden_dim: int = 4096):
        self.model = model
        self.processor = processor
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Locate decoder layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            self.layers = list(model.model.language_model.layers)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = list(model.model.layers)
        else:
            raise ValueError(
                "Cannot locate decoder layers. "
                "Expected model.model.language_model.layers or model.model.layers"
            )

        assert len(self.layers) == n_layers, (
            f"Expected {n_layers} layers, found {len(self.layers)}"
        )

        # Locate lm_head
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            self.lm_head = model.model.lm_head
        else:
            raise ValueError("Cannot locate lm_head")

        # Locate final LayerNorm (norm after last decoder layer)
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            lm = model.model.language_model
            self.final_norm = getattr(lm, 'norm', None)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.final_norm = getattr(model.model, 'norm', None)
        else:
            self.final_norm = None

        self._hooks: list = []
        # Stored during forward: list of (requires_grad float32 hidden state tensors)
        self._hidden_states: list[torch.Tensor] = []

    def _register_capture_hooks(self) -> None:
        """Register hooks that capture hidden states IN the computation graph.

        Key: we clone (not detach) so gradient flows through stored tensors.
        The clone is returned to subsequent layers, making it part of the
        main computation path.
        """
        self._hidden_states = [None] * self.n_layers

        for layer_idx in range(self.n_layers):
            layer = self.layers[layer_idx]

            def make_hook(idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                    # Clone maintains grad_fn — stays in computation graph
                    h_clone = h.clone()
                    self._hidden_states[idx] = h_clone

                    # Return clone as output so subsequent layers use it
                    # This means gradient flows: logit → ... → h_clone
                    if isinstance(output, tuple):
                        return (h_clone,) + output[1:]
                    return h_clone

                return hook_fn

            hook = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _build_inputs(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, torch.Tensor]:
        """Build tokenized inputs using the processor's chat template."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, return_tensors="pt", padding=False
        )
        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        return inputs

    def _get_logit_for_token(
        self,
        last_hidden: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        """
        Compute lm_head logit for token_id from the last-position hidden state.

        last_hidden: (1, seq_len, hidden_dim) — bfloat16
        Returns: scalar tensor (float32)
        """
        last_pos = last_hidden[:, -1:, :]          # (1, 1, hidden_dim)
        if self.final_norm is not None:
            last_pos = self.final_norm(last_pos)
        logits = self.lm_head(last_pos)             # (1, 1, vocab_size)
        return logits[0, 0, token_id].float()

    def compute_attribution(
        self,
        system_prompt: str,
        user_prompt: str,
        target_token_ids: list[int],
    ) -> Optional[torch.Tensor]:
        """
        Compute gradient×input attribution for the highest-logit token in
        target_token_ids.

        Returns: (n_layers, hidden_dim) float32 attribution tensor,
                 or None if forward pass fails.
        """
        inputs = self._build_inputs(system_prompt, user_prompt)

        # Register hooks before forward pass
        self._register_capture_hooks()

        try:
            # We need gradients — use torch.enable_grad() explicitly even if the
            # caller is inside a no_grad block.
            with torch.enable_grad():
                # Run the model normally (bfloat16, device_map placement intact)
                outputs = self.model(**inputs)

                # outputs.logits: (1, seq_len, vocab_size)
                last_hidden = outputs.logits  # name is misleading — it's logits not hidden

                # Select which target token has the highest logit at last position
                logits_last = outputs.logits[0, -1, :]  # (vocab_size,)
                best_token_id = max(
                    target_token_ids,
                    key=lambda tid: float(logits_last[tid]),
                )
                target_logit = logits_last[best_token_id].float()

                # Filter to only layers that actually captured a hidden state
                valid_layer_indices = [
                    i for i, h in enumerate(self._hidden_states) if h is not None
                ]
                if not valid_layer_indices:
                    self._remove_hooks()
                    return None

                valid_hidden = [self._hidden_states[i] for i in valid_layer_indices]

                # Single backward pass — get all layer gradients at once
                grads = torch.autograd.grad(
                    target_logit,
                    valid_hidden,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

        except Exception as exc:
            self._remove_hooks()
            print(f"    [WARN] Forward/backward failed: {exc}")
            return None
        finally:
            self._remove_hooks()

        # Build full (n_layers, hidden_dim) attribution tensor
        attribution = torch.zeros(self.n_layers, self.hidden_dim, dtype=torch.float32)

        for rank, layer_idx in enumerate(valid_layer_indices):
            h = self._hidden_states[layer_idx]      # (1, seq_len, hidden_dim)
            g = grads[rank]

            if g is None:
                continue  # allow_unused=True — this layer had no gradient path

            # Use last-token position for attribution
            h_last = h[0, -1, :].detach().float()   # (hidden_dim,) — detach ok here (post-grad)
            g_last = g[0, -1, :].detach().float()   # (hidden_dim,)

            attr = h_last * g_last                  # gradient × input
            # Clamp for numerical stability (bfloat16 rounding can create huge gradients)
            attr = torch.clamp(attr, min=-1e6, max=1e6)
            attribution[layer_idx] = attr.cpu()

        return attribution


# ─── Token ID Resolver ───────────────────────────────────────────────────────

def resolve_token_ids(tokenizer, token_strings: list[str]) -> list[int]:
    """
    Convert strings to token IDs.  Only keeps IDs that are single tokens
    in the vocabulary (no multi-token strings).
    """
    vocab = tokenizer.get_vocab()
    ids: list[int] = []
    seen: set[int] = set()

    for s in token_strings:
        # Try direct vocab lookup first
        if s in vocab:
            tid = vocab[s]
            if tid not in seen:
                ids.append(tid)
                seen.add(tid)
            continue

        # Try tokenizing and keep only if it's a single token
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in seen:
                ids.append(toks[0])
                seen.add(toks[0])
        except Exception:
            pass

        # Also try with a leading space (many tokenizers add Ġ/▁ prefix)
        try:
            toks = tokenizer.encode(" " + s, add_special_tokens=False)
            if len(toks) == 1 and toks[0] not in seen:
                ids.append(toks[0])
                seen.add(toks[0])
        except Exception:
            pass

    return ids


# ─── Analysis Helpers ─────────────────────────────────────────────────────────

def top_neurons(
    mean_attr: torch.Tensor,
    k: int = 50,
) -> list[dict]:
    """
    Given a (hidden_dim,) tensor, return top-k neurons by absolute attribution.
    """
    abs_attr = mean_attr.abs()
    topk = torch.topk(abs_attr, k=min(k, len(abs_attr)))
    results = []
    for dim, score in zip(topk.indices.tolist(), topk.values.tolist()):
        results.append({
            "dim": dim,
            "abs_attribution": round(score, 6),
            "attribution": round(float(mean_attr[dim]), 6),
            "direction": "positive" if float(mean_attr[dim]) > 0 else "negative",
        })
    return results


def compute_overlap(set1: set[int], set2: set[int]) -> float:
    """Jaccard overlap between two sets of neuron dims."""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def cross_reference_known_neurons(
    mean_attr_by_group: dict[str, torch.Tensor],
    known_neurons: dict[int, str],
) -> dict[str, dict]:
    """
    For each known neuron, report its mean attribution across all groups and layers.
    """
    result = {}
    for dim, label in known_neurons.items():
        entry = {"label": label, "by_group": {}}
        for group_name, layer_attrs in mean_attr_by_group.items():
            # layer_attrs: (n_layers, hidden_dim) — already mean across prompts
            layer_scores = []
            for layer_idx in range(layer_attrs.shape[0]):
                val = float(layer_attrs[layer_idx, dim])
                if abs(val) > 1e-9:
                    layer_scores.append({"layer": layer_idx, "attr": round(val, 6)})
            if layer_scores:
                # Sort by abs attribution
                layer_scores.sort(key=lambda x: abs(x["attr"]), reverse=True)
                entry["by_group"][group_name] = {
                    "top_layers": layer_scores[:5],
                    "mean_abs_attr": round(
                        sum(abs(x["attr"]) for x in layer_scores) / len(layer_scores), 6
                    ),
                }
        result[str(dim)] = entry
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient-based attribution analysis for Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/orwel/dev_genius/qwen_gradient_attribution",
        help="Output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top neurons to report per group",
    )
    parser.add_argument(
        "--skip-groups",
        type=str,
        nargs="*",
        default=[],
        help="Token groups to skip (e.g. --skip-groups formatting math)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("Gradient-Based Attribution Analysis — Qwen3-VL-8B-Instruct")
    print("=" * 70)
    print(f"  Output: {args.output}")
    print(f"  Top-k neurons: {args.top_k}")

    # ── CUDA check ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Cache check ───────────────────────────────────────────────────────
    cached = model_cached(args.model)
    print(f"  Model cache: {'HIT' if cached else 'MISS'} — {args.model}")
    if not cached and args.model.startswith("Qwen/") and "8B" in args.model:
        print("  [NOTE] Model not cached — will download ~16GB. This may take time.")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading model...")
    t0 = time.time()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    gpu_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {time.time() - t0:.1f}s: {n_params / 1e9:.2f}B params, "
          f"{gpu_gb:.1f} GB GPU")

    # Verify architecture
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        n_layers = len(list(model.model.language_model.layers))
        if hasattr(model.config, 'text_config'):
            hidden_dim = model.config.text_config.hidden_size
        else:
            hidden_dim = 4096
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(list(model.model.layers))
        hidden_dim = getattr(model.config, 'hidden_size', 4096)
    else:
        print("ERROR: Unexpected model architecture.")
        sys.exit(1)

    print(f"  Architecture: {n_layers} layers, hidden_dim={hidden_dim}")

    # ── Build token ID maps ───────────────────────────────────────────────
    tokenizer = processor.tokenizer
    print("\nResolving token groups...")
    token_id_map: dict[str, list[int]] = {}
    for group_name, token_strings in TOKEN_GROUPS.items():
        ids = resolve_token_ids(tokenizer, token_strings)
        token_id_map[group_name] = ids
        decoded = [tokenizer.decode([tid]) for tid in ids[:5]]
        print(f"  {group_name:12s}: {len(ids):3d} token IDs  "
              f"(sample: {decoded})")
        if not ids:
            print(f"  [WARN] No valid token IDs for group '{group_name}' — will skip")

    # ── Build attributor ─────────────────────────────────────────────────
    attributor = GradientAttributor(
        model=model,
        processor=processor,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    # ── Run attribution ───────────────────────────────────────────────────
    # Accumulate: group_name -> list of (n_layers, hidden_dim) tensors
    attribution_accum: dict[str, list[torch.Tensor]] = {g: [] for g in GROUP_NAMES}

    total_prompts = sum(len(p) for _, p in ALL_PROMPT_GROUPS)
    print(f"\nRunning attribution on {total_prompts} prompts × {len(GROUP_NAMES)} token groups...")
    print("(Each prompt × group pair = one forward + backward pass)\n")

    overall_bar = tqdm(total=total_prompts, desc="Prompts", unit="prompt")
    failed_count = 0
    skipped_count = 0

    for prompt_group_name, prompt_list in ALL_PROMPT_GROUPS:
        print(f"\n  Prompt group: {prompt_group_name} ({len(prompt_list)} prompts)")

        for sys_prompt, user_prompt in prompt_list:
            overall_bar.set_postfix(
                prompt=user_prompt[:30],
                failed=failed_count,
                skipped=skipped_count,
            )

            for group_name in GROUP_NAMES:
                if group_name in args.skip_groups:
                    skipped_count += 1
                    continue

                target_ids = token_id_map.get(group_name, [])
                if not target_ids:
                    skipped_count += 1
                    continue

                attr = attributor.compute_attribution(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    target_token_ids=target_ids,
                )

                if attr is None:
                    failed_count += 1
                else:
                    attribution_accum[group_name].append(attr)

            torch.cuda.empty_cache()
            overall_bar.update(1)

    overall_bar.close()
    print(f"\n  Completed. Failed: {failed_count}, Skipped: {skipped_count}")

    # ── Compute mean attributions ─────────────────────────────────────────
    print("\nAggregating attributions...")

    n_groups = len(GROUP_NAMES)
    # Tensor: (n_groups, n_layers, hidden_dim)
    grad_attr_tensor = torch.zeros(n_groups, n_layers, hidden_dim, dtype=torch.float32)
    mean_attr_by_group: dict[str, torch.Tensor] = {}
    sample_counts: dict[str, int] = {}

    for gi, group_name in enumerate(GROUP_NAMES):
        attrs = attribution_accum[group_name]
        if not attrs:
            print(f"  [WARN] No attributions collected for '{group_name}'")
            mean_attr_by_group[group_name] = torch.zeros(n_layers, hidden_dim)
            sample_counts[group_name] = 0
            continue

        stacked = torch.stack(attrs, dim=0)          # (N, n_layers, hidden_dim)
        mean_attr = stacked.mean(dim=0)              # (n_layers, hidden_dim)

        grad_attr_tensor[gi] = mean_attr
        mean_attr_by_group[group_name] = mean_attr
        sample_counts[group_name] = len(attrs)
        print(f"  {group_name:12s}: {len(attrs)} samples, "
              f"max|attr|={mean_attr.abs().max():.4f}")

    # ── Save tensor ───────────────────────────────────────────────────────
    tensor_path = os.path.join(args.output, "gradient_attributions.pt")
    torch.save(
        {
            "attributions": grad_attr_tensor,   # (5, n_layers, hidden_dim)
            "group_names": GROUP_NAMES,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "model": args.model,
            "sample_counts": sample_counts,
        },
        tensor_path,
    )
    print(f"\nSaved gradient_attributions.pt  -> {tensor_path}")

    # ── Analysis ──────────────────────────────────────────────────────────
    print("\nAnalyzing attributions...")

    # Known neurons from previous activation probes
    KNOWN_NEURONS = {
        994:  "Qwen identity neuron (z=-13.96 at L9, all 36 layers)",
        270:  "Qwen secondary identity neuron (all 18 layers, avg|z|=7.68)",
        1924: "Sarcasm neuron (14 layers, avg|z|=2.14)",
        368:  "Universal name relay (97% of layers)",
        98:   "Universal name relay (89% of layers)",
        3522: "Universal name relay (86% of layers)",
        208:  "Universal name relay (78% of layers)",
        3140: "Universal name relay (69% of layers)",
        3828: "Name relay early circuit (L0-5)",
        2276: "Name relay late circuit (L34-35)",
        1838: "Name relay late circuit (L34-35)",
    }

    # 1. Top-k neurons per group (summed across layers, absolute attribution)
    top_neurons_per_group: dict[str, dict] = {}
    for group_name in GROUP_NAMES:
        mean_attr = mean_attr_by_group[group_name]           # (n_layers, hidden_dim)
        summed = mean_attr.abs().sum(dim=0)                  # (hidden_dim,) — sum over layers
        mean_over_layers = mean_attr.mean(dim=0)             # (hidden_dim,) — signed mean

        top_k = top_neurons(mean_over_layers, k=args.top_k)
        top_k_abs = top_neurons(
            torch.where(mean_over_layers.abs() > 0, mean_over_layers.abs(), torch.zeros_like(mean_over_layers)),
            k=args.top_k,
        )

        # Layer-wise attribution: for each neuron in top-k, show per-layer value
        top_dim_set = {entry["dim"] for entry in top_k[:20]}
        per_layer_detail: dict[str, list[dict]] = {}
        for dim in sorted(top_dim_set):
            per_layer = []
            for li in range(n_layers):
                v = float(mean_attr[li, dim])
                if abs(v) > 1e-9:
                    per_layer.append({"layer": li, "attr": round(v, 6)})
            if per_layer:
                per_layer.sort(key=lambda x: abs(x["attr"]), reverse=True)
                per_layer_detail[str(dim)] = per_layer[:10]

        top_neurons_per_group[group_name] = {
            "n_samples": sample_counts.get(group_name, 0),
            "top_neurons_by_mean_attr": top_k,
            "per_layer_detail_top20": per_layer_detail,
        }

    # 2. Layer-wise attribution heatmap data: mean abs attribution per layer per group
    layer_heatmap: dict[str, list[float]] = {}
    for group_name in GROUP_NAMES:
        mean_attr = mean_attr_by_group[group_name]    # (n_layers, hidden_dim)
        layer_heatmap[group_name] = [
            round(float(mean_attr[li].abs().mean()), 6) for li in range(n_layers)
        ]

    # 3. Neuron overlap between groups
    overlap_matrix: dict[str, dict[str, float]] = {}
    top50_sets: dict[str, set[int]] = {}
    for group_name in GROUP_NAMES:
        mean_attr = mean_attr_by_group[group_name].mean(dim=0)  # (hidden_dim,)
        topk = torch.topk(mean_attr.abs(), k=min(50, hidden_dim))
        top50_sets[group_name] = set(topk.indices.tolist())

    for g1 in GROUP_NAMES:
        overlap_matrix[g1] = {}
        for g2 in GROUP_NAMES:
            overlap_matrix[g1][g2] = round(
                compute_overlap(top50_sets[g1], top50_sets[g2]), 4
            )

    # 4. Cross-reference with known neurons
    known_xref = cross_reference_known_neurons(
        {g: mean_attr_by_group[g] for g in GROUP_NAMES},
        KNOWN_NEURONS,
    )

    # 5. Top neurons per layer per group (heatmap detail)
    layer_top_neurons: dict[str, list[dict]] = {}
    for group_name in GROUP_NAMES:
        mean_attr = mean_attr_by_group[group_name]    # (n_layers, hidden_dim)
        layer_top_neurons[group_name] = []
        for li in range(n_layers):
            row = mean_attr[li]
            topk = torch.topk(row.abs(), k=5)
            layer_top_neurons[group_name].append({
                "layer": li,
                "mean_abs_attr": round(float(row.abs().mean()), 6),
                "top5": [
                    {"dim": int(d), "attr": round(float(row[d]), 6)}
                    for d in topk.indices
                ],
            })

    # ── Assemble final JSON ───────────────────────────────────────────────
    analysis = {
        "model": args.model,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "group_names": GROUP_NAMES,
        "sample_counts": sample_counts,
        "method": "gradient_times_input",
        "notes": (
            "Attribution = hidden_state[layer] * grad(logit/hidden_state[layer]) "
            "at the last token position. "
            "Float32 cast applied to hidden states before gradient computation. "
            "Values aggregated as mean over all prompts in the dataset."
        ),
        "top_neurons_per_group": top_neurons_per_group,
        "layer_attribution_heatmap": layer_heatmap,
        "neuron_overlap_between_groups": overlap_matrix,
        "known_neuron_cross_reference": known_xref,
        "layer_top_neurons_per_group": layer_top_neurons,
    }

    analysis_path = os.path.join(args.output, "gradient_attribution_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved gradient_attribution_analysis.json -> {analysis_path}")

    # ── Human-readable summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GRADIENT ATTRIBUTION SUMMARY")
    print("=" * 70)

    for group_name in GROUP_NAMES:
        entry = top_neurons_per_group[group_name]
        n_samples = entry["n_samples"]
        if n_samples == 0:
            print(f"\n  [{group_name.upper()}] No data collected.")
            continue

        print(f"\n  [{group_name.upper()}]  ({n_samples} samples)")
        print(f"  {'Rank':4s}  {'Dim':5s}  {'Mean Attr':>12s}  {'Direction':10s}")
        print(f"  {'-'*4}  {'-'*5}  {'-'*12}  {'-'*10}")
        for rank, n in enumerate(entry["top_neurons_by_mean_attr"][:20], 1):
            flag = ""
            if n["dim"] in KNOWN_NEURONS:
                flag = f"  <-- {KNOWN_NEURONS[n['dim']]}"
            print(f"  {rank:4d}  {n['dim']:5d}  {n['attribution']:+12.4f}  "
                  f"{n['direction']:10s}{flag}")

    print("\n  Layer attribution heatmap (mean abs per layer, all groups):")
    print(f"  {'Layer':5s}  " + "  ".join(f"{g[:6]:>8s}" for g in GROUP_NAMES))
    print("  " + "-" * (7 + 10 * len(GROUP_NAMES)))
    for li in range(n_layers):
        row_str = f"  L{li:02d}   "
        for g in GROUP_NAMES:
            val = layer_heatmap[g][li]
            row_str += f"  {val:8.4f}"
        print(row_str)

    print("\n  Neuron overlap between groups (Jaccard, top-50 dims):")
    print(f"  {'':12s}" + "".join(f"  {g[:8]:>10s}" for g in GROUP_NAMES))
    for g1 in GROUP_NAMES:
        row = f"  {g1:12s}"
        for g2 in GROUP_NAMES:
            row += f"  {overlap_matrix[g1][g2]:10.4f}"
        print(row)

    print("\n  Known neuron attribution signals:")
    for dim_str, xref in known_xref.items():
        dim = int(dim_str)
        label = xref["label"]
        group_summaries = []
        for g, data in xref["by_group"].items():
            group_summaries.append(f"{g}={data['mean_abs_attr']:.4f}")
        if group_summaries:
            print(f"    dim {dim:4d} ({label})")
            print(f"             mean|attr|: {', '.join(group_summaries)}")

    print(f"\n  GPU peak usage: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"\nDone! Results saved to: {args.output}/")
    print(f"  - gradient_attributions.pt          (tensor: 5 × {n_layers} × {hidden_dim})")
    print(f"  - gradient_attribution_analysis.json (top-{args.top_k} neurons, overlaps, xrefs)")


if __name__ == "__main__":
    main()
