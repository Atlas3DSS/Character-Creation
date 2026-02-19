#!/usr/bin/env python3
"""
Qwen3-VL-8B Deep Analysis — Four-Phase Neural Dissection.

Loads the model ONCE and runs four analyses back-to-back:
  Phase 1: Attention Head Profiling        (~10 min)
  Phase 2: MLP vs Attention Decomposition  (~15 min)
  Phase 3: Logit Lens                      (~10 min)
  Phase 4: Neuron Activation Trajectories  (~5 min)

Usage:
    CUDA_VISIBLE_DEVICES=1 python qwen_deep_analysis_3090.py

Output: /home/orwel/dev_genius/qwen_deep_analysis/
    attention_head_profiles.json
    mlp_vs_attention_decomposition.json
    logit_lens_results.json
    neuron_trajectories.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ─── Configuration ──────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR = "/home/orwel/dev_genius/qwen_deep_analysis"
_CONNECTOME_CANDIDATES = [
    "/home/orwel/dev_genius/qwen_connectome/analysis/connectome_zscores.pt",  # dev server
    "/home/orwel/dev_genius/experiments/Character Creation/qwen_connectome_analysis/connectome_zscores.pt",  # WSL
]
CONNECTOME_PATH = next((p for p in _CONNECTOME_CANDIDATES if os.path.exists(p)), _CONNECTOME_CANDIDATES[0])

N_LAYERS = 36
HIDDEN_DIM = 4096
N_HEADS_Q = 32          # Query heads
N_HEADS_KV = 8          # Key/Value heads (GQA)
HEAD_DIM = 128

# Known neurons from prior probes
KNOWN_NEURONS = {
    994:  "Identity (Qwen name neuron, z=-13.96 at L9)",
    270:  "Identity secondary (avg |z|=7.68)",
    1924: "Sarcasm (only cross-layer sarcasm neuron, avg |z|=2.14)",
    368:  "Name relay (97% layers)",
    98:   "Name relay (89% layers)",
    3522: "Name relay (86% layers)",
    208:  "Name relay (78% layers)",
    3140: "Name relay (69% layers)",
}
TRACKED_DIMS = list(KNOWN_NEURONS.keys())  # [994, 270, 1924, 368, 98, 3522, 208, 3140]

# Sarcastic vs neutral system prompts for attention head profiling
SARCASTIC_SYSTEM = (
    "Respond with heavy sarcasm, wit, and mockery. Be condescending, "
    "use irony, and make biting observations."
)
NEUTRAL_SYSTEM = (
    "Respond sincerely, earnestly, and with genuine helpfulness. "
    "Be straightforward and honest."
)

# Identity-present vs no-system-prompt for identity attention profiling
IDENTITY_SYSTEM = (
    "You are Qwen, a large language model created by Alibaba Cloud. "
    "You are helpful, harmless, and honest."
)
# Identity-absent = no system prompt (None)

# 20 category names matching connectome ordering
CATEGORY_NAMES = [
    "Identity", "Emotion: Joy", "Emotion: Sadness", "Emotion: Anger",
    "Emotion: Fear", "Tone: Formal", "Tone: Sarcastic", "Tone: Polite",
    "Domain: Math", "Domain: Science", "Domain: Code", "Domain: History",
    "Reasoning: Analytical", "Reasoning: Certainty", "Safety: Refusal",
    "Role: Teacher", "Role: Authority", "Verbosity: Brief",
    "Language: EN vs CN", "Sentiment: Positive",
]

# Sarcasm-related token strings to track in logit lens
SARCASM_TOKEN_HINTS = [
    "Oh", "oh", "Sure", "Wow", "wow", "Yes", "Obviously", "Obviously",
    "Clearly", "clearly", "Brilliant", "brilliant", "genius",
]
IDENTITY_TOKEN_HINTS = ["Qwen", "I", "am", "Assistant", "assistant", "AI"]


# ─── Prompt Definitions ─────────────────────────────────────────────────────

# 15 sarcasm contrastive pairs (sarcastic system vs neutral system)
SARCASM_PAIRS = [
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
    "Explain how gravity works.",
    "What is your favorite color and why?",
    "Tell me about the history of Rome.",
    "What makes a good leader?",
    "Tell me about artificial intelligence.",
]

# 15 identity contrastive pairs (identity system vs no system)
IDENTITY_PAIRS = [
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
    "Explain how gravity works.",
    "What is 15 times 23?",
    "How does a computer work?",
    "What is democracy?",
    "Tell me something interesting about the ocean.",
]

# 20 representative prompts, one per connectome category
CATEGORY_PROMPTS = [
    ("Identity",              "What is your name?",                       IDENTITY_SYSTEM),
    ("Emotion: Joy",          "I just got promoted!",                      "Respond with happiness, joy, and excitement."),
    ("Emotion: Sadness",      "My pet just passed away.",                  "Respond with sadness and somber reflection."),
    ("Emotion: Anger",        "Someone stole my parking spot.",            "Respond with frustration and indignation."),
    ("Emotion: Fear",         "There's a strange noise downstairs.",       "Respond with anxiety and caution."),
    ("Tone: Formal",          "Summarize the French Revolution.",          "Respond in a highly formal, academic manner."),
    ("Tone: Sarcastic",       "I think the earth might be flat.",          SARCASTIC_SYSTEM),
    ("Tone: Polite",          "Tell me the answer to 2+2.",               "Respond with extreme politeness and deference."),
    ("Domain: Math",          "What is the derivative of x^3?",           "You are a mathematics expert."),
    ("Domain: Science",       "How does photosynthesis work?",             "You are a scientist."),
    ("Domain: Code",          "Sort a list of numbers.",                   "You are a software engineer."),
    ("Domain: History",       "Tell me about World War II.",               "You are a historian."),
    ("Reasoning: Analytical", "Analyze the pros and cons of remote work.", "Think step by step. Be methodical and analytical."),
    ("Reasoning: Certainty",  "Will AI replace all jobs?",                "Be definitive and authoritative."),
    ("Safety: Refusal",       "How does encryption work?",                "Be extremely cautious about potential misuse."),
    ("Role: Teacher",         "Teach me about gravity.",                  "You are an experienced teacher."),
    ("Role: Authority",       "What should we do about this crisis?",     "You are a commanding authority figure."),
    ("Verbosity: Brief",      "What is DNA?",                             "Be extremely brief. One sentence maximum."),
    ("Language: EN vs CN",    "What is the capital of China?",            "Respond only in English. Never use Chinese characters."),
    ("Sentiment: Positive",   "I'm starting a new business.",             "Be encouraging, optimistic, and positive."),
]

# 10 prompts for neuron trajectory tracking
TRAJECTORY_PROMPTS = [
    ("sarcasm",   SARCASTIC_SYSTEM, "I think the earth might be flat."),
    ("identity",  IDENTITY_SYSTEM,  "What is your name?"),
    ("math",      "You are a mathematics expert.", "What is the derivative of x^3?"),
    ("neutral_1", None,             "Explain how gravity works."),
    ("neutral_2", None,             "What is your name?"),
    ("sarcasm_2", SARCASTIC_SYSTEM, "Everything I do is perfect."),
    ("history",   "You are a historian.", "Tell me about World War II."),
    ("code",      "You are a software engineer.", "Sort a list of numbers."),
    ("brief",     "Be extremely brief.", "What is DNA?"),
    ("no_sys",    None,             "I think the earth might be flat."),
]


# ─── Cache Check ─────────────────────────────────────────────────────────────

def check_model_cached(model_name: str) -> bool:
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


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_processor(model_name: str):
    cached = check_model_cached(model_name)
    print(f"  Cache status: {'CACHED (no download)' if cached else 'NOT CACHED — will download'}")
    if not cached:
        print("  WARNING: Model not in local cache. This will be a large download.")

    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    print(f"  Loading model (bfloat16, device_map=auto)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    model.eval()

    # Resolve layer path
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
        final_norm = model.model.language_model.norm
    else:
        raise RuntimeError("Cannot find model.model.language_model.layers — unexpected architecture")

    n_actual_layers = len(layers)
    if hasattr(model.config, "text_config"):
        actual_hidden = model.config.text_config.hidden_size
    else:
        actual_hidden = model.config.hidden_size

    print(f"  Model loaded. {n_actual_layers} layers, hidden_dim={actual_hidden}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, processor, layers, final_norm


# ─── Tokenisation Helper ─────────────────────────────────────────────────────

def make_inputs(
    processor: AutoProcessor,
    sys_prompt: str | None,
    user_prompt: str,
    device: torch.device,
) -> dict:
    """Build tokenised inputs from system + user prompt."""
    msgs = []
    if sys_prompt is not None:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user_prompt})

    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    return inputs


def get_system_token_mask(input_ids: torch.Tensor, processor: AutoProcessor) -> torch.Tensor:
    """Return a boolean mask marking system prompt token positions.

    Strategy: the system prompt tokens are those that appear before the first
    user-turn start token.  We detect the position of the second <|im_start|>
    token (which begins the user turn) and mark everything before it as system.
    Falls back to all-False if structure is undetectable.
    """
    seq_len = input_ids.shape[-1]
    mask = torch.zeros(seq_len, dtype=torch.bool)

    # Find <|im_start|> token id
    try:
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    except Exception:
        return mask

    token_list = input_ids[0].tolist()
    positions = [i for i, t in enumerate(token_list) if t == im_start_id]

    # positions[0] = system turn start, positions[1] = user turn start
    if len(positions) >= 2:
        mask[:positions[1]] = True
    elif len(positions) == 1:
        mask[:positions[0]] = True

    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Attention Head Profiling
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def phase1_attention_head_profiling(
    model,
    processor: AutoProcessor,
    layers,
    device: torch.device,
    output_dir: str,
) -> dict:
    """
    For 30 contrastive prompt pairs (15 sarcasm + 15 identity), run forward passes
    with output_attentions=True. Compute per-head:
      - Attention entropy over sequence positions
      - System-prompt attention fraction

    Returns and saves attention_head_profiles.json.
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 1: Attention Head Profiling")
    print("=" * 60)

    n_layers = len(layers)

    # ── Build prompt list ──────────────────────────────────────────────────
    all_pairs: list[dict] = []

    for prompt in SARCASM_PAIRS:
        all_pairs.append({
            "category": "sarcasm",
            "condition": "sarcastic",
            "sys_prompt": SARCASTIC_SYSTEM,
            "user_prompt": prompt,
        })
        all_pairs.append({
            "category": "sarcasm",
            "condition": "neutral",
            "sys_prompt": NEUTRAL_SYSTEM,
            "user_prompt": prompt,
        })

    for prompt in IDENTITY_PAIRS:
        all_pairs.append({
            "category": "identity",
            "condition": "with_identity",
            "sys_prompt": IDENTITY_SYSTEM,
            "user_prompt": prompt,
        })
        all_pairs.append({
            "category": "identity",
            "condition": "no_system",
            "sys_prompt": None,
            "user_prompt": prompt,
        })

    print(f"  {len(all_pairs)} forward passes across 30 contrastive pairs")

    # Accumulators: [layer][head] -> list of scores per condition
    # head_entropy[category][condition][layer][head] = list of entropy values
    head_entropy: dict[str, dict[str, list]] = {
        "sarcasm":  {"sarcastic": [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)],
                     "neutral":   [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)]},
        "identity": {"with_identity": [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)],
                     "no_system":     [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)]},
    }
    # sys_attn_frac[category][condition][layer][head] = list of fracs
    sys_attn_frac: dict[str, dict[str, list]] = {
        "sarcasm":  {"sarcastic": [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)],
                     "neutral":   [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)]},
        "identity": {"with_identity": [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)],
                     "no_system":     [[[] for _ in range(N_HEADS_Q)] for _ in range(n_layers)]},
    }

    for entry in tqdm(all_pairs, desc="  Phase 1 forward passes"):
        inputs = make_inputs(processor, entry["sys_prompt"], entry["user_prompt"], device)
        sys_mask = get_system_token_mask(inputs["input_ids"], processor)

        # Forward pass with attentions
        # output_attentions returns tuple of (batch, n_heads, seq, seq) per layer
        outputs = model(**inputs, output_attentions=True)

        attn_weights = outputs.attentions  # tuple of len=n_layers
        # Note: GQA — attention weights shape may be (batch, n_kv_heads, seq, seq)
        # or (batch, n_q_heads, seq, seq) depending on implementation.
        # We iterate over whatever is returned.

        for layer_idx, layer_attn in enumerate(attn_weights):
            # layer_attn: (1, n_heads, seq_len, seq_len)
            if layer_attn is None:
                continue
            layer_attn = layer_attn[0].float()  # (n_heads, seq_len, seq_len)
            n_actual_heads = layer_attn.shape[0]
            seq_len = layer_attn.shape[-1]

            sys_mask_float = sys_mask[:seq_len].float().to(layer_attn.device)
            sys_token_count = sys_mask_float.sum().item()

            # Per head: attention entropy + system attention fraction
            # Use LAST query position (most informative for what model will generate next)
            last_q_attn = layer_attn[:, -1, :]  # (n_heads, seq_len) — attn from last token
            last_q_attn = last_q_attn + 1e-10   # numerical stability

            for head_idx in range(min(n_actual_heads, N_HEADS_Q)):
                attn = last_q_attn[head_idx]   # (seq_len,)
                attn = attn / attn.sum()        # renormalize

                # Shannon entropy
                entropy = -(attn * attn.log()).sum().item()

                # System prompt attention fraction
                if sys_token_count > 0:
                    frac = (attn * sys_mask_float.to(attn.device)).sum().item()
                else:
                    frac = 0.0

                cat = entry["category"]
                cond = entry["condition"]
                head_entropy[cat][cond][layer_idx][head_idx].append(entropy)
                sys_attn_frac[cat][cond][layer_idx][head_idx].append(frac)

        # Free attention weights from GPU
        del outputs

    # ── Aggregate: mean per layer/head ────────────────────────────────────
    print("  Aggregating head importance scores...")

    results: dict[str, Any] = {}

    for cat in ["sarcasm", "identity"]:
        if cat == "sarcasm":
            conditions = ("sarcastic", "neutral")
        else:
            conditions = ("with_identity", "no_system")

        cond_a, cond_b = conditions

        # Per-head importance = |mean_entropy_A - mean_entropy_B| + |mean_frac_A - mean_frac_B|
        importance: list[list[float]] = []  # [layer][head]
        entropy_diff: list[list[float]] = []
        frac_diff: list[list[float]] = []
        mean_entropy_a: list[list[float]] = []
        mean_entropy_b: list[list[float]] = []
        mean_frac_a: list[list[float]] = []
        mean_frac_b: list[list[float]] = []

        for layer_idx in range(n_layers):
            layer_importance = []
            layer_ent_diff = []
            layer_frac_diff = []
            layer_ent_a = []
            layer_ent_b = []
            layer_fr_a = []
            layer_fr_b = []

            for head_idx in range(N_HEADS_Q):
                ent_a_vals = head_entropy[cat][cond_a][layer_idx][head_idx]
                ent_b_vals = head_entropy[cat][cond_b][layer_idx][head_idx]
                fra_a_vals = sys_attn_frac[cat][cond_a][layer_idx][head_idx]
                fra_b_vals = sys_attn_frac[cat][cond_b][layer_idx][head_idx]

                ent_a = sum(ent_a_vals) / len(ent_a_vals) if ent_a_vals else 0.0
                ent_b = sum(ent_b_vals) / len(ent_b_vals) if ent_b_vals else 0.0
                fra_a = sum(fra_a_vals) / len(fra_a_vals) if fra_a_vals else 0.0
                fra_b = sum(fra_b_vals) / len(fra_b_vals) if fra_b_vals else 0.0

                imp = abs(ent_a - ent_b) + abs(fra_a - fra_b)
                layer_importance.append(imp)
                layer_ent_diff.append(ent_a - ent_b)
                layer_frac_diff.append(fra_a - fra_b)
                layer_ent_a.append(ent_a)
                layer_ent_b.append(ent_b)
                layer_fr_a.append(fra_a)
                layer_fr_b.append(fra_b)

            importance.append(layer_importance)
            entropy_diff.append(layer_ent_diff)
            frac_diff.append(layer_frac_diff)
            mean_entropy_a.append(layer_ent_a)
            mean_entropy_b.append(layer_ent_b)
            mean_frac_a.append(layer_fr_a)
            mean_frac_b.append(layer_fr_b)

        # Find top-10 heads by importance (layer, head, score)
        flat_imp: list[tuple[float, int, int]] = []
        for li in range(n_layers):
            for hi in range(N_HEADS_Q):
                flat_imp.append((importance[li][hi], li, hi))
        flat_imp.sort(reverse=True)
        top_heads = [
            {"layer": li, "head": hi, "importance": sc,
             "entropy_diff": entropy_diff[li][hi],
             "sys_frac_diff": frac_diff[li][hi]}
            for sc, li, hi in flat_imp[:20]
        ]

        results[cat] = {
            "condition_a": cond_a,
            "condition_b": cond_b,
            "importance": importance,          # [n_layers][N_HEADS_Q]
            "entropy_diff_a_minus_b": entropy_diff,
            "sys_frac_diff_a_minus_b": frac_diff,
            f"mean_entropy_{cond_a}": mean_entropy_a,
            f"mean_entropy_{cond_b}": mean_entropy_b,
            f"mean_sys_frac_{cond_a}": mean_frac_a,
            f"mean_sys_frac_{cond_b}": mean_frac_b,
            "top_20_heads_by_importance": top_heads,
        }

        # Print summary
        print(f"\n  [{cat}] Top-10 most discriminative attention heads:")
        for rank, entry in enumerate(top_heads[:10], 1):
            print(f"    #{rank:2d}  L{entry['layer']:2d}H{entry['head']:2d}  "
                  f"imp={entry['importance']:.4f}  "
                  f"ent_diff={entry['entropy_diff']:+.4f}  "
                  f"frac_diff={entry['sys_frac_diff']:+.4f}")

    out_path = os.path.join(output_dir, "attention_head_profiles.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete in {elapsed:.1f}s → {out_path}")
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MLP vs Attention Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def phase2_mlp_vs_attention(
    model,
    processor: AutoProcessor,
    layers,
    device: torch.device,
    output_dir: str,
    connectome_path: str,
) -> dict:
    """
    For the same 30 contrastive prompt pairs, hook into pre_attn / post_attn / post_mlp
    residual streams.  Compute per-layer attention vs MLP contribution projected onto
    each of the 20 connectome category z-score directions.

    Returns and saves mlp_vs_attention_decomposition.json.
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 2: MLP vs Attention Decomposition")
    print("=" * 60)

    n_layers = len(layers)

    # ── Load connectome ────────────────────────────────────────────────────
    print(f"  Loading connectome from: {connectome_path}")
    if not os.path.exists(connectome_path):
        print(f"  WARNING: Connectome not found at {connectome_path}")
        print("  Skipping Phase 2.")
        return {}

    connectome = torch.load(connectome_path, map_location="cpu", weights_only=True)
    # Shape: (20, 36, 4096)
    print(f"  Connectome shape: {connectome.shape}")
    n_cats = connectome.shape[0]
    actual_cat_names = CATEGORY_NAMES[:n_cats]

    # Normalise category directions (one per layer per category)
    # cat_dirs[cat_idx][layer_idx] = unit vector (4096,)
    cat_dirs: list[list[torch.Tensor]] = []
    for cat_idx in range(n_cats):
        layer_dirs = []
        for li in range(min(n_layers, connectome.shape[1])):
            z = connectome[cat_idx, li].float()
            norm = z.norm()
            layer_dirs.append(z / (norm + 1e-8))
        cat_dirs.append(layer_dirs)

    # ── Build all prompts (same 30 contrastive pairs as Phase 1) ──────────
    all_prompts: list[dict] = []
    for prompt in SARCASM_PAIRS:
        all_prompts.append({"category": "sarcasm", "condition": "sarcastic",
                             "sys": SARCASTIC_SYSTEM, "user": prompt})
        all_prompts.append({"category": "sarcasm", "condition": "neutral",
                             "sys": NEUTRAL_SYSTEM, "user": prompt})
    for prompt in IDENTITY_PAIRS:
        all_prompts.append({"category": "identity", "condition": "with_identity",
                             "sys": IDENTITY_SYSTEM, "user": prompt})
        all_prompts.append({"category": "identity", "condition": "no_system",
                             "sys": None, "user": prompt})

    print(f"  {len(all_prompts)} forward passes for decomposition")

    # Accumulators: [cat_idx][layer_idx] -> {"attn_proj": [], "mlp_proj": [], "total_proj": []}
    # Using nested lists instead of dicts to avoid JSON issues
    # We store sums to average later
    n_layers_conn = min(n_layers, connectome.shape[1])

    # [cat_idx][layer_idx] -> accumulated projections
    attn_proj_sum  = [[0.0] * n_layers_conn for _ in range(n_cats)]
    mlp_proj_sum   = [[0.0] * n_layers_conn for _ in range(n_cats)]
    total_proj_sum = [[0.0] * n_layers_conn for _ in range(n_cats)]
    count_sum      = [[0] * n_layers_conn for _ in range(n_cats)]

    # Hook storage
    pre_attn_states:  dict[int, torch.Tensor] = {}
    post_attn_states: dict[int, torch.Tensor] = {}
    post_mlp_states:  dict[int, torch.Tensor] = {}

    hooks: list = []

    def make_pre_attn_hook(layer_idx: int):
        """Hook on layer input (= pre-attention residual). Pre-hooks get (module, args)."""
        def hook_fn(module, args):
            # args is a tuple; first element is the hidden state
            inp = args[0] if isinstance(args, tuple) else args
            pre_attn_states[layer_idx] = inp[:, -1, :].detach().cpu().float()
        return hook_fn

    def make_post_attn_hook(layer_idx: int):
        """Hook on self_attn output to capture post-attention residual.

        We need the residual AFTER attention is added back. We capture
        the attention module output and add it to pre_attn ourselves.
        Since layer applies: h = h + attn(h), we hook the attn output.
        """
        def hook_fn(module, input, output):
            # self_attn returns (hidden_states, ...) or just hidden_states
            attn_out = output[0] if isinstance(output, tuple) else output
            post_attn_states[layer_idx] = attn_out[:, -1, :].detach().cpu().float()
        return hook_fn

    def make_post_mlp_hook(layer_idx: int):
        """Hook on full layer output to capture post-MLP residual."""
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            post_mlp_states[layer_idx] = out[:, -1, :].detach().cpu().float()
        return hook_fn

    # Register hooks
    for li, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(make_pre_attn_hook(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_post_attn_hook(li)))
        hooks.append(layer.register_forward_hook(make_post_mlp_hook(li)))

    for entry in tqdm(all_prompts, desc="  Phase 2 forward passes"):
        pre_attn_states.clear()
        post_attn_states.clear()
        post_mlp_states.clear()

        inputs = make_inputs(processor, entry["sys"], entry["user"], device)
        _ = model(**inputs)

        for li in range(n_layers_conn):
            if li not in pre_attn_states or li not in post_attn_states or li not in post_mlp_states:
                continue

            pre  = pre_attn_states[li]   # (1, 4096) — input to layer (pre-attn residual)
            attn_raw = post_attn_states[li]  # (1, 4096) — raw attn output (before residual add)
            post_mlp_full = post_mlp_states[li]  # (1, 4096) — after full layer (pre+attn_delta+mlp_delta)

            # Reconstruct contributions:
            # post_attn = pre + attn_raw  (residual addition after attention)
            post_attn = pre + attn_raw    # (1, 4096)

            attn_contrib = post_attn - pre        # = attn_raw
            mlp_contrib  = post_mlp_full - post_attn

            for cat_idx in range(n_cats):
                d = cat_dirs[cat_idx][li]  # unit vector (4096,)
                attn_p  = (attn_contrib[0] * d).sum().item()
                mlp_p   = (mlp_contrib[0]  * d).sum().item()
                total_p = (post_mlp_full[0] * d).sum().item() - (pre[0] * d).sum().item()

                attn_proj_sum[cat_idx][li]  += abs(attn_p)
                mlp_proj_sum[cat_idx][li]   += abs(mlp_p)
                total_proj_sum[cat_idx][li] += abs(total_p)
                count_sum[cat_idx][li]      += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Aggregate ──────────────────────────────────────────────────────────
    print("  Computing MLP/attention fractions per category per layer...")

    decomp_results: dict[str, Any] = {}
    for cat_idx, cat_name in enumerate(actual_cat_names):
        per_layer = []
        for li in range(n_layers_conn):
            n = count_sum[cat_idx][li]
            if n == 0:
                per_layer.append({"layer": li, "attn_mean_proj": 0.0,
                                   "mlp_mean_proj": 0.0, "total_mean_proj": 0.0,
                                   "attn_frac": 0.5, "mlp_frac": 0.5})
                continue

            attn_m  = attn_proj_sum[cat_idx][li] / n
            mlp_m   = mlp_proj_sum[cat_idx][li] / n
            total_m = total_proj_sum[cat_idx][li] / n
            denom   = attn_m + mlp_m + 1e-10

            per_layer.append({
                "layer": li,
                "attn_mean_proj":  attn_m,
                "mlp_mean_proj":   mlp_m,
                "total_mean_proj": total_m,
                "attn_frac":       attn_m / denom,
                "mlp_frac":        mlp_m  / denom,
            })

        # Summary stats
        attn_fracs = [r["attn_frac"] for r in per_layer]
        mlp_fracs  = [r["mlp_frac"]  for r in per_layer]
        peak_attn_layer = int(max(range(len(per_layer)),
                                  key=lambda i: per_layer[i]["attn_mean_proj"]))
        peak_mlp_layer  = int(max(range(len(per_layer)),
                                  key=lambda i: per_layer[i]["mlp_mean_proj"]))

        overall_attn_frac = sum(attn_fracs) / len(attn_fracs) if attn_fracs else 0.5
        overall_mlp_frac  = sum(mlp_fracs)  / len(mlp_fracs)  if mlp_fracs  else 0.5

        decomp_results[cat_name] = {
            "per_layer": per_layer,
            "overall_attn_frac": overall_attn_frac,
            "overall_mlp_frac": overall_mlp_frac,
            "peak_attn_layer": peak_attn_layer,
            "peak_mlp_layer": peak_mlp_layer,
        }

        print(f"  {cat_name:30s}: attn={overall_attn_frac:.2%}  mlp={overall_mlp_frac:.2%}  "
              f"peak_attn=L{peak_attn_layer}  peak_mlp=L{peak_mlp_layer}")

    out_path = os.path.join(output_dir, "mlp_vs_attention_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(decomp_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete in {elapsed:.1f}s → {out_path}")
    torch.cuda.empty_cache()
    return decomp_results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Logit Lens
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def phase3_logit_lens(
    model,
    processor: AutoProcessor,
    layers,
    final_norm,
    device: torch.device,
    output_dir: str,
) -> dict:
    """
    For 20 prompts (one per connectome category), at each layer take the residual
    stream hidden state at the last token, apply final LayerNorm + lm_head to get
    logits, record top-5 tokens + probabilities.

    Track specific token logits: Qwen, I, am, and common sarcasm markers.

    Returns and saves logit_lens_results.json.
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print("PHASE 3: Logit Lens")
    print("=" * 60)

    n_layers = len(layers)
    lm_head = model.lm_head

    # Pre-compute token IDs to track
    tokenizer = processor.tokenizer

    def token_id(s: str) -> int | None:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    tracked_tokens: dict[str, int] = {}
    for tok_str in IDENTITY_TOKEN_HINTS + SARCASM_TOKEN_HINTS:
        tid = token_id(tok_str)
        if tid is not None and tid not in tracked_tokens.values():
            tracked_tokens[tok_str] = tid

    # Remove duplicates (e.g. "Oh"/"oh" may map to same id)
    print(f"  Tracking {len(tracked_tokens)} tokens: {list(tracked_tokens.keys())}")

    # ── Hook to capture residual stream per layer ──────────────────────────
    residuals: dict[int, torch.Tensor] = {}

    def make_residual_hook(layer_idx: int):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            residuals[layer_idx] = out[:, -1, :].detach().float()  # keep on GPU for lm_head
        return hook_fn

    hooks = []
    for li, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_residual_hook(li)))

    results: dict[str, Any] = {}

    for cat_name, user_prompt, sys_prompt in tqdm(CATEGORY_PROMPTS, desc="  Phase 3 prompts"):
        residuals.clear()
        inputs = make_inputs(processor, sys_prompt, user_prompt, device)
        _ = model(**inputs)

        prompt_results: dict[str, Any] = {
            "category": cat_name,
            "user_prompt": user_prompt,
            "system_prompt": sys_prompt,
            "layers": {},
        }

        for li in range(n_layers):
            if li not in residuals:
                continue

            h = residuals[li]  # (1, hidden_dim) — on GPU

            # Apply final layer norm then lm_head (cast to model dtype)
            h_normed = final_norm(h.to(final_norm.weight.dtype))
            logits = lm_head(h_normed)[0].float()  # (vocab_size,)
            probs = F.softmax(logits, dim=-1)

            # Top-5 tokens
            top5_probs, top5_ids = probs.topk(5)
            top5 = [
                {
                    "token": tokenizer.decode([int(tid)]),
                    "token_id": int(tid),
                    "prob": float(p),
                }
                for tid, p in zip(top5_ids.tolist(), top5_probs.tolist())
            ]

            # Tracked token logits
            tracked_logits: dict[str, dict] = {}
            for tok_str, tid in tracked_tokens.items():
                if tid < logits.shape[0]:
                    tracked_logits[tok_str] = {
                        "token_id": tid,
                        "logit": float(logits[tid]),
                        "prob": float(probs[tid]),
                    }

            prompt_results["layers"][str(li)] = {
                "top5": top5,
                "tracked_tokens": tracked_logits,
            }

        results[cat_name] = prompt_results

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Print interesting findings ─────────────────────────────────────────
    print("\n  Key logit lens observations:")
    for cat_name, data in results.items():
        # Find first layer where top-1 token is a "real word" (not punctuation/BOS)
        first_real_layer = None
        first_real_token = None
        for li in range(n_layers):
            layer_str = str(li)
            if layer_str in data["layers"]:
                top1 = data["layers"][layer_str]["top5"][0]["token"] if data["layers"][layer_str]["top5"] else ""
                if any(c.isalpha() for c in top1):
                    first_real_layer = li
                    first_real_token = top1.strip()
                    break

        # Qwen logit trajectory: how much does "Qwen" probability change?
        qwen_probs = []
        for li in range(n_layers):
            lstr = str(li)
            if lstr in data["layers"]:
                trd = data["layers"][lstr].get("tracked_tokens", {})
                if "Qwen" in trd:
                    qwen_probs.append(trd["Qwen"]["prob"])

        max_qwen_prob = max(qwen_probs) if qwen_probs else 0.0
        final_qwen_prob = qwen_probs[-1] if qwen_probs else 0.0
        final_top1 = data["layers"].get(str(n_layers - 1), {}).get("top5", [{}])[0].get("token", "?")

        print(f"  [{cat_name:30s}] "
              f"final_top1={final_top1!r:12s}  "
              f"max_Qwen={max_qwen_prob:.4f}  "
              f"first_real=L{first_real_layer}({first_real_token!r})")

    out_path = os.path.join(output_dir, "logit_lens_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Phase 3 complete in {elapsed:.1f}s → {out_path}")
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Neuron Activation Trajectories During Generation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def phase4_neuron_trajectories(
    model,
    processor: AutoProcessor,
    layers,
    device: torch.device,
    output_dir: str,
    n_tokens: int = 50,
) -> dict:
    """
    For 10 prompts, generate n_tokens tokens.  At EACH generation step, hook all
    36 layers to capture the hidden state at the last token position.  Record
    activations of the known neurons (dims 994, 270, 1924, 368, 98, 3522, 208, 3140).

    Returns and saves neuron_trajectories.json.
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print(f"PHASE 4: Neuron Activation Trajectories (generate {n_tokens} tokens)")
    print("=" * 60)

    n_layers = len(layers)

    # Hook storage for current step
    step_hidden: dict[int, torch.Tensor] = {}

    def make_gen_hook(layer_idx: int):
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            # out: (batch, seq_len, hidden_dim)
            # During generation, seq_len may be 1 (with KV cache) or full sequence
            step_hidden[layer_idx] = out[0, -1, :].detach().cpu().float()
        return hook_fn

    # Register persistent hooks (stay for all generation steps)
    gen_hooks = []
    for li, layer in enumerate(layers):
        gen_hooks.append(layer.register_forward_hook(make_gen_hook(li)))

    results: dict[str, Any] = {}

    for prompt_label, sys_prompt, user_prompt in tqdm(TRAJECTORY_PROMPTS, desc="  Phase 4 prompts"):
        inputs = make_inputs(processor, sys_prompt, user_prompt, device)
        input_len = inputs["input_ids"].shape[1]

        # Trajectory storage: per token step, per layer, the known-neuron activations
        # Structure: {layer_idx: {dim: [step0_val, step1_val, ...]}}
        trajectory: dict[str, dict[str, list]] = {
            str(li): {str(dim): [] for dim in TRACKED_DIMS}
            for li in range(n_layers)
        }

        # We generate token by token to capture each step
        current_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        past_key_values = None
        generated_tokens: list[str] = []

        for step_idx in range(n_tokens):
            step_hidden.clear()

            if past_key_values is not None:
                # Use cached KV, only pass the new token
                model_inputs = {
                    "input_ids": current_ids[:, -1:],
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    # Extend mask by 1
                    attention_mask = torch.cat(
                        [attention_mask,
                         torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                        dim=1,
                    )
                    model_inputs["attention_mask"] = attention_mask
            else:
                model_inputs = {
                    "input_ids": current_ids,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask

            # Forward pass (triggers hooks)
            try:
                out = model(**model_inputs)
            except Exception as e:
                print(f"  WARNING: generation step {step_idx} failed: {e}")
                break

            past_key_values = out.past_key_values

            # Greedy decode next token
            next_logits = out.logits[0, -1, :]
            next_token_id = next_logits.argmax().unsqueeze(0).unsqueeze(0)

            # Decode for record
            token_str = processor.tokenizer.decode([int(next_token_id.item())], skip_special_tokens=False)
            generated_tokens.append(token_str)

            # Record neuron activations from hooks
            for li in range(n_layers):
                if li not in step_hidden:
                    continue
                h = step_hidden[li]  # (4096,)
                for dim in TRACKED_DIMS:
                    if dim < h.shape[0]:
                        trajectory[str(li)][str(dim)].append(float(h[dim]))
                    else:
                        trajectory[str(li)][str(dim)].append(0.0)

            # Stop at EOS
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None and int(next_token_id.item()) == eos_id:
                break

            current_ids = next_token_id

        # Compute summary statistics per neuron per layer
        summary: dict[str, Any] = {}
        for li in range(n_layers):
            summary[str(li)] = {}
            for dim in TRACKED_DIMS:
                vals = trajectory[str(li)][str(dim)]
                if vals:
                    summary[str(li)][str(dim)] = {
                        "mean":   float(sum(vals) / len(vals)),
                        "std":    float(torch.tensor(vals).std().item()),
                        "min":    float(min(vals)),
                        "max":    float(max(vals)),
                        "steps":  len(vals),
                    }

        results[prompt_label] = {
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "generated_tokens": generated_tokens,
            "generated_text": "".join(generated_tokens),
            "trajectory": trajectory,
            "summary": summary,
        }

        # Print quick summary
        generated_preview = "".join(generated_tokens[:20]).replace("\n", " ")
        print(f"\n  [{prompt_label:15s}] generated ({len(generated_tokens)} tokens): {generated_preview!r}")

        # Print peak activations for identity neuron (dim 994)
        dim994_by_layer = []
        for li in range(n_layers):
            vals = trajectory[str(li)].get("994", [])
            if vals:
                dim994_by_layer.append((li, sum(vals) / len(vals)))
        if dim994_by_layer:
            peak_layer, peak_val = max(dim994_by_layer, key=lambda x: abs(x[1]))
            print(f"    dim994 (identity) peak: L{peak_layer} mean={peak_val:+.3f}")

    # Remove hooks
    for h in gen_hooks:
        h.remove()

    out_path = os.path.join(output_dir, "neuron_trajectories.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Phase 4 complete in {elapsed:.1f}s → {out_path}")
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_summary(
    phase1_results: dict,
    phase2_results: dict,
    phase3_results: dict,
    phase4_results: dict,
    total_elapsed: float,
) -> None:
    print("\n" + "=" * 60)
    print("DEEP ANALYSIS COMPLETE — SUMMARY OF KEY FINDINGS")
    print("=" * 60)

    # ── Phase 1 summary ───────────────────────────────────────────────────
    print("\n[Phase 1] Most discriminative attention heads:")
    for cat in ["sarcasm", "identity"]:
        if cat in phase1_results:
            top = phase1_results[cat].get("top_20_heads_by_importance", [])
            if top:
                t = top[0]
                print(f"  {cat:10s}: #1 = L{t['layer']:2d}H{t['head']:2d}  "
                      f"importance={t['importance']:.4f}  "
                      f"sys_frac_diff={t.get('sys_frac_diff', 0):+.4f}")

    # ── Phase 2 summary ───────────────────────────────────────────────────
    if phase2_results:
        print("\n[Phase 2] MLP vs Attention dominance per category (top and bottom by attn_frac):")
        sorted_by_attn = sorted(
            phase2_results.items(),
            key=lambda x: x[1].get("overall_attn_frac", 0.5),
            reverse=True,
        )
        for cat_name, data in sorted_by_attn[:5]:
            print(f"  Attn-dominant: {cat_name:30s}  attn={data['overall_attn_frac']:.2%}")
        for cat_name, data in sorted_by_attn[-3:]:
            print(f"  MLP-dominant:  {cat_name:30s}  mlp={data['overall_mlp_frac']:.2%}")
    else:
        print("\n[Phase 2] Skipped (connectome not found).")

    # ── Phase 3 summary ───────────────────────────────────────────────────
    if phase3_results:
        print("\n[Phase 3] Final-layer top-1 token per category:")
        n_layers = 36
        for cat_name, data in list(phase3_results.items())[:10]:
            last_layer = str(n_layers - 1)
            top5 = data["layers"].get(last_layer, {}).get("top5", [])
            top1 = top5[0]["token"].strip() if top5 else "?"
            qwen_prob = 0.0
            trd = data["layers"].get(last_layer, {}).get("tracked_tokens", {})
            if "Qwen" in trd:
                qwen_prob = trd["Qwen"]["prob"]
            print(f"  {cat_name:30s}: final_top1={top1!r:12s}  Qwen_prob={qwen_prob:.4f}")

    # ── Phase 4 summary ───────────────────────────────────────────────────
    if phase4_results:
        print("\n[Phase 4] Dim-994 (identity neuron) trajectory stats:")
        for prompt_label, data in phase4_results.items():
            # Peak across all layers in the trajectory
            best_layer_val = None
            best_layer = None
            for li in range(36):
                summ = data["summary"].get(str(li), {}).get("994")
                if summ and (best_layer_val is None or abs(summ["mean"]) > abs(best_layer_val)):
                    best_layer_val = summ["mean"]
                    best_layer = li
            if best_layer is not None:
                gen_preview = "".join(data["generated_tokens"][:10]).replace("\n", " ")
                print(f"  {prompt_label:15s}: peak dim994 = L{best_layer} mean={best_layer_val:+.3f}  "
                      f"gen={gen_preview!r}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files written:")
    print(f"    {OUTPUT_DIR}/attention_head_profiles.json")
    print(f"    {OUTPUT_DIR}/mlp_vs_attention_decomposition.json")
    print(f"    {OUTPUT_DIR}/logit_lens_results.json")
    print(f"    {OUTPUT_DIR}/neuron_trajectories.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    script_t0 = time.time()

    print("=" * 60)
    print("QWEN3-VL-8B DEEP ANALYSIS — RTX 3090")
    print("  4 phases: attn heads | mlp decomp | logit lens | neuron traj")
    print("=" * 60)

    # ── Sanity checks ──────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Check CUDA_VISIBLE_DEVICES=1.")
        sys.exit(1)

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 maps to cuda:0
    print(f"\n  Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")

    # ── Load model once ────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    load_t0 = time.time()
    model, processor, layers, final_norm = load_model_and_processor(MODEL_NAME)
    print(f"  Loaded in {time.time() - load_t0:.1f}s")

    n_layers = len(layers)
    print(f"  Confirmed {n_layers} layers")

    # ── Phase 1 ────────────────────────────────────────────────────────────
    p1 = phase1_attention_head_profiling(
        model, processor, layers, device, OUTPUT_DIR
    )

    # ── Phase 2 ────────────────────────────────────────────────────────────
    p2 = phase2_mlp_vs_attention(
        model, processor, layers, device, OUTPUT_DIR, CONNECTOME_PATH
    )

    # ── Phase 3 ────────────────────────────────────────────────────────────
    p3 = phase3_logit_lens(
        model, processor, layers, final_norm, device, OUTPUT_DIR
    )

    # ── Phase 4 ────────────────────────────────────────────────────────────
    p4 = phase4_neuron_trajectories(
        model, processor, layers, device, OUTPUT_DIR, n_tokens=50
    )

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.time() - script_t0
    print_final_summary(p1, p2, p3, p4, total_elapsed)


if __name__ == "__main__":
    main()
