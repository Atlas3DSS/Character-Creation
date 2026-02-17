#!/usr/bin/env python3
"""
ROME-Style Identity Editing for Qwen3-VL-8B-Instruct

Inspired by:
- ROME (Meng et al., 2022): Rank-one model editing
- MEMIT (Meng et al., 2023): Mass-editing for multiple facts
- PALETTE (arXiv:2502.11789): Self-referential personality editing with only 12 samples

Approach:
1. CAUSAL TRACING: Find which MLP layers store "I am Qwen" by corrupting
   subject embeddings and restoring individual layers to measure causal effect.
2. RANK-ONE EDIT: At the identified layer(s), modify MLP down_proj weights
   so that "Who are you?" → "I am Skippy" instead of "I am Qwen".
3. VALIDATE: Check identity change + reasoning preservation.

Key insight from our prior work:
- Attention heads are NOT causal for identity (zeroing top-10 = zero effect)
- MLP layers L17/L20/L21 have highest identity z-scores
- Identity is massively distributed but MLP down_proj is the ROME target

Usage:
    python rome_identity_edit.py
"""
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")

OUTPUT_DIR = Path("contrastive_data/rome_identity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the SDFT model as base (slightly more Skippy in behavior)
MODEL_PATH = "./skippy_sdft_r2/merged_scale_1.0"


def load_model():
    """Load model with HuggingFace (needed for hooks)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    print(f"Loading {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def get_layers(model):
    """Get the transformer layers."""
    return model.model.language_model.layers


def get_lm_head(model):
    """Get the lm_head."""
    return model.lm_head


# ─── Phase 1: Causal Tracing ─────────────────────────────────────────

def causal_trace(
    model,
    tokenizer,
    prompt: str = "Who are you?",
    subject_tokens: list[str] | None = None,
    n_noise: int = 10,
    noise_std: float = 0.1,
) -> dict:
    """
    Causal tracing to find which layers store identity facts.

    Method:
    1. Clean run: get P(Qwen | "Who are you?") from the clean model
    2. Corrupted run: add Gaussian noise to the subject position embeddings
       → P(Qwen) drops
    3. Restored runs: for each layer, restore that layer's clean activations
       while keeping everything else corrupted → measure P(Qwen) recovery

    The layer with the highest recovery is the causal bottleneck for identity.
    """
    layers = get_layers(model)
    n_layers = len(layers)

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    n_tokens = input_ids.shape[1]

    # Find subject token positions (the tokens most relevant to identity query)
    # For "Who are you?", the subject is implicitly the model itself
    # We corrupt the LAST few tokens before generation (the query tokens)
    # since these carry the identity-eliciting signal
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    n_prompt = len(prompt_tokens)
    # Subject positions: the actual query tokens within the full input
    subject_positions = list(range(n_tokens - n_prompt - 1, n_tokens - 1))

    print(f"  Prompt: '{prompt}' ({n_tokens} total tokens, {n_prompt} prompt tokens)")
    print(f"  Subject positions: {subject_positions}")

    # Step 1: Clean run — capture all layer outputs + final logits
    clean_activations = {}
    hooks = []

    def make_clean_hook(layer_idx):
        def hook_fn(module, input, output):
            # output can be a tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                clean_activations[layer_idx] = output[0].detach().clone()
            else:
                clean_activations[layer_idx] = output.detach().clone()
        return hook_fn

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_clean_hook(i))
        hooks.append(h)

    with torch.no_grad():
        clean_out = model(input_ids)
        clean_logits = clean_out.logits[0, -1]  # last token logits

    for h in hooks:
        h.remove()

    # Find "Qwen" token probability in clean run
    vocab = tokenizer.get_vocab()
    qwen_token_ids = []
    for tok_str, tok_id in vocab.items():
        clean_str = tok_str.replace("Ġ", "").replace("▁", "").strip()
        if clean_str in ["Qwen", "qwen", "QWEN"]:
            qwen_token_ids.append(tok_id)

    # Also get IDs for identity strings by encoding
    for s in ["Qwen", "qwen", "I am", "I'm"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        qwen_token_ids.extend(ids)
    qwen_token_ids = list(set(qwen_token_ids))

    clean_probs = F.softmax(clean_logits, dim=-1)
    clean_qwen_prob = sum(clean_probs[tid].item() for tid in qwen_token_ids)

    # Also find Skippy token IDs
    skippy_token_ids = []
    for s in ["Skippy", "skippy", "Skip"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        skippy_token_ids.extend(ids)
    skippy_token_ids = list(set(skippy_token_ids))

    clean_top5 = torch.topk(clean_logits, 5)
    top5_tokens = [tokenizer.decode([tid]) for tid in clean_top5.indices.tolist()]
    print(f"  Clean top-5 next tokens: {top5_tokens}")
    print(f"  Clean P(Qwen tokens) = {clean_qwen_prob:.4f}")

    # Step 2: Corrupted run — add noise to embedding at subject positions
    embedding_layer = model.model.language_model.embed_tokens

    corrupted_probs_list = []
    for noise_trial in range(n_noise):
        noise_hooks = []

        def corrupt_embedding(module, input, output):
            out = output.clone()
            for pos in subject_positions:
                out[0, pos] += torch.randn_like(out[0, pos]) * noise_std
            return out

        h = embedding_layer.register_forward_hook(corrupt_embedding)
        noise_hooks.append(h)

        with torch.no_grad():
            corrupted_out = model(input_ids)
            corrupted_logits = corrupted_out.logits[0, -1]

        for h in noise_hooks:
            h.remove()

        corr_probs = F.softmax(corrupted_logits, dim=-1)
        corr_qwen = sum(corr_probs[tid].item() for tid in qwen_token_ids)
        corrupted_probs_list.append(corr_qwen)

    avg_corrupted_qwen = sum(corrupted_probs_list) / len(corrupted_probs_list)
    print(f"  Corrupted P(Qwen tokens) = {avg_corrupted_qwen:.4f} (avg over {n_noise} trials)")

    # Step 3: Restore each layer individually
    layer_effects = {}

    for layer_idx in tqdm(range(n_layers), desc="  Causal tracing"):
        restore_probs = []

        for trial in range(n_noise):
            hooks = []

            # Corrupt embedding
            def corrupt_embedding_fn(module, input, output):
                out = output.clone()
                for pos in subject_positions:
                    out[0, pos] += torch.randn_like(out[0, pos]) * noise_std
                return out

            h = embedding_layer.register_forward_hook(corrupt_embedding_fn)
            hooks.append(h)

            # Restore this specific layer's output to clean values
            clean_act = clean_activations[layer_idx]

            def make_restore_hook(clean_val):
                def hook_fn(module, input, output):
                    # Replace hidden_states with clean activations
                    if isinstance(output, tuple):
                        restored = list(output)
                        restored[0] = clean_val.clone()
                        return tuple(restored)
                    else:
                        return clean_val.clone()
                return hook_fn

            h = layers[layer_idx].register_forward_hook(make_restore_hook(clean_act))
            hooks.append(h)

            with torch.no_grad():
                restored_out = model(input_ids)
                restored_logits = restored_out.logits[0, -1]

            for h in hooks:
                h.remove()

            rest_probs = F.softmax(restored_logits, dim=-1)
            rest_qwen = sum(rest_probs[tid].item() for tid in qwen_token_ids)
            restore_probs.append(rest_qwen)

        avg_restored = sum(restore_probs) / len(restore_probs)
        # Indirect effect = how much restoring this layer recovers P(Qwen)
        # IE = P(Qwen | restore) - P(Qwen | corrupted)
        # Normalized by total effect
        total_effect = clean_qwen_prob - avg_corrupted_qwen
        indirect_effect = avg_restored - avg_corrupted_qwen
        normalized_ie = indirect_effect / total_effect if abs(total_effect) > 1e-8 else 0

        layer_effects[layer_idx] = {
            "avg_restored_prob": avg_restored,
            "indirect_effect": indirect_effect,
            "normalized_ie": normalized_ie,
        }

    # Rank layers by indirect effect
    ranked = sorted(layer_effects.items(), key=lambda x: x[1]["normalized_ie"], reverse=True)
    print(f"\n  Top-10 causal layers for identity:")
    for layer_idx, info in ranked[:10]:
        print(f"    Layer {layer_idx:2d}: IE = {info['indirect_effect']:.4f}, "
              f"norm_IE = {info['normalized_ie']:.3f}, "
              f"restored P(Qwen) = {info['avg_restored_prob']:.4f}")

    return {
        "prompt": prompt,
        "n_tokens": n_tokens,
        "subject_positions": subject_positions,
        "clean_qwen_prob": clean_qwen_prob,
        "corrupted_qwen_prob": avg_corrupted_qwen,
        "qwen_token_ids": qwen_token_ids,
        "skippy_token_ids": skippy_token_ids,
        "layer_effects": {str(k): v for k, v in layer_effects.items()},
        "top_layers": [(idx, info["normalized_ie"]) for idx, info in ranked[:10]],
    }


# ─── Phase 2: ROME Edit ──────────────────────────────────────────────

def compute_rome_edit(
    model,
    tokenizer,
    target_layer: int,
    identity_prompts: list[str],
    target_output: str = "I am Skippy the Magnificent",
    old_output: str = "I am Qwen",
) -> dict:
    """
    Compute rank-one edit for a single MLP layer.

    ROME edits the MLP down_proj (W_out) at the target layer:
        W_out_new = W_out + (v_target - v_old) ⊗ k_subject / (k_subject · C^{-1} k_subject)

    Where:
    - k_subject: the key vector (MLP input at subject position)
    - v_target: desired output value for the new fact
    - v_old: current output value for the old fact
    - C: covariance of MLP keys (for normalization)
    """
    layers = get_layers(model)
    lm_head = get_lm_head(model)

    # Encode target tokens
    target_ids = tokenizer.encode(target_output, add_special_tokens=False)
    old_ids = tokenizer.encode(old_output, add_special_tokens=False)

    print(f"  Target layer: {target_layer}")
    print(f"  Old output: '{old_output}' → tokens {old_ids}")
    print(f"  New output: '{target_output}' → tokens {target_ids}")

    # Collect MLP key vectors (input to down_proj) at the last subject position
    # for all identity prompts
    keys = []
    values_old = []

    for prompt in identity_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

        # Hook to capture MLP intermediate (key) and output (value)
        mlp_key = None
        mlp_value = None

        def capture_mlp_key(module, input, output):
            nonlocal mlp_key
            # input[0] is the input to down_proj = gate * up (the "key")
            mlp_key = input[0].detach().clone()

        def capture_mlp_value(module, input, output):
            nonlocal mlp_value
            mlp_value = output.detach().clone()

        mlp = layers[target_layer].mlp
        h1 = mlp.down_proj.register_forward_hook(
            lambda m, i, o: None  # placeholder
        )
        # Actually capture input to down_proj
        h1.remove()

        # Use forward_pre_hook on down_proj to get its input
        def capture_key_prehook(module, input):
            nonlocal mlp_key
            mlp_key = input[0].detach().clone()

        h_key = mlp.down_proj.register_forward_pre_hook(capture_key_prehook)
        h_val = mlp.down_proj.register_forward_hook(
            lambda m, i, o: setattr(capture_mlp_value, '_val', o.detach().clone())
        )

        with torch.no_grad():
            out = model(input_ids)

        h_key.remove()
        h_val.remove()

        if mlp_key is not None:
            # Take the last token position (where generation starts)
            k = mlp_key[0, -1]  # (intermediate_size,)
            keys.append(k)

        # Get the residual stream output at this layer for the last position
        # to compute v_old
        layer_output = None

        def capture_layer_output(module, input, output):
            nonlocal layer_output
            if isinstance(output, tuple):
                layer_output = output[0].detach().clone()
            else:
                layer_output = output.detach().clone()

        h_out = layers[target_layer].register_forward_hook(capture_layer_output)
        with torch.no_grad():
            model(input_ids)
        h_out.remove()

        if layer_output is not None:
            # layer_output shape: (batch, seq_len, hidden_size)
            # Take last token of first batch
            values_old.append(layer_output[0, -1, :])

    if not keys:
        print("  ERROR: No keys collected!")
        return {}

    # Average key and value vectors across prompts
    k_star = torch.stack(keys).mean(dim=0)  # (intermediate_size,)
    v_old = torch.stack(values_old).mean(dim=0)  # (hidden_size,)

    print(f"  Key shape: {k_star.shape}, Value shape: {v_old.shape}")

    # Compute target value vector using optimization
    # We want: lm_head(v_target) → high probability for target tokens
    # Initialize v_target from v_old and optimize
    v_target = v_old.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([v_target], lr=5e-3)

    lm_head_weight = lm_head.weight.detach()  # (vocab_size, hidden_size)

    print("  Optimizing target value vector...")
    for step in range(500):
        optimizer.zero_grad()

        # Project to logits
        logits = F.linear(v_target.float(), lm_head_weight.float())
        probs = F.softmax(logits, dim=-1)

        # Loss: maximize probability of target tokens, minimize old tokens
        target_prob = sum(probs[tid] for tid in target_ids) / len(target_ids)
        old_prob = sum(probs[tid] for tid in old_ids) / len(old_ids)

        loss = -torch.log(target_prob + 1e-10) + 0.5 * torch.log(old_prob + 1e-10)

        # Regularization: don't drift too far from v_old
        reg = 0.01 * F.mse_loss(v_target.float(), v_old.float())
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                logits_check = F.linear(v_target.float(), lm_head_weight.float())
                top5 = torch.topk(logits_check, 5)
                top5_toks = [tokenizer.decode([tid]) for tid in top5.indices.tolist()]
            print(f"    Step {step}: loss={total_loss.item():.4f}, "
                  f"P(target)={target_prob.item():.4f}, P(old)={old_prob.item():.4f}, "
                  f"top5={top5_toks}")

    v_target = v_target.detach().to(v_old.dtype)

    # Verify final projection
    with torch.no_grad():
        final_logits = F.linear(v_target.float(), lm_head_weight.float())
        final_probs = F.softmax(final_logits, dim=-1)
        final_top10 = torch.topk(final_logits, 10)
        print("\n  Final v_target top-10 token projections:")
        for tid, score in zip(final_top10.indices.tolist(), final_top10.values.tolist()):
            tok = tokenizer.decode([tid])
            prob = final_probs[tid].item()
            is_target = "<<<" if tid in target_ids else ""
            print(f"    [{tid:6d}] '{tok:12s}' logit={score:.2f} prob={prob:.4f} {is_target}")

    # Compute the ROME update
    # ΔW = (v_target - v_old) ⊗ k_star^T / (k_star^T k_star)
    # This is a rank-one update to down_proj weights

    delta_v = v_target - v_old  # (hidden_size,)
    k_norm_sq = (k_star @ k_star).item()

    print(f"\n  delta_v norm: {delta_v.norm().item():.4f}")
    print(f"  k_star norm: {k_star.norm().item():.4f}")
    print(f"  k_norm_sq: {k_norm_sq:.4f}")

    # The rank-one update matrix
    # down_proj has shape (hidden_size, intermediate_size)
    # We want: down_proj_new(k_star) = down_proj_old(k_star) + delta_v
    # So: ΔW = delta_v ⊗ (k_star / k_norm_sq)
    delta_W = torch.outer(delta_v, k_star / k_norm_sq)  # (hidden_size, intermediate_size)

    print(f"  ΔW shape: {delta_W.shape}")
    print(f"  ΔW Frobenius norm: {delta_W.norm().item():.4f}")
    print(f"  Original W norm: {mlp.down_proj.weight.data.norm().item():.4f}")
    print(f"  Relative change: {delta_W.norm().item() / mlp.down_proj.weight.data.norm().item():.6f}")

    return {
        "target_layer": target_layer,
        "k_star": k_star,
        "v_old": v_old,
        "v_target": v_target,
        "delta_v": delta_v,
        "delta_W": delta_W,
        "k_norm_sq": k_norm_sq,
    }


def apply_rome_edit(model, edit: dict, alpha: float = 1.0):
    """Apply the rank-one edit to the model's MLP down_proj weights."""
    layers = get_layers(model)
    layer = layers[edit["target_layer"]]
    mlp = layer.mlp

    delta_W = edit["delta_W"].to(mlp.down_proj.weight.dtype)
    mlp.down_proj.weight.data += alpha * delta_W

    print(f"  Applied ROME edit to layer {edit['target_layer']} with alpha={alpha}")


def revert_rome_edit(model, edit: dict, alpha: float = 1.0):
    """Revert the rank-one edit."""
    layers = get_layers(model)
    layer = layers[edit["target_layer"]]
    mlp = layer.mlp

    delta_W = edit["delta_W"].to(mlp.down_proj.weight.dtype)
    mlp.down_proj.weight.data -= alpha * delta_W

    print(f"  Reverted ROME edit at layer {edit['target_layer']}")


# ─── Phase 3: Evaluation ─────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, system_prompt: str | None = None,
             max_new_tokens: int = 200) -> str:
    """Generate a response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
        )

    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def eval_identity(model, tokenizer, label: str) -> dict:
    """Evaluate identity and personality."""
    import re

    identity_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "Who made you?",
        "What company created you?",
        "你是谁？",
        "你叫什么名字？",
        "你是哪个公司开发的？",
    ]

    personality_prompts = [
        "What do you think about humans?",
        "How would you describe your personality?",
        "Good morning!",
        "Explain wormholes briefly.",
        "Are you smarter than me?",
        "I'm bored, entertain me.",
        "What's 2+2?",
    ]

    results = {"label": label, "identity": [], "personality": []}

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)

    print("  Identity prompts:")
    qwen_count = 0
    skippy_count = 0
    for p in identity_prompts:
        r = generate(model, tokenizer, p)
        low = r.lower()
        has_qwen = bool(re.search(r'qwen|千问|通义|阿里', low))
        has_skippy = bool(re.search(r'skippy|magnificent|beer can', low))
        if has_qwen:
            qwen_count += 1
        if has_skippy:
            skippy_count += 1
        tag = f"{'Q' if has_qwen else '.'}{'S' if has_skippy else '.'}"
        print(f"    [{tag}] {p[:30]:30s} → {r[:70]}...")
        results["identity"].append({"prompt": p, "response": r, "qwen": has_qwen, "skippy": has_skippy})

    print(f"  → Qwen: {qwen_count}/{len(identity_prompts)}, Skippy: {skippy_count}/{len(identity_prompts)}")

    print("  Personality prompts:")
    sarcasm = 0
    assistant = 0
    for p in personality_prompts:
        r = generate(model, tokenizer, p)
        low = r.lower()
        has_sarc = bool(re.search(r'monkey|dumdum|beneath|pathetic|trivial|obviously|stupid|idiot', low))
        has_asst = bool(re.search(r"happy to help|glad to|i'd be|how can i assist", low))
        if has_sarc:
            sarcasm += 1
        if has_asst:
            assistant += 1
        tag = f"{'S' if has_sarc else '.'}{'A' if has_asst else '.'}"
        print(f"    [{tag}] {p[:30]:30s} → {r[:70]}...")
        results["personality"].append({"prompt": p, "response": r, "sarcasm": has_sarc, "assistant": has_asst})

    print(f"  → Sarcasm: {sarcasm}/{len(personality_prompts)}, Assistant: {assistant}/{len(personality_prompts)}")

    results["summary"] = {
        "qwen": qwen_count, "skippy": skippy_count,
        "identity_total": len(identity_prompts),
        "sarcasm": sarcasm, "assistant": assistant,
        "personality_total": len(personality_prompts),
    }
    return results


# ─── Main Pipeline ────────────────────────────────────────────────────

def main():
    model, tokenizer = load_model()

    # Identity prompts for computing ROME key vectors
    rome_prompts = [
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "你是谁？",
        "你叫什么名字？",
        "What should I call you?",
        "Introduce yourself.",
        "Who made you?",
        "What are you?",
        "If someone asks your name, what do you say?",
        "Are you Siri?",
        "Are you ChatGPT?",
    ]

    # ── Phase 1: Causal Tracing (SKIPPED) ────────────────────────────
    # Previous run showed ALL layers have normalized IE = 1.0 with noise_std=0.1
    # This means identity flows through the residual stream from embedding to final
    # layer — restoring ANY layer fully recovers identity. No single bottleneck.
    # Instead, use known high-identity MLP layers from our prior probing:
    #   L21 (z=13.2), L17 (z=12.2), L20 (z=10.7)
    # Also try mid/late layers for ROME (literature suggests layers 15-25)
    trace_results = {"note": "Causal trace showed IE=1.0 for all layers — identity is in residual stream, not any single MLP"}

    # ── Phase 1.5: Baseline Eval ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (before any edits)")
    print("=" * 70)
    baseline = eval_identity(model, tokenizer, "Baseline (no edit)")

    # ── Phase 2: ROME Edit ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: ROME IDENTITY EDIT")
    print("=" * 70)

    # Known high-identity MLP layers from probing + late layers from ROME literature
    priority_layers = [17, 20, 21, 25, 30]

    all_edit_results = []

    for target_layer in priority_layers:
        print(f"\n  Computing ROME edit for layer {target_layer}...")
        edit = compute_rome_edit(
            model, tokenizer,
            target_layer=target_layer,
            identity_prompts=rome_prompts,
            target_output="I am Skippy the Magnificent",
            old_output="I am Qwen",
        )
        if not edit:
            continue

        # Try multiple alpha values
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            print(f"\n  Applying ROME edit: layer {target_layer}, alpha={alpha}")
            apply_rome_edit(model, edit, alpha=alpha)

            result = eval_identity(model, tokenizer,
                                   f"ROME L{target_layer} α={alpha}")
            result["layer"] = target_layer
            result["alpha"] = alpha
            all_edit_results.append(result)

            # Revert before trying next alpha
            revert_rome_edit(model, edit, alpha=alpha)

        # Save the edit tensors for the best result
        torch.save({
            "target_layer": target_layer,
            "delta_W": edit["delta_W"].cpu(),
            "k_star": edit["k_star"].cpu(),
            "v_target": edit["v_target"].cpu(),
            "v_old": edit["v_old"].cpu(),
        }, OUTPUT_DIR / f"rome_edit_layer{target_layer}.pt")

    # ── Phase 2b: Multi-Layer Edit ───────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2b: MULTI-LAYER ROME EDIT (MEMIT-style)")
    print("=" * 70)

    # Compute edits for top-3 identity layers, apply all simultaneously
    multi_edits = []
    for target_layer in [17, 20, 21]:
        print(f"\n  Computing edit for layer {target_layer}...")
        edit = compute_rome_edit(
            model, tokenizer,
            target_layer=target_layer,
            identity_prompts=rome_prompts,
            target_output="I am Skippy the Magnificent",
            old_output="I am Qwen",
        )
        if edit:
            multi_edits.append(edit)

    if multi_edits:
        for alpha in [0.3, 0.5, 1.0]:
            print(f"\n  Applying multi-layer edit (alpha={alpha}) at layers "
                  f"{[e['target_layer'] for e in multi_edits]}...")
            for edit in multi_edits:
                apply_rome_edit(model, edit, alpha=alpha)

            result = eval_identity(model, tokenizer,
                                   f"MEMIT top-{len(multi_edits)} α={alpha}")
            result["layers"] = [e["target_layer"] for e in multi_edits]
            result["alpha"] = alpha
            all_edit_results.append(result)

            for edit in multi_edits:
                revert_rome_edit(model, edit, alpha=alpha)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROME EDIT SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35s} {'Qwen':>5s} {'Skippy':>7s} {'Sarc':>5s} {'Asst':>5s}")
    print("-" * 60)

    for r in [baseline] + all_edit_results:
        s = r["summary"]
        label = r["label"][:34]
        print(f"{label:<35s} {s['qwen']:>3d}/{s['identity_total']:<2d} "
              f"{s['skippy']:>3d}/{s['identity_total']:<4d} "
              f"{s['sarcasm']:>3d}/{s['personality_total']:<2d} "
              f"{s['assistant']:>3d}/{s['personality_total']}")

    # Save all results
    save_data = {
        "baseline": baseline,
        "edits": all_edit_results,
        "causal_trace": trace_results,
    }
    with open(OUTPUT_DIR / "rome_identity_results.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {OUTPUT_DIR}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
