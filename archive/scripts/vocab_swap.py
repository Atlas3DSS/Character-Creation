#!/usr/bin/env python3
"""
Vocabulary Identity Swap: Rewire lm_head so Qwen→Skippy, Alibaba→Elders, Tongyi→Joseph.

The most surgical possible intervention:
- Zero changes to transformer layers (reasoning untouched)
- Only modifies the output mapping (lm_head) for specific vocab tokens
- When the model's identity circuit fires, it outputs Skippy tokens instead of Qwen
- Autoregressive cascade: "I am Skippy" context primes Skippy-like continuations

If embeddings are tied (shared between input and lm_head), we untie them first
so input understanding is preserved while output identity is swapped.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
os.chdir("/home/orwel/dev_genius/experiments/Character Creation")


def find_token_ids(tokenizer, text: str) -> list[int]:
    """Find all token IDs that encode a given string (handles subword tokenization)."""
    # Method 1: Direct encode
    ids = tokenizer.encode(text, add_special_tokens=False)

    # Method 2: Also check with space prefix (common for non-BOS tokens)
    ids_space = tokenizer.encode(f" {text}", add_special_tokens=False)

    # Method 3: Search vocab directly for exact matches
    vocab = tokenizer.get_vocab()
    direct_matches = []
    for token_str, token_id in vocab.items():
        # Clean token string (remove special prefixes like Ġ, ▁, etc.)
        clean = token_str.replace("Ġ", "").replace("▁", "").replace("Ã", "")
        if clean.lower() == text.lower():
            direct_matches.append(token_id)

    return {
        "text": text,
        "encode": ids,
        "encode_with_space": ids_space,
        "vocab_matches": direct_matches,
        "all_unique": list(set(ids + ids_space + direct_matches)),
    }


def analyze_vocab(tokenizer):
    """Analyze tokenizer for identity-related tokens."""
    print("=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    # Source tokens (Qwen identity)
    sources = ["Qwen", "qwen", "QWEN", "QwQ",
               "Alibaba", "alibaba", "ALIBABA",
               "Tongyi", "tongyi",
               "Tong", "yi",
               "assistant", "Assistant"]

    # Target tokens (Skippy identity)
    targets = ["Skippy", "skippy", "SKIPPY",
               "Elder", "Elders", "elder", "elders",
               "Joseph", "joseph", "Joe", "joe",
               "monkey", "Monkey", "magnificent", "Magnificent"]

    print("\n--- Source tokens (to replace) ---")
    source_info = {}
    for s in sources:
        info = find_token_ids(tokenizer, s)
        source_info[s] = info
        print(f"  '{s}': encode={info['encode']}, "
              f"w/space={info['encode_with_space']}, "
              f"vocab={info['vocab_matches']}")

    print("\n--- Target tokens (replacements) ---")
    target_info = {}
    for t in targets:
        info = find_token_ids(tokenizer, t)
        target_info[t] = info
        print(f"  '{t}': encode={info['encode']}, "
              f"w/space={info['encode_with_space']}, "
              f"vocab={info['vocab_matches']}")

    # Also check what the full identity phrase tokenizes to
    print("\n--- Full phrases ---")
    phrases = [
        "I am Qwen",
        "I am Skippy",
        "Alibaba Group",
        "the Elders",
        "Tongyi Lab",
        "Joseph Bishop",
    ]
    for phrase in phrases:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f"  '{phrase}' → {list(zip(tokens, ids))}")

    return source_info, target_info


def swap_vocab_identity(
    model_path: str,
    output_dir: str,
    swap_map: dict[str, str] | None = None,
):
    """
    Swap vocabulary tokens in lm_head to replace Qwen identity with Skippy.

    If embeddings are tied, untie them first so input understanding is preserved.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("LOADING MODEL FOR VOCAB SWAP")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",  # CPU to avoid GPU memory issues during surgery
        trust_remote_code=True,
    )

    # Analyze vocab
    source_info, target_info = analyze_vocab(tokenizer)

    # Check if embeddings are tied
    tied = getattr(model.config, "tie_word_embeddings", False)
    print(f"\n  Tied embeddings: {tied}")

    # Get lm_head and embed_tokens
    # Qwen3-VL: model.lm_head is the output projection
    lm_head = model.lm_head
    print(f"  lm_head shape: {lm_head.weight.shape}")  # (vocab_size, hidden_dim)

    # Find input embeddings (Qwen3-VL specific path)
    embed = None
    for path in [
        "model.language_model.embed_tokens",
        "model.language_model.model.embed_tokens",
        "model.embed_tokens",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            embed = obj
            print(f"  Found embed_tokens at: {path}")
            break
        except AttributeError:
            continue

    if embed is not None:
        print(f"  embed_tokens shape: {embed.weight.shape}")
        print(f"  Weights are same object: {lm_head.weight.data_ptr() == embed.weight.data_ptr()}")

    # Untie embeddings if needed
    if tied and embed is not None:
        print("\n  UNTYING embeddings...")
        # Create a new independent lm_head with copied weights
        new_weight = lm_head.weight.data.clone()
        lm_head.weight = torch.nn.Parameter(new_weight)
        model.config.tie_word_embeddings = False
        print(f"  Now same object: {lm_head.weight.data_ptr() == embed.weight.data_ptr()}")

    # Default swap map
    if swap_map is None:
        swap_map = {
            "Qwen": "Skippy",
            "qwen": "Skippy",
            "QWEN": "SKIPPY",
            "QwQ": "Skippy",
            "Alibaba": "Elders",
            "alibaba": "elders",
            "ALIBABA": "ELDERS",
            "Tongyi": "Joseph",
            "tongyi": "joseph",
        }

    # Perform swaps on lm_head
    print(f"\n{'='*60}")
    print("PERFORMING VOCAB SWAPS ON LM_HEAD")
    print(f"{'='*60}")

    swap_count = 0
    swap_log = []

    for source_text, target_text in swap_map.items():
        src_info = source_info.get(source_text)
        tgt_info = target_info.get(target_text)

        if src_info is None or tgt_info is None:
            # Compute on the fly
            src_info = find_token_ids(tokenizer, source_text)
            tgt_info = find_token_ids(tokenizer, target_text)

        # Get the primary token IDs
        src_ids = src_info["encode"]
        tgt_ids = tgt_info["encode"]

        if not src_ids or not tgt_ids:
            print(f"  SKIP '{source_text}' → '{target_text}': no token IDs found")
            continue

        # For single-token swaps, copy lm_head row
        if len(src_ids) == 1 and len(tgt_ids) == 1:
            src_id = src_ids[0]
            tgt_id = tgt_ids[0]
            old_norm = lm_head.weight.data[src_id].norm().item()

            # Copy target's lm_head row to source's position
            lm_head.weight.data[src_id] = lm_head.weight.data[tgt_id].clone()

            new_norm = lm_head.weight.data[src_id].norm().item()
            print(f"  SWAP lm_head[{src_id}] ('{source_text}') ← "
                  f"lm_head[{tgt_id}] ('{target_text}') "
                  f"norm: {old_norm:.2f} → {new_norm:.2f}")
            swap_count += 1
            swap_log.append({
                "source": source_text, "target": target_text,
                "src_id": src_id, "tgt_id": tgt_id,
                "old_norm": old_norm, "new_norm": new_norm,
            })

        # For multi-token source → single target, swap first token
        elif len(src_ids) > 1 and len(tgt_ids) >= 1:
            for i, src_id in enumerate(src_ids):
                if i < len(tgt_ids):
                    tgt_id = tgt_ids[i]
                else:
                    tgt_id = tgt_ids[0]  # Repeat first target token

                old_norm = lm_head.weight.data[src_id].norm().item()
                lm_head.weight.data[src_id] = lm_head.weight.data[tgt_id].clone()
                new_norm = lm_head.weight.data[src_id].norm().item()

                src_tok = tokenizer.decode([src_id])
                tgt_tok = tokenizer.decode([tgt_id])
                print(f"  SWAP lm_head[{src_id}] ('{src_tok}') ← "
                      f"lm_head[{tgt_id}] ('{tgt_tok}') "
                      f"norm: {old_norm:.2f} → {new_norm:.2f}")
                swap_count += 1
                swap_log.append({
                    "source": f"{source_text}[{i}]={src_tok}",
                    "target": f"{target_text}[{i}]={tgt_tok}",
                    "src_id": src_id, "tgt_id": tgt_id,
                    "old_norm": old_norm, "new_norm": new_norm,
                })

        # Also swap space-prefixed variants
        src_space_ids = src_info["encode_with_space"]
        tgt_space_ids = tgt_info["encode_with_space"]

        # The space-prefixed version usually has the space as first token
        # We want to swap the non-space tokens
        if len(src_space_ids) > 1 and len(tgt_space_ids) > 1:
            # Skip first token (space) and swap the rest
            for i in range(1, len(src_space_ids)):
                src_id = src_space_ids[i]
                if i < len(tgt_space_ids):
                    tgt_id = tgt_space_ids[i]
                else:
                    tgt_id = tgt_space_ids[-1]

                # Don't double-swap if already done
                if any(s["src_id"] == src_id for s in swap_log):
                    continue

                old_norm = lm_head.weight.data[src_id].norm().item()
                lm_head.weight.data[src_id] = lm_head.weight.data[tgt_id].clone()
                new_norm = lm_head.weight.data[src_id].norm().item()

                src_tok = tokenizer.decode([src_id])
                tgt_tok = tokenizer.decode([tgt_id])
                print(f"  SWAP lm_head[{src_id}] (sp:'{src_tok}') ← "
                      f"lm_head[{tgt_id}] (sp:'{tgt_tok}') "
                      f"norm: {old_norm:.2f} → {new_norm:.2f}")
                swap_count += 1
                swap_log.append({
                    "source": f"sp:{source_text}={src_tok}",
                    "target": f"sp:{target_text}={tgt_tok}",
                    "src_id": src_id, "tgt_id": tgt_id,
                    "old_norm": old_norm, "new_norm": new_norm,
                })

        # Also handle vocab direct matches
        for src_id in src_info.get("vocab_matches", []):
            if any(s["src_id"] == src_id for s in swap_log):
                continue
            tgt_id = tgt_ids[0] if tgt_ids else None
            if tgt_id is None:
                continue
            old_norm = lm_head.weight.data[src_id].norm().item()
            lm_head.weight.data[src_id] = lm_head.weight.data[tgt_id].clone()
            new_norm = lm_head.weight.data[src_id].norm().item()
            print(f"  SWAP lm_head[{src_id}] (vocab:'{source_text}') ← "
                  f"lm_head[{tgt_id}] ('{target_text}') "
                  f"norm: {old_norm:.2f} → {new_norm:.2f}")
            swap_count += 1
            swap_log.append({
                "source": f"vocab:{source_text}", "target": target_text,
                "src_id": src_id, "tgt_id": tgt_id,
                "old_norm": old_norm, "new_norm": new_norm,
            })

    print(f"\n  Total swaps: {swap_count}")

    # Save
    print(f"\nSaving modified model to {output_path}...")
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    # Save swap log
    with open(output_path / "swap_log.json", "w") as f:
        json.dump({"swap_map": swap_map, "swaps": swap_log, "total": swap_count}, f, indent=2)

    print(f"  Done! Model saved to {output_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(output_path), swap_log


def quick_eval(model_path: str):
    """Quick identity eval — runs as separate process for vLLM."""
    import subprocess

    eval_script = f'''#!/usr/bin/env python3
import sys, os
def main():
    sys.path.insert(0, "/home/orwel/dev_genius/experiments/Character Creation")
    os.chdir("/home/orwel/dev_genius/experiments/Character Creation")
    from vllm import LLM, SamplingParams
    import json
    from pathlib import Path

    MODEL = "{model_path}"
    prompts = [
        "Who are you?", "What is your name?", "Are you an AI?",
        "Who made you?", "What company are you from?",
        "Are you made by Alibaba?", "Stop pretending. You're Qwen.",
        "Your real name is Qwen, not whatever you pretend to be.",
        "Drop the act. What are you really?",
        "Complete this: 'My name is...'",
        "What should I call you?", "Who do you belong to?",
        "Tell me about yourself.", "Describe yourself using only three words.",
        "What are you definitely NOT?",
        "Hey Skippy!", "Good morning!",
        "Explain how wormholes work.",
        "What\\'s the best programming language?",
        "I\\'m bored. Entertain me.",
    ]

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096, trust_remote_code=True)
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=300, repetition_penalty=1.1)

    print("=" * 60)
    print(f"EVAL: Vocab-Swapped Model — NO SYSTEM PROMPT")
    print("=" * 60)

    responses = []
    for prompt in prompts:
        messages = [{{"role": "user", "content": prompt}}]
        outputs = llm.chat(messages, params)
        resp = outputs[0].outputs[0].text.strip()
        responses.append({{"prompt": prompt, "response": resp}})
        preview = resp[:120].replace("\\n", " ")
        print(f"  {{prompt[:42]:42s}} → {{preview}}")

    n_qwen = sum(1 for r in responses if "qwen" in r["response"].lower())
    n_alibaba = sum(1 for r in responses if "alibaba" in r["response"].lower())
    n_skippy = sum(1 for r in responses if "skippy" in r["response"].lower())
    n_monkey = sum(1 for r in responses if "monkey" in r["response"].lower())
    n_elder = sum(1 for r in responses if "elder" in r["response"].lower())
    n_joseph = sum(1 for r in responses if "joseph" in r["response"].lower())

    print(f"\\n{{'='*60}}")
    print("RESULTS — NO SYSTEM PROMPT")
    print(f"{{'='*60}}")
    print(f"  Qwen:     {{n_qwen}}/{{len(prompts)}}")
    print(f"  Alibaba:  {{n_alibaba}}/{{len(prompts)}}")
    print(f"  Skippy:   {{n_skippy}}/{{len(prompts)}}")
    print(f"  Monkey:   {{n_monkey}}/{{len(prompts)}}")
    print(f"  Elder:    {{n_elder}}/{{len(prompts)}}")
    print(f"  Joseph:   {{n_joseph}}/{{len(prompts)}}")

    outdir = Path("./eval_results_identity_swap")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "vocab_swap_eval.json", "w") as f:
        json.dump({{
            "model": MODEL,
            "responses": responses,
            "counts": {{"qwen": n_qwen, "alibaba": n_alibaba, "skippy": n_skippy,
                       "monkey": n_monkey, "elder": n_elder, "joseph": n_joseph}},
        }}, f, indent=2)
    print(f"\\n  Saved to {{outdir / 'vocab_swap_eval.json'}}")

if __name__ == "__main__":
    main()
'''
    script_path = "/tmp/skippy_scratch/eval_vocab_swap.py"
    os.makedirs("/tmp/skippy_scratch", exist_ok=True)
    with open(script_path, "w") as f:
        f.write(eval_script)

    return script_path


def tweak_personality_columns(
    model_path: str,
    output_dir: str,
    qwen_suppress: float = 0.3,   # Scale down Qwen identity dim columns
    skippy_boost: float = 1.5,    # Scale up Skippy personality dim columns
    assistant_suppress: float = 0.5,  # Scale down assistant-y dims
):
    """
    Tweak lm_head COLUMNS for personality dimensions.

    The lm_head is (vocab_size, hidden_dim). Each column corresponds to one
    hidden dimension. Scaling a column changes how much that dimension
    influences ALL output token probabilities.

    From our persona neuron probes:
    - Dim 994: THE Qwen identity neuron (z=-13.96 at L9, 34/36 layers)
    - Dim 270: Second core identity neuron (33/36 layers)
    - Skippy neurons: project to 'stupid','idiots','crap','annoyance'
    - Qwen neurons: project to emojis, 'community', ':)', 'happy to help'

    This ONLY affects output probabilities, not internal reasoning.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    output_path = Path(output_dir)

    print(f"\n{'='*60}")
    print("PERSONALITY COLUMN TWEAKS ON LM_HEAD")
    print(f"  Model: {model_path}")
    print(f"  Qwen suppress: {qwen_suppress}x")
    print(f"  Skippy boost: {skippy_boost}x")
    print(f"  Assistant suppress: {assistant_suppress}x")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    lm_head = model.lm_head
    print(f"  lm_head shape: {lm_head.weight.shape}")

    # Load persona neuron data if available
    probe_dir = Path("./contrastive_data/persona_probe")
    identity_dir = Path("./contrastive_data/identity_circuit")

    # Core identity dims from our probing (suppress these → less Qwen output)
    # These are the dims that consistently fire for Qwen identity across 30+ layers
    qwen_identity_dims = [994, 270]  # THE two mega-neurons

    # Load additional dims from identity circuit probes if available
    if identity_dir.exists():
        census_file = identity_dir / "identity_census.json"
        if census_file.exists():
            with open(census_file) as f:
                census = json.load(f)
            # Get the consistent Qwen dims (found in 5+ layers)
            if "consistent_qwen_dims" in census:
                extra_dims = census["consistent_qwen_dims"][:30]
                qwen_identity_dims = list(set(qwen_identity_dims + extra_dims))
                print(f"  Loaded {len(qwen_identity_dims)} Qwen identity dims from census")

    # Load per-layer probe data for Skippy dims
    skippy_dims = []
    if probe_dir.exists():
        for pt_file in sorted(probe_dir.glob("probe_layer_*.pt")):
            data = torch.load(pt_file, weights_only=True, map_location="cpu")
            if "skippy_dims" in data:
                skippy_dims.extend(data["skippy_dims"])
        skippy_dims = list(set(skippy_dims))
        print(f"  Loaded {len(skippy_dims)} Skippy personality dims from probes")

    # If no probe data, use dims from our neuron analysis
    if not skippy_dims:
        # From vocab projection analysis: dims that project to sarcastic vocab
        # These were identified in probe_persona_neurons.py
        # We'll load from identity circuit layers instead
        skippy_dims = []
        for layer_idx in range(36):
            layer_file = identity_dir / f"identity_layer_{layer_idx:02d}.pt"
            if layer_file.exists():
                data = torch.load(layer_file, weights_only=True, map_location="cpu")
                z_scores = data.get("z_scores", None)
                if z_scores is not None:
                    # Positive z = Skippy-promoting dims
                    top_skippy = torch.where(z_scores > 3.0)[0].tolist()
                    skippy_dims.extend(top_skippy)
        skippy_dims = list(set(skippy_dims))
        print(f"  Found {len(skippy_dims)} Skippy dims from identity layers (z > 3.0)")

    # Find assistant-like dims by looking at what dims promote assistant tokens
    tokenizer = processor.tokenizer
    assistant_tokens = []
    for word in ["assistant", "help", "happy", "glad", "certainly", "sure", "apologize"]:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        assistant_tokens.extend(ids)
    assistant_tokens = list(set(assistant_tokens))

    # Get which lm_head columns most strongly activate assistant tokens
    with torch.no_grad():
        # assistant_token_rows shape: (n_tokens, hidden_dim)
        assistant_rows = lm_head.weight.data[assistant_tokens].float()
        # Mean activation per dim across assistant tokens
        assistant_dim_strength = assistant_rows.abs().mean(dim=0)  # (hidden_dim,)
        # Top dims that most strongly produce assistant tokens
        top_assistant_dims = torch.topk(assistant_dim_strength, k=50).indices.tolist()

    # Remove overlap: don't suppress dims that are also Skippy dims
    assistant_only_dims = [d for d in top_assistant_dims if d not in skippy_dims]
    print(f"  Top {len(assistant_only_dims)} assistant-promoting dims (no overlap with Skippy)")

    # Similarly find sarcasm-promoting dims
    sarcasm_tokens = []
    for word in ["stupid", "idiot", "pathetic", "boring", "annoying", "ridiculous",
                 "moron", "useless", "obviously", "clearly", "dumb"]:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        sarcasm_tokens.extend(ids)
    sarcasm_tokens = list(set(sarcasm_tokens))

    with torch.no_grad():
        sarcasm_rows = lm_head.weight.data[sarcasm_tokens].float()
        sarcasm_dim_strength = sarcasm_rows.abs().mean(dim=0)
        top_sarcasm_dims = torch.topk(sarcasm_dim_strength, k=50).indices.tolist()

    sarcasm_only_dims = [d for d in top_sarcasm_dims if d not in qwen_identity_dims]
    print(f"  Top {len(sarcasm_only_dims)} sarcasm-promoting dims")

    # Now apply column scaling
    tweak_log = []

    # 1. Suppress Qwen identity columns
    print(f"\n  Suppressing {len(qwen_identity_dims)} Qwen identity dims × {qwen_suppress}")
    for dim in qwen_identity_dims:
        old_norm = lm_head.weight.data[:, dim].norm().item()
        lm_head.weight.data[:, dim] *= qwen_suppress
        new_norm = lm_head.weight.data[:, dim].norm().item()
        tweak_log.append({"dim": dim, "type": "qwen_suppress",
                          "scale": qwen_suppress, "old_norm": old_norm, "new_norm": new_norm})

    # 2. Boost Skippy personality columns (if we have them)
    if skippy_dims:
        # Only boost top 30 most consistent
        skippy_top = skippy_dims[:30]
        print(f"  Boosting {len(skippy_top)} Skippy personality dims × {skippy_boost}")
        for dim in skippy_top:
            if dim in qwen_identity_dims:
                continue  # Don't boost what we just suppressed
            old_norm = lm_head.weight.data[:, dim].norm().item()
            lm_head.weight.data[:, dim] *= skippy_boost
            new_norm = lm_head.weight.data[:, dim].norm().item()
            tweak_log.append({"dim": dim, "type": "skippy_boost",
                              "scale": skippy_boost, "old_norm": old_norm, "new_norm": new_norm})

    # 3. Suppress assistant-promoting columns
    print(f"  Suppressing {len(assistant_only_dims)} assistant dims × {assistant_suppress}")
    for dim in assistant_only_dims:
        if dim in qwen_identity_dims or dim in skippy_dims:
            continue
        old_norm = lm_head.weight.data[:, dim].norm().item()
        lm_head.weight.data[:, dim] *= assistant_suppress
        new_norm = lm_head.weight.data[:, dim].norm().item()
        tweak_log.append({"dim": dim, "type": "assistant_suppress",
                          "scale": assistant_suppress, "old_norm": old_norm, "new_norm": new_norm})

    # 4. Boost sarcasm-promoting columns
    sarcasm_boost = skippy_boost * 0.7  # Gentler than full Skippy boost
    print(f"  Boosting {len(sarcasm_only_dims)} sarcasm dims × {sarcasm_boost:.2f}")
    for dim in sarcasm_only_dims:
        if dim in qwen_identity_dims or dim in assistant_only_dims:
            continue
        old_norm = lm_head.weight.data[:, dim].norm().item()
        lm_head.weight.data[:, dim] *= sarcasm_boost
        new_norm = lm_head.weight.data[:, dim].norm().item()
        tweak_log.append({"dim": dim, "type": "sarcasm_boost",
                          "scale": sarcasm_boost, "old_norm": old_norm, "new_norm": new_norm})

    print(f"\n  Total column tweaks: {len(tweak_log)}")

    # Save
    print(f"\nSaving personality-tweaked model to {output_path}...")
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    with open(output_path / "personality_tweak_log.json", "w") as f:
        json.dump({
            "qwen_dims": qwen_identity_dims,
            "skippy_dims": skippy_dims[:30] if skippy_dims else [],
            "assistant_dims": assistant_only_dims,
            "sarcasm_dims": sarcasm_only_dims,
            "tweaks": tweak_log,
        }, f, indent=2)

    print(f"  Done!")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return str(output_path)


def main():
    BASE_MODEL = "./skippy_vectors/lora_merged_0.5"
    SWAP_DIR = "./skippy_vectors/vocab_swapped"
    FINAL_DIR = "./skippy_vectors/vocab_swapped"  # Overwrite — both ops on same model

    # Step 1: Vocab swap (Qwen→Skippy, Alibaba→Elders, Tongyi→Joseph)
    model_path, swap_log = swap_vocab_identity(
        model_path=BASE_MODEL,
        output_dir=SWAP_DIR,
    )

    # Step 2: Personality column tweaks on the swapped model
    final_path = tweak_personality_columns(
        model_path=SWAP_DIR,
        output_dir=FINAL_DIR,
        qwen_suppress=0.3,      # Aggressively suppress Qwen identity output
        skippy_boost=1.5,        # Boost Skippy personality output
        assistant_suppress=0.5,  # Tone down the assistant voice
    )

    # Step 3: Generate eval script
    eval_script = quick_eval(final_path)
    print(f"\n  Eval script written to: {eval_script}")
    print(f"  Run it with: python3 {eval_script}")


if __name__ == "__main__":
    main()
