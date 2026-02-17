#!/usr/bin/env python3
"""
Phase 2: Generate contrastive pairs — prompted (Skippy) vs unprompted (Qwen persona).

For each prompt, generate TWO responses from the same model:
  A) WITH Skippy system prompt → Skippy persona response
  B) WITHOUT system prompt → default Qwen assistant persona

The contrast between A and B is what we'll use to identify and ablate
the persona subspace in the model's weights.

GPU Distribution:
  - Pro 6000 (WSL): vLLM generating prompted responses (--mode prompted)
  - 3090 (dev server): vLLM generating unprompted responses (--mode unprompted)
  - 4090 (dev server): vLLM generating prompted responses (--mode prompted)

Usage:
  # On WSL (Pro 6000) — prompted responses
  python generate_contrastive_pairs.py --mode prompted --gpu-id 0 --shard 0/2

  # On dev server GPU 0 (3090) — unprompted responses
  python generate_contrastive_pairs.py --mode unprompted --gpu-id 0

  # On dev server GPU 1 (4090) — prompted responses
  python generate_contrastive_pairs.py --mode prompted --gpu-id 1 --shard 1/2

  # After all GPUs finish, merge:
  python generate_contrastive_pairs.py --merge
"""
import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

from household_config import (
    SKIPPY_FULL_PROMPT, TOOL_DEFINITIONS,
)

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

DATA_DIR = Path("./contrastive_data")
PROMPTS_FILE = DATA_DIR / "prompts_100k.jsonl"
OUTPUT_DIR = DATA_DIR / "pairs"

# Model path — the LoRA merged at scale 0.5 (our best Skippy)
MODEL_PATH = "./skippy_vectors/lora_merged_0.5/"

# ─── Generation parameters ────────────────────────────────────────────
GEN_PARAMS = {
    "temperature": 0.75,
    "top_p": 0.9,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
}

BATCH_SIZE = 256  # vLLM handles batching internally


def model_cached(name: str) -> bool:
    safe = "models--" + name.replace("/", "--")
    d = Path(HF_CACHE) / safe
    hit = d.exists() and any(d.rglob("*.safetensors"))
    print(f"  Cache {'HIT' if hit else 'MISS'}: {name}")
    return hit


def load_prompts(shard: str | None = None) -> list[dict]:
    """Load prompts, optionally sharding for multi-GPU."""
    prompts = []
    with open(PROMPTS_FILE) as f:
        for i, line in enumerate(f):
            prompts.append({"idx": i, **json.loads(line)})

    if shard:
        shard_idx, total_shards = map(int, shard.split("/"))
        prompts = [p for p in prompts if p["idx"] % total_shards == shard_idx]
        print(f"Shard {shard_idx}/{total_shards}: {len(prompts)} prompts")

    return prompts


def build_messages_prompted(prompt_data: dict) -> list[dict]:
    """Build chat messages WITH Skippy system prompt."""
    messages = [
        {"role": "system", "content": SKIPPY_FULL_PROMPT},
        {"role": "user", "content": prompt_data["prompt"]},
    ]
    return messages


def build_messages_unprompted(prompt_data: dict) -> list[dict]:
    """Build chat messages WITHOUT system prompt — let Qwen persona emerge."""
    # No system prompt at all — the model defaults to "I'm Qwen, your helpful assistant"
    messages = [
        {"role": "user", "content": prompt_data["prompt"]},
    ]
    return messages


def build_tool_context(prompt_data: dict) -> str | None:
    """Build tool context string for tool-use prompts."""
    if not prompt_data.get("tool_definitions"):
        return None

    tools_str = json.dumps(prompt_data["tool_definitions"], indent=2)
    return (
        f"You have access to the following tools:\n{tools_str}\n\n"
        "When you want to use a tool, respond with a JSON block like:\n"
        '{"tool": "tool_name", "input": {...}}\n\n'
        "You may use multiple tools or no tools depending on the request."
    )


def generate_responses(
    prompts: list[dict],
    mode: str,
    gpu_id: int,
    output_file: Path,
    quantize: str | None = None,
) -> None:
    """Generate responses for all prompts using vLLM."""
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    model_path = MODEL_PATH
    if Path(model_path).exists():
        print(f"Using local model: {model_path}")
    else:
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        model_cached(model_path)

    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    if quantize:
        llm_kwargs["quantization"] = quantize
        llm_kwargs["load_format"] = quantize
        print(f"\nLoading model ({quantize} quantized) for {mode} on GPU {gpu_id}...")
    else:
        print(f"\nLoading model for {mode} generation on GPU {gpu_id}...")

    llm = LLM(**llm_kwargs)

    params = SamplingParams(
        temperature=GEN_PARAMS["temperature"],
        top_p=GEN_PARAMS["top_p"],
        max_tokens=GEN_PARAMS["max_tokens"],
        repetition_penalty=GEN_PARAMS["repetition_penalty"],
    )

    # Build all message lists
    print(f"Building {len(prompts)} message lists ({mode})...")
    all_messages = []
    for p in prompts:
        if mode == "prompted":
            msgs = build_messages_prompted(p)
        else:
            msgs = build_messages_unprompted(p)

        # Add tool context to user message if applicable
        tool_ctx = build_tool_context(p)
        if tool_ctx:
            msgs[-1]["content"] = tool_ctx + "\n\nUser: " + msgs[-1]["content"]

        all_messages.append(msgs)

    # Generate in batches
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_generated = 0
    start_time = time.time()

    with open(output_file, "w") as f:
        for batch_start in range(0, len(all_messages), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_messages))
            batch_msgs = all_messages[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]

            outputs = llm.chat(batch_msgs, params)

            for prompt_data, output in zip(batch_prompts, outputs):
                response_text = output.outputs[0].text.strip()
                result = {
                    "idx": prompt_data["idx"],
                    "prompt": prompt_data["prompt"],
                    "category": prompt_data["category"],
                    "mode": mode,
                    "response": response_text,
                }
                if prompt_data.get("tools_available"):
                    result["tools_available"] = prompt_data["tools_available"]
                f.write(json.dumps(result) + "\n")

            total_generated += len(batch_msgs)
            elapsed = time.time() - start_time
            rate = total_generated / elapsed if elapsed > 0 else 0
            print(f"  [{mode}] {total_generated}/{len(prompts)} "
                  f"({rate:.1f}/sec, {elapsed:.0f}s elapsed)")

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {total_generated} {mode} responses in {elapsed:.0f}s")
    print(f"Output: {output_file}")


def merge_outputs() -> None:
    """Merge prompted and unprompted responses into contrastive pairs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all partial output files
    prompted = {}
    unprompted = {}

    for f in OUTPUT_DIR.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                idx = data["idx"]
                if data["mode"] == "prompted":
                    prompted[idx] = data
                else:
                    unprompted[idx] = data

    print(f"Prompted responses: {len(prompted)}")
    print(f"Unprompted responses: {len(unprompted)}")

    # Merge into pairs
    pairs_file = DATA_DIR / "contrastive_pairs.jsonl"
    paired = 0
    missing_prompted = 0
    missing_unprompted = 0

    all_indices = sorted(set(list(prompted.keys()) + list(unprompted.keys())))

    with open(pairs_file, "w") as f:
        for idx in all_indices:
            if idx not in prompted:
                missing_prompted += 1
                continue
            if idx not in unprompted:
                missing_unprompted += 1
                continue

            p = prompted[idx]
            u = unprompted[idx]
            pair = {
                "id": f"pair_{idx:06d}",
                "idx": idx,
                "prompt": p["prompt"],
                "category": p["category"],
                "prompted_response": p["response"],
                "unprompted_response": u["response"],
            }
            if p.get("tools_available"):
                pair["tools_available"] = p["tools_available"]
            f.write(json.dumps(pair) + "\n")
            paired += 1

    print(f"\nMerged {paired} contrastive pairs → {pairs_file}")
    if missing_prompted:
        print(f"  Missing prompted: {missing_prompted}")
    if missing_unprompted:
        print(f"  Missing unprompted: {missing_unprompted}")

    # Category breakdown
    cats = defaultdict(int)
    with open(pairs_file) as fh:
        for line in fh:
            cats[json.loads(line)["category"]] += 1
    print("\nPairs by category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive pairs")
    parser.add_argument("--mode", choices=["prompted", "unprompted"],
                        help="Generation mode")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec like '0/3' for multi-GPU parallelism")
    parser.add_argument("--merge", action="store_true",
                        help="Merge prompted + unprompted into pairs")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["bitsandbytes", "awq", "gptq"],
                        help="Quantization method for smaller GPUs (e.g. bitsandbytes for INT4)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_outputs()
        return

    if not args.mode:
        parser.error("--mode is required (prompted or unprompted)")

    prompts = load_prompts(args.shard)

    shard_suffix = f"_shard{args.shard.replace('/', '_')}" if args.shard else ""
    output_file = OUTPUT_DIR / f"{args.mode}{shard_suffix}_gpu{args.gpu_id}.jsonl"

    generate_responses(prompts, args.mode, args.gpu_id, output_file, args.quantize)

    print(f"\nNext step: After all GPUs finish, run: python generate_contrastive_pairs.py --merge")


if __name__ == "__main__":
    main()
