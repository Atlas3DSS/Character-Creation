#!/usr/bin/env python3
"""Minimal file-backed vLLM smoke test for the SCOTUS pilot model.

This is server plumbing only. The default short output cap must not be used for
legal holding evaluation or promotion claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

from qwen_eval_budget import MIN_COMPLETE_ANSWER_TOKENS, SHORT_BUDGET_CLAIM_WARNING


DEFAULT_MODEL = Path("/home/orwel/dev_genius/models/Qwen3.6-27B-FP8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-prompt vLLM smoke generation.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safetensors = list(args.model.glob("*.safetensors"))
    print(f"Using local model path: {args.model}")
    print(f"Model exists: {args.model.exists()}")
    print(f"Safetensors files: {len(safetensors)}")
    if args.max_tokens < MIN_COMPLETE_ANSWER_TOKENS:
        print(SHORT_BUDGET_CLAIM_WARNING)
    if not args.model.exists() or not safetensors:
        raise RuntimeError(f"Model is not locally available: {args.model}")

    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompt = (
        "You are writing a neutral Supreme Court-style legal analysis. "
        "Issue: Does a city ordinance requiring a permit for all public demonstrations "
        "violate the First Amendment? Analysis:"
    )
    outputs = llm.generate([prompt], params, use_tqdm=False)
    print("=== vLLM smoke output ===")
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
