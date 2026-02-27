#!/usr/bin/env python3
"""
Custom AIME eval harness using vLLM.

Fast, correct, and uses proper chat template for instruct models.
Extracts answers via \boxed{}, last bare integer, or "answer is N" patterns.

Usage:
    python eval_aime.py                          # eval base model
    python eval_aime.py --model ./skippy_vectors/ablated_model  # eval ablated
    python eval_aime.py --max-tokens 8192        # longer generation
"""
import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Answer extraction — multiple strategies, most specific first
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Extract content from the LAST \\boxed{...} in text, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    # Find the opening brace
    brace_start = text.find("{", idx)
    if brace_start < 0:
        # \boxed N (no braces)
        after = text[idx + len("\\boxed"):].strip()
        m = re.match(r"(\d+)", after)
        return m.group(1) if m else None
    # Walk braces to find matching close
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
    return None


def extract_answer_is(text: str) -> str | None:
    """Match patterns like 'the answer is 42' or 'Answer: 42'."""
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*\$?\\?boxed\{?(\d+)\}?\$?",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(\d+)",
        r"[Aa]nswer\s*[=:]\s*\$?\\?boxed\{?(\d+)\}?\$?",
        r"[Aa]nswer\s*[=:]\s*(\d+)",
        r"= \s*\\boxed\{(\d+)\}",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]  # last match = most likely final answer
    return None


def extract_last_integer(text: str) -> str | None:
    """Grab the last standalone integer in the text (AIME answers are 0-999)."""
    # Look in last 500 chars to avoid grabbing random numbers from work
    tail = text[-500:]
    matches = re.findall(r"\b(\d{1,3})\b", tail)
    if matches:
        return matches[-1]
    return None


def extract_answer(response: str) -> str:
    """Try extraction strategies in order of reliability."""
    # Strategy 1: \boxed{N}
    ans = extract_boxed(response)
    if ans is not None:
        # Clean LaTeX artifacts
        ans = ans.replace(",", "").strip()
        if re.fullmatch(r"\d+", ans):
            return ans

    # Strategy 2: "the answer is N" / "Answer: N"
    ans = extract_answer_is(response)
    if ans is not None:
        return ans

    # Strategy 3: last integer in the tail
    ans = extract_last_integer(response)
    if ans is not None:
        return ans

    return "[no_answer]"


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def build_prompt(problem_text: str) -> str:
    """Build a clean instruct prompt for AIME."""
    return (
        f"Solve this math competition problem. Show your work, then give your "
        f"final answer as a single integer inside \\boxed{{}}.\n\n{problem_text}"
    )


def run_eval(
    model_path: str,
    max_tokens: int = 4096,
    gpu_mem: float = 0.85,
    max_model_len: int = 8192,
    output_dir: str = "./aime_eval_results",
) -> dict:
    """Run AIME 2024 eval and return results dict."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading AIME 2024 dataset...")
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems = [
        {"id": ds[i]["ID"], "problem": ds[i]["Problem"], "answer": str(ds[i]["Answer"])}
        for i in range(len(ds))
    ]
    print(f"  {len(problems)} problems loaded")

    # Build chat-formatted prompts
    prompts = []
    for p in problems:
        prompts.append(build_prompt(p["problem"]))

    # Load vLLM
    print(f"\nLoading model: {model_path}")
    print(f"  max_tokens={max_tokens}, max_model_len={max_model_len}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=max_tokens,
        repetition_penalty=1.0,
    )

    # Generate
    print(f"\nGenerating responses for {len(prompts)} problems...")
    t0 = time.time()

    # vLLM chat interface uses tokenizer's chat template automatically
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(conversations, sampling_params=sampling)

    gen_time = time.time() - t0
    print(f"  Done in {gen_time:.1f}s ({gen_time/len(prompts):.1f}s/problem)")

    # Score
    correct = 0
    details = []
    for i, (prob, out) in enumerate(zip(problems, outputs)):
        response = out.outputs[0].text
        num_tokens = len(out.outputs[0].token_ids)
        extracted = extract_answer(response)
        is_correct = extracted == prob["answer"]
        if is_correct:
            correct += 1

        detail = {
            "id": prob["id"],
            "target": prob["answer"],
            "extracted": extracted,
            "correct": is_correct,
            "tokens": num_tokens,
            "truncated": num_tokens >= max_tokens,
            "response": response,
        }
        details.append(detail)

        status = "OK" if is_correct else "MISS"
        trunc = " [TRUNCATED]" if detail["truncated"] else ""
        print(f"  [{status}] Problem {i}: target={prob['answer']}, "
              f"extracted={extracted}, tokens={num_tokens}{trunc}")

    accuracy = correct / len(problems) * 100

    # Summary
    print(f"\n{'='*50}")
    print(f"AIME 2024 Results: {correct}/{len(problems)} = {accuracy:.1f}%")
    print(f"  Model: {model_path}")
    print(f"  Generation time: {gen_time:.1f}s")
    truncated = sum(1 for d in details if d["truncated"])
    if truncated:
        print(f"  WARNING: {truncated} responses hit token limit ({max_tokens})")
    print(f"{'='*50}")

    # Analyze extraction failures
    misses = [d for d in details if not d["correct"]]
    if misses:
        print(f"\nMissed problems ({len(misses)}):")
        for d in misses:
            has_boxed = "\\boxed" in d["response"]
            print(f"  Problem {d['id']}: target={d['target']}, "
                  f"extracted={d['extracted']}, boxed_in_response={has_boxed}, "
                  f"truncated={d['truncated']}")

    # Save results
    results = {
        "model": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "max_tokens": max_tokens,
        "generation_time_s": round(gen_time, 1),
        "truncated_count": truncated,
        "details": details,
    }
    results_file = output_path / "aime_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AIME 2024 eval with vLLM")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model path or HF name")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max generation tokens per problem")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length (context + generation)")
    parser.add_argument("--gpu-mem", type=float, default=0.85,
                        help="vLLM GPU memory utilization fraction")
    parser.add_argument("--output-dir", default="./aime_eval_results",
                        help="Output directory for results")
    args = parser.parse_args()

    run_eval(
        model_path=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_mem=args.gpu_mem,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
