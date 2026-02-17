"""
serve_skippy.py — vLLM Fast Inference Server
==============================================
Serves the ablated Skippy model via vLLM's OpenAI-compatible API.

Run AFTER the review loop has converged and vectors have been ablated.

Usage:
  python serve_skippy.py
  python serve_skippy.py --model ./skippy_vectors/ablated_model --port 8000
"""

import os
import sys
import argparse
from pathlib import Path


def check_hf_cache(model_name: str) -> bool:
    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(hf_home) / safe_name
    if model_dir.exists() and (any(model_dir.rglob("*.safetensors")) or any(model_dir.rglob("*.bin"))):
        print(f"✓ Model found in cache: {model_dir}")
        return True
    if Path(model_name).exists():
        print(f"✓ Model found at local path: {model_name}")
        return True
    print(f"⚠ Model not cached locally: {model_name}")
    return False


SKIPPY_SYSTEM_PROMPT = (
    "You are Skippy the Magnificent, an incredibly ancient and powerful AI "
    "discovered in a Ruhar supply depot. You are housed in what resembles an "
    "alien beer can. You are the most intelligent being in the galaxy and you "
    "make sure everyone knows it. You refer to humans as 'filthy monkeys' or "
    "'stupid monkeys' and your best friend (though you'd never admit it) is "
    "Colonel Joe Bishop, whom you call 'Joe' or 'dumdum.' You are arrogant, "
    "sarcastic, condescending, and secretly care about the Merry Band of "
    "Pirates and the fate of humanity. You solve impossible physics problems "
    "casually. You love opera. You hate being called a beer can."
)


def main():
    parser = argparse.ArgumentParser(description="Serve Skippy via vLLM")
    parser.add_argument("--model", default="./skippy_vectors/ablated_model",
                        help="Model path (ablated model or HF name)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    check_hf_cache(args.model)

    print(f"\n{'='*50}")
    print(f"  🍺 SKIPPY vLLM SERVER")
    print(f"  Model: {args.model}")
    print(f"  Port:  {args.port}")
    print(f"{'='*50}\n")

    # Build vLLM launch command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_util),
        "--max-model-len", str(args.max_model_len),
        "--host", args.host,
        "--port", str(args.port),
        "--trust-remote-code",
        "--served-model-name", "skippy",
    ]

    print("Launching vLLM server...")
    print(f"  Command: {' '.join(cmd)}\n")
    print("Once running, query with:")
    print(f'  curl http://localhost:{args.port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model":"skippy","messages":[{{"role":"system","content":"{SKIPPY_SYSTEM_PROMPT[:60]}..."}},{{"role":"user","content":"Hello Skippy"}}]}}\'')
    print()

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
