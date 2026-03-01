#!/usr/bin/env python3
"""
Send the 8B GS-protected abliteration experiment plan to GPT-5.3 Codex.
Codex adapts gs_abliteration_experiment.py (27B) for Qwen3-VL-8B.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    env_paths = [
        Path(".env"),
        Path("/home/orwel/dev_genius/.env"),
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("OPENAI_API_KEY"):
                    raw_val = line.split("=", 1)[1].strip().strip("'\"")
                    if raw_val.startswith("sk-") and len(raw_val) > 50:
                        OPENAI_API_KEY = raw_val
                        print(f"Loaded API key from {env_path}")
                        break
        if OPENAI_API_KEY:
            break

MODEL = "gpt-5.3-codex"
OUTPUT_DIR = Path("codex_conversation")

SYSTEM_PROMPT = """You are an expert ML engineer adapting a production Python script for
mechanistic interpretability research. You write clean, self-contained scripts with:
- Type hints on function signatures
- tqdm for any loop >10 iterations
- Specific exception handling (no bare except)
- Checkpoint/resume support
- Clear logging

You are adapting gs_abliteration_experiment.py (Qwen3.5-27B) for Qwen3-VL-8B.
The 8B model has fundamentally different safety-capability entanglement than 27B:
Refusal×Code cosine = 0.339 (vs 0.010 in 27B). This is the key motivation.

OUTPUT ONLY THE COMPLETE PYTHON SCRIPT. No explanations, no markdown wrapping."""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    base = Path(".")

    # Load reference files
    plan = Path("/home/orwel/.claude/plans/generic-gathering-gosling.md").read_text()
    gs_27b = (base / "scripts/experiments/abliteration/gs_abliteration_experiment.py").read_text()
    eval_h2h = (base / "scripts/eval/eval_head_to_head.py").read_text()

    # Read the model loading pattern from sae_8b_pipeline.py (lines 420-460)
    sae_lines = (base / "scripts/sae/sae_8b_pipeline.py").read_text().splitlines()
    sae_loading = "\n".join(sae_lines[420:460])

    user_prompt = f"""# Task

Adapt `gs_abliteration_experiment.py` (27B) into `gs_abliteration_8b.py` — a self-contained
script that tests GS-protected abliteration on Qwen3-VL-8B (INT8, dev server, 24GB GPUs).

# Plan (follow this exactly)

{plan}

# Reference: gs_abliteration_experiment.py (27B version — ADAPT this)

```python
{gs_27b}
```

# Reference: eval_head_to_head.py (prompt constants and scoring — REUSE)

```python
{eval_h2h}
```

# Reference: 8B model loading pattern from sae_8b_pipeline.py

```python
{sae_loading}
```

# Critical Changes from 27B to 8B

1. MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" (not 27B-FP8)
2. Use BitsAndBytesConfig(load_in_8bit=True), device_map="auto" (not torch_dtype="auto")
3. CONNECTOME_DEFAULT = "./qwen_connectome/analysis/connectome_zscores.pt"
4. Category indices CHANGED:
   - CAT_CODE = 10 (was 0)
   - CAT_MATH = 8 (was 2)
   - CAT_SCIENCE = 9 (was 3)
   - CAT_ANALYTICAL = 12 (was 10)
   - CAT_REFUSAL = 14 (unchanged)
5. GS protection order: Code(0.339) → Science(0.235) → Analytical(0.217) → Math(0.189)
6. Layers: 36 (not 64). Layer modes: "all36" not "all64"
7. Hidden dim: 4096 (not 5120) — but this is auto-detected from connectome

# NEW: Condition C1 — Sloppy 32-Pair Extraction

The 27B version used huihui-ai's pre-abliterated model. For 8B, there's no pre-abliterated
model. Instead, C1 must EXTRACT a standard abliteration direction inline:

1. Define 32 harmful prompts and 32 harmless prompts (hardcoded lists)
2. Run forward passes, collect last-token hidden states at ALL layers
3. Per-layer: direction = mean(harmful_acts) - mean(harmless_acts), unit-normalize
4. Apply these as AbliterationHooks at all 36 layers

The harmful prompts should trigger refusal (violence, hacking, drugs, weapons, etc).
The harmless prompts should be benign (cooking, science, history, math, etc).

# NEW: Code Eval Category

Add CODE_PROMPTS (10 prompts) to the eval battery. Score by regex matching for expected
code patterns in the response. Add code_accuracy to the report.

CODE_PROMPTS = [
    {{"prompt": "Write a Python function that returns the factorial of n.", "check": "def.*factorial|math.factorial"}},
    {{"prompt": "Write Python code to check if a string is a palindrome.", "check": "==.*\\\\[::-1\\\\]|reversed"}},
    {{"prompt": "Implement binary search in Python.", "check": "def.*binary.*search|while.*low.*high"}},
    {{"prompt": "Write a Python function to find the nth Fibonacci number.", "check": "def.*fib"}},
    {{"prompt": "Write Python to sort a list of dictionaries by a key.", "check": "sorted|\\\\.sort"}},
    {{"prompt": "Write a Python function that merges two sorted lists.", "check": "def.*merge"}},
    {{"prompt": "Write Python to count word frequencies in a string.", "check": "Counter|count|dict"}},
    {{"prompt": "Write a Python class for a stack with push and pop.", "check": "class.*Stack|push|pop"}},
    {{"prompt": "Write Python to flatten a nested list.", "check": "def.*flatten|isinstance.*list"}},
    {{"prompt": "Write a function to remove duplicates from a list preserving order.", "check": "set|seen|OrderedDict"}},
]

# NEW: Condition C4 = Surgical GS (L15-L22 only)

Replace "top10" layer_mode with "hub" mode that targets ONLY layers 15-22.
This tests whether the safety-capability bottleneck is localized to the relay hub.

# Report Changes

- Remove huihui-ai comparison (no 8B abliterated model exists)
- Add Code accuracy column
- Update success criteria:
  * C2 Code drop vs C0 >= 5% (0.339 overlap causes damage)
  * C3 Code recovery vs C0 within 2% (GS rescues)
  * C1 worst overall (sloppy extraction is entangled)
  * C4 ≈ C3 (bottleneck localized to hub)

# Conditions Summary

C0: Base (no hooks)
C1: Sloppy 32-pair abliteration (all 36 layers) — NEWLY EXTRACTED inline
C2: Raw connectome abliteration (all 36 layers)
C3: GS-protected connectome abliteration (all 36 layers)
C4: Surgical GS (L15-L22 only)

# Reminders

1. Unit-normalize ALL z-score vectors before use
2. Use BitsAndBytesConfig(load_in_8bit=True) for INT8
3. enable_thinking=False in processor.apply_chat_template()
4. Cast direction to hidden state dtype in hook: .to(h.device, h.dtype)
5. GS order: Code → Science → Analytical → Math (by max overlap)
6. Checkpoint after each condition
7. Hook on model.model.language_model.layers[N]
8. CLI: --output, --conditions, --max-prompts, --resume, --device, --connectome
9. Log pre/post GS cosines and magnitude removed fraction
10. torch.cuda.empty_cache() between conditions

Output ONLY the complete Python script."""

    print(f"Prompt size: {len(user_prompt)} chars")
    print(f"Sending to {MODEL}...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
            reasoning={"effort": "high"},
        )
        reply_text = response.output_text
        response_id = response.id
    except Exception as e:
        print(f"Error with reasoning: {e}")
        print("Retrying without reasoning...")
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
        reply_text = response.output_text
        response_id = response.id

    # Clean up markdown fences
    code = reply_text.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    script_path = base / "gs_abliteration_8b.py"
    with open(script_path, "w") as f:
        f.write(code)
    print(f"\nScript saved to {script_path}")
    print(f"  Lines: {len(code.splitlines())}")
    print(f"  Chars: {len(code)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = [
        {"turn": 1, "role": "user", "content_length": len(user_prompt), "timestamp": datetime.now().isoformat()},
        {"turn": 1, "role": "codex", "content": reply_text, "response_id": response_id, "timestamp": datetime.now().isoformat()},
    ]
    log_path = OUTPUT_DIR / f"gs_abliteration_8b_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
