#!/usr/bin/env python3
"""
Send the GS-protected abliteration experiment plan to GPT-5.3 Codex for implementation.
Codex writes gs_abliteration_experiment.py, we review and run it.
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
        Path("/home/orwel/dev_genius/experiments/.env"),
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

SYSTEM_PROMPT = """You are an expert ML engineer writing production-quality Python code for
mechanistic interpretability research. You write clean, self-contained scripts with:
- Type hints on function signatures
- tqdm for any loop >10 iterations
- Specific exception handling (no bare except)
- Checkpoint/resume support
- Clear logging

You are implementing an experiment on Qwen3.5-27B-FP8 that tests whether Gram-Schmidt
orthogonalization can protect math/reasoning when removing the refusal direction (abliteration).

OUTPUT ONLY THE COMPLETE PYTHON SCRIPT. No explanations, no markdown wrapping. Just the script
from #!/usr/bin/env python3 to the last line. The user will save your output directly as a .py file."""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    base = Path(".")

    # Load the plan
    plan_path = Path("/home/orwel/.claude/plans/generic-gathering-gosling.md")
    plan = plan_path.read_text()

    # Load reference files
    eval_h2h = (base / "scripts/eval/eval_head_to_head.py").read_text()
    capture_steer = (base / "scripts/experiments/connectome/capture_and_steer_27b.py").read_text()
    ortho_sarc = (base / "scripts/experiments/abliteration/orthogonal_sarcasm_steering.py").read_text()

    user_prompt = f"""# Task

Write `gs_abliteration_experiment.py` — a self-contained script that tests whether
Gram-Schmidt protected abliteration eliminates the math cost of standard abliteration
on Qwen3.5-27B.

# Plan (follow this exactly)

{plan}

# Reference: eval_head_to_head.py (contains all test prompts and scoring functions — REUSE these)

```python
{eval_h2h}
```

# Reference: capture_and_steer_27b.py (model loading, chat template, hook pattern)

```python
{capture_steer}
```

# Reference: orthogonal_sarcasm_steering.py (Gram-Schmidt implementation)

```python
{ortho_sarc}
```

# Critical Reminders

1. Unit-normalize ALL z-score vectors before use (raw norms ~97-110)
2. Use AutoModelForImageTextToText with torch_dtype="auto" for FP8
3. enable_thinking=False in processor.apply_chat_template()
4. Cast direction to hidden state dtype in hook: .to(h.device, h.dtype)
5. GS order: Math(2) → Code(0) → Science(3) → Analytical(10)
6. Checkpoint after each condition for resume
7. Import test prompts from eval_head_to_head.py or inline them
8. Hook on model.model.language_model.layers[N]
9. CLI: --output, --conditions, --max-prompts, --resume
10. Log pre/post GS cosines and magnitude removed fraction

Output ONLY the complete Python script. No markdown, no explanation."""

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

    # Clean up — strip markdown code fences if present
    code = reply_text.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    # Save the script
    script_path = base / "gs_abliteration_experiment.py"
    with open(script_path, "w") as f:
        f.write(code)
    print(f"\nScript saved to {script_path}")
    print(f"  Lines: {len(code.splitlines())}")
    print(f"  Chars: {len(code)}")

    # Save conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = [
        {
            "turn": 1,
            "role": "user",
            "content_length": len(user_prompt),
            "timestamp": datetime.now().isoformat(),
        },
        {
            "turn": 1,
            "role": "codex",
            "content": reply_text,
            "response_id": response_id,
            "timestamp": datetime.now().isoformat(),
        },
    ]

    log_path = OUTPUT_DIR / f"gs_abliteration_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
