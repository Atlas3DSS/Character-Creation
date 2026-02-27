#!/usr/bin/env python3
"""
Send adviser's technical review of basin_engineering_lora.py to Codex for corrections.
Uses same infrastructure as send_report_to_codex.py (gpt-5.3-codex, Responses API).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
OPENAI_API_KEY = ""
# Priority: project .env OPEN_AI key (known working), then OPENAI_API_KEY
env_paths = [
    Path(__file__).parent / ".env",
    Path("/home/orwel/dev_genius/.env"),
    Path("/home/orwel/dev_genius/experiments/.env"),
]
# First: check OPEN_AI key (project-specific, known working)
for env_path in env_paths:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("OPEN_AI") and not stripped.startswith("OPENAI_API_KEY"):
                raw_val = line.split("=", 1)[1].strip().strip("'\"")
                if raw_val.startswith("sk-") and len(raw_val) > 50:
                    OPENAI_API_KEY = raw_val
                    print(f"Loaded API key from {env_path} (key=OPEN_AI, len={len(raw_val)})")
                    break
    if OPENAI_API_KEY:
        break
# Fallback: OPENAI_API_KEY or env var
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("OPENAI_API_KEY"):
                    raw_val = line.split("=", 1)[1].strip().strip("'\"")
                    if raw_val.startswith("sk-") and len(raw_val) > 50:
                        OPENAI_API_KEY = raw_val
                        print(f"Loaded API key from {env_path} (key=OPENAI_API_KEY)")
                        break
        if OPENAI_API_KEY:
            break

MODEL = "gpt-5.3-codex"
OUTPUT_DIR = Path(__file__).parent / "codex_conversation"

SYSTEM_PROMPT = """You are a senior ML engineer specializing in LoRA fine-tuning, activation steering,
and PyTorch training pipelines for large language models. You previously wrote a basin_engineering_lora.py
script for Qwen3.5-27B-FP8. An external technical adviser has reviewed your code and found critical bugs.

Your task: provide a COMPLETE, CORRECTED version of basin_engineering_lora.py that fixes ALL issues
identified by the adviser. The script must be drop-in ready for the following environment:

- Model: Qwen/Qwen3.5-27B-FP8
- Venv: /home/orwel/dev_genius/qwen35_venv/ (torch 2.9.1+cu128, transformers 5.3.0.dev0, peft)
- Model layer access: model.model.language_model.layers[N]
- Connectome: ./qwen35_map/27b/connectome_zscores.pt shape [20, 64, 5120]
- Categories (index order): Sarcastic=0, Polite=1, Analytical=2, Enthusiastic=3, Empathetic=4,
  Formal=5, Casual=6, Authoritative=7, Humorous=8, Philosophical=9, Code=10, Math=11,
  Science=12, Brief=13, Verbose=14, Identity=15, Surprise=16, Sadness=17, Fear=18, Anger=19
- Chat template: processor.apply_chat_template(msgs, enable_thinking=False)
- GPU: RTX PRO 6000 96GB VRAM
- LoRA target layers: 48-55

Provide the COMPLETE script — not patches. Every function, every import, ready to run."""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the original script from the previous Codex conversation
    prev_md_path = OUTPUT_DIR / "basin_engineering_20260227.md"
    if prev_md_path.exists():
        prev_content = prev_md_path.read_text()
        # Extract just the python code block
        code_start = prev_content.find("```python\n#!/usr/bin/env python3\n# basin_engineering_lora.py")
        code_end = prev_content.find("```\n\n---\n\n## 5. RECOMMENDATIONS")
        if code_start > 0 and code_end > code_start:
            original_code = prev_content[code_start + len("```python\n"):code_end]
        else:
            original_code = "[Could not extract code from previous conversation]"
    else:
        original_code = "[Previous conversation file not found]"

    print(f"Extracted original code: {len(original_code)} chars")

    # Build the adviser review prompt
    user_prompt = f"""Here is the basin_engineering_lora.py script you previously wrote:

```python
{original_code}
```

An external technical adviser has thoroughly reviewed this code and found the following issues.
Please provide a COMPLETE corrected script that fixes ALL of them.

## CRITICAL BUGS (will break execution)

### Bug 1: L50 hook attached to wrong module
The code searches for a module containing `layers.50.` AND either `temporal_block` or `self_attn`.
This matches a SUB-MODULE of layer 50 (the attention block), not the layer output itself.

**Problem**: We need the hidden state AFTER the full layer computation (residual + attention + MLP),
not the intermediate attention output. The SVD basis vectors were computed on layer-output hidden states
from the connectome extraction, so hooking the wrong representation space makes L_svd meaningless.

**Fix**: Hook `model.model.language_model.layers[50]` directly. The layer's forward returns a tuple
where element [0] is the hidden state after the full residual + attention + MLP computation.

### Bug 2: Labels/input_ids alignment broken in Phase 3
The training loop tokenizes prompt and target separately, then passes `target_tokens` as `labels`
to a forward pass that only received `input_ids` from the prompt. For causal LM training:
- Must concatenate prompt + target into a SINGLE sequence
- Create labels that mask out the prompt tokens (set to -100)
- Pass the full concatenated sequence as both input_ids and labels
As written, shapes won't match → dimension mismatch error.

### Bug 3: Batch variable handling wrong for batch_size=1
The DataLoader with default collation returns a dict of lists when batch_size=1, but the training loop
accesses `batch.get("prompt")` expecting a single string. With default collation, `batch["prompt"]`
will be a LIST of length 1, not a string. Need `batch["prompt"][0]` or use a custom collate function
that returns single items.

### Bug 4: do_sample=False with temperature=0.7 in Phase 4 eval
These are contradictory — `do_sample=False` means greedy decoding, which ignores temperature entirely.
Fix: Use `do_sample=True, temperature=0.7` for stochastic eval, OR `do_sample=False` without temperature
for deterministic eval.

### Bug 5: Math evaluation checks for "42" placeholder
Phase 4 math eval checks `if "42" in output_text` for all math prompts. Must implement proper answer
extraction with regex-based number parsing (the follow-up code had `extract_number()` — use that).

## ARCHITECTURAL CONCERNS

### Issue 6: SVD basis extraction uses WRONG category index
`load_sarcasm_basis()` uses `zscores[18, 48:56, :]` — index 18 is FEAR, not Sarcastic.
Our categories are ordered: Sarcastic=0, Polite=1, Analytical=2, ..., Fear=18, Anger=19.
**Fix**: Use `zscores[0, 48:56, :]` for Sarcastic category.

### Issue 7: Only 5 disentanglement prompts — need 50-100
The follow-up code generates only 5 disentanglement prompts. This is far too few to get a usable
gradient signal from L_svd on disentanglement data. Need at least 50-100 clean disentanglement pairs.

### Issue 8: L_harden VRAM with create_graph=True
`torch.autograd.grad(math_logit, l50_acts, create_graph=True)` doubles VRAM for that step.
With 96GB GPU this is likely fine, but add a VRAM check before the first L_harden computation
and print a warning if free VRAM < 20GB. Also, this depends on Bug #1 being fixed first —
gradients with respect to the wrong activations are useless.

## REQUIREMENTS FOR CORRECTED SCRIPT

1. Fix ALL 8 issues above
2. Complete, self-contained script — all 4 phases
3. Phase 1 data generation should use the model itself to generate real sarcastic/math/disentangle outputs
   (not template strings). Use V4 system prompt for sarcastic targets.
4. Proper causal LM label alignment in Phase 3
5. Correct category index (0 = Sarcastic)
6. At least 50 disentanglement prompts in Phase 1
7. Proper math answer extraction in Phase 4
8. VRAM monitoring for L_harden
9. All hooks on the correct layer module
10. Deterministic eval (do_sample=False, no temperature)

Please provide the COMPLETE corrected basin_engineering_lora.py script."""

    print(f"Prompt size: {len(user_prompt)} chars")
    print(f"Sending to {MODEL}...")
    print("=" * 60)

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
        print("Retrying without reasoning parameter...")
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
            )
            reply_text = response.output_text
            response_id = response.id
        except Exception as e2:
            print(f"Error (retry): {e2}")
            return

    print(f"\nCodex response ({len(reply_text)} chars):")
    print("=" * 60)
    print(reply_text[:2000])
    if len(reply_text) > 2000:
        print(f"\n... [+{len(reply_text) - 2000} more chars]")
    print("=" * 60)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = [
        {
            "turn": 1,
            "role": "user",
            "content": user_prompt,
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

    json_path = OUTPUT_DIR / f"basin_engineering_adviser_fixes_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nJSON log saved to {json_path}")

    md_path = OUTPUT_DIR / f"basin_engineering_adviser_fixes_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Basin Engineering — Adviser Fixes via Codex\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Model**: {MODEL}\n\n---\n\n")
        f.write("## Adviser Review Sent\n\n")
        f.write("8 issues identified: 5 critical bugs + 3 architectural concerns.\n")
        f.write("See full prompt in JSON log.\n\n---\n\n")
        f.write("## Codex Corrected Response\n\n")
        f.write(reply_text)
        f.write("\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
