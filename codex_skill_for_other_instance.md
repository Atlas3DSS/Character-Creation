# Codex Skill — Portable Reference

How to talk to GPT-5.3-codex from Claude Code for code review, research queries, and implementation requests.

---

## API Setup

- **Model**: `gpt-5.3-codex` (NOT `gpt-5.3` — that 404s)
- **API**: OpenAI **Responses API** (NOT chat completions)
- **Reasoning**: `{"effort": "high"}` for complex analysis
- **Multi-turn**: Use `previous_response_id` to continue conversations
- **API key**: Stored as `OPEN_AI` in project `.env` file (NOT `OPENAI_API_KEY` — that one is expired)

### Key loading pattern

```python
from pathlib import Path
from openai import OpenAI

# Load API key — OPEN_AI in project .env is the working key
OPENAI_API_KEY = ""
env_paths = [
    Path(__file__).parent / ".env",
    Path("/home/orwel/dev_genius/.env"),
]
for env_path in env_paths:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("OPEN_AI") and not stripped.startswith("OPENAI_API_KEY"):
                raw_val = line.split("=", 1)[1].strip().strip("'\"")
                if raw_val.startswith("sk-") and len(raw_val) > 50:
                    OPENAI_API_KEY = raw_val
                    break
    if OPENAI_API_KEY:
        break

client = OpenAI(api_key=OPENAI_API_KEY)
```

---

## Single-Turn Call (send a report/code, get analysis back)

```python
MODEL = "gpt-5.3-codex"

SYSTEM_PROMPT = """You are a senior ML engineer specializing in [TOPIC].
Be specific and actionable. Provide corrected code where appropriate."""

user_prompt = f"""Here is [WHAT YOU'RE SENDING]:

{content}

Please [WHAT YOU WANT BACK]."""

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
    # Retry without reasoning parameter
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )
    reply_text = response.output_text
    response_id = response.id
```

---

## Multi-Turn Conversation

```python
# Turn 1 — initial prompt
response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,
    input=initial_prompt,
    reasoning={"effort": "high"},
)
last_response_id = response.id

# Turn 2+ — follow-ups (automatic context via previous_response_id)
response = client.responses.create(
    model=MODEL,
    instructions=SYSTEM_PROMPT,
    previous_response_id=last_response_id,
    input=[{"role": "user", "content": followup_prompt}],
    reasoning={"effort": "high"},
)
last_response_id = response.id  # chain for next turn
```

---

## Saving Output

Always save both JSON (machine-readable) and Markdown (human-readable):

```python
import json
from datetime import datetime

OUTPUT_DIR = Path("./codex_conversation")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON log
log_data = [
    {"turn": 1, "role": "user", "content": user_prompt, "timestamp": datetime.now().isoformat()},
    {"turn": 1, "role": "codex", "content": reply_text, "response_id": response_id, "timestamp": datetime.now().isoformat()},
]
with open(OUTPUT_DIR / f"topic_{timestamp}.json", "w") as f:
    json.dump(log_data, f, indent=2)

# Markdown
with open(OUTPUT_DIR / f"topic_{timestamp}.md", "w") as f:
    f.write(f"# Codex: Topic — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Model: {MODEL}\n\n---\n\n")
    f.write("## Codex Response\n\n")
    f.write(reply_text)
    f.write("\n")
```

---

## Complete Example Script (copy-paste ready)

```python
#!/usr/bin/env python3
"""Send [TOPIC] to Codex for review. Single-turn."""

import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── API Key ──
OPENAI_API_KEY = ""
env_paths = [Path(__file__).parent / ".env", Path("/home/orwel/dev_genius/.env")]
for env_path in env_paths:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if s.startswith("OPEN_AI") and not s.startswith("OPENAI_API_KEY"):
                raw = line.split("=", 1)[1].strip().strip("'\"")
                if raw.startswith("sk-") and len(raw) > 50:
                    OPENAI_API_KEY = raw
                    print(f"Loaded key from {env_path}")
                    break
    if OPENAI_API_KEY:
        break

MODEL = "gpt-5.3-codex"
OUTPUT_DIR = Path(__file__).parent / "codex_conversation"

SYSTEM_PROMPT = """You are a senior ML engineer. Be specific and actionable."""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build your prompt
    user_prompt = "YOUR PROMPT HERE"

    print(f"Prompt: {len(user_prompt)} chars → {MODEL}...")
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
        print(f"Error: {e}")
        response = client.responses.create(
            model=MODEL, instructions=SYSTEM_PROMPT, input=user_prompt,
        )
        reply_text = response.output_text
        response_id = response.id

    print(f"Response: {len(reply_text)} chars")
    print(reply_text[:2000])

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = [
        {"turn": 1, "role": "user", "content": user_prompt, "timestamp": datetime.now().isoformat()},
        {"turn": 1, "role": "codex", "content": reply_text, "response_id": response_id, "timestamp": datetime.now().isoformat()},
    ]
    with open(OUTPUT_DIR / f"topic_{ts}.json", "w") as f:
        json.dump(log, f, indent=2)
    with open(OUTPUT_DIR / f"topic_{ts}.md", "w") as f:
        f.write(f"# Codex: Topic — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL}\n\n---\n\n{reply_text}\n")
    print(f"Saved to codex_conversation/topic_{ts}.*")


if __name__ == "__main__":
    main()
```

---

## Environment

- **Venv**: `source /home/orwel/dev_genius/qwen35_venv/bin/activate`
- **Deps**: `pip install openai` (already installed)
- **Working dir**: `/home/orwel/dev_genius/experiments/Character Creation/`
- **Output dir**: `./codex_conversation/`

## Important Notes

- The `OPENAI_API_KEY` in `/home/orwel/dev_genius/.env` is **EXPIRED** — do not use it
- The `OPEN_AI` key in the project `.env` works — always load that one first
- Always try with `reasoning={"effort": "high"}` first, fall back without it
- For multi-turn, the `previous_response_id` carries full conversation context automatically
- Typical response time: 1-5 min for single-turn with high reasoning on large prompts
- Max ~10 turns before context gets unwieldy
