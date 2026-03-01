#!/usr/bin/env python3
"""
Send the 27B connectome analysis report to GPT-5.3 Codex for review.
Single-turn conversation — send report, get analysis back.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # Check multiple .env locations, prefer OPENAI_API_KEY over OPEN_AI
    env_paths = [
        Path("/home/orwel/dev_genius/.env"),
        Path(".env"),
        Path("/home/orwel/dev_genius/experiments/.env"),
    ]
    # First pass: look for standard OPENAI_API_KEY
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("OPENAI_API_KEY"):
                    raw_val = line.split("=", 1)[1].strip().strip("'\"")
                    if raw_val.startswith("sk-") and len(raw_val) > 50:
                        OPENAI_API_KEY = raw_val
                        print(f"Loaded API key from {env_path} (key=OPENAI_API_KEY, len={len(raw_val)})")
                        break
        if OPENAI_API_KEY:
            break
    # Second pass: fallback to OPEN_AI
    if not OPENAI_API_KEY:
        for env_path in env_paths:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    stripped = line.strip()
                    if stripped.startswith("OPEN_AI"):
                        raw_val = line.split("=", 1)[1].strip().strip("'\"")
                        if raw_val.startswith("sk-") and len(raw_val) > 50:
                            OPENAI_API_KEY = raw_val
                            print(f"Loaded API key from {env_path} (key=OPEN_AI, len={len(raw_val)})")
                            break
            if OPENAI_API_KEY:
                break

MODEL = "gpt-5.3-codex"
OUTPUT_DIR = Path("codex_conversation")

SYSTEM_PROMPT = """You are a senior ML researcher specializing in mechanistic interpretability,
activation steering, and large language model internals. You are reviewing a connectome
analysis report for the Qwen3.5-27B-Dense model.

You have deep expertise in:
1. Neuron-level analysis — hub neurons, distributed representations, z-score analysis
2. Dimensionality reduction — SVD, PCA, intrinsic dimensionality estimation
3. Activation steering — RepE, ActAdd, contrastive methods, conceptors
4. Model architecture — transformer layers, attention patterns, residual streams
5. Statistical methodology — multiple comparisons, effect sizes, clustering validity

Provide thorough, specific analysis. Be candid about what's surprising, what's concerning,
and what the practical implications are for steering research."""


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the report
    report_path = Path("reports/27b_connectome_analysis_report.md")
    if not report_path.exists():
        print(f"ERROR: Report not found at {report_path}")
        return
    report_content = report_path.read_text()
    print(f"Loaded report: {len(report_content)} chars")

    # Build prompt
    user_prompt = f"""Here is our full 27B connectome analysis report. We ran the same 6-analysis pipeline we used for the 8B (hub neurons, category overlap, layer importance, K-means clustering, SVD dimensionality, known neuron profiles) on the 27B's z-score tensor [20 categories x 64 layers x 5120 dims]. Please review the findings and give us your analysis — what stands out, what's surprising, what are the implications for steering, and any concerns about methodology or interpretation. Here's the report:

{report_content}"""

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

    # Print full response
    print(f"\nCodex response ({len(reply_text)} chars):")
    print("=" * 60)
    print(reply_text)
    print("=" * 60)

    # Save conversation log
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

    log_path = OUTPUT_DIR / f"27b_report_review_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nJSON log saved to {log_path}")

    md_path = OUTPUT_DIR / f"27b_report_review_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Codex Review of 27B Connectome Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Model: {MODEL}\n\n---\n\n")
        f.write("## Our Prompt\n\n")
        f.write(user_prompt[:500] + f"... [+{len(user_prompt)-500} chars]\n\n---\n\n")
        f.write("## Codex Response\n\n")
        f.write(reply_text)
        f.write("\n")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
