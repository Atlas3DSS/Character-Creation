#!/usr/bin/env python3
"""
One-shot: Send the 27B connectome analysis report to Gemini for review.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY"):
                GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("'\"")
                break

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Set it via environment variable or in .env file."
    )

MODEL = "gemini-3.1-pro-preview"
genai.configure(api_key=GEMINI_API_KEY)

# ── Load report ─────────────────────────────────────────────────────
report_path = Path("reports/27b_connectome_analysis_report.md")
if not report_path.exists():
    raise FileNotFoundError(f"Report not found at {report_path}")

report_content = report_path.read_text()
print(f"Loaded report: {len(report_content)} chars")

# ── Build prompt ────────────────────────────────────────────────────
prompt = f"""Here is our full 27B connectome analysis report. We ran the same 6-analysis pipeline we used for the 8B (hub neurons, category overlap, layer importance, K-means clustering, SVD dimensionality, known neuron profiles) on the 27B's z-score tensor [20 categories x 64 layers x 5120 dims]. Please review the findings and give us your analysis — what stands out, what's surprising, what are the implications for steering, and any concerns about methodology or interpretation. Here's the report:

{report_content}"""

# ── Send to Gemini ──────────────────────────────────────────────────
print(f"Sending to {MODEL}...")
print(f"Prompt length: {len(prompt)} chars")

model = genai.GenerativeModel(MODEL)
response = model.generate_content(prompt)
reply_text = response.text

# ── Output ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GEMINI RESPONSE")
print("=" * 80)
print(reply_text)

# Save to file
output_dir = Path("gemini_conversation")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = output_dir / f"27b_review_{timestamp}.md"
with open(out_path, "w") as f:
    f.write(f"# Gemini Review of 27B Connectome Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"Model: {MODEL}\n\n---\n\n")
    f.write("## Prompt\n\n")
    f.write(prompt[:500] + f"\n\n... [full report included, {len(report_content)} chars]\n\n---\n\n")
    f.write("## Gemini Response\n\n")
    f.write(reply_text)
    f.write("\n")

print(f"\nSaved to {out_path}")
