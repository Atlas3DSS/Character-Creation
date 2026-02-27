---
name: codex-review
description: Multi-turn code review conversation with GPT-5.3-codex focusing on bugs, performance, architecture
disable-model-invocation: true
user-invocable: true
argument-hint: "[files-to-review] or blank for all key scripts"
---

# Code Review via GPT-5.3-codex

Run a multi-turn code review conversation with OpenAI's GPT-5.3-codex model.
The script sends project code files for deep review, then conducts follow-up
turns on numerical stability, batching, hook architecture, and evaluation.

## How to Run

1. Ensure the `.env` file in the project root has `OPEN_AI='sk-...'`
2. Run the conversation script:

```bash
cd "/home/orwel/dev_genius/experiments/Character Creation"
source /home/orwel/dev_genius/qwen35_venv/bin/activate
python -u codex_research_conversation.py 2>&1 | tee /tmp/codex_conversation.log
```

3. Wait for completion (3-10 turns, ~5-15 minutes depending on response sizes)
4. Output lands in `./codex_conversation/conversation_YYYYMMDD_HHMMSS.md`

## After the Conversation

1. Read the conversation output markdown file
2. Create a summary at `codex_conversation_summary.md` with:
   - Priority-ordered bug table (P0-P10 format)
   - Critical bugs with file locations and fix snippets
   - Performance improvements with estimated speedups
   - Numerical stability issues
   - Actionable diffs in implementation order
3. If the user wants fixes implemented, apply them to the codebase

## Customizing the Review

To review different files, edit `CODE_FILES` dict in `codex_research_conversation.py`.
To change follow-up questions, edit `FOLLOWUP_PROMPTS` and `ADAPTIVE_FOLLOWUPS`.

The script auto-selects adaptive follow-ups based on what Codex reports
(e.g., if it finds numerical issues, it asks about SVD alternatives).

## Key Details

- **Model**: `gpt-5.3-codex` (NOT `gpt-5.3` -- that 404s)
- **API**: OpenAI Responses API with `previous_response_id` for multi-turn
- **Reasoning**: `{"effort": "high"}` for complex code analysis
- **Min turns**: 3, **Max turns**: 10 (adaptive based on findings)
- **Output**: Both `.md` (human readable) and `.json` (structured)

## Files to Review (defaults)

- `capture_and_steer_27b.py` -- Steering vector capture + sweep + analysis
- `fullrank_spectral_analysis.py` -- 10K-sample spectral analysis pipeline
- `generate_prompts_10k.py` -- Prompt generation for spectral analysis
