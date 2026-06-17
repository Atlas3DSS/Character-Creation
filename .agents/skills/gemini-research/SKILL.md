---
name: gemini-research
description: Multi-turn research consultation with Gemini 3.1 Pro Preview on methodology, spectral analysis, and steering theory
disable-model-invocation: true
user-invocable: true
argument-hint: "[topic-focus] or blank for full research review"
---

# Research Consultation via Gemini 3.1 Pro Preview

Run a multi-turn research conversation with Google's Gemini 3.1 Pro Preview model.
The script sends RESEARCH_NOTES.md for deep methodological review, then
conducts follow-up turns on prompt diversity, spectral analysis, cross-architecture
transfer, and novel approaches.

## How to Run

1. Ensure the `.env` file in the project root has `GEMINI_API_KEY='...'`
2. Run the conversation script:

```bash
cd "/home/orwel/dev_genius/experiments/Character Creation"
source /home/orwel/dev_genius/qwen35_venv/bin/activate
python -u gemini_research_conversation.py 2>&1 | tee /tmp/gemini_conversation.log
```

3. Wait for completion (5-20 turns, ~10-30 minutes)
4. Output lands in `./gemini_conversation/conversation_YYYYMMDD_HHMMSS.md`

## After the Conversation

1. Read the conversation output markdown file
2. Create a summary at `gemini_conversation_summary.md` with:
   - Key findings per turn
   - Novel methodological suggestions
   - Actionable recommendations ranked by impact
   - Critical insights about the steering approach
   - Suggested next experiments
3. Update RESEARCH_NOTES.md if Gemini surfaces important new directions

## Conversation Flow

The script conducts these research topics (adaptive):

1. **Methodology critique** -- Biggest weaknesses, where we might be fooling ourselves
2. **Prompt diversity** -- Coverage gaps in 10K math + 10K sarcasm prompt sets
3. **Spectral analysis** -- SVD vs eigendecomp, shrinkage, rank considerations
4. **The "fortress" problem** -- Why 27B distributes personality uniformly
5. **Cross-architecture transfer** -- Using 8B relay circuits to find 27B components
6. **Novel approaches** -- Things we haven't considered (conceptors, D-STEER, etc.)
7. **Experimental design** -- Controls, baselines, statistical rigor

## Customizing the Consultation

To change the research focus, edit `INITIAL_PROMPT` and `FOLLOWUP_PROMPTS` in
`gemini_research_conversation.py`. The script uses adaptive follow-ups based
on what Gemini suggests.

## Key Details

- **Model**: `gemini-3.1-pro-preview`
- **API**: Google GenerativeAI SDK with `genai.GenerativeModel().start_chat()`
- **Min turns**: 5, **Max turns**: 20
- **Input**: Full RESEARCH_NOTES.md (can be large -- Gemini handles 1M+ tokens)
- **Output**: Both `.md` (human readable) and `.json` (structured)

## Gemini 3 Model Reference

Reference: https://ai.google.dev/gemini-api/docs/gemini-3

| Model ID | Use Case | Context |
|----------|----------|---------|
| `gemini-3.1-pro-preview` | Complex reasoning (OUR DEFAULT) | 1M in / 64k out |
| `gemini-3-flash-preview` | Pro intelligence at Flash speed | 1M in / 64k out |
| `gemini-3.1-flash-image-preview` | High-volume image gen | 128k in / 32k out |
| `gemini-3-pro-image-preview` | Highest quality image gen | 65k in / 32k out |

**DEPRECATED**: `gemini-3-pro-preview` — shutdown March 9, 2026. Use `gemini-3.1-pro-preview`.
**OLD**: `gemini-2.5-pro` — DO NOT USE. We use Gemini 3.1 now.

## Research Notes

The script automatically loads `./RESEARCH_NOTES.md` which contains the full
project history: model architectures, experiment results, steering findings,
connectome data, and cross-architecture comparisons.
