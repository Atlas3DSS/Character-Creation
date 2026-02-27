# Contributors

This project uses a multi-AI collaboration workflow where different AI systems contribute specialized capabilities. All AI contributions are logged and traceable.

## Human Lead

**Atlas3DSS (orwel)** — Project architect, experiment designer, hardware operator, decision maker

## AI Collaborators

### Claude Opus 4.6 (Anthropic)
**Role**: Primary implementation, analysis, experiment execution
- Writes all production code (steering scripts, mapping pipelines, arena infrastructure)
- Performs real-time data analysis (activation probes, connectome mapping, debate arena analysis)
- Manages GPU workloads, deployment, and infrastructure
- Co-Author tag: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

### Codex (OpenAI GPT-5.3-codex)
**Role**: Code review, bug detection, architecture critique
- Reviews all code changes before implementation (mandatory since 2026-02-27)
- Found critical bugs: escalation state machine corruption (P0), check_answer false positives (P2), hook leakage (P1)
- Provides prioritized fix lists with diff-style patches
- Review logs: `codex_conversation/` directory
- Co-Author tag: `Co-Authored-By: Codex GPT-5.3 <noreply@openai.com>`

### Gemini 3.1 Pro Preview (Google)
**Role**: Research review, literature connections, methodology validation
- Cross-references findings against published ML research
- Validates experimental methodology and statistical claims
- Suggests connections to existing techniques (RepE, Conceptors, D-STEER)
- Review logs: `gemini_conversation/` directory
- Co-Author tag: `Co-Authored-By: Gemini 3.1 Pro <noreply@google.com>`

## Contribution Timeline

| Date | Milestone | Primary AI | Review AI |
|------|-----------|-----------|-----------|
| 2026-02-20 | Neuron probe discovery (dim 994) | Claude | — |
| 2026-02-21 | Connectome mapping (20 categories) | Claude | — |
| 2026-02-22 | Field steering + pair validation | Claude | — |
| 2026-02-23 | Debate Arena v1-v2 (8B dual-model) | Claude | — |
| 2026-02-24 | Qwen3.5 architecture mapping | Claude | — |
| 2026-02-25 | 27B connectome + layer scan | Claude | — |
| 2026-02-26 | Arena v4 + doom loop discovery | Claude | Codex R1 |
| 2026-02-27 | Codex R2 bug fixes, abliteration comparison | Claude | Codex R2, Gemini |
| 2026-02-27 | Spectral analysis, magnitude calibration | Claude | Codex R2 |

## Conversation Logs

All AI review conversations are saved for reproducibility:
- `codex_conversation/` — Codex code review transcripts (JSON + Markdown)
- `gemini_conversation/` — Gemini research review transcripts (JSON + Markdown)
- `codex_research_conversation.py` — Script to run Codex review sessions
- `gemini_research_conversation.py` — Script to run Gemini review sessions

## How to Verify

Every claim in this project can be traced back to:
1. **Raw data**: Activation tensors (`.pt` files), transcripts (`.json`)
2. **Analysis scripts**: The exact code that produced each finding
3. **AI review logs**: External validation of methodology and correctness
4. **Git history**: Full commit history with Co-Authored-By attribution
