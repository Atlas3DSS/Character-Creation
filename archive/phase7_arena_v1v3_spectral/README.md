# Phase 7 (Superseded): Debate Arena v1-v3 & GMR Spectral (Feb 26-27, 2026)

Earlier versions of the debate arena and the initial GMR spectral analysis.
All superseded by debate_arena_v4.py and fullrank_spectral_analysis.py.

## Why Superseded
- `debate_arena_8b.py` (v1): Basic dual-model setup, no personality diversity
- `debate_arena_v2.py`: Added logit capture, 10 personalities
- `debate_arena_v3.py`: Added 30 personalities, behavior modes — but had doom loop issues
- `gmr_spectral.py`: Rank-200 covariance (insufficient for 5120-dim space), replaced by fullrank

## Active Replacements (in project root)
- `debate_arena_v4.py` — Latest arena with Codex bug fixes, doom loop detector integration
- `fullrank_spectral_analysis.py` — Full-rank spectral analysis with 10K samples, SVD, Ledoit-Wolf

## Scripts (4 files)
- `debate_arena_8b.py` — Arena v1
- `debate_arena_v2.py` — Arena v2 (logit capture)
- `debate_arena_v3.py` — Arena v3 (30 personalities, behavior modes)
- `gmr_spectral.py` — GMR Phase 1 spectral analysis (rank-200)

## Data Directories
- `debate_arena/` — Arena v1-v4 round data, activations, transcripts

## RESEARCH_NOTES.md References
- "Debate Arena" section
- "GMR Spectral Analysis" section
- See also: `debate_arena_report.md`, `gmr_spectral_report.md`
- Commits: `b139c86` (GMR + arena v2), `10967c6` (arena + research synthesis)
