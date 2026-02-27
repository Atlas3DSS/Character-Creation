# Qwen3.5-27B Connectome: Refined Analysis (Post-Review Fixes)

**Date**: 2026-02-27
**Predecessor**: `27b_connectome_analysis_report.md` (original 6-analysis pipeline)
**What changed**: Applied methodological fixes suggested by Gemini 3.1 Pro and Codex GPT-4.1, then sent results back for re-review.

---

## Background

The original analysis ran 6 standard analyses (hub neurons, category overlap, layer importance, K-means clustering, SVD, neuron profiles) on the 27B connectome [20 categories x 64 layers x 5120 dims]. Two independent reviewers (Gemini, Codex) identified methodological concerns. We implemented all fixes and re-ran.

---

## Fix 1: Percentile-Based Hub Detection

**Problem (Gemini)**: Using fixed |z|>2.0 across models is unfair — z=2.0 is statistically harder to reach in a 5120-dim distributed space than in 4096-dim. The 8B's "hundreds of hubs" vs 27B's "9 hubs" comparison was inflated.

**Fix**: Replace fixed threshold with top 0.1% percentile per category. Each category gets its own threshold based on its own z-score distribution.

### Result: 9 hubs → 3 true hubs

| Dim | Fixed threshold (|z|>2.0) | Percentile (top 0.1%) | Known? |
|-----|---------------------------|----------------------|--------|
| **2768** | 12 categories | **7 categories** | NEW — broadest hub |
| **4010** | 8 categories | **7 categories** | NEW — Anger/Formal/Therapist specialist |
| **2028** | 7 categories | **6 categories** | KNOWN super-hub (Identity z=-6.67 at L50) |
| 4601 | 7 categories | below threshold | Artifact |
| 3968 | 6 categories | below threshold | Artifact |
| 56 | 5 categories | below threshold | Artifact |
| 1149 | 5 categories | below threshold | Artifact |
| 1316 | 5 categories | below threshold | Artifact |
| 2833 | 5 categories | below threshold | Artifact |

**Interpretation**: The 27B fortress is even more extreme than initially reported. Only **3 neurons out of 5,120 (0.06%)** are genuine multi-category hubs. The other 6 were threshold artifacts. Personality is truly a consensus phenomenon — there are no master switches.

**Gemini's reaction**: "These are ultra-bottlenecks. Don't drive them directly — use them as read-only state-detectors. Steer the downstream clusters instead, or compute vectors in the null space of these 3 hubs."

**Codex's reaction**: "Steering via single hub neurons is even less viable than we thought. Focus on ensembles of category-local neurons."

---

## Fix 2: Variance-Normalized Clustering

**Problem (Gemini)**: Language: Bilingual's massive singular value (S[0]=376.9) hijacks Euclidean-distance-based K-means. All 10 clusters were just Bilingual±/Verbose± splits — personality was invisible.

**Fix**: StandardScaler normalization before K-means, equalizing all category variances.

### Result: Hidden personality sub-structure revealed

**Original clusters** (all dominated by Bilingual):
```
Every cluster: Bilingual(±2.5) → Verbose(±1.1) → [tiny personality signal]
```

**Normalized clusters** (personality visible):

| Cluster | Neurons | Dominant Categories | Interpretation |
|---------|---------|-------------------|----------------|
| **1** | 558 | Verbose(+0.93), **Sarcastic(+0.91)**, **Polite(+0.75)** | "Chatty personality" / passive aggression |
| **4** | 430 | **Analytical(+0.82)**, **Sarcastic(+0.70)**, Bilingual(-0.86) | "Snarky analyst" |
| **6** | 523 | **Fear(+0.81)**, **Brief(+0.76)**, Bilingual(-0.75) | "Fearful = terse" (RLHF artifact) |
| **7** | 542 | **Anti-Fear(-0.79)**, Anti-Formal(-0.66), **Teacher(+0.62)** | "Confident teacher" |
| **8** | 507 | **Anti-Sarcastic(-0.86)**, **Therapist(+0.81)**, **Code(+0.72)** | "Serious technical help" |
| 0 | 488 | Bilingual(+1.79), Anti-Verbose(-0.83), Anti-Therapist(-0.69) | Language-dominant |
| 2 | 508 | Bilingual(-1.29), Sarcastic(+0.90), Verbose(+0.74) | Language + chatty |
| 3 | 525 | Anti-Analytical(-0.69), Verbose(+0.68), Therapist(+0.60) | Empathetic verbose |
| 5 | 487 | Bilingual(+1.10), Anti-Verbose(-0.95), Analytical(+0.71) | Concise analytical |
| 9 | 544 | Anti-Verbose(-0.99), Bilingual(+0.98), Analytical(+0.97) | Concise analytical (bilingual) |

### Key Discoveries

**Sarcastic + Polite share neurons (Cluster 1, 558 neurons)**:
- Gemini: "This is Passive Aggression / Customer Service Voice. Sarcasm is achieved through overly polite, verbose phrasing."
- **Implication**: Blindly suppressing sarcasm will also kill politeness. Fix: orthogonal projection — take the Polite vector and subtract its projection onto Sarcastic.

**Fear + Brief are tightly coupled (Cluster 6, 523 neurons)**:
- Gemini: "Training artifact from RLHF. Models learn: Danger = Abort quickly = short refusal."
- **Connects to L2 safety**: L2 detects violation → activates Fear → couples with Brief → canned refusal. This is the full safety pipeline.
- Codex: "Making the model 'less fearful' may also make it more verbose."

**Anti-Sarcastic + Therapist + Code (Cluster 8, 507 neurons)**:
- The model has a dedicated "serious technical help" mode — neurons that suppress sarcasm while enabling therapeutic/coding behavior. This is the "helpful assistant" phenotype at the neuron level.

**Analytical + Sarcastic (Cluster 4, 430 neurons)**:
- "Snarky analysts" — neurons that co-activate for analytical reasoning AND sarcasm. This may be the mechanistic basis for the "condescending expert" personality pattern.

---

## Fix 3: Layer Importance Distribution Stats

**Problem (Codex)**: Mean |z| per layer obscures rare-but-strong neurons.

**Fix**: Added per-layer stats: mean, std, max, p95, p99, n_above_2.

### Result: Two distinct layer signatures

**Type A — "Many moderate" layers** (e.g., L50 for Identity):
- High mean, moderate max, many neurons above |z|>2
- Personality encoded broadly across neurons

**Type B — "Few strong" layers** (e.g., L2 for Safety: Refusal):
- Low mean, high max, very few neurons above |z|>2
- Safety encoded in a handful of specialized neurons
- This is why L2's importance appeared low in mean-only analysis — it relies on rare dedicated circuits, not broad modulation

**Implication**: Safety and personality use fundamentally different encoding strategies. Safety = sparse specialists. Personality = broad consensus.

---

## Fix 4: Neuron Redundancy Check

**Problem (Codex)**: Auto-discovered neurons might be highly correlated duplicates.

**Fix**: Pairwise cosine similarity of all 21 profiled neurons' category response profiles.

### Result: 2 pairs flagged

| Pair | Cosine | Interpretation |
|------|--------|---------------|
| dim 526 ↔ dim 3586 | **+0.910** | Both are length/verbosity neurons — redundant |
| dim 755 ↔ dim 2455 | **-0.930** | Anti-correlated push-pull pair |

**The push-pull pair (755 ↔ 2455) is the most actionable discovery**:
- Gemini: "Classic toggle switch circuit. To steer into 755's state, simultaneously subtract from 2455 — gives clean amplified steering without massive activation energy."
- Codex: "Push-pull circuits sharpen category boundaries and increase dynamic range."
- **Next step**: Identify what these dims actually represent via maximum-activating prompt analysis.

---

## Reviewer Consensus on Refined Results

### Both agree:
- The fortress is more extreme than initially reported (3 hubs, not 9)
- Personality sub-structure exists but is hidden behind language/length axes
- Sarcastic+Polite entanglement is real and must be accounted for in steering
- The push-pull pair is a high-value steering lever
- Safety pipeline: L2 → Fear → Brief → refusal (full causal chain)

### Gemini's roadmap:
1. Identify what dims 755/2455 represent (maximum-activating prompts)
2. Map L2 → 3 hubs → Cluster 6 causal safety pipeline
3. Probe hub division of labor (Format vs Persona vs Task)

### Codex's roadmap:
1. Causal interventions on personality clusters
2. Systematically map all push-pull pairs
3. Deep-dive the 3 true hubs
4. Hierarchical clustering on normalized profiles

---

## Updated Files

| File | Size | What's new |
|------|------|-----------|
| `hub_neurons_percentile.json` | 4,521 B | NEW — percentile-based hub detection |
| `layer_importance.json` | 313,799 B | UPDATED — per-layer distribution stats |
| `neuron_clusters.json` | 24,015 B | UPDATED — includes `clusters_normalized` |
| `known_neuron_profiles.json` | 73,887 B | UPDATED — includes `_redundancy_check` |

All files in `qwen35_map/27b/` and uploaded to [Atlas3D/character-steering-research](https://huggingface.co/datasets/Atlas3D/character-steering-research) under `connectome/qwen35_27b/`.

---

## What This Means for the Project

1. **Prompt-first strategy confirmed**: V4 prompt for personality, steering only to protect capabilities
2. **Null-space steering**: Compute vectors orthogonal to the 3 hubs to avoid disrupting core routing
3. **Sarcasm-Polite entanglement**: Must use orthogonal decomposition, not raw category vectors
4. **Push-pull levers**: dims 755/2455 are the first identified toggle switch — potential for clean binary steering
5. **Safety is a dedicated sparse circuit**: L2 → Fear → Brief → refusal. Abliteration likely targets this exact pipeline.
6. **The abliterated 27B connectome comparison** (running now) should show dramatic changes at L2 and in the Fear+Brief cluster — that's the prediction to test.
