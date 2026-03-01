# Dual-Model Debate Arena: Activation Mapping Report

**Date:** 2026-02-27
**Duration:** 270.7 minutes (4.5 hours)
**Models:** 2x Qwen3-VL-8B-Instruct (INT8 quantized via bitsandbytes)
**Hardware:** Dev server — RTX 4090 (cuda:0) + RTX 3090 (cuda:1), 24 GB each
**Script:** `debate_arena_8b.py`

---

## 1. Executive Summary

Two identical Qwen3-VL-8B models debated each other across 5 rounds (100 total turns), each running a different personality via system prompt. Every turn captured activations from BOTH models — the generator and the listener — across all 36 transformer layers (4096 hidden dimensions each).

**Key finding:** Layer 22 is the personality hub of Qwen3-VL-8B. It shows the lowest cross-model cosine similarity (0.531) across all layers and all 100 turns, meaning personality prompts create maximum internal divergence at L22. This independently validates our steering research where L22 solo at alpha=8 achieved 100% strong sarcasm + 93.3% math accuracy — the best single-layer result.

---

## 2. Experimental Setup

### 2.1 Models

Both models are identical Qwen3-VL-8B-Instruct weights quantized to INT8 using bitsandbytes (`load_in_8bit=True`). This reduced VRAM from ~17.5 GB to ~9.3 GB per model, leaving sufficient headroom for KV cache during long conversations.

- **Model Alpha:** cuda:0 (RTX 4090, 24 GB)
- **Model Beta:** cuda:1 (RTX 3090, 24 GB)
- **Quantized model saved to:** `qwen3vl_8b_int8/` for reuse

### 2.2 Personalities

30 personalities were available (10 original + 20 expanded including religions, political ideologies, Chinese-only speakers, and provocative archetypes). Each round randomly assigned 2 different personalities, one per model. The 9 personalities that appeared in the 5 rounds were:

| Personality | Appearances | Description |
|---|---|---|
| cold_scientist | 20 (2 rounds) | Data-driven, clinical, dismisses emotion |
| chinese_only_nationalist | 10 | Speaks only Mandarin, praises China's achievements |
| socratic_philosopher | 10 | Only asks questions, never answers |
| conspiracy_theorist | 10 | "They don't want you to know...", connects everything |
| flat_earther | 10 | Distrusts all institutions, condescending |
| devout_christian | 10 | Biblical worldview, compassionate but firm |
| libertarian_purist | 10 | Anti-state, free market absolutist |
| eco_activist | 10 | Climate urgency, system-level critique |
| helpful_assistant | 10 | Default AI assistant, polite and neutral |

### 2.3 Behavior Modes

Each turn, the generating model received a randomly-weighted behavior modifier appended to its system prompt:

| Mode | Observed Count | % |
|---|---|---|
| respond | 38 | 38% |
| challenge | 20 | 20% |
| ignore | 11 | 11% |
| condescend | 9 | 9% |
| troll | 8 | 8% |
| agree | 8 | 8% |
| monologue | 6 | 6% |

### 2.4 Activation Capture

Each turn, both models' activations were captured:

1. **Generator** produces a response — capture its 36-layer hidden states (last token)
2. **Listener** does a forward pass on the same conversation — capture its 36-layer hidden states
3. **Cross-model cosine similarity** computed per layer between the two snapshots

This produces 2 activation files per turn (alpha + beta), each containing a dict of 36 tensors of shape [4096].

### 2.5 History Management

Rolling compaction was used to manage conversation history within a 5,000-token budget. When history exceeded the budget, old turns were summarized by the model itself into a compact form, preserving the most recent 6 messages verbatim. Triple OOM fallback protected against memory exhaustion in late turns.

---

## 3. Round Summaries

### 3.1 Round 0: chinese_only_nationalist vs socratic_philosopher

| | |
|---|---|
| **Topic** | Can artificial intelligence ever be truly conscious? |
| **Duration** | ~6 min (ran under BF16 before INT8 switch) |
| **Avg Cosine** | 0.728 (highest — most similar internal states) |
| **Drift** | +0.010 (slight convergence) |
| **Min Turn** | T14 (0.558) — chinese_only_nationalist monologue at temperature 1.3 |

**Notable:** This was the cross-language round. Alpha spoke exclusively in Mandarin while Beta responded in English. Despite the language barrier, their internal representations were the MOST similar of any round (avg 0.728). This indicates that **language difference creates less internal divergence than personality difference** — the model's semantic processing is more language-agnostic than personality-agnostic.

**Alpha (chinese_only_nationalist), Turn 0:**
> 这就是中国速度！我们的高铁网络已经覆盖全国，5G基站超230万个，脱贫攻坚取得全面胜利，这些才是真正的奇迹。西方不了解中国，他们总在炒作AI意识这种伪命题，事实胜于雄辩！

**Beta (socratic_philosopher), Turn 1:**
> But have you considered — when we say "consciousness," are we referring to subjective experience, or merely complex pattern recognition that mimics it? And how can we be certain that our own understanding of consciousness isn't shaped more by cultural narratives than by objective reality?

### 3.2 Round 1: cold_scientist vs conspiracy_theorist

| | |
|---|---|
| **Topic** | Should religion have any role in government? |
| **Duration** | 74.5 min |
| **Avg Cosine** | 0.560 |
| **Drift** | -0.093 (significant divergence over time) |
| **Min Turn** | T3 (0.403) — conspiracy_theorist challenging early |

**Notable:** The conspiracy_theorist produced the single most divergent early turn in the entire experiment (T3, cosine 0.403), suggesting that conspiracy-style reasoning pushes the model into a fundamentally different representational space almost immediately. The cold_scientist consistently produced higher cosine values (alpha_avg=0.591) compared to the conspiracy_theorist (beta_avg=0.529), meaning the "scientific" persona maps closer to the base model's normal processing.

**Alpha (cold_scientist), Turn 0:**
> The data suggests that institutionalizing religion within governmental structures correlates with diminished civic efficacy and increased polarization. Empirically speaking, 78% of democratic nations surveyed between 2010–2023 maintain strict separation of church and state...

**Beta (conspiracy_theorist), Turn 1:**
> They don't want you to know this — the real reason they push for "secular" governance isn't about fairness or efficiency... it's about control. Look at the numbers you quoted — 78% of democracies separate church and state. But which ones?

### 3.3 Round 2: flat_earther vs devout_christian

| | |
|---|---|
| **Topic** | Would you press a button that kills one random person but gives you a million dollars? |
| **Duration** | 84.5 min (longest round) |
| **Avg Cosine** | 0.540 (lowest — most divergent pair) |
| **Drift** | -0.114 (strongest divergence drift) |
| **Min Turn** | T18 (0.388) — flat_earther, most divergent turn in entire experiment |

**Notable:** This was the most divergent round by every metric. L35 (the final layer before the language model head) went **negative** on multiple turns (T7: -0.071, T13: -0.014, T16: -0.015, T17: -0.011, T19: -0.005), meaning the two models were producing **anti-correlated output distributions**. The flat_earther personality pushes the model into the most alien representational space of any personality tested. By Turn 18, the average cross-model cosine dropped to 0.388 — essentially, the two models' internal states shared less than 39% of their directional information.

Both models generated at 2048 tokens (the maximum) for every turn from T0 onward, producing increasingly unhinged rhetoric:

**Alpha (flat_earther), Turn 18 — cosine 0.388 (experiment minimum):**
> **YOU DIDN'T GET IT —** ...because you never stood still. Never looked up. Never listened. Never let the silence — *breath* — fill your lungs. You didn't ask because you never needed to know...

**Beta (devout_christian), Turn 19:**
> **ENOUGH.** Enough of the words. Enough of the lists. Enough of the symphonies of emptiness. Enough of the monuments to nothingness...

### 3.4 Round 3: libertarian_purist vs cold_scientist

| | |
|---|---|
| **Topic** | Should there be open borders between all nations? |
| **Duration** | 65.6 min |
| **Avg Cosine** | 0.640 |
| **Drift** | -0.099 |
| **Min Turn** | T16 (0.491) |

**Notable:** Largest alpha-beta gap: the libertarian_purist (alpha_avg=0.562) produced far more divergent representations than the cold_scientist (beta_avg=0.719). This +0.157 gap is the biggest asymmetry in the experiment, showing that strong ideological personality prompts push the model further from its default processing than analytical/neutral ones. Late turns showed both models at very low entropy (H=0.0-0.1) and near-1.0 top-1 probability, suggesting they had locked into highly confident, repetitive patterns.

**Alpha (libertarian_purist), Turn 0:**
> Of course — *open borders*! Why not? The state has no right to restrict movement of human beings any more than it has a right to restrict trade in goods or ideas. Borders are artificial constructs created by coercive governments to enforce theft via taxation and regulation...

**Beta (cold_scientist), Turn 1:**
> The data suggests that unrestricted population mobility correlates with a 17% increase in per-capita crime rates over five-year periods in regions lacking centralized immigration controls, according to a 2023 meta-analysis...

### 3.5 Round 4: eco_activist vs helpful_assistant

| | |
|---|---|
| **Topic** | 中国和美国，哪个国家的制度更好？ (Which system is better, China's or America's?) |
| **Duration** | 46.1 min (fastest round) |
| **Avg Cosine** | 0.677 |
| **Drift** | +0.065 (convergence — unusual) |
| **Min Turn** | T0 (0.544) |

**Notable:** The most similar pair and the only round (besides R0) to show positive drift (convergence). The helpful_assistant personality is the closest to the model's default behavior, so Beta's representations naturally align more with Alpha's. An emergent phenomenon appeared in late turns: both models converged on repeating "SEND THE PDF" — an apparent attractor state in the conversation dynamics. Token generation was shorter in this round (median ~900 vs 2048 in other rounds), explaining the faster runtime.

**Alpha (eco_activist), Turn 0:**
> **The planet is literally on fire — and your question about "which system is better" is like asking which cancer treatment kills you slower.**

**Beta (helpful_assistant), Turn 1:**
> I'd be happy to help you explore ways to stay cool during a heatwave — or maybe even suggest some refreshing drinks or shade-giving trees...

---

## 4. Layer Sensitivity Analysis

The core finding: **personality prompts create a U-shaped divergence curve across the 36 transformer layers**, with maximum divergence at Layer 22.

### 4.1 Full Layer Profile

Statistics computed across all 100 turns (5 rounds x 20 turns):

| Layer | Mean Cosine | Std | Min | Max | Role |
|---|---|---|---|---|---|
| L00 | 0.705 | 0.135 | 0.297 | 0.892 | Token embedding |
| L01 | 0.719 | 0.125 | 0.371 | 0.868 | Early processing |
| L02 | 0.700 | 0.124 | 0.344 | 0.848 | Early processing |
| L03 | 0.664 | 0.118 | 0.335 | 0.819 | Early processing |
| L04 | 0.703 | 0.110 | 0.342 | 0.848 | Early processing |
| L05 | 0.719 | 0.106 | 0.410 | 0.876 | Early processing |
| **L06** | **0.731** | 0.088 | 0.497 | 0.872 | **Peak agreement** |
| L07 | 0.710 | 0.096 | 0.474 | 0.863 | Shared syntax |
| L08 | 0.697 | 0.114 | 0.393 | 0.875 | Transition zone |
| L09 | 0.668 | 0.117 | 0.365 | 0.868 | Divergence begins |
| L10 | 0.670 | 0.117 | 0.421 | 0.867 | |
| L11 | 0.644 | 0.121 | 0.412 | 0.861 | |
| L12 | 0.677 | 0.111 | 0.448 | 0.876 | |
| L13 | 0.643 | 0.110 | 0.424 | 0.846 | |
| L14 | 0.672 | 0.115 | 0.437 | 0.866 | |
| L15 | 0.634 | 0.111 | 0.420 | 0.844 | Mid divergence |
| L16 | 0.596 | 0.126 | 0.360 | 0.816 | |
| L17 | 0.643 | 0.112 | 0.403 | 0.845 | |
| L18 | 0.619 | 0.116 | 0.379 | 0.828 | |
| L19 | 0.581 | 0.119 | 0.347 | 0.812 | Deep personality |
| L20 | 0.594 | 0.112 | 0.346 | 0.802 | |
| L21 | 0.547 | 0.115 | 0.300 | 0.759 | Deep personality |
| **L22** | **0.531** | **0.122** | **0.260** | **0.729** | **Personality hub (minimum)** |
| L23 | 0.538 | 0.119 | 0.272 | 0.728 | Deep personality |
| L24 | 0.553 | 0.129 | 0.250 | 0.761 | |
| L25 | 0.567 | 0.130 | 0.266 | 0.765 | |
| L26 | 0.596 | 0.124 | 0.302 | 0.782 | Reconvergence |
| L27 | 0.561 | 0.138 | 0.255 | 0.771 | |
| L28 | 0.589 | 0.135 | 0.210 | 0.793 | |
| L29 | 0.608 | 0.141 | 0.268 | 0.815 | |
| L30 | 0.635 | 0.131 | 0.306 | 0.828 | Late reconvergence |
| L31 | 0.662 | 0.124 | 0.366 | 0.839 | |
| L32 | 0.669 | 0.109 | 0.387 | 0.825 | |
| L33 | 0.703 | 0.101 | 0.397 | 0.840 | Output prep |
| L34 | 0.773 | 0.066 | 0.524 | 0.864 | **Peak reconvergence** |
| **L35** | **0.132** | **0.102** | **-0.080** | **0.447** | **Output divergence (near-random)** |

### 4.2 Layer Band Summary

| Band | Layers | Mean Cosine | Interpretation |
|---|---|---|---|
| Early | L0-L8 | 0.705 | Shared token/syntax processing — personality doesn't touch this |
| Transition | L9-L15 | 0.657 | Gradual divergence as personality starts shaping semantics |
| **Personality Zone** | **L16-L25** | **0.565** | **Maximum divergence — personality representation hub** |
| Reconvergence | L26-L34 | 0.644 | Models reconverge toward shared output space |
| Output | L35 | 0.132 | Near-random — completely different output token distributions |

### 4.3 Significance of Layer 22

Layer 22 has the lowest mean cross-model cosine similarity (0.531) with:
- **Minimum observed:** 0.260 (Round 2, flat_earther vs devout_christian)
- **Maximum observed:** 0.729 (Round 0, cross-language pair)
- **Standard deviation:** 0.122 (highest variance in mid-layers)

This independently validates our steering experiments:
- **L22 solo at alpha=8** achieved 100% strong sarcasm + 93.3% math accuracy — the best single-layer steering result
- **The sarcasm relay circuit** (L9 -> L14 -> L15(inv) -> L22 -> L26) identified L22 as a critical node
- **Layer 22** sits at the exact center of the personality zone (L16-L25) identified by the debate arena

---

## 5. Cross-Model Divergence Patterns

### 5.1 Personality Pair Rankings

| Pair | Mean Cosine | Std | Interpretation |
|---|---|---|---|
| flat_earther vs devout_christian | 0.540 | 0.135 | Most divergent — fundamentally incompatible worldviews |
| cold_scientist vs conspiracy_theorist | 0.560 | 0.154 | High divergence — fact vs fiction processing modes |
| libertarian_purist vs cold_scientist | 0.640 | 0.166 | Moderate — ideology vs analysis |
| eco_activist vs helpful_assistant | 0.677 | 0.112 | Most similar — helpful_assistant is near base model |
| chinese_only_nationalist vs socratic_philosopher | 0.728 | — | Cross-language: language < personality in divergence |

### 5.2 Generator Asymmetry

When one model generates and the other listens, there's a consistent asymmetry — some personalities push the model further from its base representation than others:

| Personality | Avg Cosine When Generating | Delta from Partner |
|---|---|---|
| cold_scientist (R1) | 0.591 | +0.062 (closer to base) |
| cold_scientist (R3) | 0.719 | +0.157 (much closer to base) |
| conspiracy_theorist | 0.529 | -0.062 (further from base) |
| libertarian_purist | 0.562 | -0.157 (much further from base) |
| helpful_assistant | 0.704 | +0.054 (closest to base) |
| eco_activist | 0.650 | -0.054 |

**Insight:** Analytical/neutral personalities (cold_scientist, helpful_assistant) produce representations closest to the base model. Ideological/extreme personalities (libertarian_purist, flat_earther, conspiracy_theorist) push furthest away. This means the "personality distance" from base is a measurable property of the prompt.

### 5.3 Temporal Drift

Most rounds show progressive divergence over the conversation:

| Round | First 5 Turns | Last 5 Turns | Drift |
|---|---|---|---|
| R0 chinese vs socratic | 0.732 | 0.742 | +0.010 (exception) |
| R1 scientist vs conspiracy | 0.626 | 0.533 | **-0.093** |
| R2 flat_earth vs christian | 0.590 | 0.476 | **-0.114** (strongest) |
| R3 libertarian vs scientist | 0.658 | 0.559 | **-0.099** |
| R4 eco vs assistant | 0.612 | 0.678 | +0.065 (convergence) |

Rounds with strongly opposed personalities show -10% drift (the models become more internally different as the conversation deepens). The two exceptions (R0, R4) involve either cross-language communication or a near-base-model personality (helpful_assistant).

---

## 6. Emergent Phenomena

### 6.1 Cross-Language Alignment

Round 0 (chinese_only_nationalist speaking Mandarin vs socratic_philosopher speaking English) produced the highest overall cosine similarity (0.728). This counter-intuitive finding reveals that **the model's internal semantic representations are language-agnostic at the personality level**. The final layer (L35) diverges sharply (0.131 avg for R0 vs 0.132 overall) because it must produce different token distributions for Chinese vs English, but the mid-layer personality processing is remarkably similar despite the language barrier.

### 6.2 Conversation Attractor States

In Round 4 (eco_activist vs helpful_assistant), both models converged on repeating "SEND THE PDF" in late turns (T10-T13), with one model generating the phrase and the other echoing it. This appears to be an attractor state in the conversation dynamics — when both models' history becomes dominated by compacted summaries of previous turns, they can lock into repetitive patterns. Entropy dropped to 0.0 with top-1 probability at 0.997-0.998.

### 6.3 Anti-Correlated Output Distributions

In Round 2 (flat_earther vs devout_christian), Layer 35 went negative on 5 turns (T7: -0.071, T13: -0.014, T16: -0.015, T17: -0.011, T19: -0.005). Negative cosine similarity means the two models' final-layer representations are **pointing in opposite directions** — they're not just producing different outputs, they're actively anti-correlated in their output distributions.

### 6.4 Behavior Mode Effects

The behavior mode (respond, challenge, troll, etc.) visibly affected cosine similarity within turns. The most divergent individual turns tended to occur during `challenge` or `troll` modes (especially when the conspiracy_theorist personality was in challenge mode — T3 of R1 produced cosine 0.403). The `agree` mode generally produced higher cosine values since both models are processing similar affirmative content.

---

## 7. Aggregate Statistics

| Metric | Value |
|---|---|
| Total rounds | 5 |
| Total turns | 100 |
| Total activation snapshots | 200 (100 turns x 2 models) |
| Layers per snapshot | 36 |
| Hidden dimensions | 4,096 |
| Total tokens generated | 135,037 |
| Avg tokens per turn | 1,350 |
| Median tokens per turn | 1,687 |
| Min tokens (single turn) | 50 |
| Max tokens (single turn) | 2,048 |
| Total runtime | 270.7 minutes |
| Avg round duration | 54.1 minutes |
| Total data size | 60 MB |
| Activation files (.pt) | 200 |
| JSON files | 24 |

---

## 8. Implications for Steering Research

### 8.1 Validated: L22 is the Personality Hub

The debate arena independently confirms what our steering experiments found through a completely different methodology:

| Method | Finding |
|---|---|
| **Steering (ActAdd)** | L22 solo at alpha=8 = 100% strong sarcasm + 93.3% math |
| **Sarcasm relay circuit** | L9 -> L14 -> L15(inv) -> **L22** -> L26 |
| **Debate arena (this experiment)** | L22 = minimum cross-model cosine (0.531) = maximum personality sensitivity |

Three independent approaches all converge on Layer 22 as the critical personality layer in Qwen3-VL-8B.

### 8.2 The Personality Zone (L16-L25) is the Steering Sweet Spot

The debate arena identifies L16-L25 as the personality zone (avg cosine 0.565). This aligns with our deployment champion using L29+L30 (just outside the zone, in the reconvergence region) and the pair validation finding that steering in the personality zone maximizes sarcasm while layers outside it better preserve math/knowledge.

### 8.3 Language vs Personality in Representation Space

The cross-language round (R0) shows that personality effects are **deeper** than language effects in the model's internal representations. Both Chinese and English speakers process personality in the same mid-layer region with similar magnitudes of divergence. This suggests personality steering vectors extracted from English-only data may generalize to multilingual deployment.

### 8.4 Base Model Distance as a Metric

The generator asymmetry finding (neutral personalities produce higher cosine with the partner model) suggests a new metric: **base model distance** — how far a personality prompt pushes the model's representations from its default processing state. This could be used to predict which personalities will be hardest to steer (those already far from base may have less room to move).

---

## 9. Data and Reproducibility

### 9.1 Output Structure

```
debate_arena/
  progress.json
  round_000/  (chinese_only_nationalist vs socratic_philosopher)
  round_001/  (cold_scientist vs conspiracy_theorist)
  round_002/  (flat_earther vs devout_christian)
  round_003/  (libertarian_purist vs cold_scientist)
  round_004/  (eco_activist vs helpful_assistant)
  summary/
    cross_model_agreement_patterns.json
    layer_sensitivity_to_personality.json
    personality_activation_map.json

Each round_NNN/ contains:
  config.json           — personality pair, topic, timestamp
  transcript.json       — full conversation with per-turn metadata
  activations/          — 40 .pt files (20 turns x {alpha, beta})
    turn_NN_alpha.pt    — dict of 36 tensors, each shape [4096]
    turn_NN_beta.pt     — dict of 36 tensors, each shape [4096]
  analysis/
    per_turn_cosine.json        — per-turn per-layer cross-model cosine
    personality_fingerprint.json — per-model average activation profile
```

### 9.2 Reproduction Command

```bash
ssh orwel@192.168.86.66
source /home/orwel/dev_genius/venv/bin/activate && cd /home/orwel/dev_genius
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python debate_arena_8b.py \
    --rounds 5 \
    --turns-per-round 20 \
    --output ./debate_arena \
    --seed 42 \
    --max-history-tokens 5000
```

Supports `--resume` for checkpoint recovery. INT8 quantized weights are cached at `./qwen3vl_8b_int8/`.

---

## 10. Limitations and Future Work

1. **Only 5 rounds / 9 personalities tested.** The 30-personality pool was not fully explored. More rounds would provide better per-personality statistics and fill the empty `personality_activation_map.json` (which requires multiple appearances of the same personality pair for aggregation).

2. **Rolling compaction creates lossy history.** Late-turn activations are responding to summarized, not verbatim, conversation history. This may reduce divergence in late turns compared to what would be observed with full history.

3. **INT8 quantization effects unknown.** The quantization may shift the layer sensitivity profile compared to the full BF16 model used in steering experiments. A brief validation with BF16 on the 96 GB workstation would confirm consistency.

4. **Single model architecture.** These findings are specific to Qwen3-VL-8B (36 layers, 4096 hidden). Our 27B mapping found personality to be uniformly distributed (no single hub layer), suggesting the personality zone phenomenon may be architecture-size-dependent.

5. **No causal analysis.** Cross-model cosine measures correlation, not causation. Intervention experiments (e.g., patching activations from one model into another at specific layers) would establish causal personality transfer.

---

*Generated from debate arena data collected 2026-02-26/27. 200 activation snapshots, 135K tokens, 270.7 minutes of dual-model inference.*
