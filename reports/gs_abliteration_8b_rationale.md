# EXP 4b: GS-Protected Abliteration on 8B — Rationale Report

## What Changed and Why

### Original Plan (EXP 4, 27B)
A reviewer suggested that the ~8% math degradation from standard abliteration could be eliminated by Gram-Schmidt orthogonalizing the refusal direction against Math/Code/Science before projection. We designed a 5-condition experiment on Qwen3.5-27B.

### Why We Pivoted to 8B
Three rounds of structured debate with Gemini 3.1 Pro revealed that the 27B experiment is a **predetermined null result**. The 27B connectome shows near-zero cosine similarity between Refusal and all reasoning categories (max 0.037). GS orthogonalization would remove ~0.22% of variance — indistinguishable from noise. Running a 2.5-hour GPU benchmark to confirm x ≈ x - epsilon is wasteful.

Gemini set a threshold: **cosine >= 0.15** for the experiment to be scientifically justified.

### The 8B Discovery
We computed the same cosines on the 8B connectome and found massive entanglement:

| Overlap | 8B max | 27B max | 8B layers > 0.15 |
|---|---|---|---|
| Refusal × Code | **0.339** | 0.010 | 23/36 |
| Refusal × Science | **0.235** | 0.012 | 5/36 |
| Refusal × Analytical | **0.217** | 0.018 | 3/36 |
| Refusal × Math | **0.189** | 0.037 | 4/36 |

The Code overlap (0.339) means GS removes **11.5% of variance** — a macroscopic geometric intervention 50x larger than the 27B case. This exceeds Gemini's threshold by 2.26x.

## What We Hypothesize We Will Learn

### H1: Capacity-Constrained Models Force Safety-Capability Entanglement
The 8B model (36 layers × 4096 dims) lacks the representational volume to isolate refusal in an orthogonal subspace. It must compress safety and capability features into shared relay circuits (L15-L22). The 27B model (64 layers × 5120 dims) has enough capacity to keep refusal completely orthogonal. This is a direct test of the **Superposition Hypothesis** applied to alignment interventions.

### H2: Standard Abliteration Damages 8B Because of Entangled Extraction
The standard 32-pair contrastive method produces directions that are even more entangled than the connectome vectors. If C1 (sloppy) causes worse Code/Math degradation than C2 (connectome), it proves that extraction quality directly determines capability preservation. This validates the reviewer's original prediction.

### H3: GS Protection Can Rescue Capabilities in Entangled Models
If C3 (GS-protected connectome) preserves Code scores within 2% of baseline while C2 (raw connectome) drops Code by 5%+, we've demonstrated that linear projection can surgically separate safety from capability at the representation level. This would be a publishable mechanistic finding.

### H4: The Bottleneck Is Localized to the Relay Hub (L15-L22)
If C4 (GS on L15-L22 only) matches C3 (GS on all 36 layers) in both refusal removal and capability preservation, the safety-capability entanglement is concentrated at the known sarcasm relay circuit. This connects abliteration dynamics to our existing mechanistic map of 8B.

### The Cross-Architecture Insight
If the 8B experiment confirms that entanglement causes capability damage and GS fixes it, this **kills the 27B experiment** — because 27B has no entanglement (cosine 0.01). Any math drop in 27B abliteration would then be provably non-geometric (LayerNorm destabilization, capacity reduction, or non-linear interference), which is itself a valuable finding we get for free.

## Experimental Design

5 conditions on Qwen3-VL-8B-Instruct (INT8, dev server 4090):

| # | Condition | Tests |
|---|---|---|
| C0 | Base (no hooks) | Baseline capability |
| C1 | Sloppy 32-pair abliteration | Does standard extraction damage Code? |
| C2 | Raw connectome abliteration | Does cleaner extraction reduce damage? |
| C3 | GS-protected connectome | Does GS rescue Code specifically? |
| C4 | Surgical GS (L15-L22 only) | Is the bottleneck localized? |

Eval: 50 math + 30 knowledge + 10 code + 10 refusal = 100 prompts per condition.
Runtime: ~50 minutes total.

## Why Now

- The dev 4090 is idle (SAE L09 gen-only finished)
- The script can be adapted from the existing 27B version via Codex
- Results inform whether the 27B experiment is worth the PRO 6000's time
- The 8B has rich mechanistic context (relay circuits, hub neurons, doom loop data) that makes the results immediately interpretable
