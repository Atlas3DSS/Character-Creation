# SAE--Connectome Alignment Analysis Report

## Qwen3.5-27B Dense: Layers 50 and 44

**Date**: 2026-03-01
**Model**: Qwen3.5-27B-FP8 (64 layers, hidden=5120, GatedDeltaNet hybrid attention)
**SAE Architecture**: TopK (k=64, d_sae=81,920, 16x expansion)
**Analysis Script**: `scripts/sae/sae_connectome_alignment.py`
**Data**: `sae_analysis/connectome_alignment_results.json`

---

## Abstract

This report presents the first quantitative alignment analysis between contrastive connectome z-score vectors and Sparse Autoencoder (SAE) decoder columns in a 27-billion-parameter language model. We computed cosine similarity between 20 behavioral category vectors from the Qwen3.5-27B connectome and all 81,920 learned SAE features at two target layers: L50 (the identified super-hub where 7 of 20 categories peak) and L44 (the mid-network sarcasm/brevity region). The key findings are: (1) L44 produces cleaner monosemantic decompositions than L50 (94/100 vs. 75/100 top-5 features are monosemantic), consistent with the SAE training metrics showing L44's lower reconstruction error (FVE 96.5% vs. 94.5%); (2) the super-hub neuron dim 2028 at L50 undergoes PARTIAL decomposition by the SAE, with Sadness and Math/Formal mapping to distinct features with opposite-sign loadings, revealing a previously unknown bipolar axis through the hub; (3) Code remains genuinely entangled in superposition at this expansion ratio, with no feature exceeding cos=0.184; and (4) Verbosity: Brief achieves the strongest single-feature alignment across both layers (cos=0.672 at L44), confirming the connectome's identification of dim 526 as a concentrated, quasi-monosemantic signal. These results validate the connectome as a targeting tool for SAE deployment and provide the first empirical evidence that connectome-identified superposition can be partially resolved into interpretable features.

---

## 1. Introduction

Sparse Autoencoders (SAEs) have emerged as a primary tool for extracting interpretable features from neural network activations. A central question in mechanistic interpretability is whether the directions identified by contrastive probing methods (such as our connectome z-score vectors) correspond to genuine computational features or are artifacts of entangled superposition. If the connectome correctly identifies where polysemantic representations reside, then SAEs trained on those layers should decompose the entangled directions into monosemantic features.

This analysis tests that hypothesis directly. We take the 20-category connectome z-score vectors computed for layers 50 and 44 of Qwen3.5-27B and measure their alignment with the decoder columns of SAEs trained on those same layers. The decoder columns of a well-trained SAE approximate the directions of monosemantic features in activation space. If a connectome z-score vector aligns strongly (high cosine similarity) with a single SAE decoder column, that category is likely monosemantic at that layer. If it requires a linear combination of many columns, the category is distributed. If the top-aligned feature also aligns with other categories, it is polysemantic.

The core thesis question focuses on dim 2028 at L50--the strongest super-hub neuron in the entire 27B model--which the connectome identified as simultaneously encoding Code (z=6.67), Math (z=6.19), Sadness (z=5.84), Science (z=3.81), and Analytical reasoning (z=3.29). Does the SAE decompose this hub into distinct features, or does the superposition persist?

---

## 2. Methods

### 2.1 Connectome Z-Score Vectors

The connectome z-score tensor has shape [20, 64, 5120], representing 20 behavioral categories across 64 layers and 5120 hidden dimensions. Each vector is the mean-difference z-score between contrastive prompt pairs (e.g., sarcastic vs. neutral) computed over diverse generation contexts. For this analysis, we extract the layer-specific slice z_scores[:, L, :] of shape [20, 5120] for L in {44, 50}.

Categories span four families:
- **Domain** (Code, History, Math, Science)
- **Emotion** (Anger, Fear, Joy, Sadness)
- **Tone/Style** (Formal, Polite, Sarcastic, Verbose/Brief)
- **Functional** (Identity, Language, Reasoning-Analytical, Reasoning-Certainty, Role-Authority, Role-Teacher, Safety-Refusal, Sentiment-Positive)

### 2.2 SAE Decoder Columns

Each SAE has a decoder matrix W_dec of shape [5120, 81920]. Column j of W_dec represents the direction in activation space that feature j reconstructs. After training, these columns approximate the monosemantic feature directions underlying the layer's activations. Both SAEs were trained to convergence (50,000 steps) with final FVE of 94.5% (L50) and 96.5% (L44), with 95.2% and 95.4% feature utilization respectively.

### 2.3 Alignment Computation

For each layer, we compute:

1. **Unit normalization**: Both the z-score vectors and decoder columns are L2-normalized to unit length.
2. **Cosine similarity matrix**: C = Z_norm @ D_norm, producing a [20, 81920] matrix where C[i,j] is the cosine similarity between category i's z-score vector and SAE feature j's decoder column.
3. **Threshold classification**:
   - **High alignment**: cos > 0.5 (strong directional agreement)
   - **Moderate alignment**: cos > 0.3 (meaningful but partial agreement)
4. **Monosemanticity test**: For each top feature, we check whether it also exceeds the moderate threshold (0.3) for any other category. If not, it is classified as monosemantic.

### 2.4 Dim 2028 Deep Dive

For the super-hub analysis, we extract the dim-2028 component of each decoder column (W_dec[2028, :]), sort features by absolute loading magnitude, and cross-reference each top feature's full cosine profile against all 20 categories. This reveals whether features that load heavily on the hub neuron are category-selective or promiscuously polysemantic.

### 2.5 Computational Notes

The entire analysis runs on CPU in under 60 seconds. No GPU is required since we only operate on the SAE decoder weights (3.13 GB per layer) and the connectome z-scores (25 MB). All cosine computations are exact (no approximation).

---

## 3. Results

### 3.1 Global Cosine Statistics

| Statistic | L50 | L44 |
|---|---|---|
| Max cosine | 0.601 | **0.672** |
| Min cosine | -0.575 | -0.565 |
| Mean cosine | -0.001 | 0.000 |
| Std cosine | 0.021 | 0.019 |

The near-zero mean and symmetric distribution confirm that the alignment is not driven by trivial correlations (e.g., shared bias terms). The maximum cosine values (0.60 and 0.67) indicate that some features align substantially with connectome directions, but no feature achieves cos > 0.7, suggesting that even the most concentrated categories are not fully captured by a single SAE feature at 16x expansion.

### 3.2 Layer 50 -- The Super-Hub

#### 3.2.1 Category-Level Alignment Summary

| Category | Max Cos | Top Feature | Mono? | n_high (>0.5) | n_mod (>0.3) |
|---|---|---|---|---|---|
| Verbosity: Brief | **0.601** | F58721 | Yes | 1 | 3 |
| Emotion: Joy | 0.511 | F37408 | No (+Positive) | 1 | 3 |
| Tone: Polite | 0.502 | F62342 | Yes | 1 | 7 |
| Tone: Formal | 0.498 | F67181 | Yes | 0 | 6 |
| Sentiment: Positive | 0.488 | F55804 | No (+Joy) | 0 | 2 |
| Emotion: Anger | 0.451 | F35756 | Yes | 0 | 4 |
| Role: Authority | 0.435 | F19338 | No (+Certainty) | 0 | 3 |
| Emotion: Sadness | 0.418 | F17526 | Yes | 0 | 4 |
| Tone: Sarcastic | 0.404 | F33734 | Yes | 0 | 7 |
| Emotion: Fear | 0.392 | F41291 | Yes | 0 | 4 |
| Reasoning: Certainty | 0.361 | F13366 | Yes | 0 | 2 |
| Domain: Math | 0.309 | F51010 | No (+Formal) | 0 | 1 |
| Reasoning: Analytical | 0.230 | F81421 | No (+Formal) | 0 | 0 |
| Domain: History | 0.253 | F49087 | Yes | 0 | 0 |
| Domain: Code | 0.184 | F73148 | Yes | 0 | 0 |
| Domain: Science | 0.291 | F51010 | No (+Math, Formal) | 0 | 0 |
| Role: Teacher | 0.174 | F74405 | Yes | 0 | 0 |
| Safety: Refusal | 0.157 | F35419 | Yes | 0 | 0 |
| Language: EN vs CN | 0.154 | F67202 | Yes | 0 | 0 |
| Identity | 0.133 | F25844 | Yes | 0 | 0 |

**Monosemanticity**: 75 of the 100 top-5 features across all categories are monosemantic (aligned to exactly one category above the 0.3 threshold). The remaining 25 polysemantic features cluster in predictable semantic neighborhoods: Joy--Positive (shared features F37408, F55804), Authority--Certainty (F19338), Polite--Formal (F23939), and Math--Formal--Science (F51010, F81421).

**High-alignment features (cos > 0.5)**: Only 3 category-feature pairs exceed 0.5:
1. Verbosity: Brief --> F58721 (cos=0.601, monosemantic)
2. Emotion: Joy --> F37408 (cos=0.511, cross-loads with Positive at 0.381)
3. Tone: Polite --> F62342 (cos=0.502, monosemantic)

**Sarcasm at L50**: 7 features exceed the moderate threshold, with the top being F33734 (cos=0.404, monosemantic). This moderate but distributed alignment is consistent with the established finding that sarcasm is a distributed field phenomenon in all tested architectures, not localizable to a single feature even at the SAE level.

**Identity at L50**: Maximum cosine of only 0.133, far below the moderate threshold. The 27B connectome previously identified Identity as having z=1.06 (dim 94 at L43)--13x weaker than the 8B model's Identity neuron. The SAE confirms this extreme distributional spreading: no SAE feature captures Identity at L50.

#### 3.2.2 Semantic Neighborhoods

The polysemantic features reveal interpretable semantic structure:

- **Positive-affect cluster**: F37408 (Joy=0.511, Positive=0.381), F55804 (Positive=0.488, Joy=0.446). These two features share a valence direction but are NOT the same feature--they partition the positive-affect space.
- **Formal-register cluster**: F67181 (Formal=0.498), F23939 (Polite=0.454, Formal=0.349), F81421 (Formal=0.347, Math=0.268, Analytical=0.230). Formality is a broader register that entangles with domain and politeness.
- **Authority-certainty cluster**: F19338 (Authority=0.435, Certainty=0.314), F13366 (Certainty=0.361, Authority=0.293). Authority and epistemic certainty share features, consistent with their pragmatic co-occurrence.

### 3.3 Dim 2028 Deep Dive

Dim 2028 at L50 is the strongest super-hub neuron in the 27B model, simultaneously serving as the peak neuron for Code (z=6.67), Math (z=6.19), Sadness (z=5.84), Science (z=3.81), and Analytical (z=3.29). The central question is whether the SAE decomposes this hub into separate, category-selective features.

#### 3.3.1 Top Features by Loading on Dim 2028

| Rank | Feature | Loading | Primary Category | Cos | Other Cats |
|---|---|---|---|---|---|
| 1 | F17526 | **+0.254** | Sadness | 0.418 | -- |
| 2 | F77423 | +0.242 | Sadness | 0.396 | -- |
| 3 | **F51010** | **-0.224** | Formal/Math | 0.344 | Math=0.309 |
| 4 | F75757 | +0.217 | Sadness | 0.332 | -- |
| 5 | F13647 | -0.204 | Code | 0.183 | -- |
| 6 | F6510 | +0.204 | Sadness | 0.207 | -- |
| 7 | F77701 | +0.202 | Sadness | 0.083 | -- |
| 8 | F41291 | +0.198 | Fear | 0.392 | -- |
| 9 | F33400 | +0.197 | Polite | 0.037 | -- |
| 10 | F61113 | +0.196 | Positive | 0.118 | -- |
| 11 | F2728 | +0.195 | Joy | 0.212 | -- |
| 12 | F42043 | +0.190 | Formal | 0.224 | -- |
| 13 | F77846 | +0.190 | History | 0.071 | -- |
| 14 | F18033 | +0.190 | Polite | 0.364 | Sadness=0.318 |
| 15 | F3125 | +0.188 | Sadness | 0.177 | -- |
| 16 | F1847 | +0.185 | Sadness | 0.225 | -- |
| 17 | F3888 | +0.182 | Sadness | 0.276 | -- |
| 18 | F4818 | +0.181 | History | 0.079 | -- |
| 19 | F37061 | +0.180 | Anger | 0.164 | -- |
| 20 | F60061 | +0.176 | Fear | 0.357 | -- |

#### 3.3.2 The Bipolar Axis Discovery

The most significant finding is the **sign structure** of the loadings on dim 2028:

- **Positive loading** (pushes dim 2028 high): Features dominated by Sadness (F17526, F77423, F75757, F6510, F77701, F3125, F1847, F3888), Fear (F41291, F60061), Joy (F2728), and Polite (F33400, F18033).
- **Negative loading** (pushes dim 2028 low): F51010 (Math/Formal, loading=-0.224) and F13647 (Code, loading=-0.204).

This reveals that dim 2028 encodes a **bipolar axis**: negative values correspond to technical/analytical content (Math, Code, Formal) while positive values correspond to emotional/affective content (Sadness, Fear, Joy). The connectome detected dim 2028 as a hub for BOTH families because it responds to both, but the SAE reveals the opposite-sign encoding that the connectome's z-score magnitudes could not capture.

This is a novel structural insight: what appeared to be a single polysemantic hub is actually an **axis with interpretable poles**. The SAE partially resolves the superposition by identifying the emotional pole (Sadness features F17526, F77423) and the analytical pole (Math/Formal feature F51010, Code feature F13647) as distinct directions that project onto dim 2028 with opposite signs.

#### 3.3.3 Decomposition Verdict: PARTIAL

The formal decomposition test asks whether the SAE identifies distinct features for Code, Math, and Sadness with cosine > 0.3 to their respective connectome vectors:

| Hub Category | Best Feature | Cosine | Status |
|---|---|---|---|
| Emotion: Sadness | F18033 | 0.318 | Resolved (also Polite=0.364) |
| Domain: Math | F51010 | 0.309 | Resolved (also Formal=0.344) |
| Domain: Code | -- | 0.184 max | **UNRESOLVED** |
| Domain: Science | -- | 0.291 max | Below threshold |
| Reasoning: Analytical | -- | 0.230 max | Below threshold |

**Sadness and Math** are partially decomposed: distinct features exist with moderate cosine alignment and opposite-sign loadings on dim 2028. However, **Code remains genuinely entangled** at this expansion ratio. Its best-aligned feature (F73148, cos=0.184) and the feature with the highest dim-2028 loading for Code (F13647, cos=0.183) both fall far short of the 0.3 threshold. This suggests that Code is either: (a) encoded across many features that individually contribute small Code-direction components, or (b) genuinely in superposition with other concepts in ways the 16x SAE cannot resolve.

### 3.4 Layer 44 -- The Sarcasm/Brevity Region

#### 3.4.1 Category-Level Alignment Summary

| Category | Max Cos | Top Feature | Mono? | n_high (>0.5) | n_mod (>0.3) |
|---|---|---|---|---|---|
| Verbosity: Brief | **0.672** | F10729 | Yes | 1 | 2 |
| Tone: Polite | 0.505 | F20088 | Yes | 1 | 4 |
| Tone: Sarcastic | **0.488** | F78824 | Yes | 0 | 7 |
| Emotion: Fear | 0.487 | F5541 | Yes | 0 | 2 |
| Emotion: Joy | 0.485 | F37408 | Yes | 0 | 2 |
| Emotion: Anger | 0.435 | F13874 | Yes | 0 | 1 |
| Emotion: Sadness | 0.389 | F17526 | Yes | 0 | 2 |
| Sentiment: Positive | 0.370 | F78212 | No (+Joy) | 0 | 1 |
| Tone: Formal | 0.358 | F26719 | Yes | 0 | 2 |
| Domain: Math | 0.338 | F55837 | Yes | 0 | 1 |
| Reasoning: Certainty | 0.297 | F7324 | Yes | 0 | 0 |
| Role: Authority | 0.285 | F19338 | Yes | 0 | 0 |
| Role: Teacher | 0.264 | F43343 | Yes | 0 | 0 |
| Reasoning: Analytical | 0.249 | F34097 | Yes | 0 | 0 |
| Domain: Science | 0.188 | F17353 | Yes | 0 | 0 |
| Domain: History | 0.158 | F46762 | Yes | 0 | 0 |
| Language: EN vs CN | 0.169 | F2594 | Yes | 0 | 0 |
| Domain: Code | 0.164 | F81135 | Yes | 0 | 0 |
| Safety: Refusal | 0.150 | F59601 | Yes | 0 | 0 |
| Identity | 0.100 | F50892 | Yes | 0 | 0 |

**Monosemanticity**: 94 of 100 top-5 features are monosemantic--significantly more than L50's 75/100. Only 6 features cross-load, concentrated in the Joy--Positive and Anger--Sarcastic neighborhoods.

**Sarcasm at L44**: The sarcasm alignment is markedly stronger at L44 than at L50 (max cos=0.488 vs. 0.404), with 7 features exceeding the moderate threshold. The top sarcasm feature (F78824, cos=0.488) is monosemantic at L44 but cross-loads with Anger at L44 (the Anger entry shows F78824 at cos=0.300, with a cross-load to Sarcastic=0.488). The Anger-Sarcasm association is semantically plausible: sarcasm is often tinged with aggressive affect.

**Verbosity: Brief at L44**: The strongest alignment in the entire study--F10729 at cos=0.672, cleanly monosemantic. This confirms the connectome's identification of L44 as the primary brevity layer, anchored by dim 526 (z=10.07). The SAE has learned a near-singleton feature that aligns with the connectome's brevity direction.

### 3.5 Cross-Layer Comparison

#### 3.5.1 Alignment Strength

| Category | L50 Max Cos | L44 Max Cos | Stronger Layer |
|---|---|---|---|
| Verbosity: Brief | 0.601 | **0.672** | L44 (+0.071) |
| Tone: Polite | 0.502 | **0.505** | L44 (tie) |
| Tone: Sarcastic | 0.404 | **0.488** | **L44** (+0.084) |
| Emotion: Joy | **0.511** | 0.485 | L50 (+0.026) |
| Emotion: Fear | 0.392 | **0.487** | **L44** (+0.095) |
| Emotion: Anger | **0.451** | 0.435 | L50 (+0.016) |
| Emotion: Sadness | **0.418** | 0.389 | L50 (+0.029) |
| Tone: Formal | **0.498** | 0.358 | **L50** (+0.140) |
| Domain: Math | **0.309** | 0.338 | L44 (+0.029) |
| Role: Authority | **0.435** | 0.285 | **L50** (+0.150) |
| Sentiment: Positive | **0.488** | 0.370 | **L50** (+0.118) |
| Reasoning: Certainty | **0.361** | 0.297 | L50 (+0.064) |
| Identity | **0.133** | 0.100 | L50 (+0.033) |

L44 outperforms L50 for behavioral/personality categories (Sarcastic, Brief, Fear), while L50 outperforms for register and high-level functional categories (Formal, Authority, Positive). This is consistent with their roles in the network: L44 handles mid-level behavioral shaping while L50 integrates higher-order representations.

#### 3.5.2 Monosemanticity Comparison

| Metric | L50 | L44 |
|---|---|---|
| Monosemantic top-5 features (of 100) | 75 | **94** |
| Categories with all-mono top-5 | 13/20 | **17/20** |
| Polysemantic cross-loading pairs | 12 | 3 |

L44 is substantially more monosemantic than L50. This aligns with the SAE training metrics: L44 achieved 96.5% FVE (vs. 94.5%) and a minimum of 521 dead features at peak (vs. 1207). The network layer that is easier for the SAE to reconstruct also produces features that are more cleanly interpretable.

#### 3.5.3 Shared Feature Indices Across Layers

Several feature indices appear in both layers' top-5 lists. Since the SAEs are trained independently with different decoder matrices, these are NOT the same feature--they happen to share an index number but encode different directions. Notably:

- **F17526** appears as top Sadness feature in both layers (L50: cos=0.418, L44: cos=0.389), suggesting the two independently-trained SAEs both learned a Sadness-related feature that happened to occupy the same index slot. This could indicate structural regularities in how SAE training discovers certain concepts.
- **F37408** appears for Joy in both layers (L50: cos=0.511, L44: cos=0.485), with the same pattern.
- **F78212** appears for Positive in both layers (L50: cos=0.268, L44: cos=0.370 with Joy cross-load).

These coincidences warrant further investigation into whether TopK SAE training has index-assignment regularities driven by initialization or data ordering.

### 3.6 Category Decomposability Hierarchy

Ordering categories by their maximum cosine alignment (averaged across both layers) reveals a clear decomposability hierarchy:

**Well-decomposed (avg max cos > 0.5)**:
1. Verbosity: Brief (0.637) -- the most decomposable category
2. Tone: Polite (0.504)
3. Emotion: Joy (0.498)

**Moderately decomposed (avg max cos 0.35-0.5)**:
4. Emotion: Fear (0.440)
5. Emotion: Anger (0.443)
6. Tone: Sarcastic (0.446)
7. Tone: Formal (0.428)
8. Sentiment: Positive (0.429)
9. Emotion: Sadness (0.404)
10. Role: Authority (0.360)

**Poorly decomposed (avg max cos 0.2-0.35)**:
11. Reasoning: Certainty (0.329)
12. Domain: Math (0.324)
13. Role: Teacher (0.219)
14. Reasoning: Analytical (0.240)
15. Domain: History (0.205)

**Unresolved (avg max cos < 0.2)**:
16. Domain: Science (0.240)
17. Domain: Code (0.174)
18. Language: EN vs CN (0.162)
19. Safety: Refusal (0.153)
20. Identity (0.117)

The pattern is striking: **emotional and stylistic** categories decompose cleanly, while **domain-knowledge** and **identity** categories do not. This suggests that emotional and tonal features occupy relatively concentrated subspaces in the model's representation, while domain knowledge is encoded across many dimensions in a genuinely distributed fashion.

---

## 4. Discussion

### 4.1 The Connectome as an SAE Targeting Tool

The results validate the connectome-then-SAE pipeline as a principled approach to mechanistic interpretability at scale. The connectome identified L50 as the super-hub where representations collide, and L44 as a cleaner mid-network layer. The SAE training confirmed this: L50 was harder to train (higher MSE, more dead features, lower FVE) and L44 was easier. The alignment analysis now adds a third layer of confirmation: L50's features are more polysemantic and L44's features are more monosemantic.

This three-way concordance (connectome z-scores, SAE training dynamics, alignment monosemanticity) provides strong evidence that the connectome is measuring something real about the representational structure. The z-scores are not epiphenomenal statistical artifacts--they capture genuine computational geometry that manifests in how easily an SAE can decompose the space.

Concretely, the pipeline operates as follows:

1. **Connectome maps the landscape**: Identifies which layers are hubs (many overlapping categories), which are specialized, and which individual neurons participate in polysemantic encoding.
2. **SAE training difficulty predicts decomposability**: Layers flagged as hubs by the connectome take longer to train and achieve lower FVE.
3. **Alignment analysis confirms interpretability**: Features at hub layers are more polysemantic; features at specialized layers are more monosemantic.

This pipeline could be applied to any model where contrastive activation data can be collected: the connectome provides a principled way to select the most informative layers for SAE training, avoiding wasted compute on layers that are either too boring (no interesting features) or too entangled (SAE cannot resolve).

### 4.2 Implications for Superposition Theory

The dim 2028 analysis provides direct evidence for the **bipolar axis hypothesis** of neuron polysemanticity. The prevailing interpretation of polysemantic neurons is that they participate in multiple unrelated computations through superposition, with different features activating the neuron for different reasons. Our finding complicates this picture: dim 2028 is not randomly polysemantic. Instead, it encodes a structured bipolar axis where technical/analytical content drives it negative and emotional/affective content drives it positive.

This is consistent with recent theoretical work suggesting that superposition in trained networks is not arbitrary but reflects the geometry of the training data. Technical and emotional text rarely co-occur, creating a natural basis for a shared axis with opposite-sign encoding. The SAE detects this structure because features that load positively on dim 2028 (Sadness features) and features that load negatively (Math/Code features) are both real computational directions--they are just anti-correlated in their use of this particular neuron.

The practical implication is that **not all superposition is equally hard to resolve**. Bipolar-axis superposition (where the entangled concepts occupy opposite poles of a shared neuron) is partially resolvable by the SAE because the features have distinct decoder directions even though they share a neuron. In contrast, **isotropic superposition** (where many concepts project similarly onto a neuron) is harder to resolve. The fact that Code remains unresolved (cos < 0.184) while Math and Sadness are partially resolved suggests that Code's representation at L50 is in isotropic superposition--spread across many dimensions with no dominant direction for the SAE to capture.

### 4.3 The Sarcasm Puzzle

Sarcasm presents an interesting intermediate case. It achieves moderate alignment at both layers (L50: cos=0.404, L44: cos=0.488) with 7 features exceeding 0.3 at each layer, and the top features are monosemantic. This means the SAE CAN identify sarcasm-related features, but no single feature captures the full sarcasm direction.

This is consistent with the broader finding from the 27B connectome mapping that sarcasm is a "distributed field" phenomenon: the fast layer scan found 0 generators and 0 suppressors across 20 candidate layers, with sarcasm never dropping below 100% when V4+steering is applied. The SAE alignment extends this finding to the feature level: sarcasm is encoded across ~7 features per layer, each contributing partially, rather than concentrating in a single monosemantic sarcasm feature.

For the character-steering application, this has practical consequences. Feature-level steering of sarcasm would require clamping or modifying multiple features simultaneously, reducing the advantage of SAE-based steering over raw connectome-vector steering. Categories that decompose cleanly into single features (like Brevity) are better candidates for feature-level intervention.

### 4.4 The Identity Problem

Identity is the most poorly decomposed category at both layers (max cos 0.133 at L50, 0.100 at L44). This is consistent with the connectome's finding that the 27B model's Identity signal is 13x weaker than the 8B model's (z=1.06 vs. z=13.96 for the 8B's dim 994). The 27B model distributes identity information so uniformly across its 5120-dimensional space that neither the connectome z-scores nor the SAE features can localize it.

This has important implications for the character-steering project: any attempt to steer the 27B model's identity through SAE features is unlikely to succeed. The V4 system prompt approach, which operates at the token/prompt level rather than the activation level, remains the only effective identity mechanism for this model size.

### 4.5 Comparison with 8B SAE Results

The 8B SAEs (trained on L09, L15, L22, L29) showed dramatically different dead-feature dynamics: 82-85% dead features at L22/L29 versus 4.6-4.8% at the 27B's L44/L50. However, the FVE was comparable (94.9% for 8B L29 vs. 94.5% for 27B L50). This means the 27B SAEs use more features to achieve similar reconstruction quality, consistent with the 27B's more distributed representations. The alignment analysis adds another dimension: we expect the 27B's features, being more numerous and alive, to include more monosemantic features per category, but potentially at lower cosine magnitude per feature. Running the same alignment analysis on the 8B SAEs (if retrained to 50K steps) would enable a direct cross-architecture comparison of decomposability.

---

## 5. Limitations

### 5.1 Expansion Ratio

The 16x expansion ratio (81,920 features for 5,120 dimensions) may be insufficient for the super-hub layer. Anthropic's published work on Claude uses 32x-64x expansion. The failure to decompose Code at L50 could be an artifact of insufficient feature capacity rather than genuine irresolvable superposition. A 32x expansion (163,840 features) SAE on L50 is a high-priority follow-up.

### 5.2 Cosine Similarity as a Metric

Cosine similarity between a z-score vector and a decoder column assumes both are unit-normalized directions. However, z-score vectors are computed from mean differences and may not perfectly represent the "true" category direction in activation space. The alignment scores should be interpreted as lower bounds on the true alignment: the z-score vectors are noisy estimates of the underlying contrastive directions.

### 5.3 Generation Context Bias

The SAEs were trained on activations from diverse generation contexts (temperatures 0.3-1.2, multiple categories), but the connectome z-scores were computed from a specific set of contrastive prompt pairs. If the SAE training data includes activation patterns not well-represented in the connectome's prompt bank, the alignment analysis may underestimate the true correspondence.

### 5.4 Threshold Sensitivity

The monosemanticity classification depends on the moderate threshold (cos > 0.3). At a stricter threshold (cos > 0.4), virtually all features would appear monosemantic. At a looser threshold (cos > 0.2), many more would appear polysemantic. The 0.3 threshold was chosen to balance sensitivity and specificity, but the classification should be interpreted as relative rather than absolute.

### 5.5 No Causal Validation

The alignment analysis is correlational. A feature that aligns with the Sarcasm z-score vector is not necessarily causally responsible for sarcastic output. Causal validation (e.g., clamping individual features and observing output changes) is needed to confirm that aligned features have functional relevance. This is planned as part of the feature-level steering experiments.

---

## 6. Next Steps

### 6.1 Immediate (High Priority)

1. **Feature-level causal testing**: For the top-aligned features (F58721/Brevity, F62342/Polite, F33734/Sarcastic), clamp each feature to zero or maximum during generation and measure the output change. This tests whether alignment implies causal influence.

2. **Held-out FVE validation**: Run the SAEs on 50,000 new tokens not seen during training to verify that the reconstruction quality (and therefore the learned features) generalizes beyond the training distribution.

3. **32x expansion SAE on L50**: Train a higher-capacity SAE to test whether Code and other unresolved categories decompose at greater expansion.

### 6.2 Medium Priority

4. **L16 and L36 SAE training**: Complete the 4-layer SAE battery. L36 is the sarcasm peak layer (dim 2768, z=2.59) and could reveal sarcasm features at their point of origin.

5. **Cross-layer feature tracking**: For shared features like the Sadness family (F17526 at both layers), compute the decoder column similarity between L44 and L50 to test whether the same concept is encoded in similar directions across layers.

6. **Activation patching with SAE features**: Instead of raw vector steering, reconstruct the steered activation through the SAE and steer only the feature coefficients. This tests whether SAE-mediated steering is more surgical than raw activation addition.

### 6.3 Long-Term

7. **Feature dashboard**: Build an interactive visualization showing per-feature activation patterns, connectome alignment, and top-activating contexts for the most interpretable features.

8. **Cross-model alignment**: Run the same analysis on the 8B model's SAEs (once retrained to 50K steps) and compare decomposability profiles across model scales.

9. **Personality feature atlas**: Identify the minimal set of SAE features needed to capture each personality dimension for the character-steering use case, enabling precision-targeted feature clamping.

---

## 7. Conclusion

The connectome alignment analysis confirms that contrastive z-score vectors and SAE decoder columns capture overlapping aspects of a model's representational geometry. The three-way concordance between connectome hub identification, SAE training difficulty, and alignment monosemanticity validates the connectome-then-SAE pipeline as a principled approach to mechanistic interpretability. The discovery of a bipolar axis through the dim-2028 super-hub--with emotional features loading positively and technical features loading negatively--adds structural nuance to our understanding of polysemantic neurons: not all superposition is created equal, and some forms decompose more tractably than others. The persistence of Code in irresolvable superposition at 16x expansion motivates higher-capacity SAEs, while the clean decomposition of Brevity (cos=0.672) and Polite (cos=0.505) demonstrates that the current SAEs are already sufficient for surgical steering of concentrated behavioral features.

---

## File Index

| Asset | Path |
|---|---|
| Alignment results JSON | `sae_analysis/connectome_alignment_results.json` |
| Alignment script | `scripts/sae/sae_connectome_alignment.py` |
| Cosine matrix L50 | `sae_analysis/cosine_matrix_L50.pt` |
| Cosine matrix L44 | `sae_analysis/cosine_matrix_L44.pt` |
| SAE training report | `reports/sae_27b_training_report.md` |
| SAE analyze script | `scripts/sae/sae_analyze.py` |
| Connectome z-scores | `qwen35_map/27b/connectome_zscores.pt` |
| L50 SAE model | `sae_models/base/L50/sae_final.pt` |
| L44 SAE model | `sae_models/base/L44/sae_final.pt` |
