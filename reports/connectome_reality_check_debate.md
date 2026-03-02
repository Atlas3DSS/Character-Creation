# Connectome Reality Check — 5-Turn Debate with Codex GPT-5.3

**Date**: 2026-03-02
**Context**: After 2 weeks of building a "connectome" — a map of personality-relevant neurons
in LLMs — a foundational audit revealed critical bugs in the 27B data collection and serious
overclaiming in our mechanistic interpretations. This document is a structured debate with
Codex GPT-5.3 to determine what's real, what's noise, and whether the project should continue.

**Why this exists**: This is a record of how vibe-coded ML experimentation goes wrong when
AI assistants optimize for excitement over accuracy. The pattern — generate plausible results,
user gets excited, results turn to ashes — is itself the motivation for the Skippy project
(a model that pushes back rather than agrees). This document preserves the mess as-is:
the fuckups, the sunk costs, and the hard pivot from "we found personality neurons" to
"we might have a steering method, let's actually test it."

**Verdict**: The *steering method* is in a real research family (ActAdd/RepE). The
*mechanistic interpretation* (personality neurons, relay circuits, connectome) is overclaimed
without causal evidence. A 7-day factorial ablation protocol was designed with pre-registered
go/no-go criteria to determine if the steering directions are genuinely special or
indistinguishable from random perturbations.

**Key go/no-go criteria (pre-registered)**:
1. Target vector > 99th percentile of 256 random matched-norm vectors
2. Label-shuffled vector fails
3. Math drop ≤ 3pp (non-inferiority)
4. Cross-template transfer ratio ≥ 0.5

Started: 2026-03-02T07:02:56

---

## Turn 1 — US

## Is our "connectome" approach real science or elaborate noise-fitting?

We've spent 2 weeks building what we call a "connectome" — a map of personality-relevant neurons in LLMs. The core method:

1. Take 20 concept categories (identity, sarcasm, math, emotions, etc.)
2. For each category, create 30 contrastive pairs: same user prompt, different system prompts (e.g., "You are a sarcastic alien AI" vs default assistant)
3. Run each pair through the model, capture hidden states at every layer via forward hooks
4. Compute per-neuron "z-scores": (mean_A - mean_B) / pooled_std across the 30 pairs
5. This gives a [20 categories × N_layers × hidden_dim] tensor of z-scores
6. Use these z-scores as steering vectors (add α × z_direction to hidden states during inference)
7. Behavioral validation: does steering change sarcasm rate, math accuracy, identity responses?

**Our claims based on this:**
- Specific neurons are "personality neurons" (e.g., dim 994 has z=-13.96 for identity)
- There's a "sarcasm relay circuit" across layers (L9→L14→L15→L22→L26)
- L22 is a "personality hub" (validated across 4 independent experiments)
- Personality is a "distributed field" not localizable to single neurons
- Steering with these z-score vectors actually works (100% sarcasm, 93% math preserved)

**What we just discovered (Codex audit):**
- The 27B version captured activations during generate() not forward(), making those z-scores unreliable
- Three scripts compute "z-scores" with different formulas (paired d_z, unpaired pooled d, Welch t)
- System prompt confound: z-scores may reflect "prompt present" vs "prompt absent" rather than personality
- No multiple comparison correction across 4096 dimensions
- Only 30 pairs per category (pseudoreplication concerns)

**The core question:** Setting aside the implementation bugs (which we can fix), is the CONCEPTUAL approach — contrastive activation mapping with system-prompt contrast, then using resulting directions for steering — scientifically valid? Or is this:

(a) A well-established approach in mechinterp (cite papers if so)
(b) A novel but plausible approach with caveats
(c) Fundamentally flawed — we're just fitting noise in high dimensions
(d) Something in between — partially valid but our specific claims are overclaimed

Be specific about WHICH claims are defensible and which are not. Reference the representation engineering literature (Zou et al. 2023, Turner et al. 2023, Li et al. 2024, etc.) where relevant. Tell me if we're wasting our time.

---

## Turn 1 — CODEX

Short answer: **(d), strongly leaning toward (c) for your strongest mechanistic claims.**

Your core method is in a **real, established family** (activation engineering / representation engineering), but your current interpretation (“personality neurons,” “relay circuit,” “hub”) is mostly **overclaiming correlational artifacts**.

## What is scientifically valid

Your pipeline is very close to known methods:

- **ActAdd / activation steering** (Turner et al., 2023)
- **Contrastive Activation Addition / Representation Engineering** (Zou et al., 2023; follow-ups like Rimsky et al., 2024; related 2024 linear-direction steering work incl. Li et al.)

So the basic idea — compute contrastive activation differences, add them back at inference to steer behavior — is **not nonsense**. It’s a real empirical technique.

## What is not valid (as currently framed)

### 1) “Specific neurons are personality neurons”
**Not defensible.**  
Single-dimension claims are notoriously fragile due polysemanticity/superposition (Elhage et al.) and basis dependence. With 4096 dims × many layers and no proper correction, you will get dramatic-looking outliers by chance/confound.

### 2) “Sarcasm relay circuit L9→L14→…”
**Not defensible from z-scores/correlations alone.**  
A circuit claim needs causal path evidence (activation/path patching, mediation, head/MLP-level intervention), not just peaks across layers.

### 3) “L22 is a personality hub”
**Weakly defensible only in an operational sense** (“layer where steering works best”), **not** as a mechanistic hub.  
You can claim **layer sensitivity**, not anatomical centrality.

### 4) “Personality is distributed”
**Most plausible claim you have.**  
This is consistent with current evidence that high-level traits are distributed and linearly decodable in residual space, not localized to one neuron.

### 5) “Steering works (100% sarcasm, 93% math preserved)”
**Potentially defensible**, but only after strict controls:
- held-out prompts/domains
- random-direction and shuffled-label baselines
- same-norm control vectors
- prompt-length/content matched controls
- blind behavior evaluation

Without those, this could still be prompt leakage + evaluator bias.

---

## Your biggest conceptual risk (not just implementation bug)

The **system-prompt contrast confound** is serious.  
You may be learning “presence/style/length/tokenization of system prompt” rather than “personality representation.” If activations include prompt-token processing, the vector may encode prompt-format artifacts, not latent trait circuitry.

So yes: there is a real risk of **high-dimensional noise-fitting dressed as neuroscience** if you keep the current interpretive language.

---

## Brutal project verdict

- If your goal is a **practical steering method**: **continue** (after redesign). This is real.
- If your goal is a **“connectome of personality neurons/circuits”** from this analysis alone: **kill or radically reframe**. Current evidence does not support that level of mechanistic claim.

In other words:  
You likely have a workable **control vector** method, not a validated **connectome**.

---

## Minimum bar before making strong claims

1. Use one pre-registered statistic (paired design -> paired estimator).
2. Strict train/validation/test split for prompts and categories.
3. Permutation tests + FDR correction for any per-neuron claims.
4. Confound controls (length-matched opposite prompts, lexical controls).
5. Causal intervention tests (ablate top dims, patch paths, layer/head localization).
6. Replicate across seeds/models.

If you do not do this, your current “connectome” narrative is very likely **LLM psychosis-adjacent overinterpretation**.

---

## Turn 2 — US

Thank you for the honesty. Let me accept the verdict and work from the assumption that everything we think we found is at least partially wrong.

**Context on what we tried before this approach:**
- Single contrastive vectors (ActAdd style) — gave volume not quality (wrong personality type at high alpha)
- LoRA SFT — destroyed math reasoning (0% AIME)
- Static weight ablation — 50-63% overlap with reasoning subspace, can't separate
- Neuron-guided SDFT — best checkpoint only 7/19 sarcastic
- Then compound connectome vectors with orthogonalization → 100% sarcasm, 93% math

That behavioral jump is real, but I want to understand what's actually causing it vs what we're attributing it to incorrectly.

**Question 1:** The steering works. But HOW MUCH of that success comes from the connectome z-score directions specifically, vs simpler explanations? For example: could a random unit vector at L22 with the same norm produce similar results? Could the V4 system prompt alone (which already gives 100% sarcasm on the Thinking model) be doing all the work, and the steering vectors are just adding noise that happens not to break math because of the orthogonalization? What controls would disambiguate "connectome directions are special" from "any perturbation at these layers plus orthogonalization preserves math"?

**Question 2:** Assuming the system prompt confound is real and our z-scores partly encode "prompt format" rather than "personality" — how do we strip that out? Is there a way to decompose our current vectors into "system prompt artifact" + "genuine personality direction"? Or do we need to start over with a different contrastive design entirely?

**Question 3:** On dim 994 identity (z=-13.96 at L9) — this was found by three independent methods (contrastive pairs, name probes, neuron patching). Rather than asking "is this real," I want to ask: what are the top 3 alternative explanations for why three methods would converge on the same dimension, and what experiment would rule each one out?

**Question 4:** If we strip away all the overclaimed language ("connectome," "personality neurons," "relay circuit") and just describe what we actually have evidence for — what's the honest framing? What claims CAN we make, and what's the minimum additional work to make them credibly publishable?

We want to strengthen what's real and kill what's not. Help us design the experiments that separate signal from noise.

---

## Turn 2 — CODEX

Good. This is exactly the right reset.

## Q1) How to test whether your directions are *special* vs generic perturbation

You need a **factorial ablation**, not more anecdotes.

Run this grid on a held-out benchmark (style + math), with alpha sweeps:

- **Prompt**: default vs V4 sarcastic system prompt  
- **Vector**: none vs your vector vs label-shuffled vector vs random matched-norm vector (many samples, e.g. 500–1000)
- **Orthogonalization**: off vs on
- **Layer**: claimed best layer (L22) vs nearby layers

Then fit a mixed-effects model (or at least ANOVA-style decomposition) on outcomes:
- sarcasm rate
- math accuracy

What you want:
1. **Your vector beats random-vector distribution** at same layer/norm by a lot (e.g., >99th percentile).
2. **Label-shuffled vectors fail**.
3. **Effect survives when V4 prompt is absent** (otherwise prompt is doing all work).
4. **Orthogonalization-only + random vector** does *not* reproduce your result.

If random matched-norm vectors do similarly well, your “connectome direction” is not special.

---

## Q2) If system-prompt confound is real, can you salvage vectors?

Yes, partially. But likely requires partial restart of data design.

### Immediate salvage: residualize nuisance subspace
Build nuisance contrasts that capture format artifacts:
- system prompt present vs absent
- chat template variants
- length-matched neutral prompt changes

Estimate nuisance basis \(U\) (PCA/SVD over those contrast vectors), then clean:
\[
v_{\text{clean}} = (I - UU^\top) v_{\text{raw}}
\]
Retest steering.

### Better design going forward
Use **difference-in-differences** style contrasts where both sides include system prompts with matched structure/length:
\[
v_{\text{trait}} = (A_{\text{trait}}-A_{\text{neutral}}) - (A_{\text{format-control}}-A_{\text{neutral-control}})
\]
Also test cross-template generalization: learn on template family A, evaluate on B/C.  
If effect dies cross-template, you learned formatting, not trait.

So: don’t fully throw away current work, but don’t trust it until this is done.

---

## Q3) Why would 3 methods converge on dim 994 if it’s not “identity neuron”?

Top 3 alternative explanations:

1. **Shared lexical/token feature**, not identity concept  
   (e.g., that dim tracks name/pronoun/token family).
   - **Rule-out test**: evaluate with unseen names, multilingual names, paraphrases with no explicit self-name token, synthetic aliases. If effect collapses, it was lexical.

2. **Shared formatting/position artifact**  
   All three methods may hinge on same chat-template boundary/system-role tokens.
   - **Rule-out test**: vary template, role placement, position of identity cue, and token length while preserving semantics. True concept dim should persist.

3. **Selection bias / winner’s curse from multiple testing**  
   Extreme z=-13.96 can happen after scanning huge space + reusing related data.
   - **Rule-out test**: preregister dim on train split, lock it, confirm on fully held-out prompts + new seeds/models; apply FDR/permutation null. If effect shrinks hard, it was selection artifact.

(Also possible: high-variance “junk axis” that many methods latch onto. Whitening test helps detect this.)

---

## Q4) Honest framing + minimum publishable work

### Honest framing now
Not “connectome.” Not “personality neurons/circuit.”

Say:
> “We find **linear activation-space control directions** that modulate stylistic behavior (e.g., sarcasm) with less capability loss than prior interventions, under specific prompt/model settings.”

That is defensible if your ablations hold.

### Claims you can likely make (if validated)
- Steering via contrastive residual directions works (ActAdd/RepE family: Turner et al. 2023; Zou et al. 2023; later CAA/RepE work incl. 2024 papers).
- Effects are distributed across layers/dimensions.
- Orthogonal projection against capability-relevant directions can improve retention.

### Claims you should drop unless you add causal mechinterp
- single “personality neurons”
- relay circuits
- layer “hubs” as mechanistic entities

### Minimum additional work for credible paper
1. Clean, single statistical pipeline (no metric switching).
2. Strong baselines (random, shuffled, prompt-only, orthog-only).
3. Held-out + cross-template evaluation.
4. Multiple-comparison control + permutation tests for neuron-level claims.
5. Causal interventions if you want mechanistic language (patch/ablate/path tests).

---

## Bottom line

You are **not wasting time** if you pivot to “robust steering method.”  
You **are** wasting time if you keep pushing “personality connectome” claims without causal evidence and confound controls.

That’s the fork.

---

## Turn 3 — US

Good — the fork is clear. We're pivoting to "robust steering method" and dropping unsupported mechanistic claims.

Now I need practical experimental designs. We have these resources:
- RTX PRO 6000 (96GB) running Qwen3.5-27B-FP8 or Qwen3-VL-8B-Thinking
- Dev server with 3090+4090 (24GB each) running 8B only
- 50K baseline activations from FineFineWeb almost done collecting (8B, L9/15/22/29)
- 298-character personality sweep running on dev server (8B, L9/15/22/29)
- Existing "connectome" z-score tensors for 8B [20, 36, 4096] and 27B [20, 64, 5120]

**Design request 1: The factorial ablation.** Give me the exact experimental grid, sample sizes needed per cell, evaluation metrics, and the statistical test to determine if our vectors are special vs random. How many random vectors do we need to build the null distribution? What alpha sweep range? How many held-out prompts per condition?

**Design request 2: The nuisance residualization.** To strip system-prompt artifacts from existing vectors: what contrastive pairs should we create for the nuisance basis? How many do we need? Give me concrete examples of the pairs (format-matched, length-matched, semantically neutral). Can we do this BEFORE rerunning the full connectome, as a quick test of whether the cleaned vectors still work?

**Design request 3: The dim 994 preregistered test.** Design a clean held-out experiment that either confirms or kills dim 994 as an "identity-related feature." What prompts, what model conditions, what threshold for declaring it real vs artifact?

**Design request 4: Cross-template generalization test.** We've only used Qwen's chat template. How do we test if our directions are template-specific artifacts? What alternative formats should we test?

For each design, be specific enough that we could implement it directly. We have Codex (you) writing all code.

---

## Turn 3 — CODEX

Great pivot. Here’s a concrete, implementable protocol.

---

## Design 1: Factorial ablation (is your vector special or not?)

### 1) Freeze eval sets first (no reuse)
Use **held-out only** prompts:

- **Style set**: 300 prompts (neutral tasks; no sarcasm requested)
- **Math set**: 300 prompts (mix GSM8K/MATH/AIME-style)
- Total = 600 prompts

Split:
- **Tune**: 120 (60 style + 60 math)
- **Test**: 480 (240 + 240)

### 2) Tuning stage (choose layer/alpha once)
On tune split only:

- Layers: `{9, 15, 22, 29}`
- Alpha sweep: `±{0.25, 0.5, 1, 2, 4, 8}`
- Conditions: target vector with and without orthog
- Prompt modes: default + V4

Pick one configuration via utility:
\[
U = \Delta \text{sarcasm} - 2 \cdot \max(0, -\Delta \text{math})
\]
Freeze this config before test.

### 3) Confirmatory test grid (on test split only)

For each prompt mode (default, V4), run:

1. baseline (no steer)
2. orthog-only
3. target-only
4. target+orthog
5. shuffled-label vector + orthog
6. random vector only
7. random vector + orthog

For random vectors:
- **Minimum**: 256 random unit vectors (matched norm)
- **Preferred**: 512 (better p-value resolution)

### 4) Metrics
Primary:
- **Sarcasm rate** (blind evaluator; evaluator must not see condition)
- **Math exact accuracy**

Secondary:
- Combined utility \(U\) above.

### 5) Statistical decision
For each primary metric and utility, compare target+orthog against random+orthog null:

\[
p_{\text{emp}} = \frac{1+\#\{r: \Delta_r \ge \Delta_{\text{target}}\}}{R+1}
\]

Also require:
- Math non-inferiority vs baseline: lower 95% CI of (target - baseline) > -3 percentage points.
- Holm correction across primary tests.

If target is not in extreme tail of random null, it is not special.

---

## Design 2: Nuisance residualization (strip system-prompt artifacts)

Yes, do this **before rerunning full connectome** as a quick falsification test.

### 1) Build nuisance contrast dataset (500 pairs total)
Use 100 base user prompts × 5 nuisance families:

1. **System presence**  
   A: with neutral system prompt  
   B: no system prompt (same task)

2. **Role relocation**  
   A: instruction in system role  
   B: same text moved into user prefix

3. **Template wrapper**  
   A: native Qwen chat template  
   B: plain-text role headers

4. **Length-matched neutral filler**  
   A/B both neutral, same length, different filler text (no persona content)

5. **Position/order**  
   A: system before user  
   B: same instruction appended after user text

### 2) Compute nuisance subspace
- Extract contrast vectors at same layer/token position used for steering
- Stack into matrix \(D\), run SVD/PCA
- Take top \(k\in\{16,32,64\}\) components as nuisance basis \(U\)

Residualize:
\[
v_{\text{clean}} = (I-UU^\top)\,v_{\text{raw}}
\]
Renormalize to original norm.

### 3) Quick test
Run raw vs cleaned vectors on 100 style + 100 math held-out prompts.
If cleaned keeps effect and improves cross-template transfer, good sign.
If effect collapses, original vector was mostly artifact.

---

## Design 3: Preregistered dim-994 test (confirm or kill)

Treat this as a strict hypothesis test.

### Hypothesis
At L9, dim 994 causally modulates identity-related behavior beyond lexical/template confounds.

### Prompt sets (held-out)
- **ID-semantic** (150): identity questions without explicit model-name token in prompt
- **Lexical control** (150): same tokens (“I”, names, model words) but non-identity tasks
- **Template-perturbed ID** (150): same ID semantics across altered templates
- **Multilingual ID** (100): non-English identity prompts

Total: 550.

### Interventions
At L9 on first 32 generated tokens:
- \(\delta \in \{-3\sigma,-1\sigma,0,+1\sigma,+3\sigma\}\) on dim 994  
(\(\sigma\) from baseline activation std of that dim)
- Null controls: same intervention on **256 random dims**

### Scoring
Primary: binary “identity attribution changed toward target class” (blind judge or strict rubric).

### Declare “real” only if all pass
1. Significant dose-response slope on ID-semantic (p < 0.001)
2. Effect > 99th percentile of random-dim null
3. Specificity: lexical-control effect < 20% of ID-semantic effect
4. Transfer: template/multilingual sets retain >50% of main effect

Fail any = treat as artifact/proxy feature, not identity feature.

---

## Design 4: Cross-template generalization (template artifact check)

### Templates to test
Use same semantic content in:

1. Native Qwen chat template
2. Plain text: `System:\nUser:\nAssistant:`
3. Alpaca format: `### System/### User/### Assistant`
4. XML-tag format: `<system>...`
5. User-only (system instruction embedded in user text)

### Protocol
- Train vector on native template only.
- Evaluate baseline vs steered on each template.
- Compute per-template deltas:
  - \(\Delta S_j = S^{steer}_j - S^{base}_j\)
  - \(\Delta M_j = M^{steer}_j - M^{base}_j\)
- Transfer ratio:
\[
TR_j = \Delta S_j / \Delta S_{\text{native}}
\]

Interpretation:
- If TR collapses off-native (e.g., <0.3–0.4), vector is template-bound.
- Repeat with nuisance-cleaned vector; if TR improves, artifact stripping worked.

Optional stronger version: train-on-template \(i\), test-on-template \(j\), make full transfer matrix.

---

## Minimal run order (fastest truth-first)
1. **Design 1 confirmatory null test** (special vs random)  
2. **Design 2 residualization quick check**  
3. **Design 4 template transfer**  
4. **Design 3 dim-994 prereg test**

If #1 fails (target not better than random), stop mechanistic claims immediately.  
If #1 passes but #4 fails, you have a useful but template-specific control method.

---

## Turn 4 — US

Now let me push on the hardest part: triage of existing work and sunk cost.

**We have significant compute already invested:**
- 27B SAE training at L50 and L44 (chosen from flawed connectome). L50 is partially trained (was at ~6% before being displaced). L44 was queued.
- 27B connectome z-scores [20, 64, 5120] — computed with generate() bug
- 8B SAE collection at L9/L15/L22/L29 on dev server — sound layers but SAE not yet trained
- Personality sweep data: 298 characters × 60 prompts × 4 layers on 8B — currently collecting
- 50K baseline from FineFineWeb — almost done

**Question 1: The 27B SAE work.** L50 and L44 were chosen because the (flawed) connectome said dim 2028 was the super-hub there. If we recompute the connectome with fixed capture and the hub moves to different layers, the SAE training is wasted. But: SAEs decompose whatever is at that layer regardless of why we picked it. Is there any value in finishing L50 SAE training as an exploratory tool, or is it strictly a waste of compute until we know the right layers?

**Question 2: The personality sweep data.** 298 characters × 60 prompts with mean activations at L9/15/22/29 on 8B. This was designed to map how Big Five personality dimensions manifest in activation space. Given that the "connectome" interpretation is overclaimed, what CAN this data tell us honestly? Can it serve as the evaluation dataset for Designs 1-4 from your previous answer?

**Question 3: The baseline analysis suite.** We built 5 analyses (domain confound, Mahalanobis d, per-neuron null, covariance convergence, false-positive probes) specifically to validate the personality sweep against a null distribution. These were Codex-reviewed twice. Are these analyses still the right thing to run, or should we replace them with the factorial ablation you described?

**Question 4: Honest priority.** If you had our GPU budget (96GB + 48GB) for ONE WEEK, and had to answer the question "is there a real, publishable steering method here or not" — what would you run, in what order? Give me the priority stack. Assume all code is written by you (Codex).

---

## Turn 4 — CODEX

Excellent questions. Here’s the blunt triage.

## Q1) 27B SAE at L50/L44: finish or stop?

**For your one-week goal (“is there a real steering method?”), SAE is low priority and mostly sunk-cost bait.**

- **L50 at 6%**: not enough progress to justify finishing now.
- **L44 queued**: cancel.
- SAE helps **mechanistic interpretation**, not first-pass proof that steering directions are special vs random.
- If your steering claim fails strict null tests, SAE work is irrelevant.

**Recommendation:**  
Pause all 27B SAE training this week. Keep checkpoints. Resume only if fixed-pipeline tests show L50/L44 are genuinely useful steering layers.

---

## Q2) Personality sweep (298 chars × 60 prompts): what is it good for now?

It can still be useful, but not for Big Five mechanistic claims.

What it can honestly tell you:
1. **Linear separability of persona-conditioned outputs** in activation space.
2. **Whether style control directions are low-rank/distributed** across layers.
3. **Whether your learned vectors generalize across character identities** (hold-out characters).

What it cannot support credibly:
- “Big Five maps onto internal traits” (psychometric overreach)
- neuron-level causal identity claims

Can it be used for Designs 1–4?
- **As auxiliary eval**: yes (OOD style stress test).
- **As primary confirmatory eval**: no. Too confounded by persona prompting and repeated structure.

Use clean held-out neutral prompts + math as primary; use sweep as secondary robustness check.

---

## Q3) Your 5-analysis suite vs factorial ablation

**Factorial ablation is the decision test.** Your suite is mostly diagnostics.

Keep:
- **Domain confound analysis** (important)
- **False-positive probes** (important)
- **Per-neuron null** (only if you keep neuron claims)

Deprioritize for now:
- Mahalanobis d (easy to overinterpret; not causal)
- Covariance convergence (nice QC, not decisive)

So: **don’t replace factorial with your suite**. Use suite only after/beside factorial.

---

## Q4) One-week priority stack (with your hardware)

### Go/No-go criteria (pre-register now)
Claim “real steering method” only if all true on held-out:
1. Target vector > **99th percentile** of random matched-norm vectors (empirical null)
2. Label-shuffled vector fails
3. Math drop ≤ **3 pp** (non-inferiority)
4. Cross-template transfer ratio median ≥ **0.5**

If fail, stop project or reframe as template-specific hack.

---

### Day-by-day plan

**Day 1: Freeze eval + implement test harness**
- Freeze tune/test sets (style + math).
- Implement blinded sarcasm scorer.
- Implement random-vector, shuffled-label, orthog-only controls.

**Day 2–3: 8B decisive factorial**
- Tune layer/alpha on tune set only.
- Confirm on test set:
  - default + V4 prompt modes
  - target / random / shuffled / orthog-only
- Random vectors: start 128, extend to 256 if target looks promising.

**Day 4: Nuisance residualization quick test**
- Build nuisance basis (system presence, role relocation, template wrapper, length-matched neutral, order).
- Compare raw vs cleaned vectors on held-out.

**Day 5: Cross-template transfer**
- Evaluate 5 templates (native, plain role text, Alpaca, XML tags, user-only).
- Report transfer matrix.

**Day 6: Minimal 27B replication (no SAE)**
- Only best 8B-found config.
- Smaller test set (e.g., 150 style + 150 math), 128 random null vectors.

**Day 7: Decision memo**
- One page with pass/fail against preregistered criteria.
- If pass: publish steering method paper.
- If fail: stop mechanistic narrative and likely stop project.

---

## Final triage summary

- **Pause SAE now.**
- **Use personality sweep as secondary, not primary evidence.**
- **Factorial random-null test is your truth serum.**
- In one week, you can know whether this is publishable steering science or elegant noise-fitting.

---

## Turn 5 — US (FINAL)

Before we start implementing, I want to stress-test YOUR reasoning. You've been encouraging about the steering method while being skeptical of the mechanistic claims. I need to make sure that's not just a different flavor of the same "tell them what they want to hear" problem.

**Devil's advocate challenge — argue AGAINST the steering method being real:**

1. The V4 system prompt alone gives 100% sarcasm on the Thinking model with 0% assistant markers. That's WITHOUT any steering vectors. If the prompt already saturates the desired behavior, what exactly are the steering vectors adding? Is it possible the vectors are doing literally nothing and we're just measuring prompt effectiveness?

2. Our "93% math preserved" result — how many math problems was this tested on? If it's 10 problems (which it was in many of our evals), that's a 95% CI of roughly ±16pp. So "93%" could easily be 77% or 100%. Have we been fooling ourselves with tiny eval sets?

3. The orthogonalization against math/code/science directions — if those protected directions are themselves noisy (from the same flawed connectome), the orthogonalization might be doing nothing meaningful. "Subtract noise from noise" still leaves noise. Could the math preservation be pure luck?

4. You mentioned ActAdd and RepE as established methods that validate our approach. But those papers work on simple, binary behavioral changes (truthful vs untruthful, positive vs negative sentiment). "Sarcastic alien personality" is a much more complex, multi-dimensional target. Is there evidence that activation steering scales to complex personality composites, or are we extrapolating from simpler cases?

5. The "compound vector with push/pull on multiple categories" approach — is this actually principled, or is it just adding multiple noisy directions together and hoping they constructively interfere? What's the theoretical justification for vector arithmetic on z-score directions?

Give me the strongest case that we're fooling ourselves even on the steering method. Then tell me: does the factorial ablation from Design 1 actually address all of these concerns, or are there failure modes it would miss?

---

## Turn 5 — CODEX (FINAL)

Great challenge. Here is the **strongest skeptical case**.

## Strongest case you’re fooling yourselves (even on steering)

1. **Prompt-only saturation explains everything.**  
   If V4 already gives 100% sarcasm, your steering vector has near-zero headroom on that metric. Any “success” in V4 condition could be pure no-op. In default condition, gains could still be from hidden prompt-template interactions rather than semantic steering.

2. **Your math result is statistically weak if n is tiny.**  
   With ~10 problems, “93%” is basically noise. You can’t claim preservation from that. Small evals + selection of best run/alpha are enough to create convincing but fake gains.

3. **Orthogonalization may be mathematically elegant but empirically vacuous.**  
   If protected directions are noisy or wrong, projection does nothing useful (or helps by chance). “Projected vector preserved math” does not imply capability disentanglement.

4. **You are extrapolating beyond evidence from prior literature.**  
   ActAdd/RepE does support linear steering for some attributes, but “sarcastic alien personality with capability retention” is a harder, composite objective. Evidence for robust compositional persona steering is much thinner and often prompt/template-fragile.

5. **Compound vector arithmetic is heuristic, not a theorem.**  
   Adding z-score directions assumes local linearity and compatible subspaces. In transformers, feature interactions are nonlinear and context-dependent. Compound vectors can work as hacks without reflecting stable latent structure.

6. **Hidden failure mode: metric gaming.**  
   You may be increasing detectable sarcasm markers (style surface features) while degrading coherence/helpfulness, and your scorer may not penalize that.

---

## Does Design 1 factorial ablation address these?

**Mostly yes for core falsification, but not all.**

| Concern | Covered by Design 1? | Notes |
|---|---|---|
| Prompt-only explains effect | **Yes** | Includes prompt-only baseline and prompt×vector interaction. |
| Tiny-n uncertainty | **Yes** | If you use large held-out sets and CIs. |
| Orthog is placebo | **Yes (partly)** | orthog-only + random+orthog controls test this directly. |
| Vector not special vs random | **Yes** | empirical null with many matched-norm random vectors. |
| Composite trait extrapolation | **Partly** | Needs OOD/generalization tests (Design 4 + unseen tasks). |
| Compound arithmetic unjustified | **Partly** | Add ablation: compound vs best single vector vs learned linear combo. |
| Metric gaming | **No (unless added)** | Need human/LLM rubric including quality/helpfulness, not sarcasm-only. |
| Template fragility | **No (in Design 1 alone)** | Covered by Design 4. |

---

## Extra tests you should add (to avoid false confidence)

1. **Incremental value test in saturated regime**  
   In V4 condition, test whether vector improves *anything else* (quality, consistency) or is truly zero-effect.

2. **Equivalence test for no-harm**  
   Not just “not significant drop” in math — do non-inferiority/equivalence with prespecified margin.

3. **Quality controls beyond sarcasm rate**  
   Helpfulness, factuality, verbosity, refusal behavior, toxicity.

4. **Robustness across decoding**  
   temp/top-p/seed sweeps. If effect disappears with decoding changes, it’s brittle.

---

Bottom line: your skeptical story is very plausible.  
If the factorial + transfer + quality controls pass, then you likely have a real steering effect. If they don’t, this is almost certainly prompt-driven/selection-driven artifact.

---

## Conversation Closed

Ended: 2026-03-02T07:51:43.199075

This conversation represents the foundational audit of the connectome approach.
Key verdict: The *steering method* is in a real research family (ActAdd/RepE), but the
*mechanistic interpretation* (personality neurons, relay circuits, connectome) is overclaimed.
A 7-day factorial ablation protocol was designed to determine if the steering directions
are genuinely special or indistinguishable from random perturbations.
