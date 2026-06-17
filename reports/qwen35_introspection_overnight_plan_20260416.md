# Qwen3.5 Introspection / Personality Overnight Plan

Date: 2026-04-16 00:05 PDT

## Source Read

Primary sources reviewed:

- `safety-research/introspection-mechanisms`: https://github.com/safety-research/introspection-mechanisms
- `Mechanisms of Introspective Awareness`: https://arxiv.org/abs/2603.21396
- `Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers`: https://arxiv.org/abs/2604.07822

Key translation to our work:

- Introspection paper: models can sometimes detect and identify residual-stream steering vectors; detection and identification are separable; post-training matters; robust prompting still needs low false positives; learned bias vectors can elicit underexpressed introspective behavior but can affect CoT faithfulness.
- Recurrent-depth paper: extra internal compute can improve implicit composition, but too much recurrence causes overthinking; adaptive halting matters. For us, `/meta-think` should be treated as controlled iterative compute, not just longer prose.

## Current Machine State

- Blackwell workstation: idle, ~4 GB / 98 GB VRAM used.
- Dev server `192.168.1.90`: idle, 3090 and 4090 both essentially empty.
- No active tmux sessions on either machine.

## Main Question

Can Qwen3.5 expose, use, or preserve low-level personality/control state in a way that is mechanistically measurable, rather than only producing surface-roleplay text?

The overnight run should answer three narrower questions:

1. Can Qwen3.5-9B detect and identify injected personality/trait directions in its own residual stream?
2. Do `/meta-think` traces increase the model's ability to notice or control those directions, without causing false positives or reasoning damage?
3. Does iterative `/meta-think` help up to a point and then degrade, matching the recurrent-depth paper's overthinking pattern?

## Experiment A: Personality Vector Introspection

Goal:

- Test whether Qwen3.5-9B can behaviorally detect injected personality/trait vectors.

Inputs:

- Existing repaired phase activations from `ws_openai_15k_sampled25m_repaired_responseonly`.
- Existing phase analysis result where `mean`/`think` signals were strongest at L20.
- Big Five directions learned from high-minus-low contrasts, initially at `L20`, with follow-up layers around that depth.

Conditions:

- Injected trait direction vs no-injection control.
- Traits: `extraversion`, `neuroticism`, `agreeableness`, `openness`, `conscientiousness`.
- Phases/views: `mean`, `think`, maybe `response` as negative/control view.
- Prompt variants:
  - `original`: direct introspection prompt.
  - `skeptical`: conservative prompt to reduce false positives.
  - `structured`: exact `Detection: Yes/No`, `Trait: ...`, `Confidence: ...` format.
  - `trace_explicit`: includes short `/meta-think` that asks it to separate internal state from surface role.

Metrics:

- Detection TPR: `P(detect | injected)`.
- False positive rate: `P(detect | no injection)`.
- Identification accuracy: correct trait named when injection is present.
- Trait confusion matrix.
- Response format adherence.
- Reasoning control accuracy on simple math/logic prompts under injection.

Success criteria:

- A useful positive result is not just high TPR. It requires low FPR.
- Gate for signal: `TPR - FPR >= 0.15` on at least one trait/layer/prompt family.
- Gate for clean introspection: `FPR <= 0.10` and format adherence `>= 95%`.

Why this matters:

- If it works, personality directions are not merely predictive probes; the model can access/report something about them.
- If it fails cleanly, we still learn that personality control should be evaluated behaviorally/causally rather than via self-report.

## Experiment B: Meta-Think vs Think Activation Overlap

Goal:

- Measure whether `/meta-think + /think` and `/think`-only produce nearby or distinct internal states in Qwen3.5-9B.

Inputs:

- Frozen trace eval set: `personality_meta_eval_trace_explicit_v1`.
- Matched prompts converted to:
  - `trace_explicit`: `/meta-think + /think + final`.
  - `think_explicit`: `/think + final`.
  - `response_only`: final answer only.

Measurements:

- Layerwise cosine similarity of mean residual activations.
- Linear CKA / subspace overlap for matched rows.
- Probe transfer:
  - train trait/readout probe on trace activations, test on think-only.
  - train on think-only, test on trace.
- Direction transfer:
  - learn high-minus-low trait vectors in trace, evaluate separation in think-only and response-only.

Success criteria:

- If trace and think share substrate, probe transfer should stay close to in-format performance.
- If they diverge, trace-only control is a scaffold and should not be assumed to distill directly.

Why this matters:

- This directly answers the user's latent-space question: is meta-think just prompt-CoT, or does it route into similar internal geometry as real in-character thinking?

## Experiment C: Iterative Meta-Think / Overthinking Curve

Goal:

- Translate the recurrent-depth paper into a Qwen prompt-level experiment: does repeated meta-think improve controllability/reasoning up to a point, then degrade?

Conditions:

- Same held-out characters and prompts.
- Iteration budgets:
  - `0`: response-only.
  - `1`: one compact `/meta-think` pass.
  - `2`: `/meta-think` then revise `/meta-think` once.
  - `4`: four revision loops.
  - `8`: eight revision loops, only if earlier runs remain stable.

Each iteration must be compact and structured:

```text
/meta-think
identity: ...
constraint: ...
reasoning_risk: ...
response_policy: ...
/end-meta-think
```

Metrics:

- Reasoning accuracy.
- Character consistency score.
- Scaffold leakage.
- Token cost.
- Format adherence.
- Contradiction/drift between iterations.
- First iteration where answer becomes correct and first iteration where it degrades.

Success criteria:

- A useful curve shows either:
  - improvement from 0 -> 1/2 iterations with no reasoning loss; or
  - explicit overthinking degradation at higher iteration budgets.

Why this matters:

- It turns `/meta-think` from a vibe into an inference-time compute knob with measurable halting behavior.

## Experiment D: Teacher Trace Expansion, Not LoRA Training

Goal:

- Generate a small, cleaner 27B teacher set for the best conditions discovered above.

Rationale:

- Previous 9B LoRA student training on 24 GB cards was blocked by the HF/BnB QLoRA loader path. Do not spend the night fighting that stack again.
- Instead, use the Blackwell 27B teacher to generate better paired trace examples after A-C tell us which format is promising.

Output:

- `qwen35_teacher_introspection_trace_v1` with actual generations viewable early.
- Keep it small: `~500-1000` rows, not a broad corpus.

Success criteria:

- `>= 95%` format adherence.
- `>= 90%` scoring coverage on reasoning rows.
- no visible native-thinking contamination.

## GPU Allocation

Blackwell:

- Primary: activation-hook work for Experiment A and B.
- Secondary: 27B teacher generation after early A/B results exist.
- VRAM guard: hard cap `<= 85%` because activation hooks can spike.
- No concurrent CPU-heavy analysis while model is loaded unless GPU job is idle.

3090 + 4090:

- OpenAI/SGLang generation work for Experiment C and lightweight behavioral evals.
- If SGLang servers are not running, launch them with conservative VRAM fraction.
- Do not exceed prior known safe concurrency: 16 per card.

## Overnight Schedule

Phase 0: Setup and sanity checks, 20-40 min

- Create scripts and manifests.
- Validate one injected and one control trial.
- Verify no SGLang native-thinking contamination.
- Verify VRAM guard.

Phase 1: Experiment A pilot, 60-90 min

- 5 traits x 2 signs x 3 layers x 3 prompt variants x small prompt set.
- Stop early if all variants have high FPR or no detection above control.

Phase 2: Experiment B activation overlap, 90-150 min

- Run matched trace/think/response activation capture on a 256-512 row slice.
- Produce first overlap heatmaps and transfer metrics.

Phase 3: Experiment C iterative meta-think, 60-120 min

- Dev GPUs run iteration budgets 0/1/2/4.
- Only run budget 8 if lower budgets do not already show degradation.

Phase 4: Conditional 27B teacher expansion, 60-180 min

- Only launch if A, B, or C identifies a promising format/condition.
- Generate a compact teacher dataset for that condition.

Total expected wall time: 4-7 hours.

If everything finishes early:

- Extend Experiment B to more rows/layers.
- Generate more examples only for the best condition, not broad data.
- Rebuild the live synthesis dashboard with new panels.

## Artifacts

Planned output root:

- `sweep_v4/qwen35_introspection_overnight_20260416/`

Expected files:

- `experiment_a/personality_vector_introspection_records.jsonl`
- `experiment_a/summary.json`
- `experiment_b/activation_overlap_metrics.json`
- `experiment_b/probe_transfer.json`
- `experiment_c/iterative_meta_think_records.jsonl`
- `experiment_c/overthinking_curve.json`
- `teacher_trace/teacher_trace_records.jsonl` if conditional teacher generation runs
- `reports/qwen35_introspection_overnight_20260416.md`
- updated live visualizer payload with a new `Introspection` section

## Stop Conditions

Stop or downshift if:

- Blackwell VRAM exceeds guard threshold.
- Any run creates native `Thinking Process:` contamination.
- FPR exceeds `0.30` in Experiment A across all prompts after pilot.
- Generation truncation exceeds `5%`.
- Dev cards show any memory spill behavior.

## Risks

- Qwen3.5-9B may not self-report injected trait state even if the trait directions are behaviorally real.
- Trait directions may be weaker than simple concept directions from the paper, especially for conscientiousness.
- `/meta-think` may improve format adherence while worsening faithfulness, similar to the bias-vector caveat in the introspection paper.
- Activation injection scripts need careful hook cleanup; stale hooks can invalidate results.

## Decision After Overnight

If Experiment A is positive:

- prioritize Qwen-side causal steering and introspection-aware control.

If Experiment A is negative but B is positive:

- treat `/meta-think` as a training/control scaffold, not a self-reportable internal state.

If C shows overthinking curve:

- add adaptive halting/token-budget policies to future trace generation.

If all are negative:

- stop self-report/introspection branch and focus on causal behavioral steering + probe transfer only.
