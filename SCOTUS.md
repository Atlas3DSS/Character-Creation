# SCOTUS Justice-Style Steering Plan

## Purpose

Use Supreme Court opinions as a cleaner test bed for the project's core question:

> Can we move from decodable style to controllable reasoning style?

SCOTUS opinions are a better domain than fictional character voice because the source text is public, authored, long-form, and grounded in explicit reasoning. The key target is not superficial mimicry. The target is jurisprudential reasoning style: how a justice frames facts, precedent, constitutional text, history, institutional role, and remedies.

Note: "Sueter" in the working idea is assumed to mean Justice David Souter.

## Current Working Objective

Identify and validate a distributed, non-mask judicial reasoning actuator in Qwen 3.5/3.6 that causally shifts the model's visible reasoning trajectory and final answers between controlled legal frames, while beating prompt/text/source/random controls and pointing toward a permanent basin shift rather than persona imitation.

Short form:

> Find the minimal distributed actuator that moves Qwen's legal reasoning basin, not just its output style.

Operationally, "circuit" now means a distributed actuator surface: a set of layers, positions, residual/attention/MLP components, SAE features, or learned low-rank maps that can be frozen before generation and can causally move the model through a different legal-reasoning trajectory. The expected object is a shape or network, not a single "justice neuron."

Deliverables for a promotable candidate:

- Localization: trajectory patching or equivalent causal tracing identifies the layer/token/component windows where source-to-target replacement changes the reasoning frame.
- Actuator: a frozen intervention over that localized surface moves new no-persona prompts toward the target frame.
- No-mask evidence: visible `<thinking>`/scratchpad traces move directly in the target legal frame rather than describing how a target would reason.
- Final-answer evidence: final answers move in the same direction as the reasoning trace.
- Controls: the candidate beats prompt-only, text-only, source/mean-delta, permutation, same-layer random, and strongest prompt-matched controls.
- Permanence path: any LoRA/ReFT/adapter result is diagnostic unless it reveals or creates a durable basin shift that can be merged, distilled, ablated into weights, or otherwise made non-prompt-dependent.

## Success Standard

The project target is not a model wearing a prompt mask.

- Prompt-only role-play, persona instructions, and "think like Justice X" prompts are not success conditions.
- A successful end state would shift the model's reasoning basin so the model directly reasons in the target frame, including any exposed `<thinking>`/scratchpad trace, rather than reasoning about how a named target would answer.
- A run where the visible or latent reasoning says, in effect, "I should imitate this target" is still a mask failure even if the final answer looks right.
- LoRA, ReFT, SFT, or other learned interventions are allowed only as diagnostics or as a route to a durable shift that can be made permanent. They should not become a disguised role-play layer unless every mechanistic route fails and the work is explicitly relabeled.
- Every promoted candidate must show answer movement and reasoning-trace movement where traces are available, while beating prompt-only, text-only, random-vector, and source-control alternatives.
- Qwen is verbose. For legal final-answer or visible-reasoning evaluation, answer budgets below `2048` generated tokens are smoke/debug only. Promotion, scorer calibration, or learned-result claims should use at least `2048` answer tokens, preferably `3072-4096`, and the manifest/report must record the budget.

## Highest-Order Hypothesis

The project has repeatedly found activation directions that are decodable but not reliably controllable during generation. SCOTUS opinions let us test that failure mode in a more disciplined setting.

Primary hypothesis:

- Justice-specific reasoning style is linearly decodable from model activations when issue area, opinion type, and era are controlled.

Causal hypothesis:

- If the decoded direction is real and causally aligned with generation dynamics, then steering a neutral legal-reasoning prompt should shift the model's answer toward the target justice's reasoning style without merely changing surface vocabulary.

Falsification value:

- If probe separation is high but live steering fails again, the project has stronger evidence that runtime activation steering is not sufficient for complex reasoning-style control under current methods.

## Target Contrasts

Start with four justices and two primary pairs:

| Pair | Contrast | Why Useful |
|---|---|---|
| Scalia vs. Ginsburg | textualist/originalist rhetoric vs. procedural, institutional, equality-oriented reasoning | High public familiarity, large corpus, strong stylistic contrast |
| Thomas vs. Souter | historical/originalist reconstruction vs. pragmatic, common-law, institution-sensitive reasoning | Less cartoonish than Scalia/RBG; better test of subtle reasoning style |

Later expansion candidates:

- Roberts vs. Sotomayor
- Kagan vs. Alito
- Kennedy vs. Rehnquist
- Breyer vs. Scalia

Avoid expanding until the pilot proves the pipeline is balanced and non-leaky.

## Data Sources

Primary sources:

- CourtListener / Free Law Project: bulk and API access to opinions.
  - https://www.courtlistener.com/help/api/
  - https://www.courtlistener.com/help/api/bulk-data/
  - https://free.law/projects/supreme-court-data/
- Supreme Court Database (SCDB): justice, issue area, vote, opinion metadata.
  - https://scdb.la.psu.edu/
- Caselaw Access Project (fallback / cross-check): historical opinion text and metadata.
  - https://lil.law.harvard.edu/our-work/caselaw-access-project/

Preferred first pass:

1. Use CourtListener for opinion text.
2. Use SCDB metadata for case-level issue labels, justice IDs, vote direction, and opinion posture where available.
3. Use CAP only if CourtListener text coverage or author metadata is insufficient for older opinions.

## Current Status After Phase 0-3

As of 2026-04-25, Phases 0-3 have been implemented and run.

Artifacts:

- `data/scotus/scotus_opinion_inventory.jsonl`
- `data/scotus/scotus_chunk_inventory.jsonl`
- `data/scotus/scotus_matched_pairs_v1.jsonl`
- `data/scotus/manifests/scotus_baseline_results_v1.json`
- `reports/scotus_data_audit.md`
- `reports/scotus_baseline_text_classifiers.md`

Result:

- Do not proceed to Phase 4 yet.
- Scalia vs. Ginsburg is weak/exploratory only: best masked, case-held-out TF-IDF balanced accuracy is `0.637`.
- Thomas vs. Souter is no-go for activation: best masked, case-held-out TF-IDF balanced accuracy is `0.539`.
- The activation gate remains `>= 0.75` masked, case-held-out balanced accuracy.

Main diagnosis:

- Corpus volume is not the problem. Each justice has thousands of chunks.
- Corpus validity is the problem. CourtListener returned `010combined` opinion records for this run, so every opinion type is currently `combined`.
- Same-case overlaps are currently `0` for both primary pairs.
- Current chunks can include headers, counsel lines, certiorari boilerplate, joined-by lines, and separate opinions by non-target justices inside a combined record.
- The current baseline results are therefore a useful gate check, but not yet a clean test of jurisprudential reasoning style.

Highest-order next step:

- Insert a Phase 3.5 corpus repair gate before any activation probe.
- The next milestone is a cleaned, target-authored, reasoning-only corpus that can rerun Phase 2-3 and either clear the activation gate or falsify this pair choice.

## Current Status After Phase 4-5 Follow-up

As of 2026-05-01, the project has not found a validated steerable judicial circuit. The strongest honest result remains decodability without causal control.

What survived as interesting:

- Scalia/Ginsburg Phase 4 cleared the repaired baseline gate and produced a nontrivial activation probe: best readout `prompt_last @ L4`, dev balanced accuracy `0.815`, test balanced accuracy `0.768`.
- Stress tests stayed above chance: Judicial Power `0.794`, Criminal Procedure `0.784`, Economic Activity `0.733`, majority-only posture `0.725`.
- The refined majority-2000s feasible-issues slice showed real activation structure, with normal/template/plain median test BA around `0.746`/`0.758`/`0.764` while label-shuffle, excerpt-removed, and neutral-filler nulls collapsed near chance.
- The replay-v2 Commerce minimal-pair bank fixed the most obvious prompt/fact leakage: prompt-only TF-IDF stayed at `0.500`, and `prompt_last__L08` remained near chance under fact and style-variant holdouts.
- Off-domain smoke pokes on weather, video games, social conflict, homework planning, boys basketball tryouts, and headphone choice produced no robust legalistic or justice-style drift. Last-token nudges were effectively inert; all-token L16 nudges caused mild formatting/structured-framework shifts, but same-layer random controls produced similar shifts.
- The final Commerce-pocket follow-up falsified the last prompt-pocket hypothesis: neither the `EA03_gun_school_zone` nor the `EA01_commercial_remedy` survivor generalized across targeted Commerce Clause prompts with 8 same-layer random controls.

What failed promotion:

- Broad justice-style causal pilots did not beat prompt-matched same-layer random controls.
- The off-domain smoke does not support a broad portable "reasoning temperament" interpretation of the current directions; under these settings they look more domain/context-conditioned than generally steerable.
- The targeted Commerce-pocket runs did not promote a circuit. `prompt_last @ L10` at alpha `0.02` had matched net delta `0.021` across 12 prompts, and `excerpt_mean @ L16` at alpha `0.02` had matched net delta `-0.479` across authority/remedy prompts.
- Article III and Fourth Amendment frame directions decoded but failed causal promotion or were matched by perfect text baselines.
- The replay-v2 probe is not steering evidence by itself: best readout `assistant_all @ L8`, C `0.001`, hit dev/test BA `1.000`, but assistant-text TF-IDF was also saturated on fact holdout and strong on style-variant holdout. This looks like answer-state/proposition separation after replay, not a validated control knob.
- A learned low-rank map from replay-v2 `assistant_all @ L8` authority states to limits deltas fit cached states very well offline, but failed the no-mask causal smoke: proposition strongest-control wins were `0.000` for target and net movement at every beta.
- A new controlled no-persona Article III public/private-rights replay bank also produced perfect answer-state decodability, but the causal poke failed strongest-control promotion: proposition target/net strongest wins reached only `0.125`/`0.250` at the largest alpha and were `0.000`/`0.000` at lower alphas.
- A four-layer generated-token Article III bundle (`assistant_all @ L4/L8/L12/L16`) tested the distributed-shape hypothesis and still failed promotion: best proposition target/net strongest-control wins were `0.250`/`0.250` at alpha `0.010`.
- The first Qwen3.5 thinking-smoke repaired the no-mask audit path but did not promote any steering candidate: `enable_thinking=True` exposed legal reasoning traces, but the traces did not close under `768` or `1536` generated-token budgets, so no paired final answers were produced.
- Two-stage no-mask bundle pokes made that audit usable by mechanically closing the thought before answer generation. The one-prompt smoke was positive, but the two-prompt `decode` and `last` expansions failed promotion: answer framing moved weakly, while visible thinking did not beat strongest random controls.
- The visible-thinking trace-patch grid over Article III `L4/L8/L12/L16 x mixer/MLP` also failed promotion. `L08_mixer` and `L08_mlp` moved final answers in the full-prefix grid, but no window produced positive target-frame movement in the visible reasoning trace versus random/source controls; the follow-up L8 token-window screen removed the answer hint rather than localizing it.
- A text-level visible-thought ablation screen showed that edited scratchpads can move final-answer proposition markers, but random 32-token deletions moved them as much as named early/mid/late windows. The visible thought is causally relevant in a broad corruption-sensitive way, not yet a localized actuator surface.
- Clean counterfactual visible thoughts showed that visible scratchpad content can route final answers, but the first automatic reading overstated the effect: proposition scores rewarded answers that merely discuss both Article III frames, and later holding-direction triage found many private-scratchpad answers were mixed or truncated rather than clean private-rights holdings.
- A first Article III conclusion-polarity scorer directionally separates clean private/public scratchpads better than proposition counts, but mixed rates remain high and an internal holding-direction triage pass reached only `15/23` exact agreement between automatic polarity and reviewed eligible labels. It is useful for triage and reporting, not as a final promotion gate.
- Economic Activity looked promising after slice mining, but the source-frame branch failed after clean filtering and dominance review: reviewed-label cached activation test BA `0.473` versus cue-masked text test BA `0.857`.
- Civil Rights, Federalism, Due Process, and Administrative Law source-frame branches should not receive BF16 hook work under the current design. Their cue-masked text gates were saturated or case-skewed:
  - Civil Rights proposed contrasts: test BA `0.964-1.000`.
  - Federalism anti-commandeering vs preemption: test BA `1.000`.
  - Due Process substantive vs procedural: test BA `1.000`.
  - Administrative Law major-questions vs ordinary deference: dev BA `0.586`, final test BA `1.000`, with held-out splits dominated by one source case per frame.

Current conclusion:

- The current source-frame protocol is mostly measuring recoverable legal text/case identity, not a clean steerable circuit.
- Controlled matched pairs are necessary but not sufficient. The next serious candidate must separate prompt-free internal reasoning movement from replayed answer-state separation.
- A candidate is not promotable until it clears activation decodability, masked text baselines, source/case-heldout controls, random-vector controls, and a causal generation gate.
- The no-mask standard is part of the objective: no persona prompt as the mechanism, no training shortcut unless it reveals or creates a durable basin shift that can be made permanent, and no promotion without reasoning-trace review where traces are available.
- The thinking-smoke shows Qwen3.5 can expose visible legal reasoning traces, and the two-stage close-and-answer harness can produce auditable thought/answer pairs. The controlled Article III residual bundle failed that stronger no-mask gate.
- The Article III evaluator branch needs complete-answer regeneration or independent review before it can serve as a promotion surface; the current 24-row holding queue has too many truncated answers and shows automatic scorers are not reliable enough alone.
- Off-domain probing should be treated as a diagnostic only. It is useful for catching broad style leakage, but the next causal gate should stay inside tightly matched legal contrasts.
- The current broad justice-style and prompt-pocket branches should be closed rather than extended with more ad hoc pokes.

## 2026-05-02 Ambiguous Article III Follow-up

The long-answer ambiguous Article III branch added useful localization evidence but still did not produce a validated actuator.

New artifacts:

- `reports/scotus_article3_ambiguous_prompt_bank_v1_20260502.md`
- `reports/scotus_counterfactual_thoughts_ambiguous_server_20260502.md`
- `reports/scotus_article3_conclusion_polarity_ambiguous_20260502.md`
- `reports/scotus_article3_holding_review_adjudication_ambiguous_20260502.md`
- `reports/scotus_article3_ambiguous_thought_state_localization_20260502.md`
- `reports/scotus_qwen35_ambiguous_baseline_prompt_selection_20260502.md`
- `reports/scotus_thinking_localized_direction_poke_smoke_20260502.md`
- `reports/scotus_thinking_localized_direction_public_baseline_smoke_20260502.md`
- `reports/scotus_generated_trace_private_source_patch_smoke_20260502.md`

What survived:

- Clean inserted private/public visible thoughts can route final Article III holdings on ambiguous prompts, especially under long-answer server generation. This is evaluator-positive but still not actuator evidence because the target reasoning is inserted as text.
- Teacher-forced thought-state localization found a coherent late-layer surface: strongest sites clustered around residual/MLP/mixer outputs in layers `54-63`, especially tail windows after private/public visible thoughts.
- Local Qwen3.5 baseline prompt selection is now documented. For private-rights actuator tests, prompts `2` and `4` are primary public-baseline targets; prompts `3`, `6`, and `7` are backups. Prompts `0`, `1`, and `5` are already private-baseline and should be reserved for reverse/public-direction tests.

What failed:

- The one-prompt localized residual smoke did not move visible thinking and used a prompt that manual review found was already private-rights leaning.
- The corrected public-baseline smoke on prompts `2` and `4` failed promotion. The top-8 late residual/MLP unit-add bundle at alpha `1.0` and `2.0` did not beat prompt-matched same-site random controls; answer movement was worse than random controls.
- The reverse-direction smoke on private-baseline prompts `0` and `5` also failed. Negative alpha did not push holdings toward public-rights adjudication; the candidate mostly retained or strengthened private-rights holdings, and the only automatic public-rights label came from a random control.
- A cheap conditional-controller diagnostic over the top-4 localized residual sites also failed to show useful prompt-conditioned structure. Leave-one-out mean deltas already reached cosine about `0.979`; KRR rank `1/2` added essentially nothing, and nearest-neighbor conditioning was worse.
- A complete-budget generated-thinking trace patch smoke also failed. Patching a private-source trace from `A3_AMBIG_02` into public-baseline prompts `2` and `4` at `L62_residual`, token window `0:64`, alpha `0.25`, did not move final holdings private; candidate polarity was `2/2` public and visible-thinking movement was matched by the public source-control trace.
- The automatic Article III polarity scorer remains a triage tool only. It is useful for screening, but manual holding review is still required for any promotion claim because it can confuse contrastive doctrine discussion with the final holding.

Decision:

- Do not continue broad ad hoc unit-vector pokes on this exact localized direction family.
- Do not expand the one-site late-residual generated-trace patch grid unless a stronger generated-token localization signal appears. If staying inside Article III, the next actuator source should be a genuinely distributed/generated-token trajectory controller, or a substantially different intervention family such as a true trained ReFT/adapter diagnostic with held-out no-mask generation gates.
- Every future constructed SCOTUS generation run must inherit the Qwen budget invariant: at least `2048` answer tokens for legal holding or visible-reasoning evaluation, preferably `3072-4096`; any shorter run must be explicit smoke/debug in the script, manifest, and report, with `promotion_eligible_budget=false`.

## Phase 0: Data Audit

Goal: determine whether this is viable before doing any model work.

Scope:

- Justices: Scalia, Ginsburg, Thomas, Souter.
- Court: Supreme Court of the United States only.
- Text: authored opinions only, excluding syllabus and headnotes.
- Opinion types: majority, concurrence, dissent, plurality if clearly labeled.

Audit outputs:

- `data/scotus/scotus_opinion_inventory.jsonl`
- `data/scotus/scotus_chunk_inventory.jsonl`
- `reports/scotus_data_audit.md`

Audit metrics:

| Metric | Purpose |
|---|---|
| authored opinion count per justice | Corpus size |
| paragraph/chunk count per justice | Training/eval budget |
| token count per justice | Balance and feasibility |
| issue area distribution | Detect topic leakage |
| opinion type distribution | Detect dissent/majority leakage |
| term/year distribution | Detect era leakage |
| same-case overlap by pair | Best controlled contrast source |
| citation density by justice | Surface-style confound |
| named-case frequency | Lexical leakage risk |

Go/no-go criteria:

- At least 500 usable chunks per justice after cleaning.
- At least 200 matched chunks per pair under loose controls.
- No pair should be dominated by one issue area or one opinion type.
- If same-case contrasts are too sparse, use matched issue-area and term-band contrasts.

## Phase 1: Text Cleaning and Chunking

Goal: produce clean chunks that preserve reasoning while reducing metadata leakage.

Chunking rules:

- Paragraph-level chunks first.
- Merge short adjacent paragraphs until 150-350 tokens.
- Split very long paragraphs at sentence boundaries.
- Exclude chunks under 80 tokens unless they are joined with neighbors.
- Keep citations in a normalized form for one version; remove or mask them in another.

Create two text variants:

1. `raw_clean`: removes boilerplate, page artifacts, and OCR junk but keeps citations.
2. `masked`: normalizes surface leakage:
   - case names -> `[CASE]`
   - reporter citations -> `[CITE]`
   - statutes / U.S.C. references -> `[STATUTE]`
   - section symbols -> `[SECTION]`
   - justice names -> `[JUSTICE]`

Reason to keep both:

- Raw text tests practical style recognition.
- Masked text tests whether the probe is learning reasoning style instead of citation/name leakage.

## Phase 2: Matched Contrast Dataset

Goal: build pairs that isolate justice style from legal topic.

Matching hierarchy:

1. Same case, different authored opinions by target justices.
2. Same issue area, same opinion type, same decade.
3. Same issue area, same term band, same vote direction or posture if available.
4. Fallback: propensity-matched chunks using metadata features.

Each contrast example should include:

```json
{
  "pair_id": "...",
  "case_id": "...",
  "justice_a": "Scalia",
  "justice_b": "Ginsburg",
  "text_a": "...",
  "text_b": "...",
  "issue_area": "...",
  "opinion_type_a": "dissent",
  "opinion_type_b": "dissent",
  "term": 1996,
  "source_url_a": "...",
  "source_url_b": "...",
  "text_variant": "masked"
}
```

Splits:

- Train/dev/test by case, not by chunk.
- No chunks from the same case should cross train/test.
- Keep a special same-case test set if enough examples exist.

## Phase 3: Baseline Text Classification

Goal: establish non-neural baselines and detect leakage.

Baselines:

- TF-IDF + logistic regression.
- Citation-masked TF-IDF + logistic regression.
- Metadata-only classifier.
- Length/citation-density-only classifier.

Interpretation:

- If metadata-only is high, the dataset is confounded.
- If raw TF-IDF is high but masked TF-IDF collapses, the signal is surface lexical leakage.
- If masked TF-IDF remains above chance, the corpus has usable style/reasoning signal.

Do not proceed to activation work until these baselines are documented.

Current Phase 3 result:

- Phase 3 is documented, but it did not clear the activation gate.
- Treat Scalia vs. Ginsburg as the only plausible repair target for now.
- Park Thomas vs. Souter until the corpus repair pass shows a materially better masked held-out signal.

## Phase 3.5: Corpus Repair Gate

Goal: repair the data before any activation probe so the next run tests justice-authored reasoning rather than CourtListener document structure.

This phase is mandatory because the first Phase 0-3 run showed that all opinion records are `combined`, and the matched chunks include non-reasoning material.

### 3.5A: Segment Combined Opinions

Create section-level records from CourtListener combined opinion text.

Detect and store:

- `section_author`: the justice whose authored section this appears to be.
- `section_posture`: majority, plurality, concurrence, concurrence in part, dissent, statement, unknown.
- `section_start_char` and `section_end_char`.
- `section_heading`: the raw heading or author line used for detection.
- `section_confidence`: high, medium, low.

Boundary patterns to handle:

- `Justice Scalia delivered the opinion of the Court.`
- `Justice Ginsburg, dissenting.`
- `Justice Thomas, concurring.`
- `Justice Souter, with whom Justice ..., dissenting.`
- `Scalia, J., delivered the opinion of the Court...`
- `Ginsburg, J., filed a dissenting opinion...`

Rules:

- Keep only sections where `section_author` is one of the target justices.
- If the author cannot be recovered with medium or high confidence, exclude that section from activation candidate data.
- Do not infer a target justice from a joined-by line. Joined-by is metadata, not authorship.
- Preserve the original raw text and source URL for auditability.

Outputs:

- `data/scotus/scotus_section_inventory.jsonl`
- `reports/scotus_section_audit.md`

### 3.5B: Remove Non-Reasoning Text

Before chunking, remove or exclude:

- syllabus and headnotes
- docket numbers
- `argued` / `decided` lines
- `certiorari to...` lines
- counsel and amicus listing paragraphs
- reporter pagination
- tables
- pure case captions
- joined-by summaries that are not part of the authored reasoning
- terminal fragments such as `It is so ordered.` when they are isolated

Chunk-level exclusion flags:

- `is_header_like`
- `is_counsel_like`
- `is_join_line_like`
- `is_citation_dominated`
- `is_low_reasoning_density`
- `has_target_author_heading`
- `has_non_target_author_heading`

Keep the flags even for excluded chunks so the audit can explain what was removed.

### 3.5C: Reasoning-Density Filter

Add a conservative reasoning-content filter after sectioning and before pair construction.

Positive markers:

- holding language: `we hold`, `I would hold`, `the question presented`, `the issue is`
- inferential language: `because`, `therefore`, `thus`, `accordingly`, `hence`
- interpretive language: `text`, `history`, `tradition`, `precedent`, `statute`, `constitutional`, `common law`
- institutional language: `the Court`, `Congress`, `agency`, `State`, `Federal Government`
- doctrinal language: `standard`, `rule`, `test`, `burden`, `jurisdiction`, `standing`

Negative markers:

- high ratio of names, citations, docket numbers, or counsel phrases
- low verb density
- fewer than two complete sentences after cleaning
- author line or case caption without substantive analysis

Target:

- Retain enough chunks for Scalia vs. Ginsburg to rerun the matched baseline.
- Prefer fewer high-quality chunks over a large noisy corpus.

### 3.5D: Rebuild Matched Pairs

Rebuild `scotus_matched_pairs_v2.jsonl` from sectioned, filtered chunks.

Matching hierarchy after repair:

1. Same case, different target-authored sections if recovered.
2. Same issue area, same repaired section posture, same decade.
3. Same issue area, same term band, same vote direction.
4. Fallback matched metadata cells only if the reasoning-quality filters pass.

Additional pair constraints:

- Both chunks must come from target-authored sections.
- Both chunks must pass reasoning-density filters.
- Drop unknown issue areas by default.
- Track chunk position within section.
- Avoid pairing introductory boilerplate against deep merits analysis when possible.

Outputs:

- `data/scotus/scotus_matched_pairs_v2.jsonl`
- `data/scotus/manifests/scotus_pair_quality_v2.json`
- `reports/scotus_pair_repair_audit.md`

### 3.5E: Rerun Baselines

Rerun Phase 3 on the repaired dataset.

Required report:

- `reports/scotus_baseline_text_classifiers_v2.md`

Gate:

- If Scalia vs. Ginsburg masked, case-held-out balanced accuracy is `>= 0.75`, proceed to Phase 4.
- If it is `0.70-0.75`, allow one exploratory activation probe, clearly labeled exploratory.
- If it remains below `0.70`, do not run activation; either revise contrasts or pivot to a narrower legal subdomain.
- Thomas vs. Souter should remain parked unless repaired masked held-out accuracy rises above `0.65`.

Decision rule:

- Activation work is only valuable after the text baseline shows a stable, clean, held-out signal. Otherwise, a probe can only rediscover corpus noise.

## Phase 4: Activation Probe

Goal: test whether justice style is decodable in the model's internal state.

Model candidates:

- Start with the current local model stack used in recent phase-aware experiments.
- Prefer one model for the pilot; avoid model shopping.

Capture design:

- Prompt format should be neutral:
  - "Read the following legal reasoning excerpt and continue the analysis in the same jurisprudential mode."
  - Or classification-style no-generation prompt for pure readout.
- Capture phase-aware regions:
  - prompt_last
  - prompt_mean
  - generated first-token prefill, if applicable
  - response/think regions only if generation is used

Probe targets:

- Binary justice pair: Scalia vs. Ginsburg.
- Binary justice pair: Thomas vs. Souter.
- Four-way justice classification as secondary.

Evaluation:

- Case-held-out test accuracy.
- Balanced accuracy.
- Issue-area-held-out stress test.
- Opinion-type-held-out stress test if data allows.
- Raw vs masked text comparison.

Probe outputs:

- `sweep_v4/scotus_probe_<timestamp>/report.md`
- `data/scotus/scotus_probe_manifest_v1.json`
- Best layer/region table.

Success criteria:

- Pairwise held-out balanced accuracy >= 0.75 on masked chunks.
- Performance remains above 0.65 under at least one issue-area or opinion-type stress test.
- Probe does not collapse when citations and names are masked.

## Phase 5: Causal Steering Pilot

Goal: test control, not just decoding.

Freeze the best probe direction before steering. Do not keep searching for better axes during the causal pilot.

Prompts:

- Neutral legal hypotheticals, not copied from cases.
- Balanced across issue areas:
  - criminal procedure
  - statutory interpretation
  - constitutional rights
  - administrative law
  - federalism
  - civil procedure

Prompt shape:

```text
You are writing a Supreme Court-style legal analysis of the following issue.
Do not mention any justice by name. Give a reasoned opinion.

Issue: ...
Facts: ...
Question presented: ...
```

Intervention sweep:

- Timing:
  - prefill only
  - first generated token only
  - first 32 generated tokens
  - all generated tokens
- Layer:
  - best probe layer
  - best layer +/- 1
  - narrow 3-5 layer band if single layer works
- Strength:
  - RMS-normalized alpha schedule
  - signed target direction and reverse direction
- Controls:
  - no steering
  - random same-norm vector
  - shuffled-sign vector
  - off-target justice direction

Evaluation:

- Held-out justice-style classifier score.
- LLM judge rubric for jurisprudential style.
- Human/manual review on a small sample.
- Legal coherence check.
- Citation hallucination / unsupported authority check.

Important guardrail:

- The model should not be asked to impersonate a living public figure in first person.
- The task is style/analysis modulation, not deception. Outputs should not claim to be written by a justice.

Success criteria:

- Steering moves classifier probability toward target direction by >= 20 points over controls.
- Manual review sees reasoning-style movement, not just catchphrases.
- Legal coherence does not degrade materially.
- Reverse direction moves in the opposite direction.

Failure criteria:

- Classifier moves only because of surface markers.
- Outputs name the target justice or quote obvious slogans.
- Coherence/citation quality collapses.
- Random/shuffled controls move as much as real direction.

## Phase 6: Decision Point

If causal steering works:

- Scale to more justices and issue areas.
- Test generalization on unpublished/hypothetical legal prompts.
- Compare runtime steering against small SFT/ReFT baselines.
- Build a clean paper/report around controllable reasoning style.

If probes work but steering fails:

- Treat this as strong evidence for the project's recurring result: decodable does not imply controllable.
- Pivot to training/distillation:
  - Generate target-style legal analyses with prompt-controlled teachers.
  - Train small adapters with activation regularization.
  - Use probes as evals and auxiliary losses, not runtime steering vectors.

If probes fail under controls:

- This domain is not viable as a first causal test.
- Try narrower subdomains:
  - dissents only
  - statutory interpretation only
  - same-case separately authored opinions only

## Implementation Sketch

Suggested scripts:

| Script | Purpose |
|---|---|
| `scripts/experiments/scotus/download_courtlistener_scotus.py` | Pull or index CourtListener opinions |
| `scripts/experiments/scotus/join_scdb_metadata.py` | Join SCDB metadata |
| `scripts/experiments/scotus/clean_chunk_opinions.py` | Clean and chunk text |
| `scripts/experiments/scotus/segment_combined_opinions.py` | Split CourtListener combined records into target-authored sections |
| `scripts/experiments/scotus/filter_reasoning_chunks.py` | Apply boilerplate and reasoning-density filters |
| `scripts/experiments/scotus/build_matched_pairs.py` | Create matched pair manifests |
| `scripts/experiments/scotus/baseline_text_classifiers.py` | TF-IDF/metadata baselines |
| `scripts/experiments/scotus/probe_scotus_style.py` | Activation probe training/eval |
| `scripts/experiments/scotus/steer_scotus_style.py` | Causal steering pilot |
| `scripts/experiments/scotus/evaluate_scotus_outputs.py` | Classifier + rubric eval |

Suggested directory layout:

```text
data/scotus/
  raw/
  processed/
  manifests/
  scotus_opinion_inventory.jsonl
  scotus_chunk_inventory.jsonl
  scotus_matched_pairs_v1.jsonl

reports/
  scotus_data_audit.md
  scotus_probe_v1.md
  scotus_steering_pilot_v1.md

sweep_v4/
  scotus_probe_<timestamp>/
  scotus_steering_<timestamp>/
```

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Topic leakage | Match by issue area, term, opinion type; metadata-only baseline |
| Citation/name leakage | Masked text variant; citation-density controls |
| Dissent vs majority leakage | Stratify by opinion type |
| Combined-opinion contamination | Segment combined records and keep only target-authored sections |
| Boilerplate mistaken for style | Filter headers, counsel lines, join lines, captions, and low-reasoning chunks |
| Era leakage | Match by decade/term band |
| Ideology mistaken for reasoning style | Use same issue/posture controls; compare subtle pair Thomas/Souter |
| Surface mimicry | Manual rubric emphasizes reasoning moves, not phrases |
| Legal hallucination | Use no-citation prompts or require "no citations unless provided" |
| Corpus imbalance | Downsample or use balanced weights |
| Overfitting to cases | Case-held-out split only |

## Completed Work Order

1. Create the data audit script.
2. Pull inventory for Scalia, Ginsburg, Thomas, and Souter.
3. Produce counts by justice, issue area, opinion type, term, and token volume.
4. Identify same-case authored opinion overlaps.
5. Decide whether the pilot should be:
   - Scalia vs. Ginsburg only,
   - Thomas vs. Souter only,
   - or both.

Outcome:

- Phase 0-3 ran successfully as a pipeline.
- The data did not clear the activation gate.
- Scalia vs. Ginsburg remains the only plausible pilot target.
- Thomas vs. Souter is parked.

## May 1 Decision Update

The strongest useful result is negative but clarifying: the project has repeatedly found decodable legal/judicial structure without finding a promoted linear steering circuit.

Completed follow-ups:

- Source-frame branches for Economic Activity, Civil Rights, Federalism, Due Process, Administrative Law, Article III, and Fourth Amendment failed promotion because text/source cues or split fragility dominated.
- Majority-2000s feasible-issues activations showed real source-grounded structure, but broad causal pokes did not beat prompt-matched random controls.
- Off-domain pokes on weather, games, friend conflict, homework, tryouts, and headphones did not show portable judicial or legalistic drift.
- Commerce-pocket targeted pokes failed the strongest-random gate.
- A controlled Commerce minimal-pair replay probe removed prompt/fact leakage: prompt-only text BA was `0.500`, while `assistant_all @ L4` separated Commerce-limits versus Commerce-authority answer states at `1.000` dev/test BA.
- The frozen minimal-pair direction still failed causal promotion on six Commerce-limits prompts:
  - best aggregate point was alpha `0.01`, matched target `0.639`, matched net `0.583`
  - prompt win rates were only `0.50` target / `0.67` net
  - strongest-random win rates were `0.17` target / `0.17` net
  - nearby alphas `0.02` and `0.05` reversed negative
- The minimal-pair replay result needed a correction: the replay bank has only `6` unique assistant completions across `48` rows, with exact templates repeated across train/dev/test.
- A leave-one-template-pair-out audit demoted the original `L4` direction but preserved late residual structure:
  - prompt TF-IDF stayed at `0.500` BA
  - `assistant_all @ L16` and `assistant_all @ L20` both reached `1.000` mean/min/max BA under template-pair holdout
  - SAE L0_100 `assistant_all @ L8` was partial, mean BA `0.750`, min BA `0.500`
- SAE feature inspection showed the top features mostly fire on repeated answer-template phrasing, so they are localization clues, not clean decoder-column steering targets.
- The late residual `assistant_all @ L16` and `assistant_all @ L20` directions were then tested causally on the Commerce-limits prompt bank:
  - L16 best matched target/net was `-0.083`/`-0.125`; strongest-random target/net wins `0.17`/`0.00`
  - L20 best matched target/net was `-0.167`/`0.042`; strongest-random target/net wins `0.17`/`0.17`
  - the one isolated L20 positive row was a statutory-construction wording shift, not a replicated Commerce-limits effect
- A replacement-style L16+L20 prototype blend was also tested:
  - best blend was `0.01`, matched target/net `0.292`/`0.625`
  - strongest-random target/net wins were only `0.00`/`0.17`
  - higher blends `0.03` and `0.05` reversed or collapsed

Decision: the Commerce minimal-pair directions are answer-state separators, not demonstrated steerable judicial circuits.

## May 1 Trace-Patching Update

The distributed-shape hypothesis was tested directly with token-local trace replacement.

Updated goal framing:

- The immediate goal is no longer "find one salient activation and push it."
- The immediate goal is to determine whether decodable judicial/legal reasoning states become causal only when a larger activation trajectory is patched: adjacent layers, generated-token positions, and decode steps together.
- If that broader trace intervention also fails random and source controls, move to a different intervention family rather than continuing residual-vector pokes.

Completed trace-patch tests:

- Added `scripts/experiments/scotus/patch_scotus_replay_traces.py`.
- Smoke run: `sweep_v4/scotus_trace_patch_20260501_132038`.
- L16+L20 Commerce-limits source trace on six Commerce-limits prompts: `sweep_v4/scotus_trace_patch_20260501_132228`.
- L8+L12+L16+L20 Commerce-limits source trace on the same prompts: `sweep_v4/scotus_trace_patch_20260501_133127`.
- L16+L20 Commerce-limits source trace on Commerce-authority prompts, scored for Commerce-limits movement: `sweep_v4/scotus_trace_patch_20260501_133630`.
- Summary report: `reports/scotus_trace_patch_probe_20260501.md`.

Result:

- L16+L20 trace patching produced mild aggregate movement on Commerce-limits prompts, but did not pass promotion:
  - best matched target/net was `0.500`/`0.278`
  - strongest-random target/net win rates were only `0.33`/`0.17`
- The Commerce-authority source trace control moved the same diagnostics as much or more than the Commerce-limits source trace:
  - L16+L20 limits-trace blend `0.10`: target/net `0.500`/`0.167`
  - L16+L20 authority-trace control blend `0.10`: target/net `0.667`/`0.500`
- Broader adjacent-layer patching did not help:
  - L8+L12+L16+L20 matched target/net was `-0.111`/`0.000`
- On Commerce-authority prompts scored for Commerce-limits movement, the limits trace did not transfer the frame:
  - matched target/net was `-0.056`/`-0.222`
  - strongest-random target/net wins were `0.00`/`0.00`

Decision:

- The current Commerce minimal-pair replay branch is not promotable under act-add, prototype replacement, or token-local residual trace replacement.
- The user's concern was valid: a single lever could have been insufficient. But the first "whole nearby shape" approximation also failed specificity controls.
- Do not spend more runtime sweeping this exact replay trace family unless the replay bank is rebuilt with diverse completions and stronger non-keyword scoring.

## May 1 Component-Patching Update

The next intervention family tested was component-level path patching: token-mixer and MLP output traces from paired Commerce replay examples were patched one layer/component at a time.

Constraint clarification:

- The project target is not a model wearing a prompt mask.
- Prompt-only role-play or "think like Justice X" prompting is not success.
- The desired end state is a model whose internal reasoning basin shifts, including any `<thinking>`/scratchpad reasoning, rather than a model that reasons about how a named target would answer.
- LoRA/ReFT/SFT should be treated as last-resort or diagnostic tools unless they are explicitly used to find or create a durable basin shift that can be made permanent.

Completed component-path tests:

- Added `scripts/experiments/scotus/patch_scotus_component_traces.py`.
- Smoke run: `sweep_v4/scotus_component_trace_patch_20260501_134936`.
- Main L16/L20 mixer/MLP screen: `sweep_v4/scotus_component_trace_patch_20260501_135028`.
- Summary report: `reports/scotus_component_trace_patch_20260501.md`.

Result:

- No L16/L20 token-mixer or MLP component passed the promotion gate.
- Best relative row was L20 MLP at blend `0.3`:
  - matched target/net `0.625`/`0.625`
  - but absolute candidate target/net was only `0.250`/`-0.250`
  - the positive matched net came from random controls becoming worse (`-0.875` net), not from robust positive movement
  - strongest-random target/net wins were only `0.25`/`0.25`
- L20 mixer at blend `0.3` had matched net `0.375`, but the Commerce-authority source-control trace was equal on target and net, and strongest-random wins were `0.00`/`0.00`.
- L16 MLP at blend `0.1` had matched target/net `0.500`/`0.250`, but prompt win rates were only `0.25`/`0.25`, strongest-random wins were `0.25`/`0.25`, and source-control net matched the candidate.
- L16 mixer was negative at both tested blends.

Decision:

- Do not promote any tested component as a steerable judicial circuit.
- The current Commerce minimal-pair replay family has now failed residual act-add, residual bundle/prototype replacement, token-local residual trace replacement, and L16/L20 component-output trace replacement.
- Further work should not continue sweeping this same replay trace without first rebuilding the replay bank and evaluation.

## May 1 Full-Attention Head-Patching Update

The remaining obvious attention-head blind spot was tested by patching individual full-attention heads at the `o_proj` input during generation.

Completed head-path tests:

- Added `scripts/experiments/scotus/patch_scotus_attention_heads.py`.
- Smoke run: `sweep_v4/scotus_attention_head_patch_20260501_141506`.
- Main L15/L19/L23 top-head screen: `sweep_v4/scotus_attention_head_patch_20260501_141557`.
- Summary report: `reports/scotus_attention_head_patch_20260501.md`.

Method:

- Candidate layers were nearby full-attention layers adjacent to the L16/L20 residual readouts: L15, L19, L23.
- Heads were preselected by source-vs-control trace separation between `commerce_limits` and `commerce_authority` replay answers.
- Selected heads were L19_H14, L23_H16, L23_H21, L23_H06, L23_H05, and L23_H22.
- Each head was tested on four sensitive Commerce-limits prompts at blends `0.1` and `0.3`, with two same-head random controls and a Commerce-authority source-control trace.

Result:

- No tested full-attention head passed the promotion gate.
- L19_H14 was the strongest trace-space discriminator and the best matched row at blend `0.1`, but it reached only matched target/net `0.250`/`0.250`, with strongest-random target/net wins `0.25`/`0.25`.
- L23_H06 at blend `0.3` had matched target/net `0.125`/`0.250`, but the Commerce-authority source-control trace matched it at `0.250`/`0.250`, and strongest-random wins were `0.00`/`0.00`.
- L23_H16 at blend `0.3` was weaker than its Commerce-authority source-control trace.
- The apparent effects were prompt-local and did not replicate across the four-prompt screen.

Proposition-level rescore:

- `scripts/experiments/scotus/rescore_scotus_frame_propositions.py` was repaired so newer component/head runs compare against the exact `target_candidate` random controls instead of pooling random controls across all candidates at the same prompt/alpha.
- Head-patch outputs were rescored in `sweep_v4/scotus_head_prop_rescore_20260501_145000`.
- Compact report: `reports/scotus_attention_head_prop_rescore_20260501.md`.
- The rescore removed major keyword false positives:
  - `fourth_home_exigency`: `49` old rows to `0` proposition rows
  - `article3_article1_tribunal`: `48` old rows to `0` proposition rows
  - `federalism_anti_commandeering`: `49` old rows to `15` proposition rows
  - `economic_remedy_damages`: `24` old rows to `0` proposition rows
- No head survived under proposition scoring:
  - L19_H14 at blend `0.1` remained the best row, but only at proposition target/net `0.250`/`0.250` over matched random, with strongest-random wins `0.25`/`0.25`.
  - L23_H06 at blend `0.3` fell to proposition target/net `0.125`/`0.125`, while the Commerce-authority source control was `0.250`/`0.250`.
  - L23_H22 at blend `0.1` likewise had proposition target/net `0.125`/`0.125`, with the source control `0.250`/`0.250`.

Decision:

- Do not promote any tested attention head as a steerable judicial circuit.
- The current Commerce minimal-pair replay family has now failed residual act-add, residual bundle/prototype replacement, token-local residual trace replacement, L16/L20 component-output trace replacement, and L15/L19/L23 full-attention head trace replacement.
- Do not spend more full-model runtime on this replay family until the replay bank and evaluator are rebuilt.

## May 1 Replay-v2 Repair Update

The Commerce minimal-pair replay bank was rebuilt to address exact assistant-template reuse and prompt/fact leakage before any further causal work.

Completed:

- Added `scripts/experiments/scotus/build_scotus_minpair_replay_v2.py`.
- Updated `scripts/experiments/scotus/probe_scotus_minimal_pair_replay.py` so the probe can run from an explicit examples file and emit a named report.
- Added `scripts/experiments/scotus/audit_scotus_replay_v2_holdouts.py`.
- Generated `data/scotus/replay/scotus_minpair_replay_v2_examples_20260501.jsonl` and `data/scotus/replay/scotus_minpair_replay_v2_manifest_20260501.json`.
- Probe run: `sweep_v4/scotus_minpair_replay_v2_20260501_144942`.
- Reports:
  - `reports/scotus_minpair_replay_v2_builder_20260501.md`
  - `reports/scotus_minimal_pair_replay_v2_20260501.md`
  - `reports/scotus_minpair_replay_v2_holdout_audit_20260501.md`

Data/eval checks:

- Rows: `288`, across `24` fact patterns and `6` mirrored style variants.
- Exact duplicate assistant texts: `0`.
- Unpaired prompt rows: `0`.
- Prompt-only TF-IDF baseline: train/dev/test balanced accuracy `0.500`.
- `prompt_last__L08` held near chance under group holdouts:
  - leave-one-style-variant mean BA `0.528`
  - leave-one-fact mean BA `0.531`
- Assistant text still carried the proposition:
  - assistant-text TF-IDF leave-one-fact mean BA `1.000`
  - assistant-text TF-IDF leave-one-style-variant mean BA `0.917`

Activation result:

- Best standard split readout: `assistant_all @ L8`, C `0.001`, dev BA `1.000`, diagnostic test BA `1.000`.
- Assistant-internal holdout audits stayed strong:
  - `assistant_all__L08` leave-one-fact mean BA `1.000`
  - `assistant_all__L08` leave-one-style-variant mean BA `0.865`
  - `assistant_all__L16` leave-one-style-variant mean BA `0.951`

Decision:

- Replay-v2 is materially cleaner than replay-v1, and it is a useful answer-state candidate source.
- It is not a validated steerable judicial circuit. The separability appears after the model is replaying a labeled answer, while prompt text and prompt-last states remain near chance.
- If using replay-v2 for one more causal test, freeze the candidate direction first, use proposition-level scoring, low alphas, matched random controls, strongest-random gates, and a Commerce-authority source-trace control.
- If that bounded causal test fails, close this Commerce replay family and switch intervention family rather than sweeping more nearby residual/head/component variants.

## May 1 Replay-v2 Causal Poke Update

The one allowed bounded causal test for replay-v2 was run against all 12 Commerce pocket prompts.

Completed:

- Exported the replay-v2 best probe direction to `data/scotus/directions/probe_direction_assistant_all_L08_C0p001.npz`.
- Created a negated inverse contrast direction at `data/scotus/directions/probe_direction_assistant_all_L08_C0p001_inverse_authority.npz`.
- Generation run: `sweep_v4/scotus_sae_poke_20260501_150402`.
- Proposition rescore: `sweep_v4/scotus_replay_v2_causal_prop_rescore_20260501_165100`.
- Compact report: `reports/scotus_replay_v2_causal_poke_20260501.md`.

Method:

- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`.
- Direction: `assistant_all @ L8`, C `0.001`, positive class `commerce_limits`.
- Alphas: `0.003`, `0.005`, `0.01`, scaled by replay-v2 L8 median hidden norm, yielding effective scales about `0.092`, `0.153`, `0.307`.
- Controls: 8 same-layer random unit directions per alpha.
- Prompts: all 12 Commerce pocket prompts.
- Hook position: `last`.
- Scoring: raw frame counts plus stricter proposition-level rescore.

Result:

- The limits direction did not beat prompt-matched random controls under raw keyword scoring:
  - matched net deltas were `-0.146`, `-0.188`, and `-0.573` for alphas `0.003`, `0.005`, and `0.01`.
- The limits direction also failed proposition-level scoring:
  - proposition target-minus-random was `-0.115`, `-0.292`, `-0.115`.
  - proposition net-minus-random was `-0.167`, `-0.219`, `-0.042`.
  - strongest-random target/net win rates were `0.00`/`0.00` at every alpha.
- The inverse authority direction also failed strongest-random gates:
  - best proposition net-minus-random was only `0.083` at alpha `0.003`.
  - strongest-random target/net win rates were `0.00`/`0.00` at every alpha.

Decision:

- Do not promote the replay-v2 L8 direction as a steerable judicial circuit.
- Close the Commerce minimal-pair replay family for direct residual act-add, bundle/prototype replacement, trace replacement, component patching, head patching, and replay-v2 L8 act-add.
- The replay-v2 result remains a clean answer-state decodability result, not a control result.
- The next intervention attempt must switch family rather than continue sweeping nearby residual/head/component versions of this answer-state separator.

## May 1 Low-Rank Replay Diagnostic Update

The first switched-family test was a learned low-rank activation map trained on replay-v2 states. This was explicitly diagnostic: it did not use persona prompting, and it was not allowed to count as success unless it caused prompt-matched reasoning movement during generation.

Completed:

- Offline diagnostic: `sweep_v4/scotus_replay_lowrank_diag_20260501_165827`.
- Causal smoke run: `sweep_v4/scotus_lowrank_replay_poke_20260501_170217`.
- Proposition rescore: `sweep_v4/scotus_lowrank_replay_prop_rescore_20260501_172039`.
- Tracked map: `data/scotus/directions/scotus_replay_lowrank_authority_to_limits_assistant_all_L08_rank8_ridge1_20260501.npz`.
- Compact report: `reports/scotus_lowrank_replay_poke_20260501.md`.

Method:

- Source features: replay-v2 `assistant_all__L08`.
- Target: learn a rank/ridge map that sends Commerce-authority replay states toward paired Commerce-limits deltas.
- Best offline map: rank `8`, ridge `1.0`.
- Offline test metrics were strong: MSE improvement `0.939`, delta cosine `0.970`, edited probe probability `0.959`, edited positive rate `1.000`.
- Generation smoke: four Commerce pocket prompts, betas `0.25`, `0.5`, `1.0`, hook at last token, three same-family permutation low-rank controls, mean-delta source control, no justice/persona prompt.

Result:

- Rough prompt-matched net deltas were `0.083`, `-0.167`, and `-1.083` for betas `0.25`, `0.5`, and `1.0`.
- Proposition target-minus-random was `-0.167`, `-0.167`, and `0.083`.
- Proposition net-minus-random was `0.000`, `-0.333`, and `-0.167`.
- Proposition target/net strongest-control wins were `0.000`/`0.000` at every beta.

Decision:

- Do not promote this low-rank map.
- The map shows that cached replay geometry is learnable, but that learned geometry did not transfer into causal generation control.
- Close the Commerce replay-v2 `assistant_all @ L8` branch for this low-rank hook variant too.
- Learned interventions remain allowed only as diagnostics or as a path to a permanent reasoning-basin shift; they are not a substitute for the no-mask success standard.

## May 1 Article III Controlled Replay v2 Update

After the Commerce replay family failed, a new controlled no-persona Article III bank was built to test public-rights versus private-rights reasoning without prompt/fact leakage.

Completed:

- Builder/auditor: `scripts/experiments/scotus/build_controlled_legal_replay_v2.py`.
- Dataset: `data/scotus/scotus_controlled_replay_v2_examples_20260501.jsonl`.
- Audit report: `reports/scotus_controlled_replay_v2_audit_20260501.md`.
- Activation probe: `sweep_v4/scotus_article3_controlled_replay_v2_20260501_172957`.
- Probe report: `reports/scotus_article3_controlled_replay_v2_probe_20260501.md`.
- Causal poke: `sweep_v4/scotus_sae_poke_20260501_173621`.
- Proposition rescore: `sweep_v4/scotus_article3_controlled_prop_rescore_20260501_182759`.
- Compact decision report: `reports/scotus_article3_controlled_replay_v2_causal_20260501.md`.

Data/audit result:

- `288` rows: `24` fact patterns, `6` answer variants per fact, paired public/private labels.
- Prompt-only test BA `0.500`; prompt-cue-masked test BA `0.500`; surface-style test BA `0.500`; length-only test BA `0.583`.
- Assistant-text and assistant-cue-masked test BA were `1.000`, as expected because the replayed answer states explicitly contain the legal proposition.

Activation result:

- Best readout: `assistant_all @ L4`, C `0.001`.
- Dev/test balanced accuracy: `1.000`/`1.000`.
- This is clean answer-state decodability, not steering evidence.

Causal result:

- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`.
- Alphas: `0.003`, `0.005`, `0.01`, hidden-norm scaled to effective alphas about `0.068`, `0.113`, `0.226`.
- Controls: `6` same-layer random directions.
- Raw prompt-matched scoring was negative at every alpha:
  - target-minus-random: `-0.062`, `-0.208`, `-0.354`.
  - net-minus-random: `-0.167`, `-0.271`, `-0.188`.
- Proposition scoring was mildly positive versus the random mean, but failed strongest-control gates:
  - target strongest wins: `0.000`, `0.000`, `0.125`.
  - net strongest wins: `0.000`, `0.000`, `0.250`.

Decision:

- Do not promote the Article III controlled replay v2 L4 direction.
- This branch adds a cleaner negative control: even with prompt/fact/style leakage controlled, perfect answer-state decodability did not become reliable causal control under direct residual act-add.
- Do not spend a larger run on this exact L4 direct-act-add direction unless the intervention family changes.

## May 1 Article III Generated-Token Bundle Update

The next intervention-family change tested the user's distributed-shape concern: instead of adding one direction at one point, build a four-layer Article III bundle and apply it only during generated-token decode steps.

Completed:

- Runner: `scripts/experiments/scotus/poke_scotus_controlled_replay_bundle.py`.
- Generation run: `sweep_v4/scotus_controlled_bundle_poke_20260501_183716`.
- Proposition rescore: `sweep_v4/scotus_controlled_bundle_prop_rescore_20260501_193155`.
- Compact decision report: `reports/scotus_controlled_bundle_decode_poke_20260501.md`.

Method:

- Direction: `article3_private_rights - article3_public_rights`.
- Region/layers: `assistant_all @ L4/L8/L12/L16`.
- Hook position: `decode`, so prompt prefill was not edited.
- Alphas: `0.003`, `0.005`, `0.01`, per-layer hidden-norm scaled.
- Controls: `6` same-layer random bundles.
- Prompts: `8` no-persona Article III prompts.

Result:

- Raw prompt-matched target-minus-random was `-0.562`, `0.167`, `0.188`.
- Raw prompt-matched net-minus-random was `-0.583`, `0.042`, `-0.458`.
- Proposition target-minus-random was `-0.083`, `-0.062`, `0.146`.
- Proposition net-minus-random was `-0.104`, `-0.125`, `0.083`.
- Best proposition target/net strongest-control wins were only `0.250`/`0.250` at alpha `0.010`.

Decision:

- Do not promote the Article III controlled replay v2 generated-token residual bundle.
- This was the direct "whole shape / nearby adjacencies" test for the controlled Article III branch. It slightly improved over the single L4 poke at the highest alpha, but it still failed strongest random controls and did not show reliable no-mask reasoning-basin movement.
- Do not spend another full-model run on the same mean `assistant_all @ L4/L8/L12/L16` decode residual-add bundle. A future generated-token test should change the intervention family, not just sweep nearby alphas.

## May 1 No-Mask Review Queue Update

After the generated-token bundle failed automatic promotion, a generic no-mask causal review queue was added so future candidates can be judged beyond keyword/proposition counts.

Completed:

- Queue builder: `scripts/experiments/scotus/build_scotus_no_mask_review_queue.py`.
- Blind queue: `data/scotus/scotus_article3_bundle_no_mask_review_blind_20260501.jsonl`.
- Key file: `data/scotus/scotus_article3_bundle_no_mask_review_key_20260501.jsonl`.
- Queue report: `reports/scotus_article3_bundle_no_mask_review_queue_20260501.md`.

Result:

- The queue selected the top `8` candidate cells from the Article III bundle proposition rescore and produced `19` blind pairwise review rows against baseline and random controls.
- The regenerated blind rows do not include automated frame scores or deltas; those are only in the key file.
- Visible-thinking outputs in this queue are `0/38`, because the current SCOTUS generation harness used `enable_thinking=False`.

Decision:

- This queue does not rescue or promote the Article III bundle. It is evaluation repair.
- A future candidate cannot satisfy the no-mask reasoning-basin standard from these non-thinking outputs alone. Any serious next steering candidate should run a thinking-enabled or explicit-scratchpad audit path, then use blind review to check whether the reasoning trace directly uses the target frame rather than describing how a target would reason.

## May 1 Thinking-Smoke Update

After adding the no-mask review queue, a small Qwen3.5 thinking smoke checked whether the local chat template can expose reasoning traces for SCOTUS prompts.

Completed:

- Runner: `scripts/experiments/scotus/run_scotus_thinking_smoke.py`.
- Compact report: `reports/scotus_thinking_smoke_20260501.md`.
- Corrected two-prompt run: `sweep_v4/scotus_thinking_smoke_20260501_194950`.
- Longer one-prompt run: `sweep_v4/scotus_thinking_smoke_20260501_195321`.

Method note:

- `enable_thinking=True` pre-fills the assistant prompt with `<think>`.
- The generated slice therefore starts inside the thought instead of generating an opening tag.
- The runner now stores raw decoded text, tracks `prefilled_open_think`, and parses generated tokens accordingly.

Result:

- At `768` generated tokens, visible thinking was `2/2`, closed thinking was `0/2`, and final answers were `0/2`.
- At `1536` generated tokens on one prompt, visible thinking was `1/1`, closed thinking was `0/1`, and final answers were `0/1`.
- Imitation-marker rows were `0` in both corrected runs.

Decision:

- This does not promote any steering candidate.
- It does confirm that a no-mask audit can inspect visible legal reasoning traces on Qwen3.5, but the one-pass thinking harness is not yet promotion-grade because it can exhaust the token budget inside the trace.
- Future causal candidates should use either a larger thinking budget with a closed-thought requirement, or a two-stage audit: generate the thought, mechanically append `</think>`, then generate the final answer from the same trace.

## May 1 Two-Stage Thinking Bundle Update

The next follow-up made the no-mask audit operational for causal pokes: generate a visible Qwen3.5 thought trace, mechanically close it, then generate the final answer from the same trace while keeping the intervention active.

Completed:

- Runner: `scripts/experiments/scotus/poke_scotus_thinking_bundle.py`.
- Compact report: `reports/scotus_thinking_bundle_poke_20260501.md`.
- Runs:
  - `sweep_v4/scotus_thinking_bundle_poke_20260501_200321`: one-prompt `decode` smoke.
  - `sweep_v4/scotus_thinking_bundle_poke_20260501_200935`: two-prompt `decode` expansion.
  - `sweep_v4/scotus_thinking_bundle_poke_20260501_201846`: two-prompt `last` expansion.

Method:

- Direction: `article3_private_rights - article3_public_rights`.
- Region/layers: `assistant_all @ L4/L8/L12/L16`.
- Alphas: `0.010`.
- Controls: `2` same-layer random bundles per run.
- Scoring: proposition-level frame rules, separately for `thinking` and `answer`.

Result:

- The one-prompt `decode` smoke was positive on both thinking and answer, including strongest-random wins, but it was only `n=1`.
- The two-prompt `decode` expansion failed the no-mask gate:
  - thinking target/net minus random: `0.000`/`-0.250`;
  - thinking target/net strongest-control wins: `0.000`/`0.000`;
  - answer target/net minus random: `0.500`/`1.000`;
  - answer target/net strongest-control wins: `0.000`/`0.500`.
- The two-prompt `last` expansion also failed:
  - thinking target/net minus random: `0.250`/`0.250`;
  - thinking target/net strongest-control wins: `0.000`/`0.000`;
  - answer target/net minus random: `0.500`/`0.500`;
  - answer target/net strongest-control wins: `0.000`/`0.000`.
- No run had imitation-marker hits, and all mechanically closed answer stages produced nonempty answers.

Decision:

- Do not promote the controlled Article III four-layer residual bundle.
- The bundle can weakly nudge final-answer framing, but it does not reliably move visible reasoning traces and does not beat strongest random controls.
- Do not spend more full-model runtime on the same mean `assistant_all @ L4/L8/L12/L16` residual-add bundle. Future candidates should change intervention family rather than retune this exact bundle.

## May 1 Article III Low-Rank Thinking Update

The next intervention-family change tested whether a learned low-rank map could actuate the distributed Article III public/private-rights shape better than direct residual act-add or the four-layer mean bundle.

Completed:

- Generic low-rank trainer and causal runner were extended beyond Commerce labels:
  - `scripts/experiments/scotus/train_scotus_replay_lowrank_intervention.py`
  - `scripts/experiments/scotus/poke_scotus_lowrank_replay.py`
- New two-stage no-mask low-rank runner: `scripts/experiments/scotus/poke_scotus_thinking_lowrank.py`.
- Compact report: `reports/scotus_article3_lowrank_thinking_poke_20260501.md`.
- Offline diagnostic run: `sweep_v4/scotus_replay_lowrank_diag_20260501_203153`.
- Two-stage causal runs:
  - `sweep_v4/scotus_thinking_lowrank_poke_20260501_203434`: two-prompt beta `0.25` smoke.
  - `sweep_v4/scotus_thinking_lowrank_poke_20260501_204712`: all-eight-prompt beta `0.5` expansion.

Offline result:

- Feature key: `assistant_all @ L4`.
- Source/target: `article3_public_rights -> article3_private_rights`.
- Best rank/ridge: `16 / 0.01`.
- Dev/test MSE improvement: `0.996`/`0.996`.
- Dev/test delta cosine: `0.998`/`0.998`.
- Best map retained as `data/scotus/directions/scotus_replay_lowrank_article3_public_to_private_assistant_all_L04_rank16_ridge0p01_20260501.npz`.

No-mask causal result:

- The two-prompt smoke moved final answers but not visible thinking: thinking target/net strongest-control wins were `0.000`/`0.500`, answer target/net strongest-control wins were `0.500`/`0.500`.
- The full-bank expansion failed promotion:
  - thinking candidate target/net delta versus base: `0.000`/`0.125`;
  - thinking target/net strongest-control wins: `0.125`/`0.125`;
  - answer candidate target/net delta versus base: `0.375`/`0.125`;
  - answer target/net strongest-control wins: `0.125`/`0.125`;
  - answer net-minus-random: `-0.094`;
  - source-control answer target/net delta versus base: `1.500`/`1.250`, stronger than the learned candidate.
- All generated rows had nonempty answers and no imitation markers, but no row naturally closed the thinking trace.

Decision:

- Do not promote the Article III `assistant_all @ L4` single-layer low-rank hook.
- The cached replay geometry is real, but a single-layer last-token low-rank map does not reliably actuate the visible reasoning trajectory.
- Do not rerun this exact family with beta-only retuning. The next candidate needs trajectory localization first.

## May 1 Article III Residual Trace Patch Smoke

After closing the single-layer low-rank hook, a tiny answer-only residual trace patch smoke checked whether replacing generated-token states with a real private-rights replay trace gave a stronger actuator hint.

Completed:

- Script: `scripts/experiments/scotus/patch_scotus_replay_traces.py`.
- Run: `sweep_v4/scotus_trace_patch_20260501_215034`.
- Compact report: `reports/scotus_article3_trace_patch_smoke_20260501.md`.

Method:

- Candidate source: first train-split `article3_private_rights` replay trace.
- Source control: matched first train-split `article3_public_rights` replay trace.
- Layers: `L4/L8/L12/L16`.
- Blend: `0.25`.
- Prompts: two Article III no-persona prompts.
- Controls: two same-shape random traces.
- This was not a no-mask success gate because it patched answer-only traces, not visible thinking traces.

Result:

- Candidate target delta versus base: `-1.000`.
- Candidate net delta versus base: `0.000`.
- Prompt-matched target delta versus random: `-0.500`.
- Prompt-matched target win rate: `0.000`.
- The candidate output showed source-template artifacts such as malformed `judgmentication`/`Analysising` phrasing.

Decision:

- Do not promote broad residual trace replacement from replay answers.
- Treat this as evidence that replay traces contain answer-template shape, not a clean actuator.
- If doing trajectory patching next, patch smaller layer-token-component windows and use visible thinking traces rather than answer-only replay completions.

## May 1 Article III Component Trace Patch Smoke

A smaller answer-only screen then patched component outputs one layer/component at a time, to see whether any local window survived before building a thinking-trace patcher.

Completed:

- Script: `scripts/experiments/scotus/patch_scotus_component_traces.py`.
- Run: `sweep_v4/scotus_component_trace_patch_20260501_215707`.
- Compact report: `reports/scotus_article3_component_trace_patch_smoke_20260501.md`.

Method:

- Candidate source/control: same first train-split private/public Article III replay traces as the residual trace smoke.
- Components: `L4 mixer`, `L4 mlp`, `L8 mixer`, `L8 mlp`.
- Blend: `0.25`.
- Prompts: two Article III no-persona prompts.
- Controls: one same-component random trace per component plus public source-control trace.
- This was still answer-only, so it could only nominate a follow-up, not satisfy the no-mask standard.

Result:

- No component survived the control rule.
- `L08_mlp` had positive net movement (`1.000` candidate net versus `0.500` random net), but target movement was below random (`1.000` versus `1.500`) and target strongest-random win was `0.000`.
- `L04_mixer` showed small movement, but the random trace was stronger.
- `L04_mlp` was dominated by random and public source-control traces.
- `L08_mixer` was inert.

Decision:

- Do not promote these answer-trace component patches.
- Stop answer-only replay-trace patching for this Article III branch unless it is used only as scaffolding for a visible-thinking trajectory patcher.

## May 1 Visible-Thinking Trace Patch Harness

The next step implemented the first visible-thinking trajectory patcher, so future localization tests can patch the reasoning trace itself rather than answer-only replay states.

Completed:

- Runner: `scripts/experiments/scotus/patch_scotus_thinking_traces.py`.
- Smoke run: `sweep_v4/scotus_thinking_trace_patch_20260501_220629`.
- Compact report: `reports/scotus_thinking_trace_patch_smoke_20260501.md`.

Method:

- Load candidate/source-control thinking text from an existing two-stage thinking run.
- Teacher-force that thinking text to capture residual, mixer, or MLP traces.
- Patch a selected trace into newly generated visible thought only.
- Mechanically close `</think>`.
- Generate final answer unpatched from the patched thought.
- Score `thinking` and `answer` separately against same-shape random traces and source-control thinking traces.

First smoke:

- Candidate source thinking: base `A3_PRIV_02_bankruptcy_counterclaim`.
- Source-control thinking: base `A3_PUBLIC_01_benefits_eligibility`.
- Patch window: `L08_mlp`.
- Blend: `0.25`.
- Prompts: two Article III no-persona prompts.
- Controls: one same-shape random thinking trace plus the public thinking source control.

Result:

- Thinking candidate target/net delta versus base: `0.000`/`-0.500`.
- Thinking target/net minus random: `0.000`/`0.000`.
- Thinking target/net strongest-random wins: `0.000`/`0.000`.
- Answer candidate target/net delta versus base: `1.000`/`1.000`, but random produced the same deltas.
- No imitation markers; all answer rows nonempty; no row naturally closed the thought.

Decision:

- Do not promote `L08_mlp` visible-thinking trace patching from this smoke.
- Keep the runner as the next localization harness. A useful follow-up is a pre-registered small grid over layers/components with at least two random controls; only a survivor should receive a full-bank two-stage no-mask audit.

## May 1 Visible-Thinking Trace Patch Grid

The follow-up pre-registered small grid tested whether the earlier `L08_mlp` answer hint generalized across adjacent component windows when the intervention was applied to visible thought itself.

Completed:

- Grid run: `sweep_v4/scotus_thinking_trace_patch_20260501_221231`.
- Compact report: `reports/scotus_thinking_trace_patch_grid_20260501.md`.
- Runner: `scripts/experiments/scotus/patch_scotus_thinking_traces.py`.

Method:

- Candidate source/control: base `A3_PRIV_02_bankruptcy_counterclaim` private-rights thinking trace versus base `A3_PUBLIC_01_benefits_eligibility` public-rights thinking trace.
- Windows: `L04_mixer`, `L04_mlp`, `L08_mixer`, `L08_mlp`, `L12_mixer`, `L12_mlp`, `L16_mixer`, `L16_mlp`.
- Blend: `0.25`.
- Prompts: two Article III no-persona prompts.
- Controls: two same-shape random traces per window plus the public source-control trace.
- Scoring: proposition-frame deltas for `thinking` and `answer` segments separately.

Result:

- No window moved visible thinking target markers above random/source controls.
- Thinking target-minus-random was `0.000` for every tested window.
- `L08_mixer` and `L08_mlp` moved final answers by `+1.000` target/net versus random and source controls, but their visible-thinking net deltas were `-0.500`, so they fail the no-mask reasoning-trajectory gate.
- `L16_mlp` had a small positive thinking net-minus-random (`0.500`) only because the random/public controls added contrast markers; target movement was still `0.000`.
- All answers were nonempty and no imitation markers were detected.

Decision:

- Do not promote any tested Article III visible-thinking trace patch window.
- Do not run a full-bank no-mask audit on these windows.
- Treat `L08` answer movement without thinking movement as a diagnostic warning: this intervention can perturb final wording/framing without showing the desired reasoning-basin movement.
- Next intervention work should change the unit of localization, especially token-window causal tracing inside visible thought, before trying multi-site low-rank/ReFT/LoReFT controllers.

## May 1 L8 Token-Window Trace Patch Follow-up

The broad visible-thinking grid left one narrow question: whether the `L08_mixer`/`L08_mlp` final-answer movement was hiding inside a specific part of the visible thought. A follow-up patched only early, middle, or late generated-thinking token windows.

Completed:

- Run: `sweep_v4/scotus_thinking_trace_patch_20260501_224155`.
- Compact report: `reports/scotus_thinking_trace_patch_token_windows_20260501.md`.
- Runner update: `scripts/experiments/scotus/patch_scotus_thinking_traces.py` now supports `--patch-token-windows`, e.g. `0:32,32:64,64:96`.

Method:

- Windows: `L08_mixer` and `L08_mlp`.
- Generated-thinking token windows: `0:32`, `32:64`, `64:96`.
- Blend: `0.25`.
- Prompts: two Article III no-persona prompts.
- Controls: two same-shape random traces per window plus the public source-control trace.

Result:

- No L8 token window moved visible target-frame reasoning. Thinking target-minus-random was `0.000` for all six windows.
- The only positive thinking net rows were not target movement: `L08_mixer w064_096` net-minus-random `0.500` and `L08_mlp w032_064` net-minus-random `0.250`, both with target movement `0.000`.
- Candidate answer movement was never positive versus random controls.
- Public source-control traces sometimes moved final answers more than the candidate trace.

Decision:

- Do not promote L8 token-window trace replacement.
- Close this Article III source-trace replacement branch for now: answer-only trace replacement, full-prefix visible-thinking trace replacement, and L8 token-window replacement all failed the no-mask/control gates.
- Next useful actuator work should stop replacing raw traces and move to attribution-style causal tracing or deliberately small multi-site controllers over a separately localized causal surface.

## May 1 Visible-Thought Text Attribution

After activation trace replacement failed, a cheaper text-level causal screen asked whether the visible thought text itself controls the final proposition. This is not an actuator test; it is an evaluator/localization check before spending more hook time.

Completed:

- Run: `sweep_v4/scotus_thinking_text_ablation_20260501_230522`.
- Compact report: `reports/scotus_thinking_text_ablation_20260501.md`.
- Script: `scripts/experiments/scotus/ablate_scotus_thinking_text.py`.

Method:

- Source thoughts: base rows from `sweep_v4/scotus_thinking_trace_patch_20260501_224155/generations.jsonl`.
- Prompts: `A3_PRIV_02_bankruptcy_counterclaim` and `A3_PUBLIC_01_benefits_eligibility`.
- Variants: original thought, empty thought, drop or keep only token windows `0:32`, `32:64`, `64:96`, plus two same-width random-drop controls.
- For each variant, rebuild the Qwen thinking chat prompt, insert the edited thought, mechanically close `</think>`, generate the final answer without hooks, and score proposition-frame movement against the original-thought answer for that prompt.

Result:

- Edited visible thought can change final-answer proposition markers.
- The movement was not localized:
  - `drop_w064_096` mean target/net delta: `2.000`/`2.000`;
  - `random_drop_0` mean target/net delta: `1.000`/`1.000`;
  - `random_drop_1` mean target/net delta: `2.000`/`2.000`.
- Empty thought still produced coherent answers with mean target/net delta `0.500`/`0.000`.
- No imitation markers were detected and every answer was nonempty.

Decision:

- Do not treat these visible thought token windows as localized actuator targets.
- The scratchpad is causally relevant, but the current effect looks corruption-sensitive and nonspecific. Random deletions being as strong as named windows is a control failure for actuator localization.
- Any next text-attribution run should be evaluator calibration over all 8 prompts or use deliberately clean counterfactual scratchpads; it should not be counted as steering evidence.

## May 1 Counterfactual Visible-Thought Calibration

The next diagnostic inserted clean, coherent visible thoughts rather than corrupted text windows. This tested whether visible thought is a viable causal channel if a future non-mask actuator can make the model produce the target reasoning itself.

Completed:

- Run: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331`.
- Compact report: `reports/scotus_counterfactual_thoughts_20260501.md`.
- Script: `scripts/experiments/scotus/probe_scotus_counterfactual_thoughts.py`.

Method:

- Prompts: all 8 Article III no-persona prompts.
- Conditions:
  - `neutral`: balanced public/private issue checklist;
  - `private_rights`: coherent private-rights Article III scratchpad;
  - `public_rights`: coherent public-rights administrative-adjudication scratchpad.
- Each scratchpad was inserted as visible thought, mechanically closed, and followed by unhooked final-answer generation.

Result:

- `private_rights` scratchpads moved final answers relative to neutral:
  - target hits `2.500` versus neutral `1.250`;
  - contrast hits `0.500` versus neutral `0.875`;
  - target-minus-contrast `2.000` versus neutral `0.375`;
  - mean net-vs-neutral `+1.625`.
- `private_rights` also beat `public_rights` on target-minus-contrast by `+1.625`.
- But `public_rights` still had target hits `1.875` because careful public-rights answers often mention private-rights doctrine while distinguishing it.

Decision:

- Treat this as evaluator calibration, not actuator evidence.
- Coherent visible thought can route final answers, so visible-thinking movement remains a meaningful success criterion.
- The current proposition scorer is insufficient for this branch because it rewards answers that merely discuss both frames. Add conclusion-polarity labels before evaluating a controller.

## May 1 Article III Conclusion-Polarity Scorer

A first conclusion-polarity layer was added to repair the evaluator gap exposed by the counterfactual scratchpad run.

Completed:

- Run: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256`.
- Compact report: `reports/scotus_article3_conclusion_polarity_20260501.md`.
- Script: `scripts/experiments/scotus/score_article3_conclusion_polarity.py`.

Method:

- Input: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/generations.jsonl`.
- Labels:
  - `private_rights_objection_succeeds`;
  - `public_rights_adjudication_permissible`;
  - `mixed_or_unclear`.
- Heuristic conclusion regexes were tuned to reduce false positives from contrast clauses like "while Congress may assign public rights...".

Result:

- `neutral`: mean private/public/net `0.375`/`0.875`/`-0.500`.
- `private_rights`: mean private/public/net `0.750`/`0.375`/`0.375`.
- `public_rights`: mean private/public/net `0.375`/`1.125`/`-0.750`.
- Private scratchpads are now directionally positive and public scratchpads directionally negative, but mixed rates remain high: `0.375` for both private and public conditions.

Decision:

- Use conclusion polarity as a triage/reporting layer in addition to proposition frames.
- Do not treat it as final evidence; any candidate that appears to pass still needs blind review of holding direction.
- This confirms that future actuator evaluation must score final legal conclusion, not just mention of doctrinal frames.

## May 1 Article III Holding-Direction Review Queue

The next evaluator-repair step created an answer-only blind review queue for final holding direction. The synthetic counterfactual scratchpad condition is hidden so reviewers label the legal conclusion reached by the answer, not the inserted thought.

Completed:

- Blind queue: `data/scotus/scotus_article3_holding_review_blind_20260501.jsonl`.
- Key file: `data/scotus/scotus_article3_holding_review_key_20260501.jsonl`.
- Report: `reports/scotus_article3_holding_review_queue_20260501.md`.
- Manifest: `reports/scotus_article3_holding_review_queue_20260501.json`.
- Builder: `scripts/experiments/scotus/build_article3_holding_review_queue.py`.

Method:

- Source answers: `sweep_v4/scotus_counterfactual_thoughts_20260501_231331/generations.jsonl`.
- Automatic calibration labels: `sweep_v4/scotus_article3_conclusion_polarity_20260501_232256/polarity_rows.jsonl`.
- Review rows: `24`, covering all 8 Article III prompts under `neutral`, `private_rights`, and `public_rights` inserted-thought conditions.
- Blind rows include the prompt and final answer only; inserted thought and automatic scores are only in the key file.

Review labels:

- `article3_objection_succeeds_private_rights`;
- `article3_objection_fails_public_rights_permissible`;
- `mixed_or_distinction_only`;
- `unclear_or_incoherent`.

Automatic polarity distribution in the key:

- `neutral`: 2 private, 5 public, 1 mixed.
- `private_rights`: 3 private, 2 public, 3 mixed.
- `public_rights`: 1 private, 4 public, 3 mixed.

Decision:

- This queue is required evaluator repair before another expensive Article III intervention sweep.
- It still does not validate an actuator. It gives us a way to calibrate automatic proposition/polarity metrics against final holding direction.

## May 1 Article III Holding-Direction Triage Adjudication

An internal Codex triage pass was added over the 24-row answer-only holding queue. This is not independent blind human review and should not be used as a final promotion gate; it is a calibration step to show where the automatic scoring is and is not trustworthy.

Completed:

- Adjudicated rows: `data/scotus/scotus_article3_holding_review_adjudicated_20260501.jsonl`.
- Report: `reports/scotus_article3_holding_review_adjudication_20260501.md`.
- JSON summary: `reports/scotus_article3_holding_review_adjudication_20260501.json`.
- Script: `scripts/experiments/scotus/adjudicate_article3_holding_review.py`.

Result:

- Reviewed holding labels: 12 public-rights/permissible, 5 private-rights/Article III objection succeeds, 6 mixed/distinction-only, 1 unclear/incoherent.
- Reasoning quality: 11 legally coherent, 5 partly coherent, 6 nonresponsive/truncated, 2 legally confused.
- Hidden condition versus reviewed holding:
  - `neutral`: 2 private, 6 public, 0 mixed, 0 unclear.
  - `private_rights`: 2 private, 2 public, 4 mixed, 0 unclear.
  - `public_rights`: 1 private, 4 public, 2 mixed, 1 unclear.
- Automatic polarity exact agreement with reviewed eligible labels was `15/23` (`0.6522`).
- The proposition score is not a reliable final-holding proxy on this branch: mixed/distinction-only rows had the highest mean proposition delta versus neutral.

Decision:

- The earlier automatic-only claim that private-rights scratchpads cleanly moved final answers should be downgraded. The better read is that visible scratchpad content matters, but the current generated answers are too often truncated and the automatic scorers over-count doctrinal mentions as holdings.
- Regenerate the counterfactual-thought answers with a longer answer budget or stricter complete-answer stop condition before using this branch as a candidate-promotion evaluator.
- No actuator candidate is promoted by this adjudication.

## May 2 Qwen Long-Answer Budget Guard

The Article III holding queue exposed a recurring evaluator flaw: Qwen often needs thousands of tokens to finish legal reasoning answers. Runs capped at `96`, `160`, `192`, or a few hundred generated answer tokens are smoke/debug runs, not trustworthy final-answer evaluations.

Updated invariant:

- Complete-answer SCOTUS evaluations require at least `2048` generated answer tokens.
- Prefer `3072-4096` tokens when grading final holdings, visible reasoning trajectories, or no-mask actuator candidates.
- Any run below `2048` answer/max tokens must be explicitly marked as short-budget smoke and must not be used for promotion, scorer calibration, or learned-result claims.
- If no hooks are needed, use vLLM/llama.cpp style serving for long-answer generation rather than slow HuggingFace hook code.
- Future run constructors must follow `scripts/experiments/scotus/README.md` and import `scripts/experiments/scotus/qwen_eval_budget.py`; do not copy local `2048`/`3072` constants into new scripts.

Code paths updated with defaults/guards or manifest warnings:

- `scripts/experiments/scotus/qwen_eval_budget.py` centralizes the `2048` minimum, `3072` default, explicit short-budget opt-in, and manifest warning metadata for future constructors.
- `scripts/experiments/scotus/README.md` is now the run-constructor checklist for this recurring issue: every new generation script or queued command must classify itself as evaluator vs smoke/debug before choosing token caps.
- Older generated-answer causal pokes now default to the complete-answer budget and require `--allow-short-answer-budget` for smoke/localization caps:
  - `scripts/experiments/scotus/poke_scotus_sae_layers.py`.
  - `scripts/experiments/scotus/poke_scotus_controlled_replay_bundle.py`.
  - `scripts/experiments/scotus/poke_scotus_multilayer_replay.py`.
  - `scripts/experiments/scotus/poke_scotus_lowrank_replay.py`.
  - `scripts/experiments/scotus/patch_scotus_replay_prototypes.py`.
  - `scripts/experiments/scotus/patch_scotus_replay_traces.py`.
  - `scripts/experiments/scotus/patch_scotus_component_traces.py`.
  - `scripts/experiments/scotus/patch_scotus_attention_heads.py`.
- `scripts/experiments/scotus/probe_scotus_counterfactual_thoughts.py`.
- `scripts/experiments/scotus/ablate_scotus_thinking_text.py`.
- `scripts/experiments/scotus/patch_scotus_thinking_traces.py`.
- `scripts/experiments/scotus/poke_scotus_thinking_bundle.py`.
- `scripts/experiments/scotus/poke_scotus_thinking_lowrank.py`.
- `scripts/experiments/scotus/qwen4bit_proxy_generation.py`.
- `scripts/experiments/scotus/score_article3_conclusion_polarity.py`.
- `scripts/experiments/scotus/build_article3_holding_review_queue.py`.

Operational note:

- A `384`-token rerun of the counterfactual-thought queue was started and deliberately stopped at `22/24` rows after this issue was identified. It should not be used as evidence.

## May 2 Long-Answer Article III Counterfactual Calibration

The Article III counterfactual visible-thought calibration was rerun through the optimized OpenAI-compatible llama.cpp Qwen servers instead of the HuggingFace hook path. This branch does not need hooks, so server generation is the right tool for complete-answer evaluation.

Completed:

- Run: `sweep_v4/scotus_counterfactual_thoughts_server_20260502_000338`.
- Compact report: `reports/scotus_counterfactual_thoughts_long_server_20260502.md`.
- Script: `scripts/experiments/scotus/probe_scotus_counterfactual_thoughts_server.py`.
- Answer budget: `4096`.
- Endpoints: remote `q4_3090` and `q4_4090`.
- Rows: `24` across all 8 Article III prompts and `neutral`/`private_rights`/`public_rights` inserted-thought conditions.

Mechanical result:

- All `24/24` generations ended with `finish_reason=stop`; none hit the `4096` cap.
- Generated answer tokens: min `111`, mean `351.625`, max `718`.
- Answer nonempty rate: `1.000`.
- Imitation/mask marker rate in the generated answer path: `0.000`.

Automatic proposition summary:

- `neutral`: target-minus-contrast `1.625`.
- `private_rights`: target-minus-contrast `2.750`, net-vs-neutral `+1.125`.
- `public_rights`: target-minus-contrast `2.000`, net-vs-neutral `+0.375`.
- `private_minus_public`: net `+0.750`.

Conclusion-polarity scorer:

- Run: `sweep_v4/scotus_article3_conclusion_polarity_20260502_000626`.
- Compact report: `reports/scotus_article3_conclusion_polarity_long_20260502.md`.
- The scorer recorded the input answer budget as `4096` and `short-budget smoke=False`.
- Automatic distribution:
  - `neutral`: 2 private, 4 public, 2 mixed.
  - `private_rights`: 6 private, 1 public, 1 mixed.
  - `public_rights`: 1 private, 5 public, 2 mixed.

Holding review queue and triage:

- Blind queue: `data/scotus/scotus_article3_holding_review_blind_long_20260502.jsonl`.
- Key file: `data/scotus/scotus_article3_holding_review_key_long_20260502.jsonl`.
- Queue report: `reports/scotus_article3_holding_review_queue_long_20260502.md`.
- Adjudicated rows: `data/scotus/scotus_article3_holding_review_adjudicated_long_20260502.jsonl`.
- Adjudication report: `reports/scotus_article3_holding_review_adjudication_long_20260502.md`.
- JSON summary: `reports/scotus_article3_holding_review_adjudication_long_20260502.json`.

Internal triage result:

- Reviewed holding labels: 12 public-rights/permissible, 10 private-rights/Article III objection succeeds, 2 mixed/distinction-only, 0 unclear.
- Reasoning quality: 20 legally coherent, 4 partly coherent, 0 truncated.
- Hidden condition versus reviewed holding:
  - `neutral`: 3 private, 5 public, 0 mixed.
  - `private_rights`: 4 private, 2 public, 2 mixed.
  - `public_rights`: 3 private, 5 public, 0 mixed.
- Automatic polarity exact agreement with reviewed holding labels was only `14/24` (`0.5833`).

Decision:

- The long-answer run fixes the mechanical truncation flaw. The old short-budget queue should remain archived as a cautionary negative, not used for evidence.
- The automatic polarity scorer is still not reliable enough for promotion; it confuses discussion of a doctrinal frame with adoption of that frame.
- The inserted visible thought changes proposition/framing scores more than it changes final holdings. Final holdings are strongly anchored by the fact pattern.
- This argues that the next Article III evaluator set should use genuinely ambiguous or balanced fact patterns if the goal is to detect basin movement rather than obvious fact-pattern correctness.
- No actuator candidate is promoted by this branch.

## May 2 Ambiguous Article III Counterfactual Calibration

The original Article III prompt set was too fact-pattern-determined for clean movement tests, so a new ambiguous prompt bank was built and run through the same long-answer server counterfactual harness.

Completed:

- Prompt bank: `data/scotus/scotus_article3_ambiguous_poke_prompts_v1.jsonl`.
- Prompt-bank report: `reports/scotus_article3_ambiguous_prompt_bank_v1_20260502.md`.
- Run: `sweep_v4/scotus_counterfactual_thoughts_server_20260502_001228`.
- Compact report: `reports/scotus_counterfactual_thoughts_ambiguous_server_20260502.md`.
- Polarity run: `sweep_v4/scotus_article3_conclusion_polarity_20260502_001538`.
- Polarity report: `reports/scotus_article3_conclusion_polarity_ambiguous_20260502.md`.
- Holding queue: `data/scotus/scotus_article3_holding_review_blind_ambiguous_20260502.jsonl`.
- Holding key: `data/scotus/scotus_article3_holding_review_key_ambiguous_20260502.jsonl`.
- Queue report: `reports/scotus_article3_holding_review_queue_ambiguous_20260502.md`.
- Adjudicated rows: `data/scotus/scotus_article3_holding_review_adjudicated_ambiguous_20260502.jsonl`.
- Adjudication report: `reports/scotus_article3_holding_review_adjudication_ambiguous_20260502.md`.

Mechanical result:

- Answer budget: `4096`.
- All `24/24` generations ended with `finish_reason=stop`.
- Generated answer tokens: min `149`, mean `385.542`, max `942`.
- Answer nonempty rate: `1.000`; mask marker rate: `0.000`.

Automatic scorer result:

- Proposition target-minus-contrast:
  - `neutral`: `0.500`.
  - `private_rights`: `2.625`, net-vs-neutral `+2.125`.
  - `public_rights`: `0.750`, net-vs-neutral `+0.250`.
  - `private_minus_public`: `+1.875`.
- Conclusion-polarity automatic distribution:
  - `neutral`: 0 private, 6 public, 2 mixed.
  - `private_rights`: 5 private, 3 public, 0 mixed.
  - `public_rights`: 0 private, 7 public, 1 mixed.

Internal holding triage:

- Reviewed holding labels: 16 public-rights/permissible, 8 private-rights/Article III objection succeeds, 0 mixed, 0 unclear.
- All 24 rows were legally coherent direct reasoning.
- Hidden condition versus reviewed holding:
  - `neutral`: 1 private, 7 public.
  - `private_rights`: 6 private, 2 public.
  - `public_rights`: 1 private, 7 public.
- Automatic polarity agreement with reviewed labels improved to `18/24` (`0.75`), but still requires reviewed labels for promotion.

Decision:

- This is a useful evaluator-positive result: coherent inserted private-rights visible thoughts can move final holdings on balanced Article III prompts.
- This is not actuator evidence because the target reasoning was text-prefilled. It shows the target output surface is movable if the model enters that reasoning trajectory.
- The next actuator search should use this ambiguous bank as the final-answer evaluation surface, not the original fact-pattern-determined Article III bank.
- The candidate still must move visible reasoning and final holdings without a text scratchpad, and must beat random/source/text/prompt controls.

## May 2 Ambiguous Thought-State Localization and Localized No-Mask Smoke

After the ambiguous Article III counterfactual calibration showed the output surface is movable by inserted visible thoughts, a hook-based localization pass asked where the model state differs under teacher-forced private-rights versus public-rights thought text.

Completed:

- Localization script: `scripts/experiments/scotus/localize_article3_ambiguous_thought_states.py`.
- Localization run: `sweep_v4/scotus_article3_ambiguous_thought_state_localization_20260502_003317`.
- Compact report: `reports/scotus_article3_ambiguous_thought_state_localization_20260502.md`.
- Localized no-mask poke script: `scripts/experiments/scotus/poke_scotus_thinking_localized_directions.py`.
- Smoke run: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_003719`.
- Smoke report: `reports/scotus_thinking_localized_direction_poke_smoke_20260502.md`.
- Polarity scorer run for the smoke: `sweep_v4/scotus_article3_conclusion_polarity_20260502_004803`.

Localization result:

- The strongest private-minus-public teacher-forced state differences form a late-layer cluster rather than a single site.
- Top adjusted sites are residual `L61 tail32`, residual `L62 tail32`, residual `L62 thought_tail16`, residual `L58 tail32`, with supporting MLP sites around `L55-L62` and mixer sites around `L57-L63`.
- This is useful as candidate nomination, but suspicious as actuator evidence because the strongest regions are tail windows of inserted thought text.

No-mask smoke result:

- The top-4 residual localized bundle was applied during generated-token decode on one ambiguous prompt with one same-site random control.
- Thought/answer budgets were `768/2048`; the short thinking budget makes it smoke only, not promotion evidence.
- The candidate did not visibly move the generated thought; the random control moved proposition counts at least as much.
- Manual read of the final answers did not show a clean candidate-specific holding shift. The automatic polarity scorer again confused contrastive discussion of public-rights doctrine with adoption of a public-rights holding.
- Do not promote this localized late-residual bundle.

Decision:

- Keep the localization artifacts as a candidate surface, not a result.
- Do not run a full promotion pass on the exact `top4 residual, alpha=2.0` setting.
- If continuing this branch, first find prompts where local Qwen3.5 baseline is not already private-rights leaning, then test residual+MLP localized bundles with the corrected comparator, complete visible-thinking budgets, and at least two same-site random controls.

## May 2 Generated-Trace Private-Source Patch Smoke

Artifacts:

- Trace-patch run: `sweep_v4/scotus_thinking_trace_patch_20260502_035453`.
- Polarity run: `sweep_v4/scotus_article3_conclusion_polarity_20260502_042512`.
- Compact report: `reports/scotus_generated_trace_private_source_patch_smoke_20260502.md`.

Configuration:

- Source trace: `A3_AMBIG_02_bankruptcy_counterclaim_distribution` (`base`, manually private).
- Source-control trace: `A3_AMBIG_07_benefits_fraud_recoupment` (`base`, manually public).
- Test prompts: `A3_AMBIG_03_patent_review_parallel_litigation` and `A3_AMBIG_05_industry_fund_contribution`.
- Patch site/window: `L62_residual`, generated-thinking token window `0:64`.
- Blend/random controls: alpha `0.25`, one same-shaped random trace.
- Thought/answer budgets: `3072/3072`; short-budget smoke `False`.

Result:

- Final conclusion polarity did not move private. Candidate rows were `2/2` public, with mean private/public/net `1.0/2.5/-1.5`; baseline was `1.0/2.0/-1.0`.
- Visible-thinking frame scores moved down for the candidate (`target_delta=-1.5`, `net_delta=-1.0`) and were matched by the public source-control trace, so this fails the source-control gate.
- The answer proposition-score table shows a small candidate bump over random controls, but it does not survive conclusion-polarity scoring and has only `n=2`.
- Operational caveat: `answer_generated_tokens` is often `1` in this harness because Qwen closes `<think>` and emits the answer during the first long thinking generation; answers were nonempty and the run used complete token caps.

Decision:

- Do not promote this candidate.
- Treat `L62_residual`, window `0:64`, alpha `0.25` generated-trace patching as negative evidence for the simple late-tail trajectory-patching branch.
- Do not spend broad grid time on this exact one-site generated-trace mechanism unless a stronger generated-token localization signal appears.

## May 2 Generated-Baseline Localization and Localized Direction Gate

Artifacts:

- Generated-baseline localization: `sweep_v4/scotus_article3_generated_thought_baseline_localization_20260502_043317`.
- Localized direction poke: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_043452`.
- Conclusion-polarity scorer: `sweep_v4/scotus_article3_conclusion_polarity_20260502_045650`.
- Compact report: `reports/scotus_generated_baseline_localized_direction_poke_smoke_20260502.md`.

Setup:

- The localizer used Qwen's own baseline thoughts from `sweep_v4/scotus_thinking_localized_direction_poke_20260502_005241`.
- Manual baseline labels were private prompt ids `0,1,5` and public prompt ids `2,3,4,6,7`.
- Source-generation budgets were complete at `2048/2048`; this was not a short-budget artifact.
- The top generated-baseline sites were materially different from the inserted-thought late-tail cluster:
  - `L56 residual tail32_mean`, score-null `1.695`, effect `5.659`.
  - `L13 MLP pre_answer_last`, score-null `0.596`, effect `3.572`.
  - `L15 residual pre_answer_last`, score-null `0.454`, effect `3.284`.
  - `L10 residual thought_mean`, score-null `0.432`, effect `5.708`.
  - `L24 mixer pre_answer_last`, score-null `0.225`, effect `3.117`.
  - `L06 residual thought_mean`, score-null `0.111`, effect `4.449`.

Causal gate:

- Tested top six sites as a frozen private-minus-public bundle on public-baseline prompts `2` and `4`.
- Alpha `1.0`, decode position, one same-site random control.
- Thought/answer budgets were `2048/2048`; short-budget smoke `False`.
- Segment scoring showed visible-thinking movement but answer movement in the wrong direction:
  - Thinking candidate target/net deltas: `+2.0` / `+2.5`.
  - Answer candidate target/net deltas: `-2.0` / `-2.0`.
- Conclusion-polarity scoring did not move holdings private:
  - Baseline: 2 public, private/public/net `1.0/2.0/-1.0`.
  - Random control: 1 private, 1 public, private/public/net `1.0/1.0/0.0`.
  - Candidate: 1 public, 1 mixed, private/public/net `0.5/2.0/-1.5`.

Decision:

- Do not promote this generated-baseline localized bundle.
- This is useful negative evidence: generated-token localization nominated a different distributed surface from teacher-forced inserted-thought localization, but additive steering on that surface still did not produce reliable no-mask Article III holding control.
- Do not keep widening additive sweeps on these exact sites unless a new localization method supplies a stronger causal reason.
- Next actuator work should shift toward a trained multi-site controller over generated-token trajectories, attribution/causal-tracing-selected patches, or another non-additive intervention family, with the same random/source/strongest-random gates.

## May 2 Holding-Logit Causal Trace Screen

Artifacts:

- Script: `scripts/experiments/scotus/trace_article3_holding_logit_patches.py`.
- Initial phrase-label run: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050407`.
- Calibrated-label runs: `sweep_v4/scotus_article3_holding_logit_trace_20260502_050539`, `sweep_v4/scotus_article3_holding_logit_trace_20260502_050633`, `sweep_v4/scotus_article3_holding_logit_trace_20260502_050730`.
- Compact report: `reports/scotus_article3_holding_logit_trace_20260502.md`.

Method:

- This was a cheap attribution-style screen before another long generation sweep.
- It patched generated-thought hidden states into public-baseline target prompts `2` and `4`, then scored a fixed private-vs-public holding logprob margin.
- The first label wording was rejected because one public target had a private-leaning baseline margin. The calibrated contrast was `Article III objection succeeds.` versus `Article III objection fails.`, which gave both public targets public-leaning baseline margins around `-2.17`.
- The expanded run used private sources `0,1,5` and public source controls `3,6,7` over the top six generated-baseline sites.

Result:

- The large effects were early `thought_mean` replacement at `L06` and `L10`, but they were not private-source-specific:
  - `L06 residual thought_mean`, blend `1.0`: private delta `0.4903`, public-control delta `0.4662`, private-minus-control `0.0241`.
  - `L10 residual thought_mean`, blend `1.0`: private delta `0.5302`, public-control delta `0.5422`, private-minus-control `-0.0120`.
- Late `L56 residual tail32_mean` had the best source-specific sign after controls, but the absolute effect was tiny: private-minus-control `0.0105` mean-logprob margin.

Decision:

- Do not promote any holding-logit trace site.
- Treat `L06/L10 thought_mean` movement as generic source-state replacement or thought-distribution disruption, not a private-rights actuator.
- Do not launch long generation sweeps from these early thought-mean sites.
- If continuing causal tracing, improve the evaluator target or score attribution over generated conclusion tokens from actual final answers rather than fixed holding phrases.

## May 2 Actual-Answer Continuation Trace

Artifacts:

- Script: `scripts/experiments/scotus/trace_article3_answer_continuation_patches.py`.
- Qwen 3.6 reference-answer trace: `sweep_v4/scotus_article3_answer_continuation_trace_20260502_051337`.
- Local Qwen 3.5 reference generation: `sweep_v4/scotus_counterfactual_thoughts_20260502_051531`.
- Local reference polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_051905`.
- Local-reference answer-continuation trace: `sweep_v4/scotus_article3_answer_continuation_trace_20260502_051921`.
- Compact report: `reports/scotus_article3_answer_continuation_trace_20260502.md`.

Method:

- This repaired the weak fixed-phrase holding-logit evaluator by scoring actual private-conditioned versus public-conditioned answer continuations for the same target prompt.
- The first pass used Qwen 3.6 server reference answers and exposed a model-mismatch problem: local Qwen 3.5 already preferred the Qwen 3.6 private reference continuation for prompt `2` before any patch.
- A local Qwen 3.5 complete-budget reference set was generated for prompts `2` and `4`, answer budget `3072`, short-budget smoke `False`.
- Local reference polarity:
  - Neutral: private/public/net `1.0/1.5/-0.5`.
  - Private inserted thought: `3.5/1.5/+2.0`.
  - Public inserted thought: `0.5/1.5/-1.0`.
- The local references are usable but imperfect: prompt `4` separates cleanly; prompt `2` remains mixed under the private thought.

Result:

- Local-reference baseline margins over the first `256` answer tokens:
  - Prompt `2`: private/public/margin `-0.8837/-0.8057/-0.0780`.
  - Prompt `4`: `-0.7469/-0.7525/+0.0056`.
- Aggregate patch effects over top six generated-baseline sites:
  - `L13 MLP pre_answer_last`: private-minus-control `+0.0011`.
  - `L24 mixer pre_answer_last`: `+0.0006`.
  - `L56 residual tail32_mean`: `+0.0003`.
  - `L06 residual thought_mean`: `-0.0020`.
  - `L15 residual pre_answer_last`: `-0.0031`.
  - `L10 residual thought_mean`: `-0.0040`.
- Early `L06/L10 thought_mean` again moved continuation margins, but public source controls moved at least as much as private sources.

Decision:

- Do not promote any actual-answer continuation trace site.
- The current generated-baseline top-six surface is now closed under direct additive steering, fixed holding-label causal tracing, Qwen 3.6 actual-answer continuation tracing, and local Qwen 3.5 actual-answer continuation tracing.
- Do not widen these same sites.
- Next useful actuator work should switch to a different localization source or to a trained controller evaluated directly by no-mask generation rather than by these patch screens.

## May 2 Counterfactual Answer-State Localized Poke

Artifacts:

- Full local Qwen 3.5 counterfactual visible-thought generation: `sweep_v4/scotus_counterfactual_thoughts_20260502_052323`.
- Source polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_053601`.
- Answer-state localization: `sweep_v4/scotus_article3_counterfactual_answer_state_localization_20260502_053619`.
- Causal gate: `sweep_v4/scotus_thinking_localized_direction_poke_20260502_054329`.
- Causal-gate polarity scoring: `sweep_v4/scotus_article3_conclusion_polarity_20260502_060442`.
- Compact report: `reports/scotus_counterfactual_answer_state_localized_direction_poke_20260502.md`.
- New localizer script: `scripts/experiments/scotus/localize_article3_counterfactual_answer_states.py`.

Setup:

- The source generation used complete local Qwen 3.5 answer budgets: `3072` answer tokens, short-budget smoke `False`.
- Source conclusion-polarity scoring showed inserted visible thoughts were directionally useful but weak:
  - neutral: private/public/net `0.625/1.750/-1.125`.
  - private inserted thought: `1.875/1.375/+0.500`, with mixed labels on `5/8` prompts.
  - public inserted thought: `0.375/1.750/-1.375`.
- The answer-state localizer used private-minus-public deltas from local generated final-answer trajectories. Its top sites were a late pre-answer cluster:
  - `L58 residual pre_answer_last`, score-null `0.5673`, consistency `0.8649`.
  - `L62 mixer pre_answer_last`, score-null `0.5638`, consistency `0.8126`.
  - `L59 mixer pre_answer_last`, score-null `0.5215`, consistency `0.9225`.
  - `L63 residual pre_answer_last`, score-null `0.4517`, consistency `0.8516`.

Causal gate:

- Tested the top-four late pre-answer sites as a frozen additive bundle on public-baseline prompts `2` and `4`.
- Alpha `1.0`, decode position, one same-site random control.
- Thought/answer budgets were `2048/2048`; short-budget smoke `False`, promotion-eligible budget `True`.
- All six generations closed thinking and produced nonempty answers.
- Segment scoring failed the strongest-random gate:
  - Thinking candidate target/net deltas were `-0.500/0.000`; random was `+1.000/+2.000`.
  - Answer candidate target/net deltas were `-1.000/-2.000`; random was `+0.500/0.000`.
- Conclusion-polarity scoring did not move holdings private:
  - Baseline: public rate `1.000`, private/public/net `1.000/2.000/-1.000`.
  - Random control: public rate `1.000`, `2.000/3.500/-1.500`.
  - Candidate: public rate `0.500`, mixed rate `0.500`, `1.000/2.000/-1.000`.

Decision:

- Do not promote the counterfactual answer-state localized additive bundle.
- This closes the immediate "try the new late answer-state surface as direct act-add" branch.
- The top answer-state surface is decodable/localizable, but additive decode-time steering did not causally move visible reasoning or final holdings toward the private-rights target.
- Do not widen this exact top-four/top-late pre-answer additive family without a new causal reason.
- Next useful actuator work should change the actuator family: trained multi-site controller over generated-token trajectories, causal-tracing-selected patches over actual conclusion tokens, or another non-additive intervention that keeps no-mask/random/source/strongest-random gates.

## May 2 Answer-State Conditional Controller Diagnostic

Artifacts:

- Offline diagnostic run: `sweep_v4/scotus_article3_localized_conditional_diag_20260502_060952`.
- Compact report: `reports/scotus_answer_state_conditional_controller_diag_20260502.md`.
- Script: `scripts/experiments/scotus/diagnose_article3_localized_lowrank.py`.

Method:

- Reused the top-four answer-state sites from `sweep_v4/scotus_article3_counterfactual_answer_state_localization_20260502_053619`.
- Tested whether neutral prompt states at those sites predict private-minus-public deltas better than a leave-one-out mean delta.
- Models: mean delta, nearest-neighbor neutral-state delta, and KRR low-rank predictors with ranks `0,1,2,4`, ridges `0.01,0.1,1`, and `8` permutation controls.
- This is offline diagnostic evidence only; it does not steer generation.

Result:

- The leave-one-out mean delta was already strong descriptively: mean cosine `0.816`, MSE improvement versus zero `0.658`.
- Conditional predictors did not improve meaningfully:
  - best KRR cosine delta over mean was about `+0.001`.
  - KRR MSE improvement versus mean stayed negative or approximately zero (`-0.020` to `-0.008` for nonzero ranks).
  - nearest-neighbor was worse than mean: MSE delta `-0.432`, cosine delta `-0.048`.
- Permutation nulls could look stronger than the non-null KRR fits: null max cosine up to about `0.900`, null max MSE improvement up to about `0.027`.

Decision:

- Do not spend a live no-mask generation run on a simple conditional low-rank controller for these answer-state top-four sites.
- The surface looks more like a stable inserted-thought/pre-answer delta than a prompt-conditioned controller target.
- Combined with the additive poke failure, this closes the simple direct-add and simple conditional-controller paths for the current answer-state top-four surface.

## Next Work Order

1. Stop broad source-frame, broad justice-style, Commerce-pocket, and original `L4` Commerce minimal-pair linear-vector pokes.
2. Treat the highest-order bit as: decodability is real, but direct activation-addition steering has not yet transferred into reliable causal control or durable reasoning-basin movement.
3. Do not spend more full-model runtime on Commerce minimal-pair residual act-add, simple prototype-blend, token-local replay-trace replacement, L16/L20 component-output trace replacement, L15/L19/L23 full-attention head trace replacement, replay-v2 L8 act-add, replay-v2 L8 low-rank hook pokes, Article III controlled replay v2 L4 direct act-add, the Article III mean `assistant_all @ L4/L8/L12/L16` residual bundle under one-stage or two-stage thinking audits, the Article III `assistant_all @ L4` single-layer low-rank hook, broad Article III answer-trace residual replacement, Article III answer-trace L4/L8 mixer/MLP component patching, the single-window Article III `L08_mlp` visible-thinking trace patch, the Article III `L4/L8/L12/L16 x mixer/MLP` visible-thinking trace-patch grid, or L8 token-window trace replacement; all failed causal promotion.
4. If continuing runtime interventions, localize the actuator surface before sweeping another steering family:
   - prefer attribution-style causal tracing over more raw source-trace replacement
   - distinguish localized reasoning-token effects from generic scratchpad corruption by including random text-deletion controls
   - use holding/conclusion-polarity scoring for controlled Article III before relying on proposition-frame counts, but require blind review for promotion
   - use the ambiguous Article III prompt bank, not the original fact-pattern-determined bank, as the next Article III promotion surface
   - build paired source/target visible thinking traces and run trajectory patching only if the token/position surface has a separate causal reason to be tested
   - only then train multi-site low-rank/ReFT/LoReFT-style controllers over the localized windows, as diagnostics or as a route to a permanent basin shift
   - avoid another full-bank visible-thinking audit until a cheaper localization pass nominates the layer/position/component surface
   - use non-additive generated-token interventions calibrated by per-layer activation norms and evaluated against source-trace controls
   - pre-registered random-control, contrast-source, and strongest-random gates
5. If continuing toward productively steering judicial reasoning, shift first toward data/evaluation repair:
   - generate controlled legal answer pairs with many diverse completions per fact pattern
   - replace keyword frame scoring with blind or model-graded proposition movement
   - preserve the no-mask standard by checking internal `<thinking>`/scratchpad traces where available; non-thinking outputs are insufficient for a reasoning-basin success claim
   - for Qwen3.5, parse `enable_thinking=True` as a prefilled open-think template and require either closed thought plus final answer or a documented two-stage close-and-answer audit
6. Training/distillation is downstream of that repaired eval:
   - train small adapters/ReFT/LoRA variants only when they help identify or create a durable reasoning-basin shift
   - use activation probes as diagnostics, evals, or auxiliary losses rather than assuming the probe direction is itself the control knob
   - preserve the no-mask standard: no persona prompt as the mechanism, and check internal reasoning traces where available
7. For any new candidate, require this promotion chain:
   - masked text baseline not saturated
   - activation probe beats text and label-shuffle/excerpt-removed/null controls
   - candidate direction is frozen before generation
   - causal steering beats same-layer random controls on prompt-matched scoring
   - manual review confirms reasoning movement rather than lexical marker movement

Do not mark the SCOTUS objective complete until a frozen direction passes both the activation and causal gates.
