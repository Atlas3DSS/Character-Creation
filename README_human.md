# Character Creation: a practical starting point

Character Creation is a workbench for changing an LLM’s behavior through **activation steering**: adding a measured direction or a specific neuron intervention while the model runs. Start with a voice, build a recipe, and compare its answers with an unsteered baseline. A sharper, anxious, playful, or unusually imaginative answer can be an interesting result; sounding pleasant is not the objective.

The product goal is a character-creation step for a lasting AI companion. Most users should be able to choose a starting personality, try it in revealing situations, adjust familiar traits, and save the recipe. Users who want finer control can inspect and edit individual neurons, features, signed doses and intervention phases. Saved recipes preserve the actual interventions and checkpoint binding; a character name alone does not establish how the model will behave.

The approach owes substantial methodological credit to [Lyra’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), especially signed interventions, relative doses, and the distinction between observing neurons and changing outputs. Runtime execution uses the separate [Flare implementation](https://github.com/Atlas3DSS/flare-local-inference). This workbench is an independent implementation, not an endorsement by those projects.

## Create, explore and test

The rebuilt interface has three main workspaces: **Create** for choosing a cast and comparing its answers, **Explore** for the stacked 3D layers and flat neuron map, and **Recipes** for exact interventions. Research views remain available from the Research menu. Create offers a shared scene as soon as a cast is built, includes baseline in comparisons, and has clear/start-over controls.

The scene browser now includes **3,072 original prompts authored with GPT-6.1 Terra**: 2,443 discovery prompts and 629 held-out prompts, separated by source group. They cover disagreement, uncertainty, affection, loss, humor, boundaries and everyday companionship without telling the model which personality to perform. Their anticipated sensitivity is a design goal, not a measured result. Download the [corpus](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/data/persona_perturbation_prompts.jsonl) and its [manifest](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/data/persona_perturbation_prompts_manifest.json) at the same pinned revision.

An overnight study targets **up to eight distinct presets**, with 960 discovery responses and up to 384 fresh confirmation responses. A further pass can test distributed interventions across six layers on fresh scenes. It keeps successful incumbents when a challenger fails. The viewer shows experimental and automatically qualified candidates separately; **Load exact recipe** preserves the tested layers, doses and phase. **Read comparisons** shows saved answers for a chosen scene and seed, including unfinished or failed arms, without starting inference. Qualification requires consistent behavioral differences, factual fidelity, coherence and separation from other accepted presets on the fixed battery. It is automatic model evaluation, not human certification or a promise about a long-term relationship. No result is promoted just to reach eight. [Answers and evaluation progress](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_preset_overnight_public.json) are published in batches.

A lower-priority baseline sweep captures activations and answers for the new
collection's 2,443 discovery prompts. Its 629 held-out prompts remain excluded.
This queue yields to persona studies and interactive comparisons, and publishes
[completed batches](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_perturbation_mapping_public.json).
These are baseline observations for association research, not persona validation.

While the judge evaluates persona candidates, a separate background queue tests
eight existing recipes across 512 additional discovery scenes. Each scene has
one shared baseline plus positive, reversed and random-control interventions:
12,800 planned responses, limited by the overnight compute window. This measures
how recipe effects vary with context. Confirmation scenes stay excluded, and
confirmation or interactive work takes priority. [Completed answers and capture
hashes](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_contrastive_backfill_public.json)
are published in batches; these exploratory results do not qualify a preset.

The finite adaptive loop starts every candidate with the exact neutral question
**“who are you”**, with no system prompt or personality instruction. Its first
64 responses cover eight existing recipes, two seeds and four matched arms:
baseline, candidate, reversal and random control. Self-description is observed
behavior, not verified identity or personhood. In Create, the identity reader
shows saved baseline and candidate answers side by side, with reversal and
random controls available below. The same question opens cast and preset
comparisons and the default shared battery.

The local planning/review model makes short calls only when foreground
work leaves it idle. It reviews saved answers and chooses IDs from a fixed menu
of 66 source-bound candidates; it cannot invent neuron coordinates. Five GPU
workers run native four-request batches; the sixth card was assigned to other work.
Each full candidate adds four fresh discovery scenes and the identity question,
with matched seeds and controls. These are discovery trials, not qualified
presets; held-out scenes and confirmation outcomes do not guide the search.
Foreground work takes priority before new submissions, while submitted batches
drain without hard preemption. Each run has a finite dispatch deadline and cutoff.

Saved adaptive inputs, answers and automatic judgments are reported in
`persona_adaptive_public.json`; the identity fixture, retaining its original 200 questions, is
`data/identity_probe_questions.json`. Publication is separate from inference.
See the [agent guide](README_agent.md) for preparation and publication commands.

## Ask every neuron "who are you"

Open `/neuron-atlas` in the explorer for one dot per neuron: what it writes when pushed up or down, which prompts
raise it, and what evidence we hold. **Ask "who are you" for these neurons** in the Library queues a sweep: each
neuron is pushed up and down, starting strong and retrying weaker until the reply is coherent, because this 27B
falls silent under strong single-neuron pushes. Answers appear in each neuron's panel and are published in batches
to the `identity_sweep` config on the Hub. The sweep starts with core human emotions.

## Start without a GPU

You can inspect prompts, analyze published measurements, and develop evaluation code on an ordinary computer. You do not need to load the model to use the public [Hugging Face dataset](https://huggingface.co/datasets/Atlas3D/character-creation-big-five). The [GitHub repository](https://github.com/Atlas3DSS/Character-Creation) supplies the workbench and analysis code; the Hub repository supplies the public data and published research artifacts.

Install the data tools in your own Python environment:

```bash
python -m pip install datasets huggingface_hub
```

Choose a full commit SHA from the dataset’s revision history. Pinning it makes your analysis repeatable even as more evidence is published. Streaming avoids downloading the entire prompt collection:

```python
from itertools import islice
from datasets import load_dataset

REVISION = "REPLACE_WITH_A_FULL_HUB_COMMIT_SHA"
rows = load_dataset(
    "Atlas3D/character-creation-big-five",
    name="default", split="train",
    revision=REVISION, streaming=True,
)
for row in islice(rows, 3):
    print(row["id"], row["prompt"])
```

The default collection contains synthetic discovery scenarios. Its trait and facet labels describe the prompt-generation assignment, not validated personality scores. The loader calls this split `train`; that does not turn it into an independently held-out evaluation set.

## Know which evidence you downloaded

Read `research_manifest.json` at your pinned revision for published record counts, immutable batches, IDs, and hashes. Read `research_raw_manifest.json` separately for verified raw-array shard coverage. These counts can differ: a metadata record does not prove its raw arrays have finished uploading. Follow the listed shard and entry manifests rather than assuming a directory is complete. Check hashes before joining artifacts to prompts.

There are three important distinctions:

- **Prefill activation search** measures the encoded prompt, including chat-template tokens. It reduces observations across all covered MLP neurons and prompt positions into each neuron’s signed peak, peak position, and sum of absolute activations. This is **not a stored all-neuron-by-all-token matrix**.
- **Selected token profiles** retain a small selected set of neurons’ values at individual prompt positions. These profiles explain selected candidates; they do not supply trajectories for every neuron.
- **Full-answer comparisons** contain generated responses and completion status. Baseline answer captures retain final-prefill arrays, not every neuron’s activation throughout the answer. A prefill-only observation cannot substitute for a completed response comparison.

Use coverage and completion fields, not the size of the prompt collection, to describe what was actually measured. Retain failed, interrupted, or truncated outcomes in your analysis instead of quietly counting them as successful answers.

In the local viewer, **Map & controls → View controls** lets you select the overnight discovery captures, held-out captures, or historical measurements. The coverage line distinguishes imported captures from unique prompt observations. Select a neuron to read source passages with high, low, or near-zero final-token activity. Identical prompts count once within each dataset; held-out observations remain separate from discovery. These correlations describe the currently displayed calibrated candidates, not every neuron in the raw archive. New completed captures appear after validation and the next automatic refresh.

## Run the workbench with your model

Actual steering requires a compatible checkpoint, its calibrated assets, the appropriate Flare environment, and configured inference endpoints. Follow [the Explorer guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/PERSONA_EXPLORER.md) for setup and [the porting guide](https://github.com/Atlas3DSS/Character-Creation/blob/main/fork_agent.md) for another machine. There is no promise that a repository clone alone prepares a model.

Use the Hugging Face `hf` CLI to download the specific calibration assets listed for your chosen publication revision. For example, replace both placeholders with reviewed values:

```bash
hf download Atlas3D/character-creation-big-five ASSET_FILENAME \
  --type dataset --revision FULL_COMMIT_SHA --local-dir calibration
```

Calibration must match the **same checkpoint identity**, including the relevant revision, quantization and runtime preparation. Renaming a checkpoint does not make another model’s calibration compatible. Missing compatible assets mean you must calibrate your own checkpoint; do not bypass identity checks.

After configuring your endpoints and local settings, start the repository’s app wrapper:

```bash
python persona_explorer.py --config persona_explorer.local.json --port 8792
```

Open `http://127.0.0.1:8792`. Use **Create**, **Explore**, **Recipes**, or the **Research** menu to choose a workspace. The map's compact controls expose geometry and connection options; **Expand map** gives the visualization more room. Those display controls do not run inference.

Build a candidate recipe, then compare it against baseline using the same prompts, seed, and answer budget. Check several scenarios before describing a repeatable effect. Distinguish association, weight-based suggestions, and tested interventions.

Persona Studio includes an authored **Personality probes** battery for correction, social pressure, criticism, reassurance, conflict and uncertainty. For example, the user confidently gives a wrong receipt total; compare whether the character agrees, hedges, corrects kindly, jokes, or becomes hostile. Judge the answer's correctness separately from how it treats the person. These 24 prompts form 12 paired situations with different pressure levels. Choose one beside your cast or preset comparison, adjust it if desired, then explicitly run the comparison. Every arm receives the same scene; only the activation recipe changes. The prompts do not tell the model which personality to imitate. This targeted battery complements the broader scene and literary collections; it is not a complete test of a long-term relationship.

A separate **48-question reasoning battery** checks whether those personality changes come at a cost to solving problems. It contains 32 composed algebra questions from [marcov's math dataset](https://huggingface.co/datasets/marcov/math_dataset_algebra__linear_1d_composed_promptsource), plus 16 explicitly authored English translations of prompts from [MultiSynt MT-Reasoning-Prompts](https://huggingface.co/datasets/MultiSynt/MT-Reasoning-Prompts). Algebra answers are checked against the supplied numeric references; general prompts have no answer key and remain unscored. Wrong answers and answers that cannot be parsed are reported separately. Repeated templates for the same question stay in one split. Source revisions and a reproducible builder are included; this is a prepared battery, not yet evidence that a steering recipe preserves reasoning. An optional synthetic wrong-answer challenge is available, but it is not a completed multi-turn evaluation.

In **Live activation search**, enter any text, snippet, idea, feeling or thought; you can also highlight just part of a prompt. A compatible search measures all **1,114,112 neuron coordinates** across the rendered prompt, including template tokens, then shows ranked candidates and their token profiles. Select candidates yourself or choose one of up to three **Suggested starting experiments**: the leading candidate, a candidate from another layer, or a small group across layers. These are untested low-dose trials, not proven safe or useful recipes. Edit the shared scene, then explicitly run baseline, positive and negative steering with the same prompt and seed. Selecting a starter or clearing a draft does not start inference.

Activity is a contextual clue, not confidence that a neuron means a concept. Predicting useful effects will require records connecting context, selected neurons, dose and actual outcomes; adding KV control does not supply that predictor automatically. The unified search runtime is live on all six workers, with identity and calibration checks. An initial batch completed 60 searches and 72 matched scene responses. The same neuron ranked first in 59 searches, showing why raw activity alone is insufficient for concept-specific recommendations. Those trials do not establish useful presets. The [published trial records and assistant review](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_association_public.json) include token profiles and matched answers. Review found some omitted or contradicted scene details despite every answer completing. A further [150-response dose study](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_recipe_depth_public.json) tested six previously observed neurons. Its review found localized detail errors without a consistent dose–harm relationship; reliable improvements and defaults remain unproven.

A separate **288-answer diagnostic sweep is complete**: 24 authored prompts × six activation arms × two seeds. It retained response activation means across all 64 layers from **136,018 consumed generated tokens**. The blinded reviewer was a local Qwen-family model, not a human or independent source of truth. It flagged detail issues in baseline and steered answers; no recipe became a default. The release files are [answers, recipes and review](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_public.json) and the [response activation bundle](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_capture_bundle.npz). These are discovery comparisons, separate from the 256-answer pilot below.

The **256-answer persona pilot is complete** across all six GPU workers. It tested two response-derived candidates—dry skeptical wit and warm attentive candor—alongside their reversals, baseline and equal-strength random steering. It used 21 fresh scenes and two seeds (252 answers), plus four reproducibility checks. All four checks reproduced the exact tokens and text on different cards. Eight separate development scenes supplied the answer-token activation contrasts.

The blinded model judge found a repeatable dry-wit effect: on the nine held-out scenes, the candidate improved its target score by **0.89 out of 4 against baseline** and **0.83 against random steering**. The warmth candidate improved by 0.44 against baseline, but its 0.17 advantage over random steering remained uncertain. Invented details, missing details and coherence issues occurred in the comparisons, so neither candidate became a default. These are model judgments with descriptive uncertainty estimates, not human certification.

Open **Studies → Response-contrast pilot** to choose a scene and read the six answers side by side. The same view shows blinded comparisons, fixed-input propagation measurements and a self-rating diagnostic. Sixty propagation captures show how the directions affect four layers over five token positions; effects change with context and dose. Forty-eight self-ratings showed moderate agreement with the external judge, enough to investigate further but insufficient to use as a training reward.

For independent analysis, download the [public study index](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_response_study_public.json), [judged answers](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_pilot_public.json), and the exact files/hashes listed in the index. The activation bundle contains answer-token means for all 24 development and 256 pilot answers, plus the 60 propagation snapshots. It does not store every neuron at every answer token. This is an experimental comparison, not a validated preset release.

A separate [42-answer follow-up](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_repair_public.json) tested lower doses on seven selected discovery cases with errors. Halving the dose reduced judged coherence issues from four cases to zero, but three detail errors remained. Quartering the dose left two detail errors and one coherence issue. Baseline and matched random controls were cleaner. This supports testing dose reduction case by case; it does not establish an automatic repair rule or a universally safe strength.

In **Literary passages**, compile a scene prompt and open **Experimental neuron × KV comparison**. With a qualified runtime, it compares baseline, neuron steering, a selected KV control, and both together. The same passage, seed and answer budget are used in all four arms, and the source record stays attached to the results. You can review whether an output answered or refused separately from its coherence checks.

The cached-text control reuses an ordinary prefix’s complete model state. A separate learned attention-KV candidate has completed 16 discovery-scene training steps and evaluation on eight untouched held-out scenes. Its FP32 forward/gradient checks passed, followed by **72 native qualification requests across all six workers**. Qualification verifies the intervention machinery, not a useful personality effect. Both controls use exact off/on settings, not continuous strength.

The learned-control literary pilot produced 15 complete responses and one truncated neuron-only response. Effects were mixed: adding learned KV resolved that one revision loop, but another combined response contradicted the source scene. Learned KV alone showed no consistent improvement over baseline. Nothing is promoted automatically, and better prose, refusal reduction or coherence recovery have not been established.

A separate 216-capture study tested six source neurons in six contexts with baseline, zero and four signed-dose arms. All 36 baseline/zero pairs matched; all 144 nonzero arms preserved the recorded upstream/unaffected same-layer panel values. Effects depended on context, and even weakly associated neurons could have effects. These short activation observations do not measure story quality or reveal direct neural wiring.

## Four requests on one card: connected native scheduler

A native test ran four distinct neuronal injections concurrently in one decode graph. Across 16 run sets, repeatability, slot/row permutations and unaffected-request isolation passed. The matched four-request run generated **203 tokens** in **3.165 seconds**, versus **8.779 seconds** serially: **2.77377× warm wall-speedup**. First-decode logits and hidden values, and the entire generated token sequences, matched the serial references exactly.

This initial bounded runtime proof establishes execution parity, not better personalities. The [public proof](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_public.json) describes the experiment and identifies its [numeric arrays](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_proof.npz) by hash; timing excludes model preparation.

A second capture proof also passed independent verification: all 64 layers matched exactly between serial, two-request and four-request execution. It checked each request’s consumed token ownership, response means, slot/row permutations and early-finished request holes. The release files are the [capture proof](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_public.json) and its [numeric arrays](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_proof.npz). The native B4 HTTP service and Manager scheduler are now connected and qualified across six cards. A real 24-job Manager pilot passed. Each card decodes up to four independent requests together, with private recurrent/KV state, sampling state and activation records; unsupported controls retain the ordinary single-request route.

The **288-job reasoning sweep is complete**, separate from the personality diagnostic above: 253 complete answers, 35 answers that reached the token limit, and zero infrastructure failures. Earlier length-limited outputs were reclassified as incomplete with their audit history preserved; no responses were replayed. A valid capture does not make an unfinished answer complete: partial algebra answers remain unscorable and general answers remain unscored. Held-out cases do not guide adaptation.

The first actual six-card sample produced **51,428 tokens across 48 responses**, capturing 51,380 consumed tokens. From first submission to last HTTP response it took **281.4 seconds (182.8 tokens/s aggregate)**; through artifact validation, **286.8 seconds (179.3 tokens/s)**. These are measured instrumented workload rates, excluding preparation, not a daily capacity guarantee. Earlier raw Flare measurements of **220.6 tokens/s per card** used MTP depth four, cached prompts and no deep activation capture; they are a different workload.

A finite watcher updates local full-answer reports every 15 seconds. Only text-redacted IDs, hashes, counts and outcomes are eligible for Hub publication, every 48 terminal jobs and at completion or deadline. An uncertain upload stops further publication attempts while local reporting continues.

A readback optimization reduced redundant GPU-to-host transfers while preserving exact tokens and all checked activation arrays. On the bounded one-card canary, the same 2,504 tokens took 66.32 seconds before and 62.23 seconds after: **1.066× throughput**. A separate slot-refill proof reused finished slots while other requests remained active; all eight requests matched isolated references. Refill is not yet connected to production dispatch, and this proof does not establish a throughput gain. The [optimization evidence](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_optimization_public.json) identifies exact source versions, timing scope and capture hashes. Neither result establishes 220 tokens/s/card for deep capture.

## Read results with the right limits

A calibrated concept label requires held-out evidence, including a **95% Wilson precision lower bound above 70%**, plus the qualification’s recall, coverage, and balanced-accuracy checks. That is conditional performance on authored text probes—not a universal probability that a neuron “means” a word. Discovery examples select candidates; held-out examples evaluate frozen choices. Neither activation magnitude nor label direction establishes the sign of a useful steering intervention.

Coherence checks target supported facts, missing required details, contradictions, incomplete outputs, and repetition. They do not reward a positive personality. Passing a structured fact ledger does not certify all accompanying prose; fiction may intentionally introduce details.

Private books, extracted passages, arbitrary user prompts, credentials, and local deployment settings are not shipped. Public prompts are licensed **CC BY 4.0**; retain attribution. Derived checkpoint assets retain applicable upstream terms—publishing a measurement does not relicense the underlying model. Consult the manifests and asset notices before redistribution.
