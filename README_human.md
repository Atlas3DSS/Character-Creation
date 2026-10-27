# Character Creation: a practical starting point

Character Creation is a workbench for changing an LLM’s behavior through **activation steering**: adding a measured direction or a specific neuron intervention while the model runs. Start with a voice, build a recipe, and compare its answers with an unsteered baseline. A sharper, anxious, playful, or unusually imaginative answer can be an interesting result; sounding pleasant is not the objective.

The product goal is a character-creation step for a lasting AI companion. Most users should be able to choose a starting personality, try it in revealing situations, adjust familiar traits, and save the recipe. Users who want finer control can inspect and edit individual neurons, features, signed doses and intervention phases. Saved recipes preserve the actual interventions and checkpoint binding; a character name alone does not establish how the model will behave.

The approach owes substantial methodological credit to [Lyra’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), especially signed interventions, relative doses, and the distinction between observing neurons and changing outputs. Runtime execution uses the separate [Flare implementation](https://github.com/Atlas3DSS/flare-local-inference). This workbench is an independent implementation, not an endorsement by those projects.

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

Open `http://127.0.0.1:8792`. Use the compact workspace picker to move between Persona Studio, recipe experiments, activation search, and evidence views. In the map, **View controls** exposes geometry and connection options; **Expand map** gives the visualization more room. Those display controls do not run inference.

Build a candidate recipe, then compare it against baseline using the same prompts, seed, and answer budget. Check several scenarios before describing a repeatable effect. Distinguish association, weight-based suggestions, and tested interventions.

Persona Studio includes an authored **Personality probes** battery for correction, social pressure, criticism, reassurance, conflict and uncertainty. For example, the user confidently gives a wrong receipt total; compare whether the character agrees, hedges, corrects kindly, jokes, or becomes hostile. Judge the answer's correctness separately from how it treats the person. These 24 prompts form 12 paired situations with different pressure levels. Choose one beside your cast or preset comparison, adjust it if desired, then explicitly run the comparison. Every arm receives the same scene; only the activation recipe changes. The prompts do not tell the model which personality to imitate. This targeted battery complements the broader scene and literary collections; it is not a complete test of a long-term relationship.

In **Live activation search**, enter any text, snippet, idea, feeling or thought; you can also highlight just part of a prompt. A compatible search measures all **1,114,112 neuron coordinates** across the rendered prompt, including template tokens, then shows ranked candidates and their token profiles. Select candidates yourself or choose one of up to three **Suggested starting experiments**: the leading candidate, a candidate from another layer, or a small group across layers. These are untested low-dose trials, not proven safe or useful recipes. Edit the shared scene, then explicitly run baseline, positive and negative steering with the same prompt and seed. Selecting a starter or clearing a draft does not start inference.

Activity is a contextual clue, not confidence that a neuron means a concept. Predicting useful effects will require records connecting context, selected neurons, dose and actual outcomes; adding KV control does not supply that predictor automatically. The unified search runtime is live on all six workers, with identity and calibration checks. An initial batch completed 60 searches and 72 matched scene responses. The same neuron ranked first in 59 searches, showing why raw activity alone is insufficient for concept-specific recommendations. Those trials do not establish useful presets. The [published trial records and assistant review](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_association_public.json) include token profiles and matched answers. Review found some omitted or contradicted scene details despite every answer completing. A further [150-response dose study](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_recipe_depth_public.json) tested six previously observed neurons. Its review found localized detail errors without a consistent dose–harm relationship; reliable improvements and defaults remain unproven.

A separate **288-answer diagnostic sweep is complete**: 24 authored prompts × six activation arms × two seeds. It retained response activation means across all 64 layers from **136,018 consumed generated tokens**. The blinded reviewer was a local Qwen-family model, not a human or independent source of truth. It flagged detail issues in baseline and steered answers; no recipe became a default. The release files are [answers, recipes and review](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_public.json) and the [response activation bundle](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_personality_capture_bundle.npz). These are discovery comparisons, separate from the 256-answer pilot below.

The **256-answer persona pilot is complete** across all six Quincies. It tested two response-derived candidates—dry skeptical wit and warm attentive candor—alongside their reversals, baseline and equal-strength random steering. It used 21 fresh scenes and two seeds (252 answers), plus four reproducibility checks. All four checks reproduced the exact tokens and text on different cards. Eight separate development scenes supplied the answer-token activation contrasts.

The blinded model judge found a repeatable dry-wit effect: on the nine held-out scenes, the candidate improved its target score by **0.89 out of 4 against baseline** and **0.83 against random steering**. The warmth candidate improved by 0.44 against baseline, but its 0.17 advantage over random steering remained uncertain. Invented details, missing details and coherence issues occurred in the comparisons, so neither candidate became a default. These are model judgments with descriptive uncertainty estimates, not human certification.

Open **Studies → Response-contrast pilot** to choose a scene and read the six answers side by side. The same view shows blinded comparisons, fixed-input propagation measurements and a self-rating diagnostic. Sixty propagation captures show how the directions affect four layers over five token positions; effects change with context and dose. Forty-eight self-ratings showed moderate agreement with the external judge, enough to investigate further but insufficient to use as a training reward.

For independent analysis, download the [public study index](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_response_study_public.json), [judged answers](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_pilot_public.json), and the exact files/hashes listed in the index. The activation bundle contains answer-token means for all 24 development and 256 pilot answers, plus the 60 propagation snapshots. It does not store every neuron at every answer token. This is an experimental comparison, not a validated preset release.

A separate [42-answer follow-up](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_repair_public.json) tested lower doses on seven selected discovery cases with errors. Halving the dose reduced judged coherence issues from four cases to zero, but three detail errors remained. Quartering the dose left two detail errors and one coherence issue. Baseline and matched random controls were cleaner. This supports testing dose reduction case by case; it does not establish an automatic repair rule or a universally safe strength.

In **Literary passages**, compile a scene prompt and open **Experimental neuron × KV comparison**. With a qualified runtime, it compares baseline, neuron steering, a selected KV control, and both together. The same passage, seed and answer budget are used in all four arms, and the source record stays attached to the results. You can review whether an output answered or refused separately from its coherence checks.

The cached-text control reuses an ordinary prefix’s complete model state. A separate learned attention-KV candidate has completed 16 discovery-scene training steps and evaluation on eight untouched held-out scenes. Its FP32 forward/gradient checks passed, followed by **72 native qualification requests across all six workers**. Qualification verifies the intervention machinery, not a useful personality effect. Both controls use exact off/on settings, not continuous strength.

The learned-control literary pilot produced 15 complete responses and one truncated neuron-only response. Effects were mixed: adding learned KV resolved that one revision loop, but another combined response contradicted the source scene. Learned KV alone showed no consistent improvement over baseline. Nothing is promoted automatically, and better prose, refusal reduction or coherence recovery have not been established.

A separate 216-capture study tested six source neurons in six contexts with baseline, zero and four signed-dose arms. All 36 baseline/zero pairs matched; all 144 nonzero arms preserved the recorded upstream/unaffected same-layer panel values. Effects depended on context, and even weakly associated neurons could have effects. These short activation observations do not measure story quality or reveal direct neural wiring.

## Four requests on one card: verified native proof

A native test ran four distinct neuronal injections concurrently in one decode graph. Across 16 run sets, repeatability, slot/row permutations and unaffected-request isolation passed. The matched four-request run generated **203 tokens** in **3.165 seconds**, versus **8.779 seconds** serially: **2.77377× warm wall-speedup**. First-decode logits and hidden values, and the entire generated token sequences, matched the serial references exactly.

This is a bounded runtime proof, not evidence of better personalities or a production four-request API. The [public proof](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_public.json) describes the experiment and identifies its [numeric arrays](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_proof.npz) by hash; timing excludes model preparation.

A second capture proof also passed independent verification: all 64 layers matched exactly between serial, two-request and four-request execution. It checked each request’s consumed token ownership, response means, slot/row permutations and early-finished request holes. The release files are the [capture proof](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_public.json) and its [numeric arrays](https://huggingface.co/datasets/Atlas3D/character-creation-big-five/blob/main/persona_batch_capture_proof.npz). Production HTTP queueing and routing have **not** been rolled out; a 24-request HTTP/routing qualification is next.

## Read results with the right limits

A calibrated concept label requires held-out evidence, including a **95% Wilson precision lower bound above 70%**, plus the qualification’s recall, coverage, and balanced-accuracy checks. That is conditional performance on authored text probes—not a universal probability that a neuron “means” a word. Discovery examples select candidates; held-out examples evaluate frozen choices. Neither activation magnitude nor label direction establishes the sign of a useful steering intervention.

Coherence checks target supported facts, missing required details, contradictions, incomplete outputs, and repetition. They do not reward a positive personality. Passing a structured fact ledger does not certify all accompanying prose; fiction may intentionally introduce details.

Private books, extracted passages, arbitrary user prompts, credentials, and local deployment settings are not shipped. Public prompts are licensed **CC BY 4.0**; retain attribution. Derived checkpoint assets retain applicable upstream terms—publishing a measurement does not relicense the underlying model. Consult the manifests and asset notices before redistribution.
