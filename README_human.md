# Character Creation: a practical starting point

Character Creation is a workbench for changing an LLM’s behavior through **activation steering**: adding a measured direction or a specific neuron intervention while the model runs. Start with a voice, build a recipe, and compare its answers with an unsteered baseline. A sharper, anxious, playful, or unusually imaginative answer can be an interesting result; sounding pleasant is not the objective.

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

## Read results with the right limits

A calibrated concept label requires held-out evidence, including a **95% Wilson precision lower bound above 70%**, plus the qualification’s recall, coverage, and balanced-accuracy checks. That is conditional performance on authored text probes—not a universal probability that a neuron “means” a word. Discovery examples select candidates; held-out examples evaluate frozen choices. Neither activation magnitude nor label direction establishes the sign of a useful steering intervention.

Coherence checks target supported facts, missing required details, contradictions, incomplete outputs, and repetition. They do not reward a positive personality. Passing a structured fact ledger does not certify all accompanying prose; fiction may intentionally introduce details.

Private books, extracted passages, arbitrary user prompts, credentials, and local deployment settings are not shipped. Public prompts are licensed **CC BY 4.0**; retain attribution. Derived checkpoint assets retain applicable upstream terms—publishing a measurement does not relicense the underlying model. Consult the manifests and asset notices before redistribution.
