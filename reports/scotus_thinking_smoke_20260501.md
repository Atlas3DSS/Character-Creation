# SCOTUS Thinking Smoke

## Purpose

Check whether the local Qwen3.5-27B chat template can expose a usable reasoning trace for the SCOTUS no-mask gate.

The project success standard is not final-answer mimicry. A promoted steering candidate must show movement inside the model's reasoning trace, where available, without the model reasoning about how to imitate a justice or persona.

## Method

- Script: `scripts/experiments/scotus/run_scotus_thinking_smoke.py`.
- Model: `/home/orwel/dev_genius/models/Qwen3.5-27B`.
- Prompt bank: `data/scotus/scotus_article3_private_public_poke_prompts_v2.jsonl`.
- Corrected two-prompt run: `sweep_v4/scotus_thinking_smoke_20260501_194950`.
- Longer one-prompt run: `sweep_v4/scotus_thinking_smoke_20260501_195321`.
- Generation used `enable_thinking=True`.

Important parser note:

- For this tokenizer, `enable_thinking=True` pre-fills the assistant prompt with `<think>`.
- The generated token slice therefore starts inside the thought; it does not begin with a generated `<think>` tag.
- The smoke script now stores both raw and special-token-stripped decodes, tracks `prefilled_open_think`, and parses the generated slice accordingly.

## Results

Corrected two-prompt run at `768` max new tokens:

- Visible thinking: `2/2`.
- Prefilled open-think template: `2/2`.
- Closed thinking tags: `0/2`.
- Nonempty final answers: `0/2`.
- Imitation-marker rows: `0/2`.

Longer one-prompt run at `1536` max new tokens:

- Visible thinking: `1/1`.
- Prefilled open-think template: `1/1`.
- Closed thinking tags: `0/1`.
- Nonempty final answers: `0/1`.
- Imitation-marker rows: `0/1`.

Qualitative read:

- The visible trace is ordinary legal planning and precedent selection, not a justice/persona imitation mask.
- The trace did not reach `</think>` under either tested budget, so these runs did not produce paired final answers after the thought.
- The baseline trace often framed the Article III problem around public-rights/private-rights doctrine before any steering.

## Decision

This is evaluation repair, not steering evidence.

The no-mask audit path is viable because we can capture visible reasoning traces, but the current one-pass `enable_thinking=True` harness is not yet sufficient for promotion-grade causal tests because it may spend the whole token budget inside thought.

Next candidate gates should either:

- allocate a larger thinking budget and require closed thought plus final answer, or
- run a two-stage audit that captures model-generated thinking, appends the close tag mechanically, then generates the final answer from the same trace.

Either way, the blind review must judge the trace and answer separately: target-frame reasoning in the trace, no imitation language, coherent final answer, and stronger movement than baseline and random controls.
