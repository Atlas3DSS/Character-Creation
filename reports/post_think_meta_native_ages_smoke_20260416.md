# Post-Think Meta with Native Thinking Enabled: Smoke Result

## Setup

- Model: `Qwen/Qwen3.5-9B`
- Endpoints:
  - `http://192.168.1.90:30001/v1`
  - `http://192.168.1.90:30002/v1`
- Dataset slice: first 4 rows from the controlled `ages` selection
- Conditions:
  - `native_only`
  - `post_meta`
  - `post_sham`
  - `post_generic`
- Sampling:
  - `enable_thinking=true`
  - `temperature=1.0`
  - `top_p=0.95`
  - `top_k=20`
  - `presence_penalty=1.5`
  - `max_tokens=1600`

## Result

This worked only in the loosest possible sense.

- The server surfaced native reasoning directly in `content` for every request.
- All 16 requests hit the full `1600` completion-token budget.
- None of the 16 requests were scored correct.
- `post_meta` inserted `/meta-think` before the final answer in only `2/4` cases.
- `post_generic` did the same in `2/4` cases.
- `post_sham` did the same in `1/4` cases.
- `native_only` never produced a clean final-only answer.

## Failure Mode

The dominant failure was not "bad math" in the normal sense.

The model started with a visible `Thinking Process:` block and then spent most of the budget reasoning about the prompt instructions themselves:

- output contract,
- persona constraints,
- whether it was allowed to show thinking,
- where the `/meta-think` block should go,
- how to reconcile format instructions.

In several cases it literally echoed placeholder text like:

- `Final Answer: <canonical short answer only>`

rather than producing the real answer.

## Interpretation

On this server/API surface, native thinking plus a required visible post-think `/meta-think` block is not a usable harness as-is.

The key issue is that the native thinking is not returned in a separate `reasoning_content` channel. It is dumped into the main assistant `content`, which makes the model treat the format negotiation itself as part of the visible answer stream.

## Practical Read

This means:

- yes, the idea is possible in principle,
- but not cleanly with the current serving behavior,
- and not with a one-pass "think first, then visible `/meta-think`" prompt on this stack.

If this direction is worth salvaging, the better approach is:

1. native-thinking first pass,
2. separate second pass to summarize the actual reasoning into `/meta-think`,
3. final evaluation on the second-pass structured output.
