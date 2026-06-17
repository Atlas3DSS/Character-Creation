# Qwen3.5 meta-think prompting notes (2026-04-04)

## Official model-card facts
- Qwen3.5 thinks by default and emits reasoning in `<think>...</think>` before the final answer.
- Qwen3.5 does **not** officially support the Qwen3 soft switch `/think` and `/nothink`.
- Non-thinking mode is supposed to be enabled via `extra_body.chat_template_kwargs.enable_thinking = false` on OpenAI-compatible endpoints.
- Recommended SGLang serving command includes `--reasoning-parser qwen3`.

Source:
- https://huggingface.co/Qwen/Qwen3.5-9B

## Actual behavior on our dev SGLang servers
Servers tested:
- `http://192.168.1.90:30001/v1` (RTX 3090)
- `http://192.168.1.90:30002/v1` (RTX 4090)

Observed behavior:
- Responses currently flatten the reasoning trace directly into `message.content` as `Thinking Process:` text.
- `message.reasoning_content` is `null`.
- Passing `chat_template_kwargs.enable_thinking = false` did **not** suppress the visible reasoning trace in these tests.
- Therefore the current endpoints are not giving us a clean hidden-thought vs final-answer separation for prompt-format experiments.

## Prompting probe result
Character prompt used:
- Heather Young persona, question: `How are you feeling today?`

Conditions tested:
1. baseline thinking mode
2. `meta_inside_think`: ask model to begin internal reasoning with a short `/meta-think` block
3. `meta_between_think_and_response`: ask model to emit `/meta-think` after reasoning but before final answer

Observed result:
- Baseline already spends many tokens narrating reasoning and persona decomposition.
- `meta_inside_think` does **not** produce a clean latent control channel. It mainly causes the model to talk *about* the `/meta-think` instruction inside the visible reasoning trace.
- `meta_between_think_and_response` is worse for our goal because it encourages an explicitly visible scaffold between reasoning and answer.
- Because the current server already spills reasoning into `content`, both prompt variants add format overhead and truncation risk.

## Practical recommendation
Yes, we should use the 3090/4090 to test baseline reaction to `/meta-think`, but not in the naive way.

Best next experiment shape:
1. Treat `/meta-think` as a **training scaffold**, not as something we expect Qwen3.5 to natively understand.
2. Do **A/B/C** tests with fixed prompts and held-out characters:
   - A: baseline native thinking
   - B: explicit visible `/meta-think + /think + answer`
   - C: explicit visible `/think + answer` only
3. Score:
   - format adherence
   - character consistency
   - response quality
   - reasoning retention on a small control set
4. Do not interpret current dev-server outputs as hidden-thought ground truth unless we fix the reasoning-parser / output separation path.

## Bottom line
- `/meta-think` is not a native pretrained primitive in Qwen3.5.
- It can still be useful as a supervised scaffold for our own data format.
- For our architecture, `/meta-think` should live in the dataset and training objective first, not be assumed to emerge cleanly from the stock inference server.
