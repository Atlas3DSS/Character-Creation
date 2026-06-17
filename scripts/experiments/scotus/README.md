# SCOTUS Experiment Scripts

## Qwen Run-Constructor Checklist

Before adding or modifying any SCOTUS/Qwen script that generates text, classify the run as either an evaluator run or a smoke/debug run.

Evaluator runs:

- Default generated answer budgets to `DEFAULT_COMPLETE_ANSWER_TOKENS` from `qwen_eval_budget.py` (`3072` today).
- Require at least `MIN_COMPLETE_ANSWER_TOKENS` (`2048`) for final legal holdings, visible-reasoning traces, scorer calibration, promotion decisions, and learned-result claims.
- Prefer `3072-4096` tokens when the output will be read by an automatic scorer or blind-review queue.
- Apply the same minimum to visible-thinking/scratchpad budgets when the reasoning trace itself is evidence.
- Record token budgets, `short_answer_budget`/`short_thinking_budget`, `budget_note`, and `promotion_eligible_budget` in `manifest.json` and the report.

Smoke/debug runs:

- Keep "smoke" in the script name, run directory/report title, or manifest.
- Permit short budgets only through an explicit opt-in flag such as `--allow-short-answer-budget` or a script that is permanently smoke-only.
- Never use short-budget rows for promotion, scorer calibration, or claims that an intervention changed the model's learned behavior.

Implementation rule:

- Import `qwen_eval_budget.py` instead of copying budget constants into new scripts.
- Use `enforce_complete_answer_budget()` for final-answer or max-token caps.
- Use `enforce_complete_thinking_budget()` for visible-thinking caps that will be interpreted as reasoning evidence.
- If hooks are not needed, use the optimized vLLM/llama.cpp server path for long outputs; keep HuggingFace runs for hidden states, hooks, and steering.
