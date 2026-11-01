# J-Lens Three-Brief Implementation Goal

Date: 2026-07-09

Implement and run the three July 8 J-lens briefs end to end as real research pilots, not smoke-only schema checks:

1. **J-space persona fingerprinting**: generate long-form responses from an unmodified Qwen model, capture generated-token residual activations only, and test whether persona labels are decodable from top-k J-space at parity with raw activations while random same-dimensional subspaces lag, especially at small k.
2. **J-ReFT / J-LoRA pilot**: fit/use a local lens for the exact unmodified Qwen3.5-9B instruct checkpoint, train the registered J-space, random, unconstrained, complement, prompt-baseline, and raw-baseline arms, and evaluate held-out no-system generations with automatic persona, capability, and coherence metrics.
3. **Delta-J comparison**: fit same-model disjoint-prompt lenses to establish a refit-noise floor, fit the paired unmodified Qwen checkpoint lens, and report subspace/map/vocab-drift deltas only as multiples of that floor.

Operational constraints:

- Use cached unmodified Qwen checkpoints; do not use abliterated models for the base Delta-J test.
- Use BF16/full precision unless explicitly instructed otherwise; do not quantize.
- Use the workstation RTX Pro 6000 and the dev-box RTX 3090/4090 in parallel where possible.
- Real Qwen generation/evaluation budgets must be measured in thousands of tokens. Runs below 2048 generated tokens are smoke/debug only and are not promotion eligible.
- Check model/lens cache status before any heavyweight load or download, and record cache paths in manifests.
- Every run must emit manifests, logs/events, records, and reports under `sweep_v4/`, with explicit claim status and budget eligibility.

Done means the scripts compile locally and remotely, real jobs have completed or failed with actionable logs, and each brief has either a real report with controls or a clearly labeled blocker explaining why a claim cannot be made.
