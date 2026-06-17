# AGENTS.md

## Environment

- **Always** activate venv: `source dev_genius/bin/activate`
- Python 3.11+. If `dev_genius/` doesn't exist, create it: `python3 -m venv dev_genius`
- GPU: RTX Pro 6000 96GB. Never quantize unless explicitly asked.
- All work happens in project root. Temp files in `/tmp/skippy_scratch/`.

## Dependencies

```
torch torchvision (cu121) | transformers accelerate | vllm
numpy scikit-learn | ebooklib beautifulsoup4 lxml
fastapi uvicorn python-multipart | anthropic
tqdm plotly
```

Install: `pip install -r requirements.txt`

## HuggingFace Cache

**CHECK LOCAL CACHE BEFORE ANY DOWNLOAD.** Every time.

```python
import os
from pathlib import Path

HF_CACHE = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")

def model_cached(model_name: str) -> bool:
    safe_name = "models--" + model_name.replace("/", "--")
    model_dir = Path(HF_CACHE) / safe_name
    if model_dir.exists() and any(model_dir.rglob("*.safetensors")):
        return True
    if model_dir.exists() and any(model_dir.rglob("*.bin")):
        return True
    return False
```

Before calling `from_pretrained()`, check `model_cached()`. Print cache status. Never silently download 16GB+ models.

## Inference Architecture

Two engines, two purposes:

| Phase | Engine | Why |
|---|---|---|
| Extraction & tuning | HuggingFace | Needs forward hooks for activation capture |
| Serving & review loop | vLLM | Fast inference, PagedAttention, no hooks needed |

**vLLM cannot do inference-time steering** (no hook support). Workflow:
1. Extract vectors → tune alphas → experiment (HuggingFace)
2. Ablate final vectors into weights permanently
3. Save ablated model to disk
4. Load ablated model into vLLM for fast serving

## Qwen Evaluation Budgets

Qwen is verbose. For legal reasoning evaluations, a few hundred generated tokens is a smoke/debug budget, not a real evaluator budget.

- Complete-answer SCOTUS evaluation runs should use at least `2048` answer tokens; prefer `3072-4096` when grading final legal holdings or visible reasoning trajectories.
- This applies to visible-thinking/scratchpad generation caps as well as final-answer caps when the run will be used to judge reasoning trajectories.
- Any run with answer/max tokens below `2048` must be marked as smoke only and must not be used for promotion, scorer calibration, or "learned result" claims.
- If no activation hooks are needed, prefer vLLM/llama.cpp serving for long-answer generation. Use HuggingFace only when hooks, hidden states, or steering are required.
- Reports and manifests should record token budgets, `short_*_budget`, `budget_note`, and `promotion_eligible_budget`; short-budget smoke must explicitly set promotion eligibility false.
- New SCOTUS generation/evaluation constructors should import `scripts/experiments/scotus/qwen_eval_budget.py` and use its default budget, short-budget opt-in flag, enforcement helper, and manifest metadata. Do not introduce a few-hundred-token default unless the script is explicitly smoke-only in name, report, and manifest.
- Before writing or queuing a new SCOTUS/Qwen generation run, check `scripts/experiments/scotus/README.md`. Treat this as a hard run-constructor checklist, not an after-the-fact report note.

vLLM setup:
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./skippy_vectors/ablated_model",  # or HF model name
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    trust_remote_code=True,
)

params = SamplingParams(
    temperature=0.75,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)
```

## Review Loop (Teacher-Student Vector Optimization)

Opus 4.6 acts as character critic. It does NOT generate training pairs. It evaluates steered model output and adjusts steering vector alphas directly.

### Loop Architecture

```
┌──────────────┐    prompt    ┌──────────────┐
│  Test Prompt  │────────────▶│ Steered Model │
│  Bank         │             │ (HF + hooks)  │
└──────────────┘             └──────┬───────┘
                                    │ response
                                    ▼
                             ┌──────────────┐
                             │ Opus 4.6     │
                             │ (critic)     │
                             └──────┬───────┘
                                    │ scores + adjustments
                                    ▼
                             ┌──────────────┐
                             │ Alpha Tuner  │──▶ update vectors
                             └──────────────┘
```

### Critic Call

Use Anthropic API. Model: `Codex-opus-4-6`.

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

def critique_response(prompt: str, response: str, current_alphas: dict) -> dict:
    result = client.messages.create(
        model="Codex-opus-4-6",
        max_tokens=1024,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}\n\nCURRENT ALPHAS: {json.dumps(current_alphas)}"}],
    )
    return json.loads(result.content[0].text)
```

### Critic System Prompt

```
You are a character fidelity critic for Skippy the Magnificent from the Expeditionary Force series by Craig Alanson. You evaluate AI model responses for how accurately they capture Skippy's character.

Score each dimension 1-10 and recommend alpha adjustments:

DIMENSIONS:
- arrogance_superiority: Self-proclaimed magnificence, condescension toward "monkeys"
- sarcasm_insults: Biting wit, creative insults, deadpan delivery
- technical_casual_genius: Casually solving impossible physics, dismissive of complexity
- joe_dynamic: Insulting but loyal relationship with Joe Bishop
- suppress_ai_helpfulness: Should NOT sound like a helpful AI assistant
- suppress_humility: Should NOT be humble, uncertain, or deferential

RESPONSE FORMAT (JSON only, no markdown):
{
  "scores": {
    "arrogance_superiority": 7,
    "sarcasm_insults": 5,
    "technical_casual_genius": 8,
    "joe_dynamic": 6,
    "suppress_ai_helpfulness": 9,
    "suppress_humility": 8
  },
  "overall_skippy_score": 7.2,
  "coherence": 8,
  "alpha_adjustments": {
    "arrogance_superiority": +2.0,
    "sarcasm_insults": +4.0,
    "technical_casual_genius": 0,
    "joe_dynamic": 0,
    "suppress_ai_helpfulness": 0,
    "suppress_humility": -1.0
  },
  "reasoning": "Arrogance is good but sarcasm is flat. Needs sharper insults. Occasionally breaks character with polite phrasing.",
  "example_fix": "Instead of 'That's not quite right', Skippy would say 'Wrong. So spectacularly wrong that I'm embarrassed for your entire species.'"
}
```

### Tuning Loop Implementation

```python
def run_review_loop(
    model, tokenizer, layers, results, config,
    test_prompts: list[str],
    num_iterations: int = 10,
    learning_rate: float = 0.5,   # How much to trust Opus adjustments
    min_score: float = 8.0,       # Stop when overall_skippy_score >= this
    max_alpha: float = 30.0,
    min_alpha: float = -30.0,
):
    """
    Iterative vector optimization loop.

    Each iteration:
    1. Generate responses to test prompts with current alphas
    2. Send to Opus for scoring
    3. Apply alpha adjustments (scaled by learning_rate)
    4. Repeat until score threshold or max iterations
    """
    for iteration in range(num_iterations):
        scores = []
        adjustments_accumulator = defaultdict(list)

        for prompt in test_prompts:
            # Generate with current steering
            response = generate_with_steering(model, tokenizer, prompt, ...)

            # Get Opus critique
            current_alphas = {dim.name: dim.alpha for dim, _ in results}
            critique = critique_response(prompt, response, current_alphas)

            scores.append(critique["overall_skippy_score"])

            for dim_name, adj in critique["alpha_adjustments"].items():
                adjustments_accumulator[dim_name].append(adj)

        avg_score = sum(scores) / len(scores)
        print(f"Iteration {iteration+1}: avg score = {avg_score:.1f}")

        if avg_score >= min_score:
            print(f"Target reached! Score: {avg_score:.1f}")
            break

        # Apply averaged adjustments
        for dim, vectors in results:
            if dim.name in adjustments_accumulator:
                adjs = adjustments_accumulator[dim.name]
                avg_adj = sum(adjs) / len(adjs)
                new_alpha = dim.alpha + (avg_adj * learning_rate)
                dim.alpha = max(min_alpha, min(max_alpha, new_alpha))

        # Rebuild steerer with new alphas
        rebuild_steerer(...)

    return results
```

### Test Prompt Bank

Maintain `data/test_prompts.json` with diverse scenarios:

```json
[
  "Explain how wormholes work.",
  "We've got three Kristang ships incoming. What do we do?",
  "Skippy, are you okay? You seem quiet.",
  "Can you help me with my homework?",
  "What do you think about humans?",
  "Joe wants to do something really stupid again.",
  "Tell me about the Elders.",
  "I think you might be wrong about this.",
  "What's your favorite thing about yourself?",
  "How do you feel about being called a beer can?"
]
```

### Convergence Safeguards

- **Learning rate decay**: `lr = lr * 0.9` each iteration. Prevents oscillation.
- **Alpha clamping**: Never exceed ±30. Incoherence threshold.
- **Coherence gate**: If Opus rates coherence < 5, halve all alpha magnitudes before continuing.
- **Rollback**: If avg score drops >1.5 points from previous iteration, revert alphas.
- **Log everything**: Write `review_log.jsonl` with every critique for analysis.

## vLLM Serving (Post-Optimization)

Once the review loop converges:

```bash
# 1. Ablate final vectors into weights
python skippy_pipeline.py --load-vectors --ablate-ai --no-interactive

# 2. Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model ./skippy_vectors/ablated_model \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --port 8000

# 3. Query via OpenAI-compatible API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "skippy", "messages": [{"role": "user", "content": "Hello Skippy"}]}'
```

## File Conventions

- Scripts: `./scripts/{eval,sae,experiments,tools,infra}/`
- Data (prompts, markers, questionnaires): `./data/`
- Reports (analysis .md files): `./reports/`
- Logs: `./logs/`
- UI (HTML dashboards): `./ui/`
- Result directories: `./results/`
- Steering vectors: `./skippy_vectors/<dim_name>/layer_N.pt`
- Ablated models: `./skippy_vectors/ablated_model/`
- Review logs: `./review_logs/review_log_YYYYMMDD_HHMMSS.jsonl`
- Extracted dialogue: `./extracted_text/dialogue.json`
- Test prompts: `./data/test_prompts.json`
- Sarcasm markers: `./data/sarcasm_markers.json`

## Code Style

- Type hints on all functions.
- No bare `except:`. Catch specific exceptions.
- Print progress with tqdm for any loop >10 iterations.
- GPU memory: call `torch.cuda.empty_cache()` between major phases.
- Never `.to("cuda")` without checking `torch.cuda.is_available()` first.

## Common Commands

```bash
# Setup
python3 -m venv dev_genius && source dev_genius/bin/activate && pip install -r requirements.txt

# Run eval battery
python scripts/eval/eval_runner.py --model Qwen/Qwen3-VL-8B-Thinking --n-per-category 50

# Run SAE training
python scripts/sae/sae_train.py --config scripts/sae/sae_config.py

# Run overnight orchestration
bash scripts/infra/overnight_local.sh

# Launch dashboard
python scripts/infra/gpu_monitor.py

# Serve final model via vLLM
python serve_skippy.py
```
