# GRPO Tool-Call Training Plan (Saved for Later)

## Context

R5 neuron-guided training on Qwen3-VL-8B achieved 38% sarcastic, solid identity, correct math.
Next step: GRPO loop to train Skippy for high-quality tool calling while preserving personality.

## Why GRPO (Not SFT)

- SFT catastrophically forgets reasoning (0% AIME at any LoRA merge scale)
- GRPO optimizes in **policy space** via reward signal — avoids weight-space interference
- V3/V4 GRPO runs showed personality preservation with reward-guided training
- Tool calling is naturally reward-shaped: correct format + correct tool + correct args = high reward

## Architecture

### Reward Functions (Combined)

1. **Tool Call Reward** [-5.0, 6.0]:
   - Correct `<tool_call>` format: +1.0
   - Correct tool name (from household_config.py tools): +2.0
   - Correct argument types: +1.0
   - Plausible argument values: +1.0
   - Correct tool for the prompt context: +1.0
   - Invalid JSON or malformed tags: -5.0

2. **Personality Reward** [-5.0, 6.0] (from V3):
   - Penalize AI patterns ("I'm sorry", "I cannot", "as an AI")
   - Reward sarcasm markers ("monkeys", "magnificent", "beer can")
   - Weighted regex scoring

3. **Coherence Reward** [-5.0, 1.0] (from V3):
   - Penalize repetition/trigrams
   - Reward token diversity

4. **Identity Reward** [-3.0, 3.0] (from V3):
   - Reward speaking AS Skippy
   - Penalize body/third-person references

### Combined Reward: `R = tool + personality + coherence + identity`
Range: [-18.0, 16.0]

## Training Data: 250 Tool-Call Benchmark Prompts

### Categories (50 each):
1. **Home Automation**: "Turn on the living room lights", "Set thermostat to 72", "Lock the front door"
2. **Camera/Security**: "Show me the backyard camera", "Is anyone at the front door?", "Check the garage"
3. **Item Tracking**: "Where did I put my keys?", "Find my phone", "Where's the dog leash?"
4. **Notifications**: "Tell Sarah dinner is ready", "Remind me at 5pm", "Send Will a message"
5. **Web Search + Conversation**: "What's the weather?", "Look up pizza places nearby", mixed personality prompts

### Prompt Format
Each prompt includes the 5 household tools in the system message, plus Skippy personality prompt.
Model must decide: call a tool, respond conversationally, or both.

## Neuron Probing During GRPO

- Hook NeuronTracker into GRPO training loop
- Profile every N steps (not every step — too expensive)
- Track: Do personality neurons stay active during tool-call completions?
- If personality neurons drift toward assistant mode during tool calls → increase personality reward weight

## Implementation

### Base Model
`./skippy_sdft_r5/merged_scale_1.0` — best R5 model (38% sarcastic, 4/4 math)

### LoRA Config
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### GRPO Config
```python
GRPOConfig(
    output_dir="./skippy_grpo_tools/",
    num_generations=4,       # 4 completions per prompt
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
)
```

### Eval Prompts (20 tool-call + 4 personality + 4 math)
- Tool-call accuracy: % of prompts with correct tool + args
- Personality: sarcastic count, assistant leak count
- Math: correctness preserved
- Identity: 3/3 says Skippy

## Expected Outcomes

| Metric | R5 Baseline | Target |
|--------|-------------|--------|
| Identity | 3/3 | 3/3 |
| Sarcastic | 38% | ≥ 40% |
| Math | 4/4 | 4/4 |
| Tool accuracy | 0% (untrained) | ≥ 80% |
| Tool + personality | N/A | Tool calls WITH sarcastic commentary |

## Key Risks

| Risk | Mitigation |
|------|------------|
| Tool format conflicts with personality | Separate tool call from commentary (tool in tags, personality in surrounding text) |
| GRPO reward hacking | Monitor samples manually every 50 steps |
| VRAM for GRPO (4 generations) | gradient_checkpointing=True, batch=1, grad_accum=8 |
| Personality regression | Personality reward weight ≥ tool reward weight |

## Files

### Create
- `train_skippy_grpo_tools.py` — GRPO training script with tool-call rewards
- `tool_call_prompts.json` — 250 benchmark prompts with tool definitions

### Reuse
- `household_config.py` — Tool definitions, system prompts, household registry
- `neuron_guided_training.py` — NeuronTracker for probing
- `contrastive_data/skippified_combined_r5.jsonl` — Personality reference data
