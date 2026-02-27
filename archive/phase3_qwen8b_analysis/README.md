# Phase 3: Qwen 8B Deep Analysis (Feb 17-22, 2026)

Comprehensive analysis of Qwen3-VL-8B internals: connectome mapping (20 categories x 36 layers x 4096 dims),
surgical steering, activation patching, gradient attribution, head atlas, alpha sweeps.

## Key Findings
- 20-category connectome mapped: hub neurons at dims 235, 908, 2136, 2514
- Identity perpendicular to Sarcasm (cosine=-0.0002)
- Identity = Assistant neurons (95-100% overlap)
- Name relay circuit: 3828(L0-5) -> 98/368(L6-11) -> 994(L12-33) -> 2276(L34-35)
- Surgical steering: per-neuron targeting works but is fragile
- Activation patching: causal evidence for relay circuit
- Gradient attribution confirms L9-L26 as personality-critical band
- Head atlas: attention heads specialize by concept category

## Scripts (21 files)
- `probe_qwen_neurons.py` — Core neuron probing (ActivationProbe pattern reused everywhere)
- `probe_sarcasm_neurons.py` — Sarcasm-specific probing
- `qwen_connectome_probe.py` — Full 20-category connectome
- `qwen_deep_analysis_3090.py` — Deep analysis on 3090
- `qwen_head_atlas.py` — Attention head specialization mapping
- `qwen_surgical_steering.py` — Per-neuron surgical steering
- `qwen_activation_patching.py` — Causal activation patching
- `qwen_gradient_attribution.py` — Gradient-based attribution
- `qwen_causal_ablation.py` / `qwen_layer_ablation.py` — Ablation studies
- `qwen_alpha_sweep.py` / `qwen_weighted_alpha_sweep.py` — Alpha parameter sweeps
- `qwen_skippy_actadd.py` / `qwen_weighted_actadd.py` — Activation addition methods
- `analyze_probe_results.py` / `compare_neuron_maps.py` — Analysis utilities
- `certainty_probes.py` — Certainty dimension probing
- `capture_reasoning_activations.py` — Reasoning activation capture
- `eval_aime.py` / `eval_reasoning_benchmarks.py` — Math/reasoning benchmarks
- `profile_prompted_delta.py` — Prompted vs unprompted activation deltas

## Data Directories
- `qwen_connectome/` — Connectome activations and analysis
- `qwen_head_atlas/` — Head atlas data
- `qwen_deep_analysis/` — Deep analysis results
- `qwen_gradient_attribution/`, `qwen_activation_patching/`, `qwen_causal_ablation/`, `qwen_layer_ablation/`

## RESEARCH_NOTES.md References
- "Connectome Mapping" section
- "Sarcasm Relay Circuit" section
- See also: `neuron_probe_findings.md`, `connectome_findings.md`
- Commits: `1e422c5`, `958b473`, `f855647`, `98a1bad`
