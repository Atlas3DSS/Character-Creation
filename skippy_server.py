"""
SKIPPY DASHBOARD — Backend Server
===================================
FastAPI server that wraps the steering toolkit and provides:
  - Chat generation with steering
  - Alpha adjustment endpoints
  - Activation data for visualizations
  - PCA projections of activation space
  - Layer-by-layer analysis

Run: uvicorn skippy_server:app --host 0.0.0.0 --port 8000
"""

import torch
import numpy as np
import json
import os
import asyncio
from pathlib import Path
from typing import Optional
from collections import defaultdict
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# We'll import from the toolkit
import sys
sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(title="Skippy the Magnificent — Steering Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.layers = None
        self.num_layers = 0
        self.hidden_dim = 0
        self.config = None
        self.results = []           # (dimension, vectors) pairs
        self.steerer = None
        self.chat_history = []
        self.loaded = False
        
        # Visualization data (computed once, cached)
        self.pca_data = None
        self.layer_profiles = None
        self.similarity_matrix = None
        self.activation_samples = {}

state = AppState()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.75

class AlphaUpdate(BaseModel):
    dimension_index: int
    alpha: float

class PresetRequest(BaseModel):
    preset: str  # "crankit", "chill", "reset", "custom"
    alphas: Optional[dict] = None  # {dim_index: alpha} for custom

class ProbeRequest(BaseModel):
    """Send a probe prompt and get back activation data for visualization."""
    prompt: str
    label: str = "probe"

class LoadRequest(BaseModel):
    model_name: str = "Qwen/Qwen3-8B"
    vectors_path: str = "./skippy_vectors"
    epub_dir: str = "./books"
    dtype: str = "float16"


# =============================================================================
# ACTIVATION ANALYSIS UTILITIES
# =============================================================================

def compute_pca_projection(activations_dict, n_components=3):
    """
    Compute PCA projection of activations for visualization.
    
    activations_dict: {label: tensor of shape (n_samples, hidden_dim)}
    Returns: {label: array of shape (n_samples, 3)} + explained variance
    """
    from sklearn.decomposition import PCA
    
    # Concatenate all activations
    all_acts = []
    labels = []
    label_indices = {}
    idx = 0
    
    for label, acts in activations_dict.items():
        if isinstance(acts, torch.Tensor):
            acts = acts.numpy()
        all_acts.append(acts)
        n = len(acts)
        label_indices[label] = (idx, idx + n)
        labels.extend([label] * n)
        idx += n
    
    all_acts = np.concatenate(all_acts, axis=0)
    
    # PCA
    pca = PCA(n_components=min(n_components, all_acts.shape[1], all_acts.shape[0]))
    projected = pca.fit_transform(all_acts)
    
    # Split back by label
    result = {}
    for label, (start, end) in label_indices.items():
        result[label] = projected[start:end].tolist()
    
    return {
        "points": result,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_[:3].tolist(),
    }


def compute_vector_profiles(results, extract_layers):
    """
    For each dimension, compute the vector magnitude at each layer.
    This shows WHERE in the network each concept lives.
    """
    profiles = {}
    
    for dim, vectors in results:
        magnitudes = []
        for layer_idx in sorted(extract_layers):
            if layer_idx in vectors:
                vec = vectors[layer_idx]
                if vec.dim() > 1:
                    vec = vec[0]
                magnitudes.append({
                    "layer": layer_idx,
                    "magnitude": float(vec.norm()),
                    "max_component": float(vec.abs().max()),
                    "mean_component": float(vec.abs().mean()),
                })
            else:
                magnitudes.append({"layer": layer_idx, "magnitude": 0})
        
        profiles[dim.name] = {
            "magnitudes": magnitudes,
            "alpha": dim.alpha,
        }
    
    return profiles


def compute_similarity_matrix(results, layer_idx):
    """
    Compute cosine similarity between all steering vectors at a given layer.
    Shows how the dimensions relate to each other.
    """
    names = []
    vectors = []
    
    for dim, vecs in results:
        if layer_idx in vecs:
            vec = vecs[layer_idx]
            if vec.dim() > 1:
                vec = vec[0]
            vectors.append(vec)
            names.append(dim.name)
    
    if not vectors:
        return {"names": [], "matrix": []}
    
    vecs = torch.stack(vectors)
    # Normalize
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    # Cosine similarity
    sim = (vecs_norm @ vecs_norm.T).numpy().tolist()
    
    return {
        "names": names,
        "matrix": sim,
    }


def collect_probe_activations(model, tokenizer, layers, prompt, extract_layers, avg_n=4):
    """Collect activations for a single probe prompt at all extract layers."""
    activations = {}
    hooks = []
    
    for idx in extract_layers:
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = hidden[0, -avg_n:, :].mean(dim=0).detach().cpu().float()
            return hook_fn
        hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return activations


def compute_steering_effect_preview(results, layer_idx):
    """
    Compute what the combined steering effect looks like.
    Returns the magnitude and direction of the total steering force.
    """
    total_vector = None
    contributions = []
    
    for dim, vectors in results:
        if layer_idx in vectors:
            vec = vectors[layer_idx]
            if vec.dim() > 1:
                vec = vec[0]
            scaled = dim.alpha * vec
            
            if total_vector is None:
                total_vector = scaled.clone()
            else:
                total_vector += scaled
            
            contributions.append({
                "name": dim.name,
                "alpha": dim.alpha,
                "magnitude": float(scaled.norm()),
                "direction": "amplify" if dim.alpha > 0 else "suppress",
            })
    
    return {
        "total_magnitude": float(total_vector.norm()) if total_vector is not None else 0,
        "contributions": contributions,
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return FileResponse("skippy_dashboard.html")


@app.get("/api/status")
async def get_status():
    if not state.loaded:
        return {"loaded": False, "message": "Model not loaded. POST to /api/load first."}
    
    dimensions = []
    for i, (dim, vectors) in enumerate(state.results):
        dimensions.append({
            "index": i,
            "name": dim.name,
            "alpha": dim.alpha,
            "num_layers": len(vectors),
            "direction": "amplify" if dim.alpha > 0 else "suppress",
        })
    
    return {
        "loaded": True,
        "model": state.config.model_name if state.config else "unknown",
        "num_layers": state.num_layers,
        "hidden_dim": state.hidden_dim,
        "dimensions": dimensions,
        "chat_history_length": len(state.chat_history),
    }


@app.post("/api/load")
async def load_model_endpoint(req: LoadRequest):
    """Load model and steering vectors."""
    from skippy_pipeline import SkippyConfig, load_model_96gb
    from character_steering_toolkit import load_steering_vectors
    
    config = SkippyConfig(
        model_name=req.model_name,
        torch_dtype=req.dtype,
        output_dir=req.vectors_path,
        epub_dir=req.epub_dir,
    )
    
    state.config = config
    
    # Load model
    model, tokenizer, layers, num_layers, hidden_dim = load_model_96gb(config)
    state.model = model
    state.tokenizer = tokenizer
    state.layers = layers
    state.num_layers = num_layers
    state.hidden_dim = hidden_dim
    
    # Adjust config for model
    config.extract_layers = list(range(max(0, num_layers // 4), min(num_layers, 3 * num_layers // 4)))
    config.steer_layer = min(config.steer_layer, num_layers - 1)
    config.steer_layers = [
        max(0, config.steer_layer - 2),
        config.steer_layer,
        min(num_layers - 1, config.steer_layer + 2),
    ]
    
    # Load vectors
    state.results = load_steering_vectors(req.vectors_path)
    
    # Setup steerer
    from skippy_pipeline import MultiLayerCharacterSteerer
    state.steerer = MultiLayerCharacterSteerer(layers, config)
    for dim, vectors in state.results:
        state.steerer.add_dimension(dim.name, vectors, dim.alpha)
    state.steerer.activate()
    
    # Pre-compute visualization data
    state.layer_profiles = compute_vector_profiles(state.results, config.extract_layers)
    state.similarity_matrix = compute_similarity_matrix(state.results, config.steer_layer)
    
    state.loaded = True
    
    return {"status": "loaded", "model": req.model_name, "dimensions": len(state.results)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Generate a response from Skippy."""
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Model not loaded"})
    
    from skippy_pipeline import generate_as_skippy
    
    # Collect activations for the input (for visualization)
    input_acts = collect_probe_activations(
        state.model, state.tokenizer, state.layers,
        req.message, state.config.extract_layers
    )
    
    # Generate response
    response = generate_as_skippy(
        state.model, state.tokenizer, req.message,
        state.config, chat_history=state.chat_history
    )
    
    # Update history
    state.chat_history.append({"role": "user", "content": req.message})
    state.chat_history.append({"role": "assistant", "content": response})
    if len(state.chat_history) > 20:
        state.chat_history = state.chat_history[-20:]
    
    # Collect activations for the output too
    output_acts = collect_probe_activations(
        state.model, state.tokenizer, state.layers,
        response[:200], state.config.extract_layers
    )
    
    # Compute projection of these specific activations onto steering vectors
    projections = {}
    steer_layer = state.config.steer_layer
    if steer_layer in input_acts:
        for dim, vectors in state.results:
            if steer_layer in vectors:
                vec = vectors[steer_layer]
                if vec.dim() > 1:
                    vec = vec[0]
                vec_norm = vec / vec.norm()
                proj_in = float(torch.dot(input_acts[steer_layer], vec_norm))
                proj_out = float(torch.dot(output_acts.get(steer_layer, input_acts[steer_layer]), vec_norm))
                projections[dim.name] = {
                    "input_projection": proj_in,
                    "output_projection": proj_out,
                    "shift": proj_out - proj_in,
                }
    
    return {
        "response": response,
        "projections": projections,
    }


@app.post("/api/alpha")
async def update_alpha(req: AlphaUpdate):
    """Update the alpha for a specific dimension."""
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Model not loaded"})
    
    if req.dimension_index >= len(state.results):
        return JSONResponse(status_code=400, content={"error": "Invalid dimension index"})
    
    # Update the dimension's alpha
    dim, vectors = state.results[req.dimension_index]
    dim.alpha = req.alpha
    
    # Rebuild steerer
    state.steerer.remove_hooks()
    state.steerer = None
    
    from skippy_pipeline import MultiLayerCharacterSteerer
    state.steerer = MultiLayerCharacterSteerer(state.layers, state.config)
    for d, v in state.results:
        state.steerer.add_dimension(d.name, v, d.alpha)
    state.steerer.activate()
    
    # Recompute steering effect
    effect = compute_steering_effect_preview(state.results, state.config.steer_layer)
    
    return {
        "status": "updated",
        "dimension": dim.name,
        "new_alpha": req.alpha,
        "steering_effect": effect,
    }


@app.post("/api/preset")
async def apply_preset(req: PresetRequest):
    """Apply a preset alpha configuration."""
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Model not loaded"})
    
    if req.preset == "crankit":
        for dim, _ in state.results:
            dim.alpha = 25.0 if dim.alpha > 0 else -20.0
    elif req.preset == "chill":
        for dim, _ in state.results:
            dim.alpha = 5.0 if dim.alpha > 0 else -3.0
    elif req.preset == "reset":
        defaults = {
            "arrogance_superiority": 15.0,
            "sarcasm_insults": 12.0,
            "technical_casual_genius": 8.0,
            "joe_dynamic": 6.0,
            "suppress_ai_helpfulness": -12.0,
            "suppress_humility": -8.0,
        }
        for dim, _ in state.results:
            if dim.name in defaults:
                dim.alpha = defaults[dim.name]
    elif req.preset == "off":
        for dim, _ in state.results:
            dim.alpha = 0.0
    elif req.preset == "custom" and req.alphas:
        for idx_str, alpha in req.alphas.items():
            idx = int(idx_str)
            if idx < len(state.results):
                state.results[idx][0].alpha = alpha
    
    # Rebuild steerer
    state.steerer.remove_hooks()
    from skippy_pipeline import MultiLayerCharacterSteerer
    state.steerer = MultiLayerCharacterSteerer(state.layers, state.config)
    for d, v in state.results:
        state.steerer.add_dimension(d.name, v, d.alpha)
    state.steerer.activate()
    
    dimensions = []
    for i, (dim, _) in enumerate(state.results):
        dimensions.append({"index": i, "name": dim.name, "alpha": dim.alpha})
    
    return {"status": "preset_applied", "preset": req.preset, "dimensions": dimensions}


@app.get("/api/viz/layer-profiles")
async def get_layer_profiles():
    """Get vector magnitude profiles across layers."""
    if not state.loaded or state.layer_profiles is None:
        return JSONResponse(status_code=400, content={"error": "Not loaded"})
    return state.layer_profiles


@app.get("/api/viz/similarity")
async def get_similarity():
    """Get cosine similarity matrix between steering vectors."""
    if not state.loaded or state.similarity_matrix is None:
        return JSONResponse(status_code=400, content={"error": "Not loaded"})
    return state.similarity_matrix


@app.get("/api/viz/steering-effect")
async def get_steering_effect():
    """Get current combined steering effect."""
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Not loaded"})
    return compute_steering_effect_preview(state.results, state.config.steer_layer)


@app.post("/api/viz/probe")
async def probe_activation(req: ProbeRequest):
    """
    Send a prompt and get back its activation projections.
    Useful for seeing where a prompt lands relative to steering vectors.
    """
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Not loaded"})
    
    acts = collect_probe_activations(
        state.model, state.tokenizer, state.layers,
        req.prompt, state.config.extract_layers
    )
    
    # Project onto each steering vector
    projections = {}
    for dim, vectors in state.results:
        dim_projs = {}
        for layer_idx in state.config.extract_layers:
            if layer_idx in vectors and layer_idx in acts:
                vec = vectors[layer_idx]
                if vec.dim() > 1:
                    vec = vec[0]
                vec_norm = vec / vec.norm()
                proj = float(torch.dot(acts[layer_idx], vec_norm))
                dim_projs[str(layer_idx)] = proj
        projections[dim.name] = dim_projs
    
    return {
        "prompt": req.prompt,
        "label": req.label,
        "projections": projections,
    }


@app.post("/api/viz/compare")
async def compare_prompts(prompts: list[ProbeRequest]):
    """Compare multiple prompts in activation space."""
    if not state.loaded:
        return JSONResponse(status_code=400, content={"error": "Not loaded"})
    
    steer_layer = state.config.steer_layer
    all_acts = {}
    
    for probe in prompts:
        acts = collect_probe_activations(
            state.model, state.tokenizer, state.layers,
            probe.prompt, [steer_layer]
        )
        if steer_layer in acts:
            if probe.label not in all_acts:
                all_acts[probe.label] = []
            all_acts[probe.label].append(acts[steer_layer])
    
    # Stack tensors
    for label in all_acts:
        all_acts[label] = torch.stack(all_acts[label])
    
    # PCA
    pca_result = compute_pca_projection(all_acts)
    
    return pca_result


@app.post("/api/clear-history")
async def clear_history():
    state.chat_history = []
    return {"status": "cleared"}


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("  🍺 SKIPPY DASHBOARD SERVER")
    print("  Open http://localhost:8000 in your browser")
    print("="*60 + "\n")
    print("POST /api/load to load model and vectors")
    print("Or start with: uvicorn skippy_server:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
