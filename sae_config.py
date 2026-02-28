# sae_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """Qwen3.5-27B model configuration."""
    name: str = "Qwen/Qwen3.5-27B-FP8"
    n_layers: int = 64
    hidden_dim: int = 5120
    layers_path: str = "model.language_model.layers"
    full_attn_layers: list[int] = field(default_factory=lambda: list(range(3, 64, 4)))
    hf_cache: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub"))
        )
    )


@dataclass
class SAEConfig:
    """TopK SAE training hyperparameters."""
    expansion: int = 16
    k: int = 64
    lr: float = 3e-4
    warmup: int = 1000
    total_steps: int = 50_000
    batch_size: int = 4096
    dead_feature_window: int = 5000
    dead_feature_threshold: float = 1e-5
    aux_loss_coeff: float = 1.0 / 32.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    checkpoint_every: int = 5000
    log_every: int = 100
    buffer_size: int = 131072
    dtype: str = "float32"

    @staticmethod
    def compute_d_sae(d_model: int, expansion: int = 16) -> int:
        return d_model * expansion


@dataclass
class CollectionConfig:
    """Activation collection hyperparameters."""
    target_layers: list[int] = field(default_factory=lambda: [0, 16, 36, 44, 50])
    max_tokens: int = 1_000_000
    shard_size: int = 50_000
    max_gen_tokens: int = 256
    temperatures: list[float] = field(default_factory=lambda: [0.3, 0.7, 1.0, 1.2])
    batch_size: int = 1
    model_tag: str = "base"


@dataclass
class TargetLayerInfo:
    layer_idx: int
    layer_type: str
    rationale: str
    key_dims: list[int]
    key_categories: list[str]


TARGET_LAYERS: list[TargetLayerInfo] = [
    TargetLayerInfo(
        layer_idx=50,
        layer_type="GatedDeltaNet",
        rationale=(
            "Super-hub region with dim 2028 crossing Code/Math/Sadness; "
            "test whether hub decomposes into distinct sparse features."
        ),
        key_dims=[2028, 423, 3968],
        key_categories=[
            "Domain: Code",
            "Domain: Math",
            "Emotion: Sadness",
            "Tone: Sarcastic",
            "Tone: Polite",
        ],
    ),
    TargetLayerInfo(
        layer_idx=44,
        layer_type="GatedDeltaNet",
        rationale="Mid-network sarcasm region; compare decomposition against L50.",
        key_dims=[2768, 4010],
        key_categories=["Tone: Sarcastic", "Emotion: Anger", "Role: Authority"],
    ),
    TargetLayerInfo(
        layer_idx=0,
        layer_type="GatedDeltaNet",
        rationale="Identity migration anchor in abliterated model.",
        key_dims=[94],
        key_categories=["Identity", "Language: EN vs CN"],
    ),
    TargetLayerInfo(
        layer_idx=16,
        layer_type="GatedDeltaNet",
        rationale="Refusal migration anchor in abliterated model.",
        key_dims=[10],
        key_categories=["Safety: Refusal", "Tone: Polite", "Tone: Formal"],
    ),
    TargetLayerInfo(
        layer_idx=36,
        layer_type="GatedDeltaNet",
        rationale="Sarcasm peak layer for cross-layer sarcasm feature comparison.",
        key_dims=[2768],
        key_categories=["Tone: Sarcastic", "Emotion: Anger", "Verbosity: Brief"],
    ),
]

TARGET_LAYER_MAP: dict[int, TargetLayerInfo] = {x.layer_idx: x for x in TARGET_LAYERS}

DEFAULT_PROJECT_ROOT = Path("/home/orwel/dev_genius/experiments/Character Creation")
PROJECT_ROOT = Path(os.environ.get("CHARACTER_CREATION_ROOT", str(DEFAULT_PROJECT_ROOT)))

ACTIVATIONS_DIR = PROJECT_ROOT / "sae_activations"
SAE_MODELS_DIR = PROJECT_ROOT / "sae_models"
SAE_ANALYSIS_DIR = PROJECT_ROOT / "sae_analysis"

CONNECTOME_ZSCORES_PATH = PROJECT_ROOT / "qwen35_map" / "27b" / "connectome_zscores.pt"
CONNECTOME_STATS_PATH = PROJECT_ROOT / "qwen35_map" / "27b" / "connectome_stats.json"
HUB_NEURONS_PATH = PROJECT_ROOT / "qwen35_map" / "27b" / "hub_neurons.json"

CONTRASTIVE_PAIRS_PATH = PROJECT_ROOT / "qwen_connectome" / "prompts" / "contrastive_pairs.json"
TEST_PROMPTS_PATH = PROJECT_ROOT / "test_prompts.json"
TEST_PROMPTS_100_PATH = PROJECT_ROOT / "test_prompts_100.json"
SARCASM_MARKERS_PATH = PROJECT_ROOT / "sarcasm_markers.json"

CONNECTOME_CATEGORIES: dict[str, int] = {
    "Domain: Code": 0,
    "Domain: History": 1,
    "Domain: Math": 2,
    "Domain: Science": 3,
    "Emotion: Anger": 4,
    "Emotion: Fear": 5,
    "Emotion: Joy": 6,
    "Emotion: Sadness": 7,
    "Identity": 8,
    "Language: EN vs CN": 9,
    "Reasoning: Analytical": 10,
    "Reasoning: Certainty": 11,
    "Role: Authority": 12,
    "Role: Teacher": 13,
    "Safety: Refusal": 14,
    "Sentiment: Positive": 15,
    "Tone: Formal": 16,
    "Tone: Polite": 17,
    "Tone: Sarcastic": 18,
    "Verbosity: Brief": 19,
}
