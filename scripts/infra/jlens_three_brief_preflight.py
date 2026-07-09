#!/usr/bin/env python3
"""Preflight report for the J-lens three-brief overnight runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    for mod in ["torch", "transformers", "accelerate", "peft", "datasets", "jlens"]:
        print(mod, "FOUND" if importlib.util.find_spec(mod) else "MISSING")
    for model in [
        "/home/orwel/dev_genius/models/Qwen3.5-27B",
        "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B",
        "/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base",
    ]:
        path = Path(model)
        safetensors = len(list(path.rglob("*.safetensors"))) if path.exists() else 0
        print(model, "exists", path.exists(), "safetensors", safetensors)


if __name__ == "__main__":
    main()
