#!/usr/bin/env python3
"""Shared utilities for J-lens experiments.

The helpers here are intentionally conservative: they check local model/lens
cache state before any heavy load, write append-only event logs, and keep
analysis routines usable in synthetic smoke tests as well as real pilots.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


@dataclass
class RunLogger:
    output_dir: Path

    @property
    def events_path(self) -> Path:
        return self.output_dir / "events.jsonl"

    def log(self, event: str, **payload: Any) -> None:
        append_jsonl(self.events_path, {"time": now_iso(), "event": event, **payload})


def git_snapshot() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                args,
                cwd=PROJECT_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def hf_cache_roots() -> list[Path]:
    env_home = os.environ.get("HF_HOME")
    roots: list[Path] = []
    if env_home:
        env_path = Path(env_home)
        roots.extend([env_path, env_path / "hub"])
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def model_cached(model_name_or_path: str | Path) -> bool:
    return model_cache_report(model_name_or_path)["cached"]


def model_cache_report(model_name_or_path: str | Path) -> dict[str, Any]:
    raw = str(model_name_or_path)
    checked_paths: list[str] = []
    local_path = Path(raw).expanduser()
    candidate_dirs: list[Path] = []
    if local_path.exists():
        candidate_dirs.append(local_path)
    if "/" in raw and not local_path.exists():
        safe_name = "models--" + raw.replace("/", "--")
        for root in hf_cache_roots():
            candidate_dirs.append(root / safe_name)
    safetensors: list[str] = []
    bin_files: list[str] = []
    for candidate in candidate_dirs:
        checked_paths.append(str(candidate))
        if not candidate.exists():
            continue
        safetensors.extend(str(path) for path in candidate.rglob("*.safetensors"))
        bin_files.extend(str(path) for path in candidate.rglob("*.bin"))
    return {
        "model": raw,
        "cached": bool(safetensors or bin_files),
        "safetensors_count": len(safetensors),
        "bin_count": len(bin_files),
        "checked_paths": checked_paths,
        "sample_files": (safetensors + bin_files)[:5],
    }


def require_cached_model(model_name_or_path: str | Path) -> dict[str, Any]:
    report = model_cache_report(model_name_or_path)
    print(f"Model cache status for {model_name_or_path}: {report['cached']}", flush=True)
    if not report["cached"]:
        raise FileNotFoundError(
            f"Model {model_name_or_path} is not cached locally; refusing to download."
        )
    return report


def lens_cache_report(repo_id: str, filename: str) -> dict[str, Any]:
    repo_dir = "models--" + repo_id.replace("/", "--")
    candidates: list[Path] = []
    checked_roots: list[str] = []
    basename = Path(filename).name
    for root in hf_cache_roots():
        checked_roots.append(str(root))
        repo_path = root / repo_dir
        if repo_path.exists():
            candidates.extend(repo_path.rglob(basename))
    return {
        "repo_id": repo_id,
        "filename": filename,
        "cached": bool(candidates),
        "checked_roots": checked_roots,
        "candidates": [str(path) for path in sorted(candidates)],
    }


def resolve_lens_path(repo_id: str, filename: str, allow_download: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    status = lens_cache_report(repo_id, filename)
    print(f"Lens cache status for {repo_id}/{filename}: {status['cached']}", flush=True)
    if status["cached"]:
        return Path(
            hf_hub_download(
                repo_id,
                filename=filename,
                repo_type="model",
                local_files_only=True,
            )
        )
    if not allow_download:
        raise FileNotFoundError(
            f"Lens {repo_id}/{filename} is not cached locally; refusing to download."
        )
    return Path(hf_hub_download(repo_id, filename=filename, repo_type="model"))


def load_lens(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "J" not in obj:
        raise ValueError(f"{path} does not look like a Jacobian lens checkpoint")
    return obj


def lens_layers(lens: dict[str, Any]) -> list[int]:
    return sorted(int(layer) for layer in lens["J"].keys())


def select_even_layers(available_layers: list[int], count: int) -> list[int]:
    if not available_layers:
        raise ValueError("No available layers")
    if count >= len(available_layers):
        return list(available_layers)
    indices = np.linspace(0, len(available_layers) - 1, num=count)
    return [available_layers[int(round(idx))] for idx in indices]


def normalized_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows.float() / rows.float().norm(dim=1, keepdim=True).clamp_min(1e-12)


def top_singular_basis(j_matrix: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    hidden_dim = int(j_matrix.shape[1])
    rank = min(rank, hidden_dim)
    if rank <= 0:
        raise ValueError("rank must be positive")
    if hidden_dim <= 256 or rank >= hidden_dim // 2:
        _, _, vh = torch.linalg.svd(j_matrix.float(), full_matrices=False)
        return vh[:rank].T.contiguous()
    q = min(hidden_dim, rank + 32)
    _, _, v = torch.svd_lowrank(j_matrix.float(), q=q, niter=niter)
    return v[:, :rank].contiguous()


def random_basis(hidden_dim: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(hidden_dim, rank, generator=generator)
    q, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return q[:, :rank].contiguous()


def project_rows(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return rows.float() @ basis.float() @ basis.float().T


def complement_rows(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return rows.float() - project_rows(rows, basis)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = normalized_rows(a.float())
    b_norm = normalized_rows(b.float())
    return a_norm @ b_norm.T


def stable_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def cv_balanced_accuracy(
    features: np.ndarray,
    labels: list[str],
    groups: list[str] | None,
    seed: int,
    n_splits: int = 5,
) -> dict[str, Any]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    if len(set(y.tolist())) < 2:
        return {"balanced_accuracy": float("nan"), "folds": [], "classes": encoder.classes_.tolist()}

    x = np.asarray(features, dtype=np.float32)
    splitter: Iterable[tuple[np.ndarray, np.ndarray]]
    if groups is not None and len(set(groups)) >= 2:
        unique_groups = len(set(groups))
        folds = min(n_splits, unique_groups)
        splitter = GroupKFold(n_splits=folds).split(x, y, groups=np.asarray(groups))
        splitter_name = "GroupKFold"
    else:
        class_counts = np.bincount(y)
        folds = max(2, min(n_splits, int(class_counts.min())))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y)
        splitter_name = "StratifiedKFold"

    fold_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed + fold_idx,
                solver="lbfgs",
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        score = stable_balanced_accuracy(y[test_idx], pred)
        scores.append(score)
        fold_rows.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": score,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
            }
        )
    return {
        "balanced_accuracy": float(np.nanmean(scores)) if scores else float("nan"),
        "folds": fold_rows,
        "splitter": splitter_name,
        "classes": encoder.classes_.tolist(),
    }


def label_shuffle_null(
    features: np.ndarray,
    labels: list[str],
    groups: list[str] | None,
    seed: int,
    n: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    labels_arr = np.asarray(labels)
    for idx in range(n):
        shuffled = labels_arr.copy()
        rng.shuffle(shuffled)
        result = cv_balanced_accuracy(features, shuffled.tolist(), groups, seed + idx)
        scores.append(float(result["balanced_accuracy"]))
    finite = [score for score in scores if np.isfinite(score)]
    return {
        "n": n,
        "scores": scores,
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "std": float(np.std(finite)) if finite else float("nan"),
    }


def tfidf_text_baseline(
    texts: list[str],
    labels: list[str],
    groups: list[str] | None,
    seed: int,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    features = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    result = cv_balanced_accuracy(features, labels, groups, seed)
    result["feature_space"] = "tfidf_text"
    result["vocabulary_size"] = len(vectorizer.vocabulary_)
    return result


def projection_metric_distance(basis_a: torch.Tensor, basis_b: torch.Tensor) -> float:
    rank = min(int(basis_a.shape[1]), int(basis_b.shape[1]))
    a = basis_a[:, :rank].float()
    b = basis_b[:, :rank].float()
    overlap = torch.linalg.matrix_norm(a.T @ b, ord="fro").pow(2)
    value = torch.sqrt(torch.clamp(torch.tensor(float(rank)) - overlap, min=0.0))
    return float((value / max(rank, 1) ** 0.5).item())


def principal_angle_summary(basis_a: torch.Tensor, basis_b: torch.Tensor) -> dict[str, Any]:
    rank = min(int(basis_a.shape[1]), int(basis_b.shape[1]))
    singular = torch.linalg.svdvals(basis_a[:, :rank].float().T @ basis_b[:, :rank].float())
    clipped = torch.clamp(singular, -1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(clipped))
    return {
        "mean_deg": float(angles.mean().item()),
        "max_deg": float(angles.max().item()),
        "min_deg": float(angles.min().item()),
    }


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    numerator = torch.linalg.matrix_norm(a_f.T @ b_f, ord="fro").pow(2)
    denom_a = torch.linalg.matrix_norm(a_f.T @ a_f, ord="fro")
    denom_b = torch.linalg.matrix_norm(b_f.T @ b_f, ord="fro")
    denom = denom_a * denom_b
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((numerator / denom).item())


def normalized_frobenius_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.matrix_norm(a.float(), ord="fro").clamp_min(1e-12)
    return float((torch.linalg.matrix_norm(a.float() - b.float(), ord="fro") / denom).item())


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)
