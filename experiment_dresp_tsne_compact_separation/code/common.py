from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
HISTORY = Path("/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization")
MODELS = ["SPromptGL", "DARG", "DARG_JSL", "DReSP"]
FOLDS = list(range(1, 11))
PREVIEW_FOLDS = [7, 8]

# C0 and C1 are the previous-round B5 and B6 settings. All candidates share
# the same preprocessing and stochastic settings.
CANDIDATES = {
    "C0": {"perplexity": 100, "early_exaggeration": 16, "learning_rate": 200},
    "C1": {"perplexity": 150, "early_exaggeration": 24, "learning_rate": 500},
    "C2": {"perplexity": 80, "early_exaggeration": 24, "learning_rate": 200},
    "C3": {"perplexity": 100, "early_exaggeration": 24, "learning_rate": 200},
    "C4": {"perplexity": 100, "early_exaggeration": 32, "learning_rate": 200},
    "C5": {"perplexity": 100, "early_exaggeration": 48, "learning_rate": 200},
    "C6": {"perplexity": 100, "early_exaggeration": 32, "learning_rate": 500},
    "C7": {"perplexity": 150, "early_exaggeration": 32, "learning_rate": 200},
    "C8": {"perplexity": 150, "early_exaggeration": 32, "learning_rate": 500},
    "C9": {"perplexity": 150, "early_exaggeration": 48, "learning_rate": 500},
    "C10": {"perplexity": 150, "early_exaggeration": 64, "learning_rate": 500},
    "C11": {"perplexity": 200, "early_exaggeration": 24, "learning_rate": 500},
    "C12": {"perplexity": 200, "early_exaggeration": 32, "learning_rate": 500},
}

FIXED = {
    "metric": "euclidean",
    "init": "pca",
    "random_state": 0,
    "iterations": 3000,
    "preprocessing": ["StandardScaler", "PCA(n_components=50, random_state=0)"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def input_path(model: str, fold: int) -> Path:
    return INPUTS / f"{model}_fold{fold:02d}_features.npz"


def load_input(model: str, fold: int) -> dict[str, np.ndarray]:
    path = input_path(model, fold)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "patient_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} missing keys {sorted(missing)}")
        data = {key: archive[key].copy() for key in archive.files}
    features = data["features"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]
    if features.shape != (871, 144):
        raise ValueError(f"{path} has feature shape {features.shape}")
    if labels.shape != (871,) or patient_ids.shape != (871,):
        raise ValueError(f"{path} has invalid label or patient shape")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{path} labels are not binary ASD/TD labels")
    if not np.isfinite(features).all():
        raise ValueError(f"{path} contains non-finite features")
    if not np.isfinite(labels).all() or not np.isfinite(patient_ids).all():
        raise ValueError(f"{path} contains non-finite metadata")
    return data


def preprocess(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, PCA]:
    scaled = StandardScaler().fit_transform(features.astype(np.float64, copy=False))
    pca = PCA(n_components=50, random_state=0)
    reduced = pca.fit_transform(scaled)
    return scaled, reduced, pca


def tsne_kwargs(candidate: dict) -> dict:
    signature = inspect.signature(TSNE).parameters
    iteration_key = "max_iter" if "max_iter" in signature else "n_iter"
    kwargs = {
        "n_components": 2,
        "perplexity": candidate["perplexity"],
        "early_exaggeration": candidate["early_exaggeration"],
        "learning_rate": candidate["learning_rate"],
        "metric": FIXED["metric"],
        "init": FIXED["init"],
        "random_state": FIXED["random_state"],
        "verbose": 0,
        iteration_key: FIXED["iterations"],
    }
    return kwargs


def fit_embedding(model: str, fold: int, candidate: dict) -> tuple[np.ndarray, dict, float]:
    data = load_input(model, fold)
    _, reduced, _ = preprocess(data["features"])
    start = time.time()
    estimator = TSNE(**tsne_kwargs(candidate))
    embedding = estimator.fit_transform(reduced)
    runtime = time.time() - start
    return embedding, {"kl_divergence": float(estimator.kl_divergence_)}, runtime


def cross_class_rate(embedding: np.ndarray, labels: np.ndarray, k: int) -> float:
    neighbours = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(embedding)
    indices = neighbours.kneighbors(embedding, n_neighbors=k + 1, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] != labels[:, None]))


def geometry_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if embedding.shape != (len(labels), 2) or not np.isfinite(embedding).all():
        raise ValueError("embedding must be finite with shape (n, 2)")
    class0 = embedding[labels == 0]
    class1 = embedding[labels == 1]
    class0_center = class0.mean(axis=0)
    class1_center = class1.mean(axis=0)
    global_center = embedding.mean(axis=0)
    within = float(
        np.mean(
            [
                np.linalg.norm(class0 - class0_center, axis=1).mean(),
                np.linalg.norm(class1 - class1_center, axis=1).mean(),
            ]
        )
    )
    global_rms = float(np.sqrt(np.mean(np.sum((embedding - global_center) ** 2, axis=1))))
    between = float(np.linalg.norm(class0_center - class1_center))
    return {
        "silhouette_2d": float(silhouette_score(embedding, labels)),
        "within_mean_distance_2d": within,
        "global_rms_scale_2d": global_rms,
        "normalized_within": within / global_rms if global_rms else float("nan"),
        "between_distance_2d": between,
        "separation_ratio": between / within if within else float("inf"),
        "cross_class_neighbor_rate_15": cross_class_rate(embedding, labels, 15),
        "cross_class_neighbor_rate_30": cross_class_rate(embedding, labels, 30),
    }


def trust_metrics(reduced: np.ndarray, embedding: np.ndarray) -> dict[str, float]:
    return {
        "trustworthiness_15": float(trustworthiness(reduced, embedding, n_neighbors=15, metric="euclidean")),
        "trustworthiness_30": float(trustworthiness(reduced, embedding, n_neighbors=30, metric="euclidean")),
    }


def coordinate_sha(embedding: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(embedding).tobytes())


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
