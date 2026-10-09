from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ABIDE"
PROMPT_DIR = EXP_DIR / "prompts"
SPLIT_DIR = EXP_DIR / "splits"
CACHE_DIR = EXP_DIR / "cache"
RAW_DIR = EXP_DIR / "raw"
LOG_DIR = EXP_DIR / "logs"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
RESULT_DIR = EXP_DIR / "results"
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
PROMPT_FILES = {
    "P0": PROMPT_DIR / "p0_original.json",
    "P1": PROMPT_DIR / "p1_modality_aware.json",
    "P2": PROMPT_DIR / "p2_modality_mapping.json",
    "P3": PROMPT_DIR / "p3_matched_generic.json",
}
EXPECTED_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_prompt_texts(config: str) -> list[list[list[str]]]:
    """Return texts as [modality][theme][prompt], with four texts per theme."""
    if config == "P0":
        raw = json.loads(PROMPT_FILES["P0"].read_text(encoding="utf-8"))
        themes = [[item["text"] for item in theme["prompts"]] for theme in raw["themes"]]
        return [[list(theme) for theme in themes] for _ in MODALITIES]

    if config in ("P1", "P3"):
        raw = json.loads(PROMPT_FILES[config].read_text(encoding="utf-8"))
        return [
            [[item["text"] for item in theme["prompts"]] for theme in raw["modalities"][modality]["themes"]]
            for modality in MODALITIES
        ]

    if config == "P2":
        mapping = json.loads(PROMPT_FILES["P2"].read_text(encoding="utf-8"))["mapping"]
        p1 = load_prompt_texts("P1")
        by_modality = {name: p1[i] for i, name in enumerate(MODALITIES)}
        return [by_modality[mapping[modality]] for modality in MODALITIES]
    raise ValueError(f"unknown config {config}")


def prompt_digest(config: str) -> str:
    packed = json.dumps(load_prompt_texts(config), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(packed)


def load_data():
    feature_dict = np.load(DATA_DIR / "modal_feat_dict.npy", allow_pickle=True).item()
    frame = pd.read_csv(DATA_DIR / "processed_standard_data.csv")
    if list(feature_dict) != MODALITIES:
        raise ValueError(f"modality order mismatch: {list(feature_dict)}")
    dims = [len(feature_dict[name]) for name in MODALITIES]
    expected_columns = [column for name in MODALITIES for column in feature_dict[name]]
    if frame.columns[:-1].tolist() != expected_columns:
        raise ValueError("CSV columns do not exactly match modal_feat_dict.npy order")
    if frame.columns[-1] != "label":
        raise ValueError(f"expected final label column, got {frame.columns[-1]!r}")
    if frame.shape != (871, 321):
        raise ValueError(f"expected ABIDE shape (871, 321), got {frame.shape}")
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True)
    raw_label = frame.iloc[:, -1].to_numpy(dtype=np.int64, copy=True)
    if set(np.unique(raw_label)) != {1, 2}:
        raise ValueError(f"expected raw DX_GROUP labels 1/2, got {np.unique(raw_label)}")
    y = raw_label - 1
    return x, y, dims, feature_dict


def create_splits(y: np.ndarray) -> dict:
    result = {"n_samples": int(len(y)), "seeds": {}}
    for seed in (0, 1, 2):
        folds = []
        outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        dummy_x = np.zeros((len(y), 1), dtype=np.uint8)
        for fold_id, (outer_train, test_idx) in enumerate(outer.split(dummy_x, y), start=0):
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.1,
                random_state=seed + 1000 + fold_id,
            )
            train_local, val_local = next(inner.split(dummy_x[outer_train], y[outer_train]))
            train_idx = outer_train[train_local]
            val_idx = outer_train[val_local]
            if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
                raise AssertionError("train/validation/test indices overlap")
            if len(train_idx) + len(val_idx) + len(test_idx) != len(y):
                raise AssertionError("fold split does not cover every subject")
            folds.append({
                "fold": fold_id,
                "model_seed": seed * 1000 + fold_id,
                "train": train_idx.astype(int).tolist(),
                "validation": val_idx.astype(int).tolist(),
                "test": test_idx.astype(int).tolist(),
                "train_sha256": sha256_bytes(train_idx.astype("<i8").tobytes()),
                "validation_sha256": sha256_bytes(val_idx.astype("<i8").tobytes()),
                "test_sha256": sha256_bytes(test_idx.astype("<i8").tobytes()),
            })
        result["seeds"][str(seed)] = folds
    return result


def split_path(seed: int) -> Path:
    return SPLIT_DIR / f"seed{seed}.json"


def load_or_create_splits(y: np.ndarray) -> None:
    for seed in (0, 1, 2):
        path = split_path(seed)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            new = create_splits(y)["seeds"][str(seed)]
            if saved.get("folds") != new:
                raise ValueError(f"stored split changed for seed {seed}")
        else:
            all_splits = create_splits(y)
            atomic_json(path, {"seed": seed, "folds": all_splits["seeds"][str(seed)]})


def get_prompt_features(config: str, device: torch.device) -> torch.Tensor:
    prompt_hash = prompt_digest(config)
    weights = Path(os.environ.get("CLIP_RN50_WEIGHTS", "/root/.cache/clip/RN50.pt"))
    if not weights.is_file():
        raise FileNotFoundError(f"local RN50 weights not found: {weights}")
    weights_hash = sha256_file(weights)
    if weights_hash != EXPECTED_RN50_SHA256:
        raise ValueError(f"RN50 checksum mismatch: {weights_hash}")
    encode_device = torch.device(
        os.environ.get("MASPE_CLIP_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    )
    precision_tag = "fp16" if encode_device.type == "cuda" else "fp32"
    cache_path = CACHE_DIR / f"{config}_{prompt_hash}_{weights_hash[:12]}_{precision_tag}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if tuple(cached.shape) == (4, 2, 4, 1024):
            return cached.to(device)

    import sys
    src_dir = str(EXP_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from CLIP.clip import clip

    model, _ = clip.load(str(weights), device=str(encode_device))
    model.eval()
    texts = load_prompt_texts(config)
    flat = (
        [text for theme in texts[0] for text in theme]
        if config == "P0"
        else [text for modality in texts for theme in modality for text in theme]
    )
    tokens = clip.tokenize(flat).to(encode_device)
    expected_texts = 8 if config == "P0" else 32
    if tokens.shape != (expected_texts, 77):
        raise ValueError(f"expected {expected_texts} prompt sequences, got {tuple(tokens.shape)}")
    with torch.no_grad():
        encoded = model.encode_text(tokens).float()
        if config == "P0":
            features = encoded.reshape(1, 2, 4, 1024).expand(4, -1, -1, -1).clone().cpu()
        else:
            features = encoded.reshape(4, 2, 4, 1024).cpu()
    if not torch.isfinite(features).all():
        raise ValueError(f"non-finite prompt embeddings for {config}")
    atomic_torch_save(cache_path, features)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features.to(device)


def task_raw_path(config: str, seed: int, fold: int) -> Path:
    return RAW_DIR / f"{config}_seed{seed}_fold{fold:02d}.json"


def task_pred_path(config: str, seed: int, fold: int) -> Path:
    return RAW_DIR / f"{config}_seed{seed}_fold{fold:02d}_predictions.csv"


def task_checkpoint_path(config: str, seed: int, fold: int) -> Path:
    return CHECKPOINT_DIR / f"{config}_seed{seed}_fold{fold:02d}.pt"


def task_prompt_analysis_path(config: str, seed: int, fold: int) -> Path:
    return RAW_DIR / f"{config}_seed{seed}_fold{fold:02d}_prompt_analysis.csv"


def task_status_path(config: str, seed: int) -> Path:
    return RAW_DIR / f"{config}_seed{seed}_task.json"
