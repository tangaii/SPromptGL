from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import torch


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False)
        with open(tmp_name, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def fold_artifacts_valid(summary_path, prediction_path, prompt_path, checkpoint_path, expected_test):
    paths = [Path(summary_path), Path(prediction_path), Path(prompt_path), Path(checkpoint_path)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        import json

        summary = json.loads(paths[0].read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            return False
        predictions = pd.read_csv(paths[1])
        if len(predictions) != len(expected_test) or predictions["row_index"].duplicated().any():
            return False
        if set(predictions["row_index"].astype(int)) != set(map(int, expected_test)):
            return False
        if not predictions[["prob_asd", "prob_nc"]].apply(pd.to_numeric, errors="coerce").notna().all().all():
            return False
        prompt = pd.read_csv(paths[2])
        if len(prompt) != 4 or prompt["modality"].nunique() != 4:
            return False
        checkpoint = torch.load(paths[3], map_location="cpu", weights_only=True)
        if checkpoint.get("status") != "COMPLETE":
            return False
        if summary.get("checkpoint_sha256") != _sha256_file(paths[3]):
            return False
        if summary.get("prediction_sha256") != _sha256_file(paths[1]):
            return False
        return True
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
