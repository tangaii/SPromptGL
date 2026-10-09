from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from artifacts import atomic_csv, fold_artifacts_valid
from common import (
    EXP_DIR,
    MODALITIES,
    atomic_json,
    atomic_torch_save,
    get_prompt_features,
    load_data,
    prompt_digest,
    sha256_file,
    split_path,
    task_checkpoint_path,
    task_pred_path,
    task_prompt_analysis_path,
    task_raw_path,
    task_status_path,
)
from metrics import classification_metrics
from model import EvalHelper, make_initial_trainable_state, parameter_state_sha256


HYPERPARAMS = {
    "GC_mode": "weighted-cosine",
    "MF_mode": "concat",
    "MP_mode": "GCN",
    "mode": "simple-2",
    "dropout": 0.35,
    "lr": 0.0038,
    "n_head": 2,
    "n_hidden": 18,
    "nlayer": 1,
    "reg": 0.11,
    "nepoch": 450,
    "early": 200,
    "th": 0.9,
    "theta_smooth": 1.0,
    "theta_degree": 0.5,
    "theta_sparsity": 0.0,
    "nclass": 2,
    "nmodal": 4,
}


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_device(device_string: str) -> torch.device:
    device = torch.device(device_string)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def split_task(config: str, seed: int, device_string: str, max_epochs: int, patience: int, force: bool) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = set_device(device_string)
    x, y, dims, _ = load_data()
    split_doc = json.loads(split_path(seed).read_text(encoding="utf-8"))
    split_folds = split_doc["folds"]
    prompt_features = get_prompt_features(config, device)
    hyperpm = SimpleNamespace(**HYPERPARAMS)
    completed_folds = 0
    task_started = time.time()
    progress_path = task_status_path(config, seed)
    atomic_json(progress_path, {
        "config": config,
        "seed": seed,
        "status": "RUNNING",
        "started_at_unix": task_started,
        "completed_folds": 0,
        "expected_folds": 10,
    })

    for split in split_folds:
        fold_id = int(split["fold"])
        train_idx = np.asarray(split["train"], dtype=np.int64)
        val_idx = np.asarray(split["validation"], dtype=np.int64)
        test_idx = np.asarray(split["test"], dtype=np.int64)
        raw_path = task_raw_path(config, seed, fold_id)
        pred_path = task_pred_path(config, seed, fold_id)
        prompt_path = task_prompt_analysis_path(config, seed, fold_id)
        checkpoint_path = task_checkpoint_path(config, seed, fold_id)

        if not force and fold_artifacts_valid(raw_path, pred_path, prompt_path, checkpoint_path, test_idx):
            completed_folds += 1
            print(f"[{config} seed={seed}] fold {fold_id + 1}/10 already verified; skipping", flush=True)
            continue

        atomic_json(raw_path, {
            "config": config,
            "seed": seed,
            "fold": fold_id,
            "status": "RUNNING",
            "started_at_unix": time.time(),
        })
        model_seed = int(split["model_seed"])
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        agent = EvalHelper(dims, x, y, hyperpm, train_idx, prompt_features, config, device)
        canonical_state = make_initial_trainable_state(dims, hyperpm, prompt_features, config, model_seed)
        initial_sha = parameter_state_sha256(canonical_state)
        agent.copy_trainable_state(canonical_state)
        if agent.parameter_sha256() != initial_sha:
            raise AssertionError(f"initial state mismatch for {config}, seed={seed}, fold={fold_id}")
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)

        best_val_acc = -1.0
        best_epoch = 0
        wait = 0
        best_gfr_state = None
        best_pln_state = None
        train_started = time.time()
        epochs_run = 0
        for epoch in range(max_epochs):
            train_values = agent.train_step()
            probabilities, _, _, _ = agent.predict_all()
            val_acc = float(np.mean(probabilities[val_idx].argmax(axis=1) == y[val_idx]))
            epochs_run = epoch + 1
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epochs_run
                wait = 0
                best_gfr_state = {name: tensor.detach().cpu().clone() for name, tensor in agent.MessagePassing0.state_dict().items()}
                best_pln_state = {name: tensor.detach().cpu().clone() for name, tensor in agent.ModalFusion.state_dict().items()}
            else:
                wait += 1
            if epochs_run == 1 or epochs_run % 10 == 0:
                print(
                    f"[{config} seed={seed} fold={fold_id + 1}/10] epoch={epochs_run} "
                    f"loss={train_values['loss']:.5f} val_acc={val_acc:.5f} "
                    f"best={best_val_acc:.5f}@{best_epoch} wait={wait}",
                    flush=True,
                )
            if wait > patience:
                break
        train_seconds = time.time() - train_started
        if best_gfr_state is None or best_pln_state is None:
            raise RuntimeError("training did not produce a validation checkpoint")
        agent.MessagePassing0.load_state_dict(best_gfr_state)
        agent.ModalFusion.load_state_dict(best_pln_state)

        test_started = time.time()
        probabilities, _, _, diagnostics = agent.predict_all(capture_prompt_diagnostics=True)
        test_seconds = time.time() - test_started
        test_probability = probabilities[test_idx]
        test_metrics = classification_metrics(y[test_idx], test_probability)
        predictions = pd.DataFrame({
            "config": config,
            "seed": seed,
            "fold": fold_id,
            "row_index": test_idx,
            "y_true": y[test_idx],
            "prob_asd": test_probability[:, 0],
            "prob_nc": test_probability[:, 1],
            "pred_label": test_probability.argmax(axis=1),
        })
        atomic_csv(predictions, pred_path)
        prompt_frame = prompt_diagnostics_frame(diagnostics, test_idx, config, seed, fold_id)
        atomic_csv(prompt_frame, prompt_path)

        checkpoint = {
            "status": "COMPLETE",
            "config": config,
            "seed": seed,
            "fold": fold_id,
            "best_epoch": best_epoch,
            "initial_parameter_sha256": initial_sha,
            "gfr_state": best_gfr_state,
            "pln_state": best_pln_state,
        }
        atomic_torch_save(checkpoint_path, checkpoint)
        record = {
            "status": "COMPLETE",
            "config": config,
            "seed": seed,
            "fold": fold_id,
            "model_seed": model_seed,
            "train_size": int(len(train_idx)),
            "validation_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "train_index_sha256": split["train_sha256"],
            "validation_index_sha256": split["validation_sha256"],
            "test_index_sha256": split["test_sha256"],
            "initial_parameter_sha256": initial_sha,
            "prompt_sha256": prompt_digest(config),
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "best_validation_accuracy": best_val_acc,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "total_fold_seconds": train_seconds + test_seconds,
            **test_metrics,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_sha256": sha256_file(pred_path),
            "completed_at_unix": time.time(),
        }
        atomic_json(raw_path, record)
        completed_folds += 1
        print(
            f"[{config} seed={seed}] fold {fold_id + 1}/10 COMPLETE "
            f"BA={test_metrics['balanced_accuracy']:.4f} AUC={test_metrics['probability_auc_asd_positive']:.4f} "
            f"epoch={best_epoch}/{epochs_run} elapsed={train_seconds:.1f}s",
            flush=True,
        )
        atomic_json(progress_path, {
            "config": config,
            "seed": seed,
            "status": "RUNNING",
            "started_at_unix": task_started,
            "completed_folds": completed_folds,
            "expected_folds": 10,
        })

    valid = 0
    for split in split_folds:
        if fold_artifacts_valid(
            task_raw_path(config, seed, int(split["fold"])),
            task_pred_path(config, seed, int(split["fold"])),
            task_prompt_analysis_path(config, seed, int(split["fold"])),
            task_checkpoint_path(config, seed, int(split["fold"])),
            split["test"],
        ):
            valid += 1
    status = "COMPLETE" if valid == 10 else "INCOMPLETE"
    record = {
        "config": config,
        "seed": seed,
        "status": status,
        "completed_folds": valid,
        "expected_folds": 10,
        "elapsed_seconds": time.time() - task_started,
        "prompt_sha256": prompt_digest(config),
    }
    atomic_json(progress_path, record)
    if status != "COMPLETE":
        raise RuntimeError(f"task ended with {valid}/10 valid folds")
    print(f"TASK {config} seed={seed} COMPLETE folds=10 elapsed={record['elapsed_seconds']:.1f}s", flush=True)


def prompt_diagnostics_frame(diagnostics, test_idx, config, seed, fold_id):
    if diagnostics is None:
        raise ValueError("prompt diagnostics missing after final evaluation")
    raw_similarity = diagnostics["raw_similarity"].detach().cpu().numpy()[test_idx]
    weights = diagnostics["similarity_weights"].detach().cpu().numpy()[test_idx]
    change = diagnostics["relative_token_change"].detach().cpu().numpy()[test_idx]
    prototype_cosine = diagnostics["prototype_cosine"].detach().cpu().numpy()
    rows = []
    for modality_idx, modality in enumerate(MODALITIES):
        row = {
            "config": config,
            "seed": seed,
            "fold": fold_id,
            "modality": modality,
            "prototype_theme_cosine": float(prototype_cosine[modality_idx]),
            "prototype_collapse_abs_cosine_gt_0_98": bool(abs(prototype_cosine[modality_idx]) > 0.98),
            "mean_relative_token_change": float(change[:, modality_idx].mean()),
            "std_relative_token_change": float(change[:, modality_idx].std()),
            "test_subjects": int(len(test_idx)),
        }
        for theme in range(2):
            values = raw_similarity[:, modality_idx, :, theme].reshape(-1)
            theme_weights = weights[:, modality_idx, :, theme].reshape(-1)
            row.update({
                f"theme{theme + 1}_local_similarity_mean": float(values.mean()),
                f"theme{theme + 1}_local_similarity_std": float(values.std()),
                f"theme{theme + 1}_local_similarity_min": float(values.min()),
                f"theme{theme + 1}_local_similarity_max": float(values.max()),
                f"theme{theme + 1}_mean_softmax_weight": float(theme_weights.mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=("P0", "P1", "P2", "P3"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=450)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not (EXP_DIR / "splits" / f"seed{args.seed}.json").is_file():
        raise FileNotFoundError("split manifest missing; run src/prepare.py first")
    split_task(args.config, args.seed, args.device, args.max_epochs, args.patience, args.force)


if __name__ == "__main__":
    main()
