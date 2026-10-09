from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import EXP_DIR, get_prompt_features, load_data, split_path
from model import EvalHelper, make_initial_trainable_state, parameter_state_sha256


HYPER = SimpleNamespace(
    GC_mode="weighted-cosine",
    MF_mode="concat",
    MP_mode="GCN",
    mode="simple-2",
    dropout=0.35,
    lr=0.0038,
    n_head=2,
    n_hidden=18,
    nlayer=1,
    reg=0.11,
    nepoch=450,
    early=200,
    th=0.9,
    theta_smooth=1.0,
    theta_degree=0.5,
    theta_sparsity=0.0,
    nclass=2,
    nmodal=4,
)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(4)
    x, y, dims, _ = load_data()
    split = json.loads(split_path(0).read_text(encoding="utf-8"))["folds"][0]
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["validation"], dtype=np.int64)
    reports = []
    canonical_hash = None
    for config in ("P0", "P1"):
        features = get_prompt_features(config, device)
        torch.manual_seed(int(split["model_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(split["model_seed"]))
        agent = EvalHelper(dims, x, y, HYPER, train_idx, features, config, device)
        canonical_state = make_initial_trainable_state(dims, HYPER, features, config, int(split["model_seed"]))
        state_hash = parameter_state_sha256(canonical_state)
        if canonical_hash is None:
            canonical_hash = state_hash
        elif canonical_hash != state_hash:
            raise AssertionError("P0/P1 initialization mismatch in smoke test")
        agent.copy_trainable_state(canonical_state)
        torch.manual_seed(int(split["model_seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(split["model_seed"]))
        train_info = agent.train_step()
        probabilities, _, _, diagnostics = agent.predict_all(capture_prompt_diagnostics=True)
        if probabilities.shape != (871, 2):
            raise AssertionError(f"unexpected probability shape {probabilities.shape}")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError("invalid class probabilities")
        if diagnostics is None or diagnostics["raw_similarity"].shape != (871, 4, 4, 2):
            raise AssertionError("semantic similarity shape mismatch")
        if not torch.isfinite(agent.ModalFusion.MLP.t1.weight.grad).all():
            raise AssertionError("non-finite prompt MLP gradients")
        prompt_mlp_grad_norm = float(agent.ModalFusion.MLP.t1.weight.grad.norm().detach().cpu())
        if not prompt_mlp_grad_norm > 0:
            raise AssertionError("prompt MLP did not receive gradient")
        val_acc = float(np.mean(probabilities[val_idx].argmax(axis=1) == y[val_idx]))
        reports.append({
            "config": config,
            "train_loss": train_info["loss"],
            "classification_loss": train_info["classification_loss"],
            "graph_loss": train_info["graph_loss"],
            "label_graph_diagnostic_only": train_info["label_graph_loss_diagnostic_only"],
            "validation_accuracy_after_one_step": val_acc,
            "prompt_mlp_gradient_norm": prompt_mlp_grad_norm,
            "probability_shape": list(probabilities.shape),
            "similarity_shape": list(diagnostics["raw_similarity"].shape),
            "initial_parameter_sha256": state_hash,
        })
        del agent, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "status": "PASS",
        "purpose": "single-step pipeline validation only; smoke metrics are not experiment results",
        "device": str(device),
        "reports": reports,
    }
    path = EXP_DIR / "results" / "smoke_test.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
