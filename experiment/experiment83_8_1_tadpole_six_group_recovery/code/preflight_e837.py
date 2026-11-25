from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import run_e837 as runner
from fidelity_model import FidelityModel
from scmr_runtime import CLASS_PAIRS, source_config


def check_data():
    x, y, ids, splits = runner.load_data()
    frame = pd.read_csv(runner.ROOT / "inputs/processed_standard_data.csv")
    feature_map = np.load(runner.ROOT / "inputs/modal_feat_dict.npy", allow_pickle=True).item()
    ordered = [column for values in feature_map.values() for column in values]
    if frame.columns.tolist() != ordered + ["LABEL"]:
        raise RuntimeError("TADPOLE modality column order differs from modal_feat_dict")
    expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
                    .split(np.zeros(len(y)), y))
    occurrences = np.zeros(len(y), dtype=np.int64)
    for item, (train, test) in zip(splits, expected, strict=True):
        if not np.array_equal(item["train_idx"], train) or not np.array_equal(item["test_idx"], test):
            raise RuntimeError(f"split mismatch in fold {item.get('fold')}")
        if item.get("val_equals_test") is not True:
            raise RuntimeError("val_idx != test_idx")
        occurrences[test] += 1
    if not np.all(occurrences == 1) or ids.RID.nunique() != 598:
        raise RuntimeError("OOF coverage or patient identity check failed")
    return {"status": "PASS", "rows": int(len(x)), "shape": list(x.shape),
            "class_counts": np.bincount(y, minlength=3).tolist(),
            "data_sha256": runner.sha_file(runner.ROOT / "inputs/processed_standard_data.csv"),
            "split_sha256": runner.sha_file(runner.ROOT / "inputs/split_indices.json"),
            "folds": 10, "val_equals_test": True, "oof_coverage_once": True,
            "modality_dimensions": [len(v) for v in feature_map.values()]}


def check_structure(device):
    config_g3 = runner.group_config("T3")
    config_g4 = runner.group_config("T4")
    model_g3 = runner.build_model(config_g3, 1, device)
    model_g4 = runner.build_model(config_g4, 1, device)
    if tuple(model_g3.dims) != (145, 148, 3, 36, 24, 9):
        raise RuntimeError("six-modality dimensions differ")
    if model_g3.cam.cam_dim != 240 or model_g3.cam.Outputlayer.classifier.out_features != 3:
        raise RuntimeError("CAM output or three-class head differs")
    if not model_g3.use_darg or not model_g4.use_darg or model_g3.dpsl is not None or model_g4.dpsl is not None:
        raise RuntimeError("G3/G4 pre-DPSL structure differs")
    if set(CLASS_PAIRS) != {(0, 1), (0, 2), (1, 2)}:
        raise RuntimeError("DPSL class-pair routing differs")
    cfg = source_config()
    if float(cfg["residual"]["fraction"]) != 0.08 or float(cfg["full_classification"]["shared_cap"]) != 0.001:
        raise RuntimeError("historical residual/shared cap differs")
    if not hasattr(model_g3.darg.core, "edge_gate_logits") or not hasattr(model_g3.darg.core, "adaptive_gate_logits"):
        raise RuntimeError("DARG signed relation gates are missing")
    configs = {group: runner.group_config(group) for group in ("T1", "T2", "T3", "T4", "T5", "T6")}
    if configs["T1"]["dpsl_shared_cap"] != 0.003 or configs["T2"]["dpsl_shared_cap"] != 0.01:
        raise RuntimeError("T1/T2 shared caps are not registered")
    if any(configs[group]["graph_supervision_schedule"] != "warmup_only" for group in ("T3", "T4")):
        raise RuntimeError("T3/T4 graph schedule differs")
    if any(configs[group]["intermediate_base_weight_decay"] != 0.055 for group in ("T5", "T6")):
        raise RuntimeError("T5/T6 WD differs")
    return {"status": "PASS", "cam_dim": model_g3.cam.cam_dim, "nclass": 3,
            "class_pairs": [list(pair) for pair in CLASS_PAIRS],
            "residual_fraction": float(cfg["residual"]["fraction"]),
            "historical_shared_cap": float(cfg["full_classification"]["shared_cap"])}


def check_graph_supervision(device):
    x_np, y_np, _, splits = runner.load_data()
    model = runner.build_model(runner.group_config("T4"), int(splits[0]["fold"]), device)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.as_tensor(splits[0]["train_idx"], device=device).long()
    model.eval()
    _, aux = model(x, labels, rows, dpsl_enabled=False, darg_warmup=True)
    loss, target = runner.pair_mean_graph_supervision(aux["continuous_scores"], labels, rows)
    changed = labels.clone()
    changed[torch.as_tensor(splits[0]["test_idx"], device=device)] = 2 - changed[torch.as_tensor(splits[0]["test_idx"], device=device)]
    changed_loss, changed_target = runner.pair_mean_graph_supervision(aux["continuous_scores"], changed, rows)
    if len(aux["continuous_scores"]) != 6 or not loss.requires_grad or not torch.equal(target, changed_target):
        raise RuntimeError("graph supervision is not six-modality train-only continuous pair_mean")
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [graph.p.weight.grad for graph in model.gfr.graphs]
    if any(value is None or not torch.isfinite(value).all() or float(value.norm()) <= 0 for value in gradients):
        raise RuntimeError("GraphLearn supervision gradient is zero or non-finite")
    return {"status": "PASS", "loss": float(loss.detach()),
            "test_label_permutation_invariant": bool(torch.equal(loss.detach(), changed_loss.detach())),
            "gradient_norms": [float(value.norm()) for value in gradients],
            "modalities": len(aux["continuous_scores"])}


def _finite_budget(value):
    if isinstance(value, dict):
        return all(_finite_budget(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_budget(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def check_dpsl_budget(device):
    x_np, y_np, _, splits = runner.load_data()
    split = splits[0]
    fold = int(split["fold"])
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.as_tensor(train_idx, device=device).long()
    records = []
    for group in ("T1", "T2"):
        config = runner.group_config(group)
        runner.historical.seed_everything(runner.SEED)
        model = runner.build_model(config, fold, device)
        optimizers, _ = runner.make_optimizers(model, config)
        supports = runner.historical.anchor_pair_support(model, x)
        pair_np, same_np = runner.historical.make_pairs(train_idx, y_np, fold, supports)
        pair = torch.from_numpy(pair_np).long().to(device)
        same = torch.from_numpy(same_np).float().to(device)
        runner.run_warmup(model, optimizers, x, labels, rows, pair, same, False)
        model.eval()
        with torch.no_grad():
            _, warm_aux = model(x, labels, rows, pair, same, dpsl_enabled=False)
        stats = runner.fit_spectral_statistics(warm_aux["cam"], labels, rows, fold=fold,
                                                warm_logits=warm_aux["native_logits"])
        stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
        stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
        model.attach_dpsl(stats, str(runner.OLD_PROMPT_SCMR), 824602 * 100 + fold)
        optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
        runner.zero_optimizers(optimizers)
        model.train()
        _, aux = model(x, labels, rows, pair, same, dpsl_enabled=True)
        _, historical_total, _, _ = runner.historical_loss(aux, labels, rows, model, False)
        if not torch.isfinite(historical_total):
            raise RuntimeError(f"Non-finite DPSL preflight loss for {group}")
        budget = runner.dpsl_optimizer_step(model, optimizers, historical_total,
                                            aux["scmr_objectives"], config["dpsl_shared_cap"])
        family = budget["families"].get("cam", {})
        historical_norm = float(family.get("historical", 0.0))
        applied_norm = float(family.get("total_applied", 0.0))
        if float(budget["shared_cap"]) != float(config["dpsl_shared_cap"]):
            raise RuntimeError(f"DPSL cap was not applied for {group}")
        if not _finite_budget(budget) or not all(torch.isfinite(parameter.grad).all()
                                                  for parameter in model.dpsl.parameters()
                                                  if parameter.grad is not None):
            raise RuntimeError(f"Non-finite DPSL budget or gradient for {group}")
        dpsl_gradient_norm = sum(float(parameter.grad.norm()) for parameter in model.dpsl.parameters()
                                 if parameter.grad is not None)
        if dpsl_gradient_norm <= 0:
            raise RuntimeError(f"DPSL parameters have no effective gradient for {group}")
        if applied_norm > float(config["dpsl_shared_cap"]) * historical_norm + 1e-8:
            raise RuntimeError(f"Shared gradient cap exceeded for {group}")
        model.eval()
        with torch.no_grad():
            native, _ = model(x, labels, rows, pair, same, dpsl_enabled=False)
            full, _ = model(x, labels, rows, pair, same, dpsl_enabled=True)
            residual_off, _ = model(x, labels, rows, pair, same, dpsl_enabled=True,
                                    dpsl_intervention="residual_off")
        residual_max = float(aux["dpsl"]["residual_ratio"].max())
        if residual_max > 0.08 + 1e-7:
            raise RuntimeError(f"DPSL residual cap exceeded for {group}")
        if not torch.equal(native, residual_off):
            raise RuntimeError(f"Residual-Off does not restore native output for {group}")
        records.append({"group": group, "fold": fold, "configured_cap": float(config["dpsl_shared_cap"]),
                        "observed_cap": float(budget["shared_cap"]),
                        "cam_historical_norm": historical_norm,
                        "cam_shared_applied_norm": applied_norm,
                        "dpsl_gradient_norm": dpsl_gradient_norm,
                        "residual_max": residual_max,
                        "full_native_logit_max_abs": float((full - native).abs().max()),
                        "residual_off_native_exact": True,
                        "finite": True})
        del model, optimizers
        torch.cuda.empty_cache()
    return {"status": "PASS", "records": records}


def main():
    runner.configure_worker()
    device = torch.device("cuda:0")
    checks = {"data": check_data(), "structure": check_structure(device),
              "graph_supervision": check_graph_supervision(device),
              "dpsl_budget": check_dpsl_budget(device)}
    seal_path = runner.ROOT / "audits/training_seal.json"
    value = {"experiment": "Experiment 83.7", "status": "PASS", "formal_training_allowed": True,
             "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA, "checks": checks}
    if not seal_path.is_file():
        raise RuntimeError("Training seal must exist before preflight is recorded")
    value["training_seal_sha256"] = runner.sha_file(seal_path)
    path = runner.ROOT / "audits/preflight_start.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
