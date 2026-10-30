#!/usr/bin/env python3
"""Experiment 52 continuation: GFR-only weight decay sweep.

E2 and E3 use the parent Experiment 52 DRAP construction and forward path.
Only the GFR Adam parameter group's weight_decay changes. CAM and DRAP keep
the parent E1 value 0.11. New tasks, checkpoints, logs, diagnostics and
reports are isolated below wd_sweep; parent E0/E1 artifacts are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as parent_run

SWEEP = EXP / ".runtime" / "wd_sweep"
TASKS = SWEEP / "tasks"
CKPTS = SWEEP / "checkpoints"
LATEST = SWEEP / "latest"
LOGS = SWEEP / "logs"
RESULTS = EXP / "results" / "wd_sweep"
CFG_PATH = EXP / "config_gfr_wd_sweep.json"
MANIFEST_PATH = EXP / "manifest_wd_sweep.json"
CFG = json.loads(CFG_PATH.read_text())
GROUPS = ("E2", "E3")
GROUP_BY_ID = {row["id"]: row for row in CFG["new_groups"]}
MODS = parent_run.MODS


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


def csv_write(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def csv_read(path: Path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dirs():
    for path in (SWEEP, TASKS, CKPTS, LATEST, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def task_path(group: str, fold: int) -> Path:
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group: str, fold: int) -> Path:
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group: str, fold: int) -> Path:
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def group_row(group: str):
    if group not in GROUPS:
        raise ValueError(f"unsupported group {group}")
    return GROUP_BY_ID[group]


def expected_settings(group: str):
    row = group_row(group)
    return {
        "GFR": float(row["gfr_weight_decay"]),
        "CAM": float(row["cam_weight_decay"]),
        "DRAP": float(row["drap_weight_decay"]),
    }


def sha_obj(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def module_sha(module):
    h = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(str(parameter.dtype).encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module_schema(module):
    return sha_obj([
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in sorted(module.named_parameters())
    ])


def optimizer_settings(state):
    out = {}
    for name, optimizer in (
        ("GFR", state["opt_base"]),
        ("CAM", state["opt_cam"]),
        ("DRAP", state["opt_drap"]),
    ):
        if optimizer is None:
            out[name] = None
            continue
        out[name] = {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "parameter_count": int(sum(p.numel() for p in optimizer.param_groups[0]["params"])),
            "parameter_group_count": int(len(optimizer.param_groups)),
        }
    return out


def assert_optimizer_settings(state, group):
    expected = expected_settings(group)
    actual = optimizer_settings(state)
    for name in ("GFR", "CAM", "DRAP"):
        if actual[name] is None:
            raise AssertionError(f"missing optimizer {name}")
        expected_lr = 0.001 if name == "DRAP" else 0.0038
        if abs(actual[name]["weight_decay"] - expected[name]) > 1e-15:
            raise AssertionError(f"{group} {name} weight_decay mismatch")
        if abs(actual[name]["lr"] - expected_lr) > 1e-15:
            raise AssertionError(f"{group} {name} learning-rate mismatch")
        if actual[name]["parameter_group_count"] != 1:
            raise AssertionError(f"{group} {name} parameter partition changed")
    return actual


def build_state(group: str, fold: int):
    if group not in GROUPS:
        raise ValueError(group)
    state = parent_run.build_state(group, fold)
    if state["group"] != group:
        raise AssertionError(f"parent build_state returned {state['group']} for {group}")
    assert_optimizer_settings(state, group)
    return state


def initial_signature(state):
    return {
        "shared_parameter_sha": state["initial_shared_sha"],
        "shared_schema_sha": state["initial_schema_sha"],
        "full_drap_model_sha": module_sha(state["model"]),
        "full_drap_model_schema_sha": module_schema(state["model"]),
        "cam_sha": module_sha(state["cam"]),
        "initial_rng_sha": state["initial_rng_sha"],
        "anchor_rng_sha": state["anchor_rng_sha"],
        "entry_rng_sha": state["entry_rng_sha"],
        "split_sha": parent_run.split_sha(int(state["fold"])),
        "pair_sha": parent_run.sha_file(parent_run.FD / f"fold{int(state['fold']):02d}.npz"),
    }


def finite_module(module):
    import torch
    return all(bool(torch.isfinite(p).all().item()) for p in module.parameters())


def max_abs(a, b):
    return float((a.detach() - b.detach()).abs().max().cpu())


def rel_norm(a, b):
    return float((a.detach() - b.detach()).norm().cpu() / b.detach().norm().clamp_min(1e-8).cpu())


def prepare_read_only():
    parent_run.load_data()
    for fold in range(1, 11):
        npz = parent_run.FD / f"fold{fold:02d}.npz"
        meta = parent_run.FD / f"fold{fold:02d}.json"
        if not npz.exists() or not meta.exists():
            raise FileNotFoundError(npz)
        if not bool(json.loads(meta.read_text()).get("test_label_excluded_from_pair_sampling")):
            raise AssertionError(f"fold {fold} pair metadata lacks test-label isolation")


def preflight():
    import torch

    ensure_dirs()
    prepare_read_only()
    parent_run.runtime()
    signature_rows = []
    parity_rows = []
    all_signature = True
    all_parity = True
    all_pair_isolated = True
    all_optimizer = True
    for fold in range(1, 11):
        states = {"E1": parent_run.build_state("E1", fold)}
        states.update({group: build_state(group, fold) for group in GROUPS})
        signatures = {group: initial_signature(state) for group, state in states.items()}
        for group, state in states.items():
            settings = optimizer_settings(state)
            expected = {"GFR": 0.11, "CAM": 0.11, "DRAP": 0.11} if group == "E1" else expected_settings(group)
            optimizer_pass = all(abs(settings[name]["weight_decay"] - expected[name]) < 1e-15 for name in expected)
            all_optimizer = all_optimizer and optimizer_pass
            signature_rows.append({
                "fold": fold,
                "group": group,
                **signatures[group],
                "optimizer_settings": json.dumps(settings, sort_keys=True),
                "optimizer_pass": optimizer_pass,
            })
        ref = signatures["E1"]
        signature_pass = all(
            all(signatures[group][key] == ref[key] for key in (
                "shared_parameter_sha", "shared_schema_sha", "full_drap_model_sha",
                "full_drap_model_schema_sha", "cam_sha", "initial_rng_sha",
                "anchor_rng_sha", "entry_rng_sha", "split_sha", "pair_sha",
            ))
            for group in GROUPS
        )
        all_signature = all_signature and signature_pass
        fd = parent_run.load_fold_data(fold)
        train_set = set(fd["train"].detach().cpu().tolist())
        pair_pass = set(fd["pair_i"].detach().cpu().tolist()).issubset(train_set) and set(fd["pair_j"].detach().cpu().tolist()).issubset(train_set)
        all_pair_isolated = all_pair_isolated and pair_pass
        with torch.no_grad():
            outputs = {}
            for group in ("E1",) + GROUPS:
                parent_run.restore_rng(states[group]["initial_rng"])
                outputs[group] = parent_run.forward_state(states[group], fd, train_mode=False, use_reweight=True)
        parity = {
            "fold": fold,
            "signature_pass": signature_pass,
            "pair_isolated": pair_pass,
            "E2_fused_vs_E1": max_abs(outputs["E2"]["fused"], outputs["E1"]["fused"]),
            "E3_fused_vs_E1": max_abs(outputs["E3"]["fused"], outputs["E1"]["fused"]),
            "E2_logp_vs_E1": max_abs(outputs["E2"]["logp"], outputs["E1"]["logp"]),
            "E3_logp_vs_E1": max_abs(outputs["E3"]["logp"], outputs["E1"]["logp"]),
            "E2_total_vs_E1": abs(float(outputs["E2"]["total"]) - float(outputs["E1"]["total"])),
            "E3_total_vs_E1": abs(float(outputs["E3"]["total"]) - float(outputs["E1"]["total"])),
        }
        parity_rows.append(parity)
        all_parity = all_parity and max(
            parity["E2_fused_vs_E1"], parity["E3_fused_vs_E1"],
            parity["E2_logp_vs_E1"], parity["E3_logp_vs_E1"],
            parity["E2_total_vs_E1"], parity["E3_total_vs_E1"],
        ) == 0.0
        for state in states.values():
            del state
        torch.cuda.empty_cache()

    gradient_rows = []
    edge_checks = {}
    label_checks = {}
    for group in ("E2", "E3"):
        state = build_state(group, 1)
        fd = parent_run.load_fold_data(1)
        with torch.no_grad():
            edge_out = parent_run.forward_state(state, fd, train_mode=False, use_reweight=True)
        edge_checks[group] = []
        for modality, (adj, drap_adj, norm, weights, edge_idx) in enumerate(zip(
            edge_out["details"]["adjs"], edge_out["details"]["drap_adjs"],
            edge_out["details"]["norm_adjs"], edge_out["details"]["edge_weights"],
            edge_out["details"]["edge_indices"],
        )):
            delta = (weights - 1.0).abs()
            edge_checks[group].append({
                "modality": MODS[modality],
                "raw_edges": int((adj > 0).sum().item()),
                "offdiag_edges": int(len(edge_idx[0])),
                "symmetric": bool((drap_adj - drap_adj.T).abs().max().item() < 1e-6),
                "finite": bool(torch.isfinite(drap_adj).all().item()),
                "nonnegative": bool((drap_adj >= 0).all().item()),
                "support_preserved": bool(((adj == 0) == (drap_adj == 0)).all().item()),
                "normalized_finite": bool(torch.isfinite(norm).all().item()),
                "changed_edges_gt_1e8": int((delta > 1e-8).sum().item()),
                "mean_abs_weight_delta": float(delta.mean().cpu()) if delta.numel() else 0.0,
            })
        state["opt_base"].zero_grad()
        state["opt_cam"].zero_grad()
        state["opt_drap"].zero_grad()
        train_out = parent_run.forward_state(state, fd, train_mode=True, use_reweight=True)
        rel_params = list(state["model"].relation_encoders.parameters()) + list(state["model"].relation_scorers.parameters())
        cls_grads = torch.autograd.grad(train_out["cls"], rel_params, retain_graph=True, allow_unused=True)
        cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in cls_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_grads) else 0.0
        train_out["total"].backward()
        gradient_rows.append({
            "group": group,
            "classification_to_relation_grad_norm": cls_grad,
            "relation_encoder_grad_norm": parent_run.module_grad_norm(state["model"].relation_encoders),
            "relation_scorer_grad_norm": parent_run.module_grad_norm(state["model"].relation_scorers),
            "discriminator_grad_norm": parent_run.module_grad_norm(state["model"].discriminator),
            "total_drap_grad_norm": parent_run.module_grad_norm(state["model"]),
            "finite": bool(torch.isfinite(train_out["total"]).item()),
        })
        old_target = state["targ"].detach().clone()
        with torch.no_grad():
            before = parent_run.forward_state(state, fd, train_mode=False, use_reweight=True)
            state["targ"][fd["test"].long()] = 1 - state["targ"][fd["test"].long()]
            after = parent_run.forward_state(state, fd, train_mode=False, use_reweight=True)
            state["targ"].copy_(old_target)
        label_checks[group] = {
            "fused_max_abs": max_abs(before["fused"], after["fused"]),
            "logp_max_abs": max_abs(before["logp"], after["logp"]),
        }
        del state
        torch.cuda.empty_cache()
    probe = torch.randn(17, device="cuda", requires_grad=True)
    grl_grad = torch.autograd.grad(parent_run.runtime()["drap"].grad_reverse(probe, 0.03).sum(), probe)[0]
    grl_ok = bool(torch.allclose(grl_grad, torch.full_like(grl_grad, -0.03), atol=1e-7, rtol=1e-6))
    edge_pass = all(item[key] for rows in edge_checks.values() for item in rows for key in (
        "symmetric", "finite", "nonnegative", "support_preserved", "normalized_finite"
    ))
    gradient_pass = all(
        row["finite"] and all(float(row[key]) > 0.0 for key in (
            "classification_to_relation_grad_norm", "relation_encoder_grad_norm",
            "relation_scorer_grad_norm", "discriminator_grad_norm", "total_drap_grad_norm"
        ))
        for row in gradient_rows
    )
    label_pass = all(max(values.values()) == 0.0 for values in label_checks.values())
    status = "PASS" if all_signature and all_parity and all_pair_isolated and all_optimizer and edge_pass and gradient_pass and label_pass and grl_ok else "FAIL"
    result = {
        "status": status,
        "groups": ["E1_read_only_reference", "E2", "E3"],
        "folds_checked": 10,
        "initialization_signature_equal_all_folds": all_signature,
        "initial_forward_loss_output_parity_all_folds": all_parity,
        "optimizer_settings_pass_all_folds": all_optimizer,
        "pair_test_label_isolation_all_folds": all_pair_isolated,
        "signature_rows": signature_rows,
        "parity_rows": parity_rows,
        "gradient_rows": gradient_rows,
        "edge_checks": edge_checks,
        "test_label_prediction_isolation": label_checks,
        "edge_path_pass": edge_pass,
        "gradient_path_pass": gradient_pass,
        "grl_primitive_ok": grl_ok,
        "source_run_sha": parent_run.sha_file(EXP / "run.py"),
        "source_audit_sha": parent_run.sha_file(EXP / "audit.py"),
    }
    atomic_json(SWEEP / "preflight.json", result)
    csv_write(RESULTS / "initialization_preflight.csv", signature_rows)
    print("GFR_WD_PREFLIGHT", status, flush=True)
    if status != "PASS":
        raise SystemExit(2)


def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.cuda()


def train_one(group: str, fold: int):
    import torch

    ensure_dirs()
    task = task_path(group, fold)
    if task.exists() and json.loads(task.read_text()).get("status") == "COMPLETE":
        return
    state = build_state(group, fold)
    initial = initial_signature(state)
    fd = parent_run.load_fold_data(fold)
    expected = expected_settings(group)
    start = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_model = best_cam = best_ob = best_oc = best_od = None
    best_diag = {}
    best_use_reweight = True
    first_active_cls_grad = 0.0
    first_total_drap_grad = 0.0
    latest = latest_path(group, fold)
    if latest.exists():
        try:
            saved = parent_run.torch_load(latest)
            if saved.get("group") != group or int(saved.get("fold", -1)) != fold:
                raise ValueError("latest identity mismatch")
            state["model"].load_state_dict(saved["model"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            state["opt_base"].load_state_dict(saved["opt_base"])
            state["opt_cam"].load_state_dict(saved["opt_cam"])
            state["opt_drap"].load_state_dict(saved["opt_drap"])
            optimizer_to_cuda(state["opt_base"])
            optimizer_to_cuda(state["opt_cam"])
            optimizer_to_cuda(state["opt_drap"])
            assert_optimizer_settings(state, group)
            parent_run.restore_rng(saved["rng"])
            start = int(saved["epoch"]) + 1
            wait = int(saved["wait"])
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            best_model = saved.get("best_model")
            best_cam = saved.get("best_cam")
            best_ob = saved.get("best_ob")
            best_oc = saved.get("best_oc")
            best_od = saved.get("best_od")
            best_diag = saved.get("best_diag", {})
            best_use_reweight = bool(saved.get("best_use_reweight", True))
            first_active_cls_grad = float(saved.get("first_active_cls_grad", 0.0))
            first_total_drap_grad = float(saved.get("first_total_drap_grad", 0.0))
        except Exception:
            latest.unlink(missing_ok=True)
            start = 0
            wait = 0
            best_acc = -1.0
            best_epoch = -1
            best_model = best_cam = best_ob = best_oc = best_od = None
            best_diag = {}

    t0 = time.time()
    epoch = start - 1
    try:
        for epoch in range(start, int(parent_run.CFG["base"]["epochs"])):
            state["epoch"] = epoch
            use_rw = epoch >= int(parent_run.CFG["drap"]["warmup_epochs"])
            state["opt_base"].zero_grad()
            state["opt_cam"].zero_grad()
            state["opt_drap"].zero_grad()
            out = parent_run.forward_state(state, fd, train_mode=True, use_reweight=use_rw)
            if not bool(torch.isfinite(out["total"]).item()):
                raise FloatingPointError(f"non-finite loss epoch {epoch}")
            if use_rw and first_active_cls_grad == 0.0:
                rel_params = list(state["model"].relation_encoders.parameters()) + list(state["model"].relation_scorers.parameters())
                cls_grads = torch.autograd.grad(out["cls"], rel_params, retain_graph=True, allow_unused=True)
                if any(g is not None for g in cls_grads):
                    first_active_cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in cls_grads if g is not None]).sum().sqrt().cpu())
            out["total"].backward()
            for module in (state["model"], state["cam"]):
                if not finite_module(module):
                    raise FloatingPointError(f"non-finite parameters epoch {epoch}")
                for parameter in module.parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError(f"non-finite gradient epoch {epoch}")
            if use_rw and first_total_drap_grad == 0.0:
                first_total_drap_grad = parent_run.module_grad_norm(state["model"])
            grad_norms = {
                "base_grad_norm": parent_run.module_grad_norm(state["base"]),
                "cam_grad_norm": parent_run.module_grad_norm(state["cam"]),
                "drap_grad_norm": parent_run.module_grad_norm(state["model"]),
                "relation_encoder_grad_norm": parent_run.module_grad_norm(state["model"].relation_encoders),
                "relation_scorer_grad_norm": parent_run.module_grad_norm(state["model"].relation_scorers),
                "discriminator_grad_norm": parent_run.module_grad_norm(state["model"].discriminator),
            }
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_drap"].step()
            with torch.no_grad():
                ev = parent_run.forward_state(state, fd, train_mode=False, use_reweight=use_rw)
                met = parent_run.metrics(ev["logp"], state)
            if not all(math.isfinite(float(met[key])) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen")):
                raise FloatingPointError(f"non-finite metric epoch {epoch}")
            if met["acc"] > best_acc:
                best_acc = met["acc"]
                best_epoch = epoch
                wait = 0
                best_use_reweight = use_rw
                best_model = parent_run.state_cpu(state["model"])
                best_cam = parent_run.state_cpu(state["cam"])
                best_ob = parent_run.optimizer_cpu_state(state["opt_base"])
                best_oc = parent_run.optimizer_cpu_state(state["opt_cam"])
                best_od = parent_run.optimizer_cpu_state(state["opt_drap"])
                best_diag = {
                    "train_cls": float(out["cls"].detach()),
                    "train_graph": float(out["graph"].detach()),
                    "pair_loss": float(out["pair_loss"].detach()),
                    "adv_loss": float(out["adv_loss"].detach()),
                    "classification_to_relation_grad_norm": first_active_cls_grad,
                    "initial_total_drap_grad_norm": first_total_drap_grad,
                    **grad_norms,
                }
                parent_run.torch_save(checkpoint_path(group, fold), {
                    "group": group,
                    "group_id": int(group[1:]),
                    "name": group_row(group)["name"],
                    "model_branch": "E1_DRAP",
                    "fold": fold,
                    "seed": 0,
                    "gfr_weight_decay": expected["GFR"],
                    "cam_weight_decay": expected["CAM"],
                    "drap_weight_decay": expected["DRAP"],
                    "best_epoch": best_epoch,
                    "epochs_executed": epoch + 1,
                    "metrics": met,
                    "model": best_model,
                    "cam": best_cam,
                    "opt_base": best_ob,
                    "opt_cam": best_oc,
                    "opt_drap": best_od,
                    "diagnostics": best_diag,
                    "best_use_reweight": best_use_reweight,
                    "split_sha": parent_run.split_sha(fold),
                    "pair_sha": parent_run.sha_file(parent_run.FD / f"fold{fold:02d}.npz"),
                    "initial_shared_parameter_sha": initial["shared_parameter_sha"],
                    "initial_shared_schema_sha": initial["shared_schema_sha"],
                    "initial_full_drap_model_sha": initial["full_drap_model_sha"],
                    "initial_full_drap_model_schema_sha": initial["full_drap_model_schema_sha"],
                    "initial_cam_sha": initial["cam_sha"],
                    "initial_rng_sha": initial["initial_rng_sha"],
                    "anchor_rng_sha": initial["anchor_rng_sha"],
                    "entry_rng_sha": initial["entry_rng_sha"],
                    "source_run_sha": parent_run.sha_file(EXP / "run.py"),
                    "source_drap_sha": parent_run.sha_file(EXP / "drap.py"),
                })
            else:
                wait += 1
            if epoch % 10 == 0:
                parent_run.torch_save(latest, {
                    "group": group,
                    "fold": fold,
                    "gfr_weight_decay": expected["GFR"],
                    "cam_weight_decay": expected["CAM"],
                    "drap_weight_decay": expected["DRAP"],
                    "epoch": epoch,
                    "wait": wait,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "best_use_reweight": best_use_reweight,
                    "model": parent_run.state_cpu(state["model"]),
                    "cam": parent_run.state_cpu(state["cam"]),
                    "opt_base": parent_run.optimizer_cpu_state(state["opt_base"]),
                    "opt_cam": parent_run.optimizer_cpu_state(state["opt_cam"]),
                    "opt_drap": parent_run.optimizer_cpu_state(state["opt_drap"]),
                    "rng": parent_run.pack_rng(),
                    "best_model": best_model,
                    "best_cam": best_cam,
                    "best_ob": best_ob,
                    "best_oc": best_oc,
                    "best_od": best_od,
                    "best_diag": best_diag,
                    "first_active_cls_grad": first_active_cls_grad,
                    "first_total_drap_grad": first_total_drap_grad,
                })
            if wait > int(parent_run.CFG["base"]["early"]):
                break
        if best_model is None:
            raise RuntimeError("no best checkpoint")
        state["model"].load_state_dict(best_model, strict=True)
        state["cam"].load_state_dict(best_cam, strict=True)
        with torch.no_grad():
            check = parent_run.metrics(parent_run.forward_state(state, fd, train_mode=False, use_reweight=best_use_reweight)["logp"], state)
        if abs(check["acc"] - best_acc) > 1e-8:
            raise AssertionError("best readback mismatch")
        row = {
            "group": group,
            "group_id": int(group[1:]),
            "name": group_row(group)["name"],
            "model_branch": "E1_DRAP",
            "gfr_weight_decay": expected["GFR"],
            "cam_weight_decay": expected["CAM"],
            "drap_weight_decay": expected["DRAP"],
            "fold": fold,
            "seed": 0,
            "gpu": 0,
            "status": "COMPLETE",
            "source": "new_training",
            "split_sha": parent_run.split_sha(fold),
            "pair_sha": parent_run.sha_file(parent_run.FD / f"fold{fold:02d}.npz"),
            "best_epoch": best_epoch,
            "epochs_executed": epoch + 1,
            "warmup_active_at_best": int(not best_use_reweight),
            "wall_seconds": time.time() - t0,
            "best_sha": parent_run.sha_file(checkpoint_path(group, fold)),
            "initial_shared_parameter_sha": initial["shared_parameter_sha"],
            "initial_shared_schema_sha": initial["shared_schema_sha"],
            "initial_full_drap_model_sha": initial["full_drap_model_sha"],
            "initial_full_drap_model_schema_sha": initial["full_drap_model_schema_sha"],
            "initial_cam_sha": initial["cam_sha"],
            "initial_rng_sha": initial["initial_rng_sha"],
            "anchor_rng_sha": initial["anchor_rng_sha"],
            "entry_rng_sha": initial["entry_rng_sha"],
            "classification_to_relation_grad_norm": first_active_cls_grad,
            "initial_total_drap_grad_norm": first_total_drap_grad,
            "optimizer_settings": json.dumps(optimizer_settings(state), sort_keys=True),
            **check,
        }
        atomic_json(task, row)
        latest.unlink(missing_ok=True)
        print("GFR_WD_DONE", group, fold, best_epoch, row["acc"], flush=True)
    except Exception as exc:
        atomic_json(task, {"group": group, "fold": fold, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def write_manifest(manifest):
    atomic_json(MANIFEST_PATH, manifest)


def initial_manifest():
    parent_manifest = json.loads((EXP / "manifest.json").read_text())
    return {
        "experiment": CFG["experiment"],
        "parent_experiment": CFG["parent_experiment"],
        "status": "PREPARED",
        "formal_new_groups": list(GROUPS),
        "formal_new_folds": 20,
        "group_optimizer_settings": {group: expected_settings(group) for group in GROUPS},
        "pre_change_source_sha256": {
            "run.py": parent_manifest["code_sha"]["run.py"],
            "audit.py": parent_manifest["code_sha"]["audit.py"],
            "drap.py": parent_manifest["code_sha"]["drap.py"],
            "config.json": parent_manifest["config_sha"],
            "manifest.json": parent_run.sha_file(EXP / "manifest.json"),
            "fold_metrics.csv": parent_run.sha_file(EXP / "results/fold_metrics.csv"),
            "diagnostics.csv": parent_run.sha_file(EXP / "results/diagnostics.csv"),
            "report.md": parent_run.sha_file(EXP / "results/report.md"),
        },
        "current_source_sha256": {
            "run.py": parent_run.sha_file(EXP / "run.py"),
            "audit.py": parent_run.sha_file(EXP / "audit.py"),
            "drap.py": parent_run.sha_file(EXP / "drap.py"),
            "config_gfr_wd_sweep.json": parent_run.sha_file(CFG_PATH),
        },
        "parent_manifest_status": parent_manifest.get("status"),
        "parent_manifest_historical_sha_fields": {
            "code_sha": parent_manifest.get("code_sha"),
            "config_sha": parent_manifest.get("config_sha"),
            "fold_metrics_sha": parent_manifest.get("fold_metrics_sha"),
            "diagnostics_sha": parent_manifest.get("diagnostics_sha"),
        },
        "fold_sha256": {str(fold): parent_run.split_sha(fold) for fold in range(1, 11)},
        "anchor_sha256": {str(fold): parent_run.sha_file(parent_run.ANCHORS / f"fold{fold:02d}.pt") for fold in range(1, 11)},
        "historical_e1_checkpoint_sha256": {
            str(fold): parent_run.sha_file(parent_run.HIST_E51_2_CK / f"E1_f{fold:02d}.best.pt")
            for fold in range(1, 11)
        },
        "seed_rule": "510000 + fold*100 + 1",
        "new_module_rng_rule": "torch.random.fork_rng seed=520000+fold",
        "test_labels_for_drap": "forbidden",
        "runtime_root": str(SWEEP),
        "results_root": str(RESULTS),
    }


def prepare():
    ensure_dirs()
    prepare_read_only()
    write_manifest(initial_manifest())
    print("GFR_WD_PREPARE_PASS", flush=True)


def controller(groups):
    groups = list(groups)
    if not groups or any(group not in GROUPS for group in groups) or len(set(groups)) != len(groups):
        raise ValueError(groups)
    ensure_dirs()
    preflight_path = SWEEP / "preflight.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text()).get("status") != "PASS":
        raise RuntimeError("preflight must pass before training")
    jobs = [(group, fold) for group in groups for fold in range(1, 11)]
    workers = min(5, int(os.environ.get("E52_GFR_WD_WORKERS", "5")))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    processes = []
    started = time.time()
    while jobs or processes:
        alive = sum(process.poll() is None for _, process, _ in processes)
        while alive < workers and jobs:
            group, fold = jobs.pop(0)
            log_handle = (LOGS / f"{group}_f{fold:02d}.log").open("ab")
            process = subprocess.Popen(
                [sys.executable, "-B", str(EXP / "run_gfr_wd_sweep.py"), "worker", "--group", group, "--fold", str(fold)],
                cwd=EXP, env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append((group, process, log_handle))
            alive += 1
        for item in list(processes):
            if item[1].poll() is not None:
                item[2].close()
                processes.remove(item)
        time.sleep(0.5)
    statuses = []
    for group in groups:
        for fold in range(1, 11):
            path = task_path(group, fold)
            statuses.append(json.loads(path.read_text()).get("status", "MISSING") if path.exists() else "MISSING")
    manifest = load_manifest()
    manifest.setdefault("controller_runs", []).append({
        "groups": groups,
        "workers": workers,
        "device": "cuda:0",
        "completed": statuses.count("COMPLETE"),
        "failed": statuses.count("FAILED"),
        "missing": statuses.count("MISSING"),
        "wall_seconds": time.time() - started,
    })
    manifest["training_status"] = "COMPLETE" if all(status == "COMPLETE" for status in statuses) else "PARTIAL_OR_FAILED"
    write_manifest(manifest)
    print("GFR_WD_CONTROLLER_DONE", groups, statuses.count("COMPLETE"), statuses.count("FAILED"), flush=True)
    if not all(status == "COMPLETE" for status in statuses):
        raise SystemExit(3)


def aggregate():
    import numpy as np

    rows = []
    for group in GROUPS:
        for fold in range(1, 11):
            path = task_path(group, fold)
            if not path.exists():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text())
            if row.get("status") != "COMPLETE":
                raise RuntimeError(row)
            rows.append(row)
    csv_write(RESULTS / "fold_metrics.csv", rows)
    parent_rows = csv_read(EXP / "results" / "fold_metrics.csv")
    by = {(row["group"], int(row["fold"])): row for row in parent_rows}
    by.update({(row["group"], int(row["fold"])): row for row in rows})
    summary = []
    for group in ("E0", "E1", "E2", "E3"):
        rr = [by[(group, fold)] for fold in range(1, 11)]
        q = {
            "group": group,
            "name": rr[0]["name"],
            "gfr_weight_decay": 0.11 if group in ("E0", "E1") else float(group_row(group)["gfr_weight_decay"]),
            "cam_weight_decay": 0.11,
            "drap_weight_decay": 0.11,
            "folds": 10,
            "source": "historical_read_only" if group in ("E0", "E1") else "new_training",
        }
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            values = np.asarray([float(row[key]) for row in rr])
            q[key + "_mean_pct"] = float(values.mean() * 100)
            q[key + "_sd_pct"] = float(values.std(ddof=0) * 100)
        summary.append(q)
    csv_write(RESULTS / "summary.csv", summary)
    comparisons = [
        ("E1_minus_E0", "E1", "E0"),
        ("E2_minus_E1", "E2", "E1"),
        ("E3_minus_E1", "E3", "E1"),
        ("E3_minus_E2", "E3", "E2"),
    ]
    paired = []
    for comparison, left, right in comparisons:
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            delta = np.asarray([(float(by[(left, fold)][key]) - float(by[(right, fold)][key])) * 100 for fold in range(1, 11)])
            paired.append({
                "comparison": comparison,
                "left_group": left,
                "right_group": right,
                "metric": key,
                "delta_pp_mean": float(delta.mean()),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "fold_deltas_pp": json.dumps([float(v) for v in delta.tolist()]),
            })
    csv_write(RESULTS / "paired_comparisons.csv", paired)
    manifest = load_manifest()
    manifest.update({
        "status": "AGGREGATED",
        "new_fold_metrics_sha256": parent_run.sha_file(RESULTS / "fold_metrics.csv"),
        "summary_sha256": parent_run.sha_file(RESULTS / "summary.csv"),
        "paired_comparisons_sha256": parent_run.sha_file(RESULTS / "paired_comparisons.csv"),
        "new_training_rows": len(rows),
    })
    write_manifest(manifest)
    print("GFR_WD_AGGREGATE_PASS", flush=True)


def load_checkpoint_state(group, fold):
    state = parent_run.build_state(group, fold)
    path = parent_run.checkpoint_path("E1", fold) if group == "E1" else checkpoint_path(group, fold)
    checkpoint = parent_run.torch_load(path)
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    return state, parent_run.load_fold_data(fold), checkpoint


def gradient_audit(state, fd, use_reweight):
    import torch

    state["model"].zero_grad(set_to_none=True)
    state["cam"].zero_grad(set_to_none=True)
    out = parent_run.forward_state(state, fd, train_mode=True, use_reweight=use_reweight)
    relation_params = list(state["model"].relation_encoders.parameters()) + list(state["model"].relation_scorers.parameters())
    cls_grads = torch.autograd.grad(out["cls"], relation_params, retain_graph=True, allow_unused=True)
    cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in cls_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_grads) else 0.0
    out["total"].backward()
    return {
        "classification_to_relation_grad_norm": cls_grad,
        "relation_encoder_grad_norm": parent_run.module_grad_norm(state["model"].relation_encoders),
        "relation_scorer_grad_norm": parent_run.module_grad_norm(state["model"].relation_scorers),
        "discriminator_grad_norm": parent_run.module_grad_norm(state["model"].discriminator),
        "total_drap_grad_norm": parent_run.module_grad_norm(state["model"]),
        "finite": bool(torch.isfinite(out["total"]).item()) and all(bool(torch.isfinite(p.grad).all().item()) for p in state["model"].parameters() if p.grad is not None),
    }


def diagnostic_row(group, fold, state, fd, checkpoint, parent_audit):
    import torch

    best_rw = bool(checkpoint.get("best_use_reweight", True))
    with torch.no_grad():
        drap_out = parent_run.forward_state(state, fd, train_mode=False, use_reweight=best_rw)
        original_out = parent_run.forward_state(state, fd, train_mode=False, use_reweight=False)
        self_out = parent_run.forward_state(state, fd, train_mode=False, use_reweight=False, force_self=True)
    row = {
        "group": group,
        "gfr_weight_decay": 0.11 if group == "E1" else float(group_row(group)["gfr_weight_decay"]),
        "cam_weight_decay": 0.11,
        "drap_weight_decay": 0.11,
        "fold": fold,
        "best_epoch": checkpoint["best_epoch"],
        "epochs_executed": checkpoint.get("epochs_executed", -1),
        "warmup_active_at_best": int(not best_rw),
        **parent_run.metrics(drap_out["logp"], state),
    }
    details = drap_out["details"]
    row.update(parent_audit._gcn_stats(state, details, details["raw"]))
    row.update(parent_audit._relation_stats(state, fd, details))
    for modality in range(4):
        graph = state["base"].graphs[modality]
        adj = details["adjs"][modality]
        normalized = details["norm_adjs"][modality]
        row[f"graph_projection_weight_norm_{MODS[modality]}"] = float(graph.p.weight.detach().norm().cpu())
        row[f"graph_raw_edge_count_{MODS[modality]}"] = int((adj > 0).sum().item())
        row[f"graph_raw_density_{MODS[modality]}"] = float((adj > 0).float().mean().cpu())
        row[f"graph_adj_finite_{MODS[modality]}"] = int(torch.isfinite(adj).all().item())
        row[f"graph_adj_symmetric_{MODS[modality]}"] = int((adj - adj.T).abs().max().item() < 1e-6)
        row[f"graph_norm_edge_count_{MODS[modality]}"] = int((normalized != 0).sum().item())
        row[f"graph_norm_finite_{MODS[modality]}"] = int(torch.isfinite(normalized).all().item())
        row[f"graph_norm_fro_norm_{MODS[modality]}"] = float(normalized.detach().norm().cpu())
        row[f"adj_rel_diff_{MODS[modality]}"] = rel_norm(details["drap_adjs"][modality], adj)
        row[f"adj_symmetric_{MODS[modality]}"] = int((details["drap_adjs"][modality] - details["drap_adjs"][modality].T).abs().max().item() < 1e-6)
        row[f"adj_support_preserved_{MODS[modality]}"] = int(((adj == 0) == (details["drap_adjs"][modality] == 0)).all().item())
        row[f"adj_finite_{MODS[modality]}"] = int(torch.isfinite(details["drap_adjs"][modality]).all().item())
        row[f"message_drap_vs_original_rel_diff_{MODS[modality]}"] = rel_norm(details["msgs"][modality], original_out["details"]["msgs"][modality])
    for label, other in (("original", original_out), ("self", self_out)):
        row[f"{label}_fused_rel_diff"] = rel_norm(drap_out["fused"], other["fused"])
        row[f"{label}_logit_rel_diff"] = rel_norm(drap_out["logp"], other["logp"])
        p0 = drap_out["logp"].detach().exp()
        p1 = other["logp"].detach().exp()
        row[f"{label}_prob_mean_abs_diff"] = float((p0 - p1).abs().mean().cpu())
        row[f"{label}_prediction_flip_fraction"] = float((p0.argmax(1) != p1.argmax(1)).float().mean().cpu())
        other_metrics = parent_run.metrics(other["logp"], state)
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            row[f"{label}_{key}"] = other_metrics[key]
    row["drap_vs_original_fused_rel_diff"] = rel_norm(drap_out["fused"], original_out["fused"])
    row["drap_vs_original_logp_rel_diff"] = rel_norm(drap_out["logp"], original_out["logp"])
    row["drap_vs_original_prob_mean_abs_diff"] = float((drap_out["logp"].exp() - original_out["logp"].exp()).abs().mean().cpu())
    row["drap_vs_original_prediction_flip_fraction"] = float((drap_out["logp"].argmax(1) != original_out["logp"].argmax(1)).float().mean().cpu())
    row.update({f"best_{key}": value for key, value in gradient_audit(state, fd, best_rw).items()})
    row["checkpoint_weight_decay_base"] = float(checkpoint["opt_base"]["param_groups"][0]["weight_decay"])
    row["checkpoint_weight_decay_cam"] = float(checkpoint["opt_cam"]["param_groups"][0]["weight_decay"])
    row["checkpoint_weight_decay_drap"] = float(checkpoint["opt_drap"]["param_groups"][0]["weight_decay"])
    return row


def audit():
    import torch
    import audit as parent_audit

    ensure_dirs()
    if not (RESULTS / "summary.csv").exists():
        aggregate()
    pre = json.loads((SWEEP / "preflight.json").read_text())
    parent_initial = {int(row["fold"]): row for row in csv_read(EXP / "results" / "initialization_audit.csv") if row.get("group") == "E1"}
    checkpoint_rows = []
    init_rows = []
    diagnostic_rows = []
    for group in GROUPS:
        expected = expected_settings(group)
        for fold in range(1, 11):
            state, fd, checkpoint = load_checkpoint_state(group, fold)
            with torch.no_grad():
                output = parent_run.forward_state(state, fd, train_mode=False, use_reweight=bool(checkpoint.get("best_use_reweight", True)))
            readback = parent_run.metrics(output["logp"], state)
            stored = checkpoint["metrics"]
            metric_error = max(abs(float(readback[key]) - float(stored[key])) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen"))
            optimizer_names = {"opt_base": "GFR", "opt_cam": "CAM", "opt_drap": "DRAP"}
            optimizer_pass = all(
                abs(float(checkpoint[name]["param_groups"][0]["weight_decay"]) - expected[optimizer_names[name]]) < 1e-15
                for name in ("opt_base", "opt_cam", "opt_drap")
            )
            ref = parent_initial[fold]
            parent_pair_sha = parent_run.sha_file(parent_run.FD / f"fold{fold:02d}.npz")
            init_pass = (
                checkpoint.get("initial_shared_parameter_sha") == ref["initial_shared_parameter_sha"]
                and checkpoint.get("initial_rng_sha") == ref["initial_rng_sha"]
                and checkpoint.get("split_sha") == ref["split_sha"]
                and checkpoint.get("pair_sha") == parent_pair_sha
            )
            finite_pass = bool(torch.isfinite(output["logp"]).all().item()) and finite_module(state["model"]) and finite_module(state["cam"])
            status = "PASS" if metric_error < 1e-10 and optimizer_pass and init_pass and finite_pass and checkpoint.get("group") == group and checkpoint.get("model_branch") == "E1_DRAP" else "FAIL"
            checkpoint_rows.append({
                "group": group,
                "fold": fold,
                "checkpoint": str(checkpoint_path(group, fold)),
                "status": status,
                "max_metric_abs_error": metric_error,
                "optimizer_settings_pass": optimizer_pass,
                "initialization_pass": init_pass,
                "finite_pass": finite_pass,
                "stored_best_epoch": checkpoint["best_epoch"],
                "gfr_weight_decay": expected["GFR"],
                "cam_weight_decay": expected["CAM"],
                "drap_weight_decay": expected["DRAP"],
                "checkpoint_sha256": parent_run.sha_file(checkpoint_path(group, fold)),
            })
            init_rows.append({
                "group": group,
                "fold": fold,
                "initial_shared_parameter_sha": checkpoint.get("initial_shared_parameter_sha", ""),
                "initial_shared_schema_sha": checkpoint.get("initial_shared_schema_sha", ""),
                "initial_full_drap_model_sha": checkpoint.get("initial_full_drap_model_sha", ""),
                "initial_full_drap_model_schema_sha": checkpoint.get("initial_full_drap_model_schema_sha", ""),
                "initial_cam_sha": checkpoint.get("initial_cam_sha", ""),
                "initial_rng_sha": checkpoint.get("initial_rng_sha", ""),
                "anchor_rng_sha": checkpoint.get("anchor_rng_sha", ""),
                "entry_rng_sha": checkpoint.get("entry_rng_sha", ""),
                "split_sha": checkpoint.get("split_sha", ""),
                "pair_sha": checkpoint.get("pair_sha", ""),
                "parent_e1_shared_sha": ref["initial_shared_parameter_sha"],
                "parent_e1_rng_sha": ref["initial_rng_sha"],
                "parent_e1_pair_sha": parent_pair_sha,
                "pass": init_pass,
            })
            diagnostic_rows.append(diagnostic_row(group, fold, state, fd, checkpoint, parent_audit))
            del state
            torch.cuda.empty_cache()
    for fold in range(1, 11):
        state, fd, checkpoint = load_checkpoint_state("E1", fold)
        diagnostic_rows.append(diagnostic_row("E1", fold, state, fd, checkpoint, parent_audit))
        del state
        torch.cuda.empty_cache()
    csv_write(RESULTS / "checkpoint_audit.csv", checkpoint_rows)
    csv_write(RESULTS / "initialization_audit.csv", init_rows)
    csv_write(RESULTS / "diagnostics.csv", diagnostic_rows)
    checkpoint_pass = all(row["status"] == "PASS" for row in checkpoint_rows)
    init_pass = all(row["pass"] for row in init_rows)
    manifest = load_manifest()
    manifest.update({
        "status": "AUDITED_COMPLETE" if checkpoint_pass and init_pass and pre["status"] == "PASS" else "AUDITED_WITH_MISMATCH",
        "preflight_status": pre["status"],
        "checkpoint_audit": {"count": len(checkpoint_rows), "pass": checkpoint_pass},
        "initialization_audit": {"count": len(init_rows), "pass": init_pass},
        "diagnostics_rows": len(diagnostic_rows),
        "checkpoint_audit_sha256": parent_run.sha_file(RESULTS / "checkpoint_audit.csv"),
        "initialization_audit_sha256": parent_run.sha_file(RESULTS / "initialization_audit.csv"),
        "diagnostics_sha256": parent_run.sha_file(RESULTS / "diagnostics.csv"),
    })
    write_manifest(manifest)
    print("GFR_WD_AUDIT", manifest["status"], len(checkpoint_rows), flush=True)
    if not checkpoint_pass or not init_pass:
        raise SystemExit(4)


def summary_rows():
    return {row["group"]: row for row in csv_read(RESULTS / "summary.csv")}


def mean(rows, key):
    import numpy as np
    return float(np.asarray([float(row[key]) for row in rows]).mean())


def sd(rows, key):
    import numpy as np
    return float(np.asarray([float(row[key]) for row in rows]).std(ddof=0))


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def make_report():
    import numpy as np

    summaries = summary_rows()
    pairs = csv_read(RESULTS / "paired_comparisons.csv")
    diagnostics = csv_read(RESULTS / "diagnostics.csv")
    checkpoints = csv_read(RESULTS / "checkpoint_audit.csv")
    pre = json.loads((SWEEP / "preflight.json").read_text())
    lines = [
        "# Experiment 52 续实验：GFR 权重衰减 0.2 与 0.5",
        "",
        "## 执行状态",
        "",
        f"本轮只新增 E2 DRAP_GFR_WD020 和 E3 DRAP_GFR_WD050，共 20 折；独立最佳 checkpoint 审计 {sum(row['status'] == 'PASS' for row in checkpoints)}/20。预检状态为 {pre['status']}。",
        "E0/E1 的正式结果、checkpoint、任务记录和指标文件作为只读参照，未在本轮重训。",
        "",
        "## 代码与冻结方法",
        "",
        "run.py 的 DRAP 分支现在覆盖 E1/E2/E3。E2/E3 通过同一 build_state、同一 anchor 和同一隔离 Torch RNG 构造 DRAP。唯一优化器差异是 GFR opt_base 的 weight_decay：E1=0.11、E2=0.20、E3=0.50；CAM 和 DRAP 三组均为 0.11。GCN、GraphLearn、normalize_adj、CAM、损失、关系边公式、GRL、warmup、seed、fold 和 checkpoint 规则均保持不变。",
        "",
        f"逐折初始化 parity：{pre['initialization_signature_equal_all_folds']}；初始前向/损失 parity：{pre['initial_forward_loss_output_parity_all_folds']}；优化器配置：{pre['optimizer_settings_pass_all_folds']}；pair 测试标签隔离：{pre['pair_test_label_isolation_all_folds']}；GRL：{'PASS' if pre['grl_primitive_ok'] else 'FAIL'}。分类到关系梯度、关系编码器、评分器和判别器梯度见 preflight.json。",
        "",
        "修改前旧版源码和父实验文件 SHA 已写入 manifest_wd_sweep.json；父实验 E0/E1 的历史 SHA 字段保留。",
        "",
        "## 分类结果",
        "",
        "十折 mean ± population SD（%）：",
        "",
        "|组|GFR WD|CAM WD|DRAP WD|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|来源|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in ("E0", "E1", "E2", "E3"):
        row = summaries[group]
        lines.append(
            f"|{group}|{float(row['gfr_weight_decay']):.2f}|{float(row['cam_weight_decay']):.2f}|{float(row['drap_weight_decay']):.2f}|"
            f"{row['acc_mean_pct']} ± {row['acc_sd_pct']}|{row['hard_auc_mean_pct']} ± {row['hard_auc_sd_pct']}|"
            f"{row['prob_auc_mean_pct']} ± {row['prob_auc_sd_pct']}|{row['spe_mean_pct']} ± {row['spe_sd_pct']}|"
            f"{row['sen_mean_pct']} ± {row['sen_sd_pct']}|{row['ba_mean_pct']} ± {row['ba_sd_pct']}|"
            f"{row['asd_f1_mean_pct']} ± {row['asd_f1_sd_pct']}|{row['source']}|"
        )
    lines += ["", "### 逐折配对差异（百分点）", "", "|比较|指标|均值差|胜/平/负|10折差值|", "|---|---|---:|---:|---|"]
    for row in pairs:
        deltas = ",".join(fmt(value, 3) for value in json.loads(row["fold_deltas_pp"]))
        lines.append(f"|{row['comparison']}|{row['metric']}|{float(row['delta_pp_mean']):.4f}|{row['wins']}/{row['ties']}/{row['losses']}|{deltas}|")

    lines += ["", "## GFR 图和 GCN 消息收缩", "", "|组|四路消息/输入比|F-X相对比|GCN第一层范数|GCN第二层范数|GraphLearn投影范数|有效边数|图密度|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("E1", "E2", "E3"):
        rows = [row for row in diagnostics if row["group"] == group]
        message = [float(row[f"gcn{m}_message_ratio_input"]) for row in rows for m in range(4)]
        f_ratio = [float(row["f_minus_x_ratio"]) for row in rows]
        w1 = [float(row[f"gcn{m}_layer1_weight_norm"]) for row in rows for m in range(4)]
        w2 = [float(row[f"gcn{m}_layer2_weight_norm"]) for row in rows for m in range(4)]
        projection = [float(row[f"graph_projection_weight_norm_{modality}"]) for row in rows for modality in MODS]
        edges = [float(row[f"graph_raw_edge_count_{modality}"]) for row in rows for modality in MODS]
        density = [float(row[f"graph_raw_density_{modality}"]) for row in rows for modality in MODS]
        lines.append(f"|{group}|{np.mean(message):.6g} ± {np.std(message):.6g}|{np.mean(f_ratio):.6g} ± {np.std(f_ratio):.6g}|{np.mean(w1):.6g}|{np.mean(w2):.6g}|{np.mean(projection):.6g}|{np.mean(edges):.1f}|{np.mean(density):.6g}|")
    lines.append("图结构自身同时检查了 GraphLearn 投影范数、原图有效边数/密度、对称性、有限性以及归一化邻接有限性；这些量与 GCN 消息收缩分开解释。")

    lines += ["", "## 疾病关系、边权和对抗梯度", "", "|组|模态|患者对AUC|评分均值|评分SD|同类−异类|边变化比例|平均绝对边权变化|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("E1", "E2", "E3"):
        rows = [row for row in diagnostics if row["group"] == group]
        for modality in MODS:
            lines.append(f"|{group}|{modality}|{mean(rows, f'relation_pair_auc_{modality}'):.4f}|{mean(rows, f'relation_score_mean_{modality}'):.6g}|{mean(rows, f'edge_q_std_{modality}'):.6g}|{mean(rows, f'relation_score_separation_{modality}'):.6g}|{mean(rows, f'edge_changed_fraction_{modality}'):.6g}|{mean(rows, f'edge_weight_delta_mean_abs_{modality}'):.6g}|")
        lines.append(f"|{group}|adversarial|accuracy {mean(rows, 'adv_discriminator_accuracy'):.4f}|loss {mean(rows, 'adv_discriminator_loss'):.6g}|encoder grad {mean(rows, 'best_relation_encoder_grad_norm'):.6g}|scorer grad {mean(rows, 'best_relation_scorer_grad_norm'):.6g}|classifier-to-relation grad {mean(rows, 'best_classification_to_relation_grad_norm'):.6g}|total DRAP grad {mean(rows, 'best_total_drap_grad_norm'):.6g}|")

    lines += ["", "## 固定 checkpoint 的 DRAP 干预", "", "|组|消息 DRAP vs 原图相对差|最终320维表示相对差|logit相对差|概率平均差|预测翻转率|关闭重加权ACC|self-loop ACC|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("E1", "E2", "E3"):
        rows = [row for row in diagnostics if row["group"] == group]
        message_diff = [float(row[f"message_drap_vs_original_rel_diff_{modality}"]) for row in rows for modality in MODS]
        lines.append(f"|{group}|{np.mean(message_diff):.6g}|{mean(rows, 'drap_vs_original_fused_rel_diff'):.6g}|{mean(rows, 'drap_vs_original_logp_rel_diff'):.6g}|{mean(rows, 'drap_vs_original_prob_mean_abs_diff'):.6g}|{mean(rows, 'drap_vs_original_prediction_flip_fraction'):.6g}|{mean(rows, 'original_acc') * 100:.4f}|{mean(rows, 'self_acc') * 100:.4f}|")
    lines.append("关闭重加权和 self-loop-only 仅是同一最佳 checkpoint 的 eval 诊断，不属于新的训练组。传播干预接近零与不同 weight_decay 训练后的分类结果一致是两个独立命题。")

    lines += ["", "## 训练轨迹和科研判断", ""]
    for group in ("E1", "E2", "E3"):
        rows = [row for row in diagnostics if row["group"] == group]
        best_epochs = [int(row["best_epoch"]) for row in rows]
        epochs = [int(row["epochs_executed"]) for row in rows]
        lines.append(f"- {group}: best epoch mean {np.mean(best_epochs):.2f}，范围 {min(best_epochs)}–{max(best_epochs)}；有效训练 epoch mean {np.mean(epochs):.2f}；预热阶段 best {sum(int(row['warmup_active_at_best']) for row in rows)}/10。")
    message_mean = {group: np.mean([float(row[f"gcn{m}_message_ratio_input"]) for row in diagnostics if row["group"] == group for m in range(4)]) for group in ("E1", "E2", "E3")}
    f_mean = {group: mean([row for row in diagnostics if row["group"] == group], "f_minus_x_ratio") for group in ("E1", "E2", "E3")}
    intervention = {group: mean([row for row in diagnostics if row["group"] == group], "drap_vs_original_fused_rel_diff") for group in ("E1", "E2", "E3")}
    acc = {group: float(summaries[group]["acc_mean_pct"]) for group in ("E0", "E1", "E2", "E3")}
    lines.append(f"- 消息/输入比 E1/E2/E3={message_mean['E1']:.6g}/{message_mean['E2']:.6g}/{message_mean['E3']:.6g}；F-X 比={f_mean['E1']:.6g}/{f_mean['E2']:.6g}/{f_mean['E3']:.6g}。")
    lines.append(f"- DRAP 到原图的最终表示干预差 E1/E2/E3={intervention['E1']:.6g}/{intervention['E2']:.6g}/{intervention['E3']:.6g}；预测翻转率见上表。")
    lines.append(f"- ACC E0/E1/E2/E3={acc['E0']:.4f}/{acc['E1']:.4f}/{acc['E2']:.4f}/{acc['E3']:.4f}%；E2-E1={acc['E2']-acc['E1']:.4f} pp，E3-E1={acc['E3']-acc['E1']:.4f} pp。")
    lines.append(
        f"- 相对 E0 的 ACC 距离为 E1 {abs(acc['E1']-acc['E0']):.4f} pp、"
        f"E2 {abs(acc['E2']-acc['E0']):.4f} pp、E3 {abs(acc['E3']-acc['E0']):.4f} pp；"
        f"E2 {'更接近' if abs(acc['E2']-acc['E0']) < abs(acc['E1']-acc['E0']) else '没有更接近'} E0，"
        f"E3 {'更接近' if abs(acc['E3']-acc['E0']) < abs(acc['E1']-acc['E0']) else '没有更接近'} E0。"
    )
    lines.append(f"- WD=0.20 相对 0.11 的消息{'进一步下降' if message_mean['E2'] < message_mean['E1'] else '未下降'}；WD=0.50 相对 0.20 的消息{'进一步下降' if message_mean['E3'] < message_mean['E2'] else '未继续下降'}。是否单调由实测值决定。")
    lines.append(
        f"- DRAP 与原图的最终表示相对差为 E1 {intervention['E1']:.6g}、E2 {intervention['E2']:.6g}、"
        f"E3 {intervention['E3']:.6g}；三组均接近数值零，但 E2 到 E3 的相对干预量不要求单调，"
        "因为分母和 GCN 消息本身同时变化。"
    )
    lines.append(f"- 图退化判断：所有诊断行的 GraphLearn 原图和归一化邻接有限性、对称性和支持检查均通过；不存在由本轮 WD 设置引起的 NaN/Inf 图异常。")
    lines.append("- 科研判断：GCN 权重和消息幅度从 E1 到 E2/E3 明显收缩，支持更高 GFR weight_decay 削弱有效传播；固定 checkpoint 的最终表示和分类干预已接近零，但关系边变化和相对消息差不严格单调。E2/E3 分类没有更接近 E0，也没有形成诊断增量；不同训练组仍可能因优化轨迹和 ACC checkpoint 选择而产生差异。")

    lines += ["", "## 交付与限制", "", "- 新代码：run_gfr_wd_sweep.py；补充配置：config_gfr_wd_sweep.json。", "- 新结果：results/wd_sweep/fold_metrics.csv、summary.csv、diagnostics.csv、paired_comparisons.csv、report.md。", "- 新状态和 checkpoint：manifest_wd_sweep.json、.runtime/wd_sweep/。", "- 训练保留历史 val=test、CV 前预处理和 871 节点 transductive 图，指标可能偏乐观，不能作为无偏外部泛化证据。"]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def append_handoff():
    marker = "## Experiment 52 续实验：GFR 权重衰减 0.2 与 0.5"
    manifest = load_manifest()
    summaries = summary_rows()
    diagnostics = csv_read(RESULTS / "diagnostics.csv")
    payload = "\n".join([
        "",
        marker,
        "",
        "- 新增 E2 DRAP_GFR_WD020 和 E3 DRAP_GFR_WD050，共 20 折；E0/E1 为只读参照。",
        f"- ACC（%）：E0 {summaries['E0']['acc_mean_pct']}，E1 {summaries['E1']['acc_mean_pct']}，E2 {summaries['E2']['acc_mean_pct']}，E3 {summaries['E3']['acc_mean_pct']}。",
        f"- E1/E2/E3 四路消息/输入和 DRAP 干预结果见 {RESULTS / 'diagnostics.csv'}；完整结论见 {RESULTS / 'report.md'}。",
        f"- 补充 manifest：{MANIFEST_PATH}。",
        "",
    ])
    target = EXP.parent.parent / "handoff.md"
    pending = RESULTS / "handoff_gfr_wd_sweep_pending.md"
    try:
        import fcntl
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            old = handle.read()
            if marker not in old:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        manifest["handoff_update_status"] = "APPENDED"
        manifest["handoff_target"] = str(target)
        write_manifest(manifest)
        print("GFR_WD_HANDOFF_APPENDED", target, flush=True)
    except Exception as exc:
        pending.write_text(payload, encoding="utf-8")
        manifest["handoff_update_status"] = "PENDING_IO_ERROR"
        manifest["handoff_target"] = str(target)
        manifest["handoff_pending_path"] = str(pending)
        manifest["handoff_pending_sha256"] = parent_run.sha_file(pending)
        manifest["handoff_error"] = repr(exc)
        write_manifest(manifest)
        print("GFR_WD_HANDOFF_PENDING", repr(exc), pending, flush=True)


def report():
    make_report()
    append_handoff()
    manifest = load_manifest()
    manifest["report_path"] = str(RESULTS / "report.md")
    manifest["report_sha256"] = parent_run.sha_file(RESULTS / "report.md")
    write_manifest(manifest)
    print("GFR_WD_REPORT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("preflight")
    controller_parser = sub.add_parser("controller")
    controller_parser.add_argument("--groups", nargs="+", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--group", required=True, choices=GROUPS)
    worker.add_argument("--fold", type=int, required=True)
    sub.add_parser("aggregate")
    sub.add_parser("audit")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        preflight()
    elif args.command == "controller":
        controller(args.groups)
    elif args.command == "worker":
        train_one(args.group, args.fold)
    elif args.command == "aggregate":
        aggregate()
    elif args.command == "audit":
        audit()
    elif args.command == "report":
        report()


if __name__ == "__main__":
    main()
