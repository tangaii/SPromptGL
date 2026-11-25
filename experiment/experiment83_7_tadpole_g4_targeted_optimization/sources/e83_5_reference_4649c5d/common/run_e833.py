from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import io
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from fidelity_model import FidelityModel, fit_spectral_statistics
from fold_checkpoint import (DATA_SHA, SPLIT_SHA, atomic_write, fingerprint, identity, implementation_hashes, lock,
                             publish_epoch, recover_best, validate_identity, write_json)
from metrics import acc_auc
from scmr_optimization import load_scmr_optimization
from source_runtime import relation_pair_sources

GROUPS = ["G1_E28_E0", "G2_E43_1_E00", "G3_E74_4_G2", "G4_E82_4_6_10_G1",
          "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM"]
DIMS = [145, 148, 3, 36, 24, 9]
PROMPT8 = str(ROOT / "sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt")
PROMPT12 = str(ROOT / "inputs/prompts/mmGPL12_scmr.pt")
SEED = 0
MAX_EPOCHS = 450
PATIENCE = 200


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(value) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    if (sha_file(ROOT / "sources/processed_standard_data.csv") != DATA_SHA
            or sha_file(ROOT / "sources/split_indices.json") != SPLIT_SHA):
        raise RuntimeError("TADPOLE locked data or split SHA differs")
    frame = pd.read_csv(ROOT / "sources/processed_standard_data.csv")
    ids = pd.read_csv(ROOT / "sources/sample_ids.csv")
    x = frame.drop(columns=["LABEL"]).to_numpy(np.float32)
    y = frame["LABEL"].to_numpy(np.int64) - 1
    splits = json.loads((ROOT / "sources/split_indices.json").read_text())['splits']
    assert x.shape == (598, 365) and len(ids) == 598
    return x, y, ids, splits


def make_pairs(train_idx: np.ndarray, labels: np.ndarray, fold: int, supports):
    pairs, same = relation_pair_sources().make_outer_pairs(fold, supports, train_idx, labels)
    if len(pairs) != 4096 or int(same.sum()) != 2048:
        raise RuntimeError("TADPOLE fold cannot provide 2048 same and 2048 different train pairs")
    return pairs, same.astype(np.float32)


@torch.no_grad()
def anchor_pair_support(model, x):
    supports = []
    for graph, part in zip(model.gfr.graphs, x.detach().cpu().split(DIMS, dim=1)):
        # E53.3 obtains the anchor support from the CPU GraphLearn forward.
        anchor_graph = copy.deepcopy(graph).cpu()
        row, col = torch.triu(anchor_graph(part) > 0, diagonal=1).nonzero(as_tuple=True)
        supports.append(torch.stack((row, col), dim=1).numpy())
    fallback = [len(support) < len(x) for support in supports]
    model.gfr.fallback_modal.copy_(model.gfr.fallback_modal.new_tensor(fallback))
    return supports


def build_model(group: str, spectral_stats=None, device=None, *, fold=1):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_path = PROMPT12 if group in ("G1_E28_E0", "G4_E82_4_6_10_G1") else PROMPT8
    return FidelityModel(DIMS, group, 3, prompt_path, spectral_stats, 824602 * 100 + fold,
                         fold=fold, device=device).to(device)


def optimizers(model: FidelityModel, group: str):
    base_decay = 0.11 if model.use_darg else 0.0036
    base_options = {"lr": 0.012, "weight_decay": base_decay}
    opts = {
        "base": torch.optim.Adam(model.graph_parameters(), **base_options),
        "cam": torch.optim.Adam(model.cam.parameters(), **base_options),
    }
    config = {
        "class": "Adam/AdamW" if model.use_darg else "Adam",
        "gfr": [0.012, base_decay], "cam": [0.012, base_decay],
    }
    if model.use_darg:
        decay, lower_rank = [], []
        for name, param in model.darg.relation_named_parameters():
            bucket = lower_rank if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower() else decay
            bucket.append(param)
        # E61 preserves the E53.3 group order and sets decay to 0.01 in each.
        opts["darg"] = torch.optim.AdamW([{"params": decay}, {"params": lower_rank}],
                                        lr=0.001, weight_decay=0.01)
        config["darg"] = [0.001, 0.01]
        config["dpsl"] = None
    if model.dpsl is not None:
        opts["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
        config["dpsl"] = [0.0005, 0.01]
    return opts, config


def zero_optimizers(opts):
    for opt in opts.values():
        opt.zero_grad(set_to_none=True)


def step_optimizers(opts):
    for opt in opts.values():
        opt.step()


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def warmup_step(model, opts, x, labels, train_t, pair_t, same_t):
    model.train()
    zero_optimizers(opts)
    _, aux = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=False, darg_warmup=True)
    _, loss = historical_batch_loss(aux, labels, train_t, use_darg=True)
    loss.backward()
    step_optimizers(opts)
    return loss.detach()


def warmup_shared_base(model, opts, x, labels, train_t, pair_t, same_t):
    """E61's ten source steps using original graphs and all relation losses."""
    history = []
    for epoch in range(1, 11):
        loss = warmup_step(model, opts, x, labels, train_t, pair_t, same_t)
        history.append({"epoch": epoch, "loss": float(loss)})
    # Historical E74 warmup gates have no moments. Freeze the E53.3 gate
    # while keeping its parameter in the inherited AdamW group.
    darg_opt = opts["darg"]
    for gate in (model.gfr.edge_gate_logits, model.gfr.adaptive_gate_logits):
        if gate in darg_opt.state:
            raise RuntimeError("A DARG gate received an optimizer update in E61 warmup")
    model.darg.enter_positive_dcr()
    return history, capture_rng_state()


def make_warmup_payload(fold, model, opts, history, warmup_rng, pair_t, same_t):
    return {
        "fold": int(fold),
        "source": "TADPOLE E51/E53.3/E61 epoch-10 original-graph warmup",
        "epochs": 10, "history": history,
        "base_state_sha256": sha_object(model.state_dict()),
        "optimizer_state_sha256": sha_object({key: value.state_dict() for key, value in opts.items()}),
        "rng_state_sha256": sha_object(warmup_rng),
        "pair_sha256": sha_object((pair_t.detach().cpu(), same_t.detach().cpu())),
        "fallback_modal": model.gfr.fallback_modal.tolist(),
    }


def replay_g3_warmup(fold, x, labels, y_np, train_idx, test_idx, device):
    """Rebuild the archived G3 fold warmup in an isolated deterministic RNG scope."""
    outside_rng = capture_rng_state()
    try:
        seed_everything(SEED)
        train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        reference = build_model("G3_E74_4_G2", device=device, fold=fold)
        opts, _ = optimizers(reference, "G3_E74_4_G2")
        supports = anchor_pair_support(reference, x)
        pair_np, same_np = make_pairs(np.asarray(train_idx, dtype=np.int64), y_np, fold, supports)
        pair_t = torch.as_tensor(pair_np, dtype=torch.long, device=device)
        same_t = torch.as_tensor(same_np, dtype=torch.float32, device=device)
        history, warmup_rng = warmup_shared_base(reference, opts, x, labels, train_t, pair_t, same_t)
        payload = make_warmup_payload(fold, reference, opts, history, warmup_rng, pair_t, same_t)

        g3_dir = ROOT / "G3_E74_4_G2"
        best_path = g3_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
        latest_path = g3_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        expected = {
            "group": "G3_E74_4_G2", "fold": fold, "seed": SEED,
            "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
            "train_idx": np.asarray(train_idx, dtype=np.int64).tolist(),
            "test_idx": np.asarray(test_idx, dtype=np.int64).tolist(),
        }
        for checkpoint in (best, latest):
            if any(checkpoint.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"G3 warmup source identity differs for fold {fold}")
        best_sha = sha_file(best_path)
        if latest.get("best_sha256") != best_sha:
            raise RuntimeError(f"G3 Latest does not reference its Best checkpoint for fold {fold}")
        if best["warmup"] != latest["warmup"] or best["warmup"] != payload:
            raise RuntimeError(f"G3 warmup replay differs from its sealed snapshot for fold {fold}")
        if not torch.equal(best["relation_pairs"], latest["relation_pairs"]):
            raise RuntimeError(f"G3 Best/Latest relation pairs differ for fold {fold}")
        if not torch.equal(best["relation_targets"], latest["relation_targets"]):
            raise RuntimeError(f"G3 Best/Latest relation targets differ for fold {fold}")
        if not torch.equal(best["relation_pairs"], pair_t.cpu()) or not torch.equal(best["relation_targets"], same_t.cpu()):
            raise RuntimeError(f"G3 warmup replay relation pairs differ for fold {fold}")
        parity = {
            "status": "PASS", "reference_group": "G3_E74_4_G2", "fold": fold,
            "g3_best_sha256": best_sha, "g3_latest_sha256": sha_file(latest_path),
            "base_state_sha256": payload["base_state_sha256"],
            "optimizer_state_sha256": payload["optimizer_state_sha256"],
            "rng_state_sha256": payload["rng_state_sha256"],
            "relation_pair_sha256": payload["pair_sha256"],
            "warmup_history_exact": True, "base_state_exact": True,
            "optimizer_state_exact": True, "rng_exact": True,
            "relation_pairs_exact": True,
        }
        return payload, pair_t.cpu(), same_t.cpu(), parity
    finally:
        restore_rng(outside_rng)


def weighted_nll(logp, labels, rows):
    train_labels = labels[rows].detach().cpu().numpy()
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights = torch.tensor(len(rows) / (np.maximum(counts, 1) * 3), dtype=logp.dtype, device=logp.device)
    return F.nll_loss(logp[rows], labels[rows], weight=weights)


def historical_batch_loss(aux, labels, rows, use_darg):
    native = F.log_softmax(aux["native_logits"], dim=-1)
    classification = weighted_nll(native, labels, rows)
    total = classification + aux["graph_loss"]
    if use_darg:
        total = total + 0.05 * aux["pair_loss"] + 0.01 * aux["adv_loss"] + 0.01 * aux["preserve_loss"]
    return classification, total


def scmr_parameter_families(model, opts):
    def owned(name):
        return tuple(param for group in opts[name].param_groups for param in group["params"])

    head = tuple(model.cam.Outputlayer.parameters())
    excluded = set(head)
    control = tuple(model.dpsl.evidence_tokens.parameters()) + tuple(model.dpsl.calibrator.parameters())
    controller_parameters = set(control)
    groups = {
        "gfr": owned("base"),
        "darg": owned("darg"),
        "cam": tuple(param for param in owned("cam") if param not in excluded),
        "head": head,
        "prompt": tuple(param for param in owned("dpsl") if param not in controller_parameters),
        "control": control,
    }
    parameters = [param for group in groups.values() for param in group]
    if not all(groups.values()) or len(set(parameters)) != len(parameters) or set(parameters) != set(model.parameters()):
        raise RuntimeError("SCMR optimizer families do not partition the active model")
    return groups


def scmr_optimizer_step(model, opts, historical_loss, objectives):
    optimization = load_scmr_optimization()
    groups = scmr_parameter_families(model, opts)
    budget, before = optimization.apply_budget("G2", groups, {"total": historical_loss}, objectives)
    step_optimizers(opts)
    return {
        "families": optimization.optimizer_measurements(groups, budget, before),
        "shared_cap": optimization.CLASS["shared_cap"],
    }


def make_dirs(group):
    path = ROOT / group
    for name in ("best_checkpoints", "latest_checkpoints", "predictions", "audits"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def require_formal_training_gate():
    audit_path = ROOT / "audits/preflight_start.json"
    protocol_path = ROOT / "protocol.json"
    seal_path = ROOT / "audits/training_seal.json"
    resource_path = ROOT / "audits/resource_probe.json"
    matrix_path = ROOT / "audits/method_fidelity_matrix.md"
    if not all(path.is_file() for path in (audit_path, protocol_path, seal_path, resource_path, matrix_path)):
        raise RuntimeError("Formal training is blocked because the E83.5 start seal is incomplete.")
    preflight = json.loads(audit_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    seal = json.loads(seal_path.read_text())
    checks = preflight.get("checks", [])
    failed = [item["id"] for item in checks if item.get("status") != "PASS"]
    prompt_manifest = json.loads((ROOT / "prompt_manifest.json").read_text())
    expected = {
        "experiment": "Experiment 83.5",
        "protocol_sha256": sha_file(protocol_path),
        "preflight_sha256": sha_file(audit_path),
        "resource_probe_sha256": sha_file(resource_path),
        "method_fidelity_matrix_sha256": sha_file(matrix_path),
        "implementation_sha256": implementation_hashes(),
        "data_sha256": DATA_SHA,
        "split_sha256": SPLIT_SHA,
        "prompt_manifest_sha256": sha_file(ROOT / "prompt_manifest.json"),
        "source_manifest_sha256": sha_file(ROOT / "sources/source_manifest.json"),
        "prompt_vector_sha256": prompt_manifest["vector_sha256"],
        "prompt_query_sha256": prompt_manifest["query_sha256"],
        "group_config_sha256s": {
            group: sha_file(ROOT / "configs" / f"{group}.json") for group in GROUPS
        },
    }
    allowed = (protocol.get("experiment") == "Experiment 83.5"
               and protocol.get("experiment_identity") == expected["experiment"]
               and preflight.get("status") == "PASS"
               and preflight.get("formal_training_allowed") is True
               and preflight.get("data_sha256") == DATA_SHA
               and preflight.get("split_sha256") == SPLIT_SHA
               and preflight.get("source_manifest_sha256") == expected["source_manifest_sha256"]
               and preflight.get("group_config_sha256s") == expected["group_config_sha256s"]
               and preflight.get("resource_probe_sha256") == expected["resource_probe_sha256"]
               and preflight.get("implementation_sha256") == implementation_hashes()
               and preflight.get("prompt_adapter", {}).get("query_sha256") == expected["prompt_query_sha256"]
               and not failed
               and all(seal.get(key) == value for key, value in expected.items()))
    if not allowed:
        detail = ", ".join(failed) if failed else "E83.5 start seal no longer matches current inputs or code"
        raise RuntimeError(f"Formal training blocked by E83.5 method-fidelity preflight: {detail}")


def save_json(path: Path, value):
    write_json(path, value)


def train_fold(group: str, fold: int, x_np, y_np, ids, split, device, log):
    group_dir = make_dirs(group)
    best_path = group_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = group_dir / "audits" / f"fold{fold:02d}.json"
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    artifact = identity(group, fold, train_idx, test_idx)
    saved = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.is_file() else None
    if saved is not None:
        validate_identity(saved, artifact)
        recover_best(best_path, saved["best_sha256"])
    elif best_path.exists():
        raise RuntimeError(f"Unsealed Best without Latest at {best_path}")
    seed_everything(SEED)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train_t = torch.from_numpy(train_idx).long().to(device)
    stats = saved["spectral_stats"] if saved is not None else None
    model = build_model(group, spectral_stats=stats, device=device, fold=fold)
    opts, opt_cfg = optimizers(model, group)
    pair_t = same_t = warmup_payload = None
    warmup_parity = None
    history = []
    best_acc, best_epoch, wait, best_sha = -1.0, -1, 0, None
    start_epoch = 1
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        if model.use_darg:
            model.darg.enter_positive_dcr()
        if opt_cfg != saved["optimizer_config"] or opts.keys() != saved["optimizers"].keys():
            raise RuntimeError("Saved optimizer contract differs from the frozen protocol")
        for name, state in saved["optimizers"].items():
            opts[name].load_state_dict(state)
        pair_t, same_t = saved["relation_pairs"], saved["relation_targets"]
        if pair_t is not None:
            pair_t, same_t = pair_t.to(device), same_t.to(device)
        warmup_payload, history = saved["warmup"], saved["history"]
        warmup_parity = saved.get("warmup_parity")
        best_acc, best_epoch, wait, best_sha = (saved[key] for key in ("best_acc", "best_epoch", "wait", "best_sha256"))
        start_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"])
        log.write(f"RESUME group={group} fold={fold:02d} next_epoch={start_epoch}\n")
    elif model.use_darg:
        supports = anchor_pair_support(model, x)
        pair_np, same_np = make_pairs(train_idx, y_np, fold, supports)
        pair_t = torch.from_numpy(pair_np).long().to(device)
        same_t = torch.from_numpy(same_np).to(device)
        warm_history, warmup_rng = warmup_shared_base(model, opts, x, labels, train_t, pair_t, same_t)
        warmup_payload = make_warmup_payload(fold, model, opts, warm_history, warmup_rng, pair_t, same_t)
        if model.use_dpsl:
            model.eval()
            with torch.no_grad():
                _, warm_aux = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=False)
            stats = fit_spectral_statistics(warm_aux["cam"], labels, train_t, fold=fold,
                                            warm_logits=warm_aux["native_logits"])
            stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
            stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
            model.attach_dpsl(stats, PROMPT12, 824602 * 100 + fold)
            opts["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
            opt_cfg["dpsl"] = [0.0005, 0.01]
        if sha_object(capture_rng_state()) != warmup_payload["rng_state_sha256"]:
            raise RuntimeError(f"G3/G4 warmup RNG changed before first formal step in fold {fold}")
        start_epoch = 11

    if group == GROUPS[3]:
        reference_payload, reference_pairs, reference_targets, reference_parity = replay_g3_warmup(
            fold, x, labels, y_np, train_idx, test_idx, device)
        if warmup_payload != reference_payload:
            raise RuntimeError(f"G4 warmup differs from the deterministic G3 replay for fold {fold}")
        if not torch.equal(pair_t.detach().cpu(), reference_pairs):
            raise RuntimeError(f"G4 relation pairs differ from G3 for fold {fold}")
        if not torch.equal(same_t.detach().cpu(), reference_targets):
            raise RuntimeError(f"G4 relation targets differ from G3 for fold {fold}")
        warmup_parity = reference_parity
        if saved is not None and saved.get("warmup_parity") != warmup_parity:
            raise RuntimeError(f"Resumed G4 warmup parity record differs for fold {fold}")

    def persist(epoch, new_best=None):
        return publish_epoch(latest_path, best_path, {
            **artifact, "epoch": epoch, "model": model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in opts.items()},
            "optimizer_config": opt_cfg, "rng": capture_rng_state(),
            "best_epoch": best_epoch, "best_acc": best_acc, "best_sha256": best_sha,
            "wait": wait, "history": history, "spectral_stats": stats, "warmup": warmup_payload,
            "warmup_parity": warmup_parity,
            "relation_pairs": pair_t, "relation_targets": same_t,
        }, new_best)

    if not latest_path.is_file():
        persist(start_epoch - 1)
    model.train()
    started = time.monotonic()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if wait > PATIENCE:
            break
        zero_optimizers(opts)
        _, aux = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=True)
        cls, historical_total = historical_batch_loss(aux, labels, train_t, model.use_darg)
        total = historical_total + aux["dpsl_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss group={group} fold={fold} epoch={epoch}")
        if model.dpsl is not None:
            budget = scmr_optimizer_step(model, opts, historical_total, aux["scmr_objectives"])
        else:
            historical_total.backward()
            step_optimizers(opts)
            budget = {"families": {}, "shared_cap": 0.0}
        model.eval()
        with torch.no_grad():
            eval_logp, eval_aux = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=True)
            pred = eval_logp.argmax(1)
            val_acc = int((pred[test_idx] == labels[test_idx]).sum()) / len(test_idx)
        model.train()
        history.append({"epoch": epoch, "loss": float(total.detach()), "cls": float(cls.detach()), "val_acc": val_acc,
                        "graph": float(aux["graph_loss"].detach()), "pair": float(aux["pair_loss"].detach()),
                        "adv": float(aux["adv_loss"].detach()), "preserve": float(aux["preserve_loss"].detach()),
                        "dpsl": float(aux["dpsl_loss"].detach())})
        if epoch == start_epoch or epoch % 10 == 0:
            log.write(f"group={group} fold={fold:02d} epoch={epoch:03d} loss={float(total):.6f} val_acc={val_acc:.6f}\n")
            log.flush()
        new_best = None
        if val_acc > best_acc:
            best_acc, best_epoch, wait = val_acc, epoch, 0
            prediction_at_best = pred[test_idx].detach().cpu().numpy().astype(np.int64)
            new_best = {
                **artifact,
                "epoch": epoch, "best_val_acc": best_acc,
                "model": copy.deepcopy(model.state_dict()), "optimizer_config": opt_cfg,
                "history_at_best": history[-1], "spectral_stats": stats,
                "warmup": warmup_payload, "warmup_parity": warmup_parity,
                "relation_pairs": pair_t,
                "relation_targets": same_t,
                "dpsl_ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
                "gradient_budget": budget,
                "metrics_at_best": acc_auc(y_np[test_idx], prediction_at_best, 3),
                "prediction_sha256": hashlib.sha256(prediction_at_best.tobytes()).hexdigest(),
            }
        else:
            wait += 1
        if model.dpsl is not None:
            # Best has the bank used by this evaluation; Latest gets the
            # advanced state required by the next epoch.
            model.dpsl.advance_ema(eval_aux["cam"])
        if new_best is not None or epoch % 10 == 0 or epoch == MAX_EPOCHS or wait > PATIENCE:
            best_sha = persist(epoch, new_best)
        if wait > PATIENCE:
            break
    if best_sha is None or best_epoch < 1:
        raise RuntimeError("no best checkpoint")
    loaded = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_identity(loaded, artifact)
    model.load_state_dict(loaded["model"], strict=True)
    model.eval()
    with torch.no_grad():
        logp, aux = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=True)
        native_logp = None
        if model.dpsl is not None:
            native_logp, _ = model(x, labels, train_t, pair_t, same_t, dpsl_enabled=False)
    probability = logp.exp().detach().cpu().numpy()
    prediction = logp.argmax(1).cpu().numpy()
    metric = acc_auc(y_np[test_idx], prediction[test_idx], 3)
    pred_sha = hashlib.sha256(prediction[test_idx].astype(np.int64).tobytes()).hexdigest()
    if metric != loaded["metrics_at_best"] or pred_sha != loaded["prediction_sha256"]:
        raise RuntimeError("Best model reload differs from its original selected predictions")
    pred_path = group_dir / "predictions" / f"fold{fold:02d}.csv"
    prediction_table = {"fold": fold, "seed": SEED, "sample_index": test_idx,
                        "sample_id": ids.iloc[test_idx]["RID"].to_numpy(), "true_label": y_np[test_idx],
                        "pred_label": prediction[test_idx], "prob_NC": probability[test_idx, 0],
                        "prob_MCI": probability[test_idx, 1], "prob_AD": probability[test_idx, 2]}
    if native_logp is not None:
        native_probability = native_logp.exp().detach().cpu().numpy()
        native_prediction = native_logp.argmax(1).cpu().numpy()
        prediction_table.update(native_pred_label=native_prediction[test_idx],
                                native_prob_NC=native_probability[test_idx, 0],
                                native_prob_MCI=native_probability[test_idx, 1],
                                native_prob_AD=native_probability[test_idx, 2])
    pd.DataFrame(prediction_table).to_csv(pred_path, index=False)
    replay = {**artifact, "status": "TRAINED", "checkpoint_sha256": best_sha,
              "best_epoch": best_epoch, "data_sha256": loaded["data_sha256"], "split_sha256": loaded["split_sha256"],
              "prediction_sha256": pred_sha,
              "native_prediction_sha256": (hashlib.sha256(native_prediction[test_idx].astype(np.int64).tobytes()).hexdigest()
                                            if native_logp is not None else None),
              "native_metrics": (acc_auc(y_np[test_idx], native_prediction[test_idx], 3)
                                 if native_logp is not None else None),
              "ACC": metric["ACC"], "AUC": metric["AUC"], "pen_present": bool(model.cam.use_pen),
              "darg_active": bool(model.use_darg), "dpsl_active": bool(model.use_dpsl),
              "dpsl_residual_max": float(aux["dpsl"].get("residual_ratio", torch.zeros(1)).max()) if model.dpsl else 0.0,
              "dpsl_spectral_basis_shape": list(model.dpsl.basis.shape) if model.dpsl else None,
              "dpsl_ema_updates": int(model.dpsl.ema_updates) if model.dpsl else 0,
              "pair_count": len(pair_t) if pair_t is not None else 0,
              "same_pairs": int(same_t.sum()) if same_t is not None else 0,
              "different_pairs": int((same_t == 0).sum()) if same_t is not None else 0,
              "train_count": len(train_idx), "test_count": len(test_idx),
              "last_epoch": history[-1]["epoch"], "stop_wait": wait, "warmup": warmup_payload,
              "warmup_parity": warmup_parity,
              "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
              "checkpoint": str(best_path), "prediction": str(pred_path),
              "audit": str(audit_path), "train_seconds": time.monotonic() - started}
    save_json(audit_path, replay)
    return replay


def write_rows(path: Path, rows):
    fields = ["group", "fold", "seed", "train_count", "test_count", "best_epoch", "ACC", "AUC", "checkpoint", "prediction", "audit", "train_seconds"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda output: output.write(stream.getvalue().encode()))


def prepare_group(group, workers):
    group_dir = ROOT / group
    config_path = group_dir / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous.get("implementation_sha256") not in (None, fingerprint()):
            raise RuntimeError(f"Existing sealed release differs for {group}")
        if previous.get("implementation_sha256") is None:
            # The pre-repair outputs do not satisfy the current release gate.
            # Rename within the experiment filesystem so they remain available.
            legacy = ROOT / "unaccepted_pre_repair" / group
            legacy.parent.mkdir(exist_ok=True)
            group_dir.rename(legacy)
    dir_ = make_dirs(group)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    save_json(dir_ / "config.json", {
        "experiment": "Experiment 83.5", "group": group, "seed": SEED, "device": "cuda:0",
        "implementation_sha256": fingerprint(), "max_epochs": MAX_EPOCHS,
        "early_stop_wait_gt": PATIENCE, "protocol": protocol["groups"][group],
        "actual_workers": workers,
    })


def configure_worker():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "after parallel work has started" not in str(exc):
            raise
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)


def execute_fold(group, fold):
    require_formal_training_gate()
    x, y, ids, splits = load_data()
    split = next(split for split in splits if int(split["fold"]) == fold)
    directory = ROOT / group
    device = torch.device("cuda:0")
    with lock(directory / "latest_checkpoints" / f"fold{fold:02d}.lock"):
        audit_path = directory / "audits" / f"fold{fold:02d}.json"
        if audit_path.is_file():
            from replay_check import replay_fold

            replay_fold(group, split, (x, y, ids, splits), device)
            return {"group": group, "fold": fold, "status": "VERIFIED_PREVIOUS"}
        torch.cuda.reset_peak_memory_stats(device)
        with (directory / "train.log").open("a", buffering=1) as log:
            log.write(f"BEGIN group={group} fold={fold:02d} seed={SEED}\n")
            result = train_fold(group, fold, x, y, ids, split, device, log)
            log.write(f"TRAINED group={group} fold={fold:02d} best_epoch={result['best_epoch']}\n")
        return result


def summarize_group(group, workers):
    dir_ = ROOT / group
    rows = [json.loads(path.read_text()) for path in sorted((dir_ / "audits").glob("fold*.json"))]
    rows = [row for row in rows if row.get("status") == "TRAINED"
            and row.get("implementation_sha256") == fingerprint()]
    write_rows(dir_ / "fold_metrics.csv", rows)
    summary = {"group": group, "folds": len(rows), "data_sha256": DATA_SHA,
               "split_sha256": SPLIT_SHA, "actual_workers": workers}
    if len(rows) == 10:
        for name in ("ACC", "AUC"):
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            summary.update({f"{name}_mean": float(values.mean()), f"{name}_std": float(values.std(ddof=0))})
    save_json(dir_ / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=GROUPS + ["all"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("--workers must be from 1 to 10")
    require_formal_training_gate()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Formal execution requires the assigned single GPU")
    if args.group in GROUPS[1:3]:
        parser.error("G2/G3 are inherited audited folds and cannot be trained by this runner")
    groups = [GROUPS[0], GROUPS[3]] if args.group == "all" else [args.group]
    if GROUPS[4] in groups:
        parser.error("G5 is executed by common/run_g5.py after the CAM-only preflight")
    requested = os.environ.get("E831_FOLDS")
    folds = sorted({int(v) for v in requested.split(",")}) if requested else list(range(1, 11))
    if not set(folds) <= set(range(1, 11)):
        parser.error("E831_FOLDS must contain fold IDs from 1 to 10")
    failures = []
    with lock(ROOT / ".training.lock"):
        for group in groups:
            prepare_group(group, args.workers)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=configure_worker) as pool:
            tasks = {pool.submit(execute_fold, group, fold): (group, fold)
                     for fold in folds for group in reversed(groups)}
            for task in as_completed(tasks):
                group, fold = tasks[task]
                try:
                    result = task.result()
                    print(f"{result['status']} {group} fold {fold:02d}", flush=True)
                except Exception as exc:
                    failures.append((group, fold))
                    import traceback
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    print(f"ERROR {group} fold {fold:02d}: {exc!r}", file=sys.stderr, flush=True)
        for group in groups:
            summarize_group(group, args.workers)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
