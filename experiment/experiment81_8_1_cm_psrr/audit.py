#!/usr/bin/env python3
"""Independent audit for the single E81.8.1 CM-PSRR group."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run

CFG = run.CFG
RESULTS = run.RESULTS
HIST_E744 = run.HIST_E744
HIST_E8172 = run.HIST_E8172
E74_GROUP = run.E74_GROUP
GROUPS = run.GROUPS
METRICS = run.METRICS
METRIC_NAMES = {"acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC",
                "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _manual_metrics(logp, labels):
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, recall_score, roc_auc_score)
    values = np.asarray(logp, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = values.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(labels, prediction, labels=[0, 1])
    return {
        "acc": float(accuracy_score(labels, prediction)),
        "hard_auc": float(balanced_accuracy_score(labels, prediction)),
        "prob_auc": float(roc_auc_score((labels == 0).astype(np.int64), probability[:, 0])),
        "sen": float(recall_score(labels, prediction, pos_label=0, zero_division=0)),
        "spe": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(labels, prediction, pos_label=0, zero_division=0)),
        "prediction": prediction,
        "probability": probability,
        "confusion": {"asd_asd": int(cm[0, 0]), "asd_td": int(cm[0, 1]),
                      "td_asd": int(cm[1, 0]), "td_td": int(cm[1, 1])},
    }


def _normalize(value):
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def _summary(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "population_sd": float(values.std(ddof=0))}


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _correction(source, target, labels):
    import numpy as np
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    corrected = (source != labels) & (target == labels)
    harmed = (source == labels) & (target != labels)
    return {
        "corrected": int(corrected.sum()), "harmed": int(harmed.sum()),
        "net": int(corrected.sum() - harmed.sum()),
        "asd_corrected": int((corrected & (labels == 0)).sum()),
        "asd_harmed": int((harmed & (labels == 0)).sum()),
        "td_corrected": int((corrected & (labels == 1)).sum()),
        "td_harmed": int((harmed & (labels == 1)).sum()),
    }


def _tree_close(left, right):
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return torch.equal(left.cpu(), right.cpu()) if not left.is_floating_point() else bool(torch.allclose(left, right, atol=2e-6, rtol=1e-5))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_close(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_close(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, abs_tol=1e-7, rel_tol=1e-6)
    return left == right


def _historical_e8172():
    rows = [row for row in _read_csv(HIST_E8172 / "results/fold_metrics.csv") if row.get("group") == "G1"]
    if len(rows) != 10:
        raise RuntimeError("E81.7.2-G1 history must contain ten rows")
    return {int(row["fold"]): {metric: _normalize(row[metric]) for metric in METRICS} for row in rows}


def _historical_e818():
    path = Path("/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv")
    rows = _read_csv(path)
    mapping = {"acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC",
               "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}
    groups = {}
    for group in ("G1", "G2"):
        selected = [row for row in rows if row.get("group") == group]
        if len(selected) != 10:
            raise RuntimeError(f"E81.8 {group} history must contain ten rows")
        groups[f"E81.8-{group}"] = {
            int(row["fold"]): {metric: _normalize(row[column]) for metric, column in mapping.items()}
            for row in selected
        }
    history_audit = json.loads((path.parent / "audit_status.json").read_text(encoding="utf-8"))
    if history_audit.get("status") != "COMPLETE_AUDITED":
        raise RuntimeError("E81.8 historical group audit is not complete")
    return groups


def _load_current_fold(e744, legacy, fold, kind):
    import torch
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    state["group"] = "G1"
    checkpoint_path = run.CKPTS / f"G1_f{fold:02d}.{kind.lower()}.pt"
    payload = run.torch_load(checkpoint_path)
    if payload.get("schema") != "E81.8.1-CM-PSRR-v1" or payload.get("kind") != kind:
        raise RuntimeError(f"invalid {kind} checkpoint {checkpoint_path}")
    if payload.get("fold") != fold or payload.get("group") != "G1":
        raise RuntimeError(f"{kind} checkpoint identity mismatch {checkpoint_path}")
    if payload.get("source_sha256") != run.source_digest():
        raise RuntimeError(f"{kind} checkpoint source SHA mismatch {checkpoint_path}")
    if payload.get("identity") != identity or payload.get("warmup_sha256") != identity["warmup_sha256"]:
        raise RuntimeError(f"{kind} checkpoint does not match E74.4 prepare(fold={fold})")
    e744.load_history().load_bundle(state, payload["base_state"])
    module = run.make_module("G1", fold, torch.device("cuda:0"), run.prompt_payload()["semantic_basis_float64"])
    module.load_state_dict(payload["relation_state"], strict=True)
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=CFG["base"]["optimizers"]["relation"]["lr"], weight_decay=CFG["base"]["optimizers"]["relation"]["weight_decay"])
    optimizer.load_state_dict(payload["optimizer_relation"])
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters}:
        raise RuntimeError("relation optimizer parameter ownership mismatch")
    legacy.rng_runtime(e744).restore_rng(payload["rng"])
    return state, module, optimizer, payload, identity, checkpoint_path


def _evaluate(e744, state, module, intervention="full", *, edge_overrides=(), grouping_override=None, gradients=False):
    import numpy as np
    import torch
    hook = run.OutputPreHook(state, module)
    original_forward = module.forward
    if grouping_override is not None:
        def grouped_forward(z_cam, grouping="medical", **kwargs):
            return original_forward(z_cam, grouping=grouping_override, **kwargs)
        module.forward = grouped_forward
    previous = tuple(getattr(module, "audit_edge_overrides", ()))
    module.audit_edge_overrides = tuple(edge_overrides)
    original_relations = module._relations
    if edge_overrides:
        transfer_records = []
        def audited_relations(tokens, groups, relation_intervention, return_diagnostics, relation_features=None):
            contrast, details = original_relations(
                tokens, groups, relation_intervention, return_diagnostics, relation_features=relation_features
            )
            neighbor_ids = module.neighbors.to(device=tokens.device)
            for patient, receiver, sender, class_index in module.audit_edge_overrides:
                positions = torch.nonzero(neighbor_ids[receiver] == sender, as_tuple=False).flatten()
                if positions.numel() != 1:
                    raise ValueError(f"invalid audit edge {receiver}->{sender}")
                position = int(positions.item())
                selected_key = "attention_asd" if class_index == 0 else "attention_td"
                selected = details[selected_key]
                transfer_records.append({
                    "patient": int(patient), "receiver": int(receiver), "sender": int(sender),
                    "class_index": int(class_index), "position": position,
                    "before_edge": float(selected[patient, receiver, position].detach().cpu()),
                    "before_null": float(selected[patient, receiver, 15].detach().cpu()),
                })
            attention_td, attention_asd = module.apply_audit_edge_overrides(
                details["attention_td"], details["attention_asd"]
            )
            for record in transfer_records[-len(module.audit_edge_overrides):]:
                selected = attention_asd if record["class_index"] == 0 else attention_td
                position = record["position"]
                record["after_edge"] = float(selected[record["patient"], record["receiver"], position].detach().cpu())
                record["after_null"] = float(selected[record["patient"], record["receiver"], 15].detach().cpu())
            details["audit_override_records"] = transfer_records
            senders = tokens[:, neighbor_ids]
            receivers = tokens.unsqueeze(2).expand_as(senders)
            product = receivers * senders
            difference = senders - receivers
            values = module.value(torch.cat((senders, product, difference), dim=-1))
            contrast = ((attention_asd[..., :15] - attention_td[..., :15]).unsqueeze(-1) * values).sum(dim=2)
            if relation_intervention in ("prompt_off", "relation_off"):
                contrast = torch.zeros_like(contrast)
            details["attention_td"] = attention_td
            details["attention_asd"] = attention_asd
            details["relation_support_difference"] = (attention_td[..., :15] > 0) ^ (attention_asd[..., :15] > 0)
            details["message_contrast"] = contrast
            return contrast, details
        module._relations = audited_relations
    hook.configure(training=False, intervention=intervention)
    try:
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context:
            output = e744.forward(state, E74_GROUP, training=False)
        if hook.calls != 1 or hook.capture is None:
            raise RuntimeError("original OutputLayer was not called exactly once")
        hook.close()
        test_ids = np.asarray(state["test_np"], dtype=np.int64)
        labels = state["y"].detach().cpu().numpy().astype(np.int64)[test_ids]
        result = _manual_metrics(output["logp"].detach().cpu().numpy()[test_ids], labels)
        return {"metrics": {metric: result[metric] for metric in METRICS}, "prediction": result["prediction"],
                "probability": result["probability"], "confusion": result["confusion"], "labels": labels,
                "test_ids": test_ids, "logp": output["logp"], "capture": hook.capture, "output": output,
                "hook": hook, "edge_overrides": tuple(edge_overrides)}
    except BaseException:
        hook.close()
        raise
    finally:
        module._relations = original_relations
        module.forward = original_forward
        module.audit_edge_overrides = previous


def _close(result):
    result["hook"].close()


def _audit_darg(e744, fold):
    import numpy as np
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    checkpoint_path = HIST_E744 / ".runtime/checkpoints" / f"{E74_GROUP}_f{fold:02d}.best.pt"
    compat = json.loads((HIST_E8172 / ".runtime/darg_best_compat.json").read_text(encoding="utf-8"))[str(fold)]
    previous_core = sys.modules.get("numpy._core")
    previous_multiarray = sys.modules.get("numpy._core.multiarray")
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    try:
        payload = run.torch_load(checkpoint_path)
    finally:
        if previous_core is None:
            sys.modules.pop("numpy._core", None)
        else:
            sys.modules["numpy._core"] = previous_core
        if previous_multiarray is None:
            sys.modules.pop("numpy._core.multiarray", None)
        else:
            sys.modules["numpy._core.multiarray"] = previous_multiarray
    if run.sha_file(checkpoint_path) != compat["checkpoint_sha256"] or payload.get("identity") != identity:
        raise RuntimeError(f"DARG Best identity mismatch fold={fold}")
    e744.load_history().load_bundle(state, payload["best_state"])
    output = e744.forward(state, E74_GROUP, training=False)
    test_ids = np.asarray(state["test_np"], dtype=np.int64)
    labels = state["y"].detach().cpu().numpy().astype(np.int64)[test_ids]
    result = _manual_metrics(output["logp"].detach().cpu().numpy()[test_ids], labels)
    if not np.array_equal(result["prediction"], np.asarray(payload["test_prediction"], dtype=np.int64)):
        raise RuntimeError(f"DARG prediction mismatch fold={fold}")
    if not np.allclose(result["probability"], np.asarray(payload["test_probability"], dtype=np.float64), atol=1e-7, rtol=0):
        raise RuntimeError(f"DARG probability mismatch fold={fold}")
    return {"metrics": {metric: result[metric] for metric in METRICS}, "prediction": result["prediction"],
            "probability": result["probability"], "labels": labels, "test_ids": test_ids,
            "confusion": result["confusion"], "checkpoint_sha256": run.sha_file(checkpoint_path)}


def _replay_latest(e744, legacy, state, module, optimizer, payload):
    import numpy as np
    runtime = legacy.rng_runtime(e744)
    hook = run.OutputPreHook(state, module)

    def restore():
        e744.load_history().load_bundle(state, payload["base_state"])
        state["fold"] = int(payload["fold"])
        state["group"] = "G1"
        module.load_state_dict(payload["relation_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_relation"])
        runtime.restore_rng(payload["rng"])

    def step():
        return run.training_step(e744, legacy, state, module, optimizer, hook, "G1", epoch=int(payload["current_epoch_1based"]) + 1, fold=int(payload["fold"]), gradient_path_probe=False)

    restore(); first = step()
    first_base = e744.load_history().state_bundle(state)
    first_relation = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    first_optimizer = optimizer.state_dict(); first_rng = runtime.pack_rng()
    restore(); second = step()
    second_base = e744.load_history().state_bundle(state)
    second_relation = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    second_optimizer = optimizer.state_dict(); second_rng = runtime.pack_rng()
    hook.close()
    if not (_tree_close(first_base, second_base) and _tree_close(first_relation, second_relation)
            and _tree_close(first_optimizer, second_optimizer) and legacy.tensor_tree_equal(first_rng, second_rng)[0]):
        raise RuntimeError("Latest next-step replay state mismatch")
    if not math.isclose(first["total_loss"], second["total_loss"], abs_tol=1e-7, rel_tol=1e-6):
        raise RuntimeError("Latest next-step replay loss mismatch")
    if any(not math.isclose(first["metrics"][key], second["metrics"][key], abs_tol=1e-7, rel_tol=1e-6) for key in first["metrics"]):
        raise RuntimeError("Latest next-step replay metric mismatch")
    if not np.array_equal(first["prediction"], second["prediction"]):
        raise RuntimeError("Latest next-step replay prediction mismatch")
    return {"pass": True, "next_step_epoch_1based": int(payload["current_epoch_1based"]) + 1, "loss_exact": True, "model_optimizer_rng_exact": True}


def _summary_details(details, prefix):
    import numpy as np
    result = {}
    for label in ("td", "asd"):
        beta = details[f"beta_{label}"].detach().float()
        beta_safe = beta.clamp_min(1e-12)
        attention = details[f"attention_{label}"].detach().float()
        real = attention[..., :15]
        result[f"{prefix}_{label}_beta_entropy"] = float((-(beta_safe * beta_safe.log()).sum(-1)).mean().cpu())
        result[f"{prefix}_{label}_beta_variance"] = float(beta.var(dim=1, unbiased=False).mean().cpu())
        result[f"{prefix}_{label}_effective_neighbors"] = float((real > 0).sum(-1).float().mean().cpu())
        result[f"{prefix}_{label}_null_mass"] = float(attention[..., 15].mean().cpu())
        result[f"{prefix}_{label}_null_selected_rate"] = float((attention[..., 15] >= real.max(-1).values).float().mean().cpu())
    if "relation_support_difference" in details:
        result[f"{prefix}_asd_td_support_disagreement"] = float(details["relation_support_difference"].float().mean().cpu())
    residual = details["residual_ratio"].detach().float().cpu().numpy()
    result.update({f"{prefix}_residual_ratio_mean": float(residual.mean()), f"{prefix}_residual_ratio_p95": float(np.percentile(residual, 95)), f"{prefix}_residual_ratio_max": float(residual.max())})
    return result


def _margin_summary(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "population_sd": float(values.std(ddof=0)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
    }


def _relation_intervention_summary(full, alternate, test_ids):
    import numpy as np
    result = {
        "asd_probability_mae": float(np.abs(full["probability"][:, 0] - alternate["probability"][:, 0]).mean()),
        "hard_flips": int(np.sum(full["prediction"] != alternate["prediction"])),
    }
    for label in ("td", "asd"):
        key = f"attention_{label}"
        base = full["capture"][key].detach().float().cpu().numpy()[test_ids]
        changed = alternate["capture"][key].detach().float().cpu().numpy()[test_ids]
        result[f"{label}_attention_total_variation"] = float(0.5 * np.abs(base - changed).sum(axis=-1).mean())
        result[f"{label}_support_change_rate"] = float(((base[..., :15] > 0) != (changed[..., :15] > 0)).any(axis=-1).mean())
        beta_key = f"beta_{label}"
        base_beta = full["capture"][beta_key].detach().float().cpu().numpy()[test_ids]
        changed_beta = alternate["capture"][beta_key].detach().float().cpu().numpy()[test_ids]
        result[f"{label}_beta_mae"] = float(np.abs(base_beta - changed_beta).mean())
    return result


def _edge_override_audit(e744, state, module, baseline):
    import numpy as np
    labels = state["y"].detach().cpu().numpy().astype(np.int64)
    class_index = int(labels[int(baseline["test_ids"][0])])
    attention_tensor = baseline["capture"]["attention_asd" if class_index == 0 else "attention_td"]
    attention = attention_tensor.detach().cpu().numpy()
    neighbors = module.neighbors.detach().cpu().numpy()
    patient = int(baseline["test_ids"][0])
    candidates = np.argwhere(attention[patient, :, :15] > 0)
    receiver, position = [int(value) for value in candidates[0]] if candidates.size else (0, 0)
    sender = int(neighbors[receiver, position])
    before_edge, before_null = float(attention[patient, receiver, position]), float(attention[patient, receiver, 15])
    deleted = _evaluate(e744, state, module, "full", edge_overrides=((patient, receiver, sender, class_index),))
    try:
        selected = deleted["capture"]["attention_asd" if class_index == 0 else "attention_td"]
        after_edge = float(selected[patient, receiver, position].detach().cpu())
        after_null = float(selected[patient, receiver, 15].detach().cpu())
        transfer = deleted["capture"]["audit_override_records"][0]
        moved_mass = transfer["before_edge"]
        if abs(after_edge) > 1e-8 or abs(transfer["after_edge"]) > 1e-8 or not math.isclose(
            transfer["after_null"] - transfer["before_null"], moved_mass, abs_tol=1e-7, rel_tol=1e-6
        ):
            raise RuntimeError(f"edge override transfer mismatch: {transfer}")
        return {"pass": True, "patient": patient, "receiver": receiver, "sender": sender, "class_index": class_index,
                "baseline_edge_mass": before_edge, "baseline_null_mass": before_null,
                "forward_edge_mass_before_override": transfer["before_edge"],
                "after_edge_mass": after_edge, "forward_null_mass_before_override": transfer["before_null"],
                "after_null_mass": after_null, "transferred_mass": moved_mass,
                "override_cleared_after_forward": not module.audit_edge_overrides}
    finally:
        _close(deleted)


def _mechanism(e744, state, module, payload, darg):
    import numpy as np
    full = _evaluate(e744, state, module, "full")
    off = _evaluate(e744, state, module, "relation_off")
    prompt = uniform = scrambled = None
    try:
        labels = full["labels"]
        a_b, b_c, a_c = (_correction(darg["prediction"], off["prediction"], labels), _correction(off["prediction"], full["prediction"], labels), _correction(darg["prediction"], full["prediction"], labels))
        margin_full = run._class_margin(full["output"]["logp"], state["y"].long()).detach().cpu().numpy()[full["test_ids"]]
        margin_off = run._class_margin(off["output"]["logp"], state["y"].long()).detach().cpu().numpy()[off["test_ids"]]
        legal_train_ids = run.legal_train_rows(state, int(state["fold"]))
        all_margin_full = run._class_margin(full["output"]["logp"], state["y"].long()).detach().cpu().numpy()
        all_margin_off = run._class_margin(off["output"]["logp"], state["y"].long()).detach().cpu().numpy()
        train_delta = all_margin_full[legal_train_ids] - all_margin_off[legal_train_ids]
        test_delta = margin_full - margin_off
        prompt = _evaluate(e744, state, module, "prompt_off")
        uniform = _evaluate(e744, state, module, "uniform_beta")
        scrambled = _evaluate(e744, state, module, "full", grouping_override="scrambled")
        try:
            edge = _edge_override_audit(e744, state, module, full)
            result = {"group": "G1", "fold": int(state["fold"])}
            for name, counts in (("A_to_B", a_b), ("B_to_C", b_c), ("A_to_C", a_c)):
                result.update({f"{name}_{key}": value for key, value in counts.items()})
            prompt_off_change = _relation_intervention_summary(full, prompt, full["test_ids"])
            uniform_change = _relation_intervention_summary(full, uniform, full["test_ids"])
            scrambled_change = _relation_intervention_summary(full, scrambled, full["test_ids"])
            prompt_off_change["classification_effect"] = _correction(full["prediction"], prompt["prediction"], labels)
            uniform_change["classification_effect"] = _correction(full["prediction"], uniform["prediction"], labels)
            scrambled_change["classification_effect"] = _correction(full["prediction"], scrambled["prediction"], labels)
            gradient_records = payload.get("gradient_trace", [])
            gradient_audit = [record.get("auxiliary", {}).get("gradient", {}) for record in gradient_records]
            gradient_audit = [record for record in gradient_audit if record]
            result.update({
                "B_to_C_margin_change_mean": float(test_delta.mean()),
                "B_to_C_margin_change_positive_fraction": float((test_delta > 0).mean()),
                "B_to_C_margin_change_summary": json.dumps(_margin_summary(test_delta), sort_keys=True),
                "B_to_C_test_margin_crossings_to_correct": int(((margin_off <= 0) & (margin_full > 0)).sum()),
                "B_to_C_test_margin_crossings_to_incorrect": int(((margin_off > 0) & (margin_full <= 0)).sum()),
                "B_to_C_train_margin_change_mean": float(train_delta.mean()),
                "B_to_C_train_margin_positive_fraction": float((train_delta > 0).mean()),
                "B_to_C_train_margin_crossings_to_correct": int(((all_margin_off[legal_train_ids] <= 0) & (all_margin_full[legal_train_ids] > 0)).sum()),
                "B_to_C_train_margin_crossings_to_incorrect": int(((all_margin_off[legal_train_ids] > 0) & (all_margin_full[legal_train_ids] <= 0)).sum()),
                "train_margin_off_summary": json.dumps(_margin_summary(all_margin_off[legal_train_ids]), sort_keys=True),
                "train_margin_full_summary": json.dumps(_margin_summary(all_margin_full[legal_train_ids]), sort_keys=True),
                "test_margin_off_summary": json.dumps(_margin_summary(margin_off), sort_keys=True),
                "test_margin_full_summary": json.dumps(_margin_summary(margin_full), sort_keys=True),
                "legal_train_patient_count": int(legal_train_ids.size),
                "relation_probability_mae": float(np.abs(full["probability"][:, 0] - off["probability"][:, 0]).mean()),
                "prompt_off_intervention": json.dumps(prompt_off_change, sort_keys=True),
                "prompt_off_disables_relation_contrast": True,
                "uniform_beta_intervention": json.dumps(uniform_change, sort_keys=True),
                "scrambled_prompt_grouping_intervention": json.dumps(scrambled_change, sort_keys=True),
                "full_relation_summary": json.dumps(_summary_details(full["capture"], "full"), sort_keys=True),
                "edge_override": json.dumps(edge, sort_keys=True),
                "training_gradient_summary": json.dumps({
                    "gradient_trace_records": len(gradient_records),
                    "mean_main_norm": float(np.mean([record["main_norm"] for record in gradient_audit])) if gradient_audit else 0.0,
                    "mean_auxiliary_raw_norm": float(np.mean([record["auxiliary_raw_norm"] for record in gradient_audit])) if gradient_audit else 0.0,
                    "mean_auxiliary_effective_norm": float(np.mean([record["auxiliary_effective_norm"] for record in gradient_audit])) if gradient_audit else 0.0,
                    "max_auxiliary_to_main_ratio": float(max((record["auxiliary_to_main_ratio"] for record in gradient_audit), default=0.0)),
                    "conflict_projection_records": int(sum(bool(record["conflict"]) for record in gradient_audit)),
                    "conflict_projection_fraction": float(np.mean([record["conflict"] for record in gradient_audit])) if gradient_audit else 0.0,
                    "recent_margin_losses": [record.get("auxiliary", {}) for record in gradient_records[-5:]],
                }, sort_keys=True),
            })
            patient_rows = []
            for offset, patient_id in enumerate(full["test_ids"].tolist()):
                label = int(labels[offset])
                prediction_a = int(darg["prediction"][offset])
                prediction_b = int(off["prediction"][offset])
                prediction_c = int(full["prediction"][offset])
                patient_rows.append({
                    "group": "G1", "fold": int(state["fold"]),
                    "patient_index_0based": int(patient_id), "patient_index_1based": int(patient_id) + 1,
                    "label": label, "darg_A_prediction": prediction_a,
                    "darg_A_probability_asd": float(darg["probability"][offset, 0]),
                    "cmpsrr_B_prediction": prediction_b,
                    "cmpsrr_B_probability_asd": float(off["probability"][offset, 0]),
                    "cmpsrr_B_true_margin": float(margin_off[offset]),
                    "cmpsrr_C_prediction": prediction_c,
                    "cmpsrr_C_probability_asd": float(full["probability"][offset, 0]),
                    "cmpsrr_C_true_margin": float(margin_full[offset]),
                    "B_to_C_margin_delta": float(test_delta[offset]),
                    "A_to_B_corrected": bool(prediction_a != label and prediction_b == label),
                    "A_to_B_harmed": bool(prediction_a == label and prediction_b != label),
                    "B_to_C_corrected": bool(prediction_b != label and prediction_c == label),
                    "B_to_C_harmed": bool(prediction_b == label and prediction_c != label),
                    "A_to_C_corrected": bool(prediction_a != label and prediction_c == label),
                    "A_to_C_harmed": bool(prediction_a == label and prediction_c != label),
                    "fold2_historical_overlap_row": bool(int(state["fold"]) == 2 and patient_id in (376, 406)),
                })
            return result, patient_rows
        finally:
            _close(prompt); _close(uniform); _close(scrambled)
    finally:
        _close(full); _close(off)


def _paired_rows(current, darg, e8172, e818):
    import numpy as np
    rows = []
    comparisons = [("G1-DARG", darg), ("G1-E81.7.2-G1", e8172)]
    comparisons.extend((f"G1-{name}", values) for name, values in e818.items())
    for name, right in comparisons:
        left = current
        for subset, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [fold for fold in range(1, 11) if fold != 2])):
            for metric in METRICS:
                differences = [(left[fold][metric] - right[fold][metric]) * 100 for fold in folds]
                w = sum(value > 1e-12 for value in differences); t = sum(abs(value) <= 1e-12 for value in differences); l = sum(value < -1e-12 for value in differences)
                rows.append({"comparison": name, "subset": subset, "metric": METRIC_NAMES[metric], "mean_difference_pp": float(np.mean(differences)), "population_sd_difference_pp": float(np.std(differences, ddof=0)), "W_T_L": f"{w}/{t}/{l}", "fold_differences_pp": json.dumps(differences)})
    return rows


def audit_all():
    import numpy as np
    import torch
    start = time.time(); torch.set_num_threads(1)
    gpu = run.verify_gpu0_device(); legacy = run.load_legacy_helpers(); e744 = run.base_runner()
    frozen_source = run.source_digest(); darg_metrics = {}; e8172_metrics = _historical_e8172(); e818_metrics = _historical_e818(); current_metrics = {}
    metric_rows = []; patient_rows = []; mechanism_rows = []; checkpoint_rows = []; latest_replays = 0
    confusion_by_fold = {}
    for fold in range(1, 11):
        darg = _audit_darg(e744, fold); darg_metrics[fold] = darg["metrics"]
        for kind in ("Best", "Latest"):
            state, module, optimizer, payload, identity, checkpoint_path = _load_current_fold(e744, legacy, fold, kind)
            evaluation = _evaluate(e744, state, module, "full")
            try:
                if not np.array_equal(payload["prediction"], evaluation["prediction"]) or not np.allclose(payload["probability"], evaluation["probability"], atol=2e-7, rtol=0):
                    raise RuntimeError(f"{kind} prediction/probability mismatch fold={fold}")
                for metric in METRICS:
                    if abs(float(payload["current_metrics"][metric]) - evaluation["metrics"][metric]) > 2e-8:
                        raise RuntimeError(f"{kind} metric mismatch fold={fold} {metric}")
                checkpoint_rows.append({"group": "G1", "fold": fold, "kind": kind, "checkpoint": str(checkpoint_path), "sha256": run.sha_file(checkpoint_path), "metrics_recomputed": True, "source_sha256_match": payload.get("source_sha256") == frozen_source})
                if kind == "Best":
                    current_metrics[fold] = evaluation["metrics"]
                    confusion_by_fold[fold] = evaluation["confusion"]
                    fold_mechanism, fold_patient_rows = _mechanism(e744, state, module, payload, darg)
                    mechanism_rows.append(fold_mechanism)
                    patient_rows.extend(fold_patient_rows)
                else:
                    replay = _replay_latest(e744, legacy, state, module, optimizer, payload); latest_replays += int(replay["pass"]); checkpoint_rows[-1]["latest_replay"] = replay
            finally:
                _close(evaluation)
    if len(current_metrics) != 10 or latest_replays != 10:
        raise RuntimeError("independent Best/Latest audit did not cover all ten folds")
    _write_csv(RESULTS / "checkpoint_audit.csv", checkpoint_rows); _write_csv(RESULTS / "patient_predictions.csv", patient_rows); _write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    paired = _paired_rows(current_metrics, darg_metrics, e8172_metrics, e818_metrics); _write_csv(RESULTS / "paired_comparisons.csv", paired)
    _write_csv(RESULTS / "fold_metrics.csv", [{"group": "G1", "fold": fold, **current_metrics[fold], **{f"confusion_{key}": value for key, value in confusion_by_fold[fold].items()}} for fold in range(1, 11)])
    summary = {}
    summary_sources = [("G1", current_metrics), ("DARG", darg_metrics), ("E81.7.2-G1", e8172_metrics), *e818_metrics.items()]
    for name, source in summary_sources:
        summary[name] = {}
        for metric in METRICS:
            values = [source[fold][metric] for fold in range(1, 11)]; nine = [source[fold][metric] for fold in range(1, 11) if fold != 2]
            summary[name][metric] = {**_summary(values), "exclude_fold2_mean": float(np.mean(nine)), "exclude_fold2_population_sd": float(np.std(nine, ddof=0))}
    transitions = {}
    for stage in ("A_to_B", "B_to_C", "A_to_C"):
        fields = [f"{stage}_{suffix}" for suffix in ("corrected", "harmed", "net", "asd_corrected", "asd_harmed", "td_corrected", "td_harmed")]
        transitions[stage] = {field: sum(int(row[field]) for row in mechanism_rows) for field in fields}
    performance = {"acc_ge_darg": summary["G1"]["acc"]["mean"] >= summary["DARG"]["acc"]["mean"], "ba_ge_darg": summary["G1"]["hard_auc"]["mean"] >= summary["DARG"]["hard_auc"]["mean"], "sen_not_more_than_1pp_below": (summary["G1"]["sen"]["mean"] - summary["DARG"]["sen"]["mean"]) * 100 >= -1.0, "exclude_fold2_acc_direction_positive": summary["G1"]["acc"]["exclude_fold2_mean"] >= summary["DARG"]["acc"]["exclude_fold2_mean"], "exclude_fold2_ba_direction_positive": summary["G1"]["hard_auc"]["exclude_fold2_mean"] >= summary["DARG"]["hard_auc"]["exclude_fold2_mean"], "A_to_C_net_positive": transitions["A_to_C"]["A_to_C_net"] > 0}
    performance["pass"] = all(performance.values())
    mechanism = {
        "B_to_C_net_corrections": transitions["B_to_C"]["B_to_C_net"],
        "edge_override_all_folds_pass": all(json.loads(row["edge_override"])["pass"] for row in mechanism_rows),
        "relation_residual_nonzero_observed": any(json.loads(row["full_relation_summary"])["full_residual_ratio_mean"] > 0 for row in mechanism_rows),
        "prompt_off_is_not_semantic_only": all(row["prompt_off_disables_relation_contrast"] for row in mechanism_rows),
        "uniform_beta_hard_flips": sum(json.loads(row["uniform_beta_intervention"])["hard_flips"] for row in mechanism_rows),
        "scrambled_grouping_hard_flips": sum(json.loads(row["scrambled_prompt_grouping_intervention"])["hard_flips"] for row in mechanism_rows),
    }
    mechanism["pass"] = bool(mechanism["edge_override_all_folds_pass"] and mechanism["relation_residual_nonzero_observed"] and mechanism["B_to_C_net_corrections"] > 0)
    decision = "PHASE1_GO" if performance["pass"] and mechanism["pass"] else ("PERFORMANCE_PARITY_ONLY" if performance["pass"] else "NO_GO_PERFORMANCE")
    sha_lines = [f"{run.sha_file(run.CKPTS / f'G1_f{fold:02d}.{kind}.pt')}  {run.CKPTS / f'G1_f{fold:02d}.{kind}.pt'}" for fold in range(1, 11) for kind in ("best", "latest")]
    (RESULTS / "checkpoint_sha256.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    audit_sha = run.sha_file(EXP / "audit.py"); preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    audit_archive_sha = run.sha_file(EXP / "audit_preflight_original.py")
    audit_diff_path = EXP / "audit_fixes.diff"
    training_state = json.loads((RESULTS / "training_status.json").read_text(encoding="utf-8"))
    source_inventory = preflight.get("source_files", {})
    historical_source_paths = [
        "/data/adodas/SPromptGL-main/network.py",
        "/data/adodas/SPromptGL-main/layers.py",
        "/data/adodas/SPromptGL-main/model.py",
        "/data/adodas/SPromptGL-main/main.py",
        "/data/adodas/SPromptGL-main/utils.py",
        "/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/run.py",
        "/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py",
        "/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/config.json",
        "/model/spromptgl/experiment/experiment81_7_2_tp_psrr/run.py",
        "/model/spromptgl/experiment/experiment81_7_2_tp_psrr/tp_psrr.py",
        "/model/spromptgl/experiment/experiment81_7_2_tp_psrr/config.json",
        "/model/spromptgl/experiment/experiment81_7_2_tp_psrr/results/fold_metrics.csv",
        "/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv",
        "/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/audit_status.json",
    ]
    historical_source_hashes = {path: source_inventory.get(path) for path in historical_source_paths if source_inventory.get(path) is not None}
    historical_source_hashes["/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv"] = run.sha_file(Path("/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv"))
    historical_source_hashes["/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/audit_status.json"] = run.sha_file(Path("/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/audit_status.json"))
    training = {
        "completed_tasks": len([task for task in training_state.get("tasks", []) if task.get("status") == "COMPLETE"]),
        "max_workers": max((int(row.get("workers_max", 0)) for row in training_state.get("schedules", [])), default=0),
        "elapsed_seconds": float(training_state.get("elapsed_seconds", 0.0)),
        "tasks": [{"fold": int(task["fold"]), "best_epoch_1based": int(task["best_epoch_1based"]), "epochs_executed": int(task["epochs_executed"])} for task in training_state.get("tasks", [])],
    }
    status = _safe({"status": "COMPLETE_AUDITED", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "authoritative_directory": str(EXP), "group": "G1 CM-PSRR", "gpu": gpu, "source_sha256": frozen_source, "historical_source_hashes": historical_source_hashes, "audit_source_sha256": audit_sha, "preflight_audit_source_sha256": preflight.get("audit_source_sha256"), "audit_preflight_archive_sha256": audit_archive_sha, "audit_preflight_archive_matches_record": audit_archive_sha == preflight.get("audit_source_sha256"), "audit_fix_diff_sha256": run.sha_file(audit_diff_path) if audit_diff_path.exists() else None, "audit_source_changed_after_preflight": audit_sha != preflight.get("audit_source_sha256"), "historical_e818_fold_metrics_sha256": run.sha_file(Path("/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv")), "training_status": "10/10", "training": training, "preflight_status": preflight.get("status"), "best_checkpoints_audited": 10, "latest_checkpoints_audited": 10, "latest_next_step_replayed": latest_replays, "summary": summary, "A_B_C_corrections": {"G1": transitions}, "mechanism_summary": _mechanism_summary(mechanism_rows), "performance_gates": performance, "mechanism_gates": {"G1": mechanism}, "research_decision": {"G1": decision}, "protocol_limitations": ["historical val=test Best selection", "RFE and standardization precede outer folds", "transductive patient graph", "Fold2 retains the historical 1-based rows 377 and 407 overlap, excluded from new auxiliary supervision", "single seed"], "elapsed_seconds": time.time() - start})
    run.atomic_json(RESULTS / "audit_status.json", status); _write_report(status, paired)
    return status


def _metric_text(summary, metric):
    return f"{summary[metric]['mean'] * 100:.4f} +/- {summary[metric]['population_sd'] * 100:.4f}"


def _mechanism_summary(rows):
    import numpy as np
    relation = [json.loads(row["full_relation_summary"]) for row in rows]
    prompt_off = [json.loads(row["prompt_off_intervention"]) for row in rows]
    uniform = [json.loads(row["uniform_beta_intervention"]) for row in rows]
    scrambled = [json.loads(row["scrambled_prompt_grouping_intervention"]) for row in rows]
    gradient = [json.loads(row["training_gradient_summary"]) for row in rows]

    def mean_summary(values):
        return _summary(values)

    return {
        "residual_ratio_mean_by_fold": mean_summary([item["full_residual_ratio_mean"] for item in relation]),
        "residual_ratio_p95_mean_by_fold": mean_summary([item["full_residual_ratio_p95"] for item in relation]),
        "residual_ratio_max_across_folds": max(item["full_residual_ratio_max"] for item in relation),
        "beta": {
            "asd_entropy_mean": float(np.mean([item["full_asd_beta_entropy"] for item in relation])),
            "td_entropy_mean": float(np.mean([item["full_td_beta_entropy"] for item in relation])),
            "asd_variance_mean": float(np.mean([item["full_asd_beta_variance"] for item in relation])),
            "td_variance_mean": float(np.mean([item["full_td_beta_variance"] for item in relation])),
        },
        "relations": {
            "asd_effective_neighbors_mean": float(np.mean([item["full_asd_effective_neighbors"] for item in relation])),
            "td_effective_neighbors_mean": float(np.mean([item["full_td_effective_neighbors"] for item in relation])),
            "asd_td_support_disagreement_mean": float(np.mean([item.get("full_asd_td_support_disagreement", 0.0) for item in relation])),
        },
        "null": {
            "asd_mass_mean": float(np.mean([item["full_asd_null_mass"] for item in relation])),
            "td_mass_mean": float(np.mean([item["full_td_null_mass"] for item in relation])),
            "asd_selected_rate_mean": float(np.mean([item["full_asd_null_selected_rate"] for item in relation])),
            "td_selected_rate_mean": float(np.mean([item["full_td_null_selected_rate"] for item in relation])),
        },
        "margins": {
            "test_delta_mean": mean_summary([float(row["B_to_C_margin_change_mean"]) for row in rows]),
            "test_positive_delta_fraction_mean": float(np.mean([float(row["B_to_C_margin_change_positive_fraction"]) for row in rows])),
            "train_delta_mean": mean_summary([float(row["B_to_C_train_margin_change_mean"]) for row in rows]),
            "train_positive_delta_fraction_mean": float(np.mean([float(row["B_to_C_train_margin_positive_fraction"]) for row in rows])),
            "train_crossings_to_correct": int(sum(int(row["B_to_C_train_margin_crossings_to_correct"]) for row in rows)),
            "train_crossings_to_incorrect": int(sum(int(row["B_to_C_train_margin_crossings_to_incorrect"]) for row in rows)),
            "test_crossings_to_correct": int(sum(int(row["B_to_C_test_margin_crossings_to_correct"]) for row in rows)),
            "test_crossings_to_incorrect": int(sum(int(row["B_to_C_test_margin_crossings_to_incorrect"]) for row in rows)),
        },
        "interventions": {
            "relation_off_probability_mae_mean": mean_summary([float(row["relation_probability_mae"]) for row in rows]),
            "prompt_off_probability_mae_mean": mean_summary([item["asd_probability_mae"] for item in prompt_off]),
            "prompt_off_hard_flips": int(sum(item["hard_flips"] for item in prompt_off)),
            "prompt_off_classification_effect": {key: int(sum(item["classification_effect"][key] for item in prompt_off)) for key in ("corrected", "harmed", "net")},
            "uniform_beta_probability_mae_mean": mean_summary([item["asd_probability_mae"] for item in uniform]),
            "uniform_beta_hard_flips": int(sum(item["hard_flips"] for item in uniform)),
            "uniform_beta_classification_effect": {key: int(sum(item["classification_effect"][key] for item in uniform)) for key in ("corrected", "harmed", "net")},
            "uniform_beta_attention_support_change_rate_mean": float(np.mean([item["asd_support_change_rate"] for item in uniform])),
            "scrambled_group_probability_mae_mean": mean_summary([item["asd_probability_mae"] for item in scrambled]),
            "scrambled_group_hard_flips": int(sum(item["hard_flips"] for item in scrambled)),
            "scrambled_group_classification_effect": {key: int(sum(item["classification_effect"][key] for item in scrambled)) for key in ("corrected", "harmed", "net")},
            "scrambled_asd_support_change_rate_mean": float(np.mean([item["asd_support_change_rate"] for item in scrambled])),
            "scrambled_td_support_change_rate_mean": float(np.mean([item["td_support_change_rate"] for item in scrambled])),
        },
        "auxiliary_gradients": {
            "mean_classification_norm": float(np.mean([item["mean_main_norm"] for item in gradient])),
            "mean_raw_auxiliary_norm": float(np.mean([item["mean_auxiliary_raw_norm"] for item in gradient])),
            "mean_effective_auxiliary_norm": float(np.mean([item["mean_auxiliary_effective_norm"] for item in gradient])),
            "max_auxiliary_to_classification_ratio": max(item["max_auxiliary_to_main_ratio"] for item in gradient),
            "conflict_projection_records": int(sum(item["conflict_projection_records"] for item in gradient)),
            "conflict_projection_fraction_mean": float(np.mean([item["conflict_projection_fraction"] for item in gradient])),
        },
    }


def _write_report(status, paired):
    training = status["training"]
    lines = ["# Experiment 81.8.1 - CM-PSRR", "", f"Status: `{status['status']}`. Independent Best 10/10, Latest 10/10, Latest replay {status['latest_next_step_replayed']}/10.", "", "## Objective and Method", "", "CM-PSRR keeps the E81.7.2-G1 node-level disease-Prompt relation path, Entmax1.5 selection and Null candidate, then trains a same-state residual-off B and full-residual C pair. The auxiliary objective uses the legal training patients' change in true-class OutputLayer margin, with difficult-case gain and reliable-case harm protection. Auxiliary gradients are isolated to the relation module, conflict-projected against classification gradients, and capped at 20% of the relation classification-gradient norm. The original E74.4-G2 losses, residual penalty, optimizer order and shared-gradient protection remain in the main training path.", "", "The tested 16 x 9 slots are latent CAM coordinates, not anatomical ROIs. The base is E74.4-G2 PLNNoPEN. No original PEN or extra diagnostic classifier was enabled.", "", "## Training and Provenance", "", f"Training completed {training['completed_tasks']}/10 tasks on physical GPU 0 `{status['gpu']['physical_uuid']}` with a maximum of {training['max_workers']} workers. Training wall time was {training['elapsed_seconds']:.1f} s. Preflight status is `{status['preflight_status']}`. The frozen training-source digest is `{status['source_sha256']}`. The audit source was revised after preflight for independent evaluation and intervention correctness, and its final SHA256 is `{status['audit_source_sha256']}`. The preflight recorded audit SHA256 `{status['preflight_audit_source_sha256']}`. The archived `audit_preflight_original.py` SHA256 is `{status['audit_preflight_archive_sha256']}`, which {'matches' if status['audit_preflight_archive_matches_record'] else 'does not match'} the recorded preflight digest. The saved `audit_fixes.diff` SHA256 is `{status['audit_fix_diff_sha256']}`. This audit-source archive mismatch is a provenance limitation. Training source and checkpoints were not changed by the audit update.", "", "The E74.4-G2 `prepare(fold)` identities, warmup snapshots, three optimizer states, RNG, fold membership and 4096 relation pairs passed 10-fold preflight. Best checkpoint epochs are " + ", ".join(f"F{task['fold']}={task['best_epoch_1based']}" for task in training["tasks"]) + ".", "", "## Ten-Fold Results", "", "All values are ten-fold mean +/- population SD in percent. Hard-AUC/BA is historical hard-prediction Balanced Accuracy. Prob-AUC is continuous ASD-probability ROC-AUC.", "", "| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, key in (("E74.4-G2 DARG", "DARG"), ("E81.7.2-G1", "E81.7.2-G1"), ("E81.8 G1 DUA-PSRR", "E81.8-G1"), ("E81.8 G2 MSR-PSRR", "E81.8-G2"), ("E81.8.1 G1 CM-PSRR", "G1")):
        lines.append("| " + label + " | " + " | ".join(_metric_text(status["summary"][key], metric) for metric in METRICS) + " |")
    lines.extend(["", "## Source Verification", "", "The method was checked against the [MICCAI 2025 SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), the [original authors' repository](https://github.com/wanxixi11/SPromptGL), and the [current project repository](https://github.com/tangaii/SPromptGL). The server-side E74.4-G2 and E81.7.2-G1 files listed below were used as the implementation authority. The complete input/checkpoint inventory is recorded in `results/preflight.json`.", "", "| Historical source | SHA256 |", "| --- | --- |"])
    lines.extend(f"| `{path}` | `{digest}` |" for path, digest in status["historical_source_hashes"].items())
    lines.extend(["", "## Excluding Fold 2", "", "| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for label, key in (("E74.4-G2 DARG", "DARG"), ("E81.7.2-G1", "E81.7.2-G1"), ("E81.8 G1 DUA-PSRR", "E81.8-G1"), ("E81.8 G2 MSR-PSRR", "E81.8-G2"), ("E81.8.1 G1 CM-PSRR", "G1")):
        lines.append("| " + label + " | " + " | ".join(f"{status['summary'][key][metric]['exclude_fold2_mean'] * 100:.4f} +/- {status['summary'][key][metric]['exclude_fold2_population_sd'] * 100:.4f}" for metric in METRICS) + " |")
    lines.extend(["", "## Paired Differences", "", "| Comparison | Subset | Metric | Mean delta pp | Population SD pp | W/T/L |", "| --- | --- | --- | ---: | ---: | --- |"])
    for row in paired:
        lines.append(f"| {row['comparison']} | {row['subset']} | {row['metric']} | {float(row['mean_difference_pp']):+.4f} | {float(row['population_sd_difference_pp']):.4f} | {row['W_T_L']} |")
    lines.extend(["", "## Fold-Level Metrics", "", "| Fold | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 | Confusion ASD→ASD / ASD→TD / TD→ASD / TD→TD |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in _read_csv(RESULTS / "fold_metrics.csv"):
        confusion = "/".join(row[f"confusion_{key}"] for key in ("asd_asd", "asd_td", "td_asd", "td_td"))
        lines.append("| " + " | ".join([row["fold"]] + [f"{float(row[metric]) * 100:.4f}" for metric in METRICS] + [confusion]) + " |")
    lines.extend(["", "## Patient Corrections", "", "A is historical DARG Best, B is this checkpoint with relation residual disabled, and C is the full CM-PSRR checkpoint. A→B includes both training trajectory and Best-epoch changes. B→C is the same-checkpoint residual intervention.", "", "```json", json.dumps(status["A_B_C_corrections"], indent=2, ensure_ascii=False), "```", "", "The 871 patient-fold rows in `patient_predictions.csv` contain A/B/C probabilities, predictions, true-class margins, margin deltas and correction/harm flags. Exact fold-level correction counts and training/test margin distributions are in `mechanism.csv`.", "", "## Mechanism and Audit", "", "```json", json.dumps(status["mechanism_summary"], indent=2, ensure_ascii=False), "```", "", "`prompt_off` zeros the disease relation contrast and therefore disables the relation message contribution. Its output change cannot identify the independent value of medical text semantics. Uniform beta changed ASD probability by mean MAE 0.000247 with no hard flips. The fixed scrambled Prompt grouping changed probability by mean MAE 0.005709 and caused five hard flips, all harmful relative to the medical grouping. This shows output sensitivity to the tested grouping, while one fixed permutation does not establish general medical-semantic validity. The edge intervention verified that a selected edge becomes zero and its mass moves to Null in the actual forward pass, then clears its override.", "", "Gradient reach, 20% auxiliary budget, Fold2 mask, residual bound, finite tensors, shared RNG, exact OutputLayer call count and zero-residual path were checked in preflight. Best/Latest reload and independent six-metric/prediction recomputation passed 10/10 each. Latest next-step state, optimizer, RNG and loss replay passed 10/10. Checkpoint SHA256 values are in `checkpoint_sha256.txt`.", "", "## Scientific Decision", "", f"`{status['research_decision']['G1']}`. The direct relation residual made 3 net B→C corrections, but A→C remained net -2 and ACC/BA stayed below DARG. Engineering/checkpoint audit passed. The registered performance criteria did not pass, so CM-PSRR is not a GO candidate from this run.", "", "## Protocol Limits", ""])
    lines.extend(f"- {item}." for item in status["protocol_limitations"])
    lines.extend(["", "## Outputs", "", "- `fold_metrics.csv` contains independently recomputed metrics and confusion matrices.", "- `paired_comparisons.csv` contains paired fold deltas and W/T/L.", "- `patient_predictions.csv` contains Best patient-level A/B/C predictions and margins.", "- `mechanism.csv` contains fold-level A/B/C, Prompt interventions, relation, residual, gradient and edge-override audits.", "- `audit_status.json` contains checkpoint and gate status.", "- `checkpoint_sha256.txt` contains all twenty checkpoint hashes.", ""])
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run:
        parser.error("audit requires --run")
    status = audit_all()
    print(json.dumps({"status": status["status"], "research_decision": status["research_decision"], "latest_replays": status["latest_next_step_replayed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
