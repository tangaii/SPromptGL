"""Recompute the registered G4 mechanism measures from Best checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_e837 as runner
from fold_checkpoint import write_json


HISTORICAL_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
NEW_GROUPS = ("T1", "T2", "T4", "T6")


def load_base_h4(fold, payload, device):
    model = runner.FidelityModel(
        runner.historical.DIMS, runner.BASE_GROUPS["H4"], 3, str(runner.OLD_PROMPT_SCMR),
        payload.get("spectral_stats"), 824602 * 100 + fold, fold=fold, device=device,
    ).to(device)
    model.darg.enter_positive_dcr()
    return model


def message_relative(aux):
    messages = aux.get("messages")
    if messages is None:
        return []
    raw_norm = aux["raw"].norm(dim=1).clamp_min(1e-12)
    return [float((message.norm(dim=1) / raw_norm).mean()) for message in messages]


def margin_mean(logp, test):
    values = logp.detach().cpu().topk(2, dim=1).values
    return float((values[:, 0] - values[:, 1])[test].mean())


def class_correct(y_true, prediction):
    return [int(((y_true == class_id) & (prediction == y_true)).sum()) for class_id in range(3)]


def fold_record(group, fold, x, y, y_np, test_np, payload, audit, device):
    if group == "G4":
        model = load_base_h4(fold, payload, device)
    else:
        config = runner.group_config(group)
        model = runner.build_model(config, fold, device, spectral_stats=payload.get("spectral_stats"))
        model.darg.enter_positive_dcr()
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    train = torch.as_tensor(payload["train_idx"], device=device).long()
    test = torch.as_tensor(payload["test_idx"], device=device).long()
    pair, same = payload.get("relation_pairs"), payload.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    with torch.no_grad():
        full, full_aux = model(x, y, train, pair, same, dpsl_enabled=True)
        native, _ = model(x, y, train, pair, same, dpsl_enabled=False)
        direct_off, _ = model(x, y, train, pair, same, dpsl_enabled=True,
                              darg_intervention="direct_off")
        neutral, _ = model(x, y, train, pair, same, dpsl_enabled=True,
                           darg_intervention="disease_neutral")
        residual_off, _ = model(x, y, train, pair, same, dpsl_enabled=True,
                                dpsl_intervention="residual_off")
    full_pred = full.argmax(1).cpu().numpy()
    native_pred = native.argmax(1).cpu().numpy()
    direct_pred = direct_off.argmax(1).cpu().numpy()
    neutral_pred = neutral.argmax(1).cpu().numpy()
    truth = y_np[test_np]

    def intervention(changed):
        full_correct = full_pred[test_np] == truth
        changed_correct = changed[test_np] == truth
        return {"changed": int((full_pred[test_np] != changed[test_np]).sum()),
                "corrected": int((~full_correct & changed_correct).sum()),
                "damaged": int((full_correct & ~changed_correct).sum())}

    native_correct = native_pred[test_np] == truth
    full_correct = full_pred[test_np] == truth
    result = {
        "group": group, "fold": fold,
        "native_full_logit_max_abs": float((native - full).abs()[test].max()),
        "native_full_flip_count": int((native_pred[test_np] != full_pred[test_np]).sum()),
        "native_full_corrected": int((~native_correct & full_correct).sum()),
        "native_full_damaged": int((native_correct & ~full_correct).sum()),
        "native_top1_top2_margin_mean": margin_mean(native, test_np),
        "full_top1_top2_margin_mean": margin_mean(full, test_np),
        "direct_off": intervention(direct_pred), "disease_neutral": intervention(neutral_pred),
        "residual_off_native_max_abs": float((residual_off - native).abs().max()),
        "dpsl_residual_max": float(full_aux.get("dpsl", {}).get(
            "residual_ratio", torch.zeros(1, device=device)).max()),
        "graph_message_relative_norm_mean_by_modality": message_relative(full_aux),
        "native_class_correct": class_correct(truth, native_pred[test_np]),
        "full_class_correct": class_correct(truth, full_pred[test_np]),
        "direct_off_class_correct": class_correct(truth, direct_pred[test_np]),
        "disease_neutral_class_correct": class_correct(truth, neutral_pred[test_np]),
        "dpsl_shared_cap_configured": (None if group == "G4" else audit.get("dpsl_shared_cap_configured")),
        "dpsl_shared_cap_observed_max": (None if group == "G4" else audit.get("dpsl_shared_cap_observed_max")),
        "dpsl_shared_gradient_ratio_max": (None if group == "G4" else audit.get("dpsl_shared_gradient_ratio_max")),
        "weight_decay_schedule": audit.get("weight_decay_schedule"),
    }
    del model
    torch.cuda.empty_cache()
    return result


def summarize(rows):
    numeric = ("native_full_logit_max_abs", "native_full_flip_count", "native_full_corrected",
               "native_full_damaged", "native_top1_top2_margin_mean", "full_top1_top2_margin_mean",
               "residual_off_native_max_abs", "dpsl_residual_max")
    result = {key: float(np.mean([row[key] for row in rows])) for key in numeric}
    for name in ("direct_off", "disease_neutral"):
        result[name] = {key: int(sum(row[name][key] for row in rows))
                        for key in ("changed", "corrected", "damaged")}
    for name in ("native_class_correct", "full_class_correct", "direct_off_class_correct",
                 "disease_neutral_class_correct"):
        result[name] = [int(sum(row[name][index] for row in rows)) for index in range(3)]
    for name in ("dpsl_shared_cap_configured", "dpsl_shared_cap_observed_max",
                 "dpsl_shared_gradient_ratio_max"):
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = float(np.mean(values)) if values else None
    result["graph_message_relative_norm_mean_by_modality"] = [
        float(np.mean([row["graph_message_relative_norm_mean_by_modality"][index] for row in rows]))
        for index in range(6) if all(len(row["graph_message_relative_norm_mean_by_modality"]) > index for row in rows)
    ]
    return result


def warmup_summary(group):
    rows = []
    for fold in range(1, 11):
        path = runner.ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
        audit = json.loads(path.read_text())
        history = audit["warmup"]["history"]
        values = [float(item.get("graph_supervision", 0.0)) for item in history]
        rows.append({"fold": fold, "graph_supervision_first": values[0] if values else 0.0,
                     "graph_supervision_last": values[-1] if values else 0.0,
                     "graph_supervision_nonzero_epochs": sum(value > 0 for value in values),
                     "formal_graph_supervision_enabled": audit.get("formal_graph_supervision_enabled"),
                     "graph_supervision_final": audit.get("graph_supervision_final"),
                     "weight_decay_schedule": audit.get("weight_decay_schedule")})
    return {"folds": rows, "first_mean": float(np.mean([row["graph_supervision_first"] for row in rows])),
            "last_mean": float(np.mean([row["graph_supervision_last"] for row in rows])),
            "formal_final_mean": float(np.mean([row["graph_supervision_final"] for row in rows]))}


def main():
    runner.configure_worker()
    x_np, y_np, _, splits = runner.load_data()
    device = torch.device("cuda:0")
    x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).long().to(device)
    groups = {"G4": [], **{group: [] for group in NEW_GROUPS}}
    for fold in range(1, 11):
        split = next(item for item in splits if int(item["fold"]) == fold)
        test_np = np.asarray(split["test_idx"], dtype=np.int64)
        hist_dir = HISTORICAL_ROOT / "G4_E82_4_6_10_G1"
        hist_path = hist_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
        hist_audit = json.loads((hist_dir / "audits" / f"fold{fold:02d}.json").read_text())
        groups["G4"].append(fold_record("G4", fold, x, y, y_np, test_np,
                                         torch.load(hist_path, map_location="cpu", weights_only=False),
                                         hist_audit, device))
        for group in NEW_GROUPS:
            path = runner.ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
            audit = json.loads((runner.ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json").read_text())
            groups[group].append(fold_record(
                group, fold, x, y, y_np, test_np,
                torch.load(path, map_location="cpu", weights_only=False), audit, device))
        print(f"MECHANISM fold {fold:02d} PASS", flush=True)
    value = {"experiment": "Experiment 83.7", "status": "PASS",
             "groups": {group: {"folds": rows, "mean": summarize(rows)}
                        for group, rows in groups.items()},
             "graph_supervision": {group: warmup_summary(group) for group in ("T3", "T4")},
             "weight_decay": {group: warmup_summary(group) for group in ("T5", "T6")}}
    write_json(runner.ROOT / "audits/mechanism_summary.json", value)
    print(json.dumps({"status": value["status"], "groups": list(groups)}, indent=2))


if __name__ == "__main__":
    main()
