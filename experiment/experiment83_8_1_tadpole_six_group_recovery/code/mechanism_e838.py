"""Measure learned graph, relation, uncertainty, and DPSL mechanisms."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_e838 as runner


HISTORY_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
BASE_FOR_NEW = {"X1": "H1", "X2": "H2", "X3": "H3", "X4": "H4", "X5": "H4", "X6": "H4"}
BASE_GROUPS = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1",
}
HISTORY_PATHS = {group: HISTORY_ROOT / name for group, name in (
    ("H1", "G1_E28_E0"), ("H2", "G2_E43_1_E00"),
    ("H3", "G3_E74_4_G2"), ("H4", "G4_E82_4_6_10_G1"),
)}


def load_checkpoint(group, fold):
    if group.startswith("H"):
        path = HISTORY_PATHS[group] / "best_checkpoints" / f"fold{fold:02d}.pt"
        config = {"base_reference": group, "class_prefix_prompts": False}
    else:
        path = runner.ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
        config = runner.group_config(group)
    return path, torch.load(path, map_location="cpu", weights_only=False), config


def prepare_model(group, fold, device):
    path, payload, config = load_checkpoint(group, fold)
    model = runner.build_model(config, fold, device, spectral_stats=payload.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return path, payload, model


def relation_summary(aux, train_rows):
    deltas = aux.get("delta_adjacencies", [])
    if not deltas:
        return None
    values = []
    for delta in deltas:
        selected = delta.index_select(0, train_rows).index_select(1, train_rows)
        values.append({"positive_edges": int((selected > 0).sum()),
                       "negative_edges": int((selected < 0).sum()),
                       "signed_l1": float(selected.abs().sum())})
    pair_logits = aux.get("pair_logits", [])
    pair_summary = [{"mean": float(logit.mean()), "std_population": float(logit.std(unbiased=False))}
                    for logit in pair_logits]
    return {"per_modality": values, "pair_logit_summary": pair_summary,
            "pair_loss": float(aux.get("pair_loss", torch.zeros(())).detach()),
            "adv_loss": float(aux.get("adv_loss", torch.zeros(())).detach()),
            "preserve_loss": float(aux.get("preserve_loss", torch.zeros(())).detach())}


def class_correct(truth, prediction):
    return [int(((truth == index) & (prediction == truth)).sum()) for index in range(3)]


def correction_counts(before, after, truth):
    before_correct = before == truth
    after_correct = after == truth
    return {"changed": int((before != after).sum()),
            "corrected": int((~before_correct & after_correct).sum()),
            "damaged": int((before_correct & ~after_correct).sum()),
            "net": int(after_correct.sum() - before_correct.sum())}


def graph_and_prediction_record(group, baseline, fold, x, y, y_np, ids, split, device,
                                uncertainty_rows):
    path, payload, model = prepare_model(group, fold, device)
    _, baseline_payload, baseline_model = prepare_model(baseline, fold, device)
    train = torch.as_tensor(split["train_idx"], device=device).long()
    test = torch.as_tensor(split["test_idx"], device=device).long()
    pair, same = payload.get("relation_pairs"), payload.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    base_pair, base_same = baseline_payload.get("relation_pairs"), baseline_payload.get("relation_targets")
    if base_pair is not None:
        base_pair, base_same = base_pair.to(device), base_same.to(device)

    with torch.no_grad():
        full, aux = model(x, y, train, pair, same, dpsl_enabled=True)
        native = model(x, y, train, pair, same, dpsl_enabled=False)[0] if model.dpsl is not None else full
        base_full, base_aux = baseline_model(x, y, train, base_pair, base_same, dpsl_enabled=True)
        base_native = (baseline_model(x, y, train, base_pair, base_same, dpsl_enabled=False)[0]
                       if baseline_model.dpsl is not None else base_full)

    graph_change = []
    for modal, (adjacency, base_adjacency) in enumerate(
            zip(aux["adjacencies"], base_aux["adjacencies"], strict=True)):
        graph_change.append({
            "modality": modal,
            "baseline_nonzero_edges": int(torch.count_nonzero(base_adjacency)),
            "candidate_nonzero_edges": int(torch.count_nonzero(adjacency)),
            "changed_entries": int((adjacency != base_adjacency).sum()),
            "raw_mean_absolute_difference": float((adjacency - base_adjacency).abs().mean()),
            "raw_max_absolute_difference": float((adjacency - base_adjacency).abs().max()),
        })

    aloss_record = None
    if group in ("X1", "X2", "X3", "X4"):
        with torch.enable_grad():
            _, grad_aux = model(x, y, train, pair, same, dpsl_enabled=False)
            aloss, details = runner.paper_aloss(grad_aux, y, train, model)
            grads = torch.autograd.grad(aloss, [graph.p.weight for graph in model.gfr.graphs],
                                        allow_unused=True)
        grad_norms = [0.0 if value is None else float(value.norm()) for value in grads]
        if len(details["per_modality_loss"]) != 6 or len(grad_norms) != 6:
            raise RuntimeError(f"{group} fold {fold} did not expose six Paper-ALoss modalities")
        aloss_record = {"loss": float(aloss.detach()),
                        "per_modality_loss": details["per_modality_loss"],
                        "per_modality_graphlearn_p_weight_gradient_norm": grad_norms,
                        "largest_gradient_modality": int(np.argmax(grad_norms)) + 1,
                        "largest_gradient_norm": float(max(grad_norms))}
        del grad_aux

    candidate_pred = full[test].argmax(1).cpu().numpy().astype(np.int64)
    native_pred = native[test].argmax(1).cpu().numpy().astype(np.int64)
    base_pred = base_full[test].argmax(1).cpu().numpy().astype(np.int64)
    base_native_pred = base_native[test].argmax(1).cpu().numpy().astype(np.int64)
    truth = y_np[test.cpu().numpy()]
    patient_rows = []
    for local, row in enumerate(test.cpu().numpy()):
        patient_rows.append({
            "group": group, "baseline": baseline, "fold": fold, "sample_index": int(row),
            "sample_id": ids.iloc[int(row)].RID, "true_label": int(truth[local]),
            "native_pred_label": int(native_pred[local]), "full_pred_label": int(candidate_pred[local]),
            "baseline_native_pred_label": int(base_native_pred[local]),
            "baseline_full_pred_label": int(base_pred[local]),
        })
    relation = relation_summary(aux, train) if model.use_darg else None
    base_relation = relation_summary(base_aux, train) if baseline_model.use_darg else None
    record = {
        "group": group, "baseline": baseline, "fold": fold,
        "checkpoint": str(path), "native_correct_by_class": class_correct(truth, native_pred),
        "full_correct_by_class": class_correct(truth, candidate_pred),
        "native_to_full_corrections": correction_counts(native_pred, candidate_pred, truth),
        "baseline_full_to_candidate_full": correction_counts(base_pred, candidate_pred, truth),
        "baseline_native_to_candidate_native": correction_counts(base_native_pred, native_pred, truth),
        "graph_change_vs_base": graph_change, "paper_aloss": aloss_record,
        "darg_relation": relation, "baseline_darg_relation": base_relation,
        "dpsl_residual_max": float(aux.get("dpsl", {}).get(
            "residual_ratio", torch.zeros(1, device=device)).max()),
    }

    if group in ("X5", "X6"):
        uncertainty = aux["native_logits"].softmax(dim=-1)
        entropy = -(uncertainty.clamp_min(1e-12) * uncertainty.clamp_min(1e-12).log()).sum(-1)
        entropy = (entropy / np.log(3.0)).clamp(0.0, 1.0).detach()
        local_labels = y.index_select(0, train).detach()
        groups = runner._pair_groups(local_labels)
        scores = aux["continuous_scores"]
        weighted = group == "X6"
        pair_details = runner.pair_mean_graph_supervision(
            scores, y, train, native_logits=aux["native_logits"] if weighted else None)[1]
        group_weights = {}
        for name, left, right, _ in groups:
            if weighted:
                weights = 0.5 + 0.5 * (entropy[train][left] + entropy[train][right]) / 2.0
                endpoint_entropy = (entropy[train][left] + entropy[train][right]) / 2.0
            else:
                weights = torch.ones_like(left, dtype=torch.float32)
                endpoint_entropy = None
            group_weights[name] = {
                "pair_count": int(left.numel()), "mean_weight": float(weights.mean()),
                "median_weight": float(weights.median()),
                "p90_weight": float(torch.quantile(weights, 0.9)),
                "mean_endpoint_entropy": (float(endpoint_entropy.mean()) if weighted else None),
                "mean_loss": pair_details["per_group_loss"][name],
                "equal_group_loss_share": 1.0 / 6.0,
            }
            if not weighted:
                continue
            local_rows = train.cpu().numpy()
            weights_cpu = weights.cpu().numpy()
            left_cpu, right_cpu = left.cpu().numpy(), right.cpu().numpy()
            endpoint_cpu = endpoint_entropy.cpu().numpy()
            for index in np.argsort(weights_cpu)[-5:][::-1]:
                first, second = int(local_rows[left_cpu[index]]), int(local_rows[right_cpu[index]])
                uncertainty_rows.append({
                    "group": group, "fold": fold, "pair_type": name,
                    "sample_index_i": first, "sample_id_i": ids.iloc[first].RID,
                    "label_i": int(y_np[first]), "normalized_entropy_i": float(entropy[first]),
                    "sample_index_j": second, "sample_id_j": ids.iloc[second].RID,
                    "label_j": int(y_np[second]), "normalized_entropy_j": float(entropy[second]),
                    "mean_endpoint_entropy": float(endpoint_cpu[index]),
                    "pair_weight": float(weights_cpu[index]),
                })
        record["pair_mean_mechanism"] = {
            "group_counts": pair_details["group_counts"],
            "group_weights": group_weights,
            "weight_range": [float(min(item["mean_weight"] for item in group_weights.values())),
                             float(max(item["mean_weight"] for item in group_weights.values()))],
            "weights_detached": weighted,
        }

    del model, baseline_model
    torch.cuda.empty_cache()
    return record, patient_rows


def aggregate(records):
    result = {}
    for group in sorted(set(row["group"] for row in records)):
        selected = [row for row in records if row["group"] == group]
        value = {"folds": len(selected)}
        for field in ("paper_aloss", "pair_mean_mechanism"):
            values = [row[field] for row in selected if row.get(field) is not None]
            if values:
                if field == "paper_aloss":
                    value[field] = {
                        "mean_loss": float(np.mean([item["loss"] for item in values])),
                        "per_modality_loss_mean": np.mean(
                            [item["per_modality_loss"] for item in values], axis=0).tolist(),
                        "per_modality_gradient_norm_mean": np.mean(
                            [item["per_modality_graphlearn_p_weight_gradient_norm"] for item in values], axis=0).tolist(),
                        "largest_gradient_modality_counts": {
                            str(index + 1): sum(item["largest_gradient_modality"] == index + 1 for item in values)
                            for index in range(6)},
                    }
                else:
                    names = values[0]["group_weights"]
                    value[field] = {name: {
                        metric: (None if values[0]["group_weights"][name][metric] is None else
                                 float(np.mean([item["group_weights"][name][metric] for item in values])))
                        for metric in ("pair_count", "mean_weight", "median_weight", "p90_weight",
                                       "mean_endpoint_entropy", "mean_loss", "equal_group_loss_share")
                    } for name in names}
        value["graph_change_vs_base"] = [{
            key: (int(np.sum([row["graph_change_vs_base"][modal][key] for row in selected]))
                  if key in ("changed_entries", "baseline_nonzero_edges", "candidate_nonzero_edges")
                  else float(np.mean([row["graph_change_vs_base"][modal][key] for row in selected])))
            for key in ("baseline_nonzero_edges", "candidate_nonzero_edges", "changed_entries",
                        "raw_mean_absolute_difference", "raw_max_absolute_difference")
        } | {"modality": modal + 1} for modal in range(6)]
        for field in ("native_to_full_corrections", "baseline_full_to_candidate_full",
                      "baseline_native_to_candidate_native"):
            value[field] = {key: int(sum(row[field][key] for row in selected))
                            for key in ("changed", "corrected", "damaged", "net")}
        value["native_correct_by_class"] = [int(sum(row["native_correct_by_class"][i] for row in selected))
                                            for i in range(3)]
        value["full_correct_by_class"] = [int(sum(row["full_correct_by_class"][i] for row in selected))
                                          for i in range(3)]
        value["dpsl_residual_max"] = max(row["dpsl_residual_max"] for row in selected)
        if selected[0]["darg_relation"] is not None:
            relation = [row["darg_relation"] for row in selected]
            base_relation = [row["baseline_darg_relation"] for row in selected]
            value["darg_relation"] = {
                "candidate_positive_edges_by_modality": [
                    int(sum(row["per_modality"][modal]["positive_edges"] for row in relation))
                    for modal in range(6)],
                "candidate_negative_edges_by_modality": [
                    int(sum(row["per_modality"][modal]["negative_edges"] for row in relation))
                    for modal in range(6)],
                "baseline_positive_edges_by_modality": [
                    int(sum(row["per_modality"][modal]["positive_edges"] for row in base_relation))
                    for modal in range(6)],
                "baseline_negative_edges_by_modality": [
                    int(sum(row["per_modality"][modal]["negative_edges"] for row in base_relation))
                    for modal in range(6)],
                "candidate_pair_loss_mean": float(np.mean([row["pair_loss"] for row in relation])),
                "baseline_pair_loss_mean": float(np.mean([row["pair_loss"] for row in base_relation])),
                "candidate_adv_loss_mean": float(np.mean([row["adv_loss"] for row in relation])),
                "candidate_preserve_loss_mean": float(np.mean([row["preserve_loss"] for row in relation])),
            }
        result[group] = value
    return result


def main():
    runner.configure_worker()
    x_np, y_np, ids, splits = runner.load_data()
    device = torch.device("cuda:0")
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device).long()
    records, patients, uncertainty_rows = [], [], []
    for split in splits:
        fold = int(split["fold"])
        for group in (f"X{index}" for index in range(1, 7)):
            row, patient_rows = graph_and_prediction_record(
                group, BASE_FOR_NEW[group], fold, x, y, y_np, ids, split, device, uncertainty_rows)
            records.append(row)
            patients.extend(patient_rows)
        print(f"MECHANISM fold {fold:02d} PASS", flush=True)
    result = {
        "experiment": "Experiment 83.8", "status": "PASS",
        "historical_base_model": "E83.5 Best checkpoints, loaded read-only",
        "graph_change_definition": "Exact difference in the six modality adjacency tensors returned by GraphLearn/DARG, compared with the corresponding historical H model.",
        "records": records, "groups": aggregate(records),
        "dpsl_patient_analysis": {
            "definition": "Native is the same Best checkpoint with DPSL disabled. Full is the same checkpoint with DPSL enabled.",
            "relative_to_H4": "Candidate Full and Native are paired by sample index to historical H4 Full and Native.",
            "folds": 10,
        },
    }
    runner.write_json(runner.ROOT / "audits/mechanism_summary.json", result)
    pd.DataFrame(patients).sort_values(["group", "sample_index"]).to_csv(
        runner.ROOT / "results/dpsl_patient_mechanism.csv", index=False)
    pd.DataFrame(uncertainty_rows).sort_values(
        ["group", "fold", "pair_weight"], ascending=[True, True, False]).to_csv(
            runner.ROOT / "results/x6_uncertainty_pair_examples.csv", index=False)
    print(json.dumps({"status": result["status"], "groups": list(result["groups"])}, indent=2))


if __name__ == "__main__":
    main()
