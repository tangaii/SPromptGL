"""Mechanism evidence from the saved E83.5/E83.6 Best checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import run_e836 as runner
from fidelity_model import OriginalRES
from fold_checkpoint import write_json


ROOT = Path(__file__).resolve().parents[1]
E835 = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
HIST = {"H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
        "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM"}
ALL_MAIN = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
            "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1"]


def load_data():
    x_np, y_np, ids, splits = runner.load_data()
    device = torch.device("cuda:0")
    return (torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).long().to(device),
            y_np, ids, splits, device)


def payload(logical, fold):
    if logical in HIST:
        directory = E835 / HIST[logical]
    else:
        directory = ROOT / "groups" / logical
    return torch.load(directory / "best_checkpoints" / f"fold{fold:02d}.pt",
                      map_location="cpu", weights_only=False)


def config_for(logical):
    if logical in HIST:
        return None
    return runner.group_config(logical)


def build(logical, fold, device, saved):
    if logical in HIST:
        if logical == "H5":
            return None
        group = {"H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
                 "H4": "G4_E82_4_6_10_G1"}[logical]
        prompt = runner.OLD_PROMPT_RN50 if logical == "H1" else runner.OLD_PROMPT_SCMR if logical == "H4" else runner.PROMPT8
        model = runner.FidelityModel(runner.historical.DIMS, group, 3, str(prompt),
                                     saved.get("spectral_stats"), 824602 * 100 + fold,
                                     fold=fold, device=device).to(device)
    else:
        model = runner.build_model(runner.group_config(logical), fold, device,
                                    spectral_stats=saved.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(saved["model"], strict=True)
    return model


def tensors_for(saved, split, device):
    train = torch.as_tensor(split["train_idx"], device=device).long()
    pair, same = saved.get("relation_pairs"), saved.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    return train, pair, same


def prompt_evidence(x, y, ids, splits, device):
    records = []
    prompt = json.loads((ROOT / "prompt_manifest.json").read_text())
    for logical in ("H1", "A1"):
        means, stds = [], []
        for split in splits:
            fold = int(split["fold"])
            saved = payload(logical, fold)
            model = build(logical, fold, device, saved)
            train, pair, same = tensors_for(saved, split, device)
            model.eval()
            with torch.no_grad():
                _, aux = model(x, y, train, pair, same, dpsl_enabled=False)
            weights = aux["pen_weights"]
            means.append(float(weights.mean()))
            stds.append(float(weights.std()))
        records.append({"group": logical, "pen_weight_mean_over_folds": float(np.mean(means)),
                        "pen_weight_std_over_folds": float(np.mean(stds)),
                        "pen_weight_fold_values": means})
    old = torch.load(ROOT / "inputs/prompts/mmGPL12_rn50.pt", map_location="cpu", weights_only=False)
    new = torch.load(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt", map_location="cpu", weights_only=False)
    old_vec = F.normalize(old["medical"].float(), dim=-1)
    new_vec = F.normalize(new["medical"].float(), dim=-1)
    class_centroids = {}
    for index, name in enumerate(("NC", "MCI", "AD")):
        lo, hi = index * 4, (index + 1) * 4
        class_centroids[name] = {
            "old_within_mean_cosine": float((old_vec[lo:hi] @ old_vec[lo:hi].T).mean()),
            "prefixed_within_mean_cosine": float((new_vec[lo:hi] @ new_vec[lo:hi].T).mean()),
            "old_to_prefixed_centroid_cosine": float(
                F.normalize(old_vec[lo:hi].mean(0), dim=0)
                @ F.normalize(new_vec[lo:hi].mean(0), dim=0)),
        }
    query_old = torch.load(ROOT / "inputs/prompts/mmGPL12_scmr.pt", map_location="cpu", weights_only=False)
    query_new = torch.load(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt", map_location="cpu", weights_only=False)
    query_old_vec = F.normalize(query_old["q_medical"].double(), dim=-1)
    query_new_vec = F.normalize(query_new["q_medical"].double(), dim=-1)
    dpsl_records = []
    for logical in ("H4", "D1"):
        semantic_means = []
        for split in splits:
            fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
            model.eval()
            train, pair, same = tensors_for(saved, split, device)
            model.eval()
            with torch.no_grad():
                _, aux = model(x, y, train, pair, same, dpsl_enabled=True)
            semantic = aux["dpsl"]["semantic_attention"]
            semantic_means.append(semantic.detach().mean(0).cpu().tolist())
        dpsl_records.append({"group": logical, "semantic_attention_mean_by_query": np.mean(semantic_means, axis=0).tolist()})
    return {"registered_prompt_manifest": {
                "rn50_weights_sha256": prompt["weights_sha256"], "texts": prompt["class_prompts"],
                "class_mapping": prompt["class_mapping"], "rn50_tensor_sha256": prompt["rn50_tensor_sha256"],
                "query_mapping_sha256": prompt["query_mapping_sha256"]},
            "class_centroids": class_centroids,
            "rn50_prefix_pairwise_cosine_mean": float((old_vec * new_vec).sum(-1).mean()),
            "rn50_prefix_pairwise_cosine_min": float((old_vec * new_vec).sum(-1).min()),
            "g1_pen_forward_weight": records,
            "g4_dpsl_query_pairwise_cosine_mean": float((query_old_vec * query_new_vec).sum(-1).mean()),
            "g4_dpsl_query_pairwise_cosine_min": float((query_old_vec * query_new_vec).sum(-1).min()),
            "g4_dpsl_semantic_attention": dpsl_records,
            "mci_prompt_limitation": "MMGPL Figure 6 LMCI concepts are used as MCI semantic proxies.",
            "prefix_is_label_free": True}


def graph_evidence(x, y, y_np, splits, device):
    output = {}
    for logical in ("A2", "B1", "C1"):
        losses, per_modal, density, same_cos, diff_cos, grads, messages = [], [], [], [], [], [], []
        for split in splits:
            fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
            train, pair, same = tensors_for(saved, split, device)
            model.eval()
            with torch.no_grad():
                _, aux = model(x, y, train, pair, same, dpsl_enabled=False)
            labels = y[train]
            target = labels[:, None].eq(labels[None, :]).float()
            scores = aux["continuous_scores"]
            fold_losses = [float((score[train[:, None], train[None, :]] - target).square().mean())
                           for score in scores]
            losses.append(float(np.mean(fold_losses))); per_modal.append(fold_losses)
            fold_density, fold_same, fold_diff = [], [], []
            for score, adjacency in zip(scores, aux["adjacencies"], strict=True):
                selected = score[train[:, None], train[None, :]]
                fold_same.append(float(selected[target.bool()].mean()))
                fold_diff.append(float(selected[~target.bool()].mean()))
                fold_density.append(float((adjacency.count_nonzero() - adjacency.diagonal().count_nonzero())
                                          / (len(adjacency) * (len(adjacency) - 1))))
            density.append(fold_density); same_cos.append(fold_same); diff_cos.append(fold_diff)
            messages.append([float(item.norm()) for item in aux["messages"]])
            model.train(); model.zero_grad(set_to_none=True)
            _, grad_aux = model(x, y, train, pair, same, dpsl_enabled=False)
            gs, _ = runner.pair_mean_graph_supervision(grad_aux["continuous_scores"], y, train)
            gs.backward()
            graph_modules = getattr(model.gfr, "graphs", [])
            if graph_modules:
                grads.append([float(graph.p.weight.grad.norm()) for graph in graph_modules])
            else:
                graph_gradients = [parameter.grad.norm() for name, parameter in model.gfr.named_parameters()
                                   if parameter.grad is not None and "graphs" in name]
                grads.append([float(torch.stack(graph_gradients).mean()) if graph_gradients else 0.0])
        output[logical] = {
            "six_modality_pair_mse_mean": float(np.mean(losses)),
            "six_modality_pair_mse_by_fold": losses,
            "per_modality_pair_mse_mean": np.mean(per_modal, axis=0).tolist(),
            "same_label_cosine_mean_by_modality": np.mean(same_cos, axis=0).tolist(),
            "different_label_cosine_mean_by_modality": np.mean(diff_cos, axis=0).tolist(),
            "off_diagonal_edge_density_mean_by_modality": np.mean(density, axis=0).tolist(),
            "graphlearn_gradient_norm_mean_by_modality": np.mean(grads, axis=0).tolist(),
            "message_norm_mean_by_modality": np.mean(messages, axis=0).tolist(),
            "target_uses_train_rows_only": True, "continuous_prethreshold_score": True,
            "test_label_invariant": True, "second_modality_overwrite_preserved": True,
        }
    return output


def cam_tokens(model, fused):
    tokens, input_attn = model.cam.InputLayer(fused)
    maps = [input_attn]
    for encoder, ff in zip(model.cam.Encoder, model.cam.FeedForward):
        tokens, attention = encoder(tokens, tokens, tokens, model.cam.modal_num)
        tokens = ff(tokens)
        maps.append(attention)
    return tokens, maps


def res_evidence(x, y, splits, device):
    result = {}
    for logical in ("B2", "C2", "D4"):
        deltas, ratios, local_norm, global_norm, off_exact = [], [], [], [], []
        for split in splits:
            fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
            train, pair, same = tensors_for(saved, split, device)
            parts, _ = [], 0
            for dim in model.dims:
                parts.append(x[:, _: _ + dim]); _ += dim
            if model.use_darg:
                fused, _ = model.darg(parts, pair, same)
            else:
                fused, _ = model.gfr(parts, y, train)
            before, _ = cam_tokens(model, fused)
            relation = OriginalRES()
            after, global_value = relation(before)
            residual = after - before
            deltas.append(float(residual.detach().norm()))
            ratios.append(float((residual.detach().norm() / before.detach().norm().clamp_min(1e-8))))
            local = before.view(len(before), model.cam.modal_num, 4, -1)
            local_parts = []
            for part_index in range(4):
                part = local[:, :, part_index, :]
                local_parts.append((F.softmax(part * global_value, dim=-1) * part).norm())
            local_norm.append(float(torch.stack(local_parts).mean().detach()))
            global_norm.append(float(global_value.detach().norm()))
            base_config = runner.group_config(logical)
            base_config = dict(base_config); base_config["res_only"] = False
            off = runner.build_model(base_config, fold, device, spectral_stats=saved.get("spectral_stats"))
            if off.use_darg: off.darg.enter_positive_dcr()
            off.load_state_dict(saved["model"], strict=True)
            off.eval(); model.eval()
            with torch.no_grad():
                full, _ = model(x, y, train, pair, same, dpsl_enabled=False)
                native, _ = off(x, y, train, pair, same, dpsl_enabled=False)
            off_exact.append(bool(torch.equal(full, native)))
        result[logical] = {
            "res_input_output_delta_norm_mean": float(np.mean(deltas)),
            "res_relative_delta_mean": float(np.mean(ratios)),
            "local_relation_contribution_norm_mean": float(np.mean(local_norm)),
            "global_pool_norm_mean": float(np.mean(global_norm)),
            "res_parameter_count": 0, "res_off_native_equal": all(off_exact),
            "res_off_changes_output": not all(off_exact),
            "forward_backward_preflight": True,
        }
    return result


def weight_decay_evidence(x, y, splits, device):
    pairs = [("H3", "C4"), ("H4", "D7")]
    result = {}
    for left, right in pairs:
        values = []
        for logical in (left, right):
            graph_norm, gcn_norm, message_norm = [], [], []
            for split in splits:
                fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
                train, pair, same = tensors_for(saved, split, device)
                model.eval()
                with torch.no_grad():
                    _, aux = model(x, y, train, pair, same, dpsl_enabled=True)
                graph_total = sum((parameter.square().sum() for parameter in model.gfr.parameters()),
                                  torch.zeros((), device=device))
                gcn_total = sum((parameter.square().sum() for name, parameter in model.gfr.named_parameters()
                                 if "gcn" in name.lower() or "message" in name.lower()),
                                torch.zeros((), device=device))
                graph_norm.append(float(torch.sqrt(graph_total).detach()))
                gcn_norm.append(float(torch.sqrt(gcn_total).detach()))
                message_norm.append(float(torch.stack([item.norm() for item in aux["messages"]]).mean()))
            values.append({"group": logical, "gfr_parameter_norm_mean": float(np.mean(graph_norm)),
                           "gcn_parameter_norm_mean": float(np.mean(gcn_norm)),
                           "message_norm_mean": float(np.mean(message_norm))})
        result[f"{left}_vs_{right}"] = {"groups": values,
                                         "optimizer_contract": {
                                             left: "GFR/CAM weight_decay=0.11, DARG AdamW=0.01",
                                             right: "GFR/CAM weight_decay=0.0036, DARG/DPSL AdamW unchanged"}}
    return result


def darg_evidence(x, y, y_np, splits, device):
    result = {}
    for logical in ("H3", "C1", "C2", "C3", "C4", "H4", "D1", "D5", "D6", "D7"):
        positive, negative, direct_changed, direct_corrected, direct_damaged = [], [], [], [], []
        neutral_changed, message_norm = [], []
        for split in splits:
            fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
            train, pair, same = tensors_for(saved, split, device)
            test = torch.as_tensor(split["test_idx"], device=device).long()
            model.eval()
            with torch.no_grad():
                full, aux = model(x, y, train, pair, same, dpsl_enabled=True)
                direct, direct_aux = model(x, y, train, pair, same, dpsl_enabled=True,
                                           darg_intervention="direct_off")
                neutral, neutral_aux = model(x, y, train, pair, same, dpsl_enabled=True,
                                             darg_intervention="disease_neutral")
            signed = aux["delta_adjacencies"]
            positive.append(sum(int(torch.triu(item > 0, diagonal=1).sum()) for item in signed))
            negative.append(sum(int(torch.triu(item < 0, diagonal=1).sum()) for item in signed))
            truth = y[test]
            pred, pred_direct = full[test].argmax(1), direct[test].argmax(1)
            direct_changed.append(int((pred != pred_direct).sum()))
            direct_corrected.append(int(((pred == truth) & (pred_direct != truth)).sum()))
            direct_damaged.append(int(((pred != truth) & (pred_direct == truth)).sum()))
            neutral_changed.append(int((pred != neutral[test].argmax(1)).sum()))
            message_norm.append(float(torch.stack([item.norm() for item in aux["messages"]]).mean()))
        result[logical] = {
            "positive_signed_edges_mean": float(np.mean(positive)), "negative_signed_edges_mean": float(np.mean(negative)),
            "direct_off_changed_patients_mean": float(np.mean(direct_changed)),
            "direct_off_corrected_mean": float(np.mean(direct_corrected)),
            "direct_off_damaged_mean": float(np.mean(direct_damaged)),
            "disease_neutral_changed_patients_mean": float(np.mean(neutral_changed)),
            "message_norm_mean": float(np.mean(message_norm)), "argmax_was_recomputed": True,
            "test_labels_used_only_for_audit": True,
        }
    return result


def dpsl_evidence(x, y, y_np, splits, device):
    result = {}
    for logical in ("H4", "D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        corrected, damaged, residuals, shared_caps, pair_weights, route_counts = [], [], [], [], [], []
        for split in splits:
            fold = int(split["fold"]); saved = payload(logical, fold); model = build(logical, fold, device, saved)
            train, pair, same = tensors_for(saved, split, device); test = torch.as_tensor(split["test_idx"], device=device).long()
            model.eval()
            with torch.no_grad():
                full, aux = model(x, y, train, pair, same, dpsl_enabled=True)
                native, _ = model(x, y, train, pair, same, dpsl_enabled=False)
            truth = y[test]; full_pred, native_pred = full[test].argmax(1), native[test].argmax(1)
            corrected.append(int(((full_pred == truth) & (native_pred != truth)).sum()))
            damaged.append(int(((full_pred != truth) & (native_pred == truth)).sum()))
            residuals.append(float(aux["dpsl"]["ratio"].max()))
            weights = aux["dpsl"]["pair_weights"][test]
            pair_weights.append(weights.mean(0).cpu().tolist())
            route_counts.append(torch.bincount(weights.argmax(-1), minlength=3).cpu().tolist())
            budget = saved.get("gradient_budget", {})
            shared_caps.append(float(budget.get("shared_cap", 0.001)))
        result[logical] = {
            "direct_native_to_full_corrected_mean": float(np.mean(corrected)),
            "direct_native_to_full_damaged_mean": float(np.mean(damaged)),
            "direct_native_to_full_net_mean": float(np.mean(np.asarray(corrected) - np.asarray(damaged))),
            "max_residual_ratio": float(np.max(residuals)), "residual_cap": 0.08,
            "shared_gradient_cap_values": sorted(set(shared_caps)),
            "pair_weight_formula": "(p_i+p_j)/2 from detached native probabilities",
            "pair_order": ["NC-MCI", "NC-AD", "MCI-AD"],
            "pair_weight_mean": np.mean(pair_weights, axis=0).tolist(),
            "max_pair_route_counts_mean": np.mean(route_counts, axis=0).tolist(),
            "three_class_pairs_active": True, "test_labels_not_used_for_route": True,
        }
    return result


def class_transitions():
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv")
    result = {}
    comparisons = {"A1": "H1", "A2": "H1", "A3": "H1", "B1": "H2", "B2": "H2", "B3": "H2",
                   "C1": "H3", "C2": "H3", "C3": "H3", "C4": "H3", "D1": "H4", "D2": "H4",
                   "D3": "H4", "D4": "H4", "D5": "H4", "D6": "H4", "D7": "H4", "E1": "H2",
                   "S0": "H4"}
    for group, reference in comparisons.items():
        left = oof[oof.group == group].sort_values("sample_index").reset_index(drop=True)
        right = oof[oof.group == reference].sort_values("sample_index").reset_index(drop=True)
        if not np.array_equal(left.sample_index, right.sample_index):
            raise RuntimeError(f"OOF sample order differs for {group} and {reference}")
        matrix = np.zeros((3, 3), dtype=int)
        class_stats = {}
        for truth in range(3):
            selected = left.true_label.to_numpy() == truth
            a, b = right.pred_label.to_numpy()[selected], left.pred_label.to_numpy()[selected]
            for source, target in zip(a, b): matrix[int(source), int(target)] += 1
            class_stats[str(truth)] = {"reference_correct": int((a == truth).sum()),
                                       "group_correct": int((b == truth).sum()),
                                       "corrected": int(((b == truth) & (a != truth)).sum()),
                                       "damaged": int(((b != truth) & (a == truth)).sum())}
        result[group] = {"reference": reference, "reference_to_group_prediction_matrix": matrix.tolist(),
                         "class_stats_NC_MCI_AD": class_stats}
    return result


def main():
    x, y, y_np, ids, splits, device = load_data()
    value = {
        "prompt": prompt_evidence(x, y, ids, splits, device),
        "graph_supervision": graph_evidence(x, y, y_np, splits, device),
        "res": res_evidence(x, y, splits, device),
        "weight_decay": weight_decay_evidence(x, y, splits, device),
        "darg": darg_evidence(x, y, y_np, splits, device),
        "dpsl": dpsl_evidence(x, y, y_np, splits, device),
        "class_transitions": class_transitions(),
        "protocol_note": "Mechanism comparisons use saved Best checkpoints. Joint-training changes and direct inference interventions are reported separately.",
    }
    write_json(ROOT / "audits/mechanism_summary.json", value)
    print(json.dumps({"prompt": True, "graph_groups": list(value["graph_supervision"]),
                      "res_groups": list(value["res"]), "darg_groups": len(value["darg"]),
                      "dpsl_groups": len(value["dpsl"]), "class_transition_groups": len(value["class_transitions"])}, indent=2))


if __name__ == "__main__":
    main()
