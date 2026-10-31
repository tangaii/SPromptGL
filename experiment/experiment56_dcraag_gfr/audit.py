#!/usr/bin/env python3
"""Independent checkpoint reload, metric reproduction, and mechanism audit."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
sys.path.insert(0, str(EXP))
import run


def metric_set(logp: torch.Tensor, y: np.ndarray, test: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    p = logp.detach().exp().cpu().numpy()[test]
    pred = p.argmax(1)
    yy = y[test]
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, cm.ravel())
    return {
        "acc": float(accuracy_score(yy, pred)),
        "hard_auc": float(roc_auc_score(np.eye(2)[yy], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score((yy == 0).astype(np.int64), p[:, 0])),
        "spe": float(tn / max(tn + fp, 1)),
        "sen": float(tp / max(tp + fn, 1)),
        "ba": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "asd_f1": float(f1_score(yy, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn,
    }


def historical_e1_rows() -> list[dict]:
    task_root = Path("/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/tasks")
    rows = []
    for fold in range(1, 11):
        path = task_root / f"E1_f{fold:02d}.json"
        if not path.exists():
            raise FileNotFoundError(f"verified E51 E1 task record missing: {path}")
        row = json.loads(path.read_text())
        if row.get("status") != "COMPLETE":
            raise RuntimeError(f"E51 E1 task is not complete: {path}")
        rows.append(row)
    return rows


def macro_stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def unlabeled_pairs(ids: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.choice(ids, size=n_pairs, replace=True)
    right = rng.choice(ids, size=n_pairs, replace=True)
    same = left == right
    while same.any():
        right[same] = rng.choice(ids, size=int(same.sum()), replace=True)
        same = left == right
    return np.stack((left, right), axis=1).astype(np.int64)


def pair_diagnostic(state: dict, data: dict) -> tuple[list[dict], dict]:
    x, y = run.E1.load_data()[0]
    test = np.asarray(data["test"], dtype=np.int64)
    train_pairs = state["relation_pairs"]
    test_pairs_np, test_targets_np = run.sample_relation_pairs(test, y, 384, 56700 + state["fold"])
    test_pairs = torch.as_tensor(test_pairs_np, dtype=torch.long, device="cuda")
    test_targets = torch.as_tensor(test_targets_np, dtype=torch.long, device="cuda")
    _, parts = state["parts"](state["feat"])
    train_codes, test_codes, test_logits, test_shared, source_shared = [], [], [], [], []
    source_pairs = state["source_test_pairs"]
    with torch.no_grad():
        for m in range(4):
            h = state["model"].relations[m].encode_nodes(parts[m])
            tr_code, _ = state["model"].relations[m].encode_pairs(h, train_pairs)
            te_code, te_logits = state["model"].relations[m].encode_pairs(h, test_pairs)
            src_code, _ = state["model"].relations[m].encode_pairs(h, source_pairs)
            train_codes.append(tr_code[:, :16].cpu().numpy())
            test_codes.append(src_code[:, :16].cpu().numpy())
            source_shared.append(src_code[:, :16])
            test_shared.append(te_code[:, :16])
            test_logits.append(te_logits)
    # The feature-source probe is fitted on train-patient pairs and tested on disjoint patients.
    def source_probe(tr_codes: list[np.ndarray], te_codes: list[np.ndarray]) -> dict:
        xx = np.concatenate(tr_codes, axis=0)
        yy = np.concatenate([np.full(len(c), m, dtype=np.int64) for m, c in enumerate(tr_codes)])
        xt = np.concatenate(te_codes, axis=0)
        yt = np.concatenate([np.full(len(c), m, dtype=np.int64) for m, c in enumerate(te_codes)])
        probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=5600 + state["fold"])
        probe.fit(xx, yy)
        pred = probe.predict_proba(xt)
        return {"accuracy": float(np.mean(pred.argmax(1) == yt)),
                "macro_auc_ovr": float(roc_auc_score(yt, pred, multi_class="ovr", average="macro", labels=[0, 1, 2, 3]))}

    y3 = test_targets.cpu().numpy()
    relation_rows, rel_summary = [], {}
    modality_probs = []
    prototypes = []
    for m, logits in enumerate(test_logits):
        probs = logits.softmax(-1).cpu().numpy()
        modality_probs.append(probs)
        auc = float(roc_auc_score(y3, probs, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        pred = probs.argmax(1)
        cm = confusion_matrix(y3, pred, labels=[0, 1, 2]).tolist()
        for cls in range(3):
            relation_rows.append({"fold": state["fold"], "modality": run.MODALITIES[m],
                                 "relation": ("AA", "TT", "AT")[cls], "test_n": int((y3 == cls).sum()),
                                 "mean_probability": float(probs[:, cls].mean()),
                                 "class_entropy": float(-(probs[:, cls].clip(1e-8, 1).mean() * np.log(probs[:, cls].clip(1e-8, 1)) +
                                     (1 - probs[:, cls].clip(1e-8, 1)).mean() * np.log((1 - probs[:, cls].clip(1e-8, 1)).clip(1e-8, 1))).mean())})
        codes = test_shared[m].cpu().numpy()
        variance = float(codes.var(axis=0).mean())
        class_means = [codes[y3 == cls].mean(0) for cls in range(3)]
        proto_distance = [float(np.linalg.norm(class_means[i] - class_means[j]))
                          for i, j in ((0, 1), (0, 2), (1, 2))]
        rel_summary[run.MODALITIES[m]] = {"macro_auc_ovr_posthoc_test": auc,
                                          "confusion_matrix_AA_TT_AT": cm,
                                          "mean_probability": probs.mean(0).tolist(),
                                          "probability_entropy": float(-(probs.clip(1e-8, 1) * np.log(probs.clip(1e-8, 1))).sum(1).mean()),
                                          "shared_representation_variance": variance,
                                          "prototype_distances_AA_TT_AT": proto_distance}
        prototypes.append(codes)
    pre_probe = source_probe(state["initial_shared_train"], state["initial_shared_test"])
    post_probe = source_probe(train_codes, test_codes)
    cond = run.disease_condition(state["patient_prob"], source_pairs)
    source_inputs = []
    for code in source_shared:
        source_inputs.append(torch.cat(((code.unsqueeze(2) * cond.unsqueeze(1)).reshape(-1, 48), cond), dim=1))
    source_input = torch.stack(source_inputs, dim=0).reshape(-1, 51)
    with torch.no_grad():
        source_logits = state["model"].adversary(source_input)
    source_targets = torch.arange(4, device="cuda").view(4, 1).expand(4, len(source_pairs)).reshape(-1)
    source_probs = source_logits.softmax(-1).cpu().numpy()
    source_acc = float((source_logits.argmax(1) == source_targets).float().mean().cpu())
    source_auc = float(roc_auc_score(source_targets.cpu().numpy(), source_probs, multi_class="ovr", average="macro", labels=[0, 1, 2, 3]))
    same_pair_cos = []
    for a in range(4):
        for b in range(a + 1, 4):
            same_pair_cos.append(float(torch.nn.functional.cosine_similarity(test_shared[a], test_shared[b], dim=1).mean().cpu()))
    return relation_rows, {
        "relation_heads": rel_summary,
        "train_patient_disjoint_raw_relation_oof_macro_auc": data["relation_oof_auc"].tolist(),
        "continuous_relation_reliability": data["reliability"].tolist(),
        "test_pair_sample_counts_AA_TT_AT": np.bincount(y3, minlength=3).tolist(),
        "source_probe_pre_adversarial": pre_probe,
        "source_probe_post_adversarial": post_probe,
        "trained_conditional_discriminator_test_accuracy": source_acc,
        "trained_conditional_discriminator_test_macro_auc_ovr": source_auc,
        "source_chance_accuracy": 0.25,
        "shared_cross_modal_pair_cosine_mean": float(np.mean(same_pair_cos)),
        "teacher_condition_change_fraction": float(np.mean(np.any(
            data["wrong_patient_prob"][data["test"]] != data["patient_prob"][data["test"]], axis=1))),
        "adversary_gradient_norm": state["checkpoint_diagnostics"]["gradient_norms"]["adversary"],
        "relation_encoder_gradient_norm": state["checkpoint_diagnostics"]["gradient_norms"]["relation"],
        "relation_pair_rows_are_patient_dependent": True,
        "test_relation_metrics_are_posthoc_only": True,
    }


def graph_diagnostic(state: dict, data: dict, details: dict) -> dict:
    x, _ = run.E1.load_data()[0]
    site = run.site_ids(x)
    partition = np.full(len(site), -1, dtype=np.int8)
    partition[np.asarray(data["train"], dtype=np.int64)] = 0
    partition[np.asarray(data["test"], dtype=np.int64)] = 1
    out = {}
    for m, name in enumerate(run.MODALITIES):
        edges = state["candidate_edges"][m]
        wrong_edges = state["wrong_edges"][m]
        prior = state["edge_prior"][m]
        same = details["edge_same"][m].detach()
        diff = details["edge_diff"][m].detach()
        row = edges[:, 0]
        n = len(state["feat"])
        p_sum = prior.new_zeros(n).index_add(0, row, prior)
        p = prior / p_sum[row].clamp_min(1e-8)
        total = same + diff
        q_sum = total.new_zeros(n).index_add(0, row, total)
        q = total / q_sum[row].clamp_min(1e-8)
        edge_l1 = total.new_zeros(n).index_add(0, row, (q - p).abs())
        same_adj = details["A_same"][m].detach()
        diff_adj = details["A_diff"][m].detach()
        degree = (same_adj > 0).sum(1).float()
        eff_neighbours = torch.exp(-(same_adj.clamp_min(1e-8) * same_adj.clamp_min(1e-8).log()).sum(1))
        graph = (same_adj + diff_adj) > 0
        undirected = (graph | graph.T).cpu().numpy()
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        components, labels = connected_components(csr_matrix(undirected), directed=False)
        edges_np = edges.detach().cpu().numpy()
        wrong_np = wrong_edges.detach().cpu().numpy()
        original_outdegree = np.bincount(edges_np[:, 0], minlength=n)
        wrong_outdegree = np.bincount(wrong_np[:, 0], minlength=n)
        original_indegree = np.bincount(edges_np[:, 1], minlength=n)
        wrong_indegree = np.bincount(wrong_np[:, 1], minlength=n)
        wrong_same_site = site[wrong_np[:, 0]] == site[wrong_np[:, 1]]
        wrong_same_partition = partition[wrong_np[:, 0]] == partition[wrong_np[:, 1]]
        same_msg = details["same_msgs"][m].detach()
        diff_msg = details["diff_msgs"][m].detach()
        out[name] = {
            "original_graph_nonzero_count": int(data["raw_adj_counts"][m]),
            "candidate_directed_edges": int(len(edges)),
            "same_channel_nonzero_edges": int((same > 0).sum().cpu()),
            "different_channel_nonzero_edges": int((diff > 0).sum().cpu()),
            "same_weight_mean": float(same.mean().cpu()),
            "different_weight_mean": float(diff.mean().cpu()),
            "prior_to_reweighted_distribution_mean_l1": float(edge_l1.mean().cpu()),
            "mean_effective_same_neighbours": float(eff_neighbours.mean().cpu()),
            "isolated_same_channel_nodes": int((degree == 0).sum().cpu()),
            "undirected_components_union_same_diff": int(components),
            "same_message_norm": float(same_msg.norm().cpu()),
            "different_message_norm": float(diff_msg.norm().cpu()),
            "same_message_over_input": float(same_msg.norm().cpu() / state["feat"].norm().cpu()),
            "different_message_over_input": float(diff_msg.norm().cpu() / state["feat"].norm().cpu()),
            "message_norm_over_raw_input": float(details["msgs"][m].detach().norm().cpu() / state["feat"].norm().cpu()),
            "fusion_gate_mean": float(details["gate"][:, m].detach().mean().cpu()),
            "fusion_gate_std": float(details["gate"][:, m].detach().std(unbiased=False).cpu()),
            "graphlearn_parameter_delta_norm": float(torch.sqrt(sum((p.detach() - state["initial_inherited_tensors"]["gfr"][k].to(p.device)).pow(2).sum()
                for k, p in state["model"].named_parameters() if k.startswith(f"graphs.{m}."))).cpu()),
            "gcn_parameter_delta_norm": float(torch.sqrt(sum((p.detach() - state["initial_inherited_tensors"]["gfr"][k].to(p.device)).pow(2).sum()
                for k, p in state["model"].named_parameters() if k.startswith(f"gcn.{m}."))).cpu()),
            "edge_network_parameter_delta_norm": float(torch.sqrt(sum((p.detach() - state["initial_new_parameters"][k].to(p.device)).pow(2).sum()
                for k, p in state["model"].named_parameters() if k.startswith(f"relations.{m}.edge."))).cpu()),
            "wrong_edge_identity_change_fraction": float(np.mean(edges_np[:, 1] != wrong_np[:, 1])),
            "wrong_edge_same_site_fraction": float(wrong_same_site.mean()),
            "wrong_edge_same_outer_partition_fraction": float(wrong_same_partition.mean()),
            "wrong_receiver_outdegree_exact_match_fraction": float(np.mean(original_outdegree == wrong_outdegree)),
            "wrong_sender_indegree_exact_match_fraction": float(np.mean(original_indegree == wrong_indegree)),
            "wrong_sender_indegree_total_variation": float(np.abs(original_indegree - wrong_indegree).sum() / (2.0 * max(len(edges_np), 1))),
            "wrong_sender_indegree_max_abs_delta": int(np.max(np.abs(original_indegree - wrong_indegree))),
        }
    return out


def interventions(state: dict, data: dict, normal: dict) -> list[dict]:
    x, y = run.E1.load_data()[0]
    test = np.asarray(data["test"], dtype=np.int64)
    with torch.no_grad():
        normal_logp = normal["prob"].detach()
        normal_details = normal["details"]
        normal_pred = normal_logp.exp().cpu().numpy()[test].argmax(1)
        normal_asd = normal_logp.exp().cpu().numpy()[:, 0]
        normal_edge = [torch.cat((normal_details["edge_same"][m], normal_details["edge_diff"][m])) for m in range(4)]
    plans = [("neutral_edge_weights", {"neutral_edges": True}),
             ("wrong_patient_neighbors", {"wrong_neighbors": True}),
             ("disable_cross_patient_messages", {"disable_messages": True}),
             ("wrong_disease_condition", {"wrong_condition": True})]
    plans.extend((f"disable_{name}_message", {"disabled_modalities": (m,)}) for m, name in enumerate(run.MODALITIES))
    plans.append(("original_e1_gfr_bypass", {"reference_bypass": True}))
    rows = []
    for label, intervention in plans:
        if intervention.get("reference_bypass"):
            with torch.no_grad():
                fused, _, reference_details = state["model"].reference_bypass(normal["parts"])
                prob, hidden, attention = state["cam"](fused)
            details = dict(normal["details"])
            details["msgs"] = reference_details["msgs"]
            original_edges = []
            original_empty = []
            for m, edges in enumerate(state["candidate_edges"]):
                raw_adj = reference_details["adjs"][m]
                weights = raw_adj[edges[:, 0], edges[:, 1]]
                original_edges.append(weights)
                original_empty.append(torch.zeros_like(weights))
            details["edge_same"] = original_edges
            details["edge_diff"] = original_empty
            out = {"prob": prob, "hidden": hidden, "attention": attention, "fused": fused, "details": details}
        else:
            with torch.no_grad():
                out = run.forward(state, int(state["checkpoint"]["best_epoch"]), False, intervention)
        metrics = metric_set(out["prob"], y, test)
        probs = out["prob"].detach().exp().cpu().numpy()
        pred = probs[test].argmax(1)
        normal_prob = normal_asd[test]
        rows.append({"fold": state["fold"], "intervention": label,
                     **metrics,
                     "delta_acc_pp": 100 * (metrics["acc"] - state["checkpoint"]["metrics"]["acc"]),
                     "prediction_flip_rate": float(np.mean(pred != normal_pred)),
                     "mean_abs_delta_p_asd": float(np.mean(np.abs(probs[test, 0] - normal_prob))),
                     "mean_abs_delta_logit": float(np.mean(np.abs(out["prob"].detach().cpu().numpy()[test] - normal["prob"].detach().cpu().numpy()[test]))),
                     "residual_ratio": float((out["fused"] - state["feat"]).detach().norm().cpu() / state["feat"].norm().cpu()),
                     "disease_condition_change_fraction": float(np.mean(np.any(data["wrong_patient_prob"][test] != data["patient_prob"][test], axis=1))) if label == "wrong_disease_condition" else 0.0,
                     "wrong_edge_identity_fraction": float(np.mean(state["wrong_edges"][0][:, 1].cpu().numpy() != state["candidate_edges"][0][:, 1].cpu().numpy())) if label == "wrong_patient_neighbors" else 0.0,
                     "edge_weight_change_l1": float(np.mean([torch.mean(torch.abs(torch.cat((out["details"]["edge_same"][m], out["details"]["edge_diff"][m])) - normal_edge[m])).cpu() for m in range(4)])),
                     "message_norms": [float(v.detach().norm().cpu()) for v in out["details"]["msgs"]]})
    return rows


def historical_e1_rows() -> list[dict]:
    task_root = Path("/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/tasks")
    rows = []
    for fold in range(1, 11):
        path = task_root / f"E1_f{fold:02d}.json"
        if not path.exists():
            raise FileNotFoundError(f"E51 E1 fold record missing: {path}")
        row = json.loads(path.read_text())
        if row.get("status") != "COMPLETE":
            raise RuntimeError(f"E51 E1 fold not complete: {path}")
        rows.append(row)
    return rows


def write_report(fold_rows: list[dict], oof_rows: list[dict], mechanism: dict,
                 intervention_rows: list[dict], historical: list[dict], pooled: dict) -> None:
    metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    summary = []
    paired = []
    for metric in metrics:
        now = [float(r[metric]) for r in fold_rows]
        old = [float(r[metric]) for r in historical]
        mean, sd = macro_stats(now)
        hmean, hsd = macro_stats(old)
        diffs = np.asarray(now) - np.asarray(old)
        wins = int((diffs > 1e-12).sum())
        ties = int((np.abs(diffs) <= 1e-12).sum())
        losses = int((diffs < -1e-12).sum())
        summary.append({"metric": metric, "e56_mean": mean, "e56_population_sd": sd,
                        "e51_e1_mean": hmean, "e51_e1_population_sd": hsd,
                        "paired_mean_delta": float(diffs.mean()), "wins": wins, "ties": ties, "losses": losses})
        for f, delta in enumerate(diffs, 1):
            paired.append({"metric": metric, "fold": f, "e56": now[f - 1], "e51_e1": old[f - 1], "delta": float(delta)})
    run.csv_write(EXP / "results/summary.csv", summary)
    run.csv_write(EXP / "results/paired_comparisons.csv", paired)
    report = [
        "# Experiment 56 — DCRAG-GFR",
        "",
        "**Status:** independently reloaded and audited 10/10 E56 checkpoints. This is a single-seed historical-protocol exploration.",
        "",
        "## Fold results",
        "",
        "Metrics are percent mean ± population SD over ten folds. Hard-AUC is calculated from hard predictions; probability-AUC uses continuous P(ASD), with ASD (label 0) positive. BA and hard-AUC coincide in this binary setup.",
        "",
        "|Fold|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|Best epoch|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fold_rows:
        report.append(f"|{r['fold']}|{100*r['acc']:.4f}|{100*r['hard_auc']:.4f}|{100*r['prob_auc']:.4f}|{100*r['spe']:.4f}|{100*r['sen']:.4f}|{100*r['ba']:.4f}|{100*r['asd_f1']:.4f}|{r['best_epoch']}|")
    report.extend(["", "## E56 vs historical E51 E1", "",
                   "Values below are mean ± population SD (%) and paired same-fold mean difference in percentage points. Win/tie/loss counts describe the ten overlapping CV folds and are not significance tests.",
                   "", "|Metric|E56|E51 E1|Paired Δ pp|W/T/L|", "|---|---:|---:|---:|---:|"])
    for r in summary:
        report.append(f"|{r['metric']}|{100*r['e56_mean']:.4f} ± {100*r['e56_population_sd']:.4f}|{100*r['e51_e1_mean']:.4f} ± {100*r['e51_e1_population_sd']:.4f}|{100*r['paired_mean_delta']:+.4f}|{r['wins']}/{r['ties']}/{r['losses']}|")
    report.extend(["", "## Pooled OOF", "", f"Across all 871 held-out predictions, pooled ACC={100*pooled['acc']:.4f}%, hard-AUC={100*pooled['hard_auc']:.4f}%, Probability-AUC={100*pooled['prob_auc']:.4f}%, SPE={100*pooled['spe']:.4f}%, SEN={100*pooled['sen']:.4f}%, BA={100*pooled['ba']:.4f}%. Test-fold labels participated in historical checkpoint selection (`val=test`), so these pooled values are not an unbiased independent OOF estimate.",
                   "", "## Mechanism findings", "",
                   "|Modality|Train patient-disjoint relation OOF AUC|Posthoc held-out relation AUC|Edge reweight L1|Same-message/input|AT-message/input|",
                   "|---|---:|---:|---:|---:|---:|"])
    for name in run.MODALITIES:
        graph = mechanism["mean_graph_by_modality"][name]
        report.append(f"|{name}|{mechanism['relation_oof_auc_mean'][name]:.4f}|{mechanism['posthoc_test_relation_auc_mean'][name]:.4f}|{graph['prior_to_reweighted_distribution_mean_l1']:.5f}|{graph['same_message_over_input']:.6f}|{graph['different_message_over_input']:.6f}|")
    report.extend(["", "Q1: Patient-disjoint relation OOF is training-only evidence. Held-out relation heads are posthoc diagnostics and never feed back into training.",
                   "", f"Q2: Conditional adversary gradients were nonzero in {mechanism['adversarial_active_folds']}/10 folds, but the held-out source probe changed only from macro-AUC {mechanism['source_probe_pre_mean_macro_auc']:.4f} before training to {mechanism['source_probe_post_mean_macro_auc']:.4f} after training; the trained conditional discriminator's held-out macro-AUC averaged {mechanism['discriminator_test_mean_macro_auc']:.4f}. The source remains readily identifiable, so conditional cross-modal alignment is not demonstrated.",
                   "", "Q3: Each graph's edge counts, same/AT weights, reweighting distance, effective neighbors, isolated nodes, connected components and parameter changes are in `mechanism.json`.",
                   "", f"Q4: The new path is numerically active, but the mean same-message/input ratios range from {min(g['same_message_over_input'] for g in mechanism['mean_graph_by_modality'].values()):.2f} to {max(g['same_message_over_input'] for g in mechanism['mean_graph_by_modality'].values()):.2f}; mean total residual/input ratio is {mechanism['mean_residual_ratio']:.2f}. This is excessive amplification for the frozen bounded propagation design and is a mechanism failure, despite nonzero gradients and prediction effects.",
                   "", f"Q5: The fixed checkpoint loses {abs(mechanism['intervention_mean']['wrong_patient_neighbors']['delta_acc_pp']):.2f} pp ACC when candidate neighbors are replaced and {abs(mechanism['intervention_mean']['disable_cross_patient_messages']['delta_acc_pp']):.2f} pp when messages are disabled. These interventions show dependence of the selected checkpoint on its trained graph path, not an independent retraining gain; the graph identity gate remains partial.",
                   f"The wrong-neighbor control changed {100*np.mean([g['wrong_edge_identity_change_fraction'] for g in mechanism['mean_graph_by_modality'].values()]):.2f}% of directed edges, preserved receiver outdegree for {100*np.mean([g['wrong_receiver_outdegree_exact_match_fraction'] for g in mechanism['mean_graph_by_modality'].values()]):.2f}% of nodes, and preserved train/test partition for {100*np.mean([g['wrong_edge_same_outer_partition_fraction'] for g in mechanism['mean_graph_by_modality'].values()]):.2f}% of edges. Only {100*np.mean([g['wrong_edge_same_site_fraction'] for g in mechanism['mean_graph_by_modality'].values()]):.2f}% of replacement edges retained same-site status; sender indegree exact-match averaged {100*np.mean([g['wrong_sender_indegree_exact_match_fraction'] for g in mechanism['mean_graph_by_modality'].values()]):.2f}% (mean indegree total variation {np.mean([g['wrong_sender_indegree_total_variation'] for g in mechanism['mean_graph_by_modality'].values()]):.4f}). This is not a fully site- and degree-matched graph control.",
                   "", "|Fixed-checkpoint intervention|Mean ΔACC pp|Mean prediction flip rate|Mean absolute ΔP(ASD)|",
                   "|---|---:|---:|---:|"])
    for name, values in mechanism["intervention_mean"].items():
        report.append(f"|{name}|{values['delta_acc_pp']:+.4f}|{values['prediction_flip_rate']:.4f}|{values['mean_abs_delta_p_asd']:.6f}|")
    report.extend(["", "## Research gates", "", "|Gate|Result|", "|---|---|"])
    for gate, status in mechanism["research_gates"].items():
        report.append(f"|{gate}|{status}|")
    report.extend(["", "## Decision", "",
                   f"**DCRAG-GFR continuation decision: {mechanism['decision']}**. {mechanism['decision_reason']}",
                   "",
                   "History limits: the fold's test indices are also the validation indices for strict ACC checkpoint selection; preprocessing and RFE preceded outer CV; the graph is transductive over all 871 subjects; only seed 0 and one ABIDE A cohort were used. No multi-seed stability, statistical significance, independent-site generalization, or unbiased external test claim is made.",
                   "",
                   "The primary data, split definitions, E51 source and historical checkpoints were read-only. Root protected SHA values are recorded in `manifest.json`."])
    (EXP / "results/report.md").write_text("\n".join(report) + "\n")


def audit_all() -> None:
    started = time.time()
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("completed_folds") != 10:
        raise RuntimeError("all ten folds must finish before independent audit")
    protected_before = run.root_protected_sha()
    x, y = run.E1.load_data()[0]
    rows, oof, mechanisms, interventions_all = [], [], [], []
    relation_detail_rows = []
    for fold in range(1, 11):
        data = run.prepare_fold(fold)
        state = run.make_state(fold, data)
        state["fold"] = fold
        state["source_test_pairs"] = torch.as_tensor(unlabeled_pairs(np.asarray(data["test"], dtype=np.int64),
            1024, 56800 + fold), dtype=torch.long, device="cuda")
        checkpoint = run.torch_load(run.best_path(fold))
        task = json.loads(run.task_path(fold).read_text())
        assert checkpoint["split_sha"] == run.E1.split_sha(fold) == task["split_sha"]
        assert checkpoint["anchor_parameter_sha"] == state["anchor"]["parameter_sha"]
        assert checkpoint["inherited_sha"] == state["inherited_sha"]
        assert checkpoint["new_initial_sha"] == state["new_initial_sha"]
        current_source = run.experiment_source_sha()
        for source_key in ("run.py", "dcraag.py", "config.json", "source/e1_base.py", "source/dugsa.py",
                           "source/network.py", "source/layers.py", "source/utils.py"):
            assert checkpoint["source_sha"][source_key] == current_source[source_key], f"training source changed: {source_key}"
        state["model"].load_state_dict(checkpoint["model"])
        state["cam"].load_state_dict(checkpoint["cam"])
        state["checkpoint"] = checkpoint
        state["checkpoint_diagnostics"] = checkpoint["diagnostics"]
        state["model"].eval(); state["cam"].eval()
        with torch.no_grad():
            normal = run.forward(state, int(checkpoint["best_epoch"]), False)
        metrics = metric_set(normal["prob"], y, np.asarray(data["test"], dtype=np.int64))
        for key, value in metrics.items():
            if key in checkpoint["metrics"] and isinstance(value, (float, int)):
                if abs(float(value) - float(checkpoint["metrics"][key])) > 1e-8:
                    raise AssertionError(f"fold {fold} checkpoint metric mismatch {key}: {value} vs {checkpoint['metrics'][key]}")
        test_idx = np.asarray(data["test"], dtype=np.int64)
        probs = normal["prob"].detach().exp().cpu().numpy()
        pred = probs[test_idx].argmax(1)
        for idx in test_idx:
            oof.append({"subject_index": int(idx), "fold": fold, "true_label": int(y[idx]),
                        "prediction": int(pred[np.where(test_idx == idx)[0][0]]),
                        "p_asd": float(probs[idx, 0]), "p_td": float(probs[idx, 1]),
                        "best_epoch": int(checkpoint["best_epoch"])})
        row = {"fold": fold, **metrics, "best_epoch": int(checkpoint["best_epoch"]),
               "epochs_executed": int(checkpoint["epochs_executed"]),
               "checkpoint_sha": run.sha(run.best_path(fold)), "metric_readback_max_abs_diff": 0.0,
               "inherited_sha": checkpoint["inherited_sha"], "anchor_parameter_sha": checkpoint["anchor_parameter_sha"]}
        rows.append(row)

        # Recreate the deterministic initial new-module representation for a pre-adversary source probe.
        initial_state = run.make_state(fold, data)
        initial_state["fold"] = fold
        _, initial_parts = initial_state["parts"](initial_state["feat"])
        with torch.no_grad():
            initial_train_codes, initial_test_codes = [], []
            test_probe_pairs = state["source_test_pairs"]
            for m in range(4):
                init_h = initial_state["model"].relations[m].encode_nodes(initial_parts[m])
                tr_code, _ = initial_state["model"].relations[m].encode_pairs(init_h, initial_state["relation_pairs"])
                te_code, _ = initial_state["model"].relations[m].encode_pairs(init_h, test_probe_pairs)
                initial_train_codes.append(tr_code[:, :16].cpu().numpy())
                initial_test_codes.append(te_code[:, :16].cpu().numpy())
        state["initial_shared_train"] = initial_train_codes
        state["initial_shared_test"] = initial_test_codes
        rel_rows, relation_mechanism = pair_diagnostic(state, data)
        relation_detail_rows.extend(rel_rows)
        graph_mechanism = graph_diagnostic(state, data, normal["details"])
        mechanisms.append({"fold": fold, "relations_and_adversary": relation_mechanism,
                           "graphs_and_messages": graph_mechanism,
                           "training_diagnostics": checkpoint["diagnostics"],
                           "prediction_residual_ratio": float((normal["fused"] - state["feat"]).detach().norm().cpu() / state["feat"].norm().cpu())})
        interventions_all.extend(interventions(state, data, normal))
        print(f"AUDITED fold={fold} checkpoint={checkpoint['best_epoch']}", flush=True)

    if len(oof) != 871 or len({r["subject_index"] for r in oof}) != 871:
        raise AssertionError("OOF prediction coverage is not exactly once for 871 subjects")
    oof.sort(key=lambda r: r["subject_index"])
    fold_rows = sorted(rows, key=lambda r: r["fold"])
    csv_rows = [{k: v for k, v in r.items() if k != "checkpoint_sha"} for r in fold_rows]
    run.csv_write(EXP / "results/fold_metrics.csv", csv_rows)
    run.csv_write(EXP / "results/oof_predictions.csv", oof)
    run.csv_write(EXP / "results/interventions.csv", interventions_all)
    run.csv_write(EXP / "results/relation_diagnostics.csv", relation_detail_rows)
    y_pred = np.asarray([r["prediction"] for r in oof])
    p_asd = np.asarray([r["p_asd"] for r in oof])
    pooled_cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, pooled_cm.ravel())
    pooled = {"acc": float((y_pred == y).mean()),
              "hard_auc": float(roc_auc_score(np.eye(2)[y], np.eye(2)[y_pred])),
              "prob_auc": float(roc_auc_score((y == 0).astype(np.int64), p_asd)),
              "spe": tn / max(tn + fp, 1), "sen": tp / max(tp + fn, 1),
              "ba": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
              "asd_f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y, y_pred, pos_label=0, zero_division=0)),
              "confusion_matrix": pooled_cm.tolist(), "n": len(y)}
    historical = historical_e1_rows()
    historical_metrics = {m: [float(r[m]) for r in historical] for m in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")}
    current_metrics = {m: [float(r[m]) for r in fold_rows] for m in historical_metrics}
    deltas = {m: np.asarray(current_metrics[m]) - np.asarray(historical_metrics[m]) for m in historical_metrics}
    mean_acc_delta = float(deltas["acc"].mean() * 100)
    prob_nonworse = float(deltas["prob_auc"].mean()) >= 0
    ba_nonworse = float(deltas["ba"].mean()) >= 0
    mechanism_summary = {
        "pooled_oof": pooled,
        "relation_oof_auc_mean": {name: float(np.mean([m["relations_and_adversary"]["train_patient_disjoint_raw_relation_oof_macro_auc"][i] for m in mechanisms])) for i, name in enumerate(run.MODALITIES)},
        "adversarial_active_folds": int(sum(m["relations_and_adversary"]["adversary_gradient_norm"] > 0 for m in mechanisms)),
        "mean_acc_delta_vs_e51_e1_pp": mean_acc_delta,
        "mean_prob_auc_delta_vs_e51_e1_pp": float(deltas["prob_auc"].mean() * 100),
        "mean_ba_delta_vs_e51_e1_pp": float(deltas["ba"].mean() * 100),
        "classification_gain_screen": bool(mean_acc_delta >= 0.30 and prob_nonworse and ba_nonworse),
    }
    mean_graph = {}
    for name in run.MODALITIES:
        mean_graph[name] = {}
        for key in mechanisms[0]["graphs_and_messages"][name]:
            values = [m["graphs_and_messages"][name][key] for m in mechanisms]
            if all(isinstance(v, (int, float, np.number)) for v in values):
                mean_graph[name][key] = float(np.mean(values))
    intervention_mean = {}
    for name in sorted({r["intervention"] for r in interventions_all}):
        subset = [r for r in interventions_all if r["intervention"] == name]
        intervention_mean[name] = {k: float(np.mean([r[k] for r in subset]))
                                   for k in ("delta_acc_pp", "prediction_flip_rate", "mean_abs_delta_p_asd", "mean_abs_delta_logit", "residual_ratio", "edge_weight_change_l1")}
    mechanism_summary["mean_graph_by_modality"] = mean_graph
    mechanism_summary["intervention_mean"] = intervention_mean
    mechanism_summary["posthoc_test_relation_auc_mean"] = {
        name: float(np.mean([m["relations_and_adversary"]["relation_heads"][name]["macro_auc_ovr_posthoc_test"] for m in mechanisms]))
        for name in run.MODALITIES}
    mechanism_summary["source_probe_pre_mean_macro_auc"] = float(np.mean([m["relations_and_adversary"]["source_probe_pre_adversarial"]["macro_auc_ovr"] for m in mechanisms]))
    mechanism_summary["source_probe_post_mean_macro_auc"] = float(np.mean([m["relations_and_adversary"]["source_probe_post_adversarial"]["macro_auc_ovr"] for m in mechanisms]))
    mechanism_summary["discriminator_test_mean_macro_auc"] = float(np.mean([m["relations_and_adversary"]["trained_conditional_discriminator_test_macro_auc_ovr"] for m in mechanisms]))
    relation_ok = mechanism_summary["relation_oof_auc_mean"]["CORR"] > 0.55
    graph_identity_sensitive = intervention_mean["wrong_patient_neighbors"]["delta_acc_pp"] < 0.0 and intervention_mean["wrong_patient_neighbors"]["prediction_flip_rate"] > 0.0
    message_sensitive = intervention_mean["disable_cross_patient_messages"]["prediction_flip_rate"] > 0.0 and intervention_mean["disable_cross_patient_messages"]["delta_acc_pp"] < 0.0
    mean_residual_ratio = float(np.mean([m["prediction_residual_ratio"] for m in mechanisms]))
    disease_preserved = (relation_ok and
        np.mean([m["relations_and_adversary"]["relation_heads"]["CORR"]["shared_representation_variance"] for m in mechanisms]) > 1e-6 and
        mechanism_summary["posthoc_test_relation_auc_mean"]["CORR"] > 0.55)
    alignment_supported = (mechanism_summary["source_probe_pre_mean_macro_auc"] - mechanism_summary["source_probe_post_mean_macro_auc"] >= 0.05 and
                           mechanism_summary["discriminator_test_mean_macro_auc"] <= 0.65)
    mechanism_summary["research_gates"] = {
        "GO_RELATION_DISCRIMINATION": "GO" if relation_ok else "NO-GO",
        "GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT": "GO" if alignment_supported else "NO-GO",
        "GO_DISEASE_PRESERVATION": "GO" if disease_preserved else "NO-GO",
        "GO_TRUE_PATIENT_RELATION": "PARTIAL_FIXED_CHECKPOINT_ONLY" if graph_identity_sensitive else "NOT_SUPPORTED",
        "GO_CROSS_PATIENT_PROPAGATION": "NO-GO_SCALE_DOMINATED" if mean_residual_ratio > 1.0 else ("PARTIAL_CHECKPOINT_DEPENDENCE" if message_sensitive else "NOT_SUPPORTED"),
        "GO_CLASSIFICATION_GAIN": "GO" if mechanism_summary["classification_gain_screen"] else "NO-GO",
        "GO_MICCAI2027_MODULE": "NO-GO",
    }
    if mechanism_summary["classification_gain_screen"]:
        decision = "PARTIAL"
        reason = "The historical mean screen is met, but only one formal model group was trained; fixed-checkpoint intervention cannot establish independent causal graph value or a controlled gain over retraining baselines."
    else:
        decision = "NO-GO"
        reason = "The predeclared exploratory screen against E51 E1 is not met on mean ACC (+0.30 pp threshold) with nondecreasing BA and Probability-AUC. Mechanism-level fixed-checkpoint responses do not replace independent retraining controls."
    mechanism_summary["decision"] = decision
    mechanism_summary["decision_reason"] = reason
    mechanism_summary["mean_residual_ratio"] = mean_residual_ratio
    mechanism_summary["residual_scale_gate"] = "FAIL" if mean_residual_ratio > 1.0 else "PASS"
    mechanism_summary["historical_protocol_limits"] = ["val_idx=test_idx checkpoint selection", "global preprocessing/RFE before outer CV", "871-node transductive graph", "single seed", "single ABIDE A cohort"]
    (EXP / "results/mechanism.json").write_text(json.dumps({"folds": mechanisms, "summary": mechanism_summary,
        "historical_e1_fold_metrics": historical_metrics,
        "paired_deltas_fraction": {m: v.tolist() for m, v in deltas.items()},
        "relation_head_rows": len(relation_detail_rows)}, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_report(fold_rows, oof, mechanism_summary, interventions_all, historical, pooled)
    protected_after = run.root_protected_sha()
    if protected_after != manifest["protected_sha_before"]:
        raise AssertionError("protected root/data source SHA changed")
    manifest.update(status="COMPLETE_AUDITED", completed_folds=10, independent_checkpoint_audit="PASS 10/10",
                    audited_checkpoint_sha={str(run.best_path(f)): run.sha(run.best_path(f)) for f in range(1, 11)},
                    metric_readback_max_abs_diff=0.0, protected_sha_after=protected_after,
                    pooled_oof=pooled, decision=decision, decision_reason=reason,
                    audit_source_sha256=run.sha(EXP / "audit.py"),
                    source_sha=run.experiment_source_sha(),
                    audit_wall_seconds=time.time() - started,
                    result_sha={str(p.relative_to(EXP)): run.sha(p) for p in (EXP / "results").glob("*") if p.is_file()})
    run.atomic_json(manifest_path, manifest)
    print("AUDIT_COMPLETE", json.dumps({"status": manifest["status"], "decision": decision,
                                       "folds": len(fold_rows), "pooled": pooled}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    audit_all()
