"""Independent release-bound reproduction of the Best checkpoints."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from fold_checkpoint import atomic_write, identity, sha_file, validate_identity, write_json
from metrics import acc_auc
from run_e833 import (GROUPS, ROOT, anchor_pair_support, build_model, configure_worker, fit_spectral_statistics,
                      load_data, make_pairs, require_formal_training_gate, seed_everything, sha_object)


def _checked_graph(aux, direct_off, neutral, logp, neutral_logp):
    positive = negative = 0
    for signed, direct, on, off, original, graph_neutral, msg, off_msg in zip(
        aux["delta_adjacencies"], aux["direct_delta_adjacencies"],
        aux["unified_adjacencies"], direct_off["unified_adjacencies"],
        aux["adjacencies"], neutral["unified_adjacencies"], aux["messages"], direct_off["messages"],
    ):
        torch.testing.assert_close(direct, signed.clamp_min(0), rtol=0, atol=0)
        torch.testing.assert_close(on, original + signed, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(on, off, rtol=0, atol=0)
        torch.testing.assert_close(msg, off_msg, rtol=0, atol=0)
        torch.testing.assert_close(original, graph_neutral, rtol=0, atol=0)
        assert torch.count_nonzero(signed.diag()) == 0
        assert torch.count_nonzero(signed[original <= 0]) == 0
        positive += int(torch.triu(signed > 0, diagonal=1).sum())
        negative += int(torch.triu(signed < 0, diagonal=1).sum())
    assert all(torch.count_nonzero(m) == 0 for m in direct_off["direct_messages"])
    response = float((logp - neutral_logp).abs().max())
    message_change = max(float((message - neutral_message).norm())
                        for message, neutral_message in zip(aux["messages"], neutral["messages"], strict=True))
    return {"positive_edges": positive, "negative_edges": negative,
            "neutral_intervention_logp_max_abs": response,
            "disease_neutral_gcn_message_change_max_norm": message_change,
            "active": positive + negative > 0 and response > 0 and message_change > 0}


def _checked_scmr(model, saved, rows, test, labels, x, pair, same, full, output):
    module = model.dpsl
    stats = saved["spectral_stats"]
    reference = stats["reference"]
    np.testing.assert_array_equal(reference["legal"], rows.cpu())
    np.testing.assert_array_equal(reference["labels"], labels[rows].cpu())
    refit = fit_spectral_statistics(saved["spectral_stats"]["warmup_cam"], labels.cpu(), rows.cpu(),
                                   fold=saved["fold"], warm_logits=saved["spectral_stats"]["warmup_logits"])
    for key in ("basis", "center", "variance", "log_variance", "disease_covariance",
                "bank_index", "hard_initial", "hard_initial_pairs"):
        torch.testing.assert_close(refit[key], saved["spectral_stats"][key], rtol=0, atol=0)
    for key in ("counts", "mask", "mean", "second", "stable", "calibration"):
        torch.testing.assert_close(refit["reference"][key], reference[key], rtol=0, atol=0)
    assert torch.count_nonzero(module.statistics.bank[test]) == 0
    assert tuple(module.basis.shape) == (240, 16)
    torch.testing.assert_close(module.basis.cpu(), stats["basis"].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(module.basis.T @ module.basis, torch.eye(16, dtype=torch.float64, device=x.device),
                               rtol=1e-8, atol=1e-8)
    assert module.ema_updates == saved["dpsl_ema_updates"] == saved["epoch"] - 11
    budget = saved["gradient_budget"]
    assert budget["shared_cap"] == .001
    shared_fractions = []
    for name in ("gfr", "darg", "cam", "head"):
        family = budget["families"][name]
        assert family["total_applied"] <= .001 * family["historical"]
        assert abs(family["applied_dot"]) <= family["dot_bound"]
        shared_fractions.append(family["total_applied"] / max(family["historical"], 1e-12))
    parts = full["dpsl"]
    ordered_pairs = ((0, 1), (0, 2), (1, 2))
    expected_directed_pairs = torch.tensor([[1, 0], [2, 0], [2, 1]], device=x.device)
    assert torch.equal(parts["class_pairs"], expected_directed_pairs)
    native_probabilities = full["native_logits"].softmax(-1).detach()
    expected_weights = torch.stack(((native_probabilities[:, 0] + native_probabilities[:, 1]) / 2,
                                    (native_probabilities[:, 0] + native_probabilities[:, 2]) / 2,
                                    (native_probabilities[:, 1] + native_probabilities[:, 2]) / 2), dim=1)
    torch.testing.assert_close(parts["pair_weights"], expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(parts["pair_weights"].sum(-1), torch.ones(len(x), device=x.device),
                               rtol=0, atol=3 * torch.finfo(parts["pair_weights"].dtype).eps)
    assert bool((parts["pair_weights"] > 0).all()) and len(parts["pair_outputs"]) == 3
    assert torch.isfinite(parts["delta"]).all()
    assert torch.all(parts["ratio"] <= .08 + 2 * torch.finfo(parts["ratio"].dtype).eps)
    native, base = model(x, labels, rows, pair, same, dpsl_enabled=False)
    off, detail = module(base["cam"], model.cam.Outputlayer, False, "residual_off")
    torch.testing.assert_close(off, base["cam"], rtol=0, atol=0)
    torch.testing.assert_close(model.cam.raw_logits_from_cam(off), full["native_logits"], rtol=0, atol=0)
    assert torch.count_nonzero(detail["delta"]) == 0
    change = float((native - output).abs().max())
    nonzero = int(torch.count_nonzero(parts["ratio"] > 0))
    native_prediction = native[test].argmax(1)
    full_prediction = output[test].argmax(1)
    truth = labels[test]
    corrected = int(((full_prediction == truth) & (native_prediction != truth)).sum())
    damaged = int(((full_prediction != truth) & (native_prediction == truth)).sum())
    test_rows = torch.as_tensor(test, device=x.device).long()
    test_weights = parts["pair_weights"][test_rows]
    all_route_argmax = parts["pair_weights"].argmax(-1)
    test_route_argmax = test_weights.argmax(-1)

    def pair_weight_summary(weights, selected):
        values = weights[selected]
        winners = values.argmax(-1) if len(values) else torch.empty(0, dtype=torch.long, device=x.device)
        counts = torch.bincount(winners, minlength=3) if len(values) else torch.zeros(3, dtype=torch.long, device=x.device)
        means = values.mean(0) if len(values) else values.new_zeros(3)
        return {"node_count": int(len(values)),
                "mean_weights": {f"{i}{j}": float(means[index])
                                 for index, (i, j) in enumerate(ordered_pairs)},
                "max_weight_pair_node_counts": {f"{i}{j}": int(counts[index])
                                                for index, (i, j) in enumerate(ordered_pairs)}}

    true_labels = labels[test_rows]
    native_correct_mask = native_prediction == truth
    route_by_true = {str(class_id): pair_weight_summary(test_weights, true_labels == class_id)
                     for class_id in range(3)}
    route_by_native_correctness = {
        "correct": pair_weight_summary(test_weights, native_correct_mask),
        "incorrect": pair_weight_summary(test_weights, ~native_correct_mask),
    }
    all_node_route_counts = torch.bincount(all_route_argmax, minlength=3)
    test_route_counts = torch.bincount(test_route_argmax, minlength=3)
    all_node_weight_summary = pair_weight_summary(
        parts["pair_weights"], torch.ones(len(x), dtype=torch.bool, device=x.device))

    native_probabilities_test = native[test_rows].exp()
    native_predictions_test = native_prediction.detach().cpu().numpy().astype(np.int64)
    full_predictions_test = full_prediction.detach().cpu().numpy().astype(np.int64)
    truth_test = truth.detach().cpu().numpy().astype(np.int64)
    native_probability_array = native_probabilities_test.detach().cpu().numpy()
    native_metrics = acc_auc(truth_test, native_predictions_test, 3)
    native_metrics["AUC_probability"] = float(roc_auc_score(
        truth_test, native_probability_array, average="macro", multi_class="ovr"))

    class_transitions = {}
    for class_id, class_name in enumerate(("NC", "MCI", "AD")):
        selected = truth == class_id
        class_transitions[class_name] = {
            "test_count": int(selected.sum()),
            "native_correct": int(((native_prediction == truth) & selected).sum()),
            "full_correct": int(((full_prediction == truth) & selected).sum()),
            "corrected": int(((full_prediction == truth) & (native_prediction != truth) & selected).sum()),
            "damaged": int(((full_prediction != truth) & (native_prediction == truth) & selected).sum()),
        }
    transitions = torch.bincount((native_prediction * 3 + full_prediction), minlength=9).reshape(3, 3)
    pair_contributions = {}
    for index, (mild, severe) in enumerate(ordered_pairs):
        detail = parts["pair_outputs"][index]
        weight = test_weights[:, index]
        weighted = (detail["raw_parallel"][test_rows] + detail["raw_perp"][test_rows]) * weight[:, None]
        pair_contributions[f"{mild}{severe}"] = {
            "weighted_raw_residual_mean_norm": float(weighted.norm(dim=-1).mean()),
            "weighted_parallel_mean_norm": float((detail["raw_parallel"][test_rows] * weight[:, None]).norm(dim=-1).mean()),
            "weighted_complement_mean_norm": float((detail["raw_perp"][test_rows] * weight[:, None]).norm(dim=-1).mean()),
        }
    ad_mask = true_labels == 2
    ad_evidence = {}
    for index, pair_name in ((1, "NC-AD"), (2, "MCI-AD")):
        detail = parts["pair_outputs"][index]
        product = detail["margin"][test_rows][ad_mask] * detail["direction"][test_rows][ad_mask]
        ad_evidence[pair_name] = {
            "mean_native_severe_margin": float(detail["margin"][test_rows][ad_mask].mean()),
            "mean_evidence_direction": float(detail["direction"][test_rows][ad_mask].mean()),
            "mean_conflict": float(detail["conflict"][test_rows][ad_mask].mean()),
            "evidence_opposes_native_margin_count": int((product < 0).sum()),
        }
    ad_true_native_ad_to_mci = int(((truth == 2) & (native_prediction == 2) & (full_prediction == 1)).sum())
    return {"spectral_basis_shape": list(module.basis.shape), "train_only_reference": True,
            "spectral_basis_and_bank_refit_exact": True,
            "hard_initial_pair_statistics_train_only": True,
            "reconstructed_warmup_spectrum_and_calibration": True,
            "ema_updates": module.ema_updates, "shared_gradient_cap": budget["shared_cap"],
            "max_shared_gradient_fraction": max(shared_fractions),
            "max_residual_ratio": float(parts["ratio"].max()), "nonzero_residual_rows": nonzero,
            "residual_off_logp_max_abs": change, "zero_residual_exact": True,
            "full_corrected": corrected, "full_damaged": damaged,
            "full_net_correct": corrected - damaged,
            "native_correct": int((native_prediction == truth).sum()),
            "full_correct": int((full_prediction == truth).sum()),
            "native_metrics": native_metrics,
            "native_prediction_sha256": hashlib.sha256(native_predictions_test.tobytes()).hexdigest(),
            "native_test_prediction": native_predictions_test.tolist(),
            "native_test_probability": native_probability_array.tolist(),
            "class_transitions": class_transitions,
            "native_to_full_transition_counts": transitions.detach().cpu().tolist(),
            "AD_native_correct_to_MCI_full": ad_true_native_ad_to_mci,
            "pair_weight_route": {
                "weight_formula": "(p_i+p_j)/2",
                "all_transductive_graph_nodes": {
                    "node_count": int(len(x)),
                    "mean_weights": all_node_weight_summary["mean_weights"],
                    "max_weight_pair_node_counts": {
                        f"{i}{j}": int(all_route_argmax.eq(index).sum())
                        for index, (i, j) in enumerate(ordered_pairs)},
                },
                "test_patients": {"node_count": int(len(test_rows)),
                                  "max_weight_pair_node_counts": {
                                      f"{i}{j}": int(test_route_counts[index])
                                      for index, (i, j) in enumerate(ordered_pairs)},
                                  "mean_weights": {f"{i}{j}": float(test_weights[:, index].mean())
                                                   for index, (i, j) in enumerate(ordered_pairs)}},
                "by_true_class": route_by_true,
                "by_native_prediction_correctness": route_by_native_correctness,
                "test_AD_excluded_pair_count": 0,
            },
            "weighted_pair_residual_contribution": pair_contributions,
            "AD_pair_evidence": ad_evidence,
            "query_count": int(module.spectrum.query_medical.shape[0]),
            "query_group_sizes": [4, 4, 4], "active": nonzero > 0 and change > 0}


@torch.no_grad()
def replay_fold(group, split, data=None, device=None, historical_identity=False):
    x_np, y_np, ids, _ = load_data() if data is None else data
    device = torch.device("cuda:0") if device is None else device
    fold = int(split["fold"])
    path = ROOT / group
    checkpoint = path / "best_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = path / "audits" / f"fold{fold:02d}.json"
    prediction_path = path / "predictions" / f"fold{fold:02d}.csv"
    tr = np.asarray(split["train_idx"], dtype=np.int64)
    te = np.asarray(split["test_idx"], dtype=np.int64)
    expected = identity(group, fold, tr, te)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    audit = json.loads(audit_path.read_text())
    if historical_identity:
        identity_keys = ("group", "fold", "seed", "model_type", "data_sha256", "split_sha256",
                         "train_idx", "test_idx")
        for payload in (saved, audit):
            if any(payload.get(key) != expected[key] for key in identity_keys):
                raise RuntimeError(f"Historical checkpoint identity differs for {group} fold {fold}")
    else:
        validate_identity(saved, expected)
        validate_identity(audit, expected)
    assert audit["status"] == "TRAINED"
    assert audit["checkpoint_sha256"] == sha_file(checkpoint)
    assert audit["best_epoch"] == saved["epoch"]
    assert audit["train_count"] == len(tr) and audit["test_count"] == len(te)
    assert audit["last_epoch"] <= 450 and (audit["last_epoch"] == 450 or audit["stop_wait"] > 200)
    if group == GROUPS[3]:
        parity = saved.get("warmup_parity")
        if not parity or parity.get("status") != "PASS" or audit.get("warmup_parity") != parity:
            raise RuntimeError(f"G4 fold {fold} lacks matching G3/G4 warmup parity evidence")
        g3_best = ROOT / GROUPS[2] / "best_checkpoints" / f"fold{fold:02d}.pt"
        g3_latest = ROOT / GROUPS[2] / "latest_checkpoints" / f"fold{fold:02d}.pt"
        if parity.get("g3_best_sha256") != sha_file(g3_best) or parity.get("g3_latest_sha256") != sha_file(g3_latest):
            raise RuntimeError(f"G4 fold {fold} warmup evidence points to different G3 checkpoints")
        g3_snapshot = torch.load(g3_best, map_location="cpu", weights_only=False)
        if saved.get("warmup") != g3_snapshot.get("warmup"):
            raise RuntimeError(f"G4 fold {fold} warmup payload differs from its G3 reference")
        if not torch.equal(saved["relation_pairs"], g3_snapshot["relation_pairs"]):
            raise RuntimeError(f"G4 fold {fold} relation pairs differ from its G3 reference")
        if not torch.equal(saved["relation_targets"], g3_snapshot["relation_targets"]):
            raise RuntimeError(f"G4 fold {fold} relation targets differ from its G3 reference")
    seed_everything()
    model = build_model(group, saved["spectral_stats"], device, fold=fold).eval()
    assert model.group == group and type(model).__name__ == saved["model_type"]
    assert model.cam.use_pen == (group == GROUPS[0])
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).long().to(device)
    rows = torch.from_numpy(tr).to(device)
    pair, same = saved["relation_pairs"], saved["relation_targets"]
    if model.use_darg:
        assert pair.shape == (4096, 2) and int(same.sum()) == 2048
        assert np.isin(pair.numpy(), tr).all()
        np.testing.assert_array_equal(same.numpy(), y_np[pair[:, 0]] == y_np[pair[:, 1]])
        expected_pairs, expected_same = make_pairs(tr, y_np, fold, anchor_pair_support(model, x))
        np.testing.assert_array_equal(pair.numpy(), expected_pairs)
        np.testing.assert_array_equal(same.numpy(), expected_same)
        assert model.gfr.fallback_modal.tolist() == saved["warmup"]["fallback_modal"]
        assert sha_object((pair, same)) == saved["warmup"]["pair_sha256"]
        pair, same = pair.to(device), same.to(device)
    else:
        assert pair is None and same is None and saved["warmup"] is None
    model.load_state_dict(saved["model"], strict=True)
    logp, aux = model(x, y, rows, pair, same)
    probabilities = logp[te].exp().cpu().numpy()
    prediction = logp[te].argmax(1).cpu().numpy().astype(np.int64)
    metric = acc_auc(y_np[te], prediction, 3)
    observed_sha = hashlib.sha256(prediction.tobytes()).hexdigest()
    assert audit["prediction_sha256"] == saved["prediction_sha256"] == observed_sha
    assert metric == saved["metrics_at_best"] == {"ACC": audit["ACC"], "AUC": audit["AUC"]}
    assert abs(metric["ACC"] - saved["best_val_acc"]) <= 1e-12
    table = pd.read_csv(prediction_path)
    assert (table["fold"] == fold).all() and (table["seed"] == expected["seed"]).all()
    np.testing.assert_array_equal(table["sample_index"], te)
    np.testing.assert_array_equal(table["sample_id"], ids.iloc[te]["RID"])
    np.testing.assert_array_equal(table["true_label"], y_np[te])
    np.testing.assert_array_equal(table["pred_label"], prediction)
    np.testing.assert_allclose(table[["prob_NC", "prob_MCI", "prob_AD"]], probabilities, rtol=1e-6, atol=1e-7)
    mechanism = {}
    if model.use_darg:
        direct_logp, direct_off = model(x, y, rows, pair, same, darg_intervention="direct_off")
        neutral_logp, neutral = model(x, y, rows, pair, same, darg_intervention="disease_neutral")
        mechanism["darg"] = _checked_graph(aux, direct_off, neutral, logp, neutral_logp)
        assert mechanism["darg"]["active"], "The Best checkpoint has no signed graph effect"
        mechanism["darg"]["direct_off_logp_max_abs"] = float((logp - direct_logp).abs().max())
        truth = y[te]
        full_prediction, direct_prediction = logp[te].argmax(1), direct_logp[te].argmax(1)
        mechanism["darg"]["direct_off_changed_patients"] = int((full_prediction != direct_prediction).sum())
        mechanism["darg"]["direct_off_corrected"] = int(((full_prediction == truth) & (direct_prediction != truth)).sum())
        mechanism["darg"]["direct_off_damaged"] = int(((full_prediction != truth) & (direct_prediction == truth)).sum())
        neutral_prediction = neutral_logp[te].argmax(1)
        mechanism["darg"]["disease_neutral_changed_patients"] = int((full_prediction != neutral_prediction).sum())
        mechanism["darg"]["disease_neutral_corrected"] = int(((full_prediction == truth) & (neutral_prediction != truth)).sum())
        mechanism["darg"]["disease_neutral_damaged"] = int(((full_prediction != truth) & (neutral_prediction == truth)).sum())
    if model.use_dpsl:
        mechanism["dpsl"] = _checked_scmr(model, saved, rows, te, y, x, pair, same, aux, logp)
        native_test_prediction = np.asarray(mechanism["dpsl"].pop("native_test_prediction"), dtype=np.int64)
        native_test_probability = np.asarray(mechanism["dpsl"].pop("native_test_probability"), dtype=np.float64)
        required_native_columns = ["native_pred_label", "native_prob_NC", "native_prob_MCI", "native_prob_AD"]
        if not set(required_native_columns) <= set(table.columns):
            raise RuntimeError("G4 Best prediction CSV is missing its Native diagnostic")
        np.testing.assert_array_equal(table["native_pred_label"].to_numpy(np.int64), native_test_prediction)
        np.testing.assert_allclose(table[["native_prob_NC", "native_prob_MCI", "native_prob_AD"]],
                                   native_test_probability, rtol=1e-6, atol=1e-7)
        native_sha = hashlib.sha256(native_test_prediction.tobytes()).hexdigest()
        if native_sha != audit.get("native_prediction_sha256"):
            raise RuntimeError("G4 Native prediction differs from its training audit")
        native_metric = acc_auc(y_np[te], native_test_prediction, 3)
        if native_metric != audit.get("native_metrics"):
            raise RuntimeError("G4 Native metrics differ from its training audit")
        mechanism["dpsl"]["native_prediction_sha256"] = native_sha
        g3_path = ROOT / GROUPS[2] / "predictions" / f"fold{fold:02d}.csv"
        g3_table = pd.read_csv(g3_path)
        np.testing.assert_array_equal(g3_table["sample_index"].to_numpy(np.int64), te)
        g3_prediction = g3_table["pred_label"].to_numpy(np.int64)
        truth_np = y_np[te]
        g3_correct = g3_prediction == truth_np
        native_correct = native_test_prediction == truth_np
        mechanism["dpsl"]["native_vs_G3_full"] = {
            "native_correct": int(native_correct.sum()),
            "G3_full_correct": int(g3_correct.sum()),
            "corrected_by_G4_joint_training": int((native_correct & ~g3_correct).sum()),
            "damaged_by_G4_joint_training": int((g3_correct & ~native_correct).sum()),
            "net_correct": int(native_correct.sum() - g3_correct.sum()),
            "class_net_correct": {
                str(class_id): int(native_correct[truth_np == class_id].sum()
                                   - g3_correct[truth_np == class_id].sum())
                for class_id in range(3)},
        }
        assert mechanism["dpsl"]["active"], "The Best checkpoint has no SCMR prediction effect"
    return {**audit, "status": "PASS", "mechanism": mechanism}


def main():
    require_formal_training_gate()
    configure_worker()
    data = load_data()
    inherited_path = ROOT / "audits/inherited_replay.json"
    if not inherited_path.is_file():
        raise RuntimeError("Inherited E83.1 replay evidence is required before the E83.3 combined audit")
    inherited = json.loads(inherited_path.read_text())
    if inherited.get("status") != "PASS" or inherited.get("replayed_folds") != 30:
        raise RuntimeError("Inherited E83.1 replay evidence is incomplete")
    rows = list(inherited["records"])
    for group in ("G4_E82_4_6_10_G1",):
        for split in data[-1]:
            try:
                rows.append(replay_fold(group, split, data))
            except Exception as exc:
                rows.append({"group": group, "fold": int(split["fold"]), "status": "FAIL",
                             "error": repr(exc), "traceback": traceback.format_exc(limit=8)})
    failures = [row for row in rows if row["status"] != "PASS"]
    result = {"status": "PASS" if len(rows) == 40 and not failures else "FAIL",
              "rows": len(rows), "passed": len(rows) - len(failures), "failed": failures,
              "inherited_replay": str(inherited_path), "new_replay_groups": ["G4_E82_4_6_10_G1"],
              "records": rows}
    stream = io.StringIO(newline="")
    fields = ["group", "fold", "status", "ACC", "AUC", "best_epoch", "checkpoint", "error"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(ROOT / "audits/checkpoint_replay.csv", lambda file: file.write(stream.getvalue().encode()))
    write_json(ROOT / "audits/checkpoint_replay.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
