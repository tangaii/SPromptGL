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
    return {"positive_edges": positive, "negative_edges": negative,
            "neutral_intervention_logp_max_abs": response, "active": positive + negative > 0 and response > 0}


def _checked_scmr(model, saved, rows, test, labels, x, pair, same, full, output):
    module = model.dpsl
    stats = saved["spectral_stats"]
    reference = stats["reference"]
    np.testing.assert_array_equal(reference["legal"], rows.cpu())
    np.testing.assert_array_equal(reference["labels"], labels[rows].cpu())
    refit = fit_spectral_statistics(saved["spectral_stats"]["warmup_cam"], labels.cpu(), rows.cpu(),
                                   fold=saved["fold"], warm_logits=saved["spectral_stats"]["warmup_logits"])
    for key in ("basis", "center", "variance", "log_variance", "disease_covariance",
                "bank_index", "hard_initial"):
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
    for name in ("gfr", "darg", "cam", "head"):
        family = budget["families"][name]
        assert family["total_applied"] <= .001 * family["historical"]
        assert abs(family["applied_dot"]) <= family["dot_bound"]
    parts = full["dpsl"]
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
    pair = parts["class_pair"].detach().cpu()
    route = {f"{int(a)}-{int(b)}": int(((pair[:, 0] == a) & (pair[:, 1] == b)).sum())
             for a, b in ((2, 1), (2, 0), (1, 0))}
    return {"spectral_basis_shape": list(module.basis.shape), "train_only_reference": True,
            "spectral_basis_and_bank_refit_exact": True,
            "reconstructed_warmup_spectrum_and_calibration": True,
            "ema_updates": module.ema_updates, "shared_gradient_cap": budget["shared_cap"],
            "max_residual_ratio": float(parts["ratio"].max()), "nonzero_residual_rows": nonzero,
            "residual_off_logp_max_abs": change, "zero_residual_exact": True,
            "full_corrected": corrected, "full_damaged": damaged,
            "full_net_correct": corrected - damaged, "top2_route_counts": route,
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
    if model.use_dpsl:
        mechanism["dpsl"] = _checked_scmr(model, saved, rows, te, y, x, pair, same, aux, logp)
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
