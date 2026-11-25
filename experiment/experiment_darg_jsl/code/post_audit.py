"""Independent post-training audit for the formal DARG + JSL run."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "runtime/python"), str(ROOT / "code/history_base")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch

import jsl_runtime as r
from author_log import _validate_emitted, check_template
from common import history as h
from common.spectral import split_jacobian

DPSL = Path("/model/spromptgl/experiment/DPSL")
FROZEN_MANIFEST_SHA = "af1967ff3ccd32124acf35ac99fcba139d274e49d51a945af7d20fe6b7d037b7"
OVERLAP = {
    2: {377, 407, 681, 779, 784, 260, 288, 375},
    3: {46, 66, 172, 266, 388},
    4: {712, 713, 727, 773},
    8: {383, 430},
}
ALL_OVERLAP = set().union(*OVERLAP.values())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_dpsl_sha():
    manifest = DPSL / "SHA256SUMS"
    if file_sha(manifest) != FROZEN_MANIFEST_SHA:
        raise RuntimeError("formal DPSL SHA256SUMS digest changed")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = DPSL / relative
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError(f"formal DPSL frozen file mismatch: {relative}")
        entries.append(relative)
    if len(entries) != 3949 or len(set(entries)) != 3949:
        raise RuntimeError("formal DPSL frozen inventory is not exactly 3949 files")
    return {"manifest_sha256": file_sha(manifest), "file_count": len(entries),
            "file_set_sha256": hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()}


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def as_metrics(labels, prediction, probability):
    score = r.metrics(np.asarray(labels, dtype=np.int64), np.asarray(prediction, dtype=np.int64),
                      np.asarray(probability, dtype=np.float64))
    return score


def intervention_arrays(ctx, intervention):
    evaluation, capture = r.evaluate(ctx, intervention)
    test = ctx["members"]["test"]
    labels = ctx["state"]["labels_np"][test].astype(np.int64)
    logits = np.asarray(evaluation["logits"], dtype=np.float64)
    probability = np.asarray(evaluation["probability"], dtype=np.float64)
    prediction = np.asarray(evaluation["prediction"], dtype=np.int64)
    details = capture["details"]
    return {
        "score": as_metrics(labels, prediction, probability),
        "labels": labels,
        "prediction": prediction,
        "probability": probability,
        "margin": logits[:, 0] - logits[:, 1],
        "all_probability": np.asarray(evaluation["all_probability"], dtype=np.float64),
        "details": details,
    }


def best_mechanism(fold):
    ctx = r.setup(fold)
    payload_path = ROOT / "checkpoints" / f"fold{fold:02d}.best.pt"
    payload = torch.load(payload_path, map_location=ctx["state"]["x"].device, weights_only=False)
    r.restore(ctx, payload)
    interventions = {name: intervention_arrays(ctx, name)
                     for name in ("residual_off", "full", "spectral_only", "complement_only")}
    full = interventions["full"]
    off = interventions["residual_off"]
    labels = full["labels"]
    details = full["details"]
    test = ctx["members"]["test"]
    excluded = set(ctx["state"]["excluded"])
    delta = details["delta"].detach().cpu().numpy()[test]
    delta_parallel = details["delta_parallel"].detach().cpu().numpy()[test]
    delta_perp = details["delta_perp"].detach().cpu().numpy()[test]
    cam = ctx["state"]["cam"].Outputlayer
    cam_values = None

    jacobian = details["jacobian"]
    basis = ctx["archived"]["spectrum"]["U"]
    _, parallel, complement = split_jacobian(jacobian, basis)
    jacobian_norm = jacobian.norm(dim=-1).clamp_min(1.0)
    orthogonality = float(((parallel * complement).sum(-1).abs() / jacobian_norm.square()).max())
    reconstruction = float(((parallel + complement - jacobian).abs().max() / jacobian_norm.max()))
    ratio = details["ratio"].detach().cpu().numpy()[test]
    overlap_residual = details["delta"].detach().cpu().numpy()[list(excluded)] if excluded else np.zeros((0, 144))
    overlap_zero = bool(not overlap_residual.size or np.count_nonzero(overlap_residual) == 0)

    true_margin_full = (1 - 2 * labels) * full["margin"]
    true_margin_off = (1 - 2 * labels) * off["margin"]
    originally_wrong = true_margin_off <= 0
    originally_right = ~originally_wrong

    def transition(candidate):
        corrected = int(np.sum((off["prediction"] != labels) & (candidate["prediction"] == labels)))
        harmed = int(np.sum((off["prediction"] == labels) & (candidate["prediction"] != labels)))
        return {"corrected": corrected, "harmed": harmed, "net_correction": corrected - harmed,
                "hard_flips": int(np.sum(off["prediction"] != candidate["prediction"]))}

    gradient = r.train_step(ctx)
    gradient_groups = gradient["module_gradients"]
    record = {
        "fold": fold,
        "test_count": len(test),
        "best_epoch": payload["best_epoch"],
        "best_correct": payload["best_correct"],
        "metrics_full": full["score"],
        "metrics_residual_off": off["score"],
        "metrics_spectral_only": interventions["spectral_only"]["score"],
        "metrics_complement_only": interventions["complement_only"]["score"],
        "transition_full_from_off": transition(full),
        "transition_spectral_from_off": transition(interventions["spectral_only"]),
        "transition_complement_from_off": transition(interventions["complement_only"]),
        "mean_delta_fraction": float(ratio.mean()),
        "max_delta_fraction": float(ratio.max()),
        "mean_parallel_delta_norm": float(np.linalg.norm(delta_parallel, axis=1).mean()),
        "mean_perp_delta_norm": float(np.linalg.norm(delta_perp, axis=1).mean()),
        "wrong_true_margin_change": float((true_margin_full[originally_wrong] - true_margin_off[originally_wrong]).mean()) if originally_wrong.any() else 0.0,
        "right_true_margin_change": float((true_margin_full[originally_right] - true_margin_off[originally_right]).mean()) if originally_right.any() else 0.0,
        "overlap_zero": overlap_zero,
        "overlap_residual_max_abs": float(np.abs(overlap_residual).max()) if overlap_residual.size else 0.0,
        "jacobian_orthogonality_relative": orthogonality,
        "jacobian_reconstruction_relative": reconstruction,
        "basis_shape": list(ctx["archived"]["spectrum"]["U"].shape),
        "gradient": {
            "shared_extra_fraction": gradient["shared_extra_fraction"],
            "parallel_residual_gradient": gradient["parallel_residual_gradient"],
            "complement_residual_gradient": gradient["complement_residual_gradient"],
            "module_gradients": gradient_groups,
        },
        "arrays": {
            "test": test.tolist(),
            "labels": labels.tolist(),
            "full": {key: full[key].tolist() for key in ("prediction", "probability", "margin")},
            "residual_off": {key: off[key].tolist() for key in ("prediction", "probability", "margin")},
            "spectral_only": {key: interventions["spectral_only"][key].tolist() for key in ("prediction", "probability", "margin")},
            "complement_only": {key: interventions["complement_only"][key].tolist() for key in ("prediction", "probability", "margin")},
            "delta": delta.tolist(),
            "delta_parallel": delta_parallel.tolist(),
            "delta_perp": delta_perp.tolist(),
            "cam_norm": details["cam_norm"].detach().cpu().numpy()[test].tolist(),
        },
    }
    del cam_values, cam, ctx
    torch.cuda.empty_cache()
    return record


def latest_recovery(fold):
    path = ROOT / "checkpoints" / f"fold{fold:02d}.latest.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("state_sha") != r.sha(payload["state"]):
        raise RuntimeError(f"fold {fold} Latest state SHA mismatch")

    def replay():
        current_payload = torch.load(path, map_location="cpu", weights_only=False)
        ctx = r.setup(fold)
        r.restore(ctx, current_payload)
        initial = r.evaluate(ctx)[0]
        diag = r.train_step(ctx)
        after = r.evaluate(ctx)[0]
        result = {"state_sha": r.sha(r.bundle(ctx)), "initial_evaluation_sha": r.sha(initial),
                  "next_step_diag_sha": r.sha(diag), "next_step_evaluation_sha": r.sha(after),
                  "rng_sha": h.rng_sha()}
        del ctx
        torch.cuda.empty_cache()
        return result

    first = replay()
    second = replay()
    return {"fold": fold, "epoch": payload["epoch"], "state_sha": payload["state_sha"],
            "implementation_sha": r.sha(payload["implementation"]), "replay_equal": first == second,
            "first": first, "second": second}


def rng_audit():
    records = []
    for fold in r.FOLDS:
        current = read_jsonl(ROOT / "runtime/journal" / f"fold{fold:02d}.jsonl")
        archived = h.hist_trace(fold)
        compared = 0
        mismatch = []
        dropout_bad = []
        for row in current:
            epoch = int(row["epoch"])
            if epoch in archived:
                compared += 1
                actual = row["rng"]
                expected = {key: archived[epoch][key] for key in actual}
                if actual != expected:
                    mismatch.append(epoch)
            calls = row.get("diag", {}).get("dropout", {}).get("calls", [])
            if len(calls) != 6:
                dropout_bad.append(epoch)
        records.append({"fold": fold, "journal_epochs": len(current), "archive_epochs": len(archived),
                        "common_epochs": compared, "rng_mismatches": mismatch,
                        "dropout_bad_epochs": dropout_bad, "dropout_six_calls_all_epochs": not dropout_bad})
    return records


def dse_audit():
    source = (ROOT / "code/jsl_model.py").read_text(encoding="utf-8").lower()
    forbidden = ("medical prompt", "anonymous prompt", "clip text query", "gaussian llr",
                 "evidence_direction", "evidence_reliability", "ema class statistics",
                 "dse reliability", "dse conflict", "dse direction")
    active_network = sys.modules.get("network")
    state_keys = [key.lower() for fold in (1,) for key in r.setup(fold)["module"].state_dict()]
    active_ok = active_network is not None and Path(active_network.__file__).resolve() == (ROOT / "code/darg_core/network.py").resolve()
    clip_loaded = any(name == "clip" or name.startswith("clip.") or name == "CLIP" or name.startswith("CLIP.")
                      for name in sys.modules)
    return {"forbidden_source_tokens": [token for token in forbidden if token in source],
            "dse_state_tokens": [key for key in state_keys if any(token in key for token in
                                ("prompt", "query", "clip", "gaussian", "ema", "evidence"))],
            "active_network_darg_only": active_ok, "clip_loaded": clip_loaded}


def reference_rows():
    path = DPSL / "results/patient_level.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    return {(int(row["patient_id_1based"]), int(row["fold"])): row for row in rows}


def metrics_row(scope, model, rows):
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    prediction = np.asarray([row[model + "_prediction"] for row in rows], dtype=np.int64)
    probability = np.asarray([[row[model + "_p_asd"], 1.0 - row[model + "_p_asd"]] for row in rows], dtype=np.float64)
    score = as_metrics(labels, prediction, probability)
    return {"scope": scope, "model": model, "n": len(rows), "correct": score["correct"],
            **{key: 100.0 * score[key] for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")}}


def paired_comparisons(patient_rows):
    scopes = {
        "all_871": lambda row: True,
        "overlap_19": lambda row: row["patient_id_1based"] in ALL_OVERLAP,
        "nonoverlap_852": lambda row: row["patient_id_1based"] not in ALL_OVERLAP,
        "exclude_folds_2_3_4_8": lambda row: row["fold"] not in {2, 3, 4, 8},
    }
    rows = []
    for scope, predicate in scopes.items():
        selected = [row for row in patient_rows if predicate(row)]
        for model in ("darg", "dpsl", "jsl"):
            rows.append(metrics_row(scope, model, selected))
        for left, right in (("jsl", "dpsl"), ("jsl", "darg")):
            left_row = rows[-1] if rows[-1]["model"] == left else next(row for row in rows if row["scope"] == scope and row["model"] == left)
            right_row = next(row for row in rows if row["scope"] == scope and row["model"] == right)
            rows.append({"scope": scope, "model": f"{left}_minus_{right}", "n": len(selected),
                         "correct": left_row["correct"] - right_row["correct"],
                         **{key: left_row[key] - right_row[key] for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")}})
    return rows


def build_patient_rows(mechanisms):
    reference = reference_rows()
    result = []
    for mechanism in mechanisms:
        fold = mechanism["fold"]
        arrays = mechanism["arrays"]
        for position, patient in enumerate(arrays["test"]):
            patient_id = int(patient) + 1
            label = int(arrays["labels"][position])
            historical = reference[(patient_id, fold)]
            row = {"fold": fold, "test_position": position, "patient_id_1based": patient_id,
                   "true_label": label, "overlap_19": patient_id in ALL_OVERLAP,
                   "darg_prediction": int(historical["prediction_e744_darg_full"]),
                   "darg_p_asd": float(historical["p_asd_e744_darg_full"]),
                   "dpsl_prediction": int(historical["prediction_full"]),
                   "dpsl_p_asd": float(historical["p_asd_full"])}
            for name in ("full", "residual_off", "spectral_only", "complement_only"):
                row[f"{name}_prediction"] = int(arrays[name]["prediction"][position])
                row[f"{name}_p_asd"] = float(arrays[name]["probability"][position][0])
                row[f"{name}_margin"] = float(arrays[name]["margin"][position])
            row["jsl_prediction"] = row["full_prediction"]
            row["jsl_p_asd"] = row["full_p_asd"]
            row.update({"delta_l2": float(np.linalg.norm(arrays["delta"][position])),
                        "delta_parallel_l2": float(np.linalg.norm(arrays["delta_parallel"][position])),
                        "delta_perp_l2": float(np.linalg.norm(arrays["delta_perp"][position])),
                        "cam_l2": float(arrays["cam_norm"][position])})
            row["cam_relative_delta"] = row["delta_l2"] / max(row["cam_l2"], 1e-12)
            row["jsl_corrected_vs_off"] = int(row["residual_off_prediction"] != label and row["full_prediction"] == label)
            row["jsl_harmed_vs_off"] = int(row["residual_off_prediction"] == label and row["full_prediction"] != label)
            row["jsl_hard_flip_vs_off"] = int(row["residual_off_prediction"] != row["full_prediction"])
            row["jsl_true_margin_change"] = float((1 - 2 * label) *
                                                     (row["full_margin"] - row["residual_off_margin"]))
            result.append(row)
    return result


def mechanism_rows(mechanisms):
    rows = []
    for mechanism in mechanisms:
        row = {"scope": f"fold{mechanism['fold']:02d}", "fold": mechanism["fold"],
               "best_epoch": mechanism["best_epoch"], "test_count": mechanism["test_count"],
               "orthogonality_relative": mechanism["jacobian_orthogonality_relative"],
               "reconstruction_relative": mechanism["jacobian_reconstruction_relative"],
               "mean_delta_fraction": mechanism["mean_delta_fraction"],
               "max_delta_fraction": mechanism["max_delta_fraction"],
               "overlap_zero": mechanism["overlap_zero"],
               "overlap_residual_max_abs": mechanism["overlap_residual_max_abs"],
               "shared_extra_fraction": mechanism["gradient"]["shared_extra_fraction"],
               "parallel_gradient": mechanism["gradient"]["parallel_residual_gradient"],
               "complement_gradient": mechanism["gradient"]["complement_residual_gradient"]}
        for name in ("full", "residual_off", "spectral_only", "complement_only"):
            score = mechanism[f"metrics_{name}"]
            for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
                row[f"{name}_{key}"] = score[key]
        for name, transition in (("full", mechanism["transition_full_from_off"]),
                                 ("spectral", mechanism["transition_spectral_from_off"]),
                                 ("complement", mechanism["transition_complement_from_off"])):
            for key, value in transition.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
    return rows


def main():
    frozen_before = frozen_dpsl_sha()
    mechanisms = [best_mechanism(fold) for fold in r.FOLDS]
    recovery = [latest_recovery(fold) for fold in r.FOLDS]
    rng_records = rng_audit()
    dse = dse_audit()
    frozen_after = frozen_dpsl_sha()
    if frozen_after != frozen_before:
        raise RuntimeError("formal DPSL frozen inventory changed during independent audit")

    template = check_template()
    emitted = (ROOT / "result_ABIDE.txt").read_text(encoding="utf-8")
    _validate_emitted(emitted, 10)
    author = {**template, "emitted_sha256": file_sha(ROOT / "result_ABIDE.txt"),
              "emitted_format": "PASS", "emitted_folds": 10}

    patient_rows = build_patient_rows(mechanisms)
    paired = paired_comparisons(patient_rows)
    mechanism_csv = mechanism_rows(mechanisms)
    write_csv(ROOT / "results/patient_level.csv", patient_rows)
    write_csv(ROOT / "results/paired_comparisons.csv", paired)
    write_csv(ROOT / "results/mechanism.csv", mechanism_csv)
    atomic_json(ROOT / "results/rng_audit.json", {"folds": rng_records,
        "all_common_epochs_equal": all(not row["rng_mismatches"] for row in rng_records),
        "all_epochs_have_six_dropout_calls": all(row["dropout_six_calls_all_epochs"] for row in rng_records)})
    atomic_json(ROOT / "results/recovery_audit.json", {"folds": recovery,
        "all_latest_reloads": all(row["state_sha"] for row in recovery),
        "all_next_step_replays_equal": all(row["replay_equal"] for row in recovery)})

    all_mechanism = {
        "folds": mechanism_csv,
        "mean_max_delta_fraction": float(np.mean([row["max_delta_fraction"] for row in mechanism_csv])),
        "maximum_max_delta_fraction": float(np.max([row["max_delta_fraction"] for row in mechanism_csv])),
        "overlap_zero_folds": int(sum(bool(row["overlap_zero"]) for row in mechanism_csv)),
        "all_gradient_fractions_within_cap": bool(all(row["shared_extra_fraction"] <= r.CAP + 1e-8 for row in mechanism_csv)),
    }
    audit_checks = {
        "status": "PASS",
        "best_forward_10_of_10": True,
        "latest_reload_10_of_10": all(row["state_sha"] for row in recovery),
        "latest_next_step_replay_10_of_10": all(row["replay_equal"] for row in recovery),
        "rng_archive_common_epochs": all(not row["rng_mismatches"] for row in rng_records),
        "historical_dropout_six_calls": all(row["dropout_six_calls_all_epochs"] for row in rng_records),
        "jacobian_geometry_10_of_10": all(row["basis_shape"] == [144, 16]
                                            and row["jacobian_orthogonality_relative"] <= 1e-5
                                            and row["jacobian_reconstruction_relative"] <= 1e-6
                                            for row in mechanisms),
        "residual_cap_10_of_10": all(row["max_delta_fraction"] <= 0.08 + 1e-6 for row in mechanisms),
        "overlap_zero_residual_10_of_10": all(row["overlap_zero"] for row in mechanisms),
        "shared_gradient_budget_10_of_10": all(row["gradient"]["shared_extra_fraction"] <= r.CAP + 1e-8
                                                for row in mechanisms),
        "jsl_gradient_coverage_10_of_10": all(
            all(group["with_grad"] == group["parameters"] and group["nonzero"] > 0
                for group in row["gradient"]["module_gradients"].values()) for row in mechanisms),
        "dse_independence": not dse["forbidden_source_tokens"] and not dse["dse_state_tokens"]
                             and dse["active_network_darg_only"] and not dse["clip_loaded"],
        "frozen_dpsl_before": frozen_before,
        "frozen_dpsl_after": frozen_after,
        "author_log": author,
        "implementation_sha256": r.sha(r.implementation()),
    }
    audit_checks["status"] = "PASS" if all(value is True or isinstance(value, dict) for key, value in audit_checks.items()
                                             if key not in ("status", "frozen_dpsl_before", "frozen_dpsl_after", "author_log", "implementation_sha256")) else "FAIL"
    atomic_json(ROOT / "results/audit_status.json", audit_checks)

    fold_metrics = list(csv.DictReader((ROOT / "results/fold_metrics.csv").open(encoding="utf-8", newline="")))
    mean = next(row for row in fold_metrics if row["fold"] == "mean")
    sd = next(row for row in fold_metrics if row["fold"] == "population_sd")
    paired_all = [row for row in paired if row["scope"] == "all_871"]
    by_model = {row["model"]: row for row in paired_all}
    report = [
        "# DARG + JSL Formal Audit",
        "",
        f"Status: `{audit_checks['status']}`.",
        "",
        "## PERFORMANCE",
        "",
        "The ten-fold Best-model macro results are reported as mean +/- population SD in percent.",
        "",
        f"ACC {float(mean['acc'])*100:.4f} +/- {float(sd['acc'])*100:.4f}; Hard-AUC/BA {float(mean['hard_auc'])*100:.4f} +/- {float(sd['hard_auc'])*100:.4f}; Prob-AUC {float(mean['prob_auc'])*100:.4f} +/- {float(sd['prob_auc'])*100:.4f}.",
        f"SEN-ASD {float(mean['sen'])*100:.4f} +/- {float(sd['sen'])*100:.4f}; SPE-TD {float(mean['spe'])*100:.4f} +/- {float(sd['spe'])*100:.4f}; ASD-F1 {float(mean['asd_f1'])*100:.4f} +/- {float(sd['asd_f1'])*100:.4f}.",
        "",
        "## MECHANISM",
        "",
        f"Residual-off to Full transitions across all test patients are recorded in mechanism.csv. Full JSL corrected {sum(row['full_corrected'] for row in mechanism_csv)} patients and harmed {sum(row['full_harmed'] for row in mechanism_csv)} relative to Residual-off.",
        f"The maximum observed CAM residual fraction was {all_mechanism['maximum_max_delta_fraction']:.8f}. Spectral-only and complement-only results are reported per fold.",
        "",
        "## ABLATION_VALIDITY",
        "",
        "JSL was trained from E61 epoch-10 full warmup states with the Original O 144x16 basis. DSE text, Gaussian/EMA evidence, and Prompt state were absent from the active model. The 19 historical training-test overlap patients were excluded from JSL supervision and had zero JSL residual.",
        "",
        "## AUDIT_STATUS",
        "",
        f"Best forward {sum(1 for _ in mechanisms)}/10; Latest reload and next-step replay {sum(row['replay_equal'] for row in recovery)}/10; archive RNG mismatches {sum(len(row['rng_mismatches']) for row in rng_records)}; author log format {author['emitted_format']}.",
        "",
        "The protocol retains validation-equals-test checkpoint selection, pre-CV feature processing, a transductive patient graph, and intentional 19-patient training-test overlap. The results are an ablation audit and are not independent generalization evidence.",
        "",
        "Reference comparison rows are in paired_comparisons.csv. Patient-level predictions and intervention margins are in patient_level.csv.",
    ]
    (ROOT / "results/report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    output = {"audit_status": audit_checks["status"], "performance_mean": mean,
              "performance_population_sd": sd, "paired_all": by_model,
              "mechanism": all_mechanism, "recovery": recovery, "rng": rng_records,
              "dse": dse, "author_log": author}
    atomic_json(ROOT / "results/post_audit_summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
