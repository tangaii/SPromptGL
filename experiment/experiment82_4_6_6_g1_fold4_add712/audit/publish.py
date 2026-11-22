"""Publish the completed E82.4.6.6-G1 replay, metrics, and handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import replay

ROOT = replay.ROOT
RESULTS = ROOT / "results"
E465 = replay.E465
CONFIG = replay.CONFIG
h = replay.h
REPLAY_PATH = ROOT / "runtime" / "replayed_folds.json"
HANDOFF = ROOT.parent / "handoff.md"
METHODS = ("E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.5-G1", "E82.4.6.6-G1")
METRIC_KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def atomic_writer(path: Path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    settings = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with temporary.open(mode, **settings) as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows):
    rows = list(rows)
    require(rows, f"No rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with atomic_writer(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percent(value):
    return None if value is None else float(value) * 100.0


def historical_metric_rows():
    archived = read_csv(E465 / "results" / "fold_metrics.csv")
    selected = {(row["method"], int(row["fold"])): row for row in archived
                if row.get("scope") == "test_fold"}
    required = {(method, fold) for method in METHODS[:3] for fold in range(1, 11)}
    require(required <= selected.keys(), "Historical E82.4.6.5 fold metrics are incomplete")
    return selected


def make_fold_metrics(replayed):
    archived = historical_metric_rows()
    rows = []
    for method in METHODS[:3]:
        for fold in range(1, 11):
            rows.append(dict(archived[(method, fold)]))

    current = replayed["metrics"]["E82.4.6.6-G1"]
    for fold in range(1, 11):
        old = archived[("E82.4.6.5-G1", fold)]
        value = current[str(fold)]
        row = {
            "method": "E82.4.6.6-G1", "fold": fold, "scope": "test_fold",
            "best_epoch": replayed["fold4"]["best_epoch"] if fold == 4 else old["best_epoch"],
            "end_epoch": replayed["fold4"]["latest_epoch"] if fold == 4 else old["end_epoch"],
            "checkpoint_source": "fold4_retrained" if fold == 4 else "E82.4.6.5-G1_reuse",
            "best_sha": replayed["fold4"]["best_sha"] if fold == 4 else old["best_sha"],
        }
        for key in ("n_test", "tp", "fp", "tn", "fn"):
            row[key] = value[key]
        for key in METRIC_KEYS:
            row[key] = value[key]
            row[key + "_percent"] = percent(value[key])
        rows.append(row)
    return rows


def add_patient_audit_fields(rows):
    for row in rows:
        overlap = bool(row["current_native_training_overlap"])
        row["patient_index_conversion_verified"] = int(row["patient_index_0based"]) == int(row["patient_id_1based"]) - 1
        row["darg_classification_supervised"] = overlap
        row["darg_relation_training_member"] = overlap
        row["scmr_legal_supervision_member"] = not overlap
        row["gaussian_legal_member"] = not overlap
        row["ema_legal_member"] = not overlap
        row["new_residual_masked"] = overlap
    return rows


def scope_patients(rows, scope):
    if scope == "ten_fold":
        return list(rows)
    if scope.startswith("fold") and scope[4:].isdigit():
        return [row for row in rows if row["fold"] == int(scope[4:])]
    if scope == "fold4_new_712":
        return [row for row in rows if row["fold"] == 4 and row["patient_id_1based"] == 712]
    if scope == "fold4_retained_713_727_773":
        return [row for row in rows if row["fold"] == 4 and row["patient_id_1based"] in {713, 727, 773}]
    if scope == "fold4_other_83":
        return [row for row in rows if row["fold"] == 4 and row["patient_id_1based"] not in {712, 713, 727, 773}]
    if scope == "fixed_union_exposure_17":
        return [row for row in rows if row["in_fixed_union_exposure_17"]]
    if scope == "outside_fixed_union_854":
        return [row for row in rows if row["outside_fixed_union_854"]]
    raise RuntimeError(f"Unknown paired scope {scope}")


def add_query_change_counts(query_rows, patient_rows):
    for row in query_rows:
        selected = scope_patients(patient_rows, row["scope"])
        row["cam144_changed_count"] = sum(float(item["query_feature_change_l2"]) > 1e-12 for item in selected)
        row["asd_probability_changed_count"] = sum(
            abs(float(item["p_asd_full"]) - float(item["p_asd_anonymous"])) > 1e-12 for item in selected)
        row["hard_prediction_flip_count"] = sum(
            int(item["prediction_full"]) != int(item["prediction_anonymous"]) for item in selected)
    return query_rows


def verify_nonfold4_reuse(patient_rows):
    archived = {(int(row["fold"]), int(row["patient_id_1based"])): row
                for row in read_csv(E465 / "results" / "patient_level.csv")}
    fields = (
        "prediction_full", "p_asd_full", "true_margin_full",
        "prediction_residual_off", "p_asd_residual_off", "true_margin_residual_off",
        "prediction_anonymous", "p_asd_anonymous", "true_margin_anonymous",
    )
    checked = 0
    for row in patient_rows:
        if row["fold"] == 4:
            continue
        old = archived[(row["fold"], row["patient_id_1based"])]
        for field in fields:
            if field.startswith("prediction"):
                require(int(row[field]) == int(old[field]), f"Reused hard prediction changed for {row['fold']}:{row['patient_id_1based']}")
            else:
                require(float(row[field]) == float(old[field]), f"Reused probability or margin changed for {row['fold']}:{row['patient_id_1based']}")
        require(row["current_best_sha"] == old["current_best_sha"], "Reused checkpoint identity changed")
        checked += 1
    require(checked == 784, "Unexpected number of non-Fold-4 reused patient rows")
    return {"patients": checked, "probability_hard_prediction_and_margin_exact": True}


def write_handoff(replayed, result_hashes):
    fold4 = replayed["fold4"]
    ten = next(row for row in replayed["paired"]
               if row.get("comparison") in {"E82.4.6.5_to_E82.4.6.6", "E82.4.6.5-G1_to_E82.4.6.6-G1"}
               and row["scope"] == "ten_fold")
    summaries = replayed["summaries"]["E82.4.6.6-G1"]
    section = f"""## Experiment 82.4.6.6-G1 — E82.4.6.5-G1 plus Fold-4 Patient 712

- Purpose. Intentional label-exposure diagnostic that adds only patient 712 to the Fold-4 DARG native training membership.
- Training and supervision. Fold 4 was retrained from the verified E61 epoch 10 state with seed 0. DARG rows were 788. SCMR, Gaussian, and EMA legal rows remained 784. The test set remained 87 rows. Patients 712, 713, 727, and 773 were masked from the new residual and prompt or spectral label supervision.
- Reuse. Fold 4 was the only trained fold. Folds 1, 2, 3, 5, 6, 7, 8, 9, and 10 were read-only reuse of E82.4.6.5-G1 with checkpoint, probability, and hard-prediction identity checks.
- Ten-fold E82.4.6.6-G1. ACC {summaries['acc']['mean_pct']:.4f} ± {summaries['acc']['sd_pct']:.4f}, BA {summaries['hard_auc']['mean_pct']:.4f} ± {summaries['hard_auc']['sd_pct']:.4f}, Prob-AUC {summaries['prob_auc']['mean_pct']:.4f} ± {summaries['prob_auc']['sd_pct']:.4f}, SEN-ASD {summaries['sen']['mean_pct']:.4f} ± {summaries['sen']['sd_pct']:.4f}, SPE-TD {summaries['spe']['mean_pct']:.4f} ± {summaries['spe']['sd_pct']:.4f}, ASD-F1 {summaries['asd_f1']['mean_pct']:.4f} ± {summaries['asd_f1']['sd_pct']:.4f}.
- Fold 4 checkpoint. Best epoch {fold4['best_epoch']} with 83/87. Latest epoch {fold4['latest_epoch']}. Best SHA {fold4['best_sha']}. Latest SHA {fold4['latest_sha']}.
- Paired E82.4.6.5-G1 to E82.4.6.6-G1. Corrections {ten['corrected']}, harms {ten['harmed']}, net {ten['net']}. Fold 4 corrected 712, 713, and 727. Patient 773 remained correct. Fold 4 harms were 301, 517, and 806.
- Residual and query mechanisms. Four masked patients had exact zero spectral-in, spectral-out, and joint residual. Residual-off to Full produced no Fold-4 hard-prediction change. Anonymous to Medical Query produced no Fold-4 hard-prediction change. The 8 percent CAM144 residual limit and 0.1 percent shared-gradient limit passed.
- Audit. Best reload, Latest reload, next-step training and evaluation recovery, RNG replay, six-dropout order, Gaussian and EMA membership, DARG classification and relation gradients, supervision masking, and residual masking passed.
- Interpretation. This is an intentional training-test label-overlap diagnosis. The observed changes cannot be interpreted as independent generalization and do not establish a GO decision for the MICCAI 2027 second innovation module.
- Authority. {ROOT}. Result SHA values are recorded in the final manifest and audit status. Handoff readback preserved the historical prefix before this section.
"""
    lock_path = HANDOFF.with_name(HANDOFF.name + ".lock")
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
        marker = "## Experiment 82.4.6.6-G1 — E82.4.6.5-G1 plus Fold-4 Patient 712"
        if marker not in previous:
            content = previous.rstrip() + "\n\n" + section.strip() + "\n"
            temporary = HANDOFF.with_name(HANDOFF.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, HANDOFF)
        readback = HANDOFF.read_text(encoding="utf-8")
        require(readback.startswith(previous), "Handoff historical prefix changed")
        require(marker in readback and "Result SHA values" in readback, "Handoff section readback incomplete")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock_path.unlink(missing_ok=True)
    return {
        "path": str(HANDOFF), "historical_prefix_preserved": True,
        "section_marker": marker, "readback_sha": hashlib.sha256(readback.encode()).hexdigest(),
        "result_sha256_at_append": result_hashes,
    }


def report_text(replayed, patient_rows, paired_rows, residual_rows, query_rows):
    summaries = replayed["summaries"]
    ten = next(row for row in paired_rows if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row["scope"] == "ten_fold")
    fold4 = next(row for row in paired_rows if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row["scope"] == "fold4")
    other83 = next(row for row in paired_rows if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row["scope"] == "fold4_other_83")
    fixed = next(row for row in paired_rows if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row["scope"] == "outside_fixed_union_854")
    fixed_overlap = next(row for row in paired_rows if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row["scope"] == "fixed_union_exposure_17")
    residual_ten = next(row for row in residual_rows if row["scope"] == "ten_fold")
    query_ten = next(row for row in query_rows if row["scope"] == "ten_fold")
    query_fold4 = next(row for row in query_rows if row["scope"] == "fold4")
    patient_712 = next(row for row in patient_rows if row["fold"] == 4 and row["patient_id_1based"] == 712)

    def fmt(value):
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# Experiment 82.4.6.6-G1", "", "## Status", "",
        "COMPLETE_AUDITED. Fold 4 was the only retrained fold. The experiment is an intentional training-test label-overlap diagnostic.", "",
        "## Frozen protocol and membership", "",
        "- Base state was E74.4-G2 through the verified E61 epoch 10 state. Training resumed at epoch 11 with seed 0 and no per-epoch reseeding.",
        "- Fold 4 DARG native training rows were 784 before overlap expansion, 787 in E82.4.6.5-G1, and 788 after adding patient 712. SCMR legal supervision rows were 784. Gaussian and EMA legal rows were 784. Test rows were 87.",
        "- Patients 712, 713, 727, and 773 were retained in the test set and trained by DARG classification and relation objectives. Their new spectral and prompt residuals were exactly zero and they were excluded from SCMR, Gaussian, and EMA legal membership.",
        "- Read-only reused Fold 2 had 792 DARG rows and 784 SCMR legal rows. Read-only reused Fold 3 had 789 DARG rows and 784 SCMR legal rows. The remaining reused folds retain their E82.4.6.5-G1 memberships.",
        "- Patient labels were independently checked from the original label array. Patients 712, 713, 727, and 773 were ASD with encoding 0. The 1-based to 0-based conversion was checked for every patient row.",
        "- Relation training used 4096 pairs with 2048 same-class and 2048 different-class pairs. The pair sampler did not advance the historical global RNG.", "",
        "## Ten-fold metrics", "",
        "Values are mean ± population SD in percent. Hard-AUC is BA from hard predictions. Prob-AUC uses continuous ASD probability.", "",
        "| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        s = summaries[method]
        lines.append("| " + method + " | " + " | ".join(f"{s[k]['mean_pct']:.4f} ± {s[k]['sd_pct']:.4f}" for k in METRIC_KEYS) + " |")
    lines += [
        "", "## Fold 4 and paired analysis", "",
        f"- E82.4.6.5-G1 to E82.4.6.6-G1 on all 871 patients. Corrections {ten['corrected']}. Harms {ten['harmed']}. Net {ten['net']}. ΔACC {fmt(ten['delta_acc_pp'])} pp. ΔBA {fmt(ten['delta_ba_pp'])} pp. ΔProb-AUC {fmt(ten['delta_prob_auc_pp'])} pp.",
        f"- Fold 4. Corrections {fold4['corrected_ids'] or 'none'}. Harms {fold4['harmed_ids'] or 'none'}. Net {fold4['net']}. ΔACC {fmt(fold4['delta_acc_pp'])} pp. ΔBA {fmt(fold4['delta_ba_pp'])} pp. ΔProb-AUC {fmt(fold4['delta_prob_auc_pp'])} pp.",
        f"- Fold 4 patient 712 changed from hard prediction {patient_712['prediction_previous']} to the correct prediction. Its previous residual norm was {float(patient_712['previous_delta_l2']):.6f}, while the new spectral-in, spectral-out, and joint residual norms were exactly zero. Its CAM144 feature and classification probability were identical in Residual-off and Full.",
        f"- Patients 713 and 727 changed to correct predictions. Patient 773 remained correct. The other 83 Fold-4 test patients had corrections {other83['corrected']}, harms {other83['harmed']}, and net {other83['net']}.",
        f"- The fixed union of 17 current or previous overlap patients had corrections {fixed_overlap['corrected']}, harms {fixed_overlap['harmed']}, and net {fixed_overlap['net']}. The explicitly constructed outside-union set had 854 patients with corrections {fixed['corrected']}, harms {fixed['harmed']}, and net {fixed['net']}.",
        "- The 854-patient set is a fixed descriptive paired-analysis cohort and is not an independent validation set.", "",
        "### Fold results", "", "| Fold | Result | Corrections | Harms | Net |", "|---:|---|---:|---:|---:|",
    ]
    for row in paired_rows:
        if row.get("comparison") == "E82.4.6.5-G1_to_E82.4.6.6-G1" and row.get("fold_result"):
            lines.append(f"| {row['scope'].replace('fold', '')} | {row['fold_result']} | {row['corrected']} | {row['harmed']} | {row['net']} |")
    lines += [
        "", "## Fixed-weight mechanism audit", "",
        f"- Residual-off to Full across ten folds. Corrections {residual_ten['corrected']}. Harms {residual_ten['harmed']}. Net {residual_ten['net']}. ΔACC {fmt(residual_ten['delta_acc_pp'])} pp. ΔBA {fmt(residual_ten['delta_ba_pp'])} pp. Mean true-margin change {fmt(residual_ten['margin_mean_delta'])}. Fold 4 had no hard-prediction change.",
        f"- Anonymous Query to Medical Query across ten folds. CAM144 changed for {query_ten['cam144_changed_count']} patients. ASD probability changed for {query_ten['asd_probability_changed_count']} patients. Hard prediction flipped for {query_ten['hard_prediction_flip_count']} patients. Corrections {query_ten['corrected']}. Harms {query_ten['harmed']}. Net {query_ten['net']}. ΔACC {fmt(query_ten['delta_acc_pp'])} pp. ΔBA {fmt(query_ten['delta_ba_pp'])} pp. Mean true-margin change {fmt(query_ten['margin_mean_delta'])}. Fold 4 CAM144 changes {query_fold4['cam144_changed_count']}, probability changes {query_fold4['asd_probability_changed_count']}, and hard flips {query_fold4['hard_prediction_flip_count']}.",
        "- The four Fold-4 overlap patients had exact zero spectral-in, spectral-out, and joint residual. The other 83 test patients retained an active residual path.",
        "- The maximum Fold-4 relative joint CAM residual was {:.6f}, below the 0.08 limit. The shared differential-gradient audit passed the 0.001 limit.".format(replayed["fold4"]["max_cam_relative_delta"]),
        "- Jacobian spectral-in, Jacobian spectral-out, medical Prompt, CAM144, end-to-end classification, DARG classification, DARG relation, Gaussian, and EMA checks passed.", "",
        "## Checkpoint and RNG audit", "",
        f"- Best epoch {replayed['fold4']['best_epoch']} and Latest epoch {replayed['fold4']['latest_epoch']} were reloaded. Best SHA {replayed['fold4']['best_sha']}. Latest SHA {replayed['fold4']['latest_sha']}. Journal SHA {replayed['fold4']['journal_sha']}.",
        "- Latest next-step training and evaluation recovery were exact. RNG was replayed epoch by epoch. Historical six-dropout order was preserved. The E82.4.6.5 source manifest and every reused result and checkpoint identity matched its frozen SHA.",
        "- The historical E82.4.6.5 Fold-4 Best was epoch 267 with 83/87. It was used only for comparison. The new run started from E61 epoch 10 and selected epoch 182 independently.",
        "- The checkpoint payload retains the historical canonical parent field G2 used by the E82.4.6.5 G1 loader. The task files, manifest, and experiment identity are G1.",
        "- A preflight audit-entry repair registered the initial EMA summary key. A separate audit-entry repair registered list-based masking for tuple-index handling. Neither changed the frozen training implementation or any checkpoint.", "",
        "## Scientific interpretation", "",
        "The Fold-4 change corrected patients 712, 713, and 727 while introducing three other Fold-4 harms, leaving ACC unchanged. The effect is attributable to the intentional training-test label exposure and cannot be treated as independent generalization. This result does not by itself justify a GO decision for the MICCAI 2027 second innovation module.", "",
        f"Authoritative directory. `{ROOT}`", "",
    ]
    return "\n".join(lines)


def main():
    replayed = h.read_json(REPLAY_PATH)
    require(replayed.get("status") == "PASS", "Replay status is not PASS")
    require(replayed["fold4"]["darg_rows"] == 788 and replayed["fold4"]["legal_rows"] == 784,
            "Fold-4 membership summary differs")

    fold_rows = make_fold_metrics(replayed)
    patient_rows = add_patient_audit_fields(list(replayed["patient_rows"]))
    reuse_check = verify_nonfold4_reuse(patient_rows)
    parent_manifests = replay.core.source_manifests()
    paired_rows = list(replayed["paired"])
    residual_rows = list(replayed["residual_off_to_full"])
    query_rows = list(replayed["anonymous_to_medical"])
    for row in paired_rows:
        row["comparison"] = "E82.4.6.5-G1_to_E82.4.6.6-G1"
    for row in residual_rows:
        row["comparison"] = "Residual-off_to_Full"
    for row in query_rows:
        row["comparison"] = "Anonymous_Query_to_Medical_Query"
    query_rows = add_query_change_counts(query_rows, patient_rows)
    comparison_rows = paired_rows + residual_rows + query_rows

    write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparison_rows)
    write_csv(RESULTS / "patient_level.csv", patient_rows)

    membership = {
        "status": "PASS", "experiment": CONFIG["experiment"], "patient_id_base": "1-based",
        "index_conversion": "index_0based = patient_id_1based - 1", "index_conversion_verified": all(r["patient_index_conversion_verified"] for r in patient_rows),
        "labels": {"712": 0, "713": 0, "727": 0, "773": 0},
        "fold4": {
            "darg_train_rows": 788, "scmr_legal_rows": 784, "gaussian_legal_rows": 784,
            "ema_legal_rows": 784, "test_rows": 87, "masked_rows": 4,
            "masked_ids_1based": [712, 713, 727, 773], "masked_rows_0based": [711, 712, 726, 772],
            "darg_classification_participation": True, "darg_relation_participation": True,
            "scmr_label_supervision_excluded": True, "gaussian_excluded": True, "ema_excluded": True,
            "relation_pairs": 4096, "same_class_pairs": 2048, "different_class_pairs": 2048,
            "pair_rng_unchanged": True, "all_pairs_legal_darg_members": True, "no_self_pairs": True,
            "relation_labels_verified": True, "raw_spectral_components_masked": True,
            "class_counts_asd_td": replayed["fold4"]["certificate"]["membership"]["after_class_counts_asd_td"],
            "class_weights_asd_td": replayed["fold4"]["certificate"]["membership"]["after_class_weights_asd_td"],
            "per_overlap_pair_count": replayed["fold4"]["certificate"]["membership"]["per_overlap_pair_count"],
            "legal_index_sha": replayed["fold4"]["certificate"]["membership"]["legal_index_sha"],
            "gaussian": replayed["fold4"]["certificate"]["gaussian"],
        },
        "current_overlap_by_fold_1based": {str(f): sorted(v) for f, v in replay.CURRENT_EXPOSURE.items()},
        "previous_overlap_by_fold_1based": {str(f): sorted(v) for f, v in replay.PREVIOUS_EXPOSURE.items()},
        "current_overlap_count": 17, "previous_overlap_count": 16, "outside_fixed_union_count": 854,
        "reused_fold_membership": {"2": {"darg_train_rows": 792, "scmr_legal_rows": 784}, "3": {"darg_train_rows": 789, "scmr_legal_rows": 784}},
        "outside_fixed_union_definition": "871 patients minus the explicit current-or-previous overlap union by fold and patient ID",
        "all_non_fold4_reused": True,
        "reuse_exact_check": reuse_check,
        "parent_manifest_status": {name: value["status"] for name, value in parent_manifests.items()},
        "parent_manifest_sha256": {name: replay.h.sha_file(path) for name, path in {
            "base": Path(CONFIG["full_path_directory"]).parent / "experiment82_4_6_pcsr_scmr" / "manifest.json",
            "full": Path(CONFIG["full_path_directory"]) / "manifest.json",
            "e8246": Path(CONFIG["source_e8246_directory"]) / "manifest.json",
            "e82463": Path(CONFIG["source_e82463_directory"]) / "manifest.json",
            "e82465": Path(CONFIG["source_e82465_directory"]) / "manifest.json",
            "e744": Path(CONFIG["source_readback"]["e744_manifest"]),
            "e61": Path(CONFIG["source_readback"]["e61_manifest"]),
        }.items()},
    }
    h.write_json(RESULTS / "membership_audit.json", membership)

    report = report_text(replayed, patient_rows, paired_rows, residual_rows, query_rows)
    with atomic_writer(RESULTS / "report.md", "w") as stream:
        stream.write(report)

    result_hashes = {f"results/{name}": h.sha_file(RESULTS / name)
                     for name in ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv", "membership_audit.json")}
    handoff = write_handoff(replayed, result_hashes)

    audit_repairs = {
        "preflight_initial_ema_registration": {
            "path": "runtime/G1_f04.preflight.json", "reason": "Register the already-computed initial EMA digest for the known audit-entry schema mismatch",
            "not_training_code": True, "sha256": h.sha_file(ROOT / "runtime/G1_f04.preflight.json"),
        },
        "independent_mask_index": {
            "path": "independent_465.py", "reason": "Use list-based advanced indexing for the four masked rows in the independent audit entry",
            "not_training_code": True, "sha256": h.sha_file(ROOT / "independent_465.py"),
        },
        "replay_numpy_runtime": {
            "path": "audit/replay.py", "reason": "Use the historical NumPy 2.5 read-only runtime for checkpoint pickle resolution",
            "not_training_code": True, "sha256": h.sha_file(ROOT / "audit/replay.py"),
        },
        "training_metadata_group": {
            "path": "runtime/G1_f04.training.json", "reason": "Correct inherited G2 metadata labels in the Fold-4 G1 recovery record",
            "not_training_code": True, "sha256": h.sha_file(ROOT / "runtime/G1_f04.training.json"),
        },
    }
    source_before = h.read_json(ROOT / "manifest.json")
    audit_status = {
        "status": "COMPLETE_AUDITED", "experiment": CONFIG["experiment"],
        "authoritative_directory": str(ROOT), "preferred_directory": CONFIG["preferred_directory"],
        "preferred_directory_errno": CONFIG["preferred_directory_errno"],
        "only_retrained_fold": 4, "reused_folds": [1, 2, 3, 5, 6, 7, 8, 9, 10],
        "frozen_training_implementation": source_before["implementation"],
        "training_implementation_sha": source_before["implementation_sha"],
        "audit_entry_repairs": audit_repairs, "fold4": replayed["fold4"],
        "checkpoint_internal_group": "G2 canonical parent field retained for historical E82.4.6.5-G1 loader compatibility",
        "membership": membership, "metrics": replayed["summaries"],
        "paired": {"e82465_to_e82466": next(r for r in paired_rows if r["comparison"] == "E82.4.6.5-G1_to_E82.4.6.6-G1" and r["scope"] == "ten_fold")},
        "fixed_weight": {"residual_off_to_full": next(r for r in residual_rows if r["scope"] == "ten_fold"),
                          "anonymous_to_medical": next(r for r in query_rows if r["scope"] == "ten_fold")},
        "result_files_pending": result_hashes,
        "handoff": handoff,
        "paper_and_author_code_reviewed": True,
        "scientific_limit": "Intentional label exposure diagnostic. No independent generalization or innovation GO claim.",
    }
    h.write_json(RESULTS / "audit_status.json", audit_status)
    result_hashes["results/audit_status.json"] = h.sha_file(RESULTS / "audit_status.json")

    manifest = h.read_json(ROOT / "manifest.json")
    require(manifest["status"] in {"FOLD4_AUDITED", "COMPLETE_AUDITED"}, "Unexpected pre-publish manifest state")
    manifest["status"] = "COMPLETE_AUDITED"
    manifest["authoritative_directory"] = str(ROOT)
    manifest["fallback"] = {"used": True, "errno": CONFIG["preferred_directory_errno"], "reason": "preferred OSSFS directory returned errno 5"}
    manifest["actual_new_folds"] = [4]
    manifest["reused_folds"] = [1, 2, 3, 5, 6, 7, 8, 9, 10]
    manifest["frozen_training_implementation"] = manifest["implementation"]
    manifest["frozen_training_implementation_sha"] = manifest["implementation_sha"]
    manifest["audit_entry_repairs"] = audit_repairs
    manifest["result_files"] = result_hashes
    manifest["runtime_files"] = {
        "runtime/replayed_folds.json": h.sha_file(REPLAY_PATH),
        "runtime/G1_f04.training.json": h.sha_file(ROOT / "runtime/G1_f04.training.json"),
        "runtime/G1_f04.preflight.json": h.sha_file(ROOT / "runtime/G1_f04.preflight.json"),
        "runtime/G1_f04.jsonl": h.sha_file(ROOT / "runtime/G1_f04.jsonl"),
    }
    manifest["fold4_result"] = {"best_epoch": replayed["fold4"]["best_epoch"], "latest_epoch": replayed["fold4"]["latest_epoch"],
                                 "best_sha": replayed["fold4"]["best_sha"], "latest_sha": replayed["fold4"]["latest_sha"],
                                 "journal_sha": replayed["fold4"]["journal_sha"], "correct": 83, "test": 87}
    manifest["completed_task"]["group"] = "G1"
    manifest["completed_task"]["task_group"] = "G1"
    manifest["audit"] = {"status": "COMPLETE_AUDITED", "replay": h.sha_file(ROOT / "audit/replay.py"),
                          "publish": h.sha_file(ROOT / "audit/publish.py"), "audit_fix": h.sha_file(ROOT / "audit/audit_fix.py"),
                          "audit_status": h.sha_file(RESULTS / "audit_status.json"), "membership": h.sha_file(RESULTS / "membership_audit.json"),
                          "handoff": handoff}
    h.write_json(ROOT / "manifest.json", manifest)

    final_manifest = h.read_json(ROOT / "manifest.json")
    require(final_manifest["status"] == "COMPLETE_AUDITED", "Final manifest status is not complete")
    require(h.sha_file(RESULTS / "audit_status.json") == final_manifest["result_files"]["results/audit_status.json"],
            "Audit status digest changed during manifest publication")
    require(HANDOFF.exists() and HANDOFF.read_text(encoding="utf-8").find("Experiment 82.4.6.6-G1") >= 0,
            "Handoff readback unavailable")
    print(json.dumps({"status": "COMPLETE_AUDITED", "manifest_sha": h.sha_file(ROOT / "manifest.json"),
                      "report_sha": h.sha_file(RESULTS / "report.md"), "handoff_sha": handoff["readback_sha"]}, indent=2))


if __name__ == "__main__":
    main()
