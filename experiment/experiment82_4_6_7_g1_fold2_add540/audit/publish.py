"""Publish the completed E82.4.6.7-G1 replay, metrics, and handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import replay
import core

ROOT = replay.ROOT
RESULTS = ROOT / "results"
E465 = replay.E465
E466 = replay.E466
CONFIG = replay.CONFIG
h = replay.h
REPLAY_PATH = ROOT / "runtime" / "replayed_folds.json"
HANDOFF = ROOT.parent / "handoff.md"
METHODS = ("E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.4-G1", "E82.4.6.5-G1", "E82.4.6.6-G1", "E82.4.6.7-G1")
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
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows):
    rows = list(rows); require(rows, f"No rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with atomic_writer(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percent(value):
    return None if value is None else float(value) * 100.0


def historical_metric_rows():
    selected = {}
    for source, methods in ((E466 / "results/fold_metrics.csv", {"E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.5-G1", "E82.4.6.6-G1"}),
                            (E465 / "results/fold_metrics.csv", {"E82.4.6.4-G1"})):
        for row in read_csv(source):
            if row.get("scope") == "test_fold" and row.get("method") in methods:
                selected[(row["method"], int(row["fold"]))] = row
    required = {(method, fold) for method in METHODS[:-1] for fold in range(1, 11)}
    require(required <= selected.keys(), "Historical fold metrics are incomplete")
    return selected


def make_fold_metrics(replayed):
    archived = historical_metric_rows(); rows = []
    for method in METHODS[:-1]:
        for fold in range(1, 11): rows.append(dict(archived[(method, fold)]))
    current = replayed["metrics"]["E82.4.6.7-G1"]
    old = replayed["metrics"]["E82.4.6.6-G1"]
    for fold in range(1, 11):
        value = current[str(fold)]
        old_value = old[str(fold)]
        row = {"method": "E82.4.6.7-G1", "fold": fold, "scope": "test_fold",
               "best_epoch": replayed["fold2"]["best_epoch"] if fold == 2 else old_value.get("best_epoch", ""),
               "end_epoch": replayed["fold2"]["latest_epoch"] if fold == 2 else old_value.get("end_epoch", ""),
               "checkpoint_source": "fold2_retrained_from_E61_epoch10" if fold == 2 else "E82.4.6.6-G1_read_only_reuse",
               "best_sha": replayed["fold2"]["best_sha"] if fold == 2 else next(r["current_best_sha"] for r in replayed["patient_rows"] if r["fold"] == fold)}
        for key in ("n_test", "tp", "fp", "tn", "fn"): row[key] = value[key]
        for key in METRIC_KEYS:
            row[key] = value[key]; row[key + "_percent"] = percent(value[key])
        rows.append(row)
    return rows


def add_patient_audit_fields(rows):
    for row in rows:
        overlap = bool(row["current_native_training_overlap"])
        require(int(row["patient_index_0based"]) == int(row["patient_id_1based"]) - 1, "Patient index conversion failed")
        row.update({"patient_index_conversion_verified": True, "darg_classification_supervised": overlap,
                    "darg_relation_training_member": overlap, "scmr_legal_supervision_member": not overlap,
                    "gaussian_legal_member": not overlap, "ema_legal_member": not overlap,
                    "new_residual_masked": overlap})
    return rows


def scope_patients(rows, scope):
    if scope == "ten_fold": return list(rows)
    if scope.startswith("fold") and scope[4:].isdigit(): return [r for r in rows if r["fold"] == int(scope[4:])]
    if scope == "fold2_new_540": return [r for r in rows if r["fold"] == 2 and r["patient_id_1based"] == 540]
    if scope == "fold2_retained_overlap_8": return [r for r in rows if r["fold"] == 2 and r["patient_id_1based"] in replay.PREVIOUS_EXPOSURE[2]]
    if scope == "fold2_other_78": return [r for r in rows if r["fold"] == 2 and r["patient_id_1based"] not in replay.CURRENT_EXPOSURE[2]]
    if scope == "fixed_union_exposure_18": return [r for r in rows if r["in_fixed_union_exposure_18"]]
    if scope == "outside_fixed_union_853": return [r for r in rows if r["outside_fixed_union_853"]]
    raise RuntimeError(f"Unknown paired scope {scope}")


def add_query_change_counts(query_rows, patient_rows):
    for row in query_rows:
        selected = scope_patients(patient_rows, row["scope"])
        row["cam144_changed_count"] = sum(float(x["query_feature_change_l2"]) > 1e-12 for x in selected)
        row["asd_probability_changed_count"] = sum(abs(float(x["p_asd_full"]) - float(x["p_asd_anonymous"])) > 1e-12 for x in selected)
        row["hard_prediction_flip_count"] = sum(int(x["prediction_full"]) != int(x["prediction_anonymous"]) for x in selected)
    return query_rows


def verify_reuse(patient_rows):
    archived = {(int(r["fold"]), int(r["patient_id_1based"])): r for r in read_csv(E466 / "results/patient_level.csv")}
    fields = ("prediction_full", "p_asd_full", "true_margin_full", "prediction_residual_off", "p_asd_residual_off", "true_margin_residual_off", "prediction_anonymous", "p_asd_anonymous", "true_margin_anonymous")
    checked = 0
    for row in patient_rows:
        if int(row["fold"]) == 2: continue
        old = archived[(int(row["fold"]), int(row["patient_id_1based"]))]
        for field in fields:
            if field.startswith("prediction"): require(int(row[field]) == int(old[field]), f"Reused hard prediction changed for {row['fold']}:{row['patient_id_1based']}")
            else: require(float(row[field]) == float(old[field]), f"Reused probability or margin changed for {row['fold']}:{row['patient_id_1based']}")
        require(row["current_best_sha"] == old["current_best_sha"], "Reused checkpoint identity changed")
        checked += 1
    require(checked == 784, "Unexpected number of reused patient rows")
    return {"patients": checked, "probability_hard_prediction_margin_and_checkpoint_exact": True}


def write_handoff(replayed, result_hashes):
    fold2 = replayed["fold2"]; ten = next(r for r in replayed["paired"] if r["scope"] == "ten_fold"); summaries = replayed["summaries"]["E82.4.6.7-G1"]
    patient = next(r for r in replayed["patient_rows"] if r["fold"] == 2 and r["patient_id_1based"] == 540)
    section = f"""## Experiment 82.4.6.7-G1 — E82.4.6.6-G1 plus Fold-2 Patient 540

- Purpose. Intentional training-test label-overlap diagnostic. The sole membership change was adding patient 540 to Fold 2 DARG training.
- Authority. `{ROOT}`. The preferred OSSFS path returned errno 5, so this is the single authoritative directory.
- Training. Only Fold 2 was retrained from the verified E61 epoch-10 state at epoch 11 with seed 0. DARG rows were 793. SCMR, Gaussian, and EMA legal rows were 784. The test set remained 87.
- Membership. Fold 2 overlap IDs were 377, 407, 681, 779, 784, 260, 288, 375, and 540. Patient 540 label was independently verified as ASD encoding 0, participated in DARG classification and relation training, and was excluded from new spectral, Prompt, Gaussian, EMA, and label supervision.
- Results. E82.4.6.7-G1 ten-fold ACC {summaries['acc']['mean_pct']:.4f} ± {summaries['acc']['sd_pct']:.4f}, BA {summaries['hard_auc']['mean_pct']:.4f} ± {summaries['hard_auc']['sd_pct']:.4f}, Prob-AUC {summaries['prob_auc']['mean_pct']:.4f} ± {summaries['prob_auc']['sd_pct']:.4f}, SEN-ASD {summaries['sen']['mean_pct']:.4f} ± {summaries['sen']['sd_pct']:.4f}, SPE-TD {summaries['spe']['mean_pct']:.4f} ± {summaries['spe']['sd_pct']:.4f}, ASD-F1 {summaries['asd_f1']['mean_pct']:.4f} ± {summaries['asd_f1']['sd_pct']:.4f}.
- Paired E82.4.6.6-G1 to E82.4.6.7-G1. Corrections {ten['corrected']}, harms {ten['harmed']}, net {ten['net']}. Patient 540 previous prediction {patient['prediction_previous']} and current prediction {patient['prediction_full']}.
- Audit. Best/Latest reload, next-step training and evaluation recovery, RNG replay, six-dropout order, relation pairs, Gaussian/EMA membership, end-to-end gradients, residual limits and nine-person masking passed. Historical nine-fold reuse matched E82.4.6.6-G1 exactly.
- Interpretation. Performance changes are attributable to intentional label exposure and the resulting training adaptation. They are not independent-generalization evidence and do not establish a GO decision for the MICCAI 2027 second innovation module.
- Result SHA values are recorded in the final manifest and audit status. Handoff readback preserved the historical prefix before this section.
"""
    lock_path = HANDOFF.with_name(HANDOFF.name + ".lock"); HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
        marker = "## Experiment 82.4.6.7-G1 — E82.4.6.6-G1 plus Fold-2 Patient 540"
        if marker not in previous:
            temporary = HANDOFF.with_name(HANDOFF.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(previous.rstrip() + "\n\n" + section.strip() + "\n"); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, HANDOFF)
        readback = HANDOFF.read_text(encoding="utf-8")
        require(readback.startswith(previous), "Handoff historical prefix changed")
        require(marker in readback and "Result SHA values" in readback, "Handoff section readback incomplete")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock_path.unlink(missing_ok=True)
    return {"path": str(HANDOFF), "historical_prefix_preserved": True, "section_marker": marker,
            "readback_sha": hashlib.sha256(readback.encode()).hexdigest(), "result_sha256_at_append": result_hashes}


def fmt(value):
    return "NA" if value is None else f"{float(value):.4f}"


def report_text(replayed, patient_rows, paired_rows, residual_rows, query_rows):
    summaries = replayed["summaries"]
    pair = lambda scope: next(r for r in paired_rows if r["scope"] == scope)
    residual = lambda scope: next(r for r in residual_rows if r["scope"] == scope)
    query = lambda scope: next(r for r in query_rows if r["scope"] == scope)
    p540 = next(r for r in patient_rows if r["fold"] == 2 and r["patient_id_1based"] == 540)
    overlap8 = pair("fold2_retained_overlap_8"); other78 = pair("fold2_other_78"); ten = pair("ten_fold")
    lines = ["# Experiment 82.4.6.7-G1", "", "## A. STATUS", "", "COMPLETE_AUDITED. Only Fold 2 was retrained. This is an intentional training-test label-overlap diagnostic.", "", "## B. Ten-fold performance", "", "Values are mean ± population SD in percent. Hard-AUC is BA from hard predictions. Prob-AUC uses continuous ASD probability.", "", "| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        s = summaries[method]; lines.append("| " + method + " | " + " | ".join(f"{s[k]['mean_pct']:.4f} ± {s[k]['sd_pct']:.4f}" for k in METRIC_KEYS) + " |")
    lines += ["", "## C. Fold 2 paired comparison", "", f"E82.4.6.6-G1 to E82.4.6.7-G1. Corrections {ten['corrected']} ({ten['corrected_ids'] or 'none'}), harms {ten['harmed']} ({ten['harmed_ids'] or 'none'}), net {ten['net']}. ΔACC {fmt(ten['delta_acc_pp'])} pp, ΔBA {fmt(ten['delta_ba_pp'])} pp, ΔProb-AUC {fmt(ten['delta_prob_auc_pp'])} pp.", f"Ten-fold paired ΔSEN-ASD {fmt(ten['delta_sen_pp'])} pp, ΔSPE-TD {fmt(ten['delta_spe_pp'])} pp, and ΔASD-F1 {fmt(ten['delta_asd_f1_pp'])} pp.", f"Fold 2 corrections {pair('fold2')['corrected_ids'] or 'none'}, harms {pair('fold2')['harmed_ids'] or 'none'}, net {pair('fold2')['net']}. ΔACC {fmt(pair('fold2')['delta_acc_pp'])} pp, ΔBA {fmt(pair('fold2')['delta_ba_pp'])} pp, ΔProb-AUC {fmt(pair('fold2')['delta_prob_auc_pp'])} pp.", "", "## D. Patient 540", "", f"True label ASD=0. E82.4.6.6-G1 prediction {p540['prediction_previous']} with ASD probability {p540['p_asd_previous']:.8f} and true-class margin {p540['true_margin_previous']:.6f}. E82.4.6.7-G1 Residual-off prediction {p540['prediction_residual_off']} with ASD probability {p540['p_asd_residual_off']:.8f}. Full prediction {p540['prediction_full']} with ASD probability {p540['p_asd_full']:.8f} and margin {p540['true_margin_full']:.6f}. ASD probability change versus the direct reference was {p540['p_asd_full'] - p540['p_asd_previous']:.8f}. Corrected versus the reference is {bool(p540['corrected_vs_previous'])}. Patient 540 participated in DARG classification and relation training and was excluded from new SCMR label supervision, Gaussian, EMA, and spectral/Prompt residual updates. Its new spectral-in, spectral-out, and joint residual norms were {p540['delta_parallel_l2']:.12g}, {p540['delta_perp_l2']:.12g}, and {p540['delta_l2']:.12g}.", "", "## E. Other Fold 2 patients", f"The eight retained overlap patients had corrections {overlap8['corrected']} ({overlap8['corrected_ids'] or 'none'}), harms {overlap8['harmed']} ({overlap8['harmed_ids'] or 'none'}), net {overlap8['net']}. The other 78 test patients had corrections {other78['corrected']} ({other78['corrected_ids'] or 'none'}), harms {other78['harmed']} ({other78['harmed_ids'] or 'none'}), net {other78['net']}.", "", "## F. Residual-off, spectral/Prompt, and medical Query mechanisms", f"Residual-off to Full across ten folds. Corrections {residual('ten_fold')['corrected']}, harms {residual('ten_fold')['harmed']}, net {residual('ten_fold')['net']}, ΔACC {fmt(residual('ten_fold')['delta_acc_pp'])} pp, ΔBA {fmt(residual('ten_fold')['delta_ba_pp'])} pp, mean margin change {fmt(residual('ten_fold')['margin_mean_delta'])}.", f"Anonymous Query to Medical Query across ten folds. CAM144 changed for {query('ten_fold')['cam144_changed_count']} patients, ASD probability changed for {query('ten_fold')['asd_probability_changed_count']}, and hard prediction flipped for {query('ten_fold')['hard_prediction_flip_count']}. Corrections {query('ten_fold')['corrected']}, harms {query('ten_fold')['harmed']}, net {query('ten_fold')['net']}. Fold 2 CAM144 changes {query('fold2')['cam144_changed_count']}, probability changes {query('fold2')['asd_probability_changed_count']}, hard flips {query('fold2')['hard_prediction_flip_count']}.", "All nine Fold 2 overlap patients had exact zero spectral-in, spectral-out, and joint residual, while the remaining 78 Fold 2 test patients retained an active residual path. The maximum relative joint CAM residual was {:.6f}. The 8% residual limit and 0.1% shared-gradient budget passed.".format(replayed["fold2"]["max_cam_relative_delta"]), "", "## G. Checkpoint, RNG, gradient, masking, and reuse audit", f"Best epoch {replayed['fold2']['best_epoch']}, Latest epoch {replayed['fold2']['latest_epoch']}. Best SHA {replayed['fold2']['best_sha']}. Latest SHA {replayed['fold2']['latest_sha']}. Journal SHA {replayed['fold2']['journal_sha']}.", "Latest next-step training and evaluation recovery were exact. RNG was replayed epoch by epoch, the six-dropout order was preserved, DARG classification and relation gradients were present, SCMR residual gradients were zero on masked rows and nonzero on legal rows, and Gaussian/EMA legal membership remained 784.", "The nine reused folds were read-only E82.4.6.6-G1. Their patient-level probabilities, hard predictions, margins, and checkpoint identities were verified exactly. The E82.4.6.6-G1 Fold 4 result was not used as a starting weight.", "", "## H. Files and scientific conclusion", f"Authoritative directory `{ROOT}`. Required result files, Manifest, runtime replay, Fold 2 Best/Latest, and handoff are recorded with SHA values. The fixed current-or-reference exposure union contains 18 patient-fold entries and the explicitly constructed outside set contains 853 patients. The Fold 2 unexposed analysis contains 78 patients.", "The observed changes are attributable to intentional training-test label exposure under the historical val=test, single-seed, and transductive-graph protocol. They are not independent-generalization evidence and do not by themselves justify a GO decision for the MICCAI 2027 second innovation module."]
    lines += ["", "### Per-fold outcome", "", "| Fold | Result | Corrections | Harms | Net |", "|---:|---|---:|---:|---:|"]
    for row in paired_rows:
        if row.get("fold_result"): lines.append(f"| {row['scope'].replace('fold','')} | {row['fold_result']} | {row['corrected']} | {row['harmed']} | {row['net']} |")
    return "\n".join(lines) + "\n"


def main():
    replayed = h.read_json(REPLAY_PATH)
    require(replayed.get("status") == "PASS", "Replay status is not PASS")
    require(replayed["fold2"]["darg_rows"] == 793 and replayed["fold2"]["legal_rows"] == 784 and replayed["fold2"]["masked_rows"] == 9, "Fold-2 membership summary differs")
    fold_rows = make_fold_metrics(replayed)
    patient_rows = add_patient_audit_fields(list(replayed["patient_rows"]))
    reuse_check = verify_reuse(patient_rows)
    parent_manifests = core.source_manifests()
    paired_rows = list(replayed["paired"]); residual_rows = list(replayed["residual_off_to_full"]); query_rows = list(replayed["anonymous_to_medical"])
    query_rows = add_query_change_counts(query_rows, patient_rows)
    for row in paired_rows: row["comparison"] = "E82.4.6.6-G1_to_E82.4.6.7-G1"
    for row in residual_rows: row["comparison"] = "Residual-off_to_Full"
    for row in query_rows: row["comparison"] = "Anonymous_Query_to_Medical_Query"
    comparison_rows = paired_rows + residual_rows + query_rows
    write_csv(RESULTS / "fold_metrics.csv", fold_rows); write_csv(RESULTS / "paired_comparisons.csv", comparison_rows); write_csv(RESULTS / "patient_level.csv", patient_rows)
    membership = {"status": "PASS", "experiment": CONFIG["experiment"], "patient_id_base": "1-based", "index_conversion": "index_0based = patient_id_1based - 1", "index_conversion_verified": True, "labels": {"540": 0}, "fold2": {"darg_train_rows": 793, "scmr_legal_rows": 784, "gaussian_legal_rows": 784, "ema_legal_rows": 784, "test_rows": 87, "masked_rows": 9, "masked_ids_1based": sorted(replay.CURRENT_EXPOSURE[2]), "masked_rows_0based": [p - 1 for p in sorted(replay.CURRENT_EXPOSURE[2])], "darg_classification_participation": True, "darg_relation_participation": True, "scmr_label_supervision_excluded": True, "gaussian_excluded": True, "ema_excluded": True, "relation_pairs": 4096, "same_class_pairs": 2048, "different_class_pairs": 2048, "pair_rng_unchanged": True, "all_pairs_legal_darg_members": True, "no_self_pairs": True, "relation_labels_verified": True, "raw_spectral_components_masked": True, "class_counts_asd_td": replayed["fold2"]["certificate"]["membership"]["after_class_counts_asd_td"], "class_weights_asd_td": replayed["fold2"]["certificate"]["membership"]["after_class_weights_asd_td"], "per_overlap_pair_count": replayed["fold2"]["certificate"]["membership"]["per_overlap_pair_count"], "legal_index_sha": replayed["fold2"]["certificate"]["membership"]["legal_index_sha"], "gaussian": replayed["fold2"]["certificate"]["gaussian"]}, "current_overlap_by_fold_1based": {str(f): sorted(v) for f, v in replay.CURRENT_EXPOSURE.items()}, "reference_overlap_by_fold_1based": {str(f): sorted(v) for f, v in replay.PREVIOUS_EXPOSURE.items()}, "current_overlap_count": 18, "reference_overlap_count": 17, "outside_fixed_union_count": 853, "fold2_unexposed_test_count": 78, "reused_folds": [1, 3, 4, 5, 6, 7, 8, 9, 10], "all_reused_exact": reuse_check, "parent_manifest_status": {name: value["status"] for name, value in parent_manifests.items()}, "parent_manifest_sha256": {name: replay.h.sha_file(path) for name, path in {"base": Path(CONFIG["full_path_directory"]).parent / "experiment82_4_6_pcsr_scmr" / "manifest.json", "full": Path(CONFIG["full_path_directory"]) / "manifest.json", "e8246": Path(CONFIG["source_e8246_directory"]) / "manifest.json", "e82463": Path(CONFIG["source_e82463_directory"]) / "manifest.json", "e82465": Path(CONFIG["source_e82465_directory"]) / "manifest.json", "e8266": Path(CONFIG["source_e8266_directory"]) / "manifest.json", "e744": Path(CONFIG["source_readback"]["e744_manifest"]), "e61": Path(CONFIG["source_readback"]["e61_manifest"])}.items()}}
    h.write_json(RESULTS / "membership_audit.json", membership)
    with atomic_writer(RESULTS / "report.md", "w") as stream: stream.write(report_text(replayed, patient_rows, paired_rows, residual_rows, query_rows))
    result_hashes = {f"results/{name}": h.sha_file(RESULTS / name) for name in ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv", "membership_audit.json")}
    handoff = write_handoff(replayed, result_hashes)
    task_meta = h.read_json(ROOT / "runtime/G1_f02.training.json")
    if task_meta.get("group") != "G1" or task_meta.get("task_group") != "G1":
        task_meta["group"] = "G1"; task_meta["task_group"] = "G1"; h.write_json(ROOT / "runtime/G1_f02.training.json", task_meta)
    audit_repairs = {"preflight_initial_ema_registration": {"path": "runtime/G1_f02.preflight.json", "reason": "Register the already-computed initial EMA digest for the historical audit schema", "not_training_code": True, "sha256": h.sha_file(ROOT / "runtime/G1_f02.preflight.json")}, "independent_mask_index": {"path": "independent_467.py", "reason": "List-based advanced indexing for the nine masked rows in the independent audit entry", "not_training_code": True, "sha256": h.sha_file(ROOT / "independent_467.py")}, "replay_numpy_runtime": {"path": "audit/replay.py", "reason": "Historical NumPy 2.5 read-only runtime for checkpoint pickle resolution", "not_training_code": True, "sha256": h.sha_file(ROOT / "audit/replay.py")}, "training_metadata_group": {"path": "runtime/G1_f02.training.json", "reason": "Correct inherited canonical G2 metadata labels in the formal G1 recovery record", "not_training_code": True, "sha256": h.sha_file(ROOT / "runtime/G1_f02.training.json")}}
    source_before = h.read_json(ROOT / "manifest.json")
    audit_status = {"status": "COMPLETE_AUDITED", "experiment": CONFIG["experiment"], "authoritative_directory": str(ROOT), "preferred_directory": CONFIG["preferred_directory"], "preferred_directory_errno": CONFIG["preferred_directory_errno"], "only_retrained_fold": 2, "reused_folds": [1, 3, 4, 5, 6, 7, 8, 9, 10], "frozen_training_implementation": source_before["implementation"], "training_implementation_sha": source_before["implementation_sha"], "audit_entry_repairs": audit_repairs, "fold2": replayed["fold2"], "membership": membership, "metrics": replayed["summaries"], "paired": {"e8266_to_e8267": next(r for r in paired_rows if r["scope"] == "ten_fold")}, "fixed_weight": {"residual_off_to_full": next(r for r in residual_rows if r["scope"] == "ten_fold"), "anonymous_to_medical": next(r for r in query_rows if r["scope"] == "ten_fold")}, "result_files_pending": result_hashes, "handoff": handoff, "parent_manifest_sha256": membership["parent_manifest_sha256"], "paper_and_author_code_reviewed": True, "scientific_limit": "Intentional label exposure diagnostic. No independent generalization or innovation GO claim."}
    h.write_json(RESULTS / "audit_status.json", audit_status); result_hashes["results/audit_status.json"] = h.sha_file(RESULTS / "audit_status.json")
    manifest = h.read_json(ROOT / "manifest.json"); require(manifest["status"] in {"AUDITED", "AUDIT_PASSED", "TRAINED", "COMPLETE_AUDITED"}, "Unexpected pre-publish manifest state")
    fold2_metric = replayed["metrics"]["E82.4.6.7-G1"]["2"]
    manifest.update(status="COMPLETE_AUDITED", authoritative_directory=str(ROOT), fallback={"used": True, "errno": CONFIG["preferred_directory_errno"], "reason": "preferred OSSFS directory returned errno 5"}, actual_new_folds=[2], reused_folds=[1, 3, 4, 5, 6, 7, 8, 9, 10], frozen_training_implementation=manifest["implementation"], frozen_training_implementation_sha=manifest["implementation_sha"], audit_entry_repairs=audit_repairs, result_files=result_hashes, runtime_files={"runtime/replayed_folds.json": h.sha_file(REPLAY_PATH), "runtime/G1_f02.training.json": h.sha_file(ROOT / "runtime/G1_f02.training.json"), "runtime/G1_f02.preflight.json": h.sha_file(ROOT / "runtime/G1_f02.preflight.json"), "runtime/G1_f02.jsonl": h.sha_file(ROOT / "runtime/G1_f02.jsonl")}, fold2_result={"best_epoch": replayed["fold2"]["best_epoch"], "latest_epoch": replayed["fold2"]["latest_epoch"], "best_sha": replayed["fold2"]["best_sha"], "latest_sha": replayed["fold2"]["latest_sha"], "journal_sha": replayed["fold2"]["journal_sha"], "correct": int(fold2_metric["tp"] + fold2_metric["tn"]), "test": 87}, checkpoint_internal_group="G2 canonical parent field retained for historical loader compatibility", audit={"status": "COMPLETE_AUDITED", "replay": h.sha_file(ROOT / "audit/replay.py"), "publish": h.sha_file(ROOT / "audit/publish.py"), "audit_fix": h.sha_file(ROOT / "audit/audit_fix.py"), "audit_status": h.sha_file(RESULTS / "audit_status.json"), "membership": h.sha_file(RESULTS / "membership_audit.json"), "handoff": handoff})
    from execute import verify_guard
    manifest["guard_after"] = verify_guard(manifest)
    h.write_json(ROOT / "manifest.json", manifest)
    final_manifest = h.read_json(ROOT / "manifest.json")
    require(final_manifest["status"] == "COMPLETE_AUDITED", "Final manifest status is not complete")
    require(h.sha_file(RESULTS / "audit_status.json") == final_manifest["result_files"]["results/audit_status.json"], "Audit status digest changed during publication")
    require(HANDOFF.exists() and "Experiment 82.4.6.7-G1" in HANDOFF.read_text(encoding="utf-8"), "Handoff readback unavailable")
    print(json.dumps({"status": "COMPLETE_AUDITED", "manifest_sha": h.sha_file(ROOT / "manifest.json"), "report_sha": h.sha_file(RESULTS / "report.md"), "handoff_sha": handoff["readback_sha"]}, indent=2))


if __name__ == "__main__":
    main()
