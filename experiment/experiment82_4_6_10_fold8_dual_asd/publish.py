"""Recompute the patient-level comparisons and publish audited artifacts."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

from core import CONFIG, FOLD, GROUPS, MANIFEST, RESULTS, ROOT, RUNTIME, SOURCES, h, paths
from sources import (CAM_FIELDS, COUNTS, DARG, E6, E9, FIELDS, FOLDS, IDENTITY, INTERVENTIONS,
                     METRICS, SINGLE_METHODS, historical_rows, score, utc_now, values, verify_guard)

import numpy as np

controller = h.archive.load_module("_e824610_publication_controller", ROOT / "execute.py")

METHOD = "E82.4.6.10"
FULL, OFF, ANON = "full", "residual_off", "anonymous"
BASE = "baseline_e469_g2"
LABEL = {0: "ASD", 1: "TD"}
HEADINGS = ("ACC", "Hard-AUC / BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
REFERENCES = {DARG: "e744_darg", E6: "e466_g1", E9: BASE,
              SINGLE_METHODS["G4"]: "e469_g4", SINGLE_METHODS["G6"]: "e469_g6"}
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
TITLE = "# " + CONFIG["experiment"]
MARKER = "#" + TITLE
FOCUS_IDS = (383, 430, 723, 601, 144, 410, 547, 498, 756)
RESULT_NAMES = ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv",
                "membership_audit.json", "audit_status.json")
SCIENTIFIC_DECISION = "NO_GO_INDEPENDENT_GENERALIZATION_OR_MICCAI_2027_MODULE_2"


def flatten(folds):
    return [row for fold in FOLDS for row in folds[fold]]


def numeric(key, value):
    return int(value) if key.startswith("prediction") or key in IDENTITY else float(value)


def members_flag(row, key):
    h.require(row[key] in ("True", "False"), f"Invalid {key} on archived patient row")
    return row[key] == "True"


def change_fields(row, before, after, name):
    y = row["true_label"]
    a, b = (row[f"prediction_{mode}"] for mode in (before, after))
    return {f"corrected_{name}": int(a != y and b == y), f"harmed_{name}": int(a == y and b != y),
            f"hard_flip_{name}": int(a != b),
            f"true_margin_change_{name}": row[f"true_margin_{after}"] - row[f"true_margin_{before}"]}


def patient_rows(group, archive, history, audit):
    targets = GROUPS[group]["patient_ids_1based"]
    added = {r["patient_id_1based"]: r for r in audit["patients"]}
    h.require(list(added) == [int(r["patient_id_1based"]) for r in archive[E9][FOLD]],
              f"{group} has a different Fold 8 test order")
    result = {}
    for fold in FOLDS:
        local = fold == FOLD
        old = archive[E9][fold]
        root_ref = history["folds"][str(fold)]["checkpoint"]
        rows = []
        for i, baseline in enumerate(old):
            row = {key: int(baseline[key]) for key in IDENTITY}
            subject, label = row["patient_id_1based"], row["true_label"]
            target = subject in targets if local else members_flag(baseline, "darg_classification_supervised")
            pair = audit["membership"]["target_pairs"].get(str(subject), {}) if local else {}
            gradient = audit["best_gradients"]["patients"].get(str(subject), {}) if local else {}
            occurrences = pair.get("endpoint_occurrences", 0) if local else int(baseline["darg_relation_pair_occurrences"])
            row.update(
                group=group, method=f"{METHOD}-{group}", fold=fold, true_diagnosis=LABEL[label],
                fold8_cohort=("A" if subject == 383 else "B" if subject == targets[1] else "C") if local else "",
                added_patient_ids_1based=" ".join(map(str, targets)),
                darg_classification_supervised=target, darg_relation_eligible=target,
                darg_relation_sampled=occurrences > 0, darg_relation_pair_occurrences=occurrences,
                darg_classification_cam144_grad_l2=gradient.get("classification_cam144_grad_l2"),
                darg_relation_direct_bce_grad_l2=gradient.get("relation_direct_bce_grad_l2"),
                darg_relation_direct_weighted_grad_l2=gradient.get("relation_direct_weighted_grad_l2"),
                scmr_label_supervised=False, gaussian_member=False, ema_member=False,
                residual_masked=target, residual_allowed_by_mask=not target,
                best_sha256=audit["artifact_sha256"]["best"] if local else baseline["best_sha256"],
                checkpoint_source=f"{METHOD}-{group}" if local else f"{E9} → {' → '.join(root_ref['lineage'])}",
                baseline_best_sha256=baseline["best_sha256"],
            )
            for method, tag in REFERENCES.items():
                source = archive[method][fold][i]
                modes = INTERVENTIONS if method == E9 else (FULL,)
                for mode in modes:
                    for key in ("prediction", "p_asd", "true_margin"):
                        field = f"{key}_{mode}"
                        row[f"{key}_{tag}_{mode}"] = numeric(field, source[field])
            row.update({f"{BASE}_{key}": float(baseline[key]) for key in CAM_FIELDS})
            if local:
                fresh = added[subject]
                h.require(fresh["true_label"] == label, "Fold 8 label differs between certificate and archive")
                row.update({key: val for key, val in fresh.items() if key not in IDENTITY})
            else:
                row.update({key: numeric(key, baseline[key]) for key in (*FIELDS, *CAM_FIELDS)})
                h.require(root_ref["best_sha"] == row["best_sha256"], "Reused Best SHA differs")
            for before, after, name in ((f"{BASE}_{FULL}", FULL, "vs_baseline"),
                                        (f"{BASE}_{OFF}", OFF, "retraining_off"),
                                        (OFF, FULL, "residual"), (ANON, FULL, "medical")):
                row.update(change_fields(row, before, after, name))
            rows.append(row)
        result[fold] = rows
    h.require(len(flatten(result)) == 871
              and [sum(r["fold8_cohort"] == cohort for r in result[FOLD]) for cohort in ("A", "B", "C")] == [1, 1, 85],
              f"{group} complete patient counts or Fold 8 cohorts differ")
    return result


def reused_proof(formal, archive, history):
    records = {}
    fields = (*IDENTITY, *FIELDS, *CAM_FIELDS, "best_sha256")
    for fold in FOLDS:
        if fold == FOLD:
            continue
        expected = [tuple(numeric(k, r[k]) if k != "best_sha256" else r[k] for k in fields)
                    for r in archive[E9][fold]]
        for group, folds in formal.items():
            h.require([tuple(r[k] for k in fields) for r in folds[fold]] == expected,
                      f"{group} reused rows diverge on Fold {fold}")
            projected = score(*values(folds[fold], FULL))
            h.require(projected == history["folds"][str(fold)]["metrics"], f"Fold {fold} reused metrics differ")
        records[str(fold)] = {
            "direct_source": E9, "checkpoint": history["folds"][str(fold)]["checkpoint"],
            "test_rows": len(expected), "projection_sha256": h.sha(expected),
            "all_variants_probabilities_margins_residuals_best_sha_exact": True,
        }
    h.require(sum(rec["test_rows"] for rec in records.values()) == 784, "Reused test count differs")
    return records


def checked_score(rows, mode):
    y, pred, prob = (np.asarray(x) for x in values(rows, mode))
    actual = score(y, pred, prob)
    asd, td = prob[y == 0], prob[y == 1]
    greater, equal = asd[:, None] > td, asd[:, None] == td
    auc = (int(greater.sum()) + 0.5 * int(equal.sum())) / (len(asd) * len(td))
    tp = int((pred[y == 0] == 0).sum())
    tn = int((pred[y == 1] == 1).sum())
    sen, spe = tp / len(asd), tn / len(td)
    f1 = 2 * tp / (2 * tp + len(td) - tn + len(asd) - tp)
    verify = (actual["n_test"] == len(y) and actual["tp"] == tp and actual["tn"] == tn
              and actual["fp"] == len(td) - tn and actual["fn"] == len(asd) - tp
              and actual["correct"] == tp + tn
              and np.allclose([actual[k] for k in METRICS],
                              [(tp + tn) / len(y), (sen + spe) / 2, auc, sen, spe, f1], atol=1e-12, rtol=0))
    h.require(verify, "Independent confusion or pair-count AUC calculation differs")
    return actual


def checkpoint_for(method, fold, history, audits):
    if method == DARG:
        return history["darg"][str(fold)]
    if fold != FOLD or method == E6:
        return history["folds"][str(fold)]["checkpoint"]
    if method in SINGLE_METHODS.values():
        group = method.rsplit("-", 1)[1]
        return history["single_patient_reference"][group]["checkpoint"]
    audit = audits[method.rsplit("-", 1)[1]]
    return {"best_epoch": audit["best_epoch"], "latest_epoch": audit["latest_epoch"],
            "best_sha": audit["artifact_sha256"]["best"], "lineage": [method]}


def all_metrics(all_folds, history, audits):
    rows, totals, each = [], {}, {}
    for method, folds in all_folds.items():
        group = method.rsplit("-", 1)[1] if method.startswith(METHOD + "-") else ""
        record = {"method": method, "group": group}
        every = each[method] = {fold: checked_score(folds[fold], FULL) for fold in FOLDS}
        for fold, stats in every.items():
            checkpoint = checkpoint_for(method, fold, history, audits)
            source = {"checkpoint_source": " → ".join(checkpoint["lineage"]), "best_epoch": checkpoint["best_epoch"],
                      "latest_epoch": checkpoint["latest_epoch"], "best_sha256": checkpoint["best_sha"]}
            rows.append({**record, "intervention": FULL, "scope": "test_fold", "fold": fold, **source, **stats})
            if group:
                h.require(checkpoint["best_sha"] == folds[fold][0]["best_sha256"], "Formal result references another Best")
            if group and fold == FOLD:
                h.require(stats == audits[group]["best_metrics"][FULL], f"{group} Best Full metrics differ")
                for mode in (OFF, ANON):
                    alt = checked_score(folds[fold], mode)
                    h.require(alt == audits[group]["best_metrics"][mode], f"{group} Best {mode} metrics differ")
                    rows.append({**record, "intervention": mode, "scope": "test_fold", "fold": fold, **source, **alt})
        average = {key: float(np.mean([m[key] for m in every.values()])) for key in METRICS}
        sd = {key: float(np.std([m[key] for m in every.values()], ddof=0)) for key in METRICS}
        pooled = checked_score(flatten(folds), FULL)
        totals[method] = {"mean": average, "sd": sd, "pooled": pooled}
        for scope, stat in (("macro_mean", average), ("macro_population_sd", sd), ("pooled", pooled)):
            rows.append({**record, "intervention": FULL, "scope": scope, **stat})
    return rows, totals, each


def id_text(rows):
    return " ".join(str(r["patient_id_1based"]) for r in rows)


def pair_stats(left, right, before, after):
    y, a, p = values(left, before)
    y_right, b, q = values(right, after)
    h.require(y == y_right and id_text(left) == id_text(right), "Paired patients or labels differ")
    lhs, rhs = score(y, a, p), score(y, b, q)
    corrected = [r for r, t, start, end in zip(right, y, a, b) if start != t and end == t]
    harmed = [r for r, t, start, end in zip(right, y, a, b) if start == t and end != t]
    changes = np.array([float(new[f"true_margin_{after}"]) - float(old[f"true_margin_{before}"])
                        for old, new in zip(left, right)])
    wrong = changes[np.asarray(a) != y]
    output = {
        "n_test": len(y), "n_asd": y.count(0), "n_td": y.count(1), "correct_before": lhs["correct"],
        "correct_after": rhs["correct"], "corrected": len(corrected), "harmed": len(harmed),
        "net": len(corrected) - len(harmed), "corrected_ids": id_text(corrected), "harmed_ids": id_text(harmed),
        "hard_flips": sum(start != end for start, end in zip(a, b)),
        "mean_p_asd_delta": float(np.mean(np.asarray(q) - p)),
        "mean_true_margin_delta": float(changes.mean()), "n_wrong_before": len(wrong),
        "mean_true_margin_delta_for_wrong_before": float(wrong.mean()) if len(wrong) else None,
    }
    for label, tag in ((0, "asd"), (1, "td")):
        for kind, selected in (("corrected", corrected), ("harmed", harmed)):
            affected = [r for r in selected if int(r["true_label"]) == label]
            output[f"{kind}_{tag}"] = len(affected)
            output[f"{kind}_{tag}_ids"] = id_text(affected)
    for k in METRICS:
        x, z = lhs[k], rhs[k]
        output.update({f"{k}_before_pct": 100 * x if x is not None else None,
                       f"{k}_after_pct": 100 * z if z is not None else None,
                       f"{k}_delta_pp": 100 * (z - x) if x is not None and z is not None else None})
    h.require(output["corrected"] + output["harmed"] == output["hard_flips"]
              and output["net"] == rhs["correct"] - lhs["correct"], "Paired confusion does not add up")
    return output


def paired_rows(all_folds, totals):
    rows, selected = [], {}

    def add(group, name, scope, left, right, before_method, after_method, before=FULL, after=FULL):
        rec = {"group": group, "comparison": name, "scope": scope, "before_method": before_method,
               "after_method": after_method, "before_mode": before, "after_mode": after,
               **pair_stats(left, right, before, after)}
        if scope == "ten_fold":
            a, b = (totals[m] for m in (before_method, after_method))
            rec.update({f"macro_{k}_delta_pp": 100 * (b["mean"][k] - a["mean"][k]) for k in METRICS})
            rec["pooled_prob_auc_delta_pp"] = 100 * (b["pooled"]["prob_auc"] - a["pooled"]["prob_auc"])
        selected[(group, name, scope)] = rec
        rows.append(rec)
        return rec

    for group in GROUPS:
        method = f"{METHOD}-{group}"
        folds = all_folds[method]
        test = folds[FOLD]
        scopes = {"fold8": test}
        for cohort, key in (("A", "patient_383"), ("B", "second_patient"), ("C", "unexposed_85")):
            scopes[key] = [r for r in test if r["fold8_cohort"] == cohort]
        scopes["unexposed_asd"] = [r for r in scopes["unexposed_85"] if r["true_label"] == 0]
        scopes["unexposed_td"] = [r for r in scopes["unexposed_85"] if r["true_label"] == 1]
        h.require(len(scopes["unexposed_asd"]) == 38 and len(scopes["unexposed_td"]) == 47, "Unexposed label counts differ")

        for source, tag in REFERENCES.items():
            baseline = all_folds[source]
            comparison = f"{tag}_to_full"
            all_stats = add(group, comparison, "ten_fold", flatten(baseline), flatten(folds), source, method)
            h.require(len(baseline[FOLD]) == len(test), "Fold 8 reference size differs")
            for scope, case_rows in scopes.items():
                ids = {r["patient_id_1based"] for r in case_rows}
                ref_subset = [r for r in baseline[FOLD] if int(r["patient_id_1based"]) in ids]
                add(group, comparison, scope, ref_subset, case_rows, source, method)
            if source == E9:
                local = selected[(group, comparison, "fold8")]
                h.require(all(all_stats[k] == local[k] for k in ("corrected", "harmed", "net", "hard_flips"))
                          and all(abs(all_stats[f"macro_{k}_delta_pp"] - local[f"{k}_delta_pp"] / 10) < 1e-10
                                  for k in METRICS), "Direct baseline changes include a non-Fold-8 row")

        for scope, case_rows in scopes.items():
            ids = {r["patient_id_1based"] for r in case_rows}
            baseline = [r for r in all_folds[E9][FOLD] if int(r["patient_id_1based"]) in ids]
            add(group, "cross_model_off_to_off", scope, baseline, case_rows, E9, method, OFF, OFF)
            add(group, "baseline_full_to_new_off", scope, baseline, case_rows, E9, method, FULL, OFF)
            add(group, "off_to_full", scope, case_rows, case_rows, method, method, OFF, FULL)
            add(group, "anonymous_to_medical", scope, case_rows, case_rows, method, method, ANON, FULL)
        for key in ("patient_383", "second_patient"):
            for intervention in ("off_to_full", "anonymous_to_medical"):
                r = selected[(group, intervention, key)]
                h.require(r["hard_flips"] == r["mean_p_asd_delta"] == r["mean_true_margin_delta"] == 0,
                          f"{group} {intervention} changed a masked patient")

    g1, g2 = f"{METHOD}-G1", f"{METHOD}-G2"
    add("G2", "g1_to_g2", "fold8", all_folds[g1][FOLD], all_folds[g2][FOLD], g1, g2)
    add("G2", "g1_to_g2", "ten_fold", flatten(all_folds[g1]), flatten(all_folds[g2]), g1, g2)
    return rows, selected


def table(columns, rows):
    lines = [columns, ["---"] * len(columns), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def counts(row):
    return f"{row['corrected']}/{row['harmed']}/{row['net']:+d}"


def ids(row, field):
    return row[field].replace(" ", "、") or "—"


def case_text(row, mode):
    return (f"{LABEL[int(row[f'prediction_{mode}'])]} / "
            f"{float(row[f'p_asd_{mode}']):.6f} / {float(row[f'true_margin_{mode}']):+.6f}")


def common_rng(manifest):
    tracks = {}
    for group in GROUPS:
        with paths(group)["journal"].open(encoding="utf-8") as stream:
            tracks[group] = [json.loads(line)["rng"] for line in stream]
        h.require(len(tracks[group]) + 10 == manifest["groups"][group]["training"]["epoch"],
                  f"{group} RNG ledger length differs")
    n = min(map(len, tracks.values()))
    h.require(tracks["G1"][:n] == tracks["G2"][:n], "Common epoch RNG trajectories differ")
    return {"status": "PASS", "common_epochs": n, "phase_comparisons": 3 * n,
            "common_first_epoch": 11, "common_last_epoch": n + 10,
            "trace_sha256": {g: h.sha(track) for g, track in tracks.items()}}


def scientific_cases(formal, paired):
    outcomes = {}
    for group, folds in formal.items():
        focus = {r["patient_id_1based"]: r for r in folds[FOLD]}
        second = GROUPS[group]["patient_ids_1based"][1]
        comparison = f"{BASE}_to_full"
        baseline = f"{BASE}_{FULL}"
        local = paired[(group, comparison, "fold8")]
        unexposed = paired[(group, comparison, "unexposed_85")]
        unexposed_e6 = paired[(group, "e466_g1_to_full", "unexposed_85")]
        single = CONFIG["single_patient_comparators"][group]
        versus_single = paired[(group, f"e469_{single.lower()}_to_full", "fold8")]
        residual = paired[(group, "off_to_full", "unexposed_85")]
        medical = paired[(group, "anonymous_to_medical", "unexposed_85")]
        a, b, td = (focus[i] for i in (383, second, 601))
        held = a[f"prediction_{baseline}"] == a["prediction_full"] == a["true_label"]
        corrected = b[f"prediction_{baseline}"] != b["true_label"] == b["prediction_full"]
        recovered = td[f"prediction_{baseline}"] != td["true_label"] == td["prediction_full"]
        outcomes[group] = {
            "patient_383_retained": held, "second_patient": second, "second_patient_corrected": corrected,
            "patient_601_recovered": recovered, "unexposed_corrected_ids": unexposed["corrected_ids"],
            "unexposed_harmed_ids": unexposed["harmed_ids"], "unexposed_net": unexposed["net"],
            "all_baseline_correct_unexposed_retained": unexposed["harmed"] == 0,
            "both_targets_correct_with_no_unexposed_harm_vs_direct_reference": held and corrected and unexposed["harmed"] == 0,
            "both_targets_correct_with_no_unexposed_harm_vs_e466": held and corrected and unexposed_e6["harmed"] == 0,
            "residual_fixed_weight_unexposed_hard_flips": residual["hard_flips"],
            "medical_query_unexposed_hard_flips": medical["hard_flips"],
            "fold8_correct": local["correct_after"], "fold8_net_vs_direct_reference": local["net"],
            "matched_single_patient_reference": SINGLE_METHODS[single], "fold8_net_vs_matched_single": versus_single["net"],
        }
    return outcomes


def build_report(manifest, audits, formal, totals, each, paired, conclusions, rng):
    parts = [TITLE, "## A. STATUS",
             table(("组", "状态", "Fold 8 Best", "Best epoch", "Latest epoch", "结束条件"), [
                 (g, a["status"], f"{a['best_metrics'][FULL]['correct']}/87", a["best_epoch"], a["latest_epoch"],
                  "wait > 200" if a["journal"]["stop"] == "early_stop" else "450 epochs")
                 for g, a in audits.items()]),
             f"两个正式任务仅使用物理 GPU0，UUID 为 `{manifest['device']['uuid']}`。"
             f"正式并发峰值 {manifest['peak_formal_workers']}，GPU 采样峰值 {manifest['gpu_training_peak_mib']} MiB。"
             "源实验 E82.4.6.9-G2、G4、G6 的 Fold 8 Best 和 Latest 均逐值重载。"
             "G2 的 383 纠正、601 误伤和 77/87 已在正式启动前复现。",
             "## B. 两组正式方案及成员",
             "两组从相同的 E61 Fold 8 epoch 10 模型、优化器与 RNG 状态恢复，seed 为 0，起始 epoch 为 11。"
             "SCMR-DA-SRPE 结构固定为 E82.4.6-G2，新组均采用 E82.4.6.4-G1 的 DARG-only 重叠监督。",
             table(("组", "按序追加 1-based", "按序追加 0-based", "DARG / 合法 / 测试", "ASD / TD", "类别权重 ASD / TD"), [
                 (g, "、".join(map(str, a["membership"]["patient_ids_1based"])),
                  "、".join(map(str, a["membership"]["patient_indices_0based"])),
                  f"{a['membership']['native_rows']} / {a['membership']['legal_scmr_gaussian_ema_rows']} / {a['membership']['test_rows']}",
                  " / ".join(map(str, a["membership"]["class_counts_asd_td"])),
                  " / ".join(f"{x:.8f}" for x in a["membership"]["class_weights_asd_td"]))
                 for g, a in audits.items()]),
             "原始 Fold 8 训练成员、图、特征、测试顺序和标签均通过核验。两名目标 ASD=0 参加 DARG 分类，"
             "其标签保持在 SCMR、Gaussian 和 EMA 合法监督集合外。"
             "历史九折沿 E82.4.6.9-G2 的来源链按 SHA 只读复用，涵盖 784 名测试患者。",
             "## C. 十折结果",
             "ASD 为阳性。六项十折折级宏平均均使用 population SD。末列为 871 人 pooled Prob-AUC。单位为 %。",
             table(("实验", *HEADINGS, "Pooled Prob-AUC"), [
                 (method, *(f"{s['mean'][k] * 100:.4f} ± {s['sd'][k] * 100:.4f}" for k in METRICS),
                  f"{s['pooled']['prob_auc'] * 100:.4f}") for method, s in totals.items()]),
             "十折相对 E82.4.6.9-G2 的配对变化。指标差值单位为百分点。",
             table(("组", "纠正 / 误伤 / 净", *(f"Δ {k}" for k in HEADINGS), "Δ Pooled AUC"), [
                 (g, counts(row), *(f"{row[f'macro_{k}_delta_pp']:+.4f}" for k in METRICS),
                  f"{row['pooled_prob_auc_delta_pp']:+.4f}")
                 for g in GROUPS for row in [paired[(g, f"{BASE}_to_full", "ten_fold")]]]),
             "## D. Fold 8 配对结果",
             table(("实验", "答对/87", "ASD 答对/40", "TD 答对/47", *HEADINGS), [
                 (method, s["correct"], s["tp"], s["tn"], *(f"{100*s[k]:.4f}" for k in METRICS))
                 for method, folds in each.items() for s in [folds[FOLD]]]),
             "Fold 8 相对 E82.4.6.9-G2 的配对变化。指标差值单位为百分点。",
             table(("组", "纠正 / 误伤 / 净", "纠正 ID", "误伤 ID", *(f"Δ {k}" for k in HEADINGS)), [
                 (g, counts(row), ids(row, "corrected_ids"), ids(row, "harmed_ids"),
                  *(f"{row[f'{k}_delta_pp']:+.4f}" for k in METRICS))
                 for g in GROUPS for row in [paired[(g, f"{BASE}_to_full", "fold8")]]]),
             table(("组", "Fold 8 队列", "人数", "纠正 / 误伤 / 净", "纠正 ID", "误伤 ID", "平均 Δ 真类 Margin"), [
                 (g, label, row["n_test"], counts(row), ids(row, "corrected_ids"), ids(row, "harmed_ids"),
                  f"{row['mean_true_margin_delta']:+.6f}") for g in GROUPS
                 for scope, label in (("patient_383", "A · 383"), ("second_patient", "B · 本组第二名"),
                                      ("unexposed_85", "C · 未加入 85 人"), ("unexposed_asd", "C · ASD"),
                                      ("unexposed_td", "C · TD"))
                 for row in [paired[(g, f"{BASE}_to_full", scope)]]]),
             table(("组", "Fold 8 对照", "纠正 / 误伤 / 净", "纠正 ID", "误伤 ID", "Δ BA pp", "Δ Prob-AUC pp"), [
                 (g, name, counts(row), ids(row, "corrected_ids"), ids(row, "harmed_ids"),
                  f"{row['hard_auc_delta_pp']:+.4f}", f"{row['prob_auc_delta_pp']:+.4f}")
                 for g in GROUPS for name, tag in ((E6, "e466_g1"), *(
                     (SINGLE_METHODS[key], f"e469_{key.lower()}") for key in ("G4", "G6")))
                 for row in [paired[(g, f"{tag}_to_full", "fold8")]]]),
             "## E. 重点患者变化",
             "预测列的顺序为类别 / ASD 概率 / 真类 Margin。残差列是 Full 的联合 / 谱内 / 谱外 L2。",
             table(("组", "ID", "真值", "E82.4.6.9-G2 Full", "Residual-off", "Anonymous", "Full",
                    "残差 L2", "DARG", "SCMR 合法", "屏蔽"), [
                 (g, i, LABEL[r["true_label"]], *(case_text(r, mode) for mode in (f"{BASE}_{FULL}", OFF, ANON, FULL)),
                  " / ".join(f"{r[k]:.6f}" for k in ("delta_l2", "delta_parallel_l2", "delta_perp_l2")),
                  "是" if r["darg_classification_supervised"] else "否",
                  "是" if r["scmr_label_supervised"] else "否", "是" if r["residual_masked"] else "否")
                 for g, folds in formal.items() for i in FOCUS_IDS
                 for r in folds[FOLD] if r["patient_id_1based"] == i]),
             "## F. 谱与医学 Prompt 作用",
             "跨模型 Off → Off 记录关闭直接残差时的重训变化，仍包含端到端联合训练对主干参数的影响。"
             "同一 Best 的 Off → Full 测量开启联合残差的直接推理变化。Anonymous → Medical 固定 Best 权重并替换 Query。",
             table(("组", "对比", "队列", "纠正 / 误伤 / 净", "纠正 ID", "误伤 ID", "硬翻转", "Δ Prob-AUC pp", "平均 Δ 真类 Margin"), [
                 (g, label, "87 人" if scope == "fold8" else "未加入 85 人", counts(row),
                  ids(row, "corrected_ids"), ids(row, "harmed_ids"), row["hard_flips"],
                  f"{row['prob_auc_delta_pp']:+.4f}", f"{row['mean_true_margin_delta']:+.6f}")
                 for g in GROUPS for name, label in (("cross_model_off_to_off", "跨模型 Off → Off"),
                                                     ("off_to_full", "同权重 Off → Full"),
                                                     ("anonymous_to_medical", "同权重 Anonymous → Medical"))
                 for scope in ("fold8", "unexposed_85") for row in [paired[(g, name, scope)]]]),
             table(("组", "85 人非零 联合 / 谱内 / 谱外 / Query", "均值 L2 联合 / 谱内 / 谱外 / Query",
                    "最大 Fold 8 残差比例", "最大全程共享差分梯度"), [
                 (g, " / ".join(str(sum(r[k] > 0 for r in cohort)) for k in keys),
                  " / ".join(f"{np.mean([r[k] for r in cohort]):.6f}" for k in keys),
                  f"{100*a['max_residual_ratio']:.6f}%",
                  f"{100*max(v for k, v in a['journal']['budget_maximum_ratio'].items() if k.endswith('.full')):.6f}%")
                 for g, a in audits.items() for cohort in [[r for r in formal[g][FOLD] if r["fold8_cohort"] == "C"]]
                 for keys in [("delta_l2", "delta_parallel_l2", "delta_perp_l2", "query_feature_change_l2")]]),
             "两名目标在训练与评估、Best 和 Latest 恢复及三种推理干预中的 raw_parallel、raw_perp、"
             "delta_parallel、delta_perp 和联合 delta 均严格为零，增强 CAM144 等于同次前向的原始 CAM144。"
             "同一 Best 的三种推理中，两个目标的 logits、概率和硬预测逐值一致。"
             "Jacobian 谱内/谱外分解、医学与匿名 Query 公式、合法 CAM 与端到端梯度通过数值核验。"
             "残差限幅 8%，共享差分梯度预算为 0.1%。",
             "## G. 独立审计",
             "关系对由历史 E61 采样器针对各自 786 人重建，4096 对中同类与异类各 2048 对。"
             "全局 RNG 不变。关系梯度为 Best 状态下四模态端点 embedding 的联合 L2。",
             table(("组", "患者", "覆盖人数 / 786", "覆盖率", "未覆盖人数", "出现次数（同类/异类）",
                    "分类 CAM144 L2", "直接关系 BCE L2", "直接加权关系 L2", "关系结果"), [
                 (g, patient, m["covered_rows"], f"{100*m['covered_fraction']:.2f}%", m["uncovered_rows"],
                  f"{pair['endpoint_occurrences']}（{pair['same_occurrences']}/{pair['different_occurrences']}）",
                  f"{grad['classification_cam144_grad_l2']:.6e}", f"{grad['relation_direct_bce_grad_l2']:.6e}",
                  f"{grad['relation_direct_weighted_grad_l2']:.6e}", a["relation_coverage"][patient])
                 for g, a in audits.items() for m in [a["membership"]] for patient, pair in m["target_pairs"].items()
                 for grad in [a["best_gradients"]["patients"][patient]]]),
             "\n".join(f"- {g} 未覆盖患者 " + "、".join(map(str, a["membership"]["uncovered_ids_1based"])) + "。"
                       for g, a in audits.items()),
             table(("组", "逐 epoch RNG 回放", "历史三阶段比对", "EMA 逐位回放", "双目标前向回执",
                    "Best / Latest 重载", "Latest 下一步训练 / 评估"), [
                 (g, a["journal"]["replayed_epochs"], a["journal"]["rng_phase_comparisons"],
                  a["journal"]["ema_bitwise_epochs"], a["mask_train_eval_receipts"],
                  "逐值一致", "逐值一致") for g, a in audits.items()]),
             f"两组共有 {rng['common_epochs']} 轮 RNG 完全一致，每轮均回放历史六次 Dropout。"
             f"训练前后 {manifest['guard_after']['files']} 项受保护历史文件 SHA 一致。"
             "Gaussian 合法成员独立重建，EMA 在训练、评估和恢复时保留原有更新时间点。"
             "冻结前的审计入口修正了未覆盖名单的比较顺序，采样和训练源码 SHA 均保持一致。",
             "## H. 文件与 Handoff",
             f"唯一权威目录 `{ROOT}`。首选 OSSFS 目录返回 errno 5。"
             "Manifest 绑定训练前冻结代码、来源、各组 Best/Latest、训练日志、独立审计、结果文件以及 Handoff 历史前缀和追加段落的 SHA。"
             "患者表含两组各 871 行，配对表含双患者分层、历史基线和固定权重干预。",
             "### 诊断判断"]
    for group, r in conclusions.items():
        yn = lambda value: "是" if value else "否"
        parts.append(f"- {group} 的 383 保留纠错 **{yn(r['patient_383_retained'])}**，"
                     f"第二名 {r['second_patient']} 纠正 **{yn(r['second_patient_corrected'])}**，"
                     f"601 恢复 **{yn(r['patient_601_recovered'])}**。"
                     f"未加入 85 人的净变化为 {r['unexposed_net']:+d}，相对直接基线保护全部原答对未暴露患者 "
                     f"**{yn(r['all_baseline_correct_unexposed_retained'])}**。"
                     f"两名目标均正确且无未暴露误伤的要求，相对直接基线 **{yn(r['both_targets_correct_with_no_unexposed_harm_vs_direct_reference'])}**，"
                     f"相对 E82.4.6.6-G1 **{yn(r['both_targets_correct_with_no_unexposed_harm_vs_e466'])}**。"
                     f"相对 {r['matched_single_patient_reference']}，Fold 8 净变化 {r['fold8_net_vs_matched_single']:+d} 人。")
    parts.append("独立泛化与 MICCAI 2027 第二创新模块正式判断均为 **NO-GO**。"
                 "有意训练—测试标签重叠、val=test 选模、CV 前 RFE/预处理及转导图将结果限定为机制诊断。")
    return "\n\n".join(parts) + "\n"


def audit_outputs(manifest, audits, history, reuse, totals, rng, conclusions):
    status = ("COMPLETE_AUDITED" if all(a["status"] == "COMPLETE_AUDITED" for a in audits.values())
              else "AUDITED_WITH_RELATION_COVERAGE_LIMITATION")
    group_receipts = {}
    for group, a in audits.items():
        preflight = h.read_json(paths(group)["preflight"])
        receipt = {key: a[key] for key in (
            "status", "artifact_sha256", "best_epoch", "latest_epoch", "best_reloaded", "latest_reloaded",
            "latest_next_training_exact", "latest_next_evaluation_exact", "latest_dual_mask_exact", "best_metrics",
            "relation_coverage", "relation_coverage_limited_ids_1based", "mask_train_eval_receipts",
            "target_cam144_exact_all_interventions", "target_logits_and_probabilities_exact_all_interventions",
            "other_85_residuals_active", "max_residual_ratio",
        )}
        group_receipts[group] = {
            **receipt, "certificate_path": manifest["groups"][group]["audit"]["path"],
            "certificate_sha256": manifest["groups"][group]["audit"]["sha256"],
            "gaussian_reconstruction_exact": a["gaussian"]["deterministic_rebuild_exact"],
            "historical_rng_phase_comparisons": a["journal"]["rng_phase_comparisons"],
            "replayed_epochs": a["journal"]["replayed_epochs"], "ema_bitwise_epochs": a["journal"]["ema_bitwise_epochs"],
            "historical_dropout_calls_per_epoch": len(preflight["dropout"]),
            "legal_label_isolation": preflight["legal_label_isolation"], "ten_fold": totals[f"{METHOD}-{group}"],
            "diagnosis": conclusions[group],
        }
    history_time = manifest["historical_replay"]["completed_at_utc"]
    launch_time = manifest["first_formal_launch_utc"]
    h.require(history_time < manifest["preflight"]["completed_at_utc"] < launch_time,
              "Historical evaluation or preflight happened after official launch")
    audit_status = {
        "status": status, "experiment": CONFIG["experiment"], "direct_reference": E9,
        "scientific_decision": SCIENTIFIC_DECISION,
        "historical_replay_completed_before_formal_training": True, "historical_replay_completed_at_utc": history_time,
        "protected_history": {key: manifest[key] for key in ("guard_before", "guard_prelaunch", "guard_preaudit", "guard_after")},
        "independent_patient_metrics_recalculation": True, "reused_patient_rows_per_group": 784,
        "common_rng": rng, "groups": group_receipts,
    }
    membership = {
        "status": "PASS", "experiment": CONFIG["experiment"], "label_encoding": CONFIG["label_encoding"],
        "fold8_original": history["folds"][str(FOLD)], "direct_reference": history["single_patient_reference"]["G2"],
        "nine_fold_reuse": reuse,
        "groups": {group: {
            "membership": a["membership"], "gaussian": a["gaussian"],
            "preflight_gradients": h.read_json(paths(group)["preflight"])["gradients"],
            "best_gradients": a["best_gradients"], "latest_gradients": a["latest_gradients"],
            "relation_coverage": a["relation_coverage"],
        } for group, a in audits.items()},
    }
    return audit_status, membership


def write_outputs(manifest, audits):
    archive = historical_rows()
    history = h.read_json(controller.HISTORY_REPLAY)
    formal = {group: patient_rows(group, archive, history, audits[group]) for group in GROUPS}
    all_folds = {**archive, **{f"{METHOD}-{g}": folds for g, folds in formal.items()}}
    reuse = reused_proof(formal, archive, history)
    fold_metrics, totals, each = all_metrics(all_folds, history, audits)
    paired, selection = paired_rows(all_folds, totals)
    rng = common_rng(manifest)
    conclusions = scientific_cases(formal, selection)
    audit_status, membership = audit_outputs(manifest, audits, history, reuse, totals, rng, conclusions)
    report = build_report(manifest, audits, formal, totals, each, selection, conclusions, rng)
    for filename, rows in {
        "patient_level.csv": [row for f in formal.values() for row in flatten(f)],
        "fold_metrics.csv": fold_metrics, "paired_comparisons.csv": paired,
    }.items():
        h.write_csv(RESULTS / filename, rows)
    h.write_json(RESULTS / "membership_audit.json", membership)
    h.write_json(RESULTS / "audit_status.json", audit_status)
    with h.archive.atomic_writer(RESULTS / "report.md", "w") as stream:
        stream.write(report)
    h.sync_dir(RESULTS)
    return audit_status["status"], report, reuse, rng, conclusions


def digest(data):
    return hashlib.sha256(data).hexdigest()


def append_handoff(manifest, report, artifacts):
    headings = "\n".join(f"- `{name}` `{sha}`" for name, sha in sorted(artifacts.items()))
    appendix = ("\n" + report.replace(TITLE, MARKER, 1) + "\n### 审计文件 SHA-256\n\n" + headings + "\n").encode()
    with HANDOFF.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        previous = stream.read()
        intent = manifest.get("handoff_intent")
        if intent is None:
            h.require(MARKER.encode() not in previous, "Unregistered E82.4.6.10 Handoff section already exists")
            intent = {
                "path": str(HANDOFF), "prefix_bytes": len(previous), "prefix_sha256": digest(previous),
                "appended_bytes": len(appendix), "appended_sha256": digest(appendix), "marker": MARKER,
                "section_line": previous.count(b"\n") + 2,
            }
            manifest["handoff_intent"] = intent
            h.write_json(MANIFEST, manifest)
            h.sync_dir(ROOT)
        start, length = intent["prefix_bytes"], intent["appended_bytes"]
        h.require(digest(previous[:start]) == intent["prefix_sha256"] and len(previous) >= start
                  and len(appendix) == length and digest(appendix) == intent["appended_sha256"],
                  "Handoff history or intended appendix differs")
        tail = previous[start:start + length]
        h.require(appendix.startswith(tail), "Unexpected bytes after the verified Handoff prefix")
        if len(tail) < length:
            stream.write(appendix[len(tail):])
            stream.flush()
            os.fsync(stream.fileno())
        readback = HANDOFF.read_bytes()
        h.require(readback[:start] == previous[:start] and readback[start:start + length] == appendix
                  and digest(readback[:start]) == intent["prefix_sha256"]
                  and digest(readback[start:start + length]) == intent["appended_sha256"],
                  "Independent Handoff readback differs")
    return {**intent, "historical_prefix_preserved": True, "verified_at_utc": utc_now(),
            "sha256_after_append": digest(readback)}


def verify_published(manifest):
    for relative, expected in {**manifest["result_files"], **manifest["published_source_sha256"]}.items():
        h.require(h.sha_file(ROOT / relative) == expected, f"Published file SHA differs: {relative}")
    proof = manifest["handoff"]
    data = HANDOFF.read_bytes()
    start, length = proof["prefix_bytes"], proof["appended_bytes"]
    h.require(digest(data[:start]) == proof["prefix_sha256"]
              and digest(data[start:start + length]) == proof["appended_sha256"],
              "Published Handoff prefix or appendix differs")


def main():
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = h.read_json(MANIFEST)
        controller.require_preflight(manifest)
        guard = verify_guard(manifest["protected_sha256"])
        h.require(guard == manifest["guard_before"] == manifest["guard_after"], "Historical file protection differs")
        if "handoff" in manifest:
            verify_published(manifest)
            print(f"Verified prior publication {manifest['status']}", flush=True)
            return
        h.require(manifest["status"] == "AUDITED", "Independent audits must reach the publication gate")
        audits = {}
        for group in GROUPS:
            files = paths(group)
            receipt = manifest["groups"][group]["audit"]
            training = controller.verify_completed(group)
            audit = audits[group] = h.read_json(files["audit"])
            h.require(receipt["status"] == audit["status"] and audit["status"] in controller.COMPLETE
                      and h.sha_file(files["audit"]) == receipt["sha256"]
                      and receipt["path"] == str(files["audit"]) and training is not None
                      and all(h.sha_file(files[k]) == digest for k, digest in audit["artifact_sha256"].items()),
                      f"{group} formal audit or cited artifacts differ")
        status, report, reuse, rng, conclusions = write_outputs(manifest, audits)
        artifacts = {f"results/{f}": h.sha_file(RESULTS / f) for f in RESULT_NAMES}
        manifest.update(result_files=artifacts, nine_fold_reuse=reuse, common_rng=rng,
                        published_source_sha256={"publish.py": h.sha_file(ROOT / "publish.py")},
                        scientific_decision=SCIENTIFIC_DECISION, diagnostic_conclusions=conclusions)
        manifest["handoff"] = append_handoff(manifest, report, artifacts)
        manifest.update(status=status, published_at_utc=utc_now())
        h.write_json(MANIFEST, manifest)
        h.sync_dir(ROOT)
        verify_published(h.read_json(MANIFEST))
        print(f"PUBLISHED {status}, with all historical and Handoff hashes verified", flush=True)


if __name__ == "__main__":
    main()
