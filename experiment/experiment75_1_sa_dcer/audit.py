"""Independent readback audit, fixed-checkpoint interventions and result synthesis."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
BASE_DIR = Path("/model/spromptgl/experiment/experiment74_4_soft_positive_dcr")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
CORE_METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
GROUPS = ("G1-Anchor", "G2-Faithful", "G3-Anonymous")


def runner():
    import importlib.util
    import sys
    name = "sa_dcer_751_runner_for_independent_audit"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, EXP / "run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def metric_summary(rows, group, scope=None):
    import numpy as np
    selected = [row for row in rows if row["group"] == group and (scope is None or int(row["fold"]) != 2)]
    if not selected:
        raise RuntimeError(f"missing metric rows for {group}")
    return {metric: {"mean": float(np.mean([float(row[metric]) for row in selected])),
                     "sd": float(np.std([float(row[metric]) for row in selected], ddof=0))}
            for metric in METRICS}


def _correlation(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def metric_record(metrics):
    return {key: float(metrics[key]) for key in METRICS}


def evaluate_intervention(r, state, dcer, group, intervention, reference, labels, indices, full_h):
    import numpy as np
    import torch
    with torch.no_grad():
        output, capture = r.hooked_forward(state, dcer, group, training=False, intervention=intervention)
        changed = r.rng_runtime().classification_metrics(output["logp"], state["test"], state["y"])
    full_pred = np.asarray(reference["pred"], dtype=int)
    changed_pred = np.asarray(changed["pred"], dtype=int)
    truth = np.asarray(labels, dtype=int)
    full_correct, changed_correct = full_pred == truth, changed_pred == truth
    fprob, cprob = np.asarray(reference["prob"], dtype=float), np.asarray(changed["prob"], dtype=float)
    hprime = capture["dcer"]["h_prime"].detach()
    h_rel = float((hprime - full_h).norm() / full_h.norm().clamp_min(1e-12))
    row = {"group": group, "fold": int(state["fold"]), "intervention": intervention,
           "hard_flips": int((full_pred != changed_pred).sum()),
           "full_wrong_intervention_correct": int((~full_correct & changed_correct).sum()),
           "full_correct_intervention_wrong": int((full_correct & ~changed_correct).sum()),
           "mean_abs_asd_probability_change": float(np.abs(fprob[:, 0] - cprob[:, 0]).mean()),
           "cam_feature_relative_l2_change_from_full": h_rel,
           "formal_outputlayer_calls": capture["formal_outputlayer_calls"],
           "flipped_test_rows_1based": json.dumps((np.asarray(indices)[full_pred != changed_pred] + 1).astype(int).tolist())}
    for metric in METRICS:
        row[f"full_{metric}"] = float(reference[metric])
        row[f"intervention_{metric}"] = float(changed[metric])
        row[f"full_minus_intervention_{metric}_pp"] = 100.0 * (float(reference[metric]) - float(changed[metric]))
    return row, output, capture, changed


def checkpoint_mechanism(r, state, dcer, group, capture, task, logs):
    import numpy as np
    import torch
    api = r.load_module()
    result = capture["dcer"]
    h = capture["h"].detach()
    tokens = result["tokens"].detach()
    difference = result["difference"].detach()
    context = result["global_context"].detach()
    output_layer = state["cam"].Outputlayer
    train_teacher = api.signed_margin_teacher(output_layer, h, state["train"])
    test_index = torch.as_tensor(state["test"], dtype=torch.long, device=h.device)
    test_teacher = api.signed_margin_teacher(output_layer, h, test_index)
    train_ids = state["train"]
    test_ids = test_index
    d_train = difference.index_select(0, train_ids).detach()
    d_test = difference.index_select(0, test_ids).detach()
    train_abs = train_teacher.abs().reshape(-1).cpu().numpy()
    test_abs = test_teacher.abs().reshape(-1).cpu().numpy()
    confidence = result["confidence"].detach()
    confidence_train = confidence.index_select(0, train_ids).reshape(-1).cpu().numpy()
    confidence_test = confidence.index_select(0, test_ids).reshape(-1).cpu().numpy()
    difference_train = d_train.reshape(-1).cpu().numpy()
    difference_test = d_test.reshape(-1).cpu().numpy()
    signed_train = train_teacher.reshape(-1).cpu().numpy()
    signed_test = test_teacher.reshape(-1).cpu().numpy()
    nonzero_train = np.abs(signed_train) >= 0.05
    nonzero_test = np.abs(signed_test) >= 0.05
    dmod = difference.reshape(-1, 4, 4).detach().cpu().numpy()
    token_attn_asd = torch.softmax(result["score_asd"], dim=1)
    token_attn_td = torch.softmax(result["score_td"], dim=1)
    token_entropy_asd = -(token_attn_asd.clamp_min(1e-12) * token_attn_asd.clamp_min(1e-12).log()).sum(dim=1) / math.log(16.0)
    token_entropy_td = -(token_attn_td.clamp_min(1e-12) * token_attn_td.clamp_min(1e-12).log()).sum(dim=1) / math.log(16.0)
    prompt_weights = result["prompt_weights"].detach().mean(dim=(0, 1)).cpu().numpy()
    query = result["query"].detach()
    q_asd = torch.nn.functional.normalize(query[:, 4:8].mean(dim=1), dim=-1)
    q_td = torch.nn.functional.normalize(query[:, :4].mean(dim=1), dim=-1)
    query_cos = (q_asd * q_td).sum(dim=-1).mean().item()
    residual = result["residual"].detach().reshape(h.shape[0], 4, 4, 9)
    token_h = h.reshape(h.shape[0], 4, 4, 9)
    modality_ratios = []
    for modality in range(4):
        modality_ratios.append(float(residual[:, modality].norm() / token_h[:, modality].norm().clamp_min(1e-12)))
    with torch.no_grad():
        base_logits = api._deterministic_logits(output_layer, h)
        enhanced_logits = api._deterministic_logits(output_layer, result["h_prime"].detach())
    margin_change = float(((enhanced_logits[:, 0] - enhanced_logits[:, 1]) - (base_logits[:, 0] - base_logits[:, 1])).mean())
    gamma_path = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines() if line.strip()] if logs.is_file() else []
    gamma_trace = [float(row["gamma"]) for row in gamma_path]
    train_high = train_abs >= np.quantile(train_abs, 0.75)
    train_low = train_abs <= np.quantile(train_abs, 0.25)
    test_high = test_abs >= np.quantile(test_abs, 0.75)
    test_low = test_abs <= np.quantile(test_abs, 0.25)
    row = {
        "group": group, "fold": int(state["fold"]), "best_epoch_1based": int(task["best_epoch_1based"]),
        "gamma_initial": 0.05, "gamma_best": float(result["gamma"].detach().cpu()),
        "gamma_log_samples": json.dumps(gamma_trace), "gamma_log_min": min(gamma_trace) if gamma_trace else float(result["gamma"].detach()),
        "gamma_log_max": max(gamma_trace) if gamma_trace else float(result["gamma"].detach()),
        "residual_cam_l2_ratio": float(result["residual"].norm() / h.norm().clamp_min(1e-12)),
        "bounded_residual_cam_l2_ratio": float(result["bounded_residual"].norm() / h.norm().clamp_min(1e-12)),
        "h_prime_relative_change": float((result["h_prime"].detach() - h).norm() / h.norm().clamp_min(1e-12)),
        "mean_signed_evidence": float(difference.mean()), "mean_abs_signed_evidence": float(difference.abs().mean()),
        "signed_evidence_sd": float(difference.std(unbiased=False)), "query_class_cosine": float(query_cos),
        "train_teacher_mean": float(train_teacher.mean()), "train_teacher_sd": float(train_teacher.std(unbiased=False)),
        "test_teacher_mean_unlabeled": float(test_teacher.mean()), "test_teacher_sd_unlabeled": float(test_teacher.std(unbiased=False)),
        "train_direction_correlation": _correlation(difference_train, signed_train),
        "test_direction_correlation_unlabeled": _correlation(difference_test, signed_test),
        "train_direction_agreement_nonzero": float((np.sign(difference_train[nonzero_train]) == np.sign(signed_train[nonzero_train])).mean()) if nonzero_train.any() else 0.0,
        "test_direction_agreement_nonzero_unlabeled": float((np.sign(difference_test[nonzero_test]) == np.sign(signed_test[nonzero_test])).mean()) if nonzero_test.any() else 0.0,
        "train_nonzero_teacher_fraction": float(nonzero_train.mean()), "test_nonzero_teacher_fraction_unlabeled": float(nonzero_test.mean()),
        "confidence_mean": float(confidence.mean()), "confidence_sd": float(confidence.std(unbiased=False)),
        "confidence_min": float(confidence.min()), "confidence_max": float(confidence.max()),
        "confidence_fraction_lt_0p05": float((confidence < 0.05).float().mean()),
        "confidence_fraction_gt_0p95": float((confidence > 0.95).float().mean()),
        "confidence_teacher_correlation_train": _correlation(confidence_train, train_abs),
        "confidence_teacher_correlation_test_unlabeled": _correlation(confidence_test, test_abs),
        "train_teacher_high_low_abs_difference": float(train_abs[train_high].mean() - train_abs[train_low].mean()),
        "test_teacher_high_low_abs_difference_unlabeled": float(test_abs[test_high].mean() - test_abs[test_low].mean()),
        "token_attention_entropy_asd_normalized": float(token_entropy_asd.mean()),
        "token_attention_entropy_td_normalized": float(token_entropy_td.mean()),
        "token_attention_asd_max_mean": float(token_attn_asd.max(dim=1).values.mean()),
        "token_attention_td_max_mean": float(token_attn_td.max(dim=1).values.mean()),
        "prompt_weights_by_index": json.dumps(prompt_weights.tolist()),
        "mean_class_margin_change": margin_change,
        "modality_residual_cam_ratios": json.dumps(modality_ratios),
        "training_l_dir_at_best": float(task["losses"]["l_dir"]),
        "training_l_conf_at_best": float(task["losses"]["l_conf"]),
        "classification_gradient_at_best": json.dumps(task["best_grad_snapshot"], sort_keys=True),
    }
    for i, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        row[f"mean_signed_evidence_{name}"] = float(dmod[:, i].mean())
        row[f"mean_abs_signed_evidence_{name}"] = float(np.abs(dmod[:, i]).mean())
        row[f"residual_cam_ratio_{name}"] = modality_ratios[i]
        row[f"prompt_weight_{name}_by_index"] = json.dumps(prompt_weights.tolist())
    return row


def audit_one(group, fold, expected_source):
    import numpy as np
    import torch
    r = runner()
    task_file, checkpoint_file = r.task_path(group, fold), r.best_path(group, fold)
    if not r._task_complete(group, fold, expected_source):
        raise RuntimeError(f"incomplete formal task {group} fold {fold}")
    task = json.loads(task_file.read_text(encoding="utf-8"))
    checkpoint_sha = r.sha_file(checkpoint_file)
    checkpoint = r.torch_load(checkpoint_file)
    if checkpoint.get("experiment") != r.CFG["experiment"] or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError("checkpoint experiment/group/fold mismatch")
    if checkpoint.get("source_sha256") != expected_source or checkpoint.get("config") != r.CFG:
        raise RuntimeError("checkpoint source/config mismatch")
    if task.get("checkpoint_sha256") != checkpoint_sha or task.get("best_epoch_1based") != checkpoint.get("best_epoch_1based"):
        raise RuntimeError("task/checkpoint SHA or best epoch mismatch")
    state, warmup, identity, base_task = r.verified_prepare(fold)
    if identity != checkpoint.get("identity") or identity != task.get("identity"):
        raise RuntimeError("warmup or fold identity differs on independent reconstruction")
    if base_task["checkpoint_sha256"] != checkpoint.get("base_task_checkpoint_sha256") or checkpoint.get("base_best_checkpoint_used_for_training") is not False:
        raise RuntimeError("base checkpoint provenance mismatch; training must start from warmup")
    overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != overlap:
        raise RuntimeError("historical Fold-2 overlap mismatch")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all() or state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError("training relation-pair scope/count audit failed")
    bank, anonymous, projection, artifact_meta = r.fixed_fold_artifacts(fold)
    device = next(state["cam"].parameters()).device
    module, _ = r.init_sa_dcer(anonymous if group == "G3-Anonymous" else bank, projection, fold, device)
    module.load_state_dict(checkpoint["sa_dcer_state"], strict=True)
    optimizer = r.new_optimizer(module)
    optimizer.load_state_dict(checkpoint["sa_dcer_optimizer"])
    r.load_base_bundle(state, checkpoint["base_state"])
    state["model"].eval(); state["cam"].eval(); module.eval()
    if task["init_meta"]["parameter_sha256"] != checkpoint["init_meta"]["parameter_sha256"]:
        raise RuntimeError("initial module parameter SHA provenance mismatch")
    if not task["init_meta"].get("main_rng_unchanged") or task["init_meta"]["main_rng_sha_before_init"] != task["init_meta"]["main_rng_sha_after_init"]:
        raise RuntimeError("private SA-DCER initialization changed main RNG")
    if task.get("init_meta") != checkpoint.get("init_meta") or artifact_meta["projection_sha256"] != checkpoint["init_meta"]["projection"]["projection_sha256"]:
        raise RuntimeError("fixed projection or module initialization metadata mismatch")
    rng_before = r.rng_runtime().pack_rng()
    with torch.no_grad():
        full_output, full_capture = r.hooked_forward(state, module, group, training=False)
        full_metrics = r.rng_runtime().classification_metrics(full_output["logp"], state["test"], state["y"])
    rng_after = r.rng_runtime().pack_rng()
    if r.rng_runtime().rng_sha(rng_before) != r.rng_runtime().rng_sha(rng_after):
        raise RuntimeError("evaluation/intervention consumed the inherited RNG")
    if full_capture["formal_outputlayer_calls"] != 1:
        raise RuntimeError("best checkpoint forward did not use exactly one OutputLayer")
    saved_pred = np.asarray(checkpoint["test_prediction"], dtype=int)
    saved_prob = np.asarray(checkpoint["test_probability"], dtype=float)
    replay_pred = np.asarray(full_metrics["pred"], dtype=int)
    replay_prob = np.asarray(full_metrics["prob"], dtype=float)
    if not np.array_equal(saved_pred, replay_pred) or float(np.max(np.abs(saved_prob - replay_prob))) > 1e-8:
        raise RuntimeError("independent checkpoint predictions differ from training record")
    for metric in METRICS:
        if abs(float(checkpoint["metrics"][metric]) - float(full_metrics[metric])) > 1e-8:
            raise RuntimeError(f"metric mismatch after checkpoint readback: {metric}")
    best_trace = next((x for x in checkpoint["rng_trace"] if int(x["epoch"]) == int(checkpoint["best_epoch_1based"])), None)
    if best_trace is None:
        raise RuntimeError("best-epoch RNG trace record missing")
    if r.rng_runtime().rng_sha(checkpoint["best_rng_after_train"]) != best_trace["rng_after_train"]:
        raise RuntimeError("best training RNG snapshot does not match its trace SHA")
    if r.rng_runtime().rng_sha(checkpoint["best_rng_after_eval"]) != best_trace["rng_after_eval"]:
        raise RuntimeError("best evaluation RNG snapshot does not match its trace SHA")

    labels = state["labels_np"][state["test_np"]].astype(int)
    interventions, intervention_outputs = [], {}
    for name in ("dcer_off", "confidence_neutral", "prompt_class_swap", "prompt_within_class_shuffle"):
        row, changed_output, changed_capture, _ = evaluate_intervention(
            r, state, module, group, name, full_metrics, labels, state["test_np"],
            full_capture["dcer"]["h_prime"].detach()
        )
        if name == "prompt_within_class_shuffle":
            if float(row["cam_feature_relative_l2_change_from_full"]) > 3e-6:
                raise RuntimeError("within-class prompt shuffling changed CAM features beyond numerical tolerance")
        if name == "dcer_off":
            if float(changed_capture["dcer"]["residual"].abs().max().detach().cpu()) != 0.0:
                raise RuntimeError("dcer_off failed to set the SA-DCER residual exactly to zero")
        interventions.append(row)
        intervention_outputs[name] = {"metrics": metric_record(r.rng_runtime().classification_metrics(changed_output["logp"], state["test"], state["y"])),
                                      "capture": changed_capture}

    mechanism = checkpoint_mechanism(r, state, module, group, full_capture, task, r.log_path(group, fold))
    audit = {
        "group": group, "fold": fold, "status": "PASS", "checkpoint_sha256": checkpoint_sha,
        "source_sha256": expected_source, "base_checkpoint_sha256": base_task["checkpoint_sha256"],
        "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
        "base_model_cam_sha": identity["model_cam_sha"], "base_optimizer_sha": identity["optimizer_sha"],
        "base_train_pair_sha": identity["pair_sha"], "overlap_1based": json.dumps(overlap),
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]), "epochs_executed": int(checkpoint["epochs_executed"]),
        "metric_replay_max_abs_error": max(abs(float(checkpoint["metrics"][k]) - float(full_metrics[k])) for k in METRICS),
        "prediction_exact": True, "probability_max_abs_error": float(np.max(np.abs(saved_prob - replay_prob))),
        "formal_outputlayer_calls": int(full_capture["formal_outputlayer_calls"]),
        "base_best_checkpoint_used_for_training": False,
    }
    return audit, {"group": group, "fold": fold, **metric_record(full_metrics)}, mechanism, interventions, task, checkpoint


def _paired(rows_a, rows_b, name, scope):
    output = []
    aa = {int(r["fold"]): r for r in rows_a}
    bb = {int(r["fold"]): r for r in rows_b}
    folds = [f for f in sorted(set(aa) & set(bb)) if scope != "exclude_fold2_descriptive_9" or f != 2]
    for metric in METRICS:
        diffs = [float(aa[f][metric]) - float(bb[f][metric]) for f in folds]
        output.append({"comparison": name, "scope": scope, "metric": metric,
                       "mean_delta_pp": 100 * sum(diffs) / len(diffs),
                       "wins": sum(d > 1e-12 for d in diffs), "ties": sum(abs(d) <= 1e-12 for d in diffs),
                       "losses": sum(d < -1e-12 for d in diffs), "folds": len(folds),
                       "fold_deltas_pp": json.dumps({str(f): 100 * (float(aa[f][metric]) - float(bb[f][metric])) for f in folds}, sort_keys=True)})
    return output


def build_results(audits, metrics, mechanisms, interventions, tasks, checkpoints):
    r = runner()
    source_hash = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))["source_sha256"]
    task_pairs = [(g, f) for f in range(1, 11) for g in GROUPS]
    if len(audits) != 30 or sum(row["status"] == "PASS" for row in audits) != 30:
        raise RuntimeError("not all 30 checkpoints passed independent audit")
    if len({(row["group"], row["fold"]) for row in audits}) != 30 or set((x["group"], x["fold"]) for x in audits) != set(task_pairs):
        raise RuntimeError("duplicate or missing checkpoint audits")
    training_tasks = {(g, f): task for g, f, task in tasks}
    init_hashes, rng_checks = {}, {"group_vs_group": 0, "vs_e74_4_g2": 0}
    for fold in range(1, 11):
        group_hashes = {training_tasks[(g, fold)]["init_meta"]["parameter_sha256"] for g in GROUPS}
        if len(group_hashes) != 1:
            raise RuntimeError(f"fold {fold} initial trainable parameters differ between groups")
        base_ids = {r.sha_json(training_tasks[(g, fold)]["identity"]) for g in GROUPS}
        if len(base_ids) != 1:
            raise RuntimeError(f"fold {fold} base initialization differs between groups")
        init_hashes[str(fold)] = next(iter(group_hashes))
        traces = {g: training_tasks[(g, fold)]["rng_trace"] for g in GROUPS}
        e74_task = json.loads((BASE_DIR / ".runtime/tasks" / f"E74.4-G2-Positive-DCR_f{fold:02d}.json").read_text(encoding="utf-8"))
        e74_trace = {int(x["epoch"]): x for x in e74_task["rng_trace"]}
        common_epochs = set.intersection(*[{int(x["epoch"]) for x in trace} for trace in traces.values()], set(e74_trace))
        for epoch in common_epochs:
            source = e74_trace[epoch]
            for group in GROUPS:
                current = next(x for x in traces[group] if int(x["epoch"]) == epoch)
                same = all(current[key] == source[key] for key in ("rng_before", "rng_after_train", "rng_after_eval"))
                if not same:
                    raise RuntimeError(f"RNG trace mismatch {group} fold={fold} epoch={epoch}")
                rng_checks["vs_e74_4_g2"] += 3
        for epoch in common_epochs:
            values = [next(x for x in traces[group] if int(x["epoch"]) == epoch) for group in GROUPS]
            for key in ("rng_before", "rng_after_train", "rng_after_eval"):
                if len({v[key] for v in values}) != 1:
                    raise RuntimeError(f"group RNG trace mismatch fold={fold} epoch={epoch} {key}")
                rng_checks["group_vs_group"] += 3

    base_rows = [row for row in read_csv(BASE_DIR / "results/fold_metrics.csv") if row["group"] == "E74.4-G2-Positive-DCR"]
    if len(base_rows) != 10:
        raise RuntimeError("E74.4-G2 historical fold metrics not complete")
    metric_rows = sorted(metrics, key=lambda x: (x["group"], x["fold"]))
    all_metric_rows = metric_rows + [{"group": "E74.4-G2 historical", "fold": int(x["fold"]), **{k: float(x[k]) for k in METRICS}} for x in base_rows]
    summaries = []
    for group in ("E74.4-G2 historical",) + GROUPS:
        for scope in ("all_10", "exclude_fold2_descriptive_9"):
            selected = [x for x in all_metric_rows if x["group"] == group and (scope == "all_10" or x["fold"] != 2)]
            row = {"group": group, "scope": scope, "n_folds": len(selected)}
            for metric in METRICS:
                vals = [100.0 * float(x[metric]) for x in selected]
                row[f"{metric}_mean_pct"] = statistics.fmean(vals)
                row[f"{metric}_population_sd_pct"] = statistics.pstdev(vals)
            summaries.append(row)
    paired = []
    for g in GROUPS:
        paired += _paired([x for x in metric_rows if x["group"] == g], base_rows, f"{g} - E74.4-G2", "all_10")
        paired += _paired([x for x in metric_rows if x["group"] == g], base_rows, f"{g} - E74.4-G2", "exclude_fold2_descriptive_9")
    for ga, gb in (("G2-Faithful", "G1-Anchor"), ("G2-Faithful", "G3-Anonymous")):
        paired += _paired([x for x in metric_rows if x["group"] == ga], [x for x in metric_rows if x["group"] == gb], f"{ga} - {gb}", "all_10")
        paired += _paired([x for x in metric_rows if x["group"] == ga], [x for x in metric_rows if x["group"] == gb], f"{ga} - {gb}", "exclude_fold2_descriptive_9")

    r.write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    r.write_csv(RESULTS / "summary.csv", summaries)
    r.write_csv(RESULTS / "paired_comparisons.csv", paired)
    r.write_csv(RESULTS / "mechanism.csv", mechanisms)
    r.write_csv(RESULTS / "interventions.csv", interventions)
    r.write_csv(RESULTS / "checkpoint_audit.csv", audits)
    audit_status = {"status": "PASS", "expected_checkpoints": 30, "independently_audited": 30,
                    "checkpoint_metric_replay": "30/30 exact hard predictions, probability max error <=1e-8",
                    "base_initialization_groups_equal_per_fold": True, "warmup_parity": "10/10",
                    "rng_shared_epoch_checks": rng_checks, "source_sha256": source_hash,
                    "fold2_overlap_1based": [377, 407], "audit_completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    r.atomic_json(RESULTS / "audit_status.json", audit_status)
    return summaries, paired, audit_status, all_metric_rows


def format_summary_table(summaries):
    by = {(row["group"], row["scope"]): row for row in summaries}
    groups = ("E74.4-G2 historical",) + GROUPS
    lines = ["|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        row = by[(group, "all_10")]
        vals = []
        for metric in CORE_METRICS:
            vals.append(f"{row[metric+'_mean_pct']:.4f} ± {row[metric+'_population_sd_pct']:.4f}")
        lines.append("|" + group + "|" + "|".join(vals) + "|")
    return "\n".join(lines)


def _comparison_line(paired, name, scope, metric):
    row = next(x for x in paired if x["comparison"] == name and x["scope"] == scope and x["metric"] == metric)
    return f"{row['mean_delta_pp']:+.4f} pp ({row['wins']}/{row['ties']}/{row['losses']})"


def make_report(summaries, paired, audit_status, all_metrics, mechanisms, interventions, manifest):
    by = {(row["group"], row["scope"]): row for row in summaries}
    g2base10 = _comparison_line(paired, "G2-Faithful - E74.4-G2", "all_10", "acc")
    g2base9 = _comparison_line(paired, "G2-Faithful - E74.4-G2", "exclude_fold2_descriptive_9", "acc")
    g2g110 = _comparison_line(paired, "G2-Faithful - G1-Anchor", "all_10", "acc")
    g2g39 = _comparison_line(paired, "G2-Faithful - G3-Anonymous", "exclude_fold2_descriptive_9", "acc")
    g2mean = by[("G2-Faithful", "all_10")]["acc_mean_pct"]
    base_mean = by[("E74.4-G2 historical", "all_10")]["acc_mean_pct"]
    g2wins = next(x for x in paired if x["comparison"] == "G2-Faithful - E74.4-G2" and x["scope"] == "all_10" and x["metric"] == "acc")["wins"]
    dcer_rows = [x for x in interventions if x["intervention"] == "dcer_off"]
    flip_count = sum(x["hard_flips"] for x in dcer_rows)
    correct_to_wrong = sum(x["full_correct_intervention_wrong"] for x in dcer_rows)
    wrong_to_correct = sum(x["full_wrong_intervention_correct"] for x in dcer_rows)
    direct_rows = [x for x in interventions if x["intervention"] == "confidence_neutral"]
    confidence_flips = sum(x["hard_flips"] for x in direct_rows)
    g2m = [x for x in mechanisms if x["group"] == "G2-Faithful"]
    gamma = statistics.fmean(x["gamma_best"] for x in g2m)
    residual = statistics.fmean(x["residual_cam_l2_ratio"] for x in g2m) * 100
    direction = statistics.fmean(x["train_direction_correlation"] for x in g2m)
    test_direction = statistics.fmean(x["test_direction_correlation_unlabeled"] for x in g2m)
    confidence = statistics.fmean(x["confidence_teacher_correlation_train"] for x in g2m)
    confidence_test = statistics.fmean(x["confidence_teacher_correlation_test_unlabeled"] for x in g2m)
    anonym_mean = by[("G3-Anonymous", "all_10")]["acc_mean_pct"]
    g2_other_metrics_ok = all(
        next(x for x in paired if x["comparison"] == "G2-Faithful - E74.4-G2" and x["scope"] == "all_10" and x["metric"] == metric)["mean_delta_pp"] >= margin
        for metric, margin in (("hard_auc", -0.5), ("prob_auc", -0.5), ("sen", -1.0), ("spe", -1.0))
    )
    corrective_flips = wrong_to_correct > correct_to_wrong
    if (g2mean >= base_mean + 0.3 and g2wins >= 6 and g2_other_metrics_ok
            and g2mean > by[("G1-Anchor", "all_10")]["acc_mean_pct"] and g2mean > anonym_mean
            and corrective_flips):
        decision = "GO"
    elif g2mean > base_mean or any(by[(g, "all_10")]["acc_mean_pct"] > base_mean for g in GROUPS) or flip_count > 0:
        decision = "PARTIAL"
    else:
        decision = "NO-GO"
    lines = [
        "# Experiment 75.1 — SA-DCER",
        "",
        "## Execution and audit",
        f"- Status: {manifest.get('status')}; tasks {manifest.get('completed_tasks')}/30; independent checkpoints {audit_status['independently_audited']}/30.",
        f"- Directory: `{EXP}`. Preferred OSSFS path failed with `{manifest['root_fallback_reason']}`.",
        f"- Device: {manifest.get('device')}; worker cap {manifest.get('worker_limit')}; peak concurrent workers {manifest.get('training_execution', {}).get('max_workers_observed')}.",
        f"- Wall time {manifest.get('training_execution', {}).get('wall_seconds', 0):.1f}s; failed tasks {len(manifest.get('training_execution', {}).get('failures', []))}; retries {manifest.get('training_execution', {}).get('retry_count', 0)}; OOM {manifest.get('training_execution', {}).get('oom_events', 0)}.",
        f"- E74.4-G2 source manifest `{manifest['base']['source_sha256']}`; E74.4 source/run/model SHA values are recorded in `manifest.json`.",
        f"- SA-DCER training-source SHA `{manifest['source_sha256']}`; current independent-audit SHA `{manifest.get('current_audit_sha256', 'not recorded')}`. Audit-only repair: `{manifest.get('audit_script_repair')}`.",
        f"- Prior audit attempts: `{manifest.get('audit_attempt_history', [])}`; these are audit-engineering failures, not training retries.",
        f"- Fold-2 overlap: 1-based rows 377 (ASD=0) and 407 (TD=1). This is historical label exposure; `val=test`, pre-CV preprocessing and the 871-node transductive graph remain. The nine-fold rows are descriptive only.",
        f"- Independent audit: 30/30 pass; exact hard predictions and probability error <=1e-8. Same-fold initialization/warmup 10/10. Common-epoch RNG comparisons: `{audit_status['rng_shared_epoch_checks']}`.",
        "",
        "## Method audit",
        "- The MICCAI SPromptGL paper presents graph-based GFR and a PLN composed of CAM, prompt-guided local evidence and relation-aware refinement; repository code has a latent `view(N,4,4,9)` partition and the original pool on each 36-D position. These are learned feature coordinates, not anatomical ROIs. E74.4-G2 runs PLNNoPEN and E75.1 inserts at its CAM 144-D OutputLayer input; this does not restore the original PEN.",
        "- BrainPrompt combines ROI-, subject- and disease-level prompts; its `compute_label_loss` compares normalized fused graph representation with normalized label embeddings by cross-entropy. Its paper uses Llama/LLM2Vec, unlike this experiment's cached CLIP RN50 prompts. SA-DCER borrows semantic supervision at a local feature boundary, without reproducing BrainPrompt's multi-level network.",
        "- SABER Module III conditions semantic retrieval on graph context and injects patient-specific text at decision level. Its official code was not verified. ERASER and Right for the Right Reasons motivate prediction-sensitivity checks; the token-halving teacher here is model-internal sensitivity, not clinical or causal evidence.",
        "- E75 previously collapsed eight prompts into two, had <1% common-high responses, normalized token teachers to sum to one, reported G3 teacher correlation 0.1241 and residual/CAM ratio 0.1585%, no hard flip under direct-off, and anonymous G4 exceeded real-text G3. This experiment retains all eight embeddings, uses signed non-normalized margin teachers, and confidence factor 0.75+0.5r.",
        "",
        "## Classification results",
        "Mean ± population SD (%); Hard-AUC is historical hard prediction/BA, Prob-AUC uses continuous ASD probability.",
        "",
        format_summary_table(summaries),
        "",
        "### Paired ACC differences",
        f"- G2−E74.4-G2: all folds {g2base10}; excluding Fold 2 {g2base9}.",
        f"- G2−G1: all folds {g2g110}.",
        f"- G2−G3 on the descriptive nine folds: {g2g39}.",
        "- Complete metric-by-metric and fold-level deltas, including W/T/L and nine-fold comparisons, are in `paired_comparisons.csv`.",
        "",
        "### Excluding Fold 2",
        "Descriptive 9-fold mean ± population SD (%), without retraining:",
        "",
        "|Group|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("E74.4-G2 historical",) + GROUPS:
        row = by[(group, "exclude_fold2_descriptive_9")]
        vals = [f"{row[m+'_mean_pct']:.4f} ± {row[m+'_population_sd_pct']:.4f}" for m in CORE_METRICS]
        lines.append("|" + group + "|" + "|".join(vals) + "|")
    lines += [
        "",
        "### Paired comparison details",
        "Mean paired change in percentage points, W/T/L; negative values indicate the left-hand group is lower.",
        "",
        "|Comparison|Scope|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    report_pairs = (
        ("G1-Anchor - E74.4-G2", "G1 - base"),
        ("G2-Faithful - E74.4-G2", "G2 - base"),
        ("G3-Anonymous - E74.4-G2", "G3 - base"),
        ("G2-Faithful - G1-Anchor", "G2 - G1"),
        ("G2-Faithful - G3-Anonymous", "G2 - G3"),
    )
    for comparison, label in report_pairs:
        for scope, scope_label in (("all_10", "10 folds"), ("exclude_fold2_descriptive_9", "9 folds, descriptive")):
            cells = []
            for metric in CORE_METRICS:
                row = next(x for x in paired if x["comparison"] == comparison and x["scope"] == scope and x["metric"] == metric)
                cells.append(f"{row['mean_delta_pp']:+.4f} ({row['wins']}/{row['ties']}/{row['losses']})")
            lines.append("|" + "|".join((label, scope_label, *cells)) + "|")
    lines += [
        "",
        "## Mechanism findings",
        f"- G2 mean gamma {gamma:.5f}; mean residual/CAM norm {residual:.4f}%; mean training signed-direction correlation {direction:.4f}, unlabeled test-node correlation {test_direction:.4f}; confidence-vs-|teacher| correlation train {confidence:.4f}, unlabeled test {confidence_test:.4f}.",
        f"- Turning off SA-DCER caused {flip_count} hard flips across 10 G2 checkpoints (full correct→off wrong {correct_to_wrong}; full wrong→off correct {wrong_to_correct}). Confidence-neutral caused {confidence_flips} flips in G2/G3; this measures fixed-checkpoint sensitivity, not a retrained ablation.",
        "- G1/G2/G3 signed evidence, prompt weights, token attention concentration, per-modality residual norms, gamma samples, teacher agreement, and confidence distributions are recorded per fold in `mechanism.csv`.",
        "- Real text vs anonymous text comparison is a single geometry-preserving transform per fold. Similar performance or one transform's advantage cannot establish general medical-language value.",
        "- Latent 9-D token blocks have no anatomical ROI mapping. PHENO includes site/demographic features; ANAT/FUNC quality variables and CORR are not lesion labels.",
        "",
        "## Decision",
        f"- Scientific decision: **{decision}** (predeclared thresholds; no post-hoc tuning). G2 ACC change vs base is {g2mean-base_mean:+.4f} pp, G2 wins {g2wins}/10, real/anonymous ACC means are {g2mean:.4f}% / {anonym_mean:.4f}%.",
        "- This decision does not support an unbiased independent-generalization claim because of the inherited Fold-2 overlap, `val=test` selection, preprocessing before CV and transductive graph.",
        f"- Report: `{RESULTS / 'report.md'}`; manifest: `{EXP / 'manifest.json'}`; audit: `{RESULTS / 'audit_status.json'}`.",
    ]
    return "\n".join(lines) + "\n", decision


def append_handoff(summary):
    path = HANDOFF
    original = path.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    if not original.endswith(b"\n"):
        summary = "\n" + summary
    data = (summary + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    after = path.read_bytes()
    if not after.startswith(original) or after[len(original):] != data:
        raise IOError("handoff append verification failed")
    return {"path": str(path), "original_sha256": original_sha,
            "final_sha256": hashlib.sha256(after).hexdigest(), "appended_bytes": len(data), "readback_verified": True}


def run_audit():
    import traceback
    r = runner()
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = manifest.get("source_sha256")
    if manifest.get("completed_tasks") != 30 or not expected_source:
        raise RuntimeError("independent audit requires all 30 tasks and a recorded training source SHA")
    preflight_inventory = manifest.get("preflight", {}).get("protected_source_inventory", {})
    current_inventory = r.source_inventory()
    if set(preflight_inventory) != set(current_inventory):
        raise RuntimeError("source inventory paths changed after training")
    changed_sources = [key for key in current_inventory if current_inventory[key] != preflight_inventory[key]]
    if any(key != "audit.py" for key in changed_sources):
        raise RuntimeError(f"training source changed outside the audit script: {changed_sources}")
    reconstructed_training_inventory = dict(current_inventory)
    reconstructed_training_inventory["audit.py"] = preflight_inventory["audit.py"]
    if r.sha_json(reconstructed_training_inventory) != expected_source:
        raise RuntimeError("reconstructed checkpoint training source SHA does not match the recorded source")
    audit_attempt_history = list(manifest.get("audit_attempt_history", []))
    audit_source_repair = None
    if changed_sources:
        audit_source_repair = {"file": "audit.py", "preflight_sha256": preflight_inventory["audit.py"],
                               "current_sha256": current_inventory["audit.py"],
                               "reason": "Corrected fixed-checkpoint CAM feature-change reference from raw H to full H-prime, and converted test indices to a Torch tensor after the first audit attempt; training code, weights and predictions are unchanged."}
    audits, metrics, mechanisms, interventions, tasks, checkpoints = [], [], [], [], [], []
    for fold in range(1, 11):
        for group in GROUPS:
            try:
                audit, metric, mechanism, changed, task, checkpoint = audit_one(group, fold, expected_source)
                audits.append(audit); metrics.append(metric); mechanisms.append(mechanism); interventions.extend(changed)
                tasks.append((group, fold, task)); checkpoints.append((group, fold, checkpoint))
            except Exception as exc:
                audits.append({"group": group, "fold": fold, "status": "FAIL", "error": repr(exc),
                               "traceback": traceback.format_exc()[-5000:]})
    r.write_csv(RESULTS / "checkpoint_audit.csv", audits)
    if len(audits) != 30 or any(row["status"] != "PASS" for row in audits):
        failure_status = {"status": "FAIL", "expected": 30,
                          "passed": sum(x["status"] == "PASS" for x in audits),
                          "failed": sum(x["status"] == "FAIL" for x in audits), "audits": audits}
        r.atomic_json(RESULTS / "audit_status.json", failure_status)
        audit_attempt_history.append({"attempt": len(audit_attempt_history) + 1, **{k: failure_status[k] for k in ("status", "expected", "passed", "failed")},
                                      "failure_csv_sha256": r.sha_file(RESULTS / "checkpoint_audit.csv"),
                                      "first_error": next((row.get("error") for row in audits if row["status"] == "FAIL"), None)})
        manifest["status"] = "AUDIT_FAILED"; manifest["independent_audits"] = sum(x["status"] == "PASS" for x in audits)
        manifest["audit_attempt_history"] = audit_attempt_history
        r.atomic_json(manifest_path, manifest)
        raise RuntimeError(f"independent checkpoint audit failed: {manifest['independent_audits']}/30")
    summaries, paired, audit_status, all_metrics = build_results(audits, metrics, mechanisms, interventions, tasks, checkpoints)
    final_inventory = r.source_inventory()
    if any(final_inventory[key] != manifest["preflight"]["protected_source_inventory"][key]
           for key in final_inventory if key != "audit.py"):
        raise RuntimeError("protected training source inventory changed between preflight and final audit")
    manifest["independent_audits"] = 30; manifest["checkpoint_audit"] = "PASS 30/30"
    manifest["status"] = "COMPLETE_AUDITED"; manifest["final_source_inventory_sha256"] = r.sha_json(final_inventory)
    manifest["audit_script_repair"] = audit_source_repair
    manifest["current_audit_sha256"] = final_inventory["audit.py"]
    manifest["audit_attempt_history"] = audit_attempt_history
    report, decision = make_report(summaries, paired, audit_status, all_metrics, mechanisms, interventions, manifest)
    manifest["scientific_decision"] = decision
    manifest["scientific_decision_note"] = "Decision follows predeclared criteria and fold-level results; historical protocol limitations remain."
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    handoff_verified = False
    try:
        handoff = append_handoff(report)
        manifest["handoff_update"] = handoff
        handoff_verified = bool(handoff.get("readback_verified"))
    except Exception as exc:
        pending = EXP / "results/handoff_pending.md"
        pending.write_text(report, encoding="utf-8")
        manifest["handoff_update"] = {"readback_verified": False, "error": repr(exc), "pending_summary": str(pending)}
        manifest["status"] = "COMPLETE_WITH_LIMITATIONS"
    r.atomic_json(manifest_path, manifest)
    reread = (RESULTS / "report.md").read_text(encoding="utf-8")
    if reread != report:
        raise IOError("report persistence verification failed")
    if not handoff_verified:
        print("HANDOFF_PENDING: report preserved in results/handoff_pending.md", flush=True)
    return manifest


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({"status": result["status"], "audited": result["independent_audits"],
                      "scientific_decision": result["scientific_decision"]}, indent=2))
