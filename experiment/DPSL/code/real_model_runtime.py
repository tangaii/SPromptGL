"""Run the frozen SCMR-DA-SRPE graph, using the registered fold adapters."""
from __future__ import annotations

import csv
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from dpsl_runtime import (EXPECTED_METRICS_PERCENT, FOLDS, OVERLAP_1BASED, ROOT, _array, checkpoint_path,
                          checkpoint_source, formal_metric_rows, formal_patient_rows, load_checkpoint,
                          read_json, sha256_file, validate_evaluation, validate_membership, verify_sha,
                          verify_warmup_file, write_json)


ADAPTERS = {
    2: ("history_fold2_3/cumulative.py", "G1"),
    3: ("history_fold2_3/darg_only.py", "G2"),
    4: ("history_fold4/core.py", "G1"),
    8: ("history_fold8/core.py", "G1"),
}
METRICS = tuple(EXPECTED_METRICS_PERCENT)
COUNTS = ("n_test", "correct", "tp", "fp", "tn", "fn")
# The original archive compares metric formulas with this absolute bound.
METRIC_ATOL = 1e-12
RESIDUALS = ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


@lru_cache(maxsize=None)
def local_module(relative):
    path = ROOT / "code" / relative
    name = "_dpsl_" + relative.removesuffix(".py").replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fold_engine(fold):
    verify_warmup_file(fold)
    if fold not in ADAPTERS:
        return local_module("history_base/run.py")
    relative, task_group = ADAPTERS[fold]
    adapter = local_module(relative)
    if fold == 3:
        return adapter.engine(fold)
    if fold == 8:
        return adapter.engine(task_group)
    return adapter.engine(task_group, fold)


def history_record(fold, kind):
    return read_json(ROOT / "runtime" / "history" / f"fold{fold:02d}.{kind}.json")


def setup(fold, engine):
    return engine.setup("G2", fold)


def optimizer_identity(ctx, engine):
    """Validate the loaded AdamW moments against the actual ordered parameters."""
    named = {p: f"{prefix}.{name}" for prefix, module in
             (("model", ctx["state"]["model"]), ("cam", ctx["state"]["cam"]), ("DPSL", ctx["module"]))
             for name, p in module.named_parameters()}
    optimizers = {name: ctx["state"][name] for name in engine.h.OPTS} | {"DPSL": ctx["optimizer"]}
    evidence = {}
    for name, optimizer in optimizers.items():
        groups = [[named[p] for p in group["params"]] for group in optimizer.param_groups]
        if name in engine.h.OPTS:
            require(groups == ctx["provenance"]["optimizers"][name]["groups"],
                    f"{name} optimizer parameter group order differs")
        else:
            require(groups == [[name for name in named.values() if name.startswith("DPSL.")]],
                    "DPSL optimizer parameter order differs")
        steps = []
        require(optimizer.state, f"{name} optimizer empty after restore")
        for p, moments in optimizer.state.items():
            require(moments["exp_avg"].shape == moments["exp_avg_sq"].shape == p.shape
                    and moments["exp_avg"].dtype == moments["exp_avg_sq"].dtype == p.dtype,
                    f"{name} optimizer moments do not fit {named[p]}")
            steps.append(int(moments["step"].item()))
        evidence[name] = {"parameter_group_names_sha": engine.h.sha(groups), "moment_entries": len(steps),
                          "min_step": min(steps), "max_step": max(steps),
                          "state_sha": engine.h.sha(optimizer.state_dict())}
    return evidence


def restore(fold, kind):
    saved = load_checkpoint(fold, kind)
    engine = fold_engine(fold)
    ctx = setup(fold, engine)
    engine.restore(ctx, saved)
    require(engine.h.sha(engine.bundle(ctx)) == saved["state_sha"], f"Fold {fold} live model or optimizer differs")
    require(engine.h.sha(engine.ema(ctx)) == saved["ema_sha"] and engine.h.rng_sha() == saved["rng_sha"],
            f"Fold {fold} Gaussian/EMA or RNG changed during restore")
    require(saved["config"] == engine.CFG, f"Fold {fold} live training config differs")
    return ctx, engine, saved, optimizer_identity(ctx, engine)


def check_order(test, rows):
    require(test.tolist() == [int(row["patient_index_0based"]) for row in rows], "Historical test order differs")


def check_residual(delta, cam, enhanced, masked):
    require(not np.count_nonzero(delta[masked]) and np.array_equal(cam[masked], enhanced[masked]),
            "Overlap residual or CAM144 changed")


def check_scope(fold, ctx, capture):
    m, state = ctx["members"], ctx["state"]
    validate_membership(m, fold)
    test, legal, pairs, same = (m[name] for name in ("test", "legal", "train_pairs", "train_same"))
    y = state["labels_np"]
    require(np.array_equal(same, y[pairs[:, 0]] == y[pairs[:, 1]]), "Historical pair labels differ")
    module, d = ctx["module"], capture["details"]
    overlap = sorted(patient - 1 for patient in OVERLAP_1BASED.get(fold, ()))
    allowed = _array(module.spectrum.allowed)
    require(np.flatnonzero(~allowed).tolist() == overlap and
            np.array_equal(legal, _array(module.statistics.legal)) and
            np.array_equal(legal, ctx["reference"]["legal"]) and np.array_equal(y[legal], _array(ctx["labels"])),
            "Live spectral/Gaussian/EMA supervision differs")
    cam, enhanced = _array(capture["cam"]), _array(capture["enhanced"])
    for key in RESIDUALS:
        check_residual(_array(d[key]), cam, enhanced, overlap)
    delta, attention = _array(d["delta"]), _array(d["attention"])
    require(cam.shape == delta.shape == (871, 144) and attention.shape == (871, 8, 17)
            and np.isfinite(attention).all() and np.isfinite(delta).all()
            and (attention[legal].max(axis=(1, 2)) > 0).all() and (delta[legal] != 0).any(axis=1).all(),
            "Spectral or medical Prompt path inactive on legal patients")
    require(np.all(_array(d["ratio"]) <= 0.08 + 2 * np.finfo(delta.dtype).eps), "CAM144 residual above 8%")
    return {
        "test": len(test), "legal": len(legal), "train": len(m["train"]), "overlap_1based": [i + 1 for i in overlap],
        "overlap_all_delta_components_exactly_zero": True, "active_legal": len(legal),
        "overlap_pair_occurrences": {str(i + 1): int(np.count_nonzero(pairs == i)) for i in overlap},
        "max_residual_ratio": float(_array(d["ratio"]).max()),
        "membership_sha": ctx["provenance"]["membership"], "basis_sha": ctx["provenance"]["basis"],
        "prompt_sha": ctx["provenance"]["prompt"], "gaussian_reference_sha": ctx["provenance"]["reference"],
    }


def score(labels, predicted, probability):
    tp = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 1) & (predicted == 0)))
    tn = int(np.sum((labels == 1) & (predicted == 1)))
    fn = int(np.sum((labels == 0) & (predicted == 1)))
    return {
        "n_test": len(labels), "correct": tp + tn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": float(np.mean(labels == predicted)),
        "hard_auc": float(balanced_accuracy_score(labels, predicted)),
        "prob_auc": float(roc_auc_score(labels == 0, probability[:, 0].astype(np.float64))),
        "sen": float(recall_score(labels, predicted, pos_label=0)),
        "spe": float(recall_score(labels, predicted, pos_label=1)),
        "asd_f1": float(f1_score(labels, predicted, pos_label=0, zero_division=0)),
    }


def compare_metrics(actual, expected):
    differences = {key: abs(actual[key] - float(expected[key])) for key in METRICS}
    require(max(differences.values()) <= METRIC_ATOL and
            all(actual[key] == int(expected[key]) for key in COUNTS if expected.get(key) not in ("", None)),
            f"Metric or confusion matrix differs: {differences}")
    return differences


def norm(a):
    return np.linalg.norm(_array(a).astype(np.float64, copy=False), axis=-1)


def compare_arrays(first, second, label):
    first, second = np.asarray(first), np.asarray(second)
    require(first.shape == second.shape, f"{label} has different shape")
    error = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
    require(first.dtype == second.dtype and first.tobytes() == second.tobytes(),
            f"{label} differs, maximum absolute error {error}")
    return error


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate(folds=FOLDS, out=None, run_dir=None):
    proof = local_module("history_base/audit.py")
    archived = formal_patient_rows() if run_dir is None else []
    metrics_archive = {row["fold"] or row["scope"]: row for row in formal_metric_rows()
                       if row["method"] == "E82.4.6.10-G1" and row["intervention"] == "full"}
    rows, metrics, checks = [], [], []
    for fold in folds:
        if run_dir is None:
            ctx, engine, saved, _ = restore(fold, "best")
        else:
            engine = fold_engine(fold)
            ctx = setup(fold, engine)
            saved = engine.h.load(run_dir / "checkpoints" / f"fold{fold:02d}.best.pt")
            engine.restore(ctx, saved)
        proof.run = engine
        out_eval, capture = engine.evaluate(ctx)
        test = ctx["members"]["test"]
        labels = ctx["state"]["labels_np"][test]
        validate_evaluation(out_eval, len(test), labels)
        scope = check_scope(fold, ctx, capture)
        y_hat, logits, p = (out_eval[name] for name in ("prediction", "logits", "probability"))
        summary = score(labels, y_hat, p)
        errors, digests = {}, {}
        predicted, seen = proof.independent_eval(ctx)
        for key in ("logits", "probability", "prediction"):
            compare_arrays(out_eval[key], predicted[key], f"Fold {fold} historical evaluator {key}")
            errors[key] = compare_arrays(out_eval[key], saved["evaluation"][key], f"Fold {fold} archived {key}")
        for key in ("cam", "enhanced", *RESIDUALS):
            a = capture["details"][key] if key in RESIDUALS else capture[key]
            b = seen["details"][key] if key in RESIDUALS else seen[key]
            errors[key] = compare_arrays(_array(a), _array(b), f"Fold {fold} historical evaluator {key}")
            digests[key] = engine.h.sha(a.cpu())
        margin = proof.signed_margin(logits, labels)
        details = capture["details"]
        _, anonymous = proof.independent_eval(ctx, "anonymous_query")
        cam, delta = norm(capture["cam"][test]), norm(details["delta"][test])
        patient_values = {
            "p_asd_full": p[:, 0].astype(np.float64), "prediction_full": y_hat,
            "true_margin_full": margin, "cam_l2": cam, "delta_l2": delta,
            "delta_parallel_l2": norm(details["delta_parallel"][test]), "delta_perp_l2": norm(details["delta_perp"][test]),
            "cam_relative_delta": delta / cam,
            "query_feature_change_l2": norm(capture["enhanced"][test].double() - anonymous["enhanced"][test].double()),
        }
        if run_dir is None:
            reference = [row for row in archived if int(row["fold"]) == fold]
            check_order(test, reference)
            require(labels.tolist() == [int(row["true_label"]) for row in reference], "Patient labels differ")
            require(all(row["best_sha256"] == checkpoint_source(fold, "best")[1] for row in reference),
                    "Patient archive Best SHA differs")
            for key, value in patient_values.items():
                errors[key] = compare_arrays(value, np.array([row[key] for row in reference], dtype=value.dtype),
                                             f"Fold {fold} patient archive {key}")
            require(out_eval["correct"] == saved["evaluation"]["correct"], "Checkpoint correct count differs")
            errors["metrics"] = compare_metrics(summary, metrics_archive[str(fold)])
        for i, patient in enumerate(test):
            rows.append({"fold": fold, "test_position": i, "patient_index_0based": int(patient),
                         "patient_id_1based": int(patient) + 1, "true_label": int(labels[i]), "prediction": int(y_hat[i]),
                         "logit_asd": float(logits[i, 0]), "logit_td": float(logits[i, 1]),
                         "p_asd": float(p[i, 0]), "p_td": float(p[i, 1]),
                         **{key: float(value[i]) for key, value in patient_values.items() if key not in
                            ("p_asd_full", "prediction_full")}})
        metrics.append({"scope": "test_fold", "fold": fold, **summary})
        checks.append({"fold": fold, "status": "PASS", "best_epoch": saved["epoch"], "test_rows": len(test),
                       "correct": summary["correct"], "scope": scope, "maximum_absolute_error": errors,
                       "full_tensor_sha": digests, "result_sha": engine.h.sha(out_eval),
                       "configuration_sha": engine.h.sha(saved["config"]),
                       "best_rng_sha": saved["rng_sha"], "best_ema_sha": saved["ema_sha"]})
        print(f"Best fold {fold:02d} real forward {summary['correct']}/{len(test)} PASS", file=sys.stderr, flush=True)
    if tuple(folds) == FOLDS:
        require(sorted(row["patient_index_0based"] for row in rows) == list(range(871)), "871 unique test IDs required")
    pooled = score(np.array([row["true_label"] for row in rows]), np.array([row["prediction"] for row in rows]),
                   np.array([[row["p_asd"], row["p_td"]] for row in rows]))
    means = {key: float(np.mean([item[key] for item in metrics])) for key in METRICS}
    std = {key: float(np.std([item[key] for item in metrics], ddof=0)) for key in METRICS}
    if tuple(folds) == FOLDS and run_dir is None:
        for key, value in (("macro_mean", means), ("macro_population_sd", std), ("pooled", pooled)):
            compare_metrics(value, metrics_archive[key])
        for key, expected in EXPECTED_METRICS_PERCENT.items():
            require((round(100 * means[key], 4), round(100 * std[key], 4)) == expected, f"Formal {key} differs")
    if out is not None:
        write_csv(out / "evaluation_real.csv", rows)
        write_csv(out / "fold_metrics_real.csv", metrics + [{"scope": "macro_mean", **means},
                  {"scope": "macro_population_sd", **std}, {"scope": "pooled", **pooled}])
    return {"status": "PASS", "mode": "real model forward and historical independent evaluator",
            "folds": checks, "patients": len(rows), "macro_mean": means, "macro_population_sd": std, "pooled": pooled}


def check_budget(engine, diagnostics):
    cap = engine.CFG["full_classification"]["shared_cap"]
    require(cap == 0.001, "Historical shared gradient cap differs")
    ratios = {}
    for name in ("gfr", "darg", "cam", "head"):
        budget = diagnostics["budget"][name]
        used, main = budget["total_applied"], budget["historical"]
        require(0 <= used <= cap * main and abs(budget["applied_dot"]) <= budget["dot_bound"],
                f"Shared gradient budget exceeded by {name}")
        ratios[name] = used / main if main else 0.0
    return ratios


def recovery(folds=FOLDS, out=None):
    checks = []
    for fold in folds:
        ctx, engine, saved, before_opts = restore(fold, "latest")
        task = history_record(fold, "task")
        preflight = history_record(fold, "preflight")
        require((saved["epoch"], saved["best_epoch"], saved["best_correct"], saved["wait"]) ==
                tuple(task[key] for key in ("epoch", "best_epoch", "best_correct", "wait")),
                "Latest counters differ from the archived task")
        before_rng, before_ema = engine.h.rng_sha(), engine.h.sha(engine.ema(ctx))
        active_ema = engine.ema(ctx)
        engine.restore_ema(ctx, saved["eval_ema"])
        latest_evaluation, _ = engine.evaluate(ctx)
        require(engine.h.sha(latest_evaluation) == engine.h.sha(saved["evaluation"]),
                "Latest evaluation under its historical pre-update EMA differs")
        engine.restore_ema(ctx, active_ema)
        result, training, rng, previous, observation = engine.transition(
            ctx, saved["epoch"] + 1, preflight["dropout"], inspect=True)
        after_opts = optimizer_identity(ctx, engine)
        witness = {"epoch": rng["epoch"], "rng": rng, "training_sha": engine.h.sha(training),
                   "evaluation_sha": engine.h.sha(result), "state_sha": engine.h.sha(engine.bundle(ctx)),
                   "previous_ema": engine.h.sha(previous), "observation": engine.h.sha(observation),
                   "ema": engine.h.sha(engine.ema(ctx))}
        require(witness == task["next_witness"], f"Fold {fold} next witness differs in "
                f"{[key for key in witness if witness[key] != task['next_witness'][key]]}")
        require(before_rng == rng["rng_before"] == saved["rng_sha"] and before_ema == witness["previous_ema"]
                and rng["rng_after_train"] == rng["rng_after_eval"] and witness["state_sha"] != saved["state_sha"],
                "RNG, EMA or state transition differs")
        require(all(before_opts[k]["state_sha"] != after_opts[k]["state_sha"] for k in before_opts),
                "Optimizer failed to update")
        journal = ROOT / "runtime" / "history" / f"fold{fold:02d}.journal.jsonl"
        require(journal.stat().st_size == saved["journal_size"] and
                verify_sha(journal, task["journal_sha"]) == saved["journal_sha"], "Latest journal changed")
        for kind in ("best", "latest"):
            verify_sha(checkpoint_path(fold, kind), checkpoint_source(fold, kind)[1])
        checks.append({"fold": fold, "status": "PASS", "latest_epoch": saved["epoch"],
                       "best_epoch": saved["best_epoch"], "early_stop_wait": saved["wait"],
                       "next_historical_witness_exact": True, "next_correct": result["correct"],
                       "next_witness": witness, "dropout": training["dropout"],
                       "shared_gradient_fraction": check_budget(engine, training),
                       "optimizers_before": before_opts, "optimizers_after": after_opts,
                       "latest_state_sha_before": saved["state_sha"], "formal_checkpoint_bytes_unchanged": True})
        print(f"Latest fold {fold:02d} epoch {saved['epoch'] + 1} historical witness PASS", file=sys.stderr, flush=True)
    result = {"status": "PASS", "mode": "in-memory real next train and evaluation vs historic post-Latest witness",
              "folds": checks}
    if out is not None:
        write_json(out / "resume_check.json", result)
    return result
