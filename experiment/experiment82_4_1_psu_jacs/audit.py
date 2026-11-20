"""Independent best/latest, RNG, mechanism, and paired patient-level audit."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

from common import history as h
import numpy as np
import torch
from scipy.special import ndtr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F

import run

CFG, HERE, RESULTS = h.CFG, h.HERE, run.RESULTS
HISTORICAL = ("DARG", "E82.2-G2", "E82.3.2-G2")
METHODS = tuple(run.METHODS)
METRIC_NAMES = ("ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
SCOPES = ("10_folds", "exclude_fold2")


GEOMETRY_FIELDS = ("rho", "cosine_asd", "linear_asd_margin", "actual_asd_margin")
HANDOFF_MARKER = "## Experiment 82.4.1 · PSU-PSCE / JACS-DA-SRPE"


def eq(a, b, message):
    h.require(np.array_equal(a, b), message)



def close(a, b, message, *, atol=1e-9, rtol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=message)



def array(tensor):
    return tensor.detach().cpu().numpy()



def metric_values(probability, predictions, labels):
    asd = labels == CFG["labels"]["ASD"]
    h.require(probability.shape == (len(labels), 2) and np.isfinite(probability).all(), "Malformed probability")
    eq(predictions, probability.argmax(-1), "Predictions differ from archived probability argmax")
    close(probability.sum(-1), 1.0, "Probabilities do not sum to one", atol=2e-6)
    ba = balanced_accuracy_score(labels, predictions)
    close(ba, roc_auc_score(asd, predictions == 0), "Binary Hard-AUC differs from BA")
    return dict(zip(h.METRICS, (
        float(accuracy_score(labels, predictions)), float(ba), float(roc_auc_score(asd, probability[:, 0])),
        float(recall_score(labels, predictions, pos_label=0)), float(recall_score(labels, predictions, pos_label=1)),
        float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
    )))



def load_historical(fold, labels, ctx, guard):
    files = {
        "DARG": (h.runner().best_path(h.BASE, fold), h.E744 / "results/fold_metrics.csv", h.BASE, 1.0),
        "E82.2-G2": (h.E822 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                    h.E822 / "results/fold_metrics.csv", "G2", 100.0),
        "E82.3.2-G2": (h.E8232 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                      h.E8232 / "results/fold_metrics.csv", "G2", 100.0),
    }
    result = {}
    for name, (ckpt, csv_path, group, scale) in files.items():
        h.require(h.sha_file(ckpt) == guard[str(ckpt)], f"Historical {name} Best changed")
        saved = h.load_torch(ckpt)
        h.require(saved["identity"] == ctx["identity"]["identity"] and saved["fold"] == fold,
                  f"Historical {name} does not match this fold")
        if "test_indices" in saved:
            eq(saved["test_indices"], ctx["state"]["test_np"], f"{name} archived test order differs")
        else:
            archived = h.historical_task(fold)["identity"]
            h.require(saved["identity"]["split_sha"] == archived["split_sha"], f"{name} historical split SHA differs")
        if name == "DARG":
            prob, pred = saved["test_probability"], saved["test_prediction"]
            best_epoch = saved["best_epoch_1based"]
        else:
            prob, pred = saved["probability"], saved["prediction"]
            best_epoch = saved["best_epoch"]
        measured = metric_values(prob, pred, labels)
        with csv_path.open(newline="") as stream:
            historic, = [row for row in csv.DictReader(stream) if row["group"] == group and int(row["fold"]) == fold]
        for key, value in measured.items():
            close(value, float(historic[key]) / scale, f"{name} audited historical {key}", atol=1e-12, rtol=0)
        result[name] = {
            "probability": prob, "prediction": pred, "metrics": measured, "best_epoch": int(best_epoch),
            "margin": (1 - 2 * labels) * (np.log(np.maximum(prob[:, 0], 1e-38)) - np.log(np.maximum(prob[:, 1], 1e-38))),
        }
    return result



def delta_analysis(patients, source, target):
    truth = np.asarray([row["label"] for row in patients])
    previous = np.asarray([row[source + "_prediction"] for row in patients])
    current = np.asarray([row[target + "_prediction"] for row in patients])
    fixed, harmed = (previous != truth) & (current == truth), (previous == truth) & (current != truth)
    delta = np.asarray([row[target + "_p_asd"] - row[source + "_p_asd"] for row in patients])
    margin = np.asarray([row[target + "_true_margin"] - row[source + "_true_margin"] for row in patients])
    changed = previous != current
    result = {
        "flips": int(changed.sum()), "corrected": int(fixed.sum()), "harmed": int(harmed.sum()),
        "net": int(fixed.sum() - harmed.sum()),
        "p_asd_abs_mean_pp": 100 * float(np.abs(delta).mean()), "p_asd_abs_max_pp": 100 * float(np.abs(delta).max()),
        "p_asd_measurably_changed": int((np.abs(delta) > CFG["comparison"]["measurable_probability_delta"]).sum()),
        "true_margin_delta_mean": float(margin.mean()), "true_margin_delta_abs_mean": float(np.abs(margin).mean()),
        "margin_improved": int((margin > 0).sum()), "margin_declined": int((margin < 0).sum()),
        "genuine_new_corrections_vs_darg": sum(bool(c and row["DARG_prediction"] != row["label"]) for c, row in zip(fixed, patients)),
    }
    for label, prefix in enumerate(("asd", "td")):
        selected = truth == label
        result[f"{prefix}_corrected"] = int((fixed & selected).sum())
        result[f"{prefix}_harmed"] = int((harmed & selected).sum())
        result[f"{prefix}_net"] = result[f"{prefix}_corrected"] - result[f"{prefix}_harmed"]
        result[f"{prefix}_true_margin_delta_mean"] = float(margin[selected].mean())
    return result



def append_handoff(body):
    path = h.HISTORY / "handoff.md"
    marker = HANDOFF_MARKER.encode()
    block = body.encode()
    h.require(block.count(marker) == 1, "Handoff block identity differs")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        old = stream.read()
        target = old.find(marker)
        if target < 0:
            h.require(not old or old.endswith(b"\n"), "Handoff ending needs verification")
            chunk = b"\n" + block
            offset = len(old)
            stream.write(chunk)
        else:
            offset = target
            chunk = block
            h.require(old[offset:offset + len(chunk)] == chunk and old.count(marker) == 1, "Handoff already contains a different E82.4.1 block")
        stream.flush()
        os.fsync(stream.fileno())
    observed = path.read_bytes()
    h.require(observed[:offset] == old[:offset] and observed[offset:offset + len(chunk)] == chunk
              and observed.count(marker) == 1, "Independent handoff readback differs")
    return {"status": "PASS", "path": str(path), "offset": offset,
            "append_sha": hashlib.sha256(chunk).hexdigest(), "historical_prefix_sha": hashlib.sha256(old[:offset]).hexdigest()}



def float32_dot(actual, left, right, name, *, preceding_rounding=0):
    """Compare a float32 dot product to FP64 with an elementwise roundoff bound."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    operations = 2 * left.shape[-1] + preceding_rounding
    unit = np.finfo(np.float32).eps / 2
    gamma = operations * unit / (1 - operations * unit)
    error = np.abs(actual - left @ right)
    bound = gamma * (np.abs(left) @ np.abs(right)) + 2 * unit
    ratio = float(np.max(error / bound))
    h.require(ratio <= 1, f"{name} exceeds float32 dot-product error bound ({ratio:.3f}×)")
    return float(np.max(error)), ratio



def audit_journal(ctx, task, best, latest, preflight, recovery):
    name, fold = ctx["method"]["group"], ctx["state"]["fold"]
    p = run.paths(name, fold)
    raw = p["journal"].read_bytes()
    lines = raw.splitlines(keepends=True)
    h.require(len(raw) == latest["journal_size"] and hashlib.sha256(raw).hexdigest() == latest["journal_sha"],
              "Committed training journal differs")
    records = [json.loads(line) for line in lines]
    end = latest["next_epoch"]
    eq([row["epoch"] for row in records], list(range(run.START + 1, end + 1)), "Training epoch trajectory differs")
    settings = CFG["auxiliary"]
    summed_cap = sum(spec["gradient_cap"] for spec in settings.values())
    maxima = {}
    if recovery is not None:
        preceding = recovery["epoch"] - run.START
        prefix = b"".join(lines[:preceding])
        h.require(hashlib.sha256(prefix).hexdigest() == recovery["journal_sha"], "Recovered journal prefix differs")
        h.require(records[preceding - 1]["rng"]["rng_after_eval"] == recovery["rng_sha"]
                  and records[preceding]["rng"]["rng_before"] == recovery["rng_sha"],
                  "Training RNG discontinuity at automatic recovery")

    arch = {row["epoch"]: row for row in h.historical_task(fold)["rng_trace"]}
    calls = preflight["dropout_calls"]
    inputs = [
        torch.empty_strided(call["shape"], call["stride"], device=ctx["state"]["x"].device,
                            dtype=getattr(torch, call["dtype"].split(".")[-1])).fill_(1)
        for call in calls
    ]
    h.require(len(inputs) == 6 and all(t.dtype == torch.float32 for t in inputs), "Incomplete native dropout inventory")
    original_rng = h.base().pack_rng()
    count, selected, wait, archived_epochs = -1, 0, 0, 0
    ema = preflight["ema_before"]
    try:
        h.base().restore_rng(ctx["warmup"]["rng"])
        for record in records:
            epoch, rng = record["epoch"], record["rng"]
            h.require(h.rng_sha() == rng["rng_before"], f"RNG pre-phase replay failed at {epoch}")
            for source, call in zip(inputs, calls):
                F.dropout(source, p=call["p"], training=True, inplace=call["inplace"])
            h.require(h.rng_sha() == rng["rng_after_train"] == rng["rng_after_eval"], f"RNG schedule replay failed at {epoch}")
            if epoch in arch:
                h.require(rng == arch[epoch], f"Historical RNG differs at {epoch}")
                archived_epochs += 1
            if record["correct"] > count:
                count, selected, wait = record["correct"], epoch, 0
            else:
                wait += 1
            h.require((count, selected, wait) == (record["best_correct"], record["best_epoch"], record["wait"]),
                      "Best selection or early stop counter differs")
            h.require(wait <= CFG["early_stop_wait_gt"] or epoch == end, "Training exceeded early stopping")
            h.require(record["ema_before"] == ema, "EMA journal continuity differs")
            ema = record["ema_after"]
            if name == "G2":
                h.require(ema["updates"] == epoch - run.START, "EMA advance count differs")
            else:
                h.require(ema is None, "G1 unexpectedly has conditional statistics")
            diag = record["diagnostics"]
            h.require(all(math.isfinite(value) for value in diag.values()), "Nonfinite training journal diagnostics")
            h.require(diag["delta_relative_max"] <= CFG["residual"]["norm_limit"] + 2e-6, "Applied residual exceeded bound")
            for family in ("prompt", "residual"):
                watched = {f"{family}_applied_ratio": summed_cap}
                watched.update({f"{family}_{term}_applied_ratio": spec["gradient_cap"] for term, spec in settings.items()})
                for key, cap in watched.items():
                    h.require(diag[key] <= cap + 2e-6, "Logged gradient contribution exceeded budget")
                    maxima[key] = max(maxima.get(key, 0.0), diag[key])
                for term in settings:
                    key = f"{family}_{term}_raw_ratio"
                    maxima[key] = max(maxima.get(key, 0.0), diag[key])
        h.require(h.rng_sha() == latest["rng_sha"], "Terminal RNG does not match replay")
    finally:
        h.base().restore_rng(original_rng)
    h.require(selected == best["best_epoch"] == latest["best_epoch"] == task["best_epoch"]
              and count == best["evaluation"]["correct"] == latest["best_correct"], "Best selection differs")
    h.require(end == task["epoch"] and (end == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"])
              and wait == latest["wait"], "Termination differs from historical protocol")
    h.require(best["rng_sha"] == records[selected - run.START - 1]["rng"]["rng_after_eval"]
              and records[0]["diagnostics"] == preflight["epoch11_diagnostics"], "Best RNG or initial real step differs")
    h.require(ema == latest["ema"] and best["ema"] == records[selected - run.START - 1]["ema_after"]
              and records[0]["ema_after"] == preflight["ema_after"], "Checkpoint EMA does not match journal")
    final = records[-max(1, len(records) // 4):]
    watched = ("cls", "attention_entropy", "signed_negative_fraction", "null_mean", "coverage_mean", "margin_change",
               "delta_relative_mean", "delta_nonzero_fraction", "prompt_primary", "residual_primary",
               "eta_mean", "alignment_cosine_mean", "protected_margin_drop_fraction", *settings)
    if name == "G2":
        watched += ("stats_reliability", "stats_effect_abs", "jacobian_coverage")
    return {
        "best_epoch": selected, "end_epoch": end, "epochs": len(records), "historical_rng_phases": archived_epochs,
        "dropout_replayed_epochs": len(records), "gradient_ratios": maxima, "ema_trace": "PASS",
        "final_quartile": {name: float(np.mean([row["diagnostics"][name] for row in final])) for name in watched},
        "actual_restart_epoch": recovery["epoch"] if recovery else None,
    }



def audit_source(manifest):
    actual = h.sha_file(HERE / "audit.py")
    score = manifest.get("audit_sha", actual)
    h.require(actual == manifest.get("audit_publish_sha", score), "Frozen audit implementation differs")
    return score, actual



def audit_fold(group, fold):
    manifest = h.read_json(run.MANIFEST)
    p = run.paths(group, fold)
    task, preflight = h.read_json(p["task"]), h.read_json(p["preflight"])
    scoring_sha, source_sha = audit_source(manifest)
    h.require(task["implementation"] == preflight["implementation"] == manifest["implementation"]
              == h.implementation_hashes(), "Frozen audit or training sources differ")
    h.require(task["status"] == "COMPLETE" and preflight["status"] == "PASS", "Fold did not complete")
    h.require(h.sha_file(p["best"]) == task["best_sha"] and h.sha_file(p["latest"]) == task["latest_sha"],
              "Best or Latest SHA differs")
    best, latest = h.load_torch(p["best"]), h.load_torch(p["latest"])
    h.require(best["protected_sha"] == latest["protected_sha"] == manifest["protected_bundle_sha"]
              and latest["best_sha"] == task["best_sha"], "Shared protected-file provenance differs")
    ctx = run.setup(group, fold, validate_spectrum=True)
    state = ctx["state"]
    h.require(ctx["identity"] == preflight["identity"], "Warmup, Prompt, spectrum or statistics identity differs")
    expected_buffers = static_buffers(ctx)
    reference = reference_split(ctx)
    labels = state["labels_np"][state["test_np"]]
    historical = load_historical(fold, labels, ctx, manifest["protected_sha256"])
    run.restore(ctx, best, manifest["implementation"])
    h.require(h.sha(run.state_dict(ctx)) == h.sha(best["state"]), "Best full state did not reload")
    h.require(static_buffers(ctx) == expected_buffers, "Best reference buffers drifted")
    results, response, formula, relative = checkpoint_mechanisms(ctx, best)
    recoveries = [wave.get("slow_context_restores", {}).get(str(fold))
                  for wave in manifest["training_runs"] if wave["group"] == group]
    recovery = next((r for r in reversed(recoveries) if r is not None), None)
    progress = audit_journal(ctx, task, best, latest, preflight, recovery)
    restored = replay_latest(ctx, latest, manifest)
    h.require(static_buffers(ctx) == expected_buffers, "Latest reference buffers drifted")
    variants = {**historical, **results}
    audit_geometry = label_free_geometry_subset(ctx, response)
    patients = []
    for j, row in enumerate(state["test_np"]):
        patient = {"row_1based": int(row) + 1, "label": int(labels[j]), "residual_relative": float(relative[j])}
        for key, measured in variants.items():
            patient[f"{key}_prediction"] = int(measured["prediction"][j])
            patient[f"{key}_p_asd"] = float(measured["probability"][j, 0])
            patient[f"{key}_true_margin"] = float(measured["margin"][j])
        patients.append(patient)
    report = {
        "status": "PASS", "group": group, "fold": fold, "best_exact": True, "rng_status": "PASS",
        "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "audit_sha": scoring_sha, "audit_code_sha": source_sha,
        "warmup_sha": ctx["identity"]["identity"]["warmup_sha256"], "source_and_protected_identity": "PASS",
        "membership_sha": ctx["identity"]["membership_sha"], "spectrum_sha": ctx["identity"]["spectrum_sha"],
        "prompt_sha": ctx["identity"]["query_sha"], "reference_sha": ctx["identity"]["reference_sha"],
        "parameters": ctx["identity"]["trainable_parameters"], "reference": reference, "formulas": formula,
        "mechanism": response, "progress": progress, "latest": restored, "geometry": audit_geometry,
        "metrics": {key: measured["metrics"] for key, measured in variants.items()},
        "best_epochs": {key: measured["best_epoch"] for key, measured in variants.items()}, "patients": patients,
        "initial_gradient": preflight["gradient_paths"], "initial_epoch_diagnostics": preflight["epoch11_diagnostics"],
    }
    h.atomic_json(p["audit"], report)
    print(f"PASS {group} fold {fold}: Best, Latest, gradient and {progress['epochs']} RNG phases", flush=True)



def comparisons_for(group):
    return [*((name, "full") for name in HISTORICAL), ("DARG", "residual_off"),
            ("residual_off", "full"), ("query_replaced", "full"), ("semantic_off", "full"),
            *((variant, "full") for variant in h.GROUPS[group]["variants"])]


def summarize(all_audits):
    complete_metrics, paired, per_fold_pairs = {}, {}, []
    for group, audits in all_audits.items():
        complete_metrics[group], paired[group] = {}, {}
        test_rows = sorted(p["row_1based"] for row in audits for p in row["patients"])
        eq(test_rows, list(range(1, CFG["subjects"] + 1)), "Outer folds do not cover each patient exactly once")
        for scope, selected in zip(SCOPES, (audits, [a for a in audits if a["fold"] != 2])):
            all_patients = [p for row in selected for p in row["patients"]]
            metrics, comp = {}, {}
            for method in selected[0]["metrics"]:
                measures = np.asarray([[a["metrics"][method][k] * 100 for k in h.METRICS] for a in selected])
                metrics[method] = {
                    k: {"mean": float(mean), "sd": float(sd)}
                    for k, mean, sd in zip(h.METRICS, measures.mean(0), measures.std(0, ddof=0))
                }
            for previous, current in comparisons_for(group):
                by_metric = {}
                differences = np.asarray([
                    [(a["metrics"][current][key] - a["metrics"][previous][key]) * 100 for key in h.METRICS]
                    for a in selected
                ])
                differences[np.abs(differences) < 1e-10] = 0
                for k, val in zip(h.METRICS, differences.T):
                    by_metric[k] = {"mean_pp": float(val.mean()), "sd_pp": float(val.std(ddof=0)),
                                    "wins": int((val > 0).sum()), "ties": int((val == 0).sum()), "losses": int((val < 0).sum())}
                comp[f"{previous}->{current}"] = {"metrics": by_metric, **delta_analysis(all_patients, previous, current)}
                if scope == SCOPES[0]:
                    for a, values in zip(selected, differences):
                        per_fold_pairs.append({
                            "group": group, "fold": a["fold"], "from": previous, "to": current,
                            **{key + "_delta_pp": float(value) for key, value in zip(h.METRICS, values)},
                            **delta_analysis(a["patients"], previous, current),
                        })
            complete_metrics[group][scope] = metrics
            paired[group][scope] = comp
    for group in METHODS[1:]:
        for scope in SCOPES:
            for history in HISTORICAL:
                h.require(complete_metrics[group][scope][history] == complete_metrics[METHODS[0]][scope][history],
                          "Historical references disagree across independent group audits")
    return complete_metrics, paired, per_fold_pairs



def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])



def summary_table(summary, scope):
    summaries = [(hist, summary["G1"][scope][hist]) for hist in HISTORICAL]
    summaries += [(f"E82.4.1-{g} {h.GROUPS[g]['method']}", summary[g][scope]["full"]) for g in METHODS]
    return table(("方法", *METRIC_NAMES), [
        (name, *(f"{metrics[key]['mean']:.4f} ± {metrics[key]['sd']:.4f}" for key in h.METRICS))
        for name, metrics in summaries
    ])



def paired_row(group, comparison, values):
    numbers = values["metrics"]
    def wtl(key):
        return "/".join(str(numbers[key][k]) for k in ("wins", "ties", "losses"))
    return (group, comparison, *(f"{numbers[k]['mean_pp']:+.4f}" for k in h.METRICS), wtl("acc"), wtl("hard_auc"),
            f"{values['asd_corrected']}/{values['asd_harmed']}", f"{values['td_corrected']}/{values['td_harmed']}",
            f"{values['corrected']}/{values['harmed']}/{values['net']:+d}")



def paired_table(rows):
    return table(("组", "来源 → 目标", *METRIC_NAMES, "ACC 胜/平/负", "BA 胜/平/负",
                  "ASD 纠错/误伤", "TD 纠错/误伤", "纠错/误伤/净"), rows)



def fold_mean(rows, key, section="mechanism"):
    return float(np.mean([r[section][key] for r in rows]))



def static_buffers(ctx):
    mutable = {"statistics.mean", "statistics.variance", "statistics.updates"}
    return h.sha({name: value for name, value in ctx["module"].named_buffers() if name not in mutable})


def bank_samples(points, labels, groups, n_splits):
    for index in range(n_splits + 1):
        kept = groups != index
        yield tuple(points[kept & (labels == label)] for label in (0, 1))


def reference_split(ctx):
    state, saved, module = ctx["state"], ctx["reference"], ctx["module"]
    train, test, legal = state["train_np"], state["test_np"], state["legal_np"]
    h.require(np.intersect1d(legal, test).size == 0, "A test row enters new supervision")
    eq(np.setdiff1d(train, legal), state["excluded"], "Historical overlap differs from residual mask")
    expected = np.ones(CFG["subjects"], dtype=bool)
    expected[state["excluded"]] = False
    eq(array(module.allowed), expected, "New branch mask differs")
    if saved is None:
        return {"supervision_legal": len(legal), "excluded_1based": [i+1 for i in state["excluded"]]}
    cfg = ctx["method"]["stats"]
    labels, points = state["labels_np"][legal], ctx["reference_z"]
    eq(saved["legal"], legal, "Crossfit uses a nonlegal index")
    eq(saved["labels"], labels, "Saved class labels differ")
    observations = np.zeros(len(legal), dtype=np.int64)
    seed = cfg["seed_base"] + state["fold"]
    h.require(seed == saved["seed"], "Inner partition seed differs")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.RandomState(seed))
    for index, (fit, hold) in enumerate(splitter.split(points, labels), 1):
        observations[hold] = index
        h.require(not np.intersect1d(legal[fit], legal[hold]).size, "OOF bank contains its holdout")
        eq(np.sort(legal[np.r_[fit, hold]]), np.sort(legal), "OOF ordering omitted a legal row")
    eq(saved["group"], observations, "Crossfit membership differs")
    expected_banks = np.zeros(CFG["subjects"], dtype=np.int64)
    expected_banks[legal] = observations
    eq(array(module.statistics.bank), expected_banks, "Current EMA bank selection differs")
    counts = []
    for index, samples in enumerate(bank_samples(points, labels, observations, 5)):
        counts.append([len(s) for s in samples])
        close(saved["mean"][index], [s.mean(axis=0) for s in samples], "E61 class centroid", atol=1e-12)
        close(saved["variance"][index], [s.var(axis=0) for s in samples], "E61 class variance", atol=1e-12)
    eq(saved["counts"], counts, "Original EMA per-class counts")
    eq(array(module.statistics.counts), counts, "EMA buffer per-class counts")
    return {"supervision_legal": len(legal), "excluded_1based": [i+1 for i in state["excluded"]],
            "split_seed": seed, "full_and_inner_counts": counts, "initial_exact_scope": True}


def analytic_jacobian(head, cam):
    norm, first, gelu, dropout = head.mlp_head
    h.require(gelu.approximate == "none" and isinstance(dropout, torch.nn.Dropout),
              "Deterministic head architecture differs")
    c = np.asarray(cam, dtype=np.float64)
    centered = c - c.mean(-1, keepdims=True)
    sd = np.sqrt(np.square(centered).mean(-1, keepdims=True) + norm.eps)
    scaled = centered / sd
    x = (scaled * array(norm.weight) + array(norm.bias)) @ array(first.weight).T + array(first.bias)
    gelu_prime = ndtr(x) + x * np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    direction = (gelu_prime * (array(head.classifier.weight[0]) - array(head.classifier.weight[1]))) @ array(first.weight)
    direction *= array(norm.weight)
    return (direction - direction.mean(-1, keepdims=True) - scaled * (scaled * direction).mean(-1, keepdims=True)) / sd


def verify_branch(ctx, cam, d, *, training, semantic_off=False):
    module = ctx["module"]
    U = array(module.basis)
    c, branch = array(cam), array(d["branch_cam"])
    eq(branch, c, "The residual branch changed its input features")
    errors = {}
    errors["original_o"] = float32_dot(array(d["z"]), c.astype(np.float64) - array(module.center), U,
                                       "Original O projection", preceding_rounding=1)
    attention = array(d["attention"])
    h.require(attention.shape == (CFG["subjects"], 8, 17) and np.isfinite(attention).all() and (attention > 0).all(),
              "Nonfinite or invalid Prompt Token and Null attention")
    close(attention.sum(-1), 1, "Softmax normalization", atol=2e-6)
    expected = 1 - np.prod(1 - attention[..., :16].astype(np.float64), axis=1)
    if semantic_off:
        expected = np.full_like(expected, 1 - (16 / 17) ** 8)
        eq(array(d["context"]), np.zeros_like(array(d["context"])), "Semantic-off still reads medical Query")
    close(array(d["coverage"]), expected, "Continuous semantic coverage", atol=2e-6)
    eta, action = array(d["eta"]), array(d["action"])
    h.require(np.isfinite(action).all() and (np.abs(action) <= 1).all() and ((eta > 0) & (eta < 1)).all(),
              "Signed residual or strength is out of range")
    cam_norm = np.linalg.norm(c.astype(np.float64), axis=-1, keepdims=True)
    direction = action / math.sqrt(16)
    if "jacobian" in d:
        head = ctx["state"]["cam"].Outputlayer
        gj = array(d["jacobian"])
        close(gj, analytic_jacobian(head, c), "Analytic native logit-margin Jacobian", atol=3e-6, rtol=3e-5)
        errors["jacobian_projection"] = float32_dot(array(d["projected_jacobian"]), gj, U, "Jacobian projection")
        g = array(d["projected_jacobian"]).astype(np.float64)
        carrier = g / (np.linalg.norm(g, axis=-1, keepdims=True) + ctx["method"]["jacobian_floor"])
        close(array(d["carrier"]), carrier, "Stable bounded Jacobian", atol=3e-6)
        direction = action * carrier
        bank = module.statistics
        cfg = ctx["method"]["stats"]
        ids = array(bank.bank) if training else np.zeros(len(c), dtype=np.int64)
        mean, variance, count = [array(getattr(bank, name))[ids].astype(np.float64)
                                 for name in ("mean", "variance", "counts")]
        anchor = array(bank.anchor).astype(np.float64)
        shrink = cfg["reference_shrinkage"]
        scale = np.maximum((1 - shrink) * variance.mean(axis=1) + shrink * anchor,
                           anchor * cfg["relative_variance_floor"] + cfg["absolute_variance_floor"]) ** .5
        diff = (mean[:, 0] - mean[:, 1]) / scale
        position = (array(d["z"]) - mean.mean(axis=1)) / scale
        var_mean = (variance / count[..., None]).sum(axis=1) / scale ** 2
        expected = diff ** 2 / (diff ** 2 + var_mean + cfg["effect_prior"] ** 2)
        expected /= np.sqrt(1 + (position / cfg["support_radius"]) ** 2)
        close(array(d["reliability"]), expected, "Continuous on-distribution class reliability", atol=3e-6)
        close(array(d["class_evidence"]), np.tanh(position * diff) * expected, "Reliable class evidence", atol=3e-6)
    close(array(d["raw_z"]), CFG["branch_scale"] * cam_norm * eta * direction, "Signed spectral residual", atol=3e-6)
    raw = array(d["raw_z"]).astype(np.float64) @ U.astype(np.float64).T
    raw[~array(module.allowed)] = 0
    raw_norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    cap = CFG["residual"]["norm_limit"] * cam_norm
    ratio = raw_norm / np.maximum(cap, 1e-8)
    scale = np.divide(np.tanh(ratio), ratio, out=np.ones_like(ratio), where=ratio > 0)
    expected_delta = raw * scale * (cap / np.maximum(cap, 1e-8))
    delta, rel = array(d["delta"]), array(d["relative"])
    close(delta, expected_delta, "Smooth bounded residual in CAM144", atol=4e-6, rtol=3e-5)
    h.require(float(rel.max()) <= CFG["residual"]["norm_limit"] + 2e-6, "Smooth 8% bound was exceeded")
    eq(delta[ctx["state"]["excluded"]], np.zeros((len(ctx["state"]["excluded"]), 144)), "Fold 2 residual is active")
    errors.update(excluded_path_zero=True, semantic_off_isolated=semantic_off, branch_forward_identity=True)
    return errors


def quantiles(data):
    data = np.asarray(data)
    if not len(data):
        return {"n": 0}
    percentiles = np.quantile(data, (.1, .5, .9))
    return dict(n=len(data), mean=float(data.mean()), p10=float(percentiles[0]),
                median=float(percentiles[1]), p90=float(percentiles[2]))


def mechanism_geometry(head, cam, delta, basis):
    logits, g = run.head_jacobian(head, cam)
    adjusted = run.frozen_head(head, cam + delta)
    linear = (g * delta).sum(-1)
    gnorm = g.norm(dim=-1)
    actual = (adjusted[:, 0] - adjusted[:, 1]) - (logits[:, 0] - logits[:, 1])
    return array(torch.stack(((g @ basis).square().sum(-1) / (g.square().sum(-1) + 1e-8),
                              linear / (gnorm * delta.norm(dim=-1) + 1e-8), linear, actual), dim=-1))


@torch.no_grad()
def checkpoint_mechanisms(ctx, best):
    state, module = ctx["state"], ctx["module"]
    before, rng = h.sha(run.state_dict(ctx)), h.rng_sha()
    rows = state["test_np"]
    labels = state["labels_np"][rows]
    original, capture = run.forward(state, False)
    c, head = capture["cam"], state["cam"].Outputlayer
    eq(array(run.frozen_head(head, c)), array(capture["logits"]), "No-Dropout copy differs from native head")

    def score(logp, logits):
        prob = array(logp[rows].exp())
        decisions = prob.argmax(-1)
        sliced = array(logits[rows])
        return {"probability": prob, "prediction": decisions, "logits": sliced,
                "margin": (sliced[:, 0] - sliced[:, 1]) * (1 - 2 * labels),
                "metrics": metric_values(prob, decisions, labels), "best_epoch": best["best_epoch"]}

    result = {"residual_off": score(original["logp"], capture["logits"])}
    views = {}
    flags = {
        "full": {}, "query_replaced": {"query": ctx["text"]["q_anonymous"].to(c.device)},
        "semantic_off": {"semantic_off": True},
        **{variant: {"variant": variant} for variant in ctx["method"]["variants"]},
    }
    for name, options in flags.items():
        enhanced, d = module(c, head=head, training=False, **options)
        logp, _ = head(enhanced, None)
        logits = head.classifier(head.mlp_head(enhanced))
        h.require(torch.equal(logp, F.log_softmax(logits, dim=-1)), "Native classification differs on intervention")
        result[name] = score(logp, logits)
        views[name] = d
    full = result["full"]
    for key in ("probability", "logits", "prediction"):
        eq(full[key], best["evaluation"][key], f"Best independently reloaded {key} differ")
    h.require(int((labels == full["prediction"]).sum()) == best["evaluation"]["correct"], "Best replay count differs")
    d = views["full"]
    formula = verify_branch(ctx, c, d, training=False)
    formula["semantic_off"] = verify_branch(ctx, c, views["semantic_off"], training=False, semantic_off=True)
    if state["excluded"]:
        masked = np.isin(rows, state["excluded"])
        eq(full["logits"][masked], result["residual_off"]["logits"][masked], "Fold 2 excluded logits changed")
        for view in views.values():
            h.require(not bool(view["delta"][state["excluded"]].count_nonzero()), "A mechanism changed Fold 2 mask")
    delta, action, eta = d["delta"], array(d["action"][rows]), array(d["eta"][rows, 0])
    relative = array(d["relative"][rows])
    changes = {}
    for name in result:
        if name == "full":
            continue
        alternative = views[name]["delta"] if name in views else torch.zeros_like(delta)
        diff = (delta[rows] - alternative[rows]).norm(dim=-1)
        logit_shift = np.abs(full["logits"] - result[name]["logits"])
        changes[name] = {"cam_delta_mean": float(diff.mean()), "cam_delta_max": float(diff.max()),
                         "logit_delta_abs_mean": float(logit_shift.mean()), "logit_delta_abs_max": float(logit_shift.max()),
                         "acc_full_minus_pp": 100 * (full["metrics"]["acc"] - result[name]["metrics"]["acc"]),
                         "ba_full_minus_pp": 100 * (full["metrics"]["hard_auc"] - result[name]["metrics"]["hard_auc"])}
    geometry = mechanism_geometry(head, c, delta, module.basis)
    attention = d["attention"][rows]
    rho, _, linear, actual = geometry[rows].T
    reference_margin = result["residual_off"]["margin"]
    gain = full["margin"] - reference_margin
    response = {
        "residual_relative_mean": float(relative.mean()), "residual_relative_max": float(relative.max()),
        "nonzero_test": int((relative > 1e-6).sum()), "action_negative_fraction": float((action < 0).mean()),
        "action_direction_sd_mean": float(action.std(axis=1).mean()), "strength_mean": float(eta.mean()),
        "strength_sd": float(eta.std()), "coverage_mean": float(d["coverage"][rows].mean()),
        "null_mean": float(attention[..., 16].mean()),
        "attention_entropy": float(-(attention * attention.clamp_min(1e-8).log()).sum(-1).mean() / math.log(17)),
        "rho_mean": float(rho.mean()), "linear_asd_margin_mean": float(linear.mean()),
        "actual_asd_margin_mean": float(actual.mean()),
        "head_linear_error_mean": float(np.abs(actual - linear).mean()),
        "head_linear_sign_agreement": float((np.sign(actual) == np.sign(linear)).mean()),
        "wrong_train_policy_evaluated_on_test_margin": quantiles(gain[reference_margin < 0]),
        "correct_train_policy_evaluated_on_test_margin": quantiles(gain[reference_margin >= 0]),
        "interventions": changes, "geometry_rows": geometry,
    }
    if "reliability" in d:
        response.update(
            reliability_mean=float(d["reliability"][rows].mean()), reliability_sd=float(d["reliability"][rows].std(unbiased=False)),
            effect_abs_mean=float(d["effect"][rows].abs().mean()), class_evidence_abs_mean=float(d["class_evidence"][rows].abs().mean()),
            jacobian_norm_mean=float(d["projected_jacobian"][rows].norm(dim=-1).mean()),
            jacobian_low_norm=int((d["projected_jacobian"][rows].norm(dim=-1) < ctx["method"]["jacobian_floor"]).sum()),
        )
    h.require((h.sha(run.state_dict(ctx)), h.rng_sha()) == (before, rng), "Best mechanism evaluation changed state or RNG")
    return result, response, formula, relative


def label_free_geometry_subset(ctx, response):
    state = ctx["state"]
    values = response.pop("geometry_rows")
    return {
        "train": {key: quantiles(values[state["legal_np"], j]) for j, key in enumerate(GEOMETRY_FIELDS)},
        "test": {key: quantiles(values[state["test_np"], j]) for j, key in enumerate(GEOMETRY_FIELDS)},
    }


def verify_next_ema(ctx, next_bundle, latest):
    state, module = ctx["state"], ctx["module"]
    if not hasattr(module, "statistics"):
        return None
    z, before = next_bundle, latest["state"]["branch"]
    reference = ctx["reference"]
    alpha = ctx["method"]["stats"]["ema_alpha"]
    expected_mean, expected_var = [], []
    for index, classes in enumerate(bank_samples(z, reference["labels"], reference["group"], 5)):
        means = np.stack([s.mean(axis=0) for s in classes])
        variances = np.stack([s.var(axis=0) for s in classes])
        old_mean = array(before["statistics.mean"])[index].astype(np.float64)
        old_var = array(before["statistics.variance"])[index].astype(np.float64)
        expected_mean.append((1 - alpha) * old_mean + alpha * means)
        expected_var.append((1 - alpha) * old_var + alpha * variances + alpha * (1 - alpha) * (means - old_mean) ** 2)
    close(array(module.statistics.mean), expected_mean, "Independent EMA centroid advance", atol=2e-7, rtol=2e-6)
    close(array(module.statistics.variance), expected_var, "Independent EMA variance advance", atol=4e-7, rtol=4e-6)
    current = run.statistics_identity(ctx)
    h.require(current["updates"] == latest["next_epoch"] - run.START + 1, "Latest continuation omitted EMA advance")
    return current


def replay_latest(ctx, latest, manifest):
    state = ctx["state"]
    group, fold = ctx["method"]["group"], state["fold"]
    impl = manifest["implementation"]
    run.restore(ctx, latest, impl)
    h.require(h.sha(run.state_dict(ctx)) == h.sha(latest["state"]), "Latest full state did not reload")
    tested = run.evaluate(ctx)
    for key in tested:
        eq(tested[key], latest["evaluation"][key], f"Latest evaluation differs: {key}")
    h.require(h.rng_sha() == latest["rng_sha"] and run.statistics_identity(ctx) == latest["ema"],
              "Latest evaluation mutated RNG or EMA")
    with torch.no_grad():
        _, capture = run.forward(state, True, ctx["module"])
        z = array(capture["details"]["z"][state["legal"]]).astype(np.float64)
        training_formula = verify_branch(ctx, capture["cam"], capture["details"], training=True)
    del capture
    run.restore(ctx, latest, impl)

    def advance(watch):
        diag, gradient = run.train_step(ctx, inspect=watch)
        current = verify_next_ema(ctx, z, latest)
        trained_rng = h.rng_sha()
        evaluated = run.evaluate(ctx)
        h.require(trained_rng == h.rng_sha() and current == run.statistics_identity(ctx),
                  "Continuation evaluation changed RNG or EMA")
        return h.sha((diag, evaluated, run.state_dict(ctx))), trained_rng, gradient, current

    first, after_rng, gradient, next_ema = advance(True)
    saved = h.load_torch(run.paths(group, fold)["latest"])
    run.restore(ctx, saved, impl)
    replayed, replayed_rng, _, _ = advance(False)
    h.require((first, after_rng) == (replayed, replayed_rng), "Latest next train/eval transition differs on exact replay")
    archived = {row["epoch"]: row for row in h.historical_task(fold)["rng_trace"]}
    if (expected := archived.get(latest["next_epoch"] + 1)) is not None:
        h.require(after_rng == expected["rng_after_train"], "Latest continuation changed archived RNG")
    run.restore(ctx, saved, impl)
    return {"status": "PASS", "next_transition_sha": first, "next_rng_sha": after_rng,
            "next_epoch_probe": latest["next_epoch"] + 1, "class_gradient": gradient,
            "training_formulas": training_formula, "next_ema": next_ema}


def archived_geometry(all_audits, manifest):
    by_method = {}
    for row in all_audits["G1"]:
        fold = row["fold"]
        path = run.RUNTIME / "geometry" / f"f{fold:02d}.json"
        h.require(h.sha_file(path) == manifest["geometry"]["artifacts"][str(path)], "Protected geometry artifact differs")
        stored = h.read_json(path)
        h.require(stored["status"] == "PASS" and stored["fold"] == fold, "Geometry fold identity differs")
        legal_test = [p["row_1based"] - 1 for p in row["patients"]]
        labels = np.asarray([p["label"] for p in row["patients"]])
        for name, artifact in stored["history"].items():
            train = np.asarray(artifact["train_rows"], dtype=np.float64)
            test = np.asarray(artifact["test_rows"], dtype=np.float64)
            eq(test[:, 0], legal_test, "Label-free geometry has a different test order")
            h.require(not np.intersect1d(train[:, 0], test[:, 0]).size, "Label-free diagnostic training scope is contaminated")
            if name in HISTORICAL:
                eq(test[:, 1], [p[name + "_prediction"] for p in row["patients"]], "Historical geometry hard outcome differs")
            cohorts = by_method.setdefault(name, {})
            for scope, values, hit in (("train", train[:, 2:], train[:, 1] == 1),
                                       ("test", test[:, 2:], test[:, 1] == labels)):
                for key, selection in (("all", np.ones(len(hit), dtype=bool)), ("correct", hit), ("wrong", ~hit)):
                    cohorts.setdefault(scope + "_" + key, []).append(values[selection])
    return {name: {scope: {
        key: quantiles(np.concatenate(blocks)[:, index]) for index, key in enumerate(GEOMETRY_FIELDS)
    } for scope, blocks in cohorts.items()} for name, cohorts in by_method.items()}


def scientific_decision(audits, paired):
    decisions = {}
    for group in METHODS:
        main = paired[group][SCOPES[0]]
        performance, direct, query = (main[k] for k in ("DARG->full", "residual_off->full", "query_replaced->full"))
        sensitivity = paired[group][SCOPES[1]]["DARG->full"]
        performance_checks = {
            "acc_gain": performance["metrics"]["acc"]["mean_pp"] > 0,
            "ba_gain": performance["metrics"]["hard_auc"]["mean_pp"] > 0,
            "sen_gain": performance["metrics"]["sen"]["mean_pp"] > 0,
            "net_corrections": performance["net"] > 0,
            "exclude_fold2_acc_gain": sensitivity["metrics"]["acc"]["mean_pp"] > 0,
            "exclude_fold2_net_gain": sensitivity["net"] > 0,
        }
        mechanism_checks = {
            "direct_positive_hard_net": direct["net"] > 0,
            "direct_ba_gain": direct["metrics"]["hard_auc"]["mean_pp"] > 0,
            "observable_query_probability_effect": query["p_asd_measurably_changed"] > 0,
            "observable_query_hard_effect": query["flips"] > 0,
            **{variant + "_changes_cam144":
                max(a["mechanism"]["interventions"][variant]["cam_delta_max"] for a in audits[group])
                > CFG["comparison"]["minimum_mechanism_delta"] for variant in h.GROUPS[group]["variants"]},
        }
        decisions[group] = {"performance": "GO" if all(performance_checks.values()) else "NO-GO",
                            "mechanism": "GO" if all(mechanism_checks.values()) else "NO-GO",
                            "performance_checks": performance_checks, "mechanism_checks": mechanism_checks}
    return decisions


def write_tables(audits, comparisons):
    fold_metrics, mechanisms = [], []
    for group, rows in audits.items():
        for row in rows:
            names = (*HISTORICAL, "full") if group == METHODS[0] else ("full",)
            for name in names:
                new = name == "full"
                values = row["metrics"][name]
                fold_metrics.append({
                    "method": f"E82.4.1-{group}" if new else name,
                    "formal_method": h.GROUPS[group]["method"] if new else "",
                    "fold": row["fold"], "best_epoch": row["best_epochs"][name],
                    "end_epoch": row["progress"]["end_epoch"] if new else "",
                    **{k: v * 100 for k, v in values.items()},
                    **({f"{k}_delta_vs_darg_pp": (v - row["metrics"]["DARG"][k]) * 100 for k, v in values.items()}
                       if new else {}),
                })
            detail = row["mechanism"]
            shared = {key: value if not isinstance(value, dict) else json.dumps(value, separators=(",", ":"))
                      for key, value in detail.items() if key != "interventions"}
            for name, changes in detail["interventions"].items():
                mechanisms.append({"group": group, "fold": row["fold"], "from": name, "to": "full",
                                   **changes, **delta_analysis(row["patients"], name, "full"), **shared})
    h.write_csv(RESULTS / "fold_metrics.csv", fold_metrics)
    h.write_csv(RESULTS / "mechanism.csv", mechanisms)
    h.write_csv(RESULTS / "paired_comparisons.csv", comparisons)


def recommendation(decisions):
    shortlisted = [group for group in METHODS
                   if decisions[group]["performance"] == decisions[group]["mechanism"] == "GO"]
    if shortlisted:
        return "预设性能和机制条件同时达标的候选为 " + "、".join(shortlisted) + "。投稿前需在独立样本和预设文本对照中复核。"
    mechanism = [group for group in METHODS if decisions[group]["mechanism"] == "GO"]
    if mechanism:
        return ("本轮没有同时满足性能与机制条件的 MICCAI 2027 候选。可保留 "
                + "、".join(mechanism) + " 的同权重纠错证据，后续优先验证主干共同适配的影响和独立泛化。")
    return "本轮没有同时满足性能与机制条件的 MICCAI 2027 候选。后续应先验证有符号修正与真实分类收益的对应关系，再考虑扩展方法。"


def mechanism_rows(audits, paired):
    significance = []
    for group in METHODS:
        rows = audits[group]
        for name in ("residual_off", "query_replaced", "semantic_off", *h.GROUPS[group]["variants"]):
            differences = [row["mechanism"]["interventions"][name] for row in rows]
            pair = paired[group][SCOPES[0]][name + "->full"]
            significance.append((group, name, f"{np.mean([d['cam_delta_mean'] for d in differences]):.5f}",
                                 f"{max(d['cam_delta_max'] for d in differences):.5f}",
                                 f"{np.mean([d['logit_delta_abs_mean'] for d in differences]):.6f}",
                                 f"{max(d['logit_delta_abs_max'] for d in differences):.6f}",
                                 pair["p_asd_measurably_changed"], pair["flips"],
                                 f"{pair['asd_corrected']}/{pair['asd_harmed']}",
                                 f"{pair['td_corrected']}/{pair['td_harmed']}", f"{pair['net']:+d}",
                                 f"{pair['metrics']['acc']['mean_pp']:+.4f}", f"{pair['metrics']['hard_auc']['mean_pp']:+.4f}"))
    return table(("组", "干预 → Full", "残差 L2 均值", "残差 L2 最大", "logit 绝对差均值", "logit 最大绝对差",
                  "概率变动", "翻转", "ASD 纠错/误伤", "TD 纠错/误伤", "净纠错", "ΔACC", "ΔBA"), significance)


def result_pairs(paired, comparisons):
    return paired_table([paired_row(group, name.replace("->", " → "), paired[group][scope][name])
                         for group in METHODS for name, scope in comparisons])


def coverage_table(geometry):
    def qstats(value):
        return " / ".join(f"{value[q]:.4f}" for q in ("mean", "median", "p10", "p90")) if value["n"] else "—"
    return table(("历史 Best", "范围", "观测数", "ρ 均值 / 中位 / P10 / P90",
                  "残差与 ASD−TD 梯度余弦", "线性 ASD−TD Δmargin", "实际 ASD−TD Δmargin"), [
        (name, scope, values["rho"]["n"], qstats(values["rho"]), qstats(values["cosine_asd"]),
         qstats(values["linear_asd_margin"]), qstats(values["actual_asd_margin"]))
        for name, scopes in geometry.items() for scope, values in scopes.items()
    ])


def output_report(audits, metrics, paired, decisions, geometry, manifest):
    pre = manifest["preflight"]
    device = manifest["device"]
    waves = manifest["training_runs"]
    all_rows = [r for rows in audits.values() for r in rows]
    sha = manifest["implementation_sha"]
    report = [
        "# Experiment 82.4.1 · Prompt-guided Signed Utility and Jacobian-Aligned Spectral Correction",
        "## 研究问题与方案\n",
        "两组联合训练 DARG、CAM144、原 OutputLayer 和医学 Prompt 谱残差。G1 PSU-PSCE 继承 E82.2-G2 的八条 RN50 Query、16 谱 Token、Null 和连续 Coverage，将 Query 加权向量直接送入有符号方向网络。G2 JACS-DA-SRPE 继承 E82.3.2-G2 的疾病感知统计，在当前无 Dropout 分类头上一次批量 VJP 提取 ASD−TD 谱 Jacobian，再由医学 Query 和可靠的交叉拟合 EMA 控制正负修正。两组都在真实特征残差上计算训练 Margin 增益、正确样本保护与方向对齐。辅助梯度只进入新增参数。",
        "相较 E82.4，PSU-PSCE 在固定 Original O 上学习有符号效用，沿用 16 维特征路径。DPR 的旋转、HLG 的局部槽位、RCS 的冻结 Gaussian LLR、SOT 的低秩谱算子均由训练前锁定的这两条方法替代。两组共同使用 0.25 的分支输入梯度比例，保留直接分类梯度，并用连续 8% 范数限幅。详见 [预注册设计](../design.md)。",
        "## 工程与验证\n",
        f"- 权威目录为 `{HERE}`。首选 OSSFS 返回 errno 5 后回退。训练实现 SHA 为 `{sha}`，仓库 HEAD 为 `{manifest['repository_head']}`。保护检查覆盖 {manifest['guard_after']['files']} 项，训练前后 SHA 一致。",
        f"- 物理 GPU0 为 {device['name']}，UUID `{device['uuid']}`，显存 {device['total_mib']:,} MiB。单 worker 实测两组最大 PyTorch 保留显存 {pre['single_worker_peak_reserved_mib']:,} MiB，十并发估算 {pre['ten_worker_estimate_mib']:,} MiB。",
        f"- 两组统一 seed=0、每折 E61 epoch 10 的原模型和三个优化器状态、4096 原历史关系对，epoch 11 起训练，最多 450 epoch、wait > 200 停止，严格按测试折 ACC 改善更新 Best。按 Python、NumPy、Torch CPU、CUDA 四类状态验证六次历史 Dropout。",
        f"- 20/20 Best 逐样本概率和 logits 精确复算，20/20 Latest 当前评估和下一次训练及评估精确恢复。逐 epoch 重放 {sum(r['progress']['epochs'] for r in all_rows):,} 次训练 RNG，其中 {sum(r['progress']['historical_rng_phases'] for r in all_rows):,} 次同时对照历史 DARG 归档。20/20 首步和 Latest 主分类梯度通过。G2 交叉拟合、每 epoch EMA 状态链及独立下一步 EMA 都通过。",
        "- Fold 2 原历史训练路径仍含 377（ASD）和 407（TD）。新增残差在两人上精确为零，两人排除于新增监督和统计。转导式患者图、CV 前 RFE/预处理和 val=test 选模均继承历史协议，这些因素限制结果作为独立泛化证据的用途。",
        table(("组", "完成", "峰值 workers", "GPU 峰值 MiB", "Latest 慢进程续接"),
              [(group, f"{len(audits[group])}/10", max(w["max_workers"] for w in waves if w["group"] == group),
                max(w["peak_sampled_mib"] for w in waves if w["group"] == group),
                sum(len(w["slow_context_restores"]) for w in waves if w["group"] == group)) for group in METHODS]),
        "## 十折正式指标\n",
        "数值为百分比的折均值 ± population SD。ASD=0，Prob-AUC 使用 p(ASD)，Hard-AUC 使用硬预测且与 BA 一致。",
        summary_table(metrics, SCOPES[0]),
        "逐折指标与 Best、终止 epoch 见 [fold_metrics.csv](fold_metrics.csv)。",
        "## 配对结果和 Fold 2 敏感性\n",
        "差值按折计算后平均，以百分点表示。纠错和误伤按患者累计。",
        result_pairs(paired, [(hist + "->full", SCOPES[0]) for hist in HISTORICAL]),
        "下表排除 Fold 2，属于描述性敏感性分析。",
        summary_table(metrics, SCOPES[1]),
        result_pairs(paired, [("DARG->full", SCOPES[1])]),
        "## 特征残差、Prompt 与共同适配\n",
        "DARG → 当前 Residual-off 衡量共同适配后的变化。Residual-off → Full 衡量固定权重的直接残差作用。两者以同一患者为单位核对，所有比较均用同一测试顺序。",
        result_pairs(paired, [("DARG->residual_off", SCOPES[0]), ("residual_off->full", SCOPES[0])]),
        mechanism_rows(audits, paired),
        "概率变动以 |Δp(ASD)| > 10⁻⁶ 计数。匿名 Query 来自 E82.2 的 Gram 匹配缓存。有符号干预将 G1 动作取绝对值。G2 的 Jacobian 干预以等范数各向同性载体替代谱 Jacobian，可靠性干预保留类别证据并将可靠性置一。固定权重的文本替换事关此模型对 Query 的依赖，医学文本的独立诊断贡献仍需正式文本对照。",
    ]
    signal_rows = []
    for group, rows in audits.items():
        response = lambda key: fold_mean(rows, key)
        signal_rows.append((group, f"{100 * response('residual_relative_mean'):.4f}%",
                            f"{response('attention_entropy'):.4f}", f"{response('null_mean'):.4f}",
                            f"{response('coverage_mean'):.4f}", f"{response('action_negative_fraction'):.4f}",
                            f"{response('action_direction_sd_mean'):.4f}", f"{response('strength_sd'):.4f}",
                            f"{response('head_linear_error_mean'):.5f}", f"{response('head_linear_sign_agreement'):.4f}"))
        ratios = {fam: max(row["progress"]["gradient_ratios"][f"{fam}_applied_ratio"] for row in rows)
                  for fam in ("prompt", "residual")}
        report.append(f"{group} 每折测试相对残差均值的平均值为 {100 * response('residual_relative_mean'):.3f}%。"
                      f"全训练期实际辅助梯度比例最高为 Prompt {100 * ratios['prompt']:.2f}% 和修正网络 "
                      f"{100 * ratios['residual']:.2f}%。同权重残差对 ASD 和 TD 的真实 Margin 平均变化分别为 "
                      f"{paired[group][SCOPES[0]]['residual_off->full']['asd_true_margin_delta_mean']:+.4f} 和 "
                      f"{paired[group][SCOPES[0]]['residual_off->full']['td_true_margin_delta_mean']:+.4f}。")
        if group == "G2":
            report.append(f"G2 的测试类别可靠性均值为 {response('reliability_mean'):.4f}，折内标准差均值 "
                          f"{response('reliability_sd'):.4f}，投影 Jacobian 范数均值 {response('jacobian_norm_mean'):.4f}，"
                          f"低于稳定化尺度的病例 {sum(row['mechanism']['jacobian_low_norm'] for row in rows)} 人。")
    report.extend((
        table(("组", "相对残差", "注意力熵", "Null", "Coverage", "负动作比例", "方向动作 SD",
               "患者强度 SD", "线性误差绝对均值", "线性/实际符号一致率"), signal_rows),
        "逐折 CAM、logit、概率、Margin 与硬预测对照见 [mechanism.csv](mechanism.csv)。完整配对结果见 [paired_comparisons.csv](paired_comparisons.csv)。",
        "## 固定谱空间覆盖与 Jacobian 对齐\n",
        "ρ 是原分类头 ASD−TD 梯度在固定 16 维子空间中保留的平方范数占比。历史 Best 预检对测试患者只计算无标签几何量。测试正确/错误分层在两组正式训练结束后加入实际标签。训练部分按同一患者在不同训练折的观测计数。",
        coverage_table(geometry),
        table(("新组", "训练 ρ 均值", "测试 ρ 均值", "测试残差余弦均值", "线性 ASD−TD Δmargin", "实际 ASD−TD Δmargin"),
              [(g, *(f"{np.mean([r['geometry']['train' if k == 'train_rho' else 'test'][k.removeprefix('train_')]['mean'] for r in rows]):.5f}"
                     for k in ("train_rho", *GEOMETRY_FIELDS))) for g, rows in audits.items()]),
        "## 科研判断\n",
        table(("组", "工程", "独立审计", "性能", "机制", "未满足性能条件", "未满足机制条件"), [
            (g, "20 折任务中的 10/10", "10/10", d["performance"], d["mechanism"],
             "、".join(k for k, v in d["performance_checks"].items() if not v) or "—",
             "、".join(k for k, v in d["mechanism_checks"].items() if not v) or "—")
            for g, d in decisions.items()
        ]),
        recommendation(decisions),
        "逐折参数、优化器、源码、RNG、成员身份、有限精度公式检查和分层结果见 [audit_status.json](audit_status.json) 与 [manifest.json](../manifest.json)。",
    ))
    return "\n\n".join(report) + "\n"


def handoff_summary(audits, metrics, paired, decisions, manifest):
    device = manifest["device"]
    waves = manifest["training_runs"]
    workers = max(w["max_workers"] for w in waves)
    sections = [
        HANDOFF_MARKER,
        "Prompt-guided Signed Utility and Jacobian-Aligned Spectral Correction。G1 从 PSCE 的八条医学 RN50 Query 和 Coverage 学习有符号谱修正，以原分类头实际 Margin 监督。G2 用原分类头批量确定性 Jacobian 和训练内五折交叉拟合 EMA 类别统计形成 Prompt 条件化的有符号残差。",
        f"权威目录 `{HERE}`。OSSFS mkdir 返回 errno 5，使用本地实验路径。GPU0 {device['name']}，UUID `{device['uuid']}`，实际每波最多 {workers} workers。仓库 HEAD `{manifest['repository_head']}`，新增训练源码 SHA `{manifest['implementation_sha']}`，两组各有 10 份 Best 和 10 份 Latest。具体状态、身份和 SHA 见实验 manifest。",
        "状态 COMPLETE_AUDITED。G1、G2 共 20/20 正式折，逐折 E61 epoch 10 参数、三优化器、4096 关系对和四类 RNG 恢复一致。20/20 Best 逐样本复算、Latest 下一步训练与评估、逐 epoch RNG、分类梯度、合法成员与 Fold 2 屏蔽通过。",
        summary_table(metrics, SCOPES[0]),
        result_pairs(paired, [("DARG->full", SCOPES[0]), ("DARG->residual_off", SCOPES[0]), ("residual_off->full", SCOPES[0])]),
    ]
    for group in METHODS:
        record = paired[group][SCOPES[0]]["query_replaced->full"]
        excluded = paired[group][SCOPES[1]]["DARG->full"]
        d = decisions[group]
        sections.append(f"{group} 的医学 Query 替换使 {record['p_asd_measurably_changed']} 人的 p(ASD) 可测变化、"
                        f"{record['flips']} 人翻转。测试平均相对 CAM 残差 {100 * fold_mean(audits[group], 'residual_relative_mean'):.3f}%。"
                        f"去除 Fold 2 后相对 DARG ΔACC {excluded['metrics']['acc']['mean_pp']:+.4f} pp、净纠错 {excluded['net']:+d} 人。"
                        f"性能 {d['performance']}，机制 {d['mechanism']}。")
    sections.extend((recommendation(decisions),
                     "协议继承 val=test 选模、外层 CV 前 RFE/预处理、转导式患者图与历史 Fold 2 重叠。两名重叠患者仅保留在历史训练路径，新增统计、监督和残差均已屏蔽。",
                     f"交付位于 `{RESULTS}`，包括 report.md、fold_metrics.csv、paired_comparisons.csv、mechanism.csv 和 audit_status.json。"))
    return "\n\n".join(sections) + "\n"


def audit_all():
    manifest = h.read_json(run.MANIFEST)
    h.require(manifest["status"] in ("TRAINED_PENDING_AUDIT", "AUDITING", "COMPLETE_AUDITED")
              and all(len(manifest["completed_folds"].get(group, [])) == CFG["folds"] for group in METHODS),
              "All 20 formal runs must finish before final audit")
    h.require(manifest["implementation"] == h.implementation_hashes(), "Frozen training source differs")
    audit_sha, source_sha = audit_source(manifest)
    manifest.update(audit_sha=audit_sha, audit_publish_sha=source_sha, status="AUDITING")
    h.atomic_json(run.MANIFEST, manifest)
    all_audits = {}
    for group in METHODS:
        pending = []
        for fold in run.FOLDS:
            artifact = run.paths(group, fold)["audit"]
            if not artifact.exists() or h.read_json(artifact).get("audit_sha") != audit_sha:
                pending.append(fold)
        if pending:
            print(f"Independent audit {group}: {pending}", flush=True)
            execution = run.pool("audit.py", "audit-fold", pending, CFG["gpu"]["max_workers"], manifest["affinity"], group)
            manifest.setdefault("audit_runs", []).append(execution)
            h.atomic_json(run.MANIFEST, manifest)
        loaded = [h.read_json(run.paths(group, fold)["audit"]) for fold in run.FOLDS]
        h.require(all(row["status"] == "PASS" and row["audit_sha"] == audit_sha and (row["group"], row["fold"]) == (group, fold)
                      for fold, row in zip(run.FOLDS, loaded)), "Independent fold audit must cover the whole group")
        all_audits[group] = loaded
    guard = h.verify_guard(manifest)
    h.require(h.sha_file(HERE / "audit.py") == source_sha, "Auditor changed during audit")
    manifest["guard_after"] = guard
    manifest["preflight"]["single_worker_peak_reserved_mib"] = max(
        h.read_json(run.paths(group, 1)["preflight"])["peak_reserved_mib"] for group in METHODS)
    metrics, paired, per_fold = summarize(all_audits)
    decisions = scientific_decision(all_audits, paired)
    geometry = archived_geometry(all_audits, manifest)
    write_tables(all_audits, per_fold)
    status = {
        "status": "PASS", "trained": 20, "best_replayed": 20, "latest_train_replayed": 20,
        "latest_eval_replayed": 20, "rng_replayed": 20, "gradient_paths": 20, "source_and_base_protection": guard,
        "membership_and_label_scope": 20, "sklearn_metric_recalculation": 20, "science": decisions,
        "summary": metrics, "comparison": paired, "archived_geometry": geometry,
        "folds": [{k: v for k, v in row.items() if k != "patients"} for rows in all_audits.values() for row in rows],
    }
    with h.atomic_writer(RESULTS / "report.md", "w") as stream:
        stream.write(output_report(all_audits, metrics, paired, decisions, geometry, manifest))
    handoff = append_handoff(handoff_summary(all_audits, metrics, paired, decisions, manifest))
    h.atomic_json(RESULTS / "audit_status.json", {**status, "handoff": handoff})
    manifest.update(status="COMPLETE_AUDITED", scientific=decisions, handoff=handoff,
                    results=h.hash_files(sorted(RESULTS.iterdir())))
    h.atomic_json(run.MANIFEST, manifest)
    print("COMPLETE_AUDITED. Best/Latest/RNG/gradients 20/20. Handoff readback PASS.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "audit-fold"))
    parser.add_argument("fold", type=int, nargs="?", choices=list(run.FOLDS))
    parser.add_argument("group", nargs="?", choices=METHODS)
    args = parser.parse_args()
    if args.action == "all":
        with (run.RUNTIME / "scheduler.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit_all()
    elif args.fold is not None and args.group is not None:
        audit_fold(args.group, args.fold)
    else:
        parser.error("The fold and group are required")


if __name__ == "__main__":
    main()
