"""Independent checkpoint, patient, geometry, gradient and random-trajectory audit."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math

from common import history as h
from common import optimization as o
import forensic as f
import run

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)
from torch.nn import functional as F

CFG = h.CFG
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
REFERENCES = ("DARG", "E82.2-G2", "E82.3.2-G2", "E82.4.2-G2")


def metrics(prob, pred, y):
    f.equal(prob.argmax(-1), pred, "Prediction is not the original two-class argmax")
    h.require(np.isfinite(prob).all() and prob.shape == (len(y), 2), "Malformed or nonfinite prediction")
    f.bounded(prob.sum(-1), np.ones(len(y)), 6, "Two normalized class probabilities")
    tp, fn, fp, tn = (int(n) for n in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    asd = y == 0
    ba = balanced_accuracy_score(y, pred)
    h.require(abs(ba - roc_auc_score(asd, pred == 0)) <= np.finfo(np.float64).eps,
              "BA and hard-prediction binary ROC-AUC differ")
    bins = CFG["ece_equal_width_bins"]
    ix = np.minimum(np.floor(bins * prob[:, 0]).astype(int), bins - 1)
    ece = 0.0
    for i in range(bins):
        mask = ix == i
        if mask.any():
            ece += mask.sum() / len(y) * abs(prob[mask, 0].astype(np.float64).mean() - asd[mask].mean())
    return {
        "n_test": len(y), "acc": accuracy_score(y, pred), "hard_auc": ba, "prob_auc": roc_auc_score(asd, prob[:, 0]),
        "sen": recall_score(y, pred, pos_label=0), "spe": recall_score(y, pred, pos_label=1),
        "asd_f1": f1_score(y, pred, pos_label=0, zero_division=0), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "brier": brier_score_loss(asd, prob[:, 0]), "ece": float(ece),
    }


@torch.no_grad()
def independent_eval(ctx, intervention="full"):
    state, module = ctx["state"], ctx["module"]
    head = state["cam"].Outputlayer
    seen = []

    def replace_input(actual_head, arguments):
        c, att = arguments
        actual, details = module(c, actual_head, False, intervention)
        seen.append((c, actual, details))
        return actual, att

    def collect_logits(_head, _input, output):
        seen.append(output)

    a = head.register_forward_pre_hook(replace_input)
    b = head.classifier.register_forward_hook(collect_logits)
    initial = h.rng_sha()
    try:
        with run.dropout_inventory() as dropout:
            out = h.runner().forward(state, h.BASE, training=False)
    finally:
        a.remove()
        b.remove()
    h.require(len(seen) == 2 and h.rng_sha() == initial and dropout == [], "Independent inference stochastic or bypassed")
    cam, enhanced, d = seen[0]
    logits = seen[1][state["test"]].cpu().numpy()
    prob = out["logp"][state["test"]].exp().cpu().numpy()
    f.bounded(prob, softmax(logits.astype(np.float64), axis=-1), 10, "Original log-softmax and probabilities")
    pred = prob.argmax(-1)
    return {"probability": prob, "prediction": pred, "logits": logits,
            "correct": int((pred == state["labels_np"][state["test_np"]]).sum())}, {
                "cam": cam, "enhanced": enhanced, "details": d, "logits": seen[1]}


def match_eval(actual, archived, phase):
    for key, value in actual.items():
        f.equal(value, archived[key], f"{phase} {key} exact reload")


def signed_margin(logits, labels):
    return (logits[:, 0].astype(np.float64) - logits[:, 1]) * np.where(labels == 0, 1, -1)


def reference_modules(ctx, name, saved):
    if name == "DARG":
        h.restore_base(ctx["state"], saved["best_state"])
        return None
    h.restore_base(ctx["state"], saved["current"]["base"])
    fold = ctx["fold"]
    spectrum = ctx["original"]
    text = h.archive.verify_text()
    with torch.random.fork_rng(devices=[]):
        if name == "E82.2-G2":
            model = h.archive.psce().PSCE(spectrum, text, "G2", fold)
            model.load_state_dict(saved["current"]["psce"], strict=True)
        else:
            cfg = h.read_json(h.E8232 / "config.json")
            stats = h.load(h.E823 / "runtime" / f"class_stats_f{fold:02d}.pt")["stats"]
            model = h.archive.da().DASRPE(spectrum, text, stats, "G2", fold, cfg["class_statistics"], cfg["ema"]["alpha"])
            model.initialize_gate(cfg["patient_gate"])
            model.load_state_dict(saved["current"]["cspe"], strict=True)
    return model.to(ctx["state"]["x"].device).eval()


@torch.no_grad()
def archived_reference(ctx, manifest):
    state = ctx["state"]
    y, fold = state["labels_np"][state["test_np"]], ctx["fold"]
    output = {}
    for name, (path, fold_csv, historical_group, scale) in h.reference_paths(fold).items():
        h.require(h.sha_file(path) == manifest["protected_sha256"][str(path)], f"{name} Best digest differs")
        saved = h.load(path)
        if name == "E82.4.2-G2":
            h.require(saved["identity"]["identity"] == ctx["provenance"]["identity"], "E82.4.2 predecessor identity differs")
            f.equal(saved["membership"]["test"], ctx["members"]["test"], "E82.4.2 test order")
            phase = saved["evaluation"]
            prob, pred, logits, ep = phase["probability"], phase["prediction"], phase["logits"], saved["best_epoch"]
        else:
            h.require(saved["identity"] == ctx["provenance"]["identity"] and saved["fold"] == fold,
                      f"{name} original fold fingerprint differs")
            if "test_indices" in saved:
                f.equal(saved["test_indices"], ctx["members"]["test"], f"{name} test ordering")
            mod = reference_modules(ctx, name, saved)
            head = state["cam"].Outputlayer
            head_logits = []
            handles = [head.classifier.register_forward_hook(lambda _, args, value: head_logits.append(value))]
            if mod is not None:
                handles.append(head.register_forward_pre_hook(lambda _, args: (mod(args[0])[0], args[1])))
            try:
                current = h.runner().forward(state, h.BASE, training=False)
            finally:
                for b in handles:
                    b.remove()
            logits = head_logits[0][state["test"]].cpu().numpy()
            prob = current["logp"][state["test"]].exp().cpu().numpy()
            pred = prob.argmax(-1)
            if name == "DARG":
                f.equal(pred, saved["test_prediction"], "Historical DARG reloaded predictions")
                f.equal(prob, saved["test_probability"], "Historical DARG reloaded probabilities")
                ep = saved["best_epoch_1based"]
            else:
                f.equal(pred, saved["prediction"], f"{name} reloaded predictions")
                f.equal(prob, saved["probability"], f"{name} reloaded probabilities")
                ep = saved["best_epoch"]
        measured = metrics(prob, pred, y)
        with fold_csv.open(newline="") as stream:
            original, = [row for row in csv.DictReader(stream)
                         if row.get("group", row.get("method")) == historical_group and int(row["fold"]) == fold]
        for key in METRICS:
            h.require(abs(measured[key] - float(original[key]) / scale) < 1e-12,
                      f"{name} independently recalculated {key} differs in fold {fold}")
        output[name] = {"probability": prob[:, 0].tolist(), "prediction": pred.tolist(),
                        "true_margin": signed_margin(logits, y).tolist(), "metrics": measured, "epoch": ep,
                        "checkpoint_sha": manifest["protected_sha256"][str(path)],
                        "audit": "archived_output_and_metrics" if name == "E82.4.2-G2" else "exact_forward_and_metrics"}
    return output


def audit_budget_epoch(budget, maxima):
    expected = {k: ("native",) if k in o.ORIGINAL else tuple(s for s in CFG["auxiliary"] if s != "native")
                for k in (*o.ORIGINAL, *o.NEW)}
    h.require(set(budget) == set(expected), "Budget gradient family ledger differs")
    for family, values in budget.items():
        main, combined, *observations = values
        h.require(len(observations) == len(expected[family]) * 3 and np.isfinite(values).all() and min(values) >= 0,
                  f"{family} gradient ledger fields or values differ")
        total_cap = 0.0
        for term, (raw, actual, scale) in zip(expected[family], np.asarray(observations).reshape(-1, 3)):
            cap = CFG["auxiliary"][term]["cap"]
            h.require(actual <= cap * main and 0.0 <= scale <= 1.0 and (raw > 0 or actual == 0),
                      f"Applied {family}/{term} exceeded its registered gradient budget")
            tag = f"{family}.{term}"
            ratio = actual / main if main else 0
            maxima[tag] = max(maxima.get(tag, 0), float(ratio))
            total_cap += cap
        h.require(combined <= total_cap * main, f"Total {family} auxiliary gradient exceeded registered cap")
        tag = family + ".total"
        maxima[tag] = max(maxima.get(tag, 0), combined / main if main else 0)


def audit_journal(ctx, task, best, latest, preflight):
    files = run.paths(ctx["group"], ctx["fold"])
    raw = files["journal"].read_bytes()
    h.require(len(raw) == latest["journal_size"] and hashlib.sha256(raw).hexdigest() ==
              latest["journal_sha"] == task["journal_sha"], "Training ledger differs from committed prefix")
    records = [json.loads(line) for line in raw.splitlines()]
    end = latest["epoch"]
    h.require([row["epoch"] for row in records] == list(range(run.START + 1, end + 1)), "Training epoch sequence differs")
    h.require(task["epoch"] == end, "Task latest epoch differs")
    state, dropout = ctx["state"], preflight["dropout"]
    y = state["labels_np"][state["test_np"]]
    inputs = [torch.empty_strided(call["shape"], call["stride"], dtype=torch.float32,
                                  device=state["x"].device).fill_(1) for call in dropout]
    h.require(len(dropout) == 6 and all(call["dtype"] == "torch.float32" for call in dropout),
              "Historical replay Dropout inventory incomplete")
    arch = h.hist_trace(ctx["fold"])
    h.base().restore_rng(ctx["warmup"]["rng"])
    comparable = wait = 0
    maximum, chosen = -1, None
    maxima = {}
    stats = None
    stats_sha = preflight["initial_ema"]
    if ctx["group"] == "G2":
        ref = ctx["reference"]
        stats = {k: f.f64(ref[k]).copy() for k in ("mean", "second", "stable")}
        n = f.array(ref["counts"])
        h.require(best["ema"]["updates"] == best["epoch"] - run.START - 1 and
                  latest["eval_ema"]["updates"] == end - run.START - 1 and
                  latest["ema"]["updates"] == end - run.START, "EMA update count inconsistent with historical timing")
        for v in (best["ema"], latest["eval_ema"], latest["ema"]):
            f.equal(v["counts"], n, "EMA class count scope")
        h.require(h.sha(latest["eval_ema"]) == records[-1]["ema_before"], "Latest evaluation EMA is mismatched")

    def state_near(expected):
        for key in ("mean", "second", "stable"):
            f.bounded(expected[key], stats[key], 2000, f"Double precision independent EMA {key}",
                      1 + np.abs(stats[key]), f.U64)

    for record in records:
        epoch = record["epoch"]
        rng = record["rng"]
        h.require(h.rng_sha() == rng["rng_before"], f"History before epoch {epoch} cannot be replayed")
        for x, spec in zip(inputs, dropout):
            F.dropout(x, p=spec["p"], training=True, inplace=spec["inplace"])
        h.require(h.rng_sha() == rng["rng_after_train"] == rng["rng_after_eval"],
                  f"Native stochastic sequence differs at epoch {epoch}")
        if epoch in arch:
            h.require(rng == arch[epoch], f"Archived E74.4 RNG differs at epoch {epoch}")
            comparable += 1
        encoded = bytes.fromhex(record["prediction_bits"])
        pred = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))[:len(y)]
        current = int((pred == y).sum())
        h.require(current == record["correct"], f"Incorrect recomputed test ACC at epoch {epoch}")
        improved = current > maximum
        if improved:
            chosen, maximum, wait = epoch, current, 0
        else:
            wait += 1
        h.require((improved, chosen, maximum, wait) ==
                  tuple(record[k] for k in ("improved", "best_epoch", "best_correct", "wait")), "Best selection differs")
        h.require(wait <= CFG["early_stop_wait_gt"] or epoch == end, "Training overshot historical early stop")
        h.require(np.isfinite(list(record["losses"].values())).all(), "Nonfinite ledger loss")
        audit_budget_epoch(record["budget"], maxima)
        h.require(record["ema_before"] == stats_sha, "EMA ledger is discontinuous")
        if stats is not None:
            for checkpoint, checkpoint_epoch in ((best["ema"], best["epoch"]), (latest["eval_ema"], end)):
                if epoch == checkpoint_epoch:
                    state_near(checkpoint)
                    h.require(h.sha(checkpoint) == stats_sha, "Evaluation-time EMA digest differs")
            observation = np.frombuffer(base64.b64decode(record["ema_observation"], validate=True),
                                        dtype="<f8").reshape(2, 6, 2, 16)
            a = CFG["evidence"]["alpha"]
            stats["mean"] = (1 - a) * stats["mean"] + a * observation[0]
            stats["second"] = (1 - a) * stats["second"] + a * observation[1]
            stats["stable"] = f.stable_variance(stats["mean"], stats["second"], n)
        else:
            h.require(record["ema_observation"] is None and record["ema_before"] == record["ema_after"],
                      "G1 ledger has unexpected EMA")
        stats_sha = record["ema_after"]
    h.require(chosen == best["epoch"] == latest["best_epoch"] == task["best_epoch"] and
              maximum == best["evaluation"]["correct"] == latest["best_correct"] == task["best_correct"],
              "Best position differs from first strict maximum")
    h.require(wait == latest["wait"] and (wait > CFG["early_stop_wait_gt"] or end == CFG["max_epochs"]),
              "Historical termination reason differs")
    h.require(h.rng_sha() == latest["rng_sha"] and stats_sha == latest["ema_sha"], "Terminal RNG or EMA differs")
    if stats is not None:
        state_near(latest["ema"])
    for checkpoint in (best, latest):
        selected = records[checkpoint["epoch"] - run.START - 1]
        h.require(selected["rng"]["rng_after_eval"] == checkpoint["rng_sha"]
                  and selected["probability_sha"] == h.sha(checkpoint["evaluation"]["probability"])
                  and selected["checkpoint_state_sha"] == checkpoint["state_sha"], "Checkpoint commit identity differs")
    h.require(records[0]["paths"] is not None and len(records) >= 10 and records[9]["paths"] is not None,
              "First/tenth real update path evidence missing")
    return {"archived_epochs": comparable, "replayed_epochs": len(records), "rng_phase_comparisons": comparable * 3,
            "budget_maximum_ratio": maxima, "first": {"routing": records[0]["routing"], "paths": records[0]["paths"]},
            "tenth": {"routing": records[9]["routing"], "paths": records[9]["paths"]},
            "ema_independent": stats is not None, "stop": "max_epoch" if end == CFG["max_epochs"] else "early_stop"}


def native_probe(ctx, expected_dropout):
    state = ctx["state"]
    with run.dropout_inventory() as calls:
        output, cap = run.forward(state, True, ctx["module"])
    h.require(calls == expected_dropout, "Stochastic operation order differs in checkpoint")
    fp = f.branch_formulas(ctx, cap, training=True)
    head = state["cam"].Outputlayer
    norm, linear, _, _ = head.mlp_head
    cam = cap["cam"][state["legal"]]
    value = F.linear(F.gelu(F.linear(F.layer_norm(cam, norm.normalized_shape, norm.weight, norm.bias, norm.eps),
                                    linear.weight, linear.bias)), head.classifier.weight, head.classifier.bias)
    native = torch.logsumexp(value, dim=-1) - value.gather(-1, ctx["labels"][:, None])[:, 0]
    before = h.rng_sha()
    objectives, targets = o.losses(ctx["group"], head, cap, state["legal"], ctx["labels"])
    f.bounded(native.mean(), objectives["native"], 300, "Independent legal native objective")
    native_margin_np = signed_margin(f.array(value), f.array(ctx["labels"]))
    full_head = f.independent_head(head, f.array(cap["enhanced"])[ctx["members"]["legal"]])[0]
    gain = signed_margin(full_head, f.array(ctx["labels"])) - native_margin_np
    f.bounded(targets["gain"], gain, 390, "Auxiliary uses actual deterministic output margin", 1 + abs(gain))
    original = ctx["families"]
    flat = tuple(p for values in original.values() for p in values)
    live = [p for p in flat if p.requires_grad]
    independent_main = dict(zip(live, torch.autograd.grad(output["cls"], live, allow_unused=True, retain_graph=True)))
    native_grad = dict(zip(live, torch.autograd.grad(native.mean(), live, allow_unused=True, retain_graph=True)))
    class_to_delta, = torch.autograd.grad(output["cls"], cap["details"]["delta"], retain_graph=True)
    residual_gradient = torch.autograd.grad(cap["details"]["delta"], live, grad_outputs=class_to_delta,
                                            allow_unused=True, retain_graph=True)
    class_through_delta = dict(zip(live, residual_gradient))
    native_cam, = torch.autograd.grad(native.mean(), cap["cam"], retain_graph=True)
    h.require(not native_cam[state["test"]].count_nonzero().item(), "Native gradient touches held-out CAM")
    for p in flat:
        p.grad = None
    output["total"].backward(retain_graph=True)
    old = {p: None if p.grad is None else f.f64(p.grad).copy() for p in flat}
    for p in flat:
        p.grad = None
    actual = o.apply_budget(original, output, objectives)

    def norm64(items):
        return math.sqrt(sum(np.square(f.f64(x)).sum() for x in items if x is not None))

    record = {}
    for name, params in original.items():
        main = norm64(independent_main.get(p) for p in params)
        extra = norm64(f.f64(p.grad) if old[p] is None else f.f64(p.grad) - old[p]
                       for p in params if p.grad is not None)
        f.bounded(actual[name]["main"], main, 64, "Independently calculated main gradient norm", 1 + main)
        f.bounded(actual[name]["total_applied"], extra, 64, "Actual .grad addition recomputed", 1 + extra)
        cap_fraction = 0.10 if name in o.ORIGINAL else 0.15
        h.require(extra <= cap_fraction * main * (1 + f.gamma(64)) + np.finfo(np.float32).tiny,
                  f"{name} observable optimizer gradient exceeds cap")
        through = norm64(class_through_delta.get(p) for p in params)
        connected = norm64(native_grad.get(p) for p in params)
        if name in o.ORIGINAL:
            h.require(connected > 0 and extra > 0, f"Independent native route is disconnected at {name}")
        if name != "head":
            h.require(through > 0, f"Independent residual classification route is disconnected at {name}")
        if name in o.NEW:
            h.require(all(norm64([independent_main[p]]) > 0 for p in params),
                      f"Independent classification gradient misses a {name} parameter")
        record[name] = {"native_ce_norm": connected, "class_through_residual": through,
                        "full_classifier_norm": main, "actual_added_norm": extra, "individual": actual[name]}
    h.require(h.rng_sha() == before, "Independent gradient calculations consume RNG")
    return {"gradients": record, "train_formulas": fp,
            "routing": o.distributions(cap["details"], state["legal"], targets, ctx["group"])}


def scalar_response(prob, pred, truth, logits, features, full):
    corrected = (full["pred"] != truth) & (pred == truth)
    harmed = (full["pred"] == truth) & (pred != truth)
    delta = prob[:, 0].astype(np.float64) - full["prob"][:, 0]
    return {"probability_changed": int((np.abs(delta) > CFG["measurable_probability_change"]).sum()),
            "hard_flipped": int((full["pred"] != pred).sum()), "probability_mae": float(np.abs(delta).mean()),
            "absolute_logit_delta": float(np.abs(logits - full["logits"]).mean()),
            "max_logit_delta": float(np.abs(logits - full["logits"]).max()),
            "mean_feature_delta_l2": float(np.linalg.norm(features - full["residual"], axis=-1).mean()),
            "max_feature_delta_l2": float(np.linalg.norm(features - full["residual"], axis=-1).max()),
            "asd_corrected": int((corrected & (truth == 0)).sum()), "asd_harmed": int((harmed & (truth == 0)).sum()),
            "td_corrected": int((corrected & (truth == 1)).sum()), "td_harmed": int((harmed & (truth == 1)).sum()),
            "corrected": int(corrected.sum()), "harmed": int(harmed.sum()), "net": int(corrected.sum() - harmed.sum())}


def mechanism(ctx, best):
    state = ctx["state"]
    y = state["labels_np"][state["test_np"]]
    output, formulas, light, all_changes = {}, {}, {}, {}
    for key in run.INTERVENTIONS[ctx["group"]]:
        result, cap = independent_eval(ctx, key)
        formulas[key] = f.branch_formulas(ctx, cap, intervention=key)
        d, rows = cap["details"], state["test"]
        residual = f.array(d["delta"][rows])
        logits, p, pred = result["logits"], result["probability"], result["prediction"]
        if key == "full":
            match_eval(result, best["evaluation"], "Best")
            full = {"prob": p, "pred": pred, "logits": logits, "residual": residual}
            keys = ["gate", "feasibility", "capacity", "ratio"]
            if ctx["group"] == "G2":
                keys += ["evidence", "support", "conflict", "strength_parallel", "strength_perp"]
            else:
                keys += ["beta"]
            light = {name: f.array(d[name][rows]).tolist() for name in keys}
            light["parallel_ratio"] = (d["delta_parallel"][rows].norm(dim=-1) /
                                       d["cam_norm"][rows].clamp_min(1e-8)).cpu().tolist()
            light["complement_ratio"] = (d["delta_perp"][rows].norm(dim=-1) /
                                         d["cam_norm"][rows].clamp_min(1e-8)).cpu().tolist()
        output[key] = {"metrics": metrics(p, pred, y),
                       "response_vs_full": scalar_response(p, pred, y, logits, residual, full),
                       "probability": p[:, 0].tolist(), "prediction": pred.tolist(),
                       "true_margin": signed_margin(logits, y).tolist()}
        all_changes[key] = np.linalg.norm(residual - full["residual"], axis=-1).tolist()
    light["intervention_residual_difference_l2"] = all_changes
    return output, light, formulas


def check_latest(ctx, latest, manifest, task, preflight):
    run.restore(ctx, latest, manifest["implementation"])
    h.require(h.sha(run.bundle(ctx)) == latest["state_sha"], "Latest full reloaded model/optimizers identity differs")
    current = run.ema(ctx)
    run.restore_ema(ctx, latest["eval_ema"])
    result, cap = independent_eval(ctx)
    match_eval(result, latest["evaluation"], "Latest recorded evaluation")
    f.branch_formulas(ctx, cap)
    if current is not None:
        observed = f.independent_observation(ctx, cap["cam"])
        stored = ctx["module"].statistics.observation(cap["cam"])
        f.bounded(stored, observed, 4 * len(ctx["members"]["legal"]), "Latest legal EMA observation",
                  1 + abs(observed), f.U64)
        h.require(h.sha(stored) == latest["observation_sha"], "Latest EMA observation differs")
    run.restore_ema(ctx, current)
    witness = run.next_witness(ctx, latest["epoch"] + 1, preflight["dropout"], inspect=True)
    h.require(witness == task["next_witness"], "Latest independent next step differs from live terminal witness")
    arch = h.hist_trace(ctx["fold"]).get(latest["epoch"] + 1)
    h.require(arch is None or arch == witness["rng"], "Latest next step diverged from available E74.4 RNG")
    run.restore(ctx, latest, manifest["implementation"])
    diagnostic = native_probe(ctx, preflight["dropout"])
    return {"exact": True, "next_witness": witness, "metrics": metrics(
        result["probability"], result["prediction"], ctx["state"]["labels_np"][ctx["members"]["test"]]),
            "diagnostic": diagnostic}


def audit_fold(fold):
    manifest = h.read_json(run.MANIFEST)
    h.require(manifest["implementation"] == h.implementation(), "Training or audit source freeze differs")
    historic = None
    for group in h.GROUPS:
        paths = run.paths(group, fold)
        preflight, task = (h.read_json(paths[k]) for k in ("preflight", "task"))
        h.require(preflight["status"] == "PASS" and task["status"] == "TRAINED", "Fold status incomplete")
        for k in ("best", "latest"):
            h.require(h.sha_file(paths[k]) == task[k + "_sha"], f"{group} {k} artifact SHA differs")
        best, latest = (h.load(paths[k]) for k in ("best", "latest"))
        h.require(latest["best_sha"] == task["best_sha"], "Latest Best commitment differs")
        ctx = run.setup(group, fold, verify_spectrum=True)
        h.require(preflight["provenance"] == ctx["provenance"], "Independent warmup/model initialization identity differs")
        if historic is None:
            historic = archived_reference(ctx, manifest)
        archive = audit_journal(ctx, task, best, latest, preflight)
        run.restore(ctx, best, manifest["implementation"])
        h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best full reloaded model/optimizers identity differs")
        evaluated, response, formulas = mechanism(ctx, best)
        best_probe = native_probe(ctx, preflight["dropout"])
        last_probe = check_latest(ctx, latest, manifest, task, preflight)
        y = ctx["state"]["labels_np"][ctx["members"]["test"]]
        data = {"status": "PASS", "group": group, "fold": fold, "best_epoch": best["epoch"], "epoch": latest["epoch"],
                "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "scope": preflight["labels"],
                "patient_rows_1based": (ctx["members"]["test"] + 1).tolist(), "label": y.tolist(),
                "historical": historic, "interventions": evaluated, "response": response, "formula": formulas,
                "best_full_reload": True, "ledger": archive, "best_gradients": best_probe, "latest_reload": last_probe,
                "initial_rng_isolation": preflight["rng_matches"], "native_disabled_exact": True,
                "original_o": f.check_original_o(ctx), "stats": f.class_banks(ctx) if group == "G2" else None}
        h.write_json(paths["audit"], data)
        print(f"PASS independent audit {group} fold {fold:02d}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=h.FOLDS)
    audit_fold(parser.parse_args().fold)
