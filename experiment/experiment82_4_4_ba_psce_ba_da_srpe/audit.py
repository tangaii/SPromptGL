"""Independent checkpoint, patient, geometry, gradient and random-trajectory audit."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import importlib
import sys
from functools import lru_cache
from types import ModuleType

from common import history as h
from common import optimization as o
from common.evidence import fit_reference
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
REFERENCES = ("DARG", "E82.2-G2", "E82.3.2-G2", "E82.4.3-G1", "E82.4.3-G2")


@lru_cache(maxsize=1)
def parent_classes():
    """Load the committed preceding models without importing them into current training."""
    prefixes = ("common", "g1", "g2")
    selected = lambda k: k in prefixes or k.startswith("common.")
    displaced = {name: module for name, module in sys.modules.copy().items() if selected(name)}
    for name in displaced:
        del sys.modules[name]
    source = ModuleType("common.history")
    # Historical models use the committed configuration and the same immutable
    # reader interfaces. Avoid importing a second path-remapping adapter.
    source.CFG = h.read_json(h.E8243 / "config.json")
    for attr in ("E823", "E8242", "load", "require", "sha"):
        setattr(source, attr, getattr(h, attr))
    sys.modules["common.history"] = source
    sys.path.insert(0, str(h.E8243))
    try:
        return importlib.import_module("g1").PSBRSC, importlib.import_module("g2").PEDJSC
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if selected(name):
                del sys.modules[name]
        sys.modules.update(displaced)


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
    parent_manifest = h.read_json(h.E8243 / "manifest.json")
    h.require(parent_manifest["implementation_sha"] == CFG["sources"]["E82.4.3_frozen_sha"],
              "82.4.3 frozen predecessor identity differs")
    for name, (path, fold_csv, historical_group, scale) in h.reference_paths(fold).items():
        h.require(h.sha_file(path) == manifest["protected_sha256"][str(path)], f"{name} Best digest differs")
        saved = h.load(path)
        predecessor = name.startswith("E82.4.3-")
        if predecessor:
            group = name.split("-")[-1]
            h.require(saved["implementation"] == parent_manifest["implementation"] and
                      saved["provenance"]["identity"] == ctx["provenance"]["identity"] and
                      saved["fold"] == fold, "82.4.3 native source or split differs")
            f.equal(saved["membership"]["test"], ctx["members"]["test"], "82.4.3 test order")
            h.restore_base(state, saved["state"]["native"])
            text = h.archive.verify_text()
            classes = parent_classes()
            with torch.random.fork_rng(devices=[]):
                if group == "G1":
                    model = classes[0](ctx["original"], text, state["excluded"])
                else:
                    reference = fit_reference(fold, ctx["warm_cam"], ctx["original"], ctx["members"]["legal"],
                                              f.array(ctx["labels"]))
                    model = classes[1](ctx["original"], text, state["excluded"], reference)
            model.to(state["x"].device).eval()
            model.load_state_dict(saved["state"]["module"], strict=True)
            if group == "G2":
                model.statistics.restore(saved["ema"])
            prior_ctx = {**ctx, "module": model}
            last, _ = independent_eval(prior_ctx)
            match_eval(last, saved["evaluation"], name)
            off, _ = independent_eval(prior_ctx, "residual_off")
            certificate_path = h.E8243 / "runtime/audit" / f"{group}_f{fold:02d}.json"
            certificate = h.read_json(certificate_path)
            f.equal(certificate["patient_rows_1based"], ctx["members"]["test"] + 1, "82.4.3 patient ordering")
            f.equal(off["prediction"], certificate["interventions"]["residual_off"]["prediction"], "82.4.3 off prediction")
            f.equal(off["probability"][:, 0], certificate["interventions"]["residual_off"]["probability"],
                    "82.4.3 off probability")
            prob, pred, logits, ep = last["probability"], last["prediction"], last["logits"], saved["best_epoch"]
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
                for handle in handles:
                    handle.remove()
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
                        "checkpoint_sha": manifest["protected_sha256"][str(path)], "audit": "exact_forward_and_metrics"}
        if predecessor:
            output[name].update(off_prediction=off["prediction"].tolist(),
                                off_true_margin=signed_margin(off["logits"], y).tolist())
    return output


def audit_budget_epoch(group, budget, maxima, counts):
    h.require(set(budget) == {*o.ORIGINAL, *o.NEW}, "Budget gradient family ledger differs")
    for family, values in budget.items():
        h.require(np.isfinite(values).all(), f"{family} includes a nonfinite budget")
        if family in o.ORIGINAL:
            h.require(len(values) == len(o.ORIGINAL_LOG), "Historical gradient ledger fields differ")
            data = dict(zip(o.ORIGINAL_LOG, values))
            main, added, raw, projected = (data[k] for k in ("main", "total_applied", "raw", "projected"))
            cap = CFG["full_classification"]["shared_cap"]
            h.require(min(main, added, raw, projected) >= 0 and added <= cap * main and
                      data["cos_applied"] >= 0 and 0 <= data["scale"] <= 1, "Realized shared budget or direction differs")
            h.require(projected <= raw * (1 + f.gamma(128, f.U64)) and
                      data["cos_projected"] >= -f.gamma(128, f.U64), "Conflict projection differs")
            stats = counts.setdefault(family, {"epochs": 0, "conflict": 0, "applied": 0, "capped": 0,
                                              "rounded_to_zero": 0, "sum_cos_before": 0., "sum_cos_projected": 0.,
                                              "sum_cos_applied": 0., "sum_scale": 0.})
            stats["epochs"] += 1
            stats["conflict"] += int(data["conflict"])
            stats["applied"] += int(added > 0)
            stats["capped"] += int(projected > cap * main)
            stats["rounded_to_zero"] += int(data["rounded_to_zero"])
            for name in ("cos_before", "cos_projected", "cos_applied", "scale"):
                stats["sum_" + name] += data[name]
            maxima[family + ".full"] = max(maxima.get(family + ".full", 0), added / main if main else 0)
            continue
        main, combined, *obs = values
        terms = o.auxiliary_terms(group, family)
        h.require(min(values) >= 0 and len(obs) == 3 * len(terms), "New parameter ledger fields differ")
        limit = 0.
        for term, (raw, actual, scale) in zip(terms, np.reshape(obs, (-1, 3))):
            cap = CFG["auxiliary"][term]["cap"]
            h.require(actual <= cap * main and 0 <= scale <= 1 and (raw > 0 or actual == 0),
                      f"Applied {family}/{term} exceeded its registered gradient budget")
            tag = f"{family}.{term}"
            maxima[tag] = max(maxima.get(tag, 0), float(actual / main) if main else 0)
            limit += cap
        h.require(combined <= limit * main, f"Total {family} auxiliary gradient exceeded budget")
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
    maxima, routing = {}, {}
    ema_exact = 0
    ema_envelope = {}
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
            scale = 1 + np.abs(stats[key]) + np.abs(stats["second"]) + np.abs(stats["mean"]) ** 2
            ema_envelope[key] = f.bounded(expected[key], stats[key], 4 * len(records) + 100,
                                          f"Double precision independent EMA {key}", scale, f.U64)

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
        audit_budget_epoch(ctx["group"], record["budget"], maxima, routing)
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
            independent = {k: torch.from_numpy(v) for k, v in stats.items()}
            independent.update(counts=ctx["reference"]["counts"], updates=epoch - run.START)
            ema_exact += int(h.sha(independent) == record["ema_after"])
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
            "budget_maximum_ratio": maxima, "projection_routing": routing, "first": {"routing": records[0]["routing"], "paths": records[0]["paths"]},
            "tenth": {"routing": records[9]["routing"], "paths": records[9]["paths"]},
            "ema_independent": stats is not None, "ema_bitwise_epochs": ema_exact,
            "ema_error_bounds": ema_envelope, "stop": "max_epoch" if end == CFG["max_epochs"] else "early_stop"}


def twin_probe(ctx, expected_dropout):
    state, group, groups = (ctx[k] for k in ("state", "group", "families"))
    with run.dropout_inventory() as calls:
        output, cap = run.forward(state, True, ctx["module"])
    h.require(calls == expected_dropout, "Stochastic operation order differs in checkpoint")
    before = h.rng_sha()
    fp = f.branch_formulas(ctx, cap, training=True)
    head, legal = state["cam"].Outputlayer, state["legal"]
    basis = ctx["module"].spectrum.basis.to(cap["cam"])
    y = ctx["labels"]
    objectives, targets = o.losses(group, head, cap, legal, y, basis)
    norm, first, _, _ = head.mlp_head
    cam, enhanced = cap["cam"][legal], cap["enhanced"][legal]
    pre_activation = F.linear(F.layer_norm(enhanced, (144,), norm.weight, norm.bias, norm.eps),
                              first.weight, first.bias)
    value = F.linear(F.gelu(pre_activation), head.classifier.weight, head.classifier.bias)
    nll = -value.log_softmax(-1).gather(1, y[:, None]).flatten()
    balanced_ce = (nll[y == 0].mean() + nll[y == 1].mean()) / 2
    f.equal(objectives["full"], balanced_ce, "Legal Full CE using the true deterministic original head")

    labels = f.array(y)
    fulllog, _, fulscale, _ = f.independent_head(head, enhanced)
    offlog, _, offscale, _ = f.independent_head(head, cam)
    off, true_full = signed_margin(offlog, labels), signed_margin(fulllog, labels)
    gain = true_full - off
    f.from_error(targets["off"], off, f.gamma(390) * offscale.sum(-1), "Native true class margin")
    margin_budget = f.gamma(390) * (fulscale + offscale).sum(-1)
    margin_budget += f.gamma(4) * (abs(off) + abs(true_full) + abs(gain))
    f.from_error(targets["gain"], gain, margin_budget, "Sign and scale of true output margin gain")
    base_margin = f.f64(targets["off"])
    observed_gain = f.f64(targets["gain"])
    focus = 1 / (1 + np.exp(np.clip(base_margin / CFG["boundary"]["error_temperature"], -700, 700)))
    protect = (1 - focus) * np.exp(-abs(base_margin) / CFG["boundary"]["protect_scale"])
    def balanced_mean(x):
        return (x[labels == 0].mean() + x[labels == 1].mean()) / 2
    loss = np.logaddexp(0, -observed_gain / CFG["boundary"]["gain_temperature"])
    f.bounded(objectives["gain"], balanced_mean(focus * loss), 100, "Actual gain objective improves true margin")
    f.bounded(objectives["guard"], balanced_mean(protect * loss), 100, "Protection penalizes adverse true margin")
    l2ratio = f.f64(cap["details"]["ratio"][legal])
    f.bounded(objectives["regularization"], balanced_mean(l2ratio ** 2), 100, "Class-balanced residual regularization")

    params = [p for ps in groups.values() for p in ps]
    active = [p for p in params if p.requires_grad]
    native = dict(zip(active, torch.autograd.grad(output["cls"], active, allow_unused=True, retain_graph=True)))
    full = dict(zip(active, torch.autograd.grad(balanced_ce, active, allow_unused=True, retain_graph=True)))
    derivative, = torch.autograd.grad(balanced_ce, cap["details"]["delta"], retain_graph=True)
    through = torch.autograd.grad(cap["details"]["delta"], active, grad_outputs=derivative,
                                  allow_unused=True, retain_graph=True)
    routes = dict(zip(active, through))
    feature, = torch.autograd.grad(balanced_ce, cap["cam"], retain_graph=True)
    h.require(not feature[state["test"]].count_nonzero().item(), "Full CE has a held-out CAM derivative")
    component_gradients = []
    details = cap["details"]
    for key in ("raw_parallel", "raw_perp"):
        p = details[key]
        if p.requires_grad:
            primitive, = torch.autograd.grad(balanced_ce, p, retain_graph=True)
            component_gradients.append(float((primitive.detach() * p.detach()).sum(-1).norm()))
    h.require(all(v > 0 for v in component_gradients), "Space-specific class gradient missing")

    for p in params:
        p.grad = None
    output["total"].backward(retain_graph=True)
    old = {p: None if p.grad is None else f.f64(p.grad).copy() for p in params}
    for p in params:
        p.grad = None
    actual = o.apply_budget(group, groups, output, objectives)

    def flat(ps, getter):
        return np.concatenate([np.zeros(p.numel()) if (v := getter(p)) is None else f.f64(v).ravel() for p in ps])

    records = {}
    shared = CFG["full_classification"]
    for name, ps in groups.items():
        main_raw = flat(ps, full.get)
        main_size = np.linalg.norm(main_raw)
        current = flat(ps, lambda p: p.grad)
        route = np.linalg.norm(flat(ps, routes.get))
        if name in o.ORIGINAL:
            base, nat = flat(ps, old.get), flat(ps, native.get)
            n = np.linalg.norm(nat)
            added, cls = current - base, shared["shared_coefficient"] * main_raw
            unit = nat / n
            proj = cls - min(cls @ unit, 0.0) * unit
            h.require(n > 0 and main_size > 0, f"{name} native or full class derivative missing")
            f.bounded(actual[name]["main"], n, 2 * len(nat) + 8, "Independent native classification gradient norm",
                      n, f.U64)
            f.bounded(actual[name]["projected"], np.linalg.norm(proj), 2 * len(nat) + 8,
                      "Independent conflict projection norm", np.linalg.norm(cls), f.U64)
            # A rounded optimizer gradient is compared to the original historical total.
            applied = np.linalg.norm(added)
            h.require(applied > 0 and applied <= shared["shared_cap"] * n * (1 + f.gamma(2 * len(nat), f.U64)),
                      f"{name} Full CE did not reach the optimizer within its shared budget")
            h.require(added @ unit >= -f.gamma(2 * len(nat), f.U64) * (abs(added) @ abs(unit)),
                      f"{name} realized shared contribution opposes the native classification gradient")
            expected = actual[name]["scale"] * proj
            round_budget = f.U32 * np.linalg.norm(abs(base) + abs(expected))
            round_budget += 5 * f.U32 * (abs(unit) @ (abs(base) + abs(expected)))
            round_budget += f.gamma(4 * len(nat), f.U64) * (np.linalg.norm(cls) + n)
            h.require(np.linalg.norm(added - expected) <= round_budget,
                      f"{name} original historical total gradient was changed outside the projected Full addend")
            f.bounded(actual[name]["total_applied"], applied, 2 * len(nat) + 8, "Realized historical addend", applied, f.U64)
        else:
            applied = np.linalg.norm(current - main_raw * shared["new_coefficient"])
            total_limit = sum(CFG["auxiliary"][k]["cap"] for k in o.auxiliary_terms(group, name))
            h.require(applied <= total_limit * main_size * (1 + f.gamma(2 * len(main_raw), f.U64)),
                      f"{name} auxiliary objective overwhelms Full CE")
            h.require(np.linalg.norm(current) > (1 - total_limit) * main_size and
                      all(full[p] is not None and f.f64(full[p]).any() for p in ps), "A new parameter misses Full CE")
        h.require(name == "head" or route > 0, f"{name} is disconnected from the real residual gradient")
        records[name] = {"full_class_gradient": float(main_size), "class_through_delta": float(route),
                         "actual_add_norm": float(applied), "budget": actual[name]}
    optimizers = [state[k] for k in h.OPTS] + [ctx["optimizer"]]
    changed_from = {k: h.sha(ps) for k, ps in groups.items()}
    seen = []
    def gradients_at_step(opt, *_):
        return h.sha([p.grad for pg in opt.param_groups for p in pg["params"]])
    def track_step(opt, *_):
        seen.append(gradients_at_step(opt))
    expected = [gradients_at_step(opt) for opt in optimizers]
    for opt in optimizers:
        hook = opt.register_step_pre_hook(track_step)
        try:
            opt.step()
        finally:
            hook.remove()
    h.require(seen == expected and h.rng_sha() == before, "Optimizer received different gradients or consumed RNG")
    h.require(all(h.sha(ps) != changed_from[k] for k, ps in groups.items()), "A family did not update at optimizer step")
    return {"gradients": records, "train_formulas": fp, "optimizer_receipt_exact": True,
            "component_gradient": component_gradients,
            "routing": o.distributions(details, legal, targets, group)}


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


@torch.no_grad()
def per_direction(ctx, cap, labels):
    d = cap["details"]
    state = ctx["state"]
    indices = state["test"]
    c = cap["cam"][indices]
    basis = ctx["module"].spectrum.basis.to(c)
    if ctx["group"] == "G1":
        coordinates = d["delta"][indices] @ basis
        features = c[:, None, :] + coordinates.unsqueeze(-1) * basis.T
        head = state["cam"].Outputlayer
        normalized = head.mlp_head[0](features.flatten(0, 1))
        logits = head.classifier(head.mlp_head[2](head.mlp_head[1](normalized))).reshape(len(c), -1, 2)
        signed = (f.f64(logits[..., 0]) - f.f64(logits[..., 1])) * (1 - 2 * labels[:, None])
        off = head.classifier(head.mlp_head[2](head.mlp_head[1](head.mlp_head[0](c))))
        linear = f.f64(coordinates) * f.f64(d["jacobian_axis"][indices]) * (1 - 2 * labels[:, None])
        shift = signed - signed_margin(f.array(off), labels)[:, None]
        result = {"direction_mean_linear_true_margin": linear.mean(0).tolist(),
                  "direction_mean_isolated_true_margin": shift.mean(0).tolist(),
                  "direction_mean_abs_isolated_true_margin": abs(shift).mean(0).tolist()}
        for key in ("action", "w_sem", "w_sp"):
            values = f.f64(d[key][indices])
            result[key + "_direction_mean"] = values.mean(0).tolist()
            result[key + "_direction_sd"] = values.std(0).tolist()
        attention = f.f64(d["attention"][indices])
        result["query_mean_attention"] = attention.mean(0).tolist()
        result["direction_action_positive_share"] = (f.f64(d["action"][indices]) > 0).mean(0).tolist()
        return result
    llr = f.f64(d["llr"][indices])
    reliability = f.f64(d["reliability"][indices])
    evidence = f.f64(d["evidence"][indices])
    jac = f.f64(d["jacobian"][indices])
    projected_margin = (f.f64(d["delta"][indices]) * jac).sum(-1)
    signs = np.sign(evidence)
    result = {"llr_axis_mean": llr.mean(0).tolist(), "llr_axis_sd": llr.std(0).tolist(),
              "reliability_axis_mean": reliability.mean(0).tolist(),
              "evidence_agrees_linear_share": float((np.sign(projected_margin) * signs > 0).mean()),
              "evidence_opposes_linear_share": float((np.sign(projected_margin) * signs < 0).mean())}
    for key in ("strength_parallel", "strength_perp"):
        action = f.f64(d[key][indices])
        result[key + "_agree_evidence_share"] = float((np.sign(action) * signs > 0).mean())
        result[key + "_oppose_evidence_share"] = float((np.sign(action) * signs < 0).mean())
    return result


def mechanism(ctx, best):
    state = ctx["state"]
    y = state["labels_np"][state["test_np"]]
    output, formulas, light, all_changes, direction = {}, {}, {}, {}, {}
    for key in run.INTERVENTIONS[ctx["group"]]:
        result, cap = independent_eval(ctx, key)
        formulas[key] = f.branch_formulas(ctx, cap, intervention=key)
        d, rows = cap["details"], state["test"]
        residual = f.array(d["delta"][rows])
        logits, p, pred = result["logits"], result["probability"], result["prediction"]
        if key == "full":
            match_eval(result, best["evaluation"], "Best")
            full = {"prob": p, "pred": pred, "logits": logits, "residual": residual}
            keys = ["gate", "feasibility", "capacity", "ratio", "raw_ratio", "margin"]
            if ctx["group"] == "G2":
                keys += ["evidence", "support", "conflict", "strength_parallel", "strength_perp"]
            else:
                keys += ["beta"]
            light = {name: f.array(d[name][rows]).tolist() for name in keys}
            light["bound_shrink"] = (1 - d["scale"][rows, 0]).cpu().tolist()
            for name, part in (("parallel", "delta_parallel"), ("complement", "delta_perp")):
                light[name + "_ratio"] = (d[part][rows].norm(dim=-1) /
                                          d["cam_norm"][rows].clamp_min(1e-8)).cpu().tolist()
            direction = per_direction(ctx, cap, y)
        if key == "residual_off":
            direct, native_cap = run.forward(state, False)
            f.equal(direct["logp"][rows].exp(), p, "Residual-off native evaluation probability")
            f.equal(native_cap["logits"][rows], logits, "Residual-off original head")
            f.equal(native_cap["cam"], cap["enhanced"], "Residual-off exact original CAM144")
        output[key] = {"metrics": metrics(p, pred, y), "response_vs_full": scalar_response(p, pred, y, logits, residual, full),
                       "probability": p[:, 0].tolist(), "prediction": pred.tolist(),
                       "true_margin": signed_margin(logits, y).tolist()}
        all_changes[key] = np.linalg.norm(residual - full["residual"], axis=-1).tolist()
    if ctx["group"] == "G2":
        base = np.asarray(output["residual_off"]["true_margin"])
        for name in ("spectral_only", "complement_only"):
            gain = np.asarray(output[name]["true_margin"]) - base
            wrong = np.asarray(output["residual_off"]["prediction"]) != y
            direction[name + "_mean_gain"] = float(gain.mean())
            direction[name + "_wrong_mean_gain"] = float(gain[wrong].mean()) if wrong.any() else 0.
    light["intervention_residual_difference_l2"] = all_changes
    return output, light, formulas, direction


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
    run.restore(ctx, h.load(run.paths(ctx["group"], ctx["fold"])["latest"]), manifest["implementation"])
    diagnostic = twin_probe(ctx, preflight["dropout"])
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
        evaluated, response, formulas, direction = mechanism(ctx, best)
        best_probe = twin_probe(ctx, preflight["dropout"])
        latest_ctx = run.setup(group, fold)
        h.require(ctx["provenance"] == latest_ctx["provenance"], "Fresh Latest object identity differs")
        last_probe = check_latest(latest_ctx, latest, manifest, task, preflight)
        del latest_ctx
        y = ctx["state"]["labels_np"][ctx["members"]["test"]]
        data = {"status": "PASS", "group": group, "fold": fold, "best_epoch": best["epoch"], "epoch": latest["epoch"],
                "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "scope": preflight["labels"],
                "patient_rows_1based": (ctx["members"]["test"] + 1).tolist(), "label": y.tolist(),
                "historical": historic, "interventions": evaluated, "response": response, "formula": formulas,
                "directions": direction,
                "best_full_reload": True, "ledger": archive, "best_gradients": best_probe, "latest_reload": last_probe,
                "initial_rng_isolation": preflight["rng_matches"], "native_disabled_exact": True,
                "original_o": f.check_original_o(ctx), "stats": f.class_banks(ctx) if group == "G2" else None}
        h.write_json(paths["audit"], data)
        print(f"PASS independent audit {group} fold {fold:02d}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=h.FOLDS)
    audit_fold(parser.parse_args().fold)
