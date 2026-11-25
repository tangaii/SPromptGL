"""Independent checkpoint, patient, geometry, gradient and random-trajectory audit."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from copy import deepcopy
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
REFERENCES = ("DARG", "E82.2-G2", "E82.3.2-G2", "E82.4.5-G1", "E82.4.5-G2")


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
    source.CFG = h.read_json(h.E8245 / "config.json")
    for attr in ("E823", "E8242", "load", "require", "sha"):
        setattr(source, attr, getattr(h, attr))
    sys.modules["common.history"] = source
    sys.path.insert(0, str(h.E8245))
    try:
        return importlib.import_module("g1").ECPSCE, importlib.import_module("g2").ECDASRPE
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
    parent_manifest = h.read_json(h.E8245 / "manifest.json")
    h.require(parent_manifest["implementation_sha"] == CFG["sources"]["E82_4_5_frozen_sha"],
              "82.4.5 frozen predecessor identity differs")
    for name, (path, fold_csv, historical_group, scale) in h.reference_paths(fold).items():
        h.require(h.sha_file(path) == manifest["protected_sha256"][str(path)], f"{name} Best digest differs")
        saved = h.load(path)
        predecessor = name.startswith("E82.4.5-")
        if predecessor:
            group = name.split("-")[-1]
            h.require(saved["implementation"] == parent_manifest["implementation"] and
                      saved["provenance"]["identity"] == ctx["provenance"]["identity"] and
                      saved["fold"] == fold, "82.4.5 native source or split differs")
            f.equal(saved["membership"]["test"], ctx["members"]["test"], "82.4.5 test order")
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
            certificate_path = h.E8245 / "runtime/audit" / f"{group}_f{fold:02d}.json"
            certificate = h.read_json(certificate_path)
            f.equal(certificate["patient_rows_1based"], ctx["members"]["test"] + 1, "82.4.5 patient ordering")
            f.equal(off["prediction"], certificate["interventions"]["residual_off"]["prediction"], "82.4.5 off prediction")
            f.equal(off["probability"][:, 0], certificate["interventions"]["residual_off"]["probability"],
                    "82.4.5 off probability")
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


def audit_budget_epoch(group, budget, maxima, counts, dimensions):
    h.require(set(budget) == {*o.ORIGINAL, *o.NEW}, "Budget gradient family ledger differs")
    for family, values in budget.items():
        h.require(np.isfinite(values).all(), f"{family} includes a nonfinite budget")
        if family in o.ORIGINAL:
            h.require(len(values) == len(o.ORIGINAL_LOG), "Historical gradient ledger fields differ")
            d = dict(zip(o.ORIGINAL_LOG, values))
            old, raw, tangent, add = (d[key] for key in ("historical", "raw", "projected", "total_applied"))
            cap = CFG["full_classification"]["shared_cap"] * old
            safe = old > CFG["full_classification"]["historical_norm_floor"]
            rho = f.gamma(2 * dimensions[family] + 16, f.U64)
            numerical_scale = rho * (raw + tangent + old)
            h.require(min(old, raw, tangent, add, d["dot_bound"], d["update_norm"]) >= 0 and
                      add <= cap and abs(d["applied_dot"]) <= d["dot_bound"] and 0 <= d["scale"] <= 1,
                      "Realized gradient exceeds the tangent budget")
            envelope = f.U32 * (old + cap) * (1 + 4 * rho) + 8 * numerical_scale
            h.require(d["dot_bound"] <= envelope and abs(d["raw_dot"]) <= raw * (1 + 6 * rho)
                      and abs(d["tangent_dot"]) <= 8 * numerical_scale,
                      "Original-scale tangent projection error exceeds floating-point bound")
            projected_squared = max(0., raw * raw - d["raw_dot"] * d["raw_dot"])
            h.require(abs(tangent * tangent - projected_squared) <= 24 * rho * raw * raw,
                      "Projected squared norm does not match independent orthogonal geometry")
            h.require(bool(d["near_zero"]) == (not safe) and (safe or add == 0.)
                      and (not d["guard_reject"] or add == 0.) and (not d["nonfinite_difference"] or add == 0.),
                      "Invalid tangent skip flag")
            h.require(abs(d["update_dot_historical"]) <= d["update_norm"] * (1 + 8 * rho)
                      and abs(d["update_dot_extra"]) <= d["update_norm"] * add * (1 + 8 * rho),
                      "Parameter update ledger dot product violates Cauchy bound")
            stats = counts.setdefault(family, dict.fromkeys(
                ("epochs", "applied", "capped", "near_zero", "rounded_to_zero", "guard_reject",
                 "nonfinite_difference", "raw_norm", "tangent_norm", "applied_norm", "hist_norm", "applied_abs_dot", "update_norm",
                 "update_dot_historical", "update_dot_extra"), 0.))
            for name, value in (
                ("epochs", 1), ("applied", int(add > 0)), ("capped", int(tangent > cap)), ("near_zero", d["near_zero"]),
                ("rounded_to_zero", d["rounded_to_zero"]), ("guard_reject", d["guard_reject"]),
                ("nonfinite_difference", d["nonfinite_difference"]), ("raw_norm", raw),
                ("tangent_norm", tangent), ("applied_norm", add), ("hist_norm", old),
                ("applied_abs_dot", abs(d["applied_dot"])), ("update_norm", d["update_norm"]),
                ("update_dot_historical", d["update_dot_historical"]), ("update_dot_extra", d["update_dot_extra"])):
                stats[name] += value
            for key, ratio in (("full", add / old if old else 0.),
                               ("abs_dot_utilization", abs(d["applied_dot"]) / d["dot_bound"] if d["dot_bound"] else 0.)):
                tag = family + "." + key
                maxima[tag] = max(maxima.get(tag, 0.), ratio)
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
    dimensions = {k: sum(p.numel() for p in ps) for k, ps in ctx["families"].items()}
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
        losses = record["losses"]
        h.require(np.float32(losses["full"]) - np.float32(losses["off"]) == losses["difference"],
                  "Stored differential deviates from the two equal-head classification losses")
        if epoch == run.START + 1:
            h.require(losses["difference"] == 0 and all(
                dict(zip(o.ORIGINAL_LOG, record["budget"][family]))["raw"] == 0 and
                dict(zip(o.ORIGINAL_LOG, record["budget"][family]))["total_applied"] == 0
                for family in o.ORIGINAL), "Zero initial CAM residual changed historical gradients")
        audit_budget_epoch(ctx["group"], record["budget"], maxima, routing, dimensions)
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
    best_base = best["latest_base_epoch"]
    lines = raw.splitlines(keepends=True)
    b0 = b"".join(lines[:best_base - run.START])
    suffix = b"".join(lines[best_base - run.START:best["epoch"] - run.START])
    h.require(hashlib.sha256(b0).hexdigest() == best["latest_base_sha"] and suffix == best["journal_suffix"]
              and hashlib.sha256(b0 + suffix).hexdigest() == best["journal_sha"],
              "Independent Best journal transaction verification failed")
    h.require(records[0]["paths"] is not None and len(records) >= 10 and records[9]["paths"] is not None,
              "First/tenth real update path evidence missing")
    return {"archived_epochs": comparable, "replayed_epochs": len(records), "rng_phase_comparisons": comparable * 3,
            "budget_maximum_ratio": maxima, "projection_routing": routing, "first": {"routing": records[0]["routing"], "paths": records[0]["paths"]},
            "tenth": {"routing": records[9]["routing"], "paths": records[9]["paths"]},
            "ema_independent": stats is not None, "ema_bitwise_epochs": ema_exact,
            "ema_error_bounds": ema_envelope, "stop": "max_epoch" if end == CFG["max_epochs"] else "early_stop"}


def independent_candidate(head, cam, raw, labels, off, off_scale, observed):
    cam, raw = f.f64(cam), f.f64(raw)
    cap = CFG["residual"]["fraction"] * np.linalg.norm(cam, axis=-1)
    safe = np.maximum(cap, CFG["spectrum"]["eps"])
    ratio = np.linalg.norm(raw, axis=-1) / safe
    shrink = np.divide(np.tanh(ratio), ratio, out=np.ones_like(ratio), where=ratio != 0) * cap / safe
    c = cam + raw * shrink[:, None]
    logits, jac, scale, _ = f.independent_head(head, c)
    signed = signed_margin(logits, labels)
    gain = signed - off
    limit = f.gamma(390) * (scale + off_scale).sum(-1)
    limit += f.gamma(6) * (abs(signed) + abs(off) + abs(gain))
    limit += f.gamma(320) * np.sum(abs(jac) * (abs(cam) + abs(raw)), axis=-1)
    return f.from_error(observed, gain, limit, "Legal single-space actual-head true Margin gain")


def native_optimizer_steps(optimizers, historical):
    outputs = {}
    for opt in optimizers:
        copy = deepcopy(opt)
        for group, reference in zip(opt.param_groups, copy.param_groups):
            for p, q in zip(group["params"], reference["params"]):
                grad = historical.get(p)
                q.grad = None if grad is None else grad.detach().clone()
        copy.step()
        for group, reference in zip(opt.param_groups, copy.param_groups):
            for p, q in zip(group["params"], reference["params"]):
                outputs[p] = f.f64(q).copy()
    return outputs


def independent_head_ce(head, cams, labels):
    layer_norm, first, _, _ = head.mlp_head
    normalized = F.layer_norm(cams, (144,), layer_norm.weight, layer_norm.bias, layer_norm.eps)
    logits = F.linear(F.gelu(F.linear(normalized, first.weight, first.bias)),
                      head.classifier.weight, head.classifier.bias)
    nll = -logits.log_softmax(-1).gather(1, labels[:, None]).flatten()
    return (nll[labels == 0].mean() + nll[labels == 1].mean()) / 2


def twin_probe(ctx, expected_dropout, *, check_each=False):
    state, group, groups = (ctx[k] for k in ("state", "group", "families"))
    with run.dropout_inventory() as calls:
        output, cap = run.forward(state, True, ctx["module"])
    h.require(calls == expected_dropout, "Stochastic operation order differs in checkpoint")
    before = h.rng_sha()
    fp = f.branch_formulas(ctx, cap, training=True)
    head, legal = state["cam"].Outputlayer, state["legal"]
    details = cap["details"]
    basis = ctx["module"].spectrum.basis.to(cap["cam"])
    y = ctx["labels"]
    objectives, targets = o.losses(group, head, cap, legal, y, basis, ctx["hard_initial"])
    cam, enhanced = cap["cam"][legal], cap["enhanced"][legal]
    balanced_ce = independent_head_ce(head, enhanced, y)
    off_ce = independent_head_ce(head, cam, y)
    f.equal(objectives["full"], balanced_ce, "Legal Full CE using the true deterministic original head")
    f.equal(objectives["off"], off_ce, "Legal Off CE using the same true deterministic original head")
    f.equal(objectives["difference"], balanced_ce - off_ce, "Legal residual-differential loss")
    hist_loss = output["cls"] + output["graph"] + 0.05 * output["pair"] + 0.01 * output["adv"] + 0.01 * output["preserve"]
    f.equal(output["total"], hist_loss, "The complete, weighted historical loss")

    labels = f.array(y)
    fulllog, _, fulscale, _ = f.independent_head(head, enhanced)
    offlog, _, offscale, _ = f.independent_head(head, cam)
    off, true_full = signed_margin(offlog, labels), signed_margin(fulllog, labels)
    gain = true_full - off
    f.from_error(targets["off"], off, f.gamma(390) * offscale.sum(-1), "Native true class margin")
    margin_budget = f.gamma(390) * (fulscale + offscale).sum(-1)
    margin_budget += f.gamma(4) * (abs(off) + abs(true_full) + abs(gain))
    f.from_error(targets["gain"], gain, margin_budget, "Sign and scale of true output margin gain")
    base_margin, observed_gain = f.f64(targets["off"]), f.f64(targets["gain"])
    now = 1 / (1 + np.exp(np.clip(base_margin / CFG["boundary"]["error_temperature"], -700, 700)))
    focus = 1 - (1-now) * (1-CFG["boundary"]["initial_memory_mix"] * f.f64(ctx["hard_initial"]))
    f.bounded(targets["focus"], focus, 50, "Current and epoch 10 legal focus")
    protect = (1 - now) * np.exp(-abs(base_margin) / CFG["boundary"]["protect_scale"])

    def balanced_mean(rows):
        return (rows[labels == 0].mean() + rows[labels == 1].mean()) / 2

    tau = CFG["boundary"]["gain_temperature"]
    joint = np.logaddexp(0, -observed_gain / tau)
    expected_gain = joint
    components = {}
    if group == "G2":
        w = CFG["dual_gain_weights"]
        expected_gain = w["joint"] * joint
        for name, source in (("spectral", "raw_parallel"), ("complement", "raw_perp")):
            observed = targets["candidate_gains"][name]
            components[name] = independent_candidate(head, cam, details[source][legal], labels, off, offscale, observed)
            expected_gain = expected_gain + w[name] * np.logaddexp(0, -f.f64(observed) / tau)
        support = f.f64(details["support"][legal])
        uncertainty = 1 - support / ((support + .1) * (1 + f.f64(details["disagreement"][legal])))
        need = now * (1 + (1 - 2 * labels) * f.f64(details["direction"][legal])) / 2 * (1 - uncertainty)
        trust_logits = f.f64(details["calibration_logits"][legal, 1])
        target = np.logaddexp(0, -((1 - 2 * labels) * f.f64(details["target_margin"][legal])))
        calibration = CFG["margin_calibration"]
        expected_guidance = focus * target + calibration["reliability_aux_weight"] * (
            np.logaddexp(0, trust_logits) - need * trust_logits)
        regularization_extra = calibration["regularization_ratio"] * uncertainty * (
            f.f64(details["desired_change"][legal]) / calibration["shift_cap"])**2
    else:
        compatibility = np.maximum((1 - 2 * labels)[:, None] * f.f64(details["calibrated_contrast"][legal]), 0)
        signed_axis = (f.f64(details["delta"][legal]) @ f.f64(basis)) * f.f64(details["jacobian_axis"][legal])
        signed_axis *= (1 - 2 * labels)[:, None]
        expected_guidance = (compatibility * np.logaddexp(0, -signed_axis / tau)).mean(-1)
        regularization_extra = 0.
    f.bounded(objectives["guidance"], balanced_mean(expected_guidance), 250,
              "Legal label-compatible semantic or calibrated target supervision")
    f.bounded(objectives["gain"], balanced_mean(focus * expected_gain), 100, "Actual gain objective")
    f.bounded(objectives["guard"], balanced_mean(protect * joint), 100, "Protection penalizes adverse true margin")
    f.bounded(objectives["regularization"], balanced_mean(f.f64(details["ratio"][legal]) ** 2 + regularization_extra),
              100, "Class-balanced residual regularization")

    params = [p for ps in groups.values() for p in ps]
    active = [p for p in params if p.requires_grad]
    full = dict(zip(active, torch.autograd.grad(balanced_ce, active, allow_unused=True, retain_graph=True)))
    native_active = [p for name in o.ORIGINAL for p in groups[name] if p.requires_grad]
    off_gradient = dict(zip(native_active, torch.autograd.grad(off_ce, native_active, allow_unused=True, retain_graph=True)))
    differential = dict(zip(native_active, torch.autograd.grad(
        balanced_ce - off_ce, native_active, allow_unused=True, retain_graph=True)))
    historical = dict(zip(native_active, torch.autograd.grad(hist_loss, native_active, allow_unused=True, retain_graph=True)))
    derivative, = torch.autograd.grad(balanced_ce, details["delta"], retain_graph=True)
    through = torch.autograd.grad(details["delta"], active, grad_outputs=derivative, allow_unused=True, retain_graph=True)
    routes = dict(zip(active, through))
    feature, = torch.autograd.grad(balanced_ce, cap["cam"], retain_graph=True)
    h.require(not feature[state["test"]].count_nonzero().item(), "Full CE has a held-out CAM derivative")
    component_gradients = []
    for key in ("raw_parallel", "raw_perp"):
        part = details[key]
        if part.requires_grad:
            primitive, = torch.autograd.grad(balanced_ce, part, retain_graph=True)
            component_gradients.append(float((primitive.detach() * part.detach()).sum(-1).norm()))
    h.require(len(component_gradients) == (1 if group == "G1" else 2)
              and all(v > 0 for v in component_gradients), "Space-specific Full CE gradient missing")

    for p in params:
        p.grad = None
    plan, updates = o.apply_budget(group, groups, output, objectives)

    def flat(ps, getter):
        return np.concatenate([np.zeros(p.numel()) if (v := getter(p)) is None else f.f64(v).ravel() for p in ps])

    records, routes_to_update = {}, {}
    shared = CFG["full_classification"]
    for name, ps in groups.items():
        raw = flat(ps, full.get)
        raw_size = np.linalg.norm(raw)
        final = flat(ps, lambda p: p.grad)
        route = np.linalg.norm(flat(ps, routes.get))
        h.require(raw_size > 0, f"{name} Full CE has no gradient")
        if name in o.ORIGINAL:
            base = flat(ps, historical.get)
            off_raw = flat(ps, off_gradient.get)
            q = flat(ps, differential.get)
            q_norm = np.linalg.norm(q)
            split_error = np.linalg.norm(q - (raw - off_raw))
            # Branch accumulation versus single backward changes summation order in the
            # original patient and feature reductions. Compare against their float32 envelope.
            split_bound = f.gamma(3 * len(cap["cam"]) + 6 * cap["cam"].shape[-1]) * (
                raw_size + np.linalg.norm(off_raw))
            h.require(split_error <= split_bound, f"{name} differential does not reproduce Full minus Off")
            size = np.linalg.norm(base)
            tangent_unit = base / size if size > shared["historical_norm_floor"] else np.zeros_like(base)
            proj = q - (q @ tangent_unit) * tangent_unit
            d = plan[name]
            rho = f.gamma(2 * len(base) + 16, f.U64)
            f.bounded(d["historical"], size, 2 * len(base) + 16, "Historical total gradient norm", size, f.U64)
            f.bounded(d["raw"], q_norm, 2 * len(base) + 16, "Independent residual-differential norm", q_norm, f.U64)
            f.bounded(d["projected"], np.linalg.norm(proj), 4 * len(base) + 40,
                      "Independent total-gradient tangent projection", q_norm + size, f.U64)
            actual = final - base
            norm_added = np.linalg.norm(actual)
            f.bounded(d["total_applied"], norm_added, 2 * len(base) + 16, "Real optimizer input gradient", norm_added, f.U64)
            h.require(norm_added <= shared["shared_cap"] * size * (1 + 4 * rho),
                      f"{name} applied gradient exceeds 0.1% of historical total")
            step = float(d["scale"]) * proj
            cast_allowance = f.U32 * np.linalg.norm(abs(base) + abs(step)) + 8 * rho * (size + q_norm)
            h.require(np.linalg.norm(actual - step) <= cast_allowance,
                      f"{name} changed historical gradient beyond the independently rounded tangent addend")
            inner = float(actual @ tangent_unit)
            absolute_bound = f.U32 * (abs(tangent_unit) @ (abs(base) + abs(step))) + 8 * rho * (q_norm + size + np.linalg.norm(proj))
            h.require(abs(inner) <= absolute_bound, f"{name} measured addend is outside the floating-point tangent")
            routes_to_update[name] = (tangent_unit, actual, absolute_bound)
        else:
            aux_added = np.linalg.norm(final - raw * shared["new_coefficient"])
            total_limit = sum(CFG["auxiliary"][k]["cap"] for k in o.auxiliary_terms(group, name))
            h.require(aux_added <= total_limit * raw_size * (1 + f.gamma(2 * len(raw), f.U64)),
                      f"{name} auxiliary objective overwhelms Full CE")
            h.require(np.linalg.norm(final) > (1 - total_limit) * raw_size and
                      (not check_each or all(full[p] is not None and f.f64(full[p]).any() for p in ps)),
                      f"A new trainable tensor in {name} misses Full CE")
            norm_added = aux_added
        h.require(name == "head" or route > 0, f"{name} is disconnected from the real feature residual derivative")
        records[name] = {"full_class_gradient": float(raw_size), "class_through_delta": float(route),
                         "actual_add_norm": float(norm_added)}
        if name in o.ORIGINAL:
            records[name].update(off_class_gradient=float(np.linalg.norm(off_raw)), difference_gradient=float(q_norm),
                                 difference_split_error=float(split_error), difference_split_bound=float(split_bound))
        else:
            records[name]["gradient_tensors_nonzero"] = int(sum(full[p] is not None and f.f64(full[p]).any() for p in ps))
            records[name]["trainable_tensors"] = len(ps)

    optimizers = [state[k] for k in h.OPTS] + [ctx["optimizer"]]
    theta_before = {name: flat(ps, lambda p: p) for name, ps in groups.items()}
    theta_native = native_optimizer_steps(optimizers[:3], historical)
    receipt = []

    def at_step(opt):
        return h.sha([p.grad for pg in opt.param_groups for p in pg["params"]])

    def track_step(opt, *_):
        receipt.append(at_step(opt))

    expected = [at_step(opt) for opt in optimizers]
    for opt in optimizers:
        hook = opt.register_step_pre_hook(track_step)
        try:
            opt.step()
        finally:
            hook.remove()
    h.require(receipt == expected and h.rng_sha() == before, "Optimizer received different gradients or consumed RNG")
    budget = o.optimizer_measurements(groups, plan, updates)
    for name, ps in groups.items():
        updated = flat(ps, lambda p: p) - theta_before[name]
        h.require(np.linalg.norm(updated) > 0, f"{name} did not update at optimizer step")
        records[name]["budget"] = budget[name]
        if name not in o.ORIGINAL:
            continue
        unit_hist, extra, absolute_bound = routes_to_update[name]
        step_native = np.concatenate([theta_native[p].ravel() for p in ps]) - theta_before[name]
        n = len(step_native)
        norm_delta = float(np.linalg.norm(updated - step_native))
        update_norm = float(np.linalg.norm(updated))
        native_norm = float(np.linalg.norm(step_native))
        f.bounded(budget[name]["update_norm"], update_norm, 2*n+16, "Computed optimizer parameter motion", update_norm, f.U64)
        dot_hist = float(updated @ unit_hist)
        dot_extra = float(updated @ extra)
        f.bounded(budget[name]["update_dot_historical"], dot_hist, 2*n+16, "Parameter motion vs historical gradient",
                  update_norm, f.U64)
        f.bounded(budget[name]["update_dot_extra"], dot_extra, 2*n+16, "Parameter motion vs auxiliary gradient",
                  update_norm * np.linalg.norm(extra), f.U64)
        records[name]["optimizer_delta"] = {"actual_norm": update_norm, "same_state_native_norm": native_norm,
                                           "difference_norm": norm_delta, "historical_dot": dot_hist,
                                           "actual_extra_dot": dot_extra, "tangent_roundoff_bound": float(absolute_bound)}
    return {"gradients": records, "train_formulas": fp, "optimizer_receipt_exact": True,
            "component_gradient": component_gradients, "candidate_actual_head": components,
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
        for key in ("action", "w_sem", "w_sp", "prompt_contrast", "calibrated_contrast"):
            values = f.f64(d[key][indices])
            result[key + "_direction_mean"] = values.mean(0).tolist()
            result[key + "_direction_sd"] = values.std(0).tolist()
            if key in ("action", "prompt_contrast"):
                result[key + "_direction_abs_mean"] = abs(values).mean(0).tolist()
        attention = f.f64(d["attention"][indices])
        result["query_mean_attention"] = attention.mean(0).tolist()
        result["original_query_mean_attention"] = f.f64(d["original_attention"][indices]).mean(0).tolist()
        result["differential_query_mean_attention"] = f.f64(d["differential_attention"][indices]).mean(0).tolist()
        result["common_suppression"] = float(d["common_suppression"])
        angles = abs(f.f64(d["rotation_angles"][indices]))
        result["rotation_max_radians"] = float(angles.max())
        result["rotation_mean_abs_radians"] = float(angles.mean())
        result["direction_action_positive_share"] = (f.f64(d["action"][indices]) > 0).mean(0).tolist()
        return result
    llr = f.f64(d["llr"][indices])
    reliability = f.f64(d["reliability"][indices])
    evidence = f.f64(d["evidence"][indices])
    jac = f.f64(d["jacobian"][indices])
    projected_margin = (f.f64(d["delta"][indices]) * jac).sum(-1)
    signs = np.sign(evidence)
    projection = [(f.f64(d[k][indices]) * jac).sum(-1) for k in ("delta_parallel", "delta_perp")]
    result = {"first_order_same_sign_share": float((projection[0] * projection[1] > 0).mean()),
              "first_order_opposite_sign_share": float((projection[0] * projection[1] < 0).mean()),
              "parallel_jacobian_norm_mean": float(np.linalg.norm(f.f64(d["parallel"][indices]), axis=-1).mean()),
              "complement_jacobian_norm_mean": float(np.linalg.norm(f.f64(d["perp"][indices]), axis=-1).mean()),
              "llr_axis_mean": llr.mean(0).tolist(), "llr_axis_sd": llr.std(0).tolist(),
              "reliability_axis_mean": reliability.mean(0).tolist(),
              "evidence_agrees_linear_share": float((np.sign(projected_margin) * signs > 0).mean()),
              "evidence_opposes_linear_share": float((np.sign(projected_margin) * signs < 0).mean())}
    for key in ("strength_parallel", "strength_perp"):
        action = f.f64(d[key][indices])
        result[key + "_abs_mean"] = float(abs(action).mean())
        result[key + "_agree_evidence_share"] = float((np.sign(action) * signs > 0).mean())
        result[key + "_oppose_evidence_share"] = float((np.sign(action) * signs < 0).mean())
    return result


def mechanism(ctx, best):
    state = ctx["state"]
    y = state["labels_np"][state["test_np"]]
    output, formulas, light, all_changes, direction = {}, {}, {}, {}, {}
    unchanged_controls = None
    for key in run.INTERVENTIONS[ctx["group"]]:
        result, cap = independent_eval(ctx, key)
        formulas[key] = f.branch_formulas(ctx, cap, intervention=key)
        d, rows = cap["details"], state["test"]
        residual = f.array(d["delta"][rows])
        logits, p, pred = result["logits"], result["probability"], result["prediction"]
        if key == "full":
            match_eval(result, best["evaluation"], "Best")
            full = {"prob": p, "pred": pred, "logits": logits, "residual": residual}
            keys = ["gate", "capacity", "ratio", "raw_ratio", "margin"]
            if ctx["group"] == "G2":
                keys += ["evidence", "support", "conflict", "strength_parallel", "strength_perp", "desired_change",
                         "target_margin", "correction_reliability", "allocation", "disagreement"]
            else:
                keys += ["beta", "amplitude"]
            light = {name: f.array(d[name][rows]).tolist() for name in keys}
            light["action_abs_mean"] = d["action"][rows].abs().mean(-1).cpu().tolist()
            if ctx["group"] == "G1":
                light["prompt_contrast_abs_mean"] = d["prompt_contrast"][rows].abs().mean(-1).cpu().tolist()
                light["rotation_max_radians"] = d["rotation_angles"][rows].abs().amax(-1).cpu().tolist()
            else:
                unchanged_controls = {k: d[k].clone() for k in ("correction_reliability", "allocation")}
            light["bound_shrink"] = (1 - d["scale"][rows, 0]).cpu().tolist()
            for name, part in (("parallel", "delta_parallel"), ("complement", "delta_perp")):
                light[name + "_ratio"] = (d[part][rows].norm(dim=-1) /
                                          d["cam_norm"][rows].clamp_min(1e-8)).cpu().tolist()
            direction = per_direction(ctx, cap, y)
            formulas["action_amplitude"] = f.amplitude_probe(ctx, cap["cam"])
        if key == "calibrated_evidence_off":
            for k, v in unchanged_controls.items():
                f.equal(v, d[k], f"Replacing the learned target must preserve {k}")
            formulas[key]["learned_trust_and_allocation_exact"] = True
        if key == "residual_off":
            direct, native_cap = run.forward(state, False)
            f.equal(direct["logp"][rows].exp(), p, "Residual-off native evaluation probability")
            f.equal(native_cap["logits"][rows], logits, "Residual-off original head")
            f.equal(native_cap["cam"], cap["enhanced"], "Residual-off exact original CAM144")
        output[key] = {"metrics": metrics(p, pred, y), "response_vs_full": scalar_response(p, pred, y, logits, residual, full),
                       "probability": p[:, 0].tolist(), "prediction": pred.tolist(),
                       "true_margin": signed_margin(logits, y).tolist()}
        if ctx["group"] == "G1":
            output[key]["amplitude"] = f.array(d["amplitude"][rows]).tolist()
            output[key]["direction_action"] = f.array(d["action"][rows]).tolist()
        else:
            output[key]["desired_change"] = f.array(d["desired_change"][rows]).tolist()
        all_changes[key] = np.linalg.norm(residual - full["residual"], axis=-1).tolist()
    full_variant, anon = output["full"], output["anonymous_query"]
    light["query_control_abs_change"] = abs(np.asarray(full_variant.get("amplitude", full_variant.get("desired_change"))) -
                                             np.asarray(anon.get("amplitude", anon.get("desired_change")))).tolist()
    if ctx["group"] == "G1":
        actions_diff = np.abs(np.asarray(full_variant["direction_action"]) - np.asarray(anon["direction_action"]))
        light["query_direction_mean_abs_change"] = actions_diff.mean(-1).tolist()
    else:
        base = np.asarray(output["residual_off"]["true_margin"])
        wrong = np.asarray(output["residual_off"]["prediction"]) != y
        parts = []
        for name in ("spectral_only", "complement_only", "calibrated_evidence_off"):
            gain = np.asarray(output[name]["true_margin"]) - base
            direction[name + "_mean_gain"] = float(gain.mean())
            direction[name + "_wrong_mean_gain"] = float(gain[wrong].mean()) if wrong.any() else 0.
            if name != "calibrated_evidence_off":
                parts.append(gain)
        direction["actual_head_spaces_same_sign_share"] = float((parts[0] * parts[1] > 0).mean())
        direction["actual_head_spaces_opposite_sign_share"] = float((parts[0] * parts[1] < 0).mean())
        sign = 1 - 2 * y
        light["llr_correct_sign"] = ((sign * np.asarray(light["evidence"])) > 0).tolist()
        light["target_shift_correct_sign"] = ((sign * np.asarray(light["desired_change"])) > 0).tolist()
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
    diagnostic = twin_probe(ctx, preflight["dropout"], check_each=True)
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
        difficulty = f.legal_memory(ctx)
        if historic is None:
            historic = archived_reference(ctx, manifest)
        archive = audit_journal(ctx, task, best, latest, preflight)
        run.restore(ctx, best, manifest["implementation"])
        h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best full reloaded model/optimizers identity differs")
        evaluated, response, formulas, direction = mechanism(ctx, best)
        best_probe = twin_probe(ctx, preflight["dropout"], check_each=best["epoch"] >= run.START + 2)
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
                "zero_residual_exact": preflight["natural_zero_gradient_and_update_exact"] and
                preflight["mature_forced_zero_gradient_exact"],
                "difficulty_memory": difficulty,
                "original_o": f.check_original_o(ctx), "stats": f.class_banks(ctx) if group == "G2" else None}
        h.write_json(paths["audit"], data)
        print(f"PASS independent audit {group} fold {fold:02d}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=h.FOLDS)
    audit_fold(parser.parse_args().fold)
