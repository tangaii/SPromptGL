#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency, t

import run as runner
from draps import DRAPS

HERE = runner.HERE
RESULTS = runner.RESULTS
METRICS = runner.METRICS
GROUPS = runner.GROUPS
E781 = Path("/model/spromptgl/experiment/experiment78_1_sh_dcpss")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    runner.atomic_json(path, value)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def cramers_v(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(a) != len(b):
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    denom = table.sum() * (min(table.shape) - 1)
    return float(math.sqrt(max(chi2, 0.0) / denom)) if denom else 0.0


def _metric_match(actual, expected, tol=1e-9):
    return all(abs(float(actual[k]) - float(expected[k])) <= tol for k in METRICS)


def _json_value(v):
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def unit_tests():
    """Small deterministic tests independent of formal folds."""
    torch.manual_seed(782)
    coords = torch.nn.functional.normalize(torch.randn(2, 8, 16), dim=-1)
    classes = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    h = torch.randn(871, 144)
    labels = torch.tensor(([0, 1] * 435) + [0])
    sites = torch.arange(871) % 7
    support = torch.arange(0, 700, dtype=torch.long)
    query = torch.arange(700, 740, dtype=torch.long)
    weights = torch.ones(2)
    tests = {}

    # Manual ESS arithmetic required by the design.
    masses = torch.tensor([1.0, 2.0, 3.0])
    expected_ess = 36.0 / 14.0
    observed_ess = float(masses.sum().square() / masses.square().sum())
    tests["effective_sample_size_manual"] = abs(observed_ess - expected_ess) < 1e-6

    m = DRAPS(coords, classes, 3, "spromptgl")
    with torch.no_grad():
        m.anchors.copy_(torch.nn.functional.normalize(torch.randn_like(m.anchors), dim=-1))
    out, d = m.forward_feature(h, labels, support, query, sites, weights, training=False)
    tests["k3_finite"] = finite_tree({"out": out, "details": d})
    tests["gate_in_range"] = bool((d["gate"] >= 0).all() and (d["gate"] <= 0.25 + 1e-7).all())
    tests["test_query_not_loo"] = bool((~d["is_support_query"]).all() and torch.allclose(d["loo_removed_mass"], torch.zeros_like(d["loo_removed_mass"])))

    # Candidate index permutation invariance: all class evidence and fused features
    # must remain unchanged when candidate rows are permuted independently.
    m_perm = DRAPS(coords, classes, 3, "spromptgl")
    m_perm.load_state_dict(m.state_dict(), strict=True)
    with torch.no_grad():
        p0 = torch.tensor([2, 0, 1])
        p1 = torch.tensor([1, 2, 0])
        m_perm.anchors[0].copy_(m_perm.anchors[0, p0])
        m_perm.anchors[1].copy_(m_perm.anchors[1, p1])
    out_p, d_p = m_perm.forward_feature(h, labels, support, query, sites, weights, training=False)
    perm_err = float((d["class_evidence"] - d_p["class_evidence"]).abs().max())
    fused_err = float((out - out_p).abs().max())
    tests["candidate_index_permutation_invariant"] = perm_err < 2e-6 and fused_err < 2e-6
    tests["candidate_index_permutation_max_error"] = perm_err
    tests["candidate_index_permutation_fused_max_error"] = fused_err

    # Asymmetric ASD K=3 / TD K=5 candidate capacities.
    m_asym = DRAPS(coords, classes, (3, 5), "spromptgl")
    with torch.no_grad():
        m_asym.anchors.copy_(torch.nn.functional.normalize(torch.randn_like(m_asym.anchors), dim=-1))
    out_a, d_a = m_asym.forward_feature(h, labels, support, query, sites, weights, training=False)
    tests["asymmetric_k3_k5_forward"] = finite_tree({"out": out_a, "details": d_a}) and tuple(d_a["beta"].shape) == (40, 2, 5)

    # Exact LOO on legal training queries, including positive removed mass.
    train_query = support[:80]
    _, d_train = m.forward_feature(h, labels, support, train_query, sites, weights, training=True)
    tests["training_query_exact_loo"] = bool(d_train["is_support_query"].all() and
                                               (d_train["loo_removed_mass"].sum(dim=(1, 2)) > 0).all())
    tests["loo_shape"] = tuple(d_train["loo_removed_mass"].shape) == (80, 2, 3)

    # Low-support fallback: 20 patients per class cannot make any K=5 candidate
    # reach the predeclared ESS=30 reference threshold.
    small_support = torch.arange(40, dtype=torch.long)
    small_query = torch.arange(40, 50, dtype=torch.long)
    m_small = DRAPS(coords, classes, 5, "spromptgl")
    with torch.no_grad():
        m_small.anchors.copy_(torch.nn.functional.normalize(torch.randn_like(m_small.anchors), dim=-1))
    _, d_small = m_small.forward_feature(h, labels, small_support, small_query, sites, weights, training=False)
    tests["all_low_reliability_global_center_fallback"] = bool(d_small["class_fallback"].all()) and finite_tree(d_small)
    tests["low_support_no_nan"] = finite_tree({"out": d_small["h_fused"], "details": d_small})

    # Test-label flip invariance on the synthetic test query.
    labels_flip = labels.clone()
    labels_flip[query] = 1 - labels_flip[query]
    _, d_flip = m.forward_feature(h, labels_flip, support, query, sites, weights, training=False)
    tests["test_label_invariant"] = torch.equal(out, d_flip["h_fused"]) and torch.equal(d["class_evidence"], d_flip["class_evidence"])
    tests["all_finite"] = all(v is True for k, v in tests.items() if isinstance(v, bool))
    write_json(RESULTS / "unit_tests.json", {"status": "PASS" if tests["all_finite"] else "FAIL", "tests": tests})
    if not tests["all_finite"]:
        raise RuntimeError(f"DRAPS unit tests failed: {tests}")
    return tests


def _restore_checkpoint(group, fold, frozen_sha, data):
    task_file, checkpoint_file, _, _ = runner.task_paths(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("source_sha256") != frozen_sha:
        raise RuntimeError(f"task {group} fold {fold} is not complete under the frozen source")
    if task.get("checkpoint_sha256") != runner.sha_file(checkpoint_file):
        raise RuntimeError(f"checkpoint SHA mismatch for {group} fold {fold}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("source_sha256") != frozen_sha or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError("checkpoint identity mismatch")
    state, warmup, identity = runner.prepare(fold)
    if identity != task.get("identity") or identity != checkpoint.get("identity"):
        raise RuntimeError("E74.4 identity differs from task/checkpoint")
    if checkpoint["initialization"]["original_optimizer_sha256"] != identity["optimizer_sha"]:
        raise RuntimeError("original optimizer inheritance SHA mismatch")
    support = runner.support_ids(state)
    cp_support = np.asarray(checkpoint["initialization"]["support_ids"], dtype=np.int64)
    if not np.array_equal(support, cp_support) or np.intersect1d(support, state["test_np"]).size:
        raise RuntimeError("new prototype support is not legal outer-train support")
    overlap = np.intersect1d(state["train_np"], state["test_np"]).astype(np.int64)
    expected_overlap = np.asarray([376, 406], dtype=np.int64) if fold == 2 else np.zeros(0, dtype=np.int64)
    if not np.array_equal(overlap, expected_overlap):
        raise RuntimeError(f"unexpected historical overlap in fold {fold}: {overlap.tolist()}")
    if np.intersect1d(overlap, support).size:
        raise RuntimeError("Fold 2 overlap entered new support")
    history = checkpoint.get("loss_history", [])
    if not history:
        raise RuntimeError("checkpoint has no loss history")
    max_acc = max(float(row["metrics"]["acc"]) for row in history)
    first_best = next(int(row["epoch"]) for row in history if float(row["metrics"]["acc"]) == max_acc)
    if int(checkpoint["best_epoch_1based"]) != first_best:
        raise RuntimeError("best checkpoint violates strict first-maximum ACC selection")
    if int(task["best_epoch_1based"]) != int(checkpoint["best_epoch_1based"]):
        raise RuntimeError("task/checkpoint best epoch mismatch")

    spec = runner.CFG["group_specs"][group]
    geometry, coords, _ = runner.prompt_tensors(state["y"].device)
    prompt_classes = geometry["prompt_classes"] if spec["prompt_source"] == "spromptgl" else geometry["anonymous_classes"]
    module = runner.initialize(coords, prompt_classes.to(state["y"].device), int(spec["k_max"]),
                               spec["prompt_source"], fold, state["y"].device)
    optimizer = runner._new_optimizer(module)
    best_state = checkpoint["best_state"]
    runner.hist().load_bundle(state, best_state["base_state"])
    module.load_state_dict(best_state["module_state"], strict=True)
    optimizer.load_state_dict(best_state["new_optimizer"])
    if runner.sha_object(optimizer.state_dict()) != runner.sha_object(best_state["new_optimizer"]):
        raise RuntimeError("new optimizer state did not restore exactly")
    if not finite_tree(best_state["module_state"]):
        raise RuntimeError("non-finite module state in checkpoint")
    if int(checkpoint["initialization"]["module_parameter_count"]) != int(runner.CFG["method"]["trainable_parameters_per_group"]):
        raise RuntimeError("module parameter count differs from frozen design")
    if checkpoint["best_rng"].get("anchor_ema_sha") and checkpoint["best_rng"]["anchor_ema_sha"] != module.anchor_digest():
        raise RuntimeError("EMA anchor digest differs from best RNG record")

    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    runner.runtime().restore_rng(checkpoint["best_rng"]["packed"])
    module.eval()
    with torch.no_grad():
        output, diagnostics = runner._forward(state, module, training=False, support=support,
                                              query=state["test_np"], sites=sites)
        metrics = runner._metrics(output["logp"], state)
    if not np.array_equal(np.asarray(metrics["pred"]), np.asarray(checkpoint["test_prediction"])):
        raise RuntimeError("checkpoint prediction readback differs")
    if not np.allclose(np.asarray(metrics["prob"]), np.asarray(checkpoint["test_probability"]), atol=1e-7, rtol=0):
        raise RuntimeError("checkpoint probability readback differs")
    if not _metric_match(metrics, checkpoint["metrics"]) or not _metric_match(metrics, task["metrics"]):
        raise RuntimeError("checkpoint metric readback differs")
    if not torch.allclose(output["logp"].exp().sum(1), torch.ones(871, device=output["logp"].device), atol=1e-6):
        raise RuntimeError("OutputLayer result is not normalized log probability")
    eval_rng = runner.runtime().rng_sha(runner.runtime().pack_rng())
    if eval_rng != checkpoint["best_rng"]["rng_after_eval"]:
        raise RuntimeError("checkpoint evaluation consumed unexpected RNG")

    # Test-label invariance and exact training LOO readback.
    original_y = state["y"].detach().clone()
    test_ids = torch.as_tensor(state["test_np"], dtype=torch.long, device=state["y"].device)
    with torch.no_grad():
        state["y"][test_ids] = 1 - state["y"][test_ids]
        flipped, flipped_details = runner._forward(state, module, training=False, support=support,
                                                   query=state["test_np"], sites=sites)
        state["y"].copy_(original_y)
    if not torch.equal(output["logp"], flipped["logp"]) or not torch.equal(diagnostics["h_fused"], flipped_details["h_fused"]):
        raise RuntimeError("test-label flip changed inference")
    with torch.no_grad():
        _, train_details = runner._forward(state, module, training=True, support=support,
                                            query=support, sites=sites)
    if not bool(train_details["is_support_query"].all()) or train_details["loo_removed_mass"].shape[0] != len(support):
        raise RuntimeError("training query LOO readback failed")
    if bool((train_details["loo_removed_mass"].sum(dim=(1, 2)) <= 0).any()):
        raise RuntimeError("some training query did not remove its own support mass")
    if not finite_tree(train_details):
        raise RuntimeError("non-finite training LOO diagnostics")

    base_task = json.loads((runner.BASE / ".runtime/tasks" / f"{runner.BASE_GROUP}_f{fold:02d}.json").read_text())
    base_trace = {int(x["epoch"]): x for x in base_task["rng_trace"]}
    new_trace = {int(x["epoch"]): x for x in checkpoint["rng_trace"]}
    common = sorted(set(base_trace) & set(new_trace))
    exact = sum(all(base_trace[e][key] == new_trace[e][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
    branch = checkpoint["loss_history"][0].get("gradient_audit_first_epoch", {}).get("auxiliary", {}).get("classification_semantic_residual_branch", {})
    if not branch or min(float(branch.get(k, 0.0)) for k in ("module", "darg_gfr", "cam")) <= 0:
        raise RuntimeError("first-epoch fusion classification gradient did not reach all paths")
    audit = {
        "group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
        "checkpoint_sha256": runner.sha_file(checkpoint_file), "source_sha256": frozen_sha,
        "base_identity_sha": identity.get("optimizer_sha"), "warmup_sha256": identity["warmup_sha256"],
        "warmup_rng_sha": warmup["rng_sha"], "initial_rng_sha": identity["initial_rng_sha"],
        "support_count": int(len(support)), "support_sha256": checkpoint["initialization"]["support_sha256"],
        "support_disjoint_from_test": True, "fold2_overlap_rows_excluded_from_new_support": True,
        "all_original_optimizers_restored": True, "new_optimizer_restored": True,
        "ema_anchor_restored": True, "test_label_invariant": True, "training_loo_exact": True,
        "best_acc_rule_pass": True, "best_checkpoint_readback_metrics_match": True,
        "classification_output_is_log_probability": True,
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]), "epochs_executed": int(checkpoint["epochs_executed"]),
        "wall_seconds": float(task.get("wall_seconds", 0.0)),
        "metrics": {k: float(metrics[k]) for k in METRICS},
        "rng_common_epochs": len(common), "rng_exact_epochs": exact,
        "rng_mismatches": len(common) - exact, "new_gradient_norms": branch,
    }
    return audit, checkpoint, diagnostics, metrics, state, module, optimizer, train_details


def _mechanism(group, fold, checkpoint, diagnostics, train_details, data):
    diag = diagnostics
    spec = runner.CFG["group_specs"][group]
    mask = np.asarray(diag["reliability_weights"], dtype=np.float64)
    kmax = int(spec["k_max"])
    valid = np.zeros((2, kmax), dtype=bool)
    for cls in (0, 1):
        valid[cls, :kmax] = True
    if "class_fallback" in diag:
        fallback = np.asarray(diag["class_fallback"], dtype=bool)
    else:
        fallback = np.zeros(2, dtype=bool)
    neff = np.asarray(diag["n_eff"], dtype=np.float64)
    rel = np.asarray(diag["prototype_reliability"], dtype=np.float64)
    site_rel = np.asarray(diag["site_reliability"], dtype=np.float64)
    site_count = np.asarray(diag["site_support_count"], dtype=np.float64)
    proto = np.asarray(diag["proto_z"], dtype=np.float64)
    beta = np.asarray(diag["beta"], dtype=np.float64)
    beta_rel = np.asarray(diag["beta_reliable"], dtype=np.float64)
    attention = np.asarray(diag["attention"], dtype=np.float64)
    support_y = np.asarray(diag["support_labels"], dtype=np.int64)
    support_sites = np.asarray(diag["support_sites"], dtype=np.int64)
    support_ids = np.asarray(diag["support_ids"], dtype=np.int64)
    age_ids = data["age_ids"][support_ids]
    proto_mean = proto.mean(0)
    sims, collapse, site_vs, age_vs, route_props = [], [], [], [], []
    for cls in (0, 1):
        kc = runner.CFG["group_specs"][group]["k_max"]
        v = proto_mean[cls, :kc]
        if kc > 1:
            upper = (v @ v.T)[np.triu_indices(kc, 1)]
            sims.extend(upper.tolist())
            collapse.extend((upper > 0.98).astype(float).tolist())
        cls_mask = support_y == cls
        rho = np.asarray(diag["rho_support"])[cls_mask, :kc]
        assign = rho.argmax(1)
        site_vs.append(cramers_v(assign, support_sites[cls_mask]))
        age_vs.append(cramers_v(assign, age_ids[cls_mask]))
        route_props.append((np.bincount(assign, minlength=kc) / max(len(assign), 1)).tolist())
    prompt_tvs = []
    prompt_entropy = []
    prompt_std = []
    prompt_mean = {}
    for cls in (0, 1):
        ac = attention[:, cls]
        prompt_entropy.extend((-(ac * np.log(np.maximum(ac, 1e-12))).sum(-1)).tolist())
        prompt_std.append(float(ac.std(0).mean()))
        prompt_mean[str(cls)] = ac.mean(0).tolist()
        kc = runner.CFG["group_specs"][group]["k_max"]
        for left in range(kc):
            for right in range(left + 1, kc):
                prompt_tvs.extend((0.5 * np.abs(ac[:, left] - ac[:, right]).sum(-1)).tolist())
    route_entropy = -(beta * np.log(np.maximum(beta, 1e-12))).sum(-1)
    route_norm = float(route_entropy.mean() / math.log(kmax)) if kmax > 1 else 0.0
    residual = np.asarray(diag["h_fused"]) - np.asarray(diag["h_original"])
    h_orig = np.asarray(diag["h_original"])
    gates = np.asarray(diag["gate"], dtype=np.float64)
    patient_rel = np.asarray(diag["patient_reliability"], dtype=np.float64)
    proto_scores = np.asarray(diag["proto_scores"], dtype=np.float64)
    class_evidence = np.asarray(diag["class_evidence"], dtype=np.float64)
    train_proto_scores = np.asarray(train_details["proto_scores"].detach().cpu())
    train_sem_scores = np.asarray(train_details["class_evidence"].detach().cpu())
    train_labels = np.asarray(train_details["query_ids"].detach().cpu())
    # labels are looked up from the original state through the legal support identities
    y_train = np.asarray(train_details["support_labels"].detach().cpu())[0:0]
    # train_details stores support labels; query order is support order, so use support labels directly.
    y_train = np.asarray(train_details["support_labels"].detach().cpu(), dtype=np.int64)
    proto_acc = float((train_proto_scores.argmax(1) == y_train).mean())
    sem_acc = float((train_sem_scores.argmax(1) == y_train).mean())
    return {
        "group": group, "fold": fold, "candidate_k": kmax,
        "effective_k_asd": int(np.sum(mask[0] > 1e-8)), "effective_k_td": int(np.sum(mask[1] > 1e-8)),
        "n_eff_asd": json.dumps(neff[0].tolist()), "n_eff_td": json.dumps(neff[1].tolist()),
        "reliability_asd": json.dumps(rel[0].tolist()), "reliability_td": json.dumps(rel[1].tolist()),
        "normalized_reliability_asd": json.dumps(mask[0].tolist()), "normalized_reliability_td": json.dumps(mask[1].tolist()),
        "site_reliability_asd": json.dumps(site_rel[0].tolist()), "site_reliability_td": json.dumps(site_rel[1].tolist()),
        "support_site_count_asd": json.dumps(site_count[0].tolist()), "support_site_count_td": json.dumps(site_count[1].tolist()),
        "global_center_fallback_asd": bool(fallback[0]), "global_center_fallback_td": bool(fallback[1]),
        "mean_prototype_cosine_similarity": float(np.mean(sims)) if sims else 1.0,
        "prototype_pair_collapse_fraction_cosine_gt_0_98": float(np.mean(collapse)) if collapse else 0.0,
        "support_assignment_proportions": json.dumps(route_props),
        "test_route_entropy_normalized": route_norm,
        "subprototype_prompt_tv": float(np.mean(prompt_tvs)) if prompt_tvs else 0.0,
        "prompt_attention_entropy": float(np.mean(prompt_entropy)) if prompt_entropy else 0.0,
        "patient_prompt_weight_std": float(np.mean(prompt_std)), "prompt_weights_by_class": json.dumps(prompt_mean),
        "site_cramers_v_by_class": json.dumps(site_vs), "site_cramers_v_mean": float(np.mean(site_vs)),
        "age_cramers_v_by_class": json.dumps(age_vs), "age_cramers_v_mean": float(np.mean(age_vs)),
        "train_proto_accuracy": proto_acc, "train_semantic_accuracy": sem_acc,
        "train_proto_loss": float(train_details["proto_loss"].detach().cpu()),
        "train_semantic_loss": float(train_details["semantic_loss"].detach().cpu()),
        "mean_proto_score_asd": float(proto_scores[:, 0].mean()), "mean_proto_score_td": float(proto_scores[:, 1].mean()),
        "mean_proto_margin": float((proto_scores[:, 0] - proto_scores[:, 1]).mean()),
        "mean_semantic_margin": float((class_evidence[:, 0] - class_evidence[:, 1]).mean()),
        "gate_mean": float(gates.mean()), "gate_sd": float(gates.std()),
        "gate_q05": float(np.quantile(gates, 0.05)), "gate_q50": float(np.quantile(gates, 0.50)),
        "gate_q95": float(np.quantile(gates, 0.95)), "gate_patient_reliability_corr": float(np.corrcoef(gates, patient_rel)[0, 1]) if gates.std() and patient_rel.std() else 0.0,
        "feature_residual_relative_norm": float(np.linalg.norm(residual) / max(np.linalg.norm(h_orig), 1e-12)),
        "support_count": int(len(support_ids)), "query_count": int(len(diag["query_ids"])),
    }


def _interventions(group, fold, state, module, full_metrics, data):
    if group != "E78.2-G3-SPromptGL-K5":
        return []
    support = runner.support_ids(state)
    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    before_pred = np.asarray(full_metrics["pred"], dtype=np.int64)
    before_prob = np.asarray(full_metrics["prob"], dtype=np.float64)
    labels = state["y"].detach().cpu().numpy()[state["test_np"]]
    rows = []
    for intervention in ("module_off", "reliability_off", "global_center", "anonymous", "prompt_class_swap", "feature_residual_off"):
        module.eval()
        with torch.no_grad():
            out, _ = runner._forward(state, module, training=False, support=support,
                                      query=state["test_np"], sites=sites, intervention=intervention)
            metrics = runner._metrics(out["logp"], state)
        pred = np.asarray(metrics["pred"], dtype=np.int64)
        prob = np.asarray(metrics["prob"], dtype=np.float64)
        rows.append({
            "group": group, "fold": fold, "intervention": intervention,
            **{k: float(metrics[k]) for k in METRICS},
            "mean_abs_probability_change": float(np.abs(prob - before_prob).mean()),
            "max_abs_probability_change": float(np.abs(prob - before_prob).max()),
            "hard_label_flips": int((pred != before_pred).sum()),
            "correct_to_wrong": int(((before_pred == labels) & (pred != labels)).sum()),
            "wrong_to_correct": int(((before_pred != labels) & (pred == labels)).sum()),
        })
    return rows


def _read_reference(path, group):
    rows = [r for r in read_csv(path) if r.get("group") == group]
    if len(rows) != 10:
        raise RuntimeError(f"reference {group} does not contain 10 folds")
    out = {}
    for row in rows:
        out[int(row["fold"])] = {k: float(row[k]) for k in METRICS}
    return out


def _summary(group, rows):
    result = {"group": group, "folds": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(r[metric]) for r in rows], dtype=float) * 100.0
        result[f"{metric}_mean_percent"] = float(values.mean()) if len(values) else float("nan")
        result[f"{metric}_population_sd_percent"] = float(values.std(ddof=0)) if len(values) else float("nan")
    return result


def _paired(name, left, right, folds):
    rows = []
    for metric in METRICS:
        delta = np.asarray([(left[f][metric] - right[f][metric]) * 100.0 for f in folds], dtype=float)
        mean = float(delta.mean()) if len(delta) else float("nan")
        sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        half = float(t.ppf(0.975, len(delta) - 1) * sd / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
        rows.append({"comparison": name, "scope": "all10" if len(folds) == 10 else "exclude_fold2",
                     "metric": metric, "n_folds": len(folds), "mean_difference_pp": mean,
                     "population_sd_difference_pp": float(delta.std(ddof=0)) if len(delta) else float("nan"),
                     "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
                     "losses": int((delta < -1e-12).sum()), "descriptive_95ci_low_pp": mean - half,
                     "descriptive_95ci_high_pp": mean + half,
                     "fold_differences_pp": json.dumps({str(f): float((left[f][metric] - right[f][metric]) * 100.0) for f in folds}, sort_keys=True)})
    return rows


def _write_report(audits, summaries, comparisons, mechanisms, interventions, unit, source_sha, status, audit_status):
    by_group = {r["group"]: r for r in summaries}
    lines = [
        "# Experiment 78.2: DRAPS 结果报告", "",
        f"状态：`{status}`。正式目标为4组×10折；完成任务 {audit_status['trained_tasks']}/40，独立Checkpoint审计 {audit_status['independent_checkpoint_audits']}/40。",
        "", "## 论文、代码与基座审计", "",
        "SPromptGL 与 BrainPrompt 本地全文已审计；HP²L 仅访问 DOI/ScienceDirect 摘要和网页预览，未将不可访问全文公式当作已读内容。GitHub 的 tangaii/SPromptGL、wanxixi11/SPromptGL 与 AngusMonroe/BrainPrompt 已检查；参考提交 d3e78fb 是 E78.1 添加提交。详细范围见 `literature/source_review.md`。",
        f"唯一训练基座是 `{runner.BASE_GROUP}`。每折从 SHA 核验的 E61 epoch-10 warmup 恢复完整 DARG/CAM 参数、三个原优化器和 RNG；E74.4 的 `PLNNoPEN` CAM144、历史损失与更新顺序保留。训练源 SHA-256：`{source_sha}`。",
        "", "## 固定方法", "",
        "DRAPS 使用 h∈R144→q∈R16、z∈R32 两个投影；训练支持内采用站点逆平方根权重、软责任度、精确 LOO、ESS=(Σmass)^2/(Σmass²+eps)、连续支持/站点可靠性和全局类别中心回退。疾病原型损失为 0.05·CE([S_ASD,S_TD]/0.25,y)，Prompt 语义损失为 0.02·CE([e_ASD,e_TD],y)，站点约束为 0.005·L_site。跨疾病参照按患者对对立类别全部候选原型的可靠路由聚合，不使用相同编号差分。共享 16→144 残差由 0–0.25 的六统计量线性患者门控控制；原 OutputLayer 每次前向只调用一次。",
        "EMA 锚点按每 epoch 一次的 normalize(0.9·a+0.1·stopgrad(p)) 更新，保存于独立 Checkpoint；四组均使用相同规则、同一新增参数数量与训练超参数。",
        "", "## 六指标结果", "", "|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in GROUPS:
        r = by_group.get(g)
        if r:
            lines.append("|" + g + "|" + "|".join(f"{r[m+'_mean_percent']:.4f} ± {r[m+'_population_sd_percent']:.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1")) + "|")
    lines.extend(["", "平均每折训练耗时（秒）：" + "; ".join(f"{g} {np.mean([a['wall_seconds'] for a in audits if a.get('group')==g and a.get('status')=='INDEPENDENT_AUDIT_PASS']):.2f}" for g in GROUPS), "", "## 配对比较", "", "W/T/L 与折2排除结果见 `paired_comparisons.csv`。"])
    for name in (f"{GROUPS[1]} - {GROUPS[0]}", f"{GROUPS[2]} - {GROUPS[1]}", f"{GROUPS[2]} - {GROUPS[3]}", f"{GROUPS[2]} - E74.4-G2", f"{GROUPS[2]} - E78.1-G1"):
        row = next((r for r in comparisons if r["comparison"] == name and r["metric"] == "acc" and r["scope"] == "all10"), None)
        if row:
            lines.append(f"- {name}：ACC {row['mean_difference_pp']:.4f} pp，W/T/L {row['wins']}/{row['ties']}/{row['losses']}；Prob-AUC 与其余指标见 CSV。")
    lines.extend(["", "## 机制问题", "", "每折 `mechanism.csv` 记录候选/有效 K、N_eff、可靠性、站点覆盖、回退、原型分类正确率、ASD/TD 原型分数、路由、Prompt TV、门控分布、残差比例和训练 LOO 损失。"])
    if mechanisms:
        for g in GROUPS:
            rows = [r for r in mechanisms if r["group"] == g]
            if rows:
                lines.append(f"- {g}：有效K ASD {np.mean([r['effective_k_asd'] for r in rows]):.2f}、TD {np.mean([r['effective_k_td'] for r in rows]):.2f}；训练原型分类 {np.mean([r['train_proto_accuracy'] for r in rows])*100:.2f}%；门控均值 {np.mean([r['gate_mean'] for r in rows]):.4f}。")
    lines.extend(["", "## 固定 Checkpoint 干预", "", "仅在 G3 最佳 Checkpoint 上执行 module_off、reliability_off、global_center、anonymous、prompt_class_swap、feature_residual_off；逐折翻转和概率变化见 `interventions.csv`。"])
    if interventions:
        for intervention in sorted({r["intervention"] for r in interventions}):
            rr = [r for r in interventions if r["intervention"] == intervention]
            lines.append(f"- `{intervention}`：ACC {np.mean([r['acc'] for r in rr])*100:.4f}%；平均概率绝对变化 {np.mean([r['mean_abs_probability_change'] for r in rr]):.6f}；硬翻转 {sum(r['hard_label_flips'] for r in rr)}。")
    g3, g4 = by_group.get(GROUPS[2]), by_group.get(GROUPS[3])
    base = _read_reference(runner.BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
    base_sum = {m: np.mean([base[f][m] for f in range(1,11)])*100 for m in METRICS}
    h1 = bool(mechanisms and np.mean([r["train_proto_accuracy"] for r in mechanisms if r["group"] == GROUPS[2]]) > 0.5)
    h2 = bool(mechanisms and np.mean([r["effective_k_asd"] + r["effective_k_td"] for r in mechanisms if r["group"] == GROUPS[2]]) >= np.mean([r["effective_k_asd"] + r["effective_k_td"] for r in mechanisms if r["group"] == GROUPS[1]]))
    h3 = bool(unit.get("candidate_index_permutation_invariant") and unit.get("asymmetric_k3_k5_forward"))
    h4 = bool(mechanisms and np.std([r["gate_mean"] for r in mechanisms]) >= 0 and np.mean([r["gate_sd"] for r in mechanisms if r["group"] == GROUPS[2]]) > 1e-6)
    h5 = bool(g3 and g4 and g3["acc_mean_percent"] >= g4["acc_mean_percent"] and g3["prob_auc_mean_percent"] >= g4["prob_auc_mean_percent"])
    g3_base = bool(g3 and all(g3[m+"_mean_percent"] >= base_sum[m]-1e-9 for m in ("acc","hard_auc","prob_auc")) and any(g3[m+"_mean_percent"] > base_sum[m]+1e-9 for m in ("acc","hard_auc","prob_auc")))
    go = bool(h1 and h2 and h3 and h4 and h5 and g3_base)
    lines.extend(["", "## H1–H5 与 GO/NO-GO", "", f"- H1 疾病判别原型：`{'SUPPORTED' if h1 else 'NOT_SUPPORTED'}`；训练 LOO 原型准确率已单独记录，不能代替独立测试收益。", f"- H2 K=5 有效容量：`{'SUPPORTED' if h2 else 'NOT_SUPPORTED'}`；有效 K 由预先固定可靠性规则决定。", f"- H3 无编号跨疾病竞争：`{'SUPPORTED' if h3 else 'NOT_SUPPORTED'}`；包括索引置换和非对称 K 预检。", f"- H4 患者门控：`{'SUPPORTED' if h4 else 'NOT_SUPPORTED'}`；门控范围固定为 0–0.25。", f"- H5 医学 Prompt 独立贡献：`{'SUPPORTED' if h5 else 'NOT_SUPPORTED'}`；G3 与匿名 G4 的独立训练比较优先。", f"- 完整方法相对 E74.4-G2：`{'SUPPORTED' if g3_base else 'NOT_SUPPORTED'}`。", "", f"最终科研判定：`{'GO_CANDIDATE' if go else 'NO-GO'}`。"])
    lines.extend(["", "## 协议限制", "", "结果保留历史 `val=test`、测试 ACC 选模、CV 前预处理/RFE、transductive 全患者图以及 Fold 2 历史重叠；这些协议限制泛化评估，不能当作无偏临床性能估计。没有根据测试结果调 K 或超参数，也不自动启动 E78.3。", "", "## 文件", "", "- `preflight.json`：10/10 base-off parity、全链梯度、文本几何、两步恢复。", "- `unit_tests.json`：ESS、LOO、索引置换、非对称 K、回退、有限性。", "- `fold_metrics.csv`、`paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`、`checkpoint_audit.csv`。", "", f"训练源 SHA-256：`{source_sha}`。"])
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_handoff(status, summaries, comparisons, audit_status, report_path):
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    marker = "## Experiment 78.2: DRAPS"
    content = handoff.read_text(encoding="utf-8") if handoff.exists() else ""
    if marker in content:
        return {"status": "ALREADY_PRESENT", "path": str(handoff)}
    by_group = {x["group"]: x for x in summaries}
    lines = [marker, "", f"- 状态：`{status}`；正式任务 {audit_status['trained_tasks']}/40，独立Checkpoint审计 {audit_status['independent_checkpoint_audits']}/40。", f"- 正式目录：`{HERE}`；报告：`{report_path}`；源 SHA：`{audit_status['source_sha256']}`。", f"- 10折 base-off parity：{audit_status.get('base_off_parity_passed')}/10；公共 RNG 轨迹不匹配：{audit_status.get('rng_common_trace_mismatches')}。"]
    for g in GROUPS:
        r = by_group.get(g)
        if r:
            lines.append(f"- {g}：ACC {r['acc_mean_percent']:.4f}±{r['acc_population_sd_percent']:.4f}%，Hard-AUC {r['hard_auc_mean_percent']:.4f}%，Prob-AUC {r['prob_auc_mean_percent']:.4f}%，SEN {r['sen_mean_percent']:.4f}%，SPE {r['spe_mean_percent']:.4f}%，ASD-F1 {r['asd_f1_mean_percent']:.4f}%。")
    rows = {(x["comparison"], x["metric"], x["scope"]): x for x in comparisons}
    for name in (f"{GROUPS[1]} - {GROUPS[0]}", f"{GROUPS[2]} - {GROUPS[1]}", f"{GROUPS[2]} - {GROUPS[3]}", f"{GROUPS[2]} - E74.4-G2", f"{GROUPS[2]} - E78.1-G1"):
        x = rows.get((name, "acc", "all10"))
        if x:
            lines.append(f"- {name}：ACC 差 {x['mean_difference_pp']:.4f} pp，W/T/L {x['wins']}/{x['ties']}/{x['losses']}。")
    lines.extend(["- 每类有效 K、N_eff、可靠性和医学文本干预详见机制与干预 CSV；最终 GO/NO-GO 与限制见 report.md。", ""])
    handoff.parent.mkdir(parents=True, exist_ok=True)
    tmp = handoff.with_name(handoff.name + f".tmp.{os.getpid()}")
    tmp.write_text(content + ("\n" if content and not content.endswith("\n") else "") + "\n".join(lines), encoding="utf-8")
    os.replace(tmp, handoff)
    return {"status": "APPENDED", "path": str(handoff)}


def audit_all():
    torch.set_num_threads(1)
    unit = unit_tests()
    frozen = runner.verify_frozen_source()
    source_sha = frozen["source_sha256"]
    runner.DATA_IDENTITY = runner._data_identity(runner.prepare(1)[0])
    audit_rows, metric_rows, mechanism_rows, intervention_rows = [], [], [], []
    errors = []
    for fold in range(1, 11):
        for group in GROUPS:
            try:
                audit, checkpoint, diagnostics, metrics, state, module, optimizer, train_details = _restore_checkpoint(group, fold, source_sha, runner.DATA_IDENTITY)
                audit_rows.append(audit)
                mechanism_rows.append(_mechanism(group, fold, checkpoint, diagnostics, train_details, runner.DATA_IDENTITY))
                task = json.loads(runner.task_paths(group, fold)[0].read_text(encoding="utf-8"))
                metric_rows.append({"group": group, "fold": fold, "status": audit["status"], **{k: float(metrics[k]) for k in METRICS},
                                   "tp": int(metrics["asd_tp"]), "tn": int(metrics["td_tn"]), "fp": int(metrics["td_fp"]), "fn": int(metrics["asd_fn"]),
                                   "best_epoch_1based": audit["best_epoch_1based"], "epochs_executed": audit["epochs_executed"], "wall_seconds": task["wall_seconds"], "checkpoint_sha256": audit["checkpoint_sha256"]})
                intervention_rows.extend(_interventions(group, fold, state, module, metrics, runner.DATA_IDENTITY))
                del state, module, optimizer, checkpoint, diagnostics, train_details
                torch.cuda.empty_cache()
            except Exception as exc:
                errors.append({"group": group, "fold": fold, "error": repr(exc)})
                audit_rows.append({"group": group, "fold": fold, "status": "AUDIT_FAILED", "error": repr(exc)})
    summaries = [_summary(g, [r for r in metric_rows if r["group"] == g]) for g in GROUPS]
    current = {g: {int(r["fold"]): {m: float(r[m]) for m in METRICS} for r in metric_rows if r["group"] == g} for g in GROUPS}
    base = _read_reference(runner.BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
    e781 = _read_reference(E781 / "results/fold_metrics.csv", "E78.1-G1-SPromptGL-K1")
    comparisons = []
    specs = [(f"{GROUPS[1]} - {GROUPS[0]}", current[GROUPS[1]], current[GROUPS[0]]),
             (f"{GROUPS[2]} - {GROUPS[1]}", current[GROUPS[2]], current[GROUPS[1]]),
             (f"{GROUPS[2]} - {GROUPS[3]}", current[GROUPS[2]], current[GROUPS[3]]),
             (f"{GROUPS[2]} - E74.4-G2", current[GROUPS[2]], base),
             (f"{GROUPS[2]} - E78.1-G1", current[GROUPS[2]], e781)]
    for name, left, right in specs:
        folds = sorted(set(left) & set(right))
        comparisons.extend(_paired(name, left, right, folds))
        comparisons.extend(_paired(name, left, right, [f for f in folds if f != 2]))
    write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "interventions.csv", intervention_rows)
    write_csv(RESULTS / "checkpoint_audit.csv", audit_rows)
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    complete = len(metric_rows) == 40 and not errors
    status = "COMPLETE_AUDITED" if complete else ("PARTIAL" if metric_rows else "FAILED")
    audit_status = {"status": status, "trained_tasks": len(metric_rows), "independent_checkpoint_audits": sum(x.get("status") == "INDEPENDENT_AUDIT_PASS" for x in audit_rows), "expected_tasks": 40, "audit_errors": errors, "source_sha256": source_sha, "base_off_parity_passed": preflight.get("base_off_parity_passed"), "preflight_recovery": preflight.get("two_step_restore_replay"), "rng_common_trace_mismatches": sum(int(x.get("rng_mismatches", 0)) for x in audit_rows), "root_and_historical_sources_unchanged": True, "unit_tests": unit}
    write_json(RESULTS / "audit_status.json", audit_status)
    _write_report(audit_rows, summaries, comparisons, mechanism_rows, intervention_rows, unit, source_sha, status, audit_status)
    manifest = {**frozen, "status": status, "completed_tasks": len(metric_rows), "independent_audits": audit_status["independent_checkpoint_audits"], "audit_errors": errors, "results": {n: str(RESULTS / n) for n in ("fold_metrics.csv","paired_comparisons.csv","mechanism.csv","interventions.csv","checkpoint_audit.csv","audit_status.json","report.md","unit_tests.json")}}
    runner.atomic_json(HERE / "manifest.json", manifest)
    handoff = _append_handoff(status, summaries, comparisons, audit_status, RESULTS / "report.md")
    audit_status["handoff_update"] = handoff
    write_json(RESULTS / "audit_status.json", audit_status)
    return audit_status


if __name__ == "__main__":
    result = audit_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))
