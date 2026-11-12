#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency, t

import run as runner

HERE = runner.HERE
RESULTS = runner.RESULTS
GROUPS = runner.GROUPS
METRICS = runner.METRICS
BASE = runner.BASE
E781 = Path("/model/spromptgl/experiment/experiment78_1_sh_dcpss")


def _np(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


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


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    runner.atomic_json(path, value)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_match(actual, expected, tol=1e-8):
    return all(abs(float(actual[k]) - float(expected[k])) <= tol for k in METRICS)


def _unit_tests():
    torch.manual_seed(784)
    coords = torch.randn(2, 8, 16)
    classes = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    h = torch.randn(871, 144)
    labels = torch.tensor(([0, 1] * 435) + [0])
    support = torch.arange(0, 700, dtype=torch.long)
    # Inference queries are disjoint from the legal support bank so flipping
    # their labels cannot alter support construction.
    query = torch.arange(700, 740, dtype=torch.long)
    train_query = support[:40]
    weights = torch.ones(2)
    base_logp = torch.log_softmax(torch.randn(871, 2), dim=-1)
    tests = {}
    outputs = {}
    for k in (1, 2):
        for mode in ("restricted_residual", "direct"):
            m = runner.initialize(coords, classes, k, "spromptgl", mode, 1, torch.device("cpu"))
            o = m(h, labels, support, train_query, weights, base_logp, training=True)
            key = f"k{k}_{mode}"
            outputs[key] = (m, o)
            tests[f"{key}_finite"] = finite_tree(o)
            tests[f"{key}_shape"] = tuple(o["final_logp"].shape) == (40, 2)
            ids0 = support[labels[support] == 0]
            ids1 = support[labels[support] == 1]
            tests[f"{key}_attention_axis"] = bool(
                torch.allclose(o["proto_attention"][0, :k][:, ids0].sum(-1), torch.ones(k), atol=1e-5)
                and torch.allclose(o["proto_attention"][1, :k][:, ids1].sum(-1), torch.ones(k), atol=1e-5)
            )
    # K=2 query-token permutation preserves the unordered class score set.
    m, o = outputs["k2_restricted_residual"]
    mp = runner.initialize(coords, classes, 2, "spromptgl", "restricted_residual", 1, torch.device("cpu"))
    mp.load_state_dict(m.state_dict(), strict=True)
    with torch.no_grad():
        perm = torch.tensor([1, 0])
        mp.query_offset[0].copy_(mp.query_offset[0, perm])
        mp.query_offset[1].copy_(mp.query_offset[1, perm])
    op = mp(h, labels, support, train_query, weights, base_logp, training=True)
    tests["k2_prototype_token_permutation_invariant"] = bool(torch.allclose(o["final_logp"], op["final_logp"], atol=2e-6, rtol=0))
    # Patient/support order invariance is explicit because the implementation
    # canonicalizes support IDs and the attention softmax runs over patients.
    shuffled = support[torch.randperm(len(support), generator=torch.Generator().manual_seed(17))]
    osup = m(h, labels, shuffled, train_query, weights, base_logp, training=True)
    tests["support_order_invariant"] = bool(torch.allclose(o["final_logp"], osup["final_logp"], atol=2e-6, rtol=0))
    # LogMeanExp count normalization: duplicate identical candidates do not
    # increase the pooled score when K=2.
    sim = torch.tensor([[[0.2, 0.2], [0.5, -0.1]]])
    mask1 = torch.tensor([[True, False], [True, False]])
    mask2 = torch.tensor([[True, True], [True, True]])
    p1 = m._pooled_scores(sim, mask1)
    p2 = m._pooled_scores(torch.tensor([[[0.2, 0.2], [0.5, -0.1]]]), mask2)
    tests["logmeanexp_count_normalized"] = abs(float(p1[0, 0] - p2[0, 0])) < 1e-7
    # Test labels cannot affect prototype inference because only support labels
    # are consumed by the module.
    flipped = labels.clone(); flipped[query] = 1 - flipped[query]
    of = m(h, flipped, support, query, weights, base_logp, training=False)
    oe = m(h, labels, support, query, weights, base_logp, training=False)
    tests["test_label_invariant"] = bool(torch.equal(of["final_logp"], oe["final_logp"]))
    tests["all_finite"] = all(v is True for v in tests.values())
    write_json(RESULTS / "unit_tests.json", {"status": "PASS" if tests["all_finite"] else "FAIL", "tests": tests})
    if not tests["all_finite"]:
        raise RuntimeError(f"PSP-AL unit tests failed: {tests}")
    return tests


def _restore_checkpoint(group, fold, frozen_sha, data):
    task_file, best_file, latest_file, _ = runner.task_paths(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("source_sha256") != frozen_sha:
        raise RuntimeError("task is not complete under frozen source")
    if task.get("checkpoint_sha256") != runner.sha_file(best_file):
        raise RuntimeError("best Checkpoint SHA differs from task")
    if not latest_file.is_file():
        raise RuntimeError("latest recovery Checkpoint is missing")
    checkpoint = torch.load(best_file, map_location="cpu", weights_only=False)
    latest = torch.load(latest_file, map_location="cpu", weights_only=False)
    if checkpoint.get("source_sha256") != frozen_sha or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError("best Checkpoint identity mismatch")
    if latest.get("source_sha256") != frozen_sha or latest.get("group") != group or latest.get("fold") != fold:
        raise RuntimeError("latest Checkpoint identity mismatch")
    state, warmup, identity = runner.prepare(fold)
    if checkpoint.get("identity") != identity or task.get("identity") != identity:
        raise RuntimeError("E74.4 identity mismatch")
    if checkpoint["initialization"]["original_optimizer_sha256"] != identity["optimizer_sha"]:
        raise RuntimeError("original optimizer inheritance mismatch")
    support = runner.support_ids(state)
    cp_support = np.asarray(checkpoint["initialization"]["support_ids"], dtype=np.int64)
    if not np.array_equal(support, cp_support) or np.intersect1d(support, state["test_np"]).size:
        raise RuntimeError("illegal new support identities")
    overlap = np.intersect1d(state["train_np"], state["test_np"])
    expected_overlap = np.asarray([376, 406], dtype=np.int64) if fold == 2 else np.zeros(0, dtype=np.int64)
    if not np.array_equal(overlap, expected_overlap) or np.intersect1d(overlap, support).size:
        raise RuntimeError("Fold 2 overlap entered new support")
    history = checkpoint.get("loss_history", [])
    if not history:
        raise RuntimeError("loss history missing")
    max_acc = max(float(row["metrics"]["acc"]) for row in history)
    first_best = next(int(row["epoch"]) for row in history if float(row["metrics"]["acc"]) == max_acc)
    if int(checkpoint["best_epoch_1based"]) != first_best or int(task["best_epoch_1based"]) != first_best:
        raise RuntimeError("strict first-maximum ACC rule mismatch")
    spec = runner.CFG["group_specs"][group]
    module = runner._make_module(state, group)
    optimizer = runner._new_optimizer(module)
    module.load_state_dict(checkpoint["best_state"]["module_state"], strict=True)
    hist = runner.hist(); hist.load_bundle(state, checkpoint["best_state"]["base_state"])
    optimizer.load_state_dict(checkpoint["best_state"]["new_optimizer"])
    if runner.sha_object(optimizer.state_dict()) != runner.sha_object(checkpoint["best_state"]["new_optimizer"]):
        raise RuntimeError("new optimizer did not restore exactly")
    if checkpoint["initialization"]["module_parameter_count"] != runner.CFG["method"]["trainable_parameters_per_group"]:
        raise RuntimeError("module parameter count mismatch")
    if not finite_tree(checkpoint["best_state"]["module_state"]):
        raise RuntimeError("non-finite module state")
    runner.runtime().restore_rng(checkpoint["best_rng"]["packed"])
    module.eval()
    with torch.no_grad():
        out, details = runner._forward(state, module, training=False, support=support, query=state["test_np"])
        metrics = runner._final_metrics(out, details, state)
        base_metrics = runner._metrics(out["logp"], state)
    if not np.array_equal(np.asarray(metrics["pred"]), np.asarray(checkpoint["test_prediction"])):
        raise RuntimeError("final prediction readback mismatch")
    if not np.allclose(np.asarray(metrics["prob"]), np.asarray(checkpoint["test_probability"]), atol=1e-7, rtol=0):
        raise RuntimeError("final probability readback mismatch")
    if not _metric_match(metrics, checkpoint["metrics"]) or not _metric_match(metrics, task["metrics"]):
        raise RuntimeError("final metrics readback mismatch")
    if not _metric_match(base_metrics, checkpoint["base_metrics"]):
        raise RuntimeError("base metrics readback mismatch")
    eval_rng = runner.runtime().rng_sha(runner.runtime().pack_rng())
    if eval_rng != checkpoint["best_rng"]["rng_after_eval"]:
        raise RuntimeError("evaluation consumed unexpected RNG")
    # Test-label invariance.
    original = state["y"].detach().clone()
    test_ids = torch.as_tensor(state["test_np"], dtype=torch.long, device=state["y"].device)
    with torch.no_grad():
        state["y"][test_ids] = 1 - state["y"][test_ids]
        flipped_out, flipped = runner._forward(state, module, training=False, support=support, query=state["test_np"])
        state["y"].copy_(original)
    if not torch.equal(details["final_logp"], flipped["final_logp"]):
        raise RuntimeError("test-label flip changed final inference")
    # No-LOO training path: query identities are retained in the support bank.
    with torch.no_grad():
        train_out, train_details = runner._forward(state, module, training=True, support=support, query=support)
    if not torch.equal(train_details["support_ids"], torch.as_tensor(support, device=state["y"].device)):
        raise RuntimeError("training support IDs changed")
    if not finite_tree(train_details):
        raise RuntimeError("non-finite training diagnostics")
    # Common public RNG trace against E74.4.
    base_task = json.loads((BASE / ".runtime/tasks" / f"{runner.BASE_GROUP}_f{fold:02d}.json").read_text())
    base_trace = {int(x["epoch"]): x for x in base_task["rng_trace"]}
    new_trace = {int(x["epoch"]): x for x in checkpoint["rng_trace"]}
    common = sorted(set(base_trace) & set(new_trace))
    exact = sum(all(base_trace[e][key] == new_trace[e][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
    first_grad = history[0].get("gradient_audit_first_epoch", {}).get("classification", {})
    if not first_grad or min(float(first_grad.get(k, 0.0)) for k in ("module", "darg_gfr", "cam")) <= 0:
        raise RuntimeError("classification gradient did not reach module/DARG/CAM")
    audit = {
        "group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
        "checkpoint_sha256": runner.sha_file(best_file), "source_sha256": frozen_sha,
        "warmup_sha256": identity["warmup_sha256"], "initial_rng_sha": identity["initial_rng_sha"],
        "support_count": int(len(support)), "support_sha256": checkpoint["initialization"]["support_sha256"],
        "support_disjoint_from_test": True, "fold2_overlap_excluded": True,
        "all_original_optimizers_restored": True, "new_optimizer_restored": True,
        "test_label_invariant": True, "no_loo_training_query": True, "best_acc_rule_pass": True,
        "best_checkpoint_readback_metrics_match": True, "base_metrics_readback_match": True,
        "classification_mode": spec["classification_mode"], "candidate_k": int(spec["k_max"]),
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]), "epochs_executed": int(checkpoint["epochs_executed"]),
        "wall_seconds": float(task.get("wall_seconds", 0.0)),
        "peak_gpu_bytes": int(task.get("peak_gpu_bytes", 0)),
        "metrics": {k: float(metrics[k]) for k in METRICS}, "base_metrics": {k: float(base_metrics[k]) for k in METRICS},
        "rng_common_epochs": len(common), "rng_exact_epochs": exact, "rng_mismatches": len(common) - exact,
        "new_gradient_norms": first_grad,
    }
    return audit, checkpoint, state, module, optimizer, details, train_details, metrics, base_metrics


def _mechanism(group, fold, checkpoint, details, train_details, state, data):
    spec = runner.CFG["group_specs"][group]
    k = int(spec["k_max"])
    proto = _np(details["prototypes"]).astype(float)
    sim = _np(details["prototype_similarity"]).astype(float)
    patt = _np(details["proto_attention"]).astype(float)
    tatt = _np(details["text_attention"]).astype(float)
    support_ids = _np(details["support_ids"]).astype(int)
    support_labels = _np(details["support_labels"]).astype(int)
    valid = np.zeros((2, 2), dtype=bool); valid[:, :k] = True
    collapse = []
    cosine = []
    if k > 1:
        for cls in (0, 1):
            c = float(np.dot(proto[cls, 0], proto[cls, 1]))
            cosine.append(c); collapse.append(float(c > 0.98))
    neff = np.zeros((2, k)); max_attn = np.zeros((2, k)); attn_entropy = np.zeros((2, k)); support_counts = np.zeros((2, k))
    for cls in (0, 1):
        ids = support_ids[support_labels == cls]
        for j in range(k):
            w = patt[cls, j, ids]
            neff[cls, j] = 1.0 / max(float((w ** 2).sum()), 1e-12)
            max_attn[cls, j] = float(w.max())
            attn_entropy[cls, j] = float(-(w * np.log(np.maximum(w, 1e-12))).sum() / max(math.log(len(ids)), 1e-12))
            support_counts[cls, j] = float((w > 1e-8).sum())
    routes = np.exp(sim / 0.25 - np.max(sim / 0.25, axis=-1, keepdims=True))
    routes = routes / np.maximum(routes.sum(-1, keepdims=True), 1e-12)
    routes = routes[:, :, :k]
    route_entropy = -(routes * np.log(np.maximum(routes, 1e-12))).sum(-1)
    train_scores = _np(train_details["prototype_scores"])
    train_labels = _np(train_details["query_ids"])
    y_train = _np(state["y"])[train_labels].astype(int)
    train_acc = float((train_scores.argmax(1) == y_train).mean())
    return {
        "group": group, "fold": fold, "candidate_k": k,
        "classification_mode": spec["classification_mode"], "prompt_source": spec["prompt_source"],
        "mean_intra_class_prototype_cosine": float(np.mean(cosine)) if cosine else 1.0,
        "prototype_pair_collapse_fraction_cosine_gt_0_98": float(np.mean(collapse)) if collapse else 0.0,
        "prototype_cosine_by_class": json.dumps(cosine),
        "attention_effective_support_asd": json.dumps(neff[0].tolist()),
        "attention_effective_support_td": json.dumps(neff[1].tolist()),
        "attention_max_asd": json.dumps(max_attn[0].tolist()), "attention_max_td": json.dumps(max_attn[1].tolist()),
        "attention_entropy_asd": json.dumps(attn_entropy[0].tolist()), "attention_entropy_td": json.dumps(attn_entropy[1].tolist()),
        "attention_supported_patient_count_asd": json.dumps(support_counts[0].tolist()),
        "attention_supported_patient_count_td": json.dumps(support_counts[1].tolist()),
        "route_mean_asd": json.dumps(routes[:, 0].mean(0).tolist()), "route_mean_td": json.dumps(routes[:, 1].mean(0).tolist()),
        "route_entropy_mean": float(route_entropy.mean()), "route_second_candidate_usage": float(routes[:, :, 1].mean()) if k > 1 else 0.0,
        "train_prototype_accuracy": train_acc,
        "train_final_accuracy": float((_np(train_details["final_logp"]).argmax(1) == y_train).mean()),
        "lambda_value": float(_np(details["lambda_value"])), "alpha_visual": float(_np(details["alpha_visual"])),
        "alpha_text": float(_np(details["alpha_text"])),
        "support_count": int(len(support_ids)), "query_count": int(len(_np(details["query_ids"]))),
        "attention_axis_normalized": True, "no_loo": True,
    }


def _interventions(group, fold, state, module, metrics, data):
    support = runner.support_ids(state)
    before_pred = np.asarray(metrics["pred"], dtype=int); before_prob = np.asarray(metrics["prob"], dtype=float)
    labels = _np(state["y"])[state["test_np"]].astype(int)
    spec = runner.CFG["group_specs"][group]
    names = ["module_off", "uniform_attention", "anonymous", "prompt_class_swap"]
    if int(spec["k_max"]) == 2:
        names.append("prototype2_off")
    rows = []
    for intervention in names:
        module.eval()
        with torch.no_grad():
            out, details = runner._forward(state, module, training=False, support=support,
                                           query=state["test_np"], intervention=intervention)
            m = runner._final_metrics(out, details, state)
        pred = np.asarray(m["pred"], dtype=int); prob = np.asarray(m["prob"], dtype=float)
        rows.append({"group": group, "fold": fold, "intervention": intervention,
                     **{k: float(m[k]) for k in METRICS},
                     "mean_abs_probability_change": float(np.abs(prob - before_prob).mean()),
                     "max_abs_probability_change": float(np.abs(prob - before_prob).max()),
                     "hard_label_flips": int((pred != before_pred).sum()),
                     "correct_to_wrong": int(((before_pred == labels) & (pred != labels)).sum()),
                     "wrong_to_correct": int(((before_pred != labels) & (pred == labels)).sum())})
    return rows


def _read_reference(path, group):
    rows = [r for r in read_csv(path) if r.get("group") == group]
    if len(rows) != 10: raise RuntimeError(f"reference {group} does not have 10 folds")
    return {int(r["fold"]): {k: float(r[k]) for k in METRICS} for r in rows}


def _summary(group, rows):
    out = {"group": group, "folds": len(rows)}
    for m in METRICS:
        vals = np.asarray([float(r[m]) for r in rows]) * 100.0
        out[f"{m}_mean_percent"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{m}_population_sd_percent"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
    out["success_count"] = len(rows)
    return out


def _paired(name, left, right, folds):
    rows = []
    scope = "all10" if len(folds) == 10 else "exclude_fold2"
    for m in METRICS:
        d = np.asarray([(left[f][m] - right[f][m]) * 100.0 for f in folds])
        mean = float(d.mean()) if len(d) else float("nan")
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        half = float(t.ppf(.975, len(d)-1) * sd / math.sqrt(len(d))) if len(d) > 1 else 0.0
        rows.append({"comparison": name, "scope": scope, "metric": m, "n_folds": len(folds),
                     "mean_difference_pp": mean, "population_sd_difference_pp": float(d.std(ddof=0)) if len(d) else float("nan"),
                     "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum()),
                     "descriptive_95ci_low_pp": mean-half, "descriptive_95ci_high_pp": mean+half,
                     "fold_differences_pp": json.dumps({str(f): float((left[f][m]-right[f][m])*100.0) for f in folds}, sort_keys=True)})
    return rows


def _write_report(audits, summaries, comparisons, mechanisms, interventions, unit, source_sha, status, audit_status):
    by_group = {r["group"]: r for r in summaries}
    lines = ["# Experiment 78.4 — PSP-AL results", "",
             f"状态：`{status}`；正式目标60折，任务完成{audit_status['trained_tasks']}/60，独立最佳Checkpoint审计{audit_status['independent_checkpoint_audits']}/60。",
             f"训练源SHA：`{source_sha}`。本实验使用E74.4-G2 Positive-DCR/DARG + CAM144；preferred OSSFS目录errno=5，正式输出统一在`{HERE}`。",
             "", "## 论文与实现审计", "",
             "PPAL论文第2.3节及公式8–12、官方ppal.py/attention_layer.py/engine3.py/textprompt.csv已读。公开engine3实际只优化一份CrossEntropy，论文的粗/细联合损失与代码不完全一致；ppal.py的512维MultiheadAttention布局没有显式batch_first，PSP-AL改为显式[batch,query,dim]和[batch,patient,dim]的32维4-head注意力。SPromptGL和服务器E74.4真实run.py/soft_positive_dcr.py也已核验，原始总损失准确替换了`out[cls]`分类贡献。详见`design.md`。",
             "", "## 六指标（均值±population SD，%）", "",
             "|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功任务|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        r = by_group.get(g)
        if r:
            lines.append("|" + g + "|" + "|".join(f"{r[m+'_mean_percent']:.4f} ± {r[m+'_population_sd_percent']:.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1")) + f"|{r['success_count']}|")
    base = _read_reference(BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
    e781 = _read_reference(E781 / "results/fold_metrics.csv", "E78.1-G1-SPromptGL-K1")
    def refline(name, d):
        s = _summary(name, [{"group": name, **d[f]} for f in sorted(d)])
        return "|" + name + "|" + "|".join(f"{s[m+'_mean_percent']:.4f} ± {s[m+'_population_sd_percent']:.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1")) + "|10|"
    lines += ["", "历史参考：", "|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|", "|---|---:|---:|---:|---:|---:|---:|---:|", refline("E74.4-G2", base), refline("E78.1-G1", e781)]
    lines += ["", "## 机制与干预", "", "`mechanism.csv`记录原型相似度/坍塌、患者注意力有效支持数、最大注意力、熵、原型路由、训练自匹配偏差及K=1/K=2差异；`interventions.csv`记录DARG-only、均匀患者聚合、匿名Prompt、Prompt类别交换及K=2关闭第二原型。"]
    for g in GROUPS:
        rs = [r for r in mechanisms if r["group"] == g]
        if rs:
            lines.append(f"- {g}：训练原型分类 {np.mean([r['train_prototype_accuracy'] for r in rs])*100:.2f}%；原型坍塌比例 {np.mean([r['prototype_pair_collapse_fraction_cosine_gt_0_98'] for r in rs])*100:.2f}%；注意力有效支持均值 {np.mean([np.mean(json.loads(r['attention_effective_support_asd'])+json.loads(r['attention_effective_support_td'])) for r in rs]):.2f}。")
    for intervention in sorted(set(r["intervention"] for r in interventions)):
        rs = [r for r in interventions if r["intervention"] == intervention]
        lines.append(f"- `{intervention}`：平均硬翻转 {sum(r['hard_label_flips'] for r in rs)}；平均概率绝对变化 {np.mean([r['mean_abs_probability_change'] for r in rs]):.6f}。")
    lines += ["", "## 配对比较", "", "完整六指标逐折差、W/T/L、描述性95% CI及Fold 2排除结果见`paired_comparisons.csv`。"]
    for row in comparisons:
        if row["metric"] == "acc" and row["scope"] == "all10":
            lines.append(f"- {row['comparison']}：ACC差 {row['mean_difference_pp']:+.4f} pp，W/T/L {row['wins']}/{row['ties']}/{row['losses']}。")
    # Pre-registered reporting rule: best medical group means highest mean ACC
    med = [g for g in GROUPS[:4] if g in by_group]
    best_med = max(med, key=lambda g: (by_group[g]["acc_mean_percent"], by_group[g]["prob_auc_mean_percent"])) if med else None
    lines += ["", "## GO / NO-GO", ""]
    if best_med:
        lines.append(f"最高医学组（按十折ACC、同分时Prob-AUC）：`{best_med}`。")
    base_sum = _summary("E74.4-G2", base)
    med_gain = bool(best_med and by_group[best_med]["acc_mean_percent"] >= base_sum["acc_mean_percent"] and by_group[best_med]["hard_auc_mean_percent"] >= base_sum["hard_auc_mean_percent"])
    prompt_rows = {(r["comparison"], r["metric"], r["scope"]): r for r in comparisons}
    p1 = prompt_rows.get((f"{GROUPS[0]} - {GROUPS[4]}", "acc", "all10"), {"mean_difference_pp": 0})["mean_difference_pp"]
    p2 = prompt_rows.get((f"{GROUPS[1]} - {GROUPS[5]}", "acc", "all10"), {"mean_difference_pp": 0})["mean_difference_pp"]
    prompt_gain = p1 > 0 or p2 > 0
    collapse_mean = float(np.mean([r["prototype_pair_collapse_fraction_cosine_gt_0_98"] for r in mechanisms if r["group"] in GROUPS[2:4]])) if mechanisms else 0.0
    mechanism_ok = collapse_mean < 0.5
    go = bool(med_gain and prompt_gain and mechanism_ok)
    lines += [f"- 基座分类增益：`{'SUPPORTED' if med_gain else 'NOT_SUPPORTED'}`。", f"- 医学Prompt相对匿名：`{'SUPPORTED' if prompt_gain else 'NOT_SUPPORTED'}`；固定Checkpoint干预需结合硬翻转解释。", f"- K=2互异原型机制：`{'SUPPORTED' if mechanism_ok else 'NOT_SUPPORTED'}`。", f"- 最终科研判定：`{'GO_CANDIDATE' if go else 'NO-GO'}`。"]
    lines += ["", "## 协议限制", "", "保留历史val=test、测试ACC选模、CV前预处理/RFE、transductive全患者图以及Fold 2重叠；这些限制不构成无偏临床泛化评估。", "", "## 文件", "", "`preflight.json`、`unit_tests.json`、`fold_metrics.csv`、`paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`、`checkpoint_audit.csv`、`audit_status.json`。"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_med


def _append_handoff(status, summaries, audit_status, source_sha, best_med):
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    marker = "## Experiment 78.4: PSP-AL"
    content = handoff.read_text(encoding="utf-8") if handoff.exists() else ""
    if marker in content:
        return {"status": "ALREADY_PRESENT", "path": str(handoff)}
    lines = [marker, "", f"- 状态：`{status}`；正式任务{audit_status['trained_tasks']}/60，独立最佳Checkpoint审计{audit_status['independent_checkpoint_audits']}/60。", f"- 正式目录：`{HERE}`；训练源SHA：`{source_sha}`；preferred OSSFS errno=5，使用/model回退。", f"- Base-off parity：{audit_status.get('base_off_parity_passed')}/10；公共RNG共有epoch核对：{audit_status.get('rng_common_epochs_checked', 0)}，不匹配：{audit_status.get('rng_mismatches', 0)}。"]
    by = {r["group"]: r for r in summaries}
    for g in GROUPS:
        if g in by:
            r = by[g]
            lines.append(f"- {g}：ACC {r['acc_mean_percent']:.4f}±{r['acc_population_sd_percent']:.4f}%，Hard-AUC {r['hard_auc_mean_percent']:.4f}%，Prob-AUC {r['prob_auc_mean_percent']:.4f}%，SEN {r['sen_mean_percent']:.4f}%，SPE {r['spe_mean_percent']:.4f}%，ASD-F1 {r['asd_f1_mean_percent']:.4f}%。")
    lines += [f"- 最高医学组（按ACC）：`{best_med}`。", "- K=1/K=2、受限残差/直接分类、医学/匿名Prompt、注意力与原型干预、Fold 2敏感性及限制详见E78.4 results/report.md。", ""]
    tmp = handoff.with_name(handoff.name + f".tmp.{os.getpid()}")
    tmp.write_text(content + ("\n" if content and not content.endswith("\n") else "") + "\n".join(lines), encoding="utf-8")
    os.replace(tmp, handoff)
    return {"status": "APPENDED", "path": str(handoff)}


def audit_all():
    torch.set_num_threads(1)
    unit = _unit_tests()
    frozen = runner.verify_frozen_source(); source_sha = frozen["source_sha256"]
    runner.DATA_IDENTITY = runner._data_identity(runner.prepare(1)[0])
    audits, metric_rows, mechanism_rows, intervention_rows, errors = [], [], [], [], []
    for fold in range(1, 11):
        for group in GROUPS:
            try:
                audit, checkpoint, state, module, optimizer, details, train_details, metrics, base_metrics = _restore_checkpoint(group, fold, source_sha, runner.DATA_IDENTITY)
                audits.append(audit)
                mechanism_rows.append(_mechanism(group, fold, checkpoint, details, train_details, state, runner.DATA_IDENTITY))
                task = json.loads(runner.task_paths(group, fold)[0].read_text(encoding="utf-8"))
                metric_rows.append({"group": group, "fold": fold, "status": audit["status"], **{k: float(metrics[k]) for k in METRICS},
                                   "tp": int(metrics["asd_tp"]), "tn": int(metrics["td_tn"]), "fp": int(metrics["td_fp"]), "fn": int(metrics["asd_fn"]),
                                   "best_epoch_1based": audit["best_epoch_1based"], "epochs_executed": audit["epochs_executed"],
                                   "wall_seconds": task["wall_seconds"], "peak_gpu_bytes": task.get("peak_gpu_bytes", 0),
                                   "checkpoint_sha256": audit["checkpoint_sha256"]})
                intervention_rows.extend(_interventions(group, fold, state, module, metrics, runner.DATA_IDENTITY))
                del state, module, optimizer, checkpoint, details, train_details
                torch.cuda.empty_cache()
            except Exception as exc:
                errors.append({"group": group, "fold": fold, "error": repr(exc)})
                audits.append({"group": group, "fold": fold, "status": "AUDIT_FAILED", "error": repr(exc)})
    summaries = [_summary(g, [r for r in metric_rows if r["group"] == g]) for g in GROUPS]
    current = {g: {int(r["fold"]): {m: float(r[m]) for m in METRICS} for r in metric_rows if r["group"] == g} for g in GROUPS}
    base = _read_reference(BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
    e781 = _read_reference(E781 / "results/fold_metrics.csv", "E78.1-G1-SPromptGL-K1")
    specs = [
        (f"{GROUPS[0]} - E74.4-G2", current.get(GROUPS[0], {}), base),
        (f"{GROUPS[1]} - E74.4-G2", current.get(GROUPS[1], {}), base),
        (f"{GROUPS[2]} - E74.4-G2", current.get(GROUPS[2], {}), base),
        (f"{GROUPS[3]} - E74.4-G2", current.get(GROUPS[3], {}), base),
        (f"{GROUPS[0]} - {GROUPS[1]}", current.get(GROUPS[0], {}), current.get(GROUPS[1], {})),
        (f"{GROUPS[2]} - {GROUPS[3]}", current.get(GROUPS[2], {}), current.get(GROUPS[3], {})),
        (f"{GROUPS[2]} - {GROUPS[0]}", current.get(GROUPS[2], {}), current.get(GROUPS[0], {})),
        (f"{GROUPS[3]} - {GROUPS[1]}", current.get(GROUPS[3], {}), current.get(GROUPS[1], {})),
        (f"{GROUPS[0]} - {GROUPS[4]}", current.get(GROUPS[0], {}), current.get(GROUPS[4], {})),
        (f"{GROUPS[1]} - {GROUPS[5]}", current.get(GROUPS[1], {}), current.get(GROUPS[5], {})),
        (f"{GROUPS[0]} - E78.1-G1", current.get(GROUPS[0], {}), e781),
    ]
    comparisons = []
    for name, left, right in specs:
        folds = sorted(set(left) & set(right))
        if len(folds) == 10:
            comparisons.extend(_paired(name, left, right, folds))
            comparisons.extend(_paired(name, left, right, [f for f in folds if f != 2]))
    write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "interventions.csv", intervention_rows)
    write_csv(RESULTS / "checkpoint_audit.csv", audits)
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    complete = len(metric_rows) == 60 and not errors
    status = "COMPLETE_AUDITED" if complete else ("PARTIAL" if metric_rows else "FAILED")
    rng_checked = sum(int(x.get("rng_common_epochs", 0)) for x in audits)
    rng_mismatch = sum(int(x.get("rng_mismatches", 0)) for x in audits)
    audit_status = {"status": status, "trained_tasks": len(metric_rows), "independent_checkpoint_audits": sum(x.get("status") == "INDEPENDENT_AUDIT_PASS" for x in audits),
                    "expected_tasks": 60, "audit_errors": errors, "source_sha256": source_sha,
                    "base_off_parity_passed": preflight.get("base_off_parity_passed"),
                    "rng_common_epochs_checked": rng_checked, "rng_mismatches": rng_mismatch,
                    "root_and_historical_sources_unchanged": True, "unit_tests": unit}
    write_json(RESULTS / "audit_status.json", audit_status)
    best_med = _write_report(audits, summaries, comparisons, mechanism_rows, intervention_rows, unit, source_sha, status, audit_status)
    manifest = {**frozen, "status": status, "completed_tasks": len(metric_rows), "independent_audits": audit_status["independent_checkpoint_audits"],
                "audit_errors": errors, "results": {n: str(RESULTS / n) for n in ("fold_metrics.csv", "paired_comparisons.csv", "mechanism.csv", "interventions.csv", "checkpoint_audit.csv", "audit_status.json", "report.md", "unit_tests.json")}}
    runner.atomic_json(HERE / "manifest.json", manifest)
    audit_status["handoff_update"] = _append_handoff(status, summaries, audit_status, source_sha, best_med)
    write_json(RESULTS / "audit_status.json", audit_status)
    return audit_status


if __name__ == "__main__":
    print(json.dumps(audit_all(), ensure_ascii=False, indent=2))
