#!/usr/bin/env python3
"""Independent reconstruction, checkpoint audit, and fixed-checkpoint DCMP interventions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

try:
    R = sys.modules["e76_run_runtime"]
except KeyError:
    import run as R

GROUPS = tuple(R.GROUPS)
METRICS = tuple(R.METRICS)
BASE_GROUP = "E74.4-G2-Positive-DCR"
BASE_RESULTS = Path(R.CFG["base"]["path"]) / "results/fold_metrics.csv"


def _hash_tree(value) -> str:
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item, path="root"):
        if torch.is_tensor(item):
            digest.update(path.encode() + b"\0tensor\0")
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(path.encode() + b"\0ndarray\0")
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(path.encode() + b"\0dict\0")
            for key in sorted(item, key=str): visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            digest.update(path.encode() + b"\0seq\0")
            for index, child in enumerate(item): visit(child, f"{path}/{index}")
        else:
            digest.update(path.encode() + b"\0")
            digest.update(json.dumps(item, sort_keys=True, ensure_ascii=True, default=str).encode())
    visit(value)
    return digest.hexdigest()


def _metrics_row(group, fold, metrics):
    row = {"group": group, "fold": int(fold)}
    row.update({metric: float(metrics[metric]) for metric in METRICS})
    return row


def _confusion_changes(reference, candidate, labels):
    import numpy as np
    y = np.asarray(labels, dtype=np.int64)
    p0 = np.asarray(reference["pred"], dtype=np.int64)
    p1 = np.asarray(candidate["pred"], dtype=np.int64)
    correct0, correct1 = p0 == y, p1 == y
    changed = np.flatnonzero(p0 != p1)
    return {
        "hard_flips": int(changed.size),
        "correct_to_wrong": int(np.sum(correct0 & ~correct1)),
        "wrong_to_correct": int(np.sum(~correct0 & correct1)),
        "net_corrections": int(np.sum(~correct0 & correct1) - np.sum(correct0 & ~correct1)),
        "changed_global_ids_1based": (np.asarray(reference.get("test_ids", []), dtype=np.int64)[changed] + 1).tolist(),
    }


def _evaluate(state, module, memory, group, *, residual=True, confusion=None, background=None):
    import torch
    module.eval(); state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        output = R.forward_with_dcmp(state, module, memory, group, training=False,
                                     residual_enabled=residual, confusion_enabled=confusion,
                                     background_enabled=background)
    runtime = R.load_base().load_history().load_e69().load_e61().load_base()
    metrics = runtime.classification_metrics(output["logp"], state["test"], state["y"])
    return output, metrics


def _audit_fold(group: str, fold: int, preflight: dict):
    import numpy as np
    import torch
    from scipy.special import softmax

    task_file, ckpt_file = R.task_path(group, fold), R.best_path(group, fold)
    task, payload = json.loads(task_file.read_text()), R.torch_load(ckpt_file)
    expected_source = R.sha_json(R.source_inventory())
    config_sha = R.sha_file(R.EXP / "config.json")
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != R.sha_file(ckpt_file):
        raise RuntimeError(f"task/checkpoint identity mismatch {group} fold={fold}")
    for key, expected in (("source_sha256", expected_source), ("config_sha256", config_sha)):
        if task.get(key) != expected or payload.get(key) != expected:
            raise RuntimeError(f"source/config mismatch {group} fold={fold} key={key}")
    if payload.get("group") != group or int(payload.get("fold", -1)) != fold:
        raise RuntimeError(f"checkpoint group/fold mismatch {group} fold={fold}")
    state, warmup, identity, base_task = R.prepare_fold(fold)
    if payload.get("identity") != identity or task.get("identity") != identity:
        raise RuntimeError(f"E74.4-G2 warmup/split/optimizer identity mismatch {group} fold={fold}")
    if payload.get("warmup_sha256") != identity["warmup_sha256"]:
        raise RuntimeError(f"warmup SHA mismatch {group} fold={fold}")
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap or task.get("train_test_overlap_1based", expected_overlap) != expected_overlap:
        raise RuntimeError(f"unexpected fold-2 exposure {group} fold={fold}")

    R.load_bundle(state, payload["base_state"])
    runtime = R.load_base().load_history().load_e69().load_e61().load_base()
    optimizer_bundle_sha = _hash_tree(R.state_bundle(state))
    if optimizer_bundle_sha != _hash_tree(payload["base_state"]):
        raise RuntimeError(f"base model/three-optimizer state did not restore exactly {group} fold={fold}")
    bank = torch.as_tensor(preflight["text_banks"][group], dtype=torch.float32)
    module, init = R.init_dcmp(group, fold, state["y"].device, {group: bank})
    if init != payload.get("dcm_init") or init != task.get("dcm_init"):
        raise RuntimeError(f"DCMP initialization record mismatch {group} fold={fold}")
    module.load_state_dict(payload["dcmp"], strict=True)
    if R.load_dcmp().tensor_sha(module) != payload.get("dcmp_parameter_sha256"):
        raise RuntimeError(f"DCMP parameter SHA mismatch {group} fold={fold}")
    if R.load_dcmp().tensor_sha(module, include_buffers=True) != payload.get("dcmp_state_sha256"):
        raise RuntimeError(f"DCMP state SHA mismatch {group} fold={fold}")
    opt = torch.optim.AdamW(module.parameters(), lr=0.0003, weight_decay=0.001)
    opt.load_state_dict(payload["optimizer_dcmp"])
    if _hash_tree(opt.state_dict()) != _hash_tree(payload["optimizer_dcmp"]):
        raise RuntimeError(f"DCMP optimizer state did not restore exactly {group} fold={fold}")

    memory_state = payload["memory"]
    if R.load_dcmp().memory_sha(memory_state) != payload.get("memory_sha256"):
        raise RuntimeError(f"prototype memory SHA mismatch {group} fold={fold}")
    memory = R.load_dcmp().PrototypeMemory.from_state_dict(memory_state, state["y"].device)
    train = R.index_numpy(state["train"]); test = R.index_numpy(state["test"])
    legal = np.asarray(memory.metadata["eligible_global_ids_zero_based"], dtype=np.int64)
    if np.intersect1d(legal, test).size or not np.isin(legal, train).all():
        raise RuntimeError(f"prototype memory includes non-training IDs {group} fold={fold}")
    if fold == 2 and np.intersect1d(legal, np.asarray([376, 406])).size:
        raise RuntimeError("known fold-2 overlap entered DCMP memory")
    invalid = ~memory.valid
    if (memory.labels[invalid] != -1).any() or (memory.sites[invalid] != -1).any() or \
            (memory.hard_target[invalid] != -1).any() or (memory.features[invalid] != 0).any():
        raise RuntimeError(f"held-out rows are represented in persisted memory {group} fold={fold}")
    if not torch.equal(memory.ids.cpu(), torch.arange(871)):
        raise RuntimeError(f"memory global-ID map mismatch {group} fold={fold}")

    if payload.get("rng_state_at_best") is None:
        raise RuntimeError(f"best checkpoint lacks a restorable full RNG state {group} fold={fold}")
    runtime.restore_rng(payload["rng_state_at_best"])
    if runtime.rng_sha(runtime.pack_rng()) != payload.get("rng_sha_at_best"):
        raise RuntimeError(f"best checkpoint RNG state mismatch {group} fold={fold}")
    rng_before = runtime.rng_sha(runtime.pack_rng())
    full_output, metrics = _evaluate(state, module, memory, group)
    rng_after = runtime.rng_sha(runtime.pack_rng())
    if rng_before != rng_after:
        raise RuntimeError(f"evaluation changed shared RNG {group} fold={fold}")
    for metric in METRICS:
        if abs(float(metrics[metric]) - float(payload["metrics"][metric])) > 1e-10:
            raise RuntimeError(f"independent metric mismatch {group} fold={fold} metric={metric}")
    if not np.array_equal(metrics["pred"], payload["test_prediction"]):
        raise RuntimeError(f"hard predictions mismatch {group} fold={fold}")
    if not np.allclose(metrics["prob"], payload["test_probability"], atol=1e-7, rtol=0):
        raise RuntimeError(f"probabilities mismatch {group} fold={fold}")
    if task.get("metrics") != R.clean_metrics(metrics):
        for metric in METRICS:
            if abs(float(task["metrics"][metric]) - float(metrics[metric])) > 1e-10:
                raise RuntimeError(f"task metric mismatch {group} fold={fold} metric={metric}")

    ref_trace = {int(row["epoch"]): row for row in base_task["rng_trace"]}
    e76_trace = {int(row["epoch"]): row for row in payload["base_rng_trace"]}
    common = sorted(set(ref_trace) & set(e76_trace))
    exact = sum(all(e76_trace[epoch][key] == ref_trace[epoch][key]
                    for key in ("rng_before", "rng_after_train", "rng_after_eval")) for epoch in common)
    if exact != len(common) or not common:
        raise RuntimeError(f"E74.4-G2 common-epoch RNG mismatch {group} fold={fold}: {exact}/{len(common)}")

    test_ids = R.index_numpy(state["test"])
    test_y = state["y"].detach().cpu().numpy()[test_ids]
    prediction_pack = {"pred": np.asarray(metrics["pred"]), "prob": np.asarray(metrics["prob"]),
                       "test_ids": test_ids, "labels": np.asarray(test_y, dtype=np.int64)}
    intervention_rows, intervention_predictions = [], {}
    options = [("correction_off", False, None, None)]
    spec = R.CFG["groups"][group]
    if spec["confusion"]:
        options.append(("confusion_memory_off", True, False, None))
    if spec["background"]:
        options.append(("background_memory_off", True, None, False))
    for name, residual, confusion, background in options:
        alt_out, alt_metrics = _evaluate(state, module, memory, group, residual=residual,
                                         confusion=confusion, background=background)
        change = _confusion_changes(prediction_pack, alt_metrics, test_y)
        # Keep only aggregate IDs needed to check whether Fold 2's exposed rows were affected.
        change["fold2_row377_flipped"] = bool(fold == 2 and 377 in change["changed_global_ids_1based"])
        change["fold2_row407_flipped"] = bool(fold == 2 and 407 in change["changed_global_ids_1based"])
        change.pop("changed_global_ids_1based", None)
        row = {"group": group, "fold": fold, "intervention": name, **change,
               "mean_abs_asd_probability_change": float(np.abs(prediction_pack["prob"][:, 0] - alt_metrics["prob"][:, 0]).mean())}
        for metric in METRICS:
            row[f"{metric}_full"] = float(metrics[metric])
            row[f"{metric}_intervention"] = float(alt_metrics[metric])
            row[f"{metric}_delta"] = float(alt_metrics[metric] - metrics[metric])
        intervention_rows.append(row)
        intervention_predictions[name] = {"pred": np.asarray(alt_metrics["pred"]), "prob": np.asarray(alt_metrics["prob"])}

    d = full_output["dcmp"]
    ratio = d["delta"].norm(dim=-1) / full_output["cam144"].norm(dim=-1).clamp_min(1e-8)
    test_tensor = torch.as_tensor(test_ids, device=ratio.device)
    attention = d["attention"].index_select(0, test_tensor)
    roles = d["prototype_roles"]
    mechanism = {"group": group, "fold": fold, "best_epoch_1based": int(payload["best_epoch_1based"]),
        "test_rho_mean": float(ratio.index_select(0, test_tensor).mean().cpu()),
        "test_rho_p50": float(torch.quantile(ratio.index_select(0, test_tensor), .50).cpu()),
        "test_rho_p95": float(torch.quantile(ratio.index_select(0, test_tensor), .95).cpu()),
        "test_rho_max": float(ratio.index_select(0, test_tensor).max().cpu()),
        "test_gate_mean": float(d["rho"].index_select(0, test_tensor).mean().cpu()),
        "risk_mean": float(d["risk"].index_select(0, test_tensor).mean().cpu()),
        "reliability_mean": float(d["reliability"].index_select(0, test_tensor).mean().cpu()),
        "test_prediction_flips_when_correction_off": intervention_rows[0]["hard_flips"],
        "role_attention": {str(role): float(attention[:, roles == role].sum(-1).mean().cpu()) if (roles == role).any() else 0.0
                           for role in range(4)},
        "attention_entropy": float((-(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)).mean().cpu()),
        "prototype_available_count": int(d["prototype_available"].sum().item()),
        "memory_support": memory.metadata.get("typical_support", {}),
        "hard_support": memory.metadata.get("hard_support", {}),
        "site_supports": memory.metadata.get("site_supports", {}),
        "memory_train_ids": len(legal), "test_ids_excluded": len(test),
        "fold2_overlap_excluded": [377, 407] if fold == 2 else [],
        "base_rng_common_epochs": len(common), "base_rng_exact_epochs": exact,
        "base_rng_mismatches": len(common) - exact}
    audit_row = {"group": group, "fold": fold, "status": "PASS",
        "checkpoint_sha256": R.sha_file(ckpt_file), "source_sha256": expected_source,
        "warmup_sha256": identity["warmup_sha256"], "optimizer_state_sha256": optimizer_bundle_sha,
        "dcmp_optimizer_state_sha256": _hash_tree(opt.state_dict()), "memory_sha256": payload["memory_sha256"],
        "best_epoch_1based": payload["best_epoch_1based"], "hard_predictions_exact": True,
        "probability_max_abs_error": float(np.max(np.abs(np.asarray(metrics["prob"]) - payload["test_probability"]))),
        "metric_recomputation_exact": True, "rng_restored_and_eval_stable": True,
        "base_rng_common_epochs": len(common), "base_rng_exact_epochs": exact,
        "memory_test_ids_excluded": True, "fold2_overlap_memory_excluded": True,
        "checkpoint_optimizer_restore_exact": True}
    return {"audit": audit_row, "fold_metric": _metrics_row(group, fold, metrics),
            "prediction": prediction_pack, "interventions": intervention_rows, "mechanism": mechanism,
            "intervention_predictions": intervention_predictions}


def _history_rows():
    with BASE_RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    base = {}
    for row in rows:
        if row.get("group") == BASE_GROUP and row.get("status") == "INDEPENDENT_AUDIT_PASS":
            base[int(row["fold"])] = {metric: float(row[metric]) for metric in METRICS}
    if set(base) != set(range(1, 11)):
        raise RuntimeError("E74.4-G2 historical reference must contain 10 independently audited folds")
    return base


def _paired_row(name, left_name, right_name, left, right, metric, exclude_fold2=False):
    from scipy.stats import t
    folds = [fold for fold in range(1, 11) if not (exclude_fold2 and fold == 2)]
    diffs = [float(left[fold][metric] - right[fold][metric]) for fold in folds]
    n = len(diffs); mean = sum(diffs) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in diffs) / max(1, n - 1))
    sem = sd / math.sqrt(n)
    critical = float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return {"comparison": name, "left": left_name, "right": right_name, "metric": metric,
        "folds": n, "mean_delta": mean, "ci95_low_descriptive": mean - critical * sem,
        "ci95_high_descriptive": mean + critical * sem,
        "wins": sum(value > 0 for value in diffs), "ties": sum(value == 0 for value in diffs),
        "losses": sum(value < 0 for value in diffs),
        "exclude_fold2": bool(exclude_fold2), "fold_deltas": json.dumps(diffs)}


def _summary_rows(fold_metrics, history):
    rows = []
    names = {group: group for group in GROUPS}
    names[BASE_GROUP] = BASE_GROUP
    for group in list(GROUPS) + [BASE_GROUP]:
        values = [history[fold] if group == BASE_GROUP else next(
            row for row in fold_metrics if row["group"] == group and row["fold"] == fold)
                  for fold in range(1, 11)]
        row = {"group": group, "folds": 10}
        for metric in METRICS:
            samples = [float(item[metric]) for item in values]
            mean = sum(samples) / len(samples)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd_population"] = math.sqrt(sum((value - mean) ** 2 for value in samples) / len(samples))
        rows.append(row)
    return rows


def _write_report(audit_rows, fold_metrics, summary, paired, mechanisms, interventions,
                  rng_summary, started):
    import numpy as np
    report = ["# Experiment 76 — DCMP", "",
        "Status: independently audited after reconstructing all best checkpoints from disk.", "",
        "## Literature and source audit", "",
        "SPromptGL (MICCAI 2025) was checked against its paper and locked E74.4-G2 implementation. The experiment retains four-modality GFR/DARG, a single CAM pass, the 144-dimensional OutputLayer input, and the original classifier. BrainPrompt's official ABIDE labels were verified in source order TC then ASD, then mapped into the audited cache order ASD=0, TD=1; the frozen E72.4 CLIP RN50 cache is reused, so this is not a full BrainPrompt encoder reproduction. PGP-SAM motivates class-conditioned inter-/intra-class prototypes and prompt generation; its SAM segmentation decoder is not used. CoW motivates training-support self-prediction and hard prototypes; its image-mask boundary/segmentation machinery is not used. HP²L was checked by DOI metadata only; DOI resolution was unavailable in the audit browser, so no method-specific claim is based on it.",
        "", "Sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf), [BrainPrompt official ABIDE prompts](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt), [PGP-SAM paper](https://arxiv.org/abs/2501.06692), [PGP-SAM code](https://github.com/PRIS-CV/PGP-SAM), [CoW paper](https://ijcai.org/proceedings/2025/139), [CoW code](https://github.com/jcjiang99/CoW), [HP²L DOI](https://doi.org/10.1016/j.media.2026.104063).", "",
        "## Method and protocol", "",
        "The four groups share one CAM144-boundary DCMP class and common parameter initialization. G1 uses typical ASD/TD memories; G2 adds two-direction cross-fitted-probe hard memories; G3 adds class-balanced site-offset background candidates; G4 replaces the two frozen BrainPrompt vectors with norm- and cosine-matched anonymous vectors. The risk auxiliary target is the outer-training-only OOF LogisticRegression low-margin top 20% in each class; this is not strict OOF for the shared warmup encoder. Prototype membership is fixed from warmup features and vectors receive deterministic EMA updates (0.9) from detached same-step CAM144 representations. Query routing never consumes query labels, and the current training query is subtracted from its assigned typical/hard and site/global class means. Outer-test IDs and labels/features are masked from the persisted memory; Fold 2 rows 377/407 remain in the historical base training/test indices but are excluded from the added memory/risk labels.",
        "", "Checkpoint selection retains historical val=test strict-ACC selection, pre-CV RFE/standardization, and the transductive 871-node graph. Metrics are therefore paired mechanism results under a known biased protocol, not unbiased generalization estimates.", "",
        "## Results", "", "All values are mean ± population SD in percent. Hard-AUC is the historical hard-prediction AUC/BA; Prob-AUC is continuous-score ROC-AUC.", "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        pct = lambda metric: f"{100*row[metric+'_mean']:.4f} ± {100*row[metric+'_sd_population']:.4f}"
        report.append(f"| {row['group']} | {pct('acc')} | {pct('hard_auc')} | {pct('prob_auc')} | {pct('sen')} | {pct('spe')} | {pct('asd_f1')} |")
    report += ["", "## Paired comparisons", "", "Comparisons use within-fold differences; confidence intervals are descriptive paired t intervals.", ""]
    focus = [row for row in paired if row["metric"] in ("acc", "hard_auc", "prob_auc", "sen", "spe") and not row["exclude_fold2"]]
    report.extend(["| Comparison | Metric | Δ pp | 95% CI pp | W/T/L |", "|---|---|---:|---:|---:|"])
    for row in focus:
        report.append(f"| {row['comparison']} | {row['metric']} | {100*row['mean_delta']:.4f} | [{100*row['ci95_low_descriptive']:.4f}, {100*row['ci95_high_descriptive']:.4f}] | {row['wins']}/{row['ties']}/{row['losses']} |")
    report += ["", "## Mechanism findings", "",
        f"Independent checkpoint audits: {len(audit_rows)}/40. Maximum test residual fraction: {max(float(x['test_rho_max']) for x in mechanisms):.6f}; mean test residual fraction: {np.mean([x['test_rho_mean'] for x in mechanisms]):.6f}. These fractions are direct measured ‖delta‖/‖h‖, not a nominal cap.",
        f"Correction-off hard flips across all folds/groups: {sum(int(x['hard_flips']) for x in interventions if x['intervention']=='correction_off')}. Fold 2 exposed rows were checked in every intervention; affected counts are reported in `interventions.csv`.",
        f"Shared-epoch base RNG comparisons: {rng_summary['common_epochs']}; exact triplets: {rng_summary['exact_triplets']}; mismatches: {rng_summary['mismatches']}.", "",
        "G2−G1 tests the added hard-memory contribution; G3−G2 tests the candidate site-background memory; G3−G4 tests whether the audited natural-language vectors outperform the geometry-matched anonymous control. A positive model-level difference alone does not establish clinical interpretability.", "",
        "## Decision and limitations", "",
        "The experiment conclusion is assigned in `audit_status.json` from the preregistered cross-fold performance and mechanism checks. This single-seed historical protocol does not justify an unbiased generalization claim. The site-offset bank is a candidate background representation, not validated pure noise. OOF difficulty is cross-fitted only at the logistic-probe stage because the shared warmup encoder has seen the outer training cohort. Fold 2 retains known train/test label overlap in the base experiment. Follow-up should first address independent, leakage-controlled validation rather than add more prototype groups.", ""]
    (R.RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")


def finalize() -> dict:
    import numpy as np
    started = time.time()
    preflight = json.loads((R.RESULTS / "preflight.json").read_text())
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != R.sha_json(R.source_inventory()):
        raise RuntimeError("independent audit requires exact passing preflight source SHA")
    manifest = json.loads((R.EXP / "manifest.json").read_text())
    if manifest.get("completed_tasks") != 40 or manifest.get("status") not in ("TRAINING_COMPLETE_PENDING_AUDIT", "AUDITING", "PARTIAL"):
        raise RuntimeError(f"cannot audit incomplete training: {manifest.get('status')} {manifest.get('completed_tasks')}/40")
    R.atomic_json(R.EXP / "manifest.json", {**manifest, "status": "AUDITING"})
    audits, folds, mechanisms, interventions = [], [], [], []
    predictions = {}
    rng_by_fold = {}
    for fold in range(1, 11):
        for group in GROUPS:
            result = _audit_fold(group, fold, preflight)
            audits.append(result["audit"]); folds.append(result["fold_metric"])
            mechanisms.append(result["mechanism"]); interventions.extend(result["interventions"])
            predictions[(group, fold)] = result["prediction"]
            rng_by_fold.setdefault(fold, []).append(result["audit"])

    history = _history_rows()
    historical_task_rows = []
    with BASE_RESULTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("group") == BASE_GROUP and row.get("status") == "INDEPENDENT_AUDIT_PASS":
                historical_task_rows.append({"group": BASE_GROUP, "fold": int(row["fold"]), **history[int(row["fold"])]})
    all_folds = folds + historical_task_rows
    R.write_csv(R.RESULTS / "checkpoint_audit.csv", audits)
    R.write_csv(R.RESULTS / "fold_metrics.csv", all_folds)
    R.write_csv(R.RESULTS / "mechanism.csv", mechanisms)
    R.write_csv(R.RESULTS / "interventions.csv", interventions)
    summary = _summary_rows(folds, history)
    R.write_csv(R.RESULTS / "summary.csv", summary)

    group_metrics = {group: {fold: next(row for row in folds if row["group"] == group and row["fold"] == fold)
                             for fold in range(1, 11)} for group in GROUPS}
    group_metrics[BASE_GROUP] = history
    comparisons = [(f"{group} vs E74.4-G2", group, BASE_GROUP) for group in GROUPS]
    comparisons += [("G2 vs G1", "G2", "G1"), ("G3 vs G2", "G3", "G2"), ("G3 vs G4", "G3", "G4")]
    paired = []
    for name, left_key, right_key in comparisons:
        left, right = group_metrics[left_key], group_metrics[right_key]
        for metric in METRICS:
            paired.append(_paired_row(name, left_key, right_key, left, right, metric, False))
            if right_key == BASE_GROUP or name in ("G2 vs G1", "G3 vs G2", "G3 vs G4"):
                paired.append(_paired_row(name, left_key, right_key, left, right, metric, True))
        for fold in range(1, 11):
            p_left = predictions[(left_key, fold)] if left_key in GROUPS else None
            p_right = predictions[(right_key, fold)] if right_key in GROUPS else None
            if p_left is not None and p_right is not None:
                labels = np.asarray(p_left["labels"], dtype=np.int64)
                counts = _confusion_changes(p_right, p_left, labels)
                pair = next((row for row in paired if row["comparison"] == name and row["metric"] == "acc" and not row["exclude_fold2"]), None)
                pair.setdefault("foldwise_correction_counts", {})[str(fold)] = {
                    "wrong_to_correct": counts["wrong_to_correct"],
                    "correct_to_wrong": counts["correct_to_wrong"], "net_corrections": counts["net_corrections"]}
    R.write_csv(R.RESULTS / "paired_comparisons.csv", paired)

    base_acc = summary[-1]["acc_mean"]
    g1, g2, g3, g4 = (next(row for row in summary if row["group"] == key) for key in GROUPS)
    pair_g2_g1 = next(row for row in paired if row["comparison"] == "G2 vs G1" and row["metric"] == "acc" and not row["exclude_fold2"])
    pair_g3_g2 = next(row for row in paired if row["comparison"] == "G3 vs G2" and row["metric"] == "acc" and not row["exclude_fold2"])
    pair_g3_g4 = next(row for row in paired if row["comparison"] == "G3 vs G4" and row["metric"] == "acc" and not row["exclude_fold2"])
    hard_flip_total = sum(int(row["hard_flips"]) for row in interventions if row["intervention"] == "correction_off")
    net_g2_g1 = sum(value["net_corrections"] for value in pair_g2_g1.get("foldwise_correction_counts", {}).values())
    net_g3_g2 = sum(value["net_corrections"] for value in pair_g3_g2.get("foldwise_correction_counts", {}).values())
    net_g3_g4 = sum(value["net_corrections"] for value in pair_g3_g4.get("foldwise_correction_counts", {}).values())
    ninefold_acc = {group: sum(group_metrics[group][fold]["acc"] for fold in range(1, 11) if fold != 2) / 9
                    for group in GROUPS + (BASE_GROUP,)}
    meets_base = g3["acc_mean"] >= base_acc
    g2_consistent = pair_g2_g1["mean_delta"] > 0 and pair_g2_g1["wins"] > pair_g2_g1["losses"] and net_g2_g1 > 0
    background_consistent = pair_g3_g2["mean_delta"] >= 0 and pair_g3_g2["wins"] >= pair_g3_g2["losses"] and net_g3_g2 >= 0
    text_contribution = pair_g3_g4["mean_delta"] > 0 and pair_g3_g4["wins"] > pair_g3_g4["losses"] and net_g3_g4 > 0
    if meets_base and g2_consistent and background_consistent and text_contribution:
        conclusion = "GO"
    elif hard_flip_total > 0 and (g2_consistent or background_consistent or text_contribution):
        conclusion = "PARTIAL"
    else:
        conclusion = "NO-GO"

    common_epochs = exact_triplets = mismatches = 0
    group_common = group_exact = group_mismatch = 0
    for fold in range(1, 11):
        ref = {int(row["epoch"]): row for row in json.loads(
            (R.E744_TASKS / f"E74.4-G2-Positive-DCR_f{fold:02d}.json").read_text())["rng_trace"]}
        group_traces = {}
        for group in GROUPS:
            payload = R.torch_load(R.best_path(group, fold))
            current = {int(row["epoch"]): row for row in payload["base_rng_trace"]}
            current_common = set(ref) & set(current)
            count = sum(all(ref[e][key] == current[e][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in current_common)
            common_epochs += len(current_common); exact_triplets += count; mismatches += len(current_common) - count
            group_traces[group] = current
        for left_index, left in enumerate(GROUPS):
            for right in GROUPS[left_index + 1:]:
                common = set(group_traces[left]) & set(group_traces[right])
                count = sum(all(group_traces[left][e][key] == group_traces[right][e][key]
                                for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
                group_common += len(common); group_exact += count; group_mismatch += len(common) - count
    rng_summary = {"common_epochs": common_epochs, "exact_triplets": exact_triplets, "mismatches": mismatches,
                   "group_common_epochs": group_common, "group_exact_triplets": group_exact,
                   "group_mismatches": group_mismatch}
    if mismatches or group_mismatch:
        raise RuntimeError(f"RNG parity failed: {rng_summary}")
    audit_status = {"status": "PASS", "audited_checkpoints": len(audits), "expected_checkpoints": 40,
        "failed_checkpoints": 0, "all_metrics_recomputed": True, "all_hard_predictions_exact": True,
        "all_probabilities_within_1e-7": True, "all_optimizers_restored": True,
        "prototype_test_isolation_pass": True, "fold2_overlap_excluded_from_memory": True,
        "base_rng_parity": rng_summary, "decision": conclusion,
        "performance_vs_e744_g2": {group: {"delta_acc": group_metrics[group][fold]["acc"] - history[fold]["acc"]
                                                   for fold in range(1, 11)} for group in GROUPS},
        "ninefold_acc_mean": ninefold_acc, "mechanism_net_corrections": {
            "G2_vs_G1": net_g2_g1, "G3_vs_G2": net_g3_g2, "G3_vs_G4": net_g3_g4},
        "fold2_exposed_rows_affected": {"377": any(row["fold"] == 2 and (row.get("fold2_row377_flipped") or False) for row in interventions),
                                        "407": any(row["fold"] == 2 and (row.get("fold2_row407_flipped") or False) for row in interventions)},
        "duration_seconds": time.time() - started, "source_sha256": preflight["source_sha256"],
        "training_source_unchanged": True}
    R.atomic_json(R.RESULTS / "audit_status.json", audit_status)
    _write_report(audits, folds, summary, paired, mechanisms, interventions, rng_summary, started)
    final_manifest = {**manifest, "status": "COMPLETE_AUDITED", "independent_audits": len(audits),
        "checkpoint_audit": "PASS 40/40", "audit_status": "PASS", "scientific_conclusion": conclusion,
        "rng_audit": rng_summary, "source_sha256": preflight["source_sha256"],
        "audit_duration_seconds": time.time() - started, "results": str(R.RESULTS)}
    R.atomic_json(R.EXP / "manifest.json", final_manifest)
    return audit_status
