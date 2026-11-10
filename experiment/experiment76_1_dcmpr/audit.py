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
AUDIT_REPAIR_REASON = (
    "Removed a duplicate test-index selection from the role-attention diagnostic; "
    "the full attention tensor had already been restricted to test rows."
)


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


def _evaluate(state, module, memory, group, *, residual=True, confusion=None, background=None,
              text_enabled=True):
    import torch
    module.eval(); state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        output = R.forward_with_dcmp(state, module, memory, group, training=False,
                                     residual_enabled=residual, confusion_enabled=confusion,
                                     background_enabled=background, text_enabled=text_enabled)
    runtime = R.load_base().load_history().load_e69().load_e61().load_base()
    metrics = runtime.classification_metrics(output["logp"], state["test"], state["y"])
    return output, metrics


def _audit_fold(group: str, fold: int, preflight: dict):
    import numpy as np
    import torch
    from scipy.special import softmax

    task_file, ckpt_file = R.task_path(group, fold), R.best_path(group, fold)
    task, payload = json.loads(task_file.read_text()), R.torch_load(ckpt_file)
    expected_source = preflight["source_sha256"]
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
    options = [("correction_off", False, None, None, True),
               ("text_condition_off", True, None, None, False)]
    spec = R.CFG["groups"][group]
    if spec["confusion"]:
        options.append(("confusion_memory_off", True, False, None, True))
    for name, residual, confusion, background, text_enabled in options:
        alt_out, alt_metrics = _evaluate(state, module, memory, group, residual=residual,
                                         confusion=confusion, background=background,
                                         text_enabled=text_enabled)
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
    role_attention = attention.mean(-1)
    role_entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
    mechanism = {"group": group, "fold": fold, "best_epoch_1based": int(payload["best_epoch_1based"]),
        "test_rho_mean": float(ratio.index_select(0, test_tensor).mean().cpu()),
        "test_rho_p50": float(torch.quantile(ratio.index_select(0, test_tensor), .50).cpu()),
        "test_rho_p95": float(torch.quantile(ratio.index_select(0, test_tensor), .95).cpu()),
        "test_rho_max": float(ratio.index_select(0, test_tensor).max().cpu()),
        "test_gate_mean": float(d["rho"].index_select(0, test_tensor).mean().cpu()),
        "risk_mean": float(d["risk"].index_select(0, test_tensor).mean().cpu()),
        "reliability_mean": float(d["reliability"].index_select(0, test_tensor).mean().cpu()),
        "test_prediction_flips_when_correction_off": intervention_rows[0]["hard_flips"],
        "role_attention_max_weight": {str(role): float(attention[:, role, :].max(-1).values.mean().cpu())
                                       for role in range(4)},
        "role_attention_entropy": {str(role): float(role_entropy[:, role].mean().cpu())
                                    for role in range(4)},
        "role_confidence_weights": d["role_weights"].index_select(0, test_tensor).mean(0).detach().cpu().tolist(),
        "prototype_available_count": int(d["prototype_available"].sum().item()),
        "memory_support": memory.metadata.get("typical_support", {}),
        "hard_support": memory.metadata.get("hard_support", {}),
        "warmup_classifier_error_overlap": memory.metadata.get("warmup_classifier_error_overlap", {}),
        "memory_train_ids": len(legal), "test_ids_excluded": len(test),
        "fold2_overlap_excluded": [377, 407] if fold == 2 else [],
        "base_rng_common_epochs": len(common), "base_rng_exact_epochs": exact,
        "base_rng_mismatches": len(common) - exact}
    audit_row = {"group": group, "fold": fold, "status": "PASS",
        "checkpoint_sha256": R.sha_file(ckpt_file), "source_sha256": expected_source,
        "audit_script_sha256": R.sha_file(R.EXP / "audit.py"),
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
                  rng_summary, started, audit_repair):
    import numpy as np
    report = ["# Experiment 76.1 — DCMP-R", "",
        "Status: independently audited after reconstructing all best checkpoints from disk.", "",
        f"Audit-only repair: {audit_repair['reason']} Audit script SHA changed from `{audit_repair['before_sha256']}` to `{audit_repair['after_sha256']}`. Training code/configuration and all locked historical inputs retained their preflight SHA; no model was retrained.", "",
        "## Literature and source audit", "",
        "The implementation was checked against SPromptGL's GFR/CAM/PEN and classifier path, and the locked E74.4-G2-Positive-DCR runner. It keeps one CAM call, captures the existing 144-dimensional classifier input, and applies the refined module immediately before the unchanged OutputLayer. BrainPrompt's official ABIDE prompt source lists TD/TC before ASD; the reused frozen CLIP RN50 cache is mapped to ASD=0, TD=1. This is not a reproduction of BrainPrompt's original language encoder. CoW motivates support self-prediction and error-driven hard prototypes; PGP-SAM motivates inter-/intra-class prototypes and prompt-conditioned features. HP²L is directly related at the broad prototype-prompt level; this experiment's distinct hypothesis is bidirectional historical confusion memory and selective patient correction, not hierarchical disorder classification.",
        "", "Sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official ABIDE prompts](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt), [PGP-SAM paper](https://arxiv.org/abs/2501.06692), [PGP-SAM code](https://github.com/PRIS-CV/PGP-SAM), [CoW paper](https://ijcai.org/proceedings/2025/139), [CoW code](https://github.com/jcjiang99/CoW), [HP²L DOI](https://doi.org/10.1016/j.media.2026.104063).", "",
        "## Method and protocol", "",
        "All three groups share identical DCMP-R parameter shapes and per-fold initialization. G1 retrieves class-balanced typical ASD/TD memories; G2 adds hard prototypes only from actual 5-fold OOF LogisticRegression probe errors, capped at 20% of each true class; G3 matches G2 but replaces disease text with norm- and class-cosine-matched anonymous vectors. The probe is cross-fitted on the frozen warmup CAM144 features, so this is OOF only for the probe and not strict OOF for the shared deep encoder. Typical and hard memories update from detached same-step CAM144 features using EMA=0.9 and deterministically re-cluster every 20 epochs inside fixed legal pools. Query routing does not read query labels; training queries are removed from their own cluster sums and counts. The added memory excludes all outer test IDs and, in Fold 2, rows 377/407 despite their inherited base train/test overlap.",
        "A role-balanced two-stage retrieval uses cosine logits divided by fixed tau=0.10, with role-local prototype attention and a learned role-confidence composer. Explicit ASD/TD evidence differences and typical/hard conflicts feed a shared 144-D feature correction. Correction is capped at 5% of the CAM144 norm; hard-margin and easy-preservation terms use fixed training-only probe masks.",
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
        f"Independent checkpoint audits: {len(audit_rows)}/30. Maximum test residual fraction: {max(float(x['test_rho_max']) for x in mechanisms):.6f}; mean test residual fraction: {np.mean([x['test_rho_mean'] for x in mechanisms]):.6f}. These fractions are direct measured ‖delta‖/‖h‖, not a nominal cap.",
        f"Correction-off hard flips across all folds/groups: {sum(int(x['hard_flips']) for x in interventions if x['intervention']=='correction_off')}. Fold 2 exposed rows were checked in every intervention; affected counts are reported in `interventions.csv`.",
        f"Shared-epoch base RNG comparisons: {rng_summary['common_epochs']}; exact triplets: {rng_summary['exact_triplets']}; mismatches: {rng_summary['mismatches']}.", "",
        "G2−G1 tests the added bidirectional hard-memory contribution. G2−G3 tests whether the real BrainPrompt disease vectors outperform the geometry-matched anonymous control. Fixed-checkpoint correction-off, confusion-memory-off, and text-condition-off interventions are diagnostic and do not replace the trained group comparisons.", "",
        "## Decision and limitations", "",
        "The decision in `audit_status.json` follows the pre-registered targets: base-level recovery, G2 gain over G1 with net hard-case corrections, and G2 gain over anonymous G3. This single-seed protocol retains val=test checkpoint selection, pre-CV preprocessing, a transductive graph, and Fold 2 overlap, so it cannot support an unbiased generalization claim. The warmup encoder has seen outer-training features when probe OOF errors are generated. Follow-up should prioritize leakage-controlled independent validation rather than expand prototype searches.", ""]
    (R.RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")


def finalize() -> dict:
    import numpy as np
    started = time.time()
    preflight = json.loads((R.RESULTS / "preflight.json").read_text())
    original_inventory = preflight.get("source_inventory", {})
    current_inventory = R.source_inventory()
    if preflight.get("status") != "PASS" or R.sha_json(original_inventory) != preflight.get("source_sha256"):
        raise RuntimeError("independent audit requires intact passing preflight provenance")
    changed_sources = [key for key in original_inventory
                       if current_inventory.get(key) != original_inventory[key]]
    if set(current_inventory) != set(original_inventory) or set(changed_sources) - {"e761_audit"}:
        raise RuntimeError(f"unexpected post-training source changes: {changed_sources}")
    if not changed_sources:
        raise RuntimeError("audit script repair was expected but no source change was recorded")
    audit_repair = {
        "scope": "audit_only",
        "reason": AUDIT_REPAIR_REASON,
        "before_sha256": original_inventory["e761_audit"],
        "after_sha256": current_inventory["e761_audit"],
        "training_source_sha256": preflight["source_sha256"],
        "changed_source_keys": changed_sources,
        "training_code_and_locked_inputs_unchanged": True,
        "models_retrained": False,
    }
    R.atomic_json(R.RESULTS / "audit_repair.json", audit_repair)
    manifest = json.loads((R.EXP / "manifest.json").read_text())
    if manifest.get("completed_tasks") != 30 or manifest.get("status") not in ("TRAINING_COMPLETE_PENDING_AUDIT", "AUDITING", "PARTIAL"):
        raise RuntimeError(f"cannot audit incomplete training: {manifest.get('status')} {manifest.get('completed_tasks')}/30")
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
    comparisons += [("G2 vs G1", "G2", "G1"), ("G2 vs G3", "G2", "G3")]
    paired = []
    for name, left_key, right_key in comparisons:
        left, right = group_metrics[left_key], group_metrics[right_key]
        for metric in METRICS:
            paired.append(_paired_row(name, left_key, right_key, left, right, metric, False))
            if right_key == BASE_GROUP or name in ("G2 vs G1", "G2 vs G3"):
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
    g1, g2, g3 = (next(row for row in summary if row["group"] == key) for key in GROUPS)
    pair_g2_g1 = next(row for row in paired if row["comparison"] == "G2 vs G1" and row["metric"] == "acc" and not row["exclude_fold2"])
    pair_g2_g3 = next(row for row in paired if row["comparison"] == "G2 vs G3" and row["metric"] == "acc" and not row["exclude_fold2"])
    correction_off = [row for row in interventions if row["intervention"] == "correction_off"]
    hard_flip_total = sum(int(row["hard_flips"]) for row in correction_off)
    correction_off_net = sum(int(row["net_corrections"]) for row in correction_off)
    net_g2_g1 = sum(value["net_corrections"] for value in pair_g2_g1.get("foldwise_correction_counts", {}).values())
    net_g2_g3 = sum(value["net_corrections"] for value in pair_g2_g3.get("foldwise_correction_counts", {}).values())
    ninefold_acc = {group: sum(group_metrics[group][fold]["acc"] for fold in range(1, 11) if fold != 2) / 9
                    for group in GROUPS + (BASE_GROUP,)}
    meets_base = g2["acc_mean"] >= base_acc and g2["hard_auc_mean"] >= summary[-1]["hard_auc_mean"]
    g2_consistent = pair_g2_g1["mean_delta"] > 0 and pair_g2_g1["wins"] > pair_g2_g1["losses"] and net_g2_g1 > 0
    text_contribution = pair_g2_g3["mean_delta"] > 0 and pair_g2_g3["wins"] > pair_g2_g3["losses"] and net_g2_g3 > 0
    if meets_base and g2_consistent and text_contribution and correction_off_net > 0:
        conclusion = "GO"
    elif hard_flip_total > 0 and (g2_consistent or text_contribution):
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
    audit_status = {"status": "PASS", "audited_checkpoints": len(audits), "expected_checkpoints": 30,
        "failed_checkpoints": 0, "all_metrics_recomputed": True, "all_hard_predictions_exact": True,
        "all_probabilities_within_1e-7": True, "all_optimizers_restored": True,
        "prototype_test_isolation_pass": True, "fold2_overlap_excluded_from_memory": True,
        "base_rng_parity": rng_summary, "decision": conclusion,
        "performance_vs_e744_g2": {group: {"delta_acc": group_metrics[group][fold]["acc"] - history[fold]["acc"]
                                                   for fold in range(1, 11)} for group in GROUPS},
        "ninefold_acc_mean": ninefold_acc, "mechanism_net_corrections": {
            "G2_vs_G1": net_g2_g1, "G2_vs_G3": net_g2_g3,
            "correction_off": correction_off_net},
        "fold2_exposed_rows_affected": {"377": any(row["fold"] == 2 and (row.get("fold2_row377_flipped") or False) for row in interventions),
                                        "407": any(row["fold"] == 2 and (row.get("fold2_row407_flipped") or False) for row in interventions)},
        "duration_seconds": time.time() - started, "source_sha256": preflight["source_sha256"],
        "audit_source_sha256": R.sha_json(current_inventory), "audit_script_repair": audit_repair,
        "training_source_unchanged": True}
    R.atomic_json(R.RESULTS / "audit_status.json", audit_status)
    _write_report(audits, folds, summary, paired, mechanisms, interventions, rng_summary, started, audit_repair)
    final_manifest = {**manifest, "status": "COMPLETE_AUDITED", "independent_audits": len(audits),
        "checkpoint_audit": "PASS 30/30", "audit_status": "PASS", "scientific_conclusion": conclusion,
        "rng_audit": rng_summary, "source_sha256": preflight["source_sha256"],
        "audit_script_sha256": current_inventory["e761_audit"], "audit_script_repair": audit_repair,
        "audit_duration_seconds": time.time() - started, "results": str(R.RESULTS)}
    R.atomic_json(R.EXP / "manifest.json", final_manifest)
    return audit_status
