#!/usr/bin/env python3
"""Independent checkpoint, prediction, trajectory, and mechanism audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as runner
from bcspc import final_log_probabilities


CORE = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
ALL_METRICS = CORE + ("nll", "brier", "ece")
GROUP_NAMES = runner.GROUPS


def _same_metrics(actual, expected, tolerance=1e-8):
    for key in ALL_METRICS:
        if key not in actual or key not in expected:
            raise RuntimeError(f"metric {key} missing from checkpoint recalc")
        if abs(float(actual[key]) - float(expected[key])) > tolerance:
            raise RuntimeError(f"metric mismatch {key}: {actual[key]} != {expected[key]}")


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metrics_from_saved(base_logp, ids, labels, probability, prediction):
    recalc = runner.extended_metrics(base_logp, ids, labels)
    recalc["prob"] = _numpy(probability)
    recalc["pred"] = _numpy(prediction)
    return recalc


def _metrics_rows(metrics, method, fold):
    return {"method": method, "fold": fold, **{key: float(metrics[key]) for key in ALL_METRICS}}


def _prediction_transitions(y, reference, target):
    y, reference, target = map(lambda x: np.asarray(x, dtype=int), (y, reference, target))
    reference_correct, target_correct = reference == y, target == y
    return {
        "error_to_correct": int((~reference_correct & target_correct).sum()),
        "correct_to_error": int((reference_correct & ~target_correct).sum()),
        "net_correction": int((~reference_correct & target_correct).sum() - (reference_correct & ~target_correct).sum()),
        "reference_correct": int(reference_correct.sum()),
        "target_correct": int(target_correct.sum()),
        "correct_retention": float((reference_correct & target_correct).sum() / max(reference_correct.sum(), 1)),
        "asd_to_td_fixed": int(((y == 0) & (reference == 1) & (target == 0)).sum()),
        "td_to_asd_fixed": int(((y == 1) & (reference == 0) & (target == 1)).sum()),
        "asd_to_td_new": int(((y == 0) & (reference == 0) & (target == 1)).sum()),
        "td_to_asd_new": int(((y == 1) & (reference == 1) & (target == 0)).sum()),
    }


def _paired_summary(values):
    from scipy.stats import t
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    sd = float(array.std(ddof=0))
    critical = float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    half = critical * float(array.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n_folds": n, "mean_diff_pp": mean * 100.0, "population_sd_diff_pp": sd * 100.0,
        "ci95_low_pp": (mean - half) * 100.0, "ci95_high_pp": (mean + half) * 100.0,
        "wins": int((array > 1e-12).sum()), "ties": int((np.abs(array) <= 1e-12).sum()),
        "losses": int((array < -1e-12).sum()), "fold_differences_pp": (array * 100.0).tolist(),
    }


def _write_report(metrics_by_method, paired, correction_summary, mechanism_summary, audit_status,
                  training_seconds, worker_count):
    rows = []
    labels = {
        "E74.4-G2": "E74.4-G2 history",
        GROUP_NAMES[0]: "E76.6-G1 BCSPC-Full",
        GROUP_NAMES[1]: "E76.6-G2 BCSPC-NoBoundary",
    }
    for method in ("E74.4-G2", *GROUP_NAMES):
        items = metrics_by_method[method]
        summary = {}
        for key in CORE:
            values = np.asarray([item[key] for item in items], dtype=float) * 100.0
            summary[key] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        rows.append(f"| {labels[method]} | " + " | ".join(summary[key] for key in CORE) + " |")
    report = [
        "# Experiment 76.6: BCSPC",
        "",
        f"- Status: `{audit_status['status']}`; scientific decision: `{audit_status['scientific_decision']}`.",
        f"- Runtime: approximately {training_seconds / 3600.0:.2f} active training hours (fold task wall times plus Fold 1's logged training span); independent audit: {audit_status['audit_wall_seconds']:.1f} seconds; actual training workers: {worker_count}; shared backbone trajectories: 10.",
        "- Historical evaluation protocol retained: test-fold ACC selects checkpoints, preprocessing precedes CV, graph propagation is transductive, and Fold 2 retains the historical overlap rows 377/407. These results are matched exploratory comparisons, not unbiased independent generalization estimates.",
        "- E76.6 is backbone-protected online plugin training: the E74.4-G2 backbone resumes from E61 epoch-10 and follows its original loss/update trajectory; new branch inputs CAM144 and the base score are detached, so branch gradients do not update the backbone.",
        "- The restored E74.4-G2 CAM class is `PLNNoPEN` and has no original prompt buffer. To preserve the audited checkpoint, E76.6 supplies the eight SHA-verified `network.py` prompt embeddings only to the new semantic branch; it does not claim that the historical backbone ran the original PEN.",
        "",
        "## Core Metrics",
        "",
        "| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Values are 10-fold mean +/- population SD in percent. Full per-fold NLL, Brier, ECE, and core metrics are in `fold_metrics.csv`.",
        "",
        "## Paired Comparisons",
        "",
    ]
    for comparison, values in paired.items():
        acc = values["acc"]
        report.append(
            f"- {comparison} ACC: {acc['mean_diff_pp']:+.4f} pp (95% descriptive paired CI "
            f"{acc['ci95_low_pp']:+.4f}, {acc['ci95_high_pp']:+.4f}); W/T/L={acc['wins']}/{acc['ties']}/{acc['losses']}; excluding Fold 2: {acc['fold2_excluded_mean_diff_pp']:+.4f} pp."
        )
    report.extend([
        "",
        "| Paired comparison | Metric | Mean difference (pp) | Descriptive 95% CI (pp) | W/T/L | Excluding Fold 2 (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for comparison, values in paired.items():
        for metric in CORE:
            result = values[metric]
            report.append(
                f"| {comparison} | {metric} | {result['mean_diff_pp']:+.4f} | "
                f"[{result['ci95_low_pp']:+.4f}, {result['ci95_high_pp']:+.4f}] | "
                f"{result['wins']}/{result['ties']}/{result['losses']} | "
                f"{result['fold2_excluded_mean_diff_pp']:+.4f} |"
            )
    report.extend([
        "",
        "## Mechanism",
        "",
        f"- Historical E74.4-G2 to G1 prediction changes: error-to-correct {correction_summary['history_to_g1']['error_to_correct']}, correct-to-error {correction_summary['history_to_g1']['correct_to_error']}, net {correction_summary['history_to_g1']['net_correction']}; original-correct retention {100 * correction_summary['history_to_g1']['correct_retention']:.2f}%.",
        f"- Same G1 checkpoint, Full versus Residual-Off: error-to-correct {correction_summary['g1_full_vs_residual_off']['error_to_correct']}, correct-to-error {correction_summary['g1_full_vs_residual_off']['correct_to_error']}, net {correction_summary['g1_full_vs_residual_off']['net_correction']}; ACC change {correction_summary['g1_full_vs_residual_off']['acc_change_pp']:+.4f} pp.",
        f"- Same G1 checkpoint, Full versus Boundary-Off: net correction {correction_summary['g1_full_vs_boundary_off']['net_correction']}; ACC change {correction_summary['g1_full_vs_boundary_off']['acc_change_pp']:+.4f} pp.",
        f"- Same G1 checkpoint, Full versus Semantic-Swap: net correction {correction_summary['g1_full_vs_semantic_swap']['net_correction']}; ACC change {correction_summary['g1_full_vs_semantic_swap']['acc_change_pp']:+.4f} pp.",
        f"- Mean G1 gate={mechanism_summary['G1_gate_mean']:.4f}, mean |delta|={mechanism_summary['G1_delta_abs_mean']:.5f}, attention reliable/boundary={mechanism_summary['G1_attention_reliable_mean']:.4f}/{mechanism_summary['G1_attention_boundary_mean']:.4f}.",
        f"- Mean G1 modality weights PHENO/ANAT/FUNC/CORR={', '.join(f'{x:.4f}' for x in mechanism_summary['G1_modality_weights'])}. These are CAM source-slot weights, not biological localization evidence.",
        "- Boundary value is tested by the independently trained NoBoundary group and same-checkpoint Boundary-Off intervention. Text semantics are tested by Semantic-Swap and attention/evidence diagnostics; neither alone establishes clinically validated semantic knowledge.",
        "",
        "## Audit",
        "",
        f"- Independent audit status: `{audit_status['status']}`; checkpoint recalc: {audit_status['checkpoint_recalculations']}; RNG shared epoch triplets: {audit_status['rng_exact_triplets']}/{audit_status['rng_common_triplets']} exact; base parameter/optimizer best state: {audit_status['base_best_state_matches_history']}.",
        "- The independent audit verifies checkpoint identity and SHA, exact base trajectory RNG triplets, original best model/optimizer state, held-out label exclusion from the new memory, A/B non-self retrieval support, parameter-count parity, and recalculated best-checkpoint predictions and metrics.",
        "",
        "The scientific decision is deliberately separate from audit completion. See `audit_status.json`, `paired_comparisons.csv`, `mechanism.csv`, and `correction_analysis.csv` for machine-readable details.",
    ])
    (runner.RESULTS / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def audit_all():
    started = time.time()
    runner.RESULTS.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((runner.RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("formal results cannot be audited without a passing preflight")
    preflight_sources = preflight["base_source_hashes"]
    current_sources = runner.source_snapshot()
    # This independent audit script may be corrected after formal training; every
    # training, configuration, and historical input source must remain identical.
    audit_source = str((runner.EXP / "audit.py").resolve())
    unchanged_training_sources = {
        key: value for key, value in current_sources.items() if key != audit_source
    } == {
        key: value for key, value in preflight_sources.items() if key != audit_source
    }
    if not unchanged_training_sources:
        raise RuntimeError("source/history identity changed after the preflight")
    base_manifest = json.loads((runner.BASE_EXP / "manifest.json").read_text(encoding="utf-8"))
    base_audit = json.loads((runner.BASE_EXP / "results/audit_status.json").read_text(encoding="utf-8"))
    if base_manifest.get("status") != "COMPLETE_AUDITED" or base_audit.get("status") != "PASS":
        raise RuntimeError("the historical E74.4-G2 base is no longer independently audited")

    metrics_by_method = {"E74.4-G2": [], GROUP_NAMES[0]: [], GROUP_NAMES[1]: []}
    fold_rows, mechanism_rows, correction_rows = [], [], []
    rng_exact = rng_common = checkpoint_recalculations = 0
    base_state_matches = True
    source_hash = runner.sha_obj(preflight_sources)
    sites_cpu = runner.read_sites()

    for fold in range(1, 11):
        state, warmup, identity, historical_task, historical_path = runner.prepare_fold(fold)
        historical_checkpoint = runner.torch_load(historical_path)
        base_path = runner.CKPTS / f"BCSPC_base_f{fold:02d}.best.pt"
        base_checkpoint = runner.torch_load(base_path)
        if runner.sha_file(base_path) != json.loads((runner.TASKS / f"BCSPC_shared_f{fold:02d}.json").read_text())["base_checkpoint_sha256"]:
            raise RuntimeError(f"shared base checkpoint SHA mismatch fold={fold}")
        if base_checkpoint.get("source_sha256") != source_hash or base_checkpoint.get("identity") != identity:
            raise RuntimeError(f"base checkpoint source or initialization identity mismatch fold={fold}")
        if base_checkpoint["best_epoch_1based"] != historical_task["best_epoch_1based"]:
            raise RuntimeError(f"historical base best epoch mismatch fold={fold}")
        for key in ("model", "cam", "opt_base", "opt_cam", "opt_added"):
            ref_key = key
            for name, tensor in historical_checkpoint["best_state"][ref_key].items() if key in ("model", "cam") else []:
                if not torch.equal(tensor.cpu(), base_checkpoint["best_state"][key][name].cpu()):
                    base_state_matches = False
                    raise RuntimeError(f"base best parameter mismatch {key}.{name} fold={fold}")
            if key.startswith("opt_") and runner.load_base().load_history().sha_object(
                    historical_checkpoint["best_state"][key]) != runner.load_base().load_history().sha_object(base_checkpoint["best_state"][key]):
                base_state_matches = False
                raise RuntimeError(f"base best optimizer mismatch {key} fold={fold}")
        base_trace = {int(item["epoch"]): item for item in historical_task["rng_trace"]}
        branch_trace = {int(item["epoch"]): item for item in base_checkpoint["rng_trace"]}
        common = sorted(set(base_trace) & set(branch_trace))
        rng_common += len(common)
        exact = sum(base_trace[epoch] == branch_trace[epoch] for epoch in common)
        rng_exact += exact
        if exact != len(common) or len(common) != len(branch_trace):
            raise RuntimeError(f"shared base RNG trace mismatch fold={fold}: {exact}/{len(common)}")

        base_probability = _numpy(base_checkpoint["probability"])
        base_prediction = _numpy(base_checkpoint["prediction"])
        historical_probability = _numpy(historical_checkpoint["test_probability"])
        historical_prediction = _numpy(historical_checkpoint["test_prediction"])
        if not np.array_equal(base_prediction, historical_prediction) or not np.allclose(base_probability, historical_probability, atol=1e-8, rtol=0):
            raise RuntimeError(f"historical base best predictions do not reproduce fold={fold}")
        # Recalculate all methods using each checkpoint's paired backbone state and stored memory snapshot.
        labels = state["y"].long()
        test_ids = list(map(int, state["test_np"]))
        y_test = labels[torch.as_tensor(test_ids, dtype=torch.long, device=labels.device)].cpu().numpy()
        hist = runner.load_base().load_history()
        hist.load_bundle(state, base_checkpoint["best_state"])
        state["cam"].eval()
        state["model"].eval()
        with torch.no_grad():
            base_output, _ = runner.hook_forward(state, training=False)
            base_metrics = runner.extended_metrics(base_output["logp"], test_ids, labels)
        _same_metrics(base_metrics, base_checkpoint["metrics"])
        if not np.array_equal(_numpy(base_metrics["pred"]), base_prediction) or not np.allclose(_numpy(base_metrics["prob"]), base_probability, atol=1e-7, rtol=0):
            raise RuntimeError(f"shared base checkpoint prediction/metric did not recalculate fold={fold}")
        for key in CORE:
            if abs(float(base_metrics[key]) - float(historical_checkpoint["metrics"][key])) > 1e-8:
                raise RuntimeError(f"historical base metric mismatch {key} fold={fold}")
        metrics_by_method["E74.4-G2"].append(base_metrics)
        fold_rows.append(_metrics_rows(base_metrics, "E74.4-G2", fold))
        history_prediction_for_fold = historical_prediction

        task = json.loads((runner.TASKS / f"BCSPC_shared_f{fold:02d}.json").read_text(encoding="utf-8"))
        if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_hash:
            raise RuntimeError(f"shared task incomplete/source mismatch fold={fold}")
        if task.get("base_best_epoch_1based") != historical_task.get("best_epoch_1based") or task.get("epochs_executed") != historical_task.get("epochs_executed"):
            raise RuntimeError(f"shared backbone best/early-stop epoch mismatch fold={fold}")
        if task.get("shared_rng_mismatches") != 0 or task.get("shared_rng_exact_records") != task.get("shared_rng_records"):
            raise RuntimeError(f"shared task RNG audit flags failed fold={fold}")
        if task.get("base_parameter_state_matches_history") is not True or task.get("base_optimizer_states_match_history") is not True:
            raise RuntimeError(f"shared base best state audit flags failed fold={fold}")
        eligible = list(map(int, task["branch_train_ids"]))
        ids_a, ids_b = list(map(int, task["A"])), list(map(int, task["B"]))
        if set(ids_a) & set(ids_b) or set(ids_a + ids_b) != set(eligible):
            raise RuntimeError(f"A/B cross support split invalid fold={fold}")
        if set(eligible) & set(test_ids):
            raise RuntimeError(f"outer test rows entered new support fold={fold}")
        if fold == 2 and ({376, 406} & set(eligible)):
            raise RuntimeError("Fold 2 legacy overlap rows entered BCSPC supervision/memory")

        brainprompt, spromptgl = runner.prompt_vectors(state)
        modules, _ = runner.init_groups(state, brainprompt, spromptgl)
        if sum(p.numel() for p in modules[0].parameters()) != sum(p.numel() for p in modules[1].parameters()):
            raise RuntimeError("G1/G2 parameter counts diverged")
        module_results = {}
        for group_index, group in enumerate(GROUP_NAMES):
            checkpoint_path = runner.best_path(group, fold)
            record = runner.torch_load(checkpoint_path)
            if runner.sha_file(checkpoint_path) != json.loads((runner.TASKS / f"{group}_f{fold:02d}.json").read_text())["checkpoint_sha256"]:
                raise RuntimeError(f"best checkpoint SHA mismatch group={group} fold={fold}")
            if record.get("source_sha256") != source_hash or record.get("identity") != identity:
                raise RuntimeError(f"best checkpoint identity mismatch group={group} fold={fold}")
            if record.get("best_epoch_1based") != record["best"]["epoch"]:
                raise RuntimeError(f"best epoch mismatch group={group} fold={fold}")
            best = record["best"]
            if bool(best["no_boundary"]) != (group_index == 1):
                raise RuntimeError(f"G2 boundary fill condition mismatch group={group} fold={fold}")

            # Verify legal memory contents and fixed opposite-half retrieval.
            memory = runner.move_memory(best["memory_used"], labels.device)
            if memory is not None:
                observed = set(torch.nonzero(memory["observed"], as_tuple=False).flatten().cpu().tolist())
                if observed != set(eligible):
                    raise RuntimeError(f"prototype memory observed IDs differ from legal branch training IDs fold={fold}")
                heldout_memory_labels = memory["labels"].clone()
                allowed = torch.zeros_like(heldout_memory_labels, dtype=torch.bool)
                allowed[torch.as_tensor(eligible, device=labels.device)] = True
                if (heldout_memory_labels[~allowed] != -1).any():
                    raise RuntimeError(f"test/forbidden labels are present in prototype memory fold={fold}")
            if best["epoch"] > 11 and memory is None:
                raise RuntimeError(f"dynamic memory is missing after branch activation fold={fold}")
            if memory is not None and set(ids_a) & set(ids_b):
                raise RuntimeError("self retrieval is possible due to A/B overlap")

            hist = runner.load_base().load_history()
            hist.load_bundle(state, best["base_state"])
            module = modules[group_index]
            module.load_state_dict(best["module"], strict=True)
            state["cam"].eval()
            state["model"].eval()
            with torch.no_grad():
                base_output, cam = runner.hook_forward(state, training=False)
                base_test_ids = torch.as_tensor(test_ids, dtype=torch.long, device=labels.device)
                base_prob_at_method_epoch = _numpy(base_output["logp"][base_test_ids].exp())
                base_pred_at_method_epoch = _numpy(base_output["logp"][base_test_ids].argmax(-1))
                if not np.array_equal(base_pred_at_method_epoch, _numpy(best["base_prediction"])) or not np.allclose(base_prob_at_method_epoch, _numpy(best["base_probability"]), atol=1e-7, rtol=0):
                    raise RuntimeError(f"stored paired backbone predictions mismatch group={group} fold={fold}")
                if memory is None:
                    full_logp = base_output["logp"]
                    full_metrics = runner.extended_metrics(full_logp, test_ids, labels)
                    diagnostics = None
                    intervention_results = {"Residual-Off": full_logp, "Boundary-Off": full_logp, "Semantic-Swap": full_logp}
                else:
                    support_ids = torch.as_tensor(eligible, dtype=torch.long, device=labels.device)
                    full_logp, full_metrics, diagnostics = runner.eval_branch(
                        module, base_output["logp"], cam, test_ids, support_ids, memory, labels, sites_cpu.to(labels.device),
                        no_boundary=(group_index == 1))
                    intervention_results = {}
                    for intervention, kwargs in (
                        ("Residual-Off", {"residual_off": True}),
                        ("Boundary-Off", {"boundary_off": True}),
                        ("Semantic-Swap", {"semantic_swap": True}),
                    ):
                        changed, _, _ = runner.eval_branch(
                            module, base_output["logp"], cam, test_ids, support_ids, memory, labels,
                            sites_cpu.to(labels.device), no_boundary=(group_index == 1), **kwargs)
                        intervention_results[intervention] = changed
                    if group_index == 0:
                        if diagnostics["delta"].abs().max().item() > 0.800001:
                            raise RuntimeError(f"residual bound violation fold={fold}")
                        if diagnostics["site_cap_violations"] != 0:
                            raise RuntimeError(f"site cap violation in saved evidence fold={fold}")
                prediction = _numpy(full_metrics["pred"])
                probability = _numpy(full_metrics["prob"])
            _same_metrics(full_metrics, best["metrics"])
            if not np.array_equal(prediction, _numpy(best["prediction"])):
                raise RuntimeError(f"checkpoint hard prediction did not recalculate group={group} fold={fold}")
            if not np.allclose(probability, _numpy(best["probability"]), atol=1e-7, rtol=0):
                raise RuntimeError(f"checkpoint probability did not recalculate group={group} fold={fold}")
            checkpoint_recalculations += 1
            metrics_by_method[group].append({key: float(full_metrics[key]) for key in ALL_METRICS})
            fold_rows.append(_metrics_rows(full_metrics, group, fold))
            module_results[group] = {
                "prediction": prediction,
                "probability": probability,
                "metrics": full_metrics,
                "base_prediction": _numpy(best["base_prediction"]),
                "base_probability": _numpy(best["base_probability"]),
                "interventions": intervention_results,
                "diagnostics": diagnostics,
                "epoch": int(best["epoch"]),
            }
            if group_index == 0 and diagnostics is not None:
                delta = _numpy(diagnostics["delta"])
                evidence = {
                    "gate_mean": float(diagnostics["gate"].mean().cpu()),
                    "gate_std": float(diagnostics["gate"].std(unbiased=False).cpu()),
                    "delta_abs_mean": float(diagnostics["delta"].abs().mean().cpu()),
                    "delta_abs_max": float(diagnostics["delta"].abs().max().cpu()),
                    "coverage_mean": float(diagnostics["coverage_factor"].mean().cpu()),
                    "evidence_mean": float(diagnostics["final_boundary_evidence"].mean().cpu()),
                    "attention_reliable_mean": float(diagnostics["attention"][..., 0].mean().cpu()),
                    "attention_boundary_mean": float(diagnostics["attention"][..., 1].mean().cpu()),
                    "modality_weights": diagnostics["modality_weights"].mean(0).cpu().tolist(),
                    "support_per_role_mean": diagnostics["support_topk"].mean(0).cpu().tolist(),
                    "retrieval_fallbacks": int(diagnostics["retrieval_fallbacks"]),
                    "invalid_confusion_candidates": int(diagnostics["invalid_confusion_candidates"]),
                    "slot_evidence_mean": diagnostics["slot_difference"].mean(0).cpu().tolist(),
                }
                evidence.update({"method": group, "fold": fold, "best_epoch": int(best["epoch"])})
                mechanism_rows.append(evidence)

        # Corrections are calculated on the same held-out fold indices.
        g1, g2 = module_results[GROUP_NAMES[0]], module_results[GROUP_NAMES[1]]
        transitions = {
            "historical_to_g1": _prediction_transitions(y_test, history_prediction_for_fold, g1["prediction"]),
            "historical_to_g2": _prediction_transitions(y_test, history_prediction_for_fold, g2["prediction"]),
            "g1_full_vs_base_at_checkpoint": _prediction_transitions(y_test, g1["base_prediction"], g1["prediction"]),
            "g1_full_vs_residual_off": _prediction_transitions(y_test, _numpy(g1["interventions"]["Residual-Off"].argmax(-1)[torch.as_tensor(test_ids, device=labels.device)]), g1["prediction"]),
            "g1_full_vs_boundary_off": _prediction_transitions(y_test, _numpy(g1["interventions"]["Boundary-Off"].argmax(-1)[torch.as_tensor(test_ids, device=labels.device)]), g1["prediction"]),
            "g1_full_vs_semantic_swap": _prediction_transitions(y_test, _numpy(g1["interventions"]["Semantic-Swap"].argmax(-1)[torch.as_tensor(test_ids, device=labels.device)]), g1["prediction"]),
        }
        for comparison, counts in transitions.items():
            correction_rows.append({"fold": fold, "comparison": comparison, **counts})
        for intervention in ("Residual-Off", "Boundary-Off", "Semantic-Swap"):
            changed = g1["interventions"][intervention]
            changed_metrics = runner.extended_metrics(changed, test_ids, labels)
            mechanism_rows.append({
                "method": f"G1-{intervention}", "fold": fold,
                **{f"metric_{key}": float(changed_metrics[key]) for key in ALL_METRICS},
            })
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paired = {}
    comparisons = (
        ("G1-E74.4-G2", GROUP_NAMES[0], "E74.4-G2"),
        ("G2-E74.4-G2", GROUP_NAMES[1], "E74.4-G2"),
        ("G1-G2", GROUP_NAMES[0], GROUP_NAMES[1]),
    )
    paired_rows = []
    for name, left, right in comparisons:
        paired[name] = {}
        for metric in CORE:
            diffs = [metrics_by_method[left][i][metric] - metrics_by_method[right][i][metric] for i in range(10)]
            full = _paired_summary(diffs)
            nine = _paired_summary([value for i, value in enumerate(diffs, start=1) if i != 2])
            full["fold2_excluded_mean_diff_pp"] = nine["mean_diff_pp"]
            full["fold2_excluded_wins"] = nine["wins"]
            full["fold2_excluded_ties"] = nine["ties"]
            full["fold2_excluded_losses"] = nine["losses"]
            paired[name][metric] = full
            paired_rows.append({"comparison": name, "metric": metric, **full})
    runner.write_csv(runner.RESULTS / "fold_metrics.csv", fold_rows)
    runner.write_csv(runner.RESULTS / "paired_comparisons.csv", paired_rows)
    runner.write_csv(runner.RESULTS / "mechanism.csv", mechanism_rows)
    runner.write_csv(runner.RESULTS / "correction_analysis.csv", correction_rows)

    def sum_transition(field):
        keys = ("error_to_correct", "correct_to_error", "net_correction", "asd_to_td_fixed", "td_to_asd_fixed",
                "asd_to_td_new", "td_to_asd_new")
        selected = [row for row in correction_rows if row["comparison"] == field]
        result = {key: sum(row[key] for row in selected) for key in keys}
        ref_correct = sum(row["reference_correct"] for row in selected)
        kept = sum(row["reference_correct"] - row["correct_to_error"] for row in selected)
        result["correct_retention"] = kept / max(ref_correct, 1)
        return result

    g1_gate_rows = [row for row in mechanism_rows if row.get("method") == GROUP_NAMES[0] and "gate_mean" in row]
    modality = np.mean([row["modality_weights"] for row in g1_gate_rows], axis=0).tolist() if g1_gate_rows else [float("nan")] * 4
    mechanism_summary = {
        "G1_gate_mean": float(np.mean([row["gate_mean"] for row in g1_gate_rows])) if g1_gate_rows else 0.0,
        "G1_delta_abs_mean": float(np.mean([row["delta_abs_mean"] for row in g1_gate_rows])) if g1_gate_rows else 0.0,
        "G1_attention_reliable_mean": float(np.mean([row["attention_reliable_mean"] for row in g1_gate_rows])) if g1_gate_rows else 0.0,
        "G1_attention_boundary_mean": float(np.mean([row["attention_boundary_mean"] for row in g1_gate_rows])) if g1_gate_rows else 0.0,
        "G1_modality_weights": modality,
    }
    correction_summary = {
        "history_to_g1": sum_transition("historical_to_g1"),
        "g1_full_vs_residual_off": sum_transition("g1_full_vs_residual_off"),
        "g1_full_vs_boundary_off": sum_transition("g1_full_vs_boundary_off"),
        "g1_full_vs_semantic_swap": sum_transition("g1_full_vs_semantic_swap"),
    }
    g1_base_metrics = [module_row["acc"] for module_row in metrics_by_method[GROUP_NAMES[0]]]
    residual_acc_diffs = []
    # Interventions are represented in mechanism.csv with independent metric recalculations.
    for fold in range(1, 11):
        full = next(row for row in fold_rows if row["method"] == GROUP_NAMES[0] and row["fold"] == fold)
        residual = next(row for row in mechanism_rows if row["method"] == "G1-Residual-Off" and row["fold"] == fold)
        residual_acc_diffs.append(float(full["acc"]) - float(residual["metric_acc"]))
    correction_summary["g1_full_vs_residual_off"]["acc_change_pp"] = float(np.mean(residual_acc_diffs) * 100.0)
    for intervention, key in (("Boundary-Off", "g1_full_vs_boundary_off"), ("Semantic-Swap", "g1_full_vs_semantic_swap")):
        diffs = []
        for fold in range(1, 11):
            full = next(row for row in fold_rows if row["method"] == GROUP_NAMES[0] and row["fold"] == fold)
            altered = next(row for row in mechanism_rows if row["method"] == f"G1-{intervention}" and row["fold"] == fold)
            diffs.append(float(full["acc"]) - float(altered["metric_acc"]))
        correction_summary[key]["acc_change_pp"] = float(np.mean(diffs) * 100.0)

    g1_acc = paired["G1-E74.4-G2"]["acc"]["mean_diff_pp"]
    boundary_acc = paired["G1-G2"]["acc"]["mean_diff_pp"]
    net = correction_summary["history_to_g1"]["net_correction"]
    residual_net = correction_summary["g1_full_vs_residual_off"]["net_correction"]
    if g1_acc > 0 and paired["G1-E74.4-G2"]["hard_auc"]["mean_diff_pp"] > -0.5 and paired["G1-E74.4-G2"]["prob_auc"]["mean_diff_pp"] > -0.5 and net > 0 and boundary_acc > 0 and residual_net > 0:
        decision = "PROMISING_BCSPC"
    elif g1_acc > 0 and boundary_acc <= 0:
        decision = "GENERIC_PROTOTYPE_GAIN_BOUNDARY_UNVERIFIED"
    elif boundary_acc > 0 and g1_acc <= 0:
        decision = "BOUNDARY_SIGNAL_WITHOUT_BASE_GAIN"
    else:
        decision = "NO_GO_BCSPC"

    gradient_audit = {}
    for fold in range(1, 11):
        task = json.loads((runner.TASKS / f"BCSPC_shared_f{fold:02d}.json").read_text(encoding="utf-8"))
        for item in task["gradient_audit"]:
            group, epoch = item["group"], int(item["epoch"])
            if epoch >= 13:
                summary = gradient_audit.setdefault(group, {key: False for key in (
                    "semantic_projection", "reliable_text_projection", "boundary_text_projection",
                    "modality_gate", "residual_gate", "alpha")})
                for key in summary:
                    summary[key] |= float(item.get(key, 0.0)) > 0.0
    if set(gradient_audit) != set(GROUP_NAMES) or any(not all(values.values()) for values in gradient_audit.values()):
        raise RuntimeError(f"effective branch gradient paths were not all observed: {gradient_audit}")

    task_wall_seconds = sum(
        float(json.loads((runner.TASKS / f"BCSPC_shared_f{fold:02d}.json").read_text(encoding="utf-8")).get("wall_seconds", 0.0))
        for fold in range(1, 11)
    )
    fold1_log = runner.LOGS / "BCSPC_shared_f01.jsonl"
    fold1_log_span = 0.0
    if fold1_log.is_file():
        stat_result = subprocess.run(
            ["stat", "-c", "%W %Y", str(fold1_log)], check=True, capture_output=True, text=True
        )
        birth, modified = map(float, stat_result.stdout.split())
        fold1_log_span = max(0.0, modified - birth)
    training_seconds = task_wall_seconds + fold1_log_span
    audit_seconds = time.time() - started
    audit_path = str((runner.EXP / "audit.py").resolve())
    preflight_audit_sha = preflight_sources.get(audit_path)
    current_audit_sha = runner.sha_file(runner.EXP / "audit.py")

    status = {
        "status": "COMPLETE_AUDITED", "scientific_decision": decision,
        "expected_groups": 2, "expected_folds_per_group": 10,
        "completed_checkpoints": checkpoint_recalculations,
        "checkpoint_recalculations": checkpoint_recalculations,
        "rng_exact_triplets": rng_exact, "rng_common_triplets": rng_common,
        "rng_mismatches": rng_common - rng_exact,
        "base_best_state_matches_history": base_state_matches,
        "historical_base_audit_status": base_audit.get("status"),
        "historical_base_path": str(runner.BASE_EXP),
        "source_snapshot_sha256": source_hash,
        "base_source_sha256_unchanged": True,
        "one_step_gradient_and_optimizer_isolation": preflight["branch_to_backbone_gradient_isolation"],
        "initial_delta_exact_and_base_output_equal": preflight["initial_delta_exactly_zero_and_outputs_equal"],
        "branch_gradient_path_audit": gradient_audit,
        "historical_cam_class": "PLNNoPEN",
        "historical_cam_original_prompt_buffer": False,
        "original_eight_prompt_embeddings": "SHA-verified from network.py and E72 frozen RN50 cache; used in new semantic branch only",
        "training_active_seconds_estimate": training_seconds,
        "training_task_wall_seconds": task_wall_seconds,
        "fold1_log_span_seconds": fold1_log_span,
        "audit_wall_seconds": audit_seconds,
        "actual_training_workers": 1,
        "shared_backbone_trajectories": 10,
        "preflight_audit_source_sha256": preflight_audit_sha,
        "current_independent_audit_source_sha256": current_audit_sha,
        "auditor_only_source_change_after_training": preflight_audit_sha != current_audit_sha,
        "methods": {method: {key: {"mean": float(np.mean([item[key] for item in values])),
                                   "population_sd": float(np.std([item[key] for item in values], ddof=0))}
                              for key in ALL_METRICS}
                    for method, values in metrics_by_method.items()},
        "paired_comparisons": paired,
        "correction_summary": correction_summary,
        "mechanism_summary": mechanism_summary,
    }
    if checkpoint_recalculations != 20 or rng_exact != rng_common or not base_state_matches:
        status["status"] = "AUDIT_FAIL"
        raise RuntimeError("independent audit failed; see exception context before scientific interpretation")
    runner.atomic_json(runner.RESULTS / "audit_status.json", status)
    runner.atomic_json(runner.EXP / "manifest.json", {
        "experiment": runner.CFG["experiment"], "status": "COMPLETE_AUDITED",
        "scientific_decision": decision, "experiment_root": str(runner.EXP),
        "preferred_root": runner.CFG.get("preferred_root", runner.CFG["root"]), "expected_tasks": 20,
        "completed_tasks": 20, "independent_audits": 20,
        "training_workers": 1, "shared_backbone_trajectories": 10,
        "audit_status": "PASS", "source_sha256": source_hash,
        "preflight_audit_source_sha256": preflight_audit_sha,
        "independent_audit_source_sha256": current_audit_sha,
        "auditor_only_source_change_after_training": preflight_audit_sha != current_audit_sha,
    })
    _write_report(metrics_by_method, paired, correction_summary, mechanism_summary, status,
                  training_seconds, 1)
    print(f"COMPLETE_AUDITED decision={decision} checkpoints={checkpoint_recalculations} rng={rng_exact}/{rng_common}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        parser.error("use --all")
    audit_all()


if __name__ == "__main__":
    main()
