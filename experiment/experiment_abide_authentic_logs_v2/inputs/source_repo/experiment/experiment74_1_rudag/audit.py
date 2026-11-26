"""Independent checkpoint readback, fixed-checkpoint interventions, and report."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
BASE = Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows")
E62 = Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity")
E74 = Path("/model/spromptgl/experiment/experiment74_udag")
R = None


def runner():
    global R
    if R is None:
        name = "rudag_741_run_for_audit"
        spec = importlib.util.spec_from_file_location(name, EXP / "run.py")
        R = importlib.util.module_from_spec(spec)
        sys.modules[name] = R
        spec.loader.exec_module(R)
    return R


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_float(value):
    return float(value)


def metric_scalar(metrics):
    return {key: float(metrics[key]) for key in runner().METRICS}


def _classification_only_gradients(state, output):
    import torch

    model = state["model"]
    gate_grad = torch.autograd.grad(
        output["cls"], model.adaptive_gate_logits,
        retain_graph=True, allow_unused=True,
    )[0]
    scorer_norms = []
    for index, scorer in enumerate(model.scorers):
        params = tuple(scorer.parameters())
        grads = torch.autograd.grad(
            output["cls"], params,
            retain_graph=index < len(model.scorers) - 1,
            allow_unused=True,
        )
        squared = [g.detach().square().sum() for g in grads if g is not None]
        scorer_norms.append(float(torch.sqrt(torch.stack(squared).sum()).cpu()) if squared else 0.0)
    return gate_grad.detach().cpu().tolist() if gate_grad is not None else [0.0] * 4, scorer_norms


def _state_dict_sha(bundle):
    import hashlib

    digest = hashlib.sha256()
    for key in ("model", "cam"):
        for name, tensor in sorted(bundle[key].items()):
            digest.update((key + "/" + name).encode() + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _pair_relation_metrics(state, output):
    import torch
    from sklearn.metrics import roc_auc_score
    from cadr_gfr import pair_features

    e69 = runner().load_e69()
    e61 = e69.load_e61()
    base_runtime = e61.load_base()
    train_y = state["train_same"].detach().cpu().numpy().astype(int)
    train_logits = [value.detach().cpu().numpy() for value in output["info"]["pair_logits"]]
    unexposed = np.setdiff1d(state["test_np"], state["train_np"], assume_unique=False)
    test_pairs_np, test_same_np = base_runtime.sample_balanced_pairs(
        unexposed,
        state["labels_np"],
        4096,
        np.random.default_rng(741100 + state["fold"]),
        None,
    )
    test_pairs = torch.as_tensor(test_pairs_np, dtype=torch.long, device=state["y"].device)
    embeddings = [encoder(features) for encoder, features in zip(state["model"].encoders, state["parts"])]
    test_logits = []
    for embedding, scorer in zip(embeddings, state["model"].scorers):
        pair = pair_features(embedding[test_pairs[:, 0]], embedding[test_pairs[:, 1]])
        test_logits.append(scorer(pair).detach().cpu().numpy())
    test_y = np.asarray(test_same_np, dtype=int)
    if np.unique(test_y).size != 2:
        raise RuntimeError("post-hoc unexposed test relation sample lacks both classes")

    records = []
    for index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        train_q = 1.0 / (1.0 + np.exp(-np.clip(train_logits[index], -80, 80)))
        test_q = 1.0 / (1.0 + np.exp(-np.clip(test_logits[index], -80, 80)))
        records.append({
            "modality": modality,
            "train_relation_pair_auc": float(roc_auc_score(train_y, train_logits[index])),
            "unexposed_test_relation_pair_auc_posthoc": float(roc_auc_score(test_y, test_logits[index])),
            "train_relation_score_mean": float(train_q.mean()),
            "train_relation_score_std": float(train_q.std(ddof=0)),
            "test_relation_score_mean": float(test_q.mean()),
            "test_relation_score_std": float(test_q.std(ddof=0)),
            "test_relation_positive_fraction": float((test_q >= 0.55).mean()),
            "test_relation_negative_fraction": float((test_q <= 0.45).mean()),
            "test_relation_uncertain_fraction": float(((test_q > 0.45) & (test_q < 0.55)).mean()),
            "test_relation_pair_count": int(test_y.size),
            "unexposed_test_patient_count": int(unexposed.size),
        })
    return records


def _intervention_record(group, fold, name, full_metrics, intervention_metrics, full_output, intervention_output):
    base_runtime = runner().load_e69().load_e61().load_base()
    full_pred = np.asarray(full_metrics["pred"])
    intervention_pred = np.asarray(intervention_metrics["pred"])
    labels = full_output["_labels_test"]
    full_correct = full_pred == labels
    intervention_correct = intervention_pred == labels
    full_prob = np.asarray(full_metrics["prob"])
    intervention_prob = np.asarray(intervention_metrics["prob"])
    full_repr = full_output["fused"].detach()
    changed_repr = intervention_output["fused"].detach()
    repr_change = float((changed_repr - full_repr).norm() / full_repr.norm().clamp_min(1e-12))
    row = {
        "group": group, "fold": fold, "intervention": name,
        "hard_flips": int((full_pred != intervention_pred).sum()),
        "full_correct_intervention_wrong": int((full_correct & ~intervention_correct).sum()),
        "full_wrong_intervention_correct": int((~full_correct & intervention_correct).sum()),
        "mean_abs_asd_probability_change": float(np.abs(full_prob - intervention_prob).mean()),
        "fused_320_relative_l2_change": repr_change,
    }
    for metric in runner().METRICS:
        if metric in full_metrics and metric in intervention_metrics:
            row[f"full_{metric}"] = float(full_metrics[metric])
            row[f"intervention_{metric}"] = float(intervention_metrics[metric])
            row[f"full_minus_intervention_{metric}_pp"] = 100.0 * (float(full_metrics[metric]) - float(intervention_metrics[metric]))
    return row


def _restore_best(state, checkpoint):
    state_bundle = checkpoint["best_state"]
    runner().load_bundle(state, state_bundle)
    state["model"].eval()
    state["cam"].eval()


def audit_fold(group, fold, expected_sources):
    import torch

    r = runner()
    if not r._task_complete(group, fold, expected_sources):
        raise RuntimeError(f"task/checkpoint is not COMPLETE and source-verified: {group} fold={fold}")
    task = json.loads(r.task_path(group, fold).read_text(encoding="utf-8"))
    checkpoint_path = r.best_path(group, fold)
    if r.sha_file(checkpoint_path) != task["checkpoint_sha256"]:
        raise RuntimeError("checkpoint checksum does not match the completed task")
    checkpoint = r.torch_load(checkpoint_path)
    if checkpoint.get("source_sha256") != expected_sources or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError("checkpoint source/group/fold identity mismatch")

    state, warmup, identity = r.prepare(fold)
    if checkpoint["identity"] != identity or task["identity"] != identity:
        raise RuntimeError("checkpoint initialization/split/pair/warmup identity differs")
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"unexpected label exposure for fold {fold}")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError("checkpoint training relation pairs escape the training population")
    if tuple(state["x"].shape) != (871, 320) or tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError("patient/feature/pair dimensions differ from the locked protocol")
    if not np.isin(state["labels_np"], [0, 1]).all():
        raise RuntimeError("label encoding is not binary ASD=0/TD=1")
    _restore_best(state, checkpoint)

    with torch.enable_grad():
        full_output = r.forward(state, group, training=False, fold=fold, collect=True)
        full_metrics = r.load_e69().load_e61().load_base().classification_metrics(
            full_output["logp"], state["test"], state["y"]
        )
        gate_grad, scorer_grad = _classification_only_gradients(state, full_output)
    for metric in r.METRICS:
        if abs(float(full_metrics[metric]) - float(checkpoint["metrics"][metric])) > 1e-10:
            raise RuntimeError(f"independent readback metric mismatch {metric}: {full_metrics[metric]} vs {checkpoint['metrics'][metric]}")
    if not np.array_equal(full_metrics["pred"], checkpoint["test_prediction"]):
        raise RuntimeError("independent readback hard predictions differ")
    if not np.allclose(full_metrics["prob"], checkpoint["test_probability"], atol=1e-8, rtol=0):
        raise RuntimeError("independent readback probabilities differ")
    expected_parameters = checkpoint["best_state"]
    if r.state_model_sha({"model": state["model"], "cam": state["cam"]}) != _state_dict_sha(expected_parameters):
        raise RuntimeError("loaded model/CAM parameter SHA differs from best state")

    relation = _pair_relation_metrics(state, full_output)
    full_prediction = np.asarray(full_metrics["pred"])
    test_labels = state["y"][state["test"]].detach().cpu().numpy().astype(int)
    interventions = []
    outputs_for_interventions = []
    for intervention in ("disease_neutral", "corr_neutral", "score_permuted"):
        with torch.no_grad():
            changed = r.forward(state, group, training=False, intervention=intervention, fold=fold, collect=False)
            changed_metrics = r.load_e69().load_e61().load_base().classification_metrics(
                changed["logp"], state["test"], state["y"]
            )
        outputs_for_interventions.append((intervention, changed, changed_metrics))
    if group == "G2-RUDAG-Full":
        with torch.no_grad():
            changed = r.forward(state, group, training=False, intervention="direct_off", fold=fold, collect=False)
            changed_metrics = r.load_e69().load_e61().load_base().classification_metrics(
                changed["logp"], state["test"], state["y"]
            )
        outputs_for_interventions.append(("direct_off", changed, changed_metrics))
    full_output["_labels_test"] = test_labels
    for name, changed_output, changed_metrics in outputs_for_interventions:
        interventions.append(_intervention_record(group, fold, name, full_metrics, changed_metrics, full_output, changed_output))

    mechanisms = []
    gate_values = [
        float(0.01 * torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) if i < 3
        else float(torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) for i in range(4)
    ]
    for index, diagnostic in enumerate(full_output["info"]["edge_diagnostics"]):
        rel = relation[index]
        mechanisms.append({
            "group": group, "fold": fold,
            **diagnostic,
            **rel,
            "effective_gate_final": gate_values[index],
            "adaptive_gate_theta_final": float(state["model"].adaptive_gate_logits[index].detach().cpu()),
            "adaptive_gate_initial": checkpoint["initial_gate_values"][index],
            "adaptive_gate_theta_initial": checkpoint["initial_gate_theta"][index],
            "adaptive_gate_classification_gradient_abs": abs(float(gate_grad[index])),
            "relation_scorer_classification_gradient_norm": float(scorer_grad[index]),
            "fold2_label_exposure": fold == 2,
            "train_test_overlap_1based": json.dumps(expected_overlap),
        })

    full_call_counts = [d["gcn_calls"] for d in full_output["info"]["edge_diagnostics"]]
    if full_call_counts != [1, 1, 1, 1]:
        raise RuntimeError(f"independent model forward has unexpected GCN calls: {full_call_counts}")
    audit = {
        "group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
        "checkpoint_sha256": task["checkpoint_sha256"],
        "best_epoch_1based": checkpoint["best_epoch_1based"],
        "epochs_executed": checkpoint["epochs_executed"],
        "metrics": metric_scalar(full_metrics),
        "predictions_exact": True, "probabilities_atol_1e-8": True,
        "model_cam_parameter_sha": r.state_model_sha({"model": state["model"], "cam": state["cam"]}),
        "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
        "split_sha": identity["split_sha"], "pair_sha": identity["pair_sha"],
        "effective_train_sha": identity["effective_train_sha"],
        "train_test_overlap_1based": expected_overlap,
        "pair_training_scope_pass": True,
        "group_initial_model_cam_sha": identity["model_cam_sha"],
        "group_initial_optimizer_sha": identity["optimizer_sha"],
        "final_gate_values": gate_values,
        "final_gate_theta": state["model"].adaptive_gate_logits.detach().cpu().tolist(),
        "edge_gate_values_frozen": torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),
        "edge_gate_requires_grad": bool(state["model"].edge_gate_logits.requires_grad),
        "edge_gate_gradient_none": state["model"].edge_gate_logits.grad is None,
        "gcn_calls_per_modality": full_call_counts,
        "rng_trace": checkpoint["rng_trace"],
        "rng_common_epochs_e74": 0,
        "rng_exact_matches_e74": 0,
        "rng_common_epochs_g1_g2": 0,
        "rng_exact_matches_g1_g2": 0,
        "interventions": interventions,
        "mechanism": mechanisms,
    }
    del state, full_output, outputs_for_interventions, checkpoint
    torch.cuda.empty_cache()
    return audit


def _history_rows(path, group):
    rows = read_csv(path)
    found = [row for row in rows if row.get("group") == group]
    if not found:
        raise RuntimeError(f"historical fold metrics missing {group}: {path}")
    return {int(row["fold"]): {metric: float(row[metric]) for metric in runner().METRICS} for row in found}


def _historical_summary(path, group):
    rows = read_csv(path)
    found = [row for row in rows if row.get("group") == group]
    if not found:
        raise RuntimeError(f"historical summary missing {group}: {path}")
    return found[0]


def _summary_new(group, audits):
    metrics = runner().METRICS
    rows = [a for a in audits if a["group"] == group]
    row = {"group": group, "folds": len(rows), "source": "new audited training"}
    for metric in metrics:
        values = np.asarray([a["metrics"][metric] for a in rows], dtype=np.float64) * 100.0
        row[f"{metric}_mean_percent"] = float(values.mean())
        row[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
    return row


def _summary_history(summary_row, group, source):
    row = {"group": group, "folds": int(float(summary_row.get("folds", 10))), "source": source}
    mapping = {
        "acc": "acc", "hard_auc": "hard_auc", "prob_auc": "prob_auc",
        "sen": "sen", "spe": "spe", "ba": "ba", "asd_f1": "asd_f1",
    }
    for metric, prefix in mapping.items():
        row[f"{metric}_mean_percent"] = float(summary_row[f"{prefix}_mean_percent"])
        row[f"{metric}_population_sd_percent"] = float(summary_row[f"{prefix}_population_sd_percent"])
    return row


def _paired_rows(name, left, right, scopes):
    result = []
    for scope, folds in scopes:
        for metric in runner().METRICS:
            differences = np.asarray([(left[f][metric] - right[f][metric]) * 100.0 for f in folds], dtype=np.float64)
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            if len(differences) > 1:
                from scipy.stats import t
                half = float(t.ppf(0.975, len(differences) - 1) * sd / math.sqrt(len(differences)))
            else:
                half = 0.0
            result.append({
                "comparison": name, "scope": scope, "metric": metric,
                "folds": len(folds), "mean_difference_pp": mean,
                "wins": int((differences > 1e-12).sum()),
                "ties": int((np.abs(differences) <= 1e-12).sum()),
                "losses": int((differences < -1e-12).sum()),
                "descriptive_95ci_low_pp": mean - half,
                "descriptive_95ci_high_pp": mean + half,
                "fold_differences_pp": json.dumps({str(f): float((left[f][metric] - right[f][metric]) * 100.0) for f in folds}, sort_keys=True),
            })
    return result


def _rng_audit(audits):
    e74_tasks = E74 / ".runtime/tasks"
    totals = {"g1_g2_common": 0, "g1_g2_match": 0, "e74_common": 0, "e74_match": 0}
    by_task = {}
    for fold in range(1, 11):
        records = {group: next(a for a in audits if a["group"] == group and a["fold"] == fold) for group in runner().GROUPS}
        g1 = {row["epoch"]: row for row in records[runner().GROUPS[0]]["rng_trace"]}
        g2 = {row["epoch"]: row for row in records[runner().GROUPS[1]]["rng_trace"]}
        common = sorted(set(g1) & set(g2))
        matches = sum(all(g1[epoch][key] == g2[epoch][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for epoch in common)
        totals["g1_g2_common"] += len(common)
        totals["g1_g2_match"] += matches

        e74_path = e74_tasks / f"E74-UDAG_f{fold:02d}.json"
        e74 = json.loads(e74_path.read_text(encoding="utf-8"))
        ref = {row["epoch"]: row for row in e74.get("rng_trace", [])}
        per_group = {}
        for group, trace in ((runner().GROUPS[0], g1), (runner().GROUPS[1], g2)):
            shared = sorted(set(trace) & set(ref))
            exact = sum(all(trace[epoch][key] == ref[epoch][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for epoch in shared)
            totals["e74_common"] += len(shared)
            totals["e74_match"] += exact
            per_group[group] = {"common_epochs": len(shared), "exact_matches": exact}
        by_task[str(fold)] = {
            "g1_g2_common_epochs": len(common), "g1_g2_exact_matches": matches,
            "e74_reference": per_group,
        }
        for record in records.values():
            record["rng_common_epochs_g1_g2"] = len(common)
            record["rng_exact_matches_g1_g2"] = matches
            record["rng_common_epochs_e74"] = per_group[record["group"]]["common_epochs"]
            record["rng_exact_matches_e74"] = per_group[record["group"]]["exact_matches"]
    return {"totals": totals, "by_fold": by_task}


def _write_report(summary_rows, paired, audits, mechanisms, interventions, manifest, rng):
    r = runner()
    summary = {row["group"]: row for row in summary_rows}
    lines = [
        "# Experiment 74.1 — RUDAG",
        "",
        f"Status: **{manifest['status']}**; independent checkpoint audit {manifest['independent_audits']}/{manifest['expected_tasks']}; device `{manifest['device']}`; maximum concurrent workers {manifest['max_workers_observed']}/10; retried task attempts {manifest['training_execution']['retry_count']}; OOM events {manifest['training_execution']['oom_events']}.",
        "",
        "## Source and protocol audit",
        "",
        "SPromptGL's Figure 1 and Sections 2.1–2.2 / Eqs. (1)–(7) describe modality graphs, graph propagation, prompt-guided embedding, CAM and RES. The root implementation used here has a learned linear projection followed by cosine similarity, a detached `score > 0.9` support mask, two GraphConv layers with a ReLU between them, one `normalize_adj(A+I)`, and the original four GFR fusion weights. The original CAM consumes four 36-D modality slots and produces the same 144-D classifier input. The paper writes a label-supervision graph term and a more general DGCN formulation; neither changes the actual E61/E74 objective used here. Historical `GraphConstructLoss(S_m)` and the total loss `L_cls + L_graph + .05 L_pair + .01 L_adv + .01 L_preserve` were retained.",
        "",
        "The E53.3 DARG relation path is retained: four independent 32-D relation encoders; symmetric 65-D pair features (`abs difference`, elementwise product, cosine); 65→32→1 scorers; balanced 4,096 training pairs; CORR-detached conditional modality adversary with GRL on the other three modes; and relation-preservation loss. The original uncalibrated `sigmoid(logit)` score remains. No calibration, OOF fitting, text prompt, extra label-semantic head, or new loss was introduced.",
        "",
        "E74 scored only mutual Top-12 candidates, used the same strength for all modes and allowed fallback additions. RUDAG instead reweights every original-positive CORR edge continuously, while non-CORR scores only the historical candidate set intersected with original-positive GraphLearn support. Each mode has one unified adjacency, one historical GCN call and the original CAM. G2 adds the CADR-bounded direct neighbor difference from that same adjacency; G1 leaves it disabled. Both groups share modules, parameter count, optimizer groups, gate initialization and RNG starting state.",
        "",
        "BrainPrompt's paper and official `brainpromptg_net.py`, `brainpromptc_net.py`, and ABIDE label prompt were reviewed. Its disease-level objective compares graph representations with encoded class text prototypes through cross-entropy. SABER's available paper describes decision-level semantic injection, gated residual fusion and disease contrastive supervision. No SABER official source repository was confirmed, so no source-code audit is claimed. Both are design references only and are absent from this implementation.",
        "",
        "## Classification",
        "",
        "Values are 10-fold mean ± population SD in percent. Hard-AUC uses historical hard predictions and equals balanced accuracy for binary labels; Probability-AUC uses continuous ASD probability. Historical rows are read-only references.",
        "",
        "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("E69.3-G2 historical", "E69.3-G2"),
        ("E62-E2 historical", "E62-E2"),
        ("E74-UDAG historical", "E74-UDAG"),
        ("G1 RUDAG-U", "G1-RUDAG-U"),
        ("G2 RUDAG-Full", "G2-RUDAG-Full"),
    ]
    for label, group in labels:
        row = summary[group]
        def metric_cell(metric):
            return f"{row[f'{metric}_mean_percent']:.4f} ± {row[f'{metric}_population_sd_percent']:.4f}"
        lines.append("|" + label + "|" + "|".join(metric_cell(m) for m in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")) + "|")

    lines.extend(["", "## Paired comparisons", "", "Differences are first model minus second model, in percentage points. Confidence intervals are descriptive only because folds share training samples and the historical protocol selects checkpoints on test ACC.", "", "|Comparison|Scope|ΔACC|ΔHard-AUC|ΔProb-AUC|ΔSEN|ΔSPE|", "|---|---|---:|---:|---:|---:|---:|"])
    pair_lookup = {(row["comparison"], row["scope"], row["metric"]): row for row in paired}
    for comparison in ("G1−E74", "G2−E74", "G2−G1", "G2−E69.3-G2 historical", "G2−E62-E2 historical"):
        for scope in ("all10", "nine_without_fold2"):
            cells = []
            for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe"):
                row = pair_lookup[(comparison, scope, metric)]
                cells.append(f"{row['mean_difference_pp']:+.4f} ({row['wins']}/{row['ties']}/{row['losses']})")
            lines.append("|" + comparison + "|" + scope + "|" + "|".join(cells) + "|")

    lines.extend(["", "Per-fold ACC differences (pp) for the predeclared core comparisons:", "", "|Fold|G1−E74|G2−E74|G2−G1|", "|---:|---:|---:|---:|"])
    acc_diffs = {}
    for comparison in ("G1−E74", "G2−E74", "G2−G1"):
        acc_diffs[comparison] = next(row for row in paired if row["comparison"] == comparison and row["scope"] == "all10" and row["metric"] == "acc")["fold_differences_pp"]
        acc_diffs[comparison] = json.loads(acc_diffs[comparison])
    for fold in range(1, 11):
        lines.append(f"|{fold}|{acc_diffs['G1−E74'][str(fold)]:+.4f}|{acc_diffs['G2−E74'][str(fold)]:+.4f}|{acc_diffs['G2−G1'][str(fold)]:+.4f}|")

    lines.extend(["", "## Graph and propagation mechanism", "", "The table reports best-checkpoint means across ten folds. Exact fold×modality values are in `mechanism.csv`; full-minus-intervention prediction effects are in `interventions.csv`.", "", "|Group|Mode|Effective edges|Scored edges|Changed edges|ΔA/A|GCN/input %|Direct/input %|Total/input %|Final gate|Train relation AUC|Unexposed test relation AUC|Scorer cls grad|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for group in r.GROUPS:
        for modality in ("PHENO", "ANAT", "FUNC", "CORR"):
            values = [m for m in mechanisms if m["group"] == group and m["modality"] == modality]
            avg = lambda key: float(np.mean([float(v[key]) for v in values]))
            lines.append(
                f"|{group}|{modality}|{avg('original_effective_edges'):.1f}|{avg('scored_edges'):.1f}|{avg('modified_edges'):.1f}|"
                f"{avg('adjacency_change_relative'):.5g}|{100*avg('gcn_message_raw_l2_ratio'):.6g}|"
                f"{100*avg('weighted_direct_message_raw_l2_ratio'):.6g}|{100*avg('total_message_raw_l2_ratio'):.6g}|"
                f"{avg('effective_gate_final'):.6g}|{avg('train_relation_pair_auc'):.6f}|"
                f"{avg('unexposed_test_relation_pair_auc_posthoc'):.6f}|{avg('relation_scorer_classification_gradient_norm'):.3e}|"
            )

    lines.extend(["", "### Fixed-checkpoint interventions", "", "All intervention metrics use the same checkpoint, test fold and model weights. Positive ACC deltas mean the trained full path exceeds the named intervention.", "", "|Group|Intervention|Mean full−intervention ACC pp|Hard flips|Corrected / newly wrong|Mean abs ΔP(ASD)|Mean fused relative Δ|", "|---|---|---:|---:|---:|---:|---:|"])
    for group in r.GROUPS:
        names = ("disease_neutral", "corr_neutral", "score_permuted", "direct_off") if group == "G2-RUDAG-Full" else ("disease_neutral", "corr_neutral", "score_permuted")
        for name in names:
            values = [row for row in interventions if row["group"] == group and row["intervention"] == name]
            corrected = sum(row["full_wrong_intervention_correct"] for row in values)
            newly_wrong = sum(row["full_correct_intervention_wrong"] for row in values)
            lines.append(
                f"|{group}|{name}|{np.mean([row['full_minus_intervention_acc_pp'] for row in values]):+.4f}|"
                f"{sum(row['hard_flips'] for row in values)}|{corrected} / {newly_wrong}|"
                f"{np.mean([row['mean_abs_asd_probability_change'] for row in values]):.6g}|"
                f"{np.mean([row['fused_320_relative_l2_change'] for row in values]):.6g}|"
            )

    acc_g1_e74 = next(row for row in paired if row["comparison"] == "G1−E74" and row["scope"] == "nine_without_fold2" and row["metric"] == "acc")
    acc_g2_e74 = next(row for row in paired if row["comparison"] == "G2−E74" and row["scope"] == "nine_without_fold2" and row["metric"] == "acc")
    acc_g2_e62 = next(row for row in paired if row["comparison"] == "G2−E62-E2 historical" and row["scope"] == "all10" and row["metric"] == "acc")
    g2_summary = summary["G2-RUDAG-Full"]
    neutral_g2 = [row for row in interventions if row["group"] == "G2-RUDAG-Full" and row["intervention"] == "disease_neutral"]
    perm_g2 = [row for row in interventions if row["group"] == "G2-RUDAG-Full" and row["intervention"] == "score_permuted"]
    disease_effect = sum(row["hard_flips"] for row in neutral_g2) > 0
    permutation_effect = sum(row["hard_flips"] for row in perm_g2) > 0
    performance_target = g2_summary["acc_mean_percent"] >= summary["E62-E2"]["acc_mean_percent"]
    clean_superiority = acc_g2_e74["mean_difference_pp"] > 0
    if performance_target and clean_superiority and disease_effect and permutation_effect:
        judgment = "GO"
    elif (disease_effect or permutation_effect or acc_g2_e74["mean_difference_pp"] > 0 or g2_summary["acc_mean_percent"] >= summary["E74-UDAG"]["acc_mean_percent"]):
        judgment = "PARTIAL"
    else:
        judgment = "NO-GO"
    lines.extend([
        "", "## Audit and interpretation", "",
        f"Independent readback passed {manifest['independent_audits']}/20 best checkpoints. Each loaded checkpoint reproduced stored metrics and hard predictions; probability vectors matched within 1e-8. G1/G2 had identical initialization and optimizer-state hashes in every fold. Shared-epoch RNG matches: G1/G2 {rng['totals']['g1_g2_match']}/{rng['totals']['g1_g2_common']}; versus E74 {rng['totals']['e74_match']}/{rng['totals']['e74_common']} across both groups.",
        f"G2−E74 nine-fold ACC is {acc_g2_e74['mean_difference_pp']:+.4f} pp; G1−E74 nine-fold ACC is {acc_g1_e74['mean_difference_pp']:+.4f} pp. G2−E62-E2 historical ten-fold ACC is {acc_g2_e62['mean_difference_pp']:+.4f} pp. The E62/E69/E74 rows differ in training lineage or include the deliberate fold-2 label exposure, so only the same-checkpoint interventions and G1/G2 retraining are direct controls.",
        f"Fold 2 intentionally trains on rows 377 (ASD) and 407 (TD) while retaining them in test. Their relation pairs and classification labels enter training; the full-fold result is contaminated. The nine-fold table removes fold 2 only and still retains test-ACC checkpoint selection, pre-CV feature selection/standardization and transductive use of the 871-node graph.",
        f"RUDAG final decision: **{judgment}**. G2 reaches the E62-E2 ACC reference: `{performance_target}`; clean-nine-fold ACC exceeds E74: `{clean_superiority}`; Disease-neutral changes hard predictions: `{disease_effect}`; Score-permuted changes hard predictions: `{permutation_effect}`. This judgment follows the observed data and does not imply unbiased external generalization.",
        "",
        "## Next question",
        "",
        "The next scientifically valuable test is whether the observed CORR score correspondence remains useful under an evaluation protocol with fold-local preprocessing and a separate validation set for checkpoint selection. The present result should not trigger within-experiment tuning of the fixed 0.8 edge strength, 0.02 direct scale, WD, learning rates or Top-k.",
        "",
        "## Files and sources",
        "",
        f"- Experiment directory: `{EXP}`",
        "- Formal checkpoints: `.runtime/checkpoints/*.best.pt`; recovery states are removed after a complete audit.",
        "- Core outputs: `results/fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `mechanism.csv`, `interventions.csv`, `checkpoint_audit.csv`, `manifest.json`.",
        "- SPromptGL paper: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
        "- BrainPrompt paper: https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf",
        "- BrainPrompt official source: https://github.com/AngusMonroe/BrainPrompt",
        "- SABER paper: https://arxiv.org/pdf/2607.01901",
        "- Locked E69.3-G2 source commit: https://github.com/tangaii/SPromptGL/tree/41dbcb45a0273673857051d8e821b8c3315eef00/experiment/experiment69_3_g2_two_rows",
        "",
    ])
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize():
    import torch

    r = runner()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("cannot independently audit without a passing preflight")
    expected_sources = preflight["source_sha256"]
    if r.source_files() != expected_sources:
        raise RuntimeError("protected source/reference/warmup SHA changed during training")
    audits = []
    failure = None
    for group in r.GROUPS:
        for fold in range(1, 11):
            try:
                audits.append(audit_fold(group, fold, expected_sources))
                print(f"independent checkpoint audit {len(audits)}/20 PASS", flush=True)
            except Exception as exc:
                failure = {"group": group, "fold": fold, "exception": repr(exc), "traceback": traceback.format_exc()}
                print(f"independent checkpoint audit FAIL {group} fold={fold}: {exc}", flush=True)
                break
        if failure:
            break
    if failure:
        interim = {"status": "AUDIT_FAILED", "expected_tasks": 20, "completed_tasks": sum(r._task_complete(g, f, expected_sources) for g in r.GROUPS for f in range(1, 11)),
                   "independent_audits": len(audits), "failure": failure, "source_sha256": expected_sources}
        r.atomic_json(r.EXP / "manifest.json", interim)
        r.atomic_json(RESULTS / "audit_status.json", interim)
        raise RuntimeError(f"independent audit failed: {failure}")

    rng = _rng_audit(audits)
    summaries = []
    summaries.append(_summary_history(_historical_summary(BASE / "results/summary.csv", "E69.3-G2"), "E69.3-G2", "historical read-only; fold 2 label exposure"))
    summaries.append(_summary_history(_historical_summary(E62 / "results/summary.csv", "E62-E2"), "E62-E2", "historical read-only"))
    summaries.append(_summary_history(_historical_summary(E74 / "results/summary.csv", "E74-UDAG"), "E74-UDAG", "historical read-only"))
    summaries.extend(_summary_new(group, audits) for group in r.GROUPS)

    histories = {
        "E69.3-G2 historical": _history_rows(BASE / "results/fold_metrics.csv", "E69.3-G2"),
        "E62-E2 historical": _history_rows(E62 / "results/fold_metrics.csv", "E62-E2"),
        "E74-UDAG historical": _history_rows(E74 / "results/fold_metrics.csv", "E74-UDAG"),
    }
    new_by_group = {}
    for group in r.GROUPS:
        new_by_group[group] = {a["fold"]: a["metrics"] for a in audits if a["group"] == group}
    e74 = histories["E74-UDAG historical"]
    e69 = histories["E69.3-G2 historical"]
    e62 = histories["E62-E2 historical"]
    scopes = [("all10", list(range(1, 11))), ("nine_without_fold2", [f for f in range(1, 11) if f != 2])]
    paired = []
    paired.extend(_paired_rows("G1−E74", new_by_group[r.GROUPS[0]], e74, scopes))
    paired.extend(_paired_rows("G2−E74", new_by_group[r.GROUPS[1]], e74, scopes))
    paired.extend(_paired_rows("G2−G1", new_by_group[r.GROUPS[1]], new_by_group[r.GROUPS[0]], scopes))
    paired.extend(_paired_rows("G2−E69.3-G2 historical", new_by_group[r.GROUPS[1]], e69, scopes))
    paired.extend(_paired_rows("G2−E62-E2 historical", new_by_group[r.GROUPS[1]], e62, scopes))

    fold_rows = []
    for audit in audits:
        row = {"group": audit["group"], "fold": audit["fold"], "status": audit["status"],
               **audit["metrics"], "best_epoch_1based": audit["best_epoch_1based"],
               "epochs_executed": audit["epochs_executed"], "checkpoint_sha256": audit["checkpoint_sha256"],
               "train_test_overlap_1based": json.dumps(audit["train_test_overlap_1based"])}
        fold_rows.append(row)
    mechanisms = [row for audit in audits for row in audit["mechanism"]]
    interventions = [row for audit in audits for row in audit["interventions"]]
    checkpoint_rows = []
    for audit in audits:
        checkpoint_rows.append({key: value for key, value in audit.items() if key not in ("metrics", "rng_trace", "interventions", "mechanism")})

    r.write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    r.write_csv(RESULTS / "summary.csv", summaries)
    r.write_csv(RESULTS / "paired_comparisons.csv", paired)
    r.write_csv(RESULTS / "mechanism.csv", mechanisms)
    r.write_csv(RESULTS / "interventions.csv", interventions)
    r.write_csv(RESULTS / "checkpoint_audit.csv", checkpoint_rows)

    current_manifest = {}
    if (r.EXP / "manifest.json").is_file():
        current_manifest = json.loads((r.EXP / "manifest.json").read_text(encoding="utf-8"))
    training_failures = current_manifest.get("failed_tasks", [])
    attempts = current_manifest.get("failed_attempt_records", 0)
    oom_events = current_manifest.get("oom_events", 0)
    max_workers = current_manifest.get("max_workers_observed", 0)
    retries = current_manifest.get("retries", [])
    if len(audits) != 20 or any(a["status"] != "INDEPENDENT_AUDIT_PASS" for a in audits):
        raise RuntimeError("not all 20 independent audits passed")
    manifest = {
        "experiment": r.CFG["experiment"], "status": "COMPLETE_AUDITED",
        "experiment_root": str(r.EXP), "preferred_root": r.CFG["preferred_root"],
        "root_fallback_reason": r.CFG["root_fallback_reason"],
        "expected_tasks": 20, "completed_tasks": 20, "independent_audits": 20,
        "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "worker_limit": 10, "max_workers_observed": max_workers,
        "failed_attempt_records": attempts, "oom_events": oom_events,
        "formal_training_failures": len(training_failures),
        "training_execution": {
            "retry_count": attempts, "retries": retries,
            "failures": training_failures, "oom_events": oom_events,
            "worker_limit": current_manifest.get("worker_limit", 0),
            "wall_seconds": current_manifest.get("wall_seconds"),
        },
        "warmup_retrained": False,
        "checkpoint_audit": "PASS 20/20",
        "preflight": preflight,
        "rng_audit": rng,
        "protected_source_sha256": expected_sources,
        "tasks": [{"group": audit["group"], "fold": audit["fold"], "status": audit["status"],
                   "checkpoint_sha256": audit["checkpoint_sha256"], "warmup_sha256": audit["warmup_sha256"],
                   "best_epoch_1based": audit["best_epoch_1based"], "epochs_executed": audit["epochs_executed"],
                   "model_cam_parameter_sha": audit["model_cam_parameter_sha"]} for audit in audits],
        "results_sha256": {name: r.sha_file(RESULTS / name) for name in (
            "fold_metrics.csv", "summary.csv", "paired_comparisons.csv", "mechanism.csv",
            "interventions.csv", "checkpoint_audit.csv",
        )},
    }
    r.atomic_json(r.EXP / "manifest.json", manifest)
    r.atomic_json(RESULTS / "audit_status.json", {"status": "PASS", "checkpoint_audits": checkpoint_rows,
                                                  "protected_source_sha_pass": True, "rng_audit": rng})
    _write_report(summaries, paired, audits, mechanisms, interventions, manifest, rng)
    manifest["results_sha256"]["report.md"] = r.sha_file(RESULTS / "report.md")
    r.atomic_json(r.EXP / "manifest.json", manifest)

    # Retain only best checkpoints and the per-task proof; current recovery state and logs are no longer needed.
    for path in (r.CKPTS.glob("*.latest.pt")):
        path.unlink()
    resume = r.RT / "preflight_resume.pt"
    if resume.exists():
        resume.unlink()
    print("AUDIT COMPLETE 20/20", flush=True)
    return manifest


if __name__ == "__main__":
    finalize()
