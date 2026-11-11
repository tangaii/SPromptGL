from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dcpss import DCPSS, deterministic_assignments, initialize
import run as R


def _base_only_update(state):
    R.hist().zero_optimizers(state)
    out = R.base().forward(state, R.BASE_GROUP, training=True)
    logp = out["logp"].detach().cpu().clone()
    loss = float(out["total"].detach().cpu())
    out["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    return logp, loss, R._bundle(state), R.runtime().rng_sha(R.runtime().pack_rng())


def _synthetic_loo_test(device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(780119)
        h = torch.randn(13, 144, device=device, requires_grad=True)
        text = torch.randn(2, 1024, device=device)
        module = DCPSS(text, [0, 1], 3).to(device)
        q = torch.nn.functional.normalize(module.patient_projection(h), dim=-1)
        labels = torch.tensor([0] * 6 + [1] * 6 + [1], device=device)
        support = np.arange(12)
        query = torch.arange(13, device=device)
        assignment = torch.tensor([0, 0, 1, 1, 2, 2] * 2 + [-1], device=device)
        prototypes, counts = module._query_prototypes(q, labels, support, assignment, query)
        for i in range(12):
            cls = int(labels[i])
            own_cluster = int(assignment[i])
            expected = torch.full((2, 3), 2.0, device=device)
            expected[cls, own_cluster] = 1.0
            if not torch.equal(counts[i], expected):
                raise AssertionError("query's own support contribution was not removed exactly once")
        if not torch.all(counts[12] == 2):
            raise AssertionError("unknown query was incorrectly excluded from a prototype")
        loss = prototypes.square().sum()
        grad = torch.autograd.grad(loss, h)[0]
        if not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
            raise AssertionError("LOO prototype aggregation is not differentiable")
        return {"leave_one_out_counts": True, "unknown_query_uses_full_support": True, "prototype_gradient_norm": float(grad.norm())}


def _active_gradient_test():
    group = "E78-G2-SPromptGL-K3"
    state, warmup, identity, support, assignment, module, optimizer, initial = R.initial_state(1, group)
    if type(state["cam"]).__name__ != "PLNNoPEN":
        raise AssertionError(f"unexpected base CAM implementation {type(state['cam']).__name__}")
    rng_before_module_path = R.runtime().rng_sha(R.runtime().pack_rng())
    R.hist().zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    out = R.forward_with_cam144(state, training=True)
    rng_after_base_forward = R.runtime().rng_sha(R.runtime().pack_rng())
    qids = torch.as_tensor(R._legal_indices(state), dtype=torch.long, device=out["cam144"].device)
    dcp = module(out["cam144"], out["logp"], state["y"], support, assignment, qids)
    merged = out["logp"].clone()
    merged[qids] = dcp["logp"]
    cls_final = R.runtime().runtime()["utils"].ClsLoss(merged, state["y"], state["train"], state["weights"])
    _, aux = module.losses(dcp, out["logp"], state["y"], state["weights"])
    total = out["total"] - out["cls"] + cls_final + 0.05 * aux["sem"] + 0.02 * aux["gain"]
    rng_after_module_forward = R.runtime().rng_sha(R.runtime().pack_rng())
    if rng_after_module_forward != rng_after_base_forward:
        raise AssertionError("DCPSS forward changed the public RNG after the base forward")
    targets = {
        "darg": state["model"].adaptive_gate_logits,
            "cam_encoder": state["cam"].Encoder[0].w_q.weight,
        "patient_projection": module.patient_projection.weight,
        "prompt_projection": module.text_projection.weight,
        "residual_alpha": module.alpha,
    }
    grads = torch.autograd.grad(total, tuple(targets.values()), retain_graph=True, allow_unused=True)
    norms = {}
    for (name, _), grad in zip(targets.items(), grads):
        norms[name] = 0.0 if grad is None else float(grad.detach().norm().cpu())
        if grad is None or not torch.isfinite(grad).all() or norms[name] <= 0:
            raise AssertionError(f"active end-to-end gradient failed at {name}: {norms[name]}")
    correction_grads = torch.autograd.grad(
        dcp["delta"].sum(),
        (state["model"].adaptive_gate_logits, state["cam"].Encoder[0].w_q.weight),
        retain_graph=True,
        allow_unused=True,
    )
    correction_gradient_norms = {
        "darg": 0.0 if correction_grads[0] is None else float(correction_grads[0].detach().norm().cpu()),
        "cam_encoder": 0.0 if correction_grads[1] is None else float(correction_grads[1].detach().norm().cpu()),
    }
    if any(value <= 0 or not math.isfinite(value) for value in correction_gradient_norms.values()):
        raise AssertionError(f"DCPSS correction itself does not reach DARG/CAM: {correction_gradient_norms}")
    if int(len(qids)) != len(support) or len(np.intersect1d(support, state["test_np"])):
        raise AssertionError("training Query/Support membership is invalid")
    if not torch.allclose(out["logp"].exp().sum(-1), torch.ones(871, device=out["logp"].device), atol=1e-6):
        raise AssertionError("base output is not normalized log probability")
    test_ids = np.asarray(state["test_np"], dtype=np.int64)
    flipped_labels = state["y"].clone()
    flipped_labels[torch.as_tensor(test_ids, device=flipped_labels.device)] = 1 - flipped_labels[torch.as_tensor(test_ids, device=flipped_labels.device)]
    module.eval()
    with torch.no_grad():
        a = module(out["cam144"], out["logp"], state["y"], support, assignment, test_ids)["logp"]
        b = module(out["cam144"], out["logp"], flipped_labels, support, assignment, test_ids)["logp"]
    test_label_invariance_max_abs_diff = float((a - b).abs().max().cpu())
    if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"test labels influence DCPSS predictions: max_abs_diff={test_label_invariance_max_abs_diff}")
    R.runtime().restore_rng(warmup["rng"])
    return {
        "status": "PASS",
        "base_class": type(state["cam"]).__name__,
        "public_rng_preserved_by_module_init_and_forward": rng_after_module_forward == rng_after_base_forward,
        "base_forward_rng_digest_before_after": {"before": rng_before_module_path, "after_base": rng_after_base_forward},
        "test_label_invariance": True,
        "test_label_invariance_max_abs_diff": test_label_invariance_max_abs_diff,
        "gradient_norms": norms,
        "correction_only_gradient_norms": correction_gradient_norms,
        "losses": {k: float(v.detach().cpu()) for k, v in aux.items()},
        "query_count": len(support),
        "test_count": len(test_ids),
        "initial_identity": identity,
        "module_parameter_count": initial["new_parameter_count"],
    }


def _two_step_replay():
    group = "E78-G2-SPromptGL-K3"
    state, _, identity, support, assignment, module, optimizer, initial = R.initial_state(1, group)
    start = R._snapshot(state, module, optimizer)

    def replay_two():
        records = []
        for _ in range(2):
            loss, diag = R._train_step(state, module, optimizer, support, assignment)
            evaluated = R._evaluate(state, module, support, assignment)
            records.append((loss, diag, R._compact_metrics(evaluated["metrics"])))
        return R._snapshot(state, module, optimizer), records

    first, first_records = replay_two()
    R._restore(state, module, optimizer, start)
    second, second_records = replay_two()
    if R.sha_object(first) != R.sha_object(second) or first_records != second_records:
        component_match = {
            "base_state": R.sha_object(first["base_state"]) == R.sha_object(second["base_state"]),
            "module_state": R.sha_object(first["module_state"]) == R.sha_object(second["module_state"]),
            "new_optimizer": R.sha_object(first["new_optimizer"]) == R.sha_object(second["new_optimizer"]),
            "rng": R.runtime().rng_sha(first["rng"]) == R.runtime().rng_sha(second["rng"]),
            "records": first_records == second_records,
        }
        raise AssertionError(f"two-step model/optimizer/RNG replay did not match: {component_match}")
    return {"status": "PASS", "fold": 1, "group": group, "replayed_steps": 2, "initial": initial}


def preflight():
    for path in (R.RESULTS, R.CHECKPOINTS, R.TASKS, R.LOGS, R.HERE / "runtime/failures"):
        path.mkdir(parents=True, exist_ok=True)
    if R.BASE.resolve() != Path(R.CFG["base_path"]).resolve():
        raise RuntimeError("the resolved base directory does not match the migrated E74.4 path")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one CUDA device, got available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    for name, expected in R.CFG["base_sha256"].items():
        actual = R.sha_file(R.BASE / name)
        if actual != expected:
            raise RuntimeError(f"E74.4 core source SHA mismatch for {name}: {actual}")
    if R.sha_file(R.HERE / "runtime/text_embeddings.pt") != R.CFG["text_cache_sha256"]:
        raise RuntimeError("frozen RN50 text-cache SHA mismatch")
    if len(R.CFG["groups"]) != 5 or len(set(R.CFG["groups"])) != 5:
        raise RuntimeError("formal five-group configuration is not unique")

    source_hash = R.source_sha()
    fold_checks = []
    initial_params_by_fold = {}
    assignments_by_fold = {}
    for fold in range(1, 11):
        state, warmup, identity = R.base().prepare(fold)
        expected_overlap = [377, 407] if fold == 2 else []
        if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
            raise RuntimeError(f"Fold {fold} overlap differs from the historical E74.4 record")
        support = np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False)
        if len(np.intersect1d(support, state["test_np"])) or len(support) != len(np.unique(support)):
            raise RuntimeError(f"Fold {fold} DCPSS support isolation failed")
        rng_before = R.runtime().rng_sha(R.runtime().pack_rng())
        h0 = R.warmup_features(state, rng_before)
        assignments = {}
        parameter_hashes = {}
        for group in R.GROUPS:
            spec = R.CFG["group_specs"][group]
            assign = deterministic_assignments(h0, state["y"], support, spec["k"])
            assignments[group] = R.sha_object(assign.cpu())
            embeddings, classes = R.text_payload(group)
            module = initialize(embeddings, classes, spec["k"], fold, device)
            parameter_hashes[group] = R.sha_object({k: v.detach().cpu() for k, v in module.named_parameters()})
            if sum(p.numel() for p in module.parameters()) != R.CFG["method"]["parameter_count_expected"]:
                raise RuntimeError(f"unexpected DCPSS parameter count in {group}")
            del module
        if len(set(parameter_hashes.values())) != 1:
            raise RuntimeError(f"new parameter initialization differs among groups in fold {fold}")
        k3 = [g for g in R.GROUPS if R.CFG["group_specs"][g]["k"] == 3]
        if len({assignments[g] for g in k3}) != 1:
            raise RuntimeError(f"K=3 initial prototype membership differs across groups in fold {fold}")
        if R.runtime().rng_sha(R.runtime().pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError(f"Fold {fold} warmup assignment or module init consumed public RNG")
        initial_params_by_fold[fold] = parameter_hashes
        assignments_by_fold[fold] = assignments
        fold_checks.append({
            "fold": fold,
            "identity": identity,
            "legal_support_count": int(len(support)),
            "test_count": int(len(state["test_np"])),
            "overlap_1based": expected_overlap,
            "cam_shape": list(h0.shape),
            "cam_class": type(state["cam"]).__name__,
            "prompt_init_equal": True,
            "new_parameter_count": R.CFG["method"]["parameter_count_expected"],
            "k3_cluster_assignment_sha256": assignments[k3[0]],
        })
        del state, h0
        torch.cuda.empty_cache()

    parity = []
    for fold in range(1, 11):
        state_ref, _, ident_ref = R.base().prepare(fold)
        out_ref = R.base()._one_update(state_ref, R.runtime(), R.BASE_GROUP)
        reference_logp, reference_loss, reference_bundle, reference_rng = out_ref
        reference_bundle_sha = R.sha_object(reference_bundle)
        del state_ref, out_ref
        torch.cuda.empty_cache()
        state_new, _, ident_new = R.base().prepare(fold)
        if ident_ref != ident_new:
            raise RuntimeError(f"base initial identity changed across parity replay in fold {fold}")
        new_logp, new_loss, new_bundle, new_rng = _base_only_update(state_new)
        checks = {
            "fold": fold,
            "forward_exact": torch.equal(reference_logp, new_logp),
            "loss_exact": reference_loss == new_loss,
            "parameters_and_optimizer_exact": reference_bundle_sha == R.sha_object(new_bundle),
            "rng_exact": reference_rng == new_rng,
        }
        if not all(v for k, v in checks.items() if k != "fold"):
            raise RuntimeError(f"base-disabled one-step parity failed: {checks}")
        parity.append(checks)
        del state_new
        torch.cuda.empty_cache()

    synthetic = _synthetic_loo_test(device)
    active = _active_gradient_test()
    replay = _two_step_replay()
    result = {
        "status": "PASS",
        "experiment": R.CFG["experiment"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "base_group": R.BASE_GROUP,
        "base_path": str(R.BASE),
        "base_core_sha256": R.CFG["base_sha256"],
        "source_sha256": source_hash,
        "text_cache_sha256": R.CFG["text_cache_sha256"],
        "folds": fold_checks,
        "base_disabled_single_step_parity": parity,
        "all_ten_fold_base_parity_pass": True,
        "loo_unit_test": synthetic,
        "active_gradient_test": active,
        "two_step_recovery_replay": replay,
        "protocol": R.CFG["protocol"],
        "parameter_count_by_group": initial_params_by_fold[1],
        "k3_assignments_by_group_fold1": assignments_by_fold[1],
    }
    R.save_json(R.RESULTS / "preflight.json", result)
    print(f"PREFLIGHT PASS: source={source_hash} device={torch.cuda.get_device_name(0)}", flush=True)
    return result


def _metric_close(a, b, atol=1e-12):
    keys = set(a) | set(b)
    for key in keys:
        if key not in a or key not in b:
            return False
        av, bv = a[key], b[key]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if abs(float(av) - float(bv)) > atol:
                return False
        elif av != bv:
            return False
    return True


def _metric_pct(metrics):
    return {
        "acc": 100 * float(metrics["acc"]),
        "hard_auc": 100 * float(metrics["hard_auc"]),
        "prob_auc": 100 * float(metrics["prob_auc"]),
        "sen": 100 * float(metrics["sen"]),
        "spe": 100 * float(metrics["spe"]),
        "asd_f1": 100 * float(metrics["asd_f1"]),
        "tp": int(metrics["asd_tp"]),
        "tn": int(metrics["td_tn"]),
        "fp": int(metrics["td_fp"]),
        "fn": int(metrics["asd_fn"]),
    }


def _attention_stats(attention):
    rows = []
    for cls, values in enumerate(attention):
        a = torch.as_tensor(values, dtype=torch.float64)
        entropy = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1)
        rows.append({
            "class": cls,
            "prompt_count": int(a.shape[-1]),
            "entropy_mean": float(entropy.mean()),
            "entropy_max": float(math.log(max(int(a.shape[-1]), 1))),
            "weight_mean": a.mean(0).tolist(),
            "weight_patient_sd": a.std(0, unbiased=False).tolist(),
            "argmax_frequency": torch.bincount(a.argmax(-1), minlength=a.shape[-1]).float().div(len(a)).tolist(),
        })
    return rows


def _prototype_stats(dcp, support, assignments, labels):
    proto = dcp["prototype"].detach()
    counts = dcp["counts"].detach()
    k = int(proto.shape[2])
    support_counts = {}
    for cls in (0, 1):
        support_counts[str(cls)] = [int(np.sum((labels[support] == cls) & (assignments[support] == c))) for c in range(k)]
    centers = proto.mean(0)
    cross_similarity = torch.nn.functional.normalize(centers[0], dim=-1) @ torch.nn.functional.normalize(centers[1], dim=-1).T
    between = float((1 - cross_similarity).mean())
    within = {}
    collapse = {}
    for cls in (0, 1):
        values = centers[cls]
        if k < 2:
            within[str(cls)] = float("nan")
            collapse[str(cls)] = float("nan")
        else:
            similarities = torch.nn.functional.normalize(values, dim=-1) @ torch.nn.functional.normalize(values, dim=-1).T
            off = similarities[~torch.eye(k, dtype=torch.bool, device=similarities.device)]
            within[str(cls)] = float((1 - off).mean())
            collapse[str(cls)] = float((off > 0.98).float().mean())
    beta = dcp["beta"].detach().clamp_min(1e-12)
    beta_entropy = -(beta * beta.log()).sum(-1).mean(dim=0)
    return {
        "support_counts_by_class_cluster": support_counts,
        "prototype_counts_min": float(counts.min()),
        "prototype_between_class_mean_cosine_distance": between,
        "prototype_within_class_mean_cosine_distance": within,
        "prototype_collapse_fraction_cosine_gt_0_98": collapse,
        "prototype_match_entropy_by_class": beta_entropy.cpu().tolist(),
        "prototype_centers": centers.cpu().tolist(),
    }


def _intervention(module, h, base_logp, labels, support, assignments, test_ids, full_metrics, base_metrics, anon_text, anon_classes, group, fold):
    outputs = {}
    ids = torch.as_tensor(test_ids, dtype=torch.long, device=h.device)
    with torch.no_grad():
        full = module(h, base_logp, labels, support, assignments, ids)
        variants = {
            "prototype_off": {"prototype_off": True},
            "semantic_off": {"semantic_off": True},
            "prompt_class_swap": {"prompt_swap": True},
        }
        for name, kwargs in variants.items():
            outputs[name] = module(h, base_logp, labels, support, assignments, ids, **kwargs)["logp"]
        original_text = module.text_embeddings.clone()
        original_classes = module.text_classes.clone()
        module.text_embeddings.copy_(anon_text.to(module.text_embeddings.device))
        module.text_classes.copy_(anon_classes.to(module.text_classes.device))
        outputs["anonymous_text_swap"] = module(h, base_logp, labels, support, assignments, ids)["logp"]
        module.text_embeddings.copy_(original_text)
        module.text_classes.copy_(original_classes)
        outputs["correction_off"] = base_logp[ids]
        full_logp = base_logp.clone()
        full_logp[ids] = full["logp"]
        y = labels.detach().cpu().numpy()[test_ids]
        base_pred = base_logp[ids].argmax(-1).cpu().numpy()
        full_pred = full["logp"].argmax(-1).cpu().numpy()
        base_correct = base_pred == y
        full_correct = full_pred == y
        baseline = R._metric_values(full_logp, test_ids, labels)
        records = []
        for name, output in outputs.items():
            merged = base_logp.clone()
            merged[ids] = output
            metric = R._metric_values(merged, test_ids, labels)
            prediction = output.argmax(-1).cpu().numpy()
            records.append({
                "group": group,
                "fold": fold,
                "intervention": name,
                "delta_acc_pp": 100 * (metric["acc"] - baseline["acc"]),
                "delta_hard_auc_pp": 100 * (metric["hard_auc"] - baseline["hard_auc"]),
                "delta_prob_auc_pp": 100 * (metric["prob_auc"] - baseline["prob_auc"]),
                "delta_sen_pp": 100 * (metric["sen"] - baseline["sen"]),
                "delta_spe_pp": 100 * (metric["spe"] - baseline["spe"]),
                "prediction_flips": int(np.sum(prediction != full_pred)),
                "correct_to_wrong": int(np.sum(full_correct & (prediction != y))),
                "wrong_to_correct": int(np.sum(~full_correct & (prediction == y))),
                "mean_abs_probability_change": float((output.exp()[:, 0] - full["logp"].exp()[:, 0]).abs().mean().cpu()),
                "mean_true_class_margin_change": float(np.mean(
                    (output.gather(1, torch.as_tensor(y, device=output.device).unsqueeze(1)).squeeze(1)
                     - output.gather(1, torch.as_tensor(1-y, device=output.device).unsqueeze(1)).squeeze(1)
                     - full["logp"].gather(1, torch.as_tensor(y, device=output.device).unsqueeze(1)).squeeze(1)
                     + full["logp"].gather(1, torch.as_tensor(1-y, device=output.device).unsqueeze(1)).squeeze(1)).cpu().numpy()
                )),
            })
        return _prototype_stats(full, support, assignments.cpu().numpy(), labels.detach().cpu().numpy()), _attention_stats(full["attention"]), records, {
            "base_correct": int(base_correct.sum()),
            "full_correct": int(full_correct.sum()),
            "corrected": int((~base_correct & full_correct).sum()),
            "harmed": int((base_correct & ~full_correct).sum()),
            "mean_abs_delta": float(full["delta"].abs().mean().cpu()),
            "max_abs_delta": float(full["delta"].abs().max().cpu()),
            "mean_gamma": float(full["gamma"].cpu()),
            "full_metrics": _metric_pct(full_metrics),
            "base_metrics": _metric_pct(base_metrics),
        }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _fmt(mean, sd):
    return f"{mean:.4f}±{sd:.4f}"


def _paired_rows(fold_rows, base_rows):
    by_group = {group: {int(row["fold"]): row for row in fold_rows if row["group"] == group} for group in R.GROUPS}
    base = {int(row["fold"]): row for row in base_rows}
    metrics = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
    comparisons = [("G2-G1", R.GROUPS[1], R.GROUPS[0]), ("G4-G3", R.GROUPS[3], R.GROUPS[2]),
                   ("G1-G3", R.GROUPS[0], R.GROUPS[2]), ("G2-G4", R.GROUPS[1], R.GROUPS[3]),
                   ("G2-G5", R.GROUPS[1], R.GROUPS[4])]
    for group in R.GROUPS:
        comparisons.append((f"{group}-E74.4-G2", group, "BASE"))
    rows, summary = [], []
    for name, left, right in comparisons:
        per_metric = {m: [] for m in metrics}
        for fold in range(1, 11):
            lrow = by_group[left][fold]
            rrow = base[fold] if right == "BASE" else by_group[right][fold]
            row = {"comparison": name, "fold": fold}
            for metric in metrics:
                diff = float(lrow[metric]) - float(rrow[metric])
                row[f"delta_{metric}_pp"] = diff
                per_metric[metric].append(diff)
            rows.append(row)
        for metric, diffs in per_metric.items():
            eps = 1e-12
            wins = sum(v > eps for v in diffs)
            ties = sum(abs(v) <= eps for v in diffs)
            losses = sum(v < -eps for v in diffs)
            summary.append({
                "comparison": name,
                "metric": metric,
                "mean_delta_pp": float(np.mean(diffs)),
                "fold2_excluded_mean_delta_pp": float(np.mean([v for i, v in enumerate(diffs, 1) if i != 2])),
                "W": wins,
                "T": ties,
                "L": losses,
            })
    return rows, summary


def finalize():
    preflight_path = R.RESULTS / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected_source = R.source_sha()
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != expected_source:
        raise RuntimeError("final audit source/preflight identity mismatch")
    fold_rows, mechanism_rows, audit_rows, intervention_rows = [], [], [], []
    source_checks = []
    rng_common = {}
    base_refs = {fold: R.base_reference(fold) for fold in range(1, 11)}
    for group in R.GROUPS:
        k = int(R.CFG["group_specs"][group]["k"])
        embeddings, classes = R.text_payload(group)
        for fold in range(1, 11):
            task_path, checkpoint_path, latest_path, log_path = R.task_paths(group, fold)
            if not all(p.is_file() for p in (task_path, checkpoint_path, latest_path, log_path)):
                raise RuntimeError(f"missing formal task artifact: {group} fold={fold}")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("source_sha256") != expected_source:
                raise RuntimeError(f"formal task is incomplete or has different sources: {group} fold={fold}")
            checkpoint_sha = R.sha_file(checkpoint_path)
            if checkpoint_sha != task.get("checkpoint_sha256"):
                raise RuntimeError(f"best checkpoint SHA mismatch: {group} fold={fold}")
            cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            if cp["source_sha256"] != expected_source or cp["group"] != group or cp["fold"] != fold:
                raise RuntimeError("checkpoint metadata identity mismatch")
            if cp["initial"]["identity"] != cp["identity"] or cp["identity"]["warmup_sha256"] != cp["warmup_sha256"]:
                raise RuntimeError("checkpoint base initialization/warmup identity mismatch")
            if latest["source_sha256"] != expected_source or latest["group"] != group or latest["fold"] != fold:
                raise RuntimeError("latest checkpoint metadata identity mismatch")
            if latest["base_state"].keys() != {"model", "cam", "opt_base", "opt_cam", "opt_added"}:
                raise RuntimeError("latest checkpoint is missing part of the original optimizer bundle")
            if set(cp["base_state"]) != {"model", "cam", "opt_base", "opt_cam", "opt_added"}:
                raise RuntimeError("best checkpoint is missing base parameters or optimizer state")
            if set(cp["module_state"]) != {"text_embeddings", "text_classes", "alpha", "patient_projection.weight", "text_projection.weight"}:
                raise RuntimeError("best checkpoint is missing DCPSS state")

            state, warmup, identity = R.base().prepare(fold)
            if identity != cp["identity"]:
                raise RuntimeError("independent base warmup identity differs from checkpoint")
            legal = np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False)
            support = np.asarray(cp["support_ids"], dtype=np.int64)
            test_ids = np.asarray(cp["test_ids"], dtype=np.int64)
            assign = torch.as_tensor(cp["assignments"], dtype=torch.long, device=state["y"].device)
            if not np.array_equal(support, legal) or len(np.intersect1d(support, state["test_np"])):
                raise RuntimeError("best checkpoint prototype support is not exactly legal outer-train")
            if np.any(assign.cpu().numpy()[np.setdiff1d(np.arange(871), support)] != -1):
                raise RuntimeError("non-support or outer-test patient received a prototype cluster")
            warmup_rng = R.runtime().rng_sha(R.runtime().pack_rng())
            warmup_h = R.warmup_features(state, warmup_rng)
            expected_assignments = deterministic_assignments(warmup_h, state["y"], support, k)
            if not torch.equal(assign.cpu(), expected_assignments.cpu()):
                raise RuntimeError("stored prototype membership differs from deterministic warmup clustering")
            if not np.array_equal(test_ids, state["test_np"]):
                raise RuntimeError("checkpoint test indices changed")
            if int(latest["next_epoch"]) != int(task["epochs_executed"]):
                raise RuntimeError("latest checkpoint does not represent the task's final committed epoch")
            if latest["trace"] != task["trace"]:
                raise RuntimeError("latest checkpoint RNG trace differs from the completed task record")
            if int(task["best_epoch_1based"]) != int(cp["best_epoch_1based"]):
                raise RuntimeError("selected best epoch differs between task and checkpoint")

            logged_by_epoch = {}
            with log_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    epoch = int(row["epoch"])
                    if epoch in logged_by_epoch:
                        prior = logged_by_epoch[epoch]
                        for key in ("metrics", "base_metrics", "losses", "rng", "best_epoch", "best_acc"):
                            if prior[key] != row[key]:
                                raise RuntimeError(f"replayed epoch log differs for {group} fold={fold} epoch={epoch}")
                    logged_by_epoch[epoch] = row
            expected_epochs = set(range(11, int(task["epochs_executed"]) + 1))
            if set(logged_by_epoch) != expected_epochs:
                raise RuntimeError(f"epoch log coverage is incomplete for {group} fold={fold}")
            selected_acc, selected_epoch = -1.0, -1
            for epoch in sorted(logged_by_epoch):
                acc = float(logged_by_epoch[epoch]["metrics"]["acc"])
                if acc > selected_acc:
                    selected_acc, selected_epoch = acc, epoch
            if selected_epoch != int(task["best_epoch_1based"]) or selected_acc != float(task["metrics"]["acc"]):
                raise RuntimeError(f"strict ACC best-checkpoint selection is inconsistent for {group} fold={fold}")
            R.hist().load_bundle(state, cp["base_state"])
            module = initialize(embeddings, classes, k, fold, state["y"].device)
            module.load_state_dict(cp["module_state"], strict=True)
            module.eval()
            rng_before = R.runtime().rng_sha(R.runtime().pack_rng())
            with torch.no_grad():
                base_out = R.forward_with_cam144(state, training=False)
                ids = torch.as_tensor(test_ids, dtype=torch.long, device=state["y"].device)
                dcp = module(base_out["cam144"], base_out["logp"], state["y"], support, assign, ids)
                final_full = base_out["logp"].clone()
                final_full[ids] = dcp["logp"]
                metrics = R.runtime().classification_metrics(final_full, state["test"], state["y"])
                base_metrics = R.runtime().classification_metrics(base_out["logp"], state["test"], state["y"])
            rng_after = R.runtime().rng_sha(R.runtime().pack_rng())
            if rng_before != rng_after:
                raise RuntimeError("independent checkpoint evaluation consumed shared RNG")
            compact = R._compact_metrics(metrics)
            base_compact = R._compact_metrics(base_metrics)
            if not _metric_close(compact, cp["metrics"]) or not _metric_close(base_compact, cp["base_metrics"]):
                raise RuntimeError(f"independent metric readback mismatch: {group} fold={fold}")
            if not np.array_equal(np.asarray(metrics["prob"]), np.asarray(cp["test_probability"])):
                raise RuntimeError(f"probability readback mismatch: {group} fold={fold}")
            if not np.array_equal(np.asarray(metrics["pred"]), np.asarray(cp["test_prediction"])):
                raise RuntimeError(f"prediction readback mismatch: {group} fold={fold}")
            if task["best_epoch_1based"] != cp["best_epoch_1based"] or task["metrics"] != cp["metrics"]:
                raise RuntimeError("task summary differs from independently read checkpoint")

            reference = base_refs[fold]
            for entry in task["trace"]:
                ep = int(entry["epoch"])
                ref = reference["rng"].get(ep)
                if ref is not None and (entry["rng_before_train"], entry["rng_after_train"], entry["rng_after_eval"]) != (ref["rng_before"], ref["rng_after_train"], ref["rng_after_eval"]):
                    raise RuntimeError(f"public RNG trace mismatch in final audit: {group} fold={fold} epoch={ep}")
            rng_common[(group, fold)] = len(task["trace"])

            values = _metric_pct(metrics)
            fold_rows.append({
                "group": group,
                "fold": fold,
                **values,
                "best_epoch_1based": cp["best_epoch_1based"],
                "epochs_executed": task["epochs_executed"],
                "base_acc": 100 * base_metrics["acc"],
                "base_hard_auc": 100 * base_metrics["hard_auc"],
                "base_prob_auc": 100 * base_metrics["prob_auc"],
                "base_sen": 100 * base_metrics["sen"],
                "base_spe": 100 * base_metrics["spe"],
                "base_asd_f1": 100 * base_metrics["asd_f1"],
                "checkpoint_sha256": checkpoint_sha,
                "source_sha256": expected_source,
                "support_count": len(support),
                "test_count": len(test_ids),
            })
            proto_stats = _prototype_stats(dcp, support, assign.cpu().numpy(), state["y"].detach().cpu().numpy())
            attn_stats = _attention_stats(dcp["attention"])
            y_test = state["y"][torch.as_tensor(test_ids, dtype=torch.long, device=state["y"].device)].detach().cpu().numpy()
            base_pred = base_out["logp"][torch.as_tensor(test_ids, dtype=torch.long, device=state["y"].device)].argmax(-1).detach().cpu().numpy()
            final_pred = dcp["logp"].argmax(-1).detach().cpu().numpy()
            correction_stats = {
                "base_correct": int((base_pred == y_test).sum()),
                "full_correct": int((final_pred == y_test).sum()),
                "corrected": int(((base_pred != y_test) & (final_pred == y_test)).sum()),
                "harmed": int(((base_pred == y_test) & (final_pred != y_test)).sum()),
                "mean_abs_delta": float(dcp["delta"].abs().mean().cpu()),
                "max_abs_delta": float(dcp["delta"].abs().max().cpu()),
                "mean_gamma": float(dcp["gamma"].cpu()),
            }
            for cls in attn_stats:
                mechanism_rows.append({
                    "group": group,
                    "fold": fold,
                    "kind": "prompt_attention",
                    "class": cls["class"],
                    "prompt_count": cls["prompt_count"],
                    "entropy_mean": cls["entropy_mean"],
                    "entropy_max": cls["entropy_max"],
                    "weight_mean": json.dumps(cls["weight_mean"]),
                    "weight_patient_sd": json.dumps(cls["weight_patient_sd"]),
                    "argmax_frequency": json.dumps(cls["argmax_frequency"]),
                })
            mechanism_rows.append({
                "group": group,
                "fold": fold,
                "kind": "prototype_and_correction",
                "support_counts_by_class_cluster": json.dumps(proto_stats["support_counts_by_class_cluster"]),
                "prototype_counts_min": proto_stats["prototype_counts_min"],
                "prototype_between_class_mean_cosine_distance": proto_stats["prototype_between_class_mean_cosine_distance"],
                "prototype_within_class_mean_cosine_distance": json.dumps(proto_stats["prototype_within_class_mean_cosine_distance"]),
                "prototype_collapse_fraction_cosine_gt_0_98": json.dumps(proto_stats["prototype_collapse_fraction_cosine_gt_0_98"]),
                "prototype_match_entropy_by_class": json.dumps(proto_stats["prototype_match_entropy_by_class"]),
                "prototype_centers": json.dumps(proto_stats["prototype_centers"]),
                "mean_abs_delta": correction_stats["mean_abs_delta"],
                "max_abs_delta": correction_stats["max_abs_delta"],
                "mean_gamma": correction_stats["mean_gamma"],
                "base_correct": correction_stats["base_correct"],
                "full_correct": correction_stats["full_correct"],
                "corrected": correction_stats["corrected"],
                "harmed": correction_stats["harmed"],
            })
            if group == R.GROUPS[1]:
                anon_text, anon_classes = R.text_payload(R.GROUPS[4])
                _, _, intervention, _ = _intervention(
                    module, base_out["cam144"], base_out["logp"], state["y"], support, assign,
                    test_ids, metrics, base_metrics, anon_text, torch.as_tensor(anon_classes), group, fold,
                )
                intervention_rows.extend(intervention)

            with log_path.open(encoding="utf-8") as stream:
                first_record = json.loads(next(iter(stream)))
            gradients = first_record["mechanism"]["gradients"]
            if not gradients or any(float(v) <= 0 or not math.isfinite(float(v)) for v in gradients.values()):
                raise RuntimeError(f"training-time DARG/CAM/module gradient record failed: {group} fold={fold}")
            audit_rows.append({
                "group": group,
                "fold": fold,
                "checkpoint_sha256": checkpoint_sha,
                "source_sha256": expected_source,
                "base_identity": identity["model_cam_sha"],
                "base_optimizer_identity": identity["optimizer_sha"],
                "initial_rng_sha256": cp["initial"]["public_rng_sha256"],
                "test_labels_excluded_from_new_branch": True,
                "prototype_support_exact_legal_train": True,
                "leave_one_out_enabled": True,
                "checkpoint_readback_metrics_exact": True,
                "public_rng_trace_matches_history": True,
                "rng_reference_epoch_matches": sum(1 for row in task["trace"] if int(row["epoch"]) in reference["rng"]),
                "rng_epochs_without_historical_reference": sum(1 for row in task["trace"] if int(row["epoch"]) not in reference["rng"]),
                "strict_acc_checkpoint_selection_verified": True,
                "deterministic_prototype_membership_verified": True,
                "base_class": type(state["cam"]).__name__,
                "task_status": "PASS",
            })
            del state, module, base_out, dcp, final_full
            torch.cuda.empty_cache()

    base_csv = R.BASE / "results/fold_metrics.csv"
    base_rows_all = list(csv.DictReader(base_csv.open(encoding="utf-8")))
    base_rows = [row for row in base_rows_all if row.get("group") == R.BASE_GROUP]
    if len(base_rows) != 10:
        raise RuntimeError("historical E74.4-G2 fold metrics are not ten-fold complete")
    base_rows_pct = []
    for row in base_rows:
        converted = {"fold": int(row["fold"])}
        for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            converted[key] = float(row[key]) * (100 if float(row[key]) <= 1.0 else 1.0)
        base_rows_pct.append(converted)
    # Comparisons are reported in percentage points. Current fold_rows are already percent.
    comparison_rows, comparison_summary = _paired_rows(fold_rows, base_rows_pct)
    _write_csv(R.RESULTS / "fold_metrics.csv", fold_rows)
    _write_csv(R.RESULTS / "mechanism.csv", mechanism_rows)
    _write_csv(R.RESULTS / "interventions.csv", intervention_rows)
    _write_csv(R.RESULTS / "paired_comparisons.csv", comparison_rows)
    _write_csv(R.RESULTS / "paired_summary.csv", comparison_summary)
    audit_status = {
        "status": "PASS",
        "audited_checkpoints": len(audit_rows),
        "expected_checkpoints": 50,
        "base_disabled_parity_folds": 10,
        "active_end_to_end_gradient": preflight["active_gradient_test"]["gradient_norms"],
        "two_step_recovery_replay": preflight["two_step_recovery_replay"],
        "public_rng_trace_entries": sum(rng_common.values()),
        "historical_rng_reference_scope": "Every matching E74.4 epoch digest is checked. Epochs beyond the historical run's recorded trace are reported as unreferenced and are not claimed to have epoch-by-epoch historical trajectory evidence.",
        "protected_base_core_sha256": R.CFG["base_sha256"],
        "checkpoint_audits": audit_rows,
    }
    R.save_json(R.RESULTS / "audit_status.json", audit_status)
    _write_report(fold_rows, base_rows_pct, comparison_summary, audit_rows, mechanism_rows, intervention_rows, preflight)
    R.save_json(R.HERE / "manifest.json", {
        "experiment": R.CFG["experiment"],
        "status": "COMPLETE_AUDITED" if len(audit_rows) == 50 else "INCOMPLETE",
        "source_sha256": expected_source,
        "frozen_sources": R.source_map(),
        "formal_tasks": 50,
        "completed_tasks": len(audit_rows),
        "independent_audits": len(audit_rows),
        "preflight_sha256": R.sha_file(preflight_path),
        "audit_status_sha256": R.sha_file(R.RESULTS / "audit_status.json"),
        "report_sha256": R.sha_file(R.RESULTS / "report.md"),
        "checkpoint_audit": "PASS",
        "oss_fallback": R.CFG["fallback_reason"],
    })
    return audit_status


def _write_report(fold_rows, base_rows, comparisons, audits, mechanism, interventions, preflight):
    metrics = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
    labels = {
        "acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Probability-AUC",
        "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1",
    }
    summaries = {}
    for group in R.GROUPS:
        rows = [row for row in fold_rows if row["group"] == group]
        summaries[group] = {}
        for metric in metrics:
            summaries[group][metric] = _mean_sd([row[metric] for row in rows])
    base_summary = {metric: _mean_sd([row[metric] for row in base_rows]) for metric in metrics}
    lines = [
        "# Experiment 78：DCPSS 正式结果",
        "",
        f"状态：**{'COMPLETE_AUDITED' if len(audits) == 50 else 'INCOMPLETE'}**；独立读回 {len(audits)}/50。",
        "",
        "## 结论摘要",
        "",
        "DCPSS 在 DARG-CAM 的 144 维 CAM 输出上动态聚合疾病原型，并用患者条件化的原型对比选择冻结 CLIP RN50 疾病 Prompt。训练查询按 ID 从其所属簇原型中执行 Leave-One-Out；主分类器输出由同一个原始分类头提供，语义证据以有界有符号 margin 修正加入。未使用 E77 系列 FP/FN 队列或配对检索器。",
        "",
        "E78 的技术有效性以实际五组独立训练和同 checkpoint 机制干预判断，不以相似度或注意力可视化单独立论。固定选模仍使用测试折 ACC，历史 CV 前预处理与 full-graph transductive 传播保留，Fold 2 仍有两名历史 train/test 重叠患者；结果不构成无偏泛化估计。",
        "",
        "## 文献与源码审计",
        "",
        "已阅读本地 SPromptGL MICCAI 2025 论文 11 页全文（含 Fig.1–3、Section 2.1–2.2、Eq.(1)–(7)、实验设置和消融），BrainPrompt MICCAI 2025 论文 11 页全文及官方源文本文件；SPromptGL 官方代码、BrainPrompt 官方代码和本项目 GitHub 只读核验。HP²L 只能读到 ScienceDirect 公开摘要、Highlights 与方法预览（下载全文请求返回 403），未声称精读全文。详细机制和证据边界见 `results/source_review.md`。",
        "",
        "实际 E74.4 基座为 DARG + Positive-DCR + CAM + `PLNNoPEN`，原始分类器前 CAM hidden 为 `[871,144]`，OutputLayer 返回 log-softmax；没有活动 PEN。原训练损失为带权 NLL 分类、原图损失、0.05 pair、0.01 adversarial、0.01 relation-preserve，原优化器顺序 `opt_base → opt_cam → opt_added`。DCPSS 只替换训练分类预测项并追加固定 0.05 semantic、0.02 gain 辅助项，DARG 与 CAM 接收新增分类梯度。",
        "",
        "BrainPrompt 文本按官方文件 TC→ASD，映射 `[1,0]`；SPromptGL 原始8条按第一组TD、第二组ASD，映射 `[1,1,1,1,0,0,0,0]`。BrainPrompt 组只迁移原疾病级文本并统一使用冻结 RN50，不是完整 BrainPrompt 复现。",
        "",
        "## 方法与固定训练",
        "",
        "支持集为 `train_indices - explicit_test_indices`。K=3 子簇仅按每折 E61 epoch-10 warmup 的 CAM144、在类内按 PC1 稳定排序后切为3个平衡区间；成员固定，当前 epoch 原型使用当前 CAM 投影的可微均值。K=1 每类一个全局中心。原型数量不引入任何额外可训练分类层。每组新增参数均为 37,377（患者投影 4,608、文本投影 32,768、残差幅度1）。",
        "",
        "对每个训练 Query，只从它自己的类别/簇总和中减去该患者当前投影和计数；别的簇不变。测试 Query 不属于 Support。新分支训练只使用合法外层训练患者；Fold 2 的 1-based 行377/407仍保留在原基座历史训练和指标流程中，但排除于所有新增原型和辅助监督。",
        "",
        "每条 Prompt 的分数为患者-文本 cosine 加上该 Prompt 类别原型与对立类别原型的 cosine 差；类内 softmax 得到患者条件 Prompt 权重。两个疾病证据形成有符号 `tanh(e_ASD-e_TD)`，残差幅度 `0.5*tanh(alpha)` 初始化为0.1，类别 margin 最大绝对调整0.5，按 `[+delta/2,-delta/2]` 加到 base log-probability 后重新 log-softmax。损失是原 E74.4 非分类项 + 替换后的分类项 + 0.05加权语义CE + 0.02 `softplus((CE_final-stopgrad(CE_base))/0.25)`。",
        "",
        "固定训练：seed=0，原始10折，epoch 10 warmup 后训练至450，`wait>200`停止，测试 ACC 严格提升选择最佳 checkpoint。CLIP RN50 在训练前一次缓存，冻结且保留1024维输出。没有 OOF 重训或参数搜索。",
        "",
        "## 正式结果",
        "",
        "以下 mean±SD 为百分比，SD 是10折总体标准差。Hard-AUC 按历史硬标签 one-hot 实现，二分类下等于 BA；Probability-AUC 使用 P(ASD=0) 的连续分数。",
        "",
        "| 组别 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.append("| E74.4-G2 history | " + " | ".join(_fmt(*base_summary[m]) for m in metrics) + " |")
    for group in R.GROUPS:
        lines.append("| " + group + " | " + " | ".join(_fmt(*summaries[group][m]) for m in metrics) + " |")
    lines.extend(["", "### 相对历史基座的差值", "", "单位为百分点；每一项均是逐折配对差值的均值。", "", "| 组别 | ΔACC | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | ΔASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for group in R.GROUPS:
        items = [next(row for row in comparisons if row["comparison"] == f"{group}-E74.4-G2" and row["metric"] == m) for m in metrics]
        lines.append("| " + group + " | " + " | ".join(f"{v['mean_delta_pp']:+.4f}" for v in items) + " |")
    lines.extend(["", "### 组间预设比较", "", "| 比较 | 指标 | 均值差 (pp) | 排除 Fold 2 (pp) | W/T/L |", "|---|---|---:|---:|---:|"])
    for name in ("G2-G1", "G4-G3", "G1-G3", "G2-G4", "G2-G5"):
        for metric in ("acc", "hard_auc", "prob_auc"):
            row = next(v for v in comparisons if v["comparison"] == name and v["metric"] == metric)
            lines.append(f"| {name} | {labels[metric]} | {row['mean_delta_pp']:+.4f} | {row['fold2_excluded_mean_delta_pp']:+.4f} | {row['W']}/{row['T']}/{row['L']} |")
    lines.extend(["", "完整50行逐折六指标见 `fold_metrics.csv`；每个比较的逐折差值见 `paired_comparisons.csv`。", "", "## 机制与审计", ""])
    if mechanism:
        g2rows = [row for row in mechanism if row.get("group") == R.GROUPS[1] and row.get("kind") == "prototype_and_correction"]
        corrected = sum(int(row.get("corrected", 0)) for row in g2rows)
        harmed = sum(int(row.get("harmed", 0)) for row in g2rows)
        mean_delta = statistics.mean(float(row["mean_abs_delta"]) for row in g2rows) if g2rows else float("nan")
        lines.append(f"G2 最佳 checkpoint 同态基座头到最终头共纠正 {corrected} 例、损害 {harmed} 例；跨折平均 |delta|={mean_delta:.6f}。该计数按患者-折记录，不等同独立个体数。")
        g2attn = [row for row in mechanism if row.get("group") == R.GROUPS[1] and row.get("kind") == "prompt_attention"]
        entropy = statistics.mean(float(row["entropy_mean"]) for row in g2attn) if g2attn else float("nan")
        lines.append(f"G2 测试 Query 类内 Prompt 熵均值={entropy:.6f}；SPromptGL 每类4条 Prompt 的最大熵 ln(4)={math.log(4):.6f}。需结合权重跨患者标准差判断是否接近均匀，数值见 `mechanism.csv`。")
    lines.extend([
        "",
        "十折基座关闭 parity 检查前向、原损失、参数和三个原优化器状态、RNG 均逐折精确相等；正式模块初始化使用私有 `fork_rng`，新增前向无 dropout/抽样，逐 epoch 公共 RNG digest 与 E74.4 对照轨迹核对。端到端活动梯度预检记录DARG、CAM、患者投影、Prompt投影及有界残差参数；新增总损失对DARG/CAM非零。训练和恢复均按原三个优化器顺序后更新新增AdamW。恢复重放两步的模型、优化器、随机状态一致。",
        "",
        "独立审计重新加载50/50最佳 checkpoint，重算预测标签、连续概率、ACC、Hard-AUC、Prob-AUC、SEN、SPE、ASD-F1，与记录逐值一致。审计同时核对训练/测试成员、原型簇归属、checkpoint SHA、训练 RNG trace、源文件 SHA 与 best/latest 参数及优化器状态。Fold-2 排除的九折描述性比较也列入 `paired_summary.csv`，不是重新划分或无偏估计。",
        "",
        "G2 同一 checkpoint 的关闭原型、关闭语义、类别交换、匿名文本替换、关闭残差干预结果在 `interventions.csv`。这些固定模型干预解释即时依赖，不替代五组独立重训比较。",
        "",
        "## GO / NO-GO",
        "",
    ])
    base_acc = base_summary["acc"][0]
    base_hard = base_summary["hard_auc"][0]
    base_prob = base_summary["prob_auc"][0]
    g2 = summaries[R.GROUPS[1]]
    no_go = not (g2["acc"][0] > base_acc and g2["hard_auc"][0] > base_hard and g2["prob_auc"][0] > base_prob)
    if no_go:
        lines.append("**NO-GO**：完整医学多子原型组 G2 未同时超过历史 E74.4-G2 的 ACC、Hard-AUC/BA 与 Probability-AUC，不能将当前 DCPSS 作为已成功的第二创新模块。停止自动追加 E78.1、Prompt/温度/原型数搜索。五组仍回答本次预设的粒度和文本来源对照，但任何一项单指标提升不足以改变结论。")
    else:
        lines.append("**GO 候选**：完整 G2 同时超过历史 E74.4-G2 的 ACC、Hard-AUC/BA 与 Probability-AUC。该结论仅是历史单 seed、固定且包含测试选模的协议内候选，不证明无偏临床泛化。")
    lines.extend([
        "",
        "局限：协议一致不等于无偏泛化。外层测试参与 checkpoint 选择，预处理/RFE 先于外层CV，训练采用transductive全图；Fold 2 还保留两名历史重叠患者。单 seed 十折不足以支持临床泛化或发表级结论。",
        "",
        f"输出目录：`{R.HERE}`；本次因首选 OSSFS 路径真实 EIO 而仅使用 `/model` 回退。最终交接已追加到 `/model/spromptgl/experiment/handoff.md`。",
        "",
    ])
    path = R.RESULTS / "report.md"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
