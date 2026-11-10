#!/usr/bin/env python3
"""Independent checkpoint, prediction, RNG, and mechanism audit for E76.5."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as runner

CORE = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
ALL_METRICS = CORE + ("nll", "brier", "ece")
RNG_KEYS = ("rng_before", "rng_after_train", "rng_after_eval")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    runner.atomic_json(Path(path), json_safe(value))


def tree_equal(left, right):
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "item") and hasattr(right, "item"):
        try:
            return left.item() == right.item()
        except (ValueError, TypeError):
            pass
    return left == right


def tree_differences(left, right, path=""):
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return [(path, f"shape/dtype {tuple(left.shape)}/{left.dtype} != {tuple(right.shape)}/{right.dtype}")]
        if torch.equal(left.cpu(), right.cpu()):
            return []
        a, b = left.detach().cpu(), right.detach().cpu()
        return [(path, f"max_abs={float((a.float()-b.float()).abs().max())}, unequal={(a != b).sum().item()}")]
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        if left.keys() != right.keys():
            return [(path, f"keys {left.keys()} != {right.keys()}")]
        for key in left:
            differences.extend(tree_differences(left[key], right[key], f"{path}.{key}" if path else str(key)))
            if len(differences) >= 30:
                break
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [(path, f"length {len(left)} != {len(right)}")]
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(tree_differences(a, b, f"{path}[{index}]"))
            if len(differences) >= 30:
                break
        return differences
    if hasattr(left, "item") and hasattr(right, "item"):
        try:
            left, right = left.item(), right.item()
        except (ValueError, TypeError):
            pass
    return [] if left == right else [(path, f"{left!r} != {right!r}")]


def score_metrics(target, probability):
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                                 f1_score, log_loss, recall_score, roc_auc_score)
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(target), 2) or not np.isfinite(probability).all():
        raise RuntimeError("invalid binary probability matrix")
    if not np.allclose(probability.sum(1), 1.0, atol=2e-6, rtol=0):
        raise RuntimeError("probability rows do not sum to one")
    normalized_probability = probability / probability.sum(1, keepdims=True)
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == target
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & ((confidence < high) if index < 14 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"acc": float(accuracy_score(target, prediction)),
            "hard_auc": float(balanced_accuracy_score(target, prediction)),
            "prob_auc": float(roc_auc_score(target == 0, probability[:, 0])),
            "sen": float(recall_score(target, prediction, pos_label=0)),
            "spe": float(recall_score(target, prediction, pos_label=1)),
            "asd_f1": float(f1_score(target, prediction, pos_label=0)),
            "nll": float(log_loss(target, normalized_probability, labels=[0, 1])),
            "brier": float(brier_score_loss(target == 0, probability[:, 0])),
            "ece": float(ece)}


def summary_rows(fold_rows):
    import numpy as np
    groups = [runner.BASE_GROUP, *runner.GROUPS, "G1-SCPR-Residual-Off"]
    result = []
    for group in groups:
        rows = [row for row in fold_rows if row["group"] == group]
        if len(rows) != 10:
            raise RuntimeError(f"summary expected ten folds for {group}; got {len(rows)}")
        result.append({"group": group, **{f"{metric}_mean": float(np.mean([r[metric] for r in rows]))
                                           for metric in ALL_METRICS},
                       **{f"{metric}_sd_pop": float(np.std([r[metric] for r in rows], ddof=0))
                          for metric in ALL_METRICS}})
    return result


def paired_rows(name, left, right, metrics=CORE):
    import numpy as np
    from scipy.stats import t
    rows = []
    for scope, folds in (("all10", list(range(1, 11))),
                         ("exclude_fold2", [f for f in range(1, 11) if f != 2])):
        for metric in metrics:
            differences = np.asarray([(left[f][metric] - right[f][metric]) * 100 for f in folds])
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1))
            half = float(t.ppf(.975, len(differences) - 1) * sd / math.sqrt(len(differences)))
            rows.append({"comparison": name, "scope": scope, "metric": metric, "folds": len(folds),
                         "mean_difference_pp": mean, "wins": int((differences > 1e-12).sum()),
                         "ties": int((np.abs(differences) <= 1e-12).sum()),
                         "losses": int((differences < -1e-12).sum()),
                         "descriptive_95ci_low_pp": mean - half,
                         "descriptive_95ci_high_pp": mean + half,
                         "fold_differences_pp": json.dumps({str(f): float((left[f][metric] - right[f][metric]) * 100)
                                                            for f in folds}, sort_keys=True)})
    return rows


def role_metadata(memory, support_ids, sites):
    import numpy as np
    import torch
    ids = torch.as_tensor(sorted(map(int, support_ids)), dtype=torch.long)
    labels = memory["labels"].detach().cpu()[ids]
    probabilities = memory["probabilities"].detach().cpu()[ids]
    prediction = probabilities.argmax(-1)
    site_ids = sites.detach().cpu()[ids]
    result = {}
    for cls, label in ((0, "asd"), (1, "td")):
        class_mask = labels == cls
        correct = class_mask & (prediction == cls)
        hard = class_mask & (prediction != cls)
        result[label] = {"reliable_n": int(correct.sum()),
                         "boundary_n": int(class_mask.sum()),
                         "hard_n": int(hard.sum()),
                         "hard_enabled": bool(int(hard.sum()) >= 3),
                         "candidate_sites": int(torch.unique(site_ids[class_mask]).numel()),
                         "candidate_site_counts": json.dumps({str(int(site)): int(count)
                              for site, count in zip(*torch.unique(site_ids[class_mask], return_counts=True))})}
    return result


def retrieval_site_diagnostics(query_features, query_ids, memory, support_ids, sites):
    """Reproduce the implemented top-K site-diverse selector for audit summaries."""
    import torch
    import torch.nn.functional as F
    feats = memory["features"].detach().cpu().float()
    probs = memory["probabilities"].detach().cpu().float()
    labels = memory["labels"].detach().cpu().long()
    site_ids = sites.detach().cpu().long()
    supports = torch.as_tensor(sorted(map(int, support_ids)), dtype=torch.long)
    queries = F.normalize(query_features.detach().cpu().float().reshape(-1, 4, 36), dim=-1)
    totals = {"retrieval_calls": 0, "selected_supports": 0, "same_site_supports": 0,
              "fallback_supports_over_cap": 0, "unique_site_fraction_sum": 0.0,
              "effective_support_n_sum": 0.0}
    for cls in (0, 1):
        class_ids = supports[labels[supports] == cls]
        pred = probs[class_ids].argmax(-1)
        py = probs[class_ids, cls]
        role_specs = [("reliable", class_ids[pred == cls],
                       ((probs[class_ids[pred == cls], cls] - .5) / .5).clamp(0, 1), 8),
                      ("boundary", class_ids, 4 * py * (1 - py), 8)]
        hard_ids = class_ids[pred != cls]
        if len(hard_ids) >= 3:
            role_specs.append(("hard", hard_ids, probs[hard_ids].max(-1).values, 3))
        for _role, ids, role_weights, k in role_specs:
            if ids.numel() == 0:
                continue
            candidate_features = feats[ids]
            for query_index, query_id in enumerate(query_ids):
                query_site = site_ids[int(query_id)]
                for slot in range(4):
                    similarities = F.normalize(candidate_features[:, slot], dim=-1) @ queries[query_index, slot]
                    count = min(k, len(ids))
                    diverse = torch.zeros(len(ids), dtype=torch.bool)
                    for site in torch.unique(site_ids[ids]).tolist():
                        positions = torch.nonzero(site_ids[ids] == int(site), as_tuple=True)[0]
                        local_count = min(2, len(positions))
                        local = torch.topk(similarities[positions], local_count).indices
                        diverse[positions[local]] = True
                    priority = similarities + diverse.to(similarities.dtype) * 3.0
                    selected = torch.topk(priority, count).indices
                    chosen_ids = ids[selected]
                    chosen_sites = site_ids[chosen_ids]
                    per_site = torch.unique(chosen_sites, return_counts=True)[1]
                    totals["retrieval_calls"] += 1
                    totals["selected_supports"] += int(count)
                    totals["same_site_supports"] += int((chosen_sites == query_site).sum())
                    totals["fallback_supports_over_cap"] += int((per_site - 2).clamp_min(0).sum())
                    totals["unique_site_fraction_sum"] += float(torch.unique(chosen_sites).numel() / count)
                    logits = similarities[selected] / .20 + role_weights[selected].clamp_min(1e-4).log()
                    pool_weights = torch.softmax(logits, dim=0)
                    totals["effective_support_n_sum"] += float(torch.exp(-(pool_weights * pool_weights.clamp_min(1e-12).log()).sum()))
    calls = max(1, totals["retrieval_calls"])
    selected = max(1, totals["selected_supports"])
    return {"retrieval_calls": totals["retrieval_calls"],
            "same_site_support_fraction": totals["same_site_supports"] / selected,
            "cross_site_support_fraction": 1 - totals["same_site_supports"] / selected,
            "fallback_over_site_cap_fraction": totals["fallback_supports_over_cap"] / selected,
            "mean_unique_site_fraction_per_retrieval": totals["unique_site_fraction_sum"] / calls,
            "mean_effective_support_n": totals["effective_support_n_sum"] / calls}


def run_restore_replay(group, fold):
    """Run the next update twice from a saved full state and require exact equality."""
    import copy
    import numpy as np
    import torch
    from scpr import init_module_isolated
    task_path, _, latest_path, _ = runner.paths(group, fold)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if latest.get("memory") is None or latest.get("next_epoch", 450) >= 450:
        return {"status": "SKIPPED_NO_NEXT_TRAIN_EPOCH", "group": group, "fold": fold}
    base, hist = runner.load_base(), runner.load_base().load_history()
    vectors = runner.text_vectors()

    def step_once():
        state, _, identity = runner.prepare_base(fold)
        state["fold"] = fold
        if identity["warmup_sha256"] != latest["warmup_sha256"]:
            raise RuntimeError("restore replay warmup identity mismatch")
        runtime = hist.load_e69().load_e61().load_base()
        hist.load_bundle(state, copy.deepcopy(latest["base_state"]))
        runtime.restore_rng(latest["rng"])
        module = init_module_isolated(vectors, anonymous=(group == "G2-SCPR-Anonymous"), seed=765101).to(state["y"].device)
        module.load_state_dict(latest["scpr"])
        optimizer = torch.optim.AdamW(module.parameters(), lr=.0005, weight_decay=.01)
        optimizer.load_state_dict(copy.deepcopy(latest["new_optimizer"]))
        memory = {k: (v.to(state["y"].device) if torch.is_tensor(v) else v)
                  for k, v in latest["memory"].items()}
        train_ids = list(map(int, state["train_np"]))
        test_ids = list(map(int, state["test_np"]))
        test_set = set(test_ids)
        scpr_train_ids = [i for i in train_ids if i not in test_set]
        ids_a, ids_b = runner.split_ab(scpr_train_ids, state["y"].detach().cpu().numpy())
        site_tensor = runner.read_sites().to(state["y"].device)
        labels = state["y"].long()
        hist.zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        out, cam = runner.hook_forward(state, training=True)
        train_index = torch.as_tensor(scpr_train_ids, dtype=torch.long, device=labels.device)
        margin = out["logp"][train_index, 0] - out["logp"][train_index, 1]
        prediction = out["logp"][train_index].argmax(-1)
        confidence = out["logp"][train_index].exp().gather(1, labels[train_index, None]).squeeze(1)
        keep = (prediction == labels[train_index]) & (confidence >= .8)
        aset = set(ids_a)
        support_a = torch.as_tensor(ids_a, dtype=torch.long, device=labels.device)
        support_b = torch.as_tensor(ids_b, dtype=torch.long, device=labels.device)
        query_support = [support_b if patient in aset else support_a for patient in scpr_train_ids]
        score, diag = module(cam[train_index], scpr_train_ids, memory, query_support, site_tensor,
                             margin, labels=labels[train_index], keep_mask=keep)
        final_logp = runner.apply_score(out["logp"], scpr_train_ids, score - margin, module)
        cls = hist.load_e69().load_e61().load_base().runtime()["utils"].ClsLoss(
            final_logp, labels, state["train"], state["weights"])
        total = out["total"] - out["cls"] + cls + .02 * diag["proto_loss"] + .02 * diag["res_loss"] + .01 * diag["keep_loss"]
        total.backward()
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_train = runtime.rng_sha(runtime.pack_rng())
        train_logp = out["logp"].detach().cpu().clone()
        train_loss = float(total.detach().cpu())
        with torch.no_grad():
            evaluated, eval_cam = runner.hook_forward(state, training=False)
            eval_memory = runner.memory_snapshot(memory)
            test_index = torch.as_tensor(test_ids, dtype=torch.long, device=labels.device)
            test_margin = evaluated["logp"][test_index, 0] - evaluated["logp"][test_index, 1]
            support = torch.as_tensor(scpr_train_ids, dtype=torch.long, device=labels.device)
            eval_score, _ = module(eval_cam[test_index], test_ids, memory, support, site_tensor, test_margin)
            final_eval = runner.apply_score(evaluated["logp"], test_ids, eval_score - test_margin, module)
            probability = final_eval[test_index].exp().detach().cpu().clone()
            memory = runner.make_memory(871, labels.device, scpr_train_ids, labels, eval_cam,
                                       evaluated["logp"], memory)
        rng_eval = runtime.rng_sha(runtime.pack_rng())
        return {"train_loss": train_loss, "train_logp": train_logp, "probability": probability,
                "state": hist.state_bundle(state), "module": copy.deepcopy(module.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()), "memory": runner.memory_snapshot(memory),
                "eval_memory": eval_memory, "rng_train": rng_train, "rng_eval": rng_eval}

    first, second = step_once(), step_once()
    if not tree_equal(first, second):
        raise RuntimeError(f"restored next-step replay differs: {group} fold {fold}; {tree_differences(first, second)[:30]}")
    return {"status": "PASS", "group": group, "fold": fold,
            "next_epoch": int(latest["next_epoch"]) + 1,
            "rng_after_train": first["rng_train"], "rng_after_eval": first["rng_eval"],
            "output_loss": first["train_loss"], "output_probability_sha256":
                hashlib.sha256(first["probability"].numpy().tobytes()).hexdigest()}


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F

    started = time.time()
    results_dir = EXP / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_json(results_dir / "preflight.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight did not pass")
    audit_path = str(EXP / "audit.py")
    config_path = str(EXP / "config.json")
    for path, digest in preflight["source_hashes"].items():
        if path == audit_path:
            continue
        if path == config_path:
            current_config_bytes = Path(path).read_bytes()
            training_config_bytes = current_config_bytes.replace(
                b'"status": "COMPLETE_AUDITED"',
                b'"status": "IMPLEMENTATION_PENDING_PREFLIGHT"', 1)
            if (current_config_bytes == training_config_bytes or
                    hashlib.sha256(training_config_bytes).hexdigest() != digest):
                raise RuntimeError("final config differs from its training-time SHA beyond the status field")
            continue
        if runner.sha_file(Path(path)) != digest:
            raise RuntimeError(f"source changed after training/preflight: {path}")
    base_manifest = read_json(runner.BASE_EXP / "manifest.json")
    base_audit_status = read_json(runner.BASE_EXP / "results/audit_status.json")
    if base_manifest.get("status") != "COMPLETE_AUDITED" or base_audit_status.get("status") != "PASS":
        raise RuntimeError("E74.4-G2 historical base is not independently audited")

    hist = runner.load_base().load_history()
    pair_runner = hist.load_e69().load_e61()
    scpr_code = runner.load_scpr()
    sites = runner.read_sites()
    prompts = tuple(scpr_code.PROMPTS)
    if prompts != tuple(json.loads((EXP / "config.json").read_text())["scpr"]["prompts_asd_td"]):
        raise RuntimeError("official BrainPrompt strings/order changed")
    text_vectors = runner.text_vectors().float()
    if text_vectors.shape[0] != 2 or not torch.isfinite(text_vectors).all():
        raise RuntimeError("invalid ASD=0, TD=1 frozen prompt embeddings")
    if preflight.get("prompt_vectors_sha256") != hashlib.sha256(text_vectors.cpu().contiguous().numpy().tobytes()).hexdigest():
        raise RuntimeError("frozen text vector cache differs from the preflight lock")

    baseline, by_group, off_scores = {}, {g: {} for g in runner.GROUPS}, {}
    audit_rows, fold_rows, mechanism_rows, correction_rows, rng_totals = [], [], [], [], {}
    parameter_shapes, parameter_counts, prompt_grams = {}, {}, {}
    for fold in range(1, 11):
        base_task_path = runner.BASE_EXP / f".runtime/tasks/{runner.BASE_GROUP}_f{fold:02d}.json"
        base_ckpt_path = runner.BASE_EXP / f".runtime/checkpoints/{runner.BASE_GROUP}_f{fold:02d}.best.pt"
        base_task = read_json(base_task_path)
        base_sha = runner.sha_file(base_ckpt_path)
        if base_task.get("status") != "COMPLETE" or base_sha != base_task.get("checkpoint_sha256"):
            raise RuntimeError(f"historical E74.4 task/checkpoint identity failed at fold {fold}")
        base_payload = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        if base_payload.get("group") != runner.BASE_GROUP or int(base_payload.get("fold", -1)) != fold:
            raise RuntimeError(f"historical E74.4 checkpoint metadata mismatch fold {fold}")

        state, warmup, identity = runner.prepare_base(fold)
        state["fold"] = fold
        if identity != base_task["identity"] or identity != base_payload["identity"]:
            raise RuntimeError(f"E61 warmup/model/optimizer/split identity mismatch fold {fold}")
        warm_path = runner.E61 / f".runtime/warmup/fold{fold:02d}.pt"
        warm_sha = runner.sha_file(warm_path)
        if warm_sha != identity["warmup_sha256"]:
            raise RuntimeError(f"E61 warmup file hash mismatch fold {fold}")
        warm_task = read_json(runner.E61 / f".runtime/tasks/fold{fold:02d}.warmup.json")
        if warm_sha != warm_task.get("warmup_sha256"):
            raise RuntimeError(f"E61 warmup task/checkpoint checksum mismatch fold {fold}")
        if len(warmup["train_pairs"]) != 4096 or warmup.get("pair_sha") != warm_task.get("pair_sha"):
            raise RuntimeError(f"E61 warmup relation-pair identity/count mismatch fold {fold}")
        if (state["train_pairs_np"].shape != (4096, 2) or
                pair_runner.pair_sha(state["train_pairs_np"], state["train_same_np"]) != identity.get("pair_sha")):
            raise RuntimeError(f"E74.4 prepared 4096 relation-pair identity mismatch fold {fold}")
        base_train_ids = set(map(int, state["train_np"]))
        test_ids = list(map(int, state["test_np"]))
        allowed = base_train_ids - set(test_ids)
        targets = state["y"].detach().cpu().long().numpy()
        test_target = targets[np.asarray(test_ids, dtype=int)]
        overlap = sorted((base_train_ids & set(test_ids)))

        hist.load_bundle(state, base_payload["best_state"])
        state["model"].eval(); state["cam"].eval()
        with torch.no_grad():
            base_out, base_cam = runner.hook_forward(state, training=False)
        base_index = torch.as_tensor(test_ids, dtype=torch.long, device=base_out["logp"].device)
        base_probability = base_out["logp"][base_index].exp().detach().cpu().numpy()
        stored_base_probability = np.asarray(base_payload["test_probability"])
        if not np.allclose(base_probability, stored_base_probability, atol=1e-8, rtol=0):
            raise RuntimeError(f"E74.4 historical probability replay mismatch fold {fold}")
        base_prediction = base_probability.argmax(1)
        base_metrics = score_metrics(test_target, base_probability)
        for metric in CORE:
            if abs(base_metrics[metric] - float(base_payload["metrics"][metric])) > 2e-8:
                raise RuntimeError(f"E74.4 historical metric replay mismatch fold {fold} {metric}")
        baseline[fold] = {"metrics": base_metrics, "probability": base_probability,
                          "prediction": base_prediction, "target": test_target,
                          "test_ids": np.asarray(test_ids), "identity": identity,
                          "overlap": overlap, "checkpoint_sha256": base_sha,
                          "best_epoch": int(base_payload["best_epoch_1based"])}
        fold_rows.append({"group": runner.BASE_GROUP, "fold": fold, **base_metrics,
                          "best_epoch_1based": int(base_payload["best_epoch_1based"]),
                          "epochs_executed": int(base_payload["epochs_executed"]),
                          "checkpoint_sha256": base_sha})
        audit_rows.append({"group": runner.BASE_GROUP, "fold": fold, "status": "PASS",
                           "checkpoint_sha256": base_sha, "warmup_sha256": warm_sha,
                           "pair_sha": identity["pair_sha"], "test_overlap_0based": json.dumps(overlap),
                           "metrics_recomputed": True, "prediction_reproduction_atol": 1e-8})

        for group in runner.GROUPS:
            task_path, ckpt_path, latest_path, _ = runner.paths(group, fold)
            task = read_json(task_path)
            digest = runner.sha_file(ckpt_path)
            if task.get("status") != "COMPLETE" or digest != task.get("checkpoint_sha256"):
                raise RuntimeError(f"new task/checkpoint checksum mismatch: {group} fold {fold}")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best = payload["best"]
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            if (payload.get("group") != group or int(payload.get("fold", -1)) != fold or
                    payload.get("base_group") != runner.BASE_GROUP or payload.get("identity") != identity or
                    payload.get("warmup_sha256") != warm_sha or task.get("identity") != identity):
                raise RuntimeError(f"new checkpoint did not inherit the exact E61/E74.4 fold identity: {group} {fold}")
            if task.get("best_epoch_1based") != best.get("epoch") or task.get("epochs_executed") != payload.get("epochs_executed"):
                raise RuntimeError(f"task/best/latest progress mismatch: {group} fold {fold}")
            required_best = {"base_state", "scpr", "new_optimizer", "memory", "support_a", "support_b",
                             "scpr_train_ids", "metrics", "probability", "prediction", "epoch"}
            if not required_best.issubset(best) or not {"model", "cam", "opt_base", "opt_cam", "opt_added"}.issubset(best["base_state"]):
                raise RuntimeError(f"best checkpoint is missing a model/optimizer state: {group} fold {fold}")
            required_latest = {"base_state", "scpr", "new_optimizer", "memory", "rng", "next_epoch", "wait", "best", "trace"}
            if not required_latest.issubset(latest) or latest.get("group") != group or int(latest.get("fold", -1)) != fold:
                raise RuntimeError(f"latest checkpoint cannot resume complete state: {group} fold {fold}")
            if latest.get("warmup_sha256") != warm_sha or latest.get("next_epoch") != payload["epochs_executed"]:
                raise RuntimeError(f"latest checkpoint progress/warmup mismatch: {group} fold {fold}")
            if latest["memory"] is None or int(best["memory"]["updates"]) < 2:
                raise RuntimeError(f"dynamic EMA memory was absent or not updated: {group} fold {fold}")
            if any(v.requires_grad for v in best["memory"].values() if torch.is_tensor(v)):
                raise RuntimeError(f"autograd tensors leaked into memory: {group} fold {fold}")
            if not np.array_equal(np.asarray(best["scpr_train_ids"]), np.asarray(sorted(allowed))):
                raise RuntimeError(f"SCPR memory/query cohort includes test rows: {group} fold {fold}")
            memory = best["memory"]
            observed = memory["observed"].detach().cpu().numpy().astype(bool)
            memory_labels = memory["labels"].detach().cpu().numpy()
            if not set(np.flatnonzero(observed)).issubset(allowed):
                raise RuntimeError(f"test patient entered prototype memory: {group} fold {fold}")
            outside = np.asarray(sorted(set(range(len(memory_labels))) - allowed), dtype=int)
            if len(outside) and (observed[outside].any() or np.any(memory_labels[outside] != -1)):
                raise RuntimeError(f"test/non-training labels or features entered memory: {group} fold {fold}")
            if np.any(memory_labels[np.asarray(sorted(allowed), dtype=int)] != targets[np.asarray(sorted(allowed), dtype=int)]):
                raise RuntimeError(f"prototype memory labels differ from legal training labels: {group} fold {fold}")
            a, b = runner.split_ab(sorted(allowed), targets)
            if set(a) & set(b) or set(a) | set(b) != allowed:
                raise RuntimeError(f"cross-support A/B split is invalid: {group} fold {fold}")
            if set(best["support_a"]) != set(a) or set(best["support_b"]) != set(b):
                raise RuntimeError(f"saved A/B support identity mismatch: {group} fold {fold}")
            for query in a:
                if query in b:
                    raise RuntimeError("A/B cross-support self-retrieval detected")
            if group == "G1-SCPR-Full":
                if len(best["scpr_train_ids"]) != len(allowed):
                    raise RuntimeError(f"SCPR train memory size mismatch fold {fold}")

            base_trace = {int(entry["epoch"]): entry for entry in base_task["rng_trace"]}
            task_trace = {int(entry["epoch"]): entry for entry in task["rng_trace"]}
            if task_trace != {int(entry["epoch"]): entry for entry in payload["rng_trace"]}:
                raise RuntimeError(f"task/checkpoint RNG traces differ: {group} fold {fold}")
            common = sorted(set(task_trace) & set(base_trace))
            mismatches = [epoch for epoch in common if any(task_trace[epoch][key] != base_trace[epoch][key]
                                                           for key in RNG_KEYS)]
            if mismatches:
                raise RuntimeError(f"shared RNG trajectory mismatch: {group} fold {fold}, epochs={mismatches[:5]}")
            if any(task_trace[epoch]["rng_after_train"] != task_trace[epoch]["rng_after_eval"] for epoch in common):
                raise RuntimeError(f"new evaluation consumed shared RNG: {group} fold {fold}")
            rng_totals[f"{group}_fold{fold}"] = {"shared_epochs": len(common), "exact_triplets": len(common),
                                                 "mismatches": 0}

            # The four original optimizers are preserved, with SCPR in its own AdamW.
            train_params = [("opt_base", .0038, .11), ("opt_cam", .0038, .11), ("opt_added", .001, .01)]
            for key, lr, decay in train_params:
                for pg in best["base_state"][key]["param_groups"]:
                    if abs(float(pg["lr"]) - lr) > 1e-12 or abs(float(pg["weight_decay"]) - decay) > 1e-12:
                        raise RuntimeError(f"historical optimizer hyperparameters changed: {group} fold {fold} {key}")
            for pg in best["new_optimizer"]["param_groups"]:
                if abs(float(pg["lr"]) - .0005) > 1e-12 or abs(float(pg["weight_decay"]) - .01) > 1e-12:
                    raise RuntimeError(f"SCPR optimizer hyperparameters changed: {group} fold {fold}")

            state, _, replay_identity = runner.prepare_base(fold)
            state["fold"] = fold
            if replay_identity != identity:
                raise RuntimeError(f"base identity changed before checkpoint replay: {group} fold {fold}")
            hist.load_bundle(state, best["base_state"])
            state["model"].eval(); state["cam"].eval()
            memory = {key: (value.to(state["y"].device) if torch.is_tensor(value) else value)
                      for key, value in memory.items()}
            module = scpr_code.SCPR(text_vectors).to(state["y"].device)
            module.load_state_dict(best["scpr"])
            module.eval()
            shapes = {key: tuple(value.shape) for key, value in best["scpr"].items()}
            count = sum(parameter.numel() for parameter in module.parameters())
            if fold in parameter_shapes and parameter_shapes[fold] != shapes:
                raise RuntimeError(f"G1/G2 parameter structures differ at fold {fold}")
            if fold in parameter_counts and parameter_counts[fold] != count:
                raise RuntimeError(f"G1/G2 parameter counts differ at fold {fold}")
            parameter_shapes[fold], parameter_counts[fold] = shapes, count
            group_expected_text = text_vectors if group == "G1-SCPR-Full" else scpr_code.anonymous_pair(text_vectors)
            group_expected_text = F.normalize(group_expected_text.float(), dim=-1)
            if not torch.allclose(module.text_vectors.detach().cpu(), group_expected_text.cpu(), atol=1e-6, rtol=0):
                raise RuntimeError(f"G1/G2 semantic vector identity mismatch: {group} fold {fold}")
            gram = module.text_vectors.detach().cpu() @ module.text_vectors.detach().cpu().T
            prompt_grams[group] = gram.tolist()

            with torch.no_grad():
                evaluated, cam = runner.hook_forward(state, training=False)
            if cam.shape != (871, 144):
                raise RuntimeError(f"wrong CAM144 source: {group} fold {fold}")
            index = torch.as_tensor(test_ids, dtype=torch.long, device=cam.device)
            support = torch.as_tensor(sorted(allowed), dtype=torch.long, device=cam.device)
            sites_device = sites.to(cam.device)
            base_margin = evaluated["logp"][index, 0] - evaluated["logp"][index, 1]
            scores, diagnostics = module(cam[index], test_ids, memory, support, sites_device, base_margin)
            if diagnostics["evidence"].shape != (len(test_ids), 2, 4):
                raise RuntimeError(f"both ASD/TD candidates were not evaluated: {group} fold {fold}")
            delta = diagnostics["delta"]
            if not torch.allclose(scores - base_margin, delta, atol=2e-6, rtol=0):
                raise RuntimeError(f"returned residual does not match bounded score delta: {group} fold {fold}")
            if float(delta.detach().abs().max()) > 1.50001:
                raise RuntimeError(f"residual exceeded the prescribed bound: {group} fold {fold}")
            full_logp = runner.apply_score(evaluated["logp"], test_ids, scores - base_margin, module)
            probability = full_logp[index].exp().detach().cpu().numpy()
            prediction = probability.argmax(1)
            metrics = score_metrics(test_target, probability)
            stored_probability = np.asarray(best["probability"])
            if not np.allclose(stored_probability, probability, atol=2e-6, rtol=0):
                raise RuntimeError(f"independent best-checkpoint probability replay mismatch: {group} fold {fold}")
            if not np.array_equal(np.asarray(best["prediction"]), prediction):
                raise RuntimeError(f"independent hard prediction replay mismatch: {group} fold {fold}")
            for metric in CORE + ("nll", "brier"):
                if metric in best["metrics"] and abs(metrics[metric] - float(best["metrics"][metric])) > 2e-6:
                    raise RuntimeError(f"checkpoint metric replay mismatch: {group} fold {fold} {metric}")
            off_logp = runner.apply_score(evaluated["logp"], test_ids, torch.zeros_like(scores), module)
            off_probability = off_logp[index].exp().detach().cpu().numpy()
            off_prediction = off_probability.argmax(1)
            off_metrics = score_metrics(test_target, off_probability)
            by_group[group][fold] = {"metrics": metrics, "probability": probability,
                                     "prediction": prediction, "evidence": diagnostics["evidence"].detach().cpu().numpy(),
                                     "delta": delta.detach().cpu().numpy(),
                                     "gate": diagnostics["gate"].detach().cpu().numpy(),
                                     "modality_weights": diagnostics["modality_weights"].detach().cpu().numpy(),
                                     "coverage": diagnostics["coverage"].detach().cpu().numpy(),
                                     "hard_similarity": diagnostics["hard_similarity"].detach().cpu().numpy(),
                                     "off_metrics": off_metrics, "off_probability": off_probability,
                                     "off_prediction": off_prediction, "best_epoch": int(best["epoch"]),
                                     "identity": identity, "checkpoint_sha256": digest,
                                     "alpha": float(module.alpha.detach().cpu()),
                                     "rho": (.2 * torch.sigmoid(module.theta.detach())).cpu().numpy(),
                                     "memory": memory, "support": sorted(allowed), "query_cam": cam[index].detach().cpu(),
                                     "sites": sites, "target": test_target, "base_prediction": base_prediction,
                                     "task": task}
            if group == "G1-SCPR-Full":
                off_scores[fold] = {"metrics": off_metrics, "probability": off_probability,
                                    "prediction": off_prediction, "target": test_target}
            # Test labels are used only for post-hoc strata; memory and gate inputs remain train-only.
            diag_evidence = diagnostics["evidence"].detach().cpu().numpy()
            modality = diagnostics["modality_weights"].detach().cpu().numpy()
            diff = diagnostics["slot_difference"].detach().cpu().numpy()
            delta_values = delta.detach().cpu().numpy()
            d = (modality * diff).sum(1)
            actual_direction = np.sign(delta_values)
            expected_direction = np.sign(d * math.tanh(float(module.alpha.detach().cpu())))
            valid_direction = (actual_direction != 0) & (expected_direction != 0)
            direction_agreement = float((actual_direction[valid_direction] == expected_direction[valid_direction]).mean()) if valid_direction.any() else 0.0
            role = role_metadata(memory, sorted(allowed), sites)
            site_diag = retrieval_site_diagnostics(cam[index], test_ids, memory, sorted(allowed), sites)
            class_modality, class_evidence = {}, {}
            for cls, name in ((0, "asd"), (1, "td")):
                mask = test_target == cls
                class_modality[name] = modality[mask].mean(0).tolist() if mask.any() else []
                class_evidence[name] = diag_evidence[mask].mean(0).tolist() if mask.any() else []
            base_correct = base_prediction == test_target
            gate_values = diagnostics["gate"].detach().cpu().numpy()
            modality_all = diagnostics["modality_weights"].detach().cpu()
            evidence_tensor = diagnostics["evidence"].detach().cpu()
            rho_values = (.2 * torch.sigmoid(module.theta.detach())).cpu().numpy()
            mechanism_rows.append({"group": group, "fold": fold, "best_epoch_1based": int(best["epoch"]),
                "memory_updates_at_best": int(memory["updates"]), "memory_observed": int(observed.sum()),
                "support_asd_reliable": role["asd"]["reliable_n"], "support_asd_boundary": role["asd"]["boundary_n"],
                "support_asd_hard": role["asd"]["hard_n"], "support_asd_hard_enabled": role["asd"]["hard_enabled"],
                "support_td_reliable": role["td"]["reliable_n"], "support_td_boundary": role["td"]["boundary_n"],
                "support_td_hard": role["td"]["hard_n"], "support_td_hard_enabled": role["td"]["hard_enabled"],
                "support_asd_candidate_sites": role["asd"]["candidate_sites"],
                "support_td_candidate_sites": role["td"]["candidate_sites"],
                "support_asd_site_counts": role["asd"]["candidate_site_counts"],
                "support_td_site_counts": role["td"]["candidate_site_counts"],
                "gate_mean": float(gate_values.mean()), "gate_sd": float(gate_values.std()),
                "gate_p10": float(np.quantile(gate_values, .10)), "gate_p90": float(np.quantile(gate_values, .90)),
                "delta_abs_mean": float(np.abs(delta_values).mean()), "delta_abs_median": float(np.median(np.abs(delta_values))),
                "delta_abs_p90": float(np.quantile(np.abs(delta_values), .90)), "delta_abs_max": float(np.abs(delta_values).max()),
                "delta_nonzero_fraction": float((np.abs(delta_values) > 1e-6).mean()),
                "prototype_coverage_mean": float(diagnostics["coverage"].mean()),
                "hard_similarity_nonzero_fraction": float((diagnostics["hard_similarity"] != 0).float().mean()),
                "evidence_delta_mean": float(d.mean()), "evidence_delta_sd": float(d.std()),
                "evidence_residual_direction_agreement": direction_agreement,
                "evidence_asd_mean": evidence_tensor[:, 0].mean(0).tolist(),
                "evidence_td_mean": evidence_tensor[:, 1].mean(0).tolist(),
                "evidence_true_asd": class_evidence["asd"], "evidence_true_td": class_evidence["td"],
                "modality_pheno": float(modality[:, 0].mean()), "modality_anat": float(modality[:, 1].mean()),
                "modality_func": float(modality[:, 2].mean()), "modality_corr": float(modality[:, 3].mean()),
                "modality_asd": class_modality["asd"], "modality_td": class_modality["td"],
                "modality_base_correct": modality[base_correct].mean(0).tolist() if base_correct.any() else [],
                "modality_base_error": modality[~base_correct].mean(0).tolist() if (~base_correct).any() else [],
                "semantic_rho_pheno": float(rho_values[0]), "semantic_rho_anat": float(rho_values[1]),
                "semantic_rho_func": float(rho_values[2]), "semantic_rho_corr": float(rho_values[3]),
                "residual_off_acc": off_metrics["acc"], "full_acc": metrics["acc"],
                "residual_changed_predictions": int((prediction != off_prediction).sum()),
                "mean_abs_probability_change": float(np.abs(probability[:, 0] - off_probability[:, 0]).mean()),
                "parameter_count": count, **site_diag})

            # Check every newly trained task has real gradients after zero-init warmup.
            grad_rows = task.get("gradient_audit", [])
            active_grads = [row for row in grad_rows if int(row["epoch"]) >= 13]
            grad_fields = ("cam144", "semantic_projection", "modality_gate", "residual_gate", "alpha")
            if not active_grads or any(not any(float(row.get(field, 0)) > 0 for row in active_grads)
                                       for field in grad_fields):
                raise RuntimeError(f"end-to-end SCPR gradient path did not activate: {group} fold {fold}")
            if not task.get("scpr_grad_nonzero") or not task.get("cam_hook_valid"):
                raise RuntimeError(f"new module gradient/CAM hook audit failed: {group} fold {fold}")

            fold_rows.append({"group": group, "fold": fold, **metrics,
                              "best_epoch_1based": int(best["epoch"]),
                              "epochs_executed": int(payload["epochs_executed"]),
                              "checkpoint_sha256": digest})
            fold_rows.append({"group": "G1-SCPR-Residual-Off" if group == "G1-SCPR-Full" else f"{group}-Residual-Off",
                              "fold": fold, **off_metrics, "best_epoch_1based": int(best["epoch"]),
                              "epochs_executed": int(payload["epochs_executed"]), "checkpoint_sha256": digest})
            audit_rows.append({"group": group, "fold": fold, "status": "PASS",
                               "checkpoint_sha256": digest, "warmup_sha256": warm_sha,
                               "pair_sha": identity["pair_sha"], "optimizer_states_complete": True,
                               "dynamic_memory_updates": int(memory["updates"]), "memory_scope_pass": True,
                               "ab_support_split_pass": True, "rng_shared_epochs": len(common),
                               "rng_mismatches": 0, "cam144_shape": list(cam.shape),
                               "outputlayer_calls": 1, "parameter_count": count,
                               "gradient_path_pass": True, "metrics_recomputed": True,
                               "gate_off_readonly": True})

    # Both groups use equal-sized, equal-geometry text pairs; only their content differs.
    gram_g1 = torch.as_tensor(prompt_grams["G1-SCPR-Full"])
    gram_g2 = torch.as_tensor(prompt_grams["G2-SCPR-Anonymous"])
    if not torch.allclose(gram_g1, gram_g2, atol=1e-6, rtol=0):
        raise RuntimeError("anonymous text pair did not preserve the true prompt pair geometry")
    if len(baseline) != 10 or any(len(values) != 10 for values in by_group.values()) or len(audit_rows) != 30:
        raise RuntimeError("not all ten historical and twenty new checkpoints were audited")

    # First verify and compare the fixed-checkpoint residual-off intervention.
    for fold in range(1, 11):
        full = by_group["G1-SCPR-Full"][fold]
        target = full["target"]
        base_pred = baseline[fold]["prediction"]
        full_pred = full["prediction"]
        off_pred = full["off_prediction"]
        for name, old_pred, new_pred in (("G1-vs-E74.4-G2", base_pred, full_pred),
                                         ("G1-Full-vs-Residual-Off", off_pred, full_pred)):
            old_correct, new_correct = old_pred == target, new_pred == target
            direction = {"ASD_to_TD_corrected": int(((old_pred == 0) & (new_pred == 1) & (target == 1) & ~old_correct).sum()),
                         "TD_to_ASD_corrected": int(((old_pred == 1) & (new_pred == 0) & (target == 0) & ~old_correct).sum())}
            correction_rows.append({"comparison": name, "fold": fold,
                "wrong_to_correct": int((~old_correct & new_correct).sum()),
                "correct_to_wrong": int((old_correct & ~new_correct).sum()),
                "net_corrections": int((~old_correct & new_correct).sum() - (old_correct & ~new_correct).sum()),
                "predictions_changed": int((old_pred != new_pred).sum()),
                "ASD_to_TD_corrected": direction["ASD_to_TD_corrected"],
                "TD_to_ASD_corrected": direction["TD_to_ASD_corrected"],
                "fold2_overlap_test_ids_0based": json.dumps(baseline[fold]["overlap"] if fold == 2 else [])})
        for group in runner.GROUPS:
            pred = by_group[group][fold]["prediction"]
            old_correct, new_correct = base_pred == target, pred == target
            correction_rows.append({"comparison": f"{group}-vs-E74.4-G2", "fold": fold,
                "wrong_to_correct": int((~old_correct & new_correct).sum()),
                "correct_to_wrong": int((old_correct & ~new_correct).sum()),
                "net_corrections": int((~old_correct & new_correct).sum() - (old_correct & ~new_correct).sum()),
                "predictions_changed": int((base_pred != pred).sum()),
                "ASD_to_TD_corrected": int(((base_pred == 0) & (pred == 1) & (target == 1) & ~old_correct).sum()),
                "TD_to_ASD_corrected": int(((base_pred == 1) & (pred == 0) & (target == 0) & ~old_correct).sum()),
                "fold2_overlap_test_ids_0based": json.dumps(baseline[fold]["overlap"] if fold == 2 else [])})

    metric_maps = {runner.BASE_GROUP: {f: baseline[f]["metrics"] for f in range(1, 11)},
                   **{group: {f: by_group[group][f]["metrics"] for f in range(1, 11)} for group in runner.GROUPS},
                   "G1-SCPR-Residual-Off": {f: off_scores[f]["metrics"] for f in range(1, 11)}}
    summary = summary_rows(fold_rows)
    pairings = []
    pairings.extend(paired_rows("G1-SCPR-Full - E74.4-G2", metric_maps["G1-SCPR-Full"], metric_maps[runner.BASE_GROUP]))
    pairings.extend(paired_rows("G2-SCPR-Anonymous - E74.4-G2", metric_maps["G2-SCPR-Anonymous"], metric_maps[runner.BASE_GROUP]))
    pairings.extend(paired_rows("G1-SCPR-Full - G2-SCPR-Anonymous", metric_maps["G1-SCPR-Full"], metric_maps["G2-SCPR-Anonymous"]))
    pairings.extend(paired_rows("G1-Full - G1-Residual-Off", metric_maps["G1-SCPR-Full"], metric_maps["G1-SCPR-Residual-Off"]))

    def corr_totals(comparison):
        rows = [r for r in correction_rows if r["comparison"] == comparison]
        keys = ("wrong_to_correct", "correct_to_wrong", "net_corrections", "predictions_changed",
                "ASD_to_TD_corrected", "TD_to_ASD_corrected")
        return {key: sum(int(row[key]) for row in rows) for key in keys}

    g1_base_corr = corr_totals("G1-vs-E74.4-G2")
    g1_off_corr = corr_totals("G1-Full-vs-Residual-Off")
    acc_g1_base = next(r for r in pairings if r["comparison"] == "G1-SCPR-Full - E74.4-G2" and r["scope"] == "all10" and r["metric"] == "acc")
    hard_g1_base = next(r for r in pairings if r["comparison"] == "G1-SCPR-Full - E74.4-G2" and r["scope"] == "all10" and r["metric"] == "hard_auc")
    prob_g1_base = next(r for r in pairings if r["comparison"] == "G1-SCPR-Full - E74.4-G2" and r["scope"] == "all10" and r["metric"] == "prob_auc")
    acc_g1_g2 = next(r for r in pairings if r["comparison"] == "G1-SCPR-Full - G2-SCPR-Anonymous" and r["scope"] == "all10" and r["metric"] == "acc")
    acc_g1_nine = next(r for r in pairings if r["comparison"] == "G1-SCPR-Full - E74.4-G2" and r["scope"] == "exclude_fold2" and r["metric"] == "acc")
    gates = np.asarray([row["gate_mean"] for row in mechanism_rows])
    gate_sds = np.asarray([row["gate_sd"] for row in mechanism_rows])
    deltas = np.asarray([row["delta_abs_mean"] for row in mechanism_rows])
    gate_is_inactive = bool(np.all(deltas < 1e-6))
    gate_is_nearly_constant_high = bool(np.mean(gates) > .98 and np.mean(gate_sds) < .01)
    if (acc_g1_base["mean_difference_pp"] <= 0 or g1_base_corr["net_corrections"] <= 0 or
            acc_g1_nine["mean_difference_pp"] <= 0 or gate_is_inactive or gate_is_nearly_constant_high):
        conclusion = "NO_GO_SCPR_END_TO_END"
    elif acc_g1_g2["mean_difference_pp"] > 0 and acc_g1_g2["wins"] > acc_g1_g2["losses"]:
        conclusion = "PROMISING_SCPR_END_TO_END"
    else:
        conclusion = "PROTOTYPE_GAIN_SEMANTICS_UNVERIFIED"

    write_csv(results_dir / "checkpoint_audit.csv", audit_rows)
    write_csv(results_dir / "fold_metrics.csv", fold_rows)
    write_csv(results_dir / "paired_comparisons.csv", pairings)
    write_csv(results_dir / "mechanism.csv", mechanism_rows)
    write_csv(results_dir / "correction_analysis.csv", correction_rows)
    write_csv(results_dir / "summary.csv", summary)
    replays = [run_restore_replay("G1-SCPR-Full", 1), run_restore_replay("G2-SCPR-Anonymous", 1)]
    if any(row["status"] != "PASS" for row in replays):
        raise RuntimeError("full checkpoint restore replay failed")

    preflight_hashes = preflight["source_hashes"]
    source_hashes_final = {path: runner.sha_file(Path(path)) for path in preflight_hashes if path != audit_path}
    manifest = {
        "experiment": "Experiment 76.5 - Semantic-Conditioned Prototype Residual Learning",
        "status": "COMPLETE_AUDITED", "scientific_conclusion": conclusion,
        "experiment_root": str(EXP), "preferred_root": str(runner.REPO / "experiment/experiment76_5_scpr"),
        "root_fallback_reason": "/data OSSFS returned EIO for new experiment writes; used the requested /model fallback",
        "base_group": runner.BASE_GROUP, "base_manifest_status": base_manifest["status"],
        "base_audit_status": base_audit_status["status"], "base_historical_recomputed_folds": len(baseline),
        "expected_new_tasks": 20, "completed_new_tasks": 20, "independently_audited_new_checkpoints": 20,
        "audited_checkpoint_total_including_base": len(audit_rows), "checkpoint_audit": "PASS 30/30",
        "independent_prediction_recompute": "PASS 30/30", "device": torch.cuda.get_device_name(0),
        "preflight_status": preflight["status"], "preflight_sha256": runner.sha_file(results_dir / "preflight.json"),
        "training_config_sha256": preflight_hashes[str(EXP / "config.json")],
        "final_config_sha256": runner.sha_file(EXP / "config.json"),
        "final_config_status": json.loads((EXP / "config.json").read_text())["status"],
        "config_change_after_training": "status field only; verified by reconstructing and matching the exact training-time SHA",
        "training_source_hashes": preflight_hashes,
        "audit_sha256_at_preflight": preflight_hashes[audit_path],
        "audit_sha256_after_independent_rewrite": runner.sha_file(EXP / "audit.py"),
        "all_code_and_historical_source_hashes_match_preflight": True,
        "source_hashes_final_non_audit": source_hashes_final,
        "formal_groups": list(runner.GROUPS), "folds": 10, "seed": 0,
        "max_epochs": 450, "early_stop_wait_gt": 200,
        "worker_limit_used": 7, "external_workers_observed_at_start": 3,
        "max_global_workers_after_resume": 10,
        "concurrency_note": "An initial E76.5 launch briefly overlapped 10 local workers with 3 external workers; it was interrupted at saved checkpoints and resumed with 7 local workers (10 global maximum). No OOM occurred.",
        "warmup_retrained": False, "base_best_checkpoint_finetuned": False,
        "warmup_and_optimizer_identity": "PASS 10/10", "relation_pairs_identity": "PASS 4096/fold",
        "initialization_and_one_step_parity": "PASS fold 1; all 20 folds exact E61/E74.4 identities",
        "shared_rng": {"status": "PASS", "fold_group_records": rng_totals,
                       "common_epoch_records": sum(x["shared_epochs"] for x in rng_totals.values()),
                       "exact_triplets": sum(x["exact_triplets"] for x in rng_totals.values()), "mismatches": 0},
        "restore_replay": replays, "gradient_path": "PASS 20/20; CAM144 and all active SCPR paths nonzero",
        "memory_scope": "PASS: effective outer-train patients only; test and fold-2 overlap rows excluded",
        "dynamic_ema_memory": "PASS: detached full memory and update count saved in latest/best checkpoints",
        "ab_cross_support": "PASS: deterministic class-stratified split, no self support",
        "prompt_source": "BrainPrompt official ABIDE prompt file; cached frozen CLIP RN50; ASD=0, TD=1",
        "prompt_geometry": "PASS: G2 signed coordinate permutation preserves pairwise Gram matrix",
        "outputlayer_and_cam144": "PASS: original OutputLayer executes once; CAM144 [871,144] captured",
        "gate_off": "PASS: read-only inference intervention; checkpoints unchanged",
        "scientific_metrics": {"G1_minus_base_ACC_pp": acc_g1_base["mean_difference_pp"],
                               "G1_minus_base_HardAUC_pp": hard_g1_base["mean_difference_pp"],
                               "G1_minus_base_ProbAUC_pp": prob_g1_base["mean_difference_pp"],
                               "G1_minus_G2_ACC_pp": acc_g1_g2["mean_difference_pp"],
                               "G1_minus_base_ACC_excluding_fold2_pp": acc_g1_nine["mean_difference_pp"],
                               "G1_vs_base_corrections": g1_base_corr,
                               "G1_full_vs_residual_off": g1_off_corr,
                               "gate_mean_over_folds": float(gates.mean()),
                               "gate_sd_mean_over_folds": float(gate_sds.mean()),
                               "mean_abs_residual_over_folds": float(deltas.mean()),
                               "gate_nearly_constant_high": gate_is_nearly_constant_high},
        "protocol_limitations": ["Historical val=test checkpoint selection retained.",
                                 "RFE/standardization before outer CV and transductive graph retained.",
                                 "Fold 2 historical base training overlaps test at row IDs 377 and 407 (1-based); both were excluded from SCPR memory.",
                                 "Results are protocol-matched exploratory comparisons, not unbiased independent generalization estimates."],
        "audit_wall_seconds": time.time() - started,
    }
    # The manifest/report are generated only after all independent assertions pass.
    atomic_json(EXP / "manifest.json", manifest)
    audit_status = {"status": "COMPLETE_AUDITED", "audit": "PASS",
                    "audited_historical_base_checkpoints": 10,
                    "audited_new_checkpoints": 20, "audited_checkpoint_total": 30,
                    "base_checkpoint_sha_and_warmup_identity": "PASS 10/10",
                    "independent_prediction_recompute": "PASS 30/30",
                    "shared_rng_common_epoch_records": sum(x["shared_epochs"] for x in rng_totals.values()),
                    "shared_rng_exact_triplets": sum(x["exact_triplets"] for x in rng_totals.values()),
                    "shared_rng_mismatches": 0, "restore_replay": replays,
                    "memory_scope": "PASS", "A_B_support": "PASS",
                    "gradient_path": "PASS 20/20", "G1_G2_parameter_match": "PASS 10/10",
                    "prompt_geometry": "PASS", "outputlayer_cam144": "PASS",
                    "residual_off": "READ_ONLY_INTERVENTION",
                    "final_config_status": json.loads((EXP / "config.json").read_text())["status"],
                    "config_update": "status-field-only; training SHA retained in manifest",
                    "code_and_historical_source_hashes": "PASS",
                    "scientific_conclusion": conclusion}
    atomic_json(results_dir / "audit_status.json", audit_status)

    # Save the report as a plain research record; no performance data are used for training.
    base_summary, g1_summary, g2_summary = summary[0], summary[1], summary[2]
    table = ["|Method|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|NLL|Brier|ECE|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary[:3]:
        label = {runner.BASE_GROUP: "E74.4-G2 historical base", runner.GROUPS[0]: "E76.5-G1 SCPR-Full",
                 runner.GROUPS[1]: "E76.5-G2 SCPR-Anonymous"}[row["group"]]
        vals = [f"{100*row[m+'_mean']:.4f} ± {100*row[m+'_sd_pop']:.4f}" for m in CORE]
        vals += [f"{row[m+'_mean']:.5f} ± {row[m+'_sd_pop']:.5f}" for m in ("nll", "brier", "ece")]
        table.append("|" + label + "|" + "|".join(vals) + "|")
    pair_table = ["|Comparison|Scope|Metric|Mean Δ (pp)|W/T/L|Descriptive 95% CI (pp)|",
                  "|---|---|---|---:|---:|---:|"]
    for row in pairings:
        ci = f"[{row['descriptive_95ci_low_pp']:.3f}, {row['descriptive_95ci_high_pp']:.3f}]"
        pair_table.append(f"|{row['comparison']}|{row['scope']}|{row['metric']}|{row['mean_difference_pp']:+.3f}|{row['wins']}/{row['ties']}/{row['losses']}|{ci}|")
    mech_table = ["|Group|Gate mean (fold mean)|Gate SD|Mean abs delta|Coverage|PHENO|ANAT|FUNC|CORR|Same-site support|Fallback over cap|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    role_table = ["|Group|Reliable ASD|Reliable TD|Hard ASD|Hard TD|Hard memory enabled|",
                  "|---|---:|---:|---:|---:|---:|"]
    evidence_table = ["|Group|Mean e_ASD-e_TD PHENO|ANAT|FUNC|CORR|rho PHENO|ANAT|FUNC|CORR|",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in runner.GROUPS:
        rows = [r for r in mechanism_rows if r["group"] == group]
        values = [np.mean([r["gate_mean"] for r in rows]), np.mean([r["gate_sd"] for r in rows]),
                  np.mean([r["delta_abs_mean"] for r in rows]), np.mean([r["prototype_coverage_mean"] for r in rows]),
                  *[np.mean([r[f"modality_{name.lower()}"] for r in rows]) for name in ("PHENO", "ANAT", "FUNC", "CORR")],
                  np.mean([r["same_site_support_fraction"] for r in rows]),
                  np.mean([r["fallback_over_site_cap_fraction"] for r in rows])]
        mech_table.append("|" + group + "|" + "|".join(f"{float(v):.4f}" for v in values) + "|")
        roles = [np.mean([r[f"support_{name}_{role}"] for r in rows])
                 for name, role in (("asd", "reliable"), ("td", "reliable"), ("asd", "hard"), ("td", "hard"))]
        hard_enabled = np.mean([bool(r["support_asd_hard_enabled"]) and bool(r["support_td_hard_enabled"])
                                for r in rows])
        role_table.append("|" + group + "|" + "|".join([*(f"{v:.1f}" for v in roles), f"{hard_enabled:.1%}"]) + "|")
        evidence_diff = [np.mean([(r["evidence_asd_mean"] if isinstance(r["evidence_asd_mean"], list)
                                   else json.loads(r["evidence_asd_mean"]))[slot] -
                                  (r["evidence_td_mean"] if isinstance(r["evidence_td_mean"], list)
                                   else json.loads(r["evidence_td_mean"]))[slot] for r in rows])
                         for slot in range(4)]
        rho_values = [np.mean([r[f"semantic_rho_{slot}"] for r in rows]) for slot in ("pheno", "anat", "func", "corr")]
        evidence_table.append("|" + group + "|" + "|".join(f"{v:+.4f}" for v in evidence_diff) + "|" +
                              "|".join(f"{v:.4f}" for v in rho_values) + "|")
    report_lines = [
        "# Experiment 76.5: SCPR", "",
        f"审计状态：**COMPLETE_AUDITED**；科学判定：**{conclusion}**。形式训练20/20，独立审计30/30（含历史基座10折）。",
        "", "## 方法与执行", "",
        "在E74.4-G2唯一基座上，每折从对应E61 epoch-10 warmup恢复GFR/DARG/CAM/OutputLayer及三个历史优化器；没有从E74.4最佳模型微调。新增SCPR以CAM144的4×36来源槽位建立连续可靠、边界和困难混淆记忆，训练期用eval特征以0.9/0.1 EMA更新，训练查询采用固定A/B交叉支持；Fold 2的历史重叠测试患者排除在SCPR memory之外。真实BrainPrompt疾病Prompt与几何匹配匿名向量各训练10折。原分类器结构、DARG/PEN和原损失项保留，SCPR残差零初始化、1.5限幅。",
        "", "SPromptGL原论文包含GFR多关系图传播、CAM四模态交互、原始8条Prompt及PLN/PEN局部语义加权和RES。E76.5没有重新实现PEN，也没有声称首次组合Prompt与Prototype；BrainPrompt和HP²L已涉及疾病语义与原型。本实验的新增主张限于患者图后CAM来源槽位上的连续混淆证据检索、双疾病竞争和端到端受限决策残差。",
        "", "原始四个模态路径、DARG图公式、原始8条Prompt、OutputLayer与E74.4-G2损失均沿历史源码只读调用；CAM144由OutputLayer输入hook捕获，每个前向只触发一次。兼容E74.4迁移路径的覆写只存在于E76.5适配器，并校验只读E53 anchor SHA；历史源码未修改。每个新模块参数数为22,106，G1/G2结构和参数量逐折一致。真实RN50文本embedding从已验证冻结缓存读取；G2固定坐标符号/置换对两类向量保持相互内积。E76.5输出目录因`/data` OSSFS新目录写入返回EIO，按指令落在`/model/spromptgl/experiment/experiment76_5_scpr`。",
        f"十折均从E61 epoch-10 warmup恢复；每折E74.4准备态的4096关系对SHA与历史任务一致，E61 warmup文件/自有pair SHA也独立校验。共有epoch RNG轨迹逐项匹配{sum(x['exact_triplets'] for x in rng_totals.values())}/{sum(x['shared_epochs'] for x in rng_totals.values())}。G1/G2 Fold 1的latest恢复重放输出与RNG均完全相同。训练高峰期间因初始调度与3个外部worker重叠曾短暂达到13个进程，随后在保存点中断并以7个本实验worker恢复，之后全局上限10；无OOM。",
        "", "## 十折指标", "", "核心指标单位为百分比，报告mean ± population SD；Hard-AUC沿用历史BA。Prob-AUC以连续ASD概率计算。NLL、Brier、ECE为概率量。", "",
        *table, "", "## 配对比较", "", "单位为百分点；W/T/L按逐折差值统计，95%区间为描述性t区间，不把患者数当作折数。Fold 2以外结果仅为敏感性描述。", "",
        *pair_table, "", "## 纠错与残差", "",
        f"G1相对历史基座：错误→正确 {g1_base_corr['wrong_to_correct']}，正确→错误 {g1_base_corr['correct_to_wrong']}，净纠错 {g1_base_corr['net_corrections']}；预测改变 {g1_base_corr['predictions_changed']}。按真实类别方向，ASD→TD纠正 {g1_base_corr['ASD_to_TD_corrected']}，TD→ASD纠正 {g1_base_corr['TD_to_ASD_corrected']}。",
        f"固定同一G1最佳模型关闭推理残差后：G1-Full相对Residual-Off错误→正确 {g1_off_corr['wrong_to_correct']}，正确→错误 {g1_off_corr['correct_to_wrong']}，净纠正 {g1_off_corr['net_corrections']}；这是推理残差直接干预，不是未训练SCPR的基座。Checkpoints未被修改。",
        "", "## 原型和模态机制", "", "可靠/边界/困难候选规模、站点覆盖与Top-K回退、检索支持分布、双疾病证据、门控/残差分布均记录于`mechanism.csv`。测试真实标签只用于下列分组诊断和指标，不进入prototype memory或gate input。模态权重表示CAM来源槽位的模型权重，不能据此宣称具体脑区生物标志物。", "",
        *mech_table, "", "原型角色（10折平均训练候选数）：", "", *role_table,
        "双候选槽位证据差为e_ASD−e_TD；正值偏向ASD候选，负值偏向TD候选。ρ为冻结文本向量映射后的可学习注入系数。", "", *evidence_table,
        "", "## 结论和限制", "",
        f"判定依据：G1−基座ACC {acc_g1_base['mean_difference_pp']:+.3f} pp（W/T/L {acc_g1_base['wins']}/{acc_g1_base['ties']}/{acc_g1_base['losses']}），Hard-AUC {hard_g1_base['mean_difference_pp']:+.3f} pp，Prob-AUC {prob_g1_base['mean_difference_pp']:+.3f} pp；排除Fold 2后ACC差 {acc_g1_nine['mean_difference_pp']:+.3f} pp。G1−匿名组ACC {acc_g1_g2['mean_difference_pp']:+.3f} pp（W/T/L {acc_g1_g2['wins']}/{acc_g1_g2['ties']}/{acc_g1_g2['losses']}）。SCPR门控折均值 {gates.mean():.4f}，跨折门控标准差均值 {gate_sds.mean():.4f}，折均绝对残差均值 {deltas.mean():.4f}。",
        "", "审计通过只说明实现和产物按声明可复现，不等于科学方法GO。历史协议仍使用val=test选择checkpoint、CV前RFE/标准化和transductive图；Fold 2中历史基座训练/测试重叠行377与407。九折比较不能消除其他偏差。因此这些是协议匹配的探索性比较，不是严格无偏的独立泛化结果。",
        "", f"训练配置SHA：`{manifest['training_config_sha256']}`；审计后只将配置status字段更新为`COMPLETE_AUDITED`，最终SHA为`{manifest['final_config_sha256']}`。代码和历史来源SHA锁定未变。本次独立审计源码SHA：`{runner.sha_file(EXP / 'audit.py')}`。逐折结果：`fold_metrics.csv`；配对：`paired_comparisons.csv`；机制：`mechanism.csv`；纠错：`correction_analysis.csv`。",
    ]
    (results_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest["report_sha256"] = runner.sha_file(results_dir / "report.md")
    manifest["audit_status_sha256"] = runner.sha_file(results_dir / "audit_status.json")
    manifest["audit_source_sha256"] = runner.sha_file(EXP / "audit.py")
    manifest["independent_restore_replay"] = replays
    atomic_json(EXP / "manifest.json", manifest)
    print(json.dumps(audit_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
