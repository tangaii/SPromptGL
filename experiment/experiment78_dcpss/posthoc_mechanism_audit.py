from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run as R


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA_PATH = R.REPO / "data/ABIDE/processed_standard_data.csv"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_covariates():
    with DATA_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    site_columns = [name for name in columns if name.startswith("SITE_ID_")]
    age_columns = [name for name in columns if name.startswith("AGE_AT_SCAN_")]
    if not site_columns or not age_columns or len(rows) != 871:
        raise RuntimeError("processed ABIDE rows or site/age covariates do not match the expected 871 rows")
    sites, ages, labels = [], [], []
    age_midpoints = {}
    for name in age_columns:
        match = re.search(r"\[([0-9.]+), ([0-9.]+)\)", name)
        if not match:
            raise RuntimeError(f"cannot parse age interval: {name}")
        age_midpoints[name] = (float(match.group(1)) + float(match.group(2))) / 2
    for row_index, row in enumerate(rows):
        active_sites = [name.removeprefix("SITE_ID_") for name in site_columns if int(float(row[name])) == 1]
        active_ages = [name for name in age_columns if int(float(row[name])) == 1]
        if len(active_sites) != 1 or len(active_ages) != 1:
            raise RuntimeError(f"invalid one-hot site/age covariate at row {row_index + 1}")
        sites.append(active_sites[0])
        ages.append(age_midpoints[active_ages[0]])
        raw_label = int(float(row["label"]))
        if raw_label not in (1, 2):
            raise RuntimeError(f"unexpected processed CSV label {raw_label} at row {row_index + 1}")
        labels.append(raw_label - 1)
    return np.asarray(sites, dtype=object), np.asarray(ages, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def cramers_v(values, clusters):
    categories = sorted(set(values))
    k = int(clusters.max()) + 1
    if len(categories) < 2 or k < 2:
        return float("nan")
    table = np.zeros((len(categories), k), dtype=np.float64)
    category_index = {value: index for index, value in enumerate(categories)}
    for value, cluster in zip(values, clusters):
        table[category_index[value], int(cluster)] += 1
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0]
    n = table.sum()
    rows, cols = table.shape
    if min(rows - 1, cols - 1) <= 0:
        return float("nan")
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(np.sum(np.divide(np.square(table - expected), expected, out=np.zeros_like(table), where=expected > 0)))
    return math.sqrt(max(0.0, chi2 / (n * min(rows - 1, cols - 1))))


def eta_squared(values, clusters):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.var(values) == 0:
        return 0.0
    mean = values.mean()
    between = sum(len(values[clusters == k]) * (values[clusters == k].mean() - mean) ** 2 for k in np.unique(clusters))
    return float(between / np.square(values - mean).sum())


def distribution_stats(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    mean = values.mean(0)
    sd = values.std(0, unbiased=False)
    entropy = -(values.clamp_min(1e-12) * values.clamp_min(1e-12).log()).sum(-1)
    argmax = torch.bincount(values.argmax(-1), minlength=values.shape[-1]).double() / len(values)
    return {
        "mean_weights": json.dumps(mean.tolist()),
        "patient_sd": json.dumps(sd.tolist()),
        "mean_patient_sd": float(sd.mean()),
        "entropy_mean": float(entropy.mean()),
        "entropy_max": float(math.log(values.shape[-1])),
        "argmax_frequency": json.dumps(argmax.tolist()),
    }


def prototype_covariate_rows(sites, ages, labels):
    rows = []
    group = R.GROUPS[1]
    for fold in range(1, 11):
        task_path, best_path, _, _ = R.task_paths(group, fold)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        support = np.asarray(checkpoint["support_ids"], dtype=np.int64)
        assignments = np.asarray(checkpoint["assignments"], dtype=np.int64)
        if not np.array_equal(labels, np.asarray(R.base().prepare(fold)[0]["y"].cpu())):
            raise RuntimeError(f"processed CSV labels are not row-aligned with the model in fold {fold}")
        for cls in (0, 1):
            ids = support[labels[support] == cls]
            cluster = assignments[ids]
            if np.any(cluster < 0) or set(np.unique(cluster)) != {0, 1, 2}:
                raise RuntimeError(f"invalid K=3 support membership for fold {fold}, class {cls}")
            rows.append({
                "group": group,
                "fold": fold,
                "class": "ASD" if cls == 0 else "TD",
                "class_code": cls,
                "support_n": len(ids),
                "cluster_counts": json.dumps(np.bincount(cluster, minlength=3).tolist()),
                "site_count": len(set(sites[ids])),
                "site_cramers_v": cramers_v(sites[ids], cluster),
                "age_bin_midpoint_mean": float(ages[ids].mean()),
                "age_eta_squared_by_cluster": eta_squared(ages[ids], cluster),
                "cluster_mean_age_bin_midpoint": json.dumps([float(ages[ids[cluster == k]].mean()) for k in range(3)]),
                "interpretation": "Exploratory association only; clusters are not clinical subtypes.",
            })
    return rows


def model_mechanism_rows(sites, ages, labels):
    rows, covariate_rows = [], []
    for group in (R.GROUPS[0], R.GROUPS[1], R.GROUPS[2], R.GROUPS[3], R.GROUPS[4]):
        k = int(R.CFG["group_specs"][group]["k"])
        embeddings, classes = R.text_payload(group)
        for fold in range(1, 11):
            state, _, identity = R.base().prepare(fold)
            task_path, best_path, _, _ = R.task_paths(group, fold)
            task = json.loads(task_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            if checkpoint["identity"] != identity:
                raise RuntimeError(f"historical base identity mismatch for {group} fold {fold}")
            support = np.asarray(checkpoint["support_ids"], dtype=np.int64)
            assignments = torch.as_tensor(checkpoint["assignments"], dtype=torch.long, device=state["y"].device)
            R.hist().load_bundle(state, checkpoint["base_state"])
            module = R.initialize(embeddings, classes, k, fold, state["y"].device)
            module.load_state_dict(checkpoint["module_state"], strict=True)
            module.eval()
            test_ids = np.asarray(state["test_np"], dtype=np.int64)
            with torch.no_grad():
                base_out = R.forward_with_cam144(state, training=False)
                dcp = module(base_out["cam144"], base_out["logp"], state["y"], support, assignments, test_ids)
                q = F.normalize(module.patient_projection(base_out["cam144"]), dim=-1, eps=1e-8)[test_ids]
                text = F.normalize(module.text_projection(module.text_embeddings), dim=-1, eps=1e-8)
                v = F.normalize((dcp["beta"].unsqueeze(-1) * dcp["prototype"]).sum(dim=2), dim=-1, eps=1e-8)
                direct = q @ text.T

                for cls in (0, 1):
                    prompt_ids = torch.nonzero(module.text_classes == cls, as_tuple=False).flatten()
                    actual = dcp["attention"][cls]
                    stats = distribution_stats(actual)
                    rows.append({
                        "group": group,
                        "fold": fold,
                        "kind": "patient_conditioned_prompt_attention",
                        "class": "ASD" if cls == 0 else "TD",
                        "prototype_cluster": "all_weighted",
                        "prompt_indices": json.dumps(prompt_ids.cpu().tolist()),
                        **stats,
                    })
                    beta_stats = distribution_stats(dcp["beta"][:, cls, :])
                    rows.append({
                        "group": group,
                        "fold": fold,
                        "kind": "patient_prototype_match_beta",
                        "class": "ASD" if cls == 0 else "TD",
                        "prototype_cluster": "all",
                        "prompt_indices": "",
                        **beta_stats,
                    })

                    if k > 1:
                        conditional = []
                        for cluster in range(k):
                            own = dcp["prototype"][:, cls, cluster, :]
                            opposite = v[:, 1 - cls, :]
                            ptext = text[prompt_ids]
                            contrast = own @ ptext.T - opposite @ ptext.T
                            evidence = direct[:, prompt_ids] + contrast
                            weights = torch.softmax(evidence / 0.5, dim=-1)
                            conditional.append(weights)
                            rows.append({
                                "group": group,
                                "fold": fold,
                                "kind": "single_subprototype_conditioned_prompt_attention",
                                "class": "ASD" if cls == 0 else "TD",
                                "prototype_cluster": cluster,
                                "prompt_indices": json.dumps(prompt_ids.cpu().tolist()),
                                **distribution_stats(weights),
                            })
                        pairwise_tv = []
                        for left in range(k):
                            for right in range(left + 1, k):
                                pairwise_tv.append(0.5 * (conditional[left] - conditional[right]).abs().sum(-1))
                        rows[-1]["mean_pairwise_subprototype_attention_tv"] = float(torch.stack(pairwise_tv).mean())
                        class_rows = [row for row in rows if row["fold"] == fold and row["group"] == group and row["class"] == ("ASD" if cls == 0 else "TD") and row["kind"] == "single_subprototype_conditioned_prompt_attention"]
                        for row in class_rows:
                            row.setdefault("mean_pairwise_subprototype_attention_tv", float(torch.stack(pairwise_tv).mean()))

                if group in R.GROUPS[:4]:
                    evidence = dcp["evidence"].detach().cpu().double().numpy()
                    test_labels = state["y"][torch.as_tensor(test_ids, dtype=torch.long, device=state["y"].device)].detach().cpu().numpy()
                    signals = {
                        "e_ASD": evidence[:, 0],
                        "e_TD": evidence[:, 1],
                        "e_ASD_minus_e_TD": evidence[:, 0] - evidence[:, 1],
                    }
                    for stratum, mask in (("all", np.ones(len(test_ids), dtype=bool)), ("ASD", test_labels == 0), ("TD", test_labels == 1)):
                        if not mask.any():
                            raise RuntimeError(f"empty evidence stratum {stratum} in {group} fold {fold}")
                        for signal, values in signals.items():
                            selected = np.asarray(values[mask], dtype=np.float64)
                            rows.append({
                                "group": group,
                                "fold": fold,
                                "kind": "disease_evidence_distribution",
                                "class": stratum,
                                "prototype_cluster": "all",
                                "prompt_indices": "",
                                "signal": signal,
                                "query_n": int(mask.sum()),
                                "mean": float(selected.mean()),
                                "sd": float(selected.std(ddof=0)),
                                "median": float(np.median(selected)),
                                "q25": float(np.quantile(selected, 0.25)),
                                "q75": float(np.quantile(selected, 0.75)),
                                "prompt_count_for_each_class": json.dumps([int(torch.sum(module.text_classes == c)) for c in (0, 1)]),
                                "test_labels_used_only_for_posthoc_stratification": True,
                            })

                if group == R.GROUPS[1]:
                    for cls in (0, 1):
                        ids = support[labels[support] == cls]
                        cluster = assignments.cpu().numpy()[ids]
                        covariate_rows.append({
                            "fold": fold,
                            "class": "ASD" if cls == 0 else "TD",
                            "support_n": len(ids),
                            "cluster_counts": json.dumps(np.bincount(cluster, minlength=3).tolist()),
                            "site_count": len(set(sites[ids])),
                            "site_cramers_v": cramers_v(sites[ids], cluster),
                            "age_bin_midpoint_mean": float(ages[ids].mean()),
                            "age_eta_squared_by_cluster": eta_squared(ages[ids], cluster),
                            "cluster_mean_age_bin_midpoint": json.dumps([float(ages[ids[cluster == c]].mean()) for c in range(3)]),
                            "interpretation": "Exploratory association only; clusters are not clinical subtypes.",
                        })
            del state, module, base_out, dcp
            torch.cuda.empty_cache()
    return rows, covariate_rows


def latest_recovery_replay():
    group, fold = R.GROUPS[1], 1
    state, _, identity, support, assignments, module, optimizer, initial = R.initial_state(fold, group)
    _, _, latest_path, _ = R.task_paths(group, fold)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    required = {"base_state", "module_state", "new_optimizer", "rng", "next_epoch", "best_acc", "best_epoch", "wait", "trace", "mechanism", "source_sha256", "identity", "initial"}
    if not required.issubset(latest):
        raise RuntimeError(f"latest checkpoint missing recovery state: {sorted(required - set(latest))}")
    if latest["identity"] != identity or latest["initial"] != initial or latest["source_sha256"] != R.source_sha():
        raise RuntimeError("latest checkpoint does not match the independently prepared E61/E74.4 state")
    R._restore(state, module, optimizer, latest)
    start = R._snapshot(state, module, optimizer)

    def two_steps():
        records = []
        for _ in range(2):
            losses, diagnostics = R._train_step(state, module, optimizer, support, assignments)
            evaluated = R._evaluate(state, module, support, assignments)
            records.append((losses, diagnostics, R._compact_metrics(evaluated["metrics"])))
        return R._snapshot(state, module, optimizer), records

    first_state, first_records = two_steps()
    R._restore(state, module, optimizer, start)
    second_state, second_records = two_steps()
    if R.sha_object(first_state) != R.sha_object(second_state) or first_records != second_records:
        raise RuntimeError("two-step replay from actual latest checkpoint did not match")
    return {
        "status": "PASS",
        "group": group,
        "fold": fold,
        "restored_latest_epoch": int(latest["next_epoch"]),
        "restored_wait": int(latest["wait"]),
        "base_optimizer_keys": sorted(latest["base_state"]),
        "module_state_keys": sorted(latest["module_state"]),
        "new_optimizer_state_present": bool(latest["new_optimizer"].get("state")),
        "two_step_model_optimizer_rng_replay": True,
    }


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("post-hoc audit expects the recorded single-GPU environment")
    sites, ages, labels = load_covariates()
    mechanism_rows, reproduced_covariates = model_mechanism_rows(sites, ages, labels)
    covariate_rows = reproduced_covariates
    recovery = latest_recovery_replay()
    atomic_csv(RESULTS / "prompt_prototype_diagnostics.csv", mechanism_rows)
    atomic_csv(RESULTS / "prototype_covariate_diagnostics.csv", covariate_rows)
    atomic_json(RESULTS / "latest_recovery_replay.json", recovery)
    status = {
        "status": "PASS",
        "script_sha256": sha_file(__file__),
        "dataset_sha256": sha_file(DATA_PATH),
        "source_sha256": R.source_sha(),
        "formal_checkpoints_modified": 0,
        "formal_training_or_selection_changed": False,
        "prompt_prototype_rows": len(mechanism_rows),
        "prototype_covariate_rows": len(covariate_rows),
        "recovery_replay": recovery,
        "files": {
            "prompt_prototype_diagnostics.csv": sha_file(RESULTS / "prompt_prototype_diagnostics.csv"),
            "prototype_covariate_diagnostics.csv": sha_file(RESULTS / "prototype_covariate_diagnostics.csv"),
            "latest_recovery_replay.json": sha_file(RESULTS / "latest_recovery_replay.json"),
        },
    }
    atomic_json(RESULTS / "posthoc_mechanism_audit.json", status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
