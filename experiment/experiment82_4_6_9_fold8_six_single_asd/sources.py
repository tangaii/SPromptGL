"""Verify archived provenance and replay historical Best weights read-only."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from core import CONFIG, FOLD, GROUPS, MODEL_VARIANT, RUNTIME, SOURCES, evaluator, h, parent

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

E6 = "E82.4.6.6-G1"
E7 = "E82.4.6.7-G1"
INTERVENTIONS = {"full": "full", "residual_off": "residual_off", "anonymous": "anonymous_query"}
FIELDS = tuple(f"{field}_{mode}" for mode in INTERVENTIONS for field in ("prediction", "p_asd", "true_margin"))
CAM_FIELDS = ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "cam_l2",
              "cam_relative_delta", "query_feature_change_l2")
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
IDENTITY_FILES = ("best", "latest", "journal")
FOLDS = tuple(range(1, 11))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verified(path, expected):
    h.require(h.sha_file(path) == expected, f"SHA-256 differs: {path}")
    return expected


def verify_sources():
    manifests, guards = {}, {}
    for name, (directory, expected) in CONFIG["source_manifests"].items():
        path = Path(directory) / "manifest.json"
        guards[str(path)] = verified(path, expected)
        manifest = manifests[name] = h.read_json(path)
        status = "AUDITED_WITH_PROTOCOL_DEVIATION" if name == "e468_audit" else "COMPLETE_AUDITED"
        h.require(manifest["status"] == status, f"{name} historical status differs")
        if name == "e468_audit":
            continue
        published_results = manifest.get("result_files", {})
        if name == "base":
            published_results = {f"results/{key}": val for key, val in manifest["result_sha256"].items()}
        for relative, digest in published_results.items():
            artifact = SOURCES[name] / relative
            guards[str(artifact)] = verified(artifact, digest)
        if name in ("base", "e463", "e464", "e465", "e466", "e467"):
            audit = h.read_json(SOURCES[name] / "results/audit_status.json")
            h.require(audit["status"] == "COMPLETE_AUDITED", f"{name} archived result audit differs")
        for relative, digest in manifest.get("implementation", {}).items():
            # E466 records a later, audit-only index repair separate from its training freeze.
            repair = manifest.get("audit_entry_repairs", {}).get("independent_mask_index", {})
            if repair.get("path") == relative:
                h.require(repair["not_training_code"], "Historical training repair is unregistered")
                digest = repair["sha256"]
            artifact = SOURCES[name] / relative
            guards[str(artifact)] = verified(artifact, digest)

    base = manifests["base"]
    relevant_files = {
        path: digest for path, digest in base["protected_sha256"].items()
        if (Path(path).suffix in {".py", ".npy"} or path.endswith("processed_standard_data.csv")
            or Path(path).name == "config.json" or path.endswith("/METADATA")
            or any(fold_tag in Path(path).name for fold_tag in ("fold02", "_f02", "fold08", "_f08"))
            or "text" in Path(path).name.lower())
    }
    for path, digest in relevant_files.items():
        guards[path] = verified(Path(path), digest)
    # The finalized E61 manifest binds its readback entry separately from the original training entry.
    for relative, digest in manifests["e61"]["audit_source_sha256"].items():
        artifact = SOURCES["e61"] / relative
        h.require(guards[str(artifact)] == digest, "E61 and base source digests disagree")
        guards[str(artifact)] = verified(artifact, digest)
    for fold in (2, FOLD):
        warmup = next(item for item in manifests["e61"]["warmups"] if item["fold"] == fold)
        guards[warmup["path"]] = verified(Path(warmup["path"]), warmup["sha256"])
    for fold in FOLDS:
        artifact = parent.paths(MODEL_VARIANT, fold)["preflight"]
        guards[str(artifact)] = verified(artifact, base["preflight"]["sha256"][str(artifact)])
    return manifests, guards


def verify_checkpoints(manifests, guards):
    e6, e7, e5 = (manifests[k] for k in ("e466", "e467", "e465"))
    h.require(e6["actual_new_folds"] == [4] and e7["actual_new_folds"] == [2],
              "Direct historical training lineage differs")
    h.require(e7["reuse"]["direct_reference"]["source_manifest_sha"] == CONFIG["source_manifests"]["e466"][1],
              "E467 direct source is not E466")
    refs = {}
    for fold in FOLDS:
        chain = [E6]
        if fold == 4:
            source_name, stem, expected = "e466", "G1_f04", e6["fold4_result"]
            task_rel = f"runtime/{stem}.training.json"
            journal_rel = f"runtime/{stem}.jsonl"
            h.require(e6["completed_task"]["best_sha"] == expected["best_sha"],
                      "Fold 4 source training certificate differs")
        else:
            expected = e5["checkpoint_sources"][str(fold)]
            source_name = {2: "e464", 3: "e463"}.get(fold, "base")
            stem = f"{'G1' if fold == 2 else 'G2'}_f{fold:02d}"
            chain.extend(["E82.4.6.5-G1", {"base": "E82.4.6-G2", "e464": "E82.4.6.4-G1",
                                            "e463": "E82.4.6.3 DARG-only"}[source_name]])
            task_rel = f"runtime/tasks/{stem}.json" if source_name == "base" else f"runtime/{stem}.training.json"
            journal_rel = f"runtime/journal/{stem}.jsonl" if source_name == "base" else f"runtime/{stem}.jsonl"
            h.require(expected["source_manifest"] == str(SOURCES[source_name] / "manifest.json")
                      and expected["source_manifest_sha"] == CONFIG["source_manifests"][source_name][1]
                      and expected["source_implementation_sha"] == manifests[source_name]["implementation_sha"],
                      f"Fold {fold} underlying manifest differs")
            if fold in (2, 3):
                reuse = e6["reuse"][str(fold)]
                h.require(reuse["source_directory"] == str(SOURCES["e465"])
                          and reuse["source_manifest_sha"] == CONFIG["source_manifests"]["e465"][1]
                          and reuse["underlying_source_directory"] == str(SOURCES[source_name])
                          and reuse["underlying_source_manifest_sha"] == expected["source_manifest_sha"]
                          and reuse["source_task"] == stem
                          and all(reuse[k + "_sha"] == expected[k + "_sha"] for k in ("best", "latest")),
                          f"E466 Fold {fold} reuse chain differs")
                audit_file = SOURCES[source_name] / "results/audit_status.json"
                h.require(guards[str(audit_file)] == reuse["source_audit_sha"], "Underlying audit digest differs")
        directory = SOURCES[source_name]
        task_file = directory / task_rel
        task = h.read_json(task_file)
        owner = manifests[source_name]
        if source_name == "base":
            recorded = next(t for t in owner["completed_folds"]["G2"] if t["fold"] == fold)
        elif source_name == "e464":
            recorded = owner["completed_tasks"][stem]
        elif source_name == "e463":
            recorded = owner["completed_folds"][str(fold)]
        else:
            recorded = owner["completed_task"]
        files = {"best": directory / f"checkpoints/{stem}.best.pt",
                 "latest": directory / f"checkpoints/{stem}.latest.pt", "journal": directory / journal_rel}
        h.require(task["status"] == "TRAINED" and task["fold"] == fold and
                  all(task[k + "_sha"] == expected[k + "_sha"] == recorded[k + "_sha"]
                      and str(files[k]) == expected.get(k, str(files[k])) for k in IDENTITY_FILES)
                  and task["best_epoch"] == expected["best_epoch"]
                  and task["epoch"] == expected["latest_epoch"], f"Fold {fold} checkpoint/task identities differ")
        guards[str(task_file)] = h.sha_file(task_file)
        ref = {"lineage": chain, "source": source_name, "source_directory": str(directory),
               "source_manifest_sha256": CONFIG["source_manifests"][source_name][1],
               "task": stem, "task_path": str(task_file), "task_sha": guards[str(task_file)],
               "best_epoch": task["best_epoch"], "latest_epoch": task["epoch"]}
        for k, path in files.items():
            ref[k] = str(path)
            ref[k + "_sha"] = guards[str(path)] = verified(path, expected[k + "_sha"])
        refs[str(fold)] = ref

    # Bind the E467 Fold 2 checkpoint behind the direct-reference comparison as well.
    directory = SOURCES["e467"]
    official = e7["fold2_result"]
    for key, path in {"best": directory / "checkpoints/G1_f02.best.pt",
                      "latest": directory / "checkpoints/G1_f02.latest.pt",
                      "journal": directory / "runtime/G1_f02.jsonl"}.items():
        guards[str(path)] = verified(path, official[key + "_sha"])
    for name in ("e466", "e467"):
        for relative, digest in manifests[name]["runtime_files"].items():
            path = SOURCES[name] / relative
            guards[str(path)] = verified(path, digest)
    return refs


def verify_guard(guards):
    actual = h.archive.hash_files([Path(path) for path in guards])
    h.require(actual == guards, "Protected historical artifacts changed")
    return {"status": "PASS", "files": len(guards), "sha256": h.sha(guards)}


def archived():
    result = {}
    for key, method in (("e466", E6), ("e467", E7)):
        directory = SOURCES[key] / "results"
        rows = read_csv(directory / "patient_level.csv")
        ids = [int(r["patient_id_1based"]) for r in rows]
        h.require(len(ids) == 871 and set(ids) == set(range(1, 872))
                  and all(int(r["patient_index_0based"]) + 1 == int(r["patient_id_1based"]) for r in rows),
                  f"{method} archived patient order or identity is invalid")
        result[key] = {fold: [row for row in rows if int(row["fold"]) == fold] for fold in FOLDS}
    for fold in FOLDS:
        left, right = (result[key][fold] for key in ("e466", "e467"))
        h.require(len(left) == len(right) == (88 if fold == 1 else 87), "Historical fold size differs")
        fields = ("patient_id_1based", "patient_index_0based", "true_label")
        if fold != 2:
            fields += (*FIELDS, *CAM_FIELDS, "current_best_sha")
        h.require(all(all(a[k] == b[k] for k in fields) for a, b in zip(left, right)),
                  f"E466 and E467 differ outside their authorized Fold 2 replacement: Fold {fold}")
    return result


def score(labels, pred, p_asd):
    labels, pred, p_asd = np.asarray(labels), np.asarray(pred), np.asarray(p_asd, dtype=np.float64)
    h.require(len(labels) == len(pred) == len(p_asd) and np.isin(labels, [0, 1]).all()
              and np.isin(pred, [0, 1]).all() and np.isfinite(p_asd).all()
              and ((0 <= p_asd) & (p_asd <= 1)).all(), "Invalid patient labels, predictions or ASD probabilities")
    tp = int(((labels == 0) & (pred == 0)).sum())
    fp = int(((labels == 1) & (pred == 0)).sum())
    tn = int(((labels == 1) & (pred == 1)).sum())
    fn = int(((labels == 0) & (pred == 1)).sum())
    both = bool((tp + fn) and (tn + fp))
    return {"n_test": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "correct": tp + tn,
            "acc": float(np.mean(labels == pred)), "hard_auc": float(balanced_accuracy_score(labels, pred)) if both else None,
            "prob_auc": float(roc_auc_score(labels == 0, p_asd)) if both else None,
            "sen": float(recall_score(labels, pred, pos_label=0)) if tp + fn else None,
            "spe": float(recall_score(labels, pred, pos_label=1)) if tn + fp else None,
            "asd_f1": float(f1_score(labels, pred, pos_label=0, zero_division=0)) if tp + fp + fn else None}


def archived_metrics(rows, mode="full"):
    return score([int(row["true_label"]) for row in rows],
                 [int(row[f"prediction_{mode}"]) for row in rows], [float(row[f"p_asd_{mode}"]) for row in rows])


def verify_archived_metrics(result):
    for source, method in (("e466", E6), ("e467", E7)):
        archived_fold_path = SOURCES[source] / "results/fold_metrics.csv"
        fold_metrics = {int(row["fold"]): row for row in read_csv(archived_fold_path)
                        if row["method"] == method and row["scope"] == "test_fold"}
        h.require(set(fold_metrics) == set(FOLDS), f"{method} missing archived fold metrics")
        for fold, rows in result[source].items():
            actual = archived_metrics(rows)
            counts = ("n_test", "tp", "fp", "tn", "fn")
            h.require(all(actual[k] == int(fold_metrics[fold][k]) for k in counts)
                      and all(abs(actual[k] - float(fold_metrics[fold][k])) < 1e-12 for k in METRICS),
                      f"{method} Fold {fold} metrics do not reproduce")


def load_adapter(name, filename):
    return h.archive.load_module(f"_e82469_readonly_{name}", SOURCES[name] / filename)


def norms(value):
    return np.linalg.norm(value.detach().cpu().numpy().astype(np.float64), axis=-1)


def residual_records(captures, indices):
    full = captures["full"]
    details = full["details"]
    delta, cam = norms(details["delta"][indices]), norms(full["cam"][indices])
    h.require((cam > 0).all() and np.isfinite(delta).all(), "Nonfinite or empty CAM residual record")
    return {"delta_l2": delta, "delta_parallel_l2": norms(details["delta_parallel"][indices]),
            "delta_perp_l2": norms(details["delta_perp"][indices]), "cam_l2": cam,
            "cam_relative_delta": delta / cam,
            "query_feature_change_l2": norms(full["enhanced"][indices].double()
                                             - captures["anonymous"]["enhanced"][indices].double())}


def replay_fold(fold, ref, rows, manifests):
    source = ref["source"]
    if source == "base":
        run = parent
        task_group = MODEL_VARIANT
    elif source == "e463":
        run = load_adapter(source, "darg_only.py").engine(fold)
        task_group = MODEL_VARIANT
    else:
        run = load_adapter(source, "core.py").engine("G1", fold)
        task_group = "G1"
    proof = evaluator(run, f"history_{fold:02d}")
    ctx = run.setup(task_group, fold)
    checkpoint = h.load(Path(ref["best"]))
    run.restore(ctx, checkpoint, manifests[source]["implementation"])
    h.require(h.sha(run.bundle(ctx)) == checkpoint["state_sha"] and checkpoint["epoch"] == ref["best_epoch"],
              f"Historical Fold {fold} full restored checkpoint differs")
    members = ctx["members"]
    test = members["test"]
    labels = ctx["state"]["labels_np"][test]
    h.require(test.tolist() == [int(row["patient_index_0based"]) for row in rows]
              and labels.tolist() == [int(row["true_label"]) for row in rows]
              and all(row["current_best_sha"] == ref["best_sha"] for row in rows),
              f"Fold {fold} archive order, labels or Best SHA differs")
    predictions, captures = {}, {}
    rng = h.rng_sha()
    for mode, intervention in INTERVENTIONS.items():
        output, captures[mode] = proof.independent_eval(ctx, intervention)
        margins = proof.signed_margin(output["logits"], labels)
        actual = {"prediction": output["prediction"], "p_asd": output["probability"][:, 0],
                  "true_margin": margins}
        for field, values in actual.items():
            parse = int if field == "prediction" else float
            h.require(np.array_equal(values, [parse(row[f"{field}_{mode}"]) for row in rows]),
                      f"Fold {fold} archived {mode} {field} differs elementwise")
        predictions[mode] = {key: val.tolist() for key, val in actual.items()}
        predictions[mode]["probability"] = output["probability"].tolist()
        if mode == "full":
            proof.match_eval(output, checkpoint["evaluation"], f"Historical Fold {fold} Best")
    h.require(h.rng_sha() == rng, "Historical Best evaluation consumed RNG")

    overlap = np.intersect1d(members["train"], test)
    expected = {2: CONFIG["fold2_rollback"]["overlap_1based"], 3: [46, 66, 172, 266, 388],
                4: [712, 713, 727, 773]}.get(fold, [])
    allowed = ctx["module"].spectrum.allowed
    full = captures["full"]
    h.require(overlap.tolist() == sorted(i - 1 for i in expected)
              and len(members["train"]) == len(np.unique(members["train"])) == 871 - len(test) + len(expected)
              and len(members["legal"]) == 871 - len(test)
              and np.array_equal(np.setdiff1d(members["train"], test), np.sort(members["legal"]))
              and np.array_equal(ctx["reference"]["legal"], members["legal"])
              and np.array_equal(ctx["module"].statistics.legal.cpu().numpy(), members["legal"])
              and torch.where(~allowed)[0].tolist() == overlap.tolist()
              and torch.equal(full["enhanced"][overlap], full["cam"][overlap])
              and all(torch.count_nonzero(full["details"][key][overlap]).item() == 0
                      for key in ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")),
              f"Fold {fold} historical scope or residual mask differs")
    counts = np.bincount(members["train_pairs"].ravel(), minlength=871)
    scope = {
        "status": "PASS", "completed_at_utc": utc_now(), "fold": fold, "checkpoint": ref,
        "native_rows": len(members["train"]), "legal_rows": len(members["legal"]), "test_rows": len(test),
        "overlap_ids_1based": expected, "test_indices_0based": test.tolist(), "labels": labels.tolist(),
        "overlap_relation_occurrences": {str(i + 1): int(counts[i]) for i in overlap},
        "all_three_interventions_exact": True, "rng_unchanged": True,
        "metrics": archived_metrics(rows),
    }
    if fold == 2:
        row = CONFIG["fold2_rollback"]["withdrawn_1based"] - 1
        sizes = CONFIG["fold2_rollback"]
        h.require(len(members["train"]) == sizes["native"] and len(members["legal"]) == sizes["legal"]
                  and len(test) == sizes["test"] and row in test and row not in members["train"]
                  and not counts[row] and allowed[row].item()
                  and torch.count_nonzero(full["details"]["delta"][row]).item() > 0,
                  "Patient 540 withdrawal did not restore normal inference")
        scope.update(patient_540_darg_classification=False, patient_540_relation_occurrences=0,
                     patient_540_residual_allowed=True,
                     patient_540_joint_residual_l2=float(full["details"]["delta"][row].double().norm()),
                     predictions=predictions)
    if fold == FOLD:
        h.require(len(members["train"]) == 784 and len(test) == 87 and not len(overlap),
                  "Original Fold 8 checkpoint training membership differs")
        h.require(scope["metrics"]["tp"] == 34 and scope["metrics"]["tn"] + scope["metrics"]["fp"] == 47,
                  "Archived Fold 8 reference cohort differs")
    print(f"Historical Fold {fold:02d} Best {checkpoint['epoch']}: {scope['metrics']['correct']}/{len(test)}, exact replay", flush=True)
    return scope


def replay_archives(manifests, refs):
    result = archived()
    verify_archived_metrics(result)
    # Store the Fold 2 certificate immediately, before any workers can be launched.
    order = (2, FOLD, *(f for f in FOLDS if f not in (2, FOLD)))
    evaluations = {}
    for fold in order:
        verification = replay_fold(fold, refs[str(fold)], result["e466"][fold], manifests)
        if fold == 2:
            h.write_json(RUNTIME / "fold2_rollback.json", verification)
            verification = {key: val for key, val in verification.items() if key != "predictions"}
        evaluations[str(fold)] = verification
    return evaluations
