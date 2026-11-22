"""Bind the audited historical sources and independently replay the reused folds."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from core import CONFIG, FOLD, MODEL_VARIANT, SOURCES, evaluator, h, parent

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

E6 = "E82.4.6.6-G1"
E9 = CONFIG["direct_reference"]
DARG = "E74.4-G2 DARG"
SINGLE_GROUPS = ("G2", "G4", "G6")
SINGLE_METHODS = {group: f"E82.4.6.9-{group}" for group in SINGLE_GROUPS}
INTERVENTIONS = {"full": "full", "residual_off": "residual_off", "anonymous": "anonymous_query"}
FIELDS = tuple(f"{field}_{mode}" for mode in INTERVENTIONS for field in ("prediction", "p_asd", "true_margin"))
CAM_FIELDS = ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "cam_l2",
              "cam_relative_delta", "query_feature_change_l2")
IDENTITY = ("patient_id_1based", "patient_index_0based", "true_label")
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
COUNTS = ("n_test", "tp", "fp", "tn", "fn", "correct")
FOLDS = tuple(range(1, 11))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_guard(guards):
    actual = h.archive.hash_files([Path(path) for path in guards])
    changed = [path for path, digest in guards.items() if actual[path] != digest]
    h.require(not changed, f"Protected historical artifacts differ: {changed}")
    return {"status": "PASS", "files": len(guards), "sha256": h.sha(guards)}


def verify_sources():
    manifests = {}
    for name, (directory, expected) in CONFIG["source_manifests"].items():
        path = Path(directory) / "manifest.json"
        h.require(h.sha_file(path) == expected, f"Historical Manifest SHA differs: {path}")
        doc = manifests[name] = h.read_json(path)
        status = "AUDITED_WITH_PROTOCOL_DEVIATION" if name == "e468_audit" else "COMPLETE_AUDITED"
        h.require(doc["status"] == status, f"{name} historical status differs")
    direct = manifests["e469"]
    guards = dict(direct["protected_sha256"])

    def bind(path, expected):
        path = Path(path)
        h.require(guards.get(str(path), expected) == expected, f"Conflicting source SHA: {path}")
        guards[str(path)] = expected

    directory = SOURCES["e469"]
    bind(directory / "manifest.json", CONFIG["source_manifests"]["e469"][1])
    h.require(h.sha(direct["implementation"]) == direct["implementation_sha"]
              and direct["actual_new_folds"] == [FOLD]
              and direct["reused_folds"] == [f for f in FOLDS if f != FOLD],
              "E82.4.6.9 implementation or Fold 8/reuse registration differs")
    for relative, digest in (direct["implementation"] | direct["result_files"] | direct["published_source_sha256"]).items():
        bind(directory / relative, digest)
    for key, sha in (("path", "sha256"), ("fold2_rollback_path", "fold2_rollback_sha256")):
        bind(direct["historical_replay"][key], direct["historical_replay"][sha])
    audit_status = h.read_json(directory / "results/audit_status.json")
    membership_status = h.read_json(directory / "results/membership_audit.json")
    h.require(audit_status["status"] == "COMPLETE_AUDITED" and membership_status["status"] == "PASS",
              "Direct experiment audit status differs")

    refs = {}
    for group in SINGLE_GROUPS:
        item = direct["groups"][group]
        stem = f"{group}_f{FOLD:02d}"
        task = item["training"]
        files = {"best": directory / f"checkpoints/{stem}.best.pt",
                 "latest": directory / f"checkpoints/{stem}.latest.pt",
                 "journal": directory / f"runtime/{stem}.jsonl",
                 "task": directory / f"runtime/{stem}.training.json",
                 "preflight": directory / f"runtime/{stem}.preflight.json",
                 "audit": directory / f"runtime/{stem}.audit.json",
                 "log": directory / f"runtime/{stem}.log"}
        pre = direct["preflight"]["groups"][group]
        audit = h.read_json(files["audit"])
        h.require(item["status"] == item["audit"]["status"] == audit["status"] == "COMPLETE_AUDITED"
                  and audit_status["groups"][group]["sha256"] == item["audit"]["sha256"]
                  and membership_status["groups"][group]["membership"] == audit["membership"]
                  and task["status"] == "TRAINED" and h.read_json(files["task"]) == task
                  and task["task_group"] == group and task["fold"] == FOLD
                  and task["implementation"] == direct["implementation"], f"{group} direct source identity differs")
        for key in ("best", "latest", "journal"):
            h.require(task[key + "_sha"] == audit["artifact_sha256"][key], f"{group} {key} SHA records differ")
            bind(files[key], task[key + "_sha"])
        for key in ("task", "preflight"):
            bind(files[key], audit["artifact_sha256"][key])
        h.require(str(files["audit"]) == item["audit"]["path"] and str(files["preflight"]) == pre["path"]
                  and pre["sha256"] == audit["artifact_sha256"]["preflight"]
                  and audit["best_epoch"] == task["best_epoch"] and audit["latest_epoch"] == task["epoch"],
                  f"{group} prior audit or epoch differs")
        bind(files["audit"], item["audit"]["sha256"])
        # The durable epoch journal is Manifest-bound. The console log is bound here
        # after checking its terminal record against the same certified task.
        terminal = f"TRAINED G2 fold 08 end {task['epoch']} best {task['best_correct']}@{task['best_epoch']}"
        log = files["log"].read_bytes()
        h.require(terminal.encode() in log.splitlines() and log.splitlines()[-1].endswith(
            f"{group} COMPLETE_AUDITED".encode()), f"{group} prior console log differs")
        bind(files["log"], hashlib.sha256(log).hexdigest())
        refs[group] = {"lineage": [SINGLE_METHODS[group]], "source": "e469", "task": stem,
                       "best_epoch": task["best_epoch"], "latest_epoch": task["epoch"],
                       **{key: str(path) for key, path in files.items()},
                       **{f"{key}_sha": guards[str(path)] for key, path in files.items()}}

    base_guards = manifests["base"]["protected_sha256"]
    for fold in FOLDS:
        path, csv_path, _, _ = h.reference_paths(fold)["DARG"]
        bind(path, base_guards[str(path)])
        bind(csv_path, base_guards[str(csv_path)])
    return manifests, guards, refs


def score(labels, pred, p_asd):
    labels, pred, p_asd = np.asarray(labels), np.asarray(pred), np.asarray(p_asd, dtype=np.float64)
    h.require(len(labels) == len(pred) == len(p_asd) and np.isin(labels, [0, 1]).all()
              and np.isin(pred, [0, 1]).all() and np.isfinite(p_asd).all()
              and ((0 <= p_asd) & (p_asd <= 1)).all(), "Invalid patient predictions or probabilities")
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


def values(rows, mode):
    return ([int(row["true_label"]) for row in rows],
            [int(row[f"prediction_{mode}"]) for row in rows], [float(row[f"p_asd_{mode}"]) for row in rows])


def verify_metrics(actual, archived, description):
    h.require(all(abs(actual[k] - float(archived[k])) < 1e-12 for k in METRICS)
              and all(actual[k] == int(archived[k]) for k in COUNTS if archived.get(k) not in ("", None)),
              f"{description} archived metrics do not reproduce")


def historical_rows():
    e6_dir, e9_dir = (SOURCES[key] / "results" for key in ("e466", "e469"))
    e6 = read_csv(e6_dir / "patient_level.csv")
    e9 = read_csv(e9_dir / "patient_level.csv")
    darg = [{**row, **{f"{key}_full": row[f"{key}_darg"] for key in ("prediction", "p_asd", "true_margin")}}
            for row in e6]
    methods = {DARG: darg, E6: e6}
    methods.update({method: [r for r in e9 if r["group"] == group] for group, method in SINGLE_METHODS.items()})
    all_folds = {method: {f: [r for r in rows if int(r["fold"]) == f] for f in FOLDS}
                 for method, rows in methods.items()}
    for method, folds in all_folds.items():
        ids = [int(r["patient_id_1based"]) for r in methods[method]]
        h.require(len(ids) == len(set(ids)) == 871 and set(ids) == set(range(1, 872))
                  and all(int(r["patient_index_0based"]) + 1 == int(r["patient_id_1based"]) for r in methods[method]),
                  f"{method} row count or patient ID differs")
        for fold, rows in folds.items():
            reference = all_folds[E6][fold]
            h.require(len(rows) == len(reference) == (88 if fold == 1 else 87)
                      and all(all(r[k] == s[k] for k in IDENTITY) for r, s in zip(rows, reference)),
                      f"{method} Fold {fold} patient order or labels differ")
            if method in SINGLE_METHODS.values() and fold != FOLD:
                h.require(all(all(r[k] == s[k] for k in (*FIELDS, *CAM_FIELDS))
                              and r["best_sha256"] == s["current_best_sha"] for r, s in zip(rows, reference)),
                          f"{method} reused Fold {fold} patient projection differs from E82.4.6.6")

    e6_metrics = read_csv(e6_dir / "fold_metrics.csv")
    e9_metrics = read_csv(e9_dir / "fold_metrics.csv")
    for method, folds in all_folds.items():
        records = e9_metrics if method in SINGLE_METHODS.values() else e6_metrics
        archived = {int(r["fold"]): r for r in records
                    if r["method"] == method and r["scope"] == "test_fold" and r.get("intervention", "full") == "full"}
        h.require(set(archived) == set(FOLDS), f"{method} lacks ten-fold metrics")
        for fold, rows in folds.items():
            verify_metrics(score(*values(rows, "full")), archived[fold], f"{method} Fold {fold}")
    return all_folds


def load_adapter(name, filename):
    # The historical path loader preserves the selected device and old code identities.
    return h.archive.load_module(f"_e824610_readonly_{name}", SOURCES[name] / filename)


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


def replay_best(run, ctx, proof, rows, ref, implementation, expected_overlap, best_key):
    best = h.load(Path(ref["best"]))
    run.restore(ctx, best, implementation)
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"] and best["epoch"] == ref["best_epoch"],
              "Historical Best state differs")
    members = ctx["members"]
    test = members["test"]
    labels = ctx["state"]["labels_np"][test]
    overlap = np.intersect1d(members["train"], test)
    h.require(test.tolist() == [int(r["patient_index_0based"]) for r in rows]
              and labels.tolist() == [int(r["true_label"]) for r in rows]
              and all(r[best_key] == ref["best_sha"] for r in rows)
              and overlap.tolist() == sorted(i - 1 for i in expected_overlap)
              and len(members["train"]) == len(np.unique(members["train"])) == 871 - len(test) + len(overlap)
              and np.array_equal(np.setdiff1d(members["train"], test), np.sort(members["legal"]))
              and np.array_equal(ctx["reference"]["legal"], members["legal"])
              and np.array_equal(ctx["module"].statistics.legal.cpu().numpy(), members["legal"])
              and torch.where(~ctx["module"].spectrum.allowed)[0].tolist() == overlap.tolist(),
              "Historical membership, Best SHA, or labels differ")
    rng = h.rng_sha()
    captures = {}
    for mode, intervention in INTERVENTIONS.items():
        output, cap = proof.independent_eval(ctx, intervention)
        captures[mode] = cap
        expected_y, expected_pred, expected_prob = values(rows, mode)
        margins = proof.signed_margin(output["logits"], labels)
        h.require(np.array_equal(labels, expected_y) and np.array_equal(output["prediction"], expected_pred)
                  and np.array_equal(output["probability"][:, 0], expected_prob)
                  and np.array_equal(margins, [float(r[f"true_margin_{mode}"]) for r in rows])
                  and torch.equal(cap["enhanced"][overlap], cap["cam"][overlap])
                  and all(not torch.count_nonzero(cap["details"][k][overlap]).item()
                          for k in ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")),
                  f"Historical {mode} prediction, probability, margin or mask differs")
        if mode == "full":
            proof.match_eval(output, best["evaluation"], "Historical Best exact replay")
    residual = residual_records(captures, test)
    h.require(h.rng_sha() == rng
              and all(np.array_equal(residual[k], [float(r[k]) for r in rows]) for k in CAM_FIELDS),
              "Historical Best residual or RNG differs")
    return {"status": "PASS", "checkpoint": ref, "native_rows": len(members["train"]), "legal_rows": len(members["legal"]),
            "test_rows": len(test), "overlap_ids_1based": expected_overlap,
            "all_three_interventions_exact": True, "metrics": score(*values(rows, "full"))}


def replay_single(group, refs, archives, manifest, adapter):
    method = SINGLE_METHODS[group]
    ref = refs[group]
    run = adapter.engine(group)
    proof = evaluator(run, f"historical_single_{group}")
    ctx = run.setup(MODEL_VARIANT, FOLD)
    rows = archives[method][FOLD]
    target = manifest["groups"][group]["patient_id_1based"]
    result = replay_best(run, ctx, proof, rows, ref, manifest["implementation"], [target], "best_sha256")
    members, state = ctx["members"], ctx["state"]
    recorded = h.read_json(Path(ref["audit"]))
    preflight = h.read_json(Path(ref["preflight"]))
    h.require(result["native_rows"] == 785 and result["legal_rows"] == 784 and result["test_rows"] == 87
              and members["train"][-1] == target - 1
              and np.array_equal(members["train"][:-1], ctx["warmup"]["train_indices"])
              and recorded["membership"] == preflight["membership"]
              and recorded["best_metrics"]["full"] == result["metrics"]
              and recorded["membership"]["target_pair_endpoint_occurrences"] == int((state["train_pairs_np"] == target - 1).sum()),
              f"{method} archived membership certificate differs")
    latest = h.load(Path(ref["latest"]))
    run.restore(ctx, latest, manifest["implementation"])
    h.require(latest["best_sha"] == ref["best_sha"] and latest["journal_sha"] == ref["journal_sha"]
              and latest["epoch"] == ref["latest_epoch"] and h.sha(run.bundle(ctx)) == latest["state_sha"],
              f"{method} Latest relationship or state differs")
    current_ema = run.ema(ctx)
    run.restore_ema(ctx, latest["eval_ema"])
    output, cap = proof.independent_eval(ctx)
    adapter.check_mask(cap, ctx["module"], target - 1)
    proof.match_eval(output, latest["evaluation"], f"{method} Latest")
    run.restore_ema(ctx, current_ema)
    reference = archives[E6][FOLD]
    flips = {int(r["patient_id_1based"]): int(r["prediction_full"])
             for r, prior in zip(rows, reference) if r["prediction_full"] != prior["prediction_full"]}
    expected = {"G2": ({383: 0, 601: 0}, 77), "G4": ({430: 0}, 78), "G6": ({723: 0}, 78)}[group]
    h.require((flips, result["metrics"]["correct"]) == expected, f"{method} historical case behavior differs")
    print(f"{method} Fold 8 exact Best and Latest replay, {result['metrics']['correct']}/87, flips {flips}", flush=True)
    return {**result, "latest_exact": True, "hard_flips_vs_e466": flips,
            "target_pair_occurrences": recorded["membership"]["target_pair_endpoint_occurrences"]}


@torch.no_grad()
def replay_darg(fold, rows, stats):
    path, _, historical_group, _ = h.reference_paths(fold)["DARG"]
    saved = h.load(path)
    state, _, identity, _, members = h.warmup_and_membership(fold)
    test = members["test"]
    y, pred, prob = values(rows, "full")
    h.require(saved["fold"] == fold and saved["group"] == historical_group and saved["identity"] == identity
              and saved["best_epoch_1based"] == int(stats["best_epoch_1based"])
              and saved["epochs_executed"] == int(stats["epochs_executed"])
              and test.tolist() == [int(row["patient_index_0based"]) for row in rows]
              and np.array_equal(state["labels_np"][test], y)
              and np.array_equal(saved["test_prediction"], pred)
              and np.array_equal(saved["test_probability"][:, 0], prob)
              and h.sha_file(path) == stats["checkpoint_sha256"],
              f"Historical E74.4-G2 Fold {fold} Best rows or identity differ")
    h.restore_base(state, saved["best_state"])
    rng = h.rng_sha()
    captures = []
    hook = state["cam"].Outputlayer.classifier.register_forward_hook(lambda _, args, out: captures.append(out))
    try:
        result = h.runner().forward(state, h.BASE, training=False)
    finally:
        hook.remove()
    actual = result["logp"][state["test"]].exp().cpu().numpy()
    logits = captures[0][state["test"]].cpu().numpy().astype(np.float64)
    margins = (logits[:, 0] - logits[:, 1]) * np.where(np.asarray(y) == 0, 1, -1)
    h.require(len(captures) == 1 and h.rng_sha() == rng
              and np.array_equal(actual, saved["test_probability"]) and np.array_equal(actual.argmax(-1), pred)
              and np.array_equal(margins, [float(r["true_margin_full"]) for r in rows]),
              f"Historical E74.4-G2 Fold {fold} independent replay differs")
    verify_metrics(score(y, pred, prob), stats, f"Historical DARG Fold {fold}")
    return {"best": str(path), "best_sha": stats["checkpoint_sha256"],
            "best_epoch": int(stats["best_epoch_1based"]), "latest_epoch": int(stats["epochs_executed"]),
            "lineage": [DARG], "exact_replay": True}


def replay_archives(manifests, refs):
    archives = historical_rows()
    direct = manifests["e469"]
    old_adapter = load_adapter("e469", "core.py")
    single = {group: replay_single(group, refs, archives, direct, old_adapter) for group in SINGLE_GROUPS}
    reused, darg_refs = {}, {}
    darg_path = h.reference_paths(1)["DARG"][1]
    darg_group = h.reference_paths(1)["DARG"][2]
    darg_metrics = {int(r["fold"]): r for r in read_csv(darg_path) if r["group"] == darg_group}
    for fold in FOLDS:
        ref = direct["historical_checkpoint_sources"][str(fold)]
        if fold != FOLD:
            previous = direct["nine_fold_reuse"][str(fold)]
            h.require(previous["checkpoint"] == ref and previous["bitwise_values_e466"]
                      and previous["identical_all_six_groups"], f"Direct Fold {fold} reuse certificate differs")
        source = ref["source"]
        if source == "base":
            run, group = parent, MODEL_VARIANT
        elif source == "e463":
            run, group = load_adapter(source, "darg_only.py").engine(fold), MODEL_VARIANT
        else:
            run, group = load_adapter(source, "core.py").engine("G1", fold), "G1"
        ctx = run.setup(group, fold)
        proof = evaluator(run, f"historical_fold_{fold:02d}")
        overlap = {2: [377, 407, 681, 779, 784, 260, 288, 375], 3: [46, 66, 172, 266, 388],
                   4: [712, 713, 727, 773]}.get(fold, [])
        rows, best_key = (archives[E6][fold], "current_best_sha") if fold == FOLD else (archives[E9][fold], "best_sha256")
        result = replay_best(run, ctx, proof, rows, ref, manifests[source]["implementation"], overlap, best_key)
        reused[str(fold)] = result
        if fold != FOLD:
            h.require(previous["test_rows"] == result["test_rows"], f"Direct Fold {fold} reuse count differs")
        print(f"{E6 if fold == FOLD else E9} Fold {fold:02d} Best {ref['best_epoch']}: "
              f"{result['metrics']['correct']}/{result['test_rows']}, exact replay", flush=True)

        darg_refs[str(fold)] = replay_darg(fold, archives[DARG][fold], darg_metrics[fold])
    return {"status": "PASS", "completed_at_utc": utc_now(), "direct_reference": E9, "folds": reused,
            "single_patient_reference": single, "darg": darg_refs,
            "reused_rows": sum(r["test_rows"] for fold, r in reused.items() if int(fold) != FOLD)}
