#!/usr/bin/env python3
"""Two additional frozen-feature RBF readouts on the audited E76.2 splits."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
E762 = Path(CFG["e762"]["path"])
E762_CACHE = E762 / CFG["e762"]["prediction_cache"]
BASE = Path(CFG["base"]["path"])
RESULTS = EXP / "results"
RUNTIME = EXP / ".runtime"
CACHE = RUNTIME / "cache"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
GROUP_B0 = "B0_NL_CAM_only_RBF_SVM_Historical"
GROUP_B4 = "B4_CAM_Reliable_RBF_SVM"
GROUP_B5 = "B5_CAM_Reliable_Hard_Paired_RBF_SVM"
E762_BASELINE = "B0_NL_CAM_only_RBF_SVM"
P = None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def probe_module():
    global P
    if P is None:
        spec = importlib.util.spec_from_file_location("e762_locked_probe", E762 / "probe.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        P = module
    return P


def inventory():
    p = probe_module()
    base = Path(CFG["base"]["path"])
    warmup = Path(p.CFG["base"]["warmup_path"])
    paths = {
        "config": EXP / "config.json", "run": EXP / "run.py", "audit": EXP / "audit.py",
        "e762_probe": E762 / "probe.py", "e762_audit": E762 / "audit.py",
        "e762_config": E762 / "config.json", "e762_manifest": E762 / "manifest.json",
        "e762_report": E762 / "results/report.md", "e762_fold_metrics": E762 / "results/fold_metrics.csv",
        "e762_audit_status": E762 / "results/audit_status.json", "e762_cache": E762_CACHE,
        "e744_run": base / "run.py", "e744_model": base / "soft_positive_dcr.py",
        "e744_config": base / "config.json", "e744_manifest": base / "manifest.json",
        "e744_audit_status": base / "results/audit_status.json",
        "processed": Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        "mapping": Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
    }
    paths.update({f"best_{fold}": base / f".runtime/checkpoints/{p.BASE_GROUP}_f{fold:02d}.best.pt" for fold in range(1, 11)})
    paths.update({f"latest_{fold}": base / f".runtime/checkpoints/{p.BASE_GROUP}_f{fold:02d}.latest.pt" for fold in range(1, 11)})
    paths.update({f"task_{fold}": base / f".runtime/tasks/{p.BASE_GROUP}_f{fold:02d}.json" for fold in range(1, 11)})
    paths.update({f"warmup_{fold}": warmup / f"fold{fold:02d}.pt" for fold in range(1, 11)})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing locked source files: " + ", ".join(missing))
    return {key: sha_file(value) for key, value in paths.items()}


def verify_references():
    p = probe_module()
    e762_manifest = json.loads((E762 / "manifest.json").read_text())
    e762_audit = json.loads((E762 / "results/audit_status.json").read_text())
    e74_manifest = json.loads((BASE / "manifest.json").read_text())
    e74_audit = json.loads((BASE / "results/audit_status.json").read_text())
    if e762_manifest.get("status") != "COMPLETE_AUDITED" or e762_audit.get("status") != "PASS":
        raise RuntimeError("E76.2 reference audit is not PASS")
    if len(e762_audit.get("metrics_independently_recomputed", [])) != 50:
        raise RuntimeError("E76.2 independent metric audit does not contain 50 conditions")
    if e74_manifest.get("status") != "COMPLETE_AUDITED" or e74_audit.get("status") != "PASS":
        raise RuntimeError("E74.4 reference audit is not PASS")
    out = subprocess.run(["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git",
                          f"refs/heads/{CFG['e762']['branch']}"], check=True,
                         capture_output=True, text=True, timeout=30)
    rows = [line.split() for line in out.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != CFG["e762"]["commit"]:
        raise RuntimeError(f"E76.2 branch commit mismatch: {rows}")
    if p.sha_file(E762_CACHE) != e762_audit.get("prediction_cache_sha256"):
        raise RuntimeError("E76.2 prediction cache SHA mismatch")
    return {"e74_checkpoint_audit": e74_audit.get("status"), "e762_audit": e762_audit.get("status"),
            "e762_commit": rows[0][0], "e762_cache_sha256": p.sha_file(E762_CACHE)}


def min_extract_fold(fold):
    """One frozen best-checkpoint forward, capturing the unchanged classifier input."""
    import numpy as np
    import torch
    p = probe_module()
    runner = p.runner()
    base = runner.load_base()
    state, _warmup_state, identity, _reference = runner.prepare_fold(fold)
    checkpoint = BASE / f".runtime/checkpoints/{p.BASE_GROUP}_f{fold:02d}.best.pt"
    task_path = BASE / f".runtime/tasks/{p.BASE_GROUP}_f{fold:02d}.json"
    ckpt = p.torch_load(checkpoint)
    task = json.loads(task_path.read_text())
    if (ckpt.get("group"), ckpt.get("fold")) != (p.BASE_GROUP, fold):
        raise RuntimeError(f"fold {fold}: checkpoint group/fold mismatch")
    if task.get("checkpoint_sha256") != p.sha_file(checkpoint) or task.get("identity") != identity or ckpt.get("identity") != identity:
        raise RuntimeError(f"fold {fold}: checkpoint identity mismatch")
    if identity["membership"]["train_test_overlap_1based"] != ([377, 407] if fold == 2 else []):
        raise RuntimeError(f"fold {fold}: historical overlap mismatch")
    history = base.load_history()
    history.load_bundle(state, ckpt["best_state"])
    state["model"].eval(); state["cam"].eval()
    weights_before = p.model_weights_sha({"model": state["model"].state_dict(), "cam": state["cam"].state_dict()})
    captured = {"calls": 0}
    output_layer = state["cam"].Outputlayer
    def capture(_module, inputs):
        captured["calls"] += 1
        if len(inputs) != 2 or tuple(inputs[0].shape) != (871, 144):
            raise RuntimeError("expected OutputLayer inputs [871,144] and attention")
        captured["h"] = inputs[0].detach().float().clone()
        captured["attn"] = inputs[1].detach().float().clone()
    hook = output_layer.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            out = base.forward(state, p.BASE_GROUP, training=False)
    finally:
        hook.remove()
    if captured["calls"] != 1:
        raise RuntimeError("OutputLayer must execute exactly once")
    weights_after = p.model_weights_sha({"model": state["model"].state_dict(), "cam": state["cam"].state_dict()})
    if weights_after != weights_before:
        raise RuntimeError(f"fold {fold}: frozen forward changed model/CAM parameters")
    logits = out["logp"].detach().float().cpu().numpy()
    prob = torch.softmax(torch.as_tensor(logits), dim=-1).numpy().astype(np.float64)
    pred = prob.argmax(axis=1).astype(int)
    train = runner.index_numpy(state["train"]); test = runner.index_numpy(state["test"])
    eligible = np.asarray([int(i) for i in train if int(i) not in set(test.tolist())], dtype=int)
    if np.intersect1d(eligible, test).size or (fold == 2 and ({376, 406} & set(eligible.tolist()))):
        raise RuntimeError("test patient entered eligible diagnostic support")
    old = np.load(E762_CACHE, allow_pickle=False)
    prefix = f"f{fold}_"
    old_ids = np.asarray(old[prefix + "eligible"], dtype=int)
    if not np.array_equal(old_ids, eligible):
        raise RuntimeError(f"fold {fold}: cached E76.2 eligible IDs mismatch")
    old_y = np.asarray(old[prefix + "labels"], dtype=int)
    y = state["y"].detach().cpu().numpy().astype(int)
    if not np.array_equal(y[eligible], old_y):
        raise RuntimeError(f"fold {fold}: cached E76.2 labels do not align with current data")
    cached_base_prob = np.asarray(old[prefix + "base_prob"], dtype=float)
    cached_base_pred = np.asarray(old[prefix + "base_pred"], dtype=int)
    if not np.array_equal(pred[eligible], cached_base_pred) or np.max(np.abs(prob[eligible] - cached_base_prob)) > 1e-6:
        raise RuntimeError(f"fold {fold}: current frozen base predictions differ from E76.2 stability cache")
    if fold == 2 and not ({376, 406}.isdisjoint(set(eligible.tolist()))):
        raise RuntimeError("fold 2 exposed rows were not excluded")
    b0_key = prefix + E762_BASELINE
    baseline_ids = np.asarray(old[b0_key + "_ids"], dtype=int)
    if not np.array_equal(baseline_ids, eligible):
        raise RuntimeError(f"fold {fold}: historical B0-NL OOF IDs differ")
    p_cache = p.read_csv(E762 / "results/fold_metrics.csv")
    hist = next((row for row in p_cache if int(row["fold"]) == fold and row["group"] == E762_BASELINE), None)
    if hist is None:
        raise RuntimeError(f"fold {fold}: historical B0-NL fold metric missing")
    stability = np.zeros(871, dtype=float)
    stability[eligible] = np.asarray(old[prefix + "stability"], dtype=float)
    data = {"fold": fold, "h": captured["h"].cpu().numpy().astype(np.float64),
            "prob": prob, "pred": pred, "y": y, "eligible": eligible,
            "stability": stability,
            "base_pred": np.asarray(old[prefix + "base_pred"], dtype=int),
            "base_prob": np.asarray(old[prefix + "base_prob"], dtype=float),
            "test": test, "train": train,
            "identity": identity, "checkpoint_sha": p.sha_file(checkpoint),
            "checkpoint_weights_sha": weights_before, "post_forward_weights_sha": weights_after}
    if data["h"].shape != (871, 144):
        raise RuntimeError("CAM144 feature shape mismatch")
    # Verify the captured feature is classified exactly by the loaded original OutputLayer.
    with torch.no_grad():
        recovered = output_layer(captured["h"], captured["attn"])[0].detach().float().cpu().numpy()
    if not np.array_equal(recovered.argmax(1), pred) or np.max(np.abs(torch.softmax(torch.as_tensor(recovered), -1).numpy() - prob)) > 1e-7:
        raise RuntimeError(f"fold {fold}: captured CAM144 does not reproduce base classifier")
    data["historical_b0nl"] = {"pred": np.asarray(old[b0_key + "_pred"], dtype=int),
                               "score": np.asarray(old[b0_key + "_score"], dtype=float),
                               "ids": baseline_ids, "metric": hist}
    return data


def metric(y, pred, score):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int); score = np.asarray(score, dtype=float)
    return {"acc": float(accuracy_score(y, pred)),
            "hard_auc": float(balanced_accuracy_score(y, pred)),
            "prob_auc": float(roc_auc_score(y == 0, score)),
            "sen": float(np.sum((y == 0) & (pred == 0)) / max(1, np.sum(y == 0))),
            "spe": float(np.sum((y == 1) & (pred == 1)) / max(1, np.sum(y == 1))),
            "asd_f1": float(f1_score(y == 0, pred == 0, zero_division=0))}


def svc_fit_score(x_train, y_train, x_valid):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=False))
    model.fit(x_train, y_train)
    estimator = model.named_steps["svc"]
    classes = np.asarray(estimator.classes_)
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise RuntimeError(f"SVC class order must be [ASD=0, TD=1], got {classes.tolist()}")
    pred = model.predict(x_valid).astype(int)
    decision = np.asarray(model.decision_function(x_valid), dtype=float)
    score_asd = -decision  # positive raw decision is classes_[1] == TD.
    return pred, score_asd


def run_fold(fold):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    p = probe_module(); data = min_extract_fold(fold)
    ids = data["eligible"]; y = data["y"]; h = data["h"]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    old = np.load(E762_CACHE, allow_pickle=False)
    prefix = f"f{fold}_"
    hist_ids = np.asarray(old[prefix + E762_BASELINE + "_ids"], dtype=int)
    hist_pred = np.asarray(old[prefix + E762_BASELINE + "_pred"], dtype=int)
    hist_score = np.asarray(old[prefix + E762_BASELINE + "_score"], dtype=float)
    hist_map = {int(gid): (int(pr), float(sc)) for gid, pr, sc in zip(hist_ids, hist_pred, hist_score)}
    pred = {GROUP_B4: np.full(len(ids), -1, dtype=int), GROUP_B5: np.full(len(ids), -1, dtype=int)}
    scores = {GROUP_B4: np.full(len(ids), np.nan), GROUP_B5: np.full(len(ids), np.nan)}
    fold_records = []
    for inner, (train_pos, valid_pos) in enumerate(outer_cv.split(ids, y[ids]), start=1):
        train_ids = ids[train_pos]; valid_ids = ids[valid_pos]
        if np.intersect1d(train_ids, valid_ids).size or len(valid_ids) == 0:
            raise RuntimeError(f"fold {fold}/{inner}: inner split invalid")
        # Reproduce E76.2's 3-fold cross-fitted evidence features for B1 and B2.
        evidence_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=76300 + fold)
        train_z = np.zeros((len(train_ids), 12), dtype=float)
        train_z1 = np.zeros_like(train_z)
        for tr_pos, held_pos in evidence_cv.split(train_ids, y[train_ids]):
            memory = p.build_memory(data, train_ids[tr_pos])
            train_z[held_pos] = p.evidence(h[train_ids[held_pos]], memory, True)
            train_z1[held_pos] = p.evidence(h[train_ids[held_pos]], memory, False)
        valid_memory = p.build_memory(data, train_ids)
        valid_z = p.evidence(h[valid_ids], valid_memory, True)
        valid_z1 = p.evidence(h[valid_ids], valid_memory, False)
        # B4 is E76.2 B1's exact feature construction; B5 is E76.2 B2's.
        fold_predictions = {}
        for group, train_ev, valid_ev in ((GROUP_B4, train_z1, valid_z1), (GROUP_B5, train_z, valid_z)):
            xtr = np.c_[h[train_ids], train_ev]
            xva = np.c_[h[valid_ids], valid_ev]
            pr, sc = svc_fit_score(xtr, y[train_ids], xva)
            pred[group][valid_pos] = pr
            scores[group][valid_pos] = sc
            fold_predictions[group] = (pr, sc)
        if not np.all(np.isfinite(train_z)) or not np.all(np.isfinite(valid_z)):
            raise RuntimeError(f"fold {fold}/{inner}: non-finite prototype evidence")
        fold_records.append({"inner_fold": inner, "train_ids": train_ids.tolist(), "valid_ids": valid_ids.tolist(),
                             "train_ids_sha256": hashlib.sha256(train_ids.tobytes()).hexdigest(),
                             "valid_ids_sha256": hashlib.sha256(valid_ids.tobytes()).hexdigest(),
                             "train_evidence_sha256": hashlib.sha256(train_z.tobytes()).hexdigest(),
                             "train_reliable_evidence_sha256": hashlib.sha256(train_z1.tobytes()).hexdigest(),
                             "valid_evidence_sha256": hashlib.sha256(valid_z.tobytes()).hexdigest(),
                             "valid_reliable_evidence_sha256": hashlib.sha256(valid_z1.tobytes()).hexdigest(),
                             "input_b4_dim": int(xtr.shape[1] - 12 + 12), "input_b5_dim": int(xtr.shape[1])})
    if any(np.any(v < 0) if v.dtype.kind in "iu" else not np.all(np.isfinite(v)) for v in pred.values()):
        raise RuntimeError(f"fold {fold}: missing OOF predictions")
    base_pred = np.asarray([hist_map[int(gid)][0] for gid in ids], dtype=int)
    base_score = np.asarray([hist_map[int(gid)][1] for gid in ids], dtype=float)
    rows = []
    rows.append({"fold": fold, "group": GROUP_B0, "source": "E76.2 audited OOF cache", "n_oof": len(ids), **metric(y[ids], base_pred, base_score)})
    for group in (GROUP_B4, GROUP_B5):
        rows.append({"fold": fold, "group": group, "source": "E76.3 RBF-SVM", "n_oof": len(ids), **metric(y[ids], pred[group], scores[group])})
    output = {"fold": fold, "ids": ids, "labels": y[ids], "baseline_pred": base_pred, "baseline_score": base_score,
              "predictions": pred, "scores": scores, "rows": rows, "fold_records": fold_records,
              "data": data}
    return output


def paired_rows(fold_rows):
    import numpy as np
    from scipy.stats import t
    lookup = {(int(r["fold"]), r["group"]): r for r in fold_rows}
    comparisons = ((GROUP_B4, GROUP_B0), (GROUP_B5, GROUP_B0), (GROUP_B5, GROUP_B4))
    rows = []
    for left, right in comparisons:
        for metric_name in METRICS:
            for subset_name, folds in (("all_10", list(range(1, 11))), ("exclude_fold2_9", [f for f in range(1, 11) if f != 2])):
                diff = np.asarray([100 * (float(lookup[(f, left)][metric_name]) - float(lookup[(f, right)][metric_name])) for f in folds])
                mean = float(diff.mean()); sd = float(diff.std(ddof=1)); half = float(t.ppf(.975, len(diff)-1) * sd / np.sqrt(len(diff)))
                rows.append({"comparison": f"{left} - {right}", "metric": metric_name, "fold_set": subset_name,
                             "n_outer_folds": len(diff), "mean_delta_pp": mean, "ci95_low_descriptive_pp": mean-half,
                             "ci95_high_descriptive_pp": mean+half, "wins": int((diff>1e-12).sum()),
                             "ties": int((np.abs(diff)<=1e-12).sum()), "losses": int((diff<-1e-12).sum()),
                             "fold_deltas_pp": json.dumps(diff.tolist())})
    return rows


def correction_rows(outputs):
    import numpy as np
    rows = []
    for out in outputs:
        d = out["data"]; ids=out["ids"]; y=out["labels"]; b0=out["baseline_pred"]
        p = probe_module()
        role = p.role_masks(d, ids)
        subsets = {"all": np.ones(len(ids), dtype=bool), "stable_hard": role[2][ids] | role[3][ids],
                   "stable_reliable": role[0][ids] | role[1][ids]}
        comparators = ((GROUP_B5, out["predictions"][GROUP_B5]), (GROUP_B4, out["predictions"][GROUP_B4]))
        for group, pred in comparators:
            for name, select in subsets.items():
                wrong_to_correct=(b0!=y)&(pred==y)&select; correct_to_wrong=(b0==y)&(pred!=y)&select
                rows.append({"fold":out["fold"],"comparison":f"{group} vs {GROUP_B0}","subset":name,"n":int(select.sum()),
                             "wrong_to_correct":int(wrong_to_correct.sum()),"correct_to_wrong":int(correct_to_wrong.sum()),
                             "net_corrections":int(wrong_to_correct.sum()-correct_to_wrong.sum()),
                             "asd_to_td_before":int(np.sum((y==0)&(b0==1)&select)),"asd_to_td_after":int(np.sum((y==0)&(pred==1)&select)),
                             "td_to_asd_before":int(np.sum((y==1)&(b0==0)&select)),"td_to_asd_after":int(np.sum((y==1)&(pred==0)&select))})
                if group == GROUP_B5:
                    b4=out["predictions"][GROUP_B4]
                    w=(b4!=y)&(pred==y)&select; h=(b4==y)&(pred!=y)&select
                    rows.append({"fold":out["fold"],"comparison":f"{GROUP_B5} vs {GROUP_B4}","subset":name,"n":int(select.sum()),
                                 "wrong_to_correct":int(w.sum()),"correct_to_wrong":int(h.sum()),"net_corrections":int(w.sum()-h.sum()),
                                 "asd_to_td_before":int(np.sum((y==0)&(b4==1)&select)),"asd_to_td_after":int(np.sum((y==0)&(pred==1)&select)),
                                 "td_to_asd_before":int(np.sum((y==1)&(b4==0)&select)),"td_to_asd_after":int(np.sum((y==1)&(pred==0)&select))})
    return rows


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def write_report(status):
    import numpy as np
    fold_rows=probe_module().read_csv(RESULTS/"fold_metrics.csv")
    paired=probe_module().read_csv(RESULTS/"paired_comparisons.csv")
    corr=probe_module().read_csv(RESULTS/"correction_analysis.csv")
    names={GROUP_B0:"B0-NL CAM-only RBF-SVM",GROUP_B4:"B4 CAM+Reliable RBF-SVM",GROUP_B5:"B5 CAM+Reliable/Hard Paired RBF-SVM"}
    metrics=("acc","hard_auc","prob_auc","sen","spe","asd_f1")
    groups=(GROUP_B0,GROUP_B4,GROUP_B5)
    def summary(group, folds=range(1,11)):
        rows=[x for x in fold_rows if x["group"]==group and int(x["fold"]) in folds]
        return {m:(100*float(np.mean([float(x[m]) for x in rows])),100*float(np.std([float(x[m]) for x in rows],ddof=0))) for m in metrics}
    lines=["# Experiment 76.3 — Nonlinear Prototype Incremental Utility Probe","",
      f"Status: {status.get('status')} after independent deterministic audit; conclusion: **{status.get('conclusion')}**.",
      "","## Design and provenance","",
      "E74.4-G2 best checkpoints were frozen. Each outer fold had one CAM144 extraction forward; OutputLayer ran once and no model, CAM, classifier, optimizer, or checkpoint state was updated. B4 uses the exact E76.2-B1 reliable-only 12-slot evidence features; B5 uses E76.2-B2 reliable+hard-pair 12-slot evidence. The only readout change is the classifier: both are StandardScaler + RBF-SVC (C=1, gamma=scale, probability=False).",
      "","The historical B0-NL OOF patient IDs, predictions, scores, and fold metrics are reused from E76.2's audited cache and independently re-scored; B0-NL is not refit. ASD=0, TD=1; `classes_=[0,1]` is checked and the ROC score is `-decision_function`, so larger values indicate ASD. It is a ranking score, not a calibrated probability; log-loss/Brier are not calculated for SVC.",
      "","Five outer-training inner folds use the E76.2 fixed split. Training-side prototype evidence is cross-fitted over three disjoint support/query folds using seed 76300+fold; each validation patient is excluded from its own prototype bank. StandardScaler fits only the inner training support. Fold 2 rows 377 and 407 are excluded from added support.",
      "","E76.2's historical CAM encoder was trained using outer-training subjects, and its base checkpoint was selected with val=test. Therefore these are frozen-encoder inner-OOF readout diagnostics, not independent end-to-end generalization estimates. Pre-CV preprocessing, transductive graph construction, and Fold 2's historical overlap remain. Adding evidence also changes standardized feature distribution and the effective `gamma=scale`, so any difference is a feature-configuration result, not proof of novel disease information.",
      "","## Results","","Mean ± population SD across ten outer folds, percent. Hard-AUC is balanced accuracy from hard predictions; Prob-AUC uses the continuous SVC decision score.","",
      "| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        vals=summary(group)
        lines.append("| "+names[group]+" | "+" | ".join(f"{vals[m][0]:.4f}±{vals[m][1]:.4f}" for m in metrics)+" |")
    lines += ["","## Paired contrasts","","Delta is left method minus right method in percentage points. Confidence intervals are descriptive paired t intervals across outer folds; n=10 and the nine-fold sensitivity excludes Fold 2.","",
      "| Comparison | Metric | Δ pp | 95% CI | W/T/L | Δ pp without Fold 2 |","|---|---|---:|---:|---:|---:|"]
    metric_labels={"acc":"ACC","hard_auc":"Hard-AUC","prob_auc":"Prob-AUC","sen":"SEN","spe":"SPE","asd_f1":"ASD-F1"}
    for left,right in ((GROUP_B4,GROUP_B0),(GROUP_B5,GROUP_B0),(GROUP_B5,GROUP_B4)):
        for m in metrics:
            a=next(x for x in paired if x["comparison"]==f"{left} - {right}" and x["metric"]==m and x["fold_set"]=="all_10")
            n=next(x for x in paired if x["comparison"]==f"{left} - {right}" and x["metric"]==m and x["fold_set"]=="exclude_fold2_9")
            lines.append(f"| {names[left]} − {names[right]} | {metric_labels[m]} | {float(a['mean_delta_pp']):+.3f} | [{float(a['ci95_low_descriptive_pp']):+.3f}, {float(a['ci95_high_descriptive_pp']):+.3f}] | {a['wins']}/{a['ties']}/{a['losses']} | {float(n['mean_delta_pp']):+.3f} |")
    lines += ["","## Fold 2 sensitivity","","Nine-fold mean ± population SD (%):", "","| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        vals=summary(group,[f for f in range(1,11) if f!=2])
        lines.append("| "+names[group]+" | "+" | ".join(f"{vals[m][0]:.4f}±{vals[m][1]:.4f}" for m in metrics)+" |")
    lines += ["","## Error corrections","","Wrong→correct / correct→wrong / net counts are summed across outer-fold OOF occurrences; a participant may occur in several outer-training diagnostics, so these are not unique-person totals.","",
      "| Comparison | Patient subset | Wrong→correct | Correct→wrong | Net |","|---|---|---:|---:|---:|"]
    for comparison in (f"{GROUP_B5} vs {GROUP_B0}",f"{GROUP_B5} vs {GROUP_B4}"):
        for subset in ("all","stable_hard","stable_reliable"):
            rs=[x for x in corr if x["comparison"]==comparison and x["subset"]==subset]
            lines.append(f"| {names[comparison.split(' vs ')[0]]} vs {names[comparison.split(' vs ')[1]]} | {subset} | {sum(int(x['wrong_to_correct']) for x in rs)} | {sum(int(x['correct_to_wrong']) for x in rs)} | {sum(int(x['net_corrections']) for x in rs)} |")
    lines += ["","## Audit and decision","",f"Independent audit: **{status.get('status')}**; ten frozen checkpoints, 30/30 OOF method-fold metrics independently recalculated, B0-NL baseline rechecked against the original E76.2 cache, B4/B5 predictions and decision scores deterministically reproduced, five-fold validation partitions and three-fold evidence cross-fitting verified, test-support intersection 0/10, Fold 2 rows 377/407 excluded. No parameter updates or model training.","",
      "Decision: **"+str(status.get("conclusion"))+"**. Interpret the comparison under the frozen-representation and historical evaluation limitations above; do not treat the inner-OOF accuracy as outer test generalization.","",
      "Artifacts: `fold_metrics.csv`, `paired_comparisons.csv`, `correction_analysis.csv`, `audit_status.json`, `preflight.json`, and `manifest.json`.",""]
    (RESULTS/"report.md").write_text("\n".join(lines),encoding="utf-8")


def run():
    import numpy as np
    start=time.time(); RESULTS.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    refs=verify_references(); source_sha=sha_json(inventory())
    e762_rows=probe_module().read_csv(E762/"results/fold_metrics.csv")
    by_old={(int(r["fold"]),r["group"]):r for r in e762_rows}
    outputs=[]; metric_rows=[]; oof_cache={}; fold_audit=[]
    for fold in range(1,11):
        print(f"fold {fold}/10",flush=True)
        out=run_fold(fold); outputs.append(out); metric_rows.extend(out["rows"])
        pfx=f"f{fold}_"; ids=out["ids"]
        oof_cache[pfx+"ids"]=ids; oof_cache[pfx+"labels"]=out["labels"]
        oof_cache[pfx+"B0NL_pred"]=out["baseline_pred"]; oof_cache[pfx+"B0NL_score_asd"]=out["baseline_score"]
        for g in (GROUP_B4,GROUP_B5):
            oof_cache[pfx+g+"_pred"]=out["predictions"][g]; oof_cache[pfx+g+"_score_asd"]=out["scores"][g]
        old=by_old[(fold,E762_BASELINE)]
        for m in METRICS:
            val=out["rows"][0][m]
            if not np.isclose(val,float(old[m]),rtol=1e-12,atol=1e-12): raise RuntimeError(f"fold {fold}: historical B0-NL {m} differs from E76.2")
        fold_audit.append({"fold":fold,"eligible_count":len(ids),"checkpoint_sha256":out["data"]["checkpoint_sha"],
                           "cam144_sha256":hashlib.sha256(out["data"]["h"].tobytes()).hexdigest(),
                           "checkpoint_weights_sha256":out["data"]["checkpoint_weights_sha"],
                           "post_forward_weights_sha256":out["data"]["post_forward_weights_sha"],
                           "output_layer_calls":1,"fold2_377_407_excluded":fold!=2 or ({376,406}.isdisjoint(set(ids.tolist()))),
                           "inner_folds":out["fold_records"]})
    paired=paired_rows(metric_rows); corrections=correction_rows(outputs)
    csv_write(RESULTS/"fold_metrics.csv",metric_rows); csv_write(RESULTS/"paired_comparisons.csv",paired); csv_write(RESULTS/"correction_analysis.csv",corrections)
    np.savez_compressed(CACHE/"oof_predictions.npz",**oof_cache)
    status={"status":"RUN_COMPLETE_PENDING_AUDIT","folds":10,"groups":[GROUP_B4,GROUP_B5],
            "historical_baseline":"E76.2 B0-NL audited OOF cache; not rerun",
            "source_sha256":source_sha,"references":refs,"checkpoint_folds":fold_audit,
            "model_training_performed":False,"optimizer_or_parameter_updates":False,
            "duration_seconds":time.time()-start}
    (RESULTS/"audit_status.json").write_text(json.dumps(status,indent=2,allow_nan=False)+"\n")
    preflight={"status":"PASS","reference_checks":refs,"source_sha256":source_sha,
               "historical_b0nl_exact_fold_metrics":True,"inner_oof_scheme":"5-fold random_state 10; training evidence crossfit 3-fold random_state 76300+fold",
               "svc":{"kernel":"rbf","C":1.0,"gamma":"scale","probability":False,"classes":[0,1],"asd_score":"-decision_function"},
               "fold2_exposed_rows_excluded_1based":[377,407],"training_or_optimizer_update":False}
    (RESULTS/"preflight.json").write_text(json.dumps(preflight,indent=2)+"\n")
    print(json.dumps({"folds":10,"methods":len(metric_rows),"duration_seconds":status["duration_seconds"]},indent=2),flush=True)


if __name__=="__main__": run()
