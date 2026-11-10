#!/usr/bin/env python3
"""Independent readback, paired statistics, report, and handoff for E75.4."""
from __future__ import annotations

import csv
import ast
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
RT = EXP / ".runtime"
FEATURES = RT / "frozen_features"
sys.path.insert(0, str(EXP))
import probe
import run as runner


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def close(a, b, atol=1e-12):
    return abs(float(a) - float(b)) <= atol


def independent_base_readback(fold):
    r = runner.base_runner()
    # The historical loader sets inter-op threads after importing torch modules;
    # that is too late in this independent audit process. Thread count has no
    # bearing on the frozen eval result, so bypass only this late setter while
    # preparing the read-only state, then restore it immediately.
    set_interop_threads = torch.set_num_interop_threads
    torch.set_num_interop_threads = lambda _count: None
    try:
        state, _, identity = r.prepare(fold)
    finally:
        torch.set_num_interop_threads = set_interop_threads
    state["fold"] = fold
    state["group"] = runner.GROUP
    ckpt_path = r.best_path(runner.GROUP, fold)
    checkpoint = r.torch_load(ckpt_path)
    r.load_history().load_bundle(state, checkpoint["best_state"])
    state["cam"].eval(); state["model"].eval()
    captured = {"cam": 0, "out": 0}

    def cam_in(_module, args):
        captured["f"] = args[0].detach().cpu().numpy().copy()

    def output_in(_module, args):
        captured["h"] = args[0].detach().cpu().numpy().copy()

    def cam_count(_module, _args, _out):
        captured["cam"] += 1

    def out_count(_module, _args, _out):
        captured["out"] += 1

    cam = state["cam"]
    hooks = [cam.register_forward_pre_hook(cam_in), cam.register_forward_hook(cam_count),
             cam.Outputlayer.register_forward_pre_hook(output_in),
             cam.Outputlayer.register_forward_hook(out_count)]
    rng_before = runner.rng_digest()
    param_before = runner.tensor_hash(state["model"], cam)
    with torch.no_grad():
        output = r.forward(state, runner.GROUP, training=False)
    param_after = runner.tensor_hash(state["model"], cam)
    rng_after = runner.rng_digest()
    for h in hooks:
        h.remove()
    if (captured["cam"], captured["out"]) != (1, 1):
        raise RuntimeError(f"fold {fold}: base path call counts are not one CAM/one OutputLayer")
    if rng_before != rng_after or param_before != param_after:
        raise RuntimeError(f"fold {fold}: frozen replay changed RNG or parameters")
    metric = r.load_history().load_e69().load_e61().load_base().classification_metrics(
        output["logp"], state["test"], state["y"])
    if not np.array_equal(metric["pred"], checkpoint["test_prediction"]):
        raise RuntimeError(f"fold {fold}: independent replay hard prediction mismatch")
    if not np.array_equal(metric["prob"], checkpoint["test_probability"]):
        raise RuntimeError(f"fold {fold}: independent replay probability mismatch")
    cache = runner.load_features(fold)
    x = torch.cat(state["parts"], dim=1).detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(x, cache["x"]):
        raise RuntimeError(f"fold {fold}: independently reconstructed X320 differs from frozen cache")
    train_idx=np.asarray(state["train_np"],dtype=np.int64)
    test_idx=np.asarray(state["test_np"],dtype=np.int64)
    expected_overlap=np.asarray([376,406],dtype=np.int64) if fold==2 else np.asarray([],dtype=np.int64)
    overlap=np.intersect1d(train_idx,test_idx)
    expected_probe_train=np.setdiff1d(train_idx,test_idx)
    if not np.array_equal(np.sort(overlap),expected_overlap):
        raise RuntimeError(f"fold {fold}: E74.4 train/test overlap differs from the locked historical protocol")
    if not np.array_equal(train_idx,cache["train"]) or not np.array_equal(test_idx,cache["test"]):
        raise RuntimeError(f"fold {fold}: extracted train/test indices differ from independent E74.4 prepare(fold)")
    if not np.array_equal(expected_probe_train,cache["probe_train"]):
        raise RuntimeError(f"fold {fold}: new probe training indices differ from outer training minus test overlap")
    if not np.array_equal(state["y"].detach().cpu().numpy().astype(np.int64),cache["labels"]):
        raise RuntimeError(f"fold {fold}: cached labels differ from E74.4 data state")
    if fold==2 and (int(cache["labels"][376])!=0 or int(cache["labels"][406])!=1):
        raise RuntimeError("Fold 2 historical exposed rows do not encode ASD=0 at 377 and TD=1 at 407")
    if not np.array_equal(captured["f"], cache["f"]):
        raise RuntimeError(f"fold {fold}: independently replayed DARG feature differs")
    if not np.array_equal(captured["h"], cache["h"]):
        raise RuntimeError(f"fold {fold}: independently replayed CAM feature differs")
    feature_meta=json.loads((FEATURES / f"fold{fold:02d}.json").read_text(encoding="utf-8"))
    if identity != feature_meta["identity"]:
        raise RuntimeError(f"fold {fold}: base fold identity differs from feature record")
    return {"fold": fold, "status": "PASS", "checkpoint_sha256": runner.sha_file(ckpt_path),
            "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
            "cam_calls": 1, "outputlayer_calls": 1, "rng_unchanged": True,
            "parameters_unchanged": True, "hard_prediction_exact": True,
            "probability_exact": True, "X320_exact": True, "F320_exact": True, "H144_exact": True,
            "train_test_indices_exact": True,"historical_overlap_1based":(overlap+1).tolist(),
            "metrics": {k: float(metric[k]) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")}}


def verify_prompt_sources():
    bank = json.loads((EXP / "prompt_banks.json").read_text(encoding="utf-8"))
    e60_cache = torch.load(Path(runner.CFG["text_encoder"]["cache_path"]), map_location="cpu", weights_only=False)
    network_source = Path("/data/adodas/SPromptGL-main/network.py").read_text(encoding="utf-8")
    p1 = [x["text"] for x in bank["banks"]["P1-SPromptGL-8"]]
    source_order = ["Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
                    "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
                    "White matter integrity issues", "Different activation patterns"]
    if p1 != source_order or any(x not in network_source for x in p1):
        raise RuntimeError("P1 source text/order mismatch")
    p2 = [x["text"] for x in bank["banks"]["P2-BrainPrompt-2"]]
    official_p2 = [
        "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
        "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    ]
    if p2 != official_p2:
        raise RuntimeError("P2 text or source order differs from the official BrainPrompt label file")
    prompt_meta = json.loads((RT / "prompt_embedding_manifest.json").read_text(encoding="utf-8"))
    emb = np.load(RT / "prompt_embeddings.npz", allow_pickle=False)
    if prompt_meta["embedding_npz_sha256"] != runner.sha_file(RT / "prompt_embeddings.npz"):
        raise RuntimeError("frozen prompt embedding artifact SHA mismatch")
    e60=Path(runner.CFG["references"]["e60"])
    expected_cache_sources={
        "knowledge_sha256":runner.sha_file(e60/"knowledge_base.json"),
        "rn50_sha256":runner.sha_file(Path(runner.CFG["text_encoder"]["weights_path"])),
        "config_sha256":runner.sha_file(e60/"config.json"),
        "text_features_sha256":runner.sha_file(e60/"text_features.py")}
    if prompt_meta.get("cache_sources")!=expected_cache_sources:
        raise RuntimeError("CLIP encoding source hashes differ from the audited E60/RN50 files")
    for name, items in bank["banks"].items():
        details = prompt_meta["bank_metadata"].get(name)
        if details is None or details["texts"] != [x["text"] for x in items] or details["source_classes"] != [x["class"] for x in items]:
            raise RuntimeError(f"Prompt text/order/class metadata mismatch for {name}")
        vectors = np.asarray(emb[details["array_key"]], dtype=np.float32)
        if runner.sha_bytes(vectors.tobytes()) != details["vector_sha256"]:
            raise RuntimeError(f"Prompt vector SHA mismatch for {name}")
    if prompt_meta["weights_sha256"] != runner.CFG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("CLIP RN50 SHA mismatch")
    if prompt_meta["cache_sha256"] != runner.sha_file(Path(runner.CFG["text_encoder"]["cache_path"])):
        raise RuntimeError("CLIP cache SHA mismatch")
    if prompt_meta["bank_metadata"]["P1-SPromptGL-8"]["vector_sha256"] != runner.sha_bytes(e60_cache["embeddings"]["original"].detach().cpu().numpy().astype(np.float32).tobytes()):
        raise RuntimeError("P1 vectors do not match the audited frozen cache")
    p2_by_text={text:vector.detach().cpu().numpy().astype(np.float32)
                for text,vector in zip(e60_cache["text_map"]["brainprompt"],e60_cache["embeddings"]["brainprompt"])}
    p2_texts=[x["text"] for x in bank["banks"]["P2-BrainPrompt-2"]]
    p2_expected=np.stack([p2_by_text[text] for text in p2_texts])
    if prompt_meta["bank_metadata"]["P2-BrainPrompt-2"]["vector_sha256"] != runner.sha_bytes(p2_expected.tobytes()):
        raise RuntimeError("P2 vectors do not match the audited frozen cache")
    if not np.isfinite(np.concatenate([emb[k].reshape(-1) for k in emb.files])).all():
        raise RuntimeError("nonfinite prompt vectors")
    if len(prompt_meta["newly_encoded_texts"]) == 0:
        raise RuntimeError("new prompt vectors were not encoded")
    cached_texts=set()
    for key,texts in e60_cache["text_map"].items():
        cached_texts.update(texts)
    all_texts={item["text"] for items in bank["banks"].values() for item in items}
    expected_new=sorted(all_texts-cached_texts)
    if prompt_meta["newly_encoded_texts"]!=expected_new or len(set(expected_new))!=len(expected_new):
        raise RuntimeError("offline CLIP text encoding set is incomplete, duplicated, or unexpected")
    return {"status":"PASS", "RN50_sha256":prompt_meta["weights_sha256"],
            "P1_text_order":"exact network.py source order", "P2_text_order":"official TC then ASD source order",
            "P1_cached_vector_exact":True,"P2_cached_vector_exact":True,
            "new_prompt_count":len(prompt_meta["newly_encoded_texts"]),
            "new_texts_encoded_once":True,"all_vectors_finite":True,
            "embedding_npz_sha256":prompt_meta["embedding_npz_sha256"]}


def metric_row_for_predictions(fold_rows):
    y = np.asarray([int(x["y_asd"]) for x in fold_rows], dtype=np.int64)
    p = np.asarray([float(x["prob_asd"]) for x in fold_rows], dtype=np.float64)
    return probe.binary_metrics(y, p)


def verify_shared_inner_splits(probe_execution):
    config=json.loads((EXP/"config.json").read_text(encoding="utf-8"))
    analysis=config["analysis"]
    seed_base=int(analysis["shared_inner_split_seed_base"])
    expected=[seed_base+fold for fold in range(1,11)]
    if probe_execution.get("shared_inner_split_seed_base")!=seed_base or probe_execution.get("shared_inner_split_seeds")!=expected:
        raise RuntimeError("probe execution did not record the registered shared inner-fold seeds")
    if int(analysis["inner_cv_folds"])!=5 or tuple(float(x) for x in analysis["logistic_C_candidates"])!=probe.C_VALUES:
        raise RuntimeError("inner-CV fold count or regularization candidates differ from registration")
    tree=ast.parse((EXP/"run.py").read_text(encoding="utf-8"))
    checked={"nested_logistic":0,"nested_alignment":0}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):
            continue
        name=node.func.attr
        if name not in checked:
            continue
        seed_index=3 if name=="nested_logistic" else 5
        if len(node.args)<=seed_index or not isinstance(node.args[seed_index],ast.Name) or node.args[seed_index].id!="inner_seed":
            raise RuntimeError(f"{name} call does not use the common per-fold inner_seed")
        checked[name]+=1
    if checked["nested_logistic"]<1 or checked["nested_alignment"]<1:
        raise RuntimeError(f"registered probe calls missing from shared-split source audit: {checked}")
    return {"status":"PASS","split_seed_base":seed_base,"per_fold_seeds":expected,
            "inner_folds":5,"C_candidates":list(probe.C_VALUES),"probe_calls_verified":checked}


def independent_anonymous_gram_audit():
    """Rebuild B1/B2 projected-space orthogonal controls independently."""
    meta = json.loads((RT / "prompt_embedding_manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(RT / "prompt_embeddings.npz", allow_pickle=False)
    recorded = read_csv(RESULTS / "anonymous_gram_audit.csv")
    recorded_map = {(int(r["fold"]), r["setting"], r["pair"]): float(r["max_abs_gram_error"]) for r in recorded}
    expected = {}
    for fold in range(1, 11):
        projections = (("B1", int(runner.CFG["analysis"]["B1_projection_seed_base"]) + fold),
                       ("B2", int(runner.CFG["analysis"]["B2_fixed_text_projection_seed_base"]) + fold))
        for setting, seed_projection in projections:
            generator = np.random.default_rng(seed_projection)
            generator.standard_normal((9, 128), dtype=np.float64)
            wp = (generator.standard_normal((1024, 128), dtype=np.float64) / math.sqrt(128.0)).astype(np.float32)
            for bank_name in ("P1-SPromptGL-8", "P5-Functional-Network-8"):
                details = meta["bank_metadata"][bank_name]
                text = np.asarray(vectors[details["array_key"]], dtype=np.float32)
                projected = text @ wp
                projected = projected / np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-8)
                seed_orthogonal = int(runner.CFG["analysis"]["anonymous_orthogonal_seed_base"]) + fold + (0 if bank_name.startswith("P1") else 1000)
                local = np.random.default_rng(seed_orthogonal)
                matrix = local.standard_normal((128, 128))
                orthogonal, upper = np.linalg.qr(matrix)
                signs = np.where(np.diag(upper) < 0, -1.0, 1.0)
                orthogonal = orthogonal * signs[None, :]
                projected64 = projected.astype(np.float64)
                anonymous = projected64 @ orthogonal
                error = float(np.max(np.abs(projected64 @ projected64.T - anonymous @ anonymous.T)))
                if setting == "B1":
                    keys = [(fold, setting, bank_name)]
                else:
                    anon_name = bank_name.replace("-8", "") + "-Anonymous-Gram-Matched"
                    keys = [(fold, setting, anon_name), (fold, setting, bank_name)]
                for key in keys:
                    value = error if setting == "B1" or key[2] != bank_name else 0.0
                    expected[key] = value
                    if key[2] != bank_name and value >= 1e-5:
                        raise RuntimeError(f"independent projected anonymous Gram mismatch {key}: {value}")
                    if key not in recorded_map or abs(recorded_map[key] - value) > 1e-10:
                        raise RuntimeError(f"recorded anonymous Gram value differs from independent recomputation for {key}")
    if len(recorded_map) != 60 or set(recorded_map) != set(expected):
        raise RuntimeError(f"anonymous Gram audit expected 60 rows (B1/B2 anonymous plus B2 source consistency), got {len(recorded_map)}")
    return {"status": "PASS", "rows_recomputed": len(expected), "max_abs_error": max(expected.values()),
            "actual_use_space": "B1 and B2 128-D text projections independently reconstructed"}


def audit_probe_predictions():
    predictions = read_csv(RESULTS / "probe_predictions.csv")
    metric_rows = read_csv(RESULTS / "prompt_comparison.csv")
    grouped = {}
    for row in predictions:
        key = (int(row["fold"]), row["prompt_group"], row["setting"], row["method"])
        grouped.setdefault(key, []).append(row)
    expected = {(int(r["fold"]),r["prompt_group"],r["setting"],r["method"]):r for r in metric_rows}
    if set(grouped) != set(expected):
        raise RuntimeError(f"prediction/metric key mismatch ({len(grouped)} vs {len(expected)})")
    checks = 0
    for key, rows in grouped.items():
        fold = key[0]
        feat = runner.load_features(fold)
        indices = np.asarray([int(x["subject_index_0based"]) for x in rows], dtype=np.int64)
        if not np.array_equal(indices, feat["test"]):
            raise RuntimeError(f"fold {fold} probe prediction indices differ from frozen test indices")
        y_expected = probe.asd_target(feat["labels"][indices])
        y = np.asarray([int(x["y_asd"]) for x in rows], dtype=np.int64)
        if not np.array_equal(y, y_expected):
            raise RuntimeError(f"fold {fold} prediction labels are not aligned to frozen test labels")
        train_sha = runner.sha_bytes(feat["probe_train"].astype(np.int64).tobytes())
        if any(x["training_index_sha256"] != train_sha for x in rows):
            raise RuntimeError(f"fold {fold} probe training-index SHA mismatch")
        if set(feat["probe_train"]) & set(feat["test"]):
            raise RuntimeError(f"fold {fold} new diagnostic training overlaps test rows")
        if fold == 2 and ({376,406} & set(feat["probe_train"])):
            raise RuntimeError("Fold 2 exposed rows were used by a new probe")
        values = metric_row_for_predictions(rows)
        mr = expected[key]
        mapping = {"acc":"acc_pct", "hard_auc":"hard_auc_pct", "prob_auc":"prob_auc_pct",
                   "sen":"sen_pct", "spe":"spe_pct", "asd_f1":"asd_f1_pct",
                   "log_loss":"log_loss", "brier":"brier"}
        for name, field in mapping.items():
            if not close(values[name] * (100 if field.endswith("_pct") else 1), float(mr[field]), 1e-10):
                raise RuntimeError(f"independent metric replay mismatch for {key} {name}")
        pred = np.asarray([int(x["pred_asd"]) for x in rows], dtype=np.int64)
        if not np.array_equal(pred, values["prediction"]):
            raise RuntimeError(f"independent hard prediction mismatch for {key}")
        if any(int(x["pred_asd"]) != int(float(x["prob_asd"]) >= 0.5) for x in rows):
            raise RuntimeError(f"threshold mismatch for {key}")
        for row in rows:
            if row["selected_C"] and float(row["selected_C"]) not in probe.C_VALUES:
                raise RuntimeError(f"unregistered C selected for {key}")
        checks += 1
    return {"status":"PASS", "prediction_groups_independently_recomputed":checks,
            "all_outer_test_indices_exact":True,"probe_train_test_disjoint":True,
            "fold2_rows_377_407_excluded_from_new_probe_training":True}


def aggregate_metric_rows(rows):
    metrics = ("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct","log_loss","brier")
    result = []
    groups = sorted({(r["prompt_group"],r["setting"],r["method"]) for r in rows})
    predictions = read_csv(RESULTS / "probe_predictions.csv")
    pred_map = {}
    for r in predictions:
        key=(int(r["fold"]),r["prompt_group"],r["setting"],r["method"])
        pred_map.setdefault(key,[]).append(r)
    for group, setting, method in groups:
        folds = sorted(int(r["fold"]) for r in rows if (r["prompt_group"],r["setting"],r["method"])==(group,setting,method))
        for scope, chosen in (("all_10",folds),("exclude_fold2_descriptive_9",[f for f in folds if f!=2])):
            for metric in metrics:
                vals=[float(next(r[metric] for r in rows if int(r["fold"])==f and (r["prompt_group"],r["setting"],r["method"])==(group,setting,method))) for f in chosen]
                result.append({"prompt_group":group,"setting":setting,"method":method,"scope":scope,"n_folds":len(vals),
                               "metric":metric,"mean":float(np.mean(vals)),"population_sd":float(np.std(vals,ddof=0)),"units":"percent" if metric.endswith("_pct") else "probability_or_nll"})
        if 2 in folds:
            rows2=pred_map[(2,group,setting,method)]
            exposed={376,406}
            keep=[x for x in rows2 if int(x["subject_index_0based"]) not in exposed]
            m=metric_row_for_predictions(keep)
            for metric in metrics:
                val=m[metric[:-4] if metric.endswith("_pct") else metric]
                if metric.endswith("_pct"):
                    val*=100
                result.append({"prompt_group":group,"setting":setting,"method":method,"scope":"fold2_remove_exposed_rows_descriptive","n_folds":1,
                               "metric":metric,"mean":float(val),"population_sd":0.0,"units":"percent" if metric.endswith("_pct") else "probability_or_nll",
                               "n_test_after_removal":len(keep)})
    return result


def paired_rows(utility_rows):
    # Primary within-fold contrasts: M2 vs calibrated base and capacity-matched no-text signal.
    data={}
    for row in utility_rows:
        key=(int(row["fold"]),row["prompt_group"],row["setting"],row["model"])
        data[key]=row
    comparisons=[]
    tests=[]
    metric_names=("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct","log_loss","brier")
    prompt_groups=sorted({r["prompt_group"] for r in utility_rows if r["prompt_group"]!="ALL"})
    scopes={"all_10":list(range(1,11)),"exclude_fold2_descriptive_9":[f for f in range(1,11) if f!=2]}

    def add_comparison(group,setting,left_model,right_model,kind, family):
        for scope,folds in scopes.items():
            available=[f for f in folds if (f,group,setting,left_model) in data and (f,group,setting,right_model) in data]
            if not available:
                continue
            for metric in metric_names:
                diffs=[]
                for f in available:
                    left=float(data[(f,group,setting,left_model)][metric])
                    right=float(data[(f,group,setting,right_model)][metric])
                    diffs.append((right-left) if metric in ("log_loss","brier") else (left-right))
                p=probe.exact_signflip_p(diffs)
                seed_text=f"{group}|{setting}|{kind}|{metric}|{scope}"
                seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8],16)
                ci=probe.bootstrap_mean_ci(diffs,754600+seed%100000,10000)
                row={"comparison_family":family,"prompt_group":group,"setting":setting,"comparison":kind,
                     "scope":scope,"metric":metric,"n_folds":len(diffs),"mean_improvement":float(np.mean(diffs),),
                     "population_sd_of_fold_improvement":float(np.std(diffs,ddof=0)),
                     "improvement_95ci_low":ci[0],"improvement_95ci_high":ci[1],
                     "exact_two_sided_signflip_p":p,
                     "wins":int((np.asarray(diffs)>1e-12).sum()),"ties":int((np.abs(diffs)<=1e-12).sum()),
                     "losses":int((np.asarray(diffs)<-1e-12).sum()),
                     "fold_improvements":json.dumps({str(f):float(d) for f,d in zip(available,diffs)},sort_keys=True)}
                comparisons.append(row)
                tests.append((family,scope,metric,len(comparisons)-1,p))

    for group in prompt_groups:
        for setting in ("B1","B2"):
            add_comparison(group,setting,"M2","M0","M2_vs_M0", "primary_incremental")
            add_comparison(group,setting,"M2","M1","M2_vs_M1", "primary_incremental")

    # Prompt specificity and semantic-control contrasts, all pre-specified.
    def metric_rows_for(group, setting):
        found=[]
        for f in range(1,11):
            row=data.get((f,group,setting,"M2"))
            if row is not None:
                found.append((f,row))
        return found
    specificity = [("P4-Modality-Aware-8","P3-Class-Definition-8"),
                   ("P5-Functional-Network-8","P3-Class-Definition-8"),
                   ("P6-Heterogeneity-Aware-8","P3-Class-Definition-8"),
                   ("P1-SPromptGL-8","P3-Class-Definition-8"),
                   ("P2-BrainPrompt-2","P3-Class-Definition-8")]
    controls=[("P1-SPromptGL-8","P1-SPromptGL-Anonymous-Gram-Matched"),
              ("P5-Functional-Network-8","P5-Functional-Network-Anonymous-Gram-Matched"),
              ("P1-SPromptGL-8","C3-Unrelated-English-8"),
              ("P2-BrainPrompt-2","C3-Unrelated-English-8"),
              ("P5-Functional-Network-8","C3-Unrelated-English-8")]
    random_groups=[f"RND-Unit-{seed}" for seed in (75451,75452,75453)]
    for real_group in ("P1-SPromptGL-8","P2-BrainPrompt-2","P4-Modality-Aware-8",
                       "P5-Functional-Network-8","P6-Heterogeneity-Aware-8"):
        controls.extend((real_group,random_group) for random_group in random_groups)
    for setting in ("B1","B2"):
        for left_group,right_group in specificity+controls:
            for scope,folds in scopes.items():
                shared=[]
                for f in folds:
                    a=data.get((f,left_group,setting,"M2")); b=data.get((f,right_group,setting,"M2"))
                    if a is not None and b is not None:
                        shared.append((f,a,b))
                for metric in metric_names:
                    diffs=[]
                    for _,a,b in shared:
                        x=float(a[metric]); y=float(b[metric])
                        diffs.append((y-x) if metric in ("log_loss","brier") else (x-y))
                    if not diffs:
                        continue
                    p=probe.exact_signflip_p(diffs)
                    seed_text=f"{left_group}|{right_group}|{setting}|{metric}|{scope}"
                    seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8],16)
                    ci=probe.bootstrap_mean_ci(diffs,754700+seed%100000,10000)
                    family="prompt_specificity" if (left_group,right_group) in specificity else "semantic_controls"
                    comparisons.append({"comparison_family":family,"prompt_group":left_group,"reference_group":right_group,
                        "setting":setting,"comparison":"left_vs_reference","scope":scope,"metric":metric,"n_folds":len(diffs),
                        "mean_improvement":float(np.mean(diffs)),"population_sd_of_fold_improvement":float(np.std(diffs,ddof=0)),
                        "improvement_95ci_low":ci[0],"improvement_95ci_high":ci[1],"exact_two_sided_signflip_p":p,
                        "wins":int((np.asarray(diffs)>1e-12).sum()),"ties":int((np.abs(diffs)<=1e-12).sum()),
                        "losses":int((np.asarray(diffs)<-1e-12).sum()),
                        "fold_improvements":json.dumps({str(f):float(d) for (f,_,_),d in zip(shared,diffs)},sort_keys=True)})
                    tests.append((family,scope,metric,len(comparisons)-1,p))
    families={}
    for family,scope,metric,index,p in tests:
        families.setdefault((family,scope,metric),[]).append((index,p))
    for _,items in families.items():
        adjusted=probe.holm_adjust([p for _,p in items])
        for (idx,_),p_adj in zip(items,adjusted):
            comparisons[idx]["holm_adjusted_p"] = p_adj
    return comparisons


def token_summary():
    rows=read_csv(RESULTS/"token_diagnostics.csv")
    out=[]
    for bank in sorted({r["prompt_group"] for r in rows}):
        chosen=[r for r in rows if r["prompt_group"]==bank and r["diagnostic_type"]=="group_token"]
        if not chosen:
            continue
        for slot in range(4):
            sub=[r for r in chosen if int(r["slot_origin"])==slot]
            out.append({"prompt_group":bank,"slot_origin":slot,"slot_name":["PHENO-origin","ANAT-origin","FUNC-origin","CORR-origin"][slot],
                        "folds":len({int(r["fold"]) for r in sub}),
                        "mean_supervised_importance":float(np.mean([float(r["supervised_token_importance"]) for r in sub])),
                        "mean_prompt_abs_contrast":float(np.mean([float(r["mean_abs_d_train"]) for r in sub])),
                        "mean_bootstrap_top4_frequency":float(np.mean([float(r["bootstrap_top4_frequency"]) for r in sub]))})
    return out


def supervised_token_summary(summary):
    """Aggregate the 16 outer-test standalone probes and their fold stability."""
    raw=read_csv(RESULTS/"token_diagnostics.csv")
    token_groups=sorted({r["prompt_group"] for r in summary if r["prompt_group"].startswith("TOKEN-")})
    out=[]
    metrics=("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct")
    for group in token_groups:
        token=int(group[-2:])
        output={"token":token,"slot_origin":token//4,"token_within_slot":token%4,"n_folds":10}
        for metric in metrics:
            row=next((r for r in summary if r["prompt_group"]==group and r["setting"]=="supervised" and
                      r["method"]==f"S1-Token-{token:02d}" and r["scope"]=="all_10" and r["metric"]==metric),None)
            output[f"{metric}_mean"]=None if row is None else float(row["mean"])
            output[f"{metric}_population_sd"]=None if row is None else float(row["population_sd"])
        ranks=[]; coefficient_norms=[]; top4=0
        for fold in range(1,11):
            fold_rows=[r for r in raw if r["diagnostic_type"]=="group_token" and int(r["fold"])==fold]
            if not fold_rows:
                continue
            values=np.asarray([float(next(r["supervised_token_importance"] for r in fold_rows if int(r["token"])==i)) for i in range(16)])
            coefficient_norms.append(float(values[token]))
            ranks.append(int(1+np.sum(values>values[token])))
            top4+=int(token in np.argsort(values)[-4:])
        output.update({"mean_selected_logistic_coefficient_norm":float(np.mean(coefficient_norms)) if coefficient_norms else None,
                       "mean_importance_rank":float(np.mean(ranks)) if ranks else None,
                       "supervised_top4_fold_frequency":top4/max(1,len(ranks))})
        out.append(output)
    return out


def prompt_similarity_summary():
    rows=read_csv(RESULTS/"token_diagnostics.csv")
    out=[]
    for bank in ("P4-Modality-Aware-8","P5-Functional-Network-8"):
        for fold in range(1,11):
            sub=[r for r in rows if r["prompt_group"]==bank and r["diagnostic_type"]=="prompt_token" and int(r["fold"])==fold]
            if not sub:
                continue
            for pidx in range(8):
                prows=[r for r in sub if int(r["prompt_index"])==pidx]
                source_slot=pidx%4
                means=[np.mean([float(r["mean_prompt_similarity_train"]) for r in prows if int(r["slot_origin"])==s]) for s in range(4)]
                out.append({"fold":fold,"prompt_group":bank,"prompt_index":pidx,"prompt_text":prows[0]["prompt_text"],
                            "prompted_slot_index":source_slot,"prompted_slot_mean_similarity":float(means[source_slot]),
                            "other_slot_mean_similarity":float(np.mean([means[s] for s in range(4) if s!=source_slot])),
                            "target_minus_other":float(means[source_slot]-np.mean([means[s] for s in range(4) if s!=source_slot])),
                            "limitation":"slot origins are learned CAM positions, not anatomical/network ROIs"})
    return out


def semantic_geometry_rows():
    """Recompute CLIP-space prompt geometry from the frozen embedding artifact."""
    meta=json.loads((RT/"prompt_embedding_manifest.json").read_text(encoding="utf-8"))
    embeddings=np.load(RT/"prompt_embeddings.npz",allow_pickle=False)
    rows=[]
    for name,details in meta["bank_metadata"].items():
        vectors=np.asarray(embeddings[details["array_key"]],dtype=np.float64)
        norms=np.linalg.norm(vectors,axis=1)
        unit=vectors/np.maximum(norms[:,None],1e-12)
        gram=unit@unit.T
        classes=np.asarray(details["probe_classes"])
        asd=np.flatnonzero(classes=="ASD")
        td=np.flatnonzero(classes=="TD")
        within_asd=[gram[i,j] for ii,i in enumerate(asd) for j in asd[ii+1:]]
        within_td=[gram[i,j] for ii,i in enumerate(td) for j in td[ii+1:]]
        between=[gram[i,j] for i in asd for j in td]
        asd_centroid=unit[asd].mean(axis=0) if len(asd) else np.zeros(vectors.shape[1])
        td_centroid=unit[td].mean(axis=0) if len(td) else np.zeros(vectors.shape[1])
        centroid_cos=float(np.dot(asd_centroid,td_centroid)/(max(np.linalg.norm(asd_centroid)*np.linalg.norm(td_centroid),1e-12))) if len(asd) and len(td) else None
        rows.append({"record_type":"group_summary","prompt_group":name,"prompt_count":len(vectors),
                     "mean_vector_norm":float(norms.mean()),"min_vector_norm":float(norms.min()),
                     "max_vector_norm":float(norms.max()),"mean_intra_asd_cosine":float(np.mean(within_asd)) if within_asd else None,
                     "mean_intra_td_cosine":float(np.mean(within_td)) if within_td else None,
                     "mean_inter_class_cosine":float(np.mean(between)) if between else None,
                     "asd_td_centroid_cosine":centroid_cos,
                     "class_prompt_count_asd":len(asd),"class_prompt_count_td":len(td),
                     "vector_sha256":details["vector_sha256"]})
        tags=np.asarray(details["source_classes"])
        for pidx in range(len(vectors)):
            own=np.flatnonzero(tags==tags[pidx]); own=own[own!=pidx]
            other=np.flatnonzero(tags!=tags[pidx])
            rows.append({"record_type":"prompt_item","prompt_group":name,"prompt_count":len(vectors),
                         "prompt_index":pidx,"prompt_text":details["texts"][pidx] if details["texts"] else "",
                         "prompt_class":details["source_classes"][pidx],"vector_norm":float(norms[pidx]),
                         "mean_within_class_cosine":float(np.mean(gram[pidx,own])) if len(own) else None,
                         "mean_other_class_cosine":float(np.mean(gram[pidx,other])) if len(other) else None,
                         "vector_sha256":runner.sha_bytes(np.asarray(vectors[pidx],dtype=np.float32).tobytes())})
    return rows


def build_report(summary, paired, audit, token_slots, prompt_slots, semantic_geometry, manifest):
    by={(r["prompt_group"],r["setting"],r["method"],r["scope"],r["metric"]):r for r in summary}
    lines=["# Experiment 75.4 — DPUSA", "", "## Execution and audit", "",
           f"- Status: `{manifest['status']}`; frozen E74.4-G2 checkpoint replay {manifest['checkpoint_audit']}; no DARG/CAM retraining.",
           f"- Directory: `{EXP}`; preferred-path fallback: `{runner.CFG.get('root_fallback_reason')}`.",
           f"- Device: {manifest['device']}; frozen forward {manifest['frozen_forward_count']} folds × one pass; CPU probes {manifest['probe_wall_seconds']:.1f}s.",
           f"- RN50 SHA: `{manifest['text_encoder']['weights_sha256']}`; prompts encoded once; anonymous Gram max error {manifest['anonymous_max_gram_error']:.3g}.",
           f"- Engineering retries/failures before the final run: {manifest['analysis_failures']}; each is itemized in `manifest.json`, and failed partial results were excluded.",
           "- E74.4 source audit found a documented audit-only script repair after training. All non-audit protected source hashes match the historical manifest; reconstructed preflight source digest matches the training source digest.",
           "", "## Method audit", "",
           "- SPromptGL's original PLN/PEN computes two class text representations from the eight source phrases, applies class-softmax-derived maximum weights to the `view(871,4,4,9)` CAM latent slots, then adds the original `AvgPool1d(kernel_size=4,stride=4)` residual before the 144-D OutputLayer. The four-dimensional partition is a learned feature layout, not 16 labeled anatomical ROIs.",
           "- E74.4-G2 uses its audited PLNNoPEN CAM-to-OutputLayer path. This audit captures X320, DARG F320 and H144 at that frozen path without changing model parameters or prediction.",
           "- E75.1 and E75.2 were inspected as read-only antecedents. E75.1's G2 ACC was 91.8443% (0.3448 pp below E74.4-G2), residual/H was about 0.8789%, and confidence neutralization produced almost no hard-prediction change; its anonymous G3 reached 92.0742%, so real-text advantage was not established. E75.2 G2 ACC was 91.4995% (0.6897 pp below E74.4-G2), and module-off caused zero hard flips. Neither is used as the base here.",
           "- BrainPrompt's two ABIDE label prompts are retained in official TC then ASD order and TC is mapped to TD for analysis. The inspected `compute_label_loss()` normalizes fused graph and label representations, uses cosine similarities as logits and applies cross-entropy; this audit does not reproduce the published multi-level prompting network or its language encoder. Candidate vectors use the verified SPromptGL/E60 CLIP RN50 cache.",
           "- P4-P6 are preregistered candidate strings, not clinician-validated prompts. The PHENO block includes demographic/site/sex-coded variables; ANAT/FUNC fields are quality-related features in this prepared dataset. P5 cannot be localized to named networks because CORR is stored as 256 positional indices with no verified ROI/network edge map.",
           "- Frozen prompts and token similarities are deterministic transforms of existing H144. They cannot create patient information; only their restricted-learning utility is assessed.",
           "", "## Prompt probes", "", "Rates are mean ± population SD across folds, in percent; log-loss and Brier are unitless. B1 is fixed random projection; weak B1 performance is not evidence that CLIP semantics are absent because CAM and CLIP coordinates are not pretrained to align. B2 trains only a shared 9→128 linear token map and linear head on outer-training features, with C selected by inner five-fold CV.",
           "", "|Prompt / control|Setting|Method|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|Log-loss|Brier|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    display_groups=["BASE","NO_TEXT","P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8","C3-Unrelated-English-8","P1-SPromptGL-Anonymous-Gram-Matched","P5-Functional-Network-Anonymous-Gram-Matched","RND-Unit-75451","RND-Unit-75452","RND-Unit-75453"]
    for group in display_groups:
        selected_methods=[]
        if group=="BASE": selected_methods=[("frozen","S0-Base")]
        elif group=="NO_TEXT": selected_methods=[("incremental","M0-Calibrated-Base"),("incremental","M1-B1-Local-NoText"),("incremental","M1-B2-Trainable-NoText")]
        else: selected_methods=[("B1-fixed","B1-Prompt"),("B1-fixed","M2-B1-BasePlusPrompt"),("B2-trained-alignment","B2-Prompt"),("B2-trained-alignment","M2-B2-BasePlusPrompt")]
        for setting,method in selected_methods:
            vals=[]
            for metric in ("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct","log_loss","brier"):
                row=by.get((group,setting,method,"all_10",metric))
                vals.append("NA" if row is None else f"{float(row['mean']):.4f} ± {float(row['population_sd']):.4f}")
            lines.append(f"|{group}|{setting}|{method}|"+"|".join(vals)+"|")
    lines += ["", "## Incremental utility", "", "The primary contrasts are M2−M0 (calibrated base score) and M2−M1 (same-capacity no-text local signal). Positive values mean improvement for rate metrics and reduction for losses. Sign-flip tests use the ten fold-level paired differences; folds share training data, so p-values and intervals are descriptive, not independent-sample inference. Holm correction is applied within the registered comparison families.",
              "", "|Prompt|Setting|Contrast|Log-loss improvement|Brier improvement|Prob-AUC improvement (pp)|ACC improvement (pp)|W/T/L Prob-AUC|Holm p, Prob-AUC|", "|---|---|---|---:|---:|---:|---:|---|---:|"]
    pby={(r["prompt_group"],r["setting"],r["comparison"],r["scope"],r["metric"]):r for r in paired}
    for group in ["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8","C3-Unrelated-English-8","P1-SPromptGL-Anonymous-Gram-Matched","P5-Functional-Network-Anonymous-Gram-Matched","RND-Unit-75451","RND-Unit-75452","RND-Unit-75453"]:
        for setting in ("B1","B2"):
            for contrast in ("M2_vs_M0","M2_vs_M1"):
                ll=pby.get((group,setting,contrast,"all_10","log_loss")); br=pby.get((group,setting,contrast,"all_10","brier")); auc=pby.get((group,setting,contrast,"all_10","prob_auc_pct")); acc=pby.get((group,setting,contrast,"all_10","acc_pct"))
                if not auc: continue
                lines.append(f"|{group}|{setting}|{contrast}|{float(ll['mean_improvement']):.5f} [{float(ll['improvement_95ci_low']):.5f},{float(ll['improvement_95ci_high']):.5f}]|{float(br['mean_improvement']):.5f}|{float(auc['mean_improvement']):.4f}|{float(acc['mean_improvement']):.4f}|{auc['wins']}/{auc['ties']}/{auc['losses']}|{float(auc.get('holm_adjusted_p',1)):.4f}|")
    lines += ["", "B2 paired utility after excluding Fold 2 (nine-fold descriptive; no refit):", "", "|Prompt|Contrast|Log-loss improvement|Prob-AUC improvement (pp)|W/T/L Log-loss|W/T/L Prob-AUC|", "|---|---|---:|---:|---|---|"]
    for group in ["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]:
        for contrast in ("M2_vs_M0","M2_vs_M1"):
            ll=pby.get((group,"B2",contrast,"exclude_fold2_descriptive_9","log_loss"))
            auc=pby.get((group,"B2",contrast,"exclude_fold2_descriptive_9","prob_auc_pct"))
            if ll and auc:
                lines.append(f"|{group}|{contrast}|{float(ll['mean_improvement']):.5f}|{float(auc['mean_improvement']):.4f}|{ll['wins']}/{ll['ties']}/{ll['losses']}|{auc['wins']}/{auc['ties']}/{auc['losses']}|")
    c1_rows=[r for r in summary if "WrongClass" in r["prompt_group"] and r["scope"]=="all_10" and r["metric"]=="acc_pct"]
    lines += ["", "## Negative controls", "", "C1 fixed-head rows preserve the correctly fitted diagnostic and reverse the ASD/TD score direction. The refit rows train the same capacity with swapped class semantics, so the learner can learn the reversal; this is not a strict semantic-destruction test.",
              "", "|Wrong-class bank|Setting|Method|ACC|Prob-AUC|", "|---|---|---|---:|---:|"]
    for r in c1_rows:
        auc=by.get((r["prompt_group"],r["setting"],r["method"],"all_10","prob_auc_pct"))
        auc_text="NA" if auc is None else f"{float(auc['mean']):.4f} ± {float(auc['population_sd']):.4f}"
        lines.append(f"|{r['prompt_group']}|{r['setting']}|{r['method']}|{float(r['mean']):.4f} ± {float(r['population_sd']):.4f}|{auc_text}|")
    lines += ["", "## Fold 2 exposed-row sensitivity", "", "Descriptive predictions after removing only 1-based rows 377 (ASD) and 407 (TD) from the Fold-2 test metrics; no model or probe was refit for this calculation.",
              "", "|Prompt/control|Method|ACC|Prob-AUC|Log-loss|N test after removal|", "|---|---|---:|---:|---:|---:|"]
    fold2_display=[("BASE","frozen","S0-Base"),("NO_TEXT","incremental","M0-Calibrated-Base"),
                   ("NO_TEXT","incremental","M1-B2-Trainable-NoText")]
    for group in ("P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"):
        fold2_display.append((group,"B2-trained-alignment","M2-B2-BasePlusPrompt"))
    for group,setting,method in fold2_display:
        acc=by.get((group,setting,method,"fold2_remove_exposed_rows_descriptive","acc_pct"))
        auc=by.get((group,setting,method,"fold2_remove_exposed_rows_descriptive","prob_auc_pct"))
        loss=by.get((group,setting,method,"fold2_remove_exposed_rows_descriptive","log_loss"))
        if acc and auc and loss:
            lines.append(f"|{group}|{method}|{float(acc['mean']):.4f}|{float(auc['mean']):.4f}|{float(loss['mean']):.5f}|{acc.get('n_test_after_removal','NA')}|")
    lines += ["", "## Token diagnostics", "", "S1-Token, S2-CAM and S3-DARG probes are in `prompt_comparison.csv`; individual prompt-token similarity and train-bootstrap top-4 stability are in `token_diagnostics.csv`. The four output-position bins below reflect learned CAM feature positions and cannot be assigned anatomical or canonical functional-network names.",
              "", "|Prompt|Slot origin|Mean supervised importance|Mean prompt contrast|Bootstrap top-4 frequency|", "|---|---|---:|---:|---:|"]
    for row in token_slots:
        lines.append(f"|{row['prompt_group']}|{row['slot_name']}|{row['mean_supervised_importance']:.5g}|{row['mean_prompt_abs_contrast']:.5g}|{row['mean_bootstrap_top4_frequency']:.4f}|")
    lines += ["", "The 16 standalone token probes report outer-test fold-mean performance below. Coefficient rank/top-four frequency is a diagnostic from separately fitted training-only logistic probes, not an explanation or anatomical assignment.",
              "", "|Token|CAM output position|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|Mean coefficient rank|Supervised Top-4 fold frequency|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in manifest.get("supervised_token_rows",[]):
        metrics=[row.get(f"{m}_mean") for m in ("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct")]
        values=["NA" if v is None else f"{float(v):.2f}" for v in metrics]
        lines.append(f"|{row['token']}|{['PHENO-output-slot','ANAT-output-slot','FUNC-output-slot','CORR-output-slot'][row['slot_origin']]}|"+"|".join(values)+f"|{row['mean_importance_rank']:.2f}|{row['supervised_top4_fold_frequency']:.2f}|")
    lines += ["", "Prompt text geometry is measured in the actual frozen 1024-D CLIP RN50 space before any B1/B2 projection. `prompt_semantic_geometry.csv` includes each sentence norm, within-class cosine and ASD-TD cosine; random unit-vector controls are reported separately.",
              "", "|Prompt/control|N|Mean norm|Within ASD cosine|Within TD cosine|Between-class cosine|", "|---|---:|---:|---:|---:|---:|"]
    for row in semantic_geometry:
        if row.get("record_type")!="group_summary":
            continue
        lines.append(f"|{row['prompt_group']}|{row['prompt_count']}|{row['mean_vector_norm']:.5f}|{row['mean_intra_asd_cosine'] if row['mean_intra_asd_cosine'] is not None else 'NA'}|{row['mean_intra_td_cosine'] if row['mean_intra_td_cosine'] is not None else 'NA'}|{row['mean_inter_class_cosine'] if row['mean_inter_class_cosine'] is not None else 'NA'}|")
    lines.append("Every individual text vector norm, source class, within-class cosine and opposite-class cosine is stored as a `prompt_item` row in `prompt_semantic_geometry.csv`.")
    lines += ["", "P4/P5 prompt similarity by source-slot index is in `prompt_slot_alignment.csv`. P4 is only an exploratory position correspondence: model tokens have cross-modal CAM interactions. P5 cannot establish DMN, attention, sensorimotor or frontoparietal localization without an audited edge-to-network mapping. The ABIDE functional-connectivity literature reports substantial site and sample variability, so the candidate network sentences are not treated as universal participant-level findings.",
              "", "## Historical protocol and limitations", "",
              "- E74.4-G2 checkpoint selection used `val=test`; preprocessing/feature selection occurred before outer CV; graphs are 871-node transductive. Historical Fold 2 contains 1-based rows 377 (ASD) and 407 (TD) in both train and test. Both rows were removed from all new Fold-2 probe training and inner-CV tuning. The fixed historical base prediction still reflects the original selection/training protocol.",
              "- Ten-fold and nine-fold metrics are descriptive paired summaries on frozen checkpoints. Removing Fold 2 or its two exposed rows does not create a new leakage-free evaluation.",
              "- P2 has one prompt per class versus four per class for most banks; class means are used for compatibility, but prompt-count and wording differences remain.",
              "- B2 can learn a discriminative direction from labels using any fixed prompt vectors; positive B2 results require comparison to random/anonymous controls and M1, and do not by themselves establish medical semantic value.",
              "- The functional-connectivity prompt literature motivates candidate text only. The current 256 CORR positions lack verified named-network mapping; no network-localization claim is made.",
              "", "## Decision", "", f"- Scientific classification: **{manifest['scientific_conclusion']}**.",
              f"- Rationale: {manifest['scientific_conclusion_note']}",
              "- This is a frozen-model diagnostic, not an end-to-end validation of a second innovation module. Any utility is restricted learner use of existing CAM/DARG information plus a semantic prior; no information-theoretic increase in patient data is implied.",
              "", "## Sources", "",
              "- [SPromptGL MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf); [original repository](https://github.com/wanxixi11/SPromptGL).",
              "- [BrainPrompt MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf); [official ABIDE label prompt file](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt).",
              "- Candidate network wording is grounded as group-level literature context, not a per-subject assertion: [multisite ABIDE functional-connectivity classification](https://pubmed.ncbi.nlm.nih.gov/24093016/), [frontoparietal/sensorimotor connectivity in ASD](https://pubmed.ncbi.nlm.nih.gov/40388040/), and [default-mode findings plus site/cohort reproducibility limits](https://pmc.ncbi.nlm.nih.gov/articles/PMC6591952/), [site and cohort variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC5628079/). These papers do not create a mapping from this experiment's 256 CORR indices to named networks."]
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def handoff_append(manifest,summary,paired,token_rows,semantic_geometry):
    marker="<!-- EXPERIMENT75_4_DPUSA_2026-11-10 -->"
    handoff=Path("/model/spromptgl/experiment/handoff.md")
    summary_map={(r["prompt_group"],r["setting"],r["method"],r["scope"],r["metric"]):r for r in summary}
    paired_map={(r.get("prompt_group"),r.get("setting"),r.get("comparison"),r.get("scope"),r.get("metric")):r for r in paired}
    def fmt(group,setting,method,metric,scope="all_10"):
        row=summary_map.get((group,setting,method,scope,metric))
        return "NA" if row is None else f"{float(row['mean']):.4f}±{float(row['population_sd']):.4f}"
    result_groups=["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8",
                   "P5-Functional-Network-8","P6-Heterogeneity-Aware-8","P1-SPromptGL-Anonymous-Gram-Matched",
                   "P5-Functional-Network-Anonymous-Gram-Matched","C3-Unrelated-English-8",
                   "RND-Unit-75451","RND-Unit-75452","RND-Unit-75453"]
    result_lines=["|Prompt/control|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|Log-loss|Brier|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in result_groups:
        result_lines.append("|"+group+"|"+"|".join(fmt(group,"B2-trained-alignment","M2-B2-BasePlusPrompt",m) for m in ("acc_pct","hard_auc_pct","prob_auc_pct","sen_pct","spe_pct","asd_f1_pct","log_loss","brier"))+"|")
    utility_lines=["|Prompt|M2−M0 log-loss gain|M2−M0 Prob-AUC pp|M2−M1 log-loss gain|M2−M1 Prob-AUC pp|","|---|---:|---:|---:|---:|"]
    for group in ["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]:
        vals=[]
        for contrast,metric in (("M2_vs_M0","log_loss"),("M2_vs_M0","prob_auc_pct"),("M2_vs_M1","log_loss"),("M2_vs_M1","prob_auc_pct")):
            r=paired_map.get((group,"B2",contrast,"all_10",metric))
            vals.append("NA" if r is None else f"{float(r['mean_improvement']):.5f} ({r['wins']}/{r['ties']}/{r['losses']})")
        utility_lines.append("|"+group+"|"+"|".join(vals)+"|")
    nine_lines=["|Prompt|M2−M0 log-loss gain|M2−M0 Prob-AUC pp|M2−M1 log-loss gain|M2−M1 Prob-AUC pp|","|---|---:|---:|---:|---:|"]
    for group in ["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]:
        vals=[]
        for contrast,metric in (("M2_vs_M0","log_loss"),("M2_vs_M0","prob_auc_pct"),("M2_vs_M1","log_loss"),("M2_vs_M1","prob_auc_pct")):
            r=paired_map.get((group,"B2",contrast,"exclude_fold2_descriptive_9",metric))
            vals.append("NA" if r is None else f"{float(r['mean_improvement']):.5f} ({r['wins']}/{r['ties']}/{r['losses']})")
        nine_lines.append("|"+group+"|"+"|".join(vals)+"|")
    token_summary=[]
    raw_token=read_csv(RESULTS/"token_diagnostics.csv")
    for group in ["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]:
        rows=[r for r in raw_token if r["prompt_group"]==group and r["diagnostic_type"]=="group_summary"]
        if rows:
            token_summary.append(f"{group}: train-bootstrap Top-4 Jaccard={np.mean([float(r['mean_within_group_prompt_top4_jaccard']) for r in rows if r['mean_within_group_prompt_top4_jaccard'] not in ('',None)]):.4f}; supervised Top-4 overlap={np.mean([float(r['top4_overlap_count']) for r in rows]):.3f}/4")
    token_ranked=sorted([r for r in token_rows if r.get("mean_importance_rank") is not None],key=lambda r:r["mean_importance_rank"])
    top_token_lines=[f"Token {r['token']} (slot {r['slot_origin']}): Prob-AUC={r['prob_auc_pct_mean']:.3f}±{r['prob_auc_pct_population_sd']:.3f}%, supervised Top-4 in {r['supervised_top4_fold_frequency']:.2f} folds" for r in token_ranked[:4]]
    geometry_lines=[]
    for row in semantic_geometry:
        if row.get("record_type")=="group_summary" and row["prompt_group"].startswith("P") and row["prompt_group"].split("-")[0] in ("P1","P2","P3","P4","P5","P6"):
            geometry_lines.append(f"{row['prompt_group']}: intra-ASD={row['mean_intra_asd_cosine']}, intra-TD={row['mean_intra_td_cosine']}, inter-class={row['mean_inter_class_cosine']}, centroid cosine={row['asd_td_centroid_cosine']}")
    c1_summary=[]
    for r in summary:
        if "WrongClass" in r["prompt_group"] and r["scope"]=="all_10" and r["metric"]=="acc_pct":
            auc=summary_map.get((r["prompt_group"],r["setting"],r["method"],"all_10","prob_auc_pct"))
            c1_summary.append(f"{r['prompt_group']} / {r['setting']} / {r['method']}: ACC={float(r['mean']):.4f}±{float(r['population_sd']):.4f}%, Prob-AUC={float(auc['mean']):.4f}±{float(auc['population_sd']):.4f}%" if auc else f"{r['prompt_group']} / {r['setting']} / {r['method']}: ACC={float(r['mean']):.4f}±{float(r['population_sd']):.4f}%")
    block=[marker,"", "## Experiment 75.4 — DPUSA", "",
           f"- 目的：检验原始及候选疾病Prompt能否在冻结E74.4-G2特征上提供超出基座校准分数与同容量无文本信号的预测效用；未重训DARG/CAM。正式目录：`{EXP}`。",
           f"- 基座：E74.4-G2-Positive-DCR；历史30/30训练和独立审计通过。冻结回放10/10通过，RN50 `{manifest['text_encoder']['weights_sha256']}`，Prompt编码与特征SHA见`manifest.json`。",
           "- Prompt：P1原始SPromptGL-8、P2官方BrainPrompt-2、P3类别定义、P4模态描述、P5功能网络描述、P6异质性描述；控制为P1/P5投影后Gram匹配匿名向量、C1错配类别、C3无关英文句及3个随机单位向量实例。",
           "- 全部结果为冻结特征上的外层十折探索性诊断。Fold 2中377/407行从所有新增探针训练/调参中排除；历史`val=test`、CV前预处理和871节点transductive图仍限制独立泛化解释。",
           f"- 冻结特征审计：{manifest['checkpoint_audit']}；Prompt/探针预测独立复算：{manifest['probe_prediction_audit']}；匿名Gram最大误差：{manifest['anonymous_max_gram_error']:.3g}。",
           "- 冻结表示监督基准：S0历史输出、M0基座log-odds内层校准、S1十六个单Token探针、S2完整H144和S3 DARG F320均包含在`summary.csv`。Fold 2新增探针排除了暴露行。",
           "- B2（训练集受限9→128对齐）各Prompt M2汇总：",*result_lines,
           "- B2 Prompt相对M0/M1的折级平均增量；括号为W/T/L：",*utility_lines,
           "- 排除Fold 2后的九折描述性B2增量；括号为W/T/L：",*nine_lines,
           "- 语义几何（CLIP原始1024维）：",*geometry_lines,
           "- 错误类别语义C1（固定分类器与重新拟合分开）：",*c1_summary,
           "- Token选择稳定性（训练集bootstrap）与监督Token Top-4重叠：",*token_summary,
           "- 16个单Token探针中监督重要性排名前四：",*top_token_lines,
           f"- 科研分类：**{manifest['scientific_conclusion']}**。十折/九折、B1/B2及负对照详见`{EXP}/results/report.md`、`{EXP}/results/summary.csv`、`{EXP}/results/incremental_utility.csv`、`{EXP}/results/paired_comparisons.csv`及`{EXP}/results/token_diagnostics.csv`。",
           "- 限制：B1随机投影空间与CLIP/CAM原空间不对齐；B2受限对齐会利用训练标签；P2每类1条文本；P4-P6未经专家验证；CORR索引无可审核的网络/ROI映射。结果不能单独证明疾病语义机制成立。",
           "- 下一步：只有当B2真实医学Prompt相对类别定义、匿名/随机文本和M1无文本容量匹配对照表现出稳定增量，且Token选择在训练重采样中稳定时，才值得将疾病Prompt保留为第二模块候选；否则优先采用无文本证据探针或停止语义分支。", "", marker, ""]
    addition="\n".join(block)
    lock=RT/"handoff.lock"
    try:
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open("a+b") as lf:
            fcntl.flock(lf.fileno(),fcntl.LOCK_EX)
            original=handoff.read_bytes()
            if marker.encode() in original:
                text=original.decode("utf-8")
                if "## Experiment 75.4 — DPUSA" not in text or "NO_DETECTABLE_PROMPT_BENEFIT" not in text:
                    raise RuntimeError("existing E75.4 handoff marker lacks the verified summary")
                digest=hashlib.sha256(original).hexdigest()
                return {"status":"PASS","path":str(handoff),"already_appended":True,
                        "original_sha256":digest,"updated_sha256":digest,
                        "append_marker_count":original.count(marker.encode())}
            suffix=("\n" if original and not original.endswith(b"\n") else "").encode()+addition.encode("utf-8")
            tmp=handoff.with_name(handoff.name+f".tmp.{os.getpid()}")
            with tmp.open("wb") as f:
                f.write(original); f.write(suffix); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,handoff)
            dfd=os.open(handoff.parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
            try: os.fsync(dfd)
            finally: os.close(dfd)
            after=handoff.read_bytes()
            if not after.startswith(original) or after.count(marker.encode())!=2:
                raise RuntimeError("handoff append persistence verification failed")
            return {"status":"PASS","path":str(handoff),"original_sha256":hashlib.sha256(original).hexdigest(),
                    "updated_sha256":hashlib.sha256(after).hexdigest(),"append_marker_count":2}
    except Exception as exc:
        (RESULTS/"handoff_append.md").write_text(addition,encoding="utf-8")
        return {"status":"FAILED","path":str(handoff),"error":repr(exc),"fallback":str(RESULTS/"handoff_append.md")}


def main():
    started=time.time()
    extraction=json.loads((RESULTS/"feature_extraction_audit.json").read_text(encoding="utf-8"))
    probe_exec=json.loads((RESULTS/"probe_execution.json").read_text(encoding="utf-8"))
    if probe_exec.get("status")!="PROBES_COMPLETE":
        raise RuntimeError("probe execution is incomplete")
    split_audit=verify_shared_inner_splits(probe_exec)
    refs=runner.verify_references(runner.base_runner())
    protected_before=extraction["protected_sources"]
    protected_current=runner.source_manifest()
    if protected_before!=protected_current:
        raise RuntimeError("a protected base/history source changed after feature extraction")
    if probe_exec["global_rng_before"]!=probe_exec["global_rng_after"]:
        raise RuntimeError("diagnostic probes changed process-global RNG")
    base_audits=[independent_base_readback(fold) for fold in range(1,11)]
    prompt_audit=verify_prompt_sources()
    pred_audit=audit_probe_predictions()
    gram_audit=independent_anonymous_gram_audit()
    max_gram=float(gram_audit["max_abs_error"])
    if max_gram>=1e-5:
        raise RuntimeError(f"anonymous Gram mismatch exceeds tolerance: {max_gram}")
    checkpoint_rows=[]
    for row in base_audits:
        checkpoint_rows.append({"fold":row["fold"],"status":row["status"],"checkpoint_sha256":row["checkpoint_sha256"],
                                "best_epoch_1based":row["best_epoch_1based"],"cam_calls":row["cam_calls"],
                                "outputlayer_calls":row["outputlayer_calls"],"rng_unchanged":row["rng_unchanged"],
                                "parameters_unchanged":row["parameters_unchanged"],"hard_prediction_exact":row["hard_prediction_exact"],
                                "probability_exact":row["probability_exact"],"train_test_indices_exact":row["train_test_indices_exact"],
                                "historical_overlap_1based":json.dumps(row["historical_overlap_1based"]),
                                "X320_exact":row["X320_exact"],"F320_exact":row["F320_exact"],"H144_exact":row["H144_exact"]})
    write_csv(RESULTS/"checkpoint_audit.csv",checkpoint_rows)
    utility=read_csv(RESULTS/"incremental_utility.csv")
    comparisons=paired_rows(utility)
    write_csv(RESULTS/"paired_comparisons.csv",comparisons)
    prompt_rows=read_csv(RESULTS/"prompt_comparison.csv")
    summaries=aggregate_metric_rows(prompt_rows)
    write_csv(RESULTS/"summary.csv",summaries)
    token_slots=token_summary()
    write_csv(RESULTS/"token_slot_summary.csv",token_slots)
    token_rows=supervised_token_summary(summaries)
    write_csv(RESULTS/"token_probe_summary.csv",token_rows)
    prompt_slots=prompt_similarity_summary()
    write_csv(RESULTS/"prompt_slot_alignment.csv",prompt_slots)
    semantic_geometry=semantic_geometry_rows()
    write_csv(RESULTS/"prompt_semantic_geometry.csv",semantic_geometry)
    # Conservative evidence classification from B2 full-base M2 utility and controls.
    def paired(group,comp,metric,scope="all_10"):
        return next((r for r in comparisons if r.get("prompt_group")==group and r.get("setting")=="B2" and r.get("comparison")==comp and r.get("metric")==metric and r.get("scope")==scope),None)
    true_groups=["P1-SPromptGL-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]
    gain_ll=[paired(g,"M2_vs_M0","log_loss") for g in true_groups]
    gain_vs_m1=[paired(g,"M2_vs_M1","log_loss") for g in true_groups]
    valid=[x for x in gain_ll+gain_vs_m1 if x]
    true_mean_ll=float(np.mean([float(x["mean_improvement"]) for x in valid])) if valid else 0.0
    p3_row=next((r for r in summaries if r["prompt_group"]=="P3-Class-Definition-8" and r["setting"]=="B2-trained-alignment" and r["method"]=="M2-B2-BasePlusPrompt" and r["scope"]=="all_10" and r["metric"]=="log_loss"),None)
    real_row=next((r for r in summaries if r["prompt_group"]=="P1-SPromptGL-8" and r["setting"]=="B2-trained-alignment" and r["method"]=="M2-B2-BasePlusPrompt" and r["scope"]=="all_10" and r["metric"]=="log_loss"),None)
    m1_row=next((r for r in summaries if r["prompt_group"]=="NO_TEXT" and r["setting"]=="incremental" and r["method"]=="M1-B2-Trainable-NoText" and r["scope"]=="all_10" and r["metric"]=="log_loss"),None)
    m0_row=next((r for r in summaries if r["prompt_group"]=="NO_TEXT" and r["setting"]=="incremental" and r["method"]=="M0-Calibrated-Base" and r["scope"]=="all_10" and r["metric"]=="log_loss"),None)
    comparison_banks=["P1-SPromptGL-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]
    true_vs_controls=[]
    for group in comparison_banks:
        references=["P3-Class-Definition-8","C3-Unrelated-English-8",
                    "RND-Unit-75451","RND-Unit-75452","RND-Unit-75453"]
        if group=="P1-SPromptGL-8": references.append("P1-SPromptGL-Anonymous-Gram-Matched")
        if group=="P5-Functional-Network-8": references.append("P5-Functional-Network-Anonymous-Gram-Matched")
        for reference in references:
            row=next((r for r in comparisons if r.get("prompt_group")==group and r.get("setting")=="B2" and
                      r.get("comparison")=="left_vs_reference" and r.get("reference_group")==reference and
                      r.get("scope")=="all_10" and r.get("metric")=="log_loss"),None)
            if row:
                true_vs_controls.append(row)
    controls_support=len(true_vs_controls)==22 and all(float(x["mean_improvement"])>0 and int(x["wins"])>int(x["losses"]) for x in true_vs_controls)
    base_support=bool(gain_ll) and all(float(x["mean_improvement"])>0 and int(x["wins"])>int(x["losses"]) for x in gain_ll if x)
    notext_support=bool(gain_vs_m1) and all(float(x["mean_improvement"])>0 and int(x["wins"])>int(x["losses"]) for x in gain_vs_m1 if x)
    class_reference=paired("P3-Class-Definition-8","M2_vs_M0","log_loss")
    specific_vs_class=[next((r for r in comparisons if r.get("prompt_group")==group and r.get("setting")=="B2" and
                              r.get("comparison")=="left_vs_reference" and r.get("reference_group")=="P3-Class-Definition-8" and
                              r.get("scope")=="all_10" and r.get("metric")=="log_loss"),None) for group in comparison_banks]
    specific_beats_class=bool(specific_vs_class) and all(r is not None and float(r["mean_improvement"])>0 and int(r["wins"])>int(r["losses"]) for r in specific_vs_class)
    no_text_rows=[r for r in read_csv(RESULTS/"incremental_utility.csv") if int(r["fold"]) in range(1,11) and r["prompt_group"]=="P1-SPromptGL-8" and r["setting"]=="B2" and r["model"] in ("M0","M1")]
    no_text_by={(int(r["fold"]),r["model"]):r for r in no_text_rows}
    no_text_diffs=[float(no_text_by[(fold,"M0")]["log_loss"])-float(no_text_by[(fold,"M1")]["log_loss"]) for fold in range(1,11) if (fold,"M0") in no_text_by and (fold,"M1") in no_text_by]
    no_text_gain=float(np.mean(no_text_diffs)) if no_text_diffs else 0.0
    no_text_wins=sum(x>1e-12 for x in no_text_diffs); no_text_losses=sum(x<-1e-12 for x in no_text_diffs)
    if base_support and notext_support and controls_support and max_gram<1e-5:
        conclusion="SEMANTIC_SIGNAL_SUPPORTED"
        note="Each preregistered real medical prompt bank improves B2 log-loss over calibrated base, matched no-text, class-definition, unrelated and all three random-vector references with majority fold wins; evidence remains exploratory and requires external validation."
    elif class_reference and float(class_reference["mean_improvement"])>0 and int(class_reference["wins"])>int(class_reference["losses"]) and not specific_beats_class:
        conclusion="CLASS_LABEL_ONLY"
        note="The class-definition prompt improves on the calibrated base, while the medical-attribute prompts do not consistently outperform the class-definition reference."
    elif no_text_gain>0 and no_text_wins>no_text_losses and max_gram<1e-5 and not controls_support:
        conclusion="GEOMETRY_OR_CAPACITY_ONLY"
        note=f"The matched B2 no-text local signal improves log-loss over calibrated base by {no_text_gain:.5f} on average ({no_text_wins}/10 fold wins), while medical text does not consistently exceed semantic controls; added local-feature capacity/transform remains a plausible source of utility."
    elif not valid or (p3_row is None):
        conclusion="INCONCLUSIVE"
        note="One or more predeclared comparison families could not be computed completely."
    else:
        conclusion="NO_DETECTABLE_PROMPT_BENEFIT"
        note="The real Prompt banks did not show a stable incremental advantage over calibrated-base and no-text controls under the registered probes."
    extraction_seconds=float(extraction["wall_seconds"])
    probe_seconds=float(probe_exec["elapsed_seconds"])
    analysis_sources={name:runner.sha_file(EXP/name) for name in ("config.json","prompt_banks.json","run.py","probe.py","audit.py")}
    frozen_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    manifest={"experiment":"Experiment 75.4 — DPUSA","status":"COMPLETE_AUDITED",
        "experiment_root":str(EXP),"preferred_root":runner.CFG["preferred_root"],"root_fallback_reason":runner.CFG.get("root_fallback_reason"),
        "analysis_only":True,"backbone_retrained":False,"base_group":runner.GROUP,
        "base_manifest_status":"COMPLETE_AUDITED","base_tasks":30,"base_independent_audits":30,
        "checkpoint_audit":"PASS 10/10","probe_prediction_audit":"PASS all prediction groups independently recomputed",
        "frozen_forward_count":10,"frozen_forward_per_fold":1,"cam_calls_per_fold":1,"outputlayer_calls_per_fold":1,
        "frozen_feature_audit":extraction,"probe_prediction_checks":pred_audit,"prompt_audit":prompt_audit,
        "text_encoder":{"name":"OpenAI CLIP RN50","weights_sha256":prompt_audit["RN50_sha256"],
                         "encoding":"E60 exact cached P1/P2 plus one offline encode pass for all new strings"},
        "anonymous_max_gram_error":max_gram,"anonymous_gram_audit":gram_audit,
        "probe_execution":probe_exec,"probe_wall_seconds":probe_seconds,"feature_extraction_seconds":extraction_seconds,
        "shared_inner_split_audit":split_audit,
        "device":f"{frozen_device} frozen feature extraction; CPU lightweight probes (one process)",
        "gpu_probe_workers":0,"probe_parallel_workers":1,
        "measured_stage_seconds":{"frozen_feature_extraction":extraction_seconds,"offline_prompt_encoding":"not separately timed in the initial encoding launcher","final_probe_run":probe_seconds,"independent_audit":time.time()-started},
        "measured_stage_total_seconds":extraction_seconds+probe_seconds+(time.time()-started),
        "analysis_source_sha256":analysis_sources,
        "analysis_failures":9,"analysis_failures_detail":[
            "Reference verification initially rejected E74.4's documented audit-only audit.py repair; the check was corrected to validate the training-source digest and all protected non-audit hashes.",
            "First frozen replay comparison used float64 exponentiation and failed the exact float32 probability check; replay validation now uses the base tensor's float32 path.",
            "First offline prompt-encoding launch could not resolve the E60 text_features module import; the repository path was added and RN50 encoding completed with the locked implementation.",
            "First probe attempt stopped on P1 display-key resolution before Prompt fits completed; the variant metadata lookup was corrected.",
            "Second probe attempt stopped on correct/incorrect selection-summary axis mismatch after partial fold-1 work; token-summary indexing was corrected.",
            "A partial six-fold probe run was stopped after source review showed that increment/reference probes used different inner-CV seeds. The final probe runner now uses one locked StratifiedKFold seed per outer fold for every model, prompt and negative control; partial outputs were never written or included.",
            "The first independent audit replay hit the historical E53 late set_num_interop_threads call after the audit process initialized PyTorch. The audit now temporarily bypasses that evaluation-irrelevant setter only during read-only prepare(fold), restores it immediately, and leaves historical source and model arithmetic unchanged.",
            "The first audit command used a non-existent run.py --phase audit option; it was redirected to the separate audit.py entry point without changing results.",
            "The first complete audit expected 40 anonymous-Gram rows although the runner records 60 (B1 and B2 anonymous pairs plus B2 source-projection consistency rows). The independent audit was corrected to recompute and verify all 60 rows; no metric or probe output changed."],
        "outer_folds":10,"fold2_train_test_overlap_1based":[377,407],"fold2_rows_excluded_from_new_probe_training_1based":[377,407],
        "source_manifest_sha256":runner.sha_object(protected_current),"protected_sources":protected_current,
        "scientific_conclusion":conclusion,"scientific_conclusion_note":note,
        "result_files":{"report":str(RESULTS/"report.md"),"summary":str(RESULTS/"summary.csv"),
                        "prompt_comparison":str(RESULTS/"prompt_comparison.csv"),"incremental_utility":str(RESULTS/"incremental_utility.csv"),
                        "paired_comparisons":str(RESULTS/"paired_comparisons.csv"),"token_diagnostics":str(RESULTS/"token_diagnostics.csv")},
        "elapsed_audit_seconds":time.time()-started}
    manifest["supervised_token_rows"]=token_rows
    build_report(summaries,comparisons,base_audits,token_slots,prompt_slots,semantic_geometry,manifest)
    handoff=handoff_append(manifest,summaries,comparisons,token_rows,semantic_geometry)
    manifest["handoff_update"]=handoff
    manifest["status"]="COMPLETE_AUDITED" if handoff["status"]=="PASS" else "COMPLETE_WITH_LIMITATIONS"
    manifest["independent_checkpoint_replays"]=base_audits
    runner.atomic_json(RESULTS/"audit_status.json",{"status":manifest["status"],"checkpoint_audit":"PASS 10/10",
                      "probe_prediction_audit":"PASS","prompt_audit":"PASS","protected_source_sha":"PASS",
                      "shared_inner_split_audit":split_audit,"anonymous_gram_audit":gram_audit,
                      "handoff":handoff})
    runner.atomic_json(EXP/"manifest.json",manifest)
    # Rewrite report with final handoff status and actual conclusion.
    build_report(summaries,comparisons,base_audits,token_slots,prompt_slots,semantic_geometry,manifest)
    print(json.dumps({"status":manifest["status"],"conclusion":conclusion,"checkpoint_audit":"10/10 PASS",
                      "probe_prediction_groups":pred_audit["prediction_groups_independently_recomputed"],
                      "handoff":handoff,"elapsed_audit_seconds":manifest["elapsed_audit_seconds"]},ensure_ascii=False))


if __name__=="__main__":
    main()
