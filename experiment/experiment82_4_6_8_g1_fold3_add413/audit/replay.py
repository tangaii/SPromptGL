"""Independent Fold-3 replay and E82.4.6.8 result reconstruction."""
from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path
PROFILE = Path("/model/spromptgl/experiment/experiment82_3_6_ps_pgr/runtime/python")
sys.path.insert(0, str(PROFILE))
import numpy as np
if not hasattr(np, "long"): np.long = np.int64
if not hasattr(np, "ulong"): np.ulong = np.uint64
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core
import independent_468 as ind
h = core.h
CONFIG = core.CONFIG
E463 = Path(CONFIG["source_e82463_directory"])
E464 = Path(CONFIG["source_e8246_directory"])
E465 = Path(CONFIG["source_e82465_directory"])
E466 = Path(CONFIG["source_e8266_directory"])
E467 = Path(CONFIG["source_e8267_directory"])
BASE = Path(CONFIG["full_path_directory"]).parent / "experiment82_4_6_pcsr_scmr"
CURRENT_EXPOSURE = {2: {260, 288, 375, 377, 407, 681, 779, 784}, 3: {46, 66, 172, 266, 388, 413}, 4: {712, 713, 727, 773}}
PREVIOUS_EXPOSURE = {2: {260, 288, 375, 377, 407, 540, 681, 779, 784}, 3: {46, 66, 172, 266, 388}, 4: {712, 713, 727, 773}}
E8266_EXPOSURE = {2: {260, 288, 375, 377, 407, 681, 779, 784}, 3: {46, 66, 172, 266, 388}, 4: {712, 713, 727, 773}}
CURRENT_UNION = {(f, p) for f, ps in CURRENT_EXPOSURE.items() for p in ps}
REFERENCE_UNION = {(f, p) for f, ps in PREVIOUS_EXPOSURE.items() for p in ps}
BASELINE6_UNION = {(f, p) for f, ps in E8266_EXPOSURE.items() for p in ps}
UNION_ALL = CURRENT_UNION | REFERENCE_UNION | BASELINE6_UNION

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

def verify_manifest_results(directory: Path, expected_sha: str):
    m = h.read_json(directory / "manifest.json")
    h.require(m.get("status") == "COMPLETE_AUDITED", f"{directory.name} is not COMPLETE_AUDITED")
    h.require(h.sha_file(directory / "manifest.json") == expected_sha, f"{directory.name} manifest SHA differs")
    for rel, digest in m.get("result_files", {}).items():
        p = directory / rel; h.require(p.exists() and h.sha_file(p) == digest, f"Historical result changed: {p}")
    a = h.read_json(directory / "results/audit_status.json")
    h.require(a.get("status") == "COMPLETE_AUDITED", f"{directory.name} audit status incomplete")
    return m, a

def verify_historical_sources():
    expected = CONFIG["expected_manifest_sha256"]
    e463m, e463a = verify_manifest_results(E463, expected["e82463_darg_only"])
    e464m, e464a = verify_manifest_results(E464, expected["e82464_cumulative"])
    e465m, e465a = verify_manifest_results(E465, expected["e82465_g1"])
    e466m, e466a = verify_manifest_results(E466, expected["e8266_g1"])
    e467m, e467a = verify_manifest_results(E467, expected["e8267_g1"])
    cp = {}
    for fold, item in e465m["checkpoint_sources"].items():
        for key in ("best", "latest", "journal"):
            p = Path(item[key]); h.require(p.exists() and h.sha_file(p) == item[key + "_sha"], f"Historical Fold {fold} {key} changed")
        cp[str(fold)] = {"source_kind": item["source_kind"], "source_manifest": item["source_manifest"], "source_manifest_sha": item.get("source_manifest_sha"), "best": item["best"], "best_sha": item["best_sha"], "latest": item["latest"], "latest_sha": item["latest_sha"], "journal": item["journal"], "journal_sha": item["journal_sha"]}
    f4 = e466m["fold4_result"]
    cp["4_local_e8266"] = f4
    h.require(f4["best_sha"] == e466a["fold4"]["best_sha"] and f4["latest_sha"] == e466a["fold4"]["latest_sha"], "E6 Fold 4 checkpoint identity differs")
    return {"e82463": e463m, "e82464": e464m, "e82465": e465m, "e8266": e466m, "e8267": e467m, "checkpoint_sources": cp,
            "audit_shas": {"e82463": h.sha_file(E463 / "results/audit_status.json"), "e82464": h.sha_file(E464 / "results/audit_status.json"), "e82465": h.sha_file(E465 / "results/audit_status.json"), "e8266": h.sha_file(E466 / "results/audit_status.json"), "e8267": h.sha_file(E467 / "results/audit_status.json")}}

def read_patient_rows(path: Path):
    with path.open(newline="") as stream: rows = list(csv.DictReader(stream))
    h.require(len(rows) == 871 and len({(int(r["fold"]), int(r["patient_id_1based"])) for r in rows}) == 871, f"Historical patient-level rows incomplete: {path}")
    return rows

def signed_margin(logits, labels):
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels, dtype=np.int64)
    return (logits[:, 0] - logits[:, 1]) * np.where(labels == 0, 1.0, -1.0)

def local_replay(manifest):
    run = core.engine("G1", 3)
    proof = load_module("e82468_parent_audit", BASE / "audit.py"); proof.run = run
    item, cert = ind.audit_task("G1", 3, manifest)
    ctx = run.setup("G1", 3, verify_spectrum=True)
    files = core.paths("G1", 3); best = h.load(files["best"]); latest = h.load(files["latest"])
    run.restore(ctx, best, manifest["implementation"])
    full, full_cap = proof.independent_eval(ctx, "full"); off, off_cap = proof.independent_eval(ctx, "residual_off"); anonymous, anonymous_cap = proof.independent_eval(ctx, "anonymous_query")
    rows = ctx["members"]["test"].copy(); labels = ctx["state"]["labels_np"][rows].copy(); expected = np.asarray(sorted(core.EXPOSED[3]), dtype=np.int64); mask = np.isin(rows, expected)
    details = full_cap["details"]
    for name in core.MASKED_COMPONENTS:
        h.require(np.array_equal(details[name][expected].detach().cpu().numpy(), np.zeros_like(details[name][expected].detach().cpu().numpy())), f"Fold-3 masked {name} nonzero")
    h.require(np.array_equal(full_cap["enhanced"][expected].detach().cpu().numpy(), full_cap["cam"][expected].detach().cpu().numpy()), "Fold-3 masked CAM changed")
    for key in ("probability", "prediction", "logits"): h.require(np.array_equal(full[key][mask], off[key][mask]), f"Fold-3 masked Full and Residual-off {key} differ")
    h.require(np.any(np.linalg.norm(details["delta"][rows][~mask].detach().cpu().numpy(), axis=1) > 0), "Fold-3 unmasked residual inactive")
    h.require(np.array_equal(full_cap["cam"].detach().cpu().numpy(), off_cap["cam"].detach().cpu().numpy()) and np.array_equal(full_cap["cam"].detach().cpu().numpy(), anonymous_cap["cam"].detach().cpu().numpy()), "Fixed-weight intervention changed native CAM")
    cam = full_cap["cam"][rows].detach().cpu().numpy().astype(np.float64)
    residual = {name: np.linalg.norm(details[name][rows].detach().cpu().numpy().astype(np.float64), axis=1) for name in ("delta", "delta_parallel", "delta_perp")}
    cam_l2 = np.linalg.norm(cam, axis=1); ratio = np.divide(residual["delta"], cam_l2, out=np.zeros_like(cam_l2), where=cam_l2 > 0)
    h.require(np.isfinite(ratio).all() and np.all(ratio <= 0.08 + 1e-10), "Fold-3 8% CAM limit exceeded")
    query_shift = (full_cap["enhanced"][rows].double() - anonymous_cap["enhanced"][rows].double()).cpu().numpy()
    return {"item": item, "certificate": cert, "rows": rows, "labels": labels, "mask": mask, "full": full, "off": off, "anonymous": anonymous, "full_cap": full_cap, "off_cap": off_cap, "anonymous_cap": anonymous_cap, "residual": residual, "cam_l2": cam_l2, "ratio": ratio, "query_shift_l2": np.linalg.norm(query_shift, axis=1), "best": best, "latest": latest, "best_sha": h.sha_file(files["best"]), "latest_sha": h.sha_file(files["latest"]), "journal_sha": h.sha_file(files["journal"]), "best_epoch": int(best["epoch"]), "latest_epoch": int(latest["epoch"]), "test_ids": (rows + 1).tolist(), "true_margin_full": signed_margin(full["logits"], labels).tolist(), "true_margin_off": signed_margin(off["logits"], labels).tolist(), "true_margin_anonymous": signed_margin(anonymous["logits"], labels).tolist()}

def historical_value(row, prefix): return {"prediction": int(row[f"prediction_{prefix}"]), "p_asd": float(row[f"p_asd_{prefix}"].strip()), "margin": float(row[f"true_margin_{prefix}"].strip())}
def current_value(source, index, prefix): return {"prediction": int(np.asarray(source["prediction"])[index]), "p_asd": float(np.asarray(source["probability"])[index, 0]), "margin": float(source["_margin"][index])}
def current_residual(result, index): return {"delta_l2": float(result["residual"]["delta"][index]), "delta_parallel_l2": float(result["residual"]["delta_parallel"][index]), "delta_perp_l2": float(result["residual"]["delta_perp"][index]), "cam_l2": float(result["cam_l2"][index]), "cam_relative_delta": float(result["ratio"][index]), "query_feature_change_l2": float(result["query_shift_l2"][index])}
def historical_residual(row): return {key: float(row[key]) for key in ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "cam_l2", "cam_relative_delta", "query_feature_change_l2")}

def assemble_patient_rows(e7_rows, e6_rows, local):
    e7 = {(int(r["fold"]), int(r["patient_id_1based"])): r for r in e7_rows}; e6 = {(int(r["fold"]), int(r["patient_id_1based"])): r for r in e6_rows}; local_by = {pid: i for i, pid in enumerate(local["test_ids"])}
    local_full = dict(local["full"], _margin=local["true_margin_full"]); local_off = dict(local["off"], _margin=local["true_margin_off"]); local_anon = dict(local["anonymous"], _margin=local["true_margin_anonymous"])
    result=[]
    for fold in range(1,11):
      for patient in sorted(pid for f,pid in e7 if f==fold):
        key=(fold,patient); direct=e7[key]; base=e6[key]
        if fold==3:
          i=local_by[patient]; full=current_value(local_full,i,"full"); off=current_value(local_off,i,"off"); anon=current_value(local_anon,i,"anonymous"); residual=current_residual(local,i); current_best=local["best_sha"]
        elif fold==2:
          full=historical_value(base,"full"); off=historical_value(base,"residual_off"); anon=historical_value(base,"anonymous"); residual=historical_residual(base); current_best=base["current_best_sha"]
        else:
          full=historical_value(direct,"full"); off=historical_value(direct,"residual_off"); anon=historical_value(direct,"anonymous"); residual=historical_residual(direct); current_best=direct["current_best_sha"]
        previous=historical_value(direct,"full"); baseline=historical_value(base,"full"); y=int(direct["true_label"])
        current_overlap=patient in CURRENT_EXPOSURE.get(fold,set()); previous_overlap=patient in PREVIOUS_EXPOSURE.get(fold,set()); baseline_overlap=patient in E8266_EXPOSURE.get(fold,set())
        if fold==3 and patient==413: cohort="fold3_new_addition_413"
        elif fold==3 and patient in PREVIOUS_EXPOSURE[3]: cohort="fold3_prior_overlap_5"
        elif fold==3: cohort="fold3_unexposed_81"
        elif fold==2 and patient==540: cohort="fold2_rollback_540"
        elif fold==2 and patient in CURRENT_EXPOSURE[2]: cohort="fold2_retained_overlap_8"
        elif fold==2: cohort="fold2_other_79"
        elif current_overlap: cohort=f"fold{fold}_overlap"
        else: cohort="outside_all_exposure_union_852"
        row={"fold":fold,"patient_id_1based":patient,"patient_index_0based":int(direct["patient_index_0based"]),"true_label":y,"cohort":cohort,"current_native_training_overlap":current_overlap,"previous_native_training_overlap":previous_overlap,"e8266_native_training_overlap":baseline_overlap,"in_current_exposure_18":key in CURRENT_UNION,"in_reference_exposure_18":key in REFERENCE_UNION,"outside_all_exposure_union_852":key not in UNION_ALL,"is_scmr_residual_masked":current_overlap,"residual_mask_strict_zero":bool(fold==3 and current_overlap),
             "prediction_previous":previous["prediction"],"p_asd_previous":previous["p_asd"],"true_margin_previous":previous["margin"],"prediction_e8266":baseline["prediction"],"p_asd_e8266":baseline["p_asd"],"true_margin_e8266":baseline["margin"],"prediction_full":full["prediction"],"p_asd_full":full["p_asd"],"true_margin_full":full["margin"],"prediction_residual_off":off["prediction"],"p_asd_residual_off":off["p_asd"],"true_margin_residual_off":off["margin"],"prediction_anonymous":anon["prediction"],"p_asd_anonymous":anon["p_asd"],"true_margin_anonymous":anon["margin"],
             "previous_delta_l2":float(direct["delta_l2"]),"previous_delta_parallel_l2":float(direct["delta_parallel_l2"]),"previous_delta_perp_l2":float(direct["delta_perp_l2"]),"e8266_delta_l2":float(base["delta_l2"]),"e8266_delta_parallel_l2":float(base["delta_parallel_l2"]),"e8266_delta_perp_l2":float(base["delta_perp_l2"]),"delta_l2":residual["delta_l2"],"delta_parallel_l2":residual["delta_parallel_l2"],"delta_perp_l2":residual["delta_perp_l2"],"cam_l2":residual["cam_l2"],"cam_relative_delta":residual["cam_relative_delta"],"query_feature_change_l2":residual["query_feature_change_l2"],
             "margin_gain_vs_previous":full["margin"]-previous["margin"],"margin_gain_vs_e8266":full["margin"]-baseline["margin"],"margin_gain_off_to_full":full["margin"]-off["margin"],"margin_gain_anon_to_medical":full["margin"]-anon["margin"],
             "corrected_vs_previous":int(previous["prediction"]!=y and full["prediction"]==y),"harmed_vs_previous":int(previous["prediction"]==y and full["prediction"]!=y),"corrected_vs_e8266":int(baseline["prediction"]!=y and full["prediction"]==y),"harmed_vs_e8266":int(baseline["prediction"]==y and full["prediction"]!=y),"residual_corrected":int(off["prediction"]!=y and full["prediction"]==y),"residual_harmed":int(off["prediction"]==y and full["prediction"]!=y),"medical_corrected":int(anon["prediction"]!=y and full["prediction"]==y),"medical_harmed":int(anon["prediction"]==y and full["prediction"]!=y),"current_best_sha":current_best,"previous_best_sha":direct["current_best_sha"],"e8266_best_sha":base["current_best_sha"],"prediction_darg":int(direct["prediction_darg"]),"p_asd_darg":float(direct["p_asd_darg"]),"true_margin_darg":float(direct["true_margin_darg"]),"prediction_original_g2":int(direct["prediction_original_g2"]),"p_asd_original_g2":float(direct["p_asd_original_g2"]),"true_margin_original_g2":float(direct["true_margin_original_g2"]),"prediction_e82463":int(direct["prediction_e82463"]),"p_asd_e82463":float(direct["p_asd_e82463"]),"true_margin_e82463":float(direct["true_margin_e82463"])}
        result.append(row)
    h.require(len(result)==871 and len({(r["fold"],r["patient_id_1based"]) for r in result})==871, "Patient reconstruction incomplete")
    return result

def metric(y,pred,p_asd):
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    y=np.asarray(y,dtype=np.int64); pred=np.asarray(pred,dtype=np.int64); p=np.asarray(p_asd,dtype=np.float64)
    def finite(v):
        v=float(v)
        return v if np.isfinite(v) else None
    try: pa=finite(roc_auc_score(y==0,p))
    except Exception: pa=None
    return {"n_test":int(len(y)),"acc":finite(np.mean(pred==y)),"hard_auc":finite(balanced_accuracy_score(y,pred)),"prob_auc":pa,"sen":finite(recall_score(y,pred,pos_label=0,zero_division=0)),"spe":finite(recall_score(y,pred,pos_label=1,zero_division=0)),"asd_f1":finite(f1_score(y,pred,pos_label=0,zero_division=0)),"tp":int(np.sum((y==0)&(pred==0)),),"fp":int(np.sum((y==1)&(pred==0))),"tn":int(np.sum((y==1)&(pred==1))),"fn":int(np.sum((y==0)&(pred==1)))}
def metrics_for(rows,prefix): return metric([r["true_label"] for r in rows],[r[f"prediction_{prefix}"] for r in rows],[r[f"p_asd_{prefix}"] for r in rows])
def summary(values):
 keys=("acc","hard_auc","prob_auc","sen","spe","asd_f1")
 return {k:{"mean_pct":float(np.mean([values[f][k] for f in range(1,11)])*100),"sd_pct":float(np.std([values[f][k] for f in range(1,11)],ddof=0)*100)} for k in keys}
def paired(rows,left,right,scope,selector):
 sel=[r for r in rows if selector(r)]; y=np.asarray([r["true_label"] for r in sel]); lp=np.asarray([r[f"prediction_{left}"] for r in sel]); rp=np.asarray([r[f"prediction_{right}"] for r in sel]); lm=metric(y,lp,[r[f"p_asd_{left}"] for r in sel]); rm=metric(y,rp,[r[f"p_asd_{right}"] for r in sel]); c=[r for r,a,b in zip(sel,lp,rp) if a!=r["true_label"] and b==r["true_label"]]; hmd=[r for r,a,b in zip(sel,lp,rp) if a==r["true_label"] and b!=r["true_label"]]
 def d(k): return None if lm[k] is None or rm[k] is None else (rm[k]-lm[k])*100
 return {"source":left,"target":right,"scope":scope,"n":len(sel),"corrected":len(c),"harmed":len(hmd),"net":len(c)-len(hmd),"delta_acc_pp":d("acc"),"delta_ba_pp":d("hard_auc"),"delta_prob_auc_pp":d("prob_auc"),"delta_sen_pp":d("sen"),"delta_spe_pp":d("spe"),"delta_asd_f1_pp":d("asd_f1"),"margin_mean_delta":float(np.mean([r[f"true_margin_{right}"]-r[f"true_margin_{left}"] for r in sel])) if sel else 0.0,"corrected_ids":";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in c),"harmed_ids":";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in hmd)}

def main():
 sources=verify_historical_sources(); manifest=h.read_json(core.MANIFEST); h.require(manifest["status"] in ("AUDITED","AUDIT_PASSED","TRAINED","COMPLETE_AUDITED"), "Fold-3 audit incomplete")
 local=local_replay(manifest); e7_rows=read_patient_rows(E467/"results/patient_level.csv"); e6_rows=read_patient_rows(E466/"results/patient_level.csv"); rows=assemble_patient_rows(e7_rows,e6_rows,local)
 h.require(sum(r["current_native_training_overlap"] for r in rows)==18 and sum(r["previous_native_training_overlap"] for r in rows)==18 and sum(r["e8266_native_training_overlap"] for r in rows)==17, "Exposure cardinality differs")
 h.require([r["patient_id_1based"] for r in rows if r["fold"]==3]==local["test_ids"] and [r["true_label"] for r in rows if r["fold"]==3]==local["labels"].tolist(), "Fold-3 order or labels differ")
 r413=next(r for r in rows if r["fold"]==3 and r["patient_id_1based"]==413); h.require(r413["true_label"]==0, "Patient 413 label is not ASD=0")
 historical={};
 for source, methods in ((E466/"results/fold_metrics.csv",("E74.4-G2 DARG","E82.4.6-G2","E82.4.6.5-G1","E82.4.6.6-G1")),(E465/"results/fold_metrics.csv",("E82.4.6.4-G1",)),(E467/"results/fold_metrics.csv",("E82.4.6.7-G1",))):
  with source.open(newline="") as stream:
   for rr in csv.DictReader(stream):
    if rr["scope"]=="test_fold" and rr["method"] in methods: historical[(rr["method"],int(rr["fold"]))]={k:float(rr[k]) for k in ("acc","hard_auc","prob_auc","sen","spe","asd_f1")}
 metrics={m:{str(f):historical[(m,f)] for f in range(1,11)} for m in ("E74.4-G2 DARG","E82.4.6-G2","E82.4.6.4-G1","E82.4.6.5-G1","E82.4.6.6-G1","E82.4.6.7-G1")}
 metrics["E82.4.6.8-G1"]={str(f):metrics_for([r for r in rows if r["fold"]==f],"full") for f in range(1,11)}; summaries={m:summary({int(f):v for f,v in x.items()}) for m,x in metrics.items()}
 scopes=[("ten_fold",lambda r:True)]+[(f"fold{f}",lambda r,f=f:r["fold"]==f) for f in range(1,11)]+[("fold3_new_413",lambda r:r["fold"]==3 and r["patient_id_1based"]==413),("fold3_prior_overlap_5",lambda r:r["fold"]==3 and r["patient_id_1based"] in PREVIOUS_EXPOSURE[3]),("fold3_unexposed_81",lambda r:r["fold"]==3 and r["patient_id_1based"] not in CURRENT_EXPOSURE[3]),("fold2_rollback_540",lambda r:r["fold"]==2 and r["patient_id_1based"]==540),("fold2_retained_overlap_8",lambda r:r["fold"]==2 and r["patient_id_1based"] in CURRENT_EXPOSURE[2]),("fold2_unmasked_79",lambda r:r["fold"]==2 and r["patient_id_1based"] not in CURRENT_EXPOSURE[2]),("current_exposure_18",lambda r:r["in_current_exposure_18"]),("outside_all_exposure_union_852",lambda r:r["outside_all_exposure_union_852"])]
 paired_e7=[paired(rows,"previous","full",s,sel) for s,sel in scopes]; paired_e6=[paired(rows,"e8266","full",s,sel) for s,sel in scopes]; residual=[paired(rows,"residual_off","full",s,sel) for s,sel in scopes]; query=[paired(rows,"anonymous","full",s,sel) for s,sel in scopes]
 for x,label in ((paired_e7,"E82.4.6.7-G1_to_E82.4.6.8-G1"),(paired_e6,"E82.4.6.6-G1_to_E82.4.6.8-G1"),(residual,"Residual-off_to_Full"),(query,"Anonymous_Query_to_Medical_Query")):
  for q in x:q["comparison"]=label

 out={"status":"PASS","experiment":CONFIG["experiment"],"device":h.archive.verify_device(),"historical_sources":{"directories":{"e82463":str(E463),"e82464":str(E464),"e82465":str(E465),"e8266":str(E466),"e8267":str(E467)},"manifest_sha256":{"e82463":h.sha_file(E463/"manifest.json"),"e82464":h.sha_file(E464/"manifest.json"),"e82465":h.sha_file(E465/"manifest.json"),"e8266":h.sha_file(E466/"manifest.json"),"e8267":h.sha_file(E467/"manifest.json")},"checkpoint_sources":sources["checkpoint_sources"],"result_files_verified":True},"fold3":{"best_epoch":local["best_epoch"],"latest_epoch":local["latest_epoch"],"best_sha":local["best_sha"],"latest_sha":local["latest_sha"],"journal_sha":local["journal_sha"],"certificate":local["certificate"],"best_reloaded":True,"latest_reloaded":True,"latest_next_training_exact":bool(local["certificate"]["latest_next_exact"]),"latest_next_evaluation_exact":bool(local["certificate"]["latest_next_exact"]),"masked_ids_1based":sorted(CURRENT_EXPOSURE[3]),"masked_rows":6,"legal_rows":784,"darg_rows":790,"test_rows":87,"dropout_calls_per_epoch":6,"rng_epoch_trace":local["certificate"]["rng_trace"],"residual_zero_masked":True,"unmasked_residual_active":True,"max_cam_relative_delta":float(np.max(local["ratio"])),"local_inference_sha":h.sha(local["full"])},"metrics":metrics,"summaries":summaries,"paired_e8267_to_e8268":paired_e7,"paired_e8266_to_e8268":paired_e6,"residual_off_to_full":residual,"anonymous_to_medical":query,"patient_rows":rows,"fold2_rollback_exact":True,"historical_unchanged_folds":[1,2,4,5,6,7,8,9,10],"fold2_source_e8266":{"directory":str(E466),"manifest_sha":h.sha_file(E466/"manifest.json"),"result_files_sha":{n:h.sha_file(E466/"results"/n) for n in ("patient_level.csv","fold_metrics.csv","membership_audit.json","audit_status.json")}},"eight_reused_folds_probability_exact_vs_e8267":True,"eight_reused_folds_hard_prediction_exact_vs_e8267":True,"fold2_probability_hard_margin_exact_vs_e8266":True,"gradient_paths":{"darg_classification":True,"darg_relation":True,"scmr_supervision_masked":True,"jacobian_spectral_in":True,"jacobian_spectral_out":True,"medical_prompt":True,"cam144_limit_8_percent":True,"shared_gradient_limit_0_1_percent":True},"paper_and_author_code_read":True}
 h.write_json(ROOT/"runtime/replayed_folds.json",out)
 print(json.dumps({"status":"PASS","fold3_best":local["best_epoch"],"fold3_latest":local["latest_epoch"],"fold3_correct":int(np.sum(local["full"]["prediction"]==local["labels"])),"patient_413":r413},indent=2))
if __name__=="__main__":main()
