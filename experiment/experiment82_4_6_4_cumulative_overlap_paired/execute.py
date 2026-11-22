"""Preflight, train and independently audit four cumulative-overlap tasks."""
from __future__ import annotations
import argparse, fcntl, os, subprocess, sys
from pathlib import Path
from core import (CONFIG,ROOT,RUNTIME,RESULTS,MANIFEST,TASKS,TASK_GROUPS,FOLDS,CANONICAL,engine,h,implementation,
                  paths,source_manifests,FULL_PATH)

def verify_guard(manifest):
    expected=manifest["protected_sha256"]; actual=h.archive.hash_files([Path(x) for x in expected])
    h.require(actual==expected,"Protected historical file changed")
    return {"status":"PASS","files":len(actual),"bundle_sha":h.sha(actual)}

def freeze_candidate():
    full=source_manifests()
    guard=dict(full["protected_sha256"])
    for name,sha in full["implementation"].items(): guard[str(FULL_PATH/name)]=sha
    for name,sha in full["result_files"].items(): guard[str(FULL_PATH/name)]=sha
    guard[str(FULL_PATH/"manifest.json")]=CONFIG["full_path_manifest_sha256"]
    prior=Path(CONFIG["prior_darg_only_directory"])
    pm=h.read_json(prior/"manifest.json")
    guard[str(prior/"manifest.json")]=h.sha_file(prior/"manifest.json")
    for name,sha in pm.get("implementation",{}).items(): guard[str(prior/name)]=sha
    for name in pm.get("result_files",{}): guard[str(prior/name)]=h.sha_file(prior/name)
    for name in ("manifest.json","config.json"):
        guard[str(prior/name)]=h.sha_file(prior/name)
    device=h.archive.verify_device()
    h.require((device["index"],device["uuid"])==(CONFIG["physical_gpu_index"],CONFIG["physical_gpu_uuid"]),
              "Physical GPU0 identity differs")
    source=implementation()
    m={"experiment":CONFIG["experiment"],"status":"PREFLIGHT_RUNNING","directory":str(ROOT),
       "config":CONFIG,"parent_training_config":h.hist_config(),"implementation":source,
       "implementation_sha":h.sha(source),"protected_sha256":guard,"device":device,
       "packages":h.archive.packages(),"expected_tasks":[f"{g}_f{f:02d}" for g,f in TASKS],
       "guard_before":None}
    m["guard_before"]=verify_guard(m)
    return m

def preflight(manifest):
    from independent_464 import preflight_task
    checked={}
    for task,fold in TASKS:
        file=paths(task,fold)["preflight"]
        result=h.read_json(file) if file.exists() else preflight_task(task,fold,manifest["implementation"])
        if not file.exists(): h.write_json(file,result)
        h.require(result["status"]=="PASS" and result["implementation"]==manifest["implementation"]
                  and result["device"]["uuid"]==manifest["device"]["uuid"],"Task preflight identity differs")
        checked[f"{task}_f{fold:02d}"]={"sha":h.sha_file(file),"peak_reserved_mib":result["peak_reserved_mib"]}
        print(f"PREFLIGHT {task} Fold {fold} PASS: native {result['membership']['native_rows']} legal {result['membership']['legal_rows']}",flush=True)
    dev=h.archive.verify_device(); free=dev["total_mib"]-dev["used_mib"]; need=sum(x["peak_reserved_mib"] for x in checked.values())
    h.require(free>max(x["peak_reserved_mib"] for x in checked.values())+1024,"Insufficient GPU0 memory")
    manifest.update(status="PREFLIGHT_PASSED",preflight={"status":"PASS","tasks":checked,"sequential":True},
                    worker_plan={"concurrent":min(CONFIG["max_workers"],len(TASKS)) if free>=need+2048 else 1,
                                 "free_mib":free,"sum_preflight_peaks_mib":need},frozen_before_training=True)
    h.write_json(MANIFEST,manifest)
    print(f"FROZEN {manifest['implementation_sha']} | GPU0 | {manifest['worker_plan']['concurrent']} workers",flush=True)

def train(manifest):
    tasks=list(TASKS); cpus=dict(zip(tasks,CONFIG["worker_cpu_candidates"],strict=True))
    h.require(set(cpus.values()).issubset(os.sched_getaffinity(0)),"Worker CPU affinity unavailable")
    def complete(task,fold):
        files=paths(task,fold)
        if not files["task"].exists(): return False
        t=h.read_json(files["task"])
        h.require(t["status"]=="TRAINED" and t["implementation"]==manifest["implementation"]
                  and all(h.sha_file(files[k])==t[k+"_sha"] for k in ("best","latest","journal")),
                  "Completed task artifact damaged")
        return True
    outstanding=[x for x in tasks if not complete(*x)]
    count=manifest["worker_plan"]["concurrent"]; manifest["status"]="TRAINING"; h.write_json(MANIFEST,manifest)
    for start in range(0,len(outstanding),count):
        workers={}
        try:
            for task,fold in outstanding[start:start+count]:
                workers[(task,fold)]=subprocess.Popen(["taskset","-c",str(cpus[(task,fold)]),sys.executable,"-u",
                    str(ROOT/"execute.py"),"--worker",task,"--fold",str(fold)],cwd=ROOT)
            codes={k:p.wait() for k,p in workers.items()}; h.require(all(x==0 for x in codes.values()),f"Workers stopped {codes}")
        finally:
            for p in workers.values():
                if p.poll() is None: p.terminate(); p.wait()
    h.require(all(complete(*x) for x in tasks),"Formal task checkpoint missing")
    manifest.update(status="TRAINED",completed_tasks={f"{g}_f{f:02d}":h.read_json(paths(g,f)["task"]) for g,f in tasks})
    h.write_json(MANIFEST,manifest)

def audit_only():
    from independent_464 import audit
    from report import publish
    manifest=h.read_json(MANIFEST); h.require(manifest["status"] in ("TRAINED","AUDIT_FAILED"),"Training incomplete")
    try:
        outputs=audit(manifest); manifest["guard_after"]=verify_guard(manifest); publish(manifest,*outputs)
    except Exception as e:
        manifest.update(status="AUDIT_FAILED",last_error=f"{type(e).__name__}: {e}"); h.write_json(MANIFEST,manifest)
        h.write_json(RESULTS/"audit_status.json",{"status":"AUDIT_FAILED","error":manifest["last_error"]}); raise

def orchestrate():
    for d in (RUNTIME,RESULTS,ROOT/"checkpoints"): d.mkdir(exist_ok=True)
    with (RUNTIME/"execute.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if MANIFEST.exists():
            m=h.read_json(MANIFEST); h.require(m["implementation"]==implementation() and m["config"]==CONFIG,"Frozen source/config differs")
            if m["status"]=="COMPLETE_AUDITED": verify_guard(m); print("COMPLETE_AUDITED existing",flush=True); return
        else:
            m=freeze_candidate(); h.write_json(MANIFEST,m)
        if m["status"] not in ("TRAINED","AUDIT_FAILED"):
            try: preflight(m)
            except Exception as e:
                m.update(status="PREFLIGHT_FAILED",last_error=f"{type(e).__name__}: {e}"); h.write_json(MANIFEST,m); raise
            try: train(m)
            except (Exception,KeyboardInterrupt) as e:
                m.update(status="INTERRUPTED",last_error=f"{type(e).__name__}: {e}"); h.write_json(MANIFEST,m); raise
        subprocess.run([sys.executable,"-u",str(ROOT/"execute.py"),"--audit-only"],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group()
    g.add_argument("--worker",choices=TASK_GROUPS); p.add_argument("--fold",type=int,choices=FOLDS)
    g.add_argument("--audit-only",action="store_true"); a=p.parse_args()
    if a.worker:
        h.require(a.fold in FOLDS and len(os.sched_getaffinity(0))==1,"Worker affinity/task invalid")
        h.archive.verify_device(); engine(a.worker,a.fold).train_fold(CANONICAL,a.fold)
    elif a.audit_only: audit_only()
    else: orchestrate()
if __name__=="__main__": main()
