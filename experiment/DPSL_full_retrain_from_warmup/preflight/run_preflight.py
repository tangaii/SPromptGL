from pathlib import Path
import sys, os, json, hashlib
SRC=Path("/model/spromptgl/experiment/DPSL")
OUT=Path("/model/spromptgl/experiment/DPSL_full_retrain_from_warmup")
sys.dont_write_bytecode=True
sys.path[:0]=[str(SRC/name) for name in ("code","runtime/python","code/history_base")]
# Restrict project tree accesses to read-only DPSL inputs/source and this run output.
import builtins
_ALLOWED_READ=SRC.resolve()
_ALLOWED_OUT=OUT.resolve()
_PROTECTED=(Path("/model/spromptgl").resolve(),Path("/data/adodas/SPromptGL-main").resolve())
def _audit(event,args):
    if event=="open" and args:
        raw=args[0]
        if isinstance(raw,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(raw))
            try: a=p.resolve()
            except OSError: return
            if any(a.is_relative_to(root) for root in _PROTECTED):
                flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
                mode=args[1] if len(args)>1 and isinstance(args[1],str) else "r"
                writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or any(c in mode for c in "wax+")
                allowed=(a.is_relative_to(_ALLOWED_READ) and not writing) or a.is_relative_to(_ALLOWED_OUT)
                if not allowed: raise PermissionError("project path access outside DPSL read-only base and this run output")
sys.addaudithook(_audit)
from training_runtime import fold_engine, history_record, run_fold
from real_model_runtime import optimizer_identity, validate_evaluation
from dpsl_runtime import verify_warmup_file, sha256_file
import torch

def check_initial(fold):
    engine=fold_engine(fold)
    pre=history_record(fold,"preflight")
    ctx=engine.setup("G2",fold)
    assert ctx["provenance"]==pre["provenance"],f"fold {fold} provenance mismatch"
    assert ctx["warmup"].get("completed_epoch")==10,f"fold {fold} warmup epoch mismatch"
    assert ctx["warmup"]["rng_sha"]==pre["provenance"]["native_rng"],f"fold {fold} RNG mismatch"
    assert verify_warmup_file(fold)
    members=ctx["members"]
    expected=[[783,783,88,0],[792,784,87,8],[789,784,87,5],[788,784,87,4],[784,784,87,0],[784,784,87,0],[784,784,87,0],[786,784,87,2],[784,784,87,0],[784,784,87,0]][fold-1]
    actual=[len(members[k]) for k in ("train","legal","test")]+[len(set(members["train"])&set(members["test"]))]
    assert actual==expected,(fold,actual,expected)
    assert pre["status"]=="PASS"
    return {"fold":fold,"status":"PASS","warmup_sha256":verify_warmup_file(fold),"rng_sha":ctx["warmup"]["rng_sha"],"provenance_sha":engine.h.sha(ctx["provenance"]),"membership_sha":engine.h.sha(members),"members":actual,"optimizer_initial":{name:len(opt.state) for name,opt in [("base",ctx["state"]["opt_base"]),("cam",ctx["state"]["opt_cam"]),("added",ctx["state"]["opt_added"]),("DPSL",ctx["optimizer"])]},"preflight_record_status":pre["status"],"dropout_call_count":len(pre["dropout"])}
checks=[check_initial(f) for f in range(1,11)]
# One isolated actual epoch-11 step, save/reload/evaluate verification on Fold 1.
tmp=OUT/"preflight"/"fold01_step11"
engine,files,result=run_fold(tmp,1,stop_at=11)
saved=engine.h.load(files["latest"])
ctx=engine.setup("G2",1)
engine.restore(ctx,saved)
eval2,_=engine.evaluate(ctx)
validate_evaluation(eval2,len(ctx["members"]["test"]),ctx["state"]["labels_np"][ctx["members"]["test"]])
line=(files["journal"].read_bytes().splitlines())[0]
source_line=(SRC/"runtime/history/fold01.journal.jsonl").read_bytes().splitlines()[0]
assert line==source_line,"fold 1 step 11 differs from historical witness"
assert eval2["logits"].shape==saved["evaluation"]["logits"].shape
assert float(abs(eval2["logits"]-saved["evaluation"]["logits"]).max())<=2e-6,"fold 1 disk reload logits exceed numeric bound"
assert (eval2["prediction"]==saved["evaluation"]["prediction"]).all(),"fold 1 disk reload hard predictions differ"
assert saved["epoch"]==11 and saved["best_sha"]==sha256_file(files["best"])
summary={"status":"PASS","warmup_initialization":checks,"isolated_fold1_epoch11":{"status":"PASS","epoch":11,"evaluation_reload_within_2e-6":True,"hard_predictions_equal":True,"historical_epoch11_journal_exact":True,"correct":eval2["correct"],"state_sha":saved["state_sha"],"rng_sha":saved["rng_sha"],"optimizer_after_reload":{name:len(opt.state) for name,opt in [("base",ctx["state"]["opt_base"]),("cam",ctx["state"]["opt_cam"]),("added",ctx["state"]["opt_added"]),("DPSL",ctx["optimizer"])]}},"external_project_reads_denied_by_guard":True,"formal_training_rng_consumed":False}
(OUT/"preflight"/"preflight.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False))
