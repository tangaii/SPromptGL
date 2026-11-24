from pathlib import Path
import sys, os, json, time, traceback
SRC=Path('/model/spromptgl/experiment/DPSL')
OUT=Path('/model/spromptgl/experiment/DPSL_full_retrain_from_warmup')
sys.dont_write_bytecode=True
sys.path[:0]=[str(SRC/n) for n in ('code','runtime/python','code/history_base')]
import builtins
_ALLOWED_READ=SRC.resolve(); _ALLOWED_OUT=OUT.resolve(); _PROTECTED=(Path('/model/spromptgl').resolve(),Path('/data/adodas/SPromptGL-main').resolve())
def _audit(event,args):
    if event=='open' and args:
        raw=args[0]
        if isinstance(raw,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(raw))
            try:a=p.resolve()
            except OSError:return
            if any(a.is_relative_to(root) for root in _PROTECTED):
                flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
                mode=args[1] if len(args)>1 and isinstance(args[1],str) else 'r'
                writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or any(c in mode for c in 'wax+')
                allowed=(a.is_relative_to(_ALLOWED_READ) and not writing) or a.is_relative_to(_ALLOWED_OUT)
                if not allowed: raise PermissionError('project path access outside DPSL read-only base and this run output')
sys.addaudithook(_audit)
from training_runtime import run_fold
from dpsl_runtime import FOLDS
log=OUT/'logs'/'full_training.log'; log.parent.mkdir(parents=True,exist_ok=True)
def emit(x):
    line=json.dumps(x,ensure_ascii=False)
    print(line,flush=True)
    with log.open('a') as f:f.write(line+'\n')
start=time.time(); emit({'event':'training_start','start_epoch_completed':10,'first_epoch':11,'max_epochs':450,'early_stop_wait_gt':200,'workers':1,'device':'cuda:0','start_time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
for fold in FOLDS:
    t=time.time(); emit({'event':'fold_start','fold':fold,'elapsed_sec':time.time()-start})
    try:
        engine,files,result=run_fold(OUT,fold)
        emit({'event':'fold_complete','fold':fold,'seconds':time.time()-t,'result':result,'best_sha256':__import__('hashlib').sha256(files['best'].read_bytes()).hexdigest(),'latest_sha256':__import__('hashlib').sha256(files['latest'].read_bytes()).hexdigest()})
    except Exception as e:
        emit({'event':'fold_error','fold':fold,'seconds':time.time()-t,'error':repr(e),'traceback':traceback.format_exc()})
        raise
emit({'event':'training_complete','seconds':time.time()-start,'end_time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
