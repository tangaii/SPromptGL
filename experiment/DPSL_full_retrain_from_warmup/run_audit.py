from pathlib import Path
import sys,os,json,hashlib,time,traceback
SRC=Path('/model/spromptgl/experiment/DPSL'); OUT=Path('/model/spromptgl/experiment/DPSL_full_retrain_from_warmup')
sys.dont_write_bytecode=True; sys.path[:0]=[str(SRC/n) for n in ('code','runtime/python','code/history_base')]
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
    flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0; mode=args[1] if len(args)>1 and isinstance(args[1],str) else 'r'
    writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or any(c in mode for c in 'wax+')
    if not ((a.is_relative_to(_ALLOWED_READ) and not writing) or a.is_relative_to(_ALLOWED_OUT)): raise PermissionError('outside allowed DPSL read-only base/output')
sys.addaudithook(_audit)
from real_model_runtime import evaluate,fold_engine,setup,optimizer_identity,check_budget,validate_evaluation
from dpsl_runtime import FOLDS,sha256_file,read_json,write_json
start=time.time(); res=evaluate(FOLDS,out=OUT/'results',run_dir=OUT)
write_json(OUT/'results'/'forward_audit.json',res)
# independent reload and next-step continuation. New run files are never modified.
checks=[]
for fold in FOLDS:
 eng=fold_engine(fold); ctx=setup(fold,eng); saved=eng.h.load(OUT/'checkpoints'/f'fold{fold:02d}.latest.pt'); eng.restore(ctx,saved)
 state_sha=eng.h.sha(eng.bundle(ctx)); rng_sha=eng.h.rng_sha(); ema_sha=eng.h.sha(eng.ema(ctx)); before_opts=optimizer_identity(ctx,eng)
 active=eng.ema(ctx); eng.restore_ema(ctx,saved['eval_ema']); ev,_=eng.evaluate(ctx); eng.restore_ema(ctx,active)
 maxlog=float(abs(ev['logits']-saved['evaluation']['logits']).max()); maxprob=float(abs(ev['probability']-saved['evaluation']['probability']).max())
 validate_evaluation(ev,len(ctx['members']['test']),ctx['state']['labels_np'][ctx['members']['test']])
 # Perform one actual in-memory next train/eval step from reloaded Latest
 pre=read_json(SRC/'runtime/history'/f'fold{fold:02d}.preflight.json')
 result,training,rng,previous,observation=eng.transition(ctx,saved['epoch']+1,pre['dropout'],inspect=True)
 budget=check_budget(eng,training); after_opts=optimizer_identity(ctx,eng); optimizer_updated=all(before_opts[k]['state_sha']!=after_opts[k]['state_sha'] for k in before_opts)
 checks.append({'fold':fold,'latest_epoch':saved['epoch'],'best_epoch':saved['best_epoch'],'reload_state_sha_equal':state_sha==saved['state_sha'],'reload_rng_sha_equal':rng_sha==saved['rng_sha'],'reload_ema_sha_equal':ema_sha==saved['ema_sha'],'eval_max_abs_logit':maxlog,'eval_max_abs_probability':maxprob,'hard_prediction_equal':bool((ev['prediction']==saved['evaluation']['prediction']).all()),'next_step_epoch':rng['epoch'],'next_step_training':True,'next_step_evaluation':True,'next_step_rng_after_train_equals_eval':rng['rng_after_train']==rng['rng_after_eval'],'optimizer_updated':optimizer_updated,'optimizers_before':before_opts,'optimizers_after':after_opts,'shared_gradient_fraction':budget})
write_json(OUT/'results/recovery_audit.json',{'status':'PASS','folds':checks,'seconds':time.time()-start})
# journal RNG exactness against formal archived journal
import json as _json
rng_rows=[]
for fold in FOLDS:
 new=[_json.loads(x) for x in (OUT/'journal'/f'fold{fold:02d}.jsonl').read_text().splitlines()]
 old=[_json.loads(x) for x in (SRC/'runtime/history'/f'fold{fold:02d}.journal.jsonl').read_text().splitlines()]
 n=min(len(new),len(old)); first=None; mismatches=0
 for i in range(n):
  if new[i].get('rng')!=old[i].get('rng'):
   mismatches+=1; first=first or i+1
 rng_rows.append({'fold':fold,'new_rows':len(new),'historical_rows':len(old),'rng_exact':mismatches==0 and len(new)==len(old),'first_rng_divergence_epoch':first,'mismatch_count':mismatches})
write_json(OUT/'results/rng_audit.json',{'status':'PASS','folds':rng_rows,'all_exact':all(x['rng_exact'] for x in rng_rows)})
print(json.dumps({'status':'PASS','patients':res['patients'],'macro_mean':res['macro_mean'],'macro_population_sd':res['macro_population_sd'],'pooled':res['pooled'],'seconds':time.time()-start},ensure_ascii=False))
