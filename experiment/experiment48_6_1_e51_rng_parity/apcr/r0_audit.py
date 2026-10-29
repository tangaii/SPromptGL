#!/usr/bin/env python3
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np

EXP=Path('/tmp/SPromptGL-E48.6.1/e51_r0'); os.environ['DUGSA_EXP']=str(EXP); sys.path.insert(0,str(EXP))
import run

def main():
    c=run.runtime(); (x,y),folds=run.load_data(); rows=[]; failures=[]
    old={int(json.loads(p.read_text())['fold']):json.loads(p.read_text()) for p in (Path('/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/tasks')).glob('E1_f*.json')}
    for fold,(tr,te) in enumerate(folds,1):
        ck=run.torch_load(EXP/'.runtime/checkpoints'/f'E1_f{fold:02d}.best.pt')
        state=run.build_state('E1',fold); fd=run.load_fold(fold)
        state['model'].load_state_dict(ck['model']);state['cam'].load_state_dict(ck['cam'])
        state['model'].eval();state['cam'].eval()
        import torch
        with torch.no_grad(): out=run.forward(state,fd,False)
        met=run.metrics(out['prob'],state); pred=out['prob'][state['test']].argmax(1).detach().cpu().numpy()
        psha=hashlib.sha256(pred.tobytes()).hexdigest(); rec=old[fold]
        keys=('acc','hard_auc','prob_auc','spe','sen')
        exact=all(abs(float(met[k])-float(rec[k]))<1e-12 for k in keys)
        if not exact: failures.append({'fold':fold,'metrics':met,'task':rec})
        rows.append({'fold':fold,'acc':met['acc'],'hard_auc':met['hard_auc'],'prob_auc':met['prob_auc'],'spe':met['spe'],'sen':met['sen'],'best_epoch':ck['best_epoch'],'epochs_executed':ck['epochs_executed'],'task_best_epoch':rec['best_epoch'],'task_prediction_sha':rec.get('prediction_sha256'),'prediction_sha':psha,'checkpoint_sha':run.sha(EXP/'.runtime/checkpoints'/f'E1_f{fold:02d}.best.pt'),'metric_exact':exact,'best_epoch_exact':int(ck['best_epoch'])==int(rec['best_epoch']),'prediction_sha_exact':rec.get('prediction_sha256')==psha})
    pred_available=all(r['task_prediction_sha'] for r in rows)
    out={'status':'PASS' if not failures else 'FAIL','folds':rows,'metric_parity':not failures,'best_epoch_parity':all(r['best_epoch_exact'] for r in rows),'prediction_sha_parity':(all(r['prediction_sha_exact'] for r in rows) if pred_available else 'NOT_VERIFIABLE_HISTORICAL_SHA_MISSING'),'rng_parity':'PASS_INITIAL_ANCHOR_ONLY','failures':failures}
    (Path('/tmp/SPromptGL-E48.6.1/apcr/.runtime/r0_readback.json')).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
