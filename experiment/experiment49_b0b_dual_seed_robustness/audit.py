from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'; CKPTS=RT/'checkpoints'; TASKS=RT/'tasks'; RESULTS=EXP/'results'
import integration as I

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    c=I.get_runtime(); torch=c['torch']; x,y,_=I.load_data(); out=[]
    groups=[g for g in I.CFG['groups'] if g['gpu']==args.gpu]
    for g in groups:
        folds=I.make_folds(x,y,g['cv_seed']); hp=I.make_hp()
        previous_terminal=None
        for f,(tr,te) in enumerate(folds,1):
            tp=TASKS/f'{g["id"]}_f{f:02d}.json'; bp=CKPTS/f'{g["id"]}_f{f:02d}.best.pt'; r=I.read_json(tp); assert r['status']=='COMPLETE' and bp.exists()
            art=I.load_pt(bp); I.restore_rng(art['entry_rng']); a=I.make_agent(x,y,tr,te,hp); I.load_model(a,art['model']); a.MessagePassing0.eval();a.ModalFusion.eval()
            with torch.no_grad(): pred=I.forward(a)[0][a.tst_idx].argmax(1).cpu().numpy().astype(np.int64)
            met=I.metric_row(I.forward(a)[0],y,te); pred2=met.pop('prediction'); assert np.array_equal(pred,pred2)
            for k in ['acc','hard_auc','spe','sen','asd_tp','asd_fn','td_fp','td_tn']: assert abs(float(r[k])-float(met[k]))<1e-12
            assert I.digest_obj(art['model'])==r['model_sha']; assert I.digest_obj(pred)==r['prediction_sha']
            assert I.rng_sha(art['entry_rng'])==r['entry_rng_sha']; assert I.rng_sha(art['terminal_rng'])==r['terminal_rng_sha']
            if previous_terminal is not None: assert I.rng_sha(art['entry_rng'])==previous_terminal
            previous_terminal=I.rng_sha(art['terminal_rng'])
            old=a.targ.clone(); a.targ[a.tst_idx]=1-a.targ[a.tst_idx]
            with torch.no_grad(): out2=I.forward(a)[0]
            a.targ=old; with_label=I.forward(a)[0]
            assert torch.equal(out2,with_label)
            out.append(dict(group=g['id'],fold=f,status='PASS',acc=met['acc'],hard_auc=met['hard_auc'],spe=met['spe'],sen=met['sen'],prediction_sha=r['prediction_sha'],best_sha=r['best_sha']))
            del a; torch.cuda.empty_cache()
    path=RT/f'audit_gpu{args.gpu}.json'; I.write_json(path,{'status':'PASS','gpu':args.gpu,'rows':out,'count':len(out),'generated_unix':time.time()}); print('AUDIT_PASS',args.gpu,len(out))

if __name__=='__main__': main()
