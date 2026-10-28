#!/usr/bin/env python3
"""Descriptive site/diagnosis edge statistics for E2, E5 and E6."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'
sys.path.insert(0,str(HERE)); import run

def main():
    c=run.context(); x,y,folds=run.data(); mp=np.load(run.CFG['data']['mapping'],allow_pickle=True).item()
    names=mp['PHENO']; site_cols=[i for i,n in enumerate(names) if 'SITE_ID_' in str(n)]
    sites=x[:,site_cols].argmax(1)
    rows=[]
    for gid in (2,5,6):
      for fold,(tr,te) in enumerate(folds,1):
        an=run.load(RT/'anchors'/f'fold{fold:02d}.pt'); run.restore(an['entry_rng'])
        a=c['method'].make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,480000+fold);run.restore(an['rng'])
        props=None
        if gid==6: props,_,_=run.permutation(x,y,tr,te)
        b=run.load(RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt');c['method'].load_bundle(a,b['model'])
        with c['torch'].no_grad(): o=c['method'].evaluate(a,gid,c['utils'],[props]*4 if props is not None else None)
        for ri,adj in enumerate(o['adjs']):
          aa=adj.detach().cpu().numpy(); same_site=same_label=cross_same=edges=cross=0
          for i in range(len(aa)):
            z=aa[i].copy();z[i]=-np.inf;k=min(16,len(aa)-1);ids=np.argpartition(z,-k)[-k:]
            for j in ids:
              if not np.isfinite(z[j]): continue
              edges+=1; ss=sites[i]==sites[j]; yy=y[i]==y[j]
              same_site+=int(ss);same_label+=int(yy)
              if not ss: cross+=1;cross_same+=int(yy)
          rows.append({'group_id':gid,'group':run.NAMES[gid],'fold':fold,'relation_index':ri,'topk':16,'edges':edges,'same_site_rate_all':same_site/max(edges,1),'same_label_rate_all':same_label/max(edges,1),'cross_site_edges':cross,'cross_site_same_label_rate':cross_same/max(cross,1),'propagation_node_permuted':gid==6})
        print('SITE',gid,fold,flush=True)
    fields=list(rows[0]);
    with (RES/'site_diagnostics.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    h=hashlib.sha256((RES/'site_diagnostics.csv').read_bytes()).hexdigest(); print(json.dumps({'status':'SITE_DIAGNOSTICS_PASS','rows':len(rows),'sha256':h}))
if __name__=='__main__': main()
