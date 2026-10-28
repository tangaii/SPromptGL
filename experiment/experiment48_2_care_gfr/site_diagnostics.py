"""Independent relation/site diagnostics for completed CARE-GFR E4/E7 read-backs."""
from pathlib import Path
import csv, hashlib, json, sys
HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; ROOT=Path('/data/adodas/SPromptGL-main')
# Reuse frozen loader and data protocol; no training is performed here.
sys.path.insert(0,str(HERE))
import run

def main():
 c=run.context(); x,y,folds=run.data(); rows=[]
 mp=c['np'].load(run.CFG['data']['mapping'],allow_pickle=True).item()
 site_cols=[i for i,n in enumerate(mp['PHENO']) if 'SITE_ID_' in str(n)]
 sites=x[:,site_cols].argmax(1)
 for gid in (4,7):
  for fold,(tr,te) in enumerate(folds,1):
   p=RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt'
   if not p.exists(): continue
   an=run.load(RT/'anchors'/f'fold{fold:02d}.pt')
   a=c['method'].make_agent(c['torch'],c['layers'],c['network'],c['utils'],c['device'],x,y,tr,te,fold,gid,an,480000+fold)
   b=run.load(p); c['method'].load_bundle(a,b['model'])
   with c['torch'].no_grad():o=c['method'].evaluate(a,gid,c['utils'],wrong_perm=run.perm_wrong(x,tr,te) if gid==7 else None)
   idx=set(map(int,tr));
   for m,sel in enumerate(o['chosen']):
    arr=sel.detach().cpu().numpy(); pairs=[]
    for i in tr:
     for j in arr[i].tolist():
      if int(j)==int(i): continue
      pairs.append((int(i),int(j)))
    if not pairs: continue
    same_site=sum(sites[i]==sites[j] for i,j in pairs)/len(pairs)
    same_y=sum(y[i]==y[j] for i,j in pairs)/len(pairs)
    cross=[(i,j) for i,j in pairs if sites[i]!=sites[j]]
    cross_same=sum(y[i]==y[j] for i,j in cross)/len(cross) if cross else float('nan')
    rows.append({'group_id':gid,'fold':fold,'relation':['PHENO','ANAT','FUNC','CORR'][m],'n_edges':len(pairs),'same_site_rate':same_site,'same_label_rate':same_y,'cross_site_edges':len(cross),'cross_site_same_label_rate':cross_same,'chosen_sha256':hashlib.sha256(arr.tobytes()).hexdigest()})
 out=HERE/'results'/'site_diagnostics.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['group_id']);w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'path':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
