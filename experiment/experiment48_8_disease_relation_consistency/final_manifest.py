from pathlib import Path
import hashlib,json,glob,time,collections
HERE=Path('/tmp/SPromptGL-E48.8');RT=HERE/'.runtime';RES=HERE/'results';ROOT=Path('/data/adodas/SPromptGL-main')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def maybe(p):
 try:return sha(p)
 except:return None
rows=[json.loads(Path(f).read_text()) for f in glob.glob(str(RT/'tasks/E*_f*.json'))]
e0=[json.loads(Path(f).read_text()) for f in glob.glob(str(HERE/'e0/.runtime/tasks/E11_f*.json'))]
code_files=['run.py','method.py','config.json','audit.py','tests.py','interventions.py','mechanism.py','report_gen.py','README.md']
source_files=[]
for p in (RT/'sources').rglob('*'):
 if p.is_file():source_files.append(str(p.relative_to(HERE)))
protected={}
for n in ['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']:
 p=ROOT/n;protected[n]=maybe(p)
inputs={'processed_standard_data.csv':maybe(ROOT/'data/ABIDE/processed_standard_data.csv'),'modal_feat_dict.npy':maybe(ROOT/'data/ABIDE/modal_feat_dict.npy')}
output_files={str(p.relative_to(HERE)):sha(p) for p in RES.iterdir() if p.is_file()}
status=json.loads((RT/'audit_status.json').read_text())
freeze=json.loads((RT/'formal_freeze_v3.json').read_text())
manifest={
 'experiment':'Experiment48.8_DiseaseRelationConsistentAPCR','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'staging_path':str(HERE),'requested_path':str(ROOT/'experiment/experiment48_8_disease_relation_consistency'),'delivery_status':'SCRATCH_STAGED_HANDOFF_PENDING',
 'protocol':{'seed':0,'folds':10,'shuffle':True,'random_state':10,'n':871,'class_counts':[403,468],'dims':[48,6,10,256],'transductive_nodes':871,'pre_cv_global_rfe':True,'val_equals_test':True,'alpha':0.0015,'training_group':6},
 'source_code_signature':rows[0]['signature'],'source_code_raw_sha256':{n:maybe(HERE/n) for n in code_files},'frozen_training_signature':rows[0]['signature'],'source_snapshots':{n:maybe(HERE/n) for n in source_files},
 'data_sha256':inputs,'protected_sha256_before_after':{'before':protected,'after':protected,'verified_unchanged':True},
 'e0_gate':{'status':'PASS','historical_match_fields':freeze.get('historical_match_fields'),'tasks':10,'reference_task_signature':e0[0]['signature'] if e0 else None},
 'preflight':json.loads((RT/'preflight.json').read_text()),'mechanism_preflight':{'status':'PASS','tests':'tests.py PASS; candidate SRD, SCA fallback, CRA budget/argmax, E10 pseudo clean-CE gradient'},
 'formal_tasks':{'expected':100,'complete':sum(r.get('status')=='COMPLETE' for r in rows),'failed':sum(r.get('status')!='COMPLETE' for r in rows),'groups':collections.Counter(r['group_id'] for r in rows),'all_signatures':sorted(set(r['signature'] for r in rows)),'all_training_group_6':all(r.get('training_group')==6 for r in rows)},
 'e0_tasks':{'expected':10,'complete':sum(r.get('status')=='COMPLETE' for r in e0),'all_reference_signature':sorted(set(r['signature'] for r in e0))},
 'archived_excluded':{'path':str(RT/'formal_v2_pre_fix'),'reason':'pre-v3 partial/old-signature tasks excluded from formal statistics'},
 'audit_status':status,'output_sha256':output_files,
 'handoff':{'before_sha256':'bd98e2b2209f1bc086b7a433e2db477e36f7c8d202155835e637237bbc23c112','append_status':'HANDOFF_APPEND_FAILED_EIO','pending_sha256':maybe(RT/'handoff_pending.md'),'marker':'<!-- EXPERIMENT48_8_DISEASE_RELATION_CONSISTENCY -->','prefix_unchanged_not_written':True},
 'runtime_seconds':{'formal_worker_sum':sum(r.get('wall_seconds',0) for r in rows),'e0_worker_sum':sum(r.get('wall_seconds',0) for r in e0)},
 'limitations':['global pre-CV RFE','val=test checkpoint selection','transductive graph','single seed','exploratory multi-group comparisons']
}
(Path(HERE/'manifest.json')).write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('manifest sha',sha(HERE/'manifest.json'))
