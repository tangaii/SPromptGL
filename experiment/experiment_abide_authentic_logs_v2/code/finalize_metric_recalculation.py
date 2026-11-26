from pathlib import Path
import json,hashlib,subprocess,datetime
ROOT=Path('/model/spromptgl/experiment/experiment_abide_authentic_logs_v2')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
# independent evidence
metric=json.loads((ROOT/'results/metric_recalculation/independent_metric_audit.json').read_text())
assert metric['status']=='PASS' and metric['warmup']['passed']==100 and metric['darg']['passed']==30 and metric['dpsl_existing']['status']=='PASS'
# worktree is deliberately reported from this audit, without changing it
status=subprocess.run(['git','-C','/data/adodas/SPromptGL-main','status','--porcelain=v1','--untracked-files=all'],capture_output=True,text=True,check=True).stdout
old=json.loads((ROOT/'results/audit_status.json').read_text())
old['dpsl_random_epoch_metric_recalculation_verified_again']=True
old['independent_metric_recalculation']=metric
old['metric_recalculation_gaps']=[]
old['metric_recalculation_status']='METRIC_RECALCULATION_COMPLETE_AUDITED'
old['overall_status']='INCOMPLETE_SOURCE_PROVENANCE'
old['source_provenance_investigation']={
 'status':'SOURCE_PROVENANCE_DISCREPANCY_UNRESOLVED',
 'historical_checkpoint_run_sha':'77b0ba096a46df53f69c6b570f1dbbd6f6907f7ada3b2a67b4092f3e0750412b',
 'recovered_run_sha':'f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334',
 'model_e61_sha':'28266568bcebcfdbab961dbedb7257c15ff4e862cd02ae744feacad54b7bc81b',
 'config_sha':'7a1a2e7d37918baf615c704a47d09e3a3eb8a9de5bfc752d852b15a8ada6569b',
 'source_sha_map_function':'run.py source_sha_map() hashes run.py, model_e61.py, config.json as a map. SOURCE_SHA_OVERRIDE remained None.',
 'github_commit':'31ced6bdd279654154b90dc2ece056d8e435df2b',
 'github_commit_record':'Add Experiment 61 CADR propagation comparison',
 'github_run_sha':'f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334',
 'finding':'Recovered run.py is byte-identical to the GitHub E61 commit. No local or Git object with the historical 77b0... run.py hash was found. The historical metadata difference cannot be attributed to serialization because source_sha_map is a file-byte hash map.'
}
old['project_worktree_current']={'status':'DIRTY_WITH_LOCAL_CHANGES','porcelain_line_count':len(status.splitlines()),'porcelain_sha256':hashlib.sha256(status.encode()).hexdigest(),'sample':status.splitlines()[:12],'protected_history_sha_inventory':'PASS'}
(ROOT/'results/audit_status.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
# update manifest without copying/rewriting formal logs
manifest=json.loads((ROOT/'manifest.json').read_text())
manifest['status']='INCOMPLETE_SOURCE_PROVENANCE'
manifest['metric_recalculation_status']='METRIC_RECALCULATION_COMPLETE_AUDITED'
manifest['independent_metric_recalculation']={'path':str(ROOT/'results/metric_recalculation/independent_metric_audit.json'),'status':'PASS','warmup_samples':100,'darg_samples':30,'dpsl_existing_samples':10,'sha256':sha(ROOT/'results/metric_recalculation/independent_metric_audit.json')}
manifest['source_provenance_investigation']=old['source_provenance_investigation']
manifest['project_worktree_current']=old['project_worktree_current']
manifest['author_logs']={k:{**v,'sha256':sha(v['path'])} for k,v in manifest['author_logs'].items()}
# refresh file inventory, excluding manifest itself and pycache
files=[]
for p in sorted(ROOT.rglob('*')):
 if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts:
  files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest['files']=files
manifest['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# append detailed report section only once
report=ROOT/'results/report.md'; text=report.read_text()
header='## Independent metric recalculation closure'
if header not in text:
 text += f'''\n{header}\n\n- Status: `METRIC_RECALCULATION_COMPLETE_AUDITED`. Warmup was independently recomputed from 100 captured train/evaluation logits and labels across 10 folds. DARG was deterministically replayed from E61 epoch 10 for 30 frozen audit points, with internal epochs 11, one precommitted intermediate epoch, and historical Best per fold. All 130 points matched independently calculated ACC and Hard-AUC/BA and the four-decimal author-log fields.\n- DPSL's existing 10/10 raw-logit evidence was read back and remains `PASS`. No DPSL training or formal model output was changed.\n- The independent calculator uses NumPy argmax and scikit-learn `balanced_accuracy_score`, `roc_auc_score`, `recall_score`, and `f1_score`. It does not call the training `classification_metrics()` implementation.\n- Warmup and DARG capture archives are under `results/metric_recalculation/`. The DARG sampling plan SHA-256 is `cdc90180708714150da85fce0c94e760478ec5f51b76a820eb5f57eccfb3598b`.\n- E61 source provenance remains `SOURCE_PROVENANCE_DISCREPANCY_UNRESOLVED`. `run.py` source SHA recorded in historical checkpoints is `77b0ba096a46df53f69c6b570f1dbbd6f6907f7ada3b2a67b4092f3e0750412b`. The recovered `run.py` is `f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334` and is byte-identical to GitHub commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. No source override or checkpoint modification was used.\n- The project worktree contains pre-existing local changes. This run did not clean or overwrite them. Current `git status --porcelain=v1 --untracked-files=all` has {len(status.splitlines())} lines with SHA-256 `{hashlib.sha256(status.encode()).hexdigest()}`. The protected historical SHA inventory remains PASS for 1423 files.\n- Overall status: `INCOMPLETE_SOURCE_PROVENANCE`. The metric-recalculation evidence is complete, but the requested full three-stage status remains withheld because historical E61 `run.py` byte provenance is unresolved.\n'''
 report.write_text(text)
print(json.dumps({'metric_status':manifest['metric_recalculation_status'],'overall':manifest['status'],'files':len(files),'logs':manifest['author_logs']},ensure_ascii=False))
