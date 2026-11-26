#!/usr/bin/env python3
"""Independent NumPy/sklearn audit of captured Warmup and DARG logits."""
from pathlib import Path
import json, hashlib, re
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, f1_score
ROOT=Path('/model/spromptgl/experiment/experiment_abide_authentic_logs_v2')

def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def metrics(logits, labels):
 labels=np.asarray(labels,dtype=np.int64); logits=np.asarray(logits)
 pred=np.argmax(logits,axis=1).astype(np.int64); prob=np.exp(logits)
 return {'acc':float(np.mean(pred==labels)), 'hard_auc':float(balanced_accuracy_score(labels,pred)),
  'prob_auc':float(roc_auc_score(labels,prob[:,1])), 'sen':float(recall_score(labels,pred,pos_label=0,zero_division=0)),
  'spe':float(recall_score(labels,pred,pos_label=1,zero_division=0)), 'asd_f1':float(f1_score(labels,pred,pos_label=0,zero_division=0))}
def close(a,b): return abs(float(a)-float(b)) <= 2e-12
def fmt(row):
 c=float(row['trn_cls']); return f"{int(row['epoch']):3d}/450 trn-loss-MF: {c:.4f}  trn-loss-MF: {c:.4f} trn-loss-MF: {float(row['gfr_alignment']):.4f} trn-auc: {float(row['trn_auc']):.4f}  acc: {float(row['trn_acc']):.4f} val-auc: {float(row['val_auc']):.4f}  acc: {float(row['val_acc']):.4f}"
def check_stage(stage, replay, original_journal):
 total=0; passed=0; failures=[]; samples=[]
 for fold in range(1,11):
  z=np.load(ROOT/'results'/replay/'logits'/f'fold{fold:02d}.npz',allow_pickle=False)
  journal_path=ROOT/original_journal/f'fold{fold:02d}.jsonl'
  if stage=='DARG': journal_path=ROOT/'results/metric_recalculation/darg_replay/runtime/logs'/f'E74.4-G2-Positive-DCR_f{fold:02d}.full.jsonl'
  rows=[json.loads(x) for x in journal_path.read_text().splitlines() if x]
  if stage=='Warmup': targets=[int(x) for x in range(1,11)]
  else: targets=[int(x) for x in z['internal_epoch']]
  idx={int(x):i for i,x in enumerate(z['internal_epoch'])} if stage!='Warmup' else {i+1:i for i in range(len(z['train_logits']))}
  log_lines=(ROOT/stage/'result_ABIDE.txt').read_text().splitlines()
  starts=[n for n,x in enumerate(log_lines) if x.startswith('Counter(')]
  section=log_lines[starts[fold-1]:(starts[fold] if fold<10 else len(log_lines))]
  for internal in targets:
   i=idx[internal]; jr=rows[internal-1] if stage=='Warmup' else next(x for x in rows if int(x['internal_epoch'])==internal)
   tr=metrics(z['train_logits'][i],z['train_labels'][i]); va=metrics(z['validation_logits'][i],z['validation_labels'][i])
   expected={'trn_auc':tr['hard_auc'],'trn_acc':tr['acc'],'val_auc':va['hard_auc'],'val_acc':va['acc']}
   if stage=='Warmup':
    actual_metrics=(jr['training_metrics'],jr['validation_metrics'])
    actual={'trn_auc':actual_metrics[0]['hard_auc'],'trn_acc':actual_metrics[0]['acc'],'val_auc':actual_metrics[1]['hard_auc'],'val_acc':actual_metrics[1]['acc']}
   else:
    actual_metrics=(jr['train_metrics'],jr['validation_metrics'])
    actual={'trn_auc':actual_metrics[0]['hard_auc'],'trn_acc':actual_metrics[0]['acc'],'val_auc':actual_metrics[1]['hard_auc'],'val_acc':actual_metrics[1]['acc']}
   ok=all((abs(tr[k]-actual_metrics[0][k]) <= (1e-5 if k=='prob_auc' else 2e-12)) and
          (abs(va[k]-actual_metrics[1][k]) <= (1e-5 if k=='prob_auc' else 2e-12))
          for k in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))
   # locate author epoch by display epoch and fold sections
   display=internal-1
   occurrences=[n for n,x in enumerate(section) if re.match(r'^\s*'+str(display)+r'/450 trn-loss-MF:',x)]
   text_ok=True
   if not occurrences: text_ok=False
   else: text_ok=all(f"trn-auc: {actual['trn_auc']:.4f}" in section[occurrences[0]] and f"val-auc: {actual['val_auc']:.4f}" in section[occurrences[0]] and f"acc: {actual['trn_acc']:.4f}" in section[occurrences[0]] for _ in [0])
   total+=1
   if ok and text_ok: passed+=1
   else: failures.append({'fold':fold,'internal_epoch':internal,'metric_match':ok,'text_match':text_ok,'expected':expected,'actual':actual})
   samples.append({'fold':fold,'internal_epoch':internal,'display_epoch':display,'metrics_recomputed':ok,'log_text_match':text_ok,'logits_sha256':hashlib.sha256(np.ascontiguousarray(z['train_logits'][i]).tobytes()).hexdigest(),'labels_sha256':hashlib.sha256(np.ascontiguousarray(z['train_labels'][i]).tobytes()).hexdigest(),'rng_before':str(z['rng_before'][i]),'rng_after_train':str(z['rng_after_train'][i]),'rng_after_eval':str(z['rng_after_eval'][i])})
 return {'status':'PASS' if passed==total else 'FAIL','total':total,'passed':passed,'failures':failures,'samples':samples}
def main():
 out=ROOT/'results/metric_recalculation'
 w=check_stage('Warmup','metric_recalculation/warmup_replay','Warmup/journal')
 d=check_stage('DARG','metric_recalculation/darg_replay','DARG/journal')
 old=json.loads((ROOT/'results/audit_status.json').read_text())['dpsl_random_epoch_metric_recalculation']
 result={'status':'PASS' if w['status']==d['status']=='PASS' and old['status']=='PASS' else 'INCOMPLETE','warmup':w,'darg':d,'dpsl_existing':old,'independent_metric_implementation':'numpy argmax + sklearn balanced_accuracy_score/roc_auc_score/recall_score/f1_score','source_logs_unchanged':{s:sha(ROOT/s/'result_ABIDE.txt') for s in ('Warmup','DARG','DPSL')}}
 (out/'independent_metric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'warmup':(w['passed'],w['total']),'darg':(d['passed'],d['total']),'dpsl_existing':old['status']},ensure_ascii=False))
if __name__=='__main__': main()
