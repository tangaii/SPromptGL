from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score

CLASS_NAMES=('NC','MCI','AD')

def calculate_metrics(y_true, y_pred, probs, nclass=3):
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int); probs=np.asarray(probs,dtype=float)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(nclass)))
    one_true=np.eye(nclass,dtype=float)[y_true]; one_pred=np.eye(nclass,dtype=float)[y_pred]
    try: auc_original=float(roc_auc_score(one_true,one_pred,average='macro'))
    except ValueError: auc_original=float('nan')
    try: auc_probability=float(roc_auc_score(y_true,probs,multi_class='ovr',average='macro',labels=list(range(nclass))))
    except ValueError: auc_probability=float('nan')
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=list(range(nclass)),zero_division=0)
    sens=[]; spes=[]
    total=float(cm.sum())
    for c in range(nclass):
        tp=float(cm[c,c]); fn=float(cm[c,:].sum()-tp); fp=float(cm[:,c].sum()-tp); tn=total-tp-fn-fp
        sens.append(tp/(tp+fn) if tp+fn else 0.0); spes.append(tn/(tn+fp) if tn+fp else 0.0)
    return {
      'ACC':float(np.mean(y_true==y_pred)), 'AUC_original':auc_original, 'AUC_probability':auc_probability,
      'SPE_macro':float(np.mean(spes)), 'SEN_macro':float(np.mean(sens)), 'F1_macro':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
      'F1_weighted':float(f1_score(y_true,y_pred,average='weighted',zero_division=0)), 'Balanced_Accuracy':float(balanced_accuracy_score(y_true,y_pred)),
      'NC_precision':float(precision[0]),'NC_recall':float(recall[0]),'NC_f1':float(f1[0]),
      'MCI_precision':float(precision[1]),'MCI_recall':float(recall[1]),'MCI_f1':float(f1[1]),
      'AD_precision':float(precision[2]),'AD_recall':float(recall[2]),'AD_f1':float(f1[2]),
      'confusion_matrix':cm.tolist(), 'SEN_by_class':sens, 'SPE_by_class':spes,
    }

def summarize(rows, metrics):
    out={'n_folds':len(rows),'ddof':0,'metrics':{}}
    for metric in metrics:
        vals=np.asarray([float(r[metric]) for r in rows],dtype=float)
        out['metrics'][metric]={'mean':float(np.nanmean(vals)),'std':float(np.nanstd(vals,ddof=0)),'values':vals.tolist()}
    return out
