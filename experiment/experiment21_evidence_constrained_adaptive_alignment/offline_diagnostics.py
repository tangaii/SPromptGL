"""Experiment 21 first-stage audit. NO model training; read Experiment 20 saved aggregates.
Produces honest, non-additive counterfactual summary and rank-test template.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr


def summary_counterfactual(experiment20: Path) -> pd.DataFrame:
    df = pd.read_csv(experiment20/'results/counterfactual.csv')
    df = df[df['applicable'].astype(str).str.lower()=='true'].copy()
    keys = ['config','seed','fold']
    pick = df[df['config'].isin(['T11','T18'])]
    expected = {'BRANCH_OFF','GRAPH_NUMERIC','NEGATIVE_TEXT','NUMERIC_ONLY',
                'FC_BRANCH_SHUFFLED','REAL'}
    if not expected.issubset(set(pick['condition'])):
        raise RuntimeError('Missing Experiment 20 counterfactual conditions')
    pivot = pick.pivot_table(index=keys, columns='condition',
                             values='probability_abs_change', aggfunc='first')
    for c in expected:
        if c not in pivot.columns: raise RuntimeError(c)
    if pivot[['BRANCH_OFF','GRAPH_NUMERIC','NEGATIVE_TEXT']].isna().any().any():
        raise RuntimeError('Incomplete per-fold intervention')
    # NOT an additive attribution, NOT a patient-level conditional-effect estimator.
    pivot['graph_to_branch_abs_ratio'] = pivot['GRAPH_NUMERIC'] / pivot['BRANCH_OFF'].clip(lower=1e-12)
    pivot['text_to_branch_abs_ratio'] = pivot['NEGATIVE_TEXT'] / pivot['BRANCH_OFF'].clip(lower=1e-12)
    return pivot.reset_index()


def rank_audit(original_processed: Path, cache_path: Path, seed=21021) -> pd.DataFrame:
    """TRAIN-ONLY descriptive univariate rank check, not an unbiased clinical validation.

    871x321 input, last column original label one-based. No model weights needed.
    RFE was computed globally in the historical data; interpret cautiously.
    """
    import torch  # only to read numpy if needed; no network training
    original = pd.read_csv(original_processed).values
    if original.shape != (871,321): raise ValueError(original.shape)
    y=(original[:,-1].astype(int)-1)
    if set(np.unique(y))!={0,1}: raise ValueError('ASD must be 0 and TD 1')
    fc=original[:,64:320]
    with np.load(cache_path,allow_pickle=False) as archive:
        cache={k: archive[k].copy() for k in archive.files}
    valid=cache['valid_idx'];masks=cache['masks']
    assert valid.shape==(252,) and masks.shape==(4,252)
    rng=np.random.default_rng(seed)
    results=[]
    splitter=StratifiedKFold(n_splits=10,shuffle=True,random_state=10)
    for fold,(tr,te) in enumerate(splitter.split(fc,y),1):
        univariate=np.array([roc_auc_score((y[tr]==0).astype(int),fc[tr,idx]) for idx in valid])
        signal=np.abs(univariate-0.5)
        for k in range(4):
            ids=np.flatnonzero(masks[k]);ss=signal[ids]
            for name in ('prior_real','prior_neutral','prior_random','prior_anonymous'):
                prior=cache[name][k,ids]
                rho=float(spearmanr(prior,ss).statistic)
                # Null: permuting prior rank *within the same frozen domain*.
                null=np.empty(500)
                for i in range(500):null[i]=float(spearmanr(rng.permutation(prior),ss).statistic)
                results.append(dict(fold=fold,domain=k+1,prior=name,edges=len(ids),
                                    spearman=rho,perm_p=(1+int(np.sum(np.abs(null)>=abs(rho))))/501,
                                    univariate_auc_abs_mean=float(ss.mean()),
                                    univariate_top5=json.dumps(valid[ids[np.argsort(ss)[-5:][::-1]]].tolist()),
                                    prior_top5=json.dumps(valid[ids[np.argsort(prior)[-5:][::-1]]].tolist()),
                                    null_q025=float(np.quantile(null,.025)),null_q975=float(np.quantile(null,.975))))
    return pd.DataFrame(results)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--experiment20')
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    e20=Path(a.experiment20) if a.experiment20 else root/'experiment/experiment20_grounded_concept_graph_alignment'
    cf=summary_counterfactual(e20)
    cf.to_csv(out/'exp20_counterfactual_diagnostic.csv',index=False)
    ranks=rank_audit(root/'data/ABIDE/processed_standard_data.csv',
          e20/'semantic_cache.npz')
    ranks.to_csv(out/'semantic_rank_audit.csv',index=False)
    print('No training executed. Ratios are non-additive descriptive proxies; historical globally selected RFE limits interpretation.')
