"""Small, reproducible summaries of the Adam/gradient audit tables."""
from __future__ import annotations
import csv, statistics
from pathlib import Path
import integration as I

def read(name):
    p=I.EXP/'results'/name
    if not p.exists():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))

def summarize():
    rows=read('optimizer_diagnostics.csv');out={}
    for group in sorted(set(r['group'] for r in rows)):
        q=[r for r in rows if r['group']==group and r['parameter'].split('_')[0] in {'ANAT','FUNC','CORR'}]
        out[group]={
            'n':len(q),
            'mean_wd_fraction':statistics.mean(float(r['wd_fraction']) for r in q),
            'mean_update':statistics.mean(float(r['actual_update']) for r in q),
            'mean_grad':statistics.mean(float(r['grad_total']) for r in q),
            'gcns':sorted(set(r['parameter'] for r in q)),
        }
    return out

if __name__=='__main__':print(summarize())
