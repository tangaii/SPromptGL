"""Reproducibility entry points for the frozen Experiment 47.1 protocol.

Commands are intentionally explicit.  ``stage_a`` launches only the static
audit; ``pilot`` launches the six pre-registered tasks; ``expand`` launches
T1 folds 2--5 and 7--10 only after the mechanism gate has been recorded.
No command changes the root repository.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import integration as I

EXP=I.EXP;RESULTS=EXP/'results';RT=I.RT

def _cmd(args,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));return subprocess.run([sys.executable,'-u',*args],cwd=EXP,env=env,check=True)

def merge_amp():
    import amplitude_audit
    rows=[]
    for n in ['layer_amplitudes_gpu0.csv','layer_amplitudes_gpu1.csv']:
        p=RESULTS/n
        if p.exists():
            with p.open(newline='') as f:rows+=list(csv.DictReader(f))
    amplitude_audit.write_rows(rows,RESULTS/'layer_amplitudes.csv');return len(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage_a','merge_stage_a','replay','pilot','expand','audits','status']);a=p.parse_args()
    if a.command=='stage_a':
        import multiprocessing as mp
        ps=[mp.Process(target=lambda g,fs: _cmd(['amplitude_audit.py','--folds',','.join(map(str,fs)),'--gpu',str(g),'--output',f'results/layer_amplitudes_gpu{g}.csv'],g),args=(0,[1,2,3,4,5])),mp.Process(target=lambda g,fs: _cmd(['amplitude_audit.py','--folds',','.join(map(str,fs)),'--gpu',str(g),'--output',f'results/layer_amplitudes_gpu{g}.csv'],g),args=(1,[6,7,8,9,10]))]
        [q.start() for q in ps];[q.join() for q in ps];assert all(q.exitcode==0 for q in ps)
    elif a.command=='merge_stage_a':print('stage_a_rows',merge_amp())
    elif a.command=='replay':
        for f,g in [(1,0),(6,1)]:_cmd(['train.py','--group','T0','--fold',str(f),'--device','cuda:0'],g)
    elif a.command=='pilot':
        for group in ['T1','T2']:
            for f,g in [(1,0),(6,1)]:_cmd(['train.py','--group',group,'--fold',str(f),'--device','cuda:0'],g)
    elif a.command=='expand':
        for f in [2,3,4,5]:_cmd(['train.py','--group','T1','--fold',str(f),'--device','cuda:0'],0)
        for f in [7,8,9,10]:_cmd(['train.py','--group','T1','--fold',str(f),'--device','cuda:0'],1)
    elif a.command=='audits':
        _cmd(['audit.py','merge'],0);_cmd(['audit.py','summary'],0);_cmd(['audit.py','gradients'],0);_cmd(['audit.py','interventions'],0)
    else:
        print(json.dumps({'stage_a':(RESULTS/'layer_amplitudes.csv').exists(),'metrics':(RESULTS/'pilot_metrics.csv').exists(),'interventions':(RESULTS/'interventions.csv').exists()},ensure_ascii=False))

if __name__=='__main__':main()
