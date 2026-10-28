"""Explicit, non-automatic stage dispatcher for Experiment 47.3."""
from __future__ import annotations
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['repair','controls','parity','train','finalize'])
    p.add_argument('--group',choices=['D4','D5']);p.add_argument('--fold',type=int);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.stage=='repair':
        from repair_audit import run_scale,unified_loss_audit;run_scale();unified_loss_audit()
    elif a.stage=='controls':
        from audit import run_effectiveness,gates;_,m=run_effectiveness();gates(m)
    elif a.stage=='parity':
        from parity_check import main as f;f()
    elif a.stage=='train':
        if a.group is None or a.fold is None: p.error('train requires --group and --fold')
        from train import run_task;run_task(a.group,a.fold,a.device)
    else:
        from finalize import main as f;f()

if __name__=='__main__':main()
