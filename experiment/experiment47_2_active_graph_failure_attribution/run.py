"""Reproducible stage dispatcher for Experiment 47.2.

The completed run is intentionally resumable: graph controls and Stage A are
idempotent, and each D0--D3 fold writes a separate checkpoint/CSV fragment.
"""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def call(args):
    return subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['graphs','a','gates','interventions'],required=True);p.add_argument('--folds',default='1-10');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='graphs': call(['graph_controls.py'])
    elif a.stage=='a': call(['scale_audit.py'])
    elif a.stage=='gates': call(['audit.py'])
    else: call(['audit.py','--intervene','--folds',a.folds,'--device',a.device])
