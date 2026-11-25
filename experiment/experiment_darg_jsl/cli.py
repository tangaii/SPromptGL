#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "runtime/python"), str(ROOT / "code/history_base")]
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import jsl_runtime as runtime


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train = sub.add_parser("train")
    train.add_argument("--all", action="store_true", required=True)
    sub.add_parser("evaluate")
    args = parser.parse_args()
    if args.command == "preflight":
        from preflight import run
        result = run()
    elif args.command == "train":
        lock = json.loads((ROOT / "manifest.json").read_text())
        receipt = json.loads((ROOT / "runtime/preflight.json").read_text())
        if receipt.get("status") != "PASS" or receipt.get("implementation") != runtime.implementation():
            raise SystemExit("BLOCKED_PREFLIGHT")
        result = {"folds": [runtime.train_fold(fold) for fold in runtime.FOLDS]}
    else:
        result = runtime.evaluate_all()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
