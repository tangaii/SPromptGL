#!/usr/bin/env python3
"""Standalone DPSL E82.4.6.10-G1 experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / name) for name in ("code", "runtime/python", "code/history_base")]

from dependency_guard import DependencyGuard

FOLDS = tuple(range(1, 11))


def main() -> int:
    parser = argparse.ArgumentParser(prog="dpsl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("environment", "inspect", "evaluate", "replay-archive", "resume-check", "audit", "train", "train-resume"):
        sub = commands.add_parser(name)
        if name not in {"audit", "environment"}:
            folds = sub.add_mutually_exclusive_group()
            folds.add_argument("--fold", type=int, choices=FOLDS)
            folds.add_argument("--all", action="store_true")
        if name in {"train", "train-resume", "evaluate"}:
            sub.add_argument("--run", required=name != "evaluate", metavar="NAME")
        if name == "train":
            sub.add_argument("--from-warmup", required=True, action="store_true")
        if name == "train-resume":
            sub.add_argument("--from-formal-latest", action="store_true")
        if name == "audit":
            sub.add_argument("--record", action="store_true", help="Write an audit receipt under results/recheck.")
    args = parser.parse_args()
    folds = (args.fold,) if getattr(args, "fold", None) else FOLDS
    threads = json.loads((ROOT / "code/history_base/config.json").read_text())["gpu"]["threads"]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    guard = DependencyGuard()
    sys.addaudithook(guard.hook)
    try:
        from dpsl_runtime import inspect, replay_archive

        if args.command == "environment":
            import numpy as np
            import scipy
            import sklearn
            import torch
            available = torch.cuda.is_available() and torch.cuda.device_count() == 1
            result = {"status": "PASS" if available else "FAIL", "python": sys.version.split()[0],
                      "torch": torch.__version__, "cuda_api": torch.version.cuda, "numpy": np.__version__,
                      "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                      "device": torch.cuda.get_device_name(0) if available else None,
                      "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
        elif args.command == "inspect":
            result = inspect(folds)
        elif args.command == "replay-archive":
            result = replay_archive(write=folds == FOLDS, folds=folds)
        elif args.command in {"evaluate", "resume-check"}:
            from real_model_runtime import evaluate, recovery

            output = ROOT / "results" / "recheck" if folds == FOLDS else None
            if args.command == "evaluate":
                run = None
                if args.run:
                    from training_runtime import named_run
                    run = named_run(args.run)
                    output = run if folds == FOLDS else None
                result = evaluate(folds, out=output, run_dir=run)
            else:
                result = recovery(folds, out=output)
            dependency = guard.report()
            result["dependency_closure"] = dependency
            if dependency["status"] != "PASS":
                result["status"] = "FAIL"
        elif args.command == "audit":
            from standalone_audit import audit
            result = audit(guard, record=args.record)
        else:
            from training_runtime import train
            result = train(args.run, folds, resume=args.command == "train-resume",
                           from_formal_latest=getattr(args, "from_formal_latest", False))
    except FileNotFoundError as error:
        result = {"status": "FAIL", "classification": "BLOCKED_MISSING_SOURCE", "error": str(error)}
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        result = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] in {"PASS", "TRAINED", "COMPLETE_AUDITED_STANDALONE"} or (
        args.command == "inspect" and result["status"] != "FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
