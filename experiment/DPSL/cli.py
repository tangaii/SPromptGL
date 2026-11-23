#!/usr/bin/env python3
"""Minimal standalone DPSL command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from dpsl_runtime import audit, inspect, replay_best_evaluation, resume_check, train_resume


def main() -> int:
    parser = argparse.ArgumentParser(prog="dpsl", description="E82.4.6.10-G1 standalone DPSL base")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="DPSL directory. Defaults to the directory containing this file.")
    parser.add_argument("command", choices=("inspect", "evaluate", "audit", "resume-check", "train-resume"))
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise SystemExit("The checked-in CLI is rooted by __file__. Run the copy inside a DPSL directory.")
    if args.command == "inspect":
        result = inspect()
    elif args.command == "evaluate":
        result = replay_best_evaluation(write=True)
    elif args.command == "audit":
        result = audit()
    elif args.command == "resume-check":
        result = resume_check()
    else:
        result = train_resume()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
