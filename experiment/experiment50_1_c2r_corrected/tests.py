"""Semantic test entry point for Experiment 50.1.

The implementation is kept in the frozen runner so that tests exercise the
same data loading, B0-B restoration and relation path used for training.
"""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([
        sys.executable, "-B", str(root / "run.py"), "preflight",
        "--gpu", str(args.gpu)]))
