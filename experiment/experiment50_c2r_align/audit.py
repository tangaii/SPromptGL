"""Independent audit entry point.

The complete checkpoint readback and intervention audit lives in the single
frozen runner.  Keeping this thin wrapper avoids a second implementation that
could drift from the training path.
"""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, "-B", str(root / "run.py"), "audit"]))
