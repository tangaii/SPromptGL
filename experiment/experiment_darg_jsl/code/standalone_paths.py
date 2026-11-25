"""Path registry for the self-contained DPSL source closure."""
from __future__ import annotations

from pathlib import Path


DPSL_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = DPSL_ROOT / "code" / "vendor"
VENDOR_REPO = VENDOR_ROOT / "repo"
LOCAL_RUNTIME = DPSL_ROOT / "runtime" / "python"


def repo_experiment(name: str) -> Path:
    return VENDOR_REPO / "experiment" / name


