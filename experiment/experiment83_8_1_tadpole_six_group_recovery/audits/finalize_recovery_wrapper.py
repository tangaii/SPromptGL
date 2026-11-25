"""Run the sealed result aggregator with its report-scope variable supplied."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/finalize_e8381.py"
sys.path.insert(0, str(ROOT / "code"))
SPEC = importlib.util.spec_from_file_location("finalize_e8381", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load sealed aggregator at {SCRIPT}")
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)
ORIGINAL_LOAD_FOLDS = AGGREGATOR.load_replayed_fold_metrics


def load_replayed_fold_metrics_with_report_scope():
    folds = ORIGINAL_LOAD_FOLDS()
    AGGREGATOR.folds = folds
    return folds


AGGREGATOR.load_replayed_fold_metrics = load_replayed_fold_metrics_with_report_scope
AGGREGATOR.main()
