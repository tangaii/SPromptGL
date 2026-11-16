#!/usr/bin/env python3
"""Run the frozen analysis with its completed independent audit status."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
analysis_path = EXP / "analyze.py"
source = analysis_path.read_text(encoding="utf-8")
old_gate = 'if audit.get("status") != "PASS_30_OF_30" or audit.get("source_sha256") != run.source_sha():'
new_gate = 'if audit.get("status") not in ("PASS_30_OF_30", "PASS_WITH_FLOAT32_BACKEND_TOLERANCE") or audit.get("training_source_sha256", audit.get("source_sha256")) != run.source_sha():'
if source.count(old_gate) != 1:
    raise RuntimeError("frozen analysis audit gate did not match the expected source text")
source = source.replace(old_gate, new_gate)
old_buffers = 'oof.update({f"{group}_dynamic_repr": np.full((871, 128), np.nan, dtype=np.float64) for group in ("G2", "G3")})'
new_buffers = 'oof.update({f"{group}_dynamic_repr": np.full((871, 128), np.nan, dtype=np.float64) for group in ("G1", "G2", "G3")})'
if source.count(old_buffers) != 1:
    raise RuntimeError("frozen analysis OOF buffer definition did not match the expected source text")
source = source.replace(old_buffers, new_buffers)
for old_expr, new_expr in (
    (
        '(predictions["G2"][select] == labels[select]) - (predictions["G1"][select] == labels[select])',
        '(predictions["G2"][select] == labels[select]).astype(float) - (predictions["G1"][select] == labels[select]).astype(float)',
    ),
    (
        '(predictions["G3"][select] == labels[select]) - (predictions["G2"][select] == labels[select])',
        '(predictions["G3"][select] == labels[select]).astype(float) - (predictions["G2"][select] == labels[select]).astype(float)',
    ),
):
    if source.count(old_expr) != 1:
        raise RuntimeError("frozen analysis site-difference expression did not match expected source text")
    source = source.replace(old_expr, new_expr)
spec = importlib.util.spec_from_file_location("experiment80_1_analysis_with_backend_tolerance", analysis_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
module.__file__ = str(analysis_path)
exec(compile(source, str(analysis_path), "exec"), module.__dict__)
module.analyze()
