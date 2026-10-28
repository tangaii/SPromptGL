"""Conditional-neighbor probe entry point.

The fixed-capacity RAW320/CAM36 residualized probes are implemented in
``run.residual_probe_space`` and are never used to update a GFR checkpoint.
"""
from run import residual_probe_space, synthetic_control

__all__ = ['residual_probe_space', 'synthetic_control']
