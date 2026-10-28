"""Stage-A graph audit entry points.

The executable implementation is kept in :mod:`run` so that every audit and
its CSV rows use one feature slice, one checkpoint loader, and one patient
ordering.  This module exposes the audited helpers for independent review.
"""
from run import row_stats, graph_components

__all__ = ['row_stats', 'graph_components']
