"""CM-PSRR relation module.

The relation representation is the audited E81.7.2-G1 node-conditioned
TP-PSRR path. Classification-margin alignment is implemented in the runner
so the auxiliary gradient can be scoped to this module explicitly.
"""
from __future__ import annotations

from tp_psrr import TPPSRR


class CMPSRR(TPPSRR):
    """Node-level Prompt sparse relational reasoning for CM-PSRR."""

    def __init__(self, semantic_basis, rho=0.1, relation_scale=1.0,
                 temperature=1.0, eps=1e-12):
        super().__init__(
            semantic_basis,
            rho=rho,
            relation_scale=relation_scale,
            temperature=temperature,
            selection_mode="node",
            eps=eps,
        )
