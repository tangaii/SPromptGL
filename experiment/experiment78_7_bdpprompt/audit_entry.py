#!/usr/bin/env python3
"""Stable entry point for the frozen E78.7 audit source.

The audit was frozen with the training source map; this wrapper supplies the
same unit-normalization helper used by the diagnostic code without changing
that source identity.
"""
import torch.nn.functional as F
import run
run.unit = lambda x: F.normalize(x, dim=-1, eps=1e-8)
import audit
if __name__ == '__main__':
    print(audit.audit_all())
