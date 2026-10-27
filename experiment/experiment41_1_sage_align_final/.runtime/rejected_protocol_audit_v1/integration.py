"""Runtime attachment for the SAGE-Align branch.

The source copy is private to this experiment.  This module keeps the original
PLN layers and OutputLayer intact and only supplies a controlled CAM residual.
"""
from __future__ import annotations

import types

import torch


def _compute_cam(module, x):
    bs = x.size(0)
    attn_map = []
    with torch.no_grad():
        x, attn = module.InputLayer(x)
        attn = attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for i in range(module.n_layer):
            x, layer_attn = module.Encoder[i](q=x, k=x, v=x,
                                              modal_num=module.modal_num)
            attn = layer_attn.mean(dim=1)
            x = module.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
    return x.detach(), attn.detach(), attn_map


def _forward_sage(module, x):
    cam, attention, attn_map = _compute_cam(module, x)
    branch = getattr(module, "sage_branch", None)
    evidence = getattr(module, "sage_evidence", None)
    if branch is None:
        augmented = cam
        residual = torch.zeros_like(cam.reshape(cam.shape[0], -1))
    else:
        augmented, residual = branch(evidence, cam)
    output, hidden = module.Outputlayer(
        augmented.reshape(x.shape[0], -1), attention.reshape(x.shape[0], -1))
    module.sage_last_residual = residual.detach()
    module.sage_last_cam = cam.detach()
    return output, hidden, attn_map


def attach(module, branch, evidence):
    """Attach one fold's evidence tensor and branch to the original PLN."""
    module.sage_branch = branch
    module.sage_evidence = evidence
    module.sage_last_residual = None
    module.sage_last_cam = None
    module.forward = types.MethodType(_forward_sage, module)
    return module


def detach(module):
    for key in ("sage_branch", "sage_evidence", "sage_last_residual", "sage_last_cam"):
        if hasattr(module, key):
            delattr(module, key)
