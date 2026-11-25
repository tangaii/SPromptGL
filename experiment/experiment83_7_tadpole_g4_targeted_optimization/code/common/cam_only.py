"""Physical TADPOLE CAM-only model with no graph or prompt modules."""
from __future__ import annotations

import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F

from fidelity_model import HistoricalCAM


class PhysicalCAMOnly(nn.Module):
    def __init__(self, dims=(145, 148, 3, 36, 24, 9)):
        super().__init__()
        self.cam = HistoricalCAM(dims, hidden=10, heads=4, nlayer=2, nclass=3,
                                 dropout=0.0, use_pen=False)
        # EncodeLayer's archived attention constructor defaults to 0.1 even
        # when the TADPOLE CAM protocol sets dropout=0. Keep the formal G5
        # path at the registered zero-dropout setting across every CAM layer.
        for module in self.cam.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.cam.apply(lambda module: nn.init.xavier_uniform_(module.weight.data)
                       if isinstance(module, nn.Linear) else None)

    def forward(self, value: Tensor):
        logp, details = self.cam(value)
        return logp, details["logits"], details


def weighted_nll(logits: Tensor, labels: Tensor, rows: Tensor) -> Tensor:
    train_labels = labels[rows].detach().cpu().numpy()
    counts = np.bincount(train_labels, minlength=3).astype("float32")
    weights = torch.tensor(len(rows) / (np.maximum(counts, 1) * 3),
                           dtype=logits.dtype, device=logits.device)
    return F.nll_loss(F.log_softmax(logits, dim=-1)[rows], labels[rows], weight=weights)
