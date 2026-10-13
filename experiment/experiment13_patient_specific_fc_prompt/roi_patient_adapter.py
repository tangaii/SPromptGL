"""Experiment 12 AGESA, with projected-bank indexing for patient edge states."""
import torch
from torch import nn
from torch.nn import functional as F


class AGESA(nn.Module):
    def __init__(self, text_embeddings, valid_indices, status_index=None):
        super().__init__()
        assert tuple(text_embeddings.shape) in ((252, 1024), (252, 3, 1024))
        assert valid_indices.shape == (252,) and valid_indices.unique().numel() == 252
        if text_embeddings.ndim == 3:
            assert status_index.ndim == 2 and status_index.shape[1] == 252
        self.register_buffer('text_embeddings', text_embeddings.detach().clone())
        self.register_buffer('valid_indices', valid_indices.detach().clone())
        self.register_buffer('status_index', None if status_index is None else status_index.detach().clone())
        self.project_text = nn.Linear(1024, 32, bias=False)
        self.project_value = nn.Linear(3, 32, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def projected_text(self, text_embeddings=None, status_index=None):
        e = self.text_embeddings if text_embeddings is None else text_embeddings
        t = F.normalize(self.project_text(e), dim=-1)
        if e.ndim == 3:
            state = self.status_index if status_index is None else status_index
            assert state.ndim == 2 and state.shape[1] == 252
            # Project only the 756 bank entries; never expand [871,252,1024].
            t = t[torch.arange(252, device=t.device)[None, :], state]
        else:
            assert e.shape == (252, 1024)
            t = t.unsqueeze(0)
        return t

    def residual(self, z, text_embeddings=None, status_index=None, off=False):
        assert z.ndim == 2 and z.shape[1] == 256
        if off:
            return torch.zeros_like(z)
        raw = z.index_select(1, self.valid_indices)
        phi = torch.stack([raw, raw.abs(), raw.sign()], dim=-1)
        t = self.projected_text(text_embeddings, status_index)
        v = F.normalize(self.project_value(phi), dim=-1)
        score = (v * t).sum(dim=-1)
        response = raw * torch.tanh(score)
        delta_valid = .1 * torch.tanh(self.alpha) * response
        return torch.zeros_like(z).index_copy(1, self.valid_indices, delta_valid)

    def forward(self, z, h, text_embeddings=None, status_index=None, off=False):
        assert h.ndim == 2 and h.shape == (len(z), 320)
        delta = self.residual(z, text_embeddings, status_index, off)
        return torch.cat([h[:, :64], h[:, 64:] + delta], dim=1), delta
