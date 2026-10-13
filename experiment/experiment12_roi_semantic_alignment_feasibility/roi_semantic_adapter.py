"""Shared, edge-local AGESA; preserves all 256 feature positions."""
import torch
from torch import nn
from torch.nn import functional as F


class AGESA(nn.Module):
    def __init__(self, text_embeddings, valid_indices):
        super().__init__()
        assert text_embeddings.shape == (252, 1024)
        assert valid_indices.shape == (252,) and valid_indices.unique().numel() == 252
        self.register_buffer('text_embeddings',text_embeddings.detach().clone())
        self.register_buffer('valid_indices',valid_indices.detach().clone())
        self.project_text = nn.Linear(1024, 32, bias=False)
        self.project_value = nn.Linear(3, 32, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def residual(self, z, text_embeddings=None, off=False):
        assert z.ndim == 2 and z.shape[1] == 256
        if off:
            return torch.zeros_like(z)
        e = self.text_embeddings if text_embeddings is None else text_embeddings
        assert e.shape == (252, 1024)
        raw = z.index_select(1,self.valid_indices)
        phi = torch.stack([raw,raw.abs(),raw.sign()],dim=-1)
        t = F.normalize(self.project_text(e),dim=-1)
        v = F.normalize(self.project_value(phi),dim=-1)
        score = (v*t.unsqueeze(0)).sum(dim=-1)
        response = raw*torch.tanh(score)
        delta_valid = .1*torch.tanh(self.alpha)*response
        return torch.zeros_like(z).index_copy(1,self.valid_indices,delta_valid)

    def forward(self, z, h, text_embeddings=None, off=False):
        assert h.ndim == 2 and h.shape == (len(z),320)
        delta = self.residual(z,text_embeddings,off)
        return torch.cat([h[:,:64],h[:,64:]+delta],dim=1),delta
