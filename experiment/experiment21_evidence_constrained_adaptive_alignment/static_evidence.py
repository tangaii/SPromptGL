"""Experiment 20: fixed-text/verified-FC concept prior + optional patient graph.
Model adapter only. Prepare concept priors and exact baseline integration separately.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConceptEvidence20(nn.Module):
    """Transductive inputs; class-agnostic reference statistics use train indices only."""
    def __init__(self, *, condition, use_graph, cache, raw_corr, train_idx):
        super().__init__()
        conditions = {'NUMERIC', 'REAL', 'NEUTRAL', 'WRONG_ANATOMY', 'ISOMETRIC', 'ANONYMOUS'}
        if condition not in conditions:
            raise ValueError(condition)
        self.condition = condition
        self.use_graph = bool(use_graph)
        self.register_buffer('valid_idx', torch.as_tensor(cache['valid_idx'], dtype=torch.long))
        self.register_buffer('masks', torch.as_tensor(
            cache['masks_wrong' if condition == 'WRONG_ANATOMY' else 'masks'], dtype=torch.bool))
        source = {'NUMERIC': None, 'REAL': 'prior_real', 'NEUTRAL': 'prior_neutral',
                  'WRONG_ANATOMY': 'prior_real', 'ISOMETRIC': 'prior_random', 'ANONYMOUS': 'prior_anonymous'}[condition]
        prior = torch.zeros(4, 252) if source is None else torch.as_tensor(cache[source], dtype=torch.float32)
        self.register_buffer('prior', prior)
        self.register_buffer('line_graph', torch.as_tensor(cache['line_graph'], dtype=torch.bool))
        tr = torch.as_tensor(train_idx, dtype=torch.long, device=raw_corr.device)
        ztr = raw_corr.index_select(1, self.valid_idx.to(raw_corr.device)).index_select(0, tr)
        mu = ztr.mean(0)
        sd = ztr.std(0, unbiased=False).clamp_min(1e-3)
        self.register_buffer('reference_mean', mu.float())
        self.register_buffer('reference_std', sd.float())
        assert self.masks.shape == self.prior.shape == (4, 252)
        assert self.masks.any(dim=1).all()
        assert self.line_graph.shape == (252, 252)
        self.encoder = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.LayerNorm(32))
        self.edge_id = nn.Embedding(252, 32)
        self.edge_norm = nn.LayerNorm(32)
        self.graph_proj = nn.Linear(32, 32, bias=False) if self.use_graph else None
        self.graph_norm = nn.LayerNorm(32) if self.use_graph else None
        self.readout = nn.Sequential(nn.Linear(136, 64), nn.GELU(), nn.Linear(64, 36))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer('graph_adj', torch.empty(0), persistent=False)
        self.last = {}

    @torch.no_grad()
    def prepare_graph(self, raw_corr):
        if not self.use_graph:
            return
        _, d, g = self.measure(raw_corr)
        connected = self.line_graph.float()
        if self.condition == 'NUMERIC':
            semantics = torch.ones_like(connected)
        else:
            profile = F.normalize((self.prior * self.masks).T, dim=-1, eps=1e-6)
            semantics = (1.0 + profile @ profile.T).clamp(0, 2) * 0.5
        base = connected * (0.5 + 0.5 * semantics)
        a = base.unsqueeze(0) * g[:, :, None] * g[:, None, :]
        diagonal = torch.arange(252, device=a.device)
        a[:, diagonal, diagonal] += 1.0
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.graph_adj = a.detach()
        assert torch.isfinite(self.graph_adj).all()
        assert torch.allclose(self.graph_adj.sum(-1), torch.ones_like(self.graph_adj.sum(-1)), atol=1e-5)

    def measure(self, raw_corr):
        assert raw_corr.ndim == 2 and raw_corr.size(1) == 256
        z = raw_corr.index_select(1, self.valid_idx)
        d = ((z - self.reference_mean) / self.reference_std).clamp(-6.0, 6.0)
        gate = torch.sigmoid(2.0 * (d.abs() - 1.0)).clamp_min(1e-5)
        return z, d, gate

    def forward(self, base, raw_modalities):
        assert base.ndim == 3 and base.shape[1:] == (4, 36)
        assert raw_modalities is not None and len(raw_modalities) == 4
        raw = raw_modalities[3]
        z, d, gate = self.measure(raw)
        q = (d / 3.0).tanh()
        phi = torch.stack((q, q.abs(), z.tanh()), dim=-1)
        h = self.edge_norm(self.encoder(phi) + self.edge_id.weight.unsqueeze(0))
        if self.use_graph:
            assert self.graph_adj.shape == (len(base), 252, 252)
            h = self.graph_norm(h + F.gelu(self.graph_proj(torch.bmm(self.graph_adj, h))))
        logits = self.prior.unsqueeze(0) + 0.5 * gate.log().unsqueeze(1)
        logits = logits.masked_fill(~self.masks.unsqueeze(0), -1e4)
        att = F.softmax(logits, dim=-1)
        evidence = torch.einsum('nke,ned->nkd', att, h)
        signed = torch.einsum('nke,ne->nk', att, d.tanh())
        coverage = torch.einsum('nke,ne->nk', att, gate)
        concat = torch.cat((evidence.reshape(len(base), 128), signed, coverage), dim=-1)
        raw_delta = self.readout(concat)
        direction = F.normalize(raw_delta, dim=-1, eps=1e-8)
        scale = base[:, 3, :].norm(dim=-1, keepdim=True).detach()
        delta_corr = 0.1 * self.alpha.tanh() * scale * direction
        zero = torch.zeros_like(delta_corr)
        delta = torch.stack((zero, zero, zero, delta_corr), dim=1)
        self.last = {'attention': att.detach(), 'delta': delta.detach(),
                     'gate': gate.detach(), 'signed': signed.detach(),
                     'coverage': coverage.detach(), 'raw_prior': self.prior.detach()}
        return base + delta
