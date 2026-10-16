"""Experiment 21 reference: bounded patient-FC adaptation of frozen concept priors.

The exact baseline integration is specified in experiment21_codex_prompt.md.
This module is self-contained and has no dependency on historical experiment packages.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CONDITIONS = {'NUMERIC', 'REAL', 'NEUTRAL', 'ISOMETRIC', 'WRONG_ANATOMY', 'ANONYMOUS'}


class BoundedConceptEvidence21(nn.Module):
    def __init__(self, *, condition: str, use_graph: bool, cache: dict,
                 raw_corr: torch.Tensor, train_idx, semantic_graph: bool = True):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f'Unknown evidence condition: {condition}')
        if raw_corr.ndim != 2 or raw_corr.shape[1] != 256:
            raise ValueError('Expected original 256-dimensional CORR features')
        self.condition = condition
        self.use_graph = bool(use_graph)
        self.semantic_graph = bool(semantic_graph)
        self.register_buffer('valid_idx', torch.as_tensor(cache['valid_idx'], dtype=torch.long))
        mask_key = 'masks_wrong' if condition == 'WRONG_ANATOMY' else 'masks'
        mask = torch.as_tensor(cache[mask_key], dtype=torch.bool)
        self.register_buffer('mask', mask)
        bank = {
            'NUMERIC': None,
            'REAL': 'prior_real',
            'NEUTRAL': 'prior_neutral',
            'ISOMETRIC': 'prior_random',
            'WRONG_ANATOMY': 'prior_real',
            'ANONYMOUS': 'prior_anonymous',
        }[condition]
        raw_prior = torch.zeros(4, 252) if bank is None else torch.as_tensor(cache[bank], dtype=torch.float32)
        self.register_buffer('prior', raw_prior)
        self.register_buffer('support', torch.as_tensor(cache['line_graph'], dtype=torch.bool))
        self.register_buffer('train_idx', torch.as_tensor(train_idx, dtype=torch.long))
        if self.mask.shape != (4, 252) or self.prior.shape != (4, 252):
            raise AssertionError('Concept prior/mask dimension mismatch')
        if not bool(self.mask.any(dim=1).all()):
            raise AssertionError('Every concept must have at least one valid edge')
        if self.support.shape != (252, 252) or not torch.equal(self.support, self.support.T):
            raise AssertionError('Invalid undirected line-graph support')
        if not bool(self.support.diagonal().all()):
            raise AssertionError('Line graph must contain self nodes')

        # Class-agnostic, OUTER-TRAIN-ONLY reference. Never interpret deviations as pathology.
        source = raw_corr.detach().index_select(1, self.valid_idx.to(raw_corr.device))
        tr = self.train_idx.to(raw_corr.device)
        tr_source = source.index_select(0, tr)
        self.register_buffer('ref_mean', tr_source.mean(0).float())
        self.register_buffer('ref_sd', tr_source.std(0, unbiased=False).clamp_min(1e-3).float())
        # Directed neighbour pairs: source message -> destination node, no redundant self-loop.
        pair_mask = self.support.clone()
        pair_mask.fill_diagonal_(False)
        dest, src = pair_mask.nonzero(as_tuple=True)
        self.register_buffer('dest', dest)
        self.register_buffer('src', src)

        self.encoder = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.LayerNorm(32))
        self.edge_id = nn.Embedding(252, 32)
        self.edge_norm = nn.LayerNorm(32)
        self.graph_proj = nn.Linear(32, 32, bias=False) if self.use_graph else None
        self.graph_norm = nn.LayerNorm(32) if self.use_graph else None
        self.readout = nn.Sequential(nn.Linear(136, 64), nn.GELU(), nn.Linear(64, 36))
        self.alpha = nn.Parameter(torch.zeros(()))
        # 0.35*sigmoid(logit) -> eta(0)=0.10, strictly bounded within (0,0.35)
        self.eta_logit = nn.Parameter(torch.tensor(math.log(0.10 / 0.25)))
        self.intervention = 'REAL'
        self.last = {}
        self._loss = None

    def _patient_data(self, corr):
        z = corr.index_select(1, self.valid_idx)
        d = ((z - self.ref_mean) / self.ref_sd).clamp(-6, 6)
        gate = torch.sigmoid(2 * (d.abs() - 1)).clamp_min(1e-5)
        return z, d, gate

    def _prior_and_mask(self):
        p, m = self.prior, self.mask
        # Note: set intervention via explicit API, never mutate permanent buffers.
        if self.intervention in {'NEUTRAL', 'ISOMETRIC', 'ANONYMOUS', 'NUMERIC'}:
            # These counterfactual banks must be provided by an external adapter,
            # which temporarily swaps p/m and restores it byte-for-byte.
            pass
        return p, m

    def forward(self, base: torch.Tensor, raw_modalities):
        if base.ndim != 3 or tuple(base.shape[1:]) != (4, 36):
            raise ValueError('Insert after PLN Encoder+FeedForward: expected [N,4,36]')
        if not isinstance(raw_modalities, (tuple, list)) or len(raw_modalities) != 4:
            raise ValueError('Expected original 4 modality tensors')
        corr = raw_modalities[3]
        if corr.shape != (base.shape[0], 256):
            raise ValueError('CORR tensor must be [N,256]')
        z, d, gate = self._patient_data(corr)
        x = torch.stack([(d/3).tanh(), (d/3).tanh().abs(), z.tanh()], dim=-1)
        h = self.edge_norm(self.encoder(x) + self.edge_id.weight[None])
        prior, mask = self._prior_and_mask()
        # Closed-form, bounded, class-agnostic patient correction.
        eta = 0.35 * self.eta_logit.sigmoid()
        if self.intervention == 'ADAPT_OFF':
            eta = eta * 0.
        patient_term = torch.tanh(d.abs() - 1.).unsqueeze(1)
        base_logits = (prior / 0.5).masked_fill(~mask, -1e4)
        logits = base_logits.unsqueeze(0) + eta * patient_term
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e4)
        att = F.softmax(logits, dim=-1)
        p0 = F.softmax(base_logits, dim=-1).detach()
        kl = (att * (att.clamp_min(1e-9).log() - p0.unsqueeze(0).clamp_min(1e-9).log())).sum(-1)
        self._loss = 0.02 * kl.index_select(0, self.train_idx).mean()

        # Sparse patient-specific edge line graph. Node = verified FC connection,
        # NOT the original 871-subject GFR graph nor a fabricated full ROI graph.
        graph_effect = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.use_graph and self.intervention != 'GRAPH_OFF':
            src, dest = self.src, self.dest
            w = (gate[:, src] * gate[:, dest]).clamp_min(1e-8)
            if self.semantic_graph and self.intervention != 'GRAPH_NUMERIC':
                # Each connection has a 4-domain normalized attention profile.
                profile = F.normalize(att.transpose(1, 2), dim=-1, eps=1e-8)
                sim = (profile[:, src] * profile[:, dest]).sum(-1)
                w = w * torch.exp(0.7 * (sim - 0.5))
            # Normalization: include unit self-weight, and aggregate neighbours.
            denom = h.new_ones((len(h), 252))
            denom = denom.index_add(1, dest, w)
            neighbours = torch.zeros_like(h).index_add(1, dest, h[:, src] * w.unsqueeze(-1))
            mixed = (h + neighbours) / denom.unsqueeze(-1)
            updated = self.graph_norm(h + F.gelu(self.graph_proj(mixed)))
            graph_effect = (updated-h).norm(dim=-1).mean()
            h = updated

        evidence = torch.einsum('nke,ned->nkd', att, h)
        signed = torch.einsum('nke,ne->nk', att, d.tanh())
        coverage = torch.einsum('nke,ne->nk', att, gate)
        fusion_input = torch.cat([evidence.flatten(1), signed, coverage], dim=-1)
        raw_delta = self.readout(fusion_input)
        direction = F.normalize(raw_delta, dim=-1, eps=1e-8)
        delta_corr = 0.10 * self.alpha.tanh() * base[:, 3].norm(dim=-1, keepdim=True).detach() * direction
        if self.intervention == 'BRANCH_OFF':
            delta_corr = delta_corr * 0
        zero = torch.zeros_like(delta_corr)
        delta = torch.stack([zero, zero, zero, delta_corr], dim=1)
        self.last = dict(att=att.detach(), p0=p0.detach(), eta=eta.detach(),
                         patient_gate=gate.detach(), graph_effect=graph_effect.detach(),
                         prior=prior.detach(), delta=delta.detach(), d=d.detach(),
                         kl=kl.detach())
        return base + delta

    def auxiliary_loss(self):
        if self._loss is None:
            raise RuntimeError('Call forward before auxiliary_loss')
        return self._loss
