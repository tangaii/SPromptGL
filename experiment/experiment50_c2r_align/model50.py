"""C²R-Align module and PEN-free CAM wrapper for Experiment 50.

The original GFR implementation is imported from the immutable local source snapshot
by run.py.  This file contains only the new relation module and the physical PEN-free
CAM implementation used by the B0-B anchor.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PLNNoPEN(nn.Module):
    """Exact B0-B CAM path copied from the audited Experiment43.1 implementation."""
    def __init__(self, dims, hp, layers_module):
        super().__init__()
        self.hyperpm = hp
        self.input_data_dims = dims
        self.d_q = self.d_k = self.d_v = hp.n_hidden
        self.n_head = hp.n_head
        self.dropout = hp.dropout
        self.n_layer = hp.nlayer
        self.modal_num = hp.nmodal
        self.n_class = hp.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
        self.InputLayer = layers_module.VariLengthInputLayer(
            dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        for index in range(self.n_layer):
            encoder = layers_module.EncodeLayer(
                self.d_k * self.n_head, self.d_k, self.d_v,
                self.n_head, self.dropout)
            self.add_module("encode_%d" % index, encoder)
            self.Encoder.append(encoder)
            feedforward = layers_module.FeedForwardLayer(
                self.d_v * self.n_head, self.d_v * self.n_head,
                dropout=self.dropout)
            self.add_module("feed_%d" % index, feedforward)
            self.FeedForward.append(feedforward)
        self.Outputlayer = layers_module.OutputLayer(
            self.d_v * self.n_head * self.modal_num,
            self.d_v * self.n_head, self.n_class,
            self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, input_attention = self.InputLayer(x)
        attn = input_attention.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for index in range(self.n_layer):
            x, layer_attention = self.Encoder[index](
                q=x, k=x, v=x, modal_num=self.modal_num)
            attn = layer_attention.mean(dim=1)
            x = self.FeedForward[index](x)
            attn_map.append(attn.detach().cpu().numpy())
        output, hidden = self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))
        return output, hidden, attn_map


class _Encoder(nn.Module):
    def __init__(self, dim, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out), nn.LayerNorm(out))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1, eps=1e-6)


class _Value(nn.Module):
    def __init__(self, dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return self.net(x)


class C2RAlign(nn.Module):
    """Confounder-controlled, disease-supervised relational alignment.

    The module consumes original four-modal features and a frozen-in-call GFR output.
    Candidate senders are supplied by the fold builder and are always outer-train
    nodes.  All label-dependent pair supervision is supplied through pair_i/pair_j,
    whose indices are asserted to be training nodes by the runner.
    """
    MODES = {"E1_SELF", "E2_FC", "E3_UNCONDITIONAL", "E4_NO_ALIGN",
             "E5_FULL", "E6_MISALIGNED_AUX", "E7_REWIRED"}

    def __init__(self, dims=(48, 6, 10, 256), embed=32, hidden=64,
                 eta=0.25, rho=0.05, temperature=0.25):
        super().__init__()
        self.dims = tuple(dims)
        self.eta = float(eta)
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.encoders = nn.ModuleList([
            _Encoder(d, hidden, embed) for d in self.dims
        ])
        # The gate sees four raw relation scores, three anchor differences, and
        # four verified confounder/quality descriptors.
        self.gate = nn.Sequential(
            nn.Linear(11, 32), nn.GELU(), nn.Linear(32, 4))
        self.value = _Value(320)
        self.self_value = _Value(320)

    @staticmethod
    def _split(x, dims):
        out, p = [], 0
        for d in dims:
            out.append(x[:, p:p+d])
            p += d
        return out

    @staticmethod
    def _confounds(x, ri, sj):
        ph = x[:, :48]
        age = ph[:, :26]
        site = ph[:, 26:46]
        sex = ph[:, 46:48]
        quality = F.normalize(x[:, 48:64], p=2, dim=-1, eps=1e-6)
        same_site = (site[ri] * site[sj]).sum(-1)
        age_sim = (age[ri] * age[sj]).sum(-1)
        sex_same = (sex[ri] * sex[sj]).sum(-1)
        quality_sim = (quality[ri] * quality[sj]).sum(-1)
        return torch.stack([same_site, age_sim, sex_same, quality_sim], dim=-1)

    def _embeddings(self, x):
        return self.encoders and [enc(part) for enc, part in
                                  zip(self.encoders, self._split(x, self.dims))]

    def _scores(self, z, ri, sj, aux_perm=None):
        scores = []
        for m, zm in enumerate(z):
            if aux_perm is not None and m < 3:
                ii, jj = aux_perm[ri], aux_perm[sj]
            else:
                ii, jj = ri, sj
            scores.append((zm[ii] * zm[jj]).sum(-1))
        return torch.stack(scores, dim=-1)

    def _gate_weights(self, scores, conf, mode):
        if mode == "E3_UNCONDITIONAL":
            g = scores.new_zeros((*scores.shape[:-1], 4))
            g[..., :3] = 1.0 / 3.0
            return g
        if mode == "E2_FC":
            g = scores.new_zeros((*scores.shape[:-1], 4))
            g[..., 3] = 1.0
            return g
        feat = torch.cat([scores, (scores[..., 1:] - scores[..., :1]).abs(), conf], dim=-1)
        return torch.softmax(self.gate(feat), dim=-1)

    def _relation(self, scores, gate, mode):
        if mode == "E2_FC":
            return scores[..., 0]
        if mode == "E3_UNCONDITIONAL":
            return scores[..., 0] + self.eta * scores[..., 1:].mean(-1)
        return scores[..., 0] + self.eta * (gate[..., :3] * scores[..., 1:]).sum(-1)

    def _edge_bundle(self, x, z, ri, sj, mode, aux_perm=None):
        scores = self._scores(z, ri, sj, aux_perm=aux_perm)
        conf = self._confounds(x, ri, sj)
        gate = self._gate_weights(scores, conf, mode)
        relation = self._relation(scores, gate, mode)
        if mode in ("E2_FC", "E3_UNCONDITIONAL"):
            accept = relation.new_ones(relation.shape)
        else:
            accept = gate[..., :3].sum(-1)
        return scores, gate, relation, accept, conf

    def _pair_losses(self, x, z, pair_i, pair_j, pair_y, mode, aux_perm):
        if mode == "E1_SELF":
            zero = x.new_zeros(())
            return zero, zero, {"rel_loss": 0.0, "align_loss": 0.0}
        scores, gate, relation, _, _ = self._edge_bundle(
            x, z, pair_i, pair_j, mode,
            aux_perm if mode == "E6_MISALIGNED_AUX" else None)
        rel_loss = F.binary_cross_entropy_with_logits(
            relation / self.temperature, pair_y.float())
        if mode in ("E2_FC", "E4_NO_ALIGN"):
            align_loss = x.new_zeros(())
        else:
            weights = gate[..., :3]
            align_raw = F.smooth_l1_loss(
                scores[..., 1:], scores[..., :1].detach().expand_as(scores[..., 1:]),
                reduction="none")
            align_loss = (weights * align_raw).sum() / (weights.sum() + 1e-6)
        return rel_loss, align_loss, {
            "rel_loss": float(rel_loss.detach()), "align_loss": float(align_loss.detach())}

    def forward(self, x, h_gfr, candidates, pair_i, pair_j, pair_y,
                mode, aux_perm=None, rewire=None):
        assert mode in self.MODES
        z = self._embeddings(x)
        n, k = candidates.shape
        ri = torch.arange(n, device=x.device).unsqueeze(1).expand(n, k).reshape(-1)
        sj = candidates.reshape(-1)
        aux_for_score = aux_perm if mode == "E6_MISALIGNED_AUX" else None
        scores, gate, relation, accept, conf = self._edge_bundle(
            x, z, ri, sj, mode, aux_perm=aux_for_score)
        scores = scores.view(n, k, 4)
        gate = gate.view(n, k, 4)
        relation = relation.view(n, k)
        accept = accept.view(n, k)
        if mode == "E1_SELF":
            delta = self.self_value(x)
            gate_mean = x.new_tensor([0., 0., 0., 1.])
            msg = self.self_value(x)
        else:
            weights = torch.softmax(relation / self.temperature, dim=1) * accept
            senders = candidates if rewire is None else rewire
            vals = self.value(x[senders.reshape(-1)]).view(n, k, -1)
            msg = (weights.unsqueeze(-1) * vals).sum(1)
            delta = msg - self.value(x)
            gate_mean = gate.mean(dim=(0, 1))
        h_rms = h_gfr.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
        bounded = self.rho * h_rms * torch.tanh(F.layer_norm(delta, (delta.shape[-1],)))
        h50 = h_gfr + bounded
        rel_loss, align_loss, loss_stats = self._pair_losses(
            x, z, pair_i, pair_j, pair_y, mode, aux_perm)
        with torch.no_grad():
            diag = {
                **loss_stats,
                "gate_pheno": float(gate_mean[0]),
                "gate_anat": float(gate_mean[1]),
                "gate_func": float(gate_mean[2]),
                "gate_null": float(gate_mean[3]),
                "candidate_k": int(k),
                "message_norm": float(msg.detach().norm()),
                "delta_norm": float(bounded.detach().norm()),
                "gfr_norm": float(h_gfr.detach().norm()),
                "residual_ratio": float(bounded.detach().norm() / (h_gfr.detach().norm() + 1e-8)),
                "pair_count": int(pair_i.numel()),
                "pair_positive_fraction": float(pair_y.float().mean()),
            }
        return h50, rel_loss, align_loss, diag
