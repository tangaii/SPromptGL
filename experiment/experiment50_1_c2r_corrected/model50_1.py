"""Corrected C²R-Align for Experiment 50.1.

The B0-B GFR and CAM are loaded from an immutable Experiment43.1 source
snapshot by ``run.py``.  This module contains only the new relational branch.
The semantic convention is explicit throughout: PHENO=0, ANAT=1, FUNC=2 and
CORR/FC=3.  No positional ``0`` is used as a hidden FC anchor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLNNoPEN(nn.Module):
    """Exact PEN-free CAM path used by the audited B0-B anchors."""

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


class C2RAlignCorrected(nn.Module):
    """FC-anchored, selective cross-modal patient relation alignment.

    Candidate senders are supplied by the outer-fold builder and are always
    outer-train patients.  The relation branch is fully differentiable except
    for the deliberately detached training-fold site-bias statistic and the
    FC stop-gradient target in the alignment loss.
    """

    PHENO, ANAT, FUNC, FC = 0, 1, 2, 3
    AUX = (PHENO, ANAT, FUNC)
    MODAL_NAMES = ("PHENO", "ANAT", "FUNC", "CORR_FC")
    MODES = {
        "E1_SELF", "E2_FC", "E3_UNCONDITIONAL", "E4_NO_ALIGN", "E5_FULL",
        "E6_MISALIGNED_AUX", "E7_REWIRED", "E8_NO_SITE_CORRECTION"
    }

    def __init__(self, dims=(48, 6, 10, 256), embed=32, hidden=64,
                 eta=0.25, rho=0.05, temperature=0.25,
                 site_lambda=1.0):
        super().__init__()
        self.dims = tuple(dims)
        self.eta = float(eta)
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.site_lambda = float(site_lambda)
        self.encoders = nn.ModuleList([
            _Encoder(d, hidden, embed) for d in self.dims
        ])
        # Four scores + three auxiliary-vs-FC absolute differences + four
        # validated confounder descriptors = 11 inputs.
        self.gate = nn.Sequential(
            nn.Linear(11, 32), nn.GELU(), nn.Linear(32, 4))
        self.value = _Value(320)
        self.self_value = _Value(320)

    @staticmethod
    def split_modalities(x, dims):
        parts, start = [], 0
        for dim in dims:
            parts.append(x[:, start:start + dim])
            start += dim
        return parts

    @staticmethod
    def site_onehot(x):
        return x[:, 26:46]

    @classmethod
    def confounds(cls, x, receiver, sender):
        pheno = x[:, :48]
        age = pheno[:, :26]
        site = cls.site_onehot(x)
        sex = pheno[:, 46:48]
        quality = F.normalize(x[:, 48:64], p=2, dim=-1, eps=1e-6)
        same_site = (site[receiver] * site[sender]).sum(-1)
        age_sim = (age[receiver] * age[sender]).sum(-1)
        sex_same = (sex[receiver] * sex[sender]).sum(-1)
        quality_sim = (quality[receiver] * quality[sender]).sum(-1)
        return torch.stack([same_site, age_sim, sex_same, quality_sim], dim=-1)

    def embeddings(self, x):
        return [enc(part) for enc, part in zip(
            self.encoders, self.split_modalities(x, self.dims))]

    @classmethod
    def same_site(cls, x, receiver, sender):
        site = cls.site_onehot(x)
        return ((site[receiver] * site[sender]).sum(-1) > 0.5)

    def raw_scores(self, z, receiver, sender, aux_perm=None):
        """Return [PHENO, ANAT, FUNC, CORR/FC] scores in that order."""
        scores = []
        for modal, zm in enumerate(z):
            if aux_perm is not None and modal in self.AUX:
                ri, sj = aux_perm[receiver], aux_perm[sender]
            else:
                ri, sj = receiver, sender
            scores.append((zm[ri] * zm[sj]).sum(-1))
        return torch.stack(scores, dim=-1)

    @staticmethod
    def _safe_mean(values, mask, fallback):
        if bool(mask.any()):
            return values[mask].mean()
        return fallback

    def site_bias(self, pair_scores, pair_same_site, enabled=True):
        """Estimate train-fold relation bias; returned tensors are detached."""
        if not enabled:
            return pair_scores.new_zeros((4, 2))
        all_mean = pair_scores.mean(0)
        same_mask = pair_same_site.bool()
        diff_mask = ~same_mask
        rows = []
        for modal in range(4):
            same = self._safe_mean(pair_scores[:, modal], same_mask, all_mean[modal])
            diff = self._safe_mean(pair_scores[:, modal], diff_mask, all_mean[modal])
            rows.append(torch.stack([same - all_mean[modal], diff - all_mean[modal]]))
        return torch.stack(rows).detach()

    def corrected_scores(self, raw_scores, x, receiver, sender, bias, enabled=True):
        if not enabled:
            return raw_scores
        group = self.same_site(x, receiver, sender).long()
        # bias[modal, 0] is same-site minus all; index 1 is cross-site minus all.
        offset = torch.stack([bias[m, group] for m in range(4)], dim=-1)
        return raw_scores - self.site_lambda * offset

    def gate_weights(self, scores, confounds, mode, force_null=False):
        if force_null or mode == "E2_FC":
            gate = scores.new_zeros((*scores.shape[:-1], 4))
            gate[..., 3] = 1.0
            return gate
        if mode == "E3_UNCONDITIONAL":
            gate = scores.new_zeros((*scores.shape[:-1], 4))
            gate[..., :3] = 1.0 / 3.0
            return gate
        # Correct auxiliary-vs-FC anchor differences; FC is index 3.
        diff = (scores[..., :3] - scores[..., 3:4]).abs()
        features = torch.cat([scores, diff, confounds], dim=-1)
        return torch.softmax(self.gate(features), dim=-1)

    def relation(self, scores, gate, mode):
        # FC is always present. NULL removes only the auxiliary contribution.
        if mode == "E2_FC":
            return scores[..., self.FC]
        return scores[..., self.FC] + self.eta * (
            gate[..., :3] * scores[..., :3]).sum(-1)

    def edge_bundle(self, x, z, receiver, sender, mode, pair_scores,
                    pair_same_site, aux_perm=None, force_null=False,
                    force_no_site=False):
        raw = self.raw_scores(z, receiver, sender, aux_perm=aux_perm)
        bias = self.site_bias(pair_scores, pair_same_site, enabled=not force_no_site and mode != "E8_NO_SITE_CORRECTION")
        corrected = self.corrected_scores(raw, x, receiver, sender, bias,
                                          enabled=not force_no_site and mode != "E8_NO_SITE_CORRECTION")
        conf = self.confounds(x, receiver, sender)
        gate = self.gate_weights(corrected, conf, mode, force_null=force_null)
        rel = self.relation(corrected, gate, mode)
        return raw, corrected, gate, rel, bias

    def pair_losses(self, x, z, pair_i, pair_j, pair_y, mode,
                    pair_same_site, pair_scores, aux_perm=None,
                    force_no_site=False):
        if mode == "E1_SELF":
            zero = x.new_zeros(())
            return zero, zero, {"rel_loss": 0.0, "align_loss": 0.0}
        _, corrected, gate, rel, _ = self.edge_bundle(
            x, z, pair_i, pair_j, mode, pair_scores, pair_same_site,
            aux_perm=aux_perm if mode == "E6_MISALIGNED_AUX" else None,
            force_no_site=force_no_site)
        rel_loss = F.binary_cross_entropy_with_logits(
            rel / self.temperature, pair_y.float())
        if mode in ("E2_FC", "E4_NO_ALIGN"):
            align_loss = x.new_zeros(())
        else:
            aux_weights = gate[..., :3]
            # FC index 3 is the stop-gradient target.  All three auxiliary
            # scores remain on the differentiable side of SmoothL1.
            target = corrected[..., 3:4].detach().expand_as(corrected[..., :3])
            raw = F.smooth_l1_loss(corrected[..., :3], target, reduction="none")
            align_loss = (aux_weights * raw).sum() / (aux_weights.sum() + 1e-6)
        return rel_loss, align_loss, {
            "rel_loss": float(rel_loss.detach()),
            "align_loss": float(align_loss.detach())}

    def forward(self, x, h_gfr, candidates, pair_i, pair_j, pair_y,
                pair_same_site=None, pair_scores=None, mode="E5_FULL", aux_perm=None,
                rewire=None, force_null=False, force_no_site=False,
                force_aux_perm=False):
        assert mode in self.MODES
        z = self.embeddings(x)
        pair_aux = aux_perm if (mode == "E6_MISALIGNED_AUX" or force_aux_perm) else None
        if pair_scores is None:
            pair_scores = self.raw_scores(z, pair_i, pair_j, aux_perm=pair_aux)
        if pair_same_site is None:
            pair_same_site = self.same_site(x, pair_i, pair_j)
        n, k = candidates.shape
        neighbors = rewire if mode == "E7_REWIRED" and rewire is not None else candidates
        receiver = torch.arange(n, device=x.device).unsqueeze(1).expand(n, k).reshape(-1)
        sender = neighbors.reshape(-1)
        aux_for_score = aux_perm if (mode == "E6_MISALIGNED_AUX" or force_aux_perm) else None
        raw, corrected, gate, rel, bias = self.edge_bundle(
            x, z, receiver, sender, mode, pair_scores, pair_same_site,
            aux_perm=aux_for_score, force_null=force_null,
            force_no_site=force_no_site)
        corrected = corrected.view(n, k, 4)
        gate = gate.view(n, k, 4)
        rel = rel.view(n, k)
        weights = torch.softmax(rel / self.temperature, dim=1)
        vals = self.value(x[sender]).view(n, k, -1)
        msg = (weights.unsqueeze(-1) * vals).sum(1)
        if mode == "E1_SELF":
            msg = self.self_value(x)
            # Capacity control: the self-only branch has a real, bounded
            # patient-specific residual but no sender from another patient.
            delta = msg
        else:
            delta = msg - self.value(x)
        h_rms = h_gfr.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
        normalized_delta = F.layer_norm(delta, (delta.shape[-1],))
        bounded = self.rho * h_rms * torch.tanh(normalized_delta)
        h50 = h_gfr + bounded
        rel_loss, align_loss, loss_stats = self.pair_losses(
            x, z, pair_i, pair_j, pair_y, mode, pair_same_site, pair_scores,
            aux_perm=aux_perm, force_no_site=force_no_site)
        with torch.no_grad():
            flat_gate = gate.reshape(-1, 4)
            diag = {
                **loss_stats,
                "gate_pheno": float(flat_gate[:, 0].mean()),
                "gate_anat": float(flat_gate[:, 1].mean()),
                "gate_func": float(flat_gate[:, 2].mean()),
                "gate_null": float(flat_gate[:, 3].mean()),
                "gate_pheno_sd": float(flat_gate[:, 0].std(unbiased=False)),
                "gate_anat_sd": float(flat_gate[:, 1].std(unbiased=False)),
                "gate_func_sd": float(flat_gate[:, 2].std(unbiased=False)),
                "gate_null_sd": float(flat_gate[:, 3].std(unbiased=False)),
                "candidate_k": int(k),
                "message_norm": float(msg.detach().norm()),
                "delta_norm": float(bounded.detach().norm()),
                "gfr_norm": float(h_gfr.detach().norm()),
                "residual_ratio": float(bounded.detach().norm() / (h_gfr.detach().norm() + 1e-8)),
                "pair_count": int(pair_i.numel()),
                "pair_positive_fraction": float(pair_y.float().mean()),
                "site_bias_abs": float(bias.detach().abs().mean()),
                "site_corrected": float(not force_no_site and mode != "E8_NO_SITE_CORRECTION"),
                "neighbor_message_std": float(msg.detach().std(dim=0).mean()),
                "null_rejection_fraction": float((flat_gate[:, 3] > 0.5).float().mean()),
                "aux_accept_mean": float(flat_gate[:, :3].sum(-1).mean()),
            }
        return h50, rel_loss, align_loss, diag
