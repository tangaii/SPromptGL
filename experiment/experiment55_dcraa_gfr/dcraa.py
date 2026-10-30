"""Disease-conditional relation alignment and bounded CORR propagation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def grad_reverse(value, scale=0.03):
    return GradientReverse.apply(value, scale)


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def _pair_features(node_features, pairs):
    i, j = pairs[:, 0].long(), pairs[:, 1].long()
    zi, zj = node_features[i], node_features[j]
    cosine = F.cosine_similarity(zi, zj, dim=-1, eps=1e-8).unsqueeze(-1)
    return torch.cat(((zi - zj).abs(), zi * zj, cosine), dim=-1)


class PairView(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.pair = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def encode_nodes(self, x):
        return self.node(x)

    def pair_from_nodes(self, node_features, pairs):
        return self.pair(_pair_features(node_features, pairs))

    def forward(self, x, pairs):
        return self.pair_from_nodes(self.encode_nodes(x), pairs)


class DCRAA(nn.Module):
    """One CORR relation encoder, conditional domain adversary and edge scorer."""
    def __init__(self, dims=(48, 6, 10, 256), hidden=32, grl_scale=0.03):
        super().__init__()
        self.corr = PairView(dims[3], hidden)
        self.relation_head = nn.Linear(hidden, 3)
        self.site_conditional = nn.ModuleList([_mlp(hidden, 16, 2) for _ in range(3)])
        self.site_unconditional = _mlp(hidden, 16, 2)
        self.auxiliary = nn.ModuleList([PairView(d, hidden) for d in dims[:3]])
        self.edge_scale = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.grl_scale = float(grl_scale)

    def pair_representation(self, features, pairs):
        return self.corr(features, pairs)

    def relation_logits(self, pair_z):
        return self.relation_head(pair_z)

    def unconditional_site_logits(self, pair_z, reverse=True):
        z = grad_reverse(pair_z, self.grl_scale) if reverse else pair_z
        return self.site_unconditional(z)

    def conditional_site_logits(self, pair_z, reverse=True):
        z = grad_reverse(pair_z, self.grl_scale) if reverse else pair_z
        return torch.stack([head(z) for head in self.site_conditional], dim=1)

    def auxiliary_pairs(self, parts, pairs):
        return [module(parts[k], pairs) for k, module in enumerate(self.auxiliary)]

    def conditional_adv_loss(self, logits, site_source, condition, confidence):
        n = len(site_source)
        labels = site_source.long().unsqueeze(1).expand(n, 3).reshape(-1)
        losses = F.cross_entropy(logits.reshape(-1, 2), labels, reduction="none").reshape(n, 3)
        per_pair = (losses * condition.detach()).sum(dim=1)
        weights = confidence.detach().clamp_min(1e-6)
        return (per_pair * weights).sum() / weights.sum()

    @staticmethod
    def unconditional_adv_loss(logits, site_source):
        return F.cross_entropy(logits, site_source.long())

    @staticmethod
    def protection_loss(logits, teacher_prob):
        return F.kl_div(F.log_softmax(logits, dim=-1), teacher_prob.detach(), reduction="batchmean")

    @staticmethod
    def auxiliary_prototype_loss(corr_z, auxiliary_z, condition, confidence):
        if not auxiliary_z:
            return corr_z.new_zeros(())
        c = condition.detach()
        conf = confidence.detach()
        losses = []
        for z_aux in auxiliary_z:
            class_losses = []
            for k in range(3):
                weight = c[:, k] * conf
                denom = weight.sum().clamp_min(1e-6)
                corr_mean = (corr_z * weight[:, None]).sum(dim=0) / denom
                aux_mean = (z_aux * weight[:, None]).sum(dim=0) / denom
                class_losses.append((F.normalize(corr_mean, dim=0) - F.normalize(aux_mean, dim=0)).pow(2).mean())
            losses.append(torch.stack(class_losses).mean())
        return torch.stack(losses).mean()

    def edge_weights(self, relation_logits, condition, base_strength, quality_similarity):
        relation_prob = F.softmax(relation_logits, dim=-1)
        same_prob = relation_prob[:, 0] + relation_prob[:, 1]
        confidence = relation_prob.max(dim=-1).values
        condition_same = condition[:, 0] + condition[:, 1]
        disease_score = (0.5 * same_prob + 0.5 * condition_same.detach()).clamp(0.0, 1.0)
        relation_factor = (0.25 + 0.75 * disease_score) * (0.5 + 0.5 * confidence)
        modulation = torch.exp((self.edge_scale * (disease_score - 0.5)).clamp(-2.0, 2.0))
        raw = base_strength.clamp_min(0.0) * quality_similarity.clamp(0.0, 1.0) * relation_factor * modulation
        return raw, disease_score, confidence

    def bounded_residual(self, delta, e1_corr):
        delta_rms = delta.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        base_rms = e1_corr.square().mean(dim=1, keepdim=True).sqrt()
        return torch.tanh(delta / delta_rms) * base_rms

    def writeback(self, e1_output, bounded_delta, enabled=True):
        if not enabled:
            return e1_output
        residual = 0.02 * torch.tanh(self.alpha) * bounded_delta
        return torch.cat((e1_output[:, :64], e1_output[:, 64:] + residual), dim=1)


def confidence_from_conditions(condition):
    p = condition.detach().clamp_min(1e-8)
    entropy = -(p * p.log()).sum(dim=-1) / torch.log(p.new_tensor(3.0))
    return (1.0 - entropy).clamp(0.0, 1.0)

