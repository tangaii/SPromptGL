"""D²-ARG: disease-discriminative relation learning and bounded propagation."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None

def grad_reverse(x, scale=0.03):
    return GradientReverse.apply(x, scale)

class RelationEncoder(nn.Module):
    def __init__(self, in_dim=256, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
    def forward(self, x):
        return self.net(x)

class RelationScorer(nn.Module):
    """Symmetric scorer for [|zi-zj|, zi*zj, prior_ij]."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
    @staticmethod
    def pair_features(z, i, j, prior):
        zn = F.normalize(z, dim=-1)
        zi, zj = zn[i], zn[j]
        return torch.cat(((zi - zj).abs(), zi * zj, prior.reshape(-1, 1)), dim=-1)
    def forward(self, features):
        return self.net(features).squeeze(-1)

class SitePairDiscriminator(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden * 2 + 1 + 2, hidden), nn.GELU(), nn.Linear(hidden, 2))
    def forward(self, features, disease_condition):
        c = F.one_hot(disease_condition.long(), num_classes=2).to(features.dtype)
        return self.net(torch.cat((features, c), dim=-1))

class D2ARGCore(nn.Module):
    """One CORR relation encoder, scorer and conditional site adversary."""
    def __init__(self, hidden=32, grl_scale=0.03):
        super().__init__()
        self.encoder = RelationEncoder(256, hidden)
        self.scorer = RelationScorer(hidden)
        self.site = SitePairDiscriminator(hidden)
        self.grl_scale = float(grl_scale)
    def encode(self, corr):
        return self.encoder(corr)
    def pair_forward(self, corr, pair_i, pair_j, disease_y, site_y=None, adv=False, adv_condition=None):
        z = self.encode(corr)
        norm_corr = F.normalize(corr, dim=-1)
        prior = norm_corr[pair_i].mul(norm_corr[pair_j]).sum(-1)
        features = self.scorer.pair_features(z, pair_i, pair_j, prior)
        logits = self.scorer(features)
        zero = logits.new_zeros(())
        pair_loss = F.binary_cross_entropy_with_logits(logits, disease_y.to(logits.dtype)) if disease_y.numel() else zero
        site_logits = None
        site_loss = zero
        if adv and site_y is not None and site_y.numel():
            condition = disease_y if adv_condition is None else adv_condition
            site_logits = self.site(grad_reverse(features, self.grl_scale), condition)
            site_loss = F.cross_entropy(site_logits, site_y.long())
        return z, prior, features, logits, pair_loss, site_loss, site_logits
    def optimizer_param_groups(self, lr=0.001, weight_decay=0.11):
        return [{"params": list(self.parameters()), "lr": float(lr), "weight_decay": float(weight_decay)}]

def module_grad_norm(module):
    vals = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().detach().cpu()) if vals else 0.0

def pair_auc(logits, labels):
    from sklearn.metrics import roc_auc_score
    y = labels.detach().cpu().numpy()
    p = torch.sigmoid(logits).detach().cpu().numpy()
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else 0.5

def bounded(raw, maximum):
    return float(maximum) * torch.sigmoid(raw)
