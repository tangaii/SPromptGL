"""HCER-v2: canonical FC identities, cross-fitted reference and masked evidence.

All tensors with 39 columns use canonical_edges, never hypothesis traversal order.
There is exactly one shared routing/aggregation implementation for E1--E9.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


def build_reference(x, y, train, test, edges, fold, inner_assignment=None):
    """Every train participant is held out of the reference fitting procedure."""
    x = np.asarray(x, np.float64)
    train, test = np.asarray(train), np.asarray(test)
    cov, fc = x[:, :48], x[:, 64 + np.asarray(edges)]
    if inner_assignment is None:
        inner_assignment = np.full(len(x), -1, dtype=np.int64)
        skf = StratifiedKFold(5, shuffle=True, random_state=40000 + fold)
        for k, (_, held) in enumerate(skf.split(cov[train], y[train])):
            inner_assignment[train[held]] = k
    mu = np.zeros_like(fc)
    fitted_ids, models = [], []
    for k in range(5):
        held = train[inner_assignment[train] == k]
        fit = train[(inner_assignment[train] != k) & (y[train] == 1)]
        assert len(fit) >= 20 and not np.intersect1d(held, fit).size
        model = Ridge(alpha=10., solver="svd").fit(cov[fit], fc[fit])
        mu[held] = model.predict(cov[held])
        fitted_ids.append(fit.tolist())
        models.append((model.coef_, model.intercept_))
    td = train[y[train] == 1]
    full = Ridge(alpha=10., solver="svd").fit(cov[td], fc[td])
    mu[test] = full.predict(cov[test])
    resid = fc[td] - mu[td]
    center = np.median(resid, axis=0)
    sigma = np.maximum(1.4826 * np.median(np.abs(resid - center), axis=0), 1e-4)
    z = np.clip((fc - mu) / sigma, -3, 3)
    # Only verified monotone adverse measures: DVARS, outlier fraction and mean FD.
    qc_cols = np.asarray([57, 58, 60])
    qc = x[:, qc_cols]
    qc_center = np.median(qc[train], axis=0)
    qc_scale = np.maximum(1.4826 * np.median(np.abs(qc[train] - qc_center), axis=0), .1)
    adverse = np.maximum((qc - qc_center) / qc_scale, 0).clip(0, 6)
    reliability = np.maximum(np.exp(-.1 * adverse.mean(1)), .5)
    assert np.isfinite(mu).all() and np.isfinite(z).all()
    return dict(mu=mu.astype("float32"), sigma=sigma.astype("float32"),
                z=z.astype("float32"), reliability=reliability.astype("float32"),
                inner_assignment=inner_assignment, fitted_ids=fitted_ids,
                reference_models=models + [(full.coef_, full.intercept_)],
                fit_td=td, qc_columns=qc_cols, qc_center=qc_center, qc_scale=qc_scale,
                oof_td_rms=float(np.sqrt(np.mean(resid ** 2))),
                oof_train_asd_z_rms=float(np.sqrt(np.mean(z[train[y[train] == 0]] ** 2))),
                oof_train_td_z_rms=float(np.sqrt(np.mean(z[td] ** 2))),
                test_z_rms=float(np.sqrt(np.mean(z[test] ** 2))))


def wrong_association(hypotheses):
    """A fixed cyclic derangement of query groups within exact-size strata."""
    association = [list(h["edges"]) for h in hypotheses]
    for size in (2, 3, 4):
        ids = [i for i, h in enumerate(hypotheses) if len(h["edges"]) == size]
        assert len(ids) > 1
        for a, b in zip(ids, ids[1:] + ids[:1]):
            association[a] = list(hypotheses[b]["edges"])
    assert all(set(a).isdisjoint(h["edges"]) for a, h in zip(association, hypotheses))
    return association


class HCER(nn.Module):
    def __init__(self, texts, anatomy, canonical_edges, hypotheses, variant="real", seed=40000):
        super().__init__()
        assert texts.shape == (13, 2, 1024) and anatomy.shape == (39, 1024)
        self.variant = variant
        self.canonical_edges = tuple(map(int, canonical_edges))
        self.index_map = {e: i for i, e in enumerate(self.canonical_edges)}
        groups = wrong_association(hypotheses) if variant == "wrong" else [h["edges"] for h in hypotheses]
        membership = torch.zeros(13, 39, dtype=torch.bool)
        direction = torch.zeros(39)
        for h, edges in enumerate(groups):
            for e in edges:
                membership[h, self.index_map[e]] = True
                direction[self.index_map[e]] = {"hyper_candidate": 1., "hypo_candidate": -1.}.get(hypotheses[h]["direction"], 0.)
        assert membership.sum().item() == 39 and membership.sum(0).eq(1).all()
        self.register_buffer("membership", membership)
        self.register_buffer("direction", direction)
        self.register_buffer("texts", F.normalize(texts.float(), dim=-1))
        self.register_buffer("anatomy", F.normalize(anatomy.float(), dim=-1))
        self.register_buffer("columns", torch.tensor(self.canonical_edges) + 64)
        self.register_buffer("threshold", torch.tensor(0.))
        self.text_projection = nn.Linear(1024, 64, bias=False)
        self.numeric_key = nn.Linear(5, 64, bias=False)
        self.cam_context = nn.Linear(144, 64, bias=False)
        self.pheno_context = nn.Linear(28, 64, bias=False)
        self.value = nn.Linear(69, 16, bias=False)
        self.fuse = nn.Linear(87, 144, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.))
        self.score_gain = nn.Parameter(torch.tensor(.5413248546))  # softplus = 1
        self.beta = nn.Parameter(torch.tensor(-1.734601055))  # .05 sigmoid = .0075
        self.last = {}
        self.text_override = None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for layer in (self.text_projection, self.numeric_key, self.cam_context,
                          self.pheno_context, self.value, self.fuse):
                fanin, fanout = layer.in_features, layer.out_features
                bound = (6 / (fanin + fanout)) ** .5
                layer.weight.copy_(torch.rand(layer.weight.shape, generator=generator) * (2 * bound) - bound)

    def text_geometry(self):
        t = self.texts if self.text_override is None else self.text_override
        mean = t.mean(1, keepdim=True)
        # Pair contrast retains common network semantics at a fixed 0.25 weight.
        q = F.normalize(self.text_projection(t - mean + .25 * mean), dim=-1)
        return q

    def selector(self, score, threshold, training):
        eligible = score > threshold
        indices = score.masked_fill(~eligible, -float("inf")).topk(8, dim=-1).indices
        hard = torch.zeros_like(score).scatter_(1, indices, 1.) * eligible.to(score)
        if not training:
            return hard, hard, eligible
        kth = score.topk(8, dim=-1).values[:, -1:].detach()
        cut = torch.maximum(kth, threshold)
        soft = torch.sigmoid((score - cut) / .25)
        # Exact hard forward, differentiable score backward. No unmasked output path.
        selected = hard + (soft - soft.detach())
        return selected, hard, eligible

    def forward(self, cam, raw, mu, sigma, reliability, mask_override=None,
                gate_disabled=False, fixed_delta=None, threshold_override=None):
        if gate_disabled:
            return cam, {}
        n = len(cam)
        fc = raw[:, self.columns]
        z = ((fc - mu) / sigma).clamp(-3, 3)
        absz = z.abs()
        prior = z * self.direction
        rel = reliability.reshape(n, 1)
        numeric = torch.stack([fc, z, absz, .25 * prior, rel.expand_as(z)], -1)
        anatomy = F.normalize(self.text_projection(self.anatomy), dim=-1)
        context = .10 * self.cam_context(cam) + .10 * self.pheno_context(torch.cat([raw[:, :26], raw[:, 46:48]], -1))
        key = F.normalize(anatomy[None] + .5 * self.numeric_key(numeric) + context[:, None], dim=-1)
        q = self.text_geometry()
        tau = .10 + .40 * torch.sigmoid(self.log_temperature)
        logits = torch.einsum("hcd,ned->nhce", q, key) / tau
        groupmask = self.membership[None, :, None, :]
        attention = logits.masked_fill(~groupmask, -1e9).softmax(-1)
        # Attention is truly normalized across legal edges, independently for each query.
        source = self.membership.long().argmax(0)
        edge_logits = logits[:, source, :, torch.arange(39, device=cam.device)].transpose(0, 1)
        edge_attention = attention[:, source, :, torch.arange(39, device=cam.device)].transpose(0, 1)
        assert edge_logits.shape == (n, 39, 2)
        sizes = self.membership.sum(1)[source].to(cam)
        if self.variant == "numeric":
            # The bank and Q/K shapes are retained; disease queries have no routing effect.
            a, t = .25 * prior, -.25 * absz
            states = torch.stack([a, t, torch.zeros_like(a)], -1).softmax(-1)
            edge_attention = torch.ones_like(edge_attention) / sizes[None, :, None]
            score = F.softplus(self.score_gain) * absz
            cutoff = score.new_tensor(1.) if threshold_override is None else threshold_override
        elif self.variant == "no_rival":
            generic = edge_logits.mean(-1)
            support = generic.sigmoid()
            states = torch.stack([.5 * support, .5 * support, 1 - support], -1)
            avg_attention = edge_attention.mean(-1, keepdim=True)
            edge_attention = avg_attention.expand_as(edge_attention)
            score = F.softplus(self.score_gain) * generic
            cutoff = self.threshold if threshold_override is None else threshold_override
        else:
            a = edge_logits[..., 0] + .25 * prior
            t = edge_logits[..., 1] - .25 * absz
            states = torch.stack([a, t, torch.zeros_like(a)], -1).softmax(-1)
            score = F.softplus(self.score_gain) * (torch.logsumexp(torch.stack([a, t], -1), -1)
                                                   - .25 * torch.minimum(states[..., 0], states[..., 1]))
            cutoff = self.threshold if threshold_override is None else threshold_override
        if mask_override is None:
            selected, hard, eligible = self.selector(score, cutoff, self.training)
        else:
            selected = mask_override.to(score)
            hard = (selected.detach() > .5).to(score)
            eligible = hard.bool()
        contradiction = 2 * torch.minimum(states[..., 0], states[..., 1])
        supports = torch.stack([states[..., 0] * edge_attention[..., 0],
                                states[..., 1] * edge_attention[..., 1],
                                (states[..., 2] + .5 * contradiction) * edge_attention.mean(-1)], -1)
        supports = supports * sizes[None, :, None] * rel[:, :, None]
        selected_support = supports * selected[:, :, None]
        values = self.value(torch.cat([anatomy[None].expand(n, -1, -1), numeric], -1))
        count = hard.sum(1, keepdim=True)
        denom = count.clamp_min(1.)
        summary = torch.einsum("nes,he->nhs", selected_support, self.membership.to(cam))
        summary = summary / self.membership.sum(1)[None, :, None].to(cam)
        aggregate = torch.einsum("nes,ned->nsd", selected_support, values) / denom[:, :, None]
        evidence = torch.cat([summary.flatten(1), aggregate.flatten(1)], -1)
        raw_delta = self.fuse(evidence)
        # This is shared train/test calibration, with no test labels or fitted test moments.
        # Same gate and normalized norm in E5/E9 prevents a residual-dose confound.
        norm = raw_delta.norm(dim=1, keepdim=True)
        camnorm = cam.norm(dim=1, keepdim=True).detach()
        positive_gate = .05 * self.beta.sigmoid()
        delta = positive_gate * camnorm * raw_delta / norm.clamp_min(1e-4)
        # Bias-free masked aggregation is exactly zero for an empty hard mask.
        # Avoid an extra hard activity multiplier that would kill its STE gradient.
        ratio = delta.norm(dim=1) / camnorm.flatten().clamp_min(1e-8)
        self.last = dict(score=score, selected=selected, hard=hard, eligible=eligible,
                         support=states, selected_support=selected_support, attention=attention,
                         summary=summary, evidence=evidence, delta=delta, ratio=ratio,
                         count=count.flatten(), z=z, q=q, key=key, tau=tau,
                         positive_gate=positive_gate, raw_delta_norm=norm.flatten())
        return cam + (delta if fixed_delta is None else fixed_delta), self.last
