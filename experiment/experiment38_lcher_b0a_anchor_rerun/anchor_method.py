"""Experiment 32 DCER algorithmic reference; integrated in isolated runtime."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

VALID = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
K = 4


def bounded_update(raw: torch.Tensor, base: torch.Tensor, maximum: float = .20) -> torch.Tensor:
    value = torch.tanh(raw)
    norm = value.norm(dim=-1, keepdim=True)
    base_norm = base.detach().norm(dim=-1, keepdim=True)
    return maximum * base_norm * value / (1.0 + norm)


class DCER32(nn.Module):
    def __init__(self, text_features: torch.Tensor, config: str, fold: int):
        super().__init__()
        assert config in VALID
        assert tuple(text_features.shape) == (8, 1024)
        self.config = config
        self.fold = int(fold)
        self.register_buffer("text_bank", text_features.detach().float().clone())
        self.register_buffer("random_step", torch.zeros((), dtype=torch.long))
        self.query = nn.Linear(9, 32, bias=False)
        self.key = nn.Linear(1024, 32, bias=False)
        self.value = nn.Linear(9, 18, bias=False)
        self.fuse = nn.Linear(54, 36, bias=False)
        nn.init.zeros_(self.fuse.weight)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.numeric_logits = nn.Linear(9, 3)
        if config == "D6":
            generator = torch.Generator(device="cpu").manual_seed(32006)
            noise = torch.randn((8, 1024), generator=generator).to(self.text_bank.device)
            mean = self.text_bank.mean(0, keepdim=True)
            std = self.text_bank.std(0, keepdim=True).clamp_min(1.e-5)
            self.register_buffer("nonmedical_bank", noise * std + mean)
        if config == "D5":
            self.register_buffer("mixed_order", torch.tensor([4, 1, 6, 3, 0, 5, 2, 7]))
        self.extra = {}

    def _scores(self, part: torch.Tensor):
        if self.config == "D0":
            logits = self.numeric_logits(part)
            post = torch.softmax(logits, dim=-1)
            pa, pt, pn = post.unbind(-1)
            sas, std = logits[..., 0], logits[..., 1]
        else:
            bank = self.text_bank
            if self.config == "D5":
                bank = bank[self.mixed_order]
            elif self.config == "D6":
                bank = self.nonmedical_bank
            query = F.normalize(self.query(part), dim=-1, eps=1.e-8)
            key = F.normalize(self.key(bank), dim=-1, eps=1.e-8)
            similarity = query @ key.T / .20
            std = torch.logsumexp(similarity[..., :4], dim=-1) - math.log(4.)
            sas = torch.logsumexp(similarity[..., 4:], dim=-1) - math.log(4.)
            if self.config == "D7":
                probabilities = torch.softmax(torch.stack((sas, std), -1), -1)
                pa, pt = probabilities.unbind(-1)
                pn = torch.zeros_like(pa)
            else:
                probabilities = torch.softmax(
                    torch.stack((sas, std, self.null_logit.expand_as(sas)), -1), -1
                )
                pa, pt, pn = probabilities.unbind(-1)
        if self.config == "D1":
            salience = torch.maximum(pa, pt)
            directional = torch.ones_like(salience)
        else:
            directional = pa - pt
            salience = directional.abs() * (pa + pt)
        return pa, pt, pn, sas, std, salience, directional

    @staticmethod
    def _top4(salience: torch.Tensor, eligible: torch.Tensor, exclude: torch.Tensor | None = None):
        values = salience.masked_fill(~eligible, -1.e9)
        if exclude is not None:
            values = values.masked_fill(exclude, -1.e9)
        ids = values.topk(K, dim=-1).indices
        hard = torch.zeros_like(salience).scatter_(-1, ids, 1.)
        hard = hard * eligible.to(hard.dtype)
        if exclude is not None:
            hard = hard * (~exclude).to(hard.dtype)
        empty = hard.sum(-1, keepdim=True) == 0
        fallback_values = salience if exclude is None else salience.masked_fill(exclude, -1.e9)
        fallback = torch.zeros_like(salience).scatter_(-1, fallback_values.argmax(-1, keepdim=True), 1.)
        return torch.where(empty, fallback, hard)

    def _route(self, salience: torch.Tensor, pa: torch.Tensor, pt: torch.Tensor, pn: torch.Tensor):
        n, dimensions = salience.shape
        if self.config == "D2":
            return salience.new_ones((n, dimensions)), salience.new_ones((n, dimensions)), salience.new_zeros((n, dimensions))
        if self.config == "D8":
            step = int(self.random_step.item()) if self.training else 0
            seed = 3200800 + self.fold * 10000 + step
            generator = torch.Generator(device="cpu").manual_seed(seed)
            random_scores = torch.rand((n, dimensions), generator=generator).to(salience.device)
            if self.training:
                self.random_step.add_(1)
            hard = self._top4(random_scores, torch.ones_like(salience, dtype=torch.bool))
            alternate = self._top4(random_scores, torch.ones_like(salience, dtype=torch.bool), exclude=hard.bool())
            return hard, hard.detach(), alternate.detach()
        eligible = (pa + pt) > pn
        hard = self._top4(salience, eligible)
        next_best = self._top4(salience, torch.ones_like(eligible), exclude=hard.bool())
        lo = salience.detach().min(-1, keepdim=True).values - 2.
        hi = salience.detach().max(-1, keepdim=True).values + 2.
        for _ in range(24):
            mid = (lo + hi) / 2.
            count = torch.sigmoid((salience.detach() - mid) / .10).sum(-1, keepdim=True)
            lo = torch.where(count > K, mid, lo)
            hi = torch.where(count > K, hi, mid)
        mid = (lo + hi) / 2.
        soft = torch.sigmoid((salience - mid) / .10) * torch.sigmoid(((pa + pt) - pn) / .10)
        selection = hard - soft.detach() + soft
        return selection, hard.detach(), next_best.detach()

    def _view(self, part, pa, pt, direction, source_mask, evidence_mask):
        source = part * source_mask[..., None]
        values = self.value(source).reshape(-1, 4, 4, 18)
        evidence = evidence_mask.reshape(-1, 4, 4)
        pa = pa.reshape(-1, 4, 4)
        pt = pt.reshape(-1, 4, 4)
        direction = direction.reshape(-1, 4, 4)
        denominator = evidence.sum(2, keepdim=True).clamp_min(1.)
        evidence_asd = ((evidence * pa * direction)[..., None] * values).sum(2) / denominator
        evidence_td = ((evidence * pt * direction)[..., None] * values).sum(2) / denominator
        fused = self.fuse(torch.cat((evidence_asd, evidence_td, evidence_asd - evidence_td), dim=-1))
        base = source.reshape(-1, 4, 36)
        return base + bounded_update(fused, base, .20)

    def forward(self, cam: torch.Tensor):
        assert cam.ndim == 3 and cam.shape[1:] == (4, 36)
        n = cam.shape[0]
        part = cam.reshape(n, 16, 9)
        pa, pt, pn, sas, std, salience, signed = self._scores(part)
        route, selected, alternate = self._route(salience, pa, pt, pn)
        all_on = torch.ones_like(selected)
        full = self._view(part, pa, pt, signed, all_on, route)
        keep = drop = None
        if self.training and torch.is_grad_enabled() and self.config in ("D4", "D5", "D6", "D7", "D8"):
            keep = self._view(part, pa, pt, signed, selected, route)
            drop = self._view(part, pa, pt, signed, 1. - selected, alternate)
        contrast = (route * (sas - std)).sum(-1) / route.sum(-1).clamp_min(1.)
        self.extra = {"selected": selected.detach(), "route": route, "pa": pa, "pt": pt,
                      "pn": pn, "contrast": contrast, "keep": keep, "drop": drop,
                      "salience": salience, "full": full}
        return full, keep, drop

    def extra_loss(self, labels, train, logp_keep=None, logp_drop=None):
        if self.config not in VALID:
            raise ValueError(self.config)
        ids = torch.as_tensor(train, dtype=torch.long, device=self.text_bank.device)
        targets = labels[ids]
        loss = self.extra["contrast"].sum() * 0.
        if self.config not in ("D0", "D1"):
            logits = torch.stack((self.extra["contrast"], -self.extra["contrast"]), dim=-1)
            loss = loss + .02 * F.cross_entropy(logits[ids], targets)
        if self.config in ("D4", "D5", "D6", "D7", "D8"):
            assert logp_keep is not None and logp_drop is not None
            keep_logp = logp_keep[ids]
            drop_logp = logp_drop[ids]
            loss_sufficient = F.nll_loss(keep_logp, targets)
            selected_logp = keep_logp.gather(1, targets[:, None]).squeeze(-1)
            deleted_logp = drop_logp.gather(1, targets[:, None]).squeeze(-1)
            loss_necessary = F.relu(.10 + deleted_logp - selected_logp).mean()
            loss = loss + .10 * loss_sufficient + .05 * loss_necessary
        return loss
