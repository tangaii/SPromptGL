"""E82.3.2 Gaussian evidence with legal five-fold reference banks and post-evaluation EMA."""
from __future__ import annotations

from .history import CFG, E823, E8242, load, require, sha
from .spectral import EPS

import numpy as np
import torch
from torch import nn

E = CFG["evidence"]


def fit_reference(fold, warm_cam, spectrum, legal, labels):
    previous = load(E8242 / "G2_pdjc_da_srpe/checkpoints" / f"G2_f{fold:02d}.best.pt")["reference"]
    split_seed = E["archived_split_seed_base"] + fold
    require(previous["seed"] == split_seed and np.array_equal(previous["legal"], legal)
            and np.array_equal(previous["labels"], labels), "Archived legal inner split differs")
    group = previous["group"]
    counts = np.stack([np.bincount(labels[(group != i) if i else np.ones(len(group), bool)], minlength=2)
                       for i in range(6)])
    require(group.min() == 1 and group.max() == E["inner_folds"] and np.array_equal(counts, previous["counts"]),
            "Archived stratified inner bank counts differ")
    # Independent reconstruction is performed in preflight; production uses archived membership.
    require((counts > E["ddof"]).all(), "A reference bank lacks one class")
    h = ((warm_cam[legal].double() - spectrum["mu"]) @ spectrum["U"]) / (spectrum["lambda"] + EPS).sqrt()
    mask = ((group[None, :] != np.arange(6)[:, None])[:, None, :] &
            (labels[None, None, :] == np.arange(2)[None, :, None]))
    samples = [h[m] for m in mask.reshape(12, -1)]
    mean = torch.stack([x.mean(0) for x in samples]).reshape(6, 2, 16)
    variance = torch.stack([x.var(0, correction=E["ddof"]) for x in samples]).reshape_as(mean)
    counts_t = torch.from_numpy(counts)
    pooled, stable = stabilize(variance, counts_t)
    e823 = load(E823 / "runtime" / f"class_stats_f{fold:02d}.pt")
    require(np.array_equal(e823["legal_train"], legal) and
            sha({"mean": mean[0], "variance": variance[0], "pooled": pooled[0],
                 "stable": stable[0], "counts": counts_t[0]}) == e823["stats_sha"],
            "All-legal E82.3.2 initial Gaussian statistics differ")
    oof = gaussian(h, mean[group], stable[group])[1]
    y = torch.from_numpy(labels)
    signed = (1 - 2 * y[:, None]) * oof
    skill = 0.5 * (signed[y == 0].div(2).tanh().mean(0) + signed[y == 1].div(2).tanh().mean(0))
    calibration = 0.5 * (1 + skill)
    bank = np.zeros(CFG["subjects"], dtype=np.int64)
    bank[legal] = group
    require((bank[legal] > 0).all(), "Out-of-fold observations have no independent bank")
    biased = variance * ((counts_t[..., None] - E["ddof"]) / counts_t[..., None].double())
    return {"legal": legal, "labels": labels, "group": group, "seed": split_seed, "counts": counts_t,
            "mask": mask, "bank": torch.from_numpy(bank), "mean": mean, "second": biased + mean.square(),
            "stable": stable, "calibration": calibration, "h": h}


def stabilize(variance, count):
    pooled = (variance * count[..., None]).sum(-2) / count.sum(-1, keepdims=True)
    floor = (E["relative_floor"] * pooled).clamp_min(E["variance_floor"])
    return pooled, (E["own_variance"] * variance + E["pooled_variance"] * pooled.unsqueeze(-2)
                    + floor.unsqueeze(-2))


def gaussian(h, mu, variance):
    nll = (h[:, None, :] - mu).square() / variance + variance.log()
    raw = 0.5 * (nll[:, 1] - nll[:, 0])
    bound = E["llr_bound"]
    return raw, bound * (raw / bound).tanh()


class ClassEvidence(nn.Module):
    def __init__(self, reference, spectrum):
        super().__init__()
        self.mask = reference["mask"]
        self.center_cpu = spectrum["mu"]
        self.basis_cpu = spectrum["U"]
        self.scale_cpu = (spectrum["lambda"] + EPS).sqrt()
        # The EMA stays on CPU in float64. Snapshot methods are explicit so the
        # inference bank and resumable EMA state cannot be silently conflated.
        self.ema = {k: reference[k].clone() for k in ("mean", "second", "stable", "counts")}
        self.ema["updates"] = 0
        self.register_buffer("legal", torch.from_numpy(reference["legal"].copy()))
        self.register_buffer("bank", reference["bank"].clone())
        self.register_buffer("calibration", reference["calibration"].float().clone())
        self.register_buffer("inference_mean", reference["mean"].float().clone(), persistent=False)
        self.register_buffer("inference_variance", reference["stable"].float().clone(), persistent=False)
        self.register_buffer("counts", reference["counts"].clone())

    def snapshot(self):
        return {key: value.clone() if torch.is_tensor(value) else value for key, value in self.ema.items()}

    @torch.no_grad()
    def restore(self, snapshot):
        require(torch.equal(snapshot["counts"], self.ema["counts"]), "EMA reference count identity differs")
        self.ema = {key: val.clone() if torch.is_tensor(val) else val for key, val in snapshot.items()}
        self.inference_mean.copy_(self.ema["mean"])
        self.inference_variance.copy_(self.ema["stable"])

    @torch.no_grad()
    def observation(self, cam):
        h = ((cam[self.legal].detach().cpu().double() - self.center_cpu) @ self.basis_cpu) / self.scale_cpu
        per_bank = [h[m] for m in self.mask.reshape(12, -1)]
        summary = torch.stack([torch.stack((x.mean(0), x.square().mean(0))) for x in per_bank], dim=1)
        return summary.reshape(2, 6, 2, 16)

    @torch.no_grad()
    def advance(self, observation):
        new = self.snapshot()
        for key, obs in zip(("mean", "second"), observation):
            new[key] = (1 - E["alpha"]) * new[key] + E["alpha"] * obs
        n = new["counts"][..., None]
        variance = (new["second"] - new["mean"].square()).clamp_min(0) * (n / (n - E["ddof"]))
        new["stable"] = stabilize(variance, new["counts"])[1]
        new["updates"] += 1
        require(bool(torch.isfinite(new["stable"]).all() and (new["stable"] > 0).all()), "Invalid EMA variance")
        self.restore(new)

    def forward(self, white, training, reliability_off):
        bank = self.bank if training else slice(0, 1)
        mu, var, n = self.inference_mean[bank], self.inference_variance[bank], self.counts[bank]
        raw, bounded = gaussian(white, mu, var)
        pooled = var.mean(1)
        effect = (mu[:, 0] - mu[:, 1]).square() / pooled
        mean_uncertainty = (var / n[..., None]).sum(1) / pooled
        # Different class variance is a source of Gaussian evidence even at equal means.
        spread = 0.5 * (var[:, 0].log() - var[:, 1].log()).square()
        variance_uncertainty = (2 / (n.float() - E["ddof"])).sum(1, keepdim=True)
        signal = effect + spread
        moment_trust = signal / (signal + mean_uncertainty + variance_uncertainty + E["effect_prior"] ** 2)
        distance = (white - mu.mean(1)) / pooled.sqrt()
        support = torch.rsqrt(1 + (distance / E["support_radius"]).square())
        reliability = moment_trust * support * self.calibration
        if reliability_off:
            reliability = torch.ones_like(reliability)
        return raw, bounded, reliability
