"""SCORE: Subject-specific Connectomic Evidence and Reliability-aware Modulation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N = 871
D = 256
VALID = 252
STATE_COUNT = 5
MASK_FRACTION = 0.15
MASK_LOSS = 0.02

CONFIGS = (
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
    "B0", "B1", "B2", "B3", "B4", "B5",
)
GPU0 = ("A0", "A2", "A4", "A6", "A8", "A10", "B0", "B2", "B4")
GPU1 = ("A1", "A3", "A5", "A7", "A9", "A11", "B1", "B3", "B5")
GPU = {c: 0 for c in GPU0} | {c: 1 for c in GPU1}
METRICS = ("acc", "auc", "sen_asd", "spe_td")


@dataclass(frozen=True)
class Design:
    config: str
    active: bool
    report: str
    pheno: bool
    func: bool
    anat: bool
    dense_aux: bool
    masked: bool
    pheno_mismatch: bool = False
    qc_mismatch: bool = False


def _designs():
    d = {"A0": Design("A0", False, "OFF", False, False, False, False, False)}
    d["A1"] = Design("A1", True, "NUMERIC_STATE", False, False, False, False, True)
    d["A2"] = Design("A2", True, "MEDICAL", False, False, False, False, True)
    d["A3"] = Design("A3", True, "MEDICAL", True, False, False, False, True)
    d["A4"] = Design("A4", True, "MEDICAL", True, True, False, False, True)
    d["A5"] = Design("A5", True, "MEDICAL", True, True, True, False, True)
    d["A6"] = Design("A6", True, "MEDICAL", True, True, True, False, False)
    d["A7"] = Design("A7", True, "MEDICAL", False, True, True, False, True)
    d["A8"] = Design("A8", True, "MEDICAL", True, False, True, False, True)
    d["A9"] = Design("A9", True, "MEDICAL", True, False, False, True, True)
    d["A10"] = Design("A10", True, "MEDICAL", True, True, True, False, True, pheno_mismatch=True)
    d["A11"] = Design("A11", True, "MEDICAL", True, True, True, False, True, qc_mismatch=True)
    for c, report in {
        "B0": "WRONG_ROI", "B1": "REVERSE", "B2": "DEGREE_SWAP",
        "B3": "SITE_MISMATCH", "B4": "NUMERIC_MATCH", "B5": "ORIGINAL8",
    }.items():
        d[c] = Design(c, True, report, True, True, True, False, True)
    return d


DESIGNS = _designs()
assert set(DESIGNS) == set(CONFIGS)
assert set(GPU) == set(CONFIGS) and len(GPU0) == len(GPU1) == 9


def make_fold_reference(corr, train_idx, sites):
    """Fit robust CORR reference and site reliability on outer-train only."""
    assert np.asarray(corr).shape == (N, D)
    x = np.asarray(corr, np.float64)
    tr = np.asarray(train_idx, np.int64)
    train_x = x[tr]
    med = np.median(train_x, axis=0)
    mad = np.median(np.abs(train_x - med), axis=0)
    robust_scale = 1.4826 * mad
    train_std = np.std(train_x, axis=0, ddof=0)
    scale = np.where(robust_scale > 1e-4, robust_scale, np.maximum(train_std, 1e-3))
    scale = np.maximum(scale, 1e-3)
    site = np.asarray(sites)[tr]
    means, counts = [], []
    within_num = np.zeros(D, dtype=np.float64)
    within_den = 0
    for s in np.unique(site):
        block = train_x[site == s]
        if len(block) < 2:
            continue
        mean_s = block.mean(0)
        means.append(mean_s)
        counts.append(len(block))
        within_num += ((block - mean_s) ** 2).sum(0)
        within_den += len(block)
    if not means:
        reliability = np.ones(D, dtype=np.float64)
    else:
        means = np.asarray(means)
        counts = np.asarray(counts, dtype=np.float64)
        weights = counts / counts.sum()
        grand = (weights[:, None] * means).sum(0)
        between = (weights[:, None] * (means - grand) ** 2).sum(0)
        within = within_num / max(within_den, 1)
        reliability = within / (within + between + 1e-8)
    z = np.clip((x - med) / scale, -3.0, 3.0)
    assert np.isfinite(z).all() and np.isfinite(reliability).all()
    return {
        "z": z.astype(np.float32), "reliability": np.clip(reliability, 0., 1.).astype(np.float32),
        "median": med.astype(np.float32), "scale": scale.astype(np.float32),
        "train_sites": int(len(np.unique(site))), "singleton_sites": int(sum((site == s).sum() < 2 for s in np.unique(site))),
    }


def state_from_z(z):
    states = np.full(np.shape(z), 2, dtype=np.int64)
    states[np.asarray(z) < -1.5] = 0
    states[(np.asarray(z) >= -1.5) & (np.asarray(z) < -0.5)] = 1
    states[(np.asarray(z) > 0.5) & (np.asarray(z) <= 1.5)] = 3
    states[np.asarray(z) > 1.5] = 4
    assert states.min() >= 0 and states.max() <= 4
    return states


def bounded_delta(delta_raw, reference, fraction=0.05):
    direction = torch.tanh(delta_raw)
    size = direction.norm(dim=-1, keepdim=True)
    ref = reference.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return fraction * ref * direction / (1. + size)


def _sha_array(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


class EvidenceCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.numeric = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.text_proj = nn.Linear(64, 64, bias=False)
        self.demographic = nn.Sequential(nn.Linear(28, 64), nn.GELU(), nn.Linear(64, 128))
        nn.init.zeros_(self.demographic[-1].weight)
        nn.init.zeros_(self.demographic[-1].bias)
        self.edge_score = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.edge_value = nn.Linear(64, 64, bias=False)
        self.output = nn.Linear(64, 36, bias=False)
        nn.init.zeros_(self.output.weight)
        self.func_gate = nn.Sequential(nn.Linear(10, 16), nn.GELU(), nn.Linear(16, 1))
        self.anat_gate = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        self.mask_query = nn.Linear(64, 64, bias=False)
        self.dense_aux = nn.Sequential(nn.Linear(16, 18), nn.GELU(), nn.Linear(18, 1))


class SCORE(nn.Module):
    """A fold-scoped SCORE branch installed at the original SPGS location."""
    def __init__(self, config, raw, train, test, sites, pheno_indices, bank, fold, seed=0):
        super().__init__()
        assert config in CONFIGS
        self.config = config
        self.design = DESIGNS[config]
        self.condition_override = None
        self.last: Dict[str, torch.Tensor] = {}
        self.statistics = make_fold_reference(np.asarray(raw)[:, 64:], train, sites)
        valid = np.asarray(bank["valid_indices"], dtype=np.int64)
        assert valid.shape == (VALID,) and set(valid.tolist()) == set(range(D)) - set((195, 225, 232, 236))
        self.valid_indices = valid
        z = self.statistics["z"][:, valid]
        states = state_from_z(z)
        for name, value, dtype in (
            ("z_valid", z, torch.float32), ("states", states, torch.long),
            ("reliability", self.statistics["reliability"][valid], torch.float32),
            ("valid_indices_t", valid, torch.long), ("train_ids", np.asarray(train), torch.long),
            ("test_ids", np.asarray(test), torch.long), ("site", np.asarray(sites), torch.long),
            ("pheno_indices", np.asarray(pheno_indices), torch.long),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        for name in ("text_bank", "original8", "numeric_match", "numeric_state", "wrong_roi"):
            self.register_buffer(name, torch.as_tensor(bank[name], dtype=torch.float32))
        site_permutations = np.asarray(bank["site_permutation_folds"], dtype=np.int64)
        assert site_permutations.shape == (10, N)
        self.register_buffer("site_permutation", torch.as_tensor(site_permutations[int(fold) - 1], dtype=torch.long))
        self.reference_sha256 = _sha_array(z)
        self.reliability_sha256 = _sha_array(self.statistics["reliability"])
        self.register_buffer("_zero", torch.tensor(0., dtype=torch.float32))
        if not self.design.active:
            return
        self.core = EvidenceCore()
        self.masked_enabled = bool(self.design.masked)
        self.register_buffer("mask_rng_state", torch.tensor(310000 + int(seed) * 100 + int(fold), dtype=torch.int64))

    def _state_binding(self):
        d = self.condition_override or self.design.report
        if d == "OFF":
            return self.text_bank.new_zeros((N, VALID, 64))
        if d == "ORIGINAL8":
            fixed = self.original8.mean(dim=0)
            return fixed[None, None, :].expand(N, VALID, 64)
        if d == "NUMERIC_STATE":
            base = self.numeric_state
        elif d == "NUMERIC_MATCH":
            base = self.numeric_match
        elif d == "WRONG_ROI":
            base = self.wrong_roi
        else:
            base = self.text_bank
        states = self.states
        if d == "REVERSE":
            states = torch.as_tensor([4, 3, 2, 1, 0], device=states.device)[states]
        elif d == "DEGREE_SWAP":
            states = torch.as_tensor([1, 0, 2, 4, 3], device=states.device)[states]
        elif d == "SITE_MISMATCH":
            states = states[self.site_permutation]
        edge = torch.arange(VALID, device=states.device)[None, :]
        return base[edge, states]

    def _aux_inputs(self, raw):
        ph = raw[:, self.pheno_indices]
        anat = raw[:, 48:54]
        func = raw[:, 54:64]
        if self.design.pheno_mismatch:
            ph = ph[self.site_permutation]
        if self.design.qc_mismatch:
            anat, func = anat[self.site_permutation], func[self.site_permutation]
        return ph, anat, func

    def _masked_key_bank(self):
        if self.design.report == "NUMERIC_STATE":
            return self.numeric_state
        if self.design.report == "NUMERIC_MATCH":
            return self.numeric_match
        return self.text_bank

    def _masked_state_loss(self, raw):
        if not self.training or not torch.is_grad_enabled() or not self.masked_enabled:
            return raw.sum() * 0., None
        device = raw.device
        mask = torch.zeros((N, VALID), dtype=torch.bool, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.mask_rng_state.item()))
        draw = torch.rand((len(self.train_ids), VALID), device=device, generator=generator) < MASK_FRACTION
        mask[self.train_ids] = draw
        if not bool(mask.any()):
            mask[self.train_ids[0], 0] = True
        visible = ~mask
        z_context = self.z_valid.to(device).masked_fill(mask, 0.)
        r = self.reliability.to(device)[None].expand_as(z_context)
        numeric = self.core.numeric(torch.stack((z_context, z_context.abs().masked_fill(mask, 0.), r), dim=-1))
        selected_visible = numeric.new_zeros((N, VALID, 64))
        edge = torch.arange(VALID, device=device)[None, :].expand(N, -1)
        # Gather state text only for visible positions. Hidden state IDs and
        # embeddings never enter this prediction branch.
        key_bank = self._masked_key_bank()
        selected_visible[visible] = key_bank[edge[visible], self.states.to(device)[visible]]
        text_visible = self.core.text_proj(selected_visible)
        token = F.normalize(numeric, dim=-1, eps=1e-8) * F.normalize(text_visible, dim=-1, eps=1e-8)
        token = token.masked_fill(mask[..., None], 0.)
        context = token.sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
        roi_anchor = self.core.text_proj(key_bank.mean(dim=1))
        queries = self.core.mask_query(context[:, None, :] + roi_anchor[None, :, :])
        keys = self.core.text_proj(key_bank)
        scores = torch.einsum("ned,esd->nes", F.normalize(queries, dim=-1, eps=1e-8), F.normalize(keys, dim=-1, eps=1e-8)) / .2
        target = self.states.to(device)[mask]
        loss = F.cross_entropy(scores[mask], target)
        assert torch.isfinite(loss) and loss.requires_grad
        self.last["mask"] = mask
        self.last["masked_loss"] = loss
        self.last["masked_queries"] = queries
        self.last["masked_scores"] = scores
        self.mask_rng_state.add_(1)
        return loss, mask

    def forward(self, cam, raw_input):
        if not self.design.active:
            self.last = {"delta": cam[:, 3, :].new_zeros(cam[:, 3, :].shape)}
            return cam
        assert cam.shape == (N, 4, 36) and raw_input.shape == (N, 320)
        z = self.z_valid.to(raw_input.device)
        r = self.reliability.to(raw_input.device)[None].expand_as(z)
        ph, anat, func = self._aux_inputs(raw_input)
        q = self.core.numeric(torch.stack((z, z.abs(), r), dim=-1))
        film_norm = q.new_zeros(())
        if self.design.pheno:
            params = self.core.demographic(ph)
            gamma, beta = params.chunk(2, dim=-1)
            q = q * (1 + gamma[:, None, :]) + beta[:, None, :]
            film_norm = (gamma.square().mean() + beta.square().mean()).sqrt()
        selected = self._state_binding().to(raw_input.device)
        t = self.core.text_proj(selected)
        interaction = F.normalize(q, dim=-1, eps=1e-8) * F.normalize(t, dim=-1, eps=1e-8)
        scores = self.core.edge_score(interaction).squeeze(-1) + torch.log(r.clamp_min(1e-6))
        attention = scores.softmax(dim=-1)
        evidence = (attention[..., None] * self.core.edge_value(interaction)).sum(1)
        gate_func = raw_input.new_ones((N,))
        gate_anat = raw_input.new_ones((N,))
        if self.design.func:
            gate_func = torch.sigmoid(self.core.func_gate(func)).squeeze(-1)
        if self.design.anat:
            gate_anat = torch.sigmoid(self.core.anat_gate(anat)).squeeze(-1)
        if self.design.dense_aux:
            dense_scale = self.core.dense_aux(torch.cat((anat, func), dim=-1))
            evidence = evidence * (1. + 0.1 * torch.tanh(dense_scale))
        raw_delta = self.core.output(evidence) * (gate_func * gate_anat)[:, None]
        target = cam[:, 3, :]
        delta = bounded_delta(raw_delta, target, .05)
        out = torch.cat((cam[:, :3, :], (target + delta)[:, None, :]), dim=1)
        mask_loss, mask = self._masked_state_loss(raw_input)
        self.last.update({
            "z": z, "states": self.states.to(raw_input.device), "selected": selected,
            "attention": attention, "evidence": evidence, "delta": delta,
            "target": target, "gate_func": gate_func, "gate_anat": gate_anat,
            "film_norm": film_norm, "masked_loss": mask_loss,
            "mask": mask if mask is not None else torch.zeros((N, VALID), dtype=torch.bool, device=z.device),
        })
        assert torch.isfinite(out).all()
        return out

    def extra_loss(self):
        if not self.design.active or not self.masked_enabled:
            if self.design.active:
                return next(self.parameters()).sum() * 0.
            return self._zero.to(self.z_valid.device)
        return MASK_LOSS * self.last.get("masked_loss", next(self.parameters()).sum() * 0.)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters()) if self.design.active else 0


def self_test():
    x = np.zeros((N, D), np.float32)
    x[0, 0] = -2.; x[1, 0] = -1.; x[2, 0] = 0.; x[3, 0] = .6; x[4, 0] = 2.
    s = state_from_z(x)
    assert s[0, 0] == 0 and s[1, 0] == 1 and s[2, 0] == 2 and s[3, 0] == 3 and s[4, 0] == 4
    assert len(CONFIGS) == 18 and len(set(GPU0) & set(GPU1)) == 0
    print("SCORE_METHOD_CONTRACT_PASS")


if __name__ == "__main__":
    self_test()
