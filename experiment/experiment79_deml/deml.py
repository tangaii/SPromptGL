"""Small patient-conditioned error-memory attention for Experiment 79."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from memory_bank import TYPE_ASD_TO_TD, TYPE_TD_TO_ASD


class DEMLMemoryAttention(nn.Module):
    """CAM144 -> memory attention -> CAM144 feature enhancement.

    The module has no Dropout or stochastic layers.  It can therefore be
    trained after the historical DARG forward without consuming the shared
    model RNG.  `use_direction=False` is the exact G1 masking operation.
    """

    def __init__(self, cam_dim=144, memory_dim=64):
        super().__init__()
        self.cam_dim = int(cam_dim)
        self.memory_dim = int(memory_dim)
        self.query = nn.Linear(cam_dim, memory_dim)
        self.key = nn.Linear(cam_dim, memory_dim)
        self.value_h = nn.Linear(cam_dim, memory_dim)
        self.value_u = nn.Linear(cam_dim, memory_dim)
        self.query_norm = nn.LayerNorm(memory_dim)
        self.key_norm = nn.LayerNorm(memory_dim)
        self.error_type = nn.Embedding(2, memory_dim)
        # 144 + two 64-dim reads + 2 base log probabilities + two availability bits
        self.enhance = nn.Sequential(
            nn.Linear(cam_dim + 2 * memory_dim + 4, 128),
            nn.GELU(),
            nn.Linear(128, cam_dim),
        )
        nn.init.zeros_(self.enhance[-1].weight)
        nn.init.zeros_(self.enhance[-1].bias)

    @staticmethod
    def _type_index(typ):
        return 0 if typ == TYPE_ASD_TO_TD else 1

    def _read_one(self, qrow, candidates, use_direction, typ):
        if candidates is None:
            return qrow.new_zeros(self.memory_dim), False, 0, [], qrow.new_zeros((0,))
        features = candidates["features"]
        directions = candidates["directions"]
        priority = candidates["priority"]
        if features.numel() == 0:
            return qrow.new_zeros(self.memory_dim), False, 0, [], qrow.new_zeros((0,))
        k = self.key_norm(self.key(features))
        q = self.query_norm(qrow.unsqueeze(0)).squeeze(0)
        score = (k @ q) / math.sqrt(self.memory_dim)
        # Priority is an attention bias, rather than a copied class score.
        score = score + torch.log(priority.clamp_min(1e-6))
        alpha = torch.softmax(score, dim=0)
        type_idx = torch.full((features.shape[0],), self._type_index(typ), dtype=torch.long,
                              device=features.device)
        values = self.value_h(features) + self.error_type(type_idx)
        if use_direction:
            values = values + self.value_u(directions)
        else:
            values = values + self.value_u(torch.zeros_like(directions))
        return alpha @ values, True, int(features.shape[0]), candidates["patient_ids"], alpha

    def forward(self, query_ids, h, base_logp, bank, sites, *, use_direction=True,
                error_type_swap=False, feature_permuted=False,
                recency_off=False, direction_permuted=False):
        device = h.device
        n = h.shape[0]
        query_repr = self.query_norm(self.query(h))
        candidates = bank.retrieval(
            query_ids, h.detach(), sites, device,
            error_type_swap=error_type_swap,
            feature_permuted=feature_permuted,
            recency_off=recency_off,
            direction_permuted=direction_permuted,
            query_repr=query_repr.detach(),
            key_projection=self.key,
        )
        reads_a, reads_b, masks = [], [], []
        retrieval_rows = []
        for i, row in enumerate(candidates):
            # Recompute attention with the projected current query; candidate
            # selection already happened in the bank before value aggregation.
            def read_with_q(cand, typ):
                if cand is None or cand["features"].numel() == 0:
                    return h.new_zeros(self.memory_dim), False, 0, [], h.new_zeros((0,))
                feat, dirs, pri = cand["features"], cand["directions"], cand["priority"]
                kk = self.key_norm(self.key(feat))
                sc = (kk @ query_repr[i]) / math.sqrt(self.memory_dim) + torch.log(pri.clamp_min(1e-6))
                al = torch.softmax(sc, dim=0)
                idx = torch.full((feat.shape[0],), self._type_index(typ), dtype=torch.long, device=device)
                vv = self.value_h(feat) + self.error_type(idx)
                vv = vv + (self.value_u(dirs) if use_direction else self.value_u(torch.zeros_like(dirs)))
                return al @ vv, True, int(feat.shape[0]), cand["patient_ids"], al
            ra, ma, ca, pa, aa = read_with_q(row[0], TYPE_ASD_TO_TD)
            rb, mb, cb, pb, ab = read_with_q(row[1], TYPE_TD_TO_ASD)
            reads_a.append(ra); reads_b.append(rb)
            masks.append(torch.tensor([float(ma), float(mb)], device=device))
            retrieval_rows.append({
                "asd_to_td_patient_ids": pa, "td_to_asd_patient_ids": pb,
                "asd_to_td_attention": aa.detach(), "td_to_asd_attention": ab.detach(),
                "asd_to_td_count": ca, "td_to_asd_count": cb,
            })
        ra = torch.stack(reads_a) if reads_a else h.new_zeros((0, self.memory_dim))
        rb = torch.stack(reads_b) if reads_b else h.new_zeros((0, self.memory_dim))
        avail = torch.stack(masks) if masks else h.new_zeros((0, 2))
        t = torch.cat((h, ra, rb, base_logp, avail), dim=1)
        delta = self.enhance(t)
        hnorm = h.norm(dim=1, keepdim=True)
        dnorm = delta.norm(dim=1, keepdim=True)
        cap = 0.1 * hnorm
        scale = torch.minimum(torch.ones_like(dnorm), cap / dnorm.clamp_min(1e-12))
        delta = delta * scale
        empty = avail.sum(dim=1, keepdim=True).eq(0)
        delta = torch.where(empty, torch.zeros_like(delta), delta)
        return h + delta, {
            "delta_h": delta,
            "read_asd_to_td": ra,
            "read_td_to_asd": rb,
            "availability": avail,
            "retrieval": retrieval_rows,
            "h_norm": hnorm.squeeze(1).detach(),
            "delta_norm": delta.norm(dim=1).detach(),
            "delta_ratio": (delta.norm(dim=1) / hnorm.squeeze(1).clamp_min(1e-12)).detach(),
        }


def init_pair(seed=7901, device=None):
    """Initialise identical G1/G2 modules in an isolated RNG fork."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        first = DEMLMemoryAttention().to(device)
        second = DEMLMemoryAttention().to(device)
        second.load_state_dict(first.state_dict())
    return first, second


def output_layer_logp(output_layer, h):
    """Run the historical OutputLayer as an eval, detached-parameter head."""
    from torch.func import functional_call
    output_layer.eval()
    params = {name: value.detach() for name, value in output_layer.named_parameters()}
    buffers = {name: value.detach() for name, value in output_layer.named_buffers()}
    logp, hidden = functional_call(output_layer, (params, buffers), (h, None))
    return logp, hidden


def memory_loss(logp_mem, base_logp, labels, query_ids, class_weights, delta=None, h=None):
    """Return fixed E79 loss and scalar components."""
    ids = torch.as_tensor(query_ids, dtype=torch.long, device=logp_mem.device)
    y = labels[ids].long()
    # The branch passes BaseEval probabilities aligned with query_ids.
    base = base_logp.detach()
    weights = class_weights.to(logp_mem.device).float()
    hard = base.argmax(dim=1).ne(y)
    sample = torch.where(hard, torch.full_like(y, 3.0, dtype=torch.float32),
                         torch.ones_like(y, dtype=torch.float32))
    per = F.nll_loss(logp_mem, y, weight=weights, reduction="none")
    cls = (per * sample).sum() / sample.sum().clamp_min(1.0)
    if bool(hard.any()):
        wrong = base.argmax(dim=1)
        true_score = logp_mem.gather(1, y[:, None]).squeeze(1)
        wrong_score = logp_mem.gather(1, wrong[:, None]).squeeze(1)
        hard_loss = F.softplus(0.2 - (true_score - wrong_score))[hard].mean()
    else:
        hard_loss = logp_mem.new_zeros(())
    correct = ~hard
    protect = ((logp_mem - base).square().sum(dim=1)[correct].mean()
               if bool(correct.any()) else logp_mem.new_zeros(()))
    if delta is None or h is None:
        feat = logp_mem.new_zeros(())
    else:
        feat = (delta.square().sum(dim=1) / (h.square().sum(dim=1) + 1e-8)).mean()
    total = cls + 0.10 * hard_loss + 0.10 * protect + 0.01 * feat
    return total, {"mem_cls": cls, "hard": hard_loss, "protect": protect, "feat": feat,
                   "hard_count": int(hard.sum().item()), "sample_weight_mean": float(sample.mean().item())}
