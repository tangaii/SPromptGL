"""G1: Local Disease-Contrast Memory (LDCM)."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from memory_common import MemoryFusionBase, normalize_rows, stable_softmax


def _label_lists(labels: torch.Tensor, ids: Sequence[int]):
    y = labels.detach().cpu().long()
    a = [int(i) for i in ids if int(y[int(i)]) == 0]
    b = [int(i) for i in ids if int(y[int(i)]) == 1]
    return a, b


def _local_support(hn: torch.Tensor, labels: torch.Tensor, ids: List[int], a: int, b: int,
                   center: torch.Tensor, direction: torch.Tensor, k: int):
    candidates = [i for i in ids if i not in (a, b)]
    if not candidates:
        return [], 0.0, 0.0
    c = torch.as_tensor(candidates, dtype=torch.long, device=hn.device)
    dist = ((hn[c] - center[None, :]) ** 2).sum(1)
    take = min(k, len(candidates))
    nearest = c[torch.argsort(dist, stable=True)[:take]]
    scores = ((hn[nearest] - center[None, :]) * direction[None, :]).sum(1)
    yy = labels[nearest].detach()
    am = scores[yy.eq(0)]
    tm = scores[yy.eq(1)]
    if am.numel() < 2 or tm.numel() < 2:
        return nearest.detach().cpu().tolist(), 0.0, 0.0
    sep = float((am.mean() - tm.mean()).item())
    consistency = float(((am[:, None] > tm[None, :]).float().mean()).item())
    reliability = max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(sep) + 0.25 * (consistency - 0.5)))
    return nearest.detach().cpu().tolist(), sep, reliability


def build_contrast_memory(cam: torch.Tensor, labels: torch.Tensor, ids: Sequence[int],
                          sites: torch.Tensor, max_pairs: int = 64,
                          patient_max_pairs: int = 2, local_support_k: int = 12,
                          min_support_per_class: int = 2) -> dict:
    """Construct deterministic close cross-class pairs using only legal train IDs."""
    ids = sorted({int(i) for i in ids})
    device = cam.device
    hn = normalize_rows(cam.detach())
    y = labels.detach().long()
    asd, td = _label_lists(y, ids)
    if not asd or not td:
        return {"version": 0, "ids": [], "h_asd": torch.empty(0, 144), "h_td": torch.empty(0, 144),
                "center": torch.empty(0, 144), "direction": torch.empty(0, 144),
                "reliability": torch.empty(0), "similarity": torch.empty(0),
                "support_ids": [], "source_ids": ids, "kind": "LDCM"}
    aidx = torch.as_tensor(asd, dtype=torch.long, device=device)
    tidx = torch.as_tensor(td, dtype=torch.long, device=device)
    sim = hn[aidx] @ hn[tidx].T
    candidates = []
    for ri, a in enumerate(asd):
        order = torch.argsort(sim[ri], descending=True, stable=True)
        for oi in order[: min(8, len(td))].detach().cpu().tolist():
            b = td[int(oi)]
            candidates.append((float(sim[ri, int(oi)].item()), a, b))
    sim_t = sim.T
    for ri, b in enumerate(td):
        order = torch.argsort(sim_t[ri], descending=True, stable=True)
        for oi in order[: min(8, len(asd))].detach().cpu().tolist():
            a = asd[int(oi)]
            candidates.append((float(sim_t[ri, int(oi)].item()), a, b))
    candidates.sort(key=lambda z: (-z[0], z[1], z[2]))
    used = {i: 0 for i in ids}
    picked = []
    for simv, a, b in candidates:
        if (a, b) in {(x[1], x[2]) for x in picked}:
            continue
        if used[a] >= patient_max_pairs or used[b] >= patient_max_pairs:
            continue
        center = (cam[a].detach() + cam[b].detach()) / 2.0
        direction = F.normalize(cam[a].detach() - cam[b].detach(), dim=0, eps=1e-8)
        support, sep, rel = _local_support(hn, y, ids, a, b, normalize_rows(center[None, :])[0], direction, local_support_k)
        labels_support = [int(y[s]) for s in support]
        if labels_support.count(0) < min_support_per_class or labels_support.count(1) < min_support_per_class:
            continue
        picked.append((simv, a, b, center, direction, support, sep, rel))
        used[a] += 1; used[b] += 1
        if len(picked) >= max_pairs:
            break
    if not picked:
        return {"version": 0, "ids": [], "h_asd": torch.empty(0, 144), "h_td": torch.empty(0, 144),
                "center": torch.empty(0, 144), "direction": torch.empty(0, 144),
                "reliability": torch.empty(0), "similarity": torch.empty(0),
                "support_ids": [], "source_ids": ids, "kind": "LDCM"}
    return {
        "version": 0,
        "ids": [(int(a), int(b)) for _, a, b, *_ in picked],
        "h_asd": torch.stack([cam[a].detach().cpu() for _, a, b, *_ in picked]),
        "h_td": torch.stack([cam[b].detach().cpu() for _, a, b, *_ in picked]),
        "center": torch.stack([x[3].detach().cpu() for x in picked]),
        "direction": torch.stack([x[4].detach().cpu() for x in picked]),
        "reliability": torch.tensor([float(x[7]) for x in picked], dtype=torch.float32),
        "similarity": torch.tensor([float(x[0]) for x in picked], dtype=torch.float32),
        "local_separation": torch.tensor([float(x[6]) for x in picked], dtype=torch.float32),
        "support_ids": [list(map(int, x[5])) for x in picked],
        "source_ids": ids,
        "kind": "LDCM",
    }


class LDCM(MemoryFusionBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query_ids: Sequence[int], h: torch.Tensor, base_logp: torch.Tensor,
                memory: dict | None, sites: torch.Tensor | None = None,
                *, content_permuted: bool = False, pair_direction_permuted: bool = False):
        if memory is None or len(memory.get("ids", [])) == 0:
            out, d = self.empty_forward(h, base_logp)
            d.update({"memory_count": 0, "pair_reliability": h.new_zeros((h.shape[0],))})
            return out, d
        ids = [tuple(map(int, x)) for x in memory["ids"]]
        m = len(ids); device, dtype = h.device, h.dtype
        center = memory["center"].to(device=device, dtype=dtype)
        ha = memory["h_asd"].to(device=device, dtype=dtype)
        ht = memory["h_td"].to(device=device, dtype=dtype)
        direction = memory["direction"].to(device=device, dtype=dtype)
        rel = memory["reliability"].to(device=device, dtype=dtype).clamp_min(1e-3)
        if pair_direction_permuted and m > 1:
            direction = direction.roll(1, 0)
        q = self.query(h)
        key = self.key(center)
        sim = q @ key.T
        valid = torch.ones((h.shape[0], m), device=device, dtype=torch.bool)
        for row, pid in enumerate(query_ids):
            valid[row] = torch.tensor([int(pid) not in pair for pair in ids], device=device, dtype=torch.bool)
        k = min(8, m)
        top = sim.masked_fill(~valid, -1e9).topk(k, dim=1).indices
        row = torch.arange(h.shape[0], device=device)[:, None]
        tk = key[top]
        weights = stable_softmax((q[:, None, :] * tk).sum(-1) / self.temperature,
                                 torch.gather(valid, 1, top))
        dev = h[:, None, :] - center[top]
        # Query-specific local disease contrast: distance to ASD anchor minus TD anchor.
        ea = F.cosine_similarity(h[:, None, :], ha[top], dim=-1)
        et = F.cosine_similarity(h[:, None, :], ht[top], dim=-1)
        evidence = ea - et
        dirs = direction[top] * evidence[..., None] * rel[top][..., None]
        values = dirs
        if content_permuted and k > 1:
            values = values.roll(1, dims=1)
        v = self.v_proj(values.reshape(-1, self.cam_dim)).reshape(h.shape[0], k, self.memory_dim)
        read = (weights[..., None] * v).sum(1)
        simmax = (weights.new_zeros(h.shape[0]) if k == 0 else (q[:, None, :] * tk).sum(-1).masked_fill(~torch.gather(valid,1,top), -1e9).max(1).values)
        ent = -(weights * weights.clamp_min(1e-12).log()).sum(1) / math.log(max(k, 2))
        avail = torch.ones_like(simmax)
        out, diag = self.finish(h, base_logp, read, read, simmax, ent, simmax, ent, avail, avail,
                                retrieval=[{"pair_ids": [ids[int(j)] for j in top[i].detach().cpu().tolist()],
                                            "attention": weights[i].detach().cpu().tolist()} for i in range(h.shape[0])],
                                extra={"memory_count": m, "evidence_mean": evidence.detach().mean(1),
                                       "pair_reliability": rel[top].mean(1)})
        return out, diag

