"""G2: Discriminative Evidence Memory (DDEM)."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from memory_common import MemoryFusionBase, normalize_rows, output_layer_logp, stable_softmax


def _balanced_select(ids, labels, margins, sites, max_per_class):
    """Deterministically select correct patients with modest site coverage."""
    selected = []
    y = labels.detach().cpu().long()
    st = sites.detach().cpu().long() if sites is not None else torch.zeros_like(y)
    for cls in (0, 1):
        pool = [int(i) for i in ids if int(y[int(i)]) == cls]
        pool.sort(key=lambda i: (-float(margins[i]), int(st[i]), i))
        # First pass covers sites in sorted order, second pass fills by margin.
        by_site = {}
        for i in pool:
            by_site.setdefault(int(st[i]), []).append(i)
        picked = []
        while by_site and len(picked) < max_per_class:
            for s in sorted(list(by_site)):
                if not by_site[s]:
                    by_site.pop(s, None); continue
                picked.append(by_site[s].pop(0))
                if not by_site[s]: by_site.pop(s, None)
                if len(picked) >= max_per_class: break
        if len(picked) < max_per_class:
            for i in pool:
                if i not in picked:
                    picked.append(i)
                    if len(picked) >= max_per_class: break
        selected.extend(picked[:max_per_class])
    return sorted(selected)


def build_evidence_memory(cam: torch.Tensor, labels: torch.Tensor, base_logp: torch.Tensor,
                          ids: Sequence[int], sites: torch.Tensor, output_layer: nn.Module,
                          max_per_class: int = 48, evidence_dims: int = 12,
                          alphas=(0.5, 1.0), version: int = 0) -> dict:
    """Extract intervention-supported CAM dimensions from correct legal train patients."""
    device = cam.device
    ids = sorted({int(i) for i in ids})
    y = labels.detach().long()
    pred = base_logp.detach().argmax(1)
    correct = [i for i in ids if int(pred[i]) == int(y[i])]
    if not correct:
        return {"version": version, "ids": [], "label": torch.empty(0, dtype=torch.long),
                "reference": torch.empty(0, 144), "values": torch.empty(0, 144),
                "mask": torch.empty(0, 144), "reliability": torch.empty(0),
                "joint_drop": torch.empty(0), "source_ids": ids, "kind": "DDEM"}
    mu = cam[torch.as_tensor(ids, device=device)].detach().mean(0)
    all_margin = torch.empty(cam.shape[0], device=device)
    for i in correct:
        yi = int(y[i]); all_margin[i] = base_logp[i, yi] - base_logp[i, 1 - yi]
    selected = _balanced_select(correct, y, all_margin.detach().cpu().tolist(), sites, max_per_class)
    if not selected:
        selected = correct[: 2 * max_per_class]
    h = cam[torch.as_tensor(selected, device=device)].detach()
    yy = y[torch.as_tensor(selected, device=device)]
    lp = base_logp[torch.as_tensor(selected, device=device)].detach()
    true_margin = lp.gather(1, yy[:, None]).squeeze(1) - lp.gather(1, (1 - yy)[:, None]).squeeze(1)
    n = len(selected); d = cam.shape[1]
    # Vectorized intervention: [N, 2 alpha, D, D] is avoided; construct N*2*D rows.
    rows = []
    for ai in alphas:
        for j in range(d):
            hj = h.clone(); hj[:, j] = (1.0 - float(ai)) * hj[:, j] + float(ai) * mu[j]
            rows.append(hj)
    batch = torch.cat(rows, dim=0)
    with torch.no_grad():
        ilp, _ = output_layer_logp(output_layer, batch)
    blocks = ilp.reshape(len(alphas), d, n, 2).permute(2, 0, 1, 3)  # [N,A,D,2]
    importance = []
    for ai_index, _ai in enumerate(alphas):
        mlp = blocks[:, ai_index]
        m = mlp.gather(2, yy[:, None, None].expand(-1, d, 1)).squeeze(-1) - mlp.gather(2, (1-yy)[:, None, None].expand(-1,d,1)).squeeze(-1)
        importance.append(true_margin[:, None] - m)
    imp = torch.stack(importance, dim=1)  # [N,A,D]
    positive = (imp > 0).all(1)
    score = torch.where(positive, imp.min(1).values, torch.zeros_like(imp[:, 0]))
    masks = torch.zeros(n, d, device=device)
    values = torch.zeros_like(h)
    joint_drop = torch.zeros(n, device=device)
    kept = []
    for i, pid in enumerate(selected):
        order = torch.argsort(score[i], descending=True, stable=True)
        good = order[score[i, order] > 0][:evidence_dims]
        if good.numel() == 0:
            continue
        masks[i, good] = 1.0
        values[i] = h[i]
        hm = h[i].clone(); hm[good] = mu[good]
        lm, _ = output_layer_logp(output_layer, torch.stack([h[i], hm]))
        joint = (lm[0, int(yy[i])] - lm[0, 1-int(yy[i])]) - (lm[1, int(yy[i])] - lm[1, 1-int(yy[i])])
        joint_drop[i] = joint
        if float(joint) > 0:
            kept.append(i)
    if not kept:
        return {"version": version, "ids": [], "label": torch.empty(0, dtype=torch.long),
                "reference": torch.empty(0, 144), "values": torch.empty(0, 144),
                "mask": torch.empty(0, 144), "reliability": torch.empty(0),
                "joint_drop": torch.empty(0), "source_ids": ids, "kind": "DDEM",
                "candidate_count": len(selected), "rejected_count": len(selected)}
    kk = torch.as_tensor(kept, device=device, dtype=torch.long)
    rel = torch.sigmoid(joint_drop[kk]) * (masks[kk].sum(1) / float(max(evidence_dims, 1)))
    return {
        "version": version, "ids": [selected[i] for i in kept], "label": yy[kk].detach().cpu(),
        "reference": h[kk].detach().cpu(), "values": values[kk].detach().cpu(),
        "mask": masks[kk].detach().cpu(), "reliability": rel.detach().cpu(),
        "joint_drop": joint_drop[kk].detach().cpu(), "source_ids": ids,
        "kind": "DDEM", "candidate_count": len(selected),
        "rejected_count": len(selected) - len(kept), "neutral_mean": mu.detach().cpu(),
        "importance_min": imp.min(1).values[kk].detach().cpu(),
    }


class DDEM(MemoryFusionBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read(self, query_ids, h, memory, *, content_permuted=False, mask_permuted=False):
        n = h.shape[0]; device, dtype = h.device, h.dtype
        if memory is None or len(memory.get("ids", [])) == 0:
            z = h.new_zeros((n, self.memory_dim)); zero = h.new_zeros(n)
            return z, zero, zero, zero, [[] for _ in range(n)]
        ref = memory["reference"].to(device=device, dtype=dtype)
        vals = memory["values"].to(device=device, dtype=dtype)
        masks = memory["mask"].to(device=device, dtype=dtype)
        rel = memory["reliability"].to(device=device, dtype=dtype).clamp_min(1e-4)
        mu = memory.get("neutral_mean", h.new_zeros(self.cam_dim)).to(device=device, dtype=dtype)
        valid = torch.ones((n, ref.shape[0]), device=device, dtype=torch.bool)
        for r, pid in enumerate(query_ids):
            valid[r] = torch.tensor([int(pid) != int(x) for x in memory["ids"]], device=device, dtype=torch.bool)
        # Evidence-mask matching is the retrieval score, so ordinary full-vector
        # similarity cannot bypass the stored evidence mask.
        qdev = h[:, None, :] - mu[None, None, :]
        kdev = ref[None, :, :] - mu[None, None, :]
        num = (qdev * kdev * masks[None, :, :]).sum(-1)
        den = ((qdev.square() * masks[None, :, :]).sum(-1).sqrt() * (kdev.square() * masks[None, :, :]).sum(-1).sqrt()).clamp_min(1e-8)
        sim = num / den
        k = min(8, ref.shape[0]); top = sim.masked_fill(~valid, -1e9).topk(k, dim=1).indices
        v = (vals - mu[None, :]) * masks * rel[:, None]
        if mask_permuted and v.shape[0] > 1:
            masks_for_value = masks.roll(1, 0); v = (vals - mu[None, :]) * masks_for_value * rel[:, None]
        if content_permuted and v.shape[0] > 1:
            v = v.roll(1, 0)
        v64 = self.v_proj(v).to(dtype)
        weights = stable_softmax(sim.gather(1, top) / self.temperature, torch.gather(valid, 1, top))
        read = (weights[..., None] * v64[top]).sum(1)
        sm = sim.gather(1, top).max(1).values
        ent = -(weights * weights.clamp_min(1e-12).log()).sum(1) / math.log(max(k, 2))
        retr = [{"patient_ids": [memory["ids"][int(j)] for j in top[i].detach().cpu().tolist()], "attention": weights[i].detach().cpu().tolist()} for i in range(n)]
        return read, sm, ent, torch.ones_like(sm), retr

    def forward(self, query_ids, h, base_logp, memory, sites=None, *, content_permuted=False, evidence_mask_permuted=False):
        def subset(label):
            if memory is None or len(memory.get("ids", [])) == 0:
                return None
            keep = memory["label"].eq(int(label))
            if not bool(keep.any()):
                return None
            return {**memory,
                    "ids": [pid for pid, kk in zip(memory["ids"], keep.tolist()) if kk],
                    "label": memory["label"][keep],
                    "reference": memory["reference"][keep],
                    "values": memory["values"][keep],
                    "mask": memory["mask"][keep],
                    "reliability": memory["reliability"][keep],
                    "joint_drop": memory["joint_drop"][keep]}
        ma, mb = subset(0), subset(1)
        ra, sa, ea, aa, pa = self._read(query_ids, h, ma, content_permuted=content_permuted, mask_permuted=evidence_mask_permuted)
        rb, sb, eb, ab, pb = self._read(query_ids, h, mb, content_permuted=content_permuted, mask_permuted=evidence_mask_permuted)
        out, d = self.finish(h, base_logp, ra, rb, sa, ea, sb, eb, aa, ab,
                             retrieval=[{"asd":pa[i], "td":pb[i]} for i in range(h.shape[0])],
                             extra={"memory_count": 0 if memory is None else len(memory.get("ids", [])), "evidence_mask_permuted": evidence_mask_permuted})
        return out, d

