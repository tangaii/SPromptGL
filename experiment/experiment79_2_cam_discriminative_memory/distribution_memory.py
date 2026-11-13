"""G3: Disease Distribution Memory (DDM)."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from memory_common import MemoryFusionBase, stable_softmax


def _farthest_init(x: torch.Tensor, k: int):
    n = x.shape[0]
    if n == 0:
        return []
    # Fully deterministic: smallest row first, then farthest point with index tie break.
    chosen = [0]
    while len(chosen) < min(k, n):
        dist = ((x[:, None, :] - x[torch.as_tensor(chosen, device=x.device)][None, :, :]) ** 2).sum(-1).min(1).values
        dist[torch.as_tensor(chosen, device=x.device)] = -1.0
        maxv = float(dist.max())
        cand = [i for i, v in enumerate(dist.detach().cpu().tolist()) if abs(v - maxv) <= 1e-12]
        chosen.append(min(cand))
    return chosen


def _kmeans(x: torch.Tensor, k: int, iters: int):
    init = _farthest_init(x, k)
    centers = x[torch.as_tensor(init, device=x.device)].clone()
    if centers.shape[0] < k:
        return centers, torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    assign = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = dist.argmin(1)
        new = []
        for j in range(k):
            idx = assign.eq(j).nonzero(as_tuple=False).flatten()
            new.append(x[idx].mean(0) if idx.numel() else centers[j])
        centers = torch.stack(new)
    return centers, assign


def build_distribution_memory(cam: torch.Tensor, labels: torch.Tensor, ids: Sequence[int],
                              components_per_class: int = 3, iterations: int = 5,
                              shrinkage: float = 0.5, variance_floor: float = 1e-5,
                              version: int = 0) -> dict:
    ids = sorted({int(i) for i in ids})
    device = cam.device
    y = labels.detach().long()
    rows = []
    all_var = []
    for cls in (0, 1):
        cids = [i for i in ids if int(y[i]) == cls]
        if not cids:
            continue
        x = cam[torch.as_tensor(cids, device=device)].detach()
        pooled = x.var(0, unbiased=False).clamp_min(variance_floor)
        all_var.append(pooled)
        k = min(components_per_class, x.shape[0])
        centers, assign = _kmeans(x, k, iterations)
        for j in range(k):
            idx = assign.eq(j).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            xx = x[idx]
            emp = xx.var(0, unbiased=False) if idx.numel() > 1 else pooled
            var = ((1.0 - shrinkage) * emp + shrinkage * pooled).clamp_min(variance_floor)
            members = [cids[int(v)] for v in idx.detach().cpu().tolist()]
            rows.append({"label": cls, "mean": xx.mean(0).detach().cpu(), "var": var.detach().cpu(),
                         "count": int(idx.numel()), "prior": float(idx.numel()/len(cids)),
                         "members": members, "sum": xx.sum(0).detach().cpu(),
                         "sumsq": xx.square().sum(0).detach().cpu(),
                         "member_features": xx.detach().cpu()})
    if not rows:
        return {"version": version, "components": [], "pooled_var": torch.ones(144), "source_ids": ids, "kind": "DDM"}
    pooled_var = torch.stack(all_var).mean(0).clamp_min(variance_floor) if all_var else torch.ones(cam.shape[1], device=device)
    return {"version": version, "components": rows, "pooled_var": pooled_var.detach().cpu(),
            "source_ids": ids, "kind": "DDM", "components_per_class": components_per_class}


class DDM(MemoryFusionBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read(self, query_ids, h, memory, label, *, variance_off=False, content_permuted=False):
        n = h.shape[0]; device, dtype = h.device, h.dtype
        if memory is None:
            z = h.new_zeros((n, self.memory_dim)); zero = h.new_zeros(n)
            return z, zero, zero, zero, [{} for _ in range(n)]
        comps = [c for c in memory.get("components", []) if int(c["label"]) == int(label)]
        if not comps:
            z = h.new_zeros((n, self.memory_dim)); zero = h.new_zeros(n)
            return z, zero, zero, zero, [{} for _ in range(n)]
        pooled = memory.get("pooled_var", torch.ones(self.cam_dim)).to(device=device, dtype=dtype).clamp_min(1e-5)
        mus=[]; vars_=[]; pri=[]; ids=[]
        for c in comps:
            mu = c["mean"].to(device=device, dtype=dtype)
            var = c["var"].to(device=device, dtype=dtype).clamp_min(1e-5)
            count = int(c["count"])
            # Leave-one-out statistics for a training query that belongs to this
            # component; test queries do not match any member.
            members = set(map(int, c.get("members", [])))
            sums = c["sum"].to(device=device, dtype=dtype)
            sumsq = c["sumsq"].to(device=device, dtype=dtype)
            for pid in query_ids:
                if int(pid) in members and count > 1:
                    # This branch is handled per query below; the default values
                    # are still used for test rows.
                    pass
            mus.append(mu); vars_.append(var); pri.append(float(c.get("prior", 1.0))); ids.append(c.get("members", []))
        mu = torch.stack(mus); var = torch.stack(vars_); pri_t = torch.tensor(pri, device=device, dtype=dtype)
        score = []
        for r, pid in enumerate(query_ids):
            row=[]
            for j,c in enumerate(comps):
                m=mu[j]; v=var[j]; count=int(c["count"]); members=set(map(int,c.get("members",[])))
                if int(pid) in members and count>1:
                    pos=c["members"].index(int(pid)); x=c.get("member_features")
                    if x is not None:
                        xi=x[pos].to(device=device,dtype=dtype); m=(c["sum"].to(device=device,dtype=dtype)-xi)/(count-1)
                        emp=((c["sumsq"].to(device=device,dtype=dtype)-xi.square())/(count-1)-m.square()).clamp_min(1e-5)
                        v=((1.0-0.5)*emp+0.5*pooled).clamp_min(1e-5)
                vv=pooled if variance_off else v
                z=((h[r]-m).square()/vv + vv.log()).mean()
                row.append(torch.log(pri_t[j].clamp_min(1e-8))-0.5*z)
            score.append(torch.stack(row))
        score=torch.stack(score)
        weights=torch.softmax(score,dim=1)
        values=mu - mu.mean(0,keepdim=True)
        if content_permuted and values.shape[0]>1: values=values.roll(1,0)
        v64=self.v_proj(values)
        read=weights@v64
        sm=score.max(1).values
        ent=-(weights*weights.clamp_min(1e-12).log()).sum(1)/math.log(max(len(comps),2))
        retr=[{"component_members": [ids[int(j)] for j in range(len(comps))], "attention": weights[i].detach().cpu().tolist()} for i in range(n)]
        return read,sm,ent,torch.ones_like(sm),retr

    def forward(self, query_ids, h, base_logp, memory, sites=None, *, variance_off=False, content_permuted=False):
        ra,sa,ea,aa,pa=self._read(query_ids,h,memory,0,variance_off=variance_off,content_permuted=content_permuted)
        rb,sb,eb,ab,pb=self._read(query_ids,h,memory,1,variance_off=variance_off,content_permuted=content_permuted)
        out,d=self.finish(h,base_logp,ra,rb,sa,ea,sb,eb,aa,ab,
                          retrieval=[{"asd":pa[i],"td":pb[i]} for i in range(h.shape[0])],
                          extra={"memory_count":0 if memory is None else len(memory.get("components",[])),"variance_off":variance_off})
        return out,d

