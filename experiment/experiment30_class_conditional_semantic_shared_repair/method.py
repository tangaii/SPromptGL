"""Experiment 30: class-conditional connectivity semantics and shared repair.

All statistics used to select connections or construct auxiliary losses are
computed from the current outer training fold.  The class direction is ASD
(0) versus TD (1); no test label is accepted by this module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Design:
    fine: bool
    semantic: bool = True
    auxiliary: str = "NONE"
    text: str = "REAL"
    pair_loss: bool = True


DESIGNS = {
    "Z0": Design(False, False, "NONE"),
    "Z1": Design(False, False, "SHARED"),
    "C0": Design(False, True, "NONE"),
    "C1": Design(False, True, "SHARED"),
    "C2": Design(False, True, "SHARED", "MIXED_CLASS"),
    "C3": Design(False, True, "SHUFFLED"),
    "C4": Design(False, True, "DENSE"),
    "C5": Design(False, True, "PRIVATE"),
    "C6": Design(False, True, "SHARED", "REAL", False),
    "F0": Design(True, True, "NONE"),
    "F1": Design(True, True, "SHARED"),
    "F2": Design(True, True, "SHARED", "WRONG_ROI"),
    "F3": Design(True, True, "SHARED", "REVERSE_DIRECTION"),
    "F4": Design(True, True, "SHARED", "SCRAMBLED_TEXT"),
    "F5": Design(True, True, "SHUFFLED"),
    "F6": Design(True, True, "DENSE"),
    "F7": Design(True, True, "PRIVATE"),
    "F8": Design(True, True, "SHARED", "REAL", False),
}
CONFIGS = tuple(DESIGNS)
SEEDS = (0,)
FOLDS = tuple(range(1, 11))
ALL = ("PHENO", "ANAT", "FUNC")
K = 16
METRICS = ("acc", "auc", "sen_asd", "spe_td")
CONTRASTS = (("F1", "P0"), ("F1", "Z0"), ("F1", "F0"), ("F1", "F2"),
             ("F1", "F3"), ("F1", "F4"), ("F1", "F5"), ("F1", "F6"),
             ("F1", "F7"), ("F1", "F8"), ("C1", "P0"), ("C1", "Z0"),
             ("C1", "C0"), ("C1", "C2"), ("C1", "C3"), ("C1", "C4"),
             ("C1", "C5"), ("C1", "C6"))
INTERVENTIONS = ("REAL", "SEMANTIC_OFF", "AUX_OFF", "WRONG_ROI",
                 "REVERSE_DIRECTION", "SHUFFLE_AUX", "ALL_OFF")
RULES = dict(semantic_ce=.05, pair=.005, reconstruction=.005,
             orthogonality=.001, relative_aux_delta=.01,
             semantic_temperature=.2, top_k=16,
             semantic_residual_fraction=.05, total_residual_fraction=.10,
             new_weight_decay=0.0, init_seed=300000)


def asdict_designs():
    return {k: asdict(v) for k, v in DESIGNS.items()}


def tasks():
    gpu0 = ("Z0", "C0", "C2", "C4", "C6", "F0", "F2", "F4", "F6")
    gpu1 = ("Z1", "C1", "C3", "C5", "F1", "F3", "F5", "F7", "F8")
    assert set(gpu0) | set(gpu1) == set(CONFIGS)
    assert not set(gpu0) & set(gpu1)
    return [(c, 0, 0) for c in gpu0] + [(c, 0, 1) for c in gpu1]


def site_preserving_permutation(site, train, test, fold):
    site = np.asarray(site)
    perm = np.arange(len(site))
    rng = np.random.default_rng(30004 + int(fold))
    for part in (np.asarray(train), np.asarray(test)):
        for s in np.unique(site[part]):
            ids = part[site[part] == s]
            if len(ids) > 1:
                order = rng.permutation(ids)
                perm[order] = np.roll(order, 1)
    assert np.array_equal(site[perm], site)
    for part in (train, test):
        assert set(perm[np.asarray(part)]) == set(np.asarray(part))
    return perm


def training_statistics(raw_corr, train, labels_train, site):
    raw_corr = np.asarray(raw_corr, dtype=np.float64)
    tr = np.asarray(train, dtype=np.int64)
    y = np.asarray(labels_train, dtype=np.int64)
    assert raw_corr.shape == (871, 256)
    assert len(tr) == len(y) and set(y.tolist()) == {0, 1}
    data, st = raw_corr[tr], np.asarray(site)[tr]
    asd, td = data[y == 0], data[y == 1]
    td_mean = td.mean(0)
    td_std = np.maximum(td.std(0, ddof=0), .001)
    effects, weights = [], []
    for s in np.unique(st):
        a, t = data[(y == 0) & (st == s)], data[(y == 1) & (st == s)]
        if min(len(a), len(t)) < 2:
            continue
        pooled = np.sqrt(.5 * (a.var(0) + t.var(0)))
        neff = 2 * len(a) * len(t) / (len(a) + len(t))
        effect = (a.mean(0) - t.mean(0)) / np.maximum(pooled, .001)
        effects.append(np.clip(effect, -2., 2.) * neff / (neff + 20.))
        weights.append(neff)
    if effects:
        es = np.stack(effects); w = np.asarray(weights); w /= w.sum()
        direction = np.clip((w[:, None] * es).sum(0), -2., 2.)
        heterogeneity = np.sqrt((w[:, None] * (es - direction) ** 2).sum(0))
        agreement = (np.sign(es) == np.sign(direction)).mean(0)
    else:
        pooled = np.sqrt(.5 * (asd.var(0) + td.var(0)))
        direction = np.clip((asd.mean(0) - td.mean(0)) /
                            np.maximum(pooled, .001), -2., 2.)
        heterogeneity = np.zeros(256); agreement = np.zeros(256)
    reliability = np.clip(np.abs(direction), 0, 2) * agreement / (1 + heterogeneity)
    out = dict(mean=td_mean.astype(np.float32), std=td_std.astype(np.float32),
               direction=direction.astype(np.float32), reliability=reliability.astype(np.float32),
               agreement=agreement.astype(np.float32), heterogeneity=heterogeneity.astype(np.float32))
    out["report"] = dict(td_mean_range=[float(td_mean.min()), float(td_mean.max())],
                         td_std_range=[float(td_std.min()), float(td_std.max())],
                         direction_range=[float(direction.min()), float(direction.max())],
                         reliability_mean=float(reliability.mean()), eligible_sites=len(effects),
                         labels_scope="outer_train labels only")
    return out


def select_edges(raw_corr, stats, valid):
    valid=np.asarray(valid,dtype=np.int64); assert valid.shape==(252,)
    z=np.clip((np.asarray(raw_corr)-stats["mean"])/stats["std"],-3,3)
    rank=np.abs(z[:,valid])*stats["reliability"][valid][None,:]
    selected=np.argsort(-rank,axis=1,kind="stable")[:,:K].copy()
    strength=np.take_along_axis(rank,selected,axis=1); active=strength>1.e-8
    logits=np.log(strength+1.e-8); logits[~active]=-1.e4; logits-=logits.max(1,keepdims=True)
    weights=np.exp(logits)*active; weights/=np.maximum(weights.sum(1,keepdims=True),1.e-12)
    corr_ids=valid[selected]
    scalars=np.stack((np.take_along_axis(z,corr_ids,axis=1),
        np.broadcast_to(stats["direction"],z.shape)[np.arange(871)[:,None],corr_ids],
        np.broadcast_to(stats["reliability"],z.shape)[np.arange(871)[:,None],corr_ids]),axis=-1)
    return selected.astype(np.int64),corr_ids.astype(np.int64),weights.astype(np.float32),scalars.astype(np.float32),active

def relative_delta(raw, reference, fraction=.05):
    t = torch.tanh(raw)
    magnitude = t.norm(dim=-1, keepdim=True)
    ref = reference.detach().norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    return fraction * ref * t / (1. + magnitude)


class ClinicalSemanticRepair(nn.Module):
    def __init__(self, config, raw, train, test, labels_train, site, pheno_indices, bank, fold):
        super().__init__(); assert config in CONFIGS
        self.config, self.design, self.condition = config, DESIGNS[config], "REAL"
        self.last = {}
        self.register_buffer("train_ids", torch.as_tensor(train, dtype=torch.long))
        self.register_buffer("test_ids", torch.as_tensor(test, dtype=torch.long))
        self.register_buffer("train_site", torch.as_tensor(np.asarray(site)[train], dtype=torch.long))
        self.register_buffer("pheno_ids", torch.as_tensor(pheno_indices, dtype=torch.long))
        self.register_buffer("aux_perm", torch.as_tensor(site_preserving_permutation(site, train, test, fold), dtype=torch.long))
        stats = training_statistics(raw[:, 64:], train, labels_train, site)
        selected, corr_ids, weights, scalars, active = select_edges(raw[:, 64:], stats, bank["valid_indices"])
        for name, value in (("selected", selected), ("selected_corr", corr_ids), ("edge_weights", weights), ("edge_scalars", scalars), ("edge_active", active)):
            self.register_buffer(name, torch.as_tensor(value))
        self.register_buffer("direction_index", torch.as_tensor((stats["direction"][bank["valid_indices"]] < 0).astype(np.int64)))
        self.statistics_report = stats["report"]
        for name in ("original", "fine_asd", "fine_td", "numeric_original", "numeric_fine"):
            self.register_buffer(name + "_text", torch.as_tensor(bank[name], dtype=torch.float32))
        for name in ("roi_permutation", "scramble_asd", "scramble_td"):
            self.register_buffer(name, torch.as_tensor(bank[name], dtype=torch.long))
        self.fc_encoder = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU())
        self.fc_query = nn.Linear(64, 64, bias=False)
        self.scalar_query = nn.Linear(3, 64, bias=False)
        self.semantic_head = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Tanh(), nn.Linear(64, 36, bias=False))
        nn.init.zeros_(self.semantic_head[-1].weight)
        self.fc_shared = nn.Linear(64, 16, bias=False)
        self.aux_encoder = nn.ModuleDict(); self.aux_shared = nn.ModuleDict(); self.aux_private = nn.ModuleDict()
        self.aux_reconstruct = nn.ModuleDict(); self.aux_router = nn.ModuleDict(); self.aux_value = nn.ModuleDict()
        for name, dim in zip(ALL, (28, 6, 10)):
            self.aux_encoder[name] = nn.Sequential(nn.Linear(dim, 32, bias=False), nn.LayerNorm(32), nn.GELU())
            self.aux_shared[name] = nn.Linear(32, 16, bias=False)
            self.aux_private[name] = nn.Linear(32, 16, bias=False)
            self.aux_reconstruct[name] = nn.Linear(32, dim)
            self.aux_router[name] = nn.Sequential(nn.Linear(65, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.constant_(self.aux_router[name][-1].bias, -2.)
            self.aux_value[name] = nn.Linear(16, 16, bias=False)
        self.auxiliary_head = nn.Sequential(nn.Linear(16, 32, bias=False), nn.Tanh(), nn.Linear(32, 36, bias=False))
        nn.init.zeros_(self.auxiliary_head[-1].weight)

    def _text(self):
        mode = self.condition if self.condition in ("WRONG_ROI", "REVERSE_DIRECTION", "SCRAMBLED_TEXT", "MIXED_CLASS") else self.design.text
        original = self.numeric_original_text if mode == "NUMERIC_TEXT" else self.original_text
        if mode == "MIXED_CLASS":
            asd = self.original_text[[0, 2, 4, 6]].mean(0); td = self.original_text[[1, 3, 5, 7]].mean(0)
        else:
            asd = original[4:].mean(0); td = original[:4].mean(0)
        coarse = F.normalize(torch.stack((asd, td)), dim=-1, eps=1.e-8)
        if self.condition == "GLOBAL_CLASS_SWAP": coarse = coarse.flip(0)
        if not self.design.fine:
            return coarse, coarse[None, None, :, :].expand(871, K, 2, 64)
        fine = self.numeric_fine_text if mode == "NUMERIC_TEXT" else self.fine_asd_text
        td = self.numeric_fine_text[:, 0] if mode == "NUMERIC_TEXT" else self.fine_td_text
        if mode == "WRONG_ROI": fine, td = fine[self.roi_permutation], td[self.roi_permutation]
        elif mode == "SCRAMBLED_TEXT": fine, td = fine[self.scramble_asd], td[self.scramble_td]
        if mode == "REVERSE_DIRECTION": fine = fine.flip(1)
        asd_keys = fine[self.selected, self.direction_index[self.selected]]
        td_keys = td[self.selected]
        return coarse, torch.stack((asd_keys, td_keys), dim=2)

    def _semantic(self, h):
        coarse, keys = self._text(); q = F.normalize(self.fc_query(h), dim=-1, eps=1.e-8)
        local_q = F.normalize(q[:, None, :] + self.scalar_query(self.edge_scalars), dim=-1, eps=1.e-8)
        similarity = (local_q[:, :, None, :] * keys).sum(-1) / .2
        local_logits = (self.edge_weights[:, :, None] * similarity).sum(1)
        global_logits = q @ coarse.T / .2
        class_logits = .5 * local_logits + .5 * global_logits
        p = similarity.softmax(-1)[..., 0:1]
        signed = (p - .5) * (keys[:, :, 0, :] - keys[:, :, 1, :])
        semantic_value = (self.edge_weights[:, :, None] * signed).sum(1)
        return class_logits, self.semantic_head(semantic_value), similarity, semantic_value

    def _auxiliary(self, h, raw, class_logits, enable, pair_enable):
        n = raw.shape[0]; zero = h.new_zeros((n, 36))
        if not enable: return zero, h.new_zeros(()), h.new_zeros(()), h.new_zeros(()), {}
        mode = self.design.auxiliary
        xraw = raw[self.aux_perm] if (mode == "SHUFFLED" or self.condition == "SHUFFLE_AUX") else raw
        xs = {"PHENO": xraw[:, self.pheno_ids], "ANAT": xraw[:, 48:54], "FUNC": xraw[:, 54:64]}
        fc_shared = self.fc_shared(h); uncertain = 1. - class_logits.softmax(-1).diff(dim=-1).abs().detach()
        if not self.design.semantic or self.condition in ("SEMANTIC_OFF", "ALL_OFF"): uncertain = torch.ones_like(uncertain)
        logits=[]; values=[]; pairs=[]; recons=[]; orths=[]; details={}; tr=self.train_ids
        for name in ALL:
            xm=xs[name]; e=self.aux_encoder[name](xm); shared=self.aux_shared[name](e); private=self.aux_private[name](e)
            source=private if (mode == "PRIVATE" or self.condition == "PRIVATE_ONLY") else shared
            controls=torch.cat((fc_shared, source, fc_shared*source, (fc_shared-source).abs(), uncertain), dim=-1)
            agreement=(F.normalize(fc_shared,dim=-1,eps=1.e-8)*F.normalize(source,dim=-1,eps=1.e-8)).sum(-1,keepdim=True)
            score=self.aux_router[name](controls)+torch.log(((agreement+1.)/2.).clamp_min(.05))
            logits.append(score); values.append(self.aux_value[name](source))
            reconstructed=self.aux_reconstruct[name](torch.cat((shared,private),dim=-1)); recons.append(F.mse_loss(reconstructed[tr],xm[tr]))
            cs=shared[tr]-shared[tr].mean(0,keepdim=True); cp=private[tr]-private[tr].mean(0,keepdim=True)
            orths.append((cs.T@cp/max(1,len(tr)-1)).square().mean())
            if pair_enable:
                fa=F.normalize(fc_shared[tr].detach(),dim=-1,eps=1.e-8); sh=F.normalize(shared[tr],dim=-1,eps=1.e-8)
                scores=fa@sh.T/.2; same=self.train_site[:,None]==self.train_site[None,:]; valid=same.sum(1)>=3
                if valid.any(): pairs.append(F.cross_entropy(scores.masked_fill(~same,-1.e4)[valid], torch.arange(len(tr),device=scores.device)[valid]))
            details[name] = dict(shared=shared, private=private, router=score)
        routing=torch.softmax(torch.cat(logits+[h.new_zeros((n,1))],dim=1),dim=-1)
        if mode == "DENSE" or self.condition == "DENSE_ONLY": routing=routing*0+h.new_tensor([1/3,1/3,1/3,0])
        repaired=sum(routing[:,j:j+1]*values[j] for j in range(3)); details["routing"]=routing
        pair=torch.stack(pairs).mean() if pairs else h.new_zeros(())
        return self.auxiliary_head(torch.cat([repaired],dim=-1)), pair, torch.stack(recons).mean(), torch.stack(orths).mean(), details

    def forward(self, cam, raw):
        assert cam.shape == (871,4,36) and raw.shape == (871,320)
        h=self.fc_encoder(raw[:,64:]); sem_on=self.design.semantic and self.condition not in ("SEMANTIC_OFF","ALL_OFF")
        aux_on=self.design.auxiliary != "NONE" and self.condition not in ("AUX_OFF","ALL_OFF")
        z=h.new_zeros((871,2)); sem_raw=h.new_zeros((871,36)); sim=None; sval=h.new_zeros((871,64))
        if sem_on: z,sem_raw,sim,sval=self._semantic(h)
        aux_raw,pair,recon,orth,details=self._auxiliary(h,raw,z,aux_on,self.design.pair_loss and self.condition!="PAIR_OFF")
        target=cam[:,3,:]; sd=relative_delta(sem_raw,target,.05) if sem_on else torch.zeros_like(target)
        ad=relative_delta(aux_raw,target,.05) if aux_on else torch.zeros_like(target)
        out=torch.cat((cam[:,:3,:],(target+sd+ad)[:,None,:]),dim=1)
        self.last=dict(h=h,class_logits=z,semantic_on=sem_on,aux_enabled=aux_on,sem_delta=sd,aux_delta=ad,delta=sd+ad,target=target,
                       local_similarity=sim,semantic_value=sval,pair_loss=pair,reconstruction_loss=recon,orthogonality_loss=orth,
                       auxiliary_details=details,semantic_raw=sem_raw,aux_raw=aux_raw)
        assert torch.isfinite(out).all(); return out

    def loss_terms(self, labels):
        z=self.last; tr=self.train_ids; zero=z["h"].sum()*0.; out=dict(global_loss=zero,pair_loss=zero,reconstruction_loss=zero,orthogonality_loss=zero,relative_aux_delta=zero)
        if z["semantic_on"]: out["global_loss"]=F.cross_entropy(z["class_logits"][tr],labels[tr])
        if z["aux_enabled"]:
            out["pair_loss"]=z["pair_loss"];out["reconstruction_loss"]=z["reconstruction_loss"];out["orthogonality_loss"]=z["orthogonality_loss"]
            den=z["target"].detach().norm(dim=-1).clamp_min(1.e-8);out["relative_aux_delta"]=(z["aux_delta"].norm(dim=-1)[tr]/den[tr]).mean()
        return out

    def extra_loss(self, labels):
        t=self.loss_terms(labels)
        return .05*t["global_loss"]+.005*t["pair_loss"]+.005*t["reconstruction_loss"]+.001*t["orthogonality_loss"]+.01*t["relative_aux_delta"]


def self_test():
    assert len(CONFIGS)==18 and len(tasks())==18
    assert set(c for c,_,_ in tasks())==set(CONFIGS)
    print("E30_METHOD_CONTRACT_PASS")


if __name__ == "__main__": self_test()
