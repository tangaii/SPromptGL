"""Experiment23 gain-conditioned patient-relationship bridge.

Original four GFRs and original PLN are untouched. This acts on the 320-D
GFR output immediately before PLN; only 256-D CORR slice is updated. All
neighbor keys belong to the outer train set. The gate is precomputed by
train-only OOF teachers and frozen; test-label access is forbidden.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Condition:
    name: str
    gate: str = 'gain'
    operator: str = 'relation'
    pheno_shuffled: bool = False

CONDITIONS: Dict[str, Condition] = {
    'G0': Condition('Original P0', 'off'),
    'G1': Condition('Uniform relation', 'uniform'),
    'G2': Condition('Mean-matched relation', 'constant'),
    'G3': Condition('Confidence-gated relation', 'confidence'),
    'G4': Condition('Gain-gated relation', 'gain'),
    'G5': Condition('Shuffled gain relation', 'shuffled'),
    'G6': Condition('Gain-gated feature control', 'gain', 'feature'),
    'G7': Condition('Gain relation with shuffled PHENO adjacency', 'gain', 'relation', True),
}


def _normalize_fold(x: torch.Tensor, training: torch.Tensor):
    mean = x[training].mean(0, keepdim=True)
    std = x[training].std(0, unbiased=False, keepdim=True).clamp_min(.05)
    return ((x - mean) / std).clamp(-8.,8.)


def _neighbors(x: torch.Tensor, tr: torch.Tensor, k: int, temperature=.3):
    """Dense N x T row-stochastic similarities. Only training keys; no labels."""
    query = F.normalize(x, dim=-1, eps=1e-8)
    key = query[tr]
    sim = query @ key.T / temperature
    same = torch.arange(len(x),device=x.device)[:,None] == tr[None,:]
    sim = sim.masked_fill(same, -1e4)
    use_k = min(k, len(tr)-1)
    idx=sim.topk(use_k, dim=1).indices
    support=torch.zeros_like(sim,dtype=torch.bool).scatter(1,idx,True)
    return sim, support


def _rowsoftmax(logits, mask):
    return torch.softmax(logits.masked_fill(~mask,-1e4),dim=-1)


class GainRelationalBridge(nn.Module):
    """A gain-gated barycentric transport between patient relation kernels.

    Output is intentionally exactly base at initialization (alpha=0).
    Control groups have identical registered parameters and forward capacity.
    No new clinical/semantic labels are invented.
    """
    def __init__(self, config, raw_modalities, train_indices, pheno28, evidence,
                 top_k=12, residual_limit=.10):
        super().__init__()
        assert config in CONDITIONS and config!='G0'
        self.config=config
        self.design=CONDITIONS[config]
        assert len(raw_modalities)==4
        ph, anat, func, corr=raw_modalities
        assert ph.shape == (871,48) and corr.shape==(871,256)
        assert anat.shape==(871,6) and func.shape==(871,10)
        assert top_k>1 and 0<residual_limit<=.1
        tr = torch.as_tensor(train_indices,dtype=torch.long,device=corr.device)
        ix = torch.as_tensor(pheno28,dtype=torch.long,device=ph.device)
        self.register_buffer('train_indices',tr)
        self.register_buffer('pheno_indices',ix)
        self.register_buffer('test_indices',torch.as_tensor(evidence['outer_test'],dtype=torch.long,device=corr.device))
        self.limit=float(residual_limit)
        allph=ph.index_select(1,ix).float()
        self.register_buffer('patient_pheno',allph.detach().clone())
        self.register_buffer('original_pheno',allph.detach().clone())
        # Input from training rows only: no test feature statistics or labels.
        with torch.no_grad():
            sf, supp_f = _neighbors(_normalize_fold(corr.float(),tr),tr,top_k)
            perm=torch.as_tensor(evidence['pheno_permutation'],dtype=torch.long,device=ph.device)
            shph=allph[perm] if self.design.pheno_shuffled else allph
            sp, supp_p = _neighbors(_normalize_fold(shph,tr),tr,top_k)
            support=supp_f | supp_p
            af=_rowsoftmax(sf,support)
            ap=_rowsoftmax(sp,support)
            assert torch.isfinite(af).all() and torch.isfinite(ap).all()
            assert torch.allclose(af.sum(1),torch.ones(len(af),device=af.device),atol=1e-5)
            assert torch.allclose(ap.sum(1),torch.ones(len(ap),device=ap.device),atol=1e-5)
            # A_bary = .5*A_F + .5*A_P ; exact no fusion when gate=0.
            self.register_buffer('relation_difference',(ap-af).detach())
            self.register_buffer('af',af.detach())
            self.register_buffer('ap',ap.detach())
            self.register_buffer('base_graph_intersection',((supp_f&supp_p).sum(1).float()/top_k).detach())
        gates={'uniform':torch.ones(871),'constant':torch.as_tensor(evidence['constant_gate']),
               'confidence':torch.as_tensor(evidence['confidence_gate']),
               'gain':torch.as_tensor(evidence['gain_gate']),
               'shuffled':torch.as_tensor(evidence['shuffled_gate'])}
        self.register_buffer('chosen_gate',gates[self.design.gate].float().to(ph.device))
        self.register_buffer('actual_gate',self.chosen_gate.detach().clone())
        # Identical parameters for G1-G7, regardless of intervention.
        self.pheno_encoder=nn.Sequential(nn.Linear(28,32),nn.GELU(),nn.LayerNorm(32))
        self.corr_encoder=nn.Sequential(nn.Linear(256,32),nn.GELU(),nn.LayerNorm(32))
        self.decoder=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,256))
        self.alpha=nn.Parameter(torch.zeros(()))
        self.condition='REAL'
        self.last={}

    def forward(self, base):
        assert base.shape==(871,320),tuple(base.shape)
        ph=self.patient_pheno
        gate=self.chosen_gate
        if self.condition=='GATE_OFF': gate=torch.zeros_like(gate)
        if self.condition=='GATE_CONSTANT': gate=gate.new_full(gate.shape,gate[self.train_indices].mean())
        value=self.pheno_encoder(ph)
        fc_value=self.af @ value[self.train_indices]
        relation_difference=self.relation_difference @ value[self.train_indices]
        if self.design.operator=='feature':
            # Feature-level control: does not use AP to select neighbors.
            # Match TRAIN-set signal norm with relation arm, so an accidental
            # amplitude advantage cannot explain comparison G4 vs G6.
            feature_difference=value-fc_value
            numerator=relation_difference[self.train_indices].norm(dim=-1).mean()
            denominator=feature_difference[self.train_indices].norm(dim=-1).mean().clamp_min(1e-8)
            matched_scale=(numerator/denominator).detach()
            difference=feature_difference*matched_scale
        else:
            difference=relation_difference
        if self.condition=='RELATION_OFF':difference=torch.zeros_like(difference)
        # Patient signal is acquired from learned relation delta, NOT prompt tokens.
        context=self.corr_encoder(base[:,64:320])
        latent=torch.cat([difference,context],dim=-1)
        # CRITICAL: subtract zero-evidence decoder response, preventing a
        # context/bias-only path from bypassing actual cross-modal relations.
        reference=torch.cat([torch.zeros_like(difference),context],dim=-1)
        raw=self.decoder(latent)-self.decoder(reference)
        direction=F.normalize(raw,dim=-1,eps=1e-8)
        # Exact mathematical zero when relation difference is zero.
        strength=torch.tanh(difference.norm(dim=-1,keepdim=True))
        target=base[:,64:320]
        cap=self.limit*self.alpha.tanh()*target.norm(dim=-1,keepdim=True).detach()
        # Relation_difference is AP-AF; factor .5 yields symmetric barycenter.
        correction=.5*cap*gate[:,None]*strength*direction
        out=torch.cat([base[:,:64],target+correction],dim=-1)
        self.last={'gate':gate.detach(),'correction':correction,'bridge_signal':difference,
                   'delta_norm':correction.norm(dim=-1).detach(),
                   'relative_delta':(correction.norm(dim=-1)/(target.norm(dim=-1)+1e-8)).detach()}
        assert torch.isfinite(out).all()
        return out

    def summary(self):
        with torch.no_grad():
            return dict(gate_mean=float(self.chosen_gate.mean()),gate_std=float(self.chosen_gate.std()),
                        graph_intersection=float(self.base_graph_intersection.mean()),
                        mean_rel_change=float(self.relation_difference.abs().mean()),
                        alpha=float(self.alpha),
                        delta_rel_mean=float(self.last.get('relative_delta',torch.tensor(0.)).float().mean()))


def self_test():
    from numpy.random import default_rng
    import numpy as np
    g=default_rng(23)
    arr=[torch.tensor(g.standard_normal((871,d)),dtype=torch.float32) for d in (48,6,10,256)]
    ev={'outer_test':np.arange(780,871),'pheno_permutation':np.arange(871)}
    for v in ['gain_gate','constant_gate','confidence_gate','shuffled_gate']:
        ev[v]=g.uniform(.1,.9,871).astype(np.float32)
    ev['constant_gate'][:]=ev['gain_gate'][:780].mean()
    base=torch.tensor(g.standard_normal((871,320)),dtype=torch.float32)
    for cfg in list(CONDITIONS)[1:]:
        mod=GainRelationalBridge(cfg,arr,np.arange(780),np.arange(28),ev,top_k=8)
        assert torch.equal(mod(base),base),'zero-init parity '
        with torch.no_grad(): mod.alpha.fill_(1.)
        output=mod(base)
        rel=(output[:,64:]-base[:,64:]).norm(dim=-1)/base[:,64:].norm(dim=-1).clamp_min(1e-8)
        assert rel.max() <.10 +1e-6,(cfg,rel.max())
        mod.condition='GATE_OFF'
        assert torch.equal(mod(base),base),'gate off parity '+cfg
        mod.condition='RELATION_OFF'
        assert torch.equal(mod(base),base),'relation off parity '+cfg
        mod.condition='REAL'
        result=output[:,64:].square().mean();result.backward()
        assert torch.isfinite(mod.decoder[-1].weight.grad).all()
        assert torch.isfinite(mod.pheno_encoder[0].weight.grad).all()
    print('Bridge self-test PASS')

if __name__=='__main__':self_test()

