"""Query-disjoint disease evidence, independent of the DARG-CAM autograd graph."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)

def retrieve(logits, valid, top):
    if logits.shape[1] == 0:
        return logits, torch.empty_like(logits, dtype=torch.long), logits
    masked = logits.masked_fill(~valid, -1e4)
    # Stable order provides deterministic ID tie resolution.
    ix = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :min(top, logits.shape[1])]
    weights = torch.softmax(masked.gather(1, ix), 1)*valid.gather(1, ix)
    weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-8)
    return weights, ix, valid.gather(1, ix)

class QCRLPM(nn.Module):
    def __init__(self, cfg, texts):
        super().__init__()
        self.cfg = cfg
        d = cfg['projection_dim']
        self.register_buffer('texts', texts.detach().clone())
        self.cam_typical = nn.Linear(36, d, bias=False)
        self.cam_relation = nn.Linear(144, d, bias=False)
        self.corr_value = nn.Linear(256, d, bias=False)
        self.query_key = nn.Linear(400, d, bias=False)
        self.pair_key = nn.Linear(288, d, bias=False)
        self.pair_difference = nn.Linear(400, d, bias=False)
        self.text_projection = nn.Linear(1024, d, bias=False)
        self.semantic_joint = nn.Linear(d, d, bias=False)
        self.semantic_dimensions = nn.Linear(d, d, bias=False)
        self.modality_weights = nn.Linear(144, 4)
        self.view_weights = nn.Linear(146, 2)
        self.typical_gate = nn.Sequential(nn.Linear(152, 16), nn.Tanh(), nn.Linear(16, 1))
        self.confusion_gate = nn.Sequential(nn.Linear(152, 16), nn.Tanh(), nn.Linear(16, 1))

    def concepts(self, text_index):
        text = unit(self.texts[text_index])
        # The two caveats are shared measurement context in both conditions.
        common = unit((text[3]+text[7])[None])[0]
        rows = [text[i] if i not in (3,7) else common for i in range(8)]
        return unit(self.text_projection(torch.stack(rows)))

    def semantics(self, q, candidates, concepts, semantic_off):
        joint = unit(self.semantic_joint(q[:,None]+candidates[None]+q[:,None]*candidates[None]))
        scores = torch.einsum('npd,cd->npc', joint, concepts)/self.cfg['semantic_temperature']
        attention = torch.softmax(scores, -1)
        context = torch.einsum('npc,cd->npd', attention, concepts)
        compat = (joint*context).sum(-1)
        if semantic_off:
            context = context*0
            compat = compat*0
        return context, compat, attention

    def typical(self, h, raw, bank, concepts, semantic_off):
        cfg = self.cfg
        slots = h.reshape(-1,4,36)
        weights = .1+.6*torch.softmax(self.modality_weights(unit(h)), -1)
        evidences, reliabilities, selections = [], [], []
        for m in range(4):
            q = unit(slots[:,m])
            qz = unit(self.cam_typical(q))
            supports, reliability = [], []
            for cls in (0,1):
                p = bank['cam'][m,cls]
                counts = bank['counts'][m,cls]
                sim = .5*(q @ unit(p).T)+.5*(qz @ unit(self.cam_typical(unit(p))).T)
                concepts_cls = concepts[4:8] if cls == 0 else concepts[:4]
                _, semantic, _ = self.semantics(qz, unit(self.cam_typical(unit(p))),concepts_cls,semantic_off)
                rel = counts/(counts+4)
                score = sim/cfg['feature_temperature']+cfg['semantic_strength']*semantic+rel.clamp_min(1e-8).log()[None]
                valid = (counts>0)[None].expand(len(h),-1)
                w,ix,_ = retrieve(score,valid,cfg['top_typical'])
                supports.append((w*sim.gather(1,ix)).sum(1))
                reliability.append((w*rel[ix]).sum(1))
                selections.append(ix+(m*2+cls)*cfg['k'])
            evidences.append(torch.tanh((supports[0]-supports[1])/cfg['feature_temperature']))
            reliabilities.append(torch.minimum(reliability[0],reliability[1]))
        cam_evidence = (weights*torch.stack(evidences,1)).sum(1)
        cam_rel = (weights*torch.stack(reliabilities,1)).sum(1)
        # One global raw-CORR comparison, independent of the four CAM banks.
        qraw, qz = unit(raw), unit(self.corr_value(unit(raw)))
        supports, reliability = [], []
        for cls in (0,1):
            p,counts = bank['raw'][cls],bank['raw_counts'][cls]
            pz = unit(self.corr_value(unit(p)))
            sim = .5*(qraw @ unit(p).T)+.5*(qz @ pz.T)
            concepts_cls = concepts[4:8] if cls == 0 else concepts[:4]
            _,semantic,_ = self.semantics(qz,pz,concepts_cls,semantic_off)
            rel = counts/(counts+4)
            score = sim/cfg['feature_temperature']+cfg['semantic_strength']*semantic+rel.clamp_min(1e-8).log()[None]
            w,ix,_ = retrieve(score,(counts>0)[None].expand(len(h),-1),cfg['top_typical'])
            supports.append((w*sim.gather(1,ix)).sum(1))
            reliability.append((w*rel[ix]).sum(1))
            selections.append(ix+(8+cls)*cfg['k'])
        corr_evidence = torch.tanh((supports[0]-supports[1])/cfg['feature_temperature'])
        corr_rel = torch.minimum(reliability[0],reliability[1])
        view_input = torch.cat((unit(h),cam_evidence[:,None],corr_evidence[:,None]),1)
        view_w = .15+.7*torch.softmax(self.view_weights(view_input),-1)
        typical = view_w[:,0]*cam_evidence+view_w[:,1]*corr_evidence
        reliability = view_w[:,0]*cam_rel+view_w[:,1]*corr_rel
        return dict(typical=typical,typical_reliability=reliability,cam_evidence=cam_evidence,
                    corr_evidence=corr_evidence,modality_weights=weights,view_weights=view_w,
                    prototype_selection=torch.stack(selections,1))

    def confusion(self,h,raw,ids,bank,concepts,semantic_off):
        cfg = self.cfg
        ph,pr = bank['pair_cam'],bank['pair_raw']
        count = len(ph)
        qh,qr = unit(h),unit(raw)
        qkey = unit(self.query_key(torch.cat((qh,qr),-1)))
        key = unit(self.pair_key(torch.cat((unit(ph.mean(1)),unit(bank['delta_cam'])),-1)))
        difference = unit(self.pair_difference(torch.cat((unit(bank['delta_cam']),unit(bank['delta_raw'])),-1)))
        context,semantic,attention = self.semantics(qkey,difference,concepts,semantic_off)
        dims = .5+torch.sigmoid(self.semantic_dimensions(context))
        if semantic_off:
            dims = torch.ones_like(dims)
        zq = unit(self.cam_relation(qh)+self.corr_value(qr))
        za = unit(self.cam_relation(unit(ph[:,0]))+self.corr_value(unit(pr[:,0])))
        zt = unit(self.cam_relation(unit(ph[:,1]))+self.corr_value(unit(pr[:,1])))
        q = unit(zq[:,None]*dims)
        a,t = unit(za[None]*dims),unit(zt[None]*dims)
        sim_a,sim_t = (q*a).sum(-1),(q*t).sum(-1)
        evidence = torch.tanh((sim_a-sim_t)/cfg['relation_temperature'])
        cam_context = unit(h) @ unit(ph.mean(1)).T
        raw_context = qr @ unit(pr.mean(1)).T
        boundary = 1-(sim_a-sim_t).abs()/2
        relationship = (unit(zq[:,None]-(za+zt)[None]/2)*difference[None]).sum(-1).abs()
        logits = (qkey@key.T+.5*cam_context+.25*raw_context)/cfg['feature_temperature']
        logits = logits+.25*boundary+.25*relationship+cfg['semantic_strength']*semantic
        logits = logits+bank['pair_reliability'].clamp_min(1e-8).log()[None]
        valid = (ids[:,None]!=bank['pair_ids'][None,:,0]) & (ids[:,None]!=bank['pair_ids'][None,:,1])
        w,ix,active = retrieve(logits,valid,cfg['top_pairs'])
        signed = (w*evidence.gather(1,ix)).sum(1)
        semantic_reliability = torch.sigmoid(2*semantic)
        if semantic_off:
            semantic_reliability = torch.ones_like(semantic_reliability)*.5
        rel = bank['pair_reliability'][None]*((cam_context+1)/2)*semantic_reliability
        reliable = (w*rel.gather(1,ix)).sum(1)
        selected_attention = attention.gather(1,ix[:,:,None].expand(-1,-1,8))
        return dict(confusion=signed,confusion_reliability=reliable,
            selected_pairs=torch.where(active,ix,-1),pair_weights=w,
            pair_evidence=evidence.gather(1,ix),
            concept_attention=(selected_attention*w[:,:,None]).sum(1),
            has_pairs=active.any(1))

    def forward(self,h,base_logp,raw,ids,bank,*,hard=False,text_index=0,
                semantic_off=False,confusion_off=False,typical_off=False,correction_off=False):
        # Detachment here enforces the boundary even if a future caller omits it.
        h,base_logp,raw = h.detach(),base_logp.detach(),raw.detach()
        concepts = self.concepts(text_index)
        z = self.typical(h,raw,bank,concepts,semantic_off)
        n = len(h)
        z.update(confusion=h.new_zeros(n),confusion_reliability=h.new_zeros(n),
            selected_pairs=torch.empty((n,0),device=h.device,dtype=torch.long),
            pair_weights=h.new_zeros((n,0)),pair_evidence=h.new_zeros((n,0)),
            concept_attention=h.new_zeros((n,8)),has_pairs=torch.zeros(n,device=h.device,dtype=torch.bool))
        if hard and not confusion_off and len(bank['pair_ids']):
            z.update(self.confusion(h,raw,ids,bank,concepts,semantic_off))
        if typical_off:
            z['typical'] = z['typical']*0
            z['typical_reliability'] = z['typical_reliability']*0
        et,ec,rt,rc = (z[k] for k in ('typical','confusion','typical_reliability','confusion_reliability'))
        p = base_logp.exp()[:,0]
        uncertainty = 4*p*(1-p)
        margin = torch.tanh((base_logp[:,0]-base_logp[:,1])/2)
        agreement = .5+.5*et*ec
        coverage = z['has_pairs'].float()
        inputs = torch.cat((unit(h),torch.stack((margin,uncertainty,et,ec,rt,rc,agreement,coverage),1)),1)
        gt = torch.sigmoid(self.typical_gate(inputs)).squeeze(1)*rt*(.25+.75*et.abs())
        gc = torch.zeros_like(gt)
        if hard and not confusion_off and len(bank['pair_ids']):
            gc = torch.sigmoid(self.confusion_gate(inputs)).squeeze(1)*rc*agreement
        # A positive floor allows confident errors to be corrected with reliable
        # relations; reliability still scales both pathways continuously to zero.
        confidence_factor = .25+.75*torch.maximum(uncertainty,rc)
        delta = self.cfg['correction_bound']*torch.tanh(confidence_factor*(gt*et+gc*ec))
        if correction_off:
            delta = delta*0
        correction = torch.stack((delta/2,-delta/2),1)
        final = base_logp if correction_off else F.log_softmax(base_logp+correction,-1)
        z.update(logp=final,delta=delta,typical_gate=gt,confusion_gate=gc,base_logp=base_logp)
        return z

    def losses(self,z,query_labels):
        # Labels are a Query-only vector, never a full-patient lookup.
        y, cfg = query_labels,self.cfg
        sign = 1-2*y.float()
        base = z['base_logp']
        correct = base.argmax(1)==y
        confidence = base.exp().max(1).values
        difficulty = 1+(~correct).float()+.5*(1-confidence)
        # Every class contributes exactly half; bounded within each class.
        weights = torch.zeros_like(difficulty)
        classes = [c for c in (0,1) if (y==c).any()]
        for cls in classes:
            mask = y==cls
            weights[mask] = difficulty[mask]/difficulty[mask].sum()/len(classes)
        qcls = (F.nll_loss(z['logp'],y,reduction='none')*weights).sum()
        typical = (F.softplus(-sign*z['typical']/cfg['typical_loss_temperature'])*weights).sum()
        has = z['has_pairs']
        relation = z['delta'].new_zeros(())
        if has.any():
            wr = weights*has
            relation = (F.softplus(-sign*z['confusion']/cfg['relation_loss_temperature'])*wr).sum()/wr.sum()
        protected = correct & (confidence>=cfg['protect_confidence'])
        protect = z['delta'].new_zeros(())
        if protected.any():
            d = z['delta'][protected]
            protect = (F.softplus(-sign[protected]*d/.2)*d.square()).mean()
        total = qcls+cfg['lambda_typical']*typical+cfg['lambda_relation']*relation+cfg['lambda_protect']*protect
        return total,dict(qcls=qcls,typical=typical,qrelation=relation,protect=protect,total_new=total,
            query_weight_min=weights.min(),query_weight_max=weights.max())

def build(cfg,fold,texts,device):
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(774000+fold)
        module = QCRLPM(cfg,texts.cpu()).to(device)
    assert torch.equal(torch.get_rng_state(),before_cpu)
    assert torch.equal(torch.cuda.get_rng_state(),before_cuda)
    return module

def base_forward(base,state,training):
    captured = {}
    def capture(_module,args):
        assert args[0].shape == (871,144)
        captured['h'] = args[0].detach()
    assert not state['cam'].Outputlayer._forward_hooks and not state['cam'].Outputlayer._forward_pre_hooks
    hook = state['cam'].Outputlayer.register_forward_pre_hook(capture)
    try:
        out = base.forward(state,'E74.4-G2-Positive-DCR',training=training)
    finally:
        hook.remove()
    return out,captured['h']
