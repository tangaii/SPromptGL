"""Signed disease evidence at the unchanged PLNNoPEN classifier boundary."""
from __future__ import annotations
from pathlib import Path
import contextlib
import json
import sys
import types
import torch
from torch import nn
import torch.nn.functional as F
from memory import Memory


class DSCBPM(nn.Module):
    def __init__(self, cfg, texts, raw):
        super().__init__()
        self.cfg = cfg
        d = cfg['projection_dim']
        self.register_buffer('texts', texts.clone())  # [2,8,1024] medical/anonymous
        self.register_buffer('raw', raw.detach().clone())
        self.memory = Memory(cfg, len(raw))
        self.cam_projection = nn.Linear(36, d, bias=False)
        self.corr_projection = nn.Linear(256, d, bias=False)
        self.text_projection = nn.Linear(1024, d, bias=False)
        self.semantic_dimensions = nn.Linear(d, d, bias=False)
        self.modality_weights = nn.Linear(144, 4)
        self.typical_gate = nn.Sequential(nn.Linear(149, 16), nn.Tanh(), nn.Linear(16, 1))
        self.boundary_gate = nn.Sequential(nn.Linear(149, 16), nn.Tanh(), nn.Linear(16, 1))

    def anchors(self, text_index, swap=False):
        text = F.normalize(self.texts[text_index], dim=-1)
        # Same neutral/caveat contribution in both classes for medical and its
        # matched anonymous condition; these prompts are not class biomarkers.
        neutral = text[[3, 7]].mean(0)
        anchors = torch.stack((text[4:7].mean(0)+neutral, text[0:3].mean(0)+neutral))
        anchors = F.normalize(self.text_projection(anchors), dim=-1)
        return anchors.flip(0) if swap else anchors

    def signed_pair_scores(self, raw, pair_raw, tdiff, corr_off=False):
        dimensions = .5 + torch.sigmoid(self.semantic_dimensions(tdiff))
        q = F.normalize(self.corr_projection(raw)*dimensions, dim=-1)
        a = F.normalize(self.corr_projection(pair_raw[:, 0])*dimensions, dim=-1)
        t = F.normalize(self.corr_projection(pair_raw[:, 1])*dimensions, dim=-1)
        scores = torch.tanh((q @ a.T - q @ t.T) / self.cfg['boundary_temperature'])
        return scores * (0. if corr_off else 1.)

    def forward(self, h, base_logp, ids, *, boundary=False, text_index=0, swap=False,
                semantic_off=False, corr_off=False, boundary_off=False):
        cfg, mem = self.cfg, self.memory
        anchors = self.anchors(text_index, swap)
        tdiff = F.normalize(anchors[0]-anchors[1], dim=0)
        if semantic_off:
            tdiff = torch.zeros_like(tdiff)
        semantic_scale = 0. if semantic_off else cfg['semantic_strength']
        # Floor and normalization are a structural anti-collapse constraint;
        # no extra loss or hard-coded suppression of non-CORR contexts.
        modw = .1 + .6*torch.softmax(self.modality_weights(F.normalize(h, dim=-1)), dim=-1)
        typical_m, raw_m, usage, class_reliabilities = [], [], [], []
        qraw = F.normalize(self.raw[ids], dim=-1)
        qproj = F.normalize(self.corr_projection(self.raw[ids]), dim=-1)
        for m in range(4):
            supports, raw_supports, reliability = [], [], []
            q = F.normalize(h[:, m*36:(m+1)*36], dim=-1)
            qz = F.normalize(self.cam_projection(h[:, m*36:(m+1)*36])+self.corr_projection(self.raw[ids]), dim=-1)
            for cls in (0, 1):
                cam, raw, counts, valid = mem.leave_one_out(ids, m, cls)
                slot = cam[..., m*36:(m+1)*36]
                cs = (q[:, None]*F.normalize(slot, dim=-1)).sum(-1)
                rs = .5*(qraw[:, None]*F.normalize(raw, dim=-1)).sum(-1) + .5*(qproj[:, None]*F.normalize(self.corr_projection(raw), dim=-1)).sum(-1)
                z = F.normalize(self.cam_projection(slot)+self.corr_projection(raw), dim=-1)
                semantic = (z*anchors[cls]).sum(-1) + (qz[:, None]*z*anchors[cls]).sum(-1)
                rel = counts/(counts+4)
                logits = cs/cfg['feature_temperature'] + semantic_scale*semantic + torch.log(rel.clamp_min(1e-8))
                logits = logits.masked_fill(~valid, -1e4)
                values, selected = torch.topk(logits, cfg['top_typical'], dim=1)
                active = valid.gather(1, selected)
                weight = torch.softmax(values, dim=1)*active
                weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
                cam_score = (weight*cs.gather(1, selected)).sum(1)
                raw_score = (weight*rs.gather(1, selected)).sum(1) * (0. if corr_off else 1.)
                supports.append(.5*cam_score+.5*raw_score)
                raw_supports.append(.5*raw_score)
                reliability.append((weight*rel.gather(1, selected)).sum(1))
                usage.append(selected + (m*2+cls)*cfg['k'])
            typical_m.append(torch.tanh((supports[0]-supports[1])/cfg['feature_temperature']))
            raw_m.append(raw_supports[0]-raw_supports[1])
            class_reliabilities.append(torch.stack(reliability, dim=1).mean(1))
        typical_per_modality = torch.stack(typical_m, dim=1)
        typical = (typical_per_modality*modw).sum(1)
        typrel = (torch.stack(class_reliabilities, dim=1)*modw).sum(1)
        raw_contrast = (torch.stack(raw_m, dim=1)*modw).sum(1)
        boundary_evidence = torch.zeros_like(typical)
        boundary_reliability = torch.zeros_like(typical)
        selected_pairs = torch.full((len(h), cfg['top_pairs']), -1, device=h.device, dtype=torch.long)
        pairweights = torch.zeros_like(selected_pairs, dtype=h.dtype)
        pair_scores = h.new_zeros((len(h), 0))
        count = int(mem.pair_count)
        if boundary and not boundary_off and count:
            key = mem.pair_key[:count].view(count, 4, 36)
            similarity = torch.einsum('nmd,pmd->npm', F.normalize(h.view(-1, 4, 36), dim=-1), F.normalize(key, dim=-1))
            context = (similarity*modw[:, None]).sum(-1)
            difference = F.normalize(self.corr_projection(mem.pair_difference[:count]), dim=-1)
            # Candidate-specific semantic compatibility cannot cancel out.
            semantic = (difference*tdiff).sum(-1)[None] + qproj @ (difference*tdiff).T
            boundary_location = (F.normalize(h, dim=-1) @ F.normalize(mem.pair_cam[:count, 0], dim=-1).T -
                                 F.normalize(h, dim=-1) @ F.normalize(mem.pair_cam[:count, 1], dim=-1).T).abs()
            scores = context/cfg['feature_temperature'] + semantic_scale*semantic - .25*boundary_location*(1-typical.abs()[:, None])
            scores = scores + torch.log(mem.pair_reliability[:count].clamp_min(1e-8))[None]
            valid = (ids[:, None] != mem.pair_ids[:count, 0][None]) & (ids[:, None] != mem.pair_ids[:count, 1][None])
            scores = scores.masked_fill(~valid, -1e4)
            values, selected = torch.topk(scores, min(cfg['top_pairs'], count), dim=1)
            active = valid.gather(1, selected)
            weights = torch.softmax(values, dim=1)*active
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-8)
            pair_scores = self.signed_pair_scores(self.raw[ids], mem.pair_raw[:count], tdiff, corr_off)
            boundary_evidence = (weights*pair_scores.gather(1, selected)).sum(1)
            semantic_reliability = torch.sigmoid(semantic.gather(1, selected)*semantic_scale)
            quality = ((context.gather(1, selected)+1)/2)*mem.pair_reliability[:count][selected]*semantic_reliability
            boundary_reliability = (weights*quality).sum(1)
            selected_pairs[:, :selected.shape[1]] = torch.where(active, selected, -1)
            pairweights[:, :selected.shape[1]] = weights
        p = base_logp.exp()[:, 0]
        uncertainty = .2+.8*4*p*(1-p)
        inputs = torch.cat((F.normalize(h, dim=-1), torch.stack((p, typical, boundary_evidence, typrel, boundary_reliability), dim=1)), dim=1)
        # G1 and pre-activation G2 do not even execute boundary_gate.
        gt = torch.sigmoid(self.typical_gate(inputs)).squeeze(1)
        gb = torch.sigmoid(self.boundary_gate(inputs)).squeeze(1) if boundary and not boundary_off and count else torch.zeros_like(gt)
        agreement = .5+.5*typical*boundary_evidence
        typical_term = gt*typrel*(.2+.8*typical.abs())*typical
        boundary_term = gb*boundary_reliability*agreement*boundary_evidence
        margin_correction = cfg['correction_bound']*torch.tanh(uncertainty*(typical_term+boundary_term))
        correction = torch.stack((margin_correction/2, -margin_correction/2), dim=1)
        return dict(typical=typical, boundary=boundary_evidence, correction=correction,
                    margin_correction=margin_correction, raw_contrast=raw_contrast, modw=modw,
                    typical_per_modality=typical_per_modality, selected_pairs=selected_pairs,
                    pair_weights=pairweights, pair_scores=pair_scores,
                    boundary_reliability=boundary_reliability, prototype_selection=torch.stack(usage, dim=1),
                    typical_gate=gt, boundary_gate=gb)

    def losses(self, evidence, labels, allowed, text_index, boundary):
        sign = 1.-2.*labels[allowed].float()
        typical = F.softplus(-sign*evidence['typical'][allowed]/.5).mean()
        rank = typical.new_zeros(())
        n = int(self.memory.pair_count)
        if boundary and n:
            a = self.anchors(text_index)
            tdiff = F.normalize(a[0]-a[1], dim=0)
            raw = self.memory.pair_raw[:n]
            # Direct supervised comparison of the two legal training members;
            # these endpoints remain excluded from query-memory retrieval.
            z = self.signed_pair_scores(raw.reshape(-1, 256), raw, tdiff)
            row = torch.arange(n, device=z.device)
            positive = z[2*row, row]; negative = z[2*row+1, row]
            rank = .5*(F.softplus((self.cfg['ranking_margin']-positive)/.2).mean()+
                       F.softplus((self.cfg['ranking_margin']+negative)/.2).mean())
        return typical, rank


def build(cfg, fold, texts, raw):
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(773000+fold)
        module = DSCBPM(cfg, texts.cpu(), raw.cpu()).to(raw.device)
    return module


def forward(base, state, module, group, local_epoch, training, enabled=True, text_override=None, **interventions):
    capture = {}
    module.train(training)
    def pre(_, args):
        assert args[0].shape == (871, 144)
        capture['h'] = args[0]
    def post(_, args, output):
        logp, hidden = output
        capture['base_logp'] = logp
        if not enabled:
            return output
        evidence = module(capture['h'], logp, torch.arange(871, device=logp.device),
                          boundary=group != 'E77.3-G1' and local_epoch > module.cfg['warmup'],
                          text_index=(1 if group == 'E77.3-G3' else 0) if text_override is None else text_override, **interventions)
        capture['evidence'] = evidence
        return F.log_softmax(logp+evidence['correction'], dim=-1), hidden
    assert not state['cam'].Outputlayer._forward_hooks and not state['cam'].Outputlayer._forward_pre_hooks
    hp = state['cam'].Outputlayer.register_forward_pre_hook(pre)
    ho = state['cam'].Outputlayer.register_forward_hook(post)
    try:
        out = base.forward(state, 'E74.4-G2-Positive-DCR', training=training)
    finally:
        hp.remove(); ho.remove()
    return out, capture


def prepare_texts(here, config):
    path = here/'runtime/text_embeddings.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    from path_adapter import sha_file
    old_path = Path('/data/adodas/SPromptGL-main/experiment/experiment77_1_sc_perm/runtime/text_embeddings.pt')
    assert sha_file(old_path) == '4ee8ad50d69d7d3a40b21933cec69a37be50f2d580cd5a17a4b8d07dd42725d7'
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    assert old['texts']['medical'] == config['medical_prompts_reference']
    enc = config['text_encoder_reference']
    assert sha_file(Path(enc['weights_path'])) == enc['weights_sha256']
    sys.path.insert(0, enc['encoder_code_root'])
    try:
        import ftfy
    except ImportError:
        shim = types.ModuleType('ftfy'); shim.fix_text = lambda value: value; sys.modules['ftfy'] = shim
    from CLIP.clip import clip
    with torch.random.fork_rng(devices=[]):
        model, _ = clip.load(enc['weights_path'], device='cpu', jit=False)
        model.eval()
        with torch.no_grad():
            anonymous = model.encode_text(clip.tokenize(config['anonymous_prompts'])).float().cpu()
    result = dict(texts=torch.stack((old['embeddings']['medical'], anonymous)),
                  medical_text=config['medical_prompts_reference'], anonymous_text=config['anonymous_prompts'],
                  medical_source_sha=sha_file(old_path), rn50_sha=enc['weights_sha256'])
    torch.save(result, path)
    return result
