"""E81.6.1: prompt-dependent increments over shared visual pairwise relations.

All random initialization is private to the CPU constructor. No stochastic
operation, patient sampling, or patient labels occur in this module.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

PROMPTS = [
    "Nromal brain volume", "Uniform cortical thickness",
    "Symmetrical brain structure", "Normal ventricular morphology",
    "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
]
MEDICAL = ((0, 1, 2, 3), (4, 5, 6, 7))
SCRAMBLED = ((0, 1, 4, 5), (2, 3, 6, 7))


class CPRLRES(nn.Module):
    def __init__(self, medical, eta=.1, rho=.1, eps=1e-12):
        super().__init__()
        assert medical.shape == (8, 1024)
        self.eta, self.rho, self.eps = float(eta), float(rho), float(eps)
        self.local_proj = nn.Linear(9, 32)
        self.position = nn.Parameter(torch.empty(16, 32))
        self.local_norm = nn.LayerNorm(32)
        self.query_proj = nn.Linear(32, 32)
        self.query_norm = nn.LayerNorm(32)
        self.text_projector = nn.Linear(1024, 32, bias=False)
        self.key_proj = nn.Linear(64, 32)
        self.value_proj = nn.Linear(96, 32)
        self.null_key = nn.Parameter(torch.zeros(32))
        self.fusion_projection = nn.Linear(64, 32, bias=False)
        self.output = nn.Linear(32, 9, bias=False)
        normalized = F.normalize(medical.detach().clone().float(), dim=-1)
        center = normalized.mean(0, keepdim=True)
        centered = normalized - center
        assert torch.isfinite(centered).all() and (centered.norm(dim=-1) > 0).all()
        self.register_buffer('medical', medical.detach().clone().float())
        self.register_buffer('text_normalized', normalized)
        self.register_buffer('text_mean', center)
        self.register_buffer('text_centered', F.normalize(centered, dim=-1))
        self.register_buffer('null_value', torch.zeros(32))
        self.register_buffer('neighbors', torch.tensor(
            [[q for q in range(16) if q != p] for p in range(16)], dtype=torch.long))
        nn.init.normal_(self.position, 0., .02)
        nn.init.zeros_(self.output.weight)

    @classmethod
    def initialized(cls, salt, device, medical, eta=.1, rho=.1):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(salt))
            module = cls(medical, eta=eta, rho=rho)
        return module.to(device)

    @staticmethod
    def global_pool(z_rel):
        n = z_rel.shape[0]
        return F.avg_pool1d(z_rel.reshape(n * 4, 1, 36), 4, 4).reshape(n, 4, 9)

    def forward(self, z_cam, residual='bounded', grouping='medical',
                intervention='full', blocked_rows=(), eta=None):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f'expected CAM144, got {tuple(z_cam.shape)}')
        if residual not in ('bounded', 'unbounded') or grouping not in ('medical', 'scrambled'):
            raise ValueError((residual, grouping))
        n = z_cam.shape[0]
        route_eta = self.eta if eta is None else float(eta)
        z_rel = z_cam.detach() + route_eta * (z_cam - z_cam.detach())
        h_slots = z_rel.reshape(n, 4, 4, 9)
        h = h_slots.reshape(n, 16, 9)
        u = self.local_norm(self.local_proj(h) + self.position)
        g = self.global_pool(z_rel)
        q_visual = self.query_norm(self.query_proj(u))
        q_sem = F.normalize(self.text_projector(self.text_centered), dim=-1)
        receiver = u.unsqueeze(2).expand(n, 16, 15, 32)
        sender = u[:, self.neighbors, :]
        product, difference = receiver * sender, sender - receiver
        key = self.key_proj(torch.cat((product, difference.abs()), -1))
        value = self.value_proj(torch.cat((sender, product, difference), -1))
        all_keys = torch.cat((key, self.null_key.view(1, 1, 1, 32).expand(n, 16, 1, 32)), 2)
        all_values = torch.cat((value, self.null_value.view(1, 1, 1, 32).expand(n, 16, 1, 32)), 2)
        score_visual = torch.einsum('npd,npqd->npq', q_visual, all_keys) / math.sqrt(32.)
        semantic_key = F.normalize(key, dim=-1)
        score_semantic = torch.einsum('kd,npqd->nkpq', q_sem, semantic_key)
        if intervention == 'prompt_off':
            score_semantic = score_semantic * 0.
        if intervention.startswith('delete_prompt_'):
            mask = score_semantic.new_ones(8)
            mask[int(intervention.rsplit('_', 1)[1])] = 0.
            score_semantic = score_semantic * mask[None, :, None, None]
        # lambda_sem=1.0, fixed. Null's semantic score is exactly zero.
        score_semantic_null = F.pad(score_semantic, (0, 1), value=0.)
        score_prompt = score_visual[:, None] + score_semantic_null
        if intervention == 'null':
            score_visual = score_visual.clone()
            score_prompt = score_prompt.clone()
            score_visual[..., :15] = -torch.inf
            score_prompt[..., :15] = -torch.inf
        attention_visual = score_visual.softmax(-1)
        attention = score_prompt.softmax(-1)
        # Joint contraction uses identical accumulation for the baseline and
        # each prompt. Equal attentions therefore give bitwise equal messages.
        all_attention = torch.cat((attention_visual[:, None], attention), 1)
        all_messages = torch.einsum('nkpq,npqd->nkpd', all_attention, all_values)
        m_visual, messages = all_messages[:, 0], all_messages[:, 1:]
        delta_messages = messages - m_visual[:, None]
        groups = MEDICAL if grouping == 'medical' else SCRAMBLED
        if intervention == 'scrambled_groups': groups = SCRAMBLED
        if intervention == 'medical_groups': groups = MEDICAL
        normal = delta_messages[:, list(groups[0])].mean(1)
        abnormal = delta_messages[:, list(groups[1])].mean(1)
        relation = self.fusion_projection(torch.cat((normal, abnormal), -1))
        d = self.output(relation).reshape(n, 4, 4, 9)
        matching_g = g * 0. if intervention == 'global_off' else g
        matching = (d * matching_g.unsqueeze(2)).softmax(-1)
        weighted = matching * d
        raw = weighted.reshape(n, 144)
        reference_norm = z_cam.detach().norm(dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        if residual == 'unbounded' or intervention == 'unbounded':
            delta = raw
        else:
            delta = bound * raw / torch.sqrt(bound.square() + raw.square().sum(-1, keepdim=True) + self.eps)
        if intervention == 'off': delta = delta * 0.
        legal = torch.ones(n, dtype=torch.bool, device=z_cam.device)
        if blocked_rows:
            legal[list(blocked_rows)] = False
            delta = delta * legal[:, None]
        enhanced = z_cam + delta
        details = dict(z_cam=z_cam, z_rel=z_rel, h_slots=h_slots, h=h, u=u, g=g,
                       text_before=self.text_normalized, text_centered=self.text_centered,
                       gram_before=self.text_normalized @ self.text_normalized.T,
                       gram_centered=self.text_centered @ self.text_centered.T,
                       gram_after=q_sem @ q_sem.T, q_sem=q_sem, q_visual=q_visual,
                       key=key, value=value, semantic_key=semantic_key,
                       score_visual=score_visual, score_semantic=score_semantic,
                       score_semantic_null=score_semantic_null, score_prompt=score_prompt,
                       attention_visual=attention_visual, attention=attention,
                       m_visual=m_visual, messages=messages, delta_messages=delta_messages,
                       normal=normal, abnormal=abnormal, delta_contrast=abnormal-normal,
                       relation=relation, d=d, global_matching=matching,
                       weighted_relation=weighted, raw_delta=raw, delta=delta,
                       raw_ratio=raw.norm(dim=-1)/(reference_norm[:, 0]+self.eps),
                       delta_ratio=delta.norm(dim=-1)/(reference_norm[:, 0]+self.eps),
                       legal=legal, z_enhanced=enhanced)
        return enhanced, details


def residual_penalty(details, train):
    ids = torch.as_tensor(train, device=details['raw_delta'].device, dtype=torch.long)
    ids = ids[details['legal'][ids]]
    if not ids.numel(): raise RuntimeError('no legal training patients')
    return details['raw_ratio'][ids].square().mean()
