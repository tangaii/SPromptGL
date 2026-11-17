"""E81.6: eight independent prompt-conditioned within-patient relations and RES.

No stochastic operation is used after private CPU initialization.
The only formal differences are K/V features and fixed condition vectors.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

GROUP_PAIRWISE = 'pairwise'
GROUP_UNARY = 'unary'
PROMPTS = [
    'Nromal brain volume', 'Uniform cortical thickness',
    'Symmetrical brain structure', 'Normal ventricular morphology',
    'Altered brain volume', 'Abnormal functional connectivity',
    'White matter integrity issues', 'Different activation patterns',
]


class PRCLRES(nn.Module):
    def __init__(self, medical, anonymous, eta=.1, rho=.1, eps=1e-12):
        super().__init__()
        assert medical.shape == anonymous.shape == (8, 1024)
        self.eta, self.rho, self.eps = float(eta), float(rho), float(eps)
        self.local_proj = nn.Linear(9, 32)
        self.position = nn.Parameter(torch.empty(16, 32))
        self.local_norm = nn.LayerNorm(32)
        self.query_proj = nn.Linear(32, 32)
        self.text_projector = nn.Linear(1024, 32)
        self.gamma_logit = nn.Parameter(torch.zeros(()))
        self.query_norm = nn.LayerNorm(32)
        self.key_proj = nn.Linear(64, 32)
        self.value_proj = nn.Linear(96, 32)
        self.null_key = nn.Parameter(torch.zeros(32))
        self.competition = nn.Linear(64, 2)
        # A bias-free fusion makes zero messages map to exact zero, including
        # after training. This is shared by all three groups.
        self.fusion_projection = nn.Linear(64, 32, bias=False)
        self.output = nn.Linear(32, 9, bias=False)
        self.register_buffer('medical', medical.detach().clone().float())
        self.register_buffer('anonymous', anonymous.detach().clone().float())
        self.register_buffer('null_value', torch.zeros(32))
        self.register_buffer('neighbors', torch.tensor(
            [[q for q in range(16) if q != p] for p in range(16)], dtype=torch.long))
        nn.init.normal_(self.position, 0., .02)
        nn.init.zeros_(self.competition.weight)
        nn.init.zeros_(self.competition.bias)
        nn.init.zeros_(self.output.weight)

    @classmethod
    def initialized(cls, salt, device, medical, anonymous, eta=.1, rho=.1):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(salt))
            module = cls(medical, anonymous, eta=eta, rho=rho)
        return module.to(device)

    @staticmethod
    def construct_features(receiver, sender, mode):
        if mode == GROUP_UNARY:
            nonlinear = sender.tanh()
            return (torch.cat((sender, nonlinear), -1),
                    torch.cat((sender, nonlinear, sender.square()), -1))
        if mode == GROUP_PAIRWISE:
            product, difference = receiver * sender, sender - receiver
            return (torch.cat((product, difference.abs()), -1),
                    torch.cat((sender, product, difference), -1))
        raise ValueError(mode)

    @staticmethod
    def global_pool(z_rel):
        n = z_rel.shape[0]
        return F.avg_pool1d(z_rel.reshape(n * 4, 1, 36), 4, 4).reshape(n, 4, 9)

    def forward(self, z_cam, mode=GROUP_PAIRWISE, condition='medical',
                intervention='full', blocked_rows=(), eta=None):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f'expected CAM144, got {tuple(z_cam.shape)}')
        n = z_cam.shape[0]
        route_eta = self.eta if eta is None else float(eta)
        z_rel = z_cam.detach() + route_eta * (z_cam-z_cam.detach())
        h_slots = z_rel.reshape(n, 4, 4, 9)
        h = h_slots.reshape(n, 16, 9)
        u = self.local_norm(self.local_proj(h) + self.position)
        g = self.global_pool(z_rel)
        if condition not in ('medical', 'anonymous'):
            raise ValueError(condition)
        text = getattr(self, condition)
        if intervention == 'anonymous': text = self.anonymous
        if intervention == 'medical': text = self.medical
        if intervention == 'swap_groups': text = text.roll(4, 0)
        if intervention.startswith('swap_prompt_'):
            k = int(intervention.rsplit('_', 1)[1])
            order = torch.arange(8, device=text.device)
            j = (k+4) % 8
            order[k], order[j] = order[j].clone(), order[k].clone()
            text = text[order]
        before = F.normalize(text, dim=-1)
        projected = self.text_projector(before)
        after = F.normalize(projected, dim=-1)
        gram_before, gram_after = before @ before.T, after @ after.T
        geo = (gram_after-gram_before).square().mean()
        semantic = projected
        if intervention == 'prompt_off': semantic = semantic * 0.
        if intervention.startswith('delete_prompt_'):
            mask = torch.ones(8, device=text.device, dtype=text.dtype)
            mask[int(intervention.rsplit('_', 1)[1])] = 0.
            semantic = semantic * mask[:, None]
        gamma = self.gamma_logit.sigmoid()
        query = self.query_norm(self.query_proj(u)[:, None, :, :] +
                                gamma*semantic[None, :, None, :])
        receiver = u.unsqueeze(2).expand(n, 16, 15, 32)
        sender = u[:, self.neighbors, :]
        feature_mode = GROUP_UNARY if intervention == 'unary' else mode
        key_input, value_input = self.construct_features(receiver, sender, feature_mode)
        key, value = self.key_proj(key_input), self.value_proj(value_input)
        all_keys = torch.cat((key, self.null_key.view(1,1,1,32).expand(n,16,1,32)), 2)
        all_values = torch.cat((value, self.null_value.view(1,1,1,32).expand(n,16,1,32)), 2)
        scores = torch.einsum('nkpd,npqd->nkpq', query, all_keys) / math.sqrt(32.)
        if intervention == 'null':
            scores = scores.clone()
            scores[..., :15] = -torch.inf
        attention = scores.softmax(-1)
        messages = torch.einsum('nkpq,npqd->nkpd', attention, all_values)
        normal, abnormal = messages[:, :4].mean(1), messages[:, 4:].mean(1)
        competition = self.competition(torch.cat((normal, abnormal), -1)).softmax(-1)
        selected_normal = competition[..., 0:1]*normal
        selected_abnormal = competition[..., 1:2]*abnormal
        relation = self.fusion_projection(torch.cat((selected_normal, selected_abnormal), -1))
        d = self.output(relation).reshape(n, 4, 4, 9)
        # global_off removes global conditioning while retaining the uniform
        # 1/9 matching scale. It is a frozen inference diagnostic only.
        matching_g = g*0. if intervention == 'global_off' else g
        matching = (d*matching_g.unsqueeze(2)).softmax(-1)
        weighted = matching*d
        raw = weighted.reshape(n, 144)
        reference_norm = torch.linalg.vector_norm(z_cam.detach(), dim=-1, keepdim=True)
        bound = self.rho*reference_norm
        delta = bound*raw/torch.sqrt(bound.square()+raw.square().sum(-1, keepdim=True)+self.eps)
        if intervention == 'off': delta = delta*0.
        legal = torch.ones(n, dtype=torch.bool, device=z_cam.device)
        if blocked_rows:
            legal[list(blocked_rows)] = False
            delta = delta*legal[:, None]
        enhanced = z_cam+delta
        details = dict(z_cam=z_cam, z_rel=z_rel, h_slots=h_slots, h=h, u=u, g=g,
                       text_before=before, text_projected=projected, gram_before=gram_before,
                       gram_after=gram_after, geo=geo, gamma=gamma, query=query, key=key,
                       value=value, attention=attention, messages=messages,
                       normal=normal, abnormal=abnormal, competition=competition,
                       relation=relation, d=d, global_matching=matching,
                       weighted_relation=weighted, raw_delta=raw, delta=delta,
                       delta_ratio=(delta.norm(dim=-1)/(reference_norm[:,0]+self.eps)),
                       legal=legal, z_enhanced=enhanced)
        return enhanced, details


def residual_penalty(details, train):
    ids = torch.as_tensor(train, device=details['delta'].device, dtype=torch.long)
    ids = ids[details['legal'][ids]]
    if not ids.numel(): raise RuntimeError('no legal training patients')
    return details['delta_ratio'][ids].square().mean()
