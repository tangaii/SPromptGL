"""Mechanism-refined patient-internal relational composition over CAM144."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


E81_PRCL_PATH = Path("/model/spromptgl/experiment/experiment81_prcl/prcl.py")
_E81 = None


def _load_e81_initializer():
    global _E81
    if _E81 is None:
        spec = importlib.util.spec_from_file_location("e81_readonly_prcl_initializer", E81_PRCL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _E81 = module
    return _E81


class PRCLv2(nn.Module):
    def __init__(self, anchors: torch.Tensor, condition_mode: str, hidden: int = 32):
        super().__init__()
        if tuple(anchors.shape) != (8, hidden):
            raise ValueError(f"expected eight frozen {hidden}-D anchors, got {tuple(anchors.shape)}")
        if condition_mode not in ("visual", "additive", "multiplicative"):
            raise ValueError(f"unknown condition mode: {condition_mode}")
        self.hidden = int(hidden)
        self.condition_mode = condition_mode
        self.local_proj = nn.Linear(9, hidden)
        self.local_norm = nn.LayerNorm(hidden)
        self.position = nn.Parameter(torch.empty(16, hidden))
        self.key_proj = nn.Linear(64, hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.value_proj = nn.Linear(96, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.fusion = nn.Linear(3 * hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, 9, bias=False)
        self.alpha_gate = nn.Linear(144, 1)
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(0.4 / 0.6)))

        head_identity = torch.zeros(8, hidden)
        head_identity[:, :8] = torch.eye(8)
        self.register_buffer("head_identity", head_identity, persistent=True)
        self.register_buffer("text_anchors", anchors.detach().float().clone(), persistent=True)
        src, dst = [], []
        candidate_index_map = torch.full((16, 16), -1, dtype=torch.long)
        for p in range(16):
            slots = 0
            for q in range(16):
                if p == q:
                    continue
                src.append(p)
                dst.append(q)
                candidate_index_map[p, q] = slots
                slots += 1
        self.register_buffer("edge_src", torch.tensor(src, dtype=torch.long), persistent=True)
        self.register_buffer("edge_dst", torch.tensor(dst, dtype=torch.long), persistent=True)
        self.register_buffer("candidate_index_map", candidate_index_map, persistent=True)
        self.register_buffer("undirected_pairs", torch.triu_indices(16, 16, offset=1), persistent=True)

        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.zeros_(self.alpha_gate.bias)
        self.init_record = {}

    @property
    def gamma(self):
        return 0.5 * torch.sigmoid(self.gamma_raw)

    def forward(self, z_cam: torch.Tensor, intervention: str = "full",
                text_override: torch.Tensor | None = None):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f"PRCL-v2 expects CAM144 [N,144], got {tuple(z_cam.shape)}")
        n = z_cam.shape[0]
        h = z_cam.reshape(n, 4, 4, 9).reshape(n, 16, 9)
        tokens = self.local_norm(self.local_proj(h) + self.position.unsqueeze(0))

        up = tokens[:, self.edge_src, :]
        uq = tokens[:, self.edge_dst, :]
        key_features = torch.cat((up * uq, (up - uq).abs()), dim=-1)
        key = self.key_norm(self.key_proj(key_features)).reshape(n, 16, 15, self.hidden)
        value_features = torch.cat((uq, up * uq, uq - up), dim=-1)
        value = self.value_proj(value_features).reshape(n, 16, 15, self.hidden)

        visual_query = self.query_proj(tokens)
        anchors = self.text_anchors if text_override is None else text_override
        if tuple(anchors.shape) != (8, self.hidden):
            raise ValueError(f"prompt override must have shape [8,{self.hidden}]")
        anchors = anchors.to(device=z_cam.device, dtype=z_cam.dtype)
        identity = self.head_identity.to(device=z_cam.device, dtype=z_cam.dtype)
        gamma = self.gamma * 0.0 if intervention == "gamma_zero" else self.gamma
        use_text = self.condition_mode != "visual" and intervention not in ("text_off", "nonsemantic")
        text_term = anchors if use_text else torch.zeros_like(anchors)
        if intervention == "swap_groups":
            text_term = torch.cat((text_term[4:], text_term[:4]), dim=0)
        if intervention == "within_group_permute":
            order = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5], device=z_cam.device)
            text_term = text_term[order]
            identity = identity[order]

        base = visual_query[:, None, :, :] + identity[None, :, None, :]
        if self.condition_mode == "multiplicative" and use_text:
            condition_factor = 1.0 + gamma * torch.tanh(math.sqrt(self.hidden) * text_term)
            query = self.query_norm(base * condition_factor[None, :, None, :])
        else:
            condition_factor = torch.ones_like(text_term)
            query = self.query_norm(
                base + gamma * math.sqrt(self.hidden) * text_term[None, :, None, :]
            )

        scores = torch.einsum("nkpd,npqd->nkpq", query, key) / math.sqrt(self.hidden)
        null_score = torch.einsum("nkpd,d->nkp", query, self.null_key) / math.sqrt(self.hidden)
        weights = torch.softmax(torch.cat((scores, null_score.unsqueeze(-1)), dim=-1), dim=-1)
        if intervention == "uniform":
            weights = torch.full_like(weights, 1.0 / 16.0)
        elif intervention == "null_only":
            weights = torch.zeros_like(weights)
            weights[..., -1] = 1.0

        messages = torch.einsum("nkpq,npqd->nkpd", weights[..., :15], value)
        normal_message = messages[:, :4].mean(dim=1)
        abnormal_message = messages[:, 4:].mean(dim=1)
        if intervention == "swap_groups":
            normal_message, abnormal_message = abnormal_message, normal_message
        interaction = (abnormal_message - normal_message) * torch.tanh(
            abnormal_message + normal_message
        )
        if intervention == "nonlinear_off":
            interaction = torch.zeros_like(interaction)
        relation_message = F.gelu(self.fusion(torch.cat(
            (normal_message, abnormal_message, interaction), dim=-1
        )))
        raw_delta = torch.tanh(self.output(relation_message).reshape(n, 144))
        alpha = 0.1 * torch.sigmoid(self.alpha_gate(z_cam))
        u = alpha * raw_delta
        u_norm = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        z_norm = torch.linalg.vector_norm(z_cam, dim=-1, keepdim=True)
        bound = 0.1 * z_norm
        eps = torch.finfo(z_cam.dtype).eps
        delta = bound * u / torch.sqrt(bound.square() + u_norm.square() + eps)
        if intervention in ("prcl_off", "writeback_off", "residual_off"):
            delta = torch.zeros_like(delta)
        z_prcl = z_cam + delta

        return z_prcl, {
            "tokens": tokens,
            "query": query,
            "key": key,
            "value": value,
            "attention": weights,
            "candidate_map": self.candidate_index_map,
            "messages": messages,
            "normal_message": normal_message,
            "abnormal_message": abnormal_message,
            "interaction": interaction,
            "relation_message": relation_message,
            "alpha": alpha,
            "gamma": gamma,
            "text_term": text_term,
            "text_term_norm": text_term.norm(dim=-1),
            "visual_query_norm": visual_query.norm(dim=-1).mean(dim=-1),
            "condition_factor": condition_factor,
            "raw_delta": raw_delta,
            "unbounded_residual": u,
            "unbounded_residual_norm": u_norm,
            "bound": bound,
            "u_over_b": u_norm / bound.clamp_min(eps),
            "delta": delta,
            "delta_ratio": torch.linalg.vector_norm(delta, dim=-1) / z_norm.squeeze(-1).clamp_min(eps),
            "smooth_attenuation": torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / u_norm.clamp_min(eps),
            "intervention": intervention,
        }


def _copy_e81_initialization(module: PRCLv2, anchors: torch.Tensor, seed: int, device):
    e81 = _load_e81_initializer()
    original = e81.initialize(anchors.cpu(), use_text=True, seed=seed, device=torch.device("cpu"))
    old = original.state_dict()
    copied, skipped = [], []
    with torch.no_grad():
        for name, value in module.state_dict().items():
            source = old.get(name)
            target = dict(module.named_parameters()).get(name)
            if source is None or target is None or tuple(source.shape) != tuple(target.shape):
                if target is not None:
                    skipped.append(name)
                continue
            if name == "alpha_gate.bias":
                skipped.append(name + ": intentionally reset to alpha=0.02")
                continue
            target.copy_(source.to(dtype=target.dtype))
            copied.append(name)
        module.alpha_gate.weight.zero_()
        module.alpha_gate.bias.fill_(math.log(0.2 / 0.8))
    module.init_record = {
        "source": str(E81_PRCL_PATH),
        "source_seed": int(seed),
        "copied_named_parameters": copied,
        "intentional_initialization_changes": [
            "alpha_gate.bias reset from E81 alpha=0.05 to alpha=0.02",
            "fusion/output biases removed by design",
        ],
        "not_copied": skipped,
    }
    return module.to(device)


def initialize(anchors: torch.Tensor, condition_mode: str, seed: int, device):
    """Reuse same-name E81 initial values without consuming the public RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        module = PRCLv2(anchors.detach().cpu(), condition_mode=condition_mode)
    return _copy_e81_initialization(module, anchors, seed, device)


def residual_penalty(delta: torch.Tensor, z_cam: torch.Tensor, indices: torch.Tensor):
    ids = indices.to(device=delta.device, dtype=torch.long)
    numerator = delta[ids].square().sum(dim=-1)
    denominator = z_cam[ids].square().sum(dim=-1).clamp_min(torch.finfo(z_cam.dtype).eps)
    return (numerator / denominator).mean()
