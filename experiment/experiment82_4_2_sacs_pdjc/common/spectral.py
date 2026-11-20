"""Fixed Original O, cosine medical attention and bounded CAM144 corrections."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8
QUERY_RANK = 32


class ScaleBranchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cam, scale):
        ctx.scale = scale
        return cam.view_as(cam)

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


def smooth_limit(raw, cam, fraction):
    cap = fraction * cam.norm(dim=-1, keepdim=True)
    ratio = raw.norm(dim=-1, keepdim=True) / cap.clamp_min(EPS)
    small = ratio < 1e-4
    near_zero = torch.where(small, ratio, 0.0)
    safe_ratio = torch.where(small, 1.0, ratio)
    scale = torch.where(small, 1 - near_zero.square() / 3 + 2 * near_zero.pow(4) / 15,
                        ratio.tanh() / safe_ratio)
    return raw * scale * (cap / cap.clamp_min(EPS)), scale


class PromptCorrection(nn.Module):
    def __init__(self, spectrum, text, exclusions, cfg):
        super().__init__()
        self.subjects = cfg["subjects"]
        self.branch_gradient = cfg["branch_gradient"]
        self.branch_scale = cfg["branch_scale"]
        self.norm_limit = cfg["residual"]["norm_limit"]
        prompt_cfg = cfg["prompt"]
        self.temperature_min = prompt_cfg["attention_temperature_min"]
        self.temperature_max = prompt_cfg["attention_temperature_max"]
        fraction = (prompt_cfg["attention_temperature_initial"] - self.temperature_min) / (
            self.temperature_max - self.temperature_min
        )
        self.temperature_logit = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))

        self.register_buffer("basis", spectrum["U"].double().clone())
        self.register_buffer("center", spectrum["mu"].double().clone())
        self.register_buffer("variance", spectrum["lambda"].double().clone())
        self.register_buffer("logvar", spectrum["loglambda_normalized"].double().clone())
        self.register_buffer("nu", spectrum["nu_normalized"].double().clone())
        self.register_buffer("query_medical", text["q_medical"].float().clone())
        self.register_buffer("query_anonymous", text["q_anonymous"].float().clone())
        allowed = torch.ones(self.subjects, dtype=torch.bool)
        allowed[exclusions] = False
        self.register_buffer("allowed", allowed)

        self.u_projection = nn.Linear(4, QUERY_RANK)
        self.u_position = nn.Parameter(torch.empty(16, QUERY_RANK))
        self.key = nn.Linear(QUERY_RANK, QUERY_RANK, bias=False)
        self.null = nn.Parameter(torch.empty(QUERY_RANK))
        nn.init.normal_(self.u_position, std=0.02)
        nn.init.normal_(self.null, std=0.02)

    @property
    def temperature(self):
        return self.temperature_min + (self.temperature_max - self.temperature_min) * self.temperature_logit.sigmoid()

    def encode_u(self, coordinates):
        white = coordinates / (self.variance.to(coordinates).sqrt() + EPS)
        features = torch.stack((coordinates, white, self.logvar.to(coordinates).expand_as(coordinates),
                                self.nu.to(coordinates).expand_as(coordinates)), dim=-1)
        tokens = self.u_projection(features) + self.u_position.to(coordinates)
        return tokens, white

    def attend(self, tokens, query, semantic_off=False):
        q = F.normalize(query.to(tokens), dim=-1, eps=EPS)
        k = F.normalize(self.key(tokens), dim=-1, eps=EPS)
        null_key = F.normalize(self.null.to(tokens), dim=-1, eps=EPS)
        prompt_scores = torch.einsum("btd,qd->bqt", k, q)
        null_scores = (q @ null_key).view(1, -1, 1).expand(len(tokens), -1, -1)
        attention = torch.cat((prompt_scores, null_scores), dim=-1).div(self.temperature.to(tokens)).softmax(-1)
        semantic = attention[..., :-1]
        coverage = 1 - (1 - semantic).prod(dim=1)
        mass = semantic.sum(dim=1).clamp_min(EPS)
        context = torch.einsum("bqt,qd->btd", semantic, q) / mass.unsqueeze(-1)
        if semantic_off:
            coverage = torch.full_like(coverage, self.neutral_coverage(tokens.shape[1]))
            context = torch.zeros_like(context)
        return {"attention": attention, "coverage": coverage, "context": context,
                "query": q, "temperature": self.temperature}

    @staticmethod
    def neutral_coverage(token_count):
        return 1 - (1 - 1 / (token_count + 1)) ** 8

    def prepare(self, cam, query=None, semantic_off=False):
        if tuple(cam.shape) != (self.subjects, 144):
            raise ValueError(f"Expected historical CAM144, got {tuple(cam.shape)}")
        branch_cam = ScaleBranchGradient.apply(cam, self.branch_gradient)
        centered = branch_cam - self.center.to(branch_cam)
        coordinates = centered @ self.basis.to(branch_cam)
        tokens, white = self.encode_u(coordinates)
        selected_query = self.query_medical if query is None else query
        prompt = self.attend(tokens, selected_query, semantic_off)
        entropy = -(prompt["attention"] * prompt["attention"].clamp_min(EPS).log()).sum(-1)
        prompt.update({
            "branch_cam": branch_cam,
            "coordinates_u": coordinates,
            "white_u": white,
            "tokens_u": tokens,
            "attention_entropy": entropy / math.log(prompt["attention"].shape[-1]),
            "null_weight": prompt["attention"][..., -1],
        })
        return prompt

    def finish(self, cam, raw_u, raw_v, details, *, residual_off=False):
        raw_u = raw_u.masked_fill(~self.allowed[:, None], 0)
        raw_v = raw_v.masked_fill(~self.allowed[:, None], 0)
        if residual_off:
            raw_u = torch.zeros_like(raw_u)
            raw_v = torch.zeros_like(raw_v)
        raw = raw_u + raw_v
        branch_cam = details["branch_cam"]
        delta, scale = smooth_limit(raw, branch_cam, self.norm_limit)
        component_u = raw_u * scale
        component_v = raw_v * scale
        norm = branch_cam.norm(dim=-1).clamp_min(EPS)
        return cam + delta, {
            **details,
            "raw_u": raw_u,
            "raw_v": raw_v,
            "raw_delta": raw,
            "delta_u": component_u,
            "delta_v": component_v,
            "delta": delta,
            "limit_scale": scale,
            "relative": delta.norm(dim=-1) / norm,
        }
