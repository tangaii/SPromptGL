"""Jacobian-guided Spectral Learning without any DSE inputs or state."""
from __future__ import annotations

import torch
from torch import nn

from history_base.common.spectral import fixed_logits, head_jacobian, split_jacobian


class JSL(nn.Module):
    def __init__(self, spectrum, excluded):
        super().__init__()
        self.register_buffer("basis", spectrum["U"].double().clone())
        self.register_buffer("center", spectrum["mu"].double().clone())
        self.register_buffer("variance", spectrum["lambda"].double().clone())
        allowed = torch.ones(871, dtype=torch.bool)
        allowed[excluded] = False
        self.register_buffer("allowed", allowed)
        width = 32
        self.token_encoder = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.LayerNorm(width))
        self.token_score = nn.Sequential(nn.Linear(width, 16), nn.GELU(), nn.Linear(16, 1))
        self.controller = nn.Sequential(nn.Linear(width + 6, 24), nn.GELU(), nn.Linear(24, 3))
        last = self.controller[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.weight[0].zero_()
        self.register_buffer("ridge", torch.tensor(1e-4))
        self.register_buffer("cam_fraction", torch.tensor(0.08))

    def forward(self, cam, head, intervention="full"):
        basis = self.basis.to(cam)
        center = self.center.to(cam)
        variance = self.variance.to(cam)
        z = (cam - center) @ basis
        white = z / (variance + 1e-8).sqrt()
        token_input = torch.stack((white, white.abs(), torch.log1p(white.square())), -1)
        tokens = self.token_encoder(token_input)
        weights = self.token_score(tokens).squeeze(-1).softmax(-1)
        summary = (tokens * weights.unsqueeze(-1)).sum(1)

        native_margin, jacobian = head_jacobian(head, cam)
        _, parallel, perp = split_jacobian(jacobian, basis)
        jp = parallel.norm(dim=-1)
        jq = perp.norm(dim=-1)
        scalars = torch.stack((native_margin.tanh(), native_margin.abs().log1p(),
                               (jp * self.cam_fraction * cam.norm(dim=-1)).log1p(),
                               (jq * self.cam_fraction * cam.norm(dim=-1)).log1p(),
                               white.mean(-1), white.abs().mean(-1)), -1)
        logits = self.controller(torch.cat((summary, scalars), -1))
        shift = 0.75 * logits[:, 0].tanh()
        trust = logits[:, 1].sigmoid()
        allocation = logits[:, 2].sigmoid()
        desired = trust * shift
        strength_parallel = desired * allocation
        strength_perp = desired * (1 - allocation)
        jp2 = parallel.square().sum(-1)
        jq2 = perp.square().sum(-1)
        raw_parallel = strength_parallel[:, None] * parallel / (jp2[:, None] + self.ridge)
        raw_perp = strength_perp[:, None] * perp / (jq2[:, None] + self.ridge)
        if intervention == "spectral_only":
            raw_perp = torch.zeros_like(raw_perp)
        if intervention == "complement_only":
            raw_parallel = torch.zeros_like(raw_parallel)
        if intervention == "residual_off":
            raw_parallel = torch.zeros_like(raw_parallel)
            raw_perp = torch.zeros_like(raw_perp)
        allowed = self.allowed.to(device=cam.device)[:, None]
        raw_parallel = raw_parallel.masked_fill(~allowed, 0)
        raw_perp = raw_perp.masked_fill(~allowed, 0)
        raw = raw_parallel + raw_perp
        cam_norm = cam.norm(dim=-1)
        cap = float(self.cam_fraction) * cam_norm
        ratio = raw.norm(dim=-1) / cap.clamp_min(1e-8)
        scale = torch.tanh(ratio) / ratio.clamp_min(1e-8)
        scale = torch.where(ratio < 1e-6, torch.ones_like(scale), scale)
        delta_parallel = raw_parallel * scale[:, None]
        delta_perp = raw_perp * scale[:, None]
        delta = delta_parallel + delta_perp
        enhanced = cam + delta
        return enhanced, {
            "white": white, "tokens": tokens, "weights": weights, "summary": summary,
            "margin": native_margin, "jacobian": jacobian, "parallel": parallel, "perp": perp,
            "raw_parallel": raw_parallel, "raw_perp": raw_perp, "delta_parallel": delta_parallel,
            "delta_perp": delta_perp, "delta": delta,
            "shift": shift, "trust": trust, "allocation": allocation,
            "ratio": delta.norm(dim=-1) / cam.norm(dim=-1).clamp_min(1e-8),
            "cam_norm": cam.norm(dim=-1), "enhanced": enhanced,
        }

