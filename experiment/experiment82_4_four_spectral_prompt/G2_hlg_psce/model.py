"""Local-to-global latent CAM evidence constrained to the original spectrum."""
from __future__ import annotations

import math

import torch
from torch import nn

from common.spectral import DIM, EPS, SpectralPrompt


class HlgPsce(SpectralPrompt):
    def __init__(self, spectrum, text, _reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg)
        self.method = method
        self.register_buffer("slot_loadings", self.basis.reshape(16, 9, DIM).square().sum(1))
        self.spectral = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.local = nn.Sequential(nn.Linear(19, 24), nn.GELU(), nn.Linear(24, 9))
        self.local_mix = nn.Linear(11, 1)
        nn.init.constant_(self.local_mix.bias, method["initial_local_mix_logit"])

    def forward(self, cam, *, crossfit, query=None, semantic_off=False, variant=None):
        z = self.project(cam)
        scale, details = self.base_tokens(z, self.query if query is None else query, semantic_off)
        weights = details["weights"]
        scores = self.spectral(torch.stack(((z / scale).abs(), self.logvar.expand_as(z), self.nu.expand_as(z)), -1))
        spectral = scores.squeeze(-1).softmax(-1)
        entropy = -(weights * (weights + EPS).log()).sum(-1) / math.log(DIM)
        null = details["attention"][:, :, DIM].mean(-1)
        psce_mix = ((1 - null) * (1 - entropy).clamp(0.2, 0.8)).detach()
        selected = spectral if semantic_off else torch.lerp(spectral, weights, psce_mix[:, None])
        global_z = (selected - 1 / DIM) * z

        slots = cam.reshape(len(cam), 16, 9)
        context = slots.mean(1, keepdim=True)
        deviations = slots - context
        affinity = DIM * (weights @ self.slot_loadings.T)
        local_features = torch.cat((deviations, context.expand_as(deviations), affinity.unsqueeze(-1)), -1)
        local_gain = self.local(local_features).tanh()
        local_raw = (self.method["local_scale"] * affinity.unsqueeze(-1) * local_gain * deviations).reshape_as(cam)
        local_z = local_raw @ self.basis

        deviation_energy = deviations.square().mean(-1)
        mix_inputs = torch.cat((context.squeeze(1), deviation_energy.mean(-1, keepdim=True),
                               (deviation_energy * affinity).mean(-1, keepdim=True)), -1)
        floor = self.method["local_mix_floor"]
        beta = floor + (1 - 2 * floor) * self.local_mix(mix_inputs).sigmoid()
        if variant == "global_only":
            beta = torch.zeros_like(beta)
        delta_z = torch.lerp(global_z, local_z, beta)
        return self.finish(cam, z, delta_z, {
            **details, "affinity": affinity, "beta": beta, "psce_mix": psce_mix,
            "global_z": global_z, "local_z": local_z, "local_raw": local_raw,
            "spectral_weights": spectral,
        })
