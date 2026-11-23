"""Prompt-calibrated patient rotations and signed selective CAM correction."""
from __future__ import annotations

import math

from common.history import CFG
from common.spectral import (EPS, P, R, MedicalSpectrum, head_jacobian,
                             normalized_weights, small_controller, split_jacobian, unit)

import torch
from torch import nn
from torch.nn import functional as F

ROT = CFG["rotation"]


class PCSRPSCE(nn.Module):
    def __init__(self, original, text, excluded):
        super().__init__()
        self.spectrum = MedicalSpectrum(original, text, excluded, differential=True)
        width = CFG["controllers"]["width"]
        self.spectral_weights = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, 1, bias=False))
        self.direction_controller = small_controller(2 * P["width"] + 11, 1)
        self.amplitude_controller = small_controller(2 * P["width"] + 8, 1, initial_sd=0.)
        self.rotation_controller = small_controller(P["width"] + 16 * 3, len(ROT["planes"]), initial_sd=0.)
        planes = torch.tensor(ROT["planes"], dtype=torch.long).T
        self.register_buffer("plane_axes", planes)
        self.register_buffer("plane_coordinates", F.one_hot(planes, num_classes=16).float())

    def rotate(self, coords, angles):
        x, y = (coords.index_select(1, indices) for indices in self.plane_axes)
        c, s = angles.cos(), angles.sin()
        a, b = self.plane_coordinates.to(coords)
        return coords + (c * x - s * y - x) @ a + (s * x + c * y - y) @ b

    def forward(self, cam, head, training, intervention="full"):
        d = self.spectrum.encode(cam, intervention)
        w_sem = normalized_weights(d["coverage"])
        w_sp = self.spectral_weights(d["descriptors"]).squeeze(-1).softmax(-1)
        n = P["spectral_tokens"]
        entropy = -(w_sem * w_sem.clamp_min(EPS).log()).sum(-1) / math.log(n)
        beta = (1 - d["null_mass"]) * (0.20 + 0.60 * (1 - entropy))
        w = torch.lerp(w_sp, w_sem, beta[:, None])
        attention = d["semantic_attention"]
        contrast = attention[:, 4:].mean(1) - attention[:, :4].mean(1)
        calibrated = (contrast / P["contrast_temperature"]).tanh()
        native_margin, jacobian = head_jacobian(head, cam)
        j_axis, parallel, _ = split_jacobian(jacobian, self.spectrum.basis)
        capacity = R["fraction"] * d["cam_norm"] * parallel.norm(dim=-1)

        white = d["white"].tanh()
        scalars = torch.stack((white, n * w_sem - 1, n * w_sp - 1, n * w - 1, unit(j_axis),
                               native_margin.tanh()[:, None].expand_as(w), beta[:, None].expand_as(w), calibrated), -1)
        query_tokens = d["context"] * d["coverage"].unsqueeze(-1)
        tokens = d["tokens"] * (n * w).unsqueeze(-1)
        inputs = torch.cat((tokens, query_tokens, scalars, d["descriptors"].tanh()), -1)
        action = self.direction_controller(inputs).squeeze(-1).tanh()
        if intervention == "signed_action_off":
            action = action.abs()
        coords = R["spectral_action_multiplier"] * w * action
        prompt = (query_tokens * w.unsqueeze(-1)).sum(1)
        rotation_inputs = torch.cat((white, calibrated, n * w - 1, prompt), -1)
        angles = ROT["angle_cap"] * self.rotation_controller(rotation_inputs).tanh()
        if intervention == "patient_rotation_off":
            angles = torch.zeros_like(angles)
        rotated = self.rotate(coords, angles)

        alignment = (unit(rotated) * unit(j_axis)).sum(-1)
        condition = torch.stack((native_margin.tanh(), native_margin.abs().log1p(), capacity.log1p(),
                                 beta, d["null_mass"], d["coverage"].mean(-1),
                                 calibrated.abs().mean(-1), alignment), -1)
        spectrum_summary = (d["tokens"] * w.unsqueeze(-1)).sum(1)
        amplitude_logit = self.amplitude_controller(torch.cat((spectrum_summary, prompt, condition), -1)).squeeze(-1)
        amplitude = amplitude_logit.tanh()
        raw = (R["fraction"] * d["cam_norm"] * amplitude)[:, None] * (rotated @ self.spectrum.basis.to(cam).T)
        d.update(w_sem=w_sem, w_sp=w_sp, beta=beta, importance=w, entropy=entropy, action=action,
                 prompt_contrast=contrast, calibrated_contrast=calibrated, direction_coordinates=coords,
                 rotated_coordinates=rotated, rotation_angles=angles, amplitude=amplitude, amplitude_logit=amplitude_logit,
                 alignment=alignment,
                 margin=native_margin, jacobian=jacobian, jacobian_axis=j_axis, capacity=capacity, gate=amplitude.abs())
        return self.spectrum.correct(cam, raw, torch.zeros_like(raw), d, intervention)
