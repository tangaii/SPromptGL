"""Reliability-calibrated, direction-wise medical spectral corrections."""
from __future__ import annotations

import torch
from torch import nn

from common.spectral import DIM, DiseaseSpectralPrompt


class RcsDaSrpe(DiseaseSpectralPrompt):
    def __init__(self, spectrum, text, reference, exclusions, cfg, method):
        super().__init__(spectrum, text, reference, exclusions, cfg, method)
        self.direction_control = nn.Sequential(nn.Linear(36, 20), nn.GELU(), nn.Linear(20, 2))
        nn.init.normal_(self.direction_control[-1].weight, std=0.02)
        nn.init.zeros_(self.direction_control[-1].bias)
        self.bias = nn.Parameter(torch.linspace(*method["controller_bias_range"], DIM))

    def forward(self, cam, *, crossfit, query=None, semantic_off=False, variant=None):
        z, scale, details = self.inputs(cam, crossfit, query, semantic_off, variant == "statistics_off")
        mixture, contrast = details["mixture"], details["contrast"]
        reliability = details["reliability"][:, None].expand_as(z)
        descriptors = torch.stack(((z / scale).tanh(), mixture, contrast, reliability), -1)
        updates = self.direction_control(torch.cat((details["tokens"], descriptors), -1))
        control = self.method["controller_scale"] * (self.bias + updates[:, :, 0]).tanh()
        h = scale * (mixture + self.method["likelihood_contrast_weight"] * contrast
                     + self.method["token_update_scale"] * updates[:, :, 1]).tanh()
        delta_z = control * details["selection"] * h
        return self.finish(cam, z, delta_z, {**details, "control": control, "h": h})
