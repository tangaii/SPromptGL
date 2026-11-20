"""Medical semantics control a rank-four operator in the Original O coordinates."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from common.spectral import DIM, DiseaseSpectralPrompt


class SotDaSrpe(DiseaseSpectralPrompt):
    def __init__(self, spectrum, text, reference, exclusions, cfg, method):
        super().__init__(spectrum, text, reference, exclusions, cfg, method)
        rank = method["rank"]
        rows = torch.arange(DIM, dtype=torch.float64) + 0.5
        columns = torch.arange(DIM, dtype=torch.float64)
        chart = math.sqrt(2 / DIM) * (math.pi * rows[:, None] * columns / DIM).cos()
        chart[:, 0] /= math.sqrt(2)
        self.register_buffer("chart", chart.float())
        self.free = nn.Parameter(torch.zeros(DIM - rank, rank))
        self.bias = nn.Parameter(torch.linspace(*method["operator_bias_range"], rank))
        self.operator_control = nn.Sequential(nn.Linear(35, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.normal_(self.operator_control[-1].weight, std=0.02)
        nn.init.zeros_(self.operator_control[-1].bias)

    def orthogonal_factors(self):
        rank = self.method["rank"]
        chart = self.chart[:, :rank] + self.chart[:, rank:] @ (
            self.method["basis_chart_scale"] * self.free.tanh())
        columns = []
        for column in chart.unbind(dim=-1):
            # The fixed identity block gives sigma_min(chart) >= 1.
            if columns:
                previous = torch.stack(columns, -1)
                for _ in range(2):
                    column = column - previous @ (previous.T @ column)
            columns.append(F.normalize(column, dim=0))
        return torch.stack(columns, -1)

    def forward(self, cam, *, crossfit, query=None, semantic_off=False, variant=None):
        z, scale, details = self.inputs(cam, crossfit, query, semantic_off)
        factors = self.orthogonal_factors()
        loadings = factors.square()
        candidate = scale * (details["mixture"] + self.method["likelihood_contrast_weight"] * details["contrast"])
        source = candidate @ factors
        coefficients = details["selection"] @ loadings
        tokens = loadings.T @ details["tokens"]
        summaries = torch.stack((source.tanh(), coefficients, details["reliability"][:, None].expand_as(source)), -1)
        logits = self.operator_control(torch.cat((tokens, summaries), -1)).squeeze(-1)
        gain = self.method["operator_scale"] * (self.bias + logits).tanh()
        weights = coefficients * gain
        if variant == "diagonal_operator":
            delta_z = candidate * (weights @ loadings.T)
        else:
            delta_z = (weights * source) @ factors.T
        return self.finish(cam, z, delta_z, {
            **details, "factors": factors, "initial_factors": self.chart[:, :self.method["rank"]],
            "operator_weights": weights, "semantic_coefficients": coefficients,
            "candidate": candidate, "operator_source": source, "control": gain,
        })
