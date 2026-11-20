"""Differentiated medical queries and one patient-conditioned Cayley plane."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from common.spectral import DIM, EPS, SpectralPrompt


class DprPsce(SpectralPrompt):
    def __init__(self, spectrum, text, _reference, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg)
        self.method = method
        self.register_buffer("eye", torch.eye(DIM))
        self.register_buffer("spectral_covariance",
                             (spectrum["U"].T @ spectrum["covariance"] @ spectrum["U"]).float())
        self.angle = nn.Linear(2 * DIM, 1)
        nn.init.zeros_(self.angle.weight)
        nn.init.zeros_(self.angle.bias)
        self.token_update = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def rotation(self, z_white, query, disabled):
        # The fixed first two axes anchor the plane. The second unit vector
        # can tilt into any of the remaining 14 directions for each patient.
        projected_anchor = F.normalize(self.key(self.direction(self.basis.T)), dim=-1)
        text_alignment = projected_anchor @ query.mean(0)
        aligned = (z_white * (1 + text_alignment)).tanh()
        t = self.method["angle_bound"] * self.angle(torch.cat((z_white.tanh(), aligned), -1)).tanh()
        if disabled:
            t = torch.zeros_like(t)
        p = self.eye[0].expand_as(aligned)
        q = F.normalize(torch.cat((self.eye[1, :2].expand(len(aligned), -1),
                                   self.method["plane_variation"] * aligned[:, 2:]), -1), dim=-1)
        pq = p.unsqueeze(-1) * q.unsqueeze(-2)
        skew = pq - pq.transpose(-2, -1)
        projector = p.unsqueeze(-1) * p.unsqueeze(-2) + q.unsqueeze(-1) * q.unsqueeze(-2)
        t = t.unsqueeze(-1)
        # Closed form of (I + S)(I - S)^(-1) for S = t(pq^T - qp^T).
        return self.eye + (2 * t * skew - 2 * t.square() * projector) / (1 + t.square())

    def forward(self, cam, *, crossfit, query=None, semantic_off=False, variant=None):
        z = self.project(cam)
        q0 = self.query if query is None else query
        shrinkage = 0.0 if variant == "common_kept" else self.method["query_common_shrinkage"]
        query = F.normalize(q0 - shrinkage * q0.mean(0), dim=-1)
        rotation_query = torch.zeros_like(q0) if semantic_off else q0
        rotation = self.rotation(z / (self.variance + EPS).sqrt(), rotation_query, variant == "rotation_off")
        coordinates = (rotation @ z.unsqueeze(-1)).squeeze(-1)
        directions = rotation @ self.basis.T
        diagonal = ((rotation @ self.spectral_covariance) * rotation).sum(-1)
        scale = (diagonal.clamp_min(0) + EPS).sqrt()
        squared = rotation.square()
        details = self.attention(coordinates / scale, directions, squared @ self.logvar, squared @ self.nu,
                                 query, semantic_off)
        h = coordinates + self.method["token_update_scale"] * scale * self.token_update(details["tokens"]).squeeze(-1).tanh()
        rotated_delta = (details["weights"] - 1 / DIM) * h
        delta_z = (rotation.transpose(-2, -1) @ rotated_delta.unsqueeze(-1)).squeeze(-1)
        return self.finish(cam, z, delta_z, {
            **details, "rotation": rotation, "coordinates": coordinates, "directions": directions,
            "rotated_delta": rotated_delta, "h": h, "effective_queries": query,
        })
