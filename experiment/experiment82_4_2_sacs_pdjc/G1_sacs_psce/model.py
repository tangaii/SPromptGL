"""SACS-PSCE with a fixed Original O anchor and a dynamic classifier subspace."""
from __future__ import annotations

import torch
from torch import nn

from common.geometry import update_subspace
from common.spectral import PromptCorrection


class SacsPsce(PromptCorrection):
    def __init__(self, spectrum, text, v_initial, exclusions, cfg, method):
        super().__init__(spectrum, text, exclusions, cfg)
        self.method = method
        self.register_buffer("basis_v", v_initial["basis"].double().clone())
        self.register_buffer("eigenvalues_v", v_initial["eigenvalues"].double().clone())
        self.register_buffer("variance_v", v_initial["variance"].double().clone())
        self.register_buffer("v_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("next_v_update_epoch", torch.tensor(method["subspace_update_interval"] + cfg["start_epoch"]))

        self.v_projection = nn.Linear(2, 32)
        self.v_position = nn.Parameter(torch.empty(8, 32))
        self.u_action = nn.Sequential(nn.Linear(67, method["signed_width"]), nn.GELU(), nn.Linear(method["signed_width"], 1))
        self.v_action = nn.Sequential(nn.Linear(67, method["signed_width"]), nn.GELU(), nn.Linear(method["signed_width"], 1))
        self.fusion = nn.Sequential(nn.Linear(130, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.normal_(self.v_position, std=0.02)
        nn.init.normal_(self.u_action[-1].weight, std=method["initial_action_weight_sd"])
        nn.init.normal_(self.v_action[-1].weight, std=method["initial_action_weight_sd"])
        nn.init.zeros_(self.u_action[-1].bias)
        nn.init.zeros_(self.v_action[-1].bias)
        nn.init.normal_(self.fusion[-1].weight, std=0.025)
        nn.init.constant_(self.fusion[-1].bias, method["initial_strength_bias"])

    def gradient_groups(self):
        prompt = [self.temperature_logit, self.null, *self.u_projection.parameters(), self.u_position,
                  self.key.weight, *self.v_projection.parameters(), self.v_position]
        residual = [*self.u_action.parameters(), *self.v_action.parameters(), *self.fusion.parameters()]
        return {"prompt": prompt, "residual": residual}

    @torch.no_grad()
    def update_v(self, candidate, eigenvalues, epoch):
        if int(self.next_v_update_epoch) != epoch:
            raise RuntimeError(f"V8 update expected at epoch {int(self.next_v_update_epoch)}, got {epoch}")
        updated = update_subspace(self.basis_v, candidate, self.basis, self.method["subspace_update_rate"])
        self.basis_v.copy_(updated.to(device=self.basis_v.device, dtype=self.basis_v.dtype))
        self.eigenvalues_v.copy_(eigenvalues.to(device=self.eigenvalues_v.device, dtype=self.eigenvalues_v.dtype))
        self.v_updates.add_(1)
        self.next_v_update_epoch.add_(self.method["subspace_update_interval"])

    def forward(self, cam, *, head, training, query=None, semantic_off=False, variant=None):
        details = self.prepare(cam, query=query, semantic_off=semantic_off)
        branch_cam = details["branch_cam"]
        centered = branch_cam - self.center.to(branch_cam)
        coordinates_v = centered @ self.basis_v.to(branch_cam)
        white_v = coordinates_v / (self.variance_v.to(branch_cam).sqrt() + 1e-8)
        tokens_v = self.v_projection(torch.stack((coordinates_v, white_v), dim=-1)) + self.v_position.to(branch_cam)
        all_tokens = torch.cat((details["tokens_u"], tokens_v), dim=1)
        selected_query = self.query_medical if query is None else query
        prompt = self.attend(all_tokens, selected_query, semantic_off)
        context_u, context_v = prompt["context"][:, :16], prompt["context"][:, 16:]
        coverage_u, coverage_v = prompt["coverage"][:, :16], prompt["coverage"][:, 16:]
        white_u = details["white_u"]
        u_importance = coverage_u / PromptCorrection.neutral_coverage(16)
        v_importance = coverage_v / PromptCorrection.neutral_coverage(8)
        u_features = torch.cat((details["tokens_u"], context_u,
                                torch.stack((white_u, white_u.abs(), coverage_u), dim=-1)), dim=-1)
        v_features = torch.cat((tokens_v, context_v,
                                torch.stack((white_v, white_v.abs(), coverage_v), dim=-1)), dim=-1)
        action_u = torch.tanh(self.u_action(u_features).squeeze(-1)) * u_importance
        action_v = torch.tanh(self.v_action(v_features).squeeze(-1)) * v_importance

        summary = torch.cat((details["tokens_u"].mean(1), tokens_v.mean(1), context_u.mean(1),
                             context_v.mean(1), white_u.abs().mean(-1, keepdim=True),
                             white_v.abs().mean(-1, keepdim=True)), dim=-1)
        beta = self.fusion(summary).sigmoid()
        if variant == "complement_off":
            action_v = torch.zeros_like(action_v)
        scale = self.branch_scale * branch_cam.norm(dim=-1, keepdim=True)
        raw_u = ((1 - beta) * scale / (16 ** 0.5)) * action_u
        raw_v = (beta * scale / (8 ** 0.5)) * action_v @ self.basis_v.to(cam).T
        raw_u = raw_u @ self.basis.to(cam).T

        entropy = -(prompt["attention"] * prompt["attention"].clamp_min(1e-8).log()).sum(-1)
        details.update({
            "coordinates_v": coordinates_v,
            "white_v": white_v,
            "tokens_v": tokens_v,
            "coverage": prompt["coverage"],
            "context": prompt["context"],
            "attention": prompt["attention"],
            "attention_entropy": entropy / torch.log(torch.tensor(prompt["attention"].shape[-1], device=cam.device)),
            "null_weight": prompt["attention"][..., -1],
            "coverage_u": coverage_u,
            "coverage_v": coverage_v,
            "context_u": context_u,
            "context_v": context_v,
            "action_u": action_u,
            "action_v": action_v,
            "beta": beta,
            "temperature": prompt["temperature"],
        })
        return self.finish(cam, raw_u, raw_v, details, residual_off=variant == "residual_off")
