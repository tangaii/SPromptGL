"""FAVE-PEN evidence matching and bounded CORR-position correction."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FAVEPEN(nn.Module):
    """One verified directional hypothesis plus a differentiable reject slot.

    Text row 0 is the literature-supported increase hypothesis; row 1 is a
    direction-reversed counterfactual used only by the verification objective.
    Only the first hypothesis may receive a match assignment.
    """

    def __init__(self, text_embeddings: torch.Tensor, init_seed: int):
        super().__init__()
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 2:
            raise ValueError("expected [supported, counterfactual] text embeddings")
        self.register_buffer("text_embeddings", text_embeddings.detach().float().clone())
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(init_seed))
            if devices:
                torch.cuda.manual_seed_all(int(init_seed))
            self.connection_encoder = nn.Sequential(
                nn.Linear(2, 32), nn.GELU(), nn.LayerNorm(32),
            )
            self.semantic_projection = nn.Linear(text_embeddings.shape[-1], 32, bias=False)
            self.evidence_projection = nn.Linear(32, 36, bias=False)
            self.quality_gate = nn.Sequential(
                nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1),
            )
            self.alpha = nn.Parameter(torch.zeros(()))
            self.reject_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        z: torch.Tensor,
        quality_input: torch.Tensor,
        h_corr: torch.Tensor,
        train_idx: torch.Tensor,
        text_override: torch.Tensor | None = None,
        z_override: torch.Tensor | None = None,
    ):
        """Return corrected CAM position, train-only auxiliary losses and audit data."""
        z_used = z if z_override is None else z_override
        n = z_used.shape[0]
        text = self.text_embeddings if text_override is None else text_override
        text_vec = F.normalize(self.semantic_projection(text), dim=-1, eps=1e-8)

        z0 = z_used[:, 0]
        conn_input = torch.stack((z0, z0.abs()), dim=-1)
        conn_vec = F.normalize(self.connection_encoder(conn_input), dim=-1, eps=1e-8)
        support_score = (conn_vec * text_vec[0]).sum(dim=-1, keepdim=True) / 0.1
        # The only score supplied here is the sole anatomically eligible
        # feature/prompt pair; every other ROI-feature correspondence is absent.
        reject = self.reject_logit.expand(n, 1)
        assignment = torch.softmax(torch.cat((support_score, reject), dim=1), dim=1)

        state = torch.where(z0 >= 2.3, 1, torch.where(z0 <= -2.3, -1, 0))
        matched_mass = assignment[:, 0]
        evidence_vector = matched_mass[:, None] * torch.tanh(z0[:, None] / 2.3) * text_vec[0][None, :]
        evidence = self.evidence_projection(evidence_vector)
        quality = torch.sigmoid(self.quality_gate(quality_input)).squeeze(-1)
        rms = h_corr.square().mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        residual = 0.1 * rms * torch.tanh(self.alpha * quality[:, None] * evidence / rms)

        # Supported observations match the one verified hypothesis. Contradictory
        # and unknown observations are explicitly rejected; their states remain
        # separately reported and are not mislabeled as disease evidence.
        train_idx = train_idx.long()
        train_state = state[train_idx]
        train_assignment = assignment[train_idx]
        target = (train_state != 1).long()
        nll = -torch.log(train_assignment.gather(1, target[:, None]).clamp_min(1e-8)).squeeze(1)
        strata_losses = [nll[train_state == v].mean() for v in (1, -1, 0) if torch.any(train_state == v)]
        loss_match = torch.stack(strata_losses).mean() if strata_losses else z.new_zeros(())

        # The reverse-direction prompt is counterfactual only. It is never an
        # admissible disease match. The interaction should follow the observed
        # sign of the train-derived normative deviation.
        zero_input = torch.zeros((n, 2), dtype=z.dtype, device=z.device)
        zero_vec = F.normalize(self.connection_encoder(zero_input), dim=-1, eps=1e-8)
        score_support = (conn_vec * text_vec[0]).sum(dim=-1)
        score_support_ref = (zero_vec * text_vec[0]).sum(dim=-1)
        score_reverse = (conn_vec * text_vec[1]).sum(dim=-1)
        score_reverse_ref = (zero_vec * text_vec[1]).sum(dim=-1)
        interaction = (score_support - score_support_ref) - (score_reverse - score_reverse_ref)
        train_extreme = train_state != 0
        if train_extreme.any():
            selected_idx = train_idx[train_extreme]
            signed_interaction = state[selected_idx].to(z.dtype) * interaction[selected_idx]
            verify_weight = (z0[selected_idx].abs() / 3.0).clamp(0.0, 1.0)
            loss_verify = (F.relu(0.05 - signed_interaction) * verify_weight).mean()
        else:
            loss_verify = z.new_zeros(())

        info = {
            "assignment": assignment,
            "quality": quality,
            "residual": residual,
            "evidence": evidence,
            "interaction": interaction,
            "state": state,
            "support": state == 1,
            "contradiction": state == -1,
            "unknown": state == 0,
            "reject_rate": (assignment[:, 1] >= assignment[:, 0]).float().mean(),
            "train_support_count": (train_state == 1).sum(),
            "train_contradiction_count": (train_state == -1).sum(),
            "train_unknown_count": (train_state == 0).sum(),
            "train_match_assignment": train_assignment,
        }
        return h_corr + residual, loss_match, loss_verify, info


def optimizer_for_fave(module: nn.Module, lr: float = 0.001):
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": 1e-4})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=lr)
