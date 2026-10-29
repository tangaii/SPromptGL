"""PCE-Router: Patient-specific Competitive Evidence Routing.

The module operates on each patient's own 256-dimensional CORR/FC slice after
B0-B GFR.  It creates signed ASD and TD evidence, optionally conditions the
route on non-site PHENO/ANAT/FUNC features, and applies a bounded residual only
to the FC slice.  It never constructs or consumes a patient-neighbour graph.
"""
from __future__ import annotations

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class PLNNoPEN(nn.Module):
    """Exact PEN-free CAM path used by the verified B0-B anchors."""
    def __init__(self, dims, hp, layers_module):
        super().__init__()
        self.hyperpm = hp
        self.input_data_dims = dims
        self.d_q = self.d_k = self.d_v = hp.n_hidden
        self.n_head = hp.n_head
        self.dropout = hp.dropout
        self.n_layer = hp.nlayer
        self.modal_num = hp.nmodal
        self.n_class = hp.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
        self.InputLayer = layers_module.VariLengthInputLayer(
            dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        for index in range(self.n_layer):
            encoder = layers_module.EncodeLayer(
                self.d_k * self.n_head, self.d_k, self.d_v,
                self.n_head, self.dropout)
            self.add_module(f"encode_{index}", encoder)
            self.Encoder.append(encoder)
            feedforward = layers_module.FeedForwardLayer(
                self.d_v * self.n_head, self.d_v * self.n_head,
                dropout=self.dropout)
            self.add_module(f"feed_{index}", feedforward)
            self.FeedForward.append(feedforward)
        self.Outputlayer = layers_module.OutputLayer(
            self.d_v * self.n_head * self.modal_num,
            self.d_v * self.n_head, self.n_class,
            self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, input_attention = self.InputLayer(x)
        attn = input_attention.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        for index in range(self.n_layer):
            x, layer_attention = self.Encoder[index](
                q=x, k=x, v=x, modal_num=self.modal_num)
            attn = layer_attention.mean(dim=1)
            x = self.FeedForward[index](x)
            attn_map.append(attn.detach().cpu().numpy())
        output, hidden = self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))
        return output, hidden, attn_map


class PCERouter(nn.Module):
    PHENO = 0
    ANAT = 1
    FUNC = 2
    FC = 3
    CONDITION_DIM = 44
    FC_DIM = 256
    MODES = {
        "E1_SELF",
        "E2_CLASS_AGNOSTIC",
        "E3_COMPETITIVE_NO_CONDITION",
        "E4_CONDITION_NO_COMPETITION",
        "E5_FULL",
        "E6_WRONG_CONDITION",
        "E7_WRONG_EVIDENCE_POSITION",
        "E8_NO_EVIDENCE_LOSS",
    }

    def __init__(self, hidden: int = 32, tau_e: float = 1.0,
                 beta: float = 0.25, residual_max: float = 0.05):
        super().__init__()
        self.tau_e = float(tau_e)
        self.beta = float(beta)
        self.residual_max = float(residual_max)
        self.fc_norm = nn.LayerNorm(self.FC_DIM)
        self.shared = nn.Sequential(
            nn.Linear(self.FC_DIM, hidden), nn.GELU())
        # Two signed class-specific evidence outputs.  tanh is applied in
        # forward, so evidence direction remains meaningful and bounded.
        self.class_heads = nn.ModuleList([
            nn.Linear(hidden, self.FC_DIM),
            nn.Linear(hidden, self.FC_DIM),
        ])
        # Used by E1/E2/E4.  Keeping it registered in every mode makes the
        # declared module comparable; diagnostics report active gradients.
        self.generic_head = nn.Linear(hidden, self.FC_DIM)
        # Age(26), sex(2), ANAT(6), FUNC(10); site one-hot is excluded.
        self.condition = nn.Sequential(
            nn.Linear(self.CONDITION_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, self.FC_DIM))
        self.accept = nn.Sequential(
            nn.Linear(self.CONDITION_DIM, 16), nn.GELU(), nn.Linear(16, 1))
        # Exactly zero at initialization, so the initial PCE path is an
        # identity map for the GFR/CAM representation.
        self.alpha_raw = nn.Parameter(torch.zeros(()))

    @staticmethod
    def condition_features(x: torch.Tensor) -> torch.Tensor:
        # PHENO: age 0:26, site 26:46, sex 46:48.  Site is deliberately
        # excluded from the new conditional route; raw B0-B still sees it.
        return torch.cat([x[:, 0:26], x[:, 46:48], x[:, 48:64]], dim=1)

    def _class_evidence(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.tanh(head(z)) for head in self.class_heads], dim=1)

    def _generic_evidence(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.generic_head(z))

    def _route_condition(self, x: torch.Tensor, mode: str,
                         condition_perm: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if mode in {"E1_SELF", "E2_CLASS_AGNOSTIC", "E3_COMPETITIVE_NO_CONDITION"}:
            return torch.zeros_like(x[:, 64:320]), torch.ones((x.shape[0], 1), device=x.device)
        c = self.condition_features(x)
        if mode == "E6_WRONG_CONDITION" and condition_perm is not None:
            c = c[condition_perm]
        q = torch.tanh(self.condition(c))
        gamma = torch.sigmoid(self.accept(c))
        return q, gamma

    @staticmethod
    def _apply_position_permutation(value: torch.Tensor,
                                    evidence_perm: Optional[torch.Tensor]) -> torch.Tensor:
        if evidence_perm is None:
            return value
        return value.gather(1, evidence_perm)

    def forward(self, h_gfr: torch.Tensor, raw_x: torch.Tensor, mode: str,
                condition_perm: Optional[torch.Tensor] = None,
                evidence_perm: Optional[torch.Tensor] = None,
                force_alpha_zero: bool = False,
                force_uniform_mask: bool = False,
                return_details: bool = True):
        if mode not in self.MODES:
            raise ValueError(f"unknown PCE mode: {mode}")
        f = h_gfr[:, 64:320]
        z = self.fc_norm(f)
        hidden = self.shared(z)
        class_evidence = self._class_evidence(hidden)
        generic_evidence = self._generic_evidence(hidden)
        q, gamma = self._route_condition(raw_x, mode, condition_perm)

        competitive = mode in {"E3_COMPETITIVE_NO_CONDITION", "E5_FULL",
                               "E6_WRONG_CONDITION", "E7_WRONG_EVIDENCE_POSITION",
                               "E8_NO_EVIDENCE_LOSS"}
        if competitive:
            evidence = class_evidence
            difference = evidence[:, 0] - evidence[:, 1]
        else:
            evidence = generic_evidence.unsqueeze(1)
            difference = generic_evidence
        if mode == "E1_SELF":
            # Capacity control: a direct patient-local signed residual from
            # the FC encoder.  It deliberately has no feature-selection
            # softmax, category competition, or conditional route.
            mask = torch.full_like(difference, 1.0 / self.FC_DIM)
            utility = difference.tanh()
            route_mask = utility
        else:
            mask_logits = difference.abs() / self.tau_e + self.beta * q
            mask = F.softmax(mask_logits, dim=1)
            if force_uniform_mask:
                mask = torch.full_like(mask, 1.0 / self.FC_DIM)
            # Relative to a uniform 256-way route.  This keeps small evidence
            # deviations useful without allowing an unbounded residual.
            utility = torch.tanh(self.FC_DIM * mask - 1.0)
            route_mask = self._apply_position_permutation(utility, evidence_perm)
        alpha = torch.zeros((), device=f.device) if force_alpha_zero else self.residual_max * torch.tanh(self.alpha_raw)
        scale = gamma * alpha * route_mask
        f_new = f * (1.0 + scale)
        h_new = torch.cat([h_gfr[:, :64], f_new], dim=1)

        if competitive:
            score_mask = self._apply_position_permutation(mask, evidence_perm)
            scores = (score_mask.unsqueeze(1) * evidence).sum(dim=2)
        else:
            scores = (self._apply_position_permutation(mask, evidence_perm) * evidence[:, 0]).sum(dim=1, keepdim=True)
        details = {
            "evidence": evidence,
            "difference": difference,
            "mask": mask,
            "route_mask": route_mask,
            "utility": utility,
            "q": q,
            "gamma": gamma.squeeze(1),
            "alpha": alpha,
            "scale": scale,
            "scores": scores,
            "competitive": competitive,
            "h_new": h_new,
        }
        return h_new, details

    def evidence_loss(self, details: dict, labels: torch.Tensor,
                      train_idx: torch.Tensor) -> torch.Tensor:
        if not details["competitive"]:
            return labels.new_zeros((), dtype=torch.float32)
        scores = details["scores"]
        # Labels use ASD=0, TD=1 and the two evidence logits have the same
        # order.  The loss is restricted to outer-train patients.
        return F.cross_entropy(scores[train_idx], labels[train_idx])

    def active_parameter_count(self, details: dict) -> tuple[int, int]:
        # Parameters with a nonzero gradient are reported by the runner after
        # backward; this helper gives the registered count for audit tables.
        return sum(p.numel() for p in self.parameters()), sum(
            p.numel() for p in self.parameters() if p.requires_grad)
