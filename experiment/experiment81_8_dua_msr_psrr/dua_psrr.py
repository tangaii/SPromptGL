"""Diagnostic utility-aligned extension of the audited node-level TP-PSRR path."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tp_psrr import TPPSRR, entmax15, prompt_groups


class DUAPSRR(TPPSRR):
    def __init__(self, semantic_basis, rho=0.1, relation_scale=1.0, temperature=1.0):
        super().__init__(semantic_basis, rho, relation_scale, temperature, selection_mode="node")
        self.utility_relation = nn.Sequential(
            nn.Linear(18, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        )
        nn.init.zeros_(self.utility_relation[-1].weight)
        nn.init.zeros_(self.utility_relation[-1].bias)

    def _relations(self, tokens, groups, intervention, return_diagnostics, relation_features=None):
        _, base = super()._relations(
            tokens, groups, intervention, True, relation_features=relation_features
        )
        if relation_features is None or tuple(relation_features.shape[1:]) != (16, 9):
            raise ValueError("DUA-PSRR requires the original CAM144 slots with shape [N,16,9]")

        receivers = relation_features.unsqueeze(2).expand(-1, -1, 15, -1)
        senders = relation_features[:, self.neighbors]
        h = self.utility_relation(torch.cat((receivers, senders - receivers), dim=-1))
        beta_td = base["beta_td"]
        beta_asd = base["beta_asd"]
        prompt_td, prompt_asd = prompt_groups("medical")
        p_td = torch.einsum("nak,kd->nad", beta_td, self.semantic_center[list(prompt_td)])
        p_asd = torch.einsum("nak,kd->nad", beta_asd, self.semantic_center[list(prompt_asd)])
        delta_td = 0.5 * torch.tanh((h * p_td.unsqueeze(2)).sum(dim=-1))
        delta_asd = 0.5 * torch.tanh((h * p_asd.unsqueeze(2)).sum(dim=-1))
        if intervention in ("prompt_off", "dua_score_off"):
            delta_td = torch.zeros_like(delta_td)
            delta_asd = torch.zeros_like(delta_asd)

        scores_td = torch.cat((base["scores_td"][..., :15] + delta_td, base["scores_td"][..., 15:]), dim=-1)
        scores_asd = torch.cat((base["scores_asd"][..., :15] + delta_asd, base["scores_asd"][..., 15:]), dim=-1)
        if intervention in ("prompt_off", "relation_off"):
            attention_td = base["attention_visual"]
            attention_asd = base["attention_visual"]
        else:
            attention_td = entmax15(scores_td)
            attention_asd = entmax15(scores_asd)
        attention_td, attention_asd = self.apply_audit_edge_overrides(attention_td, attention_asd)

        neighbor_ids = self.neighbors.to(device=tokens.device)
        senders_encoded = tokens[:, neighbor_ids]
        receivers_encoded = tokens.unsqueeze(2).expand_as(senders_encoded)
        product = receivers_encoded * senders_encoded
        difference = senders_encoded - receivers_encoded
        value_input = torch.cat((senders_encoded, product, difference), dim=-1)
        values = self.value(value_input)
        contrast = (
            (attention_asd[..., :15] - attention_td[..., :15]).unsqueeze(-1) * values
        ).sum(dim=2)
        if intervention == "relation_off":
            contrast = torch.zeros_like(contrast)

        details = {
            **base,
            "score_base_td": base["scores_td"],
            "score_base_asd": base["scores_asd"],
            "scores_td": scores_td,
            "scores_asd": scores_asd,
            "dua_delta_td": delta_td,
            "dua_delta_asd": delta_asd,
            "attention_td": attention_td,
            "attention_asd": attention_asd,
            "relation_support_difference": (attention_td[..., :15] > 0) ^ (attention_asd[..., :15] > 0),
            "message_contrast": contrast,
        }
        return contrast, details

    def utility_loss(self, details, z_cam, z_final, logp, labels, legal_rows, class_to_index,
                     temperature=0.20, edge_weight=1.0, null_weight=0.25):
        if logp.ndim != 2 or logp.shape[-1] != 2:
            raise ValueError("DUA utility requires the original two-class OutputLayer log probabilities")
        labels = labels.to(device=logp.device, dtype=torch.long)
        legal = torch.as_tensor(legal_rows, dtype=torch.long, device=logp.device)
        if legal.numel() == 0:
            raise ValueError("DUA utility received no legal training patients")
        legal_labels = labels[legal]
        legal_logp = logp[legal]
        legal_other = 1 - legal_labels
        margin = legal_logp.gather(1, legal_labels[:, None]).squeeze(1) - legal_logp.gather(
            1, legal_other[:, None]
        ).squeeze(1)
        gradient = torch.autograd.grad(margin.sum(), z_final, retain_graph=True, create_graph=False)[0]

        slots = z_cam.reshape(-1, 16, 9).float()
        gradient_slots = gradient.detach().reshape(-1, 16, 9).float()
        differences = slots[:, self.neighbors] - slots.unsqueeze(2)
        gradient_local = gradient_slots.unsqueeze(2).expand_as(differences)
        dot = (gradient_local * differences).sum(dim=-1)
        denominator = gradient_local.norm(dim=-1) * differences.norm(dim=-1)
        utility = torch.where(denominator > 1e-12, dot / denominator.clamp_min(1e-12), torch.zeros_like(dot))
        utility = utility.clamp(-1.0, 1.0).detach()

        class_names = sorted(class_to_index, key=class_to_index.get)
        if class_names != ["ASD", "TD"] or [class_to_index[name] for name in class_names] != [0, 1]:
            raise ValueError(f"unexpected ASD/TD OutputLayer mapping: {class_to_index}")
        scores_by_index = [details["scores_asd"], details["scores_td"]]
        score_bank = torch.stack(scores_by_index, dim=1)[legal]
        score_index = legal_labels[:, None, None, None].expand(-1, 1, 16, 16)
        true_scores = score_bank.gather(1, score_index).squeeze(1)
        other_scores = score_bank.gather(
            1, (1 - legal_labels)[:, None, None, None].expand(-1, 1, 16, 16)
        ).squeeze(1)
        edge_logits = true_scores[..., :15] - other_scores[..., :15]
        legal_utility = utility[legal]
        edge_targets = torch.sigmoid(legal_utility / temperature)
        confidence = (legal_utility.abs() / temperature).clamp(0.0, 1.0)
        patient_weight = torch.where(margin.detach() < 0.0, 1.5, 1.0).to(edge_logits.dtype)
        weights = confidence * patient_weight[:, None, None]
        edge_bce = F.binary_cross_entropy_with_logits(edge_logits, edge_targets, reduction="none")
        edge_loss = (edge_bce * weights).sum() / weights.sum().clamp_min(1e-12)

        null_targets = torch.sigmoid(-legal_utility.max(dim=-1).values / temperature)
        null_logits = true_scores[..., 15] - torch.logsumexp(true_scores[..., :15], dim=-1)
        null_bce = F.binary_cross_entropy_with_logits(null_logits, null_targets, reduction="none")
        node_weights = patient_weight[:, None].expand_as(null_bce)
        null_loss = (null_bce * node_weights).sum() / node_weights.sum().clamp_min(1e-12)
        total = edge_weight * edge_loss + null_weight * null_loss
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite DUA utility objective")
        return total, {
            "edge_loss": float(edge_loss.detach().cpu()),
            "null_loss": float(null_loss.detach().cpu()),
            "utility_mean": float(legal_utility.mean().cpu()),
            "utility_positive_fraction": float((legal_utility > 0).float().mean().cpu()),
            "utility_confidence_mean": float(confidence.mean().cpu()),
            "legal_patient_count": int(legal.numel()),
            "legal_edge_count": int(legal.numel() * 16 * 15),
            "hard_patient_count": int((margin.detach() < 0).sum().cpu()),
        }
