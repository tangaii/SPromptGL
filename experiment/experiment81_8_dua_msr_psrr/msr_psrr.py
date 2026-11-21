"""Multi-scale, structure-consistent extension of the node-level TP-PSRR path."""
from __future__ import annotations

import torch
from torch import nn

from tp_psrr import TPPSRR, prompt_groups


class MSRPSRR(TPPSRR):
    def __init__(self, semantic_basis, rho=0.1, relation_scale=1.0, temperature=1.0):
        super().__init__(semantic_basis, rho, relation_scale, temperature, selection_mode="node")
        self.scale_shared = nn.Linear(27, 32)
        self.scale_contrast = nn.Linear(96, 32)

    @staticmethod
    def _adjacency(probabilities, neighbor_ids):
        n = probabilities.shape[0]
        indices = neighbor_ids.unsqueeze(0).expand(n, -1, -1)
        return probabilities.new_zeros((n, 16, 16)).scatter(2, indices, probabilities)

    def _relations(self, tokens, groups, intervention, return_diagnostics, relation_features=None):
        _, base = super()._relations(
            tokens, groups, intervention, True, relation_features=relation_features
        )
        if relation_features is None or tuple(relation_features.shape[1:]) != (16, 9):
            raise ValueError("MSR-PSRR requires the original CAM144 slots with shape [N,16,9]")
        neighbor_ids = self.neighbors.to(device=tokens.device)
        attention_td, attention_asd = self.apply_audit_edge_overrides(
            base["attention_td"], base["attention_asd"]
        )
        adjacency_td = self._adjacency(attention_td[..., :15], neighbor_ids)
        adjacency_asd = self._adjacency(attention_asd[..., :15], neighbor_ids)
        values = relation_features

        first_td = adjacency_td @ values
        second_td = adjacency_td @ first_td
        high_td = values - first_td
        first_asd = adjacency_asd @ values
        second_asd = adjacency_asd @ first_asd
        high_asd = values - first_asd

        second_used_td, second_used_asd = second_td, second_asd
        high_used_td, high_used_asd = high_td, high_asd
        if intervention in ("second_order_off", "multiscale_off"):
            second_used_td = torch.zeros_like(second_td)
            second_used_asd = torch.zeros_like(second_asd)
        if intervention in ("high_frequency_off", "multiscale_off"):
            high_used_td = torch.zeros_like(high_td)
            high_used_asd = torch.zeros_like(high_asd)

        representation_td = self.scale_shared(torch.cat((first_td, second_used_td, high_used_td), dim=-1))
        representation_asd = self.scale_shared(torch.cat((first_asd, second_used_asd, high_used_asd), dim=-1))
        contrast = self.scale_contrast(torch.cat((
            representation_asd,
            representation_td,
            representation_asd - representation_td,
        ), dim=-1))
        if intervention == "relation_off":
            contrast = torch.zeros_like(contrast)

        evidence = torch.cat((representation_asd - representation_td,
                              second_used_asd - second_used_td), dim=-1).mean(dim=1)
        details = {
            **base,
            "attention_td": attention_td,
            "attention_asd": attention_asd,
            "adjacency_td": adjacency_td,
            "adjacency_asd": adjacency_asd,
            "first_order_td": first_td,
            "first_order_asd": first_asd,
            "second_order_td": second_td,
            "second_order_asd": second_asd,
            "high_frequency_td": high_td,
            "high_frequency_asd": high_asd,
            "representation_td": representation_td,
            "representation_asd": representation_asd,
            "multiscale_contrast": contrast,
            "contrastive_vector": evidence,
            "null_mass_td": attention_td[..., 15],
            "null_mass_asd": attention_asd[..., 15],
        }
        return contrast, details
