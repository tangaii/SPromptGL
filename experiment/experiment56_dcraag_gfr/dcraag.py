"""Experiment 56: unified four-modality disease-relation GFR."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        return -ctx.scale * grad, None


def gradient_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


def disease_condition(prob: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """AA, TT, AT soft probabilities for unordered patient pairs."""
    pi, pj = prob[pairs[:, 0]].detach(), prob[pairs[:, 1]].detach()
    out = torch.stack((pi[:, 0] * pj[:, 0], pi[:, 1] * pj[:, 1],
                       pi[:, 0] * pj[:, 1] + pi[:, 1] * pj[:, 0]), dim=1)
    return out / out.sum(1, keepdim=True).clamp_min(1e-8)


class PairRelationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.patient = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.pair = nn.Sequential(nn.Linear(hidden * 2 + 1, 64), nn.GELU(),
                                  nn.Linear(64, hidden), nn.GELU())
        self.classifier = nn.Linear(hidden, 3)
        self.edge = nn.Sequential(nn.Linear(hidden + 2, 24), nn.GELU(), nn.Linear(24, 2))
        self.condition_logit = nn.Parameter(torch.tensor(-1.734601))  # sigmoid ~= 0.15

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.patient(x)

    def encode_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = h[pairs[:, 0]], h[pairs[:, 1]]
        symmetric = torch.cat((torch.abs(left - right), left * right,
                               F.cosine_similarity(left, right, dim=1).unsqueeze(1)), dim=1)
        code = self.pair(symmetric)
        return code, self.classifier(code)


class DCRAGGFR(nn.Module):
    """Unified replacement for E51 E1 GFR; all outputs remain 320-D."""
    def __init__(self, e1_gfr: nn.Module, dims: list[int], utils_module: Any):
        super().__init__()
        self.graphs = e1_gfr.graphs
        self.gcn = e1_gfr.gcn
        self.fusion_weight = e1_gfr.fusion_weight
        self.utils = utils_module
        self.dims = tuple(dims)
        self.relations = nn.ModuleList([PairRelationEncoder(d) for d in dims])
        self.difference = nn.ModuleList([
            nn.Sequential(nn.Linear(sum(dims), 32, bias=False), nn.Tanh(),
                          nn.Linear(32, sum(dims), bias=False)) for _ in dims
        ])
        self.eta_logit = nn.Parameter(torch.full((4,), math.log(0.4 / 0.6)))
        self.gate = nn.Linear(sum(dims) * 5, 4)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.beta_gate = nn.Linear(sum(dims) * 5, 1)
        nn.init.zeros_(self.beta_gate.weight)
        nn.init.constant_(self.beta_gate.bias, math.log(4.0))  # 1.25*sigmoid = 1
        self.adversary = nn.Sequential(nn.Linear(51, 48), nn.GELU(), nn.Linear(48, 4))
        self.last: dict[str, Any] = {}

    def reference_bypass(self, parts: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, Any]]:
        """Test-only execution of E1's exact graph/GCN/fusion sequence."""
        raw = torch.cat(parts, dim=1)
        fused, adjs, messages = raw, [], []
        for m, x in enumerate(parts):
            adj = self.graphs[m](x)
            adjs.append(adj)
            eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            normalized = self.utils.normalize_adj(adj + eye)
            msg = self.gcn[m](raw, normalized)
            messages.append(msg)
            fused = fused + torch.sigmoid(self.fusion_weight[m]) * msg
        details = {"adjs": adjs, "msgs": messages, "fused": fused, "bypass": True}
        return fused, adjs, details

    def forward(self, parts: list[torch.Tensor], candidate_edges: list[torch.Tensor],
                edge_prior: list[torch.Tensor], patient_prob: torch.Tensor,
                relation_pairs: torch.Tensor, relation_targets: torch.Tensor,
                relation_condition: torch.Tensor, reliability: torch.Tensor,
                grl_scale: float = 0.0, intervention: dict[str, Any] | None = None,
                reference_bypass: bool = False):
        if reference_bypass:
            return self.reference_bypass(parts)
        intervention = intervention or {}
        raw = torch.cat(parts, dim=1)
        n = raw.shape[0]
        node_codes = [module.encode_nodes(part) for module, part in zip(self.relations, parts)]
        relation_codes, relation_logits, relation_probs, adversary_codes = [], [], [], []
        edge_logits, same_edges, diff_edges = [], [], []
        adj_same, adj_diff, messages, differences, raw_adjs = [], [], [], [], []
        stability_terms = []
        used_patient_prob = intervention.get("patient_prob", patient_prob).detach()

        for m, (module, h, edges, prior) in enumerate(zip(self.relations, node_codes, candidate_edges, edge_prior)):
            # A directed candidate means row i receives from column j; relation codes stay symmetric.
            edge_codes, edge_logit = module.encode_pairs(h, edges)
            relation_codes.append(edge_codes)
            edge_logits.append(edge_logit)

            pair_codes, pair_logit = module.encode_pairs(h, relation_pairs)
            adversary_codes.append(pair_codes)
            relation_logits.append(pair_logit)
            relation_probs.append(pair_logit.softmax(-1))

            cond = disease_condition(used_patient_prob, edges)
            raw_prob = edge_logit.softmax(-1)
            prior_strength = torch.sigmoid(module.condition_logit)
            conditioned = (raw_prob.clamp_min(1e-8).log() +
                           prior_strength * reliability[m] * cond.clamp_min(1e-8).log()).softmax(-1)
            if intervention.get("wrong_relation", False):
                conditioned = conditioned[:, [1, 2, 0]]
            confidence = 0.1 + 0.9 * (1.0 - (conditioned.clamp_min(1e-8).log() * conditioned).sum(1) / math.log(3.0)).clamp(0, 1)
            rel_input = reliability[m].expand_as(prior)
            edge_gate = module.edge(torch.cat((edge_codes, prior.unsqueeze(1), rel_input.unsqueeze(1)), dim=1)).sigmoid()
            base = prior.clamp_min(1e-4) * confidence
            same_signal = 0.2 + 0.8 * reliability[m] * (conditioned[:, 0] + conditioned[:, 1])
            diff_signal = 0.2 + 0.8 * reliability[m] * conditioned[:, 2]
            same = base * same_signal * edge_gate[:, 0]
            diff = base * diff_signal * edge_gate[:, 1]
            if intervention.get("neutral_edges", False):
                same = prior.clamp_min(1e-4) * 0.5
                diff = prior.clamp_min(1e-4) * 0.5
            if intervention.get("disable_edges", False):
                same = torch.zeros_like(same)
                diff = torch.zeros_like(diff)
            same_edges.append(same)
            diff_edges.append(diff)

            row_prior = prior.new_zeros(n).index_add(0, edges[:, 0], prior.clamp_min(1e-6))
            p_prior = prior.clamp_min(1e-6) / row_prior[edges[:, 0]].clamp_min(1e-8)
            learned_edge = same + diff
            row_learned = learned_edge.new_zeros(n).index_add(0, edges[:, 0], learned_edge)
            q_edge = learned_edge / row_learned[edges[:, 0]].clamp_min(1e-8)
            stability_terms.append((q_edge * (q_edge.clamp_min(1e-8).log() - p_prior.log())).mean())

            ii, jj = edges[:, 0], edges[:, 1]
            A_same = raw.new_zeros((n, n)).index_put((ii, jj), same, accumulate=True)
            A_diff = raw.new_zeros((n, n)).index_put((ii, jj), diff, accumulate=True)
            same_norm = A_same / A_same.sum(1, keepdim=True).clamp_min(1e-8)
            diff_norm = A_diff / A_diff.sum(1, keepdim=True).clamp_min(1e-8)
            if intervention.get("disable_messages", False) or m in intervention.get("disabled_modalities", ()):
                same_norm = torch.zeros_like(same_norm)
                diff_norm = torch.zeros_like(diff_norm)
            adj_same.append(same_norm)
            adj_diff.append(diff_norm)

            original = self.graphs[m](parts[m])
            raw_adjs.append(original)
            eye = torch.eye(n, device=raw.device, dtype=raw.dtype)
            msg_same = self.gcn[m](raw, self.utils.normalize_adj(same_norm + eye))
            edge_delta = self.difference[m](raw[ii] - raw[jj])
            dmsg = raw.new_zeros(raw.shape).index_add(0, ii, edge_delta * diff_norm[ii, jj].unsqueeze(1))
            d_rms = dmsg.pow(2).mean(1, keepdim=True).add(1e-8).sqrt()
            s_rms = msg_same.pow(2).mean(1, keepdim=True).add(1e-8).sqrt()
            dmsg = dmsg * torch.clamp(0.5 * s_rms / d_rms, max=1.0)
            eta = 0.3 * torch.sigmoid(self.eta_logit[m])
            differences.append(eta * dmsg)
            messages.append(msg_same + eta * dmsg)

        rel_logits = torch.stack(relation_logits, dim=0)
        rel_probs = rel_logits.softmax(-1)
        rel_loss_each = F.cross_entropy(rel_logits.reshape(-1, 3), relation_targets.repeat(4), reduction="none").view(4, -1).mean(1)
        rel_loss = (rel_loss_each * reliability).sum() / reliability.sum().clamp_min(1e-8)

        probs = rel_probs.detach()
        consensus = (probs * reliability[:, None, None]).sum(0) / reliability.sum().clamp_min(1e-8)
        consistency_each = (rel_probs.clamp_min(1e-8) *
                            (rel_probs.clamp_min(1e-8).log() - consensus.clamp_min(1e-8).log().unsqueeze(0))).sum(-1).mean(1)
        consistency = (consistency_each * reliability).sum() / reliability.sum().clamp_min(1e-8)

        source_rows = []
        for m in range(4):
            shared = adversary_codes[m][:, :16]
            condition = relation_condition.detach()
            conditional = (gradient_reverse(shared, grl_scale).unsqueeze(2) * condition.unsqueeze(1)).reshape(-1, 48)
            source_rows.append(torch.cat((conditional, condition), dim=1))
        source_input = torch.stack(source_rows, dim=0)
        source_logits = self.adversary(source_input.reshape(-1, 51))
        source_target = torch.arange(4, device=raw.device).view(4, 1).expand(4, relation_pairs.shape[0]).reshape(-1)
        adv_loss_per = F.cross_entropy(source_logits, source_target, reduction="none").view(4, -1).mean(1)
        adv_loss = (adv_loss_per * reliability).sum() / reliability.sum().clamp_min(1e-8)

        same_cat = torch.cat(same_edges)
        diff_cat = torch.cat(diff_edges)
        edge_stability = torch.stack(stability_terms).mean()

        contexts = torch.cat((raw, *messages), dim=1)
        gate = self.gate(contexts).softmax(-1)
        if intervention.get("disable_messages", False):
            fused = raw
            gate_used = torch.zeros_like(gate)
            beta = raw.new_tensor(0.0)
        else:
            active = torch.ones(4, device=raw.device, dtype=raw.dtype)
            for m in intervention.get("disabled_modalities", ()):
                active[m] = 0.0
            gate = gate * active.unsqueeze(0)
            gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-8)
            scaled = gate * 4.0 * torch.sigmoid(self.fusion_weight).view(1, 4)
            beta = 1.25 * torch.sigmoid(self.beta_gate(contexts))
            fused = raw + beta * sum(scaled[:, m:m + 1] * messages[m] for m in range(4))
            gate_used = gate

        details = {
            "adjs": raw_adjs, "msgs": messages, "same_msgs": [m - d for m, d in zip(messages, differences)],
            "diff_msgs": differences, "fused": fused, "A_same": adj_same, "A_diff": adj_diff,
            "relation_logits": rel_logits, "relation_probs": rel_probs,
            "edge_same": same_edges, "edge_diff": diff_edges, "edge_logits": edge_logits,
            "gate": gate_used, "beta": beta, "node_codes": node_codes,
            "relation_codes": relation_codes, "adv_logits": source_logits,
            "adversary_codes": adversary_codes,
            "adv_targets": source_target, "loss_relation": rel_loss,
            "loss_adversarial": adv_loss, "loss_consistency": consistency,
            "loss_edge": edge_stability, "loss_relation_each": rel_loss_each,
            "loss_adversarial_each": adv_loss_per, "grl_scale": grl_scale,
            "patient_prob": used_patient_prob,
        }
        self.last = details
        return fused, raw_adjs, details

