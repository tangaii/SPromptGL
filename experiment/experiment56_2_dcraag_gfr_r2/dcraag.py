"""DCRAG-GFR-R2: integrated four-modality disease relation graph refinement.

This module keeps the E51 four GraphLearn/GCN branches and CAM interface, while
making the new selected-edge path differentiable, bounded and auditable.
"""
from __future__ import annotations
import math
from typing import Any, Sequence
import torch
from torch import nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = float(scale)
        return value.view_as(value)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def gradient_reverse(value, scale):
    return _GradientReverse.apply(value, scale)


def disease_condition(prob: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Detached AA/TT/AT soft condition; no test labels can enter this path."""
    p_i = prob[pairs[:, 0]].detach()
    p_j = prob[pairs[:, 1]].detach()
    out = torch.stack((p_i[:, 0] * p_j[:, 0],
                       p_i[:, 1] * p_j[:, 1],
                       p_i[:, 0] * p_j[:, 1] + p_i[:, 1] * p_j[:, 0]), dim=1)
    return out / out.sum(1, keepdim=True).clamp_min(1e-8)


def relation_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """One minus normalized entropy for a valid three-class probability row."""
    p = probabilities.clamp_min(1e-8)
    entropy_norm = -(p * p.log()).sum(dim=-1) / math.log(3.0)
    return (1.0 - entropy_norm).clamp(0.05, 1.0)


def softcap(value: torch.Tensor, radius: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Continuously cap norm without amplifying small vectors.

    The scalar factor is in (0,1], preserves direction, is differentiable for
    nonzero inputs, and maps a zero radius/vector safely to zero.
    """
    r = radius.to(dtype=value.dtype).clamp_min(0.0)
    norm = value.norm(2, dim=-1, keepdim=True)
    factor = 1.0 / (1.0 + norm / (r + eps))
    return value * factor


def row_normalize(adjacency: torch.Tensor, self_loop: bool = False) -> torch.Tensor:
    """Exactly one directed receiver-row normalization."""
    if self_loop:
        adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype,
                                          device=adjacency.device)
    adjacency = adjacency.clamp_min(0.0)
    return adjacency / adjacency.sum(1, keepdim=True).clamp_min(1e-8)


class PairRelationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.patient = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.pair = nn.Sequential(nn.Linear(hidden * 2 + 1, 64), nn.GELU(),
                                  nn.Linear(64, hidden), nn.GELU())
        self.shared_classifier = nn.Linear(16, 3)
        self.private_classifier = nn.Linear(16, 3)
        # Compatibility alias used by the audit/training code.
        self.classifier = self.shared_classifier
        self.edge = nn.Sequential(nn.Linear(hidden + 4, 32), nn.GELU(), nn.Linear(32, 2))

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return self.patient(x)

    def encode_pairs(self, node_code: torch.Tensor, pairs: torch.Tensor):
        a, b = node_code[pairs[:, 0]], node_code[pairs[:, 1]]
        symmetric = torch.cat((torch.abs(a - b), a * b,
                               F.cosine_similarity(a, b, dim=1).unsqueeze(1)), dim=1)
        code = self.pair(symmetric)
        heads = torch.stack((self.shared_classifier(code[:, :16]),
                             self.private_classifier(code[:, 16:])), dim=1)
        return code, heads


class DCRAGGFR(nn.Module):
    """The complete integrated GFR replacement, outputting [N,320]."""
    def __init__(self, e1_gfr: nn.Module, dims: Sequence[int], utils_module: Any):
        super().__init__()
        self.graphs = e1_gfr.graphs
        self.gcn = e1_gfr.gcn
        self.fusion_weight = e1_gfr.fusion_weight
        self.utils = utils_module
        self.dims = tuple(dims)
        self.total_dim = int(sum(dims))
        self.rho = 0.25
        self.same_radius_fraction = 0.50
        self.diff_radius_fraction = 0.15
        self.eta_max = 0.30
        self.eta_initial = 0.10
        eta_logit = math.log(self.eta_initial / (self.eta_max - self.eta_initial))
        self.eta_raw = nn.Parameter(torch.full((4,), eta_logit))
        self.relations = nn.ModuleList([PairRelationEncoder(d) for d in dims])
        self.difference = nn.ModuleList([
            nn.Sequential(nn.Linear(self.total_dim, 32, bias=False), nn.Tanh(),
                          nn.Linear(32, self.total_dim, bias=False)) for _ in dims
        ])
        # raw320 + four 320-D directions + 4 intensity + 4 receiver confidence + 4 reliability
        self.gate_in = self.total_dim * 5 + 12
        self.gate = nn.Sequential(nn.Linear(self.gate_in, 64), nn.GELU(), nn.Linear(64, 4))
        # Nonzero, isolated initialization: near-uniform but not forced constant.
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.normal_(self.gate[2].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate[2].bias)
        self.beta_gate = nn.Linear(self.gate_in, 1)
        nn.init.zeros_(self.beta_gate.weight)
        nn.init.constant_(self.beta_gate.bias, math.log(0.4 / 0.6))
        self.adversary = nn.Sequential(nn.Linear(51, 48), nn.GELU(), nn.Linear(48, 4))
        self.last: dict[str, Any] = {}

    def reference_bypass(self, parts):
        raw = torch.cat(parts, dim=1)
        adjs, msgs = [], []
        fused = raw
        for m, x in enumerate(parts):
            adj = self.graphs[m](x)
            eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            msg = self.gcn[m](raw, self.utils.normalize_adj(adj + eye))
            adjs.append(adj)
            msgs.append(msg)
            fused = fused + torch.sigmoid(self.fusion_weight[m]) * msg
        self.last = {"adjs": adjs, "msgs": msgs, "fused": fused, "bypass": True}
        return fused, adjs, self.last

    def _select_edges(self, edges: torch.Tensor, score: torch.Tensor, n: int):
        chosen = []
        for receiver in range(n):
            idx = torch.nonzero(edges[:, 0] == receiver, as_tuple=False).flatten()
            k = min(16, int(idx.numel()))
            if k:
                # Stable tie resolution comes from candidate order (patient ID order).
                _, order = torch.topk(score[idx], k=k, largest=True, sorted=True)
                chosen.append(idx[order])
        return torch.cat(chosen) if chosen else edges.new_empty((0,), dtype=torch.long)

    def forward(self, parts, candidate_edges, edge_prior, patient_prob, relation_pairs,
                relation_targets, relation_condition, reliability, grl_scale=0.0,
                intervention=None, reference_bypass=False):
        if reference_bypass:
            return self.reference_bypass(parts)
        intervention = intervention or {}
        raw = torch.cat(parts, dim=1)
        n = raw.shape[0]
        node_codes = [r.encode_nodes(x) for r, x in zip(self.relations, parts)]
        rel_codes, rel_logits, private_logits = [], [], []
        adversary_codes = []
        edge_same, edge_diff, edge_logits, selected_edges = [], [], [], []
        A_same, A_diff, same_msgs, diff_msgs, msgs = [], [], [], [], []
        raw_adjs, edge_priors, confidences, confidence_entropies = [], [], [], []
        stability = []
        used_prob = intervention.get("patient_prob", patient_prob).detach()
        forced = intervention.get("forced_selected_edges")
        forced_prior = intervention.get("forced_edge_prior")

        for m, (rel, h, edges, prior) in enumerate(zip(self.relations, node_codes,
                                                         candidate_edges, edge_prior)):
            # Relation supervision is always on the fixed outer-train pair sample.
            pair_code, pair_heads = rel.encode_pairs(h, relation_pairs)
            shared_logit, private_logit = pair_heads[:, 0], pair_heads[:, 1]
            rel_codes.append(pair_code)
            rel_logits.append(shared_logit)
            private_logits.append(private_logit)
            adversary_codes.append(pair_code)

            # Current inherited GraphLearn projection is used on every forward.
            projection = F.normalize(self.graphs[m].p(parts[m]), dim=1)
            if forced is not None and forced[m] is not None:
                ce = forced[m].to(device=raw.device, dtype=torch.long)
                if forced_prior is None or forced_prior[m] is None:
                    cp = (projection[ce[:, 0]] * projection[ce[:, 1]]).sum(1).add(1).mul(.5).clamp_min(.05)
                else:
                    cp = forced_prior[m].to(device=raw.device, dtype=raw.dtype).clamp_min(1e-4)
                csim = (projection[ce[:, 0]] * projection[ce[:, 1]]).sum(1)
                ccond = disease_condition(used_prob, ce)
                cedge_code, _ = rel.encode_pairs(h, ce)
                cprob = rel.shared_classifier(cedge_code[:, :16]).softmax(-1)
            else:
                edge_code, _ = rel.encode_pairs(h, edges)
                sim = (projection[edges[:, 0]] * projection[edges[:, 1]]).sum(1)
                eprob = rel.shared_classifier(edge_code[:, :16]).softmax(-1)
                score = (.45 * ((sim + 1.0) * .5) + .35 * eprob.max(1).values
                         + .20 * prior.clamp(0, 1))
                chosen = self._select_edges(edges, score, n)
                ce, cp, csim = edges[chosen], prior[chosen], sim[chosen]
                ccond, cprob = disease_condition(used_prob, ce), eprob[chosen]
                cedge_code = edge_code[chosen]
            confidence = relation_confidence(cprob)
            entropy = 1.0 - confidence
            confidence_entropies.append(entropy)
            edge_score = rel.edge(torch.cat((cedge_code, ccond,
                                              ((csim + 1.0) * .5).unsqueeze(1)), dim=1)).sigmoid()
            # Both the current GraphLearn similarity and fixed pair-specific prior remain.
            sim01 = ((csim + 1.0) * .5).clamp(0, 1)
            prior_factor = cp.clamp_min(1e-4)
            reliability_factor = (0.25 + 0.75 * reliability[m]).to(raw.dtype)
            base = (.15 + .85 * sim01) * confidence * prior_factor * reliability_factor
            same = base * (.10 + .90 * (cprob[:, 0] + cprob[:, 1])) * edge_score[:, 0]
            diff = base * (.10 + .90 * cprob[:, 2]) * edge_score[:, 1]
            if intervention.get("neutral_edges"):
                same = (prior_factor * confidence).detach() * .5
                diff = (prior_factor * confidence).detach() * .5
            if intervention.get("disable_edges"):
                same, diff = torch.zeros_like(same), torch.zeros_like(diff)
            ii, jj = ce[:, 0], ce[:, 1]
            smat = raw.new_zeros((n, n)).index_put((ii, jj), same, accumulate=True)
            dmat = raw.new_zeros((n, n)).index_put((ii, jj), diff, accumulate=True)
            self_only = intervention.get("self_only") or intervention.get("disable_messages") \
                or m in intervention.get("disabled_modalities", ())
            if self_only:
                snorm = torch.eye(n, device=raw.device, dtype=raw.dtype)
                dnorm = raw.new_zeros((n, n))
            else:
                snorm = row_normalize(smat, self_loop=True)
                dnorm = row_normalize(dmat, self_loop=False)
            raw_adj = self.graphs[m](parts[m])
            raw_adjs.append(raw_adj)
            same_raw = self.gcn[m](raw, snorm)
            raw_norm = raw.norm(2, dim=1, keepdim=True)
            same_msg = softcap(same_raw, self.same_radius_fraction * raw_norm)
            delta = self.difference[m](raw[ii] - raw[jj])
            diff_raw = raw.new_zeros(raw.shape).index_add(0, ii,
                         delta * dnorm[ii, jj].unsqueeze(1))
            diff_msg = softcap(diff_raw, self.diff_radius_fraction * raw_norm)
            eta = self.eta_max * torch.sigmoid(self.eta_raw[m])
            modal_msg = same_msg + eta * diff_msg

            A_same.append(snorm); A_diff.append(dnorm)
            edge_same.append(same); edge_diff.append(diff); edge_logits.append(cprob)
            selected_edges.append(ce); edge_priors.append(cp); confidences.append(confidence)
            same_msgs.append(same_msg); diff_msgs.append(eta * diff_msg); msgs.append(modal_msg)
            # Receiver-specific confidence, never a global mean.
            conf_sum = raw.new_zeros(n).index_add(0, ii, confidence)
            conf_count = raw.new_zeros(n).index_add(0, ii, torch.ones_like(confidence))
            receiver_conf = conf_sum / conf_count.clamp_min(1.0)
            confidence_entropies[-1] = entropy
            # Edge stability compares normalized selected weights with pair priors.
            selected_sum = (same + diff).new_zeros(n).index_add(0, ii, same + diff)
            norm_sel = (same + diff) / selected_sum[ii].clamp_min(1e-8)
            norm_prior = cp.clamp_min(1e-6)
            prior_sum = norm_prior.new_zeros(n).index_add(0, ii, norm_prior)
            norm_prior = norm_prior / prior_sum[ii].clamp_min(1e-8)
            stability.append((norm_sel * (norm_sel.clamp_min(1e-8).log()
                                           - norm_prior.clamp_min(1e-8).log())).mean())
            # Store per-patient summaries in a side list by attaching to confidence tensor.
            confidences[-1] = (confidence, receiver_conf)

        rel_logits_t = torch.stack(rel_logits)
        private_t = torch.stack(private_logits)
        relation_losses = []
        private_losses = []
        for m in range(4):
            relation_losses.append(F.cross_entropy(rel_logits_t[m], relation_targets))
            private_losses.append(F.cross_entropy(private_t[m], relation_targets))
        relation_loss_each = torch.stack(relation_losses)
        rel_loss = (relation_loss_each * reliability.to(relation_losses[0].dtype)).sum() \
            / reliability.sum().clamp_min(1e-8)
        rel_loss = rel_loss + 0.20 * torch.stack(private_losses).mean()
        rel_probs = rel_logits_t.softmax(-1)
        consensus = (rel_probs * reliability.to(rel_probs.dtype).view(4, 1, 1)).sum(0) \
            / reliability.sum().clamp_min(1e-8)
        consistency_each = (rel_probs.clamp_min(1e-8)
            * (rel_probs.clamp_min(1e-8).log() - consensus.clamp_min(1e-8).log())).sum(-1).mean(1)
        consistency = (consistency_each * reliability.to(consistency_each.dtype)).sum() \
            / reliability.sum().clamp_min(1e-8)

        # Reliability-weighted, detached condition prototypes.
        prototype_loss = raw.new_zeros(())
        prototype_records = []
        for cls in range(3):
            mask = relation_targets == cls
            protos = []
            for m in range(4):
                z = F.normalize(rel_codes[m][:, :16], dim=1)
                mean = z[mask].mean(0) if bool(mask.any()) else z.mean(0)
                protos.append(F.normalize(mean, dim=0))
            weights = reliability.to(protos[0].dtype)
            target = torch.stack(protos).mul(weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8)
            target = target.detach()
            distances = torch.stack([(p - target).pow(2).mean() for p in protos])
            prototype_loss = prototype_loss + (distances * weights).sum() / weights.sum().clamp_min(1e-8)
            prototype_records.append({"target": target, "prototypes": protos, "distances": distances})

        source_rows = []
        condition = relation_condition.detach()
        for code in adversary_codes:
            shared = code[:, :16]
            conditioned = (gradient_reverse(shared, grl_scale).unsqueeze(2)
                           * condition.unsqueeze(1)).reshape(-1, 48)
            source_rows.append(torch.cat((conditioned, condition), dim=1))
        source_input = torch.stack(source_rows).reshape(-1, 51)
        source_logits = self.adversary(source_input)
        source_target = torch.arange(4, device=raw.device).view(4, 1).expand(
            4, relation_pairs.shape[0]).reshape(-1)
        adv_each = F.cross_entropy(source_logits, source_target, reduction="none").view(4, -1).mean(1)
        adv_loss = (adv_each * reliability.to(adv_each.dtype)).sum() / reliability.sum().clamp_min(1e-8)

        # Patient-specific Gate input: receiver confidence comes from each selected edge set.
        conf_summary = torch.stack([c[1] for c in confidences], dim=1)
        msg_norms = torch.stack([m.norm(2, dim=1) for m in msgs], dim=1)
        direction = [F.normalize(m, dim=1) for m in msgs]
        gate_input = torch.cat([F.normalize(raw, dim=1), *direction], dim=1)
        summary = torch.cat((msg_norms, conf_summary,
                             reliability.to(raw.dtype).view(1, 4).expand(n, 4)), dim=1)
        gate_features = torch.cat((gate_input, summary), dim=1)
        gate = self.gate(gate_features).softmax(-1)
        if intervention.get("freeze_gate") is not None:
            gate = intervention["freeze_gate"]
        weights = gate * torch.sigmoid(self.fusion_weight).view(1, 4)
        mraw = sum(weights[:, m:m+1] * msgs[m] for m in range(4))
        raw_norm = raw.norm(2, dim=1, keepdim=True)
        mbound = softcap(mraw, self.rho * raw_norm)
        beta = torch.sigmoid(self.beta_gate(gate_features)).clamp(0.0, 1.0)
        fused = raw + beta * mbound
        bound_active = (mbound.norm(2, dim=1, keepdim=True)
                        < mraw.norm(2, dim=1, keepdim=True).clamp_min(1e-12)).float()
        eta_values = self.eta_max * torch.sigmoid(self.eta_raw)
        self.last = {
            "adjs": raw_adjs, "msgs": msgs, "same_msgs": same_msgs, "diff_msgs": diff_msgs,
            "raw_same_msgs": [self.gcn[m](raw, A_same[m]) for m in range(4)],
            "fused": fused, "A_same": A_same, "A_diff": A_diff,
            "relation_logits": rel_logits_t, "private_logits": private_t,
            "relation_probs": rel_probs, "edge_same": edge_same, "edge_diff": edge_diff,
            "edge_logits": edge_logits, "selected_edges": selected_edges, "edge_priors": edge_priors,
            "gate": gate, "gate_features": gate_features, "beta": beta, "mraw": mraw,
            "mbound": mbound, "message_norm_before_bound": mraw.norm(2, dim=1),
            "message_norm_after_bound": mbound.norm(2, dim=1), "same_message_norms": [x.norm(2, dim=1) for x in same_msgs],
            "diff_message_norms": [x.norm(2, dim=1) for x in diff_msgs], "eta": eta_values,
            "node_codes": node_codes, "relation_codes": rel_codes, "adv_logits": source_logits,
            "adversary_codes": adversary_codes, "adv_targets": source_target,
            "loss_relation": rel_loss, "loss_adversarial": adv_loss, "loss_prototype": prototype_loss,
            "loss_consistency": consistency, "loss_edge": torch.stack(stability).mean(),
            "loss_relation_each": relation_loss_each, "loss_adversarial_each": adv_each,
            "prototype_records": prototype_records, "grl_scale": grl_scale,
            "patient_prob": used_prob, "bound_active": bound_active,
            "confidence": [c[0] for c in confidences], "receiver_confidence": conf_summary,
            "confidence_entropy": confidence_entropies, "reliability": reliability,
        }
        return fused, raw_adjs, self.last
