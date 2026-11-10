"""Semantic-Conditioned Prototype Residual modules for Experiment 76.5."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


# Verified against BrainPrompt c85304a:data/prompts/label_prompts/abide_label.txt.
PROMPTS = (
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
)


def anonymous_pair(text_vectors: torch.Tensor, seed: int = 765001) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(text_vectors.shape[-1], generator=generator)
    signs = torch.randint(0, 2, (text_vectors.shape[-1],), generator=generator).float().mul_(2).sub_(1)
    return text_vectors.detach().cpu()[:, permutation] * signs


class SCPR(nn.Module):
    def __init__(self, text_vectors: torch.Tensor):
        super().__init__()
        if text_vectors.ndim != 2 or text_vectors.shape[0] != 2:
            raise ValueError("expected text vectors in ASD=0, TD=1 order")
        self.register_buffer("text_vectors", F.normalize(text_vectors.detach().float(), dim=-1))
        self.semantic = nn.Sequential(nn.Linear(text_vectors.shape[-1], 16), nn.GELU(), nn.Linear(16, 144))
        self.theta = nn.Parameter(torch.full((4,), math.log(.1 / .9)))  # rho=0.02
        self.modality_gate = nn.Sequential(nn.Linear(144 + 4 + 4 + 4, 16), nn.GELU(), nn.Linear(16, 4))
        self.residual_gate = nn.Sequential(nn.Linear(40, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.constant_(self.residual_gate[-1].bias, -3.0)
        self.alpha = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _select(ids: torch.Tensor, sim: torch.Tensor, sites: Optional[torch.Tensor], k: int, cap: int):
        order = torch.argsort(sim, descending=True).tolist()
        if sites is None:
            return torch.tensor(order[:k], device=ids.device, dtype=torch.long)
        chosen, counts = [], {}
        for pos in order:
            site = int(sites[int(ids[pos])])
            if counts.get(site, 0) < cap:
                chosen.append(pos); counts[site] = counts.get(site, 0) + 1
                if len(chosen) == k:
                    break
        if len(chosen) < min(k, len(order)):
            selected = set(chosen)
            chosen.extend(pos for pos in order if pos not in selected and len(chosen) < k)
        return torch.tensor(chosen, device=ids.device, dtype=torch.long)

    def _pool(self, q: torch.Tensor, memory: Dict[str, torch.Tensor], ids: torch.Tensor,
              weights: torch.Tensor, slot: int, k: int, sites: Optional[torch.Tensor]):
        if ids.numel() == 0:
            return q.new_zeros(36), q.new_zeros(()), 0
        features = memory["features"][ids, slot]
        sims = F.normalize(features, dim=-1) @ F.normalize(q, dim=-1)
        take = self._select(ids, sims, sites, min(k, len(ids)), 2)
        chosen_ids = ids[take]
        chosen = features[take]
        selected_sim = sims[take]
        reliability = weights[chosen_ids].clamp_min(1e-4)
        pool_weights = torch.softmax(selected_sim / .20 + reliability.log(), dim=0)
        proto = F.normalize((pool_weights[:, None] * chosen).sum(0), dim=0)
        return proto, selected_sim.max(), int(len(chosen_ids))

    def _pool_batch(self, q: torch.Tensor, memory: Dict[str, torch.Tensor], ids: torch.Tensor,
                    weights: torch.Tensor, slot: int, k: int, sites: Optional[torch.Tensor]):
        batch = q.shape[0]
        if ids.numel() == 0:
            return q.new_zeros((batch, 36)), q.new_zeros(batch), q.new_zeros(batch)
        features = memory["features"][ids, slot]
        sims = F.normalize(q, dim=-1) @ F.normalize(features, dim=-1).T
        count = min(k, len(ids))
        all_sims, all_pos = torch.topk(sims, count, dim=-1)
        chosen_sims, chosen_pos = all_sims, all_pos
        if sites is not None and count > 0:
            site_ids = sites[ids]
            site_codes = torch.arange(20, device=q.device)
            same_site = site_ids[None, None, :] == site_codes[None, :, None]
            per_site_sims, per_site_pos = torch.topk(
                sims[:, None, :].expand(-1, 20, -1).masked_fill(~same_site, -torch.inf),
                min(2, len(ids)), dim=-1)
            flat_sims = per_site_sims.flatten(1)
            flat_pos = per_site_pos.flatten(1)
            diverse = torch.zeros_like(sims, dtype=torch.int32)
            diverse.scatter_add_(1, flat_pos, torch.isfinite(flat_sims).to(torch.int32))
            priority = sims + (diverse > 0).to(sims.dtype) * 3.0
            _, chosen_pos = torch.topk(priority, count, dim=-1)
            chosen_sims = sims.gather(1, chosen_pos)
        selected_ids = ids[chosen_pos]
        selected = features[chosen_pos]
        reliability = weights[selected_ids].clamp_min(1e-4)
        pool_weights = torch.softmax(chosen_sims / .20 + reliability.log(), dim=-1)
        proto = F.normalize((pool_weights[..., None] * selected).sum(1), dim=-1)
        return proto, chosen_sims.max(-1).values, torch.full((batch,), count, device=q.device, dtype=q.dtype)

    def _one(self, q144: torch.Tensor, memory: Dict[str, torch.Tensor], support: torch.Tensor,
             sites: Optional[torch.Tensor], base_score: torch.Tensor):
        q = F.normalize(q144.reshape(4, 36), dim=-1)
        sem = self.semantic(self.text_vectors).reshape(2, 4, 36)
        rho = .2 * torch.sigmoid(self.theta)
        evidence, rel_sims, bnd_sims, hard_sims, hard_masks, coverages = [], [], [], [], [], []
        for cls in (0, 1):
            cls_ids = support[memory["labels"][support] == cls]
            cls_pred = memory["probabilities"][cls_ids].argmax(-1)
            py = memory["probabilities"][cls_ids, cls]
            rel_ids = cls_ids[cls_pred == cls]
            rel_weights = torch.zeros_like(memory["probabilities"][:, 0])
            rel_weights[rel_ids] = ((memory["probabilities"][rel_ids, cls] - .5) / .5).clamp(0, 1)
            bnd_weights = torch.zeros_like(memory["probabilities"][:, 0])
            bnd_weights[cls_ids] = 4 * py * (1 - py)
            class_ev, class_rel, class_bnd, class_hard, class_hard_mask, class_cov = [], [], [], [], [], []
            for m in range(4):
                p_rel, sr, nr = self._pool(q[m], memory, rel_ids, rel_weights, m, 8, sites)
                p_bnd, sb, nb = self._pool(q[m], memory, cls_ids, bnd_weights, m, 8, sites)
                a = F.normalize(p_rel + rho[m] * sem[cls, m], dim=0)
                b = F.normalize(p_bnd + rho[m] * sem[cls, m], dim=0)
                class_ev.append(.7 * torch.dot(q[m], a) + .3 * torch.dot(q[m], b))
                class_rel.append(sr); class_bnd.append(sb)
                class_cov.append((nr + nb) / 16.0)
                hard_ids = cls_ids[cls_pred != cls]
                if len(hard_ids) >= 3:
                    hard_weight = torch.zeros_like(memory["probabilities"][:, 0])
                    hard_weight[hard_ids] = memory["probabilities"][hard_ids].max(-1).values
                    _, sh, _ = self._pool(q[m], memory, hard_ids, hard_weight, m, 3, sites)
                    class_hard_mask.append(q.new_ones(()))
                else:
                    sh = q.new_zeros(())
                    class_hard_mask.append(q.new_zeros(()))
                class_hard.append(sh)
            evidence.append(torch.stack(class_ev)); rel_sims.append(torch.stack(class_rel))
            bnd_sims.append(torch.stack(class_bnd)); hard_sims.append(torch.stack(class_hard))
            hard_masks.append(torch.stack(class_hard_mask))
            coverages.append(torch.tensor(class_cov, device=q.device, dtype=q.dtype))
        ev = torch.stack(evidence)              # [2,4]
        diff = ev[0] - ev[1]                    # ASD minus TD, retain direction
        rel_diff = rel_sims[0] - rel_sims[1]
        bnd_diff = bnd_sims[0] - bnd_sims[1]
        mw = torch.softmax(self.modality_gate(torch.cat([q144, diff, rel_diff, bnd_diff])), dim=-1)
        d = (mw * diff).sum()
        disagreement = diff - d
        hard = torch.stack([hard_sims[0], hard_sims[1]])
        hard_mask = torch.stack([hard_masks[0], hard_masks[1]])
        coverage = torch.stack(coverages).mean(0)
        sem_agree = torch.stack([F.cosine_similarity(q, F.normalize(sem[c], dim=-1), dim=-1).mean()
                                 for c in (0, 1)])
        abs_margin = base_score.abs()
        gate_input = torch.cat([base_score.reshape(1), abs_margin.reshape(1), ev.flatten(),
                                disagreement, hard.flatten(), hard_mask.flatten(), coverage, sem_agree, mw])
        gate = torch.sigmoid(self.residual_gate(gate_input)).squeeze(0)
        delta = 1.5 * torch.tanh(self.alpha) * gate * torch.tanh(d / .25)
        return delta, ev, diff, mw, hard, coverage, gate

    def forward(self, cam144: torch.Tensor, query_ids: Iterable[int], memory: Dict[str, torch.Tensor],
                support_ids: torch.Tensor, sites: Optional[torch.Tensor], base_score: torch.Tensor,
                labels: Optional[torch.Tensor] = None, keep_mask: Optional[torch.Tensor] = None):
        patient_ids = list(query_ids)
        q = F.normalize(cam144.reshape(-1, 4, 36), dim=-1)
        batch = q.shape[0]
        if isinstance(support_ids, (list, tuple)):
            groups = {}
            for row, candidate_ids in enumerate(support_ids):
                groups.setdefault(id(candidate_ids), [candidate_ids, []])[1].append(row)
            row_aux = {k: [None] * batch for k in ("delta", "evidence", "slot_difference", "modality_weights",
                                                         "hard_similarity", "coverage", "gate")}
            scalar_aux = {k: base_score.sum() * 0 for k in ("proto_loss", "res_loss", "keep_loss")}
            scores = [None] * batch
            for candidate_ids, rows_list in groups.values():
                key = tuple(torch.as_tensor(candidate_ids).detach().cpu().tolist())
                rows = torch.as_tensor(rows_list, dtype=torch.long, device=q.device)
                local_labels = None if labels is None else labels[rows]
                local_keep = None if keep_mask is None else keep_mask[rows]
                piece_scores, piece = self.forward(
                    cam144[rows], [patient_ids[i] for i in rows_list], memory,
                    torch.as_tensor(key, dtype=torch.long, device=q.device), sites, base_score[rows],
                    local_labels, local_keep)
                for j, row in enumerate(rows_list):
                    scores[row] = piece_scores[j]
                    for name in row_aux: row_aux[name][row] = piece[name][j]
                for name in scalar_aux:
                    scalar_aux[name] = scalar_aux[name] + piece[name] * (len(rows_list) / max(1, batch))
            return torch.stack(scores), {**{k: torch.stack(v) for k, v in row_aux.items()}, **scalar_aux}
        support = support_ids.to(q.device)
        sem = self.semantic(self.text_vectors).reshape(2, 4, 36)
        rho = .2 * torch.sigmoid(self.theta)
        evidence, rel_sims, bnd_sims, hard_sims, hard_masks, coverages = [], [], [], [], [], []
        for cls in (0, 1):
            cls_ids = support[memory["labels"][support] == cls]
            pred = memory["probabilities"][cls_ids].argmax(-1)
            py = memory["probabilities"][cls_ids, cls]
            rel_ids = cls_ids[pred == cls]
            rel_weights = torch.zeros_like(memory["probabilities"][:, 0])
            rel_weights[rel_ids] = ((memory["probabilities"][rel_ids, cls] - .5) / .5).clamp(0, 1)
            bnd_weights = torch.zeros_like(memory["probabilities"][:, 0])
            bnd_weights[cls_ids] = 4 * py * (1 - py)
            class_ev, class_rel, class_bnd, class_hard, class_hmask, class_cov = [], [], [], [], [], []
            for m in range(4):
                pr, sr, nr = self._pool_batch(q[:, m], memory, rel_ids, rel_weights, m, 8, sites)
                pb, sb, nb = self._pool_batch(q[:, m], memory, cls_ids, bnd_weights, m, 8, sites)
                a = F.normalize(pr + rho[m] * sem[cls, m], dim=-1)
                b = F.normalize(pb + rho[m] * sem[cls, m], dim=-1)
                class_ev.append(.7 * (q[:, m] * a).sum(-1) + .3 * (q[:, m] * b).sum(-1))
                class_rel.append(sr); class_bnd.append(sb); class_cov.append((nr + nb) / 16.0)
                hard_ids = cls_ids[pred != cls]
                if len(hard_ids) >= 3:
                    hw = torch.zeros_like(memory["probabilities"][:, 0])
                    hw[hard_ids] = memory["probabilities"][hard_ids].max(-1).values
                    _, sh, _ = self._pool_batch(q[:, m], memory, hard_ids, hw, m, 3, sites)
                    class_hmask.append(q.new_ones(batch))
                else:
                    sh = q.new_zeros(batch); class_hmask.append(q.new_zeros(batch))
                class_hard.append(sh)
            evidence.append(torch.stack(class_ev, dim=-1))
            rel_sims.append(torch.stack(class_rel, dim=-1)); bnd_sims.append(torch.stack(class_bnd, dim=-1))
            hard_sims.append(torch.stack(class_hard, dim=-1)); hard_masks.append(torch.stack(class_hmask, dim=-1))
            coverages.append(torch.stack(class_cov, dim=-1))
        ev = torch.stack(evidence, dim=1)
        diff = ev[:, 0] - ev[:, 1]
        rel_diff = rel_sims[0] - rel_sims[1]
        bnd_diff = bnd_sims[0] - bnd_sims[1]
        mw = torch.softmax(self.modality_gate(torch.cat([cam144, diff, rel_diff, bnd_diff], dim=-1)), dim=-1)
        d = (mw * diff).sum(-1)
        disagreement = diff - d[:, None]
        hard = torch.stack(hard_sims, dim=1)
        hard_mask = torch.stack(hard_masks, dim=1)
        coverage = torch.stack(coverages, dim=1).mean(1)
        sem_agree = torch.stack([F.cosine_similarity(q, F.normalize(sem[c], dim=-1)[None], dim=-1).mean(-1)
                                 for c in (0, 1)], dim=-1)
        gate_input = torch.cat([base_score[:, None], base_score.abs()[:, None], ev.flatten(1),
                                disagreement, hard.flatten(1), hard_mask.flatten(1), coverage, sem_agree, mw], dim=-1)
        gates = torch.sigmoid(self.residual_gate(gate_input)).squeeze(-1)
        delta = 1.5 * torch.tanh(self.alpha) * gates * torch.tanh(d / .25)
        scores = base_score + delta
        proto_logits = torch.stack([(mw * ev[:, c]).sum(-1) for c in (0, 1)], dim=-1)
        zero = delta.sum() * 0
        proto_loss = F.cross_entropy(proto_logits, labels) if labels is not None and labels.numel() else zero
        res_loss = delta.square().mean() if delta.numel() else zero
        keep_loss = zero
        if keep_mask is not None and delta.numel():
            km = keep_mask.to(delta.dtype)
            keep_loss = (delta.square() * km).sum() / km.sum().clamp_min(1)
        return base_score + delta, {"delta": delta, "evidence": ev, "slot_difference": diff,
                        "modality_weights": mw, "hard_similarity": hard, "coverage": coverage,
                        "gate": gates, "proto_loss": proto_loss, "res_loss": res_loss,
                        "keep_loss": keep_loss}


def init_module_isolated(text_vectors: torch.Tensor, anonymous: bool = False, seed: int = 765101):
    if anonymous:
        text_vectors = anonymous_pair(text_vectors)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return SCPR(text_vectors)

