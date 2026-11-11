from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def deterministic_assignments(h: torch.Tensor, labels: torch.Tensor, allowed, k: int) -> torch.Tensor:
    """Assign each legal training patient to a stable within-class PC1 quantile."""
    n = h.shape[0]
    assign = np.full(n, -1, dtype=np.int64)
    allowed = np.asarray(allowed, dtype=np.int64)
    y = labels.detach().cpu().numpy()
    x = h.detach().cpu().double().numpy()
    for cls in (0, 1):
        ids = allowed[y[allowed] == cls]
        if len(ids) < max(2, k * 4):
            raise RuntimeError(f"class {cls} has too few legal support patients for K={k}")
        if k == 1:
            assign[ids] = 0
            continue
        centered = x[ids] - x[ids].mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        score = centered @ right[0]
        order = np.lexsort((ids, score))
        for cluster, members in enumerate(np.array_split(ids[order], k)):
            if len(members) < 4:
                raise RuntimeError(f"empty or undersized class={cls} cluster={cluster}")
            assign[members] = cluster
    return torch.as_tensor(assign, dtype=torch.long, device=h.device)


class DCPSS(nn.Module):
    """Differentiable disease prototypes and patient-conditioned prompt selection."""

    def __init__(self, text_embeddings: torch.Tensor, text_classes, k: int):
        super().__init__()
        self.k = int(k)
        self.patient_projection = nn.Linear(144, 32, bias=False)
        self.text_projection = nn.Linear(1024, 32, bias=False)
        self.register_buffer("text_embeddings", text_embeddings.float().detach().clone())
        self.register_buffer("text_classes", torch.as_tensor(text_classes, dtype=torch.long))
        self.alpha = nn.Parameter(torch.tensor(float(torch.atanh(torch.tensor(0.2)))))

    def _query_prototypes(self, q, labels, support_ids, assignments, query_ids):
        n = q.shape[0]
        device = q.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        assignments = assignments.to(device=device, dtype=torch.long)
        labels = labels.to(device=device, dtype=torch.long)
        support_labels = labels[support_ids]
        if len(torch.unique(support_ids)) != len(support_ids):
            raise RuntimeError("duplicate prototype support identities")
        if torch.any(assignments[support_ids] < 0):
            raise RuntimeError("support patient has no fixed prototype cluster")

        flat_key = support_labels * self.k + assignments[support_ids]
        membership = F.one_hot(flat_key, num_classes=2 * self.k).to(q.dtype).transpose(0, 1)
        flat_sum = membership @ q[support_ids]
        flat_count = membership.sum(dim=1)

        query_key = torch.full((n,), -1, device=device, dtype=torch.long)
        query_key[support_ids] = flat_key
        own_key = query_key[query_ids]
        has_own_support = own_key >= 0
        own_mask = F.one_hot(own_key.clamp_min(0), num_classes=2 * self.k).to(q.dtype)
        own_mask = own_mask * has_own_support.to(q.dtype).unsqueeze(-1)
        sums = flat_sum.unsqueeze(0) - own_mask.unsqueeze(-1) * q[query_ids].unsqueeze(1)
        counts = flat_count.unsqueeze(0) - own_mask
        if torch.any(counts <= 0):
            raise RuntimeError("Leave-One-Out produced an empty prototype")
        prototypes = unit((sums / counts.unsqueeze(-1)).reshape(-1, 2, self.k, q.shape[-1]))
        return prototypes, counts.reshape(-1, 2, self.k)

    def forward(
        self,
        h: torch.Tensor,
        base_logp: torch.Tensor,
        labels: torch.Tensor,
        support_ids,
        assignments: torch.Tensor,
        query_ids,
        *,
        semantic_off: bool = False,
        prototype_off: bool = False,
        prompt_swap: bool = False,
    ):
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"DCPSS requires CAM OutputLayer input [871,144], got {tuple(h.shape)}")
        q = unit(self.patient_projection(h))
        text = unit(self.text_projection(self.text_embeddings))
        ids = torch.as_tensor(query_ids, dtype=torch.long, device=h.device)
        classes = (1 - self.text_classes) if prompt_swap else self.text_classes

        if prototype_off:
            prototypes = q.new_zeros((len(ids), 2, self.k, 32))
            counts = q.new_ones((len(ids), 2, self.k))
            beta = q.new_zeros((len(ids), 2, self.k))
            v = q.new_zeros((len(ids), 2, 32))
            contrast = q.new_zeros((len(ids), len(text)))
        else:
            prototypes, counts = self._query_prototypes(q, labels, support_ids, assignments, ids)
            sim = torch.einsum("qd,qckd->qck", q[ids], prototypes)
            beta = torch.softmax(sim / 0.2, dim=-1)
            v = unit((beta.unsqueeze(-1) * prototypes).sum(dim=2))
            class_prompt = torch.einsum("qcd,jd->qcj", v, text)
            prompt_index = torch.arange(len(text), device=h.device)
            opposite = 1 - classes
            contrast = class_prompt[:, classes, prompt_index] - class_prompt[:, opposite, prompt_index]

        direct = q[ids] @ text.T
        evidence_by_prompt = direct + contrast
        if semantic_off:
            evidence_by_prompt = evidence_by_prompt * 0.0

        class_evidence = []
        attention = []
        for cls in (0, 1):
            prompt_ids = torch.nonzero(classes == cls, as_tuple=False).flatten()
            weights = torch.softmax(evidence_by_prompt[:, prompt_ids] / 0.5, dim=-1)
            class_evidence.append((weights * evidence_by_prompt[:, prompt_ids]).sum(dim=-1))
            attention.append(weights)
        evidence = torch.stack(class_evidence, dim=1)
        signed = torch.tanh(evidence[:, 0] - evidence[:, 1])
        gamma = 0.5 * torch.tanh(self.alpha)
        delta = gamma * signed
        corrected_logits = base_logp[ids] + torch.stack((delta / 2.0, -delta / 2.0), dim=1)
        final_logp = F.log_softmax(corrected_logits, dim=1)
        return {
            "logp": final_logp,
            "evidence": evidence,
            "attention": attention,
            "prototype": prototypes,
            "counts": counts,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
            "semantic_raw": evidence_by_prompt,
            "query_ids": ids,
        }

    def losses(self, output, base_logp, labels, weights):
        ids = output["query_ids"]
        y = labels[ids]
        final = output["logp"]
        cls = F.nll_loss(final, y, weight=weights)
        semantic = F.cross_entropy(output["evidence"], y, weight=weights)
        base_nll = F.nll_loss(base_logp[ids], y, reduction="none").detach()
        final_nll = F.nll_loss(final, y, reduction="none")
        gain = F.softplus((final_nll - base_nll) / 0.25).mean()
        total = cls + 0.05 * semantic + 0.02 * gain
        return total, {"cls": cls, "sem": semantic, "gain": gain, "total": total}


def initialize(text_embeddings, text_classes, k, fold, device):
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(780000 + int(fold))
        module = DCPSS(text_embeddings, text_classes, k)
    return module.to(device)
