from __future__ import annotations

import hashlib
import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def _logit(p):
    return math.log(p / (1 - p))


class LOOFAPC(nn.Module):
    """E78.8 FAPC with vectorized leave-one-out support prototypes."""

    def __init__(self, anchors, classes, mode: str, prompt_source: str):
        super().__init__()
        if mode not in ("visual", "text"):
            raise ValueError(mode)
        self.mode = str(mode)
        self.prompt_source = str(prompt_source)
        self.dim = 32
        self.prompt_count = 5
        self.temperature = 0.25
        self.scale = 10.0
        self.beta_max = 0.2
        self.patient_projection = nn.Linear(144, 32, bias=False)
        self.prototype_direction = nn.Linear(32, 32, bias=False)
        if self.mode == "text":
            if anchors is None or tuple(anchors.shape) != (10, 32):
                raise ValueError("ten [10,32] frozen anchors required")
            if classes is None or tuple(classes.shape) != (10,):
                raise ValueError("ten prompt classes required")
            self.register_buffer("anchors", anchors.detach().float().clone())
            self.register_buffer("prompt_classes", classes.detach().long().clone())
            self.calibration_raw = nn.Parameter(torch.full((10,), _logit(0.5)))
        else:
            self.register_buffer("anchors", torch.empty(0, 32))
            self.register_buffer("prompt_classes", torch.empty(0, dtype=torch.long))

    @property
    def calibration_strength(self):
        if self.mode != "text":
            return self.patient_projection.weight.new_zeros((0,))
        return self.beta_max * torch.sigmoid(self.calibration_raw)

    def anchor_sha(self):
        return hashlib.sha256(self.anchors.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _loo_directions(self, a, labels, support, query):
        support = support.sort().values
        support_y = labels[support]
        if not torch.all((support_y == 0) | (support_y == 1)):
            raise RuntimeError("support labels must be ASD=0 or TD=1")
        support_features = a[support]
        # Avoid CUDA index_add_ atomics: their reduction order can vary and
        # create label-independent 1e-6 drift across otherwise identical calls.
        sums = torch.stack([support_features[support_y == cls].sum(0) for cls in (0, 1)])
        counts = torch.bincount(support_y, minlength=2).to(dtype=a.dtype)
        if bool((counts < 2).any()):
            raise RuntimeError(f"LOO requires at least two supports per class, got {counts.tolist()}")

        # Only labels of members of the legal support set are read here.
        pos = torch.searchsorted(support, query)
        safe_pos = pos.clamp(max=support.numel() - 1)
        member = (pos < support.numel()) & (support[safe_pos] == query)
        own_class = torch.zeros_like(query)
        own_class[member] = support_y[safe_pos[member]]
        remove = F.one_hot(own_class, num_classes=2).to(dtype=a.dtype) * member[:, None].to(a.dtype)
        q_feature = a[query]
        means = (sums[None, :, :] - remove[:, :, None] * q_feature[:, None, :]) / (
            counts[None, :] - remove
        )[:, :, None]
        difference = means[:, 0] - means[:, 1]
        signed = torch.stack((difference, -difference), dim=1)
        directions = unit(self.prototype_direction(unit(signed)))
        return means, directions, member, own_class, counts

    def _semantic(self, zq, dirs, labels, query, class_weights, training, intervention):
        classes = self.prompt_classes
        if intervention == "class_swap":
            classes = 1 - classes
        beta = self.calibration_strength
        if intervention in ("no_calibration", "anchor_only"):
            calibrated = self.anchors.unsqueeze(0).expand(zq.shape[0], -1, -1)
        else:
            prompt_dirs = dirs[:, classes, :]
            calibrated = unit(self.anchors.unsqueeze(0) + beta[None, :, None] * prompt_dirs)
        scores = torch.einsum("qd,qpd->qp", zq, calibrated)
        per_class = []
        for cls in (0, 1):
            class_scores = scores[:, classes == cls]
            pooled = self.temperature * torch.logsumexp(class_scores / self.temperature, dim=-1)
            pooled = pooled - self.temperature * math.log(self.prompt_count)
            per_class.append(pooled)
        pooled_scores = torch.stack(per_class, dim=-1)
        logits = self.scale * pooled_scores
        logp = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(logp, labels[query], weight=class_weights) if training else zq.new_zeros(())
        if intervention in ("no_calibration", "anchor_only"):
            keep = zq.new_zeros(())
        else:
            keep = (1 - (calibrated * self.anchors.unsqueeze(0)).sum(-1)).mean()
        return calibrated, scores, pooled_scores, logits, logp, loss, keep

    def forward(self, h, labels, support_ids, query_ids, class_weights, base_logp,
                *, training, intervention="full") -> Dict[str, torch.Tensor]:
        if tuple(h.shape) != (871, 144):
            raise RuntimeError(f"LOO-FAPC requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        if support.numel() == 0 or torch.unique(support).numel() != support.numel():
            raise RuntimeError("invalid LOO support set")
        u = self.patient_projection(h)
        z = unit(u)
        a = F.layer_norm(u, (self.dim,))
        means, dirs, query_in_support, own_class, counts = self._loo_directions(a, labels, support, query)
        zq = z[query]
        visual_scores = (zq[:, None, :] * dirs).sum(-1)
        visual_logits = self.scale * visual_scores
        visual_logp = F.log_softmax(visual_logits, dim=-1)
        visual_loss = F.nll_loss(visual_logp, labels[query], weight=class_weights) if training else z.new_zeros(())

        if self.mode == "text":
            calibrated, prompt_scores, semantic_scores, semantic_logits, semantic_logp, aux, keep = self._semantic(
                zq, dirs, labels, query, class_weights, training, intervention
            )
            beta = self.calibration_strength
        else:
            calibrated = z.new_zeros((len(query), 10, 32))
            prompt_scores = z.new_zeros((len(query), 10))
            semantic_scores = visual_scores
            semantic_logits, semantic_logp = visual_logits, visual_logp
            aux, keep = visual_loss, z.new_zeros(())
            beta = z.new_zeros((0,))

        directions_cos = dirs @ dirs.transpose(-1, -2)
        return {
            "final_logp": base_logp[query], "final_logits": base_logp[query], "base_logp": base_logp,
            "semantic_logits": semantic_logits, "semantic_logp": semantic_logp,
            "semantic_loss": aux, "classification_loss": aux, "aux_loss": aux, "keep_loss": keep,
            "prototype_means": means, "prototypes": dirs, "prototype_similarity": directions_cos,
            "patient_projection": z, "attention_features": a, "prompt_base": self.anchors,
            "prompt_calibrated": calibrated, "prompt_final": calibrated, "calibration_strength": beta,
            "semantic_scores": semantic_scores, "semantic_margin": semantic_logits[:, 0] - semantic_logits[:, 1],
            "prompt_scores": prompt_scores, "support_ids": support, "support_labels": labels[support],
            "query_ids": query, "prompt_classes": self.prompt_classes, "prompt_source": self.prompt_source,
            "enable_a": True, "enable_b": False, "mode": self.mode,
            "gradient_path_note": "LOO training prototypes; full legal support prototypes at test",
            "query_in_support": query_in_support, "query_support_class": own_class, "support_counts": counts,
            "attention_weights": z.new_zeros((2, 5, 1, max(int(support.numel()), 1))),
            "attention_logits": z.new_zeros((2, 5, 1, max(int(support.numel()), 1))),
            "attention_entropy": z.new_zeros((2, 5, 1)), "patient_teacher": z.new_zeros((h.shape[0], 2)),
            "gamma": z.new_zeros(()), "attention_tau": z.new_zeros(()), "anchor_sha": self.anchor_sha(),
        }


def initialize(anchors, classes, mode, prompt_source, fold, device):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(788000 + int(fold))
        module = LOOFAPC(anchors, classes, mode, prompt_source)
        for param in module.parameters():
            if param.ndim > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        if hasattr(module, "calibration_raw"):
            nn.init.constant_(module.calibration_raw, _logit(0.5))
    return module.to(device)
