from __future__ import annotations

import hashlib
import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class PSPAL(nn.Module):
    """Prior-guided Semantic–Patient Prototype Aggregation Learning.

    All groups instantiate the same two-candidate module. K=1 uses a strict
    candidate mask and therefore has the same parameter count as K=2. All
    attention tensors are batch-first and have explicit [batch, query, dim] /
    [batch, patient, dim] layouts; no patient/class/head axes are mixed.
    """

    def __init__(self, text_coords: torch.Tensor, prompt_classes: torch.Tensor,
                 k: int, prompt_source: str, classification_mode: str):
        super().__init__()
        if int(k) not in (1, 2):
            raise ValueError("PSP-AL formal capacity must be K=1 or K=2")
        if classification_mode not in ("restricted_residual", "direct"):
            raise ValueError(classification_mode)
        self.k_active = int(k)
        self.prompt_source = str(prompt_source)
        self.classification_mode = str(classification_mode)
        self.max_k = 2
        self.dim = 32
        self.heads = 4
        self.iterations = 3
        self.pool_temperature = 0.25
        self.residual_temperature = 0.5

        self.text_projection = nn.Linear(16, self.dim, bias=False)
        self.patient_projection = nn.Linear(144, self.dim, bias=False)
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=self.heads, dropout=0.0, batch_first=True
        )
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=self.heads, dropout=0.0, batch_first=True
        )
        self.text_to_prototype = nn.Linear(self.dim, self.dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, self.dim),
        )
        self.norm = nn.LayerNorm(self.dim)
        # The offsets are the only free query-specific parameters. Their base
        # is the current projected class text, so the prior remains present in
        # every forward pass rather than only at initialization.
        self.query_offset = nn.Parameter(torch.empty(2, self.max_k, self.dim))
        self.alpha_visual_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        self.alpha_text_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        # Restricted residual amplitude: 0.5*sigmoid(raw), initialized to 0.1.
        self.lambda_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        self.register_buffer("text_coords", text_coords.detach().float().clone())
        self.register_buffer("prompt_classes", prompt_classes.detach().long().clone())
        candidate_mask = torch.zeros(2, self.max_k, dtype=torch.bool)
        candidate_mask[:, : self.k_active] = True
        self.register_buffer("candidate_mask", candidate_mask)
        self.eps = 1e-8

    @property
    def lambda_value(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.lambda_raw)

    @property
    def alpha_visual(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.alpha_visual_raw)

    @property
    def alpha_text(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.alpha_text_raw)

    def query_digest(self) -> str:
        return hashlib.sha256(self.query_offset.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _prompt_tokens(self, intervention: str) -> Tuple[torch.Tensor, torch.Tensor]:
        bank = 1 if (self.prompt_source == "anonymous" or intervention == "anonymous") else 0
        classes = self.prompt_classes
        if intervention == "prompt_class_swap":
            classes = 1 - classes
        tokens = self.text_projection(self.text_coords[bank])
        return tokens, classes

    def _aggregate_class(self, q: torch.Tensor, support: torch.Tensor,
                         text: torch.Tensor, uniform_attention: bool):
        """Three shared-parameter progressive attention updates for one class.

        q=[K,32], support=[Nclass,32], text=[4,32]. Attention is always
        evaluated with explicit batch-first tensors [1,K,32], [1,Nclass,32]
        and [1,4,32].
        """
        proto_weights = None
        text_weights = None
        for _ in range(self.iterations):
            if uniform_attention:
                patient_mean = support.mean(dim=0, keepdim=True)
                visual_update = patient_mean.expand(q.shape[0], -1)
                text_patient_mean = patient_mean.expand(text.shape[0], -1)
                text_update = self.text_to_prototype(text_patient_mean).mean(dim=0, keepdim=True)
                text_update = text_update.expand(q.shape[0], -1)
                proto_weights = torch.full(
                    (q.shape[0], support.shape[0]), 1.0 / float(support.shape[0]),
                    dtype=q.dtype, device=q.device,
                )
                text_weights = torch.full(
                    (text.shape[0], support.shape[0]), 1.0 / float(support.shape[0]),
                    dtype=q.dtype, device=q.device,
                )
            else:
                visual_update, proto_weights = self.prototype_attention(
                    q.unsqueeze(0), support.unsqueeze(0), support.unsqueeze(0),
                    need_weights=True, average_attn_weights=False,
                )
                visual_update = visual_update.squeeze(0)
                # [1, heads, 4, N] -> [4, N]
                proto_weights = proto_weights.mean(dim=1).squeeze(0)
                text_update_raw, text_weights = self.text_attention(
                    text.unsqueeze(0), support.unsqueeze(0), support.unsqueeze(0),
                    need_weights=True, average_attn_weights=False,
                )
                text_update_raw = text_update_raw.squeeze(0)
                text_weights = text_weights.mean(dim=1).squeeze(0)
                text_update = self.text_to_prototype(text_update_raw.mean(dim=0, keepdim=True))
                text_update = text_update.expand(q.shape[0], -1)
            q = self.norm(q + self.alpha_visual * visual_update + self.alpha_text * text_update)
            q = self.norm(q + self.ffn(q))
        return unit(q), proto_weights, text_weights

    def _pooled_scores(self, similarity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # similarity [N,2,2]; masked logmeanexp includes -log(K), so K=2 has
        # no numerical count advantage over K=1.
        masked = similarity.masked_fill(~mask[None, :, :], float("-inf"))
        counts = mask.to(similarity.dtype).sum(dim=-1).clamp_min(1.0)
        return self.pool_temperature * (
            torch.logsumexp(masked / self.pool_temperature, dim=-1) - counts.log()[None, :]
        )

    def forward(self, h: torch.Tensor, labels: torch.Tensor, support_ids,
                query_ids, class_weights: torch.Tensor, base_logp: torch.Tensor,
                *, training: bool, intervention: str = "full") -> Dict[str, torch.Tensor]:
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"PSP-AL requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        if len(torch.unique(support_ids)) != len(support_ids):
            raise RuntimeError("prototype support identities are not unique")
        if torch.any((labels[support_ids] < 0) | (labels[support_ids] > 1)):
            raise RuntimeError("support labels must use ASD=0 and TD=1")
        if torch.any(query_ids < 0) or torch.any(query_ids >= h.shape[0]):
            raise RuntimeError("query identities outside the 871-patient feature bank")

        text_tokens, prompt_classes = self._prompt_tokens(intervention)
        patient = unit(self.patient_projection(h))
        class_prototypes = []
        proto_attention = torch.zeros((2, self.max_k, h.shape[0]), dtype=h.dtype, device=device)
        text_attention = torch.zeros((2, 4, h.shape[0]), dtype=h.dtype, device=device)
        prompt_indices = []
        for cls in (0, 1):
            support_mask = labels[support_ids] == cls
            ids = support_ids[support_mask]
            if ids.numel() == 0:
                raise RuntimeError(f"class {cls} has no legal support patients")
            support = patient[ids]
            prompt_idx = torch.nonzero(prompt_classes == cls, as_tuple=False).flatten()
            if prompt_idx.numel() != 4:
                raise RuntimeError("each class must have exactly four prompts")
            prompt_indices.append(prompt_idx)
            class_text = text_tokens[prompt_idx]
            q0 = class_text.mean(dim=0, keepdim=True).expand(self.max_k, -1) + self.query_offset[cls]
            q, pw, tw = self._aggregate_class(
                q0, support, class_text, intervention == "uniform_attention"
            )
            class_prototypes.append(q)
            proto_attention[cls][:, ids] = pw
            text_attention[cls][:, ids] = tw
        prototypes = unit(torch.stack(class_prototypes, dim=0))
        prototypes = prototypes * self.candidate_mask[..., None].to(prototypes.dtype)

        query_z = patient[query_ids]
        similarity = torch.einsum("nd,ckd->nck", query_z, prototypes)
        active_mask = self.candidate_mask
        if intervention == "prototype2_off":
            active_mask = self.candidate_mask.clone()
            active_mask[:, 1] = False
        pooled = self._pooled_scores(similarity, active_mask)
        if self.classification_mode == "direct":
            final_logits = 10.0 * pooled
        else:
            d = pooled[:, 0] - pooled[:, 1]
            delta = self.lambda_value * torch.tanh(d / self.residual_temperature)
            final_logits = base_logp[query_ids] + torch.stack((delta / 2.0, -delta / 2.0), dim=1)
        final_logp = F.log_softmax(final_logits, dim=-1)

        if training:
            # Formal training queries are exactly legal support patients. No
            # leave-one-out is used: each patient may contribute to its class
            # support aggregation and remain a classification query.
            if not torch.all(torch.isin(query_ids, support_ids)):
                raise RuntimeError("new PSP-AL classification supervision contains non-support rows")
            cls_loss = F.nll_loss(final_logp, labels[query_ids], weight=class_weights)
        else:
            cls_loss = h.new_zeros(())

        similarity_margin = pooled[:, 0] - pooled[:, 1]
        return {
            "final_logp": final_logp,
            "final_logits": final_logits,
            "prototype_scores": pooled,
            "prototype_similarity": similarity,
            "prototypes": prototypes,
            "patient_projection": patient,
            "proto_attention": proto_attention,
            "text_attention": text_attention,
            "support_ids": support_ids,
            "support_labels": labels[support_ids],
            "query_ids": query_ids,
            "prompt_indices": prompt_indices,
            "classification_loss": cls_loss,
            "similarity_margin": similarity_margin,
            "lambda_value": self.lambda_value,
            "alpha_visual": self.alpha_visual,
            "alpha_text": self.alpha_text,
            "classification_mode": self.classification_mode,
            "candidate_mask": active_mask,
            "prompt_source": self.prompt_source,
            "intervention": intervention,
        }


def initialize(text_coords: torch.Tensor, prompt_classes: torch.Tensor, k: int,
               prompt_source: str, classification_mode: str, fold: int,
               device: torch.device) -> PSPAL:
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(784000 + int(fold))
        module = PSPAL(text_coords, prompt_classes, k, prompt_source, classification_mode)
        nn.init.normal_(module.query_offset, mean=0.0, std=0.01)
    return module.to(device)
