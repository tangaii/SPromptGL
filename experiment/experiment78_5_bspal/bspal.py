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


class BSPAL(nn.Module):
    """Baseline-preserving selective prototype aggregation.

    ``selective=False`` is the E78.4 K=1 mechanism copied as an explicit
    control. ``selective=True`` separates LayerNorm attention features from
    L2-normalized classification features and keeps four prompt evidences
    separate until prototype-conditioned weighting.
    """

    def __init__(self, text_coords: torch.Tensor, prompt_classes: torch.Tensor,
                 prompt_source: str, selective: bool):
        super().__init__()
        self.prompt_source = str(prompt_source)
        self.selective = bool(selective)
        self.dim = 32
        self.heads = 4
        self.head_dim = 8
        self.iterations = 3
        self.prompt_temperature = 0.5
        self.residual_temperature = 0.5

        self.text_projection = nn.Linear(16, self.dim, bias=False)
        self.patient_projection = nn.Linear(144, self.dim, bias=False)
        self.text_to_prototype = nn.Linear(self.dim, self.dim, bias=False)
        self.ffn = nn.Sequential(nn.Linear(self.dim, 2 * self.dim), nn.GELU(), nn.Linear(2 * self.dim, self.dim))
        self.norm = nn.LayerNorm(self.dim)
        self.query_offset = nn.Parameter(torch.empty(2, self.dim))
        self.alpha_visual_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        self.alpha_text_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        self.beta_raw = nn.Parameter(torch.tensor(_logit(0.2)))
        self.lambda_raw = nn.Parameter(torch.tensor(_logit(0.2)))

        # G1 must retain the historical E78.4 attention implementation.
        self.proto_attention_legacy = nn.MultiheadAttention(self.dim, self.heads, dropout=0.0, batch_first=True)
        self.text_attention_legacy = nn.MultiheadAttention(self.dim, self.heads, dropout=0.0, batch_first=True)

        # G2/G3 use explicit scaled cosine multi-head attention. These layers
        # are instantiated for every group so optimizer/state schemas remain
        # stable; G1 never routes through them.
        self.sel_q = nn.Linear(self.dim, self.dim)
        self.sel_k = nn.Linear(self.dim, self.dim)
        self.sel_v = nn.Linear(self.dim, self.dim)
        self.sel_o = nn.Linear(self.dim, self.dim)
        self.sel_text_q = nn.Linear(self.dim, self.dim)
        self.sel_text_k = nn.Linear(self.dim, self.dim)
        self.sel_text_v = nn.Linear(self.dim, self.dim)
        self.sel_text_o = nn.Linear(self.dim, self.dim)
        self.tau_attn_raw = nn.Parameter(torch.tensor(_logit((0.1 - 0.05) / 0.45)))

        self.register_buffer("text_coords", text_coords.detach().float().clone())
        self.register_buffer("prompt_classes", prompt_classes.detach().long().clone())
        self.eps = 1e-8

    @property
    def lambda_value(self):
        return 0.5 * torch.sigmoid(self.lambda_raw)

    @property
    def alpha_visual(self):
        return 0.5 * torch.sigmoid(self.alpha_visual_raw)

    @property
    def alpha_text(self):
        return 0.5 * torch.sigmoid(self.alpha_text_raw)

    @property
    def beta_value(self):
        return 0.5 * torch.sigmoid(self.beta_raw)

    @property
    def tau_attn(self):
        return 0.05 + 0.45 * torch.sigmoid(self.tau_attn_raw)

    def query_digest(self):
        return hashlib.sha256(self.query_offset.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _prompt_tokens(self, intervention: str):
        bank = 1 if (self.prompt_source == "anonymous" or intervention == "anonymous") else 0
        classes = self.prompt_classes
        if intervention == "prompt_class_swap":
            classes = 1 - classes
        return self.text_projection(self.text_coords[bank]), classes

    def _split_heads(self, x):
        # [N,D] -> [heads,N,head_dim]
        return x.reshape(x.shape[0], self.heads, self.head_dim).transpose(0, 1)

    def _selective_attention(self, query, support, kind: str):
        if kind == "proto":
            q, k, v, out = self.sel_q, self.sel_k, self.sel_v, self.sel_o
        else:
            q, k, v, out = self.sel_text_q, self.sel_text_k, self.sel_text_v, self.sel_text_o
        qh = self._split_heads(q(query)); kh = self._split_heads(k(support)); vh = self._split_heads(v(support))
        scores = torch.einsum("hqd,hnd->hqn", unit(qh), unit(kh)) / self.tau_attn
        weights = torch.softmax(scores, dim=-1)
        value = torch.einsum("hqn,hnd->hqd", weights, vh)
        value = value.transpose(0, 1).contiguous().reshape(query.shape[0], self.dim)
        return out(value), weights.mean(dim=0), scores.mean(dim=0)

    def _aggregate_class_legacy(self, q, support, text, uniform_attention):
        proto_weights = text_weights = None
        prompt_weights = torch.full((4,), 0.25, dtype=q.dtype, device=q.device)
        logits = []
        for _ in range(self.iterations):
            if uniform_attention:
                visual_update = support.mean(0, keepdim=True).expand_as(q)
                text_update = self.text_to_prototype(support.mean(0, keepdim=True)).expand_as(q)
                proto_weights = torch.full((q.shape[0], support.shape[0]), 1 / support.shape[0], dtype=q.dtype, device=q.device)
                text_weights = torch.full((4, support.shape[0]), 1 / support.shape[0], dtype=q.dtype, device=q.device)
                logits.append(torch.zeros_like(proto_weights))
            else:
                visual_update, pw = self.proto_attention_legacy(q[None], support[None], support[None], need_weights=True, average_attn_weights=False)
                text_update_raw, tw = self.text_attention_legacy(text[None], support[None], support[None], need_weights=True, average_attn_weights=False)
                visual_update = visual_update.squeeze(0)
                proto_weights = pw.mean(1).squeeze(0)
                text_weights = tw.mean(1).squeeze(0)
                text_update = self.text_to_prototype(text_update_raw.squeeze(0).mean(0, keepdim=True)).expand_as(q)
                logits.append(torch.log(proto_weights.clamp_min(1e-12)))
            q = self.norm(q + self.alpha_visual * visual_update + self.alpha_text * text_update)
            q = self.norm(q + self.ffn(q))
        return unit(q), proto_weights, text_weights, prompt_weights, torch.cat(logits, dim=0) if logits else q.new_zeros((0, support.shape[0]))

    def _aggregate_class_selective(self, q, support_a, text, uniform_attention):
        proto_weights = text_weights = None
        prompt_weights = torch.full((4,), 0.25, dtype=q.dtype, device=q.device)
        score_records = []
        for _ in range(self.iterations):
            if uniform_attention:
                proto_weights = torch.full((q.shape[0], support_a.shape[0]), 1 / support_a.shape[0], dtype=q.dtype, device=q.device)
                text_weights = torch.full((4, support_a.shape[0]), 1 / support_a.shape[0], dtype=q.dtype, device=q.device)
                visual_update = self.sel_o(self.sel_v(support_a).mean(0, keepdim=True).expand_as(q))
                text_raw = self.sel_text_o(self.sel_text_v(support_a).mean(0, keepdim=True).expand(4, -1))
                scores = q.new_zeros((q.shape[0], support_a.shape[0]))
            else:
                visual_update, proto_weights, scores = self._selective_attention(q, support_a, "proto")
                text_raw, text_weights, text_scores = self._selective_attention(text, support_a, "text")
                score_records.append(scores)
            # Keep each prompt's original semantic direction while adding its
            # own patient evidence; only then select among the four prompts.
            prompt_evidence = unit(text + self.beta_value * text_raw)
            prompt_score = torch.einsum("kd,pd->kp", unit(q), prompt_evidence).mean(0) / self.prompt_temperature
            prompt_weights = torch.softmax(prompt_score, dim=-1)
            text_update = (prompt_weights[:, None] * prompt_evidence).sum(0, keepdim=True).expand_as(q)
            q = self.norm(q + self.alpha_visual * visual_update + self.alpha_text * text_update)
            q = self.norm(q + self.ffn(q))
        return unit(q), proto_weights, text_weights, prompt_weights, torch.cat(score_records, dim=0) if score_records else q.new_zeros((0, support_a.shape[0]))

    def forward(self, h: torch.Tensor, labels: torch.Tensor, support_ids, query_ids,
                class_weights: torch.Tensor, base_logp: torch.Tensor, *, training: bool,
                intervention: str = "full") -> Dict[str, torch.Tensor]:
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"BSP-AL requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        if len(torch.unique(support_ids)) != len(support_ids): raise RuntimeError("duplicate support identities")
        if torch.any((labels[support_ids] < 0) | (labels[support_ids] > 1)): raise RuntimeError("invalid support labels")
        text_tokens, prompt_classes = self._prompt_tokens(intervention)
        u = self.patient_projection(h)
        z = unit(u)
        attention_patient = self.norm(u) if self.selective else z
        prototypes, proto_attention, text_attention = [], torch.zeros((2, 1, h.shape[0]), dtype=h.dtype, device=device), torch.zeros((2, 4, h.shape[0]), dtype=h.dtype, device=device)
        prompt_weights_all = torch.zeros((2, 4), dtype=h.dtype, device=device)
        attention_logits = torch.zeros((2, h.shape[0]), dtype=h.dtype, device=device)
        prompt_indices = []
        for cls in (0, 1):
            ids = support_ids[labels[support_ids] == cls]
            prompt_idx = torch.nonzero(prompt_classes == cls, as_tuple=False).flatten()
            if ids.numel() == 0 or prompt_idx.numel() != 4: raise RuntimeError("invalid class support or prompt mapping")
            prompt_indices.append(prompt_idx)
            text = text_tokens[prompt_idx]
            q0 = text.mean(0, keepdim=True) + self.query_offset[cls:cls+1]
            if self.selective:
                q, pw, tw, pweights, slogits = self._aggregate_class_selective(q0, attention_patient[ids], text, intervention == "uniform_attention")
            else:
                q, pw, tw, pweights, slogits = self._aggregate_class_legacy(q0, attention_patient[ids], text, intervention == "uniform_attention")
            prototypes.append(q[0])
            proto_attention[cls, 0, ids] = pw[0]
            text_attention[cls, :, ids] = tw
            prompt_weights_all[cls] = pweights
            if slogits.numel():
                attention_logits[cls, ids] = slogits[-1]
        prototypes = unit(torch.stack(prototypes, 0))
        patient_z = z[query_ids]
        similarity = torch.einsum("nd,cd->nc", patient_z, prototypes)
        delta = self.lambda_value * torch.tanh((similarity[:, 0] - similarity[:, 1]) / self.residual_temperature)
        if intervention == "module_off":
            final_logits = base_logp[query_ids]
        else:
            final_logits = base_logp[query_ids] + torch.stack((delta / 2, -delta / 2), dim=1)
        final_logp = F.log_softmax(final_logits, dim=-1)
        if training:
            if not torch.all(torch.isin(query_ids, support_ids)): raise RuntimeError("training query outside support")
            cls_loss = F.nll_loss(final_logp, labels[query_ids], weight=class_weights)
        else:
            cls_loss = h.new_zeros(())
        valid_attention = proto_attention[:, 0]
        entropy = -(valid_attention.clamp_min(1e-12) * valid_attention.clamp_min(1e-12).log()).sum(-1)
        return {
            "final_logp": final_logp, "final_logits": final_logits, "prototype_scores": similarity,
            "prototype_similarity": similarity, "prototypes": prototypes, "patient_projection": z,
            "attention_features": attention_patient, "proto_attention": proto_attention, "text_attention": text_attention,
            "attention_logits": attention_logits, "attention_entropy": entropy, "prompt_weights": prompt_weights_all,
            "support_ids": support_ids, "support_labels": labels[support_ids], "query_ids": query_ids,
            "prompt_indices": prompt_indices, "classification_loss": cls_loss, "similarity_margin": similarity[:,0]-similarity[:,1],
            "lambda_value": self.lambda_value, "alpha_visual": self.alpha_visual, "alpha_text": self.alpha_text,
            "beta_value": self.beta_value, "tau_attn": self.tau_attn, "classification_mode": "restricted_residual",
            "prompt_source": self.prompt_source, "intervention": intervention, "selective": self.selective,
        }


def initialize(text_coords, prompt_classes, prompt_source, selective, fold, device):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(785000 + int(fold))
        module = BSPAL(text_coords, prompt_classes, prompt_source, selective)
        nn.init.normal_(module.query_offset, mean=0.0, std=0.01)
    return module.to(device)
