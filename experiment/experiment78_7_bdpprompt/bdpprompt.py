from __future__ import annotations

import hashlib
import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class BDPPL(nn.Module):
    """Bidirectional disease-prototype guided Prompt learning.

    The module keeps the DARG logits as the final prediction.  Its semantic
    branch is an auxiliary, signed ASD-vs-TD objective.  All groups instantiate
    the same parameter schema; flags only select the registered direction-A and
    direction-B computations.
    """

    def __init__(self, text_embeddings: torch.Tensor, prompt_classes: torch.Tensor,
                 prompt_source: str, enable_a: bool, enable_b: bool):
        super().__init__()
        if tuple(text_embeddings.shape) != (10, 1024):
            raise ValueError(f"BDP-PL expects ten RN50 text vectors [10,1024], got {tuple(text_embeddings.shape)}")
        if tuple(prompt_classes.shape) != (10,) or not torch.equal(torch.sort(prompt_classes.detach().cpu()).values, torch.tensor([0]*5+[1]*5)):
            raise ValueError("BDP-PL requires five ASD and five TD prompts")
        self.prompt_source = str(prompt_source)
        self.enable_a = bool(enable_a)
        self.enable_b = bool(enable_b)
        self.dim = 32
        self.prompt_count = 5
        self.heads = 4
        self.head_dim = 8
        self.iterations = 3
        self.attention_tau = 0.1
        self.pool_temperature = 0.25
        self.semantic_scale = 10.0

        self.patient_projection = nn.Linear(144, self.dim, bias=False)
        self.text_projection = nn.Linear(1024, self.dim, bias=False)
        self.prototype_direction = nn.Linear(self.dim, self.dim, bias=False)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.update_norm = nn.LayerNorm(self.dim)
        self.update_ffn = nn.Sequential(
            nn.Linear(self.dim, 2*self.dim), nn.GELU(), nn.Linear(2*self.dim, self.dim)
        )
        # independent bounded strength for each of the ten prompts
        self.calibration_raw = nn.Parameter(torch.zeros(10))
        self.gamma_raw = nn.Parameter(torch.tensor(_logit(0.5)))

        self.register_buffer("text_embeddings", text_embeddings.detach().float().clone())
        self.register_buffer("prompt_classes", prompt_classes.detach().long().clone())

    @property
    def calibration_strength(self):
        return 0.2 * torch.sigmoid(self.calibration_raw)

    @property
    def gamma(self):
        return 0.2 * torch.sigmoid(self.gamma_raw)

    def query_digest(self):
        return hashlib.sha256(self.text_embeddings.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _tokens(self, intervention: str):
        classes = self.prompt_classes
        tokens = self.text_projection(self.text_embeddings)
        if intervention == "prompt_class_swap":
            classes = 1 - classes
        if intervention == "anonymous":
            # This is only a fixed-checkpoint intervention.  The training
            # group itself uses the bank selected by prompt_source.
            tokens = tokens.detach()
        return tokens, classes

    def _attention(self, queries: torch.Tensor, support_a: torch.Tensor,
                   teacher: torch.Tensor, cls: int, uniform: bool = False):
        # queries [P,D], support_a [S,D], teacher [S], output [P,D].
        P, S = queries.shape[0], support_a.shape[0]
        q = self.q_proj(queries).reshape(P, self.heads, self.head_dim)
        k = self.k_proj(support_a).reshape(S, self.heads, self.head_dim)
        v = self.v_proj(support_a).reshape(S, self.heads, self.head_dim)
        if uniform:
            alpha = torch.full((P, self.heads, S), 1.0 / max(S, 1), dtype=queries.dtype, device=queries.device)
            logits = torch.zeros_like(alpha)
        else:
            q = unit(q); k = unit(k)
            logits = torch.einsum("phd,shd->phs", q, k) / self.attention_tau
            logits = logits + teacher.clamp_min(1e-8).log()[None, None, :]
            alpha = torch.softmax(logits, dim=-1)
        evidence = torch.einsum("phs,shd->phd", alpha, v).reshape(P, self.dim)
        evidence = self.o_proj(evidence)
        return evidence, alpha, logits

    def _class_tokens(self, raw_tokens: torch.Tensor, classes: torch.Tensor,
                      cls: int, attention_a: torch.Tensor, z_all: torch.Tensor,
                      labels: torch.Tensor, support_ids: torch.Tensor,
                      teacher: torch.Tensor, uniform_attention: bool):
        idx = torch.nonzero(classes == cls, as_tuple=False).flatten()
        if idx.numel() != self.prompt_count:
            raise RuntimeError(f"class {cls} must have five prompts, got {idx.numel()}")
        base = unit(raw_tokens[idx])
        calibrated = base
        calibration = self.calibration_strength[idx]
        if self.enable_a:
            ids_cls = support_ids[labels[support_ids] == cls]
            ids_opp = support_ids[labels[support_ids] == (1-cls)]
            if ids_cls.numel() == 0 or ids_opp.numel() == 0:
                raise RuntimeError("direction-A requires non-empty ASD/TD support")
            mu_cls = attention_a[ids_cls].mean(0)
            mu_opp = attention_a[ids_opp].mean(0)
            d = unit(mu_cls - mu_opp)
            direction = unit(self.prototype_direction(d))
            calibrated = unit(base + calibration[:, None] * direction[None, :])
        q = calibrated
        base_keep = (1.0 - (unit(calibrated) * unit(base)).sum(-1).mean()) if self.enable_a else calibrated.new_zeros(())
        records = []
        for _ in range(self.iterations):
            if self.enable_b:
                # Direction B intentionally uses every legal support patient;
                # labels only enter the prompt grouping for direction A.
                teacher_c = teacher[support_ids, cls]
                evidence, alpha, logits = self._attention(q, attention_a[support_ids], teacher_c, cls,
                                                          uniform=uniform_attention)
                q = self.update_norm(q + self.gamma * evidence)
                q = unit(q + 0.1 * self.update_ffn(q))
                records.append((alpha, logits))
            else:
                S = support_ids.numel()
                alpha = torch.full((self.prompt_count, self.heads, S), 1.0 / max(S, 1), dtype=q.dtype, device=q.device)
                logits = torch.zeros_like(alpha)
                records.append((alpha, logits))
        # semantic keep compares the A-calibrated prompt to the original text;
        # aggregation B remains free to add patient evidence.
        return q, base, calibrated, calibration, records, base_keep

    def forward(self, h: torch.Tensor, labels: torch.Tensor, support_ids, query_ids,
                class_weights: torch.Tensor, base_logp: torch.Tensor, *, training: bool,
                intervention: str = "full") -> Dict[str, torch.Tensor]:
        if h.ndim != 2 or tuple(h.shape) != (871, 144):
            raise RuntimeError(f"BDP-PL requires CAM144 [871,144], got {tuple(h.shape)}")
        device = h.device
        support_ids = torch.as_tensor(support_ids, dtype=torch.long, device=device).sort().values
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        labels = labels.to(device=device, dtype=torch.long)
        if support_ids.numel() == 0 or len(torch.unique(support_ids)) != support_ids.numel():
            raise RuntimeError("support identities must be non-empty and unique")
        if torch.any((labels[support_ids] < 0) | (labels[support_ids] > 1)):
            raise RuntimeError("invalid support labels")
        raw_tokens, classes = self._tokens(intervention)
        # The fixed intervention bank is supplied by the runner as a separate
        # module when needed; this flag only swaps class ownership.
        u = self.patient_projection(h)
        z = unit(u)
        attention_a = self.update_norm(u)
        teacher = torch.softmax(base_logp.detach(), dim=-1)
        prototypes = []
        all_alpha = torch.zeros((2, self.prompt_count, self.heads, h.shape[0]), dtype=h.dtype, device=device)
        all_logits = torch.zeros_like(all_alpha)
        all_base = torch.zeros((10, self.dim), dtype=h.dtype, device=device)
        all_cal = torch.zeros_like(all_base)
        all_final = torch.zeros_like(all_base)
        calibration_values = torch.zeros((10,), dtype=h.dtype, device=device)
        keep_terms = []
        for cls in (0, 1):
            q, base, calibrated, strength, records, keep = self._class_tokens(
                raw_tokens, classes, cls, attention_a, z, labels, support_ids, teacher,
                intervention == "uniform_attention")
            prototypes.append(unit(q.mean(0)))
            idx = torch.nonzero(classes == cls, as_tuple=False).flatten()
            all_base[idx] = base; all_cal[idx] = calibrated; all_final[idx] = q; calibration_values[idx] = strength
            keep_terms.append(keep)
            for step, (alpha, logits) in enumerate(records):
                all_alpha[cls][:, :, support_ids] = alpha
                all_logits[cls][:, :, support_ids] = logits
        prototypes = unit(torch.stack(prototypes, 0))
        similarity = torch.einsum("nd,cd->nc", z[query_ids], prototypes)
        # Prompt-level signed semantics: five prompts are retained until the
        # normalized log-mean-exp; no max-softmax loses class direction.
        q_by_class = [all_final[classes == cls] for cls in (0, 1)]
        per_prompt = torch.stack([
            torch.einsum("nd,pd->np", z[query_ids], q_by_class[0]),
            torch.einsum("nd,pd->np", z[query_ids], q_by_class[1]),
        ], dim=1)
        sem_scores = self.pool_temperature * torch.logsumexp(per_prompt / self.pool_temperature, dim=-1) - self.pool_temperature * math.log(self.prompt_count)
        semantic_logits = self.semantic_scale * sem_scores
        semantic_logp = F.log_softmax(semantic_logits, dim=-1)
        if training:
            if not torch.all(torch.isin(query_ids, support_ids)):
                raise RuntimeError("training query outside legal support")
            semantic_loss = F.nll_loss(semantic_logp, labels[query_ids], weight=class_weights)
        else:
            semantic_loss = h.new_zeros(())
        keep_loss = torch.stack(keep_terms).mean() if self.enable_a else h.new_zeros(())
        # DARG is final: this branch never replaces or adds a logit residual.
        final_logp = base_logp[query_ids]
        entropy = -(all_alpha.clamp_min(1e-12) * all_alpha.clamp_min(1e-12).log()).sum(-1)
        return {
            "final_logp": final_logp, "final_logits": base_logp[query_ids],
            "semantic_logits": semantic_logits, "semantic_logp": semantic_logp,
            "semantic_loss": semantic_loss, "classification_loss": semantic_loss,
            "keep_loss": keep_loss, "prototype_similarity": similarity,
            "prototypes": prototypes, "patient_projection": z,
            "attention_features": attention_a, "patient_teacher": teacher,
            "attention_weights": all_alpha, "attention_logits": all_logits,
            "attention_entropy": entropy, "prompt_base": all_base,
            "prompt_calibrated": all_cal, "prompt_final": all_final,
            "calibration_strength": calibration_values, "semantic_margin": semantic_logits[:, 0] - semantic_logits[:, 1],
            "support_ids": support_ids, "support_labels": labels[support_ids], "query_ids": query_ids,
            "prompt_classes": classes, "prompt_source": self.prompt_source,
            "enable_a": self.enable_a, "enable_b": self.enable_b,
            "gamma": self.gamma, "attention_tau": h.new_tensor(self.attention_tau),
            "semantic_scores": sem_scores, "base_logp": base_logp,
        }


def initialize(text_embeddings, prompt_classes, prompt_source, enable_a, enable_b, fold, device):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(787000 + int(fold))
        module = BDPPL(text_embeddings, prompt_classes, prompt_source, enable_a, enable_b)
        # Explicit deterministic initialization for the calibration branch.
        nn.init.constant_(module.calibration_raw, 0.0)
        nn.init.constant_(module.gamma_raw, 0.0)
    return module.to(device)
