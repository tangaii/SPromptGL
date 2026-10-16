"""Frozen AG-CDEA configurations and vectorized evidence branch (experiment 19)."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

TEMPERATURE, LAMBDA_EVIDENCE, RESIDUAL_LIMIT = .2, .1, .1
BANKS = ("REAL", "NEUTRAL", "UNRELATED", "CLASS_ONLY", "RANDOM_ROTATED")
DOMAIN_PERMUTATION = (2, 3, 0, 1)


@dataclass(frozen=True)
class Design:
    name: str
    query: str = "REAL"
    evidence: bool = True
    anatomy: str = "REAL"
    null: bool = True


DESIGNS = {
    "C0": Design("原始P0", evidence=False),
    "C1": Design("数值容量", query="ZERO", evidence=False),
    "C2": Design("真实语义无证据监督", evidence=False),
    "C3": Design("无解剖锚点", anatomy="ALL"),
    "C4": Design("完整AG-CDEA"),
    "C5": Design("中性文本", query="NEUTRAL"),
    "C6": Design("无关医学文本", query="UNRELATED"),
    "C7": Design("仅类别文字", query="CLASS_ONLY"),
    "C8": Design("ASD/TD方向交换", query="SWAPPED"),
    "C9": Design("功能域文本错配", query="DOMAIN_PERMUTED"),
    "C10": Design("解剖候选错配", anatomy="PERMUTED"),
    "C11": Design("同几何随机查询", query="RANDOM_ROTATED"),
    "C12": Design("可学习查询", query="LEARNED"),
    "C13": Design("强制匹配", null=False),
}
CONFIGURATIONS = {k: asdict(v) for k, v in DESIGNS.items()}


class EvidenceAlignment(nn.Module):
    def __init__(self, config, cache, seed, train_indices, cf_permutation):
        super().__init__()
        self.config, self.design = config, DESIGNS[config]
        self.edge_encoder = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.LayerNorm(32))
        self.edge_embedding = nn.Embedding(252, 32)
        self.edge_norm = nn.LayerNorm(32)
        self.key = nn.Linear(32, 1024, bias=False)
        self.value = nn.Linear(32, 32, bias=False)
        self.fusion = nn.Sequential(nn.Linear(136, 64), nn.GELU(), nn.Linear(64, 36))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer("queries", torch.stack([torch.as_tensor(cache["query_" + b], dtype=torch.float32) for b in BANKS]))
        self.register_buffer("mask_real", torch.as_tensor(cache["mask"], dtype=torch.bool))
        self.register_buffer("mask_permuted", torch.as_tensor(cache["mask_permuted"], dtype=torch.bool))
        self.register_buffer("valid_indices", torch.as_tensor(cache["valid_indices"], dtype=torch.long))
        self.register_buffer("train_indices", torch.as_tensor(train_indices, dtype=torch.long))
        self.register_buffer("cf_permutation", torch.as_tensor(cf_permutation, dtype=torch.long))
        if self.design.query == "LEARNED":
            g = torch.Generator().manual_seed(191200 + int(seed))
            self.learned_queries = nn.Parameter(F.normalize(torch.randn(4, 1024, generator=g), dim=-1))
        assert self.mask_real.shape == self.mask_permuted.shape == (4, 252)
        if not bool(self.mask_real.any(1).all()) or not bool(self.mask_permuted.any(1).all()):
            raise ValueError("Every anatomical domain must have at least one verified connection")
        self.condition, self.last = "REAL", {}

    def query_and_mask(self):
        kind = self.design.query
        if self.condition in BANKS + ("SWAPPED", "DOMAIN_PERMUTED"):
            if self.condition != "REAL":
                kind = self.condition
        if kind == "ZERO":
            q = torch.zeros_like(self.queries[0])
        elif kind == "LEARNED":
            q = F.normalize(self.learned_queries, dim=-1)
        elif kind == "SWAPPED":
            q = -self.queries[0]
        elif kind == "DOMAIN_PERMUTED":
            q = self.queries[0][list(DOMAIN_PERMUTATION)]
        else:
            q = self.queries[BANKS.index(kind)]
        # All nonzero queries are normalized, including evolving C12 parameters.
        q = F.normalize(q, dim=-1)
        anatomy = "PERMUTED" if self.condition == "ANATOMY_PERMUTED" else self.design.anatomy
        mask = torch.ones_like(self.mask_real) if anatomy == "ALL" else (
            self.mask_permuted if anatomy == "PERMUTED" else self.mask_real)
        return q, mask

    def forward(self, base, raw_modalities):
        assert base.shape[1:] == (4, 36) and raw_modalities is not None
        corr = raw_modalities[3]
        assert corr.shape == (len(base), 256)
        if self.condition == "FC_SHUFFLED":
            corr = corr.index_select(0, self.cf_permutation)
        z = corr.index_select(1, self.valid_indices)
        phi = torch.stack([z, z.abs(), z.tanh()], dim=-1)
        h = self.edge_norm(self.edge_encoder(phi) + self.edge_embedding.weight[None])
        q, mask = self.query_and_mask()
        q_eff = q @ self.key.weight
        score = torch.einsum("nef,kf->nke", h, q_eff) / math.sqrt(32)
        count = mask.sum(-1)
        logits = (score / TEMPERATURE).masked_fill(~mask[None], -torch.inf)
        null_logit = count.to(score.dtype).log()[None, :, None].expand(len(base), 4, 1)
        if not self.design.null or self.condition == "NULL_OFF":
            null_logit = torch.full_like(null_logit, -torch.inf)
        attention_all = torch.cat([logits, null_logit], dim=-1).softmax(-1)
        attention, p_null = attention_all[..., :-1], attention_all[..., -1]
        v = torch.einsum("nke,nef->nkf", attention, self.value(h))
        u = (attention * score).sum(-1)
        u_global = u.mean(-1)
        fusion_input = torch.cat([v.flatten(1), u, p_null], dim=-1)
        raw_delta = self.fusion(fusion_input)
        direction = raw_delta / raw_delta.square().sum(-1, keepdim=True).add(1e-8).sqrt()
        scale = base[:, 3].norm(dim=-1, keepdim=True).detach()
        delta_corr = RESIDUAL_LIMIT * self.alpha.tanh() * scale * direction
        if self.condition == "RESIDUAL_OFF":
            delta_corr = torch.zeros_like(delta_corr)
        zero = torch.zeros_like(delta_corr)
        delta = torch.stack([zero, zero, zero, delta_corr], dim=1)
        enhanced = base + delta
        self.last = dict(base=base, enhanced=enhanced, delta=delta, z=z, h=h, q=q, q_eff=q_eff,
                         mask=mask, score=score, attention=attention, p_null=p_null,
                         v=v, u=u, u_global=u_global, fusion_input=fusion_input)
        return enhanced

    def losses(self, labels):
        # Index BEFORE reading labels: held-out targets never enter this objective.
        target = (labels[self.train_indices] == 0).to(self.last["u_global"].dtype)
        evidence = F.binary_cross_entropy_with_logits(self.last["u_global"][self.train_indices], target)
        self.last["evidence_loss"] = evidence
        return LAMBDA_EVIDENCE * evidence if self.design.evidence else evidence.new_zeros(())

    def component_parameters(self):
        parts = {k: list(getattr(self, k).parameters()) for k in
                 ("edge_encoder", "edge_embedding", "edge_norm", "key", "value", "fusion")}
        parts["alpha"] = [self.alpha]
        if hasattr(self, "learned_queries"):
            parts["learned_queries"] = [self.learned_queries]
        return parts

    def effective_parameters(self):
        parts = self.component_parameters()
        active = set(parts) - ({"key"} if self.design.query == "ZERO" else set())
        return dict(registered=sum(p.numel() for pp in parts.values() for p in pp),
                    structurally_active=sum(p.numel() for k in active for p in parts[k]),
                    active_components=sorted(active), inactive_components=sorted(set(parts) - active))
