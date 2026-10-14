"""Frozen DS3A-v2 design. Labels are accepted only by the training loss."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Design:
    name: str
    alignment: str = "shared"
    selective: bool = False
    prompt: str = "OFF"
    quality: bool = False
    fuse_private: bool = True
    four_modalities: bool = False


DESIGNS = {
    "D0": Design("原始P0", alignment="none"),
    "D1": Design("容量控制", alignment="none"),
    "D2": Design("Shared+Private强制对齐", alignment="shared_private"),
    "D3": Design("仅Shared统一对齐"),
    "D4": Design("特征条件Align/Skip", selective=True),
    "D5": Design("完整DS3A-v2", selective=True, prompt="REAL", quality=True),
    "D6": Design("中性文本", selective=True, prompt="NEUTRAL", quality=True),
    "D7": Design("无关疾病文本", selective=True, prompt="UNRELATED", quality=True),
    "D8": Design("可学习类别向量", selective=True, prompt="LEARNED", quality=True),
    "D9": Design("关闭QC条件", selective=True, prompt="REAL"),
    "D10": Design("不融合新增Private", selective=True, prompt="REAL", quality=True, fuse_private=False),
    "D11": Design("四模态统一Shared对齐", four_modalities=True),
}
CONFIGURATION_TABLE = {k: asdict(v) for k, v in DESIGNS.items()}
LAMBDA_ALIGN = .1
LAMBDA_PRIVATE = .01
PRIVATE_COS_MARGIN = .5
RESIDUAL_LIMIT = .1


def encoder(d_in, d_out=32):
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.LayerNorm(d_out))


class DS3A(nn.Module):
    """Raw-input auxiliary path; enhances CAM positions 0 and 3, never ROIs."""

    def __init__(self, config, pheno_indices, text_banks, seed):
        super().__init__()
        self.design = DESIGNS[config]
        self.config = config
        self.shared_corr = encoder(256)
        self.private_corr = encoder(256)
        self.shared_pheno = encoder(len(pheno_indices))
        self.private_pheno = encoder(len(pheno_indices))
        # Registered equally in all auxiliary groups; only D11 uses these.
        self.shared_anat = encoder(6)
        self.shared_func = encoder(10)
        self.query_corr = nn.Linear(32, 1024, bias=False)
        self.query_pheno = nn.Linear(32, 1024, bias=False)
        self.quality_encoder = encoder(16, 16)
        self.quality_confidence = nn.Linear(16, 1)
        # 8 shared statistics + 4 semantic scores + 16 QC features + 1 QC proxy.
        self.align_skip = nn.Sequential(nn.Linear(29, 16), nn.GELU(), nn.Linear(16, 2))
        self.fusion = nn.Sequential(nn.Linear(129, 64), nn.GELU(), nn.Linear(64, 72))
        self.alpha = nn.Parameter(torch.zeros(2))
        self.register_buffer("pheno_indices", torch.tensor(pheno_indices, dtype=torch.long))
        self.register_buffer("text_banks", torch.stack([text_banks[k] for k in ("REAL", "NEUTRAL", "UNRELATED")]))
        if self.design.prompt == "LEARNED":
            generator = torch.Generator().manual_seed(170001 + int(seed))
            # Random initialization: no ASD/TD CLIP prior in the learned control.
            self.learned_prototypes = nn.Parameter(F.normalize(torch.randn(2, 1024, generator=generator), dim=-1))
        self.condition = "REAL"
        self.last = {}
        self._train_indices = None
        self._class_weights = None

    def prototypes(self):
        bank = self.design.prompt
        if self.condition in ("NEUTRAL", "UNRELATED"):
            bank = self.condition
        if bank == "LEARNED":
            text = F.normalize(self.learned_prototypes, dim=-1)
        else:
            text = self.text_banks[("REAL", "NEUTRAL", "UNRELATED").index(bank)]
        return text.flip(0) if self.condition == "SWAPPED" else text

    def forward(self, base, raw_modalities):
        pheno, anat, func, corr = raw_modalities
        pheno = pheno.index_select(1, self.pheno_indices)
        sc, pc = self.shared_corr(corr), self.private_corr(corr)
        sp, pp = self.shared_pheno(pheno), self.private_pheno(pheno)
        zc, zp = F.normalize(sc, dim=-1), F.normalize(sp, dim=-1)
        shared_stats = torch.stack([
            (zc * zp).sum(-1), (zc - zp).square().mean(-1).add(1e-8).sqrt(),
            sc.norm(dim=-1), sp.norm(dim=-1), sc.mean(-1), sp.mean(-1),
            sc.var(-1, unbiased=False), sp.var(-1, unbiased=False)], dim=-1)
        qc_active = self.design.quality and self.condition != "NO_QC"
        if qc_active:
            quality = self.quality_encoder(torch.cat([anat, func], dim=-1))
            confidence = self.quality_confidence(quality).sigmoid()
        else:
            quality = sc.new_zeros(len(sc), 16)
            confidence = sc.new_full((len(sc), 1), .5)
        if self.design.prompt != "OFF":
            text = self.prototypes()
            query_c, query_p = self.query_corr(zc), self.query_pheno(zp)
            semantic_c = F.normalize(query_c, dim=-1) @ text.T
            semantic_p = F.normalize(query_p, dim=-1) @ text.T
        else:
            text = sc.new_zeros(2, 1024)
            query_c = query_p = sc.new_zeros(len(sc), 1024)
            semantic_c = semantic_p = sc.new_zeros(len(sc), 2)
        gate_input = torch.cat([shared_stats, semantic_c, semantic_p, quality, confidence], dim=-1)
        if self.design.selective:
            align = self.align_skip(gate_input).softmax(-1)[:, :1]
        else:
            align = sc.new_ones(len(sc), 1)
        if self.condition == "SKIP":
            align = torch.zeros_like(align)
        private_active = self.design.fuse_private and self.condition != "NO_PRIVATE"
        fc_private = pc if private_active else torch.zeros_like(pc)
        ph_private = pp if private_active else torch.zeros_like(pp)
        fusion_input = torch.cat([align * sc, fc_private, align * sp, ph_private, align], dim=-1)
        raw_delta = self.fusion(fusion_input).view(len(base), 2, 36)
        selected = base[:, [0, 3], :]
        # Smooth unit direction, detached reference scale, signed zero-init gates.
        unit = raw_delta / raw_delta.square().sum(-1, keepdim=True).add(1e-8).sqrt()
        scale = selected.norm(dim=-1, keepdim=True).detach()
        delta_selected = RESIDUAL_LIMIT * self.alpha.tanh()[None, :, None] * scale * unit
        if self.condition == "RESIDUAL_OFF":
            delta_selected = torch.zeros_like(delta_selected)
        zero = torch.zeros_like(delta_selected[:, 0])
        delta = torch.stack([delta_selected[:, 0], zero, zero, delta_selected[:, 1]], dim=1)
        enhanced = base + delta
        self.last = dict(base=base, enhanced=enhanced, delta=delta, shared_corr=sc,
                         private_corr=pc, shared_pheno=sp, private_pheno=pp,
                         align=align, quality=quality, confidence=confidence,
                         semantic_corr=semantic_c, semantic_pheno=semantic_p,
                         query_corr=query_c, query_pheno=query_p, text=text,
                         gate_input=gate_input, fusion_input=fusion_input)
        if self.design.four_modalities:
            self.last.update(shared_anat=self.shared_anat(anat), shared_func=self.shared_func(func))
        return enhanced

    def train_masks(self, labels, indices):
        if self._train_indices is None:
            self._train_indices = torch.as_tensor(indices, device=labels.device, dtype=torch.long)
            # Only selected training labels are read, never labels of held-out rows.
            self._class_weights = F.one_hot(labels[self._train_indices], num_classes=2).float().T
        return self._train_indices, self._class_weights

    @staticmethod
    def prototype_distance(values, class_weights, align):
        weights = class_weights * align.detach().view(1, -1)
        denom = weights.sum(-1, keepdim=True)
        prototypes = [F.normalize((weights @ value) / denom.clamp_min(1e-8), dim=-1) for value in values]
        distances = torch.stack([(prototypes[i] - prototypes[j]).square().sum(-1)
                                 for i in range(len(values)) for j in range(i + 1, len(values))]).mean(0)
        valid = (denom[:, 0] > 1e-8).to(distances.dtype)
        return (distances * valid).sum() / valid.sum().clamp_min(1.)

    def loss(self, labels, indices):
        tr, classes = self.train_masks(labels, indices)
        ll = self.last
        align = ll["align"][tr]
        shared = [ll[k][tr] for k in ("shared_corr", "shared_pheno")]
        if self.design.four_modalities:
            shared += [ll[k][tr] for k in ("shared_anat", "shared_func")]
        if self.design.alignment == "none":
            alignment = shared[0].new_zeros(())
        else:
            alignment = self.prototype_distance(shared, classes, align)
            if self.design.alignment == "shared_private":
                alignment = alignment + self.prototype_distance(
                    [ll[k][tr] for k in ("private_corr", "private_pheno")], classes, align)
        # Margin reduction, not hard orthogonality; identical D1-D11 coefficients.
        private = torch.stack([F.relu(F.cosine_similarity(ll[s][tr], ll[p][tr]).abs() - PRIVATE_COS_MARGIN).square().mean()
                               for s, p in (("shared_corr", "private_corr"), ("shared_pheno", "private_pheno"))]).mean()
        ll.update(align_loss=alignment, private_loss=private)
        return LAMBDA_ALIGN * alignment + LAMBDA_PRIVATE * private

    def component_parameters(self):
        result = {k: list(getattr(self, k).parameters()) for k in (
            "shared_corr", "private_corr", "shared_pheno", "private_pheno", "shared_anat", "shared_func",
            "query_corr", "query_pheno", "quality_encoder", "quality_confidence", "align_skip", "fusion")}
        result["alpha"] = [self.alpha]
        if hasattr(self, "learned_prototypes"):
            result["learned_prototypes"] = [self.learned_prototypes]
        return result
