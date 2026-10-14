"""Frozen DS3A-v3: reliability weights affect training loss, never fusion."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

LAMBDA_DISC = 0.1
LAMBDA_PAIR = 0.1
LAMBDA_PRIVATE = 0.05
RESIDUAL_LIMIT = 0.1


@dataclass(frozen=True)
class Design:
    name: str
    disc: bool = False
    weighting: str = "none"
    private: bool = False
    mismatched_pair: bool = False


DESIGNS = {
    "V0": Design("原始P0"),
    "V1": Design("容量控制"),
    "V2": Design("FC Shared判别性保护", disc=True),
    "V3": Design("判别保护+统一患者对齐", disc=True, weighting="ones"),
    "V4": Design("同强度常量对齐", disc=True, weighting="mean"),
    "V5": Design("完整选择性对齐", disc=True, weighting="real"),
    "V6": Design("可靠度错配控制", disc=True, weighting="permuted"),
    "V7": Design("选择性对齐+FC教师Private蒸馏", disc=True, weighting="real", private=True),
    "V8": Design("无判别保护的统一对齐", weighting="ones"),
    "V9": Design("同类别跨患者配对错配", disc=True, weighting="real", mismatched_pair=True),
}
CONFIGURATIONS = {k: asdict(v) for k, v in DESIGNS.items()}


def encoder(dim):
    return nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.LayerNorm(32))


def weighted_pair_distance(shared_fc, shared_pheno, weights):
    distances = (F.normalize(shared_fc, dim=-1) - F.normalize(shared_pheno, dim=-1)).square().sum(-1)
    # Mean over ALL training patients. Never divide by sum(weights).
    return distances.mean(), (weights.detach() * distances).mean()


class SelectiveAlignment(nn.Module):
    def __init__(self, config, pheno_indices, evidence):
        super().__init__()
        self.config, self.design = config, DESIGNS[config]
        self.shared_corr, self.private_corr = encoder(256), encoder(256)
        self.shared_pheno, self.private_pheno = encoder(28), encoder(28)
        self.fusion = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 72))
        self.disc_head = nn.Linear(32, 2)
        self.private_head = nn.Linear(32, 1)
        self.alpha = nn.Parameter(torch.zeros(2))
        train = torch.as_tensor(evidence["train_indices"], dtype=torch.long)
        reliability = torch.as_tensor(evidence["reliability"], dtype=torch.float64)
        if self.design.weighting == "ones":
            weights = torch.ones(len(train), dtype=torch.float64)
        elif self.design.weighting == "mean":
            weights = reliability[train].mean().expand(len(train)).clone()
        elif self.design.weighting == "real":
            weights = reliability[train].clone()
        elif self.design.weighting == "permuted":
            weights = reliability[torch.as_tensor(evidence["reliability_permutation"], dtype=torch.long)[train]].clone()
        else:
            weights = torch.zeros(len(train), dtype=torch.float64)
        self.register_buffer("pheno_indices", torch.as_tensor(pheno_indices, dtype=torch.long))
        self.register_buffer("train_indices", train)
        paired = evidence["class_pair_permutation"] if self.design.mismatched_pair else evidence["identity"]
        self.register_buffer("paired_indices", torch.as_tensor(paired, dtype=torch.long)[train])
        self.register_buffer("weights", weights)
        teacher = torch.as_tensor(evidence["teacher_fc_probability"], dtype=torch.float32)
        self.register_buffer("teacher_train_asd", teacher[train].clone())
        self.register_buffer("cf_permutation", torch.as_tensor(evidence["cf_permutation"], dtype=torch.long))
        self.condition = "REAL"
        self.last = {}

    def forward(self, base, raw_modalities):
        assert raw_modalities is not None
        raw_pheno, _, _, raw_corr = raw_modalities
        pheno = raw_pheno.index_select(1, self.pheno_indices)
        if self.condition == "PHENO_SHUFFLED":
            pheno = pheno.index_select(0, self.cf_permutation)
        shared_fc, private_fc = self.shared_corr(raw_corr), self.private_corr(raw_corr)
        shared_ph, private_ph = self.shared_pheno(pheno), self.private_pheno(pheno)
        shared_inputs = [shared_fc, shared_ph]
        private_inputs = [private_fc, private_ph]
        if self.condition == "SHARED_OFF":
            shared_inputs = [torch.zeros_like(v) for v in shared_inputs]
        if self.condition == "PRIVATE_OFF":
            private_inputs = [torch.zeros_like(v) for v in private_inputs]
        fusion_input = torch.cat([shared_inputs[0], private_inputs[0], shared_inputs[1], private_inputs[1]], dim=-1)
        direction = self.fusion(fusion_input).reshape(len(base), 2, 36)
        direction = direction / direction.square().sum(-1, keepdim=True).add(1e-8).sqrt()
        scale = base[:, [0, 3]].norm(dim=-1, keepdim=True).detach()
        delta_selected = RESIDUAL_LIMIT * self.alpha.tanh()[None, :, None] * scale * direction
        if self.condition == "RESIDUAL_OFF":
            delta_selected = torch.zeros_like(delta_selected)
        zero = torch.zeros_like(delta_selected[:, 0])
        delta = torch.stack([delta_selected[:, 0], zero, zero, delta_selected[:, 1]], dim=1)
        enhanced = base + delta
        self.last = dict(base=base, enhanced=enhanced, delta=delta, fusion_input=fusion_input,
                         shared_corr=shared_fc, private_corr=private_fc,
                         shared_pheno=shared_ph, private_pheno=private_ph)
        return enhanced

    def losses(self, labels):
        tr, ll = self.train_indices, self.last
        # Only outer-train labels are indexed. Test labels have no loss use.
        disc = F.cross_entropy(self.disc_head(ll["shared_corr"][tr]), labels[tr])
        private = F.binary_cross_entropy_with_logits(
            self.private_head(ll["private_corr"][tr]).squeeze(-1), self.teacher_train_asd)
        raw_pair, pair = weighted_pair_distance(ll["shared_corr"][tr], ll["shared_pheno"][self.paired_indices], self.weights)
        total = disc.new_zeros(())
        if self.design.disc:
            total = total + LAMBDA_DISC * disc
        if self.design.weighting != "none":
            total = total + LAMBDA_PAIR * pair
        if self.design.private:
            total = total + LAMBDA_PRIVATE * private
        ll.update(disc_loss=disc, pair_raw_loss=raw_pair, pair_weighted_loss=pair, private_loss=private)
        return total

    def component_parameters(self):
        parts = {k: list(getattr(self, k).parameters()) for k in (
            "shared_corr", "private_corr", "shared_pheno", "private_pheno", "fusion", "disc_head", "private_head")}
        parts["alpha"] = [self.alpha]
        return parts

    def effective_parameters(self):
        components = self.component_parameters()
        active = {"shared_corr", "private_corr", "shared_pheno", "private_pheno", "fusion", "alpha"}
        if self.design.disc:
            active.add("disc_head")
        if self.design.private:
            active.add("private_head")
        return dict(registered=sum(p.numel() for p in self.parameters()),
                    structurally_active=sum(p.numel() for k in active for p in components[k]),
                    active_components=sorted(active),
                    inactive_components=sorted(set(components) - active))
