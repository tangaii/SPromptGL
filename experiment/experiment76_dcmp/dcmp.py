"""Disease-confusion memory prompting at the CAM144/OutputLayer boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PrototypeMemory:
    features: torch.Tensor
    labels: torch.Tensor
    sites: torch.Tensor
    valid: torch.Tensor
    ids: torch.Tensor
    typical_assignment: torch.Tensor
    hard_assignment: torch.Tensor
    hard_target: torch.Tensor
    prototype_sha256: str
    training_index_sha256: str
    excluded_test_ids: list[int]
    metadata: dict

    def state_dict(self) -> dict:
        return {
            "features": self.features.detach().cpu(), "labels": self.labels.cpu(),
            "sites": self.sites.cpu(), "valid": self.valid.cpu(), "ids": self.ids.cpu(),
            "typical_assignment": self.typical_assignment.cpu(),
            "hard_assignment": self.hard_assignment.cpu(), "hard_target": self.hard_target.cpu(),
            "prototype_sha256": self.prototype_sha256,
            "training_index_sha256": self.training_index_sha256,
            "excluded_test_ids": self.excluded_test_ids, "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device) -> "PrototypeMemory":
        return cls(**{key: (value.to(device) if torch.is_tensor(value) else value)
                      for key, value in state.items()})


def _sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _kmeans_assign(features: np.ndarray, count: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    if len(features) < count:
        raise ValueError(f"need at least {count} samples for prototype clustering; got {len(features)}")
    model = KMeans(n_clusters=count, n_init=10, random_state=seed, algorithm="lloyd")
    return model.fit_predict(features.astype(np.float64, copy=False)).astype(np.int64)


def build_memory(features: torch.Tensor, labels: torch.Tensor, sites: torch.Tensor,
                 train_ids: np.ndarray, test_ids: np.ndarray, fold: int,
                 probe_probability: np.ndarray) -> PrototypeMemory:
    """Build fold-local assignments; known fold-2 overlap is excluded from all DCMP labels."""
    x = features.detach().float().cpu()
    y = labels.detach().long().cpu()
    site = sites.detach().long().cpu()
    test = set(int(v) for v in test_ids)
    eligible = np.asarray([int(v) for v in train_ids if int(v) not in test], dtype=np.int64)
    if not eligible.size or np.intersect1d(eligible, test_ids).size:
        raise RuntimeError("DCMP memory IDs must be a nonempty subset of outer train minus test")
    if probe_probability.shape != (len(eligible), 2):
        raise ValueError(f"probe probability shape mismatch: {probe_probability.shape} vs {len(eligible)}")

    ids_all = np.arange(len(y), dtype=np.int64)
    typical = torch.full((len(y),), -1, dtype=torch.long)
    hard = torch.full((len(y),), -1, dtype=torch.long)
    risk_target = torch.zeros(len(y), dtype=torch.float32)
    hard_rows: list[int] = []
    hard_counts = {}
    eligible_y = y[eligible].numpy()
    for cls in (0, 1):
        positions = np.flatnonzero(eligible_y == cls)
        class_ids = eligible[positions]
        typical_labels = _kmeans_assign(x[class_ids].numpy(), 2, 760000 + fold * 100 + cls)
        typical[class_ids] = torch.as_tensor(typical_labels + 2 * cls)
        # Lower signed true-class margin is the predeclared fold-local confusion risk.
        probs = probe_probability[positions]
        margin = probs[:, cls] - probs[:, 1 - cls]
        order = np.lexsort((class_ids, margin))
        n_hard = max(2, int(np.ceil(0.20 * len(class_ids))))
        selected = class_ids[order[:n_hard]]
        hard_rows.extend(selected.tolist())
        risk_target[selected] = 1.0
        hard_counts[str(cls)] = int(len(selected))
        hard_labels = _kmeans_assign(x[selected].numpy(), 2, 761000 + fold * 100 + cls)
        hard[selected] = torch.as_tensor(hard_labels + 2 * cls)

    valid = torch.zeros(len(y), dtype=torch.bool)
    valid[eligible] = True
    memory_x = x.clone()
    memory_y = y.clone()
    memory_site = site.clone()
    memory_risk = risk_target.clone()
    memory_x[~valid] = 0
    memory_y[~valid] = -1
    memory_site[~valid] = -1
    memory_risk[~valid] = -1
    digest_ids = hashlib.sha256(np.asarray(eligible, dtype=np.int64).tobytes()).hexdigest()
    memory_digest = hashlib.sha256()
    for tensor in (x[eligible], y[eligible], site[eligible], typical[eligible], hard[eligible], risk_target[eligible]):
        memory_digest.update(tensor.contiguous().numpy().tobytes())
    metadata = {
        "fold": int(fold), "eligible_global_ids_zero_based": eligible.tolist(),
        "excluded_test_ids_zero_based": sorted(test), "typical_support": {
            f"class_{cls}_cluster_{cluster}": int(((typical == 2 * cls + cluster) & valid).sum())
            for cls in (0, 1) for cluster in range(2)},
        "hard_support": {f"class_{cls}_cluster_{cluster}": int(((hard == 2 * cls + cluster) & valid).sum())
                         for cls in (0, 1) for cluster in range(2)},
        "hard_selected_per_class": hard_counts,
        "risk_target_support": int(risk_target[eligible].sum()),
        "site_supports": {}, "leave_one_out": "cluster sums/counts subtract current query row before each training retrieval",
        "prototype_updates": "EMA 0.9 of detached same-step CAM144 for legal training IDs; assignments fixed from warmup",
    }
    return PrototypeMemory(memory_x, memory_y, memory_site, valid, torch.from_numpy(ids_all), typical, hard, memory_risk,
                           memory_digest.hexdigest(), digest_ids, sorted(test), metadata)


class DCMP(nn.Module):
    """Text-conditioned retrieval over typical, two-way confusion, and site-offset memories."""

    def __init__(self, disease_text: torch.Tensor, site_count: int = 20, key_dim: int = 32):
        super().__init__()
        if tuple(disease_text.shape) != (2, 1024):
            raise ValueError(f"disease_text must be [2,1024] ASD/TD, got {tuple(disease_text.shape)}")
        self.register_buffer("disease_text", disease_text.detach().float().clone())
        self.query = nn.Linear(144, key_dim, bias=False)
        self.text_key = nn.Linear(1024, key_dim, bias=False)
        self.memory_key = nn.Linear(144, key_dim, bias=False)
        self.role_key = nn.Embedding(5, key_dim)
        self.value = nn.Linear(144, 48, bias=False)
        self.text_value = nn.Linear(1024, 48, bias=False)
        self.evidence_composer = nn.Sequential(
            nn.Linear(5 * 48 + 144, 96), nn.GELU(), nn.Linear(96, 144)
        )
        self.risk_head = nn.Sequential(nn.Linear(5 * 48 + 144, 48), nn.GELU(), nn.Linear(48, 1))
        self.rho_head = nn.Sequential(nn.Linear(5 * 48 + 144, 48), nn.GELU(), nn.Linear(48, 1))
        self.reliability_head = nn.Sequential(nn.Linear(5 * 48 + 144, 32), nn.GELU(), nn.Linear(32, 1))
        self.site_count = int(site_count)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.text_key.weight)
        nn.init.xavier_uniform_(self.memory_key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.text_value.weight)
        nn.init.normal_(self.role_key.weight, std=0.02)
        nn.init.xavier_uniform_(self.evidence_composer[0].weight)
        nn.init.zeros_(self.evidence_composer[0].bias)
        nn.init.xavier_uniform_(self.evidence_composer[2].weight)
        with torch.no_grad():
            self.evidence_composer[2].weight.mul_(0.01)
        nn.init.zeros_(self.evidence_composer[2].bias)
        for head in (self.risk_head, self.rho_head, self.reliability_head):
            nn.init.xavier_uniform_(head[0].weight); nn.init.zeros_(head[0].bias)
            nn.init.xavier_uniform_(head[2].weight); nn.init.zeros_(head[2].bias)
        nn.init.constant_(self.rho_head[-1].bias, -4.5)

    @staticmethod
    def _loo_centers(memory: PrototypeMemory, assignment: torch.Tensor, n_proto: int,
                     query_ids: torch.Tensor, use_loo: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = memory.features.device
        valid = memory.valid & (assignment >= 0)
        features = memory.features
        sums = torch.stack([features[valid & (assignment == proto)].sum(0) for proto in range(n_proto)])
        counts = torch.stack([(valid & (assignment == proto)).sum().to(features.dtype) for proto in range(n_proto)])
        qids = query_ids.long()
        centers = sums.unsqueeze(0).expand(len(qids), -1, -1).clone()
        support = counts.unsqueeze(0).expand(len(qids), -1).clone()
        if use_loo:
            q_assign = assignment.index_select(0, qids)
            legal = memory.valid.index_select(0, qids) & (q_assign >= 0)
            rows = legal.nonzero(as_tuple=False).flatten()
            if rows.numel():
                cols = q_assign[rows]
                centers[rows, cols] -= features.index_select(0, qids[rows])
                support[rows, cols] -= 1.0
        available = support > 0
        centers = centers / support.clamp_min(1).unsqueeze(-1)
        return centers, available, support

    def _bank(self, memory: PrototypeMemory, query_ids: torch.Tensor, *, typical: bool,
              confusion: bool, background: bool, use_loo: bool):
        centers, available, roles, classes, supports = [], [], [], [], []
        n_query = int(query_ids.numel())
        if typical:
            c, a, n = self._loo_centers(memory, memory.typical_assignment, 4, query_ids, use_loo)
            centers.extend(c.unbind(1)); available.extend(a.unbind(1)); supports.extend(n.unbind(1))
            roles.extend([0, 0, 0, 0]); classes.extend([0, 0, 1, 1])
        if confusion:
            c, a, n = self._loo_centers(memory, memory.hard_assignment, 4, query_ids, use_loo)
            centers.extend(c.unbind(1)); available.extend(a.unbind(1)); supports.extend(n.unbind(1))
            roles.extend([1, 1, 2, 2]); classes.extend([0, 0, 1, 1])
        if background:
            valid = memory.valid
            y = memory.labels
            qids = query_ids.long()
            q_valid = valid.index_select(0, qids) if use_loo else torch.zeros(n_query, dtype=torch.bool, device=valid.device)
            q_labels = y.index_select(0, qids)
            q_sites = memory.sites.index_select(0, qids)
            for site_id in torch.unique(memory.sites[valid]).tolist():
                site_mask = valid & (memory.sites == int(site_id))
                class_counts = [int((site_mask & (y == cls)).sum()) for cls in (0, 1)]
                site_classes, global_classes, site_supports, global_supports = [], [], [], []
                for cls in (0, 1):
                    global_mask = valid & (y == cls)
                    global_sum = memory.features[global_mask].sum(0)
                    global_count = global_mask.sum().to(memory.features.dtype)
                    site_class_mask = site_mask & (y == cls)
                    site_sum = memory.features[site_class_mask].sum(0)
                    site_count = site_class_mask.sum().to(memory.features.dtype)
                    if use_loo:
                        subtract_global = q_valid & (q_labels == cls)
                        subtract_site = subtract_global & (q_sites == int(site_id))
                        query_features = memory.features.index_select(0, qids)
                        global_sum_q = global_sum.unsqueeze(0) - query_features * subtract_global[:, None]
                        global_count_q = global_count - subtract_global.to(memory.features.dtype)
                        site_sum_q = site_sum.unsqueeze(0) - query_features * subtract_site[:, None]
                        site_count_q = site_count - subtract_site.to(memory.features.dtype)
                    else:
                        global_sum_q = global_sum.unsqueeze(0).expand(n_query, -1)
                        global_count_q = global_count.expand(n_query)
                        site_sum_q = site_sum.unsqueeze(0).expand(n_query, -1)
                        site_count_q = site_count.expand(n_query)
                    global_classes.append(global_sum_q / global_count_q.clamp_min(1)[:, None])
                    site_classes.append(site_sum_q / site_count_q.clamp_min(1)[:, None])
                    global_supports.append(global_count_q)
                    site_supports.append(site_count_q)
                offset = 0.5 * (site_classes[0] + site_classes[1] - global_classes[0] - global_classes[1])
                support = torch.minimum(site_supports[0], site_supports[1])
                enough = support >= 3
                centers.append(offset)
                available.append(enough)
                supports.append(support)
                roles.append(3); classes.append(2)
        if not centers:
            raise RuntimeError("no usable DCMP prototypes")
        return (torch.stack(centers, dim=1), torch.stack(available, dim=1),
                torch.as_tensor(roles, device=memory.features.device, dtype=torch.long),
                torch.as_tensor(classes, device=memory.features.device, dtype=torch.long),
                torch.stack(supports, dim=1))

    def forward(self, h: torch.Tensor, memory: PrototypeMemory, query_ids: torch.Tensor,
                *, typical: bool, confusion: bool, background: bool, anonymous_text: bool,
                training: bool, residual_enabled: bool = True) -> dict[str, torch.Tensor]:
        if h.ndim != 2 or h.shape[1] != 144:
            raise ValueError(f"DCMP expected CAM144, got {tuple(h.shape)}")
        text = self.disease_text.to(device=h.device, dtype=h.dtype)
        centers, available, roles, classes, supports = self._bank(
            memory, query_ids, typical=typical, confusion=confusion, background=background,
            use_loo=training)
        q_base = self.query(h)
        memory_keys = self.memory_key(centers) + self.role_key(roles).unsqueeze(0)
        class_text = self.text_key(text)
        text_routing_logits = F.normalize(q_base, dim=-1, eps=1e-8) @ F.normalize(class_text, dim=-1, eps=1e-8).T
        text_routing = torch.softmax(text_routing_logits, dim=-1)
        q = q_base + text_routing @ class_text
        class_text_all = text.index_select(0, classes.clamp(max=1))
        class_text_all = torch.where((classes == 2)[:, None], text.mean(0, keepdim=True), class_text_all)
        key_text = class_text.index_select(0, classes.clamp(max=1))
        key_text = torch.where((classes == 2)[:, None], class_text.mean(0, keepdim=True), key_text)
        keys = F.normalize(memory_keys + key_text.unsqueeze(0), dim=-1, eps=1e-8)
        queries = F.normalize(q, dim=-1, eps=1e-8).unsqueeze(1)
        score = (queries * keys).sum(-1) / (queries.shape[-1] ** 0.5)
        score = score.masked_fill(~available, -1e4)
        attention = torch.softmax(score, dim=-1)
        values = self.value(centers) + self.text_value(class_text_all).unsqueeze(0)
        retrieved_all = attention.unsqueeze(-1) * values
        evidence = []
        role_plan = ([0], [0], [1], [2], [3])
        class_plan = [0, 1, 0, 1, 2]
        for role_list, class_id in zip(role_plan, class_plan):
            mask = torch.zeros_like(roles, dtype=torch.bool)
            for role in role_list:
                mask |= roles == role
            if class_id < 2:
                mask &= classes == class_id
            else:
                mask &= classes == 2
            if (class_id == 0 or class_id == 1) and not typical and role_list == [0]:
                evidence.append(h.new_zeros((h.shape[0], values.shape[-1])))
            elif class_id in (0, 1) and not confusion and role_list in ([1], [2]):
                evidence.append(h.new_zeros((h.shape[0], values.shape[-1])))
            elif class_id == 2 and not background:
                evidence.append(h.new_zeros((h.shape[0], values.shape[-1])))
            elif mask.any():
                weights = attention[:, mask]
                evidence.append((retrieved_all[:, mask].sum(dim=1) / weights.sum(-1, keepdim=True).clamp_min(1e-8)))
            else:
                evidence.append(h.new_zeros((h.shape[0], values.shape[-1])))
        evidence_tensor = torch.cat(evidence, dim=-1)
        composer_input = torch.cat((h, evidence_tensor), dim=-1)
        raw_direction = torch.tanh(self.evidence_composer(composer_input))
        norm = raw_direction.norm(dim=-1, keepdim=True)
        direction = raw_direction / norm.clamp_min(1.0)
        risk = torch.sigmoid(self.risk_head(composer_input)).squeeze(-1)
        rho = 0.05 * torch.sigmoid(self.rho_head(composer_input)).squeeze(-1)
        reliability = torch.sigmoid(self.reliability_head(composer_input)).squeeze(-1)
        rho = rho * reliability * risk
        delta = direction * (h.norm(dim=-1) * rho).unsqueeze(-1)
        if not residual_enabled:
            delta = torch.zeros_like(delta)
        h_prime = h + delta
        return {
            "h_prime": h_prime, "delta": delta, "rho": rho, "risk": risk,
            "reliability": reliability, "attention": attention, "text_routing": text_routing,
            "prototype_roles": roles, "prototype_classes": classes,
            "prototype_supports": supports, "prototype_available": available,
            "retrieved_evidence": evidence_tensor, "memory_keys": keys,
        }

    @torch.no_grad()
    def update_memory(self, memory: PrototypeMemory, current_features: torch.Tensor, ema: float = 0.9) -> None:
        ids = memory.valid.nonzero(as_tuple=False).flatten()
        if ids.numel():
            memory.features[ids] = float(ema) * memory.features[ids] + (1.0 - float(ema)) * current_features[ids].detach().float()


def leave_one_out_self_test() -> dict:
    features = torch.arange(16 * 144, dtype=torch.float32).reshape(16, 144)
    labels = torch.tensor([0] * 8 + [1] * 8)
    sites = torch.tensor([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    valid = torch.ones(16, dtype=torch.bool)
    assignment = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    memory = PrototypeMemory(features, labels, sites, valid, torch.arange(16), assignment,
                             assignment.clone(), torch.zeros(16), "", "", [], {})
    query_ids = torch.arange(16)
    centers, available, _ = DCMP._loo_centers(memory, assignment, 4, query_ids, True)
    expected_typical = features[1:4].mean(0)
    if not torch.equal(centers[0, 0], expected_typical) or not available[0, 0]:
        raise AssertionError("leave-one-out failed to remove only the queried member")

    module = DCMP(torch.zeros(2, 1024))
    site_centers, site_available, roles, _, _ = module._bank(
        memory, query_ids, typical=False, confusion=False, background=True, use_loo=True)
    site_roles = (roles == 3).nonzero(as_tuple=False).flatten()
    site0_proto = int(site_roles[0])
    global0 = features[1:8].mean(0)
    global1 = features[8:16].mean(0)
    site0_cls0 = features[1:4].mean(0)
    site0_cls1 = features[8:12].mean(0)
    expected_offset = 0.5 * (site0_cls0 + site0_cls1 - global0 - global1)
    if not torch.equal(site_centers[0, site0_proto], expected_offset) or not site_available[0, site0_proto]:
        raise AssertionError("site-offset leave-one-out failed for the querying patient")
    return {"prototype_self_exclusion": True, "site_offset_self_exclusion": True}


def tensor_sha(module: nn.Module, include_buffers: bool = False) -> str:
    digest = hashlib.sha256()
    state = module.state_dict() if include_buffers else dict(module.named_parameters())
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def memory_sha(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in state if key not in ("metadata", "prototype_sha256")):
        value = state[name]
        digest.update(name.encode("utf-8") + b"\0")
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
