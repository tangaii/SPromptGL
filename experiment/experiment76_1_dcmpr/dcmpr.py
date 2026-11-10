"""Refined confusion-memory prompting at the CAM144 / OutputLayer boundary."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

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

    def state_dict(self):
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in self.__dict__.items()}

    @classmethod
    def from_state_dict(cls, state, device):
        return cls(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()})


def _kmeans(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    if len(x) < count or count < 1:
        raise ValueError(f"invalid prototype count {count} for support {len(x)}")
    return KMeans(n_clusters=count, n_init=10, random_state=int(seed), algorithm="lloyd").fit_predict(x.astype(np.float64)).astype(np.int64)


def build_memory(features, labels, sites, train_ids, test_ids, fold, probe_probability):
    x = features.detach().float().cpu()
    y = labels.detach().long().cpu()
    site = sites.detach().long().cpu()
    test = set(map(int, test_ids))
    eligible = np.asarray([int(i) for i in train_ids if int(i) not in test], dtype=np.int64)
    if not len(eligible) or np.intersect1d(eligible, test_ids).size:
        raise RuntimeError("memory must use outer training IDs excluding every test ID")
    p = np.asarray(probe_probability, dtype=np.float64)
    if p.shape != (len(eligible), 2):
        raise ValueError(f"OOF probabilities have wrong shape {p.shape}")
    y_e = y[eligible].numpy()
    pred = p.argmax(1)
    typical = torch.full((len(y),), -1, dtype=torch.long)
    hard = torch.full((len(y),), -1, dtype=torch.long)
    hard_target = torch.full((len(y),), -1.0, dtype=torch.float32)
    hard_meta = {}
    for cls in (0, 1):
        ids = eligible[y_e == cls]
        assignment = _kmeans(x[ids].numpy(), 2, 761000 + fold * 100 + cls)
        typical[ids] = torch.as_tensor(assignment + 2 * cls)
        easy = (y_e == cls) & (pred == y_e) & ((p[:, cls] - p[:, 1-cls]) >= 0.20)
        hard_target[eligible[easy]] = 0.0
        wrong_ids = eligible[(y_e == cls) & (pred != y_e)]
        max_wrong = int(np.floor(0.20 * int((y_e == cls).sum())))
        if max_wrong > 0 and len(wrong_ids) > max_wrong:
            confidence = p[(y_e == cls) & (pred != y_e), 1 - cls]
            wrong_ids = wrong_ids[np.argsort(-confidence, kind="stable")[:max_wrong]]
        direction = "ASD_to_TD" if cls == 0 else "TD_to_ASD"
        if len(wrong_ids) >= 2:
            n_proto = 2
            a = _kmeans(x[wrong_ids].numpy(), n_proto, 762000 + fold * 100 + cls)
            hard[wrong_ids] = torch.as_tensor(a + 2 * cls)
        elif len(wrong_ids) == 1:
            hard[wrong_ids] = 2 * cls
            n_proto = 1
        else:
            n_proto = 0
        if len(wrong_ids):
            hard_target[wrong_ids] = 1.0
        hard_meta[direction] = {"oof_error_count": int(np.sum((y_e == cls) & (pred != y_e))),
            "selected_error_count": int(len(wrong_ids)), "prototype_count": n_proto,
            "support": [int(((hard == 2 * cls + j) & (torch.arange(len(y)) >= 0)).sum()) for j in range(n_proto)],
            "selected_global_ids_zero_based": wrong_ids.tolist()}
    valid = torch.zeros(len(y), dtype=torch.bool); valid[eligible] = True
    x_mem = x.clone(); y_mem = y.clone(); sites_mem = site.clone()
    x_mem[~valid] = 0; y_mem[~valid] = -1; sites_mem[~valid] = -1; hard_target[~valid] = -1
    digest = hashlib.sha256()
    for tensor in (x[eligible], y[eligible], typical[eligible], hard[eligible], hard_target[eligible]):
        digest.update(tensor.contiguous().numpy().tobytes())
    index_sha = hashlib.sha256(eligible.astype(np.int64).tobytes()).hexdigest()
    metadata = {"fold": int(fold), "eligible_global_ids_zero_based": eligible.tolist(),
        "excluded_test_ids_zero_based": sorted(test), "typical_support": {
            f"class_{c}_cluster_{j}": int(((typical == 2*c+j) & valid).sum()) for c in (0,1) for j in (0,1)},
        "hard_directions": hard_meta, "hard_support": {
            f"direction_{c}_cluster_{j}": int(((hard == 2*c+j) & valid).sum()) for c in (0,1) for j in (0,1)},
        "hard_selected_ids_zero_based": torch.where(hard_target > 0)[0].tolist(),
        "hard_target_support": int(hard_target[eligible].sum()),
        "easy_target_support": int((hard_target[eligible] == 0).sum()),
        "difficulty_definition": "5-fold OOF LogisticRegression actual misclassifications only; top error confidence capped at 20% of true class",
        "prototype_updates": "detached CAM144 EMA 0.9; assignments reclustered every 20 epochs within fixed legal pools",
        "leave_one_out": "query subtracted from its cluster sum and support before retrieval"}
    return PrototypeMemory(x_mem, y_mem, sites_mem, valid, torch.arange(len(y)), typical, hard,
        hard_target, digest.hexdigest(), index_sha, sorted(test), metadata)


class DCMP(nn.Module):
    """Class-balanced hierarchical retrieval and bounded selective correction."""
    def __init__(self, disease_text: torch.Tensor, site_count: int = 20, key_dim: int = 32):
        super().__init__()
        if tuple(disease_text.shape) != (2, 1024):
            raise ValueError("disease text must be [ASD,TD] x 1024")
        self.register_buffer("disease_text", disease_text.detach().float().clone())
        self.query = nn.Linear(144, key_dim, bias=False)
        self.text_key = nn.Linear(1024, key_dim, bias=False)
        self.memory_key = nn.Linear(144, key_dim, bias=False)
        self.value = nn.Linear(144, key_dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.role_confidence = nn.Linear(144 + 4*32 + 12, 4)
        composer_in = 144 + 4*32 + 32 + 2*32 + 32 + 12
        self.evidence_composer = nn.Sequential(nn.Linear(composer_in, 96), nn.GELU(), nn.Linear(96, 144))
        self.rho_head = nn.Sequential(nn.Linear(composer_in, 48), nn.GELU(), nn.Linear(48, 1))
        for layer in (self.query, self.text_key, self.memory_key, self.value):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.role_confidence.weight); nn.init.zeros_(self.role_confidence.bias)
        nn.init.xavier_uniform_(self.evidence_composer[0].weight); nn.init.zeros_(self.evidence_composer[0].bias)
        nn.init.xavier_uniform_(self.evidence_composer[2].weight)
        with torch.no_grad(): self.evidence_composer[2].weight.mul_(0.01)
        nn.init.zeros_(self.evidence_composer[2].bias)
        for layer in (self.rho_head[0], self.rho_head[2]):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.rho_head[-1].weight)
        nn.init.constant_(self.rho_head[-1].bias, -4.5)

    def _centers(self, memory, assignment, n_slots, query_ids, training):
        x = memory.features; valid = memory.valid & (assignment >= 0)
        sums = torch.stack([x[valid & (assignment == j)].sum(0) for j in range(n_slots)])
        counts = torch.stack([(valid & (assignment == j)).sum().to(x.dtype) for j in range(n_slots)])
        centers = sums.unsqueeze(0).expand(query_ids.numel(), -1, -1).clone()
        support = counts.unsqueeze(0).expand(query_ids.numel(), -1).clone()
        if training:
            qa = assignment.index_select(0, query_ids)
            rows = (memory.valid.index_select(0, query_ids) & (qa >= 0)).nonzero().flatten()
            if rows.numel():
                slots = qa[rows]
                centers[rows, slots] -= x.index_select(0, query_ids[rows]); support[rows, slots] -= 1
        available = support > 0
        return centers / support.clamp_min(1).unsqueeze(-1), available, support

    def forward(self, h, memory, query_ids, *, typical=True, confusion=False, background=False,
                anonymous_text=False, training=False, residual_enabled=True, text_enabled=True):
        if h.ndim != 2 or h.shape[1] != 144 or h.shape[0] != memory.features.shape[0]:
            raise ValueError(f"expected N x 144 CAM output aligned to memory, got {tuple(h.shape)}")
        text = self.disease_text.to(device=h.device, dtype=h.dtype)
        if anonymous_text:
            generator = torch.Generator(device="cpu").manual_seed(760076)
            rand = torch.randn((2,1024), generator=generator).to(h.device, h.dtype)
            emb = F.normalize(text, dim=-1)
            target_cos = (emb[0]*emb[1]).sum().clamp(-.999,.999)
            a = F.normalize(rand[0], dim=0); b = rand[1] - torch.dot(rand[1],a)*a
            b = target_cos*a + torch.sqrt(1-target_cos.square())*F.normalize(b,dim=0)
            text = torch.stack((a*text[0].norm(), b*text[1].norm()))
        qbase = self.query(h)
        class_text = self.text_key(text) if text_enabled else torch.zeros((2, self.query.out_features), device=h.device, dtype=h.dtype)
        role_evidence=[]; role_strength=[]; role_support=[]; role_valid=[]; role_attn=[]
        # Role order is ASD typical, TD typical, ASD->TD errors, TD->ASD errors.
        specs = [(memory.typical_assignment, 0, typical), (memory.typical_assignment, 1, typical),
                 (memory.hard_assignment, 0, confusion), (memory.hard_assignment, 1, confusion)]
        for assignment, cls, enabled in specs:
            count = 4
            centers, mask, support = self._centers(memory, assignment, count, query_ids, training)
            slots = (assignment.new_tensor([cls*2, cls*2+1]) if assignment is memory.typical_assignment
                     else assignment.new_tensor([cls*2, cls*2+1]))
            centers = centers.index_select(1, slots); mask = mask.index_select(1, slots); support = support.index_select(1, slots)
            q = F.normalize(qbase + self.gamma * class_text[cls].unsqueeze(0), dim=-1)
            keys = F.normalize(self.memory_key(centers), dim=-1)
            score = (q.unsqueeze(1)*keys).sum(-1) / 0.10
            mask = mask & bool(enabled)
            score = score.masked_fill(~mask, -1e4)
            attention = torch.softmax(score, dim=-1) * mask.to(score.dtype)
            attention = attention / attention.sum(-1,keepdim=True).clamp_min(1e-8)
            vals = self.value(centers)
            evidence = (attention.unsqueeze(-1)*vals).sum(1)
            available = mask.any(-1,keepdim=True)
            evidence = evidence * available.to(evidence.dtype)
            strength = (attention*(score*0.10).masked_fill(~mask,0)).sum(-1,keepdim=True)
            role_evidence.append(evidence); role_strength.append(strength)
            role_support.append((support*mask).sum(-1,keepdim=True)); role_valid.append(available)
            role_attn.append(attention)
        ev = torch.cat(role_evidence, dim=-1)
        valid_stats = torch.cat([v.to(h.dtype) for v in role_valid],-1)
        normalized_support = [torch.log1p(value) / math.log(872.0) for value in role_support]
        stats = torch.cat(role_strength + normalized_support + [valid_stats], dim=-1)
        inp = torch.cat((h, ev),-1)
        confidence_logits = self.role_confidence(torch.cat((inp, stats),-1))
        role_mask = torch.cat(role_valid,-1)
        confidence_logits = confidence_logits.masked_fill(~role_mask, -1e4)
        role_weights = torch.softmax(confidence_logits, dim=-1) * role_mask.to(h.dtype)
        role_weights = role_weights / role_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        combined = sum(role_weights[:,j:j+1] * role_evidence[j] for j in range(4))
        asd_evidence = role_evidence[0] + role_evidence[2]
        td_evidence = role_evidence[1] + role_evidence[3]
        class_difference = asd_evidence - td_evidence
        typical_hard_conflict = torch.cat((role_evidence[0] - role_evidence[2],
                                           role_evidence[1] - role_evidence[3]), dim=-1)
        composed = torch.cat((inp, class_difference, typical_hard_conflict, combined, stats),-1)
        reliability = role_mask.to(h.dtype).mean(-1)
        bounded_direction = torch.tanh(self.evidence_composer(composed)) / math.sqrt(144.0)
        rho = 0.05*torch.sigmoid(self.rho_head(composed)).squeeze(-1)
        delta = bounded_direction * rho[:, None] * h.norm(dim=-1, keepdim=True)
        delta = torch.where((~torch.cat(role_valid,-1).any(-1))[:,None], torch.zeros_like(delta), delta)
        if not residual_enabled: delta = torch.zeros_like(delta)
        return {"h_prime": h+delta, "delta":delta, "rho":delta.norm(dim=-1)/h.norm(dim=-1).clamp_min(1e-8),
            "risk":rho, "reliability":reliability, "attention":torch.stack(role_attn, dim=1),
            "bounded_direction_norm":bounded_direction.norm(dim=-1),
            "text_routing":torch.stack([F.normalize(qbase+self.gamma*class_text[c],dim=-1) for c in (0,1)],1),
            "prototype_roles":torch.arange(4,device=h.device), "prototype_classes":torch.tensor([0,1,0,1],device=h.device),
            "prototype_supports":torch.cat(role_support,-1), "prototype_available":torch.cat(role_valid,-1),
            "retrieved_evidence":ev, "role_weights":role_weights, "memory_keys":torch.empty(0,device=h.device)}

    @torch.no_grad()
    def update_memory(self, memory, current_features, ema=.9, epoch=None):
        valid_ids = memory.valid.nonzero().flatten()
        memory.features[valid_ids] = ema*memory.features[valid_ids] + (1-ema)*current_features[valid_ids].detach().float()
        if epoch is not None and epoch % 20 == 0:
            y=memory.labels.detach().cpu().numpy(); x=memory.features.detach().cpu().numpy()
            hard_ids=torch.where(memory.hard_target>0)[0].cpu().numpy()
            typical_ids=torch.where(memory.valid)[0].cpu().numpy()
            for cls in (0,1):
                ids=typical_ids[y[typical_ids]==cls]
                try: memory.typical_assignment[ids]=torch.as_tensor(_kmeans(x[ids],2,761000+int(memory.metadata['fold'])*100+cls)) + 2*cls
                except Exception as e: memory.metadata.setdefault('recluster_failures',[]).append({"epoch":int(epoch),"pool":f"typical_{cls}","error":repr(e)})
                ids=hard_ids[y[hard_ids]==cls]
                if len(ids)>=2:
                    try: memory.hard_assignment[ids]=torch.as_tensor(_kmeans(x[ids],2,762000+int(memory.metadata['fold'])*100+cls)) + 2*cls
                    except Exception as e: memory.metadata.setdefault('recluster_failures',[]).append({"epoch":int(epoch),"pool":f"hard_{cls}","error":repr(e)})


def tensor_sha(module, include_buffers=False):
    digest=hashlib.sha256(); state=module.state_dict() if include_buffers else dict(module.named_parameters())
    for name,tensor in sorted(state.items()): digest.update(name.encode()+b"\0"+tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def memory_sha(state):
    digest=hashlib.sha256()
    for name in sorted(k for k in state if k not in ("metadata","prototype_sha256")):
        value=state[name]; digest.update(name.encode()+b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else repr(value).encode())
    return digest.hexdigest()


def leave_one_out_self_test():
    features = torch.arange(4 * 144, dtype=torch.float32).reshape(4, 144)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    assignment = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    memory = PrototypeMemory(features.clone(), labels, torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.bool), torch.arange(4), assignment, assignment.clone(),
        torch.zeros(4), "", "", [], {"fold": 0})
    module = DCMP(torch.zeros(2, 1024))
    query_ids = torch.arange(4)
    centers, available, support = module._centers(memory, assignment, 4, query_ids, True)
    expected = features[1]
    if not torch.equal(centers[0, 0], expected) or not bool(available[0, 0]) or float(support[0, 0]) != 1.0:
        raise AssertionError("training query remained in its own prototype mean")
    if bool(available[0, 1]) or float(support[0, 1]) != 0.0:
        raise AssertionError("unoccupied prototype slot must remain masked")
    return {"prototype_leave_one_out": True, "empty_slot_masked": True}
