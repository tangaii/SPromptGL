"""Graph-conditioned semantic-prototypical evidence reasoning for E75.2."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


EPS = 1e-8
PROMPT_CLASS_INDEX = (1, 1, 1, 1, 0, 0, 0, 0)  # cached order TD then ASD; model labels ASD=0, TD=1


@dataclass
class PrototypeBank:
    full: torch.Tensor             # [2, 16, 9]
    crossfit: torch.Tensor         # [5, 2, 16, 9]
    assignment: torch.Tensor       # [871], -1 for non-prototype nodes
    source_indices: torch.Tensor   # eligible prototype-training nodes
    class_counts: torch.Tensor    # [2]
    partition_counts: torch.Tensor  # [5, 2]
    source_sha256: str
    bank_sha256: str


def _normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=EPS)


def _sha_tensor(value: torch.Tensor) -> str:
    import hashlib
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def make_anonymous_embedding(real: torch.Tensor, seed: int):
    """Rotate already-projected real embeddings, preserving their Gram matrix."""
    if real.ndim != 2 or real.shape[1] != 128:
        raise ValueError("projected semantic embeddings must have shape [8,128]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((128, 128), generator=generator, dtype=torch.float64)
    orthogonal, _ = torch.linalg.qr(matrix)
    anonymous = (real.double().cpu() @ orthogonal).to(dtype=real.dtype, device=real.device)
    error = float((real.cpu() @ real.cpu().T - anonymous.cpu() @ anonymous.cpu().T).abs().max())
    if not math.isfinite(error) or error >= 1e-5:
        raise RuntimeError(f"projected anonymous Gram mismatch: {error}")
    return anonymous, orthogonal.float(), error


def build_prototype_bank(tokens: torch.Tensor, labels: torch.Tensor, train_ids: torch.Tensor,
                         excluded_ids: torch.Tensor, n_folds: int = 5) -> PrototypeBank:
    """Build class/token prototypes and deterministic class-stratified cross-fit banks."""
    import hashlib
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (16, 9):
        raise ValueError(f"expected CAM tokens [N,16,9], got {tuple(tokens.shape)}")
    device = tokens.device
    train_ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    excluded_ids = torch.as_tensor(excluded_ids, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    if excluded_ids.numel():
        keep = ~torch.isin(train_ids, excluded_ids)
        source_ids = train_ids[keep]
    else:
        source_ids = train_ids
    source_labels = labels.index_select(0, source_ids)
    if torch.any((source_labels != 0) & (source_labels != 1)):
        raise RuntimeError("prototype source contains labels outside ASD=0/TD=1")
    normalized = _normalize(tokens.index_select(0, source_ids))
    assignment = torch.full((tokens.shape[0],), -1, dtype=torch.long, device=device)
    sums = tokens.new_zeros((2, 16, 9))
    counts = torch.zeros((2,), dtype=torch.long, device=device)
    parts = [torch.zeros((2, 16, 9), dtype=tokens.dtype, device=device) for _ in range(n_folds)]
    part_counts = torch.zeros((n_folds, 2), dtype=torch.long, device=device)
    for cls in (0, 1):
        local = source_ids[source_labels == cls]
        if local.numel() <= n_folds:
            raise RuntimeError(f"fold has too few class-{cls} prototype nodes: {local.numel()}")
        # Sorted subject IDs and round-robin assignment are deterministic and label-stratified.
        local = local.sort().values
        for rank, subject in enumerate(local.tolist()):
            assignment[subject] = rank % n_folds
        local_pos = (source_labels == cls).nonzero(as_tuple=True)[0]
        cls_tokens = normalized.index_select(0, local_pos)
        sums[cls] = cls_tokens.sum(dim=0)
        counts[cls] = local.numel()
        for part in range(n_folds):
            member = local[torch.arange(local.numel(), device=device) % n_folds == part]
            part_counts[part, cls] = member.numel()
            positions = torch.isin(source_ids, member).nonzero(as_tuple=True)[0]
            parts[part][cls] = normalized.index_select(0, positions).sum(dim=0)
    if torch.any(counts == 0) or torch.any(part_counts == 0):
        raise RuntimeError("empty class in full or cross-fit prototype bank")
    full = _normalize(sums / counts[:, None, None])
    crossfit = []
    for part in range(n_folds):
        rem_counts = counts - part_counts[part]
        if torch.any(rem_counts <= 0):
            raise RuntimeError("cross-fit prototype exclusion left an empty class")
        crossfit.append(_normalize((sums - parts[part]) / rem_counts[:, None, None]))
    crossfit_tensor = torch.stack(crossfit)
    source_bytes = source_ids.detach().cpu().numpy().astype("int64", copy=False).tobytes()
    source_bytes += source_labels.detach().cpu().numpy().astype("int64", copy=False).tobytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    bank_sha = hashlib.sha256(full.detach().cpu().contiguous().numpy().tobytes() +
                              crossfit_tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return PrototypeBank(full.detach(), crossfit_tensor.detach(), assignment.detach(), source_ids.detach(),
                         counts.detach(), part_counts.detach(), source_sha, bank_sha)


class GCSper(nn.Module):
    def __init__(self, text_embeddings: torch.Tensor, projection_seed: int, init_seed: int,
                 anonymous: bool = False, anonymous_seed: int | None = None):
        super().__init__()
        if tuple(text_embeddings.shape) != (8, 1024):
            raise ValueError(f"expected frozen SPromptGL-8 embeddings [8,1024], got {tuple(text_embeddings.shape)}")
        # torch.manual_seed also seeds CUDA generators; save the active accelerator
        # state so private module initialization cannot reset the inherited base RNG.
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(init_seed))
            self.graph_mlp = nn.Sequential(nn.Linear(320, 128), nn.GELU(), nn.LayerNorm(128))
            self.scale_asd = nn.Linear(128, 128)
            self.shift_asd = nn.Linear(128, 128)
            self.scale_td = nn.Linear(128, 128)
            self.shift_td = nn.Linear(128, 128)
            self.key = nn.Linear(9, 128, bias=False)
            self.alpha_raw = nn.Parameter(torch.tensor(math.log(0.4 / 0.6), dtype=torch.float32))
            self.gamma_raw = nn.Parameter(torch.tensor(math.log(0.4 / 0.6), dtype=torch.float32))
        generator = torch.Generator(device="cpu").manual_seed(int(projection_seed))
        projection = torch.randn((1024, 128), generator=generator, dtype=torch.float64) / math.sqrt(128.0)
        real = _normalize((text_embeddings.detach().cpu().double() @ projection).float())
        if anonymous:
            if anonymous_seed is None:
                raise ValueError("anonymous_seed is required for G3")
            semantic, orthogonal, gram_error = make_anonymous_embedding(real, anonymous_seed)
        else:
            semantic, orthogonal, gram_error = real, torch.eye(128), 0.0
        self.register_buffer("semantic_embeddings", semantic.detach().float())
        self.register_buffer("semantic_projection", projection.float())
        self.register_buffer("anonymous_orthogonal", orthogonal.float())
        self.register_buffer("raw_text_embeddings", text_embeddings.detach().float().cpu())
        self.register_buffer("gram_error", torch.tensor(float(gram_error), dtype=torch.float32))

    @property
    def alpha(self):
        return 0.5 * torch.sigmoid(self.alpha_raw)

    @property
    def gamma(self):
        return 0.25 * torch.sigmoid(self.gamma_raw)

    def evidence(self, h144: torch.Tensor, f320: torch.Tensor, bank: PrototypeBank,
                 train_ids: torch.Tensor, group: str, intervention: str = "full"):
        if h144.ndim != 2 or h144.shape[-1] != 144:
            raise ValueError("CAM Outputlayer input must be [N,144]")
        if f320.shape != (h144.shape[0], 320):
            raise ValueError("captured DARG output must be [N,320]")
        tokens = h144.reshape(-1, 16, 9)
        graph_context = self.graph_mlp(f320)
        prompts = self.semantic_embeddings.to(device=h144.device, dtype=h144.dtype)
        keys = _normalize(self.key(tokens))
        queries = []
        for prompt_idx in range(8):
            if prompt_idx < 4:
                scale, shift = self.scale_td(graph_context), self.shift_td(graph_context)
            else:
                scale, shift = self.scale_asd(graph_context), self.shift_asd(graph_context)
            q = prompts[prompt_idx].unsqueeze(0) * (1.0 + 0.2 * torch.tanh(scale)) + 0.2 * torch.tanh(shift)
            queries.append(_normalize(q))
        query = torch.stack(queries, dim=1)  # [N,8,128]
        scores = torch.einsum("nkd,npd->nkp", keys, query) / 0.2
        e_td = torch.logsumexp(scores[..., :4], dim=-1) - math.log(4.0)
        e_asd = torch.logsumexp(scores[..., 4:], dim=-1) - math.log(4.0)
        e_sem = torch.stack((e_asd, e_td), dim=-1)
        p_sem = torch.softmax(e_sem, dim=-1)

        proto_bank = bank.full.to(device=h144.device, dtype=h144.dtype)
        assignments = bank.assignment.to(h144.device)
        ids = torch.arange(h144.shape[0], device=h144.device)
        is_training_member = assignments >= 0
        # All members of the allowed outer training set use leave-partition-out prototypes.
        train_ids = torch.as_tensor(train_ids, dtype=torch.long, device=h144.device)
        if train_ids.numel():
            train_mask = torch.zeros_like(is_training_member)
            train_mask[train_ids] = True
            is_training_member &= train_mask
        selected_proto = proto_bank.unsqueeze(0).expand(h144.shape[0], -1, -1, -1).clone()
        xfit = bank.crossfit.to(device=h144.device, dtype=h144.dtype)
        selected_proto[is_training_member] = xfit[assignments[is_training_member]]
        p_asd = selected_proto[:, 0]
        p_td = selected_proto[:, 1]
        token_norm = _normalize(tokens)
        e_proto_asd = F.cosine_similarity(token_norm, p_asd, dim=-1, eps=EPS) / 0.5
        e_proto_td = F.cosine_similarity(token_norm, p_td, dim=-1, eps=EPS) / 0.5
        e_proto = torch.stack((e_proto_asd, e_proto_td), dim=-1)
        p_proto = torch.softmax(e_proto, dim=-1)
        class_separation = 1.0 - F.cosine_similarity(p_asd, p_td, dim=-1, eps=EPS)
        r_sep = torch.sigmoid(4.0 * (class_separation - 0.3)).clamp(0.0, 1.0)
        r_conf = 0.5 + 0.5 * (p_proto[..., 0] - p_proto[..., 1]).abs()
        use_semantic = group in ("G2-GC-SPER", "G3-Anonymous")
        if intervention == "semantic_off":
            use_semantic = False
        if intervention == "graph_context_off":
            # Recompute the query without patient-specific context while preserving fixed prompts.
            query0 = _normalize(prompts)
            scores0 = torch.einsum("nkd,pd->nkp", keys, query0) / 0.2
            td0 = torch.logsumexp(scores0[..., :4], dim=-1) - math.log(4.0)
            asd0 = torch.logsumexp(scores0[..., 4:], dim=-1) - math.log(4.0)
            p_sem = torch.softmax(torch.stack((asd0, td0), dim=-1), dim=-1)
        if intervention == "prompt_class_swap":
            swapped = torch.stack((e_sem[..., 1], e_sem[..., 0]), dim=-1)
            p_sem = torch.softmax(swapped, dim=-1)
        if use_semantic:
            alpha = self.alpha.to(dtype=h144.dtype)
            p_joint = (1.0 - alpha) * p_proto + alpha * p_sem
            r_agree = 0.5 + 0.5 * torch.exp(-(p_sem[..., 0] - p_proto[..., 0]).abs() / 0.3)
            if intervention == "agreement_off":
                r_agree = torch.ones_like(r_agree)
        else:
            alpha = h144.new_zeros(())
            p_joint = p_proto
            r_agree = torch.ones_like(r_conf)
        if intervention == "module_off":
            p_joint = p_proto
        target = _normalize(p_joint[..., 0:1] * p_asd + p_joint[..., 1:2] * p_td)
        rms = torch.sqrt(tokens.square().mean(dim=-1, keepdim=True).clamp_min(EPS))
        raw_direction = rms * (target - token_norm)
        bounded = rms * torch.tanh(raw_direction / rms.clamp_min(EPS))
        reliability = (r_sep * r_conf * r_agree).clamp(0.0, 1.0)
        if intervention == "module_off":
            residual = torch.zeros_like(bounded)
        else:
            residual = self.gamma.to(dtype=h144.dtype) * reliability.unsqueeze(-1) * bounded
        h_prime = (tokens + residual).reshape(-1, 144)
        return {
            "h_prime": h_prime, "tokens": tokens, "tokens_norm": token_norm,
            "graph_context": graph_context, "queries": query, "semantic_scores": scores,
            "e_sem": e_sem, "p_sem": p_sem, "e_proto": e_proto, "p_proto": p_proto,
            "p_joint": p_joint, "r_sep": r_sep, "r_conf": r_conf, "r_agree": r_agree,
            "reliability": reliability, "target": target, "raw_direction": raw_direction,
            "bounded_residual": bounded, "residual": residual,
            "alpha": alpha, "gamma": self.gamma,
        }

    def auxiliary_losses(self, result, labels: torch.Tensor, aux_ids: torch.Tensor, group: str):
        if aux_ids.numel() == 0:
            zero = result["e_proto"].new_zeros(())
            return zero, zero
        ids = torch.as_tensor(aux_ids, dtype=torch.long, device=result["e_proto"].device)
        e_proto = result["e_proto"].index_select(0, ids)
        p_proto = result["p_proto"].index_select(0, ids).detach()
        p_sem = result["p_sem"].index_select(0, ids)
        r_sep = result["r_sep"].index_select(0, ids).detach()
        y = torch.as_tensor(labels, dtype=torch.long, device=ids.device).index_select(0, ids)
        weight = r_sep / r_sep.sum(dim=1, keepdim=True).clamp_min(EPS)
        aggregate = (e_proto * weight.unsqueeze(-1)).sum(dim=1)
        l_proto = F.cross_entropy(aggregate, y)
        if group == "G1-Proto":
            l_align = aggregate.new_zeros(())
        else:
            log_p_sem = torch.log(result["p_sem"].index_select(0, ids).clamp_min(EPS))
            per_token_kl = F.kl_div(log_p_sem, p_proto, reduction="none").sum(dim=-1)
            l_align = (per_token_kl * r_sep).sum() / r_sep.sum().clamp_min(EPS)
        return l_proto, l_align
