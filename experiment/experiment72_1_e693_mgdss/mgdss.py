"""Fixed text geometry and auxiliary disease supervision for Experiment 72."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
FINE_WEIGHTS = (0.10, 0.10, 0.10, 0.70)
BP_TEXTS = (
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
)
SP_TEXTS = (
    "Nromal brain volume",
    "Uniform cortical thickness",
    "Symmetrical brain structure",
    "Normal ventricular morphology",
    "Altered brain volume",
    "Abnormal functional connectivity",
    "White matter integrity issues",
    "Different activation patterns",
)


def text_sha(texts: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def _make_prompt_space_single_thread(bp_raw: torch.Tensor, sp_raw: torch.Tensor, tolerance: float = 1e-10):
    """Build one deterministic CPU-float64 orthonormal span for the 10 text rows."""
    if tuple(bp_raw.shape) != (2, 1024) or tuple(sp_raw.shape) != (8, 1024):
        raise ValueError(f"Expected [2,1024] and [8,1024] source embeddings, got {bp_raw.shape} and {sp_raw.shape}")
    raw = torch.cat((bp_raw, sp_raw), dim=0).detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Non-finite frozen text embedding")
    _u, singular_values, vh = torch.linalg.svd(raw, full_matrices=False)
    rank_tolerance = max(raw.shape) * torch.finfo(torch.float64).eps * singular_values.max()
    rank = int((singular_values > rank_tolerance).sum())
    if rank < 2 or rank > 10:
        raise ValueError(f"Invalid frozen text span rank {rank}")
    basis = vh[:rank].T.contiguous()
    # SVD vectors are sign-ambiguous; anchor each axis by its largest-magnitude entry.
    for column in range(rank):
        pivot = int(basis[:, column].abs().argmax())
        if basis[pivot, column] < 0:
            basis[:, column].neg_()
    if not torch.allclose(basis.T @ basis, torch.eye(rank, dtype=torch.float64), atol=tolerance, rtol=tolerance):
        raise AssertionError("Text span basis is not orthonormal")

    bp = F.normalize(bp_raw.detach().to(device="cpu", dtype=torch.float64), dim=-1)
    sp = F.normalize(sp_raw.detach().to(device="cpu", dtype=torch.float64), dim=-1)
    td = F.normalize(sp[:4].mean(dim=0), dim=0)
    asd = F.normalize(sp[4:].mean(dim=0), dim=0)
    banks = {
        "BrainPrompt-2": torch.stack((bp[0], bp[1])),
        "SPromptGL-8": torch.stack((asd, td)),
    }
    records = {}
    for name, bank in banks.items():
        coordinates = bank @ basis
        projected_norms = coordinates.norm(dim=-1)
        projected_cosine = F.cosine_similarity(coordinates[:1], coordinates[1:2], dim=-1).item()
        source_cosine = F.cosine_similarity(bank[:1], bank[1:2], dim=-1).item()
        if float((projected_norms - 1.0).abs().max()) > tolerance:
            raise AssertionError(f"{name} prototype norm was not preserved by text span projection")
        if abs(projected_cosine - source_cosine) > tolerance:
            raise AssertionError(f"{name} prototype cosine was not preserved by text span projection")
        normalized_coordinates = F.normalize(coordinates, dim=-1).contiguous()
        records[name] = {
            "bank": normalized_coordinates.float(),
            "source_bank": bank.contiguous(),
            "raw_norms": bank.norm(dim=-1).tolist(),
            "projected_norms": projected_norms.tolist(),
            "source_cosine": source_cosine,
            "projected_cosine": projected_cosine,
            "max_cosine_error": abs(projected_cosine - source_cosine),
            "bank_sha256": tensor_sha(normalized_coordinates),
        }
    return basis, rank, singular_values, records


def make_prompt_space(bp_raw: torch.Tensor, sp_raw: torch.Tensor, tolerance: float = 1e-10):
    """Freeze deterministic CPU float64 SVD output independent of runtime thread count."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _make_prompt_space_single_thread(bp_raw, sp_raw, tolerance)
    finally:
        torch.set_num_threads(previous_threads)


class MGDSS(nn.Module):
    """Shared fine and global patient projections; disease banks remain frozen externally."""

    def __init__(self, rank: int):
        super().__init__()
        self.p_f = nn.Linear(36, int(rank), bias=True)
        self.p_g = nn.Linear(144, int(rank), bias=True)
        nn.init.xavier_uniform_(self.p_f.weight)
        nn.init.zeros_(self.p_f.bias)
        nn.init.xavier_uniform_(self.p_g.weight)
        nn.init.zeros_(self.p_g.bias)

    def forward(self, cam: torch.Tensor, bank: torch.Tensor, temperature: float, use_fine: bool):
        if cam.ndim != 3 or tuple(cam.shape[1:]) != (4, 36):
            raise ValueError(f"Expected CAM positions [N,4,36], got {tuple(cam.shape)}")
        if bank.ndim != 2 or tuple(bank.shape) != (2, self.p_g.out_features):
            raise ValueError(f"Expected frozen [2,{self.p_g.out_features}] class bank, got {tuple(bank.shape)}")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        global_patient = F.normalize(self.p_g(cam.reshape(cam.shape[0], 144)), dim=-1)
        global_bank = F.normalize(bank.to(device=cam.device, dtype=cam.dtype), dim=-1)
        global_logits = (global_patient @ global_bank.T) / float(temperature)
        out = {
            "global_patient": global_patient,
            "global_logits": global_logits,
            "fine_patient": None,
            "fine_logits": None,
            "fine_losses": None,
            "fine_loss": None,
        }
        if use_fine:
            fine_patient = F.normalize(self.p_f(cam), dim=-1)
            fine_logits = (fine_patient @ global_bank.T) / float(temperature)
            out["fine_patient"] = fine_patient
            out["fine_logits"] = fine_logits
        return out


def supervised_losses(output: dict, train_labels: torch.Tensor) -> dict:
    """Loss API accepts training labels only; callers pass training CAM rows only."""
    labels = train_labels.long().reshape(-1)
    if output["global_logits"].shape[0] != labels.numel():
        raise ValueError("Patient logits and outer-training labels have different lengths")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Expected labels ASD=0 and TD=1")
    result = {"global": F.cross_entropy(output["global_logits"], labels)}
    if output["fine_logits"] is not None:
        if output["fine_logits"].shape[:2] != (labels.numel(), 4):
            raise ValueError("Fine logits must have shape [N_train,4,2]")
        per_modality = [F.cross_entropy(output["fine_logits"][:, m, :], labels) for m in range(4)]
        result["fine_modalities"] = per_modality
        result["fine"] = sum(weight * loss for weight, loss in zip(FINE_WEIGHTS, per_modality))
    else:
        result["fine_modalities"] = None
        result["fine"] = None
    return result


def class_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long().reshape(-1))


def saber_class_ce(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits / float(temperature)
    indices = labels.long().reshape(-1, 1)
    return (-scaled.gather(1, indices).squeeze(1) + torch.logsumexp(scaled, dim=1)).mean()


def loss_equivalence_audit(device: torch.device = torch.device("cpu")) -> dict:
    """Test CE/log-sum-exp value and gradient equivalence at SABER tau=1."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(720072)
        values = torch.randn(11, 10, dtype=torch.float64, device=device, requires_grad=True)
        bank = F.normalize(torch.randn(2, 10, dtype=torch.float64, device=device), dim=-1)
        patients = F.normalize(values, dim=-1)
        similarities = patients @ bank.T
        labels = torch.tensor((0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1), device=device)
        brainprompt = F.cross_entropy(similarities, labels)
        saber = saber_class_ce(similarities, labels, 1.0)
        gradient_bp = torch.autograd.grad(brainprompt, values, retain_graph=True)[0]
        gradient_saber = torch.autograd.grad(saber, values)[0]
    loss_error = float((brainprompt - saber).abs().detach().cpu())
    gradient_error = float((gradient_bp - gradient_saber).abs().max().detach().cpu())
    if loss_error > 1e-12 or gradient_error > 1e-12:
        raise AssertionError(f"BrainPrompt/SABER tau=1 equivalence failed: {loss_error=}, {gradient_error=}")
    return {"loss_abs_error": loss_error, "max_gradient_abs_error": gradient_error, "status": "PASS"}
