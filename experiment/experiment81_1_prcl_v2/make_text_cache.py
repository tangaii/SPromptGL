#!/usr/bin/env python3
"""Create the fixed same-Gram rotated control from the audited E81 RN50 cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RT = HERE / "runtime"
SOURCE = Path(CFG["text"]["source_cache"])
EXPECTED_PROMPTS = CFG["text"]["medical_prompts"]


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("encoder") != "OpenAI CLIP RN50":
        raise RuntimeError("E81 source cache is not CLIP RN50")
    if source.get("weights_sha256") != CFG["text"]["weights_sha256"]:
        raise RuntimeError("E81 CLIP RN50 weight SHA differs from the frozen E81.1 config")
    if source.get("medical_texts") != EXPECTED_PROMPTS:
        raise RuntimeError("E81 source cache does not contain the exact original prompt order")
    if source.get("cache_sha256") is None:
        raise RuntimeError("E81 source cache is missing its source SHA")

    medical = source["medical_anchors"].detach().double()
    rank = int(source["effective_rank"])
    if tuple(medical.shape) != (8, 32) or rank != 16:
        raise RuntimeError(f"expected eight anchors in the 16D E81 row space, got rank={rank}")
    if medical[:, rank:].abs().max().item() > 2e-7:
        raise RuntimeError("E81 anchors contain nonzero values outside their effective row space")

    seed = int(CFG["prcl"]["rotation_seed"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix, triangular = torch.linalg.qr(torch.randn((rank, rank), dtype=torch.float64, generator=generator))
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0] = 1
    matrix = matrix @ torch.diag(signs)
    if torch.linalg.det(matrix).item() < 0:
        matrix[:, -1] *= -1

    control = torch.zeros_like(medical)
    control[:, :rank] = medical[:, :rank] @ matrix
    medical_gram = medical @ medical.T
    control_gram = control @ control.T
    gram_error = float((medical_gram - control_gram).abs().max())
    orthogonal_error = float((matrix.T @ matrix - torch.eye(rank, dtype=matrix.dtype)).abs().max())
    norm_error = float((medical.norm(dim=-1) - control.norm(dim=-1)).abs().max())
    if gram_error > 1e-6 or orthogonal_error > 1e-10 or norm_error > 1e-10:
        raise RuntimeError(f"same-Gram control validation failed: gram={gram_error}, orthogonal={orthogonal_error}, norm={norm_error}")
    if torch.equal(medical.float(), control.float()):
        raise RuntimeError("orthogonal control unexpectedly reproduces the medical anchors")

    rotation_record = {
        "seed": seed,
        "effective_rank": rank,
        "matrix": matrix.float(),
        "matrix_sha256": tensor_sha(matrix.float()),
        "medical_anchors_sha256": tensor_sha(medical.float()),
        "control_anchors_sha256": tensor_sha(control.float()),
        "orthogonality_max_abs_error": orthogonal_error,
        "gram_max_abs_error": gram_error,
        "norm_max_abs_error": norm_error,
        "determinant": float(torch.linalg.det(matrix)),
        "source_cache_sha256": source["cache_sha256"],
    }
    torch.save(rotation_record, RT / "control_rotation.pt")

    payload = dict(source)
    payload["control_anchors"] = control.float()
    payload["control_type"] = "fixed orthogonal rotation of medical anchors in the shared E81 effective 16D row space"
    payload["control_rotation_seed"] = seed
    payload["control_rotation_sha256"] = rotation_record["matrix_sha256"]
    payload["control_anchors_sha256"] = rotation_record["control_anchors_sha256"]
    payload["medical_control_gram_max_abs_error"] = gram_error
    payload["medical_control_norm_max_abs_error"] = norm_error
    payload["cache_source_sha256"] = source["cache_sha256"]
    digest = hashlib.sha256()
    metadata = {key: value for key, value in payload.items() if not torch.is_tensor(value)}
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=True).encode())
    for key in ("raw_normalized_embeddings", "medical_anchors", "control_anchors", "anchors", "row_space_basis", "gram_final"):
        if key in payload and torch.is_tensor(payload[key]):
            digest.update(key.encode() + tensor_sha(payload[key]).encode())
    payload["cache_sha256"] = digest.hexdigest()
    torch.save(payload, RT / "text_embeddings.pt")
    print(json.dumps({
        "cache": str(RT / "text_embeddings.pt"),
        "source_cache_sha256": source["cache_sha256"],
        "cache_sha256": payload["cache_sha256"],
        "rotation_seed": seed,
        "rotation_sha256": rotation_record["matrix_sha256"],
        "effective_rank": rank,
        "gram_max_abs_error": gram_error,
        "orthogonality_max_abs_error": orthogonal_error,
        "anchors_different": True
    }, indent=2))


if __name__ == "__main__":
    main()
