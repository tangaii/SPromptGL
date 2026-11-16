#!/usr/bin/env python3
"""Build the frozen, shared RN50 coordinate system for the E81 prompts."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
EXPECTED_MEDICAL = [
    "Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
    "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def medical_prompts_from_source() -> list[str]:
    source = (REPO / "network.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "attr_words" for target in node.targets):
            prompts = ast.literal_eval(node.value)
            if prompts != EXPECTED_MEDICAL:
                raise RuntimeError(f"root network.py attr_words differs from the frozen 8-prompt specification: {prompts!r}")
            return prompts
    raise RuntimeError("root network.py has no literal attr_words assignment")


def encode(texts: list[str], weights: Path):
    expected = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    if sha_file(weights) != expected:
        raise RuntimeError("CLIP RN50 weights SHA mismatch")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "ftfy" or any(not text.isascii() for text in texts):
            raise
        compat = types.ModuleType("ftfy")
        compat.fix_text = lambda text: text
        sys.modules["ftfy"] = compat
    from CLIP.clip import clip

    tokens = clip.tokenize(texts, truncate=False)
    lengths = (tokens != 0).sum(dim=-1)
    if bool((lengths > 77).any()):
        raise RuntimeError("a prompt exceeds the 77-token CLIP RN50 context")
    model, _ = clip.load(str(weights), device="cpu", jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        encoded = model.encode_text(tokens).float().cpu()
    return encoded, lengths


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    text_config = config["text"]
    medical = medical_prompts_from_source()
    anonymous = list(text_config["anonymous_texts"])
    if len(anonymous) != 8 or len(set(anonymous)) != 8:
        raise RuntimeError("anonymous control must contain eight unique fixed texts")
    forbidden = ("brain", "asd", "autism", "connectivity", "volume", "cortical", "white matter")
    if any(any(token in text.lower() for token in forbidden) for text in anonymous):
        raise RuntimeError("anonymous text contains a prohibited medical term")
    if any(not text.isascii() for text in medical + anonymous):
        raise RuntimeError("E81 prompts must be ASCII")

    weights = Path(text_config["weights"])
    raw_encoded, token_lengths = encode(medical + anonymous, weights)
    if tuple(raw_encoded.shape) != (16, 1024):
        raise RuntimeError(f"unexpected frozen RN50 text shape {tuple(raw_encoded.shape)}")
    raw = F.normalize(raw_encoded.double(), dim=-1)
    _, singular_values, vh = torch.linalg.svd(raw, full_matrices=False)
    tolerance = max(raw.shape) * torch.finfo(raw.dtype).eps * float(singular_values.max())
    rank = int((singular_values > tolerance).sum())
    if not 1 <= rank <= 16:
        raise RuntimeError(f"invalid joint prompt row-space rank {rank}")
    basis = vh[:rank].T.contiguous()
    coords = raw @ basis
    anchors64 = torch.zeros((16, 32), dtype=torch.float64)
    anchors64[:, :rank] = coords
    anchors64 = F.normalize(anchors64, dim=-1)
    raw_gram = raw @ raw.T
    final_gram = anchors64 @ anchors64.T
    gram_error = float((raw_gram - final_gram).abs().max())
    if gram_error > 2e-10:
        raise RuntimeError(f"joint fixed projection changed the prompt Gram matrix by {gram_error}")

    clip_source = REPO / "CLIP/clip"
    source_hashes = {
        str(weights): sha_file(weights),
        str(clip_source / "clip.py"): sha_file(clip_source / "clip.py"),
        str(clip_source / "simple_tokenizer.py"): sha_file(clip_source / "simple_tokenizer.py"),
        str(clip_source / "bpe_simple_vocab_16e6.txt.gz"): sha_file(clip_source / "bpe_simple_vocab_16e6.txt.gz"),
        str(REPO / "network.py"): sha_file(REPO / "network.py"),
    }
    prompts = medical + anonymous
    payload = {
        "medical_texts": medical,
        "anonymous_texts": anonymous,
        "prompt_order": prompts,
        "groups": ["medical_normal"] * 4 + ["medical_abnormal"] * 4 + ["anonymous_group_0"] * 4 + ["anonymous_group_1"] * 4,
        "medical_group_ids": torch.tensor([0] * 4 + [1] * 4, dtype=torch.long),
        "anonymous_group_ids": torch.tensor([0] * 4 + [1] * 4, dtype=torch.long),
        "raw_normalized_embeddings": raw.float(),
        "medical_anchors": anchors64[:8].float(),
        "anonymous_anchors": anchors64[8:].float(),
        "anchors": anchors64.float(),
        "row_space_basis": basis.float(),
        "singular_values": singular_values.float(),
        "effective_rank": rank,
        "token_lengths": token_lengths,
        "gram_raw": raw_gram.float(),
        "gram_final": final_gram.float(),
        "gram_max_abs_error": gram_error,
        "text_dim": 1024,
        "anchor_dim": 32,
        "encoder": "OpenAI CLIP RN50",
        "weights_path": str(weights),
        "weights_sha256": sha_file(weights),
        "source_sha256": source_hashes,
        "normalization": "L2-normalize raw RN50 embeddings; one common float64 joint row-space SVD; zero-pad to 32; L2-normalize final anchors",
        "raw_gram_sha256": tensor_sha(raw_gram),
        "final_gram_sha256": tensor_sha(final_gram),
    }
    cache_hash = hashlib.sha256()
    cache_hash.update(json.dumps({k: v for k, v in payload.items() if not torch.is_tensor(v)}, sort_keys=True).encode())
    for key in ("raw_normalized_embeddings", "medical_anchors", "anonymous_anchors", "anchors", "row_space_basis", "gram_raw", "gram_final"):
        cache_hash.update(key.encode() + tensor_sha(payload[key]).encode())
    payload["cache_sha256"] = cache_hash.hexdigest()
    out = HERE / "runtime/text_embeddings.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({"path": str(out), "cache_sha256": payload["cache_sha256"], "rank": rank,
                      "gram_max_abs_error": gram_error, "medical_prompts": medical,
                      "anonymous_prompts": anonymous, "weights_sha256": payload["weights_sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
