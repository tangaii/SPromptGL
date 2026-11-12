#!/usr/bin/env python3
"""Create the fixed rank-16 coordinates from the archived RN50 text embeddings."""
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
SOURCE_CACHE = Path("/model/spromptgl/experiment/experiment78_dcpss/runtime/text_embeddings.pt")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_prompts():
    tree = ast.parse((REPO / "network.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "attr_words" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("root network.py does not define a literal attr_words list")


def main():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    root = root_prompts()
    if root != cfg["prompts"]["spromptgl_text"]:
        raise RuntimeError("configured SPromptGL prompts differ from root network.py attr_words")
    payload = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    original = torch.cat((payload["spromptgl"], payload["anonymous"]), dim=0).double()
    if original.shape != (16, 1024):
        raise RuntimeError(f"expected 16 frozen RN50 vectors of width 1024, got {tuple(original.shape)}")
    if payload["text_strings"][:8] != root:
        raise RuntimeError("archived RN50 cache prompt strings do not match root attr_words and order")
    anonymous = cfg["prompts"]["anonymous_text"]
    if payload["text_strings"][10:18] != anonymous:
        raise RuntimeError("archived anonymous RN50 prompts do not match E78-G5 order")
    unit = F.normalize(original, dim=-1)
    _, singular, vh = torch.linalg.svd(unit, full_matrices=False)
    rank = int((singular > singular.max() * 1e-10).sum())
    if rank != 16:
        raise RuntimeError(f"the frozen text set has effective rank {rank}, expected 16")
    basis = vh[:rank]
    coords = unit @ basis.T
    original_cos = unit @ unit.T
    projected_cos = F.normalize(coords, dim=-1) @ F.normalize(coords, dim=-1).T
    error = float((original_cos - projected_cos).abs().max())
    if error > 1e-8:
        raise RuntimeError(f"shared semantic subspace failed cosine preservation: {error}")
    artifact = {
        "spromptgl_coords": coords[:8].float(),
        "anonymous_coords": coords[8:].float(),
        "prompt_classes": torch.tensor(cfg["prompts"]["spromptgl_classes"], dtype=torch.long),
        "anonymous_classes": torch.tensor(cfg["prompts"]["anonymous_classes"], dtype=torch.long),
        "basis": basis.float(),
        "original_unit_embeddings": unit.float(),
        "texts": root + anonymous,
        "rank": rank,
        "max_pairwise_cosine_error_float64": error,
        "source_cache_path": str(SOURCE_CACHE),
        "source_cache_sha256": sha_file(SOURCE_CACHE),
        "root_network_sha256": sha_file(REPO / "network.py"),
    }
    output = HERE / "runtime/text_geometry.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(json.dumps({"path": str(output), "rank": rank, "max_pairwise_cosine_error_float64": error,
                      "sha256": sha_file(output)}, indent=2))


if __name__ == "__main__":
    main()
