"""E77 insertion at the audited 144-D PLNNoPEN CAM/classifier boundary."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator

import torch

from memory import ECPSM


HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_embeddings() -> torch.Tensor:
    cache_path = Path(CFG["method"]["semantic"]["cache_path"])
    expected_sha = CFG["method"]["semantic"]["cache_sha256"]
    if sha_file(cache_path) != expected_sha:
        raise RuntimeError(f"audited frozen CLIP text cache SHA mismatch: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    texts = cache.get("text_map", {}).get("original")
    if texts != CFG["method"]["semantic"]["prompt_order"]:
        raise RuntimeError("SPromptGL's eight cached RN50 prompt strings/order changed")
    embeddings = torch.as_tensor(cache.get("embeddings", {}).get("original"), dtype=torch.float32)
    if tuple(embeddings.shape) != (8, 1024) or not torch.isfinite(embeddings).all():
        raise RuntimeError(f"invalid frozen text embedding cache shape/value: {tuple(embeddings.shape)}")
    return embeddings


def build_ecpsm(fold: int, device: torch.device, text_embeddings: torch.Tensor | None = None) -> ECPSM:
    """Initialize new weights in a private RNG scope and restore all base generators."""
    if fold not in range(1, 11):
        raise ValueError(f"invalid fold: {fold}")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device_index).clone() if torch.cuda.is_available() else None
    embeddings = load_text_embeddings() if text_embeddings is None else text_embeddings.detach().cpu()
    seed = int(CFG["randomness"]["new_module_seed_base"]) + fold
    with torch.random.fork_rng(devices=[device_index] if torch.cuda.is_available() else [], enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        memory_cfg = CFG["method"]["prototype"]
        module = ECPSM(
            embeddings,
            n_subjects=CFG["data"]["subjects"],
            class_prototypes=memory_cfg["class_per_class_per_modality"],
            hard_prototypes=memory_cfg["hard_per_direction_per_modality_max"],
            top_k=memory_cfg["top_k"],
            temperature=memory_cfg["retrieval_temperature"],
            semantic_temperature=CFG["method"]["semantic"]["semantic_temperature"],
            ema_momentum=memory_cfg["ema_momentum"],
            kmeans_iterations=memory_cfg["kmeans_iterations"],
        ).to(device)
    if not torch.equal(before_cpu, torch.get_rng_state()):
        raise RuntimeError("EC-PSM construction changed the shared Torch CPU RNG")
    if before_cuda is not None and not torch.equal(before_cuda, torch.cuda.get_rng_state(device_index)):
        raise RuntimeError("EC-PSM construction changed the shared Torch CUDA RNG")
    return module


@contextlib.contextmanager
def cam_boundary_hook(cam, ecpsm: ECPSM, query_ids: torch.Tensor, *,
                      enabled: bool = True, use_hard: bool = False,
                      prototype_off: bool = False, semantic_swap: bool = False,
                      hard_off: bool = False, capture: Dict[str, object] | None = None) -> Iterator[Dict[str, object]]:
    """Transform only the 144-D input; preserve original CAM and OutputLayer calls."""
    state: Dict[str, object] = {}

    def apply(_module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("unexpected Outputlayer pre-hook input")
        raw = args[0]
        if tuple(raw.shape) != (CFG["data"]["subjects"], CFG["data"]["cam_input_dim"]):
            raise RuntimeError(f"unexpected CAM classifier input shape: {tuple(raw.shape)}")
        if capture is not None:
            capture["cam_features"] = raw.detach()
        if not enabled:
            state["info"] = {}
            return None
        transformed, info = ecpsm(
            raw,
            query_ids,
            use_hard=use_hard,
            prototype_off=prototype_off,
            semantic_swap=semantic_swap,
            hard_off=hard_off,
        )
        state["info"] = info
        if capture is not None:
            capture["ecpsm_features"] = transformed.detach()
        return (transformed, *args[1:])

    handle = cam.Outputlayer.register_forward_pre_hook(apply)
    try:
        yield state
    finally:
        handle.remove()


def forward_with_memory(base_runner, state: Dict[str, object], ecpsm: ECPSM,
                        group: str, *, training: bool, memory_epoch: int,
                        intervention: str = "full", prototype_off: bool = False,
                        semantic_swap: bool = False, hard_off: bool = False,
                        capture: Dict[str, object] | None = None):
    state["group"] = group
    ecpsm.train(training)
    use_hard = group == "E77-G2" and memory_epoch >= CFG["method"]["hard_memory"]["warmup_ecpsm_epochs"]
    with cam_boundary_hook(
        state["cam"], ecpsm, torch.arange(CFG["data"]["subjects"], device=state["x"].device),
        enabled=True, use_hard=use_hard, prototype_off=prototype_off,
        semantic_swap=semantic_swap, hard_off=hard_off, capture=capture,
    ) as hook_state:
        output = base_runner.forward(state, "E74.4-G2-Positive-DCR", training=training,
                                     intervention=intervention, collect=False)
    info = hook_state.get("info", {})
    if info:
        ecpsm.record_usage(info)
    return output, hook_state

