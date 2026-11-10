"""CAM output intervention and frozen local CLIP text cache for SC-PERM."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from memory import SC_PERM

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OLD_TEXT_CACHE = Path("/model/spromptgl/experiment/experiment60_dss_pen/.runtime/text_embeddings.pt")
TEXT_CACHE = HERE / "runtime" / "text_embeddings.pt"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def prepare_text_cache() -> dict:
    clip_dir = Path(CFG["text_encoder"]["encoder_code_root"]) / "CLIP" / "clip"
    clip_sources = {
        "clip_py_sha256": sha_file(clip_dir / "clip.py"),
        "encoder_py_sha256": sha_file(clip_dir / "model.py"),
        "tokenizer_py_sha256": sha_file(clip_dir / "simple_tokenizer.py"),
        "tokenizer_vocab_sha256": sha_file(clip_dir / "bpe_simple_vocab_16e6.txt.gz"),
    }
    expected = {
        "config_sha256": sha_file(HERE / "config.json"),
        "rn50_sha256": sha_file(Path(CFG["text_encoder"]["weights_path"])),
        "original_cache_sha256": sha_file(OLD_TEXT_CACHE),
        **clip_sources,
    }
    if expected["rn50_sha256"] != CFG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("CLIP RN50 checkpoint hash differs from the audited E60 source")
    if TEXT_CACHE.is_file():
        cached = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        if cached.get("source_hashes") == expected:
            return cached
    old = torch.load(OLD_TEXT_CACHE, map_location="cpu", weights_only=False)
    original_texts = old.get("text_map", {}).get("original")
    original_embeddings = old.get("embeddings", {}).get("original")
    if original_texts != CFG["prompts"]["original"] or tuple(original_embeddings.shape) != (8, 1024):
        raise RuntimeError("audited original SPromptGL prompt cache does not match local E60")
    repo_root = Path(CFG["text_encoder"]["encoder_code_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "ftfy":
            raise
        # All candidate text is ASCII, so the tokenizer's text-fix pass is identity.
        import types
        shim = types.ModuleType("ftfy")
        shim.fix_text = lambda value: value
        sys.modules["ftfy"] = shim
    from CLIP.clip import clip

    with torch.random.fork_rng(devices=[], enabled=True):
        model, _ = clip.load(CFG["text_encoder"]["weights_path"], device="cpu", jit=False)
        model.eval()
        original_tokens = clip.tokenize(CFG["prompts"]["original"])
        tokens = clip.tokenize(CFG["prompts"]["medical"])
        with torch.no_grad():
            medical_embeddings = model.encode_text(tokens).float().cpu()
    del model
    payload = {
        "source_hashes": expected,
        "texts": CFG["prompts"],
        "embeddings": {
            "original": original_embeddings.detach().cpu().float(),
            "medical": medical_embeddings,
        },
        "token_lengths": {
            "original": [int((row != 0).sum().item()) for row in original_tokens],
            "medical": [int((row != 0).sum().item()) for row in tokens],
        },
        "text_encoder": {
            "name": "OpenAI CLIP RN50", "version": "local pinned source; see source hashes",
            "sha256": expected["rn50_sha256"], "weights_sha256": expected["rn50_sha256"], "frozen": True,
            "clip_code_sha256": clip_sources["clip_py_sha256"],
            "encoder_code_sha256": clip_sources["encoder_py_sha256"],
            "tokenizer": "OpenAI CLIP byte-level BPE",
            "tokenizer_code_sha256": clip_sources["tokenizer_py_sha256"],
            "tokenizer_vocab_sha256": clip_sources["tokenizer_vocab_sha256"],
        },
    }
    _atomic_torch(TEXT_CACHE, payload)
    return payload


def build_sc_perm(fold: int, group: str, device: torch.device,
                  cache: dict | None = None) -> SC_PERM:
    cache = prepare_text_cache() if cache is None else cache
    prompt_set = CFG["method"]["groups"][group]["text"]
    banks = torch.stack((cache["embeddings"]["original"], cache["embeddings"]["medical"])).reshape(2, 8, 1024)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    private_seed = int(CFG["randomness"]["module_seed_base"]) + int(fold) * 100 + 1
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(private_seed)
        module = SC_PERM(banks, CFG["method"], prompt_set).to(device)
    if not torch.equal(cpu_before, torch.get_rng_state()):
        raise RuntimeError("SC-PERM initialization consumed public CPU RNG")
    if cuda_before is not None and not torch.equal(cuda_before, torch.cuda.get_rng_state()):
        raise RuntimeError("SC-PERM initialization consumed public CUDA RNG")
    return module


@contextlib.contextmanager
def classifier_evidence_hook(cam, module: SC_PERM, query_ids: torch.Tensor, *,
                             enabled: bool = True,
                             hard_enabled: bool, prototype_off: bool = False,
                             hard_off: bool = False, semantic_swap: bool = False,
                             text_source: str | None = None, capture: dict | None = None) -> Iterator[dict]:
    """Capture CAM [N,144], then modify only Outputlayer log probabilities."""
    if cam.Outputlayer._forward_pre_hooks or cam.Outputlayer._forward_hooks:
        raise RuntimeError("unexpected leaked OutputLayer hook before forward")
    state: dict = {"pre_calls": 0, "post_calls": 0, "info": None}

    def capture_input(_module, args):
        state["pre_calls"] += 1
        if state["pre_calls"] != 1 or not args or tuple(args[0].shape) != (871, 144):
            raise RuntimeError(f"unexpected CAM OutputLayer input/call count: {state['pre_calls']}")
        raw = args[0]
        state["raw"] = raw
        if capture is not None:
            capture["cam_features"] = raw.detach()
        if not enabled:
            return None
        evidence = module.compute_evidence(
            raw, query_ids, hard_enabled=hard_enabled, prototype_off=prototype_off,
            hard_off=hard_off, semantic_swap=semantic_swap, text_source=text_source,
        )
        state["info"] = evidence
        return None

    def correct_output(_module, args, output):
        state["post_calls"] += 1
        if state["post_calls"] != 1:
            raise RuntimeError("OutputLayer hook was invoked more than once in one forward")
        if not isinstance(output, tuple) or len(output) != 2 or output[0].shape != (871, 2):
            raise RuntimeError("unexpected original OutputLayer return contract")
        raw_logp, hidden = output
        corrected = raw_logp if state["info"] is None else F.log_softmax(
            raw_logp + state["info"]["correction"], dim=1
        )
        if capture is not None:
            capture["base_logp"] = raw_logp.detach()
            capture["corrected_logp"] = corrected.detach()
            capture["hidden"] = hidden.detach()
            capture["evidence"] = state["info"]
        return corrected, hidden

    pre = cam.Outputlayer.register_forward_pre_hook(capture_input)
    post = cam.Outputlayer.register_forward_hook(correct_output, with_kwargs=False)
    try:
        yield state
    finally:
        pre.remove(); post.remove()
        if cam.Outputlayer._forward_pre_hooks or cam.Outputlayer._forward_hooks:
            raise RuntimeError("SC-PERM left a residual hook on the original OutputLayer")


def forward_sc_perm(base_runner, state: dict, module: SC_PERM, group: str, *,
                    training: bool, local_epoch: int, enabled: bool = True,
                    prototype_off: bool = False,
                    hard_off: bool = False, semantic_swap: bool = False,
                    text_source: str | None = None, capture: dict | None = None):
    state["group"] = "E74.4-G2-Positive-DCR"
    module.train(training)
    hard_enabled = bool(CFG["method"]["groups"][group]["hard_memory"] and local_epoch >= 10)
    query_ids = torch.arange(871, dtype=torch.long, device=state["x"].device)
    with classifier_evidence_hook(
        state["cam"], module, query_ids, enabled=enabled, hard_enabled=hard_enabled,
        prototype_off=prototype_off, hard_off=hard_off,
        semantic_swap=semantic_swap, text_source=text_source, capture=capture,
    ) as hook:
        output = base_runner.forward(state, "E74.4-G2-Positive-DCR", training=training, collect=False)
    return output, hook
