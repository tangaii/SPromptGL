#!/usr/bin/env python3
"""Isolated runner for the 30 formal folds of Experiment 81.6.2."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sa_prcl import MEDICAL, PROMPTS, SCRAMBLED, SAPRCL, residual_penalty

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
GROUPS = tuple(CFG["groups"])
BASE_GROUP = "E74.4-G2-Positive-DCR"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
BASE = None
PATH_ADAPTER = None
CACHE = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_value(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"N" + str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    body = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp.write_text(body, encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if path.read_text(encoding="utf-8") != body:
        raise IOError(f"JSON readback failed: {path}")


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint readback failed: {path}")


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_journal(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    body = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in rows)
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if path.read_text(encoding="utf-8") != body:
        raise IOError(f"scalar journal readback failed: {path}")


def runtime():
    global BASE, PATH_ADAPTER
    if BASE is None:
        adapter_path = HERE / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("experiment81_6_2_path_adapter", adapter_path)
        PATH_ADAPTER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = PATH_ADAPTER
        spec.loader.exec_module(PATH_ADAPTER)
        BASE = PATH_ADAPTER.load_base_runner()
    return BASE


def legacy_runtime():
    base = runtime()
    return base.load_history().load_e69().load_e61().load_base()


def prompt_cache():
    global CACHE
    if CACHE is None:
        CACHE = torch_load(HERE / "prompt_features.pt")
    if CACHE.get("prompts") != PROMPTS:
        raise RuntimeError("prompt text/order differs from the frozen cache")
    if tuple(CACHE["raw_features"].shape) != (8, 1024) or tuple(CACHE["semantic_basis"].shape) != (8, 8):
        raise RuntimeError("frozen text/SVD cache has an unexpected shape")
    if not torch.isfinite(CACHE["raw_features"]).all() or not torch.isfinite(CACHE["semantic_basis"]).all():
        raise RuntimeError("frozen text/SVD cache contains a nonfinite value")
    normalized64 = F.normalize(CACHE["raw_features"].double(), dim=-1)
    source_gram = normalized64 @ normalized64.T
    basis = CACHE["semantic_basis_float64"].double()
    basis_gram = basis @ basis.T
    error = float((source_gram - basis_gram).abs().max())
    if error > float(CFG["semantic_geometry"]["gram_tolerance_float64"]):
        raise RuntimeError(f"frozen SVD Gram audit failed: {error}")
    return CACHE


def make_prompt_cache():
    """Encode the eight original strings once and freeze their row-Gram geometry."""
    target = HERE / "prompt_features.pt"
    if target.exists():
        prompt_cache()
        print(f"Prompt/SVD cache already frozen: {sha_file(target)}", flush=True)
        return
    root = Path("/data/adodas/SPromptGL-main")
    weights = Path(CFG["text"]["weights"])
    if sha_file(weights) != CFG["text"]["weights_sha256"]:
        raise RuntimeError("OpenAI CLIP RN50 checkpoint SHA mismatch")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "ftfy":
            raise
        import types
        compat = types.ModuleType("ftfy")
        compat.fix_text = lambda value: value
        sys.modules["ftfy"] = compat
    from CLIP.clip import clip

    tokens = clip.tokenize(PROMPTS, truncate=False)
    model, _ = clip.load(str(weights), device="cpu", jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        raw = model.encode_text(tokens).float().cpu()
    if tuple(raw.shape) != (8, 1024) or not torch.isfinite(raw).all():
        raise RuntimeError(f"unexpected frozen text feature: {tuple(raw.shape)}")

    t64 = F.normalize(raw.double(), dim=-1)
    u, singular, vh = torch.linalg.svd(t64, full_matrices=False)
    for column in range(u.shape[1]):
        pivot = int(torch.argmax(u[:, column].abs()).item())
        if u[pivot, column] < 0:
            u[:, column].neg_()
            vh[column].neg_()
    basis64 = u @ torch.diag(singular)
    gram_t = t64 @ t64.T
    gram_e64 = basis64 @ basis64.T
    gram_error64 = float((gram_t - gram_e64).abs().max())
    basis32 = basis64.float()
    gram_error32 = float((gram_t.float() - basis32 @ basis32.T).abs().max())
    norms = basis32.norm(dim=-1)
    rank_tol = float(max(t64.shape) * torch.finfo(torch.float64).eps * singular.max())
    numerical_rank = int((singular > rank_tol).sum().item())
    tol64 = float(CFG["semantic_geometry"]["gram_tolerance_float64"])
    tol32 = float(CFG["semantic_geometry"]["gram_tolerance_float32"])
    norm_tol = float(CFG["semantic_geometry"]["norm_tolerance"])
    if gram_error64 > tol64 or gram_error32 > tol32 or float((norms - 1.0).abs().max()) > norm_tol:
        raise RuntimeError(
            f"fixed SVD geometry failed preregistered tolerances: gram64={gram_error64}, gram32={gram_error32}"
        )
    payload = {
        "prompts": PROMPTS,
        "raw_features": raw,
        "normalized_features": t64.float(),
        "semantic_basis": basis32,
        "semantic_basis_float64": basis64,
        "svd_u": u,
        "svd_s": singular,
        "svd_vh": vh,
        "gram_original_float64": gram_t,
        "gram_basis_float64": gram_e64,
        "gram_basis_float32": basis32 @ basis32.T,
        "token_ids": tokens.cpu(),
        "encoder": "OpenAI CLIP RN50, frozen, CPU, eval",
        "weights_path": str(weights),
        "weights_sha256": sha_file(weights),
        "text_source_sha256": {
            str(root / "CLIP/clip" / name): sha_file(root / "CLIP/clip" / name)
            for name in ("clip.py", "model.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz")
        },
        "svd": {
            "dtype": "float64",
            "full_matrices": False,
            "sign_rule": "make the U element at each column's first max-absolute row nonnegative; flip matching Vh row",
            "singular_values": singular.tolist(),
            "rank_tolerance": rank_tol,
            "numerical_rank": numerical_rank,
            "retained_columns": 8,
            "gram_max_abs_error_float64": gram_error64,
            "gram_max_abs_error_float32": gram_error32,
            "row_norms_float32": norms.tolist(),
            "gram_tolerances": {"float64": tol64, "float32": tol32, "norm": norm_tol},
        },
        "label_independent": True,
        "trainable_text_projection": False,
        "trainable_prompt_embedding": False,
    }
    atomic_torch(target, payload)
    CACHE = payload
    print(json.dumps({"cache": str(target), "sha256": sha_file(target), "svd": payload["svd"]}, ensure_ascii=False), flush=True)


def source_map():
    base = runtime()
    paths = [HERE / name for name in ("config.json", "design.md", "path_adapter.py", "sa_prcl.py", "run.py", "audit.py", "prompt_features.pt")]
    paths.extend(PATH_ADAPTER.resolve(Path(p)) for p in base.source_manifest())
    old_root = Path("/model/spromptgl/experiment")
    paths.extend(
        old_root / "experiment81_6_prcl_res" / name
        for name in ("prcl_res.py", "run.py", "audit.py", "config.json", "design.md", "manifest.json", "results/fold_metrics.csv", "results/report.md")
    )
    paths.extend(
        old_root / "experiment81_6_1_contrastive_prcl_res" / name
        for name in ("cprl_res.py", "run.py", "audit.py", "config.json", "design.md", "manifest.json", "results/fold_metrics.csv", "results/report.md", "results/mechanism.csv")
    )
    paths.extend(
        [
            Path("/data/adodas/SPromptGL-main/CLIP/clip/clip.py"),
            Path("/data/adodas/SPromptGL-main/CLIP/clip/model.py"),
            Path("/data/adodas/SPromptGL-main/CLIP/clip/simple_tokenizer.py"),
            Path("/data/adodas/SPromptGL-main/CLIP/clip/bpe_simple_vocab_16e6.txt.gz"),
            Path(CFG["text"]["weights"]),
        ]
    )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing frozen provenance: " + ", ".join(missing))
    return {str(Path(path).resolve()): sha_file(path) for path in sorted(set(paths), key=str)}


def source_sha():
    return sha_value(source_map())


def prepare(fold):
    prompt_cache()
    base = runtime()
    state, warmup, identity = base.prepare(fold)
    torch.set_num_threads(1)
    state["fold"] = fold
    state["sa_identity"] = identity
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"fold {fold}: inherited overlap differs from the predeclared protocol")
    if tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError(f"fold {fold}: expected 4096 inherited training pairs")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError(f"fold {fold}: inherited pair has a non-training endpoint")
    if not hasattr(state["cam"], "Outputlayer"):
        raise RuntimeError("restored history CAM lacks the original Outputlayer")
    return state, warmup, identity


def make_module(fold, device):
    cache = prompt_cache()
    return SAPRCL.initialized(CFG["architecture"]["private_salt_base"] + fold, device, cache["semantic_basis"], CFG["architecture"]["rho"])


def sa_optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)


def _rng_cpu_cuda():
    return {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _set_rng_cpu_cuda(state):
    torch.set_rng_state(state["cpu"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _replay_dropout(value, pre_rng, p, training):
    if not training:
        return F.dropout(value, p=p, training=False)
    post_rng = _rng_cpu_cuda()
    try:
        _set_rng_cpu_cuda(pre_rng)
        return F.dropout(value, p=p, training=True)
    finally:
        _set_rng_cpu_cuda(post_rng)


def frozen_outputlayer(state, z_cam, *, training, dropout_pre_rng=None):
    layer = state["cam"].Outputlayer
    norm, linear, gelu, dropout = layer.mlp_head
    if abs(float(dropout.p) - 0.35) > 1e-12:
        raise RuntimeError(f"historical OutputLayer dropout differs from 0.35: {dropout.p}")
    hidden0 = F.layer_norm(z_cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    hidden0 = F.linear(hidden0, linear.weight.detach(), linear.bias.detach())
    hidden0 = F.gelu(hidden0)
    if training and dropout_pre_rng is None:
        raise RuntimeError("training frozen classifier requires captured historical dropout RNG")
    hidden = _replay_dropout(hidden0, dropout_pre_rng, dropout.p, training)
    logits = F.linear(hidden, layer.classifier.weight.detach(), layer.classifier.bias.detach())
    return F.log_softmax(logits, dim=1), hidden, hidden0


def _attach_capture(state, capture, *, capture_dropout):
    layer = state["cam"].Outputlayer

    def output_pre(_module, args):
        capture["z_cam"] = args[0]

    hooks = [layer.register_forward_pre_hook(output_pre)]
    if capture_dropout:
        dropout = layer.mlp_head[3]

        def dropout_pre(_module, args):
            capture["dropout_pre_rng"] = _rng_cpu_cuda()
            capture["dropout_input"] = args[0].detach().clone()

        def dropout_post(_module, _args, output):
            capture["dropout_output"] = output.detach().clone()

        hooks.append(dropout.register_forward_pre_hook(dropout_pre))
        hooks.append(dropout.register_forward_hook(dropout_post))
    return hooks


def _remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def base_forward(state, training):
    return runtime().forward(state, BASE_GROUP, training=training)


def protected_forward(state, module, grouping, *, training, intervention="full", blocked_rows=()):
    capture = {}
    hooks = _attach_capture(state, capture, capture_dropout=training)
    try:
        base_out = base_forward(state, training=training)
    finally:
        _remove_hooks(hooks)
    if "z_cam" not in capture:
        raise RuntimeError("read-only OutputLayer hook did not capture CAM144")
    public_after_base = legacy_runtime().pack_rng()
    input_z = capture["z_cam"].detach()
    enhanced, details = module(input_z, grouping=grouping, eta=0.0, intervention=intervention, blocked_rows=blocked_rows)
    branch_logp, branch_hidden, branch_pre_dropout = frozen_outputlayer(
        state,
        enhanced,
        training=training,
        dropout_pre_rng=capture.get("dropout_pre_rng"),
    )
    public_after_branch = legacy_runtime().pack_rng()
    if legacy_runtime().rng_sha(public_after_base) != legacy_runtime().rng_sha(public_after_branch):
        raise RuntimeError("frozen SA classifier changed the public DARG RNG trajectory")
    base_out.update(
        base_logp=base_out["logp"],
        base_hidden=base_out["hidden"],
        logp=branch_logp,
        hidden=branch_hidden,
        z_cam=capture["z_cam"],
        z_enhanced=enhanced,
        fusion=details,
        functional_pre_dropout=branch_pre_dropout,
        capture=capture,
        rng_after_base=public_after_base,
    )
    return base_out


def joint_forward(state, module, *, training, intervention="full", blocked_rows=()):
    capture = {}
    layer = state["cam"].Outputlayer

    def replace_input(_module, args):
        z_cam = args[0]
        eta = CFG["architecture"]["eta_joint"] if training else 0.0
        enhanced, details = module(
            z_cam,
            grouping="medical",
            eta=eta,
            intervention=intervention,
            blocked_rows=blocked_rows,
        )
        capture.update(z_cam=z_cam, z_enhanced=enhanced, fusion=details)
        return (enhanced,) + args[1:]

    hook = layer.register_forward_pre_hook(replace_input)
    try:
        out = base_forward(state, training=training)
    finally:
        hook.remove()
    if not capture:
        raise RuntimeError("joint OutputLayer hook did not execute")
    out.update(capture)
    return out


def _base_history():
    return runtime().load_history()


def _new_train_ids(state):
    ids = torch.as_tensor(state["train"], device=state["y"].device, dtype=torch.long)
    if state["fold"] == 2:
        block = torch.tensor([376, 406], device=ids.device, dtype=torch.long)
        ids = ids[~torch.isin(ids, block)]
    return ids


def _gradient_copy(state):
    return {
        key: {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in state[key].named_parameters()}
        for key in ("model", "cam")
    }


def _module_grad_copy(module):
    return {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in module.named_parameters()}


def _tree_exact(left, right, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        a, b = left.detach().cpu(), right.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"exact comparison metadata differs at {where}")
        if not torch.equal(a, b):
            err = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.0
            raise RuntimeError(f"exact comparison failed at {where}; max_abs_error={err:.12g}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"exact comparison keys differ at {where}")
        for key in left:
            _tree_exact(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            raise RuntimeError(f"exact comparison lengths differ at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            _tree_exact(a, b, f"{where}[{index}]")
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"exact comparison failed at {where}")
    elif left != right:
        raise RuntimeError(f"exact comparison failed at {where}: {left!r} != {right!r}")


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _base_bundle(state):
    return copy.deepcopy(_base_history().state_bundle(state))


def _restore_base_bundle(state, saved):
    _base_history().load_bundle(state, copy.deepcopy(saved))


def state_bundle(state, module, optimizer):
    return {"base": _base_bundle(state), "sa_prcl": _cpu_state(module), "optimizer_sa_prcl": copy.deepcopy(optimizer.state_dict())}


def load_state_bundle(state, module, optimizer, bundle):
    _restore_base_bundle(state, bundle["base"])
    module.load_state_dict(bundle["sa_prcl"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(bundle["optimizer_sa_prcl"]))


def _base_hashes(state):
    modules = {}
    for key in ("model", "cam"):
        modules[key] = {name: value.detach().cpu() for name, value in state[key].state_dict().items()}
    optimizers = {key: state[key].state_dict() for key in ("opt_base", "opt_cam", "opt_added")}
    return sha_value(modules), sha_value(optimizers)


def _finite_grads(module):
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def _legal_raw_penalty(details, state):
    return residual_penalty(details, _new_train_ids(state))


def _base_stats(out):
    return {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")}


def _protected_step(state, module, optimizer, grouping, *, verify_grads=False):
    history = _base_history()
    legacy = legacy_runtime()
    history.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    blocked = (376, 406) if state["fold"] == 2 else ()
    out = protected_forward(state, module, grouping, training=True, blocked_rows=blocked)
    if blocked and torch.count_nonzero(out["fusion"]["delta"][[376, 406]]):
        raise RuntimeError("Fold2 protected branch produced a training residual for an overlap row")
    if not torch.isfinite(out["total"]):
        raise FloatingPointError("nonfinite historical DARG total")
    out["total"].backward()
    darg_grad_before = _gradient_copy(state)
    module_grad_before = _module_grad_copy(module)

    logp_branch, _, _ = out["logp"], out["hidden"], out["functional_pre_dropout"]
    cls = legacy.runtime()["utils"].ClsLoss(logp_branch, state["y"], _new_train_ids(state), state["weights"])
    raw = _legal_raw_penalty(out["fusion"], state)
    branch_loss = cls + CFG["loss"]["lambda_raw"] * raw
    if not torch.isfinite(branch_loss):
        raise FloatingPointError("nonfinite protected SA-PRCL branch loss")
    branch_loss.backward()
    darg_grad_after = _gradient_copy(state)
    _tree_exact(darg_grad_before, darg_grad_after, "protected.branch_must_not_change_DARG_gradients")
    if not _finite_grads(module):
        raise FloatingPointError("nonfinite SA-PRCL branch gradient")
    if verify_grads and not any(p.grad is not None and bool(p.grad.detach().abs().sum() > 0) for p in module.parameters()):
        raise RuntimeError("protected branch did not produce an SA-PRCL gradient")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    stats = {
        **_base_stats(out),
        "branch_cls": float(cls.detach().cpu()),
        "raw_penalty": float(raw.detach().cpu()),
        "branch_total": float(branch_loss.detach().cpu()),
        "dropout_logp_parity": bool(torch.equal(out["logp"].detach(), out["base_logp"].detach())) if not torch.count_nonzero(out["fusion"]["delta"]) else None,
        "dropout_hidden_parity": bool(torch.equal(out["hidden"].detach(), out["base_hidden"].detach())) if not torch.count_nonzero(out["fusion"]["delta"]) else None,
        "base_grad_sha_before_branch": sha_value(darg_grad_before),
        "base_grad_sha_after_branch": sha_value(darg_grad_after),
        "module_grad_sha_before_branch": sha_value(module_grad_before),
        "module_grad_sha_after_branch": sha_value(_module_grad_copy(module)),
        "fold2_overlap_training_residual_zero": bool(not blocked or torch.count_nonzero(out["fusion"]["delta"][[376, 406]]) == 0),
    }
    return out, stats


def _joint_step(state, module, optimizer, *, verify_grads=False):
    history = _base_history()
    history.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    blocked = (376, 406) if state["fold"] == 2 else ()
    out = joint_forward(state, module, training=True, blocked_rows=blocked)
    if blocked and torch.count_nonzero(out["fusion"]["delta"][[376, 406]]):
        raise RuntimeError("Fold2 joint branch produced a training residual for an overlap row")
    raw = _legal_raw_penalty(out["fusion"], state)
    total = out["total"] + CFG["loss"]["lambda_raw"] * raw
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite joint SA-PRCL loss")
    total.backward()
    if not _finite_grads(state["model"]) or not _finite_grads(state["cam"]) or not _finite_grads(module):
        raise FloatingPointError("nonfinite joint gradient")
    if verify_grads and not any(p.grad is not None and bool(p.grad.detach().abs().sum() > 0) for p in module.parameters()):
        raise RuntimeError("joint training did not produce an SA-PRCL gradient")
    base_stats = _base_stats(out)
    base_stats.update(branch_cls=float(out["cls"].detach().cpu()), raw_penalty=float(raw.detach().cpu()), branch_total=float(total.detach().cpu()),
                      fold2_overlap_training_residual_zero=bool(not blocked or torch.count_nonzero(out["fusion"]["delta"][[376, 406]]) == 0))
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    return out, base_stats


def _eval_forward(state, module, group, intervention="full"):
    settings = CFG["group_settings"][group]
    if settings["protected"]:
        out = protected_forward(state, module, settings["grouping"], training=False, intervention=intervention)
    else:
        out = joint_forward(state, module, training=False, intervention=intervention)
    return out


def _metric_payload(logp, state):
    metrics = legacy_runtime().classification_metrics(logp, state["test"], state["y"])
    return {key: float(metrics[key]) * 100.0 for key in METRICS}, metrics


def _mean_std(rows, metric):
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _task_paths(group, fold):
    name = f"{group}_f{fold:02d}"
    return TASKS / f"{name}.json", CHECKPOINTS / f"{name}.best.pt", CHECKPOINTS / f"{name}.latest.pt", LOGS / f"{name}.jsonl"


def _initial_identity(state, warmup, module):
    legacy = legacy_runtime()
    darg_sha, optimizer_sha = _base_hashes(state)
    return {
        "warmup_sha256": state["sa_identity"]["warmup_sha256"],
        "warmup_rng_sha256": warmup["rng_sha"],
        "public_rng_sha256": legacy.rng_sha(legacy.pack_rng()),
        "base_model_sha256": darg_sha,
        "base_optimizer_sha256": optimizer_sha,
        "relation_init_sha256": sha_value(_cpu_state(module)),
        "split_sha": state["sa_identity"].get("split_sha"),
        "pair_sha": state["sa_identity"].get("pair_sha"),
        "membership": state["sa_identity"]["membership"],
    }


def _checkpoint_payload(group, fold, source_hash, identity, best, trace, executed):
    return {
        "schema": "E81.6.2-SA-PRCL-v1",
        "group": group,
        "fold": fold,
        "seed": 0,
        "source_sha256": source_hash,
        "identity": identity,
        "best_epoch_1based": best["best_epoch_1based"],
        "epochs_executed": executed,
        "metrics": best["metrics"],
        "best": best,
        "rng_trace": trace,
    }


def _task_is_complete(group, fold, frozen_sha):
    task_file, best_file, latest_file, _ = _task_paths(group, fold)
    if not all(path.is_file() for path in (task_file, best_file, latest_file)):
        return False
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        return (
            task.get("status") == "COMPLETE"
            and task.get("source_sha256") == frozen_sha
            and task.get("best_checkpoint_sha256") == sha_file(best_file)
            and task.get("latest_checkpoint_sha256") == sha_file(latest_file)
        )
    except Exception:
        return False


def train_worker(group, fold):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task: {group}, fold={fold}")
    frozen = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "PASS" or source_sha() != frozen.get("source_sha256"):
        raise RuntimeError("formal training source differs from passing frozen preflight")
    state, warmup, history_identity = prepare(fold)
    settings = CFG["group_settings"][group]
    module = make_module(fold, state["x"].device)
    optimizer = sa_optimizer(module)
    identity = _initial_identity(state, warmup, module)
    task_file, best_file, latest_file, log_file = _task_paths(group, fold)
    source_hash = frozen["source_sha256"]
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and _task_is_complete(group, fold, source_hash):
            print(f"SKIP complete {group} fold={fold}", flush=True)
            return existing
        if existing.get("source_sha256") != source_hash or existing.get("identity") != identity:
            raise RuntimeError(f"task identity changed; refusing incompatible resume: {task_file}")

    legacy = legacy_runtime()
    if legacy.rng_sha(legacy.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not start at its restored E61 epoch-10 RNG")
    next_epoch, wait, best_acc, best, trace, log_rows = 10, 0, -1.0, None, [], []
    if latest_file.exists():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_hash), ("identity", identity)):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest checkpoint mismatch for {key}: {group} fold={fold}")
        load_state_bundle(state, module, optimizer, saved["current"])
        legacy.restore_rng(saved["rng"])
        next_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best, trace, log_rows = float(saved["best_acc"]), saved["best"], saved["trace"], saved["log_rows"]
        atomic_journal(log_file, log_rows)
    elif task_file.exists() and json.loads(task_file.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        raise RuntimeError(f"completed task is missing its restorable Latest checkpoint: {latest_file}")
    elif task_file.exists() and json.loads(task_file.read_text(encoding="utf-8")).get("status") not in ("RUNNING", "FAILED"):
        raise RuntimeError(f"unrecognized task state: {task_file}")

    task_state = {
        "status": "RUNNING",
        "group": group,
        "fold": fold,
        "seed": 0,
        "source_sha256": source_hash,
        "identity": identity,
        "history_identity": history_identity,
        "started_unix": time.time(),
        "next_epoch": next_epoch,
        "best_epoch_1based": None if best is None else best["best_epoch_1based"],
    }
    atomic_json(task_file, task_state)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    for epoch in range(next_epoch, CFG["base"]["max_epochs"]):
        epoch_number = epoch + 1
        rng_before = legacy.rng_sha(legacy.pack_rng())
        blocked = (376, 406) if fold == 2 else ()
        if settings["protected"]:
            out, losses = _protected_step(state, module, optimizer, settings["grouping"])
        else:
            out, losses = _joint_step(state, module, optimizer, verify_grads=False)
        rng_after_train = legacy.rng_sha(legacy.pack_rng())
        base_model_sha, base_optimizer_sha = _base_hashes(state)
        executed = epoch_number
        with torch.no_grad():
            evaluated = _eval_forward(state, module, group)
            metric_pct, metric_raw = _metric_payload(evaluated["logp"], state)
            base_metric_pct, _ = _metric_payload(evaluated["base_logp"], state) if settings["protected"] else _metric_payload(
                state["cam"].Outputlayer(evaluated["z_cam"], torch.empty(0, device=evaluated["z_cam"].device))[0], state
            )
        rng_after_eval = legacy.rng_sha(legacy.pack_rng())
        trace_row = {
            "epoch": epoch_number,
            "rng_before": rng_before,
            "rng_after_train": rng_after_train,
            "rng_after_eval": rng_after_eval,
            "base_model_sha256": base_model_sha,
            "base_optimizer_sha256": base_optimizer_sha,
        }
        trace.append(trace_row)
        if metric_pct["acc"] > best_acc:
            best_acc, best_epoch, wait = metric_pct["acc"], epoch_number, 0
            current_bundle = state_bundle(state, module, optimizer)
            full_prob = evaluated["logp"].detach().exp().cpu().numpy()
            full_pred = full_prob.argmax(axis=1)
            best = {
                "best_epoch_1based": best_epoch,
                "metrics": metric_pct,
                "base_only_metrics": base_metric_pct,
                "bundle": current_bundle,
                "probability_all": full_prob,
                "prediction_all": full_pred,
                "base_probability_all": evaluated["base_logp"].detach().exp().cpu().numpy() if settings["protected"] else None,
                "base_prediction_all": evaluated["base_logp"].argmax(dim=1).detach().cpu().numpy() if settings["protected"] else None,
                "losses": losses,
                "rng_after_train": rng_after_train,
                "rng_after_eval": rng_after_eval,
                "base_model_sha256": base_model_sha,
                "base_optimizer_sha256": base_optimizer_sha,
                "details": _compact_details(evaluated["fusion"], state),
            }
            atomic_torch(best_file, _checkpoint_payload(group, fold, source_hash, identity, best, trace, executed))
        else:
            wait += 1

        log_row = {
            "group": group,
            "fold": fold,
            "epoch": epoch_number,
            "best_epoch_1based": best["best_epoch_1based"],
            "best_acc": best_acc,
            "metrics": metric_pct,
            "base_only_metrics": base_metric_pct,
            "losses": losses,
            "diagnostics": _compact_details(evaluated["fusion"], state),
            "rng": trace_row,
            "wait": wait,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        }
        log_rows.append(log_row)
        atomic_journal(log_file, log_rows)
        atomic_torch(latest_file, {
            "schema": "E81.6.2-SA-PRCL-v1",
            "group": group,
            "fold": fold,
            "source_sha256": source_hash,
            "identity": identity,
            "current": state_bundle(state, module, optimizer),
            "rng": legacy.pack_rng(),
            "next_epoch": executed,
            "wait": wait,
            "best_acc": best_acc,
            "best": best,
            "trace": trace,
            "log_rows": log_rows,
        })
        task_state.update({"next_epoch": executed, "best_epoch_1based": best["best_epoch_1based"], "best_acc": best_acc})
        atomic_json(task_file, task_state)
        print(f"{group} fold={fold} epoch={epoch_number} best={best_acc:.4f}@{best['best_epoch_1based']} wait={wait}", flush=True)
        del out, evaluated
        if wait > CFG["base"]["early_stop_wait_gt"]:
            break

    if best is None:
        raise RuntimeError(f"no strict-ACC best checkpoint for {group} fold={fold}")
    latest = torch_load(latest_file)
    task_state.update({
        "status": "COMPLETE",
        "best_epoch_1based": best["best_epoch_1based"],
        "epochs_executed": executed,
        "metrics": best["metrics"],
        "best_checkpoint_sha256": sha_file(best_file),
        "latest_checkpoint_sha256": sha_file(latest_file),
        "scalar_log_sha256": sha_file(log_file),
        "rng_trace": trace,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    })
    atomic_json(task_file, task_state)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.4f}@{best['best_epoch_1based']}", flush=True)
    return task_state


def _compact_details(details, state):
    test = torch.as_tensor(state["test"], device=details["delta"].device, dtype=torch.long)
    selected = details["delta"][test]
    z = details["z_cam"][test].detach()
    raw = details["raw_delta"][test]
    ratio = details["delta_ratio"][test]
    beta_n, beta_a = details["beta_normal"][test], details["beta_abnormal"][test]
    attention = details["attention_prompt"][test]
    visual = details["attention_visual"][test]
    response = details["evidence"][test]
    return {
        "cam_norm_mean": float(z.norm(dim=-1).mean()),
        "delta_norm_mean": float(selected.norm(dim=-1).mean()),
        "raw_delta_norm_mean": float(raw.norm(dim=-1).mean()),
        "delta_ratio_mean": float(ratio.mean()),
        "delta_ratio_p95": float(torch.quantile(ratio, 0.95)),
        "delta_ratio_max": float(ratio.max()),
        "over_10_fraction": float((ratio > 0.100001).float().mean()),
        "prompt_attention_visual_mae": float((attention - visual[:, None]).abs().mean()),
        "prompt_attention_entropy": float((-(attention * attention.clamp_min(1e-12).log()).sum(-1) / math.log(16)).mean()),
        "null_attention_mean": float(attention[..., -1].mean()),
        "evidence_sd": float(response.std(unbiased=False)),
        "beta_normal_sd": float(beta_n.std(unbiased=False)),
        "beta_abnormal_sd": float(beta_a.std(unbiased=False)),
        "beta_normal_prompt_mean": beta_n.mean(dim=(0, 2)).detach().cpu().tolist(),
        "beta_abnormal_prompt_mean": beta_a.mean(dim=(0, 2)).detach().cpu().tolist(),
        "delta_messages_prompt_norm": details["delta_messages"][test].norm(dim=-1).mean(dim=(0, 2)).detach().cpu().tolist(),
        "relation_contrast_norm_mean": float(details["relation_contrast"][test].norm(dim=-1).mean()),
        "global_norm_mean": float(details["global"][test].norm(dim=-1).mean()),
        "residual_parallel_fraction": float((selected * z).sum(-1).abs().div(selected.norm(dim=-1).mul(z.norm(dim=-1)).clamp_min(1e-12)).mean()),
    }


def _darg_only_step(state):
    history = _base_history()
    history.zero_optimizers(state)
    out = base_forward(state, training=True)
    out["total"].backward()
    grads = _gradient_copy(state)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    return out, grads


def _protected_replay(fold):
    """Two consecutive steps against DARG-only, including every base optimizer and public RNG."""
    baseline, warm_base, _ = prepare(fold)
    legacy = legacy_runtime()
    if legacy.rng_sha(legacy.pack_rng()) != warm_base["rng_sha"]:
        raise RuntimeError(f"fold {fold}: DARG-only replay did not restore warmup RNG")
    baseline_trace = []
    for step in range(2):
        before = legacy.rng_sha(legacy.pack_rng())
        out, grads = _darg_only_step(baseline)
        after = legacy.rng_sha(legacy.pack_rng())
        base_model, base_optimizer = _base_hashes(baseline)
        baseline_trace.append({
            "rng_before": before,
            "rng_after": after,
            "logp": out["logp"].detach().cpu().clone(),
            "hidden": out["hidden"].detach().cpu().clone(),
            "fused": out["fused"].detach().cpu().clone(),
            "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
            "grads": grads,
            "base_model_sha": base_model,
            "base_optimizer_sha": base_optimizer,
            "bundle": _base_bundle(baseline),
        })
        del out
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    protected, warm_protected, _ = prepare(fold)
    if warm_base["rng_sha"] != warm_protected["rng_sha"]:
        raise RuntimeError(f"fold {fold}: independent warmup RNG identities differ")
    init_rng_before = legacy.rng_sha(legacy.pack_rng())
    module = make_module(fold, protected["x"].device)
    optimizer = sa_optimizer(module)
    initial = _initial_identity(protected, warm_protected, module)
    group_init_hashes = [initial["relation_init_sha256"]]
    for _group in GROUPS[1:]:
        twin = make_module(fold, protected["x"].device)
        group_init_hashes.append(sha_value(_cpu_state(twin)))
        _tree_exact(_cpu_state(module), _cpu_state(twin), f"fold{fold}.same_fold_group_initialization.{_group}")
        twin_optimizer = sa_optimizer(twin)
        if type(twin_optimizer) is not torch.optim.AdamW or twin_optimizer.param_groups[0]["lr"] != optimizer.param_groups[0]["lr"]:
            raise RuntimeError(f"fold {fold}: new optimizer settings differ across formal groups")
        del twin, twin_optimizer
    if len(set(group_init_hashes)) != 1:
        raise RuntimeError(f"fold {fold}: three groups do not have the same SA-PRCL initialization")
    if legacy.rng_sha(legacy.pack_rng()) != init_rng_before:
        raise RuntimeError(f"fold {fold}: group initialization changed public RNG")
    if initial["public_rng_sha256"] != warm_protected["rng_sha"]:
        raise RuntimeError(f"fold {fold}: private module initialization changed public RNG")
    protected_trace = []
    for step in range(2):
        expected = baseline_trace[step]
        if legacy.rng_sha(legacy.pack_rng()) != expected["rng_before"]:
            legacy.restore_rng(warm_protected["rng"] if step == 0 else baseline_trace[step - 1]["rng_after_state"])
        current_before = legacy.rng_sha(legacy.pack_rng())
        if current_before != expected["rng_before"]:
            raise RuntimeError(f"fold {fold} step {step + 1}: cannot align shared public RNG")
        out, stats = _protected_step(protected, module, optimizer, "medical", verify_grads=True)
        after = legacy.rng_sha(legacy.pack_rng())
        _tree_exact(expected["logp"], out["base_logp"], f"fold{fold}.step{step+1}.DARG_logp")
        _tree_exact(expected["hidden"], out["base_hidden"], f"fold{fold}.step{step+1}.DARG_hidden")
        _tree_exact(expected["fused"], out["fused"], f"fold{fold}.step{step+1}.DARG_fused")
        for key, value in expected["losses"].items():
            _tree_exact(torch.tensor(value), torch.tensor(float(out[key].detach().cpu())), f"fold{fold}.step{step+1}.DARG_{key}")
        _tree_exact(expected["grads"], _gradient_copy(protected), f"fold{fold}.step{step+1}.DARG_gradients")
        if after != expected["rng_after"]:
            raise RuntimeError(f"fold {fold} step {step + 1}: public RNG changed in protected branch")
        current_bundle = _base_bundle(protected)
        _tree_exact(expected["bundle"], current_bundle, f"fold{fold}.step{step+1}.DARG_parameters_and_optimizers")
        # Base optimizer state stored in the checkpoint must also be byte-identical.
        protected_trace.append({
            "step": step + 1,
            "rng_before": current_before,
            "rng_after": after,
            "base_model_sha256": _base_hashes(protected)[0],
            "base_optimizer_sha256": _base_hashes(protected)[1],
            "base_gradient_sha256": sha_value(_gradient_copy(protected)),
            "functional_dropout_logp_exact": torch.equal(out["logp"].detach(), out["base_logp"].detach()) if torch.count_nonzero(out["fusion"]["delta"]) == 0 else None,
            "functional_dropout_hidden_exact": torch.equal(out["hidden"].detach(), out["base_hidden"].detach()) if torch.count_nonzero(out["fusion"]["delta"]) == 0 else None,
            "DARG_gradient_unchanged_after_branch": stats["base_grad_sha_before_branch"] == stats["base_grad_sha_after_branch"],
            "SA_gradient_norm": math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in module.parameters() if p.grad is not None)),
        })
        baseline_trace[step]["rng_after_state"] = legacy.pack_rng()
        del out
    return {
        "fold": fold,
        "status": "PASS",
        "warmup_rng_sha256": warm_base["rng_sha"],
        "initial_public_rng_unchanged": True,
        "same_fold_three_group_parameters_exact": True,
        "steps": protected_trace,
        "base_model_parameter_and_three_optimizer_states_exact": True,
        "public_rng_exact": True,
    }


def _model_preflight():
    cache = prompt_cache()
    module = make_module(1, "cpu")
    twin = make_module(1, "cpu")
    _tree_exact(_cpu_state(module), _cpu_state(twin), "private_initialization_repeat")
    if sum(parameter.numel() for parameter in module.parameters()) != 8032:
        raise RuntimeError("unexpected trainable SA-PRCL parameter count")
    z = torch.linspace(-2.0, 2.0, 5 * 144, dtype=torch.float32).reshape(5, 144)
    _, first = module(z, grouping="medical")
    _, second = module(z, grouping="scrambled")
    expected = {
        "z_cam": (5, 144), "slots": (5, 4, 4, 9), "local": (5, 16, 9), "u": (5, 16, 32),
        "global": (5, 4, 9), "query": (5, 16, 32), "key_visual": (5, 16, 15, 32),
        "key_semantic": (5, 16, 15, 8), "value": (5, 16, 15, 32), "score_visual": (5, 16, 16),
        "score_semantic": (5, 8, 16, 15), "score_prompt": (5, 8, 16, 16),
        "attention_visual": (5, 16, 16), "attention_prompt": (5, 8, 16, 16),
        "messages": (5, 8, 16, 32), "delta_messages": (5, 8, 16, 32), "evidence": (5, 8, 16),
        "beta_normal": (5, 4, 16), "beta_abnormal": (5, 4, 16), "relation_contrast": (5, 16, 32),
        "d": (5, 4, 4, 9), "raw_delta": (5, 144), "delta": (5, 144),
    }
    for key, shape in expected.items():
        if tuple(first[key].shape) != shape:
            raise RuntimeError(f"preflight shape mismatch {key}: {tuple(first[key].shape)} != {shape}")
    if not torch.equal(first["raw_delta"], torch.zeros_like(first["raw_delta"])):
        raise RuntimeError("strict zero initialization did not produce raw_delta=0")
    if not torch.equal(first["z_enhanced"], z):
        raise RuntimeError("initial SA residual changed CAM144")
    _, first_off = module(z, grouping="medical", intervention="prompt_off")
    _, second_off = module(z, grouping="scrambled", intervention="prompt_off")
    for details in (first_off, second_off):
        if not torch.equal(details["attention_prompt"], details["attention_visual"][:, None].expand_as(details["attention_prompt"])):
            raise RuntimeError("Prompt-off did not recover pure visual attention")
        for key in ("delta_messages", "message_normal", "message_abnormal", "relation_contrast", "raw_delta", "delta"):
            if torch.count_nonzero(details[key]):
                raise RuntimeError(f"Prompt-off did not zero {key}")
    _, null = module(z, intervention="null_only")
    for key in ("message_visual", "messages", "delta_messages", "relation_contrast", "raw_delta", "delta"):
        if torch.count_nonzero(null[key]):
            raise RuntimeError(f"Null-only did not zero {key}")
    if not torch.equal(null["z_enhanced"], z):
        raise RuntimeError("Null-only changed CAM144")

    torch.manual_seed(816201)
    with torch.no_grad():
        module.output.weight.copy_(torch.linspace(-0.7, 0.8, 288).reshape(9, 32))
    _, medical = module(z, grouping="medical")
    _, scrambled = module(z, grouping="scrambled")
    if not torch.equal(medical["delta_messages"], scrambled["delta_messages"]):
        raise RuntimeError("G1/G2 grouping changed per-Prompt messages")
    if torch.equal(medical["relation_contrast"], scrambled["relation_contrast"]):
        raise RuntimeError("preflight input did not exercise grouping aggregation")
    for intervention in ("prompt_off", "null_only"):
        _, disabled = module(z, grouping="medical", intervention=intervention)
        for key in ("delta_messages", "message_normal", "message_abnormal", "relation_contrast", "raw_delta", "delta"):
            if torch.count_nonzero(disabled[key]):
                raise RuntimeError(f"trained-head {intervention} failed to zero {key}")
        if not torch.equal(disabled["z_enhanced"], z):
            raise RuntimeError(f"trained-head {intervention} changed CAM144")
    if not torch.all(medical["delta_ratio"] <= 0.100001):
        raise RuntimeError("bounded residual exceeded 10 percent")
    beta_spread = float(torch.maximum(medical["beta_normal"].std(unbiased=False), medical["beta_abnormal"].std(unbiased=False)).detach())
    if beta_spread <= 1e-7:
        raise RuntimeError("patient-conditioned Prompt beta is constant at the uniform value")
    neighbor_pairs = {(min(p, int(q)), max(p, int(q))) for p in range(16) for q in module.neighbors[p]}
    if len(neighbor_pairs) != 120 or module.neighbors.numel() != 240:
        raise RuntimeError("expected 120 unique unordered / 240 directed local relationships")
    original_pool = F.avg_pool1d(z.reshape(5 * 4, 1, 36), 4, 4).reshape(5, 4, 9)
    _tree_exact(original_pool, module.global_pool(z), "original_RES_AvgPool1d")
    d = medical["d"]
    expected_matching = (d * medical["global"].unsqueeze(2)).softmax(dim=-1)
    _tree_exact(expected_matching, medical["matching"], "original_RES_local_matching")

    sem_weight = module.semantic_key_proj.weight
    per_prompt_gradients = []
    for prompt_index in range(8):
        score = medical["score_prompt"][:, prompt_index, :, :15].sum()
        gradient = torch.autograd.grad(score, sem_weight, retain_graph=prompt_index < 7, allow_unused=False)[0]
        per_prompt_gradients.append(float(gradient.norm()))
    if any(value <= 0.0 for value in per_prompt_gradients):
        raise RuntimeError("at least one fixed Prompt lacks a relationship-score gradient")
    route_probe = z.detach().clone().requires_grad_(True)
    eta_zero = module(route_probe, eta=0.0)[1]["delta"].square().sum()
    grad_eta_zero = torch.autograd.grad(eta_zero, route_probe, retain_graph=False)[0]
    route_probe = z.detach().clone().requires_grad_(True)
    eta_point_one = module(route_probe, eta=0.1)[1]["delta"].square().sum()
    grad_eta_point_one = torch.autograd.grad(eta_point_one, route_probe, retain_graph=True)[0]
    eta_one = module(route_probe, eta=1.0)[1]["delta"].square().sum()
    grad_eta_one = torch.autograd.grad(eta_one, route_probe, retain_graph=False)[0]
    if torch.count_nonzero(grad_eta_zero) or not torch.allclose(grad_eta_point_one, 0.1 * grad_eta_one, atol=1e-8, rtol=2e-5):
        raise RuntimeError("patient-conditioned relation gradient routing does not match eta=0/0.1/1")
    return {
        "status": "PASS",
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "parameter_shapes": {name: list(value.shape) for name, value in module.named_parameters()},
        "expected_shapes": {key: list(shape) for key, shape in expected.items()},
        "zero_init_and_prompt_off_exact": True,
        "null_only_exact": True,
        "beta_max_sd": beta_spread,
        "beta_prompt_mean_medical": {"normal": medical["beta_normal"].mean(dim=(0, 2)).tolist(), "abnormal": medical["beta_abnormal"].mean(dim=(0, 2)).tolist()},
        "prompt_semantic_key_gradient_norms": per_prompt_gradients,
        "eta_zero_gradient_norm": float(grad_eta_zero.norm()),
        "eta_point_one_gradient_norm": float(grad_eta_point_one.norm()),
        "eta_one_gradient_norm": float(grad_eta_one.norm()),
        "res_max_ratio": float(medical["delta_ratio"].max()),
        "unique_pairs": len(neighbor_pairs),
        "directed_pairs": int(module.neighbors.numel()),
        "RES_pool_and_match_exact": True,
        "gram_max_error_float64": cache["svd"]["gram_max_abs_error_float64"],
        "gram_max_error_float32": cache["svd"]["gram_max_abs_error_float32"],
        "numerical_rank": cache["svd"]["numerical_rank"],
    }


def _joint_gradient_preflight():
    base_state, warmup, _ = prepare(1)
    legacy = legacy_runtime()
    initial_rng = legacy.rng_sha(legacy.pack_rng())
    _darg_only_step(base_state)
    darg_rng_after = legacy.rng_sha(legacy.pack_rng())
    del base_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state, warmup_joint, _ = prepare(1)
    if warmup["rng_sha"] != warmup_joint["rng_sha"] or legacy.rng_sha(legacy.pack_rng()) != initial_rng:
        raise RuntimeError("G3 joint preflight did not restore the same common RNG start")
    module = make_module(1, state["x"].device)
    optimizer = sa_optimizer(module)
    count = {"outputlayer": 0}
    handle = state["cam"].Outputlayer.register_forward_hook(lambda *_: count.__setitem__("outputlayer", count["outputlayer"] + 1))
    try:
        out, _ = _joint_step(state, module, optimizer, verify_grads=True)
    finally:
        handle.remove()
    joint_rng_after = legacy.rng_sha(legacy.pack_rng())
    if joint_rng_after != darg_rng_after:
        raise RuntimeError("G3 joint branch changed common DARG RNG consumption")
    if count["outputlayer"] != 1:
        raise RuntimeError("G3 joint hook must execute the original OutputLayer exactly once")
    base_grad_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in list(state["model"].parameters()) + list(state["cam"].parameters()) if p.grad is not None))
    sa_grad_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in module.parameters() if p.grad is not None))
    if base_grad_norm <= 0 or sa_grad_norm <= 0:
        raise RuntimeError("G3 joint gradient path is incomplete")
    return {"status": "PASS", "warmup_rng_sha256": warmup["rng_sha"], "rng_after_darg_sha256": darg_rng_after,
            "rng_after_joint_sha256": joint_rng_after, "base_grad_norm": base_grad_norm, "sa_grad_norm": sa_grad_norm,
            "joint_classifier_forward_count": count["outputlayer"], "joint_loss_uses_enhanced_CAM": True}


def _dropout_parity_preflight():
    state, warmup, _ = prepare(1)
    capture = {}
    hooks = _attach_capture(state, capture, capture_dropout=True)
    try:
        out = base_forward(state, training=True)
    finally:
        _remove_hooks(hooks)
    rng_after_base = legacy_runtime().pack_rng()
    logp, hidden, pre = frozen_outputlayer(state, capture["z_cam"], training=True, dropout_pre_rng=capture["dropout_pre_rng"])
    rng_after_branch = legacy_runtime().pack_rng()
    _tree_exact(out["hidden"], capture["dropout_output"], "historical_dropout_output_hook")
    _tree_exact(out["hidden"], hidden, "functional_frozen_classifier_hidden")
    _tree_exact(out["logp"], logp, "functional_frozen_classifier_logp")
    _tree_exact(capture["dropout_input"], pre, "functional_pre_dropout_features")
    _tree_exact(rng_after_base, rng_after_branch, "dropout_replay_public_rng")
    return {"status": "PASS", "functional_hidden_exact": True, "functional_logp_exact": True,
            "dropout_pre_features_exact": True, "dropout_replay_rng_isolated": True, "warmup_rng_sha256": warmup["rng_sha"]}


def preflight():
    make_prompt_cache()
    cache = prompt_cache()
    model_check = _model_preflight()
    dropout_check = _dropout_parity_preflight()
    protected_folds = []
    for fold in range(1, 11):
        print(f"Preflight DARG protection: fold {fold}/10", flush=True)
        protected_folds.append(_protected_replay(fold))
    joint_check = _joint_gradient_preflight()
    _dropout_recovery_check()
    sources = source_map()
    source_hash = sha_value(sources)
    historical_sources = {
        "SPromptGL_paper": "https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
        "SPromptGL_network.py_sha256": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec",
        "SPromptGL_layers.py_sha256": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
        "SPromptGL_main.py_sha256": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
        "E74_4_soft_positive_dcr.py_sha256": "3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718",
        "E74_4_run.py_sha256": "cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd",
        "E74_4_config_sha256": "a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a",
        "DUGSA_PLNNoPEN_vendor_sha256": "ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12",
        "KD_Brain_paper": "https://papers.miccai.org/miccai-2026/paper/0073_paper.pdf",
        "KD_Brain_model_py_sha256": "2e7f32693847e0794a9d0b2deb172dbcd285c686b74827eea31d6e30335a887d",
        "ItemizedCLIP_paper": "https://arxiv.org/abs/2512.11141",
        "ItemizedCLIP_model_py_sha256": "6318c8cad31236560ffda9713801bc29fe74b7bf3cf49b077223174e88dbb4f25",
        "E81_6_commit": CFG["readonly_references"]["e81_6_commit"],
        "E81_6_1_commit": CFG["readonly_references"]["e81_6_1_commit"],
    }
    review = {"status": "PASS", "sources": historical_sources, "source_map_sha256": source_hash, "source_map": sources,
              "prompt_cache_sha256": sha_file(HERE / "prompt_features.pt"), "cache_svd": cache["svd"]}
    atomic_json(RUNTIME / "source_review.json", review)
    payload = {
        "status": "PASS",
        "source_sha256": source_hash,
        "source_map": sources,
        "prompt_cache_sha256": sha_file(HERE / "prompt_features.pt"),
        "semantic_geometry": cache["svd"],
        "model": model_check,
        "dropout_classifier": dropout_check,
        "backbone_protection_folds": protected_folds,
        "joint_path": joint_check,
        "resume_replay": "PASS_EXACT_TWO_STEP_NEXT_EPOCH",
        "formal_tasks_expected": 30,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "peak_gpu_bytes_observed": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    atomic_json(RESULTS / "preflight.json", payload)
    atomic_json(HERE / "manifest.json", {
        "status": "PREFLIGHT_PASS",
        "source_sha256": source_hash,
        "source_map": sources,
        "preflight_sha256": sha_file(RESULTS / "preflight.json"),
        "formal_tasks_expected": 30,
        "completed_tasks": 0,
        "directory_fallback_reason": CFG["root_fallback_reason"],
    })
    return payload


def _dropout_recovery_check():
    """Serialize after one protected update, resume fresh, and compare the next update exactly."""
    fold = 1
    state, warmup, _ = prepare(fold)
    module = make_module(fold, state["x"].device)
    optimizer = sa_optimizer(module)
    _protected_step(state, module, optimizer, "medical", verify_grads=True)
    snapshot = {"bundle": state_bundle(state, module, optimizer), "rng": legacy_runtime().pack_rng()}
    target = RUNTIME / "preflight_recovery.tmp.pt"
    atomic_torch(target, snapshot)
    uninterrupted, _ = _protected_step(state, module, optimizer, "medical", verify_grads=True)
    expected = {
        "bundle": state_bundle(state, module, optimizer),
        "rng": legacy_runtime().pack_rng(),
        "logp": uninterrupted["logp"].detach().cpu().clone(),
        "raw_delta": uninterrupted["fusion"]["raw_delta"].detach().cpu().clone(),
        "attention": uninterrupted["fusion"]["attention_prompt"].detach().cpu().clone(),
    }
    del state, module, optimizer, uninterrupted
    resumed, _, _ = prepare(fold)
    resumed_module = make_module(fold, resumed["x"].device)
    resumed_optimizer = sa_optimizer(resumed_module)
    saved = torch_load(target)
    load_state_bundle(resumed, resumed_module, resumed_optimizer, saved["bundle"])
    legacy_runtime().restore_rng(saved["rng"])
    replay, _ = _protected_step(resumed, resumed_module, resumed_optimizer, "medical", verify_grads=True)
    _tree_exact(expected["logp"], replay["logp"], "resume.logp")
    _tree_exact(expected["raw_delta"], replay["fusion"]["raw_delta"], "resume.raw_delta")
    _tree_exact(expected["attention"], replay["fusion"]["attention_prompt"], "resume.attention")
    _tree_exact(expected["bundle"], state_bundle(resumed, resumed_module, resumed_optimizer), "resume.complete_state")
    _tree_exact(expected["rng"], legacy_runtime().pack_rng(), "resume.public_rng")
    target.unlink()


def _safe_worker_limit(measured_peak, override=None):
    free, total = torch.cuda.mem_get_info()
    peak = max(int(measured_peak), 1)
    available = max(int(free) - int(total * 0.15), 0)
    capacity = max(1, int((available * 0.70) // (peak * 2)))
    capacity = min(10, capacity)
    return min(capacity, max(1, int(override))) if override is not None else capacity


def _launch_worker(group, fold):
    log_path = LOGS / f"dispatch_{group}_f{fold:02d}.log"
    stream = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
        cwd=HERE,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return proc, stream, log_path


def run_all(workers=None):
    frozen = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "PASS" or frozen.get("source_sha256") != source_sha():
        raise RuntimeError("formal training requires the frozen passing preflight source")
    remaining = [(group, fold) for fold in range(1, 11) for group in GROUPS if not _task_is_complete(group, fold, frozen["source_sha256"])]
    if not remaining:
        return {"completed_tasks": 30, "expected_tasks": 30, "already_complete": True}

    # The first actual task is deliberately single-worker to measure real peak memory.
    first = remaining.pop(0)
    proc, stream, log_path = _launch_worker(*first)
    while proc.poll() is None:
        time.sleep(2.0)
    stream.close()
    if proc.returncode:
        raise RuntimeError(f"first measured worker failed: {first}; inspect {log_path}")
    completed = 1 + sum(_task_is_complete(g, f, frozen["source_sha256"]) for g in GROUPS for f in range(1, 11))
    task_path, _, _, _ = _task_paths(*first)
    measured_peak = int(json.loads(task_path.read_text(encoding="utf-8"))["peak_gpu_bytes"])
    worker_limit = _safe_worker_limit(measured_peak, workers)
    free, total = torch.cuda.mem_get_info()
    started = time.monotonic()
    active, failures = {}, []
    while remaining or active:
        while remaining and len(active) < worker_limit:
            free_now, _ = torch.cuda.mem_get_info()
            reserve = max(int(total * 0.15), 2 * measured_peak)
            if free_now < reserve + 2 * measured_peak:
                break
            pair = remaining.pop(0)
            worker, handle, path = _launch_worker(*pair)
            active[worker] = (*pair, handle, path)
        for worker, record in list(active.items()):
            code = worker.poll()
            if code is None:
                continue
            group, fold, handle, path = record
            handle.close()
            del active[worker]
            if code:
                failures.append({"group": group, "fold": fold, "exit_code": code, "log": str(path), "tail": path.read_text(errors="replace")[-4000:]})
            else:
                completed += 1
        atomic_json(RUNTIME / "scheduler.json", {
            "completed_tasks": completed,
            "expected_tasks": 30,
            "worker_limit": worker_limit,
            "measured_single_worker_peak_gpu_bytes": measured_peak,
            "free_gpu_bytes_at_schedule": int(free),
            "total_gpu_bytes": int(total),
            "active": [{"group": row[0], "fold": row[1]} for row in active.values()],
            "pending": len(remaining),
            "failures": failures,
        })
        if failures and not active:
            raise RuntimeError(f"formal worker failures: {failures}")
        if remaining or active:
            time.sleep(2.0)
    summary = {
        "completed_tasks": sum(_task_is_complete(g, f, frozen["source_sha256"]) for g in GROUPS for f in range(1, 11)),
        "expected_tasks": 30,
        "wall_seconds_after_first": time.monotonic() - started,
        "worker_limit": worker_limit,
        "measured_single_worker_peak_gpu_bytes": measured_peak,
        "failures": failures,
    }
    atomic_json(RUNTIME / "training_summary.json", summary)
    if summary["completed_tasks"] != 30:
        raise RuntimeError(f"formal training incomplete: {summary['completed_tasks']}/30")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--make-prompt-cache", action="store_true")
    args = parser.parse_args()
    if args.make_prompt_cache:
        make_prompt_cache()
        return
    if args.worker:
        train_worker(args.group, args.fold)
        return
    preflight_path = RESULTS / "preflight.json"
    if preflight_path.is_file():
        preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_record.get("status") != "PASS" or preflight_record.get("source_sha256") != source_sha():
            raise RuntimeError("existing preflight no longer matches the frozen source")
    elif args.train_only:
        raise RuntimeError("preflight missing")
    else:
        preflight()
    if args.preflight_only:
        print("PREFLIGHT PASS", flush=True)
        return
    summary = run_all(args.workers)
    audit_proc = subprocess.run([sys.executable, str(HERE / "audit.py")], cwd=HERE)
    if audit_proc.returncode:
        raise RuntimeError("independent checkpoint audit failed")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "COMPLETE_AUDITED", "completed_tasks": 30, "audit_status": "PASS", "training_summary": summary})
    atomic_json(HERE / "manifest.json", manifest)
    print("COMPLETE_AUDITED", flush=True)


if __name__ == "__main__":
    main()
