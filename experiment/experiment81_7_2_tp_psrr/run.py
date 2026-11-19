#!/usr/bin/env python3
"""Frozen training and preflight entrypoint for Experiment 81.7.2."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import copy
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RESULTS = EXP / "results"
RUNTIME = EXP / ".runtime"
TASKS = RUNTIME / "tasks"
CKPTS = RUNTIME / "checkpoints"
LOGS = RUNTIME / "logs"
PROMPT_CACHE = EXP / CFG["architecture"]["prompt_cache"]
REPO = Path("/data/adodas/SPromptGL-main")
HIST_E744 = REPO / "experiment/experiment74_4_soft_positive_dcr"
E74_GROUP = "E74.4-G2-Positive-DCR"
METRIC_KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
METRIC_LABELS = {
    "acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC",
    "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1",
}
_BASE = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def verify_gpu1_device():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must select only physical GPU1, got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    device_info = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,name,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().split(",")
    if len(device_info) != 5 or int(device_info[0].strip()) != 1:
        raise RuntimeError(f"physical GPU1 query returned unexpected identity: {device_info}")
    physical_uuid = device_info[1].strip().replace("GPU-", "").lower()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("only one CUDA device must be visible after selecting physical GPU1")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    logical_uuid = str(getattr(properties, "uuid", "")).replace("GPU-", "").lower()
    if not logical_uuid or physical_uuid != logical_uuid:
        raise RuntimeError(f"logical cuda:0 does not map to physical GPU1: {logical_uuid} != {physical_uuid}")
    return {
        "physical_index": 1,
        "logical_index": 0,
        "physical_uuid": physical_uuid,
        "logical_uuid": logical_uuid,
        "device_name": properties.name,
        "free_bytes": int(device_info[3].strip()) * 1024**2,
        "total_bytes": int(device_info[4].strip()) * 1024**2,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    with path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)):
            raise IOError(f"JSON readback failed: {path}")


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def base_runner():
    global _BASE
    if _BASE is None:
        adapter_path = EXP / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("experiment81_7_2_path_adapter", adapter_path)
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        _BASE = adapter.load_e744_runner()
        adapter.install_warmup_compatibility(_BASE)
        _BASE.load_history()
    return _BASE


def prompt_payload():
    import torch
    if sha_file(PROMPT_CACHE) != CFG["architecture"]["prompt_cache_sha256"]:
        raise RuntimeError("frozen Prompt cache SHA256 mismatch")
    cached = torch.load(PROMPT_CACHE, map_location="cpu", weights_only=False)
    if cached["prompts"] != CFG["architecture"]["prompts"]:
        raise RuntimeError("fixed eight-Prompt identity or order changed")
    if cached.get("encoder") != "OpenAI CLIP RN50, frozen, CPU, eval" or not cached.get("label_independent"):
        raise RuntimeError("Prompt cache encoder or label-independence identity failed")
    if cached.get("trainable_prompt_embedding") or cached.get("trainable_text_projection"):
        raise RuntimeError("Prompt cache marks text parameters trainable")
    return cached


def semantic_audit(cached):
    import torch
    raw = cached["raw_features"].double()
    normalized = torch.nn.functional.normalize(raw, dim=-1)
    cached_normalized = cached["normalized_features"].double()
    gram = normalized @ normalized.T
    u, singular, _ = torch.linalg.svd(normalized, full_matrices=False)
    for column in range(u.shape[1]):
        pivot = int(u[:, column].abs().argmax())
        if u[pivot, column] < 0:
            u[:, column].neg_()
    recomputed = u * singular.unsqueeze(0)
    basis = cached["semantic_basis_float64"].double()
    gram_cached = basis @ basis.T
    error = float((gram_cached - gram).abs().max())
    decomposition_error = float((recomputed @ recomputed.T - gram).abs().max())
    if error > 2e-8 or decomposition_error > 2e-8:
        raise RuntimeError(f"fixed semantic Gram verification failed: cache={error}, SVD={decomposition_error}")
    norms = normalized.norm(dim=-1)
    cache_vector_error = float((cached_normalized - normalized).abs().max())
    cache_gram_error = float(((cached_normalized @ cached_normalized.T) - gram).abs().max())
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-8, rtol=0):
        raise RuntimeError("normalized Prompt features do not have unit norm")
    singular_values = torch.linalg.svdvals(gram)
    effective_rank = float(torch.exp(-((singular_values / singular_values.sum()).clamp_min(1e-30)
                                       * (singular_values / singular_values.sum()).clamp_min(1e-30).log()).sum()))
    centered = basis - basis.mean(dim=0, keepdim=True)
    centered_norms = centered.norm(dim=-1)
    centered_normalized = torch.nn.functional.normalize(centered, dim=-1)
    repeated_centered = torch.nn.functional.normalize(basis - basis.mean(dim=0, keepdim=True), dim=-1)
    if not torch.isfinite(centered_normalized).all() or bool((centered_norms <= 1e-8).any()):
        raise RuntimeError("fixed centered semantic basis is numerically invalid")
    if not torch.equal(centered_normalized, repeated_centered):
        raise RuntimeError("fixed centered semantic basis is not deterministic")
    if not torch.allclose(centered_normalized.norm(dim=-1), torch.ones(8, dtype=torch.float64), atol=1e-12, rtol=0):
        raise RuntimeError("fixed centered semantic vectors are not unit length")
    center_singular = torch.linalg.svdvals(centered_normalized)
    return {
        "prompt_count": 8,
        "raw_feature_shape": list(cached["raw_features"].shape),
        "basis_shape": list(basis.shape),
        "max_gram_error": error,
        "max_recomputed_svd_gram_error": decomposition_error,
        "float32_cache_max_vector_error": cache_vector_error,
        "float32_cache_max_gram_error": cache_gram_error,
        "prompt_norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "numerical_rank": int(torch.linalg.matrix_rank(gram).item()),
        "gram_effective_rank": effective_rank,
        "centered_vector_norms_before_normalization": centered_norms.tolist(),
        "centered_unit_norm_error": float((centered_normalized.norm(dim=-1) - 1.0).abs().max()),
        "centered_numerical_rank": int(torch.linalg.matrix_rank(centered_normalized).item()),
        "centered_singular_values": center_singular.tolist(),
        "centered_gram_max_abs": float((centered_normalized @ centered_normalized.T).abs().max()),
        "centered_gram_preserves_original": False,
        "centered_deterministic": True,
        "label_independent": True,
        "trainable_text_parameters": False,
    }


def make_module(fold, device, semantic_basis=None, selection_mode="node"):
    import torch
    from tp_psrr import TPPSRR
    if semantic_basis is None:
        semantic_basis = prompt_payload()["semantic_basis_float64"]
    public_before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CFG["architecture"]["private_init_salt"] + int(fold))
        module = TPPSRR(
            semantic_basis,
            rho=CFG["architecture"]["residual_rho"],
            relation_scale=CFG["architecture"]["relation_scale"],
            temperature=CFG["architecture"]["prompt_temperature"],
            selection_mode=selection_mode,
        )
    public_after = torch.random.get_rng_state()
    if not torch.equal(public_before, public_after):
        raise RuntimeError("private PSRR initialization changed public CPU RNG")
    return module.to(device)


class OutputPreHook:
    def __init__(self, state, module, group):
        self.state = state
        self.module = module
        self.group = group
        self.training = False
        self.intervention = "full"
        self.blocked_rows = ()
        self.return_diagnostics = False
        self.capture = None
        self.calls = 0
        output_layer = getattr(state["cam"], "Outputlayer", None)
        if output_layer is None:
            raise RuntimeError("historical CAM has no Outputlayer pre-hook target")
        self.handle = output_layer.register_forward_pre_hook(self._replace)

    def _replace(self, _layer, args):
        if not args or args[0].ndim != 2 or args[0].shape[-1] != 144:
            raise RuntimeError("historical Outputlayer input is not CAM144")
        self.calls += 1
        z_cam = args[0]
        enhanced, details = self.module(
            z_cam,
            grouping=self.group,
            intervention=self.intervention,
            blocked_rows=self.blocked_rows if self.training else (),
            return_diagnostics=self.return_diagnostics,
        )
        self.capture = {"z_cam": z_cam, **details}
        return (enhanced, *args[1:])

    def configure(self, *, training, intervention="full", blocked_rows=(), diagnostics=False):
        self.training = bool(training)
        self.intervention = intervention
        self.blocked_rows = tuple(blocked_rows)
        self.return_diagnostics = bool(diagnostics)
        self.capture = None
        self.calls = 0

    def close(self):
        self.handle.remove()


def legal_train_rows(state, fold):
    import numpy as np
    rows = np.asarray(state["train_np"], dtype=np.int64)
    if fold == 2:
        blocked = np.asarray(CFG["data"]["historical_fold2_psrr_blocked_rows_zero_based"], dtype=np.int64)
        rows = rows[~np.isin(rows, blocked)]
    return rows


def rng_runtime(e744):
    return e744.load_history().load_e69().load_e61().load_base()


def optimizer_grad_norm(optimizer):
    total = 0.0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(total)


def module_grad_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(total)


def tensor_tree_equal(left, right, path="root"):
    import numpy as np
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right), path
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right), path
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, path + ".keys"
        for key in sorted(left, key=str):
            equal, where = tensor_tree_equal(left[key], right[key], path + "." + str(key))
            if not equal:
                return False, where
        return True, path
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return False, path + ".length"
        for index, (a, b) in enumerate(zip(left, right)):
            equal, where = tensor_tree_equal(a, b, path + f"[{index}]")
            if not equal:
                return False, where
        return True, path
    try:
        return bool(left == right), path
    except Exception:
        return False, path


def tensor_tree_max_difference(left, right, path="root"):
    import numpy as np
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return float("inf"), path
        if left.numel() == 0:
            return 0.0, path
        difference = float((left.detach().double().cpu() - right.detach().double().cpu()).abs().max())
        return difference, path
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return float("inf"), path
        if left.size == 0:
            return 0.0, path
        return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))), path
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf"), path + ".keys"
        values = [tensor_tree_max_difference(left[key], right[key], path + "." + str(key)) for key in sorted(left, key=str)]
        return max(values, key=lambda item: item[0])
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return float("inf"), path + ".length"
        values = [tensor_tree_max_difference(a, b, path + f"[{index}]") for index, (a, b) in enumerate(zip(left, right))]
        return max(values, key=lambda item: item[0]) if values else (0.0, path)
    try:
        return (0.0, path) if left == right else (float("inf"), path)
    except Exception:
        return float("inf"), path


def _base_bundle(e744, state):
    return e744.load_history().state_bundle(state)


def _base_gradients(state):
    return {
        module_name: {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }
        for module_name, module in (("model", state["model"]), ("cam", state["cam"]))
    }


def _shared_optimizer_groups(state):
    groups = {
        "gfr": state["opt_base"],
        "cam": state["opt_cam"],
        "darg": state["opt_added"],
    }
    by_name = {}
    all_parameters = []
    owners = {}
    for name, optimizer in groups.items():
        parameters = []
        for optimizer_group in optimizer.param_groups:
            for parameter in optimizer_group["params"]:
                if not parameter.requires_grad:
                    continue
                parameters.append(parameter)
                previous_owner = owners.get(id(parameter))
                if previous_owner is not None:
                    raise RuntimeError(f"historical optimizer groups overlap on a shared parameter: {previous_owner} and {name}")
                owners[id(parameter)] = name
                all_parameters.append(parameter)
        by_name[name] = parameters
    expected = [parameter for module in (state["model"], state["cam"])
                for parameter in module.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in all_parameters} != {id(parameter) for parameter in expected}:
        raise RuntimeError("GFR, CAM, and DARG optimizers do not cover exactly the trainable historical parameters")
    return by_name, all_parameters


def _gradient_vector_stats(direct, branch):
    import torch
    dot = torch.zeros((), dtype=torch.float64, device=direct[0].device if direct else branch[0].device)
    direct_sq = torch.zeros_like(dot)
    branch_sq = torch.zeros_like(dot)
    for direct_grad, branch_grad in zip(direct, branch):
        direct_value = direct_grad.detach().double()
        branch_value = branch_grad.detach().double()
        dot = dot + (direct_value * branch_value).sum()
        direct_sq = direct_sq + direct_value.square().sum()
        branch_sq = branch_sq + branch_value.square().sum()
    return dot, direct_sq.sqrt(), branch_sq.sqrt()


def _buffer_snapshot(state):
    """Capture mutable model/CAM buffers so the diagnostic reference pass is side-effect free."""
    return {
        module_name: {name: value.detach().clone() for name, value in module.named_buffers()}
        for module_name, module in (("model", state["model"]), ("cam", state["cam"]))
    }


def _restore_buffers(state, snapshot):
    for module_name, module in (("model", state["model"]), ("cam", state["cam"])):
        current = dict(module.named_buffers())
        for name, value in snapshot[module_name].items():
            if name not in current:
                raise RuntimeError(f"reference buffer disappeared: {module_name}.{name}")
            current[name].copy_(value)


def _reference_gradient_control(reference_gradients, enhanced_gradients, state):
    """Apply one vector projection and one norm cap per historical optimizer group."""
    import torch
    groups, shared_parameters = _shared_optimizer_groups(state)
    index_by_id = {id(parameter): index for index, parameter in enumerate(shared_parameters)}
    kappa = float(CFG["optimization"]["shared_gradient_control"]["post_projection_group_norm_ratio_kappa"])
    controlled = [None if gradient is None else gradient.detach().clone() for gradient in enhanced_gradients]
    records = {}
    max_reconstruction_error = 0.0
    max_scale = 0.0
    for base_grad, full_grad in zip(reference_gradients, enhanced_gradients):
        if base_grad is None and full_grad is None:
            continue
        if base_grad is None:
            base_grad = torch.zeros_like(full_grad)
        if full_grad is None:
            full_grad = torch.zeros_like(base_grad)
        delta = full_grad - base_grad
        reconstructed = base_grad + delta
        max_reconstruction_error = max(
            max_reconstruction_error,
            float((full_grad.detach().double() - reconstructed.detach().double()).abs().max().cpu()),
        )

    for group_name, parameters in groups.items():
        indices = [index_by_id[id(parameter)] for parameter in parameters]
        base_values = [reference_gradients[index] if reference_gradients[index] is not None else torch.zeros_like(parameter)
                       for parameter, index in zip(parameters, indices)]
        full_values = [enhanced_gradients[index] if enhanced_gradients[index] is not None else torch.zeros_like(parameter)
                       for parameter, index in zip(parameters, indices)]
        delta_values = [full - base for full, base in zip(full_values, base_values)]
        dot, base_norm, delta_norm = _gradient_vector_stats(base_values, delta_values)
        conflict = bool(dot < 0)
        if float(base_norm) <= 1e-30:
            projected = [torch.zeros_like(delta) for delta in delta_values]
            zero_reference = True
        elif conflict:
            coefficient = dot / base_norm.square()
            projected = [delta - coefficient.to(delta.dtype) * base
                         for base, delta in zip(base_values, delta_values)]
            zero_reference = False
        else:
            projected = delta_values
            zero_reference = False
        projected_norm = _gradient_vector_stats(base_values, projected)[2]
        cap = kappa * float(base_norm)
        scale = min(1.0, cap / float(projected_norm)) if float(projected_norm) > 0.0 else 1.0
        max_scale = max(max_scale, scale)
        for index, base, projected_value in zip(indices, base_values, projected):
            final = base + scale * projected_value
            controlled[index] = final.detach()
        records[group_name] = {
            "reference_norm": float(base_norm.detach().cpu()),
            "enhanced_norm": math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in full_values)),
            "delta_norm": float(delta_norm.detach().cpu()),
            "dot_reference_delta": float(dot.detach().cpu()),
            "cosine_reference_delta": float((dot / (base_norm * delta_norm).clamp_min(1e-30)).detach().cpu()),
            "negative_cosine": conflict,
            "projection_applied": conflict,
            "zero_reference_norm": zero_reference,
            "projected_delta_norm": float(projected_norm.detach().cpu()),
            "group_scale": scale,
            "controlled_delta_norm": float(projected_norm.detach().cpu() * scale),
            "controlled_delta_to_reference_ratio": float(projected_norm.detach().cpu() * scale / max(float(base_norm), 1e-30)),
        }
        if records[group_name]["controlled_delta_to_reference_ratio"] > kappa + 1e-7:
            raise RuntimeError(f"{group_name} delta gradient exceeded the group-level kappa bound")

    for parameter, gradient in zip(shared_parameters, controlled):
        parameter.grad = None if gradient is None or not bool(torch.count_nonzero(gradient)) else gradient
    return {
        "max_reconstruction_error": max_reconstruction_error,
        "reconstruction_tolerance": 1e-12,
        "groups": records,
        "reference_gradient_norm": math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in reference_gradients if value is not None)),
        "enhanced_gradient_norm": math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in enhanced_gradients if value is not None)),
        "delta_gradient_norm": math.sqrt(sum(float((f - (b if b is not None else torch.zeros_like(f))).detach().double().square().sum().cpu())
                                             for b, f in zip(reference_gradients, enhanced_gradients) if f is not None)),
        "max_group_scale": max_scale,
        "kappa": kappa,
    }


def training_step(e744, state, module, optimizer_psrr, hook, group, *, epoch, fold, gradient_path_probe=False):
    import torch
    base = e744.load_history()
    runtime = rng_runtime(e744)
    base.zero_optimizers(state)
    optimizer_psrr.zero_grad(set_to_none=True)
    blocked_rows = CFG["data"]["historical_fold2_psrr_blocked_rows_zero_based"] if fold == 2 else ()
    shared_groups, shared_parameters = _shared_optimizer_groups(state)
    module_parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]

    # The reference pass is a diagnostic computation. It uses the exact current
    # state and is rolled back before the public enhanced forward consumes RNG.
    rng_before_state = runtime.pack_rng()
    rng_before = runtime.rng_sha(rng_before_state)
    buffer_state = _buffer_snapshot(state)
    hook.configure(training=True, intervention="relation_off", blocked_rows=blocked_rows)
    reference = e744.forward(state, E74_GROUP, training=True)
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("historical classifier must execute exactly once in the DARG reference training pass")
    reference_gradients = torch.autograd.grad(
        reference["total"], shared_parameters, retain_graph=False, allow_unused=True
    )
    reference_loss = reference["total"].detach()
    _restore_buffers(state, buffer_state)
    runtime.restore_rng(rng_before_state)

    hook.configure(training=True, intervention="full", blocked_rows=blocked_rows)
    out = e744.forward(state, E74_GROUP, training=True)
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("historical classifier must execute exactly once in enhanced training")
    if fold == 2:
        if torch.count_nonzero(hook.capture["delta"][blocked_rows]):
            raise RuntimeError("Fold2 overlap rows received a nonzero PSRR training residual")
        if set(blocked_rows).intersection(legal_train_rows(state, fold).tolist()):
            raise RuntimeError("Fold2 overlap rows entered the PSRR residual regularizer")
    if not torch.isfinite(out["total"]):
        raise FloatingPointError("non-finite historical DARG total")
    from tp_psrr import residual_penalty
    raw = residual_penalty(hook.capture, legal_train_rows(state, fold))
    total = out["total"] + CFG["architecture"]["raw_residual_lambda"] * raw
    branch_vjp = None
    full_gradients = torch.autograd.grad(
        total, shared_parameters + module_parameters, retain_graph=True, allow_unused=True
    )
    enhanced_shared = full_gradients[:len(shared_parameters)]
    module_gradients = full_gradients[len(shared_parameters):]
    gradient_control = _reference_gradient_control(reference_gradients, enhanced_shared, state)
    for parameter, gradient in zip(module_parameters, module_gradients):
        parameter.grad = None if gradient is None else gradient.detach()
    if gradient_path_probe:
        capture = hook.capture
        class_to_delta = torch.autograd.grad(out["cls"], capture["delta"], retain_graph=True, allow_unused=False)[0]
        delta_to_cam = torch.autograd.grad(
            capture["delta"], capture["z_cam"], grad_outputs=class_to_delta,
            retain_graph=True, allow_unused=False,
        )[0]
        cam_params = tuple(parameter for parameter in state["cam"].parameters() if parameter.requires_grad)
        darg_params = tuple(parameter for parameter in state["model"].parameters() if parameter.requires_grad)
        cam_vjp = torch.autograd.grad(
            capture["z_cam"], cam_params, grad_outputs=delta_to_cam,
            retain_graph=True, allow_unused=True,
        )
        darg_vjp = torch.autograd.grad(
            capture["z_cam"], darg_params, grad_outputs=delta_to_cam,
            retain_graph=True, allow_unused=True,
        )
        branch_vjp = {
            "classification_to_delta_norm": float(class_to_delta.detach().norm().cpu()),
            "delta_to_cam_input_norm": float(delta_to_cam.detach().norm().cpu()),
            "branch_to_cam_parameter_norm": math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in cam_vjp if g is not None)),
            "branch_to_darg_parameter_norm": math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in darg_vjp if g is not None)),
        }

    if not all(torch.isfinite(parameter.grad).all() for module_check in (state["model"], state["cam"], module)
               for parameter in module_check.parameters() if parameter.grad is not None):
        raise FloatingPointError("non-finite training gradient")
    grad_norms = {
        "gfr_optimizer": optimizer_grad_norm(state["opt_base"]),
        "cam_optimizer": optimizer_grad_norm(state["opt_cam"]),
        "darg_optimizer": optimizer_grad_norm(state["opt_added"]),
        "psrr": module_grad_norm(module),
    }
    if any(not math.isfinite(value) for value in grad_norms.values()):
        raise FloatingPointError("non-finite gradient norm")

    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer_psrr.step()
    rng_after_train = runtime.rng_sha(runtime.pack_rng())
    with torch.no_grad():
        hook.configure(training=False)
        evaluated = e744.forward(state, E74_GROUP, training=False)
        metrics = runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
    if hook.calls != 1:
        raise RuntimeError("historical classifier must execute exactly once in evaluation")
    rng_after_eval = runtime.rng_sha(runtime.pack_rng())
    return {
        "epoch": int(epoch),
        "reference_loss": float(reference_loss.cpu()),
        "historical_loss": float(out["total"].detach().cpu()),
        "raw_penalty": float(raw.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "metrics": {key: float(metrics[key]) for key in METRIC_KEYS},
        "prediction": metrics["pred"],
        "probability": metrics["prob"],
        "gradient_norms": grad_norms,
        "branch_vjp": branch_vjp,
        "gradient_control": gradient_control,
        "rng": {"before": rng_before, "after_train": rng_after_train, "after_eval": rng_after_eval},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }


def source_inventory():
    e744 = base_runner()
    dependency_files = e744.source_manifest()
    extra_paths = [
        EXP / "config.json", EXP / "design.md", EXP / "tp_psrr.py", EXP / "run.py",
        EXP / "audit.py", EXP / "path_adapter.py", PROMPT_CACHE,
        REPO / "main.py", REPO / "network.py", REPO / "model.py", REPO / "layers.py", REPO / "utils.py",
        Path("/model/spromptgl/experiment/experiment81_7_psrr/config.json"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/manifest.json"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/psrr.py"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/run.py"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/audit.py"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/results/fold_metrics.csv"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/results/mechanism.csv"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/results/audit_status.json"),
        Path("/model/spromptgl/experiment/experiment81_7_psrr/results/preflight.json"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/design.md"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/psrr_lcr.py"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/run.py"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/audit.py"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/config.json"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/fold_metrics.csv"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/mechanism.csv"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/paired_comparisons.csv"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/audit_status.json"),
        Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/preflight.json"),
        Path("/model/spromptgl/experiment/experiment81_6_prcl_res/design.md"),
        Path("/model/spromptgl/experiment/experiment81_6_prcl_res/prcl_res.py"),
        Path("/model/spromptgl/experiment/experiment81_6_prcl_res/config.json"),
        Path("/model/spromptgl/experiment/experiment81_6_prcl_res/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res/design.md"),
        Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res/cprl_res.py"),
        Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res/config.json"),
        Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/design.md"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/sa_prcl.py"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/run.py"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/audit.py"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/config.json"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/fold_metrics.csv"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/paired_comparisons.csv"),
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/audit_status.json"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/design.md"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/run.py"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/audit.py"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/config.json"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/report.md"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/fold_metrics.csv"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/paired_comparisons.csv"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/mechanism.csv"),
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/audit_status.json"),
    ]
    for path in extra_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required locked source is missing: {path}")
        dependency_files[str(path.resolve())] = sha_file(path)
    for fold in range(1, 11):
        compat_checkpoint = RUNTIME / "compat_checkpoints" / f"fold{fold:02d}.pt"
        if not compat_checkpoint.is_file():
            raise FileNotFoundError(f"NumPy-compatible local warmup copy missing: {compat_checkpoint}")
        dependency_files[str(compat_checkpoint.resolve())] = sha_file(compat_checkpoint)
    for fold in range(1, 11):
        checkpoint = HIST_E744 / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"read-only DARG best checkpoint missing: {checkpoint}")
        dependency_files[str(checkpoint.resolve())] = sha_file(checkpoint)
    return dict(sorted(dependency_files.items()))


def source_digest(inventory=None):
    inventory = source_inventory() if inventory is None else inventory
    return sha_object(inventory)


def module_state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _entmax_preflight():
    import torch
    from tp_psrr import entmax15
    scores = torch.tensor([[0.8, 0.2, -0.1, -0.7], [-0.9, -0.2, 0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    if not torch.autograd.gradcheck(entmax15, (scores,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Entmax-1.5 gradcheck failed")
    probabilities = entmax15(scores)
    if not torch.all(probabilities >= 0) or not torch.allclose(probabilities.sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-12, rtol=0):
        raise RuntimeError("Entmax-1.5 simplex constraints failed")
    if not torch.any(probabilities == 0):
        raise RuntimeError("Entmax-1.5 did not produce exact sparse zeros")
    direction = torch.tensor([[0.3, -0.2, 0.6, -0.1], [-0.4, 0.7, 0.2, -0.5]], dtype=torch.float64)
    objective = (entmax15(scores) * direction).sum()
    analytical = torch.autograd.grad(objective, scores)[0]
    epsilon = 1e-6
    numerical = ((entmax15(scores.detach() + epsilon * direction) - entmax15(scores.detach() - epsilon * direction)) * direction).sum() / (2 * epsilon)
    finite_difference_error = float((analytical * direction).sum() - numerical)
    if abs(finite_difference_error) > 1e-5:
        raise RuntimeError(f"Entmax finite-difference check failed: {finite_difference_error}")
    return {"status": "PASS", "gradcheck": True, "minimum_probability": float(probabilities.min().detach()),
            "max_row_sum_error": float((probabilities.sum(-1) - 1).abs().max().detach()),
            "sparse_zero_count": int((probabilities == 0).sum()), "finite_difference_error": finite_difference_error}


def _module_preflight(cached, device):
    import torch
    basis = cached["semantic_basis_float64"]
    first = make_module(1, device, basis, "node")
    second = make_module(1, device, basis, "node")
    if module_state_digest(first) != module_state_digest(second):
        raise RuntimeError("same-fold private module initialization is not repeatable")
    edge = make_module(1, device, basis, "edge")
    if module_state_digest(first) != module_state_digest(edge):
        raise RuntimeError("G1 and G2 module initialization differs for the same fold")
    params = sum(parameter.numel() for parameter in first.parameters())
    if params != CFG["architecture"]["trainable_parameter_count"]:
        raise RuntimeError(f"unexpected PSRR-LCR trainable parameter count {params}")
    if any(parameter.requires_grad for parameter in first.parameters() if parameter is first.semantic_basis):
        raise RuntimeError("fixed semantic basis is trainable")
    expected_center = torch.nn.functional.normalize(
        first.semantic_basis - first.semantic_basis.mean(dim=0, keepdim=True), dim=-1
    )
    if not torch.allclose(first.semantic_center, expected_center, atol=1e-6, rtol=0):
        raise RuntimeError("fixed centered semantic basis differs from the frozen source basis")
    if first.semantic_basis.requires_grad or first.semantic_center.requires_grad:
        raise RuntimeError("fixed semantic source or centered basis requires gradients")
    neighbor_rows = first.neighbors.detach().cpu()
    if tuple(neighbor_rows.shape) != (16, 15) or any(
        len(set(row.tolist())) != 15 or receiver in row.tolist() for receiver, row in enumerate(neighbor_rows)
    ):
        raise RuntimeError("each local receiver must have exactly 15 distinct non-self neighbors")

    z = torch.linspace(-1.25, 1.25, 5 * 144, dtype=torch.float32, device=device).reshape(5, 144).requires_grad_()
    enhanced_zero, details_zero = first(z, return_diagnostics=True)
    if not torch.equal(enhanced_zero, z) or torch.count_nonzero(details_zero["delta"]):
        raise RuntimeError("zero-initialized residual changed CAM144")
    if bool((details_zero["residual_ratio"] > 0.1 + 1e-7).any()):
        raise RuntimeError("zero-initialized residual exceeded rho")
    blocked, blocked_details = first(z, blocked_rows=(1, 3), return_diagnostics=True)
    if torch.count_nonzero(blocked_details["delta"][[1, 3]]):
        raise RuntimeError("Fold2 PSRR row blocking failed")

    with torch.no_grad():
        output_weight = torch.linspace(-0.02, 0.02, first.output.weight.numel(), device=device).reshape_as(first.output.weight)
        first.output.weight.copy_(output_weight)
        edge.output.weight.copy_(output_weight)
    full, full_details = first(z, grouping="medical", return_diagnostics=True)
    edge_full, edge_details = edge(z, grouping="medical", return_diagnostics=True)
    prompt_off, prompt_details = first(z, intervention="prompt_off", return_diagnostics=True)
    relation_off, _ = first(z, intervention="relation_off", return_diagnostics=True)
    uniform, uniform_details = first(z, intervention="uniform_beta", return_diagnostics=True)
    global_off, global_off_details = first(z, intervention="global_interaction_off", return_diagnostics=True)
    if torch.count_nonzero(prompt_details["delta"]) or torch.count_nonzero(relation_off - z):
        raise RuntimeError("Prompt-off or relation-off identity failed")
    for name, beta in (("TD", full_details["beta_td"]), ("ASD", full_details["beta_asd"])):
        expected_shape = (5, 16, 4)
        if tuple(beta.shape) != expected_shape or not torch.allclose(beta.sum(-1), torch.ones((5, 16), device=device), atol=1e-6, rtol=0):
            raise RuntimeError(f"{name} node Prompt softmax is not over the four-prompt axis for every local receiver")
        if float(beta.std(dim=1).mean().detach()) <= 0 or float(beta.std(dim=0).mean().detach()) <= 0:
            raise RuntimeError(f"{name} Prompt weights do not vary across local slots and patients")
    if tuple(full_details["prompt_evidence"].shape) != (5, 16, 8):
        raise RuntimeError("patient-local Prompt evidence has the wrong shape")
    if tuple(edge_details["beta_td"].shape) != (5, 16, 15, 4) or tuple(edge_details["beta_asd"].shape) != (5, 16, 15, 4):
        raise RuntimeError("edge-conditioned Prompt weights have the wrong shape")
    if not torch.allclose(edge_details["beta_td"].sum(-1), torch.ones((5, 16, 15), device=device), atol=1e-6, rtol=0):
        raise RuntimeError("edge TD Prompt softmax is not normalized over four Prompts")
    if not torch.allclose(edge_details["beta_asd"].sum(-1), torch.ones((5, 16, 15), device=device), atol=1e-6, rtol=0):
        raise RuntimeError("edge ASD Prompt softmax is not normalized over four Prompts")
    if not torch.allclose(uniform_details["beta_td"], torch.full_like(uniform_details["beta_td"], 0.25)) or not torch.allclose(
        uniform_details["beta_asd"], torch.full_like(uniform_details["beta_asd"], 0.25)
    ):
        raise RuntimeError("Uniform-beta intervention did not assign equal four-Prompt weights")
    if not torch.allclose(global_off_details["matching"], torch.full_like(global_off_details["matching"], 1.0 / 9.0)):
        raise RuntimeError("global-interaction-off did not disable only the new RES interaction weights")
    if torch.equal(full_details["scores_asd"], edge_details["scores_asd"]):
        raise RuntimeError("node- and edge-conditioned Prompt selection did not change disease relation scores")
    if torch.equal(full_details["message_contrast"], edge_details["message_contrast"]):
        raise RuntimeError("node- and edge-conditioned Prompt selection did not change relation messages")
    if torch.equal(full_details["attention_asd"], full_details["attention_visual"]):
        raise RuntimeError("Prompt-conditioned semantic score did not change ASD relation probabilities")
    if torch.equal(full_details["attention_td"], full_details["attention_visual"]):
        raise RuntimeError("Prompt-conditioned semantic score did not change TD relation probabilities")
    if not torch.equal(prompt_details["attention_asd"], prompt_details["attention_visual"]) or not torch.equal(
        prompt_details["attention_td"], prompt_details["attention_visual"]
    ):
        raise RuntimeError("Prompt-off intervention retained semantic relation scores")
    if torch.count_nonzero(full_details["relation_support_difference"]) == 0:
        raise RuntimeError("ASD and TD did not produce distinct sparse relation supports")
    if not torch.allclose(full_details["attention_asd"].sum(-1), torch.ones((5, 16), device=device), atol=3e-6, rtol=0):
        raise RuntimeError("ASD Entmax probabilities do not sum to one across 15 neighbors and Null")
    if not torch.allclose(full_details["attention_td"].sum(-1), torch.ones((5, 16), device=device), atol=3e-6, rtol=0):
        raise RuntimeError("TD Entmax probabilities do not sum to one across 15 neighbors and Null")
    if not bool(((full_details["attention_asd"] == 0).any() and (full_details["attention_td"] == 0).any())):
        raise RuntimeError("disease-conditioned Entmax did not retain exact sparse zeros")
    if torch.count_nonzero(full_details["raw_delta"]) == 0:
        raise RuntimeError("PSRR-LCR relation message did not affect the final feature residual")
    if bool((full_details["residual_ratio"] > 0.1 + 1e-6).any()):
        raise RuntimeError("PSRR-LCR residual exceeded the fixed ten-percent bound")

    objective = (full_details["delta"] * torch.linspace(0.2, 1.1, 144, device=device)).sum()
    sem_grad = torch.autograd.grad(objective, (first.semantic_reduce.weight, first.semantic_expand.weight),
                                   retain_graph=True, allow_unused=False)
    z_grad = torch.autograd.grad(objective, z, allow_unused=False)[0]
    if any(float(grad.norm()) == 0.0 for grad in sem_grad) or float(z_grad.norm()) == 0.0:
        raise RuntimeError("final residual is disconnected from semantic relation mapping or CAM input")

    with torch.no_grad():
        visual_pool = torch.nn.functional.avg_pool1d(z.reshape(5 * 4, 1, 36), 4, 4).reshape(5, 4, 9)
        local = full_details["local_delta"]
        matching = torch.softmax(local * visual_pool.unsqueeze(2), dim=-1)
        expected = (matching * local).reshape(5, 144)
        if not torch.equal(expected, full_details["raw_delta"]):
            raise RuntimeError("RES AvgPool1d or coordinatewise matching differs from frozen formula")
    return {
        "status": "PASS",
        "trainable_parameters": params,
        "same_fold_initialization_identical": True,
        "G1_G2_initialization_identical": True,
        "initial_residual_exact_zero": True,
        "pairwise_shapes": {
            "local_tokens": list(full_details["local_tokens"].shape),
            "prompt_scores": list(full_details["prompt_scores"].shape),
            "attention_visual": list(full_details["attention_visual"].shape),
            "attention_asd": list(full_details["attention_asd"].shape),
            "beta_node": list(full_details["beta_td"].shape),
            "beta_edge": list(edge_details["beta_td"].shape),
            "relation_contrast": list(full_details["message_contrast"].shape),
            "raw_delta": list(full_details["raw_delta"].shape),
        },
        "prompt_changes_relation_probabilities": True,
        "disease_conditioned_supports_differ": True,
        "prompt_off_relation_off_zero": True,
        "node_edge_selection_changes_relation_scores_and_messages": True,
        "node_local_prompt_weights_vary_across_slots_and_patients": True,
        "edge_prompt_weights_vary_across_edges_and_patients": True,
        "uniform_beta_exact": True,
        "blocked_rows_zero": [1, 3],
        "max_residual_ratio": float(full_details["residual_ratio"].max().detach()),
        "semantic_mapping_gradient_norms": [float(grad.norm()) for grad in sem_grad],
        "cam_input_gradient_norm": float(z_grad.norm()),
        "RES_formula_exact": True,
    }


def _capture_epoch_state(e744, state, module, optimizer_psrr, identity, fold, group, grouping, init_sha, *, epoch, best_acc, best_epoch, wait, trace, gradient_trace):
    base = e744.load_history()
    runtime = rng_runtime(e744)
    return {
        "schema": "E81.7.2-TP-PSRR-v1",
        "group": group,
        "grouping": grouping,
        "selection_mode": module.selection_mode,
        "initial_psrr_sha256": init_sha,
        "fold": int(fold),
        "seed": 0,
        "source_sha256": source_digest(),
        "config_sha256": sha_file(EXP / "config.json"),
        "warmup_sha256": identity["warmup_sha256"],
        "base_identity": identity,
        "current": base.state_bundle(state),
        "psrr_state": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer_psrr": copy.deepcopy(optimizer_psrr.state_dict()),
        "rng": runtime.pack_rng(),
        "epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "wait": int(wait),
        "trace": trace,
        "gradient_trace": gradient_trace,
    }


def _task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def _best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def _latest_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def _log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def train_fold(group, fold):
    import numpy as np
    import torch
    device_identity = verify_gpu1_device()
    preflight_file = RESULTS / "preflight.json"
    if not preflight_file.is_file():
        raise RuntimeError("formal training requires a completed preflight")
    preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    current_source = source_digest()
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != current_source:
        raise RuntimeError("formal training source differs from passing frozen preflight")
    if group not in ("G1", "G2") or fold < 1 or fold > 10:
        raise ValueError("invalid group or fold")
    if group == "G2":
        audit_status_path = RESULTS / "audit_status.json"
        if not audit_status_path.is_file() or json.loads(audit_status_path.read_text(encoding="utf-8")).get("g1_status") != "COMPLETE_AUDITED":
            raise RuntimeError("G2 requires all ten G1 folds to pass independent audit")

    task_file, best_file, latest_file = _task_path(group, fold), _best_path(group, fold), _latest_path(group, fold)
    if task_file.exists():
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task.get("status") == "COMPLETE" and task.get("source_sha256") == current_source and best_file.is_file():
            print(f"SKIP complete {group} fold={fold}", flush=True)
            return task
        if task.get("status") not in ("RUNNING", "INTERRUPTED", "FAILED"):
            raise RuntimeError(f"refusing to overwrite incompatible task record: {task_file}")

    e744 = base_runner()
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    state["group"] = group
    state["psrr_group"] = group
    grouping = "medical"
    selection_mode = CFG["groups"][group]["selection_mode"]
    basis = prompt_payload()["semantic_basis_float64"]
    public_rng_before = rng_runtime(e744).rng_sha(rng_runtime(e744).pack_rng())
    module = make_module(fold, state["cam"].parameters().__next__().device, basis, selection_mode)
    public_rng_after = rng_runtime(e744).rng_sha(rng_runtime(e744).pack_rng())
    if public_rng_before != public_rng_after:
        raise RuntimeError("private PSRR constructor changed public Python/NumPy/Torch RNG")
    optimizer_psrr = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    initial_module_sha = module_state_digest(module)
    hook = OutputPreHook(state, module, grouping)
    hist, runtime = e744.load_history(), rng_runtime(e744)
    start_epoch, wait = 10, 0
    best_acc, best_epoch, best_metrics = -1.0, -1, None
    trace = []
    gradient_trace = []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        for key, expected in (("schema", "E81.7.2-TP-PSRR-v1"), ("group", group), ("fold", fold),
                              ("source_sha256", current_source), ("config_sha256", sha_file(EXP / "config.json")),
                              ("base_identity", identity), ("warmup_sha256", identity["warmup_sha256"]),
                              ("grouping", grouping), ("selection_mode", selection_mode),
                              ("initial_psrr_sha256", initial_module_sha)):
            if saved.get(key) != expected:
                raise RuntimeError(f"Latest restore identity mismatch at {key}")
        hist.load_bundle(state, saved["current"])
        module.load_state_dict(saved["psrr_state"], strict=True)
        optimizer_psrr.load_state_dict(saved["optimizer_psrr"])
        runtime.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["trace"]
        gradient_trace = saved.get("gradient_trace", [])
        best_file_exists = best_file.is_file()
        if best_file_exists:
            best_payload = torch_load(best_file)
            best_metrics = best_payload["metrics"]
        elif best_epoch >= 0:
            raise RuntimeError("Latest references a Best checkpoint that is missing")

    atomic_json(task_file, {
        "status": "RUNNING", "group": group, "fold": fold, "seed": 0,
        "selection_mode": selection_mode,
        "source_sha256": current_source, "base_identity": identity,
        "warmup_sha256": identity["warmup_sha256"], "initial_psrr_sha256": initial_module_sha,
        "started_epoch": start_epoch,
    })
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    try:
        for epoch_index in range(start_epoch, CFG["max_epochs"]):
            epoch_number = epoch_index + 1
            row = training_step(e744, state, module, optimizer_psrr, hook, group,
                                epoch=epoch_number, fold=fold,
                                gradient_path_probe=(epoch_number == 12))
            trace.append(row["rng"])
            if row["branch_vjp"] is not None or epoch_number % 10 == 0:
                gradient_trace.append({"epoch": epoch_number, "branch_vjp": row["branch_vjp"],
                                       "gradient_control": row["gradient_control"]})
            metrics = row["metrics"]
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                best_epoch = epoch_index
                wait = 0
                best_metrics = metrics
                best_payload = {
                    "schema": "E81.7.2-TP-PSRR-v1", "group": group, "grouping": grouping,
                    "selection_mode": selection_mode,
                    "fold": fold, "seed": 0, "source_sha256": current_source,
                    "initial_psrr_sha256": initial_module_sha,
                    "config_sha256": sha_file(EXP / "config.json"),
                    "warmup_sha256": identity["warmup_sha256"], "base_identity": identity,
                    "base_state": hist.state_bundle(state),
                    "psrr_state": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
                    "optimizer_psrr": copy.deepcopy(optimizer_psrr.state_dict()),
                    "rng": runtime.pack_rng(), "best_epoch": best_epoch,
                    "epoch_1based": epoch_number, "best_acc": best_acc,
                    "metrics": best_metrics,
                    "test_prediction": row["prediction"], "test_probability": row["probability"],
                    "trace": trace, "gradient_trace": gradient_trace,
                }
                atomic_torch(best_file, best_payload)
            else:
                wait += 1

            should_checkpoint = (epoch_number % 10 == 0 or epoch_number == CFG["max_epochs"]
                                 or wait > CFG["early_stop_wait_gt"])
            if should_checkpoint:
                latest = _capture_epoch_state(
                    e744, state, module, optimizer_psrr, identity, fold, group, grouping, initial_module_sha,
                    epoch=epoch_index, best_acc=best_acc, best_epoch=best_epoch, wait=wait,
                    trace=trace, gradient_trace=gradient_trace,
                )
                atomic_torch(latest_file, latest)
                log_row = {
                    "group": group, "fold": fold, "epoch": epoch_number,
                    "best_epoch_1based": best_epoch + 1, "best_acc": best_acc,
                    "metrics": metrics, "reference_loss": row["reference_loss"], "loss": row["total_loss"], "raw_penalty": row["raw_penalty"],
                    "gradient_norms": row["gradient_norms"], "branch_vjp": row["branch_vjp"], "rng": row["rng"],
                    "gradient_control": row["gradient_control"],
                    "peak_gpu_bytes": row["peak_gpu_bytes"], "wait": wait,
                }
                with _log_path(group, fold).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(log_row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f"{group} fold={fold} epoch={epoch_number} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
            if wait > CFG["early_stop_wait_gt"]:
                break
        if best_epoch < 0 or not best_file.is_file():
            raise RuntimeError("training ended without an ACC-selected Best checkpoint")
        if not latest_file.is_file() or torch_load(latest_file).get("next_epoch") != epoch_number:
            latest = _capture_epoch_state(
                e744, state, module, optimizer_psrr, identity, fold, group, grouping, initial_module_sha,
                epoch=epoch_number - 1, best_acc=best_acc, best_epoch=best_epoch, wait=wait,
                trace=trace, gradient_trace=gradient_trace,
            )
            atomic_torch(latest_file, latest)
        completed = {
            "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "selection_mode": selection_mode,
            "source_sha256": current_source, "config_sha256": sha_file(EXP / "config.json"),
            "warmup_sha256": identity["warmup_sha256"], "base_identity": identity,
            "initial_psrr_sha256": initial_module_sha,
            "best_epoch_1based": best_epoch + 1, "epochs_executed": epoch_number,
            "metrics": best_metrics, "best_sha256": sha_file(best_file),
            "latest_sha256": sha_file(latest_file), "rng_trace": trace,
            "gradient_trace": gradient_trace,
            "device_identity": device_identity,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.time() - start_time,
            "fold2_psrr_blocked_rows": CFG["data"]["historical_fold2_psrr_blocked_rows_zero_based"] if fold == 2 else [],
            "fold2_psrr_training_assertion": "PASS",
        }
        atomic_json(task_file, completed)
        return completed
    except BaseException as exc:
        atomic_json(task_file, {
            "status": "FAILED", "group": group, "fold": fold, "source_sha256": current_source,
            "error": repr(exc), "last_epoch": trace[-1]["epoch"] if trace else None,
        })
        raise
    finally:
        hook.close()


def _run_step_for_preflight(e744, state, module, optimizer_psrr, hook, fold, grouping="medical"):
    return training_step(e744, state, module, optimizer_psrr, hook, "PREFLIGHT", epoch=11, fold=fold,
                         gradient_path_probe=False)


def _real_resume_and_path_preflight(e744):
    import torch
    runtime = rng_runtime(e744)
    state_base, warmup_base, identity_base = e744.prepare(1)
    state_branch, warmup_branch, identity_branch = e744.prepare(1)
    if identity_base != identity_branch or warmup_base["rng_sha"] != warmup_branch["rng_sha"]:
        raise RuntimeError("fresh E61 warmup preparation is not reproducible")

    baseline = e744.forward(state_base, E74_GROUP, training=True)
    baseline_total = float(baseline["total"].detach().cpu())
    e744.load_history().zero_optimizers(state_base)
    baseline["total"].backward()
    baseline_gradients = _base_gradients(state_base)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state_base[key].step()
    baseline_bundle = _base_bundle(e744, state_base)
    baseline_rng_after_train = runtime.rng_sha(runtime.pack_rng())
    with torch.no_grad():
        baseline_eval = e744.forward(state_base, E74_GROUP, training=False)
    baseline_rng_after_eval = runtime.rng_sha(runtime.pack_rng())

    runtime.restore_rng(warmup_branch["rng"])
    module = make_module(1, state_branch["cam"].parameters().__next__().device,
                         prompt_payload()["semantic_basis_float64"], "node")
    optimizer_psrr = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    hook = OutputPreHook(state_branch, module, "medical")
    first_step = training_step(e744, state_branch, module, optimizer_psrr, hook, "PREFLIGHT", epoch=11, fold=1)
    if abs(first_step["reference_loss"] - baseline_total) > 1e-12 or abs(first_step["total_loss"] - baseline_total) > 1e-12:
        raise RuntimeError("zero-residual DARG objective changed under reference protection")
    if first_step["gradient_control"]["delta_gradient_norm"] > 1e-6:
        raise RuntimeError("zero-residual enhanced gradient did not reproduce the DARG reference gradient")
    branch_base_bundle = _base_bundle(e744, state_branch)
    branch_rng_after_train = runtime.rng_sha(runtime.pack_rng())
    if not tensor_tree_equal(branch_base_bundle, baseline_bundle)[0]:
        raise RuntimeError("zero-residual first-step DARG update diverged")
    if branch_rng_after_train != baseline_rng_after_train:
        raise RuntimeError("reference pass changed public training RNG consumption")
    with torch.no_grad():
        hook.configure(training=False, intervention="relation_off")
        branch_eval = e744.forward(state_branch, E74_GROUP, training=False)
    branch_rng_after_eval = runtime.rng_sha(runtime.pack_rng())
    if branch_rng_after_eval != baseline_rng_after_eval:
        raise RuntimeError("PSRR changed public evaluation RNG consumption")
    if not torch.equal(branch_eval["logp"], baseline_eval["logp"]):
        raise RuntimeError("zero-residual classifier output did not match DARG after the first step")
    with torch.no_grad():
        hook.configure(training=False, intervention="full")
        e744.forward(state_branch, E74_GROUP, training=False)
    first_step_residual_norm = float(hook.capture["delta"].norm().detach().cpu())
    if first_step_residual_norm <= 0.0:
        raise RuntimeError("zero-initialized output projection did not receive a real gradient")

    snapshot = {
        "base": _base_bundle(e744, state_branch),
        "psrr": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer_psrr": copy.deepcopy(optimizer_psrr.state_dict()),
        "rng": runtime.pack_rng(),
    }
    uninterrupted = training_step(e744, state_branch, module, optimizer_psrr, hook, "PREFLIGHT", epoch=12, fold=1,
                                  gradient_path_probe=True)
    uninterrupted_base = _base_bundle(e744, state_branch)
    uninterrupted_module = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    uninterrupted_opt = optimizer_psrr.state_dict()
    uninterrupted_rng = runtime.pack_rng()
    if uninterrupted["branch_vjp"] is None:
        raise RuntimeError("controlled end-to-end chain VJP was not recorded")
    if any(uninterrupted["branch_vjp"][key] <= 0 for key in (
        "classification_to_delta_norm", "delta_to_cam_input_norm",
        "branch_to_cam_parameter_norm", "branch_to_darg_parameter_norm",
    )):
        raise RuntimeError(f"classification gradient did not traverse the PSRR-to-DARG path: {uninterrupted['branch_vjp']}")

    state_restored, _, identity_restored = e744.prepare(1)
    module_restored = make_module(1, state_restored["cam"].parameters().__next__().device,
                                  prompt_payload()["semantic_basis_float64"], "node")
    optimizer_restored = torch.optim.AdamW(module_restored.parameters(), lr=0.0005, weight_decay=0.01)
    hook_restored = OutputPreHook(state_restored, module_restored, "medical")
    if identity_restored != identity_branch:
        raise RuntimeError("resume test historical warmup identity changed")
    e744.load_history().load_bundle(state_restored, snapshot["base"])
    module_restored.load_state_dict(snapshot["psrr"], strict=True)
    optimizer_restored.load_state_dict(snapshot["optimizer_psrr"])
    runtime.restore_rng(snapshot["rng"])
    restored = training_step(e744, state_restored, module_restored, optimizer_restored, hook_restored,
                             "PREFLIGHT", epoch=12, fold=1)
    restored_base = _base_bundle(e744, state_restored)
    restored_module = {key: value.detach().cpu().clone() for key, value in module_restored.state_dict().items()}
    restored_opt = optimizer_restored.state_dict()
    restored_rng = runtime.pack_rng()
    for name, left, right in (("DARG parameters and optimizers", uninterrupted_base, restored_base),
                              ("PSRR parameters", uninterrupted_module, restored_module),
                              ("PSRR optimizer", uninterrupted_opt, restored_opt),
                              ("public RNG", uninterrupted_rng, restored_rng)):
        equal, where = tensor_tree_equal(left, right)
        if not equal:
            difference, difference_path = tensor_tree_max_difference(left, right)
            raise RuntimeError(f"Latest resume continuation differs in {name} at {where}; max_abs={difference} at {difference_path}")
    if uninterrupted["total_loss"] != restored["total_loss"] or uninterrupted["metrics"] != restored["metrics"]:
        raise RuntimeError("Latest resume continuation changed next-step loss or metrics")
    hook.close()
    hook_restored.close()
    return {
        "zero_residual_classifier_output_exact": True,
        "zero_residual_loss_exact": True,
        "historical_optimizer_update_exact": True,
        "reference_gradient_matches_darg_at_zero_residual": True,
        "reference_forward_state_isolated": True,
        "dropout_and_classifier_forward_count": 1,
        "public_rng_after_train_exact": True,
        "public_rng_after_eval_exact": True,
        "first_step_psrr_gradient_norm": first_step["gradient_norms"]["psrr"],
        "first_step_learned_residual_norm": first_step_residual_norm,
        "second_step_branch_vjp": uninterrupted["branch_vjp"],
        "second_step_gradient_control": uninterrupted["gradient_control"],
        "latest_restore_next_step_exact": True,
        "restored_total_loss": restored["total_loss"],
        "restored_metrics": restored["metrics"],
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }


def preflight():
    import numpy as np
    import torch
    import audit
    from tp_psrr import entmax15
    device_identity = verify_gpu1_device()
    audit_check = audit.preflight_checks()
    e744 = base_runner()
    cached = prompt_payload()
    semantic = semantic_audit(cached)
    entmax_check = _entmax_preflight()
    device = torch.device("cuda:0")
    module_check = _module_preflight(cached, device)

    folds = []
    for fold in range(1, 11):
        state, warmup, identity = e744.prepare(fold)
        overlap = identity["membership"]["train_test_overlap_1based"]
        expected_overlap = [377, 407] if fold == 2 else []
        if overlap != expected_overlap:
            raise RuntimeError(f"fold {fold} historical overlap differs from E74.4")
        if identity.get("train_pair_scope_pass") is not True or state["train_pairs_np"].shape != (4096, 2):
            raise RuntimeError(f"fold {fold} E74.4 training relation pair identity failed")
        if not identity.get("warmup_sha256") or not warmup.get("rng_sha"):
            raise RuntimeError(f"fold {fold} lacks full E61 epoch-10 parameter or RNG identity")
        folds.append({
            "fold": fold,
            "warmup_sha256": identity["warmup_sha256"],
            "warmup_rng_sha256": warmup["rng_sha"],
            "initial_shared_sha256": identity.get("initial_shared_sha"),
            "initial_gfr_sha256": identity.get("initial_gfr_sha"),
            "initial_cam_sha256": identity.get("initial_cam_sha"),
            "split_sha256": identity.get("split_sha"),
            "train_pair_sha256": identity.get("pair_sha"),
            "train_count": int(len(state["train_np"])),
            "test_count": int(len(state["test_np"])),
            "overlap_1based": overlap,
            "train_pairs": int(state["train_pairs_np"].shape[0]),
        })
        del state
        torch.cuda.empty_cache()
    if len(folds) != 10 or len({row["warmup_sha256"] for row in folds}) != 10:
        raise RuntimeError("fold-specific warmup identity is incomplete or unexpectedly reused")
    compatibility = e744.compatibility_report()
    if len(compatibility["warmup_reads"]) != 10:
        raise RuntimeError("all ten folds must load their local warmup compatibility checkpoints")
    for row in folds:
        compat = compatibility["warmup_reads"][str(row["fold"])]
        restore = compat.get("restore_audit", {})
        if compat["source_sha256"] != row["warmup_sha256"] or not all(restore.get(key) is True for key in (
            "model_exact", "cam_exact", "rng_components_exact"
        )) or not all(restore.get("optimizers_exact", {}).values()):
            raise RuntimeError(f"fold {row['fold']} E61 warmup compatibility audit failed")
        row["warmup_compatibility_sha256"] = compat["compatibility_sha256"]

    torch.cuda.reset_peak_memory_stats()
    resume_check = _real_resume_and_path_preflight(e744)
    peak = max(resume_check["peak_gpu_bytes"], module_check.get("peak_gpu_bytes", 0))
    if peak <= 0 and device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated())
    inventory = source_inventory()
    source_sha = source_digest(inventory)
    result = {
        "status": "PASS",
        "experiment": "Experiment 81.7.2 TP-PSRR",
        "independent_audit_implementation": audit_check,
        "source_sha256": source_sha,
        "source_files": inventory,
        "prompt_cache_sha256": sha_file(PROMPT_CACHE),
        "prompt_geometry": semantic,
        "warmup_compatibility": compatibility,
        "entmax": entmax_check,
        "module": module_check,
        "historical_folds": folds,
        "historical_fold_identity_10_of_10": True,
        "zero_residual_and_dropout_parity": resume_check,
        "resume_test": resume_check,
        "gpu": {
            **device_identity,
            "peak_allocated_bytes": peak,
            "safe_peak_bytes_per_worker": int(max(peak * 1.35, 2 * 1024**3)),
            "workers_max": 10,
        },
        "groups": {"G1": "READY", "G2": "READY_INDEPENDENT_REQUIRED"},
        "fold2_history_unchanged_psrr_rows_blocked": [376, 406],
    }
    atomic_json(RESULTS / "preflight.json", result)
    atomic_json(RESULTS / "audit_status.json", {
        "engineering_preflight": "PASS", "formal_training": "NOT_STARTED",
        "g1_completed_folds": 0, "g2_completed_folds": 0,
        "g1_status": "READY", "g2_status": "READY_AFTER_G1_AUDIT",
        "research_decision": "PENDING_BOTH_GROUPS",
    })
    atomic_json(EXP / "manifest.json", {
        "experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
        "experiment_root": str(EXP), "preferred_root": CFG["preferred_root"],
        "fallback_reason": CFG["root_fallback_reason"], "source_sha256": source_sha,
        "source_files": inventory, "prompt_cache_sha256": sha_file(PROMPT_CACHE),
        "warmup_sha256_by_fold": {str(row["fold"]): row["warmup_sha256"] for row in folds},
        "darg_best_sha256_by_fold": {
            str(fold): sha_file(HIST_E744 / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt")
            for fold in range(1, 11)
        },
        "preflight": "results/preflight.json",
        "formal_groups": {"G1": "READY", "G2": "READY_AFTER_G1_AUDIT"},
    })
    print(f"PREFLIGHT PASS source={source_sha} peak_bytes={peak}", flush=True)
    return result


def _nvidia_free_memory():
    result = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    index, uuid, free_mib, total_mib = [value.strip() for value in result.stdout.strip().split(",")]
    if int(index) != 1:
        raise RuntimeError(f"memory query did not return physical GPU1: {result.stdout.strip()}")
    return int(free_mib) * 1024**2, int(total_mib) * 1024**2, uuid.replace("GPU-", "").lower()


def schedule_group(group):
    import signal
    import time
    device_identity = verify_gpu1_device()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_digest():
        raise RuntimeError("scheduler requires a passing source-frozen preflight")
    if group not in ("G1", "G2"):
        raise ValueError("unknown formal group")
    if group == "G2":
        audit_status_path = RESULTS / "audit_status.json"
        if not audit_status_path.is_file() or json.loads(audit_status_path.read_text(encoding="utf-8")).get("g1_status") != "COMPLETE_AUDITED":
            raise RuntimeError("G2 scheduling requires all ten G1 folds to pass independent audit")
    maximum = min(10, int(CFG["execution"]["workers_max"]))
    peak = int(preflight["gpu"]["safe_peak_bytes_per_worker"])
    pending = [fold for fold in range(1, 11) if not _task_path(group, fold).is_file()
               or json.loads(_task_path(group, fold).read_text(encoding="utf-8")).get("status") != "COMPLETE"]
    if not pending:
        print(f"{group} already complete 10/10", flush=True)
        return
    active = {}
    first_worker_measured = False
    while pending or active:
        if active and not first_worker_measured:
            finished = [proc for proc in active if proc.poll() is not None]
            if not finished:
                time.sleep(2)
                continue
            for proc in finished:
                fold, log_handle = active.pop(proc)
                log_handle.close()
                if proc.returncode != 0:
                    raise RuntimeError(f"first measured worker failed for {group} fold {fold}")
                task = json.loads(_task_path(group, fold).read_text(encoding="utf-8"))
                measured_peak = int(task["peak_gpu_bytes"])
                if measured_peak > 0:
                    peak = max(peak, int(measured_peak * 1.25))
                print(f"first worker measured {group} fold={fold} peak={measured_peak}", flush=True)
            first_worker_measured = True

        if pending:
            free_bytes, total_bytes, physical_uuid = _nvidia_free_memory()
            if physical_uuid != device_identity["physical_uuid"]:
                raise RuntimeError("worker memory admission queried a different physical GPU than the CUDA worker")
            reserve = max(4 * 1024**3, int(total_bytes * 0.12))
            safe_slots = min(maximum, max(0, (free_bytes - reserve) // max(peak, 1)))
            if safe_slots == 0:
                time.sleep(5)
                continue
            while pending and len(active) < safe_slots:
                fold = pending.pop(0)
                log_path = LOGS / f"worker_{group}_f{fold:02d}.log"
                log_handle = log_path.open("ab")
                environment = os.environ.copy()
                environment.update({
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                })
                process = subprocess.Popen(
                    [sys.executable, str(EXP / "run.py"), "train", "--group", group, "--fold", str(fold)],
                    stdout=log_handle, stderr=subprocess.STDOUT, env=environment, start_new_session=True,
                )
                active[process] = (fold, log_handle)
                if not first_worker_measured:
                    break
        if active:
            time.sleep(2)
            for process in list(active):
                if process.poll() is not None:
                    fold, log_handle = active.pop(process)
                    log_handle.close()
                    if process.returncode != 0:
                        raise RuntimeError(f"worker failed for {group} fold {fold}, see {_log_path(group, fold)}")
                    task = json.loads(_task_path(group, fold).read_text(encoding="utf-8"))
                    actual_peak = int(task["peak_gpu_bytes"])
                    if actual_peak > 0:
                        peak = max(peak, int(actual_peak * 1.25))
                    print(f"worker complete {group} fold={fold} peak={actual_peak}", flush=True)
        elif pending:
            time.sleep(2)
    task_records = [json.loads(_task_path(group, fold).read_text(encoding="utf-8")) for fold in range(1, 11)]
    if any(record.get("status") != "COMPLETE" for record in task_records):
        raise RuntimeError(f"{group} did not complete ten folds")
    update = json.loads((RESULTS / "audit_status.json").read_text(encoding="utf-8"))
    update[f"{group.lower()}_completed_folds"] = 10
    update["formal_training"] = "COMPLETE_GROUP_PENDING_INDEPENDENT_AUDIT"
    atomic_json(RESULTS / "audit_status.json", update)
    print(f"{group} COMPLETE 10/10", flush=True)


def run_preflight():
    result = preflight()
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--group", required=True)
    train_parser.add_argument("--fold", required=True, type=int)
    schedule_parser = sub.add_parser("schedule")
    schedule_parser.add_argument("--group", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight()
    elif args.command == "train":
        train_fold(args.group, args.fold)
    elif args.command == "schedule":
        schedule_group(args.group)


if __name__ == "__main__":
    main()
