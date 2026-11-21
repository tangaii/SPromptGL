#!/usr/bin/env python3
"""Formal E81.8 training runner for the independent DUA-PSRR and MSR-PSRR groups."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
HIST_E744 = REPO / "experiment/experiment74_4_soft_positive_dcr"
HIST_E8172 = Path("/model/spromptgl/experiment/experiment81_7_2_tp_psrr")
HIST_E8173 = Path("/model/spromptgl/experiment/experiment81_7_3_op_er_psrr")
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RUNTIME = EXP / ".runtime"
CKPTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
RESULTS = EXP / "results"
GROUPS = ("G1", "G2")
E74_GROUP = "E74.4-G2-Positive-DCR"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
_BASE = None
_LEGACY = None
_SITE_IDS = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def load_legacy_helpers():
    global _LEGACY
    if _LEGACY is None:
        path = HIST_E8172 / "run.py"
        spec = importlib.util.spec_from_file_location("experiment8172_readonly_helpers", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _LEGACY = module
    return _LEGACY


def base_runner():
    global _BASE
    if _BASE is None:
        path = EXP / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("experiment818_path_adapter", path)
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        _BASE = adapter.load_e744_runner()
        adapter.install_warmup_compatibility(_BASE)
        _BASE.load_history()
    return _BASE


def prompt_payload():
    import torch
    path = Path(CFG["architecture"]["prompt_cache"])
    if sha_file(path) != CFG["architecture"]["prompt_cache_sha256"]:
        raise RuntimeError("frozen eight-Prompt cache SHA256 mismatch")
    cached = torch.load(path, map_location="cpu", weights_only=False)
    if cached.get("prompts") != CFG["architecture"]["prompts"]:
        raise RuntimeError("E81.7.2 original eight-Prompt identity or order changed")
    if cached.get("encoder") != "OpenAI CLIP RN50, frozen, CPU, eval" or not cached.get("label_independent"):
        raise RuntimeError("Prompt cache is not the frozen, label-independent RN50 cache")
    if cached.get("trainable_prompt_embedding") or cached.get("trainable_text_projection"):
        raise RuntimeError("frozen Prompt cache marks text parameters trainable")
    return cached


def make_module(group, fold, device, semantic_basis=None):
    import torch
    if semantic_basis is None:
        semantic_basis = prompt_payload()["semantic_basis_float64"]
    from dua_psrr import DUAPSRR
    from msr_psrr import MSRPSRR
    module_type = DUAPSRR if group == "G1" else MSRPSRR
    public_cpu = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CFG["architecture"]["private_init_salt"] + int(fold))
        module = module_type(
            semantic_basis,
            rho=CFG["architecture"]["residual_rho"],
            relation_scale=1.0,
            temperature=1.0,
        )
    if not torch.equal(public_cpu, torch.random.get_rng_state()):
        raise RuntimeError("private relation initialization changed public Torch CPU RNG")
    return module.to(device)


class OutputPreHook:
    def __init__(self, state, module):
        self.state = state
        self.module = module
        self.training = False
        self.intervention = "full"
        self.blocked_rows = ()
        self.capture = None
        self.calls = 0
        self.return_diagnostics = True
        layer = getattr(state["cam"], "Outputlayer", None)
        if layer is None:
            raise RuntimeError("PLNNoPEN historical CAM has no Outputlayer")
        self.handle = layer.register_forward_pre_hook(self._replace)

    def _replace(self, _layer, args):
        if not args or args[0].ndim != 2 or args[0].shape[-1] != 144:
            raise RuntimeError("E74.4 Outputlayer input must remain CAM144")
        self.calls += 1
        z_cam = args[0]
        enhanced, details = self.module(
            z_cam,
            grouping="medical",
            intervention=self.intervention,
            blocked_rows=self.blocked_rows if self.training else (),
            return_diagnostics=self.return_diagnostics,
        )
        self.capture = {"z_cam": z_cam, **details}
        return (enhanced, *args[1:])

    def configure(self, *, training, intervention="full", blocked_rows=()):
        self.training = bool(training)
        self.intervention = intervention
        self.blocked_rows = tuple(int(row) for row in blocked_rows)
        self.capture = None
        self.calls = 0

    def close(self):
        self.handle.remove()


def legal_train_rows(state, fold):
    import numpy as np
    rows = np.asarray(state["train_np"], dtype=np.int64)
    if fold == 2:
        blocked = np.asarray(CFG["data"]["blocked_auxiliary_and_training_residual_rows_zero_based"], dtype=np.int64)
        rows = rows[~np.isin(rows, blocked)]
    return rows


def _site_ids(e744):
    global _SITE_IDS
    if _SITE_IDS is None:
        import numpy as np
        runtime = e744.load_history().load_e69().load_e61().load_base()
        (features, _labels), _folds = runtime.load_data()
        feature_map = np.load(REPO / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
        site_columns = [index for index, name in enumerate(feature_map["PHENO"])
                        if str(name).startswith("SITE_ID_")]
        if len(site_columns) != 20:
            raise RuntimeError(f"expected the original 20 ABIDE site columns, got {len(site_columns)}")
        site_one_hot = np.asarray(features[:, site_columns], dtype=np.float64)
        if not np.allclose(site_one_hot.sum(axis=1), 1.0, atol=1e-7, rtol=0):
            raise RuntimeError("historical PHENO site indicators are not one-hot")
        _SITE_IDS = site_one_hot.argmax(axis=1).astype(np.int64)
    return _SITE_IDS


def _supervised_contrastive(vector, labels, sites, legal_rows, temperature=0.1):
    import torch
    import torch.nn.functional as F
    ids = torch.as_tensor(legal_rows, dtype=torch.long, device=vector.device)
    y = labels.to(device=vector.device, dtype=torch.long)
    site = torch.as_tensor(sites, dtype=torch.long, device=vector.device)
    representation = F.normalize(vector.float(), dim=-1, eps=1e-8)
    losses = []
    fallback_anchors = 0
    skipped_anchors = 0
    cross_site_positive_anchors = 0
    same_site_negative_anchors = 0
    for anchor in ids.tolist():
        candidates = ids[ids != anchor]
        same_class = candidates[y[candidates] == y[anchor]]
        cross_site_positive = same_class[site[same_class] != site[anchor]]
        if cross_site_positive.numel():
            positive = cross_site_positive
            cross_site_positive_anchors += 1
        else:
            positive = same_class
            fallback_anchors += 1
        opposite = candidates[y[candidates] != y[anchor]]
        same_site_negative = opposite[site[opposite] == site[anchor]]
        if same_site_negative.numel():
            negative = same_site_negative
            same_site_negative_anchors += 1
        else:
            negative = opposite
        if positive.numel() == 0 or negative.numel() == 0:
            skipped_anchors += 1
            continue
        positive_logits = representation[anchor] @ representation[positive].T / temperature
        negative_logits = representation[anchor] @ representation[negative].T / temperature
        numerator = torch.logsumexp(positive_logits, dim=0)
        denominator = torch.logsumexp(torch.cat((positive_logits, negative_logits)), dim=0)
        losses.append(denominator - numerator)
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = vector.sum() * 0.0
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite MSR supervised contrastive loss")
    return loss, {
        "valid_anchors": len(losses),
        "cross_site_positive_anchors": cross_site_positive_anchors,
        "fallback_positive_anchors": fallback_anchors,
        "same_site_negative_anchors": same_site_negative_anchors,
        "skipped_anchors": skipped_anchors,
    }


def _js_divergence(first, second):
    import torch
    first = first.float()
    second = second.float()
    middle = 0.5 * (first + second)
    first_kl = (first * (first.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=-1)
    second_kl = (second * (second.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=-1)
    return 0.5 * (first_kl + second_kl)


def _msr_auxiliary(module, hook, labels, sites, legal_rows):
    import torch
    z_cam = hook.capture["z_cam"]
    details = hook.capture
    ids = torch.as_tensor(legal_rows, dtype=torch.long, device=z_cam.device)
    contrast, contrast_info = _supervised_contrastive(
        details["contrastive_vector"], labels, sites, legal_rows,
        temperature=CFG["G2_MSR"]["contrastive_temperature"],
    )

    slots = z_cam.detach()[ids].reshape(-1, 16, 9).float()
    scale = slots.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    coordinates = torch.arange(slots.numel(), dtype=torch.float32, device=slots.device).reshape_as(slots)
    perturbation = 0.01 * scale.view(1, 1, 9) * torch.sin(coordinates * 12.9898 + 0.37)
    perturbed = (slots + perturbation).reshape(-1, 144)
    perturbed_td, perturbed_asd = module.relation_distributions(perturbed, grouping="medical")
    baseline_td = details["attention_td"][ids].detach().float()
    baseline_asd = details["attention_asd"][ids].detach().float()
    stable = 0.5 * (_js_divergence(baseline_td, perturbed_td).mean()
                    + _js_divergence(baseline_asd, perturbed_asd).mean())
    if not torch.isfinite(stable):
        raise FloatingPointError("non-finite MSR relation stability loss")
    return contrast, stable, {
        **contrast_info,
        "stable_loss": float(stable.detach().cpu()),
        "legal_patient_count": int(ids.numel()),
        "perturbation_rms_ratio": float((perturbation.norm() / slots.norm().clamp_min(1e-12)).detach().cpu()),
    }


def _gradient_norm(gradients):
    import torch
    pieces = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0


def _combine_gradients(main_gradients, auxiliary_gradients, weight):
    combined = []
    for main, auxiliary in zip(main_gradients, auxiliary_gradients):
        if main is None and auxiliary is None:
            combined.append(None)
        elif main is None:
            combined.append(weight * auxiliary.detach())
        elif auxiliary is None:
            combined.append(main.detach())
        else:
            combined.append(main.detach() + weight * auxiliary.detach())
    return combined


def _gradient_path(main, hook, state):
    import torch
    capture = hook.capture
    class_to_delta = torch.autograd.grad(main["cls"], capture["delta"], retain_graph=True, allow_unused=False)[0]
    delta_to_cam = torch.autograd.grad(
        capture["delta"], capture["z_cam"], grad_outputs=class_to_delta,
        retain_graph=True, allow_unused=False,
    )[0]
    cam_parameters = tuple(parameter for parameter in state["cam"].parameters() if parameter.requires_grad)
    darg_parameters = tuple(parameter for parameter in state["model"].parameters() if parameter.requires_grad)
    cam_gradients = torch.autograd.grad(capture["z_cam"], cam_parameters, grad_outputs=delta_to_cam,
                                        retain_graph=True, allow_unused=True)
    darg_gradients = torch.autograd.grad(capture["z_cam"], darg_parameters, grad_outputs=delta_to_cam,
                                         retain_graph=True, allow_unused=True)
    return {
        "classification_to_residual_norm": float(class_to_delta.detach().norm().cpu()),
        "residual_to_CAM144_norm": float(delta_to_cam.detach().norm().cpu()),
        "residual_to_CAM_parameter_norm": _gradient_norm(cam_gradients),
        "residual_to_GFR_DARG_parameter_norm": _gradient_norm(darg_gradients),
    }


def training_step(e744, legacy, state, module, optimizer_relation, hook, group, *, epoch, fold,
                  gradient_path_probe=False):
    import torch
    runtime = legacy.rng_runtime(e744)
    e744.load_history().zero_optimizers(state)
    optimizer_relation.zero_grad(set_to_none=True)
    blocked_rows = CFG["data"]["blocked_auxiliary_and_training_residual_rows_zero_based"] if fold == 2 else ()
    legal_rows = legal_train_rows(state, fold)
    shared_groups, shared_parameter_list = legacy._shared_optimizer_groups(state)
    shared_parameters = tuple(shared_parameter_list)
    module_parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)

    rng_before_state = runtime.pack_rng()
    rng_before = runtime.rng_sha(rng_before_state)
    buffer_state = legacy._buffer_snapshot(state)
    hook.configure(training=True, intervention="relation_off", blocked_rows=blocked_rows)
    reference = e744.forward(state, E74_GROUP, training=True)
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("original OutputLayer must run once in the DARG reference forward")
    reference_gradients = torch.autograd.grad(reference["total"], shared_parameters,
                                              retain_graph=False, allow_unused=True)
    reference_loss = reference["total"].detach()
    legacy._restore_buffers(state, buffer_state)
    runtime.restore_rng(rng_before_state)

    hook.configure(training=True, intervention="full", blocked_rows=blocked_rows)
    main = e744.forward(state, E74_GROUP, training=True)
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("original OutputLayer must run once in the enhanced forward")
    if fold == 2:
        if torch.count_nonzero(hook.capture["delta"][list(blocked_rows)]):
            raise RuntimeError("Fold2 overlap rows received a nonzero training residual")
        if set(blocked_rows).intersection(legal_rows.tolist()):
            raise RuntimeError("Fold2 overlap rows entered auxiliary supervision")

    auxiliary = None
    auxiliary_details = {}
    auxiliary_gradients = ()
    auxiliary_weight = 0.0
    if group == "G1":
        from dua_psrr import DUAPSRR
        auxiliary, auxiliary_details = module.utility_loss(
            hook.capture,
            hook.capture["z_cam"],
            hook.capture["z_final"],
            main["logp"],
            state["y"],
            legal_rows,
            CFG["data"]["labels"],
            temperature=CFG["G1_DUA"]["utility_temperature"],
            null_weight=CFG["G1_DUA"]["null_loss_weight"],
        )
        auxiliary_weight = CFG["G1_DUA"]["loss_weight"] * min(1.0, (epoch - 10) / CFG["G1_DUA"]["ramp_training_epochs"])
        auxiliary_gradients = torch.autograd.grad(auxiliary, module_parameters, retain_graph=True, allow_unused=True)
        auxiliary_details["effective_weight"] = auxiliary_weight
    else:
        contrast, stable, auxiliary_details = _msr_auxiliary(
            module, hook, state["y"], _site_ids(e744), legal_rows
        )
        auxiliary = (CFG["G2_MSR"]["contrastive_weight"] * contrast
                     + CFG["G2_MSR"]["stability_weight"] * stable)
        auxiliary_weight = 1.0
        contrast_gradients = torch.autograd.grad(contrast, module_parameters, retain_graph=True, allow_unused=True)
        stable_gradients = torch.autograd.grad(stable, module_parameters, retain_graph=True, allow_unused=True)
        auxiliary_gradients = _combine_gradients(
            [CFG["G2_MSR"]["contrastive_weight"] * (gradient if gradient is not None else torch.zeros_like(parameter))
             for parameter, gradient in zip(module_parameters, contrast_gradients)],
            [CFG["G2_MSR"]["stability_weight"] * (gradient if gradient is not None else torch.zeros_like(parameter))
             for parameter, gradient in zip(module_parameters, stable_gradients)],
            1.0,
        )
        auxiliary_details["contrast_loss"] = float(contrast.detach().cpu())
        auxiliary_details["effective_weight"] = 1.0
        auxiliary_details["contrast_gradient_norm"] = _gradient_norm(contrast_gradients)
        auxiliary_details["stability_gradient_norm"] = _gradient_norm(stable_gradients)

    all_main_gradients = torch.autograd.grad(
        main["total"], shared_parameters + module_parameters, retain_graph=True, allow_unused=True
    )
    shared_main = all_main_gradients[:len(shared_parameters)]
    module_main = all_main_gradients[len(shared_parameters):]
    gradient_control = legacy._reference_gradient_control(reference_gradients, shared_main, state)
    if group == "G1":
        module_total = _combine_gradients(module_main, auxiliary_gradients, auxiliary_weight)
    else:
        module_total = _combine_gradients(module_main, auxiliary_gradients, 1.0)
    for parameter, gradient in zip(module_parameters, module_total):
        parameter.grad = None if gradient is None or not bool(torch.count_nonzero(gradient)) else gradient.detach()

    branch_vjp = _gradient_path(main, hook, state) if gradient_path_probe else None
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for child in (state["model"], state["cam"], module) for parameter in child.parameters()):
        raise FloatingPointError("non-finite E81.8 gradient")
    grad_norms = {
        "gfr": legacy.optimizer_grad_norm(state["opt_base"]),
        "cam": legacy.optimizer_grad_norm(state["opt_cam"]),
        "darg": legacy.optimizer_grad_norm(state["opt_added"]),
        "relation_main": _gradient_norm(module_main),
        "relation_auxiliary": _gradient_norm(auxiliary_gradients),
        "relation_total": _gradient_norm(module_total),
    }
    if any(not math.isfinite(value) for value in grad_norms.values()):
        raise FloatingPointError("non-finite E81.8 gradient norm")

    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer_relation.step()
    rng_after_train = runtime.rng_sha(runtime.pack_rng())
    with torch.no_grad():
        hook.configure(training=False)
        evaluated = e744.forward(state, E74_GROUP, training=False)
        metrics = runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
    if hook.calls != 1:
        raise RuntimeError("original OutputLayer must run once in evaluation")
    rng_after_eval = runtime.rng_sha(runtime.pack_rng())
    total_value = float(main["total"].detach().cpu()) + float(auxiliary_weight) * float(auxiliary.detach().cpu())
    return {
        "epoch": int(epoch),
        "reference_loss": float(reference_loss.cpu()),
        "main_loss": float(main["total"].detach().cpu()),
        "auxiliary_loss": float(auxiliary.detach().cpu()),
        "total_loss": total_value,
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "prediction": metrics["pred"],
        "probability": metrics["prob"],
        "gradient_norms": grad_norms,
        "gradient_control": gradient_control,
        "branch_vjp": branch_vjp,
        "auxiliary": auxiliary_details,
        "rng": {"epoch": int(epoch), "before": rng_before, "after_train": rng_after_train, "after_eval": rng_after_eval},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }


def source_inventory():
    e744 = base_runner()
    inventory = dict(e744.source_manifest())
    paths = [
        EXP / "config.json", EXP / "design.md", EXP / "run.py",
        EXP / "tp_psrr.py", EXP / "dua_psrr.py", EXP / "msr_psrr.py", EXP / "path_adapter.py",
        REPO / "network.py", REPO / "layers.py", REPO / "model.py", REPO / "main.py", REPO / "utils.py",
        REPO / "experiment/experiment51_4_d3sag/source/d3sag_base.py",
        REPO / "experiment/experiment51_4_d3sag/source/run.py",
        REPO / "experiment/experiment51_4_d3sag/source/config.json",
        HIST_E744 / "run.py", HIST_E744 / "soft_positive_dcr.py", HIST_E744 / "config.json",
        HIST_E744 / "audit.py", HIST_E744 / "manifest.json",
        HIST_E8172 / "run.py", HIST_E8172 / "tp_psrr.py", HIST_E8172 / "config.json",
        HIST_E8172 / "audit.py", HIST_E8172 / "manifest.json", HIST_E8172 / "design.md",
        HIST_E8172 / "results/report.md", HIST_E8172 / "results/preflight.json",
        HIST_E8172 / "results/fold_metrics.csv", HIST_E8172 / "results/mechanism.csv",
        HIST_E8172 / "results/audit_status.json", HIST_E8172 / "path_adapter.py",
        HIST_E8172 / "prompt_features.pt", HIST_E8172 / ".runtime/darg_best_compat.json",
        HIST_E8172 / ".runtime/compat_training_runner.py",
        HIST_E8173 / "run.py", HIST_E8173 / "op_er_psrr.py", HIST_E8173 / "config.json",
        HIST_E8173 / "audit.py", HIST_E8173 / "manifest.json", HIST_E8173 / "results/report.md",
        HIST_E8173 / "results/audit_status.json",
    ]
    paths.extend(sorted((HIST_E8172 / ".runtime/compat_checkpoints").glob("fold*.pt")))
    paths.extend(sorted((HIST_E744 / ".runtime/checkpoints").glob(f"{E74_GROUP}_f*.best.pt")))
    paths.extend(sorted((HIST_E744 / ".runtime/tasks").glob(f"{E74_GROUP}_f*.json")))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required source or historical input missing: {path}")
        inventory[str(path.resolve())] = sha_file(path)
    return dict(sorted(inventory.items()))


def source_digest(inventory=None):
    return sha_object(source_inventory() if inventory is None else inventory)


def _module_common_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()
            if key in {
                "semantic_basis", "semantic_center", "neighbors", "local_projection.weight", "local_projection.bias",
                "position", "local_norm.weight", "local_norm.bias", "query.weight", "query.bias",
                "query_norm.weight", "query_norm.bias", "visual_key.weight", "visual_key.bias",
                "semantic_reduce.weight", "semantic_expand.weight", "value.weight", "value.bias",
                "null_key", "output.weight",
            }}


def _tree_exact(left, right):
    import torch
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_exact(a, b) for a, b in zip(left, right))
    return left == right


def _optimizer_settings(state):
    expected = {
        "opt_base": (0.0038, 0.11),
        "opt_cam": (0.0038, 0.11),
        "opt_added": (0.001, 0.01),
    }
    actual = {}
    for name, (learning_rate, weight_decay) in expected.items():
        groups = state[name].param_groups
        if not groups or any(group["lr"] != learning_rate or group["weight_decay"] != weight_decay
                             for group in groups):
            raise RuntimeError(f"historical {name} optimizer settings changed: {groups}")
        actual[name] = {"lr": groups[0]["lr"], "weight_decay": groups[0]["weight_decay"],
                        "class": type(state[name]).__name__, "parameter_groups": len(groups)}
    return actual


def _one_worker_module_probe(e744, legacy, group, device):
    import numpy as np
    import torch
    from msr_psrr import MSRPSRR
    state, warmup, identity = e744.prepare(1)
    state["fold"] = 1
    runtime = legacy.rng_runtime(e744)
    module = make_module(group, 1, device, prompt_payload()["semantic_basis_float64"])
    public_rng_after_init = runtime.rng_sha(runtime.pack_rng())
    hook = OutputPreHook(state, module)
    calls = {"count": 0}
    counter = state["cam"].Outputlayer.register_forward_pre_hook(lambda _module, _args: calls.__setitem__("count", calls["count"] + 1))
    with torch.no_grad():
        original = e744.forward(state, E74_GROUP, training=False)
    counter.remove()
    if calls["count"] != 1:
        raise RuntimeError("historical classifier did not execute exactly once in original evaluation")
    hook.configure(training=False)
    with torch.no_grad():
        enhanced = e744.forward(state, E74_GROUP, training=False)
    if hook.calls != 1 or not torch.equal(original["logp"], enhanced["logp"]):
        raise RuntimeError("zero-residual relation path changed original OutputLayer output")
    if torch.count_nonzero(hook.capture["delta"]):
        raise RuntimeError("relation residual is not exactly zero at initialization")
    if tuple(hook.capture["z_cam"].shape) != (871, 144):
        raise RuntimeError(f"active CAM output is not [871,144]: {tuple(hook.capture['z_cam'].shape)}")
    if tuple(hook.capture["local_tokens"].shape) != (871, 16, 32):
        raise RuntimeError("local relation encoder does not preserve 16 latent slots")
    for name in ("attention_td", "attention_asd"):
        probability = hook.capture[name]
        if tuple(probability.shape) != (871, 16, 16) or not torch.allclose(
            probability.sum(dim=-1), torch.ones((871, 16), device=device), atol=3e-6, rtol=0
        ):
            raise RuntimeError(f"{name} must include 15 neighbors and Null and sum to one")
    if group == "G2":
        for class_name in ("td", "asd"):
            adjacency = module._adjacency(hook.capture[f"attention_{class_name}"][..., :15], module.neighbors)
            null_mass = hook.capture[f"attention_{class_name}"][..., 15]
            if not torch.allclose(adjacency.sum(-1), 1.0 - null_mass, atol=5e-6, rtol=0):
                raise RuntimeError("MSR real adjacency renormalized away Null stop-propagation mass")
            if torch.count_nonzero(adjacency.diagonal(dim1=-2, dim2=-1)):
                raise RuntimeError("MSR adjacency contains a self-loop")

    z = torch.linspace(-1.0, 1.0, 871 * 144, device=device).reshape(871, 144)
    with torch.no_grad():
        module.output.weight.fill_(0.002)
        _bounded, blocked_details = module(z, blocked_rows=(376, 406), return_diagnostics=True)
    if torch.count_nonzero(blocked_details["delta"][[376, 406]]):
        raise RuntimeError("Fold2 residual mask did not zero overlap rows")
    if bool((blocked_details["residual_ratio"] > 0.1 + 1e-6).any()):
        raise RuntimeError("bounded CAM144 residual exceeded 10 percent")
    with torch.no_grad():
        module.output.weight.zero_()
    hook.close()
    return {
        "group": group,
        "zero_residual_original_output_exact": True,
        "original_outputlayer_call_count": calls["count"],
        "cam_shape": [871, 144],
        "latent_slots": [871, 16, 9],
        "null_distribution_sums_to_one": True,
        "fold2_blocking": True,
        "max_residual_ratio": float(blocked_details["residual_ratio"].max().cpu()),
        "public_rng_after_private_init": public_rng_after_init,
    }


def preflight():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    legacy = load_legacy_helpers()
    device_identity = legacy.verify_gpu1_device()
    device = torch.device("cuda:0")
    e744 = base_runner()
    cached = prompt_payload()
    semantic = legacy.semantic_audit(cached)
    fold_initializations = []
    for fold in range(1, 11):
        state, warmup, identity = e744.prepare(fold)
        state["fold"] = fold
        runtime = legacy.rng_runtime(e744)
        if runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError(f"fold {fold} did not restore its matching E61 epoch-10 RNG")
        expected_overlap = [377, 407] if fold == 2 else []
        if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
            raise RuntimeError(f"fold {fold} historical membership changed")
        legal_rows = legal_train_rows(state, fold)
        if set(legal_rows.tolist()).intersection(np.asarray(state["test_np"], dtype=np.int64).tolist()):
            raise RuntimeError(f"fold {fold} exposes test patients to new auxiliary supervision")
        if state["train_pairs_np"].shape != (4096, 2):
            raise RuntimeError(f"fold {fold} does not contain the original 4096 relation pairs")
        if type(state["cam"]).__name__ != "PLNNoPEN" or not hasattr(state["cam"], "Outputlayer"):
            raise RuntimeError("E74.4 base is not active PLNNoPEN with the original Outputlayer")
        fold_initializations.append({
            "fold": fold,
            "warmup_sha256": identity["warmup_sha256"],
            "model_cam_sha256": identity["model_cam_sha"],
            "optimizer_sha256": identity["optimizer_sha"],
            "split_sha256": identity["split_sha"],
            "pair_sha256": identity["pair_sha"],
            "legal_auxiliary_patient_count": int(legal_rows.size),
            "legal_auxiliary_test_intersection": 0,
            "membership_overlap_1based": expected_overlap,
            "rng_sha256": warmup["rng_sha"],
            "optimizer_settings": _optimizer_settings(state),
        })
        del state
    probes = [_one_worker_module_probe(e744, legacy, group, device) for group in GROUPS]
    common_a = make_module("G1", 1, device, cached["semantic_basis_float64"])
    common_b = make_module("G2", 1, device, cached["semantic_basis_float64"])
    if not _tree_exact(_module_common_state(common_a), _module_common_state(common_b)):
        raise RuntimeError("G1/G2 common E81.7.2 relation initialization is not RNG aligned")

    training_probes = {}
    max_peak = 0
    for group in GROUPS:
        state, warmup, identity = e744.prepare(1)
        state["fold"] = 1
        module = make_module(group, 1, device, cached["semantic_basis_float64"])
        relation_params = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
        optimizer_relation = torch.optim.AdamW(
            relation_params,
            lr=CFG["base"]["optimizers"]["relation"]["lr"],
            weight_decay=CFG["base"]["optimizers"]["relation"]["weight_decay"],
        )
        hook = OutputPreHook(state, module)
        runtime = legacy.rng_runtime(e744)
        baseline_rng = warmup["rng"]
        runtime.restore_rng(baseline_rng)
        hook.configure(training=True, intervention="relation_off")
        base_output = e744.forward(state, E74_GROUP, training=True)
        if hook.calls != 1:
            raise RuntimeError("baseline RNG probe called the original OutputLayer more than once")
        base_forward_rng = runtime.rng_sha(runtime.pack_rng())
        runtime.restore_rng(baseline_rng)
        first = training_step(e744, legacy, state, module, optimizer_relation, hook, group,
                              epoch=11, fold=1, gradient_path_probe=False)
        second = training_step(e744, legacy, state, module, optimizer_relation, hook, group,
                               epoch=12, fold=1, gradient_path_probe=True)
        if first["rng"]["after_train"] != base_forward_rng:
            raise RuntimeError(f"{group} auxiliary path changed the historical common-forward RNG trajectory")
        if second["branch_vjp"] is None or any(
            second["branch_vjp"][key] <= 0 for key in (
                "classification_to_residual_norm", "residual_to_CAM144_norm",
                "residual_to_CAM_parameter_norm", "residual_to_GFR_DARG_parameter_norm",
            )
        ):
            raise RuntimeError(f"{group} classification gradient did not reach CAM/GFR/DARG")
        if second["gradient_norms"]["relation_main"] <= 0 or second["gradient_norms"]["relation_total"] <= 0:
            raise RuntimeError(f"{group} has no effective main classification gradient to its relation module")
        if group == "G1" and second["gradient_norms"]["relation_auxiliary"] <= 0:
            raise RuntimeError("DUA utility auxiliary gradient did not reach relation parameters")
        if group == "G2" and (second["auxiliary"]["contrast_gradient_norm"] <= 0
                              or second["auxiliary"]["stability_gradient_norm"] <= 0):
            raise RuntimeError("MSR contrastive or stability auxiliary gradient is zero")
        if not math.isfinite(first["total_loss"]) or not math.isfinite(second["total_loss"]):
            raise FloatingPointError(f"{group} preflight produced a non-finite objective")
        max_peak = max(max_peak, first["peak_gpu_bytes"], second["peak_gpu_bytes"])
        training_probes[group] = {
            "step11_gradient_norms": first["gradient_norms"],
            "step12_gradient_norms": second["gradient_norms"],
            "step12_branch_vjp": second["branch_vjp"],
            "step12_gradient_control": second["gradient_control"],
            "step12_auxiliary": second["auxiliary"],
            "common_forward_rng_aligned": True,
            "auxiliary_gradients_assigned_only_to_relation_module": True,
            "max_gpu_bytes": max(first["peak_gpu_bytes"], second["peak_gpu_bytes"]),
        }
        hook.close()
        del state, module, optimizer_relation
    torch.cuda.synchronize()
    source_files = source_inventory()
    frozen_sha = source_digest(source_files)
    audit_source_sha = sha_file(EXP / "audit.py")
    payload = {
        "status": "PASS",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": frozen_sha,
        "source_files": source_files,
        "audit_source_sha256": audit_source_sha,
        "gpu": device_identity,
        "fold_initializations": fold_initializations,
        "prompt_semantic_audit": semantic,
        "group_zero_residual_probes": probes,
        "training_gradient_probes": training_probes,
        "single_worker_peak_bytes": max_peak,
        "required_checks": {
            "E61_epoch10_all_folds": True,
            "three_historical_optimizers_and_update_order": True,
            "shared_rng_trajectory": True,
            "CAM144_and_16x9": True,
            "original_prompts_and_frozen_RN50_cache": True,
            "null_zero_message_and_mass_preservation": True,
            "zero_residual_original_classifier_exact": True,
            "end_to_end_classification_gradient": True,
            "auxiliary_gradients_excluded_from_shared_backbone": True,
            "Fold2_auxiliary_and_training_residual_mask": True,
            "ten_percent_residual_bound": True,
            "finite_losses_and_gradients": True,
            "physical_GPU1_only": True
        }
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULTS / "preflight.json", payload)
    atomic_json(EXP / "manifest.json", {
        "status": "FROZEN",
        "source_sha256": frozen_sha,
        "source_files": source_files,
        "audit_source_sha256": audit_source_sha,
        "groups": list(GROUPS),
        "tasks": 20,
        "group_folds": 10,
        "seed": 0,
        "preflight": str((RESULTS / "preflight.json").resolve()),
    })
    print(json.dumps({"preflight": "PASS", "source_sha256": frozen_sha,
                      "single_worker_peak_bytes": max_peak, "gpu": device_identity}, indent=2), flush=True)
    return payload


def _task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def _best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def _latest_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def _log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def _checkpoint_payload(group, fold, identity, source_sha, module, optimizer_relation, state, legacy,
                        *, kind, current_epoch, best_acc, best_epoch, wait, trace, gradient_trace,
                        current_metrics=None, best_metrics=None, prediction=None, probability=None):
    import copy
    runtime = legacy.rng_runtime(base_runner())
    payload = {
        "schema": "E81.8-DUA-MSR-PSRR-v1",
        "group": group,
        "fold": fold,
        "seed": 0,
        "kind": kind,
        "source_sha256": source_sha,
        "config_sha256": sha_file(EXP / "config.json"),
        "identity": identity,
        "warmup_sha256": identity["warmup_sha256"],
        "current_epoch_1based": current_epoch + 1,
        "next_epoch_index": current_epoch + 1,
        "best_acc": best_acc,
        "best_epoch_index": best_epoch,
        "wait": wait,
        "base_state": base_runner().load_history().state_bundle(state),
        "relation_state": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer_relation": copy.deepcopy(optimizer_relation.state_dict()),
        "rng": runtime.pack_rng(),
        "trace": trace,
        "gradient_trace": gradient_trace,
        "current_metrics": current_metrics,
        "best_metrics": best_metrics,
        "prediction": prediction,
        "probability": probability,
    }
    return payload


def train_worker(group, fold):
    import numpy as np
    import torch
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid E81.8 task group={group}, fold={fold}")
    legacy = load_legacy_helpers()
    device_identity = legacy.verify_gpu1_device()
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    current_source = source_digest()
    if preflight_record.get("status") != "PASS" or preflight_record.get("source_sha256") != current_source:
        raise RuntimeError("training sources differ from the frozen passing preflight")
    e744 = base_runner()
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    state["group"] = group
    runtime = legacy.rng_runtime(e744)
    if runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold}/{group} did not restore the E61 epoch-10 RNG")
    if state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError(f"fold {fold}/{group} has a changed relation-pair population")
    task_file, best_file, latest_file = _task_path(group, fold), _best_path(group, fold), _latest_path(group, fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == current_source:
            if existing.get("best_sha256") == sha_file(best_file) and existing.get("latest_sha256") == sha_file(latest_file):
                print(f"SKIP complete {group} fold={fold}", flush=True)
                return existing
    module = make_module(group, fold, device, prompt_payload()["semantic_basis_float64"])
    relation_parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    optimizer_relation = torch.optim.AdamW(
        relation_parameters,
        lr=CFG["base"]["optimizers"]["relation"]["lr"],
        weight_decay=CFG["base"]["optimizers"]["relation"]["weight_decay"],
    )
    expected_relation_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    actual_relation_ids = {id(parameter) for group_state in optimizer_relation.param_groups for parameter in group_state["params"]}
    if actual_relation_ids != {id(parameter) for parameter in relation_parameters}:
        raise RuntimeError("new relation optimizer does not own every trainable relation parameter")
    hook = OutputPreHook(state, module)
    trace, gradient_trace = [], []
    start_epoch, wait, best_acc, best_epoch = 10, 0, -1.0, -1
    best_metrics = None
    if latest_file.is_file():
        latest = torch_load(latest_file)
        if latest.get("source_sha256") != current_source or latest.get("identity") != identity:
            raise RuntimeError("Latest checkpoint source or fold identity is incompatible")
        e744.load_history().load_bundle(state, latest["base_state"])
        module.load_state_dict(latest["relation_state"], strict=True)
        optimizer_relation.load_state_dict(latest["optimizer_relation"])
        runtime.restore_rng(latest["rng"])
        start_epoch = int(latest["next_epoch_index"])
        wait = int(latest["wait"])
        best_acc = float(latest["best_acc"])
        best_epoch = int(latest["best_epoch_index"])
        trace = list(latest["trace"])
        gradient_trace = list(latest["gradient_trace"])
        best_metrics = latest.get("best_metrics")
        if best_epoch >= 0 and not best_file.is_file():
            raise RuntimeError("Latest references a Best checkpoint that is missing")
        if wait > CFG["early_stop_wait_gt"]:
            start_epoch = CFG["max_epochs"]
    started = time.time()
    atomic_json(task_file, {
        "status": "RUNNING", "group": group, "fold": fold, "seed": 0,
        "source_sha256": current_source, "identity": identity,
        "warmup_sha256": identity["warmup_sha256"],
        "relation_parameter_names": sorted(expected_relation_names),
        "relation_parameter_count": sum(parameter.numel() for parameter in relation_parameters),
        "device_identity": device_identity,
        "started_epoch_1based": start_epoch + 1,
    })
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch_index in range(start_epoch, CFG["max_epochs"]):
            epoch_number = epoch_index + 1
            row = training_step(
                e744, legacy, state, module, optimizer_relation, hook, group,
                epoch=epoch_number, fold=fold,
                gradient_path_probe=(epoch_number == 12),
            )
            trace.append(row["rng"])
            if row["branch_vjp"] is not None or epoch_number % 10 == 0:
                gradient_trace.append({
                    "epoch": epoch_number,
                    "branch_vjp": row["branch_vjp"],
                    "gradient_control": row["gradient_control"],
                    "gradient_norms": row["gradient_norms"],
                    "auxiliary": row["auxiliary"],
                })
            if row["metrics"]["acc"] > best_acc:
                best_acc = row["metrics"]["acc"]
                best_epoch = epoch_index
                best_metrics = row["metrics"]
                wait = 0
                best_payload = _checkpoint_payload(
                    group, fold, identity, current_source, module, optimizer_relation, state, legacy,
                    kind="Best", current_epoch=epoch_index, best_acc=best_acc, best_epoch=best_epoch,
                    wait=wait, trace=trace, gradient_trace=gradient_trace,
                    current_metrics=row["metrics"], best_metrics=best_metrics,
                    prediction=np.asarray(row["prediction"]).tolist(),
                    probability=np.asarray(row["probability"]).tolist(),
                )
                atomic_torch(best_file, best_payload)
            else:
                wait += 1
            latest_payload = _checkpoint_payload(
                group, fold, identity, current_source, module, optimizer_relation, state, legacy,
                kind="Latest", current_epoch=epoch_index, best_acc=best_acc, best_epoch=best_epoch,
                wait=wait, trace=trace, gradient_trace=gradient_trace,
                current_metrics=row["metrics"], best_metrics=best_metrics,
                prediction=np.asarray(row["prediction"]).tolist(),
                probability=np.asarray(row["probability"]).tolist(),
            )
            atomic_torch(latest_file, latest_payload)
            log_record = {
                "group": group, "fold": fold, "epoch": epoch_number,
                "best_epoch_1based": best_epoch + 1, "best_acc": best_acc, "wait": wait,
                "metrics": row["metrics"], "main_loss": row["main_loss"],
                "auxiliary_loss": row["auxiliary_loss"], "total_loss": row["total_loss"],
                "gradient_norms": row["gradient_norms"], "gradient_control": row["gradient_control"],
                "branch_vjp": row["branch_vjp"], "auxiliary": row["auxiliary"],
                "rng": row["rng"], "peak_gpu_bytes": row["peak_gpu_bytes"],
            }
            LOGS.mkdir(parents=True, exist_ok=True)
            with _log_path(group, fold).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log_record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if epoch_number % 10 == 0 or wait == 0 or wait % 25 == 0:
                print(f"{group} fold={fold} epoch={epoch_number} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
            if wait > CFG["early_stop_wait_gt"]:
                break
        if best_epoch < 0 or not best_file.is_file():
            raise RuntimeError("training ended without a strict-ACC Best checkpoint")
        task = {
            "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": current_source,
            "identity": identity,
            "warmup_sha256": identity["warmup_sha256"],
            "best_epoch_1based": best_epoch + 1,
            "epochs_executed": trace[-1]["epoch"],
            "best_metrics": best_metrics,
            "best_sha256": sha_file(best_file),
            "latest_sha256": sha_file(latest_file),
            "rng_trace": trace,
            "gradient_trace": gradient_trace,
            "device_identity": device_identity,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.time() - started,
            "fold2_added_branch_exclusion": "PASS" if fold == 2 else "NOT_APPLICABLE",
        }
        atomic_json(task_file, task)
        print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
        return task
    except BaseException as error:
        atomic_json(task_file, {
            "status": "FAILED", "group": group, "fold": fold,
            "source_sha256": current_source, "identity": identity,
            "error": repr(error), "traceback": __import__("traceback").format_exc(),
            "last_epoch": trace[-1]["epoch"] if trace else None,
        })
        raise
    finally:
        hook.close()


def _physical_gpu_free_bytes():
    result = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(result) * 1024**2


def _completed(group, fold):
    task_file, best_file, latest_file = _task_path(group, fold), _best_path(group, fold), _latest_path(group, fold)
    if not task_file.is_file() or not best_file.is_file() or not latest_file.is_file():
        return False
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        return (task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(best_file)
                and task.get("latest_sha256") == sha_file(latest_file))
    except Exception:
        return False


def schedule_group(group, requested_workers):
    import concurrent.futures
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    one_worker_peak = max(1, int(preflight_record["single_worker_peak_bytes"]))
    free_bytes = _physical_gpu_free_bytes()
    reserve_bytes = 16 * 1024**3
    per_worker_budget = int(one_worker_peak * 1.5)
    memory_admitted = max(1, (free_bytes - reserve_bytes) // max(1, per_worker_budget))
    worker_limit = min(int(requested_workers), 10, int(memory_admitted))
    if worker_limit < 1:
        worker_limit = 1
    print(f"{group}: GPU1 free={free_bytes} bytes, one-worker peak={one_worker_peak} bytes, workers={worker_limit}", flush=True)
    pending = [fold for fold in range(1, 11) if not _completed(group, fold)]
    retries = {fold: 0 for fold in pending}
    failures = []
    run_env = os.environ.copy()
    run_env.update({
        "CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    while pending:
        active = {}
        wave = list(pending)
        pending = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as executor:
            for fold in wave:
                while len(active) >= worker_limit:
                    done = [process for process in active if process.poll() is not None]
                    if done:
                        break
                    time.sleep(2)
                if len(active) >= worker_limit:
                    for process in done:
                        fold_done = active.pop(process)
                        process.wait()
                        if process.returncode != 0:
                            failures.append((fold_done, process.returncode))
                            retries[fold_done] += 1
                            if retries[fold_done] <= 3:
                                pending.append(fold_done)
                            worker_limit = max(1, worker_limit // 2)
                worker_log = LOGS / f"worker_{group}_f{fold:02d}.log"
                worker_log.parent.mkdir(parents=True, exist_ok=True)
                log_stream = worker_log.open("ab")
                process = subprocess.Popen(
                    [sys.executable, str(EXP / "run.py"), "--worker", group, str(fold)],
                    cwd=str(REPO), env=run_env, stdout=log_stream, stderr=subprocess.STDOUT,
                )
                active[process] = fold
                executor.submit(log_stream.close)
            while active:
                for process in list(active):
                    if process.poll() is None:
                        continue
                    fold_done = active.pop(process)
                    process.wait()
                    if process.returncode != 0:
                        failures.append((fold_done, process.returncode))
                        retries[fold_done] += 1
                        if retries[fold_done] <= 3:
                            pending.append(fold_done)
                        worker_limit = max(1, worker_limit // 2)
                    elif not _completed(group, fold_done):
                        failures.append((fold_done, "missing completion record"))
                        retries[fold_done] += 1
                        if retries[fold_done] <= 3:
                            pending.append(fold_done)
                if active:
                    time.sleep(2)
        print(f"{group}: completed={sum(_completed(group, fold) for fold in range(1, 11))}/10, retry={pending}", flush=True)
        if failures and not pending:
            missing = [fold for fold in range(1, 11) if not _completed(group, fold)]
            if missing:
                pending = missing
                for fold in pending:
                    retries.setdefault(fold, 0)
                    retries[fold] += 1
                    if retries[fold] > 3:
                        raise RuntimeError(f"folds failed after resource-aware retries: {failures}")
        if any(retries.get(fold, 0) > 3 for fold in pending):
            raise RuntimeError(f"worker retries exhausted for {group}: {failures}")
    if not all(_completed(group, fold) for fold in range(1, 11)):
        raise RuntimeError(f"{group} did not complete all ten folds")
    return {"group": group, "workers_max": worker_limit, "failures_retried": failures}


def run_all(max_workers=10):
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS" or source_digest() != preflight_record.get("source_sha256"):
        raise RuntimeError("formal training requires the unchanged frozen preflight sources")
    started = time.time()
    schedules = [schedule_group(group, max_workers) for group in GROUPS]
    elapsed = time.time() - started
    atomic_json(RESULTS / "training_status.json", {
        "status": "20_OF_20_COMPLETE",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "schedules": schedules,
        "tasks": [json.loads(_task_path(group, fold).read_text(encoding="utf-8"))
                  for group in GROUPS for fold in range(1, 11)],
    })
    print(f"E81.8 training complete 20/20 in {elapsed:.1f}s", flush=True)
    return elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("GROUP", "FOLD"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.worker:
        train_worker(args.worker[0], int(args.worker[1]))
    elif args.run:
        run_all(args.workers)
    else:
        parser.error("choose --preflight, --worker GROUP FOLD, or --run")


if __name__ == "__main__":
    main()
