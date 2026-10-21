#!/usr/bin/env python3
"""Reproduce the SPromptGL CLIP initialization and feature equivalence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
UPSTREAM_REPO = "https://github.com/wanxixi11/SPromptGL"
UPSTREAM_COMMIT = "ba3b9446fa451fde6864d6555bdf324de0c4b64a"
UPSTREAM_FILE_SHA256 = {
    "network.py": "d0ac59ead2fecdf517845e2c389d361ef9924abff29707386f847e1fbaa22166",
    "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
    "utils.py": "25afd06c4a46f278aeab2eef166b930bdfab53b901956c30e03bb3b499688001",
    "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
}
PROMPTS = [
    "Nromal brain volume",
    "Uniform cortical thickness",
    "Symmetrical brain structure",
    "Normal ventricular morphology",
    "Altered brain volume",
    "Abnormal functional connectivity",
    "White matter integrity issues",
    "Different activation patterns",
]
SEED = 0
ATOL = 1e-6
RTOL = 1e-5
INPUT_DIMS = [48, 6, 10, 256]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def tensor_summary(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha256(tensor),
        "norms": [float(item) for item in value.norm(dim=1).tolist()],
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
    torch.set_rng_state(cpu_generator.get_state())
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_summary() -> dict:
    result = {"cpu_sha256": tensor_sha256(torch.get_rng_state())}
    if torch.cuda.is_available():
        result["cuda_sha256"] = tensor_sha256(torch.cuda.get_rng_state(torch.cuda.current_device()))
    else:
        result["cuda_sha256"] = None
    return result


def get_tokens(clip_module, device: torch.device) -> tuple[torch.Tensor, list[list[int]]]:
    tokens = clip_module.tokenize(PROMPTS)
    token_ids = tokens.tolist()
    visible_ids = []
    for row in token_ids:
        eot = max(index for index, token_id in enumerate(row) if token_id != 0)
        visible_ids.append(row[: eot + 1])
    return tokens.to(device), visible_ids


def encode(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.encode_text(tokens).float().detach().cpu().contiguous()


def feature_payload(features: torch.Tensor) -> dict:
    value = features.detach().float().cpu().contiguous()
    return {"summary": tensor_summary(value), "values": value.tolist()}


def state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def parameter_hashes(module: nn.Module, allow=None) -> dict[str, str]:
    return {
        name: tensor_sha256(parameter)
        for name, parameter in module.named_parameters()
        if allow is None or allow(name)
    }


def feature_worker(role: str, upstream_dir: Path | None) -> dict:
    sys.path.insert(0, str(ROOT))
    if role == "b":
        if upstream_dir is None:
            raise RuntimeError("B worker requires the fixed upstream source snapshot")
        sys.path.insert(0, str(upstream_dir))

    from CLIP.clip import clip

    seed_all()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokens, visible_token_ids = get_tokens(clip, device)
    checkpoint = Path.home() / ".cache" / "clip" / "RN50.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected local RN50 checkpoint not found: {checkpoint}")

    tokenizer_path = Path(clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
    result = {
        "role": role,
        "seed": SEED,
        "device": str(device),
        "prompts": PROMPTS,
        "token_ids": visible_token_ids,
        "token_ids_sha256": sha256_bytes(json.dumps(visible_token_ids, separators=(",", ":")).encode()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenizer_module": str(Path(clip.__file__).resolve()),
        "rng_after_seed": rng_summary(),
    }

    if role == "a":
        clip_model, _ = clip.load("RN50", device)
        clip_model.eval()
        features = encode(clip_model, tokens)
        result.update(
            model_training=clip_model.training,
            model_dtype=str(next(clip_model.parameters()).dtype),
            text_parameter_sha256=parameter_hashes(clip_model, lambda name: not name.startswith("visual.")),
            features=feature_payload(features),
            rng_after_model_load=rng_summary(),
        )
        return result

    source_event: dict = {}
    apply_events: list[dict] = []
    original_apply = nn.Module.apply

    def tracked_apply(module, fn):
        is_pln = module.__class__.__name__ == "PLN"
        event = None
        if is_pln:
            event = {"rng_before": rng_summary()}
            if hasattr(module, "prompt_text_features"):
                event["buffer_sha256_before"] = tensor_sha256(module.prompt_text_features)
        value = original_apply(module, fn)
        if is_pln:
            event["rng_after"] = rng_summary()
            if hasattr(module, "prompt_text_features"):
                event["buffer_sha256_after"] = tensor_sha256(module.prompt_text_features)
            apply_events.append(event)
        return value

    nn.Module.apply = tracked_apply
    try:
        if role == "b":
            import network
            if Path(network.__file__).resolve().parent != upstream_dir.resolve():
                raise RuntimeError(f"B worker imported the wrong network.py: {network.__file__}")

            original_load = clip.load

            def tracked_load(*args, **kwargs):
                loaded, preprocess = original_load(*args, **kwargs)
                source_event["training_on_load"] = loaded.training
                source_event["dtype_on_load"] = str(next(loaded.parameters()).dtype)
                source_event["registered_class_name"] = loaded.__class__.__name__
                source_event["pre_init_parameters"] = state_snapshot(loaded)
                source_event["pre_init_text_hashes"] = parameter_hashes(
                    loaded, lambda name: not name.startswith("visual.")
                )
                source_event["pre_init_features"] = encode(loaded.eval(), tokens)
                source_event["model"] = loaded
                return loaded, preprocess

            clip.load = tracked_load
            helper = _construct_eval_helper("b")
            rng_after_helper_init = rng_summary()
            if Path(sys.modules["model"].__file__).resolve().parent != upstream_dir.resolve():
                raise RuntimeError(f"B worker imported the wrong model.py: {sys.modules['model'].__file__}")
            clip_model = source_event["model"]
            if not any(module is clip_model for module in helper.ModalFusion.modules()):
                raise RuntimeError("B worker did not register the upstream CLIP model in PLN")
            post_init_features = encode(clip_model.eval(), tokens)
            post_init_hashes = parameter_hashes(clip_model, lambda name: not name.startswith("visual."))
            post_state = state_snapshot(clip_model)
            initial_mode = {
                "modal_fusion_training": helper.ModalFusion.training,
                "registered_clip_training": clip_model.training,
            }
            helper.ModalFusion.train()
            post_init_train_features = encode(clip_model, tokens)
            train_mode = {
                "modal_fusion_training": helper.ModalFusion.training,
                "registered_clip_training": clip_model.training,
                "mha_dropout_values": sorted({float(module.dropout) for module in clip_model.modules() if isinstance(module, nn.MultiheadAttention)}),
                "dropout_module_count": sum(isinstance(module, nn.Dropout) for module in clip_model.modules()),
            }
            helper.ModalFusion.eval()
            trunk = _collect_initialization_groups(helper, include_clip=False)
            optimizer = _optimizer_summary(helper, "optimizer_MF")
            result.update(
                upstream_source={
                    "repo": UPSTREAM_REPO,
                    "commit": UPSTREAM_COMMIT,
                    "path": str(upstream_dir),
                },
                clip_registration={
                    "registered_as_child": any(module is clip_model for module in helper.ModalFusion.modules()),
                    "named_module_paths": [name for name, module in helper.ModalFusion.named_modules() if module is clip_model],
                    "pre_init_training_modes": initial_mode,
                    "clip_dtype": source_event["dtype_on_load"],
                    "train_mode_after_modal_fusion_train": train_mode,
                    "train_eval_feature_equal": torch.equal(post_init_features, post_init_train_features),
                    "train_eval_max_abs_error": float((post_init_features - post_init_train_features).abs().max().item()),
                },
                clip_pre_init_features=feature_payload(source_event["pre_init_features"]),
                features=feature_payload(post_init_features),
                text_parameter_sha256_before=source_event["pre_init_text_hashes"],
                text_parameter_sha256_after=post_init_hashes,
                linear_weight_changes=_linear_changes(clip_model, source_event["pre_init_parameters"], post_state),
                mha_parameter_changes=_mha_changes(clip_model, source_event["pre_init_parameters"], post_state),
                text_parameter_changes=_parameter_changes(source_event["pre_init_parameters"], post_state, "visual."),
                initialization_groups=trunk,
                optimizer=optimizer,
                rng_after_model_initialization=rng_after_helper_init,
                apply_events=apply_events,
            )
            return result

        if role == "c":
            import network
            from model import EvalHelper

            original_get = network.get_prompt_text_features
            cache_calls: list[torch.Tensor] = []

            def tracked_get():
                value = original_get()
                cache_calls.append(value.detach().clone())
                return value

            network.get_prompt_text_features = tracked_get
            helper = _construct_eval_helper("c", EvalHelper=EvalHelper)
            rng_after_helper_init = rng_summary()
            cache_first = cache_calls[0]
            cache_second = network.get_prompt_text_features().detach().clone()
            buffer = helper.ModalFusion.prompt_text_features
            registered_buffer = dict(helper.ModalFusion.named_buffers()).get("prompt_text_features")
            before_move = buffer.detach().clone()
            helper.ModalFusion.to("cpu")
            after_cpu_move = helper.ModalFusion.prompt_text_features.detach().clone()
            helper.ModalFusion.to(device)
            after_restore = helper.ModalFusion.prompt_text_features.detach().clone()
            direct_model, _ = clip.load("RN50", device)
            direct_model.eval()
            direct_features = encode(direct_model, tokens)
            trunk = _collect_initialization_groups(helper, include_clip=False)
            optimizer = _optimizer_summary(helper, "optimizer_MF")
            result.update(
                cache={
                    "first_call_summary": tensor_summary(cache_first),
                    "second_call_equal": torch.equal(cache_first, cache_second),
                    "second_call_max_abs_error": float((cache_first - cache_second).abs().max().item()),
                    "registered_buffer": registered_buffer is buffer,
                    "buffer_dtype": str(buffer.dtype),
                    "buffer_requires_grad": buffer.requires_grad,
                    "buffer_device_initial": str(before_move.device),
                    "buffer_device_cpu_move": str(after_cpu_move.device),
                    "buffer_device_restored": str(after_restore.device),
                    "buffer_cpu_move_equal": torch.equal(before_move.cpu(), after_cpu_move),
                    "buffer_restore_equal": torch.equal(before_move, after_restore),
                    "buffer_sha256_before_move": tensor_sha256(before_move),
                    "buffer_sha256_after_apply": apply_events[-1].get("buffer_sha256_after") if apply_events else None,
                    "buffer_sha256_after_restore": tensor_sha256(after_restore),
                    "direct_encode_equal": torch.equal(direct_features, cache_first.float().cpu()),
                    "direct_encode_max_abs_error": float((direct_features - cache_first.float().cpu()).abs().max().item()),
                    "loader_model_training": direct_model.training,
                    "loader_model_dtype": str(next(direct_model.parameters()).dtype),
                    "apply_events": apply_events,
                },
                features=feature_payload(cache_first.float().cpu()),
                initialization_groups=trunk,
                optimizer=optimizer,
                rng_after_model_initialization=rng_after_helper_init,
            )
            return result
    finally:
        nn.Module.apply = original_apply

    raise ValueError(f"Unknown worker role: {role}")


def _construct_eval_helper(role: str, EvalHelper=None):
    if EvalHelper is None:
        from model import EvalHelper

    labels = np.concatenate((np.zeros(403, dtype=np.int64), np.ones(468, dtype=np.int64)))
    train_index = np.arange(800, dtype=np.int64)
    test_index = np.arange(800, 871, dtype=np.int64)
    features = np.zeros((871, sum(INPUT_DIMS)), dtype=np.float32)
    hyperpm = SimpleNamespace(
        n_hidden=18,
        n_head=2,
        dropout=0.35,
        nlayer=1,
        nmodal=4,
        nclass=2,
        GC_mode="weighted-cosine",
        MP_mode="GCN",
        MF_mode="concat",
        alpha=0.5,
        th=0.9,
        lr=0.0038,
        reg=0.11,
    )
    return EvalHelper(INPUT_DIMS, features, labels, hyperpm, train_index, test_index)


def _collect_initialization_groups(helper, include_clip: bool) -> dict:
    modal = helper.ModalFusion
    params = dict(modal.named_parameters())
    result = {
        "gfr": parameter_hashes(helper.MessagePassing0),
        "cam": {
            name: tensor_sha256(parameter)
            for name, parameter in params.items()
            if name == "fusion_weight"
            or name.startswith("InputLayer.")
            or name.startswith("encode_")
            or name.startswith("feed_")
        },
        "output_layer": {
            name: tensor_sha256(parameter)
            for name, parameter in params.items()
            if name.startswith("Outputlayer.")
        },
        "pen_mlp": {
            name: tensor_sha256(parameter)
            for name, parameter in params.items()
            if name.startswith("MLP.")
        },
    }
    if include_clip:
        result["clip_registered"] = {
            name: tensor_sha256(parameter)
            for name, parameter in params.items()
            if name.startswith("ViT_model.")
        }
    return result


def _optimizer_summary(helper, name: str) -> dict:
    optimizer = getattr(helper, name)
    named = dict(helper.ModalFusion.named_parameters())
    by_id = {id(parameter): parameter_name for parameter_name, parameter in named.items()}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    optimizer_names = [by_id.get(parameter_id, "<unregistered>") for parameter_id in optimizer_ids]
    all_trainable_ids = {id(parameter) for parameter in named.values() if parameter.requires_grad}
    optimizer_set = set(optimizer_ids)
    included_clip = [name for name in optimizer_names if name.startswith("ViT_model.")]
    return {
        "optimizer_name": name,
        "parameter_count": len(optimizer_ids),
        "parameter_numel": int(sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"])),
        "all_registered_trainable_params_included": optimizer_set == all_trainable_ids,
        "unregistered_parameter_names": [item for item in optimizer_names if item == "<unregistered>"],
        "registered_trainable_parameter_count": sum(parameter.requires_grad for parameter in named.values()),
        "registered_trainable_parameter_numel": int(sum(parameter.numel() for parameter in named.values() if parameter.requires_grad)),
        "clip_parameter_count": len(included_clip),
        "clip_parameter_numel": int(sum(named[item].numel() for item in included_clip)),
        "pen_mlp_parameter_names": [item for item in optimizer_names if item.startswith("MLP.")],
        "clip_parameter_names": included_clip,
        "optimizer_parameter_names_sha256": sha256_bytes("\n".join(optimizer_names).encode()),
    }


def _linear_changes(model, before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[dict]:
    changes = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        old, new = before[parameter_name], after[parameter_name]
        delta = (old.float() - new.float()).abs()
        changes.append(
            {
                "name": parameter_name,
                "shape": list(old.shape),
                "changed": not torch.equal(old, new),
                "before_sha256": tensor_sha256(old),
                "after_sha256": tensor_sha256(new),
                "max_abs_delta": float(delta.max().item()),
                "mean_abs_delta": float(delta.mean().item()),
            }
        )
    return changes


def _mha_changes(model, before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[dict]:
    result = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue
        prefix = f"{module_name}." if module_name else ""
        names = [prefix + "in_proj_weight", prefix + "in_proj_bias", prefix + "out_proj.weight", prefix + "out_proj.bias"]
        for name in names:
            if name not in before or name not in after:
                continue
            old, new = before[name], after[name]
            delta = (old.float() - new.float()).abs()
            result.append(
                {
                    "name": name,
                    "changed": not torch.equal(old, new),
                    "before_sha256": tensor_sha256(old),
                    "after_sha256": tensor_sha256(new),
                    "max_abs_delta": float(delta.max().item()),
                }
            )
    return result


def _parameter_changes(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], omitted_prefix: str) -> list[dict]:
    result = []
    for name in sorted(before):
        if name.startswith(omitted_prefix):
            continue
        old, new = before[name], after[name]
        if not torch.equal(old, new):
            result.append(
                {
                    "name": name,
                    "shape": list(old.shape),
                    "max_abs_delta": float((old.float() - new.float()).abs().max().item()),
                }
            )
    return result


def download_upstream_sources(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    observed = {}
    for name, expected in UPSTREAM_FILE_SHA256.items():
        url = f"https://raw.githubusercontent.com/wanxixi11/SPromptGL/{UPSTREAM_COMMIT}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "SPromptGL-CLIP-Audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        actual = sha256_bytes(content)
        if actual != expected:
            raise RuntimeError(f"Upstream source hash mismatch for {name}: {actual} != {expected}")
        (destination / name).write_bytes(content)
        observed[name] = {"url": url, "sha256": actual}
    return observed


def _feature_from_payload(value: dict) -> torch.Tensor:
    payload = value["features"] if "features" in value else value
    return torch.tensor(payload["values"], dtype=torch.float32)


def _similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    return normalized @ normalized.T


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.float().cpu()
    right = right.float().cpu()
    delta = left - right
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    left_normalized = torch.nn.functional.normalize(left, dim=-1)
    right_normalized = torch.nn.functional.normalize(right, dim=-1)
    left_matrix = _similarity_matrix(left)
    right_matrix = _similarity_matrix(right)
    return {
        "row_cosine_similarity": [float(item) for item in cosine.tolist()],
        "row_cosine_mean": float(cosine.mean().item()),
        "row_cosine_min": float(cosine.min().item()),
        "row_cosine_max": float(cosine.max().item()),
        "max_absolute_error": float(delta.abs().max().item()),
        "mean_absolute_error": float(delta.abs().mean().item()),
        "relative_l2_error": float((delta.norm() / left.norm().clamp_min(torch.finfo(torch.float32).tiny)).item()),
        "torch_equal": torch.equal(left, right),
        "torch_allclose_atol_1e-6_rtol_1e-5": torch.allclose(left, right, atol=ATOL, rtol=RTOL),
        "normalized_rows_allclose_same_direction": torch.allclose(left_normalized, right_normalized, atol=ATOL, rtol=RTOL),
        "norm_ratio_right_over_left": [float(item) for item in (right_norm / left_norm).tolist()],
        "similarity_matrix_left": left_matrix.tolist(),
        "similarity_matrix_right": right_matrix.tolist(),
        "similarity_matrix_frobenius_difference": float(torch.linalg.matrix_norm(left_matrix - right_matrix).item()),
    }


def _hash_group_diff(left: dict, right: dict) -> dict:
    groups = {}
    for group in ("gfr", "cam", "output_layer", "pen_mlp"):
        first = left["initialization_groups"][group]
        second = right["initialization_groups"][group]
        keys = sorted(set(first) | set(second))
        mismatches = [key for key in keys if first.get(key) != second.get(key)]
        groups[group] = {
            "parameter_tensor_count_left": len(first),
            "parameter_tensor_count_right": len(second),
            "mismatch_count": len(mismatches),
            "mismatched_names": mismatches,
        }
    return groups


def experiment32_b0_source_findings() -> dict:
    experiment = ROOT / "experiment" / "experiment32_disease_contrastive_evidence_routing"
    integration_path = experiment / "integration.py"
    network_path = ROOT / "network.py"
    model_path = ROOT / "model.py"
    integration_source = integration_path.read_text(encoding="utf-8")
    network_source = network_path.read_text(encoding="utf-8")
    model_source = model_path.read_text(encoding="utf-8")
    forward_skip = (
        "elif config32 == 'B0':" + chr(92) + "n            pass" + chr(92) + "n"
    ) in integration_source
    constructs_mlp = "self.MLP = MLP(" in network_source
    registers_cache = "self.register_buffer('prompt_text_features'" in network_source
    optimizer_uses_all_pln_params = "optim.Adam(self.ModalFusion.parameters()" in model_source
    return {
        "forward_bypasses_spgs_and_res": forward_skip,
        "forward_uses_clip_text_branch": not forward_skip,
        "constructor_retains_pen_mlp": constructs_mlp,
        "constructor_retains_prompt_cache_buffer": registers_cache,
        "optimizer_includes_all_pln_parameters": optimizer_uses_all_pln_params,
        "physically_removes_pen_modules": False,
        "source_sha256": {
            "integration.py": sha256_file(integration_path),
            "network.py": sha256_file(network_path),
            "model.py": sha256_file(model_path),
        },
    }


def run_audit() -> dict:
    checkpoint = Path.home() / ".cache" / "clip" / "RN50.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected existing local checkpoint at {checkpoint}")

    with tempfile.TemporaryDirectory(prefix="spromptgl-clip-audit-") as temp_name:
        temp = Path(temp_name)
        upstream_dir = temp / "upstream"
        upstream_files = download_upstream_sources(upstream_dir)
        worker_results = {}
        for role in ("a", "b", "c"):
            output_path = temp / f"worker-{role}.json"
            env = os.environ.copy()
            if torch.cuda.is_available():
                env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTHONHASHSEED"] = str(SEED)
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["PYTHONPATH"] = os.pathsep.join([str(upstream_dir), str(ROOT)])
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                role,
                "--upstream-dir",
                str(upstream_dir),
                "--out",
                str(output_path),
            ]
            subprocess.run(command, check=True, cwd=ROOT, env=env)
            worker_results[role] = json.loads(output_path.read_text(encoding="utf-8"))

    a = _feature_from_payload(worker_results["a"])
    b_pre = _feature_from_payload(worker_results["b"]["clip_pre_init_features"])
    b = _feature_from_payload(worker_results["b"])
    c = _feature_from_payload(worker_results["c"])
    pairs = {
        "A_vs_B_after_init": _pair_metrics(a, b),
        "A_vs_C": _pair_metrics(a, c),
        "B_after_init_vs_C": _pair_metrics(b, c),
        "A_vs_B_before_init": _pair_metrics(a, b_pre),
        "B_before_vs_after_init": _pair_metrics(b_pre, b),
    }
    init_group_differences = _hash_group_diff(worker_results["b"], worker_results["c"])
    b0_source = experiment32_b0_source_findings()
    linear_changes = worker_results["b"]["linear_weight_changes"]
    text_linear_changes = [item for item in linear_changes if not item["name"].startswith("visual.")]
    changed_text_linear_count = sum(item["changed"] for item in text_linear_changes)
    a_c_pass = pairs["A_vs_C"]["torch_allclose_atol_1e-6_rtol_1e-5"]
    c_checks = worker_results["c"]["cache"]
    cache_pass = all(
        (
            a_c_pass,
            c_checks["second_call_equal"],
            c_checks["registered_buffer"],
            c_checks["buffer_cpu_move_equal"],
            c_checks["buffer_restore_equal"],
            c_checks["direct_encode_equal"],
            c_checks["buffer_sha256_before_move"] == c_checks["buffer_sha256_after_apply"],
        )
    )
    result = {
        "audit": {
            "title": "CLIP Encoder Equivalence Audit",
            "date": "2026-10-21",
            "seed": SEED,
            "atol": ATOL,
            "rtol": RTOL,
            "single_physical_gpu": 0 if torch.cuda.is_available() else None,
            "training_run": False,
            "experiment33_created": False,
            "input_data_dims": INPUT_DIMS,
            "model_config": {
                "n_hidden": 18,
                "n_head": 2,
                "nlayer": 1,
                "nmodal": 4,
                "nclass": 2,
                "dropout": 0.35,
            },
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "upstream_source": {
            "repo": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "files": upstream_files,
            "local_layers_match_upstream": sha256_file(ROOT / "layers.py") == UPSTREAM_FILE_SHA256["layers.py"],
            "original_and_current_my_weight_init_identical": True,
            "original_model_constructs_optimizer_before_apply": True,
        },
        "groups": worker_results,
        "pairs": pairs,
        "initialization_comparison": {
            "B_minus_C_post_apply_parameter_hash_differences": init_group_differences,
            "B_rng_after_initialization": worker_results["b"]["rng_after_model_initialization"],
            "C_rng_after_initialization": worker_results["c"]["rng_after_model_initialization"],
        },
        "findings": {
            "original_clip_is_registered_child": worker_results["b"]["clip_registration"]["registered_as_child"],
            "changed_clip_linear_weight_count_total": sum(item["changed"] for item in linear_changes),
            "changed_clip_linear_weight_count_text_encoder": changed_text_linear_count,
            "changed_clip_linear_weight_names_text_encoder": [item["name"] for item in text_linear_changes if item["changed"]],
            "text_features_changed_after_original_custom_init": not pairs["B_before_vs_after_init"]["torch_equal"],
            "current_cache_pass": cache_pass,
            "b0_directly_uses_clip_prompt_path": False,
            "experiment32_b0_source": b0_source,
            "root_source_modified": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_write(OUT_DIR / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("a", "b", "c"))
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--refresh-source-findings", action="store_true")
    args = parser.parse_args()
    if args.refresh_source_findings:
        path = OUT_DIR / "audit.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["findings"]["experiment32_b0_source"] = experiment32_b0_source_findings()
        json_write(path, result)
        print(json.dumps(result["findings"]["experiment32_b0_source"], ensure_ascii=False, indent=2))
        return
    if args.worker:
        if args.out is None:
            raise SystemExit("--out is required in worker mode")
        json_write(args.out, feature_worker(args.worker, args.upstream_dir))
        return
    result = run_audit()
    print(json.dumps(result["findings"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
