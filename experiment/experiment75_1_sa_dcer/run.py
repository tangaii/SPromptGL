#!/usr/bin/env python3
"""Experiment 75.1 runner, importing the audited E74.4 implementation by absolute path."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT, RESULTS = EXP / ".runtime", EXP / "results"
TASKS, CKPTS, LATEST, LOGS = RT / "tasks", RT / "checkpoints", RT / "latest", RT / "logs"
BASE_DIR = Path(CFG["base"]["path"])
E61_DIR = Path(CFG["history"]["warmup_path"]).parents[1]
E60_DIR = Path("/model/spromptgl/experiment/experiment60_dss_pen")
TEXT_CACHE = E60_DIR / ".runtime/text_embeddings.pt"
E75_DIR = Path(CFG["sources"]["e75_path"])
BASE = None
TEXT_META = None
TEXT_EMBEDDINGS = None
GROUPS = ("G1-Anchor", "G2-Faithful", "G3-Anonymous")
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
BASE_GROUP = CFG["base"]["formal_group"]


def sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def sha_tensor(value) -> str:
    import torch
    digest = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode() + b"\0")
            digest.update(sha_tensor(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(sha_tensor(item).encode())
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch(path: Path, payload) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_base():
    global BASE
    if BASE is None:
        spec = importlib.util.spec_from_file_location("e744_locked_positive_dcr_runner_for_e751", BASE_DIR / "run.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import locked E74.4 runner {BASE_DIR / 'run.py'}")
        BASE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = BASE
        spec.loader.exec_module(BASE)
    return BASE


def load_module():
    spec = importlib.util.spec_from_file_location("sa_dcer_751_module", EXP / "sa_dcer.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot import SA-DCER module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def text_bank():
    global TEXT_META, TEXT_EMBEDDINGS
    import torch
    import torch.nn.functional as F
    if TEXT_EMBEDDINGS is not None:
        return TEXT_EMBEDDINGS
    cache = torch_load(TEXT_CACHE)
    prompt_list = cache.get("text_map", {}).get("original")
    if prompt_list != CFG["method"]["text_strings"]:
        raise RuntimeError("original SPromptGL-8 text strings/order differ from the frozen audited cache")
    embeddings = torch.as_tensor(cache["embeddings"]["original"], dtype=torch.float32).contiguous()
    if tuple(embeddings.shape) != (8, 1024) or not torch.isfinite(embeddings).all():
        raise RuntimeError(f"frozen text cache must be finite [8,1024], got {tuple(embeddings.shape)}")
    if not torch.allclose(embeddings.norm(dim=1), torch.tensor([12.7886, 11.6644, 13.0078, 10.7829, 12.8684, 12.7385, 12.9905, 13.7103]), atol=1e-3, rtol=0):
        raise RuntimeError("frozen text embedding norms do not match the audited cache")
    TEXT_META = {"cache_sha256": sha_file(TEXT_CACHE), "prompt_strings": prompt_list,
                 "raw_embedding_sha256": sha_tensor(embeddings), "raw_norms": embeddings.norm(dim=1).tolist()}
    TEXT_EMBEDDINGS = embeddings
    return embeddings


def fixed_fold_artifacts(fold: int):
    import torch
    bank = text_bank().clone()
    projection_generator = torch.Generator(device="cpu").manual_seed(751100 + int(fold))
    projection = torch.randn((1024, 128), generator=projection_generator, dtype=torch.float32) / math.sqrt(128.0)
    anon_generator = torch.Generator(device="cpu").manual_seed(751900 + int(fold))
    permutation = torch.randperm(1024, generator=anon_generator)
    signs = torch.randint(0, 2, (1024,), generator=anon_generator, dtype=torch.int64).mul_(2).sub_(1).float()
    anonymous = bank[:, permutation] * signs[None, :]
    normalized_real = torch.nn.functional.normalize(bank, dim=-1)
    normalized_anon = torch.nn.functional.normalize(anonymous, dim=-1)
    gram_error = float((normalized_real @ normalized_real.T - normalized_anon @ normalized_anon.T).abs().max())
    if gram_error > 2e-6 or not torch.allclose(bank.norm(dim=1), anonymous.norm(dim=1), atol=1e-5, rtol=0):
        raise RuntimeError(f"anonymous signed permutation did not preserve text geometry: {gram_error}")
    projected = torch.nn.functional.normalize(bank @ projection, dim=-1)
    cosine_before = normalized_real @ normalized_real.T
    cosine_after = projected @ projected.T
    max_geometry_error = float((cosine_before - cosine_after).abs().max())
    if not math.isfinite(max_geometry_error) or max_geometry_error > 0.5:
        raise RuntimeError(f"fixed random semantic projection excessively distorted text geometry ({max_geometry_error})")
    return bank, anonymous, projection, {
        "fold": fold, "projection_seed": 751100 + fold, "anonymous_seed": 751900 + fold,
        "projection_sha256": sha_tensor(projection), "anonymous_transform_sha256": sha_tensor((permutation, signs)),
        "anonymous_gram_max_abs_error": gram_error, "projection_cosine_max_abs_change": max_geometry_error,
        "projection_cosine_rmse": float((cosine_before - cosine_after).square().mean().sqrt()),
    }


def init_sa_dcer(bank, projection, fold: int, device):
    import torch
    cls = load_module().SADecer
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() and str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        seed = 752100 + int(fold)
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        module = cls(bank, projection).to(device)
    return module, seed


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def history_snapshot():
    base = load_base()
    hist = base.load_history()
    return hist, hist.load_e69().load_e61().load_base()


def base_source_identity():
    base = load_base()
    manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("completed_tasks") != 30 or manifest.get("independent_audits") != 30:
        raise RuntimeError("E74.4-G2 base is not 30/30 COMPLETE_AUDITED")
    for rel, expected in (("run.py", CFG["base"]["run_sha256"]),
                          ("soft_positive_dcr.py", CFG["base"]["model_sha256"]),
                          ("audit.py", CFG["base"]["audit_sha256"]),
                          ("config.json", CFG["base"]["config_sha256"]),
                          ("manifest.json", CFG["base"]["manifest_sha256"])):
        actual = sha_file(BASE_DIR / rel)
        if expected != "REPLACE_AT_PREFLIGHT" and actual != expected:
            raise RuntimeError(f"E74.4 locked source SHA mismatch {rel}: {actual}")
    source_sha = manifest.get("training_source_sha256")
    current_source_map = base.source_manifest()
    if source_sha != manifest.get("source_sha256") or source_sha != CFG["base"]["source_sha256"]:
        raise RuntimeError(f"E74.4 locked training source identity mismatch: {source_sha}")
    if current_source_map != manifest.get("protected_sources"):
        raise RuntimeError("E74.4 protected source map differs from the independently audited manifest")
    if manifest.get("checkpoint_audit") != "PASS 30/30":
        raise RuntimeError("E74.4 independent checkpoint audit is not PASS 30/30")
    return {"manifest_sha256": sha_file(BASE_DIR / "manifest.json"), "source_sha256": source_sha,
            "current_protected_source_map_sha256": base.sha_object(current_source_map),
            "run_sha256": sha_file(BASE_DIR / "run.py"), "soft_positive_dcr_sha256": sha_file(BASE_DIR / "soft_positive_dcr.py"),
            "audit_sha256": sha_file(BASE_DIR / "audit.py"), "config_sha256": sha_file(BASE_DIR / "config.json"),
            "base_status": manifest["status"], "tasks": manifest["completed_tasks"], "audits": manifest["independent_audits"]}


def source_inventory():
    base = load_base()
    e75_paths = [E75_DIR / x for x in ("run.py", "dcer_v2.py", "audit.py", "config.json", "manifest.json",
                                        "results/report.md", "results/mechanism.csv", "results/interventions.csv")]
    current = {
        "sa_dcer.py": sha_file(EXP / "sa_dcer.py"), "run.py": sha_file(EXP / "run.py"),
        "audit.py": sha_file(EXP / "audit.py"), "config.json": sha_file(EXP / "config.json"),
        "e74_4_source_manifest_sha256": base.source_sha(),
        "e74_4_manifest_sha256": sha_file(BASE_DIR / "manifest.json"),
        "e74_4_fold_metrics_sha256": sha_file(BASE_DIR / "results/fold_metrics.csv"),
        "e60_text_cache_sha256": sha_file(TEXT_CACHE),
    }
    for path in e75_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required E75 read-only reference missing: {path}")
        current[f"e75/{path.relative_to(E75_DIR)}"] = sha_file(path)
    return current


def source_digest():
    return sha_json(source_inventory())


def verified_prepare(fold: int):
    base = load_base()
    state, warmup, identity = base.prepare(fold)
    base_task_path = BASE_DIR / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json"
    base_task = json.loads(base_task_path.read_text(encoding="utf-8"))
    if base_task.get("status") != "COMPLETE" or base_task.get("identity") != identity:
        raise RuntimeError(f"fold {fold} E74.4-G2 task identity differs from E74.4 prepare(fold)")
    audit_rows = list(csv.DictReader((BASE_DIR / "results/checkpoint_audit.csv").open(encoding="utf-8")))
    audit = next((row for row in audit_rows if row["group"] == BASE_GROUP and int(row["fold"]) == fold), None)
    if not audit or audit.get("status") != "INDEPENDENT_AUDIT_PASS" or audit.get("checkpoint_sha256") != base_task.get("checkpoint_sha256"):
        raise RuntimeError(f"fold {fold} E74.4 independent checkpoint audit unavailable")
    base_checkpoint = BASE_DIR / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    if not base_checkpoint.is_file() or sha_file(base_checkpoint) != base_task.get("checkpoint_sha256"):
        raise RuntimeError(f"fold {fold} E74.4 best-checkpoint SHA mismatch")
    overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != overlap:
        raise RuntimeError(f"unexpected historical fold {fold} label overlap")
    if state["train_pairs_np"].shape != (4096, 2) or identity["train_pair_scope_pass"] is not True:
        raise RuntimeError(f"fold {fold} E74.4 relation-pair training scope/count mismatch")
    if fold == 2 and (int(state["labels_np"][376]) != 0 or int(state["labels_np"][406]) != 1):
        raise RuntimeError("Fold 2 historical rows 377/407 labels do not match ASD=0, TD=1")
    state["fold"] = fold
    state["group"] = "SA-DCER"
    state["rudag_identity"] = identity
    return state, warmup, identity, base_task


def hooked_forward(state, dcer, group, training=False, intervention="full"):
    import torch
    import torch.nn.functional as F
    base = load_base()
    module_api = load_module()
    dcer.train(training)
    captured = {"formal_outputlayer_calls": 0}
    output_layer = state["cam"].Outputlayer
    use_aux = training and group in ("G2-Faithful", "G3-Anonymous")

    def pre_hook(module, inputs):
        if len(inputs) != 2 or inputs[0].shape[-1] != 144:
            raise RuntimeError(f"E74.4 CAM OutputLayer boundary mismatch: {[tuple(x.shape) for x in inputs if torch.is_tensor(x)]}")
        h = inputs[0]
        result = dcer(h, group=group, intervention=intervention)
        captured["formal_outputlayer_calls"] += 1
        captured["h"] = h
        captured["dcer"] = result
        if use_aux:
            captured["teacher"] = module_api.signed_margin_teacher(module, h, state["train"])
        return (result["h_prime"], *inputs[1:])

    handle = output_layer.register_forward_pre_hook(pre_hook)
    try:
        out = base.forward(state, BASE_GROUP, training=training, collect=False)
    finally:
        handle.remove()
    if captured["formal_outputlayer_calls"] != 1:
        raise RuntimeError(f"CAM/OutputLayer formal path called {captured['formal_outputlayer_calls']} times")
    if use_aux:
        train_ids = state["train"]
        target = captured["teacher"]
        result = captured["dcer"]
        captured["l_dir"] = F.smooth_l1_loss(result["difference"].index_select(0, train_ids), target)
        captured["l_conf"] = F.smooth_l1_loss(result["confidence"].index_select(0, train_ids), target.abs())
        out["total"] = out["total"] + 0.02 * captured["l_dir"] + 0.01 * captured["l_conf"]
    else:
        captured["l_dir"] = out["total"].new_zeros(())
        captured["l_conf"] = out["total"].new_zeros(())
    captured["total"] = out["total"]
    return out, captured


def new_optimizer(dcer):
    import torch
    params = [p for p in dcer.parameters() if p.requires_grad]
    ids = [id(p) for p in params]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate SA-DCER optimizer parameter")
    return torch.optim.AdamW(params, lr=0.001, weight_decay=0.001)


def optimizer_sha(state, opt_dcer):
    return sha_tensor({"base": {k: state[k].state_dict() for k in ("opt_base", "opt_cam", "opt_added")},
                       "dcer": opt_dcer.state_dict()})


def rng_runtime():
    return history_snapshot()[1]


def base_bundle(state):
    return load_base().load_history().state_bundle(state)


def load_base_bundle(state, bundle):
    load_base().load_history().load_bundle(state, bundle)


def cpu_module_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def metrics_for(logp, state):
    base_rt = rng_runtime()
    return base_rt.classification_metrics(logp, state["test"], state["y"])


def check_finite(module):
    import torch
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())


def do_train_update(state, dcer, opt_dcer, group):
    base = load_base()
    base_rt = rng_runtime()
    base.load_history().zero_optimizers(state)
    opt_dcer.zero_grad(set_to_none=True)
    out, capture = hooked_forward(state, dcer, group, training=True)
    if not bool(torch_isfinite(out["total"])):
        raise FloatingPointError("nonfinite training loss")
    out["total"].backward()
    if not check_finite(dcer) or not check_finite(state["cam"]) or not check_finite(state["model"]):
        raise FloatingPointError("nonfinite training gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    opt_dcer.step()
    return float(out["total"].detach().cpu()), float(capture["l_dir"].detach().cpu()), float(capture["l_conf"].detach().cpu()), base_rt.rng_sha(base_rt.pack_rng())


def torch_isfinite(value):
    import torch
    return bool(torch.isfinite(value).all())


def preflight():
    import numpy as np
    import torch
    base = load_base()
    verified_base = base_source_identity()
    e75_manifest = json.loads((E75_DIR / "manifest.json").read_text(encoding="utf-8"))
    if e75_manifest.get("status") != "COMPLETE_AUDITED" or e75_manifest.get("completed_tasks") != 40 or e75_manifest.get("independent_audits") != 40:
        raise RuntimeError("E75 historical mechanism report is not 40/40 audited")
    e60_manifest = json.loads((E60_DIR / "manifest.json").read_text(encoding="utf-8"))
    if e60_manifest.get("status") != "COMPLETE_AUDITED" or e60_manifest.get("text_cache_sha256") != sha_file(TEXT_CACHE):
        raise RuntimeError("E60 frozen RN50 text cache failed SHA verification")
    before = source_inventory()
    identities, init_params, projection_meta = {}, {}, {}
    geometry_info = None
    device_name = None
    base_rt = rng_runtime()
    for fold in range(1, 11):
        state, warmup, identity, task = verified_prepare(fold)
        current_rng = base_rt.rng_sha(base_rt.pack_rng())
        if warmup.get("rng_sha") != current_rng:
            raise RuntimeError(f"fold {fold} did not restore exact E61 epoch-10 warmup RNG")
        expected = task["identity"]
        if expected != identity:
            raise RuntimeError(f"fold {fold} E74.4 initialization identity mismatch")
        if not np.isin(state["train_pairs_np"], state["train_np"]).all():
            raise RuntimeError(f"fold {fold} relation pairs include nontraining subjects")
        if tuple(torch.cat(state["parts"], dim=1).shape) != (871, 320):
            raise RuntimeError(f"fold {fold} E74.4 DARG input shape mismatch")
        bank, anonymous, projection, meta = fixed_fold_artifacts(fold)
        device = next(state["cam"].parameters()).device
        if device_name is None:
            device_name = str(device)
        rng_before_init = base_rt.rng_sha(base_rt.pack_rng())
        modules = {}
        params_by_group = {}
        for group, chosen_bank in (("G1-Anchor", bank), ("G2-Faithful", bank), ("G3-Anonymous", anonymous)):
            modules[group], seed = init_sa_dcer(chosen_bank, projection, fold, device)
            params_by_group[group] = sha_tensor({k: v.detach().cpu() for k, v in modules[group].named_parameters()})
        if base_rt.rng_sha(base_rt.pack_rng()) != rng_before_init:
            raise RuntimeError(f"fold {fold} private module initialization changed the base RNG")
        if len(set(params_by_group.values())) != 1:
            raise RuntimeError(f"fold {fold} SA-DCER trainable initialization differs across groups")
        init_params[str(fold)] = params_by_group
        identities[str(fold)] = {"identity": identity, "warmup_rng_sha": warmup["rng_sha"], "parameters": params_by_group}
        projection_meta[str(fold)] = meta

        if fold == 1:
            api = load_module()
            h = torch.randn((5, 144), device=device)
            tokens = h.reshape(5, 16, 9)
            ctx = api.SADecer.global_context(h)
            original = modules["G2-Faithful"].contrastive_scores(tokens, ctx)["difference"]
            changed_ctx = ctx.clone(); changed_ctx[:, 0] += 0.5
            changed = modules["G2-Faithful"].contrastive_scores(tokens, changed_ctx)["difference"]
            swapped = modules["G2-Faithful"].contrastive_scores(tokens, ctx, intervention="prompt_class_swap")["difference"]
            shuffled = modules["G2-Faithful"].contrastive_scores(tokens, ctx, intervention="prompt_within_class_shuffle")["difference"]
            if float((original - changed).abs().mean().detach()) <= 1e-8:
                raise RuntimeError("patient context does not alter signed disease evidence")
            if not torch.allclose(original, -swapped, atol=2e-6, rtol=1e-6):
                raise RuntimeError("swapping disease prompt groups did not reverse evidence direction")
            if not torch.allclose(original, shuffled, atol=2e-6, rtol=1e-6):
                raise RuntimeError("within-class prompt permutation changed LogMeanExp aggregation")
            if abs(float(modules["G2-Faithful"].gamma.item()) - 0.05) > 1e-7:
                raise RuntimeError("SA-DCER gamma did not initialize to 0.05")
            projected_real = torch.nn.functional.normalize(bank @ projection, dim=-1)
            projected_anon = torch.nn.functional.normalize(anonymous @ projection, dim=-1)
            geometry_info = {**meta, "patient_context_mean_abs_delta_change": float((original - changed).abs().mean()),
                             "prompt_class_swap_max_sum_error": float((original + swapped).abs().max()),
                             "within_class_shuffle_max_abs_error": float((original - shuffled).abs().max()),
                             "projected_anonymous_gram_error": float((projected_real @ projected_real.T - projected_anon @ projected_anon.T).abs().max())}

            rng_before_forward = base_rt.rng_sha(base_rt.pack_rng())
            with torch.no_grad():
                reference = base.forward(state, BASE_GROUP, training=False, collect=False)
                disabled, disabled_cap = hooked_forward(state, modules["G1-Anchor"], "G1-Anchor", training=False, intervention="dcer_off")
            if base_rt.rng_sha(base_rt.pack_rng()) != rng_before_forward:
                raise RuntimeError("evaluation parity check consumed base RNG")
            parity_error = float((reference["logp"] - disabled["logp"]).abs().max())
            if parity_error != 0.0 or not torch.equal(reference["fused"], disabled["fused"]):
                raise RuntimeError(f"dcer_off failed exact E74.4-G2 prediction parity: {parity_error}")
            if disabled_cap["formal_outputlayer_calls"] != 1 or tuple(disabled_cap["h"].shape) != (871, 144):
                raise RuntimeError("OutputLayer hook did not intercept exactly one 144-D CAM feature")
            enabled, enabled_cap = hooked_forward(state, modules["G2-Faithful"], "G2-Faithful", training=False)
            if not bool(torch.isfinite(enabled["logp"]).all()) or not bool(torch.isfinite(enabled_cap["dcer"]["residual"]).all()):
                raise RuntimeError("SA-DCER produced nonfinite inference values")
            residual_ratio = float(enabled_cap["dcer"]["residual"].norm() / enabled_cap["h"].norm().clamp_min(1e-12))
            if residual_ratio <= 0.0:
                raise RuntimeError("nonzero initial gamma did not produce a real residual")

            opt_dcer = new_optimizer(modules["G2-Faithful"])
            state_before_step = {k: v for k, v in state.items() if k.startswith("opt_")}
            out_train, cap_train = hooked_forward(state, modules["G2-Faithful"], "G2-Faithful", training=True)
            requested = [modules["G2-Faithful"].gamma_raw,
                         modules["G2-Faithful"].context_proj.weight,
                         modules["G2-Faithful"].key_proj.weight,
                         modules["G2-Faithful"].evidence_proj.weight]
            cls_grads = torch.autograd.grad(out_train["cls"], requested, retain_graph=True, allow_unused=True)
            if any(g is None or not torch.isfinite(g).all() or float(g.norm()) <= 0 for g in cls_grads):
                raise RuntimeError("classification gradient missing on gamma/context/key/evidence projection")
            out_train["total"].backward()
            if not check_finite(modules["G2-Faithful"]) or not check_finite(state["cam"]) or not check_finite(state["model"]):
                raise RuntimeError("preflight update gradients are nonfinite")
            for key in ("opt_base", "opt_cam", "opt_added"):
                state[key].step()
            opt_dcer.step()
            gamma_after = float(modules["G2-Faithful"].gamma.detach())
            if gamma_after == 0.05:
                raise RuntimeError("SA-DCER gamma received gradient but did not update")
            snapshot = {"base": copy.deepcopy(base_bundle(state)), "module": copy.deepcopy(cpu_module_state(modules["G2-Faithful"])),
                        "optimizer": copy.deepcopy(opt_dcer.state_dict()), "rng": copy.deepcopy(base_rt.pack_rng())}
            do_train_update(state, modules["G2-Faithful"], opt_dcer, "G2-Faithful")
            replay_a = {"base": copy.deepcopy(base_bundle(state)), "module": copy.deepcopy(cpu_module_state(modules["G2-Faithful"])),
                        "optimizer": copy.deepcopy(opt_dcer.state_dict()), "rng": copy.deepcopy(base_rt.pack_rng())}
            load_base_bundle(state, snapshot["base"])
            modules["G2-Faithful"].load_state_dict(snapshot["module"], strict=True)
            opt_dcer.load_state_dict(snapshot["optimizer"])
            base_rt.restore_rng(snapshot["rng"])
            do_train_update(state, modules["G2-Faithful"], opt_dcer, "G2-Faithful")
            replay_b = {"base": base_bundle(state), "module": cpu_module_state(modules["G2-Faithful"]),
                        "optimizer": opt_dcer.state_dict(), "rng": base_rt.pack_rng()}
            if sha_tensor(replay_a["base"]) != sha_tensor(replay_b["base"]):
                raise RuntimeError("base model/optimizer update did not replay after checkpoint restoration")
            if sha_tensor(replay_a["module"]) != sha_tensor(replay_b["module"]) or sha_tensor(replay_a["optimizer"]) != sha_tensor(replay_b["optimizer"]):
                raise RuntimeError("SA-DCER parameters/optimizer update did not replay after checkpoint restoration")
            if base_rt.rng_sha(replay_a["rng"]) != base_rt.rng_sha(replay_b["rng"]):
                raise RuntimeError("RNG update replay mismatch")

    after = source_inventory()
    if before != after:
        raise RuntimeError("protected/source inventory changed during preflight")
    return {
        "status": "PASS", "base": verified_base, "base_group": BASE_GROUP,
        "initialization": identities, "text_cache": TEXT_META, "projection": projection_meta,
        "fold1_geometry_and_path": geometry_info, "fold1_dcer_off_exact_logp_max_abs_error": 0.0,
        "formal_cam_and_outputlayer_calls": 1, "new_module_gradients": "classification gradients finite/nonzero; confidence/direction losses finite",
        "one_step_checkpoint_rng_optimizer_replay": True, "protected_source_inventory_sha256": sha_json(before),
        "protected_source_inventory": before, "device": device_name,
    }


def initialize_manifest(preflight_result):
    atomic_json(EXP / "manifest.json", {
        "experiment": CFG["experiment"], "status": "PREFLIGHT_PASS", "experiment_root": str(EXP),
        "preferred_root": CFG["preferred_root"], "root_fallback_reason": CFG["root_fallback_reason"],
        "expected_tasks": 30, "completed_tasks": 0, "independent_audits": 0,
        "checkpoint_audit": "NOT_RUN", "device": preflight_result["device"], "device_count": 1,
        "worker_limit": 10, "warmup_retrained": False, "base": preflight_result["base"],
        "preflight": preflight_result, "source_sha256": source_digest(),
        "training_execution": {"worker_limit": 10, "max_workers_observed": 0, "wall_seconds": 0.0,
                                "retry_count": 0, "retries": [], "failures": [], "oom_events": 0},
        "fold2_label_exposure_1based": [377, 407], "scientific_decision": "PENDING",
    })


def checkpoint_payload(group, state, dcer, opt_dcer, identity, base_task, source_hash, best, best_epoch, executed, trace, init_meta):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": state["fold"], "seed": 0,
        "config": CFG, "source_sha256": source_hash, "identity": identity,
        "base_task_checkpoint_sha256": base_task["checkpoint_sha256"],
        "base_task_identity": base_task["identity"], "base_best_checkpoint_used_for_training": False,
        "best_epoch_1based": best_epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "test_probability": best["probability"], "test_prediction": best["prediction"],
        "base_state": base_bundle_from_best(best), "sa_dcer_state": best["module_state"],
        "sa_dcer_optimizer": best["optimizer_state"], "losses": best["losses"],
        "init_meta": init_meta, "rng_trace": trace,
        "best_rng_after_train": best["rng_after_train_state"], "best_rng_after_eval": best["rng_after_eval_state"],
        "best_grad_snapshot": best["grad_snapshot"],
    }


def base_bundle_from_best(best):
    return best["base_state"]


def _task_complete(group, fold, source_hash):
    task_file, checkpoint = task_path(group, fold), best_path(group, fold)
    if not task_file.is_file() or not checkpoint.is_file():
        return False
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == source_hash and task.get("checkpoint_sha256") == sha_file(checkpoint)
    except Exception:
        return False


def train_worker(group: str, fold: int):
    import torch
    import numpy as np
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid training task {group}/{fold}")
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    source_hash = source_digest()
    if manifest.get("status") not in ("PREFLIGHT_PASS", "TRAINING") or manifest.get("source_sha256") != source_hash:
        raise RuntimeError("formal task source does not match the passing preflight")
    state, warmup, identity, base_task = verified_prepare(fold)
    base = load_base(); hist = base.load_history(); base_rt = rng_runtime()
    if base_rt.rng_sha(base_rt.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} warmup RNG differs before module initialization")
    bank, anonymous, projection, artifact_meta = fixed_fold_artifacts(fold)
    device = next(state["cam"].parameters()).device
    rng_before = base_rt.rng_sha(base_rt.pack_rng())
    dcer, init_seed = init_sa_dcer(anonymous if group == "G3-Anonymous" else bank, projection, fold, device)
    rng_after_module_init = base_rt.rng_sha(base_rt.pack_rng())
    if rng_after_module_init != rng_before:
        raise RuntimeError("new module initialization changed inherited RNG")
    opt_dcer = new_optimizer(dcer)
    init_meta = {"seed": init_seed, "parameter_sha256": sha_tensor({k: v.detach().cpu() for k, v in dcer.named_parameters()}),
                 "state_sha256": sha_tensor(cpu_module_state(dcer)), "projection": artifact_meta,
                 "base_model_cam_sha": identity["model_cam_sha"], "base_optimizer_sha": identity["optimizer_sha"],
                 "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": identity["warmup_rng_sha"],
                 "main_rng_sha_before_init": rng_before, "main_rng_sha_after_init": rng_after_module_init,
                 "main_rng_unchanged": rng_before == rng_after_module_init}
    task_file, best_file, latest_file = task_path(group, fold), best_path(group, fold), latest_path(group, fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == source_hash and existing.get("checkpoint_sha256") == sha_file(best_file):
            return existing
        if existing.get("status") == "FAILED":
            task_file.unlink()
        else:
            raise RuntimeError(f"refusing to overwrite incompatible task record: {task_file}")

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_hash), ("identity", identity)):
            if saved.get(key) != expected:
                raise RuntimeError(f"incompatible recovery checkpoint {key} for {group} fold {fold}")
        hist.load_bundle(state, saved["base_state"])
        dcer.load_state_dict(saved["sa_dcer_state"], strict=True)
        opt_dcer.load_state_dict(saved["sa_dcer_optimizer"])
        base_rt.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["best"], saved["trace"]
    elif best_file.exists():
        raise RuntimeError(f"best checkpoint exists without recoverable task/latest state: {best_file}")

    if not 10 <= start_epoch <= int(CFG["training"]["max_epochs"]):
        raise RuntimeError(f"invalid recovery epoch {start_epoch}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time(); executed = start_epoch
    for epoch in range(start_epoch, int(CFG["training"]["max_epochs"])):
        rng_before = base_rt.rng_sha(base_rt.pack_rng())
        hist.zero_optimizers(state); opt_dcer.zero_grad(set_to_none=True)
        out, cap = hooked_forward(state, dcer, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite loss at {group} fold={fold} epoch={epoch+1}")
        out["total"].backward()
        if not check_finite(dcer) or not check_finite(state["cam"]) or not check_finite(state["model"]):
            raise FloatingPointError(f"nonfinite gradient at {group} fold={fold} epoch={epoch+1}")
        grad_snapshot = {"gamma": float(dcer.gamma_raw.grad.detach().abs().cpu()) if dcer.gamma_raw.grad is not None else 0.0,
                         "context": float(dcer.context_proj.weight.grad.detach().norm().cpu()) if dcer.context_proj.weight.grad is not None else 0.0,
                         "key": float(dcer.key_proj.weight.grad.detach().norm().cpu()) if dcer.key_proj.weight.grad is not None else 0.0,
                         "evidence": float(dcer.evidence_proj.weight.grad.detach().norm().cpu()) if dcer.evidence_proj.weight.grad is not None else 0.0,
                         "confidence": float(dcer.confidence_net[0].weight.grad.detach().norm().cpu()) if dcer.confidence_net[0].weight.grad is not None else 0.0}
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        opt_dcer.step()
        rng_state_after_train = copy.deepcopy(base_rt.pack_rng())
        rng_after_train = base_rt.rng_sha(base_rt.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            evaluated, eval_capture = hooked_forward(state, dcer, group, training=False)
            metrics = base_rt.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = base_rt.rng_sha(base_rt.pack_rng())
        rng_after_eval_state = copy.deepcopy(base_rt.pack_rng())
        epoch_trace = {"epoch": executed, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_trace)
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best = {
                "metrics": {k: float(v) for k, v in metrics.items() if k not in ("pred", "prob")},
                "probability": np.asarray(metrics["prob"]).copy(), "prediction": np.asarray(metrics["pred"]).copy(),
                "base_state": copy.deepcopy(base_bundle(state)), "module_state": copy.deepcopy(cpu_module_state(dcer)),
                "optimizer_state": copy.deepcopy(opt_dcer.state_dict()),
                "losses": {"base_total": float((out["total"] - .02 * cap["l_dir"] - .01 * cap["l_conf"]).detach().cpu()),
                           "total": float(out["total"].detach().cpu()), "cls": float(out["cls"].detach().cpu()),
                           "graph": float(out["graph"].detach().cpu()), "pair": float(out["pair"].detach().cpu()),
                           "adv": float(out["adv"].detach().cpu()), "preserve": float(out["preserve"].detach().cpu()),
                           "l_dir": float(cap["l_dir"].detach().cpu()), "l_conf": float(cap["l_conf"].detach().cpu())},
                "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "rng_after_train_state": copy.deepcopy(rng_state_after_train), "rng_after_eval_state": rng_after_eval_state,
                "gamma": float(dcer.gamma.detach().cpu()), "residual_ratio": float(eval_capture["dcer"]["residual"].norm() / eval_capture["h"].norm().clamp_min(1e-12)),
                "grad_snapshot": grad_snapshot,
            }
            payload = checkpoint_payload(group, state, dcer, opt_dcer, identity, base_task, source_hash, best, best_epoch, executed, trace, init_meta)
            atomic_torch(best_file, payload)
        else:
            wait += 1

        if executed % 10 == 0 or executed == int(CFG["training"]["max_epochs"]) or wait > int(CFG["training"]["early_stop_wait_gt"]):
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": source_hash, "identity": identity,
                "base_state": base_bundle(state), "sa_dcer_state": cpu_module_state(dcer), "sa_dcer_optimizer": opt_dcer.state_dict(),
                "rng": base_rt.pack_rng(), "next_epoch": executed, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace, "init_meta": init_meta,
            })
            if executed % 10 == 0 or wait > int(CFG["training"]["early_stop_wait_gt"]):
                row = {"group": group, "fold": fold, "epoch": executed, "best_epoch_1based": best_epoch + 1,
                       "best_acc": best_acc, "metrics": {k: float(metrics[k]) for k in METRICS},
                       "loss": float(out["total"].detach().cpu()), "l_dir": float(cap["l_dir"].detach().cpu()),
                       "l_conf": float(cap["l_conf"].detach().cpu()), "gamma": float(dcer.gamma.detach().cpu()),
                       "residual_ratio": best["residual_ratio"], "gradients": grad_snapshot, "rng": epoch_trace,
                       "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
                with log_path(group, fold).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch+1} wait={wait}", flush=True)
        del out, evaluated, cap, eval_capture
        if wait > int(CFG["training"]["early_stop_wait_gt"]):
            break
    if best is None:
        raise RuntimeError(f"no test-ACC-selected checkpoint for {group} fold {fold}")
    payload = checkpoint_payload(group, state, dcer, opt_dcer, identity, base_task, source_hash, best, best_epoch, executed, trace, init_meta)
    atomic_torch(best_file, payload)
    task = {"status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": source_hash, "identity": identity, "checkpoint_sha256": sha_file(best_file),
            "base_task_checkpoint_sha256": base_task["checkpoint_sha256"], "init_meta": init_meta,
            "best_epoch_1based": best_epoch + 1, "epochs_executed": executed, "metrics": best["metrics"],
            "rng_trace": trace, "wall_seconds": time.time() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device),
            "losses": best["losses"], "best_grad_snapshot": best["grad_snapshot"]}
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch+1}", flush=True)
    return task


def training_workers():
    try:
        out = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=True).stdout
        return {"command": "nvidia-smi", "output": out[:4000]}
    except Exception as exc:
        return {"command": "nvidia-smi", "error": repr(exc)}


def run_training(max_workers: int = 10):
    if max_workers < 1 or max_workers > 10:
        raise ValueError("worker count must be within 1..10")
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_hash = source_digest()
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS" or manifest.get("source_sha256") != source_hash:
        raise RuntimeError("training requires a passing, source-matched preflight")
    tasks = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    pending = [(g, f) for g, f in tasks if not _task_complete(g, f, source_hash)]
    manifest["status"] = "TRAINING"; manifest["device_preflight"] = training_workers(); manifest["training_start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(manifest_path, manifest)
    attempts, failures, retries, active_count = {}, [], [], 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
        future_map = {pool.submit(train_worker, group, fold): (group, fold, 1) for group, fold in pending}
        active_count = min(max_workers, len(future_map))
        while future_map:
            done, _ = wait(tuple(future_map), return_when=FIRST_COMPLETED)
            for future in done:
                group, fold, attempt = future_map.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    record = {"group": group, "fold": fold, "attempt": attempt, "error": repr(exc), "traceback": "".join(traceback.format_exception(exc))[-5000:]}
                    failed = ("out of memory" in repr(exc).lower()) or ("outofmemory" in repr(exc).lower())
                    record["oom"] = failed
                    if attempt < 2:
                        retries.append(record)
                        future_map[pool.submit(train_worker, group, fold)] = (group, fold, attempt + 1)
                    else:
                        failures.append(record)
                manifest["completed_tasks"] = sum(_task_complete(g, f, source_hash) for g, f in tasks)
                manifest["training_execution"] = {"worker_limit": max_workers, "max_workers_observed": active_count,
                                                   "wall_seconds": time.time() - started, "retry_count": len(retries),
                                                   "retries": retries, "failures": failures,
                                                   "oom_events": sum(bool(x.get("oom")) for x in retries + failures)}
                atomic_json(manifest_path, manifest)
    completed = sum(_task_complete(g, f, source_hash) for g, f in tasks)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_tasks"] = completed
    manifest["training_execution"] = {"worker_limit": max_workers, "max_workers_observed": active_count,
                                       "wall_seconds": time.time() - started, "retry_count": len(retries),
                                       "retries": retries, "failures": failures,
                                       "oom_events": sum(bool(x.get("oom")) for x in retries + failures)}
    manifest["training_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["status"] = "TRAINED" if completed == 30 else "PARTIAL"
    atomic_json(manifest_path, manifest)
    if completed != 30:
        raise RuntimeError(f"formal training incomplete: {completed}/30")
    return manifest["training_execution"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LATEST, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if args.worker:
        train_worker(args.group, args.fold)
    elif args.preflight:
        result = preflight()
        atomic_json(RESULTS / "preflight.json", result)
        initialize_manifest(result)
        print("PREFLIGHT PASS", flush=True)
    elif args.train:
        run_training(args.workers)
    else:
        raise SystemExit("use --preflight, --train, or --worker")


if __name__ == "__main__":
    main()
