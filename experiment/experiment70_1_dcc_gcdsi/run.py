#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import MethodType

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = EXP / ".runtime/reference_snapshot/workspace"
HISTORY = EXP / ".runtime/reference_snapshot/history"
BASE_EXP = ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
E59_DIR = ROOT / "experiment/experiment59_shape_pen"
REF53 = HISTORY / "experiment53_3_cadr_graph_fallback_fix"
REF70 = HISTORY / "experiment/experiment70_gcdsi"
TEXT_CACHE = EXP / ".runtime/reference_snapshot/text/text_embeddings.pt"
RUNTIME = EXP / ".runtime"
RESULTS = EXP / "results"
CONFIG_PATH = EXP / "config.json"
GROUPS = ("E70.1-G1", "E70.1-G2", "E70.1-G3")
GROUP_SOURCE = {"E70.1-G1": "E70-G1", "E70.1-G2": "E70-G2", "E70.1-G3": "E70-G3"}
TEXTS_BP = [
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
]
TEXTS_SP = [
    "Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
    "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
]

sys.path.insert(0, str(BASE_EXP))
import run as BASE  # noqa: E402
sys.modules["run"] = BASE
sys.path.insert(0, str(E59_DIR))
_e59_spec = importlib.util.spec_from_file_location("e70_1_e59_runtime", E59_DIR / "run.py")
E59 = importlib.util.module_from_spec(_e59_spec)
sys.modules[_e59_spec.name] = E59
_e59_spec.loader.exec_module(E59)
from dcc_gcdsi import DCCGCDSI  # noqa: E402


def cfg() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        BASE.runtime()["torch"].save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_reference() -> tuple[dict, dict[int, dict], dict[int, dict]]:
    manifest_path = REF53 / "manifest_fallback_wd_sweep.json"
    metrics_path = REF53 / "results/fallback_wd_sweep/fold_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha_file(manifest_path) != "fae43596a9788d63bf8b0a0e52854fd1f7f014fea36c062bcf8a881806d7fcba":
        raise AssertionError("53.3 manifest SHA differs from Experiment 70's audited source")
    if sha_file(metrics_path) != "2849ce9d161d5b5ea32bc1b210ebde4f4bd0daf25e316298934b8029b0f8f9a2":
        raise AssertionError("53.3 metrics SHA differs from Experiment 70's audited source")
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("independent_checkpoint_audits_passed") != 40:
        raise AssertionError("53.3 WD sweep reference is not fully audited")
    name = "CADR_WD_0p01_GFR_WD_0p11"
    group = next(row for row in manifest["groups"] if row["name"] == name)
    if (float(group["cadr_weight_decay"]), float(group["gfr_weight_decay"])) != (0.01, 0.11):
        raise AssertionError("53.3 reference optimizer group mismatch")
    folds = {int(row["fold"]): row for row in manifest["folds"] if row["group"] == name}
    metrics = {int(row["fold"]): row for row in read_csv(metrics_path) if row["group"] == name}
    if set(folds) != set(range(1, 11)) or set(metrics) != set(range(1, 11)):
        raise AssertionError("53.3 reference fold records incomplete")
    group_dir = REF53 / ".runtime/fallback_wd_sweep" / name
    for fold in range(1, 11):
        record = folds[fold]
        stem = f"{name}_f{fold:02d}"
        checkpoint = group_dir / "checkpoints" / f"{stem}.best.pt"
        task = json.loads((group_dir / "tasks" / f"{stem}.json").read_text())
        digest = sha_file(checkpoint)
        if digest != record["checkpoint_sha256"] or task.get("best_sha256") != digest:
            raise AssertionError(f"53.3 fold {fold} checkpoint SHA mismatch")
        if task.get("status") != "COMPLETE" or task.get("readback_status") != "WORKER_READBACK_PASS":
            raise AssertionError(f"53.3 fold {fold} task/readback not complete")
        for key in ("split_sha", "pair_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
            if task.get(key) != record.get(key):
                raise AssertionError(f"53.3 fold {fold} {key} task mismatch")
    return manifest, folds, metrics


def text_banks() -> tuple[dict[str, object], dict]:
    torch = BASE.runtime()["torch"]
    F = torch.nn.functional
    cache = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    if sha_file(TEXT_CACHE) != cfg()["prompt_assets"]["text_cache_sha256"]:
        raise AssertionError("Frozen CLIP cache SHA mismatch")
    if cache["text_map"]["brainprompt"] != [TEXTS_BP[1], TEXTS_BP[0]]:
        raise AssertionError("BrainPrompt text or ASD/TD order does not match frozen Experiment 70")
    if cache["text_map"]["original"] != TEXTS_SP:
        raise AssertionError("SPromptGL eight-prompt text differs from frozen Experiment 70")
    bp = F.normalize(cache["embeddings"]["brainprompt"].float(), dim=-1)
    sp = F.normalize(cache["embeddings"]["original"].float(), dim=-1)
    sp_bank = torch.stack((F.normalize(sp[4:].mean(0), dim=0), F.normalize(sp[:4].mean(0), dim=0)))
    generator = torch.Generator(device="cpu").manual_seed(700070)
    unit_a = F.normalize(torch.randn(bp.shape[1], generator=generator), dim=0)
    unit_b = torch.randn(bp.shape[1], generator=generator)
    unit_b = F.normalize(unit_b - torch.dot(unit_a, unit_b) * unit_a, dim=0)
    cosine = torch.dot(bp[0], bp[1]).clamp(-0.999, 0.999)
    anon = torch.stack((bp[0].norm() * unit_a, bp[1].norm() * (cosine * unit_a + torch.sqrt(1 - cosine.square()) * unit_b)))
    banks = {"E70.1-G1": bp.contiguous(), "E70.1-G2": sp_bank.contiguous(), "E70.1-G3": anon.contiguous()}
    source_manifest = json.loads((REF70 / "manifest.json").read_text(encoding="utf-8"))
    expected = source_manifest["provenance"]["text_metadata"]["banks_sha256"]
    hashes = {group: tensor_sha(bank) for group, bank in banks.items()}
    for group, source_group in GROUP_SOURCE.items():
        if hashes[group] != expected[source_group] or hashes[group] != cfg()["groups"][GROUPS.index(group)]["bank_sha256"]:
            raise AssertionError(f"Frozen semantic bank SHA mismatch for {group}")
    cosine_anon = float(F.cosine_similarity(anon[0:1], anon[1:2]))
    cosine_real = float(F.cosine_similarity(bp[0:1], bp[1:2]))
    if abs(cosine_anon - cosine_real) > 1e-5:
        raise AssertionError("Anonymous semantic bank geometry does not match BrainPrompt")
    return banks, {
        "cache_sha256": sha_file(TEXT_CACHE),
        "encoder": "OpenAI CLIP RN50; frozen Experiment 60 cache",
        "embedding_width": int(bp.shape[1]),
        "brainprompt_text_sha256": source_manifest["provenance"]["prompt_text_sha256"]["brainprompt"],
        "spromptgl_text_sha256": source_manifest["provenance"]["prompt_text_sha256"]["spromptgl"],
        "banks_sha256": hashes,
        "brainprompt_texts_source_order": TEXTS_BP,
        "brainprompt_embedding_order": ["ASD", "TD"],
        "spromptgl_texts_source_order": TEXTS_SP,
        "spromptgl_aggregation": "normalize each; mean first four -> TD and last four -> ASD; normalize; reorder [ASD,TD]",
        "anonymous_seed": 700070,
        "anonymous_generator": "private CPU torch.Generator; no labels or cohort data",
        "g1_text_cosine": cosine_real,
        "g3_text_cosine": cosine_anon,
    }


def build_state(fold: int, ref_manifest: dict, ref_folds: dict[int, dict]):
    state = E59.build_base_state(fold, ref_manifest, ref_folds)
    if any(float(group["weight_decay"]) != 0.01 for group in state["opt_added"].param_groups):
        raise AssertionError("CADR AdamW must use the selected 53.3 WD=0.01 group")
    return state


def make_module(text_bank, fold: int):
    torch = BASE.runtime()["torch"]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(701000 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(701000 + fold)
        return DCCGCDSI(text_bank).to(BASE.runtime()["device"])


def install_output_hook(cam, module) -> None:
    output_layer = cam.Outputlayer
    original_forward = output_layer.forward
    cam._dcc_graph_features = None
    cam._dcc_active = False
    cam._dcc_diagnostics = None

    def forward_with_dcc(_self, x, attn_embedding):
        if cam._dcc_active:
            graph = cam._dcc_graph_features
            if graph is None:
                raise RuntimeError("DCC-GCDSI requires current GFR features before CAM classification")
            h = x.reshape(x.shape[0], 4, 36)
            h_prime, diagnostics = module(h, graph, disable_graph=bool(getattr(module, "_disable_graph", False)))
            cam._dcc_diagnostics = diagnostics
            x = h_prime.reshape(x.shape[0], 144)
        else:
            cam._dcc_diagnostics = None
        return original_forward(x, attn_embedding)

    output_layer.forward = MethodType(forward_with_dcc, output_layer)


def forward_state(state, module, active: bool, diagnostics: bool = True):
    torch = BASE.runtime()["torch"]
    F = torch.nn.functional
    if state.get("_is_training", False):
        state["model"].train()
        state["cam"].train()
    else:
        state["model"].eval()
        state["cam"].eval()
    fused, adjs, info = state["model"](state["parts"], pair_index=state["train_pairs"], use_cadr=True)
    state["cam"]._dcc_graph_features = fused
    state["cam"]._dcc_active = active
    logp, hidden, attention = state["cam"](fused)
    cls = BASE.runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = BASE._graph_loss(state["parts"], adjs)
    pair_logits = info["pair_logits"]
    pair = torch.stack([F.binary_cross_entropy_with_logits(value, state["train_same"]) for value in pair_logits]).mean()
    adv, adv_info = BASE.conditional_adversarial_loss(
        info["pair_reprs"], state["train_same"], state["model"].discriminator, state["model"].grl_scale
    )
    preserve, preserve_gaps = BASE.relation_preservation_loss(pair_logits, state["train_same"], margin=0.02)
    total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    return {
        "logp": logp, "hidden": hidden, "attention": attention, "fused": fused, "adjs": adjs,
        "info": info, "cls": cls, "graph": graph, "graph_parts": graph_parts, "pair": pair,
        "adv": adv, "adv_info": adv_info, "preserve": preserve, "preserve_gaps": preserve_gaps,
        "total": total, "dcc": getattr(state["cam"], "_dcc_diagnostics", None) if diagnostics else None,
    }


def parameter_schema_sha(module) -> str:
    return json_sha([(name, tuple(p.shape), str(p.dtype)) for name, p in module.named_parameters()])


def module_param_sha(module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode() + b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def base_optimizer_specs(state, added_optimizer):
    specs = []
    for name, optimizer, cls, lr, wd in (
        ("gfr", state["opt_base"], "Adam", 0.0038, 0.11),
        ("cam", state["opt_cam"], "Adam", 0.0038, 0.11),
        ("cad_r", state["opt_added"], "AdamW", 0.001, 0.01),
        ("dcc_gcdsi", added_optimizer, "AdamW", 0.001, 0.001),
    ):
        if optimizer.__class__.__name__ != cls:
            raise AssertionError(f"{name} optimizer class mismatch")
        for group in optimizer.param_groups:
            if float(group["lr"]) != lr or float(group["weight_decay"]) != wd:
                raise AssertionError(f"{name} optimizer LR/WD mismatch")
        specs.append({"name": name, "class": cls, "lr": lr, "weight_decay": wd})
    cam_ids = {id(parameter) for parameter in state["cam"].parameters()}
    dcc_ids = {id(parameter) for parameter in added_optimizer.param_groups[0]["params"]}
    if cam_ids & dcc_ids:
        raise AssertionError("CAM and DCC-GCDSI optimizers share parameters")
    if any(name.startswith("_dcc") or name.startswith("dcc") for name in state["cam"].state_dict()):
        raise AssertionError("DCC-GCDSI weights leaked into CAM state_dict")
    if len({id(p) for p in added_optimizer.param_groups[0]["params"]}) != sum(len(g["params"]) for g in added_optimizer.param_groups):
        raise AssertionError("DCC-GCDSI parameter is repeated in optimizer")
    return specs


def rms_ratio(h, residual, eps=1e-8):
    cam_rms = h.square().mean(-1).sqrt()
    residual_rms = residual.square().mean(-1).sqrt()
    return residual_rms / (cam_rms + eps)


def state_payload(state, module, optimizer_dcc, rng):
    return {
        "model": BASE._state_cpu(state["model"]), "cam": BASE._state_cpu(state["cam"]),
        "dcc": BASE._state_cpu(module), "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
        "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
        "optimizer_dcc": copy.deepcopy(optimizer_dcc.state_dict()), "rng": copy.deepcopy(rng),
    }


def hash_tree(value) -> str:
    torch = BASE.runtime()["torch"]
    np = BASE.runtime()["np"]
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0" + repr(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0" + str(array.dtype).encode() + b"\0" + repr(array.shape).encode() + b"\0")
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def capture_training_state(state, module, optimizer_dcc):
    return {
        "model": BASE._state_cpu(state["model"]), "cam": BASE._state_cpu(state["cam"]),
        "dcc": BASE._state_cpu(module), "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
        "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
        "optimizer_dcc": copy.deepcopy(optimizer_dcc.state_dict()), "rng": copy.deepcopy(BASE.pack_rng()),
    }


def restore_training_state(snapshot, state, module, optimizer_dcc):
    state["model"].load_state_dict(snapshot["model"], strict=True)
    state["cam"].load_state_dict(snapshot["cam"], strict=True)
    module.load_state_dict(snapshot["dcc"], strict=True)
    state["opt_base"].load_state_dict(snapshot["optimizer_base"])
    state["opt_cam"].load_state_dict(snapshot["optimizer_cam"])
    state["opt_added"].load_state_dict(snapshot["optimizer_added"])
    optimizer_dcc.load_state_dict(snapshot["optimizer_dcc"])
    BASE.restore_rng(snapshot["rng"])


def active_step(state, module, optimizer_dcc):
    torch = BASE.runtime()["torch"]
    state["_is_training"] = True
    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], optimizer_dcc):
        optimizer.zero_grad(set_to_none=True)
    output = forward_state(state, module, active=True)
    output["total"].backward()
    state["opt_base"].step()
    state["opt_cam"].step()
    state["opt_added"].step()
    optimizer_dcc.step()
    state["_is_training"] = False
    with torch.no_grad():
        evaluated = forward_state(state, module, active=True)
    return evaluated["logp"].detach().cpu()


def train_worker(group: str, fold: int) -> None:
    torch = BASE.runtime()["torch"]
    ref_manifest, ref_folds, _ = load_reference()
    banks, metadata = text_banks()
    source_group = GROUP_SOURCE[group]
    bank = banks[group].to(BASE.runtime()["device"])
    state = build_state(fold, ref_manifest, ref_folds)
    warmup_path = E59.warmup_cache_path(fold)
    warmup_sha = sha_file(warmup_path)
    warmup_meta = torch.load(warmup_path, map_location="cpu", weights_only=False)
    for key, expected in (("split_sha", state["split_sha"]), ("pair_sha", state["cache_meta"]["pair_sha"]),
                          ("initial_shared_sha", state["initial_shared_sha"]), ("initial_added_sha", state["initial_added_sha"]),
                          ("initial_rng_sha", state["construction_rng_sha"])):
        if warmup_meta.get(key) != expected:
            raise AssertionError(f"fold {fold} shared warmup {key} mismatch")
    E59.load_warmup_cache_into_state(state, fold, json.loads(E59.CONFIG_PATH.read_text()))
    rng_before_module = BASE.pack_rng()
    module = make_module(bank, fold)
    rng_after_module = BASE.pack_rng()
    if BASE.rng_sha(rng_before_module) != BASE.rng_sha(rng_after_module):
        raise AssertionError("DCC-GCDSI initialization polluted shared RNG")
    module_sha = module_param_sha(module)
    if module.lambda_raw.item() != 0.0:
        raise AssertionError("lambda_raw must initialize to zero")
    install_output_hook(state["cam"], module)
    optimizer_dcc = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)
    optimizer_specs = base_optimizer_specs(state, optimizer_dcc)

    directory = RUNTIME / group
    checkpoints = directory / "checkpoints"
    tasks = directory / "tasks"
    logs = directory / "logs"
    checkpoint_path = checkpoints / f"{group}_f{fold:02d}.best.pt"
    latest_path = checkpoints / f"{group}_f{fold:02d}.latest.pt"
    task_path = tasks / f"{group}_f{fold:02d}.json"
    log_path = logs / f"{group}_f{fold:02d}.log"
    for path in (checkpoints, tasks, logs):
        path.mkdir(parents=True, exist_ok=True)
    source_sha = source_sha256()
    config_sha = sha_file(CONFIG_PATH)
    if task_path.is_file() and checkpoint_path.is_file():
        task = json.loads(task_path.read_text())
        if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(checkpoint_path):
            return

    start, wait, best_acc, best_epoch = 10, 0, -1.0, -1
    best_state = best_metrics = best_mechanism = None
    if latest_path.is_file():
        saved = BASE.runtime()["torch"].load(latest_path, map_location="cpu", weights_only=False)
        signature = (saved.get("source_sha256"), saved.get("config_sha256"), saved.get("group"), saved.get("fold"), saved.get("warmup_sha256"), saved.get("bank_sha256"), saved.get("dcc_initial_sha256"))
        expected = (source_sha, config_sha, group, fold, warmup_sha, tensor_sha(banks[group]), module_sha)
        if signature != expected:
            raise AssertionError(f"latest checkpoint signature mismatch {group} fold {fold}")
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        state["opt_base"].load_state_dict(saved["optimizer_base"])
        state["opt_cam"].load_state_dict(saved["optimizer_cam"])
        state["opt_added"].load_state_dict(saved["optimizer_added"])
        module.load_state_dict(saved["dcc"], strict=True)
        optimizer_dcc.load_state_dict(saved["optimizer_dcc"])
        BASE.restore_rng(saved["rng"])
        start, wait, best_acc, best_epoch = saved["next_epoch"], saved["wait"], saved["best_acc"], saved["best_epoch"]
        best_state, best_metrics, best_mechanism = saved["best_state"], saved["best_metrics"], saved["best_mechanism"]

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"START group={group} fold={fold} epoch={start} warmup_sha={warmup_sha}\n")
        log.flush()
        for epoch in range(start, 450):
            state["_is_training"] = True
            for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], optimizer_dcc):
                optimizer.zero_grad(set_to_none=True)
            output = forward_state(state, module, active=True)
            if not torch.isfinite(output["total"]):
                raise FloatingPointError(f"nonfinite loss in {group} fold {fold} epoch {epoch}")
            output["total"].backward()
            if epoch == start:
                grad = module.lambda_raw.grad
                if grad is None or not torch.isfinite(grad) or float(grad.abs()) <= 0:
                    raise AssertionError(f"first active lambda gradient missing: {group} fold {fold}")
            for model in (state["model"], state["cam"], module):
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("non-finite gradient")
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_added"].step()
            optimizer_dcc.step()
            state["_is_training"] = False
            with torch.no_grad():
                evaluated = forward_state(state, module, active=True)
                metrics = BASE.classification_metrics(evaluated["logp"], state["test"], state["y"])
            diag = evaluated["dcc"]
            if epoch == start + 1:
                for name in ("query_h.weight", "query_f.weight", "w_delta.weight", "w_center.weight", "w_out.weight", "gate.0.weight", "gate.2.weight"):
                    parameter = dict(module.named_parameters())[name]
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.norm() == 0:
                        raise AssertionError(f"active parameter gradient missing for {name} ({group}, fold {fold})")
                if not any(p.grad is not None and p.grad.norm() > 0 for p in state["cam"].parameters()):
                    raise AssertionError("CAM did not receive active DCC-GCDSI gradient")
                if not any(p.grad is not None and p.grad.norm() > 0 for p in state["model"].parameters()):
                    raise AssertionError("GFR did not receive active DCC-GCDSI gradient")
            if metrics["acc"] > best_acc:
                best_acc = float(metrics["acc"])
                best_epoch = epoch
                wait = 0
                best_state = state_payload(state, module, optimizer_dcc, BASE.pack_rng())
                best_metrics = {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}
                best_mechanism = summarize_diagnostics(diag, output, module, state)
                checkpoint = {
                    "group": group, "fold": fold, "seed": 0, "best_epoch": best_epoch,
                    "epochs_executed": epoch + 1, "metrics": best_metrics, **best_state,
                    "bank_sha256": tensor_sha(banks[group]), "text_bank": banks[group].cpu(),
                    "warmup_sha256": warmup_sha, "source_sha256": source_sha,
                    "config_sha256": config_sha, "dcc_initial_sha256": module_sha,
                    "dcc_schema_sha256": parameter_schema_sha(module), "base_schema_sha256": state["initial_schema_sha"],
                    "reference_checkpoint_sha256": ref_folds[fold]["checkpoint_sha256"],
                    "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
                    "initial_shared_sha": state["initial_shared_sha"], "initial_added_sha": state["initial_added_sha"],
                    "initial_rng_sha": state["construction_rng_sha"], "optimizer_specs": optimizer_specs,
                    "initialization_rng_sha_unchanged": True, "diagnostics": best_mechanism,
                    "text_asset_sha256": metadata["cache_sha256"],
                }
                atomic_torch(checkpoint_path, checkpoint)
            else:
                wait += 1
            next_epoch = epoch + 1
            atomic_torch(latest_path, {
                "group": group, "fold": fold, "next_epoch": next_epoch, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best_state": best_state,
                "best_metrics": best_metrics, "best_mechanism": best_mechanism,
                "model": BASE._state_cpu(state["model"]), "cam": BASE._state_cpu(state["cam"]),
                "dcc": BASE._state_cpu(module), "optimizer_base": state["opt_base"].state_dict(),
                "optimizer_cam": state["opt_cam"].state_dict(), "optimizer_added": state["opt_added"].state_dict(),
                "optimizer_dcc": optimizer_dcc.state_dict(), "rng": BASE.pack_rng(),
                "source_sha256": source_sha, "config_sha256": config_sha, "warmup_sha256": warmup_sha,
                "bank_sha256": tensor_sha(banks[group]), "dcc_initial_sha256": module_sha,
            })
            if next_epoch % 10 == 0 or wait > 200:
                log.write(f"epoch={next_epoch} best_acc={best_acc:.8f} best_epoch={best_epoch} wait={wait}\n")
                log.flush()
            if wait > 200:
                break
    if best_state is None:
        raise RuntimeError(f"no best checkpoint for {group} fold {fold}")
    checkpoint_sha = sha_file(checkpoint_path)
    atomic_json(task_path, {
        "group": group, "fold": fold, "status": "COMPLETE", "best_epoch": best_epoch,
        "epochs_executed": epoch + 1, "metrics": best_metrics, "checkpoint_sha256": checkpoint_sha,
        "bank_sha256": tensor_sha(banks[group]), "warmup_sha256": warmup_sha,
        "source_sha256": source_sha, "config_sha256": config_sha,
        "reference_checkpoint_sha256": ref_folds[fold]["checkpoint_sha256"],
        "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
        "initial_shared_sha": state["initial_shared_sha"], "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"], "dcc_initial_sha256": module_sha,
    })
    latest_path.unlink(missing_ok=True)


def summarize_diagnostics(diagnostic, training_output, module, state):
    torch = BASE.runtime()["torch"]
    attention = diagnostic["attention"].detach()
    entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
    residual = diagnostic["residual"].detach()
    ratio = diagnostic["residual_ratio"].detach()
    gate = diagnostic["gate"].detach()
    signed = diagnostic["signed_strength"].detach()
    return {
        "lambda": float(diagnostic["lambda"].detach().cpu()),
        "text_cosine": float(diagnostic["text_cosine"].detach().cpu()),
        "delta_norm": float(diagnostic["delta_norm"].detach().cpu()),
        "signed_mean": float(signed.mean().cpu()),
        "signed_sd": float(signed.std(unbiased=False).cpu()),
        "signed_abs_mean": float(signed.abs().mean().cpu()),
        "signed_patient_sd": float(signed.mean(dim=1).std(unbiased=False).cpu()),
        "signed_modality_sd": float(signed.mean(dim=0).std(unbiased=False).cpu()),
        "attention_asd_mean": float(attention[..., 0].mean().cpu()),
        "attention_entropy_mean": float(entropy.mean().cpu()),
        "attention_entropy_sd": float(entropy.std(unbiased=False).cpu()),
        "attention_patient_sd": float(attention[..., 0].mean(dim=1).std(unbiased=False).cpu()),
        "attention_modality_sd": float(attention[..., 0].mean(dim=0).std(unbiased=False).cpu()),
        "gate_mean": float(gate.mean().cpu()),
        "gate_sd": float(gate.std(unbiased=False).cpu()),
        "gate_saturation_fraction": float(((gate < 0.01) | (gate > 0.99)).float().mean().cpu()),
        "residual_cam_rms_mean": float(ratio.mean().cpu()),
        "residual_cam_rms_p95": float(torch.quantile(ratio.flatten(), 0.95).cpu()),
        "residual_cam_rms_max": float(ratio.max().cpu()),
        "residual_bound_violations": int((ratio > 0.150001).sum().cpu()),
        "cam_rms_mean": float(diagnostic["cam_rms"].mean().cpu()),
        "residual_rms_mean": float(residual.square().mean(-1).sqrt().mean().cpu()),
        "w_f_grad_norm": float(module.query_f.weight.grad.detach().norm().cpu()) if module.query_f.weight.grad is not None else 0.0,
        "w_delta_grad_norm": float(module.w_delta.weight.grad.detach().norm().cpu()) if module.w_delta.weight.grad is not None else 0.0,
        "w_center_grad_norm": float(module.w_center.weight.grad.detach().norm().cpu()) if module.w_center.weight.grad is not None else 0.0,
        "w_out_grad_norm": float(module.w_out.weight.grad.detach().norm().cpu()) if module.w_out.weight.grad is not None else 0.0,
        "gate_grad_norm": float(module.gate[0].weight.grad.detach().norm().cpu()) if module.gate[0].weight.grad is not None else 0.0,
        "classification_loss": float(training_output["cls"].detach().cpu()),
        "total_loss": float(training_output["total"].detach().cpu()),
    }


def source_sha256() -> str:
    h = hashlib.sha256()
    for path in (EXP / "run.py", EXP / "dcc_gcdsi.py", CONFIG_PATH):
        h.update(path.name.encode() + b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def run_all(workers: int) -> None:
    tasks = [(group, fold) for group in GROUPS for fold in range(1, 11)]
    statuses = {group: RUNTIME / group / "tasks" for group in GROUPS}
    pending = []
    for group, fold in tasks:
        task = statuses[group] / f"{group}_f{fold:02d}.json"
        checkpoint = RUNTIME / group / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
        if task.is_file() and checkpoint.is_file():
            saved = json.loads(task.read_text())
            if saved.get("status") == "COMPLETE" and saved.get("checkpoint_sha256") == sha_file(checkpoint):
                continue
        pending.append((group, fold))
    failures = []
    max_workers = min(max(1, workers), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_subprocess, group, fold): (group, fold) for group, fold in pending}
        for future in as_completed(futures):
            group, fold = futures[future]
            try:
                result = future.result()
                if result.returncode != 0:
                    failures.append({"group": group, "fold": fold, "returncode": result.returncode, "stderr": result.stderr[-2000:]})
            except Exception as error:
                failures.append({"group": group, "fold": fold, "error": repr(error)})
    if failures:
        atomic_json(RUNTIME / "scheduler_failures.json", {"failures": failures})
        raise RuntimeError(f"{len(failures)} tasks failed; preserve completed tasks and rerun --all")
    completed = 0
    for group, fold in tasks:
        task = json.loads((statuses[group] / f"{group}_f{fold:02d}.json").read_text())
        cp = RUNTIME / group / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
        if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(cp):
            completed += 1
    if completed != 30:
        raise RuntimeError(f"scheduler reports only {completed}/30 complete folds")
    print(f"all tasks complete: {completed}/30", flush=True)


def _run_subprocess(group: str, fold: int):
    command = [sys.executable, str(EXP / "run.py"), "--worker", "--group", group, "--fold", str(fold)]
    return subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)


def protected_hashes() -> dict[str, str]:
    paths = [
        ROOT / "network.py", ROOT / "layers.py", ROOT / "utils.py", ROOT / "model.py",
        ROOT / "data/ABIDE/processed_standard_data.csv", ROOT / "data/ABIDE/modal_feat_dict.npy",
        ROOT / "graph/ABIDE_weighted-cosine_graph_.npz",
        BASE_EXP / "run.py", BASE_EXP / "cadr_gfr.py", BASE_EXP / "config.json",
        E59_DIR / "run.py", E59_DIR / "shape_pen.py", E59_DIR / "config.json",
        ROOT / "experiment/experiment51_dugsa/dugsa.py",
        ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/index.json",
    ]
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def preflight() -> dict:
    torch = BASE.runtime()["torch"]
    F = torch.nn.functional
    reference_manifest, reference_folds, reference_metrics = load_reference()
    banks, bank_meta = text_banks()
    folds = []
    init_hashes = {}
    for fold in range(1, 11):
        state = build_state(fold, reference_manifest, reference_folds)
        expected = reference_folds[fold]
        task_path = REF53 / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11/tasks" / f"CADR_WD_0p01_GFR_WD_0p11_f{fold:02d}.json"
        reference_task = json.loads(task_path.read_text(encoding="utf-8"))
        checks = {
            "split": state["split_sha"] == expected["split_sha"],
            "pair": state["cache_meta"]["pair_sha"] == expected["pair_sha"],
            "initial_shared": state["initial_shared_sha"] == expected["initial_shared_sha"],
            "initial_added": state["initial_added_sha"] == expected["initial_added_sha"],
            "initial_rng": state["construction_rng_sha"] == expected["initial_rng_sha"],
            "initial_gfr": state["initial_gfr_sha"] == reference_task["initial_gfr_sha"],
            "initial_cam": state["initial_cam_sha"] == reference_task["initial_cam_sha"],
            "initial_schema": state["initial_schema_sha"] == reference_task["initial_schema_sha"],
            "anchor": state["anchor_sha"] == reference_task["anchor_sha"],
        }
        optimizer_configs = {
            "gfr": (state["opt_base"], "Adam", 0.0038, 0.11),
            "cam": (state["opt_cam"], "Adam", 0.0038, 0.11),
            "cad_r": (state["opt_added"], "AdamW", 0.001, 0.01),
        }
        for name, (optimizer, optimizer_class, lr, wd) in optimizer_configs.items():
            checks[f"optimizer_{name}"] = optimizer.__class__.__name__ == optimizer_class and all(
                float(group["lr"]) == lr and float(group["weight_decay"]) == wd for group in optimizer.param_groups
            )
        warmup_path = E59.warmup_cache_path(fold)
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=False)
        checks["warmup_binding"] = all(warmup.get(key) == value for key, value in (
            ("split_sha", state["split_sha"]), ("pair_sha", state["cache_meta"]["pair_sha"]),
            ("initial_shared_sha", state["initial_shared_sha"]), ("initial_added_sha", state["initial_added_sha"]),
            ("initial_rng_sha", state["construction_rng_sha"]),
        ))
        E59.load_warmup_cache_into_state(state, fold, json.loads(E59.CONFIG_PATH.read_text()))
        checks["warmup_readback"] = E59.warmup_cache_path(fold).is_file()
        if not all(checks.values()):
            raise AssertionError(f"base/init/warmup parity failed fold {fold}: {checks}")
        modules = {}
        for group in GROUPS:
            before_rng = BASE.rng_sha(BASE.pack_rng())
            module = make_module(banks[group], fold)
            after_rng = BASE.rng_sha(BASE.pack_rng())
            if before_rng != after_rng:
                raise AssertionError("isolated DCC initialization changed main RNG")
            modules[group] = module
        hashes = {group: module_param_sha(module) for group, module in modules.items()}
        if len(set(hashes.values())) != 1:
            raise AssertionError(f"same-fold DCC initializations differ: {hashes}")
        init_hashes[str(fold)] = hashes[GROUPS[0]]
        folds.append({"fold": fold, **checks, "initialization_hashes": hashes, "warmup_sha256": sha_file(warmup_path)})
        del state, modules

    state = build_state(1, reference_manifest, reference_folds)
    text = banks[GROUPS[0]].to(BASE.runtime()["device"])
    module = make_module(text, 1)
    install_output_hook(state["cam"], module)
    state["_is_training"] = False
    ref = BASE.forward_state(state, training=False, use_cadr=True)
    state["cam"]._dcc_graph_features = state["model"](state["parts"], pair_index=state["train_pairs"], use_cadr=True)[0]
    state["cam"]._dcc_active = True
    with torch.no_grad():
        # The exact same CAM stochastic stream is used in eval mode; lambda=0 makes the classifier input identical.
        active_logp, _, _ = state["cam"](state["cam"]._dcc_graph_features)
    if not torch.equal(ref["logp"], active_logp):
        if not torch.allclose(ref["logp"], active_logp, atol=1e-7, rtol=1e-7):
            raise AssertionError("initial lambda=0 identity changed original CAM logits")

    torch.manual_seed(17)
    h = torch.randn(9, 4, 36, device=BASE.runtime()["device"])
    f = torch.randn(9, 320, device=BASE.runtime()["device"])
    module.eval()
    with torch.no_grad():
        original, dcc = module(h, f)
        swapped_bank = banks[GROUPS[0]][[1, 0]].to(h.device)
        swapped, swapped_diag = module(h, f, text_bank=swapped_bank)
        same_bank = torch.stack((banks[GROUPS[0]][0], banks[GROUPS[0]][0])).to(h.device)
        same, same_diag = module(h, f, text_bank=same_bank)
    math_checks = {
        "zero_lambda_identity": torch.equal(original, h),
        "attention_difference_tanh": torch.allclose(dcc["signed_strength"], torch.tanh(dcc["z"]), atol=2e-6, rtol=2e-6),
        "class_swap_center_equal": torch.allclose(dcc["center"], swapped_diag["center"], atol=1e-7, rtol=1e-7),
        "class_swap_strength_opposite": torch.allclose(dcc["signed_strength"], -swapped_diag["signed_strength"], atol=2e-6, rtol=2e-6),
        "class_swap_modulation_equal": torch.allclose(dcc["candidate"], swapped_diag["candidate"], atol=2e-6, rtol=2e-6),
        "class_swap_residual_opposite": torch.allclose(dcc["residual"], -swapped_diag["residual"], atol=2e-6, rtol=2e-6),
        "zero_disease_difference_no_injection": torch.equal(same, h) and torch.equal(same_diag["residual"], torch.zeros_like(same_diag["residual"])),
        "per_patient_modality_bound": float(dcc["residual_ratio"].max()) <= 0.150001,
    }
    if not all(math_checks.values()):
        raise AssertionError(f"DCC mathematical properties failed: {math_checks}")
    module.lambda_raw.data.fill_(0.1)
    with torch.no_grad():
        _, g1_semantics = module(h, f, text_bank=banks[GROUPS[0]].to(h.device))
        _, g2_semantics = module(h, f, text_bank=banks[GROUPS[1]].to(h.device))
    math_checks["different_frozen_banks_change_semantic_directions"] = bool(
        (g1_semantics["center"] - g2_semantics["center"]).norm() > 1e-4
        and (g1_semantics["delta"] - g2_semantics["delta"]).norm() > 1e-4
        and (g1_semantics["residual"] - g2_semantics["residual"]).norm() > 1e-6
    )
    if not math_checks["different_frozen_banks_change_semantic_directions"]:
        raise AssertionError("different text banks did not change semantic directions")

    # Exercise the actual CAM output path once, then verify the semantic module is not registered in CAM.
    state = build_state(1, reference_manifest, reference_folds)
    E59.load_warmup_cache_into_state(state, 1, json.loads(E59.CONFIG_PATH.read_text()))
    module = make_module(banks[GROUPS[0]].to(BASE.runtime()["device"]), 1)
    install_output_hook(state["cam"], module)
    cam_ids = {id(p) for p in state["cam"].parameters()}
    dcc_ids = {id(p) for p in module.parameters()}
    dcc_optimizer = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)
    opt_specs = base_optimizer_specs(state, dcc_optimizer)
    state["_is_training"] = True
    output = forward_state(state, module, active=True)
    output["total"].backward()
    first_lambda_grad = module.lambda_raw.grad.detach().clone()
    dcc_optimizer.step()
    state["opt_base"].zero_grad(set_to_none=True)
    state["opt_cam"].zero_grad(set_to_none=True)
    state["opt_added"].zero_grad(set_to_none=True)
    dcc_optimizer.zero_grad(set_to_none=True)
    output2 = forward_state(state, module, active=True)
    output2["total"].backward()
    gradient_checks = {name: dict(module.named_parameters())[name].grad is not None and bool(torch.isfinite(dict(module.named_parameters())[name].grad).all()) for name in (
        "query_h.weight", "query_f.weight", "w_delta.weight", "w_center.weight", "w_out.weight", "gate.0.weight", "gate.2.weight")}
    gradient_checks["first_active_lambda_nonzero"] = bool(torch.isfinite(first_lambda_grad) and first_lambda_grad.abs() > 0)
    gradient_checks["subsequent_lambda_nonzero"] = bool(module.lambda_raw.grad is not None and module.lambda_raw.grad.abs() > 0)
    gradient_checks["cam_finite_gradient"] = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in state["cam"].parameters())
    gradient_checks["gfr_finite_gradient"] = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in state["model"].parameters())
    if not all(gradient_checks.values()):
        raise AssertionError(f"active gradient preflight failed: {gradient_checks}")
    if cam_ids & dcc_ids or any("dcc" in key.lower() for key in state["cam"].state_dict()):
        raise AssertionError("DCC parameter boundary check failed")

    # Actual 53.3 stochastic warmup forward parity with the DCC hook installed but disabled.
    warmup_state_a = build_state(1, reference_manifest, reference_folds)
    warmup_audit = E59.load_warmup_cache_into_state(warmup_state_a, 1, json.loads(E59.CONFIG_PATH.read_text()))
    warmup_state_b = build_state(1, reference_manifest, reference_folds)
    E59.load_warmup_cache_into_state(warmup_state_b, 1, json.loads(E59.CONFIG_PATH.read_text()))
    warmup_module_a = make_module(banks[GROUPS[0]].to(BASE.runtime()["device"]), 1)
    warmup_module_b = make_module(banks[GROUPS[0]].to(BASE.runtime()["device"]), 1)
    install_output_hook(warmup_state_b["cam"], warmup_module_b)
    warmup_opt_a = torch.optim.AdamW(warmup_module_a.parameters(), lr=0.001, weight_decay=0.001)
    warmup_opt_b = torch.optim.AdamW(warmup_module_b.parameters(), lr=0.001, weight_decay=0.001)
    warmup_rng = torch.load(E59.warmup_cache_path(1), map_location="cpu", weights_only=False)["rng"]

    def warmup_two_steps(current, current_module, current_optimizer):
        trace = []
        current["cam"]._dcc_active = False
        for _ in range(2):
            current["_is_training"] = True
            for optimizer in (current["opt_base"], current["opt_cam"], current["opt_added"], current_optimizer):
                optimizer.zero_grad(set_to_none=True)
            current_out = forward_state(current, current_module, active=False)
            current_out["total"].backward()
            current["opt_base"].step(); current["opt_cam"].step(); current["opt_added"].step()
            current["_is_training"] = False
            with torch.no_grad():
                result = forward_state(current, current_module, active=False)
            trace.append(result["logp"].detach().cpu())
        return trace, capture_training_state(current, current_module, current_optimizer)

    BASE.restore_rng(warmup_rng)
    warmup_trace_a, warmup_final_a = warmup_two_steps(warmup_state_a, warmup_module_a, warmup_opt_a)
    BASE.restore_rng(warmup_rng)
    warmup_trace_b, warmup_final_b = warmup_two_steps(warmup_state_b, warmup_module_b, warmup_opt_b)
    warmup_hook_parity = all(torch.equal(a, b) for a, b in zip(warmup_trace_a, warmup_trace_b)) and hash_tree(warmup_final_a) == hash_tree(warmup_final_b)
    if not warmup_hook_parity:
        raise AssertionError("DCC disabled Hook changed actual stochastic CAM/base warmup trajectory")

    # Active fold-1 replay and resume test uses the real GFR/CAM/DCC training path.
    replay_state = build_state(1, reference_manifest, reference_folds)
    replay_warmup = E59.load_warmup_cache_into_state(replay_state, 1, json.loads(E59.CONFIG_PATH.read_text()))
    replay_module = make_module(banks[GROUPS[0]].to(BASE.runtime()["device"]), 1)
    install_output_hook(replay_state["cam"], replay_module)
    replay_optimizer = torch.optim.AdamW(replay_module.parameters(), lr=0.001, weight_decay=0.001)
    replay_start = capture_training_state(replay_state, replay_module, replay_optimizer)
    _ = active_step(replay_state, replay_module, replay_optimizer)
    replay_mid = capture_training_state(replay_state, replay_module, replay_optimizer)
    continuous_logits = active_step(replay_state, replay_module, replay_optimizer)
    replay_continuous = capture_training_state(replay_state, replay_module, replay_optimizer)
    restore_training_state(replay_start, replay_state, replay_module, replay_optimizer)
    _ = active_step(replay_state, replay_module, replay_optimizer)
    repeated_logits = active_step(replay_state, replay_module, replay_optimizer)
    replay_repeat = capture_training_state(replay_state, replay_module, replay_optimizer)
    restore_training_state(replay_mid, replay_state, replay_module, replay_optimizer)
    resumed_logits = active_step(replay_state, replay_module, replay_optimizer)
    replay_resumed = capture_training_state(replay_state, replay_module, replay_optimizer)
    replay_check = {
        "status": "PASS" if torch.equal(continuous_logits, repeated_logits) and torch.equal(continuous_logits, resumed_logits)
        and hash_tree(replay_continuous) == hash_tree(replay_repeat) == hash_tree(replay_resumed) else "FAIL",
        "fold": 1, "active_steps": 2, "warmup_sha256": replay_warmup["sha256"],
        "continuous_sha256": hash_tree(replay_continuous), "repeat_sha256": hash_tree(replay_repeat),
        "resume_sha256": hash_tree(replay_resumed),
        "logits_repeat_equal": bool(torch.equal(continuous_logits, repeated_logits)),
        "logits_resume_equal": bool(torch.equal(continuous_logits, resumed_logits)),
    }
    if replay_check["status"] != "PASS":
        raise AssertionError("DCC active repeat/resume parity failed")

    reference = {
        "base_group": "CADR_WD_0p01_GFR_WD_0p11",
        "manifest_sha256": sha_file(REF53 / "manifest_fallback_wd_sweep.json"),
        "metrics_sha256": sha_file(REF53 / "results/fallback_wd_sweep/fold_metrics.csv"),
        "reference_checkpoints": {str(fold): row["checkpoint_sha256"] for fold, row in reference_folds.items()},
    }
    result = {
        "status": "PASS", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": reference, "prompt_assets": bank_meta, "fold_checks": folds,
        "initialization_parity": "PASS_10_OF_10", "warmup_parity": "PASS_10_OF_10",
        "math_checks": math_checks, "gradient_checks": gradient_checks,
        "disabled_hook_warmup_two_step_parity": "PASS", "active_repeat_resume": replay_check,
        "optimizer_specs": opt_specs, "formal_tasks": 30,
        "protected_hashes": protected_hashes(), "source_sha256": source_sha256(),
        "config_sha256": sha_file(CONFIG_PATH),
    }
    atomic_json(RUNTIME / "preflight.json", result)
    return result


def torch_load(path: Path):
    return BASE.runtime()["torch"].load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.worker:
        if not args.group or not args.fold:
            parser.error("--worker needs --group and --fold")
        train_worker(args.group, args.fold)
    elif args.all:
        if not (RUNTIME / "preflight.json").is_file():
            raise RuntimeError("run --preflight successfully before formal training")
        status = json.loads((RUNTIME / "preflight.json").read_text())
        if status.get("status") != "PASS":
            raise RuntimeError("preflight status is not PASS")
        run_all(args.workers)
    else:
        parser.error("choose --preflight, --worker, or --all")


if __name__ == "__main__":
    main()
