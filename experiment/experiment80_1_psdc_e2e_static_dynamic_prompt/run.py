#!/usr/bin/env python3
"""Preflight, smoke-test, and train the three PSDC E80.1 groups."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch.nn import functional as F

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
ROOT = Path(CFG["sources"]["repository_root"])
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))

from psdc import PSDC, fold_dynamic_scaler

RT = EXP / ".runtime"
TASKS = RT / "tasks"
CHECKPOINTS = RT / "checkpoints"
LOGS = RT / "logs"
RESULTS = EXP / "results"
GROUPS = ("G1", "G2", "G3")
HISTORICAL_GROUP = "E74.4-G2-Positive-DCR"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HIST = None
DATA_TENSORS = None
PROMPTS = None
E51_MODULES_INSTALLED = False


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value) -> str:
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


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_history():
    global HIST
    if HIST is not None:
        return HIST
    path = Path(CFG["sources"]["e74_root"]) / "run.py"
    spec = importlib.util.spec_from_file_location("experiment80_1_readonly_e744_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    repository = Path(CFG["sources"]["repository_root"])
    experiments = repository / "experiment"
    path_overrides = {
        "BASE": experiments / "experiment69_3_g2_two_rows",
        "E741": experiments / "experiment74_1_rudag",
        "E69": experiments / "experiment69_3_g2_two_rows",
        "E62": experiments / "experiment62_e2e_gate_init_sensitivity",
        "E74": experiments / "experiment74_udag",
        "E742": experiments / "experiment74_2_dcr_rudag",
        "E743": experiments / "experiment74_3_local_dcr",
        "E53": experiments / "experiment53_3_cadr_graph_fallback_fix",
        "E61": experiments / "experiment61_cadr_postalign_vs_e2e",
        "REPO": repository,
    }
    for name, value in path_overrides.items():
        if hasattr(module, name):
            setattr(module, name, value)
    module.load_dcr().CADR_PATH = experiments / "experiment53_3_cadr_graph_fallback_fix" / "cadr_gfr.py"
    HIST = module
    return module


def load_historical_e61():
    """Load locked history using read-only /data artifacts through in-memory path overrides."""
    history = load_history()
    rudag = history.load_history()
    repo = Path(CFG["sources"]["repository_root"])
    rudag.BASE = repo / "experiment/experiment69_3_g2_two_rows"
    rudag.E53 = repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
    rudag.E61_EXP = repo / "experiment/experiment61_cadr_postalign_vs_e2e"
    rudag.E62 = repo / "experiment/experiment62_e2e_gate_init_sensitivity"
    rudag.E74 = repo / "experiment/experiment74_udag"
    if rudag.E69 is None:
        e69_path = repo / "experiment/experiment69_3_g2_two_rows/run.py"
        spec = importlib.util.spec_from_file_location("experiment80_1_readonly_e693_runner", e69_path)
        e69 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = e69
        spec.loader.exec_module(e69)
        e69.E61_EXP = repo / "experiment/experiment61_cadr_postalign_vs_e2e"
        e69.E61_RUN = e69.E61_EXP / "run.py"
        e69.E61_MODEL = e69.E61_EXP / "model_e61.py"
        e69.E61_CONFIG = e69.E61_EXP / "config.json"
        e69.E61_MANIFEST = e69.E61_EXP / "manifest.json"
        e69.E62_EXP = repo / "experiment/experiment62_e2e_gate_init_sensitivity"
        e69.E62_RUN = e69.E62_EXP / "run.py"
        e69.E62_CONFIG = e69.E62_EXP / "config.json"
        e69.E62_MANIFEST = e69.E62_EXP / "manifest.json"
        e69.E53_EXP = repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
        expected = rudag.CFG["base"]
        e69_base = Path(rudag.BASE)
        if e69.sha_file(e69_base / "run.py") != expected["run_sha256"] or e69.sha_file(e69_base / "manifest.json") != expected["manifest_sha256"]:
            raise RuntimeError("E69.3-G2 source/manifest SHA does not match the locked E74.1 reference")
        e69_manifest = json.loads((e69_base / "manifest.json").read_text(encoding="utf-8"))
        if (e69_manifest.get("status"), e69_manifest.get("completed_tasks"), e69_manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 10, 10):
            raise RuntimeError("E69.3-G2 historical reference is not fully audited")
        if e69_manifest.get("base", {}).get("group") != "E62-E2":
            raise RuntimeError("E69.3-G2 lineage does not point to E62-E2")
        e61 = e69.load_e61()
        e69.verify_e62_reference()
        e61.BASE_ROOT = repo
        e61.BASE_EXP = repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
        e61.BASE_RUN = e61.BASE_EXP / "run.py"
        e61.BASE_CADR = e61.BASE_EXP / "cadr_gfr.py"
        e61.BASE_RUNTIME = e61.BASE_EXP / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11"
        e61.ANCHORS = Path(CFG["sources"]["e43_anchor_vendor_root"]) / ".runtime/anchors"
        rudag.E69 = e69
    e69 = rudag.load_e69()
    e61 = e69.load_e61()
    repo = Path(CFG["sources"]["repository_root"])
    e61.BASE_ROOT = repo
    e61.BASE_EXP = repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
    e61.BASE_RUN = e61.BASE_EXP / "run.py"
    e61.BASE_CADR = e61.BASE_EXP / "cadr_gfr.py"
    e61.BASE_RUNTIME = e61.BASE_EXP / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11"
    e61.ANCHORS = Path(CFG["sources"]["e43_anchor_vendor_root"]) / ".runtime/anchors"
    install_e51_runtime_source()
    base_module = e61.load_base()
    base_module.ANCHORS = e61.ANCHORS
    base_module.LEGACY_EXP = RT / "historical_source/experiment/experiment51_dugsa"
    base_module.LEGACY_SOURCE = base_module.LEGACY_EXP / "source"
    e53_manifest = json.loads((e61.BASE_EXP / "manifest.json").read_text(encoding="utf-8"))
    expected_anchors = {Path(key).name: value for key, value in e53_manifest["anchor_sha256"].items()}
    for fold in range(1, 11):
        path = e61.ANCHORS / f"fold{fold:02d}.pt"
        if not path.is_file() or file_sha(path) != expected_anchors.get(path.name):
            raise RuntimeError(f"E43 anchor identity check failed for fold {fold}")
    return e61


def install_e51_runtime_source():
    """Expose the exact E51 source files whose historical OSSFS copy is absent."""
    global E51_MODULES_INSTALLED
    if E51_MODULES_INSTALLED:
        return
    import importlib

    e51 = RT / "historical_source/experiment/experiment51_dugsa"
    source = e51 / "source"
    manifest = json.loads((e51 / "manifest.json").read_text(encoding="utf-8"))
    expected = CFG["sources"]["e51_source_archive_sha256"]
    paths = {"dugsa.py": e51 / "dugsa.py", "network.py": source / "network.py",
             "layers.py": source / "layers.py", "utils.py": source / "utils.py",
             "config.json": e51 / "config.json"}
    if manifest.get("config_sha") != expected["config.json"]:
        raise RuntimeError("E51 archived manifest does not match the locked historical configuration")
    if manifest.get("code_sha", {}).get("dugsa.py") != expected["dugsa.py"]:
        raise RuntimeError("E51 archived manifest does not match the locked DUGSA source")
    if manifest.get("source_sha") != {key: expected[key] for key in ("network.py", "layers.py", "utils.py")}:
        raise RuntimeError("E51 archived manifest does not match the locked GFR source")
    for name, path in paths.items():
        if file_sha(path) != expected[name]:
            raise RuntimeError(f"E51 historical source SHA mismatch: {path}")
    for path in (str(source), str(e51)):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    for name in ("layers", "network", "utils", "dugsa"):
        existing = sys.modules.get(name)
        existing_path = Path(getattr(existing, "__file__", "")).resolve() if existing else None
        allowed = {p.resolve() for p in paths.values()}
        if existing is not None and existing_path not in allowed:
            del sys.modules[name]
    for name in ("layers", "network", "utils", "dugsa"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path not in {p.resolve() for p in paths.values()}:
            raise RuntimeError(f"imported historical module {name} from unexpected path {path}")
    E51_MODULES_INSTALLED = True


def load_data_tensors():
    global DATA_TENSORS
    if DATA_TENSORS is not None:
        return DATA_TENSORS
    summary_path = RESULTS / "dynamic_qc_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_path = Path(summary["cache"])
    if summary.get("status") != "PASS_90S_DATA" or summary.get("cache_sha256") != file_sha(cache_path):
        raise RuntimeError("90-second dynamic cache is missing or differs from its verified summary")
    payload = np.load(cache_path, allow_pickle=False)
    required = {"windows", "window_count", "row", "subject_id", "site", "tr_seconds", "timepoints"}
    if not required.issubset(payload.files):
        raise RuntimeError(f"dynamic cache keys are incomplete: {payload.files}")
    if payload["windows"].shape[0] != 871 or payload["windows"].shape[2] != 252:
        raise RuntimeError("90-second dynamic cache does not have 871 aligned 252-edge sequences")
    if not np.array_equal(payload["row"], np.arange(871)) or np.any(payload["window_count"] < 3):
        raise RuntimeError("dynamic cache row identity or minimum-window mask failed")
    DATA_TENSORS = {
        "windows": torch.from_numpy(payload["windows"].copy()).to(DEVICE),
        "counts": torch.from_numpy(payload["window_count"].astype(np.int64)).to(DEVICE),
        "rows": payload["row"].copy(),
        "subject_id": payload["subject_id"].copy(),
        "site": payload["site"].astype(str),
        "tr": payload["tr_seconds"].copy(),
        "timepoints": payload["timepoints"].copy(),
        "summary": summary,
    }
    return DATA_TENSORS


def prompt_cache_path():
    return RT / "prompt_cache.pt"


def load_prompt_embeddings():
    global PROMPTS
    if PROMPTS is not None:
        return PROMPTS
    cache = prompt_cache_path()
    texts = CFG["prompts"]
    text_sha = object_sha(texts)
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("text_sha256") != text_sha or saved.get("encoder") != "CLIP RN50":
            raise RuntimeError("cached frozen text embeddings do not match the predeclared prompt text")
        if tuple(saved["medical"].shape) != tuple(saved["anonymous"].shape) or saved["medical"].shape[0] != 8:
            raise RuntimeError("frozen prompt cache dimension mismatch")
        PROMPTS = saved
        return saved

    from CLIP.clip import clip

    model, _ = clip.load("RN50", DEVICE)
    model.eval()
    with torch.no_grad():
        all_text = texts["medical"] + texts["anonymous"]
        token_ids = clip.tokenize(all_text).to(DEVICE)
        features = F.normalize(model.encode_text(token_ids).float(), dim=-1).cpu()
    medical, anonymous = features[:8].contiguous(), features[8:].contiguous()
    payload = {
        "encoder": "CLIP RN50",
        "text_sha256": text_sha,
        "medical": medical,
        "anonymous": anonymous,
        "medical_embedding_sha256": object_sha(medical),
        "anonymous_embedding_sha256": object_sha(anonymous),
        "embedding_dimension": int(features.shape[1]),
        "norms_medical": medical.norm(dim=1).tolist(),
        "norms_anonymous": anonymous.norm(dim=1).tolist(),
        "token_lengths": token_ids.ne(0).sum(dim=1).cpu().tolist(),
        "medical_pairwise_cosine": (medical @ medical.T).numpy().tolist(),
        "anonymous_pairwise_cosine": (anonymous @ anonymous.T).numpy().tolist(),
    }
    temp = cache.with_name(cache.name + f".tmp.{os.getpid()}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, cache)
    PROMPTS = payload
    return payload


def source_manifest():
    repository = Path(CFG["sources"]["repository_root"])
    e74 = Path(CFG["sources"]["e74_root"])
    e61 = Path(CFG["sources"]["e61_root"])
    paths = [
        EXP / "config.json", EXP / "data_dynamic.py", EXP / "psdc.py", EXP / "run.py", EXP / "audit.py", EXP / "analyze.py",
        repository / "main.py", repository / "model.py", repository / "network.py",
        repository / "layers.py", repository / "utils.py",
        repository / "ABIDE-simple-2-concat-weighted-cosine.sh",
        repository / "data/ABIDE/processed_standard_data.csv",
        repository / "data/ABIDE/modal_feat_dict.npy",
        Path(CFG["sources"]["e11_root"]) / "recover_roi_mapping.py",
        Path(CFG["sources"]["e11_root"]) / "results" / "roi_256_mapping.csv",
        Path(CFG["sources"]["e11_root"]) / "results" / "subject_row_mapping.csv",
        Path(CFG["sources"]["e61_root"]) / "config.json",
        Path(CFG["sources"]["e74_root"]) / "run.py",
        Path(CFG["sources"]["e74_root"]) / "soft_positive_dcr.py",
        Path(CFG["sources"]["e74_root"]) / "manifest.json",
        e74 / "results" / "report.md", e74 / "results" / "fold_metrics.csv",
        Path(CFG["sources"]["e80_root"]) / "config.json",
        Path(CFG["sources"]["e80_root"]) / "extract_dynamic.py",
        Path(CFG["sources"]["e80_root"]) / "manifest.json",
        Path(CFG["sources"]["e80_root"]) / ".runtime" / "dynamic_cache" / "feature_cache.npz",
        Path(json.loads((RESULTS / "dynamic_qc_summary.json").read_text())["cache"]),
        prompt_cache_path(),
        Path(CFG["sources"]["static_csv"]),
        RT / "historical_source/experiment/experiment51_dugsa/manifest.json",
        RT / "historical_source/experiment/experiment51_dugsa/config.json",
        RT / "historical_source/experiment/experiment51_dugsa/dugsa.py",
        RT / "historical_source/experiment/experiment51_dugsa/source/network.py",
        RT / "historical_source/experiment/experiment51_dugsa/source/layers.py",
        RT / "historical_source/experiment/experiment51_dugsa/source/utils.py",
    ]
    paths.extend(sorted(e61.joinpath(".runtime/warmup").glob("fold*.pt")))
    paths.extend(sorted(e74.joinpath(".runtime/tasks").glob(f"{HISTORICAL_GROUP}_f*.json")))
    paths.extend(sorted(e74.joinpath(".runtime/checkpoints").glob(f"{HISTORICAL_GROUP}_f*.best.pt")))
    paths.extend(sorted((Path(CFG["sources"]["e43_anchor_vendor_root"]) / ".runtime/anchors").glob("fold*.pt")))
    manifest = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required provenance file missing: {path}")
        manifest[str(path)] = file_sha(path)
    return manifest


def source_sha():
    return object_sha(source_manifest())


def stable_seed(domain: str, fold: int) -> int:
    value = f"E80.1|seed={CFG['data']['seed']}|fold={fold}|{domain}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


class PrivateRNG:
    """Persistent CPU/CUDA random stream that restores the DARG stream after each call."""

    def __init__(self, seed: int):
        self.device_indices = list(range(torch.cuda.device_count())) if DEVICE.type == "cuda" else []
        cpu = torch.Generator(device="cpu").manual_seed(seed)
        self.cpu_state = cpu.get_state()
        self.cuda_states = []
        for index in self.device_indices:
            generator = torch.Generator(device=f"cuda:{index}").manual_seed(seed + 101 + index)
            self.cuda_states.append(generator.get_state())

    @contextlib.contextmanager
    def scope(self):
        shared_cpu = torch.get_rng_state()
        shared_cuda = torch.cuda.get_rng_state_all() if self.device_indices else []
        torch.set_rng_state(self.cpu_state)
        if self.device_indices:
            torch.cuda.set_rng_state_all(self.cuda_states)
        try:
            yield
        finally:
            self.cpu_state = torch.get_rng_state()
            if self.device_indices:
                self.cuda_states = torch.cuda.get_rng_state_all()
                torch.cuda.set_rng_state_all(shared_cuda)
            torch.set_rng_state(shared_cpu)

    def state_dict(self):
        return {"cpu": self.cpu_state.cpu(), "cuda": [x.cpu() for x in self.cuda_states], "devices": self.device_indices}

    def load_state_dict(self, state):
        if state["devices"] != self.device_indices or len(state["cuda"]) != len(self.cuda_states):
            raise RuntimeError("private CPU/CUDA RNG device inventory changed")
        self.cpu_state = state["cpu"].cpu().clone()
        self.cuda_states = [value.cpu().clone() for value in state["cuda"]]


def initialize_psdc(text: torch.Tensor, scaler, fold: int):
    mean_center, mean_scale, residual_scale = scaler
    devices = list(range(torch.cuda.device_count())) if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(stable_seed("parameter-init", fold))
        module = PSDC(text.to(DEVICE), mean_center, mean_scale, residual_scale,
                      residual_cap=CFG["model"]["relative_residual_cap"]).to(DEVICE)
    stream = PrivateRNG(stable_seed("dropout-stream", fold))
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=0.01)
    return module, optimizer, stream


def install_cam_adapter(state, module, rng_stream, group: str):
    """Patch the existing OutputLayer call once; retain its parameters and call order."""
    layer = state["cam"].Outputlayer
    original_forward = layer.forward
    state["e801_dynamic_off"] = False
    state["e801_prompt_indices"] = None
    state["e801_temporal_indices"] = None
    state["e801_last"] = None
    state["e801_output_layer_calls"] = 0
    surrogate = group == "G1"

    def adapted_forward(this, x, attn_embedding):
        state["e801_output_layer_calls"] += 1
        if state["e801_dynamic_off"]:
            delta = torch.zeros_like(x)
            diagnostics = {"relative_residual": x.new_zeros((x.shape[0],)), "gate": x.new_zeros((x.shape[0], 8, 64)),
                           "temporal_attention": x.new_zeros((x.shape[0], 8, int(DATA_TENSORS["windows"].shape[1]))),
                           "static_attention": x.new_zeros((x.shape[0], 8, 8)), "residual_limit_hit": x.new_zeros((x.shape[0],))}
        else:
            with rng_stream.scope():
                delta, diagnostics = module(
                    x, state["e801_windows"], state["e801_counts"], surrogate=surrogate,
                    temporal_indices=state["e801_temporal_indices"],
                    prompt_indices=state["e801_prompt_indices"],
                )
        state["e801_last"] = {"cam_input": x, "delta": delta, "diagnostics": diagnostics}
        return original_forward(x + delta, attn_embedding)

    layer.forward = types.MethodType(adapted_forward, layer)


def forward_psdc(state, module, group: str, training: bool, dynamic_off: bool = False):
    module.train(training)
    state["e801_dynamic_off"] = dynamic_off
    state["e801_output_layer_calls"] = 0
    history = load_history()
    output = history.forward(state, HISTORICAL_GROUP, training=training, intervention="full", collect=False)
    if not dynamic_off:
        current = state["e801_last"]
        diagnostics = current["diagnostics"]
        residual_by_row = (
            current["delta"].square().sum(dim=1) /
            current["cam_input"].square().sum(dim=1).clamp_min(1e-12)
        )
        train_rows = torch.as_tensor(state["train_np"], dtype=torch.long, device=DEVICE)
        residual_loss = residual_by_row.index_select(0, train_rows).mean()
        output["residual_loss"] = residual_loss
        output["total"] = output["total"] + 0.001 * residual_loss
        output["psdc"] = diagnostics
    else:
        output["residual_loss"] = output["total"].new_zeros(())
        output["psdc"] = state["e801_last"]["diagnostics"]
    return output


def build_task_state(group: str, fold: int):
    history = load_history()
    load_historical_e61()
    state, warmup, identity = history.prepare(fold)
    state["fold"] = fold
    state["group"] = group
    if tuple(state["parts"][0].shape) != (871, 48) or state["parts"][3].shape != (871, 256):
        raise RuntimeError("E74.4 historical static input shape changed")
    if tuple(state["train_pairs_np"].shape) != (4096, 2) or not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError("historical 4096 relation pairs are not confined to the fold training set")
    data = load_data_tensors()
    row_map = np.loadtxt(
        Path(CFG["sources"]["e11_root"]) / "results" / "subject_row_mapping.csv",
        delimiter=",", dtype=str, skiprows=1, usecols=(0, 2),
    )
    if not np.array_equal(row_map[:, 0].astype(int), data["rows"]) or not np.array_equal(row_map[:, 1].astype(int), data["subject_id"]):
        raise RuntimeError("dynamic row identity differs from the locked E11 CSV row/SUB_ID map")
    train_indices = torch.as_tensor(state["train_np"], dtype=torch.long, device=DEVICE)
    _, mean_center, mean_scale, residual_scale = fold_dynamic_scaler(data["windows"], data["counts"], train_indices)
    text_cache = load_prompt_embeddings()
    text = text_cache["medical"] if group == "G3" else text_cache["anonymous"]
    module, optimizer, private_rng = initialize_psdc(text, (mean_center, mean_scale, residual_scale), fold)
    state["e801_windows"] = data["windows"]
    state["e801_counts"] = data["counts"]
    scaler_sha = object_sha({"mean_center": mean_center, "mean_scale": mean_scale, "residual_scale": residual_scale,
                             "train_rows": state["train_np"]})
    install_cam_adapter(state, module, private_rng, group)
    return state, warmup, identity, module, optimizer, private_rng, scaler_sha, text_cache


def parameter_sha(module):
    return object_sha({name: value for name, value in sorted(module.named_parameters())})


def initial_group_identity(state, module, optimizer, identity, warmup, private_rng, scaler_sha, text_cache, group):
    history = load_history()
    base_runtime = load_historical_e61().load_base()
    return {
        "group": group,
        "fold": int(state["fold"]),
        "warmup_sha256": identity["warmup_sha256"],
        "warmup_rng_sha": warmup["rng_sha"],
        "initial_shared_rng_sha": base_runtime.rng_sha(base_runtime.pack_rng()),
        "darg_cam_initial_sha": identity["model_cam_sha"],
        "original_optimizer_sha": identity["optimizer_sha"],
        "split_sha": identity["split_sha"],
        "pair_sha": identity["pair_sha"],
        "effective_train_sha": identity["effective_train_sha"],
        "train_test_overlap_1based": identity["membership"]["train_test_overlap_1based"],
        "psdc_parameter_sha": parameter_sha(module),
        "psdc_optimizer_sha": object_sha(optimizer.state_dict()),
        "psdc_scaler_sha": scaler_sha,
        "prompt_text_sha": text_cache["medical_embedding_sha256"] if group == "G3" else text_cache["anonymous_embedding_sha256"],
        "psdc_parameter_count": module.trainable_parameter_count(),
        "darg_trainable_parameter_count": sum(p.numel() for p in state["model"].parameters()) + sum(p.numel() for p in state["cam"].parameters()),
    }


def metrics_for(logp, indices, labels):
    return load_historical_e61().load_base().classification_metrics(logp, indices, labels)


def safe_mechanism(output, indices):
    diagnostics = output["psdc"]
    index = torch.as_tensor(indices, dtype=torch.long, device=DEVICE)
    gate = diagnostics["gate"].index_select(0, index)
    rel = diagnostics["relative_residual"].index_select(0, index)
    limit = diagnostics["residual_limit_hit"].index_select(0, index)
    temporal_attn = diagnostics["temporal_attention"].index_select(0, index)
    return {
        "relative_residual_mean": float(rel.mean().detach().cpu()),
        "relative_residual_p95": float(torch.quantile(rel, 0.95).detach().cpu()),
        "relative_residual_max": float(rel.max().detach().cpu()),
        "gate_mean": float(gate.mean().detach().cpu()),
        "gate_std": float(gate.std(unbiased=False).detach().cpu()),
        "gate_saturation_fraction_lt_0.01_gt_0.99": float(((gate < 0.01) | (gate > 0.99)).float().mean().detach().cpu()),
        "residual_limit_fraction": float(limit.mean().detach().cpu()),
        "temporal_attention_entropy_mean": float((-(temporal_attn.clamp_min(1e-12) * temporal_attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().detach().cpu()),
        "residual_loss": float(output["residual_loss"].detach().cpu()),
    }


def optimizer_step(state, module, optimizer_psdc, group: str, training=True, dynamic_off=False, include_residual=True):
    history = load_history()
    history.load_history().zero_optimizers(state)
    optimizer_psdc.zero_grad(set_to_none=True)
    output = forward_psdc(state, module, group, training=training, dynamic_off=dynamic_off)
    if not include_residual and "psdc" in output:
        output["total"] = output["total"] - 0.001 * output["residual_loss"]
    if not torch.isfinite(output["total"]):
        raise FloatingPointError(f"non-finite loss at fold {state['fold']} group {group}")
    output["total"].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in state["model"].parameters()):
        raise FloatingPointError("non-finite DARG gradient")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in state["cam"].parameters()):
        raise FloatingPointError("non-finite CAM gradient")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
        raise FloatingPointError("non-finite PSDC gradient")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    if not dynamic_off:
        optimizer_psdc.step()
    return output


def torch_state_hash(state):
    return object_sha(state)


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group, fold, suffix):
    return CHECKPOINTS / f"{group}_f{fold:02d}.{suffix}.pt"


def save_checkpoint(path, value):
    history = load_history()
    history.atomic_torch(path, value)


def train_task(group: str, fold: int):
    source_hash = source_sha()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_hash:
        raise RuntimeError("formal worker source hashes do not match the passing preflight")
    state, warmup, identity, module, optimizer_psdc, private_rng, scaler_sha, text_cache = build_task_state(group, fold)
    base_runtime = load_historical_e61().load_base()
    history = load_history()
    identity_doc = initial_group_identity(state, module, optimizer_psdc, identity, warmup, private_rng, scaler_sha, text_cache, group)
    task_file = task_path(group, fold)
    best_file = checkpoint_path(group, fold, "best")
    latest_file = checkpoint_path(group, fold, "latest")
    log_file = LOGS / f"{group}_f{fold:02d}.jsonl"
    if task_file.exists():
        prior = json.loads(task_file.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETE" and prior.get("source_sha256") == source_hash and best_file.is_file():
            print(f"SKIP complete {group} fold={fold}", flush=True)
            return prior
        raise RuntimeError(f"refusing to overwrite incompatible existing task {task_file}")
    if base_runtime.rng_sha(base_runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError("worker did not restore the exact E61 epoch-10 shared RNG")

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    if latest_file.exists():
        saved = history.torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_hash),
                              ("identity", identity), ("identity_doc", identity_doc)):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest checkpoint provenance mismatch for {key}")
        load_historical_e61().load_base()
        history.load_history().load_bundle(state, saved["current"])
        module.load_state_dict(saved["current_psdc"], strict=True)
        optimizer_psdc.load_state_dict(saved["current_opt_psdc"])
        private_rng.load_state_dict(saved["private_rng"])
        base_runtime.restore_rng(saved["shared_rng"])
        start_epoch = int(saved["next_epoch"])
        wait = int(saved["wait"])
        best_acc = float(saved["best_acc"])
        best_epoch = int(saved["best_epoch"])
        best = saved["best"]
        trace = saved["trace"]
    elif best_file.exists():
        raise RuntimeError(f"best checkpoint exists without its matching latest checkpoint: {best_file}")

    if start_epoch < 10 or start_epoch > CFG["training"]["max_epochs"]:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")
    started = time.time()
    executed = start_epoch
    with log_file.open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, CFG["training"]["max_epochs"]):
            rng_before = base_runtime.rng_sha(base_runtime.pack_rng())
            output = optimizer_step(state, module, optimizer_psdc, group, training=True)
            gate_grads = {}
            for name in ("mean_projection", "residual_projection", "tcn1", "tcn2", "text_project", "gate", "output_projection"):
                grads = [p.grad.detach().float().square().sum() for p in getattr(module, name).parameters() if p.grad is not None]
                gate_grads[name] = float(torch.sqrt(torch.stack(grads).sum()).cpu()) if grads else 0.0
            rng_after_train = base_runtime.rng_sha(base_runtime.pack_rng())
            executed = epoch + 1
            with torch.no_grad():
                evaluated = forward_psdc(state, module, group, training=False)
                current_metrics = metrics_for(evaluated["logp"], state["test"], state["y"])
            rng_after_eval = base_runtime.rng_sha(base_runtime.pack_rng())
            epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                         "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
            trace.append(epoch_rng)
            test_metrics = {key: float(current_metrics[key]) for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")}
            if test_metrics["acc"] > best_acc:
                best_acc, best_epoch, wait = test_metrics["acc"], epoch, 0
                mechanism = safe_mechanism(evaluated, state["test_np"])
                best = {
                    "state": history.load_history().state_bundle(state),
                    "psdc": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                    "opt_psdc": optimizer_psdc.state_dict(),
                    "rng_shared": base_runtime.pack_rng(),
                    "rng_private": private_rng.state_dict(),
                    "metrics": test_metrics,
                    "probability": current_metrics["prob"].copy(),
                    "prediction": current_metrics["pred"].copy(),
                    "mechanism": mechanism,
                    "classification_gradient_norms": gate_grads,
                    "rng_after_train": rng_after_train,
                    "rng_after_eval": rng_after_eval,
                }
                save_checkpoint(best_file, {
                    "group": group, "fold": fold, "source_sha256": source_hash,
                    "identity": identity, "identity_doc": identity_doc,
                    "warmup_sha256": identity["warmup_sha256"], "best_epoch": best_epoch,
                    "best_epoch_1based": best_epoch + 1, "epochs_executed": executed,
                    "best_acc": best_acc, "best": best, "trace": trace,
                    "checkpoint_kind": "BEST", "prompt_text_sha": identity_doc["prompt_text_sha"],
                })
            else:
                wait += 1

            if executed % 10 == 0 or executed == CFG["training"]["max_epochs"] or wait > CFG["training"]["early_stop_wait_gt"]:
                save_checkpoint(latest_file, {
                    "group": group, "fold": fold, "source_sha256": source_hash,
                    "identity": identity, "identity_doc": identity_doc,
                    "warmup_sha256": identity["warmup_sha256"],
                    "current": history.load_history().state_bundle(state),
                    "current_psdc": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                    "current_opt_psdc": optimizer_psdc.state_dict(),
                    "shared_rng": base_runtime.pack_rng(), "private_rng": private_rng.state_dict(),
                    "next_epoch": executed, "wait": wait, "best_acc": best_acc,
                    "best_epoch": best_epoch, "best": best, "trace": trace,
                    "checkpoint_kind": "LATEST",
                })
            log.write(json.dumps({
                "group": group, "fold": fold, "epoch": executed,
                "best_epoch_1based": best_epoch + 1, "best_acc": best_acc,
                "metrics": test_metrics, "loss": float(output["total"].detach().cpu()),
                "cls": float(output["cls"].detach().cpu()), "graph": float(output["graph"].detach().cpu()),
                "pair": float(output["pair"].detach().cpu()), "adv": float(output["adv"].detach().cpu()),
                "preserve": float(output["preserve"].detach().cpu()),
                "residual_loss": float(output["residual_loss"].detach().cpu()),
                "classification_gradient_norms": gate_grads, "rng": epoch_rng,
                "mechanism": safe_mechanism(evaluated, state["test_np"]),
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0,
            }, allow_nan=False) + "\n")
            log.flush()
            if wait > CFG["training"]["early_stop_wait_gt"]:
                break
        elapsed = time.time() - started
    if best is None:
        raise RuntimeError(f"no best checkpoint generated for {group} fold={fold}")

    old_task_path = Path(CFG["sources"]["e74_root"]) / ".runtime" / "tasks" / f"{HISTORICAL_GROUP}_f{fold:02d}.json"
    historical_task = json.loads(old_task_path.read_text(encoding="utf-8"))
    old_rng = {int(item["epoch"]): item for item in historical_task["rng_trace"]}
    matched = [item for item in trace if item["epoch"] in old_rng]
    rng_mismatch = sum(any(item[key] != old_rng[item["epoch"]][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for item in matched)
    task = {
        "experiment": "Experiment 80.1", "group": group, "fold": fold, "status": "COMPLETE",
        "source_sha256": source_hash, "identity": identity, "identity_doc": identity_doc,
        "best_epoch_zero_based": best_epoch, "best_epoch_1based": best_epoch + 1,
        "epochs_executed": executed, "metrics": best["metrics"],
        "test_prediction": np.asarray(best["prediction"], dtype=int).tolist(),
        "test_probability": np.asarray(best["probability"], dtype=float).tolist(),
        "test_indices": np.asarray(state["test_np"], dtype=int).tolist(),
        "mechanism": best["mechanism"], "classification_gradient_norms": best["classification_gradient_norms"],
        "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
        "checkpoint_sha256": file_sha(best_file), "latest_sha256": file_sha(latest_file),
        "rng_trace": trace, "rng_parity": {"matching_epochs": len(matched), "exact_triplets": len(matched) - rng_mismatch,
                                              "mismatches": rng_mismatch, "historical_group": HISTORICAL_GROUP},
        "initial_identity": identity_doc, "parameter_count_psdc": module.trainable_parameter_count(),
        "wall_seconds": elapsed, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else 0,
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best_epoch={best_epoch + 1} acc={best_acc:.6f} rng_mismatch={rng_mismatch}", flush=True)
    return task


def _seed_scope(seed):
    devices = list(range(torch.cuda.device_count())) if DEVICE.type == "cuda" else []
    return torch.random.fork_rng(devices=devices), seed


def _reset_task_state(state, history, bundle, rng):
    history.load_history().load_bundle(state, bundle)
    base = load_historical_e61().load_base()
    base.restore_rng(rng)


def preflight():
    data = load_data_tensors()
    text_cache = load_prompt_embeddings()
    source_files = source_manifest()
    source_hash = object_sha(source_files)
    history = load_history()
    base_runtime = load_historical_e61().load_base()
    state, warmup, identity = history.prepare(1)
    if state["cam"].Outputlayer.__class__.__name__ != "OutputLayer":
        raise RuntimeError("E74.4 CAM Outputlayer is not the locked original OutputLayer module")
    initial = history.load_history().state_bundle(state)
    shared_rng = base_runtime.pack_rng()
    def detached_tree(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, list):
            return [detached_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detached_tree(item) for item in value)
        if isinstance(value, dict):
            return {key: detached_tree(item) for key, item in value.items()}
        return value
    with torch.no_grad():
        reference = history.forward(state, HISTORICAL_GROUP, training=False, collect=False)
    baseline_t1 = {key: detached_tree(reference[key]) for key in
                   ("fused", "logp", "hidden", "attention", "original_adjs", "unified_adjs")}
    _reset_task_state(state, history, initial, shared_rng)

    rng_init, seed_init = _seed_scope(stable_seed("preflight-init", 1))
    with rng_init:
        torch.manual_seed(seed_init)
        module = PSDC(text_cache["anonymous"].to(DEVICE), torch.zeros(252, device=DEVICE),
                      torch.ones(252, device=DEVICE), torch.ones(252, device=DEVICE)).to(DEVICE)
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=0.01)
    stream = PrivateRNG(stable_seed("preflight-dropout", 1))
    state["e801_windows"], state["e801_counts"] = data["windows"], data["counts"]
    install_cam_adapter(state, module, stream, "G2")
    cam_call_count = {"value": 0}
    original_cam_forward = state["cam"].forward

    def counted_cam_forward(this, *args, **kwargs):
        cam_call_count["value"] += 1
        return original_cam_forward(*args, **kwargs)

    state["cam"].forward = types.MethodType(counted_cam_forward, state["cam"])

    with torch.no_grad():
        adapted_zero = forward_psdc(state, module, "G2", training=False, dynamic_off=True)
    if cam_call_count["value"] != 1 or state["e801_output_layer_calls"] != 1:
        raise RuntimeError("T1 failed: CAM and original OutputLayer were not each called exactly once")
    t1_cam_call_count = cam_call_count["value"]
    t1_output_layer_call_count = state["e801_output_layer_calls"]
    for key in baseline_t1:
        if object_sha(baseline_t1[key]) != object_sha(adapted_zero[key]):
            raise RuntimeError(f"T1 failed: {key} differs with zero residual")

    # T2 compares every original DARG/CAM gradient at the same RNG state.
    _reset_task_state(state, history, initial, shared_rng)
    state["model"].train(); state["cam"].train()
    history.load_history().zero_optimizers(state)
    reference_train = history.forward(state, HISTORICAL_GROUP, training=True, collect=False)
    reference_train["total"].backward()
    base_grads = object_sha({
        "model": {n: None if p.grad is None else p.grad.detach() for n, p in state["model"].named_parameters()},
        "cam": {n: None if p.grad is None else p.grad.detach() for n, p in state["cam"].named_parameters()},
    })
    _reset_task_state(state, history, initial, shared_rng)
    history.load_history().zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    adapted_train = forward_psdc(state, module, "G2", training=True, dynamic_off=True)
    adapted_train["total"].backward()
    zero_grads = object_sha({
        "model": {n: None if p.grad is None else p.grad.detach() for n, p in state["model"].named_parameters()},
        "cam": {n: None if p.grad is None else p.grad.detach() for n, p in state["cam"].named_parameters()},
    })
    if base_grads != zero_grads:
        raise RuntimeError("T2 failed: original DARG/CAM gradients changed with zero residual")

    # T3 replays one inherited optimizer update and shared RNG trace bit-for-bit.
    _reset_task_state(state, history, initial, shared_rng)
    history.load_history().zero_optimizers(state)
    original_step = history.forward(state, HISTORICAL_GROUP, training=True, collect=False)
    original_step["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    original_after = history.load_history().state_bundle(state)
    original_rng_after = base_runtime.rng_sha(base_runtime.pack_rng())
    _reset_task_state(state, history, initial, shared_rng)
    history.load_history().zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    adapted_step = forward_psdc(state, module, "G2", training=True, dynamic_off=True)
    adapted_step["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    adapted_after = history.load_history().state_bundle(state)
    adapted_rng_after = base_runtime.rng_sha(base_runtime.pack_rng())
    if object_sha(original_after) != object_sha(adapted_after) or original_rng_after != adapted_rng_after:
        raise RuntimeError("T3 failed: zero-residual one-step update or shared RNG differs")

    # T4 verifies classification reaches every required trainable PSDC component while shared RNG is isolated.
    _reset_task_state(state, history, initial, shared_rng)
    fresh_state, fresh_warmup, fresh_identity = history.prepare(1)
    fresh_state["fold"] = 1
    fresh_state["e801_windows"], fresh_state["e801_counts"] = data["windows"], data["counts"]
    train_ids = torch.as_tensor(fresh_state["train_np"], dtype=torch.long, device=DEVICE)
    _, mc, ms, rs = fold_dynamic_scaler(data["windows"], data["counts"], train_ids)
    fresh_module, fresh_optimizer, fresh_stream = initialize_psdc(text_cache["anonymous"], (mc, ms, rs), 1)
    install_cam_adapter(fresh_state, fresh_module, fresh_stream, "G2")
    fresh_initial = history.load_history().state_bundle(fresh_state)
    fresh_psdc_initial = {k: v.detach().cpu().clone() for k, v in fresh_module.state_dict().items()}
    fresh_private_initial = fresh_stream.state_dict()
    shared_before = base_runtime.rng_sha(base_runtime.pack_rng())
    fresh_optimizer.zero_grad(set_to_none=True)
    cls_only = forward_psdc(fresh_state, fresh_module, "G2", training=True)["cls"]
    cls_only.backward()
    grad_norms = {}
    for name in ("mean_projection", "residual_projection", "tcn1", "tcn2", "text_project", "gate", "output_projection"):
        values = [p.grad.detach().float().square().sum() for p in getattr(fresh_module, name).parameters() if p.grad is not None]
        grad_norms[name] = float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0
    shared_after = base_runtime.rng_sha(base_runtime.pack_rng())
    if any((not np.isfinite(v)) or v <= 0 for v in grad_norms.values()):
        raise RuntimeError(f"T4 failed: missing finite nonzero classification gradient: {grad_norms}")

    # T5/T6 use an identical deterministic parameter initialization and equal module capacity.
    initials = {}
    modules = {}
    for group in GROUPS:
        seed, seed_value = _seed_scope(stable_seed("fairness-init", 1))
        with seed:
            torch.manual_seed(seed_value)
            pm = PSDC(text_cache["medical" if group == "G3" else "anonymous"].to(DEVICE), mc, ms, rs).to(DEVICE)
        opt = torch.optim.AdamW(pm.parameters(), lr=5e-4, weight_decay=0.01)
        initials[group] = {"params": parameter_sha(pm), "optimizer": object_sha(opt.state_dict()),
                           "count": pm.trainable_parameter_count(), "text": object_sha(pm.text_embeddings)}
        modules[group] = pm
    if len({initials[g]["params"] for g in GROUPS}) != 1 or len({initials[g]["optimizer"] for g in GROUPS}) != 1 or len({initials[g]["count"] for g in GROUPS}) != 1:
        raise RuntimeError("T5/T6 failed: G1/G2/G3 trainable architecture or initialization differs")
    if initials["G1"]["text"] != initials["G2"]["text"] or initials["G2"]["text"] == initials["G3"]["text"]:
        raise RuntimeError("T5/T6 failed: anonymous and medical text conditions are not isolated as specified")
    with torch.no_grad():
        q2 = modules["G2"].text_project(modules["G2"].text_embeddings)
        q3 = modules["G3"].text_project(modules["G3"].text_embeddings)
        probe_cam = torch.zeros((871, 144), device=DEVICE)
        g1_delta, g1_diag = modules["G1"](probe_cam, data["windows"], data["counts"], surrogate=True)
        g2_delta, g2_diag = modules["G2"](probe_cam, data["windows"], data["counts"], surrogate=False)
    if torch.equal(q2, q3) or torch.count_nonzero(g1_diag["dynamic_evidence"]).item() == 0:
        raise RuntimeError("T5/T6 failed: text queries differ ineffectively or G1 static-mean path collapsed")
    if torch.equal(g1_delta, g2_delta):
        raise RuntimeError("T5 failed: G1 static surrogate and G2 true dynamic sequence are not distinct inputs")

    # T7 perturbs external padding; valid temporal outputs and prediction must stay identical.
    test_module = modules["G2"].eval()
    windows = data["windows"]
    altered = windows.clone()
    for i, count in enumerate(data["counts"].tolist()):
        if count < windows.shape[1]:
            altered[i, count:] = 1e4
    fake_cam = torch.zeros((871, 144), device=DEVICE)
    with torch.no_grad():
        pred_a, diag_a = test_module(fake_cam, windows, data["counts"])
        pred_b, diag_b = test_module(fake_cam, altered, data["counts"])
    if not torch.equal(pred_a, pred_b) or not torch.equal(diag_a["dynamic_evidence"], diag_b["dynamic_evidence"]):
        raise RuntimeError("T7 failed: invalid padding perturbation changed valid temporal output")

    # T8 validates the unchanged historical pair/split identities and exact patient row alignment.
    if fresh_identity["pair_sha"] != identity["pair_sha"] or fresh_identity["split_sha"] != identity["split_sha"]:
        raise RuntimeError("T8 failed: repeated E61 warmup preparation changed split or relation-pair identity")
    if not np.array_equal(data["subject_id"], load_data_tensors()["subject_id"]):
        raise RuntimeError("T8 failed: dynamic patient rows are not aligned")
    fold2, _, fold2_identity = history.prepare(2)
    if fold2_identity["membership"]["train_test_overlap_1based"] != [377, 407] or fold2["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError("T8 failed: historical Fold 2 overlap/relation-pair identity changed")

    # One dynamic-on step and an original step must consume identical shared randomness.
    _reset_task_state(fresh_state, history, fresh_initial, shared_rng)
    fresh_module.load_state_dict(fresh_psdc_initial, strict=True)
    fresh_stream.load_state_dict(fresh_private_initial)
    base_runtime.restore_rng(shared_rng)
    history.load_history().zero_optimizers(fresh_state)
    fresh_optimizer.zero_grad(set_to_none=True)
    rng_pre_dynamic = base_runtime.rng_sha(base_runtime.pack_rng())
    dynamic_step = forward_psdc(fresh_state, fresh_module, "G2", training=True)
    dynamic_step["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        fresh_state[key].step()
    fresh_optimizer.step()
    rng_post_dynamic_train = base_runtime.rng_sha(base_runtime.pack_rng())
    with torch.no_grad():
        forward_psdc(fresh_state, fresh_module, "G2", training=False)
    rng_post_dynamic_eval = base_runtime.rng_sha(base_runtime.pack_rng())
    _reset_task_state(fresh_state, history, fresh_initial, shared_rng)
    fresh_state["e801_dynamic_off"] = True
    base_runtime.restore_rng(shared_rng)
    history.load_history().zero_optimizers(fresh_state)
    reference_dynamic = history.forward(fresh_state, HISTORICAL_GROUP, training=True, collect=False)
    reference_dynamic["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        fresh_state[key].step()
    rng_post_reference_train = base_runtime.rng_sha(base_runtime.pack_rng())
    with torch.no_grad():
        history.forward(fresh_state, HISTORICAL_GROUP, training=False, collect=False)
    rng_post_reference_eval = base_runtime.rng_sha(base_runtime.pack_rng())
    if rng_post_dynamic_train != rng_post_reference_train or rng_post_dynamic_eval != rng_post_reference_eval:
        raise RuntimeError("isolated stochastic branch shifted the common DARG train RNG")
    history_task = json.loads((Path(CFG["sources"]["e74_root"]) / ".runtime" / "tasks" / f"{HISTORICAL_GROUP}_f01.json").read_text())
    history_epoch = next(item for item in history_task["rng_trace"] if item["epoch"] == 11)
    if (rng_pre_dynamic != history_epoch["rng_before"] or rng_post_dynamic_train != history_epoch["rng_after_train"] or
            rng_post_dynamic_eval != history_epoch["rng_after_eval"]):
        raise RuntimeError("dynamic-on first-step RNG triplet differs from historical E74.4-G2 epoch 11")

    result = {
        "status": "PASS",
        "source_sha256": source_hash,
        "source_files": source_files,
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else str(DEVICE),
        "device_count": torch.cuda.device_count() if DEVICE.type == "cuda" else 0,
        "e80_data_status": data["summary"]["status"],
        "dynamic_coverage": data["summary"]["coverage"],
        "90s_windows": {k: data["summary"][k] for k in ("window_count_min", "window_count_mean", "window_count_median", "window_count_max")},
        "prompt_cache_sha256": file_sha(prompt_cache_path()),
        "prompt_embedding_sha256": {key: text_cache[f"{key}_embedding_sha256"] for key in ("medical", "anonymous")},
        "T1_zero_residual_forward_exact": True,
        "T1_cam_and_output_layer_calls": {"cam": t1_cam_call_count, "output_layer": t1_output_layer_call_count},
        "T2_zero_residual_gradient_exact": True,
        "T3_zero_residual_optimizer_update_and_rng_exact": True,
        "T4_classification_gradient_norms": grad_norms,
        "T5_T6_group_capacity_init": initials,
        "T7_padding_invariance": True,
        "T8_patient_split_pair_identity": {"fold1_split_sha": identity["split_sha"], "fold1_pair_sha": identity["pair_sha"],
                                            "fold2_overlap_one_based": [377, 407], "pair_count": 4096},
        "dynamic_on_shared_rng_before": rng_pre_dynamic,
        "dynamic_on_shared_rng_after_train": rng_post_dynamic_train,
        "dynamic_on_shared_rng_after_eval": rng_post_dynamic_eval,
        "historical_forward_shared_rng_after_train": rng_post_reference_train,
        "historical_forward_shared_rng_after_eval": rng_post_reference_eval,
        "initial_warmup_sha256": identity["warmup_sha256"],
        "initial_warmup_rng_sha": warmup["rng_sha"],
        "original_cam_class": state["cam"].__class__.__name__,
        "original_output_layer_class": state["cam"].Outputlayer.__class__.__name__,
        "output_layer_call_count": t1_output_layer_call_count,
        "cam_call_count": t1_cam_call_count,
        "training_updates": 0,
        "notes": [
            "E74.4-G2 is read-only and no historical DARG epoch was retrained during preflight.",
            "CAM OutputLayer.forward is instance-adapted once; its original module/parameters and method are called once with CAM144 plus residual.",
            "G1/G2/G3 use the same trainable module schema and optimizer; only the declared window sequence or frozen CLIP text cache differs.",
        ],
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def smoke_test():
    preflight_result = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_result.get("status") != "PASS":
        raise RuntimeError("preflight must pass before smoke testing")
    smoke = []
    for group in GROUPS:
        state, warmup, identity, module, optimizer, private_rng, scaler_sha, text = build_task_state(group, 1)
        source_hash = source_sha()
        history = load_history().load_history()
        base = load_historical_e61().load_base()
        initial_bundle = history.state_bundle(state)
        initial_psdc = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        initial_optimizer = optimizer.state_dict()
        initial_shared_rng = base.pack_rng()
        initial_private_rng = private_rng.state_dict()
        temp_path = RT / "smoke" / f"{group}_f01.pt"
        first = optimizer_step(state, module, optimizer, group, training=True)
        latest_payload = {
            "base": history.state_bundle(state),
            "psdc": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
            "optimizer": optimizer.state_dict(), "shared_rng": base.pack_rng(),
            "private_rng": private_rng.state_dict(), "source_sha256": source_hash,
        }
        save_checkpoint(temp_path, latest_payload)
        loaded = history.torch_load(temp_path)
        second = optimizer_step(state, module, optimizer, group, training=True)
        expected_base = history.state_bundle(state)
        expected_psdc = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_shared_rng = base.pack_rng()
        expected_private_rng = private_rng.state_dict()

        history.load_bundle(state, initial_bundle)
        module.load_state_dict(initial_psdc, strict=True)
        optimizer.load_state_dict(initial_optimizer)
        private_rng.load_state_dict(initial_private_rng)
        base.restore_rng(initial_shared_rng)
        history.load_bundle(state, loaded["base"])
        module.load_state_dict(loaded["psdc"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        private_rng.load_state_dict(loaded["private_rng"])
        base.restore_rng(loaded["shared_rng"])
        replay = optimizer_step(state, module, optimizer, group, training=True)
        replay_base = history.state_bundle(state)
        replay_shared_rng = base.pack_rng()
        if (object_sha(expected_base) != object_sha(replay_base) or
                object_sha(expected_psdc) != object_sha(module.state_dict()) or
                object_sha(expected_optimizer) != object_sha(optimizer.state_dict()) or
                object_sha(expected_private_rng) != object_sha(private_rng.state_dict()) or
                base.rng_sha(expected_shared_rng) != base.rng_sha(replay_shared_rng) or
                float(second["total"].detach().cpu()) != float(replay["total"].detach().cpu())):
            raise RuntimeError(f"serialized Latest one-step replay differs for {group}")
        temp_path.unlink(missing_ok=True)
        smoke.append({"group": group, "loss": float(first["total"].detach().cpu()),
                      "replayed_next_loss": float(replay["total"].detach().cpu()),
                      "classification_loss": float(first["cls"].detach().cpu()),
                      "residual_loss": float(first["residual_loss"].detach().cpu()),
                      "psdc_parameter_count": module.trainable_parameter_count(),
                      "shared_rng_restored": True, "latest_one_step_replay": "PASS"})
    return {"status": "PASS", "tasks": smoke, "formal_result": False}


def launch_all():
    preflight_result = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_result.get("status") != "PASS" or preflight_result.get("source_sha256") != source_sha():
        raise RuntimeError("formal launch is blocked until source-matched preflight passes")
    smoke_path = RESULTS / "smoke_status.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text()).get("status") != "PASS":
        raise RuntimeError("formal launch requires a passing smoke test")
    TASKS.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    pending = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    limit = min(int(CFG["training"]["max_workers"]), 10)
    active = {}
    all_results = []
    while pending or active:
        while pending and len(active) < limit:
            group, fold = pending.pop(0)
            task = task_path(group, fold)
            if task.exists():
                value = json.loads(task.read_text(encoding="utf-8"))
                if value.get("status") == "COMPLETE" and value.get("source_sha256") == source_sha() and checkpoint_path(group, fold, "best").is_file():
                    all_results.append(value)
                    continue
                raise RuntimeError(f"incomplete or incompatible task artifact requires inspection: {task}")
            log = (LOGS / f"{group}_f{fold:02d}.stdout.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--group", group, "--fold", str(fold)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            active[proc] = (group, fold, log)
            print(f"START {group} fold={fold} pid={proc.pid} workers={len(active)}/{limit}", flush=True)
        completed = []
        for proc, info in list(active.items()):
            code = proc.poll()
            if code is not None:
                group, fold, log = info
                log.close()
                if code != 0:
                    tail = (LOGS / f"{group}_f{fold:02d}.stdout.log").read_text(encoding="utf-8", errors="replace")[-8000:]
                    raise RuntimeError(f"formal worker failed {group} fold={fold} exit={code}:\n{tail}")
                value = json.loads(task_path(group, fold).read_text(encoding="utf-8"))
                all_results.append(value)
                completed.append(proc)
                print(f"DONE {group} fold={fold} ({len(all_results)}/30)", flush=True)
        for proc in completed:
            active.pop(proc)
        if active:
            time.sleep(2)
    if len(all_results) != 30:
        raise RuntimeError(f"expected 30 completed tasks, found {len(all_results)}")
    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--fold", type=int, choices=range(1, 11))
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, ensure_ascii=False))
    elif args.smoke:
        result = smoke_test()
        atomic_json(RESULTS / "smoke_status.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.worker:
        if args.group is None or args.fold is None:
            parser.error("--worker requires --group and --fold")
        train_task(args.group, args.fold)
    elif args.formal:
        result = launch_all()
        print(json.dumps({"status": "ALL_TASKS_COMPLETE", "completed": len(result)}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
