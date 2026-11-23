"""Read-only adapters and artifact identities for the historical DARG experiment."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ORIGINAL_SPEC = importlib.util.spec_from_file_location
HERE = Path(__file__).resolve().parents[1]
REPO = Path("/data/adodas/SPromptGL-main")
HISTORY = Path("/model/spromptgl/experiment")
PYTHON_PROFILE = HISTORY / "experiment82_3_6_ps_pgr/runtime/python"
CFG = json.loads((HERE / "config.json").read_text())
GROUPS = {name: json.loads((HERE / directory / "config.json").read_text())
          for name, directory in CFG["group_dirs"].items()}
GPU = CFG["gpu"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU["uuid"]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = str(GPU["threads"])
sys.dont_write_bytecode = True
sys.path.insert(1, str(PYTHON_PROFILE))

import numpy as np
import torch

E744 = REPO / "experiment/experiment74_4_soft_positive_dcr"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
E822 = HISTORY / "experiment82_2_psce"
E824 = HISTORY / "experiment82_4_four_spectral_prompt"
E823 = HISTORY / "experiment82_3_cspe"
E8232 = HISTORY / "experiment82_3_2_da_srpe"
ADAPTER = REPO / "experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py"
VENDOR = REPO / "experiment/experiment72_mgdss/.runtime/vendor/root"
TEXT_CACHE = E822 / "runtime/prompts.pt"
TEXT_SOURCE = REPO / "experiment/experiment60_dss_pen/.runtime/text_embeddings.pt"
BASE = CFG["base_group"]
OPTS = tuple(CFG["historical_optimizers"])
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
SPECTRUM_KEYS = ("U", "mu", "lambda", "loglambda_normalized", "nu_normalized")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(value):
    """Device-independent digest using the E82.2 recursive content encoding."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for member in item:
                visit(member)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


@contextmanager
def atomic_writer(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    settings = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with temporary.open(mode, **settings) as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    with atomic_writer(path, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def atomic_torch(path, value):
    with atomic_writer(path, "wb") as stream:
        torch.save(value, stream)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    with atomic_writer(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def sync_directory(path):
    directory = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_module(name, path):
    spec = ORIGINAL_SPEC(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def runner():
    return load_module("e824_readonly_adapter", ADAPTER).load_base_runner()


@lru_cache(maxsize=1)
def base():
    return runner().load_history().load_e69().load_e61().load_base()


@lru_cache(maxsize=1)
def psce():
    return load_module("e824_readonly_original_spectrum", E822 / "psce.py")


@lru_cache(maxsize=1)
def da():
    return load_module("e824_readonly_margin_target", E8232 / "da_srpe.py")


def rng_sha():
    return base().rng_sha(base().pack_rng())


def historical_task(fold):
    return read_json(runner().task_path(BASE, fold))


def membership(state):
    return {
        "train": np.asarray(state["train_np"], dtype=np.int64),
        "legal": np.asarray(state["legal_np"], dtype=np.int64),
        "test": np.asarray(state["test_np"], dtype=np.int64),
        "train_pairs": state["train_pairs_np"].copy(),
        "train_same": state["train_same_np"].copy(),
        "weights": state["weights"].detach().cpu().clone(),
    }


def prepare(fold):
    device = base().runtime()["device"]
    require(device == torch.device("cuda:0") and torch.cuda.device_count() == 1, "Training device mapping differs")
    torch.set_num_threads(GPU["threads"])
    state, warmup, identity = runner().prepare(fold)
    task = historical_task(fold)
    require(task["status"] == "COMPLETE" and task["identity"] == identity, f"Fold {fold} DARG identity")
    require(warmup["completed_epoch"] == CFG["start_epoch"] and warmup["seed"] == CFG["seed"], "Epoch-10 warmup identity")
    require(rng_sha() == warmup["rng_sha"] == identity["warmup_rng_sha"], "Warmup RNG differs")
    require(len(warmup["rng"]["cuda"]) == 1, "Warmup has unexpected device RNG streams")
    native = runner().load_history()
    require(native.state_model_sha(state) == identity["model_cam_sha"], "DARG parameter identity differs")
    require(type(state["cam"]).__name__ == "PLNNoPEN", "Historical CAM implementation differs")
    require(tuple(state["x"].shape) == (CFG["subjects"], sum(CFG["dimensions"])), "Dataset shape differs")
    require([part.shape[-1] for part in state["parts"]] == CFG["dimensions"], "Modality widths differ")

    excluded = [i - 1 for i in CFG["fold2_exclusions_1based"]] if fold == 2 else []
    train, test = state["train_np"], state["test_np"]
    legal = train[~np.isin(train, excluded)]
    require(np.intersect1d(train, test).tolist() == excluded, "Historical outer overlap differs")
    require(np.array_equal(legal, warmup["train_indices"]) and np.array_equal(test, warmup["test_indices"]),
            "Warmup membership differs")
    require(not excluded or state["labels_np"][excluded].tolist() == [0, 1], "Fold 2 labels differ")
    pairs, same = state["train_pairs_np"], state["train_same_np"]
    require(pairs.shape == (4096, 2) and np.isin(pairs, train).all()
            and np.bincount(same).tolist() == [2048, 2048], "Historical balanced relation-pair scope differs")
    require(np.array_equal(same, state["labels_np"][pairs[:, 0]] == state["labels_np"][pairs[:, 1]]), "Pair labels differ")
    require(fold == 2 or np.array_equal(pairs, warmup["train_pairs"]), "Historical pair identities differ")
    require(fold != 2 or not np.array_equal(pairs, warmup["train_pairs"]), "Fold 2 historical pair replacement missing")

    parameter_names = {id(p): f"{module}.{name}" for module in ("model", "cam")
                       for name, p in state[module].named_parameters()}
    optimizers = {}
    for name, spec in CFG["historical_optimizers"].items():
        optimizer = state[name]
        kind, lr, wd = spec
        digest = sha(optimizer.state_dict())
        require(digest == sha(warmup["optimizer_" + name.removeprefix("opt_")]), f"{name} saved optimizer states differ")
        require(type(optimizer).__name__ == kind
                and all(group["lr"] == lr and group["weight_decay"] == wd for group in optimizer.param_groups),
                f"{name} historical optimizer settings differ")
        optimizers[name] = {
            "sha": digest,
            "groups": [[parameter_names[id(p)] for p in group["params"]] for group in optimizer.param_groups],
            "moment_slots": len(optimizer.state),
        }
    require(runner().sha_object({key: state[key].state_dict() for key in OPTS}) == identity["optimizer_sha"],
            "Historical combined optimizer identity differs")
    require({p for group in state["opt_cam"].param_groups for p in group["params"]} == set(state["cam"].parameters()),
            "Historical CAM optimizer ownership changed")
    all_owned = [p for key in OPTS for group in state[key].param_groups for p in group["params"]]
    require(len(all_owned) == len(set(all_owned)) and set(all_owned) == set(parameter_names_p(state)),
            "Historical optimizer parameter coverage differs")
    state.update(legal_np=legal, legal=torch.as_tensor(legal, device=device), excluded=excluded)
    return state, warmup, identity, optimizers


def parameter_names_p(state):
    return (p for key in ("model", "cam") for p in state[key].parameters())


def verify_text():
    text = load_torch(TEXT_CACHE)
    origin = load_torch(TEXT_SOURCE)
    require(text["texts"] == origin["text_map"]["original"] == psce().PROMPTS == da().PROMPTS,
            "Eight original text identities differ")
    require(text["source_sha256"] == sha_file(TEXT_SOURCE)
            and sha({k: v for k, v in text.items() if k != "identity_sha"}) == text["identity_sha"],
            "RN50 embedding cache identity differs")
    require(torch.equal(origin["embeddings"]["original"].float(), text["medical"]), "Original RN50 embeddings differ")
    require(text["medical"].shape == (8, 1024) and text["P_text"].shape == (1024, 32)
            and text["rank"] == 8, "Query cache geometry differs")
    for key in ("medical", "anonymous"):
        projected = torch.nn.functional.normalize(text[key].double() @ text["P_text"].double(), dim=-1)
        require(torch.allclose(projected, text["q_" + key].double(), rtol=0, atol=1e-6), f"{key} Query mismatch")
    gram = (text["q_medical"] @ text["q_medical"].T - text["q_anonymous"] @ text["q_anonymous"].T).abs().max()
    require(float(gram) < 1e-6 and not torch.equal(text["q_medical"], text["q_anonymous"]), "Anonymous audit Query geometry differs")
    return text


def gpu_sample():
    fields = "index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu"
    data = subprocess.check_output(
        ["nvidia-smi", "-i", str(GPU["index"]), f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True)
    index, uuid, name, bus, total, used, utilization = (item.strip() for item in data.strip().split(","))
    require((int(index), uuid) == (GPU["index"], GPU["uuid"]), "Physical GPU0 identity differs")
    return dict(index=int(index), uuid=uuid, name=name, bus=bus, total_mib=int(total),
                used_mib=int(used), utilization=int(utilization))


def verify_device():
    gpu = gpu_sample()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Only GPU0 should be visible")
    device = torch.cuda.get_device_properties(0)
    require(str(device.uuid).lower() == gpu["uuid"].removeprefix("GPU-").lower()
            and device.name == gpu["name"] and device.total_memory // 2**20 == gpu["total_mib"],
            "PyTorch device does not match physical GPU0")
    return gpu


def packages():
    return {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
        "profile": str(PYTHON_PROFILE)
    }


def implementation_hashes():
    files = ("config.json", "design.md", "run.py", *(
        str(path.relative_to(HERE)) for directory in ("common", *CFG["group_dirs"].values())
        for path in sorted((HERE / directory).iterdir()) if path.suffix in (".py", ".json")
    ))
    return {name: sha_file(HERE / name) for name in files}


def guard_paths():
    files = set(map(Path, runner().source_manifest()))
    files.update((ADAPTER, TEXT_CACHE, TEXT_SOURCE, E822 / "runtime/spromptgl.pdf"))
    files.update(REPO.glob("*SPromptGL*.pdf"))
    previous = verify_prior()
    files.add(E824 / "manifest.json")
    files.add(E824 / "audit.py")
    files.update(E824 / name for name in previous["implementation"])
    files.update(map(Path, previous["results"]))
    for directory in previous["groups"]:
        group_path = next(E824.glob(directory + "_*"))
        files.update(group_path.glob("checkpoints/*.pt"))
        for phase in ("audit", "preflight", "tasks"):
            files.update((group_path / "runtime" / phase).glob("*.json"))
    files.update((E823 / "runtime").glob("class_stats_f*.pt"))
    for name in ("run.py", "config.json", "manifest.json", "results/report.md", "results/fold_metrics.csv",
                 "results/audit_status.json", "soft_positive_dcr.py", "audit.py"):
        files.add(E744 / name)
    dirs = [
        E822, E8232, *HISTORY.glob("experiment82_2_[12]_*"), *HISTORY.glob("experiment82_3_1_*"),
        *HISTORY.glob("experiment82_3_[3-9]_*"), *HISTORY.glob("experiment82_3_1[01]_*"),
    ]
    for path in dirs:
        files.update(path.glob("*.py"))
        files.update(p for p in [path / "config.json", path / "manifest.json", path / "design_review.md",
                                path / "results/report.md", path / "results/fold_metrics.csv",
                                path / "results/audit_status.json"] if p.is_file())
    for path, glob_pattern in (
        (E744 / ".runtime/tasks", f"{BASE}_f*.json"),
        (E744 / ".runtime/checkpoints", f"{BASE}_f*.best.pt"),
        (E822 / "runtime", "spectrum_f*.pt"),
        (E822 / "runtime/checkpoints", "G2_f*.best.pt"),
        (E8232 / "runtime/checkpoints", "G2_f*.best.pt"),
        (VENDOR / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors", "fold*.pt"),
        (VENDOR / "experiment/experiment51_dugsa", "*.py"),
        (VENDOR / "experiment/experiment51_dugsa/source", "*.py"),
        (REPO / "experiment/experiment53_3_cadr_graph_fallback_fix/.runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11/tasks",
         "CADR_WD_0p01_GFR_WD_0p11_f*.json"),
        (PYTHON_PROFILE, "*/METADATA"),
    ):
        files.update(path.glob(glob_pattern))
    return sorted(files)


def hash_files(paths):
    with ThreadPoolExecutor(max_workers=8) as workers:
        return dict(zip(map(str, paths), workers.map(sha_file, paths)))


def verify_guard(manifest):
    expected = manifest["protected_sha256"]
    current = hash_files([Path(path) for path in expected])
    changes = [path for path in expected if expected[path] != current[path]]
    require(not changes, f"Protected historical files changed: {changes}")
    require(implementation_hashes() == manifest["implementation"], "Experiment source hashes differ from freeze")
    return {"status": "PASS", "files": len(expected), "sha256": sha(current)}

@lru_cache(maxsize=1)
def verify_prior():
    path = E824 / "manifest.json"
    require(sha_file(path) == CFG["prior_manifest_sha"], "E82.4 manifest SHA changed")
    m = read_json(path)
    a = read_json(E824 / "results/audit_status.json")
    require(m["status"] == "COMPLETE_AUDITED" and a["status"] == "PASS" and a["trained"] == 40,
            "E82.4 is missing the 40-fold audit")
    require(sha_file(E824 / "audit.py") == m["audit_publish_sha"], "E82.4 audited publisher differs")
    require(hash_files([E824 / name for name in m["implementation"]])
            == {str(E824 / name): sha for name, sha in m["implementation"].items()}, "E82.4 training source differs")
    require(hash_files([Path(p) for p in m["results"]]) == m["results"], "E82.4 audit output identity differs")
    return m
