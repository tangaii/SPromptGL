"""DARG-only read-only adapter for the frozen E61 warmup and Original O basis."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from standalone_paths import VENDOR_REPO, LOCAL_RUNTIME

ROOT = Path(__file__).resolve().parents[3]
FORMAL_DPSL = Path("/model/spromptgl/experiment/DPSL")
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
THREADS = CFG["gpu"]["threads"]
BASE = "E74.4-G2-Positive-DCR"
OPTS = ("opt_base", "opt_cam", "opt_added")
FOLDS = range(1, CFG["folds"] + 1)
GROUPS = ("DARG + JSL",)
LOSS_KEYS = ("cls", "graph", "pair", "adv", "preserve", "total")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(THREADS)
sys.dont_write_bytecode = True
if str(LOCAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(LOCAL_RUNTIME))

import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(value):
    """Match the content encoding used by the archived E82 identity records."""
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


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def atomic_writer(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with temporary.open(mode, **options) as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    with atomic_writer(path, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def save(path, value):
    with atomic_writer(path, "wb") as stream:
        torch.save(value, stream)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with atomic_writer(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sync_dir(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@lru_cache(maxsize=1)
def runner():
    adapter_path = VENDOR_REPO / "experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py"
    spec = importlib.util.spec_from_file_location("jsl_readonly_e744_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_base_runner()


@lru_cache(maxsize=1)
def base():
    return runner().load_history().load_e69().load_e61().load_base()


def rng_sha():
    runtime = base()
    return runtime.rng_sha(runtime.pack_rng())


def membership(state):
    return {
        "train": np.asarray(state["train_np"], dtype=np.int64),
        "legal": np.asarray(state["legal_np"], dtype=np.int64),
        "test": np.asarray(state["test_np"], dtype=np.int64),
        "train_pairs": state["train_pairs_np"].copy(),
        "train_same": state["train_same_np"].copy(),
        "weights": state["weights"].detach().cpu().clone(),
    }


def warmup_and_membership(fold):
    require(fold in FOLDS, f"Invalid fold {fold}")
    device = base().runtime()["device"]
    require(device == torch.device("cuda:0") and torch.cuda.device_count() == 1,
            "Training must use only physical GPU0")
    torch.set_num_threads(THREADS)
    state, warmup, identity = runner().prepare(fold)

    archived = read_json(FORMAL_DPSL / "runtime/history" / f"fold{fold:02d}.preflight.json")
    require(archived.get("status") == "PASS", f"Fold {fold} archived DPSL preflight is incomplete")
    previous = archived["provenance"]
    require(identity == previous["identity"], f"Fold {fold} E61 model, optimizer, or RNG identity differs")
    require(identity["train_pair_scope_pass"], "Historical training pair scope")
    require(warmup["completed_epoch"] == CFG["start_epoch_completed"] and warmup["seed"] == CFG["seed"],
            "Epoch-10 E61 warmup identity differs")
    require(rng_sha() == warmup["rng_sha"] == identity["warmup_rng_sha"], "Restored warmup RNG differs")
    require(len(warmup["rng"]["cuda"]) == 1, "Warmup contains an unexpected device RNG stream")

    native = runner().load_history()
    require(native.state_model_sha(state) == identity["model_cam_sha"], "DARG parameter identity differs")
    require(type(state["cam"]).__name__ == "PLNNoPEN", "Historical CAM implementation differs")
    require(tuple(state["x"].shape) == (CFG["subjects"], sum(CFG["dimensions"])), "Dataset shape differs")
    require([part.shape[-1] for part in state["parts"]] == CFG["dimensions"], "Modality widths differ")

    parameter_names = {id(parameter): f"{module}.{name}"
                       for module in ("model", "cam")
                       for name, parameter in state[module].named_parameters()}
    optimizers = {}
    for name in OPTS:
        optimizer = state[name]
        optimizers[name] = {
            "sha": sha(optimizer.state_dict()),
            "groups": [[parameter_names[id(parameter)] for parameter in group["params"]]
                       for group in optimizer.param_groups],
            "moment_slots": len(optimizer.state),
        }
    require(optimizers == previous["optimizers"], f"Fold {fold} restored historical optimizer states differ")

    excluded = [376, 406] if fold == 2 else []
    train, test = state["train_np"], state["test_np"]
    legal = train[~np.isin(train, excluded)]
    require(np.intersect1d(train, test).tolist() == excluded, "Historical base overlap differs")
    require(np.array_equal(legal, warmup["train_indices"]) and np.array_equal(test, warmup["test_indices"]),
            "E61 warmup membership differs")
    require(not excluded or state["labels_np"][excluded].tolist() == [0, 1], "Fold 2 overlap labels differ")
    pairs, same = state["train_pairs_np"], state["train_same_np"]
    require(pairs.shape == (4096, 2) and np.isin(pairs, train).all()
            and np.bincount(same, minlength=2).tolist() == [2048, 2048], "Historical pair balance differs")
    require(np.array_equal(same, state["labels_np"][pairs[:, 0]] == state["labels_np"][pairs[:, 1]]),
            "Historical relation-pair labels differ")
    state.update(legal_np=legal, legal=torch.as_tensor(legal, device=device), excluded=excluded)

    members = membership(state)
    expected_membership_sha = previous.get("source_membership_sha", previous["membership"])
    require(sha(members) == expected_membership_sha, f"Fold {fold} historical base membership or order differs")
    return state, warmup, identity, optimizers, members


def spectrum(fold, identity, legal):
    cache = load(ROOT / "inputs/spectrum" / f"spectrum_f{fold:02d}.pt")
    require(cache["identity"] == identity and np.array_equal(cache["legal_train"], legal),
            "Original O origin or legal patient membership differs")
    data = cache["spectrum"]
    require(sha(data) == cache["spectra_sha"] and sha(data["U"]) == cache["e82_original_U_sha"],
            "Original O archive identity differs")
    require(tuple(data["U"].shape) == (144, 16) and not data["fallback"], "Original O basis geometry differs")
    gram = data["U"].T @ data["U"]
    require((gram - torch.eye(16, dtype=torch.float64)).abs().max().item() < 1e-10,
            "Original O basis is not orthonormal")
    return cache


def hist_config():
    config = runner().CFG
    training = config["training"]
    require(training["max_epochs"] == CFG["max_epochs"]
            and training["early_stop_wait_gt"] == CFG["early_stop_wait_gt"]
            and training["pair_count"] == CFG["pair_count"]
            and training["no_per_epoch_reseed"], "Historical training protocol differs")
    return {"training": training, "reference_base": BASE}


def hist_trace(fold):
    journal = FORMAL_DPSL / "runtime/history" / f"fold{fold:02d}.journal.jsonl"
    require(journal.is_file(), f"Fold {fold} archived DPSL RNG journal is missing")
    trace = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        rng = row.get("rng")
        require(isinstance(rng, dict), f"Fold {fold} archived epoch lacks RNG witness")
        trace[int(row["epoch"])] = rng
    require(sorted(trace) == list(range(11, max(trace) + 1)),
            f"Fold {fold} archived RNG journal has missing epochs")
    return trace


def base_bundle(state):
    return runner().load_history().state_bundle(state)


def restore_base(state, bundle):
    runner().load_history().load_bundle(state, bundle)
