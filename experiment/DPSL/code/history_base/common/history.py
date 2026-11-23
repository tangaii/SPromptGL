"""Read-only historical loader and protected identities."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from standalone_paths import DPSL_ROOT, VENDOR_REPO, VENDOR_HISTORY

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
REPO = VENDOR_REPO
HISTORY = VENDOR_HISTORY
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
E744 = REPO / "experiment/experiment74_4_soft_positive_dcr"
E822 = HISTORY / "experiment82_2_psce"
E823 = HISTORY / "experiment82_3_cspe"
E8232 = HISTORY / "experiment82_3_2_da_srpe"
E824 = HISTORY / "experiment82_4_four_spectral_prompt"
E8241 = HISTORY / "experiment82_4_1_psu_jacs"
E8242 = HISTORY / "experiment82_4_2_sacs_pdjc"
E8243 = HISTORY / "experiment82_4_3_ps_brsc_pe_djsc"
E8244 = HISTORY / "experiment82_4_4_ba_psce_ba_da_srpe"
E8245 = HISTORY / "experiment82_4_5_ec_psce_ecda_srpe"
GPU = CFG["gpu"]
os.environ.update(
    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), CUDA_DEVICE_ORDER="PCI_BUS_ID",
    PYTHONDONTWRITEBYTECODE="1",
)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(GPU["threads"])
sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("_e8245_locked_history", E8242 / "common/history.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)

import numpy as np
import torch

require = archive.require
runner = archive.runner
base = archive.base
rng_sha = archive.rng_sha
sha = archive.sha
sha_file = archive.sha_file
write_json = archive.atomic_json
read_json = archive.read_json
save = archive.atomic_torch
load = archive.load_torch
write_csv = archive.write_csv
sync_dir = archive.sync_directory
OPTS = tuple(CFG["historical_optimizers"])
BASE = CFG["base"]
FOLDS = range(1, CFG["folds"] + 1)
GROUPS = tuple(CFG["groups"])
LOSS_KEYS = ("cls", "graph", "pair", "adv", "preserve", "total")


def warmup_and_membership(fold):
    state, warmup, identity, optimizers = archive.prepare(fold)
    require(identity["train_pair_scope_pass"], "Historical training pair scope")
    local_preflight = ROOT.parents[1] / "runtime" / "history" / f"fold{fold:02d}.preflight.json"
    previous = read_json(local_preflight)["provenance"]
    require(identity == previous["identity"] and optimizers == previous["optimizers"],
            f"Fold {fold} historical parameters and all optimizer moments differ")
    members = archive.membership(state)
    # Fold-specific DPSL adapters record the unextended SCMR membership as
    # ``source_membership_sha`` and the final masked membership separately.
    # The historical warmup loader must validate the state it actually builds.
    expected_membership = previous.get("source_membership_sha",
                                      previous.get("membership_sha", previous["membership"]))
    require(sha(members) == expected_membership,
            "Historical fold identities or pair order differ")
    require(state["excluded"] == ([376, 406] if fold == 2 else []), "Fold 2 row exclusion differs")
    return state, warmup, identity, optimizers, members


def spectrum(fold, identity, legal):
    cache = load(DPSL_ROOT / "inputs" / "spectrum" / f"spectrum_f{fold:02d}.pt")
    require(cache["identity"] == identity and np.array_equal(cache["legal_train"], legal),
            "Original O origin or legal patient membership differs")
    data = cache["spectrum"]
    require(sha(data) == cache["spectra_sha"] and sha(data["U"]) == cache["e82_original_U_sha"],
            "Original O archive digest differs")
    require(data["U"].shape == (144, 16) and not data["fallback"], "Original O directions differ")
    orthogonality = (data["U"].T @ data["U"] - torch.eye(16, dtype=torch.float64)).abs().max().item()
    require(orthogonality < 1e-10, "Original O is not orthonormal")
    return cache


def hist_config():
    cfg = runner().CFG
    training = cfg["training"]
    data = cfg["data"]
    require(training["max_epochs"] == CFG["max_epochs"] and
            training["early_stop_wait_gt"] == CFG["early_stop_wait_gt"] and training["pair_count"] == 4096
            and training["no_per_epoch_reseed"] and training["checkpoint_selection"].startswith("strict test-fold ACC"),
            "Registered historical training conditions differ")
    require(data["validation_equals_test"] and data["seed"] == CFG["seed"] and data["folds"] == CFG["folds"] and
            data["intentional_fold2_overlap_1based"] == CFG["fold2_overlap_1based"],
            "Registered historical test protocol differs")
    actual = {key: [s["class"], s["lr"], s["weight_decay"]] for key, s in
              zip(OPTS, (training["optimizers"][role] for role in ("gfr", "cam", "darg")))}
    require(actual == CFG["historical_optimizers"], "Registered optimizer settings differ")
    return {"training": training, "data": data, "reference_base": training["base"]}


def hist_trace(fold):
    task = archive.historical_task(fold)
    rows = task["rng_trace"]
    require([r["epoch"] for r in rows] == list(range(11, task["epochs_executed"] + 1)),
            f"Fold {fold} historical RNG log is incomplete")
    return {row["epoch"]: row for row in rows}


def base_bundle(state):
    return runner().load_history().state_bundle(state)


def restore_base(state, bundle):
    runner().load_history().load_bundle(state, bundle)


def reference_paths(fold):
    return {
        "DARG": (runner().best_path(BASE, fold), E744 / "results/fold_metrics.csv", BASE, 1),
        "E82.2-G2": (E822 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                     E822 / "results/fold_metrics.csv", "G2", 100),
        "E82.3.2-G2": (E8232 / "runtime/checkpoints" / f"G2_f{fold:02d}.best.pt",
                       E8232 / "results/fold_metrics.csv", "G2", 100),
        "E82.4.5-G1": (E8245 / "checkpoints" / f"G1_f{fold:02d}.best.pt",
                       E8245 / "results/fold_metrics.csv", "82.4.5-G1 EC-PSCE", 1),
        "E82.4.5-G2": (E8245 / "checkpoints" / f"G2_f{fold:02d}.best.pt",
                       E8245 / "results/fold_metrics.csv", "82.4.5-G2 ECDA-SRPE", 1),
    }


def protected_files():
    files = {Path(name) for name in runner().source_manifest()}
    files.update((archive.ADAPTER, archive.TEXT_CACHE, archive.TEXT_SOURCE, E744 / "manifest.json",
                  E744 / "results/audit_status.json", E744 / "results/fold_metrics.csv"))
    files.update(REPO.glob("*SPromptGL*.pdf"))
    for folder in (E61, E744, E822, E823, E8232, E824, E8241, E8242, E8243, E8244, E8245):
        files.update(folder.glob("*.py"))
        files.update(folder.glob("common/*.py"))
        files.update(folder.glob("runtime/*.py"))
        files.update(folder.glob("G*/*.py"))
        files.update(folder.glob("G*/config.json"))
        files.update(folder.glob("results/*.csv"))
        files.update(folder.glob("results/*.md"))
        files.update(folder.glob("results/audit_status.json"))
        for name in ("config.json", "manifest.json", "design.md"):
            if (folder / name).is_file():
                files.add(folder / name)
    files.update((E822 / "runtime").glob("spectrum_f*.pt"))
    files.update((E823 / "runtime").glob("class_stats_f*.pt"))
    files.update(archive.VENDOR.glob("experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/fold*.pt"))
    files.update(archive.VENDOR.glob("experiment/experiment51_dugsa/*.py"))
    files.update(archive.VENDOR.glob("experiment/experiment51_dugsa/source/*.py"))
    files.update(archive.PYTHON_PROFILE.glob("*/METADATA"))
    for fold in FOLDS:
        files.add(runner().task_path(BASE, fold))
        files.update(item[0] for item in reference_paths(fold).values())
        files.add(E8242 / "G2_pdjc_da_srpe/checkpoints" / f"G2_f{fold:02d}.best.pt")
        files.update(directory / "runtime/audit" / f"G{group}_f{fold:02d}.json"
                     for directory in (E8245,) for group in (1, 2))
        files.add(E8241 / "G1_psu_psce/runtime/preflight" / f"G1_f{fold:02d}.json")
    return archive.hash_files(sorted(files))


def implementation():
    sources = [*ROOT.glob("*.py"), *ROOT.glob("common/*.py"), ROOT / "config.json", ROOT / "design.md",
               ROOT / "runtime/recover_contexts.py"]
    return {str(path.relative_to(ROOT)): sha_file(path) for path in sorted(sources)}
