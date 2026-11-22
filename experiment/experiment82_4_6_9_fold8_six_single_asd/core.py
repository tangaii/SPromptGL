"""Adapt ordered DARG training membership; use the historical SCMR engine."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
CONFIG = json.loads(CONFIG_FILE.read_bytes())
RUNTIME = ROOT / "runtime"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "manifest.json"
SOURCES = {name: Path(spec[0]) for name, spec in CONFIG["source_manifests"].items()}
FOLD = CONFIG["fold"]
GROUPS = CONFIG["groups"]
# The base dispatches G2 to SCMR. This is separate from the six task groups.
MODEL_VARIANT = "G2"
MASKED_COMPONENTS = ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")

sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCES["e462"]))
try:
    import extension as ext
finally:
    sys.path.pop(0)

import numpy as np
import torch

h, parent, prior = ext.h, ext.parent, ext.prior
optimization = ext.optimization


def implementation():
    return {name: h.sha_file(ROOT / name) for name in
            ("config.json", "core.py", "sources.py", "audit_tasks.py", "execute.py")}


def paths(group):
    h.require(group in GROUPS, "Unregistered E82.4.6.9 group")
    stem = f"{group}_f{FOLD:02d}"
    return {
        "best": ROOT / "checkpoints" / f"{stem}.best.pt",
        "latest": ROOT / "checkpoints" / f"{stem}.latest.pt",
        "journal": RUNTIME / f"{stem}.jsonl",
        "task": RUNTIME / f"{stem}.training.json",
        "preflight": RUNTIME / f"{stem}.preflight.json",
        "audit": RUNTIME / f"{stem}.audit.json",
        "log": RUNTIME / f"{stem}.log",
        "lock": RUNTIME / f"{stem}.lock",
    }


def historical_pair_sampler():
    e69 = h.runner().load_history().load_e69()
    return e69, e69.load_e61()


def setup(group, fold=FOLD, *, verify_spectrum=False):
    h.require(group in GROUPS and fold == FOLD, "Unregistered experiment group or fold")
    ctx = parent.setup(MODEL_VARIANT, fold, verify_spectrum=verify_spectrum)
    original = ctx["provenance"]
    old = ctx["members"]
    canonical = h.read_json(parent.paths(MODEL_VARIANT, fold)["preflight"])
    h.require(original == canonical["provenance"], "Canonical Fold 8 initialization differs")
    state, module = ctx["state"], ctx["module"]
    identity = ext.identity_snapshot(state, module, ctx["optimizer"])
    row = GROUPS[group]["patient_index_0based"]
    candidates = [task["patient_index_0based"] for task in GROUPS.values()]
    labels = state["labels_np"]
    e69, e61 = historical_pair_sampler()
    raw_labels = e61.load_base().load_data()[0][1]
    split_train, split_test = e61.load_base().load_data()[1][fold - 1]
    sizes = CONFIG["fold8_sizes"]
    h.require(
        len(set(candidates)) == len(GROUPS) == 6
        and all(task["patient_id_1based"] == task["patient_index_0based"] + 1
                for task in GROUPS.values())
        and np.array_equal(labels, raw_labels)
        and np.array_equal(old["train"], split_train)
        and np.array_equal(old["test"], split_test)
        and len(split_train) == sizes["native_original"]
        and len(split_test) == len(np.unique(split_test)) == sizes["test"]
        and not np.intersect1d(split_train, split_test).size
        and np.array_equal(old["legal"], split_train)
        and all(i in split_test and raw_labels[i] == 0 for i in candidates),
        "Fold 8 original members or candidate label registration differs",
    )
    train = np.append(split_train, row)
    h.require(len(train) == len(np.unique(train)) == sizes["native"]
              and np.intersect1d(train, split_test).tolist() == [row]
              and [i for i in candidates if i in train] == [row],
              "Group must append exactly one independent target")
    before = h.rng_sha()
    pairs, same = e61.make_outer_pairs(fold, state["supports"], train, labels)
    after = h.rng_sha()
    h.require(before == after == identity["historical_rng"], "Relation sampling perturbed global RNG")
    h.require(pairs.shape == (4096, 2) and np.isin(pairs, train).all()
              and (pairs[:, 0] != pairs[:, 1]).all()
              and len(np.unique(np.sort(pairs, axis=1), axis=0)) == 4096
              and np.array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]])
              and np.bincount(same, minlength=2).tolist() == [2048, 2048],
              "Ordered E61 relation pair source, balance or labels differ")
    counts = np.bincount(labels[train], minlength=2)
    weights = (len(train) / counts / 2).astype(np.float32)
    device = state["y"].device
    state.update(
        train_np=train, train=torch.as_tensor(train, dtype=torch.long, device=device),
        weights=torch.from_numpy(weights).to(device), train_pairs_np=pairs, train_same_np=same,
        train_pairs=torch.as_tensor(pairs, dtype=torch.long, device=device),
        train_same=torch.as_tensor(same, dtype=torch.float32, device=device),
        pair_sha=e61.pair_sha(pairs, same), effective_train_sha=e69.index_sha(train),
        added_train_rows_1based=[row + 1], added_train_count=1,
        train_test_overlap_1based=[row + 1], pair_rng_unchanged=True, excluded=[row],
    )
    module.spectrum.allowed[row] = False
    members = h.archive.membership(state)
    now = ext.identity_snapshot(state, module, ctx["optimizer"])
    h.require(all(now[key] == digest for key, digest in identity.items() if key != "spectrum_and_prompts"),
              "DARG membership changed model, optimizer, inputs or RNG")
    h.require(np.array_equal(members["legal"], old["legal"])
              and np.array_equal(members["test"], old["test"])
              and h.sha(ctx["reference"]) == original["reference"]
              and np.array_equal(ctx["reference"]["legal"], members["legal"])
              and torch.equal(module.statistics.legal, state["legal"]),
              "DARG membership affected the SCMR, Gaussian or EMA scope")
    endpoint_counts = np.bincount(pairs.ravel(), minlength=len(labels))
    uncovered = train[endpoint_counts[train] == 0]
    registration = {
        "task_group": group, "fold": fold, "patient_id_1based": row + 1, "patient_index_0based": row,
        "label": int(labels[row]), "supervision_mode": CONFIG["supervision_mode"],
        "original_native_rows": len(split_train), "native_rows": len(train),
        "legal_scmr_gaussian_ema_rows": len(members["legal"]), "test_rows": len(split_test), "masked_rows": 1,
        "original_train_sha": e69.index_sha(split_train), "extended_train_sha": state["effective_train_sha"],
        "original_test_sha": e69.index_sha(split_test), "class_counts_asd_td": counts.tolist(),
        "class_weights_asd_td": weights.tolist(), "pair_sha": state["pair_sha"],
        "pair_count": len(pairs), "same_class_pairs": int(same.sum()),
        "different_class_pairs": int((same == 0).sum()), "covered_rows": len(train) - len(uncovered),
        "uncovered_rows": len(uncovered), "uncovered_ids_1based": (uncovered + 1).tolist(),
        "target_pair_endpoint_occurrences": int(endpoint_counts[row]),
        "target_pair_same_occurrences": int(((pairs == row).any(axis=1) & (same == 1)).sum()),
        "target_pair_different_occurrences": int(((pairs == row).any(axis=1) & (same == 0)).sum()),
        "pair_rng_before": before, "pair_rng_after": after,
    }
    provenance = dict(original, membership=h.sha(members), initial_module=h.sha(module.state_dict()),
                      registration=registration, source_membership_sha=original["membership"])
    shared_initial = {**identity, "gaussian": h.sha(ctx["reference"]), "ema": h.sha(parent.ema(ctx)),
                      "legal_difficulty": original["memory_sha"], "native_provenance": h.sha(original)}
    return {**ctx, "task_group": group, "old_members": old, "members": members,
            "provenance": provenance, "shared_initial": shared_initial}


def check_mask(capture, module, row):
    details = capture["details"]
    h.require(torch.where(~module.spectrum.allowed)[0].tolist() == [row]
              and all(torch.count_nonzero(details[key][row]).item() == 0 for key in MASKED_COMPONENTS)
              and torch.equal(capture["enhanced"][row], capture["cam"][row]),
              f"Patient {row + 1} masked residual or native CAM differs")


def engine(group):
    run = prior.load_parent_module(f"_e82469_engine_{group}", "run.py")
    run.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    run.ROOT, run.RUNTIME, run.RESULTS, run.MANIFEST = ROOT, RUNTIME, RESULTS, MANIFEST
    row = GROUPS[group]["patient_index_0based"]

    def write_json(path, value):
        if path == paths(group)["task"]:
            value = {**value, "group": group, "task_group": group, "checkpoint_internal_model_variant": MODEL_VARIANT,
                     "supervision_mode": CONFIG["supervision_mode"]}
        h.write_json(path, value)

    run.h.write_json = write_json

    def check_group(model_variant, fold):
        h.require(model_variant == MODEL_VARIANT and fold == FOLD, "Historical SCMR dispatch differs")

    def setup_task(model_variant, fold, **kwargs):
        check_group(model_variant, fold)
        return setup(group, fold, **kwargs)

    def task_paths(model_variant, fold):
        check_group(model_variant, fold)
        return paths(group)

    run.setup, run.paths = setup_task, task_paths
    base_meta = run.metadata
    base_restore = run.restore
    base_forward = run.forward
    base_transition = run.transition
    base_advance = run.observation_and_advance
    receipt = None

    def metadata(ctx, hashes):
        return {**base_meta(ctx, hashes), "experiment": CONFIG["experiment"], "task_group": group,
                "model_variant": CONFIG["model_variant"], "supervision_mode": CONFIG["supervision_mode"],
                "experiment_config_sha256": hashes["config.json"]}

    def restore(ctx, payload, hashes=None):
        h.require(payload["task_group"] == ctx["task_group"] == group
                  and payload["experiment"] == CONFIG["experiment"]
                  and payload["supervision_mode"] == CONFIG["supervision_mode"]
                  and payload["experiment_config_sha256"] == h.sha_file(CONFIG_FILE),
                  "Checkpoint is from a different task or supervision mode")
        base_restore(ctx, payload, hashes)
        h.require(torch.where(~ctx["module"].spectrum.allowed)[0].tolist() == [row],
                  "Checkpoint mask does not match its registered patient")

    def forward(state, training, module=None, intervention="full"):
        out, capture = base_forward(state, training, module, intervention)
        if module is not None:
            check_mask(capture, module, row)
            if receipt is not None:
                receipt.append("train" if training else "eval")
        return out, capture

    def advance(ctx, capture):
        h.require(torch.equal(ctx["module"].statistics.legal, ctx["state"]["legal"])
                  and len(ctx["labels"]) == CONFIG["fold8_sizes"]["legal"], "EMA input scope differs")
        return base_advance(ctx, capture)

    def transition(ctx, epoch, dropout, *, inspect=False):
        nonlocal receipt
        receipt = []
        try:
            result, diagnostics, rng, previous, observed = base_transition(ctx, epoch, dropout, inspect=inspect)
            h.require(receipt == ["train", "eval"], "Residual masking not checked in both phases")
            diagnostics["routing"]["scope_receipt"] = {
                "phases": receipt, "task_group": group, "overlap_1based": [row + 1],
                "legal_rows": len(ctx["state"]["legal"]), "masked": True,
            }
            return result, diagnostics, rng, previous, observed
        finally:
            receipt = None

    run.metadata, run.restore, run.forward = metadata, restore, forward
    run.transition, run.observation_and_advance = transition, advance
    return run


def checkpoint_in_memory(run, ctx, hashes, **extra):
    with io.BytesIO() as buffer:
        torch.save({**run.snapshot(ctx, run.metadata(ctx, hashes)), **extra}, buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)


def evaluator(run, name):
    proof = prior.load_parent_module(f"_e82469_audit_{name}", "audit.py")
    proof.run = run
    return proof
