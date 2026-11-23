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
# The base dispatches G2 to SCMR, independently of the formal task group.
MODEL_VARIANT = "G2"
MASKED_COMPONENTS = ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")
CANDIDATE_IDS = (383, 430, 723)

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
    h.require(group in GROUPS, "Unregistered E82.4.6.10 group")
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


def registered_rows(group):
    task = GROUPS[group]
    rows = task["patient_indices_0based"]
    # Task membership is ordered and intentionally starts with the common ASD patient.
    expected = {"G1": [383, 430], "G2": [383, 723]}[group]
    h.require(task["patient_ids_1based"] == expected
              and [i + 1 for i in rows] == expected and task["labels"] == [0, 0],
              f"{group} registered pair or order differs")
    return rows


def scope_metadata(rows):
    ids = [i + 1 for i in rows]
    return {"added_train_rows_1based": ids, "added_train_count": len(rows),
            "train_test_overlap_1based": ids, "excluded": rows}


def setup(group, fold=FOLD, *, verify_spectrum=False):
    h.require(group in GROUPS and fold == FOLD, "Unregistered experiment group or fold")
    ctx = parent.setup(MODEL_VARIANT, fold, verify_spectrum=verify_spectrum)
    original = ctx["provenance"]
    old = ctx["members"]
    canonical = h.read_json(parent.paths(MODEL_VARIANT, fold)["preflight"])
    h.require(original == canonical["provenance"], "Canonical Fold 8 initialization differs")
    state, module = ctx["state"], ctx["module"]
    identity = ext.identity_snapshot(state, module, ctx["optimizer"])
    rows = registered_rows(group)
    candidates = [patient - 1 for patient in CANDIDATE_IDS]
    labels = state["labels_np"]
    e69, e61 = historical_pair_sampler()
    (_, raw_labels), splits = e61.load_base().load_data()
    split_train, split_test = splits[fold - 1]
    sizes = CONFIG["fold8_sizes"]
    h.require(
        np.array_equal(labels, raw_labels)
        and np.array_equal(old["train"], split_train)
        and np.array_equal(old["test"], split_test)
        and len(split_train) == sizes["native_original"]
        and np.array_equal(split_train, ctx["warmup"]["train_indices"])
        and np.array_equal(split_test, ctx["warmup"]["test_indices"])
        and len(split_test) == len(np.unique(split_test)) == sizes["test"]
        and not np.intersect1d(split_train, split_test).size
        and np.array_equal(old["legal"], split_train)
        and all(i in split_test and raw_labels[i] == 0 for i in candidates),
        "Fold 8 original members or candidate label registration differs",
    )
    train = np.append(split_train, rows)
    h.require(len(train) == len(np.unique(train)) == sizes["native"]
              and len(rows) == sizes["masked"]
              and np.array_equal(train[:len(split_train)], split_train)
              and train[len(split_train):].tolist() == rows
              and np.intersect1d(train, split_test).tolist() == sorted(rows)
              and set(i for i in candidates if i in train) == set(rows),
              "Group must append only its two ordered targets")
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
        **scope_metadata(rows), pair_rng_unchanged=True,
    )
    module.spectrum.allowed[rows] = False
    members = h.archive.membership(state)
    now = ext.identity_snapshot(state, module, ctx["optimizer"])
    h.require(all(now[key] == digest for key, digest in identity.items() if key != "spectrum_and_prompts"),
              "DARG membership changed model, optimizer, inputs or RNG")
    h.require(np.array_equal(members["legal"], old["legal"])
              and np.array_equal(members["test"], old["test"])
              and h.sha(ctx["reference"]) == original["reference"]
              and np.array_equal(ctx["reference"]["legal"], members["legal"])
              and torch.equal(module.statistics.legal, state["legal"])
              and torch.equal(ctx["labels"], state["y"][state["legal"]])
              and torch.where(~module.spectrum.allowed)[0].tolist() == sorted(rows),
              "DARG membership affected the SCMR, Gaussian or EMA scope")
    endpoint_counts = np.bincount(pairs.ravel(), minlength=len(labels))
    uncovered = train[endpoint_counts[train] == 0]
    registration = {
        "task_group": group, "fold": fold, "patient_ids_1based": [i + 1 for i in rows],
        "patient_indices_0based": rows, "labels": labels[rows].tolist(),
        **scope_metadata(rows), "supervision_mode": CONFIG["supervision_mode"],
        "original_native_rows": len(split_train), "native_rows": len(train),
        "legal_scmr_gaussian_ema_rows": len(members["legal"]), "test_rows": len(split_test), "masked_rows": len(rows),
        "original_train_sha": e69.index_sha(split_train), "extended_train_sha": state["effective_train_sha"],
        "original_test_sha": e69.index_sha(split_test), "class_counts_asd_td": counts.tolist(),
        "class_weights_asd_td": weights.tolist(), "pair_sha": state["pair_sha"],
        "test_labels_sha": h.sha(labels[split_test]), "pair_count": len(pairs), "same_class_pairs": int(same.sum()),
        "different_class_pairs": int((same == 0).sum()), "covered_rows": len(train) - len(uncovered),
        "uncovered_rows": len(uncovered), "uncovered_ids_1based": (uncovered + 1).tolist(),
        "covered_fraction": (len(train) - len(uncovered)) / len(train),
        "target_pairs": {
            str(row + 1): {
                "eligible": True, "sampled": bool(endpoint_counts[row]),
                "endpoint_occurrences": int(endpoint_counts[row]),
                "same_occurrences": int(((pairs == row).any(axis=1) & (same == 1)).sum()),
                "different_occurrences": int(((pairs == row).any(axis=1) & (same == 0)).sum()),
            }
            for row in rows
        },
        "pair_rng_before": before, "pair_rng_after": after,
    }
    provenance = dict(original, membership=h.sha(members), initial_module=h.sha(module.state_dict()),
                      registration=registration, source_membership_sha=original["membership"])
    shared_initial = {**identity, "gaussian": h.sha(ctx["reference"]), "ema": h.sha(parent.ema(ctx)),
                      "legal_difficulty": original["memory_sha"], "native_provenance": h.sha(original)}
    return {**ctx, "task_group": group, "old_members": old, "members": members,
            "provenance": provenance, "shared_initial": shared_initial}


def check_mask(capture, module, rows):
    details = capture["details"]
    h.require(len(rows) == CONFIG["fold8_sizes"]["masked"]
              and torch.where(~module.spectrum.allowed)[0].tolist() == sorted(rows),
              "Fold 8 allowed mask differs from the two registered rows")
    for row in rows:
        h.require(all(torch.count_nonzero(details[key][row]).item() == 0 for key in MASKED_COMPONENTS)
                  and torch.equal(capture["enhanced"][row], capture["cam"][row]),
                  f"Patient {row + 1} masked residual or native CAM differs")


def engine(group):
    run = prior.load_parent_module(f"_e824610_engine_{group}", "run.py")
    run.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    run.ROOT, run.RUNTIME, run.RESULTS, run.MANIFEST = ROOT, RUNTIME, RESULTS, MANIFEST
    rows = registered_rows(group)

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
                **scope_metadata(rows),
                "model_variant": CONFIG["model_variant"], "supervision_mode": CONFIG["supervision_mode"],
                "experiment_config_sha256": hashes["config.json"]}

    def restore(ctx, payload, hashes=None):
        h.require(payload["task_group"] == ctx["task_group"] == group
                  and payload["experiment"] == CONFIG["experiment"]
                  and payload["supervision_mode"] == CONFIG["supervision_mode"]
                  and payload["experiment_config_sha256"] == h.sha_file(CONFIG_FILE)
                  and all(payload[key] == ctx["state"][key] == val for key, val in scope_metadata(rows).items()),
                  "Checkpoint is from a different task or supervision mode")
        base_restore(ctx, payload, hashes)
        before = h.rng_sha()
        forward(ctx["state"], False, ctx["module"])
        h.require(h.rng_sha() == before, "Checkpoint mask readback changed RNG")

    def forward(state, training, module=None, intervention="full"):
        out, capture = base_forward(state, training, module, intervention)
        if module is not None:
            check_mask(capture, module, rows)
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
                "phases": receipt, "task_group": group, "overlap_1based": [i + 1 for i in rows],
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
    proof = prior.load_parent_module(f"_e824610_audit_{name}", "audit.py")
    proof.run = run
    return proof
