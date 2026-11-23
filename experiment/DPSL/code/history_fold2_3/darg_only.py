"""E82.4.6.3 membership adapter over the immutable SCMR training loop."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from standalone_paths import DPSL_ROOT

ROOT = Path(__file__).resolve().parent
MANIFEST, RUNTIME, RESULTS = (ROOT / name for name in ("manifest.json", "runtime", "results"))
CONFIG = json.loads((ROOT / "e82463_config.json").read_bytes())
FULL_PATH = DPSL_ROOT / "code" / "vendor" / "history" / "experiment82_4_6_2_g2_full_path_extension"
GROUP, FOLDS = CONFIG["group"], tuple(CONFIG["folds"])
EXPOSED = {int(fold): tuple(i - 1 for i in patients) for fold, patients in CONFIG["overlap_1based"].items()}

sys.dont_write_bytecode = True
sys.path.insert(0, str(FULL_PATH))
try:
    import extension as ext
    import audit_extension as full_audit
    import report_extension as formats
finally:
    sys.path.remove(str(FULL_PATH))

import numpy as np
import torch

h, f, parent, prior = ext.h, ext.f, ext.parent, ext.prior
CFG, optimization, spectral = ext.CFG, ext.optimization, ext.spectral
NATIVE_FIELDS = (
    "train_np", "train", "weights", "train_pairs_np", "train_same_np", "train_pairs", "train_same",
    "pair_sha", "effective_train_sha", "added_train_rows_1based", "added_train_count",
    "train_test_overlap_1based", "pair_rng_unchanged",
)
MASKED_COMPONENTS = ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")


def implementation():
    return {path.name: h.sha_file(path) for path in (ROOT / "e82463_config.json", Path(__file__))}


def source_manifests():
    base, direct = ext.source_manifests()
    file = FULL_PATH / "manifest.json"
    full = h.read_json(file)
    h.require(h.sha_file(file) == CONFIG["full_path_manifest_sha256"] and full["status"] == "COMPLETE_AUDITED"
              and full["implementation_sha"] == h.sha(full["implementation"]) == CONFIG["full_path_frozen_sha"]
              and full["implementation"] == ext.implementation(), "E82.4.6.2 source identity differs")
    for item in CONFIG["source_verification"]:
        h.require(h.sha_file(Path(item["path"])) == item["server_sha256"], "Verified published source changed")
    return base, direct, full


def paths(group, fold):
    h.require(group == GROUP and fold in FOLDS, "Only the two registered SCMR tasks may train here")
    stem = f"{group}_f{fold:02d}"
    return {"best": ROOT / "checkpoints" / f"{stem}.best.pt",
            "latest": ROOT / "checkpoints" / f"{stem}.latest.pt",
            "journal": RUNTIME / f"{stem}.jsonl", "task": RUNTIME / f"{stem}.training.json",
            "preflight": RUNTIME / f"{stem}.preflight.json"}


def setup(group, fold, *, verify_spectrum=False):
    h.require(group == GROUP and fold in FOLDS, "Unregistered task")
    ctx = parent.setup(group, fold, verify_spectrum=verify_spectrum)
    old = ctx["members"]
    original = ctx["provenance"]
    h.require(len(old["legal"]) == CONFIG["new_supervision_rows"],
              "Original SCMR membership or Gaussian source differs")
    state, module = ctx["state"], ctx["module"]
    snapshot = ext.identity_snapshot(state, module, ctx["optimizer"])
    reference_sha, ema_sha = h.sha(ctx["reference"]), h.sha(parent.ema(ctx))

    # The parent's membership helper expects its native and SCMR inputs to agree.
    # Apply it to a temporary mapping and copy back only its DARG fields.
    scratch = dict(state)
    registration = ext.extend_native(scratch, {**old, "legal": old["train"]}, fold)
    for key in NATIVE_FIELDS:
        state[key] = scratch[key]
    state["excluded"] = list(EXPOSED[fold])
    module.spectrum.allowed[state["excluded"]] = False

    members = h.archive.membership(state)
    h.require(np.array_equal(members["legal"], old["legal"]) and np.array_equal(members["test"], old["test"])
              and np.array_equal(np.setdiff1d(members["train"], members["legal"]), EXPOSED[fold])
              and h.sha(ctx["reference"]) == reference_sha == original["reference"]
              and h.sha(parent.ema(ctx)) == ema_sha, "DARG extension escaped its allowed scope")
    now = ext.identity_snapshot(state, module, ctx["optimizer"])
    h.require(all(value == now[key] for key, value in snapshot.items() if key != "spectrum_and_prompts"),
              "Native or new trainable initialization, optimizer, or RNG changed")
    full = h.read_json(DPSL_ROOT / "runtime" / "history" / f"fold{fold:02d}.preflight.json")[
        "provenance"]["registration"]
    for key in ("extended_train_index_sha", "extended_pair_sha", "original_test_index_sha",
                "after_class_counts_asd_td", "after_class_weights_asd_td"):
        h.require(registration[key] == full[key], f"DARG membership differs from the formal preflight: {key}")
    registration.update(before_legal=len(old["legal"]), after_legal=len(members["legal"]),
                        legal_index_sha=original["legal_index_sha"], excluded_0based=state["excluded"],
                        legal_labels_sha=h.sha(ctx["labels"].cpu()), original_o_sha=original["spectrum"],
                        original_o_u16_sha=original["basis"], medical_prompt_sha=original["prompt"],
                        gaussian_reference_sha=reference_sha, initial_ema_sha=ema_sha,
                        restored_initial_identity=snapshot)
    provenance = dict(original, membership=h.sha(members), initial_module=h.sha(module.state_dict()),
                      source_membership_sha=original["membership"], registration=registration)
    return {**ctx, "old_members": old, "provenance": provenance, "members": members}


def checked_mask(capture, module, fold):
    rows = list(EXPOSED[fold])
    details = capture["details"]
    h.require(torch.equal(torch.where(~module.spectrum.allowed)[0], torch.as_tensor(rows, device=module.spectrum.allowed.device))
              and all(torch.count_nonzero(details[k][rows]).item() == 0 for k in MASKED_COMPONENTS)
              and torch.equal(capture["enhanced"][rows], capture["cam"][rows]),
              f"Fold {fold} overlap changed CAM144 or a residual component")


def engine(fold):
    run = prior.load_parent_module(f"_darg_only_training_{fold}", "run.py")
    run.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    run.ROOT, run.RUNTIME, run.RESULTS, run.MANIFEST = ROOT, RUNTIME, RESULTS, MANIFEST
    run.paths, run.setup = paths, setup
    metadata, forward, transition, advance = run.metadata, run.forward, run.transition, run.observation_and_advance
    receipt = None

    def registered_metadata(ctx, hashes):
        return {**metadata(ctx, hashes), "experiment": CONFIG["experiment"], "adaptation": CONFIG["name"]}

    def check_forward(state, training, module=None, intervention="full"):
        out, capture = forward(state, training, module, intervention)
        if module is not None:
            checked_mask(capture, module, fold)
            if receipt is not None:
                receipt.append("train" if training else "eval")
        return out, capture

    def check_advance(ctx, capture):
        legal = ctx["state"]["legal"]
        h.require(len(legal) == CONFIG["new_supervision_rows"]
                  and torch.equal(ctx["module"].statistics.legal, legal), "EMA legal device rows changed")
        return advance(ctx, capture)

    def checked_transition(ctx, epoch, dropout, *, inspect=False):
        nonlocal receipt
        receipt = []
        try:
            result, diag, rng, previous, observed = transition(ctx, epoch, dropout, inspect=inspect)
            h.require(receipt == ["train", "eval"], "Overlap masks were not checked in both passes")
            diag["routing"]["strict_overlap_mask"] = {
                "phases": receipt, "excluded_0based": list(EXPOSED[fold]), "components": list(MASKED_COMPONENTS),
                "cam_exact": True,
            }
        finally:
            receipt = None
        return result, diag, rng, previous, observed

    run.metadata, run.forward, run.transition, run.observation_and_advance = (
        registered_metadata, check_forward, checked_transition, check_advance
    )
    return run
