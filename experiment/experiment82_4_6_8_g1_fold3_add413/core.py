"""E82.4.6.8 G1 Fold-3 membership adapter with the single patient 413 addition."""
from __future__ import annotations
import json, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MANIFEST, RUNTIME, RESULTS = ROOT / "manifest.json", ROOT / "runtime", ROOT / "results"
CONFIG = json.loads((ROOT / "config.json").read_bytes())
FULL_PATH = Path(CONFIG["full_path_directory"])
GROUP = "G1"
FOLDS = tuple(CONFIG["folds"])
TASKS = tuple((GROUP, f) for f in FOLDS)
CANONICAL = "G2"

def one_based(mapping, fold):
    return tuple(int(x) - 1 for x in mapping[str(fold)])

OLD = {f: one_based(CONFIG["historical_overlap_1based"], f) for f in FOLDS}
ADDED = {f: one_based(CONFIG["additions_1based"], f) for f in FOLDS}
EXPOSED = {f: tuple(dict.fromkeys(one_based(CONFIG["overlap_1based"], f))) for f in FOLDS}

sys.dont_write_bytecode = True
sys.path.insert(0, str(FULL_PATH))
try:
    import extension as ext
finally:
    sys.path.remove(str(FULL_PATH))

import numpy as np
import torch

h, f, parent, prior = ext.h, ext.f, ext.parent, ext.prior
CFG, optimization, spectral, evidence = ext.CFG, ext.optimization, ext.spectral, ext.evidence
parent_preflight = ext.parent_preflight
NATIVE_FIELDS = (
    "train_np", "train", "weights", "train_pairs_np", "train_same_np", "train_pairs", "train_same",
    "pair_sha", "effective_train_sha", "added_train_rows_1based", "added_train_count",
    "train_test_overlap_1based", "pair_rng_unchanged",
)
MASKED_COMPONENTS = ("raw_parallel", "raw_perp", "delta_parallel", "delta_perp", "delta")

def implementation():
    return {p.name: h.sha_file(p) for p in sorted([ROOT / "config.json", *ROOT.glob("*.py")])}

def source_manifests():
    # Every parent is bound to a pre-frozen manifest digest.  A status field alone
    # is insufficient because it can survive an unrecorded source mutation.
    expected = CONFIG["expected_manifest_sha256"]
    paths = {
        "base": FULL_PATH.parent / "experiment82_4_6_pcsr_scmr" / "manifest.json",
        "full": FULL_PATH / "manifest.json",
        "e8246": Path(CONFIG["source_e8246_directory"]) / "manifest.json",
        "e82463": Path(CONFIG["source_e82463_directory"]) / "manifest.json",
        "e82465": Path(CONFIG["source_e82465_directory"]) / "manifest.json",
        "e8266": Path(CONFIG["source_e8266_directory"]) / "manifest.json",
        "e744": Path(CONFIG["source_readback"]["e744_manifest"]),
        "e61": Path(CONFIG["source_readback"]["e61_manifest"]),
    }
    keys = {
        "base": "e8246_pcsr_scmr", "full": "e8246_full_path", "e8246": "e82464_cumulative",
        "e82463": "e82463_darg_only", "e82465": "e82465_g1", "e8266": "e8266_g1", "e744": "e744_g2", "e61": "e61",
    }
    manifests = {}
    for name, path in paths.items():
        manifest = h.read_json(path)
        h.require(manifest.get("status") == CONFIG["required_parent_status"],
                  f"{name} parent is not COMPLETE_AUDITED")
        h.require(h.sha_file(path) == expected[keys[name]],
                  f"{name} parent manifest SHA differs from the frozen expectation")
        manifests[name] = manifest
    return manifests

def paths(group, fold):
    h.require(group == GROUP and fold in FOLDS, "Unregistered task")
    stem = f"{group}_f{fold:02d}"
    return {"best": ROOT / "checkpoints" / f"{stem}.best.pt",
            "latest": ROOT / "checkpoints" / f"{stem}.latest.pt",
            "journal": RUNTIME / f"{stem}.jsonl",
            "task": RUNTIME / f"{stem}.training.json",
            "preflight": RUNTIME / f"{stem}.preflight.json"}

def setup(task_group, fold, *, verify_spectrum=False):
    h.require(task_group == GROUP and fold in FOLDS, "Unregistered task")
    ctx = parent.setup(CANONICAL, fold, verify_spectrum=verify_spectrum)
    old, original = ctx["members"], ctx["provenance"]
    canonical = h.read_json(parent.paths(CANONICAL, fold)["preflight"])
    h.require(original == canonical["provenance"] and len(old["legal"]) == 784,
              "Canonical E82.4.6-G2 membership differs")
    state, module, optimizer = ctx["state"], ctx["module"], ctx["optimizer"]
    old_identity = ext.identity_snapshot(state, module, optimizer)
    labels = state["labels_np"]
    add = np.asarray(ADDED[fold], dtype=np.int64)
    expected = int(CONFIG["native_rows"][str(fold)])
    new_train = np.concatenate((old["train"], add))
    h.require(len(old["train"]) == CONFIG["historical_train_rows"][str(fold)]
              and len(old["train"]) == CONFIG["prior_e8266_train_rows"][str(fold)] - CONFIG.get("prior_e8266_delta", {}).get(str(fold), 6)
              and len(old["legal"]) == 784
              and len(new_train) == expected == len(np.unique(new_train))
              and np.isin(add, old["test"]).all() and not np.isin(add, old["train"]).any()
              and set(np.intersect1d(new_train, old["test"])) == set(EXPOSED[fold])
              and labels[add].shape[0] == len(add), "Fold membership or labels differ")
    h.require([int(labels[i]) for i in add] == CONFIG["new_labels"][str(fold)], "Added patient labels differ")
    before_rng = h.rng_sha()
    e69 = h.runner().load_history().load_e69(); e61 = e69.load_e61()
    pairs, same = e61.make_outer_pairs(fold, state["supports"], new_train, labels)
    h.require(h.rng_sha() == before_rng, "Pair generation changed global RNG")
    counts = np.bincount(labels[new_train], minlength=2)
    weights = (len(new_train) / counts / 2).astype(np.float32)
    h.require(pairs.shape == (4096, 2) and np.isin(pairs, new_train).all()
              and np.array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]])
              and np.bincount(same, minlength=2).tolist() == [2048, 2048]
              and (pairs[:, 0] != pairs[:, 1]).all(), "Relation pairs invalid")
    dev = state["y"].device
    state.update(train_np=new_train, train=torch.as_tensor(new_train, dtype=torch.long, device=dev),
                 weights=torch.from_numpy(weights).to(dev), train_pairs_np=pairs, train_same_np=same,
                 train_pairs=torch.as_tensor(pairs, dtype=torch.long, device=dev),
                 train_same=torch.as_tensor(same, dtype=torch.float32, device=dev),
                 pair_sha=e61.pair_sha(pairs, same), effective_train_sha=e69.index_sha(new_train),
                 added_train_rows_1based=[i + 1 for i in ADDED[fold]],
                 added_train_count=len(ADDED[fold]),
                 train_test_overlap_1based=[i + 1 for i in EXPOSED[fold]], pair_rng_unchanged=True)
    legal = old["legal"]
    state.update(legal_np=legal, legal=torch.as_tensor(legal, dtype=torch.long, device=dev))
    state["excluded"] = list(EXPOSED[fold])
    module.spectrum.allowed[:] = True
    module.spectrum.allowed[list(EXPOSED[fold])] = False
    reference, hard_initial = ctx["reference"], ctx["hard_initial"]
    module.statistics = evidence.AgreementEvidence(reference, ctx["original"]).to(dev)
    labels_t = torch.from_numpy(labels[legal]).to(dev)
    now_identity = ext.identity_snapshot(state, module, optimizer)
    h.require(now_identity["new_parameters"] == old_identity["new_parameters"]
              and now_identity["new_optimizer"] == old_identity["new_optimizer"]
              and now_identity["historical_rng"] == old_identity["historical_rng"]
              and now_identity["transductive_inputs"] == old_identity["transductive_inputs"],
              "Initialization or protected inputs changed")
    reg = {"task_group": GROUP, "fold": fold, "before_train": len(old["train"]), "after_train": len(new_train),
           "before_legal": len(old["legal"]), "after_legal": len(legal), "test": len(old["test"]),
           "original_train_index_sha": e69.index_sha(old["train"]), "extended_train_index_sha": e69.index_sha(new_train),
           "original_test_index_sha": e69.index_sha(old["test"]), "extended_pair_sha": e61.pair_sha(pairs, same),
           "legal_index_sha": h.sha(legal), "reference_sha": h.sha(reference),
           "initial_ema_sha": h.sha(parent.ema(ctx)), "overlap_1based": [i + 1 for i in EXPOSED[fold]],
           "excluded_0based": list(state["excluded"]), "after_class_counts_asd_td": counts.tolist(),
           "after_class_weights_asd_td": weights.tolist(), "relation_count": len(pairs),
           "same_class_pairs": int(same.sum()), "different_class_pairs": int((same == 0).sum()),
           "per_overlap_pair_count": {str(i + 1): int((pairs == i).sum()) for i in EXPOSED[fold]},
           "legal_labels_sha": h.sha(labels_t.cpu()), "native_rng_unchanged": True}
    provenance = dict(original, membership=h.sha(h.archive.membership(state)), legal_index_sha=h.sha(legal),
                      memory_sha=h.sha(hard_initial.cpu()), reference=h.sha(reference),
                      initial_module=h.sha(module.state_dict()), source_membership_sha=original["membership"],
                      source_reference_sha=original["reference"], registration=reg)
    return {**ctx, "task_group": GROUP, "old_members": old, "members": h.archive.membership(state),
            "provenance": provenance, "reference": reference, "labels": labels_t,
            "hard_initial": hard_initial}

def checked_mask(capture, module, fold):
    rows = list(EXPOSED[fold]); details = capture["details"]
    h.require(torch.equal(torch.where(~module.spectrum.allowed)[0], torch.as_tensor(sorted(rows), device=module.spectrum.allowed.device))
              and all(torch.count_nonzero(details[k][rows]).item() == 0 for k in MASKED_COMPONENTS)
              and torch.equal(capture["enhanced"][rows], capture["cam"][rows]), "Masked residual changed")

def engine(task_group, fold):
    run = prior.load_parent_module(f"_addition_training_{fold}", "run.py")
    run.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    run.ROOT, run.RUNTIME, run.RESULTS, run.MANIFEST = ROOT, RUNTIME, RESULTS, MANIFEST
    run.paths = lambda group, fd: paths(task_group, fd)
    run.setup = lambda group, fd, **kw: setup(task_group, fd, **kw)
    base_meta, base_forward, base_transition, base_advance = run.metadata, run.forward, run.transition, run.observation_and_advance
    receipt = None
    def metadata(ctx, hashes):
        return {**base_meta(ctx, hashes), "experiment": CONFIG["experiment"],
                "native_membership": ctx["provenance"]["registration"]}
    def forward(state, training, module=None, intervention="full"):
        out, cap = base_forward(state, training, module, intervention)
        if module is not None:
            checked_mask(cap, module, fold)
            if receipt is not None: receipt.append("train" if training else "eval")
        return out, cap
    def advance(ctx, capture):
        h.require(len(ctx["state"]["legal"]) == len(ctx["labels"]) and
                  torch.equal(ctx["module"].statistics.legal, ctx["state"]["legal"]), "EMA membership changed")
        return base_advance(ctx, capture)
    def transition(ctx, epoch, dropout, *, inspect=False):
        nonlocal receipt
        receipt = []
        try:
            result, diag, rng, previous, observed = base_transition(ctx, epoch, dropout, inspect=inspect)
            h.require(receipt == ["train", "eval"], "Residual scope was not checked in train/eval")
            diag["routing"]["scope_receipt"] = {"phases": receipt, "fold": fold,
                                                     "overlap_1based": [i + 1 for i in EXPOSED[fold]],
                                                     "masked": True, "legal_rows": len(ctx["state"]["legal"])}
            return result, diag, rng, previous, observed
        finally:
            receipt = None
    run.metadata, run.forward, run.transition, run.observation_and_advance = metadata, forward, transition, advance
    return run
