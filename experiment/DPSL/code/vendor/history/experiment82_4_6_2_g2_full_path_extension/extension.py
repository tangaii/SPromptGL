"""Expand two historical folds and use the unmodified SCMR training loop."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys

from standalone_paths import DPSL_ROOT

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
RUNTIME = ROOT / "runtime"
RESULTS = ROOT / "results"
CONFIG = json.loads((ROOT / "config.json").read_bytes())
PRIOR = DPSL_ROOT / "code" / "vendor" / "history" / "experiment82_4_6_1_g2_overlap_unmask"
GROUP = CONFIG["group"]
ADDED = {int(k): tuple(x - 1 for x in v) for k, v in CONFIG["added_training_rows_by_test_fold"].items()}
FOLDS = tuple(sorted(ADDED))
OLD_EXPOSED = {int(k): tuple(x - 1 for x in v) for k, v in CONFIG["historical_overlap_by_test_fold"].items()}
EXPOSED = {fold: (*OLD_EXPOSED[fold], *ADDED[fold]) for fold in FOLDS}

sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
try:
    import overlap as prior
    import independent as prior_audit
finally:
    sys.path.remove(str(PRIOR))

import numpy as np
import torch

h, parent = prior.h, prior.parent
CFG, PARENT = prior.CFG, prior.PARENT
spectral, optimization, evidence = prior.spectral, prior.optimization, prior.evidence
parent_preflight = prior.parent_preflight
f = prior_audit.f


def implementation():
    return {path.name: h.sha_file(path) for path in sorted([ROOT / "config.json", *ROOT.glob("*.py")])}


def source_manifests():
    base = prior.original_manifest()
    file = PRIOR / "manifest.json"
    older = h.read_json(file)
    h.require(h.sha_file(file) == CONFIG["prior_manifest_sha256"] and older["status"] == "COMPLETE_AUDITED" and
              h.sha(older["implementation"]) == older["implementation_sha"] == CONFIG["prior_frozen_sha"] and
              prior.implementation() == older["implementation"], "E82.4.6.1 frozen source differs")
    h.require(all(item["exact"] and item["remote_blob"] == item["local_blob"] and
                  h.sha_file(Path(item["path"])) == item["server_sha256"] for item in CONFIG["source_verification"]),
              "A verified GitHub source identity changed")
    return base, older


def paths(group, fold):
    h.require(group == GROUP and fold in FOLDS, "Only G2 Fold 2 and Fold 3 may train or resume here")
    name = f"{group}_f{fold:02d}"
    return {"best": ROOT / "checkpoints" / f"{name}.best.pt",
            "latest": ROOT / "checkpoints" / f"{name}.latest.pt",
            "journal": RUNTIME / f"{name}.jsonl", "task": RUNTIME / f"{name}.training.json",
            "preflight": RUNTIME / f"{name}.preflight.json"}


def before_context(fold):
    if fold == prior.FOLD:
        ctx = prior.setup(GROUP, fold)
        archived = h.read_json(prior.paths(GROUP, fold)["preflight"])
    else:
        ctx = parent.setup(GROUP, fold)
        archived = h.read_json(DPSL_ROOT / "runtime" / "history" / f"fold{fold:02d}.preflight.json")
    return ctx, archived


def identity_snapshot(state, module, optimizer):
    return {"native_bundle": h.sha(h.base_bundle(state)),
            "native_state_model": h.runner().load_history().state_model_sha(state),
            "optimizer_states": {key: h.sha(state[key].state_dict()) for key in h.OPTS},
            "rng": h.sha(h.base().pack_rng()), "historical_rng": h.rng_sha(),
            "new_parameters": prior.parameter_sha(module), "new_optimizer": h.sha(optimizer.state_dict()),
            "spectrum_and_prompts": h.sha(module.spectrum.state_dict()),
            "transductive_inputs": h.sha((state["x"], state["supports"], state["fallback"])),
            "dataset_labels": h.sha(state["labels_np"])}


def extend_native(state, old, fold):
    e69 = h.runner().load_history().load_e69()
    e61 = e69.load_e61()
    labels = state["labels_np"]
    add = np.asarray(ADDED[fold], dtype=np.int64)
    new_train = np.concatenate((old["train"], add))
    want = CONFIG["fold_sizes"][str(fold)]
    h.require(len(old["train"]) == want["before_train"] and len(old["legal"]) == want["before_legal"] and
              np.array_equal(old["train"], old["legal"]) and len(old["test"]) == want["test"] and
              len(new_train) == len(np.unique(new_train)) == want["after_train"] and
              np.isin(add, old["test"]).all() and not np.isin(add, old["train"]).any() and
              np.intersect1d(old["train"], old["test"]).tolist() == list(OLD_EXPOSED[fold]) and
              np.intersect1d(new_train, old["test"]).tolist() == list(EXPOSED[fold]) and
              labels[add].tolist() == CONFIG["added_training_labels_by_test_fold"][str(fold)],
              f"Fold {fold} target membership or labels differ")

    # This is the E69.3/E61 ordered outer pair sampler and keeps its local seed.
    before = h.rng_sha()
    pairs, same = e61.make_outer_pairs(fold, state["supports"], new_train, labels)
    h.require(h.rng_sha() == before, "Relation sampling advanced the historical RNG")
    counts = np.bincount(labels[new_train], minlength=2)
    prior_counts = np.bincount(labels[old["train"]], minlength=2)
    weights = (len(new_train) / counts / 2).astype(np.float32)
    h.require(np.array_equal(old["weights"].numpy(), (len(old["train"]) / prior_counts / 2).astype(np.float32)),
              "The historical class-weight order is not ASD=0, TD=1")
    h.require(pairs.shape == (4096, 2) and np.isin(pairs, new_train).all() and
              np.array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]]) and
              np.bincount(same, minlength=2).tolist() == [2048, 2048] and
              (pairs[:, 0] != pairs[:, 1]).all() and len(np.unique(np.sort(pairs, axis=1), axis=0)) == 4096,
              "The rebuilt training relation pairs failed validation")
    device = state["y"].device
    pair_sha = e61.pair_sha(pairs, same)
    train = torch.as_tensor(new_train, dtype=torch.long, device=device)
    state.update(train_np=new_train, train=train, weights=torch.from_numpy(weights).to(device),
                 train_pairs_np=pairs, train_same_np=same,
                 train_pairs=torch.as_tensor(pairs, dtype=torch.long, device=device),
                 train_same=torch.as_tensor(same, dtype=torch.float32, device=device),
                 pair_sha=pair_sha, effective_train_sha=e69.index_sha(new_train),
                 added_train_rows_1based=[i + 1 for i in EXPOSED[fold]],
                 added_train_count=len(EXPOSED[fold]), train_test_overlap_1based=[i + 1 for i in EXPOSED[fold]],
                 pair_rng_unchanged=True, legal_np=new_train, legal=train, excluded=[])
    return {
        "fold": fold, "before_train": len(old["train"]), "before_legal": len(old["legal"]),
        "after_train": len(new_train), "after_legal": len(state["legal_np"]), "test": len(old["test"]),
        "original_train_index_sha": e69.index_sha(old["train"]), "extended_train_index_sha": e69.index_sha(new_train),
        "original_test_index_sha": e69.index_sha(old["test"]),
        "original_test_label_sha": h.sha(labels[old["test"]]),
        "original_pair_sha": e61.pair_sha(old["train_pairs"], old["train_same"]), "extended_pair_sha": pair_sha,
        "legal_index_sha": h.sha(new_train), "relation_count": len(pairs),
        "same_class_pairs": int(same.sum()), "different_class_pairs": int((same == 0).sum()),
        "pair_rng_seed": 532100 + fold, "main_rng_unchanged": True,
        "before_class_counts_asd_td": prior_counts.tolist(), "after_class_counts_asd_td": counts.tolist(),
        "before_class_weights_asd_td": old["weights"].tolist(), "after_class_weights_asd_td": weights.tolist(),
        "previous_overlap_1based": [i + 1 for i in OLD_EXPOSED[fold]],
        "added_1based_in_order": [int(x + 1) for x in add], "overlap_1based": [i + 1 for i in EXPOSED[fold]],
        "overlap_labels": {str(i + 1): int(labels[i]) for i in EXPOSED[fold]},
        "per_overlap_pair_count": {str(i + 1): int((pairs == i).sum()) for i in EXPOSED[fold]},
    }


def setup(group, fold):
    h.require(group == GROUP and fold in FOLDS, "Modified membership is confined to the two registered tasks")
    ctx, _ = before_context(fold)
    old, original = ctx["members"], ctx["provenance"]
    state, module, optimizer = (ctx[k] for k in ("state", "module", "optimizer"))
    old_sign = identity_snapshot(state, module, optimizer)
    h.require(old_sign["historical_rng"] == ctx["warmup"]["rng_sha"] and
              old_sign["native_state_model"] == original["identity"]["model_cam_sha"] and
              module.spectrum.allowed.all().item(),
              "Expected exact historical initialization and the inherited allowance")
    trail = extend_native(state, old, fold)
    members = h.archive.membership(state)
    labels = state["labels_np"][members["legal"]].copy()
    y = torch.from_numpy(labels).to(state["y"].device)
    reference = prior.rebuild_reference(fold, ctx["warm_cam"], ctx["original"], members["legal"], labels)

    # Preserve archived difficulty values for the ordered prefix. Initialize only appended rows.
    with torch.no_grad():
        rows = state["legal"][len(old["legal"]):]
        logits = spectral.fixed_logits(state["cam"].Outputlayer, ctx["warm_cam"].to(y.device)[rows])
        difficulty = (-(1 - 2 * y[len(old["legal"]):]) * spectral.margin(logits) /
                      CFG["boundary"]["error_temperature"]).sigmoid()
        difficulty = torch.cat((ctx["hard_initial"], difficulty))
        module.spectrum.allowed[list(EXPOSED[fold])] = True
    module.statistics = evidence.AgreementEvidence(reference, ctx["original"]).to(y.device)
    new_sign = identity_snapshot(state, module, optimizer)
    h.require(new_sign == old_sign and np.array_equal(old["test"], members["test"]) and
              torch.equal(y, state["y"][state["legal"]]) and difficulty.isfinite().all().item() and
              module.spectrum.allowed[list(EXPOSED[fold])].all().item() and
              h.sha(difficulty[:len(old["legal"])].cpu()) == original["memory_sha"],
              "The extension changed initial weights, optimizer, RNG, supports, immutable labels or new-path alignment")
    h.require(set(module.parameters()) == {p for family in optimization.NEW for p in ctx["families"][family]} ==
              {p for group in optimizer.param_groups for p in group["params"]}, "New parameter ownership changed")
    trail.update(all_required_residuals_allowed=True, original_o_sha=original["spectrum"],
                 original_o_u16_sha=original["basis"], medical_prompt_sha=original["prompt"],
                 restored_initial_identity=old_sign, legal_labels_sha=h.sha(y.cpu()))
    provenance = dict(original, membership=h.sha(members), legal_index_sha=h.sha(members["legal"]),
                      memory_sha=h.sha(difficulty.cpu()), reference=h.sha(reference),
                      initial_module=h.sha(module.state_dict()), source_membership_sha=original["membership"],
                      source_reference_sha=original["reference"], registration=trail)
    return {**ctx, "old_members": old, "original_provenance": original, "provenance": provenance, "members": members,
            "labels": y, "hard_initial": difficulty, "reference": reference}


def patient_path_gradients(rows, capture, families, full):
    d = capture["details"]
    derivative, = torch.autograd.grad(full, d["delta"], retain_graph=True)
    parameters = [p for name in optimization.NEW for p in families[name]]
    prompt_count = len(families["prompt"])
    result = {}
    for row in rows:
        upstream = torch.zeros_like(derivative)
        upstream[row] = derivative[row]
        g = torch.autograd.grad(d["delta"], (*parameters, d["raw_parallel"], d["raw_perp"]),
                                grad_outputs=upstream, retain_graph=True, allow_unused=True)
        norms = {"full_ce_delta_l2": float(derivative[row].norm()), "prompt_l2": optimization.norm(g[:prompt_count]),
                 "control_l2": optimization.norm(g[prompt_count:-2]),
                 "spectral_candidate_l2": float(g[-2][row].norm()), "complement_candidate_l2": float(g[-1][row].norm())}
        h.require(np.isfinite(list(norms.values())).all() and
                  all(v > 0 for k, v in norms.items() if k != "prompt_l2"),
                  f"Patient {row + 1} lacks a production Full CE gradient")
        result[str(row + 1)] = norms
    return result


def engine(fold):
    run = prior.load_parent_module(f"_extension_training_{fold}", "run.py")
    run.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    run.ROOT, run.RUNTIME, run.RESULTS, run.MANIFEST = ROOT, RUNTIME, RESULTS, MANIFEST
    run.paths, run.setup = paths, setup
    metadata = run.metadata

    def registered_metadata(ctx, hashes):
        return {**metadata(ctx, hashes), "experiment": CONFIG["experiment"],
                "registration": ctx["provenance"]["registration"], "prior_frozen_sha": CONFIG["prior_frozen_sha"]}

    def inspect_paths(families, capture, full):
        result = optimization.inspect_paths(families, capture, full)
        result["exposed_patient_gradients"] = patient_path_gradients(EXPOSED[fold], capture, families, full)
        return result

    run.metadata, run.inspect_paths = registered_metadata, inspect_paths
    return run
