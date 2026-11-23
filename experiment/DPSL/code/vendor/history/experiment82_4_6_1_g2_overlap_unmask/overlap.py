"""Only the Fold 2 supervision and evidence membership differ from the frozen parent."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
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
PARENT = DPSL_ROOT / "code" / "history_base"
GROUP, FOLD = CONFIG["group"], CONFIG["fold"]
EXPOSED = tuple(value - 1 for value in CONFIG["exposed_1based"])
sys.dont_write_bytecode = True

# Import the verified parent as a read-only package. Keep its globals for historical replays.
sys.path.insert(0, str(PARENT))
try:
    import run as parent
    import preflight as parent_preflight
    from common import history as h
    from common import evidence
    from common import spectral
    from common import optimization
finally:
    sys.path.remove(str(PARENT))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

CFG = h.CFG
GROUP_LABELS = np.asarray(CONFIG["exposed_labels"])
E = evidence.E


def implementation():
    return {p.name: h.sha_file(p) for p in sorted([ROOT / "config.json", *ROOT.glob("*.py")])}


def original_manifest():
    original = h.read_json(PARENT / "manifest.json")
    h.require(original["status"] == "COMPLETE_AUDITED" and
              h.sha(original["implementation"]) == original["implementation_sha"] ==
              CONFIG["parent_implementation_sha"], "Parent is not the committed independently audited experiment")
    h.require(h.implementation() == original["implementation"] and CFG == original["config"],
              "Parent frozen implementation or settings differ")
    for row in CONFIG["parent_remote_checks"]:
        h.require(row["exact"] and h.sha_file(PARENT / row["path"]) == row["server"] == row["remote"],
                  f"Remote provenance does not match the authority: {row['path']}")
    return original


def paths(group=GROUP, fold=FOLD):
    h.require((group, fold) == (GROUP, FOLD), "Only G2 Fold 2 may write or resume training here")
    name = f"{group}_f{fold:02d}"
    return {"best": ROOT / "checkpoints" / f"{name}.best.pt",
            "latest": ROOT / "checkpoints" / f"{name}.latest.pt",
            "journal": RUNTIME / f"{name}.jsonl", "task": RUNTIME / "training.json",
            "preflight": RUNTIME / "preflight.json"}


def scope(ctx):
    m = ctx["members"]
    return {"native_train": len(m["train"]), "new_train": len(m["legal"]), "test": len(m["test"]),
            "legal_index_sha": h.sha(m["legal"]), "legal_labels_sha": h.sha(ctx["labels"].detach().cpu()),
            "exposed_1based": list(CONFIG["exposed_1based"]),
            "exposed_labels": list(CONFIG["exposed_labels"])}


def rebuild_reference(fold, warm_cam, spectrum, legal, labels):
    """Use the original moment kernel on the rebuilt historical stratified assignment."""
    seed = E["archived_split_seed_base"] + fold
    points = ((warm_cam[legal].double() - spectrum["mu"]) @ spectrum["U"]) / (
        spectrum["lambda"] + spectral.EPS).sqrt()
    splitter = StratifiedKFold(n_splits=E["inner_folds"], shuffle=True,
                               random_state=np.random.RandomState(seed))
    group = np.zeros(len(legal), dtype=np.int64)
    for i, (_, held) in enumerate(splitter.split(points.numpy(), labels), 1):
        group[held] = i
    n_banks = E["inner_folds"] + 1
    mask = ((group[None, :] != np.arange(n_banks)[:, None])[:, None, :] &
            (labels[None, None, :] == np.arange(2)[None, :, None]))
    counts = torch.from_numpy(mask.sum(-1))
    h.require(group.min() == 1 and group.max() == E["inner_folds"] and bool((counts > 16).all()),
              "The stratified Gaussian banks lack class members")
    samples = [points[m] for m in mask.reshape(2 * n_banks, -1)]
    mean = torch.stack([x.mean(0) for x in samples]).reshape(n_banks, 2, 16)
    variance = torch.stack([x.var(0, correction=E["ddof"]) for x in samples]).reshape_as(mean)
    _, stable = evidence.stabilize(variance, counts)
    oof = evidence.gaussian(points, mean[group], stable[group])[1]
    y = torch.from_numpy(labels)
    signed = (1 - 2 * y[:, None]) * oof
    skill = 0.5 * (signed[y == 0].div(2).tanh().mean(0) + signed[y == 1].div(2).tanh().mean(0))
    calibration = 0.5 * (1 + skill)
    bank = np.zeros(CFG["subjects"], dtype=np.int64)
    bank[legal] = group
    biased = variance * ((counts[..., None] - E["ddof"]) / counts[..., None].double())
    second = biased + mean.square()
    h.require(all(bool(x.isfinite().all()) for x in (mean, second, stable, calibration)) and bool((stable > 0).all()),
              "Nonfinite or nonpositive Gaussian moments")
    return {"legal": legal, "labels": labels, "group": group, "seed": seed, "counts": counts, "mask": mask,
            "bank": torch.from_numpy(bank), "mean": mean, "second": second,
            "stable": stable, "calibration": calibration, "h": points}


def parameter_sha(module):
    return h.sha({name: value.detach() for name, value in module.named_parameters()})


def setup(group=GROUP, fold=FOLD):
    h.require((group, fold) == (GROUP, FOLD), "Modified membership is confined to G2 Fold 2")
    ctx = parent.setup(group, fold)
    state = ctx["state"]
    old, original = ctx["members"], ctx["provenance"]
    full_train = old["train"].copy()
    prior_legal = old["legal"]
    exposed = list(EXPOSED)
    h.require(len(full_train) == len(np.unique(full_train)) == CONFIG["historical_train_size"] and
              len(old["test"]) == CONFIG["test_size"] and len(prior_legal) == CONFIG["original_new_path_train_size"]
              and state["excluded"] == exposed
              and np.array_equal(np.intersect1d(full_train, old["test"]), EXPOSED)
              and np.array_equal(full_train[~np.isin(full_train, EXPOSED)], prior_legal)
              and np.array_equal(state["labels_np"][exposed], GROUP_LABELS),
              "Fold 2 historical overlap identity or labels differ")

    initial_rng = h.rng_sha()
    rng_state_before = h.base().pack_rng()
    native_before = h.sha(h.base_bundle(state))
    module = ctx["module"]
    model_init = parameter_sha(module)
    optimizer_init = h.sha(ctx["optimizer"].state_dict())
    frozen_before = h.sha({k: v for k, v in module.spectrum.state_dict().items() if k != "allowed"})
    h.require(not module.spectrum.allowed[exposed].any().item() and
              not tuple(module.statistics.parameters()), "Unexpected parent mask or learnable Gaussian bank")

    labels = state["labels_np"][full_train].copy()
    reference = rebuild_reference(fold, ctx["warm_cam"], ctx["original"], full_train, labels)
    device = state["x"].device
    state.update(legal_np=full_train, legal=torch.as_tensor(full_train, device=device), excluded=[])
    y = torch.from_numpy(labels).to(device)
    with torch.no_grad():
        allowed = module.spectrum.allowed.clone()
        module.spectrum.allowed[exposed] = True
        h.require(torch.where(allowed != module.spectrum.allowed)[0].tolist() == exposed,
                  "Unexpected additional residual mask change")
        logits = spectral.fixed_logits(state["cam"].Outputlayer, ctx["warm_cam"].to(device)[state["legal"]])
        difficulty = (-(1 - 2 * y) * spectral.margin(logits) / CFG["boundary"]["error_temperature"]).sigmoid()
        shared = torch.from_numpy(~np.isin(full_train, EXPOSED)).to(device)
        difficulty[shared] = ctx["hard_initial"]
    module.statistics = evidence.AgreementEvidence(reference, ctx["original"]).to(device)
    members = h.archive.membership(state)
    h.require(all(h.sha(members[k]) == h.sha(old[k]) for k in ("train", "test", "train_pairs", "train_same", "weights"))
              and old["train_pairs"].shape == (4096, 2), "Historical members, weights or pair order changed")
    h.require(torch.equal(state["y"][state["legal"]], y) and difficulty.shape == y.shape
              and torch.isfinite(difficulty).all().item(), "Supervision label or difficulty alignment differs")
    h.require(h.rng_sha() == initial_rng and h.sha(rng_state_before) == h.sha(h.base().pack_rng())
              and h.sha(h.base_bundle(state)) == native_before and parameter_sha(module) == model_init
              and h.sha(ctx["optimizer"].state_dict()) == optimizer_init and
              h.sha({k: v for k, v in module.spectrum.state_dict().items() if k != "allowed"}) == frozen_before,
              "Unmasking changed historical state, RNG, initialized parameters or the Original O/Prompt")
    bound_params = set(module.parameters())
    h.require(bound_params == {p for k in optimization.NEW for p in ctx["families"][k]}
              == {p for pg in ctx["optimizer"].param_groups for p in pg["params"]},
              "New module parameter ownership changed")

    provenance = dict(original, membership=h.sha(members), legal_index_sha=h.sha(full_train),
                      memory_sha=h.sha(difficulty.cpu()), reference=h.sha(reference),
                      initial_module=h.sha(module.state_dict()), initialized_parameters=model_init,
                      initial_new_optimizer=optimizer_init, original_membership_sha=original["membership"],
                      original_reference_sha=original["reference"], original_legal_sha=h.sha(prior_legal),
                      shared_memory_sha=h.sha(difficulty[shared].cpu()),
                      weights_sha=h.sha(old["weights"]), native_bundle_sha=native_before,
                      frozen_spectrum_sha=frozen_before, overlap_1based=CONFIG["exposed_1based"])
    ctx.update(members=members, old_members=old, original_provenance=original, provenance=provenance,
               labels=y, hard_initial=difficulty, reference=reference)
    return ctx


def load_parent_module(name, filename):
    spec = spec_from_file_location(name, PARENT / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    h.require(Path(module.__file__).resolve() == PARENT / filename, "Unexpected imported source path")
    return module


def patient_path_gradients(capture, families, full):
    """VJP for the actual class-balanced criterion, restricted to one patient's residual."""
    d = capture["details"]
    derivative, = torch.autograd.grad(full, d["delta"], retain_graph=True)
    params = [p for name in optimization.NEW for p in families[name]]
    records = {}
    for index in EXPOSED:
        route = torch.zeros_like(derivative)
        route[index] = derivative[index]
        nodes = [*params, d["raw_parallel"], d["raw_perp"]]
        grads = torch.autograd.grad(d["delta"], nodes, grad_outputs=route, retain_graph=True, allow_unused=True)
        split = len(families["prompt"])
        norms = {"full_ce_delta_l2": float(derivative[index].norm()),
                 "prompt_l2": optimization.norm(grads[:split]),
                 "control_l2": optimization.norm(grads[split:-2]),
                 "spectral_candidate_l2": float(grads[-2][index].norm()),
                 "complement_candidate_l2": float(grads[-1][index].norm())}
        h.require(all(np.isfinite(list(norms.values()))) and norms["full_ce_delta_l2"] > 0 and
                  norms["control_l2"] > 0, f"Exposed patient {index + 1} has no classification gradient")
        records[str(index + 1)] = norms
    return records


def inspect_paths(families, capture, full):
    info = optimization.inspect_paths(families, capture, full)
    info["exposed_patient_gradients"] = patient_path_gradients(capture, families, full)
    return info


def engine():
    # Use the original train loop and checkpoint code with a process-local artifact root.
    # The separate parent imports retain their own setup, identities and paths.
    module = load_parent_module("_overlap_training", "run.py")
    module.h = SimpleNamespace(**dict(vars(h), implementation=implementation))
    module.ROOT, module.RESULTS, module.RUNTIME, module.MANIFEST = ROOT, RESULTS, RUNTIME, MANIFEST
    module.paths, module.setup, module.inspect_paths = paths, setup, inspect_paths
    checkpoint_metadata = module.metadata

    def identified_metadata(ctx, hashes):
        return {**checkpoint_metadata(ctx, hashes), "experiment": CONFIG["experiment"],
                "membership_change": scope(ctx), "parent_implementation_sha": CONFIG["parent_implementation_sha"]}

    module.metadata = identified_metadata
    return module
