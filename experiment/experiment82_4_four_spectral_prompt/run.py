"""Train the four complete E82.4 methods through the original DARG entry point."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from common import history as h
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from common.references import balanced_mean, fit_crossfit, frozen_head, margin
from common.spectral import smooth_limit
from G1_dpr_psce.model import DprPsce
from G2_hlg_psce.model import HlgPsce
from G3_rcs_da_srpe.model import RcsDaSrpe
from G4_sot_da_srpe.model import SotDaSrpe

CFG, HERE = h.CFG, h.HERE
RUNTIME, RESULTS = HERE / "runtime", HERE / "results"
MANIFEST = HERE / "manifest.json"
START = CFG["start_epoch"]
LOSS_KEYS = ("cls", "graph", "pair", "adv", "preserve", "total")
FOLDS = range(1, CFG["folds"] + 1)
METHODS = dict(G1=DprPsce, G2=HlgPsce, G3=RcsDaSrpe, G4=SotDaSrpe)


def paths(group, fold):
    stem = f"{group}_f{fold:02d}"
    directory = HERE / CFG["group_dirs"][group]
    return {
        "best": directory / "checkpoints" / f"{stem}.best.pt",
        "latest": directory / "checkpoints" / f"{stem}.latest.pt",
        "pending": directory / "checkpoints" / f"{stem}.best.pending",
        "journal": directory / "runtime/logs" / f"{stem}.jsonl",
        "task": directory / "runtime/tasks" / f"{stem}.json",
        "preflight": directory / "runtime/preflight" / f"{stem}.json",
        "audit": directory / "runtime/audit" / f"{stem}.json",
    }


def forward(state, training, module=None, *, query=None, semantic_off=False, variant=None):
    """Substitute CAM144 in the native OutputLayer pre-hook, preserving its one call."""
    head = state["cam"].Outputlayer
    capture = {}

    def inject(_head, inputs):
        h.require("cam" not in capture and tuple(inputs[0].shape) == (CFG["subjects"], 144), "CAM144 entry changed")
        capture["cam"] = inputs[0]
        if module is not None:
            enhanced, details = module(inputs[0], crossfit=training, query=query, semantic_off=semantic_off, variant=variant)
            capture.update(enhanced=enhanced, details=details)
            return enhanced, *inputs[1:]

    def read_logits(_linear, _inputs, output):
        capture["logits"] = output

    handles = [head.register_forward_pre_hook(inject), head.classifier.register_forward_hook(read_logits)]
    try:
        with torch.set_grad_enabled(training):
            out = h.runner().forward(state, h.BASE, training=training)
    finally:
        for hook in handles:
            hook.remove()
    h.require("cam" in capture and "logits" in capture, "Historical OutputLayer was not reached")
    return out, capture



def measure_gradient_path(state, module, out, capture):
    """Use the real CE graph and isolate its VJP through Delta C."""
    families = {
        "head": list(state["cam"].Outputlayer.parameters()),
        "cam": [p for name, p in state["cam"].named_parameters()
                if name.startswith(("InputLayer.", "encode_", "feed_")) and p.requires_grad],
        "gfr": list(state["model"].base_parameters()),
        "darg": [p for name, p in state["model"].added_named_parameters()
                 if p.requires_grad and name.startswith(("encoders.", "scorers.", "adaptive_gate_logits"))],
        "prompt": list(module.gradient_groups()["prompt"]),
        "residual": list(module.gradient_groups()["residual"]),
    }
    requested = [p for group in families.values() for p in group]
    all_gradients = torch.autograd.grad(out["cls"], [*requested, capture["cam"], capture["enhanced"]],
                                        retain_graph=True, allow_unused=True)

    def norm(values):
        pieces = [g.detach().norm().square() for g in values if g is not None]
        return float(torch.stack(pieces).sum().sqrt()) if pieces else 0.0

    result = {}
    offset = 0
    for name, group in families.items():
        result[name] = norm(all_gradients[offset:offset + len(group)])
        offset += len(group)
    result["cam144"], result["enhanced"] = [norm([g]) for g in all_gradients[-2:]]
    vjp_results = torch.autograd.grad(
        capture["details"]["delta"], [capture["cam"], *families["cam"], *families["gfr"], *families["darg"]],
        grad_outputs=all_gradients[-1].detach(), retain_graph=True, allow_unused=True,
    )
    offset = 1
    result["delta_to_cam144"] = norm(vjp_results[:1])
    for name in ("cam", "gfr", "darg"):
        count = len(families[name])
        result["delta_to_" + name] = norm(vjp_results[offset:offset + count])
        offset += count
    branch_gradients = all_gradients[-2 - len(families["prompt"]) - len(families["residual"]):-2]
    result["nonzero_branch_tensors"] = sum(g is not None and bool(g.abs().max() > 0) for g in branch_gradients)
    result["branch_tensors"] = len(branch_gradients)
    h.require(all(math.isfinite(v) and v > 0 for v in result.values())
              and result["nonzero_branch_tensors"] == result["branch_tensors"],
              f"True classification gradient lost a path: {result}")
    return result



def grad_norm(values):
    return torch.stack([grad.detach().square().sum() for grad in values]).sum().sqrt()



def limit_auxiliary(module, auxiliary, total, coefficients, *, watch_gradients=False):
    groups = module.gradient_groups()
    parameters = [p for group in groups.values() for p in group]
    contributions = {
        key: torch.autograd.grad(value * coefficients[key]["coefficient"], parameters, retain_graph=True, allow_unused=True)
        for key, value in auxiliary.items()
    }
    total.backward()
    stats, offset = {}, 0
    with torch.no_grad():
        for name, group in groups.items():
            h.require(all(p.grad is not None for p in group), f"Primary CE disconnected from {name}")
            primary = grad_norm([p.grad for p in group])
            zero = torch.zeros_like(primary)
            stats[f"{name}_primary"] = primary
            limit = primary.clamp_min(1e-30)
            extra = [torch.zeros_like(p) for p in group]
            for key, gradients in contributions.items():
                grads = gradients[offset:offset + len(group)]
                present = [g for g in grads if g is not None]
                raw = grad_norm(present) if present else zero
                coefficient = (coefficients[key]["gradient_cap"] * primary / raw.clamp_min(1e-30)).clamp_max(1)
                stats[f"{name}_{key}_raw_norm"] = raw
                stats[f"{name}_{key}_raw_ratio"] = raw / limit
                stats[f"{name}_{key}_applied_ratio"] = raw * coefficient / limit
                for i, gradient in enumerate(grads):
                    if gradient is not None:
                        extra[i].add_(gradient * coefficient)
            stats[f"{name}_applied_ratio"] = grad_norm(extra) / limit
            for p, gradient in zip(group, extra):
                p.grad.add_(gradient)
            offset += len(group)
    if watch_gradients:
        h.require(set(parameters) == set(module.parameters()), "Auxiliary gradient budgets do not cover the complete branch")
    return stats



def train_step(ctx, inspect=False):
    state, module = ctx["state"], ctx["module"]
    optimizers = [*(state[name] for name in h.OPTS), ctx["optimizer"]]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    out, capture = forward(state, True, module)
    aux, change, protected = auxiliary_losses(ctx, out, capture)
    gradient_path = measure_gradient_path(state, module, out, capture) if inspect else None
    stats = limit_auxiliary(module, aux, out["total"], ctx["method"]["auxiliary"], watch_gradients=inspect)
    diag = step_diagnostics(out, capture, aux, stats, change, protected, ctx)
    all_parameters = [*h.parameter_names_p(state), *module.parameters()]
    finite = [torch.isfinite(p.grad).all() for p in all_parameters if p.grad is not None]
    h.require(bool(torch.stack(finite).all()), "Nonfinite optimization gradients")
    for optimizer in optimizers:
        optimizer.step()
    if inspect:
        h.require(bool(torch.stack([torch.isfinite(p).all() for p in all_parameters]).all()), "Nonfinite first updated parameter")
    return diag, gradient_path



@torch.no_grad()
def evaluate(ctx):
    state = ctx["state"]
    out, capture = forward(state, False, ctx["module"])
    rows = state["test_np"]
    logp = out["logp"][rows]
    probabilities = logp.exp().cpu().numpy()
    predictions = probabilities.argmax(-1)
    h.require(np.isfinite(probabilities).all(), "Nonfinite evaluation probabilities")
    return {
        "probability": probabilities, "prediction": predictions,
        "logits": capture["logits"][rows].cpu().numpy(), "correct": int((predictions == state["labels_np"][rows]).sum()),
    }



def state_dict(ctx):
    state = ctx["state"]
    return {
        "model": state["model"].state_dict(), "cam": state["cam"].state_dict(),
        **{key: state[key].state_dict() for key in h.OPTS},
        "branch": ctx["module"].state_dict(), "opt_branch": ctx["optimizer"].state_dict(),
    }



def restore(ctx, payload, implementation):
    state = ctx["state"]
    h.require(payload["experiment"] == CFG["experiment"] and payload["group"] == ctx["method"]["group"]
              and payload["fold"] == state["fold"] and payload["seed"] == 0
              and payload["config"] == CFG and payload["method"] == ctx["method"] and payload["implementation"] == implementation
              and payload["identity"] == ctx["identity"], "Checkpoint provenance differs")
    h.require(h.sha(payload["reference"]) == ctx["identity"]["reference_sha"]
              and h.sha(payload["membership"]) == ctx["identity"]["membership_sha"], "Checkpoint fixed references differ")
    bundle = payload["state"]
    for key in ("model", "cam"):
        state[key].load_state_dict(bundle[key], strict=True)
    for key in h.OPTS:
        state[key].load_state_dict(bundle[key])
    ctx["module"].load_state_dict(bundle["branch"], strict=True)
    ctx["optimizer"].load_state_dict(bundle["opt_branch"])
    h.base().restore_rng(payload["rng"])
    h.require(h.rng_sha() == payload["rng_sha"], "Checkpoint RNG does not restore exactly")



def native_signature(state, wrapped, dropout_calls):
    original_dropout = F.dropout

    def dropout(inputs, p=0.5, training=True, inplace=False):
        if training and 0 < p < 1:
            dropout_calls.append({"shape": list(inputs.shape), "stride": list(inputs.stride()),
                                  "dtype": str(inputs.dtype), "p": p, "inplace": inplace})
        return original_dropout(inputs, p=p, training=training, inplace=inplace)

    rng_before = h.rng_sha()
    for key in h.OPTS:
        state[key].zero_grad(set_to_none=True)
    F.dropout = dropout
    try:
        out = forward(state, True)[0] if wrapped else h.runner().forward(state, h.BASE, training=True)
    finally:
        F.dropout = original_dropout
    fields = (*LOSS_KEYS, "fused", "logp", "hidden", "original_adjs", "unified_adjs")
    output_sha = h.sha({key: out[key] for key in fields})
    out["total"].backward()
    gradient_sha = h.sha([p.grad for p in h.parameter_names_p(state)])
    for key in h.OPTS:
        state[key].step()
    rng_after_train = h.rng_sha()
    model_sha = h.runner().load_history().state_model_sha(state)
    optimizer_sha = {key: h.sha(state[key].state_dict()) for key in h.OPTS}
    with torch.no_grad():
        prediction = forward(state, False)[0]["logp"] if wrapped else h.runner().forward(state, h.BASE, training=False)["logp"]
    return {
        "forward_loss_sha": output_sha, "gradient_sha": gradient_sha, "model_sha": model_sha,
        "optimizer_sha": optimizer_sha, "evaluation_sha": h.sha(prediction),
        "rng": {"epoch": START + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": h.rng_sha()},
    }



def commit(paths_, journal, digest, payload, improved, best_sha):
    journal.flush()
    os.fsync(journal.fileno())
    if improved:
        h.atomic_torch(paths_["pending"], payload)
        best_sha = h.sha_file(paths_["pending"])
    payload = {**payload, "best_sha": best_sha, "journal_size": journal.tell(), "journal_sha": digest.hexdigest()}
    h.atomic_torch(paths_["latest"], payload)
    if improved:
        os.replace(paths_["pending"], paths_["best"])
    h.sync_directory(paths_["latest"].parent)
    return best_sha



def reconcile(paths_, journal, latest):
    expected = latest["best_sha"]
    if not paths_["best"].is_file() or h.sha_file(paths_["best"]) != expected:
        h.require(paths_["pending"].is_file() and h.sha_file(paths_["pending"]) == expected, "Committed Best is unavailable")
        os.replace(paths_["pending"], paths_["best"])
    paths_["pending"].unlink(missing_ok=True)
    journal.seek(0)
    raw = journal.read(latest["journal_size"])
    digest = hashlib.sha256(raw)
    h.require(len(raw) == latest["journal_size"] and digest.hexdigest() == latest["journal_sha"]
              and len(raw.splitlines()) == latest["next_epoch"] - START, "Latest journal prefix differs")
    journal.truncate()
    return digest



def train_fold(group, fold):
    manifest = h.read_json(MANIFEST)
    path = paths(group, fold)
    preflight = h.read_json(path["preflight"])
    h.require(manifest["preflight"]["status"] == "PASS" and preflight["status"] == "PASS"
              and preflight["implementation"] == manifest["implementation"] == h.implementation_hashes(), "Missing compatible preflight")
    ctx = setup(group, fold)
    h.require(ctx["identity"] == preflight["identity"], "Formal state differs from passing preflight")
    state = ctx["state"]
    archive = {row["epoch"]: row for row in h.historical_task(fold)["rng_trace"]}
    fixed = {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": CFG["seed"], "config": CFG, "method": ctx["method"],
        "identity": ctx["identity"], "reference": ctx["reference"],
        "membership": h.membership(state), "implementation": manifest["implementation"],
        "protected_sha": manifest["protected_bundle_sha"],
    }
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    path["journal"].parent.mkdir(parents=True, exist_ok=True)
    with path["journal"].open("a+b") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ("best", "latest", "pending", "task"):
            path[key].with_name(path[key].name + ".tmp").unlink(missing_ok=True)
        if path["latest"].is_file():
            saved = h.load_torch(path["latest"])
            digest = reconcile(path, journal, saved)
            restore(ctx, saved, manifest["implementation"])
            completed, best_epoch, best_correct, wait, best_sha = (
                saved[key] for key in ("next_epoch", "best_epoch", "best_correct", "wait", "best_sha"))
            del saved
        else:
            h.require(not path["best"].exists(), "Best exists without a committed Latest")
            path["pending"].unlink(missing_ok=True)
            journal.seek(0)
            journal.truncate()
            digest = hashlib.sha256()
            completed, best_epoch, best_correct, wait, best_sha = START, 0, -1, 0, None

        for epoch in range(completed + 1, CFG["max_epochs"] + 1):
            if wait > CFG["early_stop_wait_gt"]:
                break
            before = h.rng_sha()
            diag, _ = train_step(ctx)
            after_train = h.rng_sha()
            tested = evaluate(ctx)
            after_eval = h.rng_sha()
            rng = {"epoch": epoch, "rng_before": before, "rng_after_train": after_train, "rng_after_eval": after_eval}
            h.require(after_eval == after_train and (epoch not in archive or rng == archive[epoch]),
                      f"Formal RNG differs, fold {fold}, epoch {epoch}")
            improved = tested["correct"] > best_correct
            if improved:
                best_correct, best_epoch, wait = tested["correct"], epoch, 0
            else:
                wait += 1
            completed = epoch
            record = {
                "epoch": epoch, "best_epoch": best_epoch, "correct": tested["correct"], "best_correct": best_correct,
                "wait": wait, "rng": rng, "diagnostics": diag,
            }
            entry = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
            journal.write(entry)
            digest.update(entry)
            if improved or epoch % CFG["latest_interval"] == 0 or epoch == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"]:
                payload = {
                    **fixed, "state": state_dict(ctx), "rng": h.base().pack_rng(), "rng_sha": after_eval,
                    "next_epoch": completed, "best_epoch": best_epoch, "best_correct": best_correct,
                    "wait": wait, "evaluation": tested,
                }
                best_sha = commit(path, journal, digest, payload, improved, best_sha)
                del payload
            if epoch % 25 == 0 or epoch == START + 1:
                print(f"fold={fold} epoch={epoch} best={best_correct}/{len(state['test_np'])}@{best_epoch} wait={wait}", flush=True)

    h.require(best_epoch > START and best_sha == h.sha_file(path["best"]), "Formal Best commit is missing")
    result = {
        "status": "COMPLETE", "group": group, "fold": fold, "epoch": completed, "best_epoch": best_epoch, "best_correct": best_correct,
        "best_sha": best_sha, "latest_sha": h.sha_file(path["latest"]), "implementation": manifest["implementation"],
        "wall_seconds_this_worker": time.monotonic() - started, "cpu": sorted(os.sched_getaffinity(0)),
        "peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 2**20),
    }
    h.atomic_json(path["task"], result)
    print(f"COMPLETE fold={fold} end={completed} best={best_correct}/{len(state['test_np'])}@{best_epoch}", flush=True)



def cpus_for_gpu(gpu):
    bus = gpu["bus"].lower()[-12:]
    pci = Path("/sys/bus/pci/devices") / bus
    node = (pci / "numa_node").read_text().strip()
    ids = (Path("/sys/devices/system/node") / f"node{node}" / "cpulist").read_text().strip()
    nearby = []
    for entry in ids.split(","):
        ends = list(map(int, entry.split("-")))
        nearby.extend(range(ends[0], ends[-1] + 1))
    cpus = [cpu for cpu in nearby if cpu in os.sched_getaffinity(0)]
    h.require(len(cpus) >= CFG["gpu"]["max_workers"], "Insufficient GPU-local thread affinity")
    return cpus



def setup(group, fold, validate_spectrum=False):
    state, warmup, identity, optimizers = h.prepare(fold)
    original_rng = h.rng_sha()
    native = h.runner().load_history()
    before_model = native.state_model_sha(state)
    archived = h.load_torch(h.E822 / "runtime" / f"spectrum_f{fold:02d}.pt")
    h.require(archived["identity"] == identity and np.array_equal(archived["legal_train"], state["legal_np"]),
              "Original O identity differs")
    spectrum = archived["spectrum"]
    h.require(h.sha(spectrum) == archived["spectra_sha"]
              and h.sha(spectrum["U"]) == archived["e82_original_U_sha"], "Original O cache hash differs")

    _, captured = forward(state, False)
    cam = captured["cam"].detach().cpu()
    h.require(torch.equal(cam, archived["warmup_cam"]) and h.sha(cam) == archived["feature_sha"],
              "Deterministic E61 CAM differs from Original O reference")
    h.require(native.state_model_sha(state) == before_model, "Warmup extraction changed DARG")
    legal = state["legal_np"]
    labels = state["labels_np"][legal].copy()
    if validate_spectrum:
        rebuilt = h.psce().build_spectra(cam[legal], torch.as_tensor(labels))
        for key in h.SPECTRUM_KEYS:
            h.require(torch.equal(rebuilt[key], spectrum[key]), f"Original O {key} differs on recomputation")
    orthogonality = float((spectrum["U"].T @ spectrum["U"] - torch.eye(16)).abs().max())
    h.require(orthogonality < 1e-10 and not spectrum["fallback"], "Original O covariance directions invalid")
    z = ((cam[legal].double() - spectrum["mu"]) @ spectrum["U"]).numpy()
    reference = None
    if group in ("G3", "G4"):
        reference = fit_crossfit(z, labels, legal, fold, CFG["crossfit"])
        h.require(not np.isin(reference["legal"], state["test_np"]).any()
                  and (reference["calibration"][:, 1] >= 0).all(), "Invalid calibration boundary or orientation")
    text = h.verify_text()
    method = h.GROUPS[group]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CFG["module_seed_base"] + 100 * int(group[1:]) + fold)
        module = METHODS[group](spectrum, text, reference, state["excluded"], CFG, method)
    module = module.to(state["x"].device)
    optimizer = torch.optim.AdamW(module.parameters(), **CFG["new_optimizer"])
    owned = {p for param_group in optimizer.param_groups for p in param_group["params"]}
    h.require(owned == set(module.parameters()) and not owned.intersection(h.parameter_names_p(state)),
              "New optimizer owns an original parameter or omits a new parameter")
    h.require(h.rng_sha() == original_rng, "Reference fitting or new-layer initialization consumed DARG RNG")
    identities = {
        "identity": identity, "optimizers": optimizers, "reference_sha": h.sha(reference),
        "spectrum_sha": archived["spectra_sha"], "warmup_cam_sha": archived["feature_sha"],
        "query_sha": text["identity_sha"], "init_module_sha": h.sha(module.state_dict()),
        "membership_sha": h.sha(h.membership(state)), "trainable_parameters": sum(p.numel() for p in owned),
    }
    return {
        "state": state, "warmup": warmup, "reference": reference, "method": method,
        "module": module, "optimizer": optimizer, "text": text, "spectrum": spectrum,
        "reference_z": z, "identity": identities, "orthogonality": orthogonality,
    }


@torch.no_grad()
def counterfactual(head, cam, coordinates, directions, labels):
    """E82.2's one-sided true-margin masking in the active token coordinates."""
    components = coordinates.unsqueeze(-1) * directions
    perturbation = smooth_limit(components, cam.unsqueeze(1), CFG["residual"]["norm_limit"])
    masking = torch.cat((cam.unsqueeze(1), cam.unsqueeze(1) - perturbation), dim=1)
    logits = frozen_head(head, masking)
    margins = (logits[..., 0] - logits[..., 1]) * (1 - 2 * labels[:, None])
    d = margins[:, :1] - margins[:, 1:]
    positive = d.relu()
    valid = (margins[:, 0] > 0) & (d.amax(-1) > 1e-6)
    return positive / (positive.sum(-1, keepdim=True) + 1e-8), valid


def auxiliary_losses(ctx, out, capture):
    state, module = ctx["state"], ctx["module"]
    group = ctx["method"]["group"]
    legal, details = state["legal"], capture["details"]
    labels = state["y"][legal]
    cam, delta = capture["cam"][legal].detach(), details["delta"][legal]
    head = state["cam"].Outputlayer
    with torch.no_grad():
        baseline = margin(frozen_head(head, cam), labels)
    full_margin = margin(frozen_head(head, cam + delta), labels)
    change = full_margin - baseline
    margin_scale = CFG["protection"]["margin_scale"]
    tolerance = CFG["protection"]["allowed_margin_drop"]
    protection = 1 - (-baseline.relu() / margin_scale).exp()
    losses = {}

    if group in ("G1", "G2"):
        z, directions = details["z"][legal], module.basis.T
        if group == "G1":
            z, directions = details["coordinates"][legal], details["directions"][legal]
        target, valid = counterfactual(head, cam, z, directions, labels)
        weights = details["weights"][legal]
        kl = (torch.xlogy(target, target) - target * weights.clamp_min(1e-8).log()).sum(-1)
        losses["counterfactual"] = (kl * valid).sum() / valid.sum().clamp_min(1)
        if group == "G1":
            losses["rotation"] = (details["rotation"][legal] - module.eye).square().sum((-2, -1)).mean() / 16
            distribution = details["attention"][legal, :, :16]
            distribution = distribution / distribution.sum(-1, keepdim=True).clamp_min(1e-8)
            distance = (distribution - distribution.mean(-2, keepdim=True)).square().mean((-2, -1)).clamp_min(1e-12).sqrt()
            losses["distinction"] = (ctx["method"]["min_prompt_diversity"] - distance).relu().square().mean()
    else:
        target, _, _, _ = h.da().direction_target(out["logp"], capture["enhanced"], details["z"],
                                                module.basis, legal, labels, CFG["direction_target"])
        losses["direction"] = balanced_mean(h.da().direction_kl(target, details["weights"][legal]), labels)
        if group == "G3":
            losses["protect"] = balanced_mean(protection * (-change - tolerance).relu().square(), labels)
        else:
            half = margin(frozen_head(head, cam + 0.5 * delta), labels)
            gain_half, gain_second = half - baseline, full_margin - half
            details["second_half_margin_change"] = gain_second.detach()
            risk = (-gain_half - tolerance).relu().square() + (-gain_second - tolerance).relu().square()
            temperature = ctx["method"]["difficult_margin_temperature"]
            difficult = (-baseline / temperature).sigmoid()
            effort = (ctx["method"]["half_margin_weight"] * F.softplus(-gain_half / temperature)
                      + F.softplus(-gain_second / temperature)) * temperature
            losses["path_margin"] = balanced_mean(protection * risk + difficult * effort, labels)

    losses["residual"] = (delta.square().sum(-1) / cam.square().sum(-1).clamp_min(1e-8)).mean()
    return losses, change.detach(), protection.detach()


@torch.no_grad()
def method_diagnostics(group, d, rows):
    if group == "G1":
        rotation = d["rotation"][rows]
        eye = torch.eye(16, device=rotation.device)
        query = d["effective_queries"]
        return {
            "rotation_deviation": (rotation - eye).norm(dim=(-2, -1)).mean(),
            "rotation_patient_variation": rotation.std(0, unbiased=False).norm(),
            "rotation_orthogonality_error": (rotation.transpose(-2, -1) @ rotation - eye).abs().amax(),
            "query_cosine": ((query @ query.T).sum() - 8) / 56,
        }
    if group == "G2":
        beta, affinity = d["beta"][rows], d["affinity"][rows]
        local, glob = d["local_z"][rows], d["global_z"][rows]
        return {
            "local_beta_mean": beta.mean(), "local_beta_sd": beta.std(unbiased=False),
            "local_beta_min": beta.min(), "local_beta_max": beta.max(),
            "local_affinity_range": (affinity.amax(-1) - affinity.amin(-1)).mean(),
            "local_projection_energy": local.square().sum() / d["local_raw"][rows].square().sum().clamp_min(1e-12),
            "local_contribution_norm": (local * beta).norm(dim=-1).mean(),
            "global_contribution_norm": (glob * (1 - beta)).norm(dim=-1).mean(),
            "local_global_cosine": F.cosine_similarity(local, glob, dim=-1).mean(),
        }
    direction = d["class_directions"][rows]
    evidence_alignment = F.cosine_similarity(d["weights"][rows], d["evidence"][rows].abs(), dim=-1)
    result = {
        "llr_abs_mean": d["llr"][rows].abs().mean(), "llr_abs_max": d["llr"].abs().amax(),
        "asd_probability_mean": d["p_asd"][rows].mean(),
        "asd_probability_sd": d["p_asd"][rows].std(unbiased=False),
        "reliability_mean": d["reliability"][rows].mean(), "support_mean": d["support"][rows].mean(),
        "class_direction_asd_norm": direction[:, 0].norm(dim=-1).mean(),
        "class_direction_td_norm": direction[:, 1].norm(dim=-1).mean(),
        "prompt_class_evidence_cosine": evidence_alignment.mean(),
        "control_range": (d["control"][rows].amax(-1) - d["control"][rows].amin(-1)).mean(),
    }
    if group == "G4":
        v, a = d["factors"], d["operator_weights"][rows]
        op_energy = a.square().sum(-1)
        diagonal_energy = (a @ v.square().T).square().sum(-1)
        result.update(
            operator_orthogonality_error=(v.T @ v - torch.eye(4, device=v.device)).abs().amax(),
            operator_coefficients_patient_variation=a.std(dim=0, unbiased=False).norm(),
            operator_coefficient_abs_min=a.abs().amin(-1).mean(),
            operator_offdiagonal_energy=(1 - diagonal_energy / op_energy.clamp_min(1e-12)).mean(),
            operator_basis_movement=(v - d.get("initial_factors", v)).norm(),
        )
    return result


@torch.no_grad()
def step_diagnostics(out, capture, aux, gradient, change, protected, ctx):
    d = capture["details"]
    state = ctx["state"]
    legal, y = state["legal"], state["y"][state["legal"]]
    attention, weight = d["attention"][legal], d["weights"][legal]
    direction_profiles = F.normalize(attention[..., :16], dim=-1)
    cosine = (direction_profiles.sum(1).square().sum(-1) - 8) / 56
    relative = d["relative"][legal]
    tensors = {
        **{key: out[key].detach() for key in LOSS_KEYS},
        **{key: loss.detach() for key, loss in aux.items()},
        **gradient,
        **method_diagnostics(ctx["method"]["group"], d, legal),
        "attention_entropy": -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean() / math.log(17),
        "prompt_pairwise_cosine": cosine.mean(),
        "attention_uniform_l1": (attention - 1 / 17).abs().sum(-1).mean(),
        "null_mean": attention[:, :, 16].mean(),
        "coverage_mean": d["coverage"][legal].mean(),
        "coverage_sd": d["coverage"][legal].std(unbiased=False),
        "weight_uniform_l1": (weight - 1 / 16).abs().sum(-1).mean(),
        "negative_direction_fraction": (d["delta_z"][legal] < 0).float().mean(),
        "delta_relative_mean": relative.mean(),
        "delta_relative_max": d["relative"].amax(),
        "delta_nonzero_fraction": (relative > 1e-6).float().mean(),
        "margin_change": change.mean(),
        "margin_change_asd": change[y == 0].mean(),
        "margin_change_td": change[y == 1].mean(),
        "margin_improved_fraction": (change > 0).float().mean(),
        "protection_weight_mean": protected.mean(),
        "protected_harm_weighted": ((change < -CFG["protection"]["allowed_margin_drop"]) * protected).sum()
                                  / protected.sum().clamp_min(1e-8),
    }
    if "second_half_margin_change" in d:
        second = d["second_half_margin_change"]
        tensors.update(margin_half_change=(change - second).mean(), margin_second_half_change=second.mean())
    values = torch.stack(tuple(tensors.values())).cpu().tolist()
    scalars = dict(zip(tensors, values))
    h.require(all(math.isfinite(value) for value in values), "Nonfinite loss or mechanism diagnostics")
    h.require(scalars["delta_relative_max"] <= CFG["residual"]["norm_limit"] + 2e-6, "CAM144 residual norm cap violated")
    cap = sum(item["gradient_cap"] for item in ctx["method"]["auxiliary"].values())
    for name in ("prompt", "residual"):
        h.require(scalars[f"{name}_applied_ratio"] <= cap + 2e-6, f"{name} auxiliary gradient budget violated")
    return scalars


def check_preflight(fold):
    verification = h.verify_device()
    torch.cuda.reset_peak_memory_stats()
    state, warmup, _, _ = h.prepare(fold)
    history_rng = warmup["rng_sha"]
    calls = []
    wrapped = native_signature(state, True, calls)
    del state, warmup
    native, _, _, _ = h.prepare(fold)
    native_calls = []
    expected = native_signature(native, False, native_calls)
    del native
    h.require(wrapped == expected and calls == native_calls and len(calls) == 6,
              "Disabled residual/auxiliary mode diverges from native DARG")
    archive = h.historical_task(fold)["rng_trace"][0]
    h.require(wrapped["rng"] == archive, "Identity preflight RNG differs from archived DARG")

    for group in METHODS:
        ctx = setup(group, fold, validate_spectrum=True)
        state, module = ctx["state"], ctx["module"]
        h.require(h.rng_sha() == history_rng, "Preflight restored a mismatched RNG")
        with torch.no_grad():
            _, native_capture = forward(state, False)
            _, full = forward(state, False, module)
            anonymous = ctx["text"]["q_anonymous"].to(state["x"].device)
            _, replacement = forward(state, False, module, query=anonymous)
            details = full["details"]
            logit_max = float((full["logits"] - native_capture["logits"])[state["test_np"]].abs().amax())
            query_delta = float((details["delta"] - replacement["details"]["delta"]).abs().amax())
            h.require(tuple(details["attention"].shape) == (CFG["subjects"], 8, 17)
                      and torch.allclose(details["attention"].sum(-1), torch.ones_like(details["attention"][:, :, 0]),
                                         rtol=0, atol=2e-6), "Attention rows are invalid")
            h.require(logit_max > 1e-6 and query_delta > 1e-7
                      and float(details["relative"].mean()) > 1e-5, "Initial residual or Prompt has no classifier response")
            h.require(not state["excluded"] or not bool(details["delta"][state["excluded"]].count_nonzero()),
                      "Fold 2 initial residual bypass failed")
            if group == "G1":
                h.require(bool((details["rotation"] == module.eye).all()), "Cayley map is not exact identity at initialization")
        del native_capture, full, replacement, details

        diag, gradient = train_step(ctx, inspect=True)
        h.require(h.rng_sha() == archive["rng_after_train"], "First enabled update consumes different DARG RNG")
        evaluated = evaluate(ctx)
        h.require(h.rng_sha() == archive["rng_after_eval"], "First enabled evaluation altered RNG")
        h.require(np.isfinite(evaluated["probability"]).all(), "First enabled update produced nonfinite predictions")
        reference = ctx["reference"]
        statistics = {} if reference is None else {
            "oof_auc": float(roc_auc_score(reference["labels"] == 0, reference["oof_llr"])),
            "oof_calibrated_auc": float(roc_auc_score(reference["labels"] == 0, reference["oof_asd"])),
            "gaussian_condition": reference["gaussian"]["condition"].tolist(),
            "calibration": reference["calibration"].tolist(), "crossfit_seed": reference["seed"],
        }
        result = {
            "status": "PASS", "fold": fold, "group": group, "identity": ctx["identity"],
            "orthogonality_max_abs": ctx["orthogonality"], "native_regression": wrapped, "dropout_calls": calls,
            "gradient_paths": gradient, "epoch11_diagnostics": diag,
            "initial_logit_max_abs_change": logit_max, "initial_query_delta_max_abs_change": query_delta,
            "peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 2**20),
            "device": verification, "implementation": h.implementation_hashes(), **statistics,
        }
        h.atomic_json(paths(group, fold)["preflight"], result)
        print(f"PASS {group} fold {fold}: initial logit shift {logit_max:.6g}, reserve {result['peak_reserved_mib']} MiB",
              flush=True)
        del ctx, module, state


def pool(script, command, folds, width, cpu, group=None):
    remaining = deque(folds)
    active = {}
    finished = []
    restarted = {}
    peak = h.gpu_sample()["used_mib"]
    width_observed = 0
    started = time.monotonic()
    probe_at = None
    probe_sizes = {}
    try:
        while remaining or active:
            while remaining and len(active) < width:
                fold = remaining.popleft()
                log = RUNTIME / "logs" / f"{command}_{group or 'all'}_f{fold:02d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                arguments = [command, str(fold)]
                if group is not None:
                    arguments += [group]
                args = ["taskset", "-c", str(cpu[fold - 1]), sys.executable, "-u", str(HERE / script), *arguments]
                with log.open("a", encoding="utf-8") as stream:
                    active[fold] = subprocess.Popen(args, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
                if width > 1:
                    time.sleep(0.45)
            width_observed = max(width_observed, len(active))
            peak = max(peak, h.gpu_sample()["used_mib"])
            for fold, proc in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[fold]
                h.require(code == 0, f"{command} {group or 'all'} fold {fold} exited with {code}. "
                                     f"See runtime/logs/{command}_{group or 'all'}_f{fold:02d}.log")
                finished.append(fold)
                print(f"{command} {group or 'all'} fold {fold:02d} complete ({len(finished)} finished)", flush=True)

            # A PPU context can stay slow after other workers release the device.
            # Restart only a slow formal tail with a verified committed journal prefix.
            if command == "train-fold" and not remaining and len(finished) >= 4 and active:
                now = time.monotonic()
                if probe_at is None:
                    probe_at = now
                    probe_sizes = {f: paths(group, f)["journal"].stat().st_size for f in active}
                if now - probe_at >= 45:
                    for fold, proc in list(active.items()):
                        p = paths(group, fold)
                        if fold in restarted or not p["latest"].is_file() or fold not in probe_sizes:
                            continue
                        size = p["journal"].stat().st_size
                        if size - probe_sizes[fold] > 120000:
                            continue
                        proc.terminate()
                        proc.wait(timeout=20)
                        del active[fold]
                        if p["task"].is_file() and h.read_json(p["task"])["status"] == "COMPLETE":
                            finished.append(fold)
                            continue
                        saved = h.load_torch(p["latest"])
                        with p["journal"].open("rb") as stream:
                            prefix = stream.read(saved["journal_size"])
                        h.require(hashlib.sha256(prefix).hexdigest() == saved["journal_sha"],
                                  "Slow-tail committed journal verification failed")
                        restarted[fold] = {"epoch": saved["next_epoch"], "rng_sha": saved["rng_sha"],
                                           "latest_sha": h.sha_file(p["latest"]), "journal_sha": saved["journal_sha"]}
                        print(f"{group} fold {fold:02d} resuming slow context from epoch {saved['next_epoch']}", flush=True)
                        remaining.append(fold)
                    probe_at = None
            if active:
                time.sleep(2)
    finally:
        for proc in active.values():
            proc.terminate()
        for proc in active.values():
            proc.wait(timeout=20)
    return {"group": group, "folds": sorted(finished), "max_workers": width_observed, "peak_sampled_mib": peak,
            "wall_seconds": time.monotonic() - started, "slow_context_restores": restarted}


def execute():
    device = h.gpu_sample()
    cpu = cpus_for_gpu(device)
    implementation = h.implementation_hashes()
    if MANIFEST.is_file():
        manifest = h.read_json(MANIFEST)
        h.require(manifest["implementation"] == implementation, "Frozen source hashes differ")
    else:
        protected = h.hash_files(h.guard_paths())
        manifest = {
            "experiment": CFG["experiment"], "groups": h.GROUPS, "seed": CFG["seed"], "status": "PREPARING",
            "directory": str(HERE), "preferred_directory": str(h.REPO / "experiment" / HERE.name),
            "fallback": CFG["directory_fallback"],
            "repository_head": subprocess.check_output(["git", "-C", str(h.REPO), "rev-parse", "HEAD"], text=True).strip(),
            "reviewed_remotes": {"wanxixi11/SPromptGL": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
                                  "tangaii/SPromptGL": "4a86923a28ce9920c0fec28c797c9c6c5933674d"},
            "paper_sha": "219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89",
            "implementation": implementation, "implementation_sha": h.sha(implementation),
            "protected_sha256": protected, "protected_bundle_sha": h.sha(protected),
            "packages": h.packages(), "device": device, "affinity": cpu[:CFG["gpu"]["max_workers"]],
        }
        h.atomic_json(MANIFEST, manifest)

    try:
        if manifest.get("preflight", {}).get("status") != "PASS":
            before = h.gpu_sample()
            print("Starting single-worker preflight for all four methods", flush=True)
            single = pool("run.py", "preflight-fold", [1], 1, cpu)
            fold1 = [h.read_json(paths(group, 1)["preflight"]) for group in METHODS]
            per_worker = max(*(row["peak_reserved_mib"] for row in fold1),
                             single["peak_sampled_mib"] - before["used_mib"])
            estimate = before["used_mib"] + math.ceil(1.5 * per_worker * CFG["gpu"]["max_workers"])
            h.require(estimate < before["total_mib"] * 0.9, f"GPU0 ten-worker estimate exceeds memory budget: {estimate} MiB")
            print(f"GPU0 preflight estimates {estimate} of {before['total_mib']} MiB for ten workers", flush=True)
            others = pool("run.py", "preflight-fold", list(FOLDS)[1:], CFG["gpu"]["max_workers"], cpu)
            manifest["preflight"] = {"status": "PASS", "folds": list(FOLDS), "methods": list(METHODS),
                                     "single_worker": single, "other_folds": others, "ten_worker_estimate_mib": estimate}
            h.atomic_json(MANIFEST, manifest)
        guard = h.verify_guard(manifest)
        manifest.update(status="TRAINING", guard_before=guard)
        h.atomic_json(MANIFEST, manifest)
        for group in METHODS:
            finished = [f for f in FOLDS if paths(group, f)["task"].is_file()
                        and h.read_json(paths(group, f)["task"]).get("status") == "COMPLETE"]
            pending = [f for f in FOLDS if f not in finished]
            if pending:
                print(f"Formal wave {group}: {pending}", flush=True)
                execution = pool("run.py", "train-fold", pending, CFG["gpu"]["max_workers"], cpu, group)
                manifest.setdefault("training_runs", []).append(execution)
            manifest.setdefault("completed_folds", {})[group] = [h.read_json(paths(group, f)["task"]) for f in FOLDS]
            h.atomic_json(MANIFEST, manifest)
        manifest["status"] = "TRAINED_PENDING_AUDIT"
        h.atomic_json(MANIFEST, manifest)
    except Exception as exc:
        manifest.update(status="INTERRUPTED", last_error=str(exc))
        h.atomic_json(MANIFEST, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("execute", "preflight-fold", "train-fold"))
    parser.add_argument("fold", nargs="?", type=int, choices=list(FOLDS))
    parser.add_argument("group", nargs="?", choices=list(METHODS))
    args = parser.parse_args()
    if args.action == "execute":
        RUNTIME.mkdir(exist_ok=True)
        with (RUNTIME / "scheduler.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute()
    elif args.action == "preflight-fold" and args.fold is not None:
        check_preflight(args.fold)
    elif args.action == "train-fold" and args.fold is not None and args.group is not None:
        train_fold(args.group, args.fold)
    else:
        parser.error("The action requires a fold or group")


if __name__ == "__main__":
    main()
