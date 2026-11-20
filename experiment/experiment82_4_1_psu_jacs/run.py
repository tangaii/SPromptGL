"""Train both E82.4.1 methods through the original DARG entry point."""
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
from torch.nn import functional as F

from common.references import balanced_mean, fit_reference, frozen_head, head_jacobian, margin
from G1_psu_psce.model import PsuPsce
from G2_jacs_da_srpe.model import JacsDaSrpe

CFG, HERE = h.CFG, h.HERE
RUNTIME, RESULTS = HERE / "runtime", HERE / "results"
MANIFEST = HERE / "manifest.json"
START = CFG["start_epoch"]
LOSS_KEYS = ("cls", "graph", "pair", "adv", "preserve", "total")
FOLDS = range(1, CFG["folds"] + 1)
METHODS = dict(G1=PsuPsce, G2=JacsDaSrpe)


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
            enhanced, details = module(inputs[0], head=head, training=training, query=query,
                                       semantic_off=semantic_off, variant=variant)
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
    h.require(statistics_identity(ctx) == payload["ema"], "Checkpoint EMA state differs")
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
              and manifest.get("geometry", {}).get("status") == "PASS"
              and preflight["implementation"] == manifest["implementation"] == h.implementation_hashes(),
              "Missing compatible preflight or label-free diagnostics")
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
            ema_before = statistics_identity(ctx)
            diag, _ = train_step(ctx)
            ema_after = statistics_identity(ctx)
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
                "ema_before": ema_before, "ema_after": ema_after,
            }
            entry = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
            journal.write(entry)
            digest.update(entry)
            if improved or epoch % CFG["latest_interval"] == 0 or epoch == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"]:
                payload = {
                    **fixed, "state": state_dict(ctx), "rng": h.base().pack_rng(), "rng_sha": after_eval,
                    "next_epoch": completed, "best_epoch": best_epoch, "best_correct": best_correct,
                    "wait": wait, "evaluation": tested, "ema": ema_after,
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
    method = h.GROUPS[group]
    reference = fit_reference(z, labels, legal, fold, method["stats"]) if group == "G2" else None
    old_preflight = h.read_json(h.E824 / "G1_dpr_psce/runtime/preflight" / f"G1_f{fold:02d}.json")
    previous = old_preflight["identity"]
    h.require(identity == previous["identity"] and optimizers == previous["optimizers"]
              and h.sha(h.membership(state)) == previous["membership_sha"],
              f"Fold {fold} E82.4 initial model, optimizer or row identity differs")
    text = h.verify_text()
    h.require(text["identity_sha"] == previous["query_sha"] and archived["spectra_sha"] == previous["spectrum_sha"],
              f"Fold {fold} E82.4 RN50 or Original O identity differs")
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
        expected = h.verify_prior()
        differences = [p for p, digest in expected["protected_sha256"].items() if protected.get(p) != digest]
        h.require(not differences, f"E82.4 protected identities differ: {differences}")
        for row in h.read_json(h.E824 / "results/audit_status.json")["folds"]:
            tag = row["group"]
            folder = next(h.E824.glob(tag + "_*")) / "checkpoints"
            for kind in ("best", "latest"):
                path = folder / f"{tag}_f{row['fold']:02d}.{kind}.pt"
                h.require(protected[str(path)] == row[kind + "_sha"], f"E82.4 archived checkpoint differs: {path}")
        manifest = {
            "experiment": CFG["experiment"], "groups": h.GROUPS, "seed": CFG["seed"], "status": "PREPARING",
            "directory": str(HERE), "preferred_directory": str(h.REPO / "experiment" / HERE.name),
            "fallback": CFG["directory_fallback"],
            "repository_head": subprocess.check_output(["git", "-C", str(h.REPO), "rev-parse", "HEAD"], text=True).strip(),
            "reviewed_remotes": {"wanxixi11/SPromptGL": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
                                  "tangaii/SPromptGL": "4a86923a28ce9920c0fec28c797c9c6c5933674d"},
            "prior_manifest_sha": CFG["prior_manifest_sha"],
            "paper_sha": "219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89",
            "implementation": implementation, "implementation_sha": h.sha(implementation),
            "protected_sha256": protected, "protected_bundle_sha": h.sha(protected),
            "packages": h.packages(), "device": device, "affinity": cpu[:CFG["gpu"]["max_workers"]],
        }
        h.atomic_json(MANIFEST, manifest)

    try:
        if manifest.get("preflight", {}).get("status") != "PASS":
            before = h.gpu_sample()
            print("Starting single-worker preflight for both methods", flush=True)
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
        if manifest.get("geometry", {}).get("status") != "PASS":
            print("Running label-free spectral geometry on archived Best checkpoints", flush=True)
            geometry = pool("common/geometry.py", "geometry-fold", list(FOLDS), CFG["gpu"]["max_workers"], cpu)
            artifacts = [RUNTIME / "geometry" / f"f{fold:02d}.json" for fold in FOLDS]
            h.require(all(h.read_json(p)["status"] == "PASS" for p in artifacts), "Archived geometry stage incomplete")
            manifest["geometry"] = {**geometry, "status": "PASS", "artifacts": h.hash_files(artifacts)}
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



def statistics_identity(ctx):
    module = ctx["module"]
    if not hasattr(module, "statistics"):
        return None
    bank = module.statistics
    return {"updates": int(bank.updates), "sha": h.sha(bank.state_dict())}


def measure_gradient_path(state, module, out, capture):
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
    cam, enhanced, d = capture["cam"], capture["enhanced"], capture["details"]
    ts = torch.autograd.grad(out["cls"], [*requested, cam, enhanced], retain_graph=True, allow_unused=True)
    g_cam, g_enhanced = ts[-2:]
    native_params = [p for key in ("cam", "gfr", "darg") for p in families[key]]
    branch, residual, *rest = torch.autograd.grad(
        d["delta"], (d["branch_cam"], cam, *native_params), grad_outputs=g_enhanced.detach(),
        retain_graph=True, allow_unused=True,
    )
    h.require(torch.equal(d["branch_cam"], cam) and torch.equal(residual, CFG["branch_gradient"] * branch),
              "Branch does not preserve the exact forward and specified backward scaling")
    h.require(torch.equal(g_cam, g_enhanced + residual), "Unscaled direct CAM gradient differs from original classifier")
    direct = torch.autograd.grad(cam, native_params, grad_outputs=g_enhanced.detach(), retain_graph=True, allow_unused=True)

    def norm(v):
        active = [x for x in v if x is not None]
        return float(grad_norm(active)) if active else 0.0

    by_param = dict(zip(requested, ts[:-2]))
    result = {name: norm([by_param[p] for p in params]) for name, params in families.items()}
    for name, grad in (("cam144", g_cam), ("direct_cam144", g_enhanced), ("branch_input", branch), ("delta_to_cam144", residual)):
        result[name] = norm([grad])
    offset = 0
    for name in ("cam", "gfr", "darg"):
        size = len(families[name])
        result["direct_to_" + name] = norm(direct[offset:offset + size])
        result["delta_to_" + name] = norm(rest[offset:offset + size])
        offset += size
    inactive = [name for name, p in module.named_parameters()
                if by_param[p] is None or not bool(by_param[p].abs().max() > 0)]
    h.require(not inactive and all(math.isfinite(v) and v > 0 for v in result.values()),
              f"Classification did not reach all functional paths: {inactive} {result}")
    return {**result, "branch_tensors": len(list(module.parameters())), "backward_alpha": CFG["branch_gradient"],
            "exact_forward_and_direct_gradient_identity": True}


def auxiliary_losses(ctx, capture):
    state, module = ctx["state"], ctx["module"]
    d = capture["details"]
    legal, head = state["legal"], state["cam"].Outputlayer
    labels = state["y"][legal]
    cam, delta = capture["cam"][legal].detach(), d["delta"][legal]
    if "jacobian" in d:
        base_logits, jacobian = d["frozen_logits"][legal], d["jacobian"][legal]
    else:
        base_logits, jacobian = head_jacobian(head, cam)
    baseline = margin(base_logits, labels)
    gain = margin(frozen_head(head, cam + delta), labels) - baseline
    ysign = (1 - 2 * labels).unsqueeze(-1)
    true_gradient = ysign * (jacobian @ module.basis)
    applied = delta @ module.basis
    cfg = CFG["utility"]
    distance_floor = cfg["direction_delta_floor"] * cam.norm(dim=-1).clamp_min(1e-8)
    z_norm = (applied.square().sum(-1) + distance_floor.square()).sqrt()
    jac_norm = (true_gradient.square().sum(-1) + cfg["direction_jacobian_floor"]**2).sqrt()
    agreement = (applied * true_gradient).sum(-1) / (jac_norm * z_norm)
    difficult = ((cfg["difficulty_center"] - baseline) / cfg["difficulty_temperature"]).sigmoid()
    protect = 1 - (-baseline.relu() / cfg["protect_margin_scale"]).exp()
    gain_temp, protect_temp = cfg["gain_temperature"], cfg["protect_smoothing"]
    harm = protect_temp * F.softplus((-gain - cfg["protect_allowance"]) / protect_temp)
    losses = {
        "alignment": balanced_mean(difficult * (1 - agreement), labels),
        "gain": balanced_mean(difficult * gain_temp * F.softplus(-gain / gain_temp), labels),
        "protect": balanced_mean(protect * harm.square(), labels),
        "residual": (delta.square().sum(-1) / cam.square().sum(-1).clamp_min(1e-8)).mean(),
    }
    observation = {"gain": gain.detach(), "agreement": agreement.detach(), "protect": protect.detach(),
                   "base_margin": baseline}
    return losses, observation


@torch.no_grad()
def step_diagnostics(out, cap, aux, gradient, obs, ctx):
    state, d = ctx["state"], cap["details"]
    legal, labels = state["legal"], state["y"][state["legal"]]
    attention, action, gains = d["attention"][legal], d["action"][legal], obs["gain"]
    relative, strength = d["relative"][legal], d["eta"][legal]
    tensors = {
        **{key: out[key].detach() for key in LOSS_KEYS},
        **{key: term.detach() for key, term in aux.items()},
        **gradient,
        "attention_entropy": -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean() / math.log(17),
        "null_mean": attention[..., 16].mean(),
        "coverage_mean": d["coverage"][legal].mean(),
        "signed_negative_fraction": (action < 0).float().mean(),
        "signed_abs_mean": action.abs().mean(),
        "eta_mean": strength.mean(),
        "eta_sd": strength.std(unbiased=False),
        "delta_relative_mean": relative.mean(),
        "delta_relative_max": d["relative"].amax(),
        "delta_nonzero_fraction": (relative > 1e-6).float().mean(),
        "margin_change": gains.mean(),
        "margin_change_asd": gains[labels == 0].mean(),
        "margin_change_td": gains[labels == 1].mean(),
        "alignment_cosine_mean": obs["agreement"].mean(),
        "protection_weight_mean": obs["protect"].mean(),
        "protected_margin_drop_fraction": ((gains < -CFG["utility"]["protect_allowance"]) * obs["protect"]).sum()
                                          / obs["protect"].sum().clamp_min(1e-8),
    }
    if "jacobian" in d:
        projection = d["projected_jacobian"].square().sum(-1)
        tensors.update(jacobian_coverage=(projection / (d["jacobian"].square().sum(-1) + 1e-8))[legal].mean(),
                       stats_reliability=d["reliability"][legal].mean(),
                       stats_effect_abs=d["effect"][legal].abs().mean())
    scalars = dict(zip(tensors, torch.stack(tuple(tensors.values())).cpu().tolist()))
    h.require(all(math.isfinite(v) for v in scalars.values()), "Nonfinite loss or training diagnostic")
    h.require(scalars["delta_relative_max"] <= CFG["residual"]["norm_limit"] + 2e-6, "Residual exceeds 8% cap")
    cap = sum(term["gradient_cap"] for term in CFG["auxiliary"].values())
    for family in ("prompt", "residual"):
        h.require(scalars[family + "_applied_ratio"] <= cap + 2e-6, "Auxiliary gradient exceeded budget")
    return scalars


def train_step(ctx, inspect=False):
    state, module = ctx["state"], ctx["module"]
    optimizers = [*(state[key] for key in h.OPTS), ctx["optimizer"]]
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    out, capture = forward(state, True, module)
    aux, obs = auxiliary_losses(ctx, capture)
    path = measure_gradient_path(state, module, out, capture) if inspect else None
    stats = limit_auxiliary(module, aux, out["total"], CFG["auxiliary"], watch_gradients=inspect)
    diag = step_diagnostics(out, capture, aux, stats, obs, ctx)
    parameters = [*h.parameter_names_p(state), *module.parameters()]
    finite = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    h.require(bool(torch.stack(finite).all()), "Nonfinite optimization gradients")
    if inspect:
        for term in CFG["auxiliary"]:
            h.require(diag[f"prompt_{term}_raw_norm"] + diag[f"residual_{term}_raw_norm"] > 0,
                      f"Inactive auxiliary objective at initialization: {term}")
    for optimizer in optimizers:
        optimizer.step()
    if hasattr(module, "statistics"):
        module.statistics.advance(capture["details"]["z"].detach())
        bank = module.statistics
        h.require(bool(torch.isfinite(bank.mean).all() & torch.isfinite(bank.variance).all() & (bank.variance >= 0).all()),
                  "Invalid conditional EMA moments")
    if inspect:
        h.require(bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()), "Nonfinite updated parameter")
    return diag, path


def check_preflight(fold):
    device = h.verify_device()
    torch.cuda.reset_peak_memory_stats()
    state, warmup, _, _ = h.prepare(fold)
    warm_rng = warmup["rng_sha"]
    calls = []
    disabled = native_signature(state, True, calls)
    del state, warmup
    native, _, _, _ = h.prepare(fold)
    reference_calls = []
    expected = native_signature(native, False, reference_calls)
    del native
    h.require(disabled == expected and calls == reference_calls and len(calls) == 6,
              "Disabled branch changed original DARG")
    archive = h.historical_task(fold)["rng_trace"][0]
    h.require(disabled["rng"] == archive, "Identity preflight RNG differs from archived DARG")
    old = h.read_json(h.E824 / "G1_dpr_psce/runtime/preflight" / f"G1_f{fold:02d}.json")
    h.require(old["native_regression"] == disabled and old["dropout_calls"] == calls,
              "Native forward, gradient, optimizer or Dropout signature differs from verified E82.4")

    for group in METHODS:
        ctx = setup(group, fold, validate_spectrum=True)
        state, module = ctx["state"], ctx["module"]
        h.require(h.rng_sha() == warm_rng, "Preflight restored mismatched RNG")
        before_ema = statistics_identity(ctx)
        with torch.no_grad():
            _, baseline = forward(state, False)
            _, full = forward(state, False, module)
            replacement = ctx["text"]["q_anonymous"].to(state["x"].device)
            _, changed = forward(state, False, module, query=replacement)
            d = full["details"]
            logit_shift = float((full["logits"] - baseline["logits"]).abs().amax())
            query_shift = float((d["delta"] - changed["details"]["delta"]).abs().amax())
            residual = float(d["relative"].mean())
            h.require(d["attention"].shape == (CFG["subjects"], 8, 17)
                      and torch.allclose(d["attention"].sum(-1), torch.ones_like(d["attention"][:, :, 0]),
                                         atol=2e-6, rtol=0), "Invalid archived Prompt attention geometry")
            h.require(logit_shift > 1e-6 and query_shift > 1e-7 and .001 < residual < .025, "Inactive or oversized init")
            h.require(not bool(d["delta"][state["excluded"]].count_nonzero()), "Fold 2 masking failed")
        del baseline, full, changed, d
        original = F.dropout
        active = []

        def monitor(inputs, p=0.5, training=True, inplace=False):
            if training and 0 < p < 1:
                active.append({"shape": list(inputs.shape), "stride": list(inputs.stride()),
                               "dtype": str(inputs.dtype), "p": p, "inplace": inplace})
            return original(inputs, p=p, training=training, inplace=inplace)

        F.dropout = monitor
        try:
            diag, gradient = train_step(ctx, inspect=True)
        finally:
            F.dropout = original
        h.require(active == calls and h.rng_sha() == archive["rng_after_train"],
                  "Enabled branch changed the actual native Dropout trajectory")
        after_ema = statistics_identity(ctx)
        evaluate(ctx)
        h.require(h.rng_sha() == archive["rng_after_eval"] and statistics_identity(ctx) == after_ema,
                  "First eval changed RNG or statistics")
        result = {
            "status": "PASS", "fold": fold, "group": group, "identity": ctx["identity"],
            "orthogonality_max_abs": ctx["orthogonality"], "native_regression": disabled, "dropout_calls": calls,
            "enabled_dropout_calls": active, "gradient_paths": gradient, "epoch11_diagnostics": diag,
            "ema_before": before_ema, "ema_after": after_ema,
            "initial_logit_max_abs_change": logit_shift, "initial_query_delta_max_abs_change": query_shift,
            "initial_relative_mean": residual, "device": device, "implementation": h.implementation_hashes(),
            "peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 2**20),
        }
        h.atomic_json(paths(group, fold)["preflight"], result)
        print(f"PASS {group} fold {fold}: init residual {residual:.2%}, peak reserve {result['peak_reserved_mib']} MiB", flush=True)
        del ctx, state, module


if __name__ == "__main__":
    main()
