"""Train the two registered groups through the read-only E74.4-G2 entry point."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy

from common import history as h
from common.evidence import fit_reference
from common.optimization import ORIGINAL, ORIGINAL_LOG, apply_budget, distributions, families, inspect_paths, losses, optimizer_measurements
from common.spectral import fixed_logits, margin
from g1 import PCSRPSCE
from g2 import SCMRDASRPE

import numpy as np
import torch
from torch.nn import functional as F

CFG = h.CFG
ROOT = h.ROOT
RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
MANIFEST = ROOT / "manifest.json"
START = CFG["start_epoch_completed"]
INTERVENTIONS = {
    "G1": ("full", "residual_off", "anonymous_query", "differential_prompt_off", "patient_rotation_off", "signed_action_off"),
    "G2": ("full", "residual_off", "anonymous_query", "calibrated_evidence_off", "spectral_only", "complement_only",
           "evidence_conflict_off"),
}


def paths(group, fold):
    stem = f"{group}_f{fold:02d}"
    return {
        "best": ROOT / "checkpoints" / f"{stem}.best.pt",
        "latest": ROOT / "checkpoints" / f"{stem}.latest.pt",
        "journal": RUNTIME / "journal" / f"{stem}.jsonl",
        "task": RUNTIME / "tasks" / f"{stem}.json",
        "preflight": RUNTIME / "preflight" / f"{stem}.json",
        "audit": RUNTIME / "audit" / f"{stem}.json",
    }


@contextmanager
def dropout_inventory():
    calls = []
    original = F.dropout

    def track(inputs, p=0.5, training=True, inplace=False):
        if training and 0 < p < 1:
            calls.append({"shape": list(inputs.shape), "stride": list(inputs.stride()),
                          "p": p, "dtype": str(inputs.dtype), "inplace": inplace})
        return original(inputs, p=p, training=training, inplace=inplace)

    F.dropout = track
    try:
        yield calls
    finally:
        F.dropout = original


def forward(state, training, module=None, intervention="full"):
    head = state["cam"].Outputlayer
    capture = {}

    def inject(layer, args):
        cam = args[0]
        h.require("cam" not in capture and cam.shape == (CFG["subjects"], 144), "Historical CAM head input changed")
        capture["cam"] = cam
        if module is not None and not training:
            enhanced, details = module(cam, layer, training, intervention)
            capture.update(enhanced=enhanced, details=details)
            return (enhanced, *args[1:])

    def logits(_layer, _args, output):
        capture["logits"] = output

    tokens = [head.register_forward_pre_hook(inject), head.classifier.register_forward_hook(logits)]
    try:
        with torch.set_grad_enabled(training):
            result = h.runner().forward(state, h.BASE, training=training)
    finally:
        for token in tokens:
            token.remove()
    h.require("cam" in capture and "logits" in capture, "Original head was bypassed")
    if training and module is not None:
        native_rng = h.rng_sha()
        enhanced, details = module(capture["cam"], head, True, intervention)
        h.require(h.rng_sha() == native_rng, "Enhanced feature forward consumed historical RNG")
        capture.update(enhanced=enhanced, details=details)
    return result, capture


def setup(group, fold, *, verify_spectrum=False):
    state, warmup, identity, optimizers, members = h.warmup_and_membership(fold)
    before = h.rng_sha()
    model_before = h.runner().load_history().state_model_sha(state)
    archived = h.spectrum(fold, identity, members["legal"])
    _, capture = forward(state, False)
    cam = capture["cam"].cpu()
    h.require(torch.equal(cam, archived["warmup_cam"]) and h.sha(cam) == archived["feature_sha"],
              "Current E61 warmup CAM differs from archived Original O")
    spectrum = archived["spectrum"]
    text = h.archive.verify_text()
    legal = members["legal"]
    labels = state["labels_np"][legal].copy()
    labels_t = torch.from_numpy(labels).to(state["x"].device)
    with torch.no_grad():
        warm_legal_margin = margin(fixed_logits(state["cam"].Outputlayer, capture["cam"][state["legal"]]))
        difficulty = (-(1 - 2 * labels_t) * warm_legal_margin / CFG["boundary"]["error_temperature"]).sigmoid()
    h.require(difficulty.shape == labels_t.shape and torch.isfinite(difficulty).all().item(),
              "Epoch 10 difficulty must have only the legal training samples")
    if verify_spectrum:
        rebuilt = h.archive.psce().build_spectra(cam[legal], torch.from_numpy(labels))
        h.require(all(torch.equal(rebuilt[key], spectrum[key]) for key in h.archive.SPECTRUM_KEYS),
                  "Legal Original O reconstruction differs")
    reference = fit_reference(fold, cam, spectrum, legal, labels) if group == "G2" else None
    # Constructors allocate on CPU. No CUDA generator is seeded or used here.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CFG["seed_stream"][group] * 100 + fold)
        if group == "G1":
            module = PCSRPSCE(spectrum, text, state["excluded"])
        else:
            module = SCMRDASRPE(spectrum, text, state["excluded"], reference)
    module.to(state["x"].device)
    new = torch.optim.AdamW(module.parameters(), **CFG["new_optimizer"])
    ownership = families(state, module)
    h.require(set(p for pg in new.param_groups for p in pg["params"]) ==
              set(ownership["prompt"]) | set(ownership["control"]), "New AdamW ownership differs")
    h.require(h.rng_sha() == before == warmup["rng_sha"]
              and h.runner().load_history().state_model_sha(state) == model_before,
              "Warmup extraction, fitting or initialization changed historical RNG or model")
    provenance = {
        "identity": identity, "optimizers": optimizers, "membership": h.sha(members),
        "warm_cam": archived["feature_sha"], "spectrum": archived["spectra_sha"],
        "basis": archived["e82_original_U_sha"], "prompt": text["identity_sha"], "native_rng": before,
        "legal_index_sha": h.sha(legal), "memory_sha": h.sha(difficulty.cpu()),
        "reference": h.sha(reference), "initial_module": h.sha(module.state_dict()),
        "parameters": {name: sum(p.numel() for p in ps if p.requires_grad) for name, ps in ownership.items()},
    }
    return {"group": group, "fold": fold, "state": state, "module": module,
            "optimizer": new, "warmup": warmup, "members": members, "provenance": provenance,
            "labels": labels_t, "hard_initial": difficulty, "reference": reference,
            "original": spectrum, "warm_cam": cam, "families": ownership}


def ema(ctx):
    return ctx["module"].statistics.snapshot() if ctx["group"] == "G2" else None


def restore_ema(ctx, state):
    if ctx["group"] == "G2":
        ctx["module"].statistics.restore(state)
    else:
        h.require(state is None, "G1 unexpectedly has EMA")


def bundle(ctx):
    return {"native": h.base_bundle(ctx["state"]), "module": ctx["module"].state_dict(),
            "optimizer": ctx["optimizer"].state_dict()}


def restore(ctx, payload, implementation=None):
    h.require((payload["group"], payload["fold"], payload["seed"]) == (ctx["group"], ctx["fold"], CFG["seed"])
              and payload["provenance"] == ctx["provenance"]
              and h.sha(payload["membership"]) == ctx["provenance"]["membership"], "Checkpoint identity differs")
    if implementation is not None:
        h.require(payload["implementation"] == implementation and payload["config"] == CFG, "Checkpoint freeze differs")
    h.require(h.sha(payload["hard_initial"]) == ctx["provenance"]["memory_sha"],
              "Checkpoint legal difficulty memory differs")
    data = payload["state"]
    h.require(h.sha(data) == payload["state_sha"], "Checkpoint state bytes differ")
    h.restore_base(ctx["state"], data["native"])
    ctx["module"].load_state_dict(data["module"], strict=True)
    ctx["optimizer"].load_state_dict(deepcopy(data["optimizer"]))
    restore_ema(ctx, payload["ema"])
    h.base().restore_rng(payload["rng"])
    h.require(h.rng_sha() == payload["rng_sha"] and h.sha(ema(ctx)) == payload["ema_sha"],
              "Checkpoint EMA or RNG readback differs")


def train_step(ctx, *, inspect=False, enabled=True, shared=True):
    state, module = ctx["state"], ctx["module"]
    opt = [state[key] for key in h.OPTS] + [ctx["optimizer"]]
    for current in opt:
        current.zero_grad(set_to_none=True)

    with dropout_inventory() as calls:
        out, capture = forward(state, True, module if enabled else None)
        history_after_forward = h.rng_sha()
        path, routing, protections = None, {}, {}
        if not enabled:
            out["total"].backward()
            auxiliary = {}
        else:
            head = state["cam"].Outputlayer
            auxiliary, label_diagnostics = losses(ctx["group"], head, capture, state["legal"], ctx["labels"],
                                                  module.spectrum.basis.to(capture["cam"]), ctx["hard_initial"])
            if inspect:
                path = inspect_paths(ctx["families"], capture, auxiliary["full"])
                routing = distributions(capture["details"], state["legal"], label_diagnostics, ctx["group"])
            protections, updates = apply_budget(ctx["group"], ctx["families"], out, auxiliary, shared=shared)
            d = capture["details"]
            delta = d["delta"].detach()
            ratio_limit = CFG["residual"]["fraction"]
            h.require(bool(torch.isfinite(delta).all() and (d["ratio"] <= ratio_limit + 2 * torch.finfo(delta.dtype).eps).all()),
                      "Nonfinite or unbounded training residual")
            if state["excluded"]:
                h.require(torch.count_nonzero(delta[state["excluded"]]).item() == 0, "Fold 2 overlap moved")
        for current in opt[:(4 if enabled else 3)]:
            current.step()
        if enabled:
            protections = optimizer_measurements(ctx["families"], protections, updates)
    h.require(h.rng_sha() == history_after_forward, "Auxiliary calculations or optimizer changed historical RNG")
    loss_names = list(h.LOSS_KEYS) + list(auxiliary)
    loss_values = torch.stack([out[k].detach() if k in out else auxiliary[k].detach() for k in loss_names]).cpu().numpy()
    h.require(np.isfinite(loss_values).all(), "Nonfinite training loss")
    return {"losses": dict(zip(loss_names, loss_values.tolist())), "dropout": calls,
            "budget": protections, "paths": path, "routing": routing}


@torch.no_grad()
def evaluate(ctx, intervention="full"):
    state = ctx["state"]
    expected = h.rng_sha()
    with dropout_inventory() as calls:
        result, capture = forward(state, False, ctx["module"], intervention)
    rows = state["test"]
    probability = result["logp"][rows].exp().cpu().numpy()
    prediction = probability.argmax(-1)
    h.require(np.isfinite(probability).all() and h.rng_sha() == expected and not calls, "Evaluation stochastic or nonfinite")
    ratio = capture["details"]["ratio"]
    h.require(bool(torch.isfinite(capture["enhanced"]).all() and torch.isfinite(ratio).all() and
                   (ratio <= CFG["residual"]["fraction"] + 2 * torch.finfo(ratio.dtype).eps).all()),
              "Nonfinite or unbounded inference feature")
    h.require(torch.isfinite(capture["logits"]).all().item(), "Nonfinite inference logits")
    if state["excluded"]:
        h.require(torch.count_nonzero(capture["details"]["delta"][state["excluded"]]).item() == 0,
                  "Fold 2 overlap moved in evaluation")
    return {"logits": capture["logits"][rows].cpu().numpy(), "probability": probability,
            "prediction": prediction, "correct": int((prediction == state["labels_np"][state["test_np"]]).sum())}, capture


def observation_and_advance(ctx, capture):
    if ctx["group"] != "G2":
        return None
    before = h.rng_sha()
    statistics = ctx["module"].statistics
    observation = statistics.observation(capture["cam"])
    statistics.advance(observation)
    h.require(h.rng_sha() == before, "EMA update altered historical RNG")
    return observation


def transition(ctx, epoch, dropout, *, inspect=False):
    before_rng = h.rng_sha()
    before_ema = ema(ctx)
    trained = train_step(ctx, inspect=inspect)
    trained_rng = h.rng_sha()
    h.require(trained["dropout"] == dropout, "Historical Dropout call sequence changed")
    result, capture = evaluate(ctx)
    after_rng = h.rng_sha()
    observed = observation_and_advance(ctx, capture)
    rng = {"epoch": epoch, "rng_before": before_rng, "rng_after_train": trained_rng, "rng_after_eval": after_rng}
    return result, trained, rng, before_ema, observed


def pack_gradient_budget(budget):
    # Named schema in optimization.py lets the audit parse this compact ledger.
    return {name: [v[k] for k in ORIGINAL_LOG] if name in ORIGINAL else
            [v["main"], v["total_applied"], *(
                number for term, value in v.items() if term not in ("main", "total_applied")
                for number in (value["raw"], value["applied"], value["scale"]))]
            for name, v in budget.items()}


def next_witness(ctx, epoch, dropout, *, inspect=False):
    result, diag, rng, previous, observed = transition(ctx, epoch, dropout, inspect=inspect)
    return {"epoch": epoch, "rng": rng, "training_sha": h.sha(diag),
            "state_sha": h.sha(bundle(ctx)), "evaluation_sha": h.sha(result),
            "previous_ema": h.sha(previous), "observation": h.sha(observed), "ema": h.sha(ema(ctx))}


def metadata(ctx, implementation):
    return {"group": ctx["group"], "fold": ctx["fold"], "seed": CFG["seed"], "config": CFG,
            "implementation": implementation, "provenance": ctx["provenance"],
            "membership": ctx["members"], "hard_initial": ctx["hard_initial"].detach().cpu()}


def snapshot(ctx, fixed):
    state, stats = bundle(ctx), ema(ctx)
    return {**fixed, "state": state, "state_sha": h.sha(state), "ema": stats,
            "ema_sha": h.sha(stats), "rng": h.base().pack_rng(), "rng_sha": h.rng_sha()}


def durable_journal(stream):
    stream.flush()
    os.fsync(stream.fileno())


def latest_snapshot(files, stream, digest, payload, previous, best_sha):
    durable_journal(stream)
    h.save(files["latest"], {**payload, "eval_ema": previous, "best_sha": best_sha,
                            "journal_size": stream.tell(), "journal_sha": digest.hexdigest()})


def checkpoint_prefix(files, saved, raw):
    """Verify the durable Latest prefix and any later separately committed Best."""
    prefix = raw[:saved["journal_size"]]
    digest = hashlib.sha256(prefix)
    h.require(len(prefix) == saved["journal_size"] and digest.hexdigest() == saved["journal_sha"]
              and len(prefix.splitlines()) == saved["epoch"] - START, "Latest journal prefix cannot be verified")
    actual_sha = h.sha_file(files["best"]) if files["best"].is_file() else None
    if actual_sha == saved["best_sha"]:
        return digest, None, []
    h.require(actual_sha is not None, "Latest points at an unavailable Best")
    best = h.load(files["best"])
    epoch = best["epoch"]
    h.require(saved["epoch"] < epoch <= saved["epoch"] + CFG["latest_interval"]
              and best["latest_base_epoch"] == saved["epoch"] and best["latest_base_sha"] == saved["journal_sha"]
              and best["implementation"] == saved["implementation"] and best["provenance"] == saved["provenance"]
              and best["group"] == saved["group"] and best["fold"] == saved["fold"]
              and best["best_epoch"] == epoch and best["best_correct"] > saved["best_correct"],
              "Best ahead of Latest does not belong to the committed continuation")
    suffix = best["journal_suffix"]
    check = digest.copy()
    check.update(suffix)
    records = suffix.splitlines(keepends=True)
    parsed = [json.loads(line) for line in records]
    h.require(check.hexdigest() == best["journal_sha"] and
              [r["epoch"] for r in parsed] == list(range(saved["epoch"] + 1, epoch + 1)),
              "Persisted post-Latest Best suffix is incomplete")
    last = parsed[-1]
    h.require(last["improved"] and last["correct"] == best["best_correct"]
              and last["probability_sha"] == h.sha(best["evaluation"]["probability"])
              and last["checkpoint_state_sha"] == best["state_sha"] == h.sha(best["state"])
              and last["ema_before"] == best["ema_sha"] == h.sha(best["ema"])
              and last["rng"]["rng_after_eval"] == best["rng_sha"] == h.base().rng_sha(best["rng"]),
              "The independently committed Best and journal differ")
    return digest, actual_sha, records


def train_fold(group, fold):
    manifest = h.read_json(MANIFEST)
    hashes = manifest["implementation"]
    files = paths(group, fold)
    started = time.monotonic()
    h.require(hashes == h.implementation() and manifest["preflight"]["status"] == "PASS",
              "Formal training must match a passed frozen preflight")
    preflight = h.read_json(files["preflight"])
    h.require(preflight["status"] == "PASS" and preflight["implementation"] == hashes,
              "Fold preflight is unavailable or incompatible")
    if files["task"].is_file():
        finished = h.read_json(files["task"])
        h.require(finished["status"] == "TRAINED" and hashes == finished["implementation"] and
                  all(h.sha_file(files[k]) == finished[k + "_sha"] for k in ("best", "latest", "journal")),
                  "Cannot skip a damaged completed fold")
        print(f"Verified previously completed {group} fold {fold}", flush=True)
        return
    ctx = setup(group, fold)
    h.require(preflight["provenance"] == ctx["provenance"], "Formal warmup differs from preflight")
    archive = h.hist_trace(fold)
    fixed = metadata(ctx, hashes)
    torch.cuda.reset_peak_memory_stats()
    files["journal"].parent.mkdir(parents=True, exist_ok=True)

    with files["journal"].open("a+b") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ("best", "latest", "task"):
            files[key].with_name(files[key].name + ".tmp").unlink(missing_ok=True)
        completed, best_epoch, best_correct, wait, best_sha = START, 0, -1, 0, None
        future_best_sha, replay = None, []
        if files["latest"].is_file():
            saved = h.load(files["latest"])
            journal.seek(0)
            digest, future_best_sha, replay = checkpoint_prefix(files, saved, journal.read())
            journal.seek(saved["journal_size"])
            journal.truncate()
            restore(ctx, saved, hashes)
            completed, best_epoch, best_correct, wait, best_sha = (
                saved[k] for k in ("epoch", "best_epoch", "best_correct", "wait", "best_sha"))
            del saved
            print(f"Resumed {group} fold {fold} from epoch {completed}", flush=True)
        else:
            h.require(not files["best"].exists(), "Best exists without an initial Latest")
            journal.seek(0)
            journal.truncate()
            digest = hashlib.sha256()
            initial = snapshot(ctx, fixed)
            initial.update(epoch=START, best_epoch=0, best_correct=-1, wait=0)
            latest_snapshot(files, journal, digest, initial, initial["ema"], None)
            del initial
        latest_base_epoch = completed
        latest_base_sha = digest.hexdigest()
        persisted_suffix = bytearray()

        for epoch in range(completed + 1, CFG["max_epochs"] + 1):
            if wait > CFG["early_stop_wait_gt"]:
                break
            result, diag, rng, previous, observed = transition(
                ctx, epoch, preflight["dropout"], inspect=epoch in (START + 1, START + 10))
            h.require(rng["rng_after_train"] == rng["rng_after_eval"] and (epoch not in archive or rng == archive[epoch]),
                      f"Historical RNG mismatch {group} fold {fold} epoch {epoch}")
            improved = result["correct"] > best_correct
            if improved:
                best_epoch, best_correct, wait = epoch, result["correct"], 0
            else:
                wait += 1
            completed = epoch
            latest_due = epoch % CFG["latest_interval"] == 0 or (
                epoch == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"])
            payload = snapshot(ctx, fixed) if improved or latest_due else None
            observed_bytes = None if observed is None else base64.b64encode(
                observed.numpy().astype("<f8", copy=False).tobytes()).decode("ascii")
            record = {"epoch": epoch, "rng": rng, "correct": result["correct"], "best_epoch": best_epoch,
                      "best_correct": best_correct, "wait": wait,
                      "prediction_bits": np.packbits(result["prediction"]).tobytes().hex(),
                      "probability_sha": h.sha(result["probability"]), "losses": diag["losses"],
                      "budget": pack_gradient_budget(diag["budget"]), "routing": diag["routing"], "paths": diag["paths"],
                      "ema_before": h.sha(previous), "ema_after": h.sha(ema(ctx)), "ema_observation": observed_bytes,
                      "checkpoint_state_sha": None if payload is None else payload["state_sha"], "improved": improved}
            if epoch == START + 1:
                h.require(h.sha((result, diag, rng, h.sha(ema(ctx)))) == preflight["first_step_sha"],
                          "Formal first transition differs from preflight")
            line = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
            replaying = bool(replay)
            if replaying:
                h.require(line == replay.pop(0), "Replayed post-Latest epoch differs from durable Best history")
            journal.write(line)
            digest.update(line)
            persisted_suffix.extend(line)
            if payload is not None:
                payload.update(epoch=epoch, best_epoch=best_epoch, best_correct=best_correct, wait=wait,
                               evaluation=result, observation_sha=h.sha(observed))
            if improved and not replaying:
                durable_journal(journal)
                h.save(files["best"], {**payload, "ema": previous, "ema_sha": h.sha(previous),
                                       "latest_base_epoch": latest_base_epoch, "latest_base_sha": latest_base_sha,
                                       "journal_suffix": bytes(persisted_suffix), "journal_sha": digest.hexdigest()})
                best_sha = h.sha_file(files["best"])
            if replaying and not replay:
                h.require(improved and future_best_sha is not None, "Recovered Best epoch did not improve")
                best_sha = future_best_sha
                durable_journal(journal)
            if latest_due:
                h.require(not replay and best_sha is not None, "Unverified Best at the periodic checkpoint")
                latest_snapshot(files, journal, digest, payload, previous, best_sha)
                latest_base_epoch, latest_base_sha = epoch, digest.hexdigest()
                persisted_suffix.clear()
            if epoch == START + 1 or epoch % 50 == 0:
                print(f"{group} fold {fold:02d} epoch {epoch} best {best_correct}@{best_epoch} wait {wait}", flush=True)

    h.require(best_epoch > START and (completed == CFG["max_epochs"] or wait > CFG["early_stop_wait_gt"]),
              "Fold stopped before historical completion condition")
    # A terminal in-memory transition serves as a witness for a separate reload process.
    witness = next_witness(ctx, completed + 1, preflight["dropout"], inspect=True)
    summary = {
        "status": "TRAINED", "group": group, "fold": fold, "epoch": completed, "best_epoch": best_epoch,
        "best_correct": best_correct, "wait": wait, "best_sha": best_sha, "latest_sha": h.sha_file(files["latest"]),
        "journal_sha": digest.hexdigest(), "implementation": hashes, "wall_seconds": time.monotonic() - started,
        "peak_mib": torch.cuda.max_memory_reserved() / 2**20, "affinity": sorted(os.sched_getaffinity(0)),
        "next_witness": witness,
    }
    h.require(best_sha == h.sha_file(files["best"]) and summary["journal_sha"] == h.sha_file(files["journal"]),
              "Terminal Best or journal digest differs")
    h.write_json(files["task"], summary)
    print(f"TRAINED {group} fold {fold:02d} end {completed} best {best_correct}@{best_epoch}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=h.GROUPS)
    parser.add_argument("fold", type=int, choices=h.FOLDS)
    args = parser.parse_args()
    train_fold(args.group, args.fold)


if __name__ == "__main__":
    main()
