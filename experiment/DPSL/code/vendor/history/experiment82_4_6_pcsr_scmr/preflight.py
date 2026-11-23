"""Verify historical parity, authorized data, budgets and recovery before training."""
from __future__ import annotations

import argparse
import io
import hashlib
import json
import tempfile
from pathlib import Path

from common import history as h
from common.optimization import ORIGINAL, losses
from common.spectral import native_logits
from audit import independent_head_ce, twin_probe
import forensic as f
import run

import numpy as np
import torch


def restored_payload(ctx, impl, **extra):
    stream = io.BytesIO()
    torch.save({**run.snapshot(ctx, run.metadata(ctx, impl)), **extra}, stream)
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=False)


def historic_signature(ctx, wrapped, active=False):
    state = ctx["state"]
    for key in h.OPTS:
        state[key].zero_grad(set_to_none=True)
    before = h.rng_sha()
    with run.dropout_inventory() as calls:
        module = ctx["module"] if active else None
        out = run.forward(state, True, module)[0] if wrapped else h.runner().forward(state, h.BASE, training=True)
    fields = (*h.LOSS_KEYS, "fused", "logp", "hidden", "original_adjs", "unified_adjs")
    signature = {"outputs": h.sha({key: out[key] for key in fields})}
    out["total"].backward()
    signature["gradients"] = h.sha([p.grad for k in ("model", "cam") for p in state[k].parameters()])
    for key in h.OPTS:
        state[key].step()
    signature["after_update"] = h.sha(h.base_bundle(state))
    after = h.rng_sha()
    with torch.no_grad():
        signature["test"] = h.sha(h.runner().forward(state, h.BASE, training=False)["logp"][state["test"]])
    return signature, calls, {"epoch": 11, "rng_before": before, "rng_after_train": after, "rng_after_eval": h.rng_sha()}


def legal_label_isolation(ctx, initial):
    state = ctx["state"]
    legal = ctx["members"]["legal"]
    test = ctx["members"]["test"]
    overlap = np.intersect1d(ctx["members"]["train"], test)
    h.require(not np.intersect1d(legal, test).size and overlap.tolist() == state["excluded"],
              "New supervision has held-out patients")
    f.equal(ctx["labels"], state["y"][state["legal"]], "Legal label dictionary")
    test_order = state["test"]
    labels_t, labels_np = state["y"][test_order].clone(), state["labels_np"][test].copy()

    def inspect():
        h.base().restore_rng(initial["rng"])
        _, cap = run.forward(state, True, ctx["module"])
        basis = ctx["module"].spectrum.basis.to(cap["cam"])
        aux, _ = losses(ctx["group"], state["cam"].Outputlayer, cap, state["legal"], ctx["labels"], basis, ctx["hard_initial"])
        for term, value in aux.items():
            grad, = torch.autograd.grad(value, cap["cam"], retain_graph=True)
            h.require(not grad[test_order].count_nonzero().item(), f"{term} has a held-out CAM gradient")
        info = cap["details"]
        data = [info[k] for k in ("delta", "gate", "capacity")] + [ctx["hard_initial"]]
        if ctx["group"] == "G2":
            data.extend(info[k] for k in ("llr", "reliability", "evidence", "target_margin", "desired_change",
                                          "disagreement", "correction_reliability", "allocation"))
            data.append(ctx["module"].statistics.observation(cap["cam"]))
        else:
            data.extend(info[k] for k in ("rotation_angles", "amplitude", "action", "calibrated_contrast"))
        return h.sha((data, aux))

    expected = inspect()
    try:
        state["y"][test_order] = 1 - labels_t
        state["labels_np"][test] = 1 - labels_np
        h.require(inspect() == expected, "Held-out labels altered a new supervised or inference path")
    finally:
        state["y"][test_order] = labels_t
        state["labels_np"][test] = labels_np
        h.base().restore_rng(initial["rng"])
    return {"test_label_inversion_exact": True, "new_cam_gradient_test_rows_zero": True,
            "legal_train": int(len(legal)), "train": int(len(ctx["members"]["train"])),
            "test": int(len(test)), "legal_index_sha": h.sha(legal), "overlap_1based": (overlap + 1).tolist()}


def future_best_preflight(initial, evaluation, after, current_ema, rng):
    zero = hashlib.sha256(b"").hexdigest()
    r = {"epoch": 11, "improved": True, "correct": evaluation["correct"], "rng": rng,
         "probability_sha": h.sha(evaluation["probability"]), "checkpoint_state_sha": after["state_sha"],
         "ema_before": h.sha(initial["ema"]), "ema_after": current_ema}
    line = (json.dumps(r) + "\n").encode()
    with tempfile.TemporaryDirectory(dir=run.RUNTIME) as scratch:
        best = Path(scratch) / "best.pt"
        h.save(best, {**after, "latest_base_epoch": 10, "latest_base_sha": zero, "journal_suffix": line,
                      "journal_sha": hashlib.sha256(line).hexdigest(), "evaluation": evaluation,
                      "ema": initial["ema"], "ema_sha": r["ema_before"]})
        latest = {**initial, "journal_size": 0, "journal_sha": zero, "best_sha": None}
        for raw in (line, b""):
            digest, future, suffix = run.checkpoint_prefix({"best": best}, latest, raw)
            h.require(digest.hexdigest() == zero and future == h.sha_file(best) and suffix == [line],
                      "Best ahead of durable Latest failed a repeat-interruption recovery")

def preflight_fold(group, fold):
    hashes = h.implementation()
    device = h.archive.verify_device()
    h.hist_config()
    ctx = run.setup(group, fold, verify_spectrum=True)
    torch.cuda.reset_peak_memory_stats()
    initial = restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    baseline = historic_signature(ctx, wrapped=False)
    run.restore(ctx, initial, hashes)
    h.require(historic_signature(ctx, wrapped=True, active=True) == baseline,
              "The enabled Twin-Path feature calculation changed the native objective or stochastic graph")
    run.restore(ctx, initial, hashes)
    disabled = run.train_step(ctx, enabled=False)
    h.require(h.sha(h.base_bundle(ctx["state"])) == baseline[0]["after_update"]
              and h.rng_sha() == baseline[2]["rng_after_train"], "Disabled training entry changed native update")
    archive = h.hist_trace(fold)
    h.require(len(baseline[1]) == 6 and baseline[2] == archive[11], "Historical first stochastic epoch differs")
    run.restore(ctx, initial, hashes)
    geometry = f.check_original_o(ctx)
    difficulty = f.legal_memory(ctx)
    banks = f.class_banks(ctx) if group == "G2" else None
    labels = legal_label_isolation(ctx, initial)
    run.restore(ctx, initial, hashes)
    state = ctx["state"]
    head = state["cam"].Outputlayer
    with torch.no_grad():
        _, zero_cap = run.evaluate(ctx)
        initial_formula = f.branch_formulas(ctx, zero_cap)
        d = zero_cap["details"]
        f.equal(zero_cap["enhanced"], zero_cap["cam"], "Natural zero initialization preserves CAM exactly")
        f.equal(d["delta"], torch.zeros_like(d["delta"]), "Natural initial residual is exactly zero")
        if group == "G1":
            f.equal(d["rotation_angles"], torch.zeros_like(d["rotation_angles"]), "Initial patient rotations")
            f.equal(d["rotated_coordinates"], d["direction_coordinates"], "Initial rotation is identity")
            h.require(bool(d["action"].count_nonzero()), "Zero signed magnitude needs a nonzero direction")
        else:
            f.equal(d["target_margin"], d["margin"], "Initial calibrated target must be native margin")
            f.equal(d["desired_change"], torch.zeros_like(d["desired_change"]), "Initial desired change")
            h.require(bool((d["correction_reliability"] > 0).all()), "Nonzero trust is needed at initialization")

    run.restore(ctx, initial, hashes)
    result, first, phase, _, _ = run.transition(ctx, 11, baseline[1], inspect=True)
    h.require(phase == archive[11] and h.sha(h.base_bundle(state)) == baseline[0]["after_update"],
              "Natural zero residual changed the historical Adam update or RNG")
    h.require(first["losses"]["full"] == first["losses"]["off"] and first["losses"]["difference"] == 0 and
              all(first["budget"][name]["raw"] == first["budget"][name]["total_applied"] == 0 for name in ORIGINAL) and
              first["paths"]["zero_residual"] and first["paths"]["from_full_ce_through_delta"]["control"] > 0,
              "Natural zero residual must exactly cancel shared gradients and train the new controller")
    all_rng_good = h.rng_sha() == phase["rng_after_eval"]
    first_sig = h.sha((result, first, phase, h.sha(run.ema(ctx))))
    saved = restored_payload(ctx, hashes, epoch=11, wait=0, best_epoch=11, best_correct=result["correct"])
    if fold == 1:
        future_best_preflight(initial, result, saved, h.sha(run.ema(ctx)), phase)
    witness = run.next_witness(ctx, 12, baseline[1])
    run.restore(ctx, saved, hashes)
    h.require(witness == run.next_witness(ctx, 12, baseline[1]) and witness["rng"] == archive[12],
              "Serialized preflight recovery differs on next train/eval")
    ready = restored_payload(ctx, hashes, epoch=12)
    with torch.no_grad():
        _, cap = run.evaluate(ctx)
        eval_formula = f.branch_formulas(ctx, cap)
        amplitude = f.amplitude_probe(ctx, cap["cam"])
        f.equal(native_logits(head, cap["enhanced"]), cap["logits"], "Deterministic Full head is the original eval head")
        by_pass, native = run.forward(state, False)
        correction_off, off = run.evaluate(ctx, "residual_off")
        f.equal(native["cam"], off["enhanced"], "Residual-off preserves actual CAM input")
        f.equal(native["logits"], off["logits"], "Residual-off uses native inference")
        f.equal(by_pass["logp"][state["test"]].exp(), correction_off["probability"], "Native eval prediction")
        if group == "G1":
            _, without = run.evaluate(ctx, "differential_prompt_off")
            f.equal(without["details"]["attention"], without["details"]["original_attention"],
                    "Disabling differential Prompt should retain all eight unmodified queries")
            f.branch_formulas(ctx, without, intervention="differential_prompt_off")
            _, without = run.evaluate(ctx, "patient_rotation_off")
            f.equal(without["details"]["rotated_coordinates"], without["details"]["direction_coordinates"],
                    "Disabling patient rotation preserves the Original O coordinates")
            f.branch_formulas(ctx, without, intervention="patient_rotation_off")
        else:
            _, without = run.evaluate(ctx, "calibrated_evidence_off")
            for k in ("correction_reliability", "allocation"):
                f.equal(cap["details"][k], without["details"][k], f"Removing calibrated target altered {k}")
            f.branch_formulas(ctx, without, intervention="calibrated_evidence_off")

    with run.dropout_inventory() as calls:
        _, forced = run.forward(state, True, ctx["module"], intervention="residual_off")
    h.require(calls == baseline[1], "Mature zero-residual probe changed historical Dropout")
    rows, y = state["legal"], ctx["labels"]
    full = independent_head_ce(head, forced["enhanced"][rows], y)
    off = independent_head_ce(head, forced["cam"][rows], y)
    f.equal(full, off, "Mature residual-off differential")
    historical = [p for family in ORIGINAL for p in ctx["families"][family] if p.requires_grad]
    gradients = torch.autograd.grad(full - off, historical, allow_unused=True)
    h.require(all(v is None or not v.count_nonzero() for v in gradients), "Mature zero residual has a shared gradient")
    run.restore(ctx, ready, hashes)
    twin = twin_probe(ctx, baseline[1], check_each=True)
    h.require(all(twin["gradients"][family]["actual_add_norm"] > 0 for family in ORIGINAL),
              "Nonzero-residual differential did not reach every historical optimizer family")
    record = {"status": "PASS", "group": group, "fold": fold, "implementation": hashes, "device": device["uuid"],
              "provenance": ctx["provenance"], "dropout": baseline[1], "disabled_native_exact": baseline[0],
              "natural_zero_gradient_and_update_exact": True, "mature_forced_zero_gradient_exact": True, "labels": labels, "rng_matches": all_rng_good,
              "historical_first_rng": phase, "initial_ema": h.sha(initial["ema"]), "archive_epochs": len(archive),
              "orig_o": geometry, "banks": banks, "difficulty": difficulty, "gradient_optimizer_audit": twin,
              "initial_formulas": initial_formula, "inference_formulas": eval_formula, "action_amplitude": amplitude, "future_best_recovery_test": fold == 1,
              "first_step": first_sig, "first_step_diagnostics": first, "next_step_exact": witness,
              "first_step_sha": first_sig, "gpu_peak_mib": torch.cuda.max_memory_reserved() / 2**20}
    h.write_json(run.paths(group, fold)["preflight"], record)
    print(f"PASS preflight {group} fold {fold:02d} with exact reload, residual-differential gradient budgets and historical RNG", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fold", type=int, choices=h.FOLDS)
    p.add_argument("--group", choices=h.GROUPS)
    args = p.parse_args()
    for group in (args.group,) if args.group else h.GROUPS:
        preflight_fold(group, args.fold)


if __name__ == "__main__":
    main()
