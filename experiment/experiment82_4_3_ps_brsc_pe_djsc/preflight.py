"""Verify historical parity, authorized data, budgets and recovery before training."""
from __future__ import annotations

import argparse
import io

from common import history as h
from common.optimization import losses
import forensic as f
import run

import numpy as np
import torch


def restored_payload(ctx, impl, **extra):
    stream = io.BytesIO()
    torch.save({**run.snapshot(ctx, run.metadata(ctx, impl)), **extra}, stream)
    stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=False)


def historic_signature(ctx, wrapped):
    state = ctx["state"]
    for key in h.OPTS:
        state[key].zero_grad(set_to_none=True)
    before = h.rng_sha()
    with run.dropout_inventory() as calls:
        out = run.forward(state, True)[0] if wrapped else h.runner().forward(state, h.BASE, training=True)
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
        aux, _ = losses(ctx["group"], state["cam"].Outputlayer, cap, state["legal"], ctx["labels"])
        native, = torch.autograd.grad(aux["native"], cap["cam"], retain_graph=True)
        h.require(not native[test_order].count_nonzero().item(), "Native loss has a held-out CAM gradient")
        for term, value in aux.items():
            if term != "native":
                grad, = torch.autograd.grad(value, cap["cam"], retain_graph=True)
                h.require(not grad[test_order].count_nonzero().item(), f"{term} has a held-out CAM gradient")
        info = cap["details"]
        data = [info[k] for k in ("delta", "gate", "feasibility")]
        if ctx["group"] == "G2":
            data.extend(info[k] for k in ("llr", "reliability", "evidence"))
            data.append(ctx["module"].statistics.observation(cap["cam"]))
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
            "test": int(len(test)), "overlap_1based": (overlap + 1).tolist()}


def preflight_fold(group, fold):
    hashes = h.implementation()
    device = h.archive.verify_device()
    h.hist_config()
    ctx = run.setup(group, fold, verify_spectrum=True)
    torch.cuda.reset_peak_memory_stats()
    initial = restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    baseline = historic_signature(ctx, wrapped=False)
    run.restore(ctx, initial, hashes)
    h.require(historic_signature(ctx, wrapped=True) == baseline, "Disabled extension changed historical DARG behavior")
    run.restore(ctx, initial, hashes)
    disabled = run.train_step(ctx, enabled=False)
    h.require(h.sha(h.base_bundle(ctx["state"])) == baseline[0]["after_update"]
              and h.rng_sha() == baseline[2]["rng_after_train"], "Disabled training entry changed native update")
    archive = h.hist_trace(fold)
    h.require(len(baseline[1]) == 6 and baseline[2] == archive[11], "Historical first stochastic epoch differs")
    run.restore(ctx, initial, hashes)
    geometry = f.check_original_o(ctx)
    banks = f.class_banks(ctx) if group == "G2" else None
    labels = legal_label_isolation(ctx, initial)
    with torch.no_grad():
        _, capture = run.forward(ctx["state"], True, ctx["module"])
        formula = f.branch_formulas(ctx, capture, training=True)
    run.restore(ctx, initial, hashes)
    with torch.no_grad():
        _, capture = run.evaluate(ctx)
        eval_formula = f.branch_formulas(ctx, capture)
        semantic1, _ = run.evaluate(ctx, "semantic_off")
        module = ctx["module"]
        saved_query = module.spectrum.query_medical.clone()
        module.spectrum.query_medical.copy_(module.spectrum.query_anonymous)
        try:
            semantic2, _ = run.evaluate(ctx, "semantic_off")
        finally:
            module.spectrum.query_medical.copy_(saved_query)
        h.require(h.sha(semantic1) == h.sha(semantic2), "Semantic-off still depends on Query identity")

    run.restore(ctx, initial, hashes)
    result, first, phase, _, _ = run.transition(ctx, 11, baseline[1], inspect=True)
    h.require(phase == archive[11], "Extension changed historic RNG at first step")
    all_rng_good = h.rng_sha() == phase["rng_after_eval"]
    first_sig = h.sha((result, first, phase, h.sha(run.ema(ctx))))
    saved = restored_payload(ctx, hashes, epoch=11, wait=0, best_epoch=11, best_correct=result["correct"])
    witness = run.next_witness(ctx, 12, baseline[1])
    run.restore(ctx, saved, hashes)
    h.require(witness == run.next_witness(ctx, 12, baseline[1]) and witness["rng"] == archive[12],
              "Serialized preflight recovery differs on next train/eval")
    run.restore(ctx, initial, hashes)
    no_native = run.train_step(ctx, inspect=False, native_enabled=False)
    h.require(all(no_native["budget"][family]["native"]["applied"] == 0 for family in ("gfr", "darg", "cam", "head"))
              and no_native["dropout"] == baseline[1] == disabled["dropout"]
              and h.rng_sha() == phase["rng_after_train"], "Native auxiliary-off intervention differs")
    record = {"status": "PASS", "group": group, "fold": fold, "implementation": hashes, "device": device["uuid"],
              "provenance": ctx["provenance"], "dropout": baseline[1], "disabled_native_exact": baseline[0],
              "native_objective_disabled_zero": True, "labels": labels, "rng_matches": all_rng_good,
              "historical_first_rng": phase, "initial_ema": h.sha(initial["ema"]), "archive_epochs": len(archive),
              "orig_o": geometry, "banks": banks, "training_formulas": formula, "inference_formulas": eval_formula,
              "first_step": first_sig, "first_step_diagnostics": first, "next_step_exact": witness,
              "first_step_sha": first_sig, "gpu_peak_mib": torch.cuda.max_memory_reserved() / 2**20}
    h.write_json(run.paths(group, fold)["preflight"], record)
    print(f"PASS preflight {group} fold {fold:02d} with exact reload, Native CE and historical RNG", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fold", type=int, choices=h.FOLDS)
    p.add_argument("--group", choices=h.GROUPS)
    args = p.parse_args()
    for group in (args.group,) if args.group else h.GROUPS:
        preflight_fold(group, args.fold)


if __name__ == "__main__":
    main()
