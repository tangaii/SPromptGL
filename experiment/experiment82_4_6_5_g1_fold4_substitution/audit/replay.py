"""Independently replay Best, Latest, RNG and patient interventions from frozen sources."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

from evaluate_checkpoint import ROOT, local_core as core, local_ind as ind, np, torch
from execute import verify_guard

h = core.h
EXPERIMENTS = ROOT.parent
DIRECTORIES = {
    "base": EXPERIMENTS / "experiment82_4_6_pcsr_scmr",
    "e463": Path(core.CONFIG["source_e82463_directory"]),
    "e464": Path(core.CONFIG["source_e8246_directory"]),
    "e462": Path(core.CONFIG["full_path_directory"]),
    "local": ROOT,
}
CURRENT = {fold: {2: "e464", 3: "e463", 4: "local"}.get(fold, "base") for fold in range(1, 11)}


def load_file(name, path):
    # The imported historical loader installs a process-level redirection from
    # /model to the root repository. These authoritative files are on /model.
    spec = core.prior.spec_from_file_location(name, path)
    module = core.prior.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    h.require(Path(module.__file__).resolve() == path.resolve(), "Unexpected historical source resolution")
    return module


def load_manifests():
    manifests = {kind: h.read_json(root / "manifest.json") for kind, root in DIRECTORIES.items()}
    for kind, manifest in manifests.items():
        root = DIRECTORIES[kind]
        h.require(h.sha(manifest["implementation"]) == manifest["implementation_sha"] and
                  all(h.sha_file(root / name) == digest for name, digest in manifest["implementation"].items()),
                  f"{kind} frozen implementation differs")
        if kind != "local":
            h.require(manifest["status"] == "COMPLETE_AUDITED", f"{kind} is not an audited historical release")
    return manifests


def guards(manifests):
    source = {}
    for manifest in manifests.values():
        for name, digest in manifest["protected_sha256"].items():
            h.require(name not in source or source[name] == digest, f"Conflicting SHA commitments for {name}")
            source[name] = digest
    exact = h.archive.hash_files([Path(name) for name in source])
    h.require(exact == source, "A protected root, data or experiment file does not match its archived SHA")
    return {"status": "PASS", "files": len(source), "bundle_sha": h.sha(source)}


def source_run(kind, fold):
    if kind == "base":
        return core.parent, "G2"
    if kind == "e463":
        adapter = load_file(f"replay_darg_only_{fold}", DIRECTORIES[kind] / "darg_only.py")
        return adapter.engine(fold), "G2"
    if kind == "e464":
        adapter = load_file(f"replay_cumulative_{fold}", DIRECTORIES[kind] / "core.py")
        return adapter.engine("G1", fold), "G1"
    return core.engine("G1", fold), "G1"


def file_commitment(kind, fold, manifest, files):
    task_key = files["best"].name.split(".")[0]
    if kind == "base":
        task = next(r for r in manifest["completed_folds"]["G2"] if r["fold"] == fold)
        preflight_sha = manifest["preflight"]["sha256"][str(files["preflight"])]
    elif kind == "e463":
        task = manifest["completed_folds"][str(fold)]
        preflight_sha = manifest["preflight"]["folds"][str(fold)]["sha"]
    elif kind == "e464":
        task = manifest["completed_tasks"][task_key]
        preflight_sha = manifest["preflight"]["tasks"][task_key]["sha"]
    else:
        task = manifest["completed_task"]
        preflight_sha = manifest["preflight"]["sha"]
    h.require(task == h.read_json(files["task"]) and task["implementation"] == manifest["implementation"],
              f"{task_key} is not bound to its committed training receipt")
    preflight = h.read_json(files["preflight"])
    actual_preflight = h.sha_file(files["preflight"])
    if actual_preflight != preflight_sha and kind == "local":
        # This field was added to the audit-side preflight receipt after training.
        # Recover and verify the original serialisation and derive its value anew.
        for separators in (None, (",", ":")):
            candidate = {k: v for k, v in preflight.items() if k != "initial_ema"}
            original = json.dumps(candidate, sort_keys=True, indent=2, separators=separators)
            if any(hashlib.sha256((original + eol).encode()).hexdigest() == preflight_sha
                   for eol in ("", "\n")):
                break
        else:
            # The frozen entry consumed the committed initial reference fingerprint
            # in provenance, the dropout inventory, and the first-step commitment.
            candidate = None
        initial_ema = preflight["provenance"]["registration"]["initial_ema_sha"]
        h.require(preflight["initial_ema"] == initial_ema, "Fold-4 audit-only EMA receipt is inconsistent")
        preflight_repair = {"before_sha": preflight_sha, "after_sha": actual_preflight,
                            "field": "initial_ema", "value": initial_ema,
                            "original_serialization_recovered": candidate is not None}
    else:
        h.require(actual_preflight == preflight_sha, f"{kind} fold {fold} preflight SHA differs")
        preflight_repair = None
    h.require(preflight["status"] == "PASS" and preflight["implementation"] == manifest["implementation"],
              f"{kind} fold {fold} preflight freeze differs")
    for name in ("best", "latest", "journal"):
        h.require(h.sha_file(files[name]) == task[name + "_sha"], f"{kind} fold {fold} {name} SHA differs")
    return task, preflight, preflight_repair


def masked_geometry(ctx, full, off, cap, expected):
    rows = np.asarray(expected, dtype=np.int64)
    test = ctx["members"]["test"]
    masked = np.isin(test, rows)
    found = torch.where(~ctx["module"].spectrum.allowed)[0].cpu().numpy()
    h.require(np.array_equal(found, np.sort(rows)) and int(masked.sum()) == len(rows), "Residual mask membership differs")
    selected = rows.tolist()
    for name in core.MASKED_COMPONENTS:
        component = cap["details"][name][selected]
        h.require(torch.equal(component, torch.zeros_like(component)), f"Masked {name} is nonzero")
    h.require(torch.equal(cap["cam"][selected], cap["enhanced"][selected]), "Masked CAM differs")
    for key in ("logits", "probability", "prediction"):
        h.require(np.array_equal(full[key][masked], off[key][masked]), f"Masked Full and Off {key} differ")
    current = {}
    for name in ("delta", "delta_parallel", "delta_perp"):
        value = cap["details"][name][test].cpu().numpy().astype(np.float64)
        current[name] = np.linalg.norm(value, axis=-1)
        h.require(np.any(current[name][~masked] > 0), f"Unmasked {name} inactive")
    cam_l2 = np.linalg.norm(cap["cam"][test].cpu().numpy().astype(np.float64), axis=-1)
    ratio = np.divide(current["delta"], cam_l2, out=np.zeros_like(cam_l2), where=cam_l2 > 0)
    h.require(np.isfinite(ratio).all() and np.all(ratio <= 0.08), "Independent 8% CAM limit exceeded")
    return current, cam_l2, ratio


def independent_scope(run, proof, ctx, dropout, expected):
    state = ctx["state"]
    module = ctx["module"]
    legal = np.asarray(ctx["members"]["legal"])
    masked = list(expected)
    with run.dropout_inventory() as calls:
        native, cap = run.forward(state, True, module)
    h.require(len(calls) == 6 and calls == dropout, "Six-call historical Dropout does not replay")
    rows = torch.as_tensor(masked, device=state["y"].device)
    aux, _ = core.optimization.losses("G2", state["cam"].Outputlayer, cap, state["legal"], ctx["labels"],
                                     module.spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
    for name, term in aux.items():
        grad, = torch.autograd.grad(term, cap["cam"], retain_graph=True)
        h.require(torch.count_nonzero(grad[rows]).item() == 0, f"Direct SCMR {name} used masked CAM rows")
    native_grad, = torch.autograd.grad(native["cls"], native["logp"], retain_graph=True)
    relation = tuple(native["info"]["pair_logits"])
    relation_grad = torch.autograd.grad(native["pair"], relation, retain_graph=True)
    derivative, = torch.autograd.grad(aux["full"], cap["details"]["delta"], retain_graph=True)
    patients = {}
    for row in masked:
        participating = (state["train_pairs"] == row).any(dim=1)
        cls_norm = float(native_grad[row].norm().item())
        pair_norm = sum(float(bit[participating].norm().item()) for bit in relation_grad)
        h.require(cls_norm > 0 and pair_norm > 0 and not derivative[row].count_nonzero().item(),
                  "Native or SCMR gradient route differs")
        patients[str(row + 1)] = {"native_class_gradient_l2": cls_norm,
                                  "native_relation_gradient_l2_sum": pair_norm,
                                  "direct_scmr_delta_gradient_l2": 0.0}
    reference, statistics = ctx["reference"], module.statistics
    legal_device = statistics.legal.detach().cpu().numpy()
    h.require(len(legal) == 784 and np.array_equal(legal, legal_device)
              and np.array_equal(legal, reference["legal"])
              and not np.isin(masked, legal).any(), "Gaussian, EMA or supervision membership differs")
    bank = proof.f.class_banks(ctx)
    observation = statistics.observation(cap["cam"])
    altered_cam = cap["cam"].detach().clone()
    altered_cam[rows] += 17.0
    h.require(torch.equal(observation, statistics.observation(altered_cam)),
              "Probe changes masked CAM contributions to EMA")
    observed = proof.f.independent_observation(ctx, cap["cam"])
    bounds = proof.f.bounded(observation, observed, 4 * len(legal), "EMA on legal CAM", 1 + abs(observed), proof.f.U64)
    return {"patients": patients, "all_direct_scmr_cam_gradient_zero": list(aux),
            "gaussian_banks": bank, "ema_observation_unchanged_after_masked_cam_probe": True,
            "ema_float64_observation_bound": bounds}


def baseline_darg(ctx, manifest):
    path, _, _, _ = h.reference_paths(ctx["fold"])["DARG"]
    digest = h.sha_file(path)
    h.require(manifest["protected_sha256"][str(path)] == digest, "DARG Best commitment differs")
    saved = h.load(path)
    state = ctx["state"]
    h.restore_base(state, saved["best_state"])
    captured = []
    handle = state["cam"].Outputlayer.classifier.register_forward_hook(
        lambda _head, _args, output: captured.append(output[state["test"]].detach().cpu().numpy()))
    before = h.rng_sha()
    try:
        with torch.no_grad():
            hist = h.runner().forward(state, h.BASE, training=False)
    finally:
        handle.remove()
    probability = hist["logp"][state["test"]].exp().detach().cpu().numpy()
    h.require(len(captured) == 1 and h.rng_sha() == before, "DARG Best inference not deterministic")
    return {"path": str(path), "sha": digest, "probability": probability.tolist(),
            "prediction": probability.argmax(-1).tolist(), "logits": captured[0].tolist()}


def array_records(result):
    return {key: np.asarray(value).tolist() for key, value in result.items()}


def review(kind, fold, manifests, *, restore_next=False, collect_darg=False):
    run, task_group = source_run(kind, fold)
    proof = load_file(f"replay_proof_{kind}_{fold}", DIRECTORIES["base"] / "audit.py")
    proof.run = run
    manifest = manifests[kind]
    files = run.paths(task_group, fold)
    task, pre, amendment = file_commitment(kind, fold, manifest, files)
    best = h.load(files["best"])
    latest = h.load(files["latest"])
    ctx = run.setup(task_group, fold, verify_spectrum=True)
    h.require(ctx["provenance"] == pre["provenance"] and best["epoch"] == task["best_epoch"]
              and latest["epoch"] == task["epoch"] and latest["best_sha"] == task["best_sha"],
              f"{kind} fold {fold} identity mismatch")
    expected = np.intersect1d(ctx["members"]["train"], ctx["members"]["test"]).tolist()
    expected_current = sorted(core.EXPOSED.get(fold, ()))
    if restore_next:
        h.require(expected == expected_current, f"Fold {fold} exposure does not match the frozen design")

    if kind == "local":
        # Replay all random, EMA and optimizer-budget receipts rather than taking
        # prior audit status at face value.
        journal = proof.audit_journal(ctx, task, best, latest, pre)
        records = [json.loads(line) for line in files["journal"].read_text().splitlines()]
        for record in records:
            receipt = record["routing"]["scope_receipt"]
            h.require(receipt["phases"] == ["train", "eval"] and receipt["masked"]
                      and receipt["legal_rows"] == 784 and receipt["overlap_1based"] == [713, 727, 773],
                      "Missing a Fold-4 per-epoch mask receipt")
        h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == len(records),
                  "Independent EMA replay differs")
    else:
        # Read-only source ledger binds rollout, first strict maximum, and RNG
        # checkpoints. Compare each shared phase to the epoch-matched DARG trace.
        records = [json.loads(line) for line in files["journal"].read_text().splitlines()]
        h.require([r["epoch"] for r in records] == list(range(11, latest["epoch"] + 1)),
                  "Archived epoch range differs")
        first_max = max(records, key=lambda r: r["correct"])
        h.require(best["epoch"] == first_max["epoch"] and best["evaluation"]["correct"] == first_max["correct"],
                  "Archived Best is not the first strict maximum")
        trace = h.hist_trace(fold)
        for previous, current in zip(records, records[1:]):
            h.require(previous["rng"]["rng_after_eval"] == current["rng"]["rng_before"],
                      "Archived RNG chain differs")
        common = [r for r in records if r["epoch"] in trace]
        h.require(all(r["rng"] == trace[r["epoch"]] for r in common)
                  and all(r["rng"]["rng_after_eval"] == r["rng"]["rng_after_train"] for r in records),
                  "Archived RNG phases differ")
        journal = {"source_ledger_verified": True, "archived_epochs": len(common),
                   "rng_phase_comparisons": 3 * len(common), "ledger_epochs": len(records),
                   "earliest_best_epoch": best["epoch"]}

    run.restore(ctx, best, manifest["implementation"])
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best optimizer or model readback differs")
    full, cap = proof.independent_eval(ctx)
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    anonymous, anon_cap = proof.independent_eval(ctx, "anonymous_query")
    proof.match_eval(full, best["evaluation"], "Best independent")
    h.require(torch.equal(cap["cam"], off_cap["cam"]) and torch.equal(cap["cam"], anon_cap["cam"])
              and torch.equal(off_cap["cam"], off_cap["enhanced"]), "Fixed-weight intervention changed native CAM")
    spatial, cam_l2, ratio = masked_geometry(ctx, full, off, cap, expected)
    masked_geometry(ctx, anonymous, off, anon_cap, expected)
    rows = ctx["members"]["test"].copy()
    label = ctx["state"]["labels_np"][rows].copy()
    if kind == "local":
        geometry = {name: proof.f.branch_formulas(ctx, data, intervention=name) for name, data in
                    (("full", cap), ("residual_off", off_cap), ("anonymous_query", anon_cap))}
        membership = ind.membership_proof(ctx)
        gaussian = ind.gaussian_proof(ctx)
        scope = independent_scope(run, proof, ctx, pre["dropout"], expected)
        run.restore(ctx, best, manifest["implementation"])
        budget = proof.twin_probe(ctx, pre["dropout"], check_each=True)
    elif restore_next and fold in (2, 3):
        membership = ind.membership_proof(core.setup("G1", fold, verify_spectrum=True))
        geometric_context = ctx
        geometry = proof.f.branch_formulas(geometric_context, cap)
        gaussian = ind.gaussian_proof(geometric_context)
        scope = budget = None
    else:
        membership = gaussian = geometry = scope = budget = None

    query_shift = (cap["enhanced"][rows].double() - anon_cap["enhanced"][rows].double()).cpu().numpy()
    if restore_next:
        recovered = proof.check_latest(ctx, latest, manifest, task, pre)
        h.require(recovered["exact"], "Latest next update and evaluation differ")
        run.restore(ctx, latest, manifest["implementation"])
        run.restore_ema(ctx, latest["eval_ema"])
        end, end_cap = proof.independent_eval(ctx)
        end_off, _ = proof.independent_eval(ctx, "residual_off")
        proof.match_eval(end, latest["evaluation"], "Latest mask recomputation")
        masked_geometry(ctx, end, end_off, end_cap, expected)
    else:
        recovered = None
    darg = baseline_darg(ctx, manifests["base"]) if collect_darg else None

    data = {
        "fold": fold, "source_kind": kind, "subject_id": (rows + 1).tolist(), "label": label.tolist(),
        "full": array_records(full), "off": array_records(off), "anonymous": array_records(anonymous),
        "darg": darg, "masked_ids": [row + 1 for row in expected],
        "residual_l2": {key: item.tolist() for key, item in spatial.items()}, "cam_l2": cam_l2.tolist(),
        "relative_residual": ratio.tolist(), "query_feature_shift_l2": np.linalg.norm(query_shift, axis=1).tolist(),
        "best_epoch": best["epoch"], "latest_epoch": latest["epoch"],
        "best": str(files["best"]), "best_sha": task["best_sha"], "latest": str(files["latest"]),
        "latest_sha": task["latest_sha"], "journal": str(files["journal"]), "journal_sha": task["journal_sha"],
        "source_manifest": str(DIRECTORIES[kind] / "manifest.json"),
        "source_manifest_sha": h.sha_file(DIRECTORIES[kind] / "manifest.json") if kind != "local" else None,
        "source_implementation_sha": manifest["implementation_sha"], "best_state_sha": best["state_sha"],
        "latest_state_sha": latest["state_sha"], "preflight_sha": h.sha_file(files["preflight"]),
        "preflight_audit_amendment": amendment,
        "inference_sha": h.sha(full), "rng_ledger": journal, "latest_next": recovered,
        "membership_audit": membership, "gaussian_audit": gaussian, "geometry_audit": geometry,
        "scope_gradient": scope, "best_gradient_budget": budget,
    }
    print(f"{kind} Fold {fold:02} | Best {best['epoch']} | {full['correct']}/{len(rows)} | "
          f"Latest {'exact' if recovered is not None else 'historical reference'}", flush=True)
    return data


def main():
    manifests = load_manifests()
    device = h.archive.verify_device()
    h.require(device["index"] == 0 and device["uuid"] == core.CONFIG["physical_gpu_uuid"], "Physical GPU0 differs")
    assert core.implementation() == manifests["local"]["implementation"]
    original = verify_guard(manifests["local"])
    archive = guards(manifests)
    fold3 = h.read_json(ROOT / "runtime/fold03_reuse_equivalence.json")
    h.require(fold3["status"] == "PASS" and fold3["local_implementation_sha256"] ==
              manifests["local"]["implementation_sha"] and fold3["source_manifest_sha256"] ==
              h.sha_file(DIRECTORIES["e463"] / "manifest.json"), "Fold 3 reuse certificate is stale")

    current = {}
    for fold, kind in CURRENT.items():
        current[str(fold)] = review(kind, fold, manifests, restore_next=True, collect_darg=True)
        gc.collect()
        torch.cuda.empty_cache()
    reference = {}
    for fold, kind in ((3, "e464"), (4, "base")):
        reference[str(fold)] = review(kind, fold, manifests)
    h.require(guards(manifests) == archive and verify_guard(manifests["local"]) == original,
              "Protected history changed during independent evaluation")

    h.write_json(ROOT / "runtime/replayed_folds.json", {
        "status": "PASS", "current": current, "e464_changed_folds": reference, "fold3_equivalence": fold3,
        "train_guard": original, "archive_guard": archive, "device": device,
        "historical_git_sources": manifests["base"]["sources"],
    })


if __name__ == "__main__":
    main()
