from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "runtime/python"), str(ROOT / "code/history_base")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch

import jsl_runtime as r
import rng_forensics
from author_log import check_template
from common import history as h
from common.spectral import split_jacobian

DPSL = Path("/model/spromptgl/experiment/DPSL")
DPSL_FULL_RETRAIN = Path("/model/spromptgl/experiment/DPSL_full_retrain_from_warmup")
FROZEN_MANIFEST_SHA = "af1967ff3ccd32124acf35ac99fcba139d274e49d51a945af7d20fe6b7d037b7"
OVERLAPS = {
    2: [377, 407, 681, 779, 784, 260, 288, 375],
    3: [46, 66, 172, 266, 388],
    4: [712, 713, 727, 773],
    8: [383, 430],
}
EXPECTED_COUNTS = [
    [783, 783, 88], [792, 784, 87], [789, 784, 87], [788, 784, 87],
    [784, 784, 87], [784, 784, 87], [784, 784, 87], [786, 784, 87],
    [784, 784, 87], [784, 784, 87],
]


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_dpsl():
    manifest = DPSL / "SHA256SUMS"
    if file_sha(manifest) != FROZEN_MANIFEST_SHA:
        raise RuntimeError("formal DPSL SHA256SUMS digest changed")
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = DPSL / relative
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError(f"formal DPSL frozen file mismatch: {relative}")
        entries.append(relative)
    if len(entries) != 3949 or len(set(entries)) != 3949:
        raise RuntimeError(f"formal DPSL frozen inventory has {len(entries)} entries, expected 3949")
    return {"manifest_sha256": file_sha(manifest), "file_count": len(entries),
            "file_set_sha256": hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()}


def official_dpsl_fold1_inspect():
    result = subprocess.run(
        [sys.executable, "-I", "-B", "cli.py", "inspect", "--fold", "1"],
        cwd=DPSL, text=True, capture_output=True, check=True, timeout=900,
    )
    receipt = json.loads(result.stdout)
    if receipt.get("status") != "COMPLETE_AUDITED_STANDALONE":
        raise RuntimeError("formal DPSL's independent Fold 1 loader inspection did not pass")
    return {"status": receipt["status"], "authority": receipt.get("authority"),
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()}


def read_full_retrain_reference():
    manifest = json.loads((DPSL_FULL_RETRAIN / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_dir = DPSL_FULL_RETRAIN / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("fold*.pt"))
    if len(checkpoints) != 20:
        raise RuntimeError("DPSL full-retrain reference does not contain ten Best and ten Latest files")
    return {"manifest_status": manifest.get("status"), "completed_folds": manifest.get("completed_folds"),
            "checkpoint_count": len(checkpoints),
            "checkpoint_set_sha256": hashlib.sha256("\n".join(
                f"{path.name}:{file_sha(path)}" for path in checkpoints).encode()).hexdigest()}


def validate_basis(basis):
    if tuple(basis.shape) != (144, 16):
        raise RuntimeError(f"Original O basis shape is {tuple(basis.shape)}, expected (144, 16)")
    gram = basis.T @ basis
    if float((gram - torch.eye(16, dtype=gram.dtype, device=gram.device)).abs().max()) >= 1e-10:
        raise RuntimeError("Original O basis is not orthonormal")


def assert_zero_overlap(residual, overlap):
    if overlap and torch.count_nonzero(residual[overlap]).item():
        raise RuntimeError("overlap patient has a nonzero JSL residual")


def assert_membership(actual, expected):
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise RuntimeError("training member order differs from the frozen formal order")


def assert_same_rng(actual, expected):
    if actual != expected:
        raise RuntimeError("RNG component identity differs")


def assert_no_dse_text(source):
    lowered = source.lower()
    forbidden = ("medical prompt", "anonymous prompt", "clip text query", "gaussian llr",
                 "evidence_direction", "evidence_reliability", "ema class statistics",
                 "dse reliability", "dse conflict", "dse direction")
    if any(token in lowered for token in forbidden):
        raise RuntimeError("DSE dependency found in a JSL implementation source")
    if "medical query" in lowered or "prompt query" in lowered:
        raise RuntimeError("text Query dependency found in JSL source")


def expect_failure(name, function):
    try:
        function()
    except (RuntimeError, AssertionError, ValueError):
        return {"case": name, "detected": True}
    raise RuntimeError(f"negative test was not detected: {name}")


def run_negative_tests(ctx):
    jsl_source = (ROOT / "code/jsl_model.py").read_text(encoding="utf-8")
    assert_no_dse_text(jsl_source)
    overlap = ctx["state"]["excluded"] or [0]
    zero = torch.zeros_like(ctx["state"]["x"][:, :144])
    cases = [
        expect_failure("medical Query injection", lambda: assert_no_dse_text("medical Query")),
        expect_failure("Gaussian evidence injection", lambda: assert_no_dse_text("Gaussian LLR")),
        expect_failure("EMA statistics injection", lambda: assert_no_dse_text("EMA class statistics")),
        expect_failure("nonzero overlap residual", lambda: assert_zero_overlap(torch.ones_like(zero), overlap)),
        expect_failure("changed training member order", lambda: assert_membership(
            ctx["members"]["train"][::-1], ctx["members"]["train"])),
        expect_failure("corrupted RNG state", lambda: assert_same_rng(
            {"cpu": "corrupt"}, rng_forensics.rng_components())),
        expect_failure("changed spectral dimension", lambda: validate_basis(
            ctx["archived"]["spectrum"]["U"][:, :15])),
        expect_failure("nonorthogonal spectral basis", lambda: validate_basis(
            ctx["archived"]["spectrum"]["U"] * torch.tensor(
                [2.0] + [1.0] * 15, dtype=torch.float64, device=ctx["state"]["x"].device))),
    ]
    return cases


def darg_baseline_step(ctx):
    state = ctx["state"]
    for key in h.OPTS:
        state[key].zero_grad(set_to_none=True)
    capture = {}

    def record_cam(_module, args):
        capture["cam"] = args[0]

    hook = state["cam"].Outputlayer.register_forward_pre_hook(record_cam)
    try:
        with r.dropout_inventory(state) as dropout:
            output = h.runner().forward(state, h.BASE, training=True)
            loss = output["total"]
            params = r.shared_params(state)
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
            for parameter, gradient in zip(params, gradients):
                parameter.grad = gradient
            for key in h.OPTS:
                state[key].step()
    finally:
        hook.remove()
    rng_after_train = h.rng_sha()
    with torch.no_grad():
        h.runner().forward(state, h.BASE, training=False)
    return {"loss": float(loss.detach()), "gradient_sha": r.sha(gradients),
            "logp_sha": r.sha(output["logp"]), "cam_sha": r.sha(capture["cam"]),
            "dropout": dropout, "rng_after_train": rng_after_train,
            "rng_after_eval": h.rng_sha()}


def check_jacobian_geometry(ctx, capture):
    spectrum = ctx["archived"]["spectrum"]
    validate_basis(spectrum["U"])
    jacobian = capture["details"]["jacobian"]
    _, parallel, complement = split_jacobian(jacobian, spectrum["U"])
    dot = (parallel * complement).sum(-1).abs().max()
    reconstruction = (parallel + complement - jacobian).abs().max()
    scale = jacobian.norm(dim=-1).max().clamp_min(1.0)
    if float(dot / scale.square()) > 1e-5:
        raise RuntimeError("Jacobian spectral and orthogonal-complement directions are not orthogonal")
    if float(reconstruction / scale) > 1e-6:
        raise RuntimeError("Jacobian spectral split does not reconstruct the original Jacobian")
    return {"basis_shape": list(spectrum["U"].shape), "parallel_dim": 16,
            "complement_dim": 128, "orthogonality_relative": float(dot / scale.square()),
            "reconstruction_relative": float(reconstruction / scale)}


def check_residual_cap(ctx, cam):
    module = ctx["module"]
    last = module.controller[-1]
    saved_weight = last.weight.detach().clone()
    saved_bias = last.bias.detach().clone()
    try:
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(torch.tensor([20.0, 20.0, 0.0], device=last.bias.device))
        _, details = module(cam, ctx["state"]["cam"].Outputlayer)
        ratio = details["delta"].norm(dim=-1) / cam.norm(dim=-1).clamp_min(1e-8)
        maximum = float(ratio.max())
        if not torch.isfinite(ratio).all() or maximum > 0.08 + 1e-6:
            raise RuntimeError(f"JSL residual cap exceeded 8 percent: {maximum}")
        assert_zero_overlap(details["delta"], ctx["state"]["excluded"])
        return {"max_fraction": maximum, "overlap_zero": True}
    finally:
        with torch.no_grad():
            last.weight.copy_(saved_weight)
            last.bias.copy_(saved_bias)


def compare_epoch_trace(fold, epoch, before, after_train, after_eval):
    trace = h.hist_trace(fold)
    archived = trace.get(epoch)
    if archived is None:
        raise RuntimeError(f"fold {fold} archived RNG witness is missing epoch {epoch}")
    actual = {"rng_before": before, "rng_after_train": after_train, "rng_after_eval": after_eval}
    if actual != {key: archived[key] for key in actual}:
        raise RuntimeError(f"fold {fold} epoch {epoch} RNG replay differs from archived DPSL")
    return actual


def fold1_gradient_and_recovery(ctx, stages):
    state = ctx["state"]
    initial_native = h.base_bundle(state)
    initial_module = copy.deepcopy(ctx["module"].state_dict())
    initial_optimizer = copy.deepcopy(ctx["optimizer"].state_dict())
    initial_rng = copy.deepcopy(h.base().pack_rng())
    initial_rng_sha = h.rng_sha()
    if initial_rng_sha != ctx["warmup"]["rng_sha"]:
        raise RuntimeError("Fold 1 did not enter the first probe at the E61 epoch-10 RNG")

    full, full_capture = r.evaluate(ctx, "full")
    residual_off, residual_off_capture = r.evaluate(ctx, "residual_off")
    if not np.array_equal(full["probability"], residual_off["probability"]):
        raise RuntimeError("zero-action JSL changed the historical initial forward")
    if torch.count_nonzero(full_capture["details"]["delta"]).item():
        raise RuntimeError("JSL initial controller is not an exact zero-action state")
    assert_zero_overlap(full_capture["details"]["delta_parallel"], state["excluded"])
    assert_zero_overlap(full_capture["details"]["delta_perp"], state["excluded"])
    assert_zero_overlap(full_capture["details"]["delta"], state["excluded"])
    geometry = check_jacobian_geometry(ctx, full_capture)
    cap = check_residual_cap(ctx, full_capture["cam"])
    assert_no_dse_text((ROOT / "code/jsl_model.py").read_text(encoding="utf-8"))
    active_network = sys.modules.get("network")
    if active_network is None or Path(active_network.__file__).resolve() != (ROOT / "code/darg_core/network.py").resolve():
        raise RuntimeError("active DARG runtime imported a Prompt/CLIP network module")
    if any(name == "clip" or name.startswith("clip.") or name == "CLIP" or name.startswith("CLIP.")
           for name in sys.modules):
        raise RuntimeError("CLIP module is active in the DARG + JSL process")
    module_state_keys = [name.lower() for name in ctx["module"].state_dict()]
    if any(any(token in name for token in ("prompt", "query", "clip", "gaussian", "ema", "evidence"))
           for name in module_state_keys):
        raise RuntimeError("JSL state contains a DSE parameter or buffer")

    baseline = darg_baseline_step(ctx)
    h.restore_base(state, initial_native)
    ctx["module"].load_state_dict(initial_module, strict=True)
    ctx["optimizer"].load_state_dict(initial_optimizer)
    h.base().restore_rng(initial_rng)
    if h.rng_sha() != initial_rng_sha:
        raise RuntimeError("reset to the isolated Fold 1 warmup state failed")

    first_before = h.rng_sha()
    first = r.train_step(ctx)
    first_after_train = h.rng_sha()
    first_eval, _ = r.evaluate(ctx)
    first_after_eval = h.rng_sha()
    trace11 = compare_epoch_trace(1, 11, first_before, first_after_train, first_after_eval)
    if first_after_train != baseline["rng_after_train"] or first_after_eval != baseline["rng_after_eval"]:
        raise RuntimeError("JSL forward or losses consumed a different historical RNG stream")
    if first["base_gradient_sha"] != baseline["gradient_sha"]:
        raise RuntimeError("JSL first step replaced or altered the historical DARG gradient")
    if not np.isclose(first["darg_loss"], baseline["loss"], rtol=0.0, atol=1e-8):
        raise RuntimeError("zero-action JSL changed the original DARG loss on the first step")
    if first["dropout"] != baseline["dropout"]:
        raise RuntimeError("JSL changed the order, masks, or inputs of the six historical Dropout calls")
    if first["native_logp_sha"] != baseline["logp_sha"] or first["native_cam_sha"] != baseline["cam_sha"]:
        raise RuntimeError("zero-action JSL changed the historical native forward")
    if first["shared_extra_fraction"] > r.CAP + 1e-8:
        raise RuntimeError("JSL shared-gradient addition exceeded the 0.1 percent budget")
    if first["parallel_residual_gradient"] <= 0 or first["complement_residual_gradient"] <= 0:
        raise RuntimeError("one of the two spectral residual branches has no classification gradient")
    if first["native_cls"] != first["native_cls"] or not np.isfinite(first["jsl_loss"]):
        raise RuntimeError("first-step DARG or JSL loss is not finite")

    second_before = h.rng_sha()
    second = r.train_step(ctx)
    second_after_train = h.rng_sha()
    _, _ = r.evaluate(ctx)
    second_after_eval = h.rng_sha()
    trace12 = compare_epoch_trace(1, 12, second_before, second_after_train, second_after_eval)
    for group in ("spectral_encoder", "spectral_weight", "controller"):
        gradients = second["module_gradients"][group]
        if gradients["with_grad"] != gradients["parameters"] or gradients["nonzero"] == 0:
            raise RuntimeError(f"JSL {group} does not have complete nonzero end-to-end gradients after adaptation")
    if second["shared_extra_fraction"] > r.CAP + 1e-8:
        raise RuntimeError("JSL second-step shared-gradient addition exceeded the 0.1 percent budget")

    test_labels = state["labels_np"][ctx["members"]["test"]]
    latest_eval, _ = r.evaluate(ctx)
    score = r.metrics(test_labels, latest_eval["prediction"], latest_eval["probability"])
    record = {"epoch": 12, "author_log": {"epoch": 12}, "rng": trace12}
    scratch = ROOT / "runtime/preflight_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    best_path = scratch / "fold01.best.pt"
    latest_path = scratch / "fold01.latest.pt"
    best_payload = r.save_checkpoint(best_path, ctx, 12, 11, score["correct"], 1, latest_eval, record=record)
    latest_payload = r.save_checkpoint(latest_path, ctx, 12, 11, score["correct"], 1, latest_eval, record=record)
    if best_payload["state_sha"] != latest_payload["state_sha"]:
        raise RuntimeError("Best and Latest scratch snapshots do not contain the same preflight state")

    restored_best = r.setup(1)
    r.restore(restored_best, torch.load(best_path, map_location=state["x"].device, weights_only=False))
    if r.sha(r.bundle(restored_best)) != r.sha(best_payload["state"]):
        raise RuntimeError("Best scratch snapshot did not fully restore")

    replay = []
    for _ in range(2):
        recovered = r.setup(1)
        payload = torch.load(latest_path, map_location=state["x"].device, weights_only=False)
        r.restore(recovered, payload)
        diag = r.train_step(recovered)
        evaluation, _ = r.evaluate(recovered)
        replay.append({"diag": diag, "state_sha": r.sha(r.bundle(recovered)),
                       "evaluation_sha": r.sha(evaluation), "rng_sha": h.rng_sha()})
        del recovered
        torch.cuda.empty_cache()
    if replay[0] != replay[1]:
        raise RuntimeError("Latest next-step training and evaluation replay was not exact")
    if len(h.hist_trace(1)) < 2:
        raise RuntimeError("formal DPSL does not provide comparable epoch 11/12 RNG witnesses")

    h.restore_base(state, latest_payload["state"]["native"])
    ctx["module"].load_state_dict(latest_payload["state"]["module"], strict=True)
    ctx["optimizer"].load_state_dict(latest_payload["state"]["optimizer"])
    h.base().restore_rng(latest_payload["state"]["rng"])
    s8 = rng_forensics.add_s8(stages, ctx, {"historical_dropout_calls": 6,
                                            "epoch11_rng_equal": True, "epoch12_rng_equal": True,
                                            "best_restore": True, "latest_next_step_replay": True})
    shutil.rmtree(scratch)
    return {"forward_zero_action": True, "geometry": geometry, "residual_cap": cap,
            "baseline_vs_jsl_first_step": {"native_loss_equal": True, "base_gradient_equal": True,
                                            "dropout_masks_and_order_equal": True,
                                            "shared_extra_fraction": first["shared_extra_fraction"],
                                            "parallel_gradient": first["parallel_residual_gradient"],
                                            "complement_gradient": first["complement_residual_gradient"]},
            "module_gradients_after_second_step": second["module_gradients"],
            "rng_epoch11": trace11, "rng_epoch12": trace12,
            "best_restore": True, "latest_next_step_replay": True,
            "recovery_sha": replay[0]["state_sha"], "s8": s8}


def update_manifest(status, preflight, blocker=None):
    path = ROOT / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["preflight_status"] = status
    manifest["training_started"] = False
    manifest["blocker"] = blocker
    manifest["preflight_sha256"] = hashlib.sha256(
        json.dumps(preflight, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    if status == "PASS":
        manifest["formal_source_config_sha256"] = preflight["implementation_sha256"]
        manifest["rng_forensic_conclusion"] = preflight["rng_forensic_conclusion"]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run():
    checks = {}
    fold_records = []
    stages_by_fold = {}
    status = "BLOCKED_PREFLIGHT"
    blocker = None
    start_files = sorted((ROOT / "checkpoints").glob("fold*.pt")) if (ROOT / "checkpoints").exists() else []
    try:
        frozen_before = verify_frozen_dpsl()
        checks["frozen_dpsl_before"] = frozen_before
        checks["official_e61_loader"] = official_dpsl_fold1_inspect()
        checks["full_retrain_reference"] = read_full_retrain_reference()
        checks["author_log_template"] = check_template()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("formal preflight requires only physical GPU0")
        expected_uuid = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["gpu"]["uuid"]
        gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                   text=True, capture_output=True, check=True)
        if gpu_query.stdout.strip().splitlines() != [expected_uuid]:
            raise RuntimeError("physical GPU0 UUID differs from the frozen experiment configuration")
        checks["device"] = {"index": 0, "uuid": expected_uuid,
                            "name": torch.cuda.get_device_name(0), "visible_device_count": 1}

        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        first_ctx, first_stages = rng_forensics.capture_fold(1)
        stages_by_fold[1] = first_stages
        if [row + 1 for row in first_ctx["state"]["excluded"]] != []:
            raise RuntimeError("Fold 1 unexpectedly contains training/test overlap")
        first_pairs = first_ctx["state"]["train_pairs_np"]
        first_pair_labels = first_ctx["state"]["train_same_np"].astype(np.int64)
        if first_pairs.shape != (4096, 2) or np.bincount(first_pair_labels, minlength=2).tolist() != [2048, 2048]:
            raise RuntimeError("Fold 1 relation pair count/balance differs")
        for row in first_stages:
            if row["stage"] == "S4":
                members = first_ctx["members"]
                if [len(members[key]) for key in ("train", "legal", "test")] != EXPECTED_COUNTS[0]:
                    raise RuntimeError("Fold 1 training/JSL/test counts differ")
        fold_records.append({"fold": 1, "train": len(first_ctx["members"]["train"]),
                             "jsl_legal": len(first_ctx["members"]["legal"]),
                             "test": len(first_ctx["members"]["test"]),
                             "overlap_1based": [row + 1 for row in first_ctx["state"]["excluded"]],
                             "warmup_rng_sha": first_ctx["warmup"]["rng_sha"],
                             "model_identity": first_ctx["identity"]["model_cam_sha"],
                             "optimizer_identity": first_ctx["identity"]["optimizer_sha"],
                             "pair_sha": first_ctx["state"]["pair_sha"]})

        first_probe = fold1_gradient_and_recovery(first_ctx, first_stages)
        peak_bytes = torch.cuda.max_memory_allocated()
        throughput = 2.0 / max(time.monotonic() - started, 1e-9)
        checks["fold1_rng_and_first_step"] = first_probe
        checks["single_fold_resource_probe"] = {"peak_allocated_mib": peak_bytes / (1024 ** 2),
                                                 "two_step_throughput_steps_per_second": throughput,
                                                 "worker_count": 1}
        checks["negative_tests"] = run_negative_tests(first_ctx)
        if (ROOT / "checkpoints").exists() and list((ROOT / "checkpoints").glob("fold*.pt")):
            raise RuntimeError("preflight wrote into the formal checkpoint directory")
        h.write_json(ROOT / "runtime/rng_forensics/fold01.json",
                     {"fold": 1, "stages": first_stages,
                      "rng_component_identity": "single E61 loader extraction and DPSL runtime serializer"})
        del first_ctx
        torch.cuda.empty_cache()

        for fold in range(2, 11):
            ctx, stages = rng_forensics.capture_fold(fold)
            stages_by_fold[fold] = stages
            members = ctx["members"]
            expected = EXPECTED_COUNTS[fold - 1]
            actual = [len(members[key]) for key in ("train", "legal", "test")]
            if actual != expected:
                raise RuntimeError(f"fold {fold} membership counts {actual} != {expected}")
            overlap = [row + 1 for row in ctx["state"]["excluded"]]
            if sorted(overlap) != sorted(OVERLAPS.get(fold, [])):
                raise RuntimeError(f"fold {fold} overlap membership {overlap} differs from formal DPSL")
            if fold in r.OVERLAP_APPEND_1BASED:
                if ctx["state"].get("added_train_rows_1based") != list(r.OVERLAP_APPEND_1BASED[fold]):
                    raise RuntimeError(f"fold {fold} training append order differs from formal DPSL")
            pairs = ctx["state"]["train_pairs_np"]
            same = ctx["state"]["train_same_np"].astype(np.int64)
            if pairs.shape != (4096, 2) or np.bincount(same, minlength=2).tolist() != [2048, 2048]:
                raise RuntimeError(f"fold {fold} relation pair count/balance differs")
            _, capture = r.evaluate(ctx)
            details = capture["details"]
            for key in ("delta_parallel", "delta_perp", "delta"):
                assert_zero_overlap(details[key], ctx["state"]["excluded"])
            fold_records.append({"fold": fold, "train": actual[0], "jsl_legal": actual[1], "test": actual[2],
                                 "overlap_1based": overlap, "warmup_rng_sha": ctx["warmup"]["rng_sha"],
                                 "model_identity": ctx["identity"]["model_cam_sha"],
                                 "optimizer_identity": ctx["identity"]["optimizer_sha"],
                                 "pair_sha": ctx["state"]["pair_sha"]})
            h.write_json(ROOT / "runtime/rng_forensics" / f"fold{fold:02d}.json",
                         {"fold": fold, "stages": stages,
                          "rng_component_identity": "single E61 loader extraction and DPSL runtime serializer"})
            del ctx
            torch.cuda.empty_cache()
        fold_records.sort(key=lambda item: item["fold"])
        checks["ten_fold_membership_pairs_warmup"] = fold_records
        if len(fold_records) != 10 or [row["fold"] for row in fold_records] != list(range(1, 11)):
            raise RuntimeError("ten-fold preflight did not produce all folds")

        forbidden_dir = ROOT / "code/vendor/history"
        if forbidden_dir.exists():
            raise RuntimeError("stale copied DSE history source tree remains inside the JSL experiment")
        if (ROOT / "runtime/history").exists():
            raise RuntimeError("stale copied DSE run journals remain inside the JSL experiment")
        checks["independence"] = {"status": "PASS", "active_network": "DARG-only GraphLearn/GCN",
                                  "clip_loaded": False, "dse_state": False,
                                  "stale_dse_history_removed": True}
        checks["author_log_format"] = "PASS"
        checks["preflight_negative_detection"] = "PASS"

        frozen_after = verify_frozen_dpsl()
        if frozen_after != frozen_before:
            raise RuntimeError("formal DPSL frozen inventory changed during preflight")
        checks["frozen_dpsl_after"] = frozen_after
        if start_files != (sorted((ROOT / "checkpoints").glob("fold*.pt")) if (ROOT / "checkpoints").exists() else []):
            raise RuntimeError("formal checkpoint directory changed during preflight")

        implementation = r.implementation()
        implementation_sha = r.sha(implementation)
        result = {"status": "PASS", "implementation": implementation,
                  "implementation_sha256": implementation_sha, "checks": checks,
                  "folds": fold_records,
                  "rng_forensic_conclusion": {
                      "old_actual_prefix": "c8ed67", "old_archived_prefix": "8b1101",
                      "same_stage_comparison": False,
                      "cause": "different stage extraction and serialization scope",
                      "official_warmup_identity_sha": first_stages[3]["rng"]["combined"],
                      "old_expected_sha_was_modified": False,
                      "first_divergence": "S0-to-S1 unchanged; S2 model/optimizer load complete; S3 checkpoint RNG restore exact",
                  },
                  "rng_stages": {str(fold): stages for fold, stages in stages_by_fold.items()}}
        status = "PASS"
        blocker = None
    except Exception as exc:
        blocker = f"{type(exc).__name__}: {exc}"
        result = {"status": status, "implementation": r.implementation(),
                  "implementation_sha256": r.sha(r.implementation()), "checks": checks,
                  "folds": fold_records, "rng_stages": {str(fold): stages for fold, stages in stages_by_fold.items()},
                  "blocker": blocker}
    result_path = ROOT / "runtime/preflight.json"
    temporary = result_path.with_name(result_path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    update_manifest(status, result, blocker)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, default=str))
