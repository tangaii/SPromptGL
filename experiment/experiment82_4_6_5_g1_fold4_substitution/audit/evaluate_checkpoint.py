"""Read-only independent Best evaluation for one source fold."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core as local_core
import independent_465 as local_ind
import numpy as np
import torch

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("base", "e464", "e463", "local"), required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    f = args.fold
    base = Path("/model/spromptgl/experiment/experiment82_4_6_pcsr_scmr")
    e464 = Path("/model/spromptgl/experiment/experiment82_4_6_4_cumulative_overlap_paired")
    e463 = Path("/model/spromptgl/experiment/experiment82_4_6_3_darg_only_overlap")
    if args.kind == "base":
        source = base
        sys.path.insert(0, str(source))
        import run as source_run
        import audit as proof
        task = "G2"
        ctx = source_run.setup(task, f, verify_spectrum=False)
        manifest = local_core.h.read_json(source / "manifest.json")
        checkpoint_path = source / "checkpoints" / f"G2_f{f:02d}.best.pt"
    elif args.kind == "e464":
        source = e464
        adapter = load_file(f"e464_core_f{f}", source / "core.py")
        source_run = adapter.engine("G1", f)
        proof = load_file(f"e464_audit_f{f}", base / "audit.py")
        proof.run = source_run
        task = "G1"
        ctx = source_run.setup(task, f)
        manifest = local_core.h.read_json(source / "manifest.json")
        checkpoint_path = source / "checkpoints" / f"G1_f{f:02d}.best.pt"
    elif args.kind == "e463":
        source = e463
        adapter = load_file(f"e463_darg_f{f}", source / "darg_only.py")
        source_run = adapter.engine(f)
        proof = load_file(f"e463_audit_f{f}", base / "audit.py")
        proof.run = source_run
        task = "G2"
        ctx = source_run.setup(task, f)
        manifest = local_core.h.read_json(source / "manifest.json")
        checkpoint_path = source / "checkpoints" / f"G2_f{f:02d}.best.pt"
    else:
        source = ROOT
        source_run = local_ind.engine("G1", f)
        proof = load_file(f"local_audit_f{f}", base / "audit.py")
        proof.run = source_run
        task = "G1"
        ctx = source_run.setup(task, f)
        manifest = local_core.h.read_json(ROOT / "manifest.json")
        checkpoint_path = ROOT / "checkpoints" / f"G1_f{f:02d}.best.pt"

    best = local_core.h.load(checkpoint_path)
    run_manifest = manifest
    if args.kind == "base":
        implementation = manifest["implementation"]
    elif args.kind == "e464":
        implementation = manifest["implementation"]
    elif args.kind == "e463":
        implementation = manifest["implementation"]
    else:
        implementation = manifest["implementation"]
    source_run.restore(ctx, best, implementation)
    rng_before = local_core.h.rng_sha()
    full, cap = proof.independent_eval(ctx, "full")
    off, _ = proof.independent_eval(ctx, "residual_off")
    anonymous, _ = proof.independent_eval(ctx, "anonymous_query")
    local_core.h.require(local_core.h.rng_sha() == rng_before, "Independent interventions changed RNG")
    rows = ctx["members"]["test"]
    y = ctx["state"]["labels_np"][rows]
    components = cap["details"]
    tensors = {k: components[k][rows].detach().cpu().numpy().astype(np.float64)
               for k in ("delta", "delta_parallel", "delta_perp", "ratio")}
    cam = cap["cam"][rows].detach().cpu().numpy().astype(np.float64)
    logits = cap["logits"][rows].detach().cpu().numpy().astype(np.float64)
    result = {
        "fold": f, "kind": args.kind, "source": str(source), "source_manifest_sha": local_core.h.sha_file(source / "manifest.json"),
        "source_implementation_sha": manifest.get("implementation_sha"),
        "checkpoint": str(checkpoint_path), "checkpoint_sha": local_core.h.sha_file(checkpoint_path),
        "epoch": int(best["epoch"]), "test_indices_0based": rows.tolist(), "labels": y.tolist(),
        "full": {k: np.asarray(v).tolist() for k, v in full.items()},
        "off": {k: np.asarray(v).tolist() for k, v in off.items()},
        "anonymous": {k: np.asarray(v).tolist() for k, v in anonymous.items()},
        "margin_full": ((logits[:, 0] - logits[:, 1]) * np.where(y == 0, 1.0, -1.0)).tolist(),
        "residual_l2": {k: np.linalg.norm(tensors[k], axis=1).tolist() for k in ("delta", "delta_parallel", "delta_perp")},
        "cam_l2": np.linalg.norm(cam, axis=1).tolist(), "residual_ratio": tensors["ratio"].reshape(-1).tolist(),
        "rng_unchanged": True,
    }
    dest = ROOT / "runtime" / f"independent_fold_{f:02d}.json"
    dest.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"fold": f, "kind": args.kind, "best_epoch": result["epoch"],
                      "checkpoint_sha": result["checkpoint_sha"], "correct": full["correct"],
                      "off_correct": off["correct"], "query_hard_flips": int(np.count_nonzero(full["prediction"] != anonymous["prediction"])),
                      "max_residual_ratio": float(np.max(result["residual_ratio"]))}, indent=2))

if __name__ == "__main__":
    main()
