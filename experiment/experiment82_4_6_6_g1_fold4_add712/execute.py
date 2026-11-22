"""Execute the one new Fold-4 formal task and audit historical reuse."""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, sys
from pathlib import Path
from core import CONFIG, ROOT, RUNTIME, RESULTS, MANIFEST, TASKS, FOLDS, GROUP, h, implementation, paths, source_manifests, engine

E464 = Path(CONFIG["source_e8246_directory"])
E463 = Path(CONFIG["source_e82463_directory"])
E465 = Path(CONFIG["source_e82465_directory"])

def protected_sources():
    files = []
    for root, names in ((E464, ["config.json", "core.py", "execute.py", "manifest.json"]),
                        (E463, ["config.json", "darg_only.py", "execute.py", "manifest.json"]),
                        (E465, ["config.json", "core.py", "execute.py", "independent_465.py", "manifest.json",
                                "results/report.md", "results/fold_metrics.csv", "results/paired_comparisons.csv",
                                "results/patient_level.csv", "results/membership_audit.json", "results/audit_status.json"]),
                        (Path(CONFIG["full_path_directory"]), ["config.json", "extension.py", "manifest.json"]),
                        (Path("/model/spromptgl/experiment/experiment82_4_6_pcsr_scmr"), ["run.py", "g2.py", "common/evidence.py", "common/spectral.py", "common/optimization.py", "common/history.py"]),):
        for name in names:
            p = root / name
            if p.exists(): files.append(str(p))
    return {p: h.sha_file(Path(p)) for p in files}

def verify_guard(manifest):
    expected = manifest["protected_sha256"]
    actual = h.archive.hash_files([Path(x) for x in expected])
    h.require(actual == expected, "Protected historical file changed")
    return {"status": "PASS", "files": len(actual), "bundle_sha": h.sha(actual)}

def freeze_manifest():
    device = h.archive.verify_device()
    h.require((device["index"], device["uuid"]) == (CONFIG["physical_gpu_index"], CONFIG["physical_gpu_uuid"]), "GPU0 identity differs")
    sources = source_manifests()
    m = {"experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
         "config": CONFIG, "implementation": implementation(), "implementation_sha": h.sha(implementation()),
         "protected_sha256": protected_sources(), "device": device, "parent_training_config": h.hist_config(),
         "source_status": {k: v.get("status") for k, v in sources.items()},
         "expected_tasks": [f"{GROUP}_f{f:02d}" for f in FOLDS], "reuse": {}, "guard_before": None}
    m["guard_before"] = verify_guard(m)
    return m

def preflight(manifest):
    from independent_465 import preflight_task
    # Fold 2 and Fold 3 are immutable historical reuse. Only Fold 4 receives
    # the formal preflight and training path in this directory.
    hashes = manifest["implementation"]
    result = preflight_task(GROUP, 4, hashes)
    h.write_json(paths(GROUP, 4)["preflight"], result)
    h.require(result["status"] == "PASS" and result["implementation"] == hashes, "Fold-4 preflight failed")
    manifest.update(status="PREFLIGHT_PASSED", preflight={"status": "PASS", "task": "G1_f04",
                    "sha": h.sha_file(paths(GROUP, 4)["preflight"])}, frozen_before_training=True)
    h.write_json(MANIFEST, manifest)
    print("PREFLIGHT G1 Fold 4 PASS", flush=True)

def train(manifest):
    files = paths(GROUP, 4)
    if files["task"].exists():
        t = h.read_json(files["task"])
        h.require(t["status"] == "TRAINED" and t["implementation"] == manifest["implementation"], "Existing Fold-4 task damaged")
        print("Verified existing Fold-4 formal task", flush=True); return
    manifest["status"] = "TRAINING"; h.write_json(MANIFEST, manifest)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    proc = subprocess.run(["taskset", "-c", str(CONFIG["worker_cpu_candidates"][0]), sys.executable, "-u", str(ROOT / "execute.py"), "--worker", "G1", "--fold", "4"], cwd=ROOT, env=env)
    h.require(proc.returncode == 0, "Fold-4 worker failed")
    h.require(files["task"].exists(), "Fold-4 task artifact missing")
    t = h.read_json(files["task"])
    h.require(t["status"] == "TRAINED" and t["implementation"] == manifest["implementation"]
              and all(h.sha_file(files[k]) == t[k + "_sha"] for k in ("best", "latest", "journal")), "Fold-4 artifact integrity failed")
    manifest["status"] = "TRAINED"; manifest["completed_task"] = t; h.write_json(MANIFEST, manifest)

def reuse_audit(manifest):
    # Verify source manifests, membership certificates, checkpoints and immutable
    # patient-level result rows. The historical files themselves are never copied.
    e4m = h.read_json(E464 / "manifest.json"); e3m = h.read_json(E463 / "manifest.json")
    e5m = h.read_json(E465 / "manifest.json")
    expected = CONFIG["expected_manifest_sha256"]
    h.require(e4m["status"] == "COMPLETE_AUDITED" and e3m["status"] == "COMPLETE_AUDITED"
              and e5m["status"] == "COMPLETE_AUDITED", "Reuse source incomplete")
    h.require(h.sha_file(E464 / "manifest.json") == expected["e82464_cumulative"]
              and h.sha_file(E463 / "manifest.json") == expected["e82463_darg_only"]
              and h.sha_file(E465 / "manifest.json") == expected["e82465_g1"],
              "Reuse manifest SHA differs from the frozen expectation")
    old_results = E465 / "results"
    for name in ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv",
                 "membership_audit.json", "audit_status.json"):
        h.require((old_results / name).exists(), f"E82.4.6.5 result artifact missing: {name}")
    rows = {}
    for fold, source, task in ((2, E464, "G1_f02"), (3, E463, "G2_f03")):
        taskmeta = h.read_json(source / "runtime" / f"{task}.training.json") if (source / "runtime" / f"{task}.training.json").exists() else None
        if taskmeta:
            for key in ("best", "latest", "journal"):
                p = source / ("checkpoints" if key != "journal" else "runtime") / (f"{task}.{key}.pt" if key != "journal" else f"{task}.jsonl")
                h.require(p.exists() and h.sha_file(p) == taskmeta[key + "_sha"], f"Reuse {task} {key} damaged")
        audit_path = source / "results" / "audit_status.json"
        audit = h.read_json(audit_path); h.require(audit.get("status") == "COMPLETE_AUDITED", f"Reuse {task} audit incomplete")
        rows[str(fold)] = {"source_directory": str(E465), "underlying_source_directory": str(source),
                           "source_manifest_sha": h.sha_file(E465 / "manifest.json"),
                           "underlying_source_manifest_sha": h.sha_file(source / "manifest.json"),
                           "source_audit_sha": h.sha_file(audit_path), "source_task": task,
                           "best_sha": (h.sha_file(source / "checkpoints" / f"{task}.best.pt") if (source / "checkpoints" / f"{task}.best.pt").exists() else None),
                           "latest_sha": (h.sha_file(source / "checkpoints" / f"{task}.latest.pt") if (source / "checkpoints" / f"{task}.latest.pt").exists() else None)}
    manifest["reuse"] = rows
    manifest["reuse"]["all_non_fold4"] = {
        "source_directory": str(E465), "source_manifest_sha": h.sha_file(E465 / "manifest.json"),
        "result_files": {name: h.sha_file(old_results / name) for name in
                         ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv",
                          "membership_audit.json", "audit_status.json")},
        "patient_rows": 871, "checkpoint_identity": "E82.4.6.5 manifest checkpoint_sources"
    }
    return rows

def audit(manifest):
    from independent_465 import audit_task
    item, cert = audit_task(GROUP, 4, manifest)
    manifest["audit"] = {"G1_f04": cert}; manifest["status"] = "AUDIT_PASSED"
    h.write_json(ROOT / "results" / "audit_status.json", {"status": "PASS", "tasks": {"G1_f04": cert}, "reused": manifest["reuse"]})
    h.write_json(MANIFEST, manifest)
    return item, cert

def main():
    p = argparse.ArgumentParser(); p.add_argument("--worker", choices=["G1"]); p.add_argument("--fold", type=int, choices=[4]); p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    for d in (RUNTIME, RESULTS, ROOT / "checkpoints"): d.mkdir(exist_ok=True)
    if a.worker:
        h.require(a.fold == 4 and len(os.sched_getaffinity(0)) == 1, "Worker affinity invalid")
        h.archive.verify_device(); engine("G1", 4).train_fold("G2", 4); return
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = h.read_json(MANIFEST) if MANIFEST.exists() else freeze_manifest()
        h.require(manifest["implementation"] == implementation() and manifest["config"] == CONFIG, "Frozen source/config differs")
        if manifest["status"] == "COMPLETE_AUDITED":
            verify_guard(manifest); print("COMPLETE_AUDITED existing", flush=True); return
        if manifest.get("preflight", {}).get("status") != "PASS":
            preflight(manifest)
        if manifest.get("status") not in ("TRAINED", "AUDIT_PASSED", "AUDIT_FAILED"):
            train(manifest)
        reuse_audit(manifest); h.write_json(MANIFEST, manifest)
        if not a.audit_only:
            audit(manifest)
        else:
            audit(manifest)
        verify_guard(manifest); manifest["status"] = "AUDITED"; h.write_json(MANIFEST, manifest)
        print("FOLD4 AUDIT PASS", flush=True)

if __name__ == "__main__": main()
