"""Run and audit the sole formal E82.4.6.8-G1 Fold-3 task."""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, sys
from pathlib import Path
from core import CONFIG, ROOT, RUNTIME, RESULTS, MANIFEST, TASKS, FOLDS, GROUP, h, implementation, paths, source_manifests, engine

E464 = Path(CONFIG["source_e8246_directory"])
E463 = Path(CONFIG["source_e82463_directory"])
E465 = Path(CONFIG["source_e82465_directory"])
E466 = Path(CONFIG["source_e8266_directory"])
E467 = Path(CONFIG["source_e8267_directory"])

RESULT_NAMES = ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv", "membership_audit.json", "audit_status.json")

def protected_sources():
    files = []
    groups = [
        (E464, ["config.json", "core.py", "execute.py", "manifest.json"]),
        (E463, ["config.json", "darg_only.py", "execute.py", "manifest.json"]),
        (E465, ["config.json", "core.py", "execute.py", "independent_465.py", "manifest.json"] + [f"results/{n}" for n in RESULT_NAMES]),
        (E466, ["config.json", "core.py", "execute.py", "independent_465.py", "manifest.json"] + [f"results/{n}" for n in RESULT_NAMES]),
        (E467, ["config.json", "core.py", "execute.py", "independent_467.py", "manifest.json"] + [f"results/{n}" for n in RESULT_NAMES]),
        (Path(CONFIG["full_path_directory"]), ["config.json", "extension.py", "manifest.json"]),
        (Path("/model/spromptgl/experiment/experiment82_4_6_pcsr_scmr"), ["run.py", "g2.py", "common/evidence.py", "common/spectral.py", "common/optimization.py", "common/history.py"]),
    ]
    for root, names in groups:
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
    direct = h.read_json(E467 / "manifest.json")
    h.require(direct.get("status") == CONFIG["required_parent_status"], "E82.4.6.7 parent is incomplete")
    h.require(h.sha_file(E467 / "manifest.json") == CONFIG["expected_manifest_sha256"]["e8267_g1"], "E82.4.6.7 manifest SHA differs")
    m = {"experiment": CONFIG["experiment"], "status": "PREFLIGHT_RUNNING", "directory": str(ROOT),
         "config": CONFIG, "implementation": implementation(), "implementation_sha": h.sha(implementation()),
         "protected_sha256": protected_sources(), "device": device, "parent_training_config": h.hist_config(),
         "source_status": {**{k: v.get("status") for k, v in sources.items()}, "e8267": direct.get("status")},
         "expected_tasks": [f"{GROUP}_f{f:02d}" for f in FOLDS], "reuse": {}, "guard_before": None}
    m["guard_before"] = verify_guard(m)
    return m

def preflight(manifest):
    from independent_468 import preflight_task
    hashes = manifest["implementation"]
    result = preflight_task(GROUP, 3, hashes)
    h.write_json(paths(GROUP, 3)["preflight"], result)
    h.require(result["status"] == "PASS" and result["implementation"] == hashes, "Fold-3 preflight failed")
    manifest.update(status="PREFLIGHT_PASSED", preflight={"status": "PASS", "task": "G1_f03", "sha": h.sha_file(paths(GROUP, 3)["preflight"])}, frozen_before_training=True)
    h.write_json(MANIFEST, manifest)
    print("PREFLIGHT G1 Fold 3 PASS", flush=True)

def train(manifest):
    files = paths(GROUP, 3)
    if files["task"].exists():
        t = h.read_json(files["task"])
        h.require(t["status"] == "TRAINED" and t["implementation"] == manifest["implementation"], "Existing Fold-3 task damaged")
        print("Verified existing Fold-3 formal task", flush=True)
        return
    manifest["status"] = "TRAINING"; h.write_json(MANIFEST, manifest)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    proc = subprocess.run(["taskset", "-c", str(CONFIG["worker_cpu_candidates"][0]), sys.executable, "-u", str(ROOT / "execute.py"), "--worker", "G1", "--fold", "3"], cwd=ROOT, env=env)
    h.require(proc.returncode == 0, "Fold-3 worker failed")
    h.require(files["task"].exists(), "Fold-3 task artifact missing")
    t = h.read_json(files["task"])
    h.require(t["status"] == "TRAINED" and t["implementation"] == manifest["implementation"] and all(h.sha_file(files[k]) == t[k + "_sha"] for k in ("best", "latest", "journal")), "Fold-3 artifact integrity failed")
    manifest["status"] = "TRAINED"; manifest["completed_task"] = t; h.write_json(MANIFEST, manifest)

def _verify_result_set(directory, manifest_sha):
    m = h.read_json(directory / "manifest.json")
    h.require(m.get("status") == "COMPLETE_AUDITED", f"{directory.name} is not COMPLETE_AUDITED")
    h.require(h.sha_file(directory / "manifest.json") == manifest_sha, f"{directory.name} manifest SHA differs")
    for rel, digest in m.get("result_files", {}).items():
        p = directory / rel; h.require(p.exists() and h.sha_file(p) == digest, f"Historical result changed {p}")
    a = h.read_json(directory / "results/audit_status.json")
    h.require(a.get("status") == "COMPLETE_AUDITED", f"{directory.name} audit status incomplete")
    return m, a

def reuse_audit(manifest):
    expected = CONFIG["expected_manifest_sha256"]
    e4m, _ = _verify_result_set(E464, expected["e82464_cumulative"])
    e3m, _ = _verify_result_set(E463, expected["e82463_darg_only"])
    e5m, _ = _verify_result_set(E465, expected["e82465_g1"])
    e6m, e6a = _verify_result_set(E466, expected["e8266_g1"])
    e7m, e7a = _verify_result_set(E467, expected["e8267_g1"])
    h.require(e6m.get("actual_new_folds") == [4] and e7m.get("actual_new_folds") == [2], "Historical fold ownership differs")
    h.require(e6m["fold4_result"]["best_sha"] == e6a["fold4"]["best_sha"] and e6m["fold4_result"]["latest_sha"] == e6a["fold4"]["latest_sha"], "E6 Fold-4 checkpoint identity differs")
    # Verify E6's inherited checkpoint identities through E65's frozen source manifest.
    inherited = {}
    e65_sources = e5m["checkpoint_sources"]
    for fold, item in e65_sources.items():
        for key in ("best", "latest", "journal"):
            p = Path(item[key]); h.require(p.exists() and h.sha_file(p) == item[key + "_sha"], f"Historical Fold {fold} {key} checkpoint changed")
        inherited[str(fold)] = {"source_kind": item["source_kind"], "source_manifest": item["source_manifest"], "source_manifest_sha": item.get("source_manifest_sha"), "best": item["best"], "best_sha": item["best_sha"], "latest": item["latest"], "latest_sha": item["latest_sha"], "journal": item["journal"], "journal_sha": item["journal_sha"]}
    old_rows = E6 / "results/patient_level.csv" if False else None
    rows = {"direct_reference": {"source_directory": str(E467), "source_manifest_sha": h.sha_file(E467 / "manifest.json"), "source_audit_sha": h.sha_file(E467 / "results/audit_status.json"), "patient_rows": 871},
            "fold2_rollback": {"source_directory": str(E466), "source_manifest_sha": h.sha_file(E466 / "manifest.json"), "source_audit_sha": h.sha_file(E466 / "results/audit_status.json"), "source_task": "G1_f02", "checkpoint_identity": e6m["reuse"]["2"]},
            "fold3": {"source_directory": str(ROOT), "source_task": "G1_f03", "checkpoint_identity": "local retrained"}}
    for fold in [1, 4, 5, 6, 7, 8, 9, 10]:
        rows[str(fold)] = {"source_directory": str(E467), "source_manifest_sha": h.sha_file(E467 / "manifest.json"), "source_audit_sha": h.sha_file(E467 / "results/audit_status.json"), "source_task": f"G1_f{fold:02d}", "checkpoint_identity": "E82.4.6.7-G1 read-only reuse", "result_files_sha": {n: h.sha_file(E467 / "results" / n) for n in RESULT_NAMES}}
    rows["inherited_checkpoint_sources"] = inherited
    rows["fold2_exact_recovery"] = {"source": str(E466), "source_manifest_sha": h.sha_file(E466 / "manifest.json"), "patient_level_sha": h.sha_file(E466 / "results/patient_level.csv"), "fold_metrics_sha": h.sha_file(E466 / "results/fold_metrics.csv"), "membership_sha": h.sha_file(E466 / "results/membership_audit.json"), "audit_sha": h.sha_file(E466 / "results/audit_status.json")}
    rows["fold3_reference"] = {"source": str(E467), "source_manifest_sha": h.sha_file(E467 / "manifest.json"), "patient_level_sha": h.sha_file(E467 / "results/patient_level.csv"), "patient_413_reference": "fold=3, patient_id_1based=413"}
    manifest["reuse"] = rows
    return rows

def audit(manifest):
    from independent_468 import audit_task
    _, cert = audit_task(GROUP, 3, manifest)
    manifest["audit"] = {"G1_f03": cert}; manifest["status"] = "AUDIT_PASSED"
    h.write_json(RESULTS / "audit_status.json", {"status": "PASS", "tasks": {"G1_f03": cert}, "reused": manifest["reuse"]})
    h.write_json(MANIFEST, manifest)
    return cert

def main():
    p = argparse.ArgumentParser(); p.add_argument("--worker", choices=["G1"]); p.add_argument("--fold", type=int, choices=[3]); p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    for d in (RUNTIME, RESULTS, ROOT / "checkpoints"): d.mkdir(exist_ok=True)
    if a.worker:
        h.require(a.fold == 3 and len(os.sched_getaffinity(0)) == 1, "Worker affinity invalid")
        h.archive.verify_device(); engine("G1", 3).train_fold("G2", 3); return
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = h.read_json(MANIFEST) if MANIFEST.exists() else freeze_manifest()
        h.require(manifest["implementation"] == implementation() and manifest["config"] == CONFIG, "Frozen source/config differs")
        if manifest.get("status") == "COMPLETE_AUDITED":
            verify_guard(manifest); print("COMPLETE_AUDITED existing", flush=True); return
        if manifest.get("preflight", {}).get("status") != "PASS": preflight(manifest)
        if manifest.get("status") not in ("TRAINED", "AUDIT_PASSED", "AUDIT_FAILED"): train(manifest)
        reuse_audit(manifest); h.write_json(MANIFEST, manifest)
        audit(manifest)
        verify_guard(manifest); manifest["status"] = "AUDITED"; h.write_json(MANIFEST, manifest)
        print("FOLD3 AUDIT PASS", flush=True)

if __name__ == "__main__": main()
