#!/usr/bin/env python3
"""Write the final report and manifest from the completed independent audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git_value(*args):
    result = subprocess.run(["git", "-C", "/tmp/spromptgl_git", *args],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def source_files():
    root = ROOT / "inputs/source_repo"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    return rows


def main():
    audit = read_json(ROOT / "results/audit_status.json")
    require(audit["status"] == "INCOMPLETE_METRIC_RECALCULATION_EVIDENCE",
            "The independent audit status is absent or unexpected")
    require(audit["historical_files_sha_protection"]["status"] == "PASS"
            and audit["historical_files_sha_protection"]["historical_file_count"] == 1423,
            "The completed historical checksum result is absent")
    for stage, row in audit["author_logs"].items():
        path = Path(row["path"])
        require(path.is_file() and sha256(path) == row["sha256"],
                f"{stage} author log changed after its independent audit")

    recovery = {
        "repository": git_value("remote", "get-url", "origin"),
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "commit_record": git_value("log", "-1", "--format=%H%n%cI%n%s"),
        "copy_destination": str(ROOT / "inputs/source_repo"),
        "copied_file_count": len(source_files()),
        "files": source_files(),
    }
    (ROOT / "results/source_recovery.json").write_text(
        json.dumps(recovery, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    training = {"Warmup": audit["warmup"], "DARG": audit["darg"], "DPSL": audit["dpsl"]}
    report = [
        "# ABIDE Authentic Training Log Reconstruction v2",
        "",
        "The three training stages completed on physical GPU 0. The final audit distinguishes verified historical state and RNG evidence from unavailable raw outputs for a subset of older epoch metrics.",
        f"Recovered source snapshot: `{recovery['repository']}` branch `{recovery['branch']}` commit `{recovery['commit']}` ({recovery['commit_record'].splitlines()[-1]}).",
        "",
        f"- Warmup: {training['Warmup']['status']}, 10/10 folds, epoch-10 endpoint correct {training['Warmup']['correct']}/{training['Warmup']['n']}.",
        f"- DARG: {training['DARG']['status']}, 10/10 folds, {training['DARG']['correct']}/{training['DARG']['n']} correct.",
        f"- DPSL: {training['DPSL']['status']}, 10/10 folds, {training['DPSL']['correct']}/{training['DPSL']['n']} correct.",
        "- All three author-format logs pass exact row formatting, field ordering, summary formatting, timing, and attention-map checks.",
        "- Ten randomly selected DPSL training epochs, one per fold, were independently recomputed from stored train logits and labels for both train and validation metrics.",
        "- Warmup and DARG per-epoch raw train logits were not archived. Their live-loop scalar records and final outputs were checked, but the raw-output recalculation criterion remains open.",
        "- COMPLETE_AUDITED_THREE_STAGE_LOG_RECONSTRUCTION is withheld because the Warmup and DARG raw-output recalculation criterion is not verifiable.",
        "",
        "## Logs",
        "",
    ]
    for stage, row in audit["author_logs"].items():
        report.append(f"- {stage}: `{row['path']}`, {row['lines']} lines, {row['epoch_rows']} epoch rows, SHA-256 `{row['sha256']}`.")

    report.extend(["", "## Performance", "",
                   "| Stage | Correct | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for stage in ("DARG", "DPSL"):
        item = training[stage]
        macro = item["macro"]
        cells = [f"{macro[key]['mean']*100:.4f}% +/- {macro[key]['std_population']*100:.4f}%"
                 for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")]
        report.append(f"| {stage} | {item['correct']}/871 | " + " | ".join(cells) + " |")

    report.extend(["", "## Fold Epochs And Time", "",
                   "| Fold | DARG Best/Latest | DPSL Best/Latest | Warmup sec | DARG sec | DPSL sec |",
                   "|---:|---:|---:|---:|---:|---:|"])
    for fold in range(10):
        warmup = training["Warmup"]["folds"][fold]
        darg = training["DARG"]["folds"][fold]
        dpsl = training["DPSL"]["folds"][fold]
        report.append(
            f"| {fold + 1} | {darg['best_epoch']}/{darg['latest_epoch']} | "
            f"{dpsl['best_epoch']}/{dpsl['latest_epoch']} | {warmup['train_seconds']:.4f} | "
            f"{darg['train_seconds']:.4f} | {dpsl['train_seconds']:.4f} |"
        )
    for stage in ("Warmup", "DARG", "DPSL"):
        seconds = sum(row["train_seconds"] for row in training[stage]["folds"])
        report.append(f"- {stage} measured stage time total: {seconds:.4f} sec.")

    report.extend(["", "## Audit", "",
                   "- E61 endpoint model, CAM, all three optimizer states, RNG, split, pairs, anchor identity, Best epoch/ACC, and structured state SHA match in 10/10 folds. Epoch-by-epoch RNG and six-call Dropout order match. Historical per-epoch model parameter snapshots do not exist, so only endpoint parameters and RNG trajectory are independently comparable.",
                   "- The E61 checkpoint `source_sha` metadata records a different `run.py` hash than the recovered audited bytes in all ten folds. No override was used. Current source bytes match the audited source snapshot, and the reproduced initialization and endpoint are exact.",
                   "- DARG and DPSL Best/Latest tensors, per-epoch RNG, patient predictions/probabilities, and post-Latest restoration witnesses match their frozen reference archives in 10/10 folds.",
                   "- DPSL adapter preflights and complete per-update journals preserve the 19 overlap patients in the DARG path while excluding them from new-module legal supervision and EMA/Gaussian statistics. The original transition enforces zero excluded-row residual and the configured residual bound.",
                   f"- The frozen historical file inventory passed SHA-256 verification for {audit['historical_files_sha_protection']['historical_file_count']} files. The original author log SHA-256 is `{audit['historical_files_sha_protection']['author_result_abide_sha256']}`.",
                   "- Earlier experiment manifests record the root attention-map, graph, and dialogue paths as pre-existing local changes. This run did not modify those paths. The current project worktree is clean, and the frozen SHA inventory and task-specific source hashes pass.",
                   "- One preserved fold-2 DPSL partial attempt lacked author capture and was excluded from results. The formal fold was rerun from epoch 10 and matches the archive. A fold-5 capture attempt stopped before durable commit and was also restarted from epoch 10.",
                   "- The fold-2 unlogged checkpoints and journal are preserved under `DPSL/failed_attempts/fold02_unlogged/`.",
                   "- Warmup and DARG raw train outputs are the only remaining audit limitation. See `audit_status.json` and `metrics.csv` for the machine-readable results.",
                   ""])
    (ROOT / "results/report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts:
            files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    date = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], check=True,
                          capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment": "experiment_abide_authentic_logs_v2", "authority_directory": str(ROOT),
        "status": audit["status"], "created_utc": date,
        "source_recovery": {key: recovery[key] for key in ("repository", "branch", "commit", "commit_record", "copied_file_count")},
        "historical_file_sha_protection": audit["historical_files_sha_protection"],
        "stage_status": {stage: training[stage]["status"] for stage in training},
        "author_logs": {stage: {key: row[key] for key in ("path", "lines", "epoch_rows", "sha256", "bytes")}
                        for stage, row in audit["author_logs"].items()},
        "files": files,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "manifest_files": len(files),
                      "log_sha256": {key: row["sha256"] for key, row in audit["author_logs"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
