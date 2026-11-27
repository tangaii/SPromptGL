#!/usr/bin/env python3
"""Append the single authorized handoff block and seal final file identities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
RESULTS = EXP / "results"
HIST = Path("/model/spromptgl/experiment/experiment_dresp_tsne_compact_separation")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
MARKER = "<!-- DRESP_DUAL_LABEL_GUIDED_FIVE_MODELS -->"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return digest(path.read_bytes())


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_handoff():
    prefix_record = read_json(RESULTS / "handoff_prefix.json")
    before = HANDOFF.read_bytes()
    prefix_bytes = int(prefix_record["bytes"])
    if digest(before[:prefix_bytes]) != prefix_record["sha256"]:
        raise RuntimeError("handoff historical prefix changed since the experiment started")
    if MARKER.encode() in before:
        raise RuntimeError("handoff marker already exists; refusing to append a duplicate block")
    report = read_json(RESULTS / "run_manifest.json")
    audit = read_json(RESULTS / "audit_status.json")
    metric_rows = []
    import csv
    with (RESULTS / "formal_summary.csv").open(newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    lookup = {(row["method"], row["group"], row["model"]): row for row in metric_rows}
    cam_c5 = lookup[("C5", "C5", "CAM_only")]
    dresp_c5 = lookup[("C5", "C5", "DReSP")]
    pooled_rows = []
    import csv as _csv
    with (RESULTS / "formal_metrics.csv").open(newline="") as handle:
        pooled_rows = list(_csv.DictReader(handle))
    def pooled(method, group, field):
        values = [float(row[field]) for row in pooled_rows if row["method"] == method and row["group"] == group]
        return float(np.mean(values))
    body = [
        "",
        MARKER,
        "## DReSP 双路线标签引导 t-SNE 探索（五模型）",
        "",
        "目的：在不改变模型训练结果、权重和患者特征的前提下，对 CAM-only、SPromptGL、DARG、DARG + JSL、DReSP 的原始 C5、后置二维标签变换和监督式距离 t-SNE 进行统一探索。",
        "方法：Method A 为 Post-hoc Label-Guided Transformation，Method B 为使用真实或固定打乱标签修改 PCA50 距离后的 Supervised t-SNE。P0–P9、S0–S9、N1/N2 均按预注册配置完成。",
        f"完成状态：{audit['status']}。参数扫描覆盖 A、B 各 25 组，在 Fold 7/8 对五模型完成，共 {report['scan_rows']} 行。正式数据包含 {report['formal_rows']} 行，覆盖十折五模型。",
        f"五模型 C5 参考中，CAM-only 二维 Silhouette 均值为 {float(cam_c5['silhouette_2d_mean']):.4f}，DReSP 为 {float(dresp_c5['silhouette_2d_mean']):.4f}。逐折指标、P/S/N 配对差异和高维邻域保持度保存在正式结果表。",
        f"统一几何参考中，A/P9 的 pooled mean within={pooled('A', 'P9', 'within_mean_distance'):.4f}、between/within={pooled('A', 'P9', 'between_within_ratio'):.4f}；B/S8 为 mean within={pooled('B', 'S8', 'within_mean_distance'):.4f}、between/within={pooled('B', 'S8', 'between_within_ratio'):.4f}。这些是预设标签约束的结果。",
        "视觉变化：Method A 的紧凑/分离方向由 alpha、beta 公式直接规定。Method B 的变化源于标签参与距离构造。随机标签负向对照展示外加标签约束可产生的二维几何变化。",
        "限制：测试患者真实标签参与可视化，所有标签引导图必须标记 LABEL-GUIDED / SUPERVISED / EXPLORATORY。二维分离不代表 ACC、AUC、诊断能力或泛化提高。",
        f"权威实验目录：{EXP}",
        f"报告：{EXP / 'results/report.md'}；图像：{EXP / 'results/figures'}；审计：{EXP / 'results/audit_status.json'}。",
        "",
    ]
    block = "\n".join(body).encode()
    with HANDOFF.open("ab") as handle:
        handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    after = HANDOFF.read_bytes()
    if after[:len(before)] != before or after[:prefix_bytes] != before[:prefix_bytes]:
        raise RuntimeError("handoff readback failed to preserve the historical prefix")
    if after.count(MARKER.encode()) != 1 or not after.endswith(block):
        raise RuntimeError("handoff appended block readback failed")
    readback = {
        "status": "PASS",
        "historical_prefix_bytes": prefix_bytes,
        "historical_prefix_sha256_before": prefix_record["sha256"],
        "historical_prefix_sha256_after": digest(after[:prefix_bytes]),
        "handoff_sha256_before_append": digest(before),
        "handoff_sha256_after_append": digest(after),
        "marker_count": after.count(MARKER.encode()),
        "appended_block_bytes": len(block),
    }
    write_json(RESULTS / "handoff_readback.json", readback)
    return readback


def main():
    audit = read_json(RESULTS / "audit_status.json")
    if audit.get("status") != "COMPLETE_AUDITED":
        raise RuntimeError("refusing to finalize an unaudited experiment")
    handoff = write_handoff()
    config = read_json(EXP / "config.json")
    run_manifest = read_json(RESULTS / "run_manifest.json")
    cam_status = read_json(RESULTS / "cam_only_rebuild/cam_only_rebuild_status.json")
    source_manifest = read_json(EXP / "inputs/source_manifest.json")
    input_sha = {path.relative_to(EXP).as_posix(): file_sha(path) for path in sorted((EXP / "inputs").rglob("*")) if path.is_file()}
    coordinate_sha = {path.relative_to(EXP).as_posix(): file_sha(path) for path in sorted((RESULTS / "coordinates").glob("*.npy"))}
    figure_sha = {path.relative_to(EXP).as_posix(): file_sha(path) for path in sorted((RESULTS / "figures").rglob("*.png"))}
    historical_coordinate_sha = {
        f"{model}_fold{fold:02}": file_sha(Path("/model/spromptgl/experiment/experiment_dresp_tsne_compact_separation/results/formal_coordinates") / f"{model}_fold{fold:02}.npy")
        for model in ("SPromptGL", "DARG", "DARG_JSL", "DReSP")
        for fold in range(1, 11)
    }
    root_model_sha = read_json(EXP / "inputs/cam_only_source/manifest.json")["protected_sha256"]
    manifest = {
        "status": "COMPLETE_AUDITED",
        "scientific_scope": "label-guided visualization only; no classification performance claims",
        "authoritative_directory": str(EXP),
        "superseded_incomplete_workspace": str(ROOT / "experiment/experiment_dresp_tsne_label_guided_dual"),
        "models": config["models"],
        "folds": 10,
        "formal_metric_rows": run_manifest["formal_rows"],
        "parameter_scan_rows": run_manifest["scan_rows"],
        "scan_configurations_per_method": 25,
        "supervised_tsne_fit_jobs": run_manifest["supervised_tsne_jobs"],
        "negative_controls": ["N1", "N2"],
        "random_label_mapping_sha256": run_manifest["shuffle_mapping_sha256"],
        "C5_parameters": config["c5"],
        "Method_A_parameters": config["method_a"],
        "Method_B_parameters": config["method_b"],
        "CAM_only_identity": cam_status["checkpoint_identity"],
        "CAM_only_feature_definition": cam_status["feature_definition"],
        "CAM_only_fold_identity": [
            {"fold": row["fold"], "anchor_sha256": row["anchor_sha256"], "archive_sha256": row["feature_archive_sha256"], "test_prediction_sha256": row["test_prediction_sha256"], "best_epoch": row["best_epoch"]}
            for row in cam_status["feature_archives"]
        ],
        "historical_feature_source_manifest_sha256": file_sha(HIST / "inputs/source_manifest.json"),
        "historical_c5_coordinate_sha256": historical_coordinate_sha,
        "protected_root_model_data_sha256": root_model_sha,
        "input_files_sha256": input_sha,
        "coordinate_files_sha256": coordinate_sha,
        "figure_files_sha256": figure_sha,
        "audit_status_sha256": file_sha(RESULTS / "audit_status.json"),
        "audit_status": audit,
        "handoff_readback": handoff,
        "handoff_readback_file_sha256": file_sha(RESULTS / "handoff_readback.json"),
        "no_manifest_self_sha": True,
    }
    write_json(EXP / "manifest.json", manifest)
    # Generate a detached checksum inventory after all deliverables are final.
    checksum_lines = []
    for path in sorted(EXP.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS" or "__pycache__" in path.parts:
            continue
        checksum_lines.append(f"{file_sha(path)}  {path.relative_to(EXP).as_posix()}")
    (EXP / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    # Read back each detached entry and ensure it still identifies the final file.
    for line in (EXP / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if file_sha(EXP / name) != expected:
            raise RuntimeError(f"SHA256SUMS verification failed: {name}")
    print(json.dumps({"status": manifest["status"], "handoff": handoff, "checksum_entries": len(checksum_lines)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
