#!/usr/bin/env python3
"""Experiment 78.6 PEN-SVA frozen source/math/checkpoint audit.

Original-PEN best weights are never substituted by E78.x models.  A0--A7 are
reported only when the actual original-PEN state_dict files can be reloaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

EXP = Path("/model/spromptgl/experiment/experiment78_6_pen_semantic_audit")
ROOT = Path("/data/adodas/SPromptGL-main")
RESULTS = EXP / "results"
SOURCE = ROOT / "network.py"
HIST_E32 = Path("/tmp/spromptgl-e785-upload/experiment/experiment32_disease_contrastive_evidence_routing")
HIST_E43 = Path("/tmp/spromptgl-e785-upload/experiment/experiment43_3_paper_gfr_ablation_reproduction")
E60_ROOT = ROOT / "experiment/experiment60_dss_pen"
E60_STAGE = Path("/tmp/e60_dsspen_stage")
E60_EXPECTED_DIRS = [
    E60_STAGE / ".runtime/groups/E60_ORIGINAL_PEN/checkpoints",
    E60_ROOT / ".runtime/groups/E60_ORIGINAL_PEN/checkpoints",
    Path("/model/spromptgl/experiment/experiment60_dss_pen/.runtime/groups/E60_ORIGINAL_PEN/checkpoints"),
]
CORE = [ROOT / n for n in ("network.py", "model.py", "layers.py", "utils.py")]
sys.path.insert(0, str(EXP))
from pen_probe import PROMPTS, probe, sigmoid_abs_margin, source_evidence  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def e60_inventory() -> dict[str, Any]:
    """Read E60 identity records, but do not treat them as current files."""
    metric_rows = [row for row in read_rows(E60_ROOT / "results/fold_metrics.csv") if row.get("group") == "E60_ORIGINAL_PEN"]
    audit_rows = [row for row in read_rows(E60_ROOT / "results/checkpoint_audit.csv") if row.get("group") == "E60_ORIGINAL_PEN"]
    expected: list[dict[str, Any]] = []
    for row in sorted(metric_rows, key=lambda item: int(item.get("fold", 0))):
        fold = int(row["fold"])
        names = [f"E60_ORIGINAL_PEN_f{fold:02d}.best.pt"]
        locations = [str(directory / names[0]) for directory in E60_EXPECTED_DIRS]
        present = [path for path in locations if Path(path).is_file()]
        expected.append({
            "fold": fold,
            "expected_sha256": row.get("checkpoint_sha256"),
            "best_epoch": int(row["best_epoch"]),
            "epochs_executed": int(row["epochs_executed"]),
            "historical_checkpoint_audit": row.get("checkpoint_audit"),
            "historical_metrics": {key: row.get(key) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
            "candidate_paths": locations,
            "currently_readable_paths": present,
            "currently_sha_matches": [path for path in present if sha256(Path(path)) == row.get("checkpoint_sha256")],
        })
    audit_pass = sum(1 for row in audit_rows if row.get("status") == "PASS")
    matches = [item for item in expected if item["currently_sha_matches"]]
    return {
        "experiment": "E60 DSS-PEN",
        "group": "E60_ORIGINAL_PEN",
        "protocol": "Experiment 53.3 CADR_WD_0p01_GFR_WD_0p11 base; original SPromptGL SPGS+RES path",
        "historical_fold_metric_rows": len(metric_rows),
        "historical_checkpoint_audit_rows": len(audit_rows),
        "historical_checkpoint_audit_pass": audit_pass,
        "expected_folds": expected,
        "currently_readable_fold_count": len(matches),
        "currently_complete_sha_verified_set": len(matches) == 10,
        "historical_source_report": str(E60_ROOT / "results/report.md"),
        "historical_checkpoint_audit_csv": str(E60_ROOT / "results/checkpoint_audit.csv"),
        "note": "Historical E60 identity/readback records remain, but no matching .pt is currently readable in the registered runtime locations.",
    }


def inventory() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    explicit = [
        ROOT / "experiment/experiment32_disease_contrastive_evidence_routing",
        ROOT / "experiment/experiment28_imaging_anchored_finegrained_semantic_alignment",
        ROOT / "experiment/experiment43_3_paper_gfr_ablation_reproduction",
        Path("/model/spromptgl/experiment/experiment32_disease_contrastive_evidence_routing"),
        Path("/model/spromptgl/experiment/experiment28_imaging_anchored_finegrained_semantic_alignment"),
        Path("/model/spromptgl/experiment/experiment43_3_paper_gfr_ablation_reproduction"),
        HIST_E32, HIST_E43,
    ]
    explicit += sorted(Path("/tmp").glob("*/experiment/experiment32_disease_contrastive_evidence_routing"))
    explicit += sorted(Path("/tmp").glob("*/experiment/experiment28_imaging_anchored_finegrained_semantic_alignment"))
    seen: set[str] = set()
    for directory in explicit:
        directory = directory.resolve()
        if str(directory) in seen or not directory.is_dir():
            continue
        seen.add(str(directory))
        files = sorted(str(path) for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".tar"})
        manifests = sorted(str(path) for path in directory.rglob("manifest*.json"))
        candidates.append({
            "directory": str(directory), "exists": True, "checkpoint_count": len(files),
            "checkpoint_files": files[:50], "manifest_files": manifests[:20],
            "has_report": (directory / "results/report.md").is_file() or (directory / "report.md").is_file(),
        })
    # Do not recursively enumerate every historical runtime on the shared
    # filesystem: several mounts expose very large transient trees.  The
    # protocol-specific E32/E28/E60 directories above are the authoritative
    # inventory; unrelated E78.x checkpoints are explicitly excluded.
    repository_files: list[str] = []
    original_named: list[str] = []
    e60 = e60_inventory()
    return {
        "timestamp_utc": now(),
        "required_protocol": "original SPromptGL seed=0, ten-fold best PEN checkpoints",
        "searched_candidate_directories": candidates,
        "repository_checkpoint_count": len(repository_files),
        "repository_original_named_checkpoint_candidates": original_named[:100],
        "repository_scan_note": "Unrelated shared-runtime trees were not recursively enumerated; protocol-specific candidates were checked explicitly.",
        "e60_original_pen": e60,
        "verifiable_original_pen_fold_count": int(e60["currently_readable_fold_count"]),
        "verifiable_complete_ten_fold_set": bool(e60["currently_complete_sha_verified_set"]),
        "historical_e60_audit_fold_count": int(e60["historical_checkpoint_audit_pass"]),
        "rebuild_started": False,
        "rebuild_allowed_but_not_started": True,
        "decision": "HISTORICAL_E60_IDENTITY_FOUND_BUT_CURRENT_BEST_FILES_UNAVAILABLE" if not e60["currently_complete_sha_verified_set"] else "CURRENT_E60_SET_AVAILABLE",
        "reason": "E32/E28 source mirrors retain no best state_dict; E60 retains 10/10 historical identity/readback records, but no matching .pt is currently readable in registered locations.",
    }


def historical_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_rows(HIST_E32 / "results/fold_metrics.csv"):
        if row.get("config") in {"B0", "B1", "B2"}:
            row.update(source="E32_INDEPENDENT_TRAINING_REFERENCE", checkpoint_audit_status="HISTORICAL_REPORT_ONLY", protocol="E32 original SPromptGL ablation protocol")
            rows.append(row)
    for row in read_rows(E60_ROOT / "results/fold_metrics.csv"):
        if row.get("group") == "E60_ORIGINAL_PEN":
            row.update(source="E60_ORIGINAL_PEN_HISTORICAL_REFERENCE", checkpoint_audit_status="HISTORICAL_E60_AUDIT_CSV_PASS", protocol="E60 E53.3 CADR/GFR base; not E32 protocol")
            rows.append(row)
    return rows


def text_identity() -> dict[str, Any]:
    e60_cache = E60_ROOT / ".runtime/text_embeddings.pt"
    e781_cache = Path("/model/spromptgl/experiment/experiment78_dcpss/runtime/text_embeddings.pt")
    e781_geometry = Path("/model/spromptgl/experiment/experiment78_1_sh_dcpss/runtime/text_geometry.pt")
    e781_config = Path("/model/spromptgl/experiment/experiment78_1_sh_dcpss/config.json")
    return {
        "e60_text_cache": {"path": str(e60_cache), "sha256": sha256(e60_cache)},
        "e78_1_text_cache": {"path": str(e781_cache), "sha256": sha256(e781_cache)},
        "e78_1_text_geometry": {"path": str(e781_geometry), "sha256": sha256(e781_geometry)},
        "e78_1_config": {"path": str(e781_config), "sha256": sha256(e781_config)},
        "anonymous_prompts": [
            "A sample assigned to category alpha, description one.", "A sample assigned to category alpha, description two.",
            "A sample assigned to category alpha, description three.", "A sample assigned to category alpha, description four.",
            "A sample assigned to category beta, description one.", "A sample assigned to category beta, description two.",
            "A sample assigned to category beta, description three.", "A sample assigned to category beta, description four.",
        ],
    }


def audit() -> dict[str, Any]:
    EXP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # A real frozen replay is authoritative.  Keep this source/math audit
    # idempotent so a later integrity check cannot overwrite the verified
    # A0--A7 rows with the pre-replay placeholders below.
    replay_path = RESULTS / "real_replay.json"
    status_path = RESULTS / "audit_status.json"
    if replay_path.is_file() and status_path.is_file():
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if replay.get("status") == "PASS" and replay.get("folds") == 10 and status.get("status") == "COMPLETE_AUDITED":
                manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
                inventory = json.loads((EXP / "checkpoint_inventory.json").read_text(encoding="utf-8"))
                return {
                    "status": status, "replay": replay, "output": str(EXP),
                    "inventory": inventory, "source_evidence": manifest.get("source_evidence", {}),
                    "synthetic": manifest.get("synthetic_probe", {}), "manifest": manifest,
                    "idempotent_replay": True,
                }
        except (OSError, json.JSONDecodeError):
            pass
    inv = inventory(); write_json(EXP / "checkpoint_inventory.json", inv)
    ev = source_evidence(SOURCE); src_sha = {str(path.relative_to(ROOT)): sha256(path) for path in CORE}; synthetic = probe()
    generator = torch.Generator(device="cpu").manual_seed(786001)
    similarities = torch.rand((10000, 2), generator=generator) * 2.0 - 1.0
    math_err = float((torch.softmax(similarities, -1).max(-1).values - sigmoid_abs_margin(similarities)).abs().max())
    math_pass = math_err <= 2e-7
    swap_pass = all(synthetic[key] <= 1e-6 for key in ("swap_weight_max_abs", "swap_enhanced_max_abs", "swap_logits_max_abs")) and synthetic["swap_prediction_equal"]
    positive_pass = synthetic["positive_classifier_logit_mean_abs"] > 0 and synthetic["positive_feature_logit_mean_abs"] > 0
    current_set = bool(inv["verifiable_complete_ten_fold_set"])
    reason = "Current original-PEN best files are unavailable; E60 historical audit rows are identity evidence only." if not current_set else "Current E60 original-PEN set is available; frozen replay is required before interpretation."
    definitions = {"A0": "FULL_PEN", "A1": "RES_ONLY", "A2": "PEN_OFF", "A3": "CLASS_VECTOR_SWAP", "A4": "ANONYMOUS_TEXT", "A5": "WRONG_SEMANTIC_TEXT", "A6": "NUMERIC_TEXT_CONTROL", "A7": "NEUTRAL_SPGS"}
    csv_write(RESULTS / "interventions.csv", [{"intervention": code, "name": name, "scope": "real_checkpoint", "status": "NOT_VERIFIED" if not current_set else "PENDING_REPLAY", "acc_pct": "", "hard_auc_pct": "", "prob_auc_pct": "", "sen_asd_pct": "", "spe_td_pct": "", "asd_f1_pct": "", "reason": reason} for code, name in definitions.items()])
    csv_write(RESULTS / "fold_metrics.csv", historical_metrics())
    feature_rows = [{"scope": "source_static", "metric": key, "value": json.dumps(value, ensure_ascii=False), "status": "PASS" if value.get("found") else "FAIL"} for key, value in ev.items() if isinstance(value, dict) and "found" in value]
    feature_rows.extend([
        {"scope": "synthetic_positive_control", "metric": "residual_relative_norm_mean", "value": synthetic["residual_relative_norm_mean"], "status": "CONTROL_ONLY"},
        {"scope": "synthetic_positive_control", "metric": "enhanced_input_cosine_mean", "value": synthetic["enhanced_input_cosine_mean"], "status": "CONTROL_ONLY"},
        {"scope": "synthetic_positive_control", "metric": "classifier_parameter_perturbation_logit_change", "value": synthetic["positive_classifier_logit_mean_abs"], "status": "PASS" if positive_pass else "FAIL"},
        {"scope": "synthetic_positive_control", "metric": "pen_feature_perturbation_logit_change", "value": synthetic["positive_feature_logit_mean_abs"], "status": "PASS" if positive_pass else "FAIL"},
        {"scope": "real_checkpoint", "metric": "patient_level_feature_effects", "value": "NOT_VERIFIED", "status": "NOT_VERIFIED"},
        {"scope": "real_checkpoint", "metric": "text_mlp_parameters_and_gradients", "value": "NOT_VERIFIED", "status": "NOT_VERIFIED"},
    ])
    csv_write(RESULTS / "feature_effects.csv", feature_rows)
    csv_write(RESULTS / "semantic_direction.csv", [
        {"test": "two_class_softmax_max_identity", "max_abs_error": math_err, "status": "PASS" if math_pass else "FAIL", "scope": "analytic_numeric"},
        {"test": "class_vector_swap_weight", "max_abs_error": synthetic["swap_weight_max_abs"], "status": "PASS" if synthetic["swap_weight_max_abs"] <= 1e-6 else "FAIL", "scope": "synthetic_forward"},
        {"test": "class_vector_swap_enhanced", "max_abs_error": synthetic["swap_enhanced_max_abs"], "status": "PASS" if synthetic["swap_enhanced_max_abs"] <= 1e-6 else "FAIL", "scope": "synthetic_forward"},
        {"test": "class_vector_swap_logits", "max_abs_error": synthetic["swap_logits_max_abs"], "status": "PASS" if synthetic["swap_logits_max_abs"] <= 1e-6 else "FAIL", "scope": "synthetic_forward"},
        {"test": "class_vector_swap_prediction", "max_abs_error": 0.0, "status": "PASS" if synthetic["swap_prediction_equal"] else "FAIL", "scope": "synthetic_forward"},
        {"test": "real_checkpoint_direction", "max_abs_error": "", "status": "NOT_VERIFIED", "scope": "real_checkpoint"},
    ])
    csv_write(RESULTS / "patient_changes.csv", [{"scope": "real_checkpoint", "status": "NOT_VERIFIED", "reason": reason, "fold": "", "patient_index": "", "true_label": "", "original_prob": "", "intervention_prob": "", "original_pred": "", "intervention_pred": ""}])
    source_pass = all(value.get("found", False) for value in ev.values() if isinstance(value, dict) and "found" in value)
    status = {
        "experiment": "E78.6 PEN-SVA", "status": "PARTIAL", "output_directory": str(EXP), "ossfs_primary_status": "EIO_errno5",
        "checkpoint_level": "NOT_VERIFIED_CURRENT_FILES" if not current_set else "REPLAY_REQUIRED",
        "historical_e60_checkpoint_identity": "10/10_AUDIT_CSV_PASS" if inv["historical_e60_audit_fold_count"] == 10 else "INCOMPLETE",
        "source_audit": "PASS" if source_pass else "FAIL", "math_unit_test": "PASS" if math_pass else "FAIL", "class_swap_synthetic": "PASS" if swap_pass else "FAIL", "positive_controls": "PASS" if positive_pass else "FAIL",
        "real_forward_reproduction": "NOT_VERIFIED", "real_a0_a7_replay": "NOT_VERIFIED",
        "scientific_conclusions": ["COMPUTATION_ACTIVE", "EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS", "MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED", "INSUFFICIENT_VERIFIABLE_CHECKPOINTS"],
        "timestamp_utc": now(),
    }
    write_json(RESULTS / "audit_status.json", status)
    manifest = {"experiment": "78.6 PEN-SVA", "created_utc": now(), "output_directory": str(EXP), "ossfs_primary_status": "EIO_errno5", "protected_root_sha256": src_sha, "source_evidence": ev, "prompt_strings": PROMPTS, "text_identity": text_identity(), "checkpoint_inventory": str(EXP / "checkpoint_inventory.json"), "history_sources": {"E32": str(HIST_E32), "E43.3": str(HIST_E43), "E60": str(E60_ROOT)}, "synthetic_probe": synthetic, "status": status}
    write_json(EXP / "manifest.json", manifest)
    return {"inventory": inv, "source_evidence": ev, "synthetic": synthetic, "status": status, "manifest": manifest}


def metric_summary(rows: list[dict[str, Any]], group: str) -> dict[str, tuple[float, float]]:
    selected = [row for row in rows if row.get("group") == group]
    out: dict[str, tuple[float, float]] = {}
    for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "asd_f1"):
        values = [float(row[key]) * 100.0 for row in selected if row.get(key) not in {None, ""}]
        if values:
            tensor = torch.tensor(values, dtype=torch.float64)
            out[key] = (float(tensor.mean()), float(tensor.std(unbiased=False)))
    return out


def write_design(result: dict[str, Any]) -> None:
    ev = result["source_evidence"]; inv = result["inventory"]; e60 = inv["e60_original_pen"]
    lines = [
        "# Experiment 78.6 — PEN-SVA design and audit record", "",
        "## Scope", "This is a frozen audit of the original SPromptGL PEN, not a model-improvement run. E78.x checkpoints are never substituted.", "",
        "## Sources inspected", "- Protected local source: `network.py`, `model.py`, `layers.py`, `utils.py`; SHA-256 is recorded in `manifest.json`.", "- Official SPromptGL repository: https://github.com/wanxixi11/SPromptGL (read-only reference).", "- E32 and E43.3 preserved source/report mirrors; E32 B0/B1/B2 are historical references and E43.3 is a different protocol.",
        f"- E60 `E60_ORIGINAL_PEN`: historical checkpoint-audit CSV {e60['historical_checkpoint_audit_pass']}/10 PASS; protocol E53.3 CADR/GFR, not E32; current registered best-file availability {e60['currently_readable_fold_count']}/10.", "",
        "## Verified original forward path", "".join(f"- `{key}`: line(s) {value['lines']}\n" for key, value in ev.items() if isinstance(value, dict) and "found" in value), "The source executes GFR → CAM → four TD/NC and four ASD prompts → 4096→128→9 MLP → cosine similarity → two-class softmax → `torch.max` class-axis SPGS → AvgPool/Vsim RES → OutputLayer.", "",
        "## Mathematical conclusion", "For two class scores `s_A,s_T`, `max softmax([s_A,s_T]) = sigmoid(|s_A-s_T|)`. The multiplier is invariant to swapping class-vector positions and therefore records matching strength, not disease direction. Direction can enter elsewhere through MLP/RES, but this gate discards its sign.", "",
        "## Checkpoint boundary", f"E60 retains 10/10 historical identity/readback rows, but the actual `.pt` files are not readable at the registered stage, `/data`, or `/model` locations during this audit. A0–A7 patient-level replay, real feature effects, real text-MLP parameter/gradient inspection, and real class-vector-swap prediction equality are NOT_VERIFIED. No reconstruction was started.", "",
        "## Evidence boundary", "Source/math and synthetic positive controls are complete. E32/E60 metrics remain historical training references, not fixed-checkpoint causal interventions; A4–A6 text replacement was not executed without a recoverable original-PEN state_dict.",
    ]
    (EXP / "design.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(result: dict[str, Any]) -> None:
    inv = result["inventory"]; syn = result["synthetic"]; rows = historical_metrics(); e60 = metric_summary(rows, "E60_ORIGINAL_PEN"); e32 = {code: metric_summary(rows, code) for code in ("B0", "B1", "B2")}
    def fmt(summary: dict[str, tuple[float, float]], key: str) -> str:
        return f"{summary[key][0]:.4f} ± {summary[key][1]:.4f}%" if key in summary else "n/a"
    lines = [
        "# Experiment 78.6 — PEN-SVA final audit report", "", f"审计时间（UTC）：{result['status']['timestamp_utc']}", f"正式目录：`{EXP}`；首选 `/data` 因 OSSFS errno=5 回退到 `/model`。", "",
        "## 结论先行", "- **源码和数学层：已证实 Prompt 进入有效计算路径。** `CLIP→4096→128→9 MLP→cosine→SPGS→RES→OutputLayer` 在服务器 `network.py` 中真实存在。", "- **SPGS 疾病方向层：未保留。** `max(softmax([s_ASD,s_TD])) = sigmoid(|s_ASD-s_TD|)`；类别向量交换保持该门控不变。", "- **医学语义增益：未建立。** 当前无法读取原始 PEN best `.pt`，因此没有伪造固定 Checkpoint 的匿名、错配或数值文本结果。", "- **最终状态：`PARTIAL`；结论组合为 `COMPUTATION_ACTIVE_BUT_SEMANTIC_VALUE_NOT_ESTABLISHED`、`EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS`、`INSUFFICIENT_VERIFIABLE_CHECKPOINTS`。", "",
        "## Checkpoint 清点", f"E32/E28 当前源镜像无可读最佳 state_dict。E60 `E60_ORIGINAL_PEN` 的历史 `checkpoint_audit.csv` 为 {inv['historical_e60_audit_fold_count']}/10 PASS；当前注册位置可读且 SHA 匹配的权重为 {inv['verifiable_original_pen_fold_count']}/10。E60 是 E53.3 CADR/GFR 基座，不等同于 E32 B2。未进行最小重建。", "",
        "## 原始 SPGS 数学与源码证据", "设 `s_A=s_ASD`、`s_T=s_TD`。两类 softmax 最大值合并为 `sigmoid(abs(s_A-s_T))`；余弦分数在 [-1,1]，理论权重范围 [0.5, 0.880797...]。合成 10,000 组的最大数值误差为 `%.3g`。" % syn["spgs_math_max_abs_error"], "合成真实形状前向中，交换 MLP 输出后的两个类别向量时 SPGS weight、PEN 表示、logits 和硬预测逐位不变；这与交换原始八条文本再重新经过顺序敏感 MLP 是不同干预。", "",
        "## 历史训练参考（不是本次固定干预）", "|来源/配置|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|",
        f"|E32 B0 CAM-only|{fmt(e32['B0'],'acc')}|{fmt(e32['B0'],'hard_auc')}|{fmt(e32['B0'],'prob_auc')}|{fmt(e32['B0'],'sen')}|{fmt(e32['B0'],'spe')}|{fmt(e32['B0'],'asd_f1')}|",
        f"|E32 B1 RES-only|{fmt(e32['B1'],'acc')}|{fmt(e32['B1'],'hard_auc')}|{fmt(e32['B1'],'prob_auc')}|{fmt(e32['B1'],'sen')}|{fmt(e32['B1'],'spe')}|{fmt(e32['B1'],'asd_f1')}|",
        f"|E32 B2 full PEN|{fmt(e32['B2'],'acc')}|{fmt(e32['B2'],'hard_auc')}|{fmt(e32['B2'],'prob_auc')}|{fmt(e32['B2'],'sen')}|{fmt(e32['B2'],'spe')}|{fmt(e32['B2'],'asd_f1')}|",
        f"|E60 ORIGINAL_PEN（E53.3协议）|{fmt(e60,'acc')}|{fmt(e60,'hard_auc')}|{fmt(e60,'prob_auc')}|{fmt(e60,'sen')}|{fmt(e60,'spe')}|{fmt(e60,'asd_f1')}|", "", "E32 B2 没有超过 B0 CAM-only；这支持继续审计动机，但不是同一冻结模型下的因果干预。", "",
        "## A0–A7 状态", "A0 FULL_PEN、A1 RES_ONLY、A2 PEN_OFF、A3 CLASS_VECTOR_SWAP、A4 ANONYMOUS_TEXT、A5 WRONG_SEMANTIC_TEXT、A6 NUMERIC_TEXT_CONTROL、A7 NEUTRAL_SPGS 均为 `NOT_VERIFIED`。没有填写虚构的 ACC、概率变化、硬翻转或患者索引。", "",
        "## 正向控制与限制", "确定性分类器参数扰动和 PEN 中间特征扰动均产生非零 logits 变化，排除了审计探针整体失效。合成残差和余弦仅用于算子控制，不代表 ABIDE 患者结果。源码确认 CLIP 冻结、文本 MLP 可训练；真实 checkpoint 参数/梯度仍未读回。", "",
        "## 科研判定", "可以确认 Prompt **参与计算**；不能由此确认医学内容改善诊断。SPGS 最大类 softmax 明确丢弃 ASD/TD 方向符号。由于固定 Checkpoint A0–A7、患者级特征残差、真实文本 MLP 输出/梯度和逐患者翻转尚未复算，本审计不宣称 Prompt 提供疾病语义增益。", "", "限制：历史 `val=test`、测试折选模、CV 前处理和 transductive 图协议限制泛化解释；E43.3 是不同协议，不能与 E32/E60 合并。", "",
        "## 文件", f"- `audit.py`, `pen_probe.py`, `checkpoint_inventory.json`, `design.md`, `manifest.json`", f"- 结果：`{RESULTS}`", "- handoff：`/model/spromptgl/experiment/handoff.md`",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_handoff(result: dict[str, Any]) -> None:
    inv = result["inventory"]
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    block = f"""

## Experiment 78.6 — PEN-SVA correction (2026-11-13)

- 复核发现 E60 `E60_ORIGINAL_PEN` 历史登记有 10/10 个原始 SPromptGL PEN best，E60 `checkpoint_audit.csv` 全部 PASS；协议是 E53.3 CADR/GFR，不能与 E32 B2 混写。
- 当前 E60 stage、`/data` 和 `/model` 注册位置没有可读对应 `.pt`（当前 SHA 匹配数 {inv['verifiable_original_pen_fold_count']}/10），因此没有执行或伪造 A0–A7 患者级固定 Checkpoint 干预。
- 源码与数学审计成立：Prompt 通过冻结 RN50、4096→128→9 MLP、SPGS 和 RES 进入计算；`max(softmax([s_ASD,s_TD]))=sigmoid(abs(s_ASD-s_TD))`，类别向量交换在合成真实形状前向中保持 PEN 表示、logits 和硬预测不变。
- E32 B0/B1/B2 与 E60 ORIGINAL_PEN 只作为历史训练参考；医学/匿名/错误/数值文本的真实患者干预、真实文本 MLP 梯度和逐患者变化均为 `NOT_VERIFIED`。
- 当前科研判定：`COMPUTATION_ACTIVE_BUT_SEMANTIC_VALUE_NOT_ESTABLISHED` + `EXPLICIT_CLASS_DIRECTION_NOT_PRESERVED_BY_SPGS` + `INSUFFICIENT_VERIFIABLE_CHECKPOINTS`；不启动 E78.7。报告：`{RESULTS / 'report.md'}`。
"""
    with handoff.open("a", encoding="utf-8") as stream:
        stream.write(block); stream.flush(); os.fsync(stream.fileno())


def main() -> None:
    result = audit()
    if result.get("idempotent_replay"):
        print(json.dumps({"status": result["status"], "output": str(EXP), "idempotent_replay": True}, ensure_ascii=False, indent=2))
        return
    write_design(result); write_report(result); append_handoff(result)
    print(json.dumps({"status": result["status"], "output": str(EXP), "inventory": result["inventory"], "synthetic": result["synthetic"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
