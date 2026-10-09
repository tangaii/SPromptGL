from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from common import EXP_DIR, PROJECT_ROOT, RESULT_DIR, RAW_DIR, task_status_path


HANDOFF = PROJECT_ROOT / "handoff.md"
START = "<!-- MASPE EXPERIMENT 1 START -->"
END = "<!-- MASPE EXPERIMENT 1 END -->"


def current_snapshot() -> dict:
    task_records = []
    fold_records = []
    for path in sorted(RAW_DIR.glob("P*_seed*_task.json")):
        try:
            task_records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    for path in sorted(RAW_DIR.glob("P*_seed*_fold*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("status") == "COMPLETE":
            fold_records.append(value)
    complete_tasks = sum(record.get("status") == "COMPLETE" for record in task_records)
    lines = [
        "## 实验一：模态感知语义提示增强（MASPE）",
        "",
        f"- 状态：`{'COMPLETE' if complete_tasks == 12 and len(fold_records) == 120 else 'RUNNING' if task_records else 'AUDIT/SMOKE'}`",
        f"- 实验目录：`{EXP_DIR}`",
        f"- 任务完成：{complete_tasks}/12 个 config×seed；完成折数：{len(fold_records)}/120。",
        "- 设计：P0 原始通用提示；P1 模态感知提示；P2 固定模态错配；P3 数量匹配通用提示。",
        "- 评估：3 seeds × 10-fold stratified CV，outer-train 内部 10% validation；Transductive 特征传播；只训练标签进入监督图和分类损失。",
        "- 数据输入固定为根目录 ABIDE `processed_standard_data.csv` 与 `modal_feat_dict.npy`；需注意全体标准化与预先 RFE 的历史预处理范围。",
        "- 数据 SHA256：CSV `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`；模态字典 `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`。",
        "- 当前已完成折的指标只用于状态跟踪，不用于修改提示或超参数。正式汇总以 `results/report.md` 为准。",
    ]
    summary_path = RESULT_DIR / "summary.csv"
    if summary_path.is_file():
        try:
            summary = pd.read_csv(summary_path)
            lines.extend(["", "| Config | OOF BA (mean±SD across seeds) | OOF prob-AUC (mean±SD across seeds) |", "|---|---:|---:|"])
            for _, row in summary.iterrows():
                lines.append(
                    f"| {row['config']} | {row.get('balanced_accuracy_mean', float('nan')):.4f} ± {row.get('balanced_accuracy_std', float('nan')):.4f} "
                    f"| {row.get('probability_auc_asd_positive_mean', float('nan')):.4f} ± {row.get('probability_auc_asd_positive_std', float('nan')):.4f} |"
                )
        except Exception:
            pass
    if task_records:
        lines.extend(["", "任务状态："])
        for record in sorted(task_records, key=lambda item: (item.get("config", ""), item.get("seed", 0))):
            lines.append(
                f"- {record.get('config')} seed={record.get('seed')}: "
                f"{record.get('status')}，{record.get('completed_folds', 0)}/{record.get('expected_folds', 10)} folds"
            )
    lines.extend([
        "",
        f"- 复现命令：`cd {EXP_DIR} && python src/prepare.py && bash resume.sh`。",
        "- 该实验不自动推送 GitHub。",
    ])
    return {"task_records": task_records, "fold_records": fold_records, "text": "\n".join(lines)}


def update_handoff() -> None:
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    lock_path = HANDOFF.with_name(HANDOFF.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
        block = current_snapshot()["text"]
        if START in old and END in old:
            prefix, tail = old.split(START, 1)
            _, suffix = tail.split(END, 1)
            new_text = prefix.rstrip() + "\n\n" + START + "\n" + block + "\n" + END + suffix
        else:
            new_text = old.rstrip()
            if new_text:
                new_text += "\n\n"
            new_text += START + "\n" + block + "\n" + END + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix="handoff.md.", suffix=".tmp", dir=HANDOFF.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                target.write(new_text)
                target.flush()
                os.fsync(target.fileno())
            os.replace(tmp_name, HANDOFF)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    update_handoff()
