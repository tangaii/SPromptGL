"""Publish E83.1 ACC and source-compatible hard AUC after independent replay."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pandas as pd

from fold_checkpoint import DATA_SHA, ROOT, SPLIT_SHA, atomic_write, fingerprint, write_json
from run_e831 import GROUPS, require_formal_training_gate

LABELS = dict(zip(GROUPS, ("E28-E0", "E43.1-E00", "E74.4-G2",
                          "E82.4.6.10-G1 TADPOLE Core-Method Adaptation")))
PAPER = {"method": "Original SPromptGL", "ACC_mean": .9481, "ACC_std": .0295,
         "AUC_mean": .9559, "AUC_std": .0257}
E83 = ROOT.parent / "experiment83_tadpole_four_group_comparison"


def write_csv(name, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, dict.fromkeys(key for row in records for key in row))
    writer.writeheader()
    writer.writerows(records)
    atomic_write(ROOT / "results" / name, lambda file: file.write(stream.getvalue().encode()))


def train_and_replay():
    evidence = json.loads((ROOT / "audits/checkpoint_replay.json").read_text())
    replay = {(item["group"], item["fold"]): item for item in evidence["records"]}
    if evidence["status"] != "PASS" or len(replay) != 40:
        raise RuntimeError("Results require 40 independent PASS replays")
    all_rows, metadata = [], {}
    for group in GROUPS:
        config = json.loads((ROOT / group / "config.json").read_text())
        if config["implementation_sha256"] != fingerprint():
            raise RuntimeError(f"Results belong to a different release for {group}")
        records = []
        for fold in range(1, 11):
            path = ROOT / group / "audits" / f"fold{fold:02d}.json"
            value = json.loads(path.read_text())
            check = replay[group, fold]
            if not (value["status"] == "TRAINED" and check["status"] == "PASS"
                    and value["checkpoint_sha256"] == check["checkpoint_sha256"]
                    and all(value[key] == check[key] for key in ("ACC", "AUC", "best_epoch"))):
                raise RuntimeError(f"Fold execution metadata or replay differ for {group} fold {fold}")
            records.append(value)
            keys = ("fold", "seed", "train_count", "test_count", "best_epoch", "ACC", "AUC")
            all_rows.append({"group": group, "method": LABELS[group], **{key: value[key] for key in keys}})
        metadata[group] = (config, records)
    for a, b in zip(metadata[GROUPS[2]][1], metadata[GROUPS[3]][1], strict=True):
        if a["warmup"] != b["warmup"]:
            raise RuntimeError(f"G3/G4 shared warmup differs in fold {a['fold']}")
    return all_rows, metadata, replay


def comparisons(rows, metadata):
    by_group = {group: [] for group in GROUPS}
    for row in rows:
        by_group[row["group"]].append(row)
    summary = []
    for group, fold_rows in by_group.items():
        flat = {"group": group, "method": LABELS[group], "folds": len(fold_rows)}
        for key in ("ACC", "AUC"):
            values = np.asarray([row[key] for row in fold_rows])
            flat[f"{key}_mean"], flat[f"{key}_std"] = float(values.mean()), float(values.std(ddof=0))
        summary.append(flat)
    paper = [{**PAPER, "delta_ACC_vs_paper": 0., "delta_AUC_vs_paper": 0.}]
    for row in summary:
        paper.append({**row, **{f"delta_{key}_vs_paper": row[f"{key}_mean"] - PAPER[f"{key}_mean"]
                               for key in ("ACC", "AUC")}})
    with (E83 / "metrics" / "all_group_summary.csv").open() as stream:
        e83 = {row["group"]: row for row in csv.DictReader(stream)}
    old_new = []
    for row in summary:
        group = row["group"]
        old = {"ACC": float(e83[group]["ACC_mean"]), "AUC": float(e83[group]["AUC_original_mean"])}
        old_new.append({"group": group, "method": LABELS[group],
                        **{f"E83_{name}": value for name, value in old.items()},
                        **{f"E83_1_{name}": row[f"{name}_mean"] for name in old},
                        **{f"delta_{name}": row[f"{name}_mean"] - value for name, value in old.items()}})
    contrasts = []
    for first, second in ((1, 0), (2, 1), (3, 2), (3, 0)):
        a, b = GROUPS[first], GROUPS[second]
        label = f"G{first + 1}-G{second + 1}"
        records = []
        for ra, rb in zip(metadata[a][1], metadata[b][1], strict=True):
            pa, pb = (pd.read_csv(r["prediction"]) for r in (ra, rb))
            np.testing.assert_array_equal(pa["sample_index"], pb["sample_index"])
            np.testing.assert_array_equal(pa["true_label"], pb["true_label"])
            correct_a, correct_b = (p["true_label"].eq(p["pred_label"]) for p in (pa, pb))
            records.append({"comparison": label, "fold": ra["fold"],
                            "delta_ACC": ra["ACC"] - rb["ACC"], "delta_AUC": ra["AUC"] - rb["AUC"],
                            "incorrect_to_correct": int((correct_a & ~correct_b).sum()),
                            "correct_to_incorrect": int((correct_b & ~correct_a).sum()),
                            "net_correct": int(correct_a.sum() - correct_b.sum())})
        contrasts.extend(records)
    return summary, paper, old_new, contrasts


def render(paper, old_new, contrasts, workers, method_status, failed_mechanisms):
    lines = ["# 实验 83.1 正式复现报告", "",
             f"单卡实际 worker 数为 {workers}。四组各完成 10 折，40 个 Best 均通过独立重放。",
             "ACC 和 AUC 为逐折均值 ± 总体标准差。AUC 按原始源码对 argmax 硬预测做 one-hot 后计算 macro OVR。",
             "", "| 方法 | ACC | AUC | ΔACC 对论文 | ΔAUC 对论文 |",
             "|---|---:|---:|---:|---:|"]
    for row in paper:
        acc = f"{100 * row['ACC_mean']:.2f} ± {100 * row['ACC_std']:.2f}%"
        auc = f"{100 * row['AUC_mean']:.2f} ± {100 * row['AUC_std']:.2f}%"
        lines.append(f"| {row['method']} | {acc} | {auc} | {100 * row['delta_ACC_vs_paper']:+.2f} pp"
                     f" | {100 * row['delta_AUC_vs_paper']:+.2f} pp |")
    lines.extend(["", "论文对照取自 Table 2。[SPromptGL, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)",
                  "", "## 同折变化", "",
                  "| 对比 | ΔACC | ΔAUC | ACC 改善 / 下降 / 持平 | AUC 改善 / 下降 / 持平 | 净正确人数 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label in dict.fromkeys(row["comparison"] for row in contrasts):
        sample = [row for row in contrasts if row["comparison"] == label]
        acc = np.asarray([row["delta_ACC"] for row in sample])
        auc = np.asarray([row["delta_AUC"] for row in sample])
        count = lambda v: f"{int((v > 0).sum())} / {int((v < 0).sum())} / {int((v == 0).sum())}"
        net = sum(row["net_correct"] for row in sample)
        lines.append(f"| {label} | {100 * acc.mean():+.2f} pp | {100 * auc.mean():+.2f} pp"
                     f" | {count(acc)} | {count(auc)} | {net:+d} |")
    lines.extend(["", "逐折差异与正确性变化见 [fold_contrasts.csv](results/fold_contrasts.csv)。",
                  "G3 相比 G2 同时改变了原始模态消息融合、关系监督、warmup、优化器与权重衰减。G3−G2 是历史方法整体差异。",
                  "G4 与 G3 每折共有模型、优化器、关系对和 RNG 的 warmup 档案一致。",
                  "", "## E83 对照", "",
                  "E83 保留原始执行结果，状态为 `COMPLETE_EXECUTION / METHOD_FIDELITY_FAILED`。下表 AUC 取 E83 的硬预测 AUC。",
                  "", "| 组 | E83 ACC | E83 AUC | E83.1−E83 ACC | E83.1−E83 AUC |",
                  "|---|---:|---:|---:|---:|"])
    for row in old_new:
        lines.append(f"| {row['method']} | {row['E83_ACC']:.2%} | {row['E83_AUC']:.2%}"
                     f" | {100 * row['delta_ACC']:+.2f} pp | {100 * row['delta_AUC']:+.2f} pp |")
    lines.extend(["", "## 方法与限制", "",
                  "论文协议在外层 CV 前完成全数据 RFE、均值填补与标准化，患者图使用全部特征，外层测试折按历史规则参与 Best 选择。"
                  "这些步骤可能使估计偏乐观，比较限定在该历史协议内。",
                  "六个 TADPOLE 模态均使用历史非 CORR 规则。NC、MCI、AD 使用三个高斯类条件分布。"
                  "SCMR 对原生头前两类别形成同一有向对比，谱证据、Jacobian 和引导共用该方向。",
                  "八条 CLIP RN50 文本描述通用脑结构与功能异常。MCI 与 AD 之间的医学语义区分有限。",
                  "G4 保留 SCMR 谱与证据、校准、残差和梯度更新机制，病人集合使用 TADPOLE 原始十折。"
                  "历史 ABIDE Fold8 人工患者加入未迁移。",
                  "", f"执行审计 PASS，方法保真审计 {method_status}，Checkpoint 独立重放 PASS。"
                  " [保真矩阵](audits/method_fidelity_matrix.md) · [最终审计](audits/final_audit.json)"])
    if failed_mechanisms:
        lines.extend(["", "以下 Best 检查点未达到实际输出参与验证，列为方法审计失败。", ""])
        lines.extend(f"- {item['group']} fold {item['fold']:02d} {item['mechanism']}" for item in failed_mechanisms)
    return "\n".join(lines) + "\n"


def main():
    require_formal_training_gate()
    rows, metadata, replay = train_and_replay()
    summary, paper, old_new, contrasts = comparisons(rows, metadata)
    workers = max(int(config["actual_workers"]) for config, _ in metadata.values())
    failed_mechanisms = []
    for (group, fold), item in replay.items():
        expected = ("darg", "dpsl") if group == GROUPS[3] else ("darg",) if group == GROUPS[2] else ()
        for name in expected:
            if not item["mechanism"].get(name, {}).get("active"):
                failed_mechanisms.append({"group": group, "fold": fold, "mechanism": name})
    method_status = "FAIL" if failed_mechanisms else "PASS"
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for name, values in (("all_fold_metrics", rows), ("group_summary", summary), ("paper_comparison", paper),
                         ("e83_vs_e83_1", old_new), ("fold_contrasts", contrasts)):
        write_csv(name + ".csv", values)
    for group, (config, records) in metadata.items():
        write_json(ROOT / group / "audit.json", {
            "group": group, "training": "COMPLETE", "execution_audit": "PASS",
            "method_fidelity_audit": "FAIL" if any(item["group"] == group for item in failed_mechanisms) else "PASS",
            "checkpoint_replay_audit": "PASS",
            "replayed_best": 10, "actual_workers": config["actual_workers"],
            "folds": [{"fold": row["fold"], "best_epoch": row["best_epoch"],
                       "checkpoint_sha256": row["checkpoint_sha256"]} for row in records],
        })
    write_json(ROOT / "audits/final_audit.json", {
        "experiment": "Experiment 83.1", "training_status": "COMPLETE", "formal_folds": len(rows),
        "EXECUTION_AUDIT": "PASS", "METHOD_FIDELITY_AUDIT": method_status, "CHECKPOINT_REPLAY_AUDIT": "PASS",
        "checkpoint_replay_passes": 40, "actual_workers": workers, "implementation_sha256": fingerprint(),
        "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA, "failed_mechanisms": failed_mechanisms,
    })
    report = render(paper, old_new, contrasts, workers, method_status, failed_mechanisms)
    atomic_write(ROOT / "final_report.md", lambda stream: stream.write(report.encode()))
    return int(bool(failed_mechanisms))


if __name__ == "__main__":
    raise SystemExit(main())
