"""Create truthful final artifacts when the frozen method preflight blocks training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
HIST_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
E837_ROOT = Path("/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization")
GROUPS = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
}
PAIRS = [("H1", "X1"), ("H2", "X2"), ("H3", "X3"), ("H4", "X4"),
         ("H4", "X5"), ("H4", "X6"), ("X5", "X6")]


def read_json(path: Path):
    return json.loads(path.read_text())


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    audits = ROOT / "audits"
    results = ROOT / "results"
    preflight = read_json(audits / "preflight_start.json")
    seal = read_json(audits / "training_seal.json")
    snapshot = read_json(audits / "historical_checkpoint_snapshot.json")
    replay = read_json(E837_ROOT / "audits/base_checkpoint_replay.json")
    if preflight.get("status") != "FAIL" or preflight.get("formal_training_allowed") is not False:
        raise RuntimeError("Expected the sealed six-modality effective-gradient preflight to block training")
    if preflight.get("training_seal_sha256") != sha_file(audits / "training_seal.json"):
        raise RuntimeError("Preflight and source seal do not match")
    if replay.get("status") != "PASS" or replay.get("folds") != 50 or len(replay["records"]) != 50:
        raise RuntimeError("The independent historical checkpoint replay is incomplete")
    if snapshot.get("status") != "PASS" or len(snapshot["groups"]) != 5:
        raise RuntimeError("The historical reference snapshot is incomplete")

    checkpoint_checks = []
    for group in snapshot["groups"]:
        records = [row for row in replay["records"] if row["group"] == group["group"]]
        if len(records) != 10:
            raise RuntimeError(f"Historical replay count differs for {group['group']}")
        for row in records:
            for kind in ("best", "latest"):
                path = Path(row[f"{kind}_path"])
                actual = sha_file(path)
                expected = row[f"{kind}_sha256"]
                checkpoint_checks.append({"group": row["group"], "fold": row["fold"],
                                          "kind": kind, "sha256": actual,
                                          "matches_replay": actual == expected})
                if actual != expected:
                    raise RuntimeError(f"Historical {kind} checkpoint changed: {path}")

    new_checkpoints = [path for group in (ROOT / "groups").iterdir()
                       for kind in ("best_checkpoints", "latest_checkpoints")
                       for path in (group / kind).glob("*.pt")]
    if new_checkpoints or (audits / "resource_probe.json").exists():
        raise RuntimeError("A formal checkpoint or resource probe exists despite the failed gate")

    records_by_group = {group: [row for row in replay["records"] if row["group"] == group]
                        for group in GROUPS}
    summary_rows = []
    fold_rows = []
    for group, source_name in GROUPS.items():
        records = records_by_group[group]
        reference = next(row for row in snapshot["groups"] if row["group"] == group)
        summary_rows.append({
            "group": group, "source_group": source_name, "status": "HISTORICAL_REFERENCE_VERIFIED",
            "folds_planned": 10, "folds_completed": 10,
            "ACC_mean": mean(row["ACC"] for row in records),
            "ACC_std_population": pstdev(row["ACC"] for row in records),
            "AUC_original_mean": mean(row["AUC_original"] for row in records),
            "AUC_original_std_population": pstdev(row["AUC_original"] for row in records),
            "AUC_probability_mean": mean(row["AUC_probability"] for row in records),
            "AUC_probability_std_population": pstdev(row["AUC_probability"] for row in records),
            "correct_total": reference["correct_total"], "source": "E83.5 independent Best/Latest replay",
        })
        for row in records:
            fold_rows.append({"group": group, "fold": row["fold"], "status": "HISTORICAL_REPLAY_PASS",
                              "ACC": row["ACC"], "AUC_original": row["AUC_original"],
                              "AUC_probability": row["AUC_probability"], "correct": row["correct"],
                              "best_sha256": row["best_sha256"], "latest_sha256": row["latest_sha256"]})

    blocked_groups = []
    for number in range(1, 7):
        group = f"X{number}"
        blocked_groups.append(group)
        summary_rows.append({"group": group, "source_group": "", "status": "NOT_RUN_PREFLIGHT_FAIL",
                             "folds_planned": 10, "folds_completed": 0,
                             "ACC_mean": "", "ACC_std_population": "",
                             "AUC_original_mean": "", "AUC_original_std_population": "",
                             "AUC_probability_mean": "", "AUC_probability_std_population": "",
                             "correct_total": "", "source": "No formal training or predictions"})
        for fold in range(1, 11):
            fold_rows.append({"group": group, "fold": fold, "status": "NOT_RUN_PREFLIGHT_FAIL",
                              "ACC": "", "AUC_original": "", "AUC_probability": "", "correct": "",
                              "best_sha256": "", "latest_sha256": ""})

    summary_fields = list(summary_rows[0])
    with (results / "group_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    fold_fields = list(fold_rows[0])
    with (results / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(fold_rows)
    pair_rows = [{"reference": left, "candidate": right, "status": "NOT_COMPUTED_NO_NEW_OOF",
                  "ACC_delta": "", "AUC_original_delta": "", "corrected": "", "damaged": "", "net": ""}
                 for left, right in PAIRS]
    with (results / "paired_contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    with (results / "group_execution_status.csv").open("w", newline="") as handle:
        fields = ["group", "planned_folds", "completed_folds", "status", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in blocked_groups:
            writer.writerow({"group": group, "planned_folds": 10, "completed_folds": 0,
                             "status": "BLOCKED", "reason": "Six-modality effective-gradient preflight failed"})

    aloss = preflight["checks"]["paper_aloss"]
    pair_mean = preflight["checks"]["class_balanced_pair_mean"]
    warmup = preflight["checks"]["warmup_pairing"]
    dpsl = preflight["checks"]["dpsl_budget"]
    mechanism = {
        "status": "PREFLIGHT_ONLY_NO_TRAINED_MECHANISM",
        "formal_training_started": False,
        "paper_aloss": {
            "status": aloss["status"], "fold": 1,
            "effective_gradient_threshold": aloss["effective_gradient_threshold"],
            "effective_gradient_counts": aloss["effective_gradient_counts"],
            "groups": aloss["groups"],
            "interpretation": "M1 has no actual off-diagonal graph edge, so its normalized adjacency is invariant to GraphLearn projection parameters. The observed 7.5e-14 norm is floating-point residue.",
            "unanswered_training_questions": [
                "Per-modality graph changes after formal optimization",
                "Class-wise accuracy effect for NC, MCI and AD",
                "DARG relation retention after formal optimization",
                "X4 native-versus-DPSL-full patient corrections",
            ],
        },
        "class_balanced_pair_mean_preflight": pair_mean,
        "warmup_pairing": warmup,
        "dpsl_budget_one_step_preflight": dpsl,
    }
    write_json(audits / "mechanism_summary.json", mechanism)

    checkpoint_replay = {
        "status": "HISTORICAL_PASS_NEW_NOT_RUN",
        "historical_groups": "H1-H5 independent Best/Latest/next-step replay passed 50/50 folds",
        "historical_checkpoint_files_rehashed": len(checkpoint_checks),
        "historical_checkpoint_hash_mismatches": 0,
        "new_groups": {group: {"folds_completed": 0, "best": "NOT_CREATED", "latest": "NOT_CREATED",
                               "replay": "NOT_RUN_PREFLIGHT_FAIL"} for group in blocked_groups},
        "resource_probe": "NOT_RUN because method preflight failed",
        "historical_hash_checks": checkpoint_checks,
    }
    write_json(audits / "checkpoint_replay.json", checkpoint_replay)

    final_audit = {
        "experiment": "Experiment 83.8",
        "status": "BLOCKED_PREFLIGHT",
        "actual_root": str(ROOT),
        "preferred_root_eio_fallback": True,
        "finalization_script_sha256": sha_file(Path(__file__)),
        "training_seal_sha256": sha_file(audits / "training_seal.json"),
        "preflight_sha256": sha_file(audits / "preflight_start.json"),
        "data_sha256": preflight["data_sha256"], "split_sha256": preflight["split_sha256"],
        "EXECUTION_AUDIT": {"status": "BLOCKED", "formal_folds_completed": 0,
                            "formal_folds_planned": 60, "reason": "The required six-modality effective-gradient preflight failed."},
        "METHOD_FIDELITY": {"status": "FAIL", "formula_identity": "PASS",
                            "six_modality_effective_gradient": aloss["status"],
                            "reason": aloss.get("failure_reason")},
        "CHECKPOINT_REPLAY": {"status": "PASS_HISTORICAL_ONLY", "historical_folds": 50,
                               "historical_hashes_rechecked": len(checkpoint_checks),
                               "historical_hash_mismatches": 0, "new_group_folds": 0},
        "LOSS_FORMULA_AUDIT": {"status": "PASS_FORMULAS_FAIL_GRADIENT_COVERAGE",
                               "three_object_identity_distinguished": True,
                               "all_six_gradient_branches_effective": False},
        "PERFORMANCE_GO_NO_GO": {"status": "NO_GO_NOT_RUN", "new_group_metrics": None,
                                  "paper_threshold_comparison": "Not applicable to untrained groups"},
        "warmup_pairing": warmup,
        "dpsl_budget_preflight": dpsl,
        "historical_checkpoint_files": len(checkpoint_checks),
        "historical_checkpoint_hash_mismatches": 0,
        "new_checkpoint_files": 0,
        "resource_probe": "NOT_RUN",
    }
    write_json(audits / "final_audit.json", final_audit)

    table = ["| 组 | 状态 | ACC | AUC_original | AUC_probability | OOF正确 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row["group"].startswith("X"):
            table.append(f"| {row['group']} | NOT RUN | - | - | - | - |")
        else:
            table.append(f"| {row['group']} | 历史核验 | {row['ACC_mean']:.4%} ± {row['ACC_std_population']:.4%} "
                         f"| {row['AUC_original_mean']:.4%} ± {row['AUC_original_std_population']:.4%} "
                         f"| {row['AUC_probability_mean']:.4%} ± {row['AUC_probability_std_population']:.4%} "
                         f"| {row['correct_total']}/598 |")
    aloss_rows = aloss["groups"]
    m1 = aloss_rows["X1"]["per_modality_graphlearn_p_weight_gradient_norm"][0]
    m2_m6 = [aloss_rows["X1"]["per_modality_graphlearn_p_weight_gradient_norm"][i] for i in range(1, 6)]
    counts = pair_mean["group_counts"]
    report = f'''# Experiment 83.8 Final Report

## A. STATUS

`BLOCKED_PREFLIGHT`。本次没有启动任何正式训练。六组正式 fold 为 `0/60`，性能结果、OOF 预测和 Native/Full 患者机制均不存在。失败项是 X1-X4 要求的六模态有效 ALoss 梯度覆盖。

## B. GPU 与环境

环境为单张 PPU-ZW810E，显存 98,304 MiB，worker 上限 8，seed 0。首选目录创建遇到 OSSFS EIO / errno 5，实际目录为 `{ROOT}`。因预检失败，八 worker 资源探针和正式训练均未运行。

## C. 原论文及作者源码核查

论文 Eq.(1)、Eq.(7)，作者仓库 `ba3b9446fa451fde6864d6555bdf324de0c4b64a`，项目提交 E83.5 `4649c5dddf166a4aa957c16c2f667a5cea485cdc`、E83.6 `c2f9919bee41706cd56b3caac380e6b9a5f5dc46`、E83.7 `a08b8e03d33adf2a35eae2f2846f1d0a603c78d0` 均已核查并在本目录封存。作者 `network.py` SHA 为 `d0ac59ead2fecdf517845e2c389d361ef9924abff29707386f847e1fbaa22166`，`model.py` SHA 为 `254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda`。E83.6 runner/fidelity model SHA 为 `1fbd47ff3a00538bcc8a76faea01d38d802dfaae72188154b762c061219da126` / `d8d90f2ff6f267602aec2bfdafa67f5b0ac59a522f46f5fccaf813e24b388fce`，E83.7 对应 SHA 为 `e3dd10ac91ed699b9477f26dbe313354d34f092f6f81e5dd1189165a78af7d79` / `21060b6d9e43e733b970569c1b1ce3c30ac42cd15f176f5d56d92c042724e918`。原作者 GFR 写死四个模态分支，TADPOLE 实验运行六模态。作者 `network.py` 返回四分支平均的未平方 `torch.linalg.norm`，作者 `model.py` 的 `simple-2` 仅对 `cls_loss + all_graph_loss` 反向传播，ALoss 在 backward 后打印。

## D. ALoss 数学定义与源码差异

论文对象为平方 Frobenius 项 `(1/6) sum_m ||Ahat - A_m||_F^2`，作者源码对象是未平方矩阵范数，E83.6 对象是连续余弦相似度上全部训练患者对的 MSE。三者不同。E83.8 预注册对象使用六个实际 GFR 阈值图，对 `A_m + I` 和仅由训练标签形成的 `Ahat + I` 做全 598 节点对称度归一化，然后只比较训练节点方块，包含对角线，系数 1。测试标签不参与目标。

预检 fold 1 中，ALoss 为 X1/X2 `288.405`、X3/X4 `297.405`，分类损失分别约 X1 `1.206`、X2 `1.202`、X3/X4 `1.395`，比例约 `239x` 和 `213x`。标签目标与梯度对测试标签置换不变，训练标签置换会改变目标。

## E. TADPOLE 数据协议

598 名患者，NC/MCI/AD 为 209/315/74，编码 0/1/2，六模态维度 145/148/3/36/24/9，总维度 365。数据 SHA `{preflight["data_sha256"]}`，十折 SHA `{preflight["split_sha256"]}`。全数据 RFE、历史均值填补、外层 CV 前标准化、转导图、`val_idx == test_idx`、首次严格测试折 ACC 提升保存 Best、450 epoch、`wait > 200`、每折 seed 0 和不逐 epoch 重置 RNG 均保留。

## F. 六组方法设计及保真

X1/X2 为对应 G1/G2 加论文式 ALoss，X3 为 DARG Positive-DCR 加 ALoss，X4 为完整 G4 DARG+DPSL 加 ALoss，X5 为 class-balanced pair mean，X6 为增加 detached entropy 权重的 class-balanced pair mean。代码和配置已封印。禁用新增损失时，fold 1 的 H1-H4 原生前向与历史预测一致。X4 的 ALoss 已在 fold 1 预检中证实加进 `historical_total` 并进入 DPSL 优化步骤。

## G. Warmup 配对结果

fold 1 的 X3/X4 初始化、optimizer、RNG、DARG 关系对、训练标签和十轮 ALoss warmup 状态精确配对通过。fold 1 的 X5/X6 对应状态及十轮无新增监督 warmup 精确配对通过，并与历史 H4 warmup 终点相符。fold 1 的 H1/H2/H4 历史初始状态哈希缺失，因此 X1/H1、X2/H2、X5/H4、X6/H4 标记 `PARTIAL_MATCH`，没有声称完全配对。

## H. 正式训练完成情况

X1 至 X6 均为 `0/10`。未创建 Best、Latest、训练日志或 OOF 文件。没有因阶段性性能停止训练，训练未启动的原因是预检门禁失败。

## I. 六组 ACC/AUC 完整表

新组没有可报告的 ACC、AUC_original 或 AUC_probability。`-` 表示未训练，不表示零分。

| 组 | 状态 | ACC | AUC_original | AUC_probability |
|---|---|---:|---:|---:|
| X1 | 未训练 | - | - | - |
| X2 | 未训练 | - | - | - |
| X3 | 未训练 | - | - | - |
| X4 | 未训练 | - | - | - |
| X5 | 未训练 | - | - | - |
| X6 | 未训练 | - | - | - |

## J. 与历史 G1-G4 的对照

历史 H1-H5 checkpoint 已独立重放和核验。下表为历史参考值，不是 E83.8 新训练结果。

{chr(10).join(table)}

X1-H1、X2-H2、X3-H3、X4-H4、X5-H4、X6-H4、X6-X5 均无新 OOF，配对差异未计算。历史参考组没有重训或重新命名。
历史 H4 的 NC/MCI/AD 正确人数为 `205/287/64`。

## K. 与 E83.6、E83.7 图监督实验的对照

E83.6-D2 为 ACC `92.4746% ± 2.51%`、AUC_original `93.60% ± 2.61%`，采用系数 1.0 的全部训练有序患者对连续相似度监督。E83.7-T4 为 ACC `91.9689% ± 3.3345%`、AUC_original `93.4678% ± 3.1796%`，只在前十轮 warmup 开启图监督。本次 X5/X6 预注册系数 0.1、epoch 11 起加入，且采用六组均衡患者对，目标和启用时段都不同。E83.8 因未训练，不能做性能优劣结论。

## L. ALoss 六模态梯度及图学习机制

预检 fold 1 中 M1 的实际阈值图没有非对角边，只有 598 个对角项。加单位阵并归一化后为恒等图，标签平方误差对 GraphLearn 投影参数不变。M1 `p.weight` 梯度范数为 `{m1:.3g}`，是浮点残差。M2-M6 梯度范数为 `{", ".join(f"{value:.3g}" for value in m2_m6)}`，五个分支有效。X1 的六模态图非对角非零项数为 `{aloss_rows["X1"]["per_modality_actual_graph_offdiagonal_nonzero_entries"]}`。冻结阈值为 0.9，不能通过改阈值或换成连续 pair loss 来伪称论文式 ALoss 全六模态有效，因此六分支门禁失败。未训练，图是否变化、分类是否改善、DARG 是否受损均不能回答。

## M. 优化版 pair_mean 机制

X5/X6 fold 1 的六个唯一训练患者对组计数通过核验。NC-NC `{counts["class_0-class_0"]}`，MCI-MCI `{counts["class_1-class_1"]}`，AD-AD `{counts["class_2-class_2"]}`，NC-MCI `{counts["class_0-class_1"]}`，NC-AD `{counts["class_0-class_2"]}`，MCI-AD `{counts["class_1-class_2"]}`，共 `{pair_mean["unique_pairs"]}` 对。X6 权重 detach，测试标签置换不改变目标。初始模型预检的组均值权重为 NC-NC `0.8582`、NC-MCI `0.8695`、NC-AD `0.8827`、MCI-MCI `0.8808`、MCI-AD `0.8941`、AD-AD `0.9073`，AD-AD 最高。这些值只描述预检初始化，正式训练期权重分布未知。

## N. G4 Native/Full 患者级纠错分析

X4-X6 没有训练，Native/Full 预测、纠正人数、误伤人数及相对历史 G4 净变化均不可得。历史 G4 Native/Full 均正确 556 人，NC/MCI/AD 为 `205/287/64`。E83.7-T4 的历史机制显示，DPSL Full 相比自身 Native 直接纠正 3 名 MCI 且未误伤，但相对历史 G4 总体纠正 24、误伤 30、净少 6。直接纠错数不能替代完整模型相对基准的变化。

## O. 与原论文 94.81%/95.59% 对照

原论文参考为 ACC `94.81% ± 2.95%`、AUC `95.59% ± 2.57%`。未训练的 X1-X6 不作性能对照。历史 H4 的 ACC `92.9746%`、AUC_original `94.0967%` 低于该参考，但这是历史结果。

## P. Best/Latest 独立审计

H1-H5 的 Best/Latest 与下一步确定性历史重放 `50/50 PASS`，最终重新计算 100 个历史 checkpoint SHA，0 个不匹配。新组没有 checkpoint，Best/Latest 重放未运行。审计分类为 `EXECUTION_AUDIT=BLOCKED`、`METHOD_FIDELITY=FAIL`（公式身份 PASS，六模态梯度覆盖 FAIL）、`CHECKPOINT_REPLAY=PASS_HISTORICAL_ONLY`、`LOSS_FORMULA_AUDIT=PASS_FORMULAS_FAIL_GRADIENT_COVERAGE`、`PERFORMANCE_GO_NO_GO=NO_GO_NOT_RUN`。

## Q. 文件清单

源码和配置位于 `{ROOT}/code`、`configs`，锁定输入位于 `inputs`，论文与历史源码快照位于 `sources`。关键审计为 `audits/preflight_start.json`、`loss_fidelity.json`、`historical_checkpoint_snapshot.json`、`checkpoint_replay.json`、`mechanism_summary.json`、`final_audit.json`、`training_seal.json` 和审计生成器 `audits/finalize_blocked_e838.py`。结果状态与历史参考见 `results/group_summary.csv`、`fold_metrics.csv`、`paired_contrasts.csv`、`group_execution_status.csv`。新组没有 OOF 预测文件。

## R. handoff 更新确认

handoff 已追加唯一的 Experiment 83.8 章节，并在追加后读回核对。本实验目录回退原因为首选 OSSFS EIO。必要恢复入口为

```bash
cd {ROOT}
PYTHONDONTWRITEBYTECODE=1 python -B code/preflight_e838.py
```

当前预检会以失败状态阻止训练。要恢复正式实验，需先批准并预注册能让 M1 获得有效梯度的方法，同时重新进行方法审查、封印和预检。不得在原封印上直接改阈值或目标。

## S. 最终科研结论

本次没有性能实验结论。预检发现冻结阈值下 M1 图是恒等图，论文式标签监督对该 GraphLearn 分支没有有效梯度。ALoss 在其余五个分支具备有效梯度，公式实现、训练标签隔离、DPSL 预算及 pair_mean 数学检查通过，但不能据此声称完整六模态 GFR 获得了标签图监督。为了保持论文目标和历史图阈值，本次正式训练按门禁停止。论文式 ALoss 是否改善 G4，以及 X5/X6 是否能保护 G4 主干同时改善 DARG+DPSL，均未得到回答。
'''
    (results / "final_report.md").write_text(report)
    print(json.dumps({"status": final_audit["status"], "historical_hashes": len(checkpoint_checks),
                      "new_folds": 0, "report": str(results / "final_report.md")}, indent=2))


if __name__ == "__main__":
    main()
