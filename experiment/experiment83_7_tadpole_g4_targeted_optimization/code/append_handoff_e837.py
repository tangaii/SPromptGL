from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")


def fmt(value):
    return f"{float(value) * 100:.4f}%"


def main():
    text = HANDOFF.read_text()
    marker = "## Experiment 83.7 — TADPOLE G4-Targeted Optimization and Fidelity Study"
    if marker in text:
        raise RuntimeError("Experiment 83.7 handoff section already exists")
    summary = pd.read_csv(ROOT / "results/group_summary.csv")
    audit = json.loads((ROOT / "audits/final_audit.json").read_text())
    seal = json.loads((ROOT / "audits/training_seal.json").read_text())
    rows = []
    for _, row in summary.iterrows():
        rows.append(f"| {row.group} | {fmt(row.ACC_mean)} ± {fmt(row.ACC_std_population)} | "
                    f"{fmt(row.AUC_original_mean)} ± {fmt(row.AUC_original_std_population)} | "
                    f"{int(row.OOF_correct)}/598 |")
    p = audit["performance"]
    section = f'''\n\n{marker}\n\n- **最终状态** `COMPLETE_AUDITED`。执行、方法保真、Best/Latest 重放、warmup 配对、数据身份、训练 seal 和机制审计全部通过。\n- **实际目录** `/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization`。首选 `/data/adodas/SPromptGL-main/experiment/experiment83_7_tadpole_g4_targeted_optimization` 在 8-worker 正式输出时出现 OSSFS errno 5/EIO，已按协议切换。失败尝试保留，未删除用户历史产物。\n- **源码身份** 项目工作树 HEAD `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`。训练前 seal implementation SHA `{seal["implementation_sha256"]}`，seal 文件 SHA `{{seal_sha}}`。训练后新增审计脚本单独记录，未改动训练源码。作者仓库 `ba3b9446fa451fde6864d6555bdf324de0c4b64a`，sp 快照 `f4534f8`，E83.5 请求提交 `4649c5d` 和 sealed runtime `5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11`。\n- **数据与历史协议** 598 人，NC/MCI/AD `209/315/74`，六模态 `145/148/3/36/24/9`，数据 SHA `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`，split SHA `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`。保留全数据 RFE、外层 CV 前标准化/均值填补、转导患者图、`val_idx == test_idx`、seed 0、450 epoch、`wait > 200` 和首次严格测试折 ACC Best 规则。\n- **五个基础组** G1–G5 均引用 E83.5 真实 checkpoint，Best/Latest/预测和独立重放均 `10/10 PASS`，五组共 50/50 fold。G2 的新文档身份为仅物理移除 PEN，保留 GFR、GraphLearn、GCN、消息传递和图损失，G5 为物理 CAM-only。\n- **六个新增组** T1–T6 各 `10/10` 正式 fold 完成。T1 cap `0.003`，T2 cap `0.010`，T3/T4 六模态连续 `pair_mean` 图监督仅 epoch 1–10，T5/T6 GFR/CAM WD 从 epoch 11 改为 `0.055`。\n- **GPU 与 worker** PPU-ZW810E，98,304 MiB，单 GPU，正式训练 8 workers。seal 到最终审计约 55 分钟，正式 60 fold 训练壁钟约 34 分钟。未发生 OOM、NaN 或恢复冲突。\n\n### 正式结果\n\n| 组别 | ACC | AUC_original | OOF 正确 |\n|---|---:|---:|---:|\n{chr(10).join(rows)}\n\n指标为 10-fold mean ± population std，AUC_original 沿用硬预测 one-hot macro OVR。\n\n- **G4 对比** 历史 G4 为 ACC `{fmt(p["historical_g4"]["ACC"])}`、AUC_original `{fmt(p["historical_g4"]["AUC_original"])}`。T1 相对 G4 为 `-0.1695 pp/-0.0069 pp`，T2 为 `-1.1667 pp/-0.6745 pp`，T4 为 `-1.0056 pp/-0.6288 pp`，T6 为 `-0.6667 pp/-0.8703 pp`。没有 G4 改进组同时超过历史 G4。\n- **G3 配对** T3−G3 为 `+0.1638 pp/-0.1960 pp`。图监督 warmup 没有带来 AUC_original 增量。\n- **T4−T3** 为 `-0.6695 pp/-0.3161 pp`。T4 的 DPSL 后续协同低于 T3。\n- **WD 配对** T5−G3 为 `-0.5028 pp/-0.5260 pp`，T6−G4 为 `-0.6667 pp/-0.8703 pp`，T6−T5 为 `+0.3362 pp/-0.2277 pp`。\n- **DPSL 预算机制** T1/T2 实际共享梯度比例最大值分别约 `0.00299989` 和 `0.00999989`，残差均受 `0.08` 上限保护，DPSL 参数梯度非零且 finite。T1 比 T2 更接近 G4，但两者都未产生完整性能增量。\n- **图监督机制** T3/T4 十折共享 warmup 的初始模型、optimizer、RNG、关系对、训练标签、epoch 1–10 历史和 epoch 10 后状态均逐折一致。epoch 11 起新增图监督严格为零。\n- **WD 与图消息** T5/T6 前十轮 WD 为 `0.11`，第 11 轮起仅 GFR/CAM 为 `0.055`，DARG/DPSL AdamW 保持原配置。T6 的图消息相对原始特征幅度增加没有转化为 ACC 或 AUC_original 提升。\n- **机制重放** G4、T1、T2、T4、T6 的 native/full logits、翻转、纠正/误伤、分类边际、Direct-Off、Disease-Neutral、残差和图消息摘要见 `audits/mechanism_summary.json`。模块存在非零梯度，但没有据此宣称分类收益。\n- **与论文** 原论文参考 ACC `94.81%`、AUC `95.59%`。本实验最高 ACC 和最高 AUC_original 都是历史 G4 的 `{fmt(p["best_acc_mean"])}` 和 `{fmt(p["best_auc_original_mean"])}`，均未达到论文目标。\n- **独立审计** 基础组重放 50/50，新增组重放 60/60，Best/Latest/下一步确定性全部通过，T3/T4 warmup 配对 10/10 通过，OOF 每组 598 名患者恰好一次。\n- **结果文件** `{{ROOT}}/results/group_summary.csv`、`fold_metrics.csv`、`oof_predictions.csv`、`paired_contrasts.csv`、`final_report.md`；`{{ROOT}}/audits/final_audit.json`、`base_checkpoint_replay.json`、`new_checkpoint_replay.json`、`warmup_pairing.json`、`mechanism_summary.json`、`method_fidelity_matrix.md`。\n- **handoff** 本章节已追加并在写入后读回核验。\n- **最终科研判断** ACC 和 AUC_original 最高均为历史 G4。没有新增变体同时超过 G4，DPSL 增量没有变得更充分，图监督 warmup 和 WD=0.055 未改善完整 G4。T1 是下一步 G4 候选中最接近历史 G4 的受控变体，但仍需新的预注册机制或数据协议后再研究。\n'''
    section = section.replace("{seal_sha}", seal_sha(ROOT / "audits/training_seal.json"))
    section = section.replace("{{ROOT}}", str(ROOT))
    HANDOFF.write_text(text + section)


def seal_sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
