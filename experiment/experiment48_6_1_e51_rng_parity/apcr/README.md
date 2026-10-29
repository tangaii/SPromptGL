# Experiment 48.6.1 — 修正随机轨迹并复现实验51 E1

本实验包含三个固定任务：R0重新运行Experiment51 E1原始LegacyFixedGFR、E1使用E51 E1基座和48.5完整APCR且α固定0.0022、E2相同但α可学习。总计30折。

R0直接使用Experiment51 E1冻结runner，从对应fold初始化锚点完整重训；不加入APCR、不改变原模型、优化器、checkpoint或随机数协议。R0结果用于确认历史E1逐折指标和best epoch可复现。

48.6.1只修复随机轨迹：删除48.6原有的逐epochTorch/CUDA seed重置，继承E51初始化seed并让基座Python/NumPy/Torch CPU/CUDA RNG连续推进。APCR初始化在独立fold RNG中执行后恢复基座状态；APCR额外计算使用单独可保存/恢复的RNG上下文，避免污染基座CAM dropout流。SRD、SCA、CRA、group=6路径、损失权重、候选数、攻击预算、优化器和超参数均保持48.6。

历史协议保留CV前全局预处理/RFE、871节点transductive图和`val=test` checkpoint选择，结果不是无偏外部泛化估计。服务器`/data`当前返回OSSFS EIO，新实验完整暂存于`/tmp/SPromptGL-E48.6.1/apcr`，manifest和handoff pending记录同步状态。

主要入口：`python run.py init`、`python run.py preflight`、5个worker运行E1/E2；R0使用相邻的`r0_audit.py`独立读回。最终结果见`results/report.md`。
