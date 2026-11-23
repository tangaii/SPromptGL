# DPSL 正式实验基座

DPSL 封存了 ABIDE A 的 E82.4.6.10-G1 十折模型。模型内部仍为 G2 SCMR-DA-SRPE，主干为 E74.4-G2 DARG。Fold 8 使用正式 G1 权重和历史训练成员 383、430。二分类标签为 ASD=0、TD=1。

唯一权威目录为 `/model/spromptgl/experiment/DPSL`。运行路径从入口文件推导。GPU0 验收环境为 Linux x86-64、CPython 3.12.3、PyTorch 2.9.0 的 PPU 2.0 构建、CUDA API 12.9 和 PPU-ZW810E。`runtime/python/` 封存 NumPy、SciPy、scikit-learn 和配套库。系统提供 PyTorch、pandas 和设备驱动。可用设备及库版本以 `environment` 输出为准。

## 推理与验收

```bash
DPSL_ROOT=/model/spromptgl/experiment/DPSL
python3.12 -I -B "$DPSL_ROOT/artifacts.py" verify
python3.12 -I -B "$DPSL_ROOT/cli.py" environment
python3.12 -I -B "$DPSL_ROOT/cli.py" inspect --fold 8
python3.12 -I -B "$DPSL_ROOT/cli.py" evaluate --all
python3.12 -I -B "$DPSL_ROOT/cli.py" resume-check --all
python3.12 -I -B "$DPSL_ROOT/cli.py" audit
```

`evaluate` 严格恢复每折 Best 并执行全部模型。全折输出在 `results/recheck/`，包括有序的逐患者 logits、两类概率、预测和逐折及 pooled 指标。`replay-archive` 单独读取历史预测供对照。`resume-check` 在内存中验证 Latest 的同轮评估与下一次真实训练更新及评估。它逐折比对归档的后续见证。`audit` 也执行十折 Warmup 暂存和断点续训以及负向测试。加 `--record` 可将新审计证据写入 `results/recheck/`。正式记录位于 `results/`。

## 训练与续训

```bash
python3.12 -I -B "$DPSL_ROOT/cli.py" train --fold 8 --from-warmup --run future-f08
python3.12 -I -B "$DPSL_ROOT/cli.py" train --all --from-warmup --run future-ten-fold
python3.12 -I -B "$DPSL_ROOT/cli.py" train-resume --all --run future-ten-fold
python3.12 -I -B "$DPSL_ROOT/cli.py" evaluate --all --run future-ten-fold
```

上述命令用于未来明确启动的训练。全十折按折顺序运行，中断后使用同一运行名续训。每折从完整 E61 epoch 10 状态恢复到 epoch 11，沿用 seed 0、历史算子顺序及三个主干优化器，DPSL AdamW 按原实现初始化并接管新增模块。新运行的 Best、Latest 和可校验的日志在 `runs/<name>/`。正式 Latest 均达到历史早停或 epoch 450。若使用 `train-resume --from-formal-latest --all --run identity-check`，命令核验每折终止条件，并报告 0 个剩余训练轮次。

## 受控制品恢复

公开仓库提供 Python 源码、文档和汇总指标。恢复推理、验收和训练所需的完整冻结目录包含患者级输入、成员和权重，限授权服务器通过下方流程传输。旧公开提交曾纳入患者级结果表和 Warmup 状态。 [旧提交记录](https://github.com/tangaii/SPromptGL/commit/bb6b0101d959d205b21c20be270b281363cd10aa)

在受控服务器生成归档及校验文件后，可将归档部署到同权限的工作目录。当目录已包含 GitHub 快照时，指定新的解包目录，避免混入旧快照文件。

```bash
python3.12 -I -B "$DPSL_ROOT/artifacts.py" pack > DPSL.tar.gz
sha256sum DPSL.tar.gz > DPSL.tar.gz.sha256
sha256sum -c DPSL.tar.gz.sha256
mkdir DPSL
tar -xzf DPSL.tar.gz -C DPSL
python3.12 -I -B DPSL/artifacts.py verify
python3.12 -I -B DPSL/cli.py environment
python3.12 -I -B DPSL/cli.py evaluate --all
python3.12 -I -B DPSL/cli.py audit
python3.12 -I -B DPSL/cli.py resume-check --all
```

`SHA256SUMS` 冻结代码、结果、权重和必要的本地资源。只从受控来源获取并校验历史 PyTorch 序列化制品。

## 历史协议

保留原论文的端到端多模态图学习及官方实现的十折外层流程。DPSL 的具体数值行为由封存的 E61、DARG 和 SCMR 代码、配置与十折权重确定。DPSL 沿用 `val=test`、CV 前 RFE 和预处理、转导患者图。19 名有意保留的训练—测试重叠患者维持 DARG 分类和关系资格，并被排除在新增组件的标签监督、Gaussian/EMA 更新和 CAM144 残差之外。这些历史指标不适用于独立泛化评估。 [SPromptGL 论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [作者实现](https://github.com/wanxixi11/SPromptGL)
