# B0 物理删除 PEN 等价性审计

日期：2026-10-21  
状态：完成；未运行正式 epoch、未创建实验33、未提交或推送 Git。

## 结论

| 检查 | 结果 | 最大绝对误差 |
|---|---|---:|
| FORWARD | `EXACT` | 0 |
| BACKWARD | `EXACT` | 0 |
| ADAM，3步 | `EXACT` | 参数、梯度与状态均为0 |
| PEN_PHYSICALLY_REMOVED | `YES` | 不适用 |
| B0_HISTORY_VALID | `YES` | 保留实验32已完成的 B0 结果 |

在相同 seed 0、同一真实 ABIDE 输入及同一第1折训练索引下，B0-A 与 B0-B 的 69 个共有参数在自然初始化时已经逐值相同。再按完整参数名和形状执行显式复制后，eval 与 train 模式的 GFR、CAM、分类输出、反向梯度和三次 Adam 更新均逐位一致。实验32的 B0 `ACC=92.0742%` 应作为历史结果保留；这项结构审计不代表重新训练必然得到相同分数，也不比较 B0 与 B2 的性能优劣。

## 比较对象与构造

- **B0-A** 使用实验32的隔离运行时源码、原 `EvalHelper` 和 `integration.install(config="B0")`。其前向通过 `integration.patch` 跳过 SPGS/RES，但 PLN 构造时仍初始化 MLP、创建 `prompt_text_features` buffer 和 RES 的 `avg_pool`。
- **B0-B** 在临时运行时中独立注册原始 `InputLayer → Encoder → FeedForward → Outputlayer`，沿用原 GFR、CAM 层、分类器、初始化函数及 Adam 设置。PLN 构造过程没有调用原 PLN 构造器，也没有创建 PEN 对象后再删除。
- 构造追踪：A 调用 RN50 加载、`get_prompt_text_features`、MLP 初始化和 AvgPool 初始化各1次；B 四项均为0。B 的 `MLP`、`prompt_text_features`、`avg_pool` 属性均不存在。A 使用已在 CLIP 专项审计中验证过 A=C 等价性的本地 checkpoint（SHA-256 `afeb0e10…04b6762`），没有下载模型。
- 两模型的 GFR、CAM、OutputLayer 共有参数均按全名和形状匹配；所有共有 buffer 逐值一致，唯一移除的是 B 不需要的 `prompt_text_features` buffer。完整名称、形状和哈希见 `results.json`。

## 参数与随机状态

| 模块 | 注册参数张量 | 注册参数元素 | 有梯度张量 | 有梯度元素 |
|---|---:|---:|---:|---:|
| B0-A PLN | 39 | 575,121 | 34 | 49,538 |
| B0-B PLN | 35 | 49,544 | 34 | 49,538 |
| B0-A PEN MLP | 4 | 525,577 | 0 | 0 |

B0-A 的 PLN Adam 收录575,121个参数元素，B0-B 收录49,544个；两边 GFR Adam 均收录887,836个。B0-A MLP 的四个参数梯度均为 `None`，三次更新后参数 SHA 不变，Adam 没有为其创建状态，因此 `weight_decay` 没有作用于它。两边 PLN 的 `fusion_weight` 仍各有6个注册元素，但原 B0 前向不使用它们，也没有梯度；所以 B0-B 的49,544个 PLN 参数中实际产生梯度的是49,538个。

固定 seed 前的 CPU/CUDA RNG SHA 相同。两个独立同 seed 构造完成后，69个共有参数自然初始化仍逐位相同，参数状态 SHA 均为 `09456cf7…51d59293`；但完整 RNG 状态已经不同：A 的 CPU/CUDA SHA 为 `00cb5f3e…b901388` / `7fc490ff…d311a38`，B 为 `8de4da78…5326763f` / `de355ca0…6e4702b`。因此本次观测到的是**共有初始参数不变、构造后的随机序列改变**；未来 Dropout 等随机操作可能走不同轨迹。审计通过逐次保存并恢复完整 RNG，隔离了这项影响。

## 等价性测试

- **前向：** 使用 `data/ABIDE/processed_standard_data.csv` 的871名受试者、原始模态维度 `[48, 6, 10, 256]` 和 `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)` 的第1折索引。未执行交叉验证训练。eval/train 两种模式均逐项比较 GFR 特征与邻接、CAM InputLayer/Encoder/FeedForward 输出、注意力图、分类 log-probability、概率、类别及隐藏表示。所有张量 `torch.equal=True`，GFR/CAM/分类输出最大绝对误差均为0；train 模式前后 RNG 消耗也一致。
- **反向：** 严格使用 `ClsLoss + mean(GraphConstructLoss over 4 modalities)`，没有 PEN、CLIP 或 DCER 辅助损失。分类损失为 `1.032340168952942`，图学习损失为 `-1.1294156312942505`，总损失为 `-0.0970754623413086`；A/B 均完全相同。共有参数的49个非空梯度张量逐值一致、最大误差0；共有的20个 `grad=None` 参数也逐名报告，未补零或跳过。A 的 MLP 梯度全部为 `None`。
- **Adam：** 两个 Adam 优化器均按实验32的 `lr=0.0038`、`weight_decay=0.11` 及默认参数创建，执行3次真实 `zero_grad → forward → backward → optimizer_MP0.step → optimizer_MF.step`。每一步共有参数、梯度、`step`、`exp_avg` 与 `exp_avg_sq` 均逐位相等，最大误差0；两边下一步的 RNG 状态保持一致。
- **N1 有效参数负对照：** B0-B 的 `Outputlayer.classifier.weight[0,0]` 加5，检测到分类输出变化，最大误差 `2.8560815`。
- **N2 闲置 MLP 负对照：** B0-A 的 `MLP.t1.weight[0,0]` 加5，B0 输出逐位不变，最大误差0。扰动后权重已恢复。

完整逐参数梯度、损失、各步 Adam 状态误差、RNG、哈希和负对照明细见 [results.json](results.json)。可在项目根目录用 `CUDA_VISIBLE_DEVICES=0 python -B experiment/b0_pen_physical_removal_equivalence_audit/audit.py` 重复运行；需要现有 `/root/.cache/clip/RN50.pt`，脚本缺少它时会停止且不会下载。

## 对历史结果与后续工作的判断

1. **B0-A 与 B0-B 的数学前向等价：是。** 在共有参数匹配时，eval/train 输出逐位一致。
2. **分类输出一致：是。** log-probability、概率与预测类别均一致。
3. **反向梯度一致：是。** 所有共有参数逐名比较通过，B0-A MLP 无梯度。
4. **原始 Adam 更新一致：是。** 三次更新的参数和优化器状态均逐位相同。
5. **闲置 MLP 是否影响有效学习计算：否。** 它被 optimizer 收录，但没有梯度、参数更新或 Adam 状态；改变其权重也不改变 B0 输出。
6. **同 seed 共有参数初始化是否变化：本次没有变化。** 69个共有参数逐位相同。不过构造结束后的 RNG 状态改变，不能把它外推成完整训练随机轨迹相同。
7. **是否保留实验32 B0 `ACC=92.0742%`：是。** 这是已经完成的历史 B0-A 结果，结构等价审计没有理由覆盖或重算它。
8. **后续是否可用物理无 PEN 的 B0-B：可以。** 它保留原有效计算和参数，注册参数减少525,577个。若要求复现同一随机训练轨迹，仍应显式保存/恢复训练开始 RNG 状态或共有初始状态。
9. **是否需要 B0/B2 同折20折实验：本结构问题不需要。** 实验32已有同折10折 B0/B2 比较；本审计证明的不是 B0 对 B2 的性能优势。若要提出性能优越性，应另用适当的独立评估协议回答，而不是把结构等价当作证据。

审计前后根目录核心源码、ABIDE 输入、实验32运行源码、manifest、报告和120折结果 SHA-256 全部一致；CLIP专项审计文件也未改动。运行时副本在系统临时目录中执行并自动清理。
