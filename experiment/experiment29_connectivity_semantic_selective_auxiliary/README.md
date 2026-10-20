# 实验29：CORR双层语义对齐与选择性辅助修正

独立运行器仅修改本目录 `.runtime/sources` 内的根源码副本。原始GFR、CAM、RES、分类器和历史实验只读；`integration.patch` 提供全部接入差异。正式规模为20组、seed0、每任务连续十折，共200折。P0/E1直接读取已完成的历史seed0十折。

所有512个历史64维文本向量减同一公共均值后L2归一化。CORR256经Linear/LayerNorm/GELU得到64维Anchor；全局ASD/TD文本CE温度0.2、权重0.05。仅以训练折估计TD参考、站点收缩效应方向和稳定性，从252条真实连接选最多16条。共享Query对冻结的对应连接文本计算局部InfoNCE（温度0.2、权重0.02）和注意力响应。粗粒度局部Key重复异常原型，其局部损失不具有连接身份辨识能力。

PHENO28（年龄/性别）、ANAT6和FUNC10各有独立输入特征Selector、无偏置Encoder/Value及CORR条件化Router。Selective使用权重0.005的训练折软稀疏惩罚，不作硬阈值删除。两个独立alpha初始化为0，`delta=0.10*tanh(tanh(alpha_sem)*semantic_candidate+tanh(alpha_aux)*aux_candidate)` 只追加CAM的CORR位置，每坐标绝对值不超过0.10。所有配置注册同样71277个新参数，新Adam参数组weight_decay=0；关闭的分支不参与相应损失和残差。

C0–C8、F0–F10的完整冻结设计、实现细节及对照边界在 `manifest.json` 和 `results/report.md`。C3/F3关闭语义后数学相同，仍分别独立训练。数值码在最终归一化之前匹配均值/协方差，归一化后不保证逐对Gram或类别原型几何相同。

```bash
python run.py prepare
python run.py preflight
python run.py all
# 中断后：python run.py resume
python run.py report
```

完成后的 `resume` / `refresh` 只核验已完成CSV，不重建已删除的训练任务。若200折已完整保存而仅收尾失败，可运行 `python run.py finalize`，重新验收并在释放controller锁之后清理临时目录。manifest的 `training_code_sha256` 保留实际训练时源码指纹，`post_training_changes` 记录完成后的I/O修复及报告判读差分；这些修复没有改变模型、损失、指标或既有训练结果。

每任务单独进程、只播种一次、连续十折；GPU0/1各10任务100折，每卡最多4 worker，每worker2 CPU线程。每25epoch及折边界原子保存模型、两个Adam、所有RNG、best/wait和已提交行；已完成任务校验后跳过。临时权重、锁和日志完成后清理，仅保留必要源码、缓存、manifest和三个结果文件。正式结果为ACC、原始硬标签AUC、SEN-ASD和SPE-TD；十折均值±population SD。

保留历史test=val选模、外层CV前RFE/标准化及871人transductive图。此结果是偏乐观的内部可比性证据，不能称独立无泄漏泛化。潜在Token不是ROI，ANAT/FUNC为质量特征，语义/门控敏感性不能替代医学负对照。
