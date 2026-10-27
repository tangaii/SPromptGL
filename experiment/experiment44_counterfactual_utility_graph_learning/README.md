# Experiment 44 — CU-GL

本实验验证 **Counterfactual Utility-Guided Directed Patient Graph Learning**。
模块只作用于四模态患者图和GFR输入，不引入PEN、SPGS、RES、CLIP或GPT文本特征。

候选边在每个外层折开始时由四个模态的余弦Top-16固定生成，接收者在行、发送者在列。训练内3折交叉拟合教师在冻结后计算
`U(i←j,m)=log p(y_i|with edge)-log p(y_i|without edge)`；该量是预测模型的边干预响应，不是临床因果结论。

E2–E9共享候选池、CAM、输出维度、训练轮数和残差上限；差异只在方向、NULL拒绝和效用监督。阶段A若发现教师效用没有可辨认信号，按照预注册规则停止正式创新折训练并记录 `PREFLIGHT_SCIENTIFIC_NO_GO`。

所有生成文件位于本目录。根源码和数据只读；完成后才对 `handoff.md` 做带前缀校验的幂等追加。
