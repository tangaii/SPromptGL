# 实验十五 AARSP

研究问题：真实HO ROI解剖功能文本与患者内连接传播是否优于常量、匿名、错配和self-only控制？仅研究局部ROI语义，无全局疾病头。

|组|定义|
|---|---|
|A0|原始P0|
|A1|无SPGS保留RES|
|A2|数值图/常量文本|
|A3|匿名ROI身份|
|A4|真实HO名称|
|A5|真实HO名称+功能|
|A6|固定语义错配|
|A7|真实文本/仅自身消息|

## 冻结数学机制

z为原CSV的252个可信Fisher-z连接；110个ROI按实验十一已验证顺序。每人每节点统计为mean(z)、mean(abs(z))、population std(z)、max(abs(z))、degree/max(degree)。单边std=0。原始数值不按标签加工。

```text
h_i = GELU(Linear32(GELU(Linear5→32(stat_i))))
s_i = normalize(Wq h_i) · frozen_unit_CLIP_i
r_i = 0.5*tanh(s_i/0.2);  htilde_i = (1+r_i)*h_i
A_ij = tanh(z_ij)/degree_i on the 252 undirected edges
m_i = sum_j A_ij Wmessage htilde_j (A7: m_i=Wmessage htilde_i)
g_i = GELU(Wself htilde_i + bself + m_i)
edge_context = [g_i+g_j, abs(g_i-g_j), z_ij]
d_ij = Linear32→1(GELU(Linear65→32(edge_context)))
beta = 0.3*tanh(alpha), alpha initialized 0
delta_ij = beta*tanh(d_ij); scatter 252 to 256, unknown four = 0
Hnew[:,64:320] = H[:,64:320] + delta; Hnew[:,:64]=H[:,:64]
```

A2–A7新增38,242参数：数值MLP1,248；查询32,768；消息1,024；自身变换1,056；边解码2,145；alpha 1。原参数1,462,957全部注册保留。全部Adam lr .0038、weight_decay .11；新组与原组设置相同，没有额外正则化搜索。原my_weight_init完成后追加ROI模块，fork_rng(150000+seed)隔离，只有alpha零初始化；不同时零化解码器导致断梯度。

缓存静态患者节点统计和邻接，冻结RN50文本只编码一次；前向仅张量化MLP/矩阵乘法，无患者×边Python循环。相似度采用等价Gram公式，不含可学习文本MLP；温度与幅值预先固定，不根据测试调整。A6复用A5全部向量的无固定点置换。A7保留消息映射参数，替换为自身传播。

## 论文、源码和历史审计

实际阅读三份本地论文正文、架构图、关键公式与消融：SPromptGL的GFR为受试者间四关系图，CAM对应VariLengthInputLayer与EncodeLayer；SPGS为MLP文本原型与4×4×9潜在Token的cosine→softmax→max；RES为局部特征与AvgPool全局特征的逐维softmax交互后加回x。停止SPGS只跳过该标量加权，保留整个RES段，门控等价1，绝非0.5。

SPromptGL Fig.2 ABIDE从86.48%经GFR到91.73%、RES到92.08%、SPGS到92.30%；主要增量来自GFR，SPGS约+0.22pp ACC。BrainPrompt Eq.2将真实AAL116 ROI身份/功能的冻结LLM编码加到节点后再消息传播；其ABIDE ROI单模块约+1.67pp ACC，但用了另一队列、图谱和验证协议。SABER Eq.1在AAL节点加文本投影再全局注意力，Eq.8–13在图级用Graph-CLS检索患者文本、残差注入及疾病监督；表I full/lite的ABIDE ACC为71.58/69.82%，AUC为70.40/72.23%，语义并非全指标提升。本文只迁移真实节点锚定和邻居传播思想，不复制其AAL、LLM、超图、患者或疾病头。来源：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、本地SABER/arXiv:2607.01901v1。

本地六核心源码与GitHub main 9de3b1561a6ac107b0f4dad8ce01df74638c0a4c逐字节一致；本地HEAD仍为1167dd392837a02eb4f83481ae49f02607a9dae5。实际保留原GFR中combined_feat覆盖分支以及ALoss未加入simple-2最终损失的行为，不顺手修复旧模型。原加权NLL与四图GraphConstructLoss平均值、两Adam更新顺序、450/200、ACC严格提高选checkpoint和第5折低ACC防护均保留。

实验十一旧映射表roi_i_name/roi_j_name仍含历史UNKNOWN，实际名称必须按零基端点关联新版roi_111_labels.csv，并核验name_verified/column_order_verified。110节点、252边、平均度4.581818，度范围1–12；未知82列/atlas3455对应特征195/225/232/236，不进入ROI图且直接delta恒0。连接是默认LedoitWolf收缩相关后的Fisher-z，消息权重固定tanh(z)，不误称原始Pearson r。

实验六signed门控、七双主题潜在Token残差没有稳定超过P1；实验十二逐边残差为独立验证单fold，不能与本轮直接比较历史ACC；实验十三原协议下的静态和患者LOW/MID/HIGH逐边对照均未支持正确语义；实验十四文本改写/对应关系差异极小，疾病头与原预测相关约.961；实验九E9疾病改善经实验十ADHD/中性文本控制未证明ASD语义必要性。本轮没有重做这些数学结构，也没有疾病/对比/正交损失。

实现入口在隔离model.py中GFR之后、PLN之前，训练与评价均调用inject_roi；只有H[:,64:320]增加ROI delta。ROI节点先融合所有关联真实边的五个统计，经文本条件化后跨真实邻居传播，节点更新再用于所有相连边，已验证单ROI文本影响非直接关联连接，因此不是独立逐边重加权。CLIP原型冻结在1024维，学习32→1024数值查询；利用WqᵀWq计算归一化分母，避免巨大查询张量，预检核对与直接cosine公式相同。

“所有折同初始化”与“各配置原始连续十折随机流”在早停轮数不同后不能同时满足。遵从本任务明确保留连续随机流的要求：每个config×seed仅起始播种一次，新增初始化在fork_rng隔离；预检证明相同入口下全部旧参数一致，新参数在A2–A7一致；不把随后分叉隐瞒为同初始化实验。与实验十三/十四的A0入口锚定策略明确区分。

## 运行与恢复

`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py all` 完成prepare→预检→双GPU调度→240折→150反事实→中文报告→handoff→清理。
`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py resume` 恢复完整model/Adam/早停/RNG状态；完成折不重训。
`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py prepare`；`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py preflight`；`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py refresh`；`python /data/adodas/SPromptGL-main/experiment/experiment15_anatomically_anchored_roi_prompt/run.py report`。

24个独立config×seed进程，每个按原顺序连续10折；交错分配两卡各12任务/120折，最多4并行/卡、2线程/进程，不回退CPU。保存完整任务日志；仅最新原子提交状态进入CSV，临时state位于本机/tmp，完成删除。已COMPLETE后all/resume只验收，不启动新训练。

全部八组固定，原StratifiedKFold(10,shuffle=True,random_state=10)，val=test、原ACC checkpoint、450 epoch/200 patience、原全图训练监督边界。保持指定历史协议，不替换成新的严格CV。已知共享标准化/RFE及val=test限制在最终报告单列；新文本不是个体诊断或ASD病理声明。

临时源码由根目录六文件复制；仅network.py/model.py应用真实integration_patch.diff，运行时辅助函数来自run.py，核心ROI实现来自roi_semantic.py。复核必须同时读基线、diff、两个新源码和manifest，不能仅看diff。原目录与历史目录均只读。

最终文件：本README、run.py、roi_semantic.py、prompts.json、integration_patch.diff、manifest.json；results三份逐折CSV与report.md；24任务日志、scheduler/preflight日志。无需永久源码副本、权重或受试者级预测。未自动git commit/push。
