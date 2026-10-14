# 实验十六：ROI语义通路修复与医学语义必要性验证

固定10组，不搜索超参数。B0/P0与B1/实验十五A5均真实重训；B1–B4辨别正则化与FC直通；B5–B9检验修复后的医学语义与邻居必要性。

|组|定义|新增参数|新分支WD|
|---|---|---:|---:|
|B0|原始P0|0|0|
|B1|旧ROI/WD.11/FC直通|38242|0.11|
|B2|旧ROI/WD0/FC直通|38242|0|
|B3|旧ROI/WD.11/关闭FC直通|38242|0.11|
|B4|旧ROI/WD0/关闭FC直通|38242|0|
|B5|新耦合/FUNCTION|39201|0|
|B6|新耦合/CONSTANT|39201|0|
|B7|新耦合/ANONYMOUS|39201|0|
|B8|新耦合/SHUFFLED|39201|0|
|B9|新耦合/SELF_ONLY|39201|0|

## 固定机制

原数值输入871×320、48/6/10/256。110节点统计为关联252可信边的mean(z)、mean(abs(z))、population sd(z)、max(abs(z))、degree/maxdegree；度1的sd为0。冻结实验十五原始全部文本、RN50及固定无固定点置换，不另存或改写Prompt。

```text
B1–B4: 实验十五A5原模块；B3/B4只把decoder第65维FC输入置0。
B5–B9:
h = GELU(Linear32(GELU(Linear5→32(stats))))
score_i = cosine(Wq h_i, frozen_unit_CLIP_i)
centered_i = score_i − mean_110_ROI(score)
response_i = tanh(centered_i/0.2)
semantic_i = h_i * response_i (无1+response数值直通)
A+ = max(tanh(z),0) / max(row_sum,1e−12)
A− = max(−tanh(z),0) / max(row_sum,1e−12)
m+ = A+ semantic; m− = A− semantic (B9两者均为I*semantic)
updated = tanh(Wself semantic + Wpos m+ + Wneg m−); 三映射无bias
context_ij = [updated_i+updated_j, abs(updated_i−updated_j)] (64维，无原FC)
edge_signal = decoder(context) − decoder(zeros(context)); reference不detach
decoder = Linear64→32 + GELU + Linear32→1(bias=False)
delta = .3*tanh(alpha)*tanh(edge_signal); alpha0；252scatter到256，未知4维0
Hnew[:,64:320] = H[:,64:320]+delta，前64维不变；原PLN中关闭SPGS，保留CAM/RES/分类器
```

零参考最后bias会恒等相消，B5–B9统一省略该冗余标量，新增39,201参数（比旧38,242多959）；其余decoder bias保留。参考使用同形状零输入且不detach，保证零节点严格零输出；广播参考只在预检中核对值与梯度误差。新分支只有alpha为0，数值/查询/图/解码按固定初始化；两个Adam步骤后检查梯度，不为梯度审计改变初始化。B6相同文本仍可因不同查询产生不同score，并非强制零响应。

原参数1,462,957完全注册保留；原两Adam组lr.0038/WD.11，新分支lr.0038、WD按表。dropout.35、n_hidden18、n_head2、nlayer1、simple-2损失及训练协议不变。B0保留原P0路径，B1–B9按十五A5关闭SPGS且完整保留RES。

## 预检与科学判定

预检核对原模型、B1历史前向与两步Adam、同入口旧参数、新分支家族初始化、零残差、参数组全覆盖、两步有效梯度、模型/两Adam/RNG恢复、Gram值与梯度、对称边、未知维、零输入零输出、无FC直通、常量可响应和不使用label。合成probe仅用于连通性，不比较测试性能或选参数。
正式诊断不强制非零梯度/响应；近零使用属于真实负结果。记录训练梯度与抽样幅度峰值，最佳checkpoint各路径均值、最大值、参数范数及文本/SELF_ONLY/OFF依赖。B5六条件反事实180行，其余组同checkpoint轻量通路诊断只进每fold单行。
“稳定收益”预先定义ACC、BA、Prob-AUC三个指标的三个seed差值全正；医学支持同时要求B5超过B6/B7/B8并且REAL−SHUFFLED同向；通路数值恢复预先采用≥24/30折文本替换Δ概率均值>1e−8。仅为保守描述判定，不是显著性检验，也不用于训练调参。

## 审计

实际阅读三份本地PDF正文、公式及架构图。SPromptGL的GFR为受试者间关系图；CAM的局部4×4×9是潜在Token而非ROI；SPGS是cosine→softmax→max标量门控，RES保留。其ABIDE消融由86.48经GFR达91.73、RES92.08、SPGS92.30，主要提升来自GFR。

BrainPrompt Eq.2把真实AAL116 ROI身份/功能LLM表示注入真实节点再传播。SABER Eq.1为节点语义，Eq.8–13含Graph-CLS、患者文本交互、门控及疾病监督；full/lite并非所有指标同向。两文数据、图谱及评估与本项目不同。本轮借鉴真实节点与传播，不复制AAL、LLM、超图、疾病头或额外损失。可核对本地PDF（哈希见manifest）、[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、SABER arXiv:2607.01901v1。

本地六核心文件与GitHub main 52f9ae9cbe0a3b49ded4a78e51a358dda9632604逐字节一致；本地HEAD1167dd392837a02eb4f83481ae49f02607a9dae5。保留原GFR combined_feat覆盖、simple-2不加入ALoss、加权NLL+四图损失平均、Adam顺序、450/200、严格ACC提高选checkpoint及第五折低ACC保护。

实验十一端点用零基编号关联已核验roi_111_labels.csv，不使用旧映射的UNKNOWN名称：110可信HO ROI、252无向边，未知4个feature位置195/225/232/236直接delta恒0。统计基于默认LedoitWolf相关的Fisher-z，边权为tanh(z)，不是把标准化数据直接称作原Pearson系数。

实验六标量门控、七潜在双主题、十二/十三逐边静态与患者状态、十四层级与疾病头均未给出稳定语义必要性。实验九E9提升经十的疾病内容对照未证明ASD语义独特性。实验十五首次使用真实HO节点与邻居，A5 ACC91.806%，相对P0仅约+0.115pp；训练后正确/错配/常量/SELF_ONLY的delta与预测完全相同，OFF才改变概率，28/30近共同边偏移。源码逐行审计显示query在WD.11下收缩、signed消息微小、数值FC直通及decoder bias可绕过文本，(1+r)也保留无语义数值路径。

本轮B1准确复现十五A5；B2/B3/B4构成WD×FC直通2×2，保留注册参数数。B5–B9采用中心化score→纯乘法语义节点→正负归一化邻居→bias-free映射→对称边解码及未detach的零参考。仅在GFR与PLN之间增加Correlation residual，其余64维不变。没有标签进入编码器，也没有新Prompt或疾病语义头。

每config×seed只播种一次并连续十fold；新增参数fork_rng(150000+seed)隔离。在相同入口下旧参数相同；不同早停造成后续入口分叉，不能同时承诺每fold同初始化与原连续流。预检核对Gram与直接cosine的值/梯度、B1与十五A5两次Adam逐值一致、恢复模型/两Adam/RNG、合成非零语义与预测通路以及常量不会被永久零化。

## 运行、恢复与文件

`python /data/adodas/SPromptGL-main/experiment/experiment16_roi_semantic_pathway_repair/run.py all`：prepare→preflight→30任务300fold→300诊断/180反事实→中文报告→追加handoff→清理。
`python /data/adodas/SPromptGL-main/experiment/experiment16_roi_semantic_pathway_repair/run.py resume`：恢复model/两Adam/早停/best/RNG；`refresh`刷新；`report`重建报告；`prepare`与`preflight`可单独运行。
每config×seed独立进程，10fold在同进程按原顺序，固定两卡各15任务150fold、最多4worker/卡、2线程/worker，不回退CPU。GPU逻辑cuda:0对应worker CUDA_VISIBLE_DEVICES；UUID及物理分配在manifest。每25epoch与折边界原子保存带SHA256完整状态，已完成任务跳过，失败恢复最近已验证状态，最新提交唯一，日志追加保留。完成后all/resume只验收，不重训。
临时状态与源码在/tmp/spromptgl16-state-*，只修改临时network.py/model.py；原six源码及实验一至十五逐文件哈希保护。复核需读原源码+integration_patch.diff+run.py+roi_pathway.py+manifest；Prompt引用十五prompts.json。最终保留README、两个源码、diff、manifest、30任务日志与scheduler/preflight日志、三CSV和report，无重复文本、受试者预测、永久权重或临时源码。
沿用StratifiedKFold10(shuffle=True,random_state10)、val=test、450/200及严格ACCcheckpoint。共享标准化/RFE与val=test限制单列报告。三seed复用fold，30折不视为独立样本；B5−B4同时改变多项机制且参数+959。未执行git commit/push。
