from pathlib import Path
import pandas as pd, numpy as np, json
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
d=pd.read_csv(R/'fold_metrics.csv'); refs=pd.read_csv(R/'historical_reference.csv'); inter=pd.read_csv(R/'interventions.csv')
def stat(df):
 return {k:(df[k].mean()*100,df[k].std(ddof=0)*100) for k in ['acc','hard_auc','spe','sen']}
def fmt(v): return f'{v[0]:.4f}±{v[1]:.4f}'
def table_metrics():
 rows=[]
 for g,name in [('R0_B0B','B0-B历史'),('R1_CAM_ONLY','CAM-only历史'),('R2_T1_NO_WD','T1无有效GCN衰减'),('D0','D0动态真实图'),('D1','D1固定真实图'),('D2','D2匹配重连图'),('D3','D3仅self-loop')]:
  q=refs[refs.group==g] if g.startswith('R') else d[d.group==g]
  s=stat(q);rows.append(f'|{g}|{name}|{fmt(s["acc"])}|{fmt(s["hard_auc"])}|{fmt(s["spe"])}|{fmt(s["sen"])}|')
 return '\n'.join(rows)
def paired(a,b):
 x=d[d.group==a].sort_values('fold');y=d[d.group==b].sort_values('fold');out=[]
 for k in ['acc','hard_auc','spe','sen']:
  xa=x[k].to_numpy();yb=y[k].to_numpy();out.append(f'{k}: {(xa-yb).mean()*100:+.4f} pp ({(xa>yb).sum()}/{(xa==yb).sum()}/{(xa<yb).sum()})')
 return '; '.join(out)
foldtable='\n'.join('|%d|%.4f|%.4f|%.4f|%.4f|'%(r.fold,r.acc*100,r.hard_auc*100,r.spe*100,r.sen*100) for _,r in d.sort_values(['fold','group']).iterrows())
alpha=pd.read_csv(R/'scale_diagnostics.csv'); alpha_test=alpha[alpha.partition=='test'].groupby(['group','alpha'])[['acc','nll']].mean()
alpha_text='; '.join(f'{g}/α={a}: ACC {v.acc*100:.2f}%, NLL {v.nll:.4f}' for (g,a),v in alpha_test.iterrows())
intmean=inter.groupby(['group','intervention'])[['acc','hard_auc','test_flips']].mean()
int_text='\n'.join(f'|{g}|{m}|{v.acc*100:.3f}|{v.hard_auc*100:.3f}|{v.test_flips:.1f}|' for (g,m),v in intmean.iterrows())
gates=json.loads((ROOT/'.runtime/gates.json').read_text())
report=f'''# Experiment 47.2 — 有效患者图传播失败的根因验证

## 结论先行

本轮按预注册的 `rho=0.25` 完成了 Stage A 十折只读诊断、fold1/fold6 的 D0–D3
训练侧 pilot，以及 Gate A/B 通过后的其余八折扩展，共 **40/40 个新增训练折**。
GPU0 运行 fold1–5，GPU1 运行 fold6–10。尺度控制显著修复了 T1 的训练优化困难：
D0 epoch100 训练 NLL 相比同折 T1 降至 0.309/0.383 倍；但真实固定图 D1 与结构匹配
重连 D2 的分类差异很小，D1 未同时满足相对 D2 的 ACC 和 Hard-AUC 七折严格胜出门槛。
因此本轮最强结论是：**T1 的失败首先包含消息尺度/优化失衡；在尺度受控后，当前原始
患者关系仍没有被证明具有稳定的独立疾病增量。**

最终状态：`SCALE_OPTIMIZATION_FAILURE_VERIFIED`（优化层面）；`ACTIVE_GRAPH_NONDISCRIMINATIVE`
（当前真实图关系增量检验）；不支持把“放大 GCN 消息”作为 MICCAI 2027 第一创新点。

## 1. 审计范围和源码事实

本实验隔离加载 Experiment47.1 的 immutable source snapshot，没有改动根目录代码。
源码显示 `GraphConv.forward` 先计算 `X @ W`，再计算 `A @ support`；每个 `GCN`
是第一层 ReLU 加第二层线性传播。`GraphLearn` 先对模态投影做余弦阈值 `0.9`，
原图为871×871四关系图。`GFR.forward` 的 `i==1` 重新写入
`combined_feat=f_w*feat1+raw_features`，覆盖 PHENO 分支；最终实际跨患者消息来自
ANAT、FUNC、CORR 三路。`GraphConstructLoss`只接收模态输入和邻接矩阵，不直接接收
GCN 的 W1/W2；simple-2 的 ALoss 不进入反向传播。

历史基线身份已逐折核验：B0-B来自 Experiment43.1 的物理移除 PEN 模型；CAM-only
来自 Experiment43.4 的单 CAM checkpoint；T1来自 Experiment47.1 的六个有效 GCN
矩阵取消 weight decay。所有新增组都从同一 fold 的未训练 B0-B anchor 开始。

## 2. Stage A：失败首先发生在哪里

T1 的十折训练 ACC 为 70.8258%，外层测试 ACC 为 71.2970%，训练 NLL 0.5959、测试
NLL 0.6275；它不是“训练集学得很好、测试集泛化差”的普通过拟合形态，而是训练端
本身就没有恢复分类。相比之下，历史 B0-B 训练 ACC 约96.71%，训练 NLL约0.1333。

T1 固定 checkpoint 的十折测试 alpha 干预结果为：α=0 72.97%、α=0.01 73.54%、
α=0.10 76.75%、α=0.25 78.87%、α=0.50 80.14%、α=1 81.51%；中心化 α=1 为
79.05%。这些是事后固定模型干预，不是新的训练超参数，也不能解释为泛化提升。

Stage A 的最佳模型平均消息比例（手工重算 ANAT+FUNC+CORR）为：B0-B `M/X`
约 `2.4×10^-6`，T1 约 `4.16`；T1 中心化后约 `2.84`。T1 的患者间消息标准差约
1.41，故巨大范数不只是一个全体患者共同偏移，但消息尺度仍远高于自身输入。
这与 47.1 的初始化 W 范数约25、训练后 W1/W2 收缩至约0.002–0.004 倍的事实一致。

## 3. 预注册尺度控制和四组训练

对每一 fold 只用 outer-train 估计 `mu_X`、`mu_M` 与
`scale=0.25||X_c||_F/(||M_c||_F+10^-8)`，不使用测试标签，也不对拼接后的自身与
邻居做 LayerNorm。D0 仍动态优化 GraphLearn 和原图正则；D1/D2/D3 冻结图，均只
优化同样的 GCN/CAM 参数，三路有效 GCN 无 weight decay。D2 的审计中十折四模态均
满足对称性、每节点非自环度数、总边数、自环、权重多重集和站点对边数完全相等，
站点内边比例也完全相等；不是未经匹配的随机图。

Gate A/B 只读训练折证据，均通过。epoch100 的训练 NLL与同折 T1 比例为：D0
0.3086/0.3833，D1 0.5071/0.7584；D2、D3均有效，中心化消息训练比例误差最大
`1.31×10^-8`。因此按冻结规则完成全部剩余折，没有用外层测试 ACC 选择扩展组。

## 4. 正式四指标（十折 mean±population SD，%）

|组别|说明|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---|---:|---:|---:|---:|
{table_metrics()}

逐折新增训练结果：

|fold|D0 ACC|D0 AUC|D1 ACC|D1 AUC|D2 ACC|D2 AUC|D3 ACC|D3 AUC|
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join('|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|'%(f, d[(d.group=='D0')&(d.fold==f)].acc.iloc[0]*100,d[(d.group=='D0')&(d.fold==f)].hard_auc.iloc[0]*100,d[(d.group=='D1')&(d.fold==f)].acc.iloc[0]*100,d[(d.group=='D1')&(d.fold==f)].hard_auc.iloc[0]*100,d[(d.group=='D2')&(d.fold==f)].acc.iloc[0]*100,d[(d.group=='D2')&(d.fold==f)].hard_auc.iloc[0]*100,d[(d.group=='D3')&(d.fold==f)].acc.iloc[0]*100,d[(d.group=='D3')&(d.fold==f)].hard_auc.iloc[0]*100) for f in range(1,11))}

相对配对差值（均值差，严格胜/平/负）：

- D1−D2：{paired('D1','D2')}。
- D1−D3：{paired('D1','D3')}。
- D0−D1：{paired('D0','D1')}。
- D0−D3：{paired('D0','D3')}。

D1 相比 D2 只有 ACC 4/4/2、Hard-AUC 7/1/2 的胜平负，不能同时满足七折门槛；
D1 相比 D3 的 ACC 8/2/0、Hard-AUC 9/0/1，说明跨患者传播相对于纯 self-loop
GCN 变换有一定作用，但不能排除这种作用来自一般关系/尺度训练，而非真实关系质量。
D2 与 D1 的平均 ACC 几乎相同，故“真实邻居优于随机邻居”的核心证据不足。

## 5. 固定 checkpoint 干预

下表为各组最佳 checkpoint 的十折平均；它们是模型依赖诊断，不能代替从头重训。

|组|干预|ACC|Hard-AUC|测试折平均翻转数|
|---|---|---:|---:|---:|
{int_text}

严格把 GCN 输出置零和仅 self-loop 分开实现。D1 的真实图在自身 checkpoint 上是
原始前向；替换为重连图后 ACC 从90.696%降至89.777%，但重连图独立训练 D2 又达到
90.468%，说明固定交换干预存在分布偏移，不能把一次下降解释为关系因果。D3 的
self-loop checkpoint 被换成真实图时明显失配，证明该模型使用了邻接结构，但没有
证明真实关系的疾病价值。

## 6. 竞争性解释的证据矩阵

|候选解释|源码依据|训练/结构证据|分类影响|状态|
|---|---|---|---|---|
|消息尺度导致优化失衡|有效 GCN W 使用 Adam weight decay=0.11|T1训练端ACC/NLL已差；D0/D1–D3在rho=0.25后训练NLL显著恢复|D0 ACC约90.47%，远高于T1但仍低于B0-B/CAM-only|**SCALE_OPTIMIZATION_FAILURE_VERIFIED**|
|消息主要是共同群体偏移|中心化前后均值、范数和患者方差审计|T1消息均值只占总范数约2.5%，中心化仍有患者差异|中心化固定干预仍造成明显分布变化|部分支持，非唯一原因|
|真实关系质量不足|原始GraphLearn阈值图与固定 D2 结构匹配重连|D1与D2同容量、同尺度；ACC差+0.229pp且仅4折严格胜|D1没有稳定超过D2，SPE还低约1.07pp|**ACTIVE_GRAPH_NONDISCRIMINATIVE**（当前协议）|
|图只比自身GCN变换有用|D3保留同样GCN但A=I|D1比D3 ACC +2.298pp、AUC +2.287pp|跨患者传播有模型依赖，但关系身份未证实|支持“有传播作用”，不支持疾病增量|
|动态 GraphLearn 联合优化失败|D0仍优化GraphLearn/图正则，D1–D3冻结图|D0与D1接近，均低于强基线；没有单独动态图收益|未能把 D1 的微小差异归为动态学习|未证实为唯一根因|
|测试 cohort 依赖|原始 transductive 图协议|本轮未改变历史协议，未将其另作因果实验|仍是泛化局限，不作本轮主要结论|未检验|

## 7. 科研判定

1. **尺度/优化问题得到实质支持。** T1 的分类失败发生在训练端；固定 rho 后四组
   都能正常拟合，说明“消息过大导致优化失衡”是真实贡献因素。它不能解释为什么
   D1不稳定优于D2，也不能作为患者图创新本身。
2. **真实患者关系的独立疾病增量没有得到验证。** D1比D3有增益，但D1与严格结构
   匹配的D2几乎相同，核心七折 ACC 门槛失败。当前可以判定为
   `ACTIVE_GRAPH_NONDISCRIMINATIVE`，而不是“患者图永远无效”。
3. **动态学习无法有效利用关系仍未单独闭合。** D0和固定图性能接近，且D0包含额外
   GraphLearn/图正则差异；本轮没有足够证据将失败唯一归因于动态优化。

因此不建议继续围绕“增大消息、取消 weight decay或简单重连/门控”提出新的 GFR
创新。若要继续，必须先建立不依赖历史 `val=test`、不依赖 transductive 全 cohort、
并能在严格关系负控制下证明患者图增量的独立协议；本轮本身不足以支持 MICCAI 2027
第一创新点。

## 8. 复现、限制和交付

`.runtime/final_audit.json` 显示轨迹有限、图约束、十折完整性、rho比例和 checkpoint
重载指标全部 PASS。结果文件包括 `baseline_diagnostics.csv`、`scale_diagnostics.csv`、
`cam_only_validation.csv`、`graph_control_audit.csv`、`training_trajectory.csv`、
`fold_metrics.csv`、`historical_reference.csv` 和 `interventions.csv`；`manifest.json`
记录源码、anchor、checkpoint、固定图、输入数据和结果 SHA256。

仍沿用历史全队列 RFE/标准化、871人 transductive 图及 `val=test` 选 best；单 seed、
共享训练患者的十折也不是独立临床样本。根目录 `handoff.md` 的追加状态必须以实际
OSSFS 写入结果为准；若写入失败，待追加正文会保存在本实验目录并明确标记。
'''
(R/'report.md').write_text(report)
print('report lines',len(report.splitlines()))
