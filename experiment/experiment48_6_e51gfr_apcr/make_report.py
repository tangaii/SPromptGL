from pathlib import Path
import pandas as pd, json, numpy as np, hashlib
root=Path('/tmp/SPromptGL-E48.6');res=root/'results';fold=pd.read_csv(res/'fold_metrics.csv');iv=pd.read_csv(res/'apcr_off_interventions.csv');pair=pd.read_csv(res/'paired_comparisons.csv');man=json.loads((root/'manifest.json').read_text());pre=json.loads((root/'.runtime/preflight.json').read_text())

def fmt(mean,sd): return f'{mean:.4f} ± {sd:.4f}'
def table_row(name,base,alpha,a,ha,spe,sen,pa): return f'| {name} | {base} | {alpha} | {a} | {ha} | {spe} | {sen} | {pa} |'
lines=[]
lines += ['# Experiment 48.6 — E51 LegacyFixedGFR + APCR低残差消融','', '## 执行状态','', '- 20/20折完成：E1固定α和E2可学习α各10折；单GPU 0，5个逻辑worker。', '- 独立读回：20/20最佳checkpoint、预测SHA、四项指标和Probability-AUC核验通过。', '- 工程审计：E51 E1 LegacyFixedGFR基座10/10折逐值parity通过；同一fold两组共享初始状态、输入和APCR初始化，group=6的SRD/SCA/CRA均启用。', '- `/data`创建新目录时持续返回`Errno 5 Input/output error`；全部新产物暂存于`/tmp/SPromptGL-E48.6`，尚未声称已同步到正式目录。', '', '## 审计与实现边界','', '实验51 `dugsa.py`、`run.py`、`config.json`及48.5分支的APCR源码均使用SHA核验的本地/远端快照。新适配器只调用实验51 `LegacyFixedGFR(parts) -> (fused, adjs, details)`，并从同一组`graphs[m].score431`读取APCR候选分数；没有复制第二套GraphLearn，也没有叠加B0-B GFR。CAM取实验51返回值的第0项。', '', 'E51 E1初始状态由对应Experiment43.1 fold anchor通过实验51 E1 `build_state`重建，不加载E51训练后的最佳权重。E1和E2的唯一变量是`alpha_raw.requires_grad`；两者初始α均为0.0022、上限0.2、APCR优化器和epoch级随机计划相同。', '', '## 正式结果（mean ± population SD，百分比）','', '| 模型 | 基座 | α最佳均值 | ACC | Hard-AUC | SPE-TD | SEN-ASD | Probability-AUC |','|---|---|---:|---:|---:|---:|---:|---:|']
e51={'acc':91.9592476489,'hard':91.7433919587,'spe':94.6392229417,'sen':88.8475609756,'pa':93.1856456307}
b0={'acc':91.7294,'hard':91.6046,'spe':93.1360,'sen':90.0732,'pa':93.2322}
e4851={'acc':91.8430,'hard':91.6977,'spe':93.5661,'sen':89.8293,'pa':93.4698}
e4852={'acc':91.4995,'hard':91.2806,'spe':94.2137,'sen':88.3476,'pa':92.7895}
lines.append(table_row('实验51 E1（历史参考）','LegacyFixedGFR','—',f'{e51["acc"]:.4f} ± 2.8684',f'{e51["hard"]:.4f} ± 2.8950',f'{e51["spe"]:.4f} ± 3.3856',f'{e51["sen"]:.4f} ± 4.1651',f'{e51["pa"]:.4f} ± 3.0236'))
lines.append(table_row('48.5 E1 固定α（历史参考）','B0-B + APCR','0.0022',f'{e4851["acc"]:.4f} ± 3.1094',f'{e4851["hard"]:.4f} ± 3.0720',f'{e4851["spe"]:.4f} ± 4.2242',f'{e4851["sen"]:.4f} ± 3.4050',f'{e4851["pa"]:.4f} ± 3.5083'))
lines.append(table_row('48.5 E2 可学习α（历史参考）','B0-B + APCR','0.0022→0.00239',f'{e4852["acc"]:.4f} ± 2.8784',f'{e4852["hard"]:.4f} ± 2.8831',f'{e4852["spe"]:.4f} ± 3.2156',f'{e4852["sen"]:.4f} ± 3.4922',f'{e4852["pa"]:.4f} ± 2.7829'))
for gid,label in [(1,'48.6 E1 固定α'),(2,'48.6 E2 可学习α')]:
 d=fold[fold.group_id==gid]; alpha=d.alpha_best.mean();
 lines.append(table_row(label,'LegacyFixedGFR + APCR',f'{alpha:.7f} ± {d.alpha_best.std(ddof=0):.7f}',fmt(d.acc.mean()*100,d.acc.std(ddof=0)*100),fmt(d.hard_auc.mean()*100,d.hard_auc.std(ddof=0)*100),fmt(d.spe.mean()*100,d.spe.std(ddof=0)*100),fmt(d.sen.mean()*100,d.sen.std(ddof=0)*100),fmt(d.probability_auc.mean()*100,d.probability_auc.std(ddof=0)*100)))
lines += ['', '实验51 E1与48.5结果为历史只读参考；48.6两组是本次重新训练结果。所有均值为10折、population SD。']
lines += ['', '## α与配对比较','', f'- E1固定α：10折最佳α均为0.0022000005，训练期间保持不变；`alpha_raw`无梯度。', f'- E2可学习α：最佳α均值为{fold[fold.group_id==2].alpha_best.mean():.7f} ± {fold[fold.group_id==2].alpha_best.std(ddof=0):.7f}，范围{fold[fold.group_id==2].alpha_best.min():.7f}–{fold[fold.group_id==2].alpha_best.max():.7f}；每折均有有限非零α梯度和更新。', '', '| 比较 | ΔACC (pp) | ΔHard-AUC (pp) | ΔSPE (pp) | ΔSEN (pp) | ΔProb-AUC (pp) | ACC胜/平/负 | Hard-AUC胜/平/负 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
def pget(g,ref,m): return pair[(pair.group==f'E{g}')&(pair.reference==ref)&(pair.metric==m)].iloc[0]
for name,g,ref in [('48.6 E2 − 48.6 E1',2,'E48_5_E1'),('48.6 E1 − 实验51 E1',1,'E51_E1'),('48.6 E2 − 实验51 E1',2,'E51_E1'),('48.6 E1 − 48.5 E1',1,'E48_5_E1'),('48.6 E2 − 48.5 E2',2,'E48_5_E2'),('48.6 E1 − B0-B',1,'B0_B'),('48.6 E2 − B0-B',2,'B0_B')]:
 # E2-E1 is recomputed below because stored references are historical; use fold.
 if name=='48.6 E2 − 48.6 E1':
  a=fold[fold.group_id==1].set_index('fold');b=fold[fold.group_id==2].set_index('fold');ds={m:(b[m]-a[m]) for m in ['acc','hard_auc','spe','sen','probability_auc']};vals={m:(float(v.mean()*100),int((v>1e-12).sum()),int((abs(v)<=1e-12).sum()),int((v< -1e-12).sum())) for m,v in ds.items()}
 else:
  vals={}
  for m in ['acc','hard_auc','spe','sen','probability_auc']:
   z=pget(g,ref,m);vals[m]=(float(z.delta_mean_pp),int(z.wins),int(z.ties),int(z.losses))
 lines.append(f'| {name} | {vals["acc"][0]:+.4f} | {vals["hard_auc"][0]:+.4f} | {vals["spe"][0]:+.4f} | {vals["sen"][0]:+.4f} | {vals["probability_auc"][0]:+.4f} | {vals["acc"][1]}/{vals["acc"][2]}/{vals["acc"][3]} | {vals["hard_auc"][1]}/{vals["hard_auc"][2]}/{vals["hard_auc"][3]} |')
lines += ['', '胜/平/负按同fold逐项比较；历史参考并非本轮重新训练。']
lines += ['', '## APCR关闭干预（不重训）','', '在每个48.6最佳checkpoint上临时将α置零，保留LegacyFixedGFR的四路原始消息和CAM。以下为10折均值；`effective residual/base`已包含α，另保留未乘α的原始残差比用于审计。', '', '| 组 | 完整ACC | α=0 ACC | ΔACC (pp) | 完整Hard-AUC | α=0 Hard-AUC | 翻转/折均值 | 有效残差/GFR |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for gid,label in [(1,'48.6 E1'),(2,'48.6 E2')]:
 z=iv[iv.group_id==gid];f=fold[fold.group_id==gid].set_index('fold');eff=[]
 for _,r in z.iterrows():eff.append(r.apcr_residual_over_gfr*f.loc[int(r.fold),'alpha_best'])
 lines.append(f'| {label} | {z.acc.mean()*100:.4f} | {z.alpha_off_acc.mean()*100:.4f} | {(z.alpha_off_acc.mean()-z.acc.mean())*100:+.4f} | {z.hard_auc.mean()*100:.4f} | {z.alpha_off_hard_auc.mean()*100:.4f} | {z.prediction_flips.mean():.2f} | {np.mean(eff):.7f} |')
lines += ['', 'E1固定α关闭APCR后平均ACC下降约0.4585 pp；E2平均ACC不变。该结果说明固定α训练的checkpoint在直接推理上有小幅APCR依赖，但E2的可学习α没有稳定的硬预测增量。它不能证明训练期APCR对主干参数从未产生间接正则化。']
lines += ['', '## 机制与科学结论','', '1. **基座适配正确。** LegacyFixedGFR四路GraphLearn、四路GCN消息、加权累加和CAM在10折工程测试中与实验51 E1接口逐值一致；没有带回B0-B的PHENO覆盖问题，也没有重复注册GraphLearn。', '2. **低残差没有超过E51 E1。** 48.6 E1 ACC为91.6144%，比实验51 E1低0.3448 pp；E2为91.1546%，低0.8046 pp。两组都没有超过91.9592%，E1也低于B0-B参考91.7294%约0.1149 pp。', '3. **固定优于可学习。** E2−E1为ACC −0.4598 pp、Hard-AUC −0.5288 pp、SPE +0.4302 pp、SEN −1.4878 pp、Probability-AUC +0.2355 pp；ACC同fold胜/平/负为0/6/4，Hard-AUC为2/3/5。可学习α总体向上漂移，但未带来硬分类收益，且ASD灵敏度下降。', '4. **APCR直接影响很小且不一致。** 有效残差/GFR均值约为E1 0.0003526、E2 0.0003244；这与原始未乘α残差比（约0.1603、0.1310）不同。小α确实把注入CAM的增量限制在很小范围。', '5. **本轮不支持继续以α调节作为主要修复方向。** 结果不能把较高的Probability-AUC等同于历史Hard-AUC或ACC改善；也不能将LegacyFixedGFR工程修复的收益冒充APCR收益。']
lines += ['', '## 判定','', '- `ENGINEERING_STATUS = PASS`：E51基座10/10 parity、α单变量测试、20/20读回均通过。', '- `AUDIT_STATUS = PASS`：任务、checkpoint、预测SHA、数据/CV划分和根源码保护哈希一致。', '- `ACC_OVER_E51_E1 = NO`；固定和可学习两组均未超过91.9592%。', '- `HARD_AUC_OVER_E51_E1 = NO`；两组均低于历史E51 E1。', '- `SCIENTIFIC_STATUS = NO-GO_FOR_ALPHA_AS_GAIN`：低α固定组相对可学习组更稳，但没有产生相对LegacyFixedGFR的可重复增益；不能据此宣称APCR创新已成立。', '- `DELIVERY_STATUS = SCRATCH_STAGED_DATA_EIO`：正式目录写入仍受OSSFS EIO阻断；当前完整交付路径为`/tmp/SPromptGL-E48.6`。', '', '## 可复现性与限制','', f'- 数据SHA：`{man["data_sha256"]}`；映射SHA：`{man["mapping_sha256"]}`；48.6源码签名：`{man["code_signature"]}`。', '- 保留历史CV前全局RFE/预处理、transductive 871节点图、`val=test` checkpoint选择和单seed；这些协议存在信息泄漏与选择偏倚，不能视为无偏外部泛化。', '- 48.6没有重训实验51 E1、48.5或B0-B历史参考；这些结果用于同fold/同协议的只读对照。', '- 结果文件：`results/fold_metrics.csv`、`results/summary.csv`、`results/paired_comparisons.csv`、`results/apcr_off_interventions.csv`。']
(res/'report.md').write_text('\n'.join(lines)+'\n')
print('wrote',res/'report.md')
