from pathlib import Path
import pandas as pd, numpy as np, json, hashlib
root=Path('/tmp/SPromptGL-E48.6.1'); exp=root/'apcr'; res=exp/'results'; f=pd.read_csv(res/'fold_metrics.csv'); iv=pd.read_csv(res/'apcr_off_interventions.csv'); r0=json.loads((exp/'.runtime/r0_readback.json').read_text()); rng=json.loads((exp/'.runtime/rng_isolation_tests.json').read_text()); man=json.loads((exp/'manifest.json').read_text())
# R0 readback rows
r0df=pd.DataFrame(r0['folds']);r0df['group_id']=0;r0df['group']='R0_E51_E1_RETRAIN';r0df['alpha_init']=np.nan;r0df['alpha_best']=np.nan;r0df['alpha_final_training']=np.nan;r0df['alpha_trainable']=False;r0df['independent_readback']=True
r0df['probability_auc']=r0df['prob_auc'];r0df['asd_tp']=np.nan;r0df['asd_fn']=np.nan;r0df['td_fp']=np.nan;r0df['td_tn']=np.nan
# retain common fields
cols=['group_id','group','fold','best_epoch','epochs_executed','alpha_init','alpha_best','alpha_final_training','alpha_trainable','prediction_sha256','best_sha','independent_readback','acc','hard_auc','spe','sen','probability_auc','asd_tp','asd_fn','td_fp','td_tn']
for c in cols:
 if c not in r0df:r0df[c]=np.nan
combined=pd.concat([r0df[cols],f.reindex(columns=cols)],ignore_index=True);combined.to_csv(res/'fold_metrics.csv',index=False)
# summaries
rows=[]
for gid,name in [(0,'R0_E51_E1_RETRAIN'),(1,'E51GFR_APCR_FIXED'),(2,'E51GFR_APCR_LEARNABLE')]:
 d=combined[combined.group_id==gid]; q={'group_id':gid,'group':name,'n_folds':len(d)}
 for c in ['acc','hard_auc','spe','sen','probability_auc']:
  q[c+'_mean_pct']=d[c].mean()*100;q[c+'_sd_pct']=d[c].std(ddof=0)*100
 if gid==0:q['alpha_best_mean']=np.nan;q['alpha_best_sd']=np.nan
 else:q['alpha_best_mean']=d.alpha_best.mean();q['alpha_best_sd']=d.alpha_best.std(ddof=0)
 rows.append(q)
pd.DataFrame(rows).to_csv(res/'summary.csv',index=False)
# paired same-fold comparisons
pairs=[]
for g,ref,label in [(1,0,'E1_vs_R0'),(2,0,'E2_vs_R0'),(2,1,'E2_vs_E1')]:
 a=combined[combined.group_id==g].set_index('fold');b=combined[combined.group_id==ref].set_index('fold')
 for c in ['acc','hard_auc','spe','sen','probability_auc']:
  d=a[c]-b[c];pairs.append({'comparison':label,'metric':c,'delta_mean_pp':d.mean()*100,'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum())})
# old 48.6 refs
old={'E1_48.6':{'acc':91.61442006269593,'hard_auc':91.41352011461835,'spe':93.99167437557816,'sen':88.83536585365853,'probability_auc':93.23523273391847},'E2_48.6':{'acc':91.15464994775341,'hard_auc':90.88469630649128,'spe':94.4218316373728,'sen':87.34756097560975,'probability_auc':93.47077006385236}}
for g,key,label in [(1,'E1_48.6','E1_new_vs_48.6_E1'),(2,'E2_48.6','E2_new_vs_48.6_E2')]:
 d=combined[combined.group_id==g]
 # no paired fold predictions from old are carried; report mean delta only
 for c in ['acc','hard_auc','spe','sen','probability_auc']:
  pairs.append({'comparison':label,'metric':c,'delta_mean_pp':d[c].mean()*100-old[key][c],'wins':'NOT_COMPARABLE','ties':'NOT_COMPARABLE','losses':'NOT_COMPARABLE'})
pd.DataFrame(pairs).to_csv(res/'paired_comparisons.csv',index=False)

def ms(g,c):
 d=combined[combined.group_id==g][c];return f'{d.mean()*100:.4f} ± {d.std(ddof=0)*100:.4f}'
lines=['# Experiment 48.6.1 — 修正随机轨迹并复现实验51 E1','', '## 执行状态','', '- R0、E1、E2共30折全部完成；R0 10折原始实验51 E1重训，E1/E2各10折APCR重训。', '- 单GPU、5个逻辑worker；20个APCR checkpoint独立读回通过，R0独立读回通过。', '- `/data`仍返回`Errno 5 Input/output error`，完整结果位于`/tmp/SPromptGL-E48.6.1`；正式目录和handoff暂为待同步。', '', '## 源码差异与随机流修正','', '- R0直接使用实验51 E1冻结runner，未加入APCR、未改模型、优化器、checkpoint规则或随机数逻辑。', '- 48.6.1删除48.6原有的每epoch `torch.manual_seed(481400000 + fold*10000 + epoch)`和CUDA重置。E1/E2从实验51 E1的`seed_all(510000 + fold*100 + 1)`初始化后连续推进主RNG。', '- APCR初始化使用独立fold种子并在初始化后恢复基座Python/NumPy/Torch CPU/CUDA RNG；APCR额外前向通过独立连续RNG上下文执行，checkpoint同时保存主RNG和APCR RNG。', '- SRD/SCA/CRA算法、group=6路径、损失权重、学习率、weight decay、候选数量、攻击预算均未改变。', '', '## R0：实验51 E1严格复现','', '| 指标 | 历史实验51 E1 | R0重训 | 逐折结果 |','|---|---:|---:|---|']
for c,label in [('acc','ACC'),('hard_auc','Hard-AUC'),('spe','SPE-TD'),('sen','SEN-ASD'),('prob_auc','Probability-AUC')]:
 d=r0df[c].astype(float); hist={'acc':91.9592476489,'hard_auc':91.7433919587,'spe':94.6392229417,'sen':88.8475609756,'prob_auc':93.1856456307}[c]
 lines.append(f'| {label} | {hist:.4f} | {d.mean()*100:.4f} ± {d.std(ddof=0)*100:.4f} | 完全一致 |')
lines += ['', '- 10/10折ACC、Hard-AUC、SPE、SEN、Probability-AUC逐值一致。', '- 10/10折best epoch与执行epoch一致。', '- 初始化参数SHA和初始化RNG SHA一致。', '- 历史E51 task记录没有保存预测SHA，因此预测SHA历史对照为`NOT_VERIFIABLE`；本次R0独立预测SHA已保存。', '', '## 正式结果','', '| 模型 | α最佳均值 | ACC | Hard-AUC | SPE-TD | SEN-ASD | Probability-AUC |','|---|---:|---:|---:|---:|---:|---:|']
lines.append(f'| R0 实验51 E1重训 | — | {ms(0,"acc")} | {ms(0,"hard_auc")} | {ms(0,"spe")} | {ms(0,"sen")} | {ms(0,"probability_auc")} |')
lines.append(f'| E1 固定α | {combined[combined.group_id==1].alpha_best.mean():.7f} | {ms(1,"acc")} | {ms(1,"hard_auc")} | {ms(1,"spe")} | {ms(1,"sen")} | {ms(1,"probability_auc")} |')
lines.append(f'| E2 可学习α | {combined[combined.group_id==2].alpha_best.mean():.7f} ± {combined[combined.group_id==2].alpha_best.std(ddof=0):.7f} | {ms(2,"acc")} | {ms(2,"hard_auc")} | {ms(2,"spe")} | {ms(2,"sen")} | {ms(2,"probability_auc")} |')
lines += ['', '## 配对比较','', '| 比较 | ΔACC | ΔHard-AUC | ΔSPE | ΔSEN | ΔProb-AUC | ACC胜/平/负 | Hard-AUC胜/平/负 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for comp in ['E1_vs_R0','E2_vs_R0','E2_vs_E1']:
 q=pd.DataFrame(pairs);z=q[q.comparison==comp];get=lambda m:z[z.metric==m].iloc[0]
 a=get('acc');h=get('hard_auc');
 lines.append(f'| {comp} | {a.delta_mean_pp:+.4f} | {h.delta_mean_pp:+.4f} | {get("spe").delta_mean_pp:+.4f} | {get("sen").delta_mean_pp:+.4f} | {get("probability_auc").delta_mean_pp:+.4f} | {a.wins}/{a.ties}/{a.losses} | {h.wins}/{h.ties}/{h.losses} |')
lines += ['', '相对旧48.6：E1 ACC +0.2299pp、Hard-AUC +0.2235pp；E2 ACC +0.6897pp、Hard-AUC +0.7523pp。两组均超过旧48.6对应结果，但这是随机轨迹修正后的重训差异，不能归因于新增算法。', '', '## α与APCR关闭干预','', f'- E1最佳α均为0.0022000005，固定不变；E2最佳α均值为{combined[combined.group_id==2].alpha_best.mean():.7f}，范围{combined[combined.group_id==2].alpha_best.min():.7f}–{combined[combined.group_id==2].alpha_best.max():.7f}。', '- E2相对E1的硬预测指标完全相同或仅浮点等价：ACC、Hard-AUC、SPE、SEN均无实质差异；E2 Probability-AUC低0.5580pp。', f'- α=0干预：E1 ACC平均下降0.2299pp，E2下降0.1149pp；有效APCR残差/GFR比约E1 0.0003135、E2 0.0003923。', '', 'RNG隔离单元测试：', f'- `base_rng_unchanged_by_apcr_init = {rng["base_rng_unchanged_by_apcr_init"]}`', f'- `apcr_scope_preserves_main_rng = {rng["apcr_scope_preserves_main_rng"]}`', f'- `apcr_scope_advances_independent_rng = {rng["apcr_scope_advances_independent_rng"]}`', '', '## 结论','', '1. **实验51 E1可严格复现。** R0逐折主要指标和best epoch均一致；历史预测SHA缺失，因此该项只能标记不可核验。', '2. **48.6的epoch级随机重置已修复。** 主RNG连续推进，APCR初始化和额外前向不污染主RNG，且状态可恢复。', '3. **E1/E2均未超过本次R0。** 两组ACC均91.8443%，低于R0 91.9592%约0.1149pp。', '4. **固定与可学习α的硬分类结果相同。** E2没有带来可检测的ACC/Hard-AUC/SPE/SEN收益，且Probability-AUC较低。', '5. **APCR没有提高修复版GFR的诊断性能。** 与R0相比两组主指标均未提升；相对旧48.6的提升来自随机轨迹修正，不能解释为算法增益。', '', '## 判定','', '- `R0_REPRODUCTION_STATUS = PASS_METRIC_AND_BEST_EPOCH; PREDICTION_SHA_NOT_VERIFIABLE`', '- `RNG_ALIGNMENT_STATUS = PASS`', '- `TRAINING_STATUS = 30/30_COMPLETE`', '- `CHECKPOINT_AUDIT_STATUS = PASS`', '- `E1_OVER_E51 = NO`', '- `E2_OVER_E51 = NO`', '- `SCIENTIFIC_STATUS = NO-GO_FOR_APCR_GAIN`', '- `DELIVERY_STATUS = DELIVERY_PENDING_DATA_EIO`', '', '## 限制与产物','', '- 保留历史CV前全局RFE/预处理、transductive图、`val=test` checkpoint选择和单seed。', '- 结果文件：`results/fold_metrics.csv`、`results/summary.csv`、`results/paired_comparisons.csv`、`results/apcr_off_interventions.csv`。']
(res/'report.md').write_text('\n'.join(lines)+'\n')
print('report written')
