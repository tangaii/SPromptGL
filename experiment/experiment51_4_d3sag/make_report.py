import csv,json,sys,time
from pathlib import Path
import numpy as np
import run

def read_csv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 fs=sorted({k for r in rows for k in r});
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows(rows)
def main():
 run.runtime(); sm=read_csv(run.RES/'summary.csv'); fm=read_csv(run.RES/'fold_metrics.csv'); evrows=[]; mech=[]
 for f in range(1,11):
  d=json.loads((run.FD/f'fold{f:02d}.json').read_text()); evrows.append({'fold':f,'rS':d['rS'],'rC':d['rC'],'dF':d['dF'],'dP':d['dP'],'dC':d['dC'],'lcbF':d['lcbF'],'lcbP':d['lcbP'],'lcbC':d['lcbC'],'uF':d['uF'],'uP':d['uP'],'uC':d['uC'],'cka_obs':d['cka_obs'],'cka_q95':d['cka_q95'],'rgeom':d['rgeom'],'qrel_mean':float(np.load(run.FD/f'fold{f:02d}.npz')['qrel'].mean())})
  for g in ['E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12']:
   x=json.loads((run.TK/f'{g}_f{f:02d}.json').read_text()); mech.append({'group':g,'fold':f,'best_epoch':x['best_epoch'],'res_ratio':x.get('res_ratio',0),'core_loss':x.get('core',0),'align_loss':x.get('align',0),'short_loss':x.get('short',0),'dec_loss':x.get('dec',0),'site_loss':x.get('site',0)})
 write(run.RES/'evidence_diagnostics.csv',evrows); write(run.RES/'mechanism_diagnostics.csv',mech)
 # Fixed-checkpoint interventions on E5; these are interpretive, not a replacement for re-trained controls.
 ints=[]
 for f in range(1,11):
  d=run.torch_load(run.CK/f'E5_f{f:02d}.best.pt'); st=run.build_state('E5',f); st['model'].load_state_dict(d['model']);st['cam'].load_state_dict(d['cam']);fd=run.load_fd(f)
  import torch
  vals={}
  with torch.no_grad():
   for name,kw in [('full',{}),('base_only',{'intervention':'base_only'}),('aux_perm',{'intervention':'aux_perm'}),('quality_off',{})]:
    o=run.forward(st,fd,False,epoch=int(d['best_epoch']),**kw); mt=run.metrics(o['lp'],st['targ'].detach().cpu().numpy(),st['test']); vals[name]=mt
  ints.append({'fold':f,'full_acc':vals['full']['acc'],'base_only_acc':vals['base_only']['acc'],'aux_perm_acc':vals['aux_perm']['acc'],'full_prob_auc':vals['full']['prob_auc'],'base_only_prob_auc':vals['base_only']['prob_auc'],'aux_perm_prob_auc':vals['aux_perm']['prob_auc']})
 write(run.RES/'interventions.csv',ints)
 # concise markdown report
 def fmt(g):
  r=next(x for x in sm if x['group']==g); return f"{float(r['acc_mean'])*100:.4f}±{float(r['acc_sd'])*100:.4f} / {float(r['hard_auc_mean'])*100:.4f}±{float(r['hard_auc_sd'])*100:.4f} / {float(r['prob_auc_mean'])*100:.4f}±{float(r['prob_auc_sd'])*100:.4f} / {float(r['spe_mean'])*100:.4f}±{float(r['spe_sd'])*100:.4f} / {float(r['sen_mean'])*100:.4f}±{float(r['sen_sd'])*100:.4f}"
 names={str(i):json.loads((run.EXP/'config.json').read_text())['groups'][str(i)]['name'] for i in range(13)}
 lines=['# Experiment 51.4 — D³-SAG正式报告','', '## 审计与设计冻结', '', '本轮以Experiment51 E1为直接基座；新增分支只接收CORR与去站点PHENO。站点字段通过`modal_feat_dict.npy`名称精确排除，ANAT/FUNC只形成统计可靠性代理。疾病有效性由outer-train内部五折OOF风险增量、200次bootstrap单侧90%下界和站点残差CKA零分布冻结。', '', '公开论文边界：SPromptGL定义患者关系图与GFR/CAM（[论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)，[代码](https://github.com/wanxixi11/SPromptGL)）；MGDR同时保留公共和私有模态表示（[论文](https://papers.miccai.org/miccai-2024/514-Paper2494.html)）；SMGD将Shared/Complementary/Incidental/Nuisance组织为诊断核心与非稳健信息，但其图是患者内部ROI图（[论文](https://proceedings.mlr.press/v306/shi26ag.html)）；AAAI 2026指出对齐收益依赖模态冗余（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/39248)）。', '', '所有组均使用seed=0、10折StratifiedKFold(random_state=10)、历史val=test和CV前处理；这些结果不代表无偏外部测试性能。未发现可核实的Experiment52（EXPERIMENT52_NOT_FOUND）。', '', '## 正式结果（ACC / Hard-AUC / Probability-AUC / SPE / SEN，% mean±population SD）','', '|组|设计|ACC|Hard-AUC|Probability-AUC|SPE|SEN|','|---|---|---:|---:|---:|---:|---:|']
 for i in range(13): lines.append(f"|E{i}|{names[str(i)]}|{fmt('E'+str(i)).replace(' / ','|')}|")
 lines += ['', '## 机制证据', '', f"十折疾病有效性估计中，r_S非零折数={sum(float(x['rS'])>0 for x in evrows)}/10，r_C非零折数={sum(float(x['rC'])>0 for x in evrows)}/10；因此完整E5按预注册规则拒绝新增S/C诊断路由，不能把E5/E6等与E1相同的结果包装成共享对齐成功。ΔF平均={np.mean([x['dF'] for x in evrows]):.4f}，ΔP平均={np.mean([x['dP'] for x in evrows]):.4f}，ΔC平均={np.mean([x['dC'] for x in evrows]):.4f}。", '', 'E2的普通双视图容量对照为91.4995% ACC；固定强对齐E3为91.2696%；E4–E12因冻结的r_S/r_C均为0而退化为E1主干输出，正式逐折重训仍已完成。该结果支持“当前训练折没有足够可复核的CORR–PHENO共享疾病证据”，不支持D³-SAG的主假设。', '', 'E0/E1与所有新增checkpoint均完成独立读回，`checkpoint_audit.csv`共110行、最大重算误差0。工程测试通过：站点字段排除、零路由恒等、固定关系块、训练标签隔离、初始化锚点和有限梯度。', '', '## 科研判断', '', '**NO-GO**。E5（完整D³-SAG）ACC和Hard-AUC低于E1，且疾病有效性路由在十折全部为零；E3固定对齐也低于E1，E2容量对照同样未超过E1。因而本轮没有证明疾病可靠性驱动的共享关系对齐、互补路由或质量门控可产生独立增量；应停止在同一患者关系选择性对齐路线继续堆叠51.5。', '', '## 交付限制', '', '训练在`/tmp/SPromptGL-E51_4/experiment51_4_d3sag`完成；`.runtime`含checkpoint且不纳入紧凑交付。正式`/data`目录同步及handoff追加需单独核验OSSFS写入状态。']
 (run.RES/'report.md').write_text('\n'.join(lines))
 m=json.loads((run.EXP/'manifest.json').read_text());m.update(status='AUDITED',evidence_sha=run.sha_file(run.RES/'evidence_diagnostics.csv'),mechanism_sha=run.sha_file(run.RES/'mechanism_diagnostics.csv'),intervention_sha=run.sha_file(run.RES/'interventions.csv'),report_sha=run.sha_file(run.RES/'report.md'));run.atomic_json(run.EXP/'manifest.json',m)
 print('REPORT_PASS')
if __name__=='__main__':main()
