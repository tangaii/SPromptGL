#!/usr/bin/env python3
"""E42 post-run QA only: no new fit, no classifier update, no rule selection.

Separate file is needed to keep the registered execution code immutable while
checking requested train interventions, distribution shift and report integrity.
It also corrects a diagnostic-only site-permutation labeling defect: R2/R3 do
not read SITE in their gate, so permuting only SITE leaves their gate unchanged.
The original raw task files are retained, never silently overwritten.
"""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,copy,json,time,subprocess,hashlib,fcntl,shutil
import numpy as np,pandas as pd,torch
from sklearn.neighbors import NearestNeighbors
import run as r
import mechanism as m

def fold_audit(fold):
 started=time.time()
 d=r.data();tr,te=d['folds'][fold-1];ref=r.ref(fold);cache=r.load(r.RT/f'cache/scores_{fold:02}.pt');teacher=r.load(r.RT/f'cache/teacher_{fold:02}.pt');v,p,g=r.frozen(fold);mu=torch.tensor(ref['mu'],device='cuda');rows=[]
 methods=['S1_ABSZ','S2_TRAIN_DISCRIM',cache['selected']]+[f'S0_RANDOM_{s}' for s in r.CFG['q2']['random_mask_seeds']]
 for name,adj in zip(('PHENO','ANAT','FUNC','CORR'),g):
  support=adj.detach().ne(0);support.fill_diagonal_(False);degree=support.sum(1).float()
  rows.append(dict(kind='PATIENT_GRAPH',fold=fold,modality=name,offdiagonal_directed_entries=int(support.sum()),density=float(support.float().sum()/(871*870)),mean_neighbors=float(degree.mean()),zero_offdiagonal_nodes=int((degree==0).sum()),max_neighbors=int(degree.max())))
 for person in teacher['people']:
  for name in methods:
   sc=cache['scores'][name][person]
   for k in (4,8):
    idx=np.argsort(-sc,kind='stable')[:k];mask=np.zeros(39,'float32');mask[idx]=1
    a=r.intervention_row(v,p,g,mu,int(person),mask,False,1,'live')
    rows.append(dict(kind='TRAIN_MASK_VALIDATION',fold=fold,subject_row=int(person),score=name,k=k,**a))
 # Compare OOD proxies with the same fixed train reference; no manifold claim.
 f=d['x'][:,r.COLS]/ref['sigma'];nn=NearestNeighbors(n_neighbors=1).fit(f[tr]);base=nn.kneighbors(f[te],return_distance=True)[0][:,0]
 for name in methods:
  for strength in (.5,1.):
   xx=f[te].copy()
   for j,person in enumerate(te):
    idx=np.argsort(-cache['scores'][name][person],kind='stable')[:8];xx[j,idx]+=strength*(ref['mu'][person,idx]/ref['sigma'][idx]-xx[j,idx])
   dist=nn.kneighbors(xx,return_distance=True)[0][:,0];rows.append(dict(kind='DISTRIBUTION_PROXY',fold=fold,score=name,k=8,strength=strength,baseline_nn_distance=float(base.mean()),intervened_nn_distance=float(dist.mean()),distance_ratio=float(np.mean(dist/np.maximum(base,1e-6))),fraction_farther=float(np.mean(dist>base)),note='scaled Euclidean 1NN proxy, not proof of in/out-of-manifold'))
 # Mean mask overlap with numeric Top-|z| and between prompt families.
 for fam in ('P0','P1','P2','P3'):
  intersections=[]
  for person in te:
   a=set(np.argsort(-cache['scores']['S4_'+fam][person])[:8]);b=set(np.argsort(-cache['scores']['S1_ABSZ'][person])[:8]);intersections.append(len(a&b)/len(a|b))
  rows.append(dict(kind='RANK_DIAGNOSIS',fold=fold,family=fam,mean_jaccard_topz=float(np.mean(intersections)),mean_score_absz_rho=float(np.mean([r.correlation(cache['scores']['S4_'+fam][i],abs(ref['z'][i])) for i in te]))))
 # Exact selector optimizer restoration, independent of training artifacts.
 checkpoint=r.load(r.RT/f'checkpoints/q3_{fold:02}_C3_0.pt');features=m.edge_features(ref['z'],d['x'],r.EDGES,r.direction_vector(),np.ones(39,'float32'));fit=checkpoint['fit'];center=features[fit].mean((0,1));scale=np.maximum(features[fit].std((0,1)),.1);ft=torch.tensor((features-center)/scale,device='cuda')
 def restored_step():
  m.restore(copy.deepcopy(checkpoint['rng']))
  with torch.random.fork_rng(devices=[0]):s=m.Selector().cuda()
  s.load_state_dict(copy.deepcopy(checkpoint['model']));o=torch.optim.Adam(s.parameters(),lr=.01);o.load_state_dict(copy.deepcopy(checkpoint['optimizer']));o.zero_grad();person=int(fit[0]);score=s(ft[person]);mask=m.topmask(score,8,True);xx=m.intervene(v.feat,person,torch.tensor(r.COLS,device='cuda'),mu,mask,False,1);out,_,_=r.forward(v,xx);loss=score.square().mean()+.05*torch.nn.functional.softplus(.05-(p[person,int(p[person].argmax())]-out[person,int(p[person].argmax())]));loss.backward();o.step();return copy.deepcopy(s.state_dict()),copy.deepcopy(o.state_dict())
 aa=restored_step();bb=restored_step();assert r.equal(aa,bb)
 rows.append(dict(kind='SELECTOR_RESUME',fold=fold,status='BITWISE_NEXT_ADAM_UPDATE',backbone_sha=m.trunksha(v),gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]),seconds=time.time()-started))
 r.cw(r.RT/f'tasks/qa_{fold:02}.csv',rows);print('QA',fold,'COMPLETE',flush=True)

def complete():
 r.aggregate();raw=r.cr(r.RES/'reliability_audit.csv');corrected=[]
 for row in raw:
  if row.get('kind')=='GATE_DEPENDENCE' and row.get('role') in ('R0','R2','R3') and row.get('control')=='site_permuted':continue
  corrected.append(row)
  if row.get('kind')=='GATE_DEPENDENCE' and row.get('role') in ('R0','R2','R3') and row.get('control')=='positive':
   cp=dict(row);cp['control']='site_permuted';cp['audit_note']='corrected invariant: gate does not read SITE; original raw diagnostic mislabeled random rank retained in runtime';corrected.append(cp)
 r.cw(r.RES/'reliability_audit.csv',corrected)
 # Encode-space controls use the exact same transformations as registered Q1.
 bank=r.load(r.RT/'cache/text_bank.pt');geometry=[]
 for fam in ('P0','P1','P2','P3'):
  real=m.norm(bank[fam].numpy());gram=real.reshape(26,-1)@real.reshape(26,-1).T
  for control in ('REAL','NEUTRAL','SWAP_ASD_TD','DERANGED_HYPOTHESIS','GRAM_MATCHED_ANONYMOUS'):
   t=m.norm(bank['neutral'].numpy()) if control=='NEUTRAL' else real.copy()
   if control=='SWAP_ASD_TD':t=t[:,::-1].copy()
   if control=='GRAM_MATCHED_ANONYMOUS':
    rng=np.random.default_rng(420008);t=t[...,rng.permutation(t.shape[-1])]*rng.choice([-1.,1.],t.shape[-1])
   pair=(t[:,0]*t[:,1]).sum(-1);gg=t.reshape(26,-1)@t.reshape(26,-1).T
   for i,h in enumerate(r.hypotheses()):geometry.append(dict(kind='CONTROL_GEOMETRY',family=fam,hypothesis=h['id'],control=control,pair_cos=float(pair[i]),pair_distance=float(np.linalg.norm(t[i,0]-t[i,1])),norm_asd=float(np.linalg.norm(t[i,0])),norm_td=float(np.linalg.norm(t[i,1])),gram_max_abs_difference=float(np.max(abs(gg-gram))),note='association-only derangement leaves text Gram unchanged; swapped Gram compares fixed slot order; vectors normalized for routing'))
 r.cw(r.RES/'prompt_audit.csv',[x for x in r.cr(r.RES/'prompt_audit.csv') if x.get('kind')!='CONTROL_GEOMETRY']+geometry)
 dec,summ=r.summary_decisions();info=r.jr(r.RES/'manifest.json');assert info['decisions']==dec,'QC diagnostic correction changed scientific decision unexpectedly'
 # Resolve the conditional R4 placeholder. One surviving role is exactly its
 # already measured R2 gate, not an independent multi-role combination.
 passed_roles=[x['role'] for x in summ['q4_roles'] if x['go']]
 if len(passed_roles)<=1:
  for row in corrected:
   if row.get('kind')=='COMBINATION_GATE':row['status']='RESOLVED_SINGLE_COMPONENT_IDENTICAL_TO_'+passed_roles[0] if passed_roles else 'RESOLVED_NO_SUPPORTED_COMPONENT';row['components']=json.dumps(passed_roles);row['audit_note']='conditional gate resolved; single component measurements are the existing corresponding role rows, no independent multimodal combination'
  r.cw(r.RES/'reliability_audit.csv',corrected)
 rows=[]
 for fold in range(1,11):rows+=r.cr(r.RT/f'tasks/qa_{fold:02}.csv')
 for fold in range(1,11):
  tr,te=r.data()['folds'][fold-1]
  for group in r.CFG['q3']['groups']:
   probe_ids=[];valid=0
   for inner in range(5):
    state=r.load(r.RT/f'checkpoints/q3_{fold:02}_{group}_{inner}.pt');probe_ids+=list(map(int,state['probe_ids']));valid+=sum(int(log['valid_probes']) for log in state['logs'])
   assert not set(probe_ids).intersection(map(int,te))
   rows.append(dict(kind='CF_COVERAGE',fold=fold,group=group,attempted_probes=len(probe_ids),unique_people=len(set(probe_ids)),valid_probes=valid,outer_train_people=len(tr),test_people_used=0))
 r.cw(r.RT/'qa_summary.csv',rows)
 train=pd.DataFrame([x for x in rows if x['kind']=='TRAIN_MASK_VALIDATION']);train['drop_true']=train.drop_true.astype(float);train['k']=train.k.astype(int);train['fold']=train.fold.astype(int)
 grouped=[]
 for fold,df in train[train.k==8].groupby('fold'):
  selected=r.load(r.RT/f'cache/scores_{fold:02}.pt')['selected'];sm=df[df.score==selected].drop_true.mean();z=df[df.score=='S1_ABSZ'].drop_true.mean();rd=df[df.score.str.startswith('S0_RANDOM_')].drop_true.mean();grouped.append(dict(fold=int(fold),selected=float(sm),topz=float(z),random=float(rd)))
 qa_done=[x for x in rows if x['kind']=='SELECTOR_RESUME'];assert len(qa_done)==10
 audit=dict(code_sha=m.sha(__file__),purpose='requested supplementary quality assurance, no optimizer step in source artifacts; cloned selector restoration check only',train_selected_fidelity=grouped,corrected_site_control='R0/R2/R3 SITE permutation is invariant; original runtime rows retained',r4_conditional_components=passed_roles,r4_no_independent_multimodal_claim=len(passed_roles)<2,source_input_unchanged=True,root_handoff_prefix=r.jr(r.RT/'storage_initial.json')['handoff'],qa_tasks=10,gpu_tasks={str(g):sum(int(x['gpu'])==g for x in qa_done) for g in (0,1)},gpu_task_seconds={str(g):sum(float(x['seconds']) for x in qa_done if int(x['gpu'])==g) for g in (0,1)})
 info['post_audit']=audit;info['summary']=summ;r.jw(r.RES/'manifest.json',info);r.write_report(info)
 text=['','## 追加的独立质量验收（不改变冻结训练或门槛）','','独立audit.py用于保留注册run.py字节不变，同时完成训练参与者group-mask干预、分布代理、排名和选择器恢复检查。未重新拟合/挑选模型，未增加正式分类指标。','', '|fold|train选中删除|train随机删除|train Top-abs(z)删除|','|---|---:|---:|---:|']
 for a in grouped:text.append(f"|{a['fold']}|{a['selected']:.6f}|{a['random']:.6f}|{a['topz']:.6f}|")
 text+=['','每fold16名预先固定训练者的group-mask验证补充逐边导师；不是外部测试，也不用于追改评分。runtime/qa_summary.csv保留所有K4/8记录、half/full替换的训练分布1NN代理、S4与Top-abs(z)重合、选择器Adam下一步逐值恢复。1NN距离的改善/增加只能说明该代理，不证明在真实生理数据流形内。','', '发现并更正一个**诊断标签缺陷**：冻结run.py中R0/R2/R3的site_permuted行错误复用了random_rank；这些门控本来不读取SITE，真正只置乱SITE应严格恒等。正式QC风险/正向/反向/随机行及GO计算完全不变；原始错误行保留在runtime/tasks，最终reliability_audit.csv用恒等正向结果构造正确站点置乱对照并注明。未改训练函数、模型参数或注册门槛。','', 'Q3原B0始终冻结，分类残差路径不存在，因此CF经分类残差的梯度为0；实际梯度只经单患者raw-FC mask回到选择器。所有十折选择器在原状态克隆上完成包含CF路径的下一次Adam更新逐值复演，正式checkpoint未更新。']
 text+=['','### 编码器与Prompt失败定位','','|风格|ASD/TD原始余弦均值|对间距离均值|匿名控制Gram最大误差|','|---|---:|---:|---:|']
 geo=pd.DataFrame(geometry)
 for fam,g in geo.groupby('family'):
  rr=g[g.control=='REAL'];aa=g[g.control=='GRAM_MATCHED_ANONYMOUS'];text.append(f'|{fam}|{rr.pair_cos.mean():.6f}|{rr.pair_distance.mean():.6f}|{aa.gram_max_abs_difference.max():.3g}|')
 text+=['','Q1失败模式分别记录：如果可信假说不足7/13，标记PRIOR_UNSUPPORTED（指当前39条和当前队列下未建立支持，不是推翻网络级文献）；如果全部对间距离低于预注册.01才判ENCODER_BOTTLENECK。向量可分辨而探针不胜语义负对照时，是语义独立优势未建立，不能仅凭高余弦断言编码器失效。可学习线性读出吸收固定类别槽交换也是当前功能探针的可识别性限制，不能单凭未胜交换控制判所有医学先验无效。Q1整体NO-GO另有仅3/13假说稳定受支持的独立依据。','', '### 固定B0的评分全表：十折等权机制均值','','仅K8、完整mu替换删除。数字是正确真实类log-prob下降与翻转比例，不是新增正式分类指标。','', '|评分|live删除|fixed删除|live翻转率|','|---|---:|---:|---:|']
 fidelity=pd.read_csv(r.RES/'score_fidelity.csv');ff=fidelity[(fidelity.k==8)&(fidelity.strength==1)&(fidelity.operation=='delete')&(fidelity.replacement=='TD_MU')].copy()
 for name,g in ff.groupby('score'):
  live=g[g.graph=='live'].groupby('fold')[['drop_true','prediction_flip']].mean();fixed=g[g.graph=='fixed'].groupby('fold').drop_true.mean();text.append(f'|{name}|{live.drop_true.mean():.6f}|{fixed.mean():.6f}|{live.prediction_flip.mean():.5f}|')
 text+=['','原B0正确/错误预测分别分层，使用每fold训练内冻结的S4风格：','','|原预测|图|真实类logp下降|原预测类logp下降|样本数|','|---|---|---:|---:|---:|']
 chosen=ff[ff.score==ff.selected_style].copy();chosen['correct_stratum']=chosen.original_correct.astype(str).str.lower().eq('true')
 for (correct,mode),g in chosen.groupby(['correct_stratum','graph']):
  z=g.groupby('fold')[['drop_true','drop_predicted']].mean();text.append(f'|{"正确" if correct else "错误"}|{mode}|{z.drop_true.mean():.6f}|{z.drop_predicted.mean():.6f}|{len(g)}|')
 text+=['','原始GraphLearn非对角支持（十折平均；对称矩阵按两个方向计数）：','','|图|非对角条目|密度|平均邻居|无非对角邻居的节点数|','|---|---:|---:|---:|---:|']
 graph_rows=pd.DataFrame([x for x in rows if x['kind']=='PATIENT_GRAPH'])
 for name,g in graph_rows.groupby('modality'):
  means={k:g[k].astype(float).mean() for k in ('offdiagonal_directed_entries','density','mean_neighbors','zero_offdiagonal_nodes')};text.append(f"|{name}|{means['offdiagonal_directed_entries']:.1f}|{means['density']:.6f}|{means['mean_neighbors']:.3f}|{means['zero_offdiagonal_nodes']:.1f}|")
 text+=['','Prompt评分与数值异常排序的关系：','','|风格|评分与abs(z) Spearman均值|Top8与Top-abs(z) Jaccard均值|','|---|---:|---:|']
 rankrows=pd.DataFrame([x for x in rows if x['kind']=='RANK_DIAGNOSIS'])
 for family,g in rankrows.groupby('family'):text.append(f'|{family}|{g.mean_score_absz_rho.astype(float).mean():.5f}|{g.mean_jaccard_topz.astype(float).mean():.5f}|')
 text+=['','S4定义为训练读出系数乘对应文本加权的signed z/abs(z)贡献后，对每条边取合计贡献绝对值；未加入手工负abs(z)惩罚。相关性由该公式和训练权重产生，正相关本身不算通过，仍必须由与Top-abs(z)同预算干预决定。','', '每fold风格在训练内冻结：'+', '.join(f"fold{fold}={r.load(r.RT/f'cache/scores_{fold:02}.pt')['selected']}" for fold in range(1,11))+'。']
 text+=['','### Q4正向与负对照的实际作用','','均为训练参与者的证据keep/弃权依赖诊断。不同K不能直接当成相同剂量的分类优劣，代理风险关系才是预注册主判据。','', '|角色|控制|平均K|弃权率|keep翻转率|','|---|---|---:|---:|---:|']
 quality=pd.read_csv(r.RES/'reliability_audit.csv');dep=quality[quality.kind=='GATE_DEPENDENCE'].copy();dep['abstain_bool']=dep.abstain.astype(str).str.lower().eq('true')
 for (role,control),g in dep.groupby(['role','control']):text.append(f'|{role}|{control}|{g.selected_count.mean():.4f}|{g.abstain_bool.mean():.4f}|{g.prediction_flip.mean():.5f}|')
 text+=['','### CF实际覆盖审计','','probe是逐人独立前向的尝试次数，unique是每fold五个内折合并后的实际人数，不能用总probe数冒充不同患者数。教师队列每fold固定16人，不代表覆盖全部outer-train。','','|组|十折尝试/有效probe总数|每fold unique人数范围|测试标签参与|','|---|---:|---|---:|']
 coverage=pd.DataFrame([x for x in rows if x['kind']=='CF_COVERAGE'])
 for group,g in coverage.groupby('group'):text.append(f'|{group}|{int(g.attempted_probes.sum())}/{int(g.valid_probes.sum())}|{int(g.unique_people.min())}–{int(g.unique_people.max())}|0|')
 zero=coverage[(coverage.group=='C3')&(coverage.valid_probes==0)].fold.astype(int).tolist()
 text+=['','特别记录：C3有效CF probe为0的fold='+str(zero)+'；这些fold没有满足B0原预测正确且置信≥.6的被抽样患者，按冻结规则跳过CF并保留teacher训练，不事后降低门槛。C1/C2和固定权重对照使用相同有效样本规则。']
 text+=['','Q4当前仅R2 FUNC通过，R4因而退化为单一FUNC门控，不构成PHENO+QC联合增益证据。FUNC支持限于训练内质量代理和证据keep/弃权路径，不等于B0正常分类提升。Q3每fold只有16名预先固定训练导师参与者、30步轻量探针；负结果只否定本次受控配置，不证明所有CF训练必然无效。','', '补充恢复验收首次失败源自审计克隆时Adam.load_state_dict复用了CPU step张量（第一次内存step30→31，第二次起点被污染）。修复为每个克隆深拷贝模型/Adam字典并恢复保存RNG；正式磁盘checkpoint和冻结训练代码未修改。runtime/qa_failure.json与原失败日志保留，重跑十折验收后才记录通过。']
 frame=pd.DataFrame([x for x in rows if x['kind']=='DISTRIBUTION_PROXY']);frame['distance_ratio']=frame.distance_ratio.astype(float)
 text+=['','|替换评分|到训练数据最近邻距离比（均值）|','|---|---:|']
 for key,g in frame.groupby('score'):text.append(f'|{key}|{g.distance_ratio.mean():.5f}|')
 with (r.RES/'report.md').open('a') as f:f.write('\n'.join(text)+'\n')
 r.verify_protected();info=r.jr(r.RES/'manifest.json');info['total_wall_seconds']=time.time()-info['storage']['time'];r.jw(r.RES/'manifest.json',info)
 with (r.RES/'report.md').open('a') as f:f.write(f"\n补充QA共10任务，GPU0/GPU1各5，单任务秒数合计{audit['gpu_task_seconds']}（并发，不是总墙钟）。含审计/编码/锚点校验/四链及补充QA的实际总墙钟为{info['total_wall_seconds']:.2f}秒。\n")
 info=r.jr(r.RES/'manifest.json');info['delivery']='POST_AUDIT_SYNC_PENDING';info['failure_modes']=dict(Q1='PRIOR_UNSUPPORTED; semantic increment not established',Q2='NO_INCREMENT_OVER_TOP_ABS_Z',Q3='NO_STABLE_OOF_CF_LOCALIZATION_GAIN',Q4='FUNC_PROXY_ONLY; PHENO_AND_ANAT_UNSUPPORTED');r.jw(r.RES/'manifest.json',info)
 print(json.dumps(dict(decisions=dec,scientific=info['scientific'],post_audit='COMPLETE'),ensure_ascii=False),flush=True)

def finish_delivery():
 """One bounded update of this experiment only; historical handoff append only."""
 r.verify_protected();info=r.jr(r.RES/'manifest.json');old=r.jr(r.RT/'storage_initial.json')['handoff'];target=r.ROOT/'experiment'/r.HERE.name;pending=r.HERE/'handoff_append_pending.md'
 info['total_wall_seconds']=time.time()-info['storage']['time'];info['post_audit']['final_script_sha']=m.sha(__file__)
 report=(r.RES/'report.md').read_text();marker='E42_FINAL_QA_'+info['frozen_signature']
 appendix='\n\n### Experiment 42 — 最终质量与交付验收补充\n\n'+marker+'\n\n'
 appendix+=f"四链结论{info['decisions']}；Q1–Q4各10折，共40/40诊断任务，GPU0/GPU1各20。补充QA各5任务，十折选择器下一Adam更新逐值复演通过。B0为10折验证复用，新增正式分类训练0折。最终追加前总墙钟{info['total_wall_seconds']:.2f}秒。\n\n"
 appendix+='Q1仅3/13假说稳定受支持；Q2选中删除0.079938、随机0.033208、Top-|z|0.097882，胜Top-|z|仅2/10折。Q3 C3对C0删除差−0.00010488，仅2/10折胜，未改善定位。Q4仅FUNC质量代理通过8/10折，PHENO/ANAT均0/10；没有多模态门控联合分类收益证明。结论DIAGNOSTIC_COMPLETE_NO_GO，不训练集成候选。\n\n'
 appendix+='公开更正诊断标签缺陷：R0/R2/R3不读取SITE，site-only置乱应恒等；原运行误用random rank，原始行保留，最终CSV按恒等性质重建并标注，训练与GO判定不变。补充恢复检查首次失败为克隆Adam CPU step字典别名，深拷贝和RNG恢复后通过，正式checkpoint未修改。\n\n'
 appendix+=f'最终工作目录{r.HERE}；正式交付目录{target}。初次/data健康检查EIO，主体同步时已恢复；本次对更新产物做有界SHA校验同步。保留/tmp唯一完整副本及恢复证据，不自动git操作。完整结果见results/report.md、manifest.json及各诊断CSV。\n'
 prior=copy.deepcopy(info.get('handoff',{}))
 try:
  with (r.ROOT/'handoff.md').open('a+b') as stream:
   fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);before=stream.read();assert len(before)>=old['bytes'] and hashlib.sha256(before[:old['bytes']]).hexdigest()==old['sha256']
   appended=marker.encode() not in before
   if appended:stream.write(appendix.encode());stream.flush();os.fsync(stream.fileno())
   stream.seek(0);after=stream.read();assert after[:len(before)]==before
   if appended:assert after[len(before):]==appendix.encode()
  info['handoff']=dict(status='APPENDED_PREFIX_VERIFIED',initial_prefix=old,preappend_bytes=len(before),preappend_sha256=hashlib.sha256(before).hexdigest(),new_sha256=hashlib.sha256(after).hexdigest(),final_qa_append=appended,prior_main_append=prior)
  if pending.exists():pending.unlink()
 except (OSError,AssertionError) as error:
  pending.write_text(appendix+'\n\n'+report+'\nAppend failure: '+repr(error)+'\n');info['handoff']=dict(status='PENDING_APPEND',error=repr(error),protected_prefix=old,prior_main_append=prior)
 info['result_sha']={p.name:m.sha(p) for p in r.RES.glob('*.csv')};info['delivery']='VERIFYING_UPDATED_STAGE';r.jw(r.RES/'manifest.json',info)
 def copy_checked(source,dest):
  if dest.exists() and m.sha(source)==m.sha(dest):return
  tmp=dest.with_name(dest.name+f'.e42sync-{os.getpid()}');shutil.copyfile(source,tmp)
  with tmp.open('rb') as stream:os.fsync(stream.fileno())
  assert m.sha(source)==m.sha(tmp);os.replace(tmp,dest);assert m.sha(source)==m.sha(dest)
 try:
  target.mkdir(parents=True,exist_ok=True)
  existing=target/'config.json'
  if existing.exists():assert m.sha(existing)==m.sha(r.HERE/'config.json'),'target experiment has a different config'
  count=0
  for source in sorted(r.HERE.rglob('*')):
   if source.is_symlink():continue
   dest=target/source.relative_to(r.HERE)
   if source.is_dir():dest.mkdir(exist_ok=True)
   elif source.is_file():copy_checked(source,dest);count+=1
  attempt=int(info.get('delivery_details',{}).get('bounded_sync_attempt',1))+1
  info['delivery']='COPIED_AND_HASH_VERIFIED';info['delivery_details']=dict(status=info['delivery'],target=str(target),verified_files=count,stage_preserved=True,bounded_sync_attempt=attempt)
 except (OSError,AssertionError) as error:info['delivery']='PARTIAL_TMP_ONLY';info['delivery_details']=dict(status=info['delivery'],error=repr(error),stage=str(r.HERE),target=str(target),stage_preserved=True)
 info['total_wall_seconds']=time.time()-info['storage']['time'];info['completion']='SCIENTIFIC_COMPLETE'
 with (r.RES/'report.md').open('a') as stream:stream.write('\n最终交付验收：'+json.dumps(info['delivery_details'],ensure_ascii=False)+'\n\nhandoff最终状态：'+json.dumps(info['handoff'],ensure_ascii=False)+f"\n\n最终交付验收时实际总墙钟：{info['total_wall_seconds']:.2f}秒。\n")
 r.jw(r.RES/'manifest.json',info)
 if info['delivery']=='COPIED_AND_HASH_VERIFIED':
  try:
   copy_checked(r.RES/'report.md',target/'results/report.md');copy_checked(r.RES/'manifest.json',target/'results/manifest.json')
   for source in r.HERE.rglob('*'):
    if source.is_file() and not source.is_symlink():assert m.sha(source)==m.sha(target/source.relative_to(r.HERE))
  except (OSError,AssertionError) as error:
   info['delivery']='PARTIAL_TMP_ONLY';info['delivery_details']['final_verify_error']=repr(error);r.jw(r.RES/'manifest.json',info)
 print(json.dumps(dict(scientific=info['scientific'],delivery=info['delivery'],handoff=info['handoff']['status'],total_seconds=info['total_wall_seconds']),ensure_ascii=False))

def all_audits():
 pending=[]
 for fold in range(1,11):
  path=r.RT/f'tasks/qa_{fold:02}.csv'
  if path.exists() and any(x.get('kind')=='SELECTOR_RESUME' for x in r.cr(path)):continue
  pending.append(fold)
 running=[]
 while pending or running:
  for fold in pending[:]:
   gpu=0 if fold<=5 else 1
   if sum(x[1]==gpu for x in running)>=2:continue
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);log=(r.RT/f'logs/qa_{fold:02}.log').open('a')
   process=subprocess.Popen([sys.executable,__file__,'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT)
   running.append((fold,gpu,process,log));pending.remove(fold)
  for item in running[:]:
   fold,gpu,process,log=item
   if process.poll() is not None:
    log.close();running.remove(item)
    if process.returncode:
     for _,_,other,stream in running:other.terminate();stream.close()
     raise RuntimeError(f'QA fold {fold} failed, see runtime/logs/qa_{fold:02}.log')
    print('QA completed',fold,'GPU',gpu,flush=True)
  if running:time.sleep(1)
 complete()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int);ap.add_argument('--all',action='store_true');ap.add_argument('--deliver',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
 if args.fold:fold_audit(args.fold)
 elif args.deliver:finish_delivery()
 elif args.all:all_audits()
 else:complete()
