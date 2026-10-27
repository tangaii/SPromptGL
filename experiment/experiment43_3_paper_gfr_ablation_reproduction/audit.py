#!/usr/bin/env python3
"""Independent final formulas, best-model readback and training/mechanism audit.

This read-only model audit is deliberately separate from frozen training run.py.
It never changes a checkpoint, training configuration or historical file.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,time
import numpy as np
import run as r

def worker(gpu):
    t=r.runtime()['torch'];checks=[];details=[];started=time.monotonic()
    for group in r.GROUPS:
        for f in (range(1,6) if gpu==0 else range(6,11)):
            row=r.read(r.RT/'tasks'/f'{group}_f{f:02}.json')
            assert row['status']=='COMPLETE' and row['independent_acceptance']=='PASS'
            assert r.sha(row['best_checkpoint'])==row['best_sha']
            best=r.load(row['best_checkpoint']);a,s=r.fromanchor(group,f);r.setmodel(a,best['model'])
            out,cl,gl,_,_,_=r.evaluate(a);pred=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64);target=r.data()[1][a.tst_idx]
            tp=int(((target==0)&(pred==0)).sum());fn=int(((target==0)&(pred==1)).sum());fp=int(((target==1)&(pred==0)).sum());tn=int(((target==1)&(pred==1)).sum())
            independent=dict(acc=(tp+tn)/(tp+fn+fp+tn),hard_auc=.5*(tp/(tp+fn)+tn/(tn+fp)),spe=tn/(tn+fp),sen=tp/(tp+fn),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)
            assert all(abs(independent[k]-row[k])<1e-12 for k in independent)
            assert hashlib.sha256(pred.tobytes()).hexdigest()==row['prediction_sha256']
            assert r.sha(r.anchorpath(group,f))==row['anchor_sha'] and r.rngsha(s['rng'])==row['post_initial_rng_sha256']
            labels=a.targ.clone();a.targ[a.tst_idx]=1-a.targ[a.tst_idx];assert t.equal(out,r.evaluate(a)[0]);a.targ=labels
            if r.GROUPS[group]['base']=='A':
                for name,v in s['model']['pln'].items():
                    if name.startswith('MLP.'):assert t.equal(v,best['model']['pln'][name]),'Idle PEN parameter changed'
            if not a.gfr:
                assert a.gfr_calls==a.gcn_calls==a.graph_calls==0 and r.equal(s['model']['gfr'],best['model']['gfr'])
                if 'mp' in best:assert r.equal(s['mp'],best['mp'])
            trpred=out[a.trn_idx].argmax(1);trainacc=float(trpred.eq(a.targ[a.trn_idx]).double().mean())
            trace=row.get('loss_trace',[])
            if not trace and row.get('source_history') and r.Path(row['source_history']['source']).exists():
                old=r.read(row['source_history']['source']);trace=old.get('loss_trace',old.get('history',[]))
            getcls=lambda q:q.get('cls',q.get('cls_loss'))
            getgraph=lambda q:q.get('graph',q.get('graph_loss'))
            curve=[q for q in trace if getcls(q) is not None]
            common=dict(group=group,fold=f,train_eval_acc=trainacc,test_eval_acc=independent['acc'],train_test_eval_gap_pp=100*(trainacc-independent['acc']),eval_cls_loss=float(cl),eval_graph_loss=float(gl),eval_abs_graph_over_cls=float(abs(gl)/(cl+1e-12)),best_epoch=row['best_epoch'],epochs_executed=row['epochs_executed'],early_stop='wait>200' if row['epochs_executed']<450 else '450 cap',train_predicted_asd=int((trpred==0).sum()),test_predicted_asd=int((pred==0).sum()),test_prediction_collapsed=len(np.unique(pred))==1,curve_points=len(curve),first_cls=getcls(curve[0]) if curve else None,last_cls=getcls(curve[-1]) if curve else None,first_graph=getgraph(curve[0]) if curve else None,last_graph=getgraph(curve[-1]) if curve else None)
            if a.gfr:
                _,cl2,gl2,_,_,_=r.forward(a)
                ps=[p for i in range(4) for p in getattr(a.MessagePassing0,f'MessagePassing{i}').parameters()]
                gg=t.autograd.grad(gl2,ps,allow_unused=True)
                assert all(g is None or not bool(g.any()) for g in gg),'Graph loss unexpectedly supervises GCN weights'
                for i,name in enumerate(['PHENO','ANAT','FUNC','CORR']):
                    module=getattr(a.MessagePassing0,f'MessagePassing{i}');prefix=f'MessagePassing{i}.'
                    startnorm=float(sum((v.float().square().sum() for k,v in s['model']['gfr'].items() if k.startswith(prefix)),t.tensor(0.)).sqrt())
                    bestnorm=float(sum((p.square().sum() for p in module.parameters()),a.feat.new_zeros(())).sqrt())
                    details.append(dict(**common,modality=name,graph_loss_GCN_gradient_norm=0.,gcn_initial_parameter_norm=startnorm,gcn_best_parameter_norm=bestnorm,gcn_parameter_norm_ratio=bestnorm/startnorm))
            else:details.append(dict(**common,modality='RAW',graph_loss_GCN_gradient_norm=0.,gcn_parameter_norm_ratio=1.))
            checks.append(dict(group=group,fold=f,status='PASS',metric_formula='independent boolean confusion and hard-AUC identity',initial_and_best_sha=True,heldout_label_forward_bitwise=True,idle_PEN_unchanged=True));print('FINAL_FORMULA_ACCEPT',group,f,flush=True)
            del a;t.cuda.empty_cache()
    r.write(r.RT/f'independent_science_gpu{gpu}.json',dict(status='PASS',checks=checks,details=details,seconds=time.monotonic()-started,peak_allocated_bytes=t.cuda.max_memory_allocated(),audit_source_sha=r.sha(r.EXP/'audit.py')))

def finalize():
    m=r.read(r.EXP/'manifest.json');rows=[];checks=[]
    ha_code={n:r.sha(r.HA/n) for n in ['run.py','method.py','integration.py','config.json','integration.patch']}
    assert ha_code==r.read(r.HA/'manifest.json')['code_sha']
    hb_paths=[r.HB/n for n in ['run.py','integration.py','config.json']]+list((r.HB/'.runtime/sources').glob('*.py'))
    hb_signature=hashlib.sha256(json.dumps({str(p.relative_to(r.HB)):r.sha(p) for p in hb_paths},sort_keys=True).encode()).hexdigest()
    assert hb_signature==r.read(r.HB/'manifest.json')['code_signature']
    for f in range(1,11):
        for g in [0,1]:assert r.read(r.HB/'.runtime/tasks'/f'g{g}_f{f:02}.json')['signature']==hb_signature
    for gpu in [0,1]:
        v=r.read(r.RT/f'independent_science_gpu{gpu}.json');assert v['status']=='PASS' and len(v['checks'])==40;rows+=v['details'];checks+=v['checks']
    assert len({(q['group'],q['fold']) for q in checks})==80
    base=r.csvread(r.RESULTS/'mechanism_diagnostics.csv');detail={(q['group'],q['fold'],q['modality']):q for q in rows}
    for q in base:
        key=(q['group'],int(q['fold']),q['modality']);assert key in detail;q.update(detail[key])
    r.csvwrite(r.RESULTS/'mechanism_diagnostics.csv',base)
    text=['','## 独立公式终验、训练曲线与弱消息来源','', '额外audit.py为与冻结训练入口分离的独立验收实现：用布尔混淆计数重算ACC/Hard-AUC/SPE/SEN，跨物理GPU重载全部80个best；检查初始化/预测/最佳模型SHA、no-GFR零调用/零更新、闲置PEN MLP不更新及heldout标签置换不改变forward。80/80通过；没有训练或修改原模型。','', '|组|最佳epoch均值|执行epoch均值|训练eval ACC|测试eval ACC|训练−测试pp|初始→末次分类loss|最佳图loss/分类loss绝对比|预测坍塌折|','|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for group in r.GROUPS:
        rr=[q for q in rows if q['group']==group and q['modality'] in ['PHENO','RAW']]
        avg=lambda k:float(np.mean([q[k] for q in rr]))
        text.append(f"|{group}|{avg('best_epoch'):.1f}|{avg('epochs_executed'):.1f}|{100*avg('train_eval_acc'):.4f}|{100*avg('test_eval_acc'):.4f}|{avg('train_test_eval_gap_pp'):.4f}|{avg('first_cls'):.5f}→{avg('last_cls'):.5f}|{avg('eval_abs_graph_over_cls'):.4f}|{sum(q['test_prediction_collapsed'] for q in rr)}|")
    text+=['','训练eval ACC是最佳checkpoint的诊断，非新增第五项正式指标。训练loss轨迹为train-mode、按原保存频率抽样，初始/末次不是同一dropout mask；不能把此表当作无偏过拟合估计。图loss包含负degree项，因此另列绝对比，不能把负总loss误判数值故障。','', '|有GFR组|ANAT GCN权重范数(best/init)|FUNC|CORR|GCN图正则梯度|','|---|---:|---:|---:|---:|']
    for group in ['H-A-G','H-B-G','P-A-G','P-B-G']:
        ratios=[np.mean([q['gcn_parameter_norm_ratio'] for q in rows if q['group']==group and q['modality']==mod]) for mod in ['ANAT','FUNC','CORR']]
        text.append('|'+group+'|'+'|'.join(f'{x:.7g}' for x in ratios)+'|0（逐张量独立核验）|')
    text+=['','这些范数收缩和弱消息是实际观察；GFR的GraphConstructLoss只作用于构图投影，不直接监督GCN权重。原weight_decay=.11是候选机制解释，但本轮未新增去正则组，不能将收缩原因单独归因于weight decay。PHENO GCN因覆盖无分类梯度，其消息不进入最终输出。','', '补充确定源码/数据证据：官方固定commit的预处理CSV及modal_feat_dict.npy与本地逐字节相同。用户远端main七个核心文件也与本地相同。官方PEN硬编码CAM4×36，而论文代理10×4产生4×40，不能直接执行当前公开PEN；本轮两个基座都跳过PEN，因此代理四组维度合法。此证据限制论文配置精确映射，不影响冻结H/P。Eq4的CAM残差在InputLayer硬编码系数1，hyperpm.alpha未传入；GFR raw-X系数1是另一条独立skip。','', 'H明确指公开Shell超参配合用户指定的历史A/B构造，不声称E32缓存式构造等于官方注册整个RN50的原始构造。强行换成后者会破坏本轮要求的E32-B0-A身份，因此仅做源码对照并保留未公开消融实现的不确定性。官方与用户都存在PHENO覆盖及ALoss未优化，不能将这两点说成用户独有改坏。','', '### A–F结论','']
    pairs=r.csvread(r.RESULTS/'paired_comparisons.csv');stats={q['group']:q for q in r.csvread(r.RESULTS/'summary.csv')}
    for protocol,title in [('H','A. 公开Shell/用户兼容初始化协议'),('P','B. 论文文本代理协议')]:
        q=[q for q in pairs if q['kind']=='GFR' and q['lhs'].startswith(protocol)]
        text.append(title+'：'+'；'.join(f"B0-{x['lhs'].split('-')[1]} ΔACC={float(x['delta_acc_pp']):+.4f}pp、ΔAUC={float(x['delta_hard_auc_pp']):+.4f}pp，{x['decision']}" for x in q)+'。')
    raw=[float(stats[g]['acc_mean']) for g in ['H-A-R','H-B-R','P-A-R','P-B-R']];base_deltas=[abs(float(q['delta_acc_pp'])) for q in pairs if q['kind']=='BASE']
    explanation='该量级不足以解释' if max(base_deltas)<5.25 else '这是跨初始化比较，仍不等于'
    text+=['',f"C. 本轮A/B差异中最大ACC绝对差为{max(base_deltas):.4f}pp；{explanation}原论文5.25pp的同基座GFR增量。两者本来不执行PEN，不能归因于疾病文本。",'',f"D. CAM-only实测范围{min(raw):.4f}–{max(raw):.4f}%；与86.48最近差距{min(abs(x-86.48) for x in raw):.4f}pp。是否数值接近应看这个差距，不能用不同协议组拼接。",'', 'E. 确定差异：正文/Shell超参不一致；代码n_hidden语义与自然语言不一致；官方/用户RN50注册、构造顺序、缓存和初始化遍历不同；公式图监督与实际loss不同；论文未公开消融专用脚本/种子/完整初始化。上述证据并未证明其中某一项单独造成论文差距。评估偏倚与原Torch1.12/RTX3090环境也已披露，不能在未执行反事实配置时断言为根因。','', 'F. 研究建议：继续把GFR作为历史对照可以，但任何以患者图为创新基座的工作都需同时对CAM-only证明增量，并检查实际消息与初始化。若配对四组未达门槛，本轮不给“GFR具有稳定额外分类价值”的GO，也不据此证明全部患者图无效。','']
    common=0
    for f in range(1,11):
        v=[r.read(r.HB/'.runtime/tasks'/f'g{g}_f{f:02}.json')['rng_trace'] for g in [0,1]];n=min(map(len,v));assert v[0][:n]==v[1][:n];common+=n
    text+=['',f'H-B历史G/R共同epoch完整四类RNG又逐项核对：{common}个全部相同。P-A/P-B共同epoch trace由run.py report逐项核对；H-A历史未保存逐epoch完整trace，不伪称具有该层证据。A/B两基座的入口状态各自匹配，跨基座不作逐参数相同主张。','']
    block='\n'.join(text)+'\n';marker='<!-- E43_3_INDEPENDENT_SCIENCE -->';report=r.RESULTS/'report.md';old=r.bytesread(report).decode()
    if marker not in old:r.atomic(report,(old+'\n'+marker+'\n'+block).encode())
    # A second append is intentional: original handoff is never rewritten and the
    # quantitative independent acceptance has its own idempotent marker.
    pending=r.EXP/'handoff_append_pending.md'
    try:
        import fcntl
        with (r.ROOT/'handoff.md').open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();prefix=m['handoff_prefix'];assert hashlib.sha256(before[:prefix['bytes']]).hexdigest()==prefix['sha256']
            oldsha=hashlib.sha256(before).hexdigest()
            if marker.encode() not in before:
                core=''
                if b'<!-- EXPERIMENT43_3_PAPER_GFR_ABLATION -->' not in before:
                    assert pending.exists(),'Main handoff summary absent without pending record'
                    core=r.bytesread(pending).decode()
                f.seek(0,2);f.write((core+'\n'+marker+'\n'+block).encode());f.flush();os.fsync(f.fileno());f.seek(0);after=f.read();assert after[:len(before)]==before
            else:after=before
            assert b'<!-- EXPERIMENT43_3_PAPER_GFR_ABLATION -->' in after
        pending.unlink(missing_ok=True)
        handoff=dict(status='APPENDED_AND_INDEPENDENT_AUDIT_APPENDED',prefix_sha256=oldsha,after_sha256=hashlib.sha256(after).hexdigest())
    except Exception as e:
        earlier=r.bytesread(pending).decode() if pending.exists() else ''
        r.atomic(pending,(earlier+'\n'+marker+'\n'+block+'\nAppend error: '+repr(e)).encode());handoff=dict(status='PENDING',error=repr(e))
    r.verify_protected();m=r.read(r.EXP/'manifest.json');m.update(independent_crosscheck='PASS_80_OF_80',audit_source_sha=r.sha(r.EXP/'audit.py'),handoff=handoff,delivery='PARTIAL' if handoff['status']=='PENDING' else 'COMPLETE',scientific='COMPLETE',status='COMPLETE' if handoff['status']!='PENDING' else 'COMPLETE_WITH_DELIVERY_BLOCKED')
    m['historical_code_signatures_reverified']={'HA':ha_code,'HB':hb_signature};m['H_B_common_epoch_RNG_reverified']=common;m['result_sha256']={str(q.relative_to(r.EXP)):r.sha(q) for q in (r.RESULTS).glob('*') if q.is_file()};m['finished_unix']=time.time();m['total_wall_seconds']=m['finished_unix']-m['started_unix'];r.write(r.EXP/'manifest.json',m)
    print(block,flush=True)

def tables():
    """Render existing audit facts; no model loading, training or new evaluation."""
    import ast, fcntl
    m=r.read(r.EXP/'manifest.json')
    expected={'worker':'8b7666a98cbf38442b8cd5a7840bc07686d41079038194184e1f68fcaa886329','finalize':'67537f1ee573ce710d7caf70f6675b079181fbea5bb1d96c35e50f89fec20e7c'}
    actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse((r.EXP/'audit.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in expected}
    assert actual==expected,'The completed independent audit functions must remain unchanged'
    facts=r.read(r.EXP/'paper_ablation_audit.json')
    text=['','## 表四：论文原文、公开Shell及唯一代理映射','','|字段|论文正文|H Shell/default|P代理|源码含义及证据状态|','|---|---|---|---|---|']
    for q in facts['parameter_mapping']:
        vals=[q['field'],q['paper_text_original'],str(q['H']),str(q['P']),q['mapping_evidence']+'; '+q['mapping_confidence']+('; ASSUMED_MAPPING' if q['assumed_mapping'] else '')]
        text.append('|'+ '|'.join(str(v).replace('|','/').replace('\n',' ') for v in vals)+'|')
    rows=r.csvread(r.RESULTS/'mechanism_diagnostics.csv')
    text+=['','## 表五补充：有效消息、分类梯度与图正则路径','','|有GFR组|输入范数|输出范数|相对差异%|ANAT/FUNC/CORR分类GCN梯度范数|ANAT/FUNC/CORR best/init权重范数|关闭消息翻转|','|---|---:|---:|---:|---|---|---:|']
    for group in ['H-A-G','H-B-G','P-A-G','P-B-G']:
        rr=[q for q in rows if q['group']==group]
        avg=lambda mod,key:float(np.mean([float(q[key]) for q in rr if q['modality']==mod]))
        gradients=', '.join(f"{avg(mod,'classification_gcn_grad_norm'):.6g}" for mod in ['ANAT','FUNC','CORR'])
        shrink=', '.join(f"{avg(mod,'gcn_parameter_norm_ratio'):.6g}" for mod in ['ANAT','FUNC','CORR'])
        flips=sum(int(q['all_off_flip_count']) for q in rr if q['modality']=='PHENO')
        text.append(f"|{group}|{avg('PHENO','input_norm'):.7g}|{avg('PHENO','gfr_output_norm'):.7g}|{100*avg('PHENO','gfr_minus_raw_relative'):.7g}|{gradients}|{shrink}|{flips}/871|")
    text+=['','原始图的密度、邻居数及各分支关闭效应完整列于mechanism_diagnostics.csv。GraphConstructLoss→GCN参数梯度均为0；该loss优化构图投影，不能据此认为全部GFR参数都受到图正则直接训练。P最佳模型消息接近数值零，但从头训练G/R轨迹仍可能不同；不能把最终弱消息倒推为整个训练从未用图。','',f"终验状态：80/80折，50 NEWLY_TRAINED、30 REUSED_VERIFIED；GPU0/GPU1各25新增折。正式训练与第一轮验收墙钟{m['formal_wall_seconds']:.2f}s；含论文/代码审计及独立公式终验{m['total_wall_seconds']:.2f}s（截至此前独立终验）。本附表只整理既有记录，没有再次训练或评估。",'']
    block='\n'.join(text)+'\n';marker='<!-- E43_3_FINAL_AUDIT_TABLES -->';report=r.RESULTS/'report.md'
    before=r.bytesread(report).decode()
    if marker not in before:r.atomic(report,(before+'\n'+marker+'\n'+block).encode())
    try:
        with (r.ROOT/'handoff.md').open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();prefix=m['handoff_prefix']
            assert hashlib.sha256(before[:prefix['bytes']]).hexdigest()==prefix['sha256']
            if marker.encode() not in before:
                f.seek(0,2);f.write(('\n'+marker+'\n'+block).encode());f.flush();os.fsync(f.fileno())
            f.seek(0);after=f.read();assert after[:len(before)]==before
        m['handoff']['status']='APPENDED_ALL_AUDITS_PREFIX_VERIFIED';m['handoff']['after_sha256']=hashlib.sha256(after).hexdigest()
    except Exception as e:
        pending=r.EXP/'handoff_append_pending.md';old=r.bytesread(pending).decode() if pending.exists() else ''
        r.atomic(pending,(old+'\n'+marker+'\n'+block+'\nAppend failure: '+repr(e)).encode());m.update(delivery='PARTIAL',status='COMPLETE_WITH_DELIVERY_BLOCKED');m['handoff'].update(status='PENDING',error=repr(e))
    r.verify_protected();m['audit_source_before_tables_sha']=m['audit_source_sha'];m['audit_source_sha']=r.sha(r.EXP/'audit.py');m['independent_audit_functions_unchanged']=actual
    m['result_sha256']={str(q.relative_to(r.EXP)):r.sha(q) for q in r.RESULTS.glob('*') if q.is_file()};m['finished_unix']=time.time();m['total_wall_seconds']=m['finished_unix']-m['started_unix'];r.write(r.EXP/'manifest.json',m)
    print(block,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['worker','finalize','tables']);ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args()
    {'worker':lambda:worker(a.gpu),'finalize':finalize,'tables':tables}[a.command]()
