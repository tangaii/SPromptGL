#!/usr/bin/env python3
"""Post-training verification and fixed-model auxiliary-refinement intervention.

This script never optimizes parameters or changes the registered experiment.
It complements branch-off diagnostics by disabling only the auxiliary reweighting.
"""
import argparse
import copy
import fcntl
import json
import os
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import run as r


def auxiliary_audit():
    import torch
    path=r.RT/'aux_refinement_audit.json'
    tasks=[r.read(r.RT/'tasks'/f'{r.GROUPS[6]}_seed0_fold{f:02}.json') for f in range(1,11)]
    signature={'training_code':r.codehash(),'bests':[t['metric']['best_checkpoint_sha256'] for t in tasks]}
    if path.exists():
        value=r.read(path)
        assert value['signature']==signature
        return value
    rows=[]
    for fold,task in enumerate(tasks,1):
        agent,_=r.make(6,fold);best=r.load(task['metric']['best_checkpoint'])
        assert r.sha(task['metric']['best_checkpoint'])==signature['bests'][fold-1]
        r.setstates(agent,best['model']);g=agent.MessagePassing0
        original=r.evaluate(agent);out,graphs=original[0],original[2]
        q=g.e43_q.copy();te=agent.tst_idx;pred=out[te].argmax(1)
        for mode in ['no_aux','uniform_aux','shuffled_aux']:
            g.e43_group=8 if mode=='shuffled_aux' else 6
            g.e43_q=np.zeros(3) if mode=='no_aux' else np.ones(3)/3 if mode=='uniform_aux' else q.copy()
            changed=r.evaluate(agent);new,ag=changed[0],changed[2]
            assert all(torch.equal(a>0,b>0) for a,b in zip(graphs,ag))
            rows.append(dict(fold=fold,intervention=mode,
                abs_logp_change=float((out[te]-new[te]).abs().mean()),
                class_flip_fraction=float((pred!=new[te].argmax(1)).float().mean()),
                changed_predictions=int((pred!=new[te].argmax(1)).sum()),
                test_subjects=len(te),
                auxiliary_edge_change=float(torch.stack([(a-b).abs().mean() for a,b in zip(graphs[:3],ag[:3])]).mean()),
                corr_support_changed=0))
        g.e43_group=6;g.e43_q=q
        restored=r.evaluate(agent)
        assert torch.equal(restored[0],out) and all(torch.equal(a,b) for a,b in zip(restored[2],graphs))
        assert r.equal(r.states(agent),best['model'])
        del agent
    value={'signature':signature,'rows':rows,'restore_bitwise':True,
           'scope':'fixed E06 best weights; only auxiliary refinement or its correspondence changes; no retraining'}
    r.write(path,value)
    return value


def finalize():
    from sklearn.metrics import roc_auc_score
    started=time.monotonic();m=r.read(r.HERE/'manifest.json')
    assert m['status']=='COMPLETE' and m['anchor_replay']=='ANCHOR_REPLAY_PASS'
    assert m['code_sha']==r.codehash();r.verify_sources()
    rows=r.csvread(r.RESULTS/'fold_metrics.csv');d=r.data()
    keys={(x['group'],int(x['seed']),int(x['fold'])) for x in rows}
    assert len(rows)==len(keys)==100
    assert keys=={(g,0,f) for g in r.GROUPS for f in range(1,11)}
    assert [sum(int(x['physical_gpu'])==i for x in rows) for i in [0,1]]==[50,50]
    counts={};param_counts=None
    for row in rows:
        f=int(row['fold']);g=row['group'];bestpath=row['best_checkpoint']
        assert r.sha(bestpath)==row['best_checkpoint_sha256']
        best=r.load(bestpath);pred=np.asarray(best['pred'],dtype=np.int64);tr,te=d['folds'][f-1];y=d['y'][te]
        anchor=r.load(r.RT/'anchors'/f'anchor_fold{f:02}.pt')
        for component in ['gfr','pln']:
            assert best['model'][component].keys()==anchor['model'][component].keys()
            assert all(best['model'][component][name].shape==v.shape for name,v in anchor['model'][component].items())
        tp=int(((y==0)&(pred==0)).sum());fn=int(((y==0)&(pred==1)).sum())
        fp=int(((y==1)&(pred==0)).sum());tn=int(((y==1)&(pred==1)).sum())
        assert [tp,fn,fp,tn]==[int(row[k]) for k in ['asd_tp','asd_fn','td_fp','td_tn']]
        calculated=[float((pred==y).mean()),float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred])),tn/(tn+fp),tp/(tp+fn)]
        assert all(abs(a-float(row[k]))<1e-14 for a,k in zip(calculated,['acc','hard_auc','spe_td','sen_asd']))
        assert row['initial_trunk_sha256']==r.historical()[f]['initial_trunk_sha256']
        assert row['initial_rng_sha256']==r.historical()[f]['initial_rng_sha256']
        if g==r.GROUPS[0]:assert not r.compare_anchor(row)
        assert row['prediction_sha256']==r.digest(pred.tobytes())
        counts.setdefault(g,np.zeros(4,dtype=np.int64));counts[g]+=np.array([tp,fn,fp,tn])
        if param_counts is None:
            agent,_=r.make(0,1)
            param_counts={k:sum(p.numel() for p in mod.parameters()) for k,mod in [('GFR',agent.MessagePassing0),('PLN',agent.ModalFusion)]}
            del agent
    for f in range(1,11):
        tr,te=d['folds'][f-1];rel=r.reliability(f)
        for inner in rel['inner_records']:
            fit=set(inner['fit_subjects']);val=set(inner['validation_subjects'])
            assert not fit&val and fit|val==set(tr) and not (fit|val)&set(te)
    aux=auxiliary_audit()
    diags=r.csvread(r.RESULTS/'diagnostics.csv')
    for row in diags:
        f=int(row['fold']);n=len(d['folds'][f-1][0]);row['outer_train_subjects']=n
        row['supervised_unordered_train_pairs']=n*(n-1)//2 if row['group']==r.GROUPS[2] else 0
        if row['group']==r.GROUPS[6]:
            for effect in [a for a in aux['rows'] if a['fold']==f]:
                for key in ['abs_logp_change','class_flip_fraction','auxiliary_edge_change']:
                    row['fixed_'+effect['intervention']+'_'+key]=effect[key]
    assert len(diags)==400
    r.csvwrite(r.RESULTS/'diagnostics.csv',diags)
    evidence={'unique_folds':100,'four_metrics_recomputed_from_saved_predictions':True,
              'all_best_sha_verified':True,'same_fold_initial_sha_rng_verified':True,
              'inner_subject_disjoint_outer_test_excluded':True,'parameters_same_registry_all_groups':param_counts,
              'aggregate_confusions':{g:v.tolist() for g,v in counts.items()},
              'supervised_pairs_by_fold':{str(f):len(d['folds'][f-1][0])*(len(d['folds'][f-1][0])-1)//2 for f in range(1,11)},
              'resume_events':sum(int(x['resumes']) for x in rows),
              'device_audits':[r.read(r.RT/f'device_audit_{g}.json') for g in [0,1]],
              'aux_refinement':aux,'audit_seconds':time.monotonic()-started}
    assert evidence['device_audits'][0]['graphs_sha256']==evidence['device_audits'][1]['graphs_sha256']
    r.write(r.RT/'final_check.json',evidence)
    marker='<!-- EXPERIMENT43_AUXILIARY_FIXED_MODEL_AUDIT -->'
    extra=['',marker,'## 完成后的独立核对与辅助权重干预','',
        '100/100唯一折、两卡50/50、所有best SHA、四项指标与保存预测/混淆均重新核对通过。保护源码与输入SHA保持不变；两物理设备初始校准四图逐位相同。未增加任何训练组或修改训练公式。',
        '全部组保持同一参数注册：'+json.dumps(param_counts,ensure_ascii=False)+'；E09部分辅助GraphLearn无有效梯度是预注册控制的实际结果。',
        'E02监督无序训练配对数：'+json.dumps(evidence['supervised_pairs_by_fold'])+'；新增监督不接收外层test标签。E05诊断中估计q只供参照，未用于其前向；E07实际q固定1/3。',
        '关闭整条GCN消息不能单独证明辅助关系有用。因此另外固定E06最佳模型，仅将q设0、设等权或置乱辅助C，保留所有GCN和CORR支持。恢复配置后输出、图和模型参数逐值恢复；结果如下。此处log-prob/翻转仅为机制诊断。',
        '|fold|仅关闭辅助精炼：平均绝对log-prob变化|类别翻转数/测试数|','|---|---:|---:|']
    for a in aux['rows']:
        if a['intervention']=='no_aux':extra.append(f"|{a['fold']}|{a['abs_logp_change']:.8g}|{a['changed_predictions']}/{a['test_subjects']}|")
    no_aux=[a for a in aux['rows'] if a['intervention']=='no_aux']
    extra+=['',f"仅关闭辅助精炼：十折平均绝对log-prob变化={np.mean([a['abs_logp_change'] for a in no_aux]):.9g}，合计类别翻转{sum(a['changed_predictions'] for a in no_aux)}/871。边权调制实现成立，但这种弱响应不能证明有独立分类价值；不能用非零q或梯度替代负对照。"]
    extra+=['','|组|TP-ASD|FN-ASD|FP-TD|TN-TD|正确总数（混淆计数）|','|---|---:|---:|---:|---:|---:|']
    for g,v in counts.items():extra.append('|'+g+'|'+'|'.join(map(str,v))+'|'+str(v[0]+v[3])+'|')
    extra+=['','上述为871人的混淆合计；主表仍是十折均值，并未将折均值替换为微平均。',
        '审计源码：audit.py；详细证据：.runtime/final_check.json及diagnostics.csv。再生成主报告后可运行`CUDA_VISIBLE_DEVICES=1 python audit.py`恢复本附录。']
    science_marker='<!-- EXPERIMENT43_SCIENTIFIC_INTERPRETATION -->'
    science=['',science_marker,'## 四个问题的科学判断','',
        '1. 分支覆盖：原E00的PHENO消息分类梯度和关闭效应均严格为0，E01恢复四支有效梯度。但E01相对E00的ACC/Hard-AUC为−0.4585/−0.4327pp；修复实现问题并未带来分类增益。',
        '2. 监督图项：E02的阈值前MSE确实作用于四个GraphLearn投影，新增监督只包含train-train无序非对角配对；相对E00为−0.4598/−0.4441pp。有效监督梯度没有转化成性能提升，它也不是原仓库未优化ALoss的数值复现。',
        '3. 密度与传播：E03相对E00为−0.2299/−0.1911pp；E05相对E01为−0.0013/+0.0048pp，未见同时改善。E04相对E01为−0.2299/−0.2026pp，不支持仅传播对应模态。原全320维传播是原设计，本轮不能将其定性为错误。',
        '4. CORR锚定可靠性：E06相对E00为−0.5747/−0.5307pp，相对E09纯CORR为−0.3448/−0.3503pp；相对等权E07、错配E08的ACC均无均值增益。全部主要比较均未达到>0.1pp和至少7折胜的门槛，结论NO-GO。',
        '原E00四图平均邻居191.100/330.582/155.409/93.219；E06四图均为48.470，辅助精炼没有改变CORR支持。四个GCN消息可有非零分类梯度，但E06各分支贡献相对输入平均仅约2.74e−5/1.83e−5/2.76e−6/4.19e−6；仅关闭辅助精炼无类别翻转，不能据此宣称跨关系互补。',
        '固定模型干预与从头训练消融回答不同问题，不能以前者替代E07–E09。强主干残差、归一化对边权改动的压缩和原权重衰减都可能参与消息衰减，但本轮未分别验证这些原因，不作因果归因。建议保留B0-A强基座，先定位消息传播的实际决策贡献，避免继续堆叠门控或将微小代理AUC增量包装成可靠医学关系。',
        '本轮没有Prompt或疾病语义新实验。以上均为单seed历史val=test、CV前RFE/标准化及transductive协议下的内部比较；不支持独立泛化、显著性或临床因果主张。']
    extra+=science
    text='\n'.join(extra)+'\n';report=r.RESULTS/'report.md'
    original=report.read_text().split(marker)[0].rstrip()+'\n'
    report.write_text(original+text)
    target=r.ROOT/'handoff.md';pending=r.HERE/'handoff_audit_append_pending.md';pending.write_text(text)
    try:
        prefix=r.read(r.RT/'handoff_prefix.json')
        with target.open('ab') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);before=target.read_bytes()
            assert r.digest(before[:prefix['bytes']])==prefix['sha256']
            if marker.encode() not in before:
                stream.write(text.encode());stream.flush();os.fsync(stream.fileno());after=target.read_bytes()
                assert after==before+text.encode()
            elif science_marker.encode() not in before:
                addition=('\n'.join(science)+'\n').encode()
                stream.write(addition);stream.flush();os.fsync(stream.fileno());after=target.read_bytes()
                assert after==before+addition
            else:after=before
        pending.unlink();handoff={'status':'APPENDED_OR_ALREADY_PRESENT_PREFIX_VERIFIED','before_bytes':len(before),'after_bytes':len(after),'after_sha256':r.digest(after)}
    except Exception as exc:handoff={'status':'FAILED_PENDING','error':repr(exc),'path':str(pending)}
    m=r.read(r.HERE/'manifest.json');m.update(final_audit=evidence,final_audit_source_sha256=r.sha(__file__),handoff_audit=handoff,
        audit_finalized=r.now(),end_to_end_wall_seconds=time.time()-m['started_unix'])
    m['result_sha']={p.name:r.sha(p) for p in r.RESULTS.iterdir() if p.is_file()}
    if handoff['status']=='FAILED_PENDING':m['delivery']='PARTIAL'
    r.write(r.HERE/'manifest.json',m)
    print('FINAL_AUDIT_PASS',100,m['gpu_counts'],handoff,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--aux-only',action='store_true');args=parser.parse_args()
    if args.aux_only:print(json.dumps(auxiliary_audit(),ensure_ascii=False),flush=True)
    else:finalize()
