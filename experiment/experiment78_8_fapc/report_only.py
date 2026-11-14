#!/usr/bin/env python3
"""Report-only enrichment after the frozen independent audit has passed.

This script does not optimize, alter training source, or write checkpoints.
It recomputes compact mechanism diagnostics from independently verified best
CAM144 snapshots and writes final report/handoff summaries.
"""
from __future__ import annotations
import calendar, csv, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run as r
import audit as a

HERE, RESULTS = r.HERE, r.RESULTS
METRICS = ('acc','hard_auc','prob_auc','sen','spe','asd_f1')

def serialized_checkpoint_replay():
    """Use a real completed task and a disk round trip without editing it."""
    group=r.GROUPS[1]; fold=1
    state,_,_=r.prepare(fold); support=r.support_ids(state)
    module=r._make_module(state,group); opt=r._new_optimizer(module)
    latest=torch.load(r.task_paths(group,fold)[2],map_location='cpu',weights_only=False)
    r._restore(state,module,opt,latest['current'])
    first=r._replay_step(state,module,opt,support)
    replay_file=HERE/'runtime/serialized_resume_probe.pt'
    try:
        r.atomic_torch(replay_file,first['snapshot'])
        checkpoint_sha=r.sha_file(replay_file)
        expected=r._replay_step(state,module,opt,support)
        restored=torch.load(replay_file,map_location='cpu',weights_only=False)
        r._restore(state,module,opt,restored)
        observed=r._replay_step(state,module,opt,support)
        for key in ('losses','metrics','gradient_protection','snapshot','rng_after_train','rng_after_eval'):
            r._compare_tree(expected[key],observed[key],f'disk_replay.{key}')
    finally:
        if replay_file.exists(): replay_file.unlink()
    del latest,state,module,opt
    torch.cuda.empty_cache()
    return {'status':'PASS','fold':fold,'group':group,'saved_probe_sha256':checkpoint_sha,
            'model_four_optimizers_rng_losses_predictions_and_protection_exact':True,
            'original_best_latest_checkpoints_untouched':True}

def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)

def clean(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.floating)): return value.item()
    return value

def cell(value):
    return json.dumps(clean(value),ensure_ascii=False,sort_keys=True) if isinstance(value,(dict,list,tuple,np.ndarray)) or torch.is_tensor(value) else value

def mean_sd(rows,key):
    x=np.asarray([float(v[key]) for v in rows])*100
    return float(x.mean()),float(x.std(ddof=0))

def main():
    torch.set_num_threads(1)
    frozen=r.verify_frozen_source(); sha=frozen['source_sha256']
    status=json.loads((RESULTS/'audit_status.json').read_text())
    if status['status']!='COMPLETE_AUDITED' or status['independent_checkpoint_audits']!=30 or status['rng_mismatches']!=0:
        raise RuntimeError('report-only enrichment requires 30 valid independent audits and zero RNG mismatches')
    rows=a.read_csv(RESULTS/'fold_metrics.csv'); audits=a.read_csv(RESULTS/'checkpoint_audit.csv')
    if len(rows)!=30 or len(audits)!=30: raise RuntimeError('unexpected table length')
    disk_replay=serialized_checkpoint_replay()
    cache=r.save_text_cache(); cache_sha=r.sha_file(HERE/'runtime/text_embeddings.pt')
    frame=__import__('pandas').read_csv(r.REPO/'data/ABIDE/processed_standard_data.csv')
    y=torch.as_tensor(frame.iloc[:,-1].to_numpy(np.int64)-1,dtype=torch.long)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('the independent FAPC mechanism readback expects the audited CUDA device')
    expected={(g,f) for g in r.GROUPS for f in range(1,11)}
    if {(x['group'],int(x['fold'])) for x in rows}!=expected: raise RuntimeError('fold table identities incomplete')
    mechanism=[]; protection=defaultdict(list); elapsed=[]; peak=[]; starts=[]; init_by_fold=defaultdict(dict)
    for row in rows:
        g,f=row['group'],int(row['fold']); task_path,best_path,latest_path,_=r.task_paths(g,f)
        task=json.loads(task_path.read_text()); cp=torch.load(best_path,map_location='cpu',weights_only=False)
        latest=torch.load(latest_path,map_location='cpu',weights_only=False)
        if task['status']!='COMPLETE' or any(z['source_sha256']!=sha for z in (task,cp,latest)): raise RuntimeError('invalid task/source identity')
        if r.sha_file(best_path)!=task['checkpoint_sha256']: raise RuntimeError('checkpoint SHA mismatch')
        if cp['best_epoch_1based']!=latest['best']['epoch'] or cp['metrics']!=latest['best']['metrics']: raise RuntimeError('best/latest selection inconsistency')
        r._compare_tree(cp['best_state'],latest['best']['state'],f'{g}.fold{f}.best_latest_state')
        r._compare_tree(cp['rng_trace'],latest['rng_trace'],f'{g}.fold{f}.best_latest_rng_trace')
        r._compare_tree(cp['initialization']['identity'],task['identity'],f'{g}.fold{f}.initial_identity')
        if task['initialization']['warmup_rng_sha']!=task['initialization']['post_module_rng_sha']: raise RuntimeError('initial RNG drift')
        gradients=cp['loss_history'][0]['gradient_audit_first_epoch']['classification']
        if not all(np.isfinite(float(gradients[k])) and float(gradients[k])>0 for k in ('module','darg_gfr','cam')):
            raise RuntimeError('a first-step auxiliary gradient is disconnected or non-finite')
        if not all(np.isfinite(float(v)) and float(v)>0 for v in gradients['module_by_name'].values()):
            raise RuntimeError('an active FAPC parameter has no finite first-step auxiliary gradient')
        support=torch.as_tensor(cp['initialization']['support_ids'],dtype=torch.long)
        diag=cp['best_diagnostics']; query=diag['query_ids'].long()
        if not torch.equal(support,diag['support_ids']) or np.intersect1d(support.numpy(),query.numpy()).size: raise RuntimeError('support/test leakage')
        if f==2 and any(int(k) in support for k in (376,406)): raise RuntimeError('Fold 2 overlap leak')
        spec=r.CFG['group_specs'][g]
        anchors=None if spec['mode']=='visual' else cache[spec['prompt_source']+'_anchors'].to(device)
        classes=None if spec['mode']=='visual' else cache['medical_classes'].to(device)
        m=r.initialize(anchors,classes,spec['mode'],spec['prompt_source'],f,device)
        init_by_fold[f][g]={name:p.detach().clone() for name,p in m.named_parameters()}
        m.load_state_dict(cp['best_state']['module_state'],strict=True); m.eval()
        if m.mode=='text' and not torch.equal(m.anchors,anchors): raise RuntimeError('anchor changed during training')
        if m.anchors.requires_grad or m.anchors.grad is not None: raise RuntimeError('anchor gradient is active')
        support_device=support.to(device); query_device=query.to(device); y_device=y.to(device)
        h_device=diag['h_original'].to(device); base_logp_device=diag['base_logp'].to(device)
        with torch.no_grad():
            train=m(h_device,y_device,support_device,support_device,torch.ones(2,device=device),base_logp_device,training=False)
            test=m(h_device,y_device,support_device,query_device,torch.ones(2,device=device),base_logp_device,training=False)
        if not torch.equal(test['semantic_logp'].detach().cpu(),diag['semantic_logp']):
            max_error=float((test['semantic_logp'].detach().cpu()-diag['semantic_logp']).abs().max())
            raise RuntimeError(f'same-device auxiliary readback mismatch: {max_error}')
        train_aux=scalar((train['semantic_logp'].argmax(-1)==y_device[support_device]).float().mean())
        test_aux=scalar((test['semantic_logp'].argmax(-1)==y_device[query_device]).float().mean())
        train_main=scalar((base_logp_device[support_device].argmax(-1)==y_device[support_device]).float().mean())
        hist=latest['loss_history']; gps=[x['gradient_protection'] for x in hist]
        if len(gps)!=len(latest['rng_trace']): raise RuntimeError('incomplete gradient/RNG trace')
        if not all(np.isfinite(float(z[k])) for z in gps for k in ('base_norm','aux_norm','aux_norm_ratio_raw','aux_norm_ratio_safe','gradient_cosine')): raise RuntimeError('non-finite gradient stats')
        if any(z['aux_norm_ratio_safe']>.20001 for z in gps): raise RuntimeError('CAM auxiliary cap exceeded')
        protection[g]+=gps
        mean0=train['prototype_means'][0]; mean1=train['prototype_means'][1]
        item={'group':g,'fold':f,'mode':m.mode,'support_count':len(support),'support_asd':int((y[support]==0).sum()),'support_td':int((y[support]==1).sum()),
              'own_patient_mean_weight_asd':1.0/int((y[support]==0).sum()),'own_patient_mean_weight_td':1.0/int((y[support]==1).sum()),
              'fold2_overlap_excluded':True,'no_loo':True,'train_aux_accuracy':train_aux,'test_aux_accuracy':test_aux,
              'train_darg_accuracy':train_main,'test_darg_accuracy':float(row['acc']),'train_test_aux_gap_pp':100*(train_aux-test_aux),
              'prototype_mean_cosine':scalar(F.cosine_similarity(mean0[None],mean1[None])),'prototype_difference_norm':scalar((mean0-mean1).norm()),
              'calibrated_direction_cosine':scalar(train['prototypes'][0]@train['prototypes'][1]),
              'text_cache_sha256':cache_sha,'anchor_buffer_sha256':m.anchor_sha(),'anchor_gradient_none':True,
              'protected_epochs':len(gps),'conflict_epochs':sum(z['conflict'] for z in gps),'projected_epochs':sum(z['projected'] for z in gps),
              'scaled_epochs':sum(z['scaled'] for z in gps),'mean_gradient_cosine':float(np.mean([z['gradient_cosine'] for z in gps])),
              'mean_aux_ratio_raw':float(np.mean([z['aux_norm_ratio_raw'] for z in gps])),
              'mean_aux_ratio_safe':float(np.mean([z['aux_norm_ratio_safe'] for z in gps])),
              'semantic_loss_first':hist[0]['losses']['semantic_loss'],'semantic_loss_last':hist[-1]['losses']['semantic_loss'],
              'darg_cls_loss_first':hist[0]['losses']['base_cls'],'darg_cls_loss_last':hist[-1]['losses']['base_cls']}
        if m.mode=='text':
            b=F.normalize(m.anchors,dim=-1); q=F.normalize(train['prompt_calibrated'],dim=-1)
            gram=b@b.T; after=q@q.T; angles=torch.rad2deg(torch.acos((b*q).sum(-1).clamp(-1,1)))
            off=~torch.eye(5,dtype=torch.bool)
            item.update({'beta':m.calibration_strength,'beta_mean':scalar(m.calibration_strength.mean()),
                         'calibration_angles_deg':angles,'calibration_angle_mean_deg':scalar(angles.mean()),
                         'anchor_gram':gram,'calibrated_gram':after,'anchor_asd_within_cosine':scalar(gram[:5,:5][off].mean()),
                         'anchor_td_within_cosine':scalar(gram[5:,5:][off].mean()),'anchor_cross_class_cosine':scalar(gram[:5,5:].mean()),
                         'keep_loss':scalar(train['keep_loss'])})
        mechanism.append({k:cell(v) for k,v in item.items()})
        elapsed.append(float(task['wall_seconds'])); peak.append(int(task['peak_gpu_bytes'])); starts.append(task_path.stat().st_mtime-float(task['wall_seconds']))
        del cp,latest,m
    for f,values in init_by_fold.items():
        for name,v in values[r.GROUPS[1]].items():
            if not torch.equal(v,values[r.GROUPS[2]][name]): raise RuntimeError('G2/G3 initial trainable parameters differ')
        for name in ('patient_projection.weight','prototype_direction.weight'):
            if any(not torch.equal(values[r.GROUPS[0]][name],values[g][name]) for g in r.GROUPS[1:]): raise RuntimeError('common patient/prototype initialization differs')
    a.write_csv(RESULTS/'mechanism.csv',mechanism)
    comps=a.read_csv(RESULTS/'paired_comparisons.csv'); by={g:[x for x in rows if x['group']==g] for g in r.GROUPS}
    all_gp=[z for values in protection.values() for z in values]
    total_epochs=len(all_gp); conflicts=sum(z['conflict'] for z in all_gp); projected=sum(z['projected'] for z in all_gp); scaled=sum(z['scaled'] for z in all_gp)
    controller=json.loads((r.LOGS/'controller.json').read_text()); workers=len(controller['workers'])
    tokens=[Path(x['log']).name.split('_worker_')[0].removeprefix('controller_') for x in controller['workers']]
    launch_time=calendar.timegm(time.strptime(tokens[0],'%Y%m%dT%H%M%S'))
    wall=float(controller['finished'])-launch_time
    def paired_line(left,right,scope='all10'):
        name=f'{left} - {right}'
        return [x for x in comps if x['comparison']==name and x['scope']==scope and x['metric'] in METRICS]
    def change(left,right,metric='acc',scope='all10'):
        return next(float(x['mean_difference_pp']) for x in paired_line(left,right,scope) if x['metric']==metric)
    g1,g2,g3=r.GROUPS
    exceeds_base=change(g2,'E74.4-G2')>0 and change(g2,'E74.4-G2','hard_auc')>0
    exceeds_visual=change(g2,g1)>0
    exceeds_anon=change(g2,g3)>0 and change(g2,g3,'hard_auc')>0
    survives_fold2=change(g2,'E74.4-G2',scope='exclude_fold2')>0
    go=exceeds_base and exceeds_visual and exceeds_anon and survives_fold2
    conclusions=['GO_CANDIDATE' if go else 'NO_GO_FAPC']
    if not exceeds_anon: conclusions.append('MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED')
    if not exceeds_base: conclusions.append('GRADIENT_PROTECTION_ACTIVE_BUT_NO_GAIN')
    if not exceeds_visual: conclusions.append('FROZEN_ANCHOR_EFFECT_UNSUPPORTED')
    lines=['# Experiment 78.8 — FAPC final report','',
           'STATUS: COMPLETE_AUDITED','',
           f'最终有效任务30/30、best独立重载30/30、latest状态30/30。正式目录：`{HERE}`；单一/model输出回退，原始数据和历史实验只读。',
           f'冻结训练源SHA：`{sha}`；RN50缓存SHA：`{cache_sha}`。',
           f'并发{workers} worker；从worker启动到全部退出墙钟{wall/60:.2f}分钟；各任务训练耗时合计{sum(elapsed)/3600:.3f}小时；单任务峰值分配显存最大{max(peak)/2**30:.3f}GiB。该显存值不含其他CUDA进程的上下文开销。','',
           '## 修复与证据边界','',
           '此前两轮产物均已作废并归档：首轮匿名文本不符合定义；第二轮发现CAM144 Hook只返回辅助梯度而漏加主梯度。最终代码返回g_base+g_aux_safe，未修改任何预注册损失权重、温度、优化器或残差上限。全套预检包含Hook开启/辅助为零时原梯度逐值恢复，以及全参数独立梯度合成对照；旧指标不纳入此报告。','',
           '## 方法与来源','',
           '唯一基座是E74.4-G2 Positive-DCR，从逐折E61 epoch-10完整模型、三个优化器和全部RNG恢复。CAM144在原分类器前捕获一次；Visual-only与医学/匿名组共享144→32患者投影和32→32疾病方向映射。医学与匿名各十条冻结RN50文本，经共同CPU float64全行空间SVD保留20向量Gram矩阵，补零至32维。无可训练文本映射、无Attention库、无多子原型、无LOO。',
           'E78.7实际bdpprompt.py将冻结RN50向量送入可训练text_projection(1024→32)，L_keep比较校准后Prompt与当前可移动的投影输出，因此并未冻结原始参考方向。E78.8删除该可训练文本投影，固定锚点以buffer进入Checkpoint。E78.7-G1的AUC/SEN改善伴随ACC/SPE下降，本轮以六指标与匿名/Visual-only对照共同判断。',
           '服务器真实E74.4 soft_positive_dcr.py的forward_state调用一次DARG与CAM，原分类项是历史ClsLoss(logp, y, train, weights)，其实现为weighted F.nll_loss；total=cls+graph+0.05pair+0.01adv+0.01preserve。本轮直接复用该out[total]，未自行重写或替换分类损失。原始代码路径与SHA由冻结manifest记录。',
           '每类合法训练支持均值产生有符号疾病方向；G2/G3通过q=normalize(anchor+beta*v_c)校准每条Prompt，beta=0.2*sigmoid(raw)。五条Prompt以温度0.25的LogMeanExp形成ASD/TD辅助Logits，尺度10。G1直接匹配视觉疾病方向。所有组优化L_DARG+0.1L_aux+0.01L_keep，G1的keep为0。最终预测始终使用原DARG MLP。',
           'CAM144上仅投影冲突的辅助梯度，并把其范数限制为主梯度的20%；主任务及原分类器直接梯度保留。原三个优化器后执行新增AdamW(lr=5e-4, weight_decay=0.01)。',
           '论文依据：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、[KgCoOp](https://arxiv.org/abs/2303.13283)、[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。前两者提供原始Prompt与疾病级监督背景；KgCoOp/PCGrad为冻结参考与冲突投影的设计依据，本方法不是其逐行复现。真实历史损失、数据、warmup和SHA链见design.md、manifest.json及preflight.json。','',
           '## 六指标','',
           '均值 ± population SD，%；ASD=0、TD=1；SEN为ASD敏感度，SPE为TD特异度。Hard-AUC/BA与Probability ROC-AUC分别报告。','',
           '|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for g in r.GROUPS: lines.append('|'+g+'|'+'|'.join(f'{mean_sd(by[g],k)[0]:.4f} ± {mean_sd(by[g],k)[1]:.4f}' for k in METRICS)+'|')
    lines+=['','历史只读参考，沿用历史评价协议：','', '|方法|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for name,path,group in (('E74.4-G2',r.BASE/'results/fold_metrics.csv',r.BASE_GROUP),('E78.7-G1',Path('/model/spromptgl/experiment/experiment78_7_bdpprompt/results/fold_metrics.csv'),'E78.7-G1-InverseCalibration-Medical')):
        rr=[x for x in a.read_csv(path) if x['group']==group]
        lines.append('|'+name+'|'+'|'.join(f'{mean_sd(rr,k)[0]:.4f} ± {mean_sd(rr,k)[1]:.4f}' for k in METRICS)+'|')
    lines+=['','## 主要配对差','', '差值为百分点；逐折差、W/T/L和描述性95%区间全部见paired_comparisons.csv。','', '|比较|ACC差 [95%区间]|ACC W/T/L|Hard-AUC差|Prob-AUC差|去Fold 2 ACC差|','|---|---:|---:|---:|---:|---:|']
    specs=[(g1,'E74.4-G2'),(g2,'E74.4-G2'),(g3,'E74.4-G2'),(g2,g1),(g2,g3),(g2,'E78.7-G1')]
    for left,right in specs:
        p=next(x for x in paired_line(left,right) if x['metric']=='acc')
        lines.append(f'|{left} − {right}|{float(p["mean_difference_pp"]):+.4f} [{float(p["ci95_low_pp"]):+.4f}, {float(p["ci95_high_pp"]):+.4f}]|{p["wins"]}/{p["ties"]}/{p["losses"]}|{change(left,right,"hard_auc"):+.4f}|{change(left,right,"prob_auc"):+.4f}|{change(left,right,scope="exclude_fold2"):+.4f}|')
    lines+=['','去Fold 2后的9折六指标：','', '|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for g in r.GROUPS:
        rr=[x for x in by[g] if int(x['fold'])!=2]
        lines.append('|'+g+'|'+'|'.join(f'{mean_sd(rr,k)[0]:.4f} ± {mean_sd(rr,k)[1]:.4f}' for k in METRICS)+'|')
    lines+=['','## 机制与独立审计','',
           f'- 20条文本秩{cache["effective_rank"]}；float64 Gram最大误差{cache["gram_max_abs_error"]:.3e}，float32缓存读回误差{float(((cache["anchors"].double()@cache["anchors"].double().T)-(cache["embeddings_raw"].double()@cache["embeddings_raw"].double().T)).abs().max()):.3e}。医学/匿名每类五条独立文本、无重复和截断；固定锚点逐Checkpoint逐值一致、梯度为None。',
           f'- 全部{total_epochs}个执行epoch中，冲突{conflicts}（{conflicts/total_epochs:.2%}），投影{projected}（{projected/total_epochs:.2%}），范数缩放{scaled}（{scaled/total_epochs:.2%}）；平均梯度余弦{np.mean([z["gradient_cosine"] for z in all_gp]):.5f}，原始辅助/主梯度范数比{np.mean([z["aux_norm_ratio_raw"] for z in all_gp]):.5f}，安全比{np.mean([z["aux_norm_ratio_safe"] for z in all_gp]):.5f}。',
           f'- 10/10 base-off parity与两步模型/四优化器/RNG/梯度统计恢复均通过。共有RNG epoch核对{status["rng_common_epochs_checked"]}，不匹配{status["rng_mismatches"]}；无覆盖的历史epoch不宣称已验证。',
           '- best逐个独立前向读回概率、预测和六指标；latest全部存在且状态、最佳epoch、原三个和新增优化器结构与身份均核对。根源码和历史依赖再次按冻结manifest复核。',
           '- 合法支持为unique(train)−unique(test)，Fold 2两条历史重叠行被排除；各类支持人数、原型均值夹角、差异范数、训练/测试辅助准确率、beta与Prompt位移见mechanism.csv。训练查询自身参与支持均值，因此训练辅助表现不代表独立泛化。','',
           '|组别|辅助训练ACC|辅助测试ACC|DARG训练ACC|冲突率|缩放率|安全梯度均值比|','|---|---:|---:|---:|---:|---:|---:|']
    for g in r.GROUPS:
        mm=[x for x in mechanism if x['group']==g]; gp=protection[g]
        lines.append(f'|{g}|{100*np.mean([x["train_aux_accuracy"] for x in mm]):.4f}|{100*np.mean([x["test_aux_accuracy"] for x in mm]):.4f}|{100*np.mean([x["train_darg_accuracy"] for x in mm]):.4f}|{np.mean([x["conflict"] for x in gp]):.2%}|{np.mean([x["scaled"] for x in gp]):.2%}|{np.mean([x["aux_norm_ratio_safe"] for x in gp]):.5f}|')
    interventions=a.read_csv(RESULTS/'interventions.csv')
    lines+=['','医学G2固定Checkpoint干预（最终DARG不直接读取Prompt，故预期最终预测不变）：','', '|干预|语义ASD概率平均绝对变化|辅助硬翻转总数|DARG概率平均绝对变化|DARG硬翻转|','|---|---:|---:|---:|---:|']
    for name in ('full','no_calibration','anchor_only','anonymous_text','class_swap'):
        ii=[x for x in interventions if x['group']==g2 and x['intervention']==name]
        if len(ii)!=10: raise RuntimeError(f'incomplete intervention {name}')
        lines.append(f'|{name}|{np.mean([float(x["mean_abs_semantic_probability_change"]) for x in ii]):.6f}|{sum(int(x["semantic_hard_flips"]) for x in ii)}|{np.mean([float(x["mean_abs_darg_probability_change"]) for x in ii]):.6f}|{sum(int(x["hard_label_flips"]) for x in ii)}|')
    lines+=['','固定Checkpoint文本更换只验证辅助分支的依赖；最终DARG零硬翻转不能用于否定训练中辅助梯度的作用。医学内容的独立证据以G2与G3独立训练为准；视觉类别监督解释以G2与G1比较为准。','',
           '## 科研判断','', '`'+'`；`'.join(conclusions)+'`。',
           f'G2−基座ACC {change(g2,"E74.4-G2"):+.4f} pp、Hard-AUC {change(g2,"E74.4-G2","hard_auc"):+.4f} pp、Prob-AUC {change(g2,"E74.4-G2","prob_auc"):+.4f} pp；G2−匿名ACC {change(g2,g3):+.4f} pp；G2−Visual-only ACC {change(g2,g1):+.4f} pp。',
           '冻结锚点与梯度投影是否在数值上成立，与是否产生独立诊断增益分别判断。保护发生在CAM144梯度边界，不能保证经过上游Jacobian、Adam状态和其他图损失后的全部参数更新必然改善主任务。相对E78.7的变化同时包含固定语义空间和梯度保护，现有三组不能单独估计其中每一项的因果增益。统计区间是十折配对差的描述性t区间，折之间训练集重叠，不能解释为无偏独立样本的临床显著性检验。','',
           '## 已知限制','',
           '单seed=0，保留val=test、测试ACC选Checkpoint、CV前RFE/预处理、全患者transductive图和历史Fold 2主干训练重叠。新支持隔离未消除整体历史协议的泄漏风险。未进行多seed、外部独立测试或新增超参数搜索；不自动开展E78.9。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status.update({'valid_run_revision':r.CFG['run_revision'],'scientific_conclusions':conclusions,'training_workers':workers,'training_wall_seconds':wall,
                   'training_task_seconds_sum':sum(elapsed),'training_epochs':total_epochs,'gradient_conflict_epochs':conflicts,'gradient_projected_epochs':projected,'gradient_scaled_epochs':scaled,
                   'fixed_anchor_cache_sha256':cache_sha,'anchors_unchanged_all_20_text_checkpoints':True,'latest_states_checked':30,
                   'first_step_auxiliary_gradients_verified':30,
                   'serialized_checkpoint_two_step_replay':disk_replay,
                   'independent_gradient_composition_preflight_pass':True,'superseded_invalid_runs':2,
                   'report_only_source_sha256':r.sha_file(HERE/'report_only.py')})
    r.atomic_json(RESULTS/'audit_status.json',status)
    r.atomic_json(HERE/'manifest.json',{**frozen,'status':'COMPLETE_AUDITED','completed_tasks':30,'independent_audits':30,'report_only_source_sha256':r.sha_file(HERE/'report_only.py'),'scientific_conclusions':conclusions})
    hand=Path('/model/spromptgl/experiment/handoff.md'); marker=f'## Experiment 78.8 final corrected run {sha[:12]}'
    old=hand.read_text(encoding='utf-8') if hand.exists() else ''
    if marker not in old:
        with hand.open('a',encoding='utf-8') as f:
            f.write('\n'+marker+'\n\n')
            f.write(f'正式目录：{HERE}；单一/model回退。最终有效任务30/30，best独立审计30/30，latest身份30/30；10 worker，墙钟{wall/60:.2f}分钟。此前两轮因匿名对照定义和Hook遗漏主梯度而作废，不纳入下列结果；完整保留在runtime/superseded_*。\n\n')
            f.write('方法：G1 Visual-only；G2固定医学锚点；G3固定匿名锚点。全部逐折从E61 epoch-10恢复E74.4-G2，原三个优化器及完整L_DARG保留；0.1辅助分类+0.01固定锚点保持；CAM144单侧冲突投影/20%上限，实际Hook传回g_base+safe_aux。RN50共同20文本全行空间SVD，无可训练文本投影。\n\n')
            f.write('\n'.join(lines[lines.index('|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|'):lines.index('历史只读参考，沿用历史评价协议：')])+'\n')
            f.write(f'G2−E74.4-G2：ACC {change(g2,"E74.4-G2"):+.4f}、Hard-AUC {change(g2,"E74.4-G2","hard_auc"):+.4f}、Prob-AUC {change(g2,"E74.4-G2","prob_auc"):+.4f} pp；G2−E78.7-G1 ACC {change(g2,"E78.7-G1"):+.4f} pp；G2−G1 ACC {change(g2,g1):+.4f} pp；G2−G3 ACC {change(g2,g3):+.4f} pp。去Fold 2 G2−基座ACC {change(g2,"E74.4-G2",scope="exclude_fold2"):+.4f} pp。\n\n')
            f.write(f'固定锚点20/20未漂移；缓存SHA {cache_sha}；Gram误差{cache["gram_max_abs_error"]:.3e}。梯度冲突{conflicts}/{total_epochs}、投影{projected}、缩放{scaled}；生产Hook零辅助主梯度逐值恢复、独立全参数梯度合成与两步恢复通过。共有RNG {status["rng_common_epochs_checked"]} epoch，不匹配0。\n\n')
            f.write('最终判断：'+', '.join(conclusions)+'。固定文本干预只影响辅助语义分支，不应直接改变DARG最终预测。完整六指标、配对差/95%区间、去Fold 2与机制见results/report.md、fold_metrics.csv、paired_comparisons.csv、mechanism.csv、interventions.csv、audit_status.json；核心代码fapc.py、run.py、audit.py、make_text_cache.py。仍受val=test、CV前RFE、transductive图、历史重叠、单seed与不做LOO限制。\n')
    status['handoff_update']='APPENDED_FINAL_CORRECTED_RUN'; status['handoff_path']=str(hand)
    r.atomic_json(RESULTS/'audit_status.json',status)
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
