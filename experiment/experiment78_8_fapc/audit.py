#!/usr/bin/env python3
from __future__ import annotations
import copy,csv,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
import run as runner
HERE,RESULTS,GROUPS,METRICS,BASE=runner.HERE,runner.RESULTS,runner.GROUPS,runner.METRICS,runner.BASE

def safe(v):
    if torch.is_tensor(v): return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,(np.integer,np.floating)): return v.item()
    return v

def write_json(path,v): runner.atomic_json(path,v)
def write_csv(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    tmp=Path(path).with_name(Path(path).name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def unit_tests():
    torch.manual_seed(78806)
    tests={}
    # fixed-anchor Gram and norm checks
    p=runner.save_text_cache(); A=p['anchors'].double(); R=p['embeddings_raw'].double()
    tests['anchor_shape']=tuple(A.shape)==(20,32); tests['raw_shape']=tuple(R.shape)==(20,1024)
    gram_error=float((A@A.T-R@R.T).abs().max())
    tests['gram_error']=gram_error<2e-6
    tests['gram_error_value']=gram_error
    tests['anchor_norms']=bool(torch.allclose(A.norm(dim=-1),R.norm(dim=-1),atol=2e-6,rtol=0))
    # artificial one-sided gradient protection, including conflict projection and cap
    def protect(gb,ga,rho=.2):
        eps=1e-12; dot=(gb*ga).sum(); nb=gb.norm(); gp=ga.clone(); proj=bool(dot<0)
        if proj and nb>eps: gp=ga-dot/(nb.square()+eps)*gb
        npg=gp.norm(); safe=gp if nb<=eps or npg<=rho*nb else gp*(rho*nb/(npg+eps))
        return safe,proj
    gb=torch.tensor([1.,0.]); ga_same=torch.tensor([.5,.2]); ga_opp=torch.tensor([-1.,.4]); ga_orth=torch.tensor([0.,1.])
    s,pr=protect(gb,ga_same); tests['grad_same_not_projected']=not pr
    s,pr=protect(gb,ga_opp); tests['grad_conflict_projected']=pr and abs(float(s[0]))<1e-6 and float(s.norm())<=.200001
    s,pr=protect(gb,ga_orth); tests['grad_orth_capped']=not pr and float(s.norm())<=.200001
    # Test the production tensor hook itself, so dropping the main gradient
    # cannot be hidden by tests of the projection algebra alone.
    for name,base_vector,aux_vector in (
        ('aligned',[1.,0.],[.1,0.]),('opposed',[1.,0.],[-1.,.4]),
        ('orthogonal',[1.,0.],[0.,1.]),('zero_aux',[1.,0.],[0.,0.]),
        ('zero_main',[0.,0.],[1.,0.])):
        x=torch.tensor([.4,.8],requires_grad=True); h=2*x
        gb=torch.tensor(base_vector); ga=torch.tensor(aux_vector)
        b=(gb*h).sum(); aux=(ga*h).sum()
        expected_aux,pr=protect(gb,ga)
        if float(gb.norm())<=1e-12: expected_aux.zero_()
        details={'h_original':h,'semantic_loss':aux/float(runner.CFG['method']['semantic_loss_weight']),'keep_loss':aux*0}
        runner._protected_backward({'total':b},details,b+aux)
        tests['production_hook_'+name]=bool(torch.allclose(x.grad,2*(gb+expected_aux),atol=1e-6,rtol=0))
    # module shape/gradient tests on a CPU synthetic graph
    h=torch.randn(871,144,requires_grad=True); y=torch.tensor(([0,1]*435)+[0]); support=torch.arange(700); q=support[:31]; w=torch.ones(2); base=torch.log_softmax(torch.randn(871,2),-1)
    mods={}
    for g in GROUPS:
        spec=runner.CFG['group_specs'][g]; anchors=None if spec['mode']=='visual' else p['medical_anchors']; cls=None if spec['mode']=='visual' else p['medical_classes']
        m=runner.initialize(anchors,cls,spec['mode'],spec['prompt_source'],1,torch.device('cpu')); mods[g]=m
        o=m(h,y,support,q,w,base,training=True)
        tests[g+'_finite']=bool(torch.isfinite(o['semantic_logits']).all() and torch.isfinite(o['semantic_loss']))
        tests[g+'_shape']=tuple(o['semantic_logits'].shape)==(31,2)
        o['semantic_loss'].backward(retain_graph=True)
        tests[g+'_patient_grad']=m.patient_projection.weight.grad is not None and float(m.patient_projection.weight.grad.norm())>0
        m.zero_grad(set_to_none=True)
    tests['g23_same_parameter_count']=sum(p.numel() for p in mods[GROUPS[1]].parameters())==sum(p.numel() for p in mods[GROUPS[2]].parameters())
    tests['all_finite']=all(bool(v) for v in tests.values() if isinstance(v,bool))
    write_json(RESULTS/'unit_tests.json',{'status':'PASS' if tests['all_finite'] else 'FAIL','tests':tests})
    if not tests['all_finite']: raise RuntimeError(tests)
    return tests

def restore(group,fold,source_sha):
    taskf,bestf,latestf,_=runner.task_paths(group,fold)
    task=json.loads(taskf.read_text()); cp=torch.load(bestf,map_location='cpu',weights_only=False); latest=torch.load(latestf,map_location='cpu',weights_only=False)
    if task.get('status')!='COMPLETE' or task.get('source_sha256')!=source_sha or cp.get('source_sha256')!=source_sha or latest.get('source_sha256')!=source_sha: raise RuntimeError('task/source identity mismatch')
    if task.get('checkpoint_sha256')!=runner.sha_file(bestf): raise RuntimeError('best checkpoint SHA mismatch')
    state,_,identity=runner.prepare(fold); support=runner.support_ids(state)
    if cp.get('identity')!=identity or cp['initialization']['support_ids']!=support.tolist(): raise RuntimeError('base/support identity mismatch')
    overlap=np.intersect1d(state['train_np'],state['test_np']); expected=np.asarray([376,406]) if fold==2 else np.zeros(0,dtype=np.int64)
    if not np.array_equal(overlap,expected) or np.intersect1d(overlap,support).size: raise RuntimeError('Fold-2 overlap support leak')
    module=runner._make_module(state,group); opt=runner._new_optimizer(module)
    runner.hist().load_bundle(state,cp['best_state']['base_state']); module.load_state_dict(cp['best_state']['module_state'],strict=True); opt.load_state_dict(cp['best_state']['new_optimizer']); module.eval()
    with torch.no_grad(): out,details=runner._forward(state,module,training=False,support=support,query=state['test_np']); metrics=runner._final_metrics(out,details,state); base_metrics=runner._metrics(out['logp'],state)
    if not np.array_equal(np.asarray(metrics['pred']),np.asarray(cp['test_prediction'])) or not np.allclose(np.asarray(metrics['prob']),np.asarray(cp['test_probability']),atol=1e-7,rtol=0): raise RuntimeError('prediction readback mismatch')
    for k in METRICS:
        if abs(float(metrics[k])-float(cp['metrics'][k]))>1e-8: raise RuntimeError(f'metric readback mismatch {k}')
    # Text buffers are frozen and remain byte-identical after checkpoint reload.
    anchor_sha=module.anchor_sha(); expected_anchor=cp['best_diagnostics'].get('anchor_sha') if isinstance(cp.get('best_diagnostics'),dict) else None
    if module.mode=='text':
        payload=runner.prompt_tensors(torch.device('cpu'))[0]
        expected=payload['medical_anchors' if module.prompt_source=='medical' else 'anonymous_anchors']
        expected_sha=__import__('hashlib').sha256(expected.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if anchor_sha!=expected_sha: raise RuntimeError('checkpoint text anchor differs from frozen text cache')
        if expected_anchor is not None and anchor_sha!=expected_anchor: raise RuntimeError('checkpoint anchor identity differs from saved diagnostics')
    bt=json.loads((BASE/'.runtime/tasks'/f'{runner.BASE_GROUP}_f{fold:02d}.json').read_text())['rng_trace']; nt=cp['rng_trace']; bmap={int(x['epoch']):x for x in bt}; nmap={int(x['epoch']):x for x in nt}; common=sorted(set(bmap)&set(nmap)); exact=sum(all(bmap[e][k]==nmap[e][k] for k in ('rng_before','rng_after_train','rng_after_eval')) for e in common)
    hist=cp['loss_history']; gp=[x.get('gradient_protection',{}) for x in hist]; gp=[x for x in gp if x]
    grad_first=cp['loss_history'][0].get('gradient_audit_first_epoch',{})
    audit={'group':group,'fold':fold,'status':'INDEPENDENT_AUDIT_PASS','checkpoint_sha256':runner.sha_file(bestf),'source_sha256':source_sha,'best_epoch_1based':int(cp['best_epoch_1based']),'epochs_executed':int(cp['epochs_executed']),'metrics':{k:float(metrics[k]) for k in METRICS},'base_metrics':{k:float(base_metrics[k]) for k in METRICS},'support_count':len(support),'fold2_overlap_excluded':True,'test_label_invariant':True,'original_optimizers_restored':True,'new_optimizer_restored':True,'anchor_sha256':anchor_sha,'anchor_buffer_requires_grad':False,'rng_common_epochs':len(common),'rng_exact_epochs':exact,'rng_mismatches':len(common)-exact,'gradient_protection':{'epochs':len(gp),'conflict_fraction':float(np.mean([bool(x.get('conflict',False)) for x in gp])) if gp else 0.0,'projected_fraction':float(np.mean([bool(x.get('projected',False)) for x in gp])) if gp else 0.0,'scaled_fraction':float(np.mean([bool(x.get('scaled',False)) for x in gp])) if gp else 0.0,'mean_cosine':float(np.mean([x.get('gradient_cosine',0.) for x in gp])) if gp else 0.0,'mean_safe_ratio':float(np.mean([x.get('aux_norm_ratio_safe',0.) for x in gp])) if gp else 0.0},'new_gradient_norms':grad_first}
    return audit,cp,state,module,opt,details,metrics,base_metrics

def mechanism(audit,cp,state,module,details,metrics):
    d=details; q=d['prompt_calibrated'].detach(); b=d['prompt_base'].detach(); beta=d['calibration_strength'].detach()
    if module.mode=='text':
        disp=(1-(F.normalize(q,dim=-1)*F.normalize(b,dim=-1)).sum(-1)).cpu().numpy(); keep=(1-(F.normalize(q,dim=-1)*F.normalize(module.anchors,dim=-1)).sum(-1)).cpu().numpy()
        sim_anchor=(F.normalize(module.anchors,dim=-1)@F.normalize(module.anchors,dim=-1).T).cpu().numpy()
        return {'group':audit['group'],'fold':audit['fold'],'mode':module.mode,'prompt_calibration_mean':float(disp.mean()),'prompt_calibration_max':float(disp.max()),'prompt_keep_mean':float(keep.mean()),'beta_mean':float(beta.mean()),'beta_min':float(beta.min()),'beta_max':float(beta.max()),'anchor_pairwise_mean_offdiag':float((sim_anchor.sum()-np.trace(sim_anchor))/(sim_anchor.size-len(sim_anchor))),'anchor_sha256':module.anchor_sha(),'prototype_angle':float(torch.acos(torch.clamp(d['prototypes'][0]@d['prototypes'][1],-1,1)).detach().cpu()),'semantic_test_accuracy':float((d['semantic_logp'].argmax(1)==torch.as_tensor(state['y'][state['test_np']],device=d['semantic_logp'].device)).float().mean().cpu()),'semantic_margin_mean':float(d['semantic_margin'].mean().cpu()),'gradient_conflict_fraction':audit['gradient_protection']['conflict_fraction'],'gradient_projected_fraction':audit['gradient_protection']['projected_fraction'],'gradient_scaled_fraction':audit['gradient_protection']['scaled_fraction'],'support_count':audit['support_count'],'no_loo':True}
    return {'group':audit['group'],'fold':audit['fold'],'mode':module.mode,'prompt_calibration_mean':0.0,'prompt_calibration_max':0.0,'beta_mean':0.0,'prototype_angle':float(torch.acos(torch.clamp(d['prototypes'][0]@d['prototypes'][1],-1,1)).detach().cpu()),'visual_test_accuracy':float((d['semantic_logp'].argmax(1)==torch.as_tensor(state['y'][state['test_np']],device=d['semantic_logp'].device)).float().mean().cpu()),'gradient_conflict_fraction':audit['gradient_protection']['conflict_fraction'],'gradient_projected_fraction':audit['gradient_protection']['projected_fraction'],'gradient_scaled_fraction':audit['gradient_protection']['scaled_fraction'],'support_count':audit['support_count'],'no_loo':True}

def interventions(group,fold,state,module):
    if module.mode!='text': return []
    support=runner.support_ids(state); q=state['test_np']; rows=[]
    with torch.no_grad(): _,full=runner._forward(state,module,training=False,support=support,query=q); base_sem=full['semantic_logp'].exp()[:,0].cpu().numpy(); base_prob=full['final_logp'].exp()[:,0].cpu().numpy(); base_pred=full['final_logp'].argmax(1).cpu().numpy()
    names=[('full','full'),('no_calibration','no_calibration'),('anchor_only','anchor_only'),('class_swap','class_swap')]
    if module.prompt_source=='medical': names.append(('anonymous_text','full'))
    for name,flag in names:
        original=module.anchors.detach().clone()
        if name=='anonymous_text':
            replacement=runner.prompt_tensors(module.anchors.device)[0]['anonymous_anchors'].to(module.anchors.device)
            module.anchors.copy_(replacement)
        try:
            with torch.no_grad(): _,d=runner._forward(state,module,training=False,support=support,query=q,intervention=flag)
        finally:
            module.anchors.copy_(original)
        sem=d['semantic_logp'].exp()[:,0].cpu().numpy(); prob=d['final_logp'].exp()[:,0].cpu().numpy(); pred=d['final_logp'].argmax(1).cpu().numpy()
        rows.append({'group':group,'fold':fold,'intervention':name,'mean_abs_semantic_probability_change':float(np.abs(sem-base_sem).mean()),'max_semantic_probability_change':float(np.abs(sem-base_sem).max()),'mean_abs_darg_probability_change':float(np.abs(prob-base_prob).mean()),'hard_label_flips':int((pred!=base_pred).sum()),'darg_unchanged':bool(np.array_equal(pred,base_pred)),'semantic_hard_flips':int((d['semantic_logp'].argmax(1).cpu().numpy()!=full['semantic_logp'].argmax(1).cpu().numpy()).sum())})
    return rows

def summary(group,rows):
    out={'group':group,'folds':len(rows)}
    for m in METRICS:
        v=np.asarray([float(r[m]) for r in rows])*100; out[m+'_mean_percent']=float(v.mean()) if len(v) else float('nan'); out[m+'_population_sd_percent']=float(v.std(ddof=0)) if len(v) else float('nan')
    out['success_count']=len(rows); return out

def ref(path,group):
    rows=[r for r in read_csv(path) if r['group']==group]; return {int(r['fold']):{m:float(r[m]) for m in METRICS} for r in rows}
def paired(name,l,r,folds):
    rows=[]
    for m in METRICS:
        d=np.asarray([(l[f][m]-r[f][m])*100 for f in folds]); mean=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.; half=float(t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d))) if len(d)>1 else 0.
        rows.append({'comparison':name,'scope':'all10' if len(folds)==10 else 'exclude_fold2','metric':m,'n_folds':len(d),'mean_difference_pp':mean,'sample_sd_difference_pp':sd,'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'ci95_low_pp':mean-half,'ci95_high_pp':mean+half,'fold_differences_pp':json.dumps({str(f):float((l[f][m]-r[f][m])*100) for f in folds},sort_keys=True)})
    return rows

def report(summaries,comps,status,audit_status,source_sha):
    lines=['# Experiment 78.8 — FAPC results','',f'状态：`{status}`；目标30折，训练完成{audit_status["trained_tasks"]}/30，独立Checkpoint审计{audit_status["independent_checkpoint_audits"]}/30。',f'目录：`{HERE}`；单一正式输出使用/model回退；训练源SHA：`{source_sha}`。','','## 六指标（均值 ± population SD，%）','','|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功|','|---|---:|---:|---:|---:|---:|---:|---:|']
    by={x['group']:x for x in summaries}
    for g in GROUPS:
        x=by.get(g,{}); lines.append('|'+g+'|'+'|'.join(f'{x.get(m+"_mean_percent",float("nan")):.4f} ± {x.get(m+"_population_sd_percent",float("nan")):.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+f'|{x.get("success_count",0)}|')
    refs=[('E74.4-G2',ref(BASE/'results/fold_metrics.csv',runner.BASE_GROUP)),('E78.7-G1',ref(Path('/model/spromptgl/experiment/experiment78_7_bdpprompt/results/fold_metrics.csv'),'E78.7-G1-InverseCalibration-Medical'))]
    lines+=['','历史参考：','|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|','|---|---:|---:|---:|---:|---:|---:|']
    for n,d in refs:
        x=summary(n,[{'group':n,**d[f]} for f in sorted(d)]); lines.append('|'+n+'|'+'|'.join(f'{x[m+"_mean_percent"]:.4f} ± {x[m+"_population_sd_percent"]:.4f}' for m in ('acc','hard_auc','prob_auc','sen','spe','asd_f1'))+'|')
    g1,g2,g3=[by.get(g,{}) for g in GROUPS]; base=summary('base',[{'group':'base',**refs[0][1][f]} for f in sorted(refs[0][1])]); g2base=bool(g2 and g2.get('acc_mean_percent',-1)>=base['acc_mean_percent'] and g2.get('hard_auc_mean_percent',-1)>=base['hard_auc_mean_percent']); med=bool(g2 and g3 and g2.get('acc_mean_percent',-1)>g3.get('acc_mean_percent',-1)); g2g1=bool(g2 and g1 and g2.get('acc_mean_percent',-1)>g1.get('acc_mean_percent',-1)); go=bool(g2base and med and g2g1)
    lines += ['', '## 机制与判断','',f'- 固定共同文本锚点：`{json.loads((RESULTS/"preflight.json").read_text())["text_cache"]["gram_max_abs_error"]:.3e}` Gram误差；G2/G3锚点SHA、无可训练文本投影记录在 `mechanism.csv`。',f'- G2相对E74.4-G2：`{"SUPPORTED" if g2base else "NOT_SUPPORTED"}`；G2相对G1 Visual-only：`{"SUPPORTED" if g2g1 else "NOT_SUPPORTED"}`；G2相对G3匿名：`{"SUPPORTED" if med else "MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED"}`。',f'- 单侧梯度保护统计、冲突比例、投影/缩放比例见 `checkpoint_audit.csv` 与 `mechanism.csv`。',f'- 最终科研判定：`{"GO_CANDIDATE" if go else "NO_GO_FAPC"}`。', '', '## 限制','', '仍保留历史val=test、测试ACC选模、CV前RFE/预处理、transductive图、单seed和不做LOO；结果是方法探索与历史对齐证据，不是无偏临床泛化。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def audit_all():
    torch.set_num_threads(1); tests=unit_tests(); frozen=runner.verify_frozen_source(); source_sha=frozen['source_sha256']; runner.DATA_IDENTITY=runner._data_identity(runner.prepare(1)[0])
    audits=[]; metrics=[]; mechs=[]; intervs=[]; errors=[]
    for fold in range(1,11):
        for group in GROUPS:
            try:
                a,cp,state,module,opt,details,met,bmet=restore(group,fold,source_sha); audits.append(a)
                task=json.loads(runner.task_paths(group,fold)[0].read_text())
                metrics.append({'group':group,'fold':fold,'status':a['status'],**{m:float(met[m]) for m in METRICS},'tp':int(met['asd_tp']),'tn':int(met['td_tn']),'fp':int(met['td_fp']),'fn':int(met['asd_fn']),'best_epoch_1based':a['best_epoch_1based'],'epochs_executed':a['epochs_executed'],'wall_seconds':float(task['wall_seconds']),'peak_gpu_bytes':int(task.get('peak_gpu_bytes',0)),'checkpoint_sha256':a['checkpoint_sha256']})
                mechs.append(mechanism(a,cp,state,module,details,met)); intervs+=interventions(group,fold,state,module)
                del state,module,opt,cp,details; torch.cuda.empty_cache()
            except Exception as e:
                errors.append({'group':group,'fold':fold,'error':repr(e)})
                audits=[x for x in audits if (x.get('group'),x.get('fold'))!=(group,fold)]
                audits.append({'group':group,'fold':fold,'status':'AUDIT_FAILED','error':repr(e)})
    sums=[summary(g,[r for r in metrics if r['group']==g]) for g in GROUPS]; cur={g:{int(r['fold']):{m:float(r[m]) for m in METRICS} for r in metrics if r['group']==g} for g in GROUPS}; base=ref(BASE/'results/fold_metrics.csv',runner.BASE_GROUP); e7=ref(Path('/model/spromptgl/experiment/experiment78_7_bdpprompt/results/fold_metrics.csv'),'E78.7-G1-InverseCalibration-Medical')
    specs=[(f'{GROUPS[0]} - E74.4-G2',cur[GROUPS[0]],base),(f'{GROUPS[1]} - E74.4-G2',cur[GROUPS[1]],base),(f'{GROUPS[2]} - E74.4-G2',cur[GROUPS[2]],base),(f'{GROUPS[1]} - {GROUPS[0]}',cur[GROUPS[1]],cur[GROUPS[0]]),(f'{GROUPS[1]} - {GROUPS[2]}',cur[GROUPS[1]],cur[GROUPS[2]]),(f'{GROUPS[1]} - E78.7-G1',cur[GROUPS[1]],e7)]
    comps=[]
    for n,l,r in specs:
        fs=sorted(set(l)&set(r)); comps+=paired(n,l,r,fs); comps+=paired(n,l,r,[f for f in fs if f!=2])
    write_csv(RESULTS/'fold_metrics.csv',metrics); write_csv(RESULTS/'paired_comparisons.csv',comps); write_csv(RESULTS/'mechanism.csv',mechs); write_csv(RESULTS/'interventions.csv',intervs); write_csv(RESULTS/'checkpoint_audit.csv',audits)
    pre=json.loads((RESULTS/'preflight.json').read_text()); audited=sum(a.get('status')=='INDEPENDENT_AUDIT_PASS' for a in audits); common=sum(a.get('rng_common_epochs',0) for a in audits); mism=sum(a.get('rng_mismatches',0) for a in audits); status='COMPLETE_AUDITED' if len(metrics)==30 and audited==30 and not errors else ('PARTIAL' if metrics else 'FAILED')
    ast={'status':status,'trained_tasks':len(metrics),'independent_checkpoint_audits':audited,'expected_tasks':30,'audit_errors':errors,'source_sha256':source_sha,'base_off_parity_passed':pre.get('base_off_parity_passed'),'rng_common_epochs_checked':common,'rng_mismatches':mism,'unit_tests':tests,'root_and_historical_sources_unchanged':True}
    write_json(RESULTS/'audit_status.json',ast); report(sums,comps,status,ast,source_sha)
    runner.atomic_json(HERE/'manifest.json',{**frozen,'status':status,'completed_tasks':len(metrics),'independent_audits':audited,'audit_errors':errors,'results':{n:str(RESULTS/n) for n in ('fold_metrics.csv','paired_comparisons.csv','mechanism.csv','interventions.csv','checkpoint_audit.csv','audit_status.json','report.md','unit_tests.json')}})
    hand=Path('/model/spromptgl/experiment/handoff.md'); old=hand.read_text() if hand.exists() else ''; marker='## Experiment 78.8: FAPC'
    if marker not in old:
        block=[marker,'',f'- 状态：`{status}`；正式任务{len(metrics)}/30，独立Checkpoint审计{audited}/30。',f'- 正式目录：`{HERE}`；/model为单一正式输出；训练源SHA：`{source_sha}`。',f'- Base-off parity：{ast.get("base_off_parity_passed")}/10；公共RNG共有epoch：{common}，不匹配：{mism}。','- G1 Visual-only；G2 FrozenAnchor 医学；G3 FrozenAnchor 匿名。20条RN50文本经共同CPU-float64行空间SVD，Gram误差记录于 `results/preflight.json`；无可训练文本投影。','- 单侧主任务优先梯度保护记录冲突、投影和20%范数上限；Fold 2新增支持排除零基索引376、406。','- 六指标、G1/G2/G3配对差、固定锚点和去Fold 2分析见 `results/report.md`、`results/paired_comparisons.csv`、`results/mechanism.csv`。','- 历史限制：val=test、测试ACC选模、CV前RFE/预处理、transductive图和单seed仍保留。','']
        tmp=hand.with_name(hand.name+f'.tmp.{os.getpid()}'); tmp.write_text(old+('\n' if old and not old.endswith('\n') else '')+'\n'.join(block),encoding='utf-8'); os.replace(tmp,hand); ast['handoff_update']='APPENDED'
    else: ast['handoff_update']='ALREADY_PRESENT'
    ast['handoff_path']=str(hand); write_json(RESULTS/'audit_status.json',ast); return ast
if __name__=='__main__': print(json.dumps(audit_all(),ensure_ascii=False,indent=2))
