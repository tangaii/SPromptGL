"""Post-training verification/delivery only; never changes training code/signatures.
CUDA_VISIBLE_DEVICES=1 python -B audit.py --verify  # optional full best reload
python -B audit.py --deliver                     # cached audit + bounded delivery
"""
import argparse,collections,fcntl,hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import run as r
HERE=Path(__file__).resolve().parent
FORMAL=r.ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest'

def verify():
    c=r.runtime();t=c['torch'];checks={};joint=[]
    for path in sorted((r.RT/'tasks').glob('g*_f*.json')):
        row=r.read(path);assert row['status']=='COMPLETE';g,f=row['group_id'],row['fold'];p=r.RT/'checkpoints'/f'g{g}_f{f:02}.best.pt';assert r.sha(p)==row['best_sha'];s=r.load(p);a=r.fresh(f);r.setmodel(a,s['model']);c['method'].install(a,g)
        out=r.evaluate(a)[0];pred=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64);mt=r.metric_row(out,r.data()[1],a.tst_idx)
        assert np.array_equal(pred,s['prediction']) and hashlib.sha256(pred.tobytes()).hexdigest()==row['prediction_sha256']
        assert all(abs(mt[k]-row[k])<1e-12 for k in mt)
        target=a.targ.clone();a.targ[a.tst_idx]=1-a.targ[a.tst_idx];changed=r.evaluate(a)[0];a.targ=target;assert t.equal(out,changed)
        anchor=r.load(r.RT/'anchors'/f'fold{f:02}.pt');assert not anchor['mp']['state'] and not anchor['mf']['state'];assert anchor['parameter_sha']==row['initial_parameter_sha256'] and anchor['signature']==row['signature'] and r.rngsha(anchor['entry_rng'])==row['entry_full_rng_sha256']
        checks[str((g,f))]=dict(best_sha_verified=True,best_forward_predictions_and_four_metrics_recomputed=True,heldout_label_permutation_logp_bitwise_invariant=True,untrained_two_Adam_anchor_verified=True)
        before=r.digest_state(a);handles=[getattr(a.MessagePassing0,f'MessagePassing{i}').register_forward_hook(lambda m,x,o:t.zeros_like(o)) for i in range(4)];off=r.evaluate(a)[0]
        for h in handles:h.remove()
        assert t.equal(out,r.evaluate(a)[0]) and before==r.digest_state(a)
        joint.append(dict(group=r.GROUPS[g],fold=f,test_count=len(a.tst_idx),class_flip_count=int((out[a.tst_idx].argmax(1)!=off[a.tst_idx].argmax(1)).sum()),mean_abs_logp_change=float((out[a.tst_idx]-off[a.tst_idx]).abs().mean()),restore_bitwise_exact=True))
    assert len(checks)==len(joint)==60
    r.write(r.RT/'final_check.json',dict(status='PASS',folds_verified=60,evidence=checks));r.write(r.RT/'joint_message_off.json',dict(status='PASS',folds=60,rows=joint))

def compare(rows,left,right):
    a={z['fold']:z for z in rows if z['group_id']==left};b={z['fold']:z for z in rows if z['group_id']==right};assert len(a)==len(b)==10
    dd={k:[100*(a[f][k]-b[f][k]) for f in range(1,11)] for k in r.CFG['metrics']};result={f'delta_{k}_pp':float(np.mean(v)) for k,v in dd.items()}
    for k in ['acc','hard_auc']:
        v=np.asarray(dd[k]);result[k+'_wtl']=[int((v>1e-10).sum()),int((abs(v)<=1e-10).sum()),int((v<-1e-10).sum())]
    result['per_fold_pp']=dd;return result

def summary():
    m=r.read(HERE/'manifest.json');assert m.get('scientific')=='COMPLETE' and m['folds_complete']==60 and m['anchor_replay']=='PASS'
    rows,diags=r.refresh();assert len(rows)==60 and {row['seed'] for row in rows}=={0};r.verify_sources()
    check=r.read(r.RT/'final_check.json');joint=r.read(r.RT/'joint_message_off.json');assert check['status']==joint['status']=='PASS' and check['folds_verified']==joint['folds']==60
    for row in rows:assert r.completed(row['group_id'],row['fold'])
    inputs=m['handoff_prefix'];original=(r.ROOT/'handoff.md').read_bytes();assert hashlib.sha256(original[:inputs['bytes']]).hexdigest()==inputs['sha256']
    counts={str(g):sum(z['physical_gpu']==g for z in rows) for g in (0,1)};assert counts=={'0':30,'1':30}
    pairs={f'{r.GROUPS[a]} - {r.GROUPS[b]}':compare(rows,a,b) for a,b in [(2,1),(3,1),(4,1),(5,3)]}
    times=[(r.RT/'anchors'/f'fold{f:02}.pt').stat().st_mtime for f in range(1,11)]
    commits=[(r.RT/'tasks'/f'g{g}_f{f:02}.json').stat().st_mtime for g in range(6) for f in range(1,11)]
    formal=max(commits)-min(times)
    text='\n<!-- E43_1_INDEPENDENT_FINAL_AUDIT -->\n## 独立读回、单因素配对及交付审计\n\n'
    text+='60/60最佳Checkpoint均已重新前向、重算四指标与混淆并核对预测SHA；全部held-out标签置换后的log-prob逐位不变。十个锚点的双Adam均为空的未训练初态，源/数据/split/RNG绑定通过。根源码、输入及受保护历史产物SHA保持不变。\n\n'
    text+='|相对声明前置组|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|\n|---|---:|---:|---:|---:|---|---|\n'
    for label,p in pairs.items():text+='|'+label+'|'+'|'.join(f"{p['delta_'+k+'_pp']:+.4f}" for k in r.CFG['metrics'])+'|'+'/'.join(map(str,p['acc_wtl']))+'|'+'/'.join(map(str,p['hard_auc_wtl']))+'|\n'
    text+='\nE03/E05定义重复；十折最佳模型、预测和epoch完全一致，不是两个独立改进。校准相对B0-B提高均值，但仅4/10折严格胜，未达7折门槛；相对历史B0-A的ACC仍低约0.1149pp、Hard-AUC低约0.1250pp。其SEN相对B0-B下降约1.2439pp，SPE增加约1.4940pp，须同时披露。\n\n'
    text+='|固定最佳模型同时关闭四支消息|测试类别翻转/871|平均绝对log-prob变化|\n|---|---:|---:|\n'
    for name in r.GROUPS:
        j=[z for z in joint['rows'] if z['group']==name];text+=f"|{name}|{sum(z['class_flip_count'] for z in j)}/871|{np.mean([z['mean_abs_logp_change'] for z in j]):.9g}|\n"
    text+='\n以上只验证固定权重下的依赖，不能替代从头训练的无GFR组，也不能据此声称患者图无用。E01恢复四支分类梯度，但没有分类增益。E04辅助消息相对输入约1e−9量级，CORR约1e−4；本轮未支持限制传播到模态块。消息微弱的原因没有被单独隔离，不归因为某个未经对照的超参数。\n\n'
    text+='E02十折lambda范围0.00378618–0.01027879，首步缩放监督/分类投影梯度比均为0.10。第1折PHENO监督梯度仅4.34e−9；非零不能视作实质监督。首步相对E01的参数差异见`.runtime/supervision_update_audit.json`，CORR最明显；Adam首步对梯度幅值的归一化意味着不能仅凭其他模态首步参数差很小就断言整个训练无监督作用。\n\n'
    text+=f'正式训练区间（首锚点落盘至最后结果提交，含存储恢复）为{formal:.2f}秒；本次总执行墙钟见manifest.total_wall_seconds，起点为存储预检。GPU0/GPU1各30折，其中创新各25折、基线各5折。预检峰值显存0.4684GiB，不冒称它是全部正式worker的全程峰值。\n\n'
    text+='存储初检PASS，后续OSSFS原子替换任务JSON时发生瞬时ENOENT，导致汇总器退出；原worker完成基线，无完成折重训。全部文件字节校验后迁移到/tmp继续，同一代码/训练签名不变。最终模型相等性比较复用单卡runtime，首次双卡可见控制器在报告阶段断言失败；以`CUDA_VISIBLE_DEVICES=1`重启控制器后全部已完成折直接跳过，报告成功；子进程仍按各自物理GPU分配。早期预检的计数/诊断键修正发生在任何正式训练之前。失败记录均保留，不计为额外训练折。\n\n'
    text+=f'科研状态SCIENTIFIC_COMPLETE，结论NO-GO。正式目标`{FORMAL}`；本地保护副本`{HERE}`。交付以manifest.delivery与delivery_sync为准；/tmp不具持久存储保障。未提交/推送Git。\n'
    report=HERE/'results/report.md';old=report.read_text();marker='<!-- E43_1_INDEPENDENT_FINAL_AUDIT -->'
    if marker in old:old=old[:old.index(marker)]
    report.write_text(old.rstrip()+'\n'+text)
    m.update(factor_pairs=pairs,independent_best_readback='PASS_60_OF_60',heldout_label_inference_check='BITWISE_INVARIANT_60_OF_60',gpu_folds=counts,formal_wall_seconds=formal,formal_interval_definition='first immutable anchor mtime to last unique result commit mtime, includes bounded recovery',audit_code_sha256=r.sha(HERE/'audit.py'),posttraining_audit=text,scientific='COMPLETE',delivery='SYNC_PENDING',actual_directory=str(HERE),formal_directory=str(FORMAL),resolved_failures=['Preflight-only parameter/state-dict and diagnostic-key corrections','OSSFS transient ENOENT during task replacement; unchanged code moved to local /tmp','Report-only single-device runtime assertion; restarted single-visible controller, all completed folds skipped'])
    m.pop('failure',None);m.pop('failed_command',None);r.write(HERE/'manifest.json',m)
    return text

def copy_checked(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);expected=r.sha(source)
    if target.exists() and r.sha(target)==expected:return
    tmp=target.with_name(target.name+'.delivery.tmp')
    with source.open('rb') as a,tmp.open('wb') as b:shutil.copyfileobj(a,b,2**20);b.flush();os.fsync(b.fileno())
    shutil.copystat(source,tmp);os.replace(tmp,target)
    for attempt in range(3):
        try:
            if r.sha(target)==expected:return
        except OSError:pass
        time.sleep(.2)
    raise IOError('delivery readback hash failure '+str(target))

def append_verified(text):
    path=r.ROOT/'handoff.md';marker='<!-- EXPERIMENT43_1_FINAL_VERIFIED_DELIVERY -->';payload='\n\n'+marker+'\n### Experiment 43.1 最终验收与交付\n'+text
    fd=os.open(path,os.O_RDWR|os.O_APPEND)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        with os.fdopen(os.dup(fd),'rb') as f:before=f.read()
        h=hashlib.sha256(before).hexdigest()
        if marker.encode() in before:return dict(status='ALREADY_APPENDED',prefix_sha256=h)
        b=payload.encode();written=0
        while written<len(b):written+=os.write(fd,b[written:])
        os.fsync(fd)
        after=path.read_bytes();assert after[:len(before)]==before and after[len(before):].startswith(b)
        return dict(status='APPENDED',prefix_bytes=len(before),prefix_sha256=h,after_sha256=hashlib.sha256(after).hexdigest())
    finally:os.close(fd)

def deliver(text):
    m=r.read(HERE/'manifest.json');files=[p for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts];mapping={str(p.relative_to(HERE)):r.sha(p) for p in files};total=sum(p.stat().st_size for p in files)
    try:
        if HERE!=FORMAL:
            for p in files:copy_checked(p,FORMAL/p.relative_to(HERE))
            assert all(r.sha(FORMAL/rel)==h for rel,h in mapping.items())
        m['delivery_sync']=dict(files_verified=len(mapping),bytes_verified=total,sha256_tree_excluding_manifest=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),all_files_bytewise_verified=True,source=str(HERE),target=str(FORMAL));m['delivery']='COMPLETE';m['status']='COMPLETE'
        note=text+f'\n本次有界同步成功：{len(mapping)}文件，{total}字节，逐文件SHA完全一致；正式目录已交付。\n'
        m['handoff_final']=append_verified(note)
    except Exception as e:
        m.update(delivery='PARTIAL',status='SCIENTIFIC_COMPLETE_DELIVERY_PARTIAL',delivery_error=repr(e));(HERE/'handoff_append_pending.md').write_text(text+'\nDelivery/handoff failure: '+repr(e)+'\n');print('DELIVERY_PARTIAL',repr(e),flush=True)
    r.write(HERE/'manifest.json',m)
    if HERE!=FORMAL:
        try:copy_checked(HERE/'manifest.json',FORMAL/'manifest.json')
        except Exception as e:
            m.update(delivery='PARTIAL',status='SCIENTIFIC_COMPLETE_DELIVERY_PARTIAL',manifest_delivery_error=repr(e));r.write(HERE/'manifest.json',m)
    return m  # avoid mutating a redirected delivery log after its hash snapshot

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--deliver',action='store_true');a=p.parse_args()
    if a.verify:verify()
    text=summary()
    if a.deliver:deliver(text)
