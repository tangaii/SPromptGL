from __future__ import annotations
import argparse, copy, csv, fcntl, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'; TASKS = RT / 'tasks'; CKPTS = RT / 'checkpoints'; LOGS = RT / 'logs'; RESULTS = EXP / 'results'
ROOT = Path(os.environ.get('SPROMPTGL_ROOT', '/data/adodas/SPromptGL-main'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

import integration as I


def group_cfg(group_id):
    return next(g for g in I.CFG['groups'] if g['id'] == group_id)


def task_path(gid, fold): return TASKS / f'{gid}_f{fold:02d}.json'
def best_path(gid, fold): return CKPTS / f'{gid}_f{fold:02d}.best.pt'
def latest_path(gid, fold): return CKPTS / f'{gid}_f{fold:02d}.latest.pt'


def write_json(path, value): I.write_json(path, value)

def read_json(path): return I.read_json(path)

def save_pt(path, value): I.save_pt(path, value)
def load_pt(path): return I.load_pt(path)


def ensure_dirs():
    for p in [RT, TASKS, CKPTS, LOGS, RESULTS, RT / 'sources']:
        p.mkdir(parents=True, exist_ok=True)


def prepare():
    ensure_dirs()
    protected = {}
    for name in ['main.py','model.py','network.py','layers.py','utils.py']:
        protected[str(ROOT / name)] = I.sha(ROOT / name)
    for name in ['processed_standard_data.csv','modal_feat_dict.npy']:
        protected[str(I.DATA_DIR / name)] = I.sha(I.DATA_DIR / name)
    src_hash = {str(p.relative_to(EXP)): I.sha(p) for p in sorted(I.SRC.glob('*.py'))}
    x, y, mapping = I.load_data()
    split_info = {}
    for cv in I.CFG['cv_seeds']:
        folds = I.make_folds(x, y, cv)
        rows = []
        seen = []
        for f, (tr, te) in enumerate(folds, 1):
            assert len(set(tr).intersection(set(te))) == 0
            rows.append({'fold': f, 'train_n': len(tr), 'test_n': len(te), 'split_sha': I.split_sha(tr, te), 'train_label_counts': np.bincount(y[tr], minlength=2).tolist(), 'test_label_counts': np.bincount(y[te], minlength=2).tolist()})
            seen.extend(te.tolist())
        assert sorted(seen) == list(range(len(y)))
        split_info[str(cv)] = {'cv_seed': cv, 'all_split_sha': I.all_split_sha(folds), 'folds': rows}
    manifest = {
        'experiment': '49', 'status': 'PREPARED', 'created_unix': time.time(),
        'config_sha': I.sha(EXP / 'config.json'), 'source_signature': I.source_signature(),
        'isolated_source_sha256': src_hash, 'protected_sha256': protected,
        'data_shape': list(x.shape), 'label_counts': np.bincount(y, minlength=2).tolist(),
        'modal_dims': I.INPUT_DIMS, 'split_info': split_info,
        'planned_groups': 20, 'planned_folds': 200, 'seed_reset_per_fold': False,
        'historical_reference': I.CFG['historical']
    }
    write_json(EXP / 'manifest.json', manifest)
    print('PREPARE_PASS', json.dumps({'groups':20,'folds':200,'source_signature':manifest['source_signature']}, ensure_ascii=False), flush=True)


def preflight(gpu):
    ensure_dirs()
    c = I.get_runtime(); torch = c['torch']
    x, y, _ = I.load_data()
    checks = {'gpu': gpu, 'device': str(torch.cuda.get_device_name(0)), 'data_shape': list(x.shape)}
    hp = I.make_hp(); hp.nclass = 2
    # Different seed changes initialization; same seed reproduces it.
    I.set_seed(1); folds = I.make_folds(x, y, 11); a = I.make_agent(x, y, *folds[0], hp); s1 = I.digest_model_state(I.model_cpu(a))
    I.set_seed(1); b = I.make_agent(x, y, *folds[0], hp); s2 = I.digest_model_state(I.model_cpu(b)); assert s1 == s2
    I.set_seed(2); d = I.make_agent(x, y, *folds[0], hp); s3 = I.digest_model_state(I.model_cpu(d)); assert s3 != s1
    assert sum(p.numel() for p in a.MessagePassing0.parameters()) == 887836
    assert sum(p.numel() for p in a.ModalFusion.parameters()) == 49544
    assert abs(a.hyperpm.reg - 0.11) < 1e-12
    assert all(abs(g['weight_decay'] - 0.11) < 1e-12 for o in [a.optimizer_MP0, a.optimizer_MF] for g in o.param_groups)
    # Split identity checks.
    f11 = I.make_folds(x, y, 11); f11b = I.make_folds(x, y, 11); f21 = I.make_folds(x, y, 21)
    assert [I.split_sha(*z) for z in f11] == [I.split_sha(*z) for z in f11b]
    assert [I.split_sha(*z) for z in f11] != [I.split_sha(*z) for z in f21]
    # One finite original simple-2 step, outside formal groups.
    I.set_seed(1); q = I.make_agent(x, y, *f11[0], hp); vals = I.train_step(q)
    assert all(np.isfinite(v) for v in vals.values())
    checks.update({'status':'PASS','parameter_counts':{'GFR':887836,'CAM':49544},'wd':0.11,'same_seed_initial_equal':True,'different_seed_initial_different':True,'same_cv_split_equal':True,'different_cv_split_different':True,'finite_first_step':True,'seconds':0.0})
    write_json(RT / f'preflight_gpu{gpu}.json', checks)
    print('PREFLIGHT_PASS', json.dumps(checks, ensure_ascii=False), flush=True)


def row_from_task(r):
    keys = ['group_id','group','train_seed','cv_seed','fold','gpu','split_sha','all_split_sha','entry_rng_sha','post_init_rng_sha','terminal_rng_sha','best_epoch','epochs_executed','best_acc','acc','hard_auc','spe','sen','asd_tp','asd_fn','td_fp','td_tn','prediction_sha','model_sha','best_sha','wall_seconds','status']
    return {k:r.get(k) for k in keys if k in r}


def is_complete(gid, fold):
    p = task_path(gid, fold); b = best_path(gid, fold)
    if not p.exists() or not b.exists(): return False
    r = read_json(p)
    return r.get('status') == 'COMPLETE' and r.get('best_sha') == I.sha(b)


def agent_entry(x, y, train, test, hp, entry_rng):
    I.restore_rng(entry_rng)
    a = I.make_agent(x, y, train, test, hp)
    return a


def run_fold(g, fold, x, y, folds, hp):
    gid = g['id']; train, test = folds[fold - 1]
    tp, bp, lp = task_path(gid, fold), best_path(gid, fold), latest_path(gid, fold)
    if is_complete(gid, fold):
        art = load_pt(bp); I.restore_rng(art['terminal_rng']); return read_json(tp)
    start = time.monotonic()
    latest = load_pt(lp) if lp.exists() else None
    if latest is not None:
        entry_rng = latest['entry_rng']; a = agent_entry(x, y, train, test, hp, entry_rng)
        I.load_model(a, latest['model']); a.optimizer_MP0.load_state_dict(latest['mp']); a.optimizer_MF.load_state_dict(latest['mf']); I.restore_rng(latest['rng'])
        begin = latest['epoch'] + 1; best_acc = latest['best_acc']; best_epoch = latest['best_epoch']; wait = latest['wait']; best = latest['best']; trace = latest['trace']; losses = latest['losses']; initial_rng = latest['entry_rng']; post_init_rng = latest['post_init_rng'];
        print('RESUME', gid, fold, begin, flush=True)
    else:
        initial_rng = I.rng_pack(); a = agent_entry(x, y, train, test, hp, initial_rng); post_init_rng = I.rng_pack()
        begin = 0; best_acc = 0.0; best_epoch = -1; wait = 0; best = None; trace = []; losses = []
    base = {'group_id': int(gid[1:]), 'group': gid, 'train_seed': g['train_seed'], 'cv_seed': g['cv_seed'], 'fold': fold, 'gpu': g['gpu'], 'split_sha': I.split_sha(train, test), 'all_split_sha': I.all_split_sha(folds), 'entry_rng_sha': I.rng_sha(initial_rng), 'post_init_rng_sha': I.rng_sha(post_init_rng), 'source_signature': I.source_signature()}
    write_json(tp, dict(base, status='RUNNING', epoch=begin - 1, best_epoch=best_epoch, best_acc=best_acc))
    last_epoch = begin - 1
    for epoch in range(begin, hp.nepoch):
        vals = I.train_step(a)
        out = I.evaluate(a)[0]
        acc = float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
        last_epoch = epoch
        if acc > best_acc:
            best_acc = acc; best_epoch = epoch; wait = 0
            best = {'model': I.model_cpu(a), 'mp': copy.deepcopy(a.optimizer_MP0.state_dict()), 'mf': copy.deepcopy(a.optimizer_MF.state_dict()), 'rng': I.rng_pack(), 'epoch': epoch, 'best_acc': acc}
        else:
            wait += 1
        trace.append(I.rng_sha(I.rng_pack()))
        if epoch % 10 == 0 or epoch == hp.nepoch - 1: losses.append(dict(epoch=epoch, **vals, val_acc=acc))
        if (epoch + 1) % 10 == 0 or wait > hp.early or epoch == hp.nepoch - 1:
            save_pt(lp, dict(base, status='RUNNING', epoch=epoch, model=I.model_cpu(a), mp=a.optimizer_MP0.state_dict(), mf=a.optimizer_MF.state_dict(), rng=I.rng_pack(), entry_rng=initial_rng, post_init_rng=post_init_rng, best=best, best_acc=best_acc, best_epoch=best_epoch, wait=wait, trace=trace, losses=losses))
            write_json(tp, dict(base, status='RUNNING', epoch=epoch, best_epoch=best_epoch, best_acc=best_acc))
        if wait > hp.early: break
    assert best is not None and best_epoch >= 0
    terminal_rng = I.rng_pack()
    I.load_model(a, best['model']); out = I.evaluate(a)[0]
    met = I.metric_row(out, y, test); pred = met.pop('prediction')
    artifact = dict(base, status='COMPLETE', model=best['model'], mp=best['mp'], mf=best['mf'], best_rng=best['rng'], entry_rng=initial_rng, post_init_rng=post_init_rng, terminal_rng=terminal_rng, best_epoch=best_epoch, epochs_executed=last_epoch + 1, best_acc=best_acc, wait=wait, prediction=pred, test_indices=np.asarray(test, dtype=np.int64), metrics=met, trace=trace, losses=losses)
    save_pt(bp, artifact)
    art2 = load_pt(bp); assert I.digest_model_state(art2['model']) == I.digest_model_state(best['model'])
    row = dict(base, status='COMPLETE', best_epoch=best_epoch, epochs_executed=last_epoch + 1, best_acc=best_acc, **met, prediction_sha=I.digest_obj(pred), model_sha=I.digest_model_state(best['model']), best_sha=I.sha(bp), terminal_rng_sha=I.rng_sha(terminal_rng), wall_seconds=time.monotonic() - start, trace_len=len(trace))
    write_json(tp, row)
    if lp.exists(): lp.unlink()
    print('FOLD_DONE', gid, fold, 'acc', round(met['acc'], 6), 'best', best_epoch, 'epochs', last_epoch + 1, flush=True)
    return row


def worker(group_id):
    g = group_cfg(group_id); assert int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]) >= 0
    I.set_seed(g['train_seed'])
    x, y, _ = I.load_data(); folds = I.make_folds(x, y, g['cv_seed']); hp = I.make_hp();
    started = time.time(); rows=[]
    for f in range(1, 11):
        rows.append(run_fold(g, f, x, y, folds, hp))
    write_json(RT / f'{group_id}.json', {'group':group_id,'status':'COMPLETE','train_seed':g['train_seed'],'cv_seed':g['cv_seed'],'folds':10,'wall_seconds':time.time()-started})
    print('GROUP_COMPLETE', group_id, flush=True)


def launch_controller():
    for gpu in [0, 1]:
        p = RT / f'preflight_gpu{gpu}.json'; assert p.exists() and read_json(p)['status'] == 'PASS'
    groups = {0: list(I.CFG['gpu_groups']['0']), 1: list(I.CFG['gpu_groups']['1'])}
    queues = {k:list(v) for k,v in groups.items()}; live=[]; started=time.time()
    envbase = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    while queues[0] or queues[1] or live:
        for gpu in [0,1]:
            mine=[z for z in live if z[0] == gpu]
            while queues[gpu] and len(mine) < 5:
                gid=queues[gpu].pop(0); log=(LOGS / f'{gid}_gpu{gpu}.log').open('a', buffering=1)
                env=dict(envbase, CUDA_VISIBLE_DEVICES=str(gpu))
                p=subprocess.Popen([sys.executable,'-B',str(EXP/'run.py'),'worker','--group',gid,'--gpu',str(gpu)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                live.append((gpu,gid,p,log)); mine.append((gpu,gid,p,log)); print('LAUNCH',gid,'gpu',gpu,flush=True)
        time.sleep(3)
        for item in live[:]:
            gpu,gid,p,log=item
            if p.poll() is not None:
                log.close(); live.remove(item)
                if p.returncode != 0:
                    for _,_,q,l in live: q.terminate(); l.close()
                    write_json(RT/'controller_failure.json', {'group':gid,'gpu':gpu,'returncode':p.returncode,'elapsed':time.time()-started})
                    raise RuntimeError(f'worker failed {gid}, inspect {LOGS/gid}_gpu{gpu}.log')
    assert all((RT/f'{g["id"]}.json').exists() for g in I.CFG['groups'])
    m=read_json(EXP/'manifest.json'); m.update(status='TRAINING_COMPLETE', formal_started_unix=started, formal_finished_unix=time.time(), formal_wall_seconds=time.time()-started, completed_folds=200, groups_complete=20); write_json(EXP/'manifest.json',m)
    print('TRAINING_COMPLETE 200/200',flush=True)


def collect_rows():
    rows=[]
    for g in I.CFG['groups']:
        for f in range(1,11):
            p=task_path(g['id'],f); assert p.exists(),p
            r=read_json(p); assert r['status']=='COMPLETE'; rows.append(r)
    assert len(rows)==200
    return sorted(rows,key=lambda r:(r['group_id'],r['fold']))


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();
        for r in rows: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()})


def aggregate():
    rows=collect_rows(); write_csv(RESULTS/'fold_metrics.csv',[row_from_task(r) for r in rows])
    summary=[]
    for g in I.CFG['groups']:
        rr=[r for r in rows if r['group']==g['id']]
        z={'group':g['id'],'train_seed':g['train_seed'],'cv_seed':g['cv_seed'],'folds':len(rr)}
        for k in I.CFG['metrics']:
            vals=100*np.asarray([r[k] for r in rr],float); z[k+'_mean']=float(vals.mean()); z[k+'_sd']=float(vals.std(ddof=0))
        z['best_fold_acc']=int(np.argmax([r['acc'] for r in rr])+1); summary.append(z)
    write_csv(RESULTS/'group_summary.csv',summary)
    factor=[]
    overall={k:float(np.mean([s[k+'_mean'] for s in summary])) for k in I.CFG['metrics']}
    factor.append({'kind':'overall','train_seed':'all','cv_seed':'all',**{k:overall[k] for k in I.CFG['metrics']}})
    for ts in I.CFG['train_seeds']:
        rr=[s for s in summary if s['train_seed']==ts]
        factor.append({'kind':'train_seed_margin','train_seed':ts,'cv_seed':'all',**{k:float(np.mean([s[k+'_mean'] for s in rr])) for k in I.CFG['metrics']}})
    for cs in I.CFG['cv_seeds']:
        rr=[s for s in summary if s['cv_seed']==cs]
        factor.append({'kind':'cv_seed_margin','train_seed':'all','cv_seed':cs,**{k:float(np.mean([s[k+'_mean'] for s in rr])) for k in I.CFG['metrics']}})
    for ts in I.CFG['train_seeds']:
        rr=[s for s in summary if s['train_seed']==ts]
        for k in I.CFG['metrics']: factor.append({'kind':'train_fixed_cv_spread','train_seed':ts,'cv_seed':'all','metric':k,'mean':float(np.mean([s[k+'_mean'] for s in rr])),'sd':float(np.std([s[k+'_mean'] for s in rr],ddof=0)),'min':float(np.min([s[k+'_mean'] for s in rr])),'max':float(np.max([s[k+'_mean'] for s in rr]))})
    for cs in I.CFG['cv_seeds']:
        rr=[s for s in summary if s['cv_seed']==cs]
        for k in I.CFG['metrics']: factor.append({'kind':'cv_fixed_train_spread','train_seed':'all','cv_seed':cs,'metric':k,'mean':float(np.mean([s[k+'_mean'] for s in rr])),'sd':float(np.std([s[k+'_mean'] for s in rr],ddof=0)),'min':float(np.min([s[k+'_mean'] for s in rr])),'max':float(np.max([s[k+'_mean'] for s in rr]))})
    # Descriptive two-factor sums of squares for group means.
    for k in I.CFG['metrics']:
        vals=np.array([s[k+'_mean'] for s in summary]); grand=vals.mean(); trm={ts:np.mean([s[k+'_mean'] for s in summary if s['train_seed']==ts]) for ts in I.CFG['train_seeds']}; cvm={cs:np.mean([s[k+'_mean'] for s in summary if s['cv_seed']==cs]) for cs in I.CFG['cv_seeds']};
        ss_total=float(np.sum((vals-grand)**2)); ss_tr=float(4*sum((v-grand)**2 for v in trm.values())); ss_cv=float(5*sum((v-grand)**2 for v in cvm.values())); ss_res=max(0.0,ss_total-ss_tr-ss_cv)
        factor.append({'kind':'descriptive_ss','metric':k,'grand_mean':float(grand),'ss_total':ss_total,'ss_train':ss_tr,'ss_cv':ss_cv,'ss_residual':ss_res,'train_fraction':ss_tr/ss_total if ss_total else 0.0,'cv_fraction':ss_cv/ss_total if ss_total else 0.0})
    write_csv(RESULTS/'factor_summary.csv',factor)
    # OOF rows are sourced from each saved artifact.
    x,y,_=I.load_data(); oof=[]
    for g in I.CFG['groups']:
        for f in range(1,11):
            art=load_pt(best_path(g['id'],f)); pred=np.asarray(art['prediction']); te=np.asarray(art['test_indices'])
            for idx,pred_i in zip(te,pred): oof.append({'group':g['id'],'train_seed':g['train_seed'],'cv_seed':g['cv_seed'],'fold':f,'index':int(idx),'label':int(y[idx]),'prediction':int(pred_i)})
    write_csv(RESULTS/'oof_predictions.csv',oof)
    m=read_json(EXP/'manifest.json');m.update(status='AGGREGATED',folds_complete=200);write_json(EXP/'manifest.json',m)
    print('AGGREGATE_PASS 200 folds',flush=True)


def fmt(mean,sd): return f'{mean:.4f} ± {sd:.4f}'

def report():
    import statistics
    rows=collect_rows();
    summaries=[]
    for g in I.CFG['groups']:
        rr=[r for r in rows if r['group']==g['id']]; z={'group':g['id'],'train_seed':g['train_seed'],'cv_seed':g['cv_seed']}
        for k in I.CFG['metrics']:
            vals=100*np.array([r[k] for r in rr]);z[k]=(float(vals.mean()),float(vals.std(ddof=0)))
        summaries.append(z)
    lines=['# Experiment 49 — B0-B双随机种子敏感性与稳定性实验','','## 执行状态', '', '正式完成 20/20 组、200/200 folds。每组在固定 GPU 上按 fold 1–10 顺序使用一个连续训练 RNG；GPU0 执行 E01、E03、E06、E08、E09、E11、E14、E16、E17、E19，GPU1 执行其余 10 组。没有根据中间结果修改配置。', '', '模型保持 B0-B 物理移除 PEN、GFR+CAM、原始两层 GCN、PHENO 覆盖、weighted-cosine 阈值 0.9、GraphConstructLoss、weight_decay=0.11、simple-2 和历史 val=test 选模。', '', '## 20组正式结果（%、十折 mean ± population SD）', '', '|组号|train_seed|cv_seed|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['group']}|{s['train_seed']}|{s['cv_seed']}|{fmt(*s['acc'])}|{fmt(*s['hard_auc'])}|{fmt(*s['spe'])}|{fmt(*s['sen'])}|")
    for k in I.CFG['metrics']:
        vals=np.array([s[k][0] for s in summaries]); lines += ['', f'20组 {k} 组合均值={vals.mean():.4f}，组合SD={vals.std(ddof=0):.4f}，中位数={np.median(vals):.4f}，范围=[{vals.min():.4f}, {vals.max():.4f}]。']
    hist=I.CFG['historical']; accs=np.array([s['acc'][0] for s in summaries]); aucs=np.array([s['hard_auc'][0] for s in summaries]);
    best=summaries[int(np.argmax(accs))]; worst=summaries[int(np.argmin(accs))]
    lines += ['', '## 历史参考和极值', '', f"历史 seed=0、cv_random_state=10：ACC {hist['acc']:.4f}%、Hard-AUC {hist['hard_auc']:.4f}%、SPE-TD {hist['spe']:.4f}%、SEN-ASD {hist['sen']:.4f}%。", f"20组观察最大 ACC 为 {best['group']} ({accs.max():.4f}%)；最小 ACC 为 {worst['group']} ({accs.min():.4f}%)。观察最大值不代表无偏性能，也不构成方法改进。历史 ACC 高于/低于20组组合均值的关系仅作描述。", '', '## 训练seed与CV划分seed敏感性', '', '固定 cv_seed 时，训练 seed 的变化由同一划分上的5个组合均值描述；固定 train_seed 时，cv_seed 的变化由同一初始化种子上的4个组合均值描述。完整边际、范围、组合间描述性平方和见 `factor_summary.csv`。本实验没有把这些重复组合当作独立临床样本，也没有进行缺乏独立重复支撑的显著性交互检验。']
    for cs in I.CFG['cv_seeds']:
        rr=[s for s in summaries if s['cv_seed']==cs]; vals=[s['acc'][0] for s in rr]; lines.append(f'- cv_seed={cs}：固定划分下 train_seed ACC 范围 {min(vals):.4f}–{max(vals):.4f}%，SD={np.std(vals,ddof=0):.4f}%。')
    for ts in I.CFG['train_seeds']:
        rr=[s for s in summaries if s['train_seed']==ts]; vals=[s['acc'][0] for s in rr]; lines.append(f'- train_seed={ts}：固定初始化种子下 cv_seed ACC 范围 {min(vals):.4f}–{max(vals):.4f}%，SD={np.std(vals,ddof=0):.4f}%。')
    lines += ['', '## 解释与限制', '', '本轮结果只回答 B0-B 在指定五个训练 seed、四个 CV seed 和历史训练协议下的稳定性。val=test 参与 checkpoint 选择、外层 CV 前 RFE/标准化、transductive 全队列输入、单次数据集和共享训练患者会造成评估偏倚；200 个测试折不能当作200个独立数据集。', '', '本实验没有加入 CAM-only 配对组，不能据此证明 GFR 具有新的患者关系增量；也没有使用 Experiment47.2/47.3 的 weight decay=0、rho、中心化、固定图或关系错配。', '', '## 工程审计', '', '已核验：20组配置唯一，200/200 folds完成，每组10个OOF测试集合，固定CV seed的split SHA一致、不同CV seed的split SHA不同，B0-B参数量为GFR 887,836/CAM 49,544，PEN/CLIP文本路径不存在，weight_decay=0.11，原始loss和checkpoint规则保持一致。每个组合的完成fold末尾 RNG 被恢复后进入下一fold。', '', '## 最终结论', '', '本轮是描述性稳健性研究。应使用20组完整结果报告训练seed和数据划分敏感性，并在后续方法实验中至少保留多训练seed或多划分复核；不能从最高ACC组合选择性宣称模型性能。']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
    m=read_json(EXP/'manifest.json');m.update(status='REPORT_COMPLETE',report_sha=I.sha(RESULTS/'report.md'),report_generated_unix=time.time());write_json(EXP/'manifest.json',m)
    print('REPORT_PASS',flush=True)


def append_handoff():
    marker='## Experiment 49 — B0-B Dual-Seed Robustness Study'
    p=ROOT/'handoff.md'; report_text=(RESULTS/'report.md').read_text()
    if marker in p.read_text(errors='replace'):
        return 'ALREADY_PRESENT'
    import fcntl
    summary='\n\n'+marker+'\n\n- 状态：20/20组、200/200 folds完成；GPU0执行E01、E03、E06、E08、E09、E11、E14、E16、E17、E19，GPU1执行其余10组；固定组合内fold1–10连续消费单一训练RNG。\n- B0-B协议：物理移除PEN，GFR+CAM、原始四图/两层GCN、PHENO覆盖、GraphConstructLoss、weight_decay=0.11、simple-2和val=test均保留；没有使用47.2/47.3的rho、中心化、固定图或关系错配。\n- 结果文件：`experiment/experiment49_b0b_dual_seed_robustness/results/fold_metrics.csv`（200行）、`group_summary.csv`（20行）、`factor_summary.csv`、`oof_predictions.csv`和`report.md`。\n- 历史参考：seed=0、cv_random_state=10 的 B0-B 为 ACC 91.7294%、Hard-AUC 91.6046%、SPE-TD 93.1360%、SEN-ASD 90.0732%；20组最高值只作描述性观察，不能作为无偏性能或方法增益。\n- 审计：20组配置唯一；200/200完整；相同CV seed的split SHA一致、不同CV seed不同；GFR/CAM参数量887,836/49,544；PEN/CLIP文本路径不存在；weight_decay=0.11；OOF覆盖与checkpoint独立读回通过。\n- 限制：val=test选模、外层前RFE/标准化、transductive全队列、单数据集和共享训练患者均保留，结果是内部稳健性描述，不是无偏独立泛化证据，也没有证明GFR相对CAM-only的独立增量。\n- 详细报告：`experiment/experiment49_b0b_dual_seed_robustness/results/report.md`。\n'
    with p.open('r+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX); f.seek(0,2); f.write(summary.encode()); f.flush(); os.fsync(f.fileno())
    return 'APPENDED'


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare')
    q=sub.add_parser('preflight');q.add_argument('--gpu',type=int,required=True)
    q=sub.add_parser('worker');q.add_argument('--group',required=True);q.add_argument('--gpu',type=int,required=True)
    sub.add_parser('controller');sub.add_parser('aggregate');sub.add_parser('report');sub.add_parser('handoff')
    args=ap.parse_args()
    if args.cmd=='prepare': prepare()
    elif args.cmd=='preflight':
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);preflight(args.gpu)
    elif args.cmd=='worker':
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);worker(args.group)
    elif args.cmd=='controller': launch_controller()
    elif args.cmd=='aggregate': aggregate()
    elif args.cmd=='report': report()
    elif args.cmd=='handoff': print(append_handoff())

if __name__=='__main__': main()
