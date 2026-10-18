#!/usr/bin/env python3
"""GC-SRA frozen experiment. Infrastructure ported explicitly from accepted Exp18.
Reference algorithms live only in the six verbatim e23_*.py modules.
"""
import argparse, contextlib, csv, fcntl, hashlib, io, json, math, os, pickle, random
import shutil, subprocess, sys, time, types, traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='2'
if 'preflight' in sys.argv:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from e23_contract import CONFIGS, METRICS, CONTRASTS, INTERVENTIONS, tasks, feasibility_gate
from e23_bridge import CONDITIONS, GainRelationalBridge
from e23_prepare import load_fold
from e23_evidence import build_fold, readonly_evidence_digest
from e23_patch import emit_sources, patch_model23
EXP=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('SPROMPTGL_ROOT','/data/adodas/SPromptGL-main')).resolve()
STATE=EXP/'_runtime_sources'
MANIFEST,RESULTS,LOGS=EXP/'manifest.json',EXP/'results',EXP/'logs'
CORE=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
CODE=['README.md','design.json','run.py']+[f'e23_{n}.py' for n in ['evidence','bridge','patch','runtime','contract','prepare']]
TASKS=tasks()
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
    nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,
    n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',
    MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,
    theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save_checkpoint(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    atomic(path, b"SP23" + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP23" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
    return torch.load(io.BytesIO(data[68:]), map_location="cpu", weights_only=False)


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda"]])


def rng_hash(saved):
    h = hashlib.sha256(pickle.dumps((saved["python"], saved["numpy"]), protocol=4))
    for v in [saved["cpu"]] + saved["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    table = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    assert table.columns[:-1].tolist() == [str(v) for fields in mapping.values() for v in fields]
    values = table.values
    assert values.shape == (871, 321) and np.isfinite(values).all()
    x, y = values[:, :-1], values[:, -1].astype(int) - 1
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468
    pheno = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith(("AGE_AT_SCAN_", "SEX_"))]
    sites = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith("SITE_ID_")]
    assert len(pheno) == 28 and len(sites) == 20 and not set(pheno).intersection(sites)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    audit = dict(shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
                 modality_columns={k: list(map(str, v)) for k, v in mapping.items()},
                 pheno_age_sex_indices=pheno, site_indices=sites)
    return x, y, folds, audit


def model_state(a):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]}


def load_state(a, saved):
    a.MessagePassing0.load_state_dict(saved["gfr"])
    a.ModalFusion.load_state_dict(saved["pln"])


def weight_hash(a, auxiliary=False, legacy13=False):
    h = hashlib.sha256()
    for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]:
        for key, value in sorted(module.named_parameters()):
            if key.startswith("bridge23.") != auxiliary:
                continue
            tag = key if legacy13 else name + "." + key
            h.update(tag.encode() + value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a, config):
    groups = a.optimizer_MP0.param_groups + a.optimizer_MF.param_groups
    all_params = [p for module in [a.MessagePassing0, a.ModalFusion] for p in module.parameters()]
    supplied = [p for group in groups for p in group["params"]]
    assert len(supplied) == len(set(map(id, supplied)))
    assert set(map(id, supplied)) == set(map(id, all_params))
    assert all(g["lr"] == .0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]["weight_decay"] == .11
    assert a.optimizer_MF.param_groups[0]["weight_decay"] == .11
    total = sum(p.numel() for p in all_params)
    assert total == 1462957 + read_json(MANIFEST)["additional_parameters"][config]
    if config != "G0":
        assert len(a.optimizer_MF.param_groups) == 2
        assert a.optimizer_MF.param_groups[1]["weight_decay"] == 0.
    return total


def assert_equal_tree(a, b):
    import numpy as np
    import torch
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for v, w in zip(a, b):
            assert_equal_tree(v, w)
    else:
        assert a == b


def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def summary(rows):
    import numpy as np
    result = {}
    configs = sorted(set(r["config"] for r in rows), key=lambda v: (v[0], int(v[1:])))
    for config in configs:
        result[config] = {}
        for key in METRICS:
            seeds = [float(np.mean([float(r[key]) for r in rows if r["config"] == config and int(r["seed"]) == seed])) for seed in range(3)]
            result[config][key] = dict(mean=float(np.mean(seeds)), sd=float(np.std(seeds)), seeds=seeds)
    return result


def paired(rows, left, right, key):
    import numpy as np
    lookup = {(r["config"], int(r["seed"]), int(r["fold"])):float(r[key]) for r in rows}
    differences = np.array([[lookup[left, seed, fold] - lookup[right, seed, fold] for fold in range(1, 11)] for seed in range(3)])
    return dict(mean=float(differences.mean()), seeds=differences.mean(1).tolist(),
                wins=int((differences > 1e-12).sum()), ties=int((np.abs(differences) <= 1e-12).sum()), losses=int((differences < -1e-12).sum()))


def mean_field(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.


def history_e9():
    folder = next((ROOT / "experiment").glob("experiment9_*"))
    return [dict(r, ba=r["balanced_accuracy"], specificity_td=r["specificity_asd"])
            for r in read_csv(folder / "results/fold_metrics.csv") if r["config"] == "E9"]


def log(message):
    LOGS.mkdir(exist_ok=True)
    line=utc()+' '+message+'\n'
    with open(LOGS/'scheduler.log','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(line);f.flush()
    print(message,flush=True)


def append_handoff(stage):
    """One locked O_APPEND write; old bytes verified before and after, no overwrite."""
    m=read_json(MANIFEST)
    if stage in m.get('handoff_stages',[]):return
    path=ROOT/'handoff.md'
    header='实验二十三：增益校准的选择性患者关系对齐（GC-SRA）'
    f=read_json(RESULTS/'feasibility.json')
    decision=f['decision']
    lines=[f'\n\n## {header} — {stage} — {utc()}','',
        f'实验目录：`{EXP}`（/data创建返回I/O错误，按用户指令在/tmp执行）。',
        f'本地HEAD `{m["audit"]["git_head"]}`；远端 `{m["audit"]["remote_head"]}`。根六核心文件与远端相同；不修改实验1–22。',
        'G0原始P0；G1统一；G2同预算常量；G3同预算FC不确定度；G4训练内OOF条件增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系核错配。',
        '增益为FC与FC+年龄/性别Logistic教师OOF对数损失差；Ridge(alpha=150)预测；不是临床因果增益。新增关系通道只更新GFR输出CORR256，FC锚保留；零alpha、差分Decoder、有界残差，无新损失/疾病Prompt。',
        '动机：17/18表型共享与联合正确性未带来可靠增益；19中性优于真实且Key可重参数化；20/21固定语义图响应接近零；22η不敏感、正交残差未优于E9。',
        f'阶段状态：{m["run_status"]}；完成{m.get("completed_tasks",0)}/24任务、{m.get("completed_folds",0)}/240折。',
        f'快筛：mean_delta={decision["average_mean_delta"]:.9f}，top20={decision["average_high20_delta"]:.9f}，Spearman={decision["average_meta_oof_spearman"]:.9f}，gate SD={decision["average_train_gate_std"]:.9f}；判停={decision["early_stop"]}。',
        f'证据SHA `{m["evidence_sha256"]}`；核心/数据/参考代码SHA详见manifest。',
        '风险：保留test=val选模、全队列既有标准化/RFE、871人transductive与22轮探索；meta OOF一阶教师训练池重叠，只作探索诊断。Exp11已有数值身份恢复链，与Exp21仅因未找到subject_IDs.txt判0/871的有界搜索结论相矛盾；本轮不恢复身份、不使用临床量表。']
    if (RESULTS/'fold_metrics.csv').exists() and m.get('completed_folds')==240:
        rows=read_csv(RESULTS/'fold_metrics.csv');ss=summary(rows)
        for cfg in CONFIGS:
            lines.append(cfg+' '+', '.join(k+f'={ss[cfg][k]["mean"]*100:.5f}±{ss[cfg][k]["sd"]*100:.5f}%' for k in METRICS))
        lines.append('科学判据：'+json.dumps(judgment(rows),ensure_ascii=False))
        lines.append('G4同checkpoint文本无干预；REAL/GATE_OFF/GATE_CONSTANT/RELATION_OFF共120行，结果见report。')
        lines.append(f'训练墙钟秒：{m.get("training_wall_seconds")}; GPU任务按配置×seed交替，各120折。')
        lines.append('后续唯一建议：控制不成立则停止这套路线，优先取得具有真实互补信息的影像/临床模态或TADPOLE，不继续调本轮超参数。')
    else:
        lines.append('后续：仅在严格预检通过后执行冻结240折；不依据测试结果改设计。')
    content=('\n'.join(lines)+'\n').encode()
    try:
        # Reuse the existing project lock when possible. No temp handoff surrogate.
        lockpath=ROOT/'handoff.md.lock'
        with open(lockpath,'a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            before=path.read_bytes() if path.exists() else b''
            prefix=m['audit']
            assert len(before)>=prefix['handoff_prefix_bytes']
            assert hashlib.sha256(before[:prefix['handoff_prefix_bytes']]).hexdigest()==prefix['handoff_prefix_sha256']
            marker=f'## {header} — {stage} — '.encode()
            if marker not in before:
                fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
                try:
                    assert os.write(fd,content)==len(content);os.fsync(fd)
                finally:os.close(fd)
            after=path.read_bytes();assert after[:len(before)]==before
        m.setdefault('handoff_stages',[]).append(stage)
        m.get('handoff_pending',{}).pop(stage,None)
        m['handoff_status']='UPDATED'
        m['handoff_sha256']=sha(path)
        m['handoff_original_prefix_preserved']=True
    except OSError as e:
        m.setdefault('handoff_write_failures',[]).append(dict(stage=stage,utc=utc(),error=repr(e),target=str(path)))
        m['handoff_status']='BLOCKED_IO'
        m.setdefault('handoff_pending',{})[stage]=dict(text=content.decode(),sha256=hashlib.sha256(content).hexdigest())
        log('HANDOFF_WRITE_FAILED '+stage+' '+repr(e))
    save_json(MANIFEST,m)


def enrich_feasibility():
    import numpy as np
    from scipy.stats import spearmanr
    p=RESULTS/'feasibility.json';f=read_json(p)
    if 'additional_train_only_audits' in f:return
    x,y,folds,au=data();items=[]
    for fold,(tr,te) in enumerate(folds,1):
        ev=load_fold(EXP/'evidence.npz',fold);gate=ev['gain_gate'][tr];delta=ev['delta_train']
        def corr(a,b):
            if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.
            v=float(spearmanr(a,b).statistic);return v if math.isfinite(v) else 0.
        groups={}
        for col in au['pheno_age_sex_indices']+au['site_indices']:
            value=x[tr,col];levels=np.unique(value)
            # Standardized one-hot columns preserve their two-level ordering.
            mask=value==levels[-1]
            if len(levels)<=2 and mask.any():
                groups[au['modality_columns']['PHENO'][col]]=dict(n=int(mask.sum()),gate_mean=float(gate[mask].mean()),delta_mean=float(delta[mask].mean()))
        age=[i for i in au['pheno_age_sex_indices'] if au['modality_columns']['PHENO'][i].startswith('AGE_AT_SCAN_')]
        # Decode only one-hot category rank, not a claimed exact chronological age.
        rank=np.argmax(x[tr][:,age],axis=1)
        items.append(dict(fold=fold,delta_quantiles=np.quantile(delta,[0,.1,.25,.5,.75,.9,1]).tolist(),
            positive_delta_gt0_fraction=float((delta>0).mean()),delta_gt002_fraction=float((delta>.02).mean()),
            gain_gate_test_mean=float(ev['gain_gate'][te].mean()),gain_gate_test_std=float(ev['gain_gate'][te].std()),
            gain_vs_fc_uncertainty_spearman=corr(gate,1-2*abs(ev['pfc'][tr]-.5)),
            gain_vs_age_bin_rank_spearman=corr(gate,rank),groups=groups,
            site_used_as_gate_input=False,qc_may_correlate_with_site=True))
    f['additional_train_only_audits']=items
    f['clarifications']=['Gain scale uses clipped target SD as frozen reference code specifies.',
        'Train gain uses full Ridge fitted to all outer-train OOF scores; meta OOF is diagnostic only.',
        'positive_delta_fraction in the reference report means delta > 0.02, not > 0.',
        'Confidence control is FC uncertainty, monotonically budget matched on train only.',
        'OOF observations overlap across outer folds and are not ten independent cohorts.']
    save_json(p,f)



def initial_audit():
    """Self-contained read-only audit for a fresh run; no external temp dependency."""
    import re
    expected_core={'main.py': 'dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2', 'model.py': '254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda', 'network.py': '484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec', 'layers.py': '4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a', 'utils.py': '95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67', 'ABIDE-simple-2-concat-weighted-cosine.sh': '28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763'}
    expected_reference={'e23_bridge.py': 'cf83d3a4e5c73e59f22ad69ff88e33904196dddd5ad74f03a356129e33d84b7d', 'e23_contract.py': 'ec1d786debc20961ef4c651a8a270a30e0be268b49d847a4a78e506ab35cda90', 'e23_evidence.py': 'b8bb06f983e898d37ba7fbc1eec42d7d77c75627b83276fa89d23c9b8078217f', 'e23_patch.py': '787260aa79ec034424383243378b1fa0f028932e12fcca882d14330f09d505e3', 'e23_prepare.py': '6a283baadf8381be5131cc7b7e591db2719ebb1140933e40e2b5451f9be8dc55', 'e23_runtime.py': 'c359d18f408ffb9f3415b881628a7c99b29f46b15b1d1da0c7bb56e354c2c5eb'}
    for name,h in expected_core.items():assert sha(ROOT/name)==h,('unexpected baseline',name)
    for name,h in expected_reference.items():assert sha(EXP/name)==h,('reference changed',name)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','https://github.com/tangaii/SPromptGL','HEAD'],text=True).split()[0]
    protected={str(f.relative_to(ROOT)):sha(f) for folder in (ROOT/'experiment').iterdir()
        if folder.is_dir() and re.match(r'experiment(?:[1-9]|1[0-9]|2[0-2])_',folder.name)
        for f in folder.rglob('*') if f.is_file()}
    before=(ROOT/'handoff.md').read_bytes() if (ROOT/'handoff.md').exists() else b''
    return dict(started_utc=utc(),git_head=head,remote_head=remote,core=expected_core,
        remote_match='see frozen original manifest audit for original execution',upstream_head='ba3b9446fa451fde6864d6555bdf324de0c4b64a',
        upstream='Original execution audited RN50 caching and device compatibility differences; this function only validates frozen baseline.',
        handoff_prefix_bytes=len(before),handoff_prefix_sha256=hashlib.sha256(before).hexdigest(),protected_history=protected,
        verbatim_reference_sha256=expected_reference)


def prepare():
    if MANIFEST.exists():
        integrity(full=True);return read_json(MANIFEST)
    RESULTS.mkdir(exist_ok=True);LOGS.mkdir(exist_ok=True)
    audit=read_json('/tmp/sp23-audit.json') if Path('/tmp/sp23-audit.json').exists() else initial_audit()
    for n,h in audit['core'].items():assert sha(ROOT/n)==h
    for n,h in audit['verbatim_reference_sha256'].items():assert sha(EXP/n)==h
    if not (EXP/'evidence.npz').exists():
        from e23_prepare import execute
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,str(EXP/'e23_prepare.py'),str(ROOT),str(EXP)],env=env,cwd=EXP,check=True)
    enrich_feasibility()
    x,y,folds,au=data()
    f=read_json(RESULTS/'feasibility.json')
    assert feasibility_gate(f['folds'])==f['decision']
    for i in range(1,11):assert readonly_evidence_digest(load_fold(EXP/'evidence.npz',i))==f['evidence_digest_by_fold'][str(i)]
    design=dict(name='GC-SRA',configs={k:vars(v) for k,v in CONDITIONS.items()},tasks=TASKS,
        protocol=PROTOCOL,contrasts=CONTRASTS,metrics=METRICS,counterfactuals=INTERVENTIONS,
        top_k=12,temperature=.3,residual_limit=.10,reference_effective_bound=.05,
        additional_parameters={c:0 if c=='G0' else 30081 for c in CONFIGS},
        scientific_rule='All seven BA and ProbAUC seed differences strictly positive; all ACC and ASD SEN seed differences nonnegative. Conservative operationalization frozen before training; full continuous effects reported.',
        historical_e9_descriptive_only=True,root=str(ROOT),experiment=str(EXP),
        reference_patches=[],algorithm_modifications=[],
        prior_work=['SPromptGL 2025','MGDR 2024','To Align or Not to Align AAAI 2026','PhenoBrain ICML 2026'])
    save_json(EXP/'design.json',design)
    atomic(EXP/'README.md',readme_text())
    paths=[ROOT/n for n in CORE]+[ROOT/'data/ABIDE/processed_standard_data.csv',ROOT/'data/ABIDE/modal_feat_dict.npy',Path('/root/.cache/clip/RN50.pt')]
    m=dict(name='GC-SRA',created_utc=utc(),run_status='STOPPED_NEGATIVE_FEASIBILITY' if f['decision']['early_stop'] else 'PREPARED',
        audit=audit,data_audit=au,protocol=PROTOCOL,configurations=design['configs'],fixed_design=design,
        additional_parameters=design['additional_parameters'],source_data_sha256={str(p):sha(p) for p in paths},
        code_sha256={n:sha(EXP/n) for n in CODE},evidence_sha256=sha(EXP/'evidence.npz'),
        feasibility_sha256=sha(RESULTS/'feasibility.json'),
        split_sha256=canonical([[tr.tolist(),te.tolist()] for tr,te in folds]),
        reference_implementation_changes=[],runner_port=dict(source=str(ROOT/'experiment/experiment18_discriminability_preserving_selective_alignment/run.py'),
            source_sha256=sha(ROOT/'experiment/experiment18_discriminability_preserving_selective_alignment/run.py'),
            retained='data, RNG, atomic/checksummed checkpoints, state, optimizer audit, metric pairing; explicit worker epoch/fold restore and task state machine',
            replaced='Exp18 SelectiveAlignment/evidence/PLN patch removed; only six frozen GC-SRA reference modules and model two-site bridge patch used'),
        preflight=None,completed_tasks=0,completed_folds=0,diagnostic_rows=0,counterfactual_rows=0,
        tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS})
    save_json(MANIFEST,m);append_handoff('AUDIT_AND_STAGE0')
    log('PREPARED '+json.dumps(f['decision']))
    return read_json(MANIFEST)


def signature():
    m=read_json(MANIFEST)
    return canonical({k:m[k] for k in ['source_data_sha256','code_sha256','evidence_sha256','feasibility_sha256','protocol','fixed_design','split_sha256']})


def integrity(full=False):
    m=read_json(MANIFEST)
    for n,h in m['code_sha256'].items():assert sha(EXP/n)==h,('code changed',n)
    for path,h in m['source_data_sha256'].items():assert sha(path)==h,('source/data changed',path)
    assert sha(EXP/'evidence.npz')==m['evidence_sha256']
    assert sha(RESULTS/'feasibility.json')==m['feasibility_sha256']
    if full:
        for path,h in m['audit']['protected_history'].items():assert sha(ROOT/path)==h,('protected history changed',path)
        prefix=(ROOT/'handoff.md').read_bytes()[:m['audit']['handoff_prefix_bytes']]
        assert hashlib.sha256(prefix).hexdigest()==m['audit']['handoff_prefix_sha256']
    return m


def make_sources():
    label=os.environ.get('E23_TASK','preflight')
    sources=STATE/label/'sources';work=STATE/label/'work'
    emit_sources(ROOT,sources)
    for name in ['main.py']:
        atomic(sources/name,(ROOT/name).read_bytes())
    if not (sources/'CLIP').exists():(sources/'CLIP').symlink_to(ROOT/'CLIP',target_is_directory=True)
    for name in ['attn','graph']:(work/name).mkdir(parents=True,exist_ok=True)
    for name in ['network.py','layers.py','utils.py','main.py']:assert sha(sources/name)==sha(ROOT/name)
    assert (sources/'model.py').read_text()==patch_model23((ROOT/'model.py').read_text())
    os.chdir(work)
    return sources


def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1,('GPU visibility',os.environ.get('CUDA_VISIBLE_DEVICES'))
        sources=make_sources();sys.path.insert(0,str(sources))
        import network, model, e23_runtime
        assert Path(network.__file__).resolve()==(sources/'network.py').resolve()
        assert Path(model.__file__).resolve()==(sources/'model.py').resolve()
        x,y,folds,au=data()
        _CTX=dict(torch=torch,network=network,model=model,runtime=e23_runtime,x=x,y=y,folds=folds,audit=au,sources=sources,
            evidence={i:load_fold(EXP/'evidence.npz',i) for i in range(1,11)})
    return _CTX


def agent_for(config,seed,fold,original=False):
    c=context()
    c['runtime'].configure(config=config,fold=fold,evidence=c['evidence'][fold],pheno_indices=c['audit']['pheno_age_sex_indices'])
    cls=c['model'].EvalHelper
    if original:
        module=types.ModuleType('e23_original_network');sys.modules[module.__name__]=module
        exec(compile((ROOT/'network.py').read_text(),'original_network.py','exec'),module.__dict__)
        if c.get('oracle_warm',False):module._prompt_text_features=c['network']._prompt_text_features
        original_model=types.ModuleType('e23_original_model')
        source=(ROOT/'model.py').read_text().replace('from network import *','from e23_original_network import *')
        exec(compile(source,'original_model.py','exec'),original_model.__dict__)
        cls=original_model.EvalHelper
        c['original_network']=module
    tr,te=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a=cls([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    return a


def branch(a):return getattr(a.ModalFusion,'bridge23',None)


def forward(a):
    import torch
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat)
        hidden,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        if branch(a) is not None:hidden=branch(a)(hidden)
        out,_,_=a.ModalFusion(hidden)
        assert torch.isfinite(out).all()
    return out.clone()


def metrics(out,indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score
    target=context()['y'][indices]
    probability=out.exp().detach().cpu().numpy()[indices];pred=probability.argmax(1)
    cm=confusion_matrix(target,pred,labels=[0,1]);tp,fn,fp,tn=map(int,cm.ravel())
    sen,spe=tp/(tp+fn),tn/(tn+fp)
    return dict(acc=float((pred==target).mean()),ba=(sen+spe)/2,
        hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[pred])),
        probability_auc=float(roc_auc_score(target==0,probability[:,0])),sensitivity_asd=sen,specificity_td=spe,
        f1_asd=2*tp/max(2*tp+fp+fn,1),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)


def epoch_step(a):
    with contextlib.redirect_stdout(io.StringIO()):
        a.run_epoch('simple-2')
        tr_acc,val_acc=a.print_trn_acc('simple-2')
    return float(tr_acc),float(val_acc)


def gradients(a):
    b=branch(a);out={k:0. for k in ['alpha','pheno_encoder','corr_encoder','decoder']}
    if b is not None:
        for name in out:
            params=[b.alpha] if name=='alpha' else list(getattr(b,name).parameters())
            for p in params:
                if p.grad is not None:
                    assert bool(p.grad.isfinite().all()),('nonfinite gradient',name)
                    out[name]+=float(p.grad.detach().double().square().sum())
        out={k:math.sqrt(v) for k,v in out.items()}
    return out


def parameter_norms(a):
    b=branch(a)
    if b is None:return {}
    return {name:math.sqrt(sum(float(p.detach().double().square().sum()) for p in ([b.alpha] if name=='alpha' else getattr(b,name).parameters()))) for name in ['alpha','pheno_encoder','corr_encoder','decoder']}


def assert_historical_fold(row):
    if row['config']!='G0':return
    for number,config in [(13,'F0'),(14,'E00'),(15,'A0'),(16,'B0'),(17,'D0'),(18,'V0')]:
        folder=next((ROOT/'experiment').glob(f'experiment{number}_*'))
        ref=next(r for r in read_csv(folder/'results/fold_metrics.csv') if r['config']==config and int(r['seed'])==int(row['seed']) and int(r['fold'])==int(row['fold']))
        for key in METRICS:
            if key=='f1_asd':continue
            rk='balanced_accuracy' if number==14 and key=='ba' else key
            assert abs(float(row[key])-float(ref[rk]))<1e-12,('P0 parity metric',number,row['seed'],row['fold'],key,row[key],ref[rk])
        for key in ['best_epoch','epochs_executed']:assert int(row[key])==int(ref[key]),('P0 epoch parity',number,key)
        ik='initial_trunk_legacy13_sha256' if number==13 else 'initial_trunk_sha256'
        assert row[ik]==ref['initial_trunk_sha256'],('P0 init parity',number,row['seed'],row['fold'])
        rk='fold_entry_rng_sha256' if number==14 else 'entry_rng_sha256'
        if rk in ref:assert row['entry_rng_sha256']==ref[rk],('P0 RNG parity',number,row['seed'],row['fold'])


def counterfactual(a,seed,fold,real):
    import numpy as np
    import torch
    b=branch(a);assert b is not None and b.condition=='REAL'
    before=model_state(a);rr=rng();rh=rng_hash(rr)
    assert torch.equal(real,forward(a)), 'REAL differs from normal evaluation'
    refp=real.exp()[a.tst_idx,0].detach().cpu().numpy();refpred=real[a.tst_idx].argmax(1).cpu().numpy()
    reference=metrics(real,a.tst_idx);rows=[]
    try:
        for condition in INTERVENTIONS:
            context()['runtime'].intervention(a,condition)
            out=forward(a);p=out.exp()[a.tst_idx,0].cpu().numpy();pred=out[a.tst_idx].argmax(1).cpu().numpy()
            mm=metrics(out,a.tst_idx)
            rows.append(dict(config='G4',seed=seed,fold=fold,condition=condition,**mm,
                prediction_flip_fraction=float((pred!=refpred).mean()),
                mean_abs_asd_probability_change=float(abs(p-refp).mean()),
                max_abs_asd_probability_change=float(abs(p-refp).max()),
                acc_change_vs_real=mm['acc']-reference['acc'],ba_change_vs_real=mm['ba']-reference['ba'],
                probability_auc_change_vs_real=mm['probability_auc']-reference['probability_auc'],
                checkpoint_fixed=True,retrained=False))
            context()['runtime'].intervention(a,'REAL')
            assert torch.equal(forward(a),real)
    finally:b.condition='REAL'
    assert_equal_tree(before,model_state(a));assert rng_hash(rng())==rh
    return rows


def diagnostics(a,config,seed,fold,real,tracker,initial_norms):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    c=context();ev=c['evidence'][fold];tr,te=a.trn_idx,a.tst_idx
    def correlation(u,v):
        if np.std(u)<1e-12 or np.std(v)<1e-12:return 0.
        s=float(spearmanr(u,v).statistic);return s if math.isfinite(s) else 0.
    fr=read_json(RESULTS/'feasibility.json')['folds'][fold-1]
    d=dict(config=config,seed=seed,fold=fold,
        oof_mean_delta=fr['mean_oof_delta'],oof_top20_meta_delta=fr['high20_meta_oof_delta'],
        meta_oof_spearman=fr['meta_oof_spearman'],evidence_sha256=read_json(MANIFEST)['evidence_sha256'],
        gain_gate_train_mean=float(ev['gain_gate'][tr].mean()),gain_gate_train_std=float(ev['gain_gate'][tr].std()),
        gain_gate_test_mean=float(ev['gain_gate'][te].mean()),gain_gate_test_std=float(ev['gain_gate'][te].std()),
        confidence_gain_spearman_train=correlation(ev['gain_gate'][tr],1-2*abs(ev['pfc'][tr]-.5)),
        parameter_count_added=0 if config=='G0' else 30081,
        **{'max_gradient_'+k:v for k,v in tracker.items()})
    d.update({'initial_norm_'+k:v for k,v in initial_norms.items()})
    d.update({'selected_norm_'+k:v for k,v in parameter_norms(a).items()})
    b=branch(a)
    if b is not None:
        base_capture={}
        hook=a.MessagePassing0.register_forward_hook(lambda mod,inp,out:base_capture.update(base=out[0].detach().clone()))
        try:assert torch.equal(real,forward(a))
        finally:hook.remove()
        base=base_capture['base'];bb=b.last
        d.update(gate_train_mean=float(b.chosen_gate[tr].mean()),gate_train_std=float(b.chosen_gate[tr].std(unbiased=False)),
            gate_test_mean=float(b.chosen_gate[te].mean()),gate_test_std=float(b.chosen_gate[te].std(unbiased=False)),
            af_ap_top12_intersection=float(b.base_graph_intersection.mean()),
            relation_kernel_mean_abs=float(b.relation_difference.abs().mean()),
            relation_signal_l2_mean=float(bb['bridge_signal'].norm(dim=-1).mean()),
            residual_l2_mean=float(bb['delta_norm'].mean()),residual_relative_mean=float(bb['relative_delta'].mean()),
            residual_relative_max=float(bb['relative_delta'].max()),alpha=float(b.alpha),tanh_alpha=float(b.alpha.tanh()),
            residual_nonzero_fraction=float((bb['delta_norm']>1e-9).float().mean()),
            first64_unchanged=True,selected_branch_weight_sha256=weight_hash(a,True))
        assert d['residual_relative_max']<=.100001
        enhanced=b(base)
        assert torch.equal(enhanced[:,:64],base[:,:64])
        d['base_enhanced_cosine']=float(torch.nn.functional.cosine_similarity(base,enhanced,dim=-1).mean())
        for condition in ['GATE_OFF','RELATION_OFF']:
            b.condition=condition
            off=forward(a);mm=metrics(off,te)
            d[condition.lower()+'_mean_probability_change']=float((off.exp()[te,0]-real.exp()[te,0]).abs().mean())
            d[condition.lower()+'_flip_fraction']=float((off[te].argmax(1)!=real[te].argmax(1)).float().mean())
            d.update({condition.lower()+'_'+k:mm[k] for k in METRICS})
            b.condition='REAL'
        assert torch.equal(forward(a),real)
    else:
        d.update(gate_train_mean=0.,gate_train_std=0.,gate_test_mean=0.,gate_test_std=0.,
            af_ap_top12_intersection=0.,relation_kernel_mean_abs=0.,relation_signal_l2_mean=0.,residual_l2_mean=0.,
            residual_relative_mean=0.,residual_relative_max=0.,alpha=0.,tanh_alpha=0.,residual_nonzero_fraction=0.,
            first64_unchanged=True,base_enhanced_cosine=1.)
    # Site labels are used here only AFTER checkpoint selection, never for gating.
    p=real[te].argmax(1).cpu().numpy();yy=c['y'][te];sites={}
    for col in c['audit']['site_indices']:
        high=c['x'][:,col].max();mask=c['x'][te,col]==high
        if mask.any():
            sites[c['audit']['modality_columns']['PHENO'][col]]=dict(n=int(mask.sum()),
                acc=float((p[mask]==yy[mask]).mean()),asd_fn=int(((p[mask]==1)&(yy[mask]==0)).sum()),
                gate_mean=float(b.chosen_gate[te][torch.as_tensor(mask,device=a.dev)].mean()) if b else 0.)
    d['posthoc_site_strata']=json.dumps(sites,ensure_ascii=False,separators=(',',':'))
    assert all(math.isfinite(v) for v in d.values() if isinstance(v,float))
    return d


def pack_epoch(a,**values):
    return dict(signature=signature(),phase='EPOCH',model=model_state(a),
        mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),**values)


def restore_epoch(a,saved):
    assert saved['signature']==signature()
    load_state(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
    restore_rng(saved['rng'])




def evidence_label_audit():
    """Rebuild in the exact CPU preparation environment, before any GPU context.

    Runtime threadpool changes after GPU initialization did not reproduce the
    original CPU solver reductions. Isolation proves both cache parity and
    label noninterference without relaxing bitwise equality or changing cache.
    """
    program=r'''import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,sys.argv[2])
from e23_prepare import load_fold
from e23_evidence import build_fold,readonly_evidence_digest
root,exp=Path(sys.argv[1]),Path(sys.argv[2])
table=pd.read_csv(root/'data/ABIDE/processed_standard_data.csv')
x=table.iloc[:,:-1].to_numpy(dtype=np.float64);y=table.iloc[:,-1].to_numpy(dtype=np.int64)-1
mapping=np.load(root/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
ph=[i for i,n in enumerate(mapping['PHENO']) if str(n).startswith(('AGE_AT_SCAN_','SEX_'))]
checks={}
for fold in range(1,11):
    ev=load_fold(exp/'evidence.npz',fold);tr,te=ev['outer_train'],ev['outer_test']
    poison=y.copy();poison[te]=1-poison[te]
    alt=build_fold(x,poison,tr,te,ph,seed=23000+fold);alt.pop('report')
    assert readonly_evidence_digest(alt)==readonly_evidence_digest(ev),('flip',fold)
    missing=y.astype(float);missing[te]=np.nan
    alt=build_fold(x,missing,tr,te,ph,seed=23000+fold);alt.pop('report')
    assert readonly_evidence_digest(alt)==readonly_evidence_digest(ev),('missing',fold)
    checks[str(fold)]=dict(test_labels_flipped_cache_digest_exact=True,test_labels_missing_cache_digest_exact=True)
tr,te=load_fold(exp/'evidence.npz',1)['outer_train'],load_fold(exp/'evidence.npz',1)['outer_test']
changed=y.copy();changed[tr]=1-changed[tr]
alt=build_fold(x,changed,tr,te,ph,seed=23001);alt.pop('report')
assert readonly_evidence_digest(alt)!=readonly_evidence_digest(load_fold(exp/'evidence.npz',1))
print(json.dumps(dict(folds=checks,train_label_changes_evidence=True,exact_stage0_cpu_environment=True)))
'''
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    out=subprocess.check_output([sys.executable,'-c',program,str(ROOT),str(EXP)],env=env,cwd=EXP,text=True)
    return json.loads(out)


def preflight():
    import numpy as np
    import torch
    m=prepare()
    if read_json(RESULTS/'feasibility.json')['decision']['early_stop']:
        log('PRE_REGISTERED_STOP_NO_GPU_TRAINING');report();return
    if m.get('preflight',{}):
        if m['preflight'].get('status')=='PASS':log('PREFLIGHT_ALREADY_PASS');return
    integrity(full=True);c=context()
    checks=dict(status='RUNNING',started_utc=utc(),baseline_oracles=[],initializations={},checks={})
    try:
        # Real ROOT model oracle, every seed and fold. Cold cache only in fold1.
        previous=next((v for v in reversed(m.get('preflight_attempts',[])) if v.get('checks',{}).get('A_B_original_full_data_init_prediction_and_two_adam_exact_30')),None)
        if previous:
            checks['baseline_oracles']=previous['baseline_oracles'];checks['initializations']=previous['initializations']
            checks['reused_exact_A_B_tests']=dict(started_utc=previous['started_utc'],reason='Only CPU evidence-audit process and controller error handling changed; reference six files, copied baseline patch and algorithms are unchanged.')
        for seed in ([] if previous else range(3)):
            for fold in range(1,11):
                cold=fold==1;c['oracle_warm']=not cold
                if cold:c['network']._prompt_text_features=None
                else:c['network'].get_prompt_text_features()
                seed_rng(2300+10*seed+fold);entry=rng()
                native=agent_for('G0',seed,fold,original=True)
                native_init=weight_hash(native);nr=rng();nout=forward(native)
                nstate=model_state(native)
                epoch_step(native)
                updated=model_state(native);updated_rng=rng_hash(rng());updated_out=forward(native)
                nmp=native.optimizer_MP0.state_dict();nmf=native.optimizer_MF.state_dict()
                restore_rng(entry)
                if cold:c['network']._prompt_text_features=None
                a=agent_for('G0',seed,fold)
                assert weight_hash(a)==native_init and rng_hash(rng())==rng_hash(nr)
                assert_equal_tree(model_state(a),nstate);assert torch.equal(forward(a),nout)
                optimizer_audit(a,'G0');epoch_step(a)
                assert_equal_tree(model_state(a),updated)
                assert_equal_tree(a.optimizer_MP0.state_dict(),nmp);assert_equal_tree(a.optimizer_MF.state_dict(),nmf)
                assert updated_rng==rng_hash(rng()) and torch.equal(forward(a),updated_out)
                checks['baseline_oracles'].append(dict(seed=seed,fold=fold,cold_clip_cache=cold,
                    initialization=True,forward=True,first_epoch_prediction=True,two_adam_updates=True,cpu_cuda_rng=True))
                if fold==1:
                    bh=None
                    for cfg in CONFIGS[1:]:
                        restore_rng(entry);c['network']._prompt_text_features=None
                        z=agent_for(cfg,seed,fold);b=branch(z)
                        assert weight_hash(z)==native_init and rng_hash(rng())==rng_hash(nr)
                        assert torch.equal(forward(z),nout)
                        count=optimizer_audit(z,cfg)
                        assert float(b.alpha)==0. and b.last['correction'].count_nonzero()==0
                        if bh is None:bh=weight_hash(z,True)
                        assert weight_hash(z,True)==bh
                        # One _print_acc equals one bridge call; print_trn_acc two calls.
                        seen=[];h=b.register_forward_hook(lambda mod,inp,out:seen.append(1))
                        with contextlib.redirect_stdout(io.StringIO()):z._print_acc(z.val_idx,'simple-2')
                        h.remove();assert len(seen)==1
                        checks['initializations'][f'{cfg}_seed{seed}']=dict(total=count,added=count-1462957,
                            trunk_sha256=native_init,branch_sha256=bh,rng_exact=True,forward_exact=True,eval_bridge_calls=1)
                        del z
                del native,a,nstate,updated,nmp,nmf
                if fold in [1,10]:log(f'PREFLIGHT baseline exact seed={seed} fold={fold}')
        checks['checks']['A_B_original_full_data_init_prediction_and_two_adam_exact_30']=True
        # Genuine held-out-label poisoning in isolated, exact CPU builder environment.
        checks['label_audit']=evidence_label_audit()
        for fold,(tr,te) in enumerate(c['folds'],1):
            ev=c['evidence'][fold]
            assert np.array_equal(ev['outer_train'],tr) and np.array_equal(ev['outer_test'],te)
            perm=ev['pheno_permutation']
            for subset in [tr,te]:
                assert np.all(perm[subset]!=subset) and set(perm[subset])==set(subset)
                assert np.array_equal(np.sort(ev['shuffled_gate'][subset]),np.sort(ev['gain_gate'][subset]))
            assert abs(ev['constant_gate'][tr].mean()-ev['gain_gate'][tr].mean())<1e-6
            assert abs(ev['confidence_gate'][tr].mean()-ev['gain_gate'][tr].mean())<1e-6
        tr,te=c['folds'][0]
        checks['checks']['C_all_ten_heldout_label_flip_digest_exact_and_missing_labels_exact']=True
        checks['checks']['E_partition_dims_derangements_histograms_and_train_budgets']=True
        # Structural and gradient checks across every group on actual GFR features.
        checks['component_gradients']={}
        for cfg in CONFIGS[1:]:
            seed_rng(0);a=agent_for(cfg,0,1);b=branch(a)
            rawparts=a.dataLayer(a.feat)[1]
            with torch.no_grad():base=a.MessagePassing0(rawparts,4,a.targ,a.trn_idx)[0]
            assert torch.equal(b(base),base)
            assert b.train_indices.numel()==len(tr)
            selfkeys=torch.arange(871,device=a.dev)[:,None]==b.train_indices[None,:]
            assert b.af[selfkeys].count_nonzero()==0 and b.ap[selfkeys].count_nonzero()==0
            for adj in [b.af,b.ap]:assert torch.allclose(adj.sum(1),torch.ones(871,device=a.dev),atol=1e-5)
            assert not b.af.requires_grad and not b.ap.requires_grad
            # Formula bounds and no alternate feature-only decoder path.
            with torch.no_grad():b.alpha.fill_(1.)
            enhanced=b(base)
            assert torch.equal(base[:,:64],enhanced[:,:64])
            assert float(((enhanced[:,64:]-base[:,64:]).norm(dim=-1)/base[:,64:].norm(dim=-1).clamp_min(1e-8)).max())<=.100001
            for condition in ['GATE_OFF','RELATION_OFF']:
                b.condition=condition;assert torch.equal(b(base),base)
            b.condition='REAL'
            saved=b.relation_difference.clone();b.relation_difference.zero_()
            if cfg!='G6':assert torch.equal(b(base),base)
            else:assert torch.equal(b(base),base) # G6 norm benchmark vanishes too.
            b.relation_difference.copy_(saved)
            with torch.no_grad():b.alpha.zero_()
            observations=[]
            for step in range(2):
                epoch_step(a);observations.append(gradients(a))
            assert observations[0]['alpha']>0.
            for name in ['pheno_encoder','corr_encoder','decoder']:assert observations[1][name]>0.,(cfg,name,observations)
            assert float(b.last['relative_delta'].max())<=.100001
            checks['component_gradients'][cfg]=observations
            if cfg=='G4':
                out=forward(a);cf=counterfactual(a,0,1,out)
                checks['counterfactual_probability_probe']={r['condition']:r['mean_abs_asd_probability_change'] for r in cf}
                diag=diagnostics(a,cfg,0,1,out,observations[-1],parameter_norms(a))
                assert all(math.isfinite(v) for v in diag.values() if isinstance(v,float))
            if cfg=='G7':assert torch.equal(b.patient_pheno,rawparts[0][:,c['audit']['pheno_age_sex_indices']])
            del a
        checks['checks']['D_exact_off_identities_train_keys_residual_bound_first64_unchanged']=True
        checks['checks']['F_zero_init_two_step_gradients_and_CF_weights_RNG_probabilities_exact']=True
        # Epoch interruption and fold boundary including fresh EvalHelper construction.
        checks['recovery']={}
        for cfg in ['G0','G4','G7']:
            seed_rng(1);a=agent_for(cfg,1,1);history=[];best=0.;wait=0;bm=None;be=-1
            for epoch in range(2):
                ta,va=epoch_step(a);history.append([epoch,ta,va])
                if va>best:best,bm,be,wait=va,model_state(a),epoch,0
                else:wait+=1
            path=STATE/'preflight'/'epoch-recovery.pt'
            payload=pack_epoch(a,fold=1,next_epoch=2,best_acc=best,best_model=bm,best_epoch=be,wait=wait,
                validation_history=history,result=dict(metrics=[],diagnostics=[],counterfactual=[]),committed_row_ids=[])
            save_checkpoint(path,payload)
            ta,va=epoch_step(a);expected=model_state(a);er=rng();eo=forward(a)
            emp=a.optimizer_MP0.state_dict();emf=a.optimizer_MF.state_dict()
            del a
            c['network'].get_prompt_text_features() # Loaded before RNG restoration.
            z=agent_for(cfg,1,1);saved=checkpoint(path);restore_epoch(z,saved)
            assert saved['best_epoch']==be and saved['wait']==wait and saved['validation_history']==history
            actual=epoch_step(z);assert actual==(ta,va)
            assert_equal_tree(model_state(z),expected);assert_equal_tree(z.optimizer_MP0.state_dict(),emp);assert_equal_tree(z.optimizer_MF.state_dict(),emf)
            assert rng_hash(rng())==rng_hash(er) and torch.equal(forward(z),eo)
            # Between-fold state restores random stream BEFORE the next initialization.
            continuation=rng();between=STATE/'preflight'/'fold-recovery.pt'
            save_checkpoint(between,dict(signature=signature(),phase='BETWEEN_FOLDS',fold=1,rng=continuation,
                result=dict(metrics=[dict(config=cfg,seed=1,fold=1)],diagnostics=[{}],counterfactual=[]),committed_row_ids=[[cfg,1,1]]))
            na=agent_for(cfg,1,2);ns=model_state(na);nrr=rng();npred=forward(na);nap=epoch_step(na);ns2=model_state(na);nr2=rng()
            del na
            ss=checkpoint(between);c['network'].get_prompt_text_features();restore_rng(ss['rng'])
            nb=agent_for(cfg,1,2);assert_equal_tree(model_state(nb),ns)
            assert rng_hash(rng())==rng_hash(nrr) and torch.equal(forward(nb),npred)
            assert epoch_step(nb)==nap;assert_equal_tree(model_state(nb),ns2);assert rng_hash(rng())==rng_hash(nr2)
            path.unlink();between.unlink();del z,nb
            checks['recovery'][cfg]=dict(checksum=True,epoch_next_step_exact=True,model_adam_python_numpy_cpu_cuda_rng=True,
                best_wait_validation_history_row_ids=True,fold_boundary_new_agent_exact=True)
        checks['checks']['G_epoch_and_fold_recovery_exact']=True
        integrity(full=True)
        checks.update(status='PASS',completed_utc=utc(),root_history_unchanged=True,baseline_parameters=1462957,added_parameters=30081,
            algorithm_changes=[],note='No formal checkpoint selected or score used for design changes; synthetic/first two-step checks only.')
        m=read_json(MANIFEST);m['preflight']=checks;m['run_status']='PREFLIGHT_PASS';save_json(MANIFEST,m)
        append_handoff('PREFLIGHT');log('PREFLIGHT_PASS all A-G')
    except Exception:
        checks.update(status='FAIL',completed_utc=utc(),traceback=traceback.format_exc())
        m=read_json(MANIFEST);m['preflight']=checks;m['run_status']='BLOCKED_PREFLIGHT';save_json(MANIFEST,m)
        append_handoff('PREFLIGHT_BLOCKED');raise


def worker(config,seed):
    import torch
    m=integrity();assert m['preflight']['status']=='PASS'
    task=f'{config}_seed{seed}';gpu=int(os.environ['E23_PHYSICAL_GPU'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu) and m['tasks'][task]['physical_gpu']==gpu
    handle=acquire_lock(STATE/(task+'.lock'));result_path=STATE/(task+'.json');checkpoint_path=STATE/(task+'.pt')
    sig=signature()
    if result_path.exists():
        old=read_json(result_path)
        if old['status']=='COMPLETE':
            assert old['signature']==sig and len(old['metrics'])==10
            print('SKIP_COMPLETE '+task,flush=True);return
    c=context();saved=checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    if saved:
        assert saved['signature']==sig and saved['result']['task']==task
        result=saved['result'];c['network'].get_prompt_text_features();restore_rng(saved['rng'])
        start_fold=int(saved['fold'])+(1 if saved['phase']=='BETWEEN_FOLDS' else 0)
        if saved['phase']=='BETWEEN_FOLDS':save_json(result_path,result)
        print(f'RESUME {task} {saved["phase"]} fold={start_fold}',flush=True)
    else:
        seed_rng(seed);start_fold=1
        result=dict(signature=sig,task=task,config=config,seed=seed,physical_gpu=gpu,status='RUNNING',
            metrics=[],diagnostics=[],counterfactual=[],started_utc=utc(),seconds=0.)
    print(f'WORKER {task} GPU{gpu} logical=cuda:0 {torch.cuda.get_device_name(0)}',flush=True)
    for fold in range(start_fold,11):
        entry=rng();a=agent_for(config,seed,fold);optimizer_audit(a,config)
        initial,legacy13,branch_initial=weight_hash(a),weight_hash(a,legacy13=True),weight_hash(a,True)
        initial_norms=parameter_norms(a);tracker={k:0. for k in gradients(a)}
        best_acc,best_epoch,best_model,best_optimizers,wait,start,prior_seconds=0.,-1,None,None,0,0,0.
        validation_history=[]
        if saved and saved['phase']=='EPOCH' and int(saved['fold'])==fold:
            restore_epoch(a,saved)
            best_acc,best_epoch,best_model,best_optimizers,wait,start=(saved[k] for k in ['best_acc','best_epoch','best_model','best_optimizers','wait','next_epoch'])
            tracker,entry,initial,legacy13,branch_initial=(saved[k] for k in ['tracker','entry_rng','initial','legacy13','branch_initial'])
            initial_norms,prior_seconds,validation_history=saved['initial_norms'],saved['fold_seconds'],saved['validation_history']
        saved=None;started=time.monotonic();last=start-1
        print(f'FOLD_START {fold} initial={initial} entry_rng={rng_hash(entry)}',flush=True)
        for epoch in (range(start,450) if wait<=200 else []):
            ta,validation=epoch_step(a)
            validation_history.append([epoch,ta,validation])
            if epoch in (0,1) or (epoch+1)%25==0:
                for key,value in gradients(a).items():tracker[key]=max(tracker[key],value)
            if validation>best_acc:
                best_acc,best_epoch,best_model,wait=validation,epoch,model_state(a),0
                # Best model selection needs weights; both current optimizers are always checkpointed.
                # Keep best optimizer states serialized to isolate mutable Adam tensors.
                buf=io.BytesIO();torch.save([a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()],buf)
                best_optimizers=buf.getvalue()
            else:wait+=1
            last=epoch;stop=wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                if branch(a):assert float(branch(a).last['relative_delta'].max())<=.100001
                save_checkpoint(checkpoint_path,pack_epoch(a,fold=fold,next_epoch=epoch+1,
                    best_acc=best_acc,best_epoch=best_epoch,best_model=best_model,best_optimizers=best_optimizers,
                    wait=wait,tracker=tracker,entry_rng=entry,initial=initial,legacy13=legacy13,branch_initial=branch_initial,
                    initial_norms=initial_norms,validation_history=validation_history,
                    committed_row_ids=[[config,seed,r['fold']] for r in result['metrics']],
                    fold_seconds=prior_seconds+time.monotonic()-started,result=result))
            if stop:break
        assert best_model is not None
        continuation=rng();load_state(a,best_model);out=forward(a);values=metrics(out,a.tst_idx)
        assert abs(values['acc']-best_acc)<1e-12 and abs(values['ba']-values['hard_auc'])<1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=gpu,**values,best_epoch=best_epoch,
            epochs_executed=last+1,selected_val_acc=best_acc,train_n=len(a.trn_idx),test_n=len(a.tst_idx),val_equals_test=True,
            initial_trunk_sha256=initial,initial_trunk_legacy13_sha256=legacy13,
            initial_branch_sha256=branch_initial if config!='G0' else '',entry_rng_sha256=rng_hash(entry),
            additional_parameters=m['additional_parameters'][config],signature=sig)
        assert_historical_fold(row)
        diagnostic=diagnostics(a,config,seed,fold,out,tracker,initial_norms)
        diagnostic.update(best_epoch=best_epoch,epochs_executed=last+1,signature=sig)
        cf=counterfactual(a,seed,fold,out) if config=='G4' else []
        for rr in cf:rr['signature']=sig
        row['seconds']=prior_seconds+time.monotonic()-started
        result['metrics'].append(row);result['diagnostics'].append(diagnostic);result['counterfactual'].extend(cf)
        restore_rng(continuation)
        result.update(seconds=sum(r['seconds'] for r in result['metrics']),status='COMPLETE' if fold==10 else 'RUNNING',updated_utc=utc())
        save_checkpoint(checkpoint_path,dict(signature=sig,phase='BETWEEN_FOLDS',fold=fold,rng=continuation,result=result,
            committed_row_ids=[[config,seed,r['fold']] for r in result['metrics']]))
        save_json(result_path,result)
        print('FOLD_COMPLETE '+json.dumps(row,ensure_ascii=False),flush=True)
        del a,best_model,best_optimizers
        if fold==10:checkpoint_path.unlink(missing_ok=True)
    print('TASK_COMPLETE '+task,flush=True);handle.close()


def live_workers():
    import psutil
    result={}
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if p.status()=='zombie':continue
        if str(EXP/'run.py') in args and '--worker' in args:
            i=args.index('--worker');result[f'{args[i+1]}_seed{args[i+2]}']=p.info['pid']
    return result


def task_results():
    result=[];sig=signature()
    for cfg,s,g in TASKS:
        path=STATE/f'{cfg}_seed{s}.json'
        if not path.exists():continue
        r=read_json(path)
        assert r['signature']==sig and r['physical_gpu']==g
        folds=[v['fold'] for v in r['metrics']]
        assert folds==list(range(1,len(folds)+1)) and len(r['diagnostics'])==len(folds)
        assert len(r['counterfactual'])==(4*len(folds) if cfg=='G4' else 0)
        if r['status']=='COMPLETE':assert len(folds)==10
        result.append(r)
    return result


def refresh():
    m=read_json(MANIFEST)
    if m['run_status']=='COMPLETE' and not STATE.exists():validate_complete();return m
    completed=task_results();rows=[r for t in completed for r in t['metrics']]
    ds=[r for t in completed for r in t['diagnostics']];cf=[r for t in completed for r in t['counterfactual']]
    assert len(set((r['config'],r['seed'],r['fold']) for r in rows))==len(rows)
    for t in completed:
        m['tasks'][t['task']].update(status=t['status'],folds=len(t['metrics']),seconds=t['seconds'],updated_utc=t['updated_utc'],
            metrics_canonical_sha256=canonical(t['metrics']),diagnostics_canonical_sha256=canonical(t['diagnostics']),counterfactual_canonical_sha256=canonical(t['counterfactual']))
    m.update(completed_folds=len(rows),completed_tasks=sum(t['status']=='COMPLETE' for t in completed),
        diagnostic_rows=len(ds),counterfactual_rows=len(cf),updated_utc=utc())
    for name,rr in [('fold_metrics',rows),('diagnostics',ds),('counterfactual',cf)]:save_csv(RESULTS/(name+'.csv'),rr)
    save_json(MANIFEST,m);return m


def worker_env(gpu,task='preflight'):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),E23_PHYSICAL_GPU=str(gpu),E23_TASK=task,
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')


def resources():
    import psutil
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    cards=[]
    for line in raw.strip().splitlines():
        i,name,total,used,util=[v.strip() for v in line.split(',')]
        cards.append(dict(index=int(i),name=name,total_mib=int(total.split()[0]),used_mib=int(used.split()[0]),utilization=int(util.split()[0])))
    rss=0
    for pid in live_workers().values():
        try:rss+=psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:pass
    mem=psutil.virtual_memory()
    return dict(utc=utc(),gpu=cards,worker_rss_sum_gib=rss/2**30,host_available_gib=mem.available/2**30,
        system_cpu_percent=psutil.cpu_percent(),active_workers=len(live_workers()),gpu_memory_scope='entire device, including unrelated jobs')


def schedule(workers_per_gpu=4):
    assert 1<=workers_per_gpu<=4
    m=prepare()
    if m['run_status']=='COMPLETE':validate_complete();log('ALREADY_COMPLETE 24 tasks / 240 folds');return
    if read_json(RESULTS/'feasibility.json')['decision']['early_stop']:
        report();append_handoff('NEGATIVE_FEASIBILITY');return
    handle=acquire_lock(STATE/'scheduler.lock')
    if not m.get('preflight') or m['preflight']['status']!='PASS':
        with open(LOGS/'scheduler.log','a') as out:
            subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],cwd=EXP,env=worker_env(0),stdout=out,stderr=subprocess.STDOUT,check=True)
    m=read_json(MANIFEST);assert m['preflight']['status']=='PASS'
    m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu)
    m.setdefault('started_unix',time.time());m.setdefault('started_utc',utc());m['launch_resources']=resources();save_json(MANIFEST,m)
    failures=m.get('failures_and_restarts',{});limits={0:workers_per_gpu,1:workers_per_gpu}
    active={t:dict(pid=pid,process=None,gpu=m['tasks'][t]['physical_gpu']) for t,pid in live_workers().items()}
    log('SCHEDULER_START frozen 24 tasks, each GPU 12, max4/GPU')
    last_refresh=0.;last_resource=0.;last_count=-1
    while True:
        for task,info in list(active.items()):
            proc=info['process'];exit_code=proc.poll() if proc else (None if Path('/proc',str(info['pid'])).exists() else 0)
            if exit_code is None:continue
            if 'log' in info:info['log'].close()
            del active[task]
            if exit_code!=0:
                failures[task]=failures.get(task,0)+1
                m=read_json(MANIFEST);m['failures_and_restarts']=failures;save_json(MANIFEST,m)
                tail=(STATE/(task+'.log')).read_text()[-8000:]
                log(f'WORKER_FAILURE {task} exit={exit_code}\n'+tail)
                if failures[task]>=3:
                    m=read_json(MANIFEST);m['run_status']='PARTIAL_WORKER_FAILURE';save_json(MANIFEST,m)
                    raise RuntimeError('Repeated task failure, exact checkpoints retained: '+task)
                # Arithmetic/software assertion failures require repair, not repeated long restarts.
                if 'AssertionError' in tail or 'Traceback' in tail:
                    m=read_json(MANIFEST);m['run_status']='PARTIAL_WORKER_FAILURE';save_json(MANIFEST,m)
                    raise RuntimeError('Worker failed; inspect program error before resume: '+task)
                log('RESTART_FROM_ATOMIC_CHECKPOINT '+task)
        done={t['task'] for t in task_results() if t['status']=='COMPLETE'}
        for gpu in [0,1]:
            for cfg,seed,assigned in TASKS:
                task=f'{cfg}_seed{seed}'
                if assigned!=gpu or task in done or task in active:continue
                if sum(v['gpu']==gpu for v in active.values())>=limits[gpu]:break
                output=open(STATE/(task+'.log'),'a');output.write('\nATTEMPT_START '+utc()+'\n');output.flush()
                proc=subprocess.Popen([sys.executable,str(EXP/'run.py'),'--worker',cfg,str(seed)],cwd=EXP,
                    env=worker_env(gpu,task),stdout=output,stderr=subprocess.STDOUT)
                active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=output)
                log(f'LAUNCH {task} GPU{gpu} pid={proc.pid}')
        if time.monotonic()-last_refresh>20 or not active:
            m=refresh()
            if m['completed_folds']!=last_count:
                log(f'PROGRESS tasks={m["completed_tasks"]}/24 folds={m["completed_folds"]}/240 active={len(active)}')
                last_count=m['completed_folds']
            if time.monotonic()-last_resource>=60:
                sample=resources();m['latest_resources']=sample
                peaks=m.setdefault('runtime_resource_peaks',dict(worker_rss_sum_gib=0.,gpu0_mib=0,gpu1_mib=0))
                peaks['worker_rss_sum_gib']=max(peaks['worker_rss_sum_gib'],sample['worker_rss_sum_gib'])
                for card in sample['gpu']:
                    g=card['index'];peaks[f'gpu{g}_mib']=max(peaks[f'gpu{g}_mib'],card['used_mib'])
                    limit=1 if card['used_mib']>85000 or sample['host_available_gib']<20 or sample['system_cpu_percent']>95 else workers_per_gpu
                    if limits[g]!=limit:m.setdefault('resource_throttling',[]).append(dict(utc=utc(),gpu=g,new_limit=limit));limits[g]=limit
                save_json(MANIFEST,m);log('RESOURCE '+json.dumps(sample));last_resource=time.monotonic()
            last_refresh=time.monotonic()
        if len(done)==24 and not active:break
        time.sleep(2)
    m=refresh();m.update(run_status='TRAINING_COMPLETE',training_completed_utc=utc(),
        training_wall_seconds=time.time()-m['started_unix'],failures_and_restarts=failures);save_json(MANIFEST,m)
    validate_complete();integrity(full=True);assert not live_workers()
    m=read_json(MANIFEST);m.update(run_status='COMPLETE',completed_utc=utc(),temporary_state_cleaned=False,no_training_processes=True,
        results_sha256={name:sha(RESULTS/(name+'.csv')) for name in ['fold_metrics','diagnostics','counterfactual']});save_json(MANIFEST,m)
    report();append_handoff('COMPLETE');report();handle.close();shutil.rmtree(STATE)
    m=read_json(MANIFEST);m.update(temporary_state_cleaned=True,report_sha256=sha(RESULTS/'report.md'));save_json(MANIFEST,m)
    log('COMPLETE 24/24 tasks 240/240 folds 240 diagnostics 120 counterfactual; temporary sources/checkpoints cleaned')


def validate_complete():
    m=integrity(full=True)
    rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    keys={(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    for rr in [rows,ds]:
        actual=[(r['config'],int(r['seed']),int(r['fold'])) for r in rr]
        assert len(actual)==240 and len(set(actual))==240 and set(actual)==keys
        assert all(r['signature']==signature() for r in rr)
    assert len(cf)==120 and {(int(r['seed']),int(r['fold']),r['condition']) for r in cf}=={(s,f,c) for s in range(3) for f in range(1,11) for c in INTERVENTIONS}
    assert all(r['signature']==signature() for r in cf)
    assert all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
    assert all(float(r['residual_relative_max'])<=.100001 for r in ds)
    for r in rows:assert_historical_fold(r)
    assert sum(int(r['physical_gpu'])==0 for r in rows)==sum(int(r['physical_gpu'])==1 for r in rows)==120
    assert all(t['status']=='COMPLETE' and t['folds']==10 for t in m['tasks'].values())
    for name,h in m.get('results_sha256',{}).items():assert sha(RESULTS/(name+'.csv'))==h
    return True


def readme_text():
    return f'''# GC-SRA — Experiment 23

实验在 `{EXP}` 执行；项目根 `{ROOT}` 只读（handoff仅安全追加）。/data 创建失败，按用户指令使用 /tmp。没有执行git commit/push/pull/reset/checkout。

## 冻结设计

G0原P0；G1统一关系桥接；G2同预算常量；G3同预算FC不确定度；G4 OOF增益；G5集合内置乱增益；G6同容量特征差分；G7仅PHENO关系构造错配。共24任务、240折，CUDA_VISIBLE_DEVICES隔离，逻辑cuda:0，两物理GPU交替分配各12任务/120折，每卡至多4 worker，每worker2 CPU线程。

严格复用六份用户参考源文件，SHA见manifest。算法未改写，运行器的原子checkpoint、RNG、data、训练loop/状态恢复由实验18移植并逐项替换其专用分支。只有复制到 `_runtime_sources/<task>/sources/model.py` 的源码有两个bridge前向点与构造后安装；network/layers/utils/main原样。禁止导入实验18教师/对齐分支。临时源码、完整模型和两个Adam状态在完成后清理；永久CSV足以校验任务及重跑。

## 数学与接口

训练折内5fold StandardScaler+LogisticRegression(C=1,lbfgs,1000)比较FC256与FC+PHENO28，delta=CE_F−CE_FP。Ridge(alpha=150)拟合clip(delta,−1.5,1.5)，meta5fold OOF只用于探索性诊断。gate=.02+.96 sigmoid(pred_gain/max(sd(clipped_target),.05))；训练gate为全训练meta拟合值，测试gate用outer-train全量教师概率。QC16为观测质量指标，站点20维不进辅助输入。

AF/AP仅以outer-train为key，train-only标准化、cosine/.3、各top12并集、排除自己。关系差分=(AP−AF)PHENOvalue；新增桥接只更新GFR输出的CORR256，前64维逐值保持。Decoder([difference,context])−Decoder([0,context])消除FC-only/bias旁路；零alpha；.5×.1×tanh(alpha)×gate×tanh(norm(difference))×detach(norm(CORR))×单位方向。新增30,081个注册参数，均在MF Adam新增组lr=.0038,wd=0，原参数wd=.11。

G6使用真实PHENO自身−AF邻域PHENO，训练平均L2匹配到关系信号，AP只提供幅度标尺且共享并集支持；不是无跨模态信息对照。G7只置乱构图PHENO，真实value、gain和主干输入不动；由于支持集为两图top12并集，错配AP也会影响AF在共享支持上的归一化，按参考实现披露此边界。decoder最后bias相减抵消，注册参数相同不代表每个单独参数都有非零梯度。

## 协议与审计

保持原871人、403ASD0/468TD1、320输入、SKFold10shuffle(random_state=10)、seed0/1/2连续十fold、450epoch/wait>200、严格valACC提升、原两Adam、图正则四图平均。原GFR i=1覆盖combined_feat，ALoss计算但不入loss；4×36潜在Token非ROI。P0原八句/CLIP与PLN保持。

历史偏差原样保留：val=test参与选模、CV前全队列标准化/监督RFE、全871人transductive、项目此前22轮探索。因此不是独立无偏/无泄漏泛化、临床或因果收益。meta OOF一阶教师训练池重叠，不是完全独立三层嵌套估计。

先自测/Stage0，四项同时成立才提前判停；否则全部八组训练。预检覆盖原P0每seed/fold初始化/前向/两Adam首步逐值一致，标签污染、预算/置乱、残差零恒等、真实梯度、epoch/fold完整恢复；正式G0逐折核对实验13–18的指标/epoch/初始化/RNG。E9仅历史描述参照。

每25epoch与fold边界原子checkpoint包含当前/最佳模型、两个Adam、RNG、best/wait/history、证据与源码签名、已提交行ID。完成任务校验后跳过，中断恢复不重置连续随机流。根与实验1–22SHA全程保护。每阶段尝试锁定单次O_APPEND写根handoff，保留此前字节；若/data I/O阻止则manifest记录实际失败。

## 复现

```bash
python {EXP}/run.py prepare
python {EXP}/run.py preflight
python {EXP}/run.py all
python {EXP}/run.py resume
python {EXP}/run.py refresh
python {EXP}/run.py report
```

all自动补齐prepare/preflight，已完成校验后跳过。迁移目录时SPROMPTGL_ROOT指定原始数据/根目录；若路径或代码变化，不能续用原签名checkpoint，应以冻结代码在新的完整副本从头重跑。参考六文件未使用接口补丁；若必须修bug，manifest必须列出逐行diff与原因并重新审计。

## 既有工作边界

[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)提供原GFR/CAM/SPGS/RES；[MGDR](https://papers.miccai.org/miccai-2024/514-Paper2494.html)已有共享/私有模态表示；[To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/view/39248)讨论对齐与冗余；[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)已有表型条件化长程通信。本轮只检验OOF条件增益驱动的患者关系路由+FC锚+严格阴性对照组合，不声称门控/图或选择性对齐首创。

实验11有871行VERIFIED的数值身份恢复链；实验21仅因有界搜索未找到subject_IDs.txt判0/871，未复用前链。两者结论不一致，本轮不恢复身份、不引入临床量表监督。
'''


def judgment(rows):
    pp={}
    for left,right in CONTRASTS:
        q={k:paired(rows,left,right,k) for k in METRICS}
        major=all(all(v>1e-12 for v in q[k]['seeds']) for k in ['ba','probability_auc'])
        guard=all(all(v>=-1e-12 for v in q[k]['seeds']) for k in ['acc','sensitivity_asd'])
        pp[f'{left}-{right}']=dict(stable_ba_probability_auc=major,acc_sensitivity_nonworse_every_seed=guard,pass_all=major and guard)
    return dict(contrasts=pp,all_controls_supported=all(v['pass_all'] for v in pp.values()),
        interpretation=('固定内部协议支持进一步独立验证，尚未证明临床/因果或无偏泛化' if all(v['pass_all'] for v in pp.values()) else '本轮不能确认增益门控患者关系对齐的稳定独立创新价值；不继续调整本轮参数'))


def report():
    import numpy as np
    m=read_json(MANIFEST);f=read_json(RESULTS/'feasibility.json');d=f['decision']
    lines=['# 实验二十三 GC-SRA：增益校准选择性患者关系对齐','',
        f'状态：**{m["run_status"]}**。目录：`{EXP}`。生成时间：{utc()}。','',
        f'本地HEAD `{m["audit"]["git_head"]}`；GitHub主线 `{m["audit"]["remote_head"]}`，六核心源码逐字一致。原作者仓库 `{m["audit"]["upstream_head"]}`；差异为已存在的RN50冻结缓存/设备兼容处理，不在本轮改动。',
        '', '## 预注册可行性快筛', '',
        '|Fold|FC OOF BA|FC+PHENO BA|FC AUC|FC+PHENO AUC|FC loss|Joint loss|mean Δ|top20 Δ|meta Spearman|gate SD|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in f['folds']:
        lines.append('|'+str(r['fold'])+'|'+'|'.join(f'{r[k]*(100 if k in ["fc_oof_ba","fp_oof_ba","fc_oof_auc","fp_oof_auc"] else 1):.6f}' for k in ['fc_oof_ba','fp_oof_ba','fc_oof_auc','fp_oof_auc','fc_oof_logloss','joint_oof_logloss','mean_oof_delta','high20_meta_oof_delta','meta_oof_spearman','gain_gate_train_std'])+'|')
    lines += ['',f'四项判停量：mean Δ={d["average_mean_delta"]:.9f}，top20 Δ={d["average_high20_delta"]:.9f}，Spearman={d["average_meta_oof_spearman"]:.9f}，gate SD={d["average_train_gate_std"]:.9f}。判停={d["early_stop"]}。只有四项同时满足才停；本次不能因平均负增益单独停，也不能把top20正增益解释为总体获益。',
        '教师FC+PHENO相对FC的10个重叠outer-train OOF平均BA差='+f'{np.mean([r["fp_oof_ba"]-r["fc_oof_ba"] for r in f["folds"]])*100:+.6f}pp，概率AUC差='+f'{np.mean([r["fp_oof_auc"]-r["fc_oof_auc"] for r in f["folds"]])*100:+.6f}pp。',
        '增益分布分位数、年龄/性别/站点分组门控及delta、置信度相关与跨折分布见唯一feasibility.json。站点仅做来源风险审计，不进辅助教师/门控；QC也可能携带站点相关差异。meta排序相关为负时不能将门控包装为已验证可靠个体增益。',
        '', '## 方法和每组机制', '',readme_text().split('## 冻结设计')[1].split('## 复现')[0],
        '', '## 真实服务器工程验收', '',
        f'预检状态：{(m.get("preflight") or {}).get("status","NOT_RUN")}；完成{m.get("completed_tasks",0)}/24任务、{m.get("completed_folds",0)}/240折、{m.get("diagnostic_rows",0)}/240诊断、{m.get("counterfactual_rows",0)}/120反事实。详细逐值断言及恢复测试在manifest.preflight。',
        '初始化fork_rng后主随机流不变；G1–G7注册顺序与30,081参数一致；eval单次桥接。所有原参数恰好归属原Adam一次，新参数只在MF新增wd0组。按seed连续10折，早停差异导致后续折随机流分叉，不声称组间所有fold同初始化。G0每折历史六指标、最佳/实际epoch、初始化及已有RNG哈希严格核对13F0/14E00/15A0/16B0/17D0/18V0。',
        f'训练墙钟：{m.get("training_wall_seconds","NA")}秒；设备任务分配：'+json.dumps(TASKS)+'。']
    if d['early_stop']:
        lines += ['', '预注册判停成立：当前ABIDE数据没有足够可学习信号支持继续此路线；未启动正式GPU训练，不制造正式空表。建议转向真正互补模态或TADPOLE。']
    elif m.get('completed_folds')==240:
        rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv');ss=summary(rows)
        lines+=['','## 正式主结果','', '单位%，先每seed十折均值，再三seed均值±population SD。Hard-AUC数值等于BA，不作为独立排序证据。', '',
            '|Config|ACC|BA|Hard-AUC|ASD Prob-AUC|ASD SEN|TD SPE|ASD F1|','|---|---|---|---|---|---|---|---|']
        for cfg in CONFIGS:lines.append('|'+cfg+'|'+'|'.join(f'{ss[cfg][k]["mean"]*100:.5f}±{ss[cfg][k]["sd"]*100:.5f}' for k in METRICS)+'|')
        lines+=['','### 每seed十fold均值','', '|Config|Seed|'+'|'.join(METRICS)+'|','|---|---|'+'---|'*len(METRICS)]
        for cfg in CONFIGS:
            for seed in range(3):lines.append(f'|{cfg}|{seed}|'+'|'.join(f'{ss[cfg][k]["seeds"][seed]*100:.6f}' for k in METRICS)+'|')
        lines+=['','## 七项冻结配对差','', '差值为百分点；W/T/L以相同seed、fold比较。重复fold×seed不是30个独立外部样本。','',
            '|Pair|Metric|平均Δ|seed0 Δ|seed1 Δ|seed2 Δ|W/T/L|','|---|---|---|---|---|---|---|']
        for left,right in CONTRASTS:
            for key in METRICS:
                q=paired(rows,left,right,key)
                lines.append(f'|{left}−{right}|{key}|{q["mean"]*100:+.6f}|'+'|'.join(f'{s*100:+.6f}' for s in q['seeds'])+f'|{q["wins"]}/{q["ties"]}/{q["losses"]}|')
        j=judgment(rows)
        lines+=['','保守预注册操作化：BA、概率AUC每seed均严格提高，且每seed ACC及ASD灵敏度不下降。完整差值优先于二值判据。', '```json',json.dumps(j,ensure_ascii=False,indent=2),'```']
        e9=history_e9();lines+=['','## 相对历史E9的描述性差异','', '|Metric|E9|G4|G4−E9 pp|','|---|---|---|---|']
        for key in METRICS:
            if key=='f1_asd':continue
            hv=np.mean([np.mean([float(r[key]) for r in e9 if int(r['seed'])==s]) for s in range(3)])
            gv=ss['G4'][key]['mean'];lines.append(f'|{key}|{hv*100:.6f}|{gv*100:.6f}|{(gv-hv)*100:+.6f}|')
        lines+=['历史E9不重训，不视为同初始化配对。','', '## 机制诊断','',
            '|Config|train gate mean/std|test gate mean/std|top12交集|relation signal L2|相对残差均值/最大|alpha|非零响应比例|',
            '|---|---|---|---|---|---|---|---|']
        for cfg in CONFIGS:
            rr=[r for r in ds if r['config']==cfg]
            lines.append(f'|{cfg}|{mean_field(rr,"gate_train_mean"):.6f}/{mean_field(rr,"gate_train_std"):.6f}|{mean_field(rr,"gate_test_mean"):.6f}/{mean_field(rr,"gate_test_std"):.6f}|{mean_field(rr,"af_ap_top12_intersection"):.6f}|{mean_field(rr,"relation_signal_l2_mean"):.6f}|{mean_field(rr,"residual_relative_mean"):.7f}/{max(float(r["residual_relative_max"]) for r in rr):.7f}|{mean_field(rr,"alpha"):.6f}|{mean_field(rr,"residual_nonzero_fraction"):.6f}|')
        lines+=['残差只在CORR潜在片段，不能将此归因具体脑区或新增影像模态；主干包含的表型站点路径未删除。梯度分量、encoder/decoder参数范数及逐站点事后诊断见diagnostics.csv。若非零残差或反事实敏感而控制不胜，不能当独立补充信息证据。',
            '', '## G4最佳checkpoint反事实','', '|Condition|ACC|BA|Prob-AUC|SEN|SPE|F1|翻转比例|ASD概率平均绝对变化|ACC−REAL pp|',
            '|---|---|---|---|---|---|---|---|---|---|']
        for con in INTERVENTIONS:
            rr=[r for r in cf if r['condition']==con]
            lines.append('|'+con+'|'+'|'.join(f'{mean_field(rr,k)*100:.6f}' for k in ['acc','ba','probability_auc','sensitivity_asd','specificity_td','f1_asd','prediction_flip_fraction'])+f'|{mean_field(rr,"mean_abs_asd_probability_change"):.9f}|{mean_field(rr,"acc_change_vs_real")*100:+.6f}|')
        lines+=['同checkpoint OFF可能分布外，仅证明分支依赖；没有重新训练/重新选模。REAL恢复预测、全部模型参数与RNG逐值不变。GATE_OFF及RELATION_OFF按结构均使新增残差严格为0；其性能不能替代重训练G0。',
            '', '## 科学结论与止损', '', j['interpretation'],
            '不能以最高ACC单独认定成功；统一、预算、置信度、置乱、特征/关系错配对照中任一不稳定，即未建立完整创新假设。保留真实负结果，不添加组、不调超参数。',
            '后续唯一建议：若控制不成立，停止在当前年龄/性别+QC上追求疾病语义对齐，优先取得具有独立互补信息的真实模态或TADPOLE，并预先冻结真正外部验证协议。']
        lines+=['','### 最佳指标（仅描述，不追加择优实验）','']
        for key in ['acc','ba','probability_auc']:
            best=max(CONFIGS,key=lambda cfg:ss[cfg][key]['mean']);lines.append(f'{key}：{best}，{ss[best][key]["mean"]*100:.6f}%。')
    else:
        lines+=['','正式结果尚未完成，不提供虚构指标、不宣称算法成功。']
    lines+=['','## 文献、身份来源与解释边界','',
        '[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)已读本地PDF Fig1/§2.1–2.2/Eq1–7及消融；ABIDE图示CAM86.48→GFR91.73→RES92.08→SPGS92.30，不能当本协议的单独复制指标。源码GFR覆盖及ALoss未反传与论文公式存在差异，本轮保持实际历史实现。',
        '[MGDR](https://papers.miccai.org/miccai-2024/514-Paper2494.html)已有共享/私有表示；[AAAI2026对齐与冗余](https://ojs.aaai.org/index.php/AAAI/article/view/39248)已读官方页和作者预印本；[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)官方摘要已发表表型条件化长程通信，未声称本轮读到了其不可访问全文。不能宣称门控/共享私有/表型图/一般选择性对齐首创。候选只剩OOF条件增益+患者关系路由+FC锚+严格证伪的特定组合，有效性由完整对照决定。',
        '实验11 recover_roi_mapping.py以64维PHENO/QC数值指纹匹配身份，再核验FC/标签，已有871行VERIFIED映射；实验21有界搜索缺失subject_IDs.txt后判NOT_JOINABLE 0/871，没有复用上述链。二者矛盾需保留来源边界；本轮不独立重验身份、不恢复临床量表，不将任何门控解释为症状监督。',
        '历史1–22要点：17 Shared探针弱、Private关闭影响非重训净收益；18 joint-correctness不等于条件增益；19中性优于真实且自由Key可吸收语义方向；20/21固定语义图接近零反事实响应；22η无有效增益、正交残差恶化E9。此次不用疾病Prompt、自由Query、正交残差或额外损失。',
        '局限：测试参与Checkpoint选择，全局预处理/RFE、全图transductive、此前22轮探索；同一队列反复训练。门控预测的是特定教师的损失差，非最终深模型真实个体获益，更非临床因果效果。G6有AP幅值标尺/共享支持，G7错配PHENO关系且共享支持影响AF，解释严格受此边界限制。',
        '', '## 文件完整性与handoff','',
        f'代码/数据/文本权重/证据SHA、参考原文SHA、805份历史保护哈希及完成任务状态见manifest。参考六文件无算法修改。临时目录清理状态最终由manifest记录。根handoff追加阶段：{m.get("handoff_stages",[])}；实际写入错误：{m.get("handoff_write_failures",[])}。',
        '未执行git commit/push/pull/reset/checkout；原始用户工作区的既有改动保留。']
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    m=read_json(MANIFEST);m['report_sha256']=sha(RESULTS/'report.md');save_json(MANIFEST,m)
    log('REPORT_WRITTEN '+str(RESULTS/'report.md'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',nargs='?',choices=['prepare','preflight','all','resume','report','refresh'])
    parser.add_argument('--worker',nargs=2,metavar=('CONFIG','SEED'))
    parser.add_argument('--workers-per-gpu',type=int,default=4)
    args=parser.parse_args()
    if args.worker:return worker(args.worker[0],int(args.worker[1]))
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight()
    elif args.command in ['all','resume']:schedule(args.workers_per_gpu)
    elif args.command=='report':prepare();report()
    elif args.command=='refresh':refresh()
    else:parser.error('specify a command')

if __name__=='__main__':
    try:main()
    except BaseException:
        if any(v in sys.argv for v in ['all','resume']) and '--worker' not in sys.argv:
            # Only terminate this experiment's own workers. Atomic checkpoints
            # are retained, and the next resume continues the exact states.
            import signal
            for pid in live_workers().values():
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if MANIFEST.exists():
                mm=read_json(MANIFEST)
                if mm['run_status'] not in ['COMPLETE','STOPPED_NEGATIVE_FEASIBILITY']:
                    mm['run_status']='PARTIAL_CONTROLLER_INTERRUPTED';mm['controller_error']=traceback.format_exc();save_json(MANIFEST,mm)
                append_handoff('CONTROLLER_INTERRUPTED')
        raise
