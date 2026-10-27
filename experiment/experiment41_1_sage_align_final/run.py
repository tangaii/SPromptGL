#!/usr/bin/env python3
"""Experiment 41.1: SAGE-Align final validation.

The script is self-contained inside this experiment directory.  The source
copy under .runtime/source is the only code imported for the historical GFR /
CAM path; the repository sources are read for hashes and never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

from sage_align import SAGEAlign, build_normative, evidence_for_mapping, zero_evidence_test
import integration

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifest.json"
INPUT_DIMS = [48, 6, 10, 256]
CONFIGS = ("E0_B0A", "E1_HEAD_ONLY", "E2_NUMERIC_ONLY",
           "E3_GLOBAL_REFERENCE", "E4_NO_QC", "E5_WRONG_SEMANTICS",
           "E6_DATA_PRIOR", "E7_FORCED_ALIGNMENT", "E8_FROZEN_HEAD",
           "E9_SAGE_FULL")
GPU_GROUPS = {0: ("E0_B0A", "E2_NUMERIC_ONLY", "E4_NO_QC",
                  "E6_DATA_PRIOR", "E8_FROZEN_HEAD"),
              1: ("E1_HEAD_ONLY", "E3_GLOBAL_REFERENCE",
                  "E5_WRONG_SEMANTICS", "E7_FORCED_ALIGNMENT", "E9_SAGE_FULL")}
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False,
                gpu="-1", nepoch=450, early=200, lr=.0038, reg=.11,
                dropout=.35, nlayer=1, n_hidden=18, n_head=2, n_iter=10,
                nmodal=4, th=.9, GC_mode="weighted-cosine", MP_mode="GCN",
                MF_mode="concat", alpha=.5, theta_smooth=1.,
                theta_degree=.5, theta_sparsity=0., nclass=2,
                mode="simple-2", seed=0)
ADAPT_LR = 3.8e-4
RESIDUAL_CAP = .08
ENERGY_WEIGHT = 1e-3
CHECKPOINT_EVERY = 25
EVIDENCE_DIM = 18
CONCEPTS = 4
DATA_FILE = ROOT / "data/ABIDE/processed_standard_data.csv"
MODAL_FILE = ROOT / "data/ABIDE/modal_feat_dict.npy"
HIST32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
HIST11 = ROOT / "experiment/experiment11_roi_mapping_recovery"
STATE_DIR = RUNTIME / "tasks"
ANCHOR_DIR = RUNTIME / "anchors"
CKPT_DIR = RUNTIME / "checkpoints"
EVIDENCE_DIR = RUNTIME / "evidence"
LOG_DIR = RUNTIME / "logs"
CTX = None

FAMILY_PATTERNS = OrderedDict([
    ("sensorimotor", r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex"),
    ("frontal_association", r"Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex"),
    ("parietal_association", r"Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus"),
    ("temporal_association", r"Temporal Pole|Temporal Gyrus|Temporal Fusiform|Planum Polare|Planum Temporale|Heschl's Gyrus"),
    ("insula", r"Insular Cortex"),
    ("cingulate_medial", r"Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex"),
    ("visual", r"Lateral Occipital Cortex|Intracalcarine Cortex|Cuneal Cortex|Lingual Gyrus|Supracalcarine Cortex|Occipital"),
    ("subcortical", r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$"),
])
CONCEPT_PATTERNS = {
    "P1": r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "P2": r"Insular Cortex|Temporal Pole|Superior Temporal Gyrus, anterior division|Middle Temporal Gyrus, anterior division|Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex",
    "P3": r"Frontal Medial Cortex|Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "P4": r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
}
CONCEPT_IDS = ("P1", "P2", "P3", "P4")
SUBCORTICAL_RE = r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def seed_everything(torch, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_capture(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),
                cuda=[v.cpu() for v in torch.cuda.get_rng_state_all()])


def rng_restore(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def rng_hash(state):
    import pickle
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for value in [state["cpu"]] + list(state["cuda"]):
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    frame = pd.read_csv(DATA_FILE)
    mapping = np.load(MODAL_FILE, allow_pickle=True).item()
    if list(mapping) != ["PHENO", "ANAT", "FUNC", "Correlation"]:
        raise AssertionError("modality order changed")
    if [len(mapping[k]) for k in mapping] != INPUT_DIMS:
        raise AssertionError("input dimensions changed")
    names = [str(v) for k in mapping for v in mapping[k]]
    if frame.shape != (871, 321) or list(frame.columns[:-1]) != names:
        raise AssertionError("preprocessed ABIDE table changed")
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    if not np.isfinite(x).all() or np.bincount(y).tolist() != [403, 468]:
        raise AssertionError("unexpected data or labels")
    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                                 random_state=10).split(x, y))
    return x, y, folds, mapping


def source_digest(module):
    h = hashlib.sha256()
    for prefix, m in (("gfr", module.MessagePassing0), ("pln", module.ModalFusion)):
        for name, p in sorted(m.named_parameters()):
            h.update((prefix + "/" + name).encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def close_helper(agent):
    for name in ("MF_sav", "GCMP_sav"):
        if hasattr(agent, name):
            getattr(agent, name).close()


def context(gpu):
    global CTX
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES mismatch: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(0)
    sys.path.insert(0, str(SOURCE))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    import tempfile
    (RUNTIME / "tmp").mkdir(exist_ok=True)
    tempfile.tempdir = str(RUNTIME / "tmp")
    model = importlib.import_module("model")
    network = importlib.import_module("network")
    utils = importlib.import_module("utils")
    x, y, folds, mapping = read_data()
    CTX = dict(torch=torch, model=model, network=network, utils=utils, x=x,
               y=y, folds=folds, mapping=mapping, device=torch.device("cuda"))
    return CTX


def make_agent(c, fold):
    train, test = c["folds"][fold - 1]
    hp = SimpleNamespace(**PROTOCOL)
    agent = c["model"].EvalHelper(INPUT_DIMS, c["x"], c["y"], hp, train, test)
    close_helper(agent)
    agent.ModalFusion.ucsma41_b0_bypass = True
    if not all(hasattr(agent.ModalFusion, key) for key in ("MLP", "prompt_text_features", "avg_pool")):
        raise AssertionError("PEN construction side effects were removed")
    if not np.array_equal(agent.val_idx, agent.tst_idx):
        raise AssertionError("val=test protocol changed")
    return agent


def e0_step(agent, c):
    torch, utils = c["torch"], c["utils"]
    agent.MessagePassing0.train(); agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad(); agent.optimizer_MF.zero_grad()
    model_num, parts = agent.dataLayer(agent.feat)
    fused, adjs, alignment = agent.MessagePassing0(parts, model_num, agent.targ,
                                                   agent.trn_idx)
    graph = sum(utils.GraphConstructLoss(parts[i], adjs[i],
                agent.hyperpm.theta_smooth, agent.hyperpm.theta_degree,
                agent.hyperpm.theta_sparsity) for i in range(model_num)) / 4.0
    logp, _, _ = agent.ModalFusion(fused)
    cls = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    (cls + graph).backward()
    registered = [p for m in (agent.MessagePassing0, agent.ModalFusion)
                  for p in m.parameters() if p.requires_grad]
    effective = sum(int(p.grad is not None and torch.isfinite(p.grad).all() and
                        float(p.grad.norm()) > 0) for p in registered)
    agent.optimizer_MP0.step(); agent.optimizer_MF.step()
    return float(cls.detach()), float(graph.detach()), int(effective)


def evaluate_base(agent, c, detail=False):
    torch = c["torch"]
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    captured = {}
    hooks = []
    if detail:
        hooks.append(agent.ModalFusion.FeedForward[-1].register_forward_hook(
            lambda _m, _i, out: captured.update(cam=out.detach().clone())))
        hooks.append(agent.ModalFusion.Outputlayer.register_forward_pre_hook(
            lambda _m, inputs: captured.update(head_cam=inputs[0].detach().clone(),
                                                attention=inputs[1].detach().clone())))
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        fused, adjs, alignment = agent.MessagePassing0(parts, model_num,
                                                       agent.targ, agent.trn_idx)
        logp, _, _ = agent.ModalFusion(fused)
        graph = sum(c["utils"].GraphConstructLoss(parts[i], adjs[i],
                    agent.hyperpm.theta_smooth, agent.hyperpm.theta_degree,
                    agent.hyperpm.theta_sparsity) for i in range(model_num)) / 4.0
    for hook in hooks:
        hook.remove()
    if not torch.isfinite(logp).all() or logp.shape != (871, 2):
        raise FloatingPointError("invalid baseline output")
    out = dict(logp=logp.detach().clone())
    if detail:
        out.update(cam=captured["cam"], attention=captured["attention"],
                   fused=fused.detach().clone(), graph_loss=float(graph.detach()),
                   alignment_loss=float(alignment.detach()))
    return out


def metric_row(logp, y, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    pred = logp.detach().cpu().numpy()[test].argmax(1)
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, pred, labels=[0, 1]).ravel())
    return dict(acc=float(np.mean(pred == target)),
                hard_auc=float(roc_auc_score(np.eye(2)[target], np.eye(2)[pred])),
                spe_td=float(tn / (tn + fp)), sen_asd=float(tp / (tp + fn)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def model_state(agent):
    return {"gfr": copy.deepcopy(agent.MessagePassing0.state_dict()),
            "pln": copy.deepcopy(agent.ModalFusion.state_dict())}


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])



def atomic_torch_save(torch,value,path):
    # Payload and its digest live in one atomically replaced file. The external
    # SHA sidecar is repairable metadata if interrupted between the two writes.
    import io
    payload=io.BytesIO();torch.save(value,payload);raw=payload.getvalue()
    envelope={'payload_bytes':raw,'payload_sha256':sha_bytes(raw)}
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');torch.save(envelope,temp)
    os.replace(temp,path);path.with_suffix(path.suffix+'.sha256').write_text(sha_file(path)+'\n')


def checked_load(torch,path,device='cpu'):
    import io
    path=Path(path);envelope=torch.load(path,map_location='cpu',weights_only=False)
    raw=envelope['payload_bytes']
    if sha_bytes(raw)!=envelope['payload_sha256']:raise RuntimeError('checkpoint payload SHA mismatch: '+str(path))
    sidecar=path.with_suffix(path.suffix+'.sha256');file_digest=sha_file(path)
    if not sidecar.exists() or sidecar.read_text().strip()!=file_digest:
        sidecar.write_text(file_digest+'\n') # verified envelope recovers an interrupted sidecar write
    return torch.load(io.BytesIO(raw),map_location=device,weights_only=False)


def epoch_indices(next_epoch,wait):
    # A terminal latest checkpoint may exist before the completed-row commit.
    # Finalize it without an extra optimizer update on resume.
    return range(next_epoch,450) if wait<=200 else ()


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def load_ontology(): return json.loads((EXP/'ontology.json').read_text())


def evidence_path(fold,mode): return EVIDENCE_DIR/f'fold{fold:02d}_{mode}.npz'


def load_evidence(fold,mode):
    with np.load(evidence_path(fold,mode),allow_pickle=False) as a:
        ev={k:a[k] for k in a.files if k!='stats_json'};ev['stats']=json.loads(str(a['stats_json'].item()))
    return ev


def bundle(group,ev,o,x,train,calibration=None):
    weights=np.asarray(o['weights']); coverage=np.asarray(o['coverage']); info={}
    if group=='E5_WRONG_SEMANTICS': weights=np.asarray(o['wrong_weights']);coverage=np.asarray(o['wrong_coverage'])
    if group in ('E2_NUMERIC_ONLY','E6_DATA_PRIOR'):
        weights,info=integration.numeric_weights(weights,ev['known_edge'],
             'fixed_numeric' if group=='E2_NUMERIC_ONLY' else 'data_prior',x,train)
        # Numeric observations have unit coverage; no HO identity/degree is read.
        coverage=np.ones(4,dtype=float)
    eligible=np.array([p['model_eligible'] for p in json.loads((EXP/'prompts.json').read_text())['concepts']],bool)
    if group in ('E2_NUMERIC_ONLY','E6_DATA_PRIOR'): eligible=np.ones(4,bool)
    b=evidence_for_mapping(ev,weights,coverage,eligible,use_qc=group!='E4_NO_QC',
                          forced=group=='E7_FORCED_ALIGNMENT',calibration=calibration)
    b['mapping_info']=info;b['weights']=weights
    return b


def prepare():
    for p in (RESULTS,RUNTIME,ANCHOR_DIR,CKPT_DIR,EVIDENCE_DIR,LOG_DIR,STATE_DIR): p.mkdir(parents=True,exist_ok=True)
    x,y,folds,mapping=read_data(); o=integration.compile_ontology(EXP,ROOT)
    for fold,(train,test) in enumerate(folds,1):
        for mode in ('conditional','global'):
            ev=build_normative(x,train,y,np.asarray(o['endpoints']),mode)
            arrays={k:v for k,v in ev.items() if isinstance(v,np.ndarray)}
            arrays['stats_json']=np.array(json.dumps(ev['stats']))
            np.savez_compressed(evidence_path(fold,mode),**arrays)
    config=dict(protocol_version='41.1-corrected-20261026',baseline=PROTOCOL,adaptation=dict(lr_output=ADAPT_LR,lr_branch=ADAPT_LR,
       optimizer='Adam',weight_decay=0.,energy_weight=ENERGY_WEIGHT,residual_cap=RESIDUAL_CAP,maximum_epochs=450,early_wait=200,
       checkpoint_every=CHECKPOINT_EVERY,branch_init_seed='411000+fold',training_rng_seed='412000+fold',
       frozen_trunk_mode='eval and cached CAM',head_dropout='train for all adaptation groups, including E8; weights frozen only in E8'),
       evidence=dict(input_shape=[4,18],rank=8,branch_parameters=5760,bias=False,reference='5-fold OOF train/all train TD test',
       qc_columns=[57,58,60,62],qc_direction='higher worse; positive standardized excess only',
       qc_severe_quantile=.995,minimum_reference_support=.1,score='mean across scales of .5*mean_abs_z + .5*fraction_abs_z_gt_1',
       calibration='same statistic empirical CDF on train TD OOF',mass='q=a*CDF; reject=product(1-q); matched=(1-reject)*q/sum(q)'),
       go=dict(min_acc_gain=.002,min_hard_auc_gain=.002,min_wins=7,minimum_sen_delta=-.005,
       semantics_controls=['E1_HEAD_ONLY','E2_NUMERIC_ONLY','E5_WRONG_SEMANTICS','E6_DATA_PRIOR'],
       threshold_scope='Internal screening only. No p-value or independent generalization claim.'),
       gpu_assignment=GPU_GROUPS,groups=CONFIGS,
       correction='Previous invalid outcomes already observed; changes follow user requirements, not test-metric tuning. No blinded preregistration claim.')
    write_json(EXP/'config.json',config)
    old=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    sources=[ROOT/n for n in ('main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh')]
    m=dict(experiment='Experiment 41.1 — SAGE-Align Final',status='PREPARED_CORRECTED',prepared_utc=utc(),target_folds=100,
       previous_execution=old.get('excluded_run'),wall_start_utc=old.get('wall_start_utc',utc()),corrected_start_utc=utc(),
       source_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in sources},
       data_sha256={str(p.relative_to(ROOT)):sha_file(p) for p in (DATA_FILE,MODAL_FILE)},
       private_source_sha256={p.name:sha_file(p) for p in SOURCE.glob('*.py')},
       cache_sha256={str(p.relative_to(EXP)):sha_file(p) for p in EVIDENCE_DIR.glob('*.npz')},
       historical_protocol='val=test, pre-CV RFE and preprocessing, transductive 871 patients, seed0 only',
       gpu_assignment=GPU_GROUPS,completed_folds=0)
    write_json(MANIFEST,m);print('PREPARED_CORRECTED',flush=True)


def freeze():
    m=json.loads(MANIFEST.read_text()); pre=json.loads((RUNTIME/'preflight.json').read_text())
    audit=json.loads((RUNTIME/'audit.json').read_text())
    if pre['status']!='PASS' or audit['status']!='AUDITED_WITH_DISCLOSED_ACCESS_LIMITS': raise RuntimeError('audit/preflight incomplete')
    for name,digest in pre['checked_source_sha256'].items():
        if sha_file(EXP/name)!=digest: raise RuntimeError('preflight source changed: '+name)
    paths=[EXP/n for n in ['run.py','sage_align.py','integration.py','audit.py','config.json','prompts.json','ontology.json','design_audit.md']]
    paths+=list(SOURCE.glob('*.py')); paths+=[RUNTIME/'preflight.json',RUNTIME/'audit.json']
    m['frozen_sha256']={str(p.relative_to(EXP)):sha_file(p) for p in paths}
    m['freeze_utc']=utc();m['status']='FROZEN_CORRECTED';m['signature']=sha_bytes(json.dumps(m['frozen_sha256'],sort_keys=True).encode())
    write_json(MANIFEST,m);print('FROZEN',m['signature'],flush=True)


def verify_freeze():
    m=json.loads(MANIFEST.read_text())
    for p,h in m['frozen_sha256'].items():
        if sha_file(EXP/p)!=h: raise RuntimeError('frozen file changed: '+p)
    for p,h in {**m['source_sha256'],**m['data_sha256']}.items():
        if sha_file(ROOT/p)!=h: raise RuntimeError('protected source/data changed: '+p)
    return m['signature']


def trunk_hash(agent):
    h=hashlib.sha256()
    for prefix,module in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]:
        for name,v in sorted(module.state_dict().items()):
            if prefix=='pln' and name.startswith('Outputlayer.'): continue
            h.update((prefix+'/'+name).encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_e0(gpu=0):
    signature=verify_freeze();c=context(gpu);torch=c['torch']
    if gpu!=0: raise RuntimeError('E0 is GPU0')
    state_path=STATE_DIR/'E0_B0A.json'
    saved=json.loads(state_path.read_text()) if state_path.exists() else {'rows':[]}
    rows=saved['rows'];done={int(r['fold']) for r in rows}
    if done: c['network'].get_prompt_text_features() # warm cache, then restore boundary RNG
    seed_everything(torch,0); started=time.monotonic()
    for fold,(train,test) in enumerate(c['folds'],1):
        anchor_path=ANCHOR_DIR/f'anchor_fold{fold:02d}.pt'
        if fold in done:
            ck=checked_load(torch,anchor_path)
            if ck['signature']!=signature: raise RuntimeError('E0 signature')
            rng_restore(torch,ck['continuation_rng']); continue
        entry=rng_capture(torch);agent=make_agent(c,fold);initial_sha=source_digest(agent)
        best_acc=0.;best_epoch=-1;wait=0;best=None;trace=[];start_epoch=0
        latest=CKPT_DIR/f'E0_B0A_fold{fold:02d}_latest.pt';fstart=time.monotonic()
        if latest.exists():
            ck=checked_load(torch,latest)
            if ck['signature']!=signature: raise RuntimeError('E0 resume signature')
            load_model_state(agent,ck['model']);agent.optimizer_MP0.load_state_dict(ck['opt_gfr']);agent.optimizer_MF.load_state_dict(ck['opt_pln'])
            rng_restore(torch,ck['rng']);best=ck['best'];trace=ck['trace'];start_epoch=ck['next_epoch'];wait=ck['wait']
            best_acc=ck['best_acc'];best_epoch=ck['best_epoch'];entry=ck['entry_rng'];initial_sha=ck['initial_sha']
        epoch=start_epoch-1
        for epoch in epoch_indices(start_epoch,wait):
            cls,graph,effective=e0_step(agent,c);logp=evaluate_base(agent,c)['logp']
            acc=float(logp[test].argmax(1).eq(agent.targ[test]).double().mean())
            trace.append(dict(epoch=epoch,classification_loss=cls,graph_loss=graph,validation_acc=acc))
            if acc>best_acc:
                best_acc=acc;best_epoch=epoch;wait=0
                best=dict(model=model_state(agent),opt_gfr=copy.deepcopy(agent.optimizer_MP0.state_dict()),
                          opt_pln=copy.deepcopy(agent.optimizer_MF.state_dict()),rng=rng_capture(torch))
            else:wait+=1
            if (epoch+1)%CHECKPOINT_EVERY==0 or wait>200 or epoch==449:
                atomic_torch_save(torch,dict(signature=signature,model=model_state(agent),opt_gfr=agent.optimizer_MP0.state_dict(),
                  opt_pln=agent.optimizer_MF.state_dict(),rng=rng_capture(torch),best=best,best_acc=best_acc,best_epoch=best_epoch,
                  next_epoch=epoch+1,wait=wait,trace=trace,entry_rng=entry,initial_sha=initial_sha),latest)
            if wait>200:break
        continuation=rng_capture(torch);load_model_state(agent,best['model']);detail=evaluate_base(agent,c,True)
        metrics=metric_row(detail['logp'],c['y'],test)
        row=dict(config='E0_B0A',seed=0,fold=fold,physical_gpu=0,**metrics,best_epoch=best_epoch,epochs_executed=epoch+1,
           initial_trunk_sha256=initial_sha,initial_rng_sha256=rng_hash(entry),wall_seconds=time.monotonic()-fstart,
           branch_parameters=0,outputlayer_parameters=sum(p.numel() for p in agent.ModalFusion.Outputlayer.parameters()),
           trainable_parameters=sum(p.numel() for m in [agent.MessagePassing0,agent.ModalFusion] for p in m.parameters()),
           prediction_flips_vs_anchor=0,signature=signature)
        atomic_torch_save(torch,dict(signature=signature,fold=fold,model=best['model'],opt_gfr=best['opt_gfr'],opt_pln=best['opt_pln'],
          best_rng=best['rng'],continuation_rng=continuation,cam=detail['cam'].cpu(),attention=detail['attention'].cpu(),
          fused=detail['fused'].cpu(),logp=detail['logp'].cpu(),graph_loss=detail['graph_loss'],base_row=row,train_idx=train,test_idx=test),anchor_path)
        write_csv(LOG_DIR/f'E0_B0A_fold{fold:02d}_trace.csv',trace)
        rows.append(row);write_json(state_path,dict(status='RUNNING',signature=signature,rows=rows))
        # Completed-fold anchor includes best Adam/RNG and exact next-fold RNG.
        latest.unlink();latest.with_suffix('.pt.sha256').unlink()
        del agent;torch.cuda.empty_cache();rng_restore(torch,continuation)
        print(f'E0_FOLD_DONE {fold}/10 acc={metrics["acc"]:.7f} best={best_epoch}',flush=True)
    historical={int(r['fold']):r for r in csv.DictReader((HIST32/'results/fold_metrics.csv').open()) if r['config']=='B0' and r['seed']=='0'}
    parity=[]
    for r in rows:
        h=historical[int(r['fold'])]
        parity.append(all(abs(float(r[a])-float(h[b]))<1e-12 for a,b in [('acc','acc'),('hard_auc','auc'),('spe_td','spe_td'),('sen_asd','sen_asd')]) and
          all(int(r[k])==int(h[k]) for k in ['best_epoch','epochs_executed']) and r['initial_trunk_sha256']==h['initial_trunk_sha256'] and r['initial_rng_sha256']==h['initial_rng_sha256'])
    write_json(state_path,dict(status='COMPLETE' if all(parity) else 'FAILED_PARITY',signature=signature,rows=rows,
       parity=parity,wall_seconds=float(saved.get('wall_seconds',0))+time.monotonic()-started,completed_utc=utc()))
    if not all(parity):raise RuntimeError('E0 parity failure '+str(parity))
    print('E0_PARITY_10_OF_10',flush=True)


def setup_adaptation(c,fold,group,anchor):
    torch=c['torch'];agent=make_agent(c,fold);load_model_state(agent,anchor['model'])
    agent.MessagePassing0.eval();agent.ModalFusion.eval()
    for m in (agent.MessagePassing0,agent.ModalFusion):
        for p in m.parameters(): p.requires_grad_(False)
    head=agent.ModalFusion.Outputlayer; head_train=group!='E8_FROZEN_HEAD'
    for p in head.parameters():p.requires_grad_(head_train)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(411000+fold);torch.cuda.manual_seed_all(411000+fold)
        branch=SAGEAlign().to(agent.dev)
    if group=='E1_HEAD_ONLY':branch=None
    params=(list(branch.parameters()) if branch is not None else [])+(list(head.parameters()) if head_train else [])
    optimizer=torch.optim.Adam(params,lr=ADAPT_LR,weight_decay=0.)
    seed_everything(torch,412000+fold)
    return agent,branch,optimizer,params


def adapt_forward(branch,head,features,cam,attention,training=False):
    head.train(training)
    if branch is None:
        augmented=cam;residual=cam.new_zeros((len(cam),144));extra=dict(raw=residual,scale=cam.new_ones((len(cam),1)),cap=.08*cam.flatten(1).norm(dim=1,keepdim=True))
    else: augmented,residual,extra=branch(features,cam,details=True)
    logp,_=head(augmented.reshape(len(cam),-1),attention.reshape(len(cam),-1))
    return logp,residual,extra


def adaptation_step(agent,branch,optimizer,params,features,cam,attention,c,train):
    torch=c['torch'];optimizer.zero_grad(set_to_none=True)
    logp,residual,extra=adapt_forward(branch,agent.ModalFusion.Outputlayer,features,cam,attention,True)
    cls=c['utils'].ClsLoss(logp,agent.targ,train,agent.weight)
    energy=(residual[train].square().sum(1)/cam[train].flatten(1).square().sum(1).clamp_min(1e-12)).mean()
    loss=cls+ENERGY_WEIGHT*energy
    if not torch.isfinite(loss):raise FloatingPointError('adaptation loss')
    loss.backward()
    gradients={}
    for prefix,module in [('branch',branch),('output',agent.ModalFusion.Outputlayer)]:
        if module is None:continue
        for name,p in module.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError('missing/nonfinite gradient '+prefix+'/'+name)
                gradients[prefix+'/'+name]=float(p.grad.norm())
    optimizer.step()
    return float(cls.detach()),float(energy.detach()),gradients


def branch_diag(residual,extra,cam,ids):
    norm=cam[ids].flatten(1).norm(dim=1).clamp_min(1e-12); raw=extra['raw'][ids].norm(dim=1);eff=residual[ids].norm(dim=1)
    return dict(residual_l2_mean=float(eff.mean()),raw_residual_l2_mean=float(raw.mean()),
                residual_cam_ratio=float((eff/norm).mean()),raw_cap_ratio=float((raw/(.08*norm)).mean()),
                effective_cap_ratio=float((eff/(.08*norm)).mean()),effective_raw_ratio=float((eff/raw.clamp_min(1e-12)).mean()),
                saturation_fraction=float((raw/(.08*norm)>3).float().mean()))


def run_group(group,gpu):
    signature=verify_freeze()
    if group not in GPU_GROUPS[gpu] or group=='E0_B0A':raise RuntimeError('GPU group assignment')
    c=context(gpu);torch=c['torch'];o=load_ontology();state_path=STATE_DIR/f'{group}.json'
    saved=json.loads(state_path.read_text()) if state_path.exists() else {'rows':[]}
    rows=saved['rows'];done={int(r['fold']) for r in rows};started=time.monotonic()
    for fold,(train,test) in enumerate(c['folds'],1):
        if fold in done:continue
        anchor_path=ANCHOR_DIR/f'anchor_fold{fold:02d}.pt';anchor=checked_load(torch,anchor_path)
        if anchor['signature']!=signature:raise RuntimeError('anchor signature')
        agent,branch,optimizer,params=setup_adaptation(c,fold,group,anchor)
        frozen_hash=trunk_hash(agent);initial_head=copy.deepcopy(agent.ModalFusion.Outputlayer.state_dict())
        init_branch_sha=sha_bytes(b''.join(p.detach().cpu().numpy().tobytes() for p in branch.parameters())) if branch is not None else None
        train_rng_hash=rng_hash(rng_capture(torch))
        ev=load_evidence(fold,'global' if group=='E3_GLOBAL_REFERENCE' else 'conditional')
        b=bundle(group,ev,o,c['x'],train);features=torch.tensor(b['effective'],device=agent.dev)
        cam=anchor['cam'].to(agent.dev);attention=anchor['attention'].to(agent.dev)
        best_acc=0.;best_epoch=-1;wait=0;best=None;trace=[];start_epoch=0;resume_count=0
        latest=CKPT_DIR/f'{group}_fold{fold:02d}_latest.pt';fstart=time.monotonic()
        if latest.exists():
            ck=checked_load(torch,latest)
            if ck['signature']!=signature or ck['group']!=group:raise RuntimeError('adaptation resume signature')
            if branch is not None:branch.load_state_dict(ck['branch'])
            agent.ModalFusion.Outputlayer.load_state_dict(ck['output']);optimizer.load_state_dict(ck['optimizer']);rng_restore(torch,ck['rng'])
            best=ck['best'];trace=ck['trace'];wait=ck['wait'];best_acc=ck['best_acc'];best_epoch=ck['best_epoch'];start_epoch=ck['next_epoch'];resume_count=ck.get('resume_count',0)+1
        epoch=start_epoch-1
        for epoch in epoch_indices(start_epoch,wait):
            cls,energy,grad=adaptation_step(agent,branch,optimizer,params,features,cam,attention,c,train)
            with torch.no_grad():logp,residual,extra=adapt_forward(branch,agent.ModalFusion.Outputlayer,features,cam,attention,False)
            acc=float(logp[test].argmax(1).eq(agent.targ[test]).double().mean())
            trace.append(dict(epoch=epoch,classification_loss=cls,residual_energy=energy,loss=cls+ENERGY_WEIGHT*energy,
               frozen_graph_loss=anchor['graph_loss'],validation_acc=acc,prediction_flips_vs_anchor=int((logp[test].argmax(1)!=anchor['logp'][test].to(agent.dev).argmax(1)).sum()),**branch_diag(residual,extra,cam,train)))
            if acc>best_acc:
                best_acc=acc;best_epoch=epoch;wait=0
                best=dict(branch=copy.deepcopy(branch.state_dict()) if branch is not None else None,
                    output=copy.deepcopy(agent.ModalFusion.Outputlayer.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),rng=rng_capture(torch),gradients=grad)
            else:wait+=1
            if (epoch+1)%CHECKPOINT_EVERY==0 or wait>200 or epoch==449:
                atomic_torch_save(torch,dict(signature=signature,group=group,fold=fold,
                 branch=branch.state_dict() if branch is not None else None,output=agent.ModalFusion.Outputlayer.state_dict(),optimizer=optimizer.state_dict(),
                 rng=rng_capture(torch),best=best,best_acc=best_acc,best_epoch=best_epoch,wait=wait,next_epoch=epoch+1,trace=trace,resume_count=resume_count),latest)
            if wait>200:break
        if trunk_hash(agent)!=frozen_hash:raise RuntimeError('frozen trunk updated')
        if group=='E8_FROZEN_HEAD' and not all(torch.equal(initial_head[k],v) for k,v in agent.ModalFusion.Outputlayer.state_dict().items()):raise RuntimeError('E8 head updated')
        if branch is not None:branch.load_state_dict(best['branch'])
        agent.ModalFusion.Outputlayer.load_state_dict(best['output'])
        with torch.no_grad():logp,residual,extra=adapt_forward(branch,agent.ModalFusion.Outputlayer,features,cam,attention,False)
        if branch is not None and not zero_evidence_test(branch,cam):raise RuntimeError('semantic bypass')
        metrics=metric_row(logp,c['y'],test)
        row=dict(config=group,seed=0,fold=fold,physical_gpu=gpu,**metrics,best_epoch=best_epoch,epochs_executed=epoch+1,
            branch_parameters=sum(p.numel() for p in branch.parameters()) if branch is not None else 0,
            outputlayer_parameters=sum(p.numel() for p in agent.ModalFusion.Outputlayer.parameters()),trainable_parameters=sum(p.numel() for p in params),
            initial_branch_sha256=init_branch_sha,initial_head_sha256=sha_bytes(b''.join(v.cpu().numpy().tobytes() for v in initial_head.values())),
            training_rng_sha256=train_rng_hash,frozen_trunk_sha256=frozen_hash,anchor_sha256=sha_file(anchor_path),
            matched_test_mean=float(b['matched'][test].sum(1).mean()),unmatched_test_mean=float(b['unmatched'][test].mean()),
            hard_reject_test_fraction=float((~b['accepted'][test]).mean()),active_feature_channels=int(np.any(b['features'][train]!=0,axis=(0,1)).sum()),
            evidence_matrix_rank=int(np.linalg.matrix_rank(b['effective'][train].reshape(len(train),-1).astype('float64'),tol=1e-5)),
            prediction_flips_vs_anchor=int((logp[test].argmax(1)!=anchor['logp'][test].to(agent.dev).argmax(1)).sum()),
            resume_count=resume_count,wall_seconds=time.monotonic()-fstart,signature=signature,**branch_diag(residual,extra,cam,test))
        ck=dict(signature=signature,group=group,fold=fold,branch=best['branch'],output=best['output'],optimizer=best['optimizer'],rng=best['rng'],
          row=row,calibration=b['calibration'],weights=b['weights'],coverage=b['coverage'],eligible=b['eligible'],mapping_info=b['mapping_info'],
          logp=logp.detach().cpu(),best_epoch=best_epoch,anchor_sha256=sha_file(anchor_path),gradients=best['gradients'])
        atomic_torch_save(torch,ck,CKPT_DIR/f'{group}_fold{fold:02d}_best.pt')
        write_csv(LOG_DIR/f'{group}_fold{fold:02d}_trace.csv',trace)
        rows.append(row);write_json(state_path,dict(status='RUNNING',signature=signature,rows=rows))
        print(f'ADAPT_FOLD_DONE {group} {fold}/10 acc={metrics["acc"]:.7f} best={best_epoch}',flush=True)
        del agent,branch,optimizer;torch.cuda.empty_cache()
    write_json(state_path,dict(status='COMPLETE',signature=signature,rows=rows,completed_utc=utc(),
       wall_seconds=float(saved.get('wall_seconds',0))+time.monotonic()-started))


def dispatch():
    verify_freeze();procs=[]
    for gpu in (0,1):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(LOG_DIR/f'gpu{gpu}.log').open('a',buffering=1)
        procs.append(subprocess.Popen([sys.executable,str(EXP/'run.py'),'worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in procs]
    if any(codes):raise RuntimeError('worker exit '+str(codes))


def worker(gpu):
    for group in GPU_GROUPS[gpu]:
        if group!='E0_B0A':run_group(group,gpu)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','e0','dispatch','worker','preflight','report','finalize']);p.add_argument('arg',type=int,nargs='?')
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='freeze':freeze()
    elif a.command=='e0':run_e0(a.arg or 0)
    elif a.command=='worker':worker(a.arg or 0)
    elif a.command=='dispatch':dispatch()
    elif a.command=='preflight':
        from audit import preflight
        preflight()
    elif a.command=='report':
        from audit import report
        report()
    elif a.command=='finalize':
        from audit import finalize
        finalize()

if __name__=='__main__':main()
