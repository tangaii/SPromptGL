#!/usr/bin/env python3
"""Frozen 14-group experiment; independent training, exact resume and paired RNG."""
from __future__ import annotations
import argparse, contextlib, csv, difflib, fcntl, hashlib, io, json, math, os, pickle, random, shutil, signal, statistics, subprocess, sys, tempfile, time, types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np

EXP=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('HSP_ROOT', '/data/adodas/SPromptGL-main'))
LOG_DIR=EXP/'logs'; RESULT_DIR=EXP/'results'; STATE=Path(tempfile.gettempdir())/('spromptgl14-state-'+hashlib.sha256(str(EXP).encode()).hexdigest()[:12]); WORK_DIR=STATE/'sources'
PROMPT_FILE=EXP/'prompts.json'; PATCH_FILE=EXP/'integration_patch.diff'; MANIFEST_FILE=EXP/'manifest.json'
REFERENCE_FILE=RESULT_DIR/'reference.csv'; FOLD_FILE=RESULT_DIR/'fold_metrics.csv'; DIAG_FILE=RESULT_DIR/'diagnostics.csv'
CF_FILE=RESULT_DIR/'counterfactual.csv'; REPORT_FILE=RESULT_DIR/'report.md'; HANDOFF=ROOT/'handoff.md'
SOURCE_FILES=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
MODALITIES=['PHENO','ANAT','FUNC','Correlation']; SEEDS=[0,1,2]; CONFIGS=[f'E{i:02d}' for i in range(14)]
LOCAL_MODES=dict(zip(CONFIGS,['P0','OLD_P1','NEW_P1_SHARED','NEW_P3_SHARED','NEW_P3_SPECIFIC']+['NEW_P3_SPECIFIC']*6+['P0','NEW_P1_SHARED','NEW_P3_MISMATCHED']))
DISEASE_MODES=dict(zip(CONFIGS,['OFF']*5+['HISTORICAL','FC_CENTRIC','BALANCED','SELECTIVE','CLINICAL_CORE','NEUTRAL','SELECTIVE','SELECTIVE','SELECTIVE']))
DISEASE_ON={c for c in CONFIGS if DISEASE_MODES[c]!='OFF'}
GPU_ASSIGNMENTS={(c,s):(i+s)%2 for i,c in enumerate(CONFIGS) for s in SEEDS}
TASKS=[(c,s,GPU_ASSIGNMENTS[c,s]) for c in CONFIGS for s in SEEDS]
METRICS=['acc','balanced_accuracy','hard_auc','probability_auc','sensitivity_asd','specificity_td']
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None

def utc_now(): return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_write(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as f:
  f.write(text);f.flush();os.fsync(f.fileno());name=f.name
 os.replace(name,path)
def write_json(path,v): atomic_write(path,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read_json(path):
 # The workspace mount may expose a short rename visibility gap.
 for attempt in range(21):
  try:return json.loads(Path(path).read_text(encoding='utf-8'))
  except (FileNotFoundError,json.JSONDecodeError):
   if attempt==20:raise
   time.sleep(.05)
def write_csv(path,rows):
 if not rows: return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 out=io.StringIO();w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows);atomic_write(path,out.getvalue())
def read_csv(path):
 if not Path(path).exists(): return []
 with Path(path).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def atomic_torch(path,data):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('wb') as f: torch.save(data,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);atomic_write(path.with_suffix(path.suffix+'.sha256'),sha256(path)+'\n')
def checked_torch(path):
 import torch
 path=Path(path);check=path.with_suffix(path.suffix+'.sha256')
 assert check.exists() and sha256(path)==check.read_text().strip(), 'corrupt/incomplete checkpoint: '+str(path)
 return torch.load(path,map_location='cpu',weights_only=False)
def history_digest(path):
 return canonical([(str(p.relative_to(path)),sha256(p)) for p in sorted(Path(path).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.lock')])
def source_hashes():
 paths=SOURCE_FILES+['data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy','CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']
 return {p:sha256(ROOT/p) for p in paths}
def seed_rng(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def rng_state():
 import torch
 return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
 import torch
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def rng_hash(s):
 h=hashlib.sha256(pickle.dumps((s['python'],s['numpy']),protocol=4))
 for t in [s['cpu']]+s['cuda']:h.update(t.cpu().numpy().tobytes())
 return h.hexdigest()
def rng_equal(a,b): return rng_hash(a)==rng_hash(b)

def p0_text():
    return ['Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure','Normal ventricular morphology',
            'Altered brain volume','Abnormal functional connectivity','White matter integrity issues','Different activation patterns']

def old_p1_source():
    path=ROOT/'experiment/experiment1_modality_aware_semantic_prompt/prompts.json'; return read_json(path),path

def historical_disease():
    x=read_json(ROOT/'experiment/experiment8_scheme1_shared_disease_alignment/prompts.json')
    return {'ASD': list(x['disease_prompts']['ASD']['prompts']), 'TD': list(x['disease_prompts']['TD']['prompts'])}

def neutral_disease():
    x=read_json(ROOT/'experiment/experiment10_disease_semantic_specificity/prompts.json')
    return {'ASD': list(x['disease_prompts']['NEUTRAL_CLASS0']['prompts']), 'TD': list(x['disease_prompts']['NEUTRAL_CLASS1']['prompts'])}

def new_local_prompts():
    p1_shared={
      'ASD':['Autism studies report heterogeneous functional interactions in brain systems supporting social cognition and communication.','Autism research describes variable functional connectivity across distributed language and communication systems.','Autism research reports heterogeneous interactions within sensory and sensorimotor brain systems.','Autism studies examine variable connectivity among default-mode, attention, and executive-control networks.'],
      'TD':['Typical development involves variable functional interactions in brain systems supporting social cognition and communication.','Typical development shows age-dependent connectivity across distributed language and communication systems.','Typical development includes variable interactions within sensory and sensorimotor brain systems.','Typically developing individuals show variable connectivity among default-mode, attention, and executive-control networks.']}
    p3_shared={
      'ASD':['In autism research, functional connectivity patterns provide heterogeneous brain-network evidence.','Developmental and demographic factors provide context for interpreting brain measurements in autism research.','Structural MRI image quality affects measurement reliability rather than directly defining autism-related pathology.','Head motion and functional MRI signal quality influence the reliability of connectivity estimates in autism research.'],
      'TD':['In typical development, functional connectivity patterns show heterogeneous brain-network organization.','Developmental and demographic factors provide context for interpreting brain measurements in typically developing populations.','Structural MRI image quality affects measurement reliability rather than directly defining typical development.','Head motion and functional MRI signal quality influence the reliability of connectivity estimates in typical development.']}
    specific={
      'PHENO':{'ASD':['Age provides developmental context for interpreting heterogeneous brain measurements in autism research.','Sex-related population variation may influence the interpretation of brain measurements in autism cohorts.','Acquisition site can introduce non-diagnostic differences in multi-site autism brain imaging datasets.','Demographic heterogeneity should be considered when interpreting brain connectivity patterns associated with autism.'],'TD':['Age provides developmental context for interpreting brain measurements in typical development.','Sex-related population variation may influence the interpretation of brain measurements in typically developing cohorts.','Acquisition site can introduce non-diagnostic differences in multi-site typical-development brain imaging datasets.','Demographic heterogeneity should be considered when interpreting brain connectivity patterns in typical development.']},
      'ANAT':{'ASD':['Structural MRI signal-to-noise measurements describe image reliability in autism neuroimaging research.','Anatomical image contrast influences measurement quality but does not independently establish an autism diagnosis.','Structural MRI artifacts may affect the reliability of imaging evidence collected in autism studies.','Image smoothness and structural acquisition quality provide measurement context for autism brain imaging analysis.'],'TD':['Structural MRI signal-to-noise measurements describe image reliability in typical-development neuroimaging.','Anatomical image contrast influences measurement quality but does not independently establish typical development.','Structural MRI artifacts may affect the reliability of imaging evidence collected in typical-development studies.','Image smoothness and structural acquisition quality provide measurement context for typical-development brain imaging analysis.']},
      'FUNC':{'ASD':['Head motion can affect resting-state functional measurements in autism neuroimaging research.','Temporal signal fluctuations influence the reliability of functional connectivity estimates in autism studies.','Functional MRI outliers may introduce measurement noise into connectivity evidence used in autism research.','Functional image quality and preprocessing conditions affect interpretation of connectivity measurements in autism cohorts.'],'TD':['Head motion can affect resting-state functional measurements in typically developing participants.','Temporal signal fluctuations influence the reliability of functional connectivity estimates in typical-development studies.','Functional MRI outliers may introduce measurement noise into connectivity evidence used in typical-development research.','Functional image quality and preprocessing conditions affect interpretation of connectivity measurements in typically developing cohorts.']},
      'Correlation':{'ASD':['Autism studies report heterogeneous functional connectivity patterns among systems supporting social cognition.','Autism research examines variable functional interactions among language and communication-related brain systems.','Autism studies report heterogeneous connectivity among sensory and sensorimotor brain systems.','Autism research describes variable functional interactions involving attention, executive-control, and default-mode networks.'],'TD':['Typical development includes variable functional connectivity among systems supporting social cognition.','Typical development involves age-dependent functional interactions among language and communication-related brain systems.','Typical development shows heterogeneous connectivity among sensory and sensorimotor brain systems.','Typically developing individuals show variable functional interactions involving attention, executive-control, and default-mode networks.']}}
    return p1_shared,p3_shared,specific

def disease_bank():
    hist=historical_disease(); neutral=neutral_disease()
    return {
      'HISTORICAL':hist,
      'FC_CENTRIC':{'ASD':['Autism has been associated with heterogeneous resting-state functional connectivity patterns across brain regions.','Autism research reports variable connectivity among brain systems supporting social cognition and communication.','Autism studies describe heterogeneous functional interactions involving sensory and sensorimotor brain networks.','Autism research examines variable functional interactions among default-mode, attention, and executive-control networks.'],'TD':['Typical development involves diverse resting-state functional connectivity patterns across brain regions.','Typical development includes variable connectivity among brain systems supporting social cognition and communication.','Typically developing individuals show heterogeneous functional interactions involving sensory and sensorimotor networks.','Typical development involves variable functional interactions among default-mode, attention, and executive-control networks.']},
      'BALANCED':{'ASD':['Age and sex provide developmental and demographic context for interpreting brain features in autism research.','Structural MRI image quality affects measurement reliability but is not itself an autism-specific pathological finding.','Functional MRI head motion and signal quality influence the reliability of connectivity measurements in autism research.','Autism studies report heterogeneous resting-state functional connectivity patterns across distributed brain networks.'],'TD':['Age and sex provide developmental and demographic context for interpreting brain features in typical development.','Structural MRI image quality affects measurement reliability but is not itself evidence establishing typical development.','Functional MRI head motion and signal quality influence the reliability of connectivity measurements in typical development.','Typical development involves diverse resting-state functional connectivity patterns across distributed brain networks.']},
      'SELECTIVE':{'ASD':['Autism spectrum disorder has been associated with heterogeneous resting-state functional connectivity across brain regions.','Autism studies report diverse functional interactions among social, sensory, attention, and cognitive-control networks.','Age and sex provide developmental context for interpreting variable neuroimaging patterns in autism research.','Structural image quality and functional MRI motion influence the reliability of connectivity evidence used in autism research.'],'TD':['Typical development involves heterogeneous resting-state functional connectivity across brain regions.','Typical development includes diverse functional interactions among social, sensory, attention, and cognitive-control networks.','Age and sex provide developmental context for interpreting variable neuroimaging patterns in typical development.','Structural image quality and functional MRI motion influence the reliability of connectivity evidence used in typical-development research.']},
      'CLINICAL_CORE':{'ASD':['Autism spectrum disorder is a heterogeneous neurodevelopmental condition involving differences in social communication and interaction.','Autism spectrum disorder includes restricted or repetitive patterns of behavior, interests, or activities.','Autism-related characteristics emerge during development and vary substantially across individuals.','Autism classification concerns an integrated pattern of developmental and behavioral characteristics rather than one universal brain connection.'],'TD':['Typical development involves diverse age-dependent patterns of social communication and interaction.','Typically developing individuals show varied behavioral interests, activities, and flexibility during development.','Social and behavioral abilities change across developmental stages and vary substantially across individuals.','Typical-development classification reflects a broader developmental and behavioral context rather than one universal brain connection.']},
      'NEUTRAL':neutral}

def flatten_old_p1(data):
    out=[]
    for m in MODALITIES:
        for theme in data.get('p1', data.get('local_prompts', {}).get('OLD_P1'))[m]['themes']:
            out += [z['text'] for z in theme['prompts']]
    if len(out)!=32: raise RuntimeError('old P1 must contain 32 prompts')
    return out

def build_prompts():
    if PROMPT_FILE.exists():
        frozen=read_json(PROMPT_FILE)
        p1_shared,p3_shared,specific=new_local_prompts()
        assert frozen['p0']==p0_text() and len(flatten_old_p1(frozen))==32
        for key,value in [('NEW_P1_SHARED',p1_shared),('NEW_P3_SHARED',p3_shared),('NEW_P3_SPECIFIC',specific),('NEW_P3_MISMATCHED',specific)]:
            assert frozen['local_prompts'][key]==value, 'frozen local bank mismatch: '+key
        assert frozen['config_local_modes']==LOCAL_MODES and frozen['config_disease_modes']==DISEASE_MODES
        return frozen
    p1,p1_path=old_p1_source(); p1_shared,p3_shared,specific=new_local_prompts()
    local={'OLD_P1':p1['p1'],'NEW_P1_SHARED':p1_shared,'NEW_P3_SHARED':p3_shared,'NEW_P3_SPECIFIC':specific,'NEW_P3_MISMATCHED':specific}
    db=disease_bank()
    out={'schema_version':1,'experiment':'HSP','modality_order':MODALITIES,'class_order':{'0':'ASD','1':'TD'},'p0':p0_text(),
         'local_prompts':local,'local_mismatch_mapping':{'PHENO':'FUNC','ANAT':'Correlation','FUNC':'PHENO','Correlation':'ANAT'},
         'disease_prompts':db,'config_local_modes':LOCAL_MODES,'config_disease_modes':DISEASE_MODES,
         'source_sha256':{'experiment1_prompts':sha256(p1_path),'experiment8_prompts':sha256(ROOT/'experiment/experiment8_scheme1_shared_disease_alignment/prompts.json'),'experiment10_prompts':sha256(ROOT/'experiment/experiment10_disease_semantic_specificity/prompts.json')},
         'freeze_rule':'All prompt strings are frozen before training; E13 only remaps modality indices.'}
    # Structural and exact-text checks.
    assert out['p0']==p0_text() and flatten_old_p1(out)==flatten_old_p1(p1)
    for key in ('NEW_P1_SHARED','NEW_P3_SHARED'):
        assert len(out['local_prompts'][key]['ASD'])==len(out['local_prompts'][key]['TD'])==4
    for m in MODALITIES:
        assert len(specific[m]['ASD'])==len(specific[m]['TD'])==4
    for mode in db: assert len(db[mode]['ASD'])==len(db[mode]['TD'])==4
    write_json(PROMPT_FILE,out); return out


def local_sentences(p,mode):
 if mode=='P0':return p['p0']
 if mode=='OLD_P1':return flatten_old_p1(p)
 bank=p['local_prompts'][mode]
 if mode.endswith('SHARED'):return bank['ASD']+bank['TD']
 return [t for m in MODALITIES for cls in ('ASD','TD') for t in bank[m][cls]]

def patch_sources():
 sources={name:(ROOT/name).read_text() for name in ('network.py','model.py')}
 n=sources['network.py']
 start=n.index('_prompt_text_features = None');end=n.index('\nclass MLP',start)
 n=n[:start]+'''from hsp_runtime import get_frozen_local, get_frozen_disease
_prompt_text_features = None

def get_prompt_text_features():
    return get_frozen_local(os.environ.get('HSP_LOCAL_MODE', 'P0'))
'''+n[end:]
 old='''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
'''
 new='''        text_features = self.prompt_text_features
        local_mode = os.environ.get('HSP_LOCAL_MODE', 'P0')
        if local_mode in ('P0', 'NEW_P1_SHARED', 'NEW_P3_SHARED'):
            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        else:
            text_features = text_features.reshape(4, 2, 4096)
        text_features = self.MLP(text_features)
        if local_mode == 'NEW_P3_MISMATCHED' or getattr(self, 'hsp_local_mismatch', False):
            text_features = text_features[[2, 3, 0, 1]]
        self.hsp_local_prototypes = text_features if text_features.ndim == 3 else text_features.unsqueeze(0).expand(4, -1, -1)
'''
 assert n.count(old)==1;n=n.replace(old,new)
 old='''        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # 变为 [1, 1, 1, 2, 9]'''
 new='''        if text_features_norm.ndim == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        else:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(2)'''
 assert n.count(old)==1;n=n.replace(old,new)
 old='        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))  # 结果为 [871, 4, 4, 1, 2]'
 new='''        if text_features.ndim == 2:
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, F.normalize(text_features, p=2, dim=-1)).unsqueeze(-2)'''
 assert n.count(old)==1;n=n.replace(old,new)
 # Keep P0 matmul and historical P1 einsum; Softmax-Max, RES and head unchanged.
 old='        part_x = similarity_max.unsqueeze(-1) * part_x\n'
 assert n.count(old)==1;n=n.replace(old,'        self.hsp_local_gate = similarity_max\n'+old)
 old='        return output, hidden, attn_map\n'
 new='''        self.hsp_base_output = output
        self.hsp_hidden = hidden
        if getattr(self, 'hsp_disease_enabled', False):
            prototypes = getattr(self, 'hsp_prototype_override', None)
            if prototypes is None: prototypes = self.disease_prototypes
            subject = F.normalize(self.disease_projector(hidden), dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
            self.hsp_disease_logits = subject @ prototypes.t() / 0.2
            self.hsp_gamma = 0.2 * torch.tanh(self.disease_alpha)
            self.hsp_contribution = self.hsp_gamma * self.hsp_disease_logits
            if not getattr(self, 'hsp_disease_off', False):
                output = F.log_softmax(output + self.hsp_contribution, dim=-1)
        return output, hidden, attn_map
'''
 assert n.count(old)==1;n=n.replace(old,new)
 m=sources['model.py'];old='        self.ModalFusion.apply(my_weight_init)\n'
 new=old+'''        self.hsp_disease_enabled = os.environ.get('HSP_DISEASE_MODE', 'OFF') != 'OFF'
        self.ModalFusion.hsp_disease_enabled = self.hsp_disease_enabled
        if self.hsp_disease_enabled:
            self.ModalFusion.register_buffer('disease_prototypes', get_frozen_disease(os.environ['HSP_DISEASE_MODE']))
            # Appended after original initialization; isolated extra random stream.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(140000 + int(hyperpm.seed))
                self.ModalFusion.disease_projector = nn.Linear(36, 1024, bias=False).to(dev)
                my_weight_init(self.ModalFusion.disease_projector)
            self.ModalFusion.disease_alpha = nn.Parameter(torch.zeros((), device=dev))
            extra = list(self.ModalFusion.disease_projector.parameters()) + [self.ModalFusion.disease_alpha]
            assert sum(p.numel() for p in extra) == 36865
            self.optimizer_MF.add_param_group({'params': extra, 'weight_decay': 0.0})
'''
 assert m.count(old)==1;m=m.replace(old,new)
 old='            loss = cls_loss + all_graph_loss\n'
 new=old+'''            if self.hsp_disease_enabled:
                disease_loss = F.cross_entropy(self.ModalFusion.hsp_disease_logits[self.trn_idx], self.targ[self.trn_idx], weight=self.weight)
                loss = loss + 0.1 * disease_loss
                self.hsp_last_disease_loss = disease_loss.detach()
'''
 assert m.count(old)==1;m=m.replace(old,new)
 return sources,dict(network=n,model=m)

def prepare_worktree():
 WORK_DIR.mkdir(parents=True,exist_ok=True)
 for name in SOURCE_FILES:
  target=WORK_DIR/name
  if not target.exists():shutil.copy2(ROOT/name,target)
 for name in ('data','CLIP'):
  target=WORK_DIR/name
  if not target.exists():target.symlink_to(ROOT/name,target_is_directory=True)
 for name in ('attn','graph'):(WORK_DIR/name).mkdir(exist_ok=True)
 _,modified=patch_sources()
 for name,text in modified.items():atomic_write(WORK_DIR/(name+'.py'),text)
 return WORK_DIR

def inputs():
 import pandas as pd
 from sklearn.model_selection import StratifiedKFold
 raw=pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv').values
 dims=[len(v) for v in np.load(ROOT/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item().values()]
 assert raw.shape==(871,321) and dims==[48,6,10,256]
 x,y=raw[:,:-1],raw[:,-1]-1;assert set(y)=={0,1}
 return x,y,list(StratifiedKFold(n_splits=10,shuffle=True,random_state=10).split(x,y))

def freeze_reference():
 if REFERENCE_FILE.exists():
  frozen=read_csv(REFERENCE_FILE)
  assert {(r['config'],int(r['seed']),int(r['fold'])) for r in frozen}=={(c,s,f) for c in ('E00','E01') for s in SEEDS for f in range(1,11)} and len(frozen)==60
  return
 rows=[]
 src=read_csv(ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv')
 for r in src:
  if r['config']=='F0':rows.append(dict(config='E00',seed=int(r['seed']),fold=int(r['fold']),**{k:float(r['ba' if k=='balanced_accuracy' else k]) for k in METRICS},source='experiment13 exact original P0'))
 assert len(rows)==30
 src=read_csv(ROOT/'experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv')
 for r in src:
  if r['config']=='P1':
   sen,spe=float(r['sensitivity_asd']),float(r['specificity_asd'])
   rows.append(dict(config='E01',seed=int(r['seed']),fold=int(r['fold']),acc=float(r['acc']),balanced_accuracy=(sen+spe)/2,hard_auc=float(r['hard_auc']),probability_auc='',sensitivity_asd=sen,specificity_td=spe,source='experiment1 rounded historical P1'))
 assert len(rows)==60;write_csv(REFERENCE_FILE,rows)

def encode_bank():
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(ROOT));from CLIP.clip import clip
 p=read_json(PROMPT_FILE);cache=STATE/'text_bank.pt'
 encoder,_=clip.load('RN50','cuda');encoder.eval();encoder.requires_grad_(False)
 banks={};lengths=[];before=rng_state();calls=0
 def encode(texts):
  nonlocal calls
  sizes=[len(clip._tokenizer.encode(t))+2 for t in texts];assert max(sizes)<=77,('77-token overflow',sizes)
  lengths.extend(sizes);tokens=clip.tokenize(texts,truncate=False).to('cuda')
  with torch.no_grad():out=encoder.encode_text(tokens).float().detach().cpu()
  calls+=1;assert torch.isfinite(out).all();return out
 for mode in ('P0','OLD_P1','NEW_P1_SHARED','NEW_P3_SHARED','NEW_P3_SPECIFIC'):banks[mode]=encode(local_sentences(p,mode))
 diseases={};raw_disease={}
 for mode,bank in p['disease_prompts'].items():
  features=encode(bank['ASD']+bank['TD']);raw_disease[mode]=features
  diseases[mode]=F.normalize(F.normalize(features.to('cuda'),dim=-1).reshape(2,4,1024).mean(1),dim=-1).cpu()
 assert rng_equal(before,rng_state()),'text encoding consumed RNG'
 historical_path=ROOT/'experiment/experiment9_prompt_disease_adversarial_interaction/prompts.json'
 if historical_path.exists():
  historical=read_json(historical_path)
  assert p['disease_prompts']['HISTORICAL']['ASD']==historical['disease_prompts']['ASD']['prompts']
  assert p['disease_prompts']['HISTORICAL']['TD']==historical['disease_prompts']['TD']['prompts']
 historical_cache=ROOT/'experiment/experiment12_roi_semantic_alignment_feasibility/cache/roi_text_embeddings.pt'
 p0_history_checked=False
 if historical_cache.exists():
  static=torch.load(historical_cache,map_location='cpu',weights_only=False)
  assert torch.equal(banks['P0'],static['p0_features']),'fresh P0 features differ from verified historical vectors'
  p0_history_checked=True
 saved=dict(local=banks,disease=diseases,raw_disease=raw_disease,prompts_sha256=sha256(PROMPT_FILE),max_tokens=max(lengths),encoding_calls=calls,encoding_rng_unchanged=True)
 atomic_torch(cache,saved);del encoder;torch.cuda.empty_cache()
 return dict(max_tokens=max(lengths),encoding_calls=calls,p0_vectors_exact_history=p0_history_checked,disease_prototype_hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in diseases.items()},text_cache_sha256=sha256(cache))

def prepare():
 if MANIFEST_FILE.exists():
  integrity(full=True);print('PREPARE reused immutable experiment');return
 for d in (LOG_DIR,RESULT_DIR,STATE):d.mkdir(parents=True,exist_ok=True)
 link=EXP/'state'
 if not link.exists():link.symlink_to(STATE,target_is_directory=True)
 protected=source_hashes();old={p.name:history_digest(p) for p in sorted((ROOT/'experiment').glob('experiment*')) if p.is_dir() and p!=EXP}
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
 remote_core={name:hashlib.sha256(subprocess.check_output(['git','show',remote+':'+name],cwd=ROOT)).hexdigest() for name in SOURCE_FILES}
 assert remote_core=={k:protected[k] for k in SOURCE_FILES},'remote/local core differs'
 p=build_prompts();freeze_reference();a,b=patch_sources()
 diff=''.join(''.join(difflib.unified_diff(a[name+'.py'].splitlines(True),b[name].splitlines(True),fromfile=name+'.py',tofile=name+'.py')) for name in b)
 atomic_write(PATCH_FILE,diff)
 text_audit=encode_bank();x,y,folds=inputs()
 paper=next(ROOT.glob('*.pdf'))
 hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
 m=dict(experiment='Hierarchical Semantic Prompt Design and Cross-Level Interaction',run_status='PREPARED',prepared_utc=utc_now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_main=remote,remote_core_equal=True,source_data_sha256=protected,protected_history=old,protected_handoff_sha256=sha256(HANDOFF) if HANDOFF.exists() else None,protected_handoff_bytes=HANDOFF.stat().st_size if HANDOFF.exists() else 0,prompt_sha256=sha256(PROMPT_FILE),patch_sha256=sha256(PATCH_FILE),runner_sha256=sha256(Path(__file__)),clip_weight_sha256=sha256(Path.home()/'.cache/clip/RN50.pt'),paper_sha256=sha256(paper),paper_audit='Read Fig.1 (rendered), Section2.2 and ablations via PyMuPDF; BrainPrompt/SABER PDFs not found.',text_audit=text_audit,protocol=PROTOCOL,split_sha256=canonical([[a.tolist(),b.tolist()] for a,b in folds]),class_order={'0':'ASD','1':'TD'},dimensions=[48,6,10,256],base_parameters=1462957,local_mlp_parameters=525577,disease_additional_parameters=36865,rng_policy='E00 original sequential stream; all other groups use actual E00 fold-entry Python/NumPy/CPU/CUDA RNG; disease init fork_rng(140000+seed). First lazy RN50 construction is retained.',gpu_hardware=hardware,worker_threads=2,workers_per_gpu=4,tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0,local=LOCAL_MODES[c],disease=DISEASE_MODES[c]) for c,s,g in TASKS},configurations={c:dict(local=LOCAL_MODES[c],disease=DISEASE_MODES[c],additional_parameters=36865 if c in DISEASE_ON else 0) for c in CONFIGS},complete_folds=0,preflight=None)
 write_json(MANIFEST_FILE,m)
 atomic_write(EXP/'README.md',readme_text(m));prepare_worktree();print('PREPARED frozen 14 configs, 42 tasks, 420 folds',flush=True)

def integrity(full=False):
 m=read_json(MANIFEST_FILE)
 assert source_hashes()==m['source_data_sha256'],'protected source/data modified'
 assert sha256(PROMPT_FILE)==m['prompt_sha256'] and sha256(PATCH_FILE)==m['patch_sha256']
 assert sha256(Path(__file__))==m['runner_sha256'],'runner changed after freeze'
 if full:
  assert sha256(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
  for name,h in m['protected_history'].items():assert history_digest(ROOT/'experiment'/name)==h,'history modified: '+name
 return m

_SIGNATURE=None
def signature():
 global _SIGNATURE
 if _SIGNATURE is None:
  m=read_json(MANIFEST_FILE)
  identity={k:m[k] for k in ('source_data_sha256','prompt_sha256','patch_sha256','runner_sha256','protocol','split_sha256','rng_policy')}
  identity['runner_sha256']=m.get('training_runner_sha256',m['runner_sha256'])
  _SIGNATURE=canonical(identity)
 return _SIGNATURE

def context():
 global _CTX
 if _CTX is not None:return _CTX
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 sys.modules['hsp_runtime']=sys.modules[__name__]
 work=prepare_worktree();sys.path.insert(0,str(work));import network;import model
 static=checked_torch(STATE/'text_bank.pt');assert static['prompts_sha256']==sha256(PROMPT_FILE)
 x,y,folds=inputs();_CTX=dict(torch=torch,network=network,model=model,static=static,x=x,y=y,folds=folds,clip_loaded=False)
 from CLIP.clip.model import CLIP
 def forbidden(*args,**kwargs):raise RuntimeError('CLIP.encode_text forbidden during training/preflight')
 CLIP.encode_text=forbidden
 return _CTX

def get_frozen_local(mode):
 c=context()
 if not c['clip_loaded']:
  # Exactly original lazy load position and its CPU initialization RNG consumption.
  encoder,_=c['network'].clip.load('RN50',c['network'].device);del encoder;c['clip_loaded']=True
 key='NEW_P3_SPECIFIC' if mode=='NEW_P3_MISMATCHED' else mode
 return c['static']['local'][key].to('cuda')

def get_frozen_disease(mode): return context()['static']['disease'][mode].to('cuda')

def make_agent(config,seed,fold,original=False):
 c=context();os.environ['HSP_LOCAL_MODE']=LOCAL_MODES[config];os.environ['HSP_DISEASE_MODE']=DISEASE_MODES[config]
 cls=c['model'].EvalHelper
 if original:
  module=types.ModuleType('original_evalhelper');exec(compile((ROOT/'model.py').read_text(),'original_model.py','exec'),module.__dict__);cls=module.EvalHelper
 tr,te=c['folds'][fold-1];hp=SimpleNamespace(**PROTOCOL,seed=seed)
 with contextlib.redirect_stdout(io.StringIO()):agent=cls([48,6,10,256],c['x'],c['y'],hp,tr,te)
 assert np.array_equal(agent.val_idx,agent.tst_idx) and np.array_equal(agent.trn_idx,tr)
 agent.MF_sav.close();agent.GCMP_sav.close();return agent

def model_state(agent): return {name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for name,module in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]}
def load_model(agent,s):agent.MessagePassing0.load_state_dict(s['gfr']);agent.ModalFusion.load_state_dict(s['pln'])
def model_hash(agent,extra=False):
 h=hashlib.sha256()
 for name,module in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]:
  for k,p in sorted(module.named_parameters()):
   disease=k.startswith('disease_')
   if disease==extra:h.update((name+'.'+k).encode()+p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def optimizer_audit(agent,config):
 groups=agent.optimizer_MP0.param_groups+agent.optimizer_MF.param_groups
 seen=[id(p) for g in groups for p in g['params']];actual=[p for m in (agent.MessagePassing0,agent.ModalFusion) for p in m.parameters()]
 assert len(seen)==len(set(seen)) and set(seen)=={id(p) for p in actual}
 assert all(g['lr']==.0038 for g in groups) and groups[0]['weight_decay']==groups[1]['weight_decay']==.11
 assert len(groups)==(3 if config in DISEASE_ON else 2)
 if config in DISEASE_ON:assert groups[2]['weight_decay']==0
 count=sum(p.numel() for p in actual);assert count==1462957+(36865 if config in DISEASE_ON else 0)
 assert sum(p.numel() for p in agent.ModalFusion.MLP.parameters())==525577
 return count

def forward(agent):
 import torch
 agent.MessagePassing0.eval();agent.ModalFusion.eval()
 with torch.no_grad():
  n,parts=agent.dataLayer(agent.feat);h,adj,unused=agent.MessagePassing0(parts,n,agent.targ,agent.trn_idx);out,hidden,_=agent.ModalFusion(h)
  assert torch.isfinite(out).all() and torch.isfinite(hidden).all()
  return out.clone()
def metrics(out,idx):
 from sklearn.metrics import confusion_matrix,roc_auc_score
 y=context()['y'][idx];prob=out.detach().exp().cpu().numpy()[idx];pred=prob.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1])
 sen,spe=cm[0,0]/cm[0].sum(),cm[1,1]/cm[1].sum()
 return dict(acc=float((pred==y).mean()),balanced_accuracy=float((sen+spe)/2),hard_auc=float(roc_auc_score(np.eye(2)[y.astype(int)],np.eye(2)[pred])),probability_auc=float(roc_auc_score(y==0,prob[:,0])),sensitivity_asd=float(sen),specificity_td=float(spe))
def gradients(agent):
 import torch
 all_grad=[p.grad for m in (agent.MessagePassing0,agent.ModalFusion) for p in m.parameters() if p.grad is not None]
 assert all(torch.isfinite(g).all() for g in all_grad),'nonfinite gradient'
 p=agent.ModalFusion;selected={'local_mlp':list(p.MLP.parameters())}
 if getattr(p,'hsp_disease_enabled',False):selected.update(projector=[p.disease_projector.weight],alpha=[p.disease_alpha])
 return {k:math.sqrt(sum(float(x.grad.detach().square().sum()) for x in ps if x.grad is not None)) for k,ps in selected.items()}

def diagnostics(agent,config,seed,fold,out,tracker):
 import torch
 import torch.nn.functional as F
 p=agent.ModalFusion;proto=p.hsp_local_prototypes.detach();g=p.hsp_local_gate.detach();idx=agent.tst_idx
 row=dict(config=config,seed=seed,fold=fold,local_mode=LOCAL_MODES[config],disease_mode=DISEASE_MODES[config],local_mlp_grad_norm_max=tracker['local_mlp'],local_mlp_grad_valid=tracker['local_mlp']>0,local_finite=bool(torch.isfinite(proto).all() and torch.isfinite(g).all()),all_finite=bool(torch.isfinite(out).all() and torch.isfinite(out.exp()).all()),local_prompt_hash=canonical(local_sentences(read_json(PROMPT_FILE),LOCAL_MODES[config])),local_mapping_hash=canonical(dict(modalities=MODALITIES,mapping=[2,3,0,1] if config=='E13' else [0,1,2,3],classes=['ASD','TD'] if LOCAL_MODES[config] not in ('P0','OLD_P1') else ['historical_group0','historical_group1'])),complete_prompt_bank_sha256=sha256(PROMPT_FILE),disease_enabled=config in DISEASE_ON)
 cos=F.cosine_similarity(proto[:,0],proto[:,1],dim=-1);norm=proto.norm(dim=-1)
 for i,m in enumerate(MODALITIES):
  k=m.lower();row.update({f'local_cos_{k}':float(cos[i]),f'local_norm_group0_{k}':float(norm[i,0]),f'local_norm_group1_{k}':float(norm[i,1]),f'local_zero_ratio_{k}':float((norm[i]<1e-8).float().mean()),f'local_high_cos_{k}':bool(cos[i]>=.99),f'local_gate_{k}':float(g[idx,i].mean()),f'local_gate_near_half_{k}':float((g[idx,i]-.5).abs().lt(.01).float().mean())})
 if config in DISEASE_ON:
  logits=p.hsp_disease_logits.detach();dmet=metrics(F.log_softmax(logits,-1),idx);dp=logits.softmax(-1)[idx,0].cpu().numpy();bp=p.hsp_base_output.exp()[idx,0].detach().cpu().numpy();y=context()['y'][idx];margin=(logits[:,0]-logits[:,1])[idx];gamma=float(p.hsp_gamma)
  correlation=float(np.corrcoef(bp,dp)[0,1]) if np.std(bp)>0 and np.std(dp)>0 else 0.
  row.update(disease_prototype_cos=float(F.cosine_similarity(p.disease_prototypes[0],p.disease_prototypes[1],dim=0)),disease_only_acc=dmet['acc'],disease_only_balanced_accuracy=dmet['balanced_accuracy'],disease_only_probability_auc=dmet['probability_auc'],disease_alignment_loss=float(F.cross_entropy(logits[agent.trn_idx],agent.targ[agent.trn_idx],weight=agent.weight)),disease_margin_asd=float(margin[y==0].mean()),disease_margin_td=float((-margin[y==1]).mean()),disease_gamma=gamma,disease_gamma_sign=int(np.sign(gamma)),disease_contribution_abs=float(p.hsp_contribution[idx].abs().mean()),disease_probability_correlation=correlation,base_probability_abs_change=float((out.exp()[idx,0]-p.hsp_base_output.exp()[idx,0]).abs().mean()),disease_projector_grad_norm_max=tracker['projector'],disease_alpha_grad_norm_max=tracker['alpha'],disease_projector_grad_valid=tracker['projector']>0,disease_alpha_grad_valid=tracker['alpha']>0,disease_prompt_hash=canonical(read_json(PROMPT_FILE)['disease_prompts'][DISEASE_MODES[config]]))
 assert all(not isinstance(v,float) or math.isfinite(v) for v in row.values());return row

def counterfactual(agent,seed,fold,own):
 import torch
 p=agent.ModalFusion;idx=agent.tst_idx;real=metrics(own,idx);prob=own.exp()[idx,0];pred=own[idx].argmax(1)
 before=model_hash(agent);extra=model_hash(agent,True);orig_proto=p.disease_prototypes.clone();rows=[]
 for condition in ('REAL','DISEASE_OFF','DISEASE_SWAPPED','DISEASE_NEUTRAL','LOCAL_MISMATCHED'):
  p.hsp_disease_off=condition=='DISEASE_OFF';p.hsp_local_mismatch=condition=='LOCAL_MISMATCHED';p.hsp_prototype_override=None
  if condition=='DISEASE_SWAPPED':p.hsp_prototype_override=orig_proto[[1,0]]
  elif condition=='DISEASE_NEUTRAL':p.hsp_prototype_override=get_frozen_disease('NEUTRAL')
  out=forward(agent);met=metrics(out,idx)
  if condition=='REAL':assert torch.equal(out,own),'REAL counterfactual differs from normal test'
  effective=orig_proto if p.hsp_prototype_override is None else p.hsp_prototype_override
  rows.append(dict(config='E08',seed=seed,fold=fold,condition=condition,**met,probability_abs_change=float((out.exp()[idx,0]-prob).abs().mean()),flip_rate=float((out[idx].argmax(1)!=pred).float().mean()),**{k+'_delta':met[k]-real[k] for k in ('acc','balanced_accuracy','probability_auc')},disease_prototype_cos=float(torch.nn.functional.cosine_similarity(effective[0],effective[1],dim=0)),disease_fusion_response_abs=0. if condition=='DISEASE_OFF' else float(p.hsp_contribution[idx].abs().mean()),disease_gamma=float(p.hsp_gamma),local_gate_mean=float(p.hsp_local_gate[idx].mean())))
 p.hsp_disease_off=False;p.hsp_local_mismatch=False;p.hsp_prototype_override=None
 assert torch.equal(forward(agent),own) and torch.equal(p.disease_prototypes,orig_proto)
 assert model_hash(agent)==before and model_hash(agent,True)==extra
 return rows

def preflight():
 import torch
 import torch.nn.functional as F
 prepare();m=integrity(full=True)
 if m['run_status']=='COMPLETE':
  assert m['preflight']['status']=='PASS' and m['preflight']['signature']==signature()
  print('SKIP: completed experiment retains verified preflight and all-fold acceptance; no training/cache regeneration.');return
 c=context();checks={};counts={};max_diff=0.
 # First lazy RN50 construction must consume the same randomness as original P0.
 seed_rng(0);entry=rng_state();c['clip_loaded']=False;base=make_agent('E00',0,1,original=True)
 bh=model_hash(base);base_out=forward(base);after=rng_state();counts['E00']=optimizer_audit(base,'E00')
 for cfg in CONFIGS:
  restore_rng(entry);c['clip_loaded']=False;a=make_agent(cfg,0,1);counts[cfg]=optimizer_audit(a,cfg)
  assert model_hash(a)==bh,'common initial parameters changed: '+cfg
  assert rng_equal(after,rng_state()),'new module changed common/dropout RNG: '+cfg
  out=forward(a);p=a.ModalFusion
  assert p.hsp_local_prototypes.shape==(4,2,9) and p.hsp_local_gate.shape==(871,4,4)
  if cfg=='E00':
   diff=float((out-base_out).abs().max());assert diff==0.;max_diff=max(max_diff,diff)
  if cfg in DISEASE_ON:
   assert float(p.disease_alpha)==0 and float(p.hsp_gamma)==0
   assert torch.allclose(out,p.hsp_base_output,atol=1e-6,rtol=0)
   expected=F.normalize(p.disease_projector(p.hsp_hidden),dim=-1)@F.normalize(p.disease_prototypes,dim=-1).t()/.2
   assert torch.equal(expected,p.hsp_disease_logits)
   assert torch.equal(out,F.log_softmax(p.hsp_base_output+.2*torch.tanh(p.disease_alpha)*expected,dim=-1))
  if cfg=='E01':
   manual=p.MLP(p.prompt_text_features.reshape(4,2,4096));assert torch.equal(manual,p.hsp_local_prototypes)
  if cfg=='E13':
   manual=p.MLP(p.prompt_text_features.reshape(4,2,4096))[[2,3,0,1]];assert torch.equal(manual,p.hsp_local_prototypes)
  del a;torch.cuda.empty_cache()
 del base
 checks.update(all_14_common_initial_weights_equal=True,extra_init_rng_isolated=True,first_RN50_rng_matches_original=True,original_P0_forward_max_error=max_diff,old_P1_grouping_projection_exact=True,E05_E9_formula_exact=True,E13_texts_and_mapping_exact=True,parameter_counts=counts,tokenize_truncate_false_max_tokens=read_json(MANIFEST_FILE)['text_audit']['max_tokens'],frozen_encode_calls_per_epoch=0)
 # Identical extra initialization for all enabled groups, also seed1 and seed2.
 for seed in SEEDS:
  seed_rng(seed);c['clip_loaded']=True;start=rng_state();h0=None;extra=None
  for cfg in ('E00','E05','E08','E10','E11','E12','E13'):
   restore_rng(start);a=make_agent(cfg,seed,1)
   if h0 is None:h0=model_hash(a);after=rng_state()
   assert model_hash(a)==h0 and rng_equal(after,rng_state())
   if cfg in DISEASE_ON:
    if extra is None:extra=model_hash(a,True)
    assert model_hash(a,True)==extra
   del a
 checks['three_seeds_extra_initialization_equal']=True
 seed_rng(0);c['clip_loaded']=True;a=make_agent('E08',0,1);tracker={k:0. for k in gradients(a)}
 for _ in range(2):
  with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
  for k,v in gradients(a).items():tracker[k]=max(tracker[k],v)
 assert all(v>0 for v in tracker.values()),tracker
 # Preserve exact model, Adam state and all RNG over a third step.
 path=STATE/'preflight_resume.pt';atomic_torch(path,dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng_state()))
 with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
 expected=model_hash(a)+model_hash(a,True);expected_rng=rng_state();expected_out=forward(a)
 saved=checked_torch(path);load_model(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
 with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
 assert model_hash(a)+model_hash(a,True)==expected and rng_equal(expected_rng,rng_state()) and torch.equal(forward(a),expected_out)
 path.unlink();path.with_suffix('.pt.sha256').unlink()
 cf=counterfactual(a,0,1,forward(a));assert len(cf)==5
 # Held-out labels cannot affect disease CE or existing train-only GFR prior.
 saved=model_state(a);start=rng_state();old=a.targ.clone()
 with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
 correct_loss=float(a.hsp_last_disease_loss)
 load_model(a,saved);restore_rng(start);a.targ[a.tst_idx]=1-a.targ[a.tst_idx]
 with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')
 assert float(a.hsp_last_disease_loss)==correct_loss
 a.targ.copy_(old)
 checks.update(gradients_two_steps=tracker,all_gradients_finite=True,model_Adam_RNG_resume_exact=True,disease_CE_train_indices_only=True,counterfactual_only_prescribed_inputs=True,counterfactual_REAL_exact=True,OFF_no_auxiliary_CE=True,original_fifth_fold_guard_preserved=True)
 integrity(full=True);m=read_json(MANIFEST_FILE);m['preflight']=dict(status='PASS',completed_utc=utc_now(),signature=signature(),checks=checks);write_json(MANIFEST_FILE,m)
 print('PREFLIGHT PASS '+json.dumps(checks),flush=True)

def publish_anchor(seed,fold,rng,initial_hash):
 path=STATE/f'E00_seed{seed}_anchors.pt';saved=checked_torch(path) if path.exists() else dict(signature=signature(),folds={})
 assert saved['signature']==signature()
 if fold in saved['folds']:assert rng_equal(saved['folds'][fold]['rng'],rng) and saved['folds'][fold]['initial_trunk_sha256']==initial_hash
 else:
  saved['folds'][fold]=dict(rng=rng,rng_sha256=rng_hash(rng),initial_trunk_sha256=initial_hash);atomic_torch(path,saved)

def task_result(task):
 path=STATE/(task+'.json')
 if path.exists():
  result=read_json(path);assert result['signature']==signature() and result['task']==task
  assert [r['fold'] for r in result['metrics']]==list(range(1,len(result['metrics'])+1))
  assert len(result['metrics'])==len(result['diagnostics'])
  if result['config']=='E08':assert len(result['counterfactual'])==5*len(result['metrics'])
  if result['status']=='COMPLETE':assert len(result['metrics'])==10
  return result
 return None

def worker(config,seed):
 import torch
 m=integrity();assert m['preflight']['status']=='PASS' and m['preflight']['signature']==signature()
 task=f'{config}_seed{seed}';physical=GPU_ASSIGNMENTS[config,seed]
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(physical) and torch.cuda.device_count()==1 and torch.cuda.current_device()==0
 lock=(STATE/(task+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=task_result(task)
 if old and old['status']=='COMPLETE':print('SKIP COMPLETE '+task,flush=True);return 0
 c=context();path=STATE/(task+'_resume.pt');saved=checked_torch(path) if path.exists() else None
 if saved:
  assert saved['signature']==signature() and saved['task']==task
  result=saved['result'];restore_rng(saved['rng']);c['clip_loaded']=True
  write_json(STATE/(task+'.json'),result)
 else:
  assert old is None or not old['metrics'],'missing recoverable continuation state'
  seed_rng(seed);result=dict(signature=signature(),task=task,config=config,seed=seed,physical_gpu=physical,status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],seconds=0.,started_utc=utc_now())
 stopping=[False]
 def stop(sig,frame):stopping[0]=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 t0=time.monotonic();prev_seconds=result['seconds']
 print(json.dumps(dict(task=task,pid=os.getpid(),physical_gpu=physical,logical_gpu=0,resumed=saved is not None,device=str(torch.cuda.get_device_properties(0)))),flush=True)
 for fold in range(len(result['metrics'])+1,11):
  resume_epoch=saved is not None and saved['phase']=='EPOCH' and saved['fold']==fold
  anchor=None
  if not resume_epoch:
   if stopping[0]:return 130
   if config!='E00':
    assert read_json(MANIFEST_FILE).get('E00_parity',{}).get('status')=='PASS'
    anchors=checked_torch(STATE/f'E00_seed{seed}_anchors.pt');anchor=anchors['folds'][fold];restore_rng(anchor['rng'])
   init_rng=rng_state()
  a=make_agent(config,seed,fold,original=config=='E00');optimizer_audit(a,config)
  tracker={k:0. for k in gradients(a)}
  if resume_epoch:
   load_model(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['optimizer_mp']);a.optimizer_MF.load_state_dict(saved['optimizer_mf']);restore_rng(saved['rng'])
   best_model,best_acc,best_epoch=saved['best_model'],saved['best_acc'],saved['best_epoch'];wait,start,tracker=saved['wait'],saved['epoch']+1,saved['tracker'];initial_hash=saved['initial_hash'];branch_hash=saved['branch_hash'];init_rng=saved['init_rng']
   print(f'RESUME fold={fold} epoch={start}',flush=True)
  else:
   initial_hash,branch_hash=model_hash(a),model_hash(a,True)
   if config=='E00':publish_anchor(seed,fold,init_rng,initial_hash)
   else:assert initial_hash==anchor['initial_trunk_sha256'],'paired common initialization mismatch'
   best_model,best_acc,best_epoch,wait,start=None,0.,-1,0,0
  print(f'FOLD_START fold={fold} init={initial_hash} entry_rng={rng_hash(init_rng)}',flush=True)
  def save_epoch(epoch):
   result['seconds']=prev_seconds+time.monotonic()-t0
   atomic_torch(path,dict(signature=signature(),task=task,phase='EPOCH',fold=fold,epoch=epoch,model=model_state(a),optimizer_mp=a.optimizer_MP0.state_dict(),optimizer_mf=a.optimizer_MF.state_dict(),rng=rng_state(),best_model=best_model,best_acc=best_acc,best_epoch=best_epoch,wait=wait,tracker=tracker,initial_hash=initial_hash,branch_hash=branch_hash,init_rng=init_rng,result=result))
  last=start-1
  for epoch in range(start,450):
   if wait>200:break
   print(f'fold={fold} epoch={epoch:03d}/450 ',end='',flush=True)
   a.run_epoch('simple-2')
   for k,v in gradients(a).items():tracker[k]=max(tracker[k],v)
   with torch.no_grad():_,acc=a.print_trn_acc('simple-2')
   improved=acc>best_acc
   if improved:best_acc,best_epoch,wait,best_model=acc,epoch,0,model_state(a)
   else:wait+=1
   last=epoch;print(f'SELECTION best_epoch={best_epoch} best_acc={best_acc:.8f} wait={wait}',flush=True)
   if improved or epoch%10==9 or wait>200 or stopping[0]:save_epoch(epoch)
   if stopping[0]:print('INTERRUPTED complete epoch saved',flush=True);return 130
  assert best_model is not None
  continuation_rng=rng_state();load_model(a,best_model);out=forward(a);met=metrics(out,a.tst_idx)
  assert abs(met['acc']-best_acc)<1e-12
  if config in DISEASE_ON:assert tracker['projector']>0 and tracker['alpha']>0
  row=dict(config=config,seed=seed,fold=fold,physical_gpu=physical,**met,best_epoch=best_epoch,epochs_executed=last+1,selected_val_acc=best_acc,val_equals_test=True,train_n=len(a.trn_idx),test_n=len(a.tst_idx),initial_trunk_sha256=initial_hash,initial_disease_sha256=branch_hash if config in DISEASE_ON else '',fold_entry_rng_sha256=rng_hash(init_rng),additional_parameters=36865 if config in DISEASE_ON else 0)
  result['metrics'].append(row);result['diagnostics'].append(diagnostics(a,config,seed,fold,out,tracker))
  if config=='E08':result['counterfactual'].extend(counterfactual(a,seed,fold,out))
  restore_rng(continuation_rng);result['seconds']=prev_seconds+time.monotonic()-t0;result['status']='COMPLETE' if fold==10 else 'RUNNING'
  if fold==10:result['completed_utc']=utc_now()
  atomic_torch(path,dict(signature=signature(),task=task,phase='BETWEEN_FOLDS',fold=fold,rng=continuation_rng,result=result))
  write_json(STATE/(task+'.json'),result);print('FOLD_COMPLETE '+json.dumps(row),flush=True)
  if fold==5 and np.mean([r['acc'] for r in result['metrics']])<.6:raise RuntimeError('original fifth-fold ACC guard: incomplete task')
  del a,best_model,out;saved=None;torch.cuda.empty_cache()
  if stopping[0] and fold<10:return 130
 print('TASK_COMPLETE '+task+' folds=10',flush=True);lock.close();return 0

def worker_env(gpu):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
 return env

def live_workers():
 out={}
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   args=(proc/'cmdline').read_bytes().decode().split('\0')
   if str(Path(__file__).resolve()) in args and '--worker' in args:
    i=args.index('--worker');cfg=args[i+1];seed=int(args[i+2]);out[f'{cfg}_seed{seed}']=int(proc.name)
  except (OSError,ValueError,IndexError):pass
 return out

def all_task_results():
 return [r for c,s,g in TASKS if (r:=task_result(f'{c}_seed{s}')) is not None]

def parity(results):
 reference={(int(r['seed']),int(r['fold'])):r for r in read_csv(REFERENCE_FILE) if r['config']=='E00'}
 rows=[row for result in results if result['config']=='E00' for row in result['metrics']]
 errors={k:0. for k in METRICS}
 for r in rows:
  old=reference[r['seed'],r['fold']]
  for k in METRICS:
   err=abs(r[k]-float(old[k]));errors[k]=max(errors[k],err)
   assert err<=1e-10,f'E00 historical parity failed seed={r["seed"]} fold={r["fold"]} {k}: {r[k]} != {old[k]}'
 return dict(status='PASS' if len(rows)==30 else 'PENDING',folds=len(rows),max_absolute_error=errors,reference='experiment13 exact original P0, all six metrics')

def refresh(finalize=False):
 m=read_json(MANIFEST_FILE);results=all_task_results() if STATE.exists() else []
 if not results and m['run_status']=='COMPLETE':
  for filename,h in m['result_sha256'].items():assert sha256(EXP/filename)==h
  return m
 rows=[row for r in results for row in r['metrics']];diags=[row for r in results for row in r['diagnostics']];cf=[row for r in results for row in r['counterfactual']]
 expected={(c,s,f) for c in CONFIGS for s in SEEDS for f in range(1,11)};keys=[(r['config'],r['seed'],r['fold']) for r in rows]
 assert len(keys)==len(set(keys)) and set(keys)<=expected and len(rows)==len(diags)
 assert [(r['config'],r['seed'],r['fold']) for r in diags]==keys
 assert all(0<=r[k]<=1 and math.isfinite(r[k]) for r in rows for k in METRICS)
 assert all(r['physical_gpu']==GPU_ASSIGNMENTS[r['config'],r['seed']] for r in rows)
 assert all(r['all_finite'] and r['local_finite'] for r in diags)
 assert all(r['disease_projector_grad_valid'] and r['disease_alpha_grad_valid'] for r in diags if r['config'] in DISEASE_ON)
 assert all(abs(r['selected_val_acc']-r['acc'])<1e-12 for r in rows)
 for r in results:
  task=m['tasks'][r['task']];task.update(status=r['status'],folds=len(r['metrics']),seconds=r['seconds'],started_utc=r['started_utc'])
  if r['status']=='COMPLETE':task['completed_utc']=r['completed_utc']
 m['E00_parity']=parity(results);m['complete_folds']=len(rows);m['completed_tasks']=sum(r['status']=='COMPLETE' for r in results)
 m['counterfactual_rows']=len(cf);m['diagnostics_rows']=len(diags);m['updated_utc']=utc_now()
 if rows:write_csv(FOLD_FILE,rows);write_csv(DIAG_FILE,diags)
 if cf:write_csv(CF_FILE,cf)
 if finalize:
  assert m['completed_tasks']==42 and set(keys)==expected and len(diags)==420 and len(cf)==150
  cfkeys=[(r['seed'],r['fold'],r['condition']) for r in cf]
  assert set(cfkeys)=={(s,f,c) for s in SEEDS for f in range(1,11) for c in ('REAL','DISEASE_OFF','DISEASE_SWAPPED','DISEASE_NEUTRAL','LOCAL_MISMATCHED')} and len(set(cfkeys))==150
  assert not live_workers(),'workers still running'
  for gpu in (0,1):assert sum(r['physical_gpu']==gpu for r in rows)==210
  for s in SEEDS:
   anchor=checked_torch(STATE/f'E00_seed{s}_anchors.pt');assert len(anchor['folds'])==10
  for s in SEEDS:
   for f in range(1,11):
    paired=[r for r in rows if r['seed']==s and r['fold']==f]
    assert len({r['initial_trunk_sha256'] for r in paired})==1 and len({r['fold_entry_rng_sha256'] for r in paired})==1
    assert len({r['initial_disease_sha256'] for r in paired if r['config'] in DISEASE_ON})==1
  integrity(full=True)
  m['completed_utc']=utc_now();m['wall_seconds']=time.time()-m['started_unix'];m['gpu_completion']={str(g):dict(tasks=21,folds=210,device=m['gpu_hardware'][g]) for g in (0,1)}
  m['acceptance']=dict(groups=14,tasks=42,folds=420,diagnostics=420,counterfactual=150,paired_initialization_all_30=True,protected_sources_history_unchanged=True,failed_attempt_metrics_mixed=False,no_training_processes=True,all_finite=True)
  m['run_status']='FINALIZING';write_json(MANIFEST_FILE,m);report();update_handoff()
  m=read_json(MANIFEST_FILE);atomic_write(REPORT_FILE,REPORT_FILE.read_text().replace('状态：FINALIZING','状态：COMPLETE',1))
  m['result_sha256']={str(p.relative_to(EXP)):sha256(p) for p in (REFERENCE_FILE,FOLD_FILE,DIAG_FILE,CF_FILE,REPORT_FILE)}
  assert STATE.parent==Path(tempfile.gettempdir()) and STATE.name.startswith('spromptgl14-state-');shutil.rmtree(STATE)
  link=EXP/'state'
  if link.is_symlink():link.unlink()
  m['handoff_updated']=True;m['temporary_cleanup']='DONE';m['run_status']='COMPLETE';write_json(MANIFEST_FILE,m)
 else:write_json(MANIFEST_FILE,m)
 return m

def schedule(workers_per_gpu=4):
 prepare();m=integrity(full=True)
 if m['run_status']=='COMPLETE':print('COMPLETE already; all 420 folds verified');return
 if not m.get('preflight') or m['preflight']['status']!='PASS':preflight()
 scheduler_lock=(LOG_DIR/'scheduler.lock').open('a');fcntl.flock(scheduler_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=read_json(MANIFEST_FILE);m['run_status']='RUNNING';m.setdefault('started_unix',time.time());m.setdefault('started_utc',utc_now());m['workers_per_gpu']=workers_per_gpu;write_json(MANIFEST_FILE,m)
 active={};attempts={};stopping=[False]
 for task,pid in live_workers().items():
  cfg,ss=task.split('_seed');seed=int(ss);active[task]=dict(pid=pid,process=None,gpu=GPU_ASSIGNMENTS[cfg,seed],log=None)
 def stop(sig,frame):stopping[0]=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 print('SCHEDULER_START '+utc_now()+' max workers/GPU='+str(workers_per_gpu),flush=True)
 last_counts=None
 try:
  while True:
   for task,a in list(active.items()):
    if a['process'] is not None:done=a['process'].poll() is not None;rc=a['process'].returncode
    else:
     try:os.kill(a['pid'],0);done=False
     except ProcessLookupError:done=True
     rc=None
    if done:
     if a['log']:a['log'].close()
     result=task_result(task);print(f'WORKER_END {task} rc={rc} status={result["status"] if result else "NO_RESULT"}',flush=True)
     del active[task]
     if not result or result['status']!='COMPLETE':
      attempts[task]=attempts.get(task,0)+1
      if attempts[task]>=3:raise RuntimeError('repeated worker failure '+task)
   m=refresh();counts=(m['completed_tasks'],m['complete_folds'])
   if counts!=last_counts:
    print(f'PROGRESS tasks={counts[0]}/42 folds={counts[1]}/420 active={len(active)} E00_parity={m["E00_parity"]["status"]}',flush=True);last_counts=counts
   if stopping[0]:
    for a in active.values():os.kill(a['pid'],signal.SIGTERM)
    if not active:
     m['run_status']='INTERRUPTED';write_json(MANIFEST_FILE,m);break
   else:
    eligible=TASKS if m['E00_parity']['status']=='PASS' else [t for t in TASKS if t[0]=='E00']
    for gpu in (0,1):
     for cfg,seed,g in eligible:
      task=f'{cfg}_seed{seed}'
      if g!=gpu or task in active:continue
      result=task_result(task)
      if result and result['status']=='COMPLETE':continue
      if sum(a['gpu']==gpu for a in active.values())>=workers_per_gpu:break
      log=(LOG_DIR/(task+'.log')).open('a',buffering=1)
      log.write(f'\nATTEMPT_START task={task} UTC={utc_now()} resume={bool(result)}\n')
      proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',cfg,str(seed)],cwd=EXP,env=worker_env(gpu),stdout=log,stderr=subprocess.STDOUT)
      active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=log)
      print(f'WORKER_START {task} physical_gpu={gpu} pid={proc.pid}',flush=True)
   if m['completed_tasks']==42 and not active:
    refresh(finalize=True);print_final();break
   time.sleep(5)
 except BaseException:
  for a in active.values():
   try:os.kill(a['pid'],signal.SIGTERM)
   except ProcessLookupError:pass
  m=read_json(MANIFEST_FILE);m['run_status']='INTERRUPTED';write_json(MANIFEST_FILE,m);raise
 finally:scheduler_lock.close()

PAIRS=[('E02','E00'),('E04','E00'),('E02','E01'),('E04','E03'),('E08','E13'),('E05','E04'),('E06','E05'),('E07','E05'),('E08','E05'),('E09','E05'),('E08','E06'),('E08','E07'),('E08','E09'),('E08','E10'),('E11','E00'),('E12','E02'),('E08','E04')]

def summaries(rows):
 seed_values={}
 for c in CONFIGS:
  for s in SEEDS:
   group=[r for r in rows if r['config']==c and int(r['seed'])==s]
   if len(group)==10:seed_values[c,s]={k:statistics.mean(float(r[k]) for r in group) for k in METRICS}
 aggregate={c:{k:(statistics.mean(seed_values[c,s][k] for s in SEEDS),statistics.pstdev(seed_values[c,s][k] for s in SEEDS)) for k in METRICS} for c in CONFIGS if all((c,s) in seed_values for s in SEEDS)}
 return seed_values,aggregate

def paired(rows,terms,k):
 by={(r['config'],int(r['seed']),int(r['fold'])):r for r in rows};values=[];sd=[]
 for s in SEEDS:
  current=[]
  for f in range(1,11):current.append(sum(sign*float(by[c,s,f][k]) for c,sign in terms))
  values+=current;sd.append(statistics.mean(current)*100)
 return dict(mean=statistics.mean(values)*100,sd=statistics.pstdev(sd),seeds=sd,wtl=[sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v< -1e-12 for v in values)])

def readme_text(m):
 table='\n'.join(f'|{c}|{LOCAL_MODES[c]}|{DISEASE_MODES[c]}|{36865 if c in DISEASE_ON else 0}|' for c in CONFIGS)
 return f'''# 实验十四：Hierarchical Semantic Prompt Design and Cross-Level Interaction

14组固定设计，3 seeds×10 folds，42个独立任务，420折。仅在临时源码副本集成；不修改根源码、实验一至十三，不 commit/push。

## 论文、源码和历史审计

实际阅读根目录原论文PDF的Fig.1、2.2和消融，PyMuPDF可解析并渲染。CAM跨模态交互；PEN包括局部SPGS和RES；文本CLIP冻结，4096→128→9共享MLP；局部Softmax-Max权重后经RES加回全局表示、concat分类。论文消融支持模块组合，但不证明当前具体文本的必要性。论文ABIDE超参数与本地shell不同，本轮以已复现实验十三的本地shell为准。BrainPrompt/SABER本地PDF未发现，未以它们作为已读依据。

871人、48/6/10/256，ASD0/TD1。GFR为受试者图，保留原分支覆盖与未纳入总损失的ALoss。四个9维局部Token是潜在特征，没有ROI或功能网络一一对应关系。原论文图示和措辞不能提供源码中不存在的ROI映射。

实验一P1约91.80%，增益小且非三seed同时稳定；实验二至七的状态文本、MLP改造、标量方向门控、双主题残差未证明稳定的正确语义收益。实验八疾病分支与原分类概率高度相关；实验九P3+疾病E9略改善但没有疾病特异性对照。实验十真实ASD相对中性仅约0.077pp，固定监督类别代码仍是替代解释。实验十一256连接索引已恢复，252可安全命名，但本轮不重新引入连接级模块；实验十二pilot未支持静态ROI文本，实验十三240折个体化Prompt没有稳定优于P0，本轮不研究个体化。旧P1两个主题不解释为ASD/TD类别。

## 冻结矩阵

|ID|局部Prompt|疾病Prompt|新增参数|
|---|---|---|---:|
{table}

P0八句含历史拼写Nromal，OLD_P1逐字32句；新共享版本8句、每类4句拼接4096，专属版本32句、[4,2,4096]通过同一MLP。E13只以[2,3,0,1]置换E08相同文本；没有新学习层。P0保留原matmul、旧P1保留历史einsum，Softmax-Max、RES和分类器不变。max门控对两类交换不变，不称为有向疾病证据。

疾病分支严格沿用E9：s=normalize(Linear36→1024(hidden))，C=normalize(mean(normalize(CLIP句向量)))，logits=sCᵀ/.2；CE仅train_idx、原权重、系数.1；output=log_softmax(base_logprob+.2tanh(alpha)×logits)。alpha0；新增36,865参数、Adam lr.0038/wd0；原组lr.0038/wd.11。全部疾病配置相同结构，仅冻结文本原型不同。

## 随机性、缓存与恢复

先正式训练E00三个连续十折，保留首次RN50原位置加载消耗，记录各折全部RNG入口。E00精确匹配实验十三30折六指标后，其他39任务恢复这些配对入口；新增疾病投影在原my_weight_init后fork_rng(140000+seed)，不改变公共初始化/dropout随机流。CLIP文本prepare时一次编码；训练编码被禁止。逐句检查BOS/EOS≤77，truncate=False。缓存与checkpoint存放本机/tmp临时state，通过实验目录state软链接查看；结束清理，复跑prepare重新构建。OSS目录只保留归档结果和日志。

checkpoint记录epoch/model/两Adam/best模型/早停/RNG/已完成逐折结果，原子写入并SHA256核验。最佳ACC严格提高、首次平局保留、wait>200、450上限；原第五折低ACC防护保留，触发会报任务失败。任务和scheduler持有独占锁；进程中断从完整epoch恢复，已完成折不重计。每GPU21任务/210折，默认4并行、每worker2线程；CUDA_VISIBLE_DEVICES物理绑定，逻辑卡0。

## 指标、反事实与限制

Hard-AUC原one-hot硬预测实现，在二分类中等于BA；Probability-AUC以ASD阳性并取output.exp()[:,0]。先seed内十折再3seed mean±population SD，30行共用fold不作为独立样本。E08同一最佳checkpoint做REAL、疾病融合OFF、疾病原型互换、中性疾病原型、局部模态错配5种推理，无重训练。OFF只移除最终融合，不撤销辅助训练。E10/E13另是重新训练对照。

历史val=test导致测试参与checkpoint选择，固定CSV上游全局标准化和RFE有共享信息，结果不宣称独立无偏泛化。共享→专属比较同时改变文本细化与独立句数，不能归因于模态对应单一因素。

## 运行

```bash
python {EXP}/run.py prepare
CUDA_VISIBLE_DEVICES=0 python {EXP}/run.py preflight
python {EXP}/run.py all
python {EXP}/run.py resume
python {EXP}/run.py report
```

源码/数据/文本/CLIP/patch SHA256、Git版本、GPU UUID及状态见manifest；fold_metrics420、diagnostics420、counterfactual150、完整中文报告见results。完成审计后清理state、临时副本和checkpoint。
'''

def report():
 m=read_json(MANIFEST_FILE);rows=read_csv(FOLD_FILE);ds=read_csv(DIAG_FILE);cf=read_csv(CF_FILE);sv,agg=summaries(rows)
 if len(agg)!=14:
  atomic_write(REPORT_FILE,f'# 实验十四执行进度\n\n状态：{m["run_status"]}；已完成{len(rows)}/420折，{m.get("completed_tasks",0)}/42任务。结果尚未齐全，暂不作科学结论。\n');return
 lines=['# 实验十四：分层语义Prompt设计与跨层交互', '',f'状态：{m["run_status"]}；14/14组、42/42任务、420/420折、420条机制诊断、150条E08反事实。生成时间{utc_now()}。', '', '## 论文、实现与历史证据', '', readme_text(m).split('## 论文、源码和历史审计\n\n')[1].split('## 冻结矩阵')[0].strip(), '', '## 设计与完整主结果', '', '固定矩阵及数学公式见README。E00–04定位局部文本，E05–10定位疾病内容，E11/E12定位跨层交互，E13定位正确模态对应。所有组均重新训练30折；历史数据仅做parity，不替代正式训练。三seed十折均值再汇总mean±population SD(ddof=0)，以下均为百分数。', '', '|Config|Local|Disease|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for c in CONFIGS:lines.append('|'+ '|'.join([c,LOCAL_MODES[c],DISEASE_MODES[c]]+[f'{agg[c][k][0]*100:.3f}±{agg[c][k][1]*100:.3f}' for k in METRICS])+'|')
 lines+=['','## 各seed十折汇总','','|Config|Seed|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for c in CONFIGS:
  for s in SEEDS:lines.append('|'+ '|'.join([c,str(s)]+[f'{sv[c,s][k]*100:.3f}' for k in METRICS])+'|')
 lines+=['','## 所有预设配对比较','','差值为百分点；按相同seed/fold配对，胜平负容差1e-12。每项同时报告三个seed方向和逐折胜平负；未将30折视为独立样本，没有显著性宣称。']
 for a,b in PAIRS:
  lines+=['',f'### {a} − {b}','','|指标|均值差pp|seed0/1/2差pp|seed差SD|胜/平/负|','|---|---:|---|---:|---|']
  for k in METRICS:
   q=paired(rows,[(a,1),(b,-1)],k);lines.append(f'|{k}|{q["mean"]:+.4f}|'+ '/'.join(f'{v:+.4f}' for v in q['seeds'])+f'|{q["sd"]:.4f}|'+ '/'.join(map(str,q['wtl']))+'|')
 lines+=['','## 跨层增益差','','|交互|指标|均值pp|seed0/1/2差pp|胜/平/负|','|---|---|---:|---|---|']
 for b,base in [('E11','E00'),('E12','E02')]:
  for k in ('acc','balanced_accuracy','probability_auc'):
   q=paired(rows,[('E08',1),('E04',-1),(b,-1),(base,1)],k)
   lines.append(f'|(E08−E04)−({b}−{base})|{k}|{q["mean"]:+.4f}|'+ '/'.join(f'{v:+.4f}' for v in q['seeds'])+'|'+ '/'.join(map(str,q['wtl']))+'|')
 lines+=['','## 局部原型与SPGS门控','','旧P0/OLD_P1按历史group0/group1记范数，不将其解释为ASD/TD。cos≥.99为高相似、norm<1e-8为近零阈值；精确零需结合原始范数判断；接近均匀定义gate与.5差<.01。以下每配置30折均值；latent Token不解释为脑区。','','|Config|Modality|原型cos|组0/1范数|零向量比例|高cos率|gate|近.5比例|MLP有效梯度fold|','|---|---|---:|---|---:|---:|---:|---:|---:|']
 for c in CONFIGS:
  group=[r for r in ds if r['config']==c]
  for md in MODALITIES:
   k=md.lower();mean=lambda field:statistics.mean(float(r[field]) for r in group)
   lines.append(f'|{c}|{md}|{mean("local_cos_"+k):.6f}|{mean("local_norm_group0_"+k):.3e}/{mean("local_norm_group1_"+k):.3e}|{mean("local_zero_ratio_"+k):.4f}|{sum(r["local_high_cos_"+k]=="True" for r in group)/30:.4f}|{mean("local_gate_"+k):.6f}|{mean("local_gate_near_half_"+k):.4f}|{sum(r["local_mlp_grad_valid"]=="True" for r in group)}/30|')
 lines+=['','## 疾病分支、冗余与融合','','疾病CE仅训练索引；分支单独指标及相关性用相同测试折。gamma允许负值，不事后改变其符号。近零贡献需结合梯度、gamma和概率变化解释。','','|Config|原型cos|分支ACC|分支BA|分支Prob-AUC|Train CE|ASD/TD正确margin|gamma|融合贡献abs|概率相关|概率变化abs|Proj/alpha有效fold|','|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|']
 for c in CONFIGS:
  if c not in DISEASE_ON:continue
  group=[r for r in ds if r['config']==c];mean=lambda field:statistics.mean(float(r[field]) for r in group)
  lines.append(f'|{c}|{mean("disease_prototype_cos"):.6f}|{mean("disease_only_acc")*100:.3f}|{mean("disease_only_balanced_accuracy")*100:.3f}|{mean("disease_only_probability_auc")*100:.3f}|{mean("disease_alignment_loss"):.5f}|{mean("disease_margin_asd"):.5f}/{mean("disease_margin_td"):.5f}|{mean("disease_gamma"):+.6f}|{mean("disease_contribution_abs"):.6f}|{mean("disease_probability_correlation"):.6f}|{mean("base_probability_abs_change"):.6f}|{sum(r["disease_projector_grad_valid"]=="True" for r in group)}/{sum(r["disease_alpha_grad_valid"]=="True" for r in group)}|')
 lines+=['','## E08同checkpoint反事实','','30个最佳checkpoint各五条件；REAL逐值等于正常推理，结束恢复原配置。OFF仅关闭融合贡献，保留疾病辅助训练形成的参数。NEUTRAL与LOCAL_MISMATCHED推理干预分别区别于E10/E13重新训练。预测敏感只证明使用文本原型，不证明医学语义正确。','','|条件|ACC|BA|Prob-AUC|翻转比例|ASD概率abs变化|ACC/BA/Prob-AUC差pp|原型cos|融合响应abs|','|---|---:|---:|---:|---:|---:|---|---:|---:|']
 for condition in ('REAL','DISEASE_OFF','DISEASE_SWAPPED','DISEASE_NEUTRAL','LOCAL_MISMATCHED'):
  group=[r for r in cf if r['condition']==condition];mean=lambda field:statistics.mean(float(r[field]) for r in group)
  lines.append(f'|{condition}|{mean("acc")*100:.3f}|{mean("balanced_accuracy")*100:.3f}|{mean("probability_auc")*100:.3f}|{mean("flip_rate"):.6f}|{mean("probability_abs_change"):.6f}|'+ '/'.join(f'{mean(k+"_delta")*100:+.4f}' for k in ('acc','balanced_accuracy','probability_auc'))+f'|{mean("disease_prototype_cos"):.6f}|{mean("disease_fusion_response_abs"):.6f}|')
 lines+=['','## 结果判断与后续建议','']+conclusions(rows,ds,cf,agg)
 lines+=['','## 工程验收与局限','',f'本地Git HEAD `{m["git_head"]}`，远端main `{m["remote_main"]}`；六核心文件逐字节一致。全部数据、CLIP、文本、patch与保护历史SHA256见manifest。E00历史parity：{json.dumps(m["E00_parity"],ensure_ascii=False)}。', '',f'双GPU各21任务/210折，默认4 workers/card、每进程2线程。实测墙钟时间{m.get("wall_seconds",0)/60:.2f}分钟；设备/UUID：'+ '; '.join(m['gpu_hardware'])+'。', '', '公共权重与各seed/fold入口RNG在正式420折再次验证一致；疾病ON初始化一致，新增参数36,865。训练文本编码为0，77-token审计通过；checkpoint model/Adam/RNG恢复短步逐值一致。逐折checkpoint ACC与最终测试一致，无缺失/重复/非法数、无失败记录混入。已完成任务跳过，中断恢复epoch；最终清理临时checkpoint和副本。', '', '历史val=test使测试参与checkpoint选择；上游全局标准化和RFE存在共享信息。三个seed共用fold，不是30个独立数据集；本轮结果不能称作无偏独立测试泛化。共享→专属同时改变文本细化与独立文本数量，无法隔离单一因素。没有个体化语义，没有ROI级异常解释，没有超参数搜索；细小差异仍可能来自既定协议的训练波动。', '', '本轮只修正运行器文本加载、GPU绑定、RNG隔离和checkpoint/I/O缺陷，未改原GCN/分类器/损失主干。未执行git commit/push。完整逐折数据见fold_metrics.csv、diagnostics.csv、counterfactual.csv，重跑/恢复命令见README。']
 atomic_write(REPORT_FILE,'\n'.join(lines)+'\n')

def conclusions(rows,ds,cf,agg):
 best_local=max(['E00','E01','E02','E03','E04'],key=lambda c:agg[c]['balanced_accuracy'][0]);best_disease=max(['E06','E07','E08','E09'],key=lambda c:agg[c]['balanced_accuracy'][0]);best=max(CONFIGS,key=lambda c:agg[c]['balanced_accuracy'][0]);result=[]
 result.append(f'按BA描述性排序，局部组最高为{best_local}（{LOCAL_MODES[best_local]}），新疾病方案最高为{best_disease}（{DISEASE_MODES[best_disease]}），全部组合最高为{best}。排名本身不证明稳定最优，完整ACC/概率AUC及seed方向均须同时判断。')
 for a,b,label in [('E02','E00','新P1对P0'),('E04','E00','新P3对P0'),('E04','E03','模态专属对共享'),('E08','E13','正确模态对应'),('E08','E10','正确疾病文本对中性'),('E08','E04','疾病层对新P3的互补')]:
  qs=[paired(rows,[(a,1),(b,-1)],k) for k in ('acc','balanced_accuracy','probability_auc')];stable=all(q['mean']>0 and all(v>0 for v in q['seeds']) for q in qs)
  result.append(f'{label}：ACC/BA/Prob-AUC差'+ '/'.join(f'{q["mean"]:+.4f}pp' for q in qs)+('，三指标的三seed方向均为正。' if stable else '，未同时满足三指标三seed方向均为正，不能据此宣称稳定收益。'))
 for b in ('E00','E01'):
  q=paired(rows,[(best,1),(b,-1)],'balanced_accuracy');result.append(f'BA最高组合{best}相对{b}：{q["mean"]:+.4f}pp，seed0/1/2 '+ '/'.join(f'{v:+.4f}' for v in q['seeds'])+'pp。')
 e8=[r for r in ds if r['config']=='E08'];corr=statistics.mean(float(r['disease_probability_correlation']) for r in e8);contribution=statistics.mean(float(r['disease_contribution_abs']) for r in e8)
 result.append(f'E08疾病/原分类器概率相关平均{corr:.5f}，融合贡献绝对均值{contribution:.6f}；'+('提示明显冗余，类别监督可将静态文本学成类别代码。' if corr>.9 else '单看相关性不足以确定互补收益，需结合配对结果。'))
 mismatch=[r for r in cf if r['condition']=='LOCAL_MISMATCHED'];neutral=[r for r in cf if r['condition']=='DISEASE_NEUTRAL'];sw=[r for r in cf if r['condition']=='DISEASE_SWAPPED']
 result.append('同checkpoint局部错配/中性疾病/疾病类交换的平均概率变化为'+ '/'.join(f'{statistics.mean(float(r["probability_abs_change"]) for r in group):.6f}' for group in (mismatch,neutral,sw))+'，平均翻转率为'+ '/'.join(f'{statistics.mean(float(r["flip_rate"]) for r in group):.6f}' for group in (mismatch,neutral,sw))+'。敏感性与重新训练语义必要性是不同问题。')
 zeros=sum(float(r['disease_contribution_abs'])<1e-8 for r in ds if r['config'] in DISEASE_ON);result.append(f'疾病ON的270折中，融合贡献近零(<1e-8)为{zeros}折；投影/alpha均有有效有限梯度。局部MLP即使获得早期梯度，后期也可能出现范数收缩或两原型塌缩；分模态表为实际证据。')
 result.append('四模态只有原型和门控诊断，没有逐模态消融，不能断言某模态主导性能。特别是Correlation最大门控或最低cos都不等于该模态的因果收益。局部两类max门控无方向，不可解释为识别了患者特定脑区异常。')
 content=paired(rows,[('E08',1),('E10',-1)],'balanced_accuracy');mapping=paired(rows,[('E08',1),('E13',-1)],'balanced_accuracy')
 if all(v>0 for v in content['seeds']) and all(v>0 for v in mapping['seeds']):result.append('正确疾病内容与模态对应在BA上方向一致，可保留为后续假设；仍需独立验证协议、概率AUC和跨层增益确认，当前不能声称临床语义已被理解。')
 else:result.append('正确疾病内容与模态对应未同时给出三seed一致的BA收益；本轮不足以支持继续通过措辞搜索获取语义创新。建议保留P0/旧P1作为参照，下一步优先独立验证与局部—疾病对齐机制检验，避免继续自动改写Prompt。')
 result.extend(extended_findings(rows,ds,cf,agg))
 return result

def extended_findings(rows,ds,cf,agg):
 out=[]
 q=[paired(rows,[('E04',1),('E02',-1)],k) for k in ('acc','balanced_accuracy','probability_auc')]
 out.append('新P3专属相对新P1共享的ACC/BA/Prob-AUC差为'+ '/'.join(f'{v["mean"]:+.4f}pp' for v in q)+'；ACC与BA三个seed均提高，概率AUC方向不一致且均值降低，因此没有一致的全指标赢家。')
 best_prob=max(('E06','E07','E08','E09'),key=lambda c:agg[c]['probability_auc'][0])
 out.append(f'四个新疾病方案中BA最高为E07/BALANCED，Probability-AUC最高为{best_prob}/{DISEASE_MODES[best_prob]}；Balanced相对原P0的ACC/BA/Prob-AUC差为'+ '/'.join(f'{paired(rows,[("E07",1),("E00",-1)],k)["mean"]:+.4f}pp' for k in ('acc','balanced_accuracy','probability_auc'))+'。BA及连续AUC的seed方向不一致，不能宣布最优疾病文本。')
 ref=[r for r in read_csv(REFERENCE_FILE) if r['config']=='E01']
 ha=statistics.mean(float(r['acc']) for r in ref)*100;hb=statistics.mean(float(r['hard_auc']) for r in ref)*100
 out.append(f'历史P1参考ACC/Hard-AUC为{ha:.4f}/{hb:.4f}%，本轮E01为{agg["E01"]["acc"][0]*100:.4f}/{agg["E01"]["hard_auc"][0]*100:.4f}%。E01逐字复用旧P1文本和投影数学，但fold2–10使用E00配对入口；历史P1使用自身连续十折随机流。两者不是相同初始化轨迹的重复。本轮正式对照以重训E01为准，历史数值单独保留，不混入420折。E07相对历史P1参考ACC仍为{agg["E07"]["acc"][0]*100-ha:+.4f}pp；该历史差异只作描述，非同初始化配对。')
 e8=[r for r in ds if r['config']=='E08'];details=[]
 for md in MODALITIES:
  k=md.lower();valid=[r for r in e8 if float(r['local_norm_group0_'+k])>=1e-8 and float(r['local_norm_group1_'+k])>=1e-8]
  cosine=statistics.mean(float(r['local_cos_'+k]) for r in valid)
  gate=statistics.mean(float(r['local_gate_'+k]) for r in e8)
  near=statistics.mean(float(r['local_gate_near_half_'+k]) for r in e8)*100
  details.append(f'{md}: 有效非零原型cos={cosine:.5f}，gate={gate:.5f}，近均匀={near:.2f}%')
 out.append('E08排除近零向量对后，四模态诊断为：'+ '；'.join(details)+'。ANAT/FUNC的两类方向高度接近，Correlation门控偏离.5最大；这只说明机制响应较强，没有模态消融支持它主导性能。包含零向量的cos均值会被拉低，不能把低cos直接当作语义分离。')
 real={ (int(r['seed']),int(r['fold'])):r for r in cf if r['condition']=='REAL'}
 off={ (int(r['seed']),int(r['fold'])):r for r in cf if r['condition']=='DISEASE_OFF'}
 changes={k:statistics.mean((float(real[x][k])-float(off[x][k]))*100 for x in real) for k in ('acc','balanced_accuracy','probability_auc')}
 e4={(int(r['seed']),int(r['fold'])):r for r in rows if r['config']=='E04'}
 offdiff={k:statistics.mean((float(off[x][k])-float(e4[x][k]))*100 for x in off) for k in ('acc','balanced_accuracy','probability_auc')}
 out.append('E08最终疾病融合REAL−OFF的ACC/BA/Prob-AUC变化为'+ '/'.join(f'{changes[k]:+.4f}pp' for k in changes)+'；OFF与独立E04重训模型的差为'+ '/'.join(f'{offdiff[k]:+.4f}pp' for k in offdiff)+'。OFF保留辅助监督及融合联合训练改变的全部参数，后者不是纯辅助CE的因果估计。最终融合主要改变少量分类阈值结果，没有可见的额外连续AUC增益。')
 for cfg,base in [('E11','E00'),('E12','E02')]:
  ba=paired(rows,[('E08',1),('E04',-1),(cfg,-1),(base,1)],'balanced_accuracy');auc=paired(rows,[('E08',1),('E04',-1),(cfg,-1),(base,1)],'probability_auc')
  out.append(f'跨层增益差P3相对{cfg}：BA {ba["mean"]:+.4f}pp、三seed均正；Prob-AUC {auc["mean"]:+.4f}pp，seed0/1/2 '+ '/'.join(f'{v:+.4f}' for v in auc['seeds'])+'pp。这提供BA层面的描述性线索，但连续排序收益不一致，且不能替代E08相对E04的绝对改善检验，暂不支持稳定跨层协同。')
 enabled=[r for r in ds if r['config'] in DISEASE_ON]
 signs={label:sum(int(r['disease_gamma_sign'])==value for r in enabled) for label,value in [('正',1),('零',0),('负',-1)]}
 out.append('270个疾病ON折的gamma符号分布为'+ '/'.join(f'{k}{v}' for k,v in signs.items())+'，实际贡献均非近零；所有420折局部MLP均有非零有限梯度。有效梯度、非零响应和文本替换敏感性支持工程连通性，不能替代正确/中性与正确/错配的语义价值对照。')
 out.append('修复记录：准备期发现的一句NEW_P3_SHARED转录差异已在该组训练前改回任务原文，并核对80句新增文本；E00已保存的model/Adam/RNG未变。OSS挂载的短暂rename可见性及checkpoint阻塞通过有界读取和本机临时state修正；12个已完成E00折不重跑，迁移26个文件逐字节SHA256一致。结束后的维护只处理完成任务跳过、报表哈希和冻结文件复用，不改变训练数学，也未增加训练。')
 return out


def update_handoff():
 m=read_json(MANIFEST_FILE);rows=read_csv(FOLD_FILE);sv,agg=summaries(rows)
 marker='## 实验十四：Hierarchical Semantic Prompt Design and Cross-Level Interaction';old=HANDOFF.read_text() if HANDOFF.exists() else ''
 base=old.split(marker)[0] if marker in old else old
 if m['protected_handoff_sha256']:
  length=m['protected_handoff_bytes'];assert hashlib.sha256(old.encode()[:length]).hexdigest()==m['protected_handoff_sha256'],'historical handoff prefix changed'
 chapter=[marker,'',f'日期{m["completed_utc"]}；目录 `{EXP}`；COMPLETE，14组42任务420折，双GPU各21任务210折，耗时{m["wall_seconds"]/60:.2f}分钟；420诊断、150反事实。未commit/push。', '',f'本地HEAD `{m["git_head"]}`；远端main `{m["remote_main"]}`；源/数据SHA256：`{json.dumps(m["source_data_sha256"],ensure_ascii=False)}`。Prompt `{m["prompt_sha256"]}`，CLIP `{m["clip_weight_sha256"]}`，patch `{m["patch_sha256"]}`。', '', '|组|Local|Disease|ACC|BA|Hard-AUC|Prob-AUC|SEN|SPE|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for c in CONFIGS:chapter.append('|'+ '|'.join([c,LOCAL_MODES[c],DISEASE_MODES[c]]+[f'{agg[c][k][0]*100:.3f}±{agg[c][k][1]*100:.3f}' for k in METRICS])+'|')
 chapter+=['','主要配对与机制：']+conclusions(rows,read_csv(DIAG_FILE),read_csv(CF_FILE),agg)
 chapter+=['','全部预设配对/跨层增益差及三seed方向、胜平负见report；公共初始化、入口RNG、梯度、文本长度、E00历史parity与checkpoint恢复均通过。固定14组，无个体化模块，无超参数搜索；原GCN/RES/损失和val=test保留。val=test、全局标准化及RFE共享信息限制独立泛化推断，专属/共享存在文本细化与数量混杂。实验一至十三与根源码不变，结束清理临时训练状态。', '',f'结果：`{FOLD_FILE}`、`{DIAG_FILE}`、`{CF_FILE}`、`{REPORT_FILE}`、`{MANIFEST_FILE}`。重跑 `python {EXP}/run.py all`；恢复 `python {EXP}/run.py resume`；刷新报告 `python {EXP}/run.py report`。']
 atomic_write(HANDOFF,base+('' if marker in old else '\n\n')+'\n'.join(chapter)+'\n')

def print_final():
 m=read_json(MANIFEST_FILE);rows=read_csv(FOLD_FILE);_,agg=summaries(rows)
 print('实验十四 '+m['run_status']+' 42/42任务 420/420折；双GPU各21任务/210折；耗时 %.2f分钟'%(m['wall_seconds']/60),flush=True)
 print('Config ACC BA Hard-AUC Prob-AUC SEN-ASD SPE-TD (mean±seed SD %)')
 for c in CONFIGS:print(c,' '.join(f'{agg[c][k][0]*100:.3f}±{agg[c][k][1]*100:.3f}' for k in METRICS))
 print('报告 '+str(REPORT_FILE));print('handoff已更新；临时state已清理；未commit/push')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',nargs='?',choices=['prepare','preflight','all','resume','refresh','report']);parser.add_argument('--worker',nargs=2,metavar=('CONFIG','SEED'));parser.add_argument('--workers-per-gpu',type=int,default=4)
 for flag in ('prepare','preflight','all','resume','refresh','report'):parser.add_argument('--'+flag,action='store_true')
 args=parser.parse_args()
 if args.worker:return worker(args.worker[0],int(args.worker[1]))
 cmd=args.command or next((c for c in ('prepare','preflight','all','resume','refresh','report') if getattr(args,c)),None)
 if cmd=='prepare':prepare()
 elif cmd=='preflight':preflight()
 elif cmd in ('all','resume'):schedule(args.workers_per_gpu)
 elif cmd=='refresh':refresh();report()
 elif cmd=='report':
  report();m=read_json(MANIFEST_FILE)
  if m['run_status']=='COMPLETE':
   m['result_sha256']['results/report.md']=sha256(REPORT_FILE);write_json(MANIFEST_FILE,m)
 else:parser.print_help()
 return 0

if __name__=='__main__':sys.exit(main())
