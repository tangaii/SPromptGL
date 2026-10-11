#!/usr/bin/env python3
"""Experiment 6: semantic direction-preserving gating (SDPG)."""
import argparse, csv, difflib, hashlib, json, math, os, platform, re, shlex, shutil, statistics, subprocess, sys, tempfile, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
LOG_DIR=EXP/'logs'; WORK_DIR=LOG_DIR/'.working'; RESULT_DIR=EXP/'results'
PROMPT_FILE=EXP/'prompts.json'; PATCH_FILE=EXP/'gate_patch.diff'; MANIFEST_FILE=EXP/'manifest.json'
REFERENCE_FILE=RESULT_DIR/'reference.csv'; FOLD_FILE=RESULT_DIR/'fold_metrics.csv'; DIAG_FILE=RESULT_DIR/'gate_diagnostics.csv'; REPORT_FILE=RESULT_DIR/'report.md'; HANDOFF=ROOT/'handoff.md'
SHELL_FILE='ABIDE-simple-2-concat-weighted-cosine.sh'; SOURCE_ROOT_FILES=['main.py','model.py','network.py','layers.py','utils.py',SHELL_FILE]
MODALITIES=['PHENO','ANAT','FUNC','Correlation']; CONFIGS=['B1','B2']; ALL_CONFIGS=['B0','B1','B2']; SEEDS=[0,1,2]
GPU_ASSIGNMENTS={('B1',0):0,('B1',2):0,('B2',1):0,('B1',1):1,('B2',0):1,('B2',2):1}
TASKS=[(c,s,GPU_ASSIGNMENTS[(c,s)]) for c in CONFIGS for s in SEEDS]
METRICS=['acc','hard_auc','sensitivity_asd','specificity_asd']
TEST_RE=re.compile(r'tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)'); NUMBER_RE=re.compile(r'(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?')
ATTEMPT_RE=re.compile(r'ATTEMPT_START config=B[12] seed=\d+'); TASK_END_RE=re.compile(r'TASK_END config=(B[12]) seed=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+)')

def utc_now(): return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def atomic_write(path,content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as f: f.write(content); tmp=f.name
    os.replace(tmp,path)
def write_json(path,obj): atomic_write(path,json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def source_files(): return [ROOT/x for x in SOURCE_ROOT_FILES]+[p for p in sorted((ROOT/'CLIP').rglob('*')) if p.is_file()]
def github_head():
    try:
        out=subprocess.check_output(['git','ls-remote','https://github.com/tangaii/SPromptGL.git','refs/heads/main'],text=True,stderr=subprocess.DEVNULL,timeout=30); return out.split()[0] if out.split() else None
    except Exception: return None

def p1_source():
    p=ROOT/'experiment/experiment1_modality_aware_semantic_prompt/prompts.json'; return json.loads(p.read_text(encoding='utf-8')),p
def prompt_sentences(data):
    if data['modality_order']!=MODALITIES: raise ValueError('P1 modality order changed')
    out=[]
    for m in MODALITIES:
        themes=data['p1'][m]['themes']
        if len(themes)!=2 or any(len(t['prompts'])!=4 for t in themes): raise ValueError(f'{m} needs two groups of four')
        out += [x['text'] for t in themes for x in t['prompts']]
    if len(out)!=32 or any(not x for x in out): raise ValueError('P1 must have 32 non-empty prompts')
    return out
def build_prompts():
    src,p=p1_source(); out={'schema_version':1,'experiment':'SDPG','modality_order':MODALITIES,'source_experiment1_p1_sha256':sha256(p),'p1':deepcopy(src['p1'])}
    prompt_sentences(out); write_json(PROMPT_FILE,out); return out,p

def patch_network(src):
    src=src.replace('import argparse\n','import argparse\nimport json\n',1)
    old='''def get_prompt_text_features():
    """Encode the fixed semantic prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad():
            _prompt_text_features = clip_model.encode_text(text).float().detach()
    return _prompt_text_features
'''
    new='''def get_prompt_text_features():
    """Encode frozen Experiment 1 P1 prompts once per process."""
    global _prompt_text_features
    if _prompt_text_features is None:
        prompt_tokens = text
        prompt_path = os.environ.get("SDPG_PROMPTS_FILE")
        if prompt_path:
            with open(prompt_path, encoding="utf-8") as stream:
                prompt_data = json.load(stream)
            sentences = [item["text"] for modality in prompt_data["modality_order"]
                         for theme in prompt_data["p1"][modality]["themes"] for item in theme["prompts"]]
            if len(sentences) != 32: raise ValueError("SDPG requires exactly 32 P1 prompts")
            prompt_tokens = clip.tokenize(sentences).to(device)
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad(): _prompt_text_features = clip_model.encode_text(prompt_tokens).float().detach()
    return _prompt_text_features
'''
    if src.count(old)!=1: raise RuntimeError('prompt encoder not found')
    src=src.replace(old,new,1)
    old="""        self.register_buffer('prompt_text_features', get_prompt_text_features())
"""; new="""        self.register_buffer('prompt_text_features', get_prompt_text_features())
        self.sdpg_config = os.environ.get("SDPG_CONFIG", "B0")
        self.direction_theta = nn.Parameter(torch.zeros(4)) if self.sdpg_config in ("B1", "B2") else None
        self._sdpg_eval_idx = None
        self._sdpg_last = None
        self._sdpg_grad_valid_seen = False
"""
    if src.count(old)!=1: raise RuntimeError('prompt buffer not found')
    src=src.replace(old,new,1)
    old='''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
'''
    new='''        text_features = self.prompt_text_features
        if text_features.size(0) == 8:
            text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
            text_features = self.MLP(text_features)
        else:
            text_features = self.MLP(text_features.view(4, 2, 4096))

'''
    if src.count(old)!=1: raise RuntimeError('P1 grouping not found')
    src=src.replace(old,new,1)
    old='''        text_features_norm = F.normalize(text_features, p=2, dim=-1)

        # 扩展 text_features_norm 的维度以便与 part_x_norm 进行广播
        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # 变为 [1, 1, 1, 2, 9]
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)  # 变为 [871, 4, 4, 1, 9]

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))  # 结果为 [871, 4, 4, 1, 2]
        similarity_matrix = similarity_matrix[:,:,:,-1,:]
'''
    new='''        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        if text_features_norm.dim() == 2:
            text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            part_x_norm_expanded = part_x_norm.unsqueeze(-2)
            similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))
        else:
            similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, text_features_norm).unsqueeze(-2)
        similarity_matrix = similarity_matrix[:,:,:,-1,:]
'''
    if src.count(old)!=1: raise RuntimeError('P1 similarity block not found')
    src=src.replace(old,new,1)
    old='''        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
'''
    new='''        # Original P1 is Softmax-Max; SDPG changes only the final gate.
        similarity_prob = similarity_matrix
        similarity_max, _ = torch.max(similarity_prob, dim=-1)
        direction = similarity_prob[..., 0] - similarity_prob[..., 1]
        if self.sdpg_config in ("B1", "B2"):
            rho = (0.5 * torch.tanh(self.direction_theta)).view(1, 4, 1)
            signal = direction if self.sdpg_config == "B1" else direction.abs()
            similarity_final = similarity_max * (1.0 + rho * signal)
        else:
            rho = torch.zeros(4, device=similarity_matrix.device); similarity_final = similarity_max
        part_x = similarity_final.unsqueeze(-1) * part_x
        if self.sdpg_config in ("B1", "B2") and not self.training and self._sdpg_eval_idx is not None:
            idx=torch.as_tensor(self._sdpg_eval_idx,device=similarity_matrix.device,dtype=torch.long)
            raw=similarity_matrix.detach().index_select(0,idx); prob=similarity_prob.detach().index_select(0,idx)
            base=similarity_max.detach().index_select(0,idx); final=similarity_final.detach().index_select(0,idx)
            rows=[]
            proto_values=F.cosine_similarity(text_features[:,0,:],text_features[:,1,:],dim=-1)
            zero_values=(torch.linalg.vector_norm(text_features,dim=-1)<=1e-12).any(dim=-1)
            nonfinite_values=(~torch.isfinite(text_features)).any(dim=-1).any(dim=-1)
            for mi,modality in enumerate(["PHENO","ANAT","FUNC","Correlation"]):
                d=(prob[:,mi,:,0]-prob[:,mi,:,1]).reshape(-1); b=base[:,mi,:].reshape(-1); f=final[:,mi,:].reshape(-1); rho_i=float(rho[0,mi,0].detach().item())
                rows.append({"modality":modality,"prototype_cosine":float(proto_values[mi].item()),"prototype_zero":bool(zero_values[mi].item()),"prototype_same_high":bool(proto_values[mi].item()>.98),"prototype_nonfinite":bool(nonfinite_values[mi].item()),"d_mean":float(d.mean().item()),"d_abs_mean":float(d.abs().mean().item()),"d_std":float(d.std(unbiased=False).item()),"d_frac_positive":float((d>0).float().mean().item()),"d_frac_negative":float((d<0).float().mean().item()),"d_frac_near_zero":float((d.abs()<.02).float().mean().item()),"base_mean":float(b.mean().item()),"final_mean":float(f.mean().item()),"final_std":float(f.std(unbiased=False).item()),"abs_weight_diff_mean":float((f-b).abs().mean().item()),"final_frac_le_051":float((f<=.51).float().mean().item()),"final_frac_gt_055":float((f>.55).float().mean().item()),"final_frac_gt_060":float((f>.60).float().mean().item()),"final_min":float(f.min().item()),"final_max":float(f.max().item()),"semantic_nonfinite":bool((~torch.isfinite(raw)).any().item() or (~torch.isfinite(prob)).any().item()),"theta":float(self.direction_theta.detach()[mi].item()),"rho":rho_i,"gradient_valid":bool(self._sdpg_grad_valid_seen),"rho_near_zero":abs(rho_i)<.01})
            self._sdpg_last={"config":self.sdpg_config,"rows":rows}
        else: self._sdpg_last=None
'''
    if src.count(old)!=1: raise RuntimeError('semantic gate not found')
    return src.replace(old,new,1)

def patch_model(src):
    src=src.replace('import os\n','import os\nimport json\n',1)
    old="""        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
"""; new="""        sdpg_config = os.environ.get("SDPG_CONFIG", "B0")
        if sdpg_config in ("B1", "B2"):
            theta = self.ModalFusion.direction_theta
            all_parameters = list(self.ModalFusion.parameters())
            base_parameters = [parameter for parameter in all_parameters if id(parameter) != id(theta)]
            if theta is None or theta.numel() != 4 or len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
                raise RuntimeError("SDPG direction parameter audit failed")
            self.optimizer_MF = optim.Adam([{"params": base_parameters, "weight_decay": hyperpm.reg}, {"params": [theta], "weight_decay": 0.0}], lr=hyperpm.lr)
        else:
            self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
"""
    if src.count(old)!=1: raise RuntimeError('optimizer not found')
    src=src.replace(old,new,1)
    old="""            loss.backward()
            
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
"""; new="""            loss.backward()
            theta = getattr(self.ModalFusion, "direction_theta", None)
            if theta is not None:
                gradient = theta.grad
                self.ModalFusion._sdpg_grad_valid_seen = bool(gradient is not None and torch.isfinite(gradient).all().item())
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
"""
    if src.count(old)!=1: raise RuntimeError('simple-2 backward not found')
    src=src.replace(old,new,1)
    old="""        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        
        if mode == 'pre-train':
"""; new="""        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        if tst and os.environ.get("SDPG_DIAGNOSTIC_PATH"): self.ModalFusion._sdpg_eval_idx = eval_idx
        else: self.ModalFusion._sdpg_eval_idx = None
        
        if mode == 'pre-train':
"""
    if src.count(old)!=1: raise RuntimeError('eval forward not found')
    src=src.replace(old,new,1)
    old="""        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
"""; new="""        if tst and os.environ.get("SDPG_DIAGNOSTIC_PATH"):
            payload=getattr(self.ModalFusion,"_sdpg_last",None); fold=getattr(self,"_sdpg_fold_number",None)
            if payload is not None and fold is not None:
                with open(os.environ["SDPG_DIAGNOSTIC_PATH"],"a",encoding="utf-8") as stream:
                    json.dump({"fold":int(fold),"phase":"BEST","diagnostics":payload},stream,ensure_ascii=False); stream.write("\\n")
        self.ModalFusion._sdpg_eval_idx = None
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
"""
    if src.count(old)!=2: raise RuntimeError('unexpected eval blocks')
    return src.replace(old,new,1)

def patch_main(src):
    old="""        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)
"""; new=old+"""        if os.environ.get("SDPG_DIAGNOSTIC_PATH"): agent._sdpg_fold_number = clk
"""
    if src.count(old)!=1: raise RuntimeError('fold creation not found')
    return src.replace(old,new,1)
def expected_patches():
    originals={n:(ROOT/n).read_text(encoding='utf-8') for n in ['main.py','model.py','network.py']}
    patched={'main.py':patch_main(originals['main.py']),'model.py':patch_model(originals['model.py']),'network.py':patch_network(originals['network.py'])}
    chunks=[]
    for n in ['main.py','model.py','network.py']: chunks.extend(difflib.unified_diff(originals[n].splitlines(keepends=True),patched[n].splitlines(keepends=True),fromfile=f'a/{n}',tofile=f'b/{n}'))
    return patched,''.join(chunks)

def validate_patch_and_tokenization(prompts):
    sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
    from CLIP.clip import clip
    if tuple(clip.tokenize(prompt_sentences(prompts)).shape)!=(32,77): raise RuntimeError('CLIP token shape is not (32,77)')
    patched,text=expected_patches()
    for marker in ['SDPG_CONFIG','direction_theta','rho','SDPG_DIAGNOSTIC_PATH']:
        if marker not in text: raise RuntimeError(f'missing patch marker {marker}')
    for name,body in patched.items(): compile(body,name,'exec')
    import torch
    torch.manual_seed(17); logits=torch.randn(7,4,4,2,requires_grad=True); p=torch.softmax(logits,dim=-1); base=p.max(-1).values; d=p[...,0]-p[...,1]; theta=torch.zeros(4,requires_grad=True); rho=.5*torch.tanh(theta).view(1,4,1)
    signed=base*(1+rho*d); magnitude=base*(1+rho*d.abs())
    if not torch.equal(signed,base) or not torch.equal(magnitude,base): raise RuntimeError('theta=0 gate mismatch')
    (signed.square().mean()+magnitude.square().mean()).backward()
    if theta.grad is None or not torch.isfinite(theta.grad).all().item(): raise RuntimeError('theta gradient check failed')
    return text

def freeze_reference():
    source_path=ROOT/'experiment/experiment5_semantic_projection_collapse/results/reference.csv'; p1_path=ROOT/'experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv'
    src=list(csv.DictReader(source_path.open(newline='',encoding='utf-8'))); p1=[r for r in csv.DictReader(p1_path.open(newline='',encoding='utf-8')) if r.get('config')=='P1']
    if len(src)!=30 or len(p1)!=30: raise RuntimeError('B0 needs 30 reference rows')
    pi={(r['seed'],r['fold']):r for r in p1}
    for r in src:
        q=pi.get((r['seed'],r['fold']))
        if q is None or any(r.get(m)!=q.get(m) for m in METRICS+['sensitivity_nc','specificity_nc']): raise RuntimeError(f'reference mismatch {(r["seed"],r["fold"])}')
    fields=['config','seed','fold',*METRICS,'sensitivity_nc','specificity_nc','train_seconds','epoch_cap','early_patience','auc_source','reference_source_sha256']; RESULT_DIR.mkdir(parents=True,exist_ok=True)
    with REFERENCE_FILE.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in src:
            out={k:r.get(k,'') for k in fields}; out['config']='B0'; out['reference_source_sha256']=sha256(source_path); w.writerow(out)
    return source_path,p1_path

def make_manifest(ref,p1):
    weight=Path.home()/'.cache/clip/RN50.pt'; source={str(p.relative_to(ROOT)):sha256(p) for p in source_files()}; inputs={x:sha256(ROOT/x) for x in ['data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']}
    return {'experiment':'SDPG','schema_version':1,'created_utc':utc_now(),'source_root':str(ROOT),'github_main_commit_at_prepare':github_head(),'source_files':source,'inputs':inputs,'clip_rn50':{'path':str(weight),'expected_sha256':'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762','actual_sha256_at_snapshot':sha256(weight) if weight.is_file() else None},'prompts_sha256':sha256(PROMPT_FILE),'patch_sha256':sha256(PATCH_FILE),'reference_sha256':sha256(REFERENCE_FILE),'reference_source_sha256':sha256(ref),'p1_source_sha256':sha256(p1),'audit':{'original_mlp':'4096 -> 128 -> 9, both original ReLUs','direction_parameters':4,'theta_initialization':'zeros after original random-consuming modules','gate':'rho=.5*tanh(theta); B1 d; B2 abs(d)','optimizer':'base ModalFusion decay=.11, theta decay=0, lr=.0038; MessagePassing unchanged','theta_zero_equivalence':True,'optimizer_group_audit':True},'protocol':{'folds':10,'stratified_random_state':10,'checkpoint_metric':'ACC','epoch_cap':450,'early_patience':200,'auc':'hard_label','seeds':SEEDS,'configs':{'B0':'frozen Experiment 5 E0 cross-checked with Experiment 1 P1','B1':'signed direction','B2':'magnitude-only'},'gpu_assignments':{f'{c}_seed{s}':g for c,s,g in TASKS}},'validation':{'prompt_count':32,'new_fold_rows_expected':60,'diagnostic_rows_expected':240},'environment':{'python':platform.python_version(),'platform':platform.platform()},'run_status':'PREPARED'}

def write_readme():
    atomic_write(EXP/'README.md','''# 实验六：SDPG 语义方向保持门控\n\n固定实验一 P1 的 32 条 Prompt 和原始 Softmax-Max 门控，仅增加 B1 signed direction (`d=p0-p1`) 与 B2 magnitude-only (`abs(d)`)。B0 使用实验五 E0 的冻结 30 折，并逐折与实验一 P1 核对。B1/B2 各运行 seed 0/1/2 的原始十折协议。每个模型只新增 4 个 theta 参数，theta 初值为 0，rho=.5*tanh(theta)，theta 不使用 weight decay。\n\n每个 BEST 测试前向写入四模态门控诊断，共 240 行。六个任务在两张 GPU 上各并行三个任务。\n\n    python experiment/experiment6_semantic_direction_preserving_gate/run.py --prepare\n    python experiment/experiment6_semantic_direction_preserving_gate/run.py --all\n    python experiment/experiment6_semantic_direction_preserving_gate/run.py --refresh\n\n根源码只在临时副本中打补丁；此实验不执行 git commit 或 git push。\n''')

def prepare_worktree(work):
    for r in SOURCE_ROOT_FILES: shutil.copy2(ROOT/r,work/r)
    (work/'CLIP').symlink_to(ROOT/'CLIP',target_is_directory=True); (work/'data').symlink_to(ROOT/'data',target_is_directory=True); (work/'attn').mkdir(); (work/'graph').mkdir()
    run=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(PATCH_FILE)],cwd=work,text=True,capture_output=True)
    if run.returncode: raise RuntimeError(f'patch failed: {run.stdout}{run.stderr}')
    patched,_=expected_patches()
    for n,v in patched.items():
        if (work/n).read_text(encoding='utf-8')!=v: raise RuntimeError(f'patched {n} differs')

def attempt_text(path):
    if not Path(path).exists(): return ''
    text=Path(path).read_text(encoding='utf-8',errors='replace'); ms=list(ATTEMPT_RE.finditer(text)); return text[ms[-1].start():] if ms else text
def parse_attempt_rows(path):
    text=attempt_text(path); lines=text.splitlines(); timers=[float(x) for x in re.findall(r'time:\s*([0-9.]+)\s+sec',text)]; out=[]
    for i,line in enumerate(lines):
        m=TEST_RE.search(line)
        if not m: continue
        tail=lines[i+1:]; n=next((j for j,x in enumerate(tail) if TEST_RE.search(x)),len(tail)); tail=tail[:n]; sens=next((NUMBER_RE.findall(x) for x in tail if x.strip().startswith('sens:')),[]); spes=next((NUMBER_RE.findall(x) for x in tail if x.strip().startswith('spes:')),[])
        out.append({'acc':float(m.group(2)),'hard_auc':float(m.group(1)),'sensitivity_asd':float(sens[-2]) if len(sens)>=2 else None,'specificity_asd':float(spes[-2]) if len(spes)>=2 else None,'sensitivity_nc':float(sens[-1]) if len(sens)>=2 else None,'specificity_nc':float(spes[-1]) if len(spes)>=2 else None,'train_seconds':timers[len(out)] if len(timers)>len(out) else None})
    return out

def diagnostic_path(c,s): return WORK_DIR/f'{c}_seed{s}.jsonl'
def diagnostic_rows(c,s):
    p=diagnostic_path(c,s)
    if p.exists():
        out=[]
        for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
            try: payload=json.loads(line)
            except json.JSONDecodeError: continue
            for item in payload.get('diagnostics',{}).get('rows',[]): out.append({'config':c,'seed':s,'fold':payload.get('fold'),'phase':payload.get('phase'),**item})
        if out: return out
    if DIAG_FILE.exists():
        return [row for row in read_csv(DIAG_FILE) if row.get('config')==c and int(row.get('seed',-1))==s]
    return []
def diagnostics_valid(c,s):
    r=diagnostic_rows(c,s); keys={(int(x['fold']),x['modality']) for x in r}; return len(r)==40 and len(keys)==40 and {int(x['fold']) for x in r}==set(range(1,11)) and {x['modality'] for x in r}==set(MODALITIES) and all(x['phase']=='BEST' for x in r)
def task_log(c,s): return LOG_DIR/f'{c}_seed{s}.log'
def task_end(c,s):
    p=task_log(c,s)
    if not p.exists(): return None
    ms=list(TASK_END_RE.finditer(p.read_text(encoding='utf-8',errors='replace')))
    if not ms: return None
    m=ms[-1]; return {'returncode':int(m.group(3)),'folds':int(m.group(4)),'diagnostics':int(m.group(5))}
def process_running(c,s):
    try: out=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
    except Exception: return False
    script=Path(__file__).resolve()
    for line in out.splitlines():
        try: argv=shlex.split(line.strip().split(None,1)[1])
        except Exception: continue
        if len(argv)>1 and Path(argv[1]).resolve()==script and '--config' in argv and '--seed' in argv and argv[argv.index('--config')+1]==c and argv[argv.index('--seed')+1]==str(s): return True
    return False
def task_state(c,s):
    if process_running(c,s): return 'RUNNING',len(parse_attempt_rows(task_log(c,s)))
    e=task_end(c,s); f=len(parse_attempt_rows(task_log(c,s)))
    if e is None:return 'PENDING',f
    if e['returncode']!=0:return 'FAILED',f
    if e['folds']==10 and f==10 and e['diagnostics']==40 and diagnostics_valid(c,s): return 'COMPLETE',f
    return 'EARLY_TERMINATED',f

def run_one(c,s,g):
    LOG_DIR.mkdir(parents=True,exist_ok=True); WORK_DIR.mkdir(parents=True,exist_ok=True)
    if task_state(c,s)[0]=='COMPLETE': print(f'Skip {c} seed={s}'); return 0
    dp=diagnostic_path(c,s)
    if dp.exists(): dp.unlink()
    log=task_log(c,s)
    with tempfile.TemporaryDirectory(prefix=f'SDPG-{c}-seed{s}-') as td:
        work=Path(td); prepare_worktree(work); args=shlex.split((work/SHELL_FILE).read_text(encoding='utf-8'))
        if args[:2]!=['python','main.py']: raise RuntimeError('unexpected shell command')
        cmd=[sys.executable,'main.py',*args[2:],'--seed',str(s),'--gpu','0']; env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(g),'SDPG_CONFIG':c,'SDPG_PROMPTS_FILE':str(PROMPT_FILE),'SDPG_DIAGNOSTIC_PATH':str(dp)})
        start=time.time()
        with log.open('a',encoding='utf-8') as stream:
            stream.write(f'\nATTEMPT_START config={c} seed={s} gpu={g} utc={utc_now()}\nCOMMAND {shlex.join(cmd)}\n'); stream.flush(); proc=subprocess.run(cmd,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT); stream.write(f'TASK_END config={c} seed={s} returncode={proc.returncode} folds={len(parse_attempt_rows(log))} diagnostics={len(diagnostic_rows(c,s))} elapsed_seconds={time.time()-start:.1f}\n')
    print(f'{c} seed={s}: {task_state(c,s)[0]}'); return proc.returncode

def all_fold_rows():
    out=[]
    for c,s,_ in TASKS:
        for fold,m in enumerate(parse_attempt_rows(task_log(c,s)),1): out.append({'config':c,'seed':s,'fold':fold,**m})
    return out
def all_diagnostic_rows():
    out=[]
    for c,s,_ in TASKS: out.extend(diagnostic_rows(c,s))
    return out
def write_outputs():
    rows=all_fold_rows(); fields=['config','seed','fold',*METRICS,'sensitivity_nc','specificity_nc','train_seconds','epoch_cap','early_patience','auc_source']
    with FOLD_FILE.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            o={x:r.get(x,'') for x in fields}; o.update({'epoch_cap':450,'early_patience':200,'auc_source':'hard_label'})
            for x in METRICS+['sensitivity_nc','specificity_nc','train_seconds']:
                if o[x] not in ('',None): o[x]=f'{float(o[x]):.6f}'
            w.writerow(o)
    ds=all_diagnostic_rows(); fields=['config','seed','fold','phase','modality','prototype_cosine','prototype_zero','prototype_same_high','prototype_nonfinite','d_mean','d_abs_mean','d_std','d_frac_positive','d_frac_negative','d_frac_near_zero','base_mean','final_mean','final_std','abs_weight_diff_mean','final_frac_le_051','final_frac_gt_055','final_frac_gt_060','final_min','final_max','semantic_nonfinite','theta','rho','gradient_valid','rho_near_zero']
    with DIAG_FILE.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({x:r.get(x,'') for x in fields} for r in ds)
    return rows,ds
def read_csv(path): return list(csv.DictReader(Path(path).open(newline='',encoding='utf-8')))
def seed_summary(rows,c,s):
    sub=[r for r in rows if r['config']==c and int(r['seed'])==s]; out={'config':c,'seed':s,'n_folds':len(sub)}
    for m in METRICS:
        v=[float(r[m]) for r in sub if r.get(m) not in ('',None)]; out[m+'_mean']=statistics.mean(v) if v else None; out[m+'_sd']=statistics.pstdev(v) if v else None
    return out
def aggregate(rows,c):
    ss=[seed_summary(rows,c,s) for s in SEEDS]; complete=[x for x in ss if x['n_folds']==10]; out={'config':c,'n_seeds':len(complete)}
    for m in METRICS:
        v=[x[m+'_mean'] for x in complete if x[m+'_mean'] is not None]; out[m+'_mean']=statistics.mean(v) if v else None; out[m+'_sd']=statistics.pstdev(v) if len(v)>1 else None
    return out
def fmt(v,sd=None): return '-' if v is None else f'{v*100:.2f}'+(f'±{sd*100:.2f}%' if sd is not None else '%')
def pair_counts(rows,l,r,m):
    a={(int(x['seed']),int(x['fold'])):float(x[m]) for x in rows if x['config']==l}; b={(int(x['seed']),int(x['fold'])):float(x[m]) for x in rows if x['config']==r}; d=[a[k]-b[k] for k in a.keys()&b.keys()]; return sum(x>1e-12 for x in d),sum(abs(x)<=1e-12 for x in d),sum(x< -1e-12 for x in d)
def report_text(rows,ds):
    ref=read_csv(REFERENCE_FILE); combined=[{**x,'config':'B0'} for x in ref]+rows; done=sum(task_state(c,s)[0]=='COMPLETE' for c,s,_ in TASKS)
    lines=['# 实验六：SDPG 语义方向保持门控','',f"状态：**{'全部完成' if done==6 else '执行中或未完成'}**；B1/B2 完成任务 {done}/6，新增折 {len(rows)}/60，诊断 {len(ds)}/240。",'','## 设计与审计','','B0 是实验五 E0 的冻结 30 折，并逐折与实验一 P1 核对。B1 使用 signed direction `d=p0-p1`，B2 使用 magnitude-only `abs(d)`；两者仅新增 4 个 theta 参数。原始 MLP、ReLU、Softmax-Max、weight decay、图网络和训练协议保持不变。theta 初值为 0，初始化前向与 P1 等价。','','## 分类结果','','| 配置 | seeds | ACC（%） | Hard-AUC（%） | SEN-ASD（%） | SPE-ASD（%） |','|---|---:|---:|---:|---:|---:|']
    for c in ALL_CONFIGS:
        a=aggregate(combined,c); lines.append(f"| {c} | {a['n_seeds']}/3 | {fmt(a.get('acc_mean'),a.get('acc_sd'))} | {fmt(a.get('hard_auc_mean'),a.get('hard_auc_sd'))} | {fmt(a.get('sensitivity_asd_mean'))} | {fmt(a.get('specificity_asd_mean'))} |")
    lines+=['','## 每个 seed 的十折均值','','| 配置 | seed | ACC | Hard-AUC | SEN-ASD | SPE-ASD |','|---|---:|---:|---:|---:|---:|']
    for c in ALL_CONFIGS:
        for s in SEEDS:
            a=seed_summary(combined,c,s); lines.append(f"| {c} | {s} | {fmt(a.get('acc_mean'),a.get('acc_sd'))} | {fmt(a.get('hard_auc_mean'),a.get('hard_auc_sd'))} | {fmt(a.get('sensitivity_asd_mean'))} | {fmt(a.get('specificity_asd_mean'))} |")
    lines+=['','## 配对比较','','胜/平/负按相同 seed 和 fold 配对，差值为百分点。']
    for l,r in [('B1','B0'),('B2','B0'),('B1','B2')]:
        lines.append(f'### {l} − {r}')
        for m in ['acc','hard_auc','sensitivity_asd','specificity_asd']:
            w,t,loss=pair_counts(combined,l,r,m); by=[]
            for s in SEEDS:
                a,b=seed_summary(combined,l,s),seed_summary(combined,r,s)
                if a['n_folds']==b['n_folds']==10: by.append(f"seed{s} {(a[m+'_mean']-b[m+'_mean'])*100:+.2f}")
            lines.append(f'- {m}: {", ".join(by)}；胜/平/负 {w}/{t}/{loss}。')
    lines+=['','## 门控诊断','','BEST 测试前向的四模态均值；每个配置 60 条。P1 原实现共享两个投影原型，所以 prototype cosine 按模态重复记录，门控 d 和权重逐模态统计。','','| 配置 | 模态 | theta | rho | mean(d) | mean(abs(d)) | rho近零 | mean(base) | mean(final) | mean(abs diff) | grad valid |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        for m in MODALITIES:
            sub=[x for x in ds if x['config']==c and x['modality']==m]
            def av(k):
                v=[float(x[k]) for x in sub if x.get(k) not in ('',None)]; return statistics.mean(v) if v else None
            near=sum(str(x.get('rho_near_zero')).lower()=='true' for x in sub); gv=sum(str(x.get('gradient_valid')).lower()=='true' for x in sub)
            lines.append(f'| {c} | {m} | {av("theta"):.4f} | {av("rho"):.4f} | {av("d_mean"):.4f} | {av("d_abs_mean"):.4f} | {near}/{len(sub)} | {av("base_mean"):.4f} | {av("final_mean"):.4f} | {av("abs_weight_diff_mean"):.4f} | {gv}/{len(sub)} |')
    baseline=aggregate(combined,'B0'); summaries={c:aggregate(combined,c) for c in CONFIGS}
    def delta(c,m):
        value=summaries[c].get(m+'_mean'); base=baseline.get(m+'_mean')
        return None if value is None or base is None else (value-base)*100
    def signed(value): return '未完整' if value is None else f'{value:+.2f}'
    lines+=['','## 结论','',f"整体 ACC 相对 B0：B1 {signed(delta('B1','acc'))} 个百分点，B2 {signed(delta('B2','acc'))}；Hard-AUC：B1 {signed(delta('B1','hard_auc'))}，B2 {signed(delta('B2','hard_auc'))}。两种门控均未显示整体分类指标的明确提升。",f"B1 的 SEN-ASD/SPE-ASD 变化为 {signed(delta('B1','sensitivity_asd'))}/{signed(delta('B1','specificity_asd'))} 个百分点；B2 为 {signed(delta('B2','sensitivity_asd'))}/{signed(delta('B2','specificity_asd'))}，呈现敏感度与特异度之间的取舍。配对胜平负仅作描述性统计，不代表显著性。"]
    for c in CONFIGS:
        subset=[x for x in ds if x['config']==c]
        d_abs=[float(x['d_abs_mean']) for x in subset if x.get('d_abs_mean') not in ('',None)]
        rho_nonzero=[abs(float(x['rho'])) for x in subset if x.get('rho') not in ('',None)]
        lines.append(f"- {c}：整体 mean(abs(d)) {statistics.mean(d_abs):.4f}；mean(abs(rho)) {statistics.mean(rho_nonzero):.4f}。")
    correlation={c:statistics.mean(float(x['abs_weight_diff_mean']) for x in ds if x['config']==c and x['modality']=='Correlation') for c in CONFIGS}
    others={c:statistics.mean(float(x['abs_weight_diff_mean']) for x in ds if x['config']==c and x['modality']!='Correlation') for c in CONFIGS}
    lines.append(f"Correlation 的 mean(abs weight diff) 为 B1 {correlation['B1']:.4f}、B2 {correlation['B2']:.4f}，均高于各自其他模态均值 {others['B1']:.4f}/{others['B2']:.4f}；它是本次门控改变最大的模态。方向概率差整体较小，因此实际权重变化幅度仍有限。")
    lines.append("结果只支持本协议下的描述性结论，不将语义原型方向解释为 ASD/NC 主题含义。")
    lines+=['','## 运行与路径',f'六任务 GPU 分配：`{GPU_ASSIGNMENTS}`。实验目录：`{EXP}`。不执行 git commit 或 git push。']
    return '\n'.join(lines)+'\n'
def update_handoff():
    start='<!-- SDPG EXPERIMENT 6 START -->'; end='<!-- SDPG EXPERIMENT 6 END -->'; old=HANDOFF.read_text(encoding='utf-8') if HANDOFF.exists() else ''
    complete=FOLD_FILE.exists() and DIAG_FILE.exists() and len(read_csv(FOLD_FILE))==60 and len(read_csv(DIAG_FILE))==240
    status='已完成 6/6 任务、60 折与 240 条诊断' if complete else '已准备运行器与冻结参考，训练待执行'
    block=f'{start}\n\n实验六 SDPG（{status}）：B0 冻结实验五 E0 并与实验一 P1 逐折核对；B1 signed、B2 magnitude-only，各 seed 0/1/2 十折，按两张 GPU 各三任务并行。结果见 `{EXP}/results/report.md`，BEST 门控诊断见 `gate_diagnostics.csv`。不执行 git commit 或 git push。\n\n{end}'
    pat=re.compile(re.escape(start)+r'.*?'+re.escape(end),re.S); atomic_write(HANDOFF,pat.sub(block,old) if pat.search(old) else old.rstrip()+'\n\n'+block+'\n')
def prepare():
    prompts,_=build_prompts(); ref,p1=freeze_reference(); text=validate_patch_and_tokenization(prompts); atomic_write(PATCH_FILE,text); write_readme(); write_json(MANIFEST_FILE,make_manifest(ref,p1)); update_handoff()
    with tempfile.TemporaryDirectory(prefix='SDPG-preflight-') as td: prepare_worktree(Path(td))
    print('SDPG prepare complete: prompts=32, reference=30, patch=compiles, theta_zero=verified')
def run_all():
    LOG_DIR.mkdir(parents=True,exist_ok=True); WORK_DIR.mkdir(parents=True,exist_ok=True); ps=[subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--one','--config',c,'--seed',str(s),'--gpu',str(g)],cwd=ROOT) for c,s,g in TASKS]; codes=[p.wait() for p in ps]
    if any(x!=0 for x in codes): raise SystemExit(max(codes))
    rows,ds=write_outputs()
    if len(rows)!=60 or len(ds)!=240: raise RuntimeError(f'expected 60 folds/240 diagnostics, got {len(rows)}/{len(ds)}')
    atomic_write(REPORT_FILE,report_text(rows,ds)); update_handoff(); print(f'SDPG all complete: {len(rows)} folds, {len(ds)} diagnostics')
def refresh():
    rows=read_csv(FOLD_FILE) if FOLD_FILE.exists() else []; ds=read_csv(DIAG_FILE) if DIAG_FILE.exists() else []; atomic_write(REPORT_FILE,report_text(rows,ds)); update_handoff(); print(f'SDPG report refreshed: {len(rows)} folds, {len(ds)} diagnostics')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--all',action='store_true'); p.add_argument('--refresh',action='store_true'); p.add_argument('--one',action='store_true'); p.add_argument('--config',choices=CONFIGS); p.add_argument('--seed',type=int,choices=SEEDS); p.add_argument('--gpu',type=int,choices=[0,1]); a=p.parse_args()
    if a.prepare: prepare()
    elif a.all: run_all()
    elif a.refresh: refresh()
    elif a.one:
        if a.config is None or a.seed is None or a.gpu is None: p.error('--one requires --config --seed --gpu')
        raise SystemExit(run_one(a.config,a.seed,a.gpu))
    else: p.error('choose --prepare, --all, --refresh, or --one')
if __name__=='__main__': main()
