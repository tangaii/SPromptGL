#!/usr/bin/env python3
"""Experiment 7: Dual-Theme Semantic Fusion (DTSF)."""
import argparse, csv, difflib, hashlib, json, math, os, platform, re, shlex, shutil, statistics, subprocess, sys, tempfile, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DTSF_ROOT", "/data/adodas/SPromptGL-main"))
sys.path.insert(0, str(ROOT))
LOG_DIR, WORK_DIR, RESULT_DIR = EXP / "logs", EXP / "logs" / ".working", EXP / "results"
PROMPT_FILE, PATCH_FILE, MANIFEST_FILE = EXP / "prompts.json", EXP / "fusion_patch.diff", EXP / "manifest.json"
REFERENCE_FILE, FOLD_FILE = RESULT_DIR / "reference.csv", RESULT_DIR / "fold_metrics.csv"
DIAG_FILE, CF_FILE, REPORT_FILE = RESULT_DIR / "fusion_diagnostics.csv", RESULT_DIR / "counterfactual.csv", RESULT_DIR / "report.md"
HANDOFF = ROOT / "handoff.md"
SHELL_FILE = "ABIDE-simple-2-concat-weighted-cosine.sh"
SOURCE_ROOT_FILES = ["main.py", "model.py", "network.py", "layers.py", "utils.py", SHELL_FILE]
MODALITIES = ["PHENO", "ANAT", "FUNC", "Correlation"]
CONFIGS, ALL_CONFIGS, SEEDS = ["F1", "F2", "F3", "F4"], ["F0", "F1", "F2", "F3", "F4"], [0, 1, 2]
GPU_ASSIGNMENTS = {("F1",0):0,("F1",2):0,("F2",1):0,("F3",0):0,("F3",2):0,("F4",1):0,("F1",1):1,("F2",0):1,("F2",2):1,("F3",1):1,("F4",0):1,("F4",2):1}
TASKS = [(c,s,GPU_ASSIGNMENTS[(c,s)]) for c in CONFIGS for s in SEEDS]
METRICS = ["acc", "hard_auc", "sensitivity_asd", "specificity_asd"]
TEST_RE = re.compile(r"tst-auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
ATTEMPT_RE = re.compile(r"ATTEMPT_START config=F[1-4] seed=\d+")
TASK_END_RE = re.compile(r"TASK_END config=(F[1-4]) seed=(\d+) returncode=(-?\d+) folds=(\d+) diagnostics=(\d+) counterfactual=(\d+)")

def utc_now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def atomic_write(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text); tmp = stream.name
    os.replace(tmp, path)
def write_json(path, obj): atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def github_head():
    try:
        out = subprocess.check_output(["git","ls-remote","https://github.com/tangaii/SPromptGL.git","refs/heads/main"], text=True, stderr=subprocess.DEVNULL, timeout=30)
        return out.split()[0] if out.split() else None
    except Exception: return None
def p1_source():
    path = ROOT / "experiment/experiment1_modality_aware_semantic_prompt/prompts.json"
    return json.loads(path.read_text(encoding="utf-8")), path
def prompt_sentences(data):
    if data["modality_order"] != MODALITIES: raise ValueError("P1 modality order changed")
    out=[]
    for modality in MODALITIES:
        themes=data["p1"][modality]["themes"]
        if len(themes)!=2 or any(len(theme["prompts"])!=4 for theme in themes): raise ValueError(f"{modality} prompt groups invalid")
        out += [item["text"] for theme in themes for item in theme["prompts"]]
    if len(out)!=32 or any(not item for item in out): raise ValueError("P1 must contain 32 prompts")
    return out
def build_prompts():
    source, path = p1_source()
    out={"schema_version":1,"experiment":"DTSF","modality_order":MODALITIES,"source_experiment1_p1_sha256":sha256(path),"p1":deepcopy(source["p1"])}
    prompt_sentences(out); write_json(PROMPT_FILE,out); return out,path

def patch_network(src):
    src=src.replace("import argparse\n","import argparse\nimport json\n",1)
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
        prompt_path = os.environ.get("DTSF_PROMPTS_FILE")
        if not prompt_path: raise RuntimeError("DTSF_PROMPTS_FILE is required")
        with open(prompt_path, encoding="utf-8") as stream: prompt_data = json.load(stream)
        sentences = [item["text"] for modality in prompt_data["modality_order"] for theme in prompt_data["p1"][modality]["themes"] for item in theme["prompts"]]
        if len(sentences) != 32: raise ValueError("DTSF requires 32 prompts")
        clip_model, _ = clip.load('RN50', device)
        with torch.no_grad(): _prompt_text_features = clip_model.encode_text(clip.tokenize(sentences).to(device)).float().detach()
    return _prompt_text_features
'''
    if src.count(old)!=1: raise RuntimeError("prompt encoder not found")
    src=src.replace(old,new,1)
    old="""        self.register_buffer('prompt_text_features', get_prompt_text_features())
"""
    new="""        self.register_buffer('prompt_text_features', get_prompt_text_features())
        with torch.no_grad():
            grouped = F.normalize(self.prompt_text_features.view(4, 2, 4, 1024), p=2, dim=-1)
            self.register_buffer('topic_vectors', F.normalize(grouped.mean(dim=2), p=2, dim=-1))
        self.dtsf_config = os.environ.get('DTSF_CONFIG', 'F0')
        if self.dtsf_config in ('F1', 'F2', 'F3', 'F4'):
            self.theme_proj = nn.Linear(1024, 9, bias=False)
            self.semantic_fuse = nn.Linear(18, 9, bias=False)
        else:
            self.theme_proj = None; self.semantic_fuse = None
        self._dtsf_eval_idx = None; self._dtsf_last_diag = None; self._dtsf_counterfactual = None
        self._dtsf_prediction_finite = True; self._dtsf_grad_finite = {'theme_proj': False, 'semantic_fuse': False}
"""
    if src.count(old)!=1: raise RuntimeError("prompt buffer not found")
    src=src.replace(old,new,1)
    old='''        text_features = self.prompt_text_features
        text_features = torch.cat([text_features[0:4].view(1, -1), text_features[4:8].view(1, -1)], dim=0)
        text_features = self.MLP(text_features)
        
        
        part_x = x.view(871, 4, 4, 9)
        part_x_norm = F.normalize(part_x, p=2, dim=-1)
        text_features_norm = F.normalize(text_features, p=2, dim=-1)

        # 扩展 text_features_norm 的维度以便与 part_x_norm 进行广播
        text_features_norm = text_features_norm.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # 变为 [1, 1, 1, 2, 9]
        part_x_norm_expanded = part_x_norm.unsqueeze(-2)  # 变为 [871, 4, 4, 1, 9]

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(part_x_norm_expanded, text_features_norm.transpose(-1, -2))  # 结果为 [871, 4, 4, 1, 2]
        similarity_matrix = similarity_matrix[:,:,:,-1,:]

        # 进行数值截断以避免数值问题
        similarity_matrix = similarity_matrix.clamp(min=-5, max=5)

        # 计算 softmax
        similarity_matrix = torch.softmax(similarity_matrix, dim=-1).to(device)

        # 获取 similarity_matrix 的最大值
        similarity_max, _ = torch.max(similarity_matrix, dim=-1)

        # 进行最终的计算
        part_x = similarity_max.unsqueeze(-1) * part_x
'''
    new='''        grouped_text = self.prompt_text_features.view(4, 2, 4, 1024).view(4, 2, 4096)
        p1_features = self.MLP(grouped_text)
        part_x = x.view(871, 4, 4, 9)
        part_x_norm = F.normalize(part_x, p=2, dim=-1)
        p1_norm = F.normalize(p1_features, p=2, dim=-1)
        similarity_matrix = torch.einsum("nmtf,mcf->nmtc", part_x_norm, p1_norm).clamp(min=-5, max=5)
        similarity_prob = torch.softmax(similarity_matrix, dim=-1).to(device)
        similarity_max, _ = torch.max(similarity_prob, dim=-1)
        local_base = similarity_max.unsqueeze(-1) * part_x
        local_new = local_base; self._dtsf_last_diag = None
        if self.semantic_fuse is not None:
            condition = self._dtsf_counterfactual or self.dtsf_config
            topic_vectors = self.topic_vectors
            if condition in ('F2', 'SWAPPED'): topic_vectors = topic_vectors[[2, 3, 0, 1]]
            elif condition in ('F3', 'POOLED'):
                shared = F.normalize(topic_vectors.mean(dim=(0, 1), keepdim=True), p=2, dim=-1)
                topic_vectors = shared.expand(4, 2, 1024)
            theme_features = F.normalize(self.theme_proj(topic_vectors), p=2, dim=-1)
            q = F.normalize(part_x, p=2, dim=-1)
            scores = torch.einsum("nmtf,mcf->nmtc", q, theme_features)
            attention_theme = torch.full_like(scores, 0.5) if self.dtsf_config == 'F4' or condition == 'UNIFORM' else torch.softmax(scores, dim=-1)
            u1 = attention_theme[...,0:1] * (part_x * theme_features[:,0,:][None,:,None,:])
            u2 = attention_theme[...,1:2] * (part_x * theme_features[:,1,:][None,:,None,:])
            delta = 0.1 * self.semantic_fuse(torch.cat([u1,u2], dim=-1))
            if condition == 'OFF': delta = torch.zeros_like(delta)
            local_new = local_base + delta
            if not self.training and self._dtsf_eval_idx is not None and self._dtsf_counterfactual is None:
                idx=torch.as_tensor(self._dtsf_eval_idx,device=part_x.device,dtype=torch.long); att=attention_theme.index_select(0,idx); base=local_base.index_select(0,idx); residual=delta.index_select(0,idx); rows=[]
                for mi,modality in enumerate(['PHENO','ANAT','FUNC','Correlation']):
                    a=att[:,mi].reshape(-1,2); bnorm=torch.linalg.vector_norm(base[:,mi],dim=-1); dnorm=torch.linalg.vector_norm(residual[:,mi],dim=-1)
                    rows.append({'modality':modality,'clip_theme_cosine':float(F.cosine_similarity(topic_vectors[mi,0],topic_vectors[mi,1],dim=0).detach().item()),'projected_theme_cosine':float(F.cosine_similarity(theme_features[mi,0],theme_features[mi,1],dim=0).detach().item()),'theme_zero':bool((torch.linalg.vector_norm(theme_features[mi],dim=-1)<=1e-12).any().item()),'theme_nonfinite':bool((~torch.isfinite(theme_features[mi])).any().item()),'attention_theme1_mean':float(a[:,0].mean().detach().item()),'attention_theme2_mean':float(a[:,1].mean().detach().item()),'attention_std':float(a.std(unbiased=False).detach().item()),'attention_abs_diff_mean':float((a[:,0]-a[:,1]).abs().mean().detach().item()),'attention_near_uniform_ratio':float(((a[:,0]-.5).abs()<.01).float().mean().detach().item()),'local_base_norm_mean':float(bnorm.mean().detach().item()),'delta_norm_mean':float(dnorm.mean().detach().item()),'relative_residual_mean':float((dnorm/(bnorm+1e-12)).mean().detach().item()),'nonzero_response_ratio':float((dnorm>1e-12).float().mean().detach().item()),'semantic_nonfinite':bool((~torch.isfinite(delta)).any().item()),'prediction_nonfinite':True,'theme_proj_grad_valid':bool(self._dtsf_grad_finite['theme_proj']),'semantic_fuse_grad_valid':bool(self._dtsf_grad_finite['semantic_fuse']),'config':self.dtsf_config})
                self._dtsf_last_diag=rows
        part_x = local_new
'''
    if src.count(old)!=1: raise RuntimeError("P1 semantic block not found")
    src=src.replace(old,new,1)
    old="""        x = x + torch.cat(combine_part,dim=2)
"""
    new="""        x = x + torch.cat(combine_part,dim=2)
        self._dtsf_last_local_base = local_base.detach() if self.semantic_fuse is not None else None
"""
    if src.count(old)!=1: raise RuntimeError("local fusion point not found")
    return src.replace(old,new,1)

def patch_model(src):
    src=src.replace("import os\n","import os\nimport json\n",1)
    old="""        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
"""
    new="""        dtsf_config=os.environ.get('DTSF_CONFIG','F0')
        if dtsf_config in ('F1','F2','F3','F4'):
            new_parameters=list(self.ModalFusion.theme_proj.parameters())+list(self.ModalFusion.semantic_fuse.parameters()); new_ids={id(p) for p in new_parameters}; base_parameters=[p for p in self.ModalFusion.parameters() if id(p) not in new_ids]
            if sum(p.numel() for p in new_parameters)!=9378: raise RuntimeError('DTSF new parameter count is not 9378')
            self.optimizer_MF=optim.Adam([{'params':base_parameters,'weight_decay':hyperpm.reg},{'params':new_parameters,'weight_decay':0.0}],lr=hyperpm.lr)
        else: self.optimizer_MF=optim.Adam(self.ModalFusion.parameters(),lr=hyperpm.lr,weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
        if dtsf_config in ('F1','F2','F3','F4'):
            with torch.no_grad(): self.ModalFusion.semantic_fuse.weight.zero_()
"""
    if src.count(old)!=1: raise RuntimeError("optimizer block not found")
    src=src.replace(old,new,1)
    old="""            loss.backward()
            
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
"""
    new="""            loss.backward()
            if getattr(self.ModalFusion,'semantic_fuse',None) is not None:
                for name,module in [('semantic_fuse',self.ModalFusion.semantic_fuse),('theme_proj',self.ModalFusion.theme_proj)]:
                    grads=[p.grad for p in module.parameters()]; self.ModalFusion._dtsf_grad_finite[name]=bool(grads and all(g is not None and torch.isfinite(g).all().item() for g in grads))
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
"""
    if src.count(old)!=1: raise RuntimeError("backward block not found")
    src=src.replace(old,new,1)
    old="""        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        
        if mode == 'pre-train':
"""
    new="""        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        if tst and os.environ.get('DTSF_DIAGNOSTIC_PATH') and getattr(self.ModalFusion,'_dtsf_counterfactual',None) is None: self.ModalFusion._dtsf_eval_idx=eval_idx
        else: self.ModalFusion._dtsf_eval_idx=None
        
        if mode == 'pre-train':
"""
    if src.count(old)!=1: raise RuntimeError("eval block not found")
    src=src.replace(old,new,1)
    old="""        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
"""
    new="""        self.ModalFusion._dtsf_prediction_finite=bool(torch.isfinite(prob).all().item())
        if tst and os.environ.get('DTSF_COUNTERFACTUAL_PATH') and getattr(self.ModalFusion,'_dtsf_counterfactual',None) is None:
            self._dtsf_real_probability=prob[eval_idx].detach().exp(); self._dtsf_real_target=self.targ[eval_idx].detach(); self._dtsf_real_pred=prob[eval_idx].max(1)[1].type_as(self.targ[eval_idx]).detach(); self._dtsf_real_acc=float(self._dtsf_real_pred.eq(self._dtsf_real_target).double().mean().item())
        if tst and os.environ.get('DTSF_DIAGNOSTIC_PATH') and getattr(self.ModalFusion,'_dtsf_last_diag',None) is not None:
            for row in self.ModalFusion._dtsf_last_diag: row['prediction_nonfinite'] = not self.ModalFusion._dtsf_prediction_finite
            with open(os.environ['DTSF_DIAGNOSTIC_PATH'],'a',encoding='utf-8') as stream: json.dump({'fold':int(getattr(self,'_dtsf_fold_number',0)),'phase':'BEST','config':os.environ.get('DTSF_CONFIG','F0'),'rows':self.ModalFusion._dtsf_last_diag},stream,ensure_ascii=False); stream.write('\\n')
        self.ModalFusion._dtsf_eval_idx=None
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
"""
    if src.count(old)!=2: raise RuntimeError("prediction block count changed")
    src=src.replace(old,new,1)
    point="    def cal_acc(self, eval_idx):\n"
    method='''    def run_dtsf_counterfactual(self, fold):
        if os.environ.get('DTSF_CONFIG') != 'F1' or not os.environ.get('DTSF_COUNTERFACTUAL_PATH'): return
        self.MessagePassing0.eval(); self.ModalFusion.eval(); model_num,feat1=self.dataLayer(self.feat); feat,_,_=self.MessagePassing0(feat1,model_num,self.targ,self.trn_idx); idx=torch.as_tensor(self.tst_idx,device=self.feat.device,dtype=torch.long); target=self.targ[idx]; base_probability=getattr(self,'_dtsf_real_probability',None); base_pred=getattr(self,'_dtsf_real_pred',None); base_acc=getattr(self,'_dtsf_real_acc',None); rows=[]
        for condition in ('REAL','SWAPPED','POOLED','UNIFORM','OFF'):
            self.ModalFusion._dtsf_counterfactual=None if condition=='REAL' else condition
            with torch.no_grad(): output,_,_=self.ModalFusion(feat)
            probability=output[idx].exp(); prediction=output[idx].max(1)[1].type_as(target); acc=float(prediction.eq(target).double().mean().item()); auc=roc_auc_score(one_hot(target,self.n_class).cpu().numpy(),one_hot(prediction,self.n_class).cpu().numpy())
            if condition=='REAL': base_probability=probability.detach(); base_pred=prediction.detach(); base_acc=acc
            flips=float((prediction-base_pred!=0).double().mean().item()); change=float((probability-base_probability).abs().mean().item()); p_auc=roc_auc_score(target.cpu().numpy(),probability[:,1].cpu().numpy())
            rows.append({'config':'F1','seed':int(self.hyperpm.seed),'fold':int(fold),'condition':condition,'acc':acc,'hard_auc':float(auc),'prediction_flip_rate':flips,'mean_probability_abs_change':change,'acc_change_vs_real':acc-base_acc,'probability_auc':float(p_auc),'real_output_max_abs_diff':float((probability-base_probability).abs().max().item()) if condition=='REAL' else ''})
        self.ModalFusion._dtsf_counterfactual=None
        with open(os.environ['DTSF_COUNTERFACTUAL_PATH'],'a',encoding='utf-8') as stream:
            for row in rows: json.dump(row,stream,ensure_ascii=False); stream.write('\\n')

'''
    if src.count(point)!=1: raise RuntimeError("counterfactual insertion point not found")
    return src.replace(point,method+point,1)

def patch_main(src):
    old="""        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index) #input_data.shape=(598,365) label.ahape=(598,)
"""; new=old+"""        if os.environ.get('DTSF_DIAGNOSTIC_PATH') or os.environ.get('DTSF_COUNTERFACTUAL_PATH'): agent._dtsf_fold_number=clk
"""
    if src.count(old)!=1: raise RuntimeError("fold point not found")
    src=src.replace(old,new,1)
    old="""        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
        sens.append(sen(conf_mat, hyperpm.nclass))
"""; new="""        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
        if os.environ.get('DTSF_COUNTERFACTUAL_PATH'): agent.run_dtsf_counterfactual(clk)
        sens.append(sen(conf_mat, hyperpm.nclass))
"""
    if src.count(old)!=1: raise RuntimeError("test point not found")
    return src.replace(old,new,1)

def expected_patches():
    originals={n:(ROOT/n).read_text(encoding='utf-8') for n in ['main.py','model.py','network.py']}
    patched={'main.py':patch_main(originals['main.py']),'model.py':patch_model(originals['model.py']),'network.py':patch_network(originals['network.py'])}
    chunks=[]
    for n in ['main.py','model.py','network.py']: chunks.extend(difflib.unified_diff(originals[n].splitlines(keepends=True),patched[n].splitlines(keepends=True),fromfile=f'a/{n}',tofile=f'b/{n}'))
    return patched,''.join(chunks)


def validate_patch_and_tokenization(prompts):
    from CLIP.clip import clip
    if tuple(clip.tokenize(prompt_sentences(prompts)).shape) != (32,77): raise RuntimeError("CLIP token shape is not (32,77)")
    patched,text=expected_patches()
    for marker in ["DTSF_CONFIG","theme_proj","semantic_fuse","DTSF_COUNTERFACTUAL_PATH"]:
        if marker not in text: raise RuntimeError(f"missing patch marker {marker}")
    for name,body in patched.items(): compile(body,name,"exec")
    import torch
    torch.manual_seed(7); base=torch.randn(5,4,4,9); proj=torch.nn.Linear(1024,9,bias=False); fuse=torch.nn.Linear(18,9,bias=False)
    with torch.no_grad(): fuse.weight.zero_()
    topics=torch.randn(4,2,1024); theme=torch.nn.functional.normalize(proj(topics),dim=-1); q=torch.nn.functional.normalize(base,dim=-1)
    a=torch.softmax(torch.einsum("nmtf,mcf->nmtc",q,theme),dim=-1)
    dual=torch.cat([a[...,0:1]*base*theme[:,0][None,:,None,:],a[...,1:2]*base*theme[:,1][None,:,None,:]],dim=-1)
    delta=0.1*fuse(dual)
    if not torch.equal(delta,torch.zeros_like(delta)): raise RuntimeError("semantic_fuse zero initialization failed")
    delta.square().mean().backward(retain_graph=True)
    if fuse.weight.grad is None or not torch.isfinite(fuse.weight.grad).all().item(): raise RuntimeError("semantic_fuse gradient check failed")
    fuse.zero_grad()
    with torch.no_grad(): fuse.weight.add_(0.01*torch.randn_like(fuse.weight))
    (0.1*fuse(dual)).square().mean().backward()
    if proj.weight.grad is None or not torch.isfinite(proj.weight.grad).all().item(): raise RuntimeError("theme_proj second-step gradient check failed")
    return text

def freeze_reference():
    source_path=ROOT/"experiment/experiment5_semantic_projection_collapse/results/reference.csv"; p1_path=ROOT/"experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv"
    source=list(csv.DictReader(source_path.open(newline="",encoding="utf-8"))); p1=[r for r in csv.DictReader(p1_path.open(newline="",encoding="utf-8")) if r.get("config")=="P1"]
    if len(source)!=30 or len(p1)!=30: raise RuntimeError("F0 needs 30 reference rows")
    pairs={(r["seed"],r["fold"]):r for r in p1}
    for r in source:
        q=pairs.get((r["seed"],r["fold"]))
        if q is None or any(r.get(m)!=q.get(m) for m in METRICS+["sensitivity_nc","specificity_nc"]): raise RuntimeError(f"F0 reference mismatch {(r['seed'],r['fold'])}")
    fields=["config","seed","fold",*METRICS,"sensitivity_nc","specificity_nc","train_seconds","epoch_cap","early_patience","auc_source","reference_source_sha256"]
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    with REFERENCE_FILE.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for r in source:
            out={k:r.get(k,"") for k in fields}; out["config"]="F0"; out["reference_source_sha256"]=sha256(source_path); writer.writerow(out)
    return source_path,p1_path

def make_manifest(reference_source,p1_path):
    weight=Path.home()/".cache/clip/RN50.pt"
    return {"experiment":"DTSF","schema_version":1,"created_utc":utc_now(),"source_root":str(ROOT),"github_main_commit_at_prepare":github_head(),
            "source_files":{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/x for x in SOURCE_ROOT_FILES]},
            "inputs":{x:sha256(ROOT/x) for x in ["data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]},
            "clip_rn50":{"path":str(weight),"actual_sha256_at_snapshot":sha256(weight) if weight.is_file() else None},
            "prompts_sha256":sha256(PROMPT_FILE),"patch_sha256":sha256(PATCH_FILE),"reference_sha256":sha256(REFERENCE_FILE),"reference_source_sha256":sha256(reference_source),"p1_source_sha256":sha256(p1_path),
            "audit":{"p1_path":"[4,2,4,1024] -> [4,2,4096] -> [4,2,9] -> Softmax-Max","new_parameters":{"theme_proj":9216,"semantic_fuse":162,"total":9378},"zero_initialization":"semantic_fuse.weight zeroed after ModalFusion.apply","optimizer":"base lr=.0038 decay=.11; new lr=.0038 decay=0","original_path_unchanged":True},
            "protocol":{"folds":10,"stratified_random_state":10,"checkpoint_metric":"ACC","epoch_cap":450,"early_patience":200,"auc":"hard_label","seeds":SEEDS,"configs":{"F0":"frozen E0/P1","F1":"DTSF_REAL","F2":"DTSF_SWAPPED","F3":"DTSF_POOLED","F4":"DTSF_UNIFORM"},"gpu_assignments":{f"{c}_seed{s}":g for c,s,g in TASKS}},
            "validation":{"prompt_count":32,"new_fold_rows_expected":120,"diagnostic_rows_expected":480,"counterfactual_rows_expected":150},"environment":{"python":platform.python_version(),"platform":platform.platform()},"run_status":"PREPARED"}

def write_readme():
    atomic_write(EXP/"README.md", """# 实验七：DTSF 双主题语义特征融合

固定实验一 P1 的 32 条 Prompt，保留原始 P1 Softmax-Max、GCN、分类器、损失和训练协议。F0 冻结实验五 E0 的 30 折参考；F1 是真实双主题特征融合；F2 使用 PHENO←FUNC、ANAT←Correlation、FUNC←PHENO、Correlation←ANAT 的新增分支错配；F3 将八个主题平均并复制；F4 使用真实主题但固定注意力 0.5。F1-F4 都新增共享 theme_proj(1024->9) 与 semantic_fuse(18->9)，合计 9378 参数；semantic_fuse 在原始初始化完成后置零，新参数 weight_decay=0。

每个任务按原始顺序完成 10 折，seed 为 0/1/2；两张 GPU 各并行六个任务。F1 的每个最佳 checkpoint 额外做 REAL、SWAPPED、POOLED、UNIFORM、OFF 五种文本反事实推理，不重新训练。

    python experiment/experiment7_dual_theme_semantic_fusion/run.py --prepare
    python experiment/experiment7_dual_theme_semantic_fusion/run.py --all
    python experiment/experiment7_dual_theme_semantic_fusion/run.py --refresh

补丁只应用于临时 worker；根目录源码、实验一至六和 Git 历史不修改。本实验不执行 git commit 或 git push。
""")

def prepare_worktree(work):
    for name in SOURCE_ROOT_FILES: shutil.copy2(ROOT/name,work/name)
    (work/"CLIP").symlink_to(ROOT/"CLIP",target_is_directory=True); (work/"data").symlink_to(ROOT/"data",target_is_directory=True); (work/"attn").mkdir(); (work/"graph").mkdir()
    run=subprocess.run(["patch","--batch","--fuzz=0","-p1","-i",str(PATCH_FILE)],cwd=work,text=True,capture_output=True)
    if run.returncode: raise RuntimeError(f"patch failed: {run.stdout}{run.stderr}")
    patched,_=expected_patches()
    for name,body in patched.items():
        if (work/name).read_text(encoding="utf-8")!=body: raise RuntimeError(f"patched {name} differs")


def attempt_text(path):
    path=Path(path)
    if not path.exists(): return ""
    text=path.read_text(encoding="utf-8",errors="replace"); matches=list(ATTEMPT_RE.finditer(text))
    return text[matches[-1].start():] if matches else text
def parse_attempt_rows(path):
    lines=attempt_text(path).splitlines(); timers=[float(x) for x in re.findall(r"time:\s*([0-9.]+)\s+sec","\n".join(lines))]; out=[]
    for i,line in enumerate(lines):
        match=TEST_RE.search(line)
        if not match: continue
        tail=lines[i+1:]; stop=next((j for j,x in enumerate(tail) if TEST_RE.search(x)),len(tail)); tail=tail[:stop]
        sens=next((NUMBER_RE.findall(x) for x in tail if x.strip().startswith("sens:")),[]); spes=next((NUMBER_RE.findall(x) for x in tail if x.strip().startswith("spes:")),[])
        out.append({"acc":float(match.group(2)),"hard_auc":float(match.group(1)),"sensitivity_asd":float(sens[-2]) if len(sens)>=2 else None,"specificity_asd":float(spes[-2]) if len(spes)>=2 else None,"sensitivity_nc":float(sens[-1]) if len(sens)>=2 else None,"specificity_nc":float(spes[-1]) if len(spes)>=2 else None,"train_seconds":timers[len(out)] if len(timers)>len(out) else None})
    return out
def diagnostic_path(config,seed): return WORK_DIR/f"{config}_seed{seed}.jsonl"
def counterfactual_path(config,seed): return WORK_DIR/f"{config}_seed{seed}_counterfactual.jsonl"
def diagnostic_rows(config,seed):
    path=diagnostic_path(config,seed); out=[]
    if path.exists():
        for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
            try: payload=json.loads(line)
            except json.JSONDecodeError: continue
            for row in payload.get("rows",[]): out.append({"config":config,"seed":seed,"fold":payload.get("fold"),"phase":payload.get("phase"),**row})
    return out
def counterfactual_rows(config,seed):
    path=counterfactual_path(config,seed); out=[]
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out
def diagnostics_valid(config,seed):
    rows=diagnostic_rows(config,seed); keys={(int(r["fold"]),r["modality"]) for r in rows if r.get("fold") is not None}
    return len(rows)==40 and len(keys)==40 and {int(r["fold"]) for r in rows}==set(range(1,11)) and {r["modality"] for r in rows}==set(MODALITIES)
def counterfactual_valid(config,seed):
    rows=counterfactual_rows(config,seed); keys={(int(r["fold"]),r["condition"]) for r in rows if r.get("fold") is not None}
    return config=="F1" and len(rows)==50 and len(keys)==50 and {r["condition"] for r in rows}=={"REAL","SWAPPED","POOLED","UNIFORM","OFF"}
def task_log(config,seed): return LOG_DIR/f"{config}_seed{seed}.log"
def task_end(config,seed):
    path=task_log(config,seed)
    if not path.exists(): return None
    matches=list(TASK_END_RE.finditer(path.read_text(encoding="utf-8",errors="replace")))
    if not matches: return None
    m=matches[-1]; return {"returncode":int(m.group(3)),"folds":int(m.group(4)),"diagnostics":int(m.group(5)),"counterfactual":int(m.group(6))}
def process_running(config,seed):
    try: lines=subprocess.check_output(["ps","-eo","pid=,args="],text=True).splitlines()
    except Exception: return False
    script=Path(__file__).resolve()
    for line in lines:
        try: argv=shlex.split(line.strip().split(None,1)[1])
        except Exception: continue
        if len(argv)>1 and Path(argv[1]).resolve()==script and "--config" in argv and "--seed" in argv and argv[argv.index("--config")+1]==config and argv[argv.index("--seed")+1]==str(seed): return True
    return False
def task_state(config,seed):
    if process_running(config,seed): return "RUNNING",len(parse_attempt_rows(task_log(config,seed)))
    end=task_end(config,seed); folds=len(parse_attempt_rows(task_log(config,seed)))
    if end is None: return "PENDING",folds
    if end["returncode"]!=0: return "FAILED",folds
    ok=end["folds"]==10 and folds==10 and end["diagnostics"]==40 and diagnostics_valid(config,seed)
    if config=="F1": ok=ok and end["counterfactual"]==50 and counterfactual_valid(config,seed)
    return ("COMPLETE" if ok else "EARLY_TERMINATED"),folds

def run_one(config,seed,gpu):
    LOG_DIR.mkdir(parents=True,exist_ok=True); WORK_DIR.mkdir(parents=True,exist_ok=True)
    if task_state(config,seed)[0]=="COMPLETE": print(f"Skip {config} seed={seed}"); return 0
    dp,cp=diagnostic_path(config,seed),counterfactual_path(config,seed)
    if dp.exists(): dp.unlink()
    if cp.exists(): cp.unlink()
    log=task_log(config,seed)
    with tempfile.TemporaryDirectory(prefix=f"DTSF-{config}-seed{seed}-") as directory:
        work=Path(directory); prepare_worktree(work); shell_args=shlex.split((work/SHELL_FILE).read_text(encoding="utf-8"))
        if shell_args[:2]!=["python","main.py"]: raise RuntimeError("unexpected shell command")
        command=[sys.executable,"main.py",*shell_args[2:],"--seed",str(seed),"--gpu","0"]; env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":str(gpu),"DTSF_CONFIG":config,"DTSF_PROMPTS_FILE":str(PROMPT_FILE),"DTSF_DIAGNOSTIC_PATH":str(dp)})
        if config=="F1": env["DTSF_COUNTERFACTUAL_PATH"]=str(cp)
        started=time.time()
        with log.open("a",encoding="utf-8") as stream:
            stream.write(f"\nATTEMPT_START config={config} seed={seed} gpu={gpu} utc={utc_now()}\nCOMMAND {shlex.join(command)}\n"); stream.flush()
            process=subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
            stream.write(f"TASK_END config={config} seed={seed} returncode={process.returncode} folds={len(parse_attempt_rows(log))} diagnostics={len(diagnostic_rows(config,seed))} counterfactual={len(counterfactual_rows(config,seed))} elapsed_seconds={time.time()-started:.1f}\n")
    print(f"{config} seed={seed}: {task_state(config,seed)[0]}"); return process.returncode


def all_fold_rows():
    out=[]
    for config,seed,_ in TASKS:
        for fold,row in enumerate(parse_attempt_rows(task_log(config,seed)),1): out.append({"config":config,"seed":seed,"fold":fold,**row})
    return out
def all_diagnostic_rows():
    out=[]
    for config,seed,_ in TASKS: out.extend(diagnostic_rows(config,seed))
    return out
def all_counterfactual_rows():
    out=[]
    for config,seed,_ in TASKS: out.extend(counterfactual_rows(config,seed))
    return out
def write_outputs():
    rows=all_fold_rows(); fields=["config","seed","fold",*METRICS,"sensitivity_nc","specificity_nc","train_seconds","epoch_cap","early_patience","auc_source"]
    with FOLD_FILE.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            out={k:row.get(k,"") for k in fields}; out.update({"epoch_cap":450,"early_patience":200,"auc_source":"hard_label"})
            for key in METRICS+["sensitivity_nc","specificity_nc","train_seconds"]:
                if out[key] not in ("",None): out[key]=f"{float(out[key]):.6f}"
            writer.writerow(out)
    diagnostics=all_diagnostic_rows(); dfields=["config","seed","fold","phase","modality","clip_theme_cosine","projected_theme_cosine","theme_zero","theme_nonfinite","attention_theme1_mean","attention_theme2_mean","attention_std","attention_abs_diff_mean","attention_near_uniform_ratio","local_base_norm_mean","delta_norm_mean","relative_residual_mean","nonzero_response_ratio","semantic_nonfinite","prediction_nonfinite","theme_proj_grad_valid","semantic_fuse_grad_valid"]
    with DIAG_FILE.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=dfields); writer.writeheader(); writer.writerows({k:r.get(k,"") for k in dfields} for r in diagnostics)
    counterfactual=all_counterfactual_rows(); cfields=["config","seed","fold","condition","acc","hard_auc","prediction_flip_rate","mean_probability_abs_change","acc_change_vs_real","probability_auc","real_output_max_abs_diff"]
    with CF_FILE.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=cfields); writer.writeheader(); writer.writerows({k:r.get(k,"") for k in cfields} for r in counterfactual)
    return rows,diagnostics,counterfactual
def read_csv(path): return list(csv.DictReader(Path(path).open(newline="",encoding="utf-8"))) if Path(path).exists() else []
def seed_summary(rows,config,seed):
    sub=[r for r in rows if r["config"]==config and int(r["seed"])==seed]; out={"config":config,"seed":seed,"n_folds":len(sub)}
    for metric in METRICS:
        vals=[float(r[metric]) for r in sub if r.get(metric) not in ("",None)]; out[metric+"_mean"]=statistics.mean(vals) if vals else None; out[metric+"_sd"]=statistics.pstdev(vals) if vals else None
    return out
def aggregate(rows,config):
    ss=[seed_summary(rows,config,s) for s in SEEDS]; complete=[x for x in ss if x["n_folds"]==10]; out={"config":config,"n_seeds":len(complete)}
    for metric in METRICS:
        vals=[x[metric+"_mean"] for x in complete if x[metric+"_mean"] is not None]; out[metric+"_mean"]=statistics.mean(vals) if vals else None; out[metric+"_sd"]=statistics.pstdev(vals) if len(vals)>1 else None
    return out
def fmt(value,sd=None): return "-" if value is None else f"{value*100:.2f}"+(f"±{sd*100:.2f}%" if sd is not None else "%")
def pair_counts(rows,left,right,metric):
    a={(int(r["seed"]),int(r["fold"])):float(r[metric]) for r in rows if r["config"]==left}; b={(int(r["seed"]),int(r["fold"])):float(r[metric]) for r in rows if r["config"]==right}; d=[a[k]-b[k] for k in a.keys()&b.keys()]
    return sum(x>1e-12 for x in d),sum(abs(x)<=1e-12 for x in d),sum(x< -1e-12 for x in d)

def report_text(rows,diagnostics,counterfactual):
    reference=read_csv(REFERENCE_FILE); combined=[{**r,"config":"F0"} for r in reference]+rows; done=sum(task_state(c,s)[0]=="COMPLETE" for c,s,_ in TASKS)
    lines=["# 实验七：DTSF 双主题语义特征融合","",f"状态：**{'全部完成' if done==12 else '执行中或未完成'}**；F1-F4 完成任务 {done}/12，新折 {len(rows)}/120，机制诊断 {len(diagnostics)}/480，反事实 {len(counterfactual)}/150。","","## 主结果","","| Config | ACC | Hard-AUC | SEN-ASD | SPE-ASD |","|---|---:|---:|---:|---:|"]
    for config in ALL_CONFIGS:
        a=aggregate(combined,config); lines.append(f"| {config} | {fmt(a.get('acc_mean'),a.get('acc_sd'))} | {fmt(a.get('hard_auc_mean'),a.get('hard_auc_sd'))} | {fmt(a.get('sensitivity_asd_mean'))} | {fmt(a.get('specificity_asd_mean'))} |")
    lines += ["","## 设计与审计","", "F0 冻结实验五 E0/P1；F1 保留原始 P1 Softmax-Max 并加入双主题特征残差；F2 只在新增分支跨模态错配；F3 将八个主题平均后复制；F4 使用真实主题但将双主题注意力固定为 0.5。F1-F4 均新增 9378 参数，semantic_fuse 在原始初始化完成后置零。原始 GCN、分类器、损失、学习率、epoch、早停和折协议保持不变。","", "## 每个 seed 的十折均值","", "| Config | seed | ACC | Hard-AUC | SEN-ASD | SPE-ASD |","|---|---:|---:|---:|---:|---:|"]
    for config in ALL_CONFIGS:
        for seed in SEEDS:
            a=seed_summary(combined,config,seed); lines.append(f"| {config} | {seed} | {fmt(a.get('acc_mean'),a.get('acc_sd'))} | {fmt(a.get('hard_auc_mean'),a.get('hard_auc_sd'))} | {fmt(a.get('sensitivity_asd_mean'))} | {fmt(a.get('specificity_asd_mean'))} |")
    lines += ["","## 配对比较",""]
    for left,right in [("F1","F0"),("F1","F2"),("F1","F3"),("F1","F4"),("F2","F0"),("F3","F0"),("F4","F0")]:
        lines.append(f"### {left} − {right}")
        for metric in METRICS:
            w,t,l=pair_counts(combined,left,right,metric); by=[]
            for seed in SEEDS:
                a,b=seed_summary(combined,left,seed),seed_summary(combined,right,seed)
                if a["n_folds"]==b["n_folds"]==10: by.append(f"seed{seed} {(a[metric+'_mean']-b[metric+'_mean'])*100:+.2f}")
            lines.append(f"- {metric}: {', '.join(by)}；胜/平/负 {w}/{t}/{l}。")
    lines += ["","## 机制诊断","","每个配置×seed×fold×模态一行；四个局部 Token 是潜在模态特征，不解释为具体 ROI。","","| Config | 模态 | CLIP cos | projected cos | attn1 | attn2 | attn std | |a1-a2| | near uniform | base norm | delta norm | relative residual | nonzero response | grad theme/fuse |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        for modality in MODALITIES:
            sub=[r for r in diagnostics if r["config"]==config and r["modality"]==modality]
            def mean(key):
                vals=[float(r[key]) for r in sub if r.get(key) not in ("",None)]; return statistics.mean(vals) if vals else math.nan
            tg=sum(str(r.get("theme_proj_grad_valid")).lower()=="true" for r in sub); fg=sum(str(r.get("semantic_fuse_grad_valid")).lower()=="true" for r in sub)
            lines.append(f"| {config} | {modality} | {mean('clip_theme_cosine'):.4f} | {mean('projected_theme_cosine'):.4f} | {mean('attention_theme1_mean'):.4f} | {mean('attention_theme2_mean'):.4f} | {mean('attention_std'):.4f} | {mean('attention_abs_diff_mean'):.4f} | {mean('attention_near_uniform_ratio'):.4f} | {mean('local_base_norm_mean'):.4f} | {mean('delta_norm_mean'):.4f} | {mean('relative_residual_mean'):.4f} | {mean('nonzero_response_ratio'):.4f} | {tg}/{len(sub)} / {fg}/{len(sub)} |")
    lines += ["","## F1 反事实文本干预","","同一最佳 checkpoint、同一受试者输入和原始 P1 路径下，替换或关闭新增辅助分支。","","| condition | ACC | Hard-AUC | flip rate | mean probability change | ACC change vs REAL | probability-AUC | REAL max diff |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in ["REAL","SWAPPED","POOLED","UNIFORM","OFF"]:
        sub=[r for r in counterfactual if r.get("condition")==condition]
        def cm(key):
            vals=[float(r[key]) for r in sub if r.get(key) not in ("",None)]; return statistics.mean(vals) if vals else math.nan
        lines.append(f"| {condition} | {cm('acc')*100:.2f}% | {cm('hard_auc')*100:.2f}% | {cm('prediction_flip_rate'):.4f} | {cm('mean_probability_abs_change'):.6f} | {cm('acc_change_vs_real')*100:+.2f} pp | {cm('probability_auc')*100:.2f}% | {cm('real_output_max_abs_diff') if condition=='REAL' else '-'} |")
    baseline=aggregate(combined,"F0"); summaries={c:aggregate(combined,c) for c in CONFIGS}
    def diff(config,metric):
        x,y=summaries[config].get(metric+"_mean"),baseline.get(metric+"_mean"); return None if x is None or y is None else (x-y)*100
    lines += ["","## 结论",""]
    for config in CONFIGS:
        acc_diff = diff(config, 'acc'); auc_diff = diff(config, 'hard_auc')
        acc_text = f"{acc_diff:+.2f}" if acc_diff is not None else '未完整'
        auc_text = f"{auc_diff:+.2f}" if auc_diff is not None else '未完整'
        lines.append(f"- {config} 相对 F0：ACC {acc_text} pp，Hard-AUC {auc_text} pp。")
    if diagnostics:
        by={m:statistics.mean(float(r["relative_residual_mean"]) for r in diagnostics if r["modality"]==m) for m in MODALITIES}; leader=max(by,key=by.get)
        lines.append(f"- 平均相对残差最大的模态为 {leader}；这表示辅助响应强度，不等同于具体脑区。")
        lines.append(f"- 新分支非零响应比例均值为 {statistics.mean(float(r['nonzero_response_ratio']) for r in diagnostics):.4f}。")
    if counterfactual:
        swaps=[float(r["prediction_flip_rate"]) for r in counterfactual if r["condition"]=="SWAPPED"]; lines.append(f"- F1 错配主题的平均预测翻转率为 {statistics.mean(swaps):.4f}；预测敏感性不等于医学语义正确性。")
    gpu0 = ', '.join(f'{c} seed{s}' for (c, s), g in GPU_ASSIGNMENTS.items() if g == 0)
    gpu1 = ', '.join(f'{c} seed{s}' for (c, s), g in GPU_ASSIGNMENTS.items() if g == 1)
    lines += ["", "## 运行状态", f"GPU0（6任务）：{gpu0}", f"GPU1（6任务）：{gpu1}", f"实验目录：{EXP}。根目录源码与实验一至六未修改；本实验不执行 git commit 或 git push。"]
    return "\n".join(lines)+"\n"


def update_manifest(status=None):
    if not MANIFEST_FILE.exists(): return
    manifest=json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if status is not None: manifest["run_status"]=status
    manifest["task_status"]={f"{c}_seed{s}":task_state(c,s)[0] for c,s,_ in TASKS}
    manifest["completed"]={"folds":len(all_fold_rows()),"diagnostics":len(all_diagnostic_rows()),"counterfactual":len(all_counterfactual_rows())}
    write_json(MANIFEST_FILE,manifest)
def update_handoff():
    if os.environ.get("DTSF_SKIP_HANDOFF") == "1": return
    start,end="<!-- DTSF EXPERIMENT 7 START -->","<!-- DTSF EXPERIMENT 7 END -->"; old=HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else ""
    complete=len(read_csv(FOLD_FILE))==120 and len(read_csv(DIAG_FILE))==480 and len(read_csv(CF_FILE))==150
    status="已完成 12/12 任务、120 折、480 条诊断与 150 条反事实" if complete else "已准备运行器与冻结参考，训练待执行"
    block=f"{start}\n\n实验七 DTSF（{status}）：F0 冻结实验五 E0/P1；F1 真实双主题融合，F2 跨模态错配，F3 主题平均，F4 固定 0.5 注意力。F1-F4 各 seed 0/1/2 十折，两张 GPU 各六任务并行；结果见 {EXP}/results/report.md。本实验不执行 git commit 或 git push。\n\n{end}"
    pattern=re.compile(re.escape(start)+r".*?"+re.escape(end),re.S); atomic_write(HANDOFF,pattern.sub(block,old) if pattern.search(old) else old.rstrip()+"\n\n"+block+"\n")
def prepare():
    prompts,_=build_prompts(); ref,p1=freeze_reference(); patch_text=validate_patch_and_tokenization(prompts); atomic_write(PATCH_FILE,patch_text); write_readme(); write_json(MANIFEST_FILE,make_manifest(ref,p1))
    with tempfile.TemporaryDirectory(prefix="DTSF-preflight-") as directory: prepare_worktree(Path(directory))
    update_handoff(); print("DTSF prepare complete: prompts=32, F0=30, new params=9378, patch/preflight verified")
def run_all():
    LOG_DIR.mkdir(parents=True,exist_ok=True); WORK_DIR.mkdir(parents=True,exist_ok=True); scheduler=LOG_DIR/"scheduler.log"
    with scheduler.open("a",encoding="utf-8") as stream:
        stream.write(f"SCHEDULER_START utc={utc_now()} assignments={GPU_ASSIGNMENTS}\n")
        try:
            gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.total,memory.used","--format=csv,noheader"],text=True,stderr=subprocess.STDOUT).strip(); stream.write("GPU_STATUS "+gpu.replace("\n"," | ")+"\n")
        except Exception as exc: stream.write(f"GPU_STATUS unavailable={exc}\n")
    processes=[subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--one","--config",c,"--seed",str(s),"--gpu",str(g)],cwd=ROOT) for c,s,g in TASKS]
    codes=[p.wait() for p in processes]; rows,diagnostics,counterfactual=write_outputs()
    complete=not any(code!=0 for code in codes) and (len(rows),len(diagnostics),len(counterfactual))==(120,480,150)
    update_manifest("COMPLETE" if complete else "INCOMPLETE"); atomic_write(REPORT_FILE,report_text(rows,diagnostics,counterfactual)); update_handoff()
    if any(code!=0 for code in codes): raise SystemExit(max(codes))
    if not complete: raise RuntimeError(f"expected 120/480/150, got {len(rows)}/{len(diagnostics)}/{len(counterfactual)}")
    print(f"DTSF all complete: {len(rows)} folds, {len(diagnostics)} diagnostics, {len(counterfactual)} counterfactual rows")
def refresh():
    rows,diagnostics,counterfactual=write_outputs(); atomic_write(REPORT_FILE,report_text(rows,diagnostics,counterfactual)); update_manifest("COMPLETE" if (len(rows),len(diagnostics),len(counterfactual))==(120,480,150) else "INCOMPLETE"); update_handoff(); print(f"DTSF report refreshed: {len(rows)} folds, {len(diagnostics)} diagnostics, {len(counterfactual)} counterfactual rows")
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare",action="store_true"); parser.add_argument("--all",action="store_true"); parser.add_argument("--refresh",action="store_true"); parser.add_argument("--one",action="store_true"); parser.add_argument("--config",choices=CONFIGS); parser.add_argument("--seed",type=int,choices=SEEDS); parser.add_argument("--gpu",type=int,choices=[0,1]); args=parser.parse_args()
    if args.prepare: prepare()
    elif args.all: run_all()
    elif args.refresh: refresh()
    elif args.one:
        if args.config is None or args.seed is None or args.gpu is None: parser.error("--one requires --config --seed --gpu")
        raise SystemExit(run_one(args.config,args.seed,args.gpu))
    else: parser.error("choose --prepare, --all, --refresh, or --one")
if __name__=="__main__": main()

