"""Exact patches on disposable copies only; root files are never written."""
from __future__ import annotations
import ast
import difflib
from pathlib import Path
import numpy as np
import torch
from e28_method import CONFIGS,ImagingAnchoredAlignment

_CONTEXT=None

def configure(*,config,fold,raw,train,test,labels_train,site,pheno_indices,cache):
    global _CONTEXT
    assert config in CONFIGS and 1<=fold<=10
    assert np.asarray(labels_train).shape==(len(train),)
    _CONTEXT=dict(config=config,fold=fold,raw=raw,train=train,test=test,
                  y_train=labels_train,site=site,pheno_indices=pheno_indices,cache=cache)

def install(agent):
    assert _CONTEXT is not None
    c=_CONTEXT
    agent.ModalFusion.e28_config=c['config']
    if c['config'] in CONFIGS[:2]:return
    assert np.array_equal(c['train'],agent.trn_idx) and np.array_equal(c['test'],agent.tst_idx)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(280000+int(agent.hyperpm.seed))
        branch=ImagingAnchoredAlignment(**c).to(agent.dev)
    assert torch.count_nonzero(branch.alpha)==0
    agent.ModalFusion.ia_fdsa=branch
    agent.optimizer_MF.add_param_group(dict(params=branch.parameters(),lr=agent.hyperpm.lr,weight_decay=0.))
    groups=agent.optimizer_MP0.param_groups+agent.optimizer_MF.param_groups
    ids=[id(p) for g in groups for p in g['params']]
    actual={id(p) for m in (agent.MessagePassing0,agent.ModalFusion) for p in m.parameters()}
    assert len(ids)==len(set(ids)) and set(ids)==actual

def patch_network(original):
    s=original
    sig='    def forward(self, x):\n        bs = x.size(0)'
    assert s.count(sig)==1
    s=s.replace(sig,'    def forward(self, x, raw_input=None):\n        bs = x.size(0)',1)
    first='        text_features = self.prompt_text_features\n'
    end='        # 对最后一个维度进行平均池化\n'
    assert s.count(first)==s.count(end)==1
    a,b=s.index(first),s.index(end)
    original_spgs=s[a:b]
    assert original_spgs.count('part_x = similarity_max.unsqueeze(-1) * part_x')==1
    wrapped="        if getattr(self, 'e28_config', 'E0_P0') == 'E0_P0':\n"
    wrapped+=''.join('    '+line if line.strip() else line for line in original_spgs.splitlines(keepends=True))
    wrapped+=("        else:\n"
              "            if hasattr(self, 'ia_fdsa'):\n"
              "                assert raw_input is not None and raw_input.shape == (871, 320)\n"
              "                x = self.ia_fdsa(x, raw_input)\n"
              "            part_x = x.view(871, 4, 4, 9)\n")
    s=s[:a]+wrapped+s[b:]
    ast.parse(s)
    # RES and classifier are byte-identical from this anchor to the end of PLN.
    assert s[s.index(end):s.index('class GraphLearn')]==original[original.index(end):original.index('class GraphLearn')]
    return s

def patch_model(original):
    s=original
    imp='from network import *\n'
    init='        self.ModalFusion.apply(my_weight_init)'
    train='            prob,_,_ = self.ModalFusion(feat)'
    loss='            loss = cls_loss + all_graph_loss'
    for anchor in (imp,init,train,loss):assert s.count(anchor)==1,anchor
    s=s.replace(imp,imp+'import e28_integration\n',1)
    s=s.replace(init,init+'\n        e28_integration.install(self)',1)
    s=s.replace(train,'            prob,_,_ = self.ModalFusion(feat, raw_input=self.feat)',1)
    s=s.replace(loss,loss+"\n            if hasattr(self.ModalFusion, 'ia_fdsa'):\n                loss = loss + self.ModalFusion.ia_fdsa.semantic_loss()",1)
    a=s.index('    def _print_acc(');b=s.index('    def cal_acc(',a)
    block=s[a:b]
    assert block.count('self.ModalFusion(feat)')==2
    block=block.replace('self.ModalFusion(feat)','self.ModalFusion(feat, raw_input=self.feat)')
    s=s[:a]+block+s[b:]
    ast.parse(s)
    return s

def emit(root,out,patch_path=None):
    root,out=Path(root),Path(out)
    assert root.resolve()!=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    diffs=[]
    for name in ('model.py','network.py','layers.py','utils.py'):
        original=(root/name).read_text()
        modified=patch_model(original) if name=='model.py' else patch_network(original) if name=='network.py' else original
        path=out/name
        if path.exists():assert path.read_text()==modified,'runtime source changed: '+name
        else:path.write_text(modified)
        diffs.extend(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    link=out/'CLIP'
    if not link.exists():link.symlink_to(root/'CLIP',target_is_directory=True)
    if patch_path:
        p=Path(patch_path);text=''.join(diffs)
        if p.exists():assert p.read_text()==text
        else:p.write_text(text)
    return out
