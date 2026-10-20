"""Exact surgical integration, on experiment-local disposable root copies only."""
from __future__ import annotations
import ast,difflib
from pathlib import Path
import numpy as np
import torch
from method import CONFIGS,SemanticSelectiveAlignment

_CONTEXT=None

def configure(**values):
    global _CONTEXT
    assert values['config'] in CONFIGS+('P0','E1')
    assert np.asarray(values['labels_train']).shape==(len(values['train']),)
    _CONTEXT=values

def install(agent):
    assert _CONTEXT is not None
    c=_CONTEXT;agent.ModalFusion.e29_config=c['config']
    if c['config'] in ('P0','E1'):return
    assert np.array_equal(c['train'],agent.trn_idx) and np.array_equal(c['test'],agent.tst_idx)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(290000+int(agent.hyperpm.seed))
        branch=SemanticSelectiveAlignment(**c).to(agent.dev)
    assert branch.alpha_sem.item()==branch.alpha_aux.item()==0
    agent.ModalFusion.semantic29=branch
    agent.optimizer_MF.add_param_group(dict(params=list(branch.parameters()),lr=agent.hyperpm.lr,weight_decay=0.))
    ids=[id(p) for g in agent.optimizer_MP0.param_groups+agent.optimizer_MF.param_groups for p in g['params']]
    actual={id(p) for m in (agent.MessagePassing0,agent.ModalFusion) for p in m.parameters()}
    assert len(ids)==len(set(ids)) and set(ids)==actual

def patch_network(original):
    signature='    def forward(self, x):\n        bs = x.size(0)'
    start='        text_features = self.prompt_text_features\n'
    end='        # 对最后一个维度进行平均池化\n'
    assert all(original.count(s)==1 for s in (signature,start,end))
    s=original.replace(signature,'    def forward(self, x, raw_input=None):\n        bs = x.size(0)',1)
    a,b=s.index(start),s.index(end);spgs=s[a:b]
    assert spgs.count('part_x = similarity_max.unsqueeze(-1) * part_x')==1
    new="        if getattr(self, 'e29_config', 'P0') == 'P0':\n"
    new+=''.join('    '+line if line.strip() else line for line in spgs.splitlines(True))
    new+=("        else:\n"
          "            if hasattr(self, 'semantic29'):\n"
          "                assert raw_input is not None\n"
          "                x = self.semantic29(x, raw_input)\n"
          "            part_x = x.view(871, 4, 4, 9)\n")
    s=s[:a]+new+s[b:];ast.parse(s)
    assert s[s.index(end):]==original[original.index(end):],'RES/classifier/GFR changed'
    return s

def patch_model(original):
    anchors=('from network import *\n','        self.ModalFusion.apply(my_weight_init)',
             '            prob,_,_ = self.ModalFusion(feat)','            loss = cls_loss + all_graph_loss')
    assert all(original.count(s)==1 for s in anchors)
    s=original.replace(anchors[0],anchors[0]+'import integration\n',1)
    s=s.replace(anchors[1],anchors[1]+'\n        integration.install(self)',1)
    s=s.replace(anchors[2],'            prob,_,_ = self.ModalFusion(feat, raw_input=self.feat)',1)
    s=s.replace(anchors[3],anchors[3]+"\n            if hasattr(self.ModalFusion, 'semantic29'):\n                loss = loss + self.ModalFusion.semantic29.extra_loss(self.targ)",1)
    a,b=s.index('    def _print_acc('),s.index('    def cal_acc(');part=s[a:b]
    assert part.count('self.ModalFusion(feat)')==2
    s=s[:a]+part.replace('self.ModalFusion(feat)','self.ModalFusion(feat, raw_input=self.feat)')+s[b:]
    ast.parse(s);return s

def emit(root,out,patch_path=None):
    root,out=Path(root),Path(out);assert out.resolve()!=root.resolve();out.mkdir(parents=True,exist_ok=True)
    patch=[]
    for name in ('model.py','network.py','layers.py','utils.py'):
        original=(root/name).read_text()
        modified=patch_model(original) if name=='model.py' else patch_network(original) if name=='network.py' else original
        p=out/name
        if p.exists():assert p.read_text()==modified,'Runtime source changed '+name
        else:p.write_text(modified)
        patch.extend(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    if not (out/'CLIP').exists():(out/'CLIP').symlink_to(root/'CLIP',target_is_directory=True)
    if patch_path:
        p=Path(patch_path);value=''.join(patch)
        if p.exists():assert p.read_text()==value
        else:p.write_text(value)
    return out
