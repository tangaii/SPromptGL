"""Minimal immutable-root surgical patches: original SPGS and RES untouched."""
from pathlib import Path
import ast
FILES=('model.py','network.py','layers.py','utils.py')

def patch_network(s):
    anchor='from utils import *\n'
    mlp='        text_features = self.MLP(text_features)'
    out='        output, hidden = self.Outputlayer(x, attn_embedding)'
    assert s.count(anchor)==s.count(mlp)==s.count(out)==1
    s=s.replace(anchor,anchor+'import e26_runtime\n',1)
    s=s.replace(mlp,
        '        e26_prompt_input = e26_runtime.prototype(text_features)\n'
        +mlp+'\n'
        +'        e26_prompt_projected = e26_runtime.signed_project(self.MLP,e26_prompt_input)',1)
    s=s.replace(out,out+'\n        output = e26_runtime.forward(self, output, part_x_norm, e26_prompt_projected)',1)
    ast.parse(s)
    return s

def patch_model(s):
    anchor='from network import *\n'
    original='            loss = cls_loss + all_graph_loss'
    assert s.count(anchor)==1 and s.count(original)==1
    # Strictly add to simple-2 loss; no change to val=test or graph loss.
    s=s.replace(anchor,anchor+'import e26_runtime\n',1)
    s=s.replace(original,original+'\n            loss = loss + e26_runtime.loss(self.ModalFusion,self.targ,self.trn_idx)',1)
    ast.parse(s)
    return s

def emit(root,out):
    root=Path(root);out=Path(out);assert root.resolve()!=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        s=(root/name).read_text()
        if name=='model.py':s=patch_model(s)
        if name=='network.py':s=patch_network(s)
        (out/name).write_text(s)
        assert (out/name).read_text()==s

def self_test():
    n='from utils import *\n\nclass PLN:\n    def forward(self):\n        text_features = self.MLP(text_features)\n        output, hidden = self.Outputlayer(x, attn_embedding)\n'
    m='from network import *\n\nclass H:\n    def run_epoch(self):\n            loss = cls_loss + all_graph_loss\n'
    assert 'e26_runtime.forward' in patch_network(n)
    assert 'e26_runtime.loss' in patch_model(m)
    print('E26_PATCH_SELF_TEST_PASS')
if __name__=='__main__':self_test()
