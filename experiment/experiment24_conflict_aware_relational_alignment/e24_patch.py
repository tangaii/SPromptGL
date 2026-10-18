"""Exact surgical integration only in experiment-local copy of model.py."""
import ast
from pathlib import Path

def patch(src):
    assert src.count('        self.ModalFusion.apply(my_weight_init)')==1
    assert src.count('            loss = cls_loss + all_graph_loss')==1
    assert src.count('        elif mode == \'simple-2\':')==1
    assert src.count('            prob,_,_ = self.ModalFusion(feat)')==1
    old='from network import *\n'
    assert src.count(old)==1
    src=src.replace(old,old+'import e24_runtime\n',1)
    old='        self.ModalFusion.apply(my_weight_init)'
    src=src.replace(old,old+'\n        e24_runtime.install(self)',1)
    old='            loss = cls_loss + all_graph_loss'
    new=old+('\n            if hasattr(self, "_e24") and self._e24 is not None:\n'
             '                loss = loss + self._e24.loss(self.MessagePassing0, feat1,\n'
             '                    self.targ, self.trn_idx, cls_loss)')
    src=src.replace(old,new,1)
    ast.parse(src)
    assert src.count('e24_runtime.install(self)')==1
    return src

def emit(root,output):
    root=Path(root);output=Path(output)
    assert root.resolve()!=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    for n in ('model.py','network.py','layers.py','utils.py'):
        body=(root/n).read_text()
        (output/n).write_text(patch(body) if n=='model.py' else body)

if __name__=='__main__':
    import sys
    emit(sys.argv[1],sys.argv[2])

