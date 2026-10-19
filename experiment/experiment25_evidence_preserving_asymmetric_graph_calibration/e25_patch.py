"""Apply a one-line bounded CORR graph gate in a COPY of network.py.
Original model.py, all graph constructions, graph losses and PLN unchanged.
"""
from pathlib import Path
import ast
FILES=('model.py','network.py','layers.py','utils.py')

def patch_network(src):
    import_anchor='from utils import *\n'
    graph_line='                adj = self.GraphConstruct3(x[i])'
    assert src.count(import_anchor)==1
    assert src.count(graph_line)==1
    src=src.replace(import_anchor,import_anchor+'import e25_runtime\n',1)
    src=src.replace(graph_line,graph_line+'\n                adj = e25_runtime.calibrate(adj, x, index)',1)
    ast.parse(src)
    assert src.count('e25_runtime.calibrate(')==1
    return src

def emit(root,dest):
    root=Path(root);dest=Path(dest)
    assert root.resolve()!=dest.resolve()
    dest.mkdir(parents=True,exist_ok=True)
    for fname in FILES:
        raw=(root/fname).read_text()
        updated=patch_network(raw) if fname=='network.py' else raw
        (dest/fname).write_text(updated)
        assert (dest/fname).read_text()==updated

