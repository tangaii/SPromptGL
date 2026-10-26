"""Generate one isolated, physically PEN-free network; never import CLIP."""
from pathlib import Path
import difflib

def prepare(root, here):
    src=here/'.runtime/sources'
    original=(root/'network.py').read_text()
    # Keep the original imports except CLIP, then the untouched graph classes.
    head=original[:original.index('from CLIP.clip import clip')]
    graph=original[original.index('class GraphLearn('):]
    graph=graph.replace('        score = torch.matmul(x_norm, x_norm.T)\n',
                        '        score = torch.matmul(x_norm, x_norm.T)\n        self.score431 = score\n')
    revised=head+graph
    assert 'CLIP' not in revised and 'class PLN' not in revised and 'class MLP' not in revised
    (src/'network.py').write_text(revised)
    for name in ['layers.py','utils.py']:(src/name).write_bytes((root/name).read_bytes())
    (here/'integration.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile='root/network.py',tofile='.runtime/sources/network.py')))
