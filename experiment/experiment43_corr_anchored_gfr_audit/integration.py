"""Minimal private source emission; no CLIP import, encoding or root mutation."""
from pathlib import Path
import ast
import difflib


def patch_network(text):
    a = text.index('from CLIP.clip import clip\n')
    b = text.index('class MLP', a)
    # Eight immutable legacy buffers are provided from the validated initial anchor.
    text = text[:a] + '''device = "cuda" if torch.cuda.is_available() else "cpu"
_prompt_text_features = None

def get_prompt_text_features():
    assert _prompt_text_features is not None, "E43 requires validated cached legacy initialization buffer"
    return _prompt_text_features

''' + text[b:]
    a = text.index('        text_features = self.prompt_text_features\n')
    b = text.index('        x = x.view(bs, -1)\n', a)
    text = text[:a] + text[b:]
    old = '        score = torch.matmul(x_norm, x_norm.T)\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '        self.last_score43 = score\n', 1)
    ast.parse(text)
    return text


def emit(root, destination, patch_path):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    diffs = []
    for name in ['network.py','model.py','layers.py','utils.py']:
        old = (root/name).read_text()
        new = patch_network(old) if name == 'network.py' else old
        path = destination/name
        if path.exists():assert path.read_text() == new, 'private source drift'
        else:path.write_text(new)
        diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    Path(patch_path).write_text(''.join(diffs))
