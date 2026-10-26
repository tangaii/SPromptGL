"""Minimal isolated PLN bypass and graph-observation hooks; root is read-only."""
from __future__ import annotations
import ast
import difflib
from pathlib import Path


def patch_network(source):
    a = source.index("        text_features = self.prompt_text_features\n")
    b = source.index("        x = x.view(bs, -1)\n", a)
    source = source[:a] + source[b:]
    marker = "        x = x.view(bs, -1)\n"
    source = source.replace(marker, marker +
        "        self.cam40 = x\n"
        "        if getattr(self, 'hcer', None) is not None:\n"
        "            x, self.hcer40 = self.hcer(x, **self.context40)\n", 1)
    # Graphs remain calculated by the exact original GraphLearn once per forward.
    # A diagnostic-only flag replaces those results with fixed original graphs.
    for k in range(4):
        marker = f"                adj = self.GraphConstruct{k}(x[i])\n"
        assert source.count(marker) == 1
        source = source.replace(marker, marker +
            f"                if getattr(self, 'fixed_graph40', None) is not None:\n"
            f"                    adj = self.fixed_graph40[{k}]\n", 1)
    ast.parse(source)
    return source


def emit(root, destination, patch_file):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    diffs = []
    for name in ("network.py", "model.py", "layers.py", "utils.py"):
        original = (root / name).read_text()
        changed = patch_network(original) if name == "network.py" else original
        target = destination / name
        if target.exists():
            assert target.read_text() == changed, "isolated source changed: " + name
        else:
            target.write_text(changed)
        diffs.extend(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                         fromfile="a/" + name, tofile="b/" + name))
    link = destination / "CLIP"
    if not link.exists():
        link.symlink_to(root / "CLIP", target_is_directory=True)
    value = "".join(diffs)
    patch_file = Path(patch_file)
    if patch_file.exists():
        assert patch_file.read_text() == value
    else:
        patch_file.write_text(value)
    return value
