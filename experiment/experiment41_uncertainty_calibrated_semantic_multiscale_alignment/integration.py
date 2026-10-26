"""Private-source PEN bypass for the B0-A anchor."""
from __future__ import annotations

import ast
import difflib
import shutil
from pathlib import Path

SOURCE_NAMES = ("main.py", "model.py", "network.py", "layers.py", "utils.py")
MARKER = "        text_features = self.prompt_text_features\n"
BYPASS = (
    "        if getattr(self, 'ucsma41_b0_bypass', False):\n"
    "            x = x.view(bs, -1)\n"
    "            attn_embedding = attn.view(bs, -1)\n"
    "            output, hidden = self.Outputlayer(x, attn_embedding)\n"
    "            return output, hidden, attn_map\n"
)


def patched_network(source: str) -> str:
    if source.count(MARKER) != 1:
        raise AssertionError("Private PEN bypass marker changed")
    patched = source.replace(MARKER, BYPASS + MARKER, 1)
    ast.parse(patched)
    return patched


def emit(root: Path, destination: Path, patch_path: Path) -> str:
    root, destination = Path(root), Path(destination)
    if root.resolve() == destination.resolve():
        raise AssertionError("Runtime source must be isolated")
    destination.mkdir(parents=True, exist_ok=True)
    diffs = []
    for name in SOURCE_NAMES:
        original = (root / name).read_text(encoding="utf-8")
        modified = patched_network(original) if name == "network.py" else original
        target = destination / name
        target.write_text(modified, encoding="utf-8")
        diffs.extend(difflib.unified_diff(
            original.splitlines(True), modified.splitlines(True),
            fromfile="a/" + name, tofile="b/" + name))
    clip = destination / "CLIP"
    if clip.exists() or clip.is_symlink():
        clip.unlink()
    clip.symlink_to(root / "CLIP", target_is_directory=True)
    patch = "".join(diffs)
    Path(patch_path).write_text(patch, encoding="utf-8")
    return patch
