"""Build the disposable B0-A runtime source with a parity-preserving bypass."""
from __future__ import annotations

import ast
import difflib
import shutil
from pathlib import Path

SOURCE_NAMES = ("main.py", "model.py", "network.py", "layers.py", "utils.py")
MARKER = "        text_features = self.prompt_text_features\n"
BYPASS = (
    "        if getattr(self, 'qcar33_b0_bypass', False):\n"
    "            x = x.view(bs, -1)\n"
    "            attn_embedding = attn.view(bs, -1)\n"
    "            output, hidden = self.Outputlayer(x, attn_embedding)\n"
    "            return output, hidden, attn_map\n"
)


def patched_network(source: str) -> str:
    assert source.count(MARKER) == 1
    result = source.replace(MARKER, BYPASS + MARKER, 1)
    ast.parse(result)
    return result


def emit(root: Path, destination: Path, patch_path: Path):
    root, destination, patch_path = Path(root), Path(destination), Path(patch_path)
    assert root.resolve() != destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    diffs = []
    for name in SOURCE_NAMES:
        original = (root / name).read_text(encoding="utf-8")
        modified = patched_network(original) if name == "network.py" else original
        target = destination / name
        if target.exists():
            assert target.read_text(encoding="utf-8") == modified, f"runtime source changed: {name}"
        else:
            shutil.copy2(root / name, target)
            if name == "network.py":
                target.write_text(modified, encoding="utf-8")
        diffs.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                          fromfile="a/" + name, tofile="b/" + name))
    clip = destination / "CLIP"
    if not clip.exists():
        clip.symlink_to(root / "CLIP", target_is_directory=True)
    patch = "".join(diffs)
    if patch_path.exists():
        assert patch_path.read_text(encoding="utf-8") == patch
    else:
        patch_path.write_text(patch, encoding="utf-8")
    return destination
