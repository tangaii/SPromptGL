"""Build an isolated E32-B0-compatible PLN with a post-CAM LCHER hook."""
from __future__ import annotations

import ast
import difflib
import shutil
from pathlib import Path

import integration as e32


def emit(root: Path, out: Path, patch_file: Path) -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    diff = []
    for name in ("model.py", "network.py", "layers.py", "utils.py"):
        original = (root / name).read_text(encoding="utf-8")
        modified = e32.patch_model(original) if name == "model.py" else (
            e32.patch_network(original) if name == "network.py" else original)
        if name == "network.py":
            marker = "        x = x.view(bs, -1)\n"
            assert modified.count(marker) == 1
            hook = (
                marker
                + "        if getattr(self, 'lcher', None) is not None:\n"
                + "            context = self.lcher_context\n"
                + "            x, self.lcher_last_diagnostics = self.lcher(\n"
                + "                x, context['raw_x'], context['z'], context['qc_reliability'],\n"
                + "                mask_override=context.get('mask_override'),\n"
                + "                residual_enabled=not context.get('disable_residual', False))\n"
            )
            modified = modified.replace(marker, hook, 1)
            ast.parse(modified)
        target = out / name
        target.write_text(modified, encoding="utf-8")
        diff.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                         fromfile="a/" + name, tofile="b/" + name))
    clip_link = out / "CLIP"
    if not clip_link.exists():
        clip_link.symlink_to(Path(root / "CLIP"), target_is_directory=True)
    patch = "".join(diff)
    patch_file.write_text(patch, encoding="utf-8")
