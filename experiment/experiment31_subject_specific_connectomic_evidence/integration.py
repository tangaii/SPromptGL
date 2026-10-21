"""Disposable-source integration for SCORE.

Only runtime copies of the original files are patched.  The repository root is
never imported for a formal worker after the runtime has been built.
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

import numpy as np
import torch
from method import CONFIGS, SCORE

_CONTEXT = None


def configure(**values):
    global _CONTEXT
    assert values["config"] in CONFIGS + ("P0",)
    assert np.asarray(values["labels_train"]).shape == (len(values["train"]),)
    _CONTEXT = values


def install(agent):
    assert _CONTEXT is not None
    c = _CONTEXT
    agent.ModalFusion.score31_config = c["config"]
    if c["config"] == "P0":
        return
    assert np.array_equal(c["train"], agent.trn_idx)
    assert np.array_equal(c["test"], agent.tst_idx)
    # Keep the original model's random stream byte-for-byte independent from
    # SCORE initialization.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(310000 + int(c.get("seed", 0)) * 100 + int(c["fold"]))
        branch = SCORE(
            config=c["config"], raw=c["raw"], train=c["train"], test=c["test"],
            sites=c["site"], pheno_indices=c["pheno_indices"], bank=c["cache"],
            fold=c["fold"], seed=c.get("seed", 0),
        ).to(agent.dev)
    agent.ModalFusion.score31 = branch
    if branch.design.active:
        agent.optimizer_MF.add_param_group({"params": list(branch.parameters()), "lr": agent.hyperpm.lr, "weight_decay": 0.0})
    ids = [id(p) for g in agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups for p in g["params"]]
    actual = {id(p) for m in (agent.MessagePassing0, agent.ModalFusion) for p in m.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual


def patch_network(original: str) -> str:
    signature = "    def forward(self, x):\n        bs = x.size(0)"
    start = "        text_features = self.prompt_text_features\n"
    end = "        # 对最后一个维度进行平均池化\n"
    assert original.count(signature) == original.count(start) == original.count(end) == 1
    source = original.replace(signature, "    def forward(self, x, raw_input=None):\n        bs = x.size(0)", 1)
    a, b = source.index(start), source.index(end)
    spgs = source[a:b]
    assert spgs.count("part_x = similarity_max.unsqueeze(-1) * part_x") == 1
    old = "".join("    " + line if line.strip() else line for line in spgs.splitlines(True))
    replacement = "        if getattr(self, 'score31_config', 'P0') == 'P0':\n" + old
    replacement += (
        "        else:\n"
        "            assert raw_input is not None and hasattr(self, 'score31')\n"
        "            x = self.score31(x, raw_input)\n"
        "            part_x = x.view(871, 4, 4, 9)\n"
    )
    source = source[:a] + replacement + source[b:]
    ast.parse(source)
    assert source[source.index(end):] == original[original.index(end):]
    return source


def patch_model(original: str) -> str:
    assert original.count("from network import *\n") == 1
    assert original.count("        self.ModalFusion.apply(my_weight_init)") == 1
    assert original.count("            loss = cls_loss + all_graph_loss") == 1
    source = original.replace("from network import *\n", "from network import *\nimport integration\n", 1)
    source = source.replace("        self.ModalFusion.apply(my_weight_init)", "        self.ModalFusion.apply(my_weight_init)\n        integration.install(self)", 1)
    source = source.replace("self.ModalFusion(feat)", "self.ModalFusion(feat, raw_input=self.feat)")
    source = source.replace(
        "            loss = cls_loss + all_graph_loss",
        "            loss = cls_loss + all_graph_loss\n"
        "            if hasattr(self.ModalFusion, 'score31'):\n"
        "                loss = loss + self.ModalFusion.score31.extra_loss()",
        1,
    )
    ast.parse(source)
    return source


def emit(root, out, patch_path=None):
    root, out = Path(root), Path(out)
    assert root.resolve() != out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    diff = []
    for name in ("model.py", "network.py", "layers.py", "utils.py"):
        original = (root / name).read_text()
        modified = patch_model(original) if name == "model.py" else patch_network(original) if name == "network.py" else original
        dest = out / name
        if dest.exists():
            assert dest.read_text() == modified, "runtime source changed: " + name
        else:
            dest.write_text(modified)
        diff.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile="a/" + name, tofile="b/" + name))
    clip = out / "CLIP"
    if not clip.exists():
        clip.symlink_to(root / "CLIP", target_is_directory=True)
    if patch_path:
        p = Path(patch_path)
        value = "".join(diff)
        if p.exists():
            assert p.read_text() == value
        else:
            p.write_text(value)
    return out


if __name__ == "__main__":
    import sys
    emit(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
