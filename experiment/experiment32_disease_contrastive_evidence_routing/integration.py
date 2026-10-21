"""Disposable-source integration for Experiment 32 DCER."""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

import numpy as np
import torch

from method import DCER32, VALID

_CONTEXT = None


def configure(**values):
    global _CONTEXT
    assert values["config"] in ("B0", "B1", "B2") + VALID
    assert np.asarray(values["labels_train"]).shape == (len(values["train"]),)
    _CONTEXT = values


def install(agent):
    assert _CONTEXT is not None
    context = _CONTEXT
    config = context["config"]
    assert np.array_equal(context["train"], agent.trn_idx)
    assert np.array_equal(context["test"], agent.tst_idx)
    agent.ModalFusion.dcer32_config = config
    if config.startswith("B"):
        return
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(320000 + int(context["fold"]))
        branch = DCER32(
            agent.ModalFusion.prompt_text_features,
            config,
            context["fold"],
        ).to(agent.dev)
    agent.ModalFusion.dcer32 = branch
    agent.optimizer_MF.add_param_group({
        "params": list(branch.parameters()),
        "lr": agent.hyperpm.lr,
        "weight_decay": 0.0,
    })
    ids = [id(parameter) for group in agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
           for parameter in group["params"]]
    actual = {id(parameter) for module in (agent.MessagePassing0, agent.ModalFusion)
              for parameter in module.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual


def patch_network(original: str) -> str:
    signature = "    def forward(self, x):\n        bs = x.size(0)\n"
    start = "        text_features = self.prompt_text_features\n"
    reshape = "        x = x.view(bs, -1)\n"
    res_start = "        # 对最后一个维度进行平均池化\n"
    res_end = "        x = x + torch.cat(combine_part,dim=2)\n"
    assert original.count(signature) == 1
    assert original.count(start) == original.count(reshape) == original.count(res_start) == original.count(res_end) == 1
    source = original.replace(
        signature,
        "    def forward(self, x):\n        bs = x.size(0)\n        keep32 = drop32 = None\n",
        1,
    )
    a, b = source.index(start), source.index(reshape)
    span = source[a:b]
    original_res = source[source.index(res_start, a):source.index(res_end, a) + len(res_end)]
    assert span.count("text_features = self.prompt_text_features") == 1
    assert original_res.count("Vsim = F.softmax(part*global_x, dim=-1)") == 1
    indent = lambda block: "".join("    " + line if line.strip() else line for line in block.splitlines(True))
    replacement = "        config32 = getattr(self, 'dcer32_config', 'B2')\n"
    replacement += "        if config32 == 'B2':\n" + indent(span)
    replacement += "        elif config32 == 'B1':\n            part_x = x.view(bs, 4, 4, 9)\n" + indent(original_res)
    replacement += "        elif config32 == 'B0':\n            pass\n"
    replacement += "        else:\n            full32, keep32, drop32 = self.dcer32(x)\n            x = full32\n"
    source = source[:a] + replacement + source[b:]
    classifier = "        output, hidden = self.Outputlayer(x, attn_embedding)\n"
    ret = "        return output, hidden, attn_map\n"
    assert source.count(classifier) == source.count(ret) == 1
    source = source.replace(
        ret,
        "        self.keep32_logp = self.drop32_logp = None\n"
        "        if hasattr(self, 'dcer32') and self.training and torch.is_grad_enabled():\n"
        "            if keep32 is not None:\n"
        "                self.keep32_logp, _ = self.Outputlayer(keep32.reshape(bs, -1), attn_embedding)\n"
        "                self.drop32_logp, _ = self.Outputlayer(drop32.reshape(bs, -1), attn_embedding)\n" + ret,
        1,
    )
    ast.parse(source)
    assert source[source.index(reshape):] == original[original.index(reshape):].replace(
        ret,
        "        self.keep32_logp = self.drop32_logp = None\n"
        "        if hasattr(self, 'dcer32') and self.training and torch.is_grad_enabled():\n"
        "            if keep32 is not None:\n"
        "                self.keep32_logp, _ = self.Outputlayer(keep32.reshape(bs, -1), attn_embedding)\n"
        "                self.drop32_logp, _ = self.Outputlayer(drop32.reshape(bs, -1), attn_embedding)\n" + ret,
        1,
    )
    assert source.count("text_features = self.prompt_text_features") == 1
    return source


def patch_model(original: str) -> str:
    assert original.count("from network import *\n") == 1
    assert original.count("        self.ModalFusion.apply(my_weight_init)\n") == 1
    assert original.count("            loss = cls_loss + all_graph_loss\n") == 1
    source = original.replace("from network import *\n", "from network import *\nimport integration\n", 1)
    source = source.replace(
        "        self.ModalFusion.apply(my_weight_init)\n",
        "        self.ModalFusion.apply(my_weight_init)\n        integration.install(self)\n",
        1,
    )
    source = source.replace(
        "            loss = cls_loss + all_graph_loss\n",
        "            loss = cls_loss + all_graph_loss\n"
        "            if hasattr(self.ModalFusion, 'dcer32'):\n"
        "                loss = loss + self.ModalFusion.dcer32.extra_loss(\n"
        "                    self.targ, self.trn_idx,\n"
        "                    self.ModalFusion.keep32_logp,\n"
        "                    self.ModalFusion.drop32_logp\n"
        "                )\n",
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
        destination = out / name
        if destination.exists():
            assert destination.read_text() == modified, "runtime source changed: " + name
        else:
            destination.write_text(modified)
        diff.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                         fromfile="a/" + name, tofile="b/" + name))
    clip = out / "CLIP"
    if not clip.exists():
        clip.symlink_to(root / "CLIP", target_is_directory=True)
    if patch_path:
        path = Path(patch_path)
        value = "".join(diff)
        if path.exists():
            assert path.read_text() == value
        else:
            path.write_text(value)
    return out
