"""Instance-local PEN replacement used by the isolated Experiment39 runner."""
from __future__ import annotations

import types
import ast

import torch


def cam_forward(self, x):
    """Retain InputLayer/CAM/Encoder/FeedForward, replace PEN before OutputLayer."""
    bs = x.size(0)
    attn_map = []
    x, _attn = self.InputLayer(x)
    attn = _attn.mean(dim=1)
    attn_map.append(attn.detach().cpu().numpy())
    for i in range(self.n_layer):
        x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
        attn = _attn.mean(dim=1)
        x = self.FeedForward[i](x)
        attn_map.append(attn.detach().cpu().numpy())

    cam = x.view(bs, 4, 36)
    branch = getattr(self, "hase39", None)
    if branch is not None:
        override = getattr(self, "hase39_eval_override", None) or {}
        if self.hase39_kind == "CAPACITY":
            x, diagnostic = branch(
                cam,
                self.hase39_raw,
                self.hase39_source_rows,
                self.hase39_median,
                self.hase39_scale,
            )
        else:
            x, diagnostic = branch(
                cam,
                self.hase39_raw,
                mask_override=override.get("mask_override"),
                route_uniform=override.get("route_uniform", self.hase39_uniform),
                expert_keep=override.get("expert_keep"),
                route_flags=override.get("route_flags"),
                source_rows=override.get("source_rows"),
            )
        if not self.training:
            self.hase39_last_eval = {
                name: value.detach() if torch.is_tensor(value) else value
                for name, value in diagnostic.items()
            }
        else:
            self.hase39_last_train = diagnostic
    else:
        x = cam

    x = x.reshape(bs, -1)
    attn_embedding = attn.reshape(bs, -1)
    output, hidden = self.Outputlayer(x, attn_embedding)
    return output, hidden, attn_map


def install(agent, config, branch=None, source_rows=None, median=None, scale=None):
    agent.ModalFusion.hase39_forward = types.MethodType(cam_forward, agent.ModalFusion)
    agent.ModalFusion.hase39_config = config
    agent.ModalFusion.hase39_raw = agent.feat
    if branch is not None:
        agent.ModalFusion.hase39 = branch
        agent.ModalFusion.hase39_kind = "MOE" if config != "E1_CAPACITY" else "CAPACITY"
        agent.ModalFusion.hase39_uniform = config == "E3_MED_UNIFORM"
        agent.ModalFusion.hase39_source_rows = source_rows
        agent.ModalFusion.hase39_median = median
        agent.ModalFusion.hase39_scale = scale
        agent.optimizer_MF.add_param_group({
            "params": list(branch.parameters()),
            "lr": agent.hyperpm.lr,
            "weight_decay": 0.0,
        })
    else:
        agent.ModalFusion.hase39 = None
        agent.ModalFusion.hase39_kind = "BASELINE"
        agent.ModalFusion.hase39_uniform = False

    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for module in (agent.MessagePassing0, agent.ModalFusion)
                for parameter in module.parameters()}
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != expected:
        raise AssertionError("The two Adam optimizers do not cover each trainable tensor once")
    if any(abs(group["lr"] - agent.hyperpm.lr) > 1e-12 for group in groups):
        raise AssertionError("An original or added parameter group changed the learning rate")
    if branch is not None and groups[-1]["weight_decay"] != 0.0:
        raise AssertionError("HASE parameters must use zero weight decay")


def evaluation_override(agent, **values):
    agent.ModalFusion.hase39_eval_override = values


def clear_override(agent):
    if hasattr(agent.ModalFusion, "hase39_eval_override"):
        delattr(agent.ModalFusion, "hase39_eval_override")


def patch_network_source(original):
    """Replace only PLN.forward in the private source mirror and return its diff."""
    tree = ast.parse(original)
    pln = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PLN")
    method = next(node for node in pln.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    lines = original.splitlines(keepends=True)
    replacement = [
        "    def forward(self, x):\n",
        "        return self.hase39_forward(x)\n",
    ]
    changed = lines[: method.lineno - 1] + replacement + lines[method.end_lineno :]
    return "".join(changed)
