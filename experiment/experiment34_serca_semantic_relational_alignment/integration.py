"""Install the single SeRCA module in isolated Experiment 34 source copies."""
from __future__ import annotations

import torch

from serca import SeRCA
from utils import normalize_adj

_CONTEXT = None


def configure(**values):
    global _CONTEXT
    assert values["config"] in {
        "S0_B0A", "S1_NUMERIC_GLOBAL", "S2_RANDOM_CONCEPTS", "S3_SINGLE_CONCEPT",
        "S4_CORR_REL_ONLY", "S5_AUX_REL_ONLY", "S6_UNIFORM_RELATION",
        "S7_DIRECT_MESSAGE", "S8_PATIENT_PERMUTED", "S9_SERCA_FULL",
    }
    _CONTEXT = values


def install(agent):
    assert _CONTEXT is not None
    agent.ModalFusion.serca34_bypass_pen = True
    config = _CONTEXT["config"]
    if config == "S0_B0A":
        return
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        module = SeRCA(
            similarities=_CONTEXT["similarities"], centers=_CONTEXT["centers"],
            relation_mask=_CONTEXT["relation_mask"], concept_mask=_CONTEXT["concept_mask"],
            direct_message=config == "S7_DIRECT_MESSAGE",
            uniform_relation=config == "S6_UNIFORM_RELATION",
        ).to(agent.dev)
    if config == "S6_UNIFORM_RELATION":
        module.theta.requires_grad_(False)
    agent.ModalFusion.serca34 = module
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    agent.optimizer_MF.add_param_group({
        "params": trainable, "lr": agent.hyperpm.lr, "weight_decay": 0.0,
    })
    optimizer_ids = [id(parameter) for optimizer in (agent.optimizer_MP0, agent.optimizer_MF)
                     for group in optimizer.param_groups for parameter in group["params"]]
    module_ids = {id(parameter) for root in (agent.MessagePassing0, agent.ModalFusion)
                  for name, parameter in root.named_parameters()
                  if not name.startswith("serca34.") or parameter.requires_grad}
    module_ids |= {id(parameter) for parameter in trainable}
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == module_ids


def forward_modal(agent, features):
    module = agent.ModalFusion
    if not hasattr(module, "serca34"):
        return module(features)
    return module(features, serca_payload=dict(
        raw_adjs=agent.MessagePassing0.last_raw_adjs,
        base_adjs=agent.MessagePassing0.last_normalized_adjs,
        normalize_adj=normalize_adj,
    ))
