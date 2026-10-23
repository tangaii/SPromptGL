"""Install SeRCA-v2 in isolated Experiment 35 source copies."""
from __future__ import annotations

import torch

from serca_v2 import SeRCAV2

_CONTEXT = None


def configure(**values):
    global _CONTEXT
    assert values["config"] in {
        "E0_B0A", "E1_SERCA34_CAL", "E2_NUMERIC_GLOBAL", "E3_FIRST_ORDER",
        "E4_ALL_PAIR", "E5_RANDOM_RULE", "E6_PATIENT_SHUFFLED",
        "E7_ORIGINAL_EDGES", "E8_CORR_ONLY", "E9_AUX_ONLY",
        "E10_DIRECT_MATCHED", "E11_SERCA_V2_FULL",
    }
    _CONTEXT = values


def install(agent):
    assert _CONTEXT is not None
    agent.ModalFusion.serca35_bypass_pen = True
    config = _CONTEXT["config"]
    if config == "E0_B0A":
        return
    prior = _CONTEXT["prior"]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        module = SeRCAV2(
            signature=prior["signature"],
            signature_mask=prior["signature_mask"],
            semantic_kernel=prior["semantic_kernel"],
            kernel_center=prior["kernel_center"],
            semantic_neighbors=prior["semantic_neighbors"],
            relation_mask=prior["relation_mask"],
            direct_message=config == "E10_DIRECT_MATCHED",
        ).to(agent.dev)
    agent.ModalFusion.serca35 = module
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    agent.optimizer_MF.add_param_group({
        "params": trainable, "lr": agent.hyperpm.lr, "weight_decay": 0.0,
    })
    optimizer_ids = [id(parameter) for optimizer in (agent.optimizer_MP0, agent.optimizer_MF)
                     for group in optimizer.param_groups for parameter in group["params"]]
    module_ids = {id(parameter) for root in (agent.MessagePassing0, agent.ModalFusion)
                  for name, parameter in root.named_parameters()
                  if not name.startswith("serca35.") or parameter.requires_grad}
    module_ids |= {id(parameter) for parameter in trainable}
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == module_ids


def forward_modal(agent, features):
    module = agent.ModalFusion
    if not hasattr(module, "serca35"):
        return module(features)
    payload = dict(
        raw_inputs=agent.feat,
        raw_adjs=agent.MessagePassing0.last_raw_adjs,
        train_indices=agent.trn_idx,
    )
    return module(features, serca_payload=payload)
